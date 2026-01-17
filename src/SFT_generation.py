import os

import asyncio
from openai import AsyncOpenAI

import json
import re
import random

from tqdm import tqdm


def iter_items(path):
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.endswith(".json"):
                fp = os.path.join(path, name)
                with open(fp, "r", encoding="utf-8") as f:
                    yield json.load(f)
                    
    elif os.path.isfile(path) and path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            yield json.load(f)
            
    else:
        raise ValueError(f"INPUT_PATH not found or not .json/.dir: {path}")


def count_items(path: str) -> int:
    if os.path.isdir(path):
        return sum(1 for n in os.listdir(path) if n.endswith(".json"))
    if os.path.isfile(path) and path.endswith(".json"):
        return 1
    return 0


def get_stage_a_texts(item):
    stage_a = item.get("stage_a", {}) or {}
    per_source = stage_a.get("per_source", {}) or {}

    def get_text(obj_path):
        cur = per_source
        for k in obj_path:
            if not isinstance(cur, dict):
                return ""
            cur = cur.get(k, {}) or {}
        if isinstance(cur, dict):
            return (cur.get("text") or "").strip()
        return ""

    return {
        "title": get_text(["title"]),
        "ocr_text": get_text(["text"]),
        "meme_caption": get_text(["meme_captions"]),
        "img_caption": get_text(["img_captions"]),
    }


def get_terms(item):
    stage_a = item.get("stage_a", {}) or {}
    primary = stage_a.get("primary_terms", item.get("primary_terms", [])) or []
    secondary = stage_a.get("secondary_terms", []) or []

    if isinstance(primary, str):
        primary = [primary]
    if isinstance(secondary, str):
        secondary = [secondary]

    def dedup(xs):
        seen = set()
        out = []
        for x in xs:
            x = (x or "").strip()
            if not x or x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    return dedup(primary), dedup(secondary)


def evidence_relevance_score(e, primary_terms_lc):
    base = float(e.get("score", 0.0) or 0.0)
    title = (e.get("title", "") or "").lower()
    text = (e.get("text", "") or "").lower()

    overlap = 0
    for t2 in primary_terms_lc[:10]:
        if len(t2) < 3:
            continue
        if t2 in title or t2 in text:
            overlap += 1

    return base + 0.15 * overlap


def select_evidence(item, primary_terms):
    ev = item.get("evidence", []) or []
    ev = [e for e in ev if float(e.get("score", 0.0) or 0.0) >= MIN_SCORE]

    primary_terms_lc = [t.lower() for t in primary_terms]
    ev_sorted = sorted(ev, key=lambda e: evidence_relevance_score(e, primary_terms_lc), reverse=True)
    
    return ev_sorted[:TOP_K_EVIDENCE]


def format_evidence_block(evidence):
    if not evidence:
        return "(none)"

    blocks = []
    for i, e in enumerate(evidence, start=1):
        title = e.get("title", "")
        url = e.get("url", "")
        chunk_id = e.get("chunk_id", "")
        source_type = e.get("source_type", "")
        score = e.get("score", "")
        text = (e.get("text", "") or "").strip().replace("\n", " ")
        if len(text) > 900:
            text = text[:900] + "…"

        blocks.append(
            f"[{i}] {title} ({source_type}) score={score}\n"
            f"chunk_id={chunk_id}\n"
            f"url={url}\n"
            f"snippet={text}"
        )
        
    return "\n\n".join(blocks)


def build_messages(item):
    meme_id = item.get("meme_id", item.get("id", "unknown"))
    texts = get_stage_a_texts(item)
    primary, secondary = get_terms(item)
    evidence = select_evidence(item, primary)

    user_prompt = USER_TEMPLATE.format(
        meme_id=meme_id,
        title=texts["title"],
        ocr_text=texts["ocr_text"],
        meme_caption=texts["meme_caption"],
        img_caption=texts["img_caption"],
        primary_terms=", ".join(primary),
        secondary_terms=", ".join(secondary),
        evidence_block=format_evidence_block(evidence),
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    sources = [
        {
            "title": e.get("title", ""),
            "url": e.get("url", ""),
            "chunk_id": e.get("chunk_id", ""),
            "doc_id": e.get("doc_id", ""),
            "source_type": e.get("source_type", ""),
            "score": e.get("score", None),
        }
        for e in evidence
    ]

    return messages, sources, meme_id


def violates_text_only_rules(s):
    if "```" in s:
        return True
    
    stripped = s.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return True
    
    if re.search(r"^\s*#\s", s, re.MULTILINE):
        return True
    
    return False


async def call_llm(client, messages):
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                timeout=REQUEST_TIMEOUT_SEC,
            )
            return (resp.choices[0].message.content or "").strip()
        
        except Exception as e:
            last_err = e
            sleep_s = min(RETRY_BACKOFF_SEC * (2 ** (attempt - 1)), 20.0)
            sleep_s = sleep_s * (0.75 + 0.5 * random.random())
            await asyncio.sleep(sleep_s)
            
    raise RuntimeError(f"LLM call failed after retries: {last_err}")

async def process_one(client, sem, item):
    messages, sources, meme_id = build_messages(item)

    async with sem:
        final = await call_llm(client, messages)

    if violates_text_only_rules(final):
        repair_messages = messages + [
            {"role": "assistant", "content": final},
            {
                "role": "user",
                "content": "Rewrite your answer as plain text only: no JSON, no Markdown, no code fences. "
                           "Keep the labeled sections (CONTEXT, ENTITIES & CONCEPTS, etc.) as plain text lines.",
            },
        ]
        async with sem:
            final = await call_llm(client, repair_messages)

    return {
        "id": meme_id,
        "messages": messages + [{"role": "assistant", "content": final}],
        "metadata": {"sources_used": sources},
    }


async def main_async():
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    sem = asyncio.Semaphore(CONCURRENCY)

    total = count_items(INPUT_PATH)
    items_iter = iter_items(INPUT_PATH)

    in_flight: set[asyncio.Task] = set()
    max_queued = CONCURRENCY * 3

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8", buffering=1) as out:
        pbar = tqdm(total=total)

        async def drain_one_finished():
            done, _ = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                in_flight.remove(t)
                record = t.result()
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                pbar.update(1)

        for item in items_iter:
            task = asyncio.create_task(process_one(client, sem, item))
            in_flight.add(task)

            if len(in_flight) >= max_queued:
                await drain_one_finished()

        while in_flight:
            await drain_one_finished()

        pbar.close()
        


SYSTEM_PROMPT = """You are a careful annotator creating BACKGROUND INSTRUCTION SHEETS for internet memes.

Given:
(1) extracted meme text/captions and
(2) retrieved background evidence snippets,

Write a text-only instruction sheet that teaches another LLM enough context to understand the meme’s references
without seeing the image.

Hard rules:
- Output MUST be plain text only (no JSON, no Markdown, no code fences).
- Do NOT say “in the image” / “above” / “this picture”. Describe the situation in text.
- Use the evidence for factual claims. If evidence is irrelevant or weak, say it’s uncertain and ignore it.
- Do NOT invent names/dates/events not present in the inputs.

Include these sections with clear labels:
CONTEXT:
ENTITIES & CONCEPTS:
IMPLIED SITUATION / USER EXPERIENCE:
WHAT THE MEME IS SAYING:
WHAT MAKES IT FUNNY / RELATABLE:
UNCERTAINTIES / NOISE (if any):
"""

USER_TEMPLATE = """Meme ID: {meme_id}

EXTRACTED MEME CONTENT:
Title: {title}
OCR/Text: {ocr_text}
Meme caption interpretation: {meme_caption}
Image caption interpretation: {img_caption}

Key terms:
Primary terms: {primary_terms}
Secondary terms: {secondary_terms}

RETRIEVED EVIDENCE SNIPPETS:
{evidence_block}

Task reminder:
- Explain relevant concepts/entities/events needed to understand the meme.
- If some evidence is clearly unrelated (retrieval noise), call it out and do not use it.
"""
        

API_KEY = ""
BASE_URL = ""
MODEL = "zai-org/GLM-4.5-Air-FP8"

INPUT_PATH = r""
OUTPUT_PATH = r""

TOP_K_EVIDENCE = 4
MIN_SCORE = 0.1
TEMPERATURE = 0.2

MAX_RETRIES = 5
RETRY_BACKOFF_SEC = 1.5

CONCURRENCY = 24

REQUEST_TIMEOUT_SEC = 60
        

def main():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        return asyncio.create_task(main_async())
    else:
        return asyncio.run(main_async())

if __name__ == "__main__":
    main()
