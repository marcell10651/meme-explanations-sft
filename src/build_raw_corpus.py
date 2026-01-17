import os

import json

import re
import requests

import time
from tqdm import tqdm


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def append_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def clean(s):
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def add_topic(topics, seen, t):
    t = clean(t)
    if not t:
        return
    
    k = t.lower()
    if k in seen:
        return
    
    if len(t) < 3:
        return
    
    if k in STOPWORDS:
        return

    if k in GENERIC_TOPICS:
        return

    if k.startswith("the "):
        return

    seen.add(k)
    topics.append(t)
    
def is_specific_token(tok):
    tok = tok.lower()
    if tok in STOPWORDS:
        return False
    
    if tok in GENERIC_TOPICS:
        return False

    if len(tok) <= 3 and not tok.isupper():
        return False
    
    return True


    
def title_matches_topic(topic, title):
    t = clean(topic).lower()
    s = clean(title).lower()

    topic_tokens = [w for w in re.findall(r"[a-z0-9]+", t) if w not in STOPWORDS]
    title_tokens = set(w for w in re.findall(r"[a-z0-9]+", s) if w not in STOPWORDS)

    if not topic_tokens:
        return False

    if not any(is_specific_token(w) for w in topic_tokens):
        return False

    covered = sum(1 for w in topic_tokens if w in title_tokens)
    if covered / len(topic_tokens) < 0.8:
        return False

    if len(topic_tokens) == 1 and len(s) >= 40:
        return False

    return True


def extract_topics_from_entry(entry, max_topics_per_entry=50):
    stage_a = entry.get("stage_a", {}) or {}
    per_source = stage_a.get("per_source", {}) or {}

    topics = []
    seen = set()
    
    for key in ["img_captions", "meme_captions", "title", "text"]:
        d = per_source.get(key, {}) or {}
        for c in (d.get("chunks", []) or []):
            add_topic(topics, seen, c)

    for t in (stage_a.get("primary_terms", []) or []):
        add_topic(topics, seen, t)
    for t in (stage_a.get("secondary_terms", []) or []):
        add_topic(topics, seen, t)

    queries = stage_a.get("queries", []) or []
    for q in queries:
        for phrase in _QUOTED.findall(q or ""):
            add_topic(topics, seen, phrase)

        q2 = (q or "").replace('"', " ")
        words = [w for w in _WORD.findall(q2) if w.lower() not in STOPWORDS]

        for w in words:
            if len(w) >= 5:
                add_topic(topics, seen, w)

        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            if not (is_specific_token(w1) and is_specific_token(w2)):
                continue
            
            phrase = f"{w1} {w2}"
            if len(phrase) >= 8:
                add_topic(topics, seen, phrase)

    text_fields = []
    for key in ["img_captions", "meme_captions", "title", "text"]:
        d = per_source.get(key, {}) or {}
        text_fields.append(d.get("text", ""))

    combined = " ".join(clean(x) for x in text_fields if x)

    caps_spans = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", combined)
    for span in caps_spans:

        if span.lower() in STOPWORDS:
            continue
        
        add_topic(topics, seen, span)

    return topics[:max_topics_per_entry]

def opensearch_titles(topic, limit, session):
    params = {
        "action": "opensearch",
        "search": topic,
        "limit": str(limit),
        "namespace": "0",
        "format": "json",
    }
    r = session.get(WIKI_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    titles = data[1] if len(data) > 1 else []
    
    return [clean(t) for t in titles if clean(t)]

def search_titles(topic, limit, session):
    params = {
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "srlimit": str(limit),
        "format": "json",
    }
    r = session.get(WIKI_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    hits = (((data.get("query") or {}).get("search")) or [])
    
    return [clean(h.get("title","")) for h in hits if clean(h.get("title",""))]


def fetch_extracts_by_titles(titles, session, chars):
    if not titles:
        return []

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "exsectionformat": "plain",
        "redirects": "1",
        "titles": "|".join(titles),
        "exchars": str(chars),
    }
    
    r = session.get(WIKI_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    pages = (data.get("query", {}) or {}).get("pages", {}) or {}
    out = []

    for page in pages.values():
        if not page or "missing" in page:
            continue

        pageid = page.get("pageid")
        canonical_title = page.get("title") or ""
        extract = clean(page.get("extract", ""))

        if not canonical_title or not extract:
            continue

        url = (
            f"https://en.wikipedia.org/?curid={pageid}"
            if pageid
            else f"https://en.wikipedia.org/wiki/{canonical_title.replace(' ', '_')}"
        )

        out.append({
            "doc_id": f"wikipedia:{canonical_title}",
            "title": canonical_title,
            "source_type": "wikipedia",
            "url": url,
            "text": extract,
        })

    return out

def main():
    seen_doc_ids = set()
    if os.path.exists(OUT_DOCS):
        for row in read_jsonl(OUT_DOCS):
            did = row.get("doc_id")
            if did:
                seen_doc_ids.add(did.lower())

    topics = []
    seen_topic = set()

    for entry in tqdm(read_jsonl(MEME_JSONL), desc="Extracting topics"):
        ts = extract_topics_from_entry(entry, max_topics_per_entry=MAX_TOPICS_PER_ENTRY)
        for t in ts:
            k = t.lower()
            if k in seen_topic:
                continue
            seen_topic.add(k)
            topics.append(t)
            if len(topics) >= MAX_TOPICS_TOTAL:
                break
        if len(topics) >= MAX_TOPICS_TOTAL:
            break

    print(f"Extracted {len(topics)} unique candidate topics (cap={MAX_TOPICS_TOTAL}).")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "meme-background-retrieval-research/0.1 (local academic use; contact: maculamaci@freemail.hu)"
    })
    new_docs = []

    for topic in tqdm(topics, desc="Fetching Wikipedia pages"):
        try:
            titles = search_titles(topic, limit=OPENSEARCH_LIMIT, session=session)
        except Exception:
            continue
        
        titles = [ti for ti in titles if title_matches_topic(topic, ti)]
        if not titles:
            continue

        titles = [
            t for t in titles
            if "(disambiguation)" not in t.lower()
            and not t.lower().startswith("list of ")
        ]
    
        try:
            docs = fetch_extracts_by_titles(titles, session=session, chars=EXTRACT_CHARS)
        except Exception:
            docs = []
    
        for doc in docs:
            did = doc["doc_id"].lower()
            if did in seen_doc_ids:
                continue
            if len(doc["text"]) < MIN_DOC_CHARS:
                continue
    
            doc["created_at"] = time.strftime("%Y-%m-%d")
            new_docs.append(doc)
            seen_doc_ids.add(did)

        time.sleep(SLEEP)


    if new_docs:
        append_jsonl(OUT_DOCS, new_docs)

    print(f"Added {len(new_docs)} new docs to {OUT_DOCS}")
    

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_\-']+")
_QUOTED = re.compile(r'"([^"]{3,120})"')

STOPWORDS = set("""
a an and are as at be by for from has have he her hers him his i if in into is it its
me my of on or our ours she that the their them they this those to was we were what when
where who why will with you your yours though
""".split())

GENERIC_TOPICS = set("""
person people someone everyone something anything nothing
man woman boy girl guy couple crowd audience
area place thing stuff
improvement format website
front back top bottom
fun still happening leaving shopping rights
""".split())


WIKI_API = "https://en.wikipedia.org/w/api.php"


MEME_JSONL = "data/meme_inputs/memes_data.jsonl"
OUT_DOCS = "data/corpus_raw/docs.jsonl"
MAX_TOPICS_TOTAL = 1500
MAX_TOPICS_PER_ENTRY = 10
OPENSEARCH_LIMIT = 1
EXTRACT_CHARS = 3500
MIN_DOC_CHARS = 600
SLEEP = 0.8


os.chdir("")

if __name__ == "__main__":
    main()
