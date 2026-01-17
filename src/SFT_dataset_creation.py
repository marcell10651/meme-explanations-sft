from pathlib import Path

import json
import re

from tqdm import tqdm


def violates_text_only_rules(s):
    if not s:
        return True
    
    if "```" in s:
        return True
    
    stripped = s.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return True
    
    if re.search(r"^\s*#\s", s, re.MULTILINE):
        return True
    
    return False

def normalize_messages(messages):
    out = []
    for m in messages:
        role = (m.get("role") or "").strip()
        content = m.get("content")
        if role not in ("system", "user", "assistant"):
            continue
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        out.append({"role": role, "content": content})
        
    return out

def to_prompt_completion(obj):
    messages = normalize_messages(obj.get("messages") or [])
    if not messages:
        return None, "no_messages"

    last_asst_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            last_asst_idx = i
            break

    if last_asst_idx is None:
        return None, "no_assistant"

    prompt = messages[:last_asst_idx]
    completion = [messages[last_asst_idx]]

    if DROP_SYSTEM:
        prompt = [m for m in prompt if m["role"] != "system"]

    if not any(m["role"] == "user" for m in prompt):
        return None, "no_user_in_prompt"

    comp_text = completion[0]["content"]
    if len(comp_text) < MIN_COMPLETION_CHARS:
        return None, "completion_too_short"

    if FILTER_BAD and violates_text_only_rules(comp_text):
        return None, "completion_format_violation"

    return {"prompt": prompt, "completion": completion}, None

def main():
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    rejected = 0

    with INPUT_JSONL.open("r", encoding="utf-8") as fin, \
         OUTPUT_JSONL.open("w", encoding="utf-8") as fout, \
         REJECTS_JSONL.open("w", encoding="utf-8") as frej:

        for line in tqdm(fin, desc="Converting"):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                frej.write(json.dumps({
                    "reject_reason": "json_parse_error",
                    "detail": str(e),
                    "line_prefix": line[:500]
                }, ensure_ascii=False) + "\n")
                rejected += 1
                continue

            rec, reason = to_prompt_completion(obj)
            if rec is None:
                frej.write(json.dumps({
                    "id": obj.get("id"),
                    "reject_reason": reason
                }, ensure_ascii=False) + "\n")
                rejected += 1
                continue

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1


INPUT_JSONL = Path(r"")
OUTPUT_JSONL = Path(r"")
REJECTS_JSONL = Path(r"")


DROP_SYSTEM = False
FILTER_BAD = True
MIN_COMPLETION_CHARS = 200



if __name__ == "__main__":
    main()
