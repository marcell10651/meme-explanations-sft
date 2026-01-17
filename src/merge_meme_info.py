import os
import json
from tqdm import tqdm

JSONL_PATH = r""
JSON_DIR = r""
OUTPUT_DIR = r""

os.makedirs(OUTPUT_DIR, exist_ok=True)

jsonl_items = []
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            jsonl_items.append(json.loads(line))

json_files = sorted(
    f for f in os.listdir(JSON_DIR) if f.endswith(".json")
)

assert len(jsonl_items) == len(json_files), (
    f"Count mismatch: JSONL={len(jsonl_items)} JSON files={len(json_files)}"
)


for idx, filename in tqdm(enumerate(json_files), total=len(json_files)):
    json_path = os.path.join(JSON_DIR, filename)
    out_path = os.path.join(OUTPUT_DIR, filename)

    with open(json_path, "r", encoding="utf-8") as f:
        item = json.load(f)

    jsonl_item = jsonl_items[idx]
    
    if "stage_a" in jsonl_item:
        item["stage_a"] = jsonl_item["stage_a"]

    item["source_jsonl_id"] = jsonl_item.get("id")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(item, f, ensure_ascii=False, indent=2)
