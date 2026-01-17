import json
import os
import time

from common import chunk_text_chars, clean_text, read_jsonl


def main():
    os.makedirs(os.path.dirname(OUT_CHUNKS), exist_ok=True)

    docs = 0
    chunks_out = 0
    t0 = time.time()

    with open(OUT_CHUNKS, "w", encoding="utf-8") as out:
        for doc in read_jsonl(RAW_DOCS):
            docs += 1

            doc_id = doc.get("doc_id", f"doc_{docs}")
            title = clean_text(doc.get("title", ""))
            url = clean_text(doc.get("url", ""))
            source_type = clean_text(doc.get("source_type", ""))
            text = doc.get("text", "") or ""

            chunks = chunk_text_chars(
                text,
                chunk_size_chars=CHUNK_SIZE_CHARS,
                overlap_chars=OVERLAP_CHARS,
            )

            for i, c in enumerate(chunks):
                row = {
                    "chunk_id": f"{doc_id}#{i:04d}",
                    "doc_id": doc_id,
                    "title": title,
                    "url": url,
                    "source_type": source_type,
                    "text": c,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                chunks_out += 1

            if docs % REPORT_EVERY == 0:
                dt = time.time() - t0
                print(f"processed docs={docs}, chunks={chunks_out}, last_doc_id={doc_id}, elapsed={dt:.1f}s")
                out.flush()

    dt = time.time() - t0
    print(f"DONE: docs={docs}, chunks={chunks_out}, elapsed={dt:.1f}s, out={OUT_CHUNKS}")



os.chdir("")

RAW_DOCS = "data\corpus_raw\docs.jsonl"
OUT_CHUNKS = "data\chunks\chunks.jsonl"
CHUNK_SIZE_CHARS = 1000
OVERLAP_CHARS = 150
REPORT_EVERY = 25

if __name__ == "__main__":
    main()
