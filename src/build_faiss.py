import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from common import connect_sqlite, l2_normalize, read_jsonl


def main():
    model = SentenceTransformer(MODEL)

    meta = connect_sqlite(OUT_META_DB)
    cur = meta.cursor()
    cur.execute("DROP TABLE IF EXISTS chunk_meta;")
    cur.execute("""
        CREATE TABLE chunk_meta(
            row_id INTEGER PRIMARY KEY,
            chunk_id TEXT UNIQUE,
            doc_id TEXT,
            title TEXT,
            url TEXT,
            source_type TEXT,
            text TEXT
        );
    """)
    meta.commit()

    chunks = list(read_jsonl(CHUNKS))
    texts = [c.get("text", "") for c in chunks]

    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding chunks"):
        batch_text = texts[i:i + BATCH_SIZE]
        emb = model.encode(batch_text, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb)
    X = np.vstack(embeddings).astype("float32")
    X = l2_normalize(X)

    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)

    if USE_GPU:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("FAISS GPU enabled.")
        except Exception as e:
            print(f"Could not enable FAISS GPU; continuing on CPU. Reason: {e}")

    index.add(X)

    rows = []
    for row_id, c in enumerate(chunks):
        rows.append((
            row_id,
            c["chunk_id"],
            c.get("doc_id", ""),
            c.get("title", ""),
            c.get("url", ""),
            c.get("source_type", ""),
            c.get("text", ""),
        ))
    cur.executemany("""
        INSERT INTO chunk_meta(row_id, chunk_id, doc_id, title, url, source_type, text)
        VALUES (?, ?, ?, ?, ?, ?, ?);
    """, rows)
    meta.commit()
    meta.close()

    if isinstance(index, faiss.IndexPreTransform) or hasattr(index, "gpu"):
        try:
            index_cpu = faiss.index_gpu_to_cpu(index)
            faiss.write_index(index_cpu, OUT_INDEX)
        except Exception:
            faiss.write_index(index, OUT_INDEX)
    else:
        faiss.write_index(index, OUT_INDEX)


os.chdir("")

CHUNKS = "data/chunks/chunks.jsonl"
OUT_INDEX = "data/index/faiss.index"
OUT_META_DB = "data/index/chunk_meta.sqlite"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64
USE_GPU = True


if __name__ == "__main__":
    main()
