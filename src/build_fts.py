import os

from tqdm import tqdm

from common import connect_sqlite, read_jsonl


def main():
    conn = connect_sqlite(OUT_DB)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS chunks_fts;")
    cur.execute("""
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id UNINDEXED,
            title,
            text,
            source_type,
            url
        );
    """)
    conn.commit()

    batch = []
    BATCH_SIZE = 2000

    for row in tqdm(read_jsonl(CHUNKS), desc="Indexing FTS5"):
        batch.append((
            row["chunk_id"],
            row.get("title", ""),
            row.get("text", ""),
            row.get("source_type", ""),
            row.get("url", ""),
        ))
        if len(batch) >= BATCH_SIZE:
            cur.executemany("INSERT INTO chunks_fts(chunk_id, title, text, source_type, url) VALUES (?, ?, ?, ?, ?);", batch)
            conn.commit()
            batch.clear()

    if batch:
        cur.executemany("INSERT INTO chunks_fts(chunk_id, title, text, source_type, url) VALUES (?, ?, ?, ?, ?);", batch)
        conn.commit()

    cur.execute("ANALYZE;")
    conn.commit()
    conn.close()


os.chdir("")

CHUNKS = "data/chunks/chunks.jsonl"
OUT_DB = "data/index/bm25.sqlite"


if __name__ == "__main__":
    main()
