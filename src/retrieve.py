import os

import json
import faiss
from sentence_transformers import SentenceTransformer
import re
import sqlite3

from common import clean_text, connect_sqlite, l2_normalize, read_jsonl


def make_fts_query(raw, max_tokens=10):
    raw = (raw or "").strip()
    if not raw:
        return ""

    raw = raw.replace("#", " ").replace(":", " ").replace("/", " ").replace("\\", " ")

    tokens = _FTS_TOKEN.findall(raw)
    if not tokens:
        return ""

    tokens = tokens[:max_tokens]

    return " AND ".join(tokens)


def fts_search(conn, query, k):
    q = make_fts_query(query)
    if not q:
        return []

    cur = conn.cursor()
    try:
        rows = cur.execute(
            """
            SELECT chunk_id, bm25(chunks_fts) AS score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?;
            """,
            (q, k),
        ).fetchall()
        
        return [(cid, -float(score)) for cid, score in rows]

    except sqlite3.OperationalError:
        toks = q.split()
        if not toks:
            return []
        
        q2 = toks[0]
        try:
            rows = cur.execute(
                """
                SELECT chunk_id, bm25(chunks_fts) AS score
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?;
                """,
                (q2, k),
            ).fetchall()
            return [(cid, -float(score)) for cid, score in rows]
        
        except sqlite3.OperationalError:
            return []



def faiss_search(index, model, query, k):
    query = clean_text(query)
    if not query:
        return []

    q = model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    q = l2_normalize(q)
    scores, ids = index.search(q, k)
    out = []
    for row_id, score in zip(ids[0].tolist(), scores[0].tolist()):
        if row_id == -1:
            continue
        out.append((int(row_id), float(score)))
        
    return out


def get_chunk_by_id(meta_conn, chunk_id):
    cur = meta_conn.cursor()
    row = cur.execute(
        "SELECT row_id, chunk_id, doc_id, title, url, source_type, text FROM chunk_meta WHERE chunk_id=?",
        (chunk_id,),
    ).fetchone()
    if not row:
        return {}
    
    row_id, chunk_id, doc_id, title, url, source_type, text = row
    
    return {
        "row_id": row_id,
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "title": title,
        "url": url,
        "source_type": source_type,
        "text": text,
    }


def get_chunk_by_rowid(meta_conn, row_id):
    cur = meta_conn.cursor()
    row = cur.execute(
        "SELECT row_id, chunk_id, doc_id, title, url, source_type, text FROM chunk_meta WHERE row_id=?",
        (row_id,),
    ).fetchone()
    if not row:
        return {}
    
    row_id, chunk_id, doc_id, title, url, source_type, text = row
    
    return {
        "row_id": row_id,
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "title": title,
        "url": url,
        "source_type": source_type,
        "text": text,
    }


def boost_if_terms_present(text, terms):
    t = (text or "").lower()
    boost = 0.0
    for term in terms:
        term = clean_text(term).lower()
        if term and term in t:
            boost += 0.10
            
    return boost


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    fts_conn = connect_sqlite(FTS_DB)
    meta_conn = connect_sqlite(META_DB)
    index = faiss.read_index(FAISS_INDEX)
    model = SentenceTransformer(MODEL)

    for i, entry in enumerate(read_jsonl(MEME_JSONL)):
        meme_id = entry.get("id") or entry.get("_id") or f"meme_{i:05d}"

        stage_a = entry.get("stage_a", {})
        queries = stage_a.get("queries", []) or []
        primary_terms = stage_a.get("primary_terms", []) or []

        cand_scores = {}
        cand_sources = {}

        for q in queries:
            for chunk_id, s in fts_search(fts_conn, q, BM25_K):
                cand_scores[chunk_id] = cand_scores.get(chunk_id, 0.0) + (1.0 * s)

            for row_id, s in faiss_search(index, model, q, DENSE_K):
                c = get_chunk_by_rowid(meta_conn, row_id)
                if not c:
                    continue
                chunk_id = c["chunk_id"]
                cand_scores[chunk_id] = cand_scores.get(chunk_id, 0.0) + (1.0 * s)

        for chunk_id in list(cand_scores.keys()):
            c = get_chunk_by_id(meta_conn, chunk_id)
            if not c:
                cand_scores.pop(chunk_id, None)
                continue

            cand_scores[chunk_id] += boost_if_terms_present(c.get("text", ""), primary_terms)

            if len(c.get("text", "")) < 200:
                cand_scores[chunk_id] -= 0.25
            cand_sources[chunk_id] = c

        ranked = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)

        selected = []
        seen_docs = set()
        for chunk_id, score in ranked:
            c = cand_sources[chunk_id]
            doc_id = c.get("doc_id", "")
            if len(selected) < FINAL_K:
                if len(seen_docs) < 4 and doc_id in seen_docs:
                    continue
                
                selected.append((c, score))
                seen_docs.add(doc_id)
            else:
                break

        if len(selected) < FINAL_K:
            selected_ids = set(x[0]["chunk_id"] for x in selected)
            for chunk_id, score in ranked:
                if chunk_id in selected_ids:
                    continue
                c = cand_sources[chunk_id]
                selected.append((c, score))
                if len(selected) >= FINAL_K:
                    break

        out = {
            "meme_id": meme_id,
            "queries": queries,
            "primary_terms": primary_terms,
            "evidence": [
                {
                    "chunk_id": c["chunk_id"],
                    "doc_id": c.get("doc_id", ""),
                    "title": c.get("title", ""),
                    "url": c.get("url", ""),
                    "source_type": c.get("source_type", ""),
                    "score": float(score),
                    "text": c.get("text", ""),
                }
                for (c, score) in selected
            ],
        }

        out_path = os.path.join(OUT_DIR, f"{meme_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    fts_conn.close()
    meta_conn.close()


os.chdir("")

_FTS_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}")

MEME_JSONL = "data/meme_inputs/memes_data.jsonl"
FTS_DB = "data/index/bm25.sqlite"
FAISS_INDEX = "data/index/faiss.index"
META_DB = "data/index/chunk_meta.sqlite"
OUT_DIR = "data/runs"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BM25_K = 40
DENSE_K = 40
FINAL_K = 10


if __name__ == "__main__":
    main()
