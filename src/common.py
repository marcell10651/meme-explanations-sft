import os

import numpy as np
import json

import re
import sqlite3


_WHITESPACE_RE = re.compile(r"\s+")
_NONPRINT_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+")


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            yield json.loads(line)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def clean_text(s):
    s = s or ""
    s = _NONPRINT_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    
    return s


def chunk_text_chars(text, chunk_size_chars=1000, overlap_chars=150):
    text = clean_text(text)
    if not text:
        return []

    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be > 0")
    if overlap_chars < 0 or overlap_chars >= chunk_size_chars:
        raise ValueError("overlap_chars must be >= 0 and < chunk_size_chars")

    n = len(text)
    chunks = []
    start = 0

    step = max(1, chunk_size_chars - overlap_chars)
    max_iters = (n // step) + 100
    iters = 0

    def snap_end(s, e):
        if e >= n:
            return n
        
        window = text[s:e]

        candidates = [
            window.rfind(". "),
            window.rfind("! "),
            window.rfind("? "),
        ]
        cut = max(candidates)
        if cut >= int(0.6 * len(window)):
            return s + cut + 1

        cut2 = max(window.rfind("\n"), window.rfind(" "))
        if cut2 >= int(0.7 * len(window)):
            return s + cut2

        return e

    def snap_start(ns):
        if ns <= 0:
            return 0
        
        if ns >= n:
            return n

        while ns < n and ns > 0 and text[ns - 1].isalnum() and text[ns].isalnum():
            ns += 1

        while ns < n and text[ns].isspace():
            ns += 1

        while ns < n and text[ns] in {")", "]", "}", ",", ";", ":", "—", "–"}:
            ns += 1
            
            while ns < n and text[ns].isspace():
                ns += 1

        return ns

    while start < n:
        iters += 1
        if iters > max_iters:
            tail = text[start:].strip()
            if tail:
                chunks.append(tail)
            break

        raw_end = min(n, start + chunk_size_chars)
        end = snap_end(start, raw_end)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        next_start = end - overlap_chars
        if next_start <= start:
            next_start = end

        start = snap_start(next_start)

    return chunks


def ensure_parent_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def connect_sqlite(path):
    ensure_parent_dir(path)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    
    return conn


def l2_normalize(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    
    return mat / norms
