import os

import pandas as pd
import json

import ast
import re
import spacy
from transformers import pipeline
from rapidfuzz import fuzz

from tqdm import tqdm


def extract_text_row(row, cols):
    result = {}
    for col in cols:
        val = row[col]
        if isinstance(val, list):
            result[col] = " ".join(map(str, val))
        else:
            result[col] = str(val)
            
    return result


def wordlike_ratio(s):
    if not s:
        return 0.0
    
    return sum(ch.isalnum() or ch.isspace() for ch in s) / len(s)


def looks_like_ocr_gibberish(text):
    if not text:
        return True

    t = text.strip()

    non_alnum = sum(not (ch.isalnum() or ch.isspace()) for ch in t)
    if non_alnum / max(1, len(t)) > 0.25:
        return True

    words = re.findall(r"[A-Za-z]{3,}", t)
    if len(words) < 3:
        return True

    upper = sum(ch.isupper() for ch in t if ch.isalpha())
    lower = sum(ch.islower() for ch in t if ch.isalpha())
    if (upper + lower) > 0:
        if upper / (upper + lower) > 0.75 and len(words) < 6:
            return True

    return False


def coerce_text(v):
    if v is None:
        return ""
    
    if isinstance(v, list):
        return ". ".join(str(x).strip() for x in v if str(x).strip())
    
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return ". ".join(str(x).strip() for x in parsed if str(x).strip())
            except Exception:
                pass
        return s
    
    return str(v)


def normalize_text(text):
    text = text or ""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def clean_ocr_garbage(text):
    t = text
    t = re.sub(r"[|_~—–\-]{2,}", " ", t)
    t = re.sub(r"[^A-Za-z0-9\s'@#:/%.]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    
    return t


def uniq_case_insensitive(items):
    seen = set()
    out = []
    for x in items:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
            
    return out


def fuzzy_dedup(items, threshold=92):
    kept = []
    for text, score in sorted(items, key=lambda x: x[1], reverse=True):
        tl = text.lower()
        
        if any(tl == k[0].lower() for k in kept):
            continue
        
        if any(fuzz.ratio(tl, k[0].lower()) >= threshold for k in kept):
            continue
        
        kept.append((text, score))
        
    return kept


def rule_signals(text):
    handles = ["@" + m for m in HANDLE_RE.findall(text)]
    hashtags = ["#" + m for m in HASHTAG_RE.findall(text)]
    urls = URL_RE.findall(text)
    
    return {
        "handles": uniq_case_insensitive(handles),
        "hashtags": uniq_case_insensitive(hashtags),
        "urls": uniq_case_insensitive(urls),
    }


def looks_wordlike(w):
    if len(w) < 3:
        return False
    
    if w in STOPWORDS or w in QUERY_STOPWORDS or w in OCR_JUNK_TOKENS:
        return False
    
    if not any(v in w for v in "aeiou"):
        return False
    
    vowels = sum(ch in "aeiou" for ch in w)
    if vowels / len(w) < 0.20:
        return False
    
    if re.search(r"(.)\1\1", w):
        return False
    
    return True


def keep_chunk(chunk, source):
    c = (chunk or "").strip()
    if not c:
        return False

    c = re.sub(r"^\[+", "", c).strip()

    low = c.lower()
    
    if "meme poster" in low:
        return False

    if low in PRONOUNS or low in STOPWORDS or low in QUERY_STOPWORDS:
        return False

    non_word = sum(not (ch.isalnum() or ch.isspace() or ch in "'-") for ch in c)
    if non_word / max(1, len(c)) > 0.10:
        return False

    if wordlike_ratio(c) < 0.88:
        return False

    words = re.findall(r"[a-z0-9]+", low)

    if words and all((w in QUERY_STOPWORDS) or (w in STOPWORDS) for w in words):
        return False

    if any(w in OCR_JUNK_TOKENS for w in words):
        return False

    alpha_words = re.findall(r"[A-Za-z]{3,}", c)
    if not alpha_words:
        return False

    toks = re.findall(r"[A-Za-z0-9]+", c)
    if toks:
        tiny = sum(len(t) <= 2 for t in toks)
        if tiny / len(toks) > 0.30:
            return False

    if len(words) == 1:
        w = words[0]
        if w in QUERY_STOPWORDS or w in STOPWORDS:
            return False
        if not any(v in w for v in "aeiou"):
            return False

    alpha_tokens = re.findall(r"[A-Za-z]+", c)
    if alpha_tokens:
        caps = sum(t.isupper() for t in alpha_tokens)
        if caps / len(alpha_tokens) > 0.80:
            if len(re.findall(r"[A-Za-z]{3,}", c)) < 2:
                return False

    if len(c) > 60:
        if len(alpha_words) < 4:
            return False

    if source == "title" and low in {"real", "though", "fact", "horrible"}:
        return False

    if source == "img_captions":
        if len(words) == 1 and words[0] in GENERIC_CAPTION_WORDS:
            return False

    if words and words[-1] in {"in", "of", "to", "as", "for", "with", "on", "at", "the", "a", "an"}:
        return False
    
    if "|" in c or "\\" in c:
        return False

    return True


def noun_chunks_spacy(nlp, text, source):
    doc = nlp(text)
    chunks = []
    
    for nc in doc.noun_chunks:
        c = nc.text.strip()
        if keep_chunk(c, source):
            chunks.append(c)
            
    return uniq_case_insensitive(chunks)


def fallback_phrases(text, source):
    t = clean_ocr_garbage(text).lower()
    words = re.findall(r"[a-z]{3,}", t)
    words = [w for w in words if looks_wordlike(w)]
    words = [w for w in words if w not in STOPWORDS and w not in OCR_JUNK_TOKENS]

    seen = set()
    content = []
    for w in words:
        if w not in seen:
            seen.add(w)
            content.append(w)

    phrases = []
    
    content = [w for w in content if looks_wordlike(w)]

    BAD_BIGRAM_SECOND = {"where", "though", "fact", "this", "that", "there"}
    for i in range(len(content) - 1):
        w1, w2 = content[i], content[i + 1]
        if w1 in TRIGGERS or w2 in TRIGGERS:
            if w2 in BAD_BIGRAM_SECOND:
                continue
            phrases.append(f"{w1} {w2}")

    phrases.extend(content)

    phrases = [p for p in phrases if keep_chunk(p, source)]

    return uniq_case_insensitive(phrases)[:25]


def extract_capitalized_anchors(per_source):
    text = " ".join(
        per_source.get(src, {}).get("text", "")
        for src in ["title", "meme_captions", "img_captions"]
    )

    candidates = re.findall(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}))*\b", text)
    
    candidates = [c for c in candidates if c.lower() not in QUERY_STOPWORDS and "meme" not in c.lower()]

    out = []
    for c in candidates:
        cl = c.lower()
        if cl in {"the", "a", "an"}:
            continue
        
        if len(c) < 3:
            continue
        
        out.append(c)
        
    return uniq_case_insensitive(out)[:5]


def is_semantic_meme(per_source):
    text = " ".join(
        info.get("text", "").lower()
        for src, info in per_source.items()
        if src in {"text", "meme_captions", "title"}
    )
    
    hits = 0
    
    for term in (PLATFORM_TERMS | ABSTRACT_TERMS):
        if term in text:
            hits += 1
            
    return hits >= 2


def primary_is_junk(primary_terms):
    good = []
    
    for t in primary_terms:
        low = t.lower()

        if len(low) < 4:
            continue
        
        if low in PRONOUNS or low in STOPWORDS:
            continue
        
        if any(w in OCR_JUNK_TOKENS for w in re.findall(r"[a-z0-9]+", low)):
            continue
        
        good.append(t)
        
    return len(set(good)) < 3


def term_specificity_score(term):
    t = term.strip()
    tl = term.lower().strip()
    words = re.findall(r"[a-z0-9]+", tl)

    if not words:
        return 0.0
    
    if all(w in STOPWORDS or w in QUERY_STOPWORDS for w in words):
        return 0.0
    
    if any(w in QUERY_STOPWORDS for w in words) and len(words) <= 2:
        return 0.0

    if tl in PRONOUNS or tl in STOPWORDS:
        return 0.0

    words = t.split()
    score = 0.15 + 0.12 * min(len(words), 5)

    if any(c.isupper() for c in t):
        score += 0.15
        
    if any(c.isdigit() for c in t):
        score += 0.10

    if tl in PLATFORM_TERMS:
        score += 0.35

    return min(1.0, score)


def build_queries_from_terms(terms, signals, max_queries=3):
    scored = [(t, term_specificity_score(t)) for t in terms]
    scored = [(t, s) for t, s in scored if s > 0.25]
    scored = fuzzy_dedup(scored, threshold=92)

    ranked = [re.sub(r'^\[+', '', t).strip() for t, _ in scored][:12]

    queries = []

    sig_terms = (signals.get("handles", []) + signals.get("hashtags", []))[:6]
    
    if sig_terms:
        queries.append(" ".join(sig_terms))

    if ranked:
        queries.append(" ".join(f"\"{t}\"" for t in ranked[:4]))

    if len(ranked) >= 6:
        queries.append(" ".join(f"\"{t}\"" for t in ranked[2:6]))

    seen = set()
    out = []
    for q in queries:
        ql = q.lower()
        if ql not in seen and q.strip():
            seen.add(ql)
            out.append(q)
        if len(out) >= max_queries:
            break
        
    return out


def is_visualish(t):
    low = t.lower()
    words = set(re.findall(r"[a-z]+", low))
    
    return len(words & GENERIC_CAPTION_WORDS) >= 1 and low not in PLATFORM_TERMS


def stage_a_extract(item, nlp):
    per_source = {}
    agg_signals = {"handles": [], "hashtags": [], "urls": []}

    for src, val in item.items():
        raw = coerce_text(val)
        txt = normalize_text(raw)
        sig = rule_signals(txt)

        if src == "text" and looks_like_ocr_gibberish(txt):
            chunks = []
        else:
            chunks = noun_chunks_spacy(nlp, txt, source=src)

        if src in PRIMARY_SOURCES and len(chunks) < 3:
            chunks.extend(fallback_phrases(txt, source=src))

        chunks = [c for c in chunks if keep_chunk(c, source=src)]
        chunks = uniq_case_insensitive(chunks)

        per_source[src] = {"text": txt, "signals": sig, "chunks": chunks}

        for k in agg_signals:
            agg_signals[k].extend(sig[k])

    for k in agg_signals:
        agg_signals[k] = uniq_case_insensitive(agg_signals[k])

    primary_terms = []
    secondary_terms = []

    for src, info in per_source.items():
        if src in PRIMARY_SOURCES:
            primary_terms.extend(info["chunks"])
        else:
            secondary_terms.extend(info["chunks"])

    if is_semantic_meme(per_source):
        secondary_terms = [t for t in secondary_terms if not is_visualish(t)]

    if not primary_is_junk(primary_terms):
        queries = build_queries_from_terms(primary_terms, agg_signals, max_queries=3)
        if len(set(primary_terms)) < 6:
            queries += build_queries_from_terms(primary_terms + secondary_terms, agg_signals, max_queries=3)
    else:
        caption_terms = []
        for src, info in per_source.items():
            if src in QUERY_SOURCES_FALLBACK:
                caption_terms.extend(info["chunks"])
        queries = build_queries_from_terms(caption_terms, agg_signals, max_queries=3)

    seen = set()
    queries = [q for q in queries if not (q.lower() in seen or seen.add(q.lower()))][:3]

    anchors = extract_capitalized_anchors(per_source)
    if anchors:
        if not any(a.lower() in queries[0].lower() for a in anchors):
            queries[0] = f"\"{anchors[0]}\" {queries[0]}"

    return {
        "per_source": per_source,
        "signals": agg_signals,
        "primary_terms": uniq_case_insensitive(primary_terms),
        "secondary_terms": uniq_case_insensitive(secondary_terms),
        "queries": queries,
    }



def build_ner_pipeline(model_name="dslim/bert-base-NER", device=0):

    return pipeline(
        "token-classification",
        model=model_name,
        aggregation_strategy="simple",
        device=device,
    )


def prepare_text_for_ner(per_source):
    parts = []
    for src in ["title", "text", "meme_captions", "img_captions"]:
        if src in per_source:
            parts.append(per_source[src].get("text", ""))
            
    return " ".join(p for p in parts if p).strip()


def needs_stage_b(stage_a):
        
    return primary_is_junk(stage_a.get("primary_terms", []))


def extract_entities_from_ner_output(ner_output, min_score=0.6):
    ents = []

    for e in ner_output:
        text = (e.get("word") or "").strip()
        if not text:
            continue

        score = float(e.get("score", 0.0))
        if score < min_score:
            continue

        text = re.sub(r"\s+", " ", text).strip()
        low = text.lower()

        if low in BAD_ENTITY_TEXT or low in PRONOUNS or low in STOPWORDS or low in QUERY_STOPWORDS:
            continue

        if any(w in OCR_JUNK_TOKENS for w in re.findall(r"[a-z0-9]+", low)):
            continue

        if len(re.sub(r"[^A-Za-z0-9]", "", text)) < 4:
            continue

        non_word = sum(not (ch.isalnum() or ch.isspace() or ch in "'-") for ch in text)
        if non_word / max(1, len(text)) > 0.12:
            continue

        toks = re.findall(r"[A-Za-z0-9]+", text)
        if toks:
            tiny = sum(len(t) <= 2 for t in toks)
            if tiny / len(toks) > 0.30:
                continue

        ents.append({
            "text": text,
            "label": e.get("entity_group") or e.get("entity"),
            "score": score
        })

    best = {}
    for ent in ents:
        k = ent["text"].lower()
        if k not in best or ent["score"] > best[k]["score"]:
            best[k] = ent

    return sorted(best.values(), key=lambda x: x["score"], reverse=True)


def anchor_queries_with_entities(queries, entities):
    entities = [e for e in entities if e["text"].lower() not in BAD_ENTITY_TEXT]
    
    if not queries or not entities:
        return queries

    entity = sorted(entities, key=lambda e: (len(e["text"]), e["score"]), reverse=True)[0]["text"]

    anchored = [q for q in queries if entity.lower() in q.lower()]
    if anchored:
        return anchored + [q for q in queries if q not in anchored]

    return [f"\"{entity}\" {queries[0]}"] + queries[1:]


def process_dataset(dataset,
                    spacy_model="en_core_web_sm",
                    ner_model="dslim/bert-base-NER",
                    ner_device=0,
                    ner_batch_size=32):

    nlp = spacy.load(spacy_model, disable=["lemmatizer", "textcat"])
    ner = build_ner_pipeline(ner_model, device=ner_device)

    results = []
    stage_b_indices = []
    stage_b_texts = []

    for i, item in tqdm(enumerate(dataset)):
        a = stage_a_extract(item, nlp)
        results.append({"stage_a": a})
        if needs_stage_b(a):
            stage_b_indices.append(i)
            stage_b_texts.append(prepare_text_for_ner(a["per_source"]))

    if stage_b_texts:
        ner_outputs = ner(stage_b_texts, batch_size=ner_batch_size)

        for idx, out in zip(stage_b_indices, ner_outputs):
            entities = extract_entities_from_ner_output(out, min_score=0.6)
            results[idx]["stage_b_entities"] = entities

            results[idx]["stage_a"]["queries"] = anchor_queries_with_entities(
                results[idx]["stage_a"]["queries"],
                entities,
            )[:3]

    return results


PRIMARY_SOURCES = {"text", "title"}
CAPTION_SOURCES = {"meme_captions", "img_captions"}
QUERY_SOURCES_FALLBACK = {"title", "meme_captions", "img_captions"}

PLATFORM_TERMS = {
    "reddit", "google", "youtube", "twitter", "x", "instagram", "facebook", "tiktok",
    "wikipedia", "netflix", "amazon", "steam", "discord"
}
ABSTRACT_TERMS = {
    "better", "worse", "fact", "truth", "opinion", "search", "answer", "answers", "faster",
    "feature", "request", "update", "bug", "fix", "button", "undo", "scroll", "top", "back"
}

OCR_JUNK_TOKENS = {
    "youcango", "cae", "lna", "sabet", "orn",
}

GENERIC_CAPTION_WORDS = {
    "person", "guy", "girl", "someone", "somebody", 
    "dog", "dogs", "flag", "banner", "room", "rooms", "standing",
    "stage", "crowd", "notes", "outfit", "picture", "photo", "image",
    "poster", "meme"
}

PRONOUNS = {"me", "i", "you", "he", "she", "we", "they", "it", "my", "your", "his", "her", "our", "their"}

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are", "was", "were",
    "it", "you", "me", "my", "your", "this", "that", "so", "if", "about", "with", "as",
    "at", "by", "from", "be", "been", "being", "should", "would", "could", "into", "out",
}

TRIGGERS = {"undo", "button", "scroll", "top", "back", "search", "google", "reddit", "site"}

QUERY_STOPWORDS = {
    "just","what","who","when","where","why","how",
    "like","lol","lmao","bro","dude","damn","really","true",
    "some","someone","somebody","thing","stuff",
    "take","takes","long","need","needs","get","gets",
    "make","makes","dont","don't","aint","ain't",
}

BAD_ENTITY_TEXT = {
    "meme", "memes", "meme poster", "poster",
    "you", "your", "they", "them", "we", "i", "me",
    "but", "when", "why", "how", "what", "who",
    "get", "gets", "got", "come", "mean", "seriously",
    "true", "lol", "lmao", "bro", "aint", "ain't", "whats", "what's",
}

QUERY_STOPWORDS |= {
    "meme", "memes", "poster", "meme poster", "the meme poster",
    "you", "your", "they", "them", "we", "i", "me", "my",
    "but", "and", "or", "this", "that", "these", "those",
    "when", "why", "how", "what", "who",
    "mean", "seriously", "probably", "had", "have", "ask", "come",
}

HANDLE_RE = re.compile(r"(?<!\w)@([A-Za-z0-9_]{2,30})")
HASHTAG_RE = re.compile(r"(?<!\w)#([A-Za-z0-9_]{2,50})")
URL_RE = re.compile(r"\bhttps?://[^\s<>]+\b|\bwww\.[^\s<>]+\b", re.IGNORECASE)


os.chdir("")


if __name__ == "__main__":
    data = pd.read_csv("meme_temp/meme_data.csv")
    data["all_text"] = data.apply(extract_text_row, axis=1, cols=["img_captions", "meme_captions", "title", "text"])
    
    outputs = process_dataset(data["all_text"], ner_device=0, ner_batch_size=16)

    for i, out in enumerate(outputs):
        print("Queries:", out["stage_a"]["queries"])
        if "stage_b_entities" in out:
            print("Entities:", out["stage_b_entities"][:5])
            
    with open("meme_inputs/memes_data.jsonl", "w", encoding="utf-8") as f:
        for item in outputs:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
