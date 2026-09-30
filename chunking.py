"""Three chunking strategies: fixed, recursive, semantic.

Each returns a list of chunk strings. CLI: python3 chunking.py [strategy ...]
writes chunks/<strategy>.jsonl for every article in data/.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import requests

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"


def fixed_chunks(text, size=1000, overlap=200):
    """Sliding window of `size` characters, stepping by size - overlap."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    step = size - overlap
    return [text[i:i + size].strip() for i in range(0, len(text), step)
            if text[i:i + size].strip()]


def recursive_chunks(text, size=1000, overlap=200, separators=("\n\n", "\n", ". ", " ")):
    """Split on the coarsest separator that works, recursing into oversized pieces,
    then merge small pieces back up to `size` with a tail overlap."""
    def split(t, seps):
        if len(t) <= size:
            return [t]
        if not seps:  # no separator left: hard cut
            return [t[i:i + size] for i in range(0, len(t), size)]
        sep, rest = seps[0], seps[1:]
        parts = t.split(sep)
        if len(parts) == 1:
            return split(t, rest)
        out = []
        for i, p in enumerate(parts):
            p = p + sep if i < len(parts) - 1 else p
            out.extend(split(p, rest) if len(p) > size else [p])
        return out

    chunks, cur = [], ""
    for piece in split(text, list(separators)):
        if cur and len(cur) + len(piece) > size:
            chunks.append(cur.strip())
            cur = cur[-overlap:] + piece if overlap else piece
        else:
            cur += piece
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if c]


def embed(texts):
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts},
                      timeout=120)
    r.raise_for_status()
    return np.array(r.json()["embeddings"])


def semantic_chunks(text, percentile=90, max_size=2000, min_size=200):
    """Split into sentences, embed them, and start a new chunk wherever the cosine
    distance between neighbouring sentences is above the `percentile` of all gaps."""
    sentences = []
    for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text):
        s = s.strip()
        if len(s) > max_size:  # e.g. lists/tables with no sentence punctuation
            sentences.extend(recursive_chunks(s, size=max_size, overlap=0))
        elif s:
            sentences.append(s)
    if len(sentences) < 3:
        return [text.strip()] if text.strip() else []

    vecs = np.vstack([embed(sentences[i:i + 64]) for i in range(0, len(sentences), 64)])
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    dist = 1 - np.sum(vecs[:-1] * vecs[1:], axis=1)  # distance between sentence i and i+1
    threshold = np.percentile(dist, percentile)

    chunks, cur = [], [sentences[0]]
    for s, d in zip(sentences[1:], dist):
        cur_len = sum(map(len, cur))
        if (d > threshold and cur_len >= min_size) or cur_len + len(s) > max_size:
            chunks.append(" ".join(cur))
            cur = []
        cur.append(s)
    chunks.append(" ".join(cur))
    return chunks


STRATEGIES = {"fixed": fixed_chunks, "recursive": recursive_chunks, "semantic": semantic_chunks}


def main():
    names = sys.argv[1:] or list(STRATEGIES)
    root = Path(__file__).parent
    out_dir = root / "chunks"
    out_dir.mkdir(exist_ok=True)
    files = sorted(p for p in (root / "data").glob("*.txt"))
    for name in names:
        fn, total = STRATEGIES[name], 0
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for p in files:
                for i, c in enumerate(fn(p.read_text(encoding="utf-8"))):
                    f.write(json.dumps({"source": p.name, "chunk_id": i, "text": c}) + "\n")
                    total += 1
                print(f"  {name}: {p.name}", flush=True)
        print(f"{name}: {total} chunks from {len(files)} files")


if __name__ == "__main__":
    main()
