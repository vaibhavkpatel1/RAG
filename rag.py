"""Index chunks with nomic-embed-text, retrieve by cosine similarity, answer with llama3.2.

CLI: python3 rag.py index [strategy ...]   (embeds chunks/<strategy>.jsonl -> indexes/<strategy>.npy)
     python3 rag.py ask <strategy> "question" [cosine|l2] [nprobe | ivf:N | hnsw:ef]   (default: exact)
     python3 rag.py ivf [strategy ...]          (k-means -> indexes/<strategy>.ivf.npz)
     python3 rag.py hnsw [strategy ...]         (graph -> indexes/<strategy>.hnsw.npz)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import requests

from hnsw import HNSWIndex
from ivf import IVFIndex

ROOT = Path(__file__).parent
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")  # main server: embeddings + RAG LLM
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = os.environ.get("RAG_MODEL", "llama3.2")

SYSTEM = ("Answer the question using only the provided context. "
          "If the context does not contain the answer, say you don't know.")


def embed(texts, prefix):
    r = requests.post(f"{OLLAMA}/api/embed", timeout=300,
                      json={"model": EMBED_MODEL, "input": [prefix + t for t in texts]})
    r.raise_for_status()
    v = np.array(r.json()["embeddings"], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def load_chunks(strategy):
    with open(ROOT / "chunks" / f"{strategy}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_index(strategy, batch=16):
    path = ROOT / "indexes" / f"{strategy}.npy"
    if path.exists():
        return
    path.parent.mkdir(exist_ok=True)
    texts = [c["text"] for c in load_chunks(strategy)]
    vecs = []
    for i in range(0, len(texts), batch):
        vecs.append(embed(texts[i:i + batch], "search_document: "))
        print(f"  {strategy}: embedded {min(i + batch, len(texts))}/{len(texts)}", flush=True)
    np.save(path, np.vstack(vecs))


def build_ivf(strategy, vecs):
    """Load indexes/<strategy>.ivf.npz, training k-means and saving it if missing or stale."""
    path = ROOT / "indexes" / f"{strategy}.ivf.npz"
    if path.exists():
        ivf = IVFIndex.load(path)
        if len(ivf.labels) == len(vecs):
            return ivf
    ivf = IVFIndex.build(vecs)
    ivf.save(path)
    return ivf


def build_hnsw(strategy, vecs):
    """Load indexes/<strategy>.hnsw.npz, building the graph and saving it if missing or stale."""
    path = ROOT / "indexes" / f"{strategy}.hnsw.npz"
    if path.exists():
        g = HNSWIndex.load(path, vecs)
        if len(g.neighbors) == len(vecs):
            return g
    g = HNSWIndex.build(vecs)
    g.save(path)
    return g


class Retriever:
    def __init__(self, strategy):
        build_index(strategy)
        self.strategy = strategy
        self.chunks = load_chunks(strategy)
        self.vecs = np.load(ROOT / "indexes" / f"{strategy}.npy")
        assert len(self.chunks) == len(self.vecs), "index out of date; delete indexes/*.npy"
        self._ivf = self._hnsw = None

    @property
    def ivf(self):
        if self._ivf is None:
            self._ivf = build_ivf(self.strategy, self.vecs)
        return self._ivf

    @property
    def hnsw(self):
        if self._hnsw is None:
            self._hnsw = build_hnsw(self.strategy, self.vecs)
        return self._hnsw

    def search(self, question, k=5, metric="cosine", nprobe=None, ef=None):
        return self.search_vec(embed([question], "search_query: ")[0], k, metric, nprobe, ef)

    def search_vec(self, q, k=5, metric="cosine", nprobe=None, ef=None):
        """Default: exact scan of every vector. nprobe=n: IVF, scan only the n nearest clusters.
        ef=n: HNSW graph walk keeping the n closest nodes (n is raised to k if smaller)."""
        if nprobe is not None and ef is not None:
            raise ValueError("pass nprobe (IVF) or ef (HNSW), not both")
        if ef is not None:
            ids = self.hnsw.candidates(q, max(ef, k))
        else:
            ids = np.arange(len(self.vecs)) if nprobe is None else self.ivf.candidates(q, nprobe)
        vecs = self.vecs[ids]
        if metric == "cosine":
            scores = vecs @ q
        elif metric == "l2":
            scores = -np.linalg.norm(vecs - q, axis=1)  # negated so higher = closer, like cosine
        else:
            raise ValueError(f"unknown metric {metric!r}; use 'cosine' or 'l2'")
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[ids[i]], "score": float(scores[i])} for i in top]


def answer(question, chunks):
    context = "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks))
    r = requests.post(f"{OLLAMA}/api/chat", timeout=600, json={
        "model": LLM_MODEL, "stream": False, "options": {"temperature": 0},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]})
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def main():
    cmd = sys.argv[1]
    if cmd == "index":
        for s in sys.argv[2:] or ["fixed", "recursive", "semantic"]:
            build_index(s)
    elif cmd == "ivf":
        for s in sys.argv[2:] or ["fixed", "recursive", "semantic"]:
            r = Retriever(s)
            print(f"{s}: {len(r.ivf.centroids)} clusters, list sizes "
                  f"min/mean/max = {min(map(len, r.ivf.lists))}/{len(r.vecs) // len(r.ivf.lists)}/"
                  f"{max(map(len, r.ivf.lists))}")
    elif cmd == "hnsw":
        for s in sys.argv[2:] or ["fixed", "recursive", "semantic"]:
            r = Retriever(s)
            deg = [len(nb) for nb in r.hnsw.neighbors]
            print(f"{s}: {len(deg)} nodes, neighbours per node min/mean/max = "
                  f"{min(deg)}/{sum(deg) / len(deg):.1f}/{max(deg)}, "
                  f"reachable from entry: {r.hnsw.reachable()}/{len(deg)}")
    elif cmd == "ask":
        strategy, q = sys.argv[2], sys.argv[3]
        metric = sys.argv[4] if len(sys.argv) > 4 else "cosine"
        mode = sys.argv[5] if len(sys.argv) > 5 else ""
        kind, _, n = mode.rpartition(":")
        nprobe = int(n) if mode and kind in ("", "ivf") else None
        ef = int(n) if kind == "hnsw" else None
        hits = Retriever(strategy).search(q, metric=metric, nprobe=nprobe, ef=ef)
        print(answer(q, hits))
        print("\nSources:", [(h["source"], h["chunk_id"], round(h["score"], 3)) for h in hits])


if __name__ == "__main__":
    main()
