"""IVF vs exact search: how much is scanned, and how much accuracy is lost.

  python3 ivf_eval.py

Per strategy and nprobe (queries = the questions in eval_set.jsonl):
  scanned    mean fraction of the index's vectors compared with the query
  overlap@5  mean fraction of the exact top-5 chunks that IVF also returns (1.0 = identical)
  recall@5   fraction of questions whose top-5 holds a gold-evidence chunk (same rule as eval.py)
"""
import numpy as np

from eval import first_hit_rank, load_eval_set, STRATEGIES
from rag import Retriever, embed

K = 5


def main():
    items = load_eval_set()
    qvecs = embed([it["question"] for it in items], "search_query: ")
    for s in STRATEGIES:
        retr = Retriever(s)
        nlist = len(retr.ivf.centroids)
        exact = [[(c["source"], c["chunk_id"]) for c in retr.search_vec(q, K)] for q in qvecs]
        print(f"\n{s}: {len(retr.vecs)} vectors, nlist={nlist}")
        print(f"{'nprobe':>7} {'scanned':>8} {'overlap@5':>10} {'recall@5':>9}")
        for nprobe in sorted({1, 2, 4, 8, 16, nlist}):
            if nprobe > nlist:
                continue
            scanned, overlap, hits = [], [], 0
            for q, ex, it in zip(qvecs, exact, items):
                scanned.append(len(retr.ivf.candidates(q, nprobe)) / len(retr.vecs))
                got = retr.search_vec(q, K, nprobe=nprobe)
                overlap.append(len({(c["source"], c["chunk_id"]) for c in got} & set(ex)) / K)
                hits += first_hit_rank(got, it) is not None
            print(f"{nprobe:>7} {np.mean(scanned):>8.1%} {np.mean(overlap):>10.2f} {hits / len(items):>9.2f}")


if __name__ == "__main__":
    main()
