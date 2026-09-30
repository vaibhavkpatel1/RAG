"""HNSW (single layer) vs exact search: how much is evaluated, and how much accuracy is lost.

  python3 hnsw_eval.py

Per strategy and ef (queries = the questions in eval_set.jsonl):
  scanned    mean fraction of the index's vectors whose distance to the query was computed
  overlap@5  mean fraction of the exact top-5 chunks that the graph search also returns
  recall@5   fraction of questions whose top-5 holds a gold-evidence chunk (same rule as eval.py)
"""
import time

import numpy as np

from eval import first_hit_rank, load_eval_set, STRATEGIES
from rag import Retriever, embed

K = 5
EFS = (5, 10, 20, 40, 80, 160)


def main():
    items = load_eval_set()
    qvecs = embed([it["question"] for it in items], "search_query: ")
    for s in STRATEGIES:
        t = time.perf_counter()
        retr = Retriever(s)
        g = retr.hnsw
        deg = [len(nb) for nb in g.neighbors]
        print(f"\n{s}: {len(retr.vecs)} vectors, M={g.M}, mean degree {np.mean(deg):.1f}, "
              f"reachable {g.reachable()}/{len(deg)}, load/build {time.perf_counter() - t:.1f}s")
        exact = [[(c["source"], c["chunk_id"]) for c in retr.search_vec(q, K)] for q in qvecs]
        print(f"{'ef':>5} {'scanned':>8} {'overlap@5':>10} {'recall@5':>9}")
        for ef in EFS:
            scanned, overlap, hits = [], [], 0
            for q, ex, it in zip(qvecs, exact, items):
                got = retr.search_vec(q, K, ef=ef)
                scanned.append(g.last_evals / len(retr.vecs))
                overlap.append(len({(c["source"], c["chunk_id"]) for c in got} & set(ex)) / K)
                hits += first_hit_rank(got, it) is not None
            print(f"{ef:>5} {np.mean(scanned):>8.1%} {np.mean(overlap):>10.2f} {hits / len(items):>9.2f}")


if __name__ == "__main__":
    main()
