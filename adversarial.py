"""Run adversarial_set.jsonl through retrieval for every chunking strategy.

  python3 adversarial.py [--k 5]

Per question and strategy it reports:
  ranks      rank (within top-k) of the first hit for each evidence item, None = missed
  covered    every evidence item is hit somewhere in the top-k (a generator *could* answer)
  colocated  some single chunk in the whole corpus is a hit for ALL evidence items
             (False = the needed facts are structurally split by this chunking)
  full_rank  rank of the first hit for each evidence item over the ENTIRE index
  gap        top-1 cosine score minus the best gold-chunk score (0 = gold chunk is top-1)
  top        source#chunk of the top-3 retrieved chunks
"""
import argparse
import json
from pathlib import Path

from eval import STRATEGIES, first_hit_rank, is_hit
from rag import Retriever, load_chunks

ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    k = ap.parse_args().k
    items = [json.loads(line) for line in open(ROOT / "adversarial_set.jsonl", encoding="utf-8")]
    for s in STRATEGIES:
        retr, all_chunks = Retriever(s), load_chunks(s)
        print(f"\n=== {s} ===")
        for it in items:
            ranked = retr.search(it["question"], len(all_chunks))
            ranks = [first_hit_rank(ranked[:k], ev) for ev in it["evidence"]]
            full = [first_hit_rank(ranked, ev) for ev in it["evidence"]]
            colocated = any(all(is_hit(c, ev) for ev in it["evidence"]) for c in all_chunks)
            gold = [c["score"] for c in ranked if any(is_hit(c, ev) for ev in it["evidence"])]
            gap = ranked[0]["score"] - max(gold) if gold else None
            top = [f"{c['source'][:22]}#{c['chunk_id']}" for c in ranked[:3]]
            print(f"{it['id']} {it['category']:24} ranks={ranks} covered={all(r for r in ranks)} "
                  f"colocated={colocated} full_rank={full} gap={gap if gap is None else round(gap, 3)} "
                  f"top={top}")


if __name__ == "__main__":
    main()
