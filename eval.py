"""Evaluation: retrieval recall@k / MRR against labeled evidence, and LLM-as-judge faithfulness.

  python3 eval.py retrieval    [--strategies fixed recursive semantic]
  python3 eval.py faithfulness [--strategies ...] [--k 3] [--judge-model M] [--judge-url U]
  python3 eval.py sanity       [--judge-model M] [--judge-url U]   # is the judge itself trustworthy?

Metrics (all implemented here, no eval framework):
  recall@k     fraction of questions where at least one of the top-k chunks is a "hit".
  MRR@k        mean of 1/rank of the first hit (0 if none in top-k).
  faithfulness fraction of the answer's claims that the judge finds supported by the retrieved
               chunks (per-question score, averaged). Measures groundedness, not correctness.

A chunk is a hit when it comes from the gold article and contains at least HIT_THRESHOLD of the
gold evidence's word trigrams. Labeling by evidence rather than chunk id makes the same eval set
comparable across chunking strategies, and a quote split across a chunk boundary can still count.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests

from rag import Retriever, answer

ROOT = Path(__file__).parent
STRATEGIES = ["fixed", "recursive", "semantic"]
KS = (1, 3, 5)
HIT_THRESHOLD = 0.6


def load_eval_set():
    with open(ROOT / "eval_set.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def trigrams(text):
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


def is_hit(chunk, item):
    if chunk["source"] != item["source"]:
        return False
    ev = trigrams(item["evidence"])
    return bool(ev) and len(ev & trigrams(chunk["text"])) / len(ev) >= HIT_THRESHOLD


def first_hit_rank(chunks, item):
    for rank, c in enumerate(chunks, 1):
        if is_hit(c, item):
            return rank
    return None


# ---------- retrieval ----------

def eval_retrieval(strategies, metric="cosine"):
    items = load_eval_set()
    print(f"{len(items)} questions; hit = same article and >= {HIT_THRESHOLD:.0%} of evidence trigrams\n")
    print(f"{'strategy':10}" + "".join(f"recall@{k:<3}" for k in KS) + f"{'MRR@5':8}misses@5")
    for s in strategies:
        retr, ranks = Retriever(s), []
        for it in items:
            ranks.append(first_hit_rank(retr.search(it["question"], max(KS), metric), it))
        recalls = [sum(r is not None and r <= k for r in ranks) / len(ranks) for k in KS]
        mrr = sum(1 / r for r in ranks if r) / len(ranks)
        missed = [it["id"] for it, r in zip(items, ranks) if r is None]
        print(f"{s:10}" + "".join(f"{x:<9.2f}" for x in recalls) + f"{mrr:<8.2f}{missed}")


# ---------- LLM-as-judge ----------

class Judge:
    def __init__(self, model, url):
        self.model, self.url = model, url

    def _ask(self, prompt, schema):
        r = requests.post(f"{self.url}/api/chat", timeout=900, json={
            "model": self.model, "stream": False, "format": schema, "options": {"temperature": 0},
            "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return json.loads(r.json()["message"]["content"])

    def claims(self, question, answer_text):
        out = self._ask(
            "Break the answer into a list of short, self-contained factual claims. "
            "Each claim must make sense on its own (replace pronouns with what they refer to). "
            "Do not add anything that is not in the answer. If the answer says it doesn't know "
            "or gives no factual claims, return an empty list.\n\n"
            f"Question: {question}\nAnswer: {answer_text}",
            {"type": "object", "properties": {"claims": {"type": "array", "items": {"type": "string"}}},
             "required": ["claims"]})
        return [c for c in out["claims"] if c.strip()]

    def supported(self, claim, context):
        out = self._ask(
            "You are checking whether a claim is supported by a context. Say supported=true only if "
            "the context explicitly states the claim or it follows directly from it. Ignore what you "
            "know from elsewhere: a claim that is true in the real world but absent from the context "
            "is NOT supported.\n\n"
            f"Context:\n{context}\n\nClaim: {claim}",
            {"type": "object", "properties": {"reason": {"type": "string"}, "supported": {"type": "boolean"}},
             "required": ["reason", "supported"]})
        return out["supported"], out["reason"]

    def faithfulness(self, question, answer_text, context):
        """Returns (score or None if no claims, per-claim details)."""
        details = []
        for c in self.claims(question, answer_text):
            ok, why = self.supported(c, context)
            details.append({"claim": c, "supported": ok, "reason": why})
        if not details:
            return None, details
        return sum(d["supported"] for d in details) / len(details), details


def eval_faithfulness(strategies, k, judge):
    items = load_eval_set()
    (ROOT / "results").mkdir(exist_ok=True)
    print(f"generator: {os.environ.get('RAG_MODEL', 'llama3.2')}   judge: {judge.model} @ {judge.url}   k={k}\n")
    for s in strategies:
        retr, path = Retriever(s), ROOT / "results" / f"faithfulness_{s}_k{k}.jsonl"
        done = {}
        if path.exists():  # resumable: skip questions already judged with this judge
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                if r["judge"] == judge.model:
                    done[r["id"]] = r
        with open(path, "a", encoding="utf-8") as f:
            for it in items:
                if it["id"] in done:
                    continue
                chunks = retr.search(it["question"], k)
                ans = answer(it["question"], chunks)
                context = "\n\n".join(c["text"] for c in chunks)
                score, details = judge.faithfulness(it["question"], ans, context)
                rec = {"id": it["id"], "judge": judge.model, "question": it["question"], "answer": ans,
                       "retrieved_hit": first_hit_rank(chunks, it) is not None,
                       "faithfulness": score, "claims": details}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done[it["id"]] = rec
                print(f"  {s} q{it['id']:<2} faithfulness={score}", flush=True)
        scored = [r["faithfulness"] for r in done.values() if r["faithfulness"] is not None]
        skipped = len(done) - len(scored)
        print(f"{s:10} mean faithfulness = {sum(scored) / len(scored):.3f} "
              f"({len(scored)} scored, {skipped} with no claims)\n")


def eval_sanity(judge):
    """Judge the gold answer against (a) its own evidence -> should be ~1, (b) another question's
    evidence -> should be ~0. If not, the judge can't be trusted for faithfulness."""
    items = load_eval_set()
    good, bad = [], []
    for i, it in enumerate(items):
        other = items[(i + 7) % len(items)]
        if other["source"] == it["source"]:
            other = items[(i + 3) % len(items)]
        g, _ = judge.faithfulness(it["question"], it["answer"], it["evidence"])
        b, _ = judge.faithfulness(it["question"], it["answer"], other["evidence"])
        good.append(g)
        bad.append(b)
        print(f"  q{it['id']:<2} matching-context={g}  mismatched-context={b}", flush=True)
    mean = lambda xs: sum(x for x in xs if x is not None) / max(1, sum(x is not None for x in xs))
    print(f"\njudge sanity: matching context {mean(good):.2f} (want ~1.0), "
          f"mismatched context {mean(bad):.2f} (want ~0.0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["retrieval", "faithfulness", "sanity"])
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES, choices=STRATEGIES)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--metric", choices=["cosine", "l2"], default="cosine", help="retrieval eval only")
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "gemma4"))
    ap.add_argument("--judge-url", default=os.environ.get("JUDGE_URL", "http://localhost:11435"))
    a = ap.parse_args()
    judge = Judge(a.judge_model, a.judge_url)
    if a.command == "retrieval":
        eval_retrieval(a.strategies, a.metric)
    elif a.command == "faithfulness":
        eval_faithfulness(a.strategies, a.k, judge)
    else:
        eval_sanity(judge)


if __name__ == "__main__":
    sys.exit(main())
