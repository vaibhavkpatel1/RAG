"""Recall-vs-latency benchmark for the IVF and HNSW indexes against exact search.

  python3 bench_ann.py              # measure -> results/recall_latency.csv, then plot
  python3 bench_ann.py --plot-only  # re-plot from the saved CSV

Queries: the questions in eval_set.jsonl and adversarial_set.jsonl, plus the first sentence of randomly
sampled chunks (fixed seed) up to N_QUERIES. Query embeddings are cached in results/ann_queries.npy.

recall@5  share of the exact top-5 chunks that the approximate index also returns (mean over queries)
latency   time of Retriever.search_vec for one query, query embedding excluded (it is identical for
          every method); best of REPEATS runs per query, then the mean over queries
scanned   mean share of the index whose distance to the query was computed
"""
import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

import numpy as np

from eval import STRATEGIES, load_eval_set
from rag import Retriever, embed, load_chunks

ROOT = Path(__file__).parent
OUT = ROOT / "results"
K, N_QUERIES, REPEATS = 5, 200, 3
NPROBES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 48)
EFS = (5, 8, 10, 15, 20, 30, 40, 60, 80, 120, 160, 240)


def load_queries():
    cache = OUT / "ann_queries.npy"
    if cache.exists():
        return np.load(cache)
    texts = [it["question"] for it in load_eval_set()]
    with open(ROOT / "adversarial_set.jsonl", encoding="utf-8") as f:
        texts += [json.loads(line)["question"] for line in f]
    pool = load_chunks("recursive")
    random.Random(0).shuffle(pool)
    for c in pool:
        if len(texts) >= N_QUERIES:
            break
        first = re.split(r"(?<=[.!?])\s+", c["text"].strip())[0][:300]
        if len(first) > 40 and "\\displaystyle" not in first:  # skip stubs and formula residue
            texts.append(first)
    q = np.vstack([embed(texts[i:i + 50], "search_query: ") for i in range(0, len(texts), 50)])
    OUT.mkdir(exist_ok=True)
    np.save(cache, q)
    return q


def timed(fn):
    best = float("inf")
    for _ in range(REPEATS):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best, out


def measure():
    qs = load_queries()
    print(f"{len(qs)} queries")
    rows = []
    for s in STRATEGIES:
        r = Retriever(s)
        n, nlist = len(r.vecs), len(r.ivf.centroids)
        r.hnsw  # build/load before timing
        exact = [{(c["source"], c["chunk_id"]) for c in r.search_vec(q, K)} for q in qs]

        def run(method, param, **kw):
            lat, rec, scanned = [], [], []
            for q, ex in zip(qs, exact):
                t, got = timed(lambda: r.search_vec(q, K, **kw))
                lat.append(t)
                rec.append(len({(c["source"], c["chunk_id"]) for c in got} & ex) / K)
                if method == "ivf":
                    scanned.append(len(r.ivf.candidates(q, param)) / n)
                elif method == "hnsw":
                    scanned.append(r.hnsw.last_evals / n)
                else:
                    scanned.append(1.0)
            rows.append(dict(strategy=s, method=method, param=param, latency_ms=np.mean(lat) * 1e3,
                             recall5=np.mean(rec), scanned=np.mean(scanned), n_vectors=n))
            print(f"  {s:9} {method:5} {param!s:>5} {rows[-1]['latency_ms']:7.3f} ms  "
                  f"recall {rows[-1]['recall5']:.3f}  scanned {rows[-1]['scanned']:.1%}", flush=True)

        run("exact", 0)
        for p in sorted({min(p, nlist) for p in NPROBES}):
            run("ivf", p, nprobe=p)
        for ef in EFS:
            run("hnsw", ef, ef=ef)
    OUT.mkdir(exist_ok=True)
    with open(OUT / "recall_latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, muted, grid, axis = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    style = {"ivf": dict(color="#2a78d6", marker="o", label="IVF (vary nprobe)"),
             "hnsw": dict(color="#eb6834", marker="s", label="HNSW, single layer (vary ef)")}
    with open(OUT / "recall_latency.csv") as f:
        rows = [{k: (float(v) if k in ("latency_ms", "recall5", "scanned")
                     else int(v) if k in ("param", "n_vectors") else v) for k, v in r.items()}
                for r in csv.DictReader(f)]

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "text.color": ink2,
                         "axes.labelcolor": ink2, "xtick.color": muted, "ytick.color": muted})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9), sharex=True, sharey=True, facecolor=surface)
    lo = min(r["recall5"] for r in rows)
    ymin = max(0.0, np.floor(lo * 10) / 10 - 0.05)
    for ax, s in zip(axes, STRATEGIES):
        ax.set_facecolor(surface)
        mine = [r for r in rows if r["strategy"] == s]
        for method in ("ivf", "hnsw"):
            pts = sorted((r for r in mine if r["method"] == method), key=lambda r: r["latency_ms"])
            x, y = [p["latency_ms"] for p in pts], [p["recall5"] for p in pts]
            st = style[method]
            ax.plot(x, y, color=st["color"], lw=2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
            ax.plot(x, y, st["marker"], color=st["color"], ms=8, mec=surface, mew=2, zorder=4)
            name = "nprobe" if method == "ivf" else "ef"
            for p in (min(pts, key=lambda r: r["param"]), max(pts, key=lambda r: r["param"])):
                ax.annotate(f"{name}={p['param']}", (p["latency_ms"], p["recall5"]),
                            textcoords="offset points", xytext=(0, -16 if method == "ivf" else 12),
                            ha="center", fontsize=8.5, color=ink2)
        ex = next(r for r in mine if r["method"] == "exact")
        ax.plot([ex["latency_ms"]], [1.0], "D", color=ink, ms=8, mec=surface, mew=2, zorder=5)
        ax.set_xscale("log")
        ax.set_ylim(ymin, 1.03)
        ax.set_title(f"{s} chunks ({mine[0]['n_vectors']:,} vectors)", loc="left", color=ink,
                     fontsize=11, fontweight="semibold")
        ax.grid(True, color=grid, lw=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(axis)
        ax.tick_params(length=0)
        ax.set_xlabel("Latency per query (ms, log scale)")
    axes[0].set_ylabel("Recall@5 vs exact search")
    handles = [plt.Line2D([], [], color=style[m]["color"], lw=2, marker=style[m]["marker"], ms=8,
                          mec=surface, mew=2, label=style[m]["label"]) for m in ("ivf", "hnsw")]
    handles.append(plt.Line2D([], [], color=ink, lw=0, marker="D", ms=8, mec=surface, mew=2,
                              label="Exact scan (recall 1.0 by definition)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01),
               labelcolor=ink)
    fig.text(0.012, 0.975, "Recall vs latency: IVF and HNSW against exact search", ha="left", va="top",
             fontsize=14, fontweight="semibold", color=ink)
    fig.text(0.012, 0.925, "Top-left is best (fast and accurate). Latency is the search step only, query "
             "embedding excluded; mean over queries, best of 3 runs each.", ha="left", va="top",
             fontsize=10, color=ink2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    fig.savefig(OUT / "recall_vs_latency.png", dpi=160, facecolor=surface)
    print("saved", OUT / "recall_vs_latency.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    if not ap.parse_args().plot_only:
        measure()
    plot()
