"""Download Wikipedia articles from a category (plus subcategories) as plain-text files."""
import json
import re
import sys
import time
from pathlib import Path

import requests

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "local-rag-builder/0.1 (personal project)"}
OUT = Path(__file__).parent / "data"

ROOT_CATEGORY = sys.argv[1] if len(sys.argv) > 1 else "Machine learning"
MAX_DEPTH = 1
MAX_ARTICLES = int(sys.argv[2]) if len(sys.argv) > 2 else 300


def api(**params):
    for attempt in range(6):
        r = requests.get(API, params={"format": "json", **params}, headers=HEADERS, timeout=30)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", 2 ** attempt * 2)))
            continue
        r.raise_for_status()
        time.sleep(0.5)
        return r.json()
    raise RuntimeError("Rate limited too many times")


def category_members(cat):
    pages, subcats, cont = [], [], {}
    while True:
        d = api(action="query", list="categorymembers", cmtitle=f"Category:{cat}",
                cmlimit=500, cmtype="page|subcat", **cont)
        for m in d["query"]["categorymembers"]:
            if m["ns"] == 0:
                pages.append(m["title"])
            elif m["ns"] == 14:
                subcats.append(m["title"].removeprefix("Category:"))
        if "continue" not in d:
            return pages, subcats
        cont = d["continue"]


def collect_titles():
    seen, frontier, titles = {ROOT_CATEGORY}, [ROOT_CATEGORY], []
    for _ in range(MAX_DEPTH + 1):
        nxt = []
        for cat in frontier:
            if len(titles) >= MAX_ARTICLES:
                break
            pages, subs = category_members(cat)
            titles += [t for t in pages if t not in titles]
            nxt += [s for s in subs if s not in seen]
            seen.update(subs)
        frontier = nxt
    return titles[:MAX_ARTICLES]


def fetch_extracts(titles):
    """Fetch full plain-text extracts; the API serves only one full extract per request."""
    for title in titles:
        d = api(action="query", prop="extracts", explaintext=1, titles=title, redirects=1)
        for p in d["query"]["pages"].values():
            if p.get("extract"):
                yield p["title"], p["pageid"], p["extract"]


def main():
    OUT.mkdir(exist_ok=True)
    titles = collect_titles()
    print(f"Found {len(titles)} articles under '{ROOT_CATEGORY}'")
    saved = 0
    for title, pageid, text in fetch_extracts(titles):
        if len(text) < 500:  # skip stubs
            continue
        slug = re.sub(r"[^\w\-]+", "_", title)[:100]
        (OUT / f"{slug}.txt").write_text(text, encoding="utf-8")
        with open(OUT / "_manifest.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"title": title, "pageid": pageid, "chars": len(text),
                                "url": f"https://en.wikipedia.org/?curid={pageid}"}) + "\n")
        saved += 1
    print(f"Saved {saved} articles to {OUT}")


if __name__ == "__main__":
    (OUT / "_manifest.jsonl").unlink(missing_ok=True)
    main()
