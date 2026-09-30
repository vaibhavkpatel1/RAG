# Adversarial retrieval queries

Set: `adversarial_set.jsonl` (5 queries). Reproduce with `python3 adversarial.py` (k=5, retrieval only —
the generator was not run). A query "fails" when no top-5 chunk is a hit (same article and >= 60% of the
evidence trigrams, the same rule as `eval.py`). `full_rank` is the gold chunk's rank over the whole index.

| id | failure type | fixed | recursive | semantic |
|----|--------------|-------|-----------|----------|
| A1 | facts split across chunks | **miss** (1993 fact at rank 6) | **miss** (rank 7) | pass (ranks 1, 2) |
| A2 | low overlap: paraphrase | **miss** (rank 10) | pass (rank 3) | pass (rank 1) |
| A3 | low overlap: formula | **miss** (rank 255) | pass (rank 1) | **miss** (rank 171) |
| A4 | ambiguous pronoun | **miss** (rank 133) | **miss** (rank 46) | **miss** (rank 97) |
| A5 | ambiguous entity | **miss** (rank 199) | **miss** (rank 336) | **miss** (rank 62) |

Only A4 and A5 fail under all three strategies. A1-A3 are strategy-dependent, which is the more useful result
for a chunking comparison.

## A1 — "How many years after joining the Université de Montréal faculty did Bengio receive the Turing Award?"
Needs two facts in `Yoshua_Bengio.txt` that are 1,985 characters apart (Turing Award at offset 134,
"faculty member ... since 1993" at offset 2119). No single fixed (1000) or recursive (~850) chunk holds both
(`colocated=False`). The question is dominated by "Bengio" and "Turing Award", so the 1993 chunk is
outranked by other Bengio chunks and lands at rank 6 (fixed) / 7 (recursive), just outside k=5. Semantic
chunks are larger (up to 2000 chars), so the two facts sit in adjacent chunks that both get retrieved.
This is a near-miss: k=7 would recover it. Arithmetic needs *both* facts, so one missing chunk is fatal.

## A2 — "Which pre-2010 collection of shrunken web photos was pulled by its authors because captions contained hateful language?"
The query shares almost no vocabulary with the article ("shrunken web photos" vs "32x32 scaled-down images",
"pulled" vs "retired", "captions" vs "labels", "hateful language" vs "slurs"), and never names the dataset.
The embedding follows the most distinctive topic, hate speech and offensive labels, to `Algorithmic_bias` and
other bias articles (top-1 is a Facebook hate-speech passage). Fixed also dilutes the gold chunk: chunk #2 is
mostly about dataset construction (Zipf distribution, noun counts) and the retirement sentence is at its very
end. Semantic chunking aligned a boundary with the "Retirement" section, so the chunk is on-topic and it wins.

## A3 — "What is the formula for F1?"
The formula was extracted from Wikipedia as flattened markup: one character per line
(`p | r | e | c | i | s | ...`), so the gold chunks are 14-53% alphabetic. Embedding models see noise, and
"F1" also appears in many unrelated ML articles. The gold chunk ranks 255 (fixed) and 171 (semantic).
Recursive ranks it first even though its gold chunk is 14% alphabetic; I did not determine why, so treat that as
luck rather than a strength of recursive chunking. This is a data-cleaning problem, not a chunking one.

## A4 — "When was it first proposed?"
No referent: "it" could be any of the 57 fixed chunks containing "proposed". The query embeds to generic
"first proposed" text, and the top-1 is an unrelated passage (a journal founded by Rashevsky). No retriever can
succeed here; the correct system behavior is to ask which entity is meant, or to abstain. The generator's
"say you don't know if the context lacks the answer" instruction is the only safeguard, and I have not tested
whether `llama3.2` follows it when handed plausible-looking but irrelevant chunks. This is the realistic
multi-turn follow-up case.

## A5 — "How is precision defined?"
Precision is defined in several articles (`Binary_classification`, `Evaluation_of_binary_classifiers`), and
the query names no context. The top-1 for fixed is an `Evaluation_of_binary_classifiers` chunk that *does* define
precision correctly. It counts as a miss only because the eval labels a single gold article. So this is a
labeling artifact, not a retrieval failure: the metric under-counts valid answers when several sources are
correct. Fixing it would need multiple accepted gold sources per question.

## Attempts that did NOT fail (not in the set)
- Two facts in the small `80_Million_Tiny_Images` article (subtracting 79,302,017 from 97,245,098): not
  co-located, but the article has only ~3 chunks, so all of them land in the top 5.
- "Which was released first, ELMo or 80 Million Tiny Images?": both named entities pulled their chunks (ranks 1-3).
- "Where was he a postdoctoral fellow, and who supervised him?": "he" was resolved anyway, because
  "postdoctoral fellow ... supervised" is rare enough phrasing to find the Bengio chunk (rank 1 for fixed and
  recursive, rank 3 for semantic).

Multi-fact questions only break when the article is large, the facts are far apart, and the query is dominated
by one fact.

## Caveats
Five hand-picked queries are anecdotes, not a benchmark. Retrieval only: I did not run generation or the
faithfulness judge on these, so I don't know what the LLM answers for the missed queries.
