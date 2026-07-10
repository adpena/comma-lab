# Task-Aware Post-Filtering — Interactive marimo Notebook

`notebook.py` is a live, reactive [marimo](https://marimo.io) companion to the paper
**"Task-Aware Compression via Differentiable Scorer Optimization"** (`docs/writeup_draft.md`).

## What it implements

A 45 KB CNN post-filter trained by backpropagating through the frozen PoseNet and SegNet
scorer networks used by the comma.ai Video Compression Challenge. Instead of optimizing
generic perceptual quality, the filter learns pixel corrections that preserve the
information the downstream autonomous-driving models actually consume.

The notebook lets you:

- inspect the **score trajectory** loaded live from `reports/results.jsonl`;
- interactively **decompose the contest score** `100·seg + √(10·pose) + 25·rate` with sliders
  and watch the marginal sensitivities update;
- compare the current additive scoring formula against a proposed multiplicative
  (Arrow/Pareto complementarity) formula;
- read the **live research findings feed** from `.omx/research/findings.md`;
- explore the observed **PoseNet–SegNet Pareto frontier**.

Every visualization auto-loads from the repository, so the notebook stays in sync with the
research record as experiments accumulate.

## Run it

```bash
uv run marimo edit paper/notebook.py        # interactive
# or, static HTML export:
uv run marimo export html paper/notebook.py -o notebook.html
```

Requires `marimo` (any recent version) and the repository checked out so the notebook can
read `reports/results.jsonl`, `.omx/research/findings.md`, and `docs/writeup_draft.md`.

## Data provenance

All numbers are measured on the contest scorer (`upstream/evaluate.py`) over the 600-sample
set; no proxy or non-authoritative scores are presented as contest scores.
