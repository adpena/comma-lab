---
name: Paper Tooling — Marimo + Quarto + LaTeX
description: Marimo for interactive site (WASM), Quarto/LaTeX for arXiv PDF, shared assets in repo
type: reference
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Marimo capabilities
- WASM HTML export → Cloudflare Pages (interactive, reactive)
- LaTeX equations work (inline + display)
- NO citations/BibTeX (open issue #4756)
- NO LaTeX export, NO arXiv formatting
- Excellent for interactive data exploration

## Quarto capabilities
- arXiv template exists: github.com/mikemahoney218/quarto-arxiv
- Single .qmd → PDF + HTML from same source
- Native BibTeX, figure captions, TOC, section numbering
- Can embed marimo islands for interactivity in HTML output
- Python code execution via jupyter kernel

## WINNER: Quarto + marimo + Manim (single source, dual output)

Pipeline:
```
paper/paper.qmd (single source)
  ├── quarto render --to html → Cloudflare Pages (interactive, marimo cells)
  └── quarto render --to pdf  → arXiv (LaTeX, BibTeX, static figures)
```

Key features:
- Official quarto-marimo plugin for reactive Python cells in HTML
- Conditional content: `::: {.content-visible when-format="html"}` for interactive-only
- quarto-arxiv template passes arXiv automated checks
- Plotly interactive in HTML, auto-converted to static PNG in PDF via kaleido
- Tufte/margin-note layouts replicate Distill.pub aesthetic

## Tools to install
```bash
brew install quarto
uv pip install jupyter plotly kaleido manim marimo
quarto install extension mikemahoney218/quarto-arxiv
quarto install extension marimo-team/quarto-marimo
```

## Alternatives considered and rejected
- Typst: beautiful PDF but HTML export is experimental/unusable
- MyST: good but web aesthetics lag behind Quarto
- Curvenote: just MyST with managed hosting (vendor lock-in)
