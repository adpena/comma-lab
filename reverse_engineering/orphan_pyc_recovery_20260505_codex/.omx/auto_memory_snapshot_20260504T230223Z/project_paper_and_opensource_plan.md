---
name: Paper, Open-Source, and Codebase Canonicalization Plan
description: arXiv-style paper (Quarto+marimo), open-source repo cleanup, lossless extraction, tone guidelines
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Paper Stack
- **Quarto**: dual-output paper — same .qmd source produces PDF (arXiv) + HTML (web)
- **marimo**: interactive reproducible notebooks embedded in the web version
- **Manim**: cinematic math animations (gradient rank, optimization surfaces)
- **Observable Plot / Altair**: interactive charts for web, static SVG for PDF

## Paper Tone
"Oxford Cambridge Harvard meets Silicon Valley — genius and cool and unpretentious
and understated clean genius that speaks for itself."
- arXiv format and rigor WITHOUT pretension
- Author has no formal background in this domain — be honest about that
- Scientific method, reproducible results, rigorous claims
- No filler, no hedging, no academic bloat
- Streamline into main body + addendum for details
- Direct, high-signal, descriptive language
- Let the work speak for itself

## Citation Style
- arXiv preprint format (not journal submission)
- BibTeX references where appropriate
- Keep citations minimal — cite what's necessary, not what's impressive
- Self-cite the repo commit history as methodology evidence

## Open-Source Plan
The repo will be open-sourced alongside the writeup. Requires:

### Codebase Cleanup
1. **Extract lossless/** — move entire `src/tac/lossless/` directory and ALL related
   CLI, tests, configs into a separate repo called `lossless-research`
   - src/tac/lossless/*.py
   - src/tac/tests/test_tac_lossless_*.py
   - Any CLI commands that reference lossless modules
   - Any imports of lossless modules from other files
2. **Remove dead code** — unused experiments, stale configs, orphaned scripts
3. **Canonicalize** — consistent naming, no code smell, proper docstrings
4. **Sensitive data** — remove credentials, private paths, internal URLs
5. **CLAUDE.md** — redact competition-sensitive instructions before open-sourcing
6. **.ralph/** and **.omx/** — decide what's public research record vs private

### What Stays in pact/
- src/tac/ (minus lossless/)
- experiments/ (training scripts, analysis)
- submissions/
- docs/ (paper, analysis)
- reports/ (visualizations, data)
- scripts/ (tooling)

### What Goes to lossless-research/
- src/tac/lossless/
- src/tac/tests/test_tac_lossless_*
- Related CLI, configs, data
