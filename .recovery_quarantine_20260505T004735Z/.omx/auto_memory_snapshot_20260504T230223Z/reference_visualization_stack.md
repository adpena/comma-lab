---
name: Visualization Stack — Dual Output (Paper + Web)
description: All visualizations must output both static (paper SVG) and interactive (web HTML). Quarto + Plotly + D3 + marimo.
type: reference
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Technology Stack (canonicalized)
- **Quarto**: dual-output paper — same .qmd source produces PDF (arXiv) + HTML (web)
- **marimo**: interactive reproducible notebooks (embedded in web version)
- **Manim**: cinematic math animations (gradient rank, optimization surfaces, training trajectories)
- **Observable Plot / Altair**: interactive charts (web) + static SVG (PDF)
- **DuckDB WASM**: in-browser SQL analytics on experiment data
- **adpena/molt**: Python→WASM compiler for GPU demos in browser
- **Rust/Zig→WASM**: high-performance interactive demos (codec visualization, etc.)
- **matplotlib**: static publication-quality fallback for PDF
- **Cloudflare**: free tier hosting for site + Pages + Workers
- **Pydantic/Monty**: data validation for experiment configs

## Output Directories
- `reports/graphs/site/{name}.html` — plotly interactive (Cloudflare site auto-deploys)
- `reports/graphs/paper/{name}.svg` — matplotlib static (Quarto PDF embeds)
- `reports/graphs/site/status.json` — live data for D3 dashboards

## Requirements for ALL Visualizations
1. DUAL format: every figure exists as both .html (interactive) and .svg (static)
2. Raw data saved as JSON for D3/marimo to consume independently
3. Publication-quality: no matplotlib defaults — custom colors, proper labels, clean axes
4. Interactive: hover tooltips with exact values, zoom, pan where appropriate
5. Consistent color scheme across all figures

## Key Figures Needed
1. **Score Timeline**: D3 interactive + static, all auth evals chronologically
2. **Drift Curve**: 11-checkpoint sweep (ep12500-16999), animated or interactive
3. **Gradient Rank**: PCA of Jacobian — rank-6 (output) vs rank-512 (embedding)
4. **Optimal Allocation**: Shannon's 3D surface (seg vs pose vs rate trade-off)
5. **TTO Before/After**: side-by-side frames with PoseNet heat map overlay
6. **Architecture Evolution**: codec→postfilter→renderer→TTO paradigm shifts

## Cloudflare Site
Already deployed. reports/graphs/site/ files auto-deploy.
URL is private (per CLAUDE.md strategic secrecy rule).
