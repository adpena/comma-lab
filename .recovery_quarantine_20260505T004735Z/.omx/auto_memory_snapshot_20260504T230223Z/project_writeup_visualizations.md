---
name: Writeup Visualization Plan
description: Generate scaling graphs, comparison GIFs, and Marimo notebook for best-writeup prize
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Format Options

**Marimo notebook** — reactive Python notebook that renders as static HTML.
- Each cell produces a figure + the code that made it
- Can embed in the Cloudflare site or submit as a standalone supplement
- `uv pip install marimo` then `marimo edit writeup.py`
- Export: `marimo export html writeup.py > writeup.html`
- Perfect for reproducibility story — reviewer can rerun every figure

**Comma-format GIFs** — match the README GIF style (tools/generate_comma_gif.py)

## Planned Visualizations

1. **Score vs h (width scaling)**: Log-linear fit + dilated breakout
2. **Score vs epoch (technique comparison)**: Standard / dilated / KL+hardframe overlaid
3. **Score component breakdown**: Stacked bar (seg/pose/rate) at each milestone
4. **Comparison GIFs**: Baseline vs ours with SegNet overlay
5. **Training trajectory timeline**: Score over dates
6. **Pareto frontier**: seg vs pose tradeoff at different operating points
7. **Boundary weight sensitivity**: Score vs boundary_weight

Data sources: reports/results.jsonl, reports/timeline.jsonl, experiment logs

**When:** After final checkpoint is confirmed. Re-run all h levels through
the same pipeline for fair comparison. Generate Marimo notebook with all
figures, export to HTML, embed in site.
