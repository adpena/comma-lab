# lab notebook

Last refreshed: `2026-04-08 18:55:54 -0500`

## executive summary

- current honest Track B floor: **`1.95`**
- current honest Track B bytes: `864,168`
- local rule-faithful estimate: `1.9923143481540724` at `935,149` bytes
- current gap to the public leaderboard:
  - matches first at `1.95`
  - `0.03` better than second (`1.98`)

This notebook is the layered version of the writeup. It is meant to be readable in two ways:
- fast skim: executive summary, notebook map, key figures
- deep dive: runnable commands, evidence links, glossary, appendices

## notebook map

1. Executive summary
2. Technique impact snapshots
3. Runnable reproduction snippets
4. Evidence index and methodology
5. Glossary
6. Appendices for deeper inspection

## technique impact snapshots

### promoted `1.95` floor

- branch: `long500_qat_ema_alpha20_h32`
- idea: keep the same tiny shipped operator, but train it much longer with QAT+EMA stability
- result: scorer-backed promotion from `1.99` to `1.95`

<video controls muted playsinline preload="metadata" width="100%">
  <source src="./media/inflated_preview.mp4" type="video/mp4">
</video>

### promoted floor zoomed inspection

This is the same promoted decode shown at a tighter crop so the visual effect is easier to inspect.

<video controls muted playsinline preload="metadata" width="100%">
  <source src="./media/inflated_zoom_preview.mp4" type="video/mp4">
</video>

### rejected ROI stack

- idea: combine ROI-style structure protection with the learned post-filter
- result: scorer-backed reject at `2.10`
- lesson: archive/distribution mismatch and SegNet regression were large enough to erase the proxy win

### rejected scorer-faithful `v2` family

- `v2 h16` official upstream proxy: `2.04`
- `v2 h32` official upstream proxy: `2.04`
- lesson: fixing the loss family improved rigor, but did not yet beat the longer-horizon QAT+EMA branch

## runnable snippets

### 1. authoritative local scorer

```bash
source .venv/bin/activate
comma-lab eval-submission robust_current --device cpu
```

### 2. official-path faithful proxy

This uses the upstream `evaluate.py` end-to-end on a temp submission tree.

```bash
uv run --with torch --with av --with safetensors --with timm --with einops \
  --with segmentation-models-pytorch --with numpy \
  python experiments/proxy_score_faithful.py \
  submissions/robust_current/postfilter_int8.pt
```

### 3. static site rebuild

```bash
python3 reports/graphs/build_dashboard.py
python3 reports/graphs/build_static_site.py
```

## methodology and evidence

- methodology: `../../docs/lab_methodology.md`
- evidence index: `./evidence_index.md`
- glossary: `./glossary.md`
- current live status: `../../reports/latest.md`

## live follow-on work

- observed at `2026-04-08 18:55:54 -0500`: local managed session `long500_qat_ema_alpha20_h24`
- latest seen checkpoint: epoch `200`, scorer `4.1066`, PoseNet `0.058720`, SegNet `0.035090`
- new saved-artifact candidate at `2026-04-08 18:33:00 -0500`: `long1000_qat_ema_alpha20_h16` finished locally with `4.5179 -> 4.1515` (`-0.3664`)
- official upstream-path proxy for that h16 artifact landed at `2026-04-08 18:55:54 -0500`: PoseNet `0.05890975`, SegNet `0.00579293`, estimated score `1.9222`
- interpretation: h24 remains weaker than the promoted h32 branch, while long1000 h16 is strong enough to justify the active authoritative scorer run

## glossary

See the standalone `glossary.md` for definitions used throughout the notebook and packet.

## appendices

### appendix A — promoted floor evidence

- `reports/raw/robust_current-current_workflow-cpu-summary.json`
- `reports/raw/robust_current-current_workflow-cpu-report.txt`
- `reports/raw/2026-04-08-long500-h32-authoritative/robust_current-long500-h32-smoke.json`

### appendix B — close misses

- `v2 h16`: official upstream proxy `2.04`
- `v2 h32`: official upstream proxy `2.04`
- `alpha10 h32`: scorer-backed `2.03`

### appendix C — remote lanes

- `mini`: official proxy/eval side lane
- `bat00`: authenticated CUDA lane under hardening
- `molt`: reachable but not yet admitted
