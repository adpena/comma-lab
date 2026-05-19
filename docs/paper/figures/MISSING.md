# Missing figures for long-form writeup notes

This file enumerates figures referenced or expected in the paper that are
NOT yet present in `docs/paper/figures/`. Resolve these before any formal
long-form paper or technical report.

## Present (committed PNGs)

| File | Bytes | Referenced in |
|---|---:|---|
| `leaderboard_comparison.png` | 85.9 KB | TBD — cross-reference not yet wired in markdown |
| `score_decomposition.png` | 75.9 KB | TBD — cross-reference not yet wired in markdown |
| `step_curve_comparison.png` | 140.0 KB | TBD — cross-reference not yet wired in markdown |

The three present PNGs are not currently `![…](…)`-linked in any
`docs/paper/0*_*.md` source. Either wire them into the relevant sections
(§4 results, §3 gradient bug, §2 method) or move them to `figures/archive/`.

## Missing (referenced or expected)

### 6-panel diagnostic visualization

Per the repository's multipane diagnostic-figure specification:

  - Row 1: GT Original | Our Reconstruction | Pixel Error (hot colormap)
  - Row 2: GT SegNet masks | Our SegNet masks | SegNet Disagreement (red)

This visualization is the single most useful piece of evidence for both the
gradient-bug story (§3) and the contest write-up (the "best write-up" prize
explicitly calls out visualizations). It requires:

- TTO frames (`tto_frames.pt` from the Modal volume custody)
- GT video (`upstream/videos/0.mkv`)
- SegNet via `(B, T, C, H, W)` input format with `T=1` for the sequence dim

Generation script template lives in the repository root at
`scripts/generate_six_panel_diagnostic.py` (not yet in
`docs/paper/figures/` source-of-truth tree).

### Gradient-flow validation chart (§3.6)

The 1ms validation check (§3.6) would benefit from a before/after gradient
norm chart on a fixed batch — proves that the fix actually flows non-zero
gradients through the YUV6 conversion. Quick to generate with the existing
TTO loop logging.

### Lagrangian annealing curve (§2.3)

The Lagrangian-annealing phenomenon (temporarily reducing constraint caps
to explore the Pareto frontier, §2.3) is referenced but not visualized.
A 2D plot of `(rate_violation, distortion_violation)` per anneal step
would make the explanation concrete.

## Long-form writeup action items

- [ ] Generate the 6-panel diagnostic (highest priority — feeds both §3
      and the contest write-up prize PR)
- [ ] Wire the three existing PNGs into the markdown source with explicit
      `![](figures/...)` calls and figure captions
- [ ] Decide whether to generate the gradient-flow validation chart and the
      Lagrangian annealing curve (both ~30 min generation work) or defer
      them with a stated TODO in the relevant section
- [ ] Run a final pass to ensure every `![…](figures/…)` reference resolves
      and every committed PNG is referenced somewhere
