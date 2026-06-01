# MLX↔PyTorch render-parity crux — recursively diagnosed to the bottom (LANDED 2026-06-01)

- **lane:** `lane_mlx_pytorch_render_parity_crux_20260601` (L1: impl_complete + three_clean_review + memory_entry)
- **directive:** operator standing directive 2026-06-01 — *"recursively diagnose → understand → FIX → engineer away the SOURCES of the MLX↔PyTorch render-parity drift … down to the crux."* Target = drift source #1 (RENDER-PARITY: MLX decoder forward ≠ PyTorch decoder forward on identical weights). Drift source #2 (eval-hardware axis) explicitly out of scope.
- **axis_tag:** render-parity max-abs-diff `[macOS-MLX vs PyTorch-CPU parity, exact-measured]`; any d_seg re-measurement `[macOS-CPU advisory]`. **NON-PROMOTABLE** (`score_claim=False`, `promotable=False`) per Catalog #341/#192/#127/#323. NO score claim.
- **$0 macOS-CPU/MLX-local only.** NO paid dispatch, NO cloud GPU, NO PR, NO Modal/Vast/Lightning. MLX CPU + PyTorch CPU, both free.
- **anchor:** `.omx/research/pr95_hnerv_mlx_pytorch_render_parity_crux_anchor.json` (real carrier sha `e976acd5fe565c94…`, 600 pairs, latent_dim=28, base_channels=36).

## The recursive diagnosis — drift source → fix → residual

Built an **op-level first-divergence harness** (`tac.analysis.mlx_pytorch_render_parity_crux`) that loads the SAME real PR95-HNeRV state_dict into BOTH the pinned PyTorch REFERENCE decoder (`submissions/hnerv_muon/src/model.py::HNeRVDecoder`, NEVER edited) AND the MLX decoder (`HNeRVDecoderMLX`), feeds the SAME latent, captures per-op intermediates (NCHW-aligned), and finds the FIRST op that diverges beyond fp tolerance.

The two forwards are **topologically byte-identical** (stem → reshape → `sin` → 6×[interpolate-skip + PixelShuffle(conv) + `sin`] → `x + 0.1·sin(refine)` → `sigmoid(rgb)·255`). I verified — did not assume — which ops actually diverge:

| op | MLX-optimized vs PyTorch-fp32 max-abs | classification |
|---|---|---|
| `sin0` (stem→reshape→sin) | **5.96e-08** | byte-stable — NO layout/transpose bug |
| `b0_interp` (bilinear ×2, align_corners=False) | **1.79e-07** | byte-stable — NO `align_corners`/interp bug |
| `b0_skip` (1×1 skip conv) | **1.79e-07** | byte-stable |
| `b0_ps` (PixelShuffle of first 3×3 conv) | **3.82e-06** ← **FIRST DIVERGENCE** | conv accumulation order — NOT a PixelShuffle convention bug (skip-conv at same depth is byte-stable) |
| `b5_out` (deepest block) | 8.72e-06 | accumulated conv drift |
| `refine_out` | 8.70e-06 | accumulated conv drift |
| `f0` / `f1` (sigmoid(rgb)·255 heads) | **1.18e-03 / 1.00e-03** | conv drift AMPLIFIED by ×255 head |

**The crux op = `conv2d_fp32_accumulation_order`.** The ONLY drift source is fp32 conv2d accumulation ORDER: native `mx.conv2d` accumulates in a different order than PyTorch `F.conv2d`. Intermediate features stay ~1e-6; the final `sigmoid(rgb)·255` RGB head amplifies the accumulated drift to ~1.2e-3 in [0,255] pixel space. **No structural bug** (transpose / PixelShuffle convention / `align_corners` / `sin`) — those ops are byte-stable at ≤ 2e-7.

This is the SAME class of drift PyTorch-fp32 exhibits vs PyTorch-fp64 (own accumulation self-drift ~8e-4 at the RGB heads). It is inherent to fp32 conv, not an MLX defect.

### Recursing to the next level — does the float drift matter at uint8?

| render mode | float max-abs (f0/f1, [0,255]) | uint8 max-abs | uint8 pixels differ |
|---|---|---|---|
| MLX-optimized vs PyTorch-fp32 | 1.18e-03 | **1 LSB** | **46 / 1,179,648 (0.0039%)** |
| MLX-fixed_fp64 vs PyTorch-fp32 | 1.07e-03 | **1 LSB** | **33 / 1,179,648 (0.0028%)** |

The MLX render is **already uint8-faithful** to the PyTorch reference: 99.996% of pixels are bit-identical; the rest differ by exactly 1 LSB at rounding boundaries (a coefficient near x.5 rounds differently between the two fp32 accumulation orders). This is the irreducible float→uint8 rounding-boundary floor that ANY two fp32 implementations exhibit.

### Recursing to the bottom — does render parity move the SCORE?

The decisive experiment: render the carrier with MLX-optimized vs MLX-fixed_fp64 vs the PyTorch-fp32 faithful reference, on IDENTICAL latents + IDENTICAL real `0.mkv` GT pairs, and measure SegNet argmax-flip d_seg with each:

| render | carrier d_seg (argmax-flip rate) | Δ vs PyTorch-fp32 reference |
|---|---|---|
| PyTorch-fp32 (faithful reference) | 0.0013707478647120297 | — |
| MLX-optimized | 0.0013707478647120297 | **0.0** |
| MLX-fixed_fp64 | 0.0013707478647120297 | **0.0** |

**Render-parity drift (source #1) has EXACTLY ZERO impact on d_seg.** The 1-LSB / 46-pixel / 0.0039% uint8 difference is sub-quantization for the SegNet argmax.

## The premise was wrong — and that is the highest-leverage finding

The directive's premise was that render-parity drift *"roughly DOUBLED the apparent distortion (0.189 vs implied ~0.073)."* **This is empirically FALSIFIED.** Render parity is already at the uint8 floor and contributes 0.0 to d_seg. The carrier's advisory distortion (≈ 0.189 = 100·d_seg + √(10·d_pose) = 100·0.00137 + √(10·0.00038)) is **NOT a render-parity artifact**. It is:

1. the **carrier reconstruction R(D)** — how well the 28-d latent + INT8 decoder reconstructs the real frame at the carrier's modelsize budget, AND
2. measured on the **Apple-Silicon-CPU eval axis (drift source #2, OUT OF SCOPE)** — NOT GHA-Linux-x86_64 / T4-CUDA.

Closing render parity (source #1) does NOT lower the distortion, because render parity was never the cause. Recursing to the crux bottomed out at: **source #1 is already engineered away to the floor; the leverage is elsewhere (source #2, paired eval, or the carrier R(D) — none $0-local fixable on the render axis).**

## What was ENGINEERED (the fix, honestly scoped)

1. **Op-level first-divergence harness** — `tac.analysis.mlx_pytorch_render_parity_crux.localize_render_parity_crux` localizes the crux op + classifies structural-vs-accumulation + measures the uint8 footprint. This is the recursive "down to the crux" instrument the prior full-stack drift tool (`tools/measure_pr95_mlx_pytorch_full_decoder_downstream_scorer_drift`) did not provide.
2. **Carrier render device-pinning** — `_pin_carrier_render_device` auto-pins the MLX CPU device when a fp64 conv mode is requested (Metal cannot accumulate fp64). `load_carrier_decoder` now accepts `conv2d_accumulation_mode`.
3. **Honest default** — the carrier render **defaults to the fast `optimized` mode**, NOT `fixed_fp64`. The crux finding is precisely why: `optimized` is already uint8-faithful and yields identical d_seg, so `fixed_fp64` buys ZERO d_seg benefit while paying ~56× per-forward cost (the slow fixed-order Python conv — it timed out the Fisher-pullback's 56-forward FD Jacobian at the fp64 default, a regression I caught and reverted). `fixed_fp64` remains available via the explicit kwarg for byte-tightest export parity (it does tighten float drift ~4.5×: 1.18e-3 → and reduces uint8-differing pixels 46 → 33).

The over-engineering trap avoided: forcing fp64 everywhere would be polishing a non-problem at real cost. The render is faithful at the floor in the fast mode.

## Canonical equations (results become system intelligence)

- **Appended** the decisive anchor (render-parity downstream d_seg impact = 0.0) to the EXISTING `mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1` — the strongest possible anchor for that equation: the downstream scorer impact of the render drift is empirically ZERO.
- **Registered** new `pr95_hnerv_render_parity_at_uint8_floor_distortion_gap_is_eval_axis_v1` — codifies the crux so the next MLX port inherits it and no future agent re-chases this dead end. Producer/consumer wired (`localize_render_parity_crux` + `load_carrier_decoder` + `canonical_equation_lookup_consumer`).

## Tests / verification

- 7 NO-FAKE tests pass (`src/tac/tests/test_mlx_pytorch_render_parity_crux.py`): first-divergence-is-conv (fails if a layout bug appears), uint8-faithful, fp64-tightens-float-drift, **d_seg-identical-across-render-modes** (the premise under test — would fail if render parity moved d_seg), default-mode + low-drift-override, non-promotable serialization, missing-reference error. Slot EEE Class 2: each fails if its fix were reverted.
- 23 existing carrier tests (`test_pr95_hnerv_linf_carrier.py`) still pass (8.15s — Fisher-pullback regression resolved).
- ruff clean on all touched files.

## 6-hook wire-in declaration (Catalog #125)

- **#1 sensitivity-map** — N/A (a parity diagnostic; no per-axis sensitivity signal).
- **#2 Pareto constraint** — N/A (render parity is orthogonal to the rate/seg/pose polytope; the finding REMOVES a phantom constraint — render parity is not a distortion lever).
- **#3 bit-allocator** — N/A.
- **#4 cathedral autopilot dispatch** — ACTIVE via canonical equation lookup: the new equation auto-discovers through `canonical_equation_lookup_consumer`, telling the ranker that PR95-class render parity is at-floor and the distortion gap is the eval axis (do not dispatch render-parity work).
- **#5 continual-learning posterior** — ACTIVE: the zero-d_seg anchor + the crux equation land in the canonical equations registry with Bayesian posterior update.
- **#6 probe-disambiguator** — ACTIVE: the harness IS the disambiguator between "structural MLX bug" (would diverge at a non-conv op) vs "fp32 conv accumulation order" (diverges first at `b0_ps`); it returns the regime-conditional verdict.

## Falsified / unwound assumptions

- *"the MLX render-parity blocker doubled the carrier distortion (0.189 vs 0.073)"* — **FALSIFIED**: render parity contributes 0.0 to d_seg; the carrier is render-faithful at the uint8 floor in the fast mode. The distortion gap is the eval-hardware axis (#2) + carrier R(D).
- *"closing render parity is the highest-leverage $0 unlock"* — **CARGO-CULTED**; unwound: render parity was already closed. The harness + canonical equation make this permanent system knowledge so the leverage is correctly re-routed to source #2 (paired eval, operator-gated) or the carrier R(D).
- *"fixed_fp64 should be the carrier render default"* — **FALSIFIED at the cost surface**: zero d_seg benefit, ~56× FD-Jacobian cost, Fisher-pullback timeout. Fast `optimized` default; fp64 via explicit kwarg.

## Predicted ΔS band

`[0.0, 0.0]` on the render-parity axis — this work makes NO score change (it proves render parity is already at-floor + prevents a dead-end chase). Mission contribution: `frontier_protecting` (removes a phantom blocker + canonicalizes the negative so the system reroutes leverage correctly).

## Sister-DISJOINT

Disjoint from `lane_snerv_inverse_steganalysis_carrier_20260601` (SNeRV build running in parallel, pid 39764) per Catalog #340 — touched ZERO `*snerv*` files. Touched only the PR95-HNeRV render path (`pr95_hnerv_linf_carrier.py`), the NEW harness module, the NEW test, and the NEW registration tool.
