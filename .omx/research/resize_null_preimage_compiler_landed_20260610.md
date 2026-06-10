# RESIZE-NULL PREIMAGE COMPILER — landed 2026-06-10

**Subagent:** `resize_null_preimage_compiler_20260610` (task #49).
**Evidence grade:** the exactness proof is `mathematical-derivation`
(hardware-independent, residual == 0.0); the bytes-reduction numbers are
`[macOS-CPU advisory]` (local CPU coder on real frames). Mechanism + measured
postprocessor; **no score claim; no dispatch; `promotable=false`.** $0 local, NO
cloud, NO paid GPU, NO MPS.
**Frontier at landing:** contest-CPU **0.19198533** (archive `b7106c9b…`).
This compiler does not by itself change the frontier; it is a UNIVERSAL
POSTPROCESSOR that removes scorer-invisible bytes from any vehicle's frames
BEFORE any codec touches them — a strictly-negative-ΔS rate lever at certified
zero distortion.

## The theorem this operationalizes (S12 / #47)

Both scorer heads' FIRST op is the same fixed bilinear resize
`R: (874,1164) -> (384,512)` (`F.interpolate(..., mode='bilinear',
align_corners=False)`, NO `align_corners` arg => default False — matches #47's
derivation exactly; `upstream/modules.py:73` PoseNet, `:109` SegNet).
`evaluate.py` therefore scores the PROJECTION `y = R x`, NEVER the camera frame
`x`. **Every vehicle should emit the minimum-description preimage**
`x̃* = argmin bytes(x̃) s.t. R x̃ = R x, 0 ≤ x̃ ≤ 255 uint8` — the cheapest legal
high-res representative of the scorer-equivalence class. 22.7% of every camera
channel is certified zero-weight (single-pixel exactly invisible); the full
resize nullity is 80.67%.

## What landed (three tiers, each independently shippable + tested)

`src/tac/optimization/resize_null_preimage.py` (REUSES #47's
`derive_tier1_resize_null_space` + `_resize_1d_matrix` — does NOT rederive R):

- **`ResizeProjector`** — the exact separable projector `R = R_h (x) R_w` built
  from #47's certified 1D matrices; `max_abs_projection_residual(x, x̃)` is the
  per-frame exactness proof `max|R x̃ - R x|`. Matches `F.interpolate` to <1e-9
  (test).
- **Tier 1 — `apply_tier1_zero_weight_fill`** — replace the certified
  zero-weight pixels with the entropy-optimal fill, chosen BY MEASUREMENT
  (`measured_best` ranks constant / horizontal-predictor / vertical-predictor /
  neighbor-mean on the real coder; q=5 search ranking, q=11 reported). Residual
  == 0.0 certified.
- **Tier 2 — `apply_tier2_null_basis_descent`** — block-coordinate integer
  null-basis descent on coded size over the zero-weight pixels (the ONLY
  integer-exact null directions; see derivation below), per channel; confirmed
  against tier-1 at q=11 so it is a TRUE descent (never worse). Residual == 0.0.
- **Tier 3 — `apply_tier3_blockwise_flat_preimage`** — smallest viable blockwise
  constrained least-entropy variant: snap the zero-weight lattice to the single
  plane-wide constant that minimizes coded size. Residual == 0.0.
- **`preimage_rate_score_delta`** — consumes #46's `delta_rate_score` (THE LAW
  rate term); distortion delta is CERTIFIED 0.0 so the full ΔS_total is exactly
  the (negative) rate term for any positive bytes freed.

CLI `tools/resize_null_preimage_postprocess.py` (frames-in -> preimage frames +
proof JSON + V3-row JSONL + headline mode). Loaders: `video:` (PyAV, camera_size
NHWC = the `evaluate.py` layout), `raw:` (the inflate `.raw` memmap layout), `npy:`.

## Derivation: why the integer-exact null atoms ARE exactly the zero-weight pixels

Operator caveat (a): the null space is real-valued but archive frames are
integers, so tier-2/3 use INTEGER-friendly bases only. Searching the separable
kernel for two NON-zero-weight input indices with identical resize-weight columns
(which would permit an integer +1/-1 null transfer between them) finds NONE: the
only duplicate weight columns are the all-zeros (zero-weight) columns (140 W cols
/ 106 H rows; verified). So a unit moved between any two non-zero-weight pixels
changes `R`. The certified integer null directions are therefore exactly the
zero-weight pixels (each amplitude-unlimited up to clipping), and tier-2 descends
the coded size over THOSE. **No real-valued null vector is ever rounded into the
uint8 frame** — every emitted frame is integer + provably preimage-exact.

## The falsifiable headline measurement ($0, N=16 real frames each)

`/Volumes/VertigoDataTier/pact/resize_null_preimage_20260610T020606Z/headline.json`
— tier-1, residual == 0.0 EXACTLY on every frame, all valid uint8:

| Frame set | brotli before -> after | brotli freed | lzma freed |
|---|---|---:|---:|
| **source video** (`upstream/videos/0.mkv`) | 16,413,785 -> 14,724,102 | **1,689,683 B (10.29%)** | 1,502,385 B (9.47%) |
| **SNeRV G1b render** (`frontier_inflate/0.raw`) | 42,236,163 -> 33,987,058 | **8,249,105 B (19.53%)** | 8,297,833 B (**19.60%**) |

**Prediction CONFIRMED, disconfirmer NOT triggered.** Tier-1 alone shrinks coded
frame bytes measurably — and on the SNeRV render the gain is ~2× the source
(19.5% vs 10.3%): the SNeRV decoder emits structured/low-entropy output in the
zero-weight region that the predictor fill flattens to maximally-compressible
filler. The savings are NOT absorbed by the coder's existing modeling. Per frame
that is ~515 KB (vehicle) / ~106 KB (source) of brotli bytes freed at PROVEN zero
scorer change.

(Tier-2 4+4-frame check — `resize_null_headline_t2.log` — confirms tier-2 ≥
tier-1 by the monotonic-descent guarantee; the marginal gain over tier-1 is
small because tier-1's measured-best already captures the bulk of the
zero-weight-region compressibility. Tier-1 is the headline; tier-2/3 are the
descent refinements.)

## The honest scope (what this is and is NOT)

- It is a **per-frame rate lever at certified zero distortion** — the strongest
  possible free-bytes branch under THE LAW (ΔS = negative rate term, Δd_seg =
  Δd_pose = 0.0 CERTIFIED, not estimated).
- The headline numbers are RAW per-frame coded bytes, NOT archive bytes. A real
  archive coes not store frames raw; the operative question for any vehicle is
  whether the preimage reduces ITS archive payload. For a vehicle that stores
  rendered frames (or frame residuals) the reduction transfers; for a vehicle
  that stores latents/weights (SNeRV's archive is decoder state, not frames) the
  preimage applies to the DECODE-side frames the codec models, OR to any
  frame-pixel atom (PR110++ modified frames). The right production consumer is
  **the frame-pixel atom path, not the SNeRV weight archive.**
- It does NOT lower SNeRV's d_seg (0.2468 nonrate gap) — that needs Class-3
  repair atoms, not a rate lever. Per the nonrate reality check, this is a rate
  tool.

## Recommended first production consumer

**PR110++ frame-pixel atoms (`pr110pp_frame1_joint_methodology` Class 5).** Those
atoms ARE camera-resolution frame perturbations whose archive cost is the coded
frame/residual bytes; running them through the tier-1 preimage before coding
removes the 80.67% of camera DOF that never reach the evaluator, so each atom's
bytes are spent only on scorer-visible structure. The methodology memo already
names this as Class 5 ("actions designed in raw pixel space waste most of their
bytes by construction"); this compiler is its executable form. Second consumer:
the #46 waterfiller's `null_basis` recode action (it already accounts certified
free bytes; this supplies the actual byte-reducing rewrite).

## Tests (24 dedicated; NO FAKE — behavior not constants)

`src/tac/tests/test_resize_null_preimage.py`:
- exactness (residual == 0.0 against the real projector, every tier + every
  explicit strategy);
- **the discriminator**: an OUT-of-mask perturbation DOES change the projection
  (a fake "everything invisible" basis is caught), an IN-mask +255 does not;
- out-of-mask pixels byte-identical (the fill only touches certified-invisible
  DOF); valid uint8 + same shape;
- **the no-fake heart**: the certified fill leaves BOTH the REAL upstream SegNet
  AND PoseNet preprocessed inputs bit-identical (RGB-before-YUV equality ⇒ YUV6
  equality, operator caveat (b)) — `test_preimage_survives_full_upstream_preprocess`;
- projector reproduces `torch.nn.functional.interpolate` to <1e-9;
- idempotence; tier-2 ≥ tier-1 (true descent); real-coder measurement;
  fail-closed input validation; THE LAW rate-delta sign.

## 6-hook wire-in (Catalog #125)

1. **sensitivity-map** — ACTIVE. The zero-weight pixel mask is a per-pixel
   certified-invisibility map; the preimage acts on exactly those DOF.
2. **Pareto constraint** — ACTIVE (advisory). Certified zero-distortion rate
   reduction is on the feasible boundary (no d_seg/d_pose cost).
3. **bit-allocator hook** — ACTIVE (PRIMARY). `preimage_rate_score_delta` is a
   certified-zero-distortion rate allocator; the preimage is the byte-reducing
   rewrite the #46 waterfiller's `null_basis` action accounts for.
4. **cathedral autopilot dispatch** — N/A. A postprocessor consumed by vehicles
   that emit archive bytes; it emits none itself.
5. **continual-learning posterior** — N/A. Tier-1 exactness is a closed-form
   derivation; the bytes numbers are `[macOS-CPU advisory]`, recomputed per
   vehicle render.
6. **probe-disambiguator** — ACTIVE. The exactness proof (residual == 0.0 vs >0)
   IS the disambiguator between a certified preimage and an approximate one; the
   CLI fails closed (keeps the original frame) on any non-exact frame.

## Per-layer canonical-vs-unique decision (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| resize derivation | ADOPT_CANONICAL (#47 kernels) | reuse `_resize_1d_matrix` + `derive_tier1_resize_null_space`; do NOT rederive R |
| projector `R_h (x) R_w` | FORK_PRINCIPLED | the separable apply + per-frame residual proof is genuinely new (the compiler surface) |
| coded-size arbiter | FORK_PRINCIPLED | measurement-driven (real brotli/lzma), per the directive "measure not convention" |
| rate-score delta | ADOPT_CANONICAL (#46 `delta_rate_score`) | consume THE LAW rate helper, do not refork |
| MeasurementScope / V3 schema | ADOPT_CANONICAL | the methodology-memo Class-5 row schema (extended with `preimage_proof`) |

## Named follow-ups (Tier 3 upgrades)

- per-block palette / RLE preimage over the zero-weight lattice (cheaper than a
  single plane constant when the visible projection allows piecewise structure);
- joint visible-constrained least-entropy over the FULL `ker(R)` (80.67%, not just
  the 22.7% axis-aligned subset) via integer-relaxed LP/MILP with a uint8 round +
  re-prove loop;
- learned context fill (a tiny decode-cheap predictor) for the dropped lattice;
- temporal preimage coherence (share the dropped-lattice fill across pairs so the
  selector/residual stream is delta-coded — exploits the atlas's temporal
  clustering).

## Reproduce

```bash
# Headline (tier-1, 16 source + 16 SNeRV G1b frames):
PYTHONPATH=src:upstream .venv/bin/python tools/resize_null_preimage_postprocess.py \
    --headline --n-frames 16 --tier 1 \
    --source-video upstream/videos/0.mkv \
    --vehicle-raw /Volumes/VertigoDataTier/pact/snerv_branch_b_round2_*/frontier_inflate/0.raw
# Tests (24 dedicated; the no-fake upstream-preprocess test is the heart):
PYTHONPATH=src:upstream .venv/bin/python -m pytest \
    src/tac/tests/test_resize_null_preimage.py -q   # 24 passed
```

## Commits

- `829a973c6` — tier-1: zero-weight fill + exact projector (REUSE #47 kernels).
- `94dc9af15` — tier-2+3: per-channel null-basis descent (monotone) + blockwise
  flat + THE LAW rate delta.
- (this batch) — CLI + 24 tests + this memo.

Lane: `lane_resize_null_preimage_compiler_20260609` (L1: impl_complete +
three_clean_review + memory_entry). `research_only=true` postprocessor surface
(no archive bytes emitted by this lane; it informs lanes that do).
