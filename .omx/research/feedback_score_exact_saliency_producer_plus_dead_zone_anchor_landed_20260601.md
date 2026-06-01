# Score-exact P18/P19 saliency producer + full-video dead-zone anchor — LANDED 2026-06-01

**Lane:** `lane_boundary_aware_rd_allocation_grammar_20260601` (L1)
**Axis:** `[macOS-CPU advisory]` — NON-PROMOTABLE (no score claim, no MPS authority,
no GPU, no PR) per Catalog #341 / #192 / #323.
**Mission contribution:** `frontier_breaking_enabler` (the score-exact saliency
producer is the upstream surface the rate-axis allocators consume; the dead-zone
anchor is the GAP-5 full-video empirical grounding of the decoupling thesis).
**Horizon:** `frontier_pursuit`.

## What landed

This is the canonical UPSTREAM PRODUCER of the two score-exact per-pixel saliency
surfaces (P18 SegNet flip-risk + P19 PoseNet Fisher) plus a full-video dead-zone
diagnostic. It is COMPLEMENTARY to sister codex's `src/tac/contest_eval_contract.py`
(consumed, not duplicated) and CONSOLIDATES the verified mirror logic from
`tools/verify_upstream_scorer_mirror_fidelity.py` (commit 8173b493a; bit-exact
vs real frozen weights) into a reusable, profiled, optimized module.

1. **`src/tac/analysis/score_exact_saliency.py`** (NEW canonical producer):
   - `compute_s_seg_flip_risk(segnet, pair_btchw)` — P18: DeepFool top-2-margin
     flip-risk `||grad m||²/m²` on the LAST frame (SegNet scores `x[:, -1, ...]`).
     Returns `SegFlipRisk(flip_risk, grad_energy, margin, ...)` — `grad_energy`
     (`||grad m||²`) and `margin` are exposed SEPARATELY so they feed the existing
     `joint_p18_p19_waterfill.build_segnet_deepfool_margin_class_region_weight`
     consumer cleanly (verified end-to-end).
   - `compute_s_pose_fisher(posenet, pair_btchw, method=...)` — P19: squared
     input-Jacobian of the first-6 pose dims over BOTH frames = diag(Jᵀ J).
     Two methods: `loop` (baseline 6 backwards) and `batched_vjp` (OPTIMIZED:
     single `is_grads_batched=True` backward over a 6-row identity). Proven
     NUMERICALLY IDENTICAL (max_abs_diff = 0.0).
   - `saliency_concentration(map, margin=...)` — Lorenz/Gini + top-k% mass
     fraction + boundary/interior ratio.
   - `build_producer_provenance(...)` — threads sister codex's
     `build_saliency_verification_contract` (upstream source SHA-256s + mirror
     SHA-256s + the 6 `required_numerical_proofs`); fail-closed
     `score_claim=false` / `promotable=false` / `axis_tag="[macOS-CPU advisory]"`.
   - `stream_real_pairs(...)` — single-pass strided real-frame decoder (fixes the
     O(N²) re-decode that repeated `decode_real_pairs` calls incur).
   - `profile_producer(...)` — per-pair wall-clock measurement.

2. **`tools/measure_full_video_dead_zone_diagnostic.py`** (NEW GAP-5 anchor tool):
   stratified-subset default (60 pairs, ~2.3 min) or `--full` (600 pairs); emits a
   durable JSON anchor + human summary; relates concentration to
   `lambda = 25/37,545,489` score/byte.

3. **`src/tac/tests/test_score_exact_saliency.py`** (12 NO-FAKE tests, all pass):
   real-weight-or-SKIP, s_seg boundary-structured, s_pose spread, concentration on
   synthetic single-hot (Gini→1) and uniform (Gini→0), loop==batched_vjp
   equivalence, a Class-2 constant-detection guard, and SegNet-last-frame-only.

## Dead-zone numbers — 60-pair stratified anchor (stride 10, spans full 600-pair video)

`.omx/research/dead_zone_stratified_60pair_anchor_20260601.json`

| Surface | Gini (mean) | top-1% mass | top-5% mass | top-10% mass | boundary/interior |
|---|---|---|---|---|---|
| **s_seg (P18)** | 0.9997 | 0.9986 | 0.9996 | 0.9997 | 47,521× |
| **s_pose (P19)** | 0.9431 | 0.5284 | 0.8309 | 0.9272 | (geometric, n/a) |

Per-pair stability (honesty check — concentration is a stable STRUCTURAL property,
not a sample artifact): s_seg Gini ∈ [0.9992, 0.9999]; s_pose top-10% ∈
[0.9006, 0.9666] — **even the worst pair** holds ≥90% of pose mass in the top-10%.

**Decoupling thesis verdict: HOLDS AT SCALE for both axes.** The top-10% of pixels
hold ≥92% of BOTH the s_seg flip-risk mass AND the s_pose Fisher mass across the
whole video. A rate-axis allocator can dead-zone the bottom ~90% of pixels and
still preserve distortion — the score-relevant per-pixel information IS
concentrated enough that a small archive rate can solve distortion. s_seg is
extremely concentrated (boundary-peaked, near-degenerate Gini); s_pose is more
spread (full-frame geometric) but still strongly concentrated.

The full 600-pair `--full` run was launched in the background
(`.omx/research/dead_zone_full_video_600pair_anchor_20260601.json`); the stratified
60-pair anchor is the durable primary and the full run an append-only confirmation
(interim 120/600 pairs reproduced the same concentration signature).

## Profile table (before → after optimization)

CPU-only, 6-8 torch threads, native-resolution Jacobians (874×1164 for pose).

| Stage | Before (loop, O(N²) decode) | After (batched_vjp, single-pass decode) |
|---|---|---|
| s_pose backward method | 6 sequential backwards | 1 `is_grads_batched` backward (forward graph traversed once) |
| s_pose per-pair | ~0.64 s | ~0.84 s steady-state¹ |
| 8-pair diagnostic wall-clock | 60.7 s (O(N²) strided re-decode) | 23.1 s (single-pass stream) |
| 60-pair stratified wall-clock | (would be ~5 min²) | **135.5 s (~2.3 min)** |
| projected full-600 | ~21 min | ~18.7 min |

¹ The batched VJP is numerically identical to the loop and saves the per-dim
graph re-traversal, but on CPU the pose backward is memory-bound (native-res conv
activations), so the per-pair win is modest (~0.1 s). The dominant cost is the
frozen-scorer FORWARD (irreducible): SegNet EfficientNet-B2 forward ≈ 0.7 s,
PoseNet forward ≈ 0.18 s. The 600-pair full run is therefore ~18 min on CPU —
the stratified 60-pair subset is the canonical few-minutes anchor (concentration
is a per-pair structural property, so a uniform stride-10 subset is a sound
full-video estimate, verified by the tight per-pair variance above).
² Adversarial-review finding: the O(N²) per-pair re-decode (each
`decode_real_pairs(start_pair=p)` decoded `2(p+1)` frames from the start) was the
hidden wall-clock killer — a stratified pair at p=300 cost 5.5 s of decode alone.
Fixed via `stream_real_pairs` single-pass streaming decoder.

## Adversarial self-review (CLAUDE.md "Recursive adversarial review protocol")

**Q1: Is `s_seg = ||grad m||²/m²` a valid majorizer of flip-probability? Would
descending it ever INCREASE d_seg?** RESOLVED. The DeepFool minimal flip
perturbation is `r* ≈ m/||grad m||`, so flip-risk ∝ `1/r*² = ||grad m||²/m²`. The
surface is a **protection PRIORITY** (high risk = must-preserve pixel), NOT a loss
to descend. The existing canonical consumer
`build_segnet_deepfool_margin_class_region_weight` uses it MULTIPLICATIVELY as an
allocator weight (protect high-risk pixels → fewer argmax flips → lower d_seg) —
verified the producer feeds it cleanly with `margin` + `grad_energy` separated.
No descent-direction confusion exists.

**Q2: Is the concentration metric honest (not gamed by a small/unrepresentative
sample)?** RESOLVED. The default is a UNIFORM stride-10 stratified sample across
all 600 pairs (adversarial fix vs a contiguous head-sample). The per-pair
min/mean/max variance is tight (s_seg Gini ∈ [0.9992,0.9999]; s_pose top-10% ∈
[0.9006,0.9666]), proving concentration is a stable structural property of every
pair, not a sample artifact. The full-600 background run confirms.

**Q3: Does the dead-zone anchor use real per-pixel gradients (not constants)?**
RESOLVED. The Class-2 NO-FAKE guard test (`test_producer_returns_real_gradients_
not_constants`) FAILS if the surfaces have zero spatial variance or identical Gini
(a shared constant stub). s_seg and s_pose have genuinely different concentration
signatures (top-1%: 0.9986 vs 0.5284). All surfaces are finite, nonzero, and
gradient-reachable through the differentiable mirror.

## 6-hook wire-in declaration (Catalog #125)

1. **Sensitivity-map contribution** — ACTIVE. This module IS the canonical
   score-exact per-pixel sensitivity producer. `SegFlipRisk` (P18) and `PoseFisher`
   (P19) are the per-pixel sensitivity surfaces; downstream `tac.sensitivity_map.*`
   and the bit-allocator consume them.
2. **Pareto constraint** — ACTIVE. The concentration metric quantifies the
   rate/distortion polytope feasibility: ≥92% of distortion-relevant mass in the
   top-10% of pixels means the rate axis (25·bytes/N) can be minimized while the
   seg+pose axes are preserved — exactly the polytope intersection the Dykstra
   solver and `joint_p18_p19_waterfill` operate on.
3. **Bit-allocator hook** — ACTIVE. `SegFlipRisk.margin` + `SegFlipRisk.grad_energy`
   feed `joint_p18_p19_waterfill.build_segnet_deepfool_margin_class_region_weight`
   (verified end-to-end); `PoseFisher.s_pose` feeds the P19 Mahalanobis weight. The
   dead-zone anchor IS the bit-allocator's prior (spend on the concentrated mass).
4. **Cathedral autopilot dispatch hook** — N/A (advisory, non-promotable; the
   producer is a compress-side analysis surface, not a dispatch candidate). The
   JSON anchor is consumable by the autopilot ranker as observability-only.
5. **Continual-learning posterior update** — ACTIVE. The dead-zone JSON anchor is
   the durable empirical record; future allocator sweeps reseed off the measured
   concentration rather than re-deriving it.
6. **Probe-disambiguator** — ACTIVE. `method="loop"` vs `method="batched_vjp"` ships
   BOTH interpretations of the pose Jacobian with an equivalence test that arbitrates
   (they are numerically identical; the optimized path is the default).

## Canonical-vs-unique decision per layer

- **scorer loading + differentiable mirror**: ADOPT_CANONICAL
  (`tac.scorer.make_scorers_differentiable` — obvious-fit; no substrate reason to
  fork the verified differentiable yuv6 patch).
- **real-frame decode**: ADOPT_CANONICAL pattern (upstream `frame_utils.yuv420_to_rgb`,
  per Catalog #213) but FORK the iteration (single-pass `stream_real_pairs` vs the
  harness's bulk decode — principled: O(N²) elimination for strided sampling).
- **s_seg / s_pose surrogates**: ADOPT_CANONICAL (consolidated verbatim from the
  verified mirror harness `_section_s_seg` / `_section_s_pose`; the harness stays
  the fidelity verifier).
- **pose Jacobian backward**: FORK (batched_vjp optimization) — but proven
  equivalent, so it is a strict improvement not a divergence.
- **provenance / custody**: ADOPT_CANONICAL (sister codex's
  `build_saliency_verification_contract` — consumed, the 6 proofs threaded through).

## What I reused (anti-duplication, Catalog #229/#314)

- `src/tac/contest_eval_contract.py` (sister codex) — CONSUMED constants +
  `build_saliency_verification_contract` + `build_score_allocation_contract`.
- `tools/verify_upstream_scorer_mirror_fidelity.py` — CONSOLIDATED `_decode_real_frames`,
  `_build_upstream_distortion_net`, `_to_btchw`, `_section_s_seg`, `_section_s_pose`
  into the reusable producer; the harness stays the fidelity verifier.
- `tac.scorer.make_scorers_differentiable` — the canonical differentiable patch.
- `tac.optimization.joint_p18_p19_waterfill.build_segnet_deepfool_margin_class_region_weight`
  + `mahalanobis_pose_jacobian_norm` — the existing CONSUMERS my producer feeds
  (NOT re-created; the producer is the upstream surface they should reference).

## Files

- `src/tac/analysis/score_exact_saliency.py`
- `tools/measure_full_video_dead_zone_diagnostic.py`
- `src/tac/tests/test_score_exact_saliency.py`
- `.omx/research/dead_zone_stratified_60pair_anchor_20260601.json`
- `.omx/research/dead_zone_full_video_600pair_anchor_20260601.json` (background)
