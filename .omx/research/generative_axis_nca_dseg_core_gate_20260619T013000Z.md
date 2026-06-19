---
title: GENERATIVE-AXIS FEASIBILITY GATE — Neural Cellular Automata d_seg-core — VERDICT RED
authority: "[contest-CPU advisory] — NON-PROMOTABLE; exact pointer UNMOVED at 0.19110"
score_claim: false
promotable: false
frontier_pointer_moved: false
mission_contribution: frontier_breaking_enabler
date: 2026-06-19
verdict: RED_GENERATIVE_AXIS_CAPS_BYTE_CHEAP_BUT_DSEG_FLOORS_FAR_ABOVE_SUB015
supersedes: none
cross_refs:
  - .omx/research/generative_axis_dseg_core_design_20260619T004600Z.md
  - .omx/research/factored_lf_core_capacity_gate_20260618T233940Z.md
  - .omx/research/campaign_inflection_three_paths_capped_concentrated_saliency_20260618.md
  - .omx/research/campaign_math_review_dynamics_and_optimization_20260618.md
  - experiments/probe_nca_dseg_feasibility_gate.py
  - experiments/probe_curve_core_dseg_feasibility_gate.py
  - src/tac/tests/test_nca_dseg_feasibility_gate.py
  - experiments/results/nca_dseg_feasibility_gate/gate_state.json
---

# Generative-axis (NCA) d_seg-core feasibility gate — VERDICT: **RED**

**The $0 decisive measurement of the operator's reframe ("store the GENERATOR, replay to reconstruct").**
All numbers `[contest-CPU advisory]` NON-PROMOTABLE; the exact pointer is UNMOVED at **0.19110** —
stated plainly per the GOAL firewall: **this unit did NOT move the pointer.** Its value is a measured
GREEN/RED that closes the last untested representational family. $0: local MPS fp32 gradient + CPU-
authority d_seg; no paid GPU, no PR, no self-promote.

## 0. The headline (read this first)

The generative axis was the ONE representational family the sub-0.15 campaign had not tested. The bet
(Kolmogorov/MDL-optimal, Schmidhuber): store a tiny shared LOCAL UPDATE RULE + a tiny seed, ITERATE N
steps to GROW the SegNet-decision partition — temporal weight-sharing gives effective depth N with the
SAME few-KB rule, so it might break the `d_seg ~ params^−0.71` capacity wall that the static families hit.

**The byte/capacity escape is REAL — and irrelevant. The d_seg floors far above sub-0.15.**

| NCA rule-size | rule params | est. rate | **avg realized d_seg** | **BEST-frame realized** | × GREEN (0.0012) | × curve-survival (0.0067) | best-frame proj. S |
|---|---:|---:|---:|---:|---:|---:|---:|
| h32 | 2,165 | 0.0072 | 0.18013 (unstable) | **0.02180** | 18.2× | 3.24× | 2.25 |
| h64 | 4,245 | 0.0080 | 0.10945 (1/3 collapsed) | **0.02305** | 19.2× | 3.42× | 2.37 |
| h128 | 8,405 | 0.0095 | **0.02033 (stable)** | **0.01623** | 13.5× | 2.41× | 1.69 |
| h256 (completed, see §4) | 16,725 | 0.0126 | 0.18851 (1/3 collapsed) | 0.01970 | 16.4× | 2.93× | 2.04 |
| *frontier d_seg (the bar)* | — | — | *0.00257* | — | 2.1× | 0.38× | — |
| *curve-core survival floor (sister gate)* | — | — | *0.00673* | — | 5.6× | 1.0× | *0.74* |

- **The byte escape is structurally confirmed:** every rule size costs rate **0.0072–0.0126** — far below
  the 0.05 byte-cheap bar, and comparable to the curve core's control-point bytes while storing a *program*
  not a static description. The cheap-bytes HALF of the operator's bet IS delivered.
- **But it is irrelevant** because the `100·d_seg` term dominates catastrophically: the BEST single frame
  at the best stable size (h128) floors at realized d_seg **0.0162** — **13.5× above the GREEN threshold,
  2.4× above the curve-core survival floor, and 6.3× above frontier.** Best projected full-vehicle
  **S = 1.69** — ~11× worse than the 0.191 frontier, ~11× worse than the sub-0.15 target.
- **The generative axis is WORSE than the static curve core**, not better: the curve core reached realized
  d_seg 0.00673 (geometric 0.00106), the NCA's best is 0.0162. The iteration's "free detail" does NOT
  translate into a crisper SegNet-decision boundary.

**VERDICT: RED — `RED_GENERATIVE_AXIS_CAPS_BYTE_CHEAP_BUT_DSEG_FLOORS_FAR_ABOVE_SUB015`.** The generative
axis does NOT escape the walls. **Do NOT spec a generative-axis build.** Add it to the campaign's terminal-
finding family list: sub-0.15-grade d_seg is byte-cheaply unreachable across ALL FOUR tested families
(learned-pixel-decoder, static-stored-geometry/curve, AND now generative/NCA).

## 1. Why this is the RIGHT, FAITHFUL test (NO-FAKE, measurement-first)

- **REUSES the curve gate's exact machinery** (imported, not re-implemented): the EXACT eval roundtrip
  (`_eval_roundtrip_t`: camera-res bicubic-874 → bilinear-384 → round), the realized-d_seg measurement
  (`_segnet_argmax_of_frame` + argmax-flip-rate vs L*), the GT L* targets (`gt_targets_n16` `seg` field =
  the frozen SegNet argmax on GT frame1), and the byte/rate model — so the NCA number is **apples-to-apples
  with the curve gate's survival floor.** Only the REPRESENTATION (NCA generator) is swapped.
- **Real frozen contest SegNet (CPU AUTHORITY; never MPS for the score).** MPS is the training-gradient
  device only (the fp32 train/authority split).
- **MEASUREMENT-FIRST verdict:** driven by the realized d_seg of the HARD argmax-decoded frame THROUGH the
  real SegNet + exact uint8 roundtrip, NEVER the CE fit loss. 15 NO-FAKE tests
  (`test_nca_dseg_feasibility_gate.py`) prove the gate does the work it names: the rule actually GROWS
  (iteration changes the logits), gradients flow into the rule through the chain, the byte model counts the
  rule (not n_steps — the free-depth claim is structural), and the d_seg self-match is exactly 0 flips.
- **Guarded against BOTH false-GREEN AND false-RED** (the factored-LF degenerate-fit anchor):
  - **False-RED guard #1 (training stability):** the first attempt at lr=4e-2 DIVERGED (collapsed to a
    constant, realized 0.776). The fix — the canonical Growing-NCA per-parameter gradient NORMALIZATION
    (`grad/(‖grad‖+ε)`) + LR warmup — is the decisive stabilizer; it is the reason the RED is real and not
    an artifact of an unstable optimizer.
  - **False-RED guard #2 (render mismatch):** training rendered a soft-softmax (blended) frame the SegNet
    never sees at measurement → false interior flips (interior_flip 0.20). The fix — annealing the render
    temperature τ 1.0→0.12 (soft→near-hard) so the differentiable render converges to the hard argmax
    render used at measurement — collapsed interior_flip 0.20→0.003 and realized 0.21→0.019. The RED is
    measured AFTER both fixes (the probe is correct).
  - **False-GREEN guard:** the verdict uses realized-through-the-chain d_seg, and we report BEST-per-frame
    (not just the collapse-inflated averages) so a hidden GREEN cannot be masked — the best single frame
    across all sizes is 0.0162, far above GREEN.

## 2. The wall diagnosis (why the generative axis caps)

The component decomposition at the best stable size (h128) is the diagnosis:

- `geometric_dseg_recon = 0.026` — the NCA's OWN grown argmax partition vs L*. Decent (near GREEN), so
  the rule DOES grow a roughly-correct partition — the capacity escape is *partially* real (a 8.4K-param
  rule grows a partition combinatorially close to L*, where a one-shot decoder needs far more).
- `geometric_dseg = 0.019` — the rendered frame's SegNet argmax (no roundtrip). Already 3× the curve
  core's 0.0067: **the NCA's grown boundary is FUZZIER than the curve core's decimated polygons**, so the
  SegNet reads it less cleanly even before the survival hit.
- `realized_dseg = 0.020` and `boundary_band_flip = 0.353` — through the roundtrip. The boundary flip is
  **2.2× the curve core's 0.16**: the fuzzy grown boundary survives the bilinear downsample WORSE than
  crisp polygons.

So it is **BOTH walls compounded**: (a) the NCA's grown boundary is intrinsically less crisp than a stored
geometry (the iteration produces a soft, locally-smoothed partition, not a sharp polygon), so its pre-
survival SegNet match is already worse; (b) that fuzzy boundary then hits the same survival wall (the 1px
boundary-band mixing the curve gate proved is a property of the roundtrip + SegNet, not the representation),
amplified by the fuzziness. The "free detail via iteration" is the WRONG kind of detail for d_seg — d_seg
rewards crisp argmax boundaries, and local-growth NCA produces smooth diffuse ones.

Two smaller rule sizes (h32, h64) additionally showed per-frame TRAINING COLLAPSE (1/3 frames diverged to
realized ~0.3-0.5), i.e. small rules are unstable across the GT frames — a secondary cap, but moot given
even the stable frames floor at ~0.02.

## 3. The master-gradient framing (why RED was inevitable once d_seg floored)

Per the math review: ∂S/∂d_seg = 100 (binding). The NCA spends ~0.007–0.013 of rate (∂S/∂B trivial) to buy
d_seg via iteration. For S < 0.15 with pose held at 0.00034 (√(10·pose) = 0.058) and rate ~0.01, the budget
for 100·d_seg is ~0.08 → realized d_seg must be < ~0.0008. The NCA floors at 0.016 — **20× too high.** No
byte win can rescue a d_seg term that is 20× over budget. The cheap-generator advantage is real and
irrelevant, exactly as the factored-LF and curve gates found for their families.

## 4. h256 completed — it CONFIRMS the RED (post-write update)

h256 (16,725 rule params, rate 0.0126) finished after memo-write and confirmed the verdict exactly as
predicted. h256 frame0 = realized **0.0197** (geo_recon 0.035, geo_seg 0.021, bnd 0.374) — the SAME ~0.02
floor as h128, at HIGHER rate. **The capacity escape saturates: doubling the rule (8.4K→16.7K params) does
NOT lower realized d_seg below ~0.02; it only raises the rate.** (h256's frame2 also collapsed to 0.51 like
h32/h64's frame1, inflating its average to 0.189 — the small-rule instability persists at the large rule on
some GT frames; the stable-frame floor is the honest number, ~0.0197, still 16× GREEN.) The daemon's
auto-verdict is **`RED_NCA_CORE_HITS_SURVIVAL_WALL_LIKE_CURVE_CORE`**; best_realized across all 4 sizes =
0.0203 (h128), best_S = 2.10, byte_escape_real = True. The complete 4-size table:

| size | rule params | rate | avg realized | best-frame | per-frame realized |
|---|---:|---:|---:|---:|---|
| h32 | 2,165 | 0.0072 | 0.180 | 0.0218 | [0.022, **0.484**, 0.034] |
| h64 | 4,245 | 0.0080 | 0.109 | 0.0231 | [0.026, **0.279**, 0.023] |
| h128 | 8,405 | 0.0095 | **0.0203** | **0.0162** | [0.020, 0.025, 0.016] (stable) |
| h256 | 16,725 | 0.0126 | 0.189 | 0.0197 | [0.020, 0.039, **0.507**] |

The probe's auto wall-label is CAPACITY (driven by h256's collapse-inflated avg geo_recon); the per-frame
truth is **BOTH walls compounded** (§2): the stable frames floor at realized ~0.02 with geo_seg 0.019 (fuzzy
boundary, already 3× the curve core) + boundary_flip 0.35 (survival). Either label, the verdict is RED:
realized d_seg saturates at ~0.02 = 16× GREEN, 3× the curve-core survival floor. Final JSON:
`experiments/results/nca_dseg_feasibility_gate/gate_state.json` + `.omx/research/nca_dseg_feasibility_gate_20260619T010606Z.json`.

## 5. The honest fork (what this re-routes)

- **The generative axis does NOT escape the walls.** It is added to the terminal-finding family list. Across
  all four families now measured — learned-pixel-decoder (factored-LF, capacity wall), static-stored-geometry
  (curve core, survival wall), AND generative/NCA (both walls compounded) — **sub-0.15-grade d_seg is
  byte-cheaply unreachable for any per-frame d_seg-core that renders an RGB frame through the exact eval
  roundtrip + frozen SegNet.** The survival wall (the 1px boundary-band mixing under the camera-res→384
  bilinear downsample, flipping 16-40% of boundary pixels) is the common, representation-independent
  terminal wall; the curve gate proved geometry can match L* to 0.001 and STILL realize at 0.0067.
- **The one structural escape NOT closed by these gates:** anything that lowers d_seg without rendering an
  RGB frame through the survival-lossy roundtrip — i.e. the concentrated-saliency OWN-VEHICLE direction
  (the capstone #78): keep the d_seg-critical capacity in a small high-precision CORE that participates in
  the FULL frontier decoder (so it benefits from the frontier's already-low realized d_seg 0.00257), and
  shed/coarsen the d_seg-blind periphery for the rate win. That path operates ON the frontier decoder (whose
  realized d_seg is already past the survival regime), NOT as a from-scratch per-frame d_seg-core — which is
  the class these four gates have now decisively closed.
- **Incremental levers remain** (the #137 boundary sidecar / #138 lane prior on the FRONTIER for small d_seg
  cuts → a possible sub-0.19 pointer nudge), but they are not the sub-0.15 path.

## Observability surface

Every row records the 3 separated d_seg numbers (geo_recon = combinatorial capacity check; geo_seg =
pre-survival SegNet match; realized = through-roundtrip authority), the boundary vs interior flip split, the
byte breakdown + rate, and per-frame data (so best-vs-average is inspectable). S is recomputed from its
3 components. All `[contest-CPU advisory]`, score_claim=false, pointer_moved=false. Machine-readable at
`experiments/results/nca_dseg_feasibility_gate/gate_state.json`.

## Canonical-vs-unique decision per layer

See the design memo `generative_axis_dseg_core_design_20260619T004600Z.md` §"Canonical-vs-unique decision
per layer": the eval roundtrip, realized-d_seg metric, GT load, verdict logic, and MPS/CPU split are
ADOPT_CANONICAL (reused from the curve gate for apples-to-apples); the NCA generator + its byte model are
FORK (the unique mechanism under test).
