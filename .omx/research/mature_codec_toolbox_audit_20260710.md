# Mature-Codec Toolbox Conformance Audit — 2026-07-10

**Operator question (verbatim):** *"Did we totally implement and are we fully exploiting the stuff you
said we didn't have that any mature codec has?"*

**Scope:** $0, report-only. Graded for OUR live vehicles: v7.5.2 (optimal single trunk, PR101-recode
rate term **0.118 S**, pointer **0.19110**) + v8 (edge-centric per-class geometric carriers, whole-scene
bitmap budget **0.346 S** → geometric projection **~0.02–0.05 S**; current v8 rate enemy **~0.074**).

**STORES CONSULTED (per §OPERATOR-PRIORITY proactive-recall):** CLAUDE.md L14–L32 canonical-leaderboard
lessons (the mature-codec entropy machinery PR95/100/101/103 actually shipped); DAG blocks
`FEED-v8-realmachinery / -ratebudget / -voronoi / -roadlane / -dsegtaper`; the June-2026
decoder-weight-rate-axis CONVERGENCE ("the $0/frozen-instance phase is EXHAUSTED; sub-0.15 is a TRAINING
problem"); `decoder_weight_rate_axis_…_synthesis_20260621`; `analytical_solve_extinctions/` (#342
SOLVE-DON'T-TRAIN); `src/tac/witness_dsl/{curriculum_dsl,campaign,lever_registry,activation_ledger}.py`;
`region_partition_codec` + `hessian_block_fp` + `scorer_exploits`.

---

## The honest one-line answer

**YES on concepts (every mature-codec element has an implemented, in-tree analog, and several are already
MEASURED through real byte-close in v8), but NO on full exploitation** — because the two facts that matter
are (1) the *generic* mature-codec entropy/RDO machinery is MEASURED-DEAD / EXHAUSTED for OUR rate object
(a perfect coder cannot cross the weight-entropy floor; the lever is the geometry-native REPRESENTATION,
not a better coder), and (2) the big *representation* levers that DO beat the bitmap (v8 geometric coders,
the #121 d_seg taper, the #207 postfilter, the #336 per-class allocator) are **BUILT-NOT-APPLIED** — built
and partially measured, but not yet wired into a single scored chain byte-close that moves the pointer.

---

## Graded table

| # | Mature-codec element | OUR analog | Grade | Provenance / pending task |
|---|---|---|---|---|
| 1 | Intra/inter PREDICTION + GOP/keyframe | per-pair latent (2 frames/latent, L19); frame-0→frame-1 ego-warp; 600 non-overlapping pairs = the GOP | **EXPLOITED** | L19; v8 horizon ξ warps 599/600 frames near-free (FEED-v8-realmachinery) |
| 2 | Motion model (our ego-screw ξ) | se(3) screw twist stored once, FiLM-conditioned; d_pose **0.001610** through byte-close (R1) | **EXPLOITED** (pose half banked-as-artifact) | R1 dxi byte-close; horizon cubic/quad FROZEN Δ≈1e-7. *verdict_scope:* ground-frame lane-ξ transport is a FORMULATION-level NO-GO (lanes dynamic, coeffs move 55–82%), NOT a ξ-family kill (FEED-v8-roadlane) |
| 3 | TRANSFORM stage (basis) | Fourier/curvelet/**directional** self-orient basis; curvelet = codim-1 optimal sparse basis (Candès-Donoho) | **BUILT — PARTIALLY APPLIED** | directional −48% d_seg MEASURED; `--self-orient` in campaign; freq-ALONG-tangent STARVED 3.2× (L25). `DsegAwareTaper` #121 + `DirectionalBasisRebalance` BUILT default-OFF |
| 4 | QUANTIZATION + per-unit RDO (λ mode decision) | KKT capacity waterfill #157 (campaign); WRQ score-aware reverse-waterfill; block-FP (`hessian_block_fp`) | **BUILT-NOT-APPLIED / MODEST** | WRQ = "MODEST + CONSTRAINED, sensitivity ~5.5× flat, 2.5-bit dyn range"; #157 pays ONLY after basis-match (dominated until basis directional). Per-block mode-decision RDO is not the witness's lever (trained generator, no per-block modes) |
| 5 | RATE CONTROL (bit-budget allocation) | whole-scene DE-SHARED per-class budget MEASURED; #336 per-class/margin-aware sensitivity | **BUILT-NOT-APPLIED** (#336) | FEED-v8-ratebudget: Road/Lane 0.204 · Road/Undriv 0.047 · … measured; #336 sensitivity allocator PENDING (not a live actuator in any chain) |
| 6 | Context-adaptive ENTROPY coding (AC/ANS) | `region_partition_codec` (JBIG/LOCO-I/CABAC + constriction range coder + brotli model); brotli-q11/zlib real coders in v8 byte-close | **EXPLOITED (real coder)** but "better coder" **PROBED-DEAD** | v8 uses real coders (brotli q11, bit-exact roundtrip). BUT DeepCABAC/order-2 arith = **EXHAUSTED**: brotli-q11 6.891 vs marginal H(W) 6.884 b/param (gap 0.007 ≤0.07 KB); temporal arith coder already realizes the piecewise-constant symbolic floor optimally (`…synthesis_20260621`). PR101-L21..L24 / PR103-L30 = MEASURED-DEAD on our object |
| 7 | In-loop filters / POSTFILTERS | A2 learned tiny-CNN score-aware postfilter (#207); PR95-L28 decode-side channel postprocess | **BUILT-NOT-APPLIED** (#207/A2) | A2 "contract EXISTS; re-open top-AIML vs exact scorer". L28 (0-byte channel subtract) is a known ~-0.0001..-0.0005 lever, not currently in witness chain |
| 8 | Perceptual/task-aware DISTORTION model | through-R frozen SegNet-argmax + PoseNet-MSE = the task distortion; Fisher metric = margin field (Pearson .978) | **EXPLOITED** (this IS the vehicle) | The whole capstone is task-space; strictly stronger than a perceptual proxy. UNIWARD steg-cost = same metric as cost |
| 9 | Encoder-side SEARCH (mode decision) | costate controller #247 SENSE + duty-to-measure activation-ledger; witness training = the encoder search | **EXPLOITED** | #247 auto-surfaced at SessionStart; activation-ledger tracks never-fired levers. Analog of encoder rate-distortion mode search |
| 10 | Error resilience | — | **N-A** | single deterministic video, no channel/packet-loss model |
| 11 | Tolerance / conformance budgets (D2) | deterministic-repro non-negotiable; byte-close; numpy-fp32 authority parity ≥0.9997; MLX-GPU bit-identity (#348) | **EXPLOITED** | D2 ledger + per-stage checkpoints + fused-R bit-exact (L70) |

**Grade counts:** EXPLOITED **6** (rows 1,2,8,9,11 + row 6 real-coder) · BUILT-NOT-APPLIED **4** (rows 3,5,7 + row 4 waterfill) · PROBED-DEAD **1** (row 6 "better entropy coder") · N-A **1** (row 10). **MISSING: 0** — nothing standard is un-built.

---

## Top-5 unexploited levers, ranked by expected rate/d_seg value

1. **Road/Lane geometric coder residual sidecar → v8 chain byte-close (#234 Wave-F).** Road/Lane = **59%**
   of the v8 bitmap budget (0.204 of 0.346). Already MEASURED **0.0275 S (7.4×)** lossless via real coder,
   but **27.5% coverage residual OWED** and NOT wired into a scored v8 chain. **Next:** build the residual
   sidecar (sparse hard-boundary px) + close the full v8 chain byte-close so the ~0.02–0.05 S projection
   becomes a real pointer-moving row. **EV:** the single largest rate lever vs the 0.074 enemy.
2. **Movable-class sparse object sites (OWED, named primitive exists).** 0.061 S combined bitmap
   (Undriv/Movable + Road/Movable) — the last 2 of 5 whole-scene edges not yet measured-geometric.
   Primitive = sparse object contours / Laguerre sites (`laguerre_logit_offset`, `power_diagram_argmax`
   in-tree). **Next:** measure geometric store for Movable pairs. **EV:** completes the whole-scene
   geometric budget (closes the "2 owed" gate on the rate thesis).
3. **#121 d_seg-aware Fourier taper — FIRE IT (BUILT default-OFF).** Duty-to-measure #1 lever (73% of
   remaining descent to sub-0.15), byte-neutral (rule-118 FREE), byte-identical when off (VERIFIED).
   **Next:** n600 A/B (taper on/off) through byte-close. **EV:** d_seg (not rate) — the dominant remaining
   S term after pose banked; ~0-byte so pure-win if it lowers d_seg.
4. **#207/A2 learned score-aware postfilter — top-AIML re-open.** Contract EXISTS, never applied to the
   witness chain; near-0-byte in-loop filter analog. **Next:** re-open vs the exact scorer at n600 (not a
   proxy). **EV:** d_seg + d_pose, near-free bytes; PROBED only at prototype grade → RE-OPEN per
   janky-prototype discipline.
5. **#336 per-class margin-aware rate-control allocator — make the measured budget a live actuator.** The
   de-shared per-class budget is MEASURED (FEED-v8-ratebudget) but sits as a table, not an allocator that
   assigns the geometric-vs-bitmap decision per edge in a chain. **Next:** wire the budget + margin
   sensitivity as the v8 chain's bit-allocation mode-decision. **EV:** rate — turns the measured 14.6×/7.4×
   per-edge wins into a governed whole-scene allocation.

**Not on the list (correctly EXHAUSTED — do NOT spend effort):** a "better entropy coder"
(DeepCABAC/order-2 arith/ANS) — brotli-q11 is already within 0.007 b/param of the weight-entropy floor and
the temporal arith coder already realizes the partition symbolic floor optimally. The lever is the
geometry-native BASIS, never the coder (FEED-v8-realmachinery meta-lesson: *even a real arithmetic coder is
a proxy if it ignores the representation's physics*).

---

## One-line for the #385 brief

Mature-codec toolbox: 0 MISSING, 6 EXPLOITED, 4 BUILT-NOT-APPLIED, 1 PROBED-DEAD (better entropy coder =
weight-entropy floor, do-not-pursue); the un-exploited value is all REPRESENTATION (v8 geometric coders
#234 residual+chain-close = 59% of the 0.074 rate enemy; #121 taper + #207 postfilter = the d_seg levers),
NOT coding machinery.
