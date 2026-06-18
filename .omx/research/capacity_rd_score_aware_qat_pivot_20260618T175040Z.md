---
title: Capacity-RD score-aware-QAT pivot — the $0 desk-calc decision gate + redirect
authority: "[contest-CPU advisory] — NON-PROMOTABLE; exact pointer UNMOVED at 0.19110"
score_claim: false
promotable: false
frontier_pointer_moved: false
mission_contribution: frontier_breaking_enabler
date: 2026-06-18
---

# Capacity-RD score-aware-QAT pivot — desk-calc gate (2026-06-18)

Executes the approved pivot (operator: **"higher capacity + score-aware FP-shrink QAT to cut
its rate"**) per the MVP-first sequence in the prompt: **$0 desk-calc FIRST to gate the
training**. Design basis: `campaign_math_review_dynamics_and_optimization_20260618.md`.

**ALL numbers `[advisory]` NON-PROMOTABLE. The exact pointer stays pointer-only at 0.19110.**
Reusable code: `tac.capacity_rd_qat` (15 tests) + `tools/capacity_rd_qat_desk_calc.py` +
`reports/capacity_rd_qat_desk_calc.json`.

## VERDICT: PROCEED — but the math REDIRECTS the target

The desk calc does **not** say STOP. It says PROCEED, and it **redirects** the target from a
*fresh* higher-capacity train to **QAT-shrinking the EXISTING 0.19110 frontier** — which is
the higher-EV reading of the same directive (the frontier IS the higher-capacity vehicle,
already trained; QAT-shrink cuts its rate).

## The two measured endpoints of the capacity↔d_seg↔byte Pareto (the inputs)
| vehicle | d_seg | d_pose | bytes | rate | S | provenance |
|---|---|---|---|---|---|---|
| base_ch20 small basis | 0.00260 | 0.000342 | 89,136 | 0.0594 | **0.3779** | `fp_shrink_ptq_bc20_n600.json` fp32 (600-pair exact) |
| frontier (pr110) | 0.00056 | ~2.9e-5 | 177,169 | 0.1180 | **0.19110** | `canonical_frontier_pointer.json` + symposium d_seg |

The frontier ALREADY bought near-frontier d_seg (0.00056) by spending capacity — its problem
is **purely rate** (62% of its score is bytes). bc20's problem is purely **d_seg** (65% of its
score). Neither alone is sub-0.15.

## Path A — fresh higher-capacity + QAT (the literal prompt reading): DOMINATED
S_QAT(p) using the QAT byte model (int4 on 70% d_seg-blind weights, +0.0003 d_seg hold):
| base_ch | dec_params | d_seg (src) | B_native | S_native | B_QAT | S_QAT |
|---|---|---|---|---|---|---|
| 20 | 83,356 | 0.00260 (MEAS) | 89,136 | 0.3779 | 59,354 | 0.3881 |
| 24 | 112,901 | 0.00164 (model) | 115,431 | 0.2993 | 76,863 | 0.3036 |
| 28 | 148,038 | 0.00109 (model) | 146,703 | 0.2648 | 97,686 | 0.2621 |
| 32 | 186,352 | 0.00077 (model) | 180,802 | 0.2554 | 120,392 | 0.2452 |
| 36 | 228,958 | 0.00056 (model) | 218,722 | 0.2601 | 145,642 | **0.2414** |

argmin S_QAT = bc36 at **0.2414**. It **beats bc20 native (0.378)** — the pivot's literal win —
but it **LOSES to the existing 0.19110 frontier** and is nowhere near sub-0.15. **Training a
fresh bc36 from scratch (days on MPS) is the WRONG bet: it cannot beat what we already have.**
(d_seg(p) above bc20 is a two-point power-law MODEL on the bc20+frontier endpoints — the single
honest extrapolation; flagged in-code.)

## Path B — QAT-shrink the EXISTING frontier (the redirect): SUB-0.15 IN MODEL
MEASURED 2026-06-18 by parsing the frontier FP11+CTXR grammar
(`pr110_payload_entropy_recode_20260610/submission_dir/archive.zip`): the **decoder section is
161,104 B = 90.9% of the 177,169 B archive** — the QAT-attackable share. Latents (8.5%), sidecar,
selector, DQS1 are kept verbatim (lossless, distortion-neutral). QAT-shrinking ONLY the decoder
section (bc20 measured int8→int-N transfer ratios applied to the decoder share):
| grid | frac_low | dec_B | arch_B | S(perfect hold) | S(+0.0003 spill) |
|---|---|---|---|---|---|
| int4 | 1.00 | 84,207 | 100,272 | **0.1399 ◀ sub-0.15** | 0.1699 |
| int4 | 0.70 | 107,276 | 123,341 | 0.1553 | 0.1853 |
| int5 | 1.00 | 103,880 | 119,945 | 0.1530 | 0.1830 |
| int5 | 0.70 | 121,047 | 137,112 | 0.1644 | 0.1944 |

**The decisive result: int4-uniform with a PERFECT distortion hold → S = 0.1399 (sub-0.15).**
Robust across the frontier d_seg uncertainty band (d_pose is backed out of the measured S, so
the distortion total is pinned: int4-perfect-hold stays 0.140–0.147 for d_seg ∈ [0.0004, 0.0008]).
Even the **+0.0003 d_seg spill** row (0.1699) **beats the 0.191 frontier** — so the path has
margin: it lowers the score even if QAT doesn't hold perfectly.

## The whole pivot reduces to ONE measurable question (the crux)
**Can QAT hold the frontier's distortion (d_seg + d_pose) through a decoder-section bit-shrink?**
- If hold is near-perfect → **sub-0.15** (0.140 at int4, 0.153 at int5).
- If it spills +0.0003 d_seg → **~0.17** (still beats 0.191).
- HONEST HEAVIER-BURDEN CAVEAT: the frontier decoder is **FP11** (~11-bit, denser than int8), so
  int4 is a BIGGER precision drop than bc20's int8→int4. The bc20 byte ratios transfer (the
  decoder dominates both archives), but the *hold* burden is heavier here — only a real QAT
  measurement closes it. PTQ will collapse (it did on bc20 int4: d_pose ×322); **that is exactly
  why the operator chose QAT** — re-train the net to tolerate the grid.

## Why this honors the design memo's math
The memo's stationarity `100·d_seg′(p) = −(25/B₀)·B′(p)` says QAT shifts B(p) down → the optimum
moves to higher capacity at the same bytes. Path B is the limiting case: take the vehicle that
ALREADY has frontier d_seg, and pull its B down via QAT. The 90.9% decoder share means the byte
lever has near-full reach. Score-aware allocation (protect the margin-saliency-critical weights at
int8, coarsen the d_seg-blind/stem-Nyquist weights to int4) is the `frac_low<1.0` rows — they
trade byte-win for hold-safety; the int4-uniform row is the aggressive bound.

## The next unit (precise, MVP-first, all pieces EXIST — search-and-familiarize done)
1. **Decisive cheap measurement FIRST (the QAT floor-to-beat): PTQ the frontier decoder.**
   Pipeline, all from existing pieces:
   - `feca_selector_reparameterize.split_fp11_member` → `pr110_payload_entropy_recode.reconstruct_raw_sections`
     → raw decoder streams (VERIFIED: 7 streams, 229,014 raw bytes).
   - vendored PR101 `codec.py:decode_decoder_compact` (DECODER_STREAM_ENDS reshape) → torch state_dict.
   - `tac.post_hoc_weight_shrink.requantize_decoder_state_dict` (int5/int4 qdq) → re-quant.
   - re-encode through PR101 codec → FP11 → `join_fp11_member` → CTXR pack → byte-close.
   - `tac.torch_vehicle.scorer_context.RealScorerContext.exact_eval` (600-pair, CPU authority).
   Adapt `experiments/probe_fp_shrink_ptq_bc20_n600.py` (it does exactly this for the bc20 codec;
   the frontier needs the FP11+PR101 codec swap). ~14 min/grid CPU. Tells us the d_seg/d_pose
   spill QAT must repair — the honest floor.
2. **Then score-aware QAT-finetune** on the frontier decoder using the EXISTING Lever-4 module
   `tac.torch_vehicle.score_aware_qat` (already built: per-tensor ∂S/∂w water-filling +
   boundary-protective STE; the MED-2 probe already measured −4.4% codec blob at equal d_seg on
   the bc20 basin). MPS fp32 gradient; CPU authority eval (async). Start int5 (gentler) before int4.
   `tac.margin_saliency_map.compute_margin_saliency_map` provides the protect-which-weights map.
3. **Byte-close + measure** S on CPU authority; compare to 0.19110 + sub-0.15. If it beats 0.19110
   byte-closed → flag for a paired dual CPU/CUDA exact eval (the pointer-mover).

**Do NOT** launch a fresh bc36 train (Path A is dominated). The frontier-QAT path reuses an
already-trained frontier and is the only modelled sub-0.15 vehicle.

## 6-hook wire-in (Catalog #125)
- #1 sensitivity-map: the score-aware QAT precision allocation consumes `margin_saliency_map` (ACTIVE next unit).
- #2 Pareto constraint: the capacity↔byte RD curve IS a Pareto constraint; this module quantifies it (ACTIVE).
- #3 bit-allocator: the QAT byte model is the bit-allocator's precision lever (ACTIVE — `score_aware_qat` Lever 4).
- #4 cathedral autopilot: N/A (desk model, not an archive-deployable surface yet).
- #5 continual-learning posterior: N/A this unit (no new exact anchor; advisory desk model).
- #6 probe-disambiguator: the desk calc IS the disambiguator between Path A (dominated) and Path B (sub-0.15).

## Files
- `src/tac/capacity_rd_qat.py` — reusable desk model (anchors + byte model + frontier section split + S_QAT).
- `tools/capacity_rd_qat_desk_calc.py` — CLI gate (prints both paths + verdict, writes JSON).
- `src/tac/tests/test_capacity_rd_qat.py` — 15 tests (arithmetic + measured-anchor consistency + gate).
- `reports/capacity_rd_qat_desk_calc.json` — the machine-readable table.
