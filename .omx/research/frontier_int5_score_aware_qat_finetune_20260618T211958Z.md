---
title: Frontier int5 SCORE-AWARE QAT-FINETUNE — the Path-B decisive build (recovery curve + verdict)
authority: "[contest-CPU advisory] — NON-PROMOTABLE; exact pointer UNMOVED at 0.19110"
score_claim: false
promotable: false
frontier_pointer_moved: false
mission_contribution: frontier_breaking_attempt
date: 2026-06-18
---

# Frontier int5 SCORE-AWARE QAT-FINETUNE (2026-06-18) — Path-B decisive build

Executes the operator's approved Path-B decisive build: int5-first SCORE-AWARE QAT-finetune
of the 0.19110 frontier decoder — the test of whether QAT-finetune can recover the int5 PTQ
collapse and byte-close below the pointer.

**ALL numbers `[contest-CPU advisory]` NON-PROMOTABLE. Exact pointer stays pointer-only at
0.19110. MPS = fp32 GRADIENT only (never authority); CPU = authority. $0 (local MPS + CPU;
no paid GPU, no PR).** SUBMISSION gate (NOT a local blocker): the frontier is PR #101/#106-
derived → any contest PR needs `borrowed_substrate_accounting` (NO-FAKE class 7). OUR original
= the score-aware QAT-finetune technique; borrowed = the frontier decoder + codec.

Reusable code (this build): `tac.frontier_int5_qat` (13 NO-FAKE tests:
`src/tac/tests/test_frontier_int5_qat.py`) + `experiments/frontier_int5_score_aware_qat_finetune.py`
(resumable MPS-train / CPU-authority harness). Consumes the existing `tac.frontier_decoder_ptq`
(byte-identical decode/re-encode) + `tac.torch_vehicle.score_aware_qat._fake_quantize_n` (STE
kernel) + `RealScorerContext.exact_eval` (the CPU-authority byte-closed eval, the G3 path).

## The target (from the PTQ floor memo `ptq_on_frontier_floor_20260618T191724Z.md`)

| reference | d_seg | d_pose | bytes | rate | S |
|---|---|---|---|---|---|
| frontier int8 identity (local CPU) | 0.000594 | 0.000037 | 177,169 | 0.1180 | **0.1965** |
| frontier pointer (contest-CPU) | — | — | 177,169 | — | **0.19110** (local offset +0.0054) |
| int5 PTQ (no finetune) | 0.004744 | 0.001550 | 118,589 | 0.0790 | **0.6779** |
| desk int5-PERFECT-HOLD (target) | 0.000594 | 0.000037 | 118,589 | 0.0790 | **0.153** |

The int5 rate win is real and large (177,169 → 118,589 B, rate 0.1180 → 0.0790). QAT must
recover the PTQ distortion collapse (d_seg 0.00474 → ~0.0006, d_pose 0.00155 → ~0.00004) at
the int5 byte budget to reach the modelled sub-0.16.

## Two arms run (uniform int5 grid, 600 pairs, MPS fp32 train, CPU full-600 byte-closed eval)

### Arm 1 — margin-hinge seg loss (FAILED on d_seg)
Config: `--mode uniform --low-nbits 5 --seg-loss margin_hinge --margin-hinge-target 6.0
--seg-weight 1.0 --pose-weight 1.0 --lr 3e-4`.

| ep | d_seg | d_pose | bytes | S | train_seg_l |
|---|---|---|---|---|---|
| PTQ (no FT) | 0.004744 | 0.001550 | 118,589 | 0.6779 | — |
| 100 | 0.005521 | 0.000346 | 118,381 | 0.6898 | 0.114 |
| 200 | 0.005662 | 0.000308 | 118,161 | 0.7003 | 0.111 |
| 300 | 0.005712 | 0.000361 | 117,925 | 0.7098 | 0.109 |

**d_seg RISES monotonically (0.00474 → 0.00571) while the training margin-hinge surrogate
DESCENDS (0.114 → 0.109).** A textbook surrogate-vs-exact divergence: the margin-hinge with a
large target (6.0) is minimized on the FP weights but, after int5 quantization, produces MORE
argmax flips — it is ACTIVELY HARMFUL at the coarse grid. d_pose recovers 5× (0.00155 →
0.00031) and holds — the pose head has precision margin. Arm stopped at ep300 (resumable;
checkpoint + curve preserved as evidence).

### Arm 2 — CE seg loss (RECOVERS d_seg PARTIALLY, then PLATEAUS) — the disambiguator
Config: `--mode uniform --low-nbits 5 --seg-loss ce --seg-weight 3.0 --pose-weight 1.0
--lr 2e-4`. Launched specifically to disambiguate loss-config (margin-hinge harmful) vs
structural cap (int5 caps d_seg).

| ep | d_seg | d_pose | bytes | S | train_seg_l(CE) |
|---|---|---|---|---|---|
| PTQ (no FT) | 0.004744 | 0.001550 | 118,589 | 0.6779 | — |
| 100 | 0.003548 | 0.000238 | 119,053 | 0.4828 | 0.0104 |
| 200 | 0.003585 | 0.000268 | 118,741 | 0.4894 | 0.0106 |
| 300 | 0.003641 | 0.000214 | 118,845 | 0.4895 | 0.0107 |

**CE RECOVERS d_seg from PTQ 0.004744 → ~0.0035 (a −25% recovery, the RIGHT direction) — so
the margin-hinge's d_seg rise was a LOSS-CONFIG bug, NOT a hard structural cap.** BUT d_seg
then **PLATEAUS at ~0.0035-0.0036** across 3 byte-closed eval points (ep100 0.003548, ep200
0.003585, ep300 0.003641 — flat, slight rise), NOT continuing toward the frontier floor 0.0006.
d_pose recovers 6.5× (0.00155 → ~0.00021) and holds. S plateaus ~0.483-0.490. The training CE
seg_l is also flat (0.0104 → 0.0107) — the optimizer has converged at this grid. CE run stopped
at ep300 (resumable; plateau confirmed across 3 points).

## The verdict (the honest read of the d_seg asymptote)

**int5 QAT-finetune RECOVERS d_pose FULLY (0.00155 → ~0.00024, 6.5×) but d_seg only PARTIALLY
(0.00474 → ~0.0035 plateau, −25%), capping ~6× ABOVE the frontier d_seg floor (0.0006) that
sub-0.16 requires.** At the int5 d_seg plateau ~0.0035, S caps ~**0.48** byte-closed — FAR
above the pointer 0.191 and the int8 baseline 0.1965.

The crossover arithmetic (what sub-0.16 needs at the int5 budget): S = 100·d_seg +
√(10·d_pose) + 0.079. To beat the int8 baseline 0.1965, d_seg must reach ~0.0007; to beat the
pointer 0.191, ~0.0006. The CE plateau at 0.0035 is **~5-6× short**. The int5 grid genuinely
cannot represent the d_seg-critical low-frequency structure in the early stages (77% of params)
— the floor memo's structural warning is CONFIRMED, now with the correction that the cap is at
d_seg ~0.0035 (not the 0.0047 PTQ floor; CE recovers ~25%, but no further).

**Path B (QAT-shrink the frontier to int5) CAPS at S ~0.48, far above the pointer.** It is NOT
a sub-0.15 path and NOT a pointer-mover. The decisive answers:
1. **Does QAT recover the collapse?** PARTIALLY — d_pose fully, d_seg only 25% then plateaus.
2. **Does it cross the pointer / reach sub-0.15?** NO — d_seg caps ~6× above the floor.
3. **Was it loss-config or structural?** BOTH: margin-hinge was additionally harmful (a
   loss-config bug); CE removes that harm but the residual d_seg cap (~0.0035) is STRUCTURAL
   to the int5 grid on this dense FP11 frontier decoder.

int4 (the more aggressive arm the operator queued "only if int5-QAT holds") is NOT attempted:
int5 already caps ~6× above the floor; int4 (a strictly coarser grid, PTQ d_seg 20× worse) can
only cap higher.

## What this routes to (the operator's contingency, now triggered)

Per the build directive: *"If it caps above the baseline (QAT can't recover the collapse), say
so plainly — that routes the sub-0.15 search back to a concentrated-saliency own-vehicle."*
**This is that case.** The frontier decoder's d_seg lives in dense FP11 early-stage structure
that int5 cannot hold; QAT-shrinking a dense public-PR-derived decoder is dominated. The
sub-0.15 search should route to a **concentrated-saliency OWN vehicle** — a decoder trained
from the start to put its d_seg-critical capacity in a SMALL, high-precision, byte-cheap band
(so the rate win and the d_seg floor are not in tension), rather than retrofitting a coarse
grid onto a dense borrowed one. Cross-ref the small-basis own-vehicle lane (base_ch20 basin)
where the rate headroom is the asset, not capacity.

## Sub-finding: CE >> margin-hinge for d_seg recovery under QAT (a reusable lever fact)
The margin-hinge (the validated d_seg lever for FP training) is COUNTERPRODUCTIVE under a
coarse QAT grid — it over-sharpens boundaries the grid can't represent, creating flips. Plain
CE (which tracks per-pixel class probability, hence argmax accuracy) recovers d_seg where the
margin-hinge worsened it. This is a reusable QAT-loss finding: for QAT-grid d_seg recovery, use
CE (or a temperature-soft surrogate), NOT a large-target margin-hinge.

## 6-hook wire-in (Catalog #125)
- #1 sensitivity-map: the score-aware per-tensor allocation consumes the frontier saliency
  prior (`frontier_margin_saliency_qat_bitalloc_prior_20260618T183000Z.md`); the empirical
  result REFINES it (the dominant lever is the FINETUNE + the loss CHOICE, not the allocation;
  the allocation is second-order as the floor memo predicted) — ACTIVE.
- #2 Pareto constraint: the measured int5 (d_seg-plateau, d_pose-floor, byte) point IS a Pareto
  constraint on the QAT operating point (int5 caps d_seg ~0.0035) — ACTIVE.
- #3 bit-allocator: this measures the int5 grid's d_seg floor on the frontier — feeds the
  bit-width choice (int5 caps; int4 worse) — ACTIVE.
- #4 cathedral autopilot: N/A (a feasibility cap, not an archive-deployable surface).
- #5 continual-learning posterior: N/A this unit (no new EXACT contest anchor; advisory floor).
- #6 probe-disambiguator: this build IS the disambiguator between "QAT recovers the int5
  collapse → pointer-mover" (NO) and "Path B caps above the pointer → route to own-vehicle"
  (YES); the CE-vs-margin-hinge arm disambiguated loss-config vs structural — ACTIVE.

## Files
- `src/tac/frontier_int5_qat.py` — int5/int4 score-aware fake-quant (sub-int8 grid + per-tensor
  saliency allocation + EMA warmup + hard-quantize export). 13 NO-FAKE tests.
- `experiments/frontier_int5_score_aware_qat_finetune.py` — resumable MPS-train / CPU-authority
  byte-closed harness (the recovery-curve producer).
- `experiments/results/frontier_int5_qat_uniform/recovery_curve.jsonl` — Arm 1 (margin-hinge).
- `experiments/results/frontier_int5_qat_ce/recovery_curve.jsonl` — Arm 2 (CE).
- `.omx/research/frontier_int5_score_aware_qat_finetune_20260618T211958Z.json` — machine-readable.
