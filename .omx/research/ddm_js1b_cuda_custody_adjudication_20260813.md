# ddm_js1b T4 custody adjudication — the C1 bar was bound to the WRONG OBJECT; T4 fields are self-consistent

verdict_scope: instance — the js1b admission-gate binding on the ddm_js1b_20260813b run; the retained T4 fields are good custody, the ADMISSION ARITHMETIC needs the corrected bars below.

**Date:** 2026-08-13 · **Owner:** MAIN · **Axis:** [contest-CUDA T4 frozen-SegNet argmax fields, n600, batch=16] COMPONENT-ONLY
**Modal:** fc-01KZX0A3FJQ6AQRN0CE1XWXE9T · run_id `ddm_js1b_20260813b` · T4 · 1,456.4 s (K arithmetic held: vs 1,350.2 projected, 1,800 budget) · rc=0
**Receipt:** `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/ddm_js1b_20260813b/FINAL_RESULT.json`
(sha `5fd65b946e2e1a5683e123554761c4216f8245a4d1cec46da2ee95b925c93a0c`); fields retained on volume
`comma-ddm-js1b-argmax-retained/ddm_js1b_20260813b` (~22.8 GB, P0 KEEP-THE-PAYLOAD) + local download in flight.

## The worker's verdict vs the corrected reading

The worker reported `BLOCKED_AXIS_MISMATCH` on both controls. Cross-referencing js1's own STAGE0_RESULT.json
splits the two controls into different findings:

| Control | Worker observed (T4) | Bar it used | What the bar actually IS | Corrected reading |
|---|---|---|---|---|
| CP135 | 34,970 flips | 34,964 (promoted row) | same object, same axis | **6-flip residual = 0.017%** — batch-shape/tie-break scale (et4: batch shape is part of the forward instrument; upstream evaluate.py's batching ≠ the worker's batch-16) |
| C1 | 27,330 flips | 17,926 | **flips of the STORED `c1_target_argmax_n600.npy` TARGET FIELD** (d_seg 0.000151969), NOT the archive decode | **WRONG OBJECT.** The comparable number for the t1r1_c1_composed ARCHIVE decode is js1's local 47,950 (d_seg 0.000406) — a definition mismatch, not a decode failure |

## The load-bearing new measurement: local renderer drift, quantified on BOTH archives

| Archive | Mac-CPU decode flips (js1) | T4 CUDA decode flips (js1b) | Local inflation |
|---|---|---|---|
| cp135 composed | 50,395 | 34,970 | **+15,425 (+44%)** |
| t1r1_c1_composed | 47,950 | 27,330 | **+20,620 (+75%)** |

The local axis inflates flips massively and NON-uniformly across archives — js1's refusal to compute the
per-edge map locally was correct, and the m23/et4 instrument law is re-confirmed with magnitudes.
The T4 fields agree with the promoted CP135 row to 6 flips in 34,964. They are the honest custody.

## Adjudication + fire order

1. **CP135 control: ADMIT WITH DERIVED TOLERANCE.** The 6-flip residual is the batch-shape class
   (et4 measured 1-pixel flips per frame between batch shapes; 6/600 frames is that order). The per-edge
   decomposition consumes the T4 field itself as its own base — the 6-flip delta does not contaminate a
   same-field edge map. Requirement carried: the edge map + rho are computed FROM the js1b fields, and any
   V0–V5 ΔS claims are measured against the SAME-instrument base (the js1b CP135 field), never mixed bases.
2. **C1 control: REBIND.** The correct C1-archive reference at this instrument does not exist as a promoted
   integer; the js1b field IS the first CUDA-custody measurement of that decode. The stored target field
   (17,926) remains what it always was: js1's optimization TARGET, not a decode control.
3. **Next step (fires when the fields download completes):** run the pinned post-step
   `experiments/ddm_js1_stage0_per_edge.py summarize --from-argmax-fields <fields>` → matched n600 per-edge
   map + rho on the CUDA axis → V0–V5 ladder adjudication under #381.

## Attempt ledger (this chain)

- Attempt 1 (fc-01KZWWTSQ32T…): WorkerError — the arm INVENTED census pin `0.hevc` (real one-object set:
  `0.mkv`; the vd1 worker reads unpinned). Never-validated-literal class. One-line fix committed. ~$0.02.
- Attempt 2 (fc-01KZWYMZTPQK…): fail-closed resume-custody refusal (worker fix changed REQUEST vs the
  recorded run root) — correct guard behavior. ~$0.01.
- Attempt 3 (fc-01KZX0A3FJQ6…): COMPLETE, 1,456 s ≈ $0.16. Chain total this arc ≈ $0.55 of the #381 envelope.

## Pointer

Effective frontier UNMOVED: cp135 S 0.16195513827824176 @ 186,252 B [contest-CUDA T4 n600].
Own-vehicle: lc2 0.16959899569230852 @ 187,226 B. This unit bought the seg leg's INSTRUMENT custody
(CUDA argmax fields for both promoted archives), not a pointer move.
