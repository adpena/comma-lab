# SG-DRF $0 single-frame FEASIBILITY PROBE — RESULT (DAG FEED-bg gate)

**Date:** 2026-06-26 · **Authority:** `[contest-CPU advisory]` (single real seg-frame, EXACT frozen
CPU-torch SegNet argmax through the CORRECTED contest R; NOT the 600-sample harness) ·
`promotion_eligible=false` · `score_claim=false` · MLX = fp32 training-gradient ONLY · MPS NEVER used.
**Verdict: FAIL the build-gate — do NOT escalate SG-DRF to a backbone now.** The flow COLLAPSED;
the matched-param coord-INR control trained cleanly. Classified IMPLEMENTATION-collapse (paradigm
intact per Catalog #307), not a clean iterative-depth disproof.

Probe: `experiments/sg_drf_single_frame_feasibility_probe.py` (reviewed; serializer-committed).
Bulk evidence (durable, NOT /tmp): `experiments/results/sg_drf_probe_20260626T155635Z/`
(`summary.json` + `argmax_maps.npz` — gitignored build-artifact tier; full summary embedded below).

## R-FIDELITY (the check)
The probe's d_seg VERDICT reuses the through-R trainer's CONTEST-EXACT authority path verbatim:
`_torch_R_to_camera_uint8` (render → bicubic-up to camera 874×1164 → uint8) → real
`SegNet.preprocess_input` (bilinear-down to 384×512, `modules.py:108-113`) → argmax-disagreement
(`modules.py:112`) == `upstream/evaluate.py` d_seg. **Self-check: pushing the GT frame1 straight
through this path gives `d_seg_GT_vs_Lstar = 0.0`** (a witness == GT scores exactly 0) ⇒ the
authority R reproduces evaluate.py. MLX training gradient uses the matching
`apply_contest_faithful_roundtrip_nhwc` (DAG FEED-bf corrected order: uint8 @ CAMERA, then bilinear
to scorer, no trailing uint8). Both arms use the IDENTICAL R, loss (margin-weighted softargmax-CE),
render res (256×384), optimizer (AdamW lr 1e-3, grad-clip 1.0, EMA 0.997), 500 epochs. Only the
backbone differs.

## ARM A (SG-DRF: conv velocity field v_θ(x,t|z) + d_z=16, N=4 deterministic Euler ODE, reflow 1×)
- params **59,683** (C=24; ≤ ARM B — NO param advantage), N=4 steps, 455 s.
- realized d_seg through R = **0.506922 = the init value — NEVER MOVED**. loss froze at exactly
  **2.9668 from ep~100** onward; post-reflow unchanged.
- realized argmax = a **SINGLE CONSTANT class** (all 196,608 px = class 2; L* has all 5 classes).
- flip_out_band_interior (ring metric) = 0.4904 (= degenerate init); flip_in_band 0.877.
- **= the §8.1 named risk #1 (through-R optimizer-divergence / dead-output collapse) MATERIALIZED**
  on the easiest possible task (one frame, full overfit). Not ring-free vs B — it has NO structure.

## ARM B (coord-INR control: single-forward RGBWitnessMLX, matched FiLM, SAME R/loss/budget)
- params **63,011**, 434 s.
- realized d_seg through R: 0.5069 → 0.2457 (ep350) → 0.0734 (ep400) → 0.0416 (ep450) →
  **0.036687 (ep500), STILL DESCENDING**.
- realized argmax has all 5 classes, close to L*; flip_in_band 0.385, interior-ring **0.0212**.
- **This is the FRESH single-frame INR floor on the CORRECTED R at 500 ep.** It is NOT comparable to
  the stale `0.004445` (a FULL multi-frame, many-epoch run); the stale number is neither beaten nor
  contradicted here — different budget/scope. The control is HEALTHY (the corrected R trains).

## HEAD-TO-HEAD
| | ARM A (SG-DRF flow) | ARM B (coord-INR) |
|---|---|---|
| realized d_seg through R | **0.506922** (collapsed) | **0.036687** (descending) |
| interior-ring (flip_out_band) | 0.4904 (degenerate) | 0.0212 |
| params | 59,683 | 63,011 |
| A beats B d_seg? | **NO** (A − B = +0.4702) | — |
| A ring-free vs B? | **NO** (constant map) | — |

## VERDICT: FAIL — do NOT escalate SG-DRF; the iterative-depth advantage did NOT materialize
The PASS gate ("the flow drives this one frame's d_seg BELOW the matched-param coord-INR floor,
ring-free") is missed by the maximum margin: the flow collapsed to a constant while the coord-INR
trained cleanly to 0.0367 (still descending) on the identical R/loss/budget/params. Per the design
memo's FAIL gate, do not build SG-DRF. **Honest classification (Catalog #307):** this is
IMPLEMENTATION-level collapse (the through-R end-to-end ODE training was unstable in this single
fixed config — §8.1 risk #1 confirmed REAL), NOT a clean paradigm-level disproof that iterative
depth can never help. The coord-INR remains the working d_seg gate. The burden of proof now sits on
a STABILITY fix (anchored RectifID guidance + zero-init early steps + decouple mechanism-A z-TTO
from weight training + lower LR / grad-checkpoint, per §4/§8.1) demonstrated at $0 BEFORE any
SG-DRF backbone investment; absent that, drop SG-DRF and stay on the coord-INR.

## NEXT (single implied action)
Stay on the coord-INR gate. IF SG-DRF is revisited at all: a $0 stability-only micro-probe
(anchored guidance + zero-init + z-TTO decoupling, single frame) is the ONLY admissible next SG-DRF
step — never a backbone build on a config that collapses on one frame. Do not relaunch a long job.

## Full summary.json (embedded, no signal loss)
```json
{
  "utc": "2026-06-26T16:11:26Z",
  "evidence_grade": "contest-CPU advisory (single-frame, exact scorer, NOT 600-sample harness)",
  "promotion_eligible": false,
  "score_claim": false,
  "render_res": [256, 384],
  "R_fidelity_d_seg_GT_vs_Lstar": 0.0,
  "arm_a_sg_drf": {"d_seg": 0.5069224039713541, "flip_in_band": 0.877293304741482,
    "flip_out_band_interior": 0.49040453951353247, "band_frac": 0.042694091796875,
    "epoch": 50, "phase": "pre_reflow", "n_params": 59683, "C": 24, "n_steps": 4, "secs": 455.3},
  "arm_b_coord_inr": {"d_seg": 0.036687215169270836, "flip_in_band": 0.38456040028591854,
    "flip_out_band_interior": 0.021172707662554326, "band_frac": 0.042694091796875,
    "epoch": 500, "n_params": 63011, "secs": 433.9},
  "head_to_head": {"a_d_seg": 0.5069224039713541, "b_d_seg": 0.036687215169270836,
    "a_minus_b": 0.4702351888020833, "a_interior_ring": 0.49040453951353247,
    "b_interior_ring": 0.021172707662554326, "a_params": 59683, "b_params": 63011,
    "param_ok_a_le_b": true, "a_beats_b_dseg": false, "a_ring_free_vs_b": false},
  "verdict": "FAIL"
}
```
