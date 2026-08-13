# QS2 R2 — FIRST ADMITTED CANDIDATE of the coupled family (2026-08-13, MAIN adjudication)

## The row [contest-CUDA T4 dual-axis component instrument, n600, batch=16 — matched worker family]
- Candidate: sha `0bb74f1d16e81138975a46c7d4f91f08e1d9fb574930255eac87ccd69f3f8b03` @ 186,286 B
  (cp135 +34 B; the qs1 compensation object recoded at 5.67 B/pair). Call
  `fc-01KZYQJE57P4FV9PS0SXMA2BXM`, run-id `ddm_qs2_dual_axis_20260813_r2`, 978.8 s T4,
  ~$0.16 (#381, ≈$3.1 of $20). Retention on volume
  `comma-ddm-js1b-argmax-retained/ddm_qs2_dual_axis_20260813_r2` (fields + pose vectors +
  pair_error_rms/pair_repeat_noise_rms w/ SHAs) — payload law honored.
- **VERDICT: ADMITTED. Net realized ΔS = −4.374914e-6** (projection −4.3749179e-6
  realized to 6 sig figs):
  - seg: 34,938 vs base 34,970 → **−32 net flips, −2.712674e-5 S** (same 189 changed
    pixels as qs1 — semantic/token/int12 lattice preservation CONFIRMED at the field level)
  - pose: d_pose 6.885829861857928e-6, deterministic repeat IDENTICAL → **+1.126177e-7 S**
    (byte-identical to qs1's leakage: the recode preserved the Schur compensation exactly)
  - rate: +34 B → **+2.263920e-5 S**
- Base = po1/pz4r worker-family pair (34,970 flips · 6.885642960696714e-6), batch-shape
  pin caveat standing. `score_claim: false`.

## Status of the win — BANKED, NOT CLAIMED
Sub-band per the 8dp-band law (|−4.37e-6| < the ±3.5e-6/side evaluate.py canonical band):
this is a full-precision COMPONENT win, unclaimable on the canonical instrument alone.
It is the existence proof + calibration input for ddm_qs3's super-band composed candidate
(target |ΔS| ≥ 1e-5). The floor pointer is UNCHANGED.

## Family arc (five rows, monotone convergence)
JO1 +2.16e-4 → re1 sign-indeterminate (8dp artifact) → js6b census 0/200 → qs1 REFUSED
+2.43e-5 → **qs2 ADMITTED −4.37e-6**. Coding 12.83→5.67 B/pair did it; efficiency
(16.9%, candidate-specific per gca1) and pair-scaling remain un-spent — qs3's levers.

## Apparatus notes
- Poller defect (2nd hand-rolled instance): ledger status 'completed' invalid → crash
  AFTER result write; result unharmed, ledger corrected to 'harvested'. Cure: canonical
  `tools/modal_harvest_poller.py` landed (parameterized; valid statuses; launch via
  launch_detached_process only). Hand-rolled pollers retired.
- qs3 consumes this file's named calibration path per its charter; no directive needed.
