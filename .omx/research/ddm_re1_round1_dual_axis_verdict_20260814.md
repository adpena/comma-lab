# ddm_re1 Round-1 dual-axis verdict — ADMITTED net −1.207e-6 (2026-08-14)

Two Modal T4 dispatches closed the re1 arc (~$0.32 total; calls
fc-01KZZVQNYVFTZ6D7YWWEARKP0Z seg r2 · fc-01KZZX0FGDJ1VWFTJHCWPXDC9B pose).
Candidate: RE1 Round-1 probability-object edit, archive sha 7be3eb94b2293062…,
186,252 B — byte-EQUAL to the cp135 base. Matched worker-family instrument
(base 34,970 flips · d_pose 6.885642960696714e-6). Status: component rows only,
score_claim=false, frontier unmoved.

## THE ROW (matched instrument, n600, deterministic repeat identical)

| leg | realized | ΔS |
|---|---|---|
| seg | 34,968 flips (−2; 4 changed px in the exact T4 field) | **−1.6954210069444444e-06** |
| pose | d_pose 6.8864537752233446e-6 (repeat identical, +8.108e-10) | **+4.885471577790318e-07** |
| rate | 0 B (byte-equal archive) | **0.0** |
| **net** | | **−1.2068738491654126e-06 ADMITTED** |

Verdict: **ADMITTED_NET_NEGATIVE_SUB_BAND_BANKED** (verdict_scope: INSTANCE —
RE1 Round-1 archive 7be3eb94 on the cp135 base). Sub the ±3.5e-6 8dp-report
band and the 1e-5 canonical naming bar — BANKED, not named.

## WHAT THIS RESOLVES

1. **#1032 errata CLOSED.** The retracted "+4.03e-6 WORSE" full-auth read was
   pure report quantization (one 8dp pose ULP). The dual-axis measurement the
   errata demanded now exists: Round-1 IS a micro-win.
2. **"Receiver-null" REFUTED.** The byte-equal edit changes 4 pixels in the
   exact T4 SegNet field. Probability-object edits inside the same byte
   envelope are a REAL zero-rate seg actuator (this instance: −2 flips).
3. **Pose tax measured tiny but real** (+4.885e-7 S): a 4-pixel frame edit
   still leaks into PoseNet. Consistent with the qs-family law — frame-1 edits
   without in-compile compensation carry a small pose tax; at this size it
   does not flip the sign.

## THE BANK (admitted zero/tiny-rate candidates on cp135)

qs2 −4.374914e-6 (+34 B) · **re1 −1.2068738e-6 (0 B)**. A composed union
projects ≈ −5.6e-6 IF non-interacting — still below the 1e-5 naming bar, so
the union fire is HELD until the banked pool projects ≥ 1e-5. Each new
sub-band admit joins this pool; the pool is the path to a named row without
new mechanism.

## APPARATUS BANKED

Committed seal builder `experiments/ddm_re1_pose_leg_seal.py` (9207d5eac0):
reuses the seg-leg candidate bytes exactly, honors the worker pose-placeholder
law (local_pose_delta literal 0.0), validates through the dispatcher hash gate
pre-fire. Versioned-rN reseal path in the RE1T dispatcher (a53f988b68) cured
the seal-drift-vs-volume-resume catch-22 without touching retained payloads.

Store: `.../round_01_singleton_best/re1_dual_axis_pose/` (sealed request sha
4884623c…, remote result + LOCAL_ADJUDICATION.json, all payloads retained on
the volume under `/ddm_js1b_retained/ddm_re1_dual_axis_pose_20260814_r1`).
Lane `ddm_re1_dual_axis_pose_n600_20260814` closed terminal. Modal ≈ $4.6/$20.
