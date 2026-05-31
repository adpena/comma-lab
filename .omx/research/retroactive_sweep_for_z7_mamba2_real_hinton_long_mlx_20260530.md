# Retroactive verdict-taint sweep — Z7-Mamba-2 REAL-Hinton-teacher long MLX run

**Date:** 2026-05-30
**Lane:** `lane_z7_mamba2_real_hinton_long_mlx_run_20260530`
**Trigger:** REAL-Hinton-teacher pose-axis path landed (extincts the Wave N+11
mock-teacher pose=0 gap). Per Catalog #348 (event-driven retroactive verdict-
taint sweep) this landing's bug-class fix (mock-teacher-config-not-real-teacher)
invalidates the evidence basis of any prior KILL / DEFER / FALSIFY verdict that
was based on the mock-teacher pose=0 anchor.

## 1. Bug-class symptom signature

The Wave N+11 stabilizer re-fire landed a canonical-equation anchor whose
`measurement_method` was
`wave_n11_stabilizer_600pair_50ep_mlx_local_mock_teacher` — pose-axis = 0 by
construction (the mock `MockTeacherLogitsProvider` has no PoseNet). Any
downstream verdict that consumed Z7-Mamba-2's pose-axis as "0" or "unmeasured"
was operating on a mock signal, NOT the real scorer-bound signal.

## 2. Pre-fix window

2026-05-28 (Wave N+9 Slot 1 L1 SCAFFOLD landing) -> 2026-05-30 (this landing).
The mock-teacher anchors are anchors #5 (mock) + #6 (Wave N+11 mock stabilizer)
on `z7_mamba2_state_space_predictive_coding_pose_axis_savings_v1`. (Anchors #3
+ #4 are real-teacher Wave N+10 RESUME at reduced lr — those are NOT mock and
are NOT tainted.)

## 3. Historical KILL/DEFER/FALSIFY verdict search

Verdicts examined (queried via the probe-outcomes ledger + the Wave N+11
HALT + composition memos):

| Verdict | Subject | Taint classification | RE-EVAL priority |
|---|---|---|---|
| HALT (commit `ee15561e9`) | Wave N+11 QUAD composition (Z6-v2 + Z7-Mamba-2 + NSCS06 v8 + Compound C) | Z7-Mamba-2's mock-teacher pose=0 anchor was flagged as phantom-provenance per Catalog #322. This landing provides the REAL-teacher path; the empirical real pose-axis anchor (op-routable #1) supersedes the mock anchor. | **MEDIUM** — the HALT was CORRECT (the mock anchor WAS phantom); this landing REMOVES Z7-Mamba-2 as the mock blocker but the QUAD composition reactivation still requires the OTHER 3 substrates to carry real anchors. Re-eval the QUAD composition ONLY after all 4 carry real anchors. NOT a reactivation of the composition by Z7-Mamba-2 alone. |
| Wave N+11 stabilizer anchor #6 (mock) | Z7-Mamba-2 pose-axis | The stability claim (50/50 no NaN) is INTACT (stability is independent of the score surface); the pose-axis=0 is a mock artifact, not a real falsification of Z7-Mamba-2's pose-axis. | **LOW** — anchor #6 is a STABILITY anchor, correctly noted as mock-teacher in its own `notes`. No verdict tainted; the real anchor ADDS (anchor 7), it does NOT supersede the stability claim. Per Catalog #110/#113 APPEND-ONLY, anchor #6 is preserved. |
| L1 SCAFFOLD (Wave N+9 Slot 1) | Z7-Mamba-2 substrate | The scaffold's `canonical_ssd_mlx_backend_not_wired` blocker (reference_s6 recurrence vs canonical SSD) is UNRELATED to the teacher; not tainted by this fix. | **NONE** — out of scope; the canonical-SSD-backend blocker is a separate lane. |

## 4. Per-finding RE-EVAL-priority assignment

- **MEDIUM**: Wave N+11 QUAD composition — re-eval ONLY when all 4 substrates
  carry real (non-mock, non-byte-identical) anchors. This landing satisfies
  Z7-Mamba-2's leg. Operator-routable to the composition lane (sister B / Wave
  N+12), NOT this lane.
- **LOW**: Wave N+11 stabilizer anchor #6 — no re-eval needed (stability claim
  intact; mock-teacher correctly self-disclosed; real anchor appends).
- **NONE**: canonical-SSD-backend blocker — separate lane, untainted.

## 5. Conclusion

This landing does NOT trigger any KILL/FALSIFY reversal. It REMOVES Z7-Mamba-2
as the mock-teacher blocker in the Wave N+11 QUAD HALT (the HALT was correct;
the mock anchor was correctly flagged phantom per Catalog #322). The QUAD
composition reactivation remains gated on the other 3 substrates per the HALT
memo — that re-eval is operator-routable to the composition lane when those land
real anchors. Per CLAUDE.md "Forbidden premature KILL": no verdict is reversed
prematurely; the real-teacher anchor is APPEND-ONLY evidence (anchor 6 -> 7).
