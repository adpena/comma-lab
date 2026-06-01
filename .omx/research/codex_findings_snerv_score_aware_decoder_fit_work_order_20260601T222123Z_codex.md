# Codex Findings: SNeRV Score-Aware Decoder-Fit Work Order

UTC: 2026-06-01T22:21:23Z
Agent: codex:gpt-5
Axis: `[macOS-CPU advisory]` / false-authority planning
Lane: `lane_snerv_score_aware_decoder_fit_work_order_20260601`

## What Landed

The waterfill precision-ladder no-go is now converted into a machine-readable
decoder-fit work order instead of living only as prose/adjudication.

New helper:

- `tac.analysis.snerv_score_aware_decoder_fit_work_order`

New CLI:

- `tools/build_snerv_score_aware_decoder_fit_work_order.py`

New tests:

- `src/tac/tests/test_snerv_score_aware_decoder_fit_work_order.py`

## Source Evidence

Input adjudication:

- `.omx/research/snerv_inverse_steg_advisory_waterfill_adjudication_20260601T221200Z.json`
- SHA-256: `eeccd2225b4ad75fd8bc70f34c909683b723b489daee57151d85c48e5ada3993`

Selected row:

- Classification: `rate_below_frontier_pose_or_seg_destroyed`
- Receiver archive replay verified: `true`
- Archive bytes: `33754`
- Step-map mode: `waterfill`
- `d_seg_linf`: `0.02264404296875`
- `d_pose_linf`: `2.1390697956085205`

## Produced Work Order

- Path: `.omx/research/snerv_score_aware_decoder_fit_work_order_20260601T222123Z.json`
- SHA-256: `07547eaca2eaa2eedef0d7e5419287918c6f1c19b54f6c01b45138fbe8a204a8`
- `ready_for_local_decoder_fit_smoke`: `true`
- `ready_for_exact_eval_dispatch`: `false`
- `score_claim`: `false`
- `promotion_eligible`: `false`
- Next action: `run_local_score_aware_decoder_fit_smoke_with_waterfill_packet_in_loop`

## Verification

```text
.venv/bin/ruff check src/tac/analysis/snerv_score_aware_decoder_fit_work_order.py src/tac/tests/test_snerv_score_aware_decoder_fit_work_order.py tools/build_snerv_score_aware_decoder_fit_work_order.py
All checks passed!

.venv/bin/python -m pytest src/tac/tests/test_snerv_score_aware_decoder_fit_work_order.py -q
3 passed in 0.17s
```

## Verdict

GO for local score-aware decoder-fit smoke only.

NO-GO for exact eval, promotion, score claim, or submission. The work order is
only an actuator handoff from replay-verified waterfill no-go into the next
fit/QAT implementation step. It deliberately refuses undercharged rows,
unreplayed rows, and distortion-promising rows that should go to packaging
instead of decoder fitting.

## Next Step

Run or implement the local decoder-fit smoke named in the work order, keeping
the waterfill packet in the training/advisory loop. Escalate only if a follow-up
adjudication satisfies the SegNet/PoseNet preservation ceilings without losing
the rate win.
