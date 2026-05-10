# T1 Modal full dispatch from ce1b4e2d — Codex ledger

Date: 2026-05-10

## Dispatch

Launched the first full 600-pair T1 Ballé end-to-end Modal T4 run after the
subset guard was hardened to avoid false auth-eval attempts.

- Lane: `t1_balle_128k_endtoend`
- Instance/job id: `t1_balle_modal_full_ce1b4e2d_20260510T1333Z`
- Modal call id: `fc-01KR91CR2V5AM6Y2DVBFTXCDVM`
- Modal app: `comma-t1-balle-endtoend`
- Local mounted commit: `ce1b4e2d8804c07abd2ee8d7beae9c95fc98e754`
- Mounted-code dirty state: `false`
- GPU: Modal T4
- Estimated cost: `$14.16`
- Predicted ETA: `2026-05-11T13:31:56Z`

Command:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach \
  experiments/modal_t1_balle_endtoend.py --execute \
  --label t1_balle_modal_full_ce1b4e2d_20260510T1333Z \
  --epochs 3000 \
  --batch-size 16 \
  --timeout-hours 24.0 \
  --cost-cap-usd 80.0 \
  --train-timeout-hours 22.5 \
  --sinkhorn-max-positions-per-chunk 2048
```

Recover:

```bash
.venv/bin/python experiments/modal_t1_balle_endtoend.py recover \
  --label t1_balle_modal_full_ce1b4e2d_20260510T1333Z
```

## Custody and score-claim status

The run is **in flight**. There is no score claim, no promotion claim, and no
rank/kill decision until recovery verifies exact contest-CUDA evidence:

- `auth_eval_schema` blockers equal zero;
- `n_samples == 600`;
- device is CUDA on contest-equivalent hardware;
- archive SHA and size match the scored packet;
- score is recomputed from components;
- dispatch claim is closed terminally.

Plan metadata recorded `contest_cuda_auth_eval_requested=true` because
`max_target_pairs=None` means full 600-pair export. This is intentionally
different from bounded guard runs, which now remain training/export smoke only.

## Next action

Do not launch a duplicate T1 run while the active claim exists. Harvest with the
recover command above, then classify the result as one of:

- `completed_t1_contest_cuda_recovered` with a valid exact score packet;
- `completed_t1_training_only_recovered_no_score_claim` if training succeeds
  without valid score evidence;
- `failed_t1_modal_recovered_no_score_claim` with the exact blocker if the
  remote script fails.
