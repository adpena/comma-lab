# DDM ET3 Checkpoints

## Durable State

Primary SSD bulk dir: `/Volumes/VertigoDataTier/pact/ddm_et3_20260806`

| file | role | SHA-256 |
|---|---|---|
| `/Volumes/VertigoDataTier/pact/ddm_et3_20260806/et3_solve_within_cvp_rows.jsonl` | per-pair checkpoint JSONL, 32 rows | `4634e5ceb62822a7ed0de2678c14da1c662485b96f85c03fd304e5057b1fe83d` |
| `/Volumes/VertigoDataTier/pact/ddm_et3_20260806/et3_solve_within_cvp_summary.json` | aggregate plus embedded rows | `3e74d83a6c78677bfd4776db92dda4fa0526665ce696465e1ca76797302e2d67` |
| `.omx/research/ddm_et3_20260806/et3_solve_within_cvp_summary.json` | durable in-repo mirror | `3e74d83a6c78677bfd4776db92dda4fa0526665ce696465e1ca76797302e2d67` |

No persisted evidence lives under `/tmp`.

## Resume Command

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_et3_solve_within_cvp_phase_field.py --resume
```

The runner loads `/Volumes/VertigoDataTier/pact/ddm_et3_20260806/et3_solve_within_cvp_rows.jsonl`, skips completed pairs by `pair`, and rewrites both summary JSON files after each completed row.

## Measurement Checkpoints

| checkpoint | value |
|---|---:|
| completed rows | 32 |
| requested n32 pairs | 32 |
| solve cap-stop rows | 32 `cap_bound` |
| DK1 realized blocks | 12488 |
| DK1 candidate scope | 12488 `EXHAUSTIVE_TAP_RADIUS_PRODUCT` |
| DK1 exact declared finite-scope blocks | 12488 |

All 32 rows hit the cap-bound solver stop, so the receipt is a cap-limited measurement, not a convergence claim. DK1 exactness is exact only for the declared finite tap-radius candidate product; no global MIQP/integer optimum is claimed.

## Validation Already Run

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile experiments/ddm_et3_solve_within_cvp_phase_field.py src/tac/optimization/lattice_native_pose_null_realizer.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/optimization/tests/test_lattice_native_pose_null_realizer.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_et3_solve_within_cvp_phase_field.py --limit 1 --cap-ladder 15 --steps 15 --eval-every 5 --cvp-tap-radius 0 --bulk-dir /Volumes/VertigoDataTier/pact/ddm_et3_20260806_smoke_clip --receipt-dir .omx/research/ddm_et3_20260806_smoke_clip
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_et3_solve_within_cvp_phase_field.py --resume
```

Smoke rows under `/Volumes/VertigoDataTier/pact/ddm_et3_20260806_smoke*` are diagnostic only and were not mixed into the authority n32 row.
