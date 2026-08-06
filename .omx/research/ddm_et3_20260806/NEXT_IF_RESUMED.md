# DDM ET3 Next If Resumed

Current decision: `HELD_POSE_BOUND_FAIL_NO_FIRE_ORDER_2`.

Do not fire full n600 byte-close from ET3 as-is. The n32 eta is green, but pose max fails the chartered gate:

| metric | value |
|---|---:|
| eta | 0.3562364031907179 |
| eta bar | 0.1710048742006269 |
| pose median | 1.00314130363039 |
| pose max | 1.128389479902771 |

## Immediate Next Work

1. If continuing this formulation, attack pose outliers first on the same fixed n32 set, starting with pairs `485`, `521`, `471`, and `48`.
2. Keep the SW1 solve-within null-basis target and DK1 CVP path fixed unless the receipt names the changed component. The observed blocker is pose max, not eta.
3. A valid fire-order revival requires a new n32 receipt with eta still above `0.1710048742006269`, pose median near 1.00, and pose max under the accepted bound. MAIN must adjudicate any full n600 byte-close launch.
4. If no pose-bound repair is attempted, leave ET3 classified as formulation-held, not family-folded. ET3 refutes neither solve-within eta nor DK1 CVP; it only blocks automatic promotion because pose outliers remain.

## Revalidation Commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile experiments/ddm_et3_solve_within_cvp_phase_field.py src/tac/optimization/lattice_native_pose_null_realizer.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/optimization/tests/test_lattice_native_pose_null_realizer.py
jq '.aggregate' .omx/research/ddm_et3_20260806/et3_solve_within_cvp_summary.json
wc -l /Volumes/VertigoDataTier/pact/ddm_et3_20260806/et3_solve_within_cvp_rows.jsonl
```

Boundary reminders: no score claim, no promotion, no `archive.zip`, no `upstream/evaluate.py`, no contest-CPU/CUDA inference from this n32 advisory row.
