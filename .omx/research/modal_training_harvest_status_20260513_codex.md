# Modal training harvest status (2026-05-13)

## Command

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/harvest_modal_calls.py
```

No dispatch was launched. This was a Modal FunctionCall recovery/harvest pass
over existing `experiments/results/lane_*_modal/modal_metadata.json` records.

## Result

- scanned Modal training metadata rows: `65`
- already harvested: `57`
- old Modal call IDs no longer available: `6`
- expired Modal cache: `1`
- old import-path failure recovered from Modal result: `1`
- active dispatch claims after the pass: `0`

Non-harvestable historical rows:

- `lane_sa_v4`: Modal `NotFoundError`
- `lane_sc_plus_plus_v4`: Modal `NotFoundError`
- `lane_so_v3`: Modal `NotFoundError`
- `mae_v_v2`: Modal `NotFoundError`
- `q_faithful_v3`: Modal `NotFoundError`
- `stc_cuda`: Modal `NotFoundError`
- `sz_phase2_v2`: Modal output cache expired
- `track1_phase_a1_score_gradient_latentalign_lr2e6_20260509T011929Z_modal`:
  `ModuleNotFoundError: No module named 'tac'`

## Classification

This is provider-state cleanup, not score movement:

- `score_claim=false`
- `dispatch_attempted=false`
- `ready_for_exact_eval_dispatch=false`

The one import-path failure is already represented by newer harvested
`track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_20260509T012628Z_modal`
artifacts, so it does not reopen the lane.

## Next action

Do not refire any of these historical training calls. Continue score-lowering
from byte-closed PacketIR/PR106 sidecar work and exact-eval candidates with
fresh dispatch claims.
