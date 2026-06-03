# Codex Findings: SNeRV Temporal Context Receiver-Real Fix

UTC: 2026-06-03T00:08Z
Author: Codex
Axis: implementation correctness / no-fake control hardening
Score claim: false
Promotion eligible: false
Ready for exact dispatch: false

## Finding

The adversarial review found that SNeRV `temporal_context` was accepted by
candidate IDs, metadata, queue controls, and archive headers, but the HF decoder
feature basis did not consume temporal LF-delta features. That made nonzero
`temporal_context` a no-op control at the implementation layer.

## Fix

`SnervModelSizeConfig.feature_count` now charges two receiver-visible LF-delta
features per temporal radius. HF decoder fitting consumes same-channel LF
timelines via `temporal_group_count`, and SNAR1 archive replay reconstructs the
dequantized LF sequence from packet bytes before calling `decode_frame`.

This keeps the feature receiver-real:

- no scorer or torch dependency in inflate;
- no hidden sidecar state;
- nonzero `temporal_context` changes decoder bytes;
- nonzero `temporal_context` requires LF sequence inputs for direct decode;
- archive replay supplies those sequence inputs from archived LF planes.

## Verification

- `src/tac/substrates/snerv_inverse_steg_carrier/tests/test_carrier.py` proves
  temporal context increases decoder capacity and changes decoded pixels.
- `src/tac/substrates/snerv_inverse_steg_carrier/tests/test_archive.py` proves
  SNAR1 replay consumes temporal context from receiver-visible LF sequence bytes.
- `src/tac/tests/test_nerv_modelsize_budget.py` now charges temporal-context
  features in nominal decoder byte accounting.

Focused verification run: 134 SNeRV substrate tests passed, plus the affected
modelsize/planner/feedback suite. Ruff passed for the touched SNeRV and NeRV
planning surfaces.

## Remaining Work

The current SNeRV native MLX export still uses uniform closed-form allocation
for LF step maps. That remains a score-lowering blocker until true P18/P19
waterfill is wired into native export and campaign admission.
