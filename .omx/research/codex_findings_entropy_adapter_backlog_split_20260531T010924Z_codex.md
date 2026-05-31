# Codex Findings: Entropy Adapter Backlog Split

UTC: 2026-05-31T01:09:24Z

## Scope

Reviewed the current archive/entropy repair path after sidecar agent harvest.
Two suspected gaps were already fixed on `main`: the proxy authority alias guard
now rejects newer exact/spend authority names, and MLX adapter-spine triage now
requires archive-bound custody, receiver proof, and a fresh contract.

The remaining live gap was in the range/ANS materializer backlog. Rows marked
`runtime_adapter_ready=true` only had a member-level decode helper. They did not
open the next contest-runtime adapter task explicitly, so the signal could be
misread as more complete than it was.

## Change

`repair_archive_variant_materializer_backlog.v1` rows now split:

- `member_decode_helper_ready`
- `contest_runtime_decoder_adapter_ready`
- `contest_runtime_adapter_integrated`
- `contest_runtime_adapter_integration_required`
- `smallest_contest_runtime_adapter_task`
- `next_materializer_action`
- `contest_runtime_adapter_integration_blockers`

For current range/ANS prototypes, the queue remains fail-closed and now routes
the smallest next executable work item as
`integrate_contest_runtime_decoder_adapter`. This preserves member-roundtrip
signal without promoting it to contest-runtime authority.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/repair_family_byte_transform_executor.py src/tac/tests/test_repair_family_materializers.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_repair_family_materializers.py`
- `.venv/bin/python tools/review_tracker.py policy-check src/tac/optimization/repair_family_byte_transform_executor.py`
- `.venv/bin/python tools/review_tracker.py policy-check src/tac/tests/test_repair_family_materializers.py`

## Next

The next highest-EV follow-up is to implement the actual contest-runtime decoder
adapter for one range/ANS prototype against a live archive/runtime pair, then
make the receiver proof exercise `inflate.sh` with decoded member consumption
rather than local member roundtrip only.
