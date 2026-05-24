# Codex Findings: Family Materializer Zero-Value Parameters

UTC: 2026-05-24T17:56:18Z

## Finding

The family-agnostic packet-member materializer dropped valid zero-valued numeric
parameters before execution because it reused the string-oriented
`ordered_unique()` helper. That helper intentionally skips falsey values, which
erased `zipfile.ZIP_STORED == 0` and produced an empty compression-method set
for stored-only packet-member runs.

The same bug class also applied to numeric materializer sweeps where zero is a
valid value, including ZIP `compresslevel=0` and Brotli `quality=0`.

## Fix Landed

- Added materializer-local integer de-duplication that preserves zero.
- Routed ZIP compression methods, ZIP compresslevels, and Brotli qualities
  through the integer-safe helper.
- Added direct regressions for stored-only packet-member recompression and
  Brotli quality zero.
- Added an end-to-end no-paid packet-member runner handoff smoke:
  plan -> generated context -> materializer -> harvest -> exact-readiness bridge
  -> dispatch-plan refusal.

## Verification

- `ruff check src/tac/optimization/family_agnostic_materializers.py src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`
- `git diff --check`
- `.venv/bin/python -m pytest src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_materializer_chain_harvest_scheduler.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_optimizer_candidate_queue.py -q`

## Carry-Forward Signal

Any optimizer, queue compiler, materializer, PacketIR pass, or learned sweep
that accepts numeric parameter grids must distinguish "missing" from valid
falsey values. This matters for family-agnostic byte shaving because zero can
mean a legitimate codec or search parameter, not "absent."
