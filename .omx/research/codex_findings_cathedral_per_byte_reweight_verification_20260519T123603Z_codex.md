# Codex Findings - Cathedral Per-Byte Reweight Verification

Date: 2026-05-19T12:36:03Z
Task id: `codex_routing_directive_comprehensive_wire_in_and_integration_pass_20260519T072000Z::WIRE_IN_2_VERIFY`

## Findings

No P0/P1 blockers found.

### Finding 1 - Wire-in #2 is runtime-active

Severity: pass

`per_byte_sensitivity_consumer` is not merely present as a helper. It is contract-compliant, auto-discovered, carries hooks `[1, 3, 4]`, and is invoked through `invoke_cathedral_consumers_on_candidates()`.

Live probe evidence on FEC6 archive `6bae0201fb08...`:

- production consumer count: 22
- per-byte consumer discovered: true
- contract-compliant: true
- hooks: sensitivity map, bit allocator, cathedral autopilot dispatch
- invocation emits `predicted_delta_adjustment=0.0`
- invocation emits `promotable=false`
- enclosing payload emits `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`

### Finding 2 - Axis custody is preserved

Severity: pass

The consumed FEC6 per-byte anchor is macOS advisory / 8-pair diagnostic evidence. The consumer emits `[macOS-CPU advisory]`, not `[contest-CPU]`, so it does not recreate the stale-authority bug class previously caught around master-gradient rows.

### Finding 3 - Stale production-count text was misleading

Severity: P3

The live production discovery count is 22. The regression guard already enforced `len(modules) >= 22`, but the test name/docstring claimed "at least 23" and cited stale "22 existing + 1" language. I changed the test name/docstring to the live guarded floor so future reviewers do not infer a missing consumer from stale text.

## Verification

```text
.venv/bin/python -m pytest src/tac/tests/test_master_gradient_per_byte_consumer.py src/tac/tests/test_per_byte_sensitivity_consumer.py -q
50 passed in 0.34s

.venv/bin/python -m pytest src/tac/tests/test_cathedral_autopilot_autonomous_loop.py -k 'cathedral_consumer or master_gradient or invocation' -q
8 passed, 162 deselected in 0.26s

.venv/bin/ruff check src/tac/tests/test_per_byte_sensitivity_consumer.py
All checks passed.
```

## Residual Risk

This closes only the cathedral visibility/reweight path. It does not create a byte-closed mutation, bit-allocation campaign, or contest-axis score authority. The next useful step is to consume the top-K byte notes in a packet-valid mutation/allocator path and require exact CUDA/CPU eval before score language.
