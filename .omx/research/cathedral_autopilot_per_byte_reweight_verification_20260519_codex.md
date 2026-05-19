# Cathedral Autopilot Per-Byte Reweight Verification - Codex

Date: 2026-05-19
Task id: `codex_routing_directive_comprehensive_wire_in_and_integration_pass_20260519T072000Z::WIRE_IN_2_VERIFY`
Source directive: `.omx/research/codex_routing_directive_comprehensive_wire_in_and_integration_pass_20260519T072000Z.md`

## Verdict

VERIFIED. Wire-in #2 does not need a manual edit to `tools/cathedral_autopilot_autonomous_loop.py`: the slot-6 `per_byte_sensitivity_consumer` satisfies the Catalog #335 auto-discovery contract, declares the intended Catalog #125 hooks, and is runtime-invoked through the cathedral autopilot consumer path.

No score claim, promotion claim, rank/kill authority, or exact-eval readiness is created by this verification. The live anchor consumed in the probe is macOS advisory / 8-pair diagnostic evidence and is correctly surfaced as advisory.

## Evidence

Focused tests:

```text
.venv/bin/python -m pytest src/tac/tests/test_master_gradient_per_byte_consumer.py src/tac/tests/test_per_byte_sensitivity_consumer.py -q
50 passed in 0.34s

.venv/bin/python -m pytest src/tac/tests/test_cathedral_autopilot_autonomous_loop.py -k 'cathedral_consumer or master_gradient or invocation' -q
8 passed, 162 deselected in 0.26s

.venv/bin/ruff check src/tac/tests/test_per_byte_sensitivity_consumer.py
All checks passed.
```

Live discovery/invocation probe:

```json
{
  "production_consumer_count": 22,
  "per_byte_in_modules": true,
  "per_byte_registration": {
    "consumer_name": "per_byte_sensitivity_consumer",
    "consumer_version": "1.0",
    "consumer_module_path": "tac.cathedral_consumers.per_byte_sensitivity_consumer",
    "consumer_hook_numbers": [1, 3, 4],
    "contract_compliant": true,
    "validation_errors": [],
    "waiver_active": false
  },
  "invocation_row": {
    "candidate_id": "codex_live_probe_per_byte_sensitivity_fec6",
    "consumer_name": "per_byte_sensitivity_consumer",
    "axis_tag": "[macOS-CPU advisory]",
    "predicted_delta_adjustment": 0.0,
    "promotable": false,
    "confidence": 0.0,
    "rationale": "per-byte sensitivity available for 6bae0201fb08: 178417 bytes (162125 non-zero, 9.1% sparse), top-100 indices ranked by L1-sum-of-abs importance (aggregate_sum=0.2708); hardware=darwin_arm64_local_cpu_advisory axis=[macOS-CPU advisory] [predicted]"
  },
  "payload_authority": {
    "score_claim": false,
    "promotion_eligible": false,
    "ready_for_exact_eval_dispatch": false,
    "evidence_grade": "[predicted, cathedral consumer invocation]"
  }
}
```

## Hook Declaration

| Hook | Status | Evidence |
| --- | --- | --- |
| #1 sensitivity map | Active | `CONSUMER_HOOK_NUMBERS` includes `HookNumber.SENSITIVITY_MAP`; payload ranks per-byte L1 sensitivity. |
| #2 Pareto constraint | N/A for this consumer | Per-pair sister consumers own Pareto surfaces; this per-byte consumer is observability-only. |
| #3 bit allocator | Active | `CONSUMER_HOOK_NUMBERS` includes `HookNumber.BIT_ALLOCATOR`; verdict notes expose top-K sensitive byte indices. |
| #4 cathedral autopilot dispatch | Active | `discover_compliant_consumer_modules()` includes the consumer; `invoke_cathedral_consumers_on_candidates()` calls it in report-only and post-loop paths. |
| #5 continual-learning posterior | Producer-side | Anchor persistence remains in `.omx/state/master_gradient_anchors.jsonl`; this consumer does not append posterior rows. |
| #6 probe disambiguator | N/A for this consumer | The payload is not a 2-mode arbitration surface. |

## Adversarial Review

1. Authority boundary preserved: the consumer returns `predicted_delta_adjustment=0.0`, `promotable=false`, and the enclosing autopilot payload has `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`.
2. Axis custody preserved: the live FEC6 anchor is `darwin_arm64_local_cpu_advisory`; the invocation row emits `[macOS-CPU advisory]`, not `[contest-CPU]`.
3. Runtime callsite present: auto-discovery is not a dead helper. `invoke_cathedral_consumers_on_candidates()` is called from both report-only and normal loop output paths.
4. Count guard corrected: the live production discovery floor is 22 consumers, not the stale "22 existing + 1 = 23" text. The test name/docstring now matches the enforced `>= 22` floor.

## Residual Risks

- The current per-byte payloads are advisory 8-pair signals. They can prioritize investigation but cannot justify score movement, promotion, rank/kill, or contest-axis claims.
- The bit-allocator hook exposes top-K indices as notes; no canonical per-byte allocator mutation is implemented by this task.
- If a future authoritative 600-pair contest-axis per-byte anchor lands, this consumer should be re-probed to verify the axis tag remains faithful and downstream rankers still keep it non-promotional until exact-eval evidence exists.

## Closure

Wire-in #2 is closed as verified. No follow-up Atom is needed for the cathedral autopilot reweight path itself. Follow-up work should target actual byte-closed mutation/allocator use of the top-K sensitivity notes, with packet proofs and exact-eval custody before any score claim.
