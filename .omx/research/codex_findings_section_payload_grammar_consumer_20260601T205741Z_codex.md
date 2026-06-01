<!-- SPDX-License-Identifier: MIT -->
# Codex Findings: Section Payload Grammar Consumer

- Timestamp UTC: 2026-06-01T20:57:41Z
- Scope: close the orphan-signal gap between generic section-payload entropy
  reports and the operator/autopilot surfaces.
- Authority: planning-only packet-compiler signal. `score_claim=false`,
  `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`.

## Landing

Added a cathedral consumer for `section_payload_grammar_optimizer.v1` reports:

- `src/tac/cathedral_consumers/section_payload_grammar_consumer/__init__.py`
- `src/tac/tests/test_section_payload_grammar_consumer.py`

The consumer classifies section grammar reports into the same fail-closed
decision families as tensor grammar:

- `record_section_payload_saturation_and_demote_format_churn`
- `bind_section_receiver_and_materialize_byte_closed_archive`
- `bind_section_receiver_and_materialize_grouped_brotli_archive`

It never promotes score authority. Every output is planning-only and carries
byte-closed archive / runtime-consumption / full-frame replay blockers.

## Optimizer Hardening

Extended `section_payload_grammar_optimizer.v1` grouped-Brotli diagnostics with
the missing byte deltas:

- `identity_grouped_brotli_bytes`
- `selected_isolated_section_bytes`
- `grouped_delta_bytes_vs_identity`
- `grouped_saved_bytes_vs_identity`
- `grouped_delta_bytes_vs_selected_isolated`
- `grouped_saved_bytes_vs_selected_isolated`

`build_section_payload_optimizer_queue(...)` now emits a grouped-order candidate
when the grouped section stream beats selected isolated sections. This makes
section order/packing a queue-consumable operation instead of a report-only
observation.

## Operator Surface

`tools/operator_briefing.py` now scans section grammar reports and consumer
results under the same SSD-aware roots used by tensor grammar, and exposes:

- JSON key: `section_payload_grammar`
- readiness key: `phase_6c_section_payload_grammar`
- text section: `Phase 6c.1b - Generic section payload grammar`

This prevents section-level entropy findings from being stranded in dated JSON
artifacts.

## HiNeRV Premise Check

The operator-routed HiNeRV note is partially supported by existing local memos,
but remains advisory:

- supported: HiNeRV is cheap-by-construction at the smoke point
  (`archive bytes ~= 40 KB` in the landed advisory), and the dense decoder-VJP
  adjoint is machine-exact enough for the G3 path;
- not supported as authority: distortion is not solved locally, and no CPU/CUDA
  exact score claim exists;
- steering implication: do not keep optimizing post-hoc latent tweaks as the
  main path. The joint P18/P19/L-inf objective belongs inside score-aware
  decoder-weight training/export for the cheap carrier.

So the correct status is `rate_positive_distortion_pending`, not
`score_lowering_proven`.

## Verification

```bash
/Users/adpena/Projects/pact/.venv/bin/python -m ruff check \
  src/tac/packet_compiler/section_payload_grammar_optimizer.py \
  src/tac/cathedral_consumers/section_payload_grammar_consumer/__init__.py \
  tools/operator_briefing.py \
  src/tac/tests/test_section_payload_grammar_optimizer.py \
  src/tac/tests/test_section_payload_grammar_consumer.py \
  src/tac/tests/test_operator_briefing.py

/Users/adpena/Projects/pact/.venv/bin/python -m pytest -q \
  src/tac/tests/test_section_payload_grammar_optimizer.py \
  src/tac/tests/test_section_payload_grammar_consumer.py \
  src/tac/tests/test_operator_briefing.py \
  -k 'section_payload'

/Users/adpena/Projects/pact/.venv/bin/python -m pytest -q \
  src/tac/tests/test_check_335_cathedral_consumer_directory_contract.py \
  src/tac/tests/test_section_payload_grammar_consumer.py
```

All three focused verification commands passed.

## Next Build

Move the HiNeRV/SNeRV campaign from post-hoc latent allocation to native
score-aware carrier fitting:

1. bind the joint P18/P19/L-inf saliency objective into decoder-weight training;
2. keep `--modelsize`/byte budget as a first-class training knob;
3. export byte-closed archive candidates;
4. replay locally before any CPU/CUDA exact auth dispatch.
