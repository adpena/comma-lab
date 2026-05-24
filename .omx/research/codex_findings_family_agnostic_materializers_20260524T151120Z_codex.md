# Codex Findings: Family-Agnostic Materializers

UTC: 2026-05-24T15:11:20Z

## Scope

Adversarial review and implementation pass over the byte-shaving materializer
registry, queue compiler, and first executable non-DQS1 materializer family.

## Findings Landed

- `archive_section_entropy_recode_v1`, `packet_member_recompress_v1`, and
  `tensor_factorize_v1` now share a canonical family-agnostic materializer CLI
  instead of leaving packet/tensor families as registry-only placeholders.
- The queue compiler now emits executable local work rows for section manifests,
  packet-member recompression, and tensor factorization, with artifact custody,
  input telemetry, pullback paths, and false-authority dispatch blockers.
- Runtime-consumption proof verification now requires candidate archive/member
  SHA binding before `receiver_contract_satisfied` can clear. A naked
  `passed=true` proof is no longer enough.
- Exact-readiness followup remains fail-closed for these candidate-only
  materializers until a chain-level receiver/inflate proof bridge is wired.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/family_agnostic_materializers.py tools/run_family_agnostic_materializer.py src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/comma_lab/scheduler/byte_shaving_materializer_registry.py src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_byte_shaving_campaign_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_byte_shaving_campaign_queue.py -q`
- `git diff --check`
- `.venv/bin/python tools/lane_maturity.py validate`
- `.venv/bin/python tools/review_gate_hook.py`

## Remaining Work

- Promote these candidate-only materializers into chain manifests that include
  runtime-consumption proof, receiver proof, same-runtime inflate parity, and
  exact-readiness refusal/dispatch bridge tests.
- Add additional executable families: tensor quantize/prune/shared-codebook,
  packet member reorder/merge/header-elide, and section reorder/header-elide.
- Feed emitted manifests into learned grouped search so MLX/CPU acquisition can
  select combinations across frame, pair, region, byte, section, member, and
  tensor axes without ad hoc hand wiring.
