# Codex Findings - OP-7 Pose-Byte Hoist Planning Manifest

Date: 2026-05-18 19:37:07 UTC
Author: Codex

## Scope

Canonical task:
`codex_routing_directive_v2_synthesis_followup_null_space_plus_hash_seed_plus_cross_stack_20260518::ITEM_7`.

Cheap-probe directive:
`.omx/research/codex_routing_directive_cheap_probe_wave_pose_axis_op1_op2_op6_op7_op10_20260518.md`.

This lands the missing OP-7 operator CLI:
`tools/hoist_pose_bytes_from_master_gradient.py`.

## Patch

- Added a planning-only OP-7 CLI that reads the aggregate master-gradient anchor
  for archive `f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd`.
- The tool calls
  `tac.master_gradient_consumers.select_pose_axis_dominant_bytes(...)` and
  emits typed `CandidateModificationSpec` rows.
- The manifest keeps all authority flags false:
  `score_claim=false`, `promotion_eligible=false`,
  `ready_for_operator_probe=false`, `ready_for_provider_dispatch=false`, and
  `ready_for_exact_eval_dispatch=false`.
- The manifest explicitly blocks on:
  `scored_archive_custody_missing`,
  `grammar_aware_pose_axis_mutation_builder_missing`,
  `packet_proofs_missing`, and
  `anchor_score_axis_dominance_not_persisted`.
- The selector no longer falls back from missing `scored_archive_sha256` to
  `archive_sha256`. Missing scored-archive custody stays `null` in every
  `CandidateModificationSpec`, and a diagnostic `gradient_byte_domain` of
  `scored_archive_bytes` is emitted as
  `diagnostic_uncustodied_gradient_subject_bytes` until grammar/custody proofs
  exist.
- The manifest records replay hashes for the anchor ledger, canonical anchor
  row JSON, gradient array, and deterministic selector sidecar.
- Ruff configuration now explicitly permits the consumer-ordered `__all__` and
  mathematical notation in `src/tac/master_gradient_consumers.py`, while the
  touched files pass the broader configured Ruff rule set.

## Live Probe

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/hoist_pose_bytes_from_master_gradient.py --top-k 128 --axis-dominance-threshold 0.7
```

Output:

```json
{"manifest_path": ".omx/research/pose_byte_hoist_op7_manifest_20260518.json", "selected_count": 128}
```

The durable manifest is committed at
`.omx/research/pose_byte_hoist_op7_manifest_20260518.json`. The selector sidecar
remains ignored experiment state but is content-addressed from the manifest.
Durable facts from the manifest:

- `selected_count=128`
- `measurement_axis=[macOS-CPU advisory]`
- `measurement_hardware=darwin_arm64_m5_max_macos_cpu_advisory`
- `score_axis_dominance_source=derived_from_gradient_tensor_at_runtime`
- `scored_archive_custody_available=false`
- `selector_sidecar_sha256=a00eaf925eb67f92e01b4a7d6813739a19cd33cdbd4344e63fdfea9b908d351c`
- first diagnostic index: `35781`
- first pose-axis share: `1.0`
- first spec `source_archive_sha256=null`, `source_archive_bytes=null`, and
  `section_name=diagnostic_uncustodied_gradient_subject_bytes`
- all candidate specs are planning-only and non-promotable

## Adversarial Review Fixes

Ohm flagged four issues before commit: scored-archive over-stamping,
non-hermetic manifest identity, nondeterministic selector-sidecar discovery, and
missing tests for the live anchor's no-scored-custody shape. The patch now:

- removes the archive-SHA fallback from the canonical selector;
- adds `scored_archive_custody_missing` blockers at spec and manifest level;
- passes a deterministic sidecar path into the selector instead of globbing the
  latest file after the fact;
- records anchor/gradient/sidecar SHA-256s in the durable manifest; and
- adds focused tests for the no-scored-custody path.

## Residual

This closes the OP-7 manifest/tooling gap, not the packet mutation gap. The next
closure is still the grammar-aware pose-axis mutation builder that resolves a
diagnostic gradient-subject byte index into archive grammar coordinates and
proves repack, ZIP headers, CRC, inflate success, and byte-consumption/no-op
closure.

The current live anchor still lacks a persisted `score_axis_dominance` field.
The tool computes the same dominance from the aggregate tensor and operating
point, records that derivation in the manifest, and leaves the persistence gap
as an explicit blocker.

The current live anchor also lacks `scored_archive_sha256` and
`scored_archive_bytes`. That is now treated as a custody blocker rather than an
implicit archive identity.

## Verification

- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_hoist_pose_bytes_from_master_gradient.py]`
  - Result: `4 passed`
- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_hoist_pose_bytes_from_master_gradient.py src/tac/tests/test_master_gradient_consumers.py]`
  - Result: `46 passed`
- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check tools/hoist_pose_bytes_from_master_gradient.py src/tac/master_gradient_consumers.py src/tac/tests/test_hoist_pose_bytes_from_master_gradient.py src/tac/tests/test_master_gradient_consumers.py pyproject.toml]`
  - Result: `All checks passed`
- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check --select F821 src/ experiments/ submissions/robust_current/ scripts/ tools/]`
  - Result: `All checks passed`
