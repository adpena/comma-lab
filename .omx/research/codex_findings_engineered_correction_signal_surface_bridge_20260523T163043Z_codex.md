# Codex Findings: Engineered-Correction Signal Surface Bridge

**UTC**: 2026-05-23T16:30:43Z
**Lane**: `lane_codex_engineered_correction_signal_surface_bridge_20260523`
**Scope**: legacy engineered-correction targeting and modern byte-shaving planner integration.

## Findings

1. `tac.master_gradient_consumers.engineered_correction_targeting(...)` already emitted a useful per-pair/per-byte sidecar:
   `master_gradient_consumer_engineered_correction_targeting_v1`.
   The sidecar named pair-specific correction leverage points but only reached a Tier-A cathedral consumer rationale string.

2. The modern byte-shaving planner had no unit type for correction-spend opportunities.
   It could model pair/frame/byte/tensor/packet/scorer-response units, but engineered corrections are different:
   they usually spend bytes to reduce distortion rather than save bytes directly.

3. Treating engineered corrections as byte-saving rows would be a score-authority bug.
   The correct canonicalization is a planning-only `correction_target` unit with `candidate_saved_bytes=0`,
   explicit false authority, and blockers for correction synthesis, readiness audit, runtime consumption proof,
   and exact auth eval.

## Landed

- `src/tac/optimization/byte_shaving_campaign.py`
  - Added `correction_target` as a first-class planning unit.
  - Added `apply_engineered_correction` / `probe_correction_neighbor` operation families and order priors.
  - Added `build_signal_surface_from_engineered_correction_targeting(...)`.
  - Preserves `engineered_correction_signal` through ranked units and combination/permutation planning.

- `src/tac/optimization/byte_shaving_signal_surface_builder.py`
  - Added `engineered_correction_targeting_paths` input plumbing.
  - Emits `engineered_correction_refs` and source-signal refs while preserving false-authority semantics.

- `tools/build_byte_shaving_signal_surface.py`
  - Added `--engineered-correction-targeting`, `--engineered-correction-max-targets`,
    and `--engineered-correction-default-delta`.

- `src/tac/cathedral_consumers/engineered_correction_targeting_consumer/__init__.py`
  - Documents the modern canonical sink so the old Tier-A consumer no longer points at an orphan route.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_signal_surface_builder.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_byte_range_entropy_recode_materializer.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_signal_surface_builder.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_pr103_lc_ac_runtime_adapter.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_cathedral_consumer_contract.py src/tac/tests/test_master_gradient_cathedral_consumer_wire_in.py src/tac/tests/test_cable_d_wire_in_master_gradient_consumers.py src/tac/tests/test_master_gradient_consumers_7_to_14.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_decoder_q_selective_runtime_controls.py src/tac/tests/test_dqs1_local_first_queue_builder.py`
- `.venv/bin/ruff check ...` on all touched planner/builder/materializer/scheduler/control files.
- `git diff --check`

## Remaining

Three xhigh explorer subagents are running to search for additional orphaned signal in:

- legacy engineered corrections and correction recode tooling;
- byte-shaving DAG/materializer/storage execution layers;
- MLX/auth-eval/master-gradient/X-ray/canonical-equation/atom/frontier surfaces.

Their findings should be consumed as the next canonicalization tranche rather than left as ad hoc prose.
