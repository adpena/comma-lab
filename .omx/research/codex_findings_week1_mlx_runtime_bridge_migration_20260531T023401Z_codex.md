# Codex Findings: Week 1 MLX Runtime Bridge Migration

UTC: 2026-05-31T02:34:01Z

## Scope

Operator directive: migrate remaining MLX/archive emitters onto the shared
runtime bridge and do not let advisory rows masquerade as score authority.

This pass landed the next contract-first migration slice:

- Z5 Rao-Ballard MLX archive exporter now emits the shared
  `tac_archive_bound_candidate_runtime_adapter_package.v1` by default.
- PACT-NeRV selector v2/v3/v4 MLX archive exporters now emit the shared
  runtime-bridge package by default.
- PACT-NeRV VQ MLX archive exporter now emits the shared runtime-bridge package
  by default.
- PR95 MLX PyTorch package/export now emits a shared archive-bound runtime
  bridge package around `archive.zip`, `inflate.sh`, `inflate.py`, vendored
  runtime files, receiver proof, replay metadata, exact blockers, and posterior
  hook.

## Mechanism-Level Fix

The bridge previously considered a generated `inflate.sh` proof valid whenever
it returned zero and produced any non-empty output at the expected path. That
was too weak for MLX smoke emitters: a tiny Z5 or PR95 package could produce a
valid local raw stream and be marked receiver-ready despite not being a
full-contest 1200-frame packet.

The bridge now supports expected receiver output name and expected receiver
output byte count. Z5 and PR95 pass the full contest raw-byte gate:

`1164 * 874 * 1200 * 3`

Tiny smoke archives therefore emit useful bridge packages but fail closed with
`*_generated_inflate_sh_output_bytes_mismatch` instead of entering exact-ready
custody.

## Contract Properties

All migrated packages preserve false authority:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

The shared adapter package still emits deterministic replay bundles,
MLX-triage request stubs, receiver proof gates, exact-axis blockers, and
posterior update hooks. MLX remains advisory; CPU/CUDA exact auth remains the
only score authority.

## Classification Tightening

The archive-bound contract now separates neural archives from predictive coding
archives. PACT-NeRV selector/VQ rows are tagged as neural/MLX/Pact-NeRV rows,
not as Z7 or generic predictive-coding rows. Z5 remains predictive-coding and
Z5-tagged.

## Audit Notes

DQS1 harvest, public-frontier intake, and the byte-shaving signal surfaces were
already on shared archive-bound contract surfaces in the current tree. Focused
DQS1/public-frontier and byte-shaving signal tests passed.

The broader `test_byte_shaving_campaign_queue.py` suite still has pre-existing
failures unrelated to this slice:

- registry expected-order drift after additional materializer kinds;
- materializer-chain fixture manifests missing fields now required by current
  strict completion contracts;
- DFL1 parity path/root rejection ordering mismatch.

Those failures were not hidden or weakened. They should be treated as a
byte-shaving queue cleanup blocker before claiming the full queue surface is
green.

## Validation

Passed:

- `ruff` on the touched bridge, contract, exporter, PR95 package tool, and tests.
- `pytest src/tac/tests/test_archive_bound_runtime_bridge_remaining_mlx_emitters.py src/tac/tests/test_archive_bound_candidate_adapter_spine.py src/tac/tests/test_pr95_mlx_pytorch_archive_package.py -q`
  - 16 passed.
- `pytest src/tac/substrates/z6_v2_cargo_cult_unwind/tests/test_z6_v2_mlx_renderer_and_bridge.py::test_export_z6_v2_mlx_archive_emits_fail_closed_candidate_package_for_truncated_smoke src/tac/tests/test_z7_mamba2_mlx_module_smoke.py::test_z7_mamba2_mlx_canonical_ssd_backend_uses_helper_and_exports_bridge -q`
  - 2 passed.
- `pytest src/tac/tests/test_public_frontier_intake.py src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_dqs1_local_first_autopilot.py -q`
  - 60 passed.
- `pytest src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_signal_surface_builder.py -q`
  - 53 passed.
- `git diff --check` on the touched slice.

Known failing check, preserved as blocker:

- `pytest src/tac/tests/test_public_frontier_intake.py src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_dqs1_local_first_autopilot.py src/tac/tests/test_byte_shaving_campaign_queue.py -q`
  - 146 passed, 8 failed.
  - Failures are all in `test_byte_shaving_campaign_queue.py` and no touched
    file in this slice modifies that queue or its tests.

## Subagent Note

An xhigh subagent spawn for the remaining emitter audit was attempted but the
local agent thread limit was reached. The audit continued locally against the
canonical code surfaces.
