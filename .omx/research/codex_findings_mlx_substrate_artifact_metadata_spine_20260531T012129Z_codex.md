# Codex Findings - MLX Substrate Artifact Metadata Spine

## Verdict

LANDED. The shared MLX score-aware long-training path now has a non-authority
substrate metadata channel that survives:

- `RendererBundle`
- `MlxScoreAwareAdapter`
- `TrainingArtifact`
- `training_artifact.json`
- canonical Provenance input hashing
- MLX posterior manifest handoff when archive bytes exist

## Gap Closed

Z7 Mamba-2 full MLX training previously carried backend-lineage truth in smoke
and archive metadata, but the shared long-training artifact path did not expose
a reusable substrate metadata channel. That made `reference_s6_mlx` versus
canonical Mamba2 SSD lineage easy to lose in long MLX runs.

## Guardrail

`substrate_artifact_metadata` is deliberately non-authoritative. It recursively
rejects readiness and score authority keys such as:

- `score_claim`
- `promotion_eligible`
- `ready_for_exact_eval_dispatch`
- `rank_or_kill_eligible`
- `promotable`
- `score_claim_valid`

Canonical custody remains owned by `TrainingArtifact` itself.

## Z7 Mamba Wiring

`experiments/train_substrate_time_traveler_l5_z7_mamba2_mlx_local.py` now
threads:

- `schema = mlx_substrate_backend_lineage.v1`
- `mamba2_mlx_backend_lineage = reference_s6_mlx`
- `canonical_ssd_mlx_backend_wired = false`
- `backend_claim_blockers = [canonical_ssd_mlx_backend_not_wired]`
- `math_fidelity_scope = trainable MLX score-aware Z7 module with reference S6 recurrence`

into the shared MLX bundle used by full training.

## Verification

- `ruff` on touched files: pass
- `pytest src/tac/substrates/_shared/mlx_score_aware/tests/test_bundle.py src/tac/substrates/_shared/mlx_score_aware/tests/test_pr95_faithful_curriculum_substrate_wire_in.py src/tac/tests/test_z7_mamba2_mlx_backend_lineage.py -q`: 36 passed
- `pytest src/tac/substrates/_shared/mlx_score_aware/tests/test_loss_adapter_harness.py src/tac/tests/test_z7_mamba2_mlx_module_smoke.py src/tac/tests/test_z7_mamba2_mlx_backend_lineage.py -q`: 20 passed
- `git diff --check`: pass
- `tools/lane_maturity.py validate`: 1549 lanes clean
- `tools/review_gate_hook.py`: pass

## Remaining MLX Work

This does not claim canonical Mamba2 SSD is wired into Z7. The blocker is now
durable, machine-readable, and carried by long-training artifacts so acquisition
and later archive-bound promotion cannot misread the current Z7 MLX recurrence
as canonical SSD.
