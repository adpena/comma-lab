# Codex Findings - MLX-First Scorer-Region Loop

Date: 2026-05-31T02:15Z
Agent: Codex
Scope: P18/P19/P11/P15 scorer-region cascade, local CPU/MLX acquisition, artifact retention, PR95 MLX packageability.

## Verdict

The queue-owned Cascade C loop now executes end to end with receiver output proof, local CPU advisory, MLX advisory, scorer-response dataset harvest, exact-readiness bridge, and artifact retention. It did not lower the current CPU frontier. It produced a useful asymmetry signal: the 12-pair MLX slice was positive, while the full 600-pair local CPU advisory was slightly worse.

This candidate must not dispatch exact auth from the current evidence. The durable next action is to use the scorer-response row as acquisition signal for grouped region/pair/operator search, not promote this patch.

## Concrete Evidence

- Queue definition: `.omx/research/scorer_region_selector_chain_full_loop_exec_20260531T0125Z/queue.json`
- Queue state: `.omx/state/experiment_queue_scorer_region_selector_chain_full_loop_exec_20260531T0125Z.sqlite`
- Queue status after repair: 16/16 steps succeeded.
- Output root: `/Volumes/VertigoDataTier/experiments/results/scorer_region_selector_chain_full_loop_exec_20260531T0125Z`

Receiver proof:

- `frame1_region_waterfill_runtime_patch/full_frame_output_change_proof/shell_inflate_output_change.json`
- `output_change_observed=true`
- `raw_shape_preserving_output_change_observed=true`
- `differing_output_count=1`
- `differing_byte_count=139421`
- `blockers=[]`

Local CPU advisory:

- `local_component_spot_check/local_cpu_advisory.json`
- `score_axis=cpu_advisory`
- `canonical_score=0.1920003362662307`
- `n_samples=600`
- `score_claim=false`
- `ready_for_exact_eval_dispatch=false`

MLX advisory:

- `local_component_spot_check/mlx_scorer_response.json`
- `score_axis=[macOS-MLX research-signal]`
- `canonical_score=0.18445031691375097`
- `n_samples=12`
- `score_claim=false`
- `ready_for_exact_eval_dispatch=false`

Scorer-response dataset:

- `local_component_spot_check/scorer_response_dataset.json`
- `best_delta.delta_vs_baseline_score=-0.0001507003870495943`
- `best_byte_budget_margin.byte_budget_margin_vs_break_even=226.3247889706514`

Exact-readiness bridge:

- `scorer_region_exact_ready_bridge_report.json`
- `output_change_proof_proven_count=1`
- `runtime_content_tree_custody_proven_count=1`
- `archive_custody_proven_count=1`
- `ready_for_exact_eval_dispatch=false`
- Remaining blockers include missing full-frame parity, exact auth, and dispatch claim. This is correct fail-closed behavior.

Artifact retention:

- Local component root reduced from about 6.0 GiB to about 468 KiB.
- Deleted certified rebuildable local CPU inflated raw output: 3,662,409,600 bytes.
- Deleted certified rebuildable canonical MLX scorer input cache: 2,831,169,438 bytes.
- Durable journal: `local_component_spot_check/artifact_retention_plan.journal.jsonl`

## Bugs Extincted

1. Receiver patch materialization copied stale `runtime_consumption_proof.json` from the source submission tree. Fixed by ignoring copied stale proof files when producing patched submission trees.
2. The MLX scorer-response queue postcondition checked `schema` even though canonical payloads use `schema_version`. Fixed in the queue builder.
3. Local component retention included local CPU scratch but not the MLX scorer input cache. Fixed in the queue builder.
4. Artifact retention recognized legacy `mlx_delta_cache` but not canonical `mlx_scorer_input_cache` / `reference_mlx_scorer_input_cache`. Fixed in `comma_lab.artifact_retention`.
5. `compact_experiment_artifacts.py` could execute deletion and only then fail while overwriting its output JSON. Fixed by self-capturing the existing output SHA before execution and using guarded overwrite semantics.
6. PR95 MLX long-training checkpoints with external latents were blocked by a package tool that only accepted PyTorch-embedded latents. Added `--latents-npy` so MLX training can remain portable through NumPy artifacts.

## PR95 / HNeRV Control Arm

The 2026-05-30 32-frame MLX checkpoint remains telemetry only and not packageable against the full PR95 source archive because it has 16 latent rows instead of the expected 600. The packageable control-arm target remains the full 600-latent artifact from the older full run. The package tool now supports external `.npy` latents so future MLX long-training checkpoints can package without forcing PyTorch-native latent custody.

## Stack-Of-Stacks Constraint

The predictive-coding stack-of-stacks should compose only validated members. Treat Z8/Z7-Mamba-2/DreamerV3 RSSM/Z6-v2/Z4-style validated predictive-coding pieces as eligible inputs, but keep the falsified Compound C renderer out of candidate construction unless a new artifact independently validates it. This is a composition guard, not a research conclusion.

## Next Acquisition Move

Use the local MLX-positive / full-CPU-negative split as a planner signal:

- expand beyond 12 pairs;
- evaluate grouped PoseNet-null, SegNet-region, RGB/YUV delta, selector-codec, and repack order interactions;
- require full local CPU advisory before exact dispatch;
- keep MLX as fast acquisition, never score authority.
