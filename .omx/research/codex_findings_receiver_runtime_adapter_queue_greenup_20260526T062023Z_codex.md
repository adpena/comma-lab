# Codex Findings - Receiver Runtime Adapter Queue Greenup - 2026-05-26T06:20:23Z

## Scope

Adversarial review and implementation pass over receiver-backed materializer
closure, local CPU/MLX targeted-component queue execution, and PR95/HNeRV MLX
reproduction status. All score-bearing artifacts in this memo are local
advisory or research-signal only unless explicitly tagged as contest auth eval.

## Landed Fixes

- `materializer_submission_closure` now copies the proof-backed adapter runtime
  when `runtime_adapter_ready=true`, validates the adapter runtime tree SHA, and
  refreshes closed source-queue runtime metadata. This prevents stale source
  runtimes from silently replacing packet-member/tensor receiver runtimes.
- Packet-member merge and tensor-factorize generated receivers now accept the
  contest-style extracted archive directory and reconstruct both `archive.zip`
  and member files before delegating to source runtime.
- Local auth eval now pins `UV_PYTHON` to the current venv interpreter, fixing
  the uv/CPython-3.14t dependency-resolution failure class.
- Candidate-queue merge now lets incoming adapter-ready rows overwrite stale
  runtime adapter contract fields and uses true-wins semantics for adapter
  booleans.
- Exact-readiness now binds `runtime_adapter_ready=true` rows to proof/row
  adapter runtime tree SHA agreement and fails closed on missing or mismatched
  runtime SHA.
- Tensor-factorize harvest now preserves runtime adapter fields from either the
  manifest or runtime-consumption proof.
- Byte-range chain harvest now preserves `candidate_runtime_tree_sha256` from
  chain runtime steps so exact-readiness can enforce the same runtime tree
  binding on byte-range recode candidates.
- Queue-owned scorer-input hash artifacts now support explicit outside-work-dir
  relative paths from the command cwd. This fixes the signal-loss class where
  `contest_auth_eval.py` wrote hash manifests under `work_dir/.../experiments`
  while queue telemetry expected the shared response path.

## Queue Execution

Queue:
`.omx/research/frontier_rate_attack_feedback_refresh_20260526T053007Z_shared_component_response_dedupe/targeted_component_correction_queue.json`

State:
`.omx/state/experiment_queue_frontier_rate_attack_feedback_materialization_requests_codex_component_correction.sqlite`

Result:

- `14/14` queue steps succeeded.
- Local CPU advisory succeeded on `[macOS-CPU advisory]`, `score_claim=false`,
  canonical score `53.610485624475956`, archive `cbe7d79124ba...`,
  archive bytes `345544`, inflate `80.45s`, evaluate `465.17s`.
- Hash-only scorer-input identity manifest was recovered at the queue-owned path
  with SHA-256 `82e80bea1f557899309c4d37d1c1b80a7267704b208bff45cca83708adba46e7`.
- MLX scorer-input cache built in `5.64s`, `600` pairs, `2.831GB`, audit verdict
  `PASS_CACHE_LOCAL_CPU_ADVISORY_IDENTITY`.
- MLX component response completed in `37.37s` wall / `31.05s` tool elapsed,
  `[macOS-MLX research-signal]`, `score_claim=false`, canonical score
  `53.62813031918261`.
- Five targeted-component correction harvests succeeded in parallel.

Current blocker is now higher-level, not receiver-runtime plumbing:
the harvest rows are blocked from budget spend because paired component deltas
are missing (`paired_reference_local_cpu_advisory_required_for_component_delta`,
`local_cpu_segnet_delta_missing`, `local_mlx_component_delta_missing`,
`measured_component_delta_missing`). This should be resolved by regenerating the
targeted queue with paired reference response as a first-class shared response,
then running candidate/reference CPU and MLX responses through the same
component-delta harvester.

## PR95/HNeRV MLX Status

Read-only sidecar audit found the PR95 lane is real but not yet 1:1 faithful.
Implemented pieces include:

- source intake under
  `experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/`;
- native MLX decoder/export/training scaffolds in
  `src/tac/local_acceleration/pr95_hnerv_mlx.py`,
  `src/tac/local_acceleration/pr95_hnerv_mlx_training.py`, and
  `src/tac/local_acceleration/pr95_hnerv_mlx_long_training.py`;
- export/package tools
  `tools/export_pr95_mlx_to_pytorch_state_dict.py` and
  `tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py`;
- eight-stage local queue spine in
  `.omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z/pr95_mlx_full_profile_queue.json`.

Not yet faithful:

- training remains RGB/RGB+YUV6 MSE local MLX, not PR95 SegNet/PoseNet
  scorer-domain loss with QAT/EMA/stage semantics;
- latest downstream drift artifact is above scorer precision:
  `aggregate_contest_score_drift_units=0.0934699467` in
  `experiments/results/pr95_mlx_full_decoder_downstream_drift_20260526T055636Z/results.json`;
- export parity is not tight enough yet (`pytorch_export_forward_parity_established=false`);
- shell inflate parity proof still lacks the full-frame file-list authority
  claim even when byte comparison passes.

## Remaining Gaps

- `renderer_payload_dfl1_v1` still lacks proof-bound candidate runtime dir/tree
  propagation. Full-frame parity cannot bypass missing runtime adapter binding.
- `byte_range_entropy_recode_v1` has the harvest-side runtime tree propagation
  fix now; it still needs an end-to-end materialized byte-range exact-readiness
  regression with `runtime_adapter_ready=true`.
- The targeted component harvester needs paired reference CPU/MLX responses to
  turn the current response rows into measured component-delta rows.
- PR95 MLX needs real scorer-domain loss closure before long-training results
  can be treated as PR95 reproduction rather than local substrate research.

## Verification

- `py_compile` on changed runtime/optimizer modules passed.
- Focused pytest suite: `17 passed in 1.44s`.
- Targeted queue worker: `14/14` succeeded, no ready steps remain.

## Next Action

Regenerate and execute the paired-reference targeted queue with the same
outside-work-dir scorer hash fix, then promote paired candidate/reference
component deltas into the acquisition model. In parallel, close DFL1
runtime-tree propagation and replace the PR95 MLX RGB/YUV6 loss smoke with a
source-faithful SegNet/PoseNet scorer-loss closure smoke.
