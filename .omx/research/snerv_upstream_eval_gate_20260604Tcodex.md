# SNeRV Upstream Eval Gate - 2026-06-04 Codex

Axis: `[upstream-cpu:false-authority]`. No promotion, rank, kill, or exact
CUDA/CPU authority is claimed.

## Pipeline Landing

This landing is not an ad hoc terminal run. The reusable gate is wired through:

- `src/comma_lab/evaluate.py::evaluate_external_submission_dir`
- `python -m comma_lab.cli eval-external-submission`
- `tools/run_snerv_upstream_eval_gate.py`
- `src/tac/tests/test_snerv_upstream_eval_gate.py`

The gate consumes an already materialized submission directory, runs upstream
`evaluate.sh`, writes stdout/stderr/report artifacts, hashes inflated output,
and deletes success-only raw output only after a rebuild certificate exists in
the gate JSON.

## Source Bundle

- Bundle JSON:
  `/Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/snerv_upstream_submission_bundle.json`
- Submission dir:
  `/Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/submission`
- Data-only archive:
  `/Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/submission/archive.zip`
- Archive bytes: `51694`
- Archive SHA-256:
  `2f57653c2e21834b731cb102a77d1fa603198f7c9ab13c0b82d94f3ad1f42ee2`

## Gate Result

- Gate JSON:
  `/Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/snerv_upstream_eval_gate.json`
- Upstream report:
  `/Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/upstream_eval_gate_20260604Tcodex/report.txt`
- Return code: `0`
- Wall seconds: `736.8017280829954`
- Score reported by upstream CPU evaluator: `90.61`
- PoseNet distortion: `162.09104919`
- SegNet distortion: `0.50314105`
- Rate: `0.00137684`
- Submission bytes charged by upstream `evaluate.py`: `51694`
- Inflated output bytes hashed before cleanup: `3662409600`
- Inflated output tree SHA-256:
  `c02b88fa5e431ba853d854bcaba109fc0c03913b441cf4184107ab7cd9f9eb2e`
- Inflated output retained: `False`
- Cleanup status: `deleted_after_success_with_manifest_certificate`
- Python environment: current Pact venv, recorded with
  `require_upstream_venv=false`; upstream scorer imports were probed before the
  gate.

## Blockers

- `paired_contest_cpu_cuda_auth_eval_missing`
- `pre_submission_compliance_gate_missing`
- `snerv_upstream_eval_gate_score_bad` in the derived planner feedback row

This result proves the upstream-shaped data-only archive is measured by
`archive.zip` size and evaluator-runnable, but it also confirms the current
SNeRV representation is scorer-bad. The byte win is real; the artifact is not a
frontier or launch candidate.

## Planner Feedback Wiring

- Candidate-feedback row:
  `/Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/upstream_eval_gate_20260604Tcodex/snerv_upstream_eval_candidate_feedback_row.json`
- Schema: `nerv_candidate_feedback_row.v1`
- Feedback kind: `upstream_eval_gate`
- Measured archive bytes carried to planner: `51694`
- Planner smoke:
  `.omx/research/nerv_long_training_campaign_plan_20260604Tupstream_feedback_gateblocked_codex.json`
- Queue smoke:
  `.omx/research/nerv_long_training_campaign_queue_20260604Tupstream_feedback_gateblocked_codex.json`

The feedback row is consumed through `--auto-candidate-feedback-root`. The
regenerated planner keeps the SNeRV row context-only with
`feedback_match_scope=family_upstream_eval_gate_context`, sets queue status to
`disabled`, and records `queue_launch_blockers=["snerv_upstream_eval_gate_score_bad"]`.

## Runnable Commands

Generic pipeline entrypoint:

```bash
uv run python -m comma_lab.cli eval-external-submission \
  --submission-dir /Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/submission \
  --upstream-root /Users/adpena/Projects/pact/upstream \
  --artifact-dir /Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/upstream_eval_gate_20260604Tcodex \
  --output-json /Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/snerv_upstream_eval_gate.json \
  --device cpu \
  --no-require-upstream-venv
```

SNeRV bundle-aware entrypoint:

```bash
uv run python tools/run_snerv_upstream_eval_gate.py \
  --bundle-json /Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/snerv_upstream_submission_bundle.json \
  --artifact-dir /Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/upstream_eval_gate_20260604Tcodex \
  --output-json /Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/snerv_upstream_eval_gate.json \
  --device cpu \
  --no-require-upstream-venv \
  --min-free-bytes 5000000000
```

Feedback harvest and auto-discovered planner smoke:

```bash
uv run python tools/harvest_snerv_upstream_eval_gate_feedback.py \
  --gate-json /Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/snerv_upstream_eval_gate.json \
  --output-json /Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/upstream_eval_gate_20260604Tcodex/snerv_upstream_eval_candidate_feedback_row.json

uv run python tools/build_nerv_long_training_campaign_plan.py \
  --hinerv-modelsize-budget /Volumes/VertigoDataTier/pact/experiments/results/nerv_modelsize_budget_scalar_skippriced_20260604T041430Z_codex/hinerv_modelsize_budget.json \
  --snerv-modelsize-budget /Volumes/VertigoDataTier/pact/experiments/results/nerv_modelsize_budget_scalar_skippriced_20260604T041430Z_codex/snerv_modelsize_budget.json \
  --auto-candidate-feedback-root /Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex \
  --output-json .omx/research/nerv_long_training_campaign_plan_20260604Tupstream_feedback_gateblocked_codex.json \
  --output-md .omx/research/nerv_long_training_campaign_plan_20260604Tupstream_feedback_gateblocked_codex.md \
  --output-queue .omx/research/nerv_long_training_campaign_queue_20260604Tupstream_feedback_gateblocked_codex.json \
  --output-snerv-lf-reroute-queue .omx/research/snerv_lf_over_ceiling_reroute_queue_20260604Tupstream_feedback_gateblocked_codex.json \
  --max-candidates-per-family 1
```
