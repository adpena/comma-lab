# Optimizer-guided PR101/A1 score-lowering materialization

generated_at: 2026-05-10T21:05:00Z
scope: local score-lowering plumbing, no remote dispatch, no score claim
evidence_grade: [offline-proxy-planning-only] until exact eval promotion

## What changed

This tranche converted the CMAES/Optuna discussion into a fail-closed,
byte-materialization path:

- `optimizer_guided_candidate_queue_v1` rows now adapt into the canonical
  `optimizer_candidate_queue_v1` planning queue without gaining dispatch
  authority.
- Generic proxy/planning adapters now reapply the full false-authority
  contract after merging untrusted source rows.
- `tools/constrained_coord_search_pr101_bias_sidecar.py` can consume a
  planning queue and materialize bounded A1/PR101 bias/sidecar variants.
- `tools/materialize_kaggle_pr101_proxy_candidate.py` can consume a
  bias-only optimizer queue row and emit the existing PR101 runtime-packet
  handoff schema.

## Generated planning artifacts

- `.omx/research/optimizer_guided_candidate_queues_20260510_codex/pr101_bias_sidecar_cmaes_queue.json`
  - 256 generated, top 32 retained, dispatch_ready=0
- `.omx/research/optimizer_guided_candidate_queues_20260510_codex/pr101_bias_sidecar_optuna_queue.json`
  - 256 generated, top 32 retained, dispatch_ready=0
- `.omx/research/optimizer_guided_candidate_queues_20260510_codex/pr101_bias_refine_cmaes_control_queue.json`
  - 128 generated, top 16 retained, dispatch_ready=0
- `.omx/research/optimizer_guided_candidate_queues_20260510_codex/score_lowering_planning_queue.json`
  - 80 unique planning rows, top 48 retained, dispatch_ready=0
- `.omx/research/optimizer_guided_candidate_queues_20260510_codex/materialized_top8_planning_queue.json`
  - 8 materialized A1 runtime variants, dispatch_ready=0
- `.omx/research/optimizer_guided_candidate_queues_20260510_codex/bias_refine_0127_runtime_packet_planning_queue.json`
  - 1 byte-closed runtime packet planning row, dispatch_ready=0

## Byte-closed local custody produced

Bias-only candidate:

- candidate_id: `bias_refine_cmaes_style_stdlib_0127`
- params:
  - `bias_b=-0.99816723921`
  - `bias_g=-1.00035431724`
  - `bias_r=-0.997347966104`
- packet manifest:
  `experiments/results/optimizer_guided_pr101_bias_refine_0127_packet_20260510_codex/proxy_runtime_packet/runtime_packet_manifest.json`
- runtime-consumption proof:
  `experiments/results/optimizer_guided_pr101_bias_refine_0127_packet_20260510_codex/proxy_runtime_packet/runtime_consumption_proof.json`
- archive bytes: 178258
- archive SHA-256:
  `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`
- runtime tree SHA-256:
  `612b2405112b64489450767d598210be359343de150a27cb937e3c268c89476d`
- runtime consumption:
  `runtime_consumption_proven_for_supported_bias_params=true`
- exact readiness:
  `ready_for_exact_eval_dispatch=false`

## Adversarial classification

This is not a score result. It is a local byte-closed packet/proof artifact
that can be promoted only after a separate exact-readiness gate and lane claim.

Sidecar rows remain blocked because existing PR101 runtime-packet tooling does
not yet consume `sidecar_f1_r -> up[:, 1, 0]`. The queue-to-A1 materializer can
write sidecar inflate variants, but exact CUDA evidence from neighboring
sidecar-like packet work is negative enough that the immediate score-lowering
path is bias-only custody first, not broad sidecar dispatch.

## Verification

- `.venv/bin/python -m pytest tests/test_constrained_coord_search_pr101_bias_sidecar.py src/tac/tests/test_optimizer_candidate_queue.py src/tac/tests/test_optimizer_guided_candidate_generation.py`
  - 37 passed
- `.venv/bin/python -m pytest tests/test_materialize_kaggle_pr101_proxy_candidate.py tests/test_build_pr101_kaggle_proxy_runtime_packet.py tests/test_prove_pr101_kaggle_proxy_runtime_consumption.py`
  - 30 passed
- `.venv/bin/python tools/audit_exact_ready_queues.py --format json --suppression-manifest .omx/research/exact_ready_queue_retraction_manifest_20260510_codex.json --warn-only`
  - passed; raw stale ready rows=5, suppressed=5, unsuppressed stale ready rows=0
- `.venv/bin/python tools/all_lanes_preflight.py --jobs 8 --timings --timings-json .omx/research/artifacts/preflight_dev_timing_optimizer_guided_score_lowering_20260510_codex.json`
  - wall=2.30s
  - failed only because new research queue JSONs were not yet tracked and the
    local `experiments/results/` runtime-source baseline had not yet been
    refreshed; follow-up commit tracks these ledgers and refreshes the baseline.

## Next gate

Do not dispatch this packet while T1 is active or without an explicit lane
claim. If T1 remains pending and the operator wants a low-cost exact test, the
next valid path is:

1. promote `bias_refine_0127_runtime_packet_planning_queue.json` through the
   exact-readiness gate from live custody;
2. claim `pr101_kaggle_proxy_runtime_packet_exact_eval` with a distinct job id;
3. run contest-CPU only if no same-lane claim conflict exists;
4. run contest-CUDA only if CPU-positive and operator policy allows it.
