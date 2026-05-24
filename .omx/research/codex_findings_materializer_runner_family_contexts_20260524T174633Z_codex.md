# Codex Findings: Materializer Runner Family Contexts

UTC: 2026-05-24T17:46:33Z
Lane: codex_materializer_runner_family_contexts_20260524
Mode: implementation landing, local-only, false-authority preserved

## Verdict

The remaining dirty materializer-runner tranche is coherent and should land as
the next executable bridge after the inverse-action operation-set compiler. It
does not claim score authority and it does not dispatch paid compute. It makes
the queue-owned final-byte path less leaf-manual by letting the runner generate
family-agnostic materializer contexts for archive-section recoding, packet
member recompression, and tensor factorization, while hardening inverse-scorer
cell candidates against invalid template archives.

## What Landed

- `tools/run_byte_shaving_materializer_campaign.py` can now auto-generate
  artifact maps for `archive_section_entropy_recode_v1`,
  `packet_member_recompress_v1`, and `tensor_factorize_v1`, not only inverse
  scorer cell candidates.
- `src/comma_lab/scheduler/final_byte_operation_contexts.py` carries common
  materializer controls into generated contexts:
  `runtime_consumption_proof`, expected-output hashes, `min_free_bytes`,
  overwrite controls, and size-regression controls.
- `src/comma_lab/scheduler/byte_shaving_campaign_queue.py` now validates
  inverse-scorer `candidate_archive_template` as a strict single-member ZIP
  before emitting work rows, wraps family-agnostic materializer commands with
  stronger receiver/candidate postconditions, and keeps exact-readiness
  follow-up local/fail-closed. Non-harvestable local candidate manifests now
  skip exact-readiness follow-up with a typed reason instead of crashing queue
  compilation.
- `tools/run_byte_shaving_materializer_campaign.py` keeps generated
  materializer outputs inside the scheduler workload root. If the run directory
  already lives under the chosen root, outputs stay per-run; otherwise they
  fall back under the declared workload root so storage preflight stays
  fail-closed.
- `src/comma_lab/scheduler/experiment_queue.py` adds
  `required_nonempty_unless_true` to JSON completion contracts so family
  receiver manifests can require explicit blockers unless the receiver contract
  is actually satisfied.
- `src/tac/hnerv_lowlevel_packer.py` converts malformed ZIP templates into the
  local packer error type, so invalid candidate templates fail through the
  normal materializer failure/reporting path.

## Authority Boundary

All new queue, context, runner, and manifest surfaces remain advisory/local:
`score_claim=false`, `promotion_eligible=false`,
`rank_or_kill_eligible=false`, and `ready_for_exact_eval_dispatch=false`.
Dispatch blockers still require exact auth eval before score claims. MLX/local
or materializer proof rows remain candidate-generation and custody signals only.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_experiment_queue.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_inverse_scorer_cell_materializer.py src/tac/tests/test_materializer_chain_harvest_scheduler.py -q`
  - Result: 235 passed, 1 existing duplicate-ZIP warning.
- `.venv/bin/python -m ruff check src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/comma_lab/scheduler/experiment_queue.py src/comma_lab/scheduler/final_byte_operation_contexts.py src/tac/hnerv_lowlevel_packer.py tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_inverse_scorer_cell_materializer.py`
  - Result: all checks passed.
- `git diff --check -- <changed-files>`
  - Result: clean.
- `.venv/bin/python tools/lane_maturity.py validate`
  - Result: 1258 lanes validated cleanly after lane registration.
- Bounded no-paid VertigoDataTier smoke:
  - Run: `/Volumes/VertigoDataTier/pact/work/materializer_smokes/ias1_runtime_policy_20260524T174535Z/materializer_campaign_run.json`
  - Result: storage preflight, proactive cleanup, runtime-policy queue, and
    local materializer step all succeeded; worker success count 3, failure
    count 0.
  - Candidate archive:
    `/Volumes/VertigoDataTier/pact/work/materializer_smokes/ias1_runtime_policy_20260524T174535Z/materializer_outputs/materializer_work_materializer_work_queue_required_inverse_scorer_cell_candidate_v1_scorer_inverse_surface_cell_materialize_inverse_scorer_cell_candidate_inverse_scorer_cell_candidate_adapter.zip`
    bytes `180375`, sha256
    `b672630c8a21e420160aabea5569292cf72f234260ed8692347613bef8902b78`.
  - Exact-readiness follow-up was intentionally skipped with
    `materializer_manifest_not_harvestable_for_exact_readiness`; the manifest
    keeps `score_claim=false` and `ready_for_exact_eval_dispatch=false`.

## Remaining Gap

This landing makes more operation families queue-consumable, but it is not the
full inverse-steganalysis optimizer. The next gap is still the closed campaign
circuit: action-functional / MLX / public-byte evidence -> operation sets ->
materializer queue -> parity/exact-readiness -> exact eval harvest -> canonical
response update -> replan.
