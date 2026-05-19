# Codex Findings: HF Jobs BUILD_1 Dispatch Custody

timestamp_utc: 2026-05-19T17:37:12Z
actor: codex_session_019de465
task_id: codex_routing_directive_comprehensive_wire_in_and_integration_pass_20260519T072000Z::BUILD_1
status: blocked_provider_credit_after_clean_dispatch_path

## Findings

BUILD_1 is now engineered through the canonical `operator_authorize.py --target hf_jobs` path:

- strict dispatch protocol passes for the HF Jobs recipe;
- strict local predeploy passes with explicit `hf_jobs_research_surrogate` authority;
- Codex pre-dispatch adversarial review returned `approve`;
- lane claim lifecycle closed correctly after provider failure;
- HF Jobs ledger now records a pre-launch `intent` row and a terminal `failed` row for pre-job-id failures.

The real dispatch reached Hugging Face Jobs but failed before returning an `hf_jobs_id`:

- provider error: `402 Payment Required`;
- cause: prepaid credit balance insufficient;
- no remote job id was created;
- no score claim, promotion claim, or CUDA authority was emitted.

## Hardened Bug Classes

1. Multiline HF Jobs JSON output no longer hides `hf_jobs_id`; `operator_authorize.py` parses full pretty JSON and falls back to canonical ledger lookup.
2. `hf_jobs` is native in the standalone dispatch protocol checker via the canonical native-platform set.
3. Advisory HF Jobs surrogate training uses an explicit `hf_jobs_research_surrogate` dispatch kind, not generic `tool` authority.
4. The SegNet surrogate trainer consumes a pinned HF dataset revision from the recipe `hub_dataset_sha`.
5. The dispatcher writes a recoverable intent row before `run_uv_job`.
6. If HF rejects before job id, the dispatcher appends a terminal failed outcome for `pending:<label>` and returns structured failure instead of an orphan traceback.

## Evidence

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/tests/test_operator_authorize_hf_jobs_dispatch.py src/tac/tests/test_dispatch_protocol_complete.py src/tac/tests/test_dispatch_protocol_tool_scope.py src/tac/tests/test_dispatch_hf_jobs_vision_training.py src/tac/tests/test_hf_jobs_segnet_surrogate_image_level_recipe.py -q -p no:cacheprovider`
  - result: `68 passed`
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/check_dispatch_protocol_complete.py --recipe .omx/operator_authorize_recipes/substrate_hf_jobs_segnet_surrogate_distillation_t4_dispatch.yaml --strict`
  - result: `dispatch_protocol_complete=true`
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/local_pre_deploy_check.py --trainer experiments/hf_jobs_segnet_surrogate_distillation.py --recipe substrate_hf_jobs_segnet_surrogate_distillation_t4_dispatch --strict`
  - result: `ALL 9 CHECKS PASSED`
- Real wrapper run:
  - `operator_authorize.py --recipe substrate_hf_jobs_segnet_surrogate_distillation_t4_dispatch --target hf_jobs --yes`
  - pre-dispatch review: `verdict=approve`
  - provider outcome: `402 Payment Required`, insufficient prepaid credits
  - claim: `failed_dispatch_rc_1`

## Next Action

Re-run the exact same `operator_authorize.py` command after Hugging Face Jobs prepaid credits are replenished. No code/design blocker remains for BUILD_1.
