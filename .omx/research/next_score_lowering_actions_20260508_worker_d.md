# Next Score-Lowering Actions - Worker D - 2026-05-08

Owner: Worker D
Scope: local implementation scout after PR106 UNIWARD dispatch-readiness work.
GPU dispatches launched: none.

## Decision

The highest-EV unblocked local action after PR106 UNIWARD CPU packet closure is
to advance the cross-paradigm ADMM x continuous-K plus Op1 finalizer runtime
surface, not to launch or duplicate any GPU work.

Reasoning:

- `arch_shrink_x0.4_lightning` is the top score-producing path, but it is
  already running under an active claim. The only safe action this turn was a
  single-shot harvest poll; it returned nonterminal `running`.
- PR106 UNIWARD now has a byte-closed CPU packet and deterministic rebuild
  proof. It still needs an explicit promotion decision plus a fresh claim
  before exact CUDA; Worker D did not launch it.
- ADMM/no-dead-K and apogee_int6 are still gated by exact-negative calibration
  and/or explicit override. Their manifests and readiness ledgers do not
  justify a new local dispatch action.
- PySR/CMAES trajectory integration is planning-only in the current reports:
  the best symbolic fit is weak on held-out data (`test_r2=0.38295`), and the
  best CMA-ES byte point (`162154`) is already near known Op1 saturation
  without archive substitution.
- Filler STC dual-layer masks are useful research but not a score-lowering
  local target yet: the current lossless dual-layer result is `46159` bytes,
  worse than the 5-class entropy lossless baseline `36595` bytes, and the
  manifest still requires score-aware detector costs / GF(q>2) / lossy AV1
  layer work.
- Cross-paradigm ADMM x Op1 already has a byte-closed CPU builder, real runtime
  packet shape, and `cuda_eval_worth_testing=true`. The concrete local action
  was to fix a builder CLI blocker, rebuild a Worker D artifact, and run local
  custody/compliance gates.

## Work Done

### Arch-shrink harvest poll

Single-shot harvester command:

```bash
.venv/bin/python experiments/arch_shrink_x0.4_lightning_harvest.py --job-name arch-shrink-x0-4-lightning-20260508T024304Z --teamspace comma-lab --user adpena --ssh-target s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai --once
```

Result:

- `status=running` at `2026-05-08T08:23:02Z`.
- `--once` exited nonterminal.
- No archive, score JSON, claim mutation, or evidence row was produced.

### PR106 UNIWARD verification

Verification command:

```bash
.venv/bin/python tools/verify_pr106_uniward_runtime_packet_sha256.py
```

Result:

- Canonical archive:
  `experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/archive.zip`
- Bytes: `150511`
- SHA-256:
  `0641b8ac8084b362b80e7c5bbe3c122946a8dbf7843fcbcd9f445aee7a56af7b`
- Rebuilt archive matched byte-identically.
- This remains `[CPU-build]`, not score evidence.

### Cross-paradigm builder blocker fixed

Initial Worker D build command:

```bash
.venv/bin/python tools/build_cross_paradigm_admm_x_op1_finalizer.py --output-root experiments/results/next_score_lowering_worker_d_20260508
```

Initial result:

- Builder successfully constructed the ADMM substrate, Op1 inner blob, CPLX
  decoder section, and PR101 latent/sidecar split.
- It then crashed while logging the archive path:
  `ValueError: 'experiments/results/next_score_lowering_worker_d_20260508/.../archive.zip' is not in the subpath of '/Users/adpena/Projects/pact'`
- Root cause: relative `--output-root` was not resolved before
  `archive_path.relative_to(REPO_ROOT)`.

Patch:

- `tools/build_cross_paradigm_admm_x_op1_finalizer.py`
  - added `_resolve_output_root()` to normalize relative output roots under
    `REPO_ROOT` and preserve absolute roots;
  - used the resolved output root before creating the timestamped build dir;
  - removed two stale unused imports and fixed lint-only f-string issues found
    by ruff.
- `src/tac/tests/test_build_cross_paradigm_admm_x_op1_finalizer.py`
  - added focused tests for repo-relative and absolute output roots.

Focused patch verification:

```bash
.venv/bin/python -m pytest src/tac/tests/test_build_cross_paradigm_admm_x_op1_finalizer.py -q
.venv/bin/python -m ruff check tools/build_cross_paradigm_admm_x_op1_finalizer.py src/tac/tests/test_build_cross_paradigm_admm_x_op1_finalizer.py
git diff --check -- tools/build_cross_paradigm_admm_x_op1_finalizer.py src/tac/tests/test_build_cross_paradigm_admm_x_op1_finalizer.py
```

Results:

- `2 passed`
- `All checks passed!`
- `git diff --check` clean

### Cross-paradigm Worker D artifact built

Successful build command:

```bash
.venv/bin/python tools/build_cross_paradigm_admm_x_op1_finalizer.py --output-root experiments/results/next_score_lowering_worker_d_20260508
```

Output:

- Artifact dir:
  `experiments/results/next_score_lowering_worker_d_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T082555Z/`
- Archive:
  `experiments/results/next_score_lowering_worker_d_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T082555Z/archive.zip`
- Bytes: `153513`
- SHA-256:
  `7bbba307b1432d8d885e22533fdda9ab5cc87a6025510b2d5098084895284897`
- ZIP member: `x`
- Member SHA-256 from strict compliance:
  `f380872cd0faf4467b6f389faca7c83ae5376e4f68b8a484f62c568243ca4180`
- Op1 inner blob: `137348` bytes
- CPLX decoder section: `137419` bytes
- PR101 latent blob: `15387` bytes
- PR101 sidecar blob: `607` bytes
- Local smoke:
  - tensors compared: `28`
  - latent pairs decoded: `600`
  - rel_err vs lossy substrate: `0.0051485448904925995`
  - rel_err vs original fp32: `0.03724457061759168`
  - max per-tensor rel_err vs lossy substrate: `0.012668838519979832`

Artifact manifest says:

- `evidence_grade="[CPU-build]"`
- `score_claim=false`
- `ready_for_exact_eval_dispatch=false`
- `cuda_eval_worth_testing=true`
- blockers:
  - `cpu_build_rel_err_proxy_not_score_evidence`
  - `exact_cuda_auth_eval_not_yet_harvested`
  - `requires_contest_auth_eval_json_before_score_promotion_rank_or_kill`

Local checks:

```bash
bash -n experiments/results/next_score_lowering_worker_d_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T082555Z/submission_dir/inflate.sh
shasum -a 256 experiments/results/next_score_lowering_worker_d_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T082555Z/archive.zip
zipinfo -1 experiments/results/next_score_lowering_worker_d_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T082555Z/archive.zip
```

Results:

- `bash -n` passed.
- SHA matched the build manifest:
  `7bbba307b1432d8d885e22533fdda9ab5cc87a6025510b2d5098084895284897`.
- ZIP contains only `x`.

Strict compliance command:

```bash
.venv/bin/python scripts/pre_submission_compliance_check.py --submission-dir experiments/results/next_score_lowering_worker_d_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T082555Z/submission_dir --archive experiments/results/next_score_lowering_worker_d_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T082555Z/archive.zip --contest-final --strict --expect-single-member x --expected-archive-sha256 7bbba307b1432d8d885e22533fdda9ab5cc87a6025510b2d5098084895284897 --expected-archive-size-bytes 153513 --dispatch-claims-md .omx/state/active_lane_dispatch_claims.md --expected-lane-id cross_paradigm_admm_continuous_k_plus_op1_finalizer --expected-job-id exact_eval_cross_paradigm_admm_x_op1_worker_d_pending
```

Result: failed closed as expected for a CPU-build packet. Positive custody
checks:

- archive exists;
- archive bytes and SHA match expected;
- member `x` is safe, unique, single, and local/central header names match;
- runtime dependency manifest is computable;
- runtime tree SHA:
  `d28c09156f61237a5a20f96881e087519db378c92f775bfec667c7a4f9c1a729`;
- external dependency roots: `[]`;
- repo-local `tac` import closure resolved with no parse errors.

Remaining strict blockers:

- `submission_dir/archive.zip` missing;
- `submission_dir/report.txt` missing;
- `submission_dir/archive_manifest.json` missing;
- `submission_dir/contest_auth_eval.json` missing;
- runtime tree cannot match auth eval until auth eval exists;
- report cannot mention archive bytes/SHA until report exists;
- no terminal dispatch-claim row for the placeholder exact-eval job.

Claim dry-run command:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim --dry-run --lane-id cross_paradigm_admm_continuous_k_plus_op1_finalizer --platform lightning --instance-job-id exact_eval_cross_paradigm_admm_x_op1_worker_d_pending --agent worker_d --status active_exact_eval --notes 'dry-run readiness check only; no dispatch from Worker D'
```

Result:

- Dry-run row would be accepted; no active same-lane conflict found by the
  helper for this lane/job identity.
- No real claim was written.

## Findings

1. Highest-EV unblocked local action completed: cross-paradigm ADMM x Op1 now
   has a Worker D rebuilt byte-closed CPU packet under an isolated artifact
   root, and the builder can accept repo-relative output roots without crashing.
2. The packet is not dispatch-ready and not score evidence. It is exact-eval
   worthy only after explicit promotion/claim workflow because its distortion
   proxy remains CPU-only and lossy coarsening just produced an exact negative
   for a related measured configuration.
3. The next non-GPU improvement, if chosen, is release-packet staging for this
   exact artifact: copy/link archive into `submission_dir`, generate
   `archive_manifest.json` and `report.txt`, then rerun strict compliance.
   Exact CUDA still requires a real Level-2 claim and should not be launched by
   Worker D.
4. PR106 UNIWARD remains ready only in the CPU-build sense. Current
   `.omx/state/active_lane_dispatch_claims.md` still contains a prestage row
   saying archive "NOT YET BUILT"; that row is stale relative to the verified
   `150511` byte packet, but it remains `pending_authorization`, not an active
   dispatch.
5. `arch_shrink_x0.4_lightning` should continue to be harvested, not
   relaunched. The latest Worker D poll found it running and nonterminal.

## Exact Commands Run

```bash
git status --short --branch
rg -n "PR106|UNIWARD|arch_shrink|Lightning harvest|next_score|HNeRV|public frontier" /Users/adpena/.codex/memories/MEMORY.md
pwd && git rev-parse --abbrev-ref HEAD && git rev-parse HEAD
nl -ba /Users/adpena/.codex/memories/MEMORY.md | sed -n '169,228p'
sed -n '1,220p' /Users/adpena/.codex/memories/rollout_summaries/2026-05-01T16-35-40-goJh-public_hnerv_frontier_replay_and_xrepack.md
rg --files .omx/research | rg '2026050[78]|next_score|frontier|pr106|uniward|arch|admm|apogee|pysr|cmaes|filler|stc'
test -e .omx/research/next_score_lowering_actions_20260508_worker_d.md; printf '%s\n' $?
sed -n '1,220p' .omx/research/roadmap_state_reconciliation_20260508_codex.md
sed -n '1,220p' .omx/research/pr106_uniward_runtime_packet_smoke_20260508_codex.md
sed -n '1,220p' .omx/research/frontier_next_dispatch_targets_20260508_worker_f3.md
sed -n '220,520p' .omx/research/frontier_next_dispatch_targets_20260508_worker_f3.md
sed -n '1,240p' .omx/research/arch_shrink_x0_4_lightning_poll_20260508_codex.md
sed -n '1,240p' .omx/research/arch_shrink_x0_4_lightning_review_20260508_worker_a.md
sed -n '1,220p' .omx/research/arch_shrink_lightning_live_strategy_20260508_worker_l3.md
sed -n '220,520p' .omx/research/arch_shrink_lightning_live_strategy_20260508_worker_l3.md
sed -n '1,240p' .omx/research/tier_a_dispatch_readiness_worker_b_20260508.md
sed -n '1,220p' .omx/research/frontier_active_dispatch_harvest_monitor_20260508_worker_a.md
sed -n '220,520p' .omx/research/frontier_active_dispatch_harvest_monitor_20260508_worker_a.md
sed -n '240,520p' .omx/research/tier_a_dispatch_readiness_worker_b_20260508.md
sed -n '1,240p' .omx/research/lossy_coarsening_exact_cuda_adversarial_review_20260508_worker_b.md
jq '.' .omx/research/lossy_coarsening_exact_cuda_result_review_20260508_codex.json
jq '[.[] | select((.lane_id // "") == "arch_shrink_x0.4_lightning" or (.job_name // "") == "arch-shrink-x0-4-lightning-20260508T024304Z") | {lane_id, job_name, submitted_at_utc, terminal_status, machine, status_at_submit: .submit_result.status_at_submit, expected_artifact_dir, expected_auth_eval_json, teamspace: .submit_result.teamspace, user: .submit_result.user}]' .omx/state/lightning_active_jobs.json
rg -n "arch_shrink_x0\\.4_lightning|arch-shrink-x0-4-lightning-20260508T024304Z|lane_apogee_int6|apogee_int6|PR106 UNIWARD|uniward|pysr|CMAES|CMA-ES|STC|dual-layer|dual_layer|filler|cross-paradigm|cross_paradigm" .omx/state/active_lane_dispatch_claims.md .omx/research reports experiments/results tools src/tac -g '!**/.git/**'
find experiments/results -maxdepth 3 \( -iname '*pysr*' -o -iname '*cmaes*' -o -iname '*cma*' -o -iname '*stc*' -o -iname '*dual*layer*' -o -iname '*filler*' -o -iname '*cross*paradigm*' -o -iname '*admm*' -o -iname '*apogee*int6*' -o -iname '*uniward*' \) -print | sort
date -u +%Y-%m-%dT%H:%M:%SZ
.venv/bin/python experiments/arch_shrink_x0.4_lightning_harvest.py --job-name arch-shrink-x0-4-lightning-20260508T024304Z --teamspace comma-lab --user adpena --ssh-target s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai --once
jq 'keys' experiments/results/pr101_pysr_symbolic_regression_20260508T075525Z_canonical_v2/pr101_pysr_symbolic_regression_report.json
jq 'keys' experiments/results/cma_pr101_real_substrate_cmaes_20260507T223229Z/cma_pr101_search_report.json
jq 'keys' reports/raw/pr101_cross_paradigm_hstack_vstack_corrected_20260508/manifest.json
jq 'keys' reports/raw/pr_alpha_mask_dual_layer_stc_empirical_20260508T075353Z/manifest.json
jq 'keys' experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/build_manifest.json
jq '{schema, evidence_grade, ready_for_exact_eval_dispatch, dispatchable, field_selection_ready_for_exact_eval_dispatch, best_equation, best_complexity, n_train, n_test, n_trajectories_unique, metrics, empirical_best, surrogate_optimum, dispatch_blockers, trajectory_sources}' experiments/results/pr101_pysr_symbolic_regression_20260508T075525Z_canonical_v2/pr101_pysr_symbolic_regression_report.json
jq '{schema, optimizer, n_evaluations, n_successful, n_failed, best_eval, pareto_frontier_count, ready_for_exact_eval_dispatch, score_claim, dispatch_blockers, parameter_space}' experiments/results/cma_pr101_real_substrate_cmaes_20260507T223229Z/cma_pr101_search_report.json
jq '{schema, evidence_grade, headline, dominant_stack, cuda_eval_worth_testing, ready_for_exact_eval_dispatch, score_claim, score_affecting_payload_changed, charged_bits_changed, dispatch_blockers, reactivation_criteria_remaining}' reports/raw/pr101_cross_paradigm_hstack_vstack_corrected_20260508/manifest.json
jq '{schema, evidence_grade, headline, n_frames, frame_shape, dual_layer_lossless_roundtrip_verified, dual_layer_compressed_bytes, entropy_coder_5class_lossless_bytes, av1_baseline_bytes, filler_stc_single_layer_lossy_bytes, ratio_vs_5class_entropy, ratio_vs_av1, ready_for_exact_eval_dispatch, cuda_eval_worth_testing, dispatch_blockers, reactivation_criteria_remaining}' reports/raw/pr_alpha_mask_dual_layer_stc_empirical_20260508T075353Z/manifest.json
jq '{lane_id, evidence_grade, archive_relpath, archive_bytes, archive_sha256, archive_byte_savings_vs_pr106_published, achieved_rel_err, achieved_rel_err_smoke_weight_identity, decoder_packed_brotli_bytes, latents_and_sidecar_brotli_bytes, ready_for_exact_eval_dispatch, cuda_eval_worth_testing, dispatch_blockers, score_claim_blockers, wire_format_identity_with_pr106_published, wire_format_diff_vs_pr106}' experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/build_manifest.json
rg --files tools experiments src/tac | rg 'pysr|cma|cmaes|stc|dual|cross_paradigm|hstack|vstack|admm|uniward'
rg -n "PySR|pysr|CMAES|CMA-ES|trajectory|field_selection|filler|dual-layer|dual_layer|STC|cross-paradigm|cross_paradigm|no_inflate_py_for_cross_paradigm_composition" .omx/research/*.md tools experiments src/tac -g '!**/.git/**'
find reports/raw -maxdepth 2 \( -iname '*pysr*' -o -iname '*cma*' -o -iname '*stc*' -o -iname '*dual*' -o -iname '*cross*paradigm*' -o -iname '*hstack*' \) -print | sort
.venv/bin/python tools/pr101_pysr_cmaes_trajectory_symbolic_regression.py --help
.venv/bin/python tools/codec_op_cma_search.py --help
.venv/bin/python tools/pr_alpha_mask_dual_layer_stc_empirical.py --help
.venv/bin/python tools/build_cross_paradigm_admm_x_op1_finalizer.py --help
.venv/bin/python tools/canonical_cross_paradigm_stack_orchestrator.py --help
find experiments/results/cross_paradigm_admm_x_op1_finalizer_20260508T062603Z -maxdepth 2 -type f | sort
jq 'keys' experiments/results/cross_paradigm_admm_x_op1_finalizer_20260508T062603Z/build_manifest.json
jq '{schema, evidence_grade, archive_relpath, archive_bytes, archive_sha256, op1_inner_blob_bytes, inner_blob_sha256, latent_blob_bytes, sidecar_blob_bytes, ready_for_exact_eval_dispatch, cuda_eval_worth_testing, dispatch_blockers, smoke}' experiments/results/cross_paradigm_admm_x_op1_finalizer_20260508T062603Z/build_manifest.json
sed -n '1,260p' .omx/research/monolithic_packet_candidate_bridge_20260508_codex.md
sed -n '1,260p' .omx/research/monolithic_packet_closure_floor_gate_20260508_worker_m1.md
jq '{archive_bytes, archive_sha256, cplx_header_bytes, cplx_decoder_section_bytes, op1_inner_blob_bytes, xparadigm_decoder_only_bytes, admm_proxy_archive_bytes, rel_err_int8_after_admm, rel_err_vs_orig_fp32_smoke, rel_err_vs_lossy_substrate_smoke, max_per_tensor_rel_err_vs_lossy_smoke, smoke_n_latent_pairs_decoded, wire_format, wire_format_doc, ready_for_exact_eval_dispatch, cuda_eval_worth_testing, dispatch_blockers, score_claim_blockers}' experiments/results/cross_paradigm_admm_x_op1_finalizer_20260508T062603Z/build_manifest.json
stat -f '%z %N' experiments/results/cross_paradigm_admm_x_op1_finalizer_20260508T062603Z/archive.zip experiments/results/cross_paradigm_admm_x_op1_finalizer_20260508T062603Z/submission_dir/inflate.sh experiments/results/cross_paradigm_admm_x_op1_finalizer_20260508T062603Z/submission_dir/inflate.py
zipinfo -1 experiments/results/cross_paradigm_admm_x_op1_finalizer_20260508T062603Z/archive.zip
sed -n '1,240p' src/tac/tests/test_canonical_cross_paradigm_stack_orchestrator.py
sed -n '1,260p' src/tac/tests/test_pr101_cross_paradigm_hstack_vstack.py
sed -n '1,260p' tools/build_cross_paradigm_admm_x_op1_finalizer.py
sed -n '260,620p' tools/build_cross_paradigm_admm_x_op1_finalizer.py
sed -n '1,260p' tools/pr101_pysr_cmaes_trajectory_symbolic_regression.py
sed -n '1,260p' tools/pr_alpha_mask_dual_layer_stc_empirical.py
sed -n '620,920p' tools/build_cross_paradigm_admm_x_op1_finalizer.py
.venv/bin/python tools/build_cross_paradigm_admm_x_op1_finalizer.py --output-root experiments/results/next_score_lowering_worker_d_20260508
rg -n "build_cross_paradigm_admm_x_op1_finalizer|cross_paradigm_admm_x_op1" src/tac/tests tools
git status --short -- tools/build_cross_paradigm_admm_x_op1_finalizer.py src/tac/tests
ls -la experiments/results/next_score_lowering_worker_d_20260508
.venv/bin/python -m pytest src/tac/tests/test_build_cross_paradigm_admm_x_op1_finalizer.py -q
git diff -- tools/build_cross_paradigm_admm_x_op1_finalizer.py src/tac/tests/test_build_cross_paradigm_admm_x_op1_finalizer.py
.venv/bin/python -m ruff check --fix tools/build_cross_paradigm_admm_x_op1_finalizer.py src/tac/tests/test_build_cross_paradigm_admm_x_op1_finalizer.py
.venv/bin/python -m pytest src/tac/tests/test_build_cross_paradigm_admm_x_op1_finalizer.py -q
.venv/bin/python -m ruff check tools/build_cross_paradigm_admm_x_op1_finalizer.py src/tac/tests/test_build_cross_paradigm_admm_x_op1_finalizer.py
git diff --check -- tools/build_cross_paradigm_admm_x_op1_finalizer.py src/tac/tests/test_build_cross_paradigm_admm_x_op1_finalizer.py
git diff -- tools/build_cross_paradigm_admm_x_op1_finalizer.py src/tac/tests/test_build_cross_paradigm_admm_x_op1_finalizer.py
.venv/bin/python tools/build_cross_paradigm_admm_x_op1_finalizer.py --output-root experiments/results/next_score_lowering_worker_d_20260508
bash -n experiments/results/next_score_lowering_worker_d_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T082555Z/submission_dir/inflate.sh
shasum -a 256 experiments/results/next_score_lowering_worker_d_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T082555Z/archive.zip
zipinfo -1 experiments/results/next_score_lowering_worker_d_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T082555Z/archive.zip
.venv/bin/python tools/verify_pr106_uniward_runtime_packet_sha256.py
.venv/bin/python -m ruff check tools/build_cross_paradigm_admm_x_op1_finalizer.py src/tac/tests/test_build_cross_paradigm_admm_x_op1_finalizer.py
.venv/bin/python scripts/pre_submission_compliance_check.py --submission-dir experiments/results/next_score_lowering_worker_d_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T082555Z/submission_dir --archive experiments/results/next_score_lowering_worker_d_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T082555Z/archive.zip --contest-final --strict --expect-single-member x --expected-archive-sha256 7bbba307b1432d8d885e22533fdda9ab5cc87a6025510b2d5098084895284897 --expected-archive-size-bytes 153513 --dispatch-claims-md .omx/state/active_lane_dispatch_claims.md --expected-lane-id cross_paradigm_admm_continuous_k_plus_op1_finalizer --expected-job-id exact_eval_cross_paradigm_admm_x_op1_worker_d_pending
.venv/bin/python tools/claim_lane_dispatch.py claim --dry-run --lane-id cross_paradigm_admm_continuous_k_plus_op1_finalizer --platform lightning --instance-job-id exact_eval_cross_paradigm_admm_x_op1_worker_d_pending --agent worker_d --status active_exact_eval --notes 'dry-run readiness check only; no dispatch from Worker D'
find experiments/results/next_score_lowering_worker_d_20260508 -maxdepth 3 -type f | sort
jq '{schema_version, lane_id, archive_relpath, archive_bytes, archive_sha256, cplx_decoder_section_bytes, op1_inner_blob_bytes, rel_err_int8_after_admm, rel_err_vs_lossy_substrate_smoke, rel_err_vs_orig_fp32_smoke, max_per_tensor_rel_err_vs_lossy_smoke, smoke_n_latent_pairs_decoded, ready_for_exact_eval_dispatch, cuda_eval_worth_testing, dispatch_blockers}' experiments/results/next_score_lowering_worker_d_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T082555Z/build_manifest.json
git status --short --branch tools/build_cross_paradigm_admm_x_op1_finalizer.py src/tac/tests/test_build_cross_paradigm_admm_x_op1_finalizer.py experiments/results/next_score_lowering_worker_d_20260508 .omx/research/next_score_lowering_actions_20260508_worker_d.md
rm -rf experiments/results/next_score_lowering_worker_d_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T082500Z
find experiments/results/next_score_lowering_worker_d_20260508 -maxdepth 3 -type f | sort
git status --short --branch tools/build_cross_paradigm_admm_x_op1_finalizer.py src/tac/tests/test_build_cross_paradigm_admm_x_op1_finalizer.py experiments/results/next_score_lowering_worker_d_20260508 .omx/research/next_score_lowering_actions_20260508_worker_d.md
git status --short --branch -- .omx/research/next_score_lowering_actions_20260508_worker_d.md tools/build_cross_paradigm_admm_x_op1_finalizer.py src/tac/tests/test_build_cross_paradigm_admm_x_op1_finalizer.py experiments/results/next_score_lowering_worker_d_20260508
find experiments/results/next_score_lowering_worker_d_20260508 -type f | sort
git diff --check -- .omx/research/next_score_lowering_actions_20260508_worker_d.md tools/build_cross_paradigm_admm_x_op1_finalizer.py src/tac/tests/test_build_cross_paradigm_admm_x_op1_finalizer.py
find experiments/results/next_score_lowering_worker_d_20260508 -type d -name __pycache__ -prune -exec rm -rf {} +
```

## Changed Files / Artifacts

- `.omx/research/next_score_lowering_actions_20260508_worker_d.md`
- `tools/build_cross_paradigm_admm_x_op1_finalizer.py`
- `src/tac/tests/test_build_cross_paradigm_admm_x_op1_finalizer.py`
- `experiments/results/next_score_lowering_worker_d_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T082555Z/archive.zip`
- `experiments/results/next_score_lowering_worker_d_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T082555Z/build_manifest.json`
- `experiments/results/next_score_lowering_worker_d_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T082555Z/submission_dir/inflate.py`
- `experiments/results/next_score_lowering_worker_d_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T082555Z/submission_dir/inflate.sh`
- `experiments/results/next_score_lowering_worker_d_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T082555Z/submission_dir/src/codec.py`
- `experiments/results/next_score_lowering_worker_d_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T082555Z/submission_dir/src/model.py`
- `experiments/results/next_score_lowering_worker_d_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T082555Z/submission_dir/src/tac/__init__.py`
- `experiments/results/next_score_lowering_worker_d_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T082555Z/submission_dir/src/tac/pr101_split_brotli_codec.py`
- `experiments/results/next_score_lowering_worker_d_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T082555Z/submission_dir/src/tac/pr101_split_brotli_codec_derivers.py`
