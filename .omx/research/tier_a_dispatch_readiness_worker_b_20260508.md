# Tier-A Dispatch Readiness - Worker B 2026-05-08

Author: Worker B
Repo: `/Users/adpena/Projects/pact`
Branch: `main`
Scope: PR106 UNIWARD-Lagrangian, Path B step 6 ADMM/no-dead-K, apogee_int6, active arch_shrink harvest.

No GPU job was launched in this pass. Existing worktree state was dirty before
this pass; unrelated edits and experiment artifacts were left untouched.

## Summary Verdict

| Candidate | Status | Reason |
|---|---|---|
| PR106 UNIWARD-Lagrangian | NO-GO | CPU sweep only; no byte-closed archive or runtime decoder for modified PR106 stream. |
| Path B step 6 ADMM/no-dead-K | NO-GO | Archive/runtime exists and wrapper bug is fixed, but manifest still requires apogee_int6 contest-CUDA anchor first; no auth eval/report/archive manifest/terminal claim. |
| apogee_int6 | NO-GO as score-lowering dispatch | Archive/runtime integrity is good and executable bit is fixed, but current predispatch sanity refuses normal dispatch; parity evidence is calibration-only. |
| arch_shrink harvest | HARVEST-BLOCKED / NO NEW DISPATCH | Active claim exists for `arch-shrink-x0-4-lightning-20260508T024304Z`; local shell lacks Lightning query env. Previous harvested job failed before auth eval. |

## Commands Run

```bash
git status --short --branch
rg -n "PR106|UNIWARD|Lagrangian|Path B|step 6|no-dead-K|ADMM|apogee_int6|arch_shrink|arch-shrink|Tier-A|Tier A" .omx experiments reports reverse_engineering scripts tools src/tac src/comma_lab -g '!**/.git/**'
find experiments/results -maxdepth 3 \( -iname '*pr106*' -o -iname '*uniward*' -o -iname '*lagrangian*' -o -iname '*admm*' -o -iname '*dead*k*' -o -iname '*apogee*int6*' -o -iname '*arch*shrink*' -o -iname '*lossy*coarsening*' \) -print | sort
shasum -a 256 experiments/results/unified_winners_stack_20260508T071803Z/stage_1_2_uniward_no_dead_k/archive.zip experiments/results/unified_winners_stack_20260508T071803Z/stage_1_2_3_uniward_no_dead_k_op1_finalizer/archive.zip experiments/results/admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T064711Z/archive.zip experiments/results/apogee_int6_repack_20260504_claude/apogee_int6_archive.zip experiments/results/lightning_batch/arch-shrink-x0-4-lightning-20260508T020205Z/archive_masks_seed.zip experiments/results/hnerv_arch_shrink_training_driver_stage_d_20260507_codex/generated_schema_codec.hngs
zipinfo -1 experiments/results/unified_winners_stack_20260508T071803Z/stage_1_2_uniward_no_dead_k/archive.zip
zipinfo -1 experiments/results/unified_winners_stack_20260508T071803Z/stage_1_2_3_uniward_no_dead_k_op1_finalizer/archive.zip
zipinfo -1 experiments/results/admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T064711Z/archive.zip
zipinfo -1 experiments/results/apogee_int6_repack_20260504_claude/apogee_int6_archive.zip
.venv/bin/python -m pytest src/tac/tests/test_pr101_unified_winners_stack_inflate_sh.py src/tac/tests/test_build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py src/tac/tests/test_dispatch_readiness_apogee_int6.py -q
.venv/bin/python -m pytest src/tac/tests/test_pr106_omega_opt_lagrangian_per_tensor_allocation.py -q
bash -n experiments/results/unified_winners_stack_20260508T071803Z/stage_1_2_uniward_no_dead_k/submission_dir/inflate.sh experiments/results/unified_winners_stack_20260508T071803Z/stage_1_2_3_uniward_no_dead_k_op1_finalizer/submission_dir/inflate.sh experiments/results/admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T064711Z/submission_dir/inflate.sh submissions/apogee_intN/inflate.sh
.venv/bin/python tools/check_dispatch_cli_shell_hazards.py --strict
.venv/bin/python tools/dispatch_readiness_apogee_int6.py --json
.venv/bin/python tools/predispatch_sanity.py --archive experiments/results/apogee_int6_repack_20260504_claude/apogee_int6_archive.zip --predicted-low 0.190 --predicted-high 0.204 --rel-err-pct 1.55 --lane-class apogee_intN --distortion-proxy-ran --readiness-evidence-json experiments/results/apogee_int6_basin_parity_20260507_claude/parity_evidence.json --json
.venv/bin/python experiments/arch_shrink_x0.4_lightning_harvest.py --once --job-name arch-shrink-x0-4-lightning-20260508T024304Z
```

Exact strict-compliance commands:

```bash
.venv/bin/python scripts/pre_submission_compliance_check.py --submission-dir experiments/results/unified_winners_stack_20260508T071803Z/stage_1_2_uniward_no_dead_k/submission_dir --archive experiments/results/unified_winners_stack_20260508T071803Z/stage_1_2_uniward_no_dead_k/archive.zip --contest-final --strict --expect-single-member x --expected-archive-sha256 fc539f935641e049f5cae443af930fbdbbec703103439abc45b2abf3a602ed13 --expected-archive-size-bytes 148378 --dispatch-claims-md .omx/state/active_lane_dispatch_claims.md --expected-lane-id lane_unified_winners_stack --expected-job-id exact_eval_unified_winners_stage12_pending
.venv/bin/python scripts/pre_submission_compliance_check.py --submission-dir experiments/results/unified_winners_stack_20260508T071803Z/stage_1_2_3_uniward_no_dead_k_op1_finalizer/submission_dir --archive experiments/results/unified_winners_stack_20260508T071803Z/stage_1_2_3_uniward_no_dead_k_op1_finalizer/archive.zip --contest-final --strict --expect-single-member x --expected-archive-sha256 45dd64d41ede9ec6dd74c82572996228face37f6184dd3ab5aa96aad7405ec06 --expected-archive-size-bytes 148494 --dispatch-claims-md .omx/state/active_lane_dispatch_claims.md --expected-lane-id lane_unified_winners_stack --expected-job-id exact_eval_unified_winners_stage123_pending
.venv/bin/python scripts/pre_submission_compliance_check.py --submission-dir experiments/results/admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T064711Z/submission_dir --archive experiments/results/admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T064711Z/archive.zip --contest-final --strict --expect-single-member x --expected-archive-sha256 b7b09089e852872bd67b4b8aa04c1b4d46168bb89343acff81796c5551d63d05 --expected-archive-size-bytes 153671 --dispatch-claims-md .omx/state/active_lane_dispatch_claims.md --expected-lane-id admm_x_lossy_coarsening_path_b_step6_no_dead_k --expected-job-id exact_eval_admm_no_dead_k_pending
.venv/bin/python scripts/pre_submission_compliance_check.py --submission-dir submissions/apogee_intN --archive experiments/results/apogee_int6_repack_20260504_claude/apogee_int6_archive.zip --contest-final --strict --expect-single-member 0.bin --expected-archive-sha256 0176a2691a4daf5991170404d30a304ae30389621c0fc54914628414aef39ff1 --expected-archive-size-bytes 170450 --dispatch-claims-md .omx/state/active_lane_dispatch_claims.md --expected-lane-id lane_apogee_int6 --expected-job-id claude_apogee_int6_lightning_retry5_121354Z
```

Exact claim dry-run commands:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim --dry-run --lane-id lane_unified_winners_stack --platform lightning --instance-job-id exact_eval_unified_winners_stage12_20260508TBD --agent worker_b --status active_exact_eval --notes 'dry-run readiness check only; no dispatch'
.venv/bin/python tools/claim_lane_dispatch.py claim --dry-run --lane-id admm_x_lossy_coarsening_path_b_step6_no_dead_k --platform lightning --instance-job-id exact_eval_admm_no_dead_k_20260508TBD --agent worker_b --status active_exact_eval --notes 'dry-run readiness check only; no dispatch'
.venv/bin/python tools/claim_lane_dispatch.py claim --dry-run --lane-id lane_apogee_int6 --platform lightning --instance-job-id exact_eval_apogee_int6_20260508TBD --agent worker_b --status active_exact_eval --notes 'dry-run readiness check only; no dispatch'
.venv/bin/python tools/claim_lane_dispatch.py claim --dry-run --lane-id arch_shrink_x0.4_lightning --platform lightning --instance-job-id exact_eval_arch_shrink_x0_4_20260508TBD --agent worker_b --status active_dispatching --notes 'dry-run readiness check only; no dispatch'
```

Verification results:

- `pytest` focused Tier-A/runtime checks: `16 passed`.
- `pytest` PR106 Lagrangian/UNIWARD CPU sweep checks: `9 passed`.
- `bash -n` candidate runtime wrappers: passed.
- `tools/check_dispatch_cli_shell_hazards.py --strict`: passed.
- `pre_submission_compliance_check.py --contest-final --strict`: failed for expected missing release/eval surfaces, while confirming archive bytes/SHA/member safety and runtime manifest computation.

## PR106 UNIWARD-Lagrangian

Actual artifact found:

- Manifest: `reports/raw/pr106_lagrangian_per_tensor_allocation_20260508T071433Z/manifest.json`
- Manifest SHA-256: `5eac8aecd9da657c826ee5fbc12e9803ce3ba2b01ac7b417f77d21904e1d2dad`
- Input archive: `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip`
- Input archive bytes: `186239`
- Input archive SHA-256: `3fefbe5dfdd738179a55ca5c995ff8f63ec2755662d60684706f20d313913f58`
- ZIP member: `0.bin`

Manifest facts:

- Evidence grade: `[CPU-prep faithful PR106-cross-substrate-test]`
- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `cuda_eval_worth_testing=false`
- Best UNIWARD vs uniform saving: `5599` bytes at `rms_target=0.02`
- Best overall proxy archive size: `123767` bytes at `rms_target=0.1`

Blockers:

- `byte_rel_err_proxy_only_no_score_test`
- `no_runtime_dequantize_path_built_for_modified_decoder`
- `missing_exact_cuda_auth_eval`
- `scale_per_tensor_fixed_at_lossless_during_K_sweep`
- `no_iterative_primal_dual_ADMM_consensus`
- `lossless_latents_and_sidecar_assumed_constant_no_joint_optimization`

Verdict: NO-GO. This is a useful PR106 substrate signal but not a dispatch
packet. Next step is to build a byte-closed PR106 runtime packet that consumes
the modified decoder stream, smoke it locally, write a build manifest, then
rerun strict pre-submission compliance before any lane claim.

## Path B Step 6 ADMM/No-Dead-K

Actual artifact found:

- Archive: `experiments/results/admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T064711Z/archive.zip`
- Archive bytes: `153671`
- Archive SHA-256: `b7b09089e852872bd67b4b8aa04c1b4d46168bb89343acff81796c5551d63d05`
- ZIP member: `x`
- Member SHA-256: `c9a31946641b452a99f2dc4d03426e0cb40c53fdc33512ced9e856a95be78319`
- Build manifest: `experiments/results/admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T064711Z/build_manifest.json`
- Submission dir: `experiments/results/admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T064711Z/submission_dir`

Manifest/runtime facts:

- Evidence grade: `[CPU-build]`
- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `cuda_eval_worth_testing=true`
- `section_K_bytes_in_wire_format=0`
- `rel_err_actual_int8=0.04153796782863332`
- `rel_err_actual_fp32_smoke=0.03616819695131534`
- Local smoke decoded `600` latent pairs.
- Strict compliance computed runtime tree SHA-256: `dd4cc11083f74b8c196ee98f685fa6d91aec0d6aabe6dc740ec41bca808e01af`

Fixes applied in this pass:

- Hardened `tools/build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py` fallback `inflate.sh` generator so future fallback runtime uses the canonical `inflate.sh <data_dir> <output_dir> <file_list>` contract.
- Added a regression test for that fallback contract.

Strict compliance blockers:

- `submission_dir/archive.zip` missing.
- `submission_dir/report.txt` missing.
- `submission_dir/archive_manifest.json` missing.
- `submission_dir/contest_auth_eval.json` missing.
- Runtime tree cannot match auth eval until auth eval exists.
- No terminal dispatch claim row for the placeholder exact-eval job.
- Build manifest still lists `apogee_int6_contest_cuda_anchor_required_first`.

Claim dry-run:

- `lane_id=admm_x_lossy_coarsening_path_b_step6_no_dead_k` would be accepted by the claim helper in dry-run mode.

Verdict: NO-GO for score dispatch. The archive/runtime surface is materially
closer after the wrapper hardening, but exact eval should wait for either the
apogee_int6 contest-CUDA calibration anchor or an explicit operator override,
then a complete release packet plus lane claim.

## Unified Winners Stack / PR101 UNIWARD + No-Dead-K

Actual artifacts found:

- Stage 1+2 archive: `experiments/results/unified_winners_stack_20260508T071803Z/stage_1_2_uniward_no_dead_k/archive.zip`
- Stage 1+2 bytes/SHA-256: `148378`, `fc539f935641e049f5cae443af930fbdbbec703103439abc45b2abf3a602ed13`
- Stage 1+2 member SHA-256: `f0b279bb288ea40eb068b9a83c78a1cd0ff31f4445c7e968d9008a022cfc0cdd`
- Stage 1+2+3 archive: `experiments/results/unified_winners_stack_20260508T071803Z/stage_1_2_3_uniward_no_dead_k_op1_finalizer/archive.zip`
- Stage 1+2+3 bytes/SHA-256: `148494`, `45dd64d41ede9ec6dd74c82572996228face37f6184dd3ab5aa96aad7405ec06`
- Stage 1+2+3 member SHA-256: `20f0ffee64a7ef71ff2ec128b287230cfb20de8a24aacf7f1e849bf6e045fa39`
- Build manifest: `experiments/results/unified_winners_stack_20260508T071803Z/build_manifest.json`

Manifest facts:

- Best archive path: Stage 1+2 at `148378` bytes.
- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `cuda_eval_worth_testing=true`
- Dispatch blockers include exact CUDA missing, CPU byte proxy only, variance proxy, no iterative primal-dual ADMM consensus, and no score-aware detector loop.

Fixes applied in this pass:

- Updated both generated unified submission `inflate.sh` files from the stale two-arg wrapper to the tested canonical three-arg contest wrapper.
- Runtime wrapper SHA-256 for both generated wrappers after fix: `1bc5139e82e2cefdf73fa67f8f1193048a7e1c9329de018d51814b43fed489bc`

Strict compliance blockers:

- `submission_dir/archive.zip` missing.
- `submission_dir/report.txt` missing.
- `submission_dir/archive_manifest.json` missing.
- `submission_dir/contest_auth_eval.json` missing.
- Runtime tree cannot match auth eval until auth eval exists.
- No terminal dispatch claim row for placeholder exact-eval jobs.

Claim dry-run:

- `lane_id=lane_unified_winners_stack` would be accepted by the claim helper in dry-run mode.

Verdict: NO-GO for score dispatch. Treat as a byte-closed CPU candidate that
needs release packet staging, exact auth eval, and adversarial review. Stage
1+2 is the smaller archive; Stage 1+2+3 is byte-worse by `116` bytes and should
not be first exact-eval spend unless a non-byte rationale is documented.

## apogee_int6

Actual artifacts found:

- Archive: `experiments/results/apogee_int6_repack_20260504_claude/apogee_int6_archive.zip`
- Archive bytes: `170450`
- Archive SHA-256: `0176a2691a4daf5991170404d30a304ae30389621c0fc54914628414aef39ff1`
- ZIP member: `0.bin`
- Member SHA-256: `4bcb81864af2e50a6366adb0c1e9c0846a0ab31f33350c1c80f3c5f7503e3424`
- Repack metadata: `experiments/results/apogee_int6_repack_20260504_claude/repack_metadata.json`
- Parity evidence: `experiments/results/apogee_int6_basin_parity_20260507_claude/parity_evidence.json`
- Runtime: `submissions/apogee_intN`

Runtime proof:

- `submissions/apogee_intN/inflate.sh` syntax check passed.
- `inflate.sh` mode fixed from `0644` to `0755`.
- Runtime tree SHA-256 from strict compliance: `e1caf818820891fb8811e018e6cb8e7aeea799f77555bc5ac27bec99556d4251`

Predispatch results:

- `tools/dispatch_readiness_apogee_int6.py --json` returned nonzero with archive integrity PASS but `all_ok=false`.
- The exact `tools/predispatch_sanity.py` command listed above returned `64`.
- Current refusal reasons:
  - `predicted_high=0.2040` is below the SHA-tied rate-distortion floor `0.3067`.
  - `apogee_intN` requires `contest_cuda_exact_eval_positive` for normal pass; scorer-basin parity is calibration-only and requires explicit override.

Strict compliance blockers:

- `submissions/apogee_intN/archive.zip` missing.
- `submissions/apogee_intN/report.txt` missing.
- `submissions/apogee_intN/archive_manifest.json` missing.
- `submissions/apogee_intN/contest_auth_eval.json` missing.
- Runtime tree cannot match auth eval until auth eval exists.

Claim status:

- Dry-run claim for `lane_apogee_int6` would be accepted.
- Existing real attempts are terminal refusals in `.omx/state/active_lane_dispatch_claims.md`, including Lightning T4 capacity refusal and Vast.ai credit refusal.

Verdict: NO-GO as normal score-lowering dispatch. It can only proceed as a
calibration exact-eval if the main agent/operator explicitly records an
override and then files a fresh claim before launch.

## active arch_shrink harvest

Artifacts/state found:

- Active jobs row: `.omx/state/lightning_active_jobs.json`, `job_name=arch-shrink-x0-4-lightning-20260508T024304Z`
- Expected archive path: `experiments/results/lightning_batch/arch-shrink-x0-4-lightning-20260508T024304Z/archive.zip`
- Expected auth eval: `experiments/results/lightning_batch/arch-shrink-x0-4-lightning-20260508T024304Z/contest_auth_eval.json`
- Local files currently present for active job: `source_manifest.json` only.
- Active claim: `lane_id=arch_shrink_x0.4_lightning`, `job=arch-shrink-x0-4-lightning-20260508T024304Z`, status `active_dispatching`.

Earlier harvested failure:

- Job `arch-shrink-x0-4-lightning-20260508T020205Z` has local logs and no auth eval.
- `train.log` fails before auth eval with:
  `RuntimeError: Q-FAITHFUL forward requires an explicit deployed pose tensor. Zero-pose fallback is forbidden because it trains a different FiLM contract than inflate/eval.`
- Existing terminal claim classifies it as `failed_no_auth_eval_json`.

Current harvest attempt:

```bash
.venv/bin/python experiments/arch_shrink_x0.4_lightning_harvest.py --once --job-name arch-shrink-x0-4-lightning-20260508T024304Z
```

Result:

```text
FATAL: missing required Lightning provider values:
  - --teamspace / $LIGHTNING_TEAMSPACE
  - --user / $LIGHTNING_USER
```

Claim dry-run:

- A new dry-run claim for `arch_shrink_x0.4_lightning` was refused with rc=3 because the active claim already exists.

Verdict: HARVEST-BLOCKED, NO NEW DISPATCH. The only safe next action is for
the coordinating main agent or the owning `claude_lab` claimant to run the
single-shot harvester from an environment with `LIGHTNING_TEAMSPACE` and
`LIGHTNING_USER` set, or to provide those values explicitly. Do not file a new
claim or launch another arch_shrink job while the active claim is open.

## Precise Next Dispatch Steps

1. PR106 UNIWARD-Lagrangian: build a real PR106 runtime packet first. Required
   output: archive path, archive bytes/SHA-256, runtime dir, local smoke proof,
   build manifest, and strict compliance report. Only then claim a lane and
   consider exact eval.
2. ADMM/no-dead-K: do not dispatch before apogee calibration lands or operator
   override is recorded. If approved, stage `archive.zip`, `report.txt`, and
   `archive_manifest.json` into a release packet, claim
   `admm_x_lossy_coarsening_path_b_step6_no_dead_k`, then run exact CUDA auth eval.
3. Unified winners Stage 1+2: if chosen over ADMM, stage the fixed runtime plus
   `archive.zip`, `report.txt`, and `archive_manifest.json`, claim
   `lane_unified_winners_stack`, then exact eval the `148378` byte Stage 1+2
   archive before the byte-worse Stage 1+2+3 archive.
4. apogee_int6: normal dispatch remains blocked. For calibration-only exact
   eval, the main agent must record an explicit override, create/verify the
   release packet, claim `lane_apogee_int6`, and then launch exact eval.
5. arch_shrink: harvest only. Resolve the missing Lightning env and run the
   harvester for `arch-shrink-x0-4-lightning-20260508T024304Z`; do not launch a
   new job while the active claim remains open.

## Supersession note - PR106 UNIWARD CPU-build closure

After this readiness pass, the coordinating agent built the missing PR106
UNIWARD runtime packet with:

```bash
.venv/bin/python tools/build_pr106_uniward_runtime_packet.py \
  --rms-target 0.05 \
  --output-dir experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke
```

Result:

- Archive: `experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/archive.zip`
- Archive bytes: `150511`
- Archive SHA-256:
  `0641b8ac8084b362b80e7c5bbe3c122946a8dbf7843fcbcd9f445aee7a56af7b`
- PR106 decoder smoke: `28` tensors, `600` latent pairs, `1200` implied frames
- Weight identity rel_err through PR106 decoder: `1.852e-08`
- Deterministic rebuild verifier:
  `.venv/bin/python tools/verify_pr106_uniward_runtime_packet_sha256.py`
  passed byte-identical rebuild.

The PR106 UNIWARD verdict is therefore updated from `NO-GO: no byte-closed
archive/runtime` to `CPU-build packet exists, still non-promotable`.

Remaining blockers before any exact CUDA auth eval:

- Fresh dispatch claim for the exact packet.
- Explicit operator/main-agent promotion decision because evidence grade is
  still `[CPU-build]`.
- Canonical exact CUDA auth eval on the exact archive bytes.
