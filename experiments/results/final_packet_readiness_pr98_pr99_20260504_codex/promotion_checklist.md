# PR98/PR99 Final-Packet Readiness

This directory is local readiness only. It does not make a score claim and it did not dispatch GPU work.

## Promotion Steps

For whichever PR98/PR99 T4 exact eval wins:

1. Harvest the terminal Lightning job through `scripts/launch_lightning_batch_job.py harvest-ssh`.
2. Require `contest_auth_eval.adjudicated.json` with CUDA, 600 samples, promotion eligibility, archive SHA/bytes matching this manifest, and runtime tree SHA matching the selected sanitized runtime.
3. Close the original and duplicate dispatch claims with terminal rows; stop the redundant duplicate if still pending/running.
4. Build the public packet with the selected exact archive copy plus the selected sanitized runtime snapshot, preserving the runtime root name used by exact eval or recording the new runtime tree hash explicitly.
5. Add `report.txt` only from the adjudicated JSON values: exact score, SegNet, PoseNet, archive bytes, archive SHA, runtime tree SHA, hardware, sample count, and eval command.
6. Run `scripts/pre_submission_compliance_check.py` on the exact packet surface with `--require-auth-eval --require-t4-equivalent --require-submission-runtime-match --expect-single-member 0.bin --require-report-archive-link --require-report-auth-score-link --source-prs PR98,PR99` plus the selected expected archive SHA/bytes, expected runtime tree SHA, selected lane/job, and adjudicated JSON.
7. Run public-release hygiene on the publish surface; do not include `.omx/state`, provider logs, local absolute paths, secrets, pycache, hidden files, or raw private manifests.

## Candidate Snapshots

### PR98 `ready`

- archive: `experiments/results/final_packet_readiness_pr98_pr99_20260504_codex/archives/pr98_archive.zip`
- archive bytes: `178392`
- archive sha256: `7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb`
- sanitized runtime: `experiments/results/final_packet_readiness_pr98_pr99_20260504_codex/runtime_snapshots/pr98_runtime`
- runtime tree sha256: `4d71b5769e9c886e8a4e1be8997014ec47fe5d5ce5519619bf16bff0ae7f2738`
- expected runtime tree sha256: `4d71b5769e9c886e8a4e1be8997014ec47fe5d5ce5519619bf16bff0ae7f2738`
- original T4 job: `exact_eval_public_pr98_hnerv_muon_finetuned_t4_20260504T0940Z`
- duplicate T4 job: `exact_eval_public_pr98_hnerv_muon_finetuned_t4_dup_20260504T0944Z`
- local blockers: `[]`

### PR99 `ready`

- archive: `experiments/results/final_packet_readiness_pr98_pr99_20260504_codex/archives/pr99_archive.zip`
- archive bytes: `178546`
- archive sha256: `278b1c7a1bd6b03a5bceddafcb3489b2624c558ad22825d9211b701333b6eefb`
- sanitized runtime: `experiments/results/final_packet_readiness_pr98_pr99_20260504_codex/runtime_snapshots/pr99_runtime`
- runtime tree sha256: `67fa8ef36f732be73d29053bc050a86a597b23d394ea07538451a8eb8303817f`
- expected runtime tree sha256: `67fa8ef36f732be73d29053bc050a86a597b23d394ea07538451a8eb8303817f`
- original T4 job: `exact_eval_public_pr99_hnerv_muon_lc_t4_20260504T0940Z`
- duplicate T4 job: `exact_eval_public_pr99_hnerv_muon_lc_t4_dup_20260504T0944Z`
- local blockers: `[]`

