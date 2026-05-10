# PR95 HNeRV Parity To T1 Trainer Wiring - Codex 2026-05-10

`research_only=false` for local trainer-preflight wiring. `score_claim=false`.
No GPU, remote provider, or exact eval job was run in this pass.

## Scope

Converted the PR95 release-view HNeRV/Muon eight-stage training discipline into
machine-readable T1 trainer evidence:

- source: `experiments/profile_pr95_hnerv_muon_intake.py`
- PR95 release-view tree:
  `experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon`
- trainer consumer:
  `experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py`

The profile now emits `trainer_parity_contract` with schema
`pr95_hnerv_muon_t1_trainer_parity_v1`. The contract records the eight-stage
order, stage hyperparameters, archive section budget, source-tree SHA-256,
stage-schedule digest, required T1 scorer-domain flags, and fail-closed
score-bearing dispatch blockers.

## PR95 Stage Discipline Captured

Release-view stage order:

1. `stage1_v328_ce` - CE, 3000 epochs, AdamW, no QAT, C1a off.
2. `stage2_v331_softplus` - tau-softplus, 5650 epochs, AdamW, no QAT, C1a off.
3. `stage3_v332_smooth` - smooth-disagreement, 1500 epochs, AdamW, no QAT.
4. `stage4_v332_qat` - smooth-disagreement, 500 epochs, QAT on.
5. `stage5_c1a_l7` - L7/C1a, 9000 epochs, lambda 0.01, sigma 0.2.
6. `stage6_lambda_sweep` - L7/C1a, 2000 epochs, lambda 0.02, sigma 0.2.
7. `stage7_sigma_sweep` - L7/C1a, 3000 epochs, lambda 0.02, sigma 0.1.
8. `stage8_muon_finetune` - Muon finetune, 5000 epochs, lambda 0.02, sigma 0.1.

T1 score-domain runs must now carry the contract in:

- `pr95_replication_provenance.json::trainer_parity_profile`
- `provenance.json::pr95_hnerv_trainer_parity`

Non-smoke scorer-domain T1 training refuses a missing or invalid
`--pr95-parity-profile`.

## Six-Hook Wire-In

- Sensitivity map: no new empirical anchor; the profile exposes archive section
  budgets and stage schedule digest for future `tac.sensitivity_map.*` anchors.
- Pareto constraint: non-binding in this local pass because no scored archive
  changed; `score_claim=false` remains explicit.
- Bit allocator: archive section budget is exported as compressed/uncompressed
  bytes plus SHA-256 per PR95 section.
- Cathedral autopilot dispatch: trainer provenance now records the parity
  profile path/status and fail-closed dispatch blockers before any non-smoke
  score-domain run.
- Continual-learning posterior: no posterior update because no empirical
  score anchor was produced.
- Probe-disambiguator: no A/B ambiguity here; the PR95 release-view source is
  the source of truth for stage order. YUV6 routing remains delegated to the
  existing `--yuv6-mode auto` probe path, with `monkey_patch_global` recorded
  as the PR95-faithful required flag.

## Remaining Score-Bearing Blocker

Exact remaining blocker to score-bearing T1/HNeRV parity dispatch:

`active_dispatch_claim_and_provider_job_not_started_no_gpu_or_remote_per_scope`

A promotable run still needs a lane claim, provider job, copied claim ledger,
packet artifact, exact CUDA auth eval JSON, archive/runtime SHA custody, sample
count, logs, and recomputed contest score. This pass intentionally stopped at
local config/manifest/preflight evidence.
