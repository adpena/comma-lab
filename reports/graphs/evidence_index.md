# evidence index

## current exact frontier

- PR100 HNeRV-LC-v2 adapter exact T4 adjudicated score:
  - `experiments/results/lightning_batch/exact_eval_public_pr100_hnerv_lc_v2_adapter_t4_20260504T1213Z/contest_auth_eval.adjudicated.json`
- PR100 adapter release packet:
  - `experiments/results/submission_packet_pr100_adapter_20260504/apogee_pr100_hnerv_lc_v2_adapter`
- PR100 packet strict compliance JSON:
  - `experiments/results/submission_packet_pr100_adapter_20260504/pre_submission_compliance.pr100.json`
- Sanitized public release manifest:
  - `reports/graphs/site/apogee_release_manifest.json`

## superseded exact anchors

- PR99 HNeRV/Muon LC adapter exact T4 adjudicated score:
  - `experiments/results/lightning_batch/exact_eval_public_pr99_hnerv_adapter_t4_20260504T0958Z/contest_auth_eval.adjudicated.json`
- PR95 stem-permutation repack exact T4 adjudicated score:
  - `experiments/results/lightning_batch/exact_eval_pr95_hnerv_muon_stemperm_t4_20260504T0906Z/contest_auth_eval.adjudicated.json`
- PR95 final submission packet:
  - `experiments/results/submission_packet_pr95_stemperm_20260504/apogee_pr95_stemperm`
- PR95 packet strict compliance JSON:
  - `experiments/results/submission_packet_pr95_stemperm_20260504/pre_submission_compliance.json`
- PR95 stem-permutation packing profile:
  - `experiments/results/pr95_hnerv_muon_packing_profile_20260504_codex/archive.pr95_repacked_stemperm.zip`
- PR95 conservative packing profile:
  - `experiments/results/pr95_hnerv_muon_packing_profile_20260504_codex/profile_pr95_hnerv_muon_packing.json`

- PR95 conservative repack exact T4 replay:
  - `experiments/results/lightning_batch/exact_eval_pr95_hnerv_muon_repacked_t4_fix2_20260504T0848Z/contest_auth_eval.adjudicated.json`
- PR95 public archive exact T4 replay:
  - `experiments/results/lightning_batch/exact_eval_public_pr95_hnerv_muon_t4_fix2_20260504T0848Z/contest_auth_eval.adjudicated.json`
- PR85+STBM1BR+PR92/RMB1 exact T4 replay:
  - `experiments/results/lightning_batch/exact_eval_pr85_stbm1br_pr92_rmb1_t4_20260504T082220Z/contest_auth_eval.adjudicated.json`
- PR85+STBM1BR exact T4 replay:
  - `experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z/contest_auth_eval.adjudicated.json`
- PR85 exact T4 replay:
  - `experiments/results/lightning_batch/exact_eval_public_pr85_adaptive_masking_joint_frame_model_t4_20260503T2332Z/contest_auth_eval.adjudicated.json`
- PR84 exact T4 replay:
  - `experiments/results/lightning_batch/exact_eval_public_pr84_adaptive_range_mask_no_router_t4_20260503T214008Z/contest_auth_eval.adjudicated.json`
- PR81 exact T4 replay:
  - `experiments/results/lightning_batch/exact_eval_public_pr81_qzs3_range_mask_t4_depsfix_20260503T195657Z/contest_auth_eval.adjudicated.json`
- C-067 historical exact T4 replay:
  - `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.adjudicated.json`

## external and fail-closed context

- PR91 static anatomy:
  - `experiments/results/public_pr91_intake_20260504_worker/pr91_archive_anatomy.json`
- PR91 static replay preflight:
  - `experiments/results/public_pr91_intake_20260504_codex/public_replay_intake_preflight.json`
- PR91 HPM1 fail-closed ledgers:
  - `.omx/research/pr91_hpm1_first16_contract_probe_20260504_codex.md`
  - `.omx/research/pr91_hpm1_probability_contract_worker_20260504.md`
  - `.omx/research/pr91_hpm1_parity_greenup_20260504_worker.md`
- PR95 exact-vs-body score drift and PR96 static context:
  - `.omx/research/public_pr95_pr96_no_dispatch_intake_20260504_codex.md`
- PR100 adapter replay harvest:
  - `.omx/research/public_hnerv_adapter_replays_20260504_codex.md`
- Harness bug-class ledger:
  - `.omx/research/contest_harness_bug_classes_20260504_codex.md`

## compliance gate

- Pre-submission gate script:
  - `scripts/pre_submission_compliance_check.py`
- Focused gate tests:
  - `src/tac/tests/test_pre_submission_compliance_check.py`
- Public replay intake hardening:
  - `.omx/research/public_replay_exact_eval_hardening_20260503_codex.md`
