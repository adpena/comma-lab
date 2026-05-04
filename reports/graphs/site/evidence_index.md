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
- PR85 exact T4 replay:
  - `experiments/results/lightning_batch/exact_eval_public_pr85_adaptive_masking_joint_frame_model_t4_20260503T2332Z/contest_auth_eval.adjudicated.json`
- C067 historical exact T4 replay:
  - `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.adjudicated.json`
- Renderer-era controls:
  - `experiments/results/lane_g_v3_landed/contest_auth_eval.json`

## external and fail-closed context

- PR100 public-source attribution:
  - `https://github.com/commaai/comma_video_compression_challenge/pull/100`
- Apogee public submission context:
  - `https://github.com/commaai/comma_video_compression_challenge/pull/107`
- PR91 HPM1 anatomy remains external until exact replay parity exists.
- PR96 and public PR body scores remain external unless exact CUDA replay lands
  for the same archive bytes and runtime tree.

## boundaries

- Only exact CUDA auth eval on `archive.zip -> inflate.sh -> upstream/evaluate.py`
  can rank a result.
- Public comments, body scores, leaderboard displays, static archive anatomy,
  GIFs, and roadmap probes are context until exact replay validates charged
  bytes and runtime custody.
- Raw provider logs, account metadata, local absolute paths, raw `.omx/state`,
  and secrets do not belong in the public bundle.
