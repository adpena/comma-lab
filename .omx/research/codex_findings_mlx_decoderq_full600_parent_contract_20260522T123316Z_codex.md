# Codex Findings: MLX Full-600 FEC6 And Decoder-Q Parent Contracts

- UTC: 20260522T123316Z
- Lane: mlx_auth_parent_contract_full600_fec6_decoderq
- Evidence grade: macOS-MLX research-signal only
- Score authority: false
- Promotion eligible: false
- Rank or kill eligible: false

## Result

FEC6 and decoder-q now both have strict full-sample local MLX parent contracts
over auth-axis tensor caches. The combined full-600 same-axis dataset is covered
by a two-contract bundle with no parent-plan blockers:

- FEC6 contract:
  `experiments/results/mlx_fec6_auth_parent_response_full600_20260522T1200Z/fec6_auth_parent_contract_strict_v1_full600cal.json`
- Decoder-q contract:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/candidate_parent_contract_strict_v1_full600cal.json`
- Bundle:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/mlx_parent_contract_bundle_full600_fec6_decoderq.json`
- Parent plan:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/parent_production_contract_plan_full600_fec6_decoderq.json`
- Parent plan status: `strict_pass`
- MLX rows covered: `1200`
- Required parent groups: `2`
- Covered parent groups: `2`
- Missing parent groups: `0`
- Blockers: none

This is production-grade local acceleration evidence for candidate generation
and exact-eval spend triage only. It is not a contest score, not promotion
authority, and not rank/kill authority.

## Full-Sample Scores

FEC6 parent:

- Response:
  `experiments/results/mlx_fec6_auth_parent_response_full600_20260522T1200Z/candidate_parent_0000_0600.json`
- Archive SHA-256:
  `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- `canonical_score` `[macOS-MLX research-signal; 600-sample local parent response; non-authoritative]`:
  `0.1920527920355189`
- `avg_posenet_dist`: `2.943360062118927e-05`
- `avg_segnet_dist`: `0.0005602942575933412`

Decoder-q parent:

- Response:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/candidate_parent_0000_0600.json`
- Archive SHA-256:
  `022ac0f391bc9408c357575496c3b680fc5cf9da6ca85d23c3ff994c370a1347`
- `canonical_score` `[macOS-MLX research-signal; 600-sample local parent response; non-authoritative]`:
  `0.1924459939299716`
- `avg_posenet_dist`: `2.9441861554460046e-05`
- `avg_segnet_dist`: `0.0005642022026101282`

Full-parent ordering matches contest-CPU auth eval:

- FEC6 CPU score `[contest-CPU]`: `0.1920513168811056`
- Decoder-q CPU score `[contest-CPU]`: `0.19244523120613244`
- CPU decoder-q minus FEC6 gap: `0.000393914325026834`
- MLX decoder-q minus FEC6 gap `[macOS-MLX research-signal]`:
  `0.00039320189445271603`
- MLX gap understatement versus CPU gap: `7.12430574117975e-07`

Decoder-q is worse than FEC6 on the full parent aggregate on both axes. The
local window dataset still has 170 improved singleton windows, so its useful
role is local window/component triage, not full-archive promotion.

## Calibration

- Calibration:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/candidate_score_calibration_cpu_full600.json`
- Rows: `2`
- MLX/CPU rank inversions: `0`
- Calibration uncertainty score: `1.475154413288493e-06`
- Recommended minimum MLX gap for spend triage:
  `7.375772066442465e-06`
- Certified pairwise decisions: `1`
- Uncertain pairwise decisions: `0`

The observed full-parent MLX gap is much larger than the calibrated minimum
gap, so the FEC6-vs-decoder-q ordering is spend-triage certified. This still
does not make MLX a score axis.

## Parity And Profile

FEC6:

- Candidate parity:
  `experiments/results/mlx_fec6_auth_parent_response_full600_20260522T1200Z/candidate_torch_parity_sweep_cpu_singleton_pairs0_600_argmax1.json`
- Reference parity reused from the full-600 reference sweep:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/reference_torch_parity_sweep_cpu_singleton_pairs0_600.json`
- Profile:
  `experiments/results/mlx_fec6_auth_parent_response_full600_20260522T1200Z/candidate_profile_cpu_singleton_pairs0_600_repeat2.json`
- Profile stability:
  `experiments/results/mlx_fec6_auth_parent_response_full600_20260522T1200Z/candidate_profile_stability_cpu_singleton_pairs0_600_repeat2.json`
- Best CPU singleton throughput: `1.1060096113505102` pairs/s

Decoder-q:

- Candidate parity:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/candidate_torch_parity_sweep_cpu_singleton_pairs0_600.json`
- Reference parity:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/reference_torch_parity_sweep_cpu_singleton_pairs0_600.json`
- Profile stability:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/candidate_profile_stability_cpu_singleton_pairs0_600_repeat2.json`

Parity uses singleton batches. The allowed one-pixel SegNet argmax tolerance
only covers near-tie windows with tiny top-2 margins; it does not relax score
authority or permit batch-shaped MLX rows to affect promotion decisions.

## Dataset

- Dataset:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/candidate_same_axis_window_response_dataset_full600.json`
- Row count: `1200`
- Family counts: `mlx_decoder_q=600`, `mlx_fec6_auth_parent=600`
- Improved decoder-q singleton windows versus FEC6 baseline: `170`
- Best decoder-q singleton delta `[macOS-MLX research-signal]`:
  `-0.0020326847010743165`
- Worst decoder-q singleton delta `[macOS-MLX research-signal]`:
  `0.00304554049762229`

## Next Action

Use this bundle as the local MLX spend-triage gate for auth-cache-backed rows
only. Next high-value work is to turn the 170 improved decoder-q singleton
windows into byte-closed candidate edits, then send only calibrated winners to
contest CPU/CUDA auth eval. Any new family must first pass the same auth-cache
identity, singleton parity, profile stability, full-sample calibration, and
parent-contract bundle gates.
