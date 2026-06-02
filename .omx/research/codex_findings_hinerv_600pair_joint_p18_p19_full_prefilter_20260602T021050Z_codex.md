# Codex Findings: HiNeRV 600-Pair Joint P18/P19 Full Prefilter

UTC: 2026-06-02T02:10:50Z

Scope: full-video MLX-first HiNeRV smoke with verified joint P18/P19 recon-weight surface, coder-aware QAT, byte-closed archive export, receiver proof, and MLX prefilter gate.

## Inputs

- Runner output: `/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_torch_joint_recon_weight_600pair_8epoch_20260602T015041Z`
- Recon-weight NPZ: `/Volumes/VertigoDataTier/pact/experiments/results/codex_joint_recon_weight_torch_600pair_20260602T012553Z/joint_p18_p19_recon_pixel_weight.npz`
- Recon-weight SHA-256: `b964a6ee4e5b2d847faf30b66c541a973eb5bf416b2f86d2236a511b3674eb0b`
- Gradient health: `pass_finite`, 1050 scorer components, 0 non-finite values.

## Result

The campaign produced a byte-closed HiNeRV archive and a full-video MLX prefilter profile, but it is not locally promising:

- Archive bytes: `72828`
- Archive SHA-256: `7e42e57829b77a01252aeebbf6e752e59f5a761fceb38e6d619f35c5e5ffb50a`
- Full-video MLX advisory score: `90.46398954065153`
- SegNet term: `50.48243627448876`
- Pose term: `39.93306009032479`
- Rate term: `0.048493175837981496`

The runner correctly refused local CPU replay and exact auth promotion:

- `local_cpu_replay_blocked_by_mlx_prefilter_score`
- `mlx_prefilter_score_not_below_local_replay_threshold`
- `mlx_score_above_hard_demote_threshold`
- `contest_cpu_cuda_exact_eval_not_executed`

This is `[macOS-MLX research-signal]` only. It is not a contest score and not a kill claim for HiNeRV as a family; it is a demotion of this 8-epoch, compact-width, full-video-smoke configuration.

## Operational Repair

A duplicate identical 600-pair process was detected and the older copy was terminated after writing:

`/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_torch_joint_recon_weight_600pair_8epoch_20260602T014851Z/superseded_duplicate_run_manifest.json`

The runner has been hardened with an active-campaign lock keyed by normalized campaign args excluding `output_dir`, so future identical expensive MLX/replay campaigns fail fast unless `--allow-duplicate-campaign` is explicit.

## Interpretation

HiNeRV training/export/archive plumbing is now real enough to demote bad configurations quickly. The training itself is not the bottleneck at this scale; the expensive tail is full-video scorer profiling. The score failure is distortion, not rate: the archive is small, but both SegNet and PoseNet remain unusable under this smoke.

## Next Engineering Consequence

Do not spend CPU/CUDA exact auth on this candidate. The next score-lowering work should move upstream into:

- longer staged score-aware training with real PoseNet protection;
- high-resolution/protected pose pathway or flow-conditioned pose channel;
- SNeRV/HiNeRV native rate-aware training with the same full-video MLX prefilter gate;
- faster full-video scorer/profile execution, because that is now the practical loop bottleneck.
