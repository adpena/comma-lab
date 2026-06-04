# HiNeRV sparse hard-pair top16 successor launch

Date: 2026-06-04T01:44:41Z  
Author: codex  
Authority: `[macOS-MLX research-signal]`, no score claim

## Launch

Started tmux session:

`hinerv_sparse_hardpair_top16_20260604T014441Z`

Output:

`/Volumes/VertigoDataTier/pact/experiments/results/hinerv_full600_pr95_pact_muon_sparse_hardpair_top16_resume_20260604T014441Z`

Log:

`/Volumes/VertigoDataTier/pact/experiments/results/launch_logs/hinerv_sparse_hardpair_top16_20260604T014441Z/run.log`

Resume checkpoint:

`/Volumes/VertigoDataTier/pact/experiments/results/hinerv_full600_pr95_pact_muon_seg16_resume_20260603T231916Z/hi_nerv_mlx_training/checkpoints/epoch011499_20260604T013350Z.meta.json`

Hitlist:

`/Volumes/VertigoDataTier/pact/experiments/results/hinerv_epoch11499_full600_mlx_replay_20260604T013738Z/hard_pair_hitlist_top16pct.json`

The successor uses the sparse hard-pair target hydration path landed in commit
`8db7aca81`, so prioritized source pairs hydrate only the requested hard-pair
targets while the model still represents the full 600-pair video.

## Input Evidence

Epoch 11,499 receiver-proven export:

- archive bytes: 121,315
- archive SHA-256: `02b40bf305613df5ee6ae44167108c6415011383508831c3be440cc84c98fdff`
- rate profile: latents 64,984 B, decoder weights 30,083 B

Epoch 11,499 full-video MLX replay:

- score: 87.16221122257681
- SegNet distortion: 0.5048249476154645
- PoseNet distortion: 133.94822467803954
- exact CPU/CUDA gate: blocked, `mlx_response_not_within_cpu_spend_band`

Hard-pair repair remains fit-axis work, not a promotion candidate. The expected
next proof is a new checkpoint/export/full-video replay showing whether sparse
hard-pair training reduces SegNet/PoseNet distortion without leaving the hard
byte ceilings.

