# HiNeRV hard-pair top16 successor launch

Axis: `[macOS-MLX research-signal]`, false-authority training run. This is not
an exact CPU/CUDA eval and is not promotion authority.

## Launch

- tmux session: `hinerv_hardpair_top16_20260604T012800Z`
- PID at launch sanity check: `83595`
- Output root: `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_full600_pr95_pact_muon_hardpair_top16_resume_20260604T012800Z`
- Log: `/Volumes/VertigoDataTier/pact/experiments/results/launch_logs/hinerv_hardpair_top16_20260604T012800Z/run.log`
- Resume checkpoint: `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_full600_pr95_pact_muon_seg16_resume_20260603T231916Z/hi_nerv_mlx_training/checkpoints/epoch008999_20260604T010957Z.meta.json`
- Hard-pair file: `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_epoch7749_full600_mlx_replay_20260604T005700Z/hard_pair_hitlist_top16pct_v2.json`
- Top-K hard pairs: `96`
- Training target: total `30000` epochs
- Checkpoint interval: `250`

## Rationale

HiNeRV epoch7749 was receiver-proofed at `121572` archive bytes and improved
full-video MLX score to `88.15309413421633`, but remained distortion-bound. The
corrected feedback row preserved the actual observed SegNet pressure
(`16.0`) and removed the bogus lower-weight recommendation. The live blocker is
therefore fit concentration on scorer-marginal hard pairs, not rate or a blind
SegNet-weight bump.

## Contract

The successor keeps the compact HiNeRV archive grammar and consumes pair-row
indices `0..599`. Source frame pairs from the scorer cache are provenance only.
Harvest checkpoints through receiver-proof export, direct MLX cache materialize,
full-video MLX replay, CPU spend gate, and exact gate only if local evidence
becomes plausibly frontier-moving.
