# HiNeRV epoch7749 full600 MLX replay

Axis: `[macOS-MLX research-signal]`, false-authority only. No CPU/CUDA exact
eval was run and this result is not rank, kill, or promotion authority.

## Candidate

- Family: `hi_nerv`
- Candidate: `hinerv_np600_ld16_ed8_dc16_mi1fi4_hfg_cnx_lg2c4_cx2k7_int7_mixed_ceil178000`
- Checkpoint: `epoch007749_20260604T005341Z`, EMA export
- Archive: `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_full600_pr95_pact_muon_seg16_resume_20260603T231916Z/epoch007749_ema_archive_export_20260604T005500Z_codex/archive.zip`
- Archive SHA-256: `d9191f19a99cf33846821806ed2b64aa0027b25ccd05afa9e45ddc69ad67224e`
- Archive bytes: `121572`
- Receiver proof: passed

## Full-Video MLX Replay

- Candidate cache: `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_epoch7749_full600_mlx_replay_20260604T005700Z/cache`
- Response: `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_epoch7749_full600_mlx_replay_20260604T005700Z/mlx_response_gpu_full600.json`
- Full-video pairs: `600`
- Score: `88.15309413421633`
- Non-rate estimate: `88.07214432956737`
- Average SegNet distance: `0.5045363616943359`
- Average PoseNet distance: `141.5152156194051`
- Rate contribution: `0.08094980464896083`
- CPU spend gate: blocked by `mlx_response_not_within_cpu_spend_band`

## Verdict

This checkpoint is rate-valid and improving versus the earlier epoch2749
full-video MLX replay (`91.57101908167377`), but distortion is still far outside
the plausible exact-gate band. Treat this as a fit failure, not a rate failure:
preserve the tiny archive grammar, increase scorer/SegNet pressure, keep the
pose guard, and use hard-pair sampling rather than spending exact CPU/CUDA.

The feedback row is recorded at
`/Volumes/VertigoDataTier/pact/experiments/results/hinerv_epoch7749_full_video_mlx_feedback_20260604T010100Z/nerv_full_video_mlx_scorer_feedback_row.json`
and consumed by
`.omx/research/nerv_long_training_campaign_plan_full_video_feedback_20260604T010215Z_codex.json`.
