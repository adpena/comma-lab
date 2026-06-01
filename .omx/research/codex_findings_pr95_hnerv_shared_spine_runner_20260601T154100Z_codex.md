# Codex Findings: PR95/HNeRV Shared Compact-Spine Runner

UTC: 2026-06-01T15:41:00Z
Agent: Codex
Axis: [macOS-MLX research-signal], false authority

## Verdict

PR95/HNeRV is now an executable family in the compact renderer MLX spine runner, not only a checkpoint/report adapter. The runner seeds from the public PR95 archive, trains against real contest-video pairs through the canonical MLX score-aware harness, and exports a PR95-compatible byte-closed archive packet through `tac.local_acceleration.pr95_hnerv_mlx.write_pr95_public_archive_zip`.

## Evidence

Command:

```bash
.venv/bin/python tools/run_compact_renderer_mlx_spine_runner.py \
  --execute-family pr95_hnerv \
  --output-dir /Volumes/VertigoDataTier/pact/compact_pr95_hnerv_mlx_spine_runner_600pair_1ep_codex_v1 \
  --num-pairs 600 \
  --epochs 1 \
  --batch-pairs 1 \
  --learning-rate 0.001 \
  --compact-ema-decay 0.9 \
  --source-video-path upstream/videos/0.mkv \
  --pr95-source-archive experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/archive.zip \
  --overwrite
```

Artifact:

- Report: `/Volumes/VertigoDataTier/pact/compact_pr95_hnerv_mlx_spine_runner_600pair_1ep_codex_v1/compact_renderer_mlx_spine_runner_report.json`
- Archive: `/Volumes/VertigoDataTier/pact/compact_pr95_hnerv_mlx_spine_runner_600pair_1ep_codex_v1/pr95_hnerv_mlx_training/pr95_public_archive.zip`
- Archive bytes: `178370`
- Archive SHA-256: `2954fcbce5ccf2b5e8569ff767e17d1ffecc89a952c56e122c836497dd73f5f6`
- Source archive SHA-256: `e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a`
- Coverage: full 600-pair base comparison coverage
- Per-epoch smoke loss: `0.2243214100599289`

## Authority

The row remains non-promotional:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

Open blockers:

- `receiver_proof_not_executed`
- `full_video_mlx_scorer_replay_not_attached`
- `contest_cpu_cuda_exact_eval_not_executed`

## Integration Finding

The shared compact runner can now execute the PR95/HNeRV archive grammar directly, which is the correct near-term carrier for rate-frontier work: tiny learned decoder plus tiny per-pair latents, rather than explicit Z8-style coefficient fields.

One real implementation gap remains: the public PR95 Stage-8 Muon continuation optimizer is not yet faithfully represented in this shared harness. The runner records `shared_mlx_scoreaware_harness_lacks_stage8_start_epoch_offset` and disables PR95 faithful-curriculum continuation instead of pretending the one-epoch smoke is a true Stage-8 continuation.

## Next Build Slice

Wire receiver proof and full-video MLX scorer replay directly into the compact runner output contract, then port the real PR95 Stage-8 continuation schedule into the same harness so longer runs preserve the public PR95 optimization semantics while still emitting archive-bound packet-spine rows.
