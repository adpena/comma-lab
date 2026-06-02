# Codex Findings: Upstream Eval + Compact VQ Pivot Audit

UTC: 2026-06-02T03:35:00Z
Agent: Codex
Axis: [macOS-MLX research-signal], false authority

## Finding

The terrible PACT-NeRV-VQ result is now treated as executable routing signal,
not only as a chat/memo observation.

Verified upstream scorer contract:

- `evaluate.py`: score is `100*d_seg + sqrt(10*d_pose) + 25*rate`.
- `modules.py`: SegNet scores only the last frame of each pair at 384x512.
- `modules.py` + `frame_utils.py`: PoseNet scores both frames through YUV6 at
  384x512 and compares the first six pose dimensions.
- `evaluate.py`: only `archive.zip` bytes are charged for rate.

Online primary-source check:

- RT-NeRV/VQ-NeRV direction is residual tokenization of shallow/inter-frame
  features plus residual-aware codebook learning and utilization repair:
  https://arxiv.org/abs/2403.12401
- HiNeRV direction is a higher-capacity hierarchical INR plus a pruning and
  quantization codec pipeline: https://arxiv.org/abs/2306.09818
- SNeRV direction is frequency split/DWT LF carriage with HF restoration:
  https://arxiv.org/abs/2501.01681

The current local PACT/VQ implementation is not that RT/VQ object. It is a
primary per-pair single-vector VQ latent carrier:

- `src/tac/substrates/pact_nerv_vq/architecture.py`: `latents` has shape
  `cfg.num_pairs x cfg.latent_dim`.
- `src/tac/substrates/pact_nerv_vq/archive.py`: archive ships one codebook and
  one index stream for `num_pairs`.
- No residual tokenizer, no shallow/inter-frame residual feature path, and no
  codebook-utilization repair are present.

## Live Artifact

Audit report:

- `/Volumes/VertigoDataTier/pact/hprc_projection_gap_repairs/pact_nerv_vq_pact_nerv_vq_pvq_projection_gap_aa80443421cfbcb2/capacity_l16_e32_k64_ch48/compact_vq_pivot_audit_20260602T033436Z.json`

Bounded runner plan with audit attached:

- `/Volumes/VertigoDataTier/pact/hprc_projection_gap_repairs/pact_nerv_vq_pact_nerv_vq_pvq_projection_gap_aa80443421cfbcb2/capacity_l16_e32_k64_ch48/hprc_spine_bounded_runner_plan_with_compact_vq_pivot_20260602T033436Z.json`

Key audit values:

- `verdict`: `pivot_or_rebuild_vq_before_more_long_run_spend`
- `best_full_video_mlx_score`: `90.66201548013069`
- `spend_recommendation`:
  `route_compact_training_budget_to_pr95_hinerv_snerv_stage8_or_rebuild_vq_as_rt_residual_token_bolton`
- blockers include:
  - `compact_vq_is_per_pair_latent_not_residual_tokenization`
  - `compact_vq_shallow_interframe_feature_path_missing`
  - `compact_vq_codebook_utilization_repair_missing`
  - `full_video_mlx_score_above_local_replay_threshold`

## Code Landed

New canonical audit:

- `src/tac/analysis/compact_vq_pivot_audit.py`
- `tools/audit_compact_vq_pivot.py`
- `src/tac/tests/test_compact_vq_pivot_audit.py`

Bounded runner integration:

- `src/tac/substrates/hprc/spine_bounded_runner.py`
- `tools/build_hprc_spine_bounded_runner.py`
- `src/tac/substrates/hprc/tests/test_spine_bounded_runner.py`

The bounded runner now consumes `compact_vq_pivot_audit.v1`, emits
`compact_vq_pivot_signal_rows`, marks matching rows
`demoted_by_compact_vq_pivot_audit`, records posterior demotion hooks, and
keeps false authority intact.

## Verification

```bash
/Users/adpena/Projects/pact/.venv/bin/python -m ruff check \
  src/tac/analysis/compact_vq_pivot_audit.py \
  tools/audit_compact_vq_pivot.py \
  src/tac/substrates/hprc/spine_bounded_runner.py \
  tools/build_hprc_spine_bounded_runner.py \
  src/tac/tests/test_compact_vq_pivot_audit.py \
  src/tac/substrates/hprc/tests/test_spine_bounded_runner.py
```

Passed.

```bash
/Users/adpena/Projects/pact/.venv/bin/python -m pytest \
  src/tac/tests/test_compact_vq_pivot_audit.py \
  src/tac/substrates/hprc/tests/test_spine_bounded_runner.py -q
```

`13 passed`.

## Next Work

Immediate score-lowering pivot:

1. Do not spend more long-run budget on current per-pair-latent PACT/VQ as a
   primary carrier.
2. Put PR95 Stage-8 faithful continuation, HiNeRV, and SNeRV score-aware
   decoder-weight fitting ahead of current PACT/VQ.
3. Re-admit VQ only as an RT/VQ-style residual-token bolt-on with residual
   tokenizer, shallow/inter-frame feature path, codebook-utilization repair,
   receiver-proofed archive bytes, and full-video scorer value-per-byte.
