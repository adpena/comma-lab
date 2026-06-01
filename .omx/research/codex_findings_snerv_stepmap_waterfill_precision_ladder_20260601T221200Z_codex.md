# Codex Findings: SNeRV Step-Map Waterfill Precision Ladder

UTC: 2026-06-01T22:12:00Z
Agent: codex:gpt-5
Axis: `[macOS-CPU advisory]` and `[codec-unit:false-authority]`
Lane: `lane_snerv_stepmap_waterfill_precision_ladder_20260601`

## What Landed

Implemented a receiver-visible SNeRV L-infinity step-map precision ladder in
`tac.analysis.snerv_step_map_coder`:

- `constant_log2_fill`: header-only run-length constant maps.
- int2/int4/int8 log2 quantized groups through the existing adaptive packet grammar.
- `fp16_steps_lzma`: fp16-protected maps in the same adaptive packet grammar.
- reverse-waterfill assignment under a target average bits/coefficient budget.

The SNeRV advisory runner now accepts `--step-map-coder-mode waterfill` and
`--step-map-waterfill-bits-per-coeff`, and the synthetic probe reports a
waterfill packet alongside uniform/adaptive packet probes.

## Verification

Focused checks:

```text
.venv/bin/ruff check src/tac/analysis/snerv_step_map_coder.py src/tac/tests/test_snerv_step_map_coder.py src/tac/substrates/snerv_inverse_steg_carrier/advisory.py src/tac/substrates/snerv_inverse_steg_carrier/tests/test_advisory_step_packet.py tools/run_snerv_inverse_steg_advisory.py tools/probe_snerv_step_map_coder.py
All checks passed!

.venv/bin/python -m pytest src/tac/tests/test_snerv_step_map_coder.py src/tac/substrates/snerv_inverse_steg_carrier/tests/test_advisory_step_packet.py -q
16 passed in 0.81s
```

Synthetic codec-unit probe:

- Path: `.omx/research/snerv_step_map_coder_waterfill_probe_20260601T221042Z.json`
- SHA-256: `6947cdb43bcadf97ac4653e5e9faa97e6e36e1831d265c2f1bf6a5b4975ff7fd`
- Waterfill packet: `13048 B`
- Groups: `constant x1`, `int4_bins16 x10`, `int8_bins256 x13`
- Max relative error: `0.232816`
- Score claim: `false`

One-pair scorer-backed advisory:

- Path: `.omx/research/snerv_inverse_steg_advisory_waterfill_20260601T221200Z.json`
- SHA-256: `bca8424b18f5153e0a7c3b0746938219846ac21f740fcdd853c7dec14e6b5d32`
- Receiver archive SHA-256: `e69b773bb5e232ce9e36615145f07fd1fb9d788004c4bf2a388d24a0b999cc37`
- Receiver replay verified: `true`
- Archive bytes: `33754 B`
- L-inf step packet: `7182 B` vs fp32-lzma baseline `11992 B`
- Waterfill groups: `int4_bins16 x3`, `int8_bins256 x3`
- `d_seg_linf`: `0.02264404296875`
- `d_pose_linf`: `2.1390697956085205`
- `score_linf`: `6.911887587116307`
- `score_l2`: `6.970343869808562`
- Beats frontier rate only: `true`

Adjudication:

- Path: `.omx/research/snerv_inverse_steg_advisory_waterfill_adjudication_20260601T221200Z.json`
- SHA-256: `eeccd2225b4ad75fd8bc70f34c909683b723b489daee57151d85c48e5ada3993`
- Classification: `rate_below_frontier_pose_or_seg_destroyed`
- Ready for exact eval dispatch: `false`
- Actionable next code move: `score_aware_stepmap_waterfill_and_decoder_fit_before_packaging`

## Verdict

NO-GO for promotion or exact-eval dispatch.

The waterfill packet is useful: it makes the step-map payload smaller than fp32
and preserves a slight L-inf advantage over the L2 baseline at this operating
point. But the live one-pair advisory still destroys PoseNet and slightly misses
the SegNet preservation ceiling. Rate is not the blocker here; fit/distortion is.

The next useful implementation move is not another post-hoc step-map packet
variant alone. The waterfill allocator must be coupled into score-aware decoder
weight fitting/training so the cheap receiver grammar is fitted to the scorer
surface before packaging.

## Blockers Preserved

- `full_600_pair_receiver_replay_missing`
- `paired_contest_cpu_cuda_auth_eval_missing`
- `not_packaged_as_contest_archive_zip`
- `frontier_comparison_is_rate_only_not_score_authority`
- `score_axis_is_macos_cpu_advisory`

## Next Step

Build the score-aware SNeRV/HiNeRV decoder-fit loop that uses the same
receiver-visible waterfill grammar during training/QAT, then rerun the local
advisory. Only byte-closed full-600 replay plus paired contest CPU/CUDA pass can
promote.
