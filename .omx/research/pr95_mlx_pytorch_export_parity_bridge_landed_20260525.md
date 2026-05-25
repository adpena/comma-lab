# PR95 MLX PyTorch Export Parity Bridge

Generated: 2026-05-25
Lane: `lane_pr95_mlx_pytorch_export_parity_bridge_20260525`
Evidence grade: `[macOS-MLX research-signal]`

## Verdict

The export bridge is now archive-based, not random-init based. It parses a
PR95/HNeRV single-member public archive packet (`0.bin`), writes the decoded
state dict to PyTorch `.pt`, and proves forward parity against the public PR95
`HNeRVDecoder` through the existing canonical MLX helpers.

This remains local implementation evidence only:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

## Canonical Tool

Tool:

```bash
tools/export_pr95_mlx_to_pytorch_state_dict.py
```

The tool delegates to:

- `tac.local_acceleration.pr95_hnerv_mlx.parse_pr95_public_archive_zip`
- `tac.local_acceleration.pr95_hnerv_mlx.write_pr95_public_archive_pytorch_export_forward_parity`
- `tac.local_acceleration.pr95_hnerv_mlx.compare_pr95_public_archive_forward_with_pytorch`

It intentionally avoids synthetic random-init checkpoint semantics. The archive
packet is the source of truth.

## Smoke Receipt

Command:

```bash
.venv/bin/python tools/export_pr95_mlx_to_pytorch_state_dict.py \
  --archive-zip .omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z/matrix/stage8/pr95_stage8_muon_adamw_mlx/seed17_c36_0666bb51ac1f/pr95_public_archive.zip \
  --output-pytorch-state-dict .omx/tmp/pr95_export_bridge_smoke/stage8.pt \
  --report-out .omx/tmp/pr95_export_bridge_smoke/stage8_report.json \
  --sample-indices 0 \
  --mlx-device cpu \
  --require-pass
```

Result:

- `parity_passed=true`
- `max_abs=3.051758e-05`
- `mean_abs=3.217933e-06`
- exported `.pt` path: `.omx/tmp/pr95_export_bridge_smoke/stage8.pt`

## Regression

```bash
.venv/bin/python -m pytest src/tac/tests/test_pr95_mlx_pytorch_export_bridge.py -q
```

Result: `1 passed`.

The test builds a tiny deterministic PR95 MLX archive packet, exports it through
the bridge, proves forward parity against the public PR95 decoder, and asserts
all false-authority fields remain false.

## Remaining Blockers

The bridge closes the archive-to-PyTorch export proof, but does not establish:

1. full-frame inflate parity against the original PR95 runtime;
2. source-faithful PR95 training loss parity, including SegNet/PoseNet losses,
   EMA, resume chaining, C1a, QAT, and schedule semantics;
3. paired contest CPU/CUDA auth eval;
4. score, promotion, rank/kill, or exact-dispatch authority.

Next highest-EV patch: add activation-level parity instrumentation for
`HNeRVDecoderMLX` versus the public PR95 PyTorch decoder, then promote the
queue harvest into planner/probe observations so Stage 1/5/8 timing and parity
become reusable substrate-training cost-model signal.
