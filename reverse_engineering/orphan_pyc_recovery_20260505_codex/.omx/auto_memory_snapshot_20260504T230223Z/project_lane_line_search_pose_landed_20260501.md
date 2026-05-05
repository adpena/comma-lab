---
name: Wave-1.5 line-search pose refinement LANDED — port of pr67_line_search.py onto Q-FAITHFUL+QZS3 stack
description: 2026-05-01. ~590 LOC port of PR #67 EthanYang's R(D)-joint coordinate-descent pose refinement tool (pr67_line_search.py, 194 LOC) onto OUR Lane Q-FAITHFUL stack — `experiments/line_search_pose_refinement.py` + `src/tac/tests/test_line_search_pose.py` (12 tests, all passing in <10s on CPU). Stack-on-top for the Wave-1 anchor (Q-FAITHFUL+QZS3 archive) currently training on Vast 35959478, ~13h to harvest. Predicted +0.001 to +0.005 score gain when stacked. Pure CPU/MPS development per task scope; no GPU dispatch.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## What landed

Two files via `tools/subagent_commit_serializer.py`:

- `experiments/line_search_pose_refinement.py` (~590 LOC) — port of pr67_line_search.py.
  Two structural improvements over pr67:
  1. **Metadata-driven byte slicing**: reads `metadata.json` written by
     `experiments/build_qpose_archive.py` instead of pr67's brittle 7-bucket
     `model_br_len` length-lookup at `pr67_inflate.py:746-768`. The lookup
     hard-codes 6 length windows; any new model size silently falls into
     the `61147` default which is wrong for QZS3 archives.
  2. **CUDA-required-by-default**: `--device cuda:0` per CLAUDE.md
     MPS-PoseNet-23x rule. `--device cpu` opt-in prints a stderr banner
     that the resulting pose_mse / objective values are `[advisory only]`
     and CANNOT be used for kill/promote decisions.

- `src/tac/tests/test_line_search_pose.py` (~12 tests, all pass in 9s):
  1. QP1 col0 round-trip (encode → decode → match)
  2. brotli-wrapped pose stream round-trip
  3. Joint-objective monotone descent (best_obj never exceeds baseline)
  4. Determinism (same seed → same refined col0)
  5. main() CLI smoke completes
  6. Refined archive preserves mask + model bytes byte-identical
  7. Zero-radius is no-op
  8. col0 boundary clamping (0, 65535)
  9. Joint-objective formula matches pr67_line_search.py:140 byte-for-byte
  10. CUDA dispatch smoke (gated `@pytest.mark.cuda`, skipped on CPU)
  11. slice_blob rejects length-mismatched payload
  12. write_refined_archive is deterministic stored ZIP
  13. Single-pass single-radius edge case

Plus `pyproject.toml`: registers `cuda` pytest marker (no more
`PytestUnknownMarkWarning`).

## Mathematical contract (mirrors pr67_line_search.py:140)

```
obj = sqrt(10 * pose_mse) + 25 * (mask_br + model_br + pose_br + overhead) / ORIGINAL_SIZE
```

Where `ORIGINAL_SIZE = 37,545,489` and `overhead = 100` (matches pr67 default).
This is EXACTLY the contest's pose+rate sub-objective being directly
optimized at compress time. The 100·seg term is held fixed because masks
aren't being re-quantized in this loop.

## Coordinate-descent loop (mirrors pr67_line_search.py:142-183)

For each `radius in [1, 2, 3, 5, 8]`:
  For each `pass in range(passes)`:
    For each frame `i`:
      - Build candidates `cand = col0[i] + arange(-radius, radius+1)`,
        clamped to [0, 65535]
      - For each candidate `v`: forward (mask[i], pose_from_col0([v]))
        through generator → bilinear (874, 1164) → PoseNet → MSE vs target[i]
      - Keep `v*` with min MSE; update `cur[i] = v*`
    Compute `obj = sqrt(10·mean(pose_mse)) + 25·archive_size/ORIGINAL_SIZE`
    Accept entire pass IFF obj < best_obj; else revert and break radius's loop

Greedy-monotone: every pass that lands strictly decreases the joint objective.

## Smoke architecture for CPU testing

The full pipeline (renderer at 384×512 + PoseNet at 874×1164) is too slow
on CPU for 60-frame sweeps. Test fixture builds:
- 6-frame archive
- Renderer with `out_h=16, out_w=24` (in-place mutation of decoded
  generator's canvas; weights unchanged because `out_h/out_w` only affect
  `make_coord_grid` size + final upsample target)
- `target_h=16, target_w=24` for the bilinear-to-PoseNet step
- `_MockPoseNet`: numpy-flavored deterministic surrogate that mirrors the
  upstream PoseNet's IO contract (`preprocess_input` + `forward` returning
  `{"pose": (B, 12)}`)

Total CPU runtime: 9s for 12 tests (1 skipped).

## Predicted score gain on Wave-1 anchor (when harvested)

Per `reference_pr67_line_search_R_D_joint_coordinate_descent_20260501`:

- **Rate term**: pr67's pose_q is ~0.9 KB AFTER score-aware refinement.
  Our naive QP1 of the same trained col0 might be 1.5-2.5 KB. Refinement
  could trim to ~0.9 KB → -0.6 to -1.6 KB on rate → -0.0004 to -0.001 score.
- **Distortion term**: smaller pose error after the candidate that minimizes
  BOTH distortion AND rate → +0.001 to +0.003 score in distortion savings.
- **Total**: ~0.001-0.005 score improvement on Wave-1 archive.

## Harvest-integration trigger (when to run this)

The Q-FAITHFUL training on Vast 35959478 (per
`project_lane_owv3_0120_orthogonal_stack_LANDED_0_997_20260501.md` follow-up)
is expected to land its first `[contest-CUDA]` archive in ~13h. The
operator one-liner (CUDA, real GT, real PoseNet) is:

```bash
.venv/bin/python experiments/line_search_pose_refinement.py \
    --archive-path  experiments/results/wave1_anchor/archive.zip \
    --metadata-path experiments/results/wave1_anchor/metadata.json \
    --output-path   experiments/results/wave1_5_refined/archive.zip \
    --output-metadata experiments/results/wave1_5_refined/metadata.json \
    --posenet-path upstream/models/posenet.safetensors \
    --gt-mkv upstream/videos/0.mkv \
    --device cuda:0 --batch-size 16 --candidate-chunk 32 \
    --radii "1,2,3,5,8" --passes 2
```

Estimated cost: $0.50 GPU on Vast 4090, 30-60 min wall clock.
Expected output: `archive.zip` smaller by ~600-1600 bytes than the input,
with the same mask + model bytes (byte-identical) and a new pose stream.

## Biggest harvest-integration risk

**The Wave-1 anchor must produce a `metadata.json` with at minimum the
keys `mask_br_bytes`, `model_br_bytes`, `pose_br_bytes`.** If the harvest
script writes a different metadata schema (e.g. embedded in a larger
provenance.json), the line-search tool will hard-error at
`load_metadata()` rather than silently mis-slice the blob. This is by
design (contracts > guesses) but means the harvest path must include a
metadata-emission step. `experiments/build_qpose_archive.py:312` already
writes the right schema — the assumption is that the Q-FAITHFUL training
anchor uses build_qpose_archive (or an equivalent emitter) for its final
archive packaging. If the anchor uses a different orchestrator, a
2-line wrapper that derives the metadata from `os.stat(p).st_size` minus
known header/footer offsets is required first.

## Cross-refs

- `reports/raw/leaderboard_intel_20260501/pr67_line_search.py` (194 LOC reference)
- `reference_pr67_line_search_R_D_joint_coordinate_descent_20260501.md`
- `reference_pr65_pr67_blob_byte_layouts_proper_reverse_engineering_20260501.md`
- `src/tac/qp1_pose_codec.py` (152 LOC, byte-for-byte equivalent to pr67's encode_qp1/decode_qp1, landed at commit cdf099c4)
- `src/tac/quantizr_qzs3_codec.py` (415 LOC, the QZS3 weight codec)
- `src/tac/quantizr_faithful_renderer.py` (337 LOC, JointFrameGenerator)
- `experiments/build_qpose_archive.py` (282 LOC, orchestrator that emits metadata.json)
- `project_lane_owv3_0120_orthogonal_stack_LANDED_0_997_20260501.md` (Wave-1 anchor parent)

## CLAUDE.md compliance

- **Auth eval EVERYWHERE**: this tool's output IS an auth-eval candidate
  archive. Operator must run `inflate.sh → upstream/evaluate.py` on the
  refined `archive.zip` to obtain the `[contest-CUDA]` score. The tool
  itself is a compress-time refinement, not an inflate-time path; the
  strict-scorer-rule is preserved (no scorer at inflate).
- **MPS-falsification rule**: `--device cuda:0` is default. `--device cpu`
  prints a banner that the result is `[advisory only]`. The CUDA-only
  test is gated `@pytest.mark.skipif(not torch.cuda.is_available())`.
- **eval_roundtrip**: N/A — this tool optimizes against the FULL forward
  pipeline (generator → bilinear-upsample → PoseNet). It IS the
  roundtrip; no proxy.
- **Forbidden patterns**: no MPS-fallback default, no invented CLI flags
  (every flag exists in `experiments/line_search_pose_refinement.py:argparse`),
  no silent-skip cascades (hard ValueError on metadata mismatch).
