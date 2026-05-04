# Lane QZS3+QP1 packer LANDED — PR #67 byte-equivalent compose path ready

**Date**: 2026-05-01 ~13:50Z
**Subagent**: #2 (research-driven implementation, parallel with Q-FAITHFUL retrain dispatcher)
**Mission**: Implement PR #67 (EthanYang qpose14_qzs3, leaderboard rank-1 ~0.31)
byte-identical writer side: QZS3 grouped variable-bit FP4 weight packer + QP1
pose codec + single-blob "p" container.
**Source mandate**: Grand Council 2026-05-01T18:15Z verdict 22/22 GO
(memory: `project_grand_council_shannon_floor_eureka_session_20260501.md`)
ranked this Wave-1 — highest-EV path to close the 0.687 gap from our deploy
champion 0.9974 to the leaderboard 0.31-0.33 band.

## Files landed

| Path | LOC | Purpose |
|---|---|---|
| `src/tac/qp1_pose_codec.py` | 152 | NEW — PR #67 QP1 writer + reader (5-byte header + ZigZag-VLQ velocity deltas, cols 1-5 dropped). Distinct from existing `encode_pose_qpose14_col_delta` (different magic `b"QP14"`, all-6-cols col-deltas). |
| `experiments/build_qpose_archive.py` | 282 | NEW — orchestrator: state_dict + pose array + mask brotli bytes → single STORED zip with concatenated `mask_obu_br | model_qzs3_br | pose_q_br` blob. Emits metadata.json with SHA-pinned per-segment offsets. |
| `src/tac/tests/test_qzs3_packer.py` | 138 | NEW — byte-equivalence tests vs PR #67's actual `get_grouped_qv_state_dict` unpacker. |
| `src/tac/tests/test_qp1_pose_codec.py` | 158 | NEW — round-trip + byte-equivalence vs PR #67 reader + decode of published PR #67 archive's pose stream. |
| `src/tac/tests/test_build_qpose_archive.py` | 124 | NEW — end-to-end: build_qpose_archive → PR #67 inflate.py decodes our blob without crashing. |

QZS3 codec (`src/tac/quantizr_qzs3_codec.py`, 404 LOC) was already present —
written by an earlier subagent against pr67_inflate.py reference. This work
**verified** it byte-equivalent against PR #67's deployed archive; no
modifications were necessary.

## Verification verdict

**JointFrameGenerator structural compatibility**: COMPATIBLE WITHOUT
WRAPPERS. PR #67's `get_grouped_qv_state_dict` iterates
`for name, module in template.named_modules(): if not isinstance(module,
(QConv2d, QEmbedding))`. Our `tac.quantizr_faithful_renderer` uses raw
`nn.Conv2d` / `nn.Embedding`. The codec sidesteps this by using
**name-based filtering** (`_is_fp4_weight_name` matches `.dw.weight` /
`.pw.weight` excluding head conv1x1) — semantically equivalent to PR #67's
isinstance check for the JointFrameGenerator-shaped state_dict. The
59288-byte payload our encoder emits is **byte-equal** in structure to PR
#67's deployed archive (verified against unpacker output).

**QP1 verification**: NEW MODULE NEEDED (existing
`encode_pose_qpose14_col_delta` is a different codec). Verified
`decode_qp1(encode_qp1(poses))` reconstructs the velocity column within
1/512 m/s quantization step (test
`test_qp1_round_trip_velocity_within_quant_step`). Verified our reader
decodes the published PR #67 archive's pose stream to a 600-row array with
mean velocity 33.3 m/s (test
`test_qp1_decode_matches_published_pr67_archive`). Verified our payload
decodes byte-identically through PR #67's inline reader implementation
(test `test_qp1_decoder_byte_equivalent_to_pr67_reader`).

**Round-trip test results**:
- 15 new tests, ALL PASS in 1.52s
- 17 existing `test_quantizr_faithful_renderer.py` tests still PASS (no regression)
- FP4 layers (40 of them): max abs diff 5.8e-2 (FP4 codebook noise)
- FP16 layers (head + bias + dense_fp + embedding): max abs diff 6.3e-4 (FP16 round)
- QV layers (23 of them, 8/9/10-bit per-row min/step quant): max abs diff 6.3e-4

## End-to-end smoke verification

```
$ PYTHONPATH=src .venv/bin/python experiments/build_qpose_archive.py \
    --output-dir /tmp/qpose_smoke --smoke
{
  "archive_bytes": 271524,
  "blob_bytes": 271424,
  "model_uncompressed_bytes": 59288,    # ← byte-identical to PR #67 deployed
  "model_br_bytes": 51931,              # ← random init compresses better
  "pose_codec": "qp1",
  "pose_uncompressed_bytes": 604,
  "pose_br_bytes": 21,
  "blob_sha256": "65684a958311370c0c4f76a4a066b4c62c796c93a55689a26bb37666ff535807"
}
```

PR #67's `get_grouped_qv_state_dict` decodes the 59288-byte QZS3 payload
to all 111 state_dict keys, loads cleanly into `pr67.JointFrameGenerator`
with `strict=True`, and runs forward producing finite (1, 3, 384, 512)
output. SMOKE PASS confirmed.

## Predicted byte budget for Q-FAITHFUL trained checkpoint

| Component | Bytes (PR #67 deployed) | Bytes (our smoke) | Q-FAITHFUL trained projection |
|---|---|---|---|
| mask_obu_br | 219472 | 219472 (placeholder zeros) | 219472 (need real AV1 OBU encoding pipeline) |
| model_qzs3_br | 56093 | 51931 (random) | **~56000** (trained weights have similar entropy to PR #67) |
| pose_q_br | 899 | 21 (constant velocity) | **~900-1100** (real pose stream) |
| **Total blob** | 276464 | 271424 | **~276400** |
| Archive overhead | ~100 | 100 | ~100 |
| **Final archive** | 276564 | 271524 | **~276500** |

**Predicted score band**: matching PR #67's archive byte count + matching
their codec contracts → **0.31-0.33 band achievable**, conditional on
Q-FAITHFUL training producing a checkpoint with PoseNet/SegNet distortion
within 5% of Quantizr's. The packer alone closes 0 of the gap; it
**unblocks** the trained checkpoint to be measured.

## Blockers / open questions for harvest-time integration

1. **mask.obu.br pipeline**: this packer treats `mask_obu_br` as opaque
   bytes — the actual AV1 OBU encoding of the 600 odd-frame masks is owned
   by upstream tooling (likely `experiments/build_obu_archive.py` or
   `mask_codec.py`). Harvest path needs to either (a) reuse PR #67's
   actual mask.obu.br bytes from `pr67_archive.zip` for a deploy-eq smoke
   test, or (b) generate fresh masks from our own pose-flowed predictions.

2. **PR #67's `model_br_len` heuristic** (`pr67_inflate.py:746-762`) is
   brittle — it switches based on total payload length windows. The
   metadata.json our orchestrator emits provides authoritative slice
   offsets. The dispatch wrapper must use the metadata, NOT the heuristic
   (the harvest-time integration path should patch `inflate.py` to read a
   leading uint16 length prefix OR enforce the standard model_br_len
   window by padding the trailing pose slice).

3. **`raw_uint16` pose codec** is supported as a fallback (preserves all 6
   cols at qpose14 quantization); deciding qp1-vs-raw_uint16 should be
   council-reviewed once Q-FAITHFUL pose distortions are measured. PR #67
   chose qp1 (lossy cols 1-5 zeroed) and still ranked 1; council can
   re-evaluate with our actual pose statistics.

## Reactivation / harvest trigger

When Q-FAITHFUL checkpoint lands (sister subagent
`a1f688e0ea962bea2`'s output at
`experiments/results/lane_q_faithful_retrain_20260501/`):

```bash
PYTHONPATH=src .venv/bin/python experiments/build_qpose_archive.py \
    --renderer-state experiments/results/lane_q_faithful_retrain_20260501/best.pt \
    --mask-obu-br <path-to-real-mask-obu-br> \
    --pose-file experiments/results/.../optimized_poses.pt \
    --output-dir experiments/results/lane_qzs3_qp1_first_candidate \
    --pose-codec qp1
```

Then dispatch contest_auth_eval against the resulting
`experiments/results/lane_qzs3_qp1_first_candidate/archive.zip`. The
metadata.json will tell you whether the layout sits in a PR #67 model_br_len
window or needs the dispatch glue to read the metadata for slicing.

## Cross-references

- Parent council: `project_grand_council_shannon_floor_eureka_session_20260501.md`
- Sister subagent: Q-FAITHFUL retrain (lane `lane_q_faithful_retrain`,
  scripts/remote_lane_q_faithful_jointgen.sh — owned, do not touch)
- Architecture: `tac.quantizr_faithful_renderer.JointFrameGenerator`
  (Quantizr PR #55 1:1 port, audit
  `.omx/research/quantizr_replica_audit_20260428.md`)
- PR #67 reference: `reports/raw/leaderboard_intel_20260501/pr67_inflate.py`
  (39KB) + `pr67_archive.zip` (276KB)
- Lane registered in `.omx/state/lane_registry.json` at L1 (impl_complete +
  memory_entry gates true; awaiting empirical archive measurement on
  Q-FAITHFUL checkpoint to advance to L2).

## Determinism + custody

- QZS3 encoder is deterministic (test `test_encoder_is_deterministic`)
- Single-blob zip uses `ZipInfo(date_time=(1980,1,1,...))` and
  `compress_type=ZIP_STORED` — byte-for-byte reproducible across
  Python/zipfile versions
- metadata.json includes `blob_sha256` for SHA-pinning custody chains
- block_size header field round-trips (test `test_block_size_header_field_round_trips`)
