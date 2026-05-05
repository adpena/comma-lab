---
name: Lane STC-AV1-residual smoke PRELIM (CPU fallback) — REQUIRES CUDA SegNet for valid measurement
description: 2026-04-29 PM. First-pass CPU smoke on the codex top-1 STC redesign (Hybrid AV1+residual). Best total 1.43MB at CRF 35 vs anchor 421KB. BUT the comparison is invalid — fallback used AV1-decoded anchor masks as "clean" reference, so the residual measures AV1 transcoding artifacts not the SegNet→AV1 quantization gap. Proper smoke requires SegNet argmax on CUDA (CLAUDE.md non-negotiable). Also identified RLE encoding inefficiency (uint32 per transition wastes ~4× bytes; varint would help).
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Smoke results (CPU fallback, NOT VALID for kill/keep)

| CRF | AV1 bytes | Residual bytes | Total | diff_frac |
|---:|---:|---:|---:|---:|
| 35 | 828,469 | 601,885 | 1,430,354 | 0.034% |
| 45 | 498,461 | 1,567,866 | 2,066,327 | 0.099% |
| 50 | 393,121 | 1,981,915 | 2,375,036 | 0.137% |
| 55 | 305,791 | 2,867,658 | 3,173,449 | 0.236% |

Anchor masks.mkv: 421,483 bytes. None of the CRF points in the fallback smoke beats the anchor.

## Why this measurement is INVALID for strategy

1. **Wrong "clean" reference**: the smoke decoded the anchor's AV1-encoded masks.mkv and used THAT as the "clean" target. So the residual measures the AV1 transcoding round-trip (anchor AV1 → decode → re-encode at new CRF → decode → compare), NOT the actual SegNet → AV1 quantization gap.
2. **CLAUDE.md non-negotiable**: SegNet output must be on CUDA for any strategic measurement. Both MPS and CPU SegNet drift from contest-CUDA. The proper "clean" reference is `SegNet_CUDA(GT_video) → argmax`, not the AV1-decoded anchor.
3. The CPU encoder/decoder portion (ffmpeg + integer arithmetic) IS deterministic across hosts. The bug is in the upstream SegNet step which never ran in this smoke.

## Engineering bugs found in the smoke encoder

1. **RLE wastes 4× bytes**: uses uint32 (4 bytes) per transition. For sparse diffs with ~80K-560K transitions, this contributes 320KB-2.2MB. Varint encoding would cut to ~1 byte average per transition (typical run-length fits in 1 byte). Fix: replace `int(rl).to_bytes(4, "little")` with varint encoding.

2. **Class blob unconstrained**: `arithmetic_qint_codec.encode_qints_arithmetic(class_array, num_symbols=5)` works but the codec wasn't designed for 5-symbol streams; could be tightened with a class prior matching observed road-dominant distribution.

3. **No flow-based prediction**: the smoke doesn't use temporal coherence. Each frame's mask is encoded independently. Lane STC-temporal redesign (codex top-3) would address this.

## Conclusion + next step

This smoke RULES OUT the AV1-anchor-vs-AV1-anchor case but does NOT rule out the proper SegNet-vs-AV1 case. Required to validate or kill the Hybrid AV1+residual redesign:

1. Run SegNet on CUDA over the GT video (Modal T4, ~$0.20)
2. Save the clean argmax as `clean_segnet_masks.pt`
3. Re-run the smoke with `--clean-masks clean_segnet_masks.pt`
4. THIS measurement is valid for kill/keep decision

If the proper CUDA smoke also shows >380KB total at all CRFs, the codec council kill threshold fires and we move to NeRV (top-2) or stored-flow (top-3) redesigns.

If proper CUDA smoke shows <380KB at any CRF, we have a viable lane.

## Cross-refs

- experiments/build_lane_stc_av1_residual_smoke.py (the smoke implementation)
- experiments/results/lane_stc_av1_residual_smoke/manifest.json (the byte counts)
- project_stc_redesign_verdict_20260429.md (the parent council verdict)
- /tmp/codex_runs/stc_redesign_brainstorm.log (the original 22-voice ranking)
