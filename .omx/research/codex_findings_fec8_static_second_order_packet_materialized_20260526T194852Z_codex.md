# Codex Findings: FEC8 Static Second-Order Packet Materialized

- timestamp_utc: `2026-05-26T19:48:52Z`
- lane_id: `lane_pr101_frame_exploit_selector_fec8_static_second_order_k16_20260526`
- artifact_dir: `experiments/results/pr101_frame_exploit_selector_fec8_static_second_order_k16_20260526T1552Z_codex`
- source_commit: `9aa2ef4b2f541243ebb497bfc7e6c77fc034296d`
- score_claim: `false`
- promotion_eligible: `false`
- rank_or_kill_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- evidence_axis: `PR101 MPS/macOS proxy only`

## Result

The FEC8 static second-order Markov selector codec is now materialized through the
PR101 FP11 packet builder as a byte-closed archive plus decode-only receiver
runtime.

Byte accounting versus the tracked FEC6 fixed-Huffman K16 packet:

| Packet | Archive bytes | Selector payload bytes | Selector index bytes | Selector code bits |
| --- | ---: | ---: | ---: | ---: |
| FEC6 fixed-Huffman K16 clean | `178517` | `249` | `243` | `1944` |
| FEC8 static second-order Markov K16 | `178507` | `239` | `231` | `1848` |

The live FEC8 archive is `10` bytes smaller than the FEC6 archive and preserves
the same fixed K16 canonical palette:

```text
none
frame0_blue_chroma_amp_1
frame0_blue_chroma_amp_3
frame0_luma_bias_+1
frame0_luma_bias_-1
frame0_luma_bias_-2
frame0_luma_bias_-4
frame0_rgb_bias_m2_p1_p1
frame0_rgb_bias_m4_p2_p2
frame0_rgb_bias_p0_m1_p1
frame0_rgb_bias_p0_m2_p2
frame0_rgb_bias_p0_p1_m1
frame0_rgb_bias_p0_p2_m2
frame0_rgb_bias_p2_m1_m1
frame0_rgb_bias_p4_m2_m2
frame0_roll_dx+0_dy+1
```

## Artifact Custody

- archive: `experiments/results/pr101_frame_exploit_selector_fec8_static_second_order_k16_20260526T1552Z_codex/archive.zip`
- archive_bytes: `178507`
- archive_sha256: `b44da5d54d34ce094c8fca7a37172b9ea6546d8a56eb960afa86252f56d11844`
- selector_payload_sha256: `5ec83edf357b32dc15b696b0a5ba796a732a1563cbd2b4685cf8fa67e34d88da`
- runtime_content_tree_sha256: `65a5e481c15ab87d6666f11e909fcbe1257e531fbae8c946b8012656ecfbfed8`
- runtime_tree_sha256: `60d52ca3119dc1be59846f09e962698b585a74b22858e7ffb750b1c4b583fc5b`

Receiver closure is deterministic and decode-only:

- `submission_dir/src/fec8_markov_decoder.py`
- `submission_dir/encoder/build_pr101_frame_exploit_selector_packet_markov.py`

The receiver does not optimize, inspect scorer state, fetch sidecars, or adapt at
eval time. The Markov prior is a static archive/runtime decoder contract.

## Reproduction

```bash
.venv/bin/python tools/build_pr101_frame_exploit_selector_packet.py \
  --artifact-dir experiments/results/frame_exploit_segnet_posenet_20260514_pr101_mps600_codex \
  --archive experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip \
  --source-runtime experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec \
  --output-dir experiments/results/pr101_frame_exploit_selector_fec8_static_second_order_k16_20260526T1552Z_codex \
  --selector-policy-mode compact_exact_k16 \
  --lane-id lane_pr101_frame_exploit_selector_fec8_static_second_order_k16_20260526 \
  --compact-selector-codec fec8_static_second_order_markov_k16 \
  --overlay-artifact-dir experiments/results/frame_exploit_segnet_posenet_20260514_pr101_cpu_top16_topmodes_v2_codex \
  --allow-nonpositive-charged-proxy
```

## Next Gate

This is still false-authority. The next promotion-relevant action is claimed
paired exact auth eval on the matching contest CPU/CUDA axes, or a fail-closed
blocker if the dispatch claim/runtime import gate refuses.
