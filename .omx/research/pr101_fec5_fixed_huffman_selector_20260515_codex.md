# PR101 FEC5 Fixed-Huffman Selector - 2026-05-15

## Classification

This is real rate-lowering work for the PR101/FEC3 CPU near-miss basin, not a
new score claim.

- axis: proxy/rate-only until exact eval
- source packet: PR101 FEC3 compact exact-K8 selector
- source archive SHA-256:
  `8866ebb655e96ccf0ffcd84feae08c131734cba8c402bfb8c661a29f289ce409`
- FEC5 archive:
  `experiments/results/pr101_frame_exploit_selector_fec5_fixed_huffman_k8_cpu_overlay_20260515_codex/archive.zip`
- FEC5 archive SHA-256:
  `13e6dd3bf26866bdb633e6ab9910f59c73d2ab6e7d21ccf6b22b2c0a9d49c523`
- FEC5 archive bytes: `178477`
- byte delta vs FEC3: `-40`
- selector payload: `249 -> 209` bytes

## Engineering Result

`tools/build_pr101_frame_exploit_selector_packet.py` now supports
`--compact-selector-codec fec5_fixed_huffman_k8` for compact PR101 selector
packets. The format uses a fixed exact-K8 palette and prefix code lengths
`[2, 2, 3, 3, 4, 4, 4, 4]`, removing the FEC3 charged palette table and
compressing the observed 600-pair selector stream from 225 fixed-bit index
bytes to 203 Huffman bitstream bytes.

Local parser verification:

- source payload equal to FEC3: `true`
- per-pair selected mode IDs equal to FEC3: `true`
- pair count: `600`
- FEC5 selector bit count: `1621`
- FEC5 avg bits/pair: `2.7016666666666667`

## Score Implication

The FEC3 exact CPU score is `0.19209788683213053`. FEC5 preserves the selector
decisions and only lowers bytes by 40, so at unchanged components the expected
CPU score movement is rate-only:

```text
40 * 25 / 37,545,489 = 0.000026634
0.19209788683213053 - 0.000026634 = about 0.19207125
```

This does not clear the operator `<0.192` threshold. It closes a real byte
leak, but selector-overhead removal alone cannot escape this basin.

## Verification

Commands run:

```bash
.venv/bin/ruff check tools/build_pr101_frame_exploit_selector_packet.py src/tac/tests/test_frame_exploit_selector_packet.py
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/tests/test_frame_exploit_selector_packet.py -q
.venv/bin/python tools/build_pr101_frame_exploit_selector_packet.py --artifact-dir experiments/results/frame_exploit_segnet_posenet_20260514_pr101_mps600_codex --archive experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip --source-runtime experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec --output-dir experiments/results/pr101_frame_exploit_selector_fec5_fixed_huffman_k8_cpu_overlay_20260515_codex --selector-policy-mode compact_exact_k8 --compact-selector-codec fec5_fixed_huffman_k8 --lane-id lane_pr101_frame_exploit_selector_fec5_fixed_huffman_k8_cpu_overlay_20260515 --overlay-artifact-dir experiments/results/frame_exploit_segnet_posenet_20260514_pr101_cpu_top16_topmodes_v2_codex
```

Test result: `19 passed`.

## Decision

The `0.19209788683213053` artifact is legitimate `[contest-CPU]`, not
`[contest-CUDA]`. FEC5 proves the current selector byte overhead was not fully
minimized, but the remaining improvement is too small to break `<0.192` by
itself. Further work must either improve component terms under exact CPU/CUDA
or move to a different representation/selector family; more byte-only work in
this exact K8 selector stream is unlikely to be decisive.
