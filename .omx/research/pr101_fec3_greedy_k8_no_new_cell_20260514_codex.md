# PR101 FEC3 greedy-K8 no-new-cell check - 2026-05-14

## Question

Operator asked whether the `0.192` PR101/FEC3 selector result was legitimate
and whether the film-grain / selector / water-fill basin had been fully
engineered and exhausted.

This check tested one narrow omission: the earlier best proxy K8 packet used
the old `FECI_runtime_implicit_compact_palette` wire format, which did not
charge compact-palette metadata in archive bytes. The relevant engineering
question was whether rebuilding greedy K8 through the corrected FEC3
archive-charged path would expose a new byte-closed cell.

## Build

Command:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/build_pr101_frame_exploit_selector_packet.py \
  --artifact-dir experiments/results/frame_exploit_segnet_posenet_20260514_pr101_mps600_codex \
  --archive experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip \
  --source-runtime experiments/results/public_pr_intake_full/public_pr101_intake_20260505_auto/source/submissions/hnerv_ft_microcodec \
  --overlay-artifact-dir experiments/results/frame_exploit_segnet_posenet_20260514_pr101_cpu_top16_topmodes_v2_codex \
  --selector-policy-mode compact_greedy_k8 \
  --output-dir experiments/results/pr101_frame_exploit_selector_fec3_greedy_k8_cpu_overlay_20260514_codex \
  --json-out experiments/results/pr101_frame_exploit_selector_fec3_greedy_k8_cpu_overlay_20260514_codex/packet_manifest.json \
  --lane-id lane_pr101_fec3_greedy_k8_cpu_overlay_20260514
```

## Artifact

- packet manifest:
  `experiments/results/pr101_frame_exploit_selector_fec3_greedy_k8_cpu_overlay_20260514_codex/packet_manifest.json`
- archive:
  `experiments/results/pr101_frame_exploit_selector_fec3_greedy_k8_cpu_overlay_20260514_codex/archive.zip`
- archive bytes: `178517`
- archive sha256:
  `39fc2cc2f7b4e66585c4f659b61142e78719a51444f1c4225a085356577d1aed`
- selector payload bytes: `249`
- wire format: `FEC3_archive_charged_static_or_dynamic_compact_palette`
- proxy axis: `PR101 MPS/macOS proxy only`
- charged proxy score: `0.1921079648509577`
- charged proxy delta: `-0.0007531655700472317`
- score claim: `false`
- promotion eligible: `false`
- contest CPU/CUDA: not run

## Classification

This is not a new score-lowering cell. It lands on the same charged proxy
surface as the already evaluated FEC3 exact-K8 packet:

- evaluated packet:
  `experiments/results/pr101_frame_exploit_selector_fec3_compact_exact_k8_cpu_overlay_20260514_codex/archive.zip`
- evaluated archive sha256:
  `8866ebb655e96ccf0ffcd84feae08c131734cba8c402bfb8c661a29f289ce409`
- evaluated `[contest-CPU]` score:
  `0.19209788683213053`
- evaluated archive bytes: `178517`

The fresh greedy-K8 packet changes palette ordering and archive SHA, but does
not reduce bytes or improve the charged proxy score. It should not be
dispatched unless a separate reason appears.

## Exhaustion Update

The current FEC3 selector overhead is not the remaining blocker by itself:

- score gap to `<0.192`: `0.00009788683213052263`
- bytes equivalent at unchanged components: about `147.01` bytes
- current FEC3 K8 selector payload: `249` bytes
- observed entropy lower bound for current K8 codes: about `199.47` bytes

So a perfect selector entropy coder can plausibly recover only about `25`
bytes before table/runtime overhead, not the full gap. The next score-lowering
move needs either:

1. a source-payload byte saving outside the selector,
2. a new postdecode mode family with real component gain at the same selector
   width, or
3. a lower-rate selector that preserves much more component gain than the
   measured procedural/segment selectors.

## Verification

Focused tests:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/tests/test_frame_exploit_selector_packet.py -q
```

Result: `18 passed`.
