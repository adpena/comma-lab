# PR101 selector guard and vectorized exact-search hardening - 2026-05-14

## Scope

Operator directive: keep pushing real score-lowering work, prioritize the PR101 near-miss, and avoid more narrative-only churn.

This pass focused on the only current artifact within roughly `1e-4` of the `<0.192` CPU target:

- Existing exact CPU anchor: `experiments/results/modal_auth_eval_cpu/archive_8866ebb655e9/modal_cpu_auth_eval_result.json`
- Axis: `[contest-CPU]`
- Score: `0.19209788683213053`
- Archive: `experiments/results/pr101_frame_exploit_selector_fec3_compact_exact_k8_cpu_overlay_20260514_codex/archive.zip`
- Archive SHA-256: `8866ebb655e96ccf0ffcd84feae08c131734cba8c402bfb8c661a29f289ce409`

## Code Change

Landed in `tools/build_pr101_frame_exploit_selector_packet.py`:

- Added `--selector-seg-guard-delta` for reproducible guard-relaxation sweeps.
- Recorded `selector_seg_guard_delta`, `selector.seg_guard_delta`, and `selector.selector_search_engine` in packet manifests.
- Replaced the slow pure-Python exact compact selector search with a deterministic NumPy-backed exact exhaustive path when NumPy is available.
- Kept the old pure-Python exhaustive path as a fallback.
- Exposed `compact_exact_k4`, `compact_exact_k8`, and `compact_exact_k16` through the CLI so exact-K sweeps are reproducible without private helper calls.

Test added in `src/tac/tests/test_frame_exploit_selector_packet.py`:

- `test_pr101_compact_exact_respects_seg_guard_delta`

Verification:

```bash
.venv/bin/python -m pytest src/tac/tests/test_frame_exploit_selector_packet.py -q
```

Result: `17 passed in 1.89s`.

Follow-up exact-K CLI verification after exposing K variants:

```bash
.venv/bin/python -m pytest src/tac/tests/test_frame_exploit_selector_packet.py -q
```

Result: `18 passed in 1.93s`.

## Guard Probe Result

Byte-closed guarded packet built:

```bash
.venv/bin/python tools/build_pr101_frame_exploit_selector_packet.py --artifact-dir experiments/results/frame_exploit_segnet_posenet_20260514_pr101_mps600_codex --archive experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip --source-runtime experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec --output-dir experiments/results/pr101_frame_exploit_selector_fec3_compact_exact_k8_top64_guard1e7_20260514_codex --selector-policy-mode compact_exact_k8 --selector-seg-guard-delta 1e-7 --lane-id lane_pr101_fec3_compact_exact_k8_top64_guard1e7_20260514 --overlay-artifact-dir experiments/results/frame_exploit_segnet_posenet_20260514_pr101_cpu_top64_topmodes_codex --allow-nonpositive-charged-proxy
```

Artifact summary:

- Archive: `experiments/results/pr101_frame_exploit_selector_fec3_compact_exact_k8_top64_guard1e7_20260514_codex/archive.zip`
- Archive SHA-256: `40ddab6b553f93ed51631ff13bd6ca0b5ea38c2931b627b0600f48027b1ac0a3`
- Bytes: `178517`
- Selector payload bytes: `249`
- Selected non-none pairs: `436/600`
- Search engine: `numpy_exact_exhaustive`
- Proxy axis: `PR101 MPS/macOS proxy only`
- Proxy charged score: `0.1921051475342287`
- Proxy uncharged score: `0.19193269006537006`

## Classification

No exact CPU/CUDA dispatch from this guarded packet.

Reason:

- The guarded packet is better than the top-64 no-guard proxy packet (`0.19211534792326052`) but still worse than the already evaluated exact CPU near-miss (`0.19209788683213053`).
- It remains above the operator target `<0.192`.
- Dispatching it would spend eval budget without a defensible path to a new frontier.

## Next Score-Lowering Implication

The remaining PR101 gap is not closed by guard relaxation. The likely next useful PR101 moves are:

- reduce selector overhead by at least about `140` charged bytes, or
- find a new frame/postdecode mode family that produces real component gain without widening the selector payload, or
- combine the selector packet with an independently verified PR101 source-payload byte saving.

The builder hardening is still valuable because it turns future `K × guard × overlay` selector sweeps from minutes into seconds and preserves exact reproducibility in packet manifests.

## Exact K16 Boundary

Direct charged exact-K16 probe after vectorizing search:

- Best base/top16/top64 guard charged proxy: about `0.19211844`
- Best uncharged proxy: about `0.19188539`
- Charged bytes: `178608`
- Selector payload bytes: `340`
- Classification: not dispatchable; the extra uncharged component gain is eaten by the 4-bit selector rate.

## Lower-Rate Procedural Selector Boundary

Measured whether selector bytes could be reduced by replacing the per-pair selector with lower-rate rules:

- Best fixed single-mode optimistic estimate: about `0.19291` charged, not viable.
- Piecewise constant segment selector over the exact-K8 palette:
  - 8 segments: about `0.19284` charged
  - 64 segments: about `0.19263` charged
  - 192 segments: about `0.19254` charged
- Classification: not viable for this artifact. The selector choices are not smooth enough in pair index to trade per-pair codes for segment endpoints without losing too much component score.
