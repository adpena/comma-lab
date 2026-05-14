# PR101 Local Sweep - Nonlocal Bias RSS

Date: 2026-05-14

Scope: local CPU/macOS advisory and byte-level PR101-family sweeps only. No paid
or GPU dispatch was run. HDM8 postfilter files owned by the main thread were not
edited or read for implementation.

## Preflight

- Read `CLAUDE.md`, `AGENTS.md`, and `PROGRAM.md` before acting.
- Checked `.omx/state/lane_registry.json` and `.omx/state/active_lane_dispatch_claims.md`.
- Checked recent directive files; no `.omx/research/*_directive_*` file was dated
  within the last 24 hours.
- Existing exact-CUDA terminal claims already retired
  `bias_refine_cmaes_0050` and `bias_refine_cmaes_0053` as non-improving
  PR101-family packets, so both were excluded from this local sweep.

## Commands

```bash
mkdir -p experiments/results/pr101_local_sweeps_20260514_nonlocal_bias_rss/logs
```

```bash
/usr/bin/time -l .venv/bin/python tools/build_pr101_nonlocal_sweep_packets.py \
  --out-dir experiments/results/pr101_local_sweeps_20260514_nonlocal_bias_rss \
  --top-k 5 \
  --exclude-candidate-id bias_refine_cmaes_0050 \
  --exclude-candidate-id bias_refine_cmaes_0053
```

```bash
/usr/bin/time -l .venv/bin/python experiments/contest_auth_eval.py \
  --archive experiments/results/pr101_local_sweeps_20260514_nonlocal_bias_rss/packets/bias_refine_cmaes_0044/archive.zip \
  --inflate-sh experiments/results/pr101_local_sweeps_20260514_nonlocal_bias_rss/packets/bias_refine_cmaes_0044/inflate.sh \
  --upstream-dir upstream \
  --video-names-file upstream/public_test_video_names.txt \
  --device cpu \
  --json-out experiments/results/pr101_local_sweeps_20260514_nonlocal_bias_rss/local_cpu_advisory/bias_refine_cmaes_0044/contest_auth_eval.macos_cpu_advisory.json \
  --allow-temp-work-dir \
  --inflate-timeout 1800 \
  --evaluate-timeout 1800 \
  --expected-runtime-tree-sha256 d4dcd529fae5f89514d97c58cb3da128413af7055db910f6085977e94e6abde3
```

First CPU eval attempt failed before scoring because `inflate.sh` invokes
`python` and the shell PATH did not include `.venv/bin`.

```bash
env PATH=/Users/adpena/Projects/pact/.venv/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  /usr/bin/time -l .venv/bin/python experiments/contest_auth_eval.py \
  --archive experiments/results/pr101_local_sweeps_20260514_nonlocal_bias_rss/packets/bias_refine_cmaes_0044/archive.zip \
  --inflate-sh experiments/results/pr101_local_sweeps_20260514_nonlocal_bias_rss/packets/bias_refine_cmaes_0044/inflate.sh \
  --upstream-dir upstream \
  --video-names-file upstream/public_test_video_names.txt \
  --device cpu \
  --json-out experiments/results/pr101_local_sweeps_20260514_nonlocal_bias_rss/local_cpu_advisory/bias_refine_cmaes_0044/contest_auth_eval.macos_cpu_advisory.json \
  --allow-temp-work-dir \
  --inflate-timeout 1800 \
  --evaluate-timeout 1800 \
  --expected-runtime-tree-sha256 d4dcd529fae5f89514d97c58cb3da128413af7055db910f6085977e94e6abde3
```

```bash
/usr/bin/time -l .venv/bin/python experiments/load_pr101_archive_to_state_dict.py \
  --archive experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip \
  --output-state-dict experiments/results/pr101_local_sweeps_20260514_nonlocal_bias_rss/pr101_decoder_state_dict.pt \
  --metadata-output experiments/results/pr101_local_sweeps_20260514_nonlocal_bias_rss/pr101_archive_state_metadata.json
```

```bash
/usr/bin/time -l .venv/bin/python tools/pr101_per_tensor_brotli_sweep.py \
  --state-dict-path experiments/results/pr101_local_sweeps_20260514_nonlocal_bias_rss/pr101_decoder_state_dict.pt \
  --output experiments/results/pr101_local_sweeps_20260514_nonlocal_bias_rss/pr101_per_tensor_brotli_sweep_focus.json \
  --quality-min 9 --quality-max 11 \
  --lgwin-min 18 --lgwin-max 24 \
  --lgblock-min 16 --lgblock-max 24
```

```bash
/usr/bin/time -l .venv/bin/python tools/pr101_omega_opt_per_tensor_codec_choice_empirical.py \
  --state-dict experiments/results/pr101_local_sweeps_20260514_nonlocal_bias_rss/pr101_decoder_state_dict.pt \
  --output-json experiments/results/pr101_local_sweeps_20260514_nonlocal_bias_rss/pr101_omega_per_tensor_codec_choice.json
```

## Artifacts

- `experiments/results/pr101_local_sweeps_20260514_nonlocal_bias_rss/summary.json`
- `experiments/results/pr101_local_sweeps_20260514_nonlocal_bias_rss/exact_ready_pr101_nonlocal_bias_queue.json`
- `experiments/results/pr101_local_sweeps_20260514_nonlocal_bias_rss/packets/*/`
- `experiments/results/pr101_local_sweeps_20260514_nonlocal_bias_rss/candidate_manifests/*`
- `experiments/results/pr101_local_sweeps_20260514_nonlocal_bias_rss/exact_ready/*`
- `experiments/results/pr101_local_sweeps_20260514_nonlocal_bias_rss/local_cpu_advisory/bias_refine_cmaes_0044/contest_auth_eval.macos_cpu_advisory.json`
- `experiments/results/pr101_local_sweeps_20260514_nonlocal_bias_rss/pr101_archive_state_metadata.json`
- `experiments/results/pr101_local_sweeps_20260514_nonlocal_bias_rss/pr101_per_tensor_brotli_sweep_focus.json`
- `experiments/results/pr101_local_sweeps_20260514_nonlocal_bias_rss/pr101_omega_per_tensor_codec_choice.json`
- `experiments/results/pr101_local_sweeps_20260514_nonlocal_bias_rss/logs/*`

## Packet Sweep

`tools/build_pr101_nonlocal_sweep_packets.py` selected five exact-ready runtime
packets, all with `score_claim=false`:

| candidate | proxy objective | archive bytes | archive sha256 | runtime tree sha256 |
| --- | ---: | ---: | --- | --- |
| `bias_refine_cmaes_0044` | 0.19285694147425023 | 178258 | `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e` | `d4dcd529fae5f89514d97c58cb3da128413af7055db910f6085977e94e6abde3` |
| `bias_refine_cmaes_0047` | 0.19286038938290923 | 178258 | `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e` | `3bbf7ec3b66322478a74cd72bdc04b931c1d943cbb31e66134124bcdea248269` |
| `bias_refine_cmaes_0046` | 0.19286058338451523 | 178258 | `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e` | `d02af5940febc46231d3361bf695cd378a9a22309f8260aeca512ea37c460a82` |
| `bias_refine_cmaes_0062` | 0.19286131350449545 | 178258 | `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e` | `cd7cb1b01d5750a7357cc23e5d58fa4dfaaf76a05291f69ef02f42c514394d97` |
| `bias_refine_cmaes_0042` | 0.19286234038533343 | 178258 | `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e` | `ad2da3f17f8ad95d902468091ed623f012fbd27a5b2b422b769a62ee09d4584e` |

Build runtime: 1.91 s real. Max RSS: 222,511,104 bytes.

## Local CPU Advisory

The top proxy candidate, `bias_refine_cmaes_0044`, completed a full local
macOS CPU advisory run:

- score axis: `[macOS-CPU advisory]`
- score: 0.1929870127024255
- avg SegNet distance: 0.00056165
- avg PoseNet distance: 0.00003286
- archive bytes: 178258
- samples: 600
- inflate elapsed: 37.2495330828242 s
- evaluate elapsed: 439.91880554216914 s
- total elapsed: 478.3543437081389 s
- max RSS: 10,880,270,336 bytes
- inflated aggregate raw sha256:
  `616df5f846c3609d00edfa23add4e5c2f9116d86896115f23db76ebb324f8f0d`

Existing PR101 local macOS CPU baseline:
`experiments/results/lane_macos_cpu_proxy_empirical_validation_20260513_20260513T154314Z/pr101/contest_auth_eval.json`
records score 0.1928610127024255, avg SegNet 0.00056039, avg PoseNet
0.00003286, and the same 178258 byte archive.

Result: `bias_refine_cmaes_0044` regressed by +0.000126 on the matching local
CPU advisory axis. This is a proxy-transfer negative, not a promotion signal.

## Byte Sweeps

Decoded source PR101 archive metadata:

- archive bytes: 178258
- archive sha256:
  `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`
- inner `x` bytes: 178158
- inner `x` sha256:
  `5f1948f9572e65f71c614d2ff15764ee416522e25cb1b06c8b1299c1306e8aaf`
- decoder blob bytes: 162164
- decoder blob sha256:
  `836d1876bffd74f77f30e387a3b4cac1dbb25929cc4d348830d36cfa2a6d48a6`
- latent blob bytes: 15387
- latent blob sha256:
  `de8a0da594f073efc43849334573ba06438bb37d53f9343ee6367659c0106bbe`

Focused per-tensor Brotli sweep:

- evals: 5292
- runtime: 12.31 s real
- max RSS: 256,770,048 bytes
- total at PR101 default per-tensor setting: 162264 bytes
- total per-tensor optimum: 162238 bytes
- apparent saving versus that per-tensor default: 26 bytes
- blocker: actual PR101 decoder blob is 162164 bytes, so the independent
  per-tensor bound is still 74 bytes worse than the authored stream.

Per-tensor codec-choice empirical sweep:

- baseline lossless per-tensor Brotli bytes: 162299
- best with sidechannel: 162327 bytes
- savings versus that baseline: -28 bytes
- all tested relative-error budgets selected zero sparsified tensors.
- blocker: no runtime exists to consume per-tensor codec IDs with CSR, and the
  sweep remains proxy-only.

## Classification

No promotion claim. No exact-CUDA score claim. No rank/kill claim.

Best local signal: `bias_refine_cmaes_0044` was the highest-signal excluded-new
proxy candidate and produced a byte-closed local CPU advisory packet, but it
regressed on the matching local axis. The actionable update is to deprioritize
adjacent nonlocal bias rows unless a stronger axis-specific predictor appears.

Byte-level sweeps show the tested PR101 decoder stream knobs are saturated: the
local lossless per-tensor Brotli grid and per-tensor codec-choice proxy do not
beat the actual PR101 decoder blob after overhead.

## Blockers And Reactivation

- RESOLVED after this sweep: the generated packet `inflate.sh` invoked bare
  `python`, which broke local automation when `python` was absent from `PATH`.
  `tools/build_pr101_kaggle_proxy_runtime_packet.py` now normalizes copied
  wrappers to use `PYTHON_BIN`, then `python3`, then `python`, with an explicit
  fail-closed error if none exists. Regression coverage:
  `tests/test_build_pr101_kaggle_proxy_runtime_packet.py::test_build_normalizes_bare_python_inflate_wrapper`.
- Remaining adjacent bias-refine rows have weak expected value after exact-CUDA
  negatives for `0050`/`0053` and local CPU regression for `0044`.
- PR101 lossless stream packing did not surface a byte-positive packet; next
  PR101-family work should move to semantic decoder transforms, PR106 inner
  decoder split-codec transfer, or scorer-aware retraining before exact-CUDA
  spend.
- Paid/GPU dispatch remains blocked by operator scope for this turn.
