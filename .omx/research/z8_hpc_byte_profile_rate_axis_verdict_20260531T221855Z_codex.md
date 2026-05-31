# Z8/HPC Byte Profile And Rate-Axis Verdict

Axis: `[macOS-CPU advisory]`
Score claim: `false`
Promotion eligible: `false`

## Scope

This memo records the first reusable byte-forensics pass for the Z8
hierarchical-predictive-coding archive lane. The durable tool is
`tools/profile_z8_hpc_archive_bytes.py`; it profiles Z8HPC1 section bytes,
wavelet pair-blob internals, detail codec mode choices, optional entropy
headroom curves, local replay rate terms, ZIP wrapper overhead, and ranked
rate-axis opportunities.

Generated artifacts:

- `.omx/research/z8_hpc_rate_axis_push_20260531T220400Z_codex/z8_hpc_archive_byte_profile.json`
- `.omx/research/z8_hpc_rate_axis_push_20260531T220400Z_codex/z8_hpc_archive_byte_profile.md`
- `.omx/research/z8_hpc_rate_axis_push_20260531T220400Z_codex/z8_hpc_rd_candidate_byte_profile.json`
- `.omx/research/z8_hpc_rate_axis_push_20260531T220400Z_codex/z8_hpc_rd_candidate_byte_profile.md`
- `.omx/research/z8_hpc_rate_axis_push_20260531T220400Z_codex/z8_hpc_artifact_byte_profile_comparison.json`

## Findings

The wavelet blob size is expected under the current Z8HPC1 grammar and is also
the binding blocker for contest-rate competitiveness.

Baseline `entropy_coded_contest_rate_export`:

- `0.bin`: `28,406,255` bytes
- `archive.zip`: `28,504,909` bytes
- `wavelet_blob`: `28,376,254` bytes (`99.894%` of `0.bin`)
- `top_ll_raw_payload_bytes`: `11,059,200`
- `detail_payload_bytes`: `19,670,586`
- `detail_codec_method_counts`: `{"qi16_static_range": 7200}`
- byte ratio to current CPU frontier archive bytes (`178,493`): `159.70x`

Best profiled existing byte artifact:

- artifact: `quantized_detail_entropy_rate_probe_20260531Tlocal`
- `0.bin`: `10,195,155` bytes
- `wavelet_blob`: `10,165,099` bytes
- `top_ll_raw_payload_bytes`: `11,059,200`
- `detail_payload_bytes`: `840,770`
- `detail_codec_method_counts`: `{"qi16_zero_rle": 7200}`
- quantization step: `0.25`
- byte ratio to current CPU frontier archive bytes (`178,493`): `57.12x`

Full-video detail headroom on the baseline artifact:

| Step | Detail bytes | Structured floor | Mean MSE |
|---:|---:|---:|---:|
| `0.03125` | `9,392,460` | `9,389,418` | `8.145e-05` |
| `0.0625` | `4,538,450` | `4,536,526` | `2.111e-04` |
| `0.125` | `1,204,737` | `1,401,246` | `3.801e-04` |
| `0.25` | `293,287` | `471,717` | `5.448e-04` |
| `0.5` | `84,611` | `139,387` | `7.618e-04` |
| `1.0` | `14,944` | `25,124` | `9.809e-04` |

## Interpretation

The current wavelet blob size is not a parser bug. Z8 is storing the full-video
predictive representation in per-pair wavelet pyramids. That means almost the
entire archive is before/at the wavelet payload entropy position.

Detail quantization plus entropy coding is a real lever, but it is not enough.
Even the aggressive `q=0.25` detail artifact remains `57.12x` current frontier
bytes because the next binding term is the top-LL float payload. Repacking ZIP
or changing after-entropy layout cannot solve this scale gap.

The profiled `rd_waterfill_full600_max_weighted_mse_5e-5` candidate remains too
large:

- `archive.zip`: `24,573,973` bytes (`137.67x` current CPU frontier bytes)
- local MLX replay rate: `0.6545119974`
- rate term `25 * rate`: `16.3627999358`
- `d_seg`: `0.0019393836`
- `d_pose`: `0.0749884955`
- Pose term `sqrt(10*d_pose)`: `0.8659589800`

This is fail-closed as non-promotable local advisory signal.

## First-Principles Rate Bound

Integer fields alone do not appear sufficient. The current CPU frontier archive
is `178,493` bytes. Any Z8 variant that stores per-pair top-LL fields at even a
small number of bytes per top-LL element stays far above that scale:

- top-LL value count across both frames is `2,764,800` values;
- one byte per value is already `2.76 MB` before headers/runtime;
- four bits per value is `1.38 MB`;
- one bit per value is `345.6 KB`, still `1.94x` the frontier archive before
  detail coefficients, top-state, runtime, and ZIP overhead;
- the profiled `q=0.25` detail-collapse artifact is `10.20 MB`, `57.12x`
  frontier bytes.

Therefore the highest-EV shift is not "find a smaller integer field" but
collapse the representation itself:

- predict/proceduralize top-LL instead of storing it per pair;
- train a small decoder plus latent/selector stream as PR95/HNeRV does;
- use hierarchical predictive coding only for residuals the scorer cannot
  infer for free;
- make Z7/Mamba, Dreamer/RSSM, Wyner-Ziv side information, and cooperative
  receiver components reduce residual entropy before archive coding rather than
  add per-pair float state.

This reframes Z8/HPC from an explicit wavelet-store lane into a generative
predictive-substrate lane. The wavelet store is useful as a teacher/analysis
surface and residual actuator, but it is not itself the final contest-rate
representation unless top-LL collapses by at least one to two additional orders
of magnitude.

## Rate-Axis Opportunity Order

1. Detail coefficient RD projection:
   materialize the full-video coarse headroom rows (`0.0625`, `0.125`,
   `0.25`, `0.5`, `1.0`) through byte-closed archives and local replay. The
   allocator must be materialization-aware, because aggregate entropy-headroom
   rows can diverge from real pair-blob overhead.

2. Top-LL payload coding:
   build top-LL RD curves and materializers, but treat plain integer fields as
   a stepping stone. Candidate transforms: per-frame delta/DC quantization,
   frame1 conditional residual from frame0, spatial predictive coding,
   Wyner-Ziv side-information residual coding, and section entropy coding. This
   is the next binding surface after detail collapse.

3. Representation collapse:
   train/proceduralize top-LL and detail residuals via a tiny decoder plus
   latent/selector stream, PR95-style HNeRV/RNeRV/Predictive-NeRV, or Z8 stack
   members that actually reduce per-pair payload. This is likely required for
   contest-rate competitiveness.

4. Global/section-level coding:
   current pair-local blobs pay repeated table/context overhead and cut off
   cross-pair entropy context. The apples-to-apples q=11 solid raw-pair brotli
   recheck is positive on the baseline (`-402,311` bytes versus independent
   pair blobs). This does not fix the rate axis by itself, but it is real and
   should become a receiver-proven section-level pair-blob materializer with an
   indexed seek table or streaming decoder.

5. Runtime/custody elision:
   non-wavelet sections are secondary (`~30 KB` on the baseline). Optimize them
   only after wavelet payload moves by one to two orders of magnitude.

## Durable Blocker

Z8/HPC cannot be exact-auth promoted until a byte-closed local artifact gets
near current frontier byte scale and preserves plausible full-video local
distortion. Current best profiled byte artifact is still `57.12x` frontier
bytes. The next promotion gate is therefore payload grammar work, not scorer
dispatch.

## Validation

- `.venv/bin/python -m ruff check tools/profile_z8_hpc_archive_bytes.py src/tac/tests/test_profile_z8_hpc_archive_bytes.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_profile_z8_hpc_archive_bytes.py`
