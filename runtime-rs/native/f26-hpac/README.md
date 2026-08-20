# F26 native HPAC/RC64 receiver

This directory contains the generic C lowering and audit bundle for the F26
token receiver. The native function consumes archive-derived model codes,
residual values, causal plans, and the original RC64 stream. No learned or
video-derived value is embedded in the C source or binary.

The authoritative semantics oracle remains the lifted Python receiver.

## Two entry surfaces

**Fused (`f26_hpac_decode_frame`)** — the original F26R surface. It decodes a
whole frame internally: integer model, probability table, RC64 coder. Unchanged
in behaviour.

**Split (`f26_hpac_frame_begin` / `f26_hpac_group_logits` /
`f26_hpac_group_commit`)** — added by `ddm_wc2c`. It stops at the corrected
float rows and hands them back, so the caller keeps the probability table, the
decode-time probability corrector, and the RC64 coder in Python.

Both surfaces call the same static kernels (`f26_group_model`, `f26_row_corrected`,
`f26_group_apply`), so agreement between them is structural rather than
coincidental.

### Why the split exists

`runtime/f26_inflate.py` hard-refused `native-hpac` because the shipped
decode-time probability corrector was wired into the Python token decoder only,
and the fused surface leaves no seam for it. That refusal was correct: an
unpatched native path decodes a different field.

The corrector is **float64** on its decision path and its identity rests on IEEE
`+ - * /` in a hand-fixed summation order (`free_corrector.py:266-279`, which
deliberately refuses `sum(axis=1)`). The measured failure mode of getting that
order wrong is `S = 27.83` — a desynchronised decoder, not a rounding wobble. So
the split moves only the **integer** model, where SIMD lanes are exact and
addition is associative, and leaves the float64 half in its audited numpy form.

`ddm_wc2c` MEASURED the shipping loop before choosing this shape
(`experiments/ddm_wc2c_token_stage_profile.py`).

## Threading

The parallel regions use a pthread pool (`f26_parallel_for`), not OpenMP.

- **Correctness**: the library is loaded into a process that already loaded
  PyTorch and therefore an OpenMP runtime. On macOS that is an immediate abort;
  the documented workaround, `KMP_DUPLICATE_LIB_OK`, is explicitly unsafe.
- **Dependencies**: OpenMP put `brew --prefix libomp` on Darwin and `-fopenmp`
  on Linux into the decode-time build line. pthreads is in libc on both, so the
  built library now links `libSystem` only.
- **Determinism**: every parallel region partitions PATCHES and each patch writes
  only its own slice. No cross-thread reduction exists, so the decoded field does
  not depend on `F26_HPAC_THREADS`. The equivalence test checks this rather than
  assuming it.

## ISA gating

`AVX2 -> scalar` on x86 selected at RUNTIME by `__builtin_cpu_supports`, so a
prebuilt library shipped to unknown silicon still selects a legal path. NEON on
arm64 is architectural — the target IS the gate. `-DF26_FORCE_SCALAR=1` builds
the portable twin with no intrinsics at all; it is the identity oracle.
`f26_hpac_dispatch_path()` reports the selection, and the caller records it as
`decode_path` so a fallback is visible in the receipt rather than silent.

MEASURED honestly: on Apple M5 Max with clang 21 at `-O3 -mcpu=native`, the hand
NEON kernels are **within noise of the auto-vectorised scalar build** (n=12
prefix: 11.598 s vs 11.373 s). They are retained for the runtime x86 dispatch and
as a floor when the compiler cannot auto-vectorise, not because they were
measured faster here. The x86 AVX2 path is **UNVERIFIED** — no x86 host was
reachable from this arm.

## Identity receipts

`ddm_wc2c` proved the split path on the full n600 field of the jg5 pointer body
(archive `f3bce5d2…`, 180,625 B) against the `[contest-CUDA T4]` receipt:

| anchor | value |
|---|---|
| `decoded_token_sha256` | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` |
| `corrected_quantized_logit_sha256` | `8269fe1aad031620b18051ad784d877bc9e6e9a4a71e775e78681955c4eec4dd` |
| `corrected_cdf_input_sha256` | `370a5e2a85ccbb1e598c84333cc851f0a8c352091fde272160826b4b04e46000` |
| `decoder_bit_position` | `910837` |

Retained under `/Volumes/APDataStore/pact/ddm_wc2/retained/`. Re-check with
`experiments/ddm_wc2c_python_reference_equivalence_test.py`, which re-evaluates
the receipts rather than trusting a prior PASS marker.

The earlier F26R receipts (MC36 archive `f0ba4bb4…`, token SHA `9ba2e52b…`) are
HISTORICAL: they were produced by the fused surface on a different archive and
before the corrector generation that made the refusal necessary. They are not
evidence about the split path.
