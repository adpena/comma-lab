# Binary/source boundary audit

Verdict: **PASS** for the `ddm_wc2c` dispatched and scalar-twin receiver binaries.

## Boundary

- The C translation unit implements only generic integer HPAC, probability,
  incremental causal-state, and RC64 algorithms, plus a pthread worker pool and a
  runtime ISA dispatch. All of it is decoder algorithm.
- Learned weights, affine codes, residual values, group plans, frame codes,
  context-convolution weights, conv-A deltas, and stream bytes enter through
  pointers populated from `archive.zip` by the Python binding. Nothing
  video-derived is compiled in.
- `strings` on both built binaries found none of the jg5 archive, token,
  corrected-logit, corrected-CDF or raw SHA prefixes, nor the historical MC36
  token prefix. Long-hex scan of the C source: 0 matches.
- `otool -L` on both binaries: `/usr/lib/libSystem.B.dylib` only. The OpenMP
  dependency is gone.

Details and exact commands: `embedded_constants_audit.txt`.

## What the lowering changes, and what it does not

The `ddm_wc2c` split surface changes **receiver execution only**. Proven on the
full n600 field of the jg5 pointer body (archive `f3bce5d2…`, 180,625 B) against
its `[contest-CUDA T4]` receipt — a local macOS-arm64 decode reproducing a T4
receipt exactly:

| anchor | expected (T4 receipt) | measured (split, local) |
|---|---|---|
| `decoded_token_sha256` | `cc10a7b0…636efb` | MATCH |
| `corrected_quantized_logit_sha256` | `8269fe1a…eec4dd` | MATCH |
| `corrected_cdf_input_sha256` | `370a5e2a…e46000` | MATCH |
| `decoder_bit_position` | `910837` | MATCH |

It does not change archive bytes, decoded tokens, rendered frames, scorer inputs,
or score authority.

## What the audit does NOT cover — stated, not omitted

- **x86.** The AVX2 kernels and the `__builtin_cpu_supports` selection are
  compiled and reviewed but **UNVERIFIED at runtime**: no x86 host was reachable
  from this arm. The scalar row is the fail-closed floor if the dispatch is ever
  wrong, and the equivalence test would catch a disagreement — but it has not
  been run on x86.
- **The T4 (sm_75) box.** No shipping-axis measurement of the split path exists.
  Every seconds figure here is `[macOS-CPU advisory]`.
- **The fused surface's historical receipts.** The MC36 F26R receipts
  (`f0ba4bb4…`, token `9ba2e52b…`) predate the corrector generation that made the
  refusal necessary. They are historical provenance for the fused path, not
  evidence about the split path, and are not cited as such.
