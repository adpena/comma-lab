# Binary source audit

`ans_backend.c` is original Pact glue implementing the public rANS recurrence
and constriction's public 24-bit fast-leaky categorical quantization rule. It
contains no learned weights, token values, probability tables, frame indices,
or other video-derived constants. The borrowed compile-at-decode pattern and
fallback shape come from codexblack PR135's granted runtime apparatus; no RC64
algorithm or source line is copied into the ANS implementation.

Route B deliberately borrows the arithmetic-coder recurrence verbatim from the
granted PR135 source at
`/Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book/src/cpr1_sub4/entropy/rc64_backend.c`
(12,222 bytes, SHA-256
`5c75e2c70b89f148bc9d117d4dbd39a24dfb2e72ec41b0a7e9b9cf490ca07ee6`).
`route_b_rc64.py` adds only the ctypes binding, an explicit wire discriminator,
and encoder/decoder snapshot-resume functions. It does not claim the RC64
recurrence as original work and does not change its frequency lattice.

The generated `rc64_backend.c` is the granted source followed by the disclosed
checkpoint ABI extension. Both native libraries are generic receiver code.
Neither source contains learned weights, token values, probability tables,
frame indices, or other video-derived constants. The counted ANS/RC64 token
fields remain in `archive.zip`.

`hpac_integer_sparse_optimized.py` is original Pact receiver glue. It copies
the settled public sparse-HPAC arithmetic and changes only the lifetime of
loop-invariant derived tensors: rounded weights, active-kernel views, exponent
powers, and gather indices are cached once. It contains no parameter values or
video-derived constants. The exact model weights are still parsed from the
counted archive at runtime. Full n600 parity passed, but this implementation is
not promoted because its measured wall time did not beat the control.
