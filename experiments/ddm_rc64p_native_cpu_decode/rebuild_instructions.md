# Rebuild instructions

Compile the generic decoder with the host C compiler:

```sh
cc -O3 -std=c11 -shared -fPIC -ffp-contract=off -fno-fast-math \
  $(test "$(uname -s)" = Darwin && printf '%s' \
    '-Wl,-install_name,@rpath/liblc2_ans.dylib') \
  ans_backend.c -o liblc2_ans.so
```

The shipping variant performs this build in a temporary directory at inflate
time.  `LC2_NATIVE_ANS_MODE=auto` falls back to the pinned constriction decoder
on a typed compile/import failure.  Parity and benchmark runs use
`LC2_NATIVE_ANS_MODE=required`, which refuses fallback.

The entrypoint resolves `cc` before building. Required mode exits 72 when the
compiler is absent; automatic mode emits a typed fallback receipt. The
`LC2_NATIVE_COMPILE_SMOKE_ONLY=1` admission path builds and loads the library
without importing constriction.

The native build cache is a sibling of `PR130_RUNTIME_DEPS_DIR`, never a child,
so creating it cannot make the inherited Python dependency bootstrap mistake a
fresh target for a pre-existing invalid installation.

The Darwin-only link option uses a path-independent install name. Reproducible
build verification compiles twice at the same canonical output path so the
required Mach-O UUID is stable. GNU linkers retain their content-derived build
ID.

Route B first concatenates the granted PR135 `rc64_backend.c` and the literal
`RC64_CHECKPOINT_EXTENSION` from `route_b_rc64.py`, then builds it with the same
compiler discipline:

```sh
cc -O3 -std=c11 -shared -fPIC -ffp-contract=off -fno-fast-math \
  rc64_backend.c -o liblc2_rc64.so
```

The wrapper resolves and compiles this library beside the ANS library. An
explicit `R6D1` archive field fails closed when the RC64 library cannot be
built or loaded; it never falls back to constriction under a different wire.
`R6C1` is the lossless decoder-checkpoint envelope used by the existing lc2
progress cache. The retained Route-B encoder checkpoints are distinct files at
25-frame boundaries and never overwrite an earlier stage.

For the evaluated cached-plan HPAC formulation, copy
`hpac_integer_sparse_optimized.py` into the runtime as
`hpac_integer_sparse.py`. Run `optimized-parity-smoke` before any full cell;
it compares every selected logit in canonical frame 0 against the settled
runtime. Promotion still requires the full n600 decoded-token SHA and empty
coder state. This exact formulation is retained for reproduction but refused
for production routing by the measured timing cells.
