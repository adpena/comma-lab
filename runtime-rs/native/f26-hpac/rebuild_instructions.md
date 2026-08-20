# Rebuild instructions

The receiver is compiled at decode time by `inflate.sh`. It links `libSystem`
(macOS) / libc (Linux) only — the OpenMP dependency was removed with the pthread
pool, so neither `brew --prefix libomp` nor `-fopenmp` is needed any more.

## Dispatched build

```sh
cc -O3 -march=native -std=c11 -shared -fPIC \
  -ffp-contract=off -fno-fast-math \
  runtime-rs/native/f26-hpac/f26_hpac_native.c -lm \
  -o f26_hpac_native.so
```

On Apple Silicon replace `-march=native` with `-mcpu=native`.

`-ffp-contract=off` and `-fno-fast-math` are NOT optional and NOT decoration.
The receiver's float path must stay IEEE: FMA contraction changes rounding, and
`-ffast-math` licenses reassociation. Both would desynchronise a decoder whose
identity rests on exact `+ - * /`.

`-march=native` is a build-host optimisation, not the ISA gate. The gate is
`__builtin_cpu_supports` at runtime, so a library built elsewhere and shipped
still selects a legal kernel. If `-march=native` is rejected by the host
compiler, drop it: the plain build is correct and the runtime dispatch still
selects AVX2 where the CPU has it.

## Portable identity twin

```sh
cc -O3 -std=c11 -shared -fPIC -ffp-contract=off -fno-fast-math \
  -DF26_FORCE_SCALAR=1 \
  runtime-rs/native/f26-hpac/f26_hpac_native.c -lm \
  -o f26_hpac_native_scalar.so
```

The twin compiles no intrinsics at all. Any disagreement between it and the
dispatched build is a refusal, whatever the speed.

## Runtime knobs

| variable | default | effect |
|---|---|---|
| `F26_HPAC_THREADS` | `4` | pthread pool width, clamped to `[1, 32]`. Decoded bytes are independent of it; only wall clock changes. |
| `F26_HPAC_NATIVE_LIBRARY` | — | path to the built library; required by the split decoder. |

## Verification

```sh
# Full-field identity + scalar-twin + thread-independence re-check.
.venv/bin/python experiments/ddm_wc2c_python_reference_equivalence_test.py

# Regenerate the receipts it reads (about 20 minutes on an M5 Max).
bash /Volumes/APDataStore/pact/ddm_wc2/work/run_n600_proof.sh
```

Both builds are compiled twice and required to be byte-identical before any
measurement is quoted.
