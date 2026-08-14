# Rebuild instructions

The runner builds twice and requires byte-identical outputs before measurement:

```sh
.venv/bin/python experiments/ddm_f26q_rc64_native_lowering.py build
```

The portable Linux receiver build used by `inflate.sh` is:

```sh
cc -O3 -march=native -std=c11 -shared -fPIC \
  -ffp-contract=off -fno-fast-math -fopenmp \
  runtime-rs/native/f26-hpac/f26_hpac_native.c -lm \
  -o f26_hpac_native.so
```

On Apple Silicon, replace `-march=native` with `-mcpu=native`. The local runner
links the same OpenMP runtime already loaded by PyTorch, rewrites its dependency
to `@rpath/libomp.dylib`, removes the output-name-dependent signature, and
ad-hoc signs both repeats with the fixed identifier `f26_hpac_native`. Exact
argv, compiler identity, source SHA-256, output SHA-256, and the RC64 reference
build are retained in `receipts/native_build_v13.json`.
