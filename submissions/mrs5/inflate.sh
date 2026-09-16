#!/bin/sh
set -eu
base=$(dirname "$0")
rm -f "$base/range_decoder.so" "$base/corrector.so" "$base/geometry.so"
cc -O2 -std=c11 -shared -fPIC "$base/range_decoder.c" -o "$base/range_decoder.so" || true
cc -O2 -std=c11 -ffp-contract=off -fno-fast-math -shared -fPIC "$base/corrector.c" -lm -o "$base/corrector.so" || true
cc -O2 -std=c11 -ffp-contract=off -fno-fast-math -shared -fPIC "$base/geometry.c" -o "$base/geometry.so" || true
exec python3 "$base/inflate.py" "$@"
