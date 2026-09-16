#!/bin/sh
# Compile the three C files, then decode. A missing or failing compiler stops here.
set -eu
base=$(dirname "$0")
rm -f "$base/range_decoder.so" "$base/corrector.so" "$base/geometry.so"
command -v cc >/dev/null 2>&1 || { echo "inflate.sh: no C compiler on PATH; this decoder needs cc" >&2; exit 1; }
cc -O2 -std=c11 -shared -fPIC "$base/range_decoder.c" -lm -o "$base/range_decoder.so"
cc -O2 -std=c11 -ffp-contract=off -fno-fast-math -shared -fPIC "$base/corrector.c" -lm -o "$base/corrector.so"
cc -O2 -std=c11 -ffp-contract=off -fno-fast-math -shared -fPIC "$base/geometry.c" -o "$base/geometry.so"
exec python3 "$base/inflate.py" "$@"
