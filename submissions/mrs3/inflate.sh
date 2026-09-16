#!/bin/sh
set -eu
base=$(dirname "$0")
rm -f "$base/range_decoder.so"
cc -O2 -std=c11 -shared -fPIC "$base/range_decoder.c" -o "$base/range_decoder.so" || true
exec python3 "$base/inflate.py" "$@"
