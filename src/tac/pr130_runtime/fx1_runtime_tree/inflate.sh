#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYBIN=${PYTHON:-python3}
EXPECTED_CONSTRICTION_VERSION=0.5.0
EXPECTED_BROTLI_VERSION=1.2.0
MODEL_SELECTOR_SHIFT=29
SPLIT_BROTLI_SELECTOR=1
DEPS_DIR=${PR130_RUNTIME_DEPS_DIR:-"$SCRIPT_DIR/.runtime-deps"}

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$DEPS_DIR:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# CONTEST-RUNTIME-PROVIDED dependencies. `inflate.py` and its siblings import numpy and
# torch as well as constriction; upstream/pyproject.toml declares numpy and upstream's selected
# cpu/cu126/cu128/cu130 dependency group provides torch, so the contest eval host has both by
# construction (evaluate.py runs torch scorers). We therefore ASSERT rather than install them —
# self-installing torch
# would pull ~2 GB inside the 30-minute whole-job budget. Absence is a NAMED, fail-closed
# error (exit 68), never a raw ModuleNotFoundError from deep inside the receiver, and never
# a silent degradation. Measured on Linux x86_64 by FX5 2026-08-09.
assert_provided_deps() {
    "$PYBIN" - <<'PY'
import importlib.util
import sys

missing = [name for name in ("numpy", "torch") if importlib.util.find_spec(name) is None]
if missing:
    sys.stderr.write(
        "PR130 runtime closure: contest-runtime dependencies absent: "
        + ", ".join(missing)
        + "\nThese are declared 'contest_runtime_provided_asserted_not_installed' in "
        "runtime-dependencies.json and are NEVER installed by this entrypoint.\n"
    )
    raise SystemExit(68)
PY
}

if ! assert_provided_deps; then
    exit 68
fi

MODEL_CODEC=unknown
NEEDS_BROTLI=1
if [ "${PR130_DEPENDENCY_SMOKE_ONLY:-0}" != 1 ] && [ -n "${1:-}" ] && [ -f "$1/p" ]; then
    MODEL_SELECTION=$(
        "$PYBIN" - "$1/p" "$MODEL_SELECTOR_SHIFT" "$SPLIT_BROTLI_SELECTOR" <<'PY'
import pathlib
import lzma
import struct
import sys

payload = pathlib.Path(sys.argv[1]).read_bytes()
if len(payload) < 4:
    raise SystemExit("PR130 dependency selection: payload is truncated before model_word")
model_word = struct.unpack_from("<I", payload)[0]
selector = (model_word >> int(sys.argv[2])) & 0b11
codecs = {0: "legacy_lzma", 1: "split_brotli", 2: "split_lzma2"}
if selector not in codecs:
    raise SystemExit("PR130 dependency selection: reserved model-codec selector 3")
needs_brotli = selector == int(sys.argv[3])

# PZ3R embeds Brotli-compressed PZ2 target streams inside the carrier even when
# the outer model bundle remains legacy XZ or split raw-LZMA2. Detect that wire
# tag before dependency provisioning. Malformed synthetic probes retain the
# outer-codec answer and fail closed later in the real receiver.
model_bytes = model_word & ((1 << 29) - 1)
models = payload[4:4 + model_bytes]
try:
    carrier = b""
    if selector == 0 and models.startswith(b"\xfd7zXZ\x00"):
        raw = lzma.decompress(models, format=lzma.FORMAT_XZ)
        semantic_bytes, carrier_bytes = struct.unpack_from("<II", raw)
        start = 8 + semantic_bytes
        carrier = raw[start:start + carrier_bytes]
    elif selector == 2 and len(models) >= 12:
        semantic_bytes, carrier_bytes, hpac_bytes = struct.unpack_from("<III", models)
        if 12 + semantic_bytes + carrier_bytes + hpac_bytes == len(models):
            start = 12 + semantic_bytes
            compressed_carrier = models[start:start + carrier_bytes]
            carrier = lzma.decompress(
                compressed_carrier,
                format=lzma.FORMAT_RAW,
                filters=[{
                    "id": lzma.FILTER_LZMA2,
                    "preset": 9 | lzma.PRESET_EXTREME,
                }],
            )
    needs_brotli = needs_brotli or carrier.startswith(b"PZ3R")
except (lzma.LZMAError, struct.error, ValueError):
    pass
print(codecs[selector], int(needs_brotli))
PY
    )
    MODEL_CODEC=${MODEL_SELECTION% *}
    NEEDS_BROTLI=${MODEL_SELECTION##* }
fi

if [ "${PR130_DEPENDENCY_SELECTION_ONLY:-0}" = 1 ]; then
    echo "PR130_DEPENDENCY_SELECTION model_codec=$MODEL_CODEC needs_brotli=$NEEDS_BROTLI"
    exit 0
fi

dependency_ready() {
    "$PYBIN" - "$EXPECTED_CONSTRICTION_VERSION" "$NEEDS_BROTLI" "$EXPECTED_BROTLI_VERSION" <<'PY'
import importlib
import importlib.metadata
import sys

import constriction

needs_brotli = sys.argv[2] == "1"
expected = {"constriction": sys.argv[1]}
if needs_brotli:
    expected["Brotli"] = sys.argv[3]
for package, wanted in expected.items():
    actual = importlib.metadata.version(package)
    if actual != wanted:
        raise SystemExit(
            f"{package} version mismatch: expected {wanted}, resolved {actual}"
        )

queue = constriction.stream.queue
stack = constriction.stream.stack
model = constriction.stream.model
required_apis = [
    (queue, "RangeDecoder"),
    (stack, "AnsCoder"),
    (model, "Categorical"),
]
if needs_brotli:
    required_apis.append((importlib.import_module("brotli"), "decompress"))
for owner, name in required_apis:
    if not hasattr(owner, name):
        raise SystemExit(f"runtime dependency lacks required API {name}")
for name in ("encode_reverse", "is_empty"):
    if not hasattr(stack.AnsCoder(), name):
        raise SystemExit(f"constriction AnsCoder lacks required API {name}")
PY
}

if ! dependency_ready >/dev/null 2>&1; then
    if [ -e "$DEPS_DIR" ]; then
        echo "PR130 dependency closure refused invalid existing target: $DEPS_DIR" >&2
        echo "Select a fresh PR130_RUNTIME_DEPS_DIR; this entrypoint will not overwrite it." >&2
        exit 65
    fi

    UVBIN=${UV:-}
    if [ -z "$UVBIN" ]; then
        UVBIN=$(command -v uv || true)
    fi
    if [ -z "$UVBIN" ] || [ ! -x "$UVBIN" ]; then
        if [ "$NEEDS_BROTLI" = 1 ]; then
            echo "PR130 requires constriction==$EXPECTED_CONSTRICTION_VERSION and Brotli==$EXPECTED_BROTLI_VERSION, but uv is unavailable." >&2
        else
            echo "PR130 requires constriction==$EXPECTED_CONSTRICTION_VERSION, but uv is unavailable." >&2
        fi
        exit 69
    fi

    DEPS_PARENT=$(dirname -- "$DEPS_DIR")
    mkdir -p -- "$DEPS_PARENT"
    INSTALL_TMP=$(mktemp -d "${DEPS_DIR}.tmp.XXXXXX")
    cleanup_install_tmp() {
        if [ -n "${INSTALL_TMP:-}" ] && [ -d "$INSTALL_TMP" ]; then
            rm -rf -- "$INSTALL_TMP"
        fi
    }
    trap cleanup_install_tmp EXIT HUP INT TERM

    if [ "$NEEDS_BROTLI" = 1 ]; then
        "$UVBIN" pip install \
            --python "$PYBIN" \
            --target "$INSTALL_TMP" \
            --no-deps \
            --only-binary :all: \
            "constriction==$EXPECTED_CONSTRICTION_VERSION" \
            "Brotli==$EXPECTED_BROTLI_VERSION"
    else
        "$UVBIN" pip install \
            --python "$PYBIN" \
            --target "$INSTALL_TMP" \
            --no-deps \
            --only-binary :all: \
            "constriction==$EXPECTED_CONSTRICTION_VERSION"
    fi

    PYTHONPATH="$INSTALL_TMP:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYBIN" - "$EXPECTED_CONSTRICTION_VERSION" "$NEEDS_BROTLI" "$EXPECTED_BROTLI_VERSION" <<'PY'
import importlib
import importlib.metadata
import sys

import constriction

needs_brotli = sys.argv[2] == "1"
expected = {"constriction": sys.argv[1]}
if needs_brotli:
    expected["Brotli"] = sys.argv[3]
for package, wanted in expected.items():
    actual = importlib.metadata.version(package)
    if actual != wanted:
        raise SystemExit(
            f"installed {package} version mismatch: expected {wanted}, got {actual}"
        )
required_apis = [
    (constriction.stream.queue, "RangeDecoder"),
    (constriction.stream.stack, "AnsCoder"),
    (constriction.stream.model, "Categorical"),
]
if needs_brotli:
    required_apis.append((importlib.import_module("brotli"), "decompress"))
for owner, name in required_apis:
    if not hasattr(owner, name):
        raise SystemExit(f"installed runtime dependency lacks required API {name}")
for name in ("encode_reverse", "is_empty"):
    if not hasattr(constriction.stream.stack.AnsCoder(), name):
        raise SystemExit(f"installed constriction AnsCoder lacks required API {name}")
PY

    mv -- "$INSTALL_TMP" "$DEPS_DIR"
    INSTALL_TMP=
    trap - EXIT HUP INT TERM
fi

if ! dependency_ready; then
    echo "PR130 dependency verification failed after bootstrap." >&2
    exit 70
fi

if [ "${PR130_DEPENDENCY_SMOKE_ONLY:-0}" = 1 ]; then
    cd -- "$SCRIPT_DIR"
    exec "$PYBIN" - "$EXPECTED_CONSTRICTION_VERSION" "$EXPECTED_BROTLI_VERSION" <<'PY'
import importlib.metadata
import sys

import brotli
import constriction
import inflate
import receiver

expected_constriction, expected_brotli = sys.argv[1:]
actual_constriction = importlib.metadata.version("constriction")
actual_brotli = importlib.metadata.version("Brotli")
if actual_constriction != expected_constriction:
    raise SystemExit(
        f"entrypoint smoke resolved constriction {actual_constriction}, "
        f"expected {expected_constriction}"
    )
if actual_brotli != expected_brotli:
    raise SystemExit(
        f"entrypoint smoke resolved Brotli {actual_brotli}, expected {expected_brotli}"
    )
if inflate.constriction is not constriction:
    raise SystemExit("inflate.py did not bind the verified constriction module")
if receiver.constriction is not constriction or receiver.brotli is not brotli:
    raise SystemExit("receiver.py did not bind the verified dependency modules")
print(
    "PR130_DEPENDENCY_READY "
    f"constriction={actual_constriction} Brotli={actual_brotli} "
    "receiver=inflate.py+receiver.py"
)
PY
fi

if [ "$#" -ne 3 ]; then
    echo "usage: inflate.sh <archive-dir> <output-dir> <video-names-file>" >&2
    exit 64
fi

ARCHIVE_DIR=$1
OUTPUT_DIR=$2
VIDEO_NAMES_FILE=$3
mkdir -p -- "$OUTPUT_DIR"

cd -- "$SCRIPT_DIR"
while IFS= read -r video_name || [ -n "$video_name" ]; do
    [ -n "$video_name" ] || continue
    base=${video_name%.*}
    "$PYBIN" inflate.py "$ARCHIVE_DIR" "$base" "$OUTPUT_DIR/$base.raw"
done < "$VIDEO_NAMES_FILE"
