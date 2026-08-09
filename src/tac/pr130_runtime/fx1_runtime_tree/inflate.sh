#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYBIN=${PYTHON:-python3}
EXPECTED_CONSTRICTION_VERSION=0.5.0
DEPS_DIR=${PR130_RUNTIME_DEPS_DIR:-"$SCRIPT_DIR/.runtime-deps"}

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$DEPS_DIR:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# CONTEST-RUNTIME-PROVIDED dependencies. `inflate.py` and its siblings import numpy and
# torch as well as constriction; upstream/pyproject.toml declares numpy and upstream/uv.lock
# pins torch 2.10.0+cpu, so the contest eval host has both by construction (evaluate.py runs
# torch scorers). We therefore ASSERT them rather than install them — self-installing torch
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

dependency_ready() {
    "$PYBIN" - "$EXPECTED_CONSTRICTION_VERSION" <<'PY'
import importlib.metadata
import sys

import constriction

expected = sys.argv[1]
actual = importlib.metadata.version("constriction")
if actual != expected:
    raise SystemExit(
        f"constriction version mismatch: expected {expected}, resolved {actual}"
    )

queue = constriction.stream.queue
model = constriction.stream.model
for owner, name in (
    (queue, "RangeDecoder"),
    (model, "Categorical"),
):
    if not hasattr(owner, name):
        raise SystemExit(f"constriction {actual} lacks required API {name}")
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
        echo "PR130 requires constriction==$EXPECTED_CONSTRICTION_VERSION, but uv is unavailable." >&2
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

    "$UVBIN" pip install \
        --python "$PYBIN" \
        --target "$INSTALL_TMP" \
        --no-deps \
        --only-binary constriction \
        "constriction==$EXPECTED_CONSTRICTION_VERSION"

    PYTHONPATH="$INSTALL_TMP:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYBIN" - "$EXPECTED_CONSTRICTION_VERSION" <<'PY'
import importlib.metadata
import sys

import constriction

expected = sys.argv[1]
actual = importlib.metadata.version("constriction")
if actual != expected:
    raise SystemExit(
        f"installed constriction version mismatch: expected {expected}, got {actual}"
    )
for owner, name in (
    (constriction.stream.queue, "RangeDecoder"),
    (constriction.stream.model, "Categorical"),
):
    if not hasattr(owner, name):
        raise SystemExit(f"installed constriction {actual} lacks required API {name}")
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
    exec "$PYBIN" - "$EXPECTED_CONSTRICTION_VERSION" <<'PY'
import importlib.metadata
import sys

import constriction
import inflate

expected = sys.argv[1]
actual = importlib.metadata.version("constriction")
if actual != expected:
    raise SystemExit(f"entrypoint smoke resolved {actual}, expected {expected}")
if inflate.constriction is not constriction:
    raise SystemExit("inflate.py did not bind the verified constriction module")
print(f"PR130_DEPENDENCY_READY constriction={actual} receiver=inflate.py")
PY
fi

cd -- "$SCRIPT_DIR"
exec "$PYBIN" inflate.py "$@"
