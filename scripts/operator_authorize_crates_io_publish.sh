#!/bin/bash
# Thin shim: delegates to the canonical operator-authorize entry point.
#
# Recipe: ``.omx/operator_authorize_recipes/crates_io_publish.yaml``
#
# Per simplification audit `a2a901c4f43d66a74` (operator approved 2026-05-12)
# + Catalog #162 ``check_operator_authorize_canonical_use``.
#
# Lane: lane_fix_g_t1c_wrapper_unification_20260512
# Cross-ref: feedback_fix_g_t1c_wrapper_unification_landed_20260512.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Phase 1: canonical entry point — recipe banner only. This platform=none
# wrapper owns the real action prompt and must not create phantom claims.
.venv/bin/python tools/operator_authorize.py \
    --recipe crates_io_publish \
    --agent "claude:operator_authorize_crates_io_publish" \
    --no-claim \
    --dry-run \
    "$@" || exit $?

# Phase 2: bespoke action sequence preserved for back-compat.
CRATE_DIR="${CRATE_DIR:-runtime-rs/tac-packet-compiler}"
if [ ! -d "$CRATE_DIR" ]; then
    for candidate in runtime-rs/crates/tac-packet-compiler runtime-rs/tac_packet_compiler tac-packet-compiler; do
        if [ -d "$candidate" ]; then
            CRATE_DIR="$candidate"
            break
        fi
    done
fi

if [ ! -d "$CRATE_DIR" ] || [ ! -f "${CRATE_DIR}/Cargo.toml" ]; then
    echo "[crates-publish] FATAL: tac-packet-compiler crate not found." >&2
    echo "  searched: runtime-rs/tac-packet-compiler, runtime-rs/crates/tac-packet-compiler, etc." >&2
    echo "  override with CRATE_DIR=<path>." >&2
    exit 1
fi

VERSION=$(grep -E '^version = ' "${CRATE_DIR}/Cargo.toml" | head -1 | sed 's/version = "\(.*\)"/\1/')
echo "[crates-publish] crate dir:  ${CRATE_DIR}"
echo "[crates-publish] version:    ${VERSION}"

read -r -p "Run 'cargo publish --dry-run' first? [Y/n] " dryrun
case "$dryrun" in
    [nN]|[nN][oO])
        ;;
    *)
        (cd "$CRATE_DIR" && cargo publish --dry-run)
        ;;
esac

read -r -p "Proceed with 'cargo publish' (REAL publish — IMMUTABLE)? [y/N] " confirm
case "$confirm" in
    [yY]|[yY][eE][sS])
        (cd "$CRATE_DIR" && cargo publish)
        echo "[crates-publish] complete — tac-packet-compiler ${VERSION} published to crates.io"
        echo "  verify at https://crates.io/crates/tac-packet-compiler/${VERSION}"
        ;;
    *)
        echo "[crates-publish] aborted — no publish"
        exit 0
        ;;
esac
