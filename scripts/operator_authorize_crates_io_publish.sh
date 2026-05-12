#!/bin/bash
# Operator authorize: publish tac-packet-compiler v0.2.0-rc1 to crates.io.
#
# Per operator F-2 in operator decision dashboard 2026-05-11: this is
# FROZEN-OPERATOR pending "don't submit PR yet" + Q5 council verdict.
#
# This script REQUIRES explicit operator confirmation + GitHub tag push
# (F-2 dependency). Per CLAUDE.md "Public Disclosure Hygiene", does NOT
# auto-publish.
#
# Per CLAUDE.md "Operator gates must be wired and used": this script is the
# canonical wrapper for the F-2 crates.io publication act.
#
# Cost: $0 (free crates.io publish).
# Risk: public-facing release. Once published, a crates.io version is
#       immutable + cannot be deleted (only YANKED). License correctness +
#       repository URL correctness are critical.
#
# Usage: bash scripts/operator_authorize_crates_io_publish.sh
#
# Lane: lane_operator_one_command_authorize_scripts L0
# Cross-ref: project_operator_decision_dashboard_20260511.md F-2

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Locate the crate directory.
CRATE_DIR="${CRATE_DIR:-runtime-rs/tac-packet-compiler}"
if [ ! -d "$CRATE_DIR" ]; then
    # Try alternate locations.
    for candidate in runtime-rs/crates/tac-packet-compiler runtime-rs/tac_packet_compiler tac-packet-compiler; do
        if [ -d "$candidate" ]; then
            CRATE_DIR="$candidate"
            break
        fi
    done
fi

if [ ! -d "$CRATE_DIR" ]; then
    echo "[crates-publish] FATAL: tac-packet-compiler crate dir not found." >&2
    echo "  searched: runtime-rs/tac-packet-compiler, runtime-rs/crates/tac-packet-compiler, etc." >&2
    echo "  override with CRATE_DIR=<path>." >&2
    exit 1
fi
if [ ! -f "${CRATE_DIR}/Cargo.toml" ]; then
    echo "[crates-publish] FATAL: ${CRATE_DIR}/Cargo.toml not found." >&2
    exit 1
fi

VERSION=$(grep -E '^version = ' "${CRATE_DIR}/Cargo.toml" | head -1 | sed 's/version = "\(.*\)"/\1/')
LICENSE=$(grep -E '^license = ' "${CRATE_DIR}/Cargo.toml" | head -1 | sed 's/license = "\(.*\)"/\1/')
REPOSITORY=$(grep -E '^repository = ' "${CRATE_DIR}/Cargo.toml" | head -1 | sed 's/repository = "\(.*\)"/\1/')

cat <<EOF

=== tac-packet-compiler crates.io publish operator confirmation ===

crate dir:               ${CRATE_DIR}
version:                 ${VERSION}
license:                 ${LICENSE}
repository:              ${REPOSITORY}

This will run (from ${CRATE_DIR}):
  cargo publish --dry-run    (pre-flight check)
  cargo publish               (actual publish — requires confirmation)

prerequisites verified:
  [ ] F-2 GitHub tag pushed (v0.2.0-rc1 visible on remote)
  [ ] LICENSE file present at repo root
  [ ] THIRD_PARTY_NOTICES.md present + current
  [ ] Cargo.toml repository URL matches GitHub remote
  [ ] cargo test passes locally
  [ ] No private infrastructure references in src/
  [ ] check_public_release_hygiene STRICT preflight has been run
  [ ] cargo login has been performed (~/.cargo/credentials.toml exists)

risk:                    crates.io versions are IMMUTABLE once published.
                         The only remediation is to yank (still visible) +
                         publish a new version. Verify license + repository
                         URL + no leaked credentials BEFORE confirming.
cost:                    \$0 (free crates.io)

EOF
read -r -p "Run 'cargo publish --dry-run' first? [Y/n] " dryrun
case "$dryrun" in
    [nN]|[nN][oO])
        ;;
    *)
        echo "[crates-publish] step 1/2: dry-run..."
        (cd "$CRATE_DIR" && cargo publish --dry-run)
        ;;
esac

read -r -p "Proceed with 'cargo publish' (REAL publish — IMMUTABLE)? [y/N] " confirm
case "$confirm" in
    [yY]|[yY][eE][sS])
        ;;
    *)
        echo "[crates-publish] aborted — no publish"
        exit 0
        ;;
esac

(cd "$CRATE_DIR" && cargo publish)
echo "[crates-publish] complete — tac-packet-compiler ${VERSION} published to crates.io"
echo "  verify at https://crates.io/crates/tac-packet-compiler/${VERSION}"
