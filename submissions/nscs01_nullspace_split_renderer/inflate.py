#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""NSCS01 contest-compliant inflate runtime entry-point.

Delegates to the vendored substrate CLI. No scorer-network imports
(strict-scorer-rule contract). Per Catalog #205 the device select is via
the canonical ``select_inflate_device(...)`` helper.

The submission directory layout is built by the trainer's ``_write_runtime``
function, which copies the substrate package + the canonical
``_shared/inflate_runtime.py`` helper into ``submission_dir/src/`` so this
shim's ``sys.path.insert`` reaches the vendored package.

For local development this submissions/ shim defers to the live src/ tree;
for contest dispatch the trainer-emitted submission_dir/ shim is the actual
artifact.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Try vendored src/ first (contest-dispatch layout); fall back to repo src/.
vendored_src = HERE / "src"
if vendored_src.is_dir():
    sys.path.insert(0, str(vendored_src))
else:
    repo_src = HERE.parent.parent / "src"
    if repo_src.is_dir():
        sys.path.insert(0, str(repo_src))

from tac.substrates.nscs01_nullspace_split_renderer.inflate import main_cli  # noqa: E402


def main() -> int:
    return main_cli()


if __name__ == "__main__":
    sys.exit(main())
