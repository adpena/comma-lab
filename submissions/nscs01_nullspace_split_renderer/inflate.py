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

This checked-in shim is a research/dev surface. A promoted packet must vendor
``src/`` beside this file; falling back to the repository tree would not be a
self-contained contest submission.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
vendored_src = HERE / "src"
if not vendored_src.is_dir():
    raise RuntimeError(
        "NSCS01 submission runtime is not packaged: missing vendored src/. "
        "Use the trainer-emitted submission_dir artifact before any contest run."
    )
sys.path.insert(0, str(vendored_src))

from tac.substrates.nscs01_nullspace_split_renderer.inflate import main_cli  # noqa: E402


def main() -> int:
    return main_cli()


if __name__ == "__main__":
    sys.exit(main())
