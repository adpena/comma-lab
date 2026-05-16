#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""NSCS01 contest-compliant inflate runtime entry-point.

Delegates to the vendored substrate CLI. No scorer-network imports
(strict-scorer-rule contract). Per Catalog #205 the device select is via
the canonical ``select_inflate_device(...)`` helper.

This checked-in file is a TEMPLATE / research-dev surface. The trainer's
``_write_runtime`` overwrites it in the emitted ``submission_dir`` AND
copies the substrate codec package +
``tac.substrates._shared.inflate_runtime`` into
``submission_dir/src/tac/...`` so the trainer-emitted submission tree is
truly self-contained per CLAUDE.md HNeRV parity discipline L4 + L9 and
Catalog #295 (NSCS06 v5 bug-class anchor; commit ``0b50ceceb``).

The fail-closed ``RuntimeError`` below guarantees this checked-in template
cannot itself fire a contest dispatch — it raises BEFORE any
``sys.path.insert(...)`` can resolve onto the operator's working tree.
Only the trainer-emitted ``submission_dir/inflate.py`` (which vendors
``src/tac/...`` alongside via ``experiments/train_substrate_nscs01_...``
``_write_runtime``) actually executes inflate work.

This is the NSCS01-style fail-closed-at-runtime research surface that
Catalog #295's gate docstring explicitly names as the canonical waiver
case (see ``src/tac/preflight.py::_scan_inflate_for_pythonpath_shim_self_contained``).
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
sys.path.insert(0, str(vendored_src))  # SUBMISSION_PYTHONPATH_SHIM_OK:nscs01-checked-in-template-fail-closes-via-RuntimeError-above-when-src-missing-trainer-emitted-submission-dir-vendors-src-tac-alongside-per-write-runtime-per-catalog-295-canonical-nscs01-fail-closed-waiver

from tac.substrates.nscs01_nullspace_split_renderer.inflate import main_cli  # noqa: E402


def main() -> int:
    return main_cli()


if __name__ == "__main__":
    sys.exit(main())
