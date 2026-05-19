#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compatibility wrapper for the submission inflate.py LOC budget audit."""

from __future__ import annotations

try:
    from tools.audit_submission_inflate_py_loc_budget import main
except ModuleNotFoundError:  # pragma: no cover - direct script execution.
    from audit_submission_inflate_py_loc_budget import main


if __name__ == "__main__":
    raise SystemExit(main())

