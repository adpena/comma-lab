#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build PR95-family HNeRV residual-atom planning artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr95_residual_atoms import (
    PR95AtomPlanError,
    emit_plan,
    main,
    sha256_file,
)

__all__ = ["PR95AtomPlanError", "emit_plan", "main", "sha256_file"]

if __name__ == "__main__":
    raise SystemExit(main())
