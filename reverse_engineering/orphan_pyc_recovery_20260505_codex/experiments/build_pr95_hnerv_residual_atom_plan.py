# pyc-recovery: human-reconstructed from experiments/build_pr95_hnerv_residual_atom_plan.py.pyc
# This is the canonical main-repo content as of 2026-05-05.
# Recovery spec preserved at: build_pr95_hnerv_residual_atom_plan.recovery_spec.json
# Original STUB has been replaced with this canonical version.
#!/usr/bin/env python3
"""Build PR95-family HNeRV residual-atom planning artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr95_residual_atoms import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
