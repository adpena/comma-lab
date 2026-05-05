#!/usr/bin/env python3
"""CLI wrapper for comma-lab research-state tracking audits."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from comma_lab.research_state import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
