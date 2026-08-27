# SPDX-License-Identifier: MIT
# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""Compatibility entry point for the renamed directional-Fourier smoke."""

from __future__ import annotations

from levelset_directional_fourier_feasibility_smoke import main

if __name__ == "__main__":
    raise SystemExit(main())
