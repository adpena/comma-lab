#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""Compatibility alias for ``tools/audit_mlx_scorer_input_cache.py``."""

from audit_mlx_scorer_input_cache import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
