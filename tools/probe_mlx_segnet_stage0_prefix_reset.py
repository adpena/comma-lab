#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""Compatibility wrapper for the generalized SegNet prefix-reset probe."""

from __future__ import annotations

from probe_mlx_segnet_prefix_reset import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
