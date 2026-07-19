#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Dedicated entrypoint for the streamed C2 integer-plane band trainer."""

from tac.boundary_math.integer_plane_banded_trainer import main


def entrypoint() -> int:
    """Run the dedicated trainer without adding any launch authority."""

    return main()


if __name__ == "__main__":
    raise SystemExit(entrypoint())
