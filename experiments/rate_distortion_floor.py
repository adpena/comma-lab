#!/usr/bin/env python3
"""Empirical rate/distortion floor — thin wrapper around ``tac rd-floor``."""
import sys; from tac.cli import main; sys.exit(main(["rd-floor"] + sys.argv[1:]) or 0)
