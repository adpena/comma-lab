#!/usr/bin/env python3
"""Checkpoint ensemble — thin wrapper around ``tac ensemble``."""
import sys; from tac.cli import main; sys.exit(main(["ensemble"] + sys.argv[1:]) or 0)
