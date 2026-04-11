#!/usr/bin/env python3
"""Benchmark mask codecs — thin wrapper around ``tac benchmark-codecs``."""
import sys; from tac.cli import main; sys.exit(main(["benchmark-codecs"] + sys.argv[1:]) or 0)
