#!/usr/bin/env python3
"""Per-video CRF optimizer — thin wrapper around ``tac crf-search``."""
import sys; from tac.cli import main; sys.exit(main(["crf-search"] + sys.argv[1:]) or 0)
