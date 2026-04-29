#!/usr/bin/env python3
# no-argparse-OK: thin shim — argparse defined in tac.deploy.vastai.cli.main()
"""Vast.ai deployment CLI. See src/tac/deploy/vastai/ for implementation."""
from tac.deploy.vastai.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
