"""Load config.env files into os.environ and return as dict.

Both compress.sh and inflate.sh source config.env as shell variables.
This module provides the same functionality for Python Click CLIs,
so `--config config.env` loads all settings before Click processes
env var bindings.

Format: KEY=VALUE lines (no export, no quotes needed). Comments with #.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_config_env(path: str | Path = "config.env", into_environ: bool = True) -> dict[str, str]:
    """Load a config.env file and optionally inject into os.environ.

    Args:
        path: Path to the config.env file.
        into_environ: If True, set values in os.environ (so Click envvar=
            bindings pick them up). Existing env vars take precedence.

    Returns:
        Dict of key-value pairs from the file.
    """
    config: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return config

    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Handle optional 'export ' prefix
        if line.startswith("export "):
            line = line[7:]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # Strip optional quotes around value
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        config[key] = value

        if into_environ and key not in os.environ:
            os.environ[key] = value

    return config
