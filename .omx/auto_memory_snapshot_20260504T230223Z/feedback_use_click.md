---
name: Use Click for CLI
description: User wants Click library for all CLI interfaces, not argparse or raw sys.argv
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
Use Click for all CLI interfaces. No raw sys.argv, no argparse.

**Why:** Click gives --help for free, type validation, composable commands, envvar binding, and is cleaner than argparse. The current inflate_postfilter.py uses positional sys.argv which is fragile and gives cryptic errors.

**How to apply:** `import click` + `@click.command()` + `@click.option("--archive-dir", envvar="ARCHIVE_DIR")` pattern. All env vars should also be Click options with `envvar=` binding so they work both ways. Add `uv pip install click` to setup.
