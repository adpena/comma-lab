#!/Users/adpena/Projects/pact/.venv/bin/python
"""Verified native build-cache adapter for resuming an RLC1 public checkpoint.

This is a compiler cache, not a compiler. It reuses only the exact source/flag
builds retained from the original literal public shell. Any other request fails.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cache = Path(os.environ["RLC1_NATIVE_CACHE"]).resolve()
    manifest = json.loads((cache / "MANIFEST.json").read_text())
    if manifest["schema"] != "rlc1_native_build_cache.v1":
        raise ValueError("unknown native cache schema")
    argv = sys.argv[1:]
    if len(argv) < 2 or argv[-2] != "-o":
        raise ValueError("unrecognized compile request")
    destination = Path(argv[-1]).resolve()
    scratch = Path(os.environ["TMPDIR"]).resolve()
    if not destination.is_relative_to(scratch):
        raise ValueError("compile destination left current scratch")
    matched = [row for row in manifest["rows"] if row["argv_template"] == [*argv[:-1], "{output}"]]
    if len(matched) != 1:
        raise ValueError("compile flags/source differ from retained build")
    row = matched[0]
    source = Path(row["source"]["path"])
    library = Path(row["retained_library"]["path"])
    if not library.resolve().is_relative_to(cache):
        raise ValueError("native cache library outside owned directory")
    if digest(source) != row["source"]["sha256"] or digest(library) != row["retained_library"]["sha256"]:
        raise ValueError("source or native build drift")
    payload = library.read_bytes()
    temporary = destination.with_suffix(destination.suffix + ".new")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    receipt = {
        "source": row["source"],
        "retained_library": row["retained_library"],
        "requested_argv": argv,
        "output_sha256": digest(destination),
        "operation": "verified compiler cache hit; no compilation performed",
    }
    destination.with_suffix(destination.suffix + ".cache.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
