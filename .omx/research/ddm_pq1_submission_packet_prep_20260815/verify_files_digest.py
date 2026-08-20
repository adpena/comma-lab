#!/usr/bin/env python3
"""Reproduce runtime_files_sha256 for this packet from its MANIFEST rows.

Run from the runtime tree root (the submission directory -- the directory
that contains inflate.py and MANIFEST.sha256):

    python3 verify_files_digest.py [path/to/evaluate.py]

This is the executable form of the digest recipe (rv17 R5-F1: the prose
recipe was under-specifiable; this script is the recipe). It rebuilds the
"Environment-free custody digest" construction of
experiments/contest_auth_eval.py: sha256 over compact JSON

    {"files": [{"bytes": ..., "relative_path": ..., "sha256": ...}, ...],
     "upstream_evaluate_py": {"bytes": ..., "relative_path": "evaluate.py",
                              "sha256": ...}}

where every entry is a JSON object with exactly those three keys, "files"
is sorted by relative_path, serialization uses sort_keys=True and
separators=(",", ":"), and the evaluate.py entry's relative_path is the
literal string "evaluate.py" regardless of where the file lives on disk.

EXPECTED_DIGEST and EXPECTED_ROWS are pinned to THIS packet generation
(gen6, archive df7fd266...). The packet stager must regenerate both if the
runtime tree ever changes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

EXPECTED_DIGEST = "e8dcbc6542d6f4752559726a6b88bd645f5974a2d941a0bbaef6f9932dc8cb8f"
EXPECTED_ROWS = 36
EVALUATE_PY_CANDIDATES = (
    "../../evaluate.py",   # contest-repo checkout: submissions/<name>/ -> repo root
    "../evaluate.py",
    "evaluate.py",
    "upstream/evaluate.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str]) -> int:
    manifest = Path("MANIFEST.sha256")
    if not manifest.exists():
        print(
            "FAIL: MANIFEST.sha256 not found in the current directory.\n"
            "Run this script from the runtime tree root (the submission\n"
            "directory containing inflate.py and MANIFEST.sha256).",
            file=sys.stderr,
        )
        return 1

    rows = []
    for line in manifest.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        declared, rel = stripped.split(None, 1)
        path = Path(rel)
        if not path.exists():
            print(f"FAIL: manifest row missing on disk: {rel}", file=sys.stderr)
            return 1
        actual = _sha256(path)
        if actual != declared:
            print(f"FAIL: sha mismatch for {rel}\n  manifest {declared}\n  on disk  {actual}", file=sys.stderr)
            return 1
        rows.append({"relative_path": rel, "bytes": path.stat().st_size, "sha256": actual})

    if len(rows) != EXPECTED_ROWS:
        print(f"FAIL: expected {EXPECTED_ROWS} manifest rows, found {len(rows)}", file=sys.stderr)
        return 1

    if len(argv) > 1:
        evaluate_py = Path(argv[1])
    else:
        evaluate_py = next((Path(c) for c in EVALUATE_PY_CANDIDATES if Path(c).exists()), None)
    if evaluate_py is None or not evaluate_py.exists():
        print(
            "FAIL: evaluate.py not found. Pass its path explicitly:\n"
            "  python3 verify_files_digest.py path/to/evaluate.py",
            file=sys.stderr,
        )
        return 1

    payload = {
        "files": sorted(rows, key=lambda row: str(row["relative_path"])),
        "upstream_evaluate_py": {
            "relative_path": "evaluate.py",
            "bytes": evaluate_py.stat().st_size,
            "sha256": _sha256(evaluate_py),
        },
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    print(f"evaluate.py: {evaluate_py} ({payload['upstream_evaluate_py']['sha256'][:16]}...)")
    print(f"runtime_files_sha256: {digest}")
    if digest != EXPECTED_DIGEST:
        print(f"FAIL: expected {EXPECTED_DIGEST}", file=sys.stderr)
        return 1
    print(f"PASS: matches the packet's pinned runtime_files_sha256 ({EXPECTED_ROWS} rows verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
