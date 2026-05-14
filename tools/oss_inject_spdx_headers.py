#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Inject `# SPDX-License-Identifier: MIT` into every in-scope `.py` file.

Idempotent: files that already have a SPDX-License-Identifier line in the first
2 KB are skipped. The header is inserted on the first non-shebang, non-encoding
line so existing `#!/usr/bin/env python3` / `# -*- coding: utf-8 -*-` lines stay
on top per PEP 263.

Scope (CLAUDE.md "OSS posture is comma.ai / openpilot style — MIT"):
- src/tac/
- experiments/
- tools/
- scripts/ (Python files only)

Excludes:
- __pycache__/
- _intake_/ (vendored public-PR clones per Catalog #109)
- vendored/
- .omx/oss_export/ (mirror)
- experiments/results/ (DERIVED_OUTPUT per Catalog #113)
- build/lib/
- reports/raw/

Per CLAUDE.md "Beauty, simplicity, and developer experience" + the comma.ai
openpilot convention of carrying SPDX headers in every project source file
(https://github.com/commaai/openpilot blob /master/.pre-commit-config.yaml).

Lane: lane_oss_release_v0_2_0_rc1_comma_openpilot_style_alex_pena_20260514.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Iterable

SCAN_ROOTS = ("src/tac", "experiments", "tools", "scripts")
EXCLUDE_MARKERS = (
    "__pycache__",
    "_intake_",
    "vendored",
    ".omx/oss_export",
    "experiments/results/",
    "build/lib/",
    "reports/raw/",
    ".venv",
    "upstream/",
)
SPDX_LINE = "# SPDX-License-Identifier: MIT\n"


def is_in_scope(path: pathlib.Path) -> bool:
    s = str(path).replace("\\", "/")
    return not any(marker in s for marker in EXCLUDE_MARKERS)


def iter_in_scope_py_files(repo_root: pathlib.Path) -> Iterable[pathlib.Path]:
    for root in SCAN_ROOTS:
        base = repo_root / root
        if not base.exists():
            continue
        for py in base.rglob("*.py"):
            if is_in_scope(py):
                yield py


def already_has_spdx(text: str) -> bool:
    """True iff first 2 KB carry an SPDX-License-Identifier line."""
    return "SPDX-License-Identifier" in text[:2048]


def insert_spdx(text: str) -> str:
    """Insert the canonical SPDX header on the first eligible line.

    PEP 263 / PEP 3120 ordering rules:
      1. shebang  (#!/usr/bin/env python3)        — MUST stay on line 1
      2. encoding (# -*- coding: utf-8 -*-)       — MUST stay on line 1 or 2
      3. SPDX header                              — insert just after the above
      4. module docstring / imports / ...
    """
    if not text:
        return SPDX_LINE + "\n"

    lines = text.splitlines(keepends=True)
    insert_at = 0

    # Skip shebang line if present
    if insert_at < len(lines) and lines[insert_at].startswith("#!"):
        insert_at += 1

    # Skip up to 2 encoding-declaration lines (PEP 263 allows it on line 1 or 2)
    for _ in range(2):
        if insert_at < len(lines):
            stripped = lines[insert_at].strip()
            if stripped.startswith("#") and ("coding:" in stripped or "coding=" in stripped):
                insert_at += 1

    new_lines = lines[:insert_at] + [SPDX_LINE] + lines[insert_at:]
    return "".join(new_lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root (default: cwd).")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without editing.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N files (for staged commits).")
    args = parser.parse_args()

    repo_root = pathlib.Path(args.repo_root).resolve()
    processed = 0
    edited = 0
    skipped_already_has = 0
    failed = 0

    for py in iter_in_scope_py_files(repo_root):
        if args.limit is not None and edited >= args.limit:
            break
        processed += 1
        try:
            text = py.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            print(f"[skip-decode-err] {py.relative_to(repo_root)}: {exc}", file=sys.stderr)
            failed += 1
            continue

        if already_has_spdx(text):
            skipped_already_has += 1
            continue

        new_text = insert_spdx(text)
        if args.dry_run:
            print(f"[would-edit] {py.relative_to(repo_root)}")
            edited += 1
            continue

        py.write_text(new_text, encoding="utf-8")
        edited += 1

    print(f"processed: {processed}")
    print(f"  already had SPDX header: {skipped_already_has}")
    print(f"  edited: {edited}")
    print(f"  failed (decode error): {failed}")
    if args.dry_run:
        print("(dry-run; no files modified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
