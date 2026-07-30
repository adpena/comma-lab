#!/usr/bin/env python
"""Venv src-custody gate: refuse when `tac` resolves outside the main repo src.

Bug class (2 measured occurrences: 2026-07-24 memory
`shared_venv_editable_install_hijack_from_arm_worktree_20260724`, and the sg1
arm start 2026-07-31 where `import tac` resolved to the eg1 worktree src for
~1 day): an arm worktree runs `uv pip install -e .` into the SHARED `.venv`,
leaving an editable finder / direct_url pointing at the worktree. Every later
shared-venv process silently imports STALE worktree code — custody poison for
any dispatch, measurement, or seal that assumes main-src provenance.

Checks (fail-closed, rc=1 loud):
  1. AUTHORITY: `import tac` resolves under <repo-root>/src/tac/. This catches
     every hijack mechanism (editable .pth finder, direct path entry, shadow
     install) because it tests the actual import machinery end-to-end.
  2. BELT: any *.dist-info/direct_url.json in site-packages whose file:// URL
     points at a directory that contains src/tac but is NOT the repo root
     (worktree editable installs leave exactly this fingerprint).

Wire-in: launcher gate 0 (launch_tr1_run and successors) + arm-start custody
check (`tac.subagent_contract` arms already verify tac.__file__; this tool is
the canonical callable form).

Usage:
    .venv/bin/python tools/check_venv_src_custody.py [--repo-root PATH]
Exit codes: 0 = custody OK · 1 = hijack detected (message names the fix).
"""

from __future__ import annotations

import argparse
import json
import site
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


def _site_package_dirs() -> list[Path]:
    dirs: list[str] = []
    try:
        dirs.extend(site.getsitepackages())
    except AttributeError:  # some embedded interpreters
        pass
    try:
        dirs.append(site.getusersitepackages())
    except AttributeError:
        pass
    return [Path(d) for d in dict.fromkeys(dirs) if Path(d).is_dir()]


def _direct_url_target(direct_url_json: Path) -> Path | None:
    try:
        url = json.loads(direct_url_json.read_text()).get("url", "")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(url, str) or not url.startswith("file://"):
        return None
    return Path(unquote(urlparse(url).path)).resolve()


def check_venv_src_custody(repo_root: Path) -> list[str]:
    """Return a list of custody failures (empty = OK)."""
    repo_root = repo_root.resolve()
    repo_src = repo_root / "src"
    failures: list[str] = []

    # Check 1 — authority: the live import machinery.
    try:
        import tac  # deliberate runtime probe of the live venv import machinery
    except ImportError as exc:
        return [f"import tac FAILED in this venv: {exc}"]
    tac_path = Path(tac.__file__).resolve()
    if repo_src not in tac_path.parents:
        failures.append(f"import tac -> {tac_path} (NOT under {repo_src}) — shared-venv hijack")

    # Check 2 — belt: worktree editable fingerprints in dist metadata.
    for sp in _site_package_dirs():
        for du in sp.glob("*.dist-info/direct_url.json"):
            target = _direct_url_target(du)
            if target is None or target == repo_root:
                continue
            if (target / "src" / "tac").is_dir():
                failures.append(
                    f"{du.parent.name}: editable direct_url -> {target} (a tac-bearing tree that is not {repo_root})"
                )
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="main repo root whose src/ must own the tac import",
    )
    args = ap.parse_args()
    failures = check_venv_src_custody(Path(args.repo_root))
    if failures:
        print(
            "VENV SRC-CUSTODY GATE: REFUSE (rc=1) — shared-venv hijack detected:",
            file=sys.stderr,
        )
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        print(
            f"  FIX: (cd {args.repo_root} && uv pip install -e . --no-deps) then re-run this gate.",
            file=sys.stderr,
        )
        return 1
    import tac

    print(f"venv src-custody OK: tac -> {Path(tac.__file__).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
