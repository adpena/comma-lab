#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# no-argparse-OK: two fixed invocations (--driver <base> <head> <out> | --resolve-index); prints usage docstring on misuse
"""Union merge for `.omx/state/lane_registry.json` (git merge driver + mid-merge CLI).

THE SHARP EDGE THIS REMOVES (2026-07-19 incident, twice in one session): every
codex-arm merge conflicts on the lane registry because both sides APPEND lanes.
The hand resolution is error-prone — one near-miss `git add`ed a
conflict-markered registry, destroying the staged :2/:3 stages (recovered only
via `git checkout -m`). Both sides' edits are additive per-lane, so a
deterministic union is the semantically correct merge.

Union semantics: start from OURS (its generated_at/from_state_hash/definitions
win); append every THEIRS lane whose identity key is absent from OURS. Lane
identity = `lane_id` if present, else `id`, else `name`, else the full-row JSON
(so even keyless rows dedup exactly). Same lane touched on BOTH sides keeps
OURS (latest local state) — acceptable because arm-side lane edits are
registration appends, not mutations of main-side lanes; a real double-edit
surfaces in `tools/lane_maturity.py validate`, which this driver runs and
FAILS CLOSED on (git then falls back to a normal conflict).

Two invocation modes:

1. git merge driver (auto-resolves; register once per clone):
     git config merge.lanereg.driver \
       '.venv/bin/python tools/merge_lane_registry.py --driver %O %A %B'
   with `.gitattributes`: `.omx/state/lane_registry.json merge=lanereg`

2. mid-merge repair (the file already shows `<<<<<<<` conflict stages):
     .venv/bin/python tools/merge_lane_registry.py --resolve-index
   Reads stages :2/:3 from the index, writes the union to the working tree.
   NEVER `git add` a conflict-markered state file — if stages were clobbered,
   recover them first with `git checkout -m -- <path>`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_RELPATH = ".omx/state/lane_registry.json"


def _lane_key(lane: object) -> str:
    if isinstance(lane, dict):
        for field in ("lane_id", "id", "name"):
            value = lane.get(field)
            if value:
                return f"{field}:{value}"
    return "row:" + json.dumps(lane, sort_keys=True)


def union_registries(ours: dict, theirs: dict) -> tuple[dict, list[str]]:
    """Return (merged, appended_keys). OURS metadata wins; THEIRS lanes union in."""
    merged = dict(ours)
    lanes_ours = ours.get("lanes")
    lanes_theirs = theirs.get("lanes")
    appended: list[str] = []
    if isinstance(lanes_ours, list) and isinstance(lanes_theirs, list):
        seen = {_lane_key(lane) for lane in lanes_ours}
        out = list(lanes_ours)
        for lane in lanes_theirs:
            key = _lane_key(lane)
            if key not in seen:
                out.append(lane)
                seen.add(key)
                appended.append(key)
        merged["lanes"] = out
    elif isinstance(lanes_ours, dict) and isinstance(lanes_theirs, dict):
        out_map = dict(lanes_ours)
        for lane_id, lane in lanes_theirs.items():
            if lane_id not in out_map:
                out_map[lane_id] = lane
                appended.append(lane_id)
        merged["lanes"] = out_map
    else:
        raise SystemExit(
            f"lane registry 'lanes' shape mismatch: ours={type(lanes_ours).__name__} "
            f"theirs={type(lanes_theirs).__name__} — refusing to guess"
        )
    return merged, appended


def _validate_or_die() -> None:
    """Fail closed: a union that breaks registry validation must NOT auto-merge."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "lane_maturity.py"), "validate"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(
            "merged lane registry FAILED lane_maturity validate — leaving the "
            "conflict for a human (git will record a normal conflict)"
        )


def run_driver(ancestor: str, current: str, other: str) -> int:
    ours = json.loads(Path(current).read_text())
    theirs = json.loads(Path(other).read_text())
    merged, appended = union_registries(ours, theirs)
    Path(current).write_text(json.dumps(merged, indent=2) + "\n")
    _validate_or_die()
    sys.stderr.write(
        f"[merge_lane_registry] union OK — appended {len(appended)} lane(s) from THEIRS\n"
    )
    return 0


def run_resolve_index() -> int:
    def stage(n: int) -> dict:
        proc = subprocess.run(
            ["git", "show", f":{n}:{REGISTRY_RELPATH}"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        if proc.returncode != 0:
            raise SystemExit(
                f"stage :{n} for {REGISTRY_RELPATH} is missing — if the "
                f"conflict-markered file was `git add`ed, recover the stages "
                f"with: git checkout -m -- {REGISTRY_RELPATH}"
            )
        return json.loads(proc.stdout)

    merged, appended = union_registries(stage(2), stage(3))
    (REPO_ROOT / REGISTRY_RELPATH).write_text(json.dumps(merged, indent=2) + "\n")
    _validate_or_die()
    print(
        f"union written to {REGISTRY_RELPATH} — appended {len(appended)} lane(s); "
        f"review then `git add {REGISTRY_RELPATH}`"
    )
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 5 and argv[1] == "--driver":
        return run_driver(argv[2], argv[3], argv[4])
    if len(argv) == 2 and argv[1] == "--resolve-index":
        return run_resolve_index()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
