#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator CLI for the ddm_sf1 fit-context gate.

Reports the census WITH its denominator, so an empty scope is legible as VACUOUS
rather than as a clean pass.  Exit 0 = every in-scope producer records what its
solved coefficients were fitted against; exit 1 = a refusal or a vacuum.

    .venv/bin/python tools/preflight_ddm_sf1_fit_context.py
    .venv/bin/python tools/preflight_ddm_sf1_fit_context.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac.optimization.ddm_sf1_fit_context_preflight import (  # noqa: E402
    PARTNER_KEYS,
    SOLVED_COEFFICIENT_KEYS,
    scan,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args(argv)

    res = scan(a.repo_root)
    if a.json:
        print(json.dumps(res, indent=1))
        return 0 if res["ok"] else 1

    print(f"ddm_sf1 fit-context gate  [scanned {res['scanned_files']} .py files]")
    print(f"  in-scope producers : {res['in_scope']}  "
          f"(emit {SOLVED_COEFFICIENT_KEYS} beside {PARTNER_KEYS})")
    print(f"  stamped            : {res['stamped']}")
    print(f"  waived             : {res['waived']}")
    print(f"  REFUSED            : {res['refused']}")
    for row in res["rows"]:
        print(f"    {row['verdict']:<18} {row['path']}")
    if res["vacuous"]:
        print("\nVACUOUS: 0 in-scope producers. This is NOT a pass -- the gate "
              "has lost its subject. An empty scan and a clean scan must not "
              "emit the same symbol.")
        return 1
    if res["refused"]:
        print("\nREFUSED. Each listed producer emits a solved coefficient whose "
              "fit partners are unrecorded, so its freshness is unknowable from "
              "the artifact. Stamp at the emit site with "
              "ddm_fs1.stamp_fit_context(), or CARRY an upstream context "
              "forward if this stage does not re-solve the coefficient.")
        return 1
    print("\nOK: every in-scope solved coefficient records its fit partners.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
