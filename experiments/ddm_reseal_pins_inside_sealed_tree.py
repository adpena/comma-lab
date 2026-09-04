#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Re-root a sealed cell config's ``source_pins`` to the sealed tree that will RUN it.

THE DEFECT (MAIN, 2026-09-04, ng2 + ng3): a cell sealed from the working tree records
``source_pins`` whose ``path`` fields point at ``/Users/adpena/Projects/pact/...``. The burn
prep's ``validate_config`` compares that dict WHOLE against ``verify_inputs()`` run in the
tree that fires the cell (the sealed ``git archive`` snapshot), whose paths point at the
snapshot -- so a content-identical seal is refused with "QBR source pins differ from live
exact inputs", and the seal tool cannot be re-run inside the snapshot (no ``.git``).

THE CURE: run ``verify_inputs()`` INSIDE the sealed tree (its own interpreter, its own
``REPO``), refuse unless every pin's ``sha256`` and ``bytes`` are IDENTICAL to the config's
(content identity is the seal; paths are where the bytes live), and write the re-rooted
config beside the original with a receipt. Nothing else in the config changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


class ResealError(RuntimeError):
    pass


def verify_inputs_inside(sealed_tree: Path) -> dict:
    script = (
        "import json,sys;"
        f"sys.path.insert(0, {str(sealed_tree)!r});"
        "from experiments import ddm_qbr1_born_fairform_burn_prep as q;"
        "print(json.dumps(q.verify_inputs()))"
    )
    py = sealed_tree / ".venv/bin/python"
    if not py.exists():
        raise ResealError(f"sealed tree has no interpreter: {py}")
    done = subprocess.run([str(py), "-c", script], cwd=sealed_tree, text=True, capture_output=True, check=False)
    if done.returncode != 0:
        raise ResealError(f"verify_inputs() failed inside the sealed tree:\n{done.stderr.strip()[-2000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def reroot(config_in: Path, sealed_tree: Path, config_out: Path, receipt_out: Path) -> dict:
    config = json.loads(config_in.read_text())
    old = config["source_pins"]
    live = verify_inputs_inside(sealed_tree)
    missing = sorted(set(old) - set(live))
    extra = sorted(set(live) - set(old))
    if missing or extra:
        raise ResealError(f"pin key sets differ: missing_in_tree={missing} extra_in_tree={extra}")
    drift = {
        k: {"config": (old[k].get("sha256"), old[k].get("bytes")), "tree": (live[k].get("sha256"), live[k].get("bytes"))}
        for k in old
        if (old[k].get("sha256"), old[k].get("bytes")) != (live[k].get("sha256"), live[k].get("bytes"))
    }
    if drift:
        raise ResealError(f"CONTENT drift, refusing to re-root (this is not a path problem): {json.dumps(drift)[:1500]}")
    rerooted = {k: {**old[k], "path": live[k]["path"]} for k in old}
    changed_paths = sorted(k for k in old if old[k].get("path") != live[k].get("path"))
    config["source_pins"] = rerooted
    text = json.dumps(config, indent=2, sort_keys=True) + "\n"
    config_out.parent.mkdir(parents=True, exist_ok=True)
    config_out.write_text(text)
    receipt = {
        "schema": "sealed_config_pin_reroot.v1",
        "written_at_utc": datetime.now(UTC).isoformat(),
        "config_in": {"path": str(config_in), "sha256": hashlib.sha256(config_in.read_bytes()).hexdigest()},
        "config_out": {"path": str(config_out), "sha256": hashlib.sha256(text.encode()).hexdigest()},
        "sealed_tree": str(sealed_tree),
        "content_identity": "every pin sha256+bytes identical between config and sealed tree (refused otherwise)",
        "paths_rerooted": changed_paths,
        "pins_total": len(old),
    }
    receipt_out.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-in", type=Path, required=True)
    ap.add_argument("--sealed-tree", type=Path, required=True)
    ap.add_argument("--config-out", type=Path, required=True)
    ap.add_argument("--receipt-out", type=Path, required=True)
    a = ap.parse_args(argv)
    receipt = reroot(a.config_in.resolve(), a.sealed_tree.resolve(), a.config_out.resolve(), a.receipt_out.resolve())
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
