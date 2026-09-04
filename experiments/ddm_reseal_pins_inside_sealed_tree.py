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

RECEIPT-INTEGRITY DEFECT FOUND AND FIXED (ddm_bh1 fresh-eyes hunt, 2026-09-04): the first
landing read ``config_in`` a SECOND time -- ``hashlib.sha256(config_in.read_bytes())`` --
AFTER ``config_out.write_text(text)`` had already run.  Nothing refused ``--config-out ==
--config-in``, so an in-place re-root silently recorded the OUTPUT bytes as the receipt's
``config_in.sha256``: a provenance record attesting to an input state that never existed.
The cure is threefold and all three are structural, not advisory: the input bytes are read
ONCE up front and BOTH the parse and the receipt sha come from that single read; the three
paths must be pairwise distinct (this tool writes BESIDE the original by contract, per the
docstring above -- in-place is refused, never silently tolerated); and both writes are
atomic (tmp + ``os.replace``) so a crash cannot leave a truncated config that a later fire
would read as sealed.  Sister of the repo's ``atomic_bytes`` discipline and of the
HISTORICAL_PROVENANCE append-only rule -- a receipt is forensic evidence, so the bytes it
attests to must be the bytes that were actually read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


class ResealError(RuntimeError):
    pass


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via tmp + ``os.replace`` so no reader sees a partial file.

    UTF-8 is pinned explicitly: bare ``write_text`` encodes with the process LOCALE, while the
    receipt's ``config_out.sha256`` is taken over ``text.encode()`` (UTF-8).  On a non-UTF-8
    locale the two would disagree and the receipt would attest to bytes the file does not hold.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


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
    # The three paths must be pairwise DISTINCT.  This tool writes BESIDE the original by
    # contract; an in-place re-root would make the receipt attest to bytes it never read.
    # Compared on the RESOLVED form so two spellings of one file cannot alias past the guard
    # (``main`` already resolves, but ``reroot`` is importable and must fail closed on its own);
    # the receipt still records the caller's own path strings.
    for left_name, left, right_name, right in (
        ("config_out", config_out, "config_in", config_in),
        ("receipt_out", receipt_out, "config_in", config_in),
        ("receipt_out", receipt_out, "config_out", config_out),
    ):
        if left.resolve() == right.resolve():
            raise ResealError(
                f"{left_name} must differ from {right_name}; this tool writes beside the "
                f"original so the receipt attests to the bytes it actually read: {left}"
            )
    # Read the input ONCE: both the parse and the receipt sha come from these exact bytes,
    # so no later write can change what the receipt says the input was.
    config_in_bytes = config_in.read_bytes()
    config_in_sha256 = hashlib.sha256(config_in_bytes).hexdigest()
    config = json.loads(config_in_bytes.decode("utf-8"))
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
    _atomic_write_text(config_out, text)
    receipt = {
        "schema": "sealed_config_pin_reroot.v1",
        "written_at_utc": datetime.now(UTC).isoformat(),
        # sha of the bytes read BEFORE any write -- never re-read after config_out lands.
        "config_in": {"path": str(config_in), "sha256": config_in_sha256},
        "config_out": {"path": str(config_out), "sha256": hashlib.sha256(text.encode()).hexdigest()},
        "sealed_tree": str(sealed_tree),
        "content_identity": "every pin sha256+bytes identical between config and sealed tree (refused otherwise)",
        "paths_rerooted": changed_paths,
        "pins_total": len(old),
    }
    _atomic_write_text(receipt_out, json.dumps(receipt, indent=2) + "\n")
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
