#!/usr/bin/env python3
"""Explicit historical move37 continuation after the live move38 promotion.

No live pointer is changed or hidden. Every stage records the observed pointer
and the separately pinned study field. The computation kernel is unchanged, so
its completed frame checkpoints remain bit-faithful resume inputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import ddm_bnd3_address_term as run
from experiments import ddm_bnd3_columnar as columnar


def bind(stage):
    selection_path = run.ROOT / "FIELD_SELECTION.json"
    selection = json.loads(selection_path.read_text())
    pointer = json.loads((run.REPO / ".omx/state/canonical_frontier_pointer.json").read_text())
    if pointer["our_local_frontier_contest_cuda"]["archive_sha256"] != selection["observed_live_archive_sha256"]:
        raise RuntimeError("LIVE_POINTER_CHANGED_AGAIN: record an explicit new study selection")
    if (
        selection["study_field_sha256"] != run.old.trace.FIELD_SHA
        or selection["study_archive_sha256"] != run.old.trace.ARCHIVE_SHA
    ):
        raise ValueError("selection does not bind charter move37")
    base = json.loads((run.ROOT / "decompose/INPUTS.json").read_text())["binding"]
    for key in ("producer", "joint", "old_codec", "source", "library"):
        if run.old.jg2.file_fact(Path(base[key]["path"])) != base[key]:
            raise ValueError(f"frozen computation source changed: {key}")
    run.old.complete_trace()
    if run.old.jg2.sha256_file(run.old.ROOT / "field.u8") != base["field_sha256"]:
        raise ValueError("study field changed")
    if run.old.jg2.sha256_file(run.old.ROOT / "runtime/archive.zip") != base["archive_sha256"]:
        raise ValueError("study archive changed")
    stage_path = run.ROOT / stage / "INPUTS.json"
    if stage_path.exists():
        if json.loads(stage_path.read_text())["binding"] != base:
            raise ValueError("stage computation binding changed")
    else:
        run.record(stage_path, {"binding": base, "research_only": True, "score_claim": False})
    run.record(
        run.ROOT / stage / "FROZEN_FIELD_RECHECK.json",
        {
            "selection": run.old.jg2.file_fact(selection_path),
            "selection_values": selection,
            "observed_live_pointer": pointer,
            "adapter": run.old.jg2.file_fact(Path(__file__)),
            "argv": sys.argv,
            "study_archive_sha256": base["archive_sha256"],
            "study_field_sha256": base["field_sha256"],
            "claim_is_about_live_field": False,
        },
    )
    return base, Path(base["library"]["path"])


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("fixed", "planar", "tests"):
        raise ValueError("mode must be fixed, planar, or tests; remaining arguments go to the documented kernel")
    mode = sys.argv.pop(1)
    run.bind = bind
    columnar.BASE_BIND = bind
    if mode == "fixed":
        run.main()
    elif mode == "planar":
        columnar.main()
    else:
        import pytest

        raise SystemExit(pytest.main(["-q", "experiments/test_ddm_bnd3_address_term.py"]))


if __name__ == "__main__":
    main()
