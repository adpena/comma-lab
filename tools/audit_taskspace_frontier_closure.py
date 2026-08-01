#!/usr/bin/env python3
"""Materialize the dynamic task-space frontier-closure audit."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from tac.witness_dsl.taskspace_frontier_closure_audit import (
    ClosureEvidencePaths,
    build_taskspace_frontier_closure_audit,
)

DEFAULT_RESEARCH = Path(".omx/research/original_taskspace_inverse_witness_codec_20260725")


def _stage_counts(run_dir: Path) -> dict[str, int]:
    checkpoints = run_dir / "checkpoints"
    counts: dict[str, int] = {}
    if not checkpoints.is_dir():
        return counts
    for path in checkpoints.glob("*.json"):
        stage = path.name.split(".", 1)[0]
        counts[stage] = counts.get(stage, 0) + 1
    return dict(sorted(counts.items()))


def _live_observations(repo_root: Path, g22_run: Path | None, g14_run: Path | None) -> dict[str, dict[str, Any]]:
    research = repo_root / DEFAULT_RESEARCH
    compiler = repo_root / "src/tac/witness_dsl/taskspace_selected_solution_compiler.py"
    observations: dict[str, dict[str, Any]] = {
        "g23_selected_solution_compiler": {
            "compiler_source_exists": compiler.is_file(),
            "focused_test_exists": any((repo_root / "src/tac/witness_dsl/tests").glob("test_taskspace_g17_*.py")),
            "runner_exists": (repo_root / "tools/run_taskspace_g17_composition.py").is_file(),
            "authority": "structural build state only",
        },
        "g22_full_population_equality": {
            "receipt_exists": (
                research / "g22_ep725_xcodec_n600_equality_replay_20260726/full_n600_decode_receipt.json"
            ).is_file(),
            "authority": "receipt existence only; strict semantic validation is a separate consumer",
        },
    }
    if g22_run is not None:
        observations["g22_full_population_equality"]["checkpoint_count"] = len(
            tuple((g22_run / "checkpoints").glob("*.json"))
        )
    if g14_run is not None:
        observations["g14_interaction_population"] = {
            "stage_counts": _stage_counts(g14_run),
            "terminal_receipt_exists": any(
                (g14_run / name).is_file() for name in ("final_receipt.json", "receipt.json", "run_receipt.json")
            ),
            "authority": "n2 advisory until a terminal receipt; never extrapolate to n600",
        }
    return observations


def _atomic_no_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError as exc:
            prior = path.read_bytes()
            if prior != payload:
                raise RuntimeError(f"refusing to replace unequal closure receipt {path}") from exc
        finally:
            tmp.unlink(missing_ok=True)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        help="optional explicit no-replace path; default is content-addressed under the campaign research root",
    )
    parser.add_argument("--g22-run", type=Path)
    parser.add_argument("--g14-run", type=Path)
    parser.add_argument("--check", action="store_true", help="print only; do not materialize")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    research = root / DEFAULT_RESEARCH
    paths = ClosureEvidencePaths(
        pointer=root / ".omx/state/canonical_frontier_pointer.json",
        ms1=root / ".omx/research/ddm_ms1_min_description_lattice_solve_20260724_receipt.json",
        ms2r=root / ".omx/research/ddm_ms2r_tolerance_capped_solve_r2_20260724T181428Z/receipt.json",
        g20=research / "ep725_lossless_xcodec_recode_20260726/receipt.json",
    )
    audit = build_taskspace_frontier_closure_audit(
        paths,
        observed_gates=_live_observations(root, args.g22_run, args.g14_run),
    )
    payload = json.dumps(audit, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    if args.check:
        print(payload.decode(), end="")
    else:
        output = args.output
        if output is None:
            output = DEFAULT_RESEARCH / (f"frontier_closure_audit_{audit['closure_identity_sha256'][:16]}.json")
        output = output if output.is_absolute() else root / output
        _atomic_no_replace(output, payload)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
