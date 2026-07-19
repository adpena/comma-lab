#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only compatibility/refusal proof for Task #537's sacred checkpoint layout."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import numpy as np


REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"
DEFAULT_SOURCE = Path(
    "/Users/adpena/Projects/pact/experiments/results/"
    "levelset_n600_witness_20260717T113932Z/levelset_resume_state.npz"
)
DEFAULT_SCRATCH = REPO / "experiments" / "results" / "resumability_537_sacred_layout_scratch"
DEFAULT_RECEIPT = REPO / ".omx" / "research" / "resumability_537_sacred_layout_replay_receipt_20260719.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_trainer() -> ModuleType:
    for path in (REPO, REPO / "src", REPO / "experiments", REPO / "upstream"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("task537_levelset_trainer", TRAINER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import trainer: {TRAINER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(source: Path, scratch: Path, receipt: Path) -> dict[str, object]:
    source = source.resolve()
    scratch = scratch.resolve()
    receipt = receipt.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if scratch.exists():
        raise FileExistsError(f"refusing to overwrite scratch: {scratch}")
    scratch.mkdir(parents=True)
    copied = scratch / "sacred_layout_copy.npz"
    mutated = scratch / "sacred_layout_without_optimizer.npz"
    source_hash_before = _sha256(source)
    shutil.copy2(source, copied)
    copied_hash = _sha256(copied)

    trainer = _load_trainer()
    full_state = trainer._load_resume_state(copied)
    compatibility = trainer._validate_resume_state_for_continuation(
        full_state, warm_start_weights_only=False,
    )

    with np.load(copied, allow_pickle=False) as z:
        arrays = {key: np.asarray(z[key]) for key in z.files if not key.startswith("optP__")}
    arrays["__resume_has_opt"] = np.asarray(0)
    tmp = mutated.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(mutated)
    mutated_hash = _sha256(mutated)
    refusal = ""
    try:
        trainer._validate_resume_state_for_continuation(
            trainer._load_resume_state(mutated), warm_start_weights_only=False,
        )
    except ValueError as exc:
        refusal = str(exc)
    if "optimizer" not in refusal.lower():
        raise AssertionError(f"mutated sidecar did not refuse on optimizer custody: {refusal!r}")

    source_hash_after = _sha256(source)
    assertions = {
        "source_copy_hash_matches": copied_hash == source_hash_before,
        "legacy_full_state_passes": compatibility.get("legacy_compatibility") is True,
        "mutated_missing_optimizer_refuses": "optimizer" in refusal.lower(),
        "source_hash_unchanged": source_hash_before == source_hash_after,
    }
    report: dict[str, object] = {
        "schema": "tac.resumability_537_sacred_layout_replay.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "authority": "READ_ONLY_LAYOUT_COMPATIBILITY_PROOF",
        "score_claim": False,
        "frontier_pointer_mutated": False,
        "source": {"path": str(source), "bytes": source.stat().st_size, "sha256": source_hash_before},
        "copy": {"sha256": copied_hash},
        "mutated_copy": {"sha256": mutated_hash, "removed_semantic_leg": "optimizer_moments"},
        "compatibility_row": compatibility,
        "refusal": refusal,
        "assertions": assertions,
        "all_pass": all(assertions.values()),
        "cleanup": {"policy": "success-only copied scratch", "scratch_deleted": False},
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["all_pass"]:
        raise AssertionError(json.dumps(assertions, indent=2))
    shutil.rmtree(scratch)
    report["cleanup"]["scratch_deleted"] = True  # type: ignore[index]
    receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    ap.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = ap.parse_args()
    report = run(args.source, args.scratch, args.receipt)
    print(json.dumps({"all_pass": report["all_pass"], "receipt": str(args.receipt)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
