"""Write the frontier anchor mirror for a COMPLETED candidate_seal.v3.

First-measurement harvests are quarantined from ``experiments/results`` by design
(``tools/modal_harvest_poller.py::build_anchor_mirror`` refuses any result carrying
``prefire_intent_sha256``): the promotion object of the pre-fire intent contract is
the completed ``candidate_seal.v3``, not the raw receipt. Nothing wrote the mirror
from that seal, so ``tac.frontier_scan`` never saw the row and the canonical
pointer could not move (rlc5 → move 44, 2026-09-10; the ninth pass-path defect of
the contract's first real run).

This tool lifts the quarantine ONLY after the completed seal validates: it copies
the receipt the seal names, drops the quarantine keys, records the seal by
``{path, bytes, sha256}`` inside the mirror, and hands the copy to the poller's own
builder/writer so every scalar is still copied, never re-derived.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(1, str(REPO))

QUARANTINE_KEYS = ("prefire_intent_sha256", "first_measurement")


def _load_poller():
    spec = importlib.util.spec_from_file_location("modal_harvest_poller", REPO / "tools" / "modal_harvest_poller.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def completed_seal_result(seal_path: Path, *, repo: Path = REPO) -> tuple[dict, Path, dict]:
    """Validate the completed v3 seal and return (receipt copy without quarantine keys, receipt path, seal)."""
    from tac.candidate_seal import validate_seal

    verdict = validate_seal(seal_path, repo=repo) if "repo" in validate_seal.__code__.co_varnames else validate_seal(seal_path)
    ok = verdict.get("ok") if isinstance(verdict, dict) else bool(verdict)
    if not ok:
        raise SystemExit(f"REFUSED: completed seal does not validate: {verdict}")
    seal = json.loads(seal_path.read_text())
    if seal.get("schema") != "candidate_seal.v3":
        raise SystemExit(f"REFUSED: not a candidate_seal.v3: {seal.get('schema')!r}")
    receipt_ref = seal.get("first_measurement_receipt") or {}
    receipt_path = Path(receipt_ref.get("path", ""))
    if not receipt_path.is_file():
        raise SystemExit(f"REFUSED: seal names no readable first_measurement_receipt: {receipt_ref!r}")
    raw = receipt_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt_ref.get("sha256") or len(raw) != receipt_ref.get("bytes"):
        raise SystemExit("REFUSED: first_measurement_receipt bytes/sha differ from the seal's reference")
    result = json.loads(raw)
    for key in QUARANTINE_KEYS:
        result.pop(key, None)
    seal_bytes = seal_path.read_bytes()
    result["completed_candidate_seal_v3"] = {
        "path": str(seal_path.resolve()), "bytes": len(seal_bytes), "sha256": hashlib.sha256(seal_bytes).hexdigest(),
        "candidate_id": seal.get("candidate_id"), "axis": seal.get("axis"),
    }
    return result, receipt_path, seal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", required=True, help="completed candidate_seal.v3 JSON (committed)")
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--label", required=True, help="mirror file label (e.g. the instance job id)")
    args = parser.parse_args(argv)
    seal_path = Path(args.seal).resolve()
    result, receipt_path, seal = completed_seal_result(seal_path)
    poller = _load_poller()
    written = poller.write_anchor_mirror(result, source_receipt=receipt_path, label=args.label, lane_id=args.lane_id, repo_root=REPO)
    if written is None:
        raise SystemExit("REFUSED: poller writer declined the completed-seal mirror")
    print(json.dumps({"mirror": str(written), "candidate_id": seal.get("candidate_id"),
                      "score": result.get("score_recomputed_from_components"), "archive_sha256": result.get("expected_archive_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
