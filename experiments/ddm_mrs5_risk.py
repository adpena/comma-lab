#!/usr/bin/env python3
"""Build ddm_mrs5's timing-risk receipt and retention manifest, then record what the
frozen pre-fire contract says about them.

WHY this file exists and what it deliberately does NOT do.

The charter asks for `candidate_prefire_timing_risk.v1` with lineage to move 53's
completed `t4_direct` leg (1,185.899645171 s) and a delta equal to the seven packet
files. The frozen contract's own `completed_t4_receiver_delta` validator
(`tac.candidate_seal.validate_prefire_risk`) requires a field this packet cannot
supply: `candidate_receiver.sha256`, computed by
`tac.candidate_seal.measure_prefire_risk_receiver_digest`, which REFUSES any receiver
whose `inflate.py` carries no top-level `ARCHIVE_SHA256` / `ARCHIVE_BYTES` assignment.
The minimal packet is unpinned by charter. So the required field is not merely absent
from this receipt — it is uncomputable for this tree, and the refusal fires before any
receipt shape can matter.

This builder therefore writes the receipt with every field it can measure honestly,
records `candidate_receiver.sha256 = null` beside the exact typed refusal, and sets
`contract_validated: false`. It never invents a digest, never adds a pin to the packet
to get past the control, and never reports the projection as timing authority. The
receipt is evidence for MAIN's adjudication, not a passing gate.

The projection itself is the charter's instrument: `source_t4_seconds * ratio`, where
`ratio = median(B)/median(A)` over the same-host interleaved A/B/A/B. Note that the
contract's own arithmetic for this mode is `source_t4 * (1 + max(0, ceiling/base - 1))`
— a one-sided clamp with no expression for a ratio below 1. Both numbers are reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.candidate_seal import (
    PrefireRefusal,
    measure_prefire_risk_receiver_digest,
    prefire_digest,
    prefire_risk_receiver_rows,
)

POLICY_LIMIT_SECONDS = 1260.0
HARD_TIMEOUT_SECONDS = 1800.0
RISK_SCHEMA = "candidate_prefire_timing_risk.v1"
RISK_DIGEST_DEFINITION = "tac.candidate_seal.measure_prefire_risk_receiver_digest.v1"
LEGACY_DIGEST_DEFINITION = "tac.decode_wall_clock.measure_receiver_digest"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_reference(path: Path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def measured_or_refusal(root: Path) -> tuple[str | None, dict | None]:
    """Return the risk digest, or None beside the exact typed refusal that fired."""
    try:
        return measure_prefire_risk_receiver_digest(root), None
    except PrefireRefusal as exc:
        return None, {
            "exception_type": type(exc).__name__,
            "code": getattr(exc, "code", None),
            "message": str(exc),
        }


def receiver_rows_or_refusal(root: Path) -> tuple[list | None, dict | None]:
    try:
        return [list(row) for row in prefire_risk_receiver_rows(root)], None
    except PrefireRefusal as exc:
        return None, {
            "exception_type": type(exc).__name__,
            "code": getattr(exc, "code", None),
            "message": str(exc),
        }


def build_ratio_block(timing_result: dict) -> dict:
    """Re-derive the ratio from the run rows; never trust the summary fields."""
    runs = timing_result["runs"]
    arm_walls: dict[str, list[float]] = {"A": [], "B": []}
    rows = []
    for run in runs:
        if run["returncode"] != 0:
            raise SystemExit(f"timing run {run['sequence_index']} returncode {run['returncode']}")
        arm_walls[run["arm"]].append(run["wall_seconds"])
        rows.append({
            "sequence_index": run["sequence_index"],
            "arm": run["arm"],
            "wall_seconds": run["wall_seconds"],
            "pair_compute_seconds": run["pair_compute_seconds"],
            "returncode": run["returncode"],
            "process_cold": run["process_cold"],
            "load_1min_before": run["before"]["load_average"][0],
            "load_1min_after": run["after"]["load_average"][0],
            "started_utc": run["before"]["utc"],
            "finished_utc": run["after"]["utc"],
        })
    a_walls, b_walls = arm_walls["A"], arm_walls["B"]
    if len(a_walls) != 2 or len(b_walls) != 2:
        raise SystemExit(f"expected two runs per arm; got A={len(a_walls)} B={len(b_walls)}")
    median_a = statistics.median(a_walls)
    median_b = statistics.median(b_walls)
    ratio = median_b / median_a
    adjacent = [b_walls[i] / a_walls[i] for i in range(2)]
    every = [b / a for b in b_walls for a in a_walls]
    return {
        "definition": "ratio = median(B)/median(A) over four cold same-host runs in fixed A/B/A/B order",
        "arm_a": "sealed receiver /Volumes/APDataStore/pact/ddm_pd6/candidate2/candidate_runtime",
        "arm_b": "submissions/mrs5 (the minimal seven-file packet)",
        "runs": rows,
        "median_a_seconds": median_a,
        "median_b_seconds": median_b,
        "ratio_median_b_over_median_a": ratio,
        "adjacent_ratios": adjacent,
        "adjacent_ratio_half_range": (max(adjacent) - min(adjacent)) / 2,
        "ratio_extreme_range": [min(every), max(every)],
        "pair_compute_ratio_sensitivity": (
            statistics.median([r["pair_compute_seconds"] for r in rows if r["arm"] == "B"])
            / statistics.median([r["pair_compute_seconds"] for r in rows if r["arm"] == "A"])
        ),
        "n_pairs_per_run": timing_result["binding"]["limit"],
        "scope": timing_result["scope"],
        "no_statistical_confidence_interval": True,
        "authority": False,
        "score_claim": False,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timing-result", required=True, help="the interleaved A/B/A/B RESULT.json")
    ap.add_argument("--source-leg", required=True, help="move 53's completed t4_direct leg JSON")
    ap.add_argument("--sealed-runtime", required=True, help="the sealed receiver that measured the leg")
    ap.add_argument("--candidate-runtime", required=True, help="submissions/mrs5")
    ap.add_argument("--risk-out", required=True)
    ap.add_argument("--retention-out", required=True)
    ap.add_argument("--store", required=True, help="the arm's APFS store root, for retained custody")
    args = ap.parse_args(argv)

    timing_result = json.loads(Path(args.timing_result).read_text(encoding="utf-8"))
    leg_path = Path(args.source_leg).resolve()
    leg = json.loads(leg_path.read_text(encoding="utf-8"))
    if leg.get("mode") != "t4_direct":
        raise SystemExit(f"source leg is not t4_direct: {leg.get('mode')!r}")
    sealed_root = Path(args.sealed_runtime).resolve()
    candidate_root = Path(args.candidate_runtime).resolve()

    source_digest, source_refusal = measured_or_refusal(sealed_root)
    candidate_digest, candidate_refusal = measured_or_refusal(candidate_root)
    sealed_rows, sealed_rows_refusal = receiver_rows_or_refusal(sealed_root)
    candidate_rows, candidate_rows_refusal = receiver_rows_or_refusal(candidate_root)

    ratio_block = build_ratio_block(timing_result)
    ratio = ratio_block["ratio_median_b_over_median_a"]
    source_t4 = leg["measured_t4_decode_seconds"]
    projected = source_t4 * ratio
    # The frozen mode's own arithmetic, stated for comparison: it clamps at zero, so a
    # candidate measured FASTER than its source projects the source's own time unchanged.
    contract_clamped = source_t4 * (1 + max(0.0, ratio - 1))

    candidate_files = sorted(p for p in candidate_root.iterdir() if p.is_file())
    delta_files = [
        {"relative_path": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in candidate_files
    ]

    risk = {
        "schema": RISK_SCHEMA,
        "mode": "completed_t4_receiver_delta",
        "authority": False,
        "timing_clearance": False,
        "score_claim": False,
        "contract_validated": False,
        "source_t4_leg": file_reference(leg_path),
        "source_t4_leg_facts": {
            "runtime_dir": leg["runtime_dir"],
            "archive_sha256": leg["archive_sha256"],
            "receiver_sha256": leg["receiver_sha256"],
            "measured_t4_decode_seconds": source_t4,
            "limit_seconds": leg["limit_seconds"],
            "t4_timing_scope": leg["t4_timing_scope"],
        },
        "source_receiver": {
            "path": str(sealed_root),
            "digest_definition": RISK_DIGEST_DEFINITION,
            "sha256": source_digest,
            "t4_direct_digest_definition": LEGACY_DIGEST_DEFINITION,
            "t4_direct_sha256": leg["receiver_sha256"],
            "refusal": source_refusal,
            "row_count": len(sealed_rows) if sealed_rows is not None else None,
        },
        "candidate_receiver": {
            "path": str(candidate_root),
            "digest_definition": RISK_DIGEST_DEFINITION,
            "sha256": candidate_digest,
            "refusal": candidate_refusal,
            "row_count": len(candidate_rows) if candidate_rows is not None else None,
            "rows_refusal": candidate_rows_refusal,
            "legacy_digest_definition": LEGACY_DIGEST_DEFINITION,
        },
        "receiver_delta": {
            "definition": (
                "the seven shipped packet files with sha256; NOT the contract's normalized "
                "receiver_delta_manifest, which cannot be built while the candidate's risk rows refuse"
            ),
            "candidate_file_count": len(delta_files),
            "candidate_total_bytes": sum(f["bytes"] for f in delta_files),
            "files": delta_files,
            "source_row_count": len(sealed_rows) if sealed_rows is not None else None,
            "source_rows_refusal": sealed_rows_refusal,
        },
        "interleaved_ratio": ratio_block,
        "calculation": {
            "source_t4_seconds": source_t4,
            "ratio_median_b_over_median_a": ratio,
            "t4_risk_ceiling_seconds": projected,
            "contract_clamped_projection_seconds": contract_clamped,
            "policy_limit_seconds": POLICY_LIMIT_SECONDS,
            "hard_timeout_seconds": HARD_TIMEOUT_SECONDS,
            "within_policy_limit": projected <= POLICY_LIMIT_SECONDS < HARD_TIMEOUT_SECONDS,
            "contract_clamped_within_policy_limit": (
                contract_clamped <= POLICY_LIMIT_SECONDS < HARD_TIMEOUT_SECONDS
            ),
            "assumption": (
                "serial-Python scaling transfers to the T4 host; the GPU stages are unchanged code; "
                "host load approximately stationary across the interleave (pd7 shares the host)"
            ),
        },
        "contract_conformance": {
            "validator": "tac.candidate_seal.validate_prefire_risk",
            "validated": False,
            "uncomputable_required_field": "candidate_receiver.sha256",
            "root_control": candidate_refusal,
            "shape_note": (
                "validate_prefire_risk checks the exact field set before anything else, so a receipt "
                "carrying these honest extra fields refuses on shape; the ROOT control is the digest "
                "refusal above, which fires for this tree under any receipt shape"
            ),
            "mode_arithmetic_note": (
                "the frozen mode computes source_t4 * (1 + max(0, ceiling/base - 1)) from cold-n600 "
                "candidate_decode_wall_clock diagnostics whose retained verdict is REFUSED; the "
                "charter's instrument is a sampled 48-pair interleave with a ratio below 1, which "
                "that arithmetic has no field for and its clamp cannot represent"
            ),
        },
    }
    risk["risk_sha256"] = prefire_digest(risk, "risk_sha256")

    store = Path(args.store).resolve()
    retained = [
        p for p in (
            store / "retained" / "p",
            candidate_root / "archive.zip",
            REPO / ".omx/research/ddm_mrs5_20260916/ARCHIVE_PARSEBACK.json",
            REPO / ".omx/research/ddm_mrs5_20260916/RAW_IDENTITY_N600.json",
        ) if p.is_file()
    ]
    retention = {
        "schema": "candidate_retention_manifest.v1",
        "arm": "ddm_mrs5",
        "store_root": str(store),
        "retained": [file_reference(p) for p in retained],
        "reclaimed": [{
            "path": str(store / "public_native/output/0.raw"),
            "bytes": 3662409600,
            "sha256": "8a14f55a6a8b141501836f511f4222dde75dca21714d923ad865b5f4757ef66b",
            "reason": (
                "certified duplicate: byte-identical to move 53's retained raw, proven by the cold "
                "n600 identity receipt before reclaim; reproducible from the retained archive"
            ),
        }],
        "score_claim": False,
    }

    for path, document in ((Path(args.risk_out), risk), (Path(args.retention_out), retention)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"WROTE {path} sha256={sha256_file(path)} bytes={path.stat().st_size}")

    print(f"ratio={ratio!r}")
    print(f"projected_t4_seconds={projected!r} policy_limit={POLICY_LIMIT_SECONDS}")
    print(f"contract_clamped_projection_seconds={contract_clamped!r}")
    print(f"candidate_risk_digest={candidate_digest!r} refusal={candidate_refusal!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
