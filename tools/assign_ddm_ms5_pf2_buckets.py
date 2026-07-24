#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Re-walk PF2 raw events and emit the SHA-bound MS5 assignment table."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.ddm_g3_score_atlas import reconstruct_v12_state  # noqa: E402
from tac.optimization.ddm_pf2_bucket_assignment import (  # noqa: E402
    ASSIGNMENT_RECEIPT_SCHEMA,
    build_assignment_table,
    canonical_bytes,
    canonical_sha256,
    validate_assignment_table,
)
from tac.optimization.direct_description_joint_descent import (  # noqa: E402
    lift_v15_archive,
)

RUN_ID = "ddm_ms5_pf2_bucket_assignment_20260724T044736Z"
LANE_ID = "lane_ddm_ms5_pf2_bucket_assignment_20260724"
PF2_SOURCE_COMMIT = "b8c81edec2161838a060d1996f8fb973ba1acb41"
EXPECTED_PF2_SHA256 = "85084f7bd3a03dbd1b9f04fe6a9b84df4948a6caf64620beef42da8924345f73"
EXPECTED_G2F_SHA256 = "47d3ca538f1b876f7639223a1a9a7714b7db2083eaa0971936b9a43a1e6d0d04"
EXPECTED_G2G_SHA256 = "fa49a2ca71cb2960b1e497d425f05c4a496cc7634c45b2e193e3977dfa0667da"
EXPECTED_J2_ARCHIVE_SHA256 = "759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df"
PF2_IMPLEMENTATION_SHA256 = {
    "tools/measure_ddm_pf2_dimension_conditioned_two_type.py": (
        "200e5f23ceb5b8af6b4302bdbaf40e858412107e11369c717cdb77e97c7ec2d2"
    ),
    "src/tac/optimization/ddm_dimension_conditioned_two_type.py": (
        "5cfd04964f76151ee2bdf6281c09855cbe2c4fa3cbdfb47f6a49a701c699b8ea"
    ),
    "src/tac/optimization/ddm_g3_score_atlas.py": ("8364e6998075f35f6a5684311b4861c0843f4ac682c11f35ec2385b1304a63b9"),
    "src/tac/optimization/ddm_g4_spatial_stationarity.py": (
        "ee50b8e304fa49c01147854bbe9bda0daf61dbee55c02317f7b629d68a4a5703"
    ),
    "src/tac/optimization/direct_description_joint_descent.py": (
        "76c66ceb6ad554898ae2b02db3230d669f104b97e617dd08881c9357154c9f51"
    ),
}

DEFAULT_PF2 = REPO_ROOT / (
    ".omx/research/ddm_pf2_dimension_conditioned_two_type_20260724T020205Z/"
    "ddm_pf2_dimension_conditioned_two_type_receipt.json"
)
DEFAULT_J2_ARCHIVE = REPO_ROOT / (
    ".omx/research/ddm_v15_scorer_solved_templates_n600_20260723T013000Z/"
    "ddm_v15_solved_templates_n600.not_a_candidate.zip.receipt-bytes"
)
DEFAULT_G2F = Path("/Volumes/VertigoDataTier/pact/evidence/g2f_chart_amplitude_20260721/receipt.json")
DEFAULT_G2G = Path("/Volumes/VertigoDataTier/pact/evidence/g2g_chart_receiver_20260721/run_20260721T1622Z/receipt.json")
DEFAULT_OUTPUT = REPO_ROOT / ".omx/research" / RUN_ID


class AssignmentRunError(RuntimeError):
    """A bound artifact changed or cannot support the requested assignment."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_bound(path: Path, expected_sha256: str, label: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise AssignmentRunError(f"{label} is absent or not a regular file: {path}")
    payload = path.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise AssignmentRunError(f"{label} SHA-256 differs: expected {expected_sha256}, observed {observed}")
    return payload


def verify_large_bound(path: Path, expected_sha256: str, expected_bytes: int, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise AssignmentRunError(f"{label} is absent or not a regular file: {path}")
    observed_bytes = path.stat().st_size
    if observed_bytes != expected_bytes:
        raise AssignmentRunError(f"{label} byte length differs: expected {expected_bytes}, observed {observed_bytes}")
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha256:
        raise AssignmentRunError(f"{label} SHA-256 differs: expected {expected_sha256}, observed {observed_sha}")
    return {"path": str(path), "bytes": observed_bytes, "sha256": observed_sha}


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _compose_pf2_input_rehash(pf2: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    custody = pf2.get("input_custody")
    if not isinstance(custody, dict):
        raise AssignmentRunError("PF2 receipt lacks input custody")
    for role, item in sorted(custody.items()):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise AssignmentRunError(f"PF2 input custody row {role} is malformed")
        path = resolve_repo_path(item["path"])
        expected = item.get("sha256")
        if not isinstance(expected, str):
            raise AssignmentRunError(f"PF2 input custody row {role} lacks SHA-256")
        payload = read_bound(path, expected, f"PF2 input {role}")
        if "bytes" in item and len(payload) != int(item["bytes"]):
            raise AssignmentRunError(f"PF2 input {role} byte length differs")
        rows.append(
            {
                "role": role,
                "path": item["path"],
                "bytes": len(payload),
                "sha256": expected,
            }
        )
    return rows


def _xi_event_ids(track_payload: bytes) -> list[int]:
    result: list[int] = []
    for line in track_payload.splitlines():
        row = json.loads(line)
        event_ids = row.get("event_ids")
        if not isinstance(event_ids, list):
            raise AssignmentRunError("G4 xi track row lacks event_ids")
        result.extend(event_ids)
    return result


def _g2g_actuator_ids(receipt: dict[str, Any]) -> list[str]:
    result: set[str] = set()
    for pair_row in receipt.get("candidate_rows", []):
        for candidate in pair_row.get("candidate_rows", []):
            symbol = candidate.get("symbol", {})
            try:
                result.add(
                    "g2g.g2cs1."
                    f"pair{int(symbol['pair_index']):03d}."
                    f"line{int(symbol['line_index']):02d}."
                    f"coefficient{int(symbol['coefficient_index']):02d}"
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise AssignmentRunError("G2G symbol vocabulary row is malformed") from exc
    if not result:
        raise AssignmentRunError("G2G receipt produced an empty actuator vocabulary")
    return sorted(result)


def _validate_g2f_direction_convention(receipt: dict[str, Any]) -> tuple[str, str]:
    rows = receipt.get("chart_bidirectional_observations")
    if not isinstance(rows, list) or not rows:
        raise AssignmentRunError("G2F receipt lacks paired chart observations")
    for row in rows:
        negative = row.get("negative")
        positive = row.get("positive")
        if (
            not isinstance(negative, dict)
            or not isinstance(positive, dict)
            or not float(negative.get("signed_amplitude", 0.0)) < 0.0
            or not float(positive.get("signed_amplitude", 0.0)) > 0.0
        ):
            raise AssignmentRunError("G2F paired-secant sign convention differs")
    return ("NEGATIVE_ONE_QUANTUM", "POSITIVE_ONE_QUANTUM")


def run(args: argparse.Namespace) -> Path:
    implementation_custody = []
    for relative, expected in PF2_IMPLEMENTATION_SHA256.items():
        source_path = REPO_ROOT / relative
        source_payload = read_bound(
            source_path,
            expected,
            f"implementation source {relative}",
        )
        implementation_custody.append(
            {
                "path": relative,
                "bytes": len(source_payload),
                "sha256": expected,
            }
        )
    pf2_payload = read_bound(args.pf2, args.pf2_sha256, "PF2 receipt")
    pf2 = json.loads(pf2_payload)
    composed_inputs = _compose_pf2_input_rehash(pf2)

    g4_binding = pf2["input_custody"]["g4_receipt"]
    g4_path = resolve_repo_path(g4_binding["path"])
    g4 = json.loads(read_bound(g4_path, g4_binding["sha256"], "G4 receipt"))
    recurrence = next(row for row in g4["outputs"] if row["path"].endswith("01_recurrence_arrays.npz"))
    tracks = next(row for row in g4["outputs"] if row["path"].endswith("xi_proxy_tracks.jsonl"))
    recurrence_path = resolve_repo_path(recurrence["path"])
    tracks_path = resolve_repo_path(tracks["path"])
    recurrence_payload = read_bound(recurrence_path, recurrence["sha256"], "G4 recurrence arrays")
    if len(recurrence_payload) != int(recurrence["bytes"]):
        raise AssignmentRunError("G4 recurrence-array byte length differs")
    tracks_payload = read_bound(tracks_path, tracks["sha256"], "G4 xi tracks")
    if len(tracks_payload) != int(tracks["bytes"]):
        raise AssignmentRunError("G4 xi-track byte length differs")
    with np.load(recurrence_path, allow_pickle=False) as stored:
        transition_counts = np.asarray(stored["transition_counts"], dtype=np.uint16)

    v12_binding = pf2["input_custody"]["v12_receipt"]
    v12_path = resolve_repo_path(v12_binding["path"])
    v12 = json.loads(read_bound(v12_path, v12_binding["sha256"], "V12 receipt"))
    state = reconstruct_v12_state(REPO_ROOT, v12, n_pairs=600)
    predicted = np.asarray(state.final_cells, dtype=np.uint8)
    target_binding = v12["target_custody"]
    target_path = Path(target_binding["cache_path"])
    target_custody = verify_large_bound(
        target_path,
        target_binding["cache_sha256"],
        int(target_binding["cache_bytes"]),
        "V12 target cache",
    )
    target = np.asarray(open_stored_npy_memmap(target_path, "lstars"), dtype=np.uint8)

    j2_payload = read_bound(args.j2_archive, args.j2_archive_sha256, "J2 source archive")
    j2_lift = lift_v15_archive(j2_payload)
    j2_actuators = [f"j2.{name}" for name in j2_lift.parameter_names]
    g2f = json.loads(read_bound(args.g2f_receipt, args.g2f_sha256, "G2F receipt"))
    direction_ids = _validate_g2f_direction_convention(g2f)
    g2g = json.loads(read_bound(args.g2g_receipt, args.g2g_sha256, "G2G receipt"))
    actuator_ids = sorted(set(j2_actuators + _g2g_actuator_ids(g2g)))

    table = build_assignment_table(
        pf2_receipt=pf2,
        pf2_receipt_sha256=args.pf2_sha256,
        predicted=predicted,
        target=target,
        transition_counts=transition_counts,
        xi_event_ids=_xi_event_ids(tracks_payload),
        actuator_vocabulary=actuator_ids,
        direction_vocabulary=direction_ids,
    )
    validate_assignment_table(table, expected_pf2_sha256=args.pf2_sha256)
    table_path = args.output_directory / "pf2_bucket_assignment_table.json"
    table_bytes = canonical_bytes(table)
    publish(table_path, table_bytes)

    receipt: dict[str, Any] = {
        "schema": ASSIGNMENT_RECEIPT_SCHEMA,
        "run_id": RUN_ID,
        "lane_id": LANE_ID,
        "pf2_source_commit": PF2_SOURCE_COMMIT,
        "input_hash_lineage": {
            "implementation_sources": implementation_custody,
            "pf2_receipt": {
                "path": str(args.pf2),
                "bytes": len(pf2_payload),
                "sha256": args.pf2_sha256,
            },
            "pf2_composed_input_rehash": composed_inputs,
            "g4_recurrence_arrays": {
                "path": str(recurrence_path),
                "bytes": len(recurrence_payload),
                "sha256": recurrence["sha256"],
            },
            "g4_xi_tracks": {
                "path": str(tracks_path),
                "bytes": len(tracks_payload),
                "sha256": tracks["sha256"],
            },
            "v12_target_cache": target_custody,
            "j2_source_archive": {
                "path": str(args.j2_archive),
                "bytes": len(j2_payload),
                "sha256": args.j2_archive_sha256,
                "lifted_receiver_actuator_count": len(j2_actuators),
            },
            "g2f_paired_secant_receipt": {
                "path": str(args.g2f_receipt),
                "bytes": args.g2f_receipt.stat().st_size,
                "sha256": args.g2f_sha256,
            },
            "g2g_chart_symbol_receipt": {
                "path": str(args.g2g_receipt),
                "bytes": args.g2g_receipt.stat().st_size,
                "sha256": args.g2g_sha256,
            },
        },
        "assignment_table": {
            "path": portable_path(table_path),
            "bytes": len(table_bytes),
            "file_sha256": hashlib.sha256(table_bytes).hexdigest(),
            "content_sha256": table["table_content_sha256"],
            "schema": table["schema"],
        },
        "round_trip": table["round_trip"],
        "coverage": table["coverage"],
        "producer_rerun": {
            "eligible": False,
            "reason": (
                "0/1200 buckets have an exact PF2-to-J2/G2G actuator and signed-direction "
                "foreign-key join; G3 hard-pair blocks are therefore not covered."
            ),
            "ms4_harness_invoked": False,
        },
        "verdict": table["verdict"],
        "verdict_scope": table["verdict_scope"],
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "score_claim": False,
        "pointer": "0.1910828242 [contest-CPU]",
        "pointer_moved": False,
        "research_only": True,
        "main_landing_review_required": True,
    }
    receipt["receipt_content_sha256"] = canonical_sha256(receipt)
    receipt_path = args.output_directory / "ddm_ms5_pf2_bucket_assignment_receipt.json"
    publish(receipt_path, canonical_bytes(receipt))
    return receipt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pf2", type=Path, default=DEFAULT_PF2)
    parser.add_argument("--pf2-sha256", default=EXPECTED_PF2_SHA256)
    parser.add_argument("--j2-archive", type=Path, default=DEFAULT_J2_ARCHIVE)
    parser.add_argument("--j2-archive-sha256", default=EXPECTED_J2_ARCHIVE_SHA256)
    parser.add_argument("--g2f-receipt", type=Path, default=DEFAULT_G2F)
    parser.add_argument("--g2f-sha256", default=EXPECTED_G2F_SHA256)
    parser.add_argument("--g2g-receipt", type=Path, default=DEFAULT_G2G)
    parser.add_argument("--g2g-sha256", default=EXPECTED_G2G_SHA256)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    path = run(parse_args())
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
