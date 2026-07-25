#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize the bounded local DDM WF7 seven-home waterfill receipt."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for local_path in (str(SRC), str(REPO)):
    if local_path not in sys.path:
        sys.path.insert(0, local_path)

from tac.optimization.ddm_runtime_exporter import _compile_seed_state  # noqa: E402
from tac.optimization.ddm_wf7_seven_home_stream_waterfill import (  # noqa: E402
    STATE_BYTES,
    STATE_SHA256,
    WF7Error,
    build_best_candidate,
    inspect_seeded_state,
    sha256_bytes,
)

CONFIG_SCHEMA: Final = "ddm_wf7_seven_home_stream_waterfill_config.v1"
RECEIPT_SCHEMA: Final = "ddm_wf7_seven_home_stream_waterfill_receipt.v1"
BOX = {
    "d_seg_max": 0.00116,
    "d_pose_max": 0.00161,
    "archive_bytes_max": 200_000,
}


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _read_bound(reference: Mapping[str, Any], label: str) -> tuple[bytes, dict[str, Any]]:
    raw_path = reference.get("path")
    expected_sha = reference.get("sha256")
    expected_bytes = reference.get("bytes")
    if not isinstance(raw_path, str) or not isinstance(expected_sha, str) or type(expected_bytes) is not int:
        raise WF7Error(f"{label} custody reference is incomplete")
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO / path
    payload = path.resolve(strict=True).read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if len(payload) != expected_bytes or observed != expected_sha:
        raise WF7Error(f"{label} custody differs")
    return payload, {"path": raw_path, "bytes": len(payload), "sha256": observed}


def _json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WF7Error(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise WF7Error(f"{label} must be a JSON object")
    return value


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise WF7Error(f"immutable output differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise WF7Error(f"stale atomic-write scratch blocks output: {temporary}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _outer_home_sizes(archive_payload: bytes) -> dict[str, int]:
    try:
        with zipfile.ZipFile(io.BytesIO(archive_payload), "r") as archive:
            infos = archive.infolist()
            stops = [info.header_offset for info in infos[1:]] + [archive.start_dir]
            rows = {info.filename: stop - info.header_offset for info, stop in zip(infos, stops, strict=True)}
            rows["__central_directory_and_eocd__"] = len(archive_payload) - archive.start_dir
            for info in infos:
                archive.read(info)
            return rows
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise WF7Error("outer-home source archive is malformed") from exc


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise WF7Error("WF7 config schema differs")
    for field, expected in (
        ("research_only", True),
        ("execution_allowed", False),
        ("score_claim", False),
        ("promotion_eligible", False),
        ("pointer_moved", False),
        ("main_landing_review_required", True),
    ):
        if config.get(field) is not expected:
            raise WF7Error(f"config.{field} must be {expected}")
    delegation = config.get("delegation")
    if not isinstance(delegation, Mapping) or delegation.get("checkpoint_key") != (
        "codex_delegate:ddm_wf7_seven_home_stream_waterfill:20260725T203257Z"
    ):
        raise WF7Error("WF7 delegation checkpoint key differs")
    if config.get("effective_frontier") != "official leaderboard displayed 0.172":
        raise WF7Error("WF7 effective-frontier correction is absent")


def _cc3_row(cc3: Mapping[str, Any], cc2: Mapping[str, Any]) -> dict[str, Any]:
    if (
        cc3.get("schema") != "ddm_cc3_mixed_coder_receiver_integration_mirror.v1"
        or cc3.get("archive", {}).get("bytes") != 136_116
        or cc3.get("source_archive", {}).get("bytes") != 139_538
        or cc3.get("admission", {}).get("mixed_vs_raw_receiver_output_byte_identical") is not True
        or cc3.get("admission", {}).get("all_135_cc2_canonical_frames_parseback_exact") is not True
    ):
        raise WF7Error("CC3 finite-row custody differs")
    race3 = cc2.get("race3")
    rows = race3.get("rows") if isinstance(race3, Mapping) else None
    if not isinstance(rows, list) or len(rows) != 27:
        raise WF7Error("CC2 physical-leaf table differs")
    selected = [row for row in rows if int(row.get("delta_bytes", 0)) < 0]
    if len(selected) != 8 or sum(int(row["delta_bytes"]) for row in selected) != -3_422:
        raise WF7Error("CC2 selected row census differs")
    v15_prefix = "composition.zip!/parent/ws1.zip!/base/ddm_v16_receiver.zip!/base/ddm_v15_receiver.zip!/"
    nested_v15 = [row for row in selected if str(row["stream_id"]).startswith(v15_prefix)]
    nested_v15_delta = sum(int(row["delta_bytes"]) for row in nested_v15)
    return {
        "row_id": "cc3_exact_recursive_composition_bundle",
        "seat_order": 0,
        "authority": "MEASURED_RECEIVER_CLOSED_N600",
        "source_archive_bytes": 139_538,
        "candidate_archive_bytes": 136_116,
        "delta_bytes": -3_422,
        "delta_d_seg": 0.0,
        "delta_d_pose": 0.0,
        "delta_joint_score": -0.0022785693375840703,
        "endpoint": cc3["endpoint"],
        "selected_leaf_count": 8,
        "nested_v15_selected_leaf_count": len(nested_v15),
        "nested_v15_leaf_delta_bytes": nested_v15_delta,
        "external_wrapper_delta_bytes": -3_422 - nested_v15_delta,
        "waterfill_relation": (
            "ALTERNATIVE_SAME_DESCRIBE_POOL_DIFFERENT_EXACT_COMPOSITION_OBJECT; "
            "SEATED_AS_FINITE_FALSIFIER_BUT_NOT_ADDED_TO_WF7_HOME_DELTAS"
        ),
        "verdict_scope": cc3["verdict_scope"],
    }


def materialize(config_path: Path) -> dict[str, Any]:
    config_payload = config_path.resolve(strict=True).read_bytes()
    config = _json(config_payload, "config")
    _validate_config(config)
    source_payloads: dict[str, bytes] = {}
    source_custody: dict[str, dict[str, Any]] = {}
    sources = config.get("sources")
    if not isinstance(sources, Mapping):
        raise WF7Error("config.sources must be an object")
    for label, reference in sources.items():
        if not isinstance(reference, Mapping):
            raise WF7Error(f"config.sources.{label} must be an object")
        payload, custody = _read_bound(reference, str(label))
        source_payloads[str(label)] = payload
        source_custody[str(label)] = custody

    source_archive = source_payloads["c1_source_archive"]
    state, _receiver, dofs = _compile_seed_state(source_archive)
    if (len(state), sha256_bytes(state)) != (STATE_BYTES, STATE_SHA256):
        raise WF7Error("reconstructed seeded-state identity differs")
    candidate, waterfill = build_best_candidate(state)
    candidate_path = Path(str(config["candidate_output_path"]))
    _write_immutable(candidate_path, candidate)

    source_homes = _outer_home_sizes(source_archive)
    state_homes = {row.name: row for row in inspect_seeded_state(state)}
    seed_delta_rows = []
    for name in state_homes:
        before = source_homes.get(name, 0)
        after = state_homes[name].counted_bytes
        seed_delta_rows.append(
            {"physical_home": name, "before_bytes": before, "after_bytes": after, "delta_bytes": after - before}
        )
    if sum(row["delta_bytes"] for row in seed_delta_rows) != 270:
        raise WF7Error("logical lane-seed delta does not reconcile to physical homes")

    cc3 = _json(source_payloads["cc3"], "CC3")
    cc2 = _json(source_payloads["cc2"], "CC2")
    cc3_row = _cc3_row(cc3, cc2)
    c1_ledger = _json(source_payloads["c1_ledger"], "C1 ledger")
    e4_harness = _json(source_payloads["e4_harness"], "E4 harness")
    if (
        c1_ledger.get("control", {}).get("seeded_archive_bytes") != STATE_BYTES
        or e4_harness.get("parsed_report", {}).get("archive_bytes") != 344_203
        or e4_harness.get("status") != "PASS"
    ):
        raise WF7Error("C1/E4 endpoint custody differs")

    c1_endpoint = {
        "d_seg": float(c1_ledger["control"]["d_seg"]),
        "d_pose": float(c1_ledger["control"]["d_pose"]),
        "state_bytes": STATE_BYTES,
        "evidence_axis": c1_ledger["evidence_axis"],
    }
    e4_endpoint = {
        "d_seg": float(e4_harness["parsed_report"]["d_seg"]),
        "d_pose": float(e4_harness["parsed_report"]["d_pose"]),
        "archive_bytes": int(e4_harness["parsed_report"]["archive_bytes"]),
        "evidence_axis": e4_harness["evidence_axis"],
    }
    cc3_endpoint = {
        "d_seg": float(cc3["endpoint"]["d_seg"]),
        "d_pose": float(cc3["endpoint"]["d_pose"]),
        "archive_bytes": int(cc3["archive"]["bytes"]),
        "evidence_axis": cc3["endpoint"]["evidence_axis"],
    }
    measured_box_rows = []
    for row_id, endpoint in (("CC3", cc3_endpoint), ("E4", e4_endpoint)):
        measured_box_rows.append(
            {
                "row_id": row_id,
                **endpoint,
                "inside_d_seg": endpoint["d_seg"] <= BOX["d_seg_max"],
                "inside_d_pose": endpoint["d_pose"] <= BOX["d_pose_max"],
                "inside_bytes": endpoint["archive_bytes"] <= BOX["archive_bytes_max"],
                "inside_box": (
                    endpoint["d_seg"] <= BOX["d_seg_max"]
                    and endpoint["d_pose"] <= BOX["d_pose_max"]
                    and endpoint["archive_bytes"] <= BOX["archive_bytes_max"]
                ),
            }
        )
    wf7_diagnostic = {
        "d_seg": c1_endpoint["d_seg"],
        "d_pose": c1_endpoint["d_pose"],
        "state_container_bytes": waterfill["candidate"]["bytes"],
        "inside_d_seg": c1_endpoint["d_seg"] <= BOX["d_seg_max"],
        "inside_d_pose": c1_endpoint["d_pose"] <= BOX["d_pose_max"],
        "inside_state_byte_ceiling": waterfill["candidate"]["bytes"] <= BOX["archive_bytes_max"],
        "inside_box": False,
        "authority": (
            "DERIVED_IDENTICAL_STATE_RECEIVER_OUTPUT_PLUS_MEASURED_STATE_CONTAINER_BYTES; "
            "NOT_AN_E4_OR_CONTEST_PACKET_TRIPLE"
        ),
        "reason": (
            "Distortion fails the box before runtime binding; WF7 restores the exact C1 state, "
            "but no E4 state-container consumer has been materialized."
        ),
    }

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "run_id": config["run_id"],
        "generated_at_utc": config["generated_at_utc"],
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "authority": dict(config["delegation"]),
        "typed_config": {
            "path": config_path.resolve().relative_to(REPO).as_posix(),
            "bytes": len(config_payload),
            "sha256": hashlib.sha256(config_payload).hexdigest(),
        },
        "source_custody": source_custody,
        "seeded_state": {"bytes": len(state), "sha256": sha256_bytes(state), "dofs": dofs},
        "logical_to_physical_lane_seed_reconciliation": {
            "logical_delta_bytes": 270,
            "physical_rows": seed_delta_rows,
            "conserved": True,
            "finding": ("EV2 lane_program_seed=270 is an accounting delta, not one contiguous physical stream."),
        },
        "finite_rows": [cc3_row, *waterfill["rows"]],
        "cc3_first_seat": cc3_row,
        "seven_home_waterfill": waterfill,
        "candidate_artifact": {
            "path": candidate_path.as_posix(),
            "bytes": len(candidate),
            "sha256": sha256_bytes(candidate),
            "rebuild_command": [
                sys.executable,
                "tools/run_ddm_wf7_seven_home_stream_waterfill.py",
                "--config",
                config_path.resolve().relative_to(REPO).as_posix(),
            ],
            "rebuildable": True,
            "cold_store": "SSD_PRIMARY",
            "score_claim": False,
        },
        "box": BOX,
        "measured_receiver_closed_box_rows": measured_box_rows,
        "wf7_state_container_diagnostic": wf7_diagnostic,
        "waterfill_decision": {
            "finite_strictly_improving_row_exists": True,
            "selected_home_count": waterfill["selected_negative_home_count"],
            "selected_state_container_bytes": waterfill["candidate"]["bytes"],
            "selected_state_container_delta_bytes": waterfill["delta_bytes"],
            "selected_state_container_delta_rate_score": waterfill["delta_rate_score"],
            "cheapest_measured_box_member": None,
            "status": "FINITE_RATE_PRICES_MEASURED_NO_BOX_MEMBER",
            "reason": (
                "CC3 and E4 measured endpoints both fail d_seg and d_pose; WF7 state recoding is lossless "
                "and therefore cannot repair either distortion leg."
            ),
        },
        "cross_granularity_firewall": {
            "pf3_or_pf3b_prices_added": False,
            "cc3_delta_added_to_wf7_delta": False,
            "reason": "same describe pool, alternative objects/granularities; deltas are falsifiers, not additive credits",
        },
        "effective_frontier": config["effective_frontier"],
        "local_baseline": "0.1910828242 [contest-CPU]",
        "pointer_moved": False,
        "score_claim": False,
        "promotion_eligible": False,
        "research_only": True,
        "execution_allowed": False,
        "main_landing_review_required": True,
        "verdict": "STREAM_PRICE_DOMAIN_NONEMPTY_RATE_ONLY;NO_613_BOX_MEMBER",
        "verdict_scope": (
            "INSTANCE x exact 134211-byte seeded C1 state x settled five-coder menu x exact CC3 composition "
            "and current E4 receiver endpoints. Lossless stream recoding is actionable for rate, but this "
            "does not close lossy stream moves, another endpoint, the describe family, or contest promotion."
        ),
        "stores_consulted": config["stores_consulted"],
        "canonical_laws": {
            "rate_delta": "delta_S_rate = 25 * delta_archive_bytes / 37545489",
            "lossless_identity": "decode(candidate)=state => delta_d_seg=delta_d_pose=0 for a deterministic receiver",
            "nonadditivity": "alternative same-pool granularities are compared on a common object; their deltas do not sum",
        },
    }
    receipt["receipt_content_sha256"] = hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
    output_dir = REPO / str(config["output_dir"])
    _write_immutable(output_dir / "receipt.json", _canonical_bytes(receipt))
    _write_immutable(output_dir / "seven_home_price_table.json", _canonical_bytes({"rows": waterfill["rows"]}))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    receipt = materialize(args.config)
    print(
        json.dumps(
            {
                "verdict": receipt["verdict"],
                "candidate": receipt["seven_home_waterfill"]["candidate"],
                "delta_bytes": receipt["seven_home_waterfill"]["delta_bytes"],
                "cheapest_measured_box_member": receipt["waterfill_decision"]["cheapest_measured_box_member"],
                "pointer_moved": receipt["pointer_moved"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
