# SPDX-License-Identifier: MIT
"""Read-only completed first-measurement custody for the release checker.

This validates historical completion evidence, not present-day permission to
fire an old intent. It never mutates or invokes pre-fire consumers.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from tac.auth_eval_schema import _read_bound_json, retained_json_reference, runtime_files_digest
from tac.candidate_seal import (
    SealContractError,
    compute_seal_sha256,
    measure_runtime_digest,
    prefire_digest,
    validate_first_measurement_runtime_custody,
)


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def validate_completed_custody(
    seal_reference: dict[str, Any], *, reconciliation: dict[str, Any],
    ledger_rows: list[dict[str, Any]], claim_rows: list[str], latest_row: str,
    lane_id: str, job_id: str, archive: dict[str, Any], submission_dir: Path,
    expected_runtime_tree_sha256: str, packet_auth_eval: dict[str, Any],
    runtime_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Bind the completed seal, ledger lifecycle and actual staged receiver bytes."""
    seal = _read_bound_json(seal_reference)
    _require(seal.get("schema") == "candidate_seal.v3" and seal.get("axis") == "contest_cuda",
             "completed CUDA v3 seal required")
    _require(seal.get("seal_sha256") == compute_seal_sha256(seal), "seal digest differs")
    receipt_ref = seal["first_measurement_receipt"]
    result = _read_bound_json(receipt_ref)
    intent = _read_bound_json(seal["prefire_intent"])
    auth = _read_bound_json(seal["first_measurement_authorization"])
    leg = _read_bound_json(seal["decode_wall_clock_reference"])
    call_id = reconciliation["call_id"]
    _require(re.fullmatch(r"fc-[A-Z0-9]+", call_id) is not None, "invalid call id")
    _require(reconciliation.get("schema") == "candidate_first_measurement_terminal_reconciliation.v1"
             and reconciliation.get("outcome") == "TERMINAL_WORKER_SUCCESS_CUSTODY_ONLY"
             and reconciliation.get("nonce_state") == "HARVESTED"
             and reconciliation.get("nonce_reusable") is False,
             "terminal success reconciliation required")
    for field, value in (("lane_id", lane_id), ("instance_job_id", job_id)):
        _require(auth.get(field) == reconciliation.get(field) == value, f"{field} differs")
    for key, seal_key in (("result", "first_measurement_receipt"), ("intent", "prefire_intent"),
                          ("authorization", "first_measurement_authorization")):
        _require(reconciliation[key] == seal[seal_key], f"reconciliation {key} differs")
    _require(intent["intent_sha256"] == prefire_digest(intent, "intent_sha256")
             == seal["prefire_intent_sha256"] == reconciliation["intent_sha256"], "intent digest differs")
    _require(auth["authorization_sha256"] == prefire_digest(auth, "authorization_sha256")
             == reconciliation["authorization_sha256"], "authorization digest differs")
    _require(auth.get("schema") == "candidate_first_measurement_authorization.v1"
             and auth.get("axis") == "contest_cuda" and auth.get("state") == "AUTHORIZED_ONCE",
             "authorization schema differs")
    intent_ref = seal["prefire_intent"]
    _require(all(auth["intent"].get(key) == value for key, value in
                 (("path", intent_ref["path"]), ("file_bytes", intent_ref["bytes"]),
                  ("file_sha256", intent_ref["sha256"]), ("digest", intent["intent_sha256"]))),
             "authorization intent file binding differs")
    _require(seal["candidate_id"] == intent["candidate_id"] == auth["candidate_id"]
             and seal["archive"] == intent["candidate"]["archive"]
             and seal["runtime"] == intent["candidate"]["runtime"]
             and seal["receiver_pins"] == intent["candidate"]["receiver_pins"], "frozen candidate differs")
    for path in (Path(seal["archive"]["path"]), submission_dir / "archive.zip"):
        ref = retained_json_reference(path)  # File identity only; does not decode the archive.
        _require(ref["sha256"] == seal["archive"]["sha256"] == archive["sha256"]
                 and type(seal["archive"]["bytes"]) is int
                 and ref["bytes"] == seal["archive"]["bytes"] == archive["bytes"], "archive differs")
    runtime = measure_runtime_digest(Path(seal["runtime"]["path"]))
    _require({"path": seal["runtime"]["path"], **runtime.to_dict()} == seal["runtime"], "sealed runtime differs on disk")
    for pin in seal["receiver_pins"]:
        name = pin["relative_path"]
        _require(name in {"inflate.sh", "inflate.py"}, "unsupported receiver pin")
        ref = retained_json_reference(submission_dir / name)
        _require(ref["sha256"] == pin["sha256"] and ref["bytes"] == pin["bytes"], "staged receiver pin differs")
    _require(len(seal["receiver_pins"]) == 2
             and {p["relative_path"] for p in seal["receiver_pins"]} == {"inflate.sh", "inflate.py"},
             "receiver pins incomplete")
    custody = validate_first_measurement_runtime_custody(
        runtime_dir=Path(seal["runtime"]["path"]), output_dir=Path(auth["output_dir"]),
        receipt_path=Path(receipt_ref["path"]))
    _require(runtime_files_digest(runtime_manifest) == custody["runtime_files_sha256"],
             "staged runtime files differ from completed seal")
    _require(custody == seal["first_measurement_runtime_custody"]
             == leg["first_measurement_runtime_custody"]
             and leg == seal["decode_wall_clock"], "runtime custody objects differ")
    for row in runtime_manifest["files"]:
        relative = Path(row["relative_path"])
        _require(not relative.is_absolute() and ".." not in relative.parts, "unsafe staged file row")
        ref = retained_json_reference(submission_dir / relative)
        _require(ref["sha256"] == row["sha256"] and ref["bytes"] == row["bytes"],
                 "staged runtime bytes changed during inspection")
    _require(auth["receipt_path"] == receipt_ref["path"]
             and leg["candidate_t4_receipt"] == {key: receipt_ref[key] for key in ("path", "sha256")}
             and leg["archive_sha256"] == archive["sha256"]
             and leg["archive_path"] == seal["archive"]["path"]
             and leg["runtime_dir"] == seal["runtime"]["path"]
             and leg["t4_runtime_sha256"] == custody["normal_projected_runtime_tree_sha256"],
             "timing leg receipt/runtime binding differs")
    _require(custody["retained_runtime_tree_sha256"] == expected_runtime_tree_sha256,
             "retained runtime tree differs from packet")
    inner = result["artifacts"]["contest_auth_eval.json"]
    if isinstance(inner, str):
        inner = json.loads(inner)
    _require(inner == packet_auth_eval, "packet evaluation differs from retained first measurement")
    _require(result.get("passed") is True and type(result.get("returncode")) is int
             and result["returncode"] == 0 and result.get("validation_errors") == []
             and result.get("expected_archive_sha256") == archive["sha256"]
             and result.get("expected_archive_size_bytes") == archive["bytes"]
             and inner.get("n_samples") == 600 and inner.get("score_axis") == "contest_cuda",
             "successful exact receipt required")
    provenance = result["artifacts"]["provenance.json"]
    if isinstance(provenance, str):
        provenance = json.loads(provenance)
    _require(runtime_files_digest(provenance["inflate_runtime_manifest"]) == custody["runtime_files_sha256"],
             "receipt files digest differs")
    for key, expected in (("call_id", call_id), ("lane_id", lane_id), ("instance_job_id", job_id),
                          ("first_measurement_authorization_sha256", auth["authorization_sha256"]),
                          ("prefire_intent_sha256", intent["intent_sha256"]),
                          ("intent_file_sha256", intent_ref["sha256"]), ("intent_file_bytes", intent_ref["bytes"]),
                          ("receipt_path", receipt_ref["path"]), ("score_axis", "contest_cuda")):
        _require(result.get(key) == expected, f"result {key} binding differs")
    nonce = _read_bound_json(reconciliation["nonce_record"])
    _require(nonce.get("state") == "HARVESTED" and nonce.get("call_id") == call_id
             and nonce.get("receipt") == receipt_ref
             and nonce.get("lane_id") == lane_id and nonce.get("instance_job_id") == job_id
             and nonce.get("prefire_intent_sha256") == intent["intent_sha256"]
             and nonce.get("intent_file_sha256") == intent_ref["sha256"]
             and nonce.get("intent_file_bytes") == intent_ref["bytes"]
             and nonce.get("authorization_sha256") == auth["authorization_sha256"]
             and nonce.get("authorization_nonce") == auth["authorization_nonce"]
             == reconciliation["authorization_nonce"], "harvested nonce binding differs")
    events = [row for row in ledger_rows if row.get("call_id") == call_id]
    dispatched = [row for row in events if row.get("event_type") == "dispatched"]
    harvested = [row for row in events if row.get("event_type") == "harvested"]
    reconciled = [row for row in events if row.get("event_type") == "reconciled_terminal_success"]
    _require(len(dispatched) == len(harvested) == len(reconciled) == 1,
             "unique dispatched/harvested/reconciled ledger events required")
    _require([row.get("event_type") for row in events] ==
             ["dispatched", "harvested", "reconciled_terminal_success"], "ledger lifecycle order differs")
    dispatch, harvest, terminal = dispatched[0], harvested[0], reconciled[0]
    _require(dispatch.get("lane_id") == lane_id and dispatch.get("instance_job_id") == job_id
             and dispatch.get("first_measurement_authorization_sha256") == auth["authorization_sha256"]
             and dispatch.get("prefire_intent_sha256") == intent["intent_sha256"]
             and dispatch.get("intent_file_sha256") == intent_ref["sha256"]
             and dispatch.get("intent_file_bytes") == intent_ref["bytes"]
             and dispatch.get("composed_archive_sha256") == archive["sha256"], "dispatch binding differs")
    _require(harvest.get("lane_id") == lane_id and harvest.get("score_axis") == "contest_cuda"
             and harvest.get("archive_sha256") == archive["sha256"]
             and harvest.get("archive_bytes") == archive["bytes"]
             and type(harvest.get("rc")) is int and harvest["rc"] == 0
             and harvest.get("harvest_result", {}).get("result_path") == receipt_ref["path"]
             and math.isclose(harvest["score"], result["score_recomputed_from_components"], rel_tol=0, abs_tol=1e-15),
             "harvest binding differs")
    terminal_result = terminal["harvest_result"]
    _require(terminal_result.get("lane_id") == lane_id and terminal_result.get("instance_job_id") == job_id
             and _read_bound_json(terminal_result["terminal_reconciliation"]) == reconciliation,
             "terminal reconciliation binding differs")
    claim = reconciliation["claim_closure"]["terminal_row"]
    text = claim["row_text"]
    raw = text.encode()
    _require(text in claim_rows and len(raw) == claim["row_bytes_without_newline"]
             and hashlib.sha256(raw).hexdigest() == claim["row_sha256_without_newline"],
             "harvested claim missing or changed")
    cells = [cell.strip() for cell in text.strip("|").split("|")]
    _require(len(cells) == 8 and cells[2] == lane_id and cells[4] == job_id
             and cells[6] == "completed_contest_cuda_exact_eval_harvested", "harvested claim identity differs")
    for token in (f"archive_sha256={archive['sha256']}", f"archive_bytes={archive['bytes']}",
                  f"runtime_tree_sha256={custody['normal_projected_runtime_tree_sha256']}",
                  f"call_id={call_id}", "axis=contest_cuda"):
        _require(token in cells[7].split(), f"harvested claim missing {token}")
    latest = [cell.strip() for cell in latest_row.strip("|").split("|")]
    _require(len(latest) == 8 and latest[2] == lane_id and latest[4] == job_id
             and latest[6].startswith("completed_pointer_move_")
             and re.findall(r"\bfc-[A-Z0-9]+\b", latest[7]) == [call_id]
             and "candidate_seal.v3 completed" in latest[7], "latest claim is not this completed seal")
    return {"valid": True, "basis": "completed_candidate_seal.v3", "seal": seal_reference,
            "receipt": receipt_ref, "call_id": call_id, "runtime_custody": custody,
            "ledger_events": [row["event_type"] for row in events]}


def inspect_completed_first_measurement_custody(
    *, repo: Path, latest_row: str, claim_rows: list[str], lane_id: str, job_id: str,
    archive: dict[str, Any], submission_dir: Path, expected_runtime_tree_sha256: str,
    packet_auth_eval: dict[str, Any], runtime_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Resolve only the current call's bound reconciliation and its seal-link file."""
    try:
        calls = re.findall(r"\bfc-[A-Z0-9]+\b", latest_row)
        _require(len(calls) == 1, "latest claim must name exactly one call")
        ledger = repo / ".omx/state/modal_call_id_ledger.jsonl"
        rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
        matches = [row for row in rows if row.get("call_id") == calls[0]
                   and row.get("event_type") == "reconciled_terminal_success"]
        _require(len(matches) == 1, "unique terminal reconciliation event required")
        ref = matches[0]["harvest_result"]["terminal_reconciliation"]
        reconciliation = _read_bound_json(ref)
        # Discovery is bounded to the reconciliation's own directory. Link booleans
        # confer no authority: every referenced object is re-read below.
        links = []
        for path in sorted(Path(ref["path"]).parent.glob("*.json")):
            payload = json.loads(path.read_text())
            if (isinstance(payload, dict)
                    and payload.get("schema") == "candidate_first_measurement_terminal_reconciliation_seal_link.v1"
                    and payload.get("reconciliation") == ref and payload.get("call_id") == calls[0]):
                links.append((path, payload))
        _require(len(links) == 1, "unique completed-seal link required")
        link_path, link = links[0]
        for key in ("intent", "authorization", "result"):
            _require(link[key] == reconciliation[key], f"seal link {key} differs")
        verified = validate_completed_custody(
            link["completed_seal"], reconciliation=reconciliation, ledger_rows=rows,
            claim_rows=claim_rows, latest_row=latest_row, lane_id=lane_id, job_id=job_id,
            archive=archive, submission_dir=submission_dir,
            expected_runtime_tree_sha256=expected_runtime_tree_sha256, packet_auth_eval=packet_auth_eval,
            runtime_manifest=runtime_manifest)
        verified.update(ledger=retained_json_reference(ledger), reconciliation=ref,
                        seal_link=retained_json_reference(link_path))
        return verified
    except (SealContractError, OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        return {"valid": False, "basis": "completed_candidate_seal.v3", "blocker": f"{type(exc).__name__}:{exc}"}
