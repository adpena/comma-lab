# SPDX-License-Identifier: MIT
"""Scorer-free move-44 integration and repinned corruption controls on real custody."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from scripts import pre_submission_compliance_check as checker
from scripts import pre_submission_first_measurement_custody as custody
from tac.auth_eval_schema import (
    required_contest_cpu_axis_refusal_blockers,
    retained_json_reference,
    staged_cpu_refusal_receiver,
)
from tac.candidate_seal import compute_seal_sha256

REPO = Path(__file__).resolve().parents[3]
PACKET = REPO / "submissions/_staging_move44_pr140_swap"
SEAL = REPO / ".omx/research/ddm_rlc5_20260910/SEAL_ddm_rlc2_counted_cure_move43_rlc5_contest_cuda_v3.json"
RECON = SEAL.parent / "custody/RUN5_TERMINAL_RECONCILIATION.json"
ARCHIVE = {"sha256": "04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e", "bytes": 180406}


@pytest.fixture(scope="module")
def real():
    if not SEAL.is_file() or not (PACKET / "shippable/archive.zip").is_file():
        pytest.skip("retained real move-44 custody unavailable; no synthetic positive")
    seal = json.loads(SEAL.read_text())
    if not Path(seal["first_measurement_receipt"]["path"]).is_file():
        pytest.skip("retained move-44 SSD receipt unavailable")
    recon = json.loads(RECON.read_text())
    command = json.loads((PACKET / "_packet/COMPLIANCE_COMMAND.json").read_text())["argv"][2:]
    args = checker.build_arg_parser().parse_args(command)
    claims = (REPO / args.dispatch_claims_md).read_text().splitlines()
    latest = next(line for line in claims if f"| {recon['instance_job_id']} |" in line)
    values = {
        "reconciliation": recon,
        "ledger_rows": [json.loads(line) for line in (REPO / ".omx/state/modal_call_id_ledger.jsonl").read_text().splitlines() if line],
        "claim_rows": claims, "latest_row": latest, "lane_id": recon["lane_id"], "job_id": recon["instance_job_id"],
        "archive": ARCHIVE, "submission_dir": PACKET / "shippable",
        "expected_runtime_tree_sha256": args.expected_runtime_tree_sha256,
        "packet_auth_eval": json.loads((PACKET / "_packet/contest_auth_eval.json").read_text()),
        "runtime_manifest": checker._submission_runtime_manifest(PACKET / "shippable"),
    }
    return seal, values, args


def test_real_move44_ten_checks_and_release_debt(real):
    _, _, args = real
    report = checker.build_report(args)
    failed = {row["name"] for row in report["checks"] if not row["passed"]}
    assert failed == {"submission_runtime_imports_within_allowlist", "hosted_archive_manifest_supplied"}
    assert len(report["checks"]) == 93 and sum(row["passed"] for row in report["checks"]) == 91
    cpu = report["contest_cpu_auth_eval"]
    assert cpu["refusal_valid"] and cpu["record"] is None and cpu["strict_formula"] is None
    assert report["dispatch_claims"]["completed_first_measurement_custody"]["valid"]
    assert report["submission_policy_adjudication"]["release_ready"] is False


def test_real_completed_seal_positive(real):
    _, values, _ = real
    verified = custody.validate_completed_custody(retained_json_reference(SEAL), **values)
    assert verified["valid"]
    assert verified["ledger_events"] == ["dispatched", "harvested", "reconciled_terminal_success"]


@pytest.mark.parametrize("field", ["sha256", "bytes"])
def test_seal_file_reference_corruption(real, field):
    _, values, _ = real
    ref = retained_json_reference(SEAL)
    ref[field] = "0" * 64 if field == "sha256" else ref[field] + 1
    with pytest.raises(ValueError, match="retained reference"):
        custody.validate_completed_custody(ref, **values)


@pytest.mark.parametrize("field", ["sha256", "bytes"])
@pytest.mark.parametrize("reference", ["first_measurement_receipt", "prefire_intent", "first_measurement_authorization", "decode_wall_clock_reference"])
def test_repinned_seal_rejects_reference_corruptions(real, tmp_path, reference, field):
    seal, values, _ = real
    changed = copy.deepcopy(seal)
    changed[reference][field] = "0" * 64 if field == "sha256" else changed[reference][field] + 1
    changed["seal_sha256"] = compute_seal_sha256(changed)
    path = tmp_path / "seal.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="retained reference"):
        custody.validate_completed_custody(retained_json_reference(path), **values)


@pytest.mark.parametrize("field", ["archive", "runtime", "receiver_pins", "first_measurement_runtime_custody", "decode_wall_clock"])
def test_repinned_seal_rejects_semantic_corruptions(real, tmp_path, field):
    seal, values, _ = real
    changed = copy.deepcopy(seal)
    if field in {"archive", "runtime"}:
        changed[field]["sha256"] = "0" * 64
    elif field == "receiver_pins":
        changed[field][0]["sha256"] = "0" * 64
    elif field == "first_measurement_runtime_custody":
        changed[field]["retained_runtime_tree_sha256"] = "0" * 64
    else:
        changed[field]["first_measurement_runtime_custody"]["runtime_files_sha256"] = "0" * 64
    changed["seal_sha256"] = compute_seal_sha256(changed)
    path = tmp_path / "seal.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match=r"differs|differ"):
        custody.validate_completed_custody(retained_json_reference(path), **values)


@pytest.mark.parametrize("event", ["dispatched", "harvested", "reconciled_terminal_success"])
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "different_call"])
def test_ledger_lifecycle_corruptions(real, event, mutation):
    _, original, _ = real
    values = copy.deepcopy(original)
    call = values["reconciliation"]["call_id"]
    row = next(r for r in values["ledger_rows"] if r.get("call_id") == call and r.get("event_type") == event)
    if mutation == "missing":
        values["ledger_rows"].remove(row)
    elif mutation == "duplicate":
        values["ledger_rows"].append(copy.deepcopy(row))
    else:
        row["call_id"] = "fc-DIFFERENT"
    with pytest.raises(ValueError, match="unique"):
        custody.validate_completed_custody(retained_json_reference(SEAL), **values)


@pytest.mark.parametrize("event,field,value", [
    ("dispatched", "instance_job_id", "different-job"),
    ("dispatched", "first_measurement_authorization_sha256", "0" * 64),
    ("harvested", "archive_sha256", "0" * 64),
    ("harvested", "archive_bytes", 180466),
    ("harvested", "rc", True),
    ("harvested", "score_axis", "contest_cpu"),
    ("harvested", "score", 0.01),
])
def test_ledger_binding_corruptions(real, event, field, value):
    _, original, _ = real
    values = copy.deepcopy(original)
    row = next(r for r in values["ledger_rows"] if r.get("call_id") == values["reconciliation"]["call_id"] and r.get("event_type") == event)
    row[field] = value
    with pytest.raises(ValueError, match="binding differs"):
        custody.validate_completed_custody(retained_json_reference(SEAL), **values)


@pytest.mark.parametrize("mutation", ["claim_missing", "latest_failed", "different_packet", "different_tree", "staged_manifest"])
def test_packet_and_claim_corruptions(real, mutation):
    _, original, _ = real
    values = copy.deepcopy(original)
    if mutation == "claim_missing":
        values["claim_rows"].remove(values["reconciliation"]["claim_closure"]["terminal_row"]["row_text"])
    elif mutation == "latest_failed":
        values["latest_row"] = values["latest_row"].replace("completed_pointer_move_", "failed_pointer_move_")
    elif mutation == "different_packet":
        values["packet_auth_eval"]["archive_size_bytes"] = 180466
    elif mutation == "different_tree":
        values["expected_runtime_tree_sha256"] = "0" * 64
    else:
        values["runtime_manifest"]["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        custody.validate_completed_custody(retained_json_reference(SEAL), **values)


@pytest.fixture
def refusal(real):
    payload = json.loads((PACKET / "_packet/CPU_AXIS_REFUSAL.json").read_text())
    remote = json.loads(Path(payload["receipt"]["path"]).read_text())
    return payload, remote["artifacts"]["contest_auth_eval.stderr.log"]


def test_staged_launcher_line_is_derived(real, refusal):
    payload, stderr = refusal
    actual = staged_cpu_refusal_receiver(PACKET / "shippable", payload["receiver"], stderr)
    assert payload["receiver"]["launcher_line"] == 83 and actual["launcher_line"] == 86


@pytest.mark.parametrize("guard", ["if False:", "if torch.cuda.is_available():", "if advisory_cpu and not torch.cuda.is_available():"])
def test_nonraising_cpu_guard_rejected(real, refusal, tmp_path, guard):
    payload, stderr = refusal
    changed = tmp_path / "packet"
    shutil.copytree(PACKET / "shippable", changed)
    source = changed / "inflate.py"
    source.write_text(source.read_text().replace(payload["receiver"]["guard_text"], guard))
    payload["receiver"]["guard_text"] = guard
    with pytest.raises(ValueError, match="CPU guard does not raise"):
        staged_cpu_refusal_receiver(changed, payload["receiver"], stderr)
    blockers = required_contest_cpu_axis_refusal_blockers(
        payload, selected_axis="contest_cuda", archive=ARCHIVE, submission_dir=changed,
        runtime_manifest=checker._submission_runtime_manifest(changed))
    assert "refusal_retained_runtime_mismatch" in blockers
    assert "refusal_guard_does_not_raise_recorded_error" in blockers


def test_equivalent_guard_form_is_parsed_not_literal_matched(refusal, tmp_path):
    payload, stderr = refusal
    changed = tmp_path / "packet"
    shutil.copytree(PACKET / "shippable", changed)
    source = changed / "inflate.py"
    guard = "if (torch.cuda.is_available() is False) and (advisory_cpu == False):"
    source.write_text(source.read_text().replace(payload["receiver"]["guard_text"], guard))
    payload["receiver"]["guard_text"] = guard
    assert staged_cpu_refusal_receiver(changed, payload["receiver"], stderr)["guard_text"] == guard


@pytest.mark.parametrize("mutation", ["wrong_sha", "wrong_size", "move43_receipt"])
def test_cpu_archive_and_previous_move_refused(real, refusal, mutation):
    _, values, _ = real
    payload, _ = refusal
    if mutation == "wrong_sha":
        payload["archive_sha256"] = "0" * 64
    elif mutation == "wrong_size":
        payload["archive_bytes"] = 180466
    else:
        old = json.loads((REPO / ".omx/research/ddm_cpx2_20260910/CPU_AXIS_REFUSAL.json").read_text())
        for field in ("receipt", "adjudication", "modal_call_id"):
            payload[field] = old[field]
    blockers = required_contest_cpu_axis_refusal_blockers(
        payload, selected_axis="contest_cuda", archive=ARCHIVE, submission_dir=PACKET / "shippable",
        runtime_manifest=values["runtime_manifest"])
    assert blockers


def test_normal_harvest_path_does_not_consult_v3(real, tmp_path, monkeypatch):
    _, values, _ = real
    row = values["reconciliation"]["claim_closure"]["terminal_row"]["row_text"]
    prior = next(r for r in values["claim_rows"] if f"| {values['job_id']} |" in r and "active_modal_auth_eval_spawned" in r)
    path = tmp_path / "claims.md"
    path.write_text("| time | agent | lane_id | platform | instance/job_id | expiry | status | notes |\n" + row + "\n" + prior + "\n")
    def forbidden(**kwargs):
        raise AssertionError("normal path consulted first-measurement custody")
    monkeypatch.setattr(custody, "inspect_completed_first_measurement_custody", forbidden)
    _, checks = checker.inspect_dispatch_claims(
        path, values["lane_id"], values["job_id"], require_successful_exact_eval_terminal=True,
        expected_archive_sha256=ARCHIVE["sha256"],
        expected_runtime_tree_sha256=real[0]["first_measurement_runtime_custody"]["normal_projected_runtime_tree_sha256"],
        submission_dir=values["submission_dir"], archive=ARCHIVE, packet_auth_eval=values["packet_auth_eval"])
    assert all(row.passed for row in checks)


@pytest.mark.parametrize("source", ["context", "request", "result", "argv", "provenance"])
def test_repinned_runtime_source_references_refuse(real, tmp_path, source):
    seal, values, _ = real
    changed = copy.deepcopy(seal)
    changed["first_measurement_runtime_custody"]["sources"][source]["file"]["sha256"] = "0" * 64
    changed["seal_sha256"] = compute_seal_sha256(changed)
    path = tmp_path / "seal.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="runtime custody objects differ"):
        custody.validate_completed_custody(retained_json_reference(path), **values)


def test_live_staged_file_rehashed_even_with_stale_manifest(real, tmp_path):
    _, original, _ = real
    values = copy.deepcopy(original)
    stage = tmp_path / "packet"
    shutil.copytree(PACKET / "shippable", stage)
    changed = stage / "runtime/rc3_shared_mixer.py"
    changed.write_bytes(changed.read_bytes() + b"\n# corruption control\n")
    values["submission_dir"] = stage
    with pytest.raises(ValueError, match="staged runtime bytes changed"):
        custody.validate_completed_custody(retained_json_reference(SEAL), **values)


@pytest.mark.parametrize("change", ["missing", "wrong_ref_sha", "duplicate", "wrong_call"])
def test_discovery_refuses_bad_reconciliation_event(real, tmp_path, change):
    _, original, _ = real
    values = copy.deepcopy(original)
    call = values["reconciliation"]["call_id"]
    rows = values["ledger_rows"]
    target = next(row for row in rows if row.get("call_id") == call and row.get("event_type") == "reconciled_terminal_success")
    if change == "missing":
        rows.remove(target)
    elif change == "wrong_ref_sha":
        target["harvest_result"]["terminal_reconciliation"]["sha256"] = "0" * 64
    elif change == "duplicate":
        rows.append(copy.deepcopy(target))
    else:
        target["call_id"] = "fc-DIFFERENT"
    ledger = tmp_path / ".omx/state/modal_call_id_ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    for key in ("reconciliation", "ledger_rows"):
        values.pop(key)
    verified = custody.inspect_completed_first_measurement_custody(repo=tmp_path, **values)
    assert verified["valid"] is False and verified["blocker"]


@pytest.mark.parametrize("field", ["call_id", "lane_id", "instance_job_id", "first_measurement_authorization_sha256", "intent_file_sha256"])
def test_result_identity_checked_after_bound_read(real, monkeypatch, field):
    # Attack the downstream identity check independently of the file-hash check.
    seal, values, _ = real
    original_read = custody._read_bound_json
    def altered_read(ref):
        payload = original_read(ref)
        if ref == seal["first_measurement_receipt"]:
            payload[field] = "different"
        return payload
    monkeypatch.setattr(custody, "_read_bound_json", altered_read)
    with pytest.raises(ValueError, match=f"result {field} binding differs"):
        custody.validate_completed_custody(retained_json_reference(SEAL), **values)
