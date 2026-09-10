# SPDX-License-Identifier: MIT
"""Move-43 integration conformance on the retained real packet, without scoring.

These tests explicitly skip where this local retained packet is unavailable;
normal CPU evaluation regressions remain portable in the existing test suite.
No synthetic runtime can stand in for the real positive custody fixture.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from tac.auth_eval_schema import (
    RAW_POLICY_CHECK,
    required_contest_cpu_axis_refusal_blockers,
    retained_json_reference,
    runtime_files_digest,
    submission_policy_adjudication,
    submission_policy_adjudication_matches,
)

REPO = Path(__file__).resolve().parents[3]
EVIDENCE = REPO / ".omx/research/ddm_cpx2_20260910"
PACKET = REPO / "submissions/_staging_move43_pr140_swap"
SHA = "7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e"
DIGEST = "ffdbde488ac17d3708b09d9b462aa62ef3b63d6d57e4198347995912df15e5ce"


@pytest.fixture(scope="module")
def real_packet():
    if not (PACKET / "archive.zip").is_file() or not (EVIDENCE / "CPU_AXIS_REFUSAL.json").is_file():
        pytest.skip("retained move-43 packet unavailable; not a synthetic positive")
    payload = json.loads((EVIDENCE / "CPU_AXIS_REFUSAL.json").read_text())
    if not Path(payload["receipt"]["path"]).is_file():
        pytest.skip("retained move-43 CPU refusal volume unavailable")
    spec = importlib.util.spec_from_file_location("cpx2_checker", REPO / "scripts/pre_submission_compliance_check.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    manifest = mod._submission_runtime_manifest(PACKET)
    archive = {"sha256": SHA, "bytes": 180466}
    assert runtime_files_digest(manifest) == DIGEST
    return mod, payload, manifest, archive


def _validate(real_packet, payload=None, *, selected_axis="contest_cuda", submission_dir=PACKET):
    _, original, manifest, archive = real_packet
    return required_contest_cpu_axis_refusal_blockers(
        original if payload is None else payload,
        selected_axis=selected_axis,
        archive=archive,
        submission_dir=submission_dir,
        runtime_manifest=manifest,
    )


def test_real_move43_positive_and_no_cpu_metrics(real_packet):
    mod, payload, _, archive = real_packet
    assert _validate(real_packet) == []
    command = json.loads((EVIDENCE / "COMPLIANCE_COMMAND.json").read_text())["argv"][2:]
    args = mod.build_arg_parser().parse_args(command)
    cuda, _ = mod.inspect_auth_eval(REPO / args.auth_eval_json, archive, args)
    record, checks = mod.inspect_contest_cpu_auth_eval(
        EVIDENCE / "CPU_AXIS_REFUSAL.json",
        archive,
        cuda,
        required=True,
        contest_final=True,
        selected_axis="contest_cuda",
        submission_dir=PACKET,
    )
    assert all(row.passed for row in checks)
    assert record["refusal_valid"] is True
    assert record["record"] is None and record["strict_formula"] is None
    assert payload["archive_sha256"] == SHA and payload["archive_bytes"] == 180466


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("archive_sha256", "0" * 64),
        ("archive_bytes", 180238),
        ("archive_bytes", True),
        ("runtime_files_sha256", "0" * 64),
        ("runtime_digest_definition", "runtime_tree_sha256"),
        ("modal_call_id", "fc-01M25W34DGHHJES03531X3VDXV"),
        ("selected_axis", "contest_cpu"),
        ("cpu_metrics_absent_by_design", False),
    ],
)
def test_binding_mutations_refused(real_packet, field, value):
    payload = copy.deepcopy(real_packet[1])
    payload[field] = value
    assert _validate(real_packet, payload)


@pytest.mark.parametrize(
    "field",
    [
        "score",
        "canonical_score",
        "score_recomputed_from_components",
        "avg_segnet_dist",
        "avg_posenet_dist",
        "pose",
        "seg",
        "rate",
        "rate_unscaled",
        "score_components",
        "metrics",
        "S",
        "d_seg",
        "d_pose",
        "n_samples",
    ],
)
def test_any_metric_field_refused_even_null(real_packet, field):
    payload = copy.deepcopy(real_packet[1])
    payload[field] = None
    assert "refusal_schema_fields_invalid_metrics_forbidden" in _validate(real_packet, payload)


@pytest.mark.parametrize("field", ["receipt", "adjudication", "receiver"])
def test_nested_metric_fields_refused(real_packet, field):
    payload = copy.deepcopy(real_packet[1])
    payload[field]["score"] = 0
    assert _validate(real_packet, payload)


@pytest.mark.parametrize("field", ["receipt", "adjudication"])
@pytest.mark.parametrize("mutation", ["sha256", "bytes", "missing", "malformed"])
def test_evidence_reference_fail_closed(real_packet, field, mutation, tmp_path):
    payload = copy.deepcopy(real_packet[1])
    if mutation == "sha256":
        payload[field]["sha256"] = "0" * 64
    elif mutation == "bytes":
        payload[field]["bytes"] += 1
    elif mutation == "missing":
        payload[field]["path"] = str(tmp_path / "missing.json")
    else:
        path = tmp_path / "bad.json"
        path.write_text("[]")
        payload[field] = retained_json_reference(path)
    assert _validate(real_packet, payload)


def test_cpu_axis_cannot_consume_refusal(real_packet):
    assert "refusal_requires_selected_contest_cuda" in _validate(real_packet, selected_axis="contest_cpu")


@pytest.mark.parametrize("field", ["archive_sha256", "archive_bytes", "runtime_files_sha256", "score"])
def test_inspector_fails_all_five_on_invalid_refusal(real_packet, field, tmp_path):
    mod, original, _, archive = real_packet
    payload = copy.deepcopy(original)
    payload[field] = 0
    path = tmp_path / "refusal.json"
    path.write_text(json.dumps(payload))
    record, checks = mod.inspect_contest_cpu_auth_eval(
        path,
        archive,
        {},
        required=True,
        contest_final=True,
        submission_dir=PACKET,
    )
    names = {
        "score_parseable",
        "archive_sha_matches",
        "archive_size_matches",
        "schema_metric_consistency",
        "runtime_tree_recorded",
    }
    selected = [c for c in checks if c.name.removeprefix("contest_cpu_auth_eval_") in names]
    assert len(selected) == 5 and not any(c.passed for c in selected)
    assert record["refusal_valid"] is False


def test_live_packet_changes_cannot_reuse_receipt(real_packet, tmp_path):
    mod, _, _, _ = real_packet
    shutil.copytree(PACKET, tmp_path / "packet")
    changed = tmp_path / "packet"
    source = changed / "inflate.py"
    source.write_text(source.read_text().replace("if not torch.cuda.is_available():", "if False:"))
    current = (mod, real_packet[1], mod._submission_runtime_manifest(changed), real_packet[3])
    blockers = _validate(current, submission_dir=changed)
    assert "refusal_retained_runtime_mismatch" in blockers
    assert "refusal_guard_does_not_raise_recorded_error" in blockers


@pytest.fixture
def reviewed_packet(real_packet):
    mod = real_packet[0]
    argv = json.loads((EVIDENCE / "COMPLIANCE_COMMAND.json").read_text())["argv"][2:]
    return mod.build_report(mod.build_arg_parser().parse_args(argv))


def test_real_policy_roundtrip_retains_release_blockers(real_packet, reviewed_packet):
    report = reviewed_packet
    assert sum(c["passed"] for c in report["checks"]) == 91
    assert len(report["checks"]) == 93
    assert report["passed"] is False
    policy = report["submission_policy_adjudication"]
    assert submission_policy_adjudication_matches(json.loads(json.dumps(policy)), current_packet_adjudication=policy)
    assert policy["release_ready"] is False
    assert policy["promotion_eligible"] is False
    assert policy["cpu_leaderboard_reproduction_eligible"] is False
    assert {row["name"] for row in policy["unresolved_release_checks"]} == {
        "submission_runtime_imports_within_allowlist",
        "hosted_archive_manifest_supplied",
    }
    assert RAW_POLICY_CHECK not in {row["name"] for row in policy["passed_checks"]}
    assert all(row["passed"] for row in policy["passed_checks"])


@pytest.mark.parametrize("failure", ["required_check", "new_failure", "missing_check", "unknown_raw", "forged_flags"])
def test_policy_cannot_launder_failures(real_packet, reviewed_packet, failure, tmp_path):
    policy = reviewed_packet["submission_policy_adjudication"]
    checks = copy.deepcopy(reviewed_packet["checks"])
    cuda_ref = policy["cuda_receipt"]
    if failure == "forged_flags":
        forged = {**policy, "release_ready": True}
        assert not submission_policy_adjudication_matches(forged, current_packet_adjudication=policy)
        return
    if failure == "required_check":
        next(c for c in checks if c["name"] == "auth_eval_archive_sha_matches")["passed"] = False
    elif failure == "new_failure":
        checks.append(asdict(real_packet[0].Check("new_packet_error", False, "error", "cannot ignore")))
    elif failure == "missing_check":
        checks = [c for c in checks if c["name"] != "contest_cpu_auth_eval_runtime_tree_recorded"]
    else:
        raw = json.loads(Path(cuda_ref["path"]).read_text())
        raw["promotion_blockers"].append("unknown_policy_blocker")
        path = tmp_path / "raw.json"
        path.write_text(json.dumps(raw))
        cuda_ref = retained_json_reference(path)
    assert (
        submission_policy_adjudication(
            archive=real_packet[3],
            runtime_files_sha256=DIGEST,
            cuda_receipt=cuda_ref,
            cpu_refusal_receipt=policy["cpu_refusal_receipt"],
            checks=checks,
        )
        is None
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "abbreviated_device",
        "abbreviated_digest",
        "abbreviated_equals_device",
        "duplicate_device",
        "duplicate_digest",
        "equals_device",
        "equals_digest",
        "bool_returncode",
        "adjudication_bool_returncode",
        "remote_metric",
        "provenance_metric",
        "different_provenance_digest",
        "cpu_eval_artifact",
        "different_outer_tree",
        "wrong_log",
    ],
)
def test_retained_evidence_corruptions_refused(real_packet, mutation, tmp_path):
    payload = copy.deepcopy(real_packet[1])
    remote = json.loads(Path(payload["receipt"]["path"]).read_text())
    adjudication = json.loads(Path(payload["adjudication"]["path"]).read_text())
    if mutation == "abbreviated_device":
        remote["command"] += ["--dev", "cuda"]
    elif mutation == "abbreviated_digest":
        remote["command"] += ["--expected-runtime-files", "0" * 64]
    elif mutation == "abbreviated_equals_device":
        remote["command"] += ["--dev=cuda"]
    elif mutation == "duplicate_device":
        remote["command"] += ["--device", "cuda"]
    elif mutation == "duplicate_digest":
        remote["command"] += ["--expected-runtime-files-sha256", "0" * 64]
    elif mutation == "equals_device":
        remote["command"] += ["--device=cuda"]
    elif mutation == "equals_digest":
        remote["command"] += ["--expected-runtime-files-sha256=" + "0" * 64]
    elif mutation == "bool_returncode":
        remote["returncode"] = True
    elif mutation == "adjudication_bool_returncode":
        adjudication["returncode"] = True
    elif mutation == "remote_metric":
        remote["canonical_score"] = 0.1
    elif mutation in {"provenance_metric", "different_provenance_digest"}:
        prov = remote["artifacts"]["provenance.json"]
        prov = json.loads(prov) if isinstance(prov, str) else prov
        if mutation == "provenance_metric":
            prov["avg_posenet_dist"] = 0.0
        else:
            prov["inflate_runtime_manifest"]["files"][0]["sha256"] = "0" * 64
        remote["artifacts"]["provenance.json"] = prov
    elif mutation == "cpu_eval_artifact":
        remote["artifacts"]["contest_auth_eval.json"] = "{}"
    elif mutation == "different_outer_tree":
        remote["expected_runtime_tree_sha256"] = "0" * 64
    else:
        remote["artifacts"]["contest_auth_eval.stderr.log"] = "some unrelated failure"
    remote_path = tmp_path / "remote.json"
    remote_path.write_text(json.dumps(remote))
    payload["receipt"] = retained_json_reference(remote_path)
    adjudication["receipt"] = payload["receipt"]
    adj_path = tmp_path / "adjudication.json"
    adj_path.write_text(json.dumps(adjudication))
    payload["adjudication"] = retained_json_reference(adj_path)
    assert _validate(real_packet, payload)


@pytest.mark.parametrize("schema", ["contest_cpu_axis_refusal.v0", "bad_version"])
def test_wrong_refusal_version_never_falls_through_to_normal_parser(real_packet, schema, tmp_path):
    mod, original, _, archive = real_packet
    payload = copy.deepcopy(original)
    payload["schema"] = schema
    path = tmp_path / "refusal.json"
    path.write_text(json.dumps(payload))
    record, checks = mod.inspect_contest_cpu_auth_eval(
        path,
        archive,
        {},
        required=True,
        contest_final=True,
        submission_dir=PACKET,
    )
    assert record["refusal_valid"] is False
    assert len([c for c in checks if not c.passed]) == 6


def test_policy_roundtrip_distinguishes_boolean_and_integer(reviewed_packet):
    policy = reviewed_packet["submission_policy_adjudication"]
    forged = {**policy, "promotion_eligible": 0}
    assert not submission_policy_adjudication_matches(forged, current_packet_adjudication=policy)
