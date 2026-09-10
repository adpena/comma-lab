"""Synthetic contract controls, never empirical timing evidence or score claims."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tac.candidate_seal import (
    SEAL_DECODE_WALL_CLOCK_INVALID,
    SEAL_DECODE_WALL_CLOCK_MISSING,
    SEAL_VALID,
    SealContractError,
    compute_seal_sha256,
    load_seal,
    measure_archive_identity,
    measure_runtime_digest,
    validate_seal,
    write_seal,
)
from tac.decode_wall_clock import (
    CALIBRATION_SCHEMA,
    LOCAL_SCHEMA,
    build_decode_wall_clock,
    inherit_decode_wall_clock,
    measure_receiver_digest,
    measure_t4_runtime_digest,
    receipt_reference,
    validate_decode_wall_clock,
)


def _json(path: Path, doc: dict) -> Path:
    path.write_text(json.dumps(doc))
    return path


def timing_fixture(runtime: Path, root: Path) -> dict:
    """Construct explicit synthetic positive control receipts outside the shipped runtime."""
    root.mkdir(parents=True, exist_ok=True)
    archive = runtime / "archive.zip"
    archive_sha = measure_archive_identity(archive).sha256
    receiver_sha = measure_receiver_digest(runtime)
    local_path = _json(root / "local.json", {
        "schema": LOCAL_SCHEMA, "axis": "macOS-CPU advisory", "measurement_kind": "cuda_path_equivalent_proxy",
        "completed": True, "cold_start": True, "cpu_threads": 4, "frames": list(range(20)), "wall_seconds": 40,
        "host": "synthetic-test-host", "command": ["synthetic-control"],
        "concurrency": {"competing_process_count": 0, "load_average": [0, 0, 0]},
        "margin_time_basis": "quiesced", "runtime_dir": str(runtime), "archive_path": str(archive),
        "runtime_sha256": measure_runtime_digest(runtime).sha256, "receiver_sha256": receiver_sha,
        "archive_sha256": archive_sha,
    })
    t4_path = _json(root / "t4.json", {
        "passed": True, "returncode": 0, "expected_archive_sha256": archive_sha,
        "expected_runtime_tree_sha256": measure_t4_runtime_digest(runtime),
        "artifacts": {
            "contest_auth_eval.json": json.dumps({"inflate_elapsed_seconds": 900, "evaluate_elapsed_seconds": 45}),
            "modal_cuda_preflight.json": json.dumps({"torch_cuda_device_name": "Tesla T4"}),
        },
    })
    calibration_path = _json(root / "calibration.json", {
        "schema": CALIBRATION_SCHEMA, "name": "synthetic-test-calibration", "local_receipt": receipt_reference(local_path),
        "t4_receipt": receipt_reference(t4_path), "t4_seconds_field": ["artifacts", "contest_auth_eval.json", "inflate_elapsed_seconds"],
        "t4_archive_sha256_field": ["expected_archive_sha256"],
        "t4_hardware_field": ["artifacts", "modal_cuda_preflight.json", "torch_cuda_device_name"],
        "t4_runtime_sha256_field": ["expected_runtime_tree_sha256"],
        "t4_runtime_digest_definition": "tac.deploy.modal.auth_eval.modal_uploaded_submission_dir_runtime_manifest",
        "t4_timing_scope": "decode", "archive_sha256": archive_sha, "receiver_sha256": receiver_sha,
        "t4_hardware": "nvidia-t4", "cpu_to_t4_ratio": 0.75,
    })
    return build_decode_wall_clock(local_receipt_path=local_path, calibration_receipt_path=calibration_path,
                                   runtime_dir=runtime, archive_path=archive)


@pytest.fixture
def case(tmp_path):
    from tac.tests.test_candidate_seal import _stage_candidate
    runtime, archive = _stage_candidate(tmp_path)
    leg = timing_fixture(runtime, tmp_path / "timing")
    return runtime, archive, leg


def _validate(case, leg=None):
    runtime, archive, original = case
    return validate_decode_wall_clock(original if leg is None else leg, runtime_dir=runtime, archive_path=archive)[0]


def test_real_receipt_arithmetic_passes(case):
    assert not _validate(case)
    assert case[2]["local_600_seconds"] == 1200
    assert case[2]["projected_t4_decode_seconds"] == 900


@pytest.mark.parametrize("field,value", [("wall_seconds", float("nan")), ("wall_seconds", 0),
    ("cold_start", False), ("resumed", True), ("checkpoint_resumed_from_frame", 20),
    ("cpu_threads", 1), ("cpu_threads", True), ("frames", list(range(19))),
    ("frames", list(range(0, 40, 2))), ("completed", False), ("measurement_kind", "token_only"),
    ("concurrency", {}), ("margin_time_basis", "estimated"), ("host", "")])
def test_bad_local_receipts_refuse_even_when_rehashed(case, field, value):
    leg = copy.deepcopy(case[2])
    path = Path(leg["local_receipt"]["path"])
    doc = json.loads(path.read_text())
    doc[field] = value
    _json(path, doc)
    leg["local_receipt"] = receipt_reference(path)
    cal_path = Path(leg["calibration_receipt"]["path"])
    cal = json.loads(cal_path.read_text())
    cal["local_receipt"] = receipt_reference(path)
    _json(cal_path, cal)
    leg["calibration_receipt"] = receipt_reference(cal_path)
    assert _validate(case, leg)


@pytest.mark.parametrize("field,value", [("local_600_seconds", 1), ("cpu_to_t4_ratio", 0.1),
    ("projected_t4_decode_seconds", float("nan")), ("limit_seconds", 1800), ("score_claim", True),
    ("mode", "grandfathered"), ("schema", "candidate_decode_wall_clock.v99")])
def test_leg_cannot_forge_arithmetic_or_authority(case, field, value):
    leg = copy.deepcopy(case[2])
    leg[field] = value
    assert _validate(case, leg)


def test_receipt_byte_drift_refuses(case):
    path = Path(case[2]["local_receipt"]["path"])
    path.write_text(path.read_text() + " ")
    assert "receipt drift" in _validate(case)[0]


def test_candidate_runtime_drift_refuses(case):
    (case[0] / "new_decode.py").write_text("def decode(): return 1\n")
    assert "receiver differs" in _validate(case)[0]


def test_only_pin_literals_normalized(case):
    receiver = case[0] / "inflate.py"
    original = measure_receiver_digest(case[0])
    text = receiver.read_text()
    archive_sha = measure_archive_identity(case[1]).sha256
    receiver.write_text(text.replace(archive_sha, "a" * 64))
    assert measure_receiver_digest(case[0]) == original
    receiver.write_text(receiver.read_text() + "\n# changed receiver\n")
    assert measure_receiver_digest(case[0]) != original


def test_inheritance_requires_matching_pointer_and_real_source(case, tmp_path):
    from tac.tests.test_candidate_seal import _stage_candidate
    source = _json(tmp_path / "source.json", case[2])
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    runtime, archive = _stage_candidate(other_dir, payload=b"new payload")
    inherited = inherit_decode_wall_clock(source_leg_path=source, runtime_dir=runtime, archive_path=archive,
        pointer_archive_sha256=case[2]["archive_sha256"])
    assert not validate_decode_wall_clock(inherited, runtime_dir=runtime, archive_path=archive,
        pointer_archive_sha256=case[2]["archive_sha256"])[0]
    assert validate_decode_wall_clock(inherited, runtime_dir=runtime, archive_path=archive,
        pointer_archive_sha256="e" * 64)[0]
    (runtime / "inflate.sh").write_text("echo changed\n")
    with pytest.raises(SealContractError, match="receiver code differs"):
        inherit_decode_wall_clock(source_leg_path=source, runtime_dir=runtime, archive_path=archive,
            pointer_archive_sha256=case[2]["archive_sha256"])


def test_legacy_seal_readable_but_fire_requires_leg(tmp_path):
    from tac.tests.test_candidate_seal import _seal, _stage_candidate, _write_pointer
    runtime, _ = _stage_candidate(tmp_path)
    path = _seal(tmp_path, runtime)
    doc = load_seal(path)
    del doc["decode_wall_clock"]
    doc["schema"] = "candidate_seal.v1"
    doc["seal_sha256"] = compute_seal_sha256(doc)
    write_seal(doc, path)
    result = validate_seal(path, pointer_path=_write_pointer(tmp_path))
    assert result.verdict == SEAL_VALID
    assert result.observed["decode_wall_clock"] == "absent"
    assert validate_seal(path, check_pointer=False, require_decode_wall_clock=True).verdict == SEAL_DECODE_WALL_CLOCK_MISSING
    doc["schema"] = "candidate_seal.v2"
    doc["seal_sha256"] = compute_seal_sha256(doc)
    write_seal(doc, path)
    assert validate_seal(path, check_pointer=False).verdict == SEAL_DECODE_WALL_CLOCK_MISSING


def test_malformed_leg_returns_refusal_not_exception(tmp_path):
    from tac.tests.test_candidate_seal import _seal, _stage_candidate
    runtime, _ = _stage_candidate(tmp_path)
    path = _seal(tmp_path, runtime)
    doc = load_seal(path)
    doc["decode_wall_clock"] = ["malformed"]
    doc["seal_sha256"] = compute_seal_sha256(doc)
    write_seal(doc, path)
    assert validate_seal(path, check_pointer=False).verdict == SEAL_DECODE_WALL_CLOCK_INVALID


def test_actual_t4_failure_refuses_even_unknown_concurrency(case):
    runtime, archive, leg = case
    local_path = Path(leg["local_receipt"]["path"])
    local = json.loads(local_path.read_text())
    local["concurrency"]["competing_process_count"] = None
    local["margin_time_basis"] = "unknown"
    _json(local_path, local)
    cal_path = Path(leg["calibration_receipt"]["path"])
    cal = json.loads(cal_path.read_text())
    t4_path = Path(cal["t4_receipt"]["path"])
    t4 = json.loads(t4_path.read_text())
    auth = json.loads(t4["artifacts"]["contest_auth_eval.json"])
    auth["inflate_elapsed_seconds"] = 1336.668977565
    t4["artifacts"]["contest_auth_eval.json"] = json.dumps(auth)
    _json(t4_path, t4)
    cal["t4_receipt"] = receipt_reference(t4_path)
    cal["local_receipt"] = receipt_reference(local_path)
    cal["cpu_to_t4_ratio"] = auth["inflate_elapsed_seconds"] / 1200
    _json(cal_path, cal)
    rejected = build_decode_wall_clock(local_receipt_path=local_path, calibration_receipt_path=cal_path,
        runtime_dir=runtime, archive_path=archive, enforce_margin=False)
    assert "measured T4 decode" in rejected["validation_problems_at_construction"][0]
    assert "1336.668978s exceeds 1260.0s" in _validate(case, rejected)[0]
    source = _json(cal_path.parent / "rejected.json", rejected)
    with pytest.raises(SealContractError):
        inherit_decode_wall_clock(source_leg_path=source, runtime_dir=runtime, archive_path=archive,
            pointer_archive_sha256=rejected["archive_sha256"])


def test_below_margin_loaded_measurement_still_refuses(case):
    leg = copy.deepcopy(case[2])
    path = Path(leg["local_receipt"]["path"])
    doc = json.loads(path.read_text())
    doc["concurrency"]["competing_process_count"] = 2
    _json(path, doc)
    leg["local_receipt"] = receipt_reference(path)
    cal_path = Path(leg["calibration_receipt"]["path"])
    cal = json.loads(cal_path.read_text())
    cal["local_receipt"] = receipt_reference(path)
    _json(cal_path, cal)
    leg["calibration_receipt"] = receipt_reference(cal_path)
    assert "quiesced" in _validate(case, leg)[0]


def test_calibration_ratio_cannot_use_unrelated_t4_archive(case):
    leg = copy.deepcopy(case[2])
    path = Path(leg["calibration_receipt"]["path"])
    cal = json.loads(path.read_text())
    t4_path = Path(cal["t4_receipt"]["path"])
    t4 = json.loads(t4_path.read_text())
    t4["expected_archive_sha256"] = "d" * 64
    _json(t4_path, t4)
    cal["t4_receipt"] = receipt_reference(t4_path)
    _json(path, cal)
    leg["calibration_receipt"] = receipt_reference(path)
    assert "T4 source archive differs" in _validate(case, leg)[0]


def test_calibration_same_archive_wrong_runtime_refuses(case):
    leg = copy.deepcopy(case[2])
    path = Path(leg["calibration_receipt"]["path"])
    cal = json.loads(path.read_text())
    t4_path = Path(cal["t4_receipt"]["path"])
    t4 = json.loads(t4_path.read_text())
    t4["expected_runtime_tree_sha256"] = "d" * 64
    _json(t4_path, t4)
    cal["t4_receipt"] = receipt_reference(t4_path)
    _json(path, cal)
    leg["calibration_receipt"] = receipt_reference(path)
    assert "T4 source runtime differs" in _validate(case, leg)[0]


def test_inherited_source_cannot_be_forged_or_chained(case, tmp_path):
    runtime, archive, leg = case
    source_path = _json(tmp_path / "source.json", leg)
    inherited = inherit_decode_wall_clock(source_leg_path=source_path, runtime_dir=runtime,
        archive_path=archive, pointer_archive_sha256=leg["archive_sha256"])
    chained_path = _json(tmp_path / "inherited.json", inherited)
    with pytest.raises(SealContractError, match="directly to a measured"):
        inherit_decode_wall_clock(source_leg_path=chained_path, runtime_dir=runtime,
            archive_path=archive, pointer_archive_sha256=leg["archive_sha256"])
    source_path.write_text(source_path.read_text() + " ")
    assert "receipt drift" in validate_decode_wall_clock(inherited, runtime_dir=runtime,
        archive_path=archive, pointer_archive_sha256=leg["archive_sha256"])[0][0]


def test_build_seal_refuses_missing_timing(case):
    from tac.candidate_seal import AdmitBar, build_seal
    from tac.tests.test_candidate_seal import _public_smoke
    runtime, archive, leg = case
    bar = AdmitBar(rule="synthetic test", net_dS_threshold=-1, pointer_axis="contest_cuda", pointer_score_at_seal=1,
        pointer_archive_sha256_at_seal=leg["archive_sha256"])
    with pytest.raises(SealContractError, match="decode wall-clock"):
        build_seal(candidate_id="missing-clock", runtime_dir=runtime, admit_bar=bar,
            public_entrypoint_smoke=_public_smoke(runtime, archive))


def _timeout_receipt(case, path):
    runtime, archive, leg = case
    runtime_sha = measure_t4_runtime_digest(runtime)
    return _json(path, {"passed": False, "returncode": 1,
        "expected_archive_sha256": leg["archive_sha256"], "expected_runtime_tree_sha256": runtime_sha,
        "artifacts": {
            "modal_cuda_preflight.json": json.dumps({"torch_cuda_device_name": "Tesla T4"}),
            "contest_auth_eval.stderr.log": "ProcessGroupTimeout: Command '['bash', '/runtime/inflate.sh']' timed out after 1800.0 seconds\n[ inflate ]\n[inflate] TIMED OUT",
            "provenance.json": json.dumps({"inflate_timeout_seconds": 1800, "archive_sha256": leg["archive_sha256"],
                "inflate_runtime_manifest": {"runtime_tree_sha256": runtime_sha}})}})


def test_observed_timeout_overrides_optimistic_projection(case, tmp_path):
    leg = copy.deepcopy(case[2])
    leg["candidate_t4_receipt"] = receipt_reference(_timeout_receipt(case, tmp_path / "timeout.json"))
    problems, observed = validate_decode_wall_clock(leg, runtime_dir=case[0], archive_path=case[1])
    assert "timeout_lower_bound 1800.000000s exceeds 1260.0s" in problems[0]
    assert observed["measured_t4_kind"] == "timeout_lower_bound"


@pytest.mark.parametrize("field", ["expected_archive_sha256", "expected_runtime_tree_sha256"])
def test_timeout_cannot_attach_to_unrelated_candidate(case, tmp_path, field):
    leg = copy.deepcopy(case[2])
    path = _timeout_receipt(case, tmp_path / "timeout.json")
    doc = json.loads(path.read_text())
    doc[field] = "d" * 64
    _json(path, doc)
    leg["candidate_t4_receipt"] = receipt_reference(path)
    assert "differs" in _validate(case, leg)[0]


def test_noninflate_timeout_is_not_decode_evidence(case, tmp_path):
    leg = copy.deepcopy(case[2])
    path = _timeout_receipt(case, tmp_path / "timeout.json")
    doc = json.loads(path.read_text())
    doc["artifacts"]["contest_auth_eval.stderr.log"] = "evaluate.py timed out after 1800.0 seconds"
    _json(path, doc)
    leg["candidate_t4_receipt"] = receipt_reference(path)
    assert "not an observed inflate timeout" in _validate(case, leg)[0]


def test_candidate_host_cannot_borrow_another_hosts_ratio(case, tmp_path):
    runtime, archive, original = case
    candidate_local_path = tmp_path / "candidate_local.json"
    local = json.loads(Path(original["local_receipt"]["path"]).read_text())
    local["host"] = "different-host"
    _json(candidate_local_path, local)
    leg = copy.deepcopy(original)
    leg["local_receipt"] = receipt_reference(candidate_local_path)
    assert "local host/platform differs" in _validate(case, leg)[0]


def test_optional_platform_identity_cannot_drift(case, tmp_path):
    candidate_local_path = tmp_path / "candidate_local.json"
    local = json.loads(Path(case[2]["local_receipt"]["path"]).read_text())
    local["platform"] = "different-platform"
    _json(candidate_local_path, local)
    leg = copy.deepcopy(case[2])
    leg["local_receipt"] = receipt_reference(candidate_local_path)
    assert "local host/platform differs" in _validate(case, leg)[0]


def test_evaluation_seconds_cannot_masquerade_as_decode(case):
    leg = copy.deepcopy(case[2])
    path = Path(leg["calibration_receipt"]["path"])
    cal = json.loads(path.read_text())
    t4_path = Path(cal["t4_receipt"]["path"])
    t4 = json.loads(t4_path.read_text())
    auth = json.loads(t4["artifacts"]["contest_auth_eval.json"])
    auth["inflate_elapsed_seconds"] = 1336.668977565
    t4["artifacts"]["contest_auth_eval.json"] = json.dumps(auth)
    _json(t4_path, t4)
    cal["t4_receipt"] = receipt_reference(t4_path)
    cal["t4_seconds_field"] = ["artifacts", "contest_auth_eval.json", "evaluate_elapsed_seconds"]
    cal["cpu_to_t4_ratio"] = 45 / 1200
    _json(path, cal)
    leg["calibration_receipt"] = receipt_reference(path)
    leg["cpu_to_t4_ratio"] = 45 / 1200
    leg["projected_t4_decode_seconds"] = 45
    assert "t4_seconds_field must name the canonical" in _validate(case, leg)[0]


def test_unsupported_whole_eval_scope_refuses(case):
    leg = copy.deepcopy(case[2])
    path = Path(leg["calibration_receipt"]["path"])
    cal = json.loads(path.read_text())
    cal["t4_timing_scope"] = "full_eval_upper_bound"
    _json(path, cal)
    leg["calibration_receipt"] = receipt_reference(path)
    assert "only completed decode calibration" in _validate(case, leg)[0]


@pytest.mark.parametrize("field,value", [("passed", False), ("passed", 1), ("returncode", 1), ("returncode", False)])
def test_failed_or_malformed_t4_source_cannot_calibrate(case, field, value):
    leg = copy.deepcopy(case[2])
    cal_path = Path(leg["calibration_receipt"]["path"])
    cal = json.loads(cal_path.read_text())
    t4_path = Path(cal["t4_receipt"]["path"])
    t4 = json.loads(t4_path.read_text())
    t4[field] = value
    _json(t4_path, t4)
    cal["t4_receipt"] = receipt_reference(t4_path)
    _json(cal_path, cal)
    leg["calibration_receipt"] = receipt_reference(cal_path)
    assert "completed successful result" in _validate(case, leg)[0]


@pytest.mark.parametrize("field", ["t4_archive_sha256_field", "t4_runtime_sha256_field", "t4_hardware_field"])
def test_calibration_identity_fields_cannot_be_redirected(case, field):
    leg = copy.deepcopy(case[2])
    path = Path(leg["calibration_receipt"]["path"])
    cal = json.loads(path.read_text())
    cal[field] = ["unrelated_identity"]
    _json(path, cal)
    leg["calibration_receipt"] = receipt_reference(path)
    assert f"{field} must name the canonical" in _validate(case, leg)[0]
