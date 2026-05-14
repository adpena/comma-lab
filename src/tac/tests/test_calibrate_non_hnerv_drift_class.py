# SPDX-License-Identifier: MIT
"""Tests for tools/calibrate_non_hnerv_drift_class.py — non-HNeRV drift calibration.

Covers:
  - validate_paired_eval custody: required fields present, substrates allowed
  - validate_paired_eval refuses macOS substrate
  - validate_paired_eval refuses unknown architecture_class
  - validate_paired_eval refuses NaN scores
  - validate_paired_eval refuses zero/negative cpu pose (division-by-zero guard)
  - derive_anchor computes observed_r_pose / observed_r_seg / score_gap
  - calibrate_class persists registry + audit log
  - calibrate_class cold-start: instantiates default profile if class missing
  - calibrate_class hot-start: updates existing profile via Welford
  - audit log appends multiple records
  - CLI returns 0 on success
  - CLI returns 2 on refusal
  - CLI accepts --paired-eval-json
  - CLI accepts explicit flags
  - NON_HNERV_CLASSES excludes HNeRV cluster
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import calibrate_non_hnerv_drift_class as calib  # noqa: E402

from tac.optimization.cuda_cpu_axis_profile_registry import (  # noqa: E402
    ArchitectureProfile,
    write_registry,
)

SHA = "f" * 64


def _good_payload(arch: str = "balle_scale_hyperprior") -> dict:
    return {
        "archive_sha256": SHA,
        "architecture_class": arch,
        "cuda": {
            "pose": 0.011,
            "seg": 0.0024,
            "score": 0.229,
            "hardware_substrate": "linux_x86_64_t4",
        },
        "cpu": {
            "pose": 0.054,
            "seg": 0.0028,
            "score": 0.196,
            "hardware_substrate": "linux_x86_64_gha_cpu",
        },
        "source": "test",
    }


# ── validate_paired_eval ───────────────────────────────────────────────────


def test_good_payload_validates():
    cuda, cpu = calib.validate_paired_eval(_good_payload())
    assert cuda["pose"] == 0.011
    assert cpu["pose"] == 0.054


def test_non_dict_payload_refused():
    with pytest.raises(calib.CalibrationRefusalError):
        calib.validate_paired_eval("not a dict")  # type: ignore


def test_missing_archive_sha256_refused():
    p = _good_payload()
    p["archive_sha256"] = ""
    with pytest.raises(calib.CalibrationRefusalError, match="archive_sha256"):
        calib.validate_paired_eval(p)


def test_unknown_architecture_class_refused():
    p = _good_payload()
    p["architecture_class"] = "made_up_class"
    with pytest.raises(calib.CalibrationRefusalError, match="architecture_class"):
        calib.validate_paired_eval(p)


def test_missing_cuda_block_refused():
    p = _good_payload()
    del p["cuda"]
    with pytest.raises(calib.CalibrationRefusalError, match="cuda and cpu blocks"):
        calib.validate_paired_eval(p)


def test_macos_cpu_substrate_refused():
    p = _good_payload()
    p["cpu"]["hardware_substrate"] = "macos_arm64"
    with pytest.raises(calib.CalibrationRefusalError, match="contest-compliant"):
        calib.validate_paired_eval(p)


def test_unknown_cuda_substrate_refused():
    p = _good_payload()
    p["cuda"]["hardware_substrate"] = "linux_x86_64_unknown_gpu"
    with pytest.raises(calib.CalibrationRefusalError, match="contest-compliant"):
        calib.validate_paired_eval(p)


def test_nan_score_refused():
    p = _good_payload()
    p["cuda"]["pose"] = float("nan")
    with pytest.raises(calib.CalibrationRefusalError, match="NaN"):
        calib.validate_paired_eval(p)


def test_non_numeric_field_refused():
    p = _good_payload()
    p["cuda"]["seg"] = "not numeric"
    with pytest.raises(calib.CalibrationRefusalError, match="numeric"):
        calib.validate_paired_eval(p)


def test_zero_cpu_pose_refused():
    p = _good_payload()
    p["cpu"]["pose"] = 0.0
    with pytest.raises(calib.CalibrationRefusalError, match="cpu pose"):
        calib.validate_paired_eval(p)


def test_negative_cpu_seg_refused():
    p = _good_payload()
    p["cpu"]["seg"] = -0.001
    with pytest.raises(calib.CalibrationRefusalError, match="cpu seg"):
        calib.validate_paired_eval(p)


# ── derive_anchor ──────────────────────────────────────────────────────────


def test_derive_anchor_observed_ratios():
    a = calib.derive_anchor(_good_payload())
    assert a["observed_r_pose"] == pytest.approx(0.011 / 0.054)
    assert a["observed_r_seg"] == pytest.approx(0.0024 / 0.0028)
    assert a["score_gap"] == pytest.approx(0.229 - 0.196)
    assert a["calibration_schema"] == calib.CALIBRATION_SCHEMA


def test_derive_anchor_carries_provenance():
    a = calib.derive_anchor(_good_payload())
    assert a["source"] == "test"
    assert "ingested_utc" in a


# ── calibrate_class persistence ────────────────────────────────────────────


def test_calibrate_class_writes_registry(tmp_path):
    reg_path = tmp_path / "registry.json"
    audit_path = tmp_path / "audit.jsonl"
    record = calib.calibrate_class(
        _good_payload(),
        registry_path=reg_path,
        audit_log_path=audit_path,
    )
    assert reg_path.is_file()
    assert audit_path.is_file()
    assert record["update"]["accepted"] in (True, False)


def test_calibrate_class_appends_audit_log(tmp_path):
    reg_path = tmp_path / "reg.json"
    audit_path = tmp_path / "audit.jsonl"
    calib.calibrate_class(_good_payload(), registry_path=reg_path,
                          audit_log_path=audit_path)
    p2 = _good_payload()
    p2["archive_sha256"] = "e" * 64
    calib.calibrate_class(p2, registry_path=reg_path, audit_log_path=audit_path)
    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec1 = json.loads(lines[0])
    rec2 = json.loads(lines[1])
    assert rec1["anchor"]["archive_sha256"] != rec2["anchor"]["archive_sha256"]


def test_calibrate_class_cold_start_creates_profile(tmp_path):
    reg_path = tmp_path / "reg.json"
    audit_path = tmp_path / "audit.jsonl"
    # Begin with empty registry
    write_registry({}, reg_path)
    payload = _good_payload(arch="mnerv")
    calib.calibrate_class(payload, registry_path=reg_path,
                          audit_log_path=audit_path)
    raw = json.loads(reg_path.read_text(encoding="utf-8"))
    assert "mnerv" in raw["profiles"]


def test_calibrate_class_hot_start_updates_existing(tmp_path):
    reg_path = tmp_path / "reg.json"
    audit_path = tmp_path / "audit.jsonl"
    profile = ArchitectureProfile(architecture_class="balle_scale_hyperprior")
    profile.r_pose_mean = 4.5
    profile.r_pose_std = 0.2
    write_registry({"balle_scale_hyperprior": profile}, reg_path)

    record = calib.calibrate_class(
        _good_payload(arch="balle_scale_hyperprior"),
        registry_path=reg_path,
        audit_log_path=audit_path,
    )
    # before/after delta should be present
    before = record["update"]["before"]
    after = record["update"]["after"]
    assert before["n_anchors"] != after["n_anchors"] or record["update"]["outlier_candidate"]


# ── _payload_from_args + CLI ───────────────────────────────────────────────


def test_payload_from_args_explicit_flags(tmp_path):
    args = type("X", (), {})()
    args.paired_eval_json = None
    args.archive_sha256 = SHA
    args.architecture_class = "mnerv"
    args.cuda_pose = 0.011
    args.cuda_seg = 0.0024
    args.cuda_score = 0.229
    args.cuda_substrate = "linux_x86_64_t4"
    args.cpu_pose = 0.054
    args.cpu_seg = 0.0028
    args.cpu_score = 0.196
    args.cpu_substrate = "linux_x86_64_gha_cpu"
    args.source = "explicit"
    payload = calib._payload_from_args(args)
    assert payload["architecture_class"] == "mnerv"
    assert payload["source"] == "explicit"


def test_payload_from_args_missing_flag_raises(tmp_path):
    args = type("X", (), {})()
    args.paired_eval_json = None
    args.archive_sha256 = None
    args.architecture_class = None
    args.cuda_pose = args.cuda_seg = args.cuda_score = args.cuda_substrate = None
    args.cpu_pose = args.cpu_seg = args.cpu_score = args.cpu_substrate = None
    args.source = None
    with pytest.raises(calib.CalibrationRefusalError, match="missing required"):
        calib._payload_from_args(args)


def test_payload_from_args_paired_json(tmp_path):
    payload_path = tmp_path / "paired.json"
    payload_path.write_text(json.dumps(_good_payload()), encoding="utf-8")
    args = type("X", (), {})()
    args.paired_eval_json = str(payload_path)
    payload = calib._payload_from_args(args)
    assert payload["architecture_class"] == "balle_scale_hyperprior"


def test_payload_from_args_paired_json_missing_file(tmp_path):
    args = type("X", (), {})()
    args.paired_eval_json = str(tmp_path / "nope.json")
    with pytest.raises(calib.CalibrationRefusalError, match="not found"):
        calib._payload_from_args(args)


def test_main_cli_explicit_flags_returns_0(tmp_path):
    reg_path = tmp_path / "reg.json"
    audit_path = tmp_path / "audit.jsonl"
    rc = calib.main([
        "--archive-sha256", SHA,
        "--architecture-class", "mnerv",
        "--cuda-pose", "0.011",
        "--cuda-seg", "0.0024",
        "--cuda-score", "0.229",
        "--cuda-substrate", "linux_x86_64_t4",
        "--cpu-pose", "0.054",
        "--cpu-seg", "0.0028",
        "--cpu-score", "0.196",
        "--cpu-substrate", "linux_x86_64_gha_cpu",
        "--registry-path", str(reg_path),
        "--audit-log-path", str(audit_path),
    ])
    assert rc == 0
    assert reg_path.is_file()


def test_main_cli_paired_json_returns_0(tmp_path):
    payload_path = tmp_path / "paired.json"
    payload_path.write_text(json.dumps(_good_payload(arch="raw_av1_yuv")),
                            encoding="utf-8")
    reg_path = tmp_path / "reg.json"
    audit_path = tmp_path / "audit.jsonl"
    rc = calib.main([
        "--paired-eval-json", str(payload_path),
        "--registry-path", str(reg_path),
        "--audit-log-path", str(audit_path),
    ])
    assert rc == 0


def test_main_cli_refusal_returns_2(tmp_path, capsys):
    reg_path = tmp_path / "reg.json"
    audit_path = tmp_path / "audit.jsonl"
    rc = calib.main([
        "--archive-sha256", SHA,
        "--architecture-class", "mnerv",
        "--cuda-pose", "0.011",
        "--cuda-seg", "0.0024",
        "--cuda-score", "0.229",
        "--cuda-substrate", "macos_arm64",  # refused substrate
        "--cpu-pose", "0.054",
        "--cpu-seg", "0.0028",
        "--cpu-score", "0.196",
        "--cpu-substrate", "linux_x86_64_gha_cpu",
        "--registry-path", str(reg_path),
        "--audit-log-path", str(audit_path),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err


# ── Surface invariants ─────────────────────────────────────────────────────


def test_non_hnerv_classes_excludes_hnerv():
    assert "hnerv_ft_microcodec" not in calib.NON_HNERV_CLASSES
    assert "hnerv_lc_v2" not in calib.NON_HNERV_CLASSES
    # 9 classes (10 total - 2 HNeRV - 1 'unknown_uncalibrated' is included
    # actually since we calibrate against it). The exact count may vary as
    # ARCHITECTURE_CLASSES grows; assert size > 5 and HNeRV excluded.
    assert len(calib.NON_HNERV_CLASSES) >= 5


def test_allowed_substrates_match_continual_learning():
    from tac.continual_learning import TAG_HARDWARE_REQUIREMENT
    cuda_allowed = TAG_HARDWARE_REQUIREMENT["[contest-CUDA]"]
    cpu_allowed = TAG_HARDWARE_REQUIREMENT["[contest-CPU GHA Linux x86_64]"]
    assert cuda_allowed == calib.ALLOWED_CUDA_SUBSTRATES
    assert cpu_allowed == calib.ALLOWED_CPU_SUBSTRATES


def test_calibration_schema_constant_stable():
    assert calib.CALIBRATION_SCHEMA == "tac_non_hnerv_drift_calibration_v1"
