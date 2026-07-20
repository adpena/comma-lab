from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import measure_einstein_kolmogorov_crux_v3 as measurement


def _write_json(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_validate_n24_receipt_requires_all_24_pairs(tmp_path):
    archive_sha = "a" * 64
    receipt_path = tmp_path / "n24.json"
    receipt = {
        "nested_levelset_report": {
            "bit_exact_roundtrip_gate": {
                "checked": True,
                "bit_exact": True,
                "gate_pairs": 2,
                "frames_compared": 4,
                "max_abs_uint8_diff": 0,
            },
            "byte_close": {"archive_zip_sha256": archive_sha},
        }
    }
    config = {
        "strict_n24_receipt_path": str(receipt_path),
        "strict_n24_receipt_sha256": _write_json(receipt_path, receipt),
    }
    with pytest.raises(ValueError, match="all 24 pairs / 48 frames"):
        measurement._validate_n24_receipt(config, archive_sha)

    gate = receipt["nested_levelset_report"]["bit_exact_roundtrip_gate"]
    gate.update({"gate_pairs": 24, "frames_compared": 48})
    config["strict_n24_receipt_sha256"] = _write_json(receipt_path, receipt)
    custody = measurement._validate_n24_receipt(config, archive_sha)
    assert custody["bit_exact_gate"]["gate_pairs"] == 24


def test_inflate_stage_resume_binds_receiver_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(measurement.byte_close, "CAMERA_H", 1)
    monkeypatch.setattr(measurement.byte_close, "CAMERA_W", 1)
    raw_path = tmp_path / "0.raw"
    raw_path.write_bytes(b"r" * 3600)
    stage_path = tmp_path / "inflate.json"
    stage = {
        "schema": measurement.INFLATE_STAGE_SCHEMA,
        "archive_sha256": "archive",
        "inflate_py_sha256": "receiver",
        "raw_bytes": 3600,
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }
    _write_json(stage_path, stage)
    _, valid = measurement._inflate_stage_valid(
        stage_path, raw_path, "archive", "receiver"
    )
    assert valid is True
    _, valid = measurement._inflate_stage_valid(
        stage_path, raw_path, "archive", "different-receiver"
    )
    assert valid is False


def test_score_stage_resume_binds_gt_and_oracle_content():
    packet = {"archive": {"sha256": "archive"}}
    inflate = {"raw_sha256": "raw"}
    oracle = {"gt_cache": {"sha256": "gt"}, "hard_oracle_module": {"sha256": "oracle"}}
    stage = {
        "schema": measurement.SCORE_STAGE_SCHEMA,
        "archive_sha256": "archive",
        "raw_sha256": "raw",
        "parity": {"pairs_scored": 600},
        "scorer": {"content_bindings": oracle},
    }
    assert measurement._score_stage_valid(stage, packet, inflate, oracle) is True
    changed = {**oracle, "hard_oracle_module": {"sha256": "changed"}}
    assert measurement._score_stage_valid(stage, packet, inflate, changed) is False


def test_existing_result_revalidates_and_recovers_cleanup(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    inflate_stage_path = tmp_path / "inflate.json"
    score_stage_path = tmp_path / "score.json"
    result_path = tmp_path / "result.json"
    raw_path = tmp_path / "0.raw"
    raw_path.write_bytes(b"certified raw")

    packet = {
        "archive": {"sha256": "archive"},
        "inflate_py": {"sha256": "receiver"},
    }
    inflate_stage = {
        "schema": measurement.INFLATE_STAGE_SCHEMA,
        "archive_sha256": "archive",
        "inflate_py_sha256": "receiver",
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }
    oracle = {
        "gt_cache": {"sha256": "gt"},
        "byte_close_tool": measurement._require_file(Path(measurement.byte_close.__file__)),
        "hard_oracle_module": measurement._require_file(Path(measurement.byte_close.twr.__file__)),
        "contest_score": measurement._require_file(Path(measurement.contest_score.__file__)),
    }
    score_stage = {
        "schema": measurement.SCORE_STAGE_SCHEMA,
        "archive_sha256": "archive",
        "raw_sha256": inflate_stage["raw_sha256"],
        "parity": {"pairs_scored": 600},
        "scorer": {"content_bindings": oracle},
    }
    _write_json(inflate_stage_path, inflate_stage)
    _write_json(score_stage_path, score_stage)
    source_files = {
        "measurement_tool": measurement._require_file(Path(measurement.__file__)),
        "byte_close_tool": oracle["byte_close_tool"],
        "contest_score": oracle["contest_score"],
        "hard_oracle_module": oracle["hard_oracle_module"],
    }
    receipt = {
        "schema": measurement.RECEIPT_SCHEMA,
        "classification": "A",
        "packet": packet,
        "config": measurement._require_file(config_path),
        "strict_n24_bit_identity": {"sha256": "n24"},
        "full_n600_decode": inflate_stage,
        "full_n600_hard_cpu_torch_oracle": score_stage,
        "source_files": source_files,
        "cleanup": {
            "target": str(raw_path),
            "bytes": raw_path.stat().st_size,
            "sha256": inflate_stage["raw_sha256"],
            "performed": False,
        },
        "verdict": {"pairs_scored_measured": 600},
    }
    _write_json(result_path, receipt)
    config = {
        "inflate_stage_receipt_path": str(inflate_stage_path),
        "score_stage_receipt_path": str(score_stage_path),
        "cleanup_raw_after_success": True,
    }
    monkeypatch.setattr(measurement, "_oracle_custody", lambda _config: oracle)
    resumed = measurement._resume_existing_result(
        config_path, config, packet, {"sha256": "n24"}, result_path
    )
    assert resumed["cleanup"]["performed"] is True
    assert raw_path.exists() is False
