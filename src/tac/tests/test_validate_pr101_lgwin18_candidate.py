# SPDX-License-Identifier: MIT
"""Focused tests for ``tools/validate_pr101_lgwin18_candidate.py``."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import types
import zipfile
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import pytest


def _load_validator_module():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    path = repo_root / "tools" / "validate_pr101_lgwin18_candidate.py"
    spec = importlib.util.spec_from_file_location(
        "validate_pr101_lgwin18_candidate", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_surgery_module():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    path = repo_root / "tools" / "pr101_archive_substitution_surgery.py"
    spec = importlib.util.spec_from_file_location(
        "pr101_archive_substitution_surgery", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_pr101_archive(
    path: pathlib.Path,
    *,
    decoder_blob: bytes,
    date_time: tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0),
) -> pathlib.Path:
    surgery = _load_surgery_module()
    latent_blob = b"\xb2" * surgery.PR101_LATENT_BLOB_LEN
    sidecar_blob = b"\xc3" * 607
    info = zipfile.ZipInfo(filename=surgery.PR101_INNER_MEMBER_NAME)
    info.compress_type = zipfile.ZIP_STORED
    info.date_time = date_time
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, decoder_blob + latent_blob + sidecar_blob)
    return path


def _install_fake_pr101_codec(monkeypatch: pytest.MonkeyPatch) -> None:
    surgery = _load_surgery_module()
    source_blob = b"\x11" * surgery.PR101_DECODER_BLOB_LEN
    lgwin_blob = b"\x22" * surgery.PR101_DECODER_BLOB_LEN

    def decode_decoder_compact(blob: bytes):
        if blob not in {source_blob, lgwin_blob}:
            raise ValueError("unexpected decoder blob")
        return {
            "stem.weight": torch.tensor([1, 2, 3], dtype=torch.int16),
            "stem.bias": torch.tensor([4], dtype=torch.int16),
        }

    def encode_decoder_compact(state_dict, *, brotli_quality=11, brotli_lgwin=None):
        assert brotli_quality == 11
        assert set(state_dict) == {"stem.weight", "stem.bias"}
        if brotli_lgwin == 18:
            return lgwin_blob
        return source_blob

    fake = types.ModuleType("tac.pr101_split_brotli_codec")
    fake.decode_decoder_compact = decode_decoder_compact
    fake.encode_decoder_compact = encode_decoder_compact
    monkeypatch.setitem(sys.modules, "tac.pr101_split_brotli_codec", fake)


def _forbidden_score_keys(report: object) -> set[str]:
    forbidden = {
        "predicted_score",
        "predicted_band",
        "score_delta_vs_anchor",
        "score_recomputed_from_components",
        "avg_segnet_dist",
        "avg_posenet_dist",
        "seg_dist",
        "pose_dist",
    }
    seen: set[str] = set()
    if isinstance(report, dict):
        for key, value in report.items():
            if key in forbidden:
                seen.add(key)
            seen.update(_forbidden_score_keys(value))
    elif isinstance(report, list):
        for value in report:
            seen.update(_forbidden_score_keys(value))
    return seen


def test_valid_candidate_report_is_deterministic_and_non_scoring(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _load_validator_module()
    surgery = _load_surgery_module()
    _install_fake_pr101_codec(monkeypatch)
    source_blob = b"\x11" * surgery.PR101_DECODER_BLOB_LEN
    lgwin_blob = b"\x22" * surgery.PR101_DECODER_BLOB_LEN
    source = _write_pr101_archive(tmp_path / "source.zip", decoder_blob=source_blob)
    candidate = _write_pr101_archive(
        tmp_path / "candidate.zip",
        decoder_blob=lgwin_blob,
    )

    first = validator.validate_candidate(
        source_archive=source,
        candidate_archive=candidate,
        target_profile="contest_one_video_replay",
        brotli_quality=11,
        brotli_lgwin=18,
    )
    second = validator.validate_candidate(
        source_archive=source,
        candidate_archive=candidate,
        target_profile="contest_one_video_replay",
        brotli_quality=11,
        brotli_lgwin=18,
    )

    assert first == second
    assert first["verdict"] == validator.PASS_VERDICT
    assert first["score_claim"] is False
    assert _forbidden_score_keys(first) == set()
    assert first["target_profile"] == "contest_one_video_replay"
    assert "not claimed" in first["target_profile_context"]["production_generalized"]
    assert first["checks"]["candidate_archive_byte_different"] is True
    assert first["checks"]["candidate_decoder_blob_byte_different"] is True
    assert first["checks"]["latent_blob_preserved"] is True
    assert first["checks"]["sidecar_blob_preserved"] is True
    assert first["decoder_parity_checks"]["decoder_state_dict_parity"]["passed"] is True
    assert first["decoder_parity_checks"]["source_lgwin_reencode"][
        "matches_candidate_decoder_blob"
    ] is True
    assert first["dispatch"]["gpu_dispatch_performed"] is False

    report_path = tmp_path / "report.json"
    validator.write_report(first, report_path)
    assert json.loads(report_path.read_text()) == first


def test_exact_noop_candidate_fails_closed(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _load_validator_module()
    surgery = _load_surgery_module()
    _install_fake_pr101_codec(monkeypatch)
    source_blob = b"\x11" * surgery.PR101_DECODER_BLOB_LEN
    source = _write_pr101_archive(tmp_path / "source.zip", decoder_blob=source_blob)
    candidate = _write_pr101_archive(
        tmp_path / "candidate.zip",
        decoder_blob=source_blob,
    )

    report = validator.validate_candidate(
        source_archive=source,
        candidate_archive=candidate,
        target_profile="contest_one_video_replay",
    )

    assert report["verdict"] == validator.FAIL_VERDICT
    assert report["checks"]["candidate_archive_byte_different"] is False
    assert report["checks"]["candidate_decoder_blob_byte_different"] is False
    assert "candidate_archive_not_byte_different_against_source" in report[
        "validation_blockers"
    ]
    assert "candidate_decoder_blob_noop_against_source" in report[
        "validation_blockers"
    ]


def test_metadata_only_archive_difference_still_fails_closed(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _load_validator_module()
    surgery = _load_surgery_module()
    _install_fake_pr101_codec(monkeypatch)
    source_blob = b"\x11" * surgery.PR101_DECODER_BLOB_LEN
    source = _write_pr101_archive(tmp_path / "source.zip", decoder_blob=source_blob)
    candidate = _write_pr101_archive(
        tmp_path / "candidate.zip",
        decoder_blob=source_blob,
        date_time=(1981, 1, 1, 0, 0, 0),
    )

    report = validator.validate_candidate(
        source_archive=source,
        candidate_archive=candidate,
        target_profile="contest_one_video_replay",
    )

    assert report["verdict"] == validator.FAIL_VERDICT
    assert report["checks"]["candidate_archive_byte_different"] is True
    assert report["checks"]["candidate_inner_member_byte_different"] is False
    assert report["checks"]["candidate_decoder_blob_byte_different"] is False
    assert "candidate_inner_member_noop_against_source" in report[
        "validation_blockers"
    ]
    assert "candidate_decoder_blob_noop_against_source" in report[
        "validation_blockers"
    ]


def test_substitution_report_mismatch_fails_closed(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _load_validator_module()
    surgery = _load_surgery_module()
    _install_fake_pr101_codec(monkeypatch)
    source_blob = b"\x11" * surgery.PR101_DECODER_BLOB_LEN
    lgwin_blob = b"\x22" * surgery.PR101_DECODER_BLOB_LEN
    source = _write_pr101_archive(tmp_path / "source.zip", decoder_blob=source_blob)
    candidate = _write_pr101_archive(
        tmp_path / "candidate.zip",
        decoder_blob=lgwin_blob,
    )
    report_path = tmp_path / "bad_substitution_report.json"
    report_path.write_text(json.dumps({"sha256_output_archive": "wrong"}))

    report = validator.validate_candidate(
        source_archive=source,
        candidate_archive=candidate,
        substitution_report=report_path,
        target_profile="contest_one_video_replay",
    )

    assert report["verdict"] == validator.FAIL_VERDICT
    assert "substitution_report_sha256_output_archive_mismatch" in report[
        "validation_blockers"
    ]


def test_production_generalized_target_profile_is_not_claimed(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _load_validator_module()
    surgery = _load_surgery_module()
    _install_fake_pr101_codec(monkeypatch)
    source_blob = b"\x11" * surgery.PR101_DECODER_BLOB_LEN
    lgwin_blob = b"\x22" * surgery.PR101_DECODER_BLOB_LEN
    source = _write_pr101_archive(tmp_path / "source.zip", decoder_blob=source_blob)
    candidate = _write_pr101_archive(
        tmp_path / "candidate.zip",
        decoder_blob=lgwin_blob,
    )

    report = validator.validate_candidate(
        source_archive=source,
        candidate_archive=candidate,
        target_profile="production_generalized",
    )

    assert report["verdict"] == validator.FAIL_VERDICT
    assert "production_generalized_not_validated_by_pr101_one_video_tool" in report[
        "validation_blockers"
    ]


def test_contest_generalized_target_profile_is_known_but_fails_closed(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _load_validator_module()
    surgery = _load_surgery_module()
    _install_fake_pr101_codec(monkeypatch)
    source_blob = b"\x11" * surgery.PR101_DECODER_BLOB_LEN
    lgwin_blob = b"\x22" * surgery.PR101_DECODER_BLOB_LEN
    source = _write_pr101_archive(tmp_path / "source.zip", decoder_blob=source_blob)
    candidate = _write_pr101_archive(
        tmp_path / "candidate.zip",
        decoder_blob=lgwin_blob,
    )

    report = validator.validate_candidate(
        source_archive=source,
        candidate_archive=candidate,
        target_profile="contest_generalized",
    )

    assert report["verdict"] == validator.FAIL_VERDICT
    assert report["target_profile_context"]["policy"]["contest_dispatch_candidate"] is True
    assert "contest_generalized_not_validated_by_pr101_one_video_tool" in report[
        "validation_blockers"
    ]
