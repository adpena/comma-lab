# SPDX-License-Identifier: MIT
"""Tests for the STC-Dasher byte-anchor archive builder."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from tools.build_stc_dasher_archive_v1 import REPO_ROOT, build_stc_dasher_candidate, main


def _write_zip(path: Path, payload: bytes) -> None:
    info = zipfile.ZipInfo("x", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def test_builder_manifest_does_not_claim_v1_score_delta(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    out_dir = tmp_path / "experiments" / "results" / "stc"
    _write_zip(source, b"payload" * 32)

    manifest = build_stc_dasher_candidate(
        source_archive=source,
        out_dir=out_dir,
        sigma=0.0,
        constraint_length=12,
        context_length=2,
        payload_bit_ratio=4,
    )
    manifest_json = json.loads((out_dir / "build_manifest.json").read_text())

    assert manifest.score_claim is False
    assert manifest.ready_for_exact_eval_dispatch is False
    assert manifest.predicted_delta_s_band is None
    assert manifest.predicted_delta_s_band_after_viterbi_inverse == (-0.010, -0.030)
    assert manifest_json["predicted_delta_s_band"] is None
    assert manifest_json["rate_negative_scaffold"] is True
    assert "full_viterbi_inverse_council_gated" in manifest_json["review_blockers"]


def test_builder_candidate_zip_has_single_deterministic_member(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    out_dir = tmp_path / "experiments" / "results" / "stc"
    _write_zip(source, b"abc" * 8)

    manifest = build_stc_dasher_candidate(
        source_archive=source,
        out_dir=out_dir,
        sigma=0.0,
        constraint_length=12,
        context_length=2,
        payload_bit_ratio=4,
    )

    with zipfile.ZipFile(manifest.candidate_archive_path) as zf:
        infos = zf.infolist()
        assert [i.filename for i in infos] == ["stc_dasher_v1.bin"]
        assert infos[0].date_time == (1980, 1, 1, 0, 0, 0)
        assert infos[0].compress_type == zipfile.ZIP_STORED


def test_builder_refuses_truncated_non_byte_closed_candidate(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    out_dir = tmp_path / "experiments" / "results" / "stc"
    _write_zip(source, b"x" * 256)

    try:
        build_stc_dasher_candidate(
            source_archive=source,
            out_dir=out_dir,
            sigma=0.0,
            constraint_length=12,
            context_length=2,
            payload_bit_ratio=4,
            max_source_bytes=64,
        )
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("expected large source to fail closed")

    assert "Refusing slow scaffold build" in message
    assert "non-byte-closed candidate" in message
    assert not (out_dir / "build_manifest.json").exists()


def test_builder_cli_rejects_non_default_payload_ratio_before_build(tmp_path: Path) -> None:
    out_dir = REPO_ROOT / "experiments" / "results" / "pytest_stc_invalid_ratio"

    rc = main(
        [
            "--source-archive",
            str(tmp_path / "missing.zip"),
            "--out-dir",
            str(out_dir),
            "--payload-bit-ratio",
            "3",
        ]
    )

    assert rc == 2
    assert not out_dir.exists()
