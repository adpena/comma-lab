"""Tests for the Alpha mask candidate builder."""
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "alpha_mask_candidate_builder.py"
SPEC = importlib.util.spec_from_file_location("alpha_mask_candidate_builder", MODULE_PATH)
assert SPEC is not None
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def _safe_archive(tmp_path: Path, mask_bytes: bytes = b"mask-bytes") -> Path:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("renderer.bin", b"renderer")
        zf.writestr("masks.mkv", mask_bytes)
    return archive


def _tiny_masks() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [0, 0, 1, 1, 1, 4],
                [0, 2, 2, 1, 4, 4],
                [3, 3, 2, 4, 4, 4],
            ],
            [
                [0, 1, 1, 1, 1, 4],
                [0, 2, 2, 2, 4, 4],
                [3, 3, 2, 4, 4, 0],
            ],
        ],
        dtype=torch.int64,
    )


def _source_meta(tmp_path: Path) -> dict[str, Any]:
    archive = _safe_archive(tmp_path)
    _member_data, source_meta = builder._read_archive_member(archive, "masks.mkv")
    return source_meta


def _install_fake_alpha4_codec(
    monkeypatch: pytest.MonkeyPatch,
    candidate_masks: torch.Tensor,
) -> None:
    def fake_encode_gray_av1(gray: torch.Tensor, output_path: Path, *, crf: int, fps: int) -> dict[str, Any]:
        payload = b"fake-alpha4-mkv\n" + gray.cpu().contiguous().numpy().tobytes()
        output_path.write_bytes(payload)
        return {
            "path": output_path,
            "bytes": payload,
            "command": ["fake-ffmpeg", "-crf", str(crf), "-r", str(fps), str(output_path)],
            "frames": int(gray.shape[0]),
            "height": int(gray.shape[1]),
            "width": int(gray.shape[2]),
        }

    def fake_decode_gray_av1(path: Path, *, expected_shape: tuple[int, int, int]) -> torch.Tensor:
        assert path.exists()
        assert tuple(candidate_masks.shape) == expected_shape
        return candidate_masks.clone()

    monkeypatch.setattr(builder, "_encode_gray_av1", fake_encode_gray_av1)
    monkeypatch.setattr(builder, "_decode_gray_av1", fake_decode_gray_av1)


def test_read_archive_member_rejects_zip_slip_member(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("masks.mkv", b"mask-bytes")
        zf.writestr("../escape", b"bad")

    with pytest.raises(ValueError, match="unsafe archive member path"):
        builder._read_archive_member(archive, "masks.mkv")


def test_read_archive_member_rejects_hidden_sidecar(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("masks.mkv", b"mask-bytes")
        zf.writestr("._masks.mkv", b"sidecar")

    with pytest.raises(ValueError, match="hidden/system archive member"):
        builder._read_archive_member(archive, "masks.mkv")


def test_read_archive_member_rejects_duplicate_member(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("masks.mkv", b"mask-bytes-a")
            zf.writestr("masks.mkv", b"mask-bytes-b")

    with pytest.raises(ValueError, match="duplicate archive member"):
        builder._read_archive_member(archive, "masks.mkv")


def test_builder_writes_candidate_payload_repair_payload_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tiny_masks()
    candidate = source.clone()
    candidate[0, 1, 1:3] = 0
    candidate[1, 2, 5] = 4
    _install_fake_alpha4_codec(monkeypatch, candidate)

    report = builder._build_candidate_artifacts_from_masks(
        masks=source,
        source_meta=_source_meta(tmp_path),
        output_dir=tmp_path / "candidate",
        config=builder.BuilderConfig(max_frames=None),
        command=["alpha_mask_candidate_builder.py", "--unit-test"],
    )

    output_dir = tmp_path / "candidate"
    manifest_path = output_dir / builder.MANIFEST_NAME
    grayscale_path = output_dir / builder.GRAYSCALE_MEMBER
    repair_path = output_dir / builder.REPAIR_MEMBER
    assert manifest_path.exists()
    assert grayscale_path.exists()
    assert repair_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest == report
    assert manifest["schema"] == "alpha_mask_candidate_builder_v1"
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["evidence_grade"] == "empirical"
    assert manifest["scorer_network_loaded"] is False
    assert "contest_auth_eval.py --device cuda" in manifest["canonical_score_source_required"]

    artifacts = {item["role"]: item for item in manifest["candidate"]["artifacts"]}
    assert artifacts["alpha4_grayscale_lut_video"]["candidate_archive_member"] == "grayscale.mkv"
    assert artifacts["alpha4_grayscale_lut_video"]["size_bytes"] == grayscale_path.stat().st_size
    assert artifacts["alpha4_grayscale_lut_video"]["sha256"] == builder._sha256_file(grayscale_path)
    assert artifacts["alpha4_residual_repair_payload"]["candidate_archive_member"] == "alpha4_residual_repair.amr1"
    assert artifacts["alpha4_residual_repair_payload"]["sha256"] == builder._sha256_file(repair_path)

    before = manifest["candidate"]["alpha4"]["agreement_before_repair"]
    after = manifest["candidate"]["repair"]["agreement_after_repair"]
    assert before["different_pixels"] == 3
    assert after["different_pixels"] == 0
    assert after["argmax_agreement"] == 1.0
    assert manifest["candidate"]["repair"]["selection"]["residual_pixel_coverage"] == 1.0
    assert manifest["candidate"]["candidate_archive_readiness"]["ready_for_exact_eval_finalist_archive_assembly"] is True

    repaired = builder._apply_repair_payload(candidate, repair_path.read_bytes())
    assert torch.equal(repaired, source)


def test_builder_crf_sweep_reuses_decode_and_writes_child_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tiny_masks()
    candidate = source.clone()
    candidate[0, 1, 1:3] = 0
    _install_fake_alpha4_codec(monkeypatch, candidate)
    decode_calls = 0

    def fake_decode_member(data: bytes, member: str, *, max_frames: int | None) -> tuple[torch.Tensor, dict[str, Any]]:
        nonlocal decode_calls
        decode_calls += 1
        assert data == b"mask-bytes"
        assert member == "masks.mkv"
        return source.clone(), {
            "decoder": "fake",
            "decoded_frames": int(source.shape[0]),
            "max_frames": max_frames,
            "truncated_by_builder": False,
        }

    monkeypatch.setattr(builder, "_decode_legacy_av1_masks_from_member", fake_decode_member)
    archive = _safe_archive(tmp_path)
    output_dir = tmp_path / "sweep"

    report = builder.build_candidate_crf_sweep_from_archive(
        archive=archive,
        mask_member="masks.mkv",
        output_dir=output_dir,
        config=builder.BuilderConfig(max_frames=None),
        crfs=(52, 60),
        command=["alpha_mask_candidate_builder.py", "--unit-test-sweep"],
    )

    assert decode_calls == 1
    sweep_manifest = output_dir / builder.SWEEP_MANIFEST_NAME
    assert sweep_manifest.exists()
    assert json.loads(sweep_manifest.read_text()) == report
    assert report["schema"] == "alpha_mask_candidate_crf_sweep_v1"
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["sweep_values"] == [52, 60]
    assert report["sweep_summary"]["candidate_count"] == 2
    assert len(report["candidate_records"]) == 2
    for crf in (52, 60):
        child_manifest = output_dir / f"crf_{crf:02d}" / builder.MANIFEST_NAME
        assert child_manifest.exists()
        child = json.loads(child_manifest.read_text())
        assert child["candidate"]["alpha4"]["crf"] == crf
        assert child["score_claim"] is False
        assert child["promotion_eligible"] is False


def test_builder_fails_closed_when_repair_cap_would_make_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tiny_masks()
    candidate = source.clone()
    candidate[0, 1, 1:3] = 0
    _install_fake_alpha4_codec(monkeypatch, candidate)

    with pytest.raises(ValueError, match="repair payload would be partial"):
        builder._build_candidate_artifacts_from_masks(
            masks=source,
            source_meta=_source_meta(tmp_path),
            output_dir=tmp_path / "candidate",
            config=builder.BuilderConfig(max_frames=None, max_repair_pixels=1),
            command=["alpha_mask_candidate_builder.py", "--unit-test"],
        )


def test_builder_can_emit_partial_repair_only_when_explicitly_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tiny_masks()
    candidate = source.clone()
    candidate[0, 1, 1:3] = 0
    _install_fake_alpha4_codec(monkeypatch, candidate)

    report = builder._build_candidate_artifacts_from_masks(
        masks=source,
        source_meta=_source_meta(tmp_path),
        output_dir=tmp_path / "candidate",
        config=builder.BuilderConfig(
            max_frames=None,
            max_repair_pixels=1,
            fail_on_partial_repair=False,
        ),
        command=["alpha_mask_candidate_builder.py", "--unit-test"],
    )

    selection = report["candidate"]["repair"]["selection"]
    assert selection["partial_repair"] is True
    assert selection["residual_pixel_coverage"] == 0.0
    assert report["candidate"]["candidate_archive_readiness"]["residual_repair_full_coverage"] is False
    assert report["candidate"]["candidate_archive_readiness"]["ready_for_exact_eval_finalist_archive_assembly"] is False


def test_output_dir_requires_force_before_overwriting_known_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tiny_masks()
    _install_fake_alpha4_codec(monkeypatch, source.clone())
    output_dir = tmp_path / "candidate"

    builder._build_candidate_artifacts_from_masks(
        masks=source,
        source_meta=_source_meta(tmp_path),
        output_dir=output_dir,
        config=builder.BuilderConfig(max_frames=None),
        command=["alpha_mask_candidate_builder.py", "--unit-test"],
    )
    with pytest.raises(FileExistsError, match="use --force"):
        builder._build_candidate_artifacts_from_masks(
            masks=source,
            source_meta=_source_meta(tmp_path),
            output_dir=output_dir,
            config=builder.BuilderConfig(max_frames=None),
            command=["alpha_mask_candidate_builder.py", "--unit-test"],
        )


def test_ffmpeg_resolver_fails_closed_for_bad_explicit_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_ffmpeg = tmp_path / "bad-ffmpeg"
    bad_ffmpeg.write_text("#!/bin/sh\nexit 1\n")
    bad_ffmpeg.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    good_ffmpeg = bin_dir / "ffmpeg"
    good_ffmpeg.write_text("#!/bin/sh\nprintf 'ffmpeg version fake\\n'\nexit 0\n")
    good_ffmpeg.chmod(0o755)

    monkeypatch.setenv("TAC_FFMPEG", str(bad_ffmpeg))
    monkeypatch.setenv("PATH", str(bin_dir))

    with pytest.raises(RuntimeError, match="not a usable ffmpeg"):
        builder._resolve_ffmpeg_binary()


def test_cli_defaults_are_bounded_and_fail_closed() -> None:
    parser = builder._build_arg_parser()
    args = parser.parse_args([])

    assert args.max_frames == builder.DEFAULT_MAX_FRAMES
    assert args.all_frames is False
    assert args.alpha4_crf_sweep is None
    assert args.allow_partial_repair is False
    assert args.force is False


def test_parse_alpha4_crf_sweep_rejects_empty_duplicate_and_out_of_range() -> None:
    assert builder._parse_alpha4_crf_sweep("52,58,63") == (52, 58, 63)

    with pytest.raises(ValueError, match="at least one"):
        builder._parse_alpha4_crf_sweep(" , ")
    with pytest.raises(ValueError, match="duplicate"):
        builder._parse_alpha4_crf_sweep("52,52")
    with pytest.raises(ValueError, match=r"\[0,63\]"):
        builder._parse_alpha4_crf_sweep("64")
