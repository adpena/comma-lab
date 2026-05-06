"""Tests for the Alpha mask codec candidate matrix."""
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
MODULE_PATH = REPO_ROOT / "experiments" / "alpha_mask_codec_candidate_matrix.py"
SPEC = importlib.util.spec_from_file_location("alpha_mask_codec_candidate_matrix", MODULE_PATH)
assert SPEC is not None
matrix = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = matrix
SPEC.loader.exec_module(matrix)


def _tiny_masks() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [0, 0, 1, 1, 1, 4, 4],
                [0, 2, 2, 1, 4, 4, 4],
                [3, 3, 2, 0, 0, 4, 4],
                [3, 0, 0, 0, 2, 2, 4],
            ],
            [
                [0, 1, 1, 1, 1, 4, 4],
                [0, 2, 2, 2, 4, 4, 0],
                [3, 3, 2, 0, 0, 4, 4],
                [3, 0, 0, 2, 2, 2, 4],
            ],
            [
                [0, 0, 1, 1, 4, 4, 4],
                [0, 2, 2, 1, 1, 4, 4],
                [3, 3, 2, 2, 0, 0, 4],
                [3, 0, 0, 2, 2, 4, 4],
            ],
        ],
        dtype=torch.int64,
    )


def _safe_archive(tmp_path: Path, mask_bytes: bytes = b"mask-bytes") -> Path:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("renderer.bin", b"renderer")
        zf.writestr("masks.mkv", mask_bytes)
    return archive


def _source_meta(tmp_path: Path) -> dict[str, Any]:
    archive = _safe_archive(tmp_path)
    _data, meta = matrix._read_archive_member(archive, "masks.mkv")
    return meta


def _exact_config() -> Any:
    return matrix.MatrixConfig(
        max_frames=None,
        families=("coco_rle", "component_boundary_delta", "transition_endpoints"),
        compression="zlib",
        zlib_level=9,
    )


def test_matrix_writes_multiple_exact_candidate_families_and_non_promotable_manifest(
    tmp_path: Path,
) -> None:
    report = matrix._build_candidate_matrix_from_masks(
        masks=_tiny_masks(),
        source_meta=_source_meta(tmp_path),
        output_dir=tmp_path / "matrix",
        config=_exact_config(),
        command=["alpha_mask_codec_candidate_matrix.py", "--unit-test"],
    )

    manifest_path = tmp_path / "matrix" / matrix.MANIFEST_NAME
    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text()) == report
    assert report["schema"] == "alpha_mask_codec_candidate_matrix_v1"
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["evidence_grade"] == "empirical"
    assert report["scorer_network_loaded"] is False
    assert "contest_auth_eval.py --device cuda" in report["canonical_score_source_required"]

    candidates = {candidate["family"]: candidate for candidate in report["candidates"]}
    assert set(candidates) == {"coco_rle", "component_boundary_delta", "transition_endpoints"}
    for family, candidate in candidates.items():
        assert candidate["score_claim"] is False
        assert candidate["promotion_eligible"] is False
        assert candidate["charged_representation"] is True
        assert candidate["diagnostic_reference"] is False
        assert candidate["exact_reconstruction"] is True
        assert candidate["agreement"]["different_pixels"] == 0
        artifact = candidate["artifact"]
        artifact_path = Path(artifact["path"])
        assert artifact_path.exists(), family
        assert artifact["size_bytes"] == artifact_path.stat().st_size
        assert artifact["sha256"] == matrix._sha256_file(artifact_path)

    ranked = report["rankings"]["exact_reconstruction_by_bytes"]
    assert [item["size_bytes"] for item in ranked] == sorted(item["size_bytes"] for item in ranked)
    assert len(ranked) == 3


def test_matrix_artifacts_are_deterministic_with_force_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "matrix"
    kwargs = {
        "masks": _tiny_masks(),
        "source_meta": _source_meta(tmp_path),
        "output_dir": output_dir,
        "config": _exact_config(),
        "command": ["alpha_mask_codec_candidate_matrix.py", "--determinism-test"],
    }
    first = matrix._build_candidate_matrix_from_masks(**kwargs)
    second = matrix._build_candidate_matrix_from_masks(**kwargs, force=True)

    assert second == first
    for member in (
        matrix.COCO_RLE_MEMBER,
        matrix.COMPONENT_BOUNDARY_MEMBER,
        matrix.TRANSITION_ENDPOINT_MEMBER,
    ):
        path = output_dir / member
        assert path.exists()
        assert matrix._sha256_file(path) in json.dumps(first["candidates"], sort_keys=True)


def test_build_from_archive_rejects_zip_slip_member_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("masks.mkv", b"mask-bytes")
        zf.writestr("../escape", b"bad")

    def fail_decode(*_args: Any, **_kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        raise AssertionError("decode should not run after unsafe zip member")

    monkeypatch.setattr(matrix, "_decode_masks_member_with_local_helper", fail_decode)
    with pytest.raises(ValueError, match="unsafe archive member path"):
        matrix.build_candidate_matrix_from_archive(
            archive=archive,
            mask_member="masks.mkv",
            output_dir=tmp_path / "out",
            config=matrix.MatrixConfig(max_frames=None, families=("coco_rle",)),
        )


def test_build_from_archive_rejects_hidden_and_duplicate_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_decode(*_args: Any, **_kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        raise AssertionError("decode should not run after invalid zip custody")

    monkeypatch.setattr(matrix, "_decode_masks_member_with_local_helper", fake_decode)

    hidden = tmp_path / "hidden.zip"
    with zipfile.ZipFile(hidden, "w") as zf:
        zf.writestr("masks.mkv", b"mask-bytes")
        zf.writestr("._masks.mkv", b"sidecar")
    with pytest.raises(ValueError, match="hidden/system archive member"):
        matrix.build_candidate_matrix_from_archive(
            archive=hidden,
            mask_member="masks.mkv",
            output_dir=tmp_path / "hidden_out",
            config=matrix.MatrixConfig(max_frames=None, families=("coco_rle",)),
        )

    duplicate = tmp_path / "duplicate.zip"
    with pytest.warns(UserWarning, match="Duplicate name"), zipfile.ZipFile(duplicate, "w") as zf:
        zf.writestr("masks.mkv", b"mask-a")
        zf.writestr("masks.mkv", b"mask-b")
    with pytest.raises(ValueError, match="duplicate archive member"):
        matrix.build_candidate_matrix_from_archive(
            archive=duplicate,
            mask_member="masks.mkv",
            output_dir=tmp_path / "dup_out",
            config=matrix.MatrixConfig(max_frames=None, families=("coco_rle",)),
        )


def test_build_from_archive_uses_decoded_masks_and_keeps_no_score_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _safe_archive(tmp_path)

    def fake_decode(
        data: bytes,
        member: str,
        *,
        config: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        assert data == b"mask-bytes"
        assert member == "masks.mkv"
        assert config.max_frames is None
        return _tiny_masks(), {"decoder": "fake-local-helper", "decoded_frames": 3}

    monkeypatch.setattr(matrix, "_decode_masks_member_with_local_helper", fake_decode)
    report = matrix.build_candidate_matrix_from_archive(
        archive=archive,
        mask_member="masks.mkv",
        output_dir=tmp_path / "archive_matrix",
        config=matrix.MatrixConfig(max_frames=None, families=("coco_rle", "transition_endpoints")),
        command=["alpha_mask_codec_candidate_matrix.py", "--archive-test"],
    )

    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["source"]["decode"]["decoder"] == "fake-local-helper"
    assert report["source"]["mask_member"]["sha256"] == matrix._sha256_bytes(b"mask-bytes")
    assert {candidate["family"] for candidate in report["candidates"]} == {
        "coco_rle",
        "transition_endpoints",
    }
