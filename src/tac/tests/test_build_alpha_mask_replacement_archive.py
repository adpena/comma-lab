# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import lzma
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "build_alpha_mask_replacement_archive.py"
SPEC = importlib.util.spec_from_file_location("build_alpha_mask_replacement_archive", MODULE_PATH)
assert SPEC is not None
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(builder)


def _sha(data: bytes) -> str:
    return builder._sha256_bytes(data)


def _write_anchor(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("renderer.bin", b"r" * 12000)
        zf.writestr("masks.mkv", b"m" * 421483)
        zf.writestr("optimized_poses.bin", b"p" * 7200)


def _write_candidate_manifest(path: Path, grayscale: Path, repair: Path | None = None) -> None:
    artifacts = [
        {
            "role": "alpha4_grayscale_lut_video",
            "candidate_archive_member": "grayscale.mkv",
            "path": str(grayscale),
            "size_bytes": grayscale.stat().st_size,
            "sha256": builder._sha256_file(grayscale),
        }
    ]
    if repair is not None:
        artifacts.append(
            {
                "role": "alpha4_residual_repair_payload",
                "candidate_archive_member": "alpha4_residual_repair.amr1",
                "path": str(repair),
                "size_bytes": repair.stat().st_size,
                "sha256": builder._sha256_file(repair),
            }
        )
    payload = {
        "schema": "alpha_mask_candidate_builder_v1",
        "candidate": {
            "candidate_archive_readiness": {
                "full_sequence_candidate": True,
                "ready_for_exact_eval_finalist_archive_assembly": True,
            },
            "alpha4": {
                "crf": 63,
                "agreement_before_repair": {"argmax_agreement": 0.9923},
            },
            "artifacts": artifacts,
        },
    }
    path.write_text(json.dumps(payload))


def _write_repair_payload(path: Path) -> None:
    alpha_builder = builder._load_alpha_builder_module()
    runs = [
        alpha_builder.RepairRun(frame_index=0, y=0, x0=0, length=2, class_id=2),
        alpha_builder.RepairRun(frame_index=0, y=0, x0=2, length=2, class_id=1),
    ]
    payload = alpha_builder._encode_repair_payload(
        runs,
        shape=(1, 1, 4),
        source_mask_sha256="a" * 64,
        candidate_mask_sha256="b" * 64,
        selection_meta={
            "total_residual_pixels": 4,
            "selected_repair_pixels": 4,
            "partial_repair": False,
        },
    )
    path.write_bytes(payload)


def _write_pair_repair_payload(path: Path) -> None:
    alpha_builder = builder._load_alpha_builder_module()
    runs = [
        alpha_builder.RepairRun(frame_index=0, y=0, x0=0, length=2, class_id=2),
        alpha_builder.RepairRun(frame_index=2, y=0, x0=0, length=1, class_id=1),
        alpha_builder.RepairRun(frame_index=3, y=0, x0=1, length=1, class_id=4),
    ]
    payload = alpha_builder._encode_repair_payload(
        runs,
        shape=(4, 1, 4),
        source_mask_sha256="a" * 64,
        candidate_mask_sha256="b" * 64,
        selection_meta={
            "total_residual_pixels": 4,
            "selected_repair_pixels": 4,
            "partial_repair": False,
        },
    )
    path.write_bytes(payload)


def test_builds_deterministic_grayscale_only_archive(tmp_path: Path) -> None:
    anchor = tmp_path / "anchor.zip"
    _write_anchor(anchor)
    grayscale = tmp_path / "grayscale.mkv"
    grayscale.write_bytes(b"g" * 197382)
    manifest = tmp_path / "alpha_manifest.json"
    _write_candidate_manifest(manifest, grayscale)

    out = tmp_path / "archive.zip"
    payload = builder.build_archive(
        anchor_archive=anchor,
        candidate_manifest_path=manifest,
        output=out,
        provenance_json=tmp_path / "manifest.json",
    )

    with zipfile.ZipFile(out) as zf:
        assert zf.namelist() == ["renderer.bin", "grayscale.mkv", "optimized_poses.bin"]
        assert zf.read("renderer.bin") == b"r" * 12000
        assert zf.read("grayscale.mkv") == b"g" * 197382
        assert zf.read("optimized_poses.bin") == b"p" * 7200
        for info in zf.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.external_attr >> 16 == 0o644

    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["runtime_contract"]["expected_inflate_mode"] == "renderer_grayscale"
    assert payload["anchor"]["replaced_member"]["sha256"] == _sha(b"m" * 421483)


def test_builds_selected_compressed_repair_archive(tmp_path: Path) -> None:
    anchor = tmp_path / "anchor.zip"
    _write_anchor(anchor)
    grayscale = tmp_path / "grayscale.mkv"
    grayscale.write_bytes(b"g" * 197382)
    repair = tmp_path / "alpha4_residual_repair.amr1"
    _write_repair_payload(repair)
    manifest = tmp_path / "alpha_manifest.json"
    _write_candidate_manifest(manifest, grayscale, repair=repair)

    out = tmp_path / "archive.zip"
    payload = builder.build_archive(
        anchor_archive=anchor,
        candidate_manifest_path=manifest,
        output=out,
        provenance_json=tmp_path / "manifest.json",
        repair_policy="class_prefix_2",
        repair_compressor="lzma_xz",
    )

    with zipfile.ZipFile(out) as zf:
        assert zf.namelist() == [
            "renderer.bin",
            "grayscale.mkv",
            "alpha4_residual_repair.amr1.xz",
            "optimized_poses.bin",
        ]
        raw = lzma.decompress(zf.read("alpha4_residual_repair.amr1.xz"), format=lzma.FORMAT_XZ)

    alpha_builder = builder._load_alpha_builder_module()
    header, runs = alpha_builder._decode_repair_payload(raw)
    assert len(runs) == 1
    assert runs[0].class_id == 2
    assert header["selection"]["partial_repair"] is True
    assert payload["candidate"]["residual_repair"]["policy"]["policy_name"] == "class_prefix_2"
    assert payload["candidate"]["residual_repair"]["compressor"] == "lzma_xz"


def test_builds_pair_index_selected_repair_archive(tmp_path: Path) -> None:
    anchor = tmp_path / "anchor.zip"
    _write_anchor(anchor)
    grayscale = tmp_path / "grayscale.mkv"
    grayscale.write_bytes(b"g" * 197382)
    repair = tmp_path / "alpha4_residual_repair.amr1"
    _write_pair_repair_payload(repair)
    manifest = tmp_path / "alpha_manifest.json"
    _write_candidate_manifest(manifest, grayscale, repair=repair)

    out = tmp_path / "archive.zip"
    payload = builder.build_archive(
        anchor_archive=anchor,
        candidate_manifest_path=manifest,
        output=out,
        provenance_json=tmp_path / "manifest.json",
        repair_policy="pair_indices_1",
        repair_compressor="raw",
    )

    with zipfile.ZipFile(out) as zf:
        raw = zf.read("alpha4_residual_repair.amr1")

    alpha_builder = builder._load_alpha_builder_module()
    header, runs = alpha_builder._decode_repair_payload(raw)
    assert [run.frame_index for run in runs] == [2, 3]
    assert header["selection"]["policy_kind"] == "pair_indices"
    assert header["selection"]["policy_details"]["selected_pair_indices"] == [1]
    assert header["selection"]["policy_details"]["selected_frame_indices"] == [2, 3]
    assert payload["candidate"]["residual_repair"]["policy"]["selected_repair_pixels"] == 2


def test_rejects_hidden_anchor_sidecar(tmp_path: Path) -> None:
    anchor = tmp_path / "anchor.zip"
    with zipfile.ZipFile(anchor, "w") as zf:
        zf.writestr("renderer.bin", b"r" * 12000)
        zf.writestr("masks.mkv", b"m" * 421483)
        zf.writestr("optimized_poses.bin", b"p" * 7200)
        zf.writestr("._masks.mkv", b"sidecar")
    grayscale = tmp_path / "grayscale.mkv"
    grayscale.write_bytes(b"g" * 197382)
    manifest = tmp_path / "alpha_manifest.json"
    _write_candidate_manifest(manifest, grayscale)

    with pytest.raises(ValueError, match="hidden/system archive member"):
        builder.build_archive(
            anchor_archive=anchor,
            candidate_manifest_path=manifest,
            output=tmp_path / "archive.zip",
            provenance_json=None,
        )
