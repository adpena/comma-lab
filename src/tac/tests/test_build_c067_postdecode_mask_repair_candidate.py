# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import lzma
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "build_c067_postdecode_mask_repair_candidate.py"
SPEC = importlib.util.spec_from_file_location("build_c067_postdecode_mask_repair_candidate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def _write_runtime_archive(path: Path, *, mask_bytes: bytes = b"source-mask") -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("renderer.bin", b"renderer")
        zf.writestr("masks.mkv", mask_bytes)
        zf.writestr("optimized_poses.bin", b"poses")


def _read_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path, "r") as zf:
        return {info.filename: zf.read(info) for info in zf.infolist()}


def test_builds_deterministic_charged_legacy_mask_repair_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = torch.tensor(
        [
            [[1, 1, 0, 0], [2, 2, 2, 0]],
            [[3, 3, 4, 4], [0, 0, 1, 1]],
        ],
        dtype=torch.int64,
    )
    candidate = torch.zeros_like(source)
    runtime = tmp_path / "runtime.zip"
    _write_runtime_archive(runtime, mask_bytes=b"source-mask")
    lossy = tmp_path / "lossy.mkv"
    lossy.write_bytes(b"lossy-mask")

    def fake_decode(data: bytes, *, member: str, max_frames: int | None) -> tuple[torch.Tensor, dict[str, Any]]:
        assert max_frames is None
        if data == b"source-mask":
            assert member == "masks.mkv"
            return source.clone(), {"decoder": "fake-source"}
        if data == b"lossy-mask":
            assert member == "masks.mkv"
            return candidate.clone(), {"decoder": "fake-lossy"}
        raise AssertionError(data)

    monkeypatch.setattr(builder, "_decode_legacy_mask_stream", fake_decode)

    out_a = tmp_path / "a" / "archive.zip"
    manifest_a = tmp_path / "a" / "manifest.json"
    report_a = builder.build_candidate(
        base_runtime_dir=None,
        base_archive=runtime,
        lossy_mask=lossy,
        output_archive=out_a,
        manifest_json=manifest_a,
        policy=builder.RepairAtomPolicy(policy="full", max_atoms=None, label="unit_full_repair"),
        repair_compressor="lzma_xz",
    )

    members = _read_members(out_a)
    assert list(members) == [
        "renderer.bin",
        "masks.mkv",
        "alpha4_residual_repair.amr1.xz",
        "optimized_poses.bin",
    ]
    assert members["renderer.bin"] == b"renderer"
    assert members["masks.mkv"] == b"lossy-mask"
    assert members["optimized_poses.bin"] == b"poses"
    for info in zipfile.ZipFile(out_a, "r").infolist():
        assert info.date_time == (1980, 1, 1, 0, 0, 0)
        assert info.external_attr >> 16 == 0o644

    raw = lzma.decompress(members["alpha4_residual_repair.amr1.xz"], format=lzma.FORMAT_XZ)
    alpha = builder._load_alpha_builder()
    header, runs = alpha._decode_repair_payload(raw)
    repaired = alpha._apply_repair_payload(candidate, raw)
    assert int((source != repaired).sum().item()) == 0
    assert header["source_mask_u8_sha256"] == alpha._tensor_u8_sha256(source)
    assert header["candidate_mask_u8_sha256"] == alpha._tensor_u8_sha256(candidate)
    assert header["selection"]["score_claim"] is False
    assert header["selection"]["selected_repair_pixels"] == int((source != candidate).sum().item())
    assert len(runs) == report_a["repair_payload"]["selection"]["selected_repair_runs"]

    manifest_payload = json.loads(manifest_a.read_text())
    charged = manifest_payload["charged_member_accounting"]["alpha4_residual_repair.amr1.xz"]
    assert charged["role"] == "charged_postdecode_amr1_mask_repair"
    assert charged["bytes"] == len(members["alpha4_residual_repair.amr1.xz"])
    assert charged["sha256"] == builder._sha256_bytes(members["alpha4_residual_repair.amr1.xz"])
    assert manifest_payload["score_claim"] is False
    assert manifest_payload["promotion_eligible"] is False
    assert manifest_payload["runtime_contract"]["legacy_masks_member"] == "masks.mkv"

    out_b = tmp_path / "b" / "archive.zip"
    builder.build_candidate(
        base_runtime_dir=None,
        base_archive=runtime,
        lossy_mask=lossy,
        output_archive=out_b,
        manifest_json=tmp_path / "b" / "manifest.json",
        policy=builder.RepairAtomPolicy(policy="full", max_atoms=None, label="unit_full_repair"),
        repair_compressor="lzma_xz",
    )
    assert out_a.read_bytes() == out_b.read_bytes()


def test_top_pixel_policy_records_selected_atoms_and_partial_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = torch.tensor([[[1, 1, 1, 0], [2, 2, 0, 0]]], dtype=torch.int64)
    candidate = torch.zeros_like(source)
    runtime = tmp_path / "runtime.zip"
    _write_runtime_archive(runtime)
    lossy = tmp_path / "lossy.mkv"
    lossy.write_bytes(b"lossy")
    monkeypatch.setattr(
        builder,
        "_decode_legacy_mask_stream",
        lambda data, *, member, max_frames: (
            source.clone() if data == b"source-mask" else candidate.clone(),
            {"decoder": "fake"},
        ),
    )

    report = builder.build_candidate(
        base_runtime_dir=None,
        base_archive=runtime,
        lossy_mask=lossy,
        output_archive=tmp_path / "archive.zip",
        policy=builder.RepairAtomPolicy(policy="top_pixels", max_atoms=1, atom_granularity="frame_class"),
        repair_compressor="raw",
    )

    selector = report["repair_selector"]
    assert selector["total_residual_pixels"] == 5
    assert selector["selected_atom_count"] == 1
    assert selector["selected_atoms"][0]["atom_id"] == "frame0000_class1"
    assert selector["selected_repair_pixels"] == 3
    assert selector["partial_repair"] is True
    assert report["repair_payload"]["archive_member"] == "alpha4_residual_repair.amr1"


def test_compressed_byte_budget_selects_largest_prefix_that_fits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = torch.tensor(
        [
            [[1, 1, 1, 0], [2, 2, 0, 0]],
            [[3, 3, 3, 3], [4, 4, 0, 0]],
        ],
        dtype=torch.int64,
    )
    candidate = torch.zeros_like(source)
    runtime = tmp_path / "runtime.zip"
    _write_runtime_archive(runtime)
    lossy = tmp_path / "lossy.mkv"
    lossy.write_bytes(b"lossy")
    monkeypatch.setattr(
        builder,
        "_decode_legacy_mask_stream",
        lambda data, *, member, max_frames: (
            source.clone() if data == b"source-mask" else candidate.clone(),
            {"decoder": "fake"},
        ),
    )

    one = builder.build_candidate(
        base_runtime_dir=None,
        base_archive=runtime,
        lossy_mask=lossy,
        output_archive=tmp_path / "one" / "archive.zip",
        policy=builder.RepairAtomPolicy(policy="top_pixels", max_atoms=1, atom_granularity="frame_class"),
        repair_compressor="raw",
    )
    two = builder.build_candidate(
        base_runtime_dir=None,
        base_archive=runtime,
        lossy_mask=lossy,
        output_archive=tmp_path / "two" / "archive.zip",
        policy=builder.RepairAtomPolicy(policy="top_pixels", max_atoms=2, atom_granularity="frame_class"),
        repair_compressor="raw",
    )
    one_bytes = one["repair_payload"]["compressed_size_bytes"]
    two_bytes = two["repair_payload"]["compressed_size_bytes"]
    assert one_bytes < two_bytes

    budgeted = builder.build_candidate(
        base_runtime_dir=None,
        base_archive=runtime,
        lossy_mask=lossy,
        output_archive=tmp_path / "budgeted" / "archive.zip",
        policy=builder.RepairAtomPolicy(
            policy="top_pixels",
            max_atoms=3,
            atom_granularity="frame_class",
            max_repair_payload_bytes=one_bytes,
        ),
        repair_compressor="raw",
    )

    budget = budgeted["repair_selector"]["compressed_byte_budget"]
    assert budget["budget_applied"] is True
    assert budget["max_repair_payload_bytes"] == one_bytes
    assert budget["compressed_repair_bytes_after_budget"] <= one_bytes
    assert budgeted["repair_selector"]["selected_atom_count"] == 1
    assert budgeted["repair_payload"]["compressed_size_bytes"] <= one_bytes
    assert budgeted["repair_payload"]["compressed_size_bytes"] < two_bytes


def test_rejects_zip_slip_runtime_archive_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("renderer.bin", b"renderer")
        zf.writestr("../masks.mkv", b"mask")
        zf.writestr("optimized_poses.bin", b"poses")
    lossy = tmp_path / "lossy.mkv"
    lossy.write_bytes(b"lossy")
    monkeypatch.setattr(
        builder,
        "_decode_legacy_mask_stream",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("decode should not run")),
    )

    with pytest.raises(ValueError, match="zip-slip archive member path"):
        builder.build_candidate(
            base_runtime_dir=None,
            base_archive=bad,
            lossy_mask=lossy,
            output_archive=tmp_path / "archive.zip",
        )


def test_rejects_duplicate_lossy_mask_member(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime.zip"
    _write_runtime_archive(runtime)
    lossy_archive = tmp_path / "lossy.zip"
    with zipfile.ZipFile(lossy_archive, "w") as zf:
        zf.writestr("masks.mkv", b"a")
        zf.writestr("masks.mkv", b"b")

    with pytest.raises(ValueError, match="expected exactly one"):
        builder.build_candidate(
            base_runtime_dir=None,
            base_archive=runtime,
            lossy_archive=lossy_archive,
            output_archive=tmp_path / "archive.zip",
        )
