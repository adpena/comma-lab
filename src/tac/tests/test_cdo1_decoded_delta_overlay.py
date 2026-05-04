from __future__ import annotations

import importlib.util
import lzma
import sys
import zipfile
import zlib
from pathlib import Path

import pytest
import torch

from experiments.plan_c067_decoded_delta_overlay_mask_topology import (
    OverlayRun,
    _encode_overlay_payload,
    _mask_tensor_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
INFLATE_RENDERER_PATH = REPO_ROOT / "submissions" / "robust_current" / "inflate_renderer.py"
BUILDER_PATH = REPO_ROOT / "experiments" / "build_c067_decoded_delta_overlay_candidate.py"
PACKER_PATH = REPO_ROOT / "experiments" / "build_renderer_packed_payload_archive.py"
UNPACKER_PATH = REPO_ROOT / "submissions" / "robust_current" / "unpack_renderer_payload.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _payload(base: torch.Tensor, overlaid: torch.Tensor, runs: list[OverlayRun]) -> bytes:
    return _encode_overlay_payload(
        runs=runs,
        header={
            "schema": "c067_decoded_delta_overlay_payload_v1",
            "producer": "unit-test",
            "score_claim": False,
            "base_mask_tensor_sha256": _mask_tensor_sha256(base.numpy().astype("uint8")),
            "reconstructed_mask_u8_sha256": _mask_tensor_sha256(
                overlaid.numpy().astype("uint8")
            ),
            "shape": [int(v) for v in base.shape],
            "pair_index_basis": "video_frame_pair_index",
            "run_struct": "u16_frame_u16_y_u16_x0_u16_length_u8_value_le",
            "run_count": len(runs),
            "selected_pixel_count": sum(int(run.length) for run in runs),
            "selected_pair_indices": [0],
        },
    )


def _runtime_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("renderer.bin", b"R" * 10_001)
        zf.writestr("masks.mkv", b"M" * 1_200)
        zf.writestr("optimized_poses.bin", b"P" * 2_000)


def _read_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path, "r") as zf:
        return {info.filename: zf.read(info) for info in zf.infolist()}


def test_inflate_renderer_applies_cdo1_overlay_and_preserves_half_frame(tmp_path: Path) -> None:
    inflate_renderer = _load_module("inflate_renderer_cdo1_test", INFLATE_RENDERER_PATH)
    base = torch.tensor(
        [
            [[0, 0, 1, 1], [2, 2, 2, 0]],
            [[1, 1, 1, 1], [0, 0, 0, 0]],
        ],
        dtype=torch.int64,
    )
    overlaid = base.clone()
    overlaid[0, 0, 1:3] = torch.tensor([3, 3])
    overlaid[1, 1, 0:2] = torch.tensor([4, 4])
    runs = [
        OverlayRun(frame_index=0, y=0, x0=1, length=2, value=3),
        OverlayRun(frame_index=1, y=1, x0=0, length=2, value=4),
    ]
    base._half_frame_only = True  # type: ignore[attr-defined]
    (tmp_path / "masks.cdo1.xz").write_bytes(lzma.compress(_payload(base, overlaid, runs)))

    result = inflate_renderer._maybe_apply_cdo1_overlay_from_archive_dir(tmp_path, base)

    assert torch.equal(result, overlaid)
    assert getattr(result, "_half_frame_only", False) is True


def test_inflate_renderer_rejects_cdo1_base_sha_mismatch(tmp_path: Path) -> None:
    inflate_renderer = _load_module("inflate_renderer_cdo1_mismatch_test", INFLATE_RENDERER_PATH)
    base = torch.zeros((1, 2, 4), dtype=torch.int64)
    overlaid = base.clone()
    overlaid[0, 0, 0] = 1
    raw = _payload(base, overlaid, [OverlayRun(frame_index=0, y=0, x0=0, length=1, value=1)])
    (tmp_path / "masks.cdo1.zlib").write_bytes(zlib.compress(raw, level=9))

    with pytest.raises(RuntimeError, match="base SHA mismatch"):
        inflate_renderer._maybe_apply_cdo1_overlay_from_archive_dir(
            tmp_path,
            torch.ones_like(base),
        )


def test_cdo1_builder_emits_deterministic_byte_closed_archive(tmp_path: Path) -> None:
    builder = _load_module("build_cdo1_candidate_test", BUILDER_PATH)
    base_archive = tmp_path / "base.zip"
    _runtime_archive(base_archive)
    base = torch.zeros((1, 2, 4), dtype=torch.int64)
    overlaid = base.clone()
    overlaid[0, 1, 1:3] = torch.tensor([2, 2])
    payload = tmp_path / "overlay.cdo1"
    payload.write_bytes(
        _payload(
            base,
            overlaid,
            [OverlayRun(frame_index=0, y=1, x0=1, length=2, value=2)],
        )
    )

    out_a = tmp_path / "a" / "archive.zip"
    out_b = tmp_path / "b" / "archive.zip"
    report = builder.build_candidate(
        base_archive=base_archive,
        overlay_payload=payload,
        output_archive=out_a,
        manifest_json=tmp_path / "a" / "manifest.json",
        overlay_compressor="lzma_xz",
        repo_root=REPO_ROOT,
    )
    builder.build_candidate(
        base_archive=base_archive,
        overlay_payload=payload,
        output_archive=out_b,
        manifest_json=tmp_path / "b" / "manifest.json",
        overlay_compressor="lzma_xz",
        repo_root=REPO_ROOT,
    )

    members = _read_members(out_a)
    assert list(members) == [
        "renderer.bin",
        "masks.mkv",
        "masks.cdo1.xz",
        "optimized_poses.bin",
    ]
    assert lzma.decompress(members["masks.cdo1.xz"]) == payload.read_bytes()
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["cdo1_overlay"]["archive_member"] == "masks.cdo1.xz"
    assert report["cdo1_overlay"]["pair_index_basis"] == "video_frame_pair_index"
    assert report["cdo1_overlay"]["selected_pair_indices"] == [0]
    assert report["cdo1_overlay"]["payload_header"]["pair_index_basis"] == (
        "video_frame_pair_index"
    )
    assert out_a.read_bytes() == out_b.read_bytes()


def test_cdo1_builder_accepts_and_reemits_single_p_payload_archive(tmp_path: Path) -> None:
    builder = _load_module("build_cdo1_candidate_packed_test", BUILDER_PATH)
    packer = _load_module("build_cdo1_candidate_packer_test", PACKER_PATH)
    unpacker = _load_module("build_cdo1_candidate_unpacker_test", UNPACKER_PATH)

    expanded_base = tmp_path / "expanded_base.zip"
    _runtime_archive(expanded_base)
    packed_base = tmp_path / "packed_base.zip"
    packer.build_packed_archive(
        expanded_base,
        packed_base,
        payload_member_name="p",
        payload_format=packer.PAYLOAD_FORMAT_RPK1_JSON,
    )

    base = torch.zeros((1, 2, 4), dtype=torch.int64)
    overlaid = base.clone()
    overlaid[0, 1, 1:3] = torch.tensor([2, 2])
    payload = tmp_path / "overlay.cdo1"
    payload.write_bytes(
        _payload(
            base,
            overlaid,
            [OverlayRun(frame_index=0, y=1, x0=1, length=2, value=2)],
        )
    )

    output_archive = tmp_path / "packed_out" / "archive.zip"
    report = builder.build_candidate(
        base_archive=packed_base,
        overlay_payload=payload,
        output_archive=output_archive,
        manifest_json=tmp_path / "packed_out" / "manifest.json",
        overlay_compressor="lzma_xz",
        pack_output_payload=True,
        packed_payload_member_name="p",
        packed_payload_format=packer.PAYLOAD_FORMAT_RPK1_JSON,
        repo_root=REPO_ROOT,
    )

    with zipfile.ZipFile(output_archive, "r") as zf:
        assert zf.namelist() == ["p"]
        zf.extractall(tmp_path / "unpacked")
    summary = unpacker.unpack_renderer_payload(tmp_path / "unpacked")
    assert {row["name"] for row in summary["members"]} == {
        "renderer.bin",
        "masks.mkv",
        "masks.cdo1.xz",
        "optimized_poses.bin",
    }
    assert (
        lzma.decompress((tmp_path / "unpacked" / "masks.cdo1.xz").read_bytes())
        == payload.read_bytes()
    )
    assert report["base_archive"]["runtime_member_contract"]["base_archive_layout"] == "packed_payload"
    assert report["output_archive"]["packaging"]["packed_output"] is True


def test_cdo1_builder_rejects_payload_without_pair_index_basis(tmp_path: Path) -> None:
    builder = _load_module("build_cdo1_candidate_pair_basis_guard_test", BUILDER_PATH)
    base_archive = tmp_path / "base.zip"
    _runtime_archive(base_archive)
    base = torch.zeros((1, 2, 4), dtype=torch.int64)
    overlaid = base.clone()
    overlaid[0, 0, 0] = 1
    payload = tmp_path / "overlay_missing_basis.cdo1"
    payload.write_bytes(
        _encode_overlay_payload(
            runs=[OverlayRun(frame_index=0, y=0, x0=0, length=1, value=1)],
            header={
                "schema": "c067_decoded_delta_overlay_payload_v1",
                "producer": "unit-test",
                "score_claim": False,
                "base_mask_tensor_sha256": _mask_tensor_sha256(
                    base.numpy().astype("uint8")
                ),
                "reconstructed_mask_u8_sha256": _mask_tensor_sha256(
                    overlaid.numpy().astype("uint8")
                ),
                "shape": [1, 2, 4],
                "run_struct": "u16_frame_u16_y_u16_x0_u16_length_u8_value_le",
                "run_count": 1,
                "selected_pixel_count": 1,
                "selected_pair_indices": [0],
            },
        )
    )

    with pytest.raises(ValueError, match="pair_index_basis"):
        builder.build_candidate(
            base_archive=base_archive,
            overlay_payload=payload,
            output_archive=tmp_path / "out" / "archive.zip",
            manifest_json=tmp_path / "out" / "manifest.json",
            overlay_compressor="lzma_xz",
            repo_root=REPO_ROOT,
        )
