from __future__ import annotations

import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO / "experiments" / "build_sjkl_c067_archive.py"
PACKER_PATH = REPO / "experiments" / "build_renderer_packed_payload_archive.py"
UNPACKER_PATH = REPO / "submissions" / "robust_current" / "unpack_renderer_payload.py"
INFLATE_RENDERER_PATH = REPO / "submissions" / "robust_current" / "inflate_renderer.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_runtime_source(path: Path) -> Path:
    pose_values = []
    for row in range(600):
        pose_values.extend([30.0 + row / 512.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pose_bytes = struct.pack("<" + "e" * len(pose_values), *pose_values)
    members = {
        "renderer.bin": b"renderer" * 512,
        "masks.mkv": b"mask" * 1024,
        "optimized_poses.bin": pose_bytes,
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in members.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            info.extra = b""
            info.comment = b""
            zf.writestr(info, data)
    return path


def _write_packed_public_floor_source(tmp_path: Path) -> Path:
    packer = _load_module(PACKER_PATH, "_sjkl_c067_test_packer")
    direct_source = _write_runtime_source(tmp_path / "runtime_source.zip")
    packed_source = tmp_path / "c067_source.zip"
    packer.build_packed_archive(
        direct_source,
        packed_source,
        payload_member_name=packer.SHORT_PAYLOAD_MEMBER_NAME,
    )
    return packed_source


def _build_sjkl_payload() -> bytes:
    from experiments.build_sjkl_residual import pack_alpha_block, pack_full_sjkl_payload
    from tac.sjkl_basis import SJKLBasis, pack_sjkl_basis

    basis = SJKLBasis(
        basis_coarse=torch.ones(1, 3, 1, 1),
        scale=torch.tensor([1.0]),
        target_h=4,
        target_w=5,
    ).renormalize()
    basis_bytes = pack_sjkl_basis(basis)
    block_bytes = pack_alpha_block(
        [np.array([0], dtype=np.uint8)],
        [1.0],
        [0.0],
        alpha_bits=6,
    )
    return pack_full_sjkl_payload(basis_bytes, block_bytes)


def test_build_sjkl_archive_includes_runtime_consumable_sjkl(tmp_path: Path) -> None:
    builder = _load_module(BUILDER_PATH, "_sjkl_c067_builder_include_test")
    unpacker = _load_module(UNPACKER_PATH, "_sjkl_c067_unpacker_include_test")
    inflate = _load_module(INFLATE_RENDERER_PATH, "_sjkl_c067_inflate_include_test")
    source = _write_packed_public_floor_source(tmp_path)
    sjkl = tmp_path / "sjkl.bin"
    sjkl.write_bytes(_build_sjkl_payload())

    manifest = builder.build_sjkl_archive(
        source_archive=source,
        sjkl_bin=sjkl,
        output_dir=tmp_path / "out",
    )

    archive_path = Path(manifest["output_archive"]["path"])
    with zipfile.ZipFile(archive_path) as zf:
        assert zf.namelist() == ["p"]
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        zf.extractall(extract_dir)

    summary = unpacker.unpack_renderer_payload(extract_dir)
    members = {member["name"]: member for member in summary["members"]}
    assert members["sjkl.bin"]["bytes"] == sjkl.stat().st_size
    assert (extract_dir / "sjkl.bin").read_bytes() == sjkl.read_bytes()

    state = inflate._load_sjkl_residual_from_archive_dir(extract_dir)
    assert state is not None
    assert int(state["qs"].shape[0]) == 1
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["sjkl_payload"]["member_name"] == "sjkl.bin"
    assert manifest["sjkl_payload"]["bytes"] == sjkl.stat().st_size
    assert "sjkl.bin" in manifest["payload_member_names"]["output_logical_runtime_members"]
    assert manifest["packed_payload"]["compression_selection"]["header_minimized"] is True
    assert manifest["packed_payload"]["compression_selection"]["member_order_optimized"] is True
    assert all("codec" not in member for member in manifest["packed_payload"]["header"]["members"])


def test_build_sjkl_archive_is_deterministic_and_writes_manifest(tmp_path: Path) -> None:
    builder = _load_module(BUILDER_PATH, "_sjkl_c067_builder_determinism_test")
    source = _write_packed_public_floor_source(tmp_path)
    sjkl = tmp_path / "sjkl.bin"
    sjkl.write_bytes(_build_sjkl_payload())

    first = builder.build_sjkl_archive(
        source_archive=source,
        sjkl_bin=sjkl,
        output_dir=tmp_path / "out1",
    )
    second = builder.build_sjkl_archive(
        source_archive=source,
        sjkl_bin=sjkl,
        output_dir=tmp_path / "out2",
    )

    first_archive = Path(first["output_archive"]["path"])
    second_archive = Path(second["output_archive"]["path"])
    assert first_archive.read_bytes() == second_archive.read_bytes()
    assert first["output_archive"]["sha256"] == second["output_archive"]["sha256"]
    assert first["output_archive"]["bytes"] == second["output_archive"]["bytes"]
    assert first["source_archive"]["bytes"] == source.stat().st_size
    assert first["sjkl_payload"]["sha256"] == second["sjkl_payload"]["sha256"]
    assert first["payload_member_names"]["output_archive_members"] == ["p"]

    manifest_path = tmp_path / "out1" / "sjkl_c067_archive_manifest.json"
    written = json.loads(manifest_path.read_text())
    assert written["schema"] == "sjkl_c067_archive_builder_v1"
    assert written["output_archive"]["sha256"] == first["output_archive"]["sha256"]
    assert written["runtime_contract"]["sidecars_required"] is False
    assert written["runtime_contract"]["score_affecting_payload_charged_in_archive"] is True
    assert written["runtime_contract"]["runtime_apply_proof"]["verified"] is True
    assert written["packed_payload"]["compression_selection"]["candidate_count"] == 24
    assert set(written["packed_payload"]["compression_selection"]["chosen_member_order"]) == {
        "renderer.bin",
        "masks.mkv",
        "optimized_poses.bin",
        "sjkl.bin",
    }


def test_build_sjkl_archive_rejects_payload_over_configured_cap(tmp_path: Path) -> None:
    builder = _load_module(BUILDER_PATH, "_sjkl_c067_builder_cap_test")
    source = _write_packed_public_floor_source(tmp_path)
    sjkl = tmp_path / "sjkl.bin"
    payload = _build_sjkl_payload()
    sjkl.write_bytes(payload)

    with pytest.raises(ValueError, match="exceeds byte cap"):
        builder.build_sjkl_archive(
            source_archive=source,
            sjkl_bin=sjkl,
            output_dir=tmp_path / "out",
            max_sjkl_bytes=len(payload) - 1,
        )


def test_build_sjkl_archive_rejects_missing_runtime_apply_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_module(BUILDER_PATH, "_sjkl_c067_builder_apply_proof_test")
    broken_runtime = tmp_path / "inflate_renderer.py"
    broken_runtime.write_text("SJKL_REQUIRE_APPLIED = 'mentioned but no proof'\n")
    original_verify = builder.verify_runtime_apply_proof
    monkeypatch.setattr(builder, "verify_runtime_apply_proof", lambda: original_verify(broken_runtime))
    source = _write_packed_public_floor_source(tmp_path)
    sjkl = tmp_path / "sjkl.bin"
    sjkl.write_bytes(_build_sjkl_payload())

    with pytest.raises(ValueError, match="runtime-apply proof is absent"):
        builder.build_sjkl_archive(
            source_archive=source,
            sjkl_bin=sjkl,
            output_dir=tmp_path / "out",
        )


def test_build_sjkl_top_level_sibling_preserves_source_payload(
    tmp_path: Path,
) -> None:
    builder = _load_module(BUILDER_PATH, "_sjkl_c067_builder_sibling_test")
    unpacker = _load_module(UNPACKER_PATH, "_sjkl_c067_unpacker_sibling_test")
    inflate = _load_module(INFLATE_RENDERER_PATH, "_sjkl_c067_inflate_sibling_test")
    source = _write_packed_public_floor_source(tmp_path)
    with zipfile.ZipFile(source) as zf:
        source_names = zf.namelist()
        source_p_bytes = zf.read("p")
    assert source_names == ["p"]
    sjkl = tmp_path / "sjkl.bin"
    sjkl.write_bytes(_build_sjkl_payload())

    manifest = builder.build_sjkl_archive(
        source_archive=source,
        sjkl_bin=sjkl,
        output_dir=tmp_path / "out",
        archive_layout="top_level_sibling",
    )

    archive_path = Path(manifest["output_archive"]["path"])
    with zipfile.ZipFile(archive_path) as zf:
        assert zf.namelist() == ["p", "sjkl.bin"]
        assert zf.read("p") == source_p_bytes
        assert zf.read("sjkl.bin") == sjkl.read_bytes()
        assert [info.date_time for info in zf.infolist()] == [
            (1980, 1, 1, 0, 0, 0),
            (1980, 1, 1, 0, 0, 0),
        ]
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        zf.extractall(extract_dir)

    summary = unpacker.unpack_renderer_payload(extract_dir)
    members = {member["name"]: member for member in summary["members"]}
    assert {"renderer.bin", "masks.mkv", "optimized_poses.bin"} <= set(members)
    assert (extract_dir / "sjkl.bin").read_bytes() == sjkl.read_bytes()
    state = inflate._load_sjkl_residual_from_archive_dir(extract_dir)
    assert state is not None

    assert manifest["packed_payload"]["archive_layout"] == "top_level_sibling"
    assert manifest["packed_payload"]["payload_format"] == "preserve_source_payload_plus_top_level_sjkl"
    assert manifest["payload_member_names"]["output_archive_members"] == ["p", "sjkl.bin"]
    assert manifest["packed_payload"]["compression_selection"]["layout_summary"][
        "preserves_source_payload_bytes"
    ] is True
    assert manifest["payload_member_names"]["output_logical_runtime_members"] == [
        "renderer.bin",
        "masks.mkv",
        "optimized_poses.bin",
        "sjkl.bin",
    ]
    assert (
        manifest["output_archive"]["delta_bytes_vs_source_archive"]
        < sjkl.stat().st_size + 128
    )
