# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path

import torch

from tac.qbf1_renderer_codec import unpack_qbf1_container
from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer
from tac.quantizr_qzs3_codec import encode_qzs3_state_dict

REPO = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO / "experiments" / "build_blockfp_c067_archive.py"
PACKER_PATH = REPO / "experiments" / "build_renderer_packed_payload_archive.py"
UNPACKER_PATH = REPO / "submissions" / "robust_current" / "unpack_renderer_payload.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_direct_qzs3_source(path: Path) -> Path:
    torch.manual_seed(0)
    model = build_quantizr_faithful_renderer().eval()
    pose_values = []
    for row in range(600):
        pose_values.extend([30.0 + row / 512.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    members = {
        "renderer.bin": encode_qzs3_state_dict(model.state_dict(), block_size=32),
        "masks.mkv": b"mask-obu" * 4096,
        "optimized_poses.bin": struct.pack("<" + "e" * len(pose_values), *pose_values),
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


def _write_packed_public_mask_first_source(tmp_path: Path) -> Path:
    packer = _load_module(PACKER_PATH, "_blockfp_c067_test_packer")
    direct_source = _write_direct_qzs3_source(tmp_path / "runtime_source.zip")
    packed_source = tmp_path / "c067_source.zip"
    packer.build_packed_archive(
        direct_source,
        packed_source,
        payload_member_name=packer.SHORT_PAYLOAD_MEMBER_NAME,
        payload_format=packer.PAYLOAD_FORMAT_PUBLIC_PR64_MASK_FIRST_LEN_TABLE,
    )
    return packed_source


def test_build_blockfp_c067_archive_transplants_qbf1_renderer(tmp_path: Path) -> None:
    builder = _load_module(BUILDER_PATH, "_blockfp_c067_builder_include_test")
    unpacker = _load_module(UNPACKER_PATH, "_blockfp_c067_unpacker_include_test")
    source = _write_packed_public_mask_first_source(tmp_path)

    summary = builder.build_blockfp_c067_archives(
        source_archive=source,
        output_dir=tmp_path / "out",
        block_sizes=(64,),
    )

    assert summary["score_claim"] is False
    assert summary["promotion_eligible"] is False
    assert summary["block_sizes"] == [64]
    candidate = summary["candidates"][0]
    archive_path = Path(candidate["archive_path"])
    manifest_path = Path(candidate["manifest_path"])
    assert archive_path.exists()
    assert manifest_path.exists()

    with zipfile.ZipFile(archive_path) as zf:
        assert zf.namelist() == ["p"]
        payload_member = zf.read("p")
    header, members = builder._parse_packed_payload_member("p", payload_member)
    assert header["payload_format"] == "public_pr64_mask_first_len_table"
    assert members["renderer.bin"].startswith(b"QBF1")
    assert members["masks.mkv"] == b"mask-obu" * 4096
    unpack_qbf1_container(members["renderer.bin"])

    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "blockfp_c067_archive_builder_v1"
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["renderer_source_member"]["wire_format"] == "QZS3"
    assert manifest["transformed_renderer_payload"]["wire_format"] == "QBF1"
    assert manifest["transformed_renderer_payload"]["block_size"] == 64
    assert manifest["runtime_contract"]["score_affecting_payload_charged_in_archive"] is True
    assert manifest["runtime_contract"]["scorer_imports_at_inflate_time"] is False
    assert manifest["output_archive"]["sha256"] == candidate["archive_sha256"]

    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(extract_dir)
    unpack_summary = unpacker.unpack_renderer_payload(extract_dir)
    unpacked = {member["name"]: member for member in unpack_summary["members"]}
    assert unpacked["renderer.bin"]["sha256"] == manifest["transformed_renderer_payload"]["sha256"]


def test_build_blockfp_c067_archive_is_deterministic(tmp_path: Path) -> None:
    builder = _load_module(BUILDER_PATH, "_blockfp_c067_builder_determinism_test")
    source = _write_packed_public_mask_first_source(tmp_path)

    first = builder.build_blockfp_c067_archives(
        source_archive=source,
        output_dir=tmp_path / "out1",
        block_sizes=(128,),
    )
    second = builder.build_blockfp_c067_archives(
        source_archive=source,
        output_dir=tmp_path / "out2",
        block_sizes=(128,),
    )

    first_archive = Path(first["candidates"][0]["archive_path"])
    second_archive = Path(second["candidates"][0]["archive_path"])
    assert first_archive.read_bytes() == second_archive.read_bytes()
    assert first["best_by_output_archive_bytes"]["sha256"] == second["best_by_output_archive_bytes"]["sha256"]
    assert first["candidates"][0]["renderer_sha256"] == second["candidates"][0]["renderer_sha256"]


def test_build_blockfp_c067_archive_fails_closed_on_non_jfg_renderer(
    tmp_path: Path,
) -> None:
    builder = _load_module(BUILDER_PATH, "_blockfp_c067_builder_failclosed_test")
    source = tmp_path / "bad_source.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in {
            "renderer.bin": b"ASYM-not-a-jfg",
            "masks.mkv": b"mask",
            "optimized_poses.bin": b"\x00" * 12,
        }.items():
            zf.writestr(name, data)

    try:
        builder.build_blockfp_c067_archives(
            source_archive=source,
            output_dir=tmp_path / "out_bad",
            block_sizes=(32,),
        )
    except ValueError as exc:
        assert "structural blocker" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unsupported renderer should fail closed")
