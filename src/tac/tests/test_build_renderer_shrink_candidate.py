from __future__ import annotations

import importlib.util
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

import brotli
import torch

from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer
from tac.quantizr_qzs3_codec import encode_qzs3_state_dict


REPO = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO / "experiments" / "build_renderer_shrink_candidate.py"


def _load_builder() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_renderer_shrink_candidate_test",
        BUILDER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_p3_source(path: Path) -> tuple[Path, dict[str, bytes]]:
    torch.manual_seed(123)
    renderer = encode_qzs3_state_dict(
        build_quantizr_faithful_renderer().eval().state_dict(),
        block_size=32,
    )
    masks = b"\x12\x00synthetic-mask-obu" * 8
    actions = struct.pack("<HBB", 0, 0, 0)
    qp1 = b"QP1" + struct.pack("<H", 1024)
    mask_br = brotli.compress(masks, quality=11, lgwin=24)
    renderer_br = brotli.compress(renderer, quality=11, lgwin=24)
    actions_br = brotli.compress(actions, quality=11, lgwin=24)
    qp1_br = brotli.compress(qp1, quality=11, lgwin=24)
    payload = (
        b"P3"
        + struct.pack("<IHH", len(mask_br), len(renderer_br), len(actions_br))
        + mask_br
        + renderer_br
        + actions_br
        + qp1_br
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o600 << 16
        zf.writestr(info, payload)
    return path, {
        "renderer.bin": renderer,
        "masks.mkv": masks,
        "seg_tile_actions.bin": actions,
        "optimized_poses.qp1": qp1,
    }


def test_renderer_shrink_preserves_pr75_non_renderer_members(tmp_path: Path) -> None:
    builder = _load_builder()
    source, members = _write_p3_source(tmp_path / "source.zip")

    summary = builder.build_renderer_shrink_candidates(
        source_archive=source,
        output_dir=tmp_path / "out",
        qzs3_block_sizes=(48,),
    )

    assert summary["score_claim"] is False
    assert summary["remote_gpu_dispatch_performed"] is False
    assert summary["source_payload_layout"] == "pr75_preserved_slices"
    candidate = summary["candidates"][0]
    assert candidate["qzs3_block_size"] == 48
    assert candidate["payload_format"] == "pr75_p3_preserve_slices"
    manifest = (tmp_path / "out" / candidate["candidate_id"] / "build_manifest.json")
    assert manifest.exists()
    with zipfile.ZipFile(candidate["archive"]) as zf:
        assert zf.namelist() == ["p"]
    loaded = builder.blockfp.extract_runtime_members(Path(candidate["archive"]))[0]
    assert loaded["masks.mkv"] == members["masks.mkv"]
    assert loaded["optimized_poses.qp1"] == members["optimized_poses.qp1"]
    assert loaded["seg_tile_actions.bin"] == members["seg_tile_actions.bin"]
    assert loaded["renderer.bin"] != members["renderer.bin"]


def test_renderer_shrink_can_transplant_external_qzs3_export(tmp_path: Path) -> None:
    builder = _load_builder()
    source, members = _write_p3_source(tmp_path / "source.zip")
    external_renderer = encode_qzs3_state_dict(
        build_quantizr_faithful_renderer().eval().state_dict(),
        block_size=96,
    )
    export_path = tmp_path / "external_renderer.bin"
    export_path.write_bytes(external_renderer)

    summary = builder.build_renderer_shrink_candidates(
        source_archive=source,
        output_dir=tmp_path / "out",
        renderer_export=export_path,
        qzs3_block_sizes=(48,),
    )

    assert summary["renderer_input"]["role"] == "external_renderer_export"
    assert summary["candidate_count"] == 2
    direct = next(
        item
        for item in summary["candidates"]
        if item["renderer_transform_kind"] == "external_export_direct"
    )
    assert direct["qzs3_block_size"] is None
    assert direct["renderer_sha256"] == builder._sha256_bytes(external_renderer)
    loaded = builder.blockfp.extract_runtime_members(Path(direct["archive"]))[0]
    assert loaded["renderer.bin"] == external_renderer
    assert loaded["masks.mkv"] == members["masks.mkv"]
    assert loaded["optimized_poses.qp1"] == members["optimized_poses.qp1"]
    assert loaded["seg_tile_actions.bin"] == members["seg_tile_actions.bin"]


def test_renderer_shrink_rebuilds_logical_p3_slices_for_opaque_public_payload(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    _source, members = _write_p3_source(tmp_path / "source.zip")
    opaque_source = tmp_path / "opaque_public.zip"
    with zipfile.ZipFile(opaque_source, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o600 << 16
        zf.writestr(info, b"opaque-public-fixed-slice")

    slices = builder.load_pr75_slices_for_renderer_shrink(
        opaque_source,
        members,
        brotli_quality=11,
    )

    assert slices is not None
    assert slices["format"] == "logical_rebrotli_p3"
    assert slices["rebuilt_from_logical_members"] is True

    payload, payload_meta = builder._build_pr75_payload(
        slices,
        renderer_bytes=members["renderer.bin"],
        brotli_quality=11,
    )
    archive = tmp_path / "candidate.zip"
    builder._write_single_member_archive(archive, payload)
    unpack_summary = builder._verify_archive(
        archive,
        expected_renderer=members["renderer.bin"],
        expected_non_renderer_members={
            name: data for name, data in members.items() if name != "renderer.bin"
        },
    )

    assert payload_meta["payload_format"] == "pr75_p3_logical_rebrotli_slices"
    assert unpack_summary["payload_format"] == "public_pr75_qzs3_qp1_segactions_p3"
