from __future__ import annotations

import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path

import torch

from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer
from tac.quantizr_qzs3_codec import encode_qzs3_state_dict

REPO = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO / "experiments" / "build_imp_c067_bridge_candidates.py"
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
    packer = _load_module(PACKER_PATH, "_imp_c067_bridge_test_packer")
    direct_source = _write_direct_qzs3_source(tmp_path / "runtime_source.zip")
    packed_source = tmp_path / "c067_source.zip"
    packer.build_packed_archive(
        direct_source,
        packed_source,
        payload_member_name=packer.SHORT_PAYLOAD_MEMBER_NAME,
        payload_format=packer.PAYLOAD_FORMAT_PUBLIC_PR64_MASK_FIRST_LEN_TABLE,
    )
    return packed_source


def test_build_imp_c067_bridge_candidates_are_deterministic(tmp_path: Path) -> None:
    builder = _load_module(BUILDER_PATH, "_imp_c067_bridge_builder_determinism_test")
    source = _write_packed_public_mask_first_source(tmp_path)

    first = builder.build_imp_c067_bridge_candidates(
        source_archive=source,
        output_dir=tmp_path / "out1",
        cycle_counts=(1, 2),
        qzs3_block_sizes=(32,),
    )
    second = builder.build_imp_c067_bridge_candidates(
        source_archive=source,
        output_dir=tmp_path / "out2",
        cycle_counts=(1, 2),
        qzs3_block_sizes=(32,),
    )

    assert first["score_claim"] is False
    assert first["promotion_eligible"] is False
    assert first["bridge_mode"] == "no_train_global_magnitude_prune"
    assert first["best_by_output_archive_bytes"]["archive_sha256"] == second[
        "best_by_output_archive_bytes"
    ]["archive_sha256"]
    assert first["best_by_output_archive_bytes"]["actual_sparsity"] == second[
        "best_by_output_archive_bytes"
    ]["actual_sparsity"]

    manifest_path = Path(first["best_by_output_archive_bytes"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "imp_c067_bridge_candidate_builder_v1"
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["imp_pruning"]["training_applied"] is False
    assert manifest["imp_pruning"]["full_imp_training_required_before_promotion"] is True
    assert manifest["renderer_source_member"]["renderer_wire_format"] == "QZS3"
    assert manifest["transformed_renderer_payload"]["wire_format"] == "QZS3"
    assert manifest["decision_support"]["exact_cuda_auth_eval_required_for_score"] is True
    assert manifest["decision_support"]["safe_to_promote_from_this_manifest"] is False


def test_build_imp_c067_bridge_archive_unpacks_pruned_qzs3(tmp_path: Path) -> None:
    builder = _load_module(BUILDER_PATH, "_imp_c067_bridge_builder_unpack_test")
    unpacker = _load_module(UNPACKER_PATH, "_imp_c067_bridge_unpacker_test")
    source = _write_packed_public_mask_first_source(tmp_path)

    summary = builder.build_imp_c067_bridge_candidates(
        source_archive=source,
        output_dir=tmp_path / "out",
        cycle_counts=(1,),
        qzs3_block_sizes=(32,),
    )

    candidate = summary["best_by_output_archive_bytes"]
    archive_path = Path(candidate["archive_path"])
    manifest = json.loads(Path(candidate["manifest_path"]).read_text())
    with zipfile.ZipFile(archive_path) as zf:
        assert zf.namelist() == ["p"]
        payload_member = zf.read("p")
    _header, members = builder.blockfp_builder._parse_packed_payload_member(
        "p",
        payload_member,
    )
    assert members["renderer.bin"].startswith(b"QZS3")

    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(extract_dir)
    unpack_summary = unpacker.unpack_renderer_payload(extract_dir)
    unpacked = {member["name"]: member for member in unpack_summary["members"]}
    assert unpacked["renderer.bin"]["sha256"] == manifest[
        "transformed_renderer_payload"
    ]["sha256"]


def test_build_imp_c067_bridge_refuses_non_qzs3_renderer(tmp_path: Path) -> None:
    builder = _load_module(BUILDER_PATH, "_imp_c067_bridge_builder_fail_test")
    source = tmp_path / "bad_source.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in {
            "renderer.bin": b"ASYM-not-current-c067",
            "masks.mkv": b"mask",
            "optimized_poses.bin": b"\x00" * 12,
        }.items():
            zf.writestr(name, data)

    try:
        builder.build_imp_c067_bridge_candidates(
            source_archive=source,
            output_dir=tmp_path / "out_bad",
            cycle_counts=(1,),
            qzs3_block_sizes=(32,),
        )
    except ValueError as exc:
        assert "requires a QZS3 JointFrameGenerator" in str(exc)
        assert "old Lane G/ASYM" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("non-QZS3 source should fail closed")
