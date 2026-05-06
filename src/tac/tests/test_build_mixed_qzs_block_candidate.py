from __future__ import annotations

import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path

import pytest
import torch

from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer
from tac.quantizr_qzs3_codec import (
    decode_mixed_qzs_block_state_dict,
    encode_qzs3_state_dict,
    _is_fp4_weight_name,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO_ROOT / "experiments" / "build_mixed_qzs_block_candidate.py"
UNPACK_PATH = REPO_ROOT / "submissions" / "robust_current" / "unpack_renderer_payload.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_mixed_qzs_block_candidate_test", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_unpacker():
    spec = importlib.util.spec_from_file_location("unpack_renderer_payload_test", UNPACK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source_archive(tmp_path: Path) -> Path:
    torch.manual_seed(12)
    model = build_quantizr_faithful_renderer().eval()
    source = tmp_path / "source.zip"
    pose_values: list[float] = []
    for row in range(4):
        pose_values.extend([20.0 + row / 512.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pose_bytes = struct.pack("<" + "e" * len(pose_values), *pose_values)
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("renderer.bin", encode_qzs3_state_dict(model, block_size=32))
        zf.writestr("masks.mkv", b"mask-bytes" * 11)
        zf.writestr("optimized_poses.bin", pose_bytes)
    return source


def test_mixed_policy_build_is_deterministic_exact_evaluable_and_records_blocks(tmp_path: Path) -> None:
    builder = _load_builder()
    unpacker = _load_unpacker()
    source = _source_archive(tmp_path)
    evidence = tmp_path / "contest_auth_eval.json"
    evidence.write_text(json.dumps({"device": "cuda", "note": "fixture only"}) + "\n")
    policy = builder.parse_block_policy("mixed:64:frame1_head=32,pose_mlp=32")

    out_a = tmp_path / "a" / "archive.zip"
    out_b = tmp_path / "b" / "archive.zip"
    meta_a = builder.build_candidate_for_policy(
        source,
        out_a,
        policy=policy,
        source_evidence_path=evidence,
    )
    meta_b = builder.build_candidate_for_policy(
        source,
        out_b,
        policy=policy,
        source_evidence_path=evidence,
    )

    assert out_a.read_bytes() == out_b.read_bytes()
    assert meta_a["output_archive_sha256"] == meta_b["output_archive_sha256"]
    assert meta_a["score_claim"] is False
    assert meta_a["promotion_eligible"] is False
    assert meta_a["exact_evaluable_archive"] is True
    assert "MQZ1 runtime decoder" in meta_a["exact_evaluable_reason"]
    assert meta_a["source_archive_sha256"]
    assert meta_a["source_evidence"]["exists"] is True
    assert meta_a["block_policy"]["wire_format"] == "MQZ1"
    assert meta_a["block_policy"]["runtime_decoder_available"] is True
    assert set(meta_a["block_policy"]["block_size_counts"]) == {"32", "64"}
    assert meta_a["renderer"]["output_renderer_format"] == "MQZ1"

    with zipfile.ZipFile(out_a) as zf:
        assert zf.namelist() == ["p"]
        assert zf.infolist()[0].date_time == builder.FIXED_ZIP_TIMESTAMP
        payload_member = zf.read("p")

    unpacked = tmp_path / "unpacked"
    unpacked.mkdir()
    (unpacked / "p").write_bytes(payload_member)
    unpack_summary = unpacker.unpack_renderer_payload(unpacked)
    renderer_payload = (unpacked / "renderer.bin").read_bytes()
    assert renderer_payload.startswith(b"MQZ1")
    assert (unpacked / "masks.mkv").read_bytes() == b"mask-bytes" * 11
    assert [item["name"] for item in unpack_summary["members"]] == [
        "renderer.bin",
        "masks.mkv",
        "optimized_poses.bin",
    ]
    decoded_state = decode_mixed_qzs_block_state_dict(renderer_payload, device="cpu")
    assert list(decoded_state) == list(build_quantizr_faithful_renderer().state_dict())


def test_global_policy_reuses_exact_evaluable_qzs3_contract_but_stays_non_promotable(tmp_path: Path) -> None:
    builder = _load_builder()
    source = _source_archive(tmp_path)
    policy = builder.parse_block_policy("global:32")

    out = tmp_path / "global" / "archive.zip"
    meta = builder.build_candidate_for_policy(source, out, policy=policy)

    assert meta["score_claim"] is False
    assert meta["promotion_eligible"] is False
    assert meta["exact_evaluable_archive"] is True
    assert "not promotable without exact CUDA auth eval" in meta["exact_evaluable_reason"]
    assert meta["block_policy"]["wire_format"] == "QZS3"
    assert meta["submission_path"]["layout"] == "pr64_mask_first_single_blob"
    with zipfile.ZipFile(out) as zf:
        assert zf.namelist() == ["p"]


def test_build_candidates_writes_summary_and_per_candidate_json(tmp_path: Path) -> None:
    builder = _load_builder()
    source = _source_archive(tmp_path)
    out_dir = tmp_path / "screen"

    summary = builder.build_candidates(
        source,
        out_dir,
        policy_specs=("global:32", "mixed:48:frame1_head=32"),
    )

    assert summary["score_claim"] is False
    assert summary["promotion_eligible"] is False
    assert summary["candidate_count"] == 2
    assert (out_dir / "summary.json").is_file()
    for item in summary["candidates"]:
        assert item["score_claim"] is False
        assert item["promotion_eligible"] is False
        assert Path(item["output_archive"]).is_file()
        provenance = Path(item["output_archive"]).with_name("build_provenance.json")
        assert provenance.is_file()


def test_component_aware_policy_protects_pose_critical_tensors() -> None:
    builder = _load_builder()
    policy = builder.parse_block_policy("component-aware-v1:frame2_all64")
    model = build_quantizr_faithful_renderer()

    assert policy.default_block_size == 32
    assert policy.component_awareness["schema"] == "component_aware_mixed_qzs_policy_v1"
    protected = {"shared_trunk", "frame1_head", "pose_mlp"}
    for key in model.state_dict():
        if not _is_fp4_weight_name(key):
            continue
        if any(key == prefix or key.startswith(prefix + ".") for prefix in protected):
            assert policy.block_size_for(key) == 32
        elif key.startswith("frame2_head."):
            assert policy.block_size_for(key) == 64


def test_component_aware_build_records_evidence_and_decodes(tmp_path: Path) -> None:
    builder = _load_builder()
    unpacker = _load_unpacker()
    source = _source_archive(tmp_path)
    policy_evidence = tmp_path / "mqz1_negative_eval.json"
    policy_evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "device": "cuda",
                "avg_posenet_dist": 0.32721686,
                "avg_segnet_dist": 0.00086619,
                "score_recomputed_from_components": 2.079372901891654,
                "promotion_eligible": False,
                "score_claim": False,
            }
        )
        + "\n"
    )
    policy = builder.parse_block_policy("component-aware-v1:frame2_pre64")

    out = tmp_path / "component_aware" / "archive.zip"
    meta = builder.build_candidate_for_policy(
        source,
        out,
        policy=policy,
        policy_evidence_paths=(policy_evidence,),
    )

    assert meta["score_claim"] is False
    assert meta["promotion_eligible"] is False
    assert meta["policy_evidence_inputs"][0]["sha256"]
    assert meta["policy_evidence_inputs"][0]["summary"]["pose_collapse_signal"] is True
    assert meta["block_policy"]["policy"]["component_awareness"]["tier"] == "frame2_pre64"
    tensor_blocks = {
        item["name"]: item["block_size"]
        for item in meta["block_policy"]["fp4_tensors"]
    }
    assert tensor_blocks["frame2_head.pre.dw.weight"] == 64
    assert tensor_blocks["frame2_head.pre.pw.weight"] == 64
    assert tensor_blocks["frame1_head.pre.dw.weight"] == 32
    assert tensor_blocks["shared_trunk.fuse.pw.weight"] == 32

    with zipfile.ZipFile(out) as zf:
        payload_member = zf.read("p")
    unpacked = tmp_path / "unpacked_component_aware"
    unpacked.mkdir()
    (unpacked / "p").write_bytes(payload_member)
    unpacker.unpack_renderer_payload(unpacked)
    renderer_payload = (unpacked / "renderer.bin").read_bytes()
    decoded_state = decode_mixed_qzs_block_state_dict(renderer_payload, device="cpu")
    assert list(decoded_state) == list(build_quantizr_faithful_renderer().state_dict())


def test_component_aware_defaults_write_summary_with_evidence(tmp_path: Path) -> None:
    builder = _load_builder()
    source = _source_archive(tmp_path)
    evidence = tmp_path / "sensitivity_ranked_renderer_blob_perturbation_basis_v1.json"
    evidence.write_text(
        json.dumps(
            {
                "atoms": [
                    {"layer_name": "renderer.head", "sensitivity_score": 10.0},
                    {"layer_name": "renderer.stem_res.conv1", "sensitivity_score": 5.0},
                ],
                "promotion_eligible": False,
                "score_claim": False,
            }
        )
        + "\n"
    )
    out_dir = tmp_path / "component_aware_defaults"

    summary = builder.build_candidates(
        source,
        out_dir,
        policy_evidence_paths=(evidence,),
    )

    assert summary["candidate_count"] == 3
    assert summary["policy_evidence_inputs"][0]["summary"]["top_atom_layers"] == [
        "renderer.head",
        "renderer.stem_res.conv1",
    ]
    assert all(item["component_awareness"] for item in summary["candidates"])


@pytest.mark.parametrize(
    "spec",
    [
        "",
        "mixed:48",
        "mixed:0:frame1_head=32",
        "mixed:48:frame1_head",
        "mixed:48:../bad=32",
        "component-aware-v1:unknown",
        "other:32",
        "global:0",
    ],
)
def test_parse_block_policy_rejects_bad_specs(spec: str) -> None:
    builder = _load_builder()
    with pytest.raises((ValueError, TypeError)):
        builder.parse_block_policy(spec)
