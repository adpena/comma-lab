from __future__ import annotations

import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

import brotli
import pytest
import torch

from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer
from tac.quantizr_qzs3_codec import encode_qzs3_state_dict


REPO = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO / "experiments" / "build_renderer_group_allocator_candidates.py"


def _load_builder() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_renderer_group_allocator_test",
        BUILDER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_p3_source(path: Path) -> tuple[Path, dict[str, bytes]]:
    torch.manual_seed(321)
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


def _write_source_evidence(path: Path, *, archive: Path, pose: float = 0.0005) -> Path:
    payload = {
        "canonical_score": 0.315,
        "score_recomputed_from_components": 0.315,
        "avg_posenet_dist": pose,
        "avg_segnet_dist": 0.0006,
        "archive_size_bytes": archive.stat().st_size,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": _sha256(archive),
            "archive_size_bytes": archive.stat().st_size,
            "device": "cuda",
            "gpu_model": "Synthetic T4",
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def test_policy_parser_rejects_high_risk_by_default() -> None:
    builder = _load_builder()
    policy = builder.parse_policy_spec("group:shared_trunk:64", source_block_size=32)

    validation = builder._validate_policy(
        policy,
        source_block_size=32,
        allow_high_risk=False,
    )

    assert validation["ok"] is False
    assert validation["risk_class"] == "high_risk_pose_or_shared"
    assert "policy_not_low_risk_frame2_only" in validation["failures"]


def test_allocator_builds_mqz1_candidate_and_preserves_non_renderer_members(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    source = builder.DEFAULT_SOURCE_ARCHIVE
    evidence = builder.DEFAULT_SOURCE_EVIDENCE
    if not source.exists() or not evidence.exists():
        pytest.skip("frontier PR75/QP1 source archive is not present in this checkout")
    source_members = builder.blockfp.extract_runtime_members(source)[0]

    summary = builder.run_allocator(
        source_archive=source,
        source_evidence_path=evidence,
        output_dir=tmp_path / "out",
        policy_specs=("group:frame2_head.pre:64",),
        max_build_candidates=1,
        skip_preflight=True,
    )

    assert summary["score_claim"] is False
    assert summary["remote_gpu_dispatch_performed"] is False
    assert summary["source_guard"]["ok"] is True
    assert summary["candidate_count"] == 1
    row = summary["candidates"][0]
    assert row["policy"]["risk_class"] == "low_risk_frame2_only"
    manifest = json.loads(Path(row["manifest"]).read_text())
    assert manifest["renderer_transform"]["output_format"] == "MQZ1"
    assert manifest["runtime_contract"]["renderer_only_transplant"] is True

    loaded = builder.blockfp.extract_runtime_members(Path(row["archive"]))[0]
    assert loaded["renderer.bin"].startswith(b"MQZ1")
    assert loaded["renderer.bin"] != source_members["renderer.bin"]
    for name, payload in source_members.items():
        if name == "renderer.bin":
            continue
        assert loaded[name] == payload


def test_allocator_fails_closed_on_anegative_source_evidence(tmp_path: Path) -> None:
    builder = _load_builder()
    source, _ = _write_p3_source(tmp_path / "source.zip")
    evidence = _write_source_evidence(
        tmp_path / "source_evidence_bad.json",
        archive=source,
        pose=0.25,
    )

    summary = builder.run_allocator(
        source_archive=source,
        source_evidence_path=evidence,
        output_dir=tmp_path / "out",
        policy_specs=("group:frame2_head.pre:64",),
        skip_preflight=True,
    )

    assert summary["source_guard"]["ok"] is False
    assert summary["source_guard"]["failure_class"] == "source_evidence_not_frontier_safe"
    assert summary["candidate_count"] == 0
    assert summary["dispatch_recommendation"]["recommendation"] == "do_not_dispatch"
