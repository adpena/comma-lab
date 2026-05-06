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
SEARCH_PATH = REPO / "experiments" / "search_renderer_parity_shrink_candidate.py"


def _load_search() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_renderer_parity_shrink_search_test",
        SEARCH_PATH,
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
    payload = (
        b"P3"
        + struct.pack(
            "<IHH",
            len(brotli.compress(masks, quality=11, lgwin=24)),
            len(brotli.compress(renderer, quality=11, lgwin=24)),
            len(brotli.compress(actions, quality=11, lgwin=24)),
        )
        + brotli.compress(masks, quality=11, lgwin=24)
        + brotli.compress(renderer, quality=11, lgwin=24)
        + brotli.compress(actions, quality=11, lgwin=24)
        + brotli.compress(qp1, quality=11, lgwin=24)
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


def test_parse_transform_spec_normalizes_candidate_ids() -> None:
    search = _load_search()

    zero = search.parse_transform_spec("zero-fp4-prefix:frame2_head:0.075")
    mixed = search.parse_transform_spec("mixed-block-prefix:frame2_head:64")

    assert zero.kind == "zero_fp4_prefix"
    assert zero.candidate_id == "zero_fp4_frame2_head_0.075"
    assert mixed.kind == "mixed_block_prefix"
    assert mixed.candidate_id == "mixed_block_frame2_head_64"


def test_zero_fp4_transform_records_structured_weight_changes() -> None:
    search = _load_search()
    state = build_quantizr_faithful_renderer().state_dict()

    transformed, meta = search._apply_zero_fp4_prefix(
        state,
        prefix="frame2_head",
        threshold_fraction=0.1,
    )

    assert meta["transform_family"] == "qzs3_same_block_fp4_threshold_zero"
    assert meta["changed_tensor_count"] > 0
    assert meta["zeroed_value_count"] > 0
    assert not torch.equal(
        transformed["frame2_head.pre.pw.weight"],
        state["frame2_head.pre.pw.weight"],
    )
    assert torch.equal(
        transformed["frame1_head.pre.pw.weight"],
        state["frame1_head.pre.pw.weight"],
    )


def test_search_builds_pr75_candidate_and_fails_closed_without_preflight(tmp_path: Path) -> None:
    search = _load_search()
    source, members = _write_p3_source(tmp_path / "source.zip")

    summary = search.run_search(
        source_archive=source,
        source_evidence_path=None,
        output_dir=tmp_path / "out",
        transform_specs=(
            search.parse_transform_spec("zero-fp4-prefix:frame2_head:0.1"),
        ),
        max_preflight_candidates=0,
        skip_preflight=True,
    )

    candidate = summary["candidates"][0]
    assert summary["score_claim"] is False
    assert summary["remote_gpu_dispatch_performed"] is False
    assert candidate["promotion_eligible"] is False
    assert candidate["pose_safety"]["preflight_ran"] is False
    assert summary["dispatch_recommendation"]["recommendation"] == "do_not_dispatch"

    archive = Path(candidate["archive"])
    manifest = archive.with_name("build_manifest.json")
    assert archive.is_file()
    assert manifest.is_file()
    with zipfile.ZipFile(archive) as zf:
        assert zf.namelist() == ["p"]
    loaded = search.blockfp.extract_runtime_members(archive)[0]
    assert loaded["masks.mkv"] == members["masks.mkv"]
    assert loaded["optimized_poses.qp1"] == members["optimized_poses.qp1"]
    assert loaded["seg_tile_actions.bin"] == members["seg_tile_actions.bin"]
    assert loaded["renderer.bin"] != members["renderer.bin"]


def test_recommendation_marks_pose_safe_byte_saver_exact_eval_ready() -> None:
    search = _load_search()
    candidate = {
        "candidate_id": "safe_small_renderer_shrink",
        "archive_bytes": 123,
        "delta_bytes_vs_source_archive": -1,
        "frontier_score_if_only_bytes_change": search.FRONTIER_SCORE - 1e-6,
        "pose_safety": {"safe_for_exact_eval_dispatch": True},
    }

    recommendation = search._recommend_dispatch([candidate])

    assert recommendation["recommendation"] == "exact_eval_ready_no_dispatch"
    assert recommendation["remote_gpu_dispatch_performed"] is False
    assert recommendation["candidate"]["candidate_id"] == "safe_small_renderer_shrink"
