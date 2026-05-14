# SPDX-License-Identifier: MIT
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
SEARCH_PATH = REPO / "experiments" / "search_pr79_renderer_parity_constrained_shrink.py"


def _load_search() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_pr79_renderer_parity_constrained_search_test",
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


def _parity(ok: bool) -> dict[str, Any]:
    return {
        "ok": ok,
        "stage": "pre_archive_render_output_parity",
        "failure_class": None if ok else "render_output_parity_unsafe",
        "output_parity": {
            "ok": ok,
            "aggregate": {
                "ok": ok,
                "mean_abs_delta": 0.0 if ok else 1.0,
                "rms_delta": 0.0 if ok else 2.0,
                "max_abs_delta": 0.0 if ok else 3.0,
            },
        },
    }


def _safe_strict_pose_safety(output_json: Path, **_kwargs: Any) -> dict[str, Any]:
    output_json.write_bytes(b"{}\n")
    return {
        "safe_for_exact_eval_dispatch": True,
        "failure_class": None,
        "fail_closed_reasons": [],
        "output_parity": {"ok": True},
    }


def test_parse_transform_spec_supports_pr79_reblock_and_zero_prefix() -> None:
    search = _load_search()

    reblock = search.parse_transform_spec("qzs3-reblock:64")
    zero = search.parse_transform_spec("zero-fp4-prefix:frame2_head.pre:0.02")

    assert reblock.kind == "qzs3_reblock"
    assert reblock.candidate_id == "qzs3_reblock_b0064"
    assert zero.kind == "zero_fp4_prefix"
    assert zero.candidate_id == "zero_fp4_frame2_head.pre_0.02"


def test_pre_archive_parity_failure_does_not_build_or_emit_archive(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    search = _load_search()
    source, _members = _write_p3_source(tmp_path / "source.zip")

    def fail_parity(**_kwargs: Any) -> dict[str, Any]:
        return _parity(False)

    def fail_if_called(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("archive build must not run after pre-archive parity fail")

    monkeypatch.setattr(search, "_build_pre_archive_parity", fail_parity)
    monkeypatch.setattr(search, "_build_temp_archive", fail_if_called)
    monkeypatch.setattr(
        search.pose_safety,
        "_load_runtime_state",
        lambda *_args, **_kwargs: {
            "renderer": object(),
            "masks": torch.zeros(2, 1, 1, dtype=torch.long),
            "poses": torch.zeros(1, 6),
        },
    )
    monkeypatch.setattr(search.pose_safety, "_select_pair_indices", lambda *_args: [0])
    monkeypatch.setattr(
        search.pose_safety,
        "_render_pair_batch",
        lambda **_kwargs: torch.zeros(1, 2, 1, 1, 3),
    )
    monkeypatch.setattr(
        search.pose_safety,
        "_summarize_frames",
        lambda label, frames: {"label": label, "shape": list(frames.shape)},
    )

    summary = search.run_search(
        source_archive=source,
        source_evidence_path=None,
        output_dir=tmp_path / "out",
        transform_specs=(search.parse_transform_spec("qzs3-reblock:64"),),
        max_pairs=1,
    )

    row = summary["candidates"][0]
    assert row["failure_class"] == "pre_archive_render_output_parity_unsafe"
    assert row["archive"] is None
    assert not list((tmp_path / "out").glob("*/archive.zip"))
    assert summary["dispatch_recommendation"]["recommendation"] == "do_not_dispatch"


def test_strict_pose_safety_pass_is_required_before_archive_is_emitted(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    search = _load_search()
    source, _members = _write_p3_source(tmp_path / "source.zip")

    monkeypatch.setattr(search, "_build_pre_archive_parity", lambda **_kwargs: _parity(True))
    monkeypatch.setattr(
        search,
        "_run_strict_pose_safety",
        lambda **_kwargs: {
            "safe_for_exact_eval_dispatch": False,
            "failure_class": "renderer_transplant_pose_safety_failed",
            "fail_closed_reasons": ["render_output_parity_unsafe"],
        },
    )
    monkeypatch.setattr(
        search.pose_safety,
        "_load_runtime_state",
        lambda *_args, **_kwargs: {
            "renderer": object(),
            "masks": torch.zeros(2, 1, 1, dtype=torch.long),
            "poses": torch.zeros(1, 6),
        },
    )
    monkeypatch.setattr(search.pose_safety, "_select_pair_indices", lambda *_args: [0])
    monkeypatch.setattr(
        search.pose_safety,
        "_render_pair_batch",
        lambda **_kwargs: torch.zeros(1, 2, 1, 1, 3),
    )
    monkeypatch.setattr(
        search.pose_safety,
        "_summarize_frames",
        lambda label, frames: {"label": label, "shape": list(frames.shape)},
    )

    summary = search.run_search(
        source_archive=source,
        source_evidence_path=None,
        output_dir=tmp_path / "out",
        transform_specs=(search.parse_transform_spec("qzs3-reblock:64"),),
        max_pairs=1,
        require_byte_saving=False,
    )

    row = summary["candidates"][0]
    assert row["failure_class"] == "strict_pose_safety_failed"
    assert row["archive"] is None
    assert row["output_archive"]["screened_bytes"] > 0
    assert not list((tmp_path / "out").glob("*/archive.zip"))


def test_safe_candidate_is_emitted_as_exact_eval_ready_no_dispatch(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    search = _load_search()
    source, members = _write_p3_source(tmp_path / "source.zip")

    monkeypatch.setattr(search, "_build_pre_archive_parity", lambda **_kwargs: _parity(True))
    monkeypatch.setattr(search, "_run_strict_pose_safety", _safe_strict_pose_safety)
    monkeypatch.setattr(
        search.pose_safety,
        "_load_runtime_state",
        lambda *_args, **_kwargs: {
            "renderer": object(),
            "masks": torch.zeros(2, 1, 1, dtype=torch.long),
            "poses": torch.zeros(1, 6),
        },
    )
    monkeypatch.setattr(search.pose_safety, "_select_pair_indices", lambda *_args: [0])
    monkeypatch.setattr(
        search.pose_safety,
        "_render_pair_batch",
        lambda **_kwargs: torch.zeros(1, 2, 1, 1, 3),
    )
    monkeypatch.setattr(
        search.pose_safety,
        "_summarize_frames",
        lambda label, frames: {"label": label, "shape": list(frames.shape)},
    )

    summary = search.run_search(
        source_archive=source,
        source_evidence_path=None,
        output_dir=tmp_path / "out",
        transform_specs=(search.parse_transform_spec("qzs3-reblock:64"),),
        max_pairs=1,
        require_byte_saving=False,
    )

    row = summary["candidates"][0]
    assert row["exact_eval_ready"] is True
    assert row["dispatch_recommendation"] == "exact_eval_ready_no_dispatch"
    assert Path(row["archive"]).is_file()
    assert summary["remote_gpu_dispatch_performed"] is False
    assert summary["dispatch_recommendation"]["recommendation"] == "exact_eval_ready_no_dispatch"
    loaded = search.shrink_builder.blockfp.extract_runtime_members(Path(row["archive"]))[0]
    assert loaded["masks.mkv"] == members["masks.mkv"]
    assert loaded["optimized_poses.qp1"] == members["optimized_poses.qp1"]
    assert loaded["seg_tile_actions.bin"] == members["seg_tile_actions.bin"]
