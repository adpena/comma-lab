from __future__ import annotations

import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest
import torch

from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer
from tac.quantizr_qzs3_codec import encode_qzs3_state_dict


REPO = Path(__file__).resolve().parents[3]
PREFLIGHT_PATH = REPO / "experiments" / "preflight_trained_renderer_transplant.py"
PACKER_PATH = REPO / "experiments" / "build_renderer_packed_payload_archive.py"
UNPACKER_PATH = REPO / "submissions" / "robust_current" / "unpack_renderer_payload.py"


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_direct_qzs3_source(path: Path) -> tuple[Path, bytes]:
    torch.manual_seed(0)
    model = build_quantizr_faithful_renderer().eval()
    renderer = encode_qzs3_state_dict(model.state_dict(), block_size=32)
    pose_values: list[float] = []
    for row in range(600):
        pose_values.extend([30.0 + row / 512.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    members = {
        "renderer.bin": renderer,
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
    return path, renderer


def _write_packed_source(tmp_path: Path) -> tuple[Path, bytes]:
    packer = _load_module(PACKER_PATH, "_trained_preflight_test_packer")
    direct_source, renderer = _write_direct_qzs3_source(tmp_path / "runtime_source.zip")
    packed_source = tmp_path / "c067_source.zip"
    packer.build_packed_archive(
        direct_source,
        packed_source,
        payload_member_name=packer.SHORT_PAYLOAD_MEMBER_NAME,
        payload_format=packer.PAYLOAD_FORMAT_PUBLIC_PR64_MASK_FIRST_LEN_TABLE,
    )
    return packed_source, renderer


def _write_changed_renderer(path: Path) -> Path:
    torch.manual_seed(0)
    model = build_quantizr_faithful_renderer().eval()
    state = model.state_dict()
    first_key = next(iter(state))
    state[first_key] = state[first_key].clone()
    state[first_key].view(-1)[0] += 0.125
    path.write_bytes(encode_qzs3_state_dict(state, block_size=32))
    return path


def _write_pose_safety_report(path: Path, *, source_sha: str, candidate_sha: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "renderer_transplant_pose_safety_preflight_v1",
                "score_claim": False,
                "promotion_eligible": False,
                "remote_gpu_dispatch_performed": False,
                "safe_for_exact_eval_dispatch": True,
                "failure_class": None,
                "fail_closed_reasons": [],
                "source_archive": {"sha256": source_sha},
                "candidate_archive": {"sha256": candidate_sha},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return path


def test_preflight_requires_trained_export_or_explicit_surrogate(tmp_path: Path) -> None:
    preflight = _load_module(PREFLIGHT_PATH, "_trained_preflight_required_test")
    source, _renderer = _write_packed_source(tmp_path)

    with pytest.raises(ValueError, match="--renderer-export is required"):
        preflight.build_preflight(
            source_archive=source,
            output_dir=tmp_path / "out",
            block_sizes=(64,),
        )


def test_preflight_source_surrogate_builds_byte_closed_non_dispatchable_json(
    tmp_path: Path,
) -> None:
    preflight = _load_module(PREFLIGHT_PATH, "_trained_preflight_surrogate_test")
    unpacker = _load_module(UNPACKER_PATH, "_trained_preflight_unpacker_test")
    source, _renderer = _write_packed_source(tmp_path)

    summary = preflight.build_preflight(
        source_archive=source,
        output_dir=tmp_path / "out",
        block_sizes=(64,),
        allow_source_renderer_surrogate=True,
    )

    assert summary["schema"] == "trained_renderer_blockfp_transplant_preflight_v1"
    assert summary["score_claim"] is False
    assert summary["promotion_eligible"] is False
    assert summary["remote_gpu_dispatch_performed"] is False
    assert summary["renderer_export"]["mode"] == "source_renderer_surrogate"
    assert summary["h100_lightning_readiness"]["ready"] is False
    assert summary["h100_lightning_readiness"]["next_commands_if_ready"] is None

    candidate = summary["candidates"][0]
    assert candidate["byte_closed"] is True
    assert candidate["runtime_compatible"] is True
    assert Path(candidate["archive_path"]).exists()
    assert Path(candidate["manifest_path"]).exists()
    assert (tmp_path / "out" / "trained_renderer_blockfp_preflight.json").exists()

    with zipfile.ZipFile(candidate["archive_path"]) as zf:
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        zf.extractall(extract_dir)
    unpack_summary = unpacker.unpack_renderer_payload(extract_dir)
    unpacked = {item["name"]: item for item in unpack_summary["members"]}
    manifest = json.loads(Path(candidate["manifest_path"]).read_text())
    assert unpacked["renderer.bin"]["sha256"] == manifest["transformed_renderer_payload"]["sha256"]
    assert manifest["runtime_contract"]["scorer_imports_at_inflate_time"] is False
    assert manifest["runtime_contract"]["score_affecting_payload_charged_in_archive"] is True


def test_preflight_changed_trained_export_blocks_h100_without_pose_safety(
    tmp_path: Path,
) -> None:
    preflight = _load_module(PREFLIGHT_PATH, "_trained_preflight_missing_pose_gate_test")
    source, _renderer = _write_packed_source(tmp_path)
    trained = _write_changed_renderer(tmp_path / "trained_renderer_qzs3.bin")

    summary = preflight.build_preflight(
        source_archive=source,
        renderer_export=trained,
        output_dir=tmp_path / "out",
        block_sizes=(64,),
    )

    assert summary["renderer_export"]["mode"] == "trained_renderer_export"
    assert summary["renderer_export"]["same_as_source_renderer"] is False
    assert summary["h100_lightning_readiness"]["ready"] is False
    assert summary["h100_lightning_readiness"]["next_commands_if_ready"] is None
    assert summary["best_by_archive_bytes"]["pose_safety_gate"]["status"] == "missing_pose_safety_report"


def test_preflight_changed_trained_export_emits_h100_command_shapes_after_pose_safety(
    tmp_path: Path,
) -> None:
    preflight = _load_module(PREFLIGHT_PATH, "_trained_preflight_ready_test")
    source, _renderer = _write_packed_source(tmp_path)
    trained = _write_changed_renderer(tmp_path / "trained_renderer_qzs3.bin")

    first = preflight.build_preflight(
        source_archive=source,
        renderer_export=trained,
        output_dir=tmp_path / "out1",
        block_sizes=(64,),
    )
    pose_safety = _write_pose_safety_report(
        tmp_path / "pose_safety.json",
        source_sha=first["source_archive"]["sha256"],
        candidate_sha=first["best_by_archive_bytes"]["archive_sha256"],
    )

    summary = preflight.build_preflight(
        source_archive=source,
        renderer_export=trained,
        output_dir=tmp_path / "out2",
        block_sizes=(64,),
        pose_safety_json=(pose_safety,),
    )

    assert summary["h100_lightning_readiness"]["ready"] is True
    assert summary["best_dispatchable_after_pose_safety"]["candidate_id"] == "trained_qbf1_b0064"
    commands = summary["h100_lightning_readiness"]["next_commands_if_ready"]
    assert commands["lane_id"] == "c067_trained_renderer_self_compression_blockfp"
    assert commands["remote_gpu_dispatch_performed"] is False
    assert "tools/claim_lane_dispatch.py" in commands["claim_command"]
    assert "--dry-run" in commands["lightning_exact_eval_dry_run_command"]
    assert "--machine" in commands["lightning_exact_eval_submit_command_shape"]
    machine_index = commands["lightning_exact_eval_submit_command_shape"].index("--machine")
    assert commands["lightning_exact_eval_submit_command_shape"][machine_index + 1] == "g7e.4xlarge"
    assert "--source-manifest" in commands["lightning_exact_eval_submit_command_shape"]
    assert "--remote-preflight-ssh-target" in commands["lightning_exact_eval_submit_command_shape"]


def test_preflight_is_deterministic_for_same_trained_export(tmp_path: Path) -> None:
    preflight = _load_module(PREFLIGHT_PATH, "_trained_preflight_determinism_test")
    source, _renderer = _write_packed_source(tmp_path)
    trained = _write_changed_renderer(tmp_path / "trained_renderer_qzs3.bin")

    first = preflight.build_preflight(
        source_archive=source,
        renderer_export=trained,
        output_dir=tmp_path / "out1",
        block_sizes=(64,),
    )
    second = preflight.build_preflight(
        source_archive=source,
        renderer_export=trained,
        output_dir=tmp_path / "out2",
        block_sizes=(64,),
    )

    first_candidate = first["candidates"][0]
    second_candidate = second["candidates"][0]
    assert Path(first_candidate["archive_path"]).read_bytes() == Path(
        second_candidate["archive_path"]
    ).read_bytes()
    assert first_candidate["archive_sha256"] == second_candidate["archive_sha256"]
