from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "build_protected_mask_reencode_candidate.py"
SPEC = importlib.util.spec_from_file_location("build_protected_mask_reencode_candidate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def _write_base_archive(path: Path, *, mask_bytes: bytes = b"source-mask-stream") -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("renderer.bin", b"renderer")
        zf.writestr("masks.mkv", mask_bytes)
        zf.writestr("optimized_poses.bin", b"poses")


def _read_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as zf:
        return {info.filename: zf.read(info) for info in zf.infolist()}


def _encode_fake_tensor(prefix: bytes, masks: torch.Tensor) -> bytes:
    shape = ",".join(str(int(value)) for value in masks.shape).encode("ascii")
    return prefix + b"\n" + shape + b"\n" + masks.to(torch.uint8).contiguous().numpy().tobytes()


def _decode_fake_tensor(payload: bytes) -> torch.Tensor:
    _prefix, shape_line, raw = payload.split(b"\n", 2)
    shape = tuple(int(part) for part in shape_line.decode("ascii").split(","))
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(shape).to(torch.int64)


def test_builds_deterministic_candidate_archive_with_adjacent_non_score_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = torch.tensor(
        [
            [[1, 0], [0, 0]],
            [[0, 0], [0, 2]],
        ],
        dtype=torch.int64,
    )
    lossy = torch.zeros_like(source)
    base = tmp_path / "base.zip"
    _write_base_archive(base)

    def fake_decode_source(data: bytes, member: str, *, max_frames: int | None) -> tuple[torch.Tensor, dict[str, Any]]:
        assert data == b"source-mask-stream"
        assert member == "masks.mkv"
        assert max_frames is None
        return source.clone(), {"decoder": "fake"}

    encoded_payload_to_decode: dict[bytes, torch.Tensor] = {}
    encode_calls = 0

    def fake_encode(
        masks: torch.Tensor,
        output_path: Path,
        *,
        crf: int,
        fps: int,
        svtav1_params: str,
    ) -> dict[str, Any]:
        nonlocal encode_calls
        encode_calls += 1
        decoded = lossy.clone() if encode_calls == 1 else masks.clone()
        payload = _encode_fake_tensor(f"fake-iter-{encode_calls}".encode("ascii"), decoded)
        encoded_payload_to_decode[payload] = decoded
        output_path.write_bytes(payload)
        return {
            "path": str(output_path),
            "bytes": payload,
            "command": ["fake-ffmpeg", "-crf", str(crf), "-r", str(fps), str(output_path)],
        }

    def fake_decode_candidate(data: bytes, member: str, *, expected_shape: tuple[int, int, int]) -> torch.Tensor:
        assert member == "masks.mkv"
        decoded = encoded_payload_to_decode[data]
        assert tuple(decoded.shape) == expected_shape
        return decoded.clone()

    monkeypatch.setattr(builder, "_decode_source_masks", fake_decode_source)
    monkeypatch.setattr(builder, "_encode_legacy_av1_masks", fake_encode)
    monkeypatch.setattr(builder, "_decode_candidate_masks", fake_decode_candidate)

    out_a = tmp_path / "a" / "archive.zip"
    manifest_a = tmp_path / "a" / "protected_mask_reencode_manifest.json"
    policy = builder.ProtectionPolicy(
        class_ids=(2,),
        foveal_boxes=(builder.RegionSpec(name="fovea", x0=0, y0=0, x1=1, y1=1, frames=(0,)),),
        label="test_policy",
    )
    report_a = builder.build_candidate(
        base_archive=base,
        output_archive=out_a,
        manifest_json=manifest_a,
        policy=policy,
        crf=56,
        fps=20,
        protection_iterations=1,
    )

    members_a = _read_members(out_a)
    assert list(members_a) == ["renderer.bin", "masks.mkv", "optimized_poses.bin"]
    assert members_a["renderer.bin"] == b"renderer"
    assert members_a["optimized_poses.bin"] == b"poses"
    final_masks = _decode_fake_tensor(members_a["masks.mkv"])
    assert int(final_masks[0, 0, 0]) == 1
    assert int(final_masks[1, 1, 1]) == 2
    assert int(final_masks[0, 0, 1]) == 0

    manifest_payload = json.loads(manifest_a.read_text())
    assert manifest_payload["score_claim"] is False
    assert manifest_payload["promotion_eligible"] is False
    assert manifest_payload["archive"]["sha256"] == report_a["archive"]["sha256"]
    assert manifest_payload["source_mask_stream"]["sha256"] == builder._sha256_bytes(b"source-mask-stream")
    assert manifest_payload["policy"]["label"] == "test_policy"
    assert manifest_payload["protection_summary"]["protected_pixels"] == 2
    assert len(manifest_payload["ffmpeg"]["encode_steps"]) == 2
    assert all("<work_dir>/" in step["ffmpeg_args"][-1] for step in manifest_payload["ffmpeg"]["encode_steps"])

    encode_calls = 0
    encoded_payload_to_decode.clear()
    out_b = tmp_path / "b" / "archive.zip"
    builder.build_candidate(
        base_archive=base,
        output_archive=out_b,
        manifest_json=tmp_path / "b" / "manifest.json",
        policy=policy,
        crf=56,
        fps=20,
        protection_iterations=1,
    )
    assert out_a.read_bytes() == out_b.read_bytes()


def test_protection_mask_supports_boundaries_regions_classes_and_pair_frames() -> None:
    source = torch.tensor(
        [
            [[0, 0, 1, 1], [0, 2, 2, 1], [3, 3, 2, 4]],
            [[0, 1, 1, 1], [0, 2, 2, 4], [3, 3, 4, 4]],
            [[4, 4, 4, 4], [0, 0, 1, 1], [2, 2, 3, 3]],
        ],
        dtype=torch.int64,
    )
    policy = builder.ProtectionPolicy(
        hard_pair_indices=(0,),
        class_ids=(2,),
        boundary_dilation=1,
        horizon_bands=(builder.RegionSpec(name="horizon", x0=0, y0=0, x1=-1, y1=1),),
        ego_boxes=(builder.RegionSpec(name="ego", x0=1, y0=1, x1=3, y1=3, frames=(0,)),),
    )

    mask, summary = builder.build_protection_mask(source, policy)

    assert mask.shape == source.shape
    assert bool(mask[0, 1, 1]) is True
    assert bool(mask[0, 0, 3]) is True
    assert bool(mask[0, 2, 2]) is True
    assert bool(mask[0].all()) is True
    assert bool(mask[1].all()) is True
    assert summary["frames_with_any_protection"] == 3
    assert summary["per_rule_pixel_counts_before_overlap"]["hard_frames"] == 24
    assert summary["per_rule_pixel_counts_before_overlap"]["class_ids"] == 7


def test_pair_frame_mode_supports_half_frame_mask_streams() -> None:
    source = torch.zeros((5, 2, 2), dtype=torch.int64)

    full_policy = builder.ProtectionPolicy(hard_pair_indices=(1,))
    full_mask, full_summary = builder.build_protection_mask(source, full_policy)
    assert bool(full_mask[2].all()) is True
    assert bool(full_mask[3].all()) is True
    assert full_summary["hard_pair_frame_mode_resolved"] == "full_frames"
    assert full_summary["expanded_hard_pair_frames"] == [2, 3]

    half_policy = builder.ProtectionPolicy(
        hard_pair_indices=(4,),
        hard_pair_frame_mode="half_frame_masks",
    )
    half_mask, half_summary = builder.build_protection_mask(source, half_policy)
    assert bool(half_mask[4].all()) is True
    assert half_summary["hard_pair_frame_mode_resolved"] == "half_frame_masks"
    assert half_summary["expanded_hard_pair_frames"] == [4]

    with pytest.raises(ValueError, match="half-frame mask stream"):
        builder.build_protection_mask(source, builder.ProtectionPolicy(hard_pair_indices=(4,)))


def test_rejects_hidden_base_archive_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "bad.zip"
    with zipfile.ZipFile(base, "w") as zf:
        zf.writestr("renderer.bin", b"renderer")
        zf.writestr("masks.mkv", b"mask")
        zf.writestr("._masks.mkv", b"sidecar")

    with pytest.raises(ValueError, match="hidden/system archive member"):
        builder.build_candidate(
            base_archive=base,
            output_archive=tmp_path / "archive.zip",
            manifest_json=tmp_path / "manifest.json",
            policy=builder.ProtectionPolicy(),
        )
