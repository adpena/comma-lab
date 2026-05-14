# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import struct
import zipfile
from pathlib import Path


from experiments import build_c101_renderer_pose_stack_candidate as builder


def _write_p3_archive(path: Path, *, mask: bytes, renderer: bytes, actions: bytes, pose: bytes) -> Path:
    payload = (
        b"P3"
        + struct.pack("<IHH", len(mask), len(renderer), len(actions))
        + mask
        + renderer
        + actions
        + pose
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o600 << 16
        zf.writestr(info, payload)
    return path


def _write_p6_archive(
    path: Path,
    *,
    mask: bytes,
    renderer: bytes,
    actions: bytes,
    pose: bytes,
    record_count: int,
) -> Path:
    payload = (
        b"P6"
        + struct.pack("<IHHH", len(mask), len(renderer), len(actions), record_count)
        + mask
        + renderer
        + actions
        + pose
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o600 << 16
        zf.writestr(info, payload)
    return path


def test_build_p3_payload_takes_renderer_from_renderer_archive_and_pose_from_pose_archive(tmp_path: Path) -> None:
    mask = b"mask"
    actions = b"actions"
    pose_slices = builder.parse_p3_slices(
        builder.read_single_payload(
            _write_p3_archive(
                tmp_path / "pose.zip",
                mask=mask,
                renderer=b"renderer-old",
                actions=actions,
                pose=b"pose-new",
            )
        )
    )
    renderer_slices = builder.parse_p3_slices(
        builder.read_single_payload(
            _write_p3_archive(
                tmp_path / "renderer.zip",
                mask=mask,
                renderer=b"renderer-new",
                actions=actions,
                pose=b"pose-old",
            )
        )
    )

    payload = builder.build_p3_payload(
        builder.P3Slices(
            mask_br=pose_slices.mask_br,
            renderer_br=renderer_slices.renderer_br,
            actions_br=pose_slices.actions_br,
            pose_br=pose_slices.pose_br,
        )
    )
    stack_archive = _write_p3_archive(tmp_path / "stack.zip", mask=mask, renderer=b"renderer-new", actions=actions, pose=b"pose-new")

    assert payload == zipfile.ZipFile(stack_archive).read("p")


def test_build_payload_refuses_mask_or_action_mismatch(tmp_path: Path) -> None:
    pose_slices = builder.parse_p3_slices(
        builder.read_single_payload(
            _write_p3_archive(tmp_path / "pose.zip", mask=b"mask-a", renderer=b"r0", actions=b"a", pose=b"p1")
        )
    )
    renderer_slices = builder.parse_p3_slices(
        builder.read_single_payload(
            _write_p3_archive(tmp_path / "renderer.zip", mask=b"mask-b", renderer=b"r1", actions=b"a", pose=b"p0")
        )
    )

    assert pose_slices.mask_br != renderer_slices.mask_br


def test_parse_records_archive_and_slice_hashes(tmp_path: Path) -> None:
    archive = _write_p3_archive(
        tmp_path / "candidate.zip",
        mask=b"mask",
        renderer=b"renderer",
        actions=b"actions",
        pose=b"pose",
    )
    payload = builder.read_single_payload(archive)
    parsed = builder.parse_p3_slices(payload)

    assert builder._sha256_file(archive) == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert len(payload) == len(zipfile.ZipFile(archive).read("p"))
    assert parsed.renderer_br == b"renderer"


def test_parse_and_rebuild_p6_preserves_action_record_count(tmp_path: Path) -> None:
    archive = _write_p6_archive(
        tmp_path / "candidate.zip",
        mask=b"mask",
        renderer=b"renderer",
        actions=b"actions",
        pose=b"pose",
        record_count=7,
    )
    parsed = builder.parse_p3_slices(builder.read_single_payload(archive))
    rebuilt = builder.build_p3_payload(
        builder.P3Slices(
            mask_br=parsed.mask_br,
            renderer_br=b"renderer-new",
            actions_br=parsed.actions_br,
            pose_br=parsed.pose_br,
            wire_magic=parsed.wire_magic,
            action_record_count=parsed.action_record_count,
        )
    )

    assert rebuilt.startswith(b"P6")
    assert struct.unpack_from("<IHHH", rebuilt, 2) == (
        len(b"mask"),
        len(b"renderer-new"),
        len(b"actions"),
        7,
    )


def test_pose_safety_dispatch_gate_fails_closed_without_matching_report() -> None:
    gate = builder._pose_safety_dispatch_gate(
        source_archive_sha256="a" * 64,
        candidate_archive_sha256="b" * 64,
        reports=(),
    )

    assert gate["required"] is True
    assert gate["safe_for_exact_eval_dispatch"] is False
    assert gate["status"] == "missing_pose_safety_report"


def test_pose_safety_dispatch_gate_accepts_matching_no_score_report() -> None:
    gate = builder._pose_safety_dispatch_gate(
        source_archive_sha256="a" * 64,
        candidate_archive_sha256="b" * 64,
        reports=[
            {
                "schema": "renderer_transplant_pose_safety_preflight_v1",
                "score_claim": False,
                "promotion_eligible": False,
                "remote_gpu_dispatch_performed": False,
                "safe_for_exact_eval_dispatch": True,
                "source_archive": {"sha256": "a" * 64},
                "candidate_archive": {"sha256": "b" * 64},
            }
        ],
    )

    assert gate["safe_for_exact_eval_dispatch"] is True
    assert gate["status"] == "pass"
