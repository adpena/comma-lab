# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

import brotli
import numpy as np

from tac.qp1_pose_codec import QPV1DimensionStream, QPV1Payload, encode_qp1, parse_qpv1

import experiments.build_qp1_pose_active_subspace_candidates as builder


def _write_zip(path: Path, payload: bytes) -> None:
    info = zipfile.ZipInfo("p", builder.FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _make_pose_words(words: list[int]) -> bytes:
    poses = np.zeros((len(words), 6), dtype=np.float32)
    poses[:, 0] = np.asarray(words, dtype=np.float32) / builder.VELOCITY_SCALE + builder.VELOCITY_OFFSET
    return encode_qp1(poses)


def _make_p6_archive(path: Path, *, words: list[int]) -> dict[str, bytes]:
    mask_raw = b"\x12\x00\x0a\x0a" + b"m" * 64
    renderer_raw = b"QZS3" + b"r" * 48
    actions_raw = b"\x05\x07\x09" + b"\x02\x04\x06"
    pose_raw = _make_pose_words(words)
    mask_br = brotli.compress(mask_raw, quality=0)
    renderer_br = brotli.compress(renderer_raw, quality=0)
    actions_br = brotli.compress(actions_raw, quality=0)
    pose_br = brotli.compress(pose_raw, quality=0)
    payload = (
        b"P6"
        + struct.pack("<IHHH", len(mask_br), len(renderer_br), len(actions_br), 2)
        + mask_br
        + renderer_br
        + actions_br
        + pose_br
    )
    _write_zip(path, payload)
    return {
        "mask_br": mask_br,
        "renderer_br": renderer_br,
        "actions_br": actions_br,
        "pose_raw": pose_raw,
    }


def _make_qp19_archive(path: Path) -> dict[str, bytes]:
    mask_raw = b"\x12\x00\x0a\x0a" + b"m" * 64
    renderer_raw = b"QZS3" + b"r" * 48
    qpv1 = QPV1Payload(
        count=6,
        pose_dim=6,
        streams=(
            QPV1DimensionStream(0, 20.0, 512.0, (11000, 11010, 10900, 11020, 11030, 11040)),
            QPV1DimensionStream(2, -1.0, 2048.0, (0, 8, -30, 12, 16, 20)),
        ),
    )
    pose_raw = qpv1.to_bytes()
    mask_br = brotli.compress(mask_raw, quality=0)
    renderer_br = brotli.compress(renderer_raw, quality=0)
    pose_br = brotli.compress(pose_raw, quality=0)
    payload = (
        b"QP19"
        + bytes([1, 0])
        + struct.pack("<III", len(mask_br), len(renderer_br), len(pose_br))
        + mask_br
        + renderer_br
        + pose_br
    )
    _write_zip(path, payload)
    return {
        "mask_br": mask_br,
        "renderer_br": renderer_br,
        "pose_raw": pose_raw,
    }


def _write_trace(path: Path, n: int) -> None:
    samples = []
    for i in range(n):
        samples.append({
            "pair_index": i,
            "score_pose_contribution_first_order": float(n - i) / 1000.0,
            "score_seg_contribution_exact": float(i) / 10000.0,
            "score_combined_contribution_first_order": float(n - i) / 900.0,
        })
    path.write_text(json.dumps({
        "score_recomputed_from_components": 0.315,
        "samples": samples,
    }))


def test_builder_rewrites_only_qp1_pose_slice_in_p6_archive(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    source_slices = _make_p6_archive(source, words=[11000, 11010, 10920, 11020, 11030, 11040])
    trace = tmp_path / "trace.json"
    _write_trace(trace, 6)
    summary = builder.build_candidates(
        source_archive=source,
        component_trace=trace,
        output_dir=tmp_path / "out",
        reference_base_archive=None,
        reference_active_archive=None,
        specs=[
            builder.CandidateSpec(
                "neighbor_pose_top3",
                "neighbor_pull",
                "pose",
                3,
                0.5,
                4,
                0.03,
                1.5,
                5.0,
            )
        ],
        force=True,
    )

    assert summary["candidate_count"] == 1
    manifest = json.loads((tmp_path / "out" / "neighbor_pose_top3" / "manifest.json").read_text())
    assert manifest["score_claim"] is False
    assert manifest["candidate"]["no_sidecars"] is True
    assert manifest["local_roundtrip_gates"]["all_passed"] is True
    assert manifest["exact_eval_command_template"][:4] == [
        ".venv/bin/python",
        "experiments/contest_auth_eval.py",
        "--archive",
        manifest["candidate"]["archive_path"],
    ]
    candidate_parts = builder.load_archive_parts(Path(manifest["candidate"]["archive_path"]))
    assert candidate_parts.mask_br == source_slices["mask_br"]
    assert candidate_parts.renderer_br == source_slices["renderer_br"]
    assert candidate_parts.actions_br == source_slices["actions_br"]
    assert brotli.decompress(candidate_parts.pose_br) != source_slices["pose_raw"]
    assert manifest["candidate"]["changed_pair_count"] > 0
    assert manifest["candidate"]["pose_float32_semantic_sha256"]


def test_reference_active_delta_candidate_uses_hard_pair_basis(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    _make_p6_archive(source, words=[12000, 12001, 12002, 12003, 12004, 12005])
    ref_base = tmp_path / "ref_base.zip"
    ref_active = tmp_path / "ref_active.zip"
    _make_p6_archive(ref_base, words=[12000, 12001, 12002, 12003, 12004, 12005])
    _make_p6_archive(ref_active, words=[12000, 12009, 12002, 12011, 12004, 12005])
    trace = tmp_path / "trace.json"
    _write_trace(trace, 6)

    summary = builder.build_candidates(
        source_archive=source,
        component_trace=trace,
        output_dir=tmp_path / "out",
        reference_base_archive=ref_base,
        reference_active_archive=ref_active,
        specs=[
            builder.CandidateSpec(
                "ref_pose_top4",
                "reference_delta",
                "pose",
                4,
                0.5,
                8,
                0.08,
                1.2,
                6.0,
            )
        ],
        force=True,
    )

    manifest = json.loads((tmp_path / "out" / "ref_pose_top4" / "manifest.json").read_text())
    changed_pairs = {item["pair_index"]: item for item in manifest["candidate"]["changed_pairs"]}
    assert set(changed_pairs) == {1, 3}
    assert changed_pairs[1]["basis"] == "reference_active_subspace_delta"
    assert changed_pairs[1]["delta_q"] == 4
    assert summary["reference_delta"]["usable"] is True
    assert summary["top_candidates"][0]["local_roundtrip_gates_passed"] is True


def test_builder_plans_qp19_qpv1_multidim_pose_candidates(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    source_slices = _make_qp19_archive(source)
    trace = tmp_path / "trace.json"
    _write_trace(trace, 6)

    summary = builder.build_candidates(
        source_archive=source,
        component_trace=trace,
        output_dir=tmp_path / "out",
        reference_base_archive=None,
        reference_active_archive=None,
        specs=[
            builder.CandidateSpec(
                "qpv1_neighbor_pose_top2",
                "neighbor_pull",
                "pose",
                2,
                0.5,
                16,
                0.03,
                1.5,
                5.0,
            )
        ],
        force=True,
    )

    manifest = json.loads((tmp_path / "out" / "qpv1_neighbor_pose_top2" / "manifest.json").read_text())
    assert summary["candidate_count"] == 1
    assert manifest["source"]["pose_codec"] == "QPV1"
    assert manifest["candidate"]["pose_codec"] == "QPV1"
    assert manifest["local_roundtrip_gates"]["all_passed"] is True
    assert {change["dim"] for change in manifest["candidate"]["changed_pairs"]} == {0, 2}
    candidate_parts = builder.load_archive_parts(Path(manifest["candidate"]["archive_path"]))
    assert candidate_parts.payload_format == "public_pr77_qp19_qzs3_pose_v1"
    assert candidate_parts.mask_br == source_slices["mask_br"]
    assert candidate_parts.renderer_br == source_slices["renderer_br"]
    roundtrip = parse_qpv1(brotli.decompress(candidate_parts.pose_br))
    assert roundtrip.count == 6
    assert [stream.dim for stream in roundtrip.streams] == [0, 2]
