from __future__ import annotations

import csv
import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path

import brotli
import numpy as np

from tac.qp1_pose_codec import encode_qp1


REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "experiments" / "compare_public_pose_manifolds.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_public_pose_manifolds", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_zip(path: Path, payload: bytes) -> None:
    info = zipfile.ZipInfo("p", (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _qp1_from_words(words: list[int]) -> bytes:
    poses = np.zeros((len(words), 6), dtype=np.float32)
    poses[:, 0] = np.asarray(words, dtype=np.float32) / 512.0 + 20.0
    return encode_qp1(poses)


def _make_p6_archive(path: Path, *, words: list[int]) -> bytes:
    mask_raw = b"\x12\x00\x0a\x0a" + b"m" * 64
    renderer_raw = b"QZS3" + b"r" * 64
    actions_raw = b"\x02\x04\x06\x08"
    pose_raw = _qp1_from_words(words)
    mask_br = brotli.compress(mask_raw, quality=0)
    renderer_br = brotli.compress(renderer_raw, quality=0)
    actions_br = brotli.compress(actions_raw, quality=0)
    pose_br = brotli.compress(pose_raw, quality=0)
    payload = (
        b"P6"
        + struct.pack("<IHHH", len(mask_br), len(renderer_br), len(actions_br), 1)
        + mask_br
        + renderer_br
        + actions_br
        + pose_br
    )
    _write_zip(path, payload)
    return pose_raw


def _write_eval(path: Path, archive: Path) -> None:
    compare = _load_module()
    path.write_text(
        json.dumps(
            {
                "archive_size_bytes": archive.stat().st_size,
                "avg_posenet_dist": 0.00049337,
                "avg_segnet_dist": 0.00060804,
                "n_samples": 600,
                "score_recomputed_from_components": 0.31514430182167497,
                "provenance": {
                    "archive_sha256": compare._sha256_path(archive),
                    "device": "cuda",
                    "gpu_model": "unit",
                    "gpu_t4_match": True,
                },
            },
            sort_keys=True,
        )
    )


def _write_trace(path: Path) -> None:
    samples = []
    for pair in range(600):
        hot = pair in {105, 164, 211, 347}
        samples.append(
            {
                "pair_index": pair,
                "frame_indices": [2 * pair, 2 * pair + 1],
                "score_pose_contribution_first_order": 0.004 if hot else 0.0001,
                "score_seg_contribution_exact": 0.002 if hot else 0.00005,
                "score_combined_contribution_first_order": 0.006 if hot else 0.00015,
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "score_claim": False,
                "n_samples": 600,
                "contest_auth_eval_cross_check": {"all_match": True},
                "samples": samples,
            },
            sort_keys=True,
        )
    )


def test_qp1_decode_source_from_p6_archive(tmp_path: Path) -> None:
    compare = _load_module()
    archive = tmp_path / "source.zip"
    _make_p6_archive(archive, words=[11000, 11001, 11010, 10990, 11005, 11007])

    source = compare.load_pose_source(
        compare.PoseSourceSpec("C102", archive),
        expected_len=6,
    )

    assert source.available is True
    assert source.word_count == 6
    assert source.words.tolist() == [11000, 11001, 11010, 10990, 11005, 11007]
    assert source.pose_stream_bytes is not None and source.pose_stream_bytes > 5
    assert source.pose_stream_sha256


def test_compare_plan_is_deterministic_and_planning_only(tmp_path: Path) -> None:
    compare = _load_module()
    c102_archive = tmp_path / "c102.zip"
    pr75_archive = tmp_path / "pr75.zip"
    pr77_archive = tmp_path / "pr77.zip"
    base_words = [12000 + (i % 7) for i in range(600)]
    for hot in (105, 164, 211, 347):
        base_words[hot] += 30
    _make_p6_archive(c102_archive, words=base_words)
    pr75_pose = _make_p6_archive(pr75_archive, words=[w + (3 if i in {105, 164} else 0) for i, w in enumerate(base_words)])
    pr77_pose = _make_p6_archive(pr77_archive, words=[w - (4 if i in {211, 347} else 0) for i, w in enumerate(base_words)])
    pr75_pose_path = tmp_path / "pr75.qp1"
    pr77_pose_path = tmp_path / "pr77.qp1"
    pr75_pose_path.write_bytes(pr75_pose)
    pr77_pose_path.write_bytes(pr77_pose)
    eval_path = tmp_path / "contest_auth_eval.json"
    trace_path = tmp_path / "component_trace.json"
    _write_eval(eval_path, c102_archive)
    _write_trace(trace_path)
    output_dir = tmp_path / "out"

    plan1 = compare.build_public_pose_manifold_compare(
        c102_archive=c102_archive,
        c102_eval=eval_path,
        c102_trace=trace_path,
        pr75_archive=pr75_archive,
        pr75_decoded_pose=pr75_pose_path,
        pr77_archive=pr77_archive,
        pr77_decoded_pose=pr77_pose_path,
        pr65_archive=None,
        output_dir=output_dir,
        ledger_md=tmp_path / "ledger.md",
    )
    first_bytes = (output_dir / "pose_manifold_compare_plan.json").read_bytes()
    plan2 = compare.build_public_pose_manifold_compare(
        c102_archive=c102_archive,
        c102_eval=eval_path,
        c102_trace=trace_path,
        pr75_archive=pr75_archive,
        pr75_decoded_pose=pr75_pose_path,
        pr77_archive=pr77_archive,
        pr77_decoded_pose=pr77_pose_path,
        pr65_archive=None,
        output_dir=output_dir,
        ledger_md=tmp_path / "ledger.md",
    )

    assert first_bytes == (output_dir / "pose_manifold_compare_plan.json").read_bytes()
    assert plan1 == plan2
    assert plan1["score_claim"] is False
    assert plan1["archive_built"] is False
    assert plan1["remote_dispatch"]["dispatched"] is False
    assert plan1["ranked_candidates"]
    assert all(candidate["exact_eval_readiness"] is False for candidate in plan1["ranked_candidates"])
    top = plan1["ranked_candidates"][0]
    for field in (
        "source_archive_sha256",
        "source_archive_bytes",
        "pose_stream_sha256",
        "pose_stream_bytes",
        "basis_id",
        "selected_pairs",
        "selected_coefs",
        "expected_benefit_proxy",
        "risk_flags",
        "exact_eval_readiness",
    ):
        assert field in top
    with (output_dir / "candidate_rankings.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["exact_eval_readiness"] == "False"
