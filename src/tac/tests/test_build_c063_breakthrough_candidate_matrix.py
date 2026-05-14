# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path

import brotli


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "build_c063_breakthrough_candidate_matrix.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_c063_breakthrough_candidate_matrix_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_archive(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("p", b"frontier-payload")
    return path


def _c063_like_outer_brotli_archive(path: Path) -> Path:
    mask_raw = b"\x12\x00\n\n" + b"m" * 37
    renderer_raw = b"QZS3" + b"r" * 29
    pose_raw = b"QP1\x00" + b"p" * 19
    raw = (
        struct.pack("<III", len(mask_raw), len(renderer_raw), len(pose_raw))
        + mask_raw
        + renderer_raw
        + pose_raw
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("p", brotli.compress(raw, quality=11))
    return path


def _line_search_metadata(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "payload_layout": {
                    "mask_bytes": 8,
                    "model_bytes": 6,
                    "pose_bytes": 2,
                },
                "score_claim": False,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return path


def _frontier_eval(path: Path, archive: Path) -> Path:
    payload = {
        "archive_size_bytes": archive.stat().st_size,
        "avg_posenet_dist": 0.00049637,
        "avg_segnet_dist": 0.00061244,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": _sha(archive),
            "device": "cuda",
            "gpu_model": "Tesla T4",
            "gpu_t4_match": True,
            "sys_argv": ["experiments/contest_auth_eval.py", "--device", "cuda"],
        },
        "score_recomputed_from_components": 0.3156230307844823,
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _component_trace(path: Path) -> Path:
    samples = [
        {
            "pair_index": idx,
            "posenet_dist": 0.01 if idx in {5, 9, 20} else 0.0001,
            "segnet_dist": 0.002 if idx in {5, 9, 20} else 0.0001,
        }
        for idx in range(600)
    ]
    payload = {
        "contest_auth_eval_cross_check": {"all_match": True},
        "n_samples": 600,
        "samples": samples,
        "score_claim": False,
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _pose_plan(path: Path) -> Path:
    payload = {
        "recommended_policies": [
            {
                "charged_bytes_estimate": 64.0,
                "expected_score_saved_sum": 0.0004,
                "policy_name": "fixture_top032",
                "selected_pair_indices": list(range(32)),
            },
            {
                "charged_bytes_estimate": 96.0,
                "expected_score_saved_sum": 0.0008,
                "policy_name": "fixture_top048",
                "selected_pair_indices": list(range(48)),
            },
        ],
        "score_claim": False,
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _pr65_archive(path: Path) -> Path:
    core = [1001, 1002, 101]
    qpost_lengths = [1, 2, 3, 4, 5, 6, 7]
    header = b"".join(int(n).to_bytes(3, "little") for n in [*core, *qpost_lengths])
    payload = (
        header
        + b"m" * core[0]
        + b"r" * core[1]
        + b"p" * core[2]
        + b"A"
        + b"B" * 2
        + b"C" * 3
        + b"D" * 4
        + b"E" * 5
        + b"F" * 6
        + b"G" * 7
        + b"R"
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("x", payload)
    return path


def test_plan_only_ranks_pose_waterfill_before_archive_screens(tmp_path: Path) -> None:
    module = _load_module()
    archive = _source_archive(tmp_path / "archive.zip")
    payload = module.build_breakthrough_matrix(
        source_archive=archive,
        frontier_eval=_frontier_eval(tmp_path / "frontier.json", archive),
        frontier_component_trace=_component_trace(tmp_path / "trace.json"),
        pose_plan_path=_pose_plan(tmp_path / "pose_plan.json"),
        line_search_source_archive=archive,
        line_search_source_metadata=_line_search_metadata(tmp_path / "metadata.json"),
        pr65_archive=tmp_path / "missing_pr65.zip",
        pr67_archive=tmp_path / "missing_pr67.zip",
        output_dir=tmp_path / "out",
        mqz_policies=(),
        qpost_specs=(),
        qpost_evidence_paths=(),
        build_archive_candidates=False,
    )

    recommended = payload["recommended_first_h100_candidate"]
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert recommended["candidate_id"] == "c063_pose_waterfill_048_pr67_pr65_basis"
    assert recommended["expected_net_score_delta_vs_c063"] < 0
    assert recommended["requires_dispatch_claim_before_remote_job"] is True
    assert recommended["line_search_source_archive"] == str(archive.resolve())
    assert recommended["line_search_source_metadata"].endswith("metadata.json")
    assert "--metadata-path" in recommended["build_command_no_dispatch_claim_included"]
    assert "pr67_public_metadata_for_pose_search" not in recommended["build_command_no_dispatch_claim_included"]
    assert "--basis-pair-indices" in recommended["build_command_no_dispatch_claim_included"]
    assert (tmp_path / "out" / "c063_breakthrough_candidate_matrix.json").is_file()
    assert (tmp_path / "out" / "exact_eval_recommendation.json").is_file()


def test_qpost_wrapper_builds_closed_archive_and_keeps_score_claim_false(tmp_path: Path) -> None:
    module = _load_module()
    archive = _source_archive(tmp_path / "archive.zip")
    payload = module.build_breakthrough_matrix(
        source_archive=archive,
        frontier_eval=_frontier_eval(tmp_path / "frontier.json", archive),
        frontier_component_trace=_component_trace(tmp_path / "trace.json"),
        pose_plan_path=_pose_plan(tmp_path / "pose_plan.json"),
        line_search_source_archive=archive,
        line_search_source_metadata=_line_search_metadata(tmp_path / "metadata.json"),
        pr65_archive=_pr65_archive(tmp_path / "pr65.zip"),
        pr67_archive=tmp_path / "missing_pr67.zip",
        output_dir=tmp_path / "out",
        mqz_policies=(),
        qpost_specs=(module.QPostSpec("fixture_qpost_bias", ("bias",), None),),
        qpost_evidence_paths=(),
        build_archive_candidates=True,
    )

    candidate = payload["archive_candidates"][0]
    out_archive = Path(candidate["output_archive"])
    assert candidate["score_claim"] is False
    assert candidate["promotion_eligible"] is False
    assert candidate["family"] == "archive_side_pr65_qpost_atoms"
    assert candidate["output_archive_sha256"] == _sha(out_archive)
    assert candidate["minimum_component_score_gain_to_beat_c063"] >= 0.0
    with zipfile.ZipFile(out_archive) as zf:
        assert zf.namelist() == ["p", "qpost.bin"]


def test_default_line_search_source_derives_c063_fixedslice(tmp_path: Path) -> None:
    module = _load_module()
    archive = _c063_like_outer_brotli_archive(tmp_path / "c063.zip")
    payload = module.build_breakthrough_matrix(
        source_archive=archive,
        frontier_eval=_frontier_eval(tmp_path / "frontier.json", archive),
        frontier_component_trace=_component_trace(tmp_path / "trace.json"),
        pose_plan_path=_pose_plan(tmp_path / "pose_plan.json"),
        pr65_archive=tmp_path / "missing_pr65.zip",
        pr67_archive=tmp_path / "missing_pr67.zip",
        output_dir=tmp_path / "out",
        mqz_policies=(),
        qpost_specs=(),
        qpost_evidence_paths=(),
        build_archive_candidates=False,
    )

    recommendation = payload["recommended_first_h100_candidate"]
    fixed_archive = Path(recommendation["line_search_source_archive"])
    fixed_metadata = Path(recommendation["line_search_source_metadata"])
    assert fixed_archive.is_file()
    assert fixed_metadata.is_file()
    assert fixed_archive.parent.name == "line_search_source_c063_fixedslice"
    assert recommendation["source_archive"] == str(archive.resolve())
    assert recommendation["line_search_source_archive"] != recommendation["source_archive"]
    assert "c063_frontier_pr64_len_table_to_pr67_fixedslice" in fixed_metadata.read_text()


def test_default_line_search_source_reuses_public_pr67_fixedslice(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    mask_br = b"mask-br"
    model_br = b"model-br"
    pose_br = b"pose-br"
    archive = tmp_path / "c067.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("p", mask_br + model_br + pose_br)

    class FakeUnpacker:
        @staticmethod
        def _parse_payload(payload: bytes):
            assert payload == mask_br + model_br + pose_br
            return (
                {
                    "payload_format": "public_pr67_qzs3_qp1_fixed_slices",
                    "members": [
                        {"name": "masks.mkv", "bytes": len(mask_br)},
                        {"name": "renderer.bin", "bytes": len(model_br)},
                        {"name": "optimized_poses.bin", "bytes": len(pose_br)},
                    ],
                },
                {
                    "masks.mkv": b"\x12\x00mask",
                    "renderer.bin": b"QZS3renderer",
                    "optimized_poses.bin": b"\x00" * 12,
                },
            )

    monkeypatch.setattr(module, "_load_unpacker_module", lambda: FakeUnpacker)
    payload = module.build_breakthrough_matrix(
        source_archive=archive,
        frontier_eval=_frontier_eval(tmp_path / "frontier.json", archive),
        frontier_component_trace=_component_trace(tmp_path / "trace.json"),
        pose_plan_path=_pose_plan(tmp_path / "pose_plan.json"),
        pr65_archive=tmp_path / "missing_pr65.zip",
        pr67_archive=tmp_path / "missing_pr67.zip",
        output_dir=tmp_path / "out",
        mqz_policies=(),
        qpost_specs=(),
        qpost_evidence_paths=(),
        build_archive_candidates=False,
    )

    recommendation = payload["recommended_first_h100_candidate"]
    fixed_archive = Path(recommendation["line_search_source_archive"])
    fixed_metadata = Path(recommendation["line_search_source_metadata"])
    with zipfile.ZipFile(fixed_archive) as zf:
        assert zf.namelist() == ["p"]
        assert zf.read("p") == mask_br + model_br + pose_br
    assert fixed_archive.parent.name == "line_search_source_c067_fixedslice"
    metadata = json.loads(fixed_metadata.read_text())
    assert metadata["payload_format"] == "frontier_public_pr67_fixedslice_reused_for_line_search"
    assert metadata["mask_br_bytes"] == len(mask_br)
    assert metadata["model_br_bytes"] == len(model_br)
    assert metadata["pose_br_bytes"] == len(pose_br)
