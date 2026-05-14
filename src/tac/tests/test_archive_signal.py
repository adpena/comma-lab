# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

from tac.archive_signal import RATE_LAMBDA, build_signal_table, render_markdown

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "build_replay_observability_signal.py"
PR79_S2_EXACT = (
    REPO_ROOT
    / "experiments/results/lightning_batch/exact_eval_pr79_s2_fixed_adaptive_actions_t4_20260503T173023Z/contest_auth_eval.json"
)
PR81_PROFILE = (
    REPO_ROOT
    / "experiments/results/public_pr81_qzs3_range_mask_intake_20260503_codex/pr81_qma9_semantic_range_mask_profile.json"
)
PR82_PROFILE = (
    REPO_ROOT
    / "experiments/results/public_pr82_henosis_frontier_intake_20260503_codex/pr82_henosis_frontier_static_profile.json"
)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_zip(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, payload in [
            ("masks.mkv", b"m" * 40),
            ("renderer.bin", b"r" * 20),
            ("optimized_poses.qp1", b"p" * 10),
        ]:
            info = zipfile.ZipInfo(name)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            zf.writestr(info, payload)
    return path


def _baseline(path: Path) -> Path:
    return _write_json(
        path,
        {
            "archive_size_bytes": 1000,
            "avg_posenet_dist": 0.001,
            "avg_segnet_dist": 0.002,
            "canonical_score": 0.25,
            "canonical_score_source": "score_recomputed_from_components",
            "n_samples": 2,
            "score_recomputed_from_components": 0.25,
        },
    )


def _pr81(path: Path) -> Path:
    return _write_json(
        path,
        {
            "archive": {"bytes": 700, "sha256": "pr81"},
            "evidence_grade": "external/planning_only",
            "score_claim": False,
            "payload_split": {
                "segments": [
                    {"name": "range_mask.qma9", "bytes": 400, "offset": 0, "codec": "qma9", "sha256": "m"},
                    {
                        "name": "split_model_reordered.br_bundle",
                        "bytes": 200,
                        "offset": 400,
                        "codec": "brotli_qzs3",
                        "sha256": "r",
                    },
                    {
                        "name": "optimized_poses.qp1.br",
                        "bytes": 60,
                        "offset": 600,
                        "codec": "brotli_qp1",
                        "sha256": "p",
                    },
                    {"name": "router_actions.3bit", "bytes": 40, "offset": 660, "codec": "packed", "sha256": "a"},
                ]
            },
            "component_byte_deltas": [
                {
                    "available": True,
                    "component": "mask_or_range_mask",
                    "delta_bytes_pr81_minus_reference": -300,
                    "pr81_bytes": 400,
                    "reference": "PR79_S2",
                    "reference_bytes": 700,
                }
            ],
        },
    )


def _pr82(path: Path) -> Path:
    return _write_json(
        path,
        {
            "evidence_grade": "empirical_static_archive_profile",
            "score_claim": False,
            "zip_container": {"archive_bytes": 1200, "archive_sha256": "pr82"},
            "compact_bundle": {
                "segments": [
                    {
                        "name": "mask",
                        "encoded_bytes": 500,
                        "encoded_offset": 24,
                        "encoded_sha256": "mask",
                        "brotli_decodable": True,
                        "decoded_bytes": 900,
                        "decoded_magic8_ascii": "..",
                    },
                    {
                        "name": "model",
                        "encoded_bytes": 400,
                        "encoded_offset": 524,
                        "encoded_sha256": "model",
                        "brotli_decodable": True,
                        "decoded_bytes": 1200,
                        "decoded_magic8_ascii": "QH0",
                    },
                ]
            },
            "anatomy": {
                "model_qh0": {
                    "top_records_by_bytes": [
                        {"name": "shared_trunk.fuse.pw", "record_bytes": 128, "quantization": "fp4", "shape": [8, 8]},
                    ]
                }
            },
        },
    )


def _trace(path: Path) -> Path:
    return _write_json(
        path,
        {
            "score_claim": False,
            "evidence_grade": "diagnostic_component_trace",
            "archive_size_bytes": 900,
            "avg_posenet_dist": 0.0012,
            "avg_segnet_dist": 0.0021,
            "score_recomputed_from_components": 0.26,
            "samples": [
                {
                    "pair_index": 0,
                    "video_name": "0.mkv",
                    "frame_indices": [0, 1],
                    "posenet_dist": 0.1,
                    "segnet_dist": 0.01,
                    "score_combined_contribution_first_order": 0.04,
                    "score_pose_contribution_first_order": 0.02,
                    "score_seg_contribution_exact": 0.02,
                },
                {
                    "pair_index": 1,
                    "video_name": "0.mkv",
                    "frame_indices": [2, 3],
                    "posenet_dist": 0.01,
                    "segnet_dist": 0.001,
                    "score_combined_contribution_first_order": 0.004,
                    "score_pose_contribution_first_order": 0.002,
                    "score_seg_contribution_exact": 0.002,
                },
            ],
        },
    )


def test_build_signal_table_ranks_static_and_component_signal(tmp_path: Path) -> None:
    table = build_signal_table(
        baseline_exact_json=_baseline(tmp_path / "baseline.json"),
        archive_paths=[_write_zip(tmp_path / "archive.zip")],
        pr81_profile_json=_pr81(tmp_path / "pr81.json"),
        pr82_profile_json=_pr82(tmp_path / "pr82.json"),
        component_trace_jsons=[_trace(tmp_path / "component_trace.json")],
        top_k=3,
    )

    assert table["score_claim"] is False
    assert table["dispatch_performed"] is False
    assert table["baseline"]["archive_bytes"] == 1000
    assert any(row["source_label"] == "PR81" and row["name"] == "range_mask.qma9" for row in table["stream_signal_rows"])
    pair = next(row for row in table["component_signal_rows"] if row["name"] == "pair_0000")
    assert pair["rate_equivalent_break_even_bytes"] == pytest.approx(0.04 / RATE_LAMBDA)
    assert table["ranked_signal_rows"][0]["priority_score"] >= table["ranked_signal_rows"][1]["priority_score"]
    markdown = render_markdown(table, top_k=4)
    assert "Replay Observability Signal" in markdown
    assert "exact CUDA auth eval" in markdown


def test_script_argparse_keeps_defaults_exposed() -> None:
    spec = importlib.util.spec_from_file_location("build_replay_observability_signal_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    args = module.build_arg_parser().parse_args(
        [
            "--baseline-exact-json",
            "baseline.json",
            "--component-trace-json",
            "trace.json",
            "--no-default-archives",
            "--no-default-component-traces",
            "--output-json",
            "out.json",
        ]
    )

    assert str(args.baseline_exact_json) == "baseline.json"
    assert [str(path) for path in args.component_trace_json] == ["trace.json"]
    assert args.no_default_archives is True
    assert args.no_default_component_traces is True
    assert str(args.output_json) == "out.json"


@pytest.mark.skipif(
    not (PR79_S2_EXACT.exists() and PR81_PROFILE.exists() and PR82_PROFILE.exists()),
    reason="stable local PR79/S2, PR81, or PR82 JSON artifacts are missing",
)
def test_actual_local_frontier_profiles_build_observability_table() -> None:
    table = build_signal_table(
        baseline_exact_json=PR79_S2_EXACT,
        pr81_profile_json=PR81_PROFILE,
        pr82_profile_json=PR82_PROFILE,
        component_trace_jsons=[],
        top_k=5,
    )

    assert table["baseline"]["score_recomputed_from_components"] == pytest.approx(0.31453355357318635)
    assert any(row["source_label"] == "PR81" for row in table["stream_signal_rows"])
    assert any(row["source_label"] == "PR82" for row in table["stream_signal_rows"])
    assert table["top_dispatch_guidance"]
