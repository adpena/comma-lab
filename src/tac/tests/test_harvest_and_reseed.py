# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_tool():
    path = REPO_ROOT / "tools" / "harvest_and_reseed.py"
    spec = importlib.util.spec_from_file_location("harvest_and_reseed_tool_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_cuda_auth_eval(path: Path, *, score: float = 0.20664588545741508) -> None:
    path.write_text(
        json.dumps(
            {
                "score_axis": "contest_cuda",
                "canonical_score": score,
                "avg_segnet_dist": 0.00064260,
                "avg_posenet_dist": 0.00003236,
                "archive_size_bytes": 186822,
                "evidence_grade": "contest-CUDA",
                "lane_tag": "[contest-CUDA]",
                "score_claim_valid": True,
                "promotion_eligible": True,
                "provenance": {
                    "archive_sha256": "c" * 64,
                    "archive_size_bytes": 186822,
                    "device": "cuda",
                    "hardware": "Modal Tesla T4 Linux x86_64",
                    "platform_system": "Linux",
                    "platform_machine": "x86_64",
                    "gpu_model": "Tesla T4",
                    "gpu_t4_match": True,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_cpu_auth_eval(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "score_axis": "contest_cpu",
                "canonical_score": 0.22809238271134513,
                "avg_segnet_dist": 0.00063196,
                "avg_posenet_dist": 0.00016402,
                "archive_size_bytes": 186822,
                "evidence_grade": "contest-CPU",
                "lane_tag": "[contest-CPU]",
                "provenance": {
                    "archive_sha256": "d" * 64,
                    "archive_size_bytes": 186822,
                    "device": "cpu",
                    "hardware": "Modal CPU Linux x86_64",
                    "platform_system": "Linux",
                    "platform_machine": "x86_64",
                },
            }
        ),
        encoding="utf-8",
    )


def test_harvest_reseed_accepts_auth_eval_custody_without_jsonl_tag(tmp_path: Path) -> None:
    tool = _load_tool()
    tool.REPO = tmp_path
    score_json = tmp_path / "contest_auth_eval.json"
    _write_cuda_auth_eval(score_json)
    meta = tmp_path / "parallel_sweep_fixture_candidate" / "repack_metadata.json"
    meta.parent.mkdir()
    meta.write_text(
        json.dumps(
            {
                "rel_err_pct_per_weight": 0.125,
                "n_intn_layers": 13,
                "archive_size_bytes": 186822,
            }
        ),
        encoding="utf-8",
    )
    harvested = tmp_path / "harvested.jsonl"
    harvested.write_text(
        json.dumps(
            {
                "label": "fixture_candidate",
                "score_json_path": str(score_json),
                "contest_cuda_score": None,
                "started_utc": "2026-05-11T00:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    anchors = tmp_path / "anchors.json"

    rc = tool.main(
        [
            "--harvested-jsonl",
            str(harvested),
            "--anchors-path",
            str(anchors),
            "--rel-err-source-meta-glob",
            "parallel_sweep_*/repack_metadata.json",
        ]
    )

    assert rc == 0
    payload = json.loads(anchors.read_text(encoding="utf-8"))
    assert len(payload) == 1
    anchor = payload[0]
    assert anchor["contest_cuda_score"] == 0.20664588545741508
    assert anchor["archive_sha256"] == "c" * 64
    assert anchor["hardware_substrate"] == "linux_x86_64_t4"
    assert anchor["avg_pose_dist"] == 0.00003236
    assert anchor["avg_seg_dist"] == 0.00064260


def test_harvest_reseed_rejects_self_tagged_cpu_auth_eval(tmp_path: Path) -> None:
    tool = _load_tool()
    tool.REPO = tmp_path
    score_json = tmp_path / "contest_auth_eval.json"
    _write_cpu_auth_eval(score_json)
    harvested = tmp_path / "harvested.jsonl"
    harvested.write_text(
        json.dumps(
            {
                "label": "cpu_disguised_as_cuda",
                "tag": "[contest-CUDA]",
                "score_json_path": str(score_json),
                "contest_cuda_score": 0.22809238271134513,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    anchors = tmp_path / "anchors.json"

    rc = tool.main(
        [
            "--harvested-jsonl",
            str(harvested),
            "--anchors-path",
            str(anchors),
            "--rel-err-source-meta-glob",
            "parallel_sweep_*/repack_metadata.json",
        ]
    )

    assert rc == 0
    assert not anchors.exists()
