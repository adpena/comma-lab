# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_tool():
    repo_root = Path(__file__).resolve().parents[3]
    tool_path = repo_root / "tools" / "analyze_device_axis_eval_matrix.py"
    spec = importlib.util.spec_from_file_location("analyze_device_axis_eval_matrix", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {tool_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _payload(
    *,
    device: str,
    scorer_device: str | None = None,
    score_axis: str,
    score: float,
    pose: float,
    seg: float,
    raw_sha: str,
    runtime_sha: str = "b" * 64,
    inflate_device_policy: str = "auto",
    score_claim: bool = False,
) -> dict:
    return {
        "score_recomputed_from_components": score,
        "avg_posenet_dist": pose,
        "avg_segnet_dist": seg,
        "archive_sha256": "a" * 64,
        "archive_size_bytes": 185_578,
        "runtime_tree_sha256": runtime_sha,
        "device": device,
        "scorer_device": scorer_device or device,
        "score_axis": score_axis,
        "n_samples": 600,
        "score_claim": score_claim,
        "score_claim_valid": score_claim,
        "promotion_eligible": False,
        "evidence_grade": "B" if score_axis.startswith("diagnostic") else "contest-CUDA",
        "provenance": {
            "device": device,
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "gpu_t4_match": device == "cuda",
            "inflate_device_policy": inflate_device_policy,
            "inflated_output_manifest": {
                "payload": {
                    "aggregate_sha256": raw_sha,
                    "raw_file_count": 1,
                    "total_bytes": 603_979_776,
                }
            },
        },
    }


def test_analyze_device_axis_matrix_preserves_raw_and_runtime_groups(tmp_path: Path) -> None:
    mod = _load_tool()
    cuda_auto = tmp_path / "cuda_auto.json"
    cuda_cpu_inflate = tmp_path / "cuda_cpu_inflate.json"
    cuda_auto.write_text(
        json.dumps(
            _payload(
                device="cuda",
                score_axis="contest_cuda",
                score=0.208983,
                pose=3.36e-5,
                seg=0.00067084,
                raw_sha="1" * 64,
                score_claim=True,
            )
        ),
        encoding="utf-8",
    )
    cuda_cpu_inflate.write_text(
        json.dumps(
            _payload(
                device="cuda",
                score_axis="diagnostic_cuda",
                score=0.208982,
                pose=3.36e-5,
                seg=0.00067083,
                raw_sha="2" * 64,
                inflate_device_policy="cpu",
            )
        ),
        encoding="utf-8",
    )

    analysis = mod.analyze_matrix(
        [
            ("cuda_auto", cuda_auto),
            ("cuda_cpu_inflate", cuda_cpu_inflate),
        ],
        baseline_label="cuda_auto",
    )

    assert analysis["schema"] == "device_axis_eval_matrix_analysis.v1"
    assert analysis["score_claim"] is False
    assert analysis["blockers"] == []
    assert set(analysis["raw_output_groups"]) == {"1" * 64, "2" * 64}
    assert analysis["entries"][0]["scorer_device"] == "cuda"
    assert analysis["runtime_tree_groups"] == {
        "b" * 64: ["cuda_auto", "cuda_cpu_inflate"]
    }
    delta = analysis["deltas_vs_baseline"][0]
    assert delta["label"] == "cuda_cpu_inflate"
    assert delta["same_archive_sha256"] is True
    assert delta["same_runtime_tree_sha256"] is True
    assert delta["same_raw_output_aggregate_sha256"] is False
    assert delta["score_delta"] < 0


def test_matrix_markdown_lists_axes_without_promoting(tmp_path: Path) -> None:
    mod = _load_tool()
    artifact = tmp_path / "cuda_auto.json"
    artifact.write_text(
        json.dumps(
            _payload(
                device="cuda",
                score_axis="contest_cuda",
                score=0.208983,
                pose=3.36e-5,
                seg=0.00067084,
                raw_sha="1" * 64,
            )
        ),
        encoding="utf-8",
    )

    markdown = mod.format_markdown(
        mod.analyze_matrix([("cuda_auto", artifact)], baseline_label="cuda_auto")
    )

    assert "score_claim: `false`" in markdown
    assert "| cuda_auto | contest_cuda | cuda | auto |" in markdown
