from __future__ import annotations

import json
from pathlib import Path

from tac.analysis.public_pr_mechanism_index import (
    build_public_pr_mechanism_index,
    render_markdown_summary,
    write_index_outputs,
)


def test_public_pr_mechanism_index_extracts_device_scores_and_mechanisms(tmp_path: Path) -> None:
    root = tmp_path / "public_pr_archive_release_view"
    pr_dir = root / "public_pr95_intake_20260505_auto" / "source" / "submissions" / "hnerv_muon"
    pr_dir.mkdir(parents=True)
    (root / "public_pr95_intake_20260505_auto" / "pr_body.md").write_text(
        """
  device: cpu
  Average PoseNet Distortion: 0.00003494
  Average SegNet Distortion: 0.00061212
  Submission file size: 178,417 bytes
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.20
""",
        encoding="utf-8",
    )
    (pr_dir / "README.md").write_text(
        "HNeRV decoder with QAT, Muon, DALI eval roundtrip, sidecar, brotli entropy.",
        encoding="utf-8",
    )

    index = build_public_pr_mechanism_index([root], min_pr=95, repo_root=tmp_path)

    assert index["score_claim"] is False
    assert index["promotion_eligible"] is False
    assert index["file_count"] == 2
    [summary] = index["summary_by_pr"]
    assert summary["pr"] == 95
    assert summary["devices"] == {"cpu": 1}
    assert summary["eval_row_count"] == 1
    assert "hnerv_renderer" in summary["mechanisms"]
    assert "quantization_aware_training" in summary["mechanisms"]
    best = summary["best_eval_row"]
    assert best["device"] == "cpu"
    assert best["archive_bytes"] == 178_417
    assert best["recomputed_score_from_rounded_components"] > 0.19


def test_public_pr_mechanism_index_dedupes_identical_mirror_files(tmp_path: Path) -> None:
    left = tmp_path / "public_pr_archive_release_view" / "public_pr103_intake_20260505_auto"
    right = tmp_path / "public_pr_archive_kaggle_mirror" / "public_pr103_intake_20260505_auto"
    left.mkdir(parents=True)
    right.mkdir(parents=True)
    body = "arithmetic range coding and HNeRV sidecar\n"
    (left / "pr_body.md").write_text(body, encoding="utf-8")
    (right / "pr_body.md").write_text(body, encoding="utf-8")

    index = build_public_pr_mechanism_index([left.parent.parent, right.parent.parent], min_pr=103, repo_root=tmp_path)

    assert index["file_count"] == 1
    [summary] = index["summary_by_pr"]
    assert summary["mechanisms"] == ["arithmetic_entropy_codec", "hnerv_renderer", "latent_sidecar_or_correction"]


def test_public_pr_mechanism_index_infers_cpu_capable_from_gpu_requirement_question(tmp_path: Path) -> None:
    root = tmp_path / "public_pr_archive_release_view" / "public_pr103_intake_20260505_auto"
    root.mkdir(parents=True)
    (root / "pr_body.md").write_text(
        """
Average PoseNet Distortion: 0.00003443
Average SegNet Distortion: 0.00057638
Submission file size: 178,223 bytes
Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.19

# does your submission require gpu for evaluation (inflation)?
no
""",
        encoding="utf-8",
    )

    index = build_public_pr_mechanism_index([root.parent], min_pr=103, repo_root=tmp_path)

    [summary] = index["summary_by_pr"]
    assert summary["devices"] == {"cpu_capable": 1}
    assert summary["best_eval_row"]["device"] == "cpu_capable"


def test_public_pr_mechanism_index_writes_json_and_markdown(tmp_path: Path) -> None:
    root = tmp_path / "public_pr_intake_full" / "public_pr106_intake_20260505_auto"
    root.mkdir(parents=True)
    (root / "pr_body.md").write_text(
        "device: cuda\nAverage PoseNet Distortion: 0.1\nAverage SegNet Distortion: 0.2\n"
        "Submission file size: 10 bytes\nFinal score: 1.0\n",
        encoding="utf-8",
    )
    index = build_public_pr_mechanism_index([root.parent], min_pr=106, repo_root=tmp_path)
    json_out = tmp_path / "index.json"
    md_out = tmp_path / "index.md"

    write_index_outputs(index, json_out=json_out, md_out=md_out)

    assert json.loads(json_out.read_text(encoding="utf-8"))["schema"] == "public_pr_mechanism_index.v1"
    assert "Public PR mechanism index" in md_out.read_text(encoding="utf-8")
    assert "| 106 |" in render_markdown_summary(index)
