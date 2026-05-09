from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "harvest_a1_bias_correction_sweep.py"
)
SPEC = importlib.util.spec_from_file_location("harvest_a1_bias_correction_sweep", MODULE_PATH)
assert SPEC is not None
harvest = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(harvest)


def report_text(submission_name: str) -> str:
    return f"""=== Evaluation config ===
  batch_size: 16
  device: cpu
  report: submissions/{submission_name}/report.txt
  seed: 1234
  submission_dir: submissions/{submission_name}
  uncompressed_dir: /gha/videos
  video_names_file: /gha/public_test_video_names.txt
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.00003286
  Average SegNet Distortion: 0.00056023
  Submission file size: 178,262 bytes
  Original uncompressed size: 37,545,489 bytes
  Compression Rate: 0.00474789
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.19
"""


def test_parse_report_requires_exact_submission_name_not_prefix() -> None:
    parsed = harvest.parse_report(
        report_text("a1_bias_v10_red_channel_only_20260509"),
        expected_submission_name="a1_bias_v1",
    )

    assert parsed["_error"].startswith("report submission_name mismatch")
    assert parsed["report_submission_name"] == "a1_bias_v10_red_channel_only_20260509"


def test_select_custodial_report_path_ignores_prefix_report(tmp_path: Path) -> None:
    wrong = tmp_path / "wrong" / "report.txt"
    wrong.parent.mkdir()
    wrong.write_text(report_text("a1_bias_v10_red_channel_only_20260509"))
    expected = tmp_path / "expected" / "report.txt"
    expected.parent.mkdir()
    expected.write_text(report_text("a1_bias_v1"))

    assert harvest.select_custodial_report_path(tmp_path, "a1_bias_v1") == expected


def test_select_custodial_report_path_refuses_duplicate_exact_reports(
    tmp_path: Path,
) -> None:
    """Two exact reports are ambiguous custody and must fail closed."""
    first = tmp_path / "a" / "report.txt"
    second = tmp_path / "b" / "report.txt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text(report_text("a1_bias_v1"), encoding="utf-8")
    second.write_text(report_text("a1_bias_v1"), encoding="utf-8")

    assert harvest.select_custodial_report_path(tmp_path, "a1_bias_v1") is None


def test_harvest_one_refuses_mismatched_report_without_score_claim(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        harvest,
        "find_run_for_submission",
        lambda _name: {
            "run_id": 255,
            "conclusion": "success",
            "createdAt": "2026-05-09T00:00:00Z",
            "artifact_name": "eval-a1_bias_v1",
        },
    )

    def fake_gh(args: list[str], capture: bool = True) -> subprocess.CompletedProcess[str]:
        del capture
        dest = Path(args[args.index("-D") + 1])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "report.txt").write_text(
            report_text("a1_bias_v10_red_channel_only_20260509")
        )
        return subprocess.CompletedProcess(["gh", *args], 0, "", "")

    monkeypatch.setattr(harvest, "gh", fake_gh)

    row = harvest.harvest_one("a1_bias_v1")

    assert row["status"] == "report_custody_failed"
    assert row["score"] is None
    assert row["tag"] == "[report_custody_failed]"
    assert "score_claim" not in row
