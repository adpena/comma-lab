# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import math
import stat
import zipfile
from pathlib import Path

import pytest

from tac.auth_eval_result import CONTEST_UNCOMPRESSED_BYTES
from tac.optimization.dp1_procedural_paired_adjudication import (
    build_dp1_procedural_paired_adjudication,
    register_dp1_procedural_paired_adjudication,
    render_markdown,
)


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _contest_score(*, seg: float, pose: float, archive_bytes: int) -> float:
    return (
        100.0 * seg
        + math.sqrt(10.0 * pose)
        + 25.0 * archive_bytes / CONTEST_UNCOMPRESSED_BYTES
    )


def _write_candidate_output(
    path: Path,
    *,
    lane_id: str,
    procedural: bool,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path / "archive.zip", "w") as zf:
        zf.writestr("0.bin", b"dp1-paired-adjudication")
    submission = path / "submission"
    submission.mkdir()
    inflate_sh = submission / "inflate.sh"
    inflate_sh.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    inflate_sh.chmod(inflate_sh.stat().st_mode | stat.S_IXUSR)
    (submission / "inflate.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    false_flags = {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    (path / "manifest.json").write_text(
        json.dumps(false_flags, sort_keys=True), encoding="utf-8"
    )
    (path / "provenance.json").write_text(
        json.dumps({**false_flags, "lane_id": lane_id}, sort_keys=True),
        encoding="utf-8",
    )
    if procedural:
        (path / "procedural_variant_provenance.json").write_text(
            json.dumps({**false_flags, "null_exploit_control": False}, sort_keys=True),
            encoding="utf-8",
        )
    return path / "archive.zip"


def _write_eval(
    path: Path,
    *,
    axis: str,
    archive: Path,
    seg: float,
    pose: float,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    archive_bytes = archive.stat().st_size
    score = _contest_score(seg=seg, pose=pose, archive_bytes=archive_bytes)
    payload = {
        "score_axis": axis,
        "evidence_grade": "contest-CPU" if axis == "contest_cpu" else "contest-CUDA",
        "lane_tag": "[contest-CPU]" if axis == "contest_cpu" else "[contest-CUDA]",
        "score_recomputed_from_components": score,
        "canonical_score": score,
        "avg_segnet_dist": seg,
        "avg_posenet_dist": pose,
        "archive_size_bytes": archive_bytes,
        "archive_sha256": _sha256_file(archive),
        "rate_unscaled": archive_bytes / CONTEST_UNCOMPRESSED_BYTES,
    }
    (path / "contest_auth_eval.json").write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8"
    )


def _write_complete_fixture(tmp_path: Path) -> dict[str, Path]:
    baseline_archive = _write_candidate_output(
        tmp_path / "baseline_out",
        lane_id="lane_dp1_baseline",
        procedural=False,
    )
    procedural_archive = _write_candidate_output(
        tmp_path / "procedural_out",
        lane_id="lane_dp1_procedural",
        procedural=True,
    )
    paths = {
        "baseline_output_dir": tmp_path / "baseline_out",
        "procedural_output_dir": tmp_path / "procedural_out",
        "baseline_cpu_dir": tmp_path / "eval" / "baseline_cpu",
        "baseline_cuda_dir": tmp_path / "eval" / "baseline_cuda",
        "procedural_cpu_dir": tmp_path / "eval" / "procedural_cpu",
        "procedural_cuda_dir": tmp_path / "eval" / "procedural_cuda",
    }
    _write_eval(
        paths["baseline_cpu_dir"],
        axis="contest_cpu",
        archive=baseline_archive,
        seg=0.0010,
        pose=0.000040,
    )
    _write_eval(
        paths["baseline_cuda_dir"],
        axis="contest_cuda",
        archive=baseline_archive,
        seg=0.0011,
        pose=0.000041,
    )
    _write_eval(
        paths["procedural_cpu_dir"],
        axis="contest_cpu",
        archive=procedural_archive,
        seg=0.0008,
        pose=0.000038,
    )
    _write_eval(
        paths["procedural_cuda_dir"],
        axis="contest_cuda",
        archive=procedural_archive,
        seg=0.0009,
        pose=0.000039,
    )
    return paths


def test_adjudication_proceeds_when_procedural_improves_both_axes(
    tmp_path: Path,
) -> None:
    paths = _write_complete_fixture(tmp_path)

    report = build_dp1_procedural_paired_adjudication(
        **paths,
        repo_root=tmp_path,
    )

    assert report["all_required_evidence_valid"] is True
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["verdict"] == "PROCEED"
    deltas = report["score_deltas_procedural_minus_baseline"]
    assert deltas["contest_cpu"] < 0.0
    assert deltas["contest_cuda"] < 0.0
    assert report["metric_value"] == pytest.approx(
        max(deltas["contest_cpu"], deltas["contest_cuda"])
    )
    assert "procedural_codebook_improves_both_axes" in report["next_action"]
    assert "contest_cpu" in render_markdown(report)


def test_adjudication_fails_closed_on_axis_mismatch(tmp_path: Path) -> None:
    paths = _write_complete_fixture(tmp_path)
    payload_path = paths["procedural_cpu_dir"] / "contest_auth_eval.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["score_axis"] = "diagnostic_cpu"
    payload_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    report = build_dp1_procedural_paired_adjudication(
        **paths,
        repo_root=tmp_path,
    )

    assert report["all_required_evidence_valid"] is False
    assert report["verdict"] == "OPERATOR_REVIEW_REQUIRED"
    assert report["metric_value"] is None
    assert any(
        blocker.startswith("procedural_contest_cpu_score_axis_mismatch")
        for blocker in report["blockers"]
    )


def test_register_probe_outcome_uses_custom_locked_ledger(tmp_path: Path) -> None:
    paths = _write_complete_fixture(tmp_path)
    report = build_dp1_procedural_paired_adjudication(
        **paths,
        repo_root=tmp_path,
    )
    ledger = tmp_path / "probe_outcomes.jsonl"

    record = register_dp1_procedural_paired_adjudication(
        report,
        evidence_path=tmp_path / "adjudication.json",
        path=ledger,
        lock_path=tmp_path / "probe_outcomes.jsonl.lock",
        agent="codex:test",
    )

    assert record["verdict"] == "PROCEED"
    assert record["blocker_status"] == "advisory"
    assert record["metric_value"] < 0.0
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert rows == [record]
