from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tac.witness_control import taskspace_single_stage_score_attempt_v1 as subject


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": subject._sha256_file(path),
    }


def _retained_row(tmp_path: Path) -> dict[str, object]:
    archive = tmp_path / "rank-zero.zip"
    archive.write_bytes(b"receiver-valid-rank-zero")
    receipt = tmp_path / "g120.json"
    receipt.write_text("{}")
    binding = _binding(receipt)
    return {
        "attempt_status": "COMPLETED",
        "attempt_identity_sha256": _sha(b"attempt"),
        "physical_stage_identity_sha256": _sha(b"stage"),
        "physical_stage_identity": {},
        "prepose_obstruction": {
            "disposition": subject.g121.RETAIN_POST_G105_POSE,
            "strict_distortion_open": True,
        },
        "retained_for_post_g105_pose": True,
        "selected_archive": _binding(archive),
        "public_runtime_tree": {
            "source_pre": {"tree_sha256": _sha(b"runtime")},
            "sealed_tree_sha256": _sha(b"runtime"),
            "source_post": {"tree_sha256": _sha(b"runtime")},
            "toctou_closed_by": (
                "sealed_content_execution_plus_source_postverify"
            ),
        },
        "g120": {
            "production_receipt": binding,
            "measurement_receipt": binding,
            "observation_receipt": binding,
        },
    }


def test_rank_zero_config_seals_no_pose_or_population_claims(
    tmp_path: Path,
) -> None:
    config = subject.RankZeroScoreAttemptConfigV1(
        run_id="rank-zero",
        resume_from=tmp_path / "run",
        g121_stage_ledger=tmp_path / "ledger.jsonl",
        expected_g121_stage_ledger_sha256=_sha(b"ledger"),
        g121_attempt_identity_sha256=_sha(b"attempt"),
        expected_runtime_tree_sha256=_sha(b"runtime"),
        evaluator_device="cpu",
        video_names_file=tmp_path / "names.txt",
        authority_axis="macOS-CPU advisory",
        competitive_target="0.172",
    )
    document = subject._rank_zero_config_document(
        config,
        command=["tool", "--resume-from", str(config.resume_from)],
    )
    assert document["attempt_mode"] == "G120_SELECTED_RANK_ZERO_ARCHIVE"
    assert document["pose_refit_run"] is False
    assert document["gauss_newton_stages"] == 0
    assert document["exhaustive_stage_coverage_claim"] is False
    assert document["pareto_claim"] is False
    assert document["cross_stage_winner_claim"] is False
    unsealed = dict(document)
    observed = unsealed.pop("config_sha256")
    assert observed == _sha(subject._canonical_json(unsealed))


def test_selected_row_is_frozen_from_exact_ledger_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _retained_row(tmp_path)
    line = subject._canonical_json(row) + b"\n"
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_bytes(line)
    config = SimpleNamespace(
        g121_stage_ledger=ledger,
        expected_g121_stage_ledger_sha256=_sha(line),
        g121_attempt_identity_sha256=row["attempt_identity_sha256"],
    )
    monkeypatch.setattr(subject.g121, "_read_stage_ledger", lambda _path: [row])
    monkeypatch.setattr(
        subject.g121,
        "_validate_completed_attempt",
        lambda value, *, require_physical_files: (
            value if require_physical_files else None
        ),
    )
    reopened: list[dict[str, object]] = []
    monkeypatch.setattr(
        subject,
        "_recursively_reopen_g120_g112",
        lambda value: reopened.append(value),
    )
    opened = subject._open_selected_g121_row(
        config=config,
        run_root=tmp_path / "run",
    )
    assert opened == row
    assert reopened == [row]
    snapshot = json.loads(
        (tmp_path / "run" / "01_selected_g121_row.json").read_text()
    )
    assert snapshot["source_ledger"]["sha256"] == _sha(line)
    assert snapshot["row"] == row
    assert snapshot["recursive_physical_validation"] is True

    ledger.write_bytes(line + line)
    with pytest.raises(
        subject.SingleStageScoreAttemptError,
        match="changed or differs",
    ):
        subject._open_selected_g121_row(
            config=config,
            run_root=tmp_path / "other",
        )


def test_rank_zero_release_copies_exact_g120_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _retained_row(tmp_path)
    runtime_sha = subject._g121_runtime_tree_sha256(row)
    config = subject.RankZeroScoreAttemptConfigV1(
        run_id="rank-zero",
        resume_from=tmp_path / "run",
        g121_stage_ledger=tmp_path / "ledger",
        expected_g121_stage_ledger_sha256=_sha(b"ledger"),
        g121_attempt_identity_sha256=str(
            row["attempt_identity_sha256"]
        ),
        expected_runtime_tree_sha256=runtime_sha,
        evaluator_device="cpu",
        video_names_file=tmp_path / "names",
        authority_axis="macOS-CPU advisory",
        competitive_target="0.172",
    )
    source = SimpleNamespace(
        git_sha="a" * 40,
        files=(),
        tree_sha256=_sha(b"source"),
        all_files_equal_git_head=True,
    )
    runtime = SimpleNamespace(
        files=(
            {
                "relative_path": "inflate.sh",
                "bytes": 7,
                "sha256": _sha(b"inflate"),
                "mode": 0o755,
            },
        ),
        payloads=(("inflate.sh", b"inflate", 0o755),),
        tree_sha256=runtime_sha,
    )
    monkeypatch.setattr(subject, "_git_head", lambda _root: "a" * 40)
    monkeypatch.setattr(
        subject,
        "capture_release_source_closure_v1",
        lambda **_kwargs: source,
    )
    monkeypatch.setattr(
        subject,
        "_explicit_source_bindings",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        subject,
        "capture_public_runtime_v1",
        lambda **_kwargs: runtime,
    )
    release, receipt = subject._materialize_rank_zero_release(
        config=config,
        run_root=tmp_path / "run",
        selected_row=row,
        command=["tool", "--resume-from", str(config.resume_from)],
    )
    assert (release / "archive.zip").read_bytes() == (
        tmp_path / "rank-zero.zip"
    ).read_bytes()
    assert receipt["pose_refit_run"] is False
    assert receipt["archive"]["copied_without_mutation"] is True
    assert receipt["exhaustive_stage_coverage_claim"] is False


def test_report_parser_preserves_component_precision() -> None:
    report = """=== Evaluation config ===
  batch_size: 16
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.00123456
  Average SegNet Distortion: 0.00098765
  Submission file size: 100,000 bytes
  Original uncompressed size: 37,545,489 bytes
  Compression Rate: 0.00266344
  Final score: 100*segnet_dist + sqrt = 0.18
"""
    parsed = subject._parse_report(report, archive_bytes=100_000)
    assert parsed["sample_count"] == 600
    assert parsed["d_pose_reported_8dp"] == "0.00123456"
    assert parsed["d_seg_reported_8dp"] == "0.00098765"
    assert float(
        parsed["score_recomputed_from_reported_8dp_components"]
    ) > 0.0
