# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from comma_lab.storage_tiers import StorageTierError
from experiments import train_substrate_snerv_scorer_loop_local as trainer


def test_snerv_scorer_loop_trainer_parser_requires_mode() -> None:
    with pytest.raises(SystemExit):
        trainer._build_parser().parse_args([])

    assert trainer.TRAINER_SCHEMA == "snerv_scorer_loop_qat_local_trainer.v1"


def test_snerv_scorer_loop_trainer_maps_kwargs() -> None:
    args = trainer._build_parser().parse_args(
        [
            "--score-loop",
            "--n-pairs",
            "4",
            "--levels",
            "1",
            "--wavelet",
            "haar",
            "--search-mode",
            "nes_pair_robust",
            "--max-trials",
            "3",
            "--byte-pressure-multiplier",
            "2.5",
            "--max-archive-byte-growth",
            "128",
            "--pose-slack",
            "0.01",
            "--seg-slack",
            "0.02",
        ]
    )

    kwargs = trainer._score_loop_kwargs_from_args(args)

    assert kwargs["n_pairs"] == 4
    assert kwargs["levels"] == 1
    assert kwargs["wavelet"] == "haar"
    assert kwargs["search_mode"] == "nes_pair_robust"
    assert kwargs["max_trials"] == 3
    assert kwargs["byte_pressure_multiplier"] == pytest.approx(2.5)
    assert kwargs["max_archive_byte_growth"] == 128
    assert kwargs["pose_slack"] == pytest.approx(0.01)
    assert kwargs["seg_slack"] == pytest.approx(0.02)


def test_snerv_scorer_loop_trainer_rejects_local_output_without_opt_in(
    tmp_path: Path,
) -> None:
    args = trainer._build_parser().parse_args(
        ["--score-loop", "--output-dir", str(tmp_path / "local")]
    )

    with pytest.raises(StorageTierError, match="local_disk_tier_disabled"):
        trainer._resolve_output_dir(args)


def test_snerv_scorer_loop_trainer_allows_explicit_local_output(
    tmp_path: Path,
) -> None:
    args = trainer._build_parser().parse_args(
        [
            "--score-loop",
            "--output-dir",
            str(tmp_path / "local"),
            "--allow-local-output-dir",
        ]
    )

    output, storage = trainer._resolve_output_dir(args)

    assert output == (tmp_path / "local").resolve(strict=False)
    assert output.is_dir()
    assert storage["schema"] == "snerv_scorer_loop_qat_explicit_output_preflight.v1"
    assert storage["score_claim"] is False
    assert storage["ready_for_exact_eval_dispatch"] is False


def test_snerv_scorer_loop_trainer_builds_false_authority_report(
    tmp_path: Path,
) -> None:
    args = trainer._build_parser().parse_args(
        ["--score-loop", "--output-dir", str(tmp_path), "--allow-local-output-dir"]
    )

    report = trainer._build_report(
        result=_FakeResult(),
        args=args,
        output_dir=tmp_path,
        storage_payload={"schema": "storage.v1", "score_claim": False},
        launch_path=tmp_path / "launch.json",
    )

    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["baseline_archive_bytes"] == 1000
    assert report["best_archive_bytes"] == 980
    assert report["accepted_improvement"] is True
    assert "paired_contest_cpu_cuda_pass_missing" in report["blockers"]
    assert "official_snerv_mfu_hfr_tub_parity_not_proven" in report["blockers"]
    markdown = trainer.render_snerv_scorer_loop_local_markdown(report)
    assert "SNeRV scorer-loop QAT local trainer" in markdown


def test_snerv_scorer_loop_trainer_cli_writes_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    research_json = tmp_path / "research.json"
    research_md = tmp_path / "research.md"
    calls: dict[str, Any] = {}

    def fake_runner(**kwargs: Any) -> _FakeResult:
        calls.update(kwargs)
        return _FakeResult()

    monkeypatch.setattr(trainer, "run_snerv_scorer_loop_decoder_qat_smoke", fake_runner)

    rc = trainer.main(
        [
            "--score-loop",
            "--output-dir",
            str(output_dir),
            "--allow-local-output-dir",
            "--research-json",
            str(research_json),
            "--research-md",
            str(research_md),
            "--n-pairs",
            "2",
            "--wavelet",
            "haar",
            "--max-trials",
            "0",
        ]
    )

    assert rc == 0
    assert calls["n_pairs"] == 2
    assert calls["wavelet"] == "haar"
    assert calls["max_trials"] == 0
    payload = json.loads(research_json.read_text(encoding="utf-8"))
    assert payload["schema"] == trainer.TRAINER_SCHEMA
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["result_sha256"]
    assert research_md.read_text(encoding="utf-8").startswith(
        "# SNeRV scorer-loop QAT local trainer"
    )
    assert (output_dir / "snerv_scorer_loop_qat_launch_preflight.json").exists()
    assert (output_dir / "snerv_scorer_loop_qat_result.json").exists()


class _FakeResult:
    def as_jsonable(self) -> dict[str, Any]:
        return {
            "schema": "snerv_scorer_loop_decoder_qat_smoke.v1",
            "axis_tag": "[macOS-CPU advisory]",
            "n_pairs": 2,
            "levels": 1,
            "wavelet": "haar",
            "qat_bits": 8,
            "search_mode": "random_signed",
            "scorer_loop_evaluations": 1,
            "baseline": {
                "archive_bytes": 1000,
                "score_linf": 9.0,
                "d_seg_linf": 0.1,
                "d_pose_linf": 0.2,
            },
            "best": {
                "archive_bytes": 980,
                "score_linf": 8.5,
                "d_seg_linf": 0.09,
                "d_pose_linf": 0.19,
            },
            "accepted_improvement": True,
            "ready_for_pose_guard_gate": True,
            "receiver_contract_satisfied": True,
            "blockers": ["local_smoke_only_not_full_600_pairs"],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
