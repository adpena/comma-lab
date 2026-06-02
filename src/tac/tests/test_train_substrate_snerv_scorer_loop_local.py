# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from comma_lab.storage_tiers import StorageTierError
from experiments import train_substrate_snerv_scorer_loop_local as trainer
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SNERV_SPECTRA_PRESERVING_ADAPTER,
)


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
            "--component-guard-mode",
            "pose_seg_hard",
            "--snerv-spectra-preserving-adapter",
            "--snerv-fc-dim",
            "12",
            "--snerv-emb-size",
            "2",
            "--snerv-mfu-scales",
            "1,3",
            "--snerv-hfr-gain",
            "0.25",
            "--snerv-temporal-context",
            "1",
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
    assert kwargs["component_guard_mode"] == "pose_seg_hard"
    assert kwargs["snerv_spectra_preserving_adapter"] is True
    assert kwargs["snerv_fc_dim"] == 12
    assert kwargs["snerv_emb_size"] == 2
    assert kwargs["snerv_mfu_scales"] == (1, 3)
    assert kwargs["snerv_hfr_gain"] == pytest.approx(0.25)
    assert kwargs["snerv_temporal_context"] == 1


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

    result = _FakeResult()
    materialization = trainer._materialize_best_packet(result, tmp_path)
    report = trainer._build_report(
        result=result,
        args=args,
        output_dir=tmp_path,
        storage_payload={"schema": "storage.v1", "score_claim": False},
        launch_path=tmp_path / "launch.json",
        best_packet_materialization=materialization,
    )

    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["best_packet_materialized"] is True
    assert report["best_packet_bytes"] == len(_FakeResult.best_packet)
    assert report["best_packet_sha256"] == hashlib.sha256(
        _FakeResult.best_packet
    ).hexdigest()
    assert Path(report["best_packet_path"]).read_bytes() == _FakeResult.best_packet
    assert report["baseline_archive_bytes"] == 1000
    assert report["best_archive_bytes"] == 980
    assert report["snerv_model_size_adapter"] == SNERV_SPECTRA_PRESERVING_ADAPTER
    assert report["snerv_mfu_scales"] == [1, 3]
    assert report["snerv_hfr_gain"] == pytest.approx(0.25)
    assert report["decoder_feature_count"] == 14
    assert report["component_guard_mode"] == "score_primary"
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
            "--snerv-spectra-preserving-adapter",
            "--snerv-mfu-scales",
            "1,3",
            "--snerv-hfr-gain",
            "0.25",
        ]
    )

    assert rc == 0
    assert calls["n_pairs"] == 2
    assert calls["wavelet"] == "haar"
    assert calls["max_trials"] == 0
    assert calls["snerv_spectra_preserving_adapter"] is True
    assert calls["snerv_mfu_scales"] == (1, 3)
    assert calls["snerv_hfr_gain"] == pytest.approx(0.25)
    assert calls["component_guard_mode"] == "score_primary"
    payload = json.loads(research_json.read_text(encoding="utf-8"))
    assert payload["schema"] == trainer.TRAINER_SCHEMA
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["best_packet_materialized"] is True
    assert payload["best_packet_bytes"] == len(_FakeResult.best_packet)
    assert payload["best_packet_sha256"] == hashlib.sha256(
        _FakeResult.best_packet
    ).hexdigest()
    assert Path(payload["best_packet_path"]).read_bytes() == _FakeResult.best_packet
    assert payload["best_packet_materialization"]["materialized"] is True
    assert (
        "snerv_native_scorer_loop_best_packet_not_materialized"
        not in payload["blockers"]
    )
    assert payload["result_sha256"]
    assert research_md.read_text(encoding="utf-8").startswith(
        "# SNeRV scorer-loop QAT local trainer"
    )
    assert (output_dir / "snerv_scorer_loop_qat_launch_preflight.json").exists()
    assert (output_dir / "snerv_scorer_loop_qat_result.json").exists()
    assert (output_dir / "best_packet.snar").read_bytes() == _FakeResult.best_packet


class _FakeResult:
    best_packet = b"SNAR1-test-best-packet"

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "schema": "snerv_scorer_loop_decoder_qat_smoke.v1",
            "axis_tag": "[macOS-CPU advisory]",
            "n_pairs": 2,
            "levels": 1,
            "wavelet": "haar",
            "snerv_model_size_adapter": SNERV_SPECTRA_PRESERVING_ADAPTER,
            "snerv_mfu_scales": [1, 3],
            "snerv_hfr_gain": 0.25,
            "snerv_temporal_context": 0,
            "decoder_feature_count": 14,
            "qat_bits": 8,
            "search_mode": "random_signed",
            "component_guard_mode": "score_primary",
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
            "best_packet_bytes": len(self.best_packet),
            "best_packet_sha256": hashlib.sha256(self.best_packet).hexdigest(),
            "ready_for_pose_guard_gate": True,
            "receiver_contract_satisfied": True,
            "blockers": ["local_smoke_only_not_full_600_pairs"],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
