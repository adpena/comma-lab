# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
)
from tools import run_compact_renderer_mlx_spine_runner as runner


def _source_video(tmp_path: Path) -> Path:
    path = tmp_path / "source.mkv"
    path.write_bytes(b"not-a-real-video-for-runner-wiring-test")
    return path


def _patch_lightweight_snerv_native_report(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_native_export(**kwargs: Any) -> dict[str, Any]:
        captured["modelsize_candidate"] = dict(kwargs["modelsize_candidate"] or {})
        return {
            "schema": "compact_runner_snerv_mlx_native_export_attachment.v1",
            "executed": True,
            "requested": True,
            "native_mlx_training_executed": True,
            "score_aware_long_training_executed": True,
            "receiver_proof_passed": True,
            "receiver_contract_satisfied": True,
            "native_mlx_full600_campaign_ready": False,
            "blockers": [],
            "source_pair_indices": [0],
            "score_aware_long_training": {
                "eval_roundtrip_ste_enabled": False,
                "teacher_binding": {"has_real_posenet_teacher": False},
            },
        }

    monkeypatch.setattr(runner, "_run_snerv_native_mlx_export_attachment", fake_native_export)
    monkeypatch.setattr(
        runner,
        "build_snerv_mlx_native_file_backed_evidence",
        lambda *_args, **_kwargs: {
            "required_pair_file_backed_export_proof_passed": False,
            "blockers": [],
        },
    )
    monkeypatch.setattr(
        runner,
        "build_snerv_mlx_native_adapter_contract",
        lambda *_args, **_kwargs: {"schema": "test_contract.v1", "blockers": []},
    )
    monkeypatch.setattr(
        runner,
        "write_nerv_candidate_feedback_files",
        lambda *_args, **_kwargs: {"schema": "test_feedback.v1"},
    )
    monkeypatch.setattr(
        runner,
        "_base_report",
        lambda *, output_dir, mode, hard_byte_ceilings, repo_root: {
            "schema": runner.COMPACT_RENDERER_MLX_SPINE_RUNNER_SCHEMA,
            "mode": mode,
            "output_dir": Path(output_dir).as_posix(),
            "hard_byte_ceilings": list(hard_byte_ceilings),
            "repo_root": Path(repo_root).as_posix(),
        },
    )
    return captured


def test_snerv_runner_binds_manual_official_skip_high_mode_to_native_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_lightweight_snerv_native_report(monkeypatch)

    report = runner.execute_snerv_inverse_steg_advisory_and_adapt(
        output_dir=tmp_path / "out",
        num_pairs=2,
        epochs=1,
        source_video_path=_source_video(tmp_path),
        hard_byte_ceilings=(178_000,),
        modelsize_candidate={
            "snerv_model_size_adapter": SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
        },
        snerv_official_skip_high_mode_override="shared_mean",
        run_native_mlx_export=True,
        snerv_score_aware_long_training_epochs=1,
        upstream_dir=tmp_path / "upstream",
        repo_root=Path.cwd(),
        allow_overwrite=True,
    )

    candidate = captured["modelsize_candidate"]
    assert candidate["official_skip_high_mode"] == "shared_mean"
    assert candidate["snerv_official_skip_high_mode"] == "shared_mean"
    assert report["snerv_mlx_native_export"]["executed"] is True


def test_snerv_runner_rejects_candidate_skip_high_cli_conflict(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        runner.CompactRendererMlxSpineRunnerError,
        match="official_skip_high_mode conflicts",
    ):
        runner.execute_snerv_inverse_steg_advisory_and_adapt(
            output_dir=tmp_path / "out",
            num_pairs=2,
            epochs=1,
            source_video_path=_source_video(tmp_path),
            hard_byte_ceilings=(178_000,),
            modelsize_candidate={
                "snerv_model_size_adapter": (
                    SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER
                ),
                "official_skip_high_mode": "full",
            },
            snerv_official_skip_high_mode_override="shared_mean",
            run_native_mlx_export=True,
            snerv_score_aware_long_training_epochs=1,
            upstream_dir=tmp_path / "upstream",
            repo_root=Path.cwd(),
            allow_overwrite=True,
        )


def test_snerv_runner_parser_exposes_official_skip_high_mode() -> None:
    args = runner._parse_args(
        [
            "--execute-family",
            "snerv",
            "--snerv-official-skip-high-mode",
            "shared_mean",
        ]
    )

    assert args.snerv_official_skip_high_mode == "shared_mean"
