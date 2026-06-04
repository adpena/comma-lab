from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

TOOL_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "export_hinerv_checkpoint_archive.py"
)


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "export_hinerv_checkpoint_archive", TOOL_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_state_path_selects_ema_and_live(tmp_path: Path) -> None:
    tool = _load_tool()
    ema_state = tmp_path / "epoch0001.ema_shadow.state.npsd"
    live_state = tmp_path / "epoch0001.live.state.npsd"
    ema_state.write_bytes(b"ema")
    live_state.write_bytes(b"live")
    meta = {
        "ema_shadow_state_path": ema_state.as_posix(),
        "live_state_path": live_state.as_posix(),
    }

    assert tool._checkpoint_state_path(meta, state_kind="ema") == ema_state.resolve(
        strict=False
    )
    assert tool._checkpoint_state_path(meta, state_kind="live") == live_state.resolve(
        strict=False
    )


def test_checkpoint_state_path_fails_closed_for_missing_state(tmp_path: Path) -> None:
    tool = _load_tool()
    meta = {"ema_shadow_state_path": (tmp_path / "missing.state.npsd").as_posix()}

    with pytest.raises(FileNotFoundError, match="checkpoint state not found"):
        tool._checkpoint_state_path(meta, state_kind="ema")


def test_checkpoint_state_path_fails_closed_for_missing_meta_key() -> None:
    tool = _load_tool()

    with pytest.raises(ValueError, match="checkpoint meta missing live_state_path"):
        tool._checkpoint_state_path({}, state_kind="live")


def test_candidate_decoder_codec_reaches_archive_export_when_runner_is_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_tool()
    state_path = tmp_path / "epoch0001.ema_shadow.state.npsd"
    state_path.write_bytes(b"state")
    meta_path = tmp_path / "checkpoint_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "global_epoch": 11,
                "ema_shadow_state_path": state_path.as_posix(),
            }
        ),
        encoding="utf-8",
    )
    startup_path = tmp_path / "startup.json"
    startup_path.write_text(
        json.dumps(
            {
                "modelsize_candidate": {
                    "candidate_id": "hinerv_candidate_codec_export",
                    "num_pairs": 2,
                    "latent_dim": 8,
                    "embed_dim": 8,
                    "decoder_channel": 6,
                    "decoder_codec": "int7_mixed",
                    "nominal_total_payload_bytes": 6,
                    "hard_byte_ceiling": 5,
                },
                "command_args": {
                    "num_pairs": 2,
                    "compact_decoder_codec": "portfolio_auto",
                },
                "hard_byte_ceilings": [5],
            }
        ),
        encoding="utf-8",
    )

    class _FakeModel:
        def __init__(self, cfg: object) -> None:
            self.cfg = cfg

    class _FakeAdapter:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def import_state_dict(self, _model: object, _path: Path) -> None:
            pass

    captured: dict[str, object] = {}

    def _fake_export(_model: object, output_dir: Path, **kwargs: object):
        captured.update(kwargs)
        output_dir.mkdir(parents=True, exist_ok=True)
        archive_path = output_dir / "archive.zip"
        archive_path.write_bytes(b"archive")
        return archive_path, "fake_sha256", archive_path.stat().st_size

    monkeypatch.setattr(tool, "HinervSubstrateMLX", _FakeModel)
    monkeypatch.setattr(tool, "RendererBundle", lambda **kwargs: kwargs)
    monkeypatch.setattr(tool, "MlxScoreAwareAdapter", _FakeAdapter)
    monkeypatch.setattr(
        tool,
        "unpack_state_dict_numpy",
        lambda _payload: {"any": np.zeros((1,), dtype=np.float32)},
    )
    monkeypatch.setattr(
        tool,
        "_modelsize_integrity_profile",
        lambda *_args, **_kwargs: {
            "schema": "hinerv_checkpoint_modelsize_integrity.v1",
            "profile_ready": True,
            "matches_candidate_controls": True,
            "blockers": [],
        },
    )
    monkeypatch.setattr(tool, "export_hi_nerv_mlx_archive", _fake_export)

    report = tool.export_checkpoint_archive(
        startup_json=startup_path,
        checkpoint_meta=meta_path,
        output_dir=tmp_path / "out",
        repo_root=tmp_path,
    )

    assert report["decoder_codec"] == "int7_mixed"
    assert captured["decoder_codec"] == "int7_mixed"
    assert (
        report["decoder_codec_resolution"]["resolution_source"]
        == "modelsize_candidate_decoder_codec"
    )
    assert (
        report["decoder_codec_resolution"][
            "modelsize_candidate_decoder_codec_propagates_to_export"
        ]
        is True
    )
    feedback = report["modelsize_byte_cap_feedback_row"]
    assert feedback["schema"] == "nerv_modelsize_byte_cap_feedback_row.v1"
    assert feedback["family"] == "hi_nerv"
    assert feedback["candidate_id"] == "hinerv_candidate_codec_export"
    assert feedback["decoder_codec"] == "int7_mixed"
    assert feedback["hard_byte_ceiling"] == 5
    assert feedback["hard_byte_ceiling_enforced_by_export"] == 5
    assert feedback["hard_byte_ceiling_measurement_bypass_enabled"] is False
    assert feedback["nominal_total_payload_bytes"] == 6
    assert feedback["measured_archive_bytes"] == 7
    assert feedback["archive_minus_nominal_bytes"] == 1
    assert feedback["archive_to_nominal_ratio"] == pytest.approx(7 / 6)
    assert feedback["calibrated_archive_overrun_bytes"] == 2
    assert feedback["required_nominal_payload_bytes_max"] == 4
    assert feedback["receiver_closed"] is False
    assert "candidate_decoder_codec_not_export_authority" not in report["blockers"]


def test_over_hard_byte_ceiling_measurement_bypass_keeps_report_false_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_tool()
    state_path = tmp_path / "epoch0001.ema_shadow.state.npsd"
    state_path.write_bytes(b"state")
    meta_path = tmp_path / "checkpoint_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "global_epoch": 11,
                "ema_shadow_state_path": state_path.as_posix(),
            }
        ),
        encoding="utf-8",
    )
    startup_path = tmp_path / "startup.json"
    startup_path.write_text(
        json.dumps(
            {
                "modelsize_candidate": {
                    "candidate_id": "hinerv_overcap_measurement",
                    "num_pairs": 2,
                    "latent_dim": 8,
                    "embed_dim": 8,
                    "decoder_channel": 6,
                    "nominal_total_payload_bytes": 120_000,
                    "hard_byte_ceiling": 178_000,
                },
                "command_args": {"num_pairs": 2},
                "hard_byte_ceilings": [216_000],
            }
        ),
        encoding="utf-8",
    )

    class _FakeModel:
        def __init__(self, cfg: object) -> None:
            self.cfg = cfg

    class _FakeAdapter:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def import_state_dict(self, _model: object, _path: Path) -> None:
            pass

    captured: dict[str, object] = {}

    def _fake_export(_model: object, output_dir: Path, **kwargs: object):
        captured.update(kwargs)
        output_dir.mkdir(parents=True, exist_ok=True)
        archive_path = output_dir / "archive.zip"
        archive_path.write_bytes(b"x" * 214_187)
        return archive_path, "fake_sha256", archive_path.stat().st_size

    monkeypatch.setattr(tool, "HinervSubstrateMLX", _FakeModel)
    monkeypatch.setattr(tool, "RendererBundle", lambda **kwargs: kwargs)
    monkeypatch.setattr(tool, "MlxScoreAwareAdapter", _FakeAdapter)
    monkeypatch.setattr(
        tool,
        "unpack_state_dict_numpy",
        lambda _payload: {"any": np.zeros((1,), dtype=np.float32)},
    )
    monkeypatch.setattr(
        tool,
        "_modelsize_integrity_profile",
        lambda *_args, **_kwargs: {
            "schema": "hinerv_checkpoint_modelsize_integrity.v1",
            "profile_ready": True,
            "matches_candidate_controls": True,
            "blockers": [],
        },
    )
    monkeypatch.setattr(tool, "export_hi_nerv_mlx_archive", _fake_export)

    report = tool.export_checkpoint_archive(
        startup_json=startup_path,
        checkpoint_meta=meta_path,
        output_dir=tmp_path / "out",
        repo_root=tmp_path,
        allow_over_hard_byte_ceiling_for_measurement=True,
    )

    assert captured["hard_byte_ceiling"] is None
    assert report["hard_byte_ceiling_enforced_by_export"] is None
    assert report["hard_byte_ceiling_requested_by_candidate_or_startup"] == 178_000
    assert report["hard_byte_ceiling_measurement_bypass_enabled"] is True
    assert "archive_bytes_exceed_tightest_hard_ceiling" in report["blockers"]
    assert "hard_byte_ceiling_export_bypassed_for_measurement" in report["blockers"]
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    feedback = report["modelsize_byte_cap_feedback_row"]
    assert feedback["hard_byte_ceiling"] == 178_000
    assert feedback["hard_byte_ceiling_enforced_by_export"] is None
    assert feedback["hard_byte_ceiling_measurement_bypass_enabled"] is True
    assert feedback["receiver_closed"] is False


def test_blockers_price_archive_and_receiver_proof() -> None:
    tool = _load_tool()

    blockers = tool._blockers(
        archive_bytes=121_690,
        hard_byte_ceilings=[178_000, 216_000],
        receiver_proof={},
        receiver_proof_requested=False,
        modelsize_integrity={"blockers": []},
        decoder_codec_resolution={"blockers": []},
    )
    assert "archive_bytes_exceed_tightest_hard_ceiling" not in blockers
    assert "receiver_proof_not_requested" in blockers
    assert "contest_cpu_cuda_exact_eval_not_executed" in blockers

    blockers = tool._blockers(
        archive_bytes=214_654,
        hard_byte_ceilings=[178_000, 216_000],
        receiver_proof={},
        receiver_proof_requested=True,
        modelsize_integrity={"blockers": []},
        decoder_codec_resolution={"blockers": []},
    )
    assert "archive_bytes_exceed_tightest_hard_ceiling" in blockers
    assert "receiver_proof_not_ready" in blockers

    blockers = tool._blockers(
        archive_bytes=121_690,
        hard_byte_ceilings=[178_000],
        receiver_proof={"runtime_consumption_proof_ready": True},
        receiver_proof_requested=True,
        modelsize_integrity={"blockers": []},
        decoder_codec_resolution={"blockers": []},
    )
    assert "archive_bytes_exceed_tightest_hard_ceiling" not in blockers
    assert "receiver_proof_not_ready" not in blockers
    assert "receiver_proof_not_requested" not in blockers


def test_summary_keeps_false_authority_fields() -> None:
    tool = _load_tool()
    report = {
        "schema": "hinerv_checkpoint_archive_export.v1",
        "candidate_id": "hinerv_np600_demo",
        "checkpoint_epoch": 9499,
        "checkpoint_state_kind": "ema",
        "archive_path": "/ssd/archive.zip",
        "archive_bytes": 121_690,
        "hard_byte_ceiling_enforced_by_export": 178_000,
        "hard_byte_ceiling_requested_by_candidate_or_startup": 178_000,
        "hard_byte_ceiling_measurement_bypass_enabled": False,
        "rate_byte_profile": {"profile_ready": True},
        "receiver_proof_ready": False,
        "blockers": ["full_video_scorer_replay_not_executed"],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }

    summary = tool._summary(report)

    assert summary["schema"] == "hinerv_checkpoint_archive_export.v1"
    assert summary["archive_bytes"] == 121_690
    assert summary["hard_byte_ceiling_enforced_by_export"] == 178_000
    assert summary["hard_byte_ceiling_measurement_bypass_enabled"] is False
    assert summary["score_claim"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["blockers"] == ["full_video_scorer_replay_not_executed"]
