from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

TOOL_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "harvest_hinerv_bytecap_successor.py"
)


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "harvest_hinerv_bytecap_successor", TOOL_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_checkpoint_meta(
    root: Path,
    *,
    name: str,
    epoch: int,
    role: str = "periodic",
    loss: float = 1.0,
) -> Path:
    state = root / f"{name}.ema.state.npsd"
    live = root / f"{name}.live.state.npsd"
    state.write_bytes(f"ema:{name}".encode("ascii"))
    live.write_bytes(f"live:{name}".encode("ascii"))
    meta = root / f"{name}.meta.json"
    meta.write_text(
        json.dumps(
            {
                "global_epoch": epoch,
                "checkpoint_role": role,
                "loss": loss,
                "ema_shadow_state_path": state.as_posix(),
                "live_state_path": live.as_posix(),
            }
        ),
        encoding="utf-8",
    )
    return meta


def test_select_checkpoint_uses_runner_checkpoint_selection(tmp_path: Path) -> None:
    tool = _load_tool()
    ckpt_dir = tmp_path / "run" / "hi_nerv_mlx_training" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    final_meta = _write_checkpoint_meta(
        ckpt_dir, name="final_epoch000010", epoch=10, role="final", loss=3.0
    )
    best_meta = _write_checkpoint_meta(
        ckpt_dir, name="best_epoch000007", epoch=7, role="best", loss=1.0
    )
    report = {
        "training_artifact": {
            "checkpoint_selection": {
                "selected_meta_path": best_meta.as_posix(),
                "final_meta_path": final_meta.as_posix(),
            }
        }
    }

    selected = tool.select_checkpoint_meta(
        run_dir=tmp_path / "run",
        runner_report=report,
        state_kind="ema",
    )

    assert selected["meta_path"] == best_meta.resolve(strict=False).as_posix()
    assert selected["source"] == "training_artifact.checkpoint_selection.selected_meta_path"
    assert selected["global_epoch"] == 7
    assert selected["checkpoint_role"] == "best"


def test_select_checkpoint_falls_back_to_latest_filesystem_meta(tmp_path: Path) -> None:
    tool = _load_tool()
    ckpt_dir = tmp_path / "run" / "hi_nerv_mlx_training" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    _write_checkpoint_meta(ckpt_dir, name="epoch000009", epoch=9)
    final_meta = _write_checkpoint_meta(
        ckpt_dir, name="final_epoch000010", epoch=10, role="final"
    )

    selected = tool.select_checkpoint_meta(
        run_dir=tmp_path / "run",
        runner_report={},
        state_kind="ema",
    )

    assert selected["meta_path"] == final_meta.resolve(strict=False).as_posix()
    assert selected["source"] == "filesystem_latest"
    assert selected["global_epoch"] == 10


def test_validate_export_proof_fails_closed_when_receiver_or_prefilter_missing(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    meta = _write_checkpoint_meta(tmp_path, name="final_epoch000001", epoch=1)
    state = Path(json.loads(meta.read_text())["ema_shadow_state_path"])

    proof = tool.validate_export_proof(
        {
            "schema": "hinerv_checkpoint_archive_export.v1",
            "archive_path": archive.as_posix(),
            "archive_sha256": tool.sha256_file(archive),
            "archive_bytes": archive.stat().st_size,
            "receiver_proof_ready": False,
            "receiver_proof_passed": False,
            "receiver_contract_satisfied": False,
            "receiver_closed": False,
            "receiver_proof_path": None,
            "local_mlx_prefilter_written": False,
            "local_mlx_prefilter_profile_path": None,
            "checkpoint_meta_path": meta.as_posix(),
            "checkpoint_state_path": state.as_posix(),
        }
    )

    assert proof["proof_ready"] is False
    assert "receiver_proof_not_ready" in proof["blockers"]
    assert "runtime_consumption_proof_not_passed" in proof["blockers"]
    assert "receiver_contract_not_satisfied" in proof["blockers"]
    assert "receiver_closed_not_satisfied" in proof["blockers"]
    assert "full_video_mlx_prefilter_profile_not_written" in proof["blockers"]
    assert tool.terminal_status_for_proof(proof) == "failed_self_harvest_proof_missing"


def test_overcap_receiver_proof_profile_closes_as_completed_measurement(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    receiver = tmp_path / "receiver_proof.json"
    receiver.write_text('{"runtime_consumption_proof_ready":true}\n', encoding="utf-8")
    profile = tmp_path / "local_mlx_prefilter_profile.json"
    profile.write_text('{"written":true}\n', encoding="utf-8")
    profile_sha = tool.sha256_file(profile)
    meta = _write_checkpoint_meta(tmp_path, name="final_epoch000001", epoch=1)
    state = Path(json.loads(meta.read_text())["ema_shadow_state_path"])

    proof = tool.validate_export_proof(
        {
            "schema": "hinerv_checkpoint_archive_export.v1",
            "archive_path": archive.as_posix(),
            "archive_sha256": tool.sha256_file(archive),
            "archive_bytes": archive.stat().st_size,
            "hard_byte_ceiling_requested_by_candidate_or_startup": 5,
            "receiver_proof_ready": True,
            "receiver_proof_passed": True,
            "receiver_contract_satisfied": True,
            "receiver_closed": True,
            "receiver_proof_path": receiver.as_posix(),
            "receiver_proof_sha256": tool.sha256_file(receiver),
            "local_mlx_prefilter_written": True,
            "local_mlx_prefilter_profile_path": profile.as_posix(),
            "local_mlx_prefilter_profile": {
                "written": True,
                "profile_sha256": profile_sha,
                "blockers": [
                    "mlx_local_replay_not_contest_auth_axis",
                    "hinerv_receiver_raw_cache_prefilter_false_authority",
                ],
                "cache_quality_gate": {
                    "fit_gate_passed": True,
                    "candidate_cache_nondegenerate": True,
                    "verdict": "CACHE_INPUTS_NONDEGENERATE_LOCAL_ONLY",
                    "blockers": [],
                },
            },
            "blockers": [
                "macos_mlx_checkpoint_export_false_authority",
                "contest_cpu_cuda_exact_eval_not_executed",
                "mlx_local_replay_not_contest_auth_axis",
                "hinerv_receiver_raw_cache_prefilter_false_authority",
                "archive_bytes_exceed_tightest_hard_ceiling",
                "hard_byte_ceiling_export_bypassed_for_measurement",
            ],
            "checkpoint_meta_path": meta.as_posix(),
            "checkpoint_state_path": state.as_posix(),
        }
    )

    assert proof["proof_ready"] is True
    assert proof["archive_overrun_bytes"] == archive.stat().st_size - 5
    assert proof["verdict"] == "overcap_receiver_proof_profiled"
    assert (
        tool.terminal_status_for_proof(proof)
        == "completed_overcap_measurement_receiver_proof_profiled"
    )


def test_validate_export_proof_requires_receiver_contract_satisfied(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    receiver = tmp_path / "receiver_proof.json"
    receiver.write_text('{"runtime_consumption_proof_ready":true}\n', encoding="utf-8")
    profile = tmp_path / "local_mlx_prefilter_profile.json"
    profile.write_text('{"written":true}\n', encoding="utf-8")
    meta = _write_checkpoint_meta(tmp_path, name="final_epoch000001", epoch=1)
    state = Path(json.loads(meta.read_text())["ema_shadow_state_path"])

    proof = tool.validate_export_proof(
        {
            "schema": "hinerv_checkpoint_archive_export.v1",
            "archive_path": archive.as_posix(),
            "archive_sha256": tool.sha256_file(archive),
            "archive_bytes": archive.stat().st_size,
            "receiver_proof_ready": True,
            "receiver_proof_passed": False,
            "receiver_contract_satisfied": False,
            "receiver_closed": False,
            "receiver_proof_path": receiver.as_posix(),
            "receiver_proof_sha256": tool.sha256_file(receiver),
            "local_mlx_prefilter_written": True,
            "local_mlx_prefilter_profile_path": profile.as_posix(),
            "local_mlx_prefilter_profile": {
                "written": True,
                "profile_sha256": tool.sha256_file(profile),
                "cache_quality_gate": {
                    "fit_gate_passed": True,
                    "candidate_cache_nondegenerate": True,
                    "verdict": "CACHE_INPUTS_NONDEGENERATE_LOCAL_ONLY",
                    "blockers": [],
                },
                "blockers": [
                    "mlx_local_replay_not_contest_auth_axis",
                    "hinerv_receiver_raw_cache_prefilter_false_authority",
                ],
            },
            "checkpoint_meta_path": meta.as_posix(),
            "checkpoint_state_path": state.as_posix(),
        }
    )

    assert proof["proof_ready"] is False
    assert "runtime_consumption_proof_not_passed" in proof["blockers"]
    assert "receiver_contract_not_satisfied" in proof["blockers"]
    assert "receiver_closed_not_satisfied" in proof["blockers"]
    assert tool.terminal_status_for_proof(proof) == "failed_self_harvest_proof_missing"


def test_validate_export_proof_rejects_prefilter_quality_blocker(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    receiver = tmp_path / "receiver_proof.json"
    receiver.write_text('{"runtime_consumption_proof_ready":true}\n', encoding="utf-8")
    profile = tmp_path / "local_mlx_prefilter_profile.json"
    profile.write_text('{"written":true}\n', encoding="utf-8")
    meta = _write_checkpoint_meta(tmp_path, name="final_epoch000001", epoch=1)
    state = Path(json.loads(meta.read_text())["ema_shadow_state_path"])

    proof = tool.validate_export_proof(
        {
            "schema": "hinerv_checkpoint_archive_export.v1",
            "archive_path": archive.as_posix(),
            "archive_sha256": tool.sha256_file(archive),
            "archive_bytes": archive.stat().st_size,
            "receiver_proof_ready": True,
            "receiver_proof_passed": True,
            "receiver_contract_satisfied": True,
            "receiver_closed": True,
            "receiver_proof_path": receiver.as_posix(),
            "receiver_proof_sha256": tool.sha256_file(receiver),
            "local_mlx_prefilter_written": True,
            "local_mlx_prefilter_profile_path": profile.as_posix(),
            "local_mlx_prefilter_profile": {
                "written": True,
                "profile_sha256": tool.sha256_file(profile),
                "cache_quality_gate": {
                    "fit_gate_passed": False,
                    "candidate_cache_nondegenerate": False,
                    "verdict": "FIT_OR_SCALE_FAILURE",
                    "blockers": ["candidate_posenet_yuv6_cache_degenerate"],
                },
                "blockers": [
                    "mlx_local_replay_not_contest_auth_axis",
                    "hinerv_receiver_raw_cache_prefilter_false_authority",
                ],
            },
            "blockers": [
                "contest_cpu_cuda_exact_eval_not_executed",
                "candidate_posenet_yuv6_cache_degenerate",
            ],
            "checkpoint_meta_path": meta.as_posix(),
            "checkpoint_state_path": state.as_posix(),
        }
    )

    assert proof["proof_ready"] is False
    assert "candidate_posenet_yuv6_cache_degenerate" in proof["blockers"]
    assert "mlx_prefilter_cache_quality_gate_not_passed" in proof["blockers"]
    assert (
        "mlx_prefilter_cache_quality_gate_degenerate_candidate_cache"
        in proof["blockers"]
    )
    assert "mlx_prefilter_cache_quality_verdict:FIT_OR_SCALE_FAILURE" in proof[
        "blockers"
    ]
    assert tool.terminal_status_for_proof(proof) == "failed_self_harvest_proof_missing"


def test_require_path_under_ssd_roots_rejects_local_path(tmp_path: Path) -> None:
    tool = _load_tool()

    with pytest.raises(tool.HarvestError) as exc:
        tool._require_path_under_ssd_roots(
            tmp_path / "run",
            roots=[Path("/Volumes/VertigoDataTier/pact")],
        )

    assert exc.value.blockers == ("path_not_ssd_backed",)


def test_maybe_close_dispatch_claim_skips_already_terminal(tmp_path: Path) -> None:
    tool = _load_tool()
    claims = tmp_path / "claims.md"
    claims.write_text(
        "\n".join(
            [
                "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |",
                "|---|---|---|---|---|---|---|---|",
                "| 2026-06-04T13:12:34Z | codex:gpt-5 | lane_x | local_mlx_tmux | job_x |  | completed_overcap_measurement_receiver_proof_profiled | already closed |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        no_close_claim=False,
        claims_path=claims,
        lane_id="lane_x",
        instance_job_id="job_x",
        platform="local_mlx_tmux",
        agent="codex:gpt-5",
        claim_ttl_hours=168.0,
    )

    result = tool.maybe_close_dispatch_claim(
        args=args,
        status="completed_overcap_measurement_receiver_proof_profiled",
        verdict={},
    )

    assert result["closed"] is False
    assert result["already_terminal"] is True
