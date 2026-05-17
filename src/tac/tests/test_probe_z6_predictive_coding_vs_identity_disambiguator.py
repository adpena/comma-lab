# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "probe_z6_predictive_coding_vs_identity_disambiguator.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("z6_disambiguator_tool", TOOL_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _stats(*, identity: bool, loss: float, archive_bytes: int = 1024) -> dict[str, object]:
    return {
        "lane_id": "lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516",
        "substrate_tag": "time_traveler_l5_z6",
        "smoke": True,
        "requested_epochs": 3,
        "epochs": 3,
        "smoke_epoch_cap": 3,
        "final_loss_proxy": loss,
        "final_recon": loss / 2.0,
        "final_residual": loss / 4.0,
        "archive_bytes": archive_bytes,
        "lambda_residual_entropy": 1.0,
        "predictor_kernel_size": 3,
        "paired_control_initialization": (
            "shared_modules_seed_order_matched_v2"
        ),
        "smoke_target_mode": "real-video",
        "smoke_ego_motion_mode": "ramp",
        "ego_motion_nonzero_fraction": 1.0,
        "ego_motion_l2": 2.714,
        "identity_predictor": identity,
        "score_claim_valid": False,
        "evidence_grade": "smoke-no-scorer",
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
    }


def test_z6_disambiguator_plan_is_fail_closed_and_paired() -> None:
    tool = _load_tool()

    payload = tool.build_plan_payload(epochs=3, device="cpu", seed=7)

    assert payload["schema"] == tool.SCHEMA
    assert payload["verdict"] == "pending_paired_smoke_stats"
    assert payload["paired_control_initialization"] == (
        "shared_modules_seed_order_matched_v2"
    )
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["rank_or_kill_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["ready_for_paid_dispatch"] is False
    assert payload["paradigm_claim_allowed"] is False
    commands = payload["paired_smoke_commands"]
    assert [row["identity_predictor"] for row in commands] == [False, True]
    assert "--smoke" in commands[0]["command"]
    assert "--smoke-ego-motion-mode" in commands[0]["command"]
    assert "--smoke-target-mode" in commands[0]["command"]
    assert "real-video" in commands[0]["command"]
    assert "real-video" in commands[1]["command"]
    assert "--identity-predictor" not in commands[0]["command"]
    assert "--identity-predictor" in commands[1]["command"]
    assert "no_contest_cpu_cuda_pair" in payload["blockers"]


def test_z6_disambiguator_compares_paired_smoke_stats(tmp_path: Path) -> None:
    tool = _load_tool()
    full_path = tmp_path / "experiments/results/z6/full/stats.json"
    identity_path = tmp_path / "experiments/results/z6/identity/stats.json"
    full_path.parent.mkdir(parents=True)
    identity_path.parent.mkdir(parents=True)
    full_path.write_text(
        json.dumps(_stats(identity=False, loss=0.10, archive_bytes=1200)),
        encoding="utf-8",
    )
    identity_path.write_text(
        json.dumps(_stats(identity=True, loss=0.15, archive_bytes=1000)),
        encoding="utf-8",
    )

    payload = tool.evaluate_stats_pair(
        full_stats_path=full_path,
        identity_stats_path=identity_path,
        repo_root=tmp_path,
    )

    assert payload["verdict"] == "full_film_predictor_proxy_lower_loss"
    assert payload["evidence_grade"] == "smoke_proxy_real_video_pair_no_scorer"
    assert payload["proxy_preferred_mode"] == "full_film_predictor"
    assert payload["score_claim"] is False
    assert payload["paradigm_claim_allowed"] is False
    assert payload["deltas"]["identity_minus_full_loss_proxy"] == pytest.approx(0.05)
    assert payload["deltas"]["full_minus_identity_archive_bytes"] == 200
    assert payload["result_review"]["classification"] == "real_video_smoke_proxy_only"
    assert payload["result_review"]["paired_control_initialization"] == (
        "shared_modules_seed_order_matched_v2"
    )
    assert "smoke_proxy_real_video_no_scorer" in payload["blockers"]
    assert [row["mode"] for row in payload["source_stats"]] == [
        "full_film_predictor",
        "identity_predictor",
    ]
    assert payload["source_stats"][0]["stats_payload"]["identity_predictor"] is False
    assert payload["source_stats"][1]["stats_payload"]["identity_predictor"] is True
    assert payload["source_stats"][0]["paired_control_initialization"] == (
        "shared_modules_seed_order_matched_v2"
    )


def test_z6_disambiguator_rejects_authoritative_or_mismatched_stats(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    full_path = tmp_path / "full.json"
    identity_path = tmp_path / "identity.json"
    full = _stats(identity=False, loss=0.10)
    identity = _stats(identity=True, loss=0.15)
    identity["score_claim_valid"] = True
    full_path.write_text(json.dumps(full), encoding="utf-8")
    identity_path.write_text(json.dumps(identity), encoding="utf-8")

    with pytest.raises(ValueError, match="score_claim_valid must be false"):
        tool.evaluate_stats_pair(
            full_stats_path=full_path,
            identity_stats_path=identity_path,
            repo_root=tmp_path,
        )

    identity["score_claim_valid"] = False
    identity["paired_control_initialization"] = "stale_unmatched_seed_order"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(ValueError, match="paired_control_initialization"):
        tool.evaluate_stats_pair(
            full_stats_path=full_path,
            identity_stats_path=identity_path,
            repo_root=tmp_path,
        )

    identity["paired_control_initialization"] = (
        "shared_modules_seed_order_matched_v2"
    )
    identity["predictor_kernel_size"] = 5
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(ValueError, match="must match predictor_kernel_size"):
        tool.evaluate_stats_pair(
            full_stats_path=full_path,
            identity_stats_path=identity_path,
            repo_root=tmp_path,
        )


def test_z6_disambiguator_rejects_wrong_substrate(tmp_path: Path) -> None:
    tool = _load_tool()
    full_path = tmp_path / "full.json"
    identity_path = tmp_path / "identity.json"
    full = _stats(identity=False, loss=0.10)
    identity = _stats(identity=True, loss=0.15)
    full["substrate_tag"] = "time_traveler_l5_z7"
    full_path.write_text(json.dumps(full), encoding="utf-8")
    identity_path.write_text(json.dumps(identity), encoding="utf-8")

    with pytest.raises(ValueError, match="substrate_tag"):
        tool.evaluate_stats_pair(
            full_stats_path=full_path,
            identity_stats_path=identity_path,
            repo_root=tmp_path,
        )


def test_z6_disambiguator_cli_writes_plan_json_and_markdown(tmp_path: Path) -> None:
    output_json = tmp_path / ".omx/research/z6_probe.json"
    output_md = tmp_path / ".omx/research/z6_probe.md"

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--repo-root",
            str(tmp_path),
            "--output-json",
            str(output_json.relative_to(tmp_path)),
            "--output-md",
            str(output_md.relative_to(tmp_path)),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["verdict"] == "pending_paired_smoke_stats"
    assert output_md.read_text(encoding="utf-8").startswith(
        "# L5 v2 Z6 identity-predictor disambiguator"
    )
    assert "score_claim=false" in proc.stdout
