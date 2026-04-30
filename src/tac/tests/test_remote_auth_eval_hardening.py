from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from tac.preflight import (
    MetaBugViolation,
    _scan_remote_lane_auth_eval_fragile_parse,
    check_launch_retry_wrapper_singleflight_and_signal_safe,
    check_remote_lane_auth_eval_json_adjudication,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_adjudicator():
    path = REPO_ROOT / "scripts" / "adjudicate_contest_auth_eval.py"
    spec = importlib.util.spec_from_file_location("_adjudicate_contest_auth_eval", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_adjudicator_uses_recomputed_json_score_not_human_formula(tmp_path: Path) -> None:
    adjudicator = _load_adjudicator()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"deterministic archive bytes")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()

    contest_json = tmp_path / "contest_auth_eval.json"
    contest_json.write_text(json.dumps({
        "final_score": 1.04,
        "score_recomputed_from_components": 1.0440481283330025,
        "avg_posenet_dist": 0.0034602,
        "avg_segnet_dist": 0.0040083,
        "archive_size_bytes": archive.stat().st_size,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": archive_sha,
            "device": "cuda",
            "gpu_model": "NVIDIA GeForce RTX 4090",
            "gpu_t4_match": False,
        },
    }))

    provenance = tmp_path / "provenance.json"
    result_copy = tmp_path / "RESULT_JSON"
    args = argparse.Namespace(
        contest_json=str(contest_json),
        provenance=str(provenance),
        archive=str(archive),
        result_copy=str(result_copy),
        baseline_score=1.05,
        baseline_archive_bytes=694074,
        predicted_band=[1.04, 1.05],
        hard_kill_above=1.05,
        delta_key="score_delta_vs_lane_g_v3",
        required_device="cuda",
        required_samples=600,
        max_sane_score=10.0,
    )

    result = adjudicator.adjudicate(args)

    assert result["score_recomputed"] == pytest.approx(1.0440481283330025)
    assert result["score_recomputed"] != 100.0
    assert result["hard_kill_triggered"] is False
    assert result["evidence_grade"] == "A score-grade"

    prov = json.loads(provenance.read_text())
    assert prov["contest_cuda_score_recomputed"] == pytest.approx(1.0440481283330025)
    assert prov["contest_cuda_score_reported_rounded"] == pytest.approx(1.04)
    assert prov["lane_status"] == "IN_PREDICTED_BAND"
    assert prov["contest_cuda_gpu_t4_match"] is False


def test_adjudicator_rejects_non_cuda_evidence(tmp_path: Path) -> None:
    adjudicator = _load_adjudicator()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    contest_json = tmp_path / "contest_auth_eval.json"
    contest_json.write_text(json.dumps({
        "final_score": 1.04,
        "score_recomputed_from_components": 1.044,
        "archive_size_bytes": archive.stat().st_size,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": archive_sha,
            "device": "cpu",
            "gpu_t4_match": False,
        },
    }))
    args = argparse.Namespace(
        contest_json=str(contest_json),
        provenance=str(tmp_path / "provenance.json"),
        archive=str(archive),
        result_copy=None,
        baseline_score=1.05,
        baseline_archive_bytes=None,
        predicted_band=[1.04, 1.05],
        hard_kill_above=1.05,
        delta_key="score_delta_vs_baseline",
        required_device="cuda",
        required_samples=600,
        max_sane_score=10.0,
    )
    with pytest.raises(SystemExit, match="expected 'cuda'"):
        adjudicator.adjudicate(args)


def test_preflight_catches_human_score_regex(tmp_path: Path) -> None:
    root = tmp_path
    scripts = root / "scripts"
    scripts.mkdir()
    bad = scripts / "remote_lane_bad.sh"
    bad.write_text(
        "#!/bin/bash\n"
        "python experiments/contest_auth_eval.py 2>&1 | tee auth_eval.log\n"
        "python - <<'PY'\n"
        "import re\n"
        "m = re.search(r'final[_ ]?score[\\s:=]+([0-9.]+)', open('auth_eval.log').read())\n"
        "PY\n"
    )
    violations = _scan_remote_lane_auth_eval_fragile_parse(bad, root)
    assert violations
    assert any("final[_ ]?score" in v for v in violations)


def test_preflight_catches_last_json_object_scrape(tmp_path: Path) -> None:
    root = tmp_path
    scripts = root / "scripts"
    scripts.mkdir()
    bad = scripts / "remote_lane_bad.sh"
    bad.write_text(
        "#!/bin/bash\n"
        "python experiments/contest_auth_eval.py 2>&1 | tee auth_eval.log\n"
        "grep -Eo '\\{.*\\}' auth_eval.log | tail -1 > RESULT_JSON\n"
    )
    violations = check_remote_lane_auth_eval_json_adjudication(
        repo_root=root, strict=False, verbose=False,
    )
    assert len(violations) == 1
    with pytest.raises(MetaBugViolation):
        check_remote_lane_auth_eval_json_adjudication(
            repo_root=root, strict=True, verbose=False,
        )


def test_live_remote_lane_scripts_avoid_fragile_auth_eval_parsers() -> None:
    violations = check_remote_lane_auth_eval_json_adjudication(
        repo_root=REPO_ROOT, strict=False, verbose=False,
    )
    assert violations == []


def test_launch_retry_wrapper_preflight_self_protection() -> None:
    violations = check_launch_retry_wrapper_singleflight_and_signal_safe(
        repo_root=REPO_ROOT, strict=False, verbose=False,
    )
    assert violations == []


def test_launch_retry_timeouts_cover_launcher_poll_windows(monkeypatch) -> None:
    path = REPO_ROOT / "scripts" / "launch_lane_with_retry.py"
    spec = importlib.util.spec_from_file_location("_launch_lane_with_retry", path)
    assert spec and spec.loader
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    assert launcher.PHASE2_WAIT_TIMEOUT_SECONDS >= 480
    assert launcher.PHASE2_SCP_TIMEOUT_SECONDS >= 600
    assert launcher.PHASE2_EXTRACT_TIMEOUT_SECONDS >= 120
    assert launcher.PHASE2_LAUNCH_TIMEOUT_SECONDS > 240

    calls: list[tuple[tuple[str, ...], int]] = []

    def fake_run_stage(cmd, timeout=300):
        calls.append((tuple(cmd), timeout))
        stage = cmd[2]
        if stage == "phase1":
            return 0, "INSTANCE_ID=123\n"
        if stage in {"phase2-wait", "phase2-scp", "phase2-extract"}:
            return 0, "ok\n"
        if stage == "phase2-launch":
            return 124, "TIMEOUT after 420s"
        raise AssertionError(cmd)

    destroyed: list[int] = []
    monkeypatch.setattr(launcher, "run_stage", fake_run_stage)
    monkeypatch.setattr(launcher, "destroy", destroyed.append)
    args = types.SimpleNamespace(
        lane_script="scripts/remote_lane_pfp16_stack.sh",
        label="lane_pfp16",
        max_dph=0.40,
        predicted_band=[1.04, 1.05],
        estimated_cost=0.50,
        allow_existing_label_prefix=True,
    )

    status, iid, log = launcher.attempt_dispatch(args, attempt=1)

    assert status == "unknown"
    assert iid == 123
    assert destroyed == []
    assert "UNKNOWN_REMOTE_STATE" in log
    assert calls[-1][1] == launcher.PHASE2_LAUNCH_TIMEOUT_SECONDS


def test_launch_retry_refuses_duplicate_live_label_prefix(monkeypatch) -> None:
    path = REPO_ROOT / "scripts" / "launch_lane_with_retry.py"
    spec = importlib.util.spec_from_file_location("_launch_lane_with_retry", path)
    assert spec and spec.loader
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    monkeypatch.setattr(
        launcher,
        "live_instances_with_label_prefix",
        lambda _label: ([{
            "id": 35905118,
            "label": "lane_sa_segmap_clone_2026-04-30_codex_a3",
            "actual_status": "running",
            "ssh_host": "ssh6.vast.ai",
            "ssh_port": 25118,
            "dph_total": 0.253,
        }], None),
    )
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(launcher, "run_stage", lambda cmd, timeout=300: calls.append(tuple(cmd)) or (0, "INSTANCE_ID=1\n"))
    args = types.SimpleNamespace(
        lane_script="scripts/remote_lane_sa_segmap_clone.sh",
        label="lane_sa_segmap_clone_2026-04-30_codex",
        max_dph=0.40,
        predicted_band=[0.40, 0.55],
        estimated_cost=1.00,
        allow_existing_label_prefix=False,
    )

    status, iid, log = launcher.attempt_dispatch(args, attempt=1)

    assert status == "unknown"
    assert iid == 35905118
    assert calls == []
    assert "UNKNOWN_EXISTING_LABEL_PREFIX" in log
    assert "No duplicate retry launched" in log


def test_launch_retry_run_stage_timeout_kills_stage_group() -> None:
    path = REPO_ROOT / "scripts" / "launch_lane_with_retry.py"
    spec = importlib.util.spec_from_file_location("_launch_lane_with_retry", path)
    assert spec and spec.loader
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    rc, out = launcher.run_stage(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        timeout=1,
    )

    assert rc == 124
    assert "TIMEOUT after 1s" in out
