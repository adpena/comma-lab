"""Tests for tools/measure_cross_machine_variance.py.

Covers F6 fix per A1 PR Council Round 1: cross-machine variance probe
(plan + harvest + statistics). The tool DOES NOT auto-dispatch — these
tests verify the planner + harvest math + refusal classes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_variance_module():
    spec = importlib.util.spec_from_file_location(
        "measure_cross_machine_variance_under_test",
        REPO_ROOT / "tools" / "measure_cross_machine_variance.py",
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["measure_cross_machine_variance_under_test"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


variance_mod = _load_variance_module()
build_plan = variance_mod.build_plan
compute_statistics = variance_mod.compute_statistics
harvest = variance_mod.harvest
harvest_eval_json = variance_mod.harvest_eval_json
VariancePlanRefused = variance_mod.VariancePlanRefused
ALLOWED_RUNNERS = variance_mod.ALLOWED_RUNNERS
FORBIDDEN_RUNNERS_REASONS = variance_mod.FORBIDDEN_RUNNERS_REASONS
VARIANCE_PLAN_SCHEMA = variance_mod.VARIANCE_PLAN_SCHEMA
VARIANCE_HARVEST_SCHEMA = variance_mod.VARIANCE_HARVEST_SCHEMA


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def archive(tmp_path: Path):
    p = tmp_path / "archive.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("0.bin", b"variance-probe-archive-bytes")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    return {"path": p, "sha": sha, "tmp": tmp_path}


@pytest.fixture
def inflate_sh(tmp_path: Path):
    p = tmp_path / "inflate.sh"
    p.write_text("#!/bin/sh\nset -euo pipefail\necho ok\n", encoding="utf-8")
    p.chmod(0o755)
    return p


def _eval_json(score: float, sha: str, hardware: str) -> dict[str, Any]:
    return {
        "archive_sha256": sha,
        "canonical_score_recomputed": score,
        "n_samples": 600,
        "device": "cpu",
        "hardware": hardware,
        "lane_tag": "[contest-CPU]",
        "score_axis": "contest_cpu",
        "evidence_semantics": "public_leaderboard_cpu_reproduction",
        "score_claim": False,
        "cpu_leaderboard_reproduction_eligible": True,
    }


# ── 1. build_plan happy path ──────────────────────────────────────────────


def test_build_plan_happy_path(archive, inflate_sh):
    plan = build_plan(
        archive_path=archive["path"],
        inflate_sh_path=inflate_sh,
        n_runs=5,
        runners=["gha-ubuntu-latest", "modal-cpu-shared"],
    )
    assert plan.archive_sha256 == archive["sha"]
    assert plan.n_runs_per_runner == 5
    assert plan.runners == ["gha-ubuntu-latest", "modal-cpu-shared"]
    assert len(plan.plan_rows) == 10  # 2 runners × 5 runs
    # GHA cost per run is $0; Modal CPU-shared $0.10 → total $0.50.
    assert plan.estimated_total_cost_usd == pytest.approx(0.50)
    payload = plan.to_json()
    assert payload["schema"] == VARIANCE_PLAN_SCHEMA
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert "f6_council_round_1_critical_remediation" in payload["claude_md_compliance_tags"]


def test_build_plan_dispatch_command_stubs_present(archive, inflate_sh):
    plan = build_plan(
        archive_path=archive["path"],
        inflate_sh_path=inflate_sh,
        n_runs=2,
        runners=["gha-ubuntu-latest", "modal-cpu-isolated"],
    )
    gha_row = next(r for r in plan.plan_rows if r.runner == "gha-ubuntu-latest")
    assert "gh workflow run contest_cpu_eval.yml" in gha_row.dispatch_command_stub
    assert "-f archive_path=" in gha_row.dispatch_command_stub
    modal_row = next(r for r in plan.plan_rows if r.runner == "modal-cpu-isolated")
    assert "modal run --detach experiments/modal_auth_eval_cpu.py" in modal_row.dispatch_command_stub
    # Every row carries the archive sha for apples-to-apples discipline.
    assert archive["sha"][:12] in gha_row.dispatch_command_stub
    assert archive["sha"][:12] in modal_row.dispatch_command_stub


def test_build_plan_output_artifact_relpath_includes_runner_and_index(
    archive, inflate_sh
):
    plan = build_plan(
        archive_path=archive["path"],
        inflate_sh_path=inflate_sh,
        n_runs=3,
        runners=["gha-ubuntu-latest"],
    )
    paths = [r.output_artifact_relpath for r in plan.plan_rows]
    assert any("run_001.json" in p for p in paths)
    assert any("run_002.json" in p for p in paths)
    assert any("run_003.json" in p for p in paths)
    assert all("gha-ubuntu-latest" in p for p in paths)


# ── 2. build_plan refusal classes ─────────────────────────────────────────


def test_build_plan_refuses_on_macos_runner(archive, inflate_sh):
    with pytest.raises(VariancePlanRefused) as excinfo:
        build_plan(
            archive_path=archive["path"],
            inflate_sh_path=inflate_sh,
            n_runs=3,
            runners=["macos-arm64"],
        )
    assert excinfo.value.reason_class == "forbidden_runner"
    assert "Catalog #192" in str(excinfo.value)


def test_build_plan_refuses_on_mps_runner(archive, inflate_sh):
    with pytest.raises(VariancePlanRefused) as excinfo:
        build_plan(
            archive_path=archive["path"],
            inflate_sh_path=inflate_sh,
            n_runs=3,
            runners=["mps"],
        )
    assert excinfo.value.reason_class == "forbidden_runner"


def test_build_plan_refuses_on_unknown_runner(archive, inflate_sh):
    with pytest.raises(VariancePlanRefused) as excinfo:
        build_plan(
            archive_path=archive["path"],
            inflate_sh_path=inflate_sh,
            n_runs=3,
            runners=["something-weird"],
        )
    assert excinfo.value.reason_class == "unknown_runner"


def test_build_plan_refuses_when_no_runners(archive, inflate_sh):
    with pytest.raises(VariancePlanRefused) as excinfo:
        build_plan(
            archive_path=archive["path"],
            inflate_sh_path=inflate_sh,
            n_runs=3,
            runners=[],
        )
    assert excinfo.value.reason_class == "no_runners"


def test_build_plan_refuses_on_invalid_n_runs(archive, inflate_sh):
    with pytest.raises(VariancePlanRefused) as excinfo:
        build_plan(
            archive_path=archive["path"],
            inflate_sh_path=inflate_sh,
            n_runs=0,
            runners=["gha-ubuntu-latest"],
        )
    assert excinfo.value.reason_class == "invalid_n_runs"


def test_build_plan_refuses_on_archive_missing(tmp_path, inflate_sh):
    with pytest.raises(VariancePlanRefused) as excinfo:
        build_plan(
            archive_path=tmp_path / "no-archive.zip",
            inflate_sh_path=inflate_sh,
            n_runs=3,
            runners=["gha-ubuntu-latest"],
        )
    assert excinfo.value.reason_class == "archive_missing"


def test_build_plan_refuses_on_inflate_missing(archive, tmp_path):
    with pytest.raises(VariancePlanRefused) as excinfo:
        build_plan(
            archive_path=archive["path"],
            inflate_sh_path=tmp_path / "no-inflate.sh",
            n_runs=3,
            runners=["gha-ubuntu-latest"],
        )
    assert excinfo.value.reason_class == "inflate_sh_missing"


# ── 3. compute_statistics math ────────────────────────────────────────────


def test_compute_statistics_zero_variance():
    """If every score is identical, std == 0 and noise floor == 0."""
    scores = {"runner_a": [0.5, 0.5, 0.5]}
    stats = compute_statistics(scores)
    assert stats["per_runner"]["runner_a"]["mean"] == pytest.approx(0.5)
    assert stats["per_runner"]["runner_a"]["std"] == pytest.approx(0.0)
    assert stats["empirical_noise_floor"] == pytest.approx(0.0)


def test_compute_statistics_per_runner_mean_and_std():
    scores = {
        "gha-ubuntu-latest": [0.193, 0.194, 0.192, 0.193, 0.194],
        "modal-cpu-shared": [0.192, 0.193, 0.193, 0.192, 0.193],
    }
    stats = compute_statistics(scores)
    gha = stats["per_runner"]["gha-ubuntu-latest"]
    assert gha["n"] == 5
    assert 0.192 < gha["mean"] < 0.195
    assert gha["std"] > 0
    assert gha["ci_95_low"] < gha["mean"] < gha["ci_95_high"]


def test_compute_statistics_pooled_std_is_finite_with_two_runners():
    scores = {"runner_a": [0.10, 0.11], "runner_b": [0.20, 0.21]}
    stats = compute_statistics(scores)
    assert stats["cross_runner_pooled_std"] > 0
    assert stats["cross_runner_total_std"] > stats["cross_runner_pooled_within_std"]
    # noise floor = 2 × total cross-runner std, including stable runner offsets.
    assert stats["empirical_noise_floor"] == pytest.approx(
        2 * stats["cross_runner_total_std"]
    )


def test_compute_statistics_counts_stable_between_runner_offset_as_noise():
    scores = {"runner_a": [0.10, 0.10, 0.10], "runner_b": [0.20, 0.20, 0.20]}
    stats = compute_statistics(scores)
    assert stats["cross_runner_pooled_within_std"] == pytest.approx(0.0)
    assert stats["cross_runner_total_std"] > 0.0
    assert stats["empirical_noise_floor"] > 0.0


def test_compute_statistics_empty_input():
    stats = compute_statistics({})
    assert stats["cross_runner_grand_mean"] is None
    assert stats["cross_runner_n_total"] == 0


def test_compute_statistics_single_run_per_runner():
    """n=1 yields std=0 (degenerate) and infinite t-CI; tool should return finite reasonable values."""
    scores = {"a": [0.193]}
    stats = compute_statistics(scores)
    assert stats["per_runner"]["a"]["std"] == 0.0
    assert stats["per_runner"]["a"]["ci_95_low"] == 0.193
    assert stats["per_runner"]["a"]["ci_95_high"] == 0.193


# ── 4. harvest_eval_json ──────────────────────────────────────────────────


def test_harvest_eval_json_extracts_score_and_runner(tmp_path, archive):
    p = tmp_path / "run.json"
    p.write_text(
        json.dumps(_eval_json(0.193, archive["sha"], "github-actions-ubuntu-latest")),
        encoding="utf-8",
    )
    row = harvest_eval_json(p, archive["sha"])
    assert row["score_value"] == pytest.approx(0.193)
    assert row["runner"] == "gha-ubuntu-latest"


def test_harvest_eval_json_refuses_on_sha_mismatch(tmp_path, archive):
    p = tmp_path / "run.json"
    p.write_text(
        json.dumps(_eval_json(0.193, "0" * 64, "github-actions-ubuntu-latest")),
        encoding="utf-8",
    )
    with pytest.raises(VariancePlanRefused) as excinfo:
        harvest_eval_json(p, archive["sha"])
    assert excinfo.value.reason_class == "archive_sha_mismatch"


def test_harvest_eval_json_refuses_on_missing_score(tmp_path, archive):
    p = tmp_path / "run.json"
    payload = _eval_json(0.193, archive["sha"], "github-actions-ubuntu-latest")
    payload.pop("canonical_score_recomputed")
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(VariancePlanRefused) as excinfo:
        harvest_eval_json(p, archive["sha"])
    assert excinfo.value.reason_class == "score_value_missing"


def test_harvest_eval_json_refuses_non_contest_cpu_metadata(tmp_path, archive):
    p = tmp_path / "run.json"
    payload = _eval_json(0.193, archive["sha"], "github-actions-ubuntu-latest")
    payload["lane_tag"] = "[diagnostic-auth-eval]"
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(VariancePlanRefused) as excinfo:
        harvest_eval_json(p, archive["sha"])
    assert excinfo.value.reason_class == "non_contest_cpu_eval_json"


def test_harvest_eval_json_refuses_runner_not_in_plan(tmp_path, archive):
    p = tmp_path / "run.json"
    p.write_text(
        json.dumps(_eval_json(0.193, archive["sha"], "modal-cpu-shared")),
        encoding="utf-8",
    )
    with pytest.raises(VariancePlanRefused) as excinfo:
        harvest_eval_json(
            p,
            archive["sha"],
            planned_runners={"gha-ubuntu-latest"},
        )
    assert excinfo.value.reason_class == "runner_not_in_plan"


def test_harvest_eval_json_refuses_on_missing_file(tmp_path, archive):
    with pytest.raises(VariancePlanRefused) as excinfo:
        harvest_eval_json(tmp_path / "no-such.json", archive["sha"])
    assert excinfo.value.reason_class == "eval_json_missing"


# ── 5. harvest happy path ─────────────────────────────────────────────────


def test_harvest_happy_path(tmp_path, archive, inflate_sh):
    plan = build_plan(
        archive_path=archive["path"],
        inflate_sh_path=inflate_sh,
        n_runs=3,
        runners=["gha-ubuntu-latest"],
    )

    eval_paths: list[Path] = []
    for i, score in enumerate([0.1932, 0.1934, 0.1933]):
        p = tmp_path / f"run_{i + 1}.json"
        p.write_text(
            json.dumps(_eval_json(score, archive["sha"], "github-actions-ubuntu-latest")),
            encoding="utf-8",
        )
        eval_paths.append(p)

    report = harvest(plan=plan.to_json(), eval_json_paths=eval_paths)
    assert report["schema"] == VARIANCE_HARVEST_SCHEMA
    assert report["n_eval_jsons_harvested"] == 3
    assert report["plan_archive_sha256"] == archive["sha"]
    assert report["score_claim"] is False
    stats = report["statistics"]
    assert stats["cross_runner_n_total"] == 3
    assert stats["empirical_noise_floor"] >= 0


def test_harvest_refuses_on_plan_missing_archive_sha(tmp_path, archive):
    p = tmp_path / "run.json"
    p.write_text(
        json.dumps(_eval_json(0.193, archive["sha"], "github-actions")),
        encoding="utf-8",
    )
    with pytest.raises(VariancePlanRefused) as excinfo:
        harvest(plan={}, eval_json_paths=[p])
    assert excinfo.value.reason_class == "plan_missing_archive_sha"
