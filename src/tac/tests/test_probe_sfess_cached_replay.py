# SPDX-License-Identifier: MIT
"""System-boundary canaries for the zero-scorer SFESS replay probe."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tac.sfess_cached_replay import SFESSError, load_cached_objective_jsonl

REPO = Path(__file__).resolve().parents[3]
PROBE_PATH = REPO / "tools/probe_sfess_cached_replay.py"


def _load_probe() -> Any:
    spec = importlib.util.spec_from_file_location("probe_sfess_cached_replay_test", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_custody_policy_and_registered_baseline_surface_are_pinned() -> None:
    probe = _load_probe()
    source, baselines = probe._load_and_verify_source_receipt()
    policy = probe._build_policy()
    policy.compile()
    table = load_cached_objective_jsonl(probe.TABLE)

    assert source["score_claim"] is False
    assert policy.research_only is True
    assert policy.produces_costate is False
    assert policy.live_gradient_fallback == "full_teacher"
    assert table.source_sha256 == probe.PINNED_UGC64_SHA256
    assert set(baselines) == {
        "exact_enumeration",
        "one_plus_one_es",
        "ugc",
        "disarm",
        "rloo",
    }
    assert baselines["exact_enumeration"]["best_s"] == pytest.approx(0.19080359202934188)
    assert probe.SOURCE_VIDEO.stat().st_size == 37_545_489
    assert hashlib.sha256(probe.SOURCE_VIDEO.read_bytes()).hexdigest() == (
        "2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9"
    )
    assert policy.source_video_custody.kind == "source_video"
    assert policy.source_video_custody.sha256 == policy.objective_context.source_video_sha256


def test_execution_source_custody_covers_loaded_sfess_code_and_base_commit_state() -> None:
    probe = _load_probe()
    custody = probe._execution_source_custody()
    by_path = {row["path"]: row for row in custody["modules"].values()}
    required = {
        "tools/probe_sfess_cached_replay.py",
        "src/tac/sfess_cached_replay.py",
        "src/tac/witness_dsl/sfess_cached_replay_policy.py",
    }

    assert required <= set(by_path)
    assert custody["module_alias_count"] == len(custody["modules"])
    assert custody["unique_source_count"] == len(
        {row["path"] for row in custody["modules"].values()}
    )
    encoded = json.dumps(
        custody["modules"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    assert custody["tree_sha256"] == hashlib.sha256(encoded).hexdigest()
    for relative in required:
        source = REPO / relative
        assert by_path[relative]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert by_path[relative]["bytes"] == source.stat().st_size
    git_state = probe._execution_source_git_state(custody)
    assert git_state["base_git_head"] == probe._git_head()
    probe_row = git_state["files"]["tools/probe_sfess_cached_replay.py"]
    assert probe_row["tracked_at_head"] is (probe_row["head_blob_oid"] is not None)
    assert probe_row["matches_head"] is (
        probe_row["head_blob_oid"] == probe_row["worktree_blob_oid"]
    )
    assert git_state["all_execution_sources_match_head"] is all(
        row["matches_head"] for row in git_state["files"].values()
    )


def test_resume_refuses_execution_source_custody_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe()
    monkeypatch.setattr(probe, "RESULTS_ROOT", tmp_path)
    run_dir = tmp_path / "sfess_cached_replay_ugc64_20260712T000000Z"
    args = argparse.Namespace(output_dir=run_dir, resume_from=None)
    _out, run = probe._initialize_or_resume_output(args)
    recorded = run["execution_source_custody"]
    assert run["command_argv_at_start"]
    assert set(run["environment_at_start"]) == {
        "OMP_NUM_THREADS",
        "PYTHONHASHSEED",
        "PYTHONPATH",
        "VECLIB_MAXIMUM_THREADS",
    }
    monkeypatch.setattr(
        probe,
        "_execution_source_custody",
        lambda: {**recorded, "tree_sha256": "f" * 64},
    )

    with pytest.raises(RuntimeError, match="execution source custody drift"):
        probe._initialize_or_resume_output(
            argparse.Namespace(output_dir=None, resume_from=run_dir)
        )


def test_cached_k5_replay_retains_only_initial_and_strict_gate_candidates(
    tmp_path: Path,
) -> None:
    probe = _load_probe()
    policy = probe._build_policy()
    compiled = policy.compile()
    table = load_cached_objective_jsonl(probe.TABLE)
    row = probe._run_sfess_k(k=5, out=tmp_path, table=table, compiled_policy=compiled)

    assert row["function_evals"] == 64
    assert row["gradient_sample_calls"] == 50
    assert row["strict_gate_calls"] == 10
    assert row["initial_calls"] == 1
    assert row["budget_padding_calls"] == 3
    assert row["best_s"] == pytest.approx(0.19080429731336374)
    retained = [
        record["value"]
        for record in row["query_records"]
        if record["purpose"] in {"initial", "strict_exact_gate"}
    ]
    assert row["best_s"] == min(retained)
    assert row["sample_values_retained_as_candidates"] is False
    assert len(row["lookup_decisions_rederived_after_completion"]) == 64
    assert all(
        decision["admitted_for_cached_lookup"]
        and not decision["live_gradient_admitted"]
        and decision["fallback_to_full_teacher"]
        for decision in row["lookup_decisions_rederived_after_completion"]
    )


def test_receipt_verdict_excludes_degenerate_k6_from_estimator_ranking(tmp_path: Path) -> None:
    probe = _load_probe()
    _source, baselines = probe._load_and_verify_source_receipt()
    policy = probe._build_policy()
    compiled = policy.compile()
    table = load_cached_objective_jsonl(probe.TABLE)
    controls = [
        probe._run_degenerate_control(k=k, out=tmp_path, table=table, compiled_policy=compiled)
        for k in (0, 6)
    ]
    sfess = [
        probe._run_sfess_k(k=k, out=tmp_path, table=table, compiled_policy=compiled)
        for k in probe.K_VALUES
    ]
    receipt = probe._build_receipt(
        run={
            "created_at_utc": "2026-07-12T00:00:00Z",
            "git_head_at_start": "a" * 40,
            "execution_source_custody": probe._execution_source_custody(),
            "execution_source_git_state_at_start": {
                "base_git_head": "a" * 40,
                "all_execution_sources_match_head": False,
                "files": {},
            },
            "command_argv_at_start": ["tools/probe_sfess_cached_replay.py", "--output-dir", "fixture"],
            "environment_at_start": {
                "OMP_NUM_THREADS": None,
                "PYTHONHASHSEED": None,
                "PYTHONPATH": "src",
                "VECLIB_MAXIMUM_THREADS": None,
            },
        },
        policy=policy,
        table=table,
        baselines=baselines,
        sfess_rows=sfess,
        controls=controls,
    )

    assert receipt["verdict"] == "NO-GO"
    assert receipt["same_budget_ranking_changed"] is False
    assert receipt["best_non_degenerate_sfess_arm"] == "sfess_k5"
    assert receipt["delta_best_sfess_minus_exact_enumeration_s"] == pytest.approx(
        7.052840218513268e-7
    )
    assert not any(row["arm"] == "structural_control_k6" for row in receipt["same_budget_ranking"])
    assert controls[1]["estimator_evidence"] is False


def test_production_sfess_resume_keeps_oracle_empty_when_trace_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe()
    policy = probe._build_policy()
    compiled = policy.compile()
    table = load_cached_objective_jsonl(probe.TABLE)
    probe._run_sfess_k(k=3, out=tmp_path, table=table, compiled_policy=compiled)
    snapshot = tmp_path / "k3_stage_snapshot.json"
    payload = json.loads(snapshot.read_text())
    payload["accepted"] += 1
    snapshot.write_text(json.dumps(payload), encoding="utf-8")

    original_oracle = probe.CountedCachedOracle
    created: list[Any] = []

    class TrackingOracle(original_oracle):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(probe, "CountedCachedOracle", TrackingOracle)
    with pytest.raises(SFESSError, match="counters disagree"):
        probe._run_sfess_k(k=3, out=tmp_path, table=table, compiled_policy=compiled)
    assert len(created) == 1
    assert created[0].calls == 0


@pytest.mark.parametrize(
    ("foreign_samples_per_gradient", "foreign_seed", "expected_error"),
    (
        (2, 396_400, "samples_per_gradient mismatch"),
        (5, 12_345, "seed mismatch"),
    ),
)
def test_production_resume_rejects_each_self_consistent_foreign_control(
    tmp_path: Path,
    foreign_samples_per_gradient: int,
    foreign_seed: int,
    expected_error: str,
) -> None:
    probe = _load_probe()
    policy = probe._build_policy()
    compiled = policy.compile()
    table = load_cached_objective_jsonl(probe.TABLE)
    snapshot = tmp_path / "k3_stage_snapshot.json"
    foreign_oracle = probe.CountedCachedOracle(
        table, budget=probe.EVAL_BUDGET, authorize_lookup=lambda mask: True
    )
    probe.SFESSFixedKSearch(
        foreign_oracle,
        probe.N_BITS,
        3,
        foreign_samples_per_gradient,
        foreign_seed,
        probe.NOISE_FLOOR_S,
    ).run(probe.EVAL_BUDGET, snapshot)

    with pytest.raises(SFESSError, match=expected_error):
        probe._run_sfess_k(k=3, out=tmp_path, table=table, compiled_policy=compiled)


def test_degenerate_resume_rejects_registered_but_semantically_wrong_purpose(
    tmp_path: Path,
) -> None:
    probe = _load_probe()
    policy = probe._build_policy()
    compiled = policy.compile()
    table = load_cached_objective_jsonl(probe.TABLE)
    probe._run_degenerate_control(k=0, out=tmp_path, table=table, compiled_policy=compiled)
    snapshot = tmp_path / "k0_stage_snapshot.json"
    payload = json.loads(snapshot.read_text())
    payload["query_records"][1]["purpose"] = "sfess_sample"
    snapshot.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="purpose schedule mismatch"):
        probe._run_degenerate_control(k=0, out=tmp_path, table=table, compiled_policy=compiled)


def test_output_boundary_refuses_tmp_and_requires_timestamped_results_path(
    tmp_path: Path,
) -> None:
    probe = _load_probe()
    with pytest.raises(RuntimeError, match="never /tmp"):
        probe._validated_output_path(tmp_path / "sfess_cached_replay_ugc64_20260712T000000Z")
    with pytest.raises(RuntimeError, match="must be named"):
        probe._validated_output_path(probe.RESULTS_ROOT / "sfess_cached_replay_ugc64_latest")
    accepted = probe.RESULTS_ROOT / "sfess_cached_replay_ugc64_20260712T000000Z"
    assert probe._validated_output_path(accepted) == accepted.resolve()


def test_forbidden_import_meter_has_positive_and_negative_canaries(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _load_probe()
    for name in list(sys.modules):
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in probe.FORBIDDEN_MODULE_PREFIXES
        ):
            monkeypatch.delitem(sys.modules, name)
    assert probe._forbidden_imports() == []
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    assert probe._forbidden_imports() == ["torch"]
    with pytest.raises(RuntimeError, match="forbidden scorer/training/cloud modules"):
        probe._require_zero_forbidden_imports("negative control")


def test_fresh_process_probe_import_loads_no_boundary_scorer_renderer_or_trainer() -> None:
    code = """
import importlib.util, json, sys
from pathlib import Path
probe_path = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location('sfess_probe_fresh_process', probe_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
custody = module._execution_source_custody()
print(json.dumps({
    'forbidden': module._forbidden_imports(),
    'source_paths': sorted({row['path'] for row in custody['modules'].values()}),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(PROBE_PATH)],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["forbidden"] == []
    assert payload["source_paths"] == [
        "src/tac/__init__.py",
        "src/tac/sfess_cached_replay.py",
        "src/tac/witness_dsl/sfess_cached_replay_policy.py",
        "tools/probe_sfess_cached_replay.py",
    ]
