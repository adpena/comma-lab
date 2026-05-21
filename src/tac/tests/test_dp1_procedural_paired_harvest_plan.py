# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import stat
import zipfile
from pathlib import Path

import pytest
import yaml

from tac.optimization.dp1_procedural_paired_harvest_plan import (
    build_dp1_procedural_paired_harvest_plan,
    render_markdown,
)


def _write_recipe(root: Path, *, variant: str, lane_id: str) -> Path:
    names = {
        "baseline": (
            "substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch"
        ),
        "procedural": (
            "substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch"
        ),
        "null_control": (
            "substrate_pretrained_driving_prior_null_exploit_codebook_modal_t4_paired_dispatch"
        ),
    }
    path = root / ".omx" / "operator_authorize_recipes" / f"{names[variant]}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "name": names[variant],
        "lane_id": lane_id,
        "dispatch_enabled": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "paired_axis": {"enabled": True},
        "remote_driver": "scripts/remote_lane_substrate_pretrained_driving_prior.sh",
        "env_overrides": {"DPP_SKIP_AUTH_EVAL": "1"},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return path


def _set_recipe_dispatch_enabled(root: Path, *, variant: str, enabled: bool) -> None:
    names = {
        "baseline": (
            "substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch"
        ),
        "procedural": (
            "substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch"
        ),
        "null_control": (
            "substrate_pretrained_driving_prior_null_exploit_codebook_modal_t4_paired_dispatch"
        ),
    }
    path = root / ".omx" / "operator_authorize_recipes" / f"{names[variant]}.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["dispatch_enabled"] = enabled
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _write_candidate_output(
    output_dir: Path,
    *,
    lane_id: str,
    procedural: bool = False,
    null_control: bool = False,
    score_claim: bool = False,
    ready_for_exact_eval_dispatch: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_dir / "archive.zip", "w") as zf:
        zf.writestr("0.bin", b"dp1")
    submission = output_dir / "submission"
    submission.mkdir()
    inflate_sh = submission / "inflate.sh"
    inflate_sh.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    inflate_sh.chmod(inflate_sh.stat().st_mode | stat.S_IXUSR)
    (submission / "inflate.py").write_text("raise SystemExit(0)\n", encoding="utf-8")

    manifest = {
        "score_claim": score_claim,
        "score_claim_valid": score_claim,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": ready_for_exact_eval_dispatch,
    }
    provenance = {
        "lane_id": lane_id,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, sort_keys=True), encoding="utf-8"
    )
    if procedural:
        procedural_payload = {
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "null_exploit_control": null_control,
        }
        (output_dir / "procedural_variant_provenance.json").write_text(
            json.dumps(procedural_payload, sort_keys=True),
            encoding="utf-8",
        )
    _write_modal_metadata(output_dir, lane_id=lane_id)


def _write_modal_metadata(
    output_dir: Path,
    *,
    lane_id: str,
    sentinels: dict[str, str] | None = None,
    head: str = "a" * 40,
) -> None:
    payload = {
        "metadata_schema": "modal_train_lane_dispatch_metadata_v2_catalog166",
        "lane_id": lane_id,
        "label": f"{lane_id}_label",
        "call_id": f"fc-{lane_id}",
        "mounted_code_git_head": head,
        "mounted_code_git_branch": "main",
        "require_clean_head": True,
        "working_tree_dirty": False,
        "score_claim": False,
        "promotion_eligible": False,
        "sentinel_files_local_sha256": sentinels
        or {
            "experiments/modal_train_lane.py": "1" * 64,
            "tools/operator_authorize.py": "2" * 64,
            "scripts/remote_lane_substrate_pretrained_driving_prior.sh": "3" * 64,
        },
    }
    (output_dir / "modal_metadata.json").write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8"
    )


def _write_training_metadata(path: Path, *, variant: str, call_id: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "call_id": call_id,
        "label": f"dp1_{variant}",
        "lane_id": f"lane_{variant}",
        "live_volume": "comma-train-lane-results",
        "live_volume_prefix": f"dp1_{variant}/",
        "dispatched_at": "2026-05-21T03:16:00",
        "max_seconds": 5400,
        "mounted_code_git_head": "a" * 40,
        "mounted_code_git_branch": "main",
        "working_tree_dirty": False,
        "working_tree_dirty_paths_count": 0,
        "sentinel_files_local_sha256": {"a.py": "0" * 64},
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_all_recipes(root: Path) -> dict[str, str]:
    lanes = {
        "baseline": "lane_dp1_original_baseline_first_paired_anchor_20260520",
        "procedural": "lane_dp1_procedural_codebook_replacement_first_paired_smoke_20260520",
        "null_control": "lane_dp1_null_exploit_codebook_replacement_control_paired_smoke_20260520",
    }
    for variant, lane_id in lanes.items():
        _write_recipe(root, variant=variant, lane_id=lane_id)
    return lanes


def test_plan_blocks_before_candidate_outputs_exist(tmp_path: Path) -> None:
    _write_all_recipes(tmp_path)

    plan = build_dp1_procedural_paired_harvest_plan(
        output_dirs={},
        repo_root=tmp_path,
    )

    assert plan["planning_only"] is True
    assert plan["score_claim"] is False
    assert plan["all_required_candidates_ready"] is False
    assert "baseline_and_procedural_paired_harvest_not_ready" in plan["top_blockers"]
    statuses = {row["variant"]: row["status"] for row in plan["candidates"]}
    assert statuses == {"baseline": "blocked", "procedural": "blocked"}
    assert all(row["paired_dispatch_plan_command"] is None for row in plan["candidates"])


def test_plan_can_include_running_training_call_status(tmp_path: Path) -> None:
    _write_all_recipes(tmp_path)
    baseline_meta = _write_training_metadata(
        tmp_path / "baseline_modal" / "modal_metadata.json",
        variant="baseline",
        call_id="fc-base",
    )
    procedural_meta = _write_training_metadata(
        tmp_path / "procedural_modal" / "modal_metadata.json",
        variant="procedural",
        call_id="fc-proc",
    )

    class _RunningCall:
        def get(self, timeout: float | None = None) -> dict:
            raise TimeoutError

    plan = build_dp1_procedural_paired_harvest_plan(
        output_dirs={},
        training_metadata_paths={
            "baseline": baseline_meta,
            "procedural": procedural_meta,
        },
        poll_training_calls=True,
        repo_root=tmp_path,
        function_call_from_id=lambda _call_id: _RunningCall(),
    )

    assert plan["training_call_status"]["status"] == "running"
    assert "dp1_training_calls_not_ready_for_harvest" in plan["top_blockers"]
    assert plan["training_call_status"]["ready_for_training_harvest"] is False
    md = render_markdown(plan)
    assert "Training-call status: `running`" in md
    assert "Baseline training call: `fc-base` / `running_or_pending`" in md


def test_plan_does_not_add_training_blocker_when_training_calls_ready(
    tmp_path: Path,
) -> None:
    _write_all_recipes(tmp_path)
    baseline_meta = _write_training_metadata(
        tmp_path / "baseline_modal" / "modal_metadata.json",
        variant="baseline",
        call_id="fc-base",
    )
    procedural_meta = _write_training_metadata(
        tmp_path / "procedural_modal" / "modal_metadata.json",
        variant="procedural",
        call_id="fc-proc",
    )

    class _FinishedCall:
        def __init__(self, call_id: str) -> None:
            self.call_id = call_id

        def get(self, timeout: float | None = None) -> dict:
            return {
                "returncode": 0,
                "elapsed_seconds": 12.0,
                "timed_out": False,
                "artifacts": {f"{self.call_id}/archive.zip": b"zip"},
                "stdout_tail": "finished",
            }

    plan = build_dp1_procedural_paired_harvest_plan(
        output_dirs={},
        training_metadata_paths={
            "baseline": baseline_meta,
            "procedural": procedural_meta,
        },
        poll_training_calls=True,
        repo_root=tmp_path,
        function_call_from_id=_FinishedCall,
    )

    assert plan["training_call_status"]["status"] == "ready_for_training_harvest"
    assert plan["training_call_status"]["ready_for_training_harvest"] is True
    assert "dp1_training_calls_not_ready_for_harvest" not in plan["top_blockers"]
    assert "baseline_and_procedural_paired_harvest_not_ready" in plan["top_blockers"]


def test_ready_plan_emits_real_paired_dispatch_commands(tmp_path: Path) -> None:
    lanes = _write_all_recipes(tmp_path)
    baseline_out = tmp_path / "baseline_out"
    procedural_out = tmp_path / "procedural_out"
    _write_candidate_output(baseline_out, lane_id=lanes["baseline"])
    _write_candidate_output(
        procedural_out,
        lane_id=lanes["procedural"],
        procedural=True,
    )

    plan = build_dp1_procedural_paired_harvest_plan(
        output_dirs={
            "baseline": baseline_out,
            "procedural": procedural_out,
        },
        repo_root=tmp_path,
    )

    assert plan["all_required_candidates_ready"] is True
    assert plan["score_claim"] is False
    assert plan["dispatch_attempted"] is False
    assert plan["paired_source_equivalence"]["checked"] is True
    assert plan["paired_source_equivalence"]["shared_sentinel_count"] == 3
    assert plan["paired_source_equivalence"]["mismatched_sentinel_files"] == []
    adjudication_cmd = plan["post_harvest_adjudication_command"]
    assert adjudication_cmd[:2] == [
        ".venv/bin/python",
        "tools/adjudicate_dp1_procedural_paired_harvest.py",
    ]
    assert "--baseline-cpu-dir" in adjudication_cmd
    assert "--procedural-cuda-dir" in adjudication_cmd
    assert "--json-out" in adjudication_cmd
    for row in plan["candidates"]:
        assert row["status"] == "ready_for_paired_dispatch_plan"
        cmd = row["paired_dispatch_plan_command"]
        execute = row["paired_dispatch_execute_command"]
        assert cmd[:2] == [".venv/bin/python", "tools/dispatch_modal_paired_auth_eval.py"]
        assert "--execute" not in cmd
        assert "--execute" in execute
        assert "--expected-runtime-tree-sha256" in cmd
        assert cmd[cmd.index("--expected-runtime-tree-sha256") + 1] == "auto"
        assert "--skip-axis-if-promotable-anchor-exists" in cmd
        assert "--json-out" in cmd
        assert cmd[cmd.index("--json-out") + 1].endswith(
            "/paired_dispatch_plan.json"
        )
        assert row["harvest_commands"]["contest_cuda"].startswith(
            ".venv/bin/python tools/recover_modal_auth_eval.py --output-dir "
        )
        assert row["harvest_commands"]["contest_cpu"].startswith(
            ".venv/bin/python tools/recover_modal_auth_eval.py --output-dir "
        )


def test_recipe_dispatch_flip_does_not_block_harvest_plan(tmp_path: Path) -> None:
    lanes = _write_all_recipes(tmp_path)
    _set_recipe_dispatch_enabled(tmp_path, variant="baseline", enabled=True)
    _set_recipe_dispatch_enabled(tmp_path, variant="procedural", enabled=True)
    baseline_out = tmp_path / "baseline_out"
    procedural_out = tmp_path / "procedural_out"
    _write_candidate_output(baseline_out, lane_id=lanes["baseline"])
    _write_candidate_output(
        procedural_out,
        lane_id=lanes["procedural"],
        procedural=True,
    )

    plan = build_dp1_procedural_paired_harvest_plan(
        output_dirs={
            "baseline": baseline_out,
            "procedural": procedural_out,
        },
        repo_root=tmp_path,
    )

    assert plan["all_required_candidates_ready"] is True
    rows = {row["variant"]: row for row in plan["candidates"]}
    assert rows["baseline"]["recipe_dispatch_enabled"] is True
    assert rows["procedural"]["recipe_dispatch_enabled"] is True
    assert all(
        "recipe_dispatch_enabled_not_false" not in row["blockers"]
        for row in plan["candidates"]
    )


def test_paired_harvest_blocks_sentinel_hash_mismatch(tmp_path: Path) -> None:
    lanes = _write_all_recipes(tmp_path)
    baseline_out = tmp_path / "baseline_out"
    procedural_out = tmp_path / "procedural_out"
    _write_candidate_output(baseline_out, lane_id=lanes["baseline"])
    _write_candidate_output(
        procedural_out,
        lane_id=lanes["procedural"],
        procedural=True,
    )
    _write_modal_metadata(
        procedural_out,
        lane_id=lanes["procedural"],
        sentinels={
            "experiments/modal_train_lane.py": "1" * 64,
            "tools/operator_authorize.py": "f" * 64,
            "scripts/remote_lane_substrate_pretrained_driving_prior.sh": "3" * 64,
        },
        head="b" * 40,
    )

    plan = build_dp1_procedural_paired_harvest_plan(
        output_dirs={
            "baseline": baseline_out,
            "procedural": procedural_out,
        },
        repo_root=tmp_path,
    )

    assert plan["all_required_candidates_ready"] is False
    assert "paired_candidate_sentinel_sha256_mismatch" in plan["top_blockers"]
    assert plan["post_harvest_adjudication_command"] is None
    assert plan["paired_source_equivalence"]["mismatched_sentinel_files"] == [
        "tools/operator_authorize.py"
    ]


def test_manifest_score_claim_blocks_dispatch_command(tmp_path: Path) -> None:
    lanes = _write_all_recipes(tmp_path)
    baseline_out = tmp_path / "baseline_out"
    procedural_out = tmp_path / "procedural_out"
    _write_candidate_output(
        baseline_out,
        lane_id=lanes["baseline"],
        score_claim=True,
    )
    _write_candidate_output(
        procedural_out,
        lane_id=lanes["procedural"],
        procedural=True,
    )

    plan = build_dp1_procedural_paired_harvest_plan(
        output_dirs={
            "baseline": baseline_out,
            "procedural": procedural_out,
        },
        repo_root=tmp_path,
    )

    baseline = next(row for row in plan["candidates"] if row["variant"] == "baseline")
    assert baseline["status"] == "blocked"
    assert "manifest_score_claim_true" in baseline["blockers"]
    assert "manifest_score_claim_valid_true" in baseline["blockers"]
    assert baseline["paired_dispatch_plan_command"] is None


def test_manifest_ready_for_exact_eval_claim_blocks_dispatch_command(
    tmp_path: Path,
) -> None:
    lanes = _write_all_recipes(tmp_path)
    baseline_out = tmp_path / "baseline_out"
    procedural_out = tmp_path / "procedural_out"
    _write_candidate_output(
        baseline_out,
        lane_id=lanes["baseline"],
        ready_for_exact_eval_dispatch=True,
    )
    _write_candidate_output(
        procedural_out,
        lane_id=lanes["procedural"],
        procedural=True,
    )

    plan = build_dp1_procedural_paired_harvest_plan(
        output_dirs={
            "baseline": baseline_out,
            "procedural": procedural_out,
        },
        repo_root=tmp_path,
    )

    baseline = next(row for row in plan["candidates"] if row["variant"] == "baseline")
    assert baseline["status"] == "blocked"
    assert "manifest_ready_for_exact_eval_dispatch_true" in baseline["blockers"]
    assert baseline["paired_dispatch_plan_command"] is None


def test_recipe_must_keep_trainer_auth_eval_skipped(tmp_path: Path) -> None:
    lanes = _write_all_recipes(tmp_path)
    recipe = tmp_path / ".omx" / "operator_authorize_recipes" / (
        "substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch.yaml"
    )
    payload = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    payload["env_overrides"]["DPP_SKIP_AUTH_EVAL"] = "0"
    recipe.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    baseline_out = tmp_path / "baseline_out"
    procedural_out = tmp_path / "procedural_out"
    _write_candidate_output(baseline_out, lane_id=lanes["baseline"])
    _write_candidate_output(
        procedural_out,
        lane_id=lanes["procedural"],
        procedural=True,
    )

    plan = build_dp1_procedural_paired_harvest_plan(
        output_dirs={
            "baseline": baseline_out,
            "procedural": procedural_out,
        },
        repo_root=tmp_path,
    )

    procedural = next(
        row for row in plan["candidates"] if row["variant"] == "procedural"
    )
    assert procedural["status"] == "blocked"
    assert "recipe_DPP_SKIP_AUTH_EVAL_not_1" in procedural["blockers"]


def test_markdown_renders_blockers_and_commands(tmp_path: Path) -> None:
    lanes = _write_all_recipes(tmp_path)
    baseline_out = tmp_path / "baseline_out"
    procedural_out = tmp_path / "procedural_out"
    _write_candidate_output(baseline_out, lane_id=lanes["baseline"])
    _write_candidate_output(
        procedural_out,
        lane_id=lanes["procedural"],
        procedural=True,
    )
    plan = build_dp1_procedural_paired_harvest_plan(
        output_dirs={
            "baseline": baseline_out,
            "procedural": procedural_out,
        },
        repo_root=tmp_path,
    )

    md = render_markdown(plan)

    assert "# DP1 Procedural Paired-Harvest Plan" in md
    assert "tools/dispatch_modal_paired_auth_eval.py" in md
    assert "tools/adjudicate_dp1_procedural_paired_harvest.py" in md
    assert "| baseline | ready_for_paired_dispatch_plan |" in md
