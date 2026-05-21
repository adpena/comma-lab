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
