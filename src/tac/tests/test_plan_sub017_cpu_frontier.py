# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_tool():
    repo_root = Path(__file__).resolve().parents[3]
    tool_path = repo_root / "tools" / "plan_sub017_cpu_frontier.py"
    spec = importlib.util.spec_from_file_location("plan_sub017_cpu_frontier", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {tool_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_falsified_relerr_schedule(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "factorized_hnerv_relerr_schedule_plan.v1",
                "schedules": [
                    {
                        "relerr_cap": 0.02,
                        "estimated_isolated_brotli_savings_bytes": 0,
                        "selected_rows": [],
                    },
                    {
                        "relerr_cap": 0.04,
                        "estimated_isolated_brotli_savings_bytes": 0,
                        "selected_rows": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_default_plan_is_non_dispatching_cpu_design(tmp_path: Path) -> None:
    mod = _load_tool()
    relerr_schedule = tmp_path / "relerr.json"
    _write_falsified_relerr_schedule(relerr_schedule)

    plan = mod.build_plan(relerr_schedule_json=relerr_schedule)

    assert plan["schema"] == "sub017_cpu_frontier_plan.v1"
    assert plan["score_claim"] is False
    assert plan["promotion_eligible"] is False
    assert plan["rank_or_kill_eligible"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert plan["dispatch_attempted"] is False
    assert "remote_dispatch_forbidden_for_this_task" in plan["global_dispatch_blockers"]
    assert (
        "posthoc_factorized_hnerv_relerr_safe_schedule_falsified"
        in plan["global_dispatch_blockers"]
    )
    assert plan["posthoc_relerr_schedule_probe"]["status"] == (
        "falsified_for_posthoc_factorization"
    )
    assert plan["byte_budget"]["target_max_archive_bytes_if_components_hold"] < plan["anchor"]["archive_bytes"]
    assert plan["byte_budget"]["required_archive_savings_bytes"] > 50_000


def test_plan_targets_stem_and_early_conv2d_with_svd_before_coarsening(
    tmp_path: Path,
) -> None:
    mod = _load_tool()
    relerr_schedule = tmp_path / "relerr.json"
    _write_falsified_relerr_schedule(relerr_schedule)

    plan = mod.build_plan(relerr_schedule_json=relerr_schedule)
    recommended = plan["recommended_cpu_candidate"]
    layer_names = {row["name"] for row in recommended["target_layers"]}
    order = [step["name"] for step in plan["safe_stacking_order"]]

    assert "stem.weight" in layer_names
    assert "blocks.0.weight" in layer_names
    assert "blocks.1.weight" in layer_names
    assert any(name.startswith("blocks.2") for name in layer_names)
    assert "rgb_0.weight" not in layer_names
    assert "rgb_1.weight" not in layer_names
    assert order == [
        "svd_low_rank_stem_and_early_conv2d",
        "continuous_k_allocation",
        "analytical_lossy_coarsening",
        "entropy_pack_and_noop_guards",
    ]
    for row in recommended["target_layers"]:
        assert row["rank"] <= min(row["matrix_shape"])
        assert "CPU SVD on dequantized fp32 weights" in row["construction"]


def test_recommended_candidate_reaches_sub017_only_as_projection(tmp_path: Path) -> None:
    mod = _load_tool()
    relerr_schedule = tmp_path / "relerr.json"
    _write_falsified_relerr_schedule(relerr_schedule)

    plan = mod.build_plan(relerr_schedule_json=relerr_schedule)
    recommended = plan["recommended_cpu_candidate"]

    assert recommended["meets_sub017_byte_budget_if_components_hold"] is True
    assert recommended["estimated_archive_bytes"] <= recommended["target_max_archive_bytes"]
    assert recommended["projected_score_if_components_hold"] < 0.17
    assert recommended["score_claim"] is False
    assert recommended["ready_for_exact_eval_dispatch"] is False
    assert "component_preservation_unproven" in recommended["dispatch_blockers"]
    assert (
        "posthoc_factorized_hnerv_relerr_safe_schedule_falsified"
        in recommended["dispatch_blockers"]
    )
    assert recommended["posthoc_relerr_probe_verdict"] == (
        "falsified_for_posthoc_factorization"
    )


def test_plan_keeps_missing_relerr_schedule_fail_closed(tmp_path: Path) -> None:
    mod = _load_tool()

    plan = mod.build_plan(relerr_schedule_json=tmp_path / "missing.json")

    assert plan["posthoc_relerr_schedule_probe"]["status"] == "missing"
    assert (
        "posthoc_factorized_hnerv_relerr_safe_schedule_missing"
        in plan["global_dispatch_blockers"]
    )
    assert (
        "posthoc_factorized_hnerv_relerr_safe_schedule_missing"
        in plan["recommended_cpu_candidate"]["dispatch_blockers"]
    )


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    mod = _load_tool()
    json_out = tmp_path / "plan.json"
    md_out = tmp_path / "plan.md"
    relerr_schedule = tmp_path / "relerr.json"
    _write_falsified_relerr_schedule(relerr_schedule)

    rc = mod.main(
        [
            "--relerr-schedule-json",
            str(relerr_schedule),
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(md_out),
        ]
    )

    assert rc == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    markdown = md_out.read_text(encoding="utf-8")
    assert payload["schema"] == "sub017_cpu_frontier_plan.v1"
    assert payload["score_claim"] is False
    assert "Posthoc Relerr Probe" in markdown
    assert "Sub-0.17 CPU Frontier Plan" in markdown
    assert payload["recommended_cpu_candidate_id"] in markdown
