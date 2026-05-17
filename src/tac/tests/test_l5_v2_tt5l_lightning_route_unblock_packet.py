# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from tac.optimization.l5_v2_measurement_schedule import (
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES,
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS,
)
from tac.optimization.l5_v2_tt5l_lightning_route_unblock_packet import (
    L5V2_TT5L_LIGHTNING_ROUTE_UNBLOCK_PACKET_SCHEMA,
    build_l5_v2_tt5l_lightning_route_unblock_packet,
    l5_v2_tt5l_lightning_route_unblock_packet_json,
    render_l5_v2_tt5l_lightning_route_unblock_packet_markdown,
)
from tac.optimization.l5_v2_tt5l_sideinfo_lightning_execution_bundle import (
    T4_LIGHTNING_EXACT_EVAL_RUNTIME_ENV,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _ten_cells() -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS:
        for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES:
            cells.append(
                {
                    "variant": variant,
                    "axis": axis,
                    "dry_run_state_file": {"stdout_core_matched": True},
                    "inflate_runtime": {"exists": True, "executable": True},
                }
            )
    return cells


def _bundle_cells() -> list[dict[str, object]]:
    command_env = " ".join(T4_LIGHTNING_EXACT_EVAL_RUNTIME_ENV)
    cells: list[dict[str, object]] = []
    for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS:
        for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES:
            cells.append(
                {
                    "variant": variant,
                    "axis": axis,
                    "dry_run_submit_command": f"dry {variant} {axis} {command_env}",
                    "non_dry_run_submit_command_template": (
                        f"submit {variant} {axis} {command_env}"
                    ),
                }
            )
    return cells


def _write_route_inputs(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path
    paths = {
        "provider": root / ".omx/research/provider.json",
        "preflight": root / ".omx/research/preflight.json",
        "bundle": root / ".omx/research/bundle.json",
        "dry_run": root / ".omx/research/dry_run.json",
        "plan": root / ".omx/research/plan.json",
        "harvest": root / ".omx/research/harvest.json",
        "effect": root / ".omx/research/effect.json",
        "architecture": root / ".omx/research/architecture.json",
        "legacy": root / ".omx/research/legacy.json",
    }
    _write_json(
        paths["provider"],
        {
            "schema": "cloud_provider_readiness_v1",
            "providers": [
                {
                    "provider": "lightning",
                    "blockers": [
                        "lightning_teamspace_missing",
                        "lightning_owner_missing",
                        "lightning_ssh_target_missing",
                        "no_dispatch_claim",
                    ],
                    "stdout_excerpt": "2026.4.23",
                }
            ],
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    _write_json(
        paths["preflight"],
        {
            "schema": "l5_v2_tt5l_sideinfo_lightning_execution_preflight_v1",
            "score_claim": False,
        },
    )
    _write_json(
        paths["bundle"],
        {
            "schema": "l5_v2_tt5l_sideinfo_lightning_execution_bundle_v1",
            "required_axes": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES),
            "required_variants": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS),
            "cells": _bundle_cells(),
            "score_claim": False,
            "ready_for_provider_dispatch": False,
        },
    )
    _write_json(
        paths["dry_run"],
        {
            "schema": (
                "l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification_v1"
            ),
            "all_dry_runs_passed": True,
            "passed_cell_count": 10,
            "cell_count": 10,
            "cells": _ten_cells(),
            "score_claim": False,
            "ready_for_provider_dispatch": False,
        },
    )
    _write_json(
        paths["plan"],
        {
            "schema": "l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_v1",
            "source_commit": "a" * 40,
            "required_axes": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES),
            "required_variants": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS),
            "cell_count": 10,
            "score_claim": False,
            "ready_for_provider_dispatch": False,
        },
    )
    _write_json(
        paths["harvest"],
        {
            "schema": "l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_v1",
            "harvested_exact_eval_artifact_count": 0,
            "missing_exact_eval_artifact_count": 10,
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    _write_json(
        paths["effect"],
        {
            "schema": "l5_v2_sideinfo_effect_curve_v1",
            "predicate_passed": False,
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    _write_json(
        paths["architecture"],
        {
            "schema": "l5_v2_architecture_lock_packet_v1",
            "architecture_lock_allowed": False,
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    _write_json(paths["legacy"], {"schema": "legacy", "score_claim": False})
    return paths


def _build(tmp_path: Path, **kwargs: object) -> dict[str, object]:
    paths = _write_route_inputs(tmp_path)
    return build_l5_v2_tt5l_lightning_route_unblock_packet(
        repo_root=tmp_path,
        current_head_commit=kwargs.pop("current_head_commit", "b" * 40),
        source_relevant_diff_paths=kwargs.pop("source_relevant_diff_paths", ()),
        provider_readiness_path=str(paths["provider"]),
        execution_preflight_path=str(paths["preflight"]),
        execution_bundle_path=str(paths["bundle"]),
        dry_run_verification_path=str(paths["dry_run"]),
        paired_axis_plan_path=str(paths["plan"]),
        harvest_cells_path=str(paths["harvest"]),
        sideinfo_effect_curve_path=str(paths["effect"]),
        legacy_alt_provider_plan_path=str(paths["legacy"]),
    )


def test_tt5l_route_unblock_packet_preserves_false_authority_and_evidence(
    tmp_path: Path,
) -> None:
    paths = _write_route_inputs(tmp_path)
    packet = build_l5_v2_tt5l_lightning_route_unblock_packet(
        repo_root=tmp_path,
        current_head_commit="b" * 40,
        provider_readiness_path=str(paths["provider"]),
        execution_preflight_path=str(paths["preflight"]),
        execution_bundle_path=str(paths["bundle"]),
        dry_run_verification_path=str(paths["dry_run"]),
        paired_axis_plan_path=str(paths["plan"]),
        harvest_cells_path=str(paths["harvest"]),
        sideinfo_effect_curve_path=str(paths["effect"]),
        legacy_alt_provider_plan_path=str(paths["legacy"]),
    )

    assert packet["schema"] == L5V2_TT5L_LIGHTNING_ROUTE_UNBLOCK_PACKET_SCHEMA
    assert packet["planning_only"] is True
    assert packet["score_claim"] is False
    assert packet["promotion_eligible"] is False
    assert packet["rank_or_kill_eligible"] is False
    assert packet["ready_for_exact_eval_dispatch"] is False
    assert packet["ready_for_provider_dispatch"] is False
    assert packet["dispatch_attempted"] is False
    assert packet["provider_spend_attempted"] is False
    assert packet["source_artifacts"]["provider_readiness"]["sha256"] == _sha(
        paths["provider"]
    )
    assert "architecture_lock_packet" not in packet["source_artifacts"]
    dry = packet["source_artifacts"][
        "sideinfo_execution_bundle_dry_run_verification"
    ]
    assert dry["all_dry_runs_passed"] is True
    assert dry["passed_cell_count"] == 10
    assert packet["verified_before_this_packet"][
        "bundle_all_10_dry_run_state_files_match_stdout_core"
    ] is True
    assert packet["verified_before_this_packet"][
        "bundle_all_10_inflate_runtimes_exist_and_are_executable"
    ] is True
    assert packet["verified_before_this_packet"][
        "bundle_t4_runtime_env_pins_embedded"
    ] is True
    assert "LIGHTNING_TEAMSPACE missing" in packet["remaining_blockers"]


def test_tt5l_route_unblock_packet_fails_closed_on_source_relevant_diff(
    tmp_path: Path,
) -> None:
    packet = _build(
        tmp_path,
        current_head_commit="b" * 40,
        source_relevant_diff_paths=("src/tac/optimization/stale.py",),
    )

    plan = packet["source_artifacts"]["sideinfo_lightning_paired_axis_plan"]
    assert plan["source_commit_matches_current_head"] is False
    assert plan["source_relevant_paths_match_current_head"] is False
    assert "paired_axis_plan:source_relevant_paths_changed" in packet["blockers"]
    assert packet["score_claim"] is False
    assert packet["ready_for_provider_dispatch"] is False


def test_tt5l_route_unblock_packet_fails_closed_on_dirty_source_relevant_diff(
    tmp_path: Path,
) -> None:
    packet = _build(
        tmp_path,
        current_head_commit="a" * 40,
        source_relevant_diff_paths=("src/tac/optimization/tt5l_sideinfo_variant_packets.py",),
    )

    plan = packet["source_artifacts"]["sideinfo_lightning_paired_axis_plan"]
    assert plan["source_commit_matches_current_head"] is True
    assert plan["source_relevant_paths_match_current_head"] is False
    assert "paired_axis_plan:source_relevant_paths_changed" in packet["blockers"]
    assert packet["score_claim"] is False
    assert packet["ready_for_provider_dispatch"] is False


def test_tt5l_route_unblock_packet_records_missing_artifact_without_authority(
    tmp_path: Path,
) -> None:
    paths = _write_route_inputs(tmp_path)
    paths["dry_run"].unlink()

    packet = build_l5_v2_tt5l_lightning_route_unblock_packet(
        repo_root=tmp_path,
        current_head_commit="b" * 40,
        provider_readiness_path=str(paths["provider"]),
        execution_preflight_path=str(paths["preflight"]),
        execution_bundle_path=str(paths["bundle"]),
        dry_run_verification_path=str(paths["dry_run"]),
        paired_axis_plan_path=str(paths["plan"]),
        harvest_cells_path=str(paths["harvest"]),
        sideinfo_effect_curve_path=str(paths["effect"]),
        legacy_alt_provider_plan_path=str(paths["legacy"]),
    )

    dry = packet["source_artifacts"][
        "sideinfo_execution_bundle_dry_run_verification"
    ]
    assert dry["exists"] is False
    assert "artifact_missing" in dry["blockers"]
    assert (
        "sideinfo_execution_bundle_dry_run_verification:artifact_missing"
        in packet["blockers"]
    )
    assert "dry_run_verification:not_all_dry_runs_passed" in packet["blockers"]
    assert packet["score_claim"] is False
    assert packet["promotion_eligible"] is False


def test_tt5l_route_unblock_packet_json_and_markdown_are_axis_labelled(
    tmp_path: Path,
) -> None:
    packet = _build(tmp_path)

    decoded = json.loads(l5_v2_tt5l_lightning_route_unblock_packet_json(packet))
    report = render_l5_v2_tt5l_lightning_route_unblock_packet_markdown(packet)

    assert decoded["schema"] == L5V2_TT5L_LIGHTNING_ROUTE_UNBLOCK_PACKET_SCHEMA
    assert "score_claim=false" in report
    assert "promotion_eligible=false" in report
    assert "[contest-CPU]" in report
    assert "[contest-CUDA]" in report
    assert "No CPU/CUDA axis may be promoted" in report


def test_tt5l_route_unblock_packet_cli_writes_json_and_markdown(
    tmp_path: Path,
) -> None:
    root = Path.cwd()
    paths = _write_route_inputs(tmp_path)
    output_json = tmp_path / "route_packet.json"
    output_md = tmp_path / "route_packet.md"

    proc = subprocess.run(
        [
            str(root / "tools" / "build_l5_v2_tt5l_lightning_route_unblock_packet.py"),
            "--repo-root",
            str(tmp_path),
            "--provider-readiness-json",
            str(paths["provider"]),
            "--execution-preflight-json",
            str(paths["preflight"]),
            "--execution-bundle-json",
            str(paths["bundle"]),
            "--dry-run-verification-json",
            str(paths["dry_run"]),
            "--paired-axis-plan-json",
            str(paths["plan"]),
            "--harvest-cells-json",
            str(paths["harvest"]),
            "--sideinfo-effect-curve-json",
            str(paths["effect"]),
            "--legacy-alt-provider-plan-json",
            str(paths["legacy"]),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--current-head-commit",
            "a" * 40,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["source_artifacts"]["sideinfo_lightning_paired_axis_plan"][
        "source_commit_matches_current_head"
    ] is True
    assert payload["score_claim"] is False
    assert "architecture_lock_packet" not in payload["source_artifacts"]
    assert "score_claim=false" in proc.stdout
    assert "remaining_route_blocker_count=" in proc.stdout
    assert output_md.is_file()
    assert "Architecture lock packet" not in output_md.read_text(encoding="utf-8")
