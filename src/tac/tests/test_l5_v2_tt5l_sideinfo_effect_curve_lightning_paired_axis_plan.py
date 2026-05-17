# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from tac.optimization.l5_v2_measurement_schedule import (
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS,
)
from tac.optimization.l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan import (
    L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_SCHEMA,
    build_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan,
    l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_json,
    render_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_markdown,
)
from tac.tests.test_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan import (
    _write_manifest,
)


def _state_paths(cell: dict[str, object]) -> tuple[Path, Path, Path]:
    return (
        Path(str(cell["state_path"])),
        Path(str(cell["dry_run_stdout_path"])),
        Path(str(cell["dry_run_stderr_path"])),
    )


def test_tt5l_lightning_paired_axis_plan_materializes_ten_exact_eval_cells(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    plan = build_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan(
        manifest=payload,
        manifest_path=manifest,
        repo_root=tmp_path,
        artifact_root="experiments/results/lightning_batch/test_tt5l_paired_axes",
        generated_at_utc="2026-05-17T00:00:00Z",
    )

    assert plan["schema"] == L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_SCHEMA
    assert plan["score_claim"] is False
    assert plan["promotion_eligible"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert plan["ready_for_provider_dispatch"] is False
    assert plan["dispatch_attempted"] is False
    assert plan["cell_count"] == len(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS) * 2
    assert plan["all_cells_dry_run_ready"] is True
    assert plan["verification"]["paired_axis_dry_run_semantics"] == "PASS"
    assert plan["verification"]["roles"] == ["exact_cpu_eval", "exact_cuda_eval"]
    assert {cell["axis"] for cell in plan["cells"]} == {"contest_cpu", "contest_cuda"}

    for cell in plan["cells"]:
        spec = cell["spec"]
        assert isinstance(spec, dict)
        command = str(spec["command"])
        assert cell["ready_for_operator_dispatch"] is True
        assert cell["ready_for_provider_dispatch"] is False
        assert cell["blockers"] == []
        assert spec["queue_metadata"]["axis"] == cell["axis"]
        assert spec["queue_metadata"]["variant"] == cell["variant"]
        assert spec["queue_metadata"]["pair_group_id"] == cell["pair_group_id"]
        assert spec["queue_metadata"]["archive_sha256"] == cell["archive_sha256"]
        state_path, stdout_path, stderr_path = _state_paths(cell)
        for path in (state_path, stdout_path, stderr_path):
            assert (tmp_path / path).is_file()
        assert (tmp_path / stderr_path).stat().st_size == 0
        assert cell["state"]["bytes"] > 0
        assert cell["dry_run_stdout"]["bytes"] > 0
        assert cell["dry_run_stderr"]["bytes"] == 0
        if cell["axis"] == "contest_cpu":
            assert cell["role"] == "exact_cpu_eval"
            assert cell["required_device"] == "cpu"
            assert spec["role"] == "exact_cpu_eval"
            assert spec["adjudication"]["required_device"] == "cpu"
            assert "--device cpu" in command
            assert "--device cuda" not in command
            assert "INFLATE_REQUIRE_CUDA=1" not in command
        else:
            assert cell["axis"] == "contest_cuda"
            assert cell["role"] == "exact_cuda_eval"
            assert cell["required_device"] == "cuda"
            assert spec["role"] == "exact_cuda_eval"
            assert spec["adjudication"]["required_device"] == "cuda"
            assert "--device cuda" in command
            assert "--device cpu" not in command
            assert "INFLATE_REQUIRE_CUDA=1" in command


def test_tt5l_lightning_paired_axis_plan_fails_closed_on_archive_sha_mismatch(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["variants"][0]["archive_sha256"] = "0" * 64

    plan = build_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan(
        manifest=payload,
        manifest_path=manifest,
        repo_root=tmp_path,
        artifact_root="experiments/results/lightning_batch/test_tt5l_paired_axes",
    )

    zero_cells = [cell for cell in plan["cells"] if cell["variant"] == "zero"]
    assert len(zero_cells) == 2
    for cell in zero_cells:
        assert cell["ready_for_operator_dispatch"] is False
        assert "variant_archive_sha_mismatch:zero" in cell["blockers"]
        assert "lightning_exact_eval_spec_not_materialized" in cell["blockers"]
        assert cell["spec"] is None
    assert plan["all_cells_dry_run_ready"] is False
    assert plan["verification"]["paired_axis_dry_run_semantics"] == "BLOCKED"
    assert "variant_archive_sha_mismatch:zero" in plan["blockers"]


def test_tt5l_lightning_paired_axis_plan_json_and_markdown_are_durable(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    plan = build_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan(
        manifest=payload,
        manifest_path=manifest,
        repo_root=tmp_path,
        artifact_root="experiments/results/lightning_batch/test_tt5l_paired_axes",
    )

    decoded = json.loads(
        l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_json(plan)
    )
    report = render_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_markdown(
        plan
    )

    assert decoded["schema"] == L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_SCHEMA
    assert "L5 v2 TT5L side-info effect curve Lightning paired-axis plan" in report
    assert "score movement" in report
    assert "[contest-CPU]" in report
    assert "[contest-CUDA]" in report
    assert "CPU cells are `exact_cpu_eval`" in report
    assert "CUDA cells are `exact_cuda_eval`" in report


def test_tt5l_lightning_paired_axis_plan_cli_writes_json_and_markdown(
    tmp_path: Path,
) -> None:
    root = Path.cwd()
    artifact_root = root / "experiments/results/time_traveler_l5_v2" / (
        f"test_sideinfo_lightning_paired_axis_{tmp_path.name}"
    )
    output_json = artifact_root / "lightning_paired_axis_plan.json"
    output_md = artifact_root / "lightning_paired_axis_plan.md"
    dry_run_root = artifact_root / "dry_runs"
    try:
        manifest = _write_manifest(artifact_root)
        proc = subprocess.run(
            [
                str(
                    root
                    / "tools"
                    / "build_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan.py"
                ),
                "--variant-manifest",
                str(manifest),
                "--output-json",
                str(output_json.relative_to(root)),
                "--output-md",
                str(output_md.relative_to(root)),
                "--repo-root",
                str(artifact_root),
                "--artifact-root",
                str(dry_run_root.relative_to(artifact_root)),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert proc.returncode == 0, proc.stdout + proc.stderr
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        assert payload["cell_count"] == len(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS) * 2
        assert payload["all_cells_dry_run_ready"] is True
        assert "score_claim=false dispatch_attempted=false" in proc.stdout
        assert output_md.is_file()
        assert (dry_run_root / "zero/contest_cpu/state.json").is_file()
        assert (dry_run_root / "zero/contest_cuda/state.json").is_file()
    finally:
        if artifact_root.exists():
            shutil.rmtree(artifact_root)
