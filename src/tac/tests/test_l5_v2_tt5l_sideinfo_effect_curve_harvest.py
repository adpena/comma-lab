# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from tac.exact_eval_custody import contest_score
from tac.optimization.l5_v2_sideinfo_effect_curve import (
    build_l5_v2_sideinfo_effect_curve,
)
from tac.optimization.l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan import (
    build_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan,
)
from tac.optimization.l5_v2_tt5l_sideinfo_effect_curve_harvest import (
    L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_SCHEMA,
    build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan,
)
from tac.tests.test_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan import (
    _write_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _sha(seed: int) -> str:
    return hashlib.sha256(f"seed:{seed}".encode()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _plan(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    manifest_path = _write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["variants"]:
        variant = row["variant"]
        nonzero = 0 if variant in {"zero", "ablated"} else 8
        row["sideinfo_liveness"] = {
            "checked": True,
            "shape": [600, 45],
            "total_values": 27_000,
            "nonzero_values": nonzero,
            "nonzero_fraction": nonzero / 27_000,
        }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plan = build_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan(
        manifest=manifest,
        manifest_path=manifest_path,
        repo_root=tmp_path,
        artifact_root="experiments/results/lightning_batch/test_tt5l_paired_axes",
        generated_at_utc="2026-05-17T00:00:00Z",
    )
    plan_path = tmp_path / ".omx/research/lightning_paired_axis_plan.json"
    _write_json(plan_path, plan)
    return plan_path, plan


def _write_exact_eval_artifact(
    root: Path,
    cell: dict[str, object],
    *,
    seed: int = 10,
    archive_sha256: str | None = None,
    archive_size_bytes: int | None = None,
    filename: str = "contest_auth_eval.json",
    seg_dist: float = 0.001,
    score_axis: str | None = None,
    pair_group_id: str | None = None,
    run_id: str | None = None,
    runtime_content_tree_sha256: str | None = None,
    inflate_device: str | None = None,
    eval_device: str | None = None,
    hardware: str | None = None,
    gpu_model: str | None = None,
) -> Path:
    local_dir = root / str(cell["local_artifact_dir"])
    artifact = local_dir / filename
    log = local_dir / "contest_auth_eval.stdout.log"
    manifest = local_dir / "inflated_outputs_manifest.json"
    raw_sha = _sha(seed + 1000)
    _write_json(manifest, {"aggregate_sha256": raw_sha})
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("ok\n", encoding="utf-8")
    archive_bytes = (
        int(cell["archive_size_bytes"])
        if archive_size_bytes is None
        else archive_size_bytes
    )
    pose_dist = 0.0
    score = contest_score(seg_dist, pose_dist, archive_bytes)
    axis = str(cell["axis"])
    is_cuda = axis == "contest_cuda"
    observed_inflate_device = inflate_device or ("cuda" if is_cuda else "cpu")
    observed_eval_device = eval_device or ("cuda" if is_cuda else "cpu")
    observed_hardware = hardware or ("Tesla T4" if is_cuda else "Linux x86_64")
    observed_gpu_model = gpu_model if gpu_model is not None else ("Tesla T4" if is_cuda else "")
    pair = pair_group_id or str(cell["pair_group_id"])
    run = run_id or str(cell["run_id"])
    _write_json(
        artifact,
        {
            "archive_size_bytes": archive_bytes,
            "avg_posenet_dist": pose_dist,
            "avg_segnet_dist": seg_dist,
            "canonical_score": score,
            "n_samples": 600,
            "provenance": {
                "archive_sha256": archive_sha256 or cell["archive_sha256"],
                "device": "cuda" if is_cuda else "cpu",
                "gpu_model": observed_gpu_model,
                "hardware": observed_hardware,
                "inflate_runtime_manifest": {"runtime_tree_sha256": _sha(seed + 1)},
                "platform_machine": "x86_64",
                "platform_system": "Linux",
                "sys_argv": [
                    "experiments/contest_auth_eval.py",
                    "--device",
                    "cuda" if is_cuda else "cpu",
                    "--pair-group-id",
                    pair,
                    "--run-id",
                    run,
                ],
            },
            "raw_output_aggregate_sha256": raw_sha,
            "runtime_content_tree_sha256": runtime_content_tree_sha256 or _sha(seed + 2),
            "score_axis": score_axis or axis,
            "inflate_device": observed_inflate_device,
            "eval_device": observed_eval_device,
        },
    )
    return artifact


def _write_complete_exact_eval_harvest(root: Path, plan: dict[str, object]) -> None:
    runtime_content_by_variant = {
        str(cell["variant"]): _sha(20_000 + idx)
        for idx, cell in enumerate(
            {str(row["variant"]): row for row in plan["cells"]}.values()
        )
    }
    for cell in plan["cells"]:
        assert isinstance(cell, dict)
        variant = str(cell["variant"])
        seg = 0.001 if variant == "trained" else 0.002
        _write_exact_eval_artifact(
            root,
            cell,
            seed=len(variant) * 10 + (1 if cell["axis"] == "contest_cuda" else 0),
            seg_dist=seg,
            runtime_content_tree_sha256=runtime_content_by_variant[variant],
        )


def test_harvest_cells_preserve_plan_identity_when_artifacts_missing(
    tmp_path: Path,
) -> None:
    plan_path, plan = _plan(tmp_path)

    payload = build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan(
        plan=plan,
        plan_path=plan_path,
        repo_root=tmp_path,
    )

    assert payload["schema"] == L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_SCHEMA
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["cell_count"] == 10
    assert payload["harvested_exact_eval_artifact_count"] == 0
    assert payload["missing_exact_eval_artifact_count"] == 10
    assert any(
        str(blocker).startswith("harvested_exact_eval_artifact_missing:trained:")
        for blocker in payload["blockers"]
    )

    by_cell = {(cell["axis"], cell["variant"]): cell for cell in payload["cells"]}
    planned = {(cell["axis"], cell["variant"]): cell for cell in plan["cells"]}
    assert set(by_cell) == set(planned)
    for key, harvested in by_cell.items():
        source = planned[key]
        assert harvested["pair_group_id"] == source["pair_group_id"]
        assert harvested["run_id"] == source["run_id"]
        assert harvested["archive_sha256"] == source["archive_sha256"]
        assert harvested["sideinfo_liveness"]["checked"] is True
        assert harvested["evidence"]["pair_group_id"] == source["pair_group_id"]
        assert harvested["evidence"]["run_id"] == source["run_id"]


def test_harvest_cells_merge_exact_eval_without_dropping_pair_identity(
    tmp_path: Path,
) -> None:
    plan_path, plan = _plan(tmp_path)
    target = next(
        cell
        for cell in plan["cells"]
        if cell["axis"] == "contest_cpu" and cell["variant"] == "trained"
    )
    artifact = _write_exact_eval_artifact(tmp_path, target)

    payload = build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan(
        plan=plan,
        plan_path=plan_path,
        repo_root=tmp_path,
    )
    trained_cpu = next(
        cell
        for cell in payload["cells"]
        if cell["axis"] == "contest_cpu" and cell["variant"] == "trained"
    )

    assert payload["harvested_exact_eval_artifact_count"] == 1
    assert trained_cpu["blockers"] == []
    assert trained_cpu["pair_group_id"] == target["pair_group_id"]
    assert trained_cpu["run_id"] == target["run_id"]
    assert trained_cpu["evidence"]["pair_group_id"] == target["pair_group_id"]
    assert trained_cpu["evidence"]["run_id"] == target["run_id"]
    assert trained_cpu["evidence"]["artifact_path"] == artifact.relative_to(tmp_path).as_posix()
    assert trained_cpu["evidence"]["axis"] == "contest_cpu"

    curve = build_l5_v2_sideinfo_effect_curve(payload["cells"], repo_root=tmp_path)
    assert curve["predicate_passed"] is False
    assert not any(
        str(blocker).startswith("tt5l_sideinfo_effect_curve_cells_missing:")
        for blocker in curve["contract_blockers"]
    )
    assert any(
        str(blocker).startswith(
            "tt5l_sideinfo_effect_curve_cell_blocked:contest_cuda:trained"
        )
        for blocker in curve["contract_blockers"]
    )


def test_harvest_cells_block_archive_sha_and_byte_mismatches(
    tmp_path: Path,
) -> None:
    plan_path, plan = _plan(tmp_path)
    target = next(
        cell
        for cell in plan["cells"]
        if cell["axis"] == "contest_cuda" and cell["variant"] == "trained"
    )
    _write_exact_eval_artifact(
        tmp_path,
        target,
        archive_sha256=_sha(999),
        archive_size_bytes=int(target["archive_size_bytes"]) + 1,
    )

    payload = build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan(
        plan=plan,
        plan_path=plan_path,
        repo_root=tmp_path,
    )
    trained_cuda = next(
        cell
        for cell in payload["cells"]
        if cell["axis"] == "contest_cuda" and cell["variant"] == "trained"
    )

    assert (
        "harvested_exact_eval_archive_sha_mismatch:trained:contest_cuda"
        in trained_cuda["blockers"]
    )
    assert (
        "harvested_exact_eval_archive_bytes_mismatch:trained:contest_cuda"
        in trained_cuda["blockers"]
    )
    assert (
        "harvested_exact_eval_archive_sha_mismatch:trained:contest_cuda"
        in payload["blockers"]
    )
    curve = build_l5_v2_sideinfo_effect_curve(payload["cells"], repo_root=tmp_path)
    assert (
        "tt5l_sideinfo_effect_curve_cell_blocked:contest_cuda:trained:"
        "harvested_exact_eval_archive_sha_mismatch:trained:contest_cuda"
        in curve["contract_blockers"]
    )


def test_harvest_cells_complete_harvest_feeds_effect_curve_builder(
    tmp_path: Path,
) -> None:
    plan_path, plan = _plan(tmp_path)
    _write_complete_exact_eval_harvest(tmp_path, plan)

    payload = build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan(
        plan=plan,
        plan_path=plan_path,
        repo_root=tmp_path,
    )

    assert payload["ready_for_effect_curve_build"] is True
    assert payload["ready_cell_count"] == 10
    assert payload["harvested_exact_eval_artifact_count"] == 10
    assert payload["blockers"] == []
    curve = build_l5_v2_sideinfo_effect_curve(payload["cells"], repo_root=tmp_path)
    assert curve["predicate_passed"] is True
    assert curve["contract_blockers"] == []
    for effect in curve["axis_effects"].values():
        assert effect["trained_beats_or_ties_best_control"] is True


def test_harvest_cells_prefers_adjudicated_artifact(tmp_path: Path) -> None:
    plan_path, plan = _plan(tmp_path)
    _write_complete_exact_eval_harvest(tmp_path, plan)
    target = next(
        cell
        for cell in plan["cells"]
        if cell["axis"] == "contest_cpu" and cell["variant"] == "zero"
    )
    _write_exact_eval_artifact(
        tmp_path,
        target,
        filename="contest_auth_eval.adjudicated.json",
        seg_dist=0.0015,
    )

    payload = build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan(
        plan=plan,
        plan_path=plan_path,
        repo_root=tmp_path,
    )
    zero_cpu = next(
        cell
        for cell in payload["cells"]
        if cell["axis"] == "contest_cpu" and cell["variant"] == "zero"
    )

    assert zero_cpu["evidence"]["artifact_path"].endswith(
        "contest_auth_eval.adjudicated.json"
    )
    assert zero_cpu["ready_for_effect_curve_build"] is True


def test_harvest_cells_block_axis_pair_and_run_mismatch_before_metadata_fallback(
    tmp_path: Path,
) -> None:
    plan_path, plan = _plan(tmp_path)
    target = next(
        cell
        for cell in plan["cells"]
        if cell["axis"] == "contest_cuda" and cell["variant"] == "trained"
    )
    _write_exact_eval_artifact(
        tmp_path,
        target,
        score_axis="contest_cpu",
        pair_group_id="pair_wrong",
        run_id="run_wrong",
    )

    payload = build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan(
        plan=plan,
        plan_path=plan_path,
        repo_root=tmp_path,
    )
    trained_cuda = next(
        cell
        for cell in payload["cells"]
        if cell["axis"] == "contest_cuda" and cell["variant"] == "trained"
    )

    assert "harvested_exact_eval_axis_mismatch:trained:contest_cuda" in trained_cuda[
        "blockers"
    ]
    assert (
        "harvested_exact_eval_pair_group_id_mismatch:trained:contest_cuda"
        in trained_cuda["blockers"]
    )
    assert "harvested_exact_eval_run_id_mismatch:trained:contest_cuda" in trained_cuda[
        "blockers"
    ]


def test_harvest_cells_block_device_and_hardware_axis_mismatch(
    tmp_path: Path,
) -> None:
    plan_path, plan = _plan(tmp_path)
    target = next(
        cell
        for cell in plan["cells"]
        if cell["axis"] == "contest_cuda" and cell["variant"] == "trained"
    )
    _write_exact_eval_artifact(
        tmp_path,
        target,
        score_axis="contest_cuda",
        inflate_device="cpu",
        eval_device="cpu",
        hardware="Linux x86_64",
        gpu_model="",
    )

    payload = build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan(
        plan=plan,
        plan_path=plan_path,
        repo_root=tmp_path,
    )
    trained_cuda = next(
        cell
        for cell in payload["cells"]
        if cell["axis"] == "contest_cuda" and cell["variant"] == "trained"
    )

    assert (
        "harvested_exact_eval_custody_hardware_not_cuda:trained:contest_cuda"
        in trained_cuda["blockers"]
    )
    assert (
        "harvested_exact_eval_custody_inflate_device_not_cuda:trained:contest_cuda"
        in trained_cuda["blockers"]
    )
    assert (
        "harvested_exact_eval_custody_eval_device_not_cuda:trained:contest_cuda"
        in trained_cuda["blockers"]
    )
    assert payload["ready_for_effect_curve_build"] is False


def test_harvest_cells_block_wrong_plan_schema_and_extra_cells(tmp_path: Path) -> None:
    plan_path, plan = _plan(tmp_path)
    plan["schema"] = "wrong_schema"
    extra = dict(plan["cells"][0])
    extra["variant"] = "unsupported"
    plan["cells"].append(extra)

    payload = build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan(
        plan=plan,
        plan_path=plan_path,
        repo_root=tmp_path,
    )

    assert "lightning_paired_axis_plan_schema_mismatch" in payload["blockers"]
    assert "planned_cell_extra:contest_cpu:unsupported" in payload["blockers"]


def test_harvest_cells_cli_writes_builder_ready_json(tmp_path: Path) -> None:
    plan_path, plan = _plan(tmp_path)
    target = next(cell for cell in plan["cells"] if cell["axis"] == "contest_cpu")
    _write_exact_eval_artifact(tmp_path, target)
    output = tmp_path / ".omx/research/harvest_cells.json"

    proc = subprocess.run(
        [
            str(
                REPO_ROOT
                / "tools/build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan.py"
            ),
            "--lightning-plan-json",
            str(plan_path),
            "--output-json",
            str(output),
            "--repo-root",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "score_claim=false promotion_eligible=false" in proc.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["cell_count"] == 10
    assert payload["harvested_exact_eval_artifact_count"] == 1
    assert payload["cells"][0]["score_claim"] is False
    report = output.with_suffix(".md")
    assert report.exists()
    assert "test_tt5l_paired_axes" in report.read_text(encoding="utf-8")
