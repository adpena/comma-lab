# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from tac.exact_eval_custody import contest_score
from tac.optimization.l5_v2_measurement_schedule import (
    L5V2_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH,
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES,
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS,
    validate_l5_v2_sideinfo_effect_curve,
)
from tac.optimization.l5_v2_sideinfo_effect_curve import (
    build_l5_v2_sideinfo_effect_curve,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _sha(seed: int) -> str:
    return hashlib.sha256(f"seed:{seed}".encode()).hexdigest()


def _write_manifest(path: Path, *, aggregate_sha: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"aggregate_sha256": aggregate_sha}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell(
    repo_root: Path,
    *,
    axis: str,
    variant: str,
    seg_dist: float,
    seed: int,
    hardware: str | None = None,
) -> dict[str, object]:
    base = repo_root / "evidence" / axis / variant
    base.mkdir(parents=True, exist_ok=True)
    artifact = base / "contest_auth_eval.json"
    log = base / "auth_eval.log"
    artifact.write_text("{}\n", encoding="utf-8")
    log.write_text("ok\n", encoding="utf-8")
    raw_sha = _sha(seed + 1000)
    manifest = base / "inflated_outputs_manifest.json"
    manifest_sha = _write_manifest(manifest, aggregate_sha=raw_sha)
    archive_bytes = 1000 + seed
    pose_dist = 0.0
    score = contest_score(seg_dist, pose_dist, archive_bytes)
    is_cuda = axis == "contest_cuda"
    return {
        "axis": axis,
        "variant": variant,
        "evidence": {
            "axis": axis,
            "archive_sha256": _sha(seed),
            "runtime_tree_sha256": _sha(seed + 1),
            "score": score,
            "seg_dist": seg_dist,
            "pose_dist": pose_dist,
            "archive_bytes": archive_bytes,
            "n_samples": 600,
            "hardware": hardware
            or ("modal-t4 cuda" if is_cuda else "linux-x86_64 cpu"),
            "inflate_device": "cuda" if is_cuda else "cpu",
            "eval_device": "cuda" if is_cuda else "cpu",
            "auth_eval_command": (
                "contest_auth_eval --device cuda"
                if is_cuda
                else "contest_auth_eval --device cpu"
            ),
            "artifact_path": str(artifact.relative_to(repo_root)),
            "log_path": str(log.relative_to(repo_root)),
            "inflated_outputs_manifest_path": str(manifest.relative_to(repo_root)),
            "inflated_outputs_manifest_sha256": manifest_sha,
            "raw_output_aggregate_sha256": raw_sha,
        },
    }


def _complete_cells(repo_root: Path, *, trained_seg: float = 0.001) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    seed = 10
    for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES:
        for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS:
            seg = trained_seg if variant == "trained" else 0.002
            cells.append(
                _cell(
                    repo_root,
                    axis=axis,
                    variant=variant,
                    seg_dist=seg,
                    seed=seed,
                )
            )
            seed += 1
    return cells


def test_sideinfo_effect_curve_builder_passes_only_with_paired_trained_win(
    tmp_path: Path,
) -> None:
    payload = build_l5_v2_sideinfo_effect_curve(
        _complete_cells(tmp_path),
        repo_root=tmp_path,
    )

    assert payload["predicate_passed"] is True
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["contract_blockers"] == []
    assert payload["effect_blockers"] == []
    assert len(payload["observed_cells"]) == 10
    assert validate_l5_v2_sideinfo_effect_curve(payload) == []
    for effect in payload["axis_effects"].values():
        assert effect["trained_beats_or_ties_best_control"] is True


def test_sideinfo_effect_curve_builder_fails_when_trained_loses(tmp_path: Path) -> None:
    payload = build_l5_v2_sideinfo_effect_curve(
        _complete_cells(tmp_path, trained_seg=0.003),
        repo_root=tmp_path,
    )

    assert payload["predicate_passed"] is False
    assert "trained_not_best_or_tied:contest_cpu" in payload["effect_blockers"]
    assert "trained_not_best_or_tied:contest_cuda" in payload["effect_blockers"]


def test_sideinfo_effect_curve_builder_keeps_invalid_custody_blockers(
    tmp_path: Path,
) -> None:
    cells = _complete_cells(tmp_path)
    cells[0] = _cell(
        tmp_path,
        axis="contest_cpu",
        variant="zero",
        seg_dist=0.002,
        seed=999,
        hardware="x86_64 cpu host",
    )

    payload = build_l5_v2_sideinfo_effect_curve(cells, repo_root=tmp_path)

    assert payload["predicate_passed"] is False
    zero_cpu = next(
        row
        for row in payload["observed_cells"]
        if row["axis"] == "contest_cpu" and row["variant"] == "zero"
    )
    assert "exact_eval_hardware_not_contest_cpu" in zero_cpu["blockers"]
    assert any(
        str(blocker).startswith("tt5l_sideinfo_effect_curve_cell_blocked")
        for blocker in validate_l5_v2_sideinfo_effect_curve(payload)
    )


def test_sideinfo_effect_curve_default_artifact_is_tracked_research_surface() -> None:
    assert L5V2_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH.startswith(".omx/research/")
    assert L5V2_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH.endswith(".json")
    assert "experiments/results/" not in L5V2_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH


def test_sideinfo_effect_curve_cli_writes_schedule_consumable_artifact(
    tmp_path: Path,
) -> None:
    artifact_root = (
        REPO_ROOT
        / "experiments"
        / "results"
        / "time_traveler_l5_v2"
        / f"test_sideinfo_effect_curve_{tmp_path.name}"
    )
    output_path = artifact_root / "tt5l_sideinfo_effect_curve.json"
    cells_path = tmp_path / "cells.json"
    cells_path.write_text(
        json.dumps(_complete_cells(tmp_path), sort_keys=True),
        encoding="utf-8",
    )
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "tools/build_l5_v2_sideinfo_effect_curve.py",
                "--repo-root",
                str(tmp_path),
                "--cell-json",
                str(cells_path),
                "--output-json",
                str(output_path.relative_to(REPO_ROOT)),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "predicate_passed=true" in proc.stdout
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert validate_l5_v2_sideinfo_effect_curve(payload) == []
    finally:
        if artifact_root.exists():
            import shutil

            shutil.rmtree(artifact_root)
