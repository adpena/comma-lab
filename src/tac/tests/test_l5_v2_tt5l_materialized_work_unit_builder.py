# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import torch

import tac.optimization.l5_staircase_v2 as l5_v2
from tac.optimization.l5_v2_tt5l_materialized_work_unit import (
    build_tt5l_materialized_paired_work_unit_plan,
    select_tt5l_variant_archive,
    tt5l_materialized_paired_work_unit_json,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_runtime(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "inflate.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")


def _write_tt5l_archive_zip(
    path: Path,
    *,
    nonzero_side_info: bool,
    num_pairs: int = 600,
) -> None:
    from tac.substrates.time_traveler_l5_autonomy.architecture import (
        TimeTravelerConfig,
        TimeTravelerSubstrate,
    )
    from tac.substrates.time_traveler_l5_autonomy.archive import pack_archive

    torch.manual_seed(0)
    cfg = TimeTravelerConfig(
        hidden_dim=8,
        num_hidden_layers=1,
        num_pairs=num_pairs,
        output_height=8,
        output_width=8,
        per_pair_side_info_bytes=45,
    )
    substrate = TimeTravelerSubstrate(cfg)
    side_info = np.zeros((cfg.num_pairs, cfg.per_pair_side_info_bytes), dtype=np.int8)
    if nonzero_side_info:
        side_info[:, :] = 1
    bin_bytes = pack_archive(
        world_model_state_dict=substrate.state_dict(),
        per_pair_side_info=side_info,
        meta={
            "int8_scale": 64.0,
            "first_omega": cfg.first_omega,
            "hidden_omega": cfg.hidden_omega,
            "coord_feature_freqs": cfg.coord_feature_freqs,
            "coord_dim": cfg.coord_dim,
            "markov_transition_band": cfg.markov_transition_band,
        },
        num_pairs=cfg.num_pairs,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
        foveation_grid_h=cfg.foveation_grid_h,
        foveation_grid_w=cfg.foveation_grid_w,
        pose_dim=cfg.pose_dim,
        per_pair_bytes=cfg.per_pair_side_info_bytes,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", bin_bytes)


def test_tt5l_materialized_work_unit_builder_outputs_valid_plan(tmp_path: Path) -> None:
    archive = tmp_path / "experiments/results/tt5l/random_lsb/archive.zip"
    runtime = tmp_path / "experiments/results/tt5l/runtime"
    _write_tt5l_archive_zip(archive, nonzero_side_info=True)
    _write_runtime(runtime)

    payload = build_tt5l_materialized_paired_work_unit_plan(
        archive=archive,
        submission_dir=runtime,
        repo_root=tmp_path,
        materialized_from={"variant": "random_lsb"},
    )
    plan_path = tmp_path / l5_v2.TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PLAN_ARTIFACT_PATH
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(tt5l_materialized_paired_work_unit_json(payload), encoding="utf-8")

    status = l5_v2._tt5l_materialized_paired_work_unit_status(repo_root=tmp_path)

    assert status["artifact_valid"] is True
    assert status["blockers"] == []
    assert status["tt5l_sideinfo_stats"]["num_pairs"] == 600
    assert status["tt5l_sideinfo_stats"]["nonzero_values"] == 27_000
    assert payload["archive"]["path"] == "experiments/results/tt5l/random_lsb/archive.zip"
    assert payload["archive"]["sha256"] == _sha256(archive)
    assert payload["runtime"]["submission_dir"] == "experiments/results/tt5l/runtime"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["ready_for_provider_dispatch"] is False
    assert payload["axes_skipped_due_to_existing_anchor"] == {
        "contest_cpu": False,
        "contest_cuda": False,
    }
    assert payload["existing_anchors_reused"] == {
        "contest_cpu": None,
        "contest_cuda": None,
    }
    assert all(
        str(tmp_path) not in token
        for command in payload["commands"].values()
        for token in command
    )


def test_select_tt5l_variant_archive_verifies_manifest_sha(tmp_path: Path) -> None:
    archive = tmp_path / "experiments/results/tt5l/random_lsb/archive.zip"
    _write_tt5l_archive_zip(archive, nonzero_side_info=True)
    manifest = tmp_path / ".omx/research/variants.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "source_archive_path": "experiments/results/tt5l/source/archive.zip",
                "source_archive_sha256": "a" * 64,
                "source_sideinfo_liveness": {"nonzero_values": 0},
                "variants": [
                    {
                        "variant": "random_lsb",
                        "archive_path": "experiments/results/tt5l/random_lsb/archive.zip",
                        "archive_sha256": _sha256(archive),
                        "archive_bytes": archive.stat().st_size,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    selected = select_tt5l_variant_archive(
        variant_manifest=manifest,
        variant="random_lsb",
        repo_root=tmp_path,
    )

    assert selected["variant"] == "random_lsb"
    assert selected["archive_path"] == "experiments/results/tt5l/random_lsb/archive.zip"
    assert selected["archive_sha256"] == _sha256(archive)
    assert selected["variant_manifest_path"] == ".omx/research/variants.json"
