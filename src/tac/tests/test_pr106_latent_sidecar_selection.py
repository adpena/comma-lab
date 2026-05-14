# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

from tac.packet_compiler.pr106_latent_sidecar_selection import (
    build_latent_candidate_grid,
    choose_latent_corrections_from_scores,
    latent_candidate_grid_npy_sha256,
    profile_latent_sidecar_topk_pareto,
    validate_score_table_manifest,
)

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "profile_pr106_latent_sidecar_topk_pareto.py"


def test_score_table_selection_is_stable_and_nonpromotable() -> None:
    candidates = build_latent_candidate_grid(latent_dim=2, delta_radius=1)
    scores = np.array(
        [
            [10.0, 9.5, 10.2, 9.9, 11.0],
            [10.0, 10.1, 9.2, 9.4, 10.3],
            [10.0, 10.0, 10.1, 10.2, 10.3],
            [10.0, 9.7, 10.4, 9.3, 10.2],
        ],
        dtype=np.float32,
    )

    dims, deltas, diagnostics = choose_latent_corrections_from_scores(
        scores,
        candidates,
        top_k=2,
    )

    assert dims.tolist() == [255, 0, 255, 1]
    assert deltas.tolist() == [0, 1, 0, -1]
    assert diagnostics["selected_nonzero_pair_count"] == 2
    assert diagnostics["selected_improvement_sum"] == 1.5


def test_topk_pareto_profile_keeps_zero_and_full_rows() -> None:
    candidates = build_latent_candidate_grid(latent_dim=2, delta_radius=1)
    scores = np.array(
        [
            [10.0, 9.5, 10.2, 9.9, 11.0],
            [10.0, 10.1, 9.2, 9.4, 10.3],
            [10.0, 10.0, 10.1, 10.2, 10.3],
            [10.0, 9.7, 10.4, 9.3, 10.2],
        ],
        dtype=np.float32,
    )

    profile = profile_latent_sidecar_topk_pareto(
        scores,
        candidates,
        top_k_values=[1],
    )

    assert profile["score_claim"] is False
    assert profile["ready_for_exact_eval_dispatch"] is False
    assert profile["selector_objective_is_exact_score"] is False
    assert profile["evaluated_top_k_values"] == [0, 1, 3]
    gains = [row["selector_improvement_sum"] for row in profile["rows"]]
    assert gains == sorted(gains)
    assert profile["rows"][0]["n_corrections"] == 0
    assert profile["rows"][-1]["n_corrections"] == 3
    assert profile["pareto_frontier"]


def test_topk_pareto_tool_writes_non_score_report(tmp_path: Path) -> None:
    candidates = build_latent_candidate_grid(latent_dim=2, delta_radius=1)
    scores = np.array(
        [
            [10.0, 9.5, 10.2, 9.9, 11.0],
            [10.0, 10.1, 9.2, 9.4, 10.3],
            [10.0, 10.0, 10.1, 10.2, 10.3],
            [10.0, 9.7, 10.4, 9.3, 10.2],
        ],
        dtype=np.float32,
    )
    table = tmp_path / "score_table.npy"
    np.save(table, scores, allow_pickle=False)
    out = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--score-table-npy",
            str(table),
            "--n-pairs",
            "4",
            "--latent-dim",
            "2",
            "--delta-radius",
            "1",
            "--top-k-values",
            "1,2",
            "--json-out",
            str(out),
        ],
        check=True,
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema"] == "pr106_latent_sidecar_topk_pareto_report_v1"
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["profile"]["score_table_shape"] == [4, len(candidates)]
    assert report["profile"]["evaluated_top_k_values"] == [0, 1, 2, 3]


def test_score_table_manifest_accepts_single_member_x_payload_match(tmp_path: Path) -> None:
    candidates = build_latent_candidate_grid(latent_dim=2, delta_radius=1)
    scores = np.zeros((4, len(candidates)), dtype=np.float32)
    table = tmp_path / "score_table.npy"
    np.save(table, scores, allow_pickle=False)
    archive = tmp_path / "archive.zip"
    payload = b"fixture-pr106-payload"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo("x", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        zf.writestr(info, payload)
    manifest = tmp_path / "score_table_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_schema": "pr106_latent_score_table_manifest_v1",
                "producer": "experiments/build_pr106_latent_score_table.py",
                "score_claim": False,
                "ready_for_builder": True,
                "source_archive_sha256": "a" * 64,
                "source_zero_bin_sha256": hashlib.sha256(payload).hexdigest(),
                "score_table_npy_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
                "candidate_grid_sha256": latent_candidate_grid_npy_sha256(candidates),
                "n_pairs": 4,
                "latent_dim": 2,
                "delta_radius": 1,
                "candidate_count": len(candidates),
                "score_table_shape": [4, len(candidates)],
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "remote_jobs_dispatched": False,
            }
        ),
        encoding="utf-8",
    )
    validated = validate_score_table_manifest(
        manifest,
        score_table_npy=table,
        source_archive=archive,
        n_pairs=4,
        latent_dim=2,
        delta_radius=1,
        candidate_count=len(candidates),
    )

    assert validated["validated_source_archive_sha256_match"] is False
    assert validated["validated_source_zero_bin_sha256_match"] is True
