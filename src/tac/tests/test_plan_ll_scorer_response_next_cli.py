from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_plan_ll_scorer_response_next_cli_accepts_null_byte_matrix(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    matrix_path = tmp_path / "null_byte_matrix.json"
    json_out = tmp_path / "plan.json"
    md_out = tmp_path / "plan.md"

    dataset_path.write_text(
        json.dumps({"schema": "scorer_response_dataset.v1", "summary": {}, "rows": []}),
        encoding="utf-8",
    )
    matrix_path.write_text(
        json.dumps(
            {
                "schema": "null_byte_master_gradient_probe_matrix_v1",
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "axis_tag": "[predicted]",
                "n_anchors_probed_ok": 1,
                "top5_replacement_candidates": [
                    {
                        "substrate_label": "fec6",
                        "codec_family": "hnerv_family",
                        "scored_archive_sha256": "a" * 64,
                        "axis": "[contest-CUDA]",
                        "anchor_index": 1,
                        "n_null_bytes": 16292,
                        "null_fraction": 0.091,
                        "predicted_delta_s_per_seed_budget": {"K=16": -0.0108375},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_ll_scorer_response_next.py"),
            "--dataset",
            str(dataset_path),
            "--null-byte-matrix",
            str(matrix_path),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "ll_null_byte_procedural_codebook_candidates" in completed.stdout
    plan = json.loads(json_out.read_text(encoding="utf-8"))
    assert plan["probes"][0]["probe_id"] == "ll_null_byte_procedural_codebook_candidates"
    assert plan["probes"][0]["null_byte_priority_rows"][0]["priority_weight"] == 0.0108375
    assert "Null-Byte Matrix Priority" in md_out.read_text(encoding="utf-8")
