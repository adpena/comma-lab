from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.repo_io import write_json


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools/plan_pr103_arithmetic_transform.py"


def test_plan_pr103_arithmetic_transform_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    manifest = tmp_path / "schema_manifest.json"
    json_out = tmp_path / "plan.json"
    md_out = tmp_path / "plan.md"
    write_json(manifest, _manifest())

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--schema-manifest",
            str(manifest),
            "--target-label",
            "blocks.1.weight",
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    plan = json.loads(json_out.read_text(encoding="utf-8"))
    assert plan["target_stream"]["label"] == "blocks.1.weight"
    assert plan["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert plan["tool_run_manifest"]["ready_for_exact_eval_dispatch"] is False
    assert "candidate_runtime_adapter_missing" in plan["readiness_blockers"]
    assert "blocks.1.weight" in md_out.read_text(encoding="utf-8")


def test_plan_pr103_arithmetic_transform_cli_fails_closed_for_unknown_target(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "schema_manifest.json"
    write_json(manifest, _manifest())

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--schema-manifest",
            str(manifest),
            "--target-label",
            "missing.weight",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "target label not found" in proc.stderr


def _manifest() -> dict:
    return {
        "planning_only": True,
        "score_claim": False,
        "ready_for_schema_review": True,
        "source_archive": {
            "path": "experiments/results/pr103/archive.zip",
            "bytes": 178223,
            "sha256": "a" * 64,
            "member_name": "x",
            "member_bytes": 178111,
            "member_sha256": "b" * 64,
        },
        "merged_arithmetic_stream": {
            "source_bytes": 153856,
            "source_sha256": "c" * 64,
            "decoded_symbol_count": 237561,
            "decoder_maybe_exhausted": True,
            "reencoded_byte_identical": True,
        },
        "next_arithmetic_schema_targets": [
            {
                "label": "stem.weight",
                "role": "ac_weight_tensor",
                "schema_index": 0,
                "symbol_count": 48384,
                "alphabet_size": 256,
                "decoded_symbols_sha256": "d" * 64,
                "observed_entropy_bytes_floor": 31582,
                "model_cross_entropy_bytes_floor": 31627,
                "model_gap_bytes_estimate": 46,
            },
            {
                "label": "blocks.1.weight",
                "role": "ac_weight_tensor",
                "schema_index": 4,
                "symbol_count": 46656,
                "alphabet_size": 256,
                "decoded_symbols_sha256": "e" * 64,
                "observed_entropy_bytes_floor": 32434,
                "model_cross_entropy_bytes_floor": 32478,
                "model_gap_bytes_estimate": 45,
            },
        ],
    }
