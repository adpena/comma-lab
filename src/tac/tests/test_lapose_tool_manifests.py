from __future__ import annotations

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

from tac.repo_io import json_text, sha256_file

REPO = Path(__file__).resolve().parents[3]


def test_lapose_cli_chain_records_deterministic_tool_manifests(tmp_path: Path) -> None:
    pair_metrics = tmp_path / "pair_metrics.json"
    component_response = tmp_path / "component_response.json"
    lite_out = tmp_path / "lapose_lite.json"
    records_out = tmp_path / "records.json"
    atoms_out = tmp_path / "atoms.json"
    pair_metrics.write_text(json.dumps(_pair_metrics()), encoding="utf-8")
    component_response.write_text(json.dumps(_component_response()), encoding="utf-8")

    _run_tool(
        "build_lapose_lite_inputs_from_pair_metrics.py",
        "--pair-metrics-json",
        str(pair_metrics),
        "--max-pairs",
        "2",
        "--json-out",
        str(lite_out),
    )
    _run_tool(
        "build_lapose_motion_records_from_component_response.py",
        "--component-response-json",
        str(component_response),
        "--latent-actions-json",
        str(lite_out),
        "--pair-opportunities-json",
        str(lite_out),
        "--json-out",
        str(records_out),
    )
    _run_tool(
        "build_lapose_motion_atom_manifest.py",
        "--records-json",
        str(records_out),
        "--base-pose-dist",
        "0.02",
        "--source",
        "lapose_cli_chain_fixture",
        "--json-out",
        str(atoms_out),
    )

    lite = json.loads(lite_out.read_text(encoding="utf-8"))
    records = json.loads(records_out.read_text(encoding="utf-8"))
    atoms = json.loads(atoms_out.read_text(encoding="utf-8"))

    _assert_tool_manifest(
        lite,
        tool="tools/build_lapose_lite_inputs_from_pair_metrics.py",
        input_paths=[pair_metrics],
        output_path=lite_out,
    )
    _assert_tool_manifest(
        records,
        tool="tools/build_lapose_motion_records_from_component_response.py",
        input_paths=[component_response, lite_out, lite_out],
        output_path=records_out,
    )
    _assert_tool_manifest(
        atoms,
        tool="tools/build_lapose_motion_atom_manifest.py",
        input_paths=[records_out],
        output_path=atoms_out,
    )
    assert atoms["ready_for_exact_eval_dispatch"] is False
    assert atoms["atom_ledger"]["ready_for_exact_eval_dispatch"] is False


def test_lapose_tool_manifest_payload_hash_ignores_wrapper_metadata(tmp_path: Path) -> None:
    pair_metrics = tmp_path / "pair_metrics.json"
    first_out = tmp_path / "first.json"
    second_out = tmp_path / "second.json"
    pair_metrics.write_text(json.dumps(_pair_metrics()), encoding="utf-8")

    base_args = [
        "build_lapose_lite_inputs_from_pair_metrics.py",
        "--pair-metrics-json",
        str(pair_metrics),
        "--max-pairs",
        "2",
        "--json-out",
    ]
    _run_tool(*base_args, str(first_out))
    _run_tool(*base_args, str(second_out))

    first = json.loads(first_out.read_text(encoding="utf-8"))
    second = json.loads(second_out.read_text(encoding="utf-8"))

    first_manifest = first["tool_run_manifest"]
    second_manifest = second["tool_run_manifest"]
    assert first_manifest["output_path"] != second_manifest["output_path"]
    assert (
        first_manifest["canonical_payload_without_tool_manifest_sha256"]
        == second_manifest["canonical_payload_without_tool_manifest_sha256"]
    )


def _run_tool(tool_name: str, *args: str) -> None:
    subprocess.run(
        [sys.executable, str(REPO / "tools" / tool_name), *args],
        check=True,
        cwd=REPO,
        text=True,
    )


def _assert_tool_manifest(
    payload: dict,
    *,
    tool: str,
    input_paths: list[Path],
    output_path: Path,
) -> None:
    manifest = payload["tool_run_manifest"]
    assert manifest["schema_version"] == 1
    assert manifest["tool"] == tool
    assert manifest["score_claim"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["output_path"] == output_path.as_posix()
    assert "--json-out" in manifest["argv"]
    assert manifest["input_files"] == [
        {
            "path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in input_paths
    ]
    without_manifest = dict(payload)
    without_manifest.pop("tool_run_manifest")
    expected = sha256(json_text(without_manifest).encode("utf-8")).hexdigest()
    assert manifest["canonical_payload_without_tool_manifest_sha256"] == expected


def _pair_metrics() -> dict:
    return {
        "schema_version": 1,
        "device": "cuda",
        "lane": "lane_w_hard_pair",
        "n_pairs": 4,
        "hardest_pair_indices": [2, 1],
        "per_pair_pose_dist": [0.1, 0.2, 0.8, 0.3],
        "per_pair_seg_dist": [0.01, 0.04, 0.08, 0.03],
        "per_pair_contrib": [1.0, 4.0, 9.0, 3.0],
    }


def _component_response() -> dict:
    return {
        "schema_version": 1,
        "score_claim": False,
        "device": "cuda",
        "promotion_eligible": False,
        "baseline_archive": {"bytes": 1000, "sha256": "2" * 64},
        "points": [
            {
                "epsilon": 0.0,
                "archive": {"bytes": 1000, "sha256": "2" * 64},
                "values": {"combined": 1.0, "segnet": 0.01, "posenet": 0.02},
            },
            {
                "epsilon": 0.5,
                "archive": {"bytes": 1012, "sha256": "3" * 64},
                "values": {"combined": 0.9, "segnet": 0.009, "posenet": 0.019},
            },
        ],
    }
