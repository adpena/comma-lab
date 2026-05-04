from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "plan_c067_postdecode_mask_repair_waterfill.py"
SPEC = importlib.util.spec_from_file_location("plan_c067_postdecode_mask_repair_waterfill", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
planner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


def _write_trace(path: Path, *, score_claim: bool = False, cross_check: bool = True) -> None:
    samples = [
        (0, 0.10, 0.20, 0.30),
        (1, 0.02, 0.03, 0.05),
        (2, 0.08, 0.17, 0.25),
        (3, 0.01, 0.01, 0.02),
    ]
    payload = {
        "schema_version": 1,
        "score_claim": score_claim,
        "evidence_grade": "diagnostic_component_trace",
        "n_samples": 4,
        "archive_size_bytes": 233_612,
        "avg_posenet_dist": 0.10,
        "avg_segnet_dist": 0.02,
        "score_recomputed_from_components": 1.5,
        "contest_auth_eval_cross_check": {
            "all_match": cross_check,
            "contest_auth_eval_json_sha256": "c" * 64,
        },
        "trace_inputs": {
            "device": "cuda:0",
            "cuda_device_name": "NVIDIA L40S",
            "archive_sha256": "d" * 64,
        },
        "samples": [
            {
                "pair_index": pair,
                "frame_indices": [2 * pair, 2 * pair + 1],
                "posenet_dist": pose,
                "segnet_dist": seg,
                "score_pose_contribution_first_order": pose_term,
                "score_seg_contribution_exact": seg_term,
                "score_combined_contribution_first_order": combined,
            }
            for pair, pose_term, seg_term, combined in samples
            for pose, seg in [(0.001 + pair * 0.001, 0.002 + pair * 0.001)]
        ],
    }
    path.write_text(json.dumps(payload))


def _manifest_payload(*, compressed_bytes: int, atoms: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema": "c067_postdecode_mask_repair_candidate_v1",
        "tool": "experiments/build_c067_postdecode_mask_repair_candidate.py",
        "score_claim": False,
        "promotion_eligible": False,
        "archive": {
            "path": "/tmp/archive.zip",
            "size_bytes": 250_000 + compressed_bytes,
            "sha256": "a" * 64,
            "delta_vs_base_bytes": -1,
            "rate_term_delta_vs_base": -0.1,
        },
        "repair_selector": {
            "atom_granularity": "frame_class",
            "selected_repair_pixels": sum(int(atom["changed_pixels"]) for atom in atoms),
            "selected_atoms": atoms,
            "policy": {
                "policy": "frame_indices",
                "atom_granularity": "frame_class",
                "max_atoms": None,
                "max_repair_payload_bytes": None,
                "pair_indices": [],
                "frame_indices": sorted({int(atom["frames"][0]) for atom in atoms}),  # type: ignore[index]
                "class_ids": [],
                "label": "unit",
            },
        },
        "repair_payload": {
            "archive_member": "alpha4_residual_repair.amr1.xz",
            "compressor": "lzma_xz",
            "compressed_size_bytes": compressed_bytes,
        },
    }


def _atom(frame: int, class_id: int, pixels: int) -> dict[str, object]:
    return {
        "atom_id": f"frame{frame:04d}_class{class_id}",
        "frames": [frame],
        "pair_indices": [frame // 2],
        "class_id": class_id,
        "changed_pixels": pixels,
    }


def test_waterfill_uses_component_trace_and_nested_compressed_byte_marginals(tmp_path: Path) -> None:
    trace = tmp_path / "component_trace.json"
    _write_trace(trace)
    atoms_prefix = [_atom(0, 1, 100), _atom(1, 1, 50)]
    atoms_full = atoms_prefix + [_atom(2, 1, 100)]
    manifest_a = tmp_path / "manifest_a.json"
    manifest_b = tmp_path / "manifest_b.json"
    manifest_a.write_text(json.dumps(_manifest_payload(compressed_bytes=300, atoms=atoms_prefix)))
    manifest_b.write_text(json.dumps(_manifest_payload(compressed_bytes=450, atoms=atoms_full)))
    output = tmp_path / "plan.json"

    payload = planner.build_waterfill_plan(
        component_trace_specs=[("save12k_l40s", trace)],
        manifest_paths=[manifest_a, manifest_b],
        output_json=output,
        budgets=(160, 350),
        expected_samples=4,
        label_prefix="unit_waterfill",
    )

    assert output.exists()
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["cuda_jobs_launched"] is False
    assert payload["evidence_grade"] == "empirical_planning_non_promotable"
    assert payload["top_atoms"][0]["atom_id"] == "frame0002_class1"
    assert payload["top_atoms"][0]["estimated_payload_bytes"] == 150.0
    assert payload["top_atoms"][0]["byte_estimate_method"] == "nested_prefix_marginal_median"

    small, large = payload["budget_policies"]
    assert small["budget_payload_bytes"] == 160
    assert small["selected_atom_count"] == 1
    assert small["selected_atoms"][0]["atom_id"] == "frame0002_class1"
    assert small["builder_contract"]["policy_json"]["policy"] == "frame_indices"
    assert small["builder_contract"]["policy_json"]["frame_indices"] == [2]
    assert small["expected_marginal_score_terms"]["break_even_under_first_order_prior"] is True

    assert large["budget_payload_bytes"] == 350
    assert large["selected_atom_count"] == 2
    assert large["builder_contract"]["policy_json"]["frame_indices"] == [0, 2]


def test_waterfill_rejects_promotable_trace_or_manifest(tmp_path: Path) -> None:
    trace = tmp_path / "component_trace.json"
    _write_trace(trace, score_claim=True)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest_payload(compressed_bytes=100, atoms=[_atom(0, 1, 10)])))

    with pytest.raises(planner.WaterfillPlanError, match="score_claim"):
        planner.build_waterfill_plan(
            component_trace_specs=[("bad", trace)],
            manifest_paths=[manifest],
            expected_samples=4,
        )

    _write_trace(trace, score_claim=False)
    payload = _manifest_payload(compressed_bytes=100, atoms=[_atom(0, 1, 10)])
    payload["promotion_eligible"] = True
    manifest.write_text(json.dumps(payload))

    with pytest.raises(planner.WaterfillPlanError, match="promotion_eligible"):
        planner.build_waterfill_plan(
            component_trace_specs=[("ok", trace)],
            manifest_paths=[manifest],
            expected_samples=4,
        )


def test_cli_writes_non_promotable_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    trace = tmp_path / "component_trace.json"
    _write_trace(trace)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest_payload(compressed_bytes=100, atoms=[_atom(0, 1, 10)])))
    output = tmp_path / "plan.json"

    rc = planner.main(
        [
            "--component-trace",
            f"trace={trace}",
            "--archive-manifest",
            str(manifest),
            "--output-json",
            str(output),
            "--budget-bytes",
            "50,100",
            "--expected-samples",
            "4",
            "--top-k",
            "4",
        ]
    )

    assert rc == 0
    written = json.loads(output.read_text())
    assert written["score_claim"] is False
    assert written["promotion_eligible"] is False
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["output_json"] == str(output)
    assert stdout["budget_policy_count"] == 2
