# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
PLANNER_PATH = REPO / "experiments" / "plan_c102_native_action_atoms.py"


def _load_planner() -> Any:
    spec = importlib.util.spec_from_file_location("c102_native_action_atoms_test", PLANNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _trace(
    planner: Any,
    samples: dict[int, tuple[float, float]],
    *,
    label: str,
    archive_bytes: int = 276_485,
) -> Any:
    return planner.LoadedTrace(
        label=label,
        path=Path(f"{label}.json"),
        archive_bytes=archive_bytes,
        archive_sha256="0" * 64,
        score=0.0,
        seg_score=sum(seg for seg, _pose in samples.values()),
        pose_score=sum(pose for _seg, pose in samples.values()),
        samples_by_pair={
            pair: {"seg": seg, "pose": pose, "combined": seg + pose}
            for pair, (seg, pose) in samples.items()
        },
    )


def _write_trace(path: Path, samples: dict[int, tuple[float, float]], *, bytes_: int) -> None:
    payload = {
        "archive_size_bytes": bytes_,
        "score_recomputed_from_components": 0.315,
        "score_seg_contribution": sum(seg for seg, _pose in samples.values()),
        "score_pose_contribution": sum(pose for _seg, pose in samples.values()),
        "samples": [
            {
                "pair_index": pair,
                "score_seg_contribution_exact": seg,
                "score_pose_contribution_first_order": pose,
                "score_combined_contribution_first_order": seg + pose,
            }
            for pair, (seg, pose) in sorted(samples.items())
        ],
        "trace_inputs": {"archive_sha256": "1" * 64},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_manifest(path: Path, *, action_id: int = 9) -> None:
    payload = {
        "selected_records": [
            {
                "pair_index": 1,
                "tile_id": 88,
                "action_id": action_id,
                "source_action_id": action_id,
                "source_index": 3,
                "transform": "identity",
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_noop_guard_marks_anchor_duplicate_not_dispatchable() -> None:
    planner = _load_planner()
    anchor = _trace(planner, {1: (0.010, 0.020)}, label="anchor")
    candidate = _trace(planner, {1: (0.008, 0.010)}, label="candidate")
    anchor_record = planner.ActionRecord(pair_index=1, tile_id=88, action_id=7)
    observation = planner.Observation(
        label="obs",
        trace=candidate,
        action_records=(planner.ActionRecord(pair_index=1, tile_id=88, action_id=7),),
        family="unit",
        evidence_role="test",
        attribution="pair_delta_equal_share",
        confidence=1.0,
    )

    atoms = planner._aggregate_atoms(
        anchor=anchor,
        observations=[observation],
        anchor_records=[anchor_record],
        pose_heavy_pairs=set(),
    )

    assert atoms[0]["classification"] == "exact_anchor_duplicate_noop"
    assert atoms[0]["no_op_status"] == "exact_anchor_duplicate"
    assert atoms[0]["dispatchable_atom"] is False


def test_policy_rows_include_required_fields_and_dispatch_false() -> None:
    planner = _load_planner()
    atom = {
        "action_id": 9,
        "atom_id": "pair001_tile088_src009_act009_identity",
        "classification": "component_positive_pose_safe",
        "evidence_sources": [{"label": "unit_trace"}],
        "mean_weighted_combined_equal_share": 0.001,
        "no_op_relative_to_c102": False,
        "no_op_status": "changes_c102_action_stream_if_built",
        "pair_index": 1,
        "source_action_id": 9,
        "tile_id": 88,
        "transform": "identity",
    }

    policies = planner._build_policy_rows(
        atoms=[atom],
        anchor_records=[planner.ActionRecord(pair_index=2, tile_id=88, action_id=7)],
        existing_manifests={},
    )

    required = {
        "candidate_policy_id",
        "source_priors",
        "records_selected",
        "charged_byte_proxy",
        "actual_builder_bytes",
        "expected_component_benefit_proxy",
        "break_even_vs_0_31",
        "no_op_status",
        "dispatchable",
    }
    assert policies
    assert required.issubset(policies[0])
    assert all(policy["dispatchable"] is False for policy in policies)


def test_build_policy_is_deterministic_with_required_outputs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    planner = _load_planner()
    anchor_trace = tmp_path / "anchor_trace.json"
    candidate_trace = tmp_path / "candidate_trace.json"
    manifest = tmp_path / "manifest.json"
    anchor_archive = tmp_path / "missing_anchor.zip"
    output_a = tmp_path / "out_a"
    output_b = tmp_path / "out_b"
    _write_trace(anchor_trace, {1: (0.010, 0.020)}, bytes_=276_485)
    _write_trace(candidate_trace, {1: (0.008, 0.010)}, bytes_=276_330)
    _write_manifest(manifest, action_id=9)

    monkeypatch.setattr(
        planner,
        "_read_archive_action_records",
        lambda _path, label: (
            planner.ActionRecord(pair_index=1, tile_id=88, action_id=7, evidence_source=label),
        ),
    )
    spec = planner.ObservationSpec(
        label="unit_positive",
        trace_path=candidate_trace,
        manifest_path=manifest,
        family="unit",
        evidence_role="test",
        attribution="pair_delta_equal_share",
        confidence=1.0,
    )

    policy_a = planner.build_policy(
        output_dir=output_a,
        anchor_trace_path=anchor_trace,
        anchor_archive=anchor_archive,
        existing_action_build_dir=tmp_path / "no_builds",
        c101_negative_eval=tmp_path / "missing_eval.json",
        specs=[spec],
    )
    policy_b = planner.build_policy(
        output_dir=output_b,
        anchor_trace_path=anchor_trace,
        anchor_archive=anchor_archive,
        existing_action_build_dir=tmp_path / "no_builds",
        c101_negative_eval=tmp_path / "missing_eval.json",
        specs=[spec],
    )

    assert policy_a == policy_b
    assert policy_a["dispatch_decision"]["exact_eval_justified"] is False
    assert (output_a / "ranked_atom_policy.json").exists()
    assert (output_a / "ranked_atoms.csv").exists()
    assert (output_a / "ranked_policies.csv").exists()
