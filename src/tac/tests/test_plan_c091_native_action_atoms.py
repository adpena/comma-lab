# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
PLANNER_PATH = REPO / "experiments" / "plan_c091_native_action_atoms.py"


def _load_planner() -> Any:
    spec = importlib.util.spec_from_file_location("c091_native_action_atoms_test", PLANNER_PATH)
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
    archive_bytes: int = 276_481,
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
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")


def _write_manifest(path: Path) -> None:
    payload = {
        "selected_records": [
            {
                "pair_index": 1,
                "tile_id": 88,
                "action_id": 7,
                "source_action_id": 9,
                "source_index": 3,
                "transform": "amp_shift_-1",
            }
        ]
    }
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")


def _write_single_p_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        info = zipfile.ZipInfo("p", (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o644 << 16
        info.create_system = 3
        zf.writestr(info, b"payload")


def test_parse_action_records_requires_four_byte_records() -> None:
    planner = _load_planner()

    records = planner._parse_action_records(
        (42).to_bytes(2, "little") + bytes([88, 7]),
        evidence_source="unit",
    )

    assert len(records) == 1
    assert records[0].pair_index == 42
    assert records[0].tile_id == 88
    assert records[0].action_id == 7

    try:
        planner._parse_action_records(b"abc", evidence_source="bad")
    except ValueError as exc:
        assert "divisible by 4" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected invalid action record length to fail")


def test_pair_deltas_use_c091_minus_candidate_sign() -> None:
    planner = _load_planner()
    anchor = _trace(planner, {1: (0.010, 0.020)}, label="anchor")
    candidate = _trace(planner, {1: (0.008, 0.025)}, label="candidate")

    deltas = planner._pair_deltas(anchor, candidate)

    assert deltas[1]["seg_delta_vs_c091"] == 0.002
    assert deltas[1]["pose_delta_vs_c091"] == -0.005000000000000001
    assert deltas[1]["combined_delta_vs_c091"] == -0.003000000000000001


def test_aggregate_marks_pose_heavy_pairs_not_dispatchable() -> None:
    planner = _load_planner()
    anchor = _trace(planner, {1: (0.010, 0.020)}, label="anchor")
    candidate = _trace(planner, {1: (0.008, 0.010)}, label="candidate")
    observation = planner.Observation(
        label="obs",
        trace=candidate,
        action_records=(
            planner.ActionRecord(
                pair_index=1,
                tile_id=88,
                action_id=7,
                source_action_id=9,
                source_index=3,
                transform="amp_shift_-1",
            ),
        ),
        family="unit",
        evidence_role="test",
        attribution="equal_share",
        confidence=1.0,
    )

    atoms = planner._aggregate_atoms(
        anchor=anchor,
        observations=[observation],
        pose_heavy_pairs={1},
    )

    assert atoms[0]["classification"] == "pose_toxic_pair"
    assert atoms[0]["dispatchable_atom"] is False


def test_build_policy_emits_no_dispatch_when_upper_bound_misses_break_even(
    tmp_path: Path,
) -> None:
    planner = _load_planner()
    anchor_trace = tmp_path / "anchor_trace.json"
    candidate_trace = tmp_path / "candidate_trace.json"
    manifest = tmp_path / "manifest.json"
    anchor_archive = tmp_path / "anchor.zip"
    output_dir = tmp_path / "out"
    _write_trace(anchor_trace, {1: (0.010, 0.020)}, bytes_=276_481)
    _write_trace(candidate_trace, {1: (0.00999, 0.01999)}, bytes_=276_329)
    _write_manifest(manifest)
    _write_single_p_zip(anchor_archive)
    spec = planner.ObservationSpec(
        label="tiny_positive",
        trace_path=candidate_trace,
        manifest_path=manifest,
        family="unit",
        evidence_role="test",
        attribution="pair_delta_equal_share",
        confidence=1.0,
    )

    policy = planner.build_policy(
        output_dir=output_dir,
        anchor_trace_path=anchor_trace,
        anchor_archive=anchor_archive,
        specs=[spec],
    )

    assert policy["dispatch_decision"]["exact_eval_justified"] is False
    assert policy["no_candidate_archives_emitted"] is True
    assert (output_dir / "ranked_atom_policy.json").exists()
    assert (output_dir / "ranked_atom_policy.csv").exists()
