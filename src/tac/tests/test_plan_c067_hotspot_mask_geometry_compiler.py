from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
PLANNER_PATH = REPO / "experiments" / "plan_c067_hotspot_mask_geometry_compiler.py"


def _load_planner():
    spec = importlib.util.spec_from_file_location("c067_hotspot_geometry_compiler_test", PLANNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _atom(
    atom_id: str,
    *,
    frame: int,
    y: int,
    x0: int,
    x1: int,
    cls: int,
    benefit: float,
    component_weight: float = 1.0,
    hard_weight: float = 1.0,
    foveal_weight: float = 1.0,
    boundary_fraction: float = 0.0,
    source_cls: int | None = None,
    candidate_cls: int | None = None,
) -> dict:
    charged = 6
    pixels = x1 - x0
    src = cls if source_cls is None else source_cls
    cand = cls if candidate_cls is None else candidate_cls
    return {
        "atom_id": atom_id,
        "atom_family": "row_run",
        "identity": {
            "frame_index": frame,
            "pair_index": frame // 2,
            "class_id": cls,
            "y": y,
            "x0": x0,
            "x1_exclusive": x1,
        },
        "pair_indices": [frame // 2],
        "frame_indices": [frame],
        "class_ids": [cls],
        "residual_pixels": pixels,
        "source_class_histogram_pixels": {str(src): pixels},
        "candidate_class_histogram_pixels": {str(cand): pixels},
        "boundary_pixel_fraction": boundary_fraction,
        "cost_model": {"estimated_charged_bytes": charged},
        "weights": {
            "component_pair_weight_pixel_mean": component_weight,
            "hard_pair_weight_pixel_mean": hard_weight,
            "foveal_weight_pixel_mean": foveal_weight,
            "weighted_residual_pixel_proxy": pixels * component_weight * foveal_weight,
        },
        "lagrangian": {
            "estimated_marginal_score_saved_proxy": benefit,
            "estimated_rate_score_cost": 0.000003995154,
            "estimated_lagrangian_net_proxy": benefit - 0.000003995154,
            "estimated_score_saved_per_charged_byte": benefit / charged,
        },
        "score_claim": False,
        "evidence_grade": "planning_only",
    }


def _write_ledger(path: Path, atoms: list[dict]) -> None:
    payload = {
        "schema": "cmg3_pixel_lagrangian_atom_ledger_v1",
        "score_claim": False,
        "evidence_grade": "planning_only",
        "atom_count": len(atoms),
        "inputs": {
            "candidate": {
                "mode": "reconstructed_from_cmg3a_adaptive_manifest",
                "base_runs_per_row": 1,
            }
        },
        "top_atoms": atoms,
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_hotspot_confusion_dense_atom_beats_arbitrary_span(tmp_path: Path) -> None:
    planner = _load_planner()
    ledger = tmp_path / "atoms.json"
    _write_ledger(
        ledger,
        [
            _atom(
                "arbitrary_span",
                frame=8,
                y=120,
                x0=0,
                x1=220,
                cls=2,
                benefit=0.0000060,
                component_weight=1.6,
            ),
            _atom(
                "hotspot_confusion",
                frame=10,
                y=174,
                x0=240,
                x1=272,
                cls=3,
                benefit=0.0000055,
                component_weight=2.8,
                hard_weight=2.0,
                foveal_weight=2.1,
                boundary_fraction=0.18,
                source_cls=3,
                candidate_cls=1,
            ),
        ],
    )

    plan = planner.build_plan(
        ledger_jsons=[ledger],
        output_json=tmp_path / "plan.json",
        candidate_sizes=(1,),
        max_arbitrary_span_pixels=512,
        min_policy_hotspot_fraction=0.0,
    )

    assert plan["score_claim"] is False
    assert plan["candidate_policies"][0]["selected_atom_ids"] == ["hotspot_confusion"]
    first = plan["selected_atoms"][0]
    assert first["atom_id"] == "hotspot_confusion"
    assert first["signals"]["confusion_fraction"] == 1.0
    assert first["field_coordinates"]["scorer_space"] == "mask_512x384"
    assert first["break_even_bytes"] > 0


def test_rejects_known_negative_global_rowspan_atom_shape(tmp_path: Path) -> None:
    planner = _load_planner()
    ledger = tmp_path / "atoms.json"
    _write_ledger(
        ledger,
        [
            _atom(
                "wide_low_signal_rowspan",
                frame=0,
                y=10,
                x0=0,
                x1=511,
                cls=1,
                benefit=0.000010,
                component_weight=1.05,
                source_cls=1,
                candidate_cls=1,
            )
        ],
    )

    plan = planner.build_plan(
        ledger_jsons=[ledger],
        output_json=tmp_path / "plan.json",
        candidate_sizes=(1,),
        max_arbitrary_span_pixels=128,
    )

    assert plan["candidate_policies"] == []
    assert plan["rejected_atoms"][0]["atom_id"] == "wide_low_signal_rowspan"
    assert "known_negative_arbitrary_row_span_shape" in plan["rejected_atoms"][0]["reject_reasons"]


def test_rejects_class_zero_for_cmg3a_nonzero_row_run_policy(tmp_path: Path) -> None:
    planner = _load_planner()
    ledger = tmp_path / "atoms.json"
    _write_ledger(
        ledger,
        [
            _atom(
                "invalid_class_zero_hotspot",
                frame=67 * 2,
                y=174,
                x0=240,
                x1=276,
                cls=0,
                benefit=0.000050,
                component_weight=4.0,
                hard_weight=3.0,
                foveal_weight=2.5,
                boundary_fraction=0.3,
                source_cls=0,
                candidate_cls=3,
            ),
            _atom(
                "valid_class_two_hotspot",
                frame=69 * 2,
                y=174,
                x0=248,
                x1=276,
                cls=2,
                benefit=0.000010,
                component_weight=3.0,
                hard_weight=2.5,
                foveal_weight=2.0,
                boundary_fraction=0.2,
                source_cls=2,
                candidate_cls=3,
            ),
        ],
    )

    plan = planner.build_plan(
        ledger_jsons=[ledger],
        output_json=tmp_path / "plan.json",
        candidate_sizes=(1,),
        min_policy_hotspot_fraction=0.0,
    )

    assert plan["candidate_policies"][0]["selected_atom_ids"] == ["valid_class_two_hotspot"]
    assert plan["candidate_policies"][0]["selected_row_run_atoms"][0]["class_id"] == 2
    rejected = {item["atom_id"]: item for item in plan["rejected_atoms"]}
    assert "builder_incompatible_class_id_for_cmg3a_nonzero_row_run" in rejected[
        "invalid_class_zero_hotspot"
    ]["reject_reasons"]
    contract = plan["configuration"]["builder_class_contract"]
    assert contract["valid_row_run_class_id_min_inclusive"] == 1
    assert contract["valid_row_run_class_id_max_exclusive"] == 5


def test_filters_full_residual_like_policy_spread(tmp_path: Path) -> None:
    planner = _load_planner()
    ledger = tmp_path / "spread_atoms.json"
    atoms = [
        _atom(
            f"spread_{idx:02d}",
            frame=idx * 2,
            y=idx,
            x0=20,
            x1=60,
            cls=2,
            benefit=0.000008,
            component_weight=1.7,
            source_cls=2,
            candidate_cls=1,
        )
        for idx in range(20)
    ]
    _write_ledger(ledger, atoms)
    negative_eval = tmp_path / "rowspan_exact_negative.json"
    negative_eval.write_text(
        json.dumps(
            {
                "score_recomputed_from_components": 24.45,
                "archive_size_bytes": 251500,
                "avg_posenet_dist": 51.9,
                "avg_segnet_dist": 0.015,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    plan = planner.build_plan(
        ledger_jsons=[ledger],
        output_json=tmp_path / "plan.json",
        exact_negative_jsons=[negative_eval],
        candidate_sizes=(20,),
        min_policy_hotspot_fraction=0.90,
        min_atom_hotspot_signal=1.20,
    )

    assert plan["candidate_policies"] == []
    filtered = plan["filtered_candidate_policies"]["known_negative_shape"][0]
    assert "known_negative_full_residual_like_pair_spread" in filtered["filter_reasons"]
    assert "known_negative_low_hotspot_density_policy" in filtered["filter_reasons"]
    assert plan["exact_negative_trace_inputs"][0]["failure_modes"] == [
        "posenet_component_collapse",
        "segnet_component_collapse",
        "large_exact_negative_score",
    ]


def test_emits_builder_compatible_policy_and_optional_command(tmp_path: Path) -> None:
    planner = _load_planner()
    ledger = tmp_path / "atoms.json"
    _write_ledger(
        ledger,
        [
            _atom(
                "hotspot",
                frame=2,
                y=20,
                x0=100,
                x1=130,
                cls=4,
                benefit=0.000010,
                component_weight=3.0,
                hard_weight=2.0,
                foveal_weight=2.0,
                boundary_fraction=0.2,
                source_cls=4,
                candidate_cls=0,
            )
        ],
    )
    output_json = tmp_path / "plan.json"

    plan = planner.build_plan(
        ledger_jsons=[ledger],
        output_json=output_json,
        candidate_sizes=(1,),
        frontier_archive=Path("frontier.zip"),
        decoded_mask_array=Path("masks.npy"),
        builder_output_dir=Path("candidate_dir"),
        target_body_bytes=166000,
    )

    assert plan["schema"] == "yousfi_fridrich_atom_field_allocator_v1"
    assert plan["compiler_schema"] == "c067_hotspot_mask_geometry_compiler_v1"
    assert plan["external_design_motivation"]["arxiv"] == "2604.06332"
    assert plan["planning_field_atoms"][1]["atom_family"] == "learnable_anisotropic_hyperbolic_foveation"
    assert plan["planning_field_atoms"][1]["score_claim"] is False
    assert plan["candidate_policies"][0]["required_base_runs_per_row"] == 1
    assert "radial_from_foe" in plan["candidate_policies"][0]["support"]["low_dimensional_field_basis"]
    assert plan["candidate_policies"][0]["selected_row_run_atoms"] == [
        {"frame_index": 2, "y": 20, "x0": 100, "x1_exclusive": 130, "class_id": 4}
    ]
    command = plan["concrete_builder_command_if_safe"]
    assert command[:2] == ["python", "experiments/build_cmg3_adaptive_runs_candidate.py"]
    assert "--field-policy-json" in command
    assert str(output_json) in command
    assert "--target-body-bytes" in command
