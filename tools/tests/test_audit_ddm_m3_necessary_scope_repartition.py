from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "audit_ddm_m3_necessary_scope_repartition.py"
SPEC = importlib.util.spec_from_file_location("ddm_m3_scope_audit", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


PATHS = {
    "c1": ROOT / ".omx/research/ddm_c1_composed_candidate_ledger_603_613_20260723.json",
    "v14": ROOT
    / ".omx/research/ddm_v14_realization_fidelity_n600_20260722T215500Z"
    / "ddm_v14_realization_fidelity_n600_receipt.json",
    "v19b": ROOT
    / ".omx/research/ddm_v19b_joint_remeasure_stack_20260723T051914Z"
    / "ddm_v19b_joint_remeasure_stack_receipt.json",
    "g2": ROOT / ".omx/research/ddm_g2_solve_diff_op_mining_n600_20260722T194000Z" / "aggregate_ledger.json",
}


def _load_sources() -> dict[str, dict]:
    return {name: json.loads(path.read_text()) for name, path in PATHS.items()}


def test_build_audit_preserves_unknowns_and_measured_lower_bound() -> None:
    sources = _load_sources()
    receipt = MODULE.build_audit(
        **sources,
        source_paths={name: str(path.relative_to(ROOT)) for name, path in PATHS.items()},
    )

    aggregate = receipt["aggregate"]
    assert aggregate["c1_assigned_residual_errors"] == 2_377_273
    assert aggregate["v19b_residual_bucket_net_flips"] == 73_945
    assert aggregate["v19b_role_bucket_net_flips"] == 29_377
    assert aggregate["current_master_counterfactual_residual_after_v19b_subtraction"] == 2_303_328
    assert aggregate["certified_infeasible_residual_errors"] is None
    assert aggregate["certified_over_scope_percent"] is None
    assert aggregate["true_necessary_scope_interval_errors"] == [0, 2_377_273]

    rows = {(row["stratum"], row["frame"]): row for row in receipt["frontier"]}
    assert rows[("Lane", "frame_1")]["multicoefficient_inverse_solve"]["status"] == (
        "MEASURED_SUBSET_ONLY_N600_UNKNOWN"
    )
    assert rows[("Road", "frame_1")]["v19b_correction_common_master"]["net_flips"] == 82_824
    assert (
        rows[("Road", "frame_1")]["v19b_correction_common_master"]["class_conditional_delta_d_seg_after_minus_before"]
        < 0
    )
    assert rows[("Undrivable", "frame_1")]["v19b_correction_common_master"]["net_flips"] == -25_191
    assert (
        rows[("Undrivable", "frame_1")]["v19b_correction_common_master"][
            "class_conditional_delta_d_seg_after_minus_before"
        ]
        > 0
    )
    assert rows[("MyCar", "frame_1")]["v19b_correction_common_master"]["net_flips"] == 16_312
    assert rows[("Road", "frame_0")]["seg_incidence"] == "ZERO_EXACT"
    assert rows[("Road", "frame_0")]["control_sites"] is None
    assert rows[("Road", "frame_0")]["frame_separability"]["r1_comparator"] == {
        "d_pose": 0.001610,
        "complete_bytes": 7_195,
        "transferable_component": False,
    }


def test_build_audit_refuses_to_promote_g2_without_receiver_delta() -> None:
    sources = _load_sources()
    sources["g2"]["candidate_admission"]["status"] = "ADMITTED"
    with pytest.raises(MODULE.AuditRefusal, match="G2_ADMISSION_STATUS_DRIFT"):
        MODULE.build_audit(
            **sources,
            source_paths={name: str(path.relative_to(ROOT)) for name, path in PATHS.items()},
        )


def test_checked_load_refuses_sha_drift(tmp_path: Path) -> None:
    path = tmp_path / "source.json"
    path.write_text("{}\n")
    with pytest.raises(MODULE.AuditRefusal, match="SHA256_MISMATCH"):
        MODULE._load_checked(path, "0" * 64)
