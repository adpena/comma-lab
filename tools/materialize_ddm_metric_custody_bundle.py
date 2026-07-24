#!/usr/bin/env python3
"""Materialize the current DDM metric-custody bundle without inventing data.

The tool rehashes every landed source, writes one receipt per required
component, and emits either BUNDLE-COMPLETE or a precise PARTIAL manifest.
It never runs a solve, scorer, training loop, paid dispatch, or exact eval.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Final

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.optimization.ddm_lambda_continuation_frontier import (  # noqa: E402
    publish_immutable_json,
)
from tac.optimization.ddm_metric_custody_bundle import (  # noqa: E402
    BUNDLE_SCHEMA,
    COMPONENT_SCHEMA,
    EVIDENCE_AXIS,
    POINTER,
    ArtifactCustody,
    ComponentId,
    CustodyStatus,
    artifact_custody,
    load_metric_custody_bundle,
)
from tac.optimization.ddm_min_description_contract import (  # noqa: E402
    LayerHome,
    StreamType,
    TypedStreamTag,
)

DEFAULT_OUTPUT = REPO / ".omx/research/ddm_ms3_metric_custody_bundle_20260724T035249Z"
PF2 = (
    REPO / ".omx/research/ddm_pf2_dimension_conditioned_two_type_20260724T020205Z/"
    "ddm_pf2_dimension_conditioned_two_type_receipt.json"
)
G3 = REPO / ".omx/research/ddm_g3_score_atlas_n600_20260722T204000Z/hard_pair_registry.json"
AT1X = REPO / ".omx/research/ddm_at1x_atlas_materialize_20260723/atlas_receipt.json"
V16 = (
    REPO / ".omx/research/ddm_v16_coupled_joint_solve_lane_fix_20260723T013500Z/"
    "ddm_v16_coupled_joint_solve_receipt.json"
)
G2F = Path("/Volumes/VertigoDataTier/pact/evidence/g2f_chart_amplitude_20260721/receipt.json")
MIN_FREE_BYTES: Final = 64 * 1024 * 1024


def _tag(stream_type: StreamType, layer_home: LayerHome) -> dict[str, Any]:
    return TypedStreamTag(
        type=stream_type,
        layer_home=layer_home,
        evaluate_py_recursion_level_cited=(f"{layer_home.value} metric analysis -> L5_verdict"),
        counted_bytes=0,
        free_receiver_code=True,
    ).to_dict()


def _artifact(path: Path, *, role: str, content_schema: str) -> ArtifactCustody:
    return artifact_custody(
        path,
        repository_root=REPO,
        role=role,
        content_schema=content_schema,
    )


def _component(
    *,
    component_id: ComponentId,
    sample_count: int,
    scorer_batch_size: int,
    input_lineage: list[ArtifactCustody],
    blockers: list[str],
    next_measurement: str,
    tag: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": COMPONENT_SCHEMA,
        "component_id": component_id.value,
        "status": CustodyStatus.PARTIAL.value,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "research_only": True,
        "sample_count": sample_count,
        "scorer_batch_size": scorer_batch_size,
        "input_lineage": [row.to_dict() for row in input_lineage],
        "data_artifact": None,
        "blockers": blockers,
        "next_measurement": next_measurement,
        "typed_stream_tags": [tag],
        "main_landing_review_required": True,
    }


def build_partial_bundle(output: Path) -> Path:
    """Write deterministic PARTIAL receipts for the current landed custody."""

    free = shutil.disk_usage(output.parent).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"storage preflight refused: free={free} required={MIN_FREE_BYTES}")
    output.mkdir(parents=True, exist_ok=True)

    pf2 = _artifact(
        PF2,
        role="pf2_typed_atlas",
        content_schema="ddm_pf2_dimension_conditioned_two_type_measurement.v1",
    )
    g3 = _artifact(
        G3,
        role="g3_hard_pair_registry",
        content_schema="ddm_g3_hard_pair_registry.v1",
    )
    at1x = _artifact(
        AT1X,
        role="at1x_contracted_atlas_control",
        content_schema="ddm_at1x_atlas_materialize_tracked_receipt.v1",
    )
    v16 = _artifact(
        V16,
        role="v16_eight_pair_batch16_pose_control",
        content_schema="ddm_v16_coupled_joint_solve_receipt.v1",
    )
    g2f = _artifact(
        G2F,
        role="g2f_n64_realized_secant_control",
        content_schema="chart_bidirectional_amplitude_ladder_receipt.v1",
    )

    components = {
        ComponentId.SEG_METRIC: _component(
            component_id=ComponentId.SEG_METRIC,
            sample_count=600,
            scorer_batch_size=16,
            input_lineage=[pf2, g3, at1x],
            blockers=[
                "FULL_N600_SEG_RANK4_ROW_GRAMS_NOT_CUSTODIED",
                "PF2_1200_BUCKET_MARGIN_FISHER_LAMBDA_RANGES_NOT_CUSTODIED",
                "AT1X_SEG_IS_CONTRACTED_ENERGY_NOT_FULL_ROW_GRAM",
            ],
            next_measurement=(
                "Measure full rank-4 Seg row-Grams and margin-Fisher lambda "
                "ranges for all 1,200 SHA-pinned PF2 buckets, hard pairs first."
            ),
            tag=_tag(StreamType.SKELETON, LayerHome.L4_SCORER_FEATURE),
        ),
        ComponentId.POSE_METRIC: _component(
            component_id=ComponentId.POSE_METRIC,
            sample_count=8,
            scorer_batch_size=16,
            input_lineage=[at1x, v16, g3],
            blockers=[
                "N600_BATCH32_POSE_OUTPUT_QUADRATICS_NOT_CUSTODIED",
                "POSE_TUBE_CONVERGENCE_FLAGS_NOT_CUSTODIED_FOR_600_PAIRS",
                "V16_CONTROL_IS_EIGHT_PAIR_BATCH16_AND_NOT_ALL_CONVERGED",
            ],
            next_measurement=(
                "Measure the exact Pose6 output quadratic and contest-budget "
                "active-tube radius for pair IDs 0..599 at canonical batch32; "
                "preserve every non-convergence flag."
            ),
            tag=_tag(StreamType.FIBER, LayerHome.L5_VERDICT),
        ),
        ComponentId.COMPOSITE_R_SECOND_ORDER: _component(
            component_id=ComponentId.COMPOSITE_R_SECOND_ORDER,
            sample_count=64,
            scorer_batch_size=16,
            input_lineage=[pf2, g2f, v16],
            blockers=[
                "N600_COMPOSITE_R_HESSIAN_ADJOINT_NOT_CUSTODIED",
                "PF2_BUCKET_COMPLETE_REALIZED_PAIRED_SECANTS_NOT_CUSTODIED",
                "G2F_REALIZED_SECANT_CONTROL_IS_N64_NOT_N600",
            ],
            next_measurement=(
                "Measure exact composite-R adjoint/Hessian and equal-amplitude "
                "positive/negative receiver-realized secants for each PF2 "
                "bucket at n600 batch32, reporting model and realized together."
            ),
            tag=_tag(StreamType.CONNECTION, LayerHome.L4_SCORER_FEATURE),
        ),
        ComponentId.DUAL_METRIC_DIAGNOSTICS: _component(
            component_id=ComponentId.DUAL_METRIC_DIAGNOSTICS,
            sample_count=0,
            scorer_batch_size=32,
            input_lineage=[pf2, g3, at1x],
            blockers=[
                "PF2_BUCKET_COMPLETE_FISHER_EUCLIDEAN_COSINE_NOT_CUSTODIED",
                "PF2_BUCKET_COMPLETE_FISHER_EUCLIDEAN_REL_NORM_NOT_CUSTODIED",
                "EUCLIDEAN_CONTROL_EXISTS_WITHOUT_MATCHED_FISHER_READBACK",
            ],
            next_measurement=(
                "Read back matched Fisher and Euclidean control vectors for "
                "all 1,200 buckets and emit signed cosine plus relative norm; "
                "retain Euclidean only as the labeled control column."
            ),
            tag=_tag(StreamType.RESIDUAL, LayerHome.L5_VERDICT),
        ),
    }

    component_refs: dict[str, dict[str, Any]] = {}
    all_blockers: list[str] = []
    for component_id, receipt in components.items():
        path = output / f"{component_id.value.lower()}_receipt.json"
        publish_immutable_json(path, receipt)
        component_refs[component_id.value] = _artifact(
            path,
            role=f"{component_id.value.lower()}_component_receipt",
            content_schema=COMPONENT_SCHEMA,
        ).to_dict()
        all_blockers.extend(receipt["blockers"])

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "bundle_id": "ddm_ms3_metric_custody_bundle_20260724T035249Z",
        "status": CustodyStatus.PARTIAL.value,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "research_only": True,
        "pointer": POINTER,
        "pointer_moved": False,
        "pf2_atlas": pf2.to_dict(),
        "g3_hard_pair_registry": g3.to_dict(),
        "component_receipts": component_refs,
        "hard_pair_order": [
            "top24",
            "top64",
            "stratified_control24",
            "full_n600",
        ],
        "consumers": [
            "ms2_typed_quotient_solve",
            "pf2r_metric_active_three_formulation",
            "rd1_dimension_duals",
        ],
        "blockers": list(dict.fromkeys(all_blockers)),
        "headline_admissibility": {
            "bundle_complete": False,
            "scorer_metric_active": False,
            "pose_tube_active": False,
            "score_claim": False,
        },
        "main_landing_review_required": True,
    }
    manifest_path = output / "BUNDLE-PARTIAL.json"
    publish_immutable_json(manifest_path, manifest)
    loaded = load_metric_custody_bundle(
        manifest_path,
        repository_root=REPO,
    )
    if loaded.complete:
        raise RuntimeError("PARTIAL materializer unexpectedly produced COMPLETE")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    path = build_partial_bundle(args.output_dir)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
