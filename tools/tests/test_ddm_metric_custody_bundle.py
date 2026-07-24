from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.optimization.ddm_metric_custody_bundle import (
    BUNDLE_SCHEMA,
    COMPONENT_SCHEMA,
    COMPOSITE_R_DATA_SCHEMA,
    DUAL_DATA_SCHEMA,
    EVIDENCE_AXIS,
    POSE_DATA_SCHEMA,
    SEG_DATA_SCHEMA,
    ArtifactCustody,
    ComponentId,
    MetricCustodyError,
    _validate_direct_blocks,
    artifact_custody,
    load_component_receipt,
    load_metric_custody_bundle,
)
from tac.optimization.ddm_metric_producers import direct_scorer_intrinsic_pair_block
from tac.optimization.ddm_min_description_contract import (
    LayerHome,
    StreamType,
    TypedStreamTag,
    build_minimum_description_headline,
)

REPO = Path(__file__).resolve().parents[2]
PF2 = (
    REPO / ".omx/research/ddm_pf2_dimension_conditioned_two_type_20260724T020205Z/"
    "ddm_pf2_dimension_conditioned_two_type_receipt.json"
)
G3 = REPO / ".omx/research/ddm_g3_score_atlas_n600_20260722T204000Z/hard_pair_registry.json"
PARTIAL = REPO / ".omx/research/ddm_ms3_metric_custody_bundle_20260724T035249Z/BUNDLE-PARTIAL.json"
RG3_DIRECT = (
    REPO / ".omx/research/ddm_rg3_residual_family_productions_20260724T110418Z/"
    "ddm_rg3_receiver_support_summary.json"
)


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _tag() -> dict[str, object]:
    return TypedStreamTag(
        type=StreamType.FIBER,
        layer_home=LayerHome.L4_SCORER_FEATURE,
        evaluate_py_recursion_level_cited="L4_scorer_feature -> L5_verdict",
        counted_bytes=0,
        free_receiver_code=True,
    ).to_dict()


def _artifact(path: Path, *, root: Path, role: str, schema: str) -> dict[str, object]:
    return artifact_custody(
        path,
        repository_root=root,
        role=role,
        content_schema=schema,
    ).to_dict()


def _component(
    *,
    component_id: ComponentId,
    root: Path,
    atlas_ref: dict[str, object],
    g3_ref: dict[str, object],
    data_path: Path,
    data_schema: str,
) -> dict[str, object]:
    return {
        "schema": COMPONENT_SCHEMA,
        "component_id": component_id.value,
        "status": "COMPLETE",
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "research_only": True,
        "sample_count": 600,
        "scorer_batch_size": 32,
        "input_lineage": [atlas_ref, g3_ref],
        "data_artifact": _artifact(
            data_path,
            root=root,
            role=f"{component_id.value.lower()}_data",
            schema=data_schema,
        ),
        "blockers": [],
        "next_measurement": "Rehash this exact complete measurement before every use.",
        "typed_stream_tags": [_tag()],
        "main_landing_review_required": True,
    }


@pytest.fixture(scope="module")
def complete_bundle(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("metric_custody")
    atlas_value = json.loads(PF2.read_text(encoding="utf-8"))
    atlas_rows = atlas_value["typed_split_atlas"]["rows"]
    atlas_ref = _artifact(
        PF2,
        root=root,
        role="pf2_typed_atlas",
        schema=atlas_value["schema"],
    )
    g3_value = json.loads(G3.read_text(encoding="utf-8"))
    g3_ref = _artifact(
        G3,
        root=root,
        role="g3_hard_pair_registry",
        schema=g3_value["schema"],
    )
    tag = _tag()

    seg_path = root / "seg.json"
    _write(
        seg_path,
        {
            "schema": SEG_DATA_SCHEMA,
            "pf2_atlas_sha256": atlas_ref["sha256"],
            "g3_hard_pair_registry_sha256": g3_ref["sha256"],
            "measurement_schedule": [
                "top24",
                "top64",
                "stratified_control24",
                "full_n600",
            ],
            "pair_count": 600,
            "scorer_batch_size": 32,
            "head_rank": 4,
            "metric_id": "MARGIN_FISHER_RANK4",
            "rows": [
                {
                    "bucket_id": source["bucket_id"],
                    "class_pair": source["class_pair"],
                    "class_stratum": source["class_stratum"],
                    "visibility": source["visibility"],
                    "g4_temporal_class": source["g4_temporal_class"],
                    "representation_type": source["representation_type"],
                    "margin_fisher_gram": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0, 0.0],
                        [0.0, 0.0, 3.0, 0.0],
                        [0.0, 0.0, 0.0, 4.0],
                    ],
                    "eigenvalues_ascending": [1.0, 2.0, 3.0, 4.0],
                    "lambda_range": [0.0, 1.0],
                    "sample_count": 600,
                    "typed_stream_tag": tag,
                }
                for source in atlas_rows
            ],
        },
    )
    pose_path = root / "pose.json"
    _write(
        pose_path,
        {
            "schema": POSE_DATA_SCHEMA,
            "pf2_atlas_sha256": atlas_ref["sha256"],
            "g3_hard_pair_registry_sha256": g3_ref["sha256"],
            "measurement_schedule": [
                "top24",
                "top64",
                "stratified_control24",
                "full_n600",
            ],
            "pair_count": 600,
            "scorer_batch_size": 32,
            "output_dimension": 6,
            "metric_surface": "EXACT_POSENET_OUTPUT_MSE_QUADRATIC",
            "rows": [
                {
                    "pair_id": pair_id,
                    "center": [0.0] * 6,
                    "rank": 1,
                    "low_rank_factors": [[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
                    "tube_radius": 1.0,
                    "converged": pair_id != 7,
                    "convergence_status": ("NON_CONVERGED_ITERATION_LIMIT" if pair_id == 7 else "CONVERGED"),
                    "typed_stream_tag": tag,
                }
                for pair_id in range(600)
            ],
        },
    )
    composite_path = root / "composite.json"
    _write(
        composite_path,
        {
            "schema": COMPOSITE_R_DATA_SCHEMA,
            "pf2_atlas_sha256": atlas_ref["sha256"],
            "g3_hard_pair_registry_sha256": g3_ref["sha256"],
            "measurement_schedule": [
                "top24",
                "top64",
                "stratified_control24",
                "full_n600",
            ],
            "pair_count": 600,
            "scorer_batch_size": 32,
            "kernel_binding": "separable_resize_full_kernel_direct_sum_v1",
            "paired_secant_pattern": "g2f_plus_minus_equal_amplitude",
            "rows": [
                {
                    "bucket_id": source["bucket_id"],
                    "dimension": 1,
                    "model_hessian": [[1.0]],
                    "adjoint_readback": [1.0],
                    "realized_secant_positive": [0.5],
                    "realized_secant_negative": [-0.5],
                    "secant_amplitude": 0.25,
                    "typed_stream_tag": tag,
                }
                for source in atlas_rows
            ],
        },
    )
    dual_path = root / "dual.json"
    _write(
        dual_path,
        {
            "schema": DUAL_DATA_SCHEMA,
            "pf2_atlas_sha256": atlas_ref["sha256"],
            "g3_hard_pair_registry_sha256": g3_ref["sha256"],
            "measurement_schedule": [
                "top24",
                "top64",
                "stratified_control24",
                "full_n600",
            ],
            "pair_count": 600,
            "primary_metric": "MARGIN_FISHER",
            "control_metric": "EUCLIDEAN_CONTROL_ONLY",
            "rows": [
                {
                    "bucket_id": source["bucket_id"],
                    "fisher_euclidean_cosine": 0.5,
                    "fisher_to_euclidean_rel_norm": 1.0,
                    "euclidean_role": "LABELED_CONTROL_ONLY",
                    "typed_stream_tag": tag,
                }
                for source in atlas_rows
            ],
        },
    )

    component_paths: dict[str, Path] = {}
    specs = (
        (ComponentId.SEG_METRIC, seg_path, SEG_DATA_SCHEMA),
        (ComponentId.POSE_METRIC, pose_path, POSE_DATA_SCHEMA),
        (
            ComponentId.COMPOSITE_R_SECOND_ORDER,
            composite_path,
            COMPOSITE_R_DATA_SCHEMA,
        ),
        (ComponentId.DUAL_METRIC_DIAGNOSTICS, dual_path, DUAL_DATA_SCHEMA),
    )
    for component_id, data_path, data_schema in specs:
        component_path = root / f"{component_id.value}.json"
        _write(
            component_path,
            _component(
                component_id=component_id,
                root=root,
                atlas_ref=atlas_ref,
                g3_ref=g3_ref,
                data_path=data_path,
                data_schema=data_schema,
            ),
        )
        component_paths[component_id.value] = component_path

    manifest_path = root / "BUNDLE-COMPLETE.json"
    _write(
        manifest_path,
        {
            "schema": BUNDLE_SCHEMA,
            "bundle_id": "synthetic_complete_contract_fixture",
            "status": "COMPLETE",
            "evidence_axis": EVIDENCE_AXIS,
            "score_claim": False,
            "research_only": True,
            "pointer": "0.1910828242 [contest-CPU]",
            "pointer_moved": False,
            "pf2_atlas": atlas_ref,
            "g3_hard_pair_registry": g3_ref,
            "component_receipts": {
                component_id: _artifact(
                    component_path,
                    root=root,
                    role=f"{component_id.lower()}_receipt",
                    schema=COMPONENT_SCHEMA,
                )
                for component_id, component_path in component_paths.items()
            },
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
            "blockers": [],
            "headline_admissibility": {
                "bundle_complete": True,
                "scorer_metric_active": True,
                "pose_tube_active": True,
                "score_claim": False,
            },
            "main_landing_review_required": True,
        },
    )
    return root, manifest_path


def test_complete_synthetic_bundle_passes_all_four_scientific_gates(
    complete_bundle: tuple[Path, Path],
) -> None:
    root, manifest_path = complete_bundle
    bundle = load_metric_custody_bundle(
        manifest_path,
        repository_root=root,
        require_complete=True,
    )
    assert bundle.complete
    assert set(bundle.components) == set(ComponentId)


def test_repository_partial_bundle_names_exact_debts_and_refuses_complete() -> None:
    bundle = load_metric_custody_bundle(PARTIAL, repository_root=REPO)
    assert not bundle.complete
    assert bundle.atlas.sha256 == ("85084f7bd3a03dbd1b9f04fe6a9b84df4948a6caf64620beef42da8924345f73")
    assert "FULL_N600_SEG_RANK4_ROW_GRAMS_NOT_CUSTODIED" in bundle.blockers
    with pytest.raises(MetricCustodyError, match="metric custody bundle is PARTIAL"):
        load_metric_custody_bundle(
            PARTIAL,
            repository_root=REPO,
            require_complete=True,
        )


def test_partial_bundle_suppresses_metric_headline_authority() -> None:
    row = build_minimum_description_headline(
        stored_problem_bytes=1,
        stored_problem_sha256="a" * 64,
        exception_bytes=0,
        exception_sha256="b" * 64,
        realized_d_seg=0.0,
        realized_d_pose=0.0,
        stored_problem_own_lineage=True,
        donor_conditioned=False,
        expansion_receiver_closed=True,
        pose_tube_active=True,
        realized_uint8_r_frozen_scorers=True,
        quotient_coordinates_only=True,
        scorer_metric_active=True,
        alternating_typed_subproblems=True,
        typed_blocks_active=True,
        per_dimension_quanta_active=True,
        typed_stream_tags=[
            TypedStreamTag(
                type=StreamType.FIBER,
                layer_home=LayerHome.L2_CHART,
                evaluate_py_recursion_level_cited="L2_chart -> L5_verdict",
                counted_bytes=1,
                free_receiver_code=True,
            )
        ],
        metric_custody_bundle_path=PARTIAL,
        metric_custody_repository_root=REPO,
    )
    assert not row["headline_eligible"]
    assert "METRIC_CUSTODY_BUNDLE_INCOMPLETE" in row["blockers"]
    assert "SCORER_METRIC_NOT_ACTIVE" in row["blockers"]
    assert "POSE_TUBE_NOT_ACTIVE_IN_SOLVE" in row["blockers"]
    assert row["joint_constraints"]["pose_tube_active"] is False
    assert row["metric_custody_bundle"]["status"] == "PARTIAL"


def test_artifact_revalidation_refuses_schema_drift(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    _write(path, {"schema": "actual.v1"})
    custody = ArtifactCustody.from_dict(_artifact(path, root=tmp_path, role="test", schema="claimed.v1"))
    with pytest.raises(MetricCustodyError, match="content schema drift"):
        custody.revalidate(repository_root=tmp_path)


def test_seg_primary_euclidean_is_refused(
    complete_bundle: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    root, _ = complete_bundle
    atlas_value = json.loads(PF2.read_text(encoding="utf-8"))
    atlas_rows = atlas_value["typed_split_atlas"]["rows"]
    atlas_ref = _artifact(
        PF2,
        root=tmp_path,
        role="pf2_typed_atlas",
        schema=atlas_value["schema"],
    )
    g3_value = json.loads(G3.read_text(encoding="utf-8"))
    g3_ref = _artifact(
        G3,
        root=tmp_path,
        role="g3_hard_pair_registry",
        schema=g3_value["schema"],
    )
    source = json.loads((root / "seg.json").read_text(encoding="utf-8"))
    source["metric_id"] = "EUCLIDEAN_IDENTITY"
    data_path = tmp_path / "seg_euclidean.json"
    _write(data_path, source)
    receipt_path = tmp_path / "seg_receipt.json"
    _write(
        receipt_path,
        _component(
            component_id=ComponentId.SEG_METRIC,
            root=tmp_path,
            atlas_ref=atlas_ref,
            g3_ref=g3_ref,
            data_path=data_path,
            data_schema=SEG_DATA_SCHEMA,
        ),
    )
    with pytest.raises(MetricCustodyError, match="non-Euclidean Fisher"):
        load_component_receipt(
            receipt_path,
            repository_root=tmp_path,
            atlas_sha256=str(atlas_ref["sha256"]),
            hard_pair_registry_sha256=str(g3_ref["sha256"]),
            atlas_rows=atlas_rows,
        )


def test_direct_block_validator_refuses_actuation_reclassification() -> None:
    rg3 = json.loads(RG3_DIRECT.read_text(encoding="utf-8"))
    residual = rg3["receiver_coordinate_derivation"]["residual"]
    expected_counts: dict[str, list[int]] = {}
    blocks = []
    for source in residual:
        pair_id = int(source["pair_id"])
        bucket_id = str(source["bucket_id"])
        counts = expected_counts.setdefault(bucket_id, [0] * 600)
        counts[pair_id] = 1
        blocks.append(
            direct_scorer_intrinsic_pair_block(
                pair_id=pair_id,
                bucket_id=bucket_id,
                head_pair_normal=[1.0, -0.5, 0.25, 0.125],
                margins=[0.2],
                probe_custody=source["rg3_probe_blocker"],
            )
        )
    _validate_direct_blocks(
        blocks,
        residual=residual,
        expected_counts=expected_counts,
    )
    blocks[0]["actuation_status"] = "REACHABLE"
    with pytest.raises(MetricCustodyError, match="identity/custody differs"):
        _validate_direct_blocks(
            blocks,
            residual=residual,
            expected_counts=expected_counts,
        )
