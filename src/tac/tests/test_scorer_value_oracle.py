"""Tests for the DDM-366 scorer-value oracle facade."""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.ddm_metric_custody_bundle import ComponentId
from tac.scorer_value_oracle import (
    COVERAGE_SCHEMA,
    DEFAULT_BINDINGS,
    DEFAULT_GAPS,
    CoverageStatus,
    DimensionRow,
    FreshnessMode,
    FreshnessStatus,
    OracleError,
    PayloadKind,
    RowBinding,
    ScorerValueOracle,
    StaleProducerError,
    TypedGap,
    TypedGapError,
)

REPO = Path(__file__).resolve().parents[3]
TEST_SCHEMA = "test.scorer.value.v1"
POSE_METRIC_DATA = Path(
    "/Volumes/VertigoDataTier/pact/ddm_ms4_metric_producers_and_measurement_"
    "20260724T042005Z/pose_metric_n600_batch32.json"
)
GC2_ROWS = (
    (
        DimensionRow.GAIN_NORMALIZATION_AFFINE,
        "normalization_affine",
        "ddm_scorer_normalization_affine.v1",
        ("input_affine", "mean_each"),
        127.5,
    ),
    (
        DimensionRow.FREQUENCY_R_PASSBAND,
        "r_frequency_passband",
        "ddm_composite_r_frequency_passband.v1",
        ("tangent_normal_anisotropy", "axis_ratio_vertical_over_horizontal_at_nyquist"),
        1.0005539392634217,
    ),
    (
        DimensionRow.YUV6_LUMA_PHASES,
        "yuv6_luma_phases",
        "ddm_yuv6_luma_phase_law.v1",
        ("input", "concatenated_pose_input_channels"),
        12,
    ),
    (
        DimensionRow.CHROMA_POSE_NULL,
        "chroma_pose_null",
        "ddm_chroma_pose_null_intersection.v1",
        ("exact_kernel", "real_dimension"),
        6,
    ),
    (
        DimensionRow.NULL_GAUGE_ENERGY,
        "null_gauge_energy",
        "ddm_null_gauge_composite_custody.v1",
        ("intersection_accounting", "measured_joint_intersection_energy"),
        None,
    ),
    (
        DimensionRow.POSE_DIMS_7_12,
        "pose_excluded_dimensions",
        "ddm_pose_objective_dimension_exclusion.v1",
        ("pose_head", "excluded_one_based_dimensions"),
        [7, 8, 9, 10, 11, 12],
    ),
    (
        DimensionRow.SCORE_AXES_WEIGHTS,
        "score_functional",
        "ddm_contest_score_functional.v1",
        ("axes", "rate", "canonical_video_set_denominator_bytes"),
        37_545_489,
    ),
)


def _write_json(path: Path, value: object) -> tuple[str, int]:
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return sha256(raw).hexdigest(), len(raw)


def _binding(
    path: Path,
    *,
    row: DimensionRow = DimensionRow.RATE_ARCHIVE_BYTES_ONLY,
    selector: tuple[str | int, ...] = (),
    sha_value: str | None = None,
    byte_count: int | None = None,
    schema: str = TEST_SCHEMA,
) -> RowBinding:
    observed = path.read_bytes() if path.is_file() else b""
    return RowBinding(
        row=row,
        producer="fixture producer",
        path=str(path),
        sha256=sha_value or sha256(observed).hexdigest(),
        bytes=byte_count if byte_count is not None else len(observed),
        schema=schema,
        validity_horizon="fixture hash valid until fixture mutation",
        value_kind="fixture_value",
        authority_scope="fixture-only authority",
        selector=selector,
        payload_kind=PayloadKind.JSON,
    )


def _gaps_except(*wrapped: DimensionRow) -> tuple[TypedGap, ...]:
    keep = set(wrapped)
    return tuple(
        TypedGap(row, "fixture producer", "fixture typed gap", "land fixture producer")
        for row in DimensionRow
        if row not in keep
    )


def _one_row_oracle(
    tmp_path: Path,
    *,
    payload: object | None = None,
    row: DimensionRow = DimensionRow.RATE_ARCHIVE_BYTES_ONLY,
    selector: tuple[str | int, ...] = (),
) -> tuple[ScorerValueOracle, Path]:
    path = tmp_path / "producer.json"
    _write_json(path, payload or {"schema": TEST_SCHEMA, "value": 7})
    binding = _binding(path, row=row, selector=selector)
    return (
        ScorerValueOracle(tmp_path, bindings=(binding,), gaps=_gaps_except(row)),
        path,
    )


def test_dimension_enum_is_exact_21_row_contract() -> None:
    assert len(DimensionRow) == 21
    assert DimensionRow.SUB_PIXEL_PLACEMENT.value == "sub-pixel placement (874-res, pre-R)"
    assert DimensionRow.SCORE_AXES_WEIGHTS.value == "score axes + weights"


def test_default_registry_covers_every_contract_row_once() -> None:
    rows = [binding.row for binding in DEFAULT_BINDINGS] + [gap.row for gap in DEFAULT_GAPS]
    assert len(rows) == len(set(rows)) == len(DimensionRow)


def test_ms4d_binding_requires_explicit_component() -> None:
    with pytest.raises(ValueError, match="require exactly one component"):
        RowBinding(
            row=DimensionRow.MARGIN_FISHER_SURROGATE,
            producer="fixture producer",
            path="fixture.json",
            sha256="0" * 64,
            bytes=1,
            schema="fixture.schema",
            validity_horizon="fixture",
            value_kind="fixture",
            authority_scope="fixture",
            payload_kind=PayloadKind.MS4D_BUNDLE,
        )


def test_json_binding_rejects_ms4d_component_selector() -> None:
    with pytest.raises(ValueError, match="require exactly one component"):
        RowBinding(
            row=DimensionRow.MARGIN_FISHER_SURROGATE,
            producer="fixture producer",
            path="fixture.json",
            sha256="0" * 64,
            bytes=1,
            schema="fixture.schema",
            validity_horizon="fixture",
            value_kind="fixture",
            authority_scope="fixture",
            ms4d_component=ComponentId.SEG_METRIC,
        )


def test_read_fresh_json_with_selector(tmp_path: Path) -> None:
    oracle, _ = _one_row_oracle(
        tmp_path,
        payload={"schema": TEST_SCHEMA, "nested": {"answer": 42}},
        selector=("nested", "answer"),
    )
    result = oracle.read(DimensionRow.RATE_ARCHIVE_BYTES_ONLY)
    assert result.value == 42
    assert result.coverage is CoverageStatus.WRAPPED
    assert result.freshness is FreshnessStatus.FRESH
    assert result.freshness_tag == "[fresh]"
    assert result.lineage[0].fresh is True
    assert result.authority_scope == "fixture-only authority"


def test_read_accepts_exact_contract_key(tmp_path: Path) -> None:
    oracle, _ = _one_row_oracle(tmp_path)
    result = oracle.read("rate (archive bytes only)")
    assert result.row is DimensionRow.RATE_ARCHIVE_BYTES_ONLY


def test_read_accepts_enum_member_name(tmp_path: Path) -> None:
    oracle, _ = _one_row_oracle(tmp_path)
    result = oracle.read("RATE_ARCHIVE_BYTES_ONLY")
    assert result.row is DimensionRow.RATE_ARCHIVE_BYTES_ONLY


def test_unknown_contract_key_is_rejected(tmp_path: Path) -> None:
    oracle, _ = _one_row_oracle(tmp_path)
    with pytest.raises(OracleError, match="unknown DDM-366"):
        oracle.read("not a contract row")


def test_unknown_freshness_mode_is_rejected(tmp_path: Path) -> None:
    oracle, _ = _one_row_oracle(tmp_path)
    with pytest.raises(OracleError, match="unknown freshness mode"):
        oracle.read(DimensionRow.RATE_ARCHIVE_BYTES_ONLY, freshness_mode="invented")


def test_hash_drift_fails_closed(tmp_path: Path) -> None:
    oracle, path = _one_row_oracle(tmp_path)
    path.write_text('{"schema":"test.scorer.value.v1","value":8}\n')
    with pytest.raises(StaleProducerError, match="stale producer artifact"):
        oracle.read(DimensionRow.RATE_ARCHIVE_BYTES_ONLY)


def test_byte_count_drift_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "producer.json"
    digest, size = _write_json(path, {"schema": TEST_SCHEMA, "value": 7})
    binding = _binding(path, sha_value=digest, byte_count=size + 1)
    oracle = ScorerValueOracle(
        tmp_path,
        bindings=(binding,),
        gaps=_gaps_except(DimensionRow.RATE_ARCHIVE_BYTES_ONLY),
    )
    with pytest.raises(StaleProducerError, match="stale producer artifact"):
        oracle.read(DimensionRow.RATE_ARCHIVE_BYTES_ONLY)


def test_missing_artifact_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"
    binding = _binding(path, sha_value="0" * 64, byte_count=1)
    oracle = ScorerValueOracle(
        tmp_path,
        bindings=(binding,),
        gaps=_gaps_except(DimensionRow.RATE_ARCHIVE_BYTES_ONLY),
    )
    with pytest.raises(StaleProducerError, match="stale producer artifact"):
        oracle.read(DimensionRow.RATE_ARCHIVE_BYTES_ONLY)


def test_stale_advisory_returns_typed_tag(tmp_path: Path) -> None:
    oracle, path = _one_row_oracle(tmp_path)
    path.write_text('{"schema":"test.scorer.value.v1","value":8}\n')
    result = oracle.read(
        DimensionRow.RATE_ARCHIVE_BYTES_ONLY,
        freshness_mode=FreshnessMode.STALE_ADVISORY,
    )
    assert result.coverage is CoverageStatus.WRAPPED
    assert result.freshness is FreshnessStatus.STALE_ADVISORY
    assert result.freshness_tag == "[stale-advisory]"
    assert result.value is None


def test_typed_gap_is_data_not_a_guessed_value(tmp_path: Path) -> None:
    oracle, _ = _one_row_oracle(tmp_path)
    result = oracle.read(DimensionRow.SCORE_AXES_WEIGHTS)
    assert result.coverage is CoverageStatus.TYPED_GAP
    assert result.freshness is FreshnessStatus.TYPED_GAP
    assert result.freshness_tag == "[typed-gap]"
    assert result.value is None
    assert result.next_action


def test_require_raises_on_typed_gap(tmp_path: Path) -> None:
    oracle, _ = _one_row_oracle(tmp_path)
    with pytest.raises(TypedGapError, match="fixture typed gap"):
        oracle.require(DimensionRow.SCORE_AXES_WEIGHTS)


def test_schema_drift_is_rejected_after_hash_passes(tmp_path: Path) -> None:
    path = tmp_path / "producer.json"
    _write_json(path, {"schema": "wrong.schema", "value": 7})
    binding = _binding(path, schema=TEST_SCHEMA)
    oracle = ScorerValueOracle(
        tmp_path,
        bindings=(binding,),
        gaps=_gaps_except(DimensionRow.RATE_ARCHIVE_BYTES_ONLY),
    )
    with pytest.raises(StaleProducerError, match="schema drift"):
        oracle.read(DimensionRow.RATE_ARCHIVE_BYTES_ONLY)


def test_schema_drift_can_be_typed_stale_advisory(tmp_path: Path) -> None:
    path = tmp_path / "producer.json"
    _write_json(path, {"schema": "wrong.schema", "value": 7})
    binding = _binding(path, schema=TEST_SCHEMA)
    oracle = ScorerValueOracle(
        tmp_path,
        bindings=(binding,),
        gaps=_gaps_except(DimensionRow.RATE_ARCHIVE_BYTES_ONLY),
    )
    result = oracle.read(
        DimensionRow.RATE_ARCHIVE_BYTES_ONLY,
        freshness_mode=FreshnessMode.STALE_ADVISORY,
    )
    assert result.freshness is FreshnessStatus.STALE_ADVISORY
    assert result.freshness_tag == "[stale-advisory]"
    assert result.value is None


def test_selector_drift_is_rejected(tmp_path: Path) -> None:
    oracle, _ = _one_row_oracle(tmp_path, selector=("missing",))
    with pytest.raises(OracleError, match="selector failed"):
        oracle.read(DimensionRow.RATE_ARCHIVE_BYTES_ONLY)


def test_coverage_report_counts_wrapped_and_gaps(tmp_path: Path) -> None:
    oracle, _ = _one_row_oracle(tmp_path)
    report = oracle.coverage_report()
    assert report["schema"] == COVERAGE_SCHEMA
    assert report["row_count"] == 21
    assert report["counts"] == {"WRAPPED": 1, "TYPED-GAP": 20}
    assert report["stale_advisory_count"] == 0


def test_coverage_report_surfaces_stale_advisory(tmp_path: Path) -> None:
    oracle, path = _one_row_oracle(tmp_path)
    path.write_text('{"schema":"test.scorer.value.v1","value":8}\n')
    report = oracle.coverage_report()
    wrapped = next(row for row in report["rows"] if row["coverage"] == "WRAPPED")
    assert wrapped["freshness"] == "STALE_ADVISORY"
    assert report["stale_advisory_count"] == 1


def test_unverified_coverage_never_claims_freshness(tmp_path: Path) -> None:
    oracle, _ = _one_row_oracle(tmp_path)
    report = oracle.coverage_report(verify=False)
    wrapped = next(row for row in report["rows"] if row["coverage"] == "WRAPPED")
    assert wrapped["freshness"] == "NOT_CHECKED"
    assert report["verified_at_consumption"] is False


def test_duplicate_binding_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "producer.json"
    _write_json(path, {"schema": TEST_SCHEMA})
    binding = _binding(path)
    with pytest.raises(OracleError, match="duplicate row"):
        ScorerValueOracle(
            tmp_path,
            bindings=(binding, binding),
            gaps=_gaps_except(DimensionRow.RATE_ARCHIVE_BYTES_ONLY),
        )


def test_binding_gap_conflict_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "producer.json"
    _write_json(path, {"schema": TEST_SCHEMA})
    binding = _binding(path)
    gaps = (
        *_gaps_except(DimensionRow.RATE_ARCHIVE_BYTES_ONLY),
        TypedGap(
            DimensionRow.RATE_ARCHIVE_BYTES_ONLY,
            "fixture",
            "conflict",
            "remove conflict",
        ),
    )
    with pytest.raises(OracleError, match="both wrapped and typed gaps"):
        ScorerValueOracle(tmp_path, bindings=(binding,), gaps=gaps)


def test_incomplete_registry_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "producer.json"
    _write_json(path, {"schema": TEST_SCHEMA})
    with pytest.raises(OracleError, match="omits DDM-366 rows"):
        ScorerValueOracle(tmp_path, bindings=(_binding(path),), gaps=())


def test_external_npz_member_is_rehashed_then_opened_read_only(tmp_path: Path) -> None:
    cache = tmp_path / "targets.npz"
    values = np.arange(12, dtype=np.uint8).reshape(3, 2, 2)
    np.savez(cache, lstars=values)
    descriptor = {
        "schema": TEST_SCHEMA,
        "target": {
            "cache_path": str(cache),
            "cache_sha256": sha256(cache.read_bytes()).hexdigest(),
            "cache_bytes": cache.stat().st_size,
        },
    }
    oracle, _ = _one_row_oracle(
        tmp_path,
        payload=descriptor,
        row=DimensionRow.CELL_LAGUERRE_ARGMAX_POLYTOPE,
        selector=("target",),
    )
    mapped = oracle.open_npz_member(DimensionRow.CELL_LAGUERRE_ARGMAX_POLYTOPE, "lstars")
    assert isinstance(mapped, np.memmap)
    assert mapped.flags.writeable is False
    assert np.array_equal(mapped, values)


def test_external_npz_hash_drift_is_rejected(tmp_path: Path) -> None:
    cache = tmp_path / "targets.npz"
    np.savez(cache, lstars=np.arange(4, dtype=np.uint8))
    descriptor = {
        "schema": TEST_SCHEMA,
        "target": {
            "cache_path": str(cache),
            "cache_sha256": "0" * 64,
            "cache_bytes": cache.stat().st_size,
        },
    }
    oracle, _ = _one_row_oracle(
        tmp_path,
        payload=descriptor,
        row=DimensionRow.CELL_LAGUERRE_ARGMAX_POLYTOPE,
        selector=("target",),
    )
    with pytest.raises(StaleProducerError, match="external NPZ lineage drift"):
        oracle.open_npz_member(DimensionRow.CELL_LAGUERRE_ARGMAX_POLYTOPE, "lstars")


def test_admission_accessors_are_thin_fresh_reads(tmp_path: Path) -> None:
    oracle, _ = _one_row_oracle(tmp_path)
    assert oracle.bucket_assignments().row is DimensionRow.RATE_ARCHIVE_BYTES_ONLY
    assert oracle.bucket_assignments().lineage[0].fresh is True


def test_default_coverage_shape_is_21_wrapped_zero_typed_gaps() -> None:
    report = ScorerValueOracle(REPO).coverage_report(verify=False)
    assert report["counts"] == {"WRAPPED": 21, "TYPED-GAP": 0}
    assert report["row_count"] == 21
    assert DEFAULT_GAPS == ()


@pytest.mark.parametrize(
    ("row", "accessor", "schema", "selector", "expected"),
    GC2_ROWS,
)
def test_gc2_gap_closure_row_is_fresh_and_content_typed(
    row: DimensionRow,
    accessor: str,
    schema: str,
    selector: tuple[str, ...],
    expected: object,
) -> None:
    result = getattr(ScorerValueOracle(REPO), accessor)()
    value = result.require_value()
    assert result.row is row
    assert result.lineage[0].fresh is True
    assert value["schema"] == schema
    selected = value
    for key in selector:
        selected = selected[key]
    assert selected == expected


@pytest.mark.parametrize(("row", "_accessor", "_schema", "_selector", "_expected"), GC2_ROWS)
def test_gc2_gap_closure_row_hash_drift_fails_closed(
    tmp_path: Path,
    row: DimensionRow,
    _accessor: str,
    _schema: str,
    _selector: tuple[str, ...],
    _expected: object,
) -> None:
    binding = next(item for item in DEFAULT_BINDINGS if item.row is row)
    source = REPO / binding.path
    stale = tmp_path / source.name
    stale.write_bytes(source.read_bytes() + b" ")
    oracle = ScorerValueOracle(
        tmp_path,
        bindings=(replace(binding, path=str(stale)),),
        gaps=_gaps_except(row),
    )
    with pytest.raises(StaleProducerError, match="stale producer artifact"):
        oracle.read(row)


def test_gc2_chroma_null_keeps_camera_preimage_authority_fail_closed() -> None:
    value = ScorerValueOracle(REPO).chroma_pose_null().require_value()
    readback = value["uint8_readback"]
    assert readback["bounded_uint8_feasible_fraction"] == pytest.approx(
        0.7505900065104166
    )
    assert readback["bit_exact_yuv6_fraction_among_feasible"] == pytest.approx(
        0.2672232311459265
    )
    assert readback["camera_preimage_certified_fraction"] is None
    assert readback["receiver_closed_fraction"] is None


def test_gc2_null_gauge_does_not_conflate_two_52_percent_measurements() -> None:
    value = ScorerValueOracle(REPO).null_gauge_energy().require_value()
    assert value["head_gauge"]["gauge_fraction_of_norm"] == pytest.approx(
        0.5235602011487402
    )
    assert value["head_gauge"]["gauge_fraction_of_energy"] == pytest.approx(
        0.27411528422690934
    )
    assert value["render_resize_null"]["blind_output_raw_energy_in_ker_a_mean"] == (
        pytest.approx(0.5242467615738868)
    )


def test_default_pf2_bucket_assignment_is_fresh() -> None:
    result = ScorerValueOracle(REPO).bucket_assignments()
    value = result.require_value()
    assert value["schema"] == "ddm_ms5_pf2_bucket_assignment_table.v1"
    assert value["bucket_count"] == 1200
    assert result.lineage[0].fresh is True


def test_default_margin_fisher_returns_validated_component_payload() -> None:
    if not POSE_METRIC_DATA.is_file():
        pytest.skip("sealed MS4D bundle's external pose component is unavailable")
    result = ScorerValueOracle(REPO).margin_fisher()
    value = result.require_value()
    assert value["component_id"] == ComponentId.SEG_METRIC.value
    assert value["component_status"] == "COMPLETE"
    assert value["data"]["schema"] == (
        "ddm_seg_metric_custody.direct_scorer_intrinsic.v2"
    )
    assert len(value["data"]["rows"]) == 1200
    assert len(result.lineage) == 2
    assert all(item.fresh for item in result.lineage)
