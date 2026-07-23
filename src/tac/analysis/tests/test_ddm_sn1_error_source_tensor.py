from __future__ import annotations

import numpy as np

from tac.analysis.ddm_sn1_error_source_tensor import (
    CLASS_NAMES,
    HEIGHT,
    N600_SITES,
    WIDTH,
    ErrorSource,
    boundary_distance_bands,
    classify_error_sources,
    curvature_bands,
    d2_margin_bands,
    decode_group_key,
    encode_group_key,
    paint_floor_mechanism_codes,
    source_budget,
    summarize_components,
    survival_wall_149,
    temporal_pattern_codes,
)


def test_error_source_partition_is_exhaustive_and_ordered() -> None:
    target = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    predicted = target.copy()
    predicted[0, :3] = 1
    current = predicted.copy()
    current[0, 0] = 0
    enriched = predicted.copy()
    enriched[0, 1] = 0
    source, residual = classify_error_sources(
        target=target,
        predicted=predicted,
        current_semantic=current,
        enriched_semantic=enriched,
    )
    assert residual.sum() == 3
    assert source[0, :3].tolist() == [
        ErrorSource.DESCRIBED_BUT_REALIZATION_LOST,
        ErrorSource.NEVER_DESCRIBED,
        ErrorSource.STRUCTURALLY_HARD_IRREDUCIBLE,
    ]


def test_d2_and_joint_group_key_roundtrip() -> None:
    target = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    predicted = target.copy()
    predicted[0, 0] = 1
    logits = np.zeros((5, HEIGHT, WIDTH), dtype=np.float32)
    logits[1, 0, 0] = 0.5
    norms = {f"{left}->{right}": 2.0 for left in CLASS_NAMES for right in CLASS_NAMES if left != right}
    thresholds = dict.fromkeys(norms, (0.1, 0.4))
    d2, margin = d2_margin_bands(
        predicted=predicted,
        target=target,
        logits=logits,
        head_norms=norms,
        sided_thresholds=thresholds,
    )
    assert d2[0, 0] == 0.25
    assert margin[0, 0] == 1
    source = np.zeros((HEIGHT, WIDTH), dtype=np.int8)
    temporal = temporal_pattern_codes(
        recurrence=np.zeros((HEIGHT, WIDTH), dtype=np.uint16),
        event_adjacent=False,
    )
    curvature = curvature_bands(target)
    key = encode_group_key(
        source=source,
        target=target,
        predicted=predicted,
        margin_band=margin,
        curvature_band=curvature,
        temporal_pattern=temporal,
        boundary_distance_band=np.zeros_like(source),
        paint_floor_mechanism=np.zeros_like(source),
    )
    decoded = decode_group_key(int(key[0, 0]))
    assert decoded == {
        "source": 0,
        "target": 0,
        "predicted": 1,
        "margin_band": 1,
        "curvature_band": 2,
        "temporal_pattern": 0,
        "boundary_distance_band": 0,
        "paint_floor_mechanism": 0,
    }


def test_boundary_distance_mechanism_and_survival_wall() -> None:
    target = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    target[:, 5:] = 1
    predicted = target.copy()
    predicted[10, 4] = 1
    predicted[10, 0] = 2
    distance, distance_band = boundary_distance_bands(target)
    assert distance[10, 4] == 0.0
    assert distance_band[10, 4] == 0
    assert distance_band[10, 0] == 1
    assert distance_band[10, 20] == 3

    margin = np.zeros((HEIGHT, WIDTH), dtype=np.int8)
    margin[10, 5] = 2
    mechanism, curve = paint_floor_mechanism_codes(
        target=target,
        predicted=predicted,
        margin_band=margin,
        boundary_distance_band=distance_band,
    )
    assert curve[10, 5]
    assert mechanism[10, 4] == 1
    assert mechanism[10, 5] == 2
    assert mechanism[10, 0] == 0
    assert mechanism[10, 20] == 2

    wall = survival_wall_149(target=target, predicted=predicted)
    assert wall["all_classes"] == {
        "sites": HEIGHT * 4,
        "errors": 1,
        "error_fraction": 1 / (HEIGHT * 4),
    }
    assert wall["by_target_class"]["Road"]["sites"] == HEIGHT * 2
    assert wall["by_target_class"]["Lane"]["sites"] == HEIGHT * 2


def test_component_and_budget_accounting() -> None:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[0, 0] = True
    mask[3:6, 3:6] = True
    summary = summarize_components(mask)
    assert summary.component_count == 2
    assert summary.pixel_mass_by_scale == {
        "POINT_LE4": 1,
        "BOUNDARY_SEGMENT_5_TO_64": 9,
        "REGION_GT64": 0,
    }
    budget = source_budget(
        counts={
            "DESCRIBED_BUT_REALIZATION_LOST": {"Road": 3},
            "NEVER_DESCRIBED": {"Undrivable": 4},
            "STRUCTURALLY_HARD_IRREDUCIBLE": {"MyCar": 5},
        },
        target_sites={"Road": 10, "Undrivable": 20, "MyCar": 30},
    )
    assert budget["total_errors"] == 12
    assert budget["global_d_seg"] == 12 / N600_SITES
