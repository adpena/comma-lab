# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import math

import pytest

from tac.derived_upstream_refresh import (
    DISPOSITION_QUEUED_HEAVY_REFRESH,
    DISPOSITION_REFRESHED_SCORER_FREE,
    RefreshRegistryError,
    load_refresh_registry_jsonl,
    require_fresh_for_consumption,
)
from tools.build_ddm_uf1_refresh_registry import (
    QO1_ARCHIVE_SHA256,
    build_outputs,
    build_refresh_rows,
    recompute_m66_gap,
)


def test_uf1_registry_contains_charter_rows() -> None:
    rows = build_refresh_rows()
    ids = {row.quantity_id for row in rows}
    assert {
        "pf2_1200_row_atlas",
        "ms3_ms6_metric_bundle",
        "margin_saliency_maps_141",
        "sensitivity_bitalloc_maps_157_336",
        "g3_score_atlas",
        "g4_stationarity_maps",
        "g2f_trust_regions",
        "r9m_advisory_to_contest_cpu_calibration_prior",
        "W_bytes_per_flip_exchange",
        "m66_gap_decomposition_inputs_qo1",
        "prefix_bias_ratios_931",
        "fiber_transport_ab_rows_891",
    } <= ids
    assert all(row.consumers for row in rows)
    assert all(row.trigger for row in rows)


def test_r9m_prior_is_not_consumable_for_qo1_without_refresh() -> None:
    row = {
        row.quantity_id: row for row in build_refresh_rows()
    }["r9m_advisory_to_contest_cpu_calibration_prior"]
    assert row.disposition == DISPOSITION_QUEUED_HEAVY_REFRESH
    with pytest.raises(RefreshRegistryError, match="QUEUED_HEAVY_REFRESH"):
        require_fresh_for_consumption(
            row,
            current_base_sha256=QO1_ARCHIVE_SHA256,
            consumer="contest-axis projection writers",
        )


def test_m66_gap_is_recomputed_from_components() -> None:
    m66 = recompute_m66_gap()
    assert m66["ours"]["S"] == pytest.approx(0.7539807296911207, abs=1e-15)
    assert m66["gap"]["seg"] == pytest.approx(0.401519, abs=1e-12)
    assert m66["gap"]["pose"] == pytest.approx(
        math.sqrt(10 * 0.00071459) - math.sqrt(10 * 2.331e-5), abs=1e-15
    )
    assert m66["gap"]["total"] == pytest.approx(
        m66["gap"]["seg"] + m66["gap"]["pose"] + m66["gap"]["rate"], abs=1e-15
    )
    assert m66["gap"]["shares"]["seg"] > 0.68
    assert m66["gap"]["rank_by_gap"] == ["seg", "rate", "pose"]


def test_build_outputs_writes_typed_receipts_without_tmp_paths(tmp_path) -> None:
    outputs = build_outputs(tmp_path)
    rows = load_refresh_registry_jsonl(outputs.registry_path)
    assert len(rows) == 12
    assert any(row.disposition == DISPOSITION_REFRESHED_SCORER_FREE for row in rows)

    summary = json.loads(outputs.summary_path.read_text(encoding="utf-8"))
    assert summary["denominators"]["quantities_found"] == 12
    assert summary["denominators"]["with_consumers"] == 12
    assert summary["denominators"]["with_triggers"] == 12
    assert (
        "r9m_advisory_to_contest_cpu_calibration_prior"
        in summary["freshness_guard"]["consumption_refusals"]
    )

    transient_token = "/" + "tmp"
    for path in outputs.output_dir.rglob("*"):
        if path.is_file():
            assert transient_token not in path.read_text(encoding="utf-8", errors="ignore")
