from __future__ import annotations

from pathlib import Path

import pytest

from tac.canonical_equations.hcov_global_gluing_20260715 import (
    HCovGluingAtlas,
    HCovGluingStatus,
    HCovOverlapRestriction,
    HCovStratumChart,
    exact_array_quadrant_atlas_from_receipt,
)
from tac.canonical_equations.rate_law_ladder_measured_20260713 import (
    D37_MI_GROSS_BITS,
    D37_MI_NET_BITS,
    D37_MI_NET_CI95_BITS,
    D37_MI_TABLE_CHARGE_BYTES,
    D37_RECEIPT_SHA256,
    D38_GLOBAL_EXTENSION_STATUS,
)


def test_d37_v9_receipt_numbers_replace_stale_predecessor() -> None:
    assert D37_RECEIPT_SHA256 == (
        "60dd6a4837706d100932416cf8fdf77fce0e7c171b1ef58fd3f1154021428308"
    )
    assert pytest.approx(467_373.90888513427) == D37_MI_GROSS_BITS
    assert pytest.approx(
        D37_MI_GROSS_BITS - 8 * D37_MI_TABLE_CHARGE_BYTES
    ) == D37_MI_NET_BITS
    assert pytest.approx(384_637.90888513427) == D37_MI_NET_BITS
    assert pytest.approx(
        (373_674.7586229076, 395_236.54874890414)
    ) == D37_MI_NET_CI95_BITS


def test_measured_exact_array_instance_is_typed_but_not_rate_descent() -> None:
    atlas = exact_array_quadrant_atlas_from_receipt()
    assert not atlas.schema_violations()
    assert atlas.measured_overlap_points == 19_660_800
    assert atlas.status() is HCovGluingStatus.TYPED_UNBOUND
    assert "receiver_section_id" in atlas.unbound_global_rate_fields()
    assert "GLOBAL_RATE_DESCENT_UNBOUND" in D38_GLOBAL_EXTENSION_STATUS
    with pytest.raises(ValueError, match="global H_cov rate descent unavailable"):
        atlas.require_global_rate_descent()


def test_gluing_type_can_represent_a_fully_bound_charged_descent() -> None:
    charts = (
        HCovStratumChart("a", "regular", "K_a", "H_a", "coeff_a"),
        HCovStratumChart("b", "regular", "K_b", "H_b", "coeff_b"),
    )
    overlap = HCovOverlapRestriction(
        "ab",
        "a",
        "b",
        "rho_a_ab",
        "rho_b_ab",
        "alpha_ab",
        True,
    )
    atlas = HCovGluingAtlas(
        cover_id="synthetic-contract-test",
        charts=charts,
        overlaps=(overlap,),
        triple_cocycles=(),
        receiver_section_id="decode_section_v1",
        charged_section_bits=17.0,
    )
    assert atlas.status() is HCovGluingStatus.GLOBAL_RATE_DESCENT_TYPED
    atlas.require_global_rate_descent()


def test_gluing_schema_refuses_unknown_overlap_chart() -> None:
    atlas = HCovGluingAtlas(
        cover_id="bad",
        charts=(HCovStratumChart("a", "regular", None, None, None),),
        overlaps=(
            HCovOverlapRestriction("ab", "a", "missing", None, None, None, None),
        ),
        triple_cocycles=(),
        receiver_section_id=None,
        charged_section_bits=None,
    )
    assert atlas.status() is HCovGluingStatus.INVALID
    assert any("unknown chart" in item for item in atlas.schema_violations())


def test_canonical_dag_no_longer_repeats_stale_d37_scalar() -> None:
    repo = Path(__file__).resolve().parents[3]
    dag = (repo / ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md").read_text()
    section = dag[dag.index("## FEED-ladder-measurables-20260713") :]
    assert "+318,586 bits" not in section
    assert "+384,637.9089 bits" in section
