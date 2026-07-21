# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

from tac.optimization.predictor_r3_causal import (
    compose_curve,
    parse_causal_rules,
    parse_static_chart_quotient,
    serialize_causal_rules,
    serialize_static_chart_quotient,
)
from tac.optimization.predictor_upgrade_xi_chart import StaticCharts, serialize_static_charts


def test_static_chart_quotient_is_exact_and_smaller() -> None:
    ru = np.zeros((8, 16), dtype=np.uint8)
    ru[:, :5] = 1
    ru[:, 11:] = 2
    charts = StaticCharts(ru, np.eye(8, 16, dtype=np.bool_), ((0, 1), (0, 2), (1, 2)))
    pxch = serialize_static_charts(charts)
    quotient = serialize_static_chart_quotient(pxch)
    assert parse_static_chart_quotient(quotient) == pxch
    assert len(quotient) < len(pxch)


def test_causal_rule_wire_format_is_canonical() -> None:
    rules = {(0, 1, 2, 3, 4, 5), (2, 0, 0, 0, 0, 0)}
    payload = serialize_causal_rules(rules)
    assert set(parse_causal_rules(payload)) == rules
    assert serialize_causal_rules(parse_causal_rules(payload)) == payload


def test_compose_curve_decomposes_knee_exactly() -> None:
    base = {"stream_grouping": {"chosen_container_bytes": 40_000}}
    causal = {
        "models": {
            "adaptive_prior_frames": {
                "parameter_bytes": 0,
                "seed_free_hits": 100,
                "introduced_false_sites": 10,
            }
        }
    }
    residual = [
        {
            "class_name": "Road",
            "stratum": "cell_interior",
            "true_boundary_misses": 990,
            "seed_free_hits": 0,
            "introduced_false_sites": 10,
            "residual_packet": {"record_count": 1000, "container_bytes": 100},
        }
    ]
    components = [
        {
            "class_name": "Lane",
            "stratum": "boundary_codim1",
            "frame": 0,
            "pixels": 2000,
            "bytes": 200,
            "description_score_per_byte": 1.0e-5,
        }
    ]
    receipt = compose_curve(base=base, causal=causal, residual_rows=residual, component_candidates=components)
    headline = receipt["headline_decomposed"]
    assert headline["knee_total_bytes"] == 40_300
    assert (
        headline["base_entropy_bytes"] + headline["boundary_exception_bytes"] + headline["component_shape_bytes"]
        == 40_300
    )
    assert receipt["knee"]["remaining_misses"] == 3_118_996
