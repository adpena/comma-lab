# SPDX-License-Identifier: MIT
"""Costate reader coverage for the ddm_vh2 canonical routing extension."""

from __future__ import annotations

from tools.costate_digest import _format_vehicle_routing_coverage


def test_vehicle_routing_coverage_surfaces_denominator_and_largest_gaps() -> None:
    report = {
        "totals": {
            "artifacts": 12,
            "harvested": 5,
            "routed": 4,
            "un_harvested": 7,
        },
        "lineages": {
            "v": {"un_harvested": 4},
            "ws": {"un_harvested": 2},
            "entropy": {"un_harvested": 0},
            "ic": {"un_harvested": 1},
        },
    }
    assert _format_vehicle_routing_coverage(report) == (
        "DDM-vehicle-harvest: 4/12 routed artifacts; harvested=5 un-harvested=7; largest gaps=v:4,ws:2,ic:1"
    )
