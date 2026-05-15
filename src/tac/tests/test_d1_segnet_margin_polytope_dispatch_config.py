# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_d1_remote_driver_passes_shrunk_margin_resolution_to_trainer() -> None:
    script = (
        REPO / "scripts/remote_lane_substrate_d1_segnet_margin_polytope.sh"
    ).read_text(encoding="utf-8")

    assert 'D1_POLYTOPE_MARGIN_H="${D1_POLYTOPE_MARGIN_H:-96}"' in script
    assert 'D1_POLYTOPE_MARGIN_W="${D1_POLYTOPE_MARGIN_W:-128}"' in script
    assert '--margin-h "$D1_POLYTOPE_MARGIN_H"' in script
    assert '--margin-w "$D1_POLYTOPE_MARGIN_W"' in script
    assert "'margin_h': $D1_POLYTOPE_MARGIN_H" in script
    assert "'margin_w': $D1_POLYTOPE_MARGIN_W" in script


def test_d1_operator_recipe_uses_shrunk_margin_resolution() -> None:
    recipe = (
        REPO
        / ".omx/operator_authorize_recipes/"
        / "substrate_d1_segnet_margin_polytope_modal_t4_dispatch.yaml"
    ).read_text(encoding="utf-8")

    assert 'D1_POLYTOPE_MARGIN_H: "96"' in recipe
    assert 'D1_POLYTOPE_MARGIN_W: "128"' in recipe
