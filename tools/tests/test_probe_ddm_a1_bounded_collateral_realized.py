# SPDX-License-Identifier: MIT
from __future__ import annotations

import math

from tools.probe_ddm_a1_bounded_collateral_realized import _source_control


def test_source_control_prices_bound_score_row() -> None:
    receipt = {
        "solved_template_ladder": [
            {
                "candidate": "v15_solved_templates",
                "archive_bytes": 100,
                "archive_sha256": "a" * 64,
                "d_seg": "0.125",
                "d_pose": "0.4",
            }
        ],
        "producer_custody": [{"sha256": "b" * 64}],
    }
    control = _source_control(receipt)
    expected = 100.0 * 0.125 + math.sqrt(10.0 * 0.4) + 25.0 * 100 / 37_545_489
    assert math.isclose(float(control["advisory_score_formula_value"]), expected, abs_tol=1e-12)
    assert control["archive_sha256"] == "a" * 64
