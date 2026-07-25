from __future__ import annotations

import json
from pathlib import Path

from tools.register_ddm_db1_margin_mass_equation import EQUATION_ID, build_equation


def test_build_equation_from_minimal_receipt(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "written_at_utc": "2026-07-25T12:41:20Z",
                "sn1_at1_margin_mass": {
                    "total_ordered_boundary_incidences": 105,
                    "total_unique_boundary_pixels": 100,
                    "total_duplicate_incidences": 5,
                    "sn1_custody": {"receipt": {"sha256": "a" * 64}},
                    "at1_custody": {"atlas_canonical_payload_sha256": "b" * 64},
                },
            }
        ),
        encoding="utf-8",
    )
    equation = build_equation(receipt)
    assert equation.equation_id == EQUATION_ID
    assert equation.python_callable_module_path.endswith(":unique_count_bounds")
    assert equation.empirical_anchors[0].residual == 0.0
