# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "probe_weight_l1_int64",
    ROOT / "tools/probe_weight_l1_int64_scorer_forward_n600.py",
)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def _write(path: Path, *, measured: bool, exact: bool) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "mixed_int64_fixedpoint_scorer_n600.v1",
                "summary": {
                    "status": "MEASURED" if measured else "INCOMPLETE",
                    "full_real_n600": measured,
                    "argmax_exact_admitted": exact,
                },
            }
        ),
        encoding="utf-8",
    )


def test_weight_l1_successor_requires_full_negative_geometry_receipt(
    tmp_path: Path,
) -> None:
    path = tmp_path / "geometry.json"
    _write(path, measured=False, exact=False)
    with pytest.raises(ValueError, match="full-n600 custody"):
        PROBE._validate_geometry_predecessor(path)
    _write(path, measured=True, exact=True)
    with pytest.raises(ValueError, match="already exact"):
        PROBE._validate_geometry_predecessor(path)
    _write(path, measured=True, exact=False)
    payload = PROBE._validate_geometry_predecessor(path)
    assert payload["summary"]["full_real_n600"] is True
