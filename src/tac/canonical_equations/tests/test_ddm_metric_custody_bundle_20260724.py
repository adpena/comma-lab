# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from tac.canonical_equations.ddm_metric_custody_bundle_20260724 import (
    COMPONENTS,
    EQUATION_ID,
    build_ddm_metric_custody_bundle_completion_v1,
    metric_custody_bundle_completion_law,
    populate_ddm_metric_custody_bundle_completion,
)
from tac.canonical_equations.registry import query_equations


def _mapping(value: object) -> dict[str, object]:
    return dict.fromkeys(COMPONENTS, value)


def test_completion_law_requires_every_exact_surface() -> None:
    assert metric_custody_bundle_completion_law(
        _mapping(True),
        _mapping(True),
        _mapping(600),
        _mapping(32),
        seg_bucket_count=1200,
        composite_r_bucket_count=1200,
        dual_bucket_count=1200,
    )
    incomplete = _mapping(True)
    incomplete["POSE_METRIC"] = False
    assert not metric_custody_bundle_completion_law(
        incomplete,
        _mapping(True),
        _mapping(600),
        _mapping(32),
        seg_bucket_count=1200,
        composite_r_bucket_count=1200,
        dual_bucket_count=1200,
    )


def test_completion_law_refuses_implicit_or_missing_components() -> None:
    bad = _mapping(True)
    bad.pop("SEG_METRIC")
    with pytest.raises(ValueError, match="exactly four"):
        metric_custody_bundle_completion_law(
            bad,
            _mapping(True),
            _mapping(600),
            _mapping(32),
            seg_bucket_count=1200,
            composite_r_bucket_count=1200,
            dual_bucket_count=1200,
        )


def test_equation_callable_imports_and_registry_round_trips(tmp_path: Path) -> None:
    equation = build_ddm_metric_custody_bundle_completion_v1()
    module_name, callable_name = equation.python_callable_module_path.split(":")
    assert getattr(importlib.import_module(module_name), callable_name) is (metric_custody_bundle_completion_law)
    ledger = tmp_path / "registry.jsonl"
    lock = tmp_path / "registry.jsonl.lock"
    populate_ddm_metric_custody_bundle_completion(
        path=ledger,
        lock_path=lock,
        agent="codex",
        subagent_id="ddm_ms3_metric_custody_bundle_20260724T035249Z",
    )
    rows = query_equations(path=ledger)
    assert [row.equation_id for row in rows] == [EQUATION_ID]
    assert rows[0].domain_of_validity["score_claim"] is False
