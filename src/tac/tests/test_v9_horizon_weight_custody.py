# SPDX-License-Identifier: MIT
"""$0 custody tests for the V9 HORIZON-iso boundary-derived trainer weight."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_TRAINER = Path("experiments/train_levelset_witness_realized_through_R_mlx.py")


def _load_pure_helpers() -> dict[str, Any]:
    """Execute only the trainer's two pure HWM helpers, avoiding MLX imports."""
    tree = ast.parse(_TRAINER.read_text())
    wanted = {
        "_resolve_hwm_v9_stage_share_weight",
        "_hwm_v9_boundary_receipt",
    }
    body: list[ast.stmt] = []
    for node in tree.body:
        if (isinstance(node, (ast.Assign, ast.AnnAssign))
                and getattr(node, "value", None) is not None
                and any(isinstance(t, ast.Name) and t.id == "HWM_V9_BOUNDARY_RECEIPT_SCHEMA"
                        for t in (node.targets if isinstance(node, ast.Assign) else [node.target]))):
            body.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            body.append(node)
    assert {node.name for node in body if isinstance(node, ast.FunctionDef)} == wanted
    namespace: dict[str, Any] = {"np": np, "Any": Any}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(_TRAINER), "exec"), namespace)
    return namespace


def test_derived_weight_realizes_requested_boundary_share() -> None:
    ns = _load_pure_helpers()
    resolve = ns["_resolve_hwm_v9_stage_share_weight"]
    receipt = ns["_hwm_v9_boundary_receipt"]
    weight = resolve(loss_horizon_raw=3.0, loss_other=9.0, requested_share=0.15)
    assert weight == pytest.approx((0.15 / 0.85) * 3.0)
    row = receipt(
        epoch=726,
        n_pairs=600,
        loss_horizon_raw=3.0,
        loss_other=9.0,
        requested_share=0.15,
        resolved_weight=weight,
    )
    assert row["schema"] == "hwm_v9_stage_share_boundary.v1"
    assert row["realized_share_at_boundary"] == pytest.approx(0.15)
    assert row["frozen_model_state"] is True
    assert row["n_pairs"] == 600


@pytest.mark.parametrize(
    "lh,lo,q",
    [(-1.0, 2.0, 0.15), (1.0, -2.0, 0.15), (1.0, 2.0, 0.0), (1.0, 2.0, 1.0)],
)
def test_derived_weight_refuses_invalid_boundary_custody(lh: float, lo: float, q: float) -> None:
    resolve = _load_pure_helpers()["_resolve_hwm_v9_stage_share_weight"]
    with pytest.raises(ValueError):
        resolve(lh, lo, q)


def test_trainer_consumes_all_pairs_and_persists_resolved_weight() -> None:
    source = _TRAINER.read_text()
    assert "_loss_terms_for_chunk(\n                        range(P), seg_form, eik_w_ep)" in source
    assert "args.seg_horizon_margin_resolved_weight = hz_w" in source
    assert "__cfg_seg_horizon_margin_resolved_weight" in source
    assert "horizon_margin_boundary_receipt.json" in source
    assert "_atomic_write_json(" in source
    assert "--seg-horizon-margin-derived-live" in source
