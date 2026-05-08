"""Tests for beta-Fisher lossy-coarsening tensor weight export."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from tac.optimization.beta_fisher_lossy_weights import (
    BetaFisherWeightError,
    ScoreWeightConfig,
    TensorWeightTarget,
    build_tensor_weight_rows,
    load_tensor_scalar_json,
    resolve_sensitivity_scalar,
    select_weighted_k_allocations,
)
from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA
from tac.sensitivity_map import save_sensitivity_map

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_tool():
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    spec = importlib.util.spec_from_file_location(
        "build_beta_fisher_lossy_coarsening_weights",
        REPO_ROOT / "tools" / "build_beta_fisher_lossy_coarsening_weights.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_beta_fisher_lossy_coarsening_weights"] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_sensitivity_scalar_maps_weight_vector_to_bias() -> None:
    sensitivities = {"block.weight": torch.tensor([2.0, 4.0, 6.0])}
    bias = TensorWeightTarget("block.bias", (3,))
    resolved = resolve_sensitivity_scalar(bias, sensitivities)

    assert resolved.matched is True
    assert resolved.source_key == "block.weight"
    assert resolved.reduction == "mean_output_channels"
    assert resolved.value == 4.0


def test_build_tensor_weight_rows_combines_fisher_boundary_and_grain() -> None:
    targets = [
        TensorWeightTarget("sensitive.weight", (2,), symbols=np.array([1, -1, 2, -2], dtype=np.int32)),
        TensorWeightTarget("grainy.weight", (2,), symbols=np.array([80, -80, 70, -70], dtype=np.int32)),
        TensorWeightTarget("plain.weight", (2,), symbols=np.array([3, -3, 3, -3], dtype=np.int32)),
    ]
    sensitivities = {
        "sensitive.weight": torch.tensor([10.0, 10.0]),
        "grainy.weight": torch.tensor([1.0, 1.0]),
        "plain.weight": torch.tensor([1.0, 1.0]),
    }
    payload = build_tensor_weight_rows(
        targets,
        sensitivities,
        config=ScoreWeightConfig(
            fisher_beta=1.0,
            boundary_alpha=1.0,
            film_grain_alpha=1.0,
        ),
        boundary_mass={"sensitive.weight": 4.0},
        film_grain_capacity={"grainy.weight": 8.0},
    )

    rows = {row["tensor_name"]: row for row in payload["per_tensor"]}
    assert rows["sensitive.weight"]["allocator_weight"] > rows["plain.weight"]["allocator_weight"]
    assert rows["grainy.weight"]["allocator_weight"] < rows["plain.weight"]["allocator_weight"]
    assert payload["allocator_input"]["weight_semantics"] == "cost = bytes + lambda * weight[t] * rel_err[t]^2"


def test_load_tensor_scalar_json_accepts_rows_and_mapping(tmp_path: Path) -> None:
    path = tmp_path / "boundary.json"
    path.write_text(
        """
        {
          "rows": [
            {"tensor_name": "a.weight", "boundary_mass": 2.5},
            {"name": "b.weight", "value": 1.5}
          ]
        }
        """,
        encoding="utf-8",
    )
    assert load_tensor_scalar_json(path, value_key="boundary_mass") == {
        "a.weight": 2.5,
        "b.weight": 1.5,
    }


def test_select_weighted_k_allocations_exports_selected_Ks() -> None:
    from tac.codec.cost_curves import TensorBlob

    blobs = [
        TensorBlob("hi.weight", np.array([1, -1, 1, -1, 2, -2], dtype=np.int32)),
        TensorBlob("lo.weight", np.array([50, -50, 60, -60, 70, -70], dtype=np.int32)),
    ]
    allocations = select_weighted_k_allocations(
        blobs,
        weights=[1000.0, 0.001],
        rms_targets=[0.25],
        k_range=[1, 2, 4],
    )

    assert len(allocations) == 1
    assert len(allocations[0]["selected_Ks"]) == 2
    assert allocations[0]["selected_K_by_tensor"][0]["tensor_name"] == "hi.weight"


def test_select_weighted_k_allocations_rejects_nonfinite_inputs() -> None:
    from tac.codec.cost_curves import TensorBlob

    blobs = [TensorBlob("x.weight", np.array([1, -1, 2, -2], dtype=np.int32))]
    try:
        select_weighted_k_allocations(
            blobs,
            weights=[float("nan")],
            rms_targets=[0.1],
            k_range=[1, 2],
        )
    except BetaFisherWeightError as exc:
        assert "weights must be finite" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected non-finite weight rejection")

    try:
        select_weighted_k_allocations(
            blobs,
            weights=[1.0],
            rms_targets=[float("nan")],
            k_range=[1, 2],
        )
    except BetaFisherWeightError as exc:
        assert "rms_targets must be non-negative" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected non-finite target rejection")


def test_tool_builds_planning_manifest_on_synthetic_pr101_state(tmp_path: Path) -> None:
    tool = _load_tool()
    rng = np.random.default_rng(7)
    state = {}
    sensitivities = {}
    for idx, (name, shape) in enumerate(FIXED_STATE_SCHEMA):
        state[name] = torch.tensor(rng.normal(0.0, 1.0, size=shape).astype(np.float32))
        # Keep the sensitivity map small but ordered and non-uniform.
        sensitivities[name] = torch.tensor([1.0 + 0.1 * idx], dtype=torch.float32)

    state_dict = tmp_path / "state_dict.pt"
    sensitivity_map = tmp_path / "sensitivity.pt"
    output_json = tmp_path / "manifest.json"
    torch.save(state, state_dict)
    save_sensitivity_map(
        sensitivity_map,
        sensitivities,
        metadata={"device": "cpu", "source": "unit-diagnostic", "is_stub": True},
    )

    rc = tool.main(
        [
            "--state-dict",
            str(state_dict),
            "--sensitivity-map",
            str(sensitivity_map),
            "--allow-diagnostic-sensitivity",
            "--rms-targets",
            "0.05",
            "--max-K",
            "4",
            "--output-json",
            str(output_json),
        ]
    )
    assert rc == 0
    manifest = tool.REPO_ROOT.joinpath(output_json).read_text() if not output_json.is_absolute() else output_json.read_text()
    assert '"score_claim": false' in manifest
    assert '"ready_for_exact_eval_dispatch": false' in manifest
    assert '"selected_Ks"' in manifest
    assert "ADMM_PATH_B_STEP6_KS" in manifest


def test_tool_rejects_diagnostic_sensitivity_without_escape_hatch(tmp_path: Path) -> None:
    tool = _load_tool()
    state = {
        name: torch.ones(shape, dtype=torch.float32)
        for name, shape in FIXED_STATE_SCHEMA
    }
    state_dict = tmp_path / "state_dict.pt"
    sensitivity_map = tmp_path / "sensitivity.pt"
    torch.save(state, state_dict)
    save_sensitivity_map(
        sensitivity_map,
        {name: torch.ones(1) for name, _shape in FIXED_STATE_SCHEMA},
        metadata={"device": "cpu", "is_stub": True},
    )

    try:
        tool.main(
            [
                "--state-dict",
                str(state_dict),
                "--sensitivity-map",
                str(sensitivity_map),
                "--rms-targets",
                "0.05",
                "--max-K",
                "2",
                "--output-json",
                str(tmp_path / "manifest.json"),
            ]
        )
    except Exception as exc:
        assert "diagnostic/stub sensitivity rejected" in str(exc)
    else:  # pragma: no cover - asserts fail loudly in normal pytest
        raise AssertionError("diagnostic sensitivity should require an explicit escape hatch")
