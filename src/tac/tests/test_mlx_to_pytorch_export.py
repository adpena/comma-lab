# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.local_acceleration.mlx_to_pytorch_export import (
    export_mlx_state_dict_to_torch_pt,
    load_pytorch_state_dict_from_pt,
)


def test_mlx_to_pytorch_export_preserves_non_float_dtypes(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pt_path = tmp_path / "state.pt"
    manifest = export_mlx_state_dict_to_torch_pt(
        {
            "indices": np.array([1, 2, 3], dtype=np.int64),
            "mask": np.array([True, False, True], dtype=np.bool_),
            "latent_fp16": np.array([0.25, -0.5], dtype=np.float16),
        },
        pt_path,
        substrate_id="unit_mlx_dtype",
        run_id="unit",
    )

    state = load_pytorch_state_dict_from_pt(pt_path)

    assert state["indices"].dtype == torch.int64
    assert state["mask"].dtype == torch.bool
    assert state["latent_fp16"].dtype == torch.float16
    assert state["indices"].tolist() == [1, 2, 3]
    assert state["mask"].tolist() == [True, False, True]
    assert manifest["dtype_policy"]["default"] == "preserve_numpy_dtype"
    assert manifest["per_tensor"]["indices"]["export_dtype"] == "int64"
    assert manifest["per_tensor"]["mask"]["export_dtype"] == "bool"
    assert manifest["per_tensor"]["latent_fp16"]["export_dtype"] == "float16"
    assert len(manifest["per_tensor"]["indices"]["sha256"]) == 64


def test_mlx_to_pytorch_export_casts_only_explicit_float32_names(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pt_path = tmp_path / "state.pt"
    manifest = export_mlx_state_dict_to_torch_pt(
        {
            "weight": np.array([1.0, 2.0], dtype=np.float64),
            "indices": np.array([1, 2], dtype=np.int64),
        },
        pt_path,
        substrate_id="unit_mlx_dtype",
        run_id="unit",
        force_float32_names=("weight",),
    )

    state = load_pytorch_state_dict_from_pt(pt_path)

    assert state["weight"].dtype == torch.float32
    assert state["indices"].dtype == torch.int64
    assert manifest["per_tensor"]["weight"]["source_dtype"] == "float64"
    assert manifest["per_tensor"]["weight"]["export_dtype"] == "float32"
    assert manifest["per_tensor"]["weight"]["forced_float32"] is True
    assert manifest["per_tensor"]["indices"]["forced_float32"] is False
    assert manifest["dtype_policy"]["force_float32_names"] == ["weight"]


def test_mlx_to_pytorch_export_rejects_unknown_force_float32_name(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="unknown tensors"):
        export_mlx_state_dict_to_torch_pt(
            {"weight": np.array([1.0], dtype=np.float32)},
            tmp_path / "state.pt",
            substrate_id="unit_mlx_dtype",
            run_id="unit",
            force_float32_names=("missing",),
        )
