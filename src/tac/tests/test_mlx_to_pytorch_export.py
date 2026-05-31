# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.local_acceleration.mlx_to_pytorch_export import (
    MLX_BRIDGE_FALSE_AUTHORITY_BLOCKERS,
    build_mlx_pytorch_forward_parity_proof,
    build_substrate_bridge_manifest,
    export_mlx_state_dict_to_torch_pt,
    load_pytorch_state_dict_from_pt,
    transpose_mlx_hwio_state_dict_to_pytorch_oihw,
)
from tac.local_acceleration.pact_nerv_ia3_export_parity import (
    PACT_NERV_IA3_MLX_PYTORCH_FORWARD_PARITY_SCHEMA,
    prove_pact_nerv_ia3_mlx_pytorch_forward_parity,
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


def test_mlx_to_pytorch_export_all_public_symbols_are_defined() -> None:
    import tac.local_acceleration.mlx_to_pytorch_export as bridge

    missing = [name for name in bridge.__all__ if not hasattr(bridge, name)]

    assert missing == []


def test_mlx_bridge_manifest_uses_shared_false_authority_blockers(
    tmp_path: Path,
) -> None:
    manifest = export_mlx_state_dict_to_torch_pt(
        {"weight": np.array([1.0], dtype=np.float32)},
        tmp_path / "state.pt",
        substrate_id="unit_bridge",
        run_id="unit",
    )

    assert manifest["blockers"] == list(MLX_BRIDGE_FALSE_AUTHORITY_BLOCKERS)
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_transpose_mlx_hwio_state_dict_to_pytorch_oihw_wrapper() -> None:
    torch = pytest.importorskip("torch")
    arr = np.arange(2 * 3 * 5 * 7, dtype=np.float32).reshape(2, 3, 5, 7)

    pytorch_sd, per_tensor = transpose_mlx_hwio_state_dict_to_pytorch_oihw(
        {"decoder.0.weight": arr, "latents": np.ones((2, 4), dtype=np.float32)}
    )

    assert pytorch_sd["decoder.0.weight"].shape == torch.Size([2, 7, 3, 5])
    assert per_tensor["decoder.0.weight"]["layout"] == "mlx_hwio_to_pytorch_oihw"
    assert per_tensor["latents"]["layout"] == "preserved"


def test_build_mlx_pytorch_forward_parity_proof_is_fail_closed() -> None:
    proof = build_mlx_pytorch_forward_parity_proof(
        mlx_output=np.array([0.25, 0.5], dtype=np.float32),
        pytorch_output=np.array([0.25, 0.5], dtype=np.float32),
        sample_pair_indices=[0, 2],
    )

    assert proof["parity_passed"] is True
    assert proof["blockers"] == list(MLX_BRIDGE_FALSE_AUTHORITY_BLOCKERS)
    assert proof["score_claim"] is False
    assert proof["promotion_eligible"] is False
    assert proof["ready_for_exact_eval_dispatch"] is False


def test_build_substrate_bridge_manifest_rejects_authority_override() -> None:
    manifest = build_substrate_bridge_manifest(
        schema_version="unit_bridge.v1",
        tool="unit",
        source_state_dict_path="source.npsd",
        output_pytorch_state_dict="state.pt",
        source_state_dict_sha256="a" * 64,
        pytorch_state_dict_sha256="b" * 64,
        pytorch_state_dict_bytes=17,
        tensor_count=2,
        config={"num_pairs": 1},
        per_tensor={"weight": {"shape": [1]}},
    )

    assert manifest["blockers"] == list(MLX_BRIDGE_FALSE_AUTHORITY_BLOCKERS)
    assert manifest["score_claim"] is False
    with pytest.raises(ValueError, match="score_claim=True forbidden"):
        build_substrate_bridge_manifest(
            schema_version="unit_bridge.v1",
            tool="unit",
            source_state_dict_path="source.npsd",
            output_pytorch_state_dict="state.pt",
            source_state_dict_sha256="a" * 64,
            pytorch_state_dict_sha256="b" * 64,
            pytorch_state_dict_bytes=17,
            tensor_count=2,
            extra_fields={"score_claim": True},
        )


def test_pact_nerv_ia3_mlx_pytorch_forward_parity_smoke(tmp_path: Path) -> None:
    pytest.importorskip("mlx.core")
    pytest.importorskip("torch")

    report = prove_pact_nerv_ia3_mlx_pytorch_forward_parity(
        pair_indices=[0, 1, 2],
        output_pt_path=tmp_path / "ia3.pt",
        seed=7,
    )

    assert report["schema"] == PACT_NERV_IA3_MLX_PYTORCH_FORWARD_PARITY_SCHEMA
    assert report["parity_passed"] is True
    assert report["max_abs_diff_255"] <= report["tolerance"]
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["export_manifest"]["file_size_bytes"] > 0
    assert (tmp_path / "ia3.pt").is_file()
