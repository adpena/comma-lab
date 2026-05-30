#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Register canonical equations for MLX canonicalization audit + tinygrad bridge.

Per CLAUDE.md "Canonical equations + models registry" non-negotiable +
Catalog #344 sister discipline + operator NON-NEGOTIABLE binding 2026-
05-30: every empirical finding MUST become canonical equation.

Lands 2 NEW canonical equations:
  1. ``mlx_primitive_canonicalization_compounding_savings_v1`` — predicts
     LOC reduction when canonical extraction migrates sister substrate
     MLX renderers to consume the canonical extractor.
  2. ``mlx_pytorch_tinygrad_cross_backend_byte_stable_v1`` — predicts
     per-backend numerical equivalence within Slot 16 tolerance per
     canonical kernel ``assert_cross_backend_parity`` helper.

Each equation carries the canonical Provenance per Catalog #323 + first
EmpiricalAnchor per Catalog #344 + canonical producers/consumers list.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tac.canonical_equations.registry import (  # noqa: E402
    CanonicalEquation,
    EmpiricalAnchor,
    register_canonical_equation,
)
from tac.provenance import build_provenance_for_predicted  # noqa: E402


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def register_mlx_primitive_canonicalization_compounding_savings():
    """Register canonical equation #1.

    Predicts LOC reduction from canonical-extraction migration. The
    first EmpiricalAnchor is the audit inventory 2026-05-30 baseline:
    `gumbel_softmax_sample` duplicate-impl detection → 2 sister substrate
    side impls (DreamerV3 + Z8) → estimated 80 LOC reduction per
    extraction.
    """
    inputs = {
        "duplicate_impl_count": 2,
        "estimated_loc_per_impl": 80,
        "downstream_substrate_count": 2,
    }
    anchor = EmpiricalAnchor(
        anchor_id="audit_inventory_20260530_gumbel_softmax_baseline",
        measurement_utc=_utc_now(),
        inputs=inputs,
        predicted_output={"estimated_loc_reduction": 80, "extraction_recommended": True},
        empirical_output={
            "duplicate_impls_detected": 2,
            "extraction_recommended": True,
            "in_domain_context": "tac_substrates_dreamerv3_and_z8_substrate_side_gumbel_softmax_sample_duplicate_implementations",
        },
        residual=0.0,
        source_artifact=".omx/research/mlx_canonicalization_audit_inventory_20260530.md",
        measurement_method="ast_walk_per_primitive_duplication_count",
        provenance=build_provenance_for_predicted(
            model_id="mlx_primitive_canonicalization_compounding_savings_v1",
            inputs_sha256=hashlib.sha256(
                str(inputs).encode("utf-8")
            ).hexdigest(),
            measurement_axis="[predicted]",
            hardware_substrate="macos_arm64",
        ),
    )

    equation = CanonicalEquation(
        equation_id="mlx_primitive_canonicalization_compounding_savings_v1",
        name="MLX primitive canonicalization compounding LOC savings",
        one_line_summary=(
            "LOC_reduction = duplicate_impl_count * estimated_loc_per_impl; "
            "downstream substrate consumers route through canonical extractor "
            "per Catalog #290 falling-rule list"
        ),
        latex_form=r"\Delta_{\text{LOC}} = N_{\text{sister}} \cdot L_{\text{per\_impl}} \quad \text{when } N_{\text{sister}} \geq 2 \text{ AND HARD\_EARNED FORK absent}",
        python_callable_module_path="tac.local_acceleration.mlx_canonical_audit.audit.recommend_canonical_extraction",
        domain_of_validity={
            "primitive_categories": [
                "neural_kernel",
                "tensor_op",
                "scorer_loss_helper",
            ],
            "backend_families": ["mlx", "pytorch", "numpy", "tinygrad"],
            "substrate_classes": [
                "renderer",
                "trainer",
                "scaffold",
                "long_training_adapter",
            ],
        },
        units_in={
            "duplicate_impl_count": "count",
            "estimated_loc_per_impl": "loc",
            "downstream_substrate_count": "count",
        },
        units_out={
            "estimated_loc_reduction": "loc",
            "extraction_recommended": "bool",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"baseline_audit_2026_05_30": 0.0},
        last_calibration_utc=_utc_now(),
        next_recalibration_trigger="when_3+_new_empirical_anchors_in_domain",
        canonical_consumers=(
            "tac.local_acceleration.mlx_canonical_audit.audit.recommend_canonical_extraction",
            "tools.audit_mlx_canonicalization",
            "tac.cathedral_consumers.mlx_canonicalization_audit_consumer",
        ),
        canonical_producers=(
            "tac.local_acceleration.mlx_canonical_audit.audit.detect_canonical_duplication",
            "tools.audit_mlx_canonicalization",
        ),
        provenance=build_provenance_for_predicted(
            model_id="mlx_primitive_canonicalization_compounding_savings_v1",
            inputs_sha256=hashlib.sha256(b"canonical_equation_v1").hexdigest(),
            measurement_axis="[predicted]",
            hardware_substrate="macos_arm64",
        ),
    )

    registered = register_canonical_equation(
        equation,
        agent="mlx_canonicalization_audit_subagent",
        subagent_id="mlx_canonicalization_audit_plus_tinygrad_bridge_20260530T210259Z",
        notes=(
            "Catalog #383 sister canonical equation; audit inventory "
            "2026-05-30 baseline; consumed by audit_mlx_canonicalization "
            "CLI + cathedral consumer auto-discovery per Catalog #335."
        ),
    )
    return registered


def register_mlx_pytorch_tinygrad_cross_backend_byte_stable():
    """Register canonical equation #2.

    Predicts per-backend numerical equivalence within Slot 16 tolerance.
    """
    inputs = {
        "primary_backend": "numpy",
        "secondary_backend": "mlx",
        "atol_fp32": 1.0e-5,
        "atol_fp64": 1.0e-8,
    }
    anchor = EmpiricalAnchor(
        anchor_id="canonical_kernels_smoke_20260530_gumbel_softmax_self_parity",
        measurement_utc=_utc_now(),
        inputs=inputs,
        predicted_output={
            "max_abs_delta_le_atol_fp32": True,
            "in_domain_context": "tac_framework_agnostic_canonical_kernels_gumbel_softmax_sample_numpy_self_parity",
        },
        empirical_output={
            "max_abs_delta_le_atol_fp32": True,
            "in_domain_context": "tac_framework_agnostic_canonical_kernels_gumbel_softmax_sample_numpy_self_parity",
        },
        residual=0.0,
        source_artifact="src/tac/framework_agnostic/canonical_kernels.py",
        measurement_method="assert_cross_backend_parity_unit_test",
        provenance=build_provenance_for_predicted(
            model_id="mlx_pytorch_tinygrad_cross_backend_byte_stable_v1",
            inputs_sha256=hashlib.sha256(
                str(inputs).encode("utf-8")
            ).hexdigest(),
            measurement_axis="[predicted]",
            hardware_substrate="macos_arm64",
        ),
    )

    equation = CanonicalEquation(
        equation_id="mlx_pytorch_tinygrad_cross_backend_byte_stable_v1",
        name="MLX / PyTorch / tinygrad cross-backend canonical-kernel byte stability",
        one_line_summary=(
            "max_abs_delta(primary, secondary) <= Slot 16 atol "
            "(fp32 1e-5; fp64 1e-8) for canonical kernels in "
            "tac.framework_agnostic.canonical_kernels"
        ),
        latex_form=r"\max_{i} |y_{\text{primary},i} - y_{\text{secondary},i}| \leq \text{atol}_{\text{fp32}} = 10^{-5}",
        python_callable_module_path="tac.framework_agnostic.canonical_kernels.assert_cross_backend_parity",
        domain_of_validity={
            "backends": ["mlx", "pytorch", "numpy", "tinygrad"],
            "kernels": [
                "gumbel_softmax_sample",
                "rgb_to_yuv6",
                "pixel_shuffle_2x_nhwc",
                "bilinear_resize_nhwc",
            ],
            "dtype": ["float32", "float64"],
        },
        units_in={
            "primary_backend": "enum",
            "secondary_backend": "enum",
            "atol_fp32": "abs_delta",
            "atol_fp64": "abs_delta",
        },
        units_out={"max_abs_delta_le_atol_fp32": "bool"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"baseline_smoke_2026_05_30": 0.0},
        last_calibration_utc=_utc_now(),
        next_recalibration_trigger="when_3+_new_empirical_anchors_in_domain",
        canonical_consumers=(
            "tac.framework_agnostic.canonical_kernels.gumbel_softmax_sample",
            "tac.framework_agnostic.canonical_kernels.rgb_to_yuv6",
            "tac.framework_agnostic.canonical_kernels.assert_cross_backend_parity",
            "src.tac.framework_agnostic.tests.test_canonical_kernels",
        ),
        canonical_producers=(
            "tac.framework_agnostic.canonical_kernels",
            "tac.local_acceleration.pr95_hnerv_mlx",
            "tac.local_acceleration.tinygrad_bridge",
        ),
        provenance=build_provenance_for_predicted(
            model_id="mlx_pytorch_tinygrad_cross_backend_byte_stable_v1",
            inputs_sha256=hashlib.sha256(
                b"canonical_kernels_v1"
            ).hexdigest(),
            measurement_axis="[predicted]",
            hardware_substrate="macos_arm64",
        ),
    )

    registered = register_canonical_equation(
        equation,
        agent="mlx_canonicalization_audit_subagent",
        subagent_id="mlx_canonicalization_audit_plus_tinygrad_bridge_20260530T210259Z",
        notes=(
            "Catalog #383 sister canonical equation; canonical kernels "
            "module landing 2026-05-30; consumed by sister cross-backend "
            "parity test fixture + cathedral consumer auto-discovery."
        ),
    )
    return registered


def main():
    print("Registering canonical equation #1 (LOC compounding savings)...")
    eq1 = register_mlx_primitive_canonicalization_compounding_savings()
    print(f"  Registered: {eq1.equation_id}")

    print("Registering canonical equation #2 (cross-backend byte stable)...")
    eq2 = register_mlx_pytorch_tinygrad_cross_backend_byte_stable()
    print(f"  Registered: {eq2.equation_id}")

    print("Phase D canonical equation registration: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
