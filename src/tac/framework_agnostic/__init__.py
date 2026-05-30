# SPDX-License-Identifier: MIT
"""Canonical framework-agnostic portability primitives + decorators.

Per operator NON-NEGOTIABLE META directive 2026-05-28 verbatim: *"remmebr
MLX first but agnostic portability via numpy and tinygrad like primitives
and helpers or decorators or whatever"*.

Sister of ``tac.substrates._shared.inflate_runtime.select_inflate_device``
per Catalog #205 at the **framework-selection surface at training-time**.

Closes the implicit **duplicate-trainer anti-pattern** empirically anchored
by the V3 sister trainer pair:

  * ``experiments/train_substrate_pact_nerv_selector_v3.py`` (PyTorch; 783 LOC)
  * ``experiments/train_substrate_pact_nerv_selector_v3_mlx_local.py`` (MLX; 720 LOC)
  * ≈95% LOC overlap; ≈1503 LOC total

Per the canonical canonical-anti-patterns registry (sister landing
2026-05-28): the anti-pattern is
``mlx_trainer_pytorch_sister_duplicated_implementation_v1`` with
canonical_unwind_path = consume :mod:`tac.framework_agnostic` primitives +
``@framework_agnostic`` / ``@mlx_first_with_numpy_fallback`` /
``@pytorch_first_with_numpy_fallback`` decorators.

Public API (narrow per CLAUDE.md "Beauty, simplicity, and developer
experience"):

  * :class:`Backend` — canonical 4-backend taxonomy + AUTO sentinel
  * :func:`select_backend` — canonical priority cascade per Catalog #205 sister
  * :func:`detect_available_backends` — runtime detection
  * :class:`BackendUnavailableError` — fail-closed selection error
  * :class:`FrameworkAgnosticTensor` — runtime_checkable Protocol per Catalog #335 sister
  * :func:`quantize_int8_per_channel` / :func:`dequantize_int8_per_channel` —
    per-channel symmetric int8 quantization (byte-deterministic across backends)
  * :func:`quantize_fp4_packed_nibbles` — canonical Quantizr unsigned-E2M1 FP4
  * :func:`brotli_compress` — canonical entropy coder per CLAUDE.md hard dep
  * :func:`mlx_state_dict_to_npz_bridge` / sister — canonical bridge contract
    per CLAUDE.md "MLX-FIRST NUMPY-PORTABLE INDIVIDUALLY-FRACTAL" 8th directive
  * ``@framework_agnostic`` / ``@mlx_first_with_numpy_fallback`` /
    ``@pytorch_first_with_numpy_fallback`` / ``@inflate_runtime_helper`` —
    canonical decorators per operator META directive

Cross-references:
  * Catalog #205 — sister gate at inflate-time device-selection surface
  * Catalog #335 — canonical cathedral consumer auto-discovery
  * Catalog #341 — Tier A canonical-routing markers
  * Catalog #287 — placeholder-rationale rejection sister discipline
  * Catalog #344 — canonical equations registry (sister equation landed
    2026-05-28: ``framework_agnostic_backend_abstraction_compounding_v1``)
  * CLAUDE.md "MLX-FIRST NUMPY-PORTABLE INDIVIDUALLY-FRACTAL" 8th standing
    directive (canonical contract this package operationalizes)
  * CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" — per-substrate
    forks are accepted when canonical abstraction suppresses score; the
    canonical helpers serve as the default reusable surface
"""
from __future__ import annotations

from tac.framework_agnostic.backend import (
    DEFAULT_ENV_VAR,
    Backend,
    BackendUnavailableError,
    detect_available_backends,
    select_backend,
)
from tac.framework_agnostic.decorators import (
    framework_agnostic,
    inflate_runtime_helper,
    mlx_first_with_numpy_fallback,
    pytorch_first_with_numpy_fallback,
)
from tac.framework_agnostic.helpers import (
    assert_no_framework_mismatch,
    convert_mlx_state_dict_to_pytorch_oihw,
    detect_available_backends_dict,
    mlx_state_dict_to_npz_bridge,
    npz_to_numpy_primitives,
    pytorch_state_dict_to_npz_bridge,
    tinygrad_state_dict_to_npz_bridge,
)
from tac.framework_agnostic.mlx_runtime import (
    MlxRuntime,
    is_mlx_runtime_available,
    mlx_array,
    mlx_compile,
    mlx_eval,
    optional_mlx_runtime,
    require_mlx_core,
    require_mlx_nn,
    require_mlx_optimizers,
    require_mlx_runtime,
    require_mlx_utils,
)
from tac.framework_agnostic.operations import (
    brotli_compress,
    dequantize_int8_per_channel,
    quantize_fp4_packed_nibbles,
    quantize_int8_per_channel,
)
from tac.framework_agnostic.tensor_protocol import (
    FrameworkAgnosticTensor,
    dtype_name,
    shape_of,
)

__all__ = [  # noqa: RUF022 - grouped by public contract surface.
    # Backend selection (Catalog #205 sister)
    "Backend",
    "BackendUnavailableError",
    "DEFAULT_ENV_VAR",
    "detect_available_backends",
    "select_backend",
    # Tensor Protocol (Catalog #335 sister)
    "FrameworkAgnosticTensor",
    "MlxRuntime",
    "dtype_name",
    "is_mlx_runtime_available",
    "shape_of",
    # Operations (canonical primitives per CLAUDE.md QAT pipeline)
    "brotli_compress",
    "dequantize_int8_per_channel",
    "quantize_fp4_packed_nibbles",
    "quantize_int8_per_channel",
    # Decorators (operator META directive 2026-05-28)
    "framework_agnostic",
    "inflate_runtime_helper",
    "mlx_array",
    "mlx_compile",
    "mlx_eval",
    "mlx_first_with_numpy_fallback",
    "optional_mlx_runtime",
    "pytorch_first_with_numpy_fallback",
    "require_mlx_core",
    "require_mlx_nn",
    "require_mlx_optimizers",
    "require_mlx_runtime",
    "require_mlx_utils",
    # Bridge helpers (CLAUDE.md 8th standing directive)
    "assert_no_framework_mismatch",
    "convert_mlx_state_dict_to_pytorch_oihw",
    "detect_available_backends_dict",
    "mlx_state_dict_to_npz_bridge",
    "npz_to_numpy_primitives",
    "pytorch_state_dict_to_npz_bridge",
    "tinygrad_state_dict_to_npz_bridge",
]
