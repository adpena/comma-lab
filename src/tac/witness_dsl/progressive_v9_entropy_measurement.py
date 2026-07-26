# SPDX-License-Identifier: MIT
"""Receiver-closed bridge for the research-only V9-conditioned PBR2 bound."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from tac.witness_dsl.factorized_v9_predictor import (
    PREDICTOR_CONTRACT_ID,
    FactorizedV9PredictorError,
    receive_factorized_v9_predictor,
)
from tac.witness_dsl.progressive_geometry_residual import (
    ProgressiveGeometryResidualError,
    apply_progressive_geometry_residual,
)


def apply_progressive_v9_entropy_measurement(
    predictor_program: bytes,
    progressive_residual: bytes,
    *,
    repository_root: Path | None = None,
    max_strata: int = 3,
) -> np.ndarray:
    """Re-derive V9 semantics from counted bytes, then apply research-only PBR2.

    This closes the decoder dependency for conditional-entropy measurement; it
    does not make the exact-target residual candidate-admissible. PBR2's own
    strict header keeps that prohibition machine-readable.
    """

    receiver = receive_factorized_v9_predictor(
        predictor_program,
        repository_root=repository_root,
    )
    try:
        return apply_progressive_geometry_residual(
            progressive_residual,
            predictor_program=predictor_program,
            predictor_contract_id=PREDICTOR_CONTRACT_ID,
            predictor_renderer_sha256=receiver.source_manifest_sha256,
            predictor_labels=receiver.decode_all_semantics(),
            source_pair_ids=receiver.source_pair_ids,
            max_strata=max_strata,
        )
    except ProgressiveGeometryResidualError as exc:
        raise FactorizedV9PredictorError("progressive V9 entropy measurement failed closure") from exc


__all__ = ["apply_progressive_v9_entropy_measurement"]
