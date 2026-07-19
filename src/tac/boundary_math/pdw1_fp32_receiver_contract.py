"""PDW1 frozen-fp32 receiver arithmetic — THE single declared cell-assignment contract.

Contract id: ``pdw1-native-f32-power-first-max.v1``

This module declares, as a wire contract, the ONE arithmetic under which PDW1
power-diagram cell assignment is evaluated (task #543 closure; the same move
that made the uint8-lattice realization exact: frozen CPU-fp32 is THE receiver
arithmetic, and every other evaluation — float64 included — is non-authoritative).

The contract, exactly:

1. Inputs are float32: quotient points ``z`` (…,4), target sites ``s`` (K,4)
   and weights ``w`` (K,), taken verbatim from the serialized PDW1 bytes
   (float32 on the wire; no promotion).
2. Scores are the expanded power scores with the common ``-||z||^2`` removed::

       score_c = f32(2) * sum_f32(s_c * z) + w_c - sum_f32(s_c * s_c)

   Every product and reduction is float32.  The reductions are NumPy
   ``np.sum(..., dtype=np.float32)`` over the trailing axis of an elementwise
   product — never a BLAS matmul, whose reduction order is not part of any
   contract.  This is byte-for-byte the arithmetic of the sealed frame-195
   diagnostic (``tools/diagnose_v10_power_diagram_frame195.py::
   native_f32_power_scores``), lifted to batch form.
3. The label is ``argmax`` over classes with NumPy's deterministic FIRST-max
   tie rule, matching the frozen scorer's first-max argmax convention and the
   PDW2 half-open-cell finding: an exact fp32 score tie is assigned to the
   lowest class index.  Cells are half-open by declaration.

Anything the contract does not fix (float64 evaluation, reassociated
reductions, alternate tie rules) is explicitly NON-AUTHORITATIVE for PDW1
labels.  The frame-195 blocker pixel is the canonical fixture: generic float64
picks class 1 by 2.53e-7 while this contract produces an exact fp32 tie and
first-max class 0 — agreeing with the frozen CPU-Torch forward (margin
4.77e-7) and the cached ``L*``.

Research-only: no score, promotion, or archive authority.
"""

from __future__ import annotations

import numpy as np

from tac.boundary_math.power_diagram_witness import (
    PowerDiagramTarget,
    PowerDiagramWitnessError,
)

CONTRACT_ID = "pdw1-native-f32-power-first-max.v1"

#: Frame-195 canonical fixture (from the sealed diagnostic receipt
#: ``.omx/research/v10_power_diagram_frame195_diagnostic_20260718.json``).
FRAME195_QUOTIENT = (
    0.0014007954159751534,
    5.777266502380371,
    1.7700754404067993,
    3.2382075786590576,
)
FRAME195_CACHED_LSTAR = 0
FRAME195_GENERIC_F64_ARGMAX = 1


def contract_f32_power_scores(points: np.ndarray, target: PowerDiagramTarget) -> np.ndarray:
    """Frozen-fp32 power scores under ``CONTRACT_ID`` (batched, (…,K) float32)."""

    z = np.asarray(points)
    if z.dtype != np.float32:
        # The contract consumes float32 points.  A float64 input would silently
        # change the products; refuse instead of converting behind the caller.
        raise PowerDiagramWitnessError(
            f"{CONTRACT_ID} requires float32 points; got dtype {z.dtype}"
        )
    if z.ndim < 1 or z.shape[-1] != target.rank:
        raise PowerDiagramWitnessError(
            f"points must have trailing dimension {target.rank}; got shape {z.shape}"
        )
    if not np.isfinite(z).all():
        raise PowerDiagramWitnessError("points contain non-finite values")
    sites = np.ascontiguousarray(target.sites, dtype=np.float32)
    weights = np.ascontiguousarray(target.weights, dtype=np.float32)
    # Elementwise product + np.sum(dtype=float32) over the trailing axis: the
    # declared reduction.  (K,4) fits a single pairwise block, so the order is
    # the sequential 0..rank-1 order of the sealed diagnostic.
    norm = np.sum(sites * sites, axis=-1, dtype=np.float32)
    dot = np.sum(z[..., None, :] * sites, axis=-1, dtype=np.float32)
    return np.asarray(np.float32(2.0) * dot + weights - norm, dtype=np.float32)


def contract_f32_assign(points: np.ndarray, target: PowerDiagramTarget) -> np.ndarray:
    """Class ids under the contract: fp32 scores + deterministic FIRST-max tie rule."""

    scores = contract_f32_power_scores(points, target)
    return np.asarray(target.class_ids)[np.argmax(scores, axis=-1)]
