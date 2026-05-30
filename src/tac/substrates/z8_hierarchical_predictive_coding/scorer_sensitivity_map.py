# SPDX-License-Identifier: MIT
"""Z8 Phase 2 M7 — canonical scorer-sensitivity-map helper.

The empirical scorer-sensitivity map is the Yousfi-grounded primitive M8's
``ScoreAwareLevelLoss.per_level_loss`` consumes as its third argument:

    loss_at_level_i = sum_pixel(
        scorer_sensitivity_at_pixel_at_level_i
        * reconstruction_error_at_pixel
    )

Per the M8 Protocol invariant (``binding_contract.py:467-470``): "implementations
must satisfy: integral over uniform sensitivity (sensitivity_map == 1 everywhere)
reduces to standard L2/L1 reconstruction loss." Path A below is the canonical
implementation of that invariant — every other production path must reduce to
Path A in the limit of uniform sensitivity.

Three honest production paths
-----------------------------

(A) ``uniform_sensitivity_map_for_level`` — trivial all-ones tensor at the
    level's resolution. Always-correct; satisfies the M8 invariant; produces
    Z8 baselines indistinguishable from generic per-level L2 loss. This is
    the L0-equivalent baseline Yousfi would accept as "the codec sees the
    scorer as uniform — your prior is empty."

(B) ``empirical_sensitivity_map_from_slot_ggg`` — `NotImplementedError`
    stub. Reactivation criteria pinned in docstring. The existing Slot GGG
    artifact (commit `32a70c051`, file
    ``experiments/results/slot_rr_canonical_real_video_mlx_macos_cpu_advisory_smoke_20260529T162630Z/smoke_output.json``)
    contains per-mode SegNet-class disagreement summaries (4 strategies ×
    canonical_menu_size) but does NOT contain per-pixel sensitivity tensors
    at PR110-frame resolution. Production path requires either (b1) a
    follow-on probe that emits per-pixel disagreement counts (sister of
    Slot GGG with per-pixel output instead of canonical-menu-size summary),
    or (b2) the canonical Wyner-Ziv decoder-side PoseNet side-information
    canonical equation per CLAUDE.md MEMORY.md task #1496.

(C) ``yousfi_uniward_finite_difference_sensitivity_map`` —
    `NotImplementedError` stub. The canonical Fridrich UNIWARD-analog
    formulation per CLAUDE.md "Fridrich inverse steganalysis": for each
    pixel (i, j), perturb ±ε and re-run real PoseNet + SegNet via canonical
    helpers (``tac.differentiable_eval_roundtrip.load_differentiable_scorers``
    + ``patch_upstream_yuv6_globally`` + ``apply_eval_roundtrip_during_training``
    per CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE"). The map at pixel
    (i, j) is ‖∂score/∂pixel(i,j)‖₂. This requires real scorer load + per-
    pixel forward-backward pass = paid-GPU primitive. Reactivation
    criteria pinned in docstring.

Honest data-domain finding (premise-verified 2026-05-30)
--------------------------------------------------------

The existing ``.omx/state/master_gradient_anchors.jsonl`` ledger contains
per-archive-byte sensitivities at shape (n_bytes, 3) ~178K rows × 3 axes,
not per-pixel at the image plane. The M8 Protocol's per-level
``(B, C, H, W)`` contract is at the image-pyramid plane, not the
archive-byte plane. Therefore the ledger cannot be directly wrapped as
sensitivity-map source — it is the wrong domain. (B) and (C) above are the
only honest production paths; (A) is the baseline that satisfies M8's
L2-reduction invariant.

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": (B)
and (C) are DEFERRED-pending-research-with-reactivation-criteria, NOT
killed. The canonical helper module ships with Path A enabled so M8 can be
designed against the API today, then (B) or (C) lands as a follow-on once
the canonical posterior anchor exists.

Per Catalog #287: every claim above is paired with adjacent source/citation
evidence; no docstring overstatement.
Per Catalog #290: ADOPT — numpy intermediate per Catalog #317. FORK —
Z8-specific Path A baseline (uniform sensitivity at level resolution);
no sister canonical helper builds per-level Yousfi sensitivity maps.
"""

from __future__ import annotations

import enum
from typing import Any

import numpy as np

from tac.substrates.z8_hierarchical_predictive_coding.binding_contract import (
    LevelDimensionContract,
)


__all__ = [
    "ScorerSensitivityMapSource",
    "Z8ScorerSensitivityMap",
    "build_z8_scorer_sensitivity_map_for_level",
    "uniform_sensitivity_map_for_level",
    "empirical_sensitivity_map_from_master_gradient",
    "EmpiricalSensitivityMapNotYetLandedError",
    "ResolutionMismatchAwaitingPerLevelProjectionError",
]


class ScorerSensitivityMapSource(enum.Enum):
    """Four honest production paths per the module docstring.

    Per CLAUDE.md "Forbidden premature KILL": EMPIRICAL_SLOT_GGG and
    FINITE_DIFFERENCE_UNIWARD ANALOG are DEFERRED-pending-research, not
    killed; reactivation criteria pinned in helper docstrings.

    EMPIRICAL_FROM_MASTER_GRADIENT (Path B2) was added 2026-05-30 per
    Phase D Yousfi-ordered cap=1-per-turn (Phase A landed
    ``extract_M_pixel`` + ``broadcast_sensitivity_map_to_channels`` at
    commit ``8a95c9cc5``; this enum value wires that canonical helper
    chain into the M8 Protocol dispatcher).
    """

    UNIFORM = "uniform"  # Path A: trivial all-ones; satisfies M8 invariant
    EMPIRICAL_SLOT_GGG = "empirical_slot_ggg"  # Path B: NotImplementedError stub
    FINITE_DIFFERENCE_UNIWARD = "finite_difference_uniward"  # Path C: stub
    EMPIRICAL_FROM_MASTER_GRADIENT = "empirical_from_master_gradient"  # Path B2


class EmpiricalSensitivityMapNotYetLandedError(NotImplementedError):
    """Path B / Path C are not yet implemented; reactivation criteria pinned.

    Per CLAUDE.md "Forbidden premature KILL without research exhaustion":
    raising this is NOT a paradigm-level kill verdict on Yousfi sensitivity;
    it is an honest IMPLEMENTATION-LEVEL deferral per Catalog #307. The
    paradigm IS the canonical Yousfi UNIWARD-analog finite-difference
    sensitivity map; the implementation requires paid-GPU compute that has
    not yet been allocated.
    """


class ResolutionMismatchAwaitingPerLevelProjectionError(NotImplementedError):
    """Path B2 gradient_tensor shape does not match level.wavelet_subband_shape.

    Per CLAUDE.md "Forbidden premature KILL without research exhaustion" +
    Yousfi-ordered cap=1-per-turn discipline: Phase D (this landing) wires
    ``extract_M_pixel`` + ``broadcast_sensitivity_map_to_channels`` end-to-end
    for the identity-resolution case where the gradient tensor's native
    ``(H, W)`` already matches ``level.wavelet_subband_shape``. The
    per-level resolution-projection step (bilinear downsample for coarser
    levels, identity for matching, raise on upsample) is canonical Phase C
    scope (``decompose_M_contest_per_level``).

    Reactivation criterion (single):

    1. **Phase C ``decompose_M_contest_per_level`` lands** in
       ``tac.master_gradient_comparison.multi_granularity`` (sister cap=1
       cycle after Phase D). The Phase C helper will accept a
       ``PerPixelSensitivityMap`` at native gradient resolution and emit
       per-level-projected maps via the canonical Mallat
       wavelet-hierarchical-prior bilinear downsample.

    Until Phase C lands, this error fires with operator-actionable shapes
    in the message. Callers can EITHER (a) provide a gradient tensor whose
    native ``(H, W)`` matches ``level.wavelet_subband_shape``, OR (b) wait
    for Phase C.
    """


def _validate_level(level: LevelDimensionContract) -> tuple[int, int]:
    """Honest input validation; return the canonical (H, W) for the level."""
    if not isinstance(level, LevelDimensionContract):
        raise TypeError(
            f"level must be LevelDimensionContract, got {type(level).__name__}"
        )
    H, W = level.wavelet_subband_shape
    if H <= 0 or W <= 0:
        raise ValueError(
            f"wavelet_subband_shape must be positive; got ({H}, {W}) at level "
            f"{level.level_index}"
        )
    return H, W


def uniform_sensitivity_map_for_level(
    level: LevelDimensionContract,
    *,
    batch_size: int = 1,
    num_channels: int = 3,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Path A: trivial all-ones sensitivity map at the level's resolution.

    Returns ``(batch_size, num_channels, H, W)`` of dtype ``dtype``, all
    ones. This is the canonical baseline that satisfies the M8 Protocol
    invariant from ``binding_contract.py:467-470``:

        "Implementations must satisfy: integral over uniform sensitivity
         (sensitivity_map == 1 everywhere) reduces to standard L2/L1
         reconstruction loss."

    When M8 multiplies reconstruction error pixel-wise by this map and
    sums, the result equals the standard per-level L2/L1 reconstruction
    loss. This is the L0-equivalent baseline — Yousfi would call it
    "the codec sees the scorer as uniform — your prior is empty."

    Per Catalog #317: numpy is the canonical portable intermediate. The
    map is framework-agnostic; downstream MLX or PyTorch trainers convert
    via their native ``asarray`` / ``from_numpy`` paths.

    Args:
        level: the per-level contract giving the (H, W) shape via
            ``level.wavelet_subband_shape``.
        batch_size: the batch dimension. Defaults to 1 so M8 can broadcast
            over arbitrary batch via standard broadcasting.
        num_channels: image-plane channel count. Defaults to 3 (RGB);
            Z8's per-level reconstruction operates on RGB pyramid levels.
        dtype: numpy dtype. Defaults to ``np.float32`` matching MLX +
            PyTorch default float; use ``np.float64`` for fp64 invariant
            tests.

    Returns:
        Sensitivity-map tensor of shape ``(batch_size, num_channels, H, W)``
        with all entries equal to ``1.0`` (in the given dtype).

    Raises:
        TypeError: ``level`` is not a ``LevelDimensionContract``.
        ValueError: ``level.wavelet_subband_shape`` has non-positive H or W,
            or batch_size / num_channels is non-positive.
    """
    H, W = _validate_level(level)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size}")
    if num_channels <= 0:
        raise ValueError(f"num_channels must be positive; got {num_channels}")
    return np.ones((batch_size, num_channels, H, W), dtype=dtype)


def empirical_sensitivity_map_from_slot_ggg(
    level: LevelDimensionContract,
    *,
    batch_size: int = 1,
    num_channels: int = 3,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Path B: empirical anchor from Slot GGG SegNet-class null-projection.

    NOT YET IMPLEMENTED. The existing Slot GGG artifact at
    ``experiments/results/slot_rr_canonical_real_video_mlx_macos_cpu_advisory_smoke_20260529T162630Z/smoke_output.json``
    contains per-mode metadata (4 strategies × ``canonical_menu_size``
    × 2-element ``frame_resolution_hw``) but does NOT contain per-pixel
    SegNet-class disagreement tensors at PR110-frame resolution.

    Reactivation criteria (any ONE of):

    1. **Sister Slot GGG follow-on probe lands** that emits per-pixel
       disagreement counts at PR110 frame resolution (1164 × 874 × 3)
       AND per-substrate level-projection helper exists at
       ``tac.substrates.z8_hierarchical_predictive_coding.sensitivity_projection.project_per_pixel_to_level``.

    2. **Canonical Wyner-Ziv decoder-side PoseNet side-information
       canonical equation registered** per MEMORY.md task #1496 with a
       per-pixel sensitivity-map projection from the side-info coder.

    3. **Operator-attended paired-CUDA RATIFICATION** of Slot GGG with
       per-pixel SegNet output tensor capture (extends Slot GGG to emit
       ``(1164, 874, 3)`` per-mode SegNet output bytes alongside the
       existing canonical-menu-size summary).

    Until any reactivation criterion holds, raises
    ``EmpiricalSensitivityMapNotYetLandedError``. M7 callers default to
    Path A (uniform) until B is enabled.

    Per Catalog #307 paradigm-vs-implementation classification: this stub
    is IMPLEMENTATION-LEVEL deferred, NOT paradigm-falsified. The Yousfi
    sensitivity-map paradigm is intact; only this specific empirical-
    anchor implementation is pending the data-side prereq.

    Args:
        level: per-level contract (unused at stub stage; pinned for API
            parity with Path A so M7 callers can swap source enum
            without changing signatures).
        batch_size: ignored at stub stage.
        num_channels: ignored at stub stage.
        dtype: ignored at stub stage.

    Raises:
        EmpiricalSensitivityMapNotYetLandedError: always, until at least
            one of the 3 reactivation criteria above is satisfied.
    """
    del level, batch_size, num_channels, dtype  # unused at stub stage
    raise EmpiricalSensitivityMapNotYetLandedError(
        "Path B (empirical_sensitivity_map_from_slot_ggg) is DEFERRED-pending-"
        "research per CLAUDE.md non-negotiable. Slot GGG artifact lacks per-"
        "pixel SegNet-class disagreement tensors at PR110 frame resolution. "
        "See docstring for the 3 reactivation criteria. Use Path A "
        "(UNIFORM) as baseline until Path B is enabled."
    )


def yousfi_uniward_finite_difference_sensitivity_map(
    level: LevelDimensionContract,
    *,
    batch_size: int = 1,
    num_channels: int = 3,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Path C: canonical Yousfi UNIWARD-analog finite-difference sensitivity.

    NOT YET IMPLEMENTED. The canonical formulation per CLAUDE.md "Fridrich
    inverse steganalysis" + "HNeRV / leaderboard-implementation parity
    discipline" lessons 1+6+8:

        sensitivity[i, j] = ‖∂score/∂pixel(i, j)‖₂

    where score = α·d_seg + sqrt(10)·d_pose + β·rate per CLAUDE.md
    "Submission auth eval" canonical formula; d_seg and d_pose come from
    REAL PoseNet + SegNet via
    ``tac.differentiable_eval_roundtrip.load_differentiable_scorers`` per
    CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE"; per-pixel derivative
    computed via finite-difference ±ε perturbation with real scorer
    forward pass at each pixel.

    Reactivation criteria (all must hold):

    1. **Paid-GPU compute allocated** for per-pixel ±ε finite-difference
       sweep. Cost estimate: 1164 × 874 × 3 = ~3M pixels × 2 forward
       passes × 1200 frames = ~7.2B scorer forward passes. At T4 ~10ms
       per scorer forward = ~20K hours. Practical implementation must
       use either (a) sparse sampling + interpolation per Daubechies
       wavelet hierarchical-prior canonical, or (b) gradient
       backpropagation through scorer for analytic per-pixel
       sensitivity (~3 orders of magnitude faster than finite-difference
       but requires ``differentiable_eval_roundtrip`` patch per
       CLAUDE.md "HNeRV parity L8").

    2. **``patch_upstream_yuv6_globally()`` called** before scorer
       construction so PoseNet gradients are not severed per CLAUDE.md
       "HNeRV parity L8" — upstream ``rgb_to_yuv6`` is
       ``@torch.no_grad()`` / in-place by default.

    3. **``eval_roundtrip=True`` active** per CLAUDE.md "eval_roundtrip
       — NON-NEGOTIABLE" so the 384 → 874 → uint8 → 384 path is
       differentiably simulated; otherwise pose-axis sensitivity has
       2-11× proxy-auth gap.

    Until ALL 3 reactivation criteria hold, raises
    ``EmpiricalSensitivityMapNotYetLandedError``.

    Per Catalog #307: IMPLEMENTATION-LEVEL deferred. The Yousfi
    UNIWARD-analog paradigm is intact; only this specific paid-GPU
    finite-difference implementation is pending the compute-side prereq.

    Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-
    negotiable: any Path C empirical anchor MUST come from real
    PoseNet + SegNet on 1:1 contest-compliant hardware; macOS-MLX
    advisory smoke is permissible for the development-loop signal but
    NEVER promotable per Catalog #192.

    Args:
        level, batch_size, num_channels, dtype: pinned for API parity;
            ignored at stub stage.

    Raises:
        EmpiricalSensitivityMapNotYetLandedError: always, until all 3
            reactivation criteria above are satisfied.
    """
    del level, batch_size, num_channels, dtype  # unused at stub stage
    raise EmpiricalSensitivityMapNotYetLandedError(
        "Path C (yousfi_uniward_finite_difference_sensitivity_map) is "
        "DEFERRED-pending-paid-GPU per CLAUDE.md non-negotiable. Requires "
        "(1) paid-GPU compute allocation, (2) patch_upstream_yuv6_globally() "
        "active per HNeRV parity L8, (3) eval_roundtrip=True per CLAUDE.md "
        "'eval_roundtrip — NON-NEGOTIABLE'. See docstring for full criteria. "
        "Use Path A (UNIFORM) as baseline until Path C is enabled."
    )


def empirical_sensitivity_map_from_master_gradient(
    level: LevelDimensionContract,
    gradient_tensor: Any,
    *,
    reduction: str = "l2_norm",
    batch_size: int = 1,
    num_channels: int = 3,
    dtype: Any = np.float32,
    pair_aggregation: str = "first",
) -> np.ndarray:
    """Path B2: empirical sensitivity from canonical master-gradient tensor.

    Phase D Yousfi-ordered wire-in (2026-05-30): consumes Phase A's canonical
    ``tac.master_gradient_comparison.extract_M_pixel`` + sister
    ``broadcast_sensitivity_map_to_channels`` to produce M8-compatible
    ``(B, C, H, W)`` tensors from a ``ContestGradientTensor`` or
    ``InflatedGradientTensor``.

    Per the M8 Protocol docstring at ``binding_contract.py`` (verbatim):

        "The map itself comes from ``tac.master_gradient`` empirical
         per-pixel sensitivity measurement (cross-substrate per Slot AA +
         sister work)."

    This helper IS that wire-in. Three honest steps:

    1. **Extract per-pixel scorer sensitivity** via
       ``extract_M_pixel(gradient_tensor, reduction=reduction)`` — reduces
       over the scorer-axis dimension (seg, pose, archive_bytes) per the
       canonical Fridrich UNIWARD scorer-blindness inverse interpretation
       (L2: ``sqrt(sum_axes(M[:, axes, h, w]**2))``).

    2. **Broadcast to (B, C, H, W)** via
       ``broadcast_sensitivity_map_to_channels(m_pixel, batch_size=...,
       num_channels=..., pair_aggregation=...)`` — channel-uniformity
       invariant preserves Fridrich per-spatial-location cost convention.

    3. **Identity-resolution invariant check**: assert the gradient
       tensor's native ``(H, W)`` matches ``level.wavelet_subband_shape``.
       Mismatched cases raise
       ``ResolutionMismatchAwaitingPerLevelProjectionError`` with
       operator-actionable reactivation criterion (Phase C
       ``decompose_M_contest_per_level``).

    Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192 + Catalog #317:
    the gradient tensor's evidence-grade tag (``measurement_axis``) is
    inherited from the source ``PerPixelSensitivityMap`` sidecar
    ``.meta.json`` (per Catalog #323 canonical Provenance). For
    ``[predicted]``-tagged input the output is non-promotable; for
    ``[contest-CUDA T4]``-tagged input the output carries the canonical
    paid-anchor provenance. This helper does NOT mutate provenance; the
    sidecar trail through ``extract_M_pixel`` is the canonical record.

    Args:
        level: per-level contract giving the (H, W) shape via
            ``level.wavelet_subband_shape``.
        gradient_tensor: ``ContestGradientTensor`` or
            ``InflatedGradientTensor`` from
            ``tac.master_gradient_comparison.multi_granularity``. Native
            shape ``(N_pairs, n_axes, H, W)``; native ``(H, W)`` MUST
            match ``level.wavelet_subband_shape`` (Phase C unblocks
            mismatched-resolution cases).
        reduction: one of ``LEGAL_PIXEL_REDUCTIONS`` (``l2_norm`` /
            ``l1_norm`` / ``max``). Defaults to ``l2_norm`` per Fridrich
            UNIWARD canonical formulation.
        batch_size: ``B`` dimension for M8 output. Defaults to 1.
        num_channels: ``C`` dimension for M8 output. Defaults to 3 (RGB).
        dtype: output numpy dtype. Defaults to ``np.float32``.
        pair_aggregation: how to reduce the ``N_pairs`` dimension to a
            single map for the level. ``first`` (default) takes pair 0
            (canonical for single-pair M8 callsites). ``mean`` averages
            across pairs. ``max`` takes per-pixel max — useful when M8
            wants the conservative dominant-pair bound. Forwarded to
            ``broadcast_sensitivity_map_to_channels``.

    Returns:
        Sensitivity-map tensor of shape ``(batch_size, num_channels, H, W)``
        of the given dtype. All entries non-negative. Channel-uniform per
        Fridrich per-spatial-location cost convention.

    Raises:
        TypeError: ``level`` is not a ``LevelDimensionContract``, OR
            ``gradient_tensor`` is not a recognized
            ``ContestGradientTensor`` / ``InflatedGradientTensor``.
        ValueError: ``reduction`` is not in ``LEGAL_PIXEL_REDUCTIONS``,
            OR ``batch_size`` / ``num_channels`` is non-positive.
        ResolutionMismatchAwaitingPerLevelProjectionError: gradient
            tensor's native ``(H, W)`` does not match
            ``level.wavelet_subband_shape``. Phase C is the canonical
            unblocker per the error's docstring.
    """
    # Lazy import to avoid module-level coupling; Phase A canonical helper.
    from tac.master_gradient_comparison.multi_granularity import (
        broadcast_sensitivity_map_to_channels,
        extract_M_pixel,
    )

    H_level, W_level = _validate_level(level)

    # Step 1: extract per-pixel sensitivity (axes-reduced per `reduction`).
    # extract_M_pixel handles its own input-type validation + reduction
    # validation; we surface those errors unchanged.
    m_pixel = extract_M_pixel(gradient_tensor, reduction=reduction)

    # Step 3 (early-fail): identity-resolution invariant. Phase D
    # ONLY supports the matched-resolution case; resolution projection
    # is canonical Phase C scope.
    n_pairs, H_grad, W_grad = m_pixel.shape()
    if (H_grad, W_grad) != (H_level, W_level):
        raise ResolutionMismatchAwaitingPerLevelProjectionError(
            f"gradient_tensor native shape ({H_grad}, {W_grad}) does not "
            f"match level.wavelet_subband_shape ({H_level}, {W_level}) at "
            f"level_index={level.level_index}. Phase D wires the canonical "
            f"identity-resolution path only; per-level resolution projection "
            f"is canonical Phase C scope (decompose_M_contest_per_level "
            f"in tac.master_gradient_comparison.multi_granularity). EITHER "
            f"(a) provide a gradient tensor whose native (H, W) matches "
            f"level.wavelet_subband_shape, OR (b) wait for Phase C."
        )

    # Step 2: broadcast (N_pairs, H, W) -> (B, C, H, W) per M8 contract.
    # broadcast_sensitivity_map_to_channels handles aggregation +
    # channel/batch shaping + dtype.
    return broadcast_sensitivity_map_to_channels(
        m_pixel,
        batch_size=batch_size,
        num_channels=num_channels,
        pair_aggregation=pair_aggregation,
        dtype=dtype,
    )


class Z8ScorerSensitivityMap:
    """Canonical M7 source dispatcher consumed by M8's per_level_loss.

    Per the M8 Protocol contract in ``binding_contract.py:419-472``: M8's
    ``per_level_loss(reconstruction, target, scorer_sensitivity_map)``
    accepts ``scorer_sensitivity_map: Any`` of shape ``(B, C, H, W)`` or
    broadcast-compatible. This dispatcher produces the correctly-shaped
    tensor per the bound source and level.

    Single-responsibility class — given a level and a source enum, the
    ``get_for_level`` method returns the canonical sensitivity tensor.
    M8's trainer holds one ``Z8ScorerSensitivityMap`` instance per
    hierarchy and queries it per-level per-step.
    """

    def __init__(
        self,
        source: ScorerSensitivityMapSource = ScorerSensitivityMapSource.UNIFORM,
    ) -> None:
        if not isinstance(source, ScorerSensitivityMapSource):
            raise TypeError(
                f"source must be ScorerSensitivityMapSource enum member; "
                f"got {type(source).__name__}"
            )
        self._source = source

    @property
    def source(self) -> ScorerSensitivityMapSource:
        return self._source

    def get_for_level(
        self,
        level: LevelDimensionContract,
        *,
        batch_size: int = 1,
        num_channels: int = 3,
        dtype: Any = np.float32,
        gradient_tensor: Any = None,
        reduction: str = "l2_norm",
        pair_aggregation: str = "first",
    ) -> np.ndarray:
        """Return the sensitivity tensor for ``level`` per the bound source.

        Dispatches to ``uniform_sensitivity_map_for_level`` (Path A) when
        ``source == UNIFORM``; to ``empirical_sensitivity_map_from_master_gradient``
        (Path B2) when ``source == EMPIRICAL_FROM_MASTER_GRADIENT`` (requires
        ``gradient_tensor``); to the Path B / Path C stubs otherwise. Stubs
        raise ``EmpiricalSensitivityMapNotYetLandedError`` until their
        per-helper reactivation criteria are satisfied.

        Args:
            level, batch_size, num_channels, dtype: per-level + shape
                contract; forwarded to the path-specific helper.
            gradient_tensor: REQUIRED when ``source ==
                EMPIRICAL_FROM_MASTER_GRADIENT`` (Path B2);
                ``ContestGradientTensor`` or ``InflatedGradientTensor``.
                Ignored for other paths.
            reduction: ``l2_norm`` (default) / ``l1_norm`` / ``max``;
                forwarded to ``extract_M_pixel`` when source is Path B2.
            pair_aggregation: ``first`` (default) / ``mean`` / ``max``;
                forwarded to ``broadcast_sensitivity_map_to_channels``
                when source is Path B2.

        Raises:
            ValueError: ``source == EMPIRICAL_FROM_MASTER_GRADIENT`` AND
                ``gradient_tensor is None``.
            ResolutionMismatchAwaitingPerLevelProjectionError: Path B2
                gradient tensor's native ``(H, W)`` does not match
                ``level.wavelet_subband_shape``.
        """
        if self._source is ScorerSensitivityMapSource.UNIFORM:
            return uniform_sensitivity_map_for_level(
                level,
                batch_size=batch_size,
                num_channels=num_channels,
                dtype=dtype,
            )
        if self._source is ScorerSensitivityMapSource.EMPIRICAL_FROM_MASTER_GRADIENT:
            if gradient_tensor is None:
                raise ValueError(
                    "source == EMPIRICAL_FROM_MASTER_GRADIENT requires a "
                    "non-None gradient_tensor (ContestGradientTensor or "
                    "InflatedGradientTensor from "
                    "tac.master_gradient_comparison.multi_granularity). "
                    "EITHER (a) pass gradient_tensor=..., OR (b) bind "
                    "source=ScorerSensitivityMapSource.UNIFORM for the "
                    "Path A baseline."
                )
            return empirical_sensitivity_map_from_master_gradient(
                level,
                gradient_tensor,
                reduction=reduction,
                batch_size=batch_size,
                num_channels=num_channels,
                dtype=dtype,
                pair_aggregation=pair_aggregation,
            )
        if self._source is ScorerSensitivityMapSource.EMPIRICAL_SLOT_GGG:
            return empirical_sensitivity_map_from_slot_ggg(
                level,
                batch_size=batch_size,
                num_channels=num_channels,
                dtype=dtype,
            )
        if self._source is ScorerSensitivityMapSource.FINITE_DIFFERENCE_UNIWARD:
            return yousfi_uniward_finite_difference_sensitivity_map(
                level,
                batch_size=batch_size,
                num_channels=num_channels,
                dtype=dtype,
            )
        # Exhaustiveness check; unreachable if enum is well-formed.
        raise RuntimeError(f"unhandled ScorerSensitivityMapSource: {self._source!r}")


def build_z8_scorer_sensitivity_map_for_level(
    level: LevelDimensionContract,
    *,
    source: ScorerSensitivityMapSource = ScorerSensitivityMapSource.UNIFORM,
    batch_size: int = 1,
    num_channels: int = 3,
    dtype: Any = np.float32,
    gradient_tensor: Any = None,
    reduction: str = "l2_norm",
    pair_aggregation: str = "first",
) -> np.ndarray:
    """Single-call canonical builder for M8 trainer callsites.

    Convenience wrapper around ``Z8ScorerSensitivityMap(source).get_for_level(level, ...)``
    for the common case where M8's trainer wants the canonical sensitivity
    tensor for a single level without holding a dispatcher instance.

    The ``gradient_tensor`` / ``reduction`` / ``pair_aggregation`` kwargs
    are forwarded through to Path B2 (``EMPIRICAL_FROM_MASTER_GRADIENT``);
    ignored for other paths.
    """
    return Z8ScorerSensitivityMap(source).get_for_level(
        level,
        batch_size=batch_size,
        num_channels=num_channels,
        dtype=dtype,
        gradient_tensor=gradient_tensor,
        reduction=reduction,
        pair_aggregation=pair_aggregation,
    )
