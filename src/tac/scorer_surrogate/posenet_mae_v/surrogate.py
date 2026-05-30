# SPDX-License-Identifier: MIT
"""PoseNet MAE-V (Multi-Axis Variant) numpy-portable surrogate.

See ``__init__.py`` for the public API + design rationale + canonical
apparatus mutation chain.

The numpy forward replicates the canonical
``tac.substrates.hinton_distilled_scorer_surrogate.LearnablePoseStudentHead``
computation exactly:

1. Per frame: average-pool ``(B, H, W, 3)`` -> ``(B, grid, grid, 3)``.
2. Flatten each frame's pooled features -> ``(B, grid*grid*3)``.
3. Concatenate both frames -> ``(B, 2*grid*grid*3)``.
4. Linear projection: ``feature @ weight + bias`` -> ``(B, pose_dims)``.

For canonical ``grid=4`` + ``pose_dims=6``: feature_dim = 96, total
params = 96 * 6 + 6 = 582 — the canonical 582-param cheap surrogate.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


CANONICAL_POSE_DIMS = 6
CANONICAL_POSE_POOL_GRID = 4

# Catalog #1265 drift-discipline ULP threshold per Slot 1303 T3 GRAND COUNCIL.
PARITY_MAX_ABS_CANONICAL_THRESHOLD = 3e-5

_SURROGATE_AXIS_TAG = "[macOS-CPU advisory]"
_SURROGATE_EVIDENCE_GRADE = "predicted"


class PoseNetMaeVSurrogateInvalidError(ValueError):
    """Raised on weight-shape / dtype / pool_grid mismatch."""


@dataclass(frozen=True)
class PoseJacobianResult:
    """Typed per-byte pose-Jacobian result with canonical-routing markers.

    Catalog #341 Tier A canonical-routing markers are baked in:
    ``score_claim=False`` + ``promotable=False`` + ``axis_tag`` =
    ``[macOS-CPU advisory]``. Downstream consumers MUST NOT promote
    these values to contest-axis claims; they are cost-discrimination
    priors only.

    The Jacobian is per-byte: index ``i`` holds the surrogate-estimated
    ``d(pose_distortion_proxy)/d(byte_i)`` magnitude (always >= 0 since
    we report ``|finite_difference|``).
    """

    per_byte_pose_jacobian_magnitude: tuple[float, ...]
    n_bytes_probed: int
    surrogate_weight_sha256: str
    axis_tag: str
    score_claim: bool
    promotable: bool
    evidence_grade: str
    measurement_utc: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.score_claim is not False:
            raise ValueError(
                "PoseJacobianResult.score_claim MUST be False per Catalog #341"
            )
        if self.promotable is not False:
            raise ValueError(
                "PoseJacobianResult.promotable MUST be False per Catalog #341"
            )
        if self.axis_tag != _SURROGATE_AXIS_TAG:
            raise ValueError(
                f"axis_tag must be {_SURROGATE_AXIS_TAG!r}, "
                f"got {self.axis_tag!r}"
            )
        if self.evidence_grade != _SURROGATE_EVIDENCE_GRADE:
            raise ValueError(
                f"evidence_grade must be {_SURROGATE_EVIDENCE_GRADE!r}, "
                f"got {self.evidence_grade!r}"
            )
        if len(self.per_byte_pose_jacobian_magnitude) != self.n_bytes_probed:
            raise ValueError(
                "per_byte_pose_jacobian_magnitude length must equal n_bytes_probed "
                f"({len(self.per_byte_pose_jacobian_magnitude)} vs "
                f"{self.n_bytes_probed})"
            )
        if any(v < 0.0 for v in self.per_byte_pose_jacobian_magnitude):
            raise ValueError(
                "per_byte_pose_jacobian_magnitude must be all-nonneg "
                "(we report |finite_difference|)"
            )
        if (
            len(self.surrogate_weight_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in self.surrogate_weight_sha256)
        ):
            raise ValueError("surrogate_weight_sha256 must be 64-char lowercase hex")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_of_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    # Include shape + dtype so weight tensors with same bytes but different
    # interpretation produce distinct sha256.
    h.update(repr((arr.shape, str(arr.dtype))).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _sha256_of_weights(weight: np.ndarray, bias: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(_sha256_of_array(weight).encode("ascii"))
    h.update(_sha256_of_array(bias).encode("ascii"))
    return h.hexdigest()


def build_canonical_provenance_for_surrogate(
    *,
    surrogate_weight_sha256: str,
    n_bytes_probed: int,
    measurement_utc: str,
    canonical_helper: str,
) -> dict[str, Any]:
    """Build canonical Provenance per Catalog #323 for surrogate outputs."""
    return {
        "schema_version": "provenance_v1",
        "kind": "predicted_from_model",
        "model_identifier": canonical_helper,
        "surrogate_weight_sha256": surrogate_weight_sha256,
        "n_bytes_probed": int(n_bytes_probed),
        "axis_tag": _SURROGATE_AXIS_TAG,
        "evidence_grade": _SURROGATE_EVIDENCE_GRADE,
        "score_claim": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "captured_at_utc": measurement_utc,
        "canonical_helper_invocation": canonical_helper,
    }


def _block_average_pool_2d(arr: np.ndarray, grid: int) -> np.ndarray:
    """Equivalent of MLX block-averaging pool used by ``LearnablePoseStudentHead``.

    Input shape ``(B, H, W, C)``; output shape ``(B, grid, grid, C)``.

    The canonical MLX head uses ``mx.mean(x.reshape(B, grid, H//grid, grid, W//grid, C), axis=(2, 4))``
    — average over each H/grid x W/grid block. Replicated here exactly for
    parity with the canonical MLX forward.
    """
    if arr.ndim != 4:
        raise PoseNetMaeVSurrogateInvalidError(
            f"expected (B, H, W, C) 4D array, got shape {arr.shape}"
        )
    b, h, w, c = arr.shape
    if h % grid != 0 or w % grid != 0:
        raise PoseNetMaeVSurrogateInvalidError(
            f"H ({h}) and W ({w}) must be divisible by grid ({grid})"
        )
    reshaped = arr.reshape(b, grid, h // grid, grid, w // grid, c)
    return reshaped.mean(axis=(2, 4))


@dataclass(frozen=True)
class PoseNetMaeVSurrogate:
    """Numpy-portable PoseNet surrogate (canonical 6-dim ego-motion).

    Attributes:
        weight: numpy float32 ``(feature_dim, pose_dims)`` where
            ``feature_dim = 2 * pool_grid * pool_grid * 3`` (96 for the
            canonical grid=4 / pose_dims=6 default).
        bias: numpy float32 ``(pose_dims,)``.
        pose_dims: number of pose dims (canonical 6 = the first 6 of the
            12-dim FastViT-T12 pose head per upstream/modules.py).
        pool_grid: coarse pool resolution per frame (canonical 4 = 4x4).
        weight_sha256: canonical sha256 of ``(weight, bias)`` for
            provenance + drift detection.
    """

    weight: np.ndarray
    bias: np.ndarray
    pose_dims: int = CANONICAL_POSE_DIMS
    pool_grid: int = CANONICAL_POSE_POOL_GRID
    weight_sha256: str = ""

    def __post_init__(self) -> None:
        # Validate shapes + dtype
        if self.pose_dims < 1:
            raise PoseNetMaeVSurrogateInvalidError(
                f"pose_dims must be >= 1, got {self.pose_dims}"
            )
        if self.pool_grid < 1:
            raise PoseNetMaeVSurrogateInvalidError(
                f"pool_grid must be >= 1, got {self.pool_grid}"
            )
        expected_feature_dim = 2 * self.pool_grid * self.pool_grid * 3
        if self.weight.shape != (expected_feature_dim, self.pose_dims):
            raise PoseNetMaeVSurrogateInvalidError(
                f"weight shape {self.weight.shape} != "
                f"({expected_feature_dim}, {self.pose_dims})"
            )
        if self.bias.shape != (self.pose_dims,):
            raise PoseNetMaeVSurrogateInvalidError(
                f"bias shape {self.bias.shape} != ({self.pose_dims},)"
            )
        if self.weight.dtype not in (np.float32, np.float64):
            raise PoseNetMaeVSurrogateInvalidError(
                f"weight dtype must be float32/float64, got {self.weight.dtype}"
            )
        if self.bias.dtype not in (np.float32, np.float64):
            raise PoseNetMaeVSurrogateInvalidError(
                f"bias dtype must be float32/float64, got {self.bias.dtype}"
            )
        # Backfill weight_sha256 if caller didn't pre-compute
        if not self.weight_sha256:
            object.__setattr__(
                self,
                "weight_sha256",
                _sha256_of_weights(self.weight, self.bias),
            )
        if (
            len(self.weight_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in self.weight_sha256)
        ):
            raise PoseNetMaeVSurrogateInvalidError(
                "weight_sha256 must be 64-char lowercase hex"
            )

    @property
    def feature_dim(self) -> int:
        return 2 * self.pool_grid * self.pool_grid * 3

    @property
    def total_params(self) -> int:
        return self.feature_dim * self.pose_dims + self.pose_dims

    def forward(
        self,
        rgb_0_bhwc: np.ndarray,
        rgb_1_bhwc: np.ndarray,
    ) -> np.ndarray:
        """Map a decoded frame pair to a (B, pose_dims) pose vector.

        Args:
            rgb_0_bhwc: decoded frame 0 ``(B, H, W, 3)`` in ``[0, 1]``.
            rgb_1_bhwc: decoded frame 1 ``(B, H, W, 3)`` in ``[0, 1]``.

        Returns:
            numpy ``(B, pose_dims)`` predicted pose vector.

        Raises:
            PoseNetMaeVSurrogateInvalidError: on shape mismatch.
        """
        if rgb_0_bhwc.shape != rgb_1_bhwc.shape:
            raise PoseNetMaeVSurrogateInvalidError(
                "rgb_0 and rgb_1 must have identical shape; "
                f"got {rgb_0_bhwc.shape} vs {rgb_1_bhwc.shape}"
            )
        if rgb_0_bhwc.ndim != 4 or rgb_0_bhwc.shape[-1] != 3:
            raise PoseNetMaeVSurrogateInvalidError(
                f"rgb_0 must be (B, H, W, 3); got {rgb_0_bhwc.shape}"
            )
        f0 = _block_average_pool_2d(rgb_0_bhwc, self.pool_grid)
        f1 = _block_average_pool_2d(rgb_1_bhwc, self.pool_grid)
        b = f0.shape[0]
        feat = np.concatenate(
            [f0.reshape(b, -1), f1.reshape(b, -1)], axis=-1
        )
        return feat @ self.weight + self.bias


def build_surrogate_from_numpy_weights(
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    pose_dims: int = CANONICAL_POSE_DIMS,
    pool_grid: int = CANONICAL_POSE_POOL_GRID,
) -> PoseNetMaeVSurrogate:
    """Convenience constructor — copies arrays + freezes the dataclass."""
    w = np.asarray(weight, dtype=np.float32).copy()
    b = np.asarray(bias, dtype=np.float32).copy()
    return PoseNetMaeVSurrogate(
        weight=w,
        bias=b,
        pose_dims=pose_dims,
        pool_grid=pool_grid,
    )


def compute_forward_parity_max_abs(
    surrogate: PoseNetMaeVSurrogate,
    canonical_forward: Callable[[np.ndarray, np.ndarray], np.ndarray],
    rgb_0_bhwc: np.ndarray,
    rgb_1_bhwc: np.ndarray,
) -> float:
    """Compute max-abs drift between surrogate forward + canonical forward.

    Catalog #1265 sister: max_abs < ``PARITY_MAX_ABS_CANONICAL_THRESHOLD``
    (3e-5) is the canonical drift-discipline threshold per Slot 1303 T3
    GRAND COUNCIL fp32 ULP boundary.

    Args:
        surrogate: this package's numpy-portable surrogate.
        canonical_forward: a callable replicating the canonical
            ``LearnablePoseStudentHead`` forward in another backend (MLX,
            PyTorch, etc.). Caller supplies it; we do not import MLX
            here so the surrogate package stays numpy-portable.
        rgb_0_bhwc, rgb_1_bhwc: input frame pair.

    Returns:
        Scalar max-abs difference (per-element across the (B, pose_dims)
        outputs).
    """
    out_surr = surrogate.forward(rgb_0_bhwc, rgb_1_bhwc)
    out_canon = canonical_forward(rgb_0_bhwc, rgb_1_bhwc)
    return float(np.max(np.abs(out_surr - np.asarray(out_canon))))


def compute_per_byte_pose_jacobian(
    surrogate: PoseNetMaeVSurrogate,
    *,
    archive_bytes: bytes,
    decoder: Callable[[bytes], tuple[np.ndarray, np.ndarray]],
    byte_indices: tuple[int, ...] | None = None,
    perturbation_magnitude: float = 1.0,
    teacher_pose: np.ndarray | None = None,
) -> PoseJacobianResult:
    """Compute per-byte ``|d(pose_distortion_proxy)/d(byte_i)|`` via FD.

    Per-byte finite-difference Jacobian extraction over a CHEAP surrogate
    forward (the 96-feature pool + linear projection costs O(N_pairs)
    multiplications per evaluation; per-byte FD is O(N_bytes_probed *
    N_pairs * 96 * 6) = tractable even on macOS-CPU advisory hardware).

    The "pose distortion proxy" is the MSE between the surrogate's
    predicted pose and either (a) the caller-supplied ``teacher_pose``
    when given (canonical Yousfi-distortion-cost-discriminator) or (b)
    the per-byte BASELINE surrogate output when ``teacher_pose is None``
    (self-referential FD, useful for detecting which bytes the surrogate
    is sensitive to in the absence of teacher pose).

    Args:
        surrogate: trained :class:`PoseNetMaeVSurrogate`.
        archive_bytes: the archive byte sequence to probe.
        decoder: a callable ``bytes -> (rgb_0_bhwc, rgb_1_bhwc)`` that
            decodes the archive bytes into a frame pair (each
            ``(B, H, W, 3)`` in ``[0, 1]``). Caller-supplied to keep
            this package decoder-agnostic.
        byte_indices: tuple of byte indices to probe. ``None`` defaults
            to ``tuple(range(len(archive_bytes)))`` (probe all bytes;
            expensive for large archives).
        perturbation_magnitude: byte-level perturbation (default 1.0 =
            single-LSB flip, capped at 255 + saturated at 0). Larger
            magnitudes amortize numerical noise but reduce locality of
            the gradient estimate.
        teacher_pose: optional canonical PoseNet teacher pose
            ``(B, pose_dims)`` for the canonical pair; ``None`` falls
            back to self-referential FD (sensitivity, not vulnerability).

    Returns:
        :class:`PoseJacobianResult` with per-byte Jacobian magnitudes +
        canonical-routing markers.

    Raises:
        ValueError: on inconsistent input shapes.
    """
    if not archive_bytes:
        raise ValueError("archive_bytes must be nonempty")
    if perturbation_magnitude == 0.0:
        raise ValueError("perturbation_magnitude must be nonzero")

    probe_indices = (
        tuple(byte_indices) if byte_indices is not None else tuple(range(len(archive_bytes)))
    )
    if any(not (0 <= i < len(archive_bytes)) for i in probe_indices):
        raise ValueError("byte_indices contain values out of range")

    # Baseline forward
    rgb_0_base, rgb_1_base = decoder(archive_bytes)
    pose_base = surrogate.forward(rgb_0_base, rgb_1_base)
    if teacher_pose is not None and teacher_pose.shape != pose_base.shape:
        raise ValueError(
            f"teacher_pose shape {teacher_pose.shape} != "
            f"baseline pose shape {pose_base.shape}"
        )

    def _proxy(pose: np.ndarray) -> float:
        if teacher_pose is not None:
            diff = pose - teacher_pose
            return float(np.mean(diff * diff))
        # Self-referential: |pose|^2 (sensitivity / response magnitude)
        return float(np.mean(pose * pose))

    base_proxy = _proxy(pose_base)
    delta = float(perturbation_magnitude)

    # Per-byte FD
    jac_magnitudes: list[float] = []
    mutable = bytearray(archive_bytes)
    for i in probe_indices:
        original = mutable[i]
        # Saturated single-byte perturbation in [0, 255]
        new_value = original + int(delta)
        if new_value > 255:
            new_value = original - int(abs(delta))
        if new_value < 0:
            new_value = original + int(abs(delta))
        new_value = max(0, min(255, new_value))
        if new_value == original:
            # Cannot perturb this byte (degenerate at boundary)
            jac_magnitudes.append(0.0)
            continue
        mutable[i] = new_value
        try:
            rgb_0_p, rgb_1_p = decoder(bytes(mutable))
            pose_p = surrogate.forward(rgb_0_p, rgb_1_p)
            proxy_p = _proxy(pose_p)
        finally:
            mutable[i] = original
        # |finite difference|
        jac_magnitudes.append(abs(proxy_p - base_proxy) / abs(new_value - original))

    utc = _utc_now()
    sha = surrogate.weight_sha256
    provenance = build_canonical_provenance_for_surrogate(
        surrogate_weight_sha256=sha,
        n_bytes_probed=len(probe_indices),
        measurement_utc=utc,
        canonical_helper=(
            "tac.scorer_surrogate.posenet_mae_v.compute_per_byte_pose_jacobian"
        ),
    )
    return PoseJacobianResult(
        per_byte_pose_jacobian_magnitude=tuple(float(v) for v in jac_magnitudes),
        n_bytes_probed=len(probe_indices),
        surrogate_weight_sha256=sha,
        axis_tag=_SURROGATE_AXIS_TAG,
        score_claim=False,
        promotable=False,
        evidence_grade=_SURROGATE_EVIDENCE_GRADE,
        measurement_utc=utc,
        provenance=provenance,
    )
