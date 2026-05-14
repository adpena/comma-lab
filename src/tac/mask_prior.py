# SPDX-License-Identifier: MIT
"""Mask class-frequency prior (Lane MOS).

Per the Cosmos research synthesis (memory: project_cosmos_mae_lyra_telescope_synthesis_20260428
and project_outstanding_work_and_stacks_20260428), Lane MOS adds a class-frequency
prior to the renderer's mask reconstruction so the argmax decision is biased
toward the empirical 5-class distribution observed in the GT masks.

The prior is a small 3-D tensor stored as ``prior.npz`` inside ``archive.zip``:
    prior.shape == (num_classes, prior_h, prior_w)
    prior.dtype == np.float16
    prior.sum(axis=0) == 1.0      # per-cell class probability simplex

At inflate time, the renderer/postfilter calls :func:`apply_prior_weighting`
to add ``alpha * log(prior)`` to its predicted logits — a SOFT bias, not a
hard projection. This costs ~20 bytes for typical low-resolution priors
(e.g. 5×6×8 fp16 = 480 bytes; 5×4×5 fp16 = 200 bytes — both well under the
rate-limit budget for a Quantizr-class archive).

Per Yousfi SegNet (upstream/modules.py): ``NUM_CLASSES = 5``. The class layout
matches ``tac.semantic_label_contract.CONTEST_SEGNET_CLASS_NAMES``:
    0 = road, 1 = lane_markings, 2 = undrivable, 3 = movable, 4 = my_car.

Functions
---------
- :func:`load_prior` — read prior.npz, validate shape/version, return numpy array.
- :func:`apply_prior_weighting` — add alpha * log(prior) to (B, C, H, W) logits.
- :func:`save_prior_to_archive` — append prior.npz to an existing archive.zip.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tac.camera import NUM_CLASSES

__all__ = [
    "PRIOR_FILENAME",
    "PRIOR_VERSION",
    "PRIOR_DTYPE",
    "load_prior",
    "apply_prior_weighting",
    "save_prior_to_archive",
]

#: Canonical filename inside archive.zip.
PRIOR_FILENAME = "prior.npz"

#: Schema version stored in the npz; bump on incompatible layout changes.
PRIOR_VERSION = 1

#: On-disk dtype for the class-probability tensor. fp16 keeps the prior
#: tiny without measurably affecting the soft bias (alpha ≤ 0.5 typical).
PRIOR_DTYPE = np.float16

# Numerical floor used when computing log(prior). The prior is normalized
# to sum to 1 across classes per spatial cell, but soft-renormalization +
# fp16 storage can yield exact zeros in unrepresented cells; clamp before log.
_EPS = 1e-6


def _coerce_prior_to_numpy(prior: np.ndarray | "torch.Tensor") -> np.ndarray:
    """Convert prior to numpy array (handles both numpy and torch inputs).

    The tests exercise both code paths: ``apply_prior_weighting(logits, prior, ...)``
    where ``prior`` is the in-memory numpy array returned by
    ``build_mask_class_prior`` AND the post-roundtrip numpy array returned by
    :func:`load_prior`.
    """
    if isinstance(prior, np.ndarray):
        return prior
    if isinstance(prior, torch.Tensor):
        return prior.detach().cpu().numpy()
    raise TypeError(
        f"prior must be numpy.ndarray or torch.Tensor, got {type(prior).__name__}"
    )


def _validate_prior_shape(prior: np.ndarray) -> None:
    """Ensure prior has shape ``(NUM_CLASSES, H, W)`` with H, W >= 1."""
    if prior.ndim != 3:
        raise ValueError(
            f"Mask class prior must be 3-D (num_classes, H, W); got shape {prior.shape}"
        )
    if prior.shape[0] != NUM_CLASSES:
        raise ValueError(
            f"Mask class prior must have shape (num_classes={NUM_CLASSES}, H, W); "
            f"got shape {prior.shape} (axis-0 = {prior.shape[0]})"
        )
    if prior.shape[1] < 1 or prior.shape[2] < 1:
        raise ValueError(
            f"Mask class prior spatial dims must be >= 1; got shape {prior.shape}"
        )


def load_prior(npz_path: str | Path) -> np.ndarray:
    """Load a mask-class prior from ``prior.npz`` and validate its layout.

    Args:
        npz_path: filesystem path or open file-like object pointing at a
            ``prior.npz`` produced by :func:`experiments.build_mask_class_prior.write_prior_npz`.

    Returns:
        ``(NUM_CLASSES, H, W)`` numpy array, dtype ``float16``.

    Raises:
        ValueError: if the file's ``prior`` array has the wrong axis-0 length,
            wrong rank, or empty spatial dims; or if the schema version does
            not match :data:`PRIOR_VERSION`.
        FileNotFoundError: if ``npz_path`` does not exist.
    """
    if isinstance(npz_path, (str, Path)):
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Mask prior not found: {npz_path}")

    data = np.load(npz_path)

    if "prior" not in data.files:
        raise ValueError(
            f"prior.npz missing 'prior' array; keys: {list(data.files)!r}"
        )
    prior = np.asarray(data["prior"])

    if "version" in data.files:
        version = int(np.asarray(data["version"]).item())
        if version != PRIOR_VERSION:
            raise ValueError(
                f"prior.npz version mismatch: file={version}, "
                f"expected={PRIOR_VERSION}. Rebuild with the current "
                "experiments/build_mask_class_prior.py."
            )

    _validate_prior_shape(prior)
    return prior


def apply_prior_weighting(
    logits: torch.Tensor,
    prior: np.ndarray | "torch.Tensor",
    alpha: float = 0.1,
) -> torch.Tensor:
    """Apply a soft class-prior bias to per-pixel mask logits.

    Adds ``alpha * log(prior)`` to ``logits`` after spatially resampling the
    (small) prior to match the logits' (H, W). This is a SOFT bias toward the
    empirical class distribution, not a hard projection — at ``alpha = 0`` it
    is a no-op; large ``alpha`` collapses toward the prior.

    Args:
        logits: ``(B, C, H, W)`` float tensor where C == :data:`NUM_CLASSES`.
        prior: ``(NUM_CLASSES, H_prior, W_prior)`` numpy array or torch tensor
            with each spatial cell summing to 1.0 across the class axis (as
            produced by :func:`experiments.build_mask_class_prior.build_mask_class_prior`).
        alpha: bias weight. ``0.1`` is a safe starting point; the lane MOS
            council will sweep this. Negative values are allowed (anti-prior)
            but unusual.

    Returns:
        ``logits + alpha * log(prior_resampled_to_HW)``, same shape/dtype/device as
        ``logits``.
    """
    if logits.dim() != 4:
        raise ValueError(
            f"logits must be (B, C, H, W); got shape {tuple(logits.shape)}"
        )
    if logits.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"logits must have C={NUM_CLASSES} on axis 1; got {logits.shape[1]}"
        )

    prior_np = _coerce_prior_to_numpy(prior)
    _validate_prior_shape(prior_np)

    # Promote to float32 on the logits' device for numerical stability,
    # then rescale back to logits.dtype on return.
    prior_t = torch.from_numpy(prior_np.astype(np.float32, copy=False)).to(
        device=logits.device, dtype=torch.float32
    )

    target_h, target_w = logits.shape[-2], logits.shape[-1]
    # Resample low-res prior up to the logits' (H, W). Bilinear keeps the
    # per-cell simplex approximately normalized; we re-clamp + log below so a
    # tiny renormalization drift is harmless.
    prior_t = prior_t.unsqueeze(0)  # (1, C, h, w)
    if prior_t.shape[-2:] != (target_h, target_w):
        prior_t = F.interpolate(
            prior_t,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

    log_prior = torch.log(prior_t.clamp(min=_EPS))  # (1, C, H, W)

    return logits + float(alpha) * log_prior.to(dtype=logits.dtype)


def save_prior_to_archive(
    prior: np.ndarray | "torch.Tensor" | str | Path,
    archive_zip_path: str | Path,
    *,
    arcname: str = PRIOR_FILENAME,
) -> int:
    """Append a prior to an existing ``archive.zip`` without rewriting it.

    The submission archive (``archive.zip``) already contains the renderer
    weights, masks.mkv, and poses. This function appends ``prior.npz`` in-place
    using ``zipfile.ZipFile(..., mode="a")`` so we do NOT have to rebuild the
    rest of the archive.

    Accepts either:
      - a numpy array / torch tensor (re-serialized through
        :func:`experiments.build_mask_class_prior.write_prior_npz`'s npz layout); or
      - a path to a pre-built ``prior.npz`` on disk (copied as bytes).

    Args:
        prior: numpy array / torch tensor with shape (NUM_CLASSES, H, W),
            OR a path to an existing prior.npz.
        archive_zip_path: path to the archive to append into. Must already exist.
        arcname: name inside the zip (default :data:`PRIOR_FILENAME`).

    Returns:
        Size in bytes of the prior entry as written into the archive.
    """
    archive_zip_path = Path(archive_zip_path)
    if not archive_zip_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_zip_path}")

    # Determine the bytes to write.
    if isinstance(prior, (str, Path)):
        prior_path = Path(prior)
        if not prior_path.exists():
            raise FileNotFoundError(f"Prior file not found: {prior_path}")
        prior_bytes = prior_path.read_bytes()
    else:
        prior_np = _coerce_prior_to_numpy(prior)
        _validate_prior_shape(prior_np)
        # Re-serialize through numpy.savez_compressed into an in-memory buffer
        # so the on-disk layout matches what load_prior expects.
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            prior=prior_np.astype(PRIOR_DTYPE, copy=False),
            version=np.array(PRIOR_VERSION, dtype=np.int32),
        )
        prior_bytes = buf.getvalue()

    # Refuse to silently overwrite an existing entry — the operator should
    # rebuild the archive cleanly if they want to replace the prior.
    with zipfile.ZipFile(archive_zip_path, mode="a", compression=zipfile.ZIP_DEFLATED) as zf:
        if arcname in zf.namelist():
            raise FileExistsError(
                f"Archive already contains {arcname!r}: {archive_zip_path}. "
                "Rebuild the archive instead of appending a duplicate."
            )
        zf.writestr(arcname, prior_bytes)

    return len(prior_bytes)
