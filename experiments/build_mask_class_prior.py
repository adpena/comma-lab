# SPDX-License-Identifier: MIT
"""Build a mask-class-frequency prior from GT masks (Lane MOS).

CLI::

    python experiments/build_mask_class_prior.py \
        --masks-mkv submissions/robust_current/masks.mkv \
        --output prior.npz \
        --prior-resolution 5x4 \
        --sigma 1.0

The script:

1. Decodes GT class masks (either an ``--masks-mkv`` AV1 file via
   ``tac.mask_codec.decode_masks_auto`` OR a pre-saved ``--masks-pt`` torch tensor).
2. Builds a per-cell class-frequency prior at low spatial resolution
   (typically 4–6 cells per axis).
3. Smooths the prior with a Gaussian kernel and renormalizes per cell
   so the class axis sums to 1.0.
4. Writes ``prior.npz`` (float16, ~20–500 bytes depending on resolution).

The resulting prior is bundled into ``archive.zip`` via
:func:`tac.mask_prior.save_prior_to_archive`. At inflate time the renderer
adds ``alpha * log(prior)`` to its predicted logits — see
:func:`tac.mask_prior.apply_prior_weighting`.

NOTE: a 5×4 prior is the default test/sanity resolution. Production lanes
should sweep resolution + alpha + sigma via the council. See
``project_outstanding_work_and_stacks_20260428`` for the Lane MOS proposal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Allow direct invocation (`python experiments/build_mask_class_prior.py`)
# AND import-as-module from the test suite. When run as a script, ensure
# the repository's src/ is on sys.path so `from tac.camera import ...` works.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from tac.camera import NUM_CLASSES  # noqa: E402
from tac.mask_prior import PRIOR_DTYPE, PRIOR_VERSION  # noqa: E402

__all__ = [
    "build_mask_class_prior",
    "write_prior_npz",
]


def _gaussian_kernel_2d(sigma: float, radius: int | None = None) -> np.ndarray:
    """Unit-sum 2-D Gaussian kernel for class-frequency smoothing."""
    if sigma <= 0:
        # Identity (no smoothing) when sigma is non-positive.
        return np.ones((1, 1), dtype=np.float32)
    if radius is None:
        radius = max(1, int(round(3.0 * sigma)))
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    g_1d = np.exp(-0.5 * (coords / sigma) ** 2)
    g_1d /= g_1d.sum()
    g_2d = np.outer(g_1d, g_1d)
    g_2d /= g_2d.sum()
    return g_2d.astype(np.float32)


def _convolve_same(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Reflect-pad 2-D convolution along the last two axes (C, H, W) → (C, H, W)."""
    if kernel.shape == (1, 1):
        return arr
    try:
        from scipy.ndimage import convolve  # type: ignore
    except ImportError:
        # Pure-numpy fallback: explicit reflect-pad + slide.
        c_dim, h, w = arr.shape
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2
        padded = np.pad(arr, ((0, 0), (ph, ph), (pw, pw)), mode="reflect")
        out = np.zeros_like(arr, dtype=np.float32)
        for dy in range(kh):
            for dx in range(kw):
                out += kernel[dy, dx] * padded[:, dy : dy + h, dx : dx + w]
        return out

    out = np.empty_like(arr, dtype=np.float32)
    for c in range(arr.shape[0]):
        out[c] = convolve(arr[c].astype(np.float32), kernel, mode="reflect")
    return out


def build_mask_class_prior(
    masks: torch.Tensor | np.ndarray,
    prior_resolution: tuple[int, int] = (5, 4),
    sigma: float = 1.0,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Compute a per-cell class-frequency prior from a stack of GT masks.

    Spatial layout convention
    -------------------------
    ``prior_resolution`` is ``(grid_w, grid_h)`` — width-first, matching the
    image-coordinate convention used elsewhere in this codebase. The returned
    array has shape ``(num_classes, grid_h, grid_w)`` so that ``axis=0`` is the
    class dimension and ``prior.sum(axis=0) == 1.0`` per spatial cell. This
    matches the test contract in
    ``src/tac/tests/test_mask_class_prior.py``.

    Args:
        masks: ``(N, H, W)`` int/long tensor (or numpy array) with values in
            ``[0, num_classes)``.
        prior_resolution: ``(grid_w, grid_h)`` low-resolution prior grid.
            Tiny grids (e.g. 5×4) are normal — the prior is meant to be ~20
            bytes inside ``archive.zip``.
        sigma: standard deviation (in cells) for the Gaussian smoothing kernel
            applied after histogram counting. Set to 0 to disable smoothing.
        num_classes: defaults to :data:`tac.camera.NUM_CLASSES` (5 per Yousfi
            SegNet).

    Returns:
        ``(num_classes, grid_h, grid_w)`` float32 numpy array, each spatial
        column summing to 1.0 along the class axis.
    """
    if isinstance(masks, torch.Tensor):
        masks_np = masks.detach().cpu().numpy()
    else:
        masks_np = np.asarray(masks)

    if masks_np.ndim != 3:
        raise ValueError(
            f"masks must be (N, H, W); got shape {masks_np.shape}"
        )
    masks_int = masks_np.astype(np.int64, copy=False)

    if masks_int.min() < 0 or masks_int.max() >= num_classes:
        raise ValueError(
            f"masks values must be in [0, {num_classes}); "
            f"got min={masks_int.min()}, max={masks_int.max()}"
        )

    grid_w, grid_h = int(prior_resolution[0]), int(prior_resolution[1])
    if grid_w < 1 or grid_h < 1:
        raise ValueError(
            f"prior_resolution must have positive dims; got {prior_resolution}"
        )

    n, h, w = masks_int.shape

    # Spatially downsample by averaging one-hot class indicators within each
    # grid cell. We bin via integer division on row/col indices, summed across
    # all N frames.
    # Compute per-pixel bin assignments.
    row_bin = np.minimum((np.arange(h) * grid_h) // h, grid_h - 1)  # (H,)
    col_bin = np.minimum((np.arange(w) * grid_w) // w, grid_w - 1)  # (W,)

    counts = np.zeros((num_classes, grid_h, grid_w), dtype=np.float64)

    # Vectorized accumulation per class. For 5 classes this is a small loop.
    for cls in range(num_classes):
        # (N, H, W) bool mask of pixels equal to this class.
        cls_mask = masks_int == cls
        # Sum over N first → (H, W) per-pixel hit counts.
        per_pixel = cls_mask.sum(axis=0).astype(np.float64)
        # Bin into the grid.
        # Use np.add.at for unbuffered scatter-accumulate.
        rb = row_bin[:, None]  # (H, 1)
        cb = col_bin[None, :]  # (1, W)
        rb_full = np.broadcast_to(rb, (h, w)).ravel()
        cb_full = np.broadcast_to(cb, (h, w)).ravel()
        np.add.at(counts[cls], (rb_full, cb_full), per_pixel.ravel())

    # Add a tiny Laplace-style smoothing so empty cells don't make log(0)
    # explode downstream. The Gaussian smoothing below also helps but only
    # spreads existing mass — Laplace adds a uniform floor.
    counts = counts + 1.0 / num_classes

    # Spatial smoothing.
    if sigma and sigma > 0:
        kernel = _gaussian_kernel_2d(sigma=sigma)
        counts = _convolve_same(counts.astype(np.float32), kernel).astype(np.float64)

    # Renormalize so class axis sums to 1.0 in every cell.
    cell_totals = counts.sum(axis=0, keepdims=True)
    cell_totals = np.maximum(cell_totals, 1e-12)
    prior = (counts / cell_totals).astype(np.float32)

    return prior


def write_prior_npz(prior: np.ndarray, output_path: str | Path) -> int:
    """Serialize a prior to ``output_path`` (npz, float16) and return file size.

    Stored keys:
      - ``prior``: ``(num_classes, H, W)`` float16 array.
      - ``version``: int32 scalar equal to :data:`tac.mask_prior.PRIOR_VERSION`.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prior_arr = np.asarray(prior)
    if prior_arr.ndim != 3:
        raise ValueError(f"prior must be 3-D; got shape {prior_arr.shape}")

    np.savez_compressed(
        output_path,
        prior=prior_arr.astype(PRIOR_DTYPE, copy=False),
        version=np.array(PRIOR_VERSION, dtype=np.int32),
    )
    return output_path.stat().st_size


def _load_masks_from_args(args: argparse.Namespace) -> torch.Tensor:
    """Load masks from --masks-mkv (AV1) or --masks-pt (saved tensor)."""
    if args.masks_pt:
        return torch.load(args.masks_pt, map_location="cpu", weights_only=True)
    if args.masks_mkv:
        from tac.mask_codec import decode_masks_auto

        path = Path(args.masks_mkv)
        # Use detect_mask_codec to robustly route .mkv / .bin / .amrc.
        from tac.mask_codec import detect_mask_codec

        codec = detect_mask_codec(path)
        return decode_masks_auto(path, codec=codec)
    raise SystemExit("Must pass --masks-mkv OR --masks-pt")


def _parse_resolution(spec: str) -> tuple[int, int]:
    """Parse '5x4' or '5,4' into (5, 4)."""
    spec = spec.strip().lower().replace(",", "x").replace(" ", "")
    parts = spec.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Expected WxH (e.g. 5x4); got {spec!r}"
        )
    return (int(parts[0]), int(parts[1]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a mask-class-frequency prior (Lane MOS).",
    )
    src = parser.add_argument_group("masks source (one required)")
    src.add_argument("--masks-mkv", type=str, default=None, help="Path to GT masks.mkv (AV1 / entropy / amrc)")
    src.add_argument("--masks-pt", type=str, default=None, help="Path to saved (N,H,W) torch tensor")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output prior.npz path",
    )
    parser.add_argument(
        "--prior-resolution",
        type=_parse_resolution,
        default=(5, 4),
        help="Grid resolution as WxH (default 5x4).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma in grid cells (default 1.0; 0 disables).",
    )
    args = parser.parse_args(argv)

    masks = _load_masks_from_args(args)
    print(
        f"[mask_prior] Loaded masks: shape={tuple(masks.shape)}, "
        f"dtype={masks.dtype}, min={int(masks.min())}, max={int(masks.max())}"
    )
    prior = build_mask_class_prior(
        masks,
        prior_resolution=args.prior_resolution,
        sigma=args.sigma,
    )
    size = write_prior_npz(prior, args.output)
    print(
        f"[mask_prior] Wrote prior {prior.shape} → {args.output} ({size} bytes, "
        f"sigma={args.sigma}, resolution={args.prior_resolution})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
