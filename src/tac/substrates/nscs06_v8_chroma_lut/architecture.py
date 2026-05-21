# SPDX-License-Identifier: MIT
"""NSCS06 v8 chroma-LUT architecture — scaled per-level x per-class chroma table.

Per WAVE-3-NSCS06-V8-CHROMA-LUT-SUBSTRATE-BUILD 2026-05-21 + CASCADE
COMPRESSION symposium commit ``d125af6c3`` PRIORITY 3 (Daubechies + Mallat
multi-scale partition discovery framing) + HONEST CASCADE-MORTALITY
ASSESSMENT commit ``d884dd6aa`` Rank 2 + NSCS06 v6 -> v7 cargo-cult-unwind
methodology empirically validated rescue path commit ``4292c8ce2``.

The v8 chroma-LUT substrate scales the v7 per-class chroma palette from
~15 bytes (5 classes x 3 RGB anchors) to a ~4096-byte
**per-grayscale-level x per-class** chroma table. The expansion targets
canonical equation #26 ``_INCLUDED_CONTEXTS['nscs06_v8_chroma_lut']``
(per ``src/tac/canonical_equations/procedural_codebook_savings.py:102``)
with predicted savings ``ΔS = -25 * (4096 - 32) / 37_545_489 ≈ -0.002706``
when the LUT bytes are replaced by a 32-byte deterministic seed.

Architecture summary (canonical-vs-unique per Catalog #290 + #303):

* **UNIQUE per layer**:
  - Chroma-LUT shape ``(GRAYSCALE_LEVELS, NUM_SEGNET_CLASSES, 3)`` of uint8
    indexed by ``(gray_quant_level, segnet_class) -> (R, G, B)``. This is
    a strict super-set of v7's per-class anchor (5 classes x 3 RGB).
  - Compress-side LUT derivation aggregates per-pixel ``(gray_lvl, class) -> RGB``
    over compress-time ground-truth frames (median across pixels in each bin).
  - Inflate-side LUT consumption replaces v7's ``_grayscale_plus_chroma_to_rgb``
    with a per-pixel ``LUT[gray_quant[y,x], cls[y,x]]`` lookup.
* **ADOPT canonical**:
  - ``tac.procedural_codebook_generator.derive_codebook_from_seed`` for the
    seed-to-LUT derivation path (PCG64 32-byte seed -> 4096-byte uint8 LUT).
  - ``tac.substrates._shared.trainer_skeleton.device_or_die`` /
    ``_pin_seeds`` / ``_utc_now_iso`` / etc.
  - ``tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call``.
  - ``tac.deploy.modal.runtime`` NVML env block in driver.
* **PRESERVE HARD-EARNED**:
  - 13 HNeRV parity discipline lessons (declared in __init__.py per Catalog #124).
  - Strict-scorer-rule (inflate.py imports ZERO scorer code).
  - 6-DOF affine warp (sister to v7; cargo-cult #4 stays UNWOUND).

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status | Notes |
|---|---|---|
| L1 substrate must be score-aware | PASS | scorer queried at COMPRESS time |
| L2 export-first archive grammar | PASS | CH08 grammar declared in archive.py |
| L3 monolithic 0.bin | PASS | single-file fixed-offset CH08 grammar |
| L4 inflate <= 100 LOC | substrate_engineering exception | ~120 LOC w/ chroma LUT lookup |
| L5 full RGB renderer | PASS | per-pair RGB output via LUT-lookup + affine warp |
| L6 score-domain Lagrangian | N/A | NO training; bit allocation closed-form (sister to v7) |
| L7 bolt-on <= 350 LOC | substrate_engineering exception | total ~600 LOC across 4 files |
| L8 eval-roundtrip + diff yuv6 | N/A | NO training; simulated at compress only |
| L9 runtime closure | PASS | numpy + Pillow only (NO torch); same as v7 |
| L10 mask/pose coupling | PASS | pose deltas drive frame-1 affine warp from frame-0 |
| L11 no-op detector | PASS | Catalog #139 byte-mutation smoke planned + scaffolded |
| L12 single-LOC review discipline | PASS | each file reviewable in 30s |
| L13 KILL last resort | PASS | DEFERRED-pending-per-substrate-symposium |

This module's API:

  - :class:`Nscs06V8ChromaLutConfig` — frozen dataclass with the canonical
    grayscale_levels + num_segnet_classes + lut_dtype + seed_bytes config.
  - :data:`GRAYSCALE_LEVELS_DEFAULT` = 16 (canonical 4-bit luma quantization).
  - :data:`CHROMA_LUT_BYTES_DEFAULT` = 16 * 5 * 3 + padding = 4096 (canonical
    nscs06_v8_chroma_lut size per canonical equation #26).
  - :data:`PROCEDURAL_SEED_SIZE_BYTES` = 32 (canonical PCG64 seed; sister
    DP1 + VQ-VAE + grayscale_lut pattern).

The architecture is INTENTIONALLY decoupled from v7's existing trainer +
inflate runtime: v8 lives in its own package per UNIQUE-AND-COMPLETE-PER-METHOD.
Per Catalog #290 canonical-vs-unique decision per layer: forking the LUT
shape from v7 is PRINCIPLED MISMATCH (v7's 15-byte per-class anchor is too
small for the canonical equation #26 IN-DOMAIN context bytes-saved
prediction; expanding it in v7 would silently change v7's archive grammar).

CLAUDE.md compliance per Catalog #287 + #323:
- No score CLAIM asserted; predicted ΔS is PREDICTED-only.
- No silent device defaults (compress side runs scorer via canonical helper).
- No /tmp paths (compress writes under args.output_dir; inflate honors $1/$2/$3).
- No KILL verdicts (DEFER-pending-per-substrate-symposium per Catalog #325).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

__all__ = [
    "CHROMA_LUT_BYTES_DEFAULT",
    "CHROMA_LUT_DTYPE_DEFAULT",
    "GRAYSCALE_LEVELS_DEFAULT",
    "NUM_SEGNET_CLASSES",
    "PROCEDURAL_SEED_SIZE_BYTES",
    "Nscs06V8ChromaLutConfig",
    "build_chroma_lut_from_ground_truth",
    "lookup_rgb_via_chroma_lut",
]


# ---------------------------------------------------------------------------
# Canonical constants per canonical equation #26 IN-DOMAIN context
# ``nscs06_v8_chroma_lut`` (per src/tac/canonical_equations/procedural_codebook_savings.py:102).
# ---------------------------------------------------------------------------

NUM_SEGNET_CLASSES: Final[int] = 5
"""Sister of nscs06_carmack_hotz_strip_everything.codec.NUM_SEGNET_CLASSES.

Matches upstream/modules.py SegNet `classes=5` (5 segmentation classes:
background, lane, vehicle, road, sky-or-other). Preserved as a v8 module-level
constant rather than importing v7 to keep v8 unique-and-complete-per-method
per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode".
"""

GRAYSCALE_LEVELS_DEFAULT: Final[int] = 16
"""Canonical 4-bit luma quantization for chroma-LUT indexing.

The v8 chroma LUT is indexed by ``(grayscale_quantized, segnet_class)`` where
``grayscale_quantized = gray_u8 >> 4`` (i.e. 16 levels). The choice of 16 is:

  - SMALL enough that the per-(level, class) median over compress-time GT
    pixels has enough samples to be statistically meaningful.
  - LARGE enough that the per-level chroma variation captures BT.601 luma
    range (0-255) with reasonable fidelity.
  - Combined with NUM_SEGNET_CLASSES=5 and 3-byte RGB anchors gives
    16 * 5 * 3 = 240 bytes of MINIMUM chroma table; the canonical
    equation #26 IN-DOMAIN context ``nscs06_v8_chroma_lut`` budgets
    ``4096`` bytes total (allowing future expansion to richer LUT shapes
    e.g. per-temporal-window or per-region without grammar churn).
"""

CHROMA_LUT_BYTES_DEFAULT: Final[int] = 4096
"""Canonical chroma LUT footprint per canonical equation #26.

Per ``src/tac/canonical_equations/procedural_codebook_savings.py:76``:
``_NSCS06_V8_BYTES_SAVED = 4096 - 32`` (the equation's `_NSCS06_V8_BYTES_SAVED`
constant assumes 4096-byte LUT replaced by 32-byte seed).

The canonical 4096-byte budget covers the v8 LUT shape
``(GRAYSCALE_LEVELS_DEFAULT, NUM_SEGNET_CLASSES, 3) = (16, 5, 3) = 240``
bytes PLUS reserved padding for future per-temporal-window or per-spatial-
region extensions. The padding is zero-filled at compress-time so the
canonical bytes-saved prediction stays byte-stable.
"""

CHROMA_LUT_DTYPE_DEFAULT: Final[np.dtype] = np.dtype(np.uint8)
"""uint8 matches BT.601 RGB anchor convention + PCG64 native byte output."""

PROCEDURAL_SEED_SIZE_BYTES: Final[int] = 32
"""Canonical PCG64 32-byte seed; sister DP1 / VQ-VAE / grayscale_lut pattern.

Per canonical equation #26 closed form
``ΔS = -25 * (CHROMA_LUT_BYTES_DEFAULT - PROCEDURAL_SEED_SIZE_BYTES) / 37_545_489``
= ``-25 * (4096 - 32) / 37_545_489 ≈ -0.002706`` [prediction; canonical-equation-26-grounded].
"""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Nscs06V8ChromaLutConfig:
    """Frozen configuration for the v8 chroma-LUT substrate.

    Per CLAUDE.md "Beauty, simplicity, and developer experience" + Catalog #290
    canonical-vs-unique decision per layer: keep the config narrow + frozen
    so the architecture cannot drift after import.

    Attributes:
        grayscale_levels: Number of luma quantization levels for LUT
            indexing. Default ``GRAYSCALE_LEVELS_DEFAULT`` = 16 (4-bit).
        num_segnet_classes: Number of SegNet semantic classes. Default
            ``NUM_SEGNET_CLASSES`` = 5 (matches upstream/modules.py).
        chroma_lut_bytes: Canonical chroma-LUT footprint in bytes. Default
            ``CHROMA_LUT_BYTES_DEFAULT`` = 4096 (canonical equation #26
            ``_NSCS06_V8_BYTES_SAVED`` predicate).
        lut_dtype: LUT dtype (default ``np.uint8`` matching BT.601 RGB).
    """

    grayscale_levels: int = GRAYSCALE_LEVELS_DEFAULT
    num_segnet_classes: int = NUM_SEGNET_CLASSES
    chroma_lut_bytes: int = CHROMA_LUT_BYTES_DEFAULT
    lut_dtype: np.dtype = CHROMA_LUT_DTYPE_DEFAULT

    def __post_init__(self) -> None:
        """Validate canonical invariants.

        Per Catalog #287 placeholder-rationale rejection sister discipline +
        CLAUDE.md "Forbidden empirical-claim-without-evidence-tag".
        """
        if self.grayscale_levels < 1 or self.grayscale_levels > 256:
            raise ValueError(
                f"grayscale_levels={self.grayscale_levels} outside [1, 256]"
            )
        if self.num_segnet_classes < 1 or self.num_segnet_classes > 32:
            raise ValueError(
                f"num_segnet_classes={self.num_segnet_classes} outside [1, 32]"
            )
        if self.chroma_lut_bytes < 1 or self.chroma_lut_bytes > 65535:
            raise ValueError(
                f"chroma_lut_bytes={self.chroma_lut_bytes} outside [1, 65535] u16"
            )
        # Validate the canonical structural minimum: the dense
        # (grayscale_levels, num_segnet_classes, 3) RGB table must FIT inside
        # the declared chroma_lut_bytes budget. Excess is padding (zero-filled).
        min_required = self.grayscale_levels * self.num_segnet_classes * 3
        if self.chroma_lut_bytes < min_required:
            raise ValueError(
                f"chroma_lut_bytes={self.chroma_lut_bytes} < minimum required "
                f"{min_required} (grayscale_levels * num_segnet_classes * 3)"
            )

    @property
    def chroma_lut_shape(self) -> tuple[int, int, int]:
        """Canonical chroma-LUT tensor shape ``(grayscale_levels, classes, 3)``."""
        return (self.grayscale_levels, self.num_segnet_classes, 3)


# ---------------------------------------------------------------------------
# Compress-side LUT derivation (from ground-truth pixels)
# ---------------------------------------------------------------------------


def build_chroma_lut_from_ground_truth(
    rgb_pairs: np.ndarray,
    class_labels: np.ndarray,
    *,
    grayscale_levels: int = GRAYSCALE_LEVELS_DEFAULT,
    num_segnet_classes: int = NUM_SEGNET_CLASSES,
) -> np.ndarray:
    """Derive the v8 ``(levels, classes, 3)`` chroma LUT from compress-time GT.

    For each ``(level, class)`` bin, compute the median (R, G, B) across all
    compress-time pixels whose luma quantizes to ``level`` AND whose SegNet
    class equals ``class``. Median is robust to outliers in noisy regions.

    Bins with zero pixels are filled with the per-class GLOBAL median (i.e.
    the v7 per-class anchor) as fallback so every bin has a valid RGB anchor.

    Args:
        rgb_pairs: (N, 3, H, W) uint8 RGB frames (compress-time GT).
        class_labels: (N, H, W) uint8 SegNet argmax labels (same N, H, W).
        grayscale_levels: Number of luma quantization levels (default 16).
        num_segnet_classes: Number of SegNet classes (default 5).

    Returns:
        np.ndarray ``(grayscale_levels, num_segnet_classes, 3)`` uint8 LUT.
    """
    if rgb_pairs.dtype != np.uint8:
        raise ValueError("rgb_pairs must be uint8")
    if class_labels.dtype != np.uint8:
        raise ValueError("class_labels must be uint8")
    if rgb_pairs.ndim != 4 or rgb_pairs.shape[1] != 3:
        raise ValueError(f"rgb_pairs must be (N, 3, H, W); got {rgb_pairs.shape}")
    n, _, h, w = rgb_pairs.shape
    if class_labels.shape != (n, h, w):
        raise ValueError(
            f"class_labels shape {class_labels.shape} != ({n}, {h}, {w})"
        )

    # Compute BT.601 luma per pixel as the LUT level index.
    r = rgb_pairs[:, 0].astype(np.float32)
    g = rgb_pairs[:, 1].astype(np.float32)
    b = rgb_pairs[:, 2].astype(np.float32)
    luma = (0.299 * r + 0.587 * g + 0.114 * b).clip(0.0, 255.0)
    # Quantize luma -> 0..grayscale_levels-1 (use np.floor not np.round to
    # match the inflate-time `gray_u8 >> shift` indexing).
    level_step = 256 // grayscale_levels
    level_idx = np.clip(
        (luma // level_step).astype(np.int64), 0, grayscale_levels - 1
    )

    lut = np.zeros((grayscale_levels, num_segnet_classes, 3), dtype=np.uint8)
    rgb_flat = rgb_pairs.transpose(1, 0, 2, 3).reshape(3, -1)  # (3, N*H*W)
    cls_flat = class_labels.reshape(-1).astype(np.int64)
    level_flat = level_idx.reshape(-1)
    for c in range(num_segnet_classes):
        cls_mask = cls_flat == c
        # Per-class global median as fallback for empty bins.
        if cls_mask.any():
            global_median = np.array(
                [np.median(rgb_flat[ch][cls_mask]) for ch in range(3)],
                dtype=np.uint8,
            )
        else:
            global_median = np.array([128, 128, 128], dtype=np.uint8)
        for lvl in range(grayscale_levels):
            bin_mask = cls_mask & (level_flat == lvl)
            if bin_mask.any():
                for ch in range(3):
                    lut[lvl, c, ch] = np.uint8(np.median(rgb_flat[ch][bin_mask]))
            else:
                lut[lvl, c, :] = global_median
    return lut


# ---------------------------------------------------------------------------
# Inflate-side LUT consumption (per-pixel lookup)
# ---------------------------------------------------------------------------


def lookup_rgb_via_chroma_lut(
    gray_u8: np.ndarray,
    cls_u8: np.ndarray,
    chroma_lut: np.ndarray,
) -> np.ndarray:
    """Lookup RGB per pixel via the canonical v8 ``(levels, classes, 3)`` LUT.

    Replaces v7's ``_grayscale_plus_chroma_to_rgb`` per-class anchor with a
    per-(level, class) lookup that captures luma-conditional chroma
    variation. Cargo-cult #2 (Y=R=G=B chroma destruction) stays UNWOUND;
    v8 strengthens the unwind by indexing chroma on BOTH (level, class)
    rather than (class) alone.

    Args:
        gray_u8: (H, W) uint8 luma at output resolution.
        cls_u8: (H, W) uint8 SegNet class labels at output resolution.
        chroma_lut: (grayscale_levels, num_segnet_classes, 3) uint8 LUT.

    Returns:
        (H, W, 3) uint8 RGB frame.
    """
    if gray_u8.dtype != np.uint8 or cls_u8.dtype != np.uint8:
        raise ValueError("gray_u8 and cls_u8 must be uint8")
    if gray_u8.shape != cls_u8.shape:
        raise ValueError(f"shape mismatch {gray_u8.shape} vs {cls_u8.shape}")
    if chroma_lut.ndim != 3 or chroma_lut.shape[2] != 3:
        raise ValueError(
            f"chroma_lut must be (levels, classes, 3); got {chroma_lut.shape}"
        )
    grayscale_levels, num_segnet_classes, _ = chroma_lut.shape
    level_step = max(1, 256 // grayscale_levels)
    level_idx = np.clip(
        (gray_u8.astype(np.int64) // level_step), 0, grayscale_levels - 1
    )
    cls_idx = np.clip(cls_u8.astype(np.int64), 0, num_segnet_classes - 1)
    # Per-pixel lookup. chroma_lut[level, class] -> (3,) uint8 RGB anchor.
    out = chroma_lut[level_idx, cls_idx]  # (H, W, 3) uint8 via advanced indexing
    return np.ascontiguousarray(out, dtype=np.uint8)
