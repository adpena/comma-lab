# SPDX-License-Identifier: MIT
"""Canonical pair-constraint batch builder (Yousfi 2019 ALASKA Pattern #2).

Origin: ``external/alaska_yousfi/src/tools/tf_utils.py:55-95`` upstream
``gen_train``/``gen_valid`` + ``input_fn`` lines 119-150.

The CANONICAL insight (Yousfi 2019 ALASKA):
Training a steganalysis detector on UNPAIRED random cover/stego batches is
DRAMATICALLY worse than training on PAIRS (same image, one cover + one
stego). The pair structure removes inter-image-content variance from the
gradient signal, letting the detector focus on the embedding signal.
Empirically Yousfi reports ~2-3pp accuracy gain on ALASKA validation just
from this batching change.

5-axis adaptation taxonomy
--------------------------
* **Axis A (contest)** -- cover/stego JPEG pair -> (frame_0, frame_1) pair
* **Axis B (problem space)** -- detector training -> generator+scorer joint
* **Axis C (math)** -- 1:1 pair structure preserved
* **Axis D (data)** -- BOSSBase pairs -> Comma2k19 + contest video pairs
* **Axis E (video)** -- temporal pair (frame_0 -> frame_1) is the canonical
  contest contract

Sister of slot Catalog #213 ``Comma2k19LocalCache`` for the canonical
data-fetch surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np

__all__ = (
    "PairConstraintBatchStrategy",
    "PairConstraintBatchConfig",
    "build_pair_constraint_batch",
    "PairConstraintBatchError",
)


class PairConstraintBatchError(ValueError):
    """Raised when pair-constraint batch construction violates an invariant."""


class PairConstraintBatchStrategy(str, Enum):
    """Canonical pair-construction strategies.

    * ``COVER_STEGO_PAIR`` -- Yousfi's canonical
      (cover, stego) pair (the ALASKA-#1-winning pattern).
    * ``COVER_STEGO_PAIR_FLIP_AUG`` -- Yousfi's canonical augmentation
      (random horizontal flip + random rot90; see ``gen_train`` upstream).
    * ``FRAME0_FRAME1_PAIR`` -- comma-video adaptation
      (frame_0, frame_1) temporal pair from upstream/videos/0.mkv.
    * ``FRAME0_FRAME1_WITH_LATENT`` -- per-pair latent-shared mode
      (matches PR101 canonical 28-d latent predicting 2 frames per latent
      per CLAUDE.md HNeRV parity L19).
    """

    COVER_STEGO_PAIR = "cover_stego_pair"
    COVER_STEGO_PAIR_FLIP_AUG = "cover_stego_pair_flip_aug"
    FRAME0_FRAME1_PAIR = "frame0_frame1_pair"
    FRAME0_FRAME1_WITH_LATENT = "frame0_frame1_with_latent"


@dataclass(frozen=True)
class PairConstraintBatchConfig:
    """Canonical pair-constraint batch config.

    Mirrors Yousfi's upstream ``input_fn`` parameter set; renamed for COMMA
    contest contract clarity.

    Attributes
    ----------
    strategy
        Which pair-construction strategy to use.
    pairs_per_batch
        How many (cover, stego) pairs per batch. Yousfi default: 32
        (-> total batch size 64). For comma video, default 16 -> batch 32
        because contest scorers are heavier than SRNet.
    augment_flip
        Random horizontal flip (axis=2 for NHWC; axis=3 for NCHW).
        Yousfi's canonical augmentation; preserves pair structure
        (both cover + stego flip together).
    augment_rot90
        Random rot90 of {0, 1, 2, 3}. Yousfi's canonical augmentation;
        preserves pair structure (same rot for cover + stego).
    seed
        Optional deterministic seed for augmentation choices.
        Default None => use numpy global state.
    """

    strategy: PairConstraintBatchStrategy = PairConstraintBatchStrategy.COVER_STEGO_PAIR_FLIP_AUG
    pairs_per_batch: int = 16
    augment_flip: bool = True
    augment_rot90: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.pairs_per_batch < 1:
            raise PairConstraintBatchError(
                f"pairs_per_batch={self.pairs_per_batch} must be >= 1"
            )
        if not isinstance(self.strategy, PairConstraintBatchStrategy):
            raise PairConstraintBatchError(
                f"strategy={self.strategy!r} must be PairConstraintBatchStrategy"
            )


def build_pair_constraint_batch(
    cover_frames: Sequence[np.ndarray],
    stego_frames: Sequence[np.ndarray],
    config: PairConstraintBatchConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a canonical pair-constraint batch.

    1:1 with Yousfi's ``gen_train`` upstream pattern:
    1. Stack cover + stego per-pair (axis=0 -> batch dim)
    2. Apply same augmentation (flip + rot90) to BOTH halves of each pair
    3. Emit labels (cover=0, stego=class_id) per pair

    Parameters
    ----------
    cover_frames
        List of cover arrays, shape ``(H, W, C)`` (NHWC; matches Yousfi
        upstream contract). Length must equal ``stego_frames``.
    stego_frames
        List of stego arrays, shape ``(H, W, C)`` (NHWC). Must have same
        shape as the paired cover.
    config
        :class:`PairConstraintBatchConfig`.

    Returns
    -------
    batch : np.ndarray
        Shape ``(2*N, H, W, C)`` where N = len(cover_frames). Interleaved
        as [cover_0, stego_0, cover_1, stego_1, ...] preserving pair
        adjacency for the pair-constraint loss.
    labels : np.ndarray
        Shape ``(2*N,)`` dtype uint8. Labels: 0 for cover, 1 for stego.

    Raises
    ------
    PairConstraintBatchError
        Length mismatch, shape mismatch, empty inputs, or invalid config.
    """
    if len(cover_frames) != len(stego_frames):
        raise PairConstraintBatchError(
            f"len(cover_frames)={len(cover_frames)} != len(stego_frames)={len(stego_frames)}"
        )
    if len(cover_frames) == 0:
        raise PairConstraintBatchError("cover_frames + stego_frames empty")
    rng = np.random.default_rng(config.seed)
    pairs: list[np.ndarray] = []
    labels: list[int] = []
    for cover, stego in zip(cover_frames, stego_frames):
        if cover.shape != stego.shape:
            raise PairConstraintBatchError(
                f"shape mismatch: cover={cover.shape} stego={stego.shape}"
            )
        rot_k = int(rng.integers(0, 4)) if config.augment_rot90 else 0
        flip = bool(rng.random() < 0.5) if config.augment_flip else False
        for arr, lbl in ((cover, 0), (stego, 1)):
            out = arr
            if rot_k:
                out = np.rot90(out, rot_k, axes=(0, 1))
            if flip:
                out = np.flip(out, axis=1)
            pairs.append(out)
            labels.append(lbl)
    return np.stack(pairs, axis=0), np.asarray(labels, dtype=np.uint8)
