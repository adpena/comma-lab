# SPDX-License-Identifier: MIT
"""Tests for canonical ALASKA pair-constraint batch builder.

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS" non-negotiable + Slot EEE rule:
tests verify BEHAVIORAL invariants (pair structure preserved, augmentation
applied identically to both halves, labels alternating) not just config
field values.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.composition.alaska_inverse_steganalysis_patterns import (
    PairConstraintBatchConfig,
    PairConstraintBatchError,
    PairConstraintBatchStrategy,
    build_pair_constraint_batch,
)


def _mk_frames(n: int, h: int = 8, w: int = 8, c: int = 3, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.uniform(0, 255, size=(h, w, c)).astype(np.float32) for _ in range(n)]


def test_pair_constraint_batch_strategy_enum_4_values() -> None:
    assert {s.value for s in PairConstraintBatchStrategy} == {
        "cover_stego_pair",
        "cover_stego_pair_flip_aug",
        "frame0_frame1_pair",
        "frame0_frame1_with_latent",
    }


def test_config_rejects_zero_pairs() -> None:
    with pytest.raises(PairConstraintBatchError, match="must be >= 1"):
        PairConstraintBatchConfig(pairs_per_batch=0)


def test_config_rejects_bad_strategy_type() -> None:
    with pytest.raises(PairConstraintBatchError, match="must be PairConstraintBatchStrategy"):
        PairConstraintBatchConfig(strategy="not_an_enum")  # type: ignore[arg-type]


def test_build_batch_shape_2n() -> None:
    """Output batch must be shape ``(2*N, H, W, C)``."""
    cover = _mk_frames(4, seed=0)
    stego = _mk_frames(4, seed=1)
    cfg = PairConstraintBatchConfig(seed=42, augment_flip=False, augment_rot90=False)
    batch, labels = build_pair_constraint_batch(cover, stego, cfg)
    assert batch.shape == (8, 8, 8, 3)
    assert labels.shape == (8,)


def test_build_batch_labels_alternating_cover_stego() -> None:
    """Labels must interleave cover=0 + stego=1 per Yousfi's gen_train."""
    cover = _mk_frames(3, seed=0)
    stego = _mk_frames(3, seed=1)
    cfg = PairConstraintBatchConfig(seed=42, augment_flip=False, augment_rot90=False)
    _, labels = build_pair_constraint_batch(cover, stego, cfg)
    # 6 labels = 3 pairs; pattern = [cover, stego, cover, stego, cover, stego]
    assert list(labels) == [0, 1, 0, 1, 0, 1]


def test_build_batch_pair_adjacency_preserved_no_aug() -> None:
    """Without augmentation, cover/stego at indices (2i, 2i+1) must be the
    original input arrays unchanged."""
    cover = _mk_frames(2, seed=0)
    stego = _mk_frames(2, seed=1)
    cfg = PairConstraintBatchConfig(seed=42, augment_flip=False, augment_rot90=False)
    batch, _ = build_pair_constraint_batch(cover, stego, cfg)
    np.testing.assert_array_equal(batch[0], cover[0])
    np.testing.assert_array_equal(batch[1], stego[0])
    np.testing.assert_array_equal(batch[2], cover[1])
    np.testing.assert_array_equal(batch[3], stego[1])


def test_build_batch_pair_aug_identical_for_both_halves() -> None:
    """The CANONICAL pair constraint: same flip+rot is applied to BOTH cover
    and stego per pair. Slot EEE substantive-distinctness test: WITHOUT
    this, the pair structure is broken and the detector signal is lost.

    Verify by constructing pairs where stego = cover + delta; after
    augmentation, since flip + rot90 are isometries that act element-wise
    AFTER REARRANGEMENT, the multiset of (stego - cover) entries must be
    invariant: ``sorted(aug(stego) - aug(cover)) == sorted(delta)``.
    """
    cover = _mk_frames(4, seed=0)
    # Use float64 + integer-multiple delta to avoid float32 rounding noise
    delta = [
        np.arange(c.size, dtype=np.float64).reshape(c.shape).astype(np.float64)
        for c in cover
    ]
    cover64 = [c.astype(np.float64) for c in cover]
    stego = [c + d for c, d in zip(cover64, delta)]
    cfg = PairConstraintBatchConfig(
        seed=123, augment_flip=True, augment_rot90=True
    )
    batch, _ = build_pair_constraint_batch(cover64, stego, cfg)
    # For each pair, batch[2i+1] - batch[2i] is the augmented residual.
    # Flip+rot90 are isometries on the element multiset, so:
    for i in range(4):
        residual = batch[2 * i + 1] - batch[2 * i]
        np.testing.assert_array_equal(
            np.sort(residual.flatten()),
            np.sort(delta[i].flatten()),
        )


def test_build_batch_rejects_length_mismatch() -> None:
    with pytest.raises(PairConstraintBatchError, match="!= len"):
        build_pair_constraint_batch(_mk_frames(3), _mk_frames(2), PairConstraintBatchConfig())


def test_build_batch_rejects_empty() -> None:
    with pytest.raises(PairConstraintBatchError, match="empty"):
        build_pair_constraint_batch([], [], PairConstraintBatchConfig())


def test_build_batch_rejects_shape_mismatch() -> None:
    cover = [np.zeros((4, 4, 3), dtype=np.float32)]
    stego = [np.zeros((8, 8, 3), dtype=np.float32)]
    with pytest.raises(PairConstraintBatchError, match="shape mismatch"):
        build_pair_constraint_batch(cover, stego, PairConstraintBatchConfig())


def test_build_batch_deterministic_with_seed() -> None:
    """Same seed -> same augmentation choices -> bytewise identical batch."""
    cover = _mk_frames(4, seed=0)
    stego = _mk_frames(4, seed=1)
    cfg = PairConstraintBatchConfig(seed=42)
    b1, l1 = build_pair_constraint_batch(cover, stego, cfg)
    b2, l2 = build_pair_constraint_batch(cover, stego, cfg)
    np.testing.assert_array_equal(b1, b2)
    np.testing.assert_array_equal(l1, l2)
