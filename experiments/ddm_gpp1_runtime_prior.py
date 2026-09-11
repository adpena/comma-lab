"""Receiver-side online mixer for a causal generic motion-boundary belief.

The belief is computed from a public model on already-rendered RGB.  This module
contains no model weights, fitted video table, scorer state, or source symbols.
Its observed/expected tables start cold and update only after a complete token
plane has been decoded.
"""

from __future__ import annotations

import numpy as np

from .tc1_shared_mixer import SCALE, TOTAL, K, Q, frequencies, log2_fixed, mix_probabilities

BELIEF_LEVELS = 17


class GenericMotionMixer:
    """One fixed-strength KT calibration bank over argmax x public belief."""

    def __init__(self, weight_q5: int) -> None:
        if not 1 <= weight_q5 <= 32:
            raise ValueError("motion prior weight must be in [1,32]")
        self.weight_q5 = weight_q5
        self.counts = np.zeros((K * BELIEF_LEVELS, K), dtype=np.int64)
        self.expected = np.zeros_like(self.counts)

    def table(self) -> np.ndarray:
        """Freeze a Q10 log-ratio table from prior completed frames."""
        ratio = (self.counts.astype(np.float64) + 0.5) / (self.expected.astype(np.float64) / TOTAL + 0.5)
        return log2_fixed(np.clip(ratio, 1 / 16, 16))

    def coding(
        self,
        base: np.ndarray,
        belief: np.ndarray,
        table: np.ndarray,
    ) -> np.ndarray:
        """Return receiver probabilities without observing the current symbols."""
        base = np.asarray(base, dtype=np.uint32)
        belief = np.asarray(belief, dtype=np.int64).reshape(-1)
        if base.ndim != 2 or base.shape[1] != K or len(base) != len(belief):
            raise ValueError("motion prior base/belief shape mismatch")
        if np.any(base.sum(axis=1, dtype=np.uint64) != TOTAL):
            raise ValueError("motion prior base frequency mass changed")
        if np.any((belief < 0) | (belief >= BELIEF_LEVELS)):
            raise ValueError("motion prior belief escaped its fixed alphabet")
        argmax = base.argmax(axis=1)
        code = argmax * BELIEF_LEVELS + belief
        phi = table[code][:, :, None]
        original = (base.astype(np.float64) / TOTAL).astype(np.float32)
        result = mix_probabilities(
            base,
            phi,
            np.full((K, 1), self.weight_q5, dtype=np.int8),
            original_rows=original,
        )
        # Prove that the public float32 receiver and the retained integer
        # encoder will use the same lattice.
        encoded = frequencies(result)
        reconstructed = frequencies((encoded.astype(np.float64) / TOTAL).astype(np.float32))
        np.testing.assert_array_equal(encoded, reconstructed)
        return result

    def observe(
        self,
        base: np.ndarray,
        belief: np.ndarray,
        truth: np.ndarray,
    ) -> None:
        """Update only after the current plane is completely decoded."""
        base = np.asarray(base, dtype=np.uint32)
        belief = np.asarray(belief, dtype=np.int64).reshape(-1)
        truth = np.asarray(truth, dtype=np.int64).reshape(-1)
        if len(base) != len(belief) or len(base) != len(truth):
            raise ValueError("motion prior observation shape mismatch")
        argmax = base.argmax(axis=1)
        code = argmax * BELIEF_LEVELS + belief
        self.counts += np.bincount(code * K + truth, minlength=self.counts.size).reshape(self.counts.shape)
        for symbol in range(K):
            self.expected[:, symbol] += np.bincount(code, weights=base[:, symbol], minlength=len(self.counts)).astype(
                np.int64
            )

    def snapshot(self) -> dict[str, np.ndarray]:
        return {
            "weight_q5": np.array([self.weight_q5], dtype=np.int64),
            "counts": self.counts.copy(),
            "expected": self.expected.copy(),
        }

    def restore(self, state: dict[str, np.ndarray]) -> None:
        if int(state["weight_q5"][0]) != self.weight_q5:
            raise ValueError("motion prior checkpoint weight changed")
        if state["counts"].shape != self.counts.shape:
            raise ValueError("motion prior checkpoint shape changed")
        self.counts = state["counts"].copy()
        self.expected = state["expected"].copy()


__all__ = ["BELIEF_LEVELS", "SCALE", "GenericMotionMixer", "Q"]
