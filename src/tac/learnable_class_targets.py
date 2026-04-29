"""Lane LCT — learnable CLASS_TARGETS for the grayscale-LUT mask codec.

van den Oord (VQ-VAE/WaveNet) prescription via senior engineer review:
the hard-coded CLASS_TO_GRAY = [0, 255, 64, 192, 128] is a fixed codebook.
Letting the codebook adapt during SegMap training (with EMA updates)
should improve seg distortion 0.5-2 percent points on the contest scorer.

Cost: 5 fp16 = 10 bytes added to the archive (negligible vs the 280KB
mask payload). The decoder simply reads these 10 bytes and passes them
to ``tac.mask_grayscale_lut.create_gaussian_softmax_lut(targets=...)``.

Constraints:
- Targets stay in [0, 255] (sigmoid-bounded).
- Targets stay separable under AV1 quantization noise (min gap >= 32
  when sorted, enforced post-update by ``enforce_separation``).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from tac.mask_grayscale_lut import CLASS_TO_GRAY, NUM_CLASSES


# Default initialization mirrors the Selfcomp hard-coded targets so that a
# zero-trained LCT model is identical to the no-LCT baseline.
_DEFAULT_TARGETS = torch.tensor(
    [float(CLASS_TO_GRAY[c]) for c in range(NUM_CLASSES)],
    dtype=torch.float32,
)


class LearnableClassTargets(nn.Module):
    """Learnable class-to-gray targets parameterized via sigmoid-bounded floats.

    The raw learnable parameter ``raw_values`` is unconstrained float; the
    forward pass applies sigmoid * 255 to map into [0, 255]. We use the
    inverse-sigmoid (logit) of the initial gray values to ensure that an
    untrained ``LearnableClassTargets()`` reproduces the Selfcomp targets
    exactly.

    Attributes:
        raw_values: nn.Parameter of shape (NUM_CLASSES,) — the optimizer
            target. Initialized so forward() == default Selfcomp targets.
    """

    def __init__(self, initial: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if initial is None:
            initial = _DEFAULT_TARGETS.clone()
        else:
            initial = torch.as_tensor(initial, dtype=torch.float32)
            if initial.shape != (NUM_CLASSES,):
                raise ValueError(
                    f"initial must have shape ({NUM_CLASSES},), got {tuple(initial.shape)}"
                )
            if not torch.all((initial >= 0) & (initial <= 255)):
                raise ValueError(
                    f"initial values must be in [0, 255], got {initial.tolist()}"
                )
        # Inverse sigmoid: logit(p) = log(p/(1-p)) where p = initial / 255.
        # Clamp to (epsilon, 1-epsilon) to avoid log(0).
        eps = 1e-4
        p = (initial / 255.0).clamp(eps, 1.0 - eps)
        raw_init = torch.log(p / (1.0 - p))
        self.raw_values = nn.Parameter(raw_init)

    def forward(self) -> torch.Tensor:
        """Return the current class targets in [0, 255], shape (NUM_CLASSES,)."""
        return torch.sigmoid(self.raw_values) * 255.0

    @torch.no_grad()
    def enforce_separation(self, min_gap: float = 32.0) -> None:
        """Enforce minimum gap between adjacent sorted targets.

        AV1 monochrome quantization noise is ~10-15 gray levels at CRF 50.
        Targets must stay >= 32 apart so the Gaussian-LUT argmax decoder
        recovers the right class even under maximal noise.

        Mutates ``raw_values`` in-place.
        """
        targets = self.forward()
        sorted_t, sort_idx = torch.sort(targets)
        # Adjust sorted positions to enforce min_gap.
        for i in range(1, NUM_CLASSES):
            if sorted_t[i] - sorted_t[i - 1] < min_gap:
                sorted_t[i] = (sorted_t[i - 1] + min_gap).clamp(max=255.0)
        # Restore original order.
        new_targets = torch.empty_like(targets)
        new_targets[sort_idx] = sorted_t
        # Convert back to raw via inverse sigmoid.
        eps = 1e-4
        p = (new_targets / 255.0).clamp(eps, 1.0 - eps)
        new_raw = torch.log(p / (1.0 - p))
        self.raw_values.copy_(new_raw)

    def serialize_to_bytes(self) -> bytes:
        """Serialize current targets as 5 fp16 = 10 bytes for archive shipping."""
        targets = self.forward().detach().to(torch.float16)
        return targets.cpu().numpy().tobytes()

    @classmethod
    def deserialize_from_bytes(cls, data: bytes) -> "LearnableClassTargets":
        """Reconstruct from 10-byte payload."""
        if len(data) != NUM_CLASSES * 2:
            raise ValueError(
                f"expected {NUM_CLASSES * 2} bytes (5 fp16), got {len(data)}"
            )
        import numpy as np

        arr = np.frombuffer(data, dtype=np.float16).copy()
        targets = torch.from_numpy(arr.astype("float32"))
        return cls(initial=targets)


__all__ = ["LearnableClassTargets"]
