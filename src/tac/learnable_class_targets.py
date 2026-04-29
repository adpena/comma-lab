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


def _logit_exact_fp64(targets: torch.Tensor) -> torch.Tensor:
    """Return fp64 raw values such that the fp64 forward path round-trips exactly.

    The forward computation ``(sigmoid(raw_fp64) * 255).to(fp32)`` lands on
    integer-valued fp32 targets exactly when ``raw_fp64`` is the analytical
    logit. Pure fp32 raw cannot achieve this — the discrete fp32 raw lattice
    skips over ~all integer-valued ``fp32(sigmoid * 255)`` outputs (verified
    empirically: 0/61 ULP candidates around the analytical raw produced
    exactly 12.0 for target=12).

    Round 6 codex finding B context: previous fp32 logit + eps=1e-4 broke
    6 LCT tests by producing 0.0255/254.9745 (endpoints) and 11.99999
    (non-endpoints). fp64 raw + fp32 cast in forward fixes both.

    Endpoints (target == 0 or 255) get raw ±200 so the fp64 sigmoid
    underflows/saturates to 0.0/1.0 exactly even after fp32 cast.
    """
    targets_fp64 = targets.to(torch.float64)
    eps = 1e-12
    p = (targets_fp64 / 255.0).clamp(eps, 1.0 - eps)
    raw = torch.log(p / (1.0 - p))
    raw = torch.where(targets_fp64 == 0.0, torch.full_like(raw, -200.0), raw)
    raw = torch.where(targets_fp64 == 255.0, torch.full_like(raw, 200.0), raw)
    return raw


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

    def __init__(
        self,
        initial: Optional[torch.Tensor] = None,
        *,
        ema_decay: float = 0.99,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if device is not None and torch.device(device).type == "mps":
            raise RuntimeError(
                "LearnableClassTargets rejects MPS device — codebook updates "
                "use float64 logit math that MPS does not support reliably; "
                "produces silent precision drift. Use cuda or cpu."
            )
        if initial is None:
            initial = _DEFAULT_TARGETS.clone()
        else:
            initial = torch.as_tensor(initial, dtype=torch.float32)
            if initial.shape != (NUM_CLASSES,):
                raise ValueError(
                    f"initial must have shape ({NUM_CLASSES},), got {tuple(initial.shape)}"
                )
        # Out-of-range initial values are CLIPPED to [0, 255] rather than
        # raising — `forward()` is sigmoid-bounded so any optimizer trajectory
        # that pushes raw_values toward ±inf is mathematically identical to a
        # clipped target. Tests expect clipping (e.g. -10 -> 0, 300 -> 255).
        initial = initial.clamp(0.0, 255.0)
        # Inverse sigmoid: logit(p) = log(p/(1-p)) where p = initial / 255.
        # eps must be small enough that sigmoid(logit(eps)) * 255 underflows to
        # exactly 0.0 in fp32 (and symmetrically 255.0 at the upper end). The
        # earlier eps=1e-4 produced 0.0255 / 254.9745 instead of 0/255 — broke
        # the exact-init contract for the default targets [0,255,64,192,128]
        # and 6 tests (Round 6 codex finding B). 1e-30 yields raw=±69.08; in
        # fp32, sigmoid(±69) saturates to 0.0 / 1.0 exactly, so endpoints round-
        # trip identically while non-endpoints (e.g. p=64/255=0.251) are
        # unaffected.
        # Find the fp32 raw_init values that produce sigmoid(raw)*255 == target
        # EXACTLY in fp32. fp32 logit + fp32 sigmoid drops 1 ULP of precision,
        # so we use ULP-search around the analytical logit. Round 6 codex
        # finding B: previous eps=1e-4 produced 0.0255/254.9745 (endpoints)
        # AND 11.99999 instead of 12.0 (non-endpoints) — both broke the
        # exact-init contract, killing 6 tests.
        raw_init = _logit_exact_fp64(initial)
        self.raw_values = nn.Parameter(raw_init)
        if not (0.0 < ema_decay < 1.0):
            raise ValueError(
                f"ema_decay must be in (0.0, 1.0); got {ema_decay}"
            )
        self.ema_decay: float = float(ema_decay)
        # van den Oord (2017) VQ-VAE persistent-buffer EMA. The naive
        # `c <- decay * c + (1 - decay) * mean(z|c)` weights every batch
        # equally regardless of the number of assignments — a batch with
        # 1 sample has the same EMA weight as a batch with 100 samples
        # (Fridrich Round 1 finding). The persistent variant tracks per-class
        # count `N_c` and accumulated sum `m_c`, producing centroid
        # `c = m_c / N_c`. This is what Quantizr uses.
        # Initialize the buffers from the current target so the first
        # update doesn't kick the codebook far from initialization.
        with torch.no_grad():
            initial_targets = (
                torch.sigmoid(self.raw_values.to(torch.float64)) * 255.0
            ).to(torch.float32)
        self.register_buffer(
            "ema_count",
            torch.ones(NUM_CLASSES, dtype=torch.float32),
        )
        self.register_buffer(
            "ema_sum",
            initial_targets.clone(),
        )
        if device is not None:
            self.to(device)

    def forward(self) -> torch.Tensor:
        """Return the current class targets in [0, 255], shape (NUM_CLASSES,).

        Sigmoid + multiply happen in fp64 then cast to fp32. Pure fp32 forward
        drops 1 ULP through the round-trip (e.g. integer target 12 returns
        11.99999), breaking the exact-init contract. fp64 keeps enough precision
        that fp32 cast lands on the integer exactly. Gradient still flows
        through the fp64 path back to the fp32 raw_values parameter.
        """
        return (torch.sigmoid(self.raw_values.to(torch.float64)) * 255.0).to(torch.float32)

    @torch.no_grad()
    def enforce_separation(self, min_gap: float = 32.0) -> "LearnableClassTargets":
        """Enforce minimum gap between adjacent sorted targets.

        AV1 monochrome quantization noise is ~10-15 gray levels at CRF 50.
        Targets must stay >= 32 apart so the Gaussian-LUT argmax decoder
        recovers the right class even under maximal noise.

        Mutates ``raw_values`` in-place and returns self for chaining.
        """
        targets = self.forward()
        sorted_t, sort_idx = torch.sort(targets)
        # Forward pass: push each element up to enforce min_gap from its
        # predecessor. May overshoot 255.
        for i in range(1, NUM_CLASSES):
            if sorted_t[i] - sorted_t[i - 1] < min_gap:
                sorted_t[i] = sorted_t[i - 1] + min_gap
        # Backward pass: if forward overshot 255, anchor at 255 and pull each
        # earlier element DOWN. This salvages the case where targets started
        # clustered near the high end (e.g. [128, 128, 129, 130, 131]) — the
        # forward-only sweep cannot satisfy min_gap because the cap at 255
        # leaves gap < min_gap; the bidirectional sweep finds a feasible
        # placement by sliding lower elements down.
        if sorted_t[NUM_CLASSES - 1] > 255.0:
            sorted_t[NUM_CLASSES - 1] = 255.0
            for i in range(NUM_CLASSES - 2, -1, -1):
                upper_bound = sorted_t[i + 1] - min_gap
                if sorted_t[i] > upper_bound:
                    sorted_t[i] = upper_bound
        # If after both sweeps the lowest target went below 0, the constraint
        # is infeasible (NUM_CLASSES * min_gap > 255 + range). Clamp to 0
        # rather than raise — caller's distortion will reflect the violation.
        sorted_t = sorted_t.clamp(0.0, 255.0)
        # Restore original order.
        new_targets = torch.empty_like(targets)
        new_targets[sort_idx] = sorted_t
        # Use the same exact-fp32 logit search as __init__ — naive fp32 logit
        # drifts 1 ULP through the sigmoid round-trip, breaking the exact-
        # separation contract (e.g. 159 - 127 came out 31.99999 vs 32 expected).
        new_raw = _logit_exact_fp64(new_targets)
        self.raw_values.copy_(new_raw)
        return self

    @torch.no_grad()
    def ema_update(
        self,
        assignments: torch.Tensor,
        gray_values: torch.Tensor,
    ) -> "LearnableClassTargets":
        """van den Oord (2017) VQ-VAE persistent-buffer EMA codebook update.

        Maintains per-class running stats:

            N_c <- decay * N_c + (1 - decay) * count(assignments == c)
            m_c <- decay * m_c + (1 - decay) * sum(gray_values[assignments == c])
            c   <- m_c / N_c

        Unlike the naive ``decay * c + (1-decay) * mean(z|c)`` form, this
        weights each batch by its actual contribution: a batch with 1 sample
        moves the codebook less than a batch with 100 samples assigned to
        the same class. Quantizr uses exactly this rule (Hinton/Vinyals/Dean
        2014 KL distill in their pipeline assumes a properly-tracked codebook).
        Round 1 codex review (Fridrich) flagged the previous batch-mean
        implementation as Medium severity — fixed here.

        Classes with zero assignments do NOT update their target this
        iteration; their N_c and m_c are still decayed (so targets gravitate
        slowly toward the existing centroid in the limit), preserving the
        VQ-VAE-2 dead-codebook handling.

        Args:
            assignments: (N,) long tensor of class IDs in [0, NUM_CLASSES).
            gray_values: (N,) float tensor of observed gray values in [0, 255].

        Returns: self (for chaining).
        """
        if assignments.dim() != 1 or gray_values.dim() != 1:
            raise ValueError(
                f"assignments and gray_values must be 1-D; got "
                f"{assignments.shape} and {gray_values.shape}"
            )
        if assignments.shape[0] != gray_values.shape[0]:
            raise ValueError(
                f"assignments ({assignments.shape}) and gray_values "
                f"({gray_values.shape}) must have matching length"
            )
        if assignments.numel() == 0:
            return self
        if not torch.all((assignments >= 0) & (assignments < NUM_CLASSES)):
            raise ValueError(
                f"assignments must be in [0, {NUM_CLASSES}); got "
                f"min={assignments.min().item()}, max={assignments.max().item()}"
            )
        decay = self.ema_decay
        gray_values = gray_values.to(torch.float32)
        # Per-class assignment count and accumulated sum for THIS batch.
        # scatter_add for efficient O(N) computation regardless of NUM_CLASSES.
        ones = torch.ones_like(gray_values)
        batch_count = torch.zeros(
            NUM_CLASSES, dtype=torch.float32, device=gray_values.device
        )
        batch_sum = torch.zeros(
            NUM_CLASSES, dtype=torch.float32, device=gray_values.device
        )
        batch_count.scatter_add_(0, assignments.long(), ones)
        batch_sum.scatter_add_(0, assignments.long(), gray_values)
        # Persistent EMA buffers — full-tensor decay even for unobserved
        # classes so the running stats forget stale data over time.
        self.ema_count.mul_(decay).add_(batch_count, alpha=1.0 - decay)
        self.ema_sum.mul_(decay).add_(batch_sum, alpha=1.0 - decay)
        # Centroid = sum / count. Laplace-smooth count by a small epsilon to
        # avoid division-by-zero on cold-start classes that have never been
        # assigned. eps=1e-6 is small enough not to bias the centroid.
        new_targets = (self.ema_sum / (self.ema_count + 1e-6)).clamp(0.0, 255.0)
        new_raw = _logit_exact_fp64(new_targets)
        self.raw_values.copy_(new_raw)
        return self

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
