"""Distillation Chain — composition/stacking primitive.

Hinton-Vinyals-Dean 2014 ("Distilling the Knowledge in a Neural Network",
https://arxiv.org/abs/1503.02531) showed that a small "student" network can
match a large "teacher's" predictions when trained on softmax outputs with
temperature ``T``. The student inherits both the dominant class and the
*dark knowledge* (relative magnitudes of non-dominant classes).

A distillation **chain** generalises this to L levels:

    teacher_0 (large) → teacher_1 (medium) → ... → student_L (small)     (D.1)

Each step compresses by a factor ~2-3x while preserving the bulk of the
score. The on-disk archive only stores the FINAL student; the intermediate
teachers are transient training artefacts.

This module implements the **planning + loss** primitives for a
distillation chain:

* :class:`DistillationLevel` — typed metadata for one level (model name,
  param count, target compression).
* :class:`DistillationChain` — chain spec (ordered list of levels).
* :func:`distillation_loss` — canonical KL distillation loss with
  temperature scaling and ``T²`` correction (Hinton 2014 §3).

The actual TRAINING is driven by the substrate's trainer skeleton; this
module is the typed/serialisable composition surface.

Source memos:
- Hinton-Vinyals-Dean 2014 KL distillation.
- Selfcomp's KL distill usage (CLAUDE.md): T=2.0 for SegNet distill.

Cross-references
----------------
- Trainer skeleton: ``tac.substrates._shared.trainer_skeleton``.
- Hypernetwork (``tac.composition.hypernetwork``) — each level can be a
  hypernet-generated student.
- Stack-of-stacks: ``tac.composition.stack_of_stacks.compose``.

Score-claim discipline (NON-NEGOTIABLE per CLAUDE.md)
-----------------------------------------------------
This module produces typed spec + a differentiable loss; it does not
modify archive bytes by itself. Substrate integration must register a
parser-section manifest entry per CLAUDE.md Catalog #124 before any
``score_claim=True``. Until paired ``[contest-CUDA]`` + ``[contest-CPU]``
anchors land on a distillation-chain-equipped substrate, every result is
``score_claim=False``, ``promotion_eligible=False``,
``ready_for_exact_eval_dispatch=False``.

HNeRV parity discipline (13 lessons)
------------------------------------
1. Score-aware: the distillation loss is differentiable wrt the student.
   The trainer wires :class:`EMA` + ``apply_eval_roundtrip=True`` per
   CLAUDE.md non-negotiables.
2. Export-first: :meth:`DistillationChain.serialize_state` deterministic.
3-6. Substrate concerns; not violated.
7. Bolt-on ≤ 300 LOC.
8-13. Standard substrate concerns; not violated.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import torch
import torch.nn.functional as F

DISTILL_MAGIC = b"DST1"
DISTILL_SCHEMA_VERSION = 1
DEFAULT_TEMPERATURE = 2.0  # Selfcomp's canonical default per CLAUDE.md.
DEFAULT_KL_WEIGHT = 1.0
DEFAULT_HARD_WEIGHT = 0.0


class DistillationError(ValueError):
    """Raised when a DistillationChain spec or loss input is invalid."""


@dataclass(frozen=True)
class DistillationLevel:
    """One level in a distillation chain.

    Args:
        name: human-readable label (e.g. "hnerv_large").
        param_count: number of parameters (bytes_estimate / dtype_bytes).
        target_archive_bytes: expected archive bytes after this level
            lands (informative; not enforced here).
        temperature: KL distillation temperature for this level
            (defaults to :data:`DEFAULT_TEMPERATURE`).
        kl_weight: weight on the KL distill loss term.
        hard_weight: weight on the hard-target (ground truth) loss term.
    """

    name: str
    param_count: int
    target_archive_bytes: int = 0
    temperature: float = DEFAULT_TEMPERATURE
    kl_weight: float = DEFAULT_KL_WEIGHT
    hard_weight: float = DEFAULT_HARD_WEIGHT

    def __post_init__(self) -> None:
        if not self.name:
            raise DistillationError("name must be non-empty")
        if self.param_count <= 0:
            raise DistillationError(
                f"param_count must be positive, got {self.param_count}"
            )
        if self.target_archive_bytes < 0:
            raise DistillationError(
                f"target_archive_bytes must be >= 0, got {self.target_archive_bytes}"
            )
        if self.temperature <= 0:
            raise DistillationError(
                f"temperature must be positive, got {self.temperature}"
            )
        if self.kl_weight < 0 or self.hard_weight < 0:
            raise DistillationError("loss weights must be non-negative")
        if self.kl_weight + self.hard_weight == 0:
            raise DistillationError(
                "at least one of kl_weight or hard_weight must be positive"
            )


@dataclass(frozen=True)
class DistillationChain:
    """An ordered chain of distillation levels (teacher → ... → student).

    Args:
        levels: tuple of :class:`DistillationLevel` in order from largest
            teacher to smallest student. Must have >= 2 entries.
        compression_factor_floor: minimum compression ratio between
            adjacent levels (e.g. 1.5 means each student must have ≤
            ``teacher.param_count / 1.5`` params). Default 1.5.
    """

    levels: tuple[DistillationLevel, ...]
    compression_factor_floor: float = 1.5

    def __post_init__(self) -> None:
        if len(self.levels) < 2:
            raise DistillationError(
                f"chain must have >= 2 levels, got {len(self.levels)}"
            )
        if self.compression_factor_floor <= 1.0:
            raise DistillationError(
                f"compression_factor_floor must be > 1.0, got "
                f"{self.compression_factor_floor}"
            )
        for i in range(len(self.levels) - 1):
            teacher = self.levels[i]
            student = self.levels[i + 1]
            ratio = teacher.param_count / max(1, student.param_count)
            if ratio < self.compression_factor_floor:
                raise DistillationError(
                    f"level {i+1} ({student.name}) does not compress level "
                    f"{i} ({teacher.name}) by >= {self.compression_factor_floor}x "
                    f"(actual: {ratio:.2f}x)"
                )

    def num_levels(self) -> int:
        return len(self.levels)

    def total_compression(self) -> float:
        """Compression of final student vs initial teacher."""
        return self.levels[0].param_count / max(1, self.levels[-1].param_count)

    def serialize_state(self) -> bytes:
        """Deterministic spec serialisation."""
        body = bytearray()
        body += DISTILL_MAGIC
        body += struct.pack("<H", DISTILL_SCHEMA_VERSION)
        body += struct.pack("<H", len(self.levels))
        body += struct.pack("<d", self.compression_factor_floor)
        for level in self.levels:
            name_bytes = level.name.encode("ascii")
            body += struct.pack("<H", len(name_bytes))
            body += name_bytes
            body += struct.pack("<Q", level.param_count)
            body += struct.pack("<Q", level.target_archive_bytes)
            body += struct.pack("<d", level.temperature)
            body += struct.pack("<d", level.kl_weight)
            body += struct.pack("<d", level.hard_weight)
        return bytes(body)

    @classmethod
    def deserialize_state(cls, payload: bytes) -> DistillationChain:
        """Inverse of :meth:`serialize_state`."""
        if len(payload) < 4 or payload[:4] != DISTILL_MAGIC:
            raise DistillationError(f"bad magic: {payload[:4]!r}")
        off = 4
        (version,) = struct.unpack_from("<H", payload, off)
        off += 2
        if version != DISTILL_SCHEMA_VERSION:
            raise DistillationError(f"unsupported schema version: {version}")
        (num_levels,) = struct.unpack_from("<H", payload, off)
        off += 2
        (compression_floor,) = struct.unpack_from("<d", payload, off)
        off += 8
        levels = []
        for _ in range(num_levels):
            (name_len,) = struct.unpack_from("<H", payload, off)
            off += 2
            name = payload[off : off + name_len].decode("ascii")
            off += name_len
            (param_count,) = struct.unpack_from("<Q", payload, off)
            off += 8
            (target_archive,) = struct.unpack_from("<Q", payload, off)
            off += 8
            (temperature,) = struct.unpack_from("<d", payload, off)
            off += 8
            (kl_weight,) = struct.unpack_from("<d", payload, off)
            off += 8
            (hard_weight,) = struct.unpack_from("<d", payload, off)
            off += 8
            levels.append(
                DistillationLevel(
                    name=name,
                    param_count=param_count,
                    target_archive_bytes=target_archive,
                    temperature=temperature,
                    kl_weight=kl_weight,
                    hard_weight=hard_weight,
                )
            )
        return cls(
            levels=tuple(levels),
            compression_factor_floor=compression_floor,
        )


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    kl_weight: float = DEFAULT_KL_WEIGHT,
    hard_weight: float = DEFAULT_HARD_WEIGHT,
    hard_targets: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Canonical Hinton-Vinyals-Dean 2014 KL distillation loss.

    Loss = kl_weight · T² · KL(softmax(t/T) || log_softmax(s/T)) +
           hard_weight · CE(s, hard_targets)

    Args:
        student_logits: student outputs of shape ``(B, C, ...)``.
        teacher_logits: teacher outputs of shape ``(B, C, ...)``.
        temperature: distillation temperature ``T``.
        kl_weight: weight on KL distill term.
        hard_weight: weight on hard-target CE term (0 → no hard targets).
        hard_targets: integer class labels of shape ``(B, ...)`` if
            ``hard_weight > 0``; else ignored.
        reduction: "mean" or "sum".

    Returns:
        Scalar loss tensor with grad flowing through ``student_logits``.
    """
    if student_logits.shape != teacher_logits.shape:
        raise DistillationError(
            f"shape mismatch: student {student_logits.shape} vs teacher {teacher_logits.shape}"
        )
    if temperature <= 0:
        raise DistillationError(f"temperature must be positive, got {temperature}")
    if kl_weight < 0 or hard_weight < 0:
        raise DistillationError("loss weights must be non-negative")
    if reduction not in {"mean", "sum"}:
        raise DistillationError(f"reduction must be 'mean'/'sum', got {reduction!r}")

    loss = student_logits.new_zeros(())
    if kl_weight > 0:
        # Hinton 2014 KL distill: scale by T² so gradient magnitudes match
        # the hard-target CE gradient as T → ∞.
        T = temperature
        soft_t = F.softmax(teacher_logits.detach() / T, dim=1)
        log_soft_s = F.log_softmax(student_logits / T, dim=1)
        # KL(p_t || p_s) ≈ -Σ p_t log p_s + Σ p_t log p_t (the log p_t term
        # is constant wrt student so we drop it; the loss is equivalent
        # to cross-entropy from teacher's soft targets).
        ce = -(soft_t * log_soft_s).sum(dim=1)
        kl_term = ce.mean() if reduction == "mean" else ce.sum()
        loss = loss + kl_weight * (T * T) * kl_term

    if hard_weight > 0:
        if hard_targets is None:
            raise DistillationError("hard_targets required when hard_weight > 0")
        if hard_targets.shape != student_logits.shape[:1] + student_logits.shape[2:]:
            raise DistillationError(
                f"hard_targets shape {hard_targets.shape} must match "
                f"student spatial dims {student_logits.shape[:1] + student_logits.shape[2:]}"
            )
        ce_hard = F.cross_entropy(
            student_logits,
            hard_targets.long(),
            reduction=reduction,
        )
        loss = loss + hard_weight * ce_hard

    return loss


__all__ = [
    "DEFAULT_HARD_WEIGHT",
    "DEFAULT_KL_WEIGHT",
    "DEFAULT_TEMPERATURE",
    "DISTILL_MAGIC",
    "DISTILL_SCHEMA_VERSION",
    "DistillationChain",
    "DistillationError",
    "DistillationLevel",
    "distillation_loss",
]
