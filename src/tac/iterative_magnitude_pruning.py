# SPDX-License-Identifier: MIT
"""Lane J-IMP — Iterative Magnitude Pruning (LTH stabilized).

10-cycle Frankle-Carbin Iterative Magnitude Pruning with Frankle-2019
weight-rewinding-to-early-epoch (NOT to init). Each cycle:

  1. Train model to convergence on current mask.
  2. Globally rank surviving conv weights by |w| and zero the lowest
     ``sparsity_increment`` fraction (default 20%).
  3. Rewind all SURVIVING weights to a snapshot taken at ~1% through
     the FIRST training cycle (Frankle 1912.05671 stabilization fix).
  4. Repeat.

After 10 cycles at 20%/cycle the cumulative sparsity is
``1 - 0.8**10 = 0.893`` (≈89%). Combined with Lane Ω-V2 per-element
4-bit quantization on the surviving weights this targets a renderer
size of ``88K × 0.107 × 4 / 8 ≈ 4.7KB`` versus the current 170KB
dense FP32 / 44KB FP4.

Sparse-CSR export breakeven analysis at our 88K-param scale:

    dense FP4:  88_000 × 4 / 8                       =  44_000 bytes
    sparse-CSR: nnz × (uint16 idx + FP4 val)
              = nnz × 2.5 bytes
              < 44_000  ⇔  nnz < 17_600  ⇔  sparsity > 80.0%

So sparse-CSR only beats dense FP4 once we are above ~80% sparsity.
At 89% sparsity (target after 10 IMP cycles): nnz ≈ 9_400 → sparse-CSR
≈ 23_500 bytes, beating dense FP4 by ~46% (saves ~20.5KB ≈ -0.014
score points at the 25× rate multiplier).

Per the LTH RESEARCH-PARK note (revised DEPLOY-WORTHY at $200-500
budget): this lane is a moonshot — at our sub-100K param scale there
is no 2024 published evidence of LTH lottery tickets actually existing.
The expected band is ``[0.85, 1.00]`` against the Lane G v3 frontier
of 1.05.

Public API
----------
``IMPState``                     — dataclass with cycle bookkeeping.
``prune_lowest_magnitude(...)``  — global magnitude prune of conv weights.
``rewind_weights_to_early_epoch(...)`` — Frankle 2019 rewind step.
``compute_actual_sparsity(...)`` — fraction of pruned weights.
``sparse_csr_export(...)``       — pack uint16 indices + FP4 values.
``sparse_csr_decode(...)``       — inverse of ``sparse_csr_export``.
``apply_mask_to_model(...)``     — zero pruned weights in-place.
``snapshot_state_dict(...)``     — clone tensors for early-epoch save.

The module deliberately operates on ANY ``nn.Module`` (not just our
renderer arch) so the same code can be retargeted to Lane I (Cool-Chic)
or Lane GH (Ghost) in a follow-up.

Design notes
------------
* "Conv weights" = ``nn.Conv2d.weight`` and ``nn.ConvTranspose2d.weight``.
  Biases, BatchNorm parameters, embedding tables, and FiLM ``nn.Linear``
  layers are NOT pruned — they are tiny relative to the conv weights and
  pruning them produces brittle distortion (Frankle 2020 §4.1 also
  excludes BN+bias from the prune set).
* The mask is a ``dict[str, torch.BoolTensor]`` keyed by parameter
  qualified name (``model.named_parameters()`` key). True = keep,
  False = pruned.
* ``prune_lowest_magnitude`` is global across all conv weights — the
  threshold is chosen so the cumulative pruned fraction across the
  union equals ``sparsity_increment``. This matches Frankle &
  Carbin's "global pruning" variant which they show outperforms
  per-layer pruning on ResNets.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

__all__ = [
    "IMPState",
    "iter_prunable_parameters",
    "snapshot_state_dict",
    "prune_lowest_magnitude",
    "apply_mask_to_model",
    "rewind_weights_to_early_epoch",
    "compute_actual_sparsity",
    "sparse_csr_export",
    "sparse_csr_decode",
    "fp4_pack_values",
    "fp4_unpack_values",
]


# ── Utility helpers ─────────────────────────────────────────────────────


def iter_prunable_parameters(
    model: nn.Module,
) -> list[tuple[str, torch.nn.Parameter]]:
    """Return the list of ``(qualified_name, parameter)`` for prunable layers.

    Prunable = ``nn.Conv2d.weight`` and ``nn.ConvTranspose2d.weight``.
    Biases and BatchNorm parameters are NOT prunable (Frankle 2020 §4.1).
    """
    out: list[tuple[str, torch.nn.Parameter]] = []
    for module_name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            qname = f"{module_name}.weight" if module_name else "weight"
            out.append((qname, module.weight))
    return out


def snapshot_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Clone every parameter tensor for later rewinding.

    Used to capture the early-epoch state so subsequent IMP cycles can
    rewind survivors back to it (Frankle 1912.05671 stabilization).
    Detaches and clones onto CPU to keep snapshot memory bounded.
    """
    snap: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        snap[name] = param.detach().cpu().clone()
    return snap


# ── IMPState dataclass ──────────────────────────────────────────────────


@dataclass
class IMPState:
    """Bookkeeping for an IMP run across cycles.

    Attributes
    ----------
    cycle_count:
        Number of IMP cycles completed so far. Cycle 0 = first prune
        applied. After N cycles the cumulative target sparsity is
        ``1 - (1 - sparsity_increment) ** N``.
    sparsity_target:
        Total target sparsity (fraction of zeroed weights) for the
        FULL run. After cycle N the actual sparsity should track
        ``1 - (1 - sparsity_increment) ** (N + 1)``.
    sparsity_increment:
        Per-cycle sparsity fraction (default 0.20 = 20% of currently
        surviving weights pruned each cycle).
    mask:
        ``dict[str, torch.BoolTensor]``. True = keep, False = pruned.
    early_epoch_weights:
        Snapshot of model state taken ~1% through the first training
        cycle. Survivors are rewound to these values after each prune.

    All tensor fields are stored on CPU to keep IMPState serializable
    via ``torch.save``.
    """

    cycle_count: int = 0
    sparsity_target: float = 0.90
    sparsity_increment: float = 0.20
    mask: dict[str, torch.Tensor] = field(default_factory=dict)
    early_epoch_weights: dict[str, torch.Tensor] = field(default_factory=dict)

    def expected_sparsity_after_cycle(self, cycle: int) -> float:
        """Cumulative sparsity expected after ``cycle`` cycles of pruning.

        Cycle 0 means "first prune applied" → 20% sparsity by default.
        Cycle 9 (10 prunes total) → ``1 - 0.8**10 = 89.3%``.
        """
        return 1.0 - (1.0 - self.sparsity_increment) ** (cycle + 1)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a torch-savable dict (no tensors lost)."""
        return {
            "cycle_count": int(self.cycle_count),
            "sparsity_target": float(self.sparsity_target),
            "sparsity_increment": float(self.sparsity_increment),
            "mask": {k: v.detach().cpu().bool() for k, v in self.mask.items()},
            "early_epoch_weights": {
                k: v.detach().cpu() for k, v in self.early_epoch_weights.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "IMPState":
        return cls(
            cycle_count=int(d.get("cycle_count", 0)),
            sparsity_target=float(d.get("sparsity_target", 0.90)),
            sparsity_increment=float(d.get("sparsity_increment", 0.20)),
            mask={k: v.bool() for k, v in d.get("mask", {}).items()},
            early_epoch_weights=dict(d.get("early_epoch_weights", {})),
        )


# ── Pruning step ────────────────────────────────────────────────────────


def prune_lowest_magnitude(
    model: nn.Module,
    sparsity_increment: float = 0.20,
    current_mask: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Globally prune the lowest-magnitude conv weights.

    The threshold is chosen so that an additional ``sparsity_increment``
    fraction of CURRENTLY-SURVIVING weights are zeroed. With
    ``sparsity_increment=0.20`` and starting from a dense model:

        cycle 0 → 20% sparsity
        cycle 1 → 36% sparsity (1 - 0.8 * 0.8)
        ...
        cycle 9 → 89.3% sparsity (1 - 0.8**10)

    Args:
        model: ``nn.Module`` whose ``Conv2d.weight`` / ``ConvTranspose2d.weight``
            tensors will be measured. The model itself is NOT mutated; call
            ``apply_mask_to_model`` to zero the weights in-place.
        sparsity_increment: fraction of currently-surviving weights to prune.
            Must be in ``(0, 1)``.
        current_mask: optional existing mask from a previous cycle. If
            provided, only weights where ``current_mask`` is True are
            considered for pruning (already-pruned weights stay pruned).
            If None, treats every weight as currently surviving.

    Returns:
        New ``mask`` dict (True = keep, False = pruned). Existing-pruned
        positions are preserved as False so cumulative sparsity is monotone.

    Raises:
        ValueError: if ``sparsity_increment`` is not in (0, 1).
    """
    if not (0.0 < sparsity_increment < 1.0):
        raise ValueError(
            f"sparsity_increment must be in (0, 1); got {sparsity_increment!r}"
        )

    prunable = iter_prunable_parameters(model)
    if not prunable:
        return current_mask if current_mask is not None else {}

    # Build initial mask if absent.
    if current_mask is None:
        current_mask = {
            name: torch.ones_like(param, dtype=torch.bool, device="cpu")
            for name, param in prunable
        }

    # Collect magnitudes of CURRENTLY SURVIVING weights into a single
    # global tensor so the threshold spans all conv layers.
    surviving_abs: list[torch.Tensor] = []
    for name, param in prunable:
        mask_t = current_mask.get(name)
        if mask_t is None:
            mask_t = torch.ones_like(param, dtype=torch.bool, device="cpu")
            current_mask[name] = mask_t
        # CPU magnitude — cheap; threshold computation happens once per cycle.
        abs_t = param.detach().cpu().abs()
        surviving_abs.append(abs_t[mask_t])
    if not surviving_abs:
        return current_mask
    flat = torch.cat([t.flatten() for t in surviving_abs])
    n_surviving = int(flat.numel())
    if n_surviving == 0:
        return current_mask
    n_to_prune = int(round(n_surviving * sparsity_increment))
    n_to_prune = max(1, min(n_to_prune, n_surviving - 1))

    # kthvalue is more memory-efficient than full sort for huge tensors,
    # but for our 88K-param renderer either is fine. kthvalue picks the
    # n_to_prune-th smallest magnitude → that defines the cutoff.
    threshold = torch.kthvalue(flat, n_to_prune).values.item()

    # Apply threshold per-layer to build the new mask.
    new_mask: dict[str, torch.Tensor] = {}
    pruned_count = 0
    for name, param in prunable:
        old_mask = current_mask[name]
        abs_t = param.detach().cpu().abs()
        # Survivors after this cycle: magnitude > threshold AND was
        # already a survivor. Strictly-greater here means weights at
        # exactly the threshold magnitude get pruned (matches kthvalue).
        survives = (abs_t > threshold) & old_mask
        # Tie-handling: if weights == threshold mass exactly equal the
        # threshold (e.g. zero-init weights or duplicated values), prefer
        # to prune them so cumulative sparsity tracks the requested
        # fraction. The strict > already does this.
        new_mask[name] = survives
        pruned_count += int((old_mask & ~survives).sum().item())

    return new_mask


# ── Mask application + rewinding ────────────────────────────────────────


def apply_mask_to_model(
    model: nn.Module,
    mask: dict[str, torch.Tensor],
) -> None:
    """Zero pruned weights in-place according to ``mask``.

    Args:
        model: model to mutate in-place.
        mask: dict from ``prune_lowest_magnitude``. Missing keys are
            treated as "fully kept" (no zeroing).
    """
    for name, param in iter_prunable_parameters(model):
        m = mask.get(name)
        if m is None:
            continue
        m_dev = m.to(param.device)
        with torch.no_grad():
            param.mul_(m_dev.to(param.dtype))


def rewind_weights_to_early_epoch(
    model: nn.Module,
    early_epoch_weights: dict[str, torch.Tensor],
    mask: dict[str, torch.Tensor],
) -> None:
    """Rewind surviving weights to the early-epoch snapshot.

    Pruned positions stay zero — only ``mask=True`` positions are copied
    from the snapshot. This is the Frankle 2019 stabilization fix
    (1912.05671): rewinding to ~1% through training instead of to
    initialization recovers LTH at ResNet-50 / ImageNet scale.

    Args:
        model: model to mutate in-place.
        early_epoch_weights: snapshot from ``snapshot_state_dict`` taken
            at ~1% through training cycle 0.
        mask: prune mask. Pruned positions are zeroed; survivors are
            copied from the snapshot.

    Raises:
        KeyError: if a prunable weight is missing from the snapshot.
    """
    prunable_names = {name for name, _ in iter_prunable_parameters(model)}
    for name, param in model.named_parameters():
        if name not in early_epoch_weights:
            # Non-prunable params (BN, biases) silently retain current state
            # if they aren't in the snapshot. This is intentional — the
            # snapshot is supposed to be complete, but legacy callers may
            # pass partial dicts.
            continue
        snap = early_epoch_weights[name].to(param.device).to(param.dtype)
        if name in prunable_names:
            m = mask.get(name)
            if m is None:
                # No mask entry → treat as fully-kept (rewind everything).
                with torch.no_grad():
                    param.copy_(snap)
                continue
            m_dev = m.to(param.device)
            with torch.no_grad():
                # Survivors get snapshot value, pruned stay zero.
                param.copy_(snap * m_dev.to(snap.dtype))
        else:
            with torch.no_grad():
                param.copy_(snap)


# ── Sparsity reporting ──────────────────────────────────────────────────


def compute_actual_sparsity(
    model: nn.Module,
    mask: dict[str, torch.Tensor] | None = None,
) -> float:
    """Fraction of conv weights set to zero (or False in mask).

    If ``mask`` is provided, the sparsity is computed from the mask
    directly (cheap, exact). Otherwise the sparsity is computed by
    counting zero-valued weights in the model itself, which is what
    matters for the EXPORTED archive size.

    Returns 0.0 when the model has no prunable weights.
    """
    if mask is not None:
        total = 0
        zero = 0
        for _name, m in mask.items():
            total += int(m.numel())
            zero += int((~m).sum().item())
        return zero / total if total > 0 else 0.0

    total = 0
    zero = 0
    for _name, param in iter_prunable_parameters(model):
        total += int(param.numel())
        zero += int((param == 0).sum().item())
    return zero / total if total > 0 else 0.0


# ── Sparse-CSR export (uint16 indices + FP4 values) ─────────────────────
#
# Layout per layer (all little-endian):
#
#     int32   nnz            (number of nonzero values)
#     int32   numel          (total elements in this tensor before sparsity)
#     int32   shape_ndim
#     int32×ndim shape
#     float32 scale          (per-tensor abs_max for FP4 dequant)
#     uint16×nnz indices     (flat indices of survivors, ascending)
#     packed FP4×nnz values  ((nnz+1)//2 bytes; high nibble = even idx)
#
# A nibble takes 4 bits; ``nibble_to_float`` maps the 16 nibble values
# to a symmetric 16-level codebook in ``[-1, 1]`` and we multiply by
# ``scale`` on decode. This is the same FP4 encoding used by Lane F.
#
# numel ≤ 65535 is REQUIRED so uint16 indices suffice. Our renderer's
# largest conv has ~16K weights (60×60×3×3); well under the cap.


_FP4_LEVELS_INT = list(range(16))  # 0..15 nibble values


def _fp4_levels_signed_float() -> list[float]:
    """Map nibble values 0..15 to signed levels in [-1, +1].

    16 evenly-spaced signed values: nibble k → (k - 7.5) / 7.5.
    """
    return [(k - 7.5) / 7.5 for k in _FP4_LEVELS_INT]


def fp4_pack_values(values_unsigned: list[int]) -> bytes:
    """Pack a list of unsigned 4-bit integers (0..15) into bytes.

    Two values per byte; first value goes into the LOW nibble for
    little-endian-natural reading (so ``unpack[0] == data[0] & 0xF``).
    """
    out = bytearray()
    for i in range(0, len(values_unsigned), 2):
        lo = values_unsigned[i] & 0xF
        if i + 1 < len(values_unsigned):
            hi = values_unsigned[i + 1] & 0xF
        else:
            hi = 0
        out.append(lo | (hi << 4))
    return bytes(out)


def fp4_unpack_values(data: bytes, count: int) -> list[int]:
    """Inverse of ``fp4_pack_values``: extract ``count`` unsigned nibbles."""
    out: list[int] = []
    for i in range(count):
        b = data[i // 2]
        if i % 2 == 0:
            out.append(b & 0xF)
        else:
            out.append((b >> 4) & 0xF)
    return out


def _quantize_to_fp4(values: torch.Tensor) -> tuple[list[int], float]:
    """Quantize a 1D float tensor to FP4 nibbles + per-tensor scale.

    Returns (nibble_list, scale). ``scale`` is per-tensor abs_max (clamped
    to >= 1e-12 to avoid div-by-zero on all-zero tensors).
    """
    if values.numel() == 0:
        return [], 0.0
    abs_max = values.detach().abs().max().clamp(min=1e-12).item()
    levels = _fp4_levels_signed_float()  # length 16
    levels_t = torch.tensor(levels, dtype=values.dtype, device=values.device)
    # For each value, find the closest level after dividing by scale.
    normed = values / abs_max
    # Compare against codebook → argmin.
    diffs = (normed.reshape(-1, 1) - levels_t.reshape(1, -1)).abs()
    nibbles = diffs.argmin(dim=1).to(torch.long).tolist()
    return nibbles, float(abs_max)


def _dequantize_from_fp4(nibbles: list[int], scale: float) -> torch.Tensor:
    """Inverse of ``_quantize_to_fp4``."""
    levels = _fp4_levels_signed_float()
    return torch.tensor(
        [levels[n] * scale for n in nibbles], dtype=torch.float32
    )


def sparse_csr_export(
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> bytes:
    """Pack a single conv weight tensor as sparse CSR (uint16 idx + FP4 val).

    Args:
        weights: dense weight tensor of shape ``(C_out, C_in, kH, kW)`` or
            similar. Any shape allowed; flattening order is row-major.
        mask: bool tensor of the same shape. True = surviving weight.

    Returns:
        Packed bytes (see module docstring for layout).

    Raises:
        ValueError: if shapes mismatch or numel exceeds the uint16 cap.
    """
    if weights.shape != mask.shape:
        raise ValueError(
            f"shape mismatch: weights={tuple(weights.shape)} "
            f"mask={tuple(mask.shape)}"
        )
    numel = int(weights.numel())
    if numel > 65535:
        raise ValueError(
            f"sparse_csr_export only supports tensors with numel<=65535 "
            f"(got {numel}); split larger tensors into per-out-channel slices "
            f"or upgrade to uint32 indices."
        )

    flat_w = weights.detach().cpu().flatten()
    flat_m = mask.detach().cpu().flatten().bool()
    idxs = torch.nonzero(flat_m, as_tuple=False).flatten().tolist()
    surviving = flat_w[flat_m]
    nibbles, scale = _quantize_to_fp4(surviving)

    buf = bytearray()
    # Header
    buf.extend(struct.pack("<I", len(idxs)))   # nnz
    buf.extend(struct.pack("<I", numel))       # numel
    shape = list(weights.shape)
    buf.extend(struct.pack("<I", len(shape)))  # ndim
    for s in shape:
        buf.extend(struct.pack("<I", int(s)))
    buf.extend(struct.pack("<f", float(scale)))
    # Indices (uint16 each)
    for i in idxs:
        buf.extend(struct.pack("<H", int(i)))
    # FP4 packed values
    buf.extend(fp4_pack_values(nibbles))
    return bytes(buf)


def sparse_csr_decode(data: bytes) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of ``sparse_csr_export``.

    Returns ``(dense_weight, mask)``: the dense FP32 weight tensor (with
    pruned positions set to 0.0) and the bool mask.
    """
    offset = 0
    (nnz,) = struct.unpack("<I", data[offset:offset + 4]); offset += 4
    (numel,) = struct.unpack("<I", data[offset:offset + 4]); offset += 4
    (ndim,) = struct.unpack("<I", data[offset:offset + 4]); offset += 4
    shape: list[int] = []
    for _ in range(ndim):
        (s,) = struct.unpack("<I", data[offset:offset + 4]); offset += 4
        shape.append(s)
    (scale,) = struct.unpack("<f", data[offset:offset + 4]); offset += 4
    idxs: list[int] = []
    for _ in range(nnz):
        (i,) = struct.unpack("<H", data[offset:offset + 2]); offset += 2
        idxs.append(i)
    nibble_bytes = (nnz + 1) // 2
    nibbles = fp4_unpack_values(data[offset:offset + nibble_bytes], nnz)
    offset += nibble_bytes

    flat = torch.zeros(numel, dtype=torch.float32)
    mask_flat = torch.zeros(numel, dtype=torch.bool)
    if nnz > 0:
        values = _dequantize_from_fp4(nibbles, scale)
        flat[torch.tensor(idxs, dtype=torch.long)] = values
        mask_flat[torch.tensor(idxs, dtype=torch.long)] = True
    return flat.reshape(shape), mask_flat.reshape(shape)


# ── Smoke test ──────────────────────────────────────────────────────────


def _smoke_test() -> None:
    """Quick correctness check; mirrors the test-suite assertions."""
    print("iterative_magnitude_pruning: smoke test")
    torch.manual_seed(0)

    class TinyConv(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, padding=1)
            self.bn = nn.BatchNorm2d(8)
            self.c2 = nn.Conv2d(8, 4, 3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.c2(self.bn(self.c1(x)))

    model = TinyConv()
    snap = snapshot_state_dict(model)
    mask0 = prune_lowest_magnitude(model, sparsity_increment=0.2)
    s0 = compute_actual_sparsity(model, mask0)
    print(f"  cycle 0 sparsity: {s0:.3f}")
    apply_mask_to_model(model, mask0)
    rewind_weights_to_early_epoch(model, snap, mask0)

    # Round-trip a single layer through sparse-CSR export.
    name, param = iter_prunable_parameters(model)[1]
    blob = sparse_csr_export(param.detach(), mask0[name])
    dense, mask_dec = sparse_csr_decode(blob)
    print(f"  layer {name}: nnz={int(mask_dec.sum().item())}, blob={len(blob)} bytes")
    assert dense.shape == param.shape

    # Cycle 1
    mask1 = prune_lowest_magnitude(model, sparsity_increment=0.2,
                                   current_mask=mask0)
    s1 = compute_actual_sparsity(model, mask1)
    print(f"  cycle 1 sparsity: {s1:.3f}")
    print("smoke test OK")


if __name__ == "__main__":
    _smoke_test()
