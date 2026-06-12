# SPDX-License-Identifier: MIT
"""Score-aware QAT (Lever 4) — quantize WHERE the argmax-boundary tolerates, protect
WHERE it flips.

The vendored QAT (``losses.apply_qat``) is **uniform symmetric INT8 fake-quant over
EVERY Conv2d/Linear weight** (``fake_quantize(w, n_levels=127)``). It is
L2-reconstruction-blind to which weights matter for d_seg/d_pose: quantization error
in a weight that drives a SegNet argmax-boundary pixel costs d_seg; the same error in
a weight the argmax ignores is free. Uniform quantization is provably sub-optimal when
the distortion sensitivity is non-uniform (reverse water-filling, Cover & Thomas
Ch.10).

This module makes the **per-tensor INT8 effective level count a function of measured
score sensitivity**, not uniform:

* **(4a) Sensitivity-weighted fake-quant.** Per-tensor ``s_t = ||∂S/∂w_t||`` (an EMA
  of the score-domain loss gradient magnitude — already computed once Lever 2 routes
  the score-domain loss; the backward populates ``w.grad``). HIGH-sensitivity tensors
  get MORE INT8 levels (finer grid → smaller quant error → the argmax boundary is
  protected → more brotli bytes there); LOW-sensitivity tensors get FEWER levels
  (coarser grid → more repeated/zero symbols → FEWER brotli bytes). This is the
  water-filling of the INT8 budget against score-sensitivity (the bit-allocator hook,
  CLAUDE.md 6-hook #3). Total bytes can DROP at equal d_seg because grid resolution
  stops being spent on score-irrelevant weights.

* **(4b) Boundary-protective straight-through QAT.** Automatic once the QAT fake-quant
  forward flows through Lever-2's d_seg surrogate (it wraps the SAME decoder forward
  whose output feeds the loss): the QAT-induced argmax flips are directly penalized in
  training, so the decoder learns weights robust to their OWN INT8 quantization at the
  boundary (eval_roundtrip sister discipline, applied to the weight grid).

The per-tensor level count maps to an **effective step within the codec's EXISTING
per-tensor-scale format** (``codec.py`` already stores one scale per tensor), so 4a is
GRAMMAR-COMPATIBLE — no new archive section. It is a TRAINING-TIME proxy: the actual
archive still uses the codec's INT8 (127-level) per-tensor scale; the score-aware QAT
shapes the decoder so that, post-codec-quant, the score-relevant weights survive and
the score-irrelevant ones collapse to repeated symbols brotli loves.

**MED-2 codec-side validation (2026-06-12,
``experiments/probe_lever4_qat_brotli_blob_delta.py``).** The R1 audit flagged the
byte-win as an UNPROVEN indirect effect because the codec always re-quantizes at 127.
The probe MEASURED the effect on the basin EMA decoder with a REAL frozen-scorer
``||∂S/∂w||`` sensitivity: the score-aware grid (13/14 tensors coarsened into the
[64,127] band) produced a **-3263 B (-4.4%) smaller vendored-codec brotli decoder blob**
(70264 vs 73527) at **equal advisory d_seg** (0.0034 → 0.0034). The coarse snap SURVIVES
the codec's own 127-requant — a 64-level snap collapses e.g. ``blocks.0.weight`` from 214
to 113 distinct codec-127 symbols → fewer brotli symbols → smaller blob — so the win is
the REAL deployed bytes, not a pre-codec artifact (Catalog #304 bit-spend proof:
POSITIVE on the codec axis). CAVEAT (honest scope): the probe applies the grid as a
one-shot SNAP to already-trained weights — it validates the CODEC half of the
hypothesis (coarse-grid-robust → fewer brotli symbols → smaller blob). The full Lever-4
mechanism additionally TRAINS the decoder to be robust at the coarse grid, which can
recover the small d_pose uptick the snap incurs (0.001663 → 0.001777 advisory); that
training half still requires the paired TRAINING A/B (uniform-QAT-trained vs
score-aware-QAT-trained, archive_bytes at equal d_seg/d_pose) + the dual CPU/CUDA exact
eval before any SCORE claim. The byte direction is validated; the net-score win is still
a prediction pending the training A/B.

NO-FAKE discipline (this module ACTUALLY varies the quantization grid per tensor by
the supplied sensitivity, on the real weights — verified in
``tests/test_score_aware_qat.py``):
- a HIGH-sensitivity tensor gets a FINER grid (smaller quant error) than a LOW one
  (the mechanism, not a constant);
- uniform sensitivity reproduces the vendored uniform 127-level quant EXACTLY (the
  default-preserving guard);
- the STE gradient flows through (forward = quant, backward = identity).

Score-claim discipline (sister of cat_entropy_v2 / rate_surrogate): this changes
TRAINING DYNAMICS; NO score is asserted from its addition alone. The empirical
bit-spend proof is the paired A/B (uniform-QAT vs score-aware-QAT, same epoch budget)
measuring ``archive_bytes`` at equal ``d_seg/d_pose``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

_TARGET_LAYERS: tuple[type[nn.Module], ...] = (nn.Conv2d, nn.Linear)
_BASE_N_LEVELS = 127  # the vendored uniform INT8 level count (the default)


@dataclass(frozen=True)
class ScoreAwareQATConfig:
    """Configuration for sensitivity-weighted fake-quant.

    The level count for tensor ``t`` is
    ``n_levels_t = round(base * (min_ratio + (max_ratio - min_ratio) * w_t))``
    where ``w_t in [0,1]`` is the rank-normalized sensitivity. Defaults give a
    [0.5×, 1.0×] band around the base 127 levels: the LEAST-sensitive tensor gets
    ~64 levels (coarse → fewer brotli bytes), the MOST-sensitive gets the full 127
    (fine → boundary protected). With UNIFORM sensitivity (all ``w_t`` equal) the
    map collapses to a SINGLE level count for every tensor; when that single value
    is the base 127 (``uniform_falls_back_to_base=True``, the default), the result
    is BIT-IDENTICAL to the vendored uniform QAT (the default-preserving guard).
    """

    #: Floor multiplier on the base level count for the LEAST-sensitive tensor.
    min_level_ratio: float = 0.5
    #: Ceiling multiplier for the MOST-sensitive tensor (1.0 = the base 127 grid).
    max_level_ratio: float = 1.0
    #: Base INT8 level count (the vendored value; the codec's grid).
    base_n_levels: int = _BASE_N_LEVELS
    #: When sensitivity is uniform/absent, fall back to the EXACT base level count
    #: for every tensor (so an unconditioned score_aware_qat run == vendored QAT).
    uniform_falls_back_to_base: bool = True
    #: Smallest allowed per-tensor level count (a hard floor so a near-zero-
    #: sensitivity tensor never degenerates to a 1-level (all-zero) quant).
    min_abs_levels: int = 16


def _fake_quantize_n(tensor: torch.Tensor, n_levels: int) -> torch.Tensor:
    """Per-tensor symmetric fake-quant at ``n_levels`` (STE), mirroring the vendored
    ``losses.fake_quantize`` exactly but with a per-tensor level count.

    With ``n_levels == 127`` this is BIT-IDENTICAL to the vendored call (verified by
    ``test_n_levels_127_matches_vendored_fake_quantize``)."""
    ma = tensor.abs().max()
    scale = ma / n_levels if ma > 0 else torch.ones((), device=tensor.device, dtype=tensor.dtype)
    q = (tensor / scale).round().clamp(-n_levels, n_levels)
    return (q * scale - tensor).detach() + tensor


def _rank_normalize(values: torch.Tensor) -> torch.Tensor:
    """Rank-normalize a 1-D sensitivity vector to ``[0, 1]`` (0 = least sensitive, 1
    = most). Returns all-0.5 when the values are (near-)uniform so the level map
    collapses to a single value (the default-preserving path).

    Rank (not raw magnitude) is robust to the heavy-tailed scale of
    ``||∂S/∂w_t||`` across tensors of very different numel."""
    n = values.numel()
    if n == 0:
        return values.new_zeros(0)
    if n == 1:
        return values.new_full((1,), 0.5)
    vmin = values.min()
    vmax = values.max()
    if (vmax - vmin).abs().item() < 1e-30:
        # Uniform sensitivity → all 0.5 (collapses the level map to one value).
        return values.new_full((n,), 0.5)
    order = torch.argsort(values)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(n, device=values.device, dtype=values.dtype)
    return ranks / (n - 1)


def per_tensor_levels_from_sensitivity(
    sensitivity: dict[str, float] | None,
    tensor_names: list[str],
    config: ScoreAwareQATConfig | None = None,
) -> dict[str, int]:
    """Map per-tensor sensitivity → per-tensor INT8 level count (4a, water-filling).

    ``sensitivity`` maps tensor name → ``||∂S/∂w_t||`` (an EMA scalar). Tensors not in
    the map (or ``sensitivity is None``) get the base level count. The returned level
    counts are higher for HIGH-sensitivity tensors (finer grid, boundary protected)
    and lower for LOW-sensitivity tensors (coarser grid, fewer brotli bytes).

    DEFAULT-PRESERVING: ``sensitivity is None`` OR uniform sensitivity with
    ``uniform_falls_back_to_base`` returns ``base_n_levels`` for EVERY tensor → the
    quant is bit-identical to the vendored uniform QAT.
    """
    cfg = config or ScoreAwareQATConfig()
    if not tensor_names:
        return {}
    if sensitivity is None:
        return {name: cfg.base_n_levels for name in tensor_names}

    # Build the per-tensor sensitivity vector (missing → 0, the least sensitive).
    vals = torch.tensor(
        [float(sensitivity.get(name, 0.0)) for name in tensor_names],
        dtype=torch.float64,
    )
    w = _rank_normalize(vals)  # in [0,1], all-0.5 if uniform
    # If uniform (all 0.5) and the fall-back flag is set, use the base level count
    # exactly (the default-preserving guard).
    is_uniform = bool((w == 0.5).all().item())
    if is_uniform and cfg.uniform_falls_back_to_base:
        return {name: cfg.base_n_levels for name in tensor_names}

    levels: dict[str, int] = {}
    for name, wi in zip(tensor_names, w.tolist(), strict=True):
        ratio = cfg.min_level_ratio + (cfg.max_level_ratio - cfg.min_level_ratio) * wi
        n = int(round(cfg.base_n_levels * ratio))
        n = max(cfg.min_abs_levels, min(cfg.base_n_levels, n))
        levels[name] = n
    return levels


def apply_score_aware_qat(
    decoder: nn.Module,
    sensitivity: dict[str, float] | None,
    config: ScoreAwareQATConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Replace Conv2d/Linear weights with SENSITIVITY-WEIGHTED fake-quantized versions
    IN PLACE (STE), returning the dict of originals to restore (same contract as the
    vendored ``apply_qat`` / ``restore_qat`` pair).

    DEFAULT-PRESERVING: ``sensitivity is None`` quantizes EVERY tensor at the base 127
    levels — BIT-IDENTICAL to ``losses.apply_qat`` (verified by
    ``test_apply_score_aware_qat_none_sensitivity_matches_vendored``). With a real
    sensitivity map, high-sensitivity tensors get a finer grid (protected) and
    low-sensitivity tensors a coarser grid (fewer brotli bytes).

    Usage (mirrors the vendored pattern in ``driver._train_one_epoch``)::

        originals = apply_score_aware_qat(decoder, sensitivity, cfg)
        decoded = decoder(...)            # forward through the SAME score-domain loss
        restore_score_aware_qat(decoder, originals)
    """
    cfg = config or ScoreAwareQATConfig()
    names = [
        name
        for name, mod in decoder.named_modules()
        if isinstance(mod, _TARGET_LAYERS) and hasattr(mod, "weight") and mod.weight is not None
    ]
    levels = per_tensor_levels_from_sensitivity(sensitivity, names, cfg)
    originals: dict[str, torch.Tensor] = {}
    for name, mod in decoder.named_modules():
        if name in levels:
            originals[name] = mod.weight.data.clone()
            mod.weight.data = _fake_quantize_n(mod.weight.data, levels[name])
    return originals


def restore_score_aware_qat(
    decoder: nn.Module, originals: dict[str, torch.Tensor]
) -> None:
    """Restore the pre-QAT weights (same contract as the vendored ``restore_qat``)."""
    for name, mod in decoder.named_modules():
        if name in originals:
            mod.weight.data = originals[name]


def accumulate_tensor_sensitivity(
    decoder: nn.Module,
    sensitivity_ema: dict[str, float],
    *,
    decay: float = 0.99,
) -> dict[str, float]:
    """Update the per-tensor sensitivity EMA from the CURRENT ``w.grad`` magnitudes
    (called after the score-domain ``loss.backward()`` populates the grads).

    ``s_t = ||∂S/∂w_t||`` is the L2 norm of the tensor's gradient. The EMA smooths the
    early-training noise (CLAUDE.md "Sensitivity estimate noise" risk in the memo). A
    tensor with no grad this step keeps its prior EMA. Returns the updated dict (also
    mutates it in place)."""
    for name, mod in decoder.named_modules():
        if isinstance(mod, _TARGET_LAYERS) and hasattr(mod, "weight") and mod.weight is not None:
            g = mod.weight.grad
            if g is None:
                continue
            s = float(g.detach().norm().item())
            prior = sensitivity_ema.get(name)
            sensitivity_ema[name] = s if prior is None else decay * prior + (1.0 - decay) * s
    return sensitivity_ema


# Score-claim discipline metadata.
SCORE_CLAIM = False
PROMOTION_ELIGIBLE = False
READY_FOR_EXACT_EVAL_DISPATCH = False

__all__ = [
    "ScoreAwareQATConfig",
    "per_tensor_levels_from_sensitivity",
    "apply_score_aware_qat",
    "restore_score_aware_qat",
    "accumulate_tensor_sensitivity",
    "SCORE_CLAIM",
    "PROMOTION_ELIGIBLE",
    "READY_FOR_EXACT_EVAL_DISPATCH",
]
