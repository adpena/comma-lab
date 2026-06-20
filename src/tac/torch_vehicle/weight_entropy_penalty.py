# SPDX-License-Identifier: MIT
"""Ballé end-to-end rate-distortion lever — entropy-penalized TRAINING of the
decoder weights (the un-integrated VCM/task-aware-compression term).

WHY THIS IS THE REAL RATE LEVER (the math)
------------------------------------------
The contest rate term is ``25·archive_bytes/37_545_489``. The decoder blob is the
dominant section, and its byte count is

    archive_bytes(decoder) ≈ Σ_tensor H(quantized weight symbols) · n_elem / 8 ,

where the symbols are exactly what the vendored codec
(``codec.quantize_state_dict`` → ``encode_decoder``) brotli-compresses: per
Conv2d/Linear ``weight`` tensor, the per-tensor symmetric INT8 grid

    scale  = max(|w|) / 127
    symbol = round(w / scale)  ∈ {-127, …, 127}.

The post-hoc coder (PR112 constriction) already sits at the lossless floor
(~7.999 bits/byte), so a BETTER CODER IS DEAD. The only way to lower the rate is
to lower ``H`` itself, which is set by TRAINING. The Ballé 2018 factorized-prior
rate term ``λ·E[−log2 p(w)]`` under a LEARNED prior pulls the weight-symbol
distribution toward low entropy → lower ``H`` → lower deployed byte floor. PR95
has only a weak MEMORYLESS shadow of this (``cat_entropy_v2`` / ``cat_lambda``);
this is the proper learned-prior, per-output-channel entropy model.

WHAT THIS MODULE DOES (NO FAKE — it ACTUALLY penalizes entropy)
---------------------------------------------------------------
For each Conv2d/Linear ``weight`` tensor on the decoder, instantiate a
per-output-channel :class:`tac.entropy_bottleneck.EntropyBottleneck` (Ballé
factorized logistic-CDF prior; the OUTPUT-CHANNEL axis ``weight.shape[0]`` is the
channel dim). ``rate_bits(decoder)`` runs each tensor's bottleneck on the
CODEC-GRID representation ``w / scale`` (so the modeled symbols ARE the integer
symbols the codec codes; ``scale`` is ``.detach()``-ed so the per-tensor scale is
NOT penalized — only the SHAPE of the integer-symbol distribution) and returns

  * ``total_bits``  — the expected codelength Σ_tensor bits_per_element · n_elem
    (in BITS) under the learned prior, and
  * ``rate_term``   — the same expectation mapped to the contest rate scale
    ``total_bits / 8 / 37_545_489 · 25`` (= the rate-term contribution if the
    learned-prior codelength were realized as archive bytes).

In ``training`` the bottleneck adds ``U(-0.5, 0.5)`` noise (Ballé's STE for
quantization) so the rate is a smooth, differentiable surrogate; in ``eval`` it
ROUNDS (the deployed integer symbols). The bottleneck params (loc / raw_scale /
raw_shape) are LEARNABLE so the prior adapts to the weight distribution — but
they must be added to the optimizer by the caller (the driver does this in
``_build_stage_runtime``); a stand-alone instance trains nothing.

SCORE-CLAIM DISCIPLINE (sister of ``cat_entropy_v2`` + ``rate_surrogate``)
--------------------------------------------------------------------------
This changes TRAINING DYNAMICS, NOT archive bytes directly. NO score is asserted
from its addition. The empirical bit-spend proof is the paired A/B (λ-on vs λ=0,
same epoch budget) measuring the REAL ``archive_bytes`` (the byte-closed codec)
at equal ``d_seg/d_pose`` — per CLAUDE.md "Bit-level deconstruction and entropy
discipline" + Catalog #304 (closed-form-rate-without-empirical-bit-spend-proof is
forbidden; the headline NO-FAKE test measures the ACTUAL post-codec weight
symbol-entropy descent, not the surrogate).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tac.entropy_bottleneck import EntropyBottleneck

# The contest rate-term constants (S = 100·d_seg + sqrt(10·d_pose) + 25·bytes/N).
# Used ONLY to map the expected codelength (bits) onto the score's rate scale so
# the lever's magnitude is comparable to d_seg/d_pose when choosing λ; it is NOT a
# score claim (the real bytes come from the byte-closed codec).
_TOTAL_VIDEO_BYTES = 37_545_489
_RATE_COEFF = 25.0
_CODEC_N_QUANT = 127  # the vendored codec's per-tensor symmetric INT8 grid (codec.N_QUANT)


def _is_coded_weight_module(mod: nn.Module) -> bool:
    """True iff ``mod`` is a Conv2d/Linear whose ``weight`` the codec codes — the
    EXACT membership ``cat_entropy_v2`` / ``quantize_state_dict`` iterate (the codec
    codes Conv2d/Linear weights; biases ride a separate path). Kept as one predicate
    so the penalty's tensor set provably matches the codec's coded set."""
    return isinstance(mod, (nn.Conv2d, nn.Linear)) and hasattr(mod, "weight")


def _coded_weight_modules(decoder: nn.Module) -> list[tuple[str, nn.Module]]:
    """The ``(name, module)`` list of coded Conv2d/Linear weight tensors, in
    ``named_modules`` order (deterministic; matches the codec's tensor order)."""
    return [(n, m) for n, m in decoder.named_modules() if _is_coded_weight_module(m)]


class WeightEntropyPenalty(nn.Module):
    """Per-decoder-weight-tensor Ballé factorized-prior weight-entropy penalty.

    Construct ONCE per run from the decoder (the bottlenecks are sized to each
    coded weight tensor's output-channel count); register it on the driver and add
    ``self.parameters()`` to the optimizer so the learned priors train. Each forward
    (:meth:`rate_bits`) returns the expected codelength of the decoder's CURRENT
    weights under the learned prior — the differentiable rate term added to the loss.

    Args:
        decoder: the decoder whose Conv2d/Linear weights will be penalized. The
            bottlenecks are keyed by ``named_modules`` name + sized to
            ``weight.shape[0]`` (the output-channel axis = the EntropyBottleneck
            channel dim). A decoder with a DIFFERENT architecture (channel counts)
            needs its own penalty instance (the per-tensor channel sizes are baked
            in at construction; a mismatch raises at :meth:`rate_bits`).
        init_scale: initial logistic scale for each per-channel prior. The default
            10.0 starts broad (does not over-penalize the codec-grid symbols, which
            span ~±127 early) and tightens as the prior learns.
    """

    def __init__(self, decoder: nn.Module, *, init_scale: float = 10.0):
        super().__init__()
        coded = _coded_weight_modules(decoder)
        if not coded:
            raise ValueError(
                "WeightEntropyPenalty: the decoder exposes NO Conv2d/Linear weight "
                "tensors to penalize (the codec codes Conv2d/Linear weights)."
            )
        # One bottleneck per coded weight tensor, keyed by a sanitized module name
        # (``ModuleDict`` keys cannot contain '.'); store the sanitize map so
        # rate_bits can re-pair each bottleneck with its module by name.
        self._bottlenecks = nn.ModuleDict()
        self._key_for_name: dict[str, str] = {}
        for name, mod in coded:
            out_ch = int(mod.weight.shape[0])
            key = self._sanitize(name)
            self._bottlenecks[key] = EntropyBottleneck(out_ch, init_scale=init_scale)
            self._key_for_name[name] = key
        self._last_total_bits: Tensor | None = None

    @staticmethod
    def _sanitize(name: str) -> str:
        """ModuleDict keys may not contain '.'; map ``blocks.0.conv`` → ``blocks__0__conv``.
        The empty top-level name (a bare Conv2d/Linear decoder) maps to ``__root__``."""
        return name.replace(".", "__") if name else "__root__"

    @staticmethod
    def _codec_grid_representation(weight: Tensor) -> Tensor:
        """The codec-grid representation ``w / scale`` of a weight tensor, where
        ``scale = max(|w|) / 127`` is the EXACT vendored codec per-tensor scale
        (``codec.quantize_state_dict``). The result is the continuous pre-round
        integer-grid value whose ``round()`` is the coded symbol ∈ {-127, …, 127}.

        ``scale`` is ``.detach()``-ed: the penalty must NOT push the per-tensor
        scale around (the scale rides 4 bytes of metadata, not the entropy-bearing
        symbol stream); it penalizes ONLY the SHAPE of the integer-symbol
        distribution. A degenerate all-zero tensor (max|w|≈0) returns zeros (the
        codec stores it as all-zero symbols — entropy 0)."""
        ma = weight.detach().abs().max()
        if ma.item() < 1e-12:
            return torch.zeros_like(weight)
        scale = (ma / _CODEC_N_QUANT).detach()
        return weight / scale

    def rate_bits(self, decoder: nn.Module) -> tuple[Tensor, Tensor]:
        """Return ``(total_bits, rate_term)`` for the decoder's CURRENT weights under
        the learned per-channel priors.

        ``total_bits`` is the expected codelength Σ_tensor (bits/element · n_elem) in
        BITS — the differentiable rate surrogate. ``rate_term`` maps it onto the
        contest rate scale ``total_bits / 8 / 37_545_489 · 25`` (comparable to the
        d_seg/d_pose loss terms when picking λ). Both carry gradient to the decoder
        weights (so the optimizer can descend the entropy) AND to the bottleneck
        priors (so the priors adapt).

        Each weight tensor is reshaped to ``(1, out_ch, in·kh·kw)`` (the
        EntropyBottleneck's ``(N, C, *)`` contract; the OUTPUT-CHANNEL axis is the
        channel dim) and fed its codec-grid representation. In ``training`` the
        bottleneck adds ``U(-0.5, 0.5)`` noise (Ballé STE); in ``eval`` it rounds.
        """
        coded = dict(_coded_weight_modules(decoder))
        total_bits = torch.zeros((), device=next(decoder.parameters()).device)
        total_bits = total_bits.to(torch.float32)
        n_seen = 0
        for name, mod in coded.items():
            key = self._key_for_name.get(name)
            if key is None:
                # A tensor the penalty was not constructed for (architecture drift):
                # refuse rather than silently skip (a silent skip would under-penalize).
                raise ValueError(
                    f"WeightEntropyPenalty has no bottleneck for coded tensor {name!r}; "
                    "the penalty must be constructed from the SAME decoder architecture "
                    "(channel counts) it scores. Build a fresh penalty for this decoder."
                )
            bottleneck: EntropyBottleneck = self._bottlenecks[key]  # type: ignore[assignment]
            w = mod.weight
            out_ch = int(w.shape[0])
            grid = self._codec_grid_representation(w)
            y = grid.reshape(1, out_ch, -1)  # (N=1, C=out_ch, *)
            _y_hat, bits_per_element = bottleneck(y)
            total_bits = total_bits + bits_per_element * float(w.numel())
            n_seen += 1
        if n_seen == 0:  # pragma: no cover - guarded at construction
            raise ValueError("WeightEntropyPenalty.rate_bits found no coded weight tensors")
        self._last_total_bits = total_bits
        rate_term = total_bits / 8.0 / float(_TOTAL_VIDEO_BYTES) * _RATE_COEFF
        return total_bits, rate_term

    def last_total_bits(self) -> Tensor:
        """The ``total_bits`` from the most recent :meth:`rate_bits` (0 before any)."""
        if self._last_total_bits is None:
            return next(self.parameters()).sum() * 0.0
        return self._last_total_bits


@torch.no_grad()
def measure_decoder_weight_symbol_entropy(decoder: nn.Module) -> float:
    """The REAL (measured, codec-faithful) mean weight symbol-entropy in BITS/WEIGHT,
    averaged across the decoder's coded Conv2d/Linear tensors, size-weighted — the
    NO-FAKE headline metric the λ>0 lever must LOWER.

    This is NOT the surrogate: it HARD-quantizes each weight tensor exactly as the
    codec does (``q = round(w / scale).clamp(-127, 127)`` with ``scale =
    max(|w|)/127``) and computes the empirical Shannon entropy of the integer
    symbols (an exact histogram, no soft assignment, no learned prior). It is the
    quantity ``H(symbols)`` that sets the codec's byte floor. A lever that lowers the
    Ballé surrogate but NOT this measured entropy would be a fake; the smoke/test
    assert THIS descends.
    """
    total_numel = 0
    weighted_entropy = 0.0
    for _name, mod in _coded_weight_modules(decoder):
        w = mod.weight.detach()
        ma = w.abs().max()
        if ma.item() < 1e-12:
            continue
        scale = ma / _CODEC_N_QUANT
        q = (w / scale).round().clamp(-_CODEC_N_QUANT, _CODEC_N_QUANT).to(torch.int64).flatten()
        numel = int(q.numel())
        # Exact symbol histogram → Shannon entropy in bits.
        counts = torch.bincount(q + _CODEC_N_QUANT, minlength=2 * _CODEC_N_QUANT + 1).to(
            torch.float64
        )
        probs = counts / counts.sum()
        nz = probs[probs > 0]
        ent = float(-(nz * torch.log2(nz)).sum().item())
        weighted_entropy += numel * ent
        total_numel += numel
    return weighted_entropy / max(total_numel, 1)


__all__ = [
    "WeightEntropyPenalty",
    "measure_decoder_weight_symbol_entropy",
]
