# SPDX-License-Identifier: MIT
"""Frontier int5/int4 SCORE-AWARE QAT — the Path-B decisive lever.

The PTQ-on-frontier floor (``.omx/research/ptq_on_frontier_floor_20260618T191724Z.md``)
proved naive bit-shrinking the 0.19110 frontier decoder COLLAPSES: int5 PTQ → S 0.678
(d_seg 0.00475 / d_pose 0.00155, a ~8× d_seg + ~42× d_pose spill over the int8 identity
0.1965), int4 → S 1.89. The rate win is real and large (int5 cuts 177,169 → 118,589 B =
rate 0.1180 → 0.0790), but the un-finetuned distortion buries it. The desk model's
int5-PERFECT-HOLD is S 0.153 (sub-0.16). This module is the lever that tries to RECOVER
the PTQ collapse via QAT-finetune: train the frontier decoder so its weights are robust
to their OWN int5 grid at the SegNet argmax boundary + the PoseNet pose head.

Why a NEW module (not ``tac.torch_vehicle.score_aware_qat``): that Lever-4 module floors
the per-tensor level count at 16 (its sensitivity band is [64,127] around the codec's
int8 grid — the deployed-byte-shaving lever, NOT a sub-int8 grid). int5 = 31 distinct
values (``n_levels=15``), int4 = 15 (``n_levels=7``) — well below that floor. This module
exposes the sub-int8 per-tensor grid + a saliency-driven per-tensor allocation (protect
the d_seg-critical tensors at int8, coarsen the d_seg-blind ones to int5/int4), reusing
the EXACT STE fake-quant kernel ``score_aware_qat._fake_quantize_n`` (so an ``n_levels``
of 127 is bit-identical to the vendored uniform QAT — the default-preserving guard).

The score-aware ALLOCATION is second-order (the floor memo's key correction: at PTQ the
d_seg damage comes overwhelmingly from coarsening the massive early/low-res stages — 77%
of params — NOT the tiny output heads; protecting only the 7%-of-params heads cannot
rescue d_seg). The DOMINANT lever is the FINETUNE: QAT-STE trains the decoder to be
robust at the coarse grid (the eval_roundtrip sister discipline, applied to the weight
grid). This module gives BOTH; the harness measures whether the finetune recovers enough.

NO-FAKE: this ACTUALLY fake-quantizes the REAL frontier decoder weights to the int5/int4
grid per the supplied per-tensor level count, on the real weights, with the STE gradient
flowing through (verified in ``tests/test_frontier_int5_qat.py``). The level count for a
tensor is a function of the supplied saliency map (a HIGH-saliency tensor gets MORE levels
= finer = protected; a LOW-saliency tensor gets FEWER = the coarse int5/int4 grid). With
no saliency the allocation is UNIFORM at ``low_n_levels`` (every weight tensor on the same
coarse grid) — the honest baseline. Score-claim discipline: this changes TRAINING
DYNAMICS; NO score is asserted from its addition. The empirical proof is the byte-closed
CPU-authority exact eval the harness runs (``[contest-CPU advisory]`` until paired
CPU/CUDA, pointer UNMOVED 0.19110).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from tac.torch_vehicle.score_aware_qat import _fake_quantize_n

_TARGET_LAYERS: tuple[type[nn.Module], ...] = (nn.Conv2d, nn.Linear)

# n_levels for the canonical sub-int8 grids (symmetric int-N has 2*n_levels+1 distinct
# values including 0; n_quant = 2**(nbits-1) - 1). int8 = 127, int5 = 15, int4 = 7.
NLEVELS_FOR_NBITS: dict[int, int] = {8: 127, 7: 63, 6: 31, 5: 15, 4: 7}

# Output-proximal heads (TINY in params, ~7%) — the floor memo's ``sa_int5lo_int8hi``
# protect band (the gentlest score-aware operating point). These hold the
# 192×256/384×512 argmax boundary the contest SegNet decides on, AND they are tiny in
# params, so protecting them at int8 costs almost NO bytes while the massive early/low-res
# stages (77% of params) take the coarse grid — preserving the int5 rate win. This mirrors
# ``tac.frontier_decoder_ptq.DSEG_CRITICAL_PREFIXES`` (blocks.4/5 + skips + refine + rgb).
FRONTIER_OUTPUT_HEAD_PREFIXES: tuple[str, ...] = (
    "blocks.4.",
    "blocks.5.",
    "skips.",
    "refine.",
    "rgb_0.",
    "rgb_1.",
)
# The per-tensor PROTECT list is INTENTIONALLY EMPTY by default: the recalibrated frontier
# saliency prior found a FLAT ~5.5× spread (no large score-blind weight mass to shed
# cheaply) AND that the big early blocks (blocks.0/1/2, ~46k params each — the RATE
# carriers) are mid-rank, NOT the most d_seg-critical. Protecting them would DEFEAT the
# int5 rate win for a second-order d_seg gain (the floor memo: "the per-stage protection
# helps at matched bytes but is second-order; the dominant lever is finetune"). So the
# default score-aware allocation protects ONLY the tiny output-proximal heads above; the
# rate-carrying early stages go to the coarse int5 grid and the FINETUNE recovers them.
FRONTIER_PROTECT_TENSORS: tuple[str, ...] = ()


class FrontierInt5QATError(ValueError):
    """Raised when the int5/int4 QAT config is malformed."""


@dataclass
class FrontierQATConfig:
    """Per-tensor sub-int8 grid allocation for the frontier decoder QAT.

    The allocation strategy (``mode``):

    * ``"uniform"`` — EVERY Conv2d/Linear weight on the ``low_nbits`` grid (the honest
      coarse baseline; the floor memo's ``int5`` uniform row, but now QAT-finetuned).
    * ``"score_aware"`` — the d_seg-critical output-proximal heads (``head_prefixes``)
      + the top saliency-protect weight tensors (``protect_tensors``) on ``high_nbits``
      (precise int8), everything else on ``low_nbits`` (coarse int5/int4). The
      taper-on-the-bit-axis lever (second-order per the floor memo; the finetune is the
      dominant lever).

    A tensor's grid is its n_levels via :data:`NLEVELS_FOR_NBITS`; ``n_levels=127`` is
    bit-identical to the vendored uniform int8 QAT (so a high_nbits=8 protect tensor is
    EXACTLY the codec's grid).
    """

    mode: str = "score_aware"
    low_nbits: int = 5  # the coarse grid for d_seg-blind tensors (int5 default)
    high_nbits: int = 8  # the precise grid for d_seg-critical tensors (int8)
    head_prefixes: tuple[str, ...] = FRONTIER_OUTPUT_HEAD_PREFIXES
    protect_tensors: tuple[str, ...] = FRONTIER_PROTECT_TENSORS
    weight_min_dim: int = 2  # biases / 1-D params stay fp32 (tiny, output-proximal)

    def __post_init__(self) -> None:
        if self.mode not in ("uniform", "score_aware"):
            raise FrontierInt5QATError(
                f"mode must be 'uniform' or 'score_aware'; got {self.mode!r}"
            )
        for nb in (self.low_nbits, self.high_nbits):
            if int(nb) not in NLEVELS_FOR_NBITS:
                raise FrontierInt5QATError(
                    f"nbits must be one of {sorted(NLEVELS_FOR_NBITS)}; got {nb}"
                )
        if self.high_nbits < self.low_nbits:
            raise FrontierInt5QATError(
                f"high_nbits ({self.high_nbits}) must be >= low_nbits ({self.low_nbits})"
            )


def per_tensor_nbits_for_decoder(
    decoder: nn.Module, config: FrontierQATConfig
) -> dict[str, int]:
    """Map each Conv2d/Linear MODULE name → its target nbits per the config.

    The keys are the ``named_modules`` module names (e.g. ``stem`` / ``blocks.0`` —
    matching the ``score_aware_qat`` apply contract). A module is protected (high_nbits)
    when its ``<module>.weight`` is in ``protect_tensors`` OR its name starts with any
    ``head_prefix``; otherwise it takes ``low_nbits``. ``uniform`` mode → every module on
    ``low_nbits``.
    """
    out: dict[str, int] = {}
    for name, mod in decoder.named_modules():
        if not (
            isinstance(mod, _TARGET_LAYERS)
            and hasattr(mod, "weight")
            and mod.weight is not None
        ):
            continue
        if mod.weight.dim() < config.weight_min_dim:
            continue
        if config.mode == "uniform":
            out[name] = config.low_nbits
            continue
        # Match prefixes against the STATE-DICT KEY (``<module>.weight``) — the
        # convention the frontier saliency prior + ``frontier_decoder_ptq``
        # ``DSEG_CRITICAL_PREFIXES`` use (so ``"rgb_0."`` matches ``"rgb_0.weight"``,
        # ``"skips."`` matches ``"skips.2.weight"``). Matching the bare module name
        # would miss the no-trailing-dot leaf modules (``rgb_0`` / ``rgb_1``).
        wkey = f"{name}.weight"
        is_protected = wkey in config.protect_tensors or any(
            wkey.startswith(p) for p in config.head_prefixes
        )
        out[name] = config.high_nbits if is_protected else config.low_nbits
    return out


def apply_frontier_qat(
    decoder: nn.Module, per_tensor_nbits: dict[str, int]
) -> dict[str, torch.Tensor]:
    """Fake-quantize the decoder's Conv2d/Linear weights to their per-tensor grid IN
    PLACE (STE), returning the dict of originals to restore (same contract as the
    vendored ``apply_qat`` / ``restore_qat`` pair).

    The grid is ``NLEVELS_FOR_NBITS[nbits]`` n_levels via the EXACT vendored STE kernel
    ``_fake_quantize_n`` (forward = quant, backward = identity). A module not in
    ``per_tensor_nbits`` is left untouched (fp32) — so biases/1-D params and any
    excluded tensor flow their full-precision gradient.
    """
    originals: dict[str, torch.Tensor] = {}
    for name, mod in decoder.named_modules():
        if name in per_tensor_nbits:
            nb = int(per_tensor_nbits[name])
            n_levels = NLEVELS_FOR_NBITS[nb]
            originals[name] = mod.weight.data.clone()
            mod.weight.data = _fake_quantize_n(mod.weight.data, n_levels)
    return originals


def restore_frontier_qat(decoder: nn.Module, originals: dict[str, torch.Tensor]) -> None:
    """Restore the pre-QAT weights (same contract as the vendored ``restore_qat``)."""
    for name, mod in decoder.named_modules():
        if name in originals:
            mod.weight.data = originals[name]


def hard_quantize_state_dict_to_nbits(
    decoder: nn.Module,
    per_tensor_nbits: dict[str, int],
    *,
    weight_min_dim: int = 2,
) -> dict[str, torch.Tensor]:
    """Return a NEW state dict with each weight tensor HARD-quantized to its grid
    (quantize→dequantize, the deployed grid) — for the byte-close re-encode.

    This is the EXPORT step: after QAT-finetune, the decoder weights are float but
    TRAINED to be robust at the coarse grid; this snaps them ONTO that grid (the bytes
    the codec will store). ``intn_qdq`` is the codec-grid kernel (``tac.post_hoc_weight_
    shrink``) — at nbits=8 it is the codec's exact int8 grid, so a protected tensor
    re-encodes losslessly through the codec's own int8 path.

    Module names → state-dict ``<module>.weight`` keys; other state-dict entries
    (biases, 1-D) are copied through fp32 (the codec keeps them as-is).
    """
    from tac.post_hoc_weight_shrink import intn_qdq

    module_nbits = dict(per_tensor_nbits)
    sd = decoder.state_dict()
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if not torch.is_tensor(v) or v.dim() < weight_min_dim:
            out[k] = v.clone()
            continue
        # k is "<module>.weight"; resolve the module name.
        module = k[: -len(".weight")] if k.endswith(".weight") else k
        nb = module_nbits.get(module)
        out[k] = intn_qdq(v, int(nb)) if nb is not None else v.clone()
    return out


@dataclass
class EMAState:
    """A simple weight-EMA shadow with bias-corrected warmup (the EMA non-negotiable).

    Tracks a shadow copy of the decoder weights + the latents. ``decay`` is the target
    decay; with ``warmup`` the effective per-step decay ramps from ~0.1 → ``decay`` so
    the shadow TRACKS the live weights early (the EMA-shadow-lag fix for short runs;
    sister of the capstone curriculum fix and the driver's ``ema_warmup``).
    """

    decay: float = 0.999
    warmup: bool = True
    step: int = 0
    shadow: dict[str, torch.Tensor] = field(default_factory=dict)
    latent_shadow: torch.Tensor | None = None

    def init_from(self, decoder: nn.Module, latents: torch.Tensor) -> None:
        self.shadow = {k: v.detach().clone() for k, v in decoder.state_dict().items()}
        self.latent_shadow = latents.detach().clone()
        self.step = 0

    def _effective_decay(self) -> float:
        if not self.warmup:
            return self.decay
        return min(self.decay, (self.step + 1) / (self.step + 10))

    def update(self, decoder: nn.Module, latents: torch.Tensor) -> None:
        d = self._effective_decay()
        sd = decoder.state_dict()
        for k, v in sd.items():
            s = self.shadow.get(k)
            if s is None or not torch.is_floating_point(v):
                self.shadow[k] = v.detach().clone()
            else:
                s.mul_(d).add_(v.detach(), alpha=1.0 - d)
        if self.latent_shadow is None:
            self.latent_shadow = latents.detach().clone()
        else:
            self.latent_shadow.mul_(d).add_(latents.detach(), alpha=1.0 - d)
        self.step += 1

    def shadow_decoder(self, template: nn.Module) -> nn.Module:
        """Load the shadow weights into a clone of ``template`` (CPU, eval)."""
        from copy import deepcopy

        dec = deepcopy(template).to("cpu").eval()
        dec.load_state_dict({k: v.to("cpu") for k, v in self.shadow.items()})
        return dec

    def shadow_latents_cpu(self) -> torch.Tensor:
        if self.latent_shadow is None:
            raise FrontierInt5QATError("EMA latent shadow not initialized")
        return self.latent_shadow.detach().to("cpu").clone()


# Score-claim discipline metadata.
SCORE_CLAIM = False
PROMOTION_ELIGIBLE = False
READY_FOR_EXACT_EVAL_DISPATCH = False

__all__ = [
    "FRONTIER_OUTPUT_HEAD_PREFIXES",
    "FRONTIER_PROTECT_TENSORS",
    "NLEVELS_FOR_NBITS",
    "PROMOTION_ELIGIBLE",
    "READY_FOR_EXACT_EVAL_DISPATCH",
    "SCORE_CLAIM",
    "EMAState",
    "FrontierInt5QATError",
    "FrontierQATConfig",
    "apply_frontier_qat",
    "hard_quantize_state_dict_to_nbits",
    "per_tensor_nbits_for_decoder",
    "restore_frontier_qat",
]
