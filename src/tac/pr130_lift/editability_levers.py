"""B2E train-for-editability levers for the PR130-lift SemanticTokenRenderer.

WHAT THIS IS
------------
The ns1 audit (``.omx/research/ddm_ns1_negative_signal_audit_and_missing_patterns_20260816.md``,
sha256 91741c062c38ab88ce7e225921a0024d6dc45dacc858658e02ee83ff08b2dba0) measured that
pose brittleness on the shipped hv1 vehicle is a TRAINED property: every post-hoc
semantic-weight edit (mixed q3/q4, FiLM row prune, the keep75-minus-keep87 marginal)
was REFUSED with pose damaged 3.8-5.0x.  Sensitivity is ~94x anisotropic and the
pose-critical subspace is LOCATED at the ``blocks.N.film.weight`` rows.

These levers train the renderer to TOLERATE those exact edits, so the refused byte
pools become harvestable.  They are the mechanism half of ns1 P1.

OBJECT SCOPE -- READ BEFORE WIRING
---------------------------------
These levers act on the **SemanticTokenRenderer** (38-tensor state:
``token_embed.weight``, ``frame_embed.weight``, ``coord_mix.*``,
``blocks.{0..3}.{dw,pw,norm,film}.*``, ``head.*``) -- the object whose weights the
mp2 edits actually touched, carried in the archive member's ``semantic`` section.

They do NOT act on the cl1 HPAC token model (``conv_a``, ``conv_b1``, ``conv_b2``,
``conv_past``, ``spm_dw``, ``spm_pw``, 5-class ``head``) carried in the ``hpac``
section and trained by ``tools/train_ddm_cl1_hpac_capacity.py``.  Those are two
different learned objects in one archive; see the b2e landing memo for the measured
separation and the charter re-pin it forces.

BYTE-IDENTITY CONTRACT (binding)
--------------------------------
Every lever is default-off.  When a lever is off:

* no tensor is read, cloned, or written by that lever;
* **no random number is drawn** -- inactive levers never touch any RNG, so a run
  with all levers off consumes the identical RNG stream as the pre-lever trainer;
* the loss contribution is exactly ``0.0`` and is not added to the graph.

Active stochastic levers draw from a DEDICATED :class:`torch.Generator` seeded from
the run seed, never from the global/default stream.  This keeps the base trainer's
own sampling bit-identical whether or not a lever is on, which is what makes a
lever A/B a clean 2x2 rather than a confounded one.

NON-AUTHORITY
-------------
Nothing here produces or implies a score.  Admission is measured only by the
edit-replay harness (``experiments/ddm_b2e_edit_replay_admission.py``) and, for any
promotable claim, by the exact contest evaluator on the exact archive bytes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

# fp16's smallest positive (subnormal) value, 2**-24 = 5.960464e-08.  A positive
# floor applied in fp32 BELOW this rounds to EXACTLY 0.0 when narrowed to fp16,
# silently re-opening the divide-by-zero / zero-scale the floor exists to close.
# The floor must therefore be re-applied AFTER the cast.
_FP16_MIN_POSITIVE = 5.960464477539063e-08

__all__ = [
    "SELECTED_MIXED_Q3_NAMES",
    "FILM_ROW_FAMILY",
    "POSE_CRITICAL_TENSORS",
    "DEFAULT_FILM_CRITICAL_MULTIPLIER",
    "EditabilityLeverConfig",
    "EditabilityLevers",
    "LeverError",
    "deployed_fake_quant",
    "mixed_bit_allocation",
    "film_row_order",
]


class LeverError(RuntimeError):
    """A lever was configured or applied outside its declared contract."""


# ---------------------------------------------------------------------------
# Pinned edit-target sets.
#
# These MIRROR the shipped constants; they are re-declared (not imported) so this
# module carries no import dependency on the ``experiments`` package, and are
# VERIFIED against the shipped definitions by
# ``src/tac/tests/test_editability_levers.py::test_constants_match_shipped_sets``.
# If that test fails, the shipped edit set moved and these must be re-derived --
# never silently patched.
# ---------------------------------------------------------------------------

#: Tensors the mp2 mixed-precision candidate coarsened to 3 bits (rest stay at 4).
#: Source: ``experiments/ddm_sm3_semantic_representation.py::SELECTED_MIXED_Q3_NAMES``.
SELECTED_MIXED_Q3_NAMES: frozenset[str] = frozenset(
    {
        "frame_embed.weight",
        "blocks.1.film.weight",
        "blocks.2.film.weight",
        "blocks.3.film.weight",
    }
)

#: FiLM weight rows the keep-ladders prune.
#: Source: ``experiments/ddm_sm3_semantic_representation.py::PRUNE_NAMES``.
FILM_ROW_FAMILY: frozenset[str] = frozenset(
    {
        "blocks.1.film.weight",
        "blocks.2.film.weight",
        "blocks.3.film.weight",
    }
)

#: The measured pose-critical subspace (ns1 section A).  A 0.28% relative
#: perturbation here did the damage of a 28.5% perturbation in ``frame_embed`` --
#: a ~94x sensitivity spread -- so robustness pressure is applied HERE first.
POSE_CRITICAL_TENSORS: frozenset[str] = FILM_ROW_FAMILY

#: Default extra perturbation applied to the pose-critical tensors under F1.
#: DERIVED, not tuned: ns1 measured Δd_pose per unit relative perturbation at
#: 0.192 (keep87, FiLM) vs 0.00205 (q3/q4, frame_embed-dominated) = 93.7x.  We
#: up-weight by sqrt(93.7) ~= 9.68 rather than the full ratio: full-ratio
#: weighting equalises measured DAMAGE, which at this anisotropy collapses the
#: FiLM rows outright; the square root equalises damage in the metric where
#: perturbation enters quadratically, which is the regime the quantiser noise
#: actually lives in.  Swept-not-assumed is the arm-matrix follow-on.
DEFAULT_FILM_CRITICAL_MULTIPLIER: float = math.sqrt(93.7)


def mixed_bit_allocation(
    names: Iterable[str],
    *,
    q3_names: Iterable[str] = SELECTED_MIXED_Q3_NAMES,
    low_bits: int = 3,
    high_bits: int = 4,
) -> dict[str, int]:
    """Return the exact mp2 mixed q3/q4 bit map over ``names``.

    Mirrors ``ddm_mp2_mixed_precision_receiver_close`` line 839:
    ``3 if name in SELECTED_MIXED_Q3_NAMES else 4``.
    """
    if not 2 <= low_bits <= 8 or not 2 <= high_bits <= 8:
        raise LeverError("semantic bit depths must be in [2, 8]")
    selected = frozenset(q3_names)
    return {name: (low_bits if name in selected else high_bits) for name in names}


def film_row_order(value: torch.Tensor) -> list[int]:
    """Row order used by the shipped keep-ladders: descending L2^2, ties by index.

    Mirrors ``ddm_mp2_mixed_precision_receiver_close`` lines 528-531 exactly, so a
    lever that protects/drops "the rows keep87 would prune" targets the same rows
    the receiver would.
    """
    if value.ndim < 2:
        raise LeverError("row order is only defined for tensors with ndim >= 2")
    rows = int(value.shape[0])
    flat = value.detach().cpu().float().reshape(rows, -1)
    norms = flat.square().sum(dim=1)
    return sorted(range(rows), key=lambda index: (-float(norms[index]), index))


def deployed_fake_quant(name: str, value: torch.Tensor, bits: int) -> torch.Tensor:
    """Differentiable straight-through fake-quant matching the DEPLOYED quantiser.

    Numerically identical (in the forward direction) to
    ``ddm_sd1_semantic_rd_curve.quantized_tensor``:

    * ``ndim < 2`` -> stored as fp16;
    * otherwise per-output-row (or per-embedding-column) absmax scale, cast to
      fp16, codes rounded and clamped to ``+/-(2^(bits-1) - 1)``.

    The straight-through estimator passes the identity gradient, so training sees
    the real deployed grid in the forward pass while remaining trainable.  This is
    the weight-space analogue of the eval_roundtrip non-negotiable.
    """
    if bits < 2 or bits > 8:
        raise LeverError(f"bits must be in [2, 8], got {bits}")
    if value.ndim < 2:
        stored = value.to(torch.float16).float()
        return value + (stored - value).detach()

    limit = (1 << (bits - 1)) - 1
    embedding = name.endswith("embed.weight")
    reduce_dims = (
        tuple(range(value.ndim - 1)) if embedding else tuple(range(1, value.ndim))
    )
    scale = value.detach().abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-8)
    scale = (scale / limit).to(torch.float16).clamp(min=_FP16_MIN_POSITIVE).float()
    codes = torch.round(value / scale).clamp(-limit, limit)
    restored = codes * scale
    return value + (restored - value).detach()


@dataclass(frozen=True)
class EditabilityLeverConfig:
    """Fully inert by default.  Every field off => byte-identical training."""

    # --- F1: weight-perturbation robustness -------------------------------
    weight_perturb_robustness: float = 0.0
    """Relative perturbation sigma applied to semantic weights, as a fraction of
    each tensor's own quantiser step.  ``0.0`` disables F1 entirely."""

    weight_perturb_shape: str = "quantization"
    """``"quantization"`` (uniform over one quantiser step -- matches the actual
    coarsening the edits apply) or ``"gaussian"``."""

    film_critical_multiplier: float = DEFAULT_FILM_CRITICAL_MULTIPLIER
    """Extra F1 pressure on :data:`POSE_CRITICAL_TENSORS`.  Inert when F1 is off."""

    # --- F2: mixed q3/q4 weight QAT ----------------------------------------
    weight_qat_q3q4: bool = False
    """Train through the EXACT mp2 mixed q3/q4 grid instead of uniform q4."""

    weight_qat_low_bits: int = 3
    weight_qat_high_bits: int = 4

    # --- F3: FiLM structured row dropout -----------------------------------
    film_row_dropout: float = 0.0
    """Probability of dropping a whole FiLM output row per step (inverted
    dropout, so expectation is preserved).  ``0.0`` disables F3."""

    film_row_dropout_protect_top: int = 0
    """Number of highest-norm rows (per :func:`film_row_order`) never dropped.
    ``0`` protects nothing."""

    # --- F4: carrier rank penalty ------------------------------------------
    carrier_rank_penalty: float = 0.0
    """Weight of the nuclear-norm surrogate on :attr:`carrier_tensors`.  ``0.0``
    disables F4 and contributes exactly ``0.0`` to the loss."""

    carrier_tensors: tuple[str, ...] = ()
    """Names the rank penalty applies to.  Empty => F4 inert even if weighted."""

    # --- shared -------------------------------------------------------------
    seed: int = 20260816
    """Seed for the DEDICATED lever RNG stream (never the global stream)."""

    def __post_init__(self) -> None:
        if self.weight_perturb_robustness < 0.0:
            raise LeverError("--weight-perturb-robustness must be >= 0")
        if self.weight_perturb_shape not in ("quantization", "gaussian"):
            raise LeverError(
                "weight_perturb_shape must be 'quantization' or 'gaussian'"
            )
        if self.film_critical_multiplier < 0.0:
            raise LeverError("film_critical_multiplier must be >= 0")
        if not 0.0 <= self.film_row_dropout < 1.0:
            raise LeverError("--film-row-dropout must be in [0, 1)")
        if self.film_row_dropout_protect_top < 0:
            raise LeverError("film_row_dropout_protect_top must be >= 0")
        if self.carrier_rank_penalty < 0.0:
            raise LeverError("--carrier-rank-penalty must be >= 0")
        for bits in (self.weight_qat_low_bits, self.weight_qat_high_bits):
            if not 2 <= bits <= 8:
                raise LeverError("weight QAT bit depths must be in [2, 8]")

    # -- activation predicates (the "off is a tracked state" surface) --------

    @property
    def f1_active(self) -> bool:
        return self.weight_perturb_robustness > 0.0

    @property
    def f2_active(self) -> bool:
        return bool(self.weight_qat_q3q4)

    @property
    def f3_active(self) -> bool:
        return self.film_row_dropout > 0.0

    @property
    def f4_active(self) -> bool:
        return self.carrier_rank_penalty > 0.0 and bool(self.carrier_tensors)

    @property
    def any_active(self) -> bool:
        return self.f1_active or self.f2_active or self.f3_active or self.f4_active

    def activation_ledger(self) -> dict[str, Any]:
        """Machine-readable per-lever state for run provenance.

        Satisfies the "off is a tracked, reasoned, surfaced state -- never a
        silent default" non-negotiable: every lever reports its state and the
        reason it is off.
        """
        return {
            "schema": "ddm_b2e_editability_lever_activation.v1",
            "any_active": self.any_active,
            "levers": {
                "F1_weight_perturb_robustness": {
                    "active": self.f1_active,
                    "sigma_in_quantiser_steps": self.weight_perturb_robustness,
                    "shape": self.weight_perturb_shape,
                    "film_critical_multiplier": self.film_critical_multiplier,
                    "reason_if_off": None if self.f1_active else "sigma == 0 (default)",
                },
                "F2_weight_qat_q3q4": {
                    "active": self.f2_active,
                    "low_bits": self.weight_qat_low_bits,
                    "high_bits": self.weight_qat_high_bits,
                    "q3_names": sorted(SELECTED_MIXED_Q3_NAMES),
                    "reason_if_off": None if self.f2_active else "flag unset (default)",
                },
                "F3_film_row_dropout": {
                    "active": self.f3_active,
                    "probability": self.film_row_dropout,
                    "protect_top": self.film_row_dropout_protect_top,
                    "row_family": sorted(FILM_ROW_FAMILY),
                    "reason_if_off": None if self.f3_active else "p == 0 (default)",
                },
                "F4_carrier_rank_penalty": {
                    "active": self.f4_active,
                    "weight": self.carrier_rank_penalty,
                    "tensors": list(self.carrier_tensors),
                    "reason_if_off": (
                        None
                        if self.f4_active
                        else "weight == 0 or no carrier tensors named (default)"
                    ),
                },
                "F5_gate_aware_conditioning": {
                    "active": False,
                    "state": "DECLARED_UNBUILT_FOLLOW_ON",
                    "reason_if_off": (
                        "requires the js8 gated application distribution, which is "
                        "not derivable from any receipt this module holds; see the "
                        "b2e landing memo NEXT_IF_RESUMED"
                    ),
                },
            },
            "seed": self.seed,
        }


class EditabilityLevers:
    """Applies the b2e levers to a SemanticTokenRenderer during training.

    Usage (caller guards on ``any_active`` so an all-off config is a no-op)::

        levers = EditabilityLevers(config)
        with levers.applied(model):
            loss = criterion(model(...), target)
        loss = loss + levers.rank_penalty(model)
    """

    def __init__(self, config: EditabilityLeverConfig) -> None:
        self._config = config
        self._generator: torch.Generator | None = None
        self._steps = 0

    @property
    def config(self) -> EditabilityLeverConfig:
        return self._config

    @property
    def steps_applied(self) -> int:
        """How many times a lever actually perturbed the model.

        A lever that is configured on but never fires is orphaned signal; this
        counter is what the run receipt reports so "held but never fired" is
        detectable rather than silent.
        """
        return self._steps

    def _rng(self, device: torch.device) -> torch.Generator:
        # Lazily created so an all-off config never constructs a generator and
        # therefore cannot perturb determinism in any way.
        if self._generator is None or self._generator.device != device:
            generator = torch.Generator(device=device)
            generator.manual_seed(self._config.seed)
            self._generator = generator
        return self._generator

    # -- F2 -----------------------------------------------------------------

    def _quantize(self, name: str, value: torch.Tensor) -> torch.Tensor:
        allocation = mixed_bit_allocation(
            [name],
            low_bits=self._config.weight_qat_low_bits,
            high_bits=self._config.weight_qat_high_bits,
        )
        return deployed_fake_quant(name, value, allocation[name])

    # -- F1 -----------------------------------------------------------------

    def _perturb(self, name: str, value: torch.Tensor) -> torch.Tensor:
        config = self._config
        sigma = config.weight_perturb_robustness
        if name in POSE_CRITICAL_TENSORS:
            sigma *= config.film_critical_multiplier
        if sigma <= 0.0 or value.ndim < 2:
            return value

        # Step size of this tensor's own deployed quantiser: perturbation is
        # measured in quantiser steps, so one sigma is directly comparable to the
        # coarsening an edit would apply.
        bits = self._config.weight_qat_high_bits
        limit = (1 << (bits - 1)) - 1
        embedding = name.endswith("embed.weight")
        reduce_dims = (
            tuple(range(value.ndim - 1)) if embedding else tuple(range(1, value.ndim))
        )
        step = (
            value.detach().abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-8)
            / limit
        )

        generator = self._rng(value.device)
        if config.weight_perturb_shape == "gaussian":
            noise = torch.randn(
                value.shape, generator=generator, device=value.device, dtype=value.dtype
            )
        else:
            noise = (
                torch.rand(
                    value.shape,
                    generator=generator,
                    device=value.device,
                    dtype=value.dtype,
                )
                - 0.5
            )
        # Straight-through: forward sees the perturbed weight, gradient reaches
        # the clean parameter.
        return value + (noise * step * sigma).detach()

    # -- F3 -----------------------------------------------------------------

    def _row_dropout(self, name: str, value: torch.Tensor) -> torch.Tensor:
        config = self._config
        if name not in FILM_ROW_FAMILY or value.ndim < 2:
            return value
        rows = int(value.shape[0])
        generator = self._rng(value.device)
        keep = (
            torch.rand(rows, generator=generator, device=value.device)
            >= config.film_row_dropout
        ).to(value.dtype)
        if config.film_row_dropout_protect_top > 0:
            protected = film_row_order(value)[: config.film_row_dropout_protect_top]
            if protected:
                keep[torch.tensor(protected, device=value.device)] = 1.0
        # Inverted dropout keeps the expectation unchanged, so F3 does not shift
        # the mean activation scale the rest of the network is trained against.
        scale = 1.0 / max(1.0 - config.film_row_dropout, 1e-6)
        mask = (keep * scale).reshape((rows,) + (1,) * (value.ndim - 1))
        return value * mask

    # -- application --------------------------------------------------------

    def transform(
        self, name: str, value: torch.Tensor, *, base_bits: int | None = None
    ) -> torch.Tensor:
        """Apply every active lever to one named tensor, in deployment order.

        ``base_bits`` is the QAT depth the caller would otherwise have applied
        uniformly (PR130's ``quantized_forward`` calls ``fake_quantize(v, bits)``
        on every parameter).  When it is supplied, quantization is applied to
        EVERY tensor unconditionally -- at the mp2 mixed map if F2 is on, at
        ``base_bits`` if it is off.

        That unconditional behaviour is load-bearing.  If quantization were
        applied only when F2 is on, enabling F1 or F3 alone would silently drop
        QAT from the run -- a catastrophic silent change that would look like a
        lever effect but would actually be the absence of quantization.
        """
        config = self._config
        result = value
        if config.f1_active:
            result = self._perturb(name, result)
        if base_bits is not None:
            bits = (
                self._mixed_bits(name)
                if config.f2_active
                else base_bits
            )
            result = deployed_fake_quant(name, result, bits)
        elif config.f2_active:
            result = self._quantize(name, result)
        if config.f3_active:
            result = self._row_dropout(name, result)
        return result

    def parameter_overrides(
        self, model: torch.nn.Module, *, base_bits: int
    ) -> dict[str, torch.Tensor]:
        """Build the ``torch.func.functional_call`` parameter dict.

        This is the PRIMARY integration point and it mirrors PR130's own
        ``quantized_forward``, which builds
        ``{name: fake_quantize(value, bits, ...)}`` and calls ``functional_call``.
        The only change is that the per-tensor bit depth comes from the lever
        config instead of one uniform ``bits``, and that F1/F3 compose in.

        Preferred over :meth:`applied` because it does not mutate the module at
        all -- so ``named_parameters()`` keeps its normal meaning throughout, and
        there is no restore path that could fail.

        With every lever off this returns exactly
        ``fake_quantize(value, base_bits, embedding)`` per tensor, which is
        bit-identical to the stock ``quantized_forward``.
        """
        return {
            name: self.transform(name, value, base_bits=base_bits)
            for name, value in model.named_parameters()
        }

    def _mixed_bits(self, name: str) -> int:
        return mixed_bit_allocation(
            [name],
            low_bits=self._config.weight_qat_low_bits,
            high_bits=self._config.weight_qat_high_bits,
        )[name]

    @contextmanager
    def applied(
        self, model: torch.nn.Module, *, base_bits: int | None = None
    ) -> Iterator[torch.nn.Module]:
        """Temporarily swap each parameter for its lever-transformed value.

        Implemented by replacing the ``nn.Parameter`` entries in each module's
        ``_parameters`` dict with plain tensors that are *functions of* the
        parameters, so autograd still reaches the originals.  The originals are
        restored unconditionally on exit, including on exception.

        Pass ``base_bits`` when substituting for PR130's ``render_quantized`` so
        the QAT depth is preserved even with every lever off (see
        :meth:`transform`).  With ``base_bits=None`` and no lever active this is
        a strict no-op: it does not touch ``_parameters``, does not read any
        tensor, and does not draw any RNG.
        """
        if not self._config.any_active and base_bits is None:
            yield model
            return

        originals: list[tuple[torch.nn.Module, str, torch.nn.Parameter]] = []
        try:
            for module_name, module in model.named_modules():
                for param_name, param in list(module._parameters.items()):
                    if param is None:
                        continue
                    qualified = f"{module_name}.{param_name}" if module_name else param_name
                    transformed = self.transform(qualified, param, base_bits=base_bits)
                    if transformed is param:
                        continue
                    originals.append((module, param_name, param))
                    del module._parameters[param_name]
                    setattr(module, param_name, transformed)
            if originals:
                self._steps += 1
            yield model
        finally:
            for module, param_name, param in originals:
                if hasattr(module, param_name):
                    delattr(module, param_name)
                module._parameters[param_name] = param

    # -- F4 -----------------------------------------------------------------

    def rank_penalty(self, model: torch.nn.Module) -> torch.Tensor:
        """Nuclear-norm surrogate on the named carrier tensors.

        Returns a zero scalar (detached, no graph) when F4 is off, so adding it
        to a loss is byte-identical to not adding it.
        """
        config = self._config
        if not config.f4_active:
            return torch.zeros((), dtype=torch.float32)

        selected = frozenset(config.carrier_tensors)
        found: list[torch.Tensor] = []
        state = dict(model.named_parameters())
        missing = sorted(selected - set(state))
        if missing:
            raise LeverError(
                f"carrier rank penalty names absent from the model: {missing}"
            )
        for name in sorted(selected):
            value = state[name]
            if value.ndim < 2:
                raise LeverError(
                    f"carrier rank penalty needs a matrix, {name} has ndim={value.ndim}"
                )
            matrix = value.reshape(int(value.shape[0]), -1)
            # Nuclear norm = sum of singular values = the convex envelope of rank
            # on the spectral-norm ball.  Normalised by the Frobenius norm so the
            # penalty measures spectral CONCENTRATION rather than raw magnitude
            # (otherwise F4 degenerates into weight decay).
            singular = torch.linalg.svdvals(matrix)
            frobenius = singular.square().sum().clamp_min(1e-12).sqrt()
            found.append(singular.sum() / frobenius)
        return config.carrier_rank_penalty * torch.stack(found).sum()


def _selftest() -> None:  # pragma: no cover - exercised by the test module
    config = EditabilityLeverConfig()
    if config.any_active:
        raise LeverError("default config must be inert")


if __name__ == "__main__":  # pragma: no cover
    _selftest()
    print("editability levers: default config is inert")
