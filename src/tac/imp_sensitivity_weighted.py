"""Lane 17 IMP β-variant — sensitivity-weighted iterative magnitude pruning.

Paradigm β extension of :mod:`tac.iterative_magnitude_pruning`. Standard IMP
prunes the lowest-magnitude weights globally with a flat sparsity target.
Sensitivity-weighted IMP additionally consults a per-Conv2d-layer score-
sensitivity vector (produced offline by
``experiments/profile_hessian_per_weight.py`` and converted via
``experiments/convert_fisher_to_owv3_sensitivity_map.py``) and:

* Layers with sensitivity > ``protect_threshold`` (default 1e-3) are
  fully PROTECTED — their weights are never pruned, regardless of magnitude.
  These are the channels that the Ω-W-V2 1.07 regression revealed: PoseNet-
  sensitive Conv2d outputs whose perturbation translates 1:1 to PoseNet
  distortion.
* Layers with sensitivity < ``aggressive_threshold`` (default 1e-5) are
  pruned AGGRESSIVELY — their per-cycle sparsity increment is multiplied
  by ``aggressive_multiplier`` (default 1.5). The standard 0.20 increment
  reaches 89.3% over 10 cycles; under aggressive scheduling these layers
  reach >95% sparsity over the same 10 cycles.
* Intermediate layers follow the vanilla IMP increment.

The sensitivity vector is per-output-channel (matching the OWV3 contract
in :mod:`tac.sensitivity_map`). Per-LAYER aggregation is the maximum
per-channel sensitivity — if ANY output channel of a layer is
score-sensitive, the whole layer is protected. This avoids the failure
mode of pruning sparse high-sensitivity channels into dense layers.

Wire format
-----------
None — IMP is a training-time pruning policy; it produces a Conv2d weight
mask consumed by :mod:`tac.imps_renderer_archive` for archive emission.
This β-variant produces the same mask schema; the IMPS magic-byte writer
is unchanged.

CLAUDE.md compliance
--------------------
* Compress-time only; no scorer load at decode time.
* Pure-math operation on (model, mask, sensitivities) → new mask.
* Sensitivity input is the canonical OWV3 contract — same artifact as
  Ω-W-V3 consumes. No duplicated sensitivity-computation surface.
* No silent defaults; thresholds + multiplier are explicit kwargs.

[prediction] Lane 17 β-variant predicted band [0.92, 1.05] [contest-CUDA]
on Lane G v3 anchor — same ballpark as vanilla Lane 17 (0.95-1.10) but
with PoseNet distortion held within +5% vs the +63.4% the unweighted
Ω-W-V2 stack produced.

References
----------
* :mod:`tac.iterative_magnitude_pruning` — vanilla IMP primitives that
  this module wraps.
* :mod:`tac.sensitivity_map` — canonical sensitivity-vector contract.
* :mod:`tac.owv3_sensitivity_weighted` — sister codec applying the same
  sensitivity-vector idea to Ω-W-V2 water-fill quantization.
* ``.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430.md``
  §"Paradigm Shift β" — math foundation.
"""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from tac.iterative_magnitude_pruning import (
    iter_prunable_parameters,
    prune_lowest_magnitude,
)
from tac.sensitivity_map import (
    SensitivityMapError,
    resolve_layer_sensitivity,
)


class SensitivityWeightedIMPError(ValueError):
    """Raised when sensitivity-weighted IMP inputs are malformed."""


def _validate_thresholds(
    protect_threshold: float,
    aggressive_threshold: float,
    aggressive_multiplier: float,
) -> None:
    if not isinstance(protect_threshold, (int, float)):
        raise SensitivityWeightedIMPError(
            "protect_threshold must be a finite scalar"
        )
    if not isinstance(aggressive_threshold, (int, float)):
        raise SensitivityWeightedIMPError(
            "aggressive_threshold must be a finite scalar"
        )
    if not isinstance(aggressive_multiplier, (int, float)):
        raise SensitivityWeightedIMPError(
            "aggressive_multiplier must be a finite scalar"
        )
    if protect_threshold <= 0.0:
        raise SensitivityWeightedIMPError("protect_threshold must be > 0")
    if aggressive_threshold < 0.0:
        raise SensitivityWeightedIMPError("aggressive_threshold must be >= 0")
    if aggressive_threshold >= protect_threshold:
        raise SensitivityWeightedIMPError(
            "aggressive_threshold must be < protect_threshold"
        )
    if aggressive_multiplier <= 0.0:
        raise SensitivityWeightedIMPError("aggressive_multiplier must be > 0")


def _module_lookup(model: nn.Module) -> dict[str, nn.Module]:
    return dict(model.named_modules())


def classify_layers_by_sensitivity(
    *,
    model: nn.Module,
    sensitivities: Mapping[str, torch.Tensor],
    protect_threshold: float = 1e-3,
    aggressive_threshold: float = 1e-5,
) -> dict[str, str]:
    """Classify each prunable layer as ``protect`` / ``aggressive`` / ``standard``.

    Per-layer sensitivity is the MAX over per-output-channel sensitivities.
    A layer with any high-sensitivity channel is protected entirely.

    Args:
        model: target model. Required.
        sensitivities: OWV3 sensitivity-map contract — keys are
            ``"<module>.weight"``, values are 1-D tensors with one
            non-negative scalar per output channel. Required.
        protect_threshold: layers with max-channel-sensitivity above this
            are protected (never pruned).
        aggressive_threshold: layers with max-channel-sensitivity below
            this are pruned with the aggressive multiplier.

    Returns:
        ``{ "<module>.weight" -> "protect"|"aggressive"|"standard" }``.

    Raises:
        SensitivityWeightedIMPError: bad inputs / missing sensitivity tensors.
    """
    _validate_thresholds(
        protect_threshold,
        aggressive_threshold,
        aggressive_multiplier=1.0,  # not used in classify path
    )

    out: dict[str, str] = {}
    name_to_module = _module_lookup(model)

    for qname, _param in iter_prunable_parameters(model):
        # Strip trailing ".weight" if present
        module_name = qname[:-7] if qname.endswith(".weight") else qname
        module = name_to_module.get(module_name)
        if module is None:
            raise SensitivityWeightedIMPError(
                f"classify_layers_by_sensitivity: cannot resolve module for {qname!r}"
            )
        try:
            sens = resolve_layer_sensitivity(
                sensitivities,
                module_name=module_name,
                weight=module.weight,
                required=True,
            )
        except SensitivityMapError as exc:
            raise SensitivityWeightedIMPError(str(exc)) from exc
        assert sens is not None
        max_sens = float(sens.max().item())
        if max_sens > protect_threshold:
            out[qname] = "protect"
        elif max_sens < aggressive_threshold:
            out[qname] = "aggressive"
        else:
            out[qname] = "standard"
    return out


def prune_with_sensitivity_weighting(
    *,
    model: nn.Module,
    sensitivities: Mapping[str, torch.Tensor],
    sparsity_increment: float = 0.20,
    aggressive_multiplier: float = 1.5,
    protect_threshold: float = 1e-3,
    aggressive_threshold: float = 1e-5,
    current_mask: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Run a sensitivity-weighted IMP cycle.

    Strategy:

    1. Classify every prunable layer via :func:`classify_layers_by_sensitivity`.
    2. Build a temporary nn.Module containing only ``standard`` layers and
       prune them with ``sparsity_increment``.
    3. Build a second temporary containing only ``aggressive`` layers and
       prune them with ``sparsity_increment * aggressive_multiplier``.
       The multiplier is capped at ``0.95`` (we never prune 100% of a
       layer because that destroys the layer entirely).
    4. ``protect`` layers retain their existing mask (or full-True if no
       prior mask).

    The thresholding step is the load-bearing one: Ω-W-V2 1.07 regression
    revealed that uniform allocation pays the full PoseNet sensitivity
    penalty. By protecting high-sensitivity layers, this β-variant trades
    a small amount of byte-savings for a large amount of distortion
    recovery.

    Args:
        model: target model. Required.
        sensitivities: OWV3 sensitivity-map contract. Required.
        sparsity_increment: vanilla per-cycle sparsity fraction. Default 0.20.
        aggressive_multiplier: multiplier on ``sparsity_increment`` for
            low-sensitivity layers. Default 1.5 → these layers prune at 0.30.
        protect_threshold: see :func:`classify_layers_by_sensitivity`.
        aggressive_threshold: see :func:`classify_layers_by_sensitivity`.
        current_mask: optional existing IMP mask.

    Returns:
        New mask dict with sensitivity-weighted pruning applied.

    Raises:
        SensitivityWeightedIMPError: bad inputs.
        ValueError: from the underlying ``prune_lowest_magnitude`` if
            sparsity_increment is out of range.
    """
    _validate_thresholds(
        protect_threshold,
        aggressive_threshold,
        aggressive_multiplier,
    )
    if not (0.0 < sparsity_increment < 1.0):
        raise SensitivityWeightedIMPError(
            f"sparsity_increment must be in (0, 1); got {sparsity_increment!r}"
        )
    aggressive_inc = min(0.95, sparsity_increment * aggressive_multiplier)
    if aggressive_inc <= 0.0 or aggressive_inc >= 1.0:
        raise SensitivityWeightedIMPError(
            f"effective aggressive sparsity_increment {aggressive_inc} "
            "must be in (0, 1) — adjust aggressive_multiplier"
        )

    classification = classify_layers_by_sensitivity(
        model=model,
        sensitivities=sensitivities,
        protect_threshold=protect_threshold,
        aggressive_threshold=aggressive_threshold,
    )

    new_mask: dict[str, torch.Tensor] = {}
    name_to_module = _module_lookup(model)

    if current_mask is None:
        current_mask = {}
        for qname, param in iter_prunable_parameters(model):
            current_mask[qname] = torch.ones_like(
                param, dtype=torch.bool, device="cpu"
            )

    # 1. Protected layers: identity copy of current mask.
    for qname, kind in classification.items():
        if kind == "protect":
            existing = current_mask.get(qname)
            if existing is None:
                module_name = qname[:-7] if qname.endswith(".weight") else qname
                module = name_to_module[module_name]
                existing = torch.ones_like(
                    module.weight, dtype=torch.bool, device="cpu"
                )
            new_mask[qname] = existing.clone()

    # 2. Standard layers: vanilla prune at sparsity_increment.
    standard_qnames = [n for n, k in classification.items() if k == "standard"]
    if standard_qnames:
        std_model = _build_subset_module(model, standard_qnames)
        std_current_mask = {
            qname: current_mask.get(qname).clone()
            if current_mask.get(qname) is not None
            else torch.ones_like(
                getattr(std_model, _qname_to_subset_attr(qname)).weight,
                dtype=torch.bool,
                device="cpu",
            )
            for qname in standard_qnames
        }
        # Map qnames into the std_model's namespace (subset attribute name)
        std_input_mask = {
            _qname_to_subset_attr(qname) + ".weight": std_current_mask[qname]
            for qname in standard_qnames
        }
        std_out = prune_lowest_magnitude(
            std_model,
            sparsity_increment=sparsity_increment,
            current_mask=std_input_mask,
        )
        for qname in standard_qnames:
            new_mask[qname] = std_out[
                _qname_to_subset_attr(qname) + ".weight"
            ]

    # 3. Aggressive layers: prune at sparsity_increment * aggressive_multiplier.
    aggressive_qnames = [
        n for n, k in classification.items() if k == "aggressive"
    ]
    if aggressive_qnames:
        agg_model = _build_subset_module(model, aggressive_qnames)
        agg_current_mask = {
            qname: current_mask.get(qname).clone()
            if current_mask.get(qname) is not None
            else torch.ones_like(
                getattr(agg_model, _qname_to_subset_attr(qname)).weight,
                dtype=torch.bool,
                device="cpu",
            )
            for qname in aggressive_qnames
        }
        agg_input_mask = {
            _qname_to_subset_attr(qname) + ".weight": agg_current_mask[qname]
            for qname in aggressive_qnames
        }
        agg_out = prune_lowest_magnitude(
            agg_model,
            sparsity_increment=aggressive_inc,
            current_mask=agg_input_mask,
        )
        for qname in aggressive_qnames:
            new_mask[qname] = agg_out[
                _qname_to_subset_attr(qname) + ".weight"
            ]

    return new_mask


def _qname_to_subset_attr(qname: str) -> str:
    """Map a dotted qname to a flat subset-module attribute name."""
    base = qname[:-7] if qname.endswith(".weight") else qname
    return base.replace(".", "__") or "root"


def _build_subset_module(
    model: nn.Module, qnames: list[str]
) -> nn.Module:
    """Build a flat ``nn.Module`` containing only the named conv layers.

    Used so :func:`prune_lowest_magnitude` (which globally pools magnitudes
    across all prunable parameters of its input model) sees only the
    subset we want to prune. The flat-module attribute name encodes the
    original qname so we can map back.
    """
    name_to_module = dict(model.named_modules())
    container = nn.Module()
    for qname in qnames:
        module_name = qname[:-7] if qname.endswith(".weight") else qname
        original = name_to_module[module_name]
        if not isinstance(original, (nn.Conv2d, nn.ConvTranspose2d)):
            raise SensitivityWeightedIMPError(
                f"_build_subset_module: {qname!r} is not Conv2d/ConvTranspose2d"
            )
        attr = _qname_to_subset_attr(qname)
        container.add_module(attr, original)
    return container


__all__ = [
    "SensitivityWeightedIMPError",
    "classify_layers_by_sensitivity",
    "prune_with_sensitivity_weighting",
]
