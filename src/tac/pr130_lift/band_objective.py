#!/usr/bin/env python3
"""Debt-proportional 1-px label-band weighting for the PR130 semantic QAT loss.

WHY THIS EXISTS (the measured routing, not a preference)
--------------------------------------------------------
``ddm_rt1`` measured that **99.22% of the seg axis sits exactly ON the transmitted
label boundary** -- a curve one pixel wide that the decoder already owns, so it
costs **zero archive bytes** to address.  ``ddm_av3`` then measured, across the
four ``ddm_lr1`` arms (three decades of learning rate, 250x of weight
displacement), that the trainer's excursion obeys ``peak_flips ~ ||dw||^0.458``
with R^2 = 0.9969.  A smooth basin gives exponent ~2; exponent ~1/2 is the
signature of a piecewise-constant argmax field perturbed **diffusively** -- only
pixels already within delta of a decision boundary can flip.  The init sits on an
argmax **plateau**, and no learning rate fixes a plateau.

The binding constraint is therefore the objective's **direction**: the trainer
descends ``curriculum_loss``, a scalar ``.mean()`` over the whole field
(``lifted/semantic_renderer_oracle.py:181``), while the metric counts one-pixel
boundary crossings.  This module supplies the per-pixel reduction that lets the
gradient budget follow the **measured per-edge debt** on exactly that curve.

WHAT IT IS NOT
--------------
Not a port of the MLX witness ``_live_margin_weight``.  That consumer weights by
margin **magnitude**; this weights by **measured per-edge S debt on a
geometrically-defined band**.  Margin-magnitude weighting is a separate variable
and does not ride this term.

THE RULE, IN ONE LINE
---------------------
``weight is proportional to measured flip density`` -- uniformly, on-band and
off-band alike::

    W_e     = flips_e / band_px_e          # per unordered class pair
    W_off   = flips_off / px_off           # the SAME rule, off the band

Nothing here is picked.  ``flips_e`` and ``flips_off`` come from measured
confusion/ring receipts; ``band_px_e`` is the geometric incidence count.

SCALE NEUTRALITY IS STRUCTURAL, NOT A CONVENTION
------------------------------------------------
Two independent guarantees, deliberately redundant:

1. :func:`band_weight_field` returns ``(1 - alpha) * 1 + alpha * W / mean(W)``,
   whose mean is **exactly 1 for every alpha and every field** -- algebraically,
   not by a post-hoc renormalisation.
2. :func:`curriculum_loss_weighted` reduces with ``(w * l).sum() / w.sum()``, so
   **any** positive weight field reduces to a weighted mean.  A caller cannot
   rescale the loss -- and therefore cannot silently rescale the effective
   learning rate -- by supplying an unnormalised weight.

That matters because ``ddm_lr1`` just spent four arms measuring what learning
rate does here.  A term that moved the effective lr would confound exactly the
quantity that ladder pinned down.

THE POSITIVE CONTROL
--------------------
``curriculum_loss_weighted(..., weight=None)`` **delegates** to the lifted
``curriculum_loss``.  Equality is therefore structural (same call, same object),
not a numerical coincidence.  The reimplemented per-pixel path is separately
controlled against the oracle with a **uniform** weight field, and the residual
of that control is measured rather than assumed -- see
``tests/test_band_objective.py``.

AXIS
----
Every number this module derives is ``[macOS-CPU advisory]`` read-back of
retained payloads.  It is never a score.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.nn.functional as F

__all__ = [
    "BAND_WEIGHT_TABLE_PATH",
    "EDGE_PAIRS",
    "N_SEMANTIC_CLASSES",
    "BandObjectiveError",
    "BandWeightTable",
    "band_weight_field",
    "band_weight_stats",
    "curriculum_loss_weighted",
    "label_boundary",
    "load_band_weight_table",
    "pair_debt_field",
]

#: comma10k canonical order: 0 Road, 1 Lane, 2 Undrivable, 3 Movable, 4 MyCar.
#: NEVER re-derived by luma-sorting the palette -- that yields a different order
#: and has bitten this repo three times.  No class index is hardcoded in the
#: weighting itself: the table is keyed by the unordered pair present at each
#: band pixel, exactly as ``ddm_rc2`` section 2.1 requires.
N_SEMANTIC_CLASSES = 5

#: The 10 unordered class pairs, in a fixed canonical order.
EDGE_PAIRS: tuple[tuple[int, int], ...] = tuple(
    itertools.combinations(range(N_SEMANTIC_CLASSES), 2)
)

_LIFTED_DIR = Path(__file__).resolve().parent / "lifted"

#: The committed, machine-generated weight table.  It travels with the package so
#: a run needs no external volume mounted, and it carries the sha256 of every
#: payload it was derived from.
BAND_WEIGHT_TABLE_PATH = Path(__file__).resolve().parent / "band_weight_table_rt1_n600.json"

BAND_WEIGHT_TABLE_SCHEMA = "ddm_rg1b_band_weight_table.v1"


class BandObjectiveError(RuntimeError):
    """Fail-closed error for band-objective instrument or custody violations."""


def _pair_key(a: int, b: int) -> str:
    lo, hi = (a, b) if a <= b else (b, a)
    return f"{lo}-{hi}"


@lru_cache(maxsize=1)
def _oracle() -> ModuleType:
    """Load the lifted PR130 oracle WITHOUT mutating it.

    Mirrors ``train_semantic_quantized_resumable._load_lifted_qat``.  Only the
    two pure functions ``curriculum_loss`` and ``target_margin`` are used, so a
    second module instance in the process is harmless -- there is no module
    state to diverge.
    """

    path = _LIFTED_DIR / "semantic_renderer_oracle.py"
    spec = importlib.util.spec_from_file_location(
        "tac.pr130_lift.dynamic.semantic_renderer_oracle", path
    )
    if spec is None or spec.loader is None:  # pragma: no cover - packaging error
        raise BandObjectiveError(f"cannot load lifted semantic oracle at {path}")
    module = importlib.util.module_from_spec(spec)
    previous = list(sys.path)
    if str(_LIFTED_DIR) not in sys.path:
        sys.path.insert(0, str(_LIFTED_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous
    return module


# =============================================================================
# the table
# =============================================================================
@dataclass(frozen=True)
class BandWeightTable:
    """Measured per-edge debt density, plus the off-band density under the same rule.

    ``flips_by_pair`` and ``band_px_by_pair`` are keyed ``"lo-hi"`` over the
    unordered class pair.  ``basis`` records WHICH confusion the numerator came
    from; the trainer's in-loop metric is ``pred != target_tokens``, so
    ``pred_vs_label`` is the matching basis and ``pred_vs_gt`` is retained only
    as the cross-check.
    """

    basis: str
    flips_by_pair: Mapping[str, int]
    band_px_by_pair: Mapping[str, int]
    off_band_flips: int
    off_band_px: int
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.basis not in ("pred_vs_label", "pred_vs_gt"):
            raise BandObjectiveError(f"unknown confusion basis {self.basis!r}")
        expected = {_pair_key(a, b) for a, b in EDGE_PAIRS}
        for name, mapping in (
            ("flips_by_pair", self.flips_by_pair),
            ("band_px_by_pair", self.band_px_by_pair),
        ):
            if set(mapping) != expected:
                missing = sorted(expected - set(mapping))
                extra = sorted(set(mapping) - expected)
                raise BandObjectiveError(
                    f"{name} must cover exactly the 10 unordered pairs "
                    f"(missing={missing}, extra={extra})"
                )
            if any(int(v) < 0 for v in mapping.values()):
                raise BandObjectiveError(f"{name} has a negative count")
        if self.off_band_px <= 0:
            raise BandObjectiveError("off_band_px must be positive")
        if self.off_band_flips < 0:
            raise BandObjectiveError("off_band_flips must be non-negative")

    # -- derived quantities -------------------------------------------------
    @property
    def weights_by_pair(self) -> dict[str, float]:
        """``W_e = flips_e / band_px_e``.

        A pair with **no band pixels** has no defined density; it is reported as
        ``0.0`` and, having no pixels, can never be selected at runtime.  A pair
        with band pixels but **zero measured flips** genuinely has zero measured
        debt, and gets ``0.0`` -- the honest reading of the receipt rather than a
        smoothing prior invented here.
        """

        out: dict[str, float] = {}
        for key, band_px in self.band_px_by_pair.items():
            flips = int(self.flips_by_pair[key])
            out[key] = (flips / band_px) if band_px > 0 else 0.0
        return out

    @property
    def off_band_weight(self) -> float:
        """``W_off = flips_off / px_off`` -- the SAME density rule, off the band."""

        return self.off_band_flips / self.off_band_px

    @property
    def band_px_total_incident(self) -> int:
        return sum(int(v) for v in self.band_px_by_pair.values())

    def concentration(self) -> dict[str, float]:
        """Ratio of each band weight to the off-band weight (observability)."""

        off = self.off_band_weight
        weights = self.weights_by_pair
        if off <= 0.0:
            return {key: float("inf") for key in weights}
        return {key: value / off for key, value in weights.items()}

    def lookup_tensor(
        self, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Symmetric ``(5, 5)`` matrix of ``W_e``; the diagonal is zero."""

        lut = torch.zeros(
            (N_SEMANTIC_CLASSES, N_SEMANTIC_CLASSES), device=device, dtype=dtype
        )
        weights = self.weights_by_pair
        for class_a, class_b in EDGE_PAIRS:
            value = weights[_pair_key(class_a, class_b)]
            lut[class_a, class_b] = value
            lut[class_b, class_a] = value
        return lut

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BAND_WEIGHT_TABLE_SCHEMA,
            "score_claim": False,
            "axis": "[macOS-CPU advisory] derived from retained payloads -- never a score",
            "basis": self.basis,
            "flips_by_pair": {k: int(v) for k, v in sorted(self.flips_by_pair.items())},
            "band_px_by_pair": {
                k: int(v) for k, v in sorted(self.band_px_by_pair.items())
            },
            "off_band_flips": int(self.off_band_flips),
            "off_band_px": int(self.off_band_px),
            "derived_weights_by_pair": dict(sorted(self.weights_by_pair.items())),
            "derived_off_band_weight": self.off_band_weight,
            "derived_concentration_vs_off_band": dict(
                sorted(self.concentration().items())
            ),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BandWeightTable:
        schema = payload.get("schema")
        if schema != BAND_WEIGHT_TABLE_SCHEMA:
            raise BandObjectiveError(
                f"band weight table schema {schema!r} != {BAND_WEIGHT_TABLE_SCHEMA!r}"
            )
        return cls(
            basis=str(payload["basis"]),
            flips_by_pair={k: int(v) for k, v in payload["flips_by_pair"].items()},
            band_px_by_pair={k: int(v) for k, v in payload["band_px_by_pair"].items()},
            off_band_flips=int(payload["off_band_flips"]),
            off_band_px=int(payload["off_band_px"]),
            provenance=dict(payload.get("provenance", {})),
        )


@lru_cache(maxsize=4)
def _load_table_cached(path: str) -> BandWeightTable:
    resolved = Path(path)
    if not resolved.exists():
        raise BandObjectiveError(f"band weight table not found: {resolved}")
    return BandWeightTable.from_dict(json.loads(resolved.read_text()))


def load_band_weight_table(path: Path | str | None = None) -> BandWeightTable:
    """Load the committed table (or an explicit override)."""

    return _load_table_cached(str(Path(path or BAND_WEIGHT_TABLE_PATH).resolve()))


def band_weight_table_sha256(path: Path | str | None = None) -> str:
    """sha256 of the table FILE -- the causal identity recorded in run config."""

    resolved = Path(path or BAND_WEIGHT_TABLE_PATH)
    if not resolved.exists():
        raise BandObjectiveError(f"band weight table not found: {resolved}")
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


# =============================================================================
# the band geometry
# =============================================================================
def label_boundary(target: torch.Tensor) -> torch.Tensor:
    """Ring-0 of the 4-neighbour label boundary; ``True`` = ON the boundary.

    VERBATIM the ``ddm_rt1`` / sq1 / gp1 ``boundary()`` convention, lifted to a
    batched torch tensor.  Verified bit-exact against the retained
    ``free_band_mask.npy`` over all 600 frames (2,551,464 band px, matching
    ``RT1_GEOMETRY.json::ring_population[0]`` exactly).

    ``target`` is ``(B, H, W)`` integer labels; the result is ``(B, H, W)`` bool.
    """

    if target.ndim != 3:
        raise BandObjectiveError(
            f"target must be (B, H, W) integer labels, got shape {tuple(target.shape)}"
        )
    band = torch.zeros_like(target, dtype=torch.bool)
    vertical = target[:, :-1, :] != target[:, 1:, :]
    band[:, :-1, :] |= vertical
    band[:, 1:, :] |= vertical
    horizontal = target[:, :, :-1] != target[:, :, 1:]
    band[:, :, :-1] |= horizontal
    band[:, :, 1:] |= horizontal
    return band


def pair_debt_field(
    target: torch.Tensor,
    table: BandWeightTable,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pixel measured debt density ``W(p)``, and the band mask.

    A pixel incident to more than one class pair (a junction) takes the **max**
    of its incident densities.  Declared DOF, with its blast radius MEASURED
    rather than assumed: junctions are **17,990 of 2,551,464 band px = 0.705%**
    on n600, so ``max`` vs ``sum`` vs ``mean`` moves under 1% of the band.  Max
    is chosen because a pixel can only flip once, so its risk is governed by its
    riskiest incident edge, and because it is bounded and tie-break free.
    """

    band = label_boundary(target)
    lut = table.lookup_tensor(device=target.device, dtype=dtype)
    debt = torch.zeros(target.shape, device=target.device, dtype=dtype)

    def accumulate(
        left: torch.Tensor, right: torch.Tensor, slices: Sequence[Any]
    ) -> None:
        differs = left != right
        if not bool(differs.any()):
            return
        value = torch.where(
            differs, lut[left.long(), right.long()], torch.zeros((), dtype=dtype, device=target.device)
        )
        for index in slices:
            debt[index] = torch.maximum(debt[index], value)

    accumulate(
        target[:, :-1, :],
        target[:, 1:, :],
        ((slice(None), slice(None, -1), slice(None)), (slice(None), slice(1, None), slice(None))),
    )
    accumulate(
        target[:, :, :-1],
        target[:, :, 1:],
        ((slice(None), slice(None), slice(None, -1)), (slice(None), slice(None), slice(1, None))),
    )
    off = torch.full((), table.off_band_weight, device=target.device, dtype=dtype)
    return torch.where(band, debt, off), band


def band_weight_field(
    target: torch.Tensor,
    table: BandWeightTable,
    alpha: float,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """The per-pixel loss weight: ``(1 - alpha) + alpha * W / mean(W)``.

    ``alpha`` is the mixing fraction between the stock uniform reduction
    (``alpha = 0``, an exact no-op) and pure measured debt density
    (``alpha = 1``).  ``mean(weight) == 1`` holds **algebraically** for every
    ``alpha`` and every field, because both mixed terms already have mean 1 --
    so this term cannot rescale the loss, and therefore cannot perturb the
    effective learning rate that ``ddm_lr1`` just measured.
    """

    if not 0.0 <= alpha <= 1.0:
        raise BandObjectiveError(f"--band-objective-weight must be in [0, 1], got {alpha}")
    debt, _ = pair_debt_field(target, table, dtype=dtype)
    mean_debt = debt.mean()
    if not bool(torch.isfinite(mean_debt)) or float(mean_debt) <= 0.0:
        raise BandObjectiveError(
            "measured debt field has non-positive mean; the table or the target "
            "field is not the one rt1 measured"
        )
    return (1.0 - alpha) + alpha * (debt / mean_debt)


def band_weight_stats(weight: torch.Tensor, band: torch.Tensor | None = None) -> dict[str, Any]:
    """Score-neutral telemetry.  Read-only, so it defaults ON at every call site.

    "Off is a tracked state, never a forgotten default": a term that is held but
    never fires must be detectable, and a term that fires must show HOW.
    """

    flat = weight.detach().float()
    stats: dict[str, Any] = {
        "mean": float(flat.mean()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "numel": int(flat.numel()),
    }
    if band is not None:
        on = band.reshape(-1)
        values = flat.reshape(-1)
        band_px = int(on.sum())
        stats.update(
            {
                "band_px": band_px,
                "band_fraction": band_px / max(int(on.numel()), 1),
                "band_mean": float(values[on].mean()) if band_px else 0.0,
                "off_band_mean": float(values[~on].mean()) if band_px < on.numel() else 0.0,
                "band_weight_mass_fraction": (
                    float(values[on].sum() / values.sum()) if float(values.sum()) else 0.0
                ),
            }
        )
    return stats


# =============================================================================
# the loss
# =============================================================================
def curriculum_loss_weighted(
    logits: torch.Tensor,
    target: torch.Tensor,
    step: int,
    total_steps: int,
    ce_fraction: float,
    softplus_fraction: float,
    weight: torch.Tensor | None = None,
    *,
    oracle: ModuleType | None = None,
) -> tuple[torch.Tensor, str]:
    """Per-pixel reproduction of the lifted ``curriculum_loss``, reduced by ``weight``.

    ``weight=None`` **delegates** to the lifted function, so the no-op case is
    bit-identical by construction rather than by numerical luck.  This is the
    control that makes the term admissible.

    ``weight`` is ``(B, H, W)`` (or broadcastable to it) and strictly positive.
    Reduction is ``(w * l).sum() / w.sum()``, i.e. a weighted mean -- so no
    weight field, normalised or not, can change the loss SCALE.
    """

    module = oracle if oracle is not None else _oracle()
    if weight is None:
        return module.curriculum_loss(
            logits, target, step, total_steps, ce_fraction, softplus_fraction
        )

    progress = step / max(total_steps - 1, 1)
    if progress < ce_fraction:
        temp = 1.0 * (0.08 ** (progress / ce_fraction))
        per_pixel = F.cross_entropy(logits / temp, target, reduction="none")
        return _weighted_mean(per_pixel, weight), "ce"

    margin = module.target_margin(logits, target)
    if progress < softplus_fraction:
        tau = 0.20
        per_pixel = F.softplus(-margin / tau) * tau
    else:
        tail = (progress - softplus_fraction) / max(1.0 - softplus_fraction, 1e-6)
        tau = 0.15 - 0.10 * tail
        per_pixel = torch.sigmoid(-margin / tau)
    phase = "softplus_margin" if progress < softplus_fraction else "expected_flip"
    return _weighted_mean(per_pixel.squeeze(1), weight), phase


def _weighted_mean(per_pixel: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if weight.shape != per_pixel.shape:
        weight = weight.expand_as(per_pixel)
    if bool((weight < 0).any()):
        raise BandObjectiveError("loss weight field must be non-negative")
    total = weight.sum()
    if float(total) <= 0.0:
        raise BandObjectiveError("loss weight field sums to zero")
    return (per_pixel * weight).sum() / total


# =============================================================================
# the derivation (step 0 of the rg1 charter, run offline against retained payloads)
# =============================================================================
_DEFAULT_TOKENS = Path(
    "/Volumes/APDataStore/pact/ddm_hv1_harvest_compose/ep0634/retained/coders/"
    "s1p25_c1p0/decoded_spatial_tokens.rc64.bin"
)
_DEFAULT_PRED = Path(
    "/Volumes/APDataStore/pact/ddm_rt1_seg_roundtrip_20260816/argmax_base.npy"
)
_DEFAULT_BAND = Path(
    "/Volumes/APDataStore/pact/ddm_rt1_seg_roundtrip_20260816/free_band_mask.npy"
)
_FRAMES, _H, _W = 600, 384, 512


def derive_table_from_payloads(
    *,
    tokens: Path = _DEFAULT_TOKENS,
    pred: Path = _DEFAULT_PRED,
    band_mask: Path | None = _DEFAULT_BAND,
    basis: str = "pred_vs_label",
    frames: int = _FRAMES,
) -> BandWeightTable:
    """Measure ``flips_e``, ``band_px_e`` and the off-band pair from the payloads.

    ``band_mask`` is the pre-registered INSTRUMENT CONTROL: when supplied, the
    boundary recomputed here must equal the retained ``free_band_mask.npy``
    frame-for-frame, or this refuses.  A mismatch means the band operator or the
    label field is not the one ``ddm_rt1`` measured, and no table is emitted.

    NUMERATOR AND DENOMINATOR SHARE ONE FOOTING.  ``flips_e`` counts band pixels
    **incident to edge e that flip**; ``band_px_e`` counts band pixels incident
    to edge e.  Both use the same incidence mask, so ``W_e`` is literally the
    flip RATE among the pixels of that edge.  The first draft of this function
    keyed the numerator on the CONFUSION pair ``(pred, label)`` while the
    denominator keyed on the GEOMETRIC pair -- a units mismatch that also
    double-counted the 264 off-band flips into the on-band numerators.  The
    confusion table is still measured, but only as ``provenance`` (it reproduces
    ``RT1_EDGESHAPE.json::confusion_pred_to_gt`` exactly, which is what validates
    this instrument against rt1's receipt).
    """

    import numpy as np  # local: keep the training import surface small

    if basis not in ("pred_vs_label", "pred_vs_gt"):
        raise BandObjectiveError(f"unknown basis {basis!r}")

    labels = np.memmap(tokens, dtype=np.uint8, mode="r", shape=(frames, _H, _W))
    predictions = np.load(pred, mmap_mode="r")
    retained = np.load(band_mask, mmap_mode="r") if band_mask is not None else None

    band_px = {_pair_key(a, b): 0 for a, b in EDGE_PAIRS}
    flips = {_pair_key(a, b): 0 for a, b in EDGE_PAIRS}
    confusion = {_pair_key(a, b): 0 for a, b in EDGE_PAIRS}
    band_px_ring0 = 0
    junction_px = 0
    on_band_flips = 0
    off_band_flips = 0
    total_flips = 0
    total_px = 0

    for index in range(frames):
        label = np.asarray(labels[index])
        boundary = _numpy_boundary(label)
        if retained is not None and not np.array_equal(
            boundary, np.asarray(retained[index]).astype(bool)
        ):
            raise BandObjectiveError(
                f"INSTRUMENT CONTROL FAILED: recomputed boundary != retained "
                f"free_band_mask at frame {index}"
            )
        band_px_ring0 += int(boundary.sum())
        total_px += label.size

        incidence = np.zeros(label.shape, dtype=np.uint16)
        for axis_a, axis_b in (
            ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
            ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ):
            left, right = label[axis_a], label[axis_b]
            differs = left != right
            if not differs.any():
                continue
            low = np.minimum(left, right)
            high = np.maximum(left, right)
            for bit, (class_a, class_b) in enumerate(EDGE_PAIRS):
                selected = differs & (low == class_a) & (high == class_b)
                if selected.any():
                    value = np.uint16(1 << bit)
                    incidence[axis_a][selected] |= value
                    incidence[axis_b][selected] |= value
        prediction = np.asarray(predictions[index])
        wrong = prediction != label
        total_flips += int(wrong.sum())
        on_band_flips += int(np.count_nonzero(wrong & boundary))
        off_band_flips += int(np.count_nonzero(wrong & ~boundary))

        popcount = np.zeros(label.shape, dtype=np.uint8)
        for bit, (class_a, class_b) in enumerate(EDGE_PAIRS):
            selected = (incidence & np.uint16(1 << bit)) != 0
            key = _pair_key(class_a, class_b)
            band_px[key] += int(selected.sum())
            # SAME incidence mask as the denominator: the flip RATE on edge e.
            flips[key] += int(np.count_nonzero(selected & wrong))
            popcount += selected.astype(np.uint8)
        junction_px += int((popcount > 1).sum())

        # Cross-check only: the confusion pair (pred, label), which is what
        # RT1_EDGESHAPE.json records.  Never used as a weight numerator.
        if wrong.any():
            for class_a, class_b in EDGE_PAIRS:
                pair = ((prediction == class_a) & (label == class_b)) | (
                    (prediction == class_b) & (label == class_a)
                )
                confusion[_pair_key(class_a, class_b)] += int(pair.sum())

    if on_band_flips + off_band_flips != total_flips:  # pragma: no cover - arithmetic
        raise BandObjectiveError(
            "on/off band flips do not partition the total: "
            f"{on_band_flips} + {off_band_flips} != {total_flips}"
        )

    provenance = {
        "frames": frames,
        "total_px": total_px,
        "band_px_ring0": band_px_ring0,
        "band_px_total_incident": sum(band_px.values()),
        "junction_px": junction_px,
        "junction_fraction_of_band": junction_px / max(band_px_ring0, 1),
        "total_flips": total_flips,
        "on_band_flips": on_band_flips,
        "on_band_flip_share": on_band_flips / max(total_flips, 1),
        "flips_by_pair_incident_sum": sum(flips.values()),
        "confusion_by_pair_cross_check": {
            k: int(v) for k, v in sorted(confusion.items())
        },
        "instrument_control_free_band_mask_parity": retained is not None,
        "sources": {
            "transmitted_labels": {
                "path": str(tokens),
                "sha256": _sha256(tokens),
            },
            "render_argmax": {"path": str(pred), "sha256": _sha256(pred)},
            **(
                {"free_band_mask": {"path": str(band_mask), "sha256": _sha256(band_mask)}}
                if band_mask is not None
                else {}
            ),
        },
        "note": (
            "flips_by_pair and band_px_by_pair are BOTH incidence counts over the "
            "same mask (a junction pixel counts toward each incident pair), so W_e "
            "is the flip rate on edge e; confusion_by_pair_cross_check is the "
            "(pred, label) pair table rt1 recorded and is NOT a weight numerator"
        ),
    }
    return BandWeightTable(
        basis=basis,
        flips_by_pair=flips,
        band_px_by_pair=band_px,
        off_band_flips=off_band_flips,
        off_band_px=total_px - band_px_ring0,
        provenance=provenance,
    )


def _numpy_boundary(label: Any) -> Any:
    import numpy as np

    band = np.zeros(label.shape, dtype=bool)
    differs = label[:-1, :] != label[1:, :]
    band[:-1, :] |= differs
    band[1:, :] |= differs
    differs = label[:, :-1] != label[:, 1:]
    band[:, :-1] |= differs
    band[:, 1:] |= differs
    return band


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_report(table: BandWeightTable) -> str:
    weights = table.weights_by_pair
    concentration = table.concentration()
    names = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}
    total_flips = int(table.provenance.get("total_flips") or 0) or (
        sum(table.flips_by_pair.values()) + table.off_band_flips
    )
    incident_sum = sum(table.flips_by_pair.values())
    lines = [
        f"basis={table.basis}  band_px={table.provenance.get('band_px_ring0')}  "
        f"off_band_px={table.off_band_px}",
        f"{'pair':<24}{'flips':>8}{'of inc':>8}{'band_px':>11}"
        f"{'W_e':>13}{'vs off':>10}",
    ]
    order = sorted(EDGE_PAIRS, key=lambda p: -weights[_pair_key(*p)])
    for class_a, class_b in order:
        key = _pair_key(class_a, class_b)
        flips = table.flips_by_pair[key]
        lines.append(
            f"{names[class_a] + '/' + names[class_b]:<24}{flips:>8}"
            f"{flips / max(incident_sum, 1):>8.3%}{table.band_px_by_pair[key]:>11}"
            f"{weights[key]:>13.7f}{concentration[key]:>10.0f}"
        )
    lines.append(
        f"{'OFF-BAND':<24}{table.off_band_flips:>8}{'':>8}"
        f"{table.off_band_px:>11}{table.off_band_weight:>13.7f}{1:>10.0f}"
    )
    on_band = int(table.provenance.get("on_band_flips") or 0)
    lines.append(
        f"partition: on_band {on_band} + off_band {table.off_band_flips} "
        f"= {total_flips} total  (on-band share "
        f"{on_band / max(total_flips, 1):.4%}; rt1 measured 99.22%)"
    )
    lines.append(
        f"incidence: pair flips sum to {incident_sum} "
        f"({incident_sum / max(on_band, 1):.4f}x on_band -- junction pixels count "
        f"toward each incident edge, as band_px does)"
    )
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Derive the debt-proportional band weight table from retained rt1 "
            "payloads (rc2 section 2.1 step 0).  Read-only on every input."
        )
    )
    parser.add_argument("--tokens", type=Path, default=_DEFAULT_TOKENS)
    parser.add_argument("--pred", type=Path, default=_DEFAULT_PRED)
    parser.add_argument("--band-mask", type=Path, default=_DEFAULT_BAND)
    parser.add_argument("--basis", choices=("pred_vs_label", "pred_vs_gt"), default="pred_vs_label")
    parser.add_argument("--frames", type=int, default=_FRAMES)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"write the table JSON here (default: print only; canonical is {BAND_WEIGHT_TABLE_PATH})",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    table = derive_table_from_payloads(
        tokens=args.tokens,
        pred=args.pred,
        band_mask=args.band_mask,
        basis=args.basis,
        frames=args.frames,
    )
    print(_format_report(table), file=sys.stderr)
    payload = json.dumps(table.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.write_text(payload)
        print(f"wrote {args.out} sha256={hashlib.sha256(payload.encode()).hexdigest()}", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
