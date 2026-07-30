# SPDX-License-Identifier: MIT
"""QA80 margin-budget producer — per-pixel flip-distance field ``d = |m| / ||Δw||``.

Scorer-free. Consumes CACHED SegNet per-pixel margin + argmax fields (no fresh
scorer pass) and the MEASURED rank-4 head pair-normals ``||w_c - w_c'||`` to
produce, per pixel, the feature-space flip distance ``d = |margin| / ||w_c -
w_c'||`` — the exact minimal L2 penultimate-patch perturbation that flips the
pairwise argmax ordering (the ``segnet_head_rank4_linear_flipdist_v1`` law).

This is the amplitude-budget field QA80 (ph3 §10.2) needs: inside the budget a
photometric edit provably cannot flip the argmax, so a low-weight pose-legible
luma/chroma term confined to the margin slack is seg-flip-free by construction
(the pp1 correction-band lemma reads this budget in pixel space).

Two producer contracts:

* :func:`exact_flip_distance_field` — the EXACT field given per-pixel
  ``(margin, winner, runner)``.  The runner-up class index is NOT in the current
  gt cache (only argmax + margin magnitude), so the exact n600 field on the BURN
  frames is the deliberately-deferred POST-BURN scorer step (a scorer pass yields
  the top-2 argsort → runner-up).
* :func:`conservative_budget_field` — a SOUND lower bound computable NOW from the
  cache (argmax + margin only): it divides ``|margin|`` by the MAX pair-norm over
  all pairs involving the argmax class, so ``d_cons <= d_exact`` for ANY
  runner-up — staying under it is provably flip-safe.

NO SegNet/PoseNet is imported or run here.  Pointer honesty:
``0.1910828242 [contest-CPU]`` UNMOVED; artifacts ``research_only``,
``score_claim=false``, ``[macOS-CPU advisory]``.  Ledger custody: QA80 in
``.omx/research/ddm_deferral_queue_ledger_20260729.md``; design ph3 §10.2;
law ``segnet_head_rank4_flipdist_20260715``; band lemma
``ddm_pp1_correction_stream_position_band_20260728``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import HEAD_PAIR_NORMS

#: Canonical comma10k class order (MEASURED; MEMORY established finding).  Defined
#: locally to avoid coupling to the actively-edited compose module that also holds it.
CLASS_ORDER: tuple[str, ...] = ("Road", "Lane", "Undrivable", "Movable", "MyCar")


class MarginBudgetError(ValueError):
    """Raised on any geometry / class-index / cache custody violation (fail-closed)."""


def _pair_norm_matrix() -> np.ndarray:
    """Symmetric ``(5, 5)`` ``||w_c - w_c'||`` matrix from the MEASURED head normals; diag = inf."""

    index = {name: i for i, name in enumerate(CLASS_ORDER)}
    matrix = np.full((5, 5), np.nan, dtype=np.float64)
    for key, value in HEAD_PAIR_NORMS.items():
        left, right = key.split("-")
        i, j = index[left], index[right]
        matrix[i, j] = matrix[j, i] = float(value)
    np.fill_diagonal(matrix, np.inf)  # no self-pair flip
    if np.isnan(matrix[np.triu_indices(5, k=1)]).any():
        raise MarginBudgetError("HEAD_PAIR_NORMS does not cover all 10 class pairs")
    return matrix


#: MEASURED pair-norm matrix (feature-space ``||w_c - w_c'||``), diagonal = inf.
PAIR_NORM_MATRIX: np.ndarray = _pair_norm_matrix()
#: MAX finite pair-norm per class (the conservative denominator).
MAX_PAIR_NORM_PER_CLASS: np.ndarray = np.where(
    np.isfinite(PAIR_NORM_MATRIX), PAIR_NORM_MATRIX, -np.inf
).max(axis=1)


def _check_class_array(name: str, arr: np.ndarray) -> None:
    if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= 5):
        raise MarginBudgetError(f"{name} class index out of range [0,5)")


def exact_flip_distance_field(
    margin: Any,
    winner: Any,
    runner: Any,
    *,
    pair_norms: np.ndarray = PAIR_NORM_MATRIX,
) -> np.ndarray:
    """Per-pixel ``d = |margin| / ||w_winner - w_runner||`` (exact rank-4 head law).

    ``margin`` is the pairwise logit margin; ``winner`` / ``runner`` are per-pixel
    class indices in ``[0, 5)`` with ``winner != runner``.  All three broadcast to
    the same shape.
    """

    margin_arr = np.abs(np.asarray(margin, dtype=np.float64))
    winner_arr = np.asarray(winner)
    runner_arr = np.asarray(runner)
    if not (margin_arr.shape == winner_arr.shape == runner_arr.shape):
        raise MarginBudgetError("margin/winner/runner shapes must match")
    _check_class_array("winner", winner_arr)
    _check_class_array("runner", runner_arr)
    if np.any(winner_arr == runner_arr):
        raise MarginBudgetError("winner == runner has no flip pair")
    denom = pair_norms[winner_arr, runner_arr]
    return margin_arr / denom


def conservative_budget_field(
    margin: Any,
    argmax: Any,
    *,
    max_pair_norm_per_class: np.ndarray = MAX_PAIR_NORM_PER_CLASS,
) -> np.ndarray:
    """SOUND per-pixel budget from CACHE (argmax + margin magnitude only).

    Divides ``|margin|`` by the MAX pair-norm over the pairs involving the argmax
    class, so ``d_cons <= d_exact`` for every possible runner-up — a provably-safe
    flip-distance lower bound.
    """

    margin_arr = np.abs(np.asarray(margin, dtype=np.float64))
    argmax_arr = np.asarray(argmax)
    if margin_arr.shape != argmax_arr.shape:
        raise MarginBudgetError("margin/argmax shapes must match")
    _check_class_array("argmax", argmax_arr)
    return margin_arr / max_pair_norm_per_class[argmax_arr]


@dataclass(frozen=True)
class MarginBudgetField:
    """Typed per-pixel flip-distance / amplitude-budget field (advisory, no score authority)."""

    field: np.ndarray
    mode: str  # "exact" | "conservative"
    source: str
    geometry: tuple[int, ...]

    def summary(self) -> dict[str, Any]:
        finite = self.field[np.isfinite(self.field)]
        quant = np.quantile(finite, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]) if finite.size else np.zeros(7)
        return {
            "mode": self.mode,
            "source": self.source,
            "geometry": list(self.geometry),
            "finite_count": int(finite.size),
            "mean": float(finite.mean()) if finite.size else 0.0,
            "quantiles_0_5_25_50_75_95_100": [float(q) for q in quant],
        }

    def save(self, out_path: Path | str) -> dict[str, Any]:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, self.field.astype(np.float32), allow_pickle=False)
        sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
        manifest = {
            "schema": "qa80_margin_budget_field.v1",
            "score_claim": False,
            "axis": "[macOS-CPU advisory]",
            "field_path": out_path.name,
            "field_sha256": sha,
            "field_dtype": "float32",
            **self.summary(),
        }
        (out_path.with_suffix(".manifest.json")).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest


def budget_field_from_gt_cache(
    npz_path: Path | str,
    *,
    mode: str = "conservative",
    pair_index: int | None = None,
) -> MarginBudgetField:
    """Produce a budget field from a cached ``gt_n*.npz`` (``margins`` + ``lstars``).

    ``mode="conservative"`` is the only cache-computable mode: the exact field
    needs the per-pixel runner-up class (a scorer pass → POST-BURN step).
    ``pair_index`` selects one pair; ``None`` uses all cached pairs.
    """

    npz_path = Path(npz_path)
    with np.load(npz_path) as data:
        if "margins" not in data.files:
            raise MarginBudgetError(f"cache {npz_path.name} lacks per-pixel 'margins'")
        if "lstars" not in data.files:
            raise MarginBudgetError(f"cache {npz_path.name} lacks per-pixel 'lstars' argmax")
        margins = np.asarray(data["margins"])
        lstars = np.asarray(data["lstars"])
    if pair_index is not None:
        margins = margins[pair_index]
        lstars = lstars[pair_index]
    if mode != "conservative":
        raise MarginBudgetError(
            "exact mode needs the per-pixel runner-up class (not in the gt cache); "
            "it is the POST-BURN scorer step — use exact_flip_distance_field(margin, winner, runner)"
        )
    field = conservative_budget_field(margins, lstars)
    return MarginBudgetField(field=field, mode="conservative", source=str(npz_path), geometry=tuple(field.shape))
