"""Per-class-pair sigma_ij LENGTH-WEIGHT resolution for the Chan-Vese length term.

The consumption path for the MEASURED Young's-law junction fit
(``junction_young_angle_sigma_fit_v1``; artifact
``experiments/results/solver_pack_20260707/junction_sigma/junction_sigma_fit.json``,
landed commit 3571e5b65; memo ``.omx/research/solver_pack_junction_sigma_powerlaw_20260707.md``).

The trainer's length term lives on the DECISION MARGIN m = phi_top1 - phi_top2 and
localizes at the {m=0} inter-class boundary via delta_eps(m)*|grad m|. At each pixel the
interface present is exactly the (argmax-top1, argmax-top2) class PAIR, so a pairwise
surface-tension matrix sigma_ij is FAITHFULLY expressible as a per-pixel multiplicative
weight sigma[top1, top2] on the length integrand — no per-class projection/approximation
is needed (the decomposition the loss already has IS per-interface).

Resolution contract (``resolve_length_sigma_matrix``):
- ``"all-ones"`` (the trainer default) -> ``None``: the caller takes its PRE-EXISTING code
  path untouched (byte-identical BY CODE PATH, not merely by multiply-by-1.0).
- ``"fitted-20260707"`` -> the deterministic MEASURED preset below (NaN-unobserved pairs
  filled with 1.0 = the all-ones null; geometric-mean-1 gauge).
- anything else -> a JSON path holding either a raw 5x5 nested list or a dict with a
  ``"sigma_matrix_5x5"`` key (so the fit tool's own JSON can be passed directly). NaN
  entries are filled with 1.0 (the null) before validation.

Validation is fail-closed: wrong shape, non-symmetric, or non-positive/non-finite
OFF-DIAGONAL entries raise ``ValueError`` (the diagonal is never gathered — argsort
top1 != top2 always — and is normalized to 1.0).

Advisory geometry calibration, NOT a score; pointer contest-CPU 0.19110 UNMOVED.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

N_CLASSES = 5  # canonical order Road0 / Lane1 / Undrivable2 / Movable3 / MyCar4

PRESET_ALL_ONES = "all-ones"
PRESET_FITTED_20260707 = "fitted-20260707"
PRESETS = (PRESET_ALL_ONES, PRESET_FITTED_20260707)

# MEASURED fitted sigma_ij, FULL precision from the primary artifact
# experiments/results/solver_pack_20260707/junction_sigma/junction_sigma_fit.json
# (commit 3571e5b65; tools/fit_junction_sigma_youngs_law.py, n600 cached GT argmax,
# geometric-mean-1 gauge, bootstrap-200 CIs; [macOS-CPU advisory] NON-PROMOTABLE).
# Registered law: tac.canonical_equations.junction_young_sigma_and_powerlaw_exit_20260707
# (rounded constants there; cross-consistency asserted in tests). Unobserved pairs
# (Lane-Movable, Undrivable-MyCar, Movable-MyCar: <30 junctions) are the all-ones null 1.0;
# the diagonal is 1.0 (never gathered).
_S_RL = 0.3771195466360733   # Road-Lane        ci95 [0.317, 0.441]  EXCLUDES 1.0
_S_RU = 1.0848087450168646   # Road-Undrivable  ci95 [0.994, 1.206]
_S_RM = 1.0062627915225708   # Road-Movable     ci95 [0.918, 1.125]
_S_RC = 1.7791690170773755   # Road-MyCar       ci95 [1.459, 2.150]  EXCLUDES 1.0
_S_LU = 0.7381986449045815   # Lane-Undrivable  ci95 [0.568, 0.922]  EXCLUDES 1.0
_S_LC = 1.764344211480968    # Lane-MyCar       ci95 [1.434, 2.119]  EXCLUDES 1.0
_S_UM = 1.0482927871960461   # Undrivable-Movable ci95 [0.955, 1.185]

FITTED_20260707_MATRIX = (
    # Road   Lane   Undriv  Movable MyCar
    (1.0, _S_RL, _S_RU, _S_RM, _S_RC),
    (_S_RL, 1.0, _S_LU, 1.0, _S_LC),
    (_S_RU, _S_LU, 1.0, _S_UM, 1.0),
    (_S_RM, 1.0, _S_UM, 1.0, 1.0),
    (_S_RC, _S_LC, 1.0, 1.0, 1.0),
)


def validate_sigma_matrix(a: np.ndarray, *, source: str = "sigma matrix") -> None:
    """Fail-closed structural validation. Raises ``ValueError`` on: wrong shape,
    non-symmetric, or any OFF-DIAGONAL entry that is non-finite or <= 0."""
    if a.shape != (N_CLASSES, N_CLASSES):
        raise ValueError(
            f"--length-sigma-matrix: {source} must be {N_CLASSES}x{N_CLASSES} "
            f"(canonical class order Road0/Lane1/Undrivable2/Movable3/MyCar4), got shape {a.shape}")
    off = ~np.eye(N_CLASSES, dtype=bool)
    vals = a[off]
    if not np.all(np.isfinite(vals)):
        raise ValueError(
            f"--length-sigma-matrix: {source} has non-finite off-diagonal entries "
            "(NaN unobserved pairs must be pre-filled with 1.0; raw NaN/inf is refused fail-closed)")
    if not np.all(vals > 0.0):
        raise ValueError(
            f"--length-sigma-matrix: {source} has non-positive off-diagonal entries "
            f"(min={vals.min()!r}); surface tensions are strictly positive weights")
    if not np.array_equal(a[off], a.T[off]):
        raise ValueError(
            f"--length-sigma-matrix: {source} is not symmetric (sigma_ij = sigma_ji; the "
            "interface between classes i and j is one interface)")


def _fill_nan_with_null(a: np.ndarray) -> np.ndarray:
    """Unobserved (NaN) pairs -> 1.0 (the all-ones null); diagonal -> 1.0 (never gathered)."""
    out = a.astype(np.float64, copy=True)
    out[np.isnan(out)] = 1.0
    np.fill_diagonal(out, 1.0)
    return out


def load_sigma_matrix_json(path: str | Path) -> np.ndarray:
    """Load a 5x5 sigma matrix from JSON: raw nested list, or a dict carrying
    ``"sigma_matrix_5x5"`` (the fit tool's own JSON passes directly). NaN filled with 1.0."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(
            f"--length-sigma-matrix: {str(p)!r} is neither a preset {PRESETS} nor an existing "
            "JSON file")
    try:
        obj = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"--length-sigma-matrix: {p} is not valid JSON: {exc}") from exc
    if isinstance(obj, dict):
        fit = obj.get("fit")
        node = fit.get("sigma_matrix_5x5") if isinstance(fit, dict) else None
        node = obj.get("sigma_matrix_5x5", node)
        if node is None:
            raise ValueError(
                f"--length-sigma-matrix: JSON dict at {p} has no 'sigma_matrix_5x5' key "
                "(top-level or under 'fit'); pass a raw 5x5 list or the fit tool's JSON")
        obj = node
    try:
        a = np.asarray(obj, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"--length-sigma-matrix: {p} does not parse as a numeric matrix: {exc}") from exc
    a = _fill_nan_with_null(a) if a.ndim == 2 and a.shape == (N_CLASSES, N_CLASSES) else a
    validate_sigma_matrix(a, source=f"JSON at {p}")
    return a


def resolve_length_sigma_matrix(spec: str) -> np.ndarray | None:
    """Resolve the ``--length-sigma-matrix`` flag value.

    Returns ``None`` for ``"all-ones"`` (the default: caller MUST take its pre-existing
    unweighted code path -> byte-identical by construction), else a validated (5,5)
    float64 numpy array (the fitted preset or a loaded JSON path).
    """
    s = str(spec).strip()
    if s == PRESET_ALL_ONES:
        return None
    if s == PRESET_FITTED_20260707:
        a = np.asarray(FITTED_20260707_MATRIX, dtype=np.float64)
        validate_sigma_matrix(a, source=f"preset {PRESET_FITTED_20260707!r}")
        return a
    return load_sigma_matrix_json(s)


def describe_length_sigma(spec: str) -> dict:
    """Provenance record for the resolved spec (defaults are DERIVED + RECORDED)."""
    a = resolve_length_sigma_matrix(spec)
    if a is None:
        return {"spec": PRESET_ALL_ONES, "active": False,
                "note": "uniform length weight (Herring 120-120-120 null); byte-identical path"}
    gm = math.exp(float(np.mean(np.log(a[~np.eye(N_CLASSES, dtype=bool)]))))
    return {"spec": str(spec), "active": True,
            "sigma_road_lane": float(a[0, 1]), "sigma_road_mycar": float(a[0, 4]),
            "offdiag_geomean": gm,
            "provenance": "junction_young_angle_sigma_fit_v1 (commit 3571e5b65)"}


__all__ = [
    "FITTED_20260707_MATRIX",
    "N_CLASSES",
    "PRESETS",
    "PRESET_ALL_ONES",
    "PRESET_FITTED_20260707",
    "describe_length_sigma",
    "load_sigma_matrix_json",
    "resolve_length_sigma_matrix",
    "validate_sigma_matrix",
]
