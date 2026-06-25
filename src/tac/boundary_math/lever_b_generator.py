# SPDX-License-Identifier: MIT
"""LEVER-B score-native seg generator — reusable checkpointed carrier + residual tools.

The lever-B smoke (`tools/lever_b_score_native_argmax_smoke.py`) PROVED the seg term is
generatable (d_seg=0.00826 over 600 pairs, a 63.8 KB quantized blob) but it did NOT save the
trained generator — only `smoke_result.json`. The campaign's reactivation of the
`closed_spec_boundary_solver.v1` (task #55, DEFERRED on the frontier base because its residual
is salt-and-pepper / unrepairable) needs the GENERATOR'S argmax residual to run the solver on.

This module is the canonical, checkpointable home for:

  1. ``ScoreNativeSegGenerator`` — the MLX coordinate-INR + FiLM-per-pair-mod + 5-class-logit head
     (identical math to the smoke; ONE definition now, the smoke can import this).
  2. ``numpy_reference_forward`` — the pure-numpy portable mirror (argmax-parity contract).
  3. ``save_generator_npz`` / ``load_generator_npz`` — the missing artifact: persist the trained
     weights + the deterministic Fourier table + per-pair mod codes as a portable .npz, so the
     solver / legal-frame bridge can reconstruct the generator's argmax WITHOUT retraining.
  4. ``generator_argmax`` — run the (numpy) generator forward over a coord grid → per-pair argmax.
  5. ``residual_component_stats`` — the DECISIVE Step-1 measurement: connected components of the
     generator's flip set (A_b_generator != L*), with the contiguity histogram that decides
     repairability (contiguous patches = repairable; salt-and-pepper = not, per the frontier-base
     finding).

COMPUTE-SUBSTRATE LAW (CLAUDE.md): the MLX forward is the fast path; the numpy forward is the
portability reference (argmax-parity, NOT bit-exact logits). NO MPS. d_seg / argmax read on the
exact frozen quantities only. Evidence ``[macOS-MLX research-signal]`` for the generator forward;
``[macOS-CPU advisory]`` once the synthesized frame is scored by the exact SegNet. Non-promotable.

NO-FAKE (class 2): the residual stats are computed from the ACTUAL generator forward over the
ACTUAL GT argmax targets — a stub that returns constant counts FAILS the tests (which assert the
component histogram matches a constructed-residual ground truth).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ── canonical Fourier-table seed (deterministic free table at inflate time) ──
_FOURIER_SEED = 0


def build_coords(h: int, w: int) -> np.ndarray:
    """The (H*W, 2) pixel coordinate grid in [-1, 1] (row-major, gx, gy)."""

    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)


def deterministic_fourier_B(n_fourier: int, fourier_sigma: float) -> np.ndarray:
    """The deterministic (seeded) Fourier feature matrix (2, n_fourier).

    NOT trained — a free deterministic table reconstructed identically at inflate time
    from the seed + (n_fourier, sigma). Identical to the smoke's ``_fourier_B``.
    """

    rng = np.random.default_rng(_FOURIER_SEED)
    return (rng.standard_normal((2, n_fourier)) * fourier_sigma).astype(np.float32)


# ---------------------------------------------------------------------------
# Deep-math directional / all-class-boundary feature helpers (2026-06-25 capstone).
#
# MEASURED on targets_n600: the witness's argmax flips concentrate at ALL inter-class
# boundaries (a codim-1 curve = union of every class edge), NOT just the class-1 lane
# (only ~19% of small-margin px are class-1; ~50% are class-0). The decisive d_seg lever
# is an oriented/anisotropic ("curvelet") Fourier basis oriented to that ALL-CLASS boundary
# tangent: high frequency ACROSS the edge, low frequency ALONG it. MEASURED win:
# -48% d_seg (dir-only) to -64% (dir + modest capacity) over the isotropic-only control
# at the n96 regime (see .omx/research/witness_capstone_deepmath_levers_20260625.md).
# These helpers are the reusable, portable (numpy/scipy, 0-byte train-time prior) home for
# that lever; the prototype is tools/witness_capstone_deepmath_smoke.py.
# ---------------------------------------------------------------------------
def all_class_boundary_mask(argmax_hw: np.ndarray) -> np.ndarray:
    """Pixels on ANY inter-class boundary (argmax differs from a 4-neighbor). Pure numpy."""

    a = np.asarray(argmax_hw)
    b = np.zeros(a.shape, dtype=bool)
    b[:-1, :] |= a[:-1, :] != a[1:, :]
    b[1:, :] |= a[:-1, :] != a[1:, :]
    b[:, :-1] |= a[:, :-1] != a[:, 1:]
    b[:, 1:] |= a[:, :-1] != a[:, 1:]
    return b


def all_class_boundary_proximity_and_tangent(
    argmax_hw: np.ndarray, tau: float = 4.0
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel boundary-proximity key ``exp(-d/tau)`` + unit tangent of the ALL-CLASS boundary.

    The distance ``d`` is the (borrowed) scipy EDT to the nearest inter-class edge; the tangent is
    the 90-degree rotation of grad(d) (the boundary normal). Returns
    ``(prox (H,W) in [0,1], tangent (H,W,2))``. Pure CPU/numpy/scipy, 0 archive bytes (a fixed
    function of the GT argmax the witness reproduces). Borrowed: scipy EDT.
    """

    from scipy import ndimage

    a = np.asarray(argmax_hw)
    h, w = a.shape
    bnd = all_class_boundary_mask(a)
    if not bnd.any():
        diag = float(np.hypot(h, w))
        prox = np.exp(-np.full((h, w), diag, np.float32) / tau).astype(np.float32)
        tang = np.zeros((h, w, 2), np.float32)
        tang[..., 0] = 1.0
        return prox, tang
    dist = ndimage.distance_transform_edt(~bnd).astype(np.float32)
    prox = np.exp(-dist / tau).astype(np.float32)
    gy = np.zeros_like(dist)
    gx = np.zeros_like(dist)
    gy[1:-1, :] = (dist[2:, :] - dist[:-2, :]) * 0.5
    gx[:, 1:-1] = (dist[:, 2:] - dist[:, :-2]) * 0.5
    tx = -gy
    ty = gx
    nrm = np.sqrt(tx * tx + ty * ty)
    flat = nrm < 1e-6
    nrm = np.maximum(nrm, 1e-8)
    tx = tx / nrm
    ty = ty / nrm
    tx[flat] = 1.0
    ty[flat] = 0.0
    return prox, np.stack([tx, ty], axis=-1).astype(np.float32)


def directional_fourier_feats(
    coords: np.ndarray,
    tangent: np.ndarray,
    n_freqs: int = 6,
    freq_across: float = 32.0,
    freq_along: float = 4.0,
) -> np.ndarray:
    """Oriented/anisotropic ("curvelet") Fourier features ``(P, 4*n_freqs)``.

    For each pixel, build the local frame ``(tangent t, normal n)`` and encode the coordinate's
    projection onto each with DIFFERENT frequency scales: HIGH across the edge (``freq_across``,
    normal) and LOW along it (``freq_along``, tangent). Deterministic (no random projection) so it
    is identical at train and inflate and adds ZERO archive bytes. ``coords``/``tangent`` are
    ``(P, 2)``. Borrowed mechanism: AFPE (Kuckelhaus 2025) / Tancik 2020 / steerable filters /
    curvelets; OURS: orienting to the all-class GT boundary tangent for the argmax witness.
    """

    coords = np.asarray(coords)
    tangent = np.asarray(tangent)
    tx = tangent[:, 0]
    ty = tangent[:, 1]
    nx = ty
    ny = -tx
    cx = coords[:, 0]
    cy = coords[:, 1]
    u_t = cx * tx + cy * ty
    u_n = cx * nx + cy * ny
    two_pi = 2.0 * np.pi
    feats: list[np.ndarray] = []
    for k in range(n_freqs):
        fa = float(freq_along) * (2.0**k)
        fc = float(freq_across) * (2.0**k)
        feats.append(np.sin(two_pi * fa * u_t))
        feats.append(np.cos(two_pi * fa * u_t))
        feats.append(np.sin(two_pi * fc * u_n))
        feats.append(np.cos(two_pi * fc * u_n))
    return np.stack(feats, axis=-1).astype(np.float32)


@dataclass(frozen=True)
class GeneratorConfig:
    """The architecture hyperparameters that fully determine the generator shape."""

    num_pairs: int
    n_fourier: int = 16
    hidden_dim: int = 96
    n_hidden: int = 4
    mod_dim: int = 32
    n_classes: int = 5
    fourier_sigma: float = 8.0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "num_pairs": self.num_pairs,
            "n_fourier": self.n_fourier,
            "hidden_dim": self.hidden_dim,
            "n_hidden": self.n_hidden,
            "mod_dim": self.mod_dim,
            "n_classes": self.n_classes,
            "fourier_sigma": self.fourier_sigma,
        }


# ---------------------------------------------------------------------------
# Pure-numpy forward (the PORTABLE reference; argmax is the score-native quantity).
# ---------------------------------------------------------------------------
def numpy_reference_forward(
    params: dict[str, np.ndarray],
    fourier_B: np.ndarray,
    coords: np.ndarray,
    mod_vec: np.ndarray,
    n_hidden: int,
    hidden_dim: int,
) -> np.ndarray:
    """Pure-numpy mirror of ``ScoreNativeSegGenerator.__call__`` (float64 accumulation).

    The contract this realizes is ARGMAX-parity (the d_seg quantity), invariant to the
    fp32-vs-fp64 accumulation drift on a trained net. Returns (P, n_classes) logits.
    """

    coords = coords.astype(np.float64)
    fourier_B = fourier_B.astype(np.float64)
    mod_vec = mod_vec.astype(np.float64)
    p = {k: v.astype(np.float64) for k, v in params.items()}
    # errstate silences benign Accelerate fast-math denormal flags on the dense
    # products; the trained-net inputs are finite, the result is exact float64.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        proj = coords @ fourier_B
        feat = np.concatenate([np.sin(proj), np.cos(proj)], axis=-1)
        h = np.maximum(feat @ p["in_proj.weight"].T + p["in_proj.bias"], 0.0)
        film = mod_vec @ p["film.weight"].T + p["film.bias"]
        film = film.reshape(n_hidden, 2, hidden_dim)
        for li in range(n_hidden):
            scale = 1.0 + film[li, 0]
            shift = film[li, 1]
            h = np.maximum((h @ p[f"hidden.{li}.weight"].T + p[f"hidden.{li}.bias"]) * scale + shift, 0.0)
        return h @ p["out.weight"].T + p["out.bias"]


def generator_argmax(
    params: dict[str, np.ndarray],
    cfg: GeneratorConfig,
    coords: np.ndarray,
    pair_idx: int,
    h: int,
    w: int,
) -> np.ndarray:
    """Per-pair generator argmax map (H, W) uint8 — the numpy-portable forward.

    ``params`` must contain ``mod`` (num_pairs, mod_dim) + ``in_proj/film/hidden.*/out``.
    Reconstructs the deterministic Fourier table from cfg (free table).
    """

    fourier_B = deterministic_fourier_B(cfg.n_fourier, cfg.fourier_sigma)
    mod_vec = np.asarray(params["mod"])[pair_idx]
    logits = numpy_reference_forward(params, fourier_B, coords, mod_vec, cfg.n_hidden, cfg.hidden_dim)
    return logits.argmax(axis=-1).reshape(h, w).astype(np.uint8)


# ---------------------------------------------------------------------------
# Checkpoint I/O — the missing artifact (portable .npz).
# ---------------------------------------------------------------------------
def save_generator_npz(path: Path, params: dict[str, np.ndarray], cfg: GeneratorConfig) -> Path:
    """Persist trained weights + per-pair mod + cfg to a portable .npz (NO /tmp)."""

    path = Path(path)
    _refuse_tmp(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {f"param::{k}": np.asarray(v).astype(np.float32) for k, v in params.items()}
    for k, v in cfg.to_dict().items():
        flat[f"cfg::{k}"] = np.asarray(v)
    np.savez(path, **flat)
    return path


def load_generator_npz(path: Path) -> tuple[dict[str, np.ndarray], GeneratorConfig]:
    """Load a generator checkpoint saved by :func:`save_generator_npz`."""

    path = Path(path)
    with np.load(path) as z:
        params = {k[len("param::"):]: z[k] for k in z.files if k.startswith("param::")}
        cfg_kv = {k[len("cfg::"):]: z[k] for k in z.files if k.startswith("cfg::")}
    cfg = GeneratorConfig(
        num_pairs=int(cfg_kv["num_pairs"]),
        n_fourier=int(cfg_kv["n_fourier"]),
        hidden_dim=int(cfg_kv["hidden_dim"]),
        n_hidden=int(cfg_kv["n_hidden"]),
        mod_dim=int(cfg_kv["mod_dim"]),
        n_classes=int(cfg_kv["n_classes"]),
        fourier_sigma=float(cfg_kv["fourier_sigma"]),
    )
    return params, cfg


_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD tier per CLAUDE.md.")


# ---------------------------------------------------------------------------
# Step-1 DECISIVE measurement: the generator residual contiguity.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResidualComponentStats:
    """Connected-component structure of the generator's flip set (A_b_generator != L*).

    This is THE decision input: the frontier base residual was 95% single-pixel scattered
    (unrepairable — the SegNet receptive field flips ~2 correct neighbours per repair). If
    the generator residual is instead CONTIGUOUS multi-pixel patches (a smooth net's
    under-fit regions), the boundary solver's contour/region atoms can pay rent.
    """

    n_flips: int
    n_components: int
    single_pixel_components: int
    single_pixel_fraction: float
    largest_component_pixels: int
    mean_component_pixels: float
    median_component_pixels: float
    size_histogram: dict[str, int]
    contiguous_fraction: float  # fraction of FLIP PIXELS in components of size >= 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_flips": self.n_flips,
            "n_components": self.n_components,
            "single_pixel_components": self.single_pixel_components,
            "single_pixel_fraction": self.single_pixel_fraction,
            "largest_component_pixels": self.largest_component_pixels,
            "mean_component_pixels": self.mean_component_pixels,
            "median_component_pixels": self.median_component_pixels,
            "size_histogram": self.size_histogram,
            "contiguous_fraction": self.contiguous_fraction,
        }


_CONN4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def residual_component_stats(base_argmax: np.ndarray, target_argmax: np.ndarray) -> ResidualComponentStats:
    """Connected components of the flip set (base != target) with the contiguity histogram.

    4-connectivity (matches the boundary solver's flip-component connectivity). The
    ``contiguous_fraction`` (flip pixels in components >= 4 px) is the single number the
    repairability decision keys on (the frontier base had ~0; a repairable base has it high).
    """

    from scipy import ndimage

    base = np.asarray(base_argmax).astype(np.int64)
    tgt = np.asarray(target_argmax).astype(np.int64)
    if base.shape != tgt.shape:
        raise ValueError(f"base {base.shape} != target {tgt.shape}")
    flip = base != tgt
    n_flips = int(flip.sum())
    labelled, n = ndimage.label(flip, structure=_CONN4)
    if n == 0:
        return ResidualComponentStats(
            n_flips=0, n_components=0, single_pixel_components=0, single_pixel_fraction=0.0,
            largest_component_pixels=0, mean_component_pixels=0.0, median_component_pixels=0.0,
            size_histogram={}, contiguous_fraction=0.0,
        )
    sizes = np.bincount(labelled.ravel())[1:]  # drop background (0)
    single = int((sizes == 1).sum())
    bands = {
        "1px": int((sizes == 1).sum()),
        "2px": int((sizes == 2).sum()),
        "3px": int((sizes == 3).sum()),
        "4-9px": int(((sizes >= 4) & (sizes <= 9)).sum()),
        "10-49px": int(((sizes >= 10) & (sizes <= 49)).sum()),
        "50+px": int((sizes >= 50).sum()),
    }
    contiguous_px = int(sizes[sizes >= 4].sum())
    return ResidualComponentStats(
        n_flips=n_flips,
        n_components=int(n),
        single_pixel_components=single,
        single_pixel_fraction=single / int(n),
        largest_component_pixels=int(sizes.max()),
        mean_component_pixels=float(sizes.mean()),
        median_component_pixels=float(np.median(sizes)),
        size_histogram=bands,
        contiguous_fraction=contiguous_px / n_flips if n_flips else 0.0,
    )


def aggregate_residual_stats(stats_list: list[ResidualComponentStats]) -> dict[str, Any]:
    """Aggregate per-pair residual stats into the campaign-level decision summary."""

    if not stats_list:
        return {"n_pairs": 0}
    tot_flips = sum(s.n_flips for s in stats_list)
    tot_comps = sum(s.n_components for s in stats_list)
    tot_single = sum(s.single_pixel_components for s in stats_list)
    tot_contig_px = sum(round(s.contiguous_fraction * s.n_flips) for s in stats_list)
    hist: dict[str, int] = {}
    for s in stats_list:
        for k, v in s.size_histogram.items():
            hist[k] = hist.get(k, 0) + v
    return {
        "n_pairs": len(stats_list),
        "total_flips": tot_flips,
        "total_components": tot_comps,
        "single_pixel_components": tot_single,
        "single_pixel_fraction": tot_single / tot_comps if tot_comps else 0.0,
        "contiguous_flip_fraction": tot_contig_px / tot_flips if tot_flips else 0.0,
        "size_histogram": hist,
        "mean_flips_per_pair": tot_flips / len(stats_list),
        "largest_component_pixels": max((s.largest_component_pixels for s in stats_list), default=0),
    }


__all__ = [
    "GeneratorConfig",
    "ResidualComponentStats",
    "aggregate_residual_stats",
    "all_class_boundary_mask",
    "all_class_boundary_proximity_and_tangent",
    "build_coords",
    "deterministic_fourier_B",
    "directional_fourier_feats",
    "generator_argmax",
    "load_generator_npz",
    "numpy_reference_forward",
    "residual_component_stats",
    "save_generator_npz",
]
