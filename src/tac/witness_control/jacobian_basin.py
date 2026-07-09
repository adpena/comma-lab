"""ξ→PoseNet Jacobian BASIN telemetry (OBSERVER). Spec: ``.omx/research/pose_jacobian_conditioning_
basin_trigger_formalization_20260709.md`` (build memo: ``pose_jacobian_basin_telemetry_build_20260709.md``).

The render coherence that gates the pose-descent basin is EXACTLY the conditioning of the ξ→PoseNet
Jacobian ``J_ξ = ∂ PoseNet(R(θ,ξ))[:6] / ∂ξ ∈ R^{6x6}``. This module is the INSTRUMENT that measures it.

Two tiers (spec §5):
  * **T0** — render-boundary-gradient energy ``mean(|∇f0|²)`` (+ per-class). ``∇source`` is the
    MULTIPLICATIVE factor in ``J_R`` (spec §2), so this is a near-free, PURE-NUMPY, monotone PROXY for
    ``σ_min(J_ξ)``. No PoseNet, no backward.
  * **T1** — the true conditioning of ``J_ξ`` (σ_min / κ / r_eff / basin_frac), computed via ``mx.vjp``
    (6 VJPs/pair = 6 rows of the 6×6 Jacobian). Authoritative; subsampled + cadenced by the caller.

DISCIPLINE (the build contract B1–B5):
  * **B1 score-neutral by construction** — every function here is a PURE READ. It takes the witness
    render (as fixed MLX/numpy arrays) + the FROZEN PoseNet, returns a metrics dict, and MUTATES NOTHING
    (no weight / optimizer / loss / gradient / RNG / EMA write). The trained artifact is byte-identical
    with the sensor ON vs OFF by construction.
  * **B5 observer-only** — this module MEASURES σ_min(epoch) and the would-have-fired epoch; it does NOT
    actuate the engage-point. The basin TRIGGER (``JACOBIAN_BASIN_ENTRY_TRIGGER``) is a run-2 lever.

The MLX core (``jacobian_xi``) is isolated so ``import mlx`` is deferred; the aggregation / conditioning /
stratification / T0 math is PURE NUMPY (unit-testable at $0, no MLX). ``NON-PROMOTABLE`` advisory: MLX is
never a score authority (CLAUDE.md).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

# The scored PoseNet output width — upstream ``modules.py`` ``compute_distortion`` uses the first
# ``out//2`` == 6 dims of the 12-dim head (MEASURED; CLAUDE.md "Exact scorer architectures").
N_POSE: int = 6

# The run-2 ACTUATOR name (spec §6 DSL leg): the ENTRY-trigger that would engage the pose-finish when
# the basin forms (``median σ_min ≥ f_basin·σ_min^plateau AND basin_frac ≥ q``). Registered here as a
# recognized criterion so the DSL HOLDS the designed lever (config-orphan discipline), but it is
# DEFAULT-OFF / duty-to-measure and is NOT wired to fire on run-1 (B5: sensor earns trust first).
JACOBIAN_BASIN_ENTRY_TRIGGER: str = "jacobian_basin"


@dataclass(frozen=True)
class JacobianBasinConfig:
    """The basin telemetry knobs (spec §6 DSL leg). BOTH tiers are OBSERVERS (read-only, B1).

    ``t0`` (near-free) rides every verdict; ``t1`` (the σ_min authority) is SUBSAMPLED to ``k_pairs``
    stratified by |ego-t| and CADENCED to every ``every``-th verdict (spec B3 default 0.16× a verdict
    eval amortized). ``sigma_floor`` / ``f_basin`` / ``quorum_q`` parametrize the (observer-only)
    would-have-fired criterion; ``f_basin=1.0`` reproduces the current TERMINAL policy exactly."""

    t0: bool = True
    t1: bool = True
    k_pairs: int = 32
    every: int = 4
    stratify_by_t: bool = True
    sigma_floor: float = 1e-4
    f_basin: float = 1.0
    quorum_q: float = 0.8

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.k_pairs < 1:
            problems.append(f"JacobianBasinConfig: k_pairs must be >= 1, got {self.k_pairs}")
        if self.every < 1:
            problems.append(f"JacobianBasinConfig: every must be >= 1, got {self.every}")
        if not (0.0 < self.f_basin <= 1.0):
            problems.append(f"JacobianBasinConfig: f_basin must be in (0,1], got {self.f_basin}")
        if not (0.0 <= self.quorum_q <= 1.0):
            problems.append(f"JacobianBasinConfig: quorum_q must be in [0,1], got {self.quorum_q}")
        if self.sigma_floor < 0.0:
            problems.append(f"JacobianBasinConfig: sigma_floor must be >= 0, got {self.sigma_floor}")
        return problems


# --------------------------------------------------------------------------- #
# T0 — render-boundary-gradient energy (PURE NUMPY, near-free σ_min proxy).
# --------------------------------------------------------------------------- #
def render_grad_energy_map(f0: Any) -> np.ndarray:
    """Per-pixel boundary-gradient energy ``|∇f0|²`` for one frame, shape ``(H,W)``.

    ``∇source`` is the MULTIPLICATIVE factor in ``J_R`` (spec §2), so where this is ~0 (flat cartoon
    interiors) the ξ-motion is INVISIBLE. Forward difference, zero-padded to keep ``(H,W)`` so a class
    argmax map aligns 1:1. Channels (RGB) are summed (total edge energy). Pure read; no MLX."""
    a = np.asarray(f0, dtype=np.float64)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim != 3:
        raise ValueError(f"render_grad_energy_map: expected (H,W) or (H,W,C), got shape {a.shape}")
    h, w, _c = a.shape
    gx = np.zeros_like(a)
    gy = np.zeros_like(a)
    if w >= 2:
        gx[:, : w - 1, :] = a[:, 1:, :] - a[:, : w - 1, :]
    if h >= 2:
        gy[: h - 1, :, :] = a[1:, :, :] - a[: h - 1, :, :]
    return np.sum(gx * gx + gy * gy, axis=2)  # (H,W) total-channel |∇|²


def render_grad_energy(f0: Any) -> float:
    """Scalar T0 proxy: ``mean(|∇f0|²)`` over all pixels (near-free)."""
    m = render_grad_energy_map(f0)
    return float(np.mean(m)) if m.size else 0.0


def render_grad_energy_per_class(f0: Any, argmax: Any, n_classes: int = 5) -> dict[int, float]:
    """Per-class T0 proxy: mean ``|∇f0|²`` restricted to pixels whose SegNet argmax is class ``c``.

    Connects to the v8 per-class carriers (each class's render coherence). ``argmax`` is the realized
    ``(H,W)`` SegNet argmax at the SAME resolution as ``f0`` (else falls back to nearest-size crop). NaN
    for a class absent from the frame (NO-FAKE: never a fabricated 0)."""
    emap = render_grad_energy_map(f0)
    am = np.asarray(argmax)
    if am.shape != emap.shape:
        h = min(emap.shape[0], am.shape[0])
        w = min(emap.shape[1], am.shape[1])
        emap = emap[:h, :w]
        am = am[:h, :w]
    out: dict[int, float] = {}
    for c in range(int(n_classes)):
        mask = am == c
        cnt = int(mask.sum())
        out[c] = float(emap[mask].mean()) if cnt else float("nan")
    return out


def t0_fields(
    f0_list: Sequence[Any], argmax_list: "Sequence[Any] | None" = None, n_classes: int = 5,
) -> dict:
    """Aggregate the T0 proxy over a set of frames. Returns ``{render_grad_energy, per_class}`` where
    ``per_class`` is a NaN-omitting class→mean-energy dict (present only when argmax maps are given)."""
    energies = [render_grad_energy(f0) for f0 in f0_list]
    out: dict[str, Any] = {
        "render_grad_energy": (float(np.mean(energies)) if energies else 0.0),
        "render_grad_energy_p10": (float(np.percentile(energies, 10.0)) if energies else 0.0),
        "n_frames": int(len(energies)),
    }
    if argmax_list is not None and len(argmax_list) == len(f0_list) and f0_list:
        acc: dict[int, list[float]] = {c: [] for c in range(int(n_classes))}
        for f0, am in zip(f0_list, argmax_list):
            for c, e in render_grad_energy_per_class(f0, am, n_classes).items():
                if math.isfinite(e):
                    acc[c].append(e)
        out["per_class"] = {c: (float(np.mean(v)) if v else None) for c, v in acc.items()}
    return out


# --------------------------------------------------------------------------- #
# T1 — the true J_ξ conditioning (mx.vjp isolated; SVD/aggregation pure numpy).
# --------------------------------------------------------------------------- #
def jacobian_xi(p_of_xi: Callable[[Any], Any], xi0: Any, *, mx: Any = None) -> np.ndarray:
    """``J_ξ = ∂ p / ∂ξ ∈ R^{6x6}`` at base point ``xi0`` via ``mx.vjp`` (6 VJPs = 6 rows).

    ``p_of_xi(xi) -> (6,)`` is the composed, MLX-differentiable ``PoseNet(R(θ,ξ))[:6]`` (the render θ is
    FROZEN — it enters ``p_of_xi`` as a stop-gradient constant; only ξ is differentiated). For the i-th
    cotangent ``e_i`` the VJP returns ``e_iᵀ J = row i`` of J; stacking the 6 rows gives the Jacobian.

    Pure read: ``mx.vjp`` differentiates ONLY w.r.t. ``xi0`` — it never touches the witness weights or
    the optimizer state (B1 score-neutral). Returns a ``(6,6)`` numpy array (advisory / NON-PROMOTABLE)."""
    if mx is None:
        import mlx.core as mx  # noqa: PLC0415
    xi = mx.array(np.asarray(xi0, dtype=np.float32).reshape(-1))
    if int(xi.shape[0]) != N_POSE:
        raise ValueError(f"jacobian_xi: xi0 must have {N_POSE} entries, got shape {tuple(xi.shape)}")
    rows: list[np.ndarray] = []
    for i in range(N_POSE):
        cot = np.zeros(N_POSE, dtype=np.float32)
        cot[i] = 1.0
        _out, (g,) = mx.vjp(p_of_xi, (xi,), (mx.array(cot),))
        mx.eval(g)
        row = np.asarray(g, dtype=np.float64).reshape(-1)
        if row.shape[0] != N_POSE:
            raise ValueError(f"jacobian_xi: VJP row has width {row.shape[0]} (expected {N_POSE})")
        rows.append(row)
    return np.stack(rows, axis=0)  # (6,6): row i = ∂p_i/∂ξ


def conditioning(jac: np.ndarray, *, rtol: float = 1e-6) -> dict:
    """SVD conditioning of ``J_ξ`` (spec §1). Returns σ_min / σ_max / cond / r_eff / rank / sigmas.

    ``σ_min`` is THE basin variable (its right singular vector is the LEAST-observable ξ direction).
    ``r_eff = exp(H(σ̃))`` is a threshold-free soft rank; ``rank`` counts σ above ``rtol·σ_max``. Pure
    numpy; non-finite J => a fail-safe all-zero conditioning (NO-FAKE: flagged, never a fabricated σ)."""
    j = np.asarray(jac, dtype=np.float64)
    if j.shape != (N_POSE, N_POSE) or not np.all(np.isfinite(j)):
        return {"sigma_min": 0.0, "sigma_max": 0.0, "cond": float("inf"),
                "r_eff": 0.0, "rank": 0, "sigmas": [0.0] * N_POSE,
                "degenerate": True}
    s = np.linalg.svd(j, compute_uv=False)
    s = np.sort(np.asarray(s, dtype=np.float64))[::-1]  # descending
    sigma_max = float(s[0])
    sigma_min = float(s[-1])
    tol = rtol * sigma_max
    rank = int(np.count_nonzero(s > tol))
    ssum = float(s.sum())
    if ssum > 0.0:
        p = s / ssum
        nz = p[p > 0.0]
        ent = float(-np.sum(nz * np.log(nz)))
        r_eff = float(math.exp(ent))
    else:
        r_eff = 0.0
    cond = (sigma_max / sigma_min) if sigma_min > 0.0 else float("inf")
    return {"sigma_min": sigma_min, "sigma_max": sigma_max, "cond": cond,
            "r_eff": r_eff, "rank": rank, "sigmas": [float(x) for x in s],
            "degenerate": False}


# --------------------------------------------------------------------------- #
# Stratified subsample + aggregation + the (observer-only) basin criterion.
# --------------------------------------------------------------------------- #
def motion_magnitude(pose6: Any) -> float:
    """Per-pair ego-motion magnitude for |ego-t| stratification: the L2 norm of the 6-dim GT pose
    target. Low-motion pairs condition LAST (d_pose anti-correlates with |ego-t|, MEASURED corr
    −0.45/−0.68), so they MUST be represented in the subsample (else the median hides the tail)."""
    p = np.asarray(pose6, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(p))


def stratified_indices_by_motion(motions: Sequence[float], k: int) -> list[int]:
    """Pick ``k`` pair indices spanning the |ego-t| range (including the low + high tails), by taking
    ``k`` evenly-spaced quantile positions of the motion-sorted order. Deterministic (no RNG => B1).
    ``k >= n`` returns all indices."""
    n = len(motions)
    if n == 0:
        return []
    if k >= n:
        return list(range(n))
    order = list(np.argsort(np.asarray(motions, dtype=np.float64), kind="stable"))
    # evenly spaced positions in [0, n-1], inclusive of both endpoints (low & high motion tails).
    pos = np.linspace(0, n - 1, num=int(k))
    picks = sorted({int(order[int(round(p))]) for p in pos})
    # linspace collisions can drop below k on tight ranges; backfill from the sorted order.
    if len(picks) < int(k):
        for idx in order:
            if idx not in picks:
                picks.append(int(idx))
            if len(picks) >= int(k):
                break
        picks = sorted(picks)
    return picks


def aggregate_conditioning(conds: Sequence[dict], *, sigma_floor: float) -> dict:
    """Aggregate per-pair conditioning dicts into the basin level: median/p10 σ_min (p10 = the tail),
    median cond, median r_eff, basin_frac = frac[σ_min ≥ sigma_floor]. Empty => a fail-safe zero row."""
    smins = np.asarray([float(c.get("sigma_min", 0.0)) for c in conds], dtype=np.float64)
    conds_k = np.asarray([float(c.get("cond", float("inf"))) for c in conds], dtype=np.float64)
    reffs = np.asarray([float(c.get("r_eff", 0.0)) for c in conds], dtype=np.float64)
    if smins.size == 0:
        return {"median_sigma_min": 0.0, "p10_sigma_min": 0.0, "median_cond": float("inf"),
                "median_r_eff": 0.0, "basin_frac": 0.0, "n_pairs": 0}
    finite_cond = conds_k[np.isfinite(conds_k)]
    return {
        "median_sigma_min": float(np.median(smins)),
        "p10_sigma_min": float(np.percentile(smins, 10.0)),
        "median_cond": (float(np.median(finite_cond)) if finite_cond.size else float("inf")),
        "median_r_eff": float(np.median(reffs)),
        "basin_frac": float(np.mean(smins >= float(sigma_floor))),
        "n_pairs": int(smins.size),
    }


def would_have_fired(
    agg: dict, *, plateau_est: float, f_basin: float, quorum_q: float,
) -> "bool | None":
    """The OBSERVER-ONLY basin criterion (spec §3): ``median σ_min ≥ f_basin·σ_min^plateau AND
    basin_frac ≥ q``. ``plateau_est`` is the running MAX of median σ_min the telemetry has observed so
    far (the plateau is only KNOWN offline at run end, so this live answer is PROVISIONAL — labeled as
    such in the row). Returns None until a positive plateau estimate exists."""
    if not (plateau_est > 0.0):
        return None
    return bool(agg.get("median_sigma_min", 0.0) >= float(f_basin) * float(plateau_est)
                and agg.get("basin_frac", 0.0) >= float(quorum_q))
