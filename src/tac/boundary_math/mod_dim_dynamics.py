"""MOD-DIM DYNAMICS telemetry — score-neutral, read-only spectral + per-dim introspection of the
witness's per-pair latent (``code``) table.

WHY (operator 2026-07-08 "will the next run provide sufficient telemetry to understand the mod dim
dynamics and how it's working under the hood and exploitation"): a post-hoc autopsy of the trained
mod-32 latents found effective rank ~17.8 and k90 ~20 (~= Whitney 2*8+1 = 17, memory L77 /
``quadratic_head_chart``) — measured ONCE, offline. This module lets the v7 run MEASURE that
CONTINUOUSLY at verdict/checkpoint cadence so mod-dim exploitation (rank saturation vs the anneal
octaves, ξ-redundant deletable dims, per-dim export bit-allocation) is observed under the hood
instead of reconstructed after the fact.

SCORE-NEUTRAL BY CONSTRUCTION (CLAUDE.md "'Off' is a tracked queue"): every function here is pure
numpy — it READS array snapshots and returns new arrays/scalars, NEVER mutates its inputs, and NEVER
touches ``np.random`` (the seeded training stream is untouched). Emitting a row therefore cannot
change a trained weight, an archive byte, d_seg, or d_pose: byte-identity is preserved. The trainer
wraps every call in a fail-open ``try/except`` so telemetry can never break a run.

Two cadences:
  * VERDICT cadence (cheap: SVD of a (2P, mod_dim) table is microseconds) — the spectral pack,
    per-dim variance, per-dim FiLM consumption, per-dim ξ correlation, latent↔ξ CCA, k90 truncate
    bytes estimate. Assembled by :func:`mod_dim_dynamics_row`.
  * CHECKPOINT cadence (heavier: per-dim zero-ablation d_seg re-render on a K-pair sample) —
    :func:`per_dim_dseg_ablation` + :func:`mod_dim_ablation_row`. The heavy render/scorer forward is
    supplied by the trainer as a callable; this module owns only the (testable) ablation loop + row.

EXPLOITATION (the "and exploitation" half): per-dim utilization × per-dim scorer sensitivity is the
Catalog #157 waterfill applied at mod-dim granularity — a per-dim export bit-allocation that feeds
the Catalog #336 master-gradient bit-allocator. :func:`per_dim_bit_allocation_hint` names it; the
k90 truncate-bytes estimate is the free-rate lever the autopsy suggested (deferral D18).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "centered_singular_values",
    "effective_rank",
    "k_energy_cutoff",
    "spectral_entropy",
    "top_k_energy_fracs",
    "per_dim_variance",
    "per_dim_film_consumption",
    "code_to_per_pair",
    "per_dim_pose_r2",
    "latent_pose_cca",
    "truncate_bytes_estimate",
    "per_dim_bit_allocation_hint",
    "per_dim_dseg_ablation",
    "mod_dim_dynamics_row",
    "mod_dim_ablation_row",
]


# ── spectral primitives ──────────────────────────────────────────────────────────────────────────
def centered_singular_values(code: np.ndarray) -> np.ndarray:
    """Descending singular values of the column-centered code matrix ``(S, D)`` (S rows = sampled
    (pair, frame) codes, D = mod_dim). Length ``min(S, D)``. READ-ONLY: operates on a centered copy;
    ``code`` is never mutated. The squared singular values are the eigenvalues of the (unscaled)
    covariance — the whole spectral pack derives from them."""
    M = np.asarray(code, np.float64)
    if M.ndim != 2 or M.shape[0] < 1:
        raise ValueError(f"code must be (S, D) with S>=1; got shape {M.shape}")
    Mc = M - M.mean(axis=0, keepdims=True)
    # (macOS-Accelerate BLAS raises a spurious FPE flag on matmul/svd; the values are correct —
    # silence it consistently with lever_b_levelset_generator's np.errstate discipline.)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return np.linalg.svd(Mc, compute_uv=False)


def effective_rank(svals: np.ndarray) -> float:
    """Participation ratio (effective rank) from singular values: ``PR = (Σλ)² / Σλ²`` with
    ``λ = s²`` the covariance eigenvalues. PR ∈ [1, min(S, D)]: 1 == rank-1 collapse, higher == energy
    spread across more independent axes. Algebraically identical to
    ``lever_b_levelset_generator.code_spectrum_participation_ratio`` (cross-checked in tests)."""
    lam = np.asarray(svals, np.float64) ** 2
    tr = float(lam.sum())
    fro2 = float((lam * lam).sum())
    return (tr * tr) / (fro2 + 1e-12)


def k_energy_cutoff(svals: np.ndarray, frac: float) -> int:
    """Smallest ``k`` such that the top-k components hold ``>= frac`` of the total spectral energy
    ``Σ s²`` (e.g. k90 at frac=0.90, k99 at 0.99). Returns 0 for an all-zero spectrum."""
    lam = np.asarray(svals, np.float64) ** 2
    total = float(lam.sum())
    if total <= 0.0:
        return 0
    cum = np.cumsum(lam) / total
    return int(np.searchsorted(cum, float(frac), side="left") + 1)


def spectral_entropy(svals: np.ndarray) -> dict[str, float]:
    """Shannon entropy of the normalized energy spectrum ``p_i = s_i² / Σ s²``. Returns ``nats`` (the
    raw entropy) and ``normalized`` (``H / ln(n)`` ∈ [0, 1], 1 == perfectly uniform / all dims equally
    live, 0 == all energy in one direction). Zero-energy dims contribute 0 (0·ln0 := 0)."""
    lam = np.asarray(svals, np.float64) ** 2
    total = float(lam.sum())
    n = int(lam.size)
    if total <= 0.0 or n == 0:
        return {"nats": 0.0, "normalized": 0.0}
    p = lam / total
    nz = p[p > 0.0]
    h = float(-(nz * np.log(nz)).sum())
    return {"nats": h, "normalized": (h / float(np.log(n))) if n > 1 else 0.0}


def top_k_energy_fracs(svals: np.ndarray, k: int = 8) -> list[float]:
    """Normalized energy fraction ``s_i² / Σ s²`` of the top-``k`` components (descending). Length
    ``min(k, len(svals))``; empty spectrum -> empty list."""
    lam = np.asarray(svals, np.float64) ** 2
    total = float(lam.sum())
    if total <= 0.0:
        return []
    kk = int(min(max(k, 0), lam.size))
    return [float(x) for x in (lam[:kk] / total)]


# ── per-dim primitives ───────────────────────────────────────────────────────────────────────────
def per_dim_variance(code: np.ndarray) -> np.ndarray:
    """Per-dimension (per-column) variance of the code table ``(S, D)`` -> ``(D,)``. Which dims VARY
    across pairs (population variance, ddof=0). READ-ONLY."""
    M = np.asarray(code, np.float64)
    if M.ndim != 2:
        raise ValueError(f"code must be 2-D (S, D); got shape {M.shape}")
    return M.var(axis=0)


def per_dim_film_consumption(film_weights: "np.ndarray | list", mod_dim: int) -> np.ndarray:
    """Per-latent-dim FiLM CONSUMPTION: the L2 norm of the FiLM layer's INPUT-weight column for each
    latent dim -> ``(mod_dim,)``. This is which dims the network actually READS (downstream weight
    energy), distinct from which dims merely VARY. A FiLM ``nn.Linear(mod_dim, ...)`` has weight
    ``(out, mod_dim)``; the column ``W[:, j]`` is how latent dim ``j`` drives the modulation.

    ``film_weights`` may be a single ``(out, mod_dim)`` matrix or a list of them (main + per-layer
    FiLM); matrices are combined in quadrature ``sqrt(Σ_k ||W_k[:, j]||²)``. Matrices whose last dim
    != ``mod_dim`` are IGNORED (e.g. a concat-code FiLM with a different input width). READ-ONLY."""
    mats = film_weights if isinstance(film_weights, (list, tuple)) else [film_weights]
    acc = np.zeros(int(mod_dim), np.float64)
    used = 0
    for w in mats:
        if w is None:
            continue
        a = np.asarray(w, np.float64)
        if a.ndim != 2 or a.shape[1] != int(mod_dim):
            continue
        acc += np.linalg.norm(a, axis=0) ** 2
        used += 1
    return np.sqrt(acc) if used else acc


def code_to_per_pair(code_table: np.ndarray) -> np.ndarray:
    """Aggregate the ``(2P, D)`` per-(pair, frame) code table to a ``(P, D)`` per-pair table by
    averaging each pair's two frame codes (rows ``2*pi`` and ``2*pi+1``). Used to align latents with
    the per-pair pose twist ``(P, 6)`` for correlation/CCA. READ-ONLY. An odd row count is truncated
    to the largest even prefix (fail-safe; the code table is always even in practice)."""
    M = np.asarray(code_table, np.float64)
    if M.ndim != 2:
        raise ValueError(f"code_table must be 2-D (2P, D); got shape {M.shape}")
    p2 = M.shape[0] - (M.shape[0] % 2)
    return M[:p2].reshape(p2 // 2, 2, M.shape[1]).mean(axis=1)


def _r2(x: np.ndarray, y: np.ndarray) -> float:
    """Coefficient of determination = squared Pearson correlation between two 1-D vectors. Returns
    0.0 when either vector is constant (undefined correlation -> no explained variance)."""
    xc = x - x.mean()
    yc = y - y.mean()
    dx = float((xc * xc).sum())
    dy = float((yc * yc).sum())
    if dx <= 0.0 or dy <= 0.0:
        return 0.0
    cov = float((xc * yc).sum())
    return float(min(1.0, max(0.0, (cov * cov) / (dx * dy))))


def per_dim_pose_r2(code_per_pair: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Per-dim ξ correlation: r² of each latent dim (columns of the ``(P, D)`` per-pair code) against
    each of the 6 pose-twist components (``(P, 6)``) -> ``(D, 6)`` matrix. A dim whose max-over-c r² is
    high is ξ-REDUNDANT (recoverable from the stored twist -> a deletable-dim candidate). READ-ONLY."""
    X = np.asarray(code_per_pair, np.float64)
    Y = np.asarray(poses, np.float64)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError(f"shape mismatch: code_per_pair {X.shape} vs poses {Y.shape}")
    d, c = X.shape[1], Y.shape[1]
    out = np.zeros((d, c), np.float64)
    for j in range(d):
        for k in range(c):
            out[j, k] = _r2(X[:, j], Y[:, k])
    return out


def latent_pose_cca(code_per_pair: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Canonical correlations between the per-pair latent table ``(P, D)`` and the pose twist
    ``(P, 6)`` — the table-level latent↔ξ redundancy probe (memory L77). Standard method: economy QR of
    the centered blocks, then the singular values of ``Qx^T Qy`` are the cosines of the principal
    angles = the canonical correlations (descending, length ``min(D, 6)``, each in [0, 1]). A high
    leading canonical correlation means a latent subspace is linearly recoverable from ξ. READ-ONLY;
    returns zeros when there are too few pairs to estimate."""
    X = np.asarray(code_per_pair, np.float64)
    Y = np.asarray(poses, np.float64)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError(f"shape mismatch: code_per_pair {X.shape} vs poses {Y.shape}")
    n = X.shape[0]
    m = int(min(X.shape[1], Y.shape[1]))
    if n < 2 or m == 0:
        return np.zeros(m, np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        Qx, _ = np.linalg.qr(Xc)
        Qy, _ = np.linalg.qr(Yc)
        s = np.linalg.svd(Qx.T @ Qy, compute_uv=False)
    return np.clip(s[:m], 0.0, 1.0)


# ── exploitation hooks ───────────────────────────────────────────────────────────────────────────
def truncate_bytes_estimate(code_bytes_full: int, k: int, mod_dim: int) -> int:
    """Projected coded-blob bytes if the latent table were TRUNCATED to ``k`` columns at export
    (linear estimate ``round(code_bytes_full * k / mod_dim)``). ESTIMATE ONLY — no behavior change;
    the real byte delta is measured by the deferred truncate-at-export A/B (deferral D18). ``k >=
    mod_dim`` returns the full bytes; a non-positive mod_dim returns the full bytes (fail-safe)."""
    if int(mod_dim) <= 0:
        return int(code_bytes_full)
    k = int(max(0, min(int(k), int(mod_dim))))
    return int(round(float(code_bytes_full) * k / float(mod_dim)))


def per_dim_bit_allocation_hint(per_dim_util: np.ndarray, per_dim_sensitivity: np.ndarray) -> list[float]:
    """Catalog #157 waterfill at mod-dim granularity: per-dim priority = ``utilization *
    |scorer sensitivity|``, normalized to sum 1 (an export bit-allocation weight per latent dim that
    the Catalog #336 master-gradient bit-allocator can consume). Dims with high utilization AND high
    d_seg sensitivity get more export bits; ξ-redundant / inert dims get fewer (or are truncation
    candidates). READ-ONLY; degenerate (all-zero) input returns a uniform allocation."""
    u = np.abs(np.asarray(per_dim_util, np.float64))
    s = np.abs(np.asarray(per_dim_sensitivity, np.float64))
    if u.shape != s.shape:
        raise ValueError(f"util {u.shape} and sensitivity {s.shape} must match")
    prio = u * s
    total = float(prio.sum())
    if total <= 0.0:
        n = int(prio.size)
        return [1.0 / n] * n if n else []
    return [float(x) for x in (prio / total)]


def per_dim_dseg_ablation(
    code_table: np.ndarray, dims: "list[int]", render_dseg_fn, *, baseline_dseg: float,
    workers: int = 0,
) -> list[float]:
    """Per-dim d_seg ATTRIBUTION by zero-ablation: for each dim ``j`` in ``dims``, zero column ``j`` of
    a COPY of the code table, hand the copy to ``render_dseg_fn`` (the trainer's K-pair render+SegNet
    forward returning mean d_seg), and record ``Δd_seg = d_seg(ablated) - baseline_dseg``. A large
    positive Δ means dim ``j`` CARRIES d_seg (removing it hurts); ~0 means the dim is d_seg-inert.

    SCORE-NEUTRAL: the ablation is on ``code_table.copy()`` — the input table is never mutated, and
    ``render_dseg_fn`` is expected to be a read-only render (the trainer supplies a closure over
    deploy weights it does not modify). ``render_dseg_fn`` failing on a dim records ``nan`` for that
    dim rather than aborting the sweep (NO-FAKE: never a fabricated Δ).

    ``workers`` (#509 burn-down batch 3, 2026-07-15): when >= 2, fan the per-dim calls across a
    ThreadPoolExecutor — this OBSERVABILITY-ONLY probe is the measured verdict-epoch tail burner
    (~443 s of the 630 s n24 ep25 tail = (mod_dim+1) x (K-pair numpy render + CPU SegNet), each
    call independent + read-only). VALUES IDENTICAL by construction: each dim's input (an
    independent copy) and output are exactly the sequential ones; ``Executor.map`` preserves dim
    order; per-dim nan semantics preserved (the try/except rides inside the mapped fn). Same
    idle-core legality as the verdict workers lever (advisory path, never read into training).
    ``workers<=1`` => the incumbent sequential loop, byte-identical."""
    M = np.asarray(code_table, np.float64)
    if M.ndim != 2:
        raise ValueError(f"code_table must be 2-D; got shape {M.shape}")

    def _one(j: int) -> float:
        cc = M.copy()
        cc[:, int(j)] = 0.0
        try:
            return float(render_dseg_fn(cc)) - float(baseline_dseg)
        except Exception:
            return float("nan")

    if int(workers) >= 2 and len(dims) > 1:
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        with ThreadPoolExecutor(max_workers=int(workers)) as ex:
            return list(ex.map(_one, [int(j) for j in dims]))
    return [_one(int(j)) for j in dims]


# ── row assemblers ───────────────────────────────────────────────────────────────────────────────
_ADVISORY = "[macOS-numpy advisory] NON-PROMOTABLE"


def mod_dim_dynamics_row(
    code_table: np.ndarray, poses: "np.ndarray | list", *,
    epoch: int, seg_form: "str | None", tau: "float | None", mod_dim: int,
    code_bytes_full: int, film_weights: "np.ndarray | list | None" = None,
    top_k: int = 8,
) -> dict:
    """Assemble the per-verdict ``{"stage": "mod_dim_dynamics", ...}`` row from a ``(2P, D)`` code
    table + the ``(P, 6)`` pose twist table. Pure (raises on malformed input); the trainer wraps this
    in a fail-open ``try/except`` so a bad snapshot never breaks a run. OBSERVABILITY-ONLY — never read
    back into training / parity / resume (byte-identical when absent)."""
    svals = centered_singular_values(code_table)
    eff = effective_rank(svals)
    k90 = k_energy_cutoff(svals, 0.90)
    k99 = k_energy_cutoff(svals, 0.99)
    ent = spectral_entropy(svals)
    var = per_dim_variance(code_table)
    per_pair = code_to_per_pair(code_table)
    poses_arr = np.asarray([np.asarray(p, np.float64).ravel()[:6] for p in poses], np.float64)
    cca = latent_pose_cca(per_pair, poses_arr)
    r2 = per_dim_pose_r2(per_pair, poses_arr)
    per_dim_xi_max_r2 = r2.max(axis=1) if r2.size else np.zeros(0)
    film_cons = (per_dim_film_consumption(film_weights, mod_dim)
                 if film_weights is not None else None)
    return {
        "stage": "mod_dim_dynamics",
        "epoch": int(epoch),
        "seg_form": (str(seg_form) if seg_form is not None else None),
        "tau": (float(tau) if tau is not None else None),
        "mod_dim": int(mod_dim),
        "spectrum": {
            "effective_rank": round(float(eff), 4),
            "k90": int(k90),
            "k99": int(k99),
            "spectral_entropy_nats": round(float(ent["nats"]), 4),
            "spectral_entropy_norm": round(float(ent["normalized"]), 4),
            "top_energy_fracs": [round(x, 5) for x in top_k_energy_fracs(svals, top_k)],
        },
        "per_dim": {
            "variance": [round(float(x), 6) for x in var],
            "film_consumption": ([round(float(x), 6) for x in film_cons]
                                 if film_cons is not None else None),
            "xi_max_r2": [round(float(x), 4) for x in per_dim_xi_max_r2],
            "xi_r2_by_component": [[round(float(x), 4) for x in row] for row in r2],
        },
        "latent_xi_cca": {
            "canonical_corrs": [round(float(x), 4) for x in cca],
            "mean": round(float(cca.mean()), 4) if cca.size else 0.0,
            "max": round(float(cca.max()), 4) if cca.size else 0.0,
        },
        "k90_truncate_bytes_estimate": truncate_bytes_estimate(code_bytes_full, k90, mod_dim),
        "code_bytes_full": int(code_bytes_full),
        "axis": _ADVISORY,
        "note": "OBSERVABILITY-ONLY (read-only latent-table SVD + per-dim util/consumption/xi-r2 + "
                "CCA; never read into training => byte-identical when absent). k90_truncate_bytes = "
                "the free-rate lever estimate (deferral D18); per_dim feeds #157/#336 waterfill.",
    }


def mod_dim_ablation_row(
    deltas: "list[float]", dims: "list[int]", per_dim_util: "list[float]", *,
    epoch: int, seg_form: "str | None", k_sample: int,
) -> dict:
    """Assemble the CHECKPOINT-cadence ``{"stage": "mod_dim_ablation", ...}`` row: per-dim zero-ablation
    Δd_seg (the direct "which dims carry score" vector) + the per-dim export bit-allocation hint
    (utilization × |Δd_seg| = #157 waterfill -> #336 consumer). ``per_dim_util`` is the per-dim FiLM
    consumption (or variance) restricted to ``dims``; when its length matches ``deltas`` the hint is
    included."""
    hint = None
    if len(per_dim_util) == len(deltas) and deltas:
        finite = np.nan_to_num(np.asarray(deltas, np.float64), nan=0.0)
        hint = per_dim_bit_allocation_hint(np.asarray(per_dim_util, np.float64), finite)
    return {
        "stage": "mod_dim_ablation",
        "epoch": int(epoch),
        "seg_form": (str(seg_form) if seg_form is not None else None),
        "k_sample": int(k_sample),
        "dims": [int(x) for x in dims],
        "delta_d_seg": [round(float(x), 6) if np.isfinite(x) else None for x in deltas],
        "bit_allocation_hint": ([round(float(x), 5) for x in hint] if hint is not None else None),
        "axis": _ADVISORY,
        "note": "OBSERVABILITY-ONLY zero-ablation d_seg attribution on a K-pair sample (read-only, "
                "on a code COPY; never read into training). bit_allocation_hint = util*|Δd_seg| "
                "(#157 waterfill at mod-dim granularity -> #336 master-gradient bit-allocator).",
    }
