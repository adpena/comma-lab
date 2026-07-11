# SPDX-License-Identifier: MIT
"""The INTERPRETABLE-BY-DESIGN prototype router — task #426, the Rudin/Daubechies co-lead
stance made architecture (PRISM arXiv 2607.00510; ProtoPNet "this looks like that" lineage,
Rudin/Chen's own family). Binding directive 2026-07-11: the λ-organ's ROUTER/head is
prototype-based so the whole thing is interpretable-by-design (explanations are CONTRACTS,
not post-hoc retrofits) AND an OBSERVATORY (the max-observability non-negotiable).

WHY prototypes and not a black-box gate. A DeepONet/MoE gate predicts λ well but cannot say
WHY — it violates the apparatus. PRISM's answer: every prediction is a **sparse non-negative
mixture of learned prototypes** anchored to **coherent neighborhoods of training examples**.
Transposed to our campaign: prototypes = recognizable TRAJECTORY REGIMES (island-birth /
Road-binding / lane-erosion / annulus-migration), each anchored to a neighborhood of actual #205
verdict states, and **the mixture weights ARE the explanation** — every λ is traceable to named
regimes + the past states that drove it (the 500× training-data attribution PRISM measures; here
it is EXACT and cheap because the mixture is sparse and the neighborhood is explicit).

Composes with what is already built (NOT a replacement). HISTORY: at n=5 intervals the
ridge SOLVE won the tournament and this lens was the interpretable head with a measured
+8% scalar cost; at n=8 (2026-07-11 adversarial-review re-arbitration) the PROTOTYPE
FAMILY is the measured winner outright (LOO 0.002716 vs ridge 0.003057; walk-forward
0.002513 vs ridge 0.003535, which LOSES to persistence) — interpretability now costs
nothing. Re-arbitrated per record accrual via ``continual_costate``; this docstring
states the mechanism, the organ ledger holds the live verdict.
Its per-prototype local λ-responses are fitted (a SLIM-style linear controller over prototypes —
Rudin's falling-rule discipline: the DECIDE stays a readable linear rule). Daubechies enters as a
**multi-scale cascade**: coarse prototypes (few, broad kernel) carry the bulk regime, fine
prototypes carry the residual correction — the wavelet coarse→fine the co-lead brings.

Prototype SUPPRESSION (PRISM): zeroing a prototype's mixture weight removes its regime-response
traceably, without refitting — controllable + auditable actuation (kill a bad regime-response by
name). Exposed as ``suppress=``.

NO-FAKE: prototype NAMES are DERIVED from each prototype's MEASURED per-class d_seg + part_frac
signature (the CLAUDE.md canonical class order), never asserted; the backtest still earns the
architecture (interpretability is a design constraint the winner must ALSO satisfy — this lens
does not get to skip the forecast gate, it gets reported honestly). CPU/numpy only; every number
[macOS advisory] NON-PROMOTABLE. No actuation surface (containment source-scan clean).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tac.witness_control.lambda_net import (
    CLASS_NAMES,
    Interval,
    N_CLASSES,
    STATE_DIM,
    lever_features,
)

#: default sparse mixture width (top-m prototypes carry a prediction — PRISM sparsity)
DEFAULT_TOPM = 2
#: default coarse/fine prototype counts (Daubechies 2-level cascade)
DEFAULT_N_COARSE = 2
DEFAULT_N_FINE = 3
#: kernel bandwidth as a fraction of the median inter-prototype distance
DEFAULT_BANDWIDTH_FRAC = 1.0


@dataclass(frozen=True)
class Prototype:
    """One interpretable regime prototype anchored to a neighborhood of training states."""

    index: int
    scale: str                      # "coarse" | "fine"
    center: np.ndarray              # (STATE_DIM,) — the regime anchor in state space
    name: str                       # DERIVED from the center's per-class signature
    neighborhood_epochs: tuple[float, ...]   # the past states this prototype speaks for
    local_response: np.ndarray      # (STATE_DIM, PHI_DIM) — the prototype's λ-response operator
    local_base: np.ndarray          # (STATE_DIM,) — its base drift
    basis: np.ndarray | None = None  # (r, STATE_DIM) BSF block chart (None = point prototype)

    def signature(self) -> str:
        c = self.center[:N_CLASSES]
        worst = int(np.argmax(c))
        dim = f" dim={self.basis.shape[0]}" if self.basis is not None else ""
        return f"{self.name} [{self.scale}{dim}] worst-class={CLASS_NAMES[worst]} " \
               f"(d_seg_by_class≈{np.round(c, 4).tolist()})"

    def to_jsonable(self) -> dict:
        """Triality-ledger serialization (the regime node's durable payload)."""
        return {
            "index": self.index, "scale": self.scale, "name": self.name,
            "center": [round(float(v), 8) for v in self.center],
            "neighborhood_epochs": list(self.neighborhood_epochs),
            "block_dim": (int(self.basis.shape[0]) if self.basis is not None else 0),
        }


def _name_from_center(center: np.ndarray, log_bytes_ref: float) -> str:
    """DERIVE a regime name from the MEASURED per-class d_seg signature (canonical order
    0=Road 1=Lane 2=Undrivable 3=Movable 4=MyCar). Never asserted — read off the anchor."""
    c = center[:N_CLASSES]
    order = np.argsort(-c)  # worst class first
    worst, second = int(order[0]), int(order[1])
    # named regimes from the harvest §3 (the measured #205 dynamics)
    if c[3] > 0.05:                       # Movable still large
        return "movable-island-unborn"
    if worst == 1 and c[1] > 0.30:        # Lane flip mass dominant + high
        return "lane-erosion"
    if worst == 0:                        # Road-binding stage
        return "road-binding"
    if worst == 2:
        return "undrivable-annulus"
    return f"mixed-{CLASS_NAMES[worst]}-{CLASS_NAMES[second]}"


@dataclass(frozen=True)
class PrototypeAttribution:
    """The OBSERVATORY row — which prototypes fired, with what weights, at what uncertainty,
    traceable to which trajectory neighborhood. This IS the explanation (a contract)."""

    epoch: float
    fired: tuple[tuple[str, float], ...]     # (prototype signature, mixture weight), sparse
    mixture_entropy: float                   # uncertainty of the routing (nats; 0 = confident)
    lambda_by_prototype: dict[str, float]    # each prototype's λ-contribution (named)
    neighborhoods: dict[str, tuple[float, ...]]   # name → the past epochs it speaks for
    suppressed: tuple[str, ...]              # any prototypes zeroed this call (auditable)
    axis_tag: str = "[macOS advisory] NON-PROMOTABLE"

    def explain(self) -> str:
        """Rudin readback: the routing decision AS its explanation."""
        parts = [f"{sig.split(' [')[0]}={w:.2f}" for sig, w in self.fired]
        return (f"λ@ep{self.epoch:.0f} routed through {' + '.join(parts)} "
                f"(uncertainty {self.mixture_entropy:.2f} nats"
                + (f"; suppressed {list(self.suppressed)}" if self.suppressed else "") + ")")


class PrototypeRouterLens:
    """The interpretable diverse-panel router: λ as a sparse non-negative prototype mixture.

    Fits (name := DERIVE) coarse + fine prototypes from the interval states, each with a local
    affine λ-response over lever features. Prediction = Σ_k w_k(x)·[base_k + Λ_k φ], w sparse
    non-negative (top-m normalized similarity). Mirrors the lambda_net model API
    (``fit`` / ``response`` / ``base``) so it drops into the tournament + panel unchanged."""

    name = "E_prototype"

    def __init__(self, n_coarse: int = DEFAULT_N_COARSE, n_fine: int = DEFAULT_N_FINE,
                 topm: int = DEFAULT_TOPM, bandwidth_frac: float = DEFAULT_BANDWIDTH_FRAC,
                 ridge: float = 1e-2, distance: str = "euclidean"):
        if distance not in ("euclidean", "bregman_kl"):
            raise ValueError(f"distance {distance!r} not in ('euclidean', 'bregman_kl')")
        self.n_coarse, self.n_fine = int(n_coarse), int(n_fine)
        self.topm, self.bandwidth_frac = int(topm), float(bandwidth_frac)
        self.ridge = float(ridge)
        self.distance = distance
        self.prototypes: list[Prototype] = []
        self._bw: float = 1.0
        self._log_bytes_ref: float = math.log(85_000.0)

    # ---- divergence (the assignment geometry; Nielsen "Six short stories" #4/#6:
    # per-class d_seg are RATES living in exponential-family geometry — the natural
    # divergence is the binary-KL Bregman divergence, not squared-Euclid. Banerjee
    # Bregman clustering keeps the MEAN as the (right-)centroid; only the ASSIGNMENT
    # divergence changes. Arbitrated by the backtest, never asserted.) ----
    def _div_rows(self, X: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Divergence of each row of X to center c (squared-Euclid or Bregman-KL)."""
        if self.distance == "euclidean":
            return np.sum((X - c[None, :]) ** 2, axis=1)
        eps = 1e-6
        p = np.clip(X[:, :N_CLASSES], eps, 1 - eps)
        q = np.clip(c[:N_CLASSES], eps, 1 - eps)[None, :]
        kl = np.sum(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)), axis=1)
        rest = np.sum((X[:, N_CLASSES:] - c[None, N_CLASSES:]) ** 2, axis=1)
        return kl + 0.5 * rest

    def _proto_div(self, x: np.ndarray, p: Prototype) -> float:
        """Divergence of a query state to one prototype (BSF blocks override to
        distance-to-SUBSPACE; point prototypes use the assignment divergence)."""
        if p.basis is not None and p.basis.size:
            r = x - p.center
            along = p.basis @ r
            resid = r - p.basis.T @ along
            # off-chart residual dominates; a weak in-plane term keeps LOCALITY so a
            # chart that happens to span the whole trajectory line cannot claim every
            # state (BSF blocks are regions, not infinite planes).
            return float(resid @ resid) + 0.25 * float(along @ along)
        return float(self._div_rows(x[None, :], p.center)[0])

    def _make_basis(self, members: np.ndarray) -> np.ndarray | None:
        """Point prototypes carry no chart (the BSF subclass fits one)."""
        return None

    # ---- fitting (interpretable-by-construction) ----
    def _kmeans(self, X: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Tiny deterministic Bregman k-means (few points, few clusters; assignment by
        ``_div_rows``, centroid = mean — the Banerjee right-centroid — seeded)."""
        k = min(k, len(X))
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))[:k]
        C = X[idx].copy()
        labels = np.zeros(len(X), dtype=int)
        for _ in range(25):
            d = np.stack([self._div_rows(X, C[j]) for j in range(k)], axis=1)
            labels = np.argmin(d, axis=1)
            newC = np.stack([X[labels == j].mean(axis=0) if np.any(labels == j) else C[j]
                             for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        return C, labels

    def _fit_local(self, intervals: list[Interval], phis: np.ndarray,
                   weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Weighted ridge for one prototype's (base, response) over its neighborhood."""
        rows, ys, ws = [], [], []
        for iv, w in zip(intervals, weights, strict=True):
            if w <= 1e-6:
                continue
            occ = phis.T @ iv.u_mean
            rows.append(np.concatenate([[1.0], occ]))
            ys.append(iv.dxdt())
            ws.append(w)
        if not rows:                                   # degenerate prototype → zero response
            return np.zeros(STATE_DIM), np.zeros((STATE_DIM, phis.shape[0]))
        Phi = np.stack(rows); Y = np.stack(ys); W = np.diag(ws)
        p = Phi.shape[1]
        # np.errstate: Apple Accelerate BLAS raises SPURIOUS fp-status flags on these
        # small matmuls (diagnosed 2026-07-11: inputs/outputs verified finite — ws≈0.6,
        # gram∈[0,5.2]; instrumented reproduction in the adversarial-review round 2).
        # Outputs are asserted finite below — a real failure still fails loudly.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            gram = Phi.T @ W @ Phi
            scale = float(np.mean(np.diag(gram))) or 1.0
            coef = np.linalg.solve(gram + self.ridge * scale * np.eye(p), Phi.T @ W @ Y)
        if not np.isfinite(coef).all():                # fail-loud: the guard the errstate owes
            raise FloatingPointError("prototype local ridge produced non-finite coefficients")
        base = coef[0].copy()
        resp = coef[1:].T.copy()                       # (STATE_DIM, PHI_DIM)
        return base, resp

    def fit(self, intervals: list[Interval], phis: np.ndarray, seed: int = 0) -> None:
        X = np.stack([iv.x0 for iv in intervals])      # (N, STATE_DIM)
        eps = np.asarray([iv.ep0 for iv in intervals])
        # coarse + fine prototype centers (Daubechies cascade: coarse regime, fine correction)
        protos: list[Prototype] = []
        for scale, k in (("coarse", self.n_coarse), ("fine", self.n_fine)):
            C, labels = self._kmeans(X, k, seed + (0 if scale == "coarse" else 17))
            for j in range(len(C)):
                mask = labels == j
                nbhd = tuple(float(e) for e in eps[mask])
                # soft membership weights for the local fit (kernel in the divergence)
                dv = self._div_rows(X, C[j])
                bw = np.median(dv[dv > 0]) if np.any(dv > 0) else 1.0
                w = np.exp(-dv / (2 * bw + 1e-12))
                base, resp = self._fit_local(intervals, phis, w)
                protos.append(Prototype(
                    index=len(protos), scale=scale, center=C[j].copy(),
                    name=_name_from_center(C[j], self._log_bytes_ref),
                    neighborhood_epochs=nbhd, local_response=resp, local_base=base,
                    basis=self._make_basis(X[mask]) if np.any(mask) else None))
        self.prototypes = protos
        centers = np.stack([p.center for p in protos])
        dd = np.stack([self._div_rows(centers, centers[j]) for j in range(len(protos))])
        self._bw = self.bandwidth_frac * (np.median(dd[dd > 0]) if np.any(dd > 0) else 1.0)

    # ---- sparse non-negative routing (the interpretable gate) ----
    def _mixture(self, x: np.ndarray, suppress: tuple[str, ...] = ()) -> np.ndarray:
        sims = np.asarray([
            0.0 if p.name in suppress else
            math.exp(-self._proto_div(x, p) / (2 * self._bw + 1e-12))
            for p in self.prototypes])
        if self.topm < len(sims):                      # PRISM sparsity: keep top-m
            cut = np.argsort(-sims)[self.topm:]
            sims = sims.copy(); sims[cut] = 0.0
        s = float(sims.sum())
        return sims / s if s > 0 else np.ones_like(sims) / len(sims)

    def response(self, x: np.ndarray, ctx: np.ndarray, phi: np.ndarray,
                 path: np.ndarray | None = None, suppress: tuple[str, ...] = ()) -> np.ndarray:
        w = self._mixture(x, suppress)
        return sum(wk * (p.local_response @ phi) for wk, p in zip(w, self.prototypes, strict=True))

    def base(self, x: np.ndarray, ctx: np.ndarray, path: np.ndarray | None = None,
             suppress: tuple[str, ...] = ()) -> np.ndarray:
        w = self._mixture(x, suppress)
        return sum(wk * p.local_base for wk, p in zip(w, self.prototypes, strict=True))

    # ---- the OBSERVATORY (the explanation is a contract) ----
    def attribute(self, x: np.ndarray, epoch: float, grad_s: np.ndarray,
                  phi_by_lever: dict[str, np.ndarray] | None = None,
                  suppress: tuple[str, ...] = ()) -> PrototypeAttribution:
        w = self._mixture(x, suppress)
        order = np.argsort(-w)
        fired = tuple((self.prototypes[i].signature(), float(w[i]))
                      for i in order if w[i] > 1e-6)
        ent = float(-sum(wi * math.log(wi) for wi in w if wi > 1e-12))
        # per-prototype λ-contribution: ⟨∂S/∂x, w_k Λ_k φ̄⟩ over a representative lever set
        phis = phi_by_lever or {"seg": lever_features("seg"),
                                "island_amplify": lever_features("island_amplify")}
        lam_by: dict[str, float] = {}
        nbhd: dict[str, tuple[float, ...]] = {}
        for i, p in enumerate(self.prototypes):
            if w[i] <= 1e-6:
                continue
            contrib = 0.0
            for phi in phis.values():
                contrib += float(grad_s @ (w[i] * (p.local_response @ phi)))
            lam_by[p.name] = lam_by.get(p.name, 0.0) + contrib / max(len(phis), 1)
            nbhd[p.name] = p.neighborhood_epochs
        return PrototypeAttribution(epoch=epoch, fired=fired, mixture_entropy=ent,
                                    lambda_by_prototype=lam_by, neighborhoods=nbhd,
                                    suppressed=tuple(suppress))

    # ---- FAITHFULNESS AUDIT (Goodfire/Eternis arXiv 2607.08046: verbalized
    # explanations are only partly faithful (ρ=0.22) while activation probes track the
    # real behavioral shift (ρ=0.57). PRISM's mixture is faithful-BY-CONSTRUCTION —
    # the weights ARE the computation — but that claim is itself AUDITED, not asserted:
    # for each fired prototype, compare its STATED λ-contribution against the REALIZED
    # counterfactual (suppress it, re-route, measure the actual λ shift). A divergence
    # means the renormalization is doing work the stated attribution conceals.) ----
    def faithfulness_audit(self, x: np.ndarray, ctx: np.ndarray, grad_s: np.ndarray,
                           phi: np.ndarray, *, rel_tol: float = 0.35) -> dict:
        """Counterfactual check: stated per-prototype λ vs realized suppress-and-remeasure.

        Returns {"rows": [...], "max_rel_gap": float, "faithful": bool}. Advisory."""
        w = self._mixture(x)
        lam_full = float(grad_s @ self.response(x, ctx, phi))
        rows: list[dict] = []
        scale = max(abs(lam_full), 1e-12)
        for i, p in enumerate(self.prototypes):
            if w[i] <= 1e-6:
                continue
            stated = float(grad_s @ (w[i] * (p.local_response @ phi)))
            lam_without = float(grad_s @ self.response(x, ctx, phi, suppress=(p.name,)))
            realized = lam_full - lam_without      # the prototype's ACTUAL causal share
            gap = abs(stated - realized)
            rows.append({"prototype": p.name, "weight": float(w[i]),
                         "stated_lambda": stated, "realized_counterfactual": realized,
                         "gap": gap, "rel_gap": gap / scale})
        max_rel = max((r["rel_gap"] for r in rows), default=0.0)
        return {"rows": rows, "max_rel_gap": float(max_rel),
                "faithful": bool(max_rel <= rel_tol), "rel_tol": rel_tol,
                "note": "stated=w_k·Λ_k·φ vs realized=suppression counterfactual; the gap "
                        "is the renormalization's hidden work (2607.08046 probe-vs-story)"}


class BlockSubspaceRouterLens(PrototypeRouterLens):
    """Candidate F — the BSF growth-spine head (Goodfire Block-Sparse Featurizers):
    each regime prototype is a MULTIDIMENSIONAL block-subspace (a local affine chart
    fitted by PCA over the regime's neighborhood states), not a 1D point/direction —
    the curved-manifold inductive bias (the level-set boundary manifold is curved +
    multidim; a point prototype flattens it). Routing = distance-to-SUBSPACE with the
    same PRISM top-m block-sparsity (few blocks fire at a time = the Morse-Smale
    sparse-active-strata structure). Interpretable: a block is a named REGION with an
    explicit chart + neighborhood — richer than a point, still a contract.

    Unconstrained-size liberation (operator 2026-07-11): the organ is never rate-scored,
    so block growth costs nothing in bytes; capacity is admitted by BACKTEST only."""

    name = "F_bsf"

    def __init__(self, *args, block_dim: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_dim = int(block_dim)

    def _make_basis(self, members: np.ndarray) -> np.ndarray | None:
        n = len(members)
        r = min(self.block_dim, n - 1, members.shape[1])
        if r < 1:
            return None                       # singleton neighborhood → point prototype
        centered = members - members.mean(axis=0, keepdims=True)
        # deterministic PCA chart via SVD (numpy, seeded-free — SVD is deterministic)
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        keep = min(r, int(np.sum(s > 1e-12)))
        if keep < 1:
            return None
        return vt[:keep].copy()               # (r, STATE_DIM) orthonormal rows
