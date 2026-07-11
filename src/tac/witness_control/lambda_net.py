# SPDX-License-Identifier: MIT
"""The learned Pontryagin adjoint — task #426: the λ-network of the Amortized-Operator
Pontryagin Loop (cluster memo ``.omx/research/amortized_operator_pontryagin_loop_cluster_20260711.md``).

THE CLUSTER (control roles; each member an amortized solve of one leg of the loop):
  * #211 FORWARD operator  — clip → witness-INR params (encoder side; corpus-gated).
  * #426 ADJOINT operator  — THIS MODULE: campaign-state → costate λ = ∂S/∂x as a FIELD
    over the design surface (levers × values), the marginal-ΔS shadow-price organ.
  * #247 CONTROLLER        — consumes λ, u* = argmin H(x,u,λ) (``control_alphabet``),
    ranks duty-to-measure, spawns bounded reasoning agents.
  * #396 TERMINAL solver   — gradient-free (EKI/MC) exact-argmax finisher on the low-dim
    gauge constants (the one member that directly touches the exact score).

WHAT IS LEARNED (the honest small-data adaptation of arXiv 2606.16303's Adjoint Neural
Regulator). The ANR recipe: a net maps state → a HORIZON of costates, trained
SELF-SUPERVISED by backpropagating the ORIGINAL objective through the rollout (no solved-λ
labels); control is recovered closed-form per step from λ; feasibility lives in an act-time
projection OUTSIDE λ-learning (= our containment governor). Mapped to campaign cadence:

    state x_t   = per-class d_seg (5) + log-bytes + context      (verdict rows, n600 advisory)
    control ū_t = measured per-lever loss-term DIRECTION SHARES  (amber clip regime: weights
                  steer direction only — harvest §3f), dense per-epoch rows
    field  Λ_ψ(x, φ_i) ∈ R^{6}  = per-class response RATE of lever i (features φ_i) at state x
    rollout x̂_{t+1} = x̂_t + Δep_t · [ base_ψ(x̂_t) + Σ_i Λ_ψ(x̂_t, φ_i) · ū_i(t) ]
    loss    = Σ_t  ‖x̂_{t+1} − x_{t+1}‖²_w  + γ‖Λ‖₁          (objective through the rollout)
    λ-readout: marginal-ΔS of lever i at state x = ⟨∂S/∂x (ANALYTIC score law), Λ_ψ(x, φ_i)⟩

The costate CHAIN is therefore hybrid-honest: ∂S/∂x is the EXACT analytic partial of the
score law (``costate_estimator.analytic_costates`` tier ANALYTIC, with per-class weights
FITTED from measured verdict rows); the learned part is ONLY the response operator
∂x/∂u — the piece finite differences cannot supply for never-fired levers. Generalization
to never-fired levers rides the lever FEATURE vector φ (class-targeting / term-targeting /
boundary-vs-bulk / regularizer-vs-data), which is why the flagship architecture is a
DeepONet-style branch(x)⊗trunk(φ) bilinear field and not a per-lever lookup.

IDENTIFIABILITY (stated, never hidden): levers whose measured share is ~constant across
the fit window are mutually CONFOUNDED (collinear with each other and the base drift);
their per-lever attribution is FEATURE-STRUCTURED (PARTIAL tier), not data-identified.
The backtest gates below only score what IS identified: held-out per-class forecast error
(vs the OLS-slope persistence heuristic = ``stage_epoch_costates``'s window extrapolation)
and alignment of the |marginal-ΔS| ranking with the harvest's measured BINDING/INERT
labels. Everything else is SPECULATIVE-UNTIL-BACKTESTED by construction.

CONTAINMENT: read-only sensors (parses run-dir telemetry), no actuation surface, no
process tokens; the act side lives in ``control_alphabet`` behind operator-GO tickets.
Torch is imported lazily (CPU, seeded, deterministic algorithms) — MPS/GPU is NEVER used
here (nothing in this module is a score; every number is [macOS advisory] NON-PROMOTABLE).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ── score law (upstream/evaluate.py) — the ANALYTIC λ half ──
SEG_WEIGHT = 100.0
POSE_WEIGHT_INNER = 10.0
RATE_NUMERATOR = 25.0
RATE_DENOMINATOR = 37_545_489.0

N_CLASSES = 5
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")

# ─────────────────────────────────────────────────────────────────────────────
# SENSORS — the observation operator (read-only; composes the canonical parsers'
# line discipline: JSON payload begins at the first '{' on the line).
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CampaignTrajectory:
    """The measured campaign path: verdict-cadence state + dense per-epoch control."""

    run_dir: str
    verdicts: tuple[dict, ...]           # rows with d_seg, d_seg_by_class, blob_bytes, ...
    loss_terms: tuple[dict, ...]         # dense per-epoch rows with `terms` dict + gnorm
    lever_names: tuple[str, ...]         # union of term names seen (stable sorted)

    @property
    def n_verdicts(self) -> int:
        return len(self.verdicts)


def _parse_jsonl_payloads(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        i = line.find("{")
        if i < 0:
            continue
        try:
            r = json.loads(line[i:])
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(r, dict):
            rows.append(r)
    return rows


def read_trajectory(run_dir: str | Path, log_name: str = "daemon.log") -> CampaignTrajectory:
    """Parse a witness run dir into the campaign trajectory (READ-ONLY)."""
    run_dir = Path(run_dir)
    rows = _parse_jsonl_payloads(run_dir / log_name)
    verdicts = [r for r in rows if r.get("stage") == "verdict"
                and isinstance(r.get("epoch"), (int, float))
                and isinstance(r.get("d_seg"), (int, float))]
    loss_terms = [r for r in rows if r.get("stage") == "loss_terms"
                  and isinstance(r.get("ep"), (int, float))
                  and isinstance(r.get("terms"), dict)]
    verdicts.sort(key=lambda r: float(r["epoch"]))
    loss_terms.sort(key=lambda r: float(r["ep"]))
    names = sorted({k for r in loss_terms for k in r["terms"]})
    return CampaignTrajectory(run_dir=str(run_dir), verdicts=tuple(verdicts),
                              loss_terms=tuple(loss_terms), lever_names=tuple(names))


# ─────────────────────────────────────────────────────────────────────────────
# LEVER FEATURES φ — the trunk coordinates of the design surface. DERIVED from the
# master-action knob table (§4 of cgauge_master_action_and_parametrization_20260711):
# class-targeting · score-term targeting · boundary-vs-bulk · regularizer-vs-data.
# A lever ABSENT from this map gets the honest uniform prior (all-classes, seg-term).
# ─────────────────────────────────────────────────────────────────────────────
#: name -> (class weights over CLASS_NAMES, term one-hot (seg,pose,rate), boundary, regularizer)
LEVER_FEATURE_MAP: dict[str, tuple[tuple[float, ...], tuple[float, ...], float, float]] = {
    "seg":              ((.2, .2, .2, .2, .2), (1, 0, 0), 0.0, 0.0),
    "pose":             ((0, 0, 0, 0, 0),      (0, 1, 0), 0.0, 0.0),
    "eikonal":          ((.2, .2, .2, .2, .2), (1, 0, 0), 1.0, 1.0),
    "length":           ((.1, .4, .1, .3, .1), (1, 0, 0), 1.0, 1.0),   # MCF erases thin/minority
    "eik_steik":        ((.2, .2, .2, .2, .2), (1, 0, 0), 1.0, 1.0),
    "boundary_distance":((.2, .2, .2, .2, .2), (1, 0, 0), 1.0, 0.0),
    "lane_edge":        ((0, 1, 0, 0, 0),      (1, 0, 0), 1.0, 0.0),
    "margin_saliency":  ((.2, .2, .2, .2, .2), (1, 0, 0), 1.0, 0.0),
    "subpix":           ((.2, .2, .2, .2, .2), (1, 0, 0), 1.0, 0.0),
    "chroma_boundary":  ((.1, .5, .1, .2, .1), (1, 0, 0), 1.0, 0.0),   # lane-repair stack
    "margin_satisfice": ((.2, .2, .2, .2, .2), (1, 0, 0), 1.0, 0.0),
    "horizon_margin":   ((.4, .1, .4, .1, 0),  (1, 0, 0), 1.0, 0.0),   # horizon band Road/Undriv
    "temporal_screw":   ((.1, .4, .1, .3, .1), (1, 0, 0), 1.0, 0.0),   # gauge/ξ-transport
    "island_amplify":   ((0, .3, 0, .7, 0),    (1, 0, 0), 0.0, 0.0),   # birth stack: Movable-led
    "area_constraint":  ((0, .3, 0, .7, 0),    (1, 0, 0), 0.0, 0.0),
    "persistence":      ((.05, .35, .05, .5, .05), (1, 0, 0), 0.0, 0.0),
    "rankfloor":        ((0, 0, 0, 0, 0),      (0, 0, 1), 0.0, 1.0),
    "code_spectral":    ((0, 0, 0, 0, 0),      (0, 0, 1), 0.0, 1.0),
    "thin_lane":        ((0, 1, 0, 0, 0),      (1, 0, 0), 1.0, 0.0),
    "margin_field_head":((.2, .2, .2, .2, .2), (1, 0, 0), 1.0, 0.0),
    "code_nuclear":     ((0, 0, 0, 0, 0),      (0, 0, 1), 0.0, 1.0),
    "weight_entropy":   ((0, 0, 0, 0, 0),      (0, 0, 1), 0.0, 1.0),
}
PHI_DIM = N_CLASSES + 3 + 2  # class weights + term one-hot + boundary + regularizer


def lever_features(name: str) -> np.ndarray:
    """φ_i for a lever name (uniform-prior fallback for unknown names — stated, not hidden)."""
    cw, term, boundary, reg = LEVER_FEATURE_MAP.get(
        name, ((.2, .2, .2, .2, .2), (1, 0, 0), 0.0, 0.0))
    return np.asarray([*cw, *term, boundary, reg], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# SCORE-LAW COMPOSITION — fit the GT-class area weights w from measured verdicts
# (d_seg = Σ_c w_c · d_seg_by_class[c]; exact lstsq, DERIVED from measured rows).
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ScoreComposition:
    class_weights: np.ndarray      # (5,) fitted GT-area weights, nonneg
    fit_residual: float            # max |predicted − measured| d_seg over the fit rows

    def d_seg_total(self, d_by_class: np.ndarray) -> float:
        return float(self.class_weights @ np.asarray(d_by_class, dtype=np.float64))

    def grad_s_wrt_state(self) -> np.ndarray:
        """∂S/∂x for x = [d_seg_by_class(5), log_bytes] — the ANALYTIC costate half.

        ∂S/∂d_c = 100·w_c (exact). ∂S/∂log_bytes = 25·bytes/N ≈ rate term itself; we
        return the rate-term coefficient at the normalizer so the readout stays linear
        (advisory ranking use only — magnitudes tiny next to seg terms, honestly so)."""
        g = np.zeros(N_CLASSES + 1, dtype=np.float64)
        g[:N_CLASSES] = SEG_WEIGHT * self.class_weights
        g[N_CLASSES] = RATE_NUMERATOR * 85_000.0 / RATE_DENOMINATOR  # d(rate)/d(log b)=rate(b)
        return g


def fit_score_composition(verdicts: tuple[dict, ...] | list[dict]) -> ScoreComposition:
    rows = [v for v in verdicts if isinstance(v.get("d_seg_by_class"), list)
            and len(v["d_seg_by_class"]) == N_CLASSES]
    if len(rows) < 2:
        raise ValueError(f"need ≥2 verdicts with d_seg_by_class; have {len(rows)}")
    D = np.asarray([v["d_seg_by_class"] for v in rows], dtype=np.float64)
    d = np.asarray([v["d_seg"] for v in rows], dtype=np.float64)
    w, *_ = np.linalg.lstsq(D, d, rcond=None)
    w = np.clip(w, 0.0, None)
    resid = float(np.max(np.abs(D @ w - d)))
    return ScoreComposition(class_weights=w, fit_residual=resid)


# ─────────────────────────────────────────────────────────────────────────────
# INTERVAL DATASET — per verdict-interval (t → t+1): start state, mean control
# shares, dense path summary, per-class Δstate/Δepoch targets.
# ─────────────────────────────────────────────────────────────────────────────
STATE_DIM = N_CLASSES + 1                 # per-class d_seg + log_bytes
#: context features appended to the state for the nets (epoch frac, mean gnorm, ep_loss log)
CTX_DIM = 3


@dataclass(frozen=True)
class Interval:
    ep0: float
    ep1: float
    x0: np.ndarray                 # (STATE_DIM,)
    x1: np.ndarray                 # (STATE_DIM,)
    ctx: np.ndarray                # (CTX_DIM,)
    u_mean: np.ndarray             # (L,) mean DIRECTION SHARES of each lever over the interval
    path: np.ndarray               # (T, L+1) dense per-epoch [shares..., gnorm/10] for the GRU

    @property
    def dep(self) -> float:
        return self.ep1 - self.ep0

    def dxdt(self) -> np.ndarray:
        return (self.x1 - self.x0) / max(self.dep, 1e-9)


def _state_from_verdict(v: dict) -> np.ndarray:
    d = np.asarray(v["d_seg_by_class"], dtype=np.float64)
    b = math.log(max(float(v.get("blob_bytes", 1.0)), 1.0))
    return np.concatenate([d, [b]])


def _shares(terms: dict, names: tuple[str, ...]) -> np.ndarray:
    m = np.asarray([abs(float(terms.get(n, 0.0))) for n in names], dtype=np.float64)
    s = float(m.sum())
    return m / s if s > 0 else m


def build_intervals(traj: CampaignTrajectory) -> list[Interval]:
    """Assemble the measured intervals (the ONLY training data — no fabrication)."""
    out: list[Interval] = []
    names = traj.lever_names
    for a, b in zip(traj.verdicts[:-1], traj.verdicts[1:], strict=False):
        e0, e1 = float(a["epoch"]), float(b["epoch"])
        if not (isinstance(a.get("d_seg_by_class"), list) and isinstance(b.get("d_seg_by_class"), list)):
            continue
        lt = [r for r in traj.loss_terms if e0 <= float(r["ep"]) < e1]
        if not lt:
            continue
        share_rows = np.stack([_shares(r["terms"], names) for r in lt])
        gn = np.asarray([[min(float(r.get("gnorm", 0.0)), 100.0) / 10.0] for r in lt])
        path = np.hstack([share_rows, gn])
        ep_loss = float(a.get("ep_loss", 0.0))
        ctx = np.asarray([e0 / 3000.0,
                          float(np.mean(gn)),
                          math.log(max(ep_loss, 1.0)) / 10.0], dtype=np.float64)
        out.append(Interval(ep0=e0, ep1=e1, x0=_state_from_verdict(a),
                            x1=_state_from_verdict(b), ctx=ctx,
                            u_mean=share_rows.mean(axis=0), path=path))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURES — four candidates for the response field Λ(x, φ_i) → R^{STATE_DIM}
# (per-class response RATE per unit control share) + base drift b(x) → R^{STATE_DIM}.
# All tiny + deterministic; torch lazily imported; benchmarked by the backtest gates.
# ─────────────────────────────────────────────────────────────────────────────
class RidgeSolveAdjoint:
    """Candidate A — the closed-form SOLVE (no SGD): Λ = M φ, base = a + C x̃.

    Linear-in-features response + affine drift, fitted by one ridge solve over the
    interval equations dx/dt = base(x) + Σ_i (M φ_i) u_i. This is the DMDc spirit of
    ``ncde_trajectory`` lifted to the lever-feature design surface, and the incumbent-
    class baseline the learned nets must beat."""

    name = "A_ridge_solve"

    def __init__(self, ridge: float = 1e-2):
        self.ridge = float(ridge)
        self.M: np.ndarray | None = None       # (STATE_DIM, PHI_DIM)
        self.a: np.ndarray | None = None       # (STATE_DIM,)
        self.C: np.ndarray | None = None       # (STATE_DIM, STATE_DIM)

    def fit(self, intervals: list[Interval], phis: np.ndarray, seed: int = 0) -> None:
        rows, ys = [], []
        for iv in intervals:
            # dx/dt = a + C x0 + M (Σ_i u_i φ_i)  — Σ_i u_i φ_i is the share-weighted
            # design-surface occupancy: linear structure makes the solve closed-form.
            occ = phis.T @ iv.u_mean                      # (PHI_DIM,)
            rows.append(np.concatenate([[1.0], iv.x0, occ]))
            ys.append(iv.dxdt())
        Phi = np.stack(rows)                              # (N, 1+STATE_DIM+PHI_DIM)
        Y = np.stack(ys)                                  # (N, STATE_DIM)
        p = Phi.shape[1]
        gram = Phi.T @ Phi
        scale = float(np.mean(np.diag(gram))) or 1.0
        coef = np.linalg.solve(gram + self.ridge * scale * np.eye(p), Phi.T @ Y)  # (p, S)
        self.a = coef[0].copy()
        self.C = coef[1:1 + STATE_DIM].T.copy()
        self.M = coef[1 + STATE_DIM:].T.copy()

    def response(self, x: np.ndarray, ctx: np.ndarray, phi: np.ndarray,
                 path: np.ndarray | None = None) -> np.ndarray:
        assert self.M is not None, "fit first"
        return self.M @ phi

    def base(self, x: np.ndarray, ctx: np.ndarray) -> np.ndarray:
        assert self.a is not None and self.C is not None
        return self.a + self.C @ x


def _torch():
    import torch
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    return torch


class _TorchFieldBase:
    """Shared self-supervised trainer: backprop the rollout objective (ANR recipe).

    loss = Σ_intervals ‖x̂1 − x1‖²_w  (+ multi-step rollout when contiguous)
           + l1 · mean|Λ|  (costate sparsity, per the ANR L1-regularizer)
    No λ labels anywhere. CPU only, seeded, deterministic."""

    name = "torch_base"

    def __init__(self, l1: float = 1e-3, epochs: int = 4000, lr: float = 3e-3):
        self.l1, self.epochs, self.lr = float(l1), int(epochs), float(lr)
        self.net = None
        self._t = None

    # subclasses build self.net with forward(xc, phi, path_stats) -> (STATE_DIM,) response
    def _build(self, t):  # pragma: no cover - abstract
        raise NotImplementedError

    def _path_stats(self, t, path: np.ndarray):
        p = t.tensor(path, dtype=t.float64)
        return t.cat([p.mean(dim=0), p.std(dim=0, unbiased=False)])

    def fit(self, intervals: list[Interval], phis: np.ndarray, seed: int = 0) -> None:
        t = _torch()
        t.manual_seed(seed)
        self._t = t
        self._build(t)
        # target scaling: per-channel weights so Lane (large deltas) does not drown MyCar
        dx = np.stack([iv.dxdt() for iv in intervals])
        w = 1.0 / (np.abs(dx).mean(axis=0) + 1e-6)
        w_t = t.tensor(w / w.mean(), dtype=t.float64)
        phis_t = t.tensor(phis, dtype=t.float64)
        opt = t.optim.Adam(self.net.parameters(), lr=self.lr)
        data = [(t.tensor(np.concatenate([iv.x0, iv.ctx]), dtype=t.float64),
                 t.tensor(iv.u_mean, dtype=t.float64),
                 iv.path,
                 t.tensor(iv.dxdt(), dtype=t.float64)) for iv in intervals]
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = t.tensor(0.0, dtype=t.float64)
            for xc, u, raw_path, y in data:
                # path encoding INSIDE the loop: for parameterized encoders (GRU) the
                # graph must be rebuilt per optimizer step.
                ps = self._path_stats(t, raw_path)
                resp = self._responses(xc, phis_t, ps)            # (L, STATE_DIM)
                base = self._base(xc, ps)
                pred = base + resp.T @ u
                loss = loss + (((pred - y) * w_t) ** 2).sum()
                loss = loss + self.l1 * resp.abs().mean()
            loss.backward()
            opt.step()

    # -- numpy-facing API (mirrors RidgeSolveAdjoint) --
    def response(self, x: np.ndarray, ctx: np.ndarray, phi: np.ndarray,
                 path: np.ndarray | None = None) -> np.ndarray:
        t = self._t
        xc = t.tensor(np.concatenate([x, ctx]), dtype=t.float64)
        ps = self._path_stats(t, path if path is not None else np.zeros((1, 23)))
        with t.no_grad():
            r = self._responses(xc, t.tensor(phi[None, :], dtype=t.float64), ps)[0]
        return r.numpy()

    def base(self, x: np.ndarray, ctx: np.ndarray,
             path: np.ndarray | None = None) -> np.ndarray:
        t = self._t
        xc = t.tensor(np.concatenate([x, ctx]), dtype=t.float64)
        ps = self._path_stats(t, path if path is not None else np.zeros((1, 23)))
        with t.no_grad():
            return self._base(xc, ps).numpy()


class MLPLambdaNet(_TorchFieldBase):
    """Candidate B — MLP field: [x,ctx,φ] → response; [x,ctx] → base. Small (h=8)."""

    name = "B_mlp"

    def _build(self, t):
        h = 8
        self.f_resp = t.nn.Sequential(
            t.nn.Linear(STATE_DIM + CTX_DIM + PHI_DIM, h, dtype=t.float64), t.nn.Tanh(),
            t.nn.Linear(h, STATE_DIM, dtype=t.float64))
        self.f_base = t.nn.Sequential(
            t.nn.Linear(STATE_DIM + CTX_DIM, h, dtype=t.float64), t.nn.Tanh(),
            t.nn.Linear(h, STATE_DIM, dtype=t.float64))
        self.net = t.nn.ModuleList([self.f_resp, self.f_base])

    def _responses(self, xc, phis_t, ps):
        t = self._t
        n = phis_t.shape[0]
        inp = t.cat([xc.expand(n, -1), phis_t], dim=1)
        return self.f_resp(inp)

    def _base(self, xc, ps):
        return self.f_base(xc)


class GRULambdaNet(_TorchFieldBase):
    """Candidate C — sequence encoder over the DENSE per-epoch path (loss-term shares +
    gnorm), fused with φ. Uses the 148-row telemetry the verdict cadence throws away."""

    name = "C_gru_path"

    def __init__(self, l1: float = 1e-3, epochs: int = 3000, lr: float = 3e-3,
                 path_dim: int = 23):
        super().__init__(l1=l1, epochs=epochs, lr=lr)
        self.path_dim = int(path_dim)

    def _build(self, t):
        h = 8
        self.gru = t.nn.GRU(self.path_dim, h, batch_first=True, dtype=t.float64)
        self.f_resp = t.nn.Sequential(
            t.nn.Linear(h + STATE_DIM + CTX_DIM + PHI_DIM, h, dtype=t.float64), t.nn.Tanh(),
            t.nn.Linear(h, STATE_DIM, dtype=t.float64))
        self.f_base = t.nn.Sequential(
            t.nn.Linear(h + STATE_DIM + CTX_DIM, h, dtype=t.float64), t.nn.Tanh(),
            t.nn.Linear(h, STATE_DIM, dtype=t.float64))
        self.net = t.nn.ModuleList([self.gru, self.f_resp, self.f_base])
        self._h_cache: dict[int, object] = {}

    def _path_stats(self, t, path: np.ndarray):
        # override: run the GRU over the dense path; return the final hidden state.
        if path.shape[1] != self.path_dim:  # tolerate width drift (pad/trim honestly)
            fixed = np.zeros((path.shape[0], self.path_dim))
            k = min(self.path_dim, path.shape[1])
            fixed[:, :k] = path[:, :k]
            path = fixed
        p = t.tensor(path[None, :, :], dtype=t.float64)
        _, hn = self.gru(p)
        return hn[0, 0]

    def _responses(self, xc, phis_t, ps):
        t = self._t
        n = phis_t.shape[0]
        ctx = t.cat([ps, xc])
        inp = t.cat([ctx.expand(n, -1), phis_t], dim=1)
        return self.f_resp(inp)

    def _base(self, xc, ps):
        t = self._t
        return self.f_base(t.cat([ps, xc]))


class DeepONetLambdaField(_TorchFieldBase):
    """Candidate D — the operator-learning shape: branch(x,ctx) ⊗ trunk(φ) bilinear field.

    λ as a FIELD over the design surface, queryable at ANY lever-feature point —
    including never-fired levers (the duty-to-measure queue). Branch and trunk each map
    to a p-dim latent; the response for class-channel c is ⟨branch_c(x), trunk(φ)⟩."""

    name = "D_deeponet"

    def _build(self, t):
        p, h = 6, 8
        self.p = p
        self.branch = t.nn.Sequential(
            t.nn.Linear(STATE_DIM + CTX_DIM, h, dtype=t.float64), t.nn.Tanh(),
            t.nn.Linear(h, p * STATE_DIM, dtype=t.float64))
        self.trunk = t.nn.Sequential(
            t.nn.Linear(PHI_DIM, h, dtype=t.float64), t.nn.Tanh(),
            t.nn.Linear(h, p, dtype=t.float64))
        self.f_base = t.nn.Sequential(
            t.nn.Linear(STATE_DIM + CTX_DIM, h, dtype=t.float64), t.nn.Tanh(),
            t.nn.Linear(h, STATE_DIM, dtype=t.float64))
        self.net = t.nn.ModuleList([self.branch, self.trunk, self.f_base])

    def _responses(self, xc, phis_t, ps):
        b = self.branch(xc).reshape(STATE_DIM, self.p)     # (S, p)
        tr = self.trunk(phis_t)                            # (L, p)
        return tr @ b.T                                    # (L, S)

    def _base(self, xc, ps):
        return self.f_base(xc)


class GPCostatePosteriorAdjoint:
    """Candidate T — the CLOSED-FORM GP costate posterior (Warnick 2025, §3.3–3.4).

    Maps the organ's costate to the paper's source-term recovery ``Ly = Q`` with the
    first-order operator ``L = d/dep`` (a 1st-order ODE, a_1=1, a_0=0), so the "source"
    Q(ep) = dx/dep IS the per-class d_seg / log-bytes RATE the harness forecasts. Per
    channel c we place a zero-mean GP prior on the CENTERED state trajectory
    ``x_c(ep) ~ GP(0, k)`` with an RBF kernel, observe the verdict rows as noisy point
    values ``z = H[x] + ε`` (H = point-evaluation; ε ~ N(0, σ_n²)), and read off the
    Bayes action under squared-error loss = the posterior mean of ``L x = x'`` in CLOSED
    FORM via differentiated-kernel linear-Gaussian conditioning:

        K   = σ_f² K_rbf(EP, EP) + σ_n² I                       (m×m)
        k_d = ∂_{u*} k(u*, ep_j) = σ_f²·(-(u*-ep_j)/ℓ²)·e^{-(u*-ep_j)²/2ℓ²}   (m,)
        μ'(u*) = k_d^T K^{-1} (z − z̄)          (Bayes action = posterior mean of x')
        σ'²(u*) = σ_f²/ℓ² − k_d^T K^{-1} k_d    (closed-form posterior variance)

    This is the paper's ``Q̂ = L_* μ_*`` / ``Cov(Q) = L_* Σ_* L_*ᵀ`` with the derivative
    kernel blocks (§3.3). By THEOREM 2 the same μ' is the Bayes action under the 0-1
    small-ball loss (loss choice is free on the Gaussian posterior) — the arm's forecast
    is loss-invariant by construction. Hyperparameters (ℓ, σ_n) are fit by type-II
    marginal likelihood on PAST-ONLY observations at each walk-forward fold (no
    look-ahead); σ_f² = training variance of the channel.

    HONEST SCOPE (stated, not hidden): this arm forecasts the TOTAL forcing Q = dx/dep
    directly from trajectory smoothness — it does NOT decompose Q into per-lever
    responses Λ_i (that is the ridge/prototype arms' job; the paper's separatrix/Theorem-3
    view is where the two-posterior decision boundary would live). So ``response`` is the
    zero field and ``base`` carries the whole forecast; the binding-AUROC gate is
    therefore N/A for this arm (no lever field) and the DISCRIMINATING test is the
    walk-forward forecast MAE vs persistence. numpy-only, deterministic, $0. Every number
    [macOS advisory] NON-PROMOTABLE, score_claim=false."""

    name = "T_gp_costate_posterior"

    def __init__(self, ridge_jitter: float = 1e-10):
        self.jitter = float(ridge_jitter)
        self._eps: np.ndarray | None = None                 # (m,) training epochs
        self._z: np.ndarray | None = None                   # (m, STATE_DIM) states
        self._zbar: np.ndarray | None = None                # (STATE_DIM,) per-channel mean
        self._hyp: list[tuple[float, float, float]] = []    # per-channel (ell, sf2, sn2)
        self._Kinv: list[np.ndarray] = []                   # per-channel K^{-1}

    @staticmethod
    def _obs_from_intervals(intervals: list[Interval]) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct the verdict-row observations (contiguous intervals share endpoints)."""
        eps = [iv.ep0 for iv in intervals] + [intervals[-1].ep1]
        xs = [iv.x0 for iv in intervals] + [intervals[-1].x1]
        # dedup epochs (defensive), keep first
        seen: dict[float, np.ndarray] = {}
        for e, x in zip(eps, xs, strict=False):
            if e not in seen:
                seen[e] = x
        e_arr = np.asarray(sorted(seen), dtype=np.float64)
        x_arr = np.stack([seen[e] for e in e_arr])
        return e_arr, x_arr

    @staticmethod
    def _rbf(ep: np.ndarray, sf2: float, ell: float) -> np.ndarray:
        d = ep[:, None] - ep[None, :]
        return sf2 * np.exp(-(d ** 2) / (2.0 * ell ** 2))

    def _mll(self, ep: np.ndarray, z: np.ndarray, sf2: float, ell: float, sn2: float) -> float:
        n = ep.shape[0]
        K = self._rbf(ep, sf2, ell) + (sn2 + self.jitter) * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -np.inf
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, z))
        logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
        return float(-0.5 * z @ alpha - 0.5 * logdet - 0.5 * n * math.log(2.0 * math.pi))

    def fit(self, intervals: list[Interval], phis: np.ndarray, seed: int = 0) -> None:
        eps, xs = self._obs_from_intervals(intervals)
        self._eps, self._z = eps, xs
        self._zbar = xs.mean(axis=0)
        n = eps.shape[0]
        span = float(eps[-1] - eps[0]) if n > 1 else 1.0
        diffs = np.diff(eps)
        med_sp = float(np.median(diffs)) if diffs.size else 1.0
        # bounded, deterministic hyperparameter grid (type-II ML, past-only → no leakage)
        ell_grid = np.geomspace(max(0.5 * med_sp, 1e-3), max(4.0 * span, med_sp), 7)
        ratio_grid = np.asarray([1e-4, 1e-3, 1e-2, 1e-1, 3e-1])  # σ_n²/σ_f²
        self._hyp, self._Kinv = [], []
        for c in range(xs.shape[1]):
            z = xs[:, c] - self._zbar[c]
            sf2 = float(np.var(z))
            if sf2 <= 0.0 or not np.isfinite(sf2):
                sf2 = 1e-18
            best = (-np.inf, ell_grid[len(ell_grid) // 2], sf2, 1e-2 * sf2)
            for ell in ell_grid:
                for ratio in ratio_grid:
                    sn2 = float(ratio) * sf2
                    mll = self._mll(eps, z, sf2, float(ell), sn2)
                    if mll > best[0]:
                        best = (mll, float(ell), sf2, sn2)
            _, ell, sf2, sn2 = best
            K = self._rbf(eps, sf2, ell) + (sn2 + self.jitter) * np.eye(n)
            self._hyp.append((ell, sf2, sn2))
            self._Kinv.append(np.linalg.inv(K))

    def _deriv_posterior(self, u_star: float, c: int) -> tuple[float, float]:
        """Closed-form (mean', var') of x_c'(u_star) — the paper's Q̂ and its covariance."""
        ell, sf2, _ = self._hyp[c]
        ep = self._eps
        z = self._z[:, c] - self._zbar[c]
        d = u_star - ep
        kexp = np.exp(-(d ** 2) / (2.0 * ell ** 2))
        k_d = sf2 * (-(d) / ell ** 2) * kexp                # cov(x'(u*), x(ep_j))
        Kinv = self._Kinv[c]
        mean = float(k_d @ (Kinv @ z))
        prior_var = sf2 / ell ** 2                          # ∂²k/∂u∂v at zero lag
        var = float(prior_var - k_d @ (Kinv @ k_d))
        return mean, max(var, 0.0)

    def base(self, x: np.ndarray, ctx: np.ndarray,
             path: np.ndarray | None = None) -> np.ndarray:
        """Bayes action = posterior-mean derivative at the query epoch (ctx[0]·3000)."""
        assert self._eps is not None, "fit first"
        u_star = float(ctx[0]) * 3000.0
        out = np.empty(STATE_DIM, dtype=np.float64)
        for c in range(STATE_DIM):
            out[c], _ = self._deriv_posterior(u_star, c)
        return out

    def response(self, x: np.ndarray, ctx: np.ndarray, phi: np.ndarray,
                 path: np.ndarray | None = None) -> np.ndarray:
        # forecast-only arm: the total forcing Q=dx/dep is carried by `base`; no per-lever
        # decomposition (Theorem-3 separatrix view, not built). Honest zero field.
        return np.zeros(STATE_DIM, dtype=np.float64)


ARCHITECTURES = ("A_ridge_solve", "B_mlp", "C_gru_path", "D_deeponet", "E_prototype",
                 "E_prototype_bregman", "F_bsf", "G_ridge_scorerprior",
                 "H_smoothed_argmax", "I_comma10k_regime", "J_adv_boundary",
                 "K_perclass_v8", "L_priormean_comma10k", "M_priormean_advb",
                 "N_aniso_coupled", "O_openpilot_geom", "P_priormean_aniso",
                 "Q_priormean_iso", "R_priormean_c10k_scorelaw",
                 "S_priormean_openpilot", "T_gp_costate_posterior")


def make_model(name: str):
    if name == "A_ridge_solve":
        return RidgeSolveAdjoint()
    if name == "B_mlp":
        return MLPLambdaNet()
    if name == "C_gru_path":
        return GRULambdaNet()
    if name == "D_deeponet":
        return DeepONetLambdaField()
    if name == "E_prototype":
        # interpretable-by-design router (PRISM 2607.00510); lazy import avoids a cycle
        from tac.witness_control.prototype_router import PrototypeRouterLens
        return PrototypeRouterLens()
    if name == "E_prototype_bregman":
        # Nielsen Bregman-geometry arm: binary-KL assignment divergence (rates live in
        # exponential-family geometry); centroid stays the mean (Banerjee right-centroid)
        from tac.witness_control.prototype_router import PrototypeRouterLens
        return PrototypeRouterLens(distance="bregman_kl")
    if name == "F_bsf":
        # Goodfire BSF growth-spine: regime prototypes as multidim block-subspaces
        from tac.witness_control.prototype_router import BlockSubspaceRouterLens
        return BlockSubspaceRouterLens()
    if name == "G_ridge_scorerprior":
        # trajectory-INDEPENDENT scorer-geometry arm (cached margin field; $0)
        from tac.witness_control.scorer_geometry import ScorerPriorRidgeAdjoint
        return ScorerPriorRidgeAdjoint()
    if name == "H_smoothed_argmax":
        # SCORER-MODEL ARM (P0 2026-07-11): smoothed-argmax metric relaxation — the
        # #428 survey's #1 (perturbed-optimizer/Gumbel through the REAL frozen SegNet's
        # cached margins; exact gradient, zero model error; ε = τ = ħ per L75)
        from tac.witness_control.scorer_model_arms import SmoothedArgmaxRidgeAdjoint
        return SmoothedArgmaxRidgeAdjoint()
    if name == "I_comma10k_regime":
        # SCORER-MODEL ARM: trajectory-INDEPENDENT comma10k regime prior (SegNet's
        # real training distribution; rarity ⇒ under-trained ⇒ λ-hot; artifact-gated)
        from tac.witness_control.scorer_model_arms import Comma10kRegimeRidgeAdjoint
        return Comma10kRegimeRidgeAdjoint()
    if name == "J_adv_boundary":
        # SCORER-MODEL ARM: adversarial-boundary (advection-ball minimal-flip) geometry
        # pair-weighted by the fitted Young σ_cc′ (#382); ball-agreement audited
        from tac.witness_control.scorer_model_arms import AdversarialBoundaryRidgeAdjoint
        return AdversarialBoundaryRidgeAdjoint()
    if name == "K_perclass_v8":
        # SCORER-MODEL ARM: per-class λ-heads reconciled through the v8 boundary-pair
        # coupling (measured adjacency ⊙ 1/σ_cc′) — feeds the #430 schedule composer
        from tac.witness_control.scorer_model_arms import PerClassCoupledRidgeAdjoint
        return PerClassCoupledRidgeAdjoint()
    if name == "L_priormean_comma10k":
        # SHRINK-TO-PRIOR cure formulation (measured: φ-rescale is INERT; the prior
        # must be the ridge TARGET, not a feature rescale — see scorer_model_arms)
        from tac.witness_control.scorer_model_arms import Comma10kPriorMeanAdjoint
        return Comma10kPriorMeanAdjoint()
    if name == "M_priormean_advb":
        from tac.witness_control.scorer_model_arms import AdvBoundaryPriorMeanAdjoint
        return AdvBoundaryPriorMeanAdjoint()
    if name == "N_aniso_coupled":
        # ANISOTROPIC-COUPLED PER-CLASS λ (P0 #433): coupling-reweight formulation of
        # the V9·CGauge physics (bulk/boundary Fisher regimes + σ_cc′ pair tension +
        # power-diagram adjacency + along/across-tangent anisotropy). Expected-near-
        # neutral (φ-recalibration absorption, measured); the prior-mean sibling P is
        # the physics-consuming formulation.
        from tac.witness_control.aniso_perclass_lambda import AnisoCoupledRidgeAdjoint
        return AnisoCoupledRidgeAdjoint()
    if name == "O_openpilot_geom":
        # openpilot ISOLATED arm (thread 2), reweight formulation
        from tac.witness_control.aniso_perclass_lambda import OpenpilotGeomRidgeAdjoint
        return OpenpilotGeomRidgeAdjoint()
    if name == "P_priormean_aniso":
        # THE P0 ARM (#433): shrink-to-prior ridge, anisotropic-coupled score-law-
        # pinned prior mean (C_phys ∘ exact smoothed-argmax gradient; 1-dof κ)
        from tac.witness_control.aniso_perclass_lambda import AnisoPriorMeanAdjoint
        return AnisoPriorMeanAdjoint()
    if name == "Q_priormean_iso":
        # the ISOTROPIC-INDEPENDENT ablation of P (same formulation, M0 = I)
        from tac.witness_control.aniso_perclass_lambda import IsoPriorMeanAdjoint
        return IsoPriorMeanAdjoint()
    if name == "R_priormean_c10k_scorelaw":
        # the OPEN comma10k-family member: rarity direction, score-law-pinned scale +
        # σ_cc′/Fisher anisotropic coupling (the cure for the measured κ failure)
        from tac.witness_control.aniso_perclass_lambda import (
            C10kScoreLawPriorMeanAdjoint,
        )
        return C10kScoreLawPriorMeanAdjoint()
    if name == "S_priormean_openpilot":
        # openpilot ISOLATED arm, prior-mean (non-absorbable) formulation
        from tac.witness_control.aniso_perclass_lambda import OpenpilotPriorMeanAdjoint
        return OpenpilotPriorMeanAdjoint()
    if name == "T_gp_costate_posterior":
        # CLOSED-FORM GP costate posterior (Warnick 2025 §3.3–3.4): differentiated-RBF
        # derivative posterior mean+cov; Ly=Q with L=d/dep; the solve-don't-train candidate
        return GPCostatePosteriorAdjoint()
    raise ValueError(f"unknown architecture {name!r}; choose from {ARCHITECTURES}")


# ─────────────────────────────────────────────────────────────────────────────
# THE λ READOUT — marginal-ΔS field over the design surface (the adjoint output).
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LambdaField:
    """The evaluated costate field at a state: per-lever marginal-ΔS per unit share·epoch.

    Negative = the lever LOWERS S (improvement). `identified` marks levers whose share
    actually varied in the fit window (data-identified) vs feature-structured (PARTIAL)."""

    epoch: float
    per_lever: dict[str, float]
    identified: dict[str, bool]
    grad_s: tuple[float, ...]              # the ANALYTIC ∂S/∂x used for the readout
    status: str                            # SPECULATIVE-UNTIL-BACKTESTED | BACKTESTED-PASS | BACKTESTED-FAIL

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.per_lever.items(), key=lambda kv: kv[1])


def evaluate_lambda_field(model, traj: CampaignTrajectory, comp: ScoreComposition,
                          intervals: list[Interval], status: str,
                          extra_levers: dict[str, np.ndarray] | None = None) -> LambdaField:
    """marginal-ΔS_i = ⟨∂S/∂x, Λ(x_latest, φ_i)⟩ at the latest verdict state."""
    last = traj.verdicts[-1]
    x = _state_from_verdict(last)
    ctx = intervals[-1].ctx if intervals else np.zeros(CTX_DIM)
    path = intervals[-1].path if intervals else None
    g = comp.grad_s_wrt_state()
    share_var = {}
    if intervals:
        U = np.stack([iv.u_mean for iv in intervals])
        for j, n in enumerate(traj.lever_names):
            share_var[n] = bool(U[:, j].std() > 1e-4 and U[:, j].max() > 1e-3)
    per, ident = {}, {}
    for n in traj.lever_names:
        r = model.response(x, ctx, lever_features(n), path)
        per[n] = float(g @ r)
        ident[n] = share_var.get(n, False)
    for n, phi in (extra_levers or {}).items():
        r = model.response(x, ctx, phi, path)
        per[n] = float(g @ r)
        ident[n] = False
    return LambdaField(epoch=float(last["epoch"]), per_lever=per, identified=ident,
                       grad_s=tuple(g.tolist()), status=status)


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST — the honesty gate ($0, trajectory history only).
# Gate 1 (forecast): leave-one-interval-out per-class Δd_seg forecast vs the OLS-slope
#   persistence heuristic (what ``stage_epoch_costates`` extrapolates today).
# Gate 2 (binding): |marginal-ΔS| ranking vs measured BINDING/INERT labels (harvest §4)
#   compared against the loss-term magnitude heuristic.
# ─────────────────────────────────────────────────────────────────────────────
#: measured binding-vs-inert labels — n205_live_telemetry_harvest_for_v9cgauge_20260711.md §4
HARVEST_BINDING_LABELS: dict[str, int] = {
    "seg": 1, "island_amplify": 1, "persistence": 1, "weight_entropy": 1,
    "area_constraint": 1,     # birth-stack member, measured working (§3a)
    "eikonal": 0, "length": 0,                       # active-small, not binding
    "chroma_boundary": 0, "temporal_screw": 0,       # PENDING (0.0 all rows)
    "pose": 0,                                       # blind by design until ep726
}


def _auroc(scores: dict[str, float], labels: dict[str, int]) -> float | None:
    pairs = [(scores[k], labels[k]) for k in labels if k in scores]
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return None
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


@dataclass(frozen=True)
class BacktestReport:
    architecture: str
    n_intervals: int
    forecast_mae_model: float          # mean |Δd_seg_pred − Δd_seg_meas| over held-out intervals
    forecast_mae_heuristic: float      # same for OLS-slope persistence extrapolation
    forecast_perclass_mae_model: float
    forecast_perclass_mae_heuristic: float
    binding_auroc_model: float | None
    binding_auroc_magnitude_heuristic: float | None
    passed: bool
    notes: tuple[str, ...]
    # walk-forward gate (deployment-faithful: train on PAST intervals only — the LOO gate
    # above trains on future intervals too, a look-ahead advantage the persistence
    # heuristic does not get; adversarial review 2026-07-11 measured that advantage as
    # decisive for A_ridge_solve on the #205 trajectory: LOO-pass but walk-forward-LOSS).
    walkforward_mae_model: float = float("nan")
    walkforward_mae_heuristic: float = float("nan")
    walkforward_perclass_mae_model: float = float("nan")
    walkforward_perclass_mae_heuristic: float = float("nan")
    passed_walkforward: bool = False
    axis_tag: str = "[macOS advisory] NON-PROMOTABLE"
    score_claim: bool = False
    promotable: bool = False

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["notes"] = list(self.notes)
        return d


def _persistence_forecast(intervals: list[Interval], hold: int) -> np.ndarray:
    """The incumbent heuristic: previous interval's measured slope persists (OLS-window
    extrapolation degenerates to this at window=2 on verdict-cadence data)."""
    prev = intervals[hold - 1] if hold > 0 else intervals[hold]
    return prev.dxdt()


def _predict_interval(model, architecture: str, iv: Interval,
                      lever_names: tuple[str, ...]) -> np.ndarray:
    pred = model.base(iv.x0, iv.ctx) if architecture == "A_ridge_solve" else \
        model.base(iv.x0, iv.ctx, iv.path)
    resp = np.stack([model.response(iv.x0, iv.ctx, lever_features(n), iv.path)
                     for n in lever_names])
    return pred + resp.T @ iv.u_mean


def backtest(traj: CampaignTrajectory, architecture: str = "D_deeponet",
             seed: int = 0) -> tuple[BacktestReport, LambdaField]:
    """Leave-one-interval-out backtest + WALK-FORWARD backtest + binding-alignment gate.

    Two forecast gates, both against the persistence heuristic:
      * LOO   — train on all-but-held-out (INCLUDES future intervals: a look-ahead
                advantage; kept for sample efficiency + comparability, labeled as such);
      * WALK-FORWARD — train on PAST intervals only, predict the next (the deployment-
                faithful protocol; the heuristic is walk-forward by construction, so this
                is the apples-to-apples gate). ``BACKTESTED-PASS`` requires BOTH when the
                walk-forward gate is computable (≥4 intervals).
    Returns (report, field) where the field is fit on ALL intervals and stamped."""
    comp = fit_score_composition(traj.verdicts)
    intervals = build_intervals(traj)
    if len(intervals) < 3:
        raise ValueError(f"need ≥3 intervals to backtest; have {len(intervals)}")
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    wcls = comp.class_weights

    errs_m, errs_h, perr_m, perr_h = [], [], [], []
    for hold in range(1, len(intervals)):        # hold-out needs a predecessor for the heuristic
        train = intervals[:hold] + intervals[hold + 1:]
        model = make_model(architecture)
        model.fit(train, phis, seed=seed)
        iv = intervals[hold]
        pred = _predict_interval(model, architecture, iv, traj.lever_names)
        meas = iv.dxdt()
        heur = _persistence_forecast(intervals, hold)
        # scalar d_seg delta = class-weighted; per-class = mean abs over the 5 channels
        errs_m.append(abs(float(wcls @ (pred[:N_CLASSES] - meas[:N_CLASSES]))) * iv.dep)
        errs_h.append(abs(float(wcls @ (heur[:N_CLASSES] - meas[:N_CLASSES]))) * iv.dep)
        perr_m.append(float(np.mean(np.abs(pred[:N_CLASSES] - meas[:N_CLASSES]))) * iv.dep)
        perr_h.append(float(np.mean(np.abs(heur[:N_CLASSES] - meas[:N_CLASSES]))) * iv.dep)

    # walk-forward: past-only training (needs ≥2 train intervals → folds start at 2)
    wf_m, wf_h, wf_pm, wf_ph = [], [], [], []
    for hold in range(2, len(intervals)):
        model = make_model(architecture)
        model.fit(intervals[:hold], phis, seed=seed)
        iv = intervals[hold]
        pred = _predict_interval(model, architecture, iv, traj.lever_names)
        meas = iv.dxdt()
        heur = intervals[hold - 1].dxdt()
        wf_m.append(abs(float(wcls @ (pred[:N_CLASSES] - meas[:N_CLASSES]))) * iv.dep)
        wf_h.append(abs(float(wcls @ (heur[:N_CLASSES] - meas[:N_CLASSES]))) * iv.dep)
        wf_pm.append(float(np.mean(np.abs(pred[:N_CLASSES] - meas[:N_CLASSES]))) * iv.dep)
        wf_ph.append(float(np.mean(np.abs(heur[:N_CLASSES] - meas[:N_CLASSES]))) * iv.dep)

    # binding gate on the full fit. REALIZED binding = |marginal-ΔS field| × mean
    # occupancy share (a pending lever with u=0 has zero REALIZED binding, whatever its
    # field value — the field value itself is the duty-to-measure/what-if ranking).
    # Honesty note: the harvest labels correlate with magnitude by construction (the
    # harvest's own evidence is term magnitude + trend), so this gate is a CONSISTENCY
    # check (necessary, not sufficient); the forecast gate is the discriminating test.
    model = make_model(architecture)
    model.fit(intervals, phis, seed=seed)
    field = evaluate_lambda_field(model, traj, comp, intervals,
                                  status="SPECULATIVE-UNTIL-BACKTESTED")
    U = np.stack([iv.u_mean for iv in intervals])
    mean_share = {n: float(U[:, j].mean()) for j, n in enumerate(traj.lever_names)}
    lam_scores = {k: abs(v) * mean_share.get(k, 0.0) for k, v in field.per_lever.items()}
    mag_scores = dict(mean_share)
    auroc_m = _auroc(lam_scores, HARVEST_BINDING_LABELS)
    auroc_h = _auroc(mag_scores, HARVEST_BINDING_LABELS)

    fm, fh = float(np.mean(errs_m)), float(np.mean(errs_h))
    pm, ph = float(np.mean(perr_m)), float(np.mean(perr_h))
    wfm = float(np.mean(wf_m)) if wf_m else float("nan")
    wfh = float(np.mean(wf_h)) if wf_h else float("nan")
    wfpm = float(np.mean(wf_pm)) if wf_pm else float("nan")
    wfph = float(np.mean(wf_ph)) if wf_ph else float("nan")
    # Binding gate = a CONSISTENCY FLOOR (AUROC ≥ 0.8), not "≥ the magnitude
    # heuristic": the harvest labels mix units — a binding RATE lever (e.g.
    # weight_entropy) has a legitimately tiny |marginal-ΔS| in score units because
    # the rate term is ~0.057/log-byte vs 23–50/d_seg-unit, so demanding the λ-field
    # out-rank a share-magnitude heuristic on those labels is a category error
    # (found + fixed in the 2026-07-11 adversarial review; the heuristic comparison
    # stays REPORTED in the notes + report fields, it just no longer gates).
    passed_loo = (fm <= fh) and (auroc_m is not None) and (auroc_m >= 0.8)
    passed_wf = bool(wf_m) and (wfm <= wfh)
    # BACKTESTED-PASS requires BOTH gates when walk-forward is computable (honesty:
    # the LOO-only pass was measured to be look-ahead-flattered on the #205 trajectory).
    passed = passed_loo and (passed_wf if wf_m else True)
    notes = (
        f"LOO forecast gate: model MAE {fm:.6f} vs persistence-heuristic {fh:.6f} "
        f"(Δd_seg units over held-out intervals; {'BEATS' if fm <= fh else 'LOSES-TO'} heuristic; "
        "LOO trains on future intervals — look-ahead-advantaged, kept for comparability)",
        f"LOO per-class MAE {pm:.6f} vs {ph:.6f}",
        f"WALK-FORWARD gate (deployment-faithful, past-only training, {len(wf_m)} folds): "
        f"model MAE {wfm:.6f} vs heuristic {wfh:.6f} "
        f"({'BEATS' if passed_wf else 'LOSES-TO'} heuristic); per-class {wfpm:.6f} vs {wfph:.6f}",
        f"binding gate: λ-field AUROC {auroc_m} vs magnitude-heuristic {auroc_h} on "
        f"{len(HARVEST_BINDING_LABELS)} harvest-labeled levers",
        "labels source: n205_live_telemetry_harvest_for_v9cgauge_20260711.md §4 (MEASURED)",
        "n=" + str(len(intervals)) + " intervals — small-data regime; every fit ridge/L1-"
        "regularized; per-lever attribution for co-constant levers is FEATURE-STRUCTURED "
        "(PARTIAL tier), not data-identified",
    )
    report = BacktestReport(
        architecture=architecture, n_intervals=len(intervals),
        forecast_mae_model=fm, forecast_mae_heuristic=fh,
        forecast_perclass_mae_model=pm, forecast_perclass_mae_heuristic=ph,
        binding_auroc_model=auroc_m, binding_auroc_magnitude_heuristic=auroc_h,
        passed=passed, notes=notes,
        walkforward_mae_model=wfm, walkforward_mae_heuristic=wfh,
        walkforward_perclass_mae_model=wfpm, walkforward_perclass_mae_heuristic=wfph,
        passed_walkforward=passed_wf)
    field = LambdaField(epoch=field.epoch, per_lever=field.per_lever,
                        identified=field.identified, grad_s=field.grad_s,
                        status="BACKTESTED-PASS" if passed else "BACKTESTED-FAIL")
    return report, field


# ─────────────────────────────────────────────────────────────────────────────
# RASHOMON λ-SET — the honest uncertainty object (Semenova-Rudin: at small n the
# set of near-optimal response models is LARGE; a point-estimate λ is epistemically
# dishonest). Jackknife ensemble of ridge SOLVEs (Ensemble-SINDy-lite): the SET's
# sign/rank agreement per lever is the System-2 escalation trigger — deliberation
# fires when near-optimal models DISAGREE ABOUT THE ACTION, not on a confidence scalar.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RashomonLambdaSet:
    per_lever_values: dict[str, tuple[float, ...]]   # λ across the jackknife ensemble
    sign_agreement: dict[str, float]                 # fraction agreeing with the majority sign
    unstable_levers: tuple[str, ...]                 # decision-relevant + sign-contested
    n_members: int

    def contested(self, min_agreement: float = 0.8) -> list[str]:
        return [k for k, a in self.sign_agreement.items() if a < min_agreement]


def rashomon_lambda_set(traj: CampaignTrajectory, comp: ScoreComposition,
                        intervals: list[Interval], *,
                        decision_quantile: float = 0.75,
                        min_agreement: float = 0.8) -> RashomonLambdaSet:
    """Jackknife-over-intervals ensemble of ridge SOLVEs → per-lever λ set."""
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    members: list[dict[str, float]] = []
    for hold in range(len(intervals)):
        train = intervals[:hold] + intervals[hold + 1:]
        if len(train) < 2:
            continue
        m = RidgeSolveAdjoint()
        m.fit(train, phis)
        f = evaluate_lambda_field(m, traj, comp, intervals,
                                  status="SPECULATIVE-UNTIL-BACKTESTED")
        members.append(f.per_lever)
    if not members:
        return RashomonLambdaSet(per_lever_values={}, sign_agreement={},
                                 unstable_levers=(), n_members=0)
    per: dict[str, tuple[float, ...]] = {}
    agree: dict[str, float] = {}
    for n in traj.lever_names:
        vals = tuple(float(m.get(n, 0.0)) for m in members)
        per[n] = vals
        signs = [1 if v > 0 else -1 if v < 0 else 0 for v in vals]
        nz = [s for s in signs if s != 0]
        if not nz:
            agree[n] = 1.0
            continue
        maj = 1 if sum(nz) >= 0 else -1
        agree[n] = sum(1 for s in nz if s == maj) / len(nz)
    mags = {n: float(np.mean(np.abs(v))) for n, v in per.items()}
    cut = float(np.quantile(list(mags.values()), decision_quantile)) if mags else 0.0
    unstable = tuple(sorted(n for n in per
                            if mags[n] >= cut and agree[n] < min_agreement))
    return RashomonLambdaSet(per_lever_values=per, sign_agreement=agree,
                             unstable_levers=unstable, n_members=len(members))


def benchmark_all(traj: CampaignTrajectory, seed: int = 0) -> dict[str, BacktestReport]:
    """Run the backtest across all four architectures (the tournament)."""
    out: dict[str, BacktestReport] = {}
    for arch in ARCHITECTURES:
        try:
            report, _ = backtest(traj, architecture=arch, seed=seed)
            out[arch] = report
        except Exception as exc:  # honest per-arch failure, not a silent skip
            out[arch] = BacktestReport(
                architecture=arch, n_intervals=0,
                forecast_mae_model=float("nan"), forecast_mae_heuristic=float("nan"),
                forecast_perclass_mae_model=float("nan"),
                forecast_perclass_mae_heuristic=float("nan"),
                binding_auroc_model=None, binding_auroc_magnitude_heuristic=None,
                passed=False, notes=(f"FAILED: {type(exc).__name__}: {exc}",))
    return out
