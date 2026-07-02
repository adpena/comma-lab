"""Activation gradient-health + CE-descent A/B on the REAL level-set witness trunk (MLX).

$0 CPU/local-MLX. Frozen-authority: this measures GRADIENT HEALTH at init (n-independent
property of arch x init x activation x real curvelet feats) + a short CE-descent trajectory
on REAL n96 pair-0 lstars (the per-coordinate SegNet argmax target the CE curriculum stage
optimizes). NOT a score; realized d_seg-through-R is a separate authority. MPS never touched.

Replicates the trainer's inline LevelSetRGBWitness trunk EXACTLY:
  h0 = act(in_proj(feats));  for li: pre = hidden[li](h)*(1+film[li,0]) + film[li,1]; h = act(pre)
  sdf = out_sdf(h)  (P,K)
with a PLUGGABLE activation so we can measure hosc{beta}, siren, finer, wire, gauss, relu, step_basis
under the SAME real front-end + SIREN init. Validated against the real model for hosc.
"""
from __future__ import annotations
import json, math, sys, time
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "experiments")
import mlx.core as mx
import mlx.nn as nn

from tac.boundary_math.lever_b_levelset_generator import (
    CurveletBankConfig, curvelet_directional_B, curvelet_feats,
)
from train_witness_realized_through_R_mlx import _build_render_coords, apply_siren_init
from tac.boundary_math.lever_b_generator import self_orientation_directional_feats  # noqa

mx.random.seed(0); np.random.seed(0)

# ---- arch (from n600_v2 launch design) ----
MOD_DIM, HIDDEN_DIM, N_HIDDEN, K = 26, 120, 4, 5
BANK = CurveletBankConfig(n_scales=4, n_orient0=6, f0=2.0, base=2.0, n_iso=4)
B = curvelet_directional_B(BANK, max_freq=64.0)
COORDS = _build_render_coords(384, 512)                 # (196608, 2)
FEATS_FULL = curvelet_feats(COORDS, B).astype(np.float32)
IN_FEAT = FEATS_FULL.shape[1]

# subsample coords for the health probe (preact dist is well-sampled; CE uses real labels)
NPROBE = 8192
idx = np.random.choice(FEATS_FULL.shape[0], NPROBE, replace=False)
FEATS = mx.array(FEATS_FULL[idx])                       # (NPROBE, in_feat)

# real labels: pair-0 lstars from gt_n96 (the CE-stage target)
GT = np.load("experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
lstar0 = GT["lstars"][0].reshape(-1)                    # (196608,)
LAB = mx.array(lstar0[idx].astype(np.int32))           # (NPROBE,)

print(json.dumps({"stage": "setup", "in_feat": int(IN_FEAT), "nprobe": NPROBE,
                  "arch": [MOD_DIM, HIDDEN_DIM, N_HIDDEN, K]}), flush=True)


def act_fn(name, beta):
    if name == "hosc":   return lambda u: mx.tanh(beta * mx.sin(1.0 * u))
    if name == "siren":  return lambda u: mx.sin(30.0 * u)
    if name == "finer":  return lambda u: mx.sin(30.0 * (mx.abs(u) + 1.0) * u)
    if name == "wire":   return lambda u: mx.cos(20.0 * u) * mx.exp(-((10.0 * u) ** 2))
    if name == "gauss":  return lambda u: mx.exp(-((10.0 * u) ** 2))
    if name == "relu":   return lambda u: nn.relu(u)
    raise ValueError(name)


def dact_fn(name, beta):
    # analytic derivative for the 1-D gradient-throughput profile
    if name == "hosc":
        return lambda u: beta * np.cos(u) * (1.0 / np.cosh(beta * np.sin(u)) ** 2)
    if name == "siren":  return lambda u: 30.0 * np.cos(30.0 * u)
    if name == "finer":  # d/du sin(w(|u|+1)u) ; approx via w*(2|u|+1)*cos(...)
        return lambda u: 30.0 * (2 * np.abs(u) + 1.0) * np.cos(30.0 * (np.abs(u) + 1.0) * u)
    if name == "wire":   return lambda u: (-20.0 * np.sin(20.0 * u)) * np.exp(-((10.0 * u) ** 2)) + np.cos(20.0 * u) * (-2 * 100.0 * u) * np.exp(-((10.0 * u) ** 2))
    if name == "gauss":  return lambda u: (-2 * 100.0 * u) * np.exp(-((10.0 * u) ** 2))
    if name == "relu":   return lambda u: (u > 0).astype(np.float64)
    raise ValueError(name)


class Trunk(nn.Module):
    """Faithful level-set trunk (pluggable act). film(code=0)=film.bias (nonzero MLX init)."""
    def __init__(self, name, beta):
        super().__init__()
        self.in_proj = nn.Linear(IN_FEAT, HIDDEN_DIM)
        self.film = nn.Linear(MOD_DIM, 2 * HIDDEN_DIM * N_HIDDEN)
        self.hidden = [nn.Linear(HIDDEN_DIM, HIDDEN_DIM) for _ in range(N_HIDDEN)]
        self.out_sdf = nn.Linear(HIDDEN_DIM, K)
        self.code = mx.zeros((MOD_DIM,))
        self._act = act_fn(name, beta)

    def sdf(self, feats):
        h = self._act(self.in_proj(feats))
        film = mx.reshape(self.film(self.code), (N_HIDDEN, 2, HIDDEN_DIM))
        for li, layer in enumerate(self.hidden):
            pre = layer(h) * (1.0 + film[li, 0]) + film[li, 1]
            h = self._act(pre)
        return self.out_sdf(h)                          # (P,K)

    def preacts(self, feats):
        outs = []
        u0 = self.in_proj(feats); outs.append(u0)
        h = self._act(u0)
        film = mx.reshape(self.film(self.code), (N_HIDDEN, 2, HIDDEN_DIM))
        for li, layer in enumerate(self.hidden):
            pre = layer(h) * (1.0 + film[li, 0]) + film[li, 1]
            outs.append(pre); h = self._act(pre)
        return outs


def build(name, beta, siren_init):
    t = Trunk(name, beta); mx.eval(t.parameters())
    if siren_init:
        omega = 1.0 if name == "hosc" else (20.0 if name == "wire" else 30.0)
        apply_siren_init(t, omega=omega); mx.eval(t.parameters())
    return t


def ce_loss(t, feats, lab, temp=1.0):
    phi = t.sdf(feats) / temp
    logZ = mx.logsumexp(phi, axis=-1)
    pick = mx.take_along_axis(phi, lab[:, None], axis=-1)[:, 0]
    return mx.mean(logZ - pick)


def measure(name, beta=4.0, siren_init=True):
    t = build(name, beta, siren_init)
    # --- preactivation distribution + gradient throughput per layer ---
    pre = t.preacts(FEATS); mx.eval(pre)
    dfn = dact_fn(name, beta)
    layer_stats = []
    for li, u in enumerate(pre):
        un = np.asarray(u, dtype=np.float64).reshape(-1)
        g = np.abs(dfn(un))
        layer_stats.append({
            "preact_std": float(un.std()),
            "mean_absgrad": float(g.mean()),
            "vanish_frac": float((g < 1e-2).mean()),   # |phi'|<0.01 => vanishing
            "dead_frac_unit": None,
        })
    # per-UNIT dead fraction at layer 1 (first hidden): a unit dead if |phi'| tiny for ~all inputs
    u1 = np.asarray(pre[1], dtype=np.float64)           # (P, HIDDEN)
    g1 = np.abs(dfn(u1.reshape(-1))).reshape(u1.shape)
    dead_units = ((g1 < 1e-2).mean(axis=0) > 0.99).mean()
    for s in layer_stats: s.pop("dead_frac_unit")

    # --- chained backprop health: grad norm reaching layer0 (in_proj) vs out_sdf, + code grad ---
    def loss_params(params):
        t.update(params)
        return ce_loss(t, FEATS, LAB, temp=1.0)
    lv, grads = mx.value_and_grad(loss_params)(t.parameters())
    mx.eval(lv, grads)
    def gnorm(d):
        tot = 0.0
        for k, v in d.items():
            if isinstance(v, dict):
                tot += gnorm(v) ** 2
            elif isinstance(v, list):
                for e in v:
                    if hasattr(e, "shape"):
                        tot += float((np.asarray(e) ** 2).sum())
                    elif isinstance(e, dict):
                        tot += gnorm(e) ** 2
            elif hasattr(v, "shape"):
                tot += float((np.asarray(v) ** 2).sum())
        return math.sqrt(tot)
    g_in = gnorm(grads["in_proj"])
    g_out = gnorm(grads["out_sdf"])
    g_hidden0 = gnorm(grads["hidden"][0]) if isinstance(grads["hidden"], list) else 0.0
    g_film = gnorm(grads["film"])
    # code gradient: grad of loss wrt code (per-pair adaptation signal)
    def loss_code(code):
        t.code = code
        return ce_loss(t, FEATS, LAB, temp=1.0)
    _, gcode = mx.value_and_grad(loss_code)(t.code)
    mx.eval(gcode)
    g_code = float(np.linalg.norm(np.asarray(gcode)))

    # --- short CE descent (200 steps, full-batch AdamW, temp=1.0) : trajectory health ---
    t2 = build(name, beta, siren_init)
    opt = __import__("mlx.optimizers", fromlist=["AdamW"]).AdamW(learning_rate=1e-3, weight_decay=1e-4)
    lg = nn.value_and_grad(t2, lambda: ce_loss(t2, FEATS, LAB, temp=1.0))
    losses = []
    for step in range(200):
        l, g = lg(); opt.update(t2, g); mx.eval(t2.parameters(), opt.state)
        if step % 25 == 0 or step == 199:
            losses.append(round(float(l), 4))
    # argmax accuracy proxy at end (per-coordinate class match; higher = better partition fit)
    phi = t2.sdf(FEATS); mx.eval(phi)
    acc = float((np.asarray(mx.argmax(phi, axis=-1)) == np.asarray(LAB)).mean())

    return {
        "act": name, "beta": beta, "siren_init": siren_init,
        "preact_std_L": [round(s["preact_std"], 3) for s in layer_stats],
        "mean_absgrad_L": [round(s["mean_absgrad"], 4) for s in layer_stats],
        "vanish_frac_L": [round(s["vanish_frac"], 3) for s in layer_stats],
        "dead_units_L1": round(float(dead_units), 3),
        "gnorm_in": round(g_in, 5), "gnorm_hidden0": round(g_hidden0, 5),
        "gnorm_out": round(g_out, 5), "gnorm_film": round(g_film, 5),
        "gnorm_code": round(g_code, 5),
        "chain_ratio_in_over_out": round(g_in / max(g_out, 1e-12), 4),
        "ce_init": losses[0], "ce_traj": losses, "ce_final": losses[-1],
        "argmax_acc_final": round(acc, 4),
    }


CONFIGS = [
    ("hosc", 4.0, True), ("hosc", 8.0, True),
    ("hosc", 1.0, True),                       # anneal-start
    ("hosc", 4.0, False), ("hosc", 8.0, False),  # NO siren-init (the DAG "diverge" regime?)
    ("siren", 4.0, True), ("finer", 4.0, True),
    ("wire", 4.0, True), ("gauss", 4.0, False), ("relu", 4.0, False),
]
rows = []
for name, beta, si in CONFIGS:
    t0 = time.time()
    try:
        r = measure(name, beta, si); r["sec"] = round(time.time() - t0, 1)
        rows.append(r); print(json.dumps(r), flush=True)
    except Exception as e:
        print(json.dumps({"act": name, "beta": beta, "siren_init": si, "ERROR": str(e)}), flush=True)

with open("/private/tmp/claude-501/-Users-adpena-Projects-pact/89ff112f-013d-43b5-b949-2a6d43b650c3/scratchpad/gradhealth_rows.json", "w") as f:
    json.dump(rows, f, indent=2)
print("DONE", len(rows), flush=True)
