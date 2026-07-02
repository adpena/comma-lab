"""1-D Gibbs + R-survival spectral A/B for INR activations (torch, canonical activation_family).

Deep-math mechanism measurement (NOT a paradigm verdict): fit each activation's tiny INR to a
1-D STEP (the codim-1 argmax-boundary prototype), then measure:
  (1) GIBBS overshoot in the flat plateaus (the L-infinity error that flips shallow-margin pixels)
  (2) R-SURVIVAL: after a low-pass (the contest R = bicubic^ -> uint8 -> bilinear v is a low-pass),
      the argmax-flip fraction vs the true step. Sinusoids ring -> alias -> flip; steps survive.
$0 CPU. torch CPU only (MPS never). Uses the REAL activation_family implementations + LearnableStepBasis.
"""
from __future__ import annotations
import json, math, sys
import numpy as np
import torch
sys.path.insert(0, "src")
from tac.substrates.siren.activation_family import (
    apply_activation_family, LearnableStepBasis, FourierKANActivation,
)
torch.manual_seed(0); np.random.seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)

N = 2048
x = torch.linspace(-1, 1, N).unsqueeze(1)                 # (N,1)
c = 0.13                                                  # edge location (off-grid, generic)
y = (x[:, 0] >= c).float()                                # step target in [0,1]
true_cls = (y >= 0.5).long()

# Fourier-feature front-end (deterministic, matches the witness spirit: sin/cos of B@x)
NF = 32
Bmat = torch.from_numpy((np.random.randn(1, NF) * 6.0).astype(np.float32))
def feats(xx):
    proj = xx @ Bmat
    return torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)  # (N, 2NF)
FEAT = feats(x); IN = FEAT.shape[1]
HID = 64

def lowpass_argmax_flip(phi_np):
    """R-survival proxy: box-decimate x4 (down) then linear up (a low-pass), threshold at 0.5,
    compare argmax(class) to the true step. Returns flip fraction (lower=better R-survival)."""
    v = phi_np.copy()
    # downsample by 4 (avg) then upsample linear back to N  == low-pass round trip
    d = v[: (N // 4) * 4].reshape(-1, 4).mean(1)
    up = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, d.size), d)
    cls = (up >= 0.5).astype(np.int64)
    return float((cls != true_cls.numpy()).mean())

def make_net(kind):
    torch.manual_seed(0)
    W1 = torch.nn.Linear(IN, HID); W2 = torch.nn.Linear(HID, HID); W3 = torch.nn.Linear(HID, 1)
    learn = None
    if kind == "step_basis":
        learn = LearnableStepBasis(num_steps=4, init_gain=1.0)
    if kind == "fkan":
        learn = FourierKANActivation(num_harmonics=5, omega=1.0)
    params = list(W1.parameters()) + list(W2.parameters()) + list(W3.parameters())
    if learn is not None: params += list(learn.parameters())
    def act(u, layer):
        if kind == "step_basis" or kind == "fkan":
            return learn(u)
        if kind == "hosc4":  return apply_activation_family(u, activation_family="hosc", omega=1.0, wire_scale=10.0, hosc_beta=4.0)
        if kind == "hosc8":  return apply_activation_family(u, activation_family="hosc", omega=1.0, wire_scale=10.0, hosc_beta=8.0)
        if kind == "siren":  return apply_activation_family(u, activation_family="siren", omega=30.0, wire_scale=10.0)
        if kind == "finer":  return apply_activation_family(u, activation_family="finer", omega=30.0, wire_scale=10.0)
        if kind == "wire":   return apply_activation_family(u, activation_family="wire", omega=30.0, wire_scale=10.0)
        if kind == "gauss":  return apply_activation_family(u, activation_family="gauss", omega=1.0, wire_scale=3.0)
        if kind == "relu":   return torch.relu(u)
        raise ValueError(kind)
    def fwd():
        h = act(W1(FEAT), 0); h = act(W2(h), 1); return W3(h)[:, 0]
    return fwd, params

def run(kind, steps=800):
    fwd, params = make_net(kind)
    opt = torch.optim.Adam(params, lr=3e-3)
    for _ in range(steps):
        opt.zero_grad(); phi = fwd(); loss = ((phi - y) ** 2).mean(); loss.backward(); opt.step()
    with torch.no_grad():
        phi = fwd().numpy()
    # flat-region L-inf overshoot (exclude a +/-2px window around the edge)
    ei = int((c + 1) / 2 * N); win = 24
    left = phi[: max(ei - win, 1)]; right = phi[min(ei + win, N - 1):]
    overshoot = float(max(np.abs(left - 0.0).max() if left.size else 0.0,
                          np.abs(right - 1.0).max() if right.size else 0.0))
    mse = float(((phi - y.numpy()) ** 2).mean())
    flip = lowpass_argmax_flip(phi)
    # direct argmax-flip (no lowpass) as fit reference
    fit_flip = float(((phi >= 0.5).astype(np.int64) != true_cls.numpy()).mean())
    return {"act": kind, "mse": round(mse, 5), "gibbs_overshoot": round(overshoot, 4),
            "fit_flip": round(fit_flip, 4), "R_survival_flip": round(flip, 4)}

KINDS = ["hosc4", "hosc8", "step_basis", "fkan", "siren", "finer", "wire", "gauss", "relu"]
rows = [run(k) for k in KINDS]
for r in rows: print(json.dumps(r), flush=True)
with open("/private/tmp/claude-501/-Users-adpena-Projects-pact/89ff112f-013d-43b5-b949-2a6d43b650c3/scratchpad/spectral_rows.json", "w") as f:
    json.dump(rows, f, indent=2)
print("DONE", flush=True)
