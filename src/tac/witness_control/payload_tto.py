# SPDX-License-Identifier: MIT
"""payload_tto — deterministic differentiable-GPU test-time optimization of the level-set witness
PAYLOAD (the per-(pair,frame) ``code`` FiLM table), task #350.

WHAT THIS IS (compress-time TOOL, not a trainer lever): a byte-close payload carries the trained
witness's per-pair ``code`` latents. This module optimizes THOSE latents directly against the
DECODED task-space objective (the trainer's own differentiable seg surrogate through render->R on
MLX-GPU with the fixed-order fused-R kernel) while the trunk weights are held FROZEN — so the
decode stays differentiable end-to-end and the payload stays small. It is v1 = LATENTS-ONLY:
trunk weights are never touched (that would need a from-checkpoint fine-tune, a different vehicle).

WHY LATENTS-ONLY IS MEMORY-SAFE AT n600: the ``code`` table row for pair ``pi`` only affects pair
``pi``'s render (FiLM is per-pair). So the optimization DECOMPOSES per-pair — the driver chunks
over pairs, holding only a few pairs' render graph resident at once, bounding RSS well under the
monolithic trainer's ~55 GiB self-orient footprint. This CORE optimizes whatever pair set it is
handed; the driver owns the chunking + the byte-close before/after verdict.

THE OBJECTIVE (stated honestly, NO-FAKE): the IN-LOOP objective is the trainer's DIFFERENTIABLE
seg surrogate (CE/tau/l7 on the rendered partition — the differentiable image of d_seg in
task-space), NOT the argmax d_seg verdict itself (argmax is non-differentiable AND the authority
SegNet is CPU-torch, not MLX — it cannot be in an MLX-GPU loop). The realized d_seg (frozen
CPU-torch SegNet on the byte-close-inflated frames) is MEASURED by the driver before/after; a
surrogate<->verdict gap is itself a finding, which is exactly what the driver's GO/INERT/HARM
bands are designed to catch.

  objective(code) = mean_pi[ loss_closure(model, pi) ] + lambda_bytes * byte_proxy(code)

``byte_proxy`` is a DIFFERENTIABLE surrogate of the counted code-table rate: mean-abs deviation of
``code`` from the trained original (staying near the original keeps the int8+brotli code chunk
close, so Delta-coded-bytes stays small). The REAL Delta-coded-bytes is re-measured via the actual
temporal-delta/int8+brotli coder (:func:`real_code_bytes`, wrapping
``lever_b_levelset_generator.quantize_levelset_blob``) at ACCEPTANCE — the in-loop term is a proxy
by construction and is labelled as such. lambda_bytes defaults to the registered KKT price
6.6586e-7 S/byte.

DETERMINISM + RESUMABILITY (the #348 exploitation): with ``--fused-r-kernel`` ON the whole render
graph is cross-process bit-identical (STAGE 0 verdict), so a seeded, fixed-iteration-order optimizer
is byte-reproducible. The optimizer is a MANUAL AdamW over the ``code`` leaf (m/v/t + code
checkpointed) so a run is resumable-from-disk and two runs are bit-identical — no MLX optimizer
state-tree subtlety, full control.

AUTHORITY: ``[macOS-MLX research-signal]`` for the in-loop objective; the DECODED d_seg verdict the
driver measures is ``[macOS-CPU advisory]``. NEVER a score. Pointer UNMOVED — this is a
compress-time actuator (means), not an exact-eval row.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# The registered KKT byte price (S per archive byte). Sourced from the canonical costate price used
# across the campaign; keep in one place so the driver + tests agree.
LAMBDA_BYTES_KKT = 6.6586e-7


def real_code_bytes(params: dict[str, np.ndarray]) -> int:
    """The MEASURED counted rate for the ``code`` table: int8-symmetric + brotli(quality=11), via the
    canonical ``quantize_levelset_blob`` accounting (the SAME coder the byte-close packs with). This
    is the acceptance-time truth for Delta-coded-bytes; the in-loop penalty is only a proxy for it."""
    from tac.boundary_math.lever_b_levelset_generator import quantize_levelset_blob

    return int(quantize_levelset_blob(params)["code_int8_brotli_bytes"])


@dataclass
class TTOConfig:
    n_iters: int = 200
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    lambda_bytes: float = LAMBDA_BYTES_KKT
    # scale that maps the (dimensionless) mean-abs-deviation byte proxy into approximate byte units,
    # so lambda_bytes (S/byte) is applied on a byte-scaled quantity. Conservative default; the REAL
    # byte accounting is re-measured at acceptance regardless of this scale.
    byte_proxy_scale: float = 1.0e5
    seed: int = 0
    ckpt_every: int = 50  # intra-run resumable checkpoint cadence


@dataclass
class TTOResult:
    n_iters_run: int
    pair_indices: list[int]
    loss_first: float
    loss_last: float
    surrogate_first: float
    surrogate_last: float
    byte_proxy_first: float
    byte_proxy_last: float
    code_sha_before: str
    code_sha_after: str
    telemetry: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _sha_arr(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.float32)).tobytes()).hexdigest()[:16]


def optimize_codes(
    model: Any,
    loss_closure: Callable[[Any, int], Any],
    pair_indices: list[int],
    *,
    cfg: TTOConfig | None = None,
    resume_path: str | Path | None = None,
    fused_r: bool = True,
) -> TTOResult:
    """Optimize ``model.code`` for the given ``pair_indices`` against ``loss_closure`` (the REAL
    differentiable seg surrogate for a pair), trunk FROZEN, deterministically.

    ``loss_closure(model, pi) -> mx scalar`` is supplied by the driver (it closes over the pair's
    real coord_feats + GT targets built by the trainer's setup). This CORE knows nothing about
    self-orient/curvelet/GT — it only moves the ``code`` rows of ``pair_indices`` and calls the
    closure, so it is reusable across any witness whose payload is a per-item code table.

    Resumable: if ``resume_path`` exists, (code, m, v, t) are restored and iteration continues
    bit-identically. Checkpoints are written every ``cfg.ckpt_every`` iters (atomic tmp+rename)."""
    import mlx.core as mx
    from train_witness_realized_through_R_mlx import set_fused_r_kernel

    set_fused_r_kernel(bool(fused_r))
    cfg = cfg or TTOConfig()
    pairs = [int(p) for p in pair_indices]

    code_orig = mx.array(model.code)            # trained payload snapshot (proxy anchor)
    mx.eval(code_orig)
    code = mx.array(model.code)
    n_rows = code.shape[0]
    # row mask: 1.0 on the (2*pi, 2*pi+1) rows we optimize, 0.0 elsewhere -> other pairs FROZEN.
    row_mask_np = np.zeros((n_rows, 1), np.float32)
    for p in pairs:
        row_mask_np[2 * p] = 1.0
        row_mask_np[2 * p + 1] = 1.0
    row_mask = mx.array(row_mask_np)

    m = mx.zeros_like(code)
    v = mx.zeros_like(code)
    t0 = 0
    if resume_path is not None and Path(resume_path).exists():
        z = np.load(resume_path)
        code = mx.array(z["code"])
        m = mx.array(z["m"])
        v = mx.array(z["v"])
        t0 = int(z["t"])
        mx.eval(code, m, v)

    code_sha_before = _sha_arr(code_orig)

    def _objective(code_table):
        model.update({"code": code_table})
        total = mx.array(0.0)
        for pi in pairs:
            total = total + loss_closure(model, pi)
        surrogate = total / float(len(pairs))
        byte_proxy = mx.mean(mx.abs((code_table - code_orig) * row_mask)) * cfg.byte_proxy_scale
        obj = surrogate + cfg.lambda_bytes * byte_proxy
        return obj, (surrogate, byte_proxy)

    grad_fn = mx.value_and_grad(lambda c: _objective(c)[0])

    def _measure(code_table) -> tuple[float, float, float]:
        obj, (surr, bp) = _objective(code_table)
        mx.eval(obj, surr, bp)
        return float(obj), float(surr), float(bp)

    b1, b2, eps, lr, wd = cfg.beta1, cfg.beta2, cfg.eps, cfg.lr, cfg.weight_decay
    obj0, surr0, bp0 = _measure(code)
    telem: list[dict] = [{"it": t0, "obj": obj0, "surrogate": surr0, "byte_proxy": bp0}]
    ckpt = None if resume_path is None else Path(resume_path)

    for it in range(t0, cfg.n_iters):
        obj, g = grad_fn(code)
        g = g * row_mask                              # freeze non-target rows
        t = it + 1
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * (g * g)
        mhat = m / (1.0 - b1 ** t)
        vhat = v / (1.0 - b2 ** t)
        code = code - lr * (mhat / (mx.sqrt(vhat) + eps) + wd * code)
        mx.eval(code, m, v)
        if ckpt is not None and (t % cfg.ckpt_every == 0 or t == cfg.n_iters):
            tmp = ckpt.with_suffix(".tmp.npz")
            np.savez(tmp, code=np.asarray(code, np.float32), m=np.asarray(m, np.float32),
                     v=np.asarray(v, np.float32), t=np.asarray(t))
            tmp.replace(ckpt)

    model.update({"code": code})
    objL, surrL, bpL = _measure(code)
    telem.append({"it": cfg.n_iters, "obj": objL, "surrogate": surrL, "byte_proxy": bpL})
    return TTOResult(
        n_iters_run=cfg.n_iters - t0, pair_indices=pairs,
        loss_first=obj0, loss_last=objL,
        surrogate_first=surr0, surrogate_last=surrL,
        byte_proxy_first=bp0, byte_proxy_last=bpL,
        code_sha_before=code_sha_before, code_sha_after=_sha_arr(code),
        telemetry=telem,
    )


def band_verdict(delta_d_seg: float, delta_bytes: float, *,
                 go_dseg: float = -5e-5, inert_abs: float = 2e-5,
                 bytes_budget: float = 2048.0) -> dict:
    """Pre-registered BINDING band (no post-hoc adjustment), scoped INSTANCE by default per req R:
      GO    = decoded Delta-d_seg <= -5e-5 AND net-positive lambda-accounting (Delta-bytes <= +2KB)
      INERT = |Delta-d_seg| < 2e-5 (a finding about latent-space flatness, scope INSTANCE)
      HARM  = Delta-d_seg > +2e-5 (investigate before concluding, scope INSTANCE)
    Net lambda-accounting uses the registered KKT price: Delta-S ~= 100*Delta-d_seg + lambda*Delta-bytes."""
    net_delta_s = 100.0 * delta_d_seg + LAMBDA_BYTES_KKT * delta_bytes
    if delta_d_seg <= go_dseg and delta_bytes <= bytes_budget and net_delta_s < 0:
        verdict = "GO"
    elif abs(delta_d_seg) < inert_abs:
        verdict = "INERT"
    elif delta_d_seg > inert_abs:
        verdict = "HARM"
    else:
        verdict = "PARTIAL"  # improved d_seg but bytes/net-accounting not net-positive
    return {
        "verdict": verdict, "scope": "INSTANCE",
        "delta_d_seg": float(delta_d_seg), "delta_bytes": float(delta_bytes),
        "net_delta_s_estimate": float(net_delta_s),
        "go_dseg": go_dseg, "inert_abs": inert_abs, "bytes_budget": bytes_budget,
    }


def write_report(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1))
