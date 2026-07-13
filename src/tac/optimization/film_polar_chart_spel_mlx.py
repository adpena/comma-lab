# SPDX-License-Identifier: MIT
"""Resumable FiLM polar-chart MCSD/SPEL finisher for the level-set witness.

The live FiLM map is represented as ``W = Q @ H0``.  ``H0`` is the frozen
positive polar factor at the finisher boundary and ``Q`` has orthonormal
columns.  This preserves the boundary function (up to fp32 factorization
roundoff) while letting the optimizer move only the conditioning isometry.

This module deliberately ships the single-loop MCSD/SPEL approximation, not
Bernstein's exact nested tangent-dual LMO.  Each step:

1. pulls ``grad_W`` back to ``grad_Q = grad_W @ H0.T``;
2. projects the gradient and momentum to ``T_Q St(m, n)``;
3. applies the Muon Newton--Schulz matrix-sign map, projects it back to the
   tangent space, and enforces the spectral-unit ball;
4. takes a step and performs a deterministic thin-QR retraction.

``FilmPolarChartSPELState`` implements the canonical resume-registry protocol.
Its Q, frozen H0, tangent momentum, Q-EMA, source fingerprint, and step counter
therefore ride the trainer's same atomic per-stage resume NPZ.  A legacy run is
unchanged because the controller is constructed and registered only when the
default-OFF DSL lever is armed.

Authority boundary: this is optimizer machinery.  It makes no score claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

METHOD_ID = "film_polar_chart_mcsd_spel_v1"
RESUME_PREFIX = "__fpc_"
RESUME_SCHEMA = 1


def muon_aspect_ratio_scale(shape: tuple[int, int]) -> float:
    """Return MLX Muon's 2-D learning-rate multiplier for ``shape``."""

    rows, cols = (int(shape[0]), int(shape[1]))
    if rows < 1 or cols < 1:
        raise ValueError(f"shape dimensions must be positive, got {shape}")
    return math.sqrt(max(1.0, rows / cols))


def _as_f32_matrix(value: Any, *, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float32)
    if out.ndim != 2 or min(out.shape) < 1:
        raise ValueError(f"{name} must be a non-empty rank-2 matrix, got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    return np.ascontiguousarray(out)


def _matrix_sha256(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def _finite_matmul(left: Any, right: Any, *, name: str) -> np.ndarray:
    """Matrix multiply with explicit finite-result authority.

    Apple's Accelerate-backed NumPy can surface stale floating-point status
    flags as divide/overflow warnings after a finite BLAS matmul.  Scope the
    warning mask to this call, then fail closed on the actual result.
    """

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = np.matmul(left, right)
    if not np.all(np.isfinite(out)):
        raise FloatingPointError(f"{name} produced a non-finite matrix")
    return np.asarray(out)


def polar_chart_numpy(weight: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the thin polar factors ``Q, H0`` of a tall full-rank matrix."""

    w = _as_f32_matrix(weight, name="weight")
    rows, cols = w.shape
    if rows < cols:
        raise ValueError(f"film polar chart requires rows >= cols, got {w.shape}")
    u, singular, vt = np.linalg.svd(w.astype(np.float64), full_matrices=False)
    if float(singular[-1]) <= np.finfo(np.float32).eps * max(float(singular[0]), 1.0):
        raise ValueError("film polar chart requires full column rank")
    q = _finite_matmul(u, vt, name="polar Q").astype(np.float32)
    h0 = _finite_matmul(vt.T * singular[None, :], vt, name="polar H0").astype(np.float32)
    return np.ascontiguousarray(q), np.ascontiguousarray(h0)


def tangent_project_numpy(q: Any, value: Any) -> np.ndarray:
    """Euclidean projection of ``value`` onto ``T_q St(m,n)``."""

    qn = _as_f32_matrix(q, name="q")
    x = _as_f32_matrix(value, name="value")
    if x.shape != qn.shape:
        raise ValueError(f"value shape {x.shape} != q shape {qn.shape}")
    qt_x = _finite_matmul(qn.T, x, name="tangent Q.T@X")
    sym = np.float32(0.5) * (qt_x + qt_x.T)
    return np.asarray(x - _finite_matmul(qn, sym, name="tangent Q@sym"), dtype=np.float32)


def qr_retract_numpy(value: Any) -> np.ndarray:
    """Deterministic thin-QR Stiefel retraction with positive R diagonal."""

    x = _as_f32_matrix(value, name="value")
    q, r = np.linalg.qr(x.astype(np.float64), mode="reduced")
    signs = np.where(np.diag(r) >= 0.0, 1.0, -1.0)
    return np.asarray(q * signs[None, :], dtype=np.float32)


def newton_schulz5_numpy(value: Any, *, steps: int = 5) -> np.ndarray:
    """NumPy-fp32 twin of MLX Muon's quintic matrix-sign approximation."""

    x = _as_f32_matrix(value, name="value")
    if int(steps) < 1:
        raise ValueError("steps must be >= 1")
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = np.asarray(x / (np.linalg.norm(x, ord="fro", keepdims=True) + np.float32(1e-7)), dtype=np.float32)
    a, b, c = np.float32(3.4445), np.float32(-4.7750), np.float32(2.0315)
    for _ in range(int(steps)):
        gram = np.asarray(_finite_matmul(x, x.T, name="NS Gram"), dtype=np.float32)
        gram2 = _finite_matmul(gram, gram, name="NS Gram squared")
        poly = np.asarray(b * gram + c * gram2, dtype=np.float32)
        x = np.asarray(a * x + _finite_matmul(poly, x, name="NS polynomial"), dtype=np.float32)
    return np.ascontiguousarray(x.T if transposed else x)


def _spectral_unit_tangent_numpy(q: np.ndarray, drive: np.ndarray, ns_steps: int) -> np.ndarray:
    direction = tangent_project_numpy(q, newton_schulz5_numpy(drive, steps=ns_steps))
    sigma = float(np.linalg.svd(direction.astype(np.float64), compute_uv=False)[0])
    if not math.isfinite(sigma):
        raise FloatingPointError("non-finite SPEL tangent spectral norm")
    return np.asarray(direction / np.float32(max(1.0, sigma)), dtype=np.float32)


@dataclass(frozen=True)
class NumpySPELStep:
    q: np.ndarray
    h0: np.ndarray
    momentum: np.ndarray
    q_ema: np.ndarray
    weight: np.ndarray
    tangent_residual_fro: float
    orthogonality_residual_fro: float
    direction_spectral_norm: float


def spel_step_numpy(
    q: Any,
    h0: Any,
    momentum: Any,
    q_ema: Any,
    grad_weight: Any,
    *,
    learning_rate: float,
    momentum_beta: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    ema_decay: float = 0.997,
) -> NumpySPELStep:
    """One deterministic NumPy-fp32 MCSD/SPEL polar-chart update."""

    qn = _as_f32_matrix(q, name="q")
    hn = _as_f32_matrix(h0, name="h0")
    mn = _as_f32_matrix(momentum, name="momentum")
    qen = _as_f32_matrix(q_ema, name="q_ema")
    gw = _as_f32_matrix(grad_weight, name="grad_weight")
    if gw.shape != qn.shape or mn.shape != qn.shape or qen.shape != qn.shape:
        raise ValueError("q, momentum, q_ema, and grad_weight must share shape")
    if hn.shape != (qn.shape[1], qn.shape[1]):
        raise ValueError("h0 must be square with q.shape[1] rows")
    lr = float(learning_rate)
    beta = float(momentum_beta)
    decay = float(ema_decay)
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not 0.0 <= beta < 1.0:
        raise ValueError("momentum_beta must be in [0,1)")
    if not 0.0 <= decay < 1.0:
        raise ValueError("ema_decay must be in [0,1)")

    grad_q = np.asarray(_finite_matmul(gw, hn.T, name="FiLM pullback"), dtype=np.float32)
    grad_tan = tangent_project_numpy(qn, grad_q)
    mom = tangent_project_numpy(qn, beta * mn + (1.0 - beta) * grad_tan)
    drive = tangent_project_numpy(qn, (1.0 - beta) * grad_tan + beta * mom) if nesterov else mom
    direction = _spectral_unit_tangent_numpy(qn, drive, int(ns_steps))
    q_next = qr_retract_numpy(qn - np.float32(lr) * direction)
    mom_next = tangent_project_numpy(q_next, mom)
    q_ema_next = np.asarray(decay * qen + (1.0 - decay) * q_next, dtype=np.float32)
    weight_next = np.asarray(_finite_matmul(q_next, hn, name="FiLM fold"), dtype=np.float32)
    qt_m = _finite_matmul(q_next.T, mom_next, name="tangent residual")
    qt_q = _finite_matmul(q_next.T, q_next, name="orthogonality residual")
    tan_res = float(np.linalg.norm(qt_m + qt_m.T, ord="fro"))
    orth_res = float(np.linalg.norm(qt_q - np.eye(q_next.shape[1]), ord="fro"))
    direction_sigma = float(np.linalg.svd(direction.astype(np.float64), compute_uv=False)[0])
    return NumpySPELStep(
        q=q_next,
        h0=hn,
        momentum=mom_next,
        q_ema=q_ema_next,
        weight=weight_next,
        tangent_residual_fro=tan_res,
        orthogonality_residual_fro=orth_res,
        direction_spectral_norm=direction_sigma,
    )


def _tangent_project_mlx(mx: Any, q: Any, value: Any) -> Any:
    qt_x = q.T @ value
    return value - q @ (0.5 * (qt_x + qt_x.T))


def _qr_retract_mlx(mx: Any, value: Any) -> Any:
    q, r = mx.linalg.qr(value)
    signs = mx.where(mx.diag(r) >= 0.0, mx.array(1.0, dtype=q.dtype), mx.array(-1.0, dtype=q.dtype))
    return q * signs[None, :]


def _newton_schulz5_mlx(mx: Any, value: Any, *, steps: int) -> Any:
    x = value
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (mx.linalg.norm(x, keepdims=True) + 1e-7)
    for _ in range(int(steps)):
        gram = x @ x.T
        poly = -4.7750 * gram + 2.0315 * (gram @ gram)
        x = 3.4445 * x + poly @ x
    return x.T if transposed else x


def spel_step_mlx_arrays(
    q: Any,
    h0: Any,
    momentum: Any,
    q_ema: Any,
    grad_weight: Any,
    *,
    learning_rate: float,
    momentum_beta: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    ema_decay: float = 0.997,
) -> tuple[Any, Any, Any, Any]:
    """MLX twin returning ``(q, momentum, q_ema, folded_weight)``."""

    import mlx.core as mx

    # The whole step is pinned to the CPU stream: (a) linalg.qr/svd have no Metal
    # kernels (MLX 0.31.x); (b) the GPU stream drifts vs the numpy-fp32 reference
    # (MEASURED 2026-07-13: GPU max-abs err 4.2e-5 on q / 2.2e-3 on the folded
    # weight vs CPU-stream 8.9e-8 / 2.4e-7 — the known fp-reorder wall, which the
    # Newton-Schulz chain amplifies). The FiLM leaf is tiny, so CPU cost is
    # negligible and the finisher keeps the bit-close determinism it exists for.
    with mx.stream(mx.cpu):
        grad_q = grad_weight @ h0.T
        grad_tan = _tangent_project_mlx(mx, q, grad_q)
        mom = _tangent_project_mlx(mx, q, float(momentum_beta) * momentum + (1.0 - float(momentum_beta)) * grad_tan)
        drive = (_tangent_project_mlx(mx, q, (1.0 - float(momentum_beta)) * grad_tan + float(momentum_beta) * mom)
                 if bool(nesterov) else mom)
        direction = _tangent_project_mlx(mx, q, _newton_schulz5_mlx(mx, drive, steps=int(ns_steps)))
        sigma = mx.linalg.svd(direction, compute_uv=False)[0]
        direction = direction / mx.maximum(mx.array(1.0, dtype=direction.dtype), sigma)
        q_next = _qr_retract_mlx(mx, q - float(learning_rate) * direction)
        mom_next = _tangent_project_mlx(mx, q_next, mom)
        q_ema_next = float(ema_decay) * q_ema + (1.0 - float(ema_decay)) * q_next
        return q_next, mom_next, q_ema_next, q_next @ h0


@dataclass
class FilmPolarChartSPELState:
    """Canonical-resumable MCSD/SPEL state for the single ``film.weight`` leaf."""

    momentum_beta: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    q: Any | None = None
    h0: Any | None = None
    momentum: Any | None = None
    q_ema: Any | None = None
    step: int = 0
    source_weight_sha256: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.momentum_beta) < 1.0:
            raise ValueError("momentum_beta must be in [0,1)")
        if int(self.ns_steps) < 1:
            raise ValueError("ns_steps must be >=1")

    @property
    def initialized(self) -> bool:
        return all(x is not None for x in (self.q, self.h0, self.momentum, self.q_ema))

    def initialize_numpy(self, weight: Any) -> np.ndarray:
        w = _as_f32_matrix(weight, name="weight")
        q, h0 = polar_chart_numpy(w)
        self.q = q
        self.h0 = h0
        self.momentum = np.zeros_like(q)
        self.q_ema = q.copy()
        self.step = 0
        self.source_weight_sha256 = _matrix_sha256(w)
        return np.asarray(_finite_matmul(q, h0, name="initial FiLM fold"), dtype=np.float32)

    def initialize_mlx(self, weight: Any) -> Any:
        import mlx.core as mx

        folded = self.initialize_numpy(np.asarray(weight, dtype=np.float32))
        self.q = mx.array(self.q)
        self.h0 = mx.array(self.h0)
        self.momentum = mx.array(self.momentum)
        self.q_ema = mx.array(self.q_ema)
        mx.eval(self.q, self.h0, self.momentum, self.q_ema)
        return mx.array(folded)

    def warm_start_momentum_numpy(self, weight_momentum: Any) -> np.ndarray:
        """Pull an outgoing AdamW first moment into the Stiefel tangent."""

        if not self.initialized:
            raise RuntimeError("polar-chart state is not initialized")
        raw = _as_f32_matrix(weight_momentum, name="weight_momentum")
        q = _as_f32_matrix(self.q, name="q")
        h0 = _as_f32_matrix(self.h0, name="h0")
        if raw.shape != q.shape:
            raise ValueError(f"weight_momentum shape {raw.shape} != q shape {q.shape}")
        pulled = _finite_matmul(raw, h0.T, name="warm momentum pullback")
        self.momentum = tangent_project_numpy(q, pulled)
        return np.asarray(self.momentum, dtype=np.float32)

    def warm_start_momentum_mlx(self, weight_momentum: Any) -> Any:
        """MLX twin of :meth:`warm_start_momentum_numpy`."""

        self._ensure_mlx()
        import mlx.core as mx

        raw = weight_momentum
        if tuple(raw.shape) != tuple(self.q.shape):
            raise ValueError(
                f"weight_momentum shape {tuple(raw.shape)} != q shape {tuple(self.q.shape)}")
        self.momentum = _tangent_project_mlx(mx, self.q, raw @ self.h0.T)
        mx.eval(self.momentum)
        return self.momentum

    def _ensure_mlx(self) -> None:
        if not self.initialized:
            raise RuntimeError("polar-chart state is not initialized")
        import mlx.core as mx

        if isinstance(self.q, np.ndarray):
            self.q = mx.array(self.q)
            self.h0 = mx.array(self.h0)
            self.momentum = mx.array(self.momentum)
            self.q_ema = mx.array(self.q_ema)
            mx.eval(self.q, self.h0, self.momentum, self.q_ema)

    def step_numpy(self, grad_weight: Any, *, learning_rate: float, ema_decay: float) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("polar-chart state is not initialized")
        out = spel_step_numpy(
            self.q,
            self.h0,
            self.momentum,
            self.q_ema,
            grad_weight,
            learning_rate=learning_rate,
            momentum_beta=self.momentum_beta,
            nesterov=self.nesterov,
            ns_steps=self.ns_steps,
            ema_decay=ema_decay,
        )
        self.q, self.h0, self.momentum, self.q_ema = out.q, out.h0, out.momentum, out.q_ema
        self.step += 1
        return out.weight

    def step_mlx(self, grad_weight: Any, *, learning_rate: float, ema_decay: float) -> Any:
        self._ensure_mlx()
        import mlx.core as mx

        self.q, self.momentum, self.q_ema, weight = spel_step_mlx_arrays(
            self.q,
            self.h0,
            self.momentum,
            self.q_ema,
            grad_weight,
            learning_rate=learning_rate,
            momentum_beta=self.momentum_beta,
            nesterov=self.nesterov,
            ns_steps=self.ns_steps,
            ema_decay=ema_decay,
        )
        self.step += 1
        mx.eval(self.q, self.momentum, self.q_ema, weight)
        return weight

    def deploy_weight_numpy(self) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("polar-chart state is not initialized")
        q_ema = qr_retract_numpy(np.asarray(self.q_ema, dtype=np.float32))
        return np.asarray(
            _finite_matmul(q_ema, np.asarray(self.h0, dtype=np.float32), name="EMA FiLM fold"),
            dtype=np.float32,
        )

    def live_weight_numpy(self) -> np.ndarray:
        """Fold the live Q state for resume-custody verification."""

        if not self.initialized:
            raise RuntimeError("polar-chart state is not initialized")
        return np.asarray(
            _finite_matmul(
                np.asarray(self.q, dtype=np.float32),
                np.asarray(self.h0, dtype=np.float32),
                name="live FiLM fold",
            ),
            dtype=np.float32,
        )

    def state_arrays(self, prefix: str) -> dict[str, Any]:
        if not self.initialized:
            return {}
        return {
            prefix + "schema": np.asarray(RESUME_SCHEMA, np.int64),
            prefix + "method": np.asarray(METHOD_ID),
            prefix + "q": np.asarray(self.q, dtype=np.float32),
            prefix + "h0": np.asarray(self.h0, dtype=np.float32),
            prefix + "momentum": np.asarray(self.momentum, dtype=np.float32),
            prefix + "q_ema": np.asarray(self.q_ema, dtype=np.float32),
            prefix + "step": np.asarray(int(self.step), np.int64),
            prefix + "momentum_beta": np.asarray(float(self.momentum_beta), np.float64),
            prefix + "nesterov": np.asarray(int(bool(self.nesterov)), np.int64),
            prefix + "ns_steps": np.asarray(int(self.ns_steps), np.int64),
            prefix + "source_weight_sha256": np.asarray(self.source_weight_sha256),
        }

    def restore_from_cfg(self, prefix: str, cfg: dict) -> bool:
        if prefix + "schema" not in cfg:
            return False
        if int(cfg[prefix + "schema"]) != RESUME_SCHEMA:
            raise ValueError("unsupported film polar-chart resume schema")
        if str(cfg.get(prefix + "method")) != METHOD_ID:
            raise ValueError("film polar-chart resume method mismatch")
        if abs(float(cfg[prefix + "momentum_beta"]) - float(self.momentum_beta)) > 1e-12:
            raise ValueError("film polar-chart momentum_beta differs from typed config")
        if bool(int(cfg[prefix + "nesterov"])) != bool(self.nesterov):
            raise ValueError("film polar-chart nesterov differs from typed config")
        if int(cfg[prefix + "ns_steps"]) != int(self.ns_steps):
            raise ValueError("film polar-chart ns_steps differs from typed config")
        q = _as_f32_matrix(cfg[prefix + "q"], name="restored q")
        h0 = _as_f32_matrix(cfg[prefix + "h0"], name="restored h0")
        momentum = _as_f32_matrix(cfg[prefix + "momentum"], name="restored momentum")
        q_ema = _as_f32_matrix(cfg[prefix + "q_ema"], name="restored q_ema")
        if momentum.shape != q.shape or q_ema.shape != q.shape or h0.shape != (q.shape[1], q.shape[1]):
            raise ValueError("film polar-chart restored state shapes are inconsistent")
        self.q, self.h0, self.momentum, self.q_ema = q, h0, momentum, q_ema
        self.step = int(cfg[prefix + "step"])
        self.source_weight_sha256 = str(cfg[prefix + "source_weight_sha256"])
        if len(self.source_weight_sha256) != 64:
            raise ValueError("film polar-chart source SHA-256 is invalid")
        return True

    def telemetry_numpy(self) -> dict[str, Any]:
        if not self.initialized:
            return {"method": METHOD_ID, "initialized": False}
        q = np.asarray(self.q, dtype=np.float32)
        mom = np.asarray(self.momentum, dtype=np.float32)
        h0 = np.asarray(self.h0, dtype=np.float32)
        qt_q = _finite_matmul(q.T, q, name="telemetry Q.T@Q")
        qt_m = _finite_matmul(q.T, mom, name="telemetry Q.T@M")
        return {
            "method": METHOD_ID,
            "initialized": True,
            "step": int(self.step),
            "q_shape": list(q.shape),
            "h0_shape": list(h0.shape),
            "orthogonality_residual_fro": float(
                np.linalg.norm(qt_q - np.eye(q.shape[1]), ord="fro")),
            "momentum_tangent_residual_fro": float(np.linalg.norm(qt_m + qt_m.T, ord="fro")),
            "source_weight_sha256": self.source_weight_sha256,
        }


def resume_payload_json(state: FilmPolarChartSPELState) -> str:
    """Small deterministic inspection helper used by receipts/tests."""

    return json.dumps(state.telemetry_numpy(), sort_keys=True, separators=(",", ":"))


__all__ = [
    "METHOD_ID",
    "RESUME_PREFIX",
    "RESUME_SCHEMA",
    "FilmPolarChartSPELState",
    "NumpySPELStep",
    "muon_aspect_ratio_scale",
    "newton_schulz5_numpy",
    "polar_chart_numpy",
    "qr_retract_numpy",
    "resume_payload_json",
    "spel_step_mlx_arrays",
    "spel_step_numpy",
    "tangent_project_numpy",
]
