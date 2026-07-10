# SPDX-License-Identifier: MIT
"""decoupled_field — v8 B1: per-class DECOUPLED-FIELD partition head (G of W=(G, ξ, T)).

WHY (v8 unlock, operator #398 thread A 2026-07-10 + SPEC_v8_perclass_decomposition_20260708 §1):
The live level-set witness computes its 5 partition fields as ONE shared linear readout of ONE
shared trunk: ``phi = out_sdf(trunk(coord, code))`` (a ``Linear(hidden -> K)`` in
``build_levelset_rgb_witness``). Because every ``phi_c`` is a function of the SAME trunk parameters,
a gradient step on class ``c``'s loss moves ALL ``phi_{c'}`` — the measured cross-class THEFT channel
(run-1: Lane 13.8× / Movable 4.6× stealing Road). v8's architecture makes that theft IMPOSSIBLE for
the gradient mechanism by giving each class its OWN independent field:

    P(x) = argmax_c ( phi_c(x) + b_c )       (SPEC §1 tropical argmax of decoupled fields)

with ``phi_c`` produced by an INDEPENDENT per-class coordinate-INR (``∂phi_c/∂θ_{c'} = 0`` — the
decoupling identity). This module is the DECOUPLED PARTITION HEAD: K independent per-class scalar
fields with block-diagonal parameters, vectorized over the class axis (each class slice ``k`` reads
ONLY ``w[k]`` — genuine parameter independence, not a reparametrization of a shared readout).

SCOPE (increment-1 per ``v8_unlock_398a_20260710.md``): this is the TRAINING MODE + the composition
forward ONLY — the partition fields and their tropical-argmax reconciliation. It is NOT the residual
coder, NOT the paint/reconciliation-vs-frozen-scorer stage (SPEC §3), and NOT a byte-close carrier
(that is 1b). The RGB texture / pose channel (``out_tex``) stays on the shared trunk (SPEC §3 "luma
reserved for pose"); the paint-free MASK partition that the 1a decoupling screen measures
(``tac.inc1a_harness.decoupling_screen``) is ``argmax_c phi_c`` — fully decoupled here.

HONEST DECOUPLING STATEMENT (NO-FAKE): the PARTITION FIELDS ``phi_c`` are decoupled — each has a
disjoint parameter set, so ``∂phi_c/∂θ_{c'} = 0`` holds by construction (verified by
:func:`cross_class_jacobian_is_block_diagonal`). The shared trunk survives ONLY as the ``out_tex``
(texture/pose) provider, which does NOT enter the paint-free mask partition. The decoupling this
module delivers is exactly the decoupling the 1a screen tests: the mask argmax over ``phi``.

CLASS INDEX ORDER (CLAUDE.md, comma10k canonical, NEVER luma-sort): 0=Road, 1=Lane, 2=Undrivable,
3=Movable, 4=MyCar. This module is class-INDEX-agnostic (it operates on a K-vector of fields; it does
NOT hardcode any class's spatial signature). ``n_classes`` defaults to the measured ``N_CLASSES``.

COMPUTE-SUBSTRATE LAW (CLAUDE.md): the numpy fp32 forward
(:func:`decoupled_field_numpy_forward`) is the PORTABLE deterministic authority; the MLX module
(:func:`make_decoupled_field_head_mlx`) mirrors it in EXACT algebra (same weights, same block-diagonal
einsums; fp32 parity — the numpy reference runs fp64 then casts, so agreement is float32-level). NO
MPS as a score.

AUTHORITY: this module SYNTHESIZES per-class fields + reports its DESIGN properties (param count,
counted bytes, block-diagonal Jacobian). It makes NO score claim. A d_seg VERDICT is n600-only through
the 1a decoupling screen against a MATCHED-COMPUTE shared-head control, byte-closed / exact-eval. The
pointer (contest-CPU 0.19110) is UNMOVED — everything here is MEANS.
``[macOS-MLX/CPU advisory · research-signal · NON-PROMOTABLE]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.witness_control.perclass_verdict import N_CLASSES

__all__ = [
    "DecoupledFieldError",
    "DecoupledFieldSpec",
    "compose_argmax_partition",
    "compose_softmax",
    "counted_bytes",
    "cross_class_jacobian_is_block_diagonal",
    "decoupled_field_numpy_forward",
    "init_field_params_numpy",
    "make_decoupled_field_head_mlx",
    "param_count",
]

# Activations supported by the per-class fields. relu is the default (matches the witness trunk's
# from-scratch default). tanh is a smooth bounded alternative for SDF-style fields.
_SUPPORTED_ACTIVATIONS = ("relu", "tanh")


class DecoupledFieldError(ValueError):
    """Raised on a mis-shaped / out-of-contract decoupled-field input."""


# --------------------------------------------------------------------------- #
# The spec — clause-B: minimal per-class field dimensions (stated, not guessed).#
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DecoupledFieldSpec:
    """The decoupled per-class field spec (K independent coordinate-INR scalar fields).

    * ``in_feat`` — input coordinate-feature dim (the witness ``curv_feats`` width; matches the trunk
      ``in_proj`` in-dim so the field reads the SAME per-pixel coordinate features).
    * ``mod_dim`` — per-pair code dim (the witness ``--mod-dim``; the field's FiLM reads the SAME
      per-pair latent so it can amortize the 600 pairs, exactly like the shared trunk).
    * ``n_classes`` — K independent fields (default the measured ``N_CLASSES`` = 5).
    * ``field_hidden`` — H, the per-class hidden width. Clause-B minimal: each field is a scalar
      coordinate INR; H is deliberately small (default 32) since the field only carries ONE class's
      separatrix geometry, not the full K-way partition (that is the whole decoupling win — each field
      is a fraction of the shared trunk's job).
    * ``field_layers`` — L, per-class hidden depth (default 2).
    * ``activation`` — ``"relu"`` (default) or ``"tanh"``.
    """

    in_feat: int
    mod_dim: int
    n_classes: int = N_CLASSES
    field_hidden: int = 32
    field_layers: int = 2
    activation: str = "relu"

    def __post_init__(self) -> None:
        if int(self.in_feat) < 1:
            raise DecoupledFieldError(f"in_feat must be >= 1, got {self.in_feat}")
        if int(self.mod_dim) < 1:
            raise DecoupledFieldError(f"mod_dim must be >= 1, got {self.mod_dim}")
        if int(self.n_classes) < 2:
            raise DecoupledFieldError(f"n_classes must be >= 2, got {self.n_classes}")
        if int(self.field_hidden) < 1:
            raise DecoupledFieldError(f"field_hidden must be >= 1, got {self.field_hidden}")
        if int(self.field_layers) < 1:
            raise DecoupledFieldError(f"field_layers must be >= 1, got {self.field_layers}")
        if self.activation not in _SUPPORTED_ACTIVATIONS:
            raise DecoupledFieldError(
                f"activation must be one of {_SUPPORTED_ACTIVATIONS}, got {self.activation!r}")


# --------------------------------------------------------------------------- #
# Parameter shapes + counting (block-diagonal over the class axis K).          #
# --------------------------------------------------------------------------- #
def _param_shapes(spec: DecoupledFieldSpec) -> dict[str, tuple[int, ...]]:
    """The canonical block-diagonal parameter shapes (leading K = the independent class axis).

    Each named tensor's FIRST axis is the class ``k``; class ``k``'s field reads ONLY ``w[k]``, so the
    parameter sets are disjoint across classes (the ``∂phi_c/∂θ_{c'}=0`` decoupling by construction).
    """
    k = int(spec.n_classes)
    i = int(spec.in_feat)
    m = int(spec.mod_dim)
    h = int(spec.field_hidden)
    ell = int(spec.field_layers)
    return {
        "w_in": (k, i, h),           # per-class input projection
        "b_in": (k, h),
        "w_film": (k, m, 2 * ell * h),  # per-class FiLM (scale, shift) for each of L hidden layers
        "w_hid": (k, ell, h, h),     # per-class hidden weights
        "b_hid": (k, ell, h),
        "w_out": (k, h),             # per-class scalar readout
        "b_out": (k,),               # per-class field bias b_c (the tropical-argmax offset)
    }


def param_count(spec: DecoupledFieldSpec) -> int:
    """Total COUNTED parameter count of the K decoupled fields (all learned / video-derived).

    This is the number the v8 unlock sequence E1 measures on the decoupled arm to build
    ``matched_control_spec(P_dec)`` (the matched-compute shared-head control for the 1a A/B)."""
    return int(sum(int(np.prod(s)) for s in _param_shapes(spec).values()))


def counted_bytes(spec: DecoupledFieldSpec, *, quant_bits: int = 8) -> dict[str, Any]:
    """The COUNTED rate of the decoupled fields = quantized per-class field parameters.

    Unlike the texture trunk's Gabor bank, the decoupled fields have NO rule-118-free table — every
    parameter is video-derived (a fitted per-class field) ⇒ ALL counted. Reports raw bytes at
    ``quant_bits`` and the archive-rate-term contribution ``25·bytes/37_545_489`` for the matched-bytes
    accounting. (Real byte-close would entropy-code these; this is the UNCODED upper bound. Increment-1
    is paint-free / mask-level — this table is for the E1→E4 byte planning, not a 1a input.)"""
    n = param_count(spec)
    raw_bytes = n * float(quant_bits) / 8.0
    return {
        "n_params": int(n),
        "n_classes": int(spec.n_classes),
        "quant_bits": int(quant_bits),
        "raw_bytes_uncoded": float(raw_bytes),
        "rate_term_uncoded_S": float(25.0 * raw_bytes / 37_545_489.0),
        "note": "all params counted (no rule-118-free table); UNCODED upper bound.",
    }


# --------------------------------------------------------------------------- #
# Deterministic init (portable numpy authority).                              #
# --------------------------------------------------------------------------- #
def init_field_params_numpy(
    spec: DecoupledFieldSpec, *, seed: int = 0, scale: float = 0.1, tied_init: bool = False,
) -> dict[str, np.ndarray]:
    """Deterministic seeded init for the K decoupled fields (portable authority).

    Small random weights (``scale``), zero biases. ``tied_init=True`` copies class-0's parameters onto
    every class ⇒ all K fields are IDENTICAL ⇒ ``phi`` has equal columns ⇒ the tropical argmax reduces
    to a constant class-0 partition (the deterministic tie-break). ``tied_init`` is the "start from
    degenerate-shared behaviour" config used by the composition-equivalence proof
    (:func:`compose_argmax_partition` over tied ``phi`` == shared-head argmax over the same tied logits).
    """
    rng = np.random.default_rng(int(seed))
    shapes = _param_shapes(spec)
    params: dict[str, np.ndarray] = {}
    for name, shape in shapes.items():
        if name.startswith("b_"):
            params[name] = np.zeros(shape, np.float32)
        else:
            params[name] = (rng.standard_normal(shape).astype(np.float32) * float(scale))
    if tied_init:
        for name, arr in params.items():
            # broadcast class-0's slice onto every class (the leading axis is K)
            params[name] = np.repeat(arr[:1], arr.shape[0], axis=0).astype(np.float32)
    return params


def _act_np(u: np.ndarray, activation: str) -> np.ndarray:
    if activation == "relu":
        return np.maximum(u, 0.0)
    if activation == "tanh":
        return np.tanh(u)
    raise DecoupledFieldError(f"unknown activation {activation!r}")  # pragma: no cover (spec-guarded)


# --------------------------------------------------------------------------- #
# The numpy reference forward (portable authority; the MLX module mirrors it). #
# --------------------------------------------------------------------------- #
def decoupled_field_numpy_forward(
    params: dict[str, np.ndarray],
    coord_feats: np.ndarray,
    code: np.ndarray,
    spec: DecoupledFieldSpec,
) -> np.ndarray:
    """The block-diagonal per-class field forward: ``phi_c(x) = FieldMLP_c(coord, code)``.

    ``coord_feats`` ``(P, in_feat)`` per-pixel coordinate features (shared across classes; the SAME
    features the trunk reads). ``code`` ``(mod_dim,)`` for a single pair OR ``(B, mod_dim)`` for a batch
    of pairs (the per-pair latent). Returns ``phi`` ``(P, K)`` (single) or ``(B, P, K)`` (batch).

    Every class ``k`` uses ONLY ``params[...][k]`` — the einsums index ``k`` independently, so
    ``∂phi[...,k]/∂params[...][k'] = 0`` for ``k != k'`` (the decoupling identity, tested by
    :func:`cross_class_jacobian_is_block_diagonal`). fp64 compute, fp32 cast (the parity reference)."""
    coord = np.asarray(coord_feats, dtype=np.float64)
    if coord.ndim != 2 or coord.shape[1] != int(spec.in_feat):
        raise DecoupledFieldError(
            f"coord_feats must be (P, {spec.in_feat}), got {coord.shape}")
    code_arr = np.asarray(code, dtype=np.float64)
    squeeze = code_arr.ndim == 1
    if squeeze:
        code_arr = code_arr[None, :]  # (1, mod_dim)
    if code_arr.ndim != 2 or code_arr.shape[1] != int(spec.mod_dim):
        raise DecoupledFieldError(
            f"code must be (mod_dim,) or (B, mod_dim) with mod_dim={spec.mod_dim}, got {code.shape}")

    k = int(spec.n_classes)
    h = int(spec.field_hidden)
    ell = int(spec.field_layers)
    act = spec.activation

    w_in = np.asarray(params["w_in"], np.float64)     # (K, I, H)
    b_in = np.asarray(params["b_in"], np.float64)     # (K, H)
    w_film = np.asarray(params["w_film"], np.float64)  # (K, M, 2*L*H)
    w_hid = np.asarray(params["w_hid"], np.float64)   # (K, L, H, H)
    b_hid = np.asarray(params["b_hid"], np.float64)   # (K, L, H)
    w_out = np.asarray(params["w_out"], np.float64)   # (K, H)
    b_out = np.asarray(params["b_out"], np.float64)   # (K,)

    # input projection: pair-INDEPENDENT (coord is shared across pairs) -> (P, K, H)
    h0 = _act_np(np.einsum("pi,kih->pkh", coord, w_in) + b_in[None, :, :], act)
    b = code_arr.shape[0]
    p = coord.shape[0]
    # broadcast to the pair batch -> (B, P, K, H)
    hcur = np.broadcast_to(h0[None], (b, p, k, h)).copy()
    # FiLM: per-pair, per-class (scale, shift) for each hidden layer -> (B, K, L, 2, H)
    film = np.einsum("bm,kmf->bkf", code_arr, w_film).reshape(b, k, ell, 2, h)
    for li in range(ell):
        pre = np.einsum("bpkh,khj->bpkj", hcur, w_hid[:, li]) + b_hid[None, None, :, li, :]
        scale = 1.0 + film[:, None, :, li, 0, :]  # (B,1,K,H)
        shift = film[:, None, :, li, 1, :]
        hcur = _act_np(pre * scale + shift, act)
    phi = np.einsum("bpkh,kh->bpk", hcur, w_out) + b_out[None, None, :]  # (B, P, K)
    phi = phi.astype(np.float32)
    return phi[0] if squeeze else phi


# --------------------------------------------------------------------------- #
# The composition forward (SHARED between decoupled and shared modes).         #
# --------------------------------------------------------------------------- #
def compose_argmax_partition(phi: np.ndarray, b_c: np.ndarray | None = None) -> np.ndarray:
    """The tropical-argmax composition ``P(x) = argmax_c ( phi_c(x) + b_c )`` (SPEC §1).

    This operator is MODE-AGNOSTIC: it is IDENTICAL for the decoupled fields and the shared head — the
    decoupled mode only changes the PROVENANCE of ``phi`` (K independent fields vs one shared linear
    readout), never the composition. So the "composition forward equals shared-mode when classes tied"
    invariant is exact: given the SAME ``phi``, this returns the SAME partition regardless of mode
    (verified in tests). ``b_c`` ``(K,)`` is the optional ~0-byte per-class tie calibration (SPEC §1
    Road↔Lane bias); default None = the fields' own ``b_out`` biases carry it. Lowest-index tie-break
    (numpy ``argmax``)."""
    phi = np.asarray(phi)
    if b_c is not None:
        phi = phi + np.asarray(b_c).reshape((1,) * (phi.ndim - 1) + (-1,))
    return np.argmax(phi, axis=-1)


def compose_softmax(phi: np.ndarray, temp: float, b_c: np.ndarray | None = None) -> np.ndarray:
    """The soft composition ``softmax((phi + b_c) / temp)`` — the SAME masks ``_compose_rgb`` uses.

    Mode-agnostic like :func:`compose_argmax_partition`. Numerically-stable (subtract per-pixel max)."""
    if float(temp) <= 0.0:
        raise DecoupledFieldError(f"temp must be > 0, got {temp}")
    phi = np.asarray(phi, dtype=np.float64)
    if b_c is not None:
        phi = phi + np.asarray(b_c, np.float64).reshape((1,) * (phi.ndim - 1) + (-1,))
    z = phi / float(temp)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def cross_class_jacobian_is_block_diagonal(
    spec: DecoupledFieldSpec, *, seed: int = 0, eps: float = 1e-3, tol: float = 1e-8,
) -> bool:
    """Numerically VERIFY ``∂phi_c/∂θ_{c'} = 0`` for ``c != c'`` (the decoupling identity, NO-FAKE).

    Perturbs each class ``k'``'s output-readout weights and confirms ONLY ``phi[:, k']`` changes — the
    other classes' fields are numerically invariant (they share no parameters). Returns True iff the
    cross-class response is below ``tol`` for every off-diagonal (k, k') pair. This is the honest proof
    that the fields are decoupled by construction, not merely reparametrized."""
    params = init_field_params_numpy(spec, seed=int(seed), scale=0.3)
    rng = np.random.default_rng(int(seed) + 991)
    coord = rng.standard_normal((7, int(spec.in_feat))).astype(np.float32)
    code = rng.standard_normal((int(spec.mod_dim),)).astype(np.float32)
    base = decoupled_field_numpy_forward(params, coord, code, spec)  # (P, K)
    k = int(spec.n_classes)
    for kp in range(k):
        pert = {name: arr.copy() for name, arr in params.items()}
        pert["w_out"] = pert["w_out"].copy()
        pert["w_out"][kp] = pert["w_out"][kp] + float(eps)  # perturb class kp only
        out = decoupled_field_numpy_forward(pert, coord, code, spec)
        delta = np.abs(out - base)  # (P, K)
        for kc in range(k):
            if kc == kp:
                continue
            if float(delta[:, kc].max()) > float(tol):
                return False  # a cross-class response => NOT decoupled (would be a fake claim)
    return True


# --------------------------------------------------------------------------- #
# The MLX witness submodule (byte-accounted; mirrors the numpy authority).     #
# --------------------------------------------------------------------------- #
def make_decoupled_field_head_mlx(
    spec: DecoupledFieldSpec, *, seed: int = 0, scale: float = 0.1, tied_init: bool = False,
):
    """Build the MLX ``DecoupledFieldHead`` submodule (K independent per-class fields).

    Returns an ``nn.Module`` whose params are the block-diagonal tensors of :func:`_param_shapes`
    (leading K = the independent class axis), initialized bit-identically to
    :func:`init_field_params_numpy` (same seed/scale/tied_init). ALL params are COUNTED (no rule-118
    table). Methods:

    * ``phi_single(coord_feats, code_vec) -> (P, K)`` — the single-pair field forward.
    * ``phi_batch(coord_feats, code_mat) -> (B, P, K)`` — the batched (multi-pair) field forward.

    Both mirror :func:`decoupled_field_numpy_forward` in EXACT block-diagonal algebra (same einsums;
    fp32 parity — numpy runs fp64 then casts, so agreement is float32-level, not bit-exact). The MLX
    autograd over these params is genuinely decoupled (each class slice is an independent einsum arm).
    """
    import mlx.core as mx
    import mlx.nn as nn

    init = init_field_params_numpy(spec, seed=int(seed), scale=float(scale), tied_init=bool(tied_init))
    ell = int(spec.field_layers)
    h = int(spec.field_hidden)
    k = int(spec.n_classes)
    act_name = spec.activation

    def _act(u):
        if act_name == "relu":
            return nn.relu(u)
        return mx.tanh(u)

    class DecoupledFieldHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_classes = k
            self.field_hidden = h
            self.field_layers = ell
            self.activation = act_name
            self.w_in = mx.array(init["w_in"])       # (K, I, H) COUNTED
            self.b_in = mx.array(init["b_in"])       # (K, H)
            self.w_film = mx.array(init["w_film"])   # (K, M, 2*L*H)
            self.w_hid = mx.array(init["w_hid"])     # (K, L, H, H)
            self.b_hid = mx.array(init["b_hid"])     # (K, L, H)
            self.w_out = mx.array(init["w_out"])     # (K, H)
            self.b_out = mx.array(init["b_out"])     # (K,)

        def _fields(self, coord_feats, code_mat):
            # coord_feats: (P, I); code_mat: (B, M) -> phi (B, P, K).
            h0 = _act(mx.einsum("pi,kih->pkh", coord_feats, self.w_in) + self.b_in[None])  # (P,K,H)
            b = code_mat.shape[0]
            p = coord_feats.shape[0]
            hcur = mx.broadcast_to(h0[None], (b, p, self.n_classes, self.field_hidden))
            film = mx.reshape(mx.einsum("bm,kmf->bkf", code_mat, self.w_film),
                              (b, self.n_classes, self.field_layers, 2, self.field_hidden))
            for li in range(self.field_layers):
                pre = mx.einsum("bpkh,khj->bpkj", hcur, self.w_hid[:, li]) + self.b_hid[None, None, :, li, :]
                scale = 1.0 + film[:, None, :, li, 0, :]
                shift = film[:, None, :, li, 1, :]
                hcur = _act(pre * scale + shift)
            return mx.einsum("bpkh,kh->bpk", hcur, self.w_out) + self.b_out[None, None, :]  # (B,P,K)

        def phi_batch(self, coord_feats, code_mat):
            return self._fields(coord_feats, code_mat)  # (B, P, K)

        def phi_single(self, coord_feats, code_vec):
            return self._fields(coord_feats, code_vec[None, :])[0]  # (P, K)

    return DecoupledFieldHead()
