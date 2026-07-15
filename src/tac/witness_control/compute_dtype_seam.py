# SPDX-License-Identifier: MIT
"""#509 burn-down batch 3 (2026-07-15): bf16/fp16 COMPUTE SEAM for the level-set witness trainer.

THE NAMED BUILD-OWED (memo ``.omx/research/wallclock_burndown_build_20260715.md`` §3 +
``m5max_unconstrained_leverage_campaign_20260715`` constraint 1): neither trainer had a dtype
seam — every witness forward/backward ran fp32 while M-series GPUs execute bf16/fp16 at ~2x
the fp32 rate (ESTIMATE-flagged until the paired bench lands). This module is the seam:

    fp32 MASTER WEIGHTS  +  low-precision COMPUTE for the witness forward/backward ONLY.

Boundary contract (the training-only relaxed-identity directive,
``feedback_wallclock_to_target_joint_objective_drift_ok_if_gradient_good_20260715``):

* **Inside the traced loss** the master fp32 params are ``astype``-cast to the compute dtype
  (gradients flow through the cast VJP and arrive back in fp32), and the module's public
  entry methods cast their fp32 array inputs down / cast every float output back UP to fp32.
  So the witness trunk computes low-precision, while everything DOWNSTREAM of the module
  boundary — the render/R operator (incl. the custom fused-R Metal kernel, which expects
  fp32), the FROZEN-SCORER forwards, and every loss reduction — receives fp32 arrays
  (bf16-valued; ``astype`` up-cast is exact).
* **Outside the traced loss** the seam is INERT: ``active`` is False, the entry shims pass
  straight through, and the module holds its fp32 masters — so EMA, checkpoints (resume-safe
  by construction: NOTHING new is persisted), the verdict render, decode, and byte-close are
  EXACTLY the incumbent fp32 paths. ``--compute-dtype fp32`` (the default) never constructs
  this object at all => byte-identical.

Gradient-quality gate (the C0 lesson, ``perparam_normalize_masks_all_norm_clipping_c0_confound
_20260715``): admission compares the POST-normalize update direction (cosine + relative norm)
against a same-master fp32 reference — ``update_direction_stats`` below is that comparator;
the trainer's ``--compute-dtype-quality-check N`` mode drives it for the first N optimizer
steps and steps with the fp32 REFERENCE update (the seam is measured ALONG the reference
trajectory, never conflated with trajectory divergence). Law leg:
``tac.canonical_equations.mixed_precision_compute_seam_20260715``. DSL leg:
``curriculum_dsl.ComputeDtype``.

means != ends: a sec/ep lever candidate; NO score claim; every number from it is
``[macOS-MLX advisory]`` NON-PROMOTABLE; pointer UNMOVED.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

COMPUTE_DTYPES = ("fp32", "bf16", "fp16")

# The witness module's public entry methods called by the loss/compose path (enumerated from
# the levelset trainer: __call__ @5475, sdf @6326/6366, call_margin @5325,
# render_lane_appearance @5333, call_batch = the micro-batch twin). Missing methods are
# skipped (base-trainer module variants lack the lane-band entries).
DEFAULT_ENTRY_METHODS = ("__call__", "call_batch", "sdf", "call_margin", "render_lane_appearance")


def _mx():
    import mlx.core as mx

    return mx


def resolve_compute_dtype(name: str):
    """'bf16'/'fp16'/'fp32' -> mlx dtype. ValueError on anything else (never-invent-values)."""
    mx = _mx()
    table = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}
    key = str(name)
    if key not in table:
        raise ValueError(f"unknown compute dtype {name!r}; expected one of {COMPUTE_DTYPES}")
    return table[key]


def _make_entry_shim(orig: Callable) -> Callable:
    """Class-method shim: when the instance's seam is ACTIVE, cast fp32 array args down to the
    compute dtype and cast float outputs back to fp32; otherwise byte-identical pass-through."""

    def shim(self, *a, **k):
        seam = getattr(self, "_cdt_seam", None)
        if seam is None or not seam.active:
            return orig(self, *a, **k)
        a2 = tuple(seam.cast_in(x) for x in a)
        return seam.cast_out_fp32(orig(self, *a2, **k))

    shim.__name__ = getattr(orig, "__name__", "cdt_shim")
    shim.__qualname__ = getattr(orig, "__qualname__", shim.__name__)
    shim._cdt_orig = orig  # introspection + idempotence marker
    return shim


class ComputeDtypeSeam:
    """Installable mixed-precision seam around ONE witness module instance.

    Construction patches ``type(model)`` entry methods (idempotent; the trainer builds the
    witness class per-instance inside its builder, so the patch is instance-scoped in
    practice) and binds itself on the instance (``model._cdt_seam``). The seam starts
    INACTIVE; only the wrapped value_and_grad callables flip it on around the traced loss."""

    def __init__(self, model: Any, dtype_name: str,
                 entry_methods: Sequence[str] = DEFAULT_ENTRY_METHODS) -> None:
        mx = _mx()
        self._mx_mod = mx
        self.model = model
        self.dtype_name = str(dtype_name)
        self.dtype = resolve_compute_dtype(dtype_name)
        if self.dtype == mx.float32:
            raise ValueError(
                "ComputeDtypeSeam is the LOW-precision seam; fp32 is the incumbent path — "
                "compose nothing instead of an inert seam (off-is-orphan rule).")
        self.active = False
        self._low_float_dtypes = (mx.bfloat16, mx.float16)
        cls = type(model)
        for name in entry_methods:
            orig = getattr(cls, name, None)
            if orig is None or getattr(orig, "_cdt_orig", None) is not None:
                continue  # absent on this module variant / already shimmed (idempotent)
            setattr(cls, name, _make_entry_shim(orig))
        model._cdt_seam = self

    # ---- casts -------------------------------------------------------------------
    def cast_in(self, x: Any) -> Any:
        """fp32 mx.array -> compute dtype; everything else (ints/bools/python) untouched."""
        mx = self._mx_mod
        if isinstance(x, mx.array) and x.dtype == mx.float32:
            return x.astype(self.dtype)
        return x

    def cast_out_fp32(self, out: Any) -> Any:
        """Low-precision float outputs -> fp32 (exact up-cast) so every consumer (render/R,
        frozen scorers, losses, fused kernels) sees fp32 arrays. Tuples/lists mapped."""
        mx = self._mx_mod
        if isinstance(out, mx.array):
            return out.astype(mx.float32) if out.dtype in self._low_float_dtypes else out
        if isinstance(out, tuple):
            return tuple(self.cast_out_fp32(o) for o in out)
        if isinstance(out, list):
            return [self.cast_out_fp32(o) for o in out]
        return out

    def cast_tree(self, tree: Any) -> Any:
        """fp32 leaves of a param tree -> compute dtype (the inside-the-trace master cast;
        the astype VJP returns fp32 gradients w.r.t. the fp32 masters)."""
        from mlx.utils import tree_map

        mx = self._mx_mod
        return tree_map(
            lambda a: a.astype(self.dtype)
            if isinstance(a, mx.array) and a.dtype == mx.float32 else a,
            tree)

    # ---- traced-loss wrappers ----------------------------------------------------
    def wrap_module_value_and_grad(self, loss_fn: Callable) -> Callable:
        """The single-path replacement for ``nn.value_and_grad(model, loss_fn)``.

        Same call convention (``vag(model, *args)`` — the model is forwarded to ``loss_fn``
        as its first arg, exactly like the nn wrapper). Differentiates w.r.t. the fp32
        MASTERS; inside the trace the masters are cast to the compute dtype and installed
        via ``model.update`` (the nn.value_and_grad pattern + one cast). The fp32 masters
        are RESTORED after every call, so the module never leaks low-precision params to
        any outside consumer (EMA / checkpoints / verdict / probes)."""
        import mlx.core as mx

        model = self.model
        seam = self

        def _inner(params, *call_args):
            model.update(seam.cast_tree(params))
            return loss_fn(*call_args)

        raw = mx.value_and_grad(_inner)

        def vag(*call_args):
            master = model.trainable_parameters()
            seam.active = True
            try:
                out = raw(master, *call_args)
            finally:
                seam.active = False
                model.update(master)
            return out

        vag._cdt_seam = seam
        return vag

    def wrap_dual_value_and_grad(self, raw_dual: Callable) -> Callable:
        """Wrap the DUAL (witness + island-seed) mx.value_and_grad: flips the seam active
        around the call and restores the fp32 masters after (the dual inner loss must
        already route its ``model.update`` through ``cast_tree``; the seed tree stays
        fp32 — it composes at the fp32 module boundary)."""
        model = self.model
        seam = self

        def vag(model_params, *rest):
            seam.active = True
            try:
                out = raw_dual(model_params, *rest)
            finally:
                seam.active = False
                model.update(model_params)
            return out

        vag._cdt_seam = seam
        return vag


# ---- gradient-quality comparator (the POST-normalize direction check; C0 lesson) ------
def _flatten_float_leaves(tree: Any, out: list[np.ndarray]) -> None:
    mx = _mx()
    if isinstance(tree, mx.array):
        out.append(np.asarray(tree.astype(mx.float32), dtype=np.float64).ravel())
        return
    if isinstance(tree, dict):
        for k in sorted(tree):
            _flatten_float_leaves(tree[k], out)
        return
    if isinstance(tree, (list, tuple)):
        for v in tree:
            _flatten_float_leaves(v, out)


def flatten_update_tree(tree: Any) -> np.ndarray:
    """Deterministic (sorted-key) fp64 flattening of an update/gradient tree."""
    leaves: list[np.ndarray] = []
    _flatten_float_leaves(tree, leaves)
    if not leaves:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(leaves)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def update_direction_stats(lowp_tree: Any, ref_tree: Any) -> dict[str, Any]:
    """POST-normalize update-direction comparison: low-precision arm vs the fp32 reference.

    Returns ``cosine`` (global flattened direction agreement), ``rel_norm``
    (||u_lowp||/||u_ref||; 1.0 = magnitude-faithful), the two norms, and per-top-level-group
    cosines (diagnosis: WHICH param group degrades first). Computed in fp64 numpy —
    deterministic, no MLX graph effect. The C0 lesson is baked into the CALLER contract:
    compare AFTER the trainer's clip+normalize pipeline (per-param normalize divides out any
    uniform per-tensor scale, so a pre-normalize comparison would grade a direction the
    optimizer never sees)."""
    u_l = flatten_update_tree(lowp_tree)
    u_r = flatten_update_tree(ref_tree)
    if u_l.shape != u_r.shape:
        raise ValueError(
            f"update trees disagree in flattened size ({u_l.size} vs {u_r.size}) — "
            "not the same parameter tree (NO-FAKE: refusing a meaningless cosine).")
    n_l, n_r = float(np.linalg.norm(u_l)), float(np.linalg.norm(u_r))
    per_group: dict[str, float] = {}
    if isinstance(lowp_tree, dict) and isinstance(ref_tree, dict):
        for k in sorted(set(lowp_tree) & set(ref_tree)):
            per_group[str(k)] = _cosine(
                flatten_update_tree(lowp_tree[k]), flatten_update_tree(ref_tree[k]))
    return {
        "cosine": _cosine(u_l, u_r),
        "rel_norm": (n_l / n_r) if n_r > 0.0 else float("inf") if n_l > 0.0 else 1.0,
        "l2_lowp": n_l,
        "l2_ref": n_r,
        "per_group_cosine": per_group,
    }


__all__ = [
    "COMPUTE_DTYPES",
    "DEFAULT_ENTRY_METHODS",
    "ComputeDtypeSeam",
    "flatten_update_tree",
    "resolve_compute_dtype",
    "update_direction_stats",
]
