# SPDX-License-Identifier: MIT
"""``mx.compile`` fusion of the level-set witness d_seg training step (EXCLUDING pose).

What this is
------------
``mx.compile`` fuses MLX's many small launches (Linear + bias + FiLM affine +
activation + softmax + palette matmul + sigmoid + the resize glue) into fewer
kernels, killing the per-launch lazy-eval overhead the CPU dry-run + literature
show (``mlx_vs_torch_mps_bench`` §2/§4, push-plan P4). It is **graph fusion of the
same math** — so the compiled step is bit-identical to the uncompiled step (it is
NOT an approximation).

Scope (operator binding): wrap the **INR trunk + the d_seg grad closure** only.
The PoseNet verdict is EXCLUDED — pose explodes under compile, and in the witness
it is ``w-pose=0`` monitoring-only on the async CPU-authority verdict thread
(already off the training step). So the function handed to ``compile_loss_and_grad``
MUST be the seg-only loss-and-grad closure; the pose path stays uncompiled.

Authority note (NO-FAKE)
------------------------
``mx.compile`` changes neither the math nor the score authority. The d_seg/d_pose
VERDICT remains the FROZEN numpy/torch-CPU scorer on the byte-closed archive; MLX
(compiled or not) is a training-gradient substrate only. ``assert_compile_bit_identical``
is the equivalence gate (compiled == uncompiled, max|Δ|=0) + a determinism check.

Composition
-----------
The compiled step composes with the custom grouped-conv backward
(``metal_grouped_conv_backward``) and, when ``--fused-r-kernel`` is on, the fused
R-operator (``metal_fused_r_operator``): ``mx.compile`` captures the surrounding
graph and the ``@mx.custom_function`` ops keep their registered VJPs. (CPU here
validates the compile bit-identity of the pure-MLX trunk; full composition with the
Metal custom ops is validated by the GPU-gated harness when the GPU frees.)
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = [
    "compile_loss_and_grad",
    "assert_compile_bit_identical",
    "build_representative_dseg_trunk",
    "representative_dseg_loss_and_grad",
]


def compile_loss_and_grad(
    loss_and_grad_fn: Callable[..., Any],
    *,
    enabled: bool = True,
    shapeless: bool = False,
) -> Callable[..., Any]:
    """Return ``mx.compile(loss_and_grad_fn)`` when ``enabled`` else the fn unchanged.

    ``enabled`` is the ``--compile-step`` flag (DEFAULT OFF until GPU-validated).
    ``shapeless=True`` reuses the compiled graph across changing input shapes (use
    only when shapes truly vary; otherwise leave False so the compile cache keys on
    shape). The wrapped function MUST be the SEG-ONLY loss-and-grad closure (pose
    excluded — see module docstring).
    """

    if not enabled:
        return loss_and_grad_fn
    import mlx.core as mx

    return mx.compile(loss_and_grad_fn, shapeless=bool(shapeless))


def _flatten_tree(tree: Any) -> list[Any]:
    """Flatten an MLX pytree (dict/list/tuple of arrays) to a list of arrays."""

    from mlx.utils import tree_flatten

    return [v for _, v in tree_flatten(tree)]


def assert_compile_bit_identical(
    loss_and_grad_fn: Callable[..., Any],
    args: tuple,
    *,
    atol: float = 1e-6,
    det_atol: float = 0.0,
    deterministic_runs: int = 3,
    shapeless: bool = False,
) -> dict[str, Any]:
    """Assert ``mx.compile`` matches the uncompiled step (same math) + is deterministic.

    Runs ``loss_and_grad_fn(*args)`` uncompiled and compiled on the SAME inputs and
    compares the loss scalar and EVERY gradient-tree leaf. ``mx.compile`` is GRAPH
    FUSION of the SAME math, but kernel fusion can reorder fused multiply-adds, so the
    compiled output can differ at the fp32 ULP scale (~2e-7) — NOT a math change. So
    equivalence uses ``atol`` (default ``1e-6``, the spec's bound), while DETERMINISM
    (re-running the SAME compiled graph on the SAME inputs) is asserted EXACT
    (``det_atol=0.0`` — byte-stable). Returns a report; RAISES on any failure.
    CPU-safe (``mx.compile`` is device-agnostic).
    """

    import numpy as np
    import mlx.core as mx

    compiled = compile_loss_and_grad(loss_and_grad_fn, enabled=True, shapeless=shapeless)

    loss_u, grads_u = loss_and_grad_fn(*args)
    loss_c, grads_c = compiled(*args)
    mx.eval(loss_u, grads_u, loss_c, grads_c)

    loss_delta = float(abs(float(np.asarray(loss_u)) - float(np.asarray(loss_c))))
    gu = _flatten_tree(grads_u)
    gc = _flatten_tree(grads_c)
    if len(gu) != len(gc):
        raise AssertionError(
            f"compiled grad tree arity {len(gc)} != uncompiled {len(gu)}"
        )
    grad_max = 0.0
    for a, b in zip(gu, gc):
        d = float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) if a.size else 0.0
        grad_max = max(grad_max, d)

    # Determinism: the compiled step repeated must be byte-stable (EXACT).
    det_max = 0.0
    base_loss = float(np.asarray(loss_c))
    base_g = [np.asarray(v).copy() for v in gc]
    for _ in range(max(0, int(deterministic_runs) - 1)):
        l_i, g_i = compiled(*args)
        mx.eval(l_i, g_i)
        det_max = max(det_max, abs(float(np.asarray(l_i)) - base_loss))
        for ref, v in zip(base_g, _flatten_tree(g_i)):
            d = float(np.max(np.abs(ref - np.asarray(v)))) if ref.size else 0.0
            det_max = max(det_max, d)

    report = {
        "loss_max_abs_delta": loss_delta,
        "grad_max_abs_delta": grad_max,
        "determinism_max_abs_delta": det_max,
        "atol": float(atol),
        "det_atol": float(det_atol),
        "n_grad_leaves": len(gu),
        "equivalent_within_tol": bool(loss_delta <= atol and grad_max <= atol),
        "deterministic": bool(det_max <= det_atol),
    }
    if loss_delta > atol:
        raise AssertionError(
            f"compiled loss diverges beyond fp-fusion tol: |Δ|={loss_delta} > atol={atol}"
        )
    if grad_max > atol:
        raise AssertionError(
            f"compiled grad diverges beyond fp-fusion tol: max|Δ|={grad_max} > atol={atol}"
        )
    if det_max > det_atol:
        raise AssertionError(
            f"compiled step non-deterministic: max|Δ| across runs={det_max} > det_atol={det_atol}"
        )
    return report


# --------------------------------------------------------------------------- #
# Representative d_seg trunk + seg-only loss (mirrors the real witness trunk;
# pose EXCLUDED) — used to demonstrate + test the compile bit-identity on CPU.
# --------------------------------------------------------------------------- #


def build_representative_dseg_trunk(
    *,
    in_feat: int = 88,
    hidden_dim: int = 96,
    n_hidden: int = 4,
    n_classes: int = 5,
    mod_dim: int = 16,
    seed: int = 0,
):
    """Return an ``nn.Module`` mirroring the witness trunk forward.

    in_proj -> relu -> n_hidden x [Linear * (1+film_scale) + film_shift -> relu]
    -> out_sdf (K) + out_tex (3) -> softmax(phi)@palette -> sigmoid(base+tex)*255.
    (FiLM via a per-code projection; same op kinds as
    ``train_levelset_witness...LevelSetRGBWitness._trunk`` / ``_compose_rgb``.)
    """

    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(int(seed))

    class _Trunk(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_hidden = int(n_hidden)
            self.hidden_dim = int(hidden_dim)
            self.in_proj = nn.Linear(int(in_feat), int(hidden_dim))
            self.hidden = [nn.Linear(int(hidden_dim), int(hidden_dim)) for _ in range(int(n_hidden))]
            self.film = nn.Linear(int(mod_dim), int(n_hidden) * 2 * int(hidden_dim))
            self.out_sdf = nn.Linear(int(hidden_dim), int(n_classes))
            self.out_tex = nn.Linear(int(hidden_dim), 3)
            self.softmax_temp = 1.0
            # Fixed palette (K,3), like the witness class-color anchors.
            pal = mx.stack(
                [mx.array([float(k), -float(k), 0.5 * float(k)]) * 2.0 for k in range(int(n_classes))]
            )
            self.palette = pal

        def __call__(self, coord_feats, code):
            h = nn.relu(self.in_proj(coord_feats))
            film = mx.reshape(self.film(code), (self.n_hidden, 2, self.hidden_dim))
            for li, layer in enumerate(self.hidden):
                scale = 1.0 + film[li, 0]
                shift = film[li, 1]
                h = nn.relu(layer(h) * scale + shift)
            phi = self.out_sdf(h)              # (P, K)
            tex = self.out_tex(h)              # (P, 3)
            soft = mx.softmax(phi / self.softmax_temp, axis=-1)
            base = soft @ self.palette         # (P, 3)
            rgb = mx.sigmoid(base + tex) * 255.0
            return rgb, phi

    return _Trunk()


def representative_dseg_loss_and_grad(
    model,
    *,
    n_classes: int = 5,
    seed: int = 0,
):
    """Return ``(loss_and_grad_fn, example_args)`` for the representative trunk.

    The loss is a SEG-ONLY surrogate: a frozen linear "scorer" maps the trunk RGB to
    K logits and a cross-entropy against a fixed target argmax is taken (mirrors the
    d_seg CE/margin loss STRUCTURE; NO pose term — pose is excluded from compile).
    """

    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    rng = np.random.default_rng(int(seed))
    P = 256
    in_feat = int(model.in_proj.weight.shape[1])
    mod_dim = int(model.film.weight.shape[1])
    coord_feats = mx.array(rng.standard_normal((P, in_feat)).astype(np.float32))
    code = mx.array(rng.standard_normal((mod_dim,)).astype(np.float32))
    # Frozen "scorer surrogate": RGB(P,3) -> logits(P,K). Constant (not a parameter).
    scorer_w = mx.array(rng.standard_normal((3, int(n_classes))).astype(np.float32) * 0.01)
    target = mx.array(rng.integers(0, int(n_classes), size=(P,)).astype(np.int32))
    tgt_oh = mx.eye(int(n_classes))[target]

    # Standard MLX pattern: loss closes over `model`; nn.value_and_grad differentiates
    # wrt model params; the transformed fn takes only the data args.
    def loss_fn(feats, c):
        rgb, _phi = model(feats, c)
        logits = rgb @ scorer_w                       # (P, K) seg-surrogate logits
        logsum = mx.logsumexp(logits, axis=-1)        # (P,)
        tgt = mx.sum(logits * tgt_oh, axis=-1)        # (P,)
        return mx.mean(logsum - tgt)                  # CE (the d_seg loss structure)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    return loss_and_grad_fn, (coord_feats, code)
