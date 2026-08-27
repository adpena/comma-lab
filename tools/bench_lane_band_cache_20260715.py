# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""M3 (#509 burn-down 3): lane-band static-cache ON/OFF per-call timing.

Micro-bench of make_lane_band_compose_fn at the real geometry (384x512, dict
margin_provider = the precomputed-GT source, the config where the cache fires).
Values already proven bit-identical by test_analytic_lane_render_band.py; this
measures the per-call wall-clock delta on MLX-GPU. [macOS-MLX advisory] MEANS only.

IMPORT-TIME SAFETY (ddm_qd1, 2026-08-03).  Everything below used to sit at
module scope: the ``sys.path`` mutation, ``import mlx.core``, the array
allocation, and both timing loops -- so 8 epochs x 24 priors x 2 calls of
MLX-GPU work fired on a bare ``import``.

This was the ONLY true import-time GPU allocator in a measured 10,348-file scan
of src/, tools/, experiments/ and scripts/.  Worth stating plainly because it
inverts the backlog row that sent me here: the other 16 hits were deliberate
module-scope ``mx.set_default_device(mx.cpu)`` CPU-PINS, not GPU leaks.  Those
carry the opposite hazard (importing one silently pins the whole process to CPU,
which would force a GPU consumer onto CPU with no error) and are left alone here.

Nothing imports this module today, so the exposure was latent rather than live.
It is still worth closing: any tool that walks and imports ``tools/*.py`` -- a
coverage sweep, an AST-plus-import lever walk, a doc generator, a broad pytest
collection -- would have run the whole benchmark as a side effect and contended
for the GPU with a live training run.

Script behaviour is unchanged.  The RNG is still ``default_rng(0)`` and the
draws happen in the same order, so the reported numbers are identical to the
pre-fix version.
"""
import json
import sys
import time

import numpy as np

H, W = 384, 512
N_PRIORS = 24          # unique pair priors touched per epoch slice
CALLS_PER_PRIOR = 2    # ~2 compose calls/pair/epoch once the band gate opens
REPS = 8               # epochs simulated


class P:
    def __init__(self, cov):
        self.coverage = cov
        self.band_recall = 0.9


def main() -> int:
    # Deferred to call time on purpose: the sys.path mutation and the MLX
    # import are themselves side effects that must not fire on a bare import.
    sys.path.insert(
        0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src")
    )
    import mlx.core as mx

    from tac.boundary_math.analytic_lane_render_band import make_lane_band_compose_fn

    rng = np.random.default_rng(0)

    priors = {i: P(rng.random((H, W), dtype=np.float32)) for i in range(N_PRIORS)}
    margins = {i: rng.standard_normal((H, W)).astype(np.float32) for i in range(N_PRIORS)}
    lane_rgb = {i: rng.random((H, W, 3), dtype=np.float32) for i in range(N_PRIORS)}
    rgb = mx.array(rng.random((1, H, W, 3), dtype=np.float32))
    mx.eval(rgb)

    def build(cache_static):
        return make_lane_band_compose_fn(
            priors, lane_rgb_provider=lane_rgb, margin_provider=margins,
            tau=0.85, eps=0.35, weight=1.0, use_mlx=True, cache_static=cache_static)

    out = {}
    for label, cache in (("cache_on", True), ("cache_off", False)):
        fn = build(cache)
        # warm one full sweep (populates cache in ON arm; compiles kernels both arms)
        for i in range(N_PRIORS):
            mx.eval(fn(rgb, i))
        t0 = time.perf_counter()
        for _ in range(REPS):
            for i in range(N_PRIORS):
                for _c in range(CALLS_PER_PRIOR):
                    mx.eval(fn(rgb, i))
        dt = time.perf_counter() - t0
        calls = REPS * N_PRIORS * CALLS_PER_PRIOR
        out[label] = {"total_s": round(dt, 4),
                      "per_call_ms": round(1e3 * dt / calls, 4),
                      "calls": calls}

    per_call_delta_ms = out["cache_off"]["per_call_ms"] - out["cache_on"]["per_call_ms"]
    # n600 projection: 600 pairs x 2 calls/pair/epoch once band gate opens
    out["n600_projection_s_per_ep_saved"] = round(per_call_delta_ms * 600 * 2 / 1e3, 3)
    out["axis"] = "[macOS-MLX advisory] micro-bench; bit-identical values (tested)"
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
