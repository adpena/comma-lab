"""Numpy REFERENCE for the normalized normal-direction curvature term kappa_n = n^T H n.

Task #318 (DE-DERIVATION). This is a *reference oracle only* — it does NOT run in any
trainer and touches no training state. Its purpose is to give the v6 eikonal symposium
(#317) a byte-parity target for an MLX `--eikonal-normal-curvature-weight` term, the
"normalized n^T H n" follow-up named (but NOT built) by the eikonal-stabilizer arbitration
(`eikonal_stabilizer_build_20260705.md` reading 4.2 + DAG FEED-05y candidate (b)).

WHY normalized (the whole point). The paper's *raw* StEik directional-divergence term is
`|grad(m)^T H(m) grad(m)|`, which carries a `|grad m|^2` prefactor. At the far-from-SDF
runaway state (|grad m| ~ 2070, MEASURED) that prefactor makes the damping term itself a
runaway (measured 575x-1431x self-amplification, an implementation-level NO-GO per #307).
The NORMALIZED form divides it out:

    n       = grad(m) / |grad(m)|                    (unit level-set normal)
    kappa_n = n^T H(m) n = (grad(m)^T H grad(m)) / |grad(m)|^2

kappa_n is the SECOND derivative of m along its own normal direction (normal-direction
curvature of the margin field). The exact scaling under m -> c*m (MEASURED in `self_test`):

    grad^T H grad  ~  c^3    (three factors linear in m)
    |grad m|^2     ~  c^2
    kappa_n = (grad^T H grad)/|grad m|^2  ~  c^1     (LINEAR in the field)
    raw StEik = |grad^T H grad|           ~  c^3     (CUBIC in the field)

So kappa_n is NOT scale-invariant (a normal 2nd-derivative carries the units of m). What the
normalization removes is the |grad m|^2 = c^2 OVER-amplification of the raw term: at the
far-from-SDF state |grad m| ~ 2000 the raw term is ~ |grad m|^2 ~ 4e6 times larger than
kappa_n, which is exactly why raw self-amplified (measured 575x-1431x) and kappa_n does not.
kappa_n scales LINEARLY with the field, the same class as the eikonal residual (|grad m|-1)
it is meant to damp -- comparably scaled, not self-amplifying relative to it. That
scaling-power law (kappa ~ c^1, raw ~ c^3, ratio ~ c^2 = |grad m|^2) is what the symposium
must reproduce in its MLX port; `self_test()` verifies it at |grad m| in {1, 100, 2000}.

Discretization matches the sibling's ViscoReg stencils: central differences on the
(H-2, W-2) interior grid. Field convention: `m` is a 2-D array m[i, j], axis 0 = rows (y),
axis 1 = cols (x). Boundary cells are returned as 0 (no valid central stencil) so a
mean/sum reduction over the term is well defined and edge-artifact-free.

Determinism: pure numpy, no RNG in the term itself (the self-test seeds its own generator).
Default-safe: importing this module runs nothing; call the functions explicitly.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "normal_direction_curvature",
    "raw_steik_directional_divergence",
    "self_test",
]

_EPS = 1e-12  # guards |grad m| = 0 (flat plateaus); tiny, does not perturb the term where grad != 0


def _central_grad(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First derivatives (m_y, m_x) via central differences on the interior; 0 on the border."""
    my = np.zeros_like(m)
    mx = np.zeros_like(m)
    my[1:-1, :] = 0.5 * (m[2:, :] - m[:-2, :])
    mx[:, 1:-1] = 0.5 * (m[:, 2:] - m[:, :-2])
    return my, mx


def _central_hessian(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Second derivatives (m_yy, m_xx, m_xy) via central differences on the interior."""
    myy = np.zeros_like(m)
    mxx = np.zeros_like(m)
    mxy = np.zeros_like(m)
    myy[1:-1, :] = m[2:, :] - 2.0 * m[1:-1, :] + m[:-2, :]
    mxx[:, 1:-1] = m[:, 2:] - 2.0 * m[:, 1:-1] + m[:, :-2]
    mxy[1:-1, 1:-1] = 0.25 * (
        m[2:, 2:] - m[2:, :-2] - m[:-2, 2:] + m[:-2, :-2]
    )
    return myy, mxx, mxy


def normal_direction_curvature(m: np.ndarray) -> np.ndarray:
    """kappa_n = n^T H n = (grad^T H grad) / |grad|^2 on the interior grid; 0 on the border.

    Scale-invariant in m (this is the whole point vs the raw StEik term).
    Returns an array the shape of `m`; border cells are 0.
    """
    m = np.asarray(m, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"expected a 2-D margin field, got shape {m.shape}")
    my, mx = _central_grad(m)
    myy, mxx, mxy = _central_hessian(m)
    # grad^T H grad = my^2*myy + 2*my*mx*mxy + mx^2*mxx
    quad = my * my * myy + 2.0 * my * mx * mxy + mx * mx * mxx
    gnorm2 = my * my + mx * mx
    kappa = np.zeros_like(m)
    interior = np.zeros_like(m, dtype=bool)
    interior[1:-1, 1:-1] = True
    denom = gnorm2 + _EPS
    kappa[interior] = quad[interior] / denom[interior]
    return kappa


def raw_steik_directional_divergence(m: np.ndarray) -> np.ndarray:
    """The paper's RAW term |grad^T H grad| (NOT normalized) — for contrast only.

    Provided so the symposium can reproduce the measured self-amplification: this term
    scales like |grad m|^2 (see self_test), which is exactly why it blew up at the
    far-from-SDF state. DO NOT use as the trainer term; use `normal_direction_curvature`.
    """
    m = np.asarray(m, dtype=np.float64)
    my, mx = _central_grad(m)
    myy, mxx, mxy = _central_hessian(m)
    quad = my * my * myy + 2.0 * my * mx * mxy + mx * mx * mxx
    return np.abs(quad)


def self_test() -> dict[str, float]:
    """Verify the scaling-power law: kappa_n ~ c^1, raw StEik ~ c^3, ratio ~ c^2 = |grad m|^2.

    This is the property that removes the self-amplification (raw is |grad m|^2 larger than
    kappa_n). Returns measured ratios; asserts the derived powers the symposium must reproduce.
    """
    rng = np.random.default_rng(318)
    h, w = 48, 64
    base = rng.standard_normal((h, w))
    # smooth it a touch so the discrete Hessian is meaningful (a real margin is not white noise)
    base = base.cumsum(axis=0).cumsum(axis=1)
    base = (base - base.mean()) / (base.std() + _EPS)

    out: dict[str, float] = {}
    ref_kappa = normal_direction_curvature(base)
    ref_raw = raw_steik_directional_divergence(base)
    interior = np.zeros_like(base, dtype=bool)
    interior[1:-1, 1:-1] = True
    ref_kappa_mag = float(np.mean(np.abs(ref_kappa[interior])) + _EPS)
    ref_raw_mag = float(np.mean(np.abs(ref_raw[interior])) + _EPS)

    for scale in (1.0, 100.0, 2000.0):
        m = scale * base
        kappa = normal_direction_curvature(m)
        raw = raw_steik_directional_divergence(m)
        kappa_ratio = float(np.mean(np.abs(kappa[interior])) / ref_kappa_mag)  # ~ c^1
        raw_ratio = float(np.mean(np.abs(raw[interior])) / ref_raw_mag)  # ~ c^3
        # removed prefactor: raw is larger than kappa by ~ |grad m|^2 = c^2
        removed_prefactor = raw_ratio / (kappa_ratio + _EPS)  # ~ c^2
        out[f"kappa_ratio_scale_{scale:g}"] = kappa_ratio
        out[f"raw_ratio_scale_{scale:g}"] = raw_ratio
        out[f"removed_prefactor_scale_{scale:g}"] = removed_prefactor
        # kappa_n scales LINEARLY (c^1): kappa_ratio ~ scale
        assert abs(kappa_ratio - scale) / scale < 1e-6, (
            f"kappa_n must scale ~c^1; got ratio {kappa_ratio} at scale {scale}"
        )
        # raw scales CUBICALLY (c^3): raw_ratio ~ scale^3
        assert abs(raw_ratio - scale**3) / scale**3 < 1e-6, (
            f"raw StEik must scale ~c^3; got ratio {raw_ratio} at scale {scale}"
        )
        # the removed prefactor is exactly |grad m|^2 ~ c^2 (the self-amplification killed):
        assert abs(removed_prefactor - scale**2) / scale**2 < 1e-6, (
            f"normalization must remove ~c^2 = |grad m|^2; got {removed_prefactor} at scale {scale}"
        )

    return out


if __name__ == "__main__":  # pragma: no cover
    results = self_test()
    print("normalized n^T H n reference — self-test PASSED")
    for k, v in results.items():
        print(f"  {k}: {v:.3e}")
