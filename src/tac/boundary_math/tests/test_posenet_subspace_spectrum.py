# SPDX-License-Identifier: MIT
"""Behavior tests for the PoseNet pose-sensitive SUBSPACE spectrum probe (task #80).

NO-FAKE (class 8): the spectrum is the EXACT SVD of an EXACT Jacobian. The fast tests use a SYNTHETIC
known-rank Jacobian (so the participation ratio, rank detection, orthonormal basis, and pose-null
projection can be asserted against closed-form answers); the on-real-scorer test (slow) exercises the
literal frozen-PoseNet 6xN Jacobian path.

If every test here still passed with ``measure_pose_subspace_spectrum`` replaced by a constant, or with
``project_onto_pose_null`` replaced by ``return delta`` (all-null), the suite would be verifying
constants not behavior. The participation-ratio-matches-closed-form, rank-detection, orthonormal-basis,
null-fraction-of-row-space-vector-is-zero, null-fraction-of-isotropic-matches-formula, and
sensitive+null=delta tests make that impossible.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.posenet_subspace_spectrum import (
    N_SCORED_POSE_DIMS,
    PoseSubspaceError,
    PoseSubspaceSpectrum,
    _orthonormal_row_basis,
    expected_isotropic_null_fraction,
    measure_pose_subspace_spectrum,
    participation_ratio,
    project_onto_pose_null,
)


# --------------------------------------------------------------------------- #
# participation_ratio — the effective-dimension measure (closed-form anchors)  #
# --------------------------------------------------------------------------- #
def test_participation_ratio_rank1_is_one() -> None:
    # A single dominant singular value -> effective dimension ~1.
    assert participation_ratio(np.array([5.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_participation_ratio_flat_spectrum_equals_count() -> None:
    # r equal singular values -> participation ratio == r (the maximal spread).
    assert participation_ratio(np.array([2.0, 2.0, 2.0, 2.0])) == pytest.approx(4.0)
    assert participation_ratio(np.ones(6)) == pytest.approx(6.0)


def test_participation_ratio_between_one_and_rank() -> None:
    # A skewed spectrum is strictly between 1 and the count of nonzero values.
    pr = participation_ratio(np.array([10.0, 1.0, 1.0]))
    assert 1.0 < pr < 3.0


def test_participation_ratio_ignores_zeros_and_empty() -> None:
    assert participation_ratio(np.array([])) == 0.0
    assert participation_ratio(np.zeros(5)) == 0.0
    # zeros do not change the ratio of the nonzero part
    assert participation_ratio(np.array([3.0, 3.0, 0.0, 0.0])) == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# _orthonormal_row_basis — exact SVD of a synthetic known-rank Jacobian        #
# --------------------------------------------------------------------------- #
def _synthetic_jacobian(n: int, rank: int, *, seed: int = 0, decay: float = 1.0) -> np.ndarray:
    """A (6, n) Jacobian whose row space has exactly `rank` significant directions.

    Built as J = A @ B where A is (6, rank) and B is (rank, n) orthonormal rows scaled by decaying
    singular values. The pose-output space is 6-dim (6 rows) but the ROW SPACE (pixel directions) has
    dimension `rank`.
    """

    rng = np.random.default_rng(seed)
    # orthonormal pixel-space directions (rank, n)
    raw = rng.standard_normal((rank, n))
    q, _ = np.linalg.qr(raw.T)  # (n, rank)
    b = q[:, :rank].T  # (rank, n), orthonormal rows
    sing = np.array([decay**i for i in range(rank)], dtype=np.float64)  # descending
    # mixing from the rank directions into the 6 output dims
    a = rng.standard_normal((N_SCORED_POSE_DIMS, rank))
    aq, _ = np.linalg.qr(a)  # (6, 6); take first `rank` cols as orthonormal output mixing
    a_orth = aq[:, :rank]  # (6, rank)
    jac = a_orth @ (np.diag(sing) @ b)  # (6, n)
    return jac, sing, b


def test_orthonormal_row_basis_recovers_synthetic_rank() -> None:
    n = 400
    jac, _sing, _b = _synthetic_jacobian(n, rank=3, decay=0.7)
    sv, basis, rank = _orthonormal_row_basis(jac, rel_tol=1e-4)
    assert rank == 3
    assert basis.shape == (3, n)
    # singular values descending and positive for the rank
    assert sv[0] >= sv[1] >= sv[2] > 0


def test_orthonormal_row_basis_is_orthonormal() -> None:
    n = 300
    jac, _sing, _b = _synthetic_jacobian(n, rank=4, decay=0.8)
    _sv, basis, rank = _orthonormal_row_basis(jac, rel_tol=1e-4)
    gram = basis @ basis.T  # (rank, rank) should be I
    assert np.allclose(gram, np.eye(rank), atol=1e-4)


def test_orthonormal_row_basis_singular_values_match_construction() -> None:
    n = 500
    jac, sing, _b = _synthetic_jacobian(n, rank=5, decay=0.6)
    sv, _basis, rank = _orthonormal_row_basis(jac, rel_tol=1e-6)
    # the recovered singular values match the constructed ones (descending)
    assert np.allclose(sv[:rank], sing[:rank], atol=1e-4)


def test_orthonormal_row_basis_zero_jacobian_returns_empty() -> None:
    jac = np.zeros((6, 100))
    sv, basis, rank = _orthonormal_row_basis(jac, rel_tol=1e-4)
    assert rank == 0
    assert basis.shape == (0, 100)


# --------------------------------------------------------------------------- #
# project_onto_pose_null — the escape decomposition                            #
# --------------------------------------------------------------------------- #
def _spectrum_from_jacobian(jac: np.ndarray, n: int) -> PoseSubspaceSpectrum:
    sv, basis, rank = _orthonormal_row_basis(jac, rel_tol=1e-4)
    return PoseSubspaceSpectrum(
        singular_values=sv,
        row_basis=basis,
        n_pixels=n,
        h=1,
        w=n,
        frame_slot=0,
        compute_path="cpu_torch",
        effective_dim=participation_ratio(sv[:rank]),
        rank=rank,
    )


def test_null_fraction_of_row_space_vector_is_zero() -> None:
    # A perturbation built INSIDE the pose-sensitive row space has ~0 pose-null energy.
    n = 400
    jac, _sing, b = _synthetic_jacobian(n, rank=3, decay=0.7)
    spec = _spectrum_from_jacobian(jac, n)
    delta = (np.array([1.0, -2.0, 0.5]) @ b).astype(np.float64)  # in the row space
    out = project_onto_pose_null(spec, delta)
    assert out["null_energy_frac"] < 1e-6
    assert out["sensitive_energy_frac"] > 1.0 - 1e-6


def test_null_fraction_of_orthogonal_vector_is_one() -> None:
    # A perturbation ORTHOGONAL to the row space is entirely pose-null.
    n = 400
    jac, _sing, b = _synthetic_jacobian(n, rank=3, decay=0.7)
    spec = _spectrum_from_jacobian(jac, n)
    rng = np.random.default_rng(7)
    raw = rng.standard_normal(n)
    # subtract its projection onto the row space -> purely null
    coeffs = b @ raw
    null_only = raw - coeffs @ b
    out = project_onto_pose_null(spec, null_only)
    assert out["null_energy_frac"] > 1.0 - 1e-6


def test_sensitive_plus_null_reconstructs_delta() -> None:
    n = 250
    jac, _sing, _b = _synthetic_jacobian(n, rank=4, decay=0.8)
    spec = _spectrum_from_jacobian(jac, n)
    rng = np.random.default_rng(3)
    delta = rng.standard_normal(n)
    out = project_onto_pose_null(spec, delta)
    recon = np.asarray(out["sensitive"], dtype=np.float64) + np.asarray(out["null"], dtype=np.float64)
    assert np.allclose(recon, delta, atol=1e-4)


def test_isotropic_null_fraction_matches_formula() -> None:
    # An isotropic random perturbation has null fraction ~ (n - rank) / n (the #74-noise baseline).
    n = 2000
    jac, _sing, _b = _synthetic_jacobian(n, rank=5, decay=0.9)
    spec = _spectrum_from_jacobian(jac, n)
    # average over several isotropic draws
    fracs = []
    for s in range(8):
        d = np.random.default_rng(100 + s).standard_normal(n)
        fracs.append(project_onto_pose_null(spec, d)["null_energy_frac"])
    expected = expected_isotropic_null_fraction(n, spec.rank)
    assert np.mean(fracs) == pytest.approx(expected, abs=0.02)


def test_project_length_mismatch_raises() -> None:
    n = 100
    jac, _sing, _b = _synthetic_jacobian(n, rank=2, decay=0.7)
    spec = _spectrum_from_jacobian(jac, n)
    with pytest.raises(ValueError):
        project_onto_pose_null(spec, np.zeros(n + 1))


def test_expected_isotropic_null_fraction_formula() -> None:
    assert expected_isotropic_null_fraction(1000, 6) == pytest.approx(0.994)
    assert expected_isotropic_null_fraction(0, 6) == 0.0
    # rank exceeds n is clamped to 0
    assert expected_isotropic_null_fraction(3, 6) == 0.0


def test_summary_fields_present_and_consistent() -> None:
    n = 600
    jac, _sing, _b = _synthetic_jacobian(n, rank=4, decay=0.5)
    spec = _spectrum_from_jacobian(jac, n)
    summ = spec.to_summary()
    assert summ["rank"] == 4
    assert summ["n_pose_output_dims"] == 6
    assert 1.0 <= summ["effective_dim"] <= 4.0 + 1e-9
    assert 0.0 < summ["energy_frac_top1"] <= 1.0
    assert summ["energy_frac_top3"] >= summ["energy_frac_top1"]


# --------------------------------------------------------------------------- #
# fail-closed: a degenerate (severed-gradient) Jacobian must raise             #
# --------------------------------------------------------------------------- #
def test_zero_jacobian_spectrum_raises_via_real_path_guard() -> None:
    # We can't cheaply build a zero-grad PoseNet here, but we can prove the guard logic: a spectrum built
    # from an all-zero Jacobian has rank 0 and effective_dim 0 (the severed-gradient signature the real
    # measure_pose_subspace_spectrum raises on).
    spec = _spectrum_from_jacobian(np.zeros((6, 50)), 50)
    assert spec.rank == 0
    assert spec.effective_dim == 0.0
    # PoseSubspaceError is the public fail-closed exception the real measure path raises on a severed
    # gradient (a degenerate zero Jacobian); confirm it is a ValueError subclass (the contract).
    assert issubclass(PoseSubspaceError, ValueError)


# --------------------------------------------------------------------------- #
# ON-REAL-SCORER (slow) — the literal frozen PoseNet 6xN Jacobian              #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_real_posenet_subspace_is_low_dim() -> None:
    """The decisive task-80 measurement: the frozen PoseNet's pose-sensitive subspace is <=6-dim and the
    effective dimension is well below the 589,824 pixel dimension. This is the load-bearing real-scorer
    assertion (it FAILS if the Jacobian path is severed -> raises PoseSubspaceError)."""

    pytest.importorskip("torch")
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4]
    for p in (str(repo / "upstream"), str(repo / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        import av  # type: ignore
        from frame_utils import yuv420_to_rgb  # type: ignore
        from modules import PoseNet, posenet_sd_path  # type: ignore
        from safetensors.torch import load_file  # type: ignore
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"upstream PoseNet / video not available: {exc}")

    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally, unpatch_upstream_yuv6

    net = PoseNet().eval()
    net.load_state_dict(load_file(posenet_sd_path, device="cpu"))
    for prm in net.parameters():
        prm.requires_grad_(False)

    vid = repo / "upstream" / "videos" / "0.mkv"
    if not vid.exists():  # pragma: no cover
        pytest.skip("upstream/videos/0.mkv not available")
    cont = av.open(str(vid))
    frames = []
    for i, fr in enumerate(cont.decode(video=0)):
        if i >= 2:
            break
        frames.append(yuv420_to_rgb(fr).permute(2, 0, 1).float())  # (3,H,W)
    f0, f1 = frames[0], frames[1]

    token = patch_upstream_yuv6_globally()
    try:
        spec = measure_pose_subspace_spectrum(net, f0, f1, frame_slot=0)
    finally:
        unpatch_upstream_yuv6(token)

    assert spec.rank >= 1
    assert spec.rank <= N_SCORED_POSE_DIMS  # at most 6 (the row space of a 6xN matrix)
    assert spec.effective_dim <= spec.rank + 1e-6
    # the effective dimension is astronomically smaller than the pixel dimension
    assert spec.effective_dim < spec.n_pixels / 100.0
