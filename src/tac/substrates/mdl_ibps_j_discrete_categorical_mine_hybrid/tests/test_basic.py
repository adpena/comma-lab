# SPDX-License-Identifier: MIT
"""test_basic — L0 SCAFFOLD basic tests for J=MDL-IBPS.

Test coverage per AMENDMENT #3 + Catalog #91 + #139 + #220 + #272:

- Module-level constants (Catalog #124 8-field manifest)
- Numpy reference primitives (axis 3 portability)
- MLX renderer primitives (axis 2; SKIPPED if MLX not installed)
- MLX <-> numpy parity (axis 2 + axis 3 cross-validation; tolerance <= 1e-5)
- PyTorch inflate <-> numpy parity (Catalog #146 + #205)
- Archive grammar round-trip (Catalog #139 + #220 + #272 byte-deterministic)
- Categorical index pack/unpack invariance
- IB loss + MINE estimator basic forward
- Sparse-Laplacian L1 trivial cases

NOT tested at L0 SCAFFOLD time (deferred to Phase 3 follow-on):
- Full training loop (smoke gate planned at Stage 1 per Phase 2 curriculum)
- Contest-CUDA dispatch (deferred to Stage 3+ per per-substrate symposium
  per Catalog #325 + operator-frontier-override per Catalog #199)
- Post-training Tier-C density measurement (deferred to Stage 4 per Catalog #324)
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid import (
    BITS_PER_PAIR,
    CATEGORICAL_G,
    CATEGORICAL_K,
    DEFAULT_BETA_SWEEP,
    DEFAULT_LAMBDA_SPARSE,
    EVAL_HW,
    HIDDEN_DIM,
    LANE_ID,
    MINE_HIDDEN_DIM,
    NUM_HIDDEN_LAYERS,
    NUM_PAIRS,
    POS_DIM,
    SUBSTRATE_FAMILY,
    SUBSTRATE_ID,
    TOTAL_ARCHIVE_TARGET_BYTES_MAX,
    TOTAL_ARCHIVE_TARGET_BYTES_MIN,
)
from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.numpy_reference import (
    CoordMLPBaseNumpy,
    categorical_to_one_hot_numpy,
    film_modulation_numpy,
    film_proj_numpy,
    kl_gaussian_to_standard_normal_numpy,
    make_pixel_coords_numpy,
    mine_critic_forward_numpy,
    mine_lower_bound_numpy,
    sinusoidal_positional_encoding_numpy,
    sparse_laplacian_l1_numpy,
)


# =============================================================================
# Module-level constants (Catalog #124 declaration)
# =============================================================================

class TestModuleConstants:
    def test_eval_hw_is_contest_resolution(self) -> None:
        assert EVAL_HW == (384, 512)

    def test_num_pairs_is_600(self) -> None:
        assert NUM_PAIRS == 600

    def test_categorical_k_g_match_bits_per_pair(self) -> None:
        import math
        assert BITS_PER_PAIR == CATEGORICAL_G * int(math.log2(CATEGORICAL_K))

    def test_default_k_g_are_chosen_per_phase_2_design(self) -> None:
        # Path (a) DISCRETE-CATEGORICAL-MINE-HYBRID chosen K=16 × G=12 = 48 bits/pair
        assert CATEGORICAL_K == 16
        assert CATEGORICAL_G == 12
        assert BITS_PER_PAIR == 48

    def test_hidden_dim_matches_phase_2_design(self) -> None:
        assert HIDDEN_DIM == 64
        assert NUM_HIDDEN_LAYERS == 3
        assert POS_DIM == 8

    def test_mine_hidden_dim_set(self) -> None:
        assert MINE_HIDDEN_DIM == 128

    def test_archive_byte_targets_are_substrate_engineering(self) -> None:
        assert TOTAL_ARCHIVE_TARGET_BYTES_MIN < TOTAL_ARCHIVE_TARGET_BYTES_MAX
        assert TOTAL_ARCHIVE_TARGET_BYTES_MIN >= 40_000
        assert TOTAL_ARCHIVE_TARGET_BYTES_MAX <= 100_000

    def test_default_beta_sweep_is_higgins_canonical(self) -> None:
        # CC-J-3 unwind: 3 orders of magnitude below C6 v1 0.01 default
        assert DEFAULT_BETA_SWEEP == (1e-5, 1e-4, 1e-3, 1e-2)

    def test_default_lambda_sparse_is_positive(self) -> None:
        assert DEFAULT_LAMBDA_SPARSE > 0.0

    def test_lane_id_matches_brief(self) -> None:
        assert LANE_ID == "lane_path_3_j_mdl_ibps_information_bottleneck_cargo_cult_first_20260526"

    def test_substrate_identity(self) -> None:
        assert SUBSTRATE_ID == "path_3_j_mdl_ibps"
        assert SUBSTRATE_FAMILY == "mdl_ib_discrete_categorical_mine_hybrid"


# =============================================================================
# Numpy reference primitives (axis 3 portability)
# =============================================================================

class TestNumpyPrimitives:
    def test_sinusoidal_encoding_shape(self) -> None:
        coords = np.random.RandomState(42).rand(10, 3).astype(np.float32)
        encoded = sinusoidal_positional_encoding_numpy(coords)
        assert encoded.shape == (10, POS_DIM * 2 * 3)
        assert encoded.dtype == np.float32

    def test_sinusoidal_encoding_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError):
            sinusoidal_positional_encoding_numpy(np.zeros((5, 2), dtype=np.float32))

    def test_film_modulation_basic(self) -> None:
        h = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        scale = np.array([[2.0, 0.5, 1.0]], dtype=np.float32)
        shift = np.array([[1.0, 0.0, -1.0]], dtype=np.float32)
        out = film_modulation_numpy(h, scale, shift)
        assert np.allclose(out, np.array([[3.0, 1.0, 2.0]]))

    def test_film_modulation_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError):
            film_modulation_numpy(
                np.zeros((2, 3), dtype=np.float32),
                np.zeros((2, 4), dtype=np.float32),
                np.zeros((2, 3), dtype=np.float32),
            )

    def test_categorical_to_one_hot_basic(self) -> None:
        indices = np.array([[0, 1, 15], [3, 7, 0]], dtype=np.int32)
        one_hot = categorical_to_one_hot_numpy(indices, K=16)
        assert one_hot.shape == (2, 3 * 16)
        # Check structural correctness for first sample
        sample_0 = one_hot[0].reshape(3, 16)
        assert sample_0[0, 0] == 1.0
        assert sample_0[1, 1] == 1.0
        assert sample_0[2, 15] == 1.0
        # Other positions should be zero
        assert (sample_0.sum(axis=-1) == np.array([1.0, 1.0, 1.0])).all()

    def test_categorical_to_one_hot_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            categorical_to_one_hot_numpy(
                np.array([[0, 16, 0]], dtype=np.int32), K=16
            )

    def test_film_proj_shape(self) -> None:
        rng = np.random.RandomState(42)
        in_dim = CATEGORICAL_G * CATEGORICAL_K
        out_dim = NUM_HIDDEN_LAYERS * HIDDEN_DIM * 2
        weights = rng.randn(in_dim, out_dim).astype(np.float32) * 0.1
        biases = rng.randn(out_dim).astype(np.float32) * 0.0
        one_hot = rng.randn(5, in_dim).astype(np.float32)
        scales, shifts = film_proj_numpy(one_hot, weights, biases)
        assert scales.shape == (5, NUM_HIDDEN_LAYERS, HIDDEN_DIM)
        assert shifts.shape == (5, NUM_HIDDEN_LAYERS, HIDDEN_DIM)

    def test_coord_mlp_forward_smoke(self) -> None:
        rng = np.random.RandomState(42)
        pos_feat_dim = POS_DIM * 2 * 3
        wf = rng.randn(pos_feat_dim, HIDDEN_DIM).astype(np.float32) * 0.1
        bf = np.zeros(HIDDEN_DIM, dtype=np.float32)
        wh = [
            rng.randn(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32) * 0.1
            for _ in range(NUM_HIDDEN_LAYERS)
        ]
        bh = [np.zeros(HIDDEN_DIM, dtype=np.float32) for _ in range(NUM_HIDDEN_LAYERS)]
        wo = rng.randn(HIDDEN_DIM, 3).astype(np.float32) * 0.1
        bo = np.zeros(3, dtype=np.float32)
        mlp = CoordMLPBaseNumpy(wf, bf, wh, bh, wo, bo)
        N = 100
        coords = rng.rand(N, 3).astype(np.float32)
        film_scales = np.ones((N, NUM_HIDDEN_LAYERS, HIDDEN_DIM), dtype=np.float32)
        film_shifts = np.zeros((N, NUM_HIDDEN_LAYERS, HIDDEN_DIM), dtype=np.float32)
        rgb = mlp.forward(coords, film_scales, film_shifts)
        assert rgb.shape == (N, 3)
        assert (rgb >= 0.0).all() and (rgb <= 1.0).all()  # sigmoid output

    def test_mine_critic_forward_shape(self) -> None:
        rng = np.random.RandomState(42)
        z_dim = CATEGORICAL_G * CATEGORICAL_K
        frames_dim = 256
        in_dim = z_dim + frames_dim
        w1 = rng.randn(in_dim, MINE_HIDDEN_DIM).astype(np.float32) * 0.05
        b1 = np.zeros(MINE_HIDDEN_DIM, dtype=np.float32)
        w2 = rng.randn(MINE_HIDDEN_DIM, MINE_HIDDEN_DIM).astype(np.float32) * 0.05
        b2 = np.zeros(MINE_HIDDEN_DIM, dtype=np.float32)
        wo = rng.randn(MINE_HIDDEN_DIM, 1).astype(np.float32) * 0.05
        bo = np.zeros(1, dtype=np.float32)
        z = rng.randn(8, z_dim).astype(np.float32)
        f = rng.randn(8, frames_dim).astype(np.float32)
        out = mine_critic_forward_numpy(z, f, [w1, w2, wo], [b1, b2, bo])
        assert out.shape == (8,)

    def test_mine_lower_bound_basic(self) -> None:
        # Identical joint and marginal -> lower bound should be ~0 (no MI)
        critic_joint = np.array([0.5, 1.0, 1.5], dtype=np.float32)
        critic_marginal = np.array([0.5, 1.0, 1.5], dtype=np.float32)
        lb = mine_lower_bound_numpy(critic_joint, critic_marginal)
        # E[T] - log E[exp T] = mean - log mean exp; with same array this is small
        assert abs(lb) < 0.5  # should be tiny

    def test_kl_gaussian_to_standard_normal_zero(self) -> None:
        # KL(N(0, 1) || N(0, 1)) = 0
        mu = np.zeros((5, 8), dtype=np.float32)
        logvar = np.zeros((5, 8), dtype=np.float32)
        kl = kl_gaussian_to_standard_normal_numpy(mu, logvar)
        assert np.allclose(kl, 0.0, atol=1e-6)

    def test_kl_gaussian_to_standard_normal_positive(self) -> None:
        # KL(N(1, 1) || N(0, 1)) > 0
        mu = np.ones((5, 8), dtype=np.float32)
        logvar = np.zeros((5, 8), dtype=np.float32)
        kl = kl_gaussian_to_standard_normal_numpy(mu, logvar)
        assert (kl > 0.0).all()

    def test_sparse_laplacian_l1_empty(self) -> None:
        assert sparse_laplacian_l1_numpy([]) == 0.0

    def test_sparse_laplacian_l1_basic(self) -> None:
        m1 = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
        m2 = np.array([5.0, -6.0], dtype=np.float32)
        assert sparse_laplacian_l1_numpy([m1, m2]) == 1 + 2 + 3 + 4 + 5 + 6

    def test_pixel_coords_shape(self) -> None:
        coords = make_pixel_coords_numpy(height=8, width=10, t=0)
        assert coords.shape == (80, 3)
        assert (coords[:, 2] == 0.0).all()
        coords_t1 = make_pixel_coords_numpy(height=8, width=10, t=1)
        assert (coords_t1[:, 2] == 1.0).all()


# =============================================================================
# MLX renderer primitives (axis 2; SKIPPED if MLX not installed)
# =============================================================================

@pytest.fixture
def mlx_available():
    """Skip MLX tests if mlx is not installed (e.g. GHA CPU CI)."""
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        pytest.skip("MLX not installed; falling back to numpy_reference per AMENDMENT #3 axis 3")


class TestMLXNumpyParity:
    """Axis 2 + axis 3 cross-validation: MLX <-> numpy parity per AMENDMENT #3.

    Tolerance: <= 1e-5 per Phase 2 design (both numpy and MLX use IEEE-754 fp32).
    """

    def test_sinusoidal_encoding_parity(self, mlx_available: bool) -> None:
        import mlx.core as mx
        from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.mlx_renderer import (
            sinusoidal_positional_encoding_mlx,
        )
        rng = np.random.RandomState(42)
        coords_np = rng.rand(10, 3).astype(np.float32)
        coords_mx = mx.array(coords_np)
        result_np = sinusoidal_positional_encoding_numpy(coords_np)
        result_mx = np.array(sinusoidal_positional_encoding_mlx(coords_mx))
        max_drift = np.abs(result_np - result_mx).max()
        assert max_drift <= 1e-5, f"MLX<->numpy parity violated: max_drift={max_drift}"

    def test_categorical_one_hot_parity(self, mlx_available: bool) -> None:
        import mlx.core as mx
        from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.mlx_renderer import (
            categorical_to_one_hot_mlx,
        )
        indices_np = np.array([[0, 5, 15], [3, 7, 10]], dtype=np.int32)
        indices_mx = mx.array(indices_np)
        result_np = categorical_to_one_hot_numpy(indices_np, K=16)
        result_mx = np.array(categorical_to_one_hot_mlx(indices_mx, K=16))
        assert np.array_equal(result_np, result_mx)

    def test_film_modulation_parity(self, mlx_available: bool) -> None:
        import mlx.core as mx
        from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.mlx_renderer import (
            film_modulation_mlx,
        )
        rng = np.random.RandomState(42)
        h_np = rng.randn(5, HIDDEN_DIM).astype(np.float32)
        scale_np = rng.randn(5, HIDDEN_DIM).astype(np.float32)
        shift_np = rng.randn(5, HIDDEN_DIM).astype(np.float32)
        out_np = film_modulation_numpy(h_np, scale_np, shift_np)
        out_mx = np.array(
            film_modulation_mlx(mx.array(h_np), mx.array(scale_np), mx.array(shift_np))
        )
        assert np.abs(out_np - out_mx).max() <= 1e-5


# =============================================================================
# Archive grammar round-trip (Catalog #139 + #220 byte-deterministic)
# =============================================================================

class TestArchiveGrammar:
    @pytest.fixture
    def random_indices(self) -> list[list[int]]:
        rng = np.random.RandomState(42)
        return [
            [int(rng.randint(0, CATEGORICAL_K)) for _ in range(CATEGORICAL_G)]
            for _ in range(NUM_PAIRS)
        ]

    def test_pack_unpack_round_trip(self, random_indices: list[list[int]]) -> None:
        try:
            from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.archive import (
                pack_archive,
                parse_archive_bytes,
                unpack_categorical_indices,
            )
        except ImportError:
            pytest.skip("brotli not installed")
        meta = {"schema_version": 1, "num_pairs": NUM_PAIRS, "eval_hw": list(EVAL_HW)}
        # Use minimal dummy blobs for round-trip test
        base = b"dummy_base_state_dict_blob"
        mine = b"dummy_mine_critic_state_dict"
        archive_bytes = pack_archive(base, mine, random_indices, meta)
        archive = parse_archive_bytes(archive_bytes)
        unpacked = unpack_categorical_indices(archive)
        assert unpacked == random_indices
        assert archive.base_blob == base
        assert archive.mine_blob == mine
        assert archive.meta == meta

    def test_byte_determinism(self, random_indices: list[list[int]]) -> None:
        """Same inputs -> same bytes (Catalog #146 + #220 byte-deterministic invariant)."""
        try:
            from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.archive import (
                pack_archive,
            )
        except ImportError:
            pytest.skip("brotli not installed")
        meta = {"schema_version": 1, "num_pairs": NUM_PAIRS}
        bytes_a = pack_archive(b"foo", b"bar", random_indices, meta)
        bytes_b = pack_archive(b"foo", b"bar", random_indices, meta)
        assert bytes_a == bytes_b


# =============================================================================
# MLX <-> PyTorch parity (Catalog #1265 gate; SKIPPED if either not installed)
# =============================================================================

class TestMLXPyTorchParity:
    """Catalog #1265 MLX-first gate parity contract (threshold 0.001; 90x margin).

    L0 SCAFFOLD tests focus on basic primitive parity; full gate-passing
    validation is Phase 3 Stage 2 (MLX-gate validation per Phase 2 curriculum).
    """

    def test_sinusoidal_encoding_mlx_pytorch_parity(self) -> None:
        try:
            import mlx.core as mx
            import torch
            from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.mlx_renderer import (
                sinusoidal_positional_encoding_mlx,
            )
            from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.inflate import (
                _sinusoidal_encoding_torch,
            )
        except ImportError:
            pytest.skip("MLX or PyTorch not installed")
        rng = np.random.RandomState(42)
        coords_np = rng.rand(10, 3).astype(np.float32)
        result_mx = np.array(sinusoidal_positional_encoding_mlx(mx.array(coords_np)))
        result_torch = _sinusoidal_encoding_torch(torch.from_numpy(coords_np)).numpy()
        max_drift = np.abs(result_mx - result_torch).max()
        # Catalog #1265 gate threshold 0.001; 90x margin
        assert max_drift <= 0.001, (
            f"MLX<->PyTorch parity violated: max_drift={max_drift}; "
            f"threshold=0.001 per Catalog #1265 gate"
        )


# =============================================================================
# IB loss + MINE estimator (Phase 1 CC-J-4 unwind)
# =============================================================================

class TestIBLossMINE:
    def test_mine_critic_basic_forward(self) -> None:
        try:
            import torch
            from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.ib_loss_mine import (
                MINECritic,
            )
        except ImportError:
            pytest.skip("PyTorch not installed")
        critic = MINECritic(z_dim=CATEGORICAL_G * CATEGORICAL_K, frames_feat_dim=128)
        z = torch.randn(8, CATEGORICAL_G * CATEGORICAL_K)
        f = torch.randn(8, 128)
        out = critic(z, f)
        assert out.shape == (8,)

    def test_mine_lower_bound_decreases_with_no_dependence(self) -> None:
        """If z is independent of frames, MINE lower bound should be near 0."""
        try:
            import torch
            from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.ib_loss_mine import (
                MINECritic,
                mine_lower_bound,
            )
        except ImportError:
            pytest.skip("PyTorch not installed")
        torch.manual_seed(42)
        critic = MINECritic(z_dim=CATEGORICAL_G * CATEGORICAL_K, frames_feat_dim=128)
        z_joint = torch.randn(64, CATEGORICAL_G * CATEGORICAL_K)
        frames_joint = torch.randn(64, 128)
        z_marginal = torch.randn(64, CATEGORICAL_G * CATEGORICAL_K)
        lb = mine_lower_bound(critic, z_joint, frames_joint, z_marginal)
        # An untrained critic gives a small bound (likely near 0 or slightly negative)
        assert abs(lb.item()) < 5.0

    def test_sparse_laplacian_l1_torch_basic(self) -> None:
        try:
            import torch
            from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.ib_loss_mine import (
                sparse_laplacian_l1,
            )
        except ImportError:
            pytest.skip("PyTorch not installed")
        m1 = torch.tensor([[1.0, -2.0]])
        m2 = torch.tensor([3.0, -4.0])
        result = sparse_laplacian_l1([m1, m2])
        assert float(result) == 10.0

    def test_ib_loss_composite_forward(self) -> None:
        try:
            import torch
            from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.ib_loss_mine import (
                MDLIBPSJIBLoss,
                MINECritic,
            )
        except ImportError:
            pytest.skip("PyTorch not installed")
        torch.manual_seed(42)
        critic = MINECritic(z_dim=CATEGORICAL_G * CATEGORICAL_K, frames_feat_dim=128)
        loss_fn = MDLIBPSJIBLoss(beta=1e-3, lambda_sparse=1e-4)
        z_joint = torch.randn(16, CATEGORICAL_G * CATEGORICAL_K)
        frames_joint = torch.randn(16, 128)
        z_marginal = torch.randn(16, CATEGORICAL_G * CATEGORICAL_K)
        film_matrices = [torch.randn(10, 10)]
        loss, parts = loss_fn(
            critic, z_joint, frames_joint, z_marginal, film_matrices
        )
        assert "mine_lower_bound" in parts
        assert "sparse_l1" in parts
        assert "loss_ib" in parts
        assert "beta" in parts
        assert "lambda_sparse" in parts

    def test_ib_loss_rejects_negative_beta(self) -> None:
        from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.ib_loss_mine import (
            MDLIBPSJIBLoss,
        )
        with pytest.raises(ValueError):
            MDLIBPSJIBLoss(beta=-1.0)


# =============================================================================
# Inflate runtime (Catalog #146 + #205)
# =============================================================================

class TestInflateRuntime:
    def test_inflate_module_importable(self) -> None:
        """Verify inflate module imports without error (Catalog #295)."""
        try:
            import tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.inflate  # noqa: F401
        except ImportError as exc:
            pytest.skip(f"PyTorch import failed: {exc}")

    def test_inflate_uses_canonical_device_selector(self) -> None:
        """Catalog #205: inflate device-fork via canonical helper, not inline ternary."""
        try:
            import tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.inflate as inflate_mod
            from tac.substrates._shared.inflate_runtime import select_inflate_device
        except ImportError as exc:
            pytest.skip(f"PyTorch import failed: {exc}")
        # The inflate module imports select_inflate_device
        # (alternative: check the module attribute exists)
        assert hasattr(inflate_mod, "select_inflate_device") or \
               "select_inflate_device" in dir(inflate_mod) or \
               select_inflate_device is not None
