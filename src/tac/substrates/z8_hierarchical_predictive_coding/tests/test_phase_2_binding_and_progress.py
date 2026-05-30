# SPDX-License-Identifier: MIT
"""Phase 2 binding contract + build progress test suite.

Verifies milestones M1-M3 of Z8_PHASE_2_BUILD_MILESTONES (status=landed)
by exercising their declared acceptance criteria:

- M1 binding_contract_landed: contract module imports, dataclass
  invariants enforced, Protocols are runtime_checkable with declared
  methods, build_canonical_contract_from_config produces valid contract.
- M2 build_progress_surface_landed: this module imports, validator
  passes, BuildMilestone invariants reject invalid states, predecessor
  consistency enforced, render produces output.
- M3 existing_config_satisfies_per_level_contract: the canonical
  Z8HierarchicalConfig defaults satisfy the contract.

The test file IS the structural verification that M1/M2/M3 are truly
landed (not phantom-landed per sister anti-pattern). A future failure
of any test here means the milestone tuple needs updating BEFORE the
substrate continues to Phase 2 work.
"""

from __future__ import annotations

import math

import pytest

from tac.substrates.z8_hierarchical_predictive_coding.binding_contract import (
    CONTEST_PAIR_COUNT,
    CONTEST_PAIR_RGB_SHAPE,
    CONTEST_SCORER_RESOLUTION,
    DeterministicStateUpdate,
    HierarchyBindingContract,
    LevelDimensionContract,
    ScoreAwareLevelLoss,
    WaveletPartition,
    WynerZivTopLevelCoder,
    _Z8ConfigShape,
    build_canonical_contract_from_config,
)
from tac.substrates.z8_hierarchical_predictive_coding.build_progress import (
    BuildMilestone,
    BuildMilestoneStatus,
    Z8_PHASE_2_BUILD_MILESTONES,
    get_in_progress_milestones,
    get_landed_milestones,
    get_next_actionable_milestones,
    get_pending_milestones,
    render_progress_summary,
    validate_milestone_tuple,
)


# ---------------------------------------------------------------------------
# M1 acceptance criteria: binding_contract module.
# ---------------------------------------------------------------------------


class TestM1BindingContractLanded:
    """M1 binding_contract_landed acceptance criteria."""

    def test_canonical_constants_match_sister_mlx_renderer(self) -> None:
        assert CONTEST_PAIR_COUNT == 600
        assert CONTEST_SCORER_RESOLUTION == (384, 512)
        assert CONTEST_PAIR_RGB_SHAPE == (2, 3, 384, 512)

    def test_level_dimension_contract_rejects_invalid_level_index(self) -> None:
        with pytest.raises(ValueError, match="level_index"):
            LevelDimensionContract(
                level_index=-1,
                num_categorical_groups=24,
                num_categorical_classes=256,
                deterministic_state_dim=64,
                wavelet_subband_shape=(384, 512),
            )

    def test_level_dimension_contract_rejects_invalid_groups(self) -> None:
        with pytest.raises(ValueError, match="num_categorical_groups"):
            LevelDimensionContract(
                level_index=0,
                num_categorical_groups=0,
                num_categorical_classes=256,
                deterministic_state_dim=64,
                wavelet_subband_shape=(384, 512),
            )
        with pytest.raises(ValueError, match="num_categorical_groups"):
            LevelDimensionContract(
                level_index=0,
                num_categorical_groups=257,
                num_categorical_classes=256,
                deterministic_state_dim=64,
                wavelet_subband_shape=(384, 512),
            )

    def test_level_dimension_contract_rejects_invalid_categories(self) -> None:
        with pytest.raises(ValueError, match="num_categorical_classes"):
            LevelDimensionContract(
                level_index=0,
                num_categorical_groups=24,
                num_categorical_classes=0,
                deterministic_state_dim=64,
                wavelet_subband_shape=(384, 512),
            )
        with pytest.raises(ValueError, match="num_categorical_classes"):
            LevelDimensionContract(
                level_index=0,
                num_categorical_groups=24,
                num_categorical_classes=65537,
                deterministic_state_dim=64,
                wavelet_subband_shape=(384, 512),
            )

    def test_level_dimension_contract_rejects_invalid_wavelet_shape(self) -> None:
        with pytest.raises(ValueError, match="wavelet_subband_shape"):
            LevelDimensionContract(
                level_index=0,
                num_categorical_groups=24,
                num_categorical_classes=256,
                deterministic_state_dim=64,
                wavelet_subband_shape=(0, 512),
            )

    def test_level_dimension_contract_one_hot_size(self) -> None:
        level = LevelDimensionContract(
            level_index=0,
            num_categorical_groups=24,
            num_categorical_classes=256,
            deterministic_state_dim=64,
            wavelet_subband_shape=(384, 512),
        )
        assert level.categorical_one_hot_size == 24 * 256

    def test_level_dimension_contract_index_bytes_u8_vs_u16(self) -> None:
        level_u8 = LevelDimensionContract(
            level_index=0,
            num_categorical_groups=24,
            num_categorical_classes=256,
            deterministic_state_dim=64,
            wavelet_subband_shape=(384, 512),
        )
        assert level_u8.categorical_index_bytes_per_pair == 24  # 1 byte/group

        level_u16 = LevelDimensionContract(
            level_index=0,
            num_categorical_groups=24,
            num_categorical_classes=1024,
            deterministic_state_dim=64,
            wavelet_subband_shape=(384, 512),
        )
        assert level_u16.categorical_index_bytes_per_pair == 48  # 2 bytes/group

    def test_hierarchy_binding_contract_requires_at_least_one_level(
        self,
    ) -> None:
        with pytest.raises(ValueError, match=">= 1 level"):
            HierarchyBindingContract(
                levels=(),
                wyner_ziv_top_level_side_info_shape=(28, 48, 64),
                score_aware_loss_sensitivity_map_shape=(3, 384, 512),
            )

    def test_hierarchy_binding_contract_enforces_contiguous_level_index(
        self,
    ) -> None:
        wrong = (
            LevelDimensionContract(
                level_index=0,
                num_categorical_groups=24,
                num_categorical_classes=256,
                deterministic_state_dim=64,
                wavelet_subband_shape=(384, 512),
            ),
            LevelDimensionContract(
                level_index=2,  # skip index 1 — invalid
                num_categorical_groups=16,
                num_categorical_classes=128,
                deterministic_state_dim=64,
                wavelet_subband_shape=(192, 256),
            ),
        )
        with pytest.raises(ValueError, match="contiguous"):
            HierarchyBindingContract(
                levels=wrong,
                wyner_ziv_top_level_side_info_shape=(28, 96, 128),
                score_aware_loss_sensitivity_map_shape=(3, 384, 512),
            )

    def test_deterministic_state_update_protocol_runtime_checkable(self) -> None:
        # Class that fully implements the Protocol satisfies isinstance.
        class GoodGRU:
            @property
            def state_dim(self) -> int:
                return 64

            def initial_state(self, batch_size):  # noqa: ANN001
                return [0.0] * batch_size * self.state_dim

            def step(self, prior_state, input_at_t):  # noqa: ANN001
                return prior_state

        assert isinstance(GoodGRU(), DeterministicStateUpdate)

        # Class missing methods does NOT satisfy.
        class BadGRU:
            @property
            def state_dim(self) -> int:
                return 64

        assert not isinstance(BadGRU(), DeterministicStateUpdate)

    def test_wavelet_partition_protocol_runtime_checkable(self) -> None:
        class GoodWavelet:
            def decompose_to_next_level(self, x):  # noqa: ANN001
                return (x, x)

            def recompose_from_next_level(self, approximation, detail):  # noqa: ANN001
                return approximation

        assert isinstance(GoodWavelet(), WaveletPartition)

    def test_wyner_ziv_protocol_runtime_checkable(self) -> None:
        class GoodWZ:
            @property
            def side_info_shape(self) -> tuple[int, int, int]:
                return (28, 48, 64)

            def encode(self, top_state, side_info) -> bytes:  # noqa: ANN001
                return b""

            def decode(self, payload, side_info):  # noqa: ANN001
                return None

        assert isinstance(GoodWZ(), WynerZivTopLevelCoder)

    def test_score_aware_level_loss_protocol_runtime_checkable(self) -> None:
        class GoodLoss:
            def per_level_loss(
                self,
                reconstruction,  # noqa: ANN001
                target,  # noqa: ANN001
                scorer_sensitivity_map,  # noqa: ANN001
            ):
                return 0.0

        assert isinstance(GoodLoss(), ScoreAwareLevelLoss)

    def test_build_canonical_contract_rejects_missing_attrs(self) -> None:
        class IncompleteConfig:
            num_levels = 3
            # Missing the other required attrs.

        with pytest.raises(ValueError, match="missing required attributes"):
            build_canonical_contract_from_config(IncompleteConfig())

    def test_build_canonical_contract_from_z8_config_shape(self) -> None:
        config = _Z8ConfigShape(
            num_levels=3,
            num_groups_per_level=(24, 16, 8),
            num_categories_per_level=(256, 128, 64),
            deterministic_state_dim=64,
            eval_size=(384, 512),
            ego_motion_dim=6,
        )
        contract = build_canonical_contract_from_config(config)
        assert contract.num_levels == 3
        assert contract.levels[0].wavelet_subband_shape == (384, 512)
        assert contract.levels[1].wavelet_subband_shape == (192, 256)
        assert contract.levels[2].wavelet_subband_shape == (96, 128)
        assert contract.levels[0].num_categorical_groups == 24
        assert contract.levels[2].num_categorical_classes == 64
        # Top-level Wyner-Ziv side-info defaults to top wavelet shape.
        assert contract.wyner_ziv_top_level_side_info_shape == (28, 96, 128)
        # Score-aware loss sensitivity map matches contest scorer resolution.
        assert contract.score_aware_loss_sensitivity_map_shape == (3, 384, 512)


# ---------------------------------------------------------------------------
# M2 acceptance criteria: build_progress module.
# ---------------------------------------------------------------------------


class TestM2BuildProgressSurfaceLanded:
    """M2 build_progress_surface_landed acceptance criteria."""

    def test_validate_milestone_tuple_passes_on_canonical(self) -> None:
        # Validator runs at import time too; explicit invocation here
        # confirms the canonical tuple is consistent.
        validate_milestone_tuple(Z8_PHASE_2_BUILD_MILESTONES)

    def test_build_milestone_rejects_empty_id(self) -> None:
        with pytest.raises(ValueError, match="milestone_id"):
            BuildMilestone(
                milestone_id="",
                description="x",
                acceptance_criteria=("ac1",),
                status=BuildMilestoneStatus.PENDING,
            )

    def test_build_milestone_rejects_empty_description(self) -> None:
        with pytest.raises(ValueError, match="description"):
            BuildMilestone(
                milestone_id="m1",
                description="",
                acceptance_criteria=("ac1",),
                status=BuildMilestoneStatus.PENDING,
            )

    def test_build_milestone_rejects_empty_acceptance_criteria(self) -> None:
        with pytest.raises(ValueError, match="acceptance criterion"):
            BuildMilestone(
                milestone_id="m1",
                description="x",
                acceptance_criteria=(),
                status=BuildMilestoneStatus.PENDING,
            )

    def test_build_milestone_landed_requires_landed_at_utc(self) -> None:
        with pytest.raises(ValueError, match="landed_at_utc"):
            BuildMilestone(
                milestone_id="m1",
                description="x",
                acceptance_criteria=("ac1",),
                status=BuildMilestoneStatus.LANDED,
            )

    def test_build_milestone_non_landed_rejects_sha(self) -> None:
        with pytest.raises(ValueError, match="status=pending"):
            BuildMilestone(
                milestone_id="m1",
                description="x",
                acceptance_criteria=("ac1",),
                status=BuildMilestoneStatus.PENDING,
                landed_commit_sha="abc123",
            )

    def test_build_milestone_non_landed_rejects_landed_at_utc(self) -> None:
        with pytest.raises(ValueError, match="status=in_progress"):
            BuildMilestone(
                milestone_id="m1",
                description="x",
                acceptance_criteria=("ac1",),
                status=BuildMilestoneStatus.IN_PROGRESS,
                landed_at_utc="2026-05-29T00:00:00Z",
            )

    def test_validate_milestone_tuple_rejects_duplicate_ids(self) -> None:
        duplicates = (
            BuildMilestone(
                milestone_id="m1",
                description="x",
                acceptance_criteria=("ac1",),
                status=BuildMilestoneStatus.PENDING,
            ),
            BuildMilestone(
                milestone_id="m1",
                description="y",
                acceptance_criteria=("ac1",),
                status=BuildMilestoneStatus.PENDING,
            ),
        )
        with pytest.raises(ValueError, match="duplicate milestone_id"):
            validate_milestone_tuple(duplicates)

    def test_validate_milestone_tuple_rejects_unknown_predecessor(self) -> None:
        bad = (
            BuildMilestone(
                milestone_id="m1",
                description="x",
                acceptance_criteria=("ac1",),
                status=BuildMilestoneStatus.PENDING,
                predecessor_milestone_ids=("nonexistent",),
            ),
        )
        with pytest.raises(ValueError, match="unknown predecessor"):
            validate_milestone_tuple(bad)

    def test_validate_milestone_tuple_rejects_in_progress_with_pending_pred(
        self,
    ) -> None:
        bad = (
            BuildMilestone(
                milestone_id="m1",
                description="x",
                acceptance_criteria=("ac1",),
                status=BuildMilestoneStatus.PENDING,
            ),
            BuildMilestone(
                milestone_id="m2",
                description="y",
                acceptance_criteria=("ac1",),
                status=BuildMilestoneStatus.IN_PROGRESS,
                predecessor_milestone_ids=("m1",),
            ),
        )
        with pytest.raises(ValueError, match="predecessors must be landed"):
            validate_milestone_tuple(bad)

    def test_get_landed_milestones_returns_only_landed(self) -> None:
        landed = get_landed_milestones(Z8_PHASE_2_BUILD_MILESTONES)
        assert len(landed) >= 1
        for m in landed:
            assert m.status == BuildMilestoneStatus.LANDED

    def test_get_pending_milestones_returns_only_pending(self) -> None:
        pending = get_pending_milestones(Z8_PHASE_2_BUILD_MILESTONES)
        assert len(pending) >= 1
        for m in pending:
            assert m.status == BuildMilestoneStatus.PENDING

    def test_get_in_progress_milestones_returns_only_in_progress(
        self,
    ) -> None:
        in_progress = get_in_progress_milestones(Z8_PHASE_2_BUILD_MILESTONES)
        for m in in_progress:
            assert m.status == BuildMilestoneStatus.IN_PROGRESS

    def test_get_next_actionable_milestones_respects_predecessors(self) -> None:
        actionable = get_next_actionable_milestones(
            Z8_PHASE_2_BUILD_MILESTONES
        )
        by_id = {m.milestone_id: m for m in Z8_PHASE_2_BUILD_MILESTONES}
        for m in actionable:
            assert m.status == BuildMilestoneStatus.PENDING
            for pred_id in m.predecessor_milestone_ids:
                assert (
                    by_id[pred_id].status == BuildMilestoneStatus.LANDED
                )

    def test_render_progress_summary_produces_markdown_table(self) -> None:
        summary = render_progress_summary()
        assert "| Status |" in summary
        assert "| ID |" in summary
        # Every milestone appears in the rendered table.
        for m in Z8_PHASE_2_BUILD_MILESTONES:
            assert m.milestone_id in summary


# ---------------------------------------------------------------------------
# M3 acceptance criteria: existing_config_satisfies_per_level_contract.
# ---------------------------------------------------------------------------


class TestM3ExistingConfigSatisfiesPerLevelContract:
    """M3 acceptance criteria: real Z8HierarchicalConfig satisfies contract."""

    def test_real_z8_config_produces_valid_contract(self) -> None:
        # Import only inside the test so MLX-unavailable environments still
        # exercise M1 + M2; M3 requires the actual config dataclass which
        # imports cleanly even without MLX runtime (MLX import inside
        # _require_mlx is deferred to model construction).
        from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
            Z8HierarchicalConfig,
        )

        config = Z8HierarchicalConfig()
        contract = build_canonical_contract_from_config(config)
        assert contract.num_levels == config.num_levels == 3

    def test_real_z8_config_categorical_byte_total_matches_sister(self) -> None:
        from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
            Z8HierarchicalConfig,
        )

        config = Z8HierarchicalConfig()
        contract = build_canonical_contract_from_config(config)
        # Sister property on Z8HierarchicalConfig computes the same thing.
        assert (
            contract.total_categorical_index_bytes_per_pair
            == config.total_latent_packing_bytes_per_pair
        )

    def test_real_z8_config_total_categorical_bits_matches_log2_math(
        self,
    ) -> None:
        from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
            Z8HierarchicalConfig,
        )

        config = Z8HierarchicalConfig()
        expected = sum(
            g * math.log2(k)
            for g, k in zip(
                config.num_groups_per_level,
                config.num_categories_per_level,
            )
        )
        assert config.total_categorical_bits_per_sample == pytest.approx(
            expected
        )

    def test_real_z8_config_wavelet_shapes_halve_correctly(self) -> None:
        from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
            Z8HierarchicalConfig,
        )

        config = Z8HierarchicalConfig()
        contract = build_canonical_contract_from_config(config)
        base_h, base_w = config.eval_size
        for i, level in enumerate(contract.levels):
            assert level.wavelet_subband_shape == (
                max(1, base_h // (2**i)),
                max(1, base_w // (2**i)),
            )
