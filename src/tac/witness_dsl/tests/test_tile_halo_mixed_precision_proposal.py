import pytest

from tac.witness_dsl.tile_halo_mixed_precision_proposal import (
    MlxMixedPrecisionTrainingProposal,
    Precision,
    ProposalState,
    TileHaloSensitivityWaterfillProposal,
)


def test_proposals_default_off_and_emit_no_flags():
    halo = TileHaloSensitivityWaterfillProposal()
    precision = MlxMixedPrecisionTrainingProposal()
    assert halo.enabled is False and halo.state is ProposalState.OFF_UNWIRED
    assert precision.enabled is False and precision.state is ProposalState.OFF_UNWIRED
    assert "argv" not in halo.to_display_dict()
    assert "argv" not in precision.to_display_dict()
    assert halo.to_display_dict()["wired"] is False


def test_enabled_requires_explicit_anchor_and_non_off_state():
    with pytest.raises(ValueError, match="measured_anchor"):
        MlxMixedPrecisionTrainingProposal(
            enabled=True, state=ProposalState.PROPOSAL_ONLY,
            forward=Precision.FP16, measured_anchor=None
        )
    with pytest.raises(ValueError, match="OFF_UNWIRED"):
        TileHaloSensitivityWaterfillProposal(enabled=True)


def test_enabled_tile_proposal_requires_derived_halo():
    with pytest.raises(ValueError, match="positive derived halo"):
        TileHaloSensitivityWaterfillProposal(
            enabled=True,
            state=ProposalState.PROPOSAL_ONLY,
            measured_anchor="receipt.json",
        )


def test_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="max_tile_fraction"):
        TileHaloSensitivityWaterfillProposal(max_tile_fraction=1.1)


def test_preregistered_go_bars_are_visible_and_n600_strict():
    halo = TileHaloSensitivityWaterfillProposal().to_display_dict()
    precision = MlxMixedPrecisionTrainingProposal().to_display_dict()
    assert halo["minimum_speedup"] == 2.0
    assert halo["exact_on_selected_tiles_required"] is True
    assert precision["minimum_speedup"] == 1.5
    assert precision["minimum_global_gradient_cosine"] == 0.99
    assert precision["minimum_pair_gradient_cosine"] == 0.99
    assert precision["required_quality_pairs"] == 600
