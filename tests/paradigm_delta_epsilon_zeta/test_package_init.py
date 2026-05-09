"""Unit tests for tac.paradigm_delta_epsilon_zeta package init.

Phase 1 scaffold — verifies the public API exports cleanly and the
provenance dataclass is well-formed.
"""
from __future__ import annotations

import pytest


def test_version_is_phase1_scaffold():
    from tac.paradigm_delta_epsilon_zeta import PARADIGM_DELTA_EPS_ZETA_VERSION

    assert PARADIGM_DELTA_EPS_ZETA_VERSION.startswith("0.1.")
    assert "phase1-scaffold" in PARADIGM_DELTA_EPS_ZETA_VERSION


def test_provenance_carries_no_empirical_anchor():
    from tac.paradigm_delta_epsilon_zeta import PARADIGM_DELTA_EPS_ZETA_PROVENANCE

    # Phase 1 scaffold must not claim empirical results.
    assert PARADIGM_DELTA_EPS_ZETA_PROVENANCE["empirical_anchor"] is None
    assert "predicted" in PARADIGM_DELTA_EPS_ZETA_PROVENANCE["predicted_score_band"]


def test_provenance_lists_compliance_tags():
    from tac.paradigm_delta_epsilon_zeta import PARADIGM_DELTA_EPS_ZETA_PROVENANCE

    tags = PARADIGM_DELTA_EPS_ZETA_PROVENANCE["compliance_tags"]
    assert "ema_0p997_snapshot_restore" in tags
    assert "eval_roundtrip_true" in tags
    assert "no_mps_authoritative" in tags
    assert "frozen_a1_encoder_designated" in tags
    assert "score_tag_predicted_only" in tags


def test_provenance_includes_t1_track():
    from tac.paradigm_delta_epsilon_zeta import PARADIGM_DELTA_EPS_ZETA_PROVENANCE

    assert "T1_balle_128k_endtoend" in PARADIGM_DELTA_EPS_ZETA_PROVENANCE["tracks"]


def test_provenance_links_to_blocker_memo():
    from tac.paradigm_delta_epsilon_zeta import PARADIGM_DELTA_EPS_ZETA_PROVENANCE

    refs = PARADIGM_DELTA_EPS_ZETA_PROVENANCE["council_memo_refs"]
    assert any("BLOCKED_scaffold_missing" in r for r in refs)


def test_public_api_exports_decoder_factory():
    from tac.paradigm_delta_epsilon_zeta import build_decoder_128k, Decoder128K

    decoder = build_decoder_128k()
    assert isinstance(decoder, Decoder128K)


def test_public_api_exports_balle_factory():
    from tac.paradigm_delta_epsilon_zeta import (
        build_balle_hyperprior,
        BalleHyperpriorWrapper,
    )

    wrapper = build_balle_hyperprior()
    assert isinstance(wrapper, BalleHyperpriorWrapper)


def test_public_api_exports_admm():
    from tac.paradigm_delta_epsilon_zeta import (
        JointLagrangianADMM,
        JointLagrangianADMMConfig,
    )

    coord = JointLagrangianADMM(JointLagrangianADMMConfig())
    assert coord.rho == 1.0


def test_frozen_a1_encoder_loader_exported():
    from tac.paradigm_delta_epsilon_zeta import (
        load_frozen_a1_encoder,
        FrozenA1Encoder,
        FrozenA1EncoderError,
        A1_CANONICAL_DIR_NAME,
    )

    assert A1_CANONICAL_DIR_NAME == "A1_canonical"
    # Must be importable + callable; we test functional behaviour in
    # test_frozen_a1_encoder.py.
    assert callable(load_frozen_a1_encoder)
    assert FrozenA1Encoder is not None
    assert issubclass(FrozenA1EncoderError, RuntimeError)


def test_provenance_is_serialisable():
    import json

    from tac.paradigm_delta_epsilon_zeta import PARADIGM_DELTA_EPS_ZETA_PROVENANCE

    # The provenance dict is meant to be embedded in archive manifests; it
    # must round-trip through JSON without exotic types.
    serialised = json.dumps(PARADIGM_DELTA_EPS_ZETA_PROVENANCE)
    restored = json.loads(serialised)
    assert restored == PARADIGM_DELTA_EPS_ZETA_PROVENANCE


def test_all_exports_in_public_namespace():
    """`__all__` must list every symbol the package re-exports."""
    import tac.paradigm_delta_epsilon_zeta as pkg

    for name in pkg.__all__:
        assert hasattr(pkg, name), f"missing public symbol {name!r}"
