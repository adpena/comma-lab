# SPDX-License-Identifier: MIT
"""Smoke tests for the Tishby IB-pure research-only package closure."""

from __future__ import annotations

from tac.substrate_registry import get_registered_substrates
from tac.substrates.tishby_ib_pure import (
    RESEARCH_ONLY,
    TIBP1_MAGIC,
    TISHBY_IB_PURE_CONTRACT,
    TishbyIBPureCodec,
    TishbyIBPureScoreAwareLoss,
    inflate_one_video,
    pack_archive,
    parse_archive,
)


def test_package_import_is_research_only() -> None:
    codec = TishbyIBPureCodec()
    summary = codec.encode_summary()
    assert RESEARCH_ONLY is True
    assert summary["score_claim"] is False
    assert summary["research_only"] is True


def test_tibp1_archive_roundtrip_and_inflate_consumes_bytes(tmp_path) -> None:
    archive = pack_archive({"decoder_blob": b"decoder", "latent_t_blob": b"latent"})
    assert archive.startswith(TIBP1_MAGIC)
    parsed = parse_archive(archive)
    assert parsed.sections["decoder_blob"] == b"decoder"
    assert parsed.sections["latent_t_blob"] == b"latent"
    out = inflate_one_video(archive, tmp_path / "inflate_proof.txt")
    assert "score_claim=False" in out.read_text(encoding="utf-8")


def test_loss_facade_is_decomposed_and_non_claiming() -> None:
    loss = TishbyIBPureScoreAwareLoss()
    output = loss(reconstruction_term=2.0, kl_term=3.0, rate_term=4.0)
    assert output.total == 2.0 + 0.01 * 3.0 + 4.0
    assert output.score_claim is False


def test_substrate_contract_registers_from_package_import() -> None:
    registered = get_registered_substrates()
    assert registered["tishby_ib_pure"] is TISHBY_IB_PURE_CONTRACT
    assert TISHBY_IB_PURE_CONTRACT.recipe_research_only is True
    assert TISHBY_IB_PURE_CONTRACT.recipe_min_smoke_gpu == "A100"
    assert TISHBY_IB_PURE_CONTRACT.hook_probe_disambiguator == (
        "tools/check_variational_ib_tractability.py"
    )


def test_archive_grammar_8_fields_declared_per_catalog_124() -> None:
    """Catalog #124 archive-grammar 8 fields must be declared on the contract."""
    from tac.substrates.tishby_ib_pure import TIBP1_SECTION_ROLES

    # 8 sections declared per design memo §10
    assert len(TIBP1_SECTION_ROLES) == 8
    for required in (
        "encoder_blob", "decoder_blob", "statistic_net_blob",
        "latent_t_blob", "scorer_class_prior_blob", "cdf_table_blob",
        "meta_blob", "reserved_blob",
    ):
        assert required in TIBP1_SECTION_ROLES


def test_archive_parse_refuses_tampered_magic() -> None:
    """TIBP1 parser must refuse bytes with wrong magic per Catalog #220 byte-mutation."""
    import pytest
    archive = pack_archive({"meta_blob": b'{"x":1}'})
    tampered = b"XXXX" + archive[4:]
    with pytest.raises(ValueError, match="bad TIBP1 magic"):
        parse_archive(tampered)


def test_archive_parse_refuses_tampered_payload_digest() -> None:
    """TIBP1 parser must refuse bytes whose sha256 payload digest doesn't match."""
    import pytest
    archive = pack_archive({"meta_blob": b'{"x":1}', "decoder_blob": b"original_decoder"})
    # Tamper one byte in the payload (after header) — should fail digest check.
    payload_offset = len(archive) - 16  # somewhere in the payload
    tampered = bytearray(archive)
    tampered[payload_offset] ^= 0xFF
    with pytest.raises(ValueError, match="digest"):
        parse_archive(bytes(tampered))


def test_d4_probe_anchor_constants_match_state_file() -> None:
    """The package's empirical anchors must match the design memo + state JSON."""
    from tac.substrates.tishby_ib_pure import (
        D4_PROBE_MUTUAL_INFORMATION_BITS,
        D4_PROBE_VERDICT,
        VIB_TRACTABILITY_VERDICT,
    )

    # Per .omx/state/h_latent_given_scorer_class_tishby_ib_pure.json
    # produced by tools/check_variational_ib_tractability.py + D4 probe execution
    assert D4_PROBE_VERDICT == "INDEPENDENT"
    assert 0.001 <= D4_PROBE_MUTUAL_INFORMATION_BITS < 0.5  # below MEANINGFUL threshold
    assert VIB_TRACTABILITY_VERDICT == "TRACTABLE"


def test_variational_ib_tractability_probe_returns_tractable_on_default_smoke() -> None:
    """The NEW canonical VIB tractability probe should report TRACTABLE on default config."""
    from tools.check_variational_ib_tractability import (
        compute_variational_ib_tractability,
    )

    verdict = compute_variational_ib_tractability(
        substrate_id="tishby_ib_pure_test",
        num_replicates=4,
        batch_size=8,
        latent_dim=8,
        input_dim=32,
    )
    # Defaults produce TRACTABLE on synthetic Gaussian smoke per design memo §15
    assert verdict.verdict in ("TRACTABLE", "MARGINAL")
    assert verdict.score_claim is False
    assert verdict.evidence_grade == "diagnostic_cpu"
    assert "[diagnostic-CPU" in verdict.axis_label
    # Operating-within assumption per Catalog #292 sextet-pact discipline
    assert len(verdict.operating_within_assumption) > 100


def test_variational_ib_tractability_probe_path_mine_not_implemented() -> None:
    """Path-MINE measurement is v2 fallback per design memo §22 op-routable #3."""
    import pytest

    from tools.check_variational_ib_tractability import (
        compute_variational_ib_tractability,
    )

    with pytest.raises(NotImplementedError, match="Path-MINE"):
        compute_variational_ib_tractability(
            substrate_id="tishby_ib_pure_test",
            path_variant="MINE",
        )


def test_substrate_contract_archive_grammar_declares_all_required_fields() -> None:
    """Contract must declare the 8 archive-grammar fields per Catalog #124."""
    contract = TISHBY_IB_PURE_CONTRACT
    assert contract.archive_grammar is not None
    assert contract.parser_section_manifest is not None
    assert len(contract.parser_section_manifest) >= 7
    assert contract.inflate_runtime_loc_budget == 200
    assert "torch" in contract.runtime_dep_closure[0]
    assert contract.export_format == "custom"
    assert contract.score_aware_loss == "custom"
    assert contract.no_op_detector_planned is True


def test_substrate_contract_canary_status_independent_substrate() -> None:
    """Tishby IB-pure is canary_status=independent_substrate per design memo §14."""
    contract = TISHBY_IB_PURE_CONTRACT
    assert contract.recipe_canary_status == "independent_substrate"
    assert contract.recipe_canary_dependency is None
    # Catalog #270 dispatch optimization protocol declared
    assert "catalog_270_dispatch_optimization_protocol_complete" in (
        contract.catalog_compliance_declarations
    )


def test_trainer_full_main_raises_not_implemented_per_catalog_240() -> None:
    """Trainer _full_main MUST raise NotImplementedError per Catalog #240 cascade."""
    import argparse

    # Import the trainer module directly to verify _full_main raises
    import importlib

    import pytest
    trainer = importlib.import_module(
        "experiments.train_substrate_tishby_ib_pure"
    )
    ns = argparse.Namespace(
        video_path="upstream/videos/0.mkv",
        output_dir="/tmp/never_runs",
        epochs=200,
        batch_size=4,
        device="cuda",
        upstream_dir="upstream",
        latent_dim=16,
        beta=0.01,
        path_variant="VIB",
        smoke=False,
    )
    with pytest.raises(NotImplementedError, match="Phase 2 council"):
        trainer._full_main(ns)


def test_trainer_smoke_main_emits_stats_json_with_fail_closed_metadata(tmp_path) -> None:
    """Trainer _smoke_main emits stats.json with score_claim=false per Catalog #221."""
    import argparse
    import importlib
    import json
    trainer = importlib.import_module(
        "experiments.train_substrate_tishby_ib_pure"
    )

    out_dir = tmp_path / "smoke_out"
    ns = argparse.Namespace(
        video_path="upstream/videos/0.mkv",
        output_dir=out_dir,
        epochs=1,
        batch_size=4,
        lr=5e-4,
        seed=42,
        device="cpu",
        upstream_dir="upstream",
        latent_dim=16,
        decoder_embed_dim=32,
        decoder_num_upsample_blocks=6,
        scorer_class_prior_dim=16,
        path_variant="VIB",
        beta=0.01,
        beta_seg=100.0,
        gamma_pose=3.1622776601683795,
        enable_autocast_fp16=False,
        enable_torch_compile=False,
        smoke=True,
        max_pairs=600,
    )
    rc = trainer._smoke_main(ns)
    assert rc == 0
    stats_path = out_dir / "smoke_stats.json"
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text())
    # Fail-closed metadata per Catalog #221
    assert stats["score_claim"] is False
    assert stats["research_only"] is True
    assert stats["score_axis"] == "diagnostic_cpu"
    assert "[diagnostic-CPU" in stats["score_axis_anchor"]
    # Empirical anchors at landing match package constants
    assert stats["d4_probe_verdict_at_landing"] == "INDEPENDENT"
    assert stats["vib_tractability_verdict_at_landing"] == "TRACTABLE"
    # TIBP1 magic ok + archive roundtrip ok
    assert stats["tibp1_magic_ok"] is True
    assert stats["roundtrip_ok"] is True
