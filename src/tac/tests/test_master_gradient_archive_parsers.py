# SPDX-License-Identifier: MIT
"""Focused tests for ``tac.master_gradient_archive_parsers`` facade.

Per the comprehensive analytical-surfaces inventory memo §2.3 TIER-1
op-routable #1: the canonical importable namespace
``tac.master_gradient_archive_parsers.*`` exposes per-archive parsers under
a stable namespace so downstream consumers don't take a dependency on
``tools/*.py``.

Per CLAUDE.md "tac stays clean; comma-lab owns research state": this facade
delegates to the canonical extractor parser surface at
``tools/extract_master_gradient.py``; tests verify the delegation contract
+ the canonical grammar registry + end-to-end detection against live
archives.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tac.master_gradient_archive_parsers as mgap

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Live archive fixtures known to exist on disk at landing time per the
# inventory memo. These are CANONICAL frontier archives; if they ever go
# missing, sister regressions in the inventory + lane registry will surface
# the deletion (per CLAUDE.md "Apples-to-apples evidence discipline").
LIVE_A1_ARCHIVE = REPO_ROOT / "submissions" / "a1" / "archive.zip"
LIVE_PR106_R2_ARCHIVE = REPO_ROOT / "submissions" / "pr106_latent_sidecar_r2" / "archive.zip"
LIVE_PR106_FORMAT0D_ARCHIVE = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "pr106_format0d_latent_score_table_materialized_20260515_codex"
    / "sidecar_archive.zip"
)


# --------------------------------------------------------------------------- #
# Registry contract                                                            #
# --------------------------------------------------------------------------- #


class TestGrammarRegistryContract:
    def test_registry_contains_canonical_8_grammars(self):
        names = mgap.list_archive_grammar_names()
        # Per CLAUDE.md "Strict-flip atomicity rule" — pin the canonical
        # registry so a future PR adding/removing parsers requires this
        # test to be updated explicitly.
        assert "fec6_fp11_selector" in names
        assert "a1_finetuned" in names
        assert "pr101_lc_v2" in names
        assert "pr106_format0d" in names
        assert "pr106_ff_packed_hnerv" in names
        assert "hnerv_lc_v2_length_prefixed" in names
        assert "pr107_apogee_length_prefixed" in names
        assert "dp1_pretrained_driving_prior" in names

    def test_anchor_emitting_grammars_pinned(self):
        # Per CLAUDE.md "Catalog #327 raw byte authority NOT landed":
        # only grammars with a wired Jacobian projector emit anchors.
        # At landing: fec6 + A1 + PR101_lc_v2 are anchor-eligible; PR106
        # format0d later gained a primary packed-HNeRV decoder projector with
        # sidecar zero-gradient v1 semantics.
        assert mgap.is_anchor_emitting_grammar("fec6_fp11_selector") is True
        assert mgap.is_anchor_emitting_grammar("a1_finetuned") is True
        assert mgap.is_anchor_emitting_grammar("pr101_lc_v2") is True
        assert mgap.is_anchor_emitting_grammar("pr106_format0d") is True
        assert mgap.is_anchor_emitting_grammar("pr107_apogee_length_prefixed") is True
        assert mgap.is_anchor_emitting_grammar("pr106_ff_packed_hnerv") is False
        assert mgap.is_anchor_emitting_grammar("hnerv_lc_v2_length_prefixed") is False
        assert mgap.is_anchor_emitting_grammar("dp1_pretrained_driving_prior") is False

    def test_unknown_grammar_returns_false(self):
        assert mgap.is_anchor_emitting_grammar("nonexistent_grammar") is False
        assert mgap.is_anchor_emitting_grammar("") is False

    def test_all_exported_parsers_callable(self):
        for parser_name in [
            "fec6_fp11_selector_grammar_parser",
            "a1_grammar_parser",
            "pr101_lc_v2_grammar_parser",
            "pr106_format0d_grammar_parser",
            "pr106_ff_packed_grammar_parser",
            "hnerv_lc_v2_grammar_parser",
            "pr107_apogee_grammar_parser",
            "dp1_pretrained_driving_prior_grammar_parser",
            "detect_archive_grammar_and_parse",
            "list_archive_grammar_contracts",
        ]:
            assert hasattr(mgap, parser_name), f"missing: {parser_name}"
            assert callable(getattr(mgap, parser_name)), f"not callable: {parser_name}"


# --------------------------------------------------------------------------- #
# Live-archive detection regression guards                                     #
# --------------------------------------------------------------------------- #


class TestLiveArchiveDetectionRegressionGuards:
    @pytest.mark.skipif(
        not LIVE_A1_ARCHIVE.exists(),
        reason="canonical A1 archive missing from disk",
    )
    def test_a1_archive_detects_as_a1_finetuned(self):
        archive_bytes = LIVE_A1_ARCHIVE.read_bytes()
        grammar_name, layout = mgap.detect_archive_grammar_and_parse(archive_bytes)
        assert grammar_name == "a1_finetuned"
        # Per CLAUDE.md + inventory memo: A1 archive sha is 87ec7ca5...
        assert layout.archive_sha256.startswith("87ec7ca5")
        assert layout.gradient_projection_supported is True
        # Canonical A1 sections: header + decoder + latent + sidecar
        section_names = [s.name for s in layout.sections]
        assert "a1_section_header" in section_names
        assert "decoder" in section_names
        assert "latent" in section_names
        assert "sidecar" in section_names

    @pytest.mark.skipif(
        not LIVE_PR106_FORMAT0D_ARCHIVE.exists(),
        reason="canonical PR106 format0d archive missing from disk",
    )
    def test_pr106_format0d_archive_detects_correctly(self):
        archive_bytes = LIVE_PR106_FORMAT0D_ARCHIVE.read_bytes()
        grammar_name, layout = mgap.detect_archive_grammar_and_parse(archive_bytes)
        assert grammar_name == "pr106_format0d"
        # Per CLAUDE.md / inventory memo: PR106 format0d frontier sha is 9cb989cef519...
        assert layout.archive_sha256.startswith("9cb989cef519")
        assert layout.gradient_projection_supported is True

    @pytest.mark.skipif(
        not LIVE_A1_ARCHIVE.exists(),
        reason="canonical A1 archive missing from disk",
    )
    def test_a1_grammar_parser_directly(self):
        archive_bytes = LIVE_A1_ARCHIVE.read_bytes()
        layout = mgap.a1_grammar_parser(archive_bytes)
        assert layout.grammar_name == "a1_finetuned"
        assert layout.gradient_projection_supported is True

    @pytest.mark.skipif(
        not LIVE_PR106_FORMAT0D_ARCHIVE.exists(),
        reason="canonical PR106 format0d archive missing from disk",
    )
    def test_pr106_format0d_grammar_parser_directly(self):
        archive_bytes = LIVE_PR106_FORMAT0D_ARCHIVE.read_bytes()
        layout = mgap.pr106_format0d_grammar_parser(archive_bytes)
        assert layout.grammar_name == "pr106_format0d"

    def test_list_archive_grammar_contracts_returns_canonical_schema(self):
        payload = mgap.list_archive_grammar_contracts()
        assert isinstance(payload, dict)
        assert payload["schema"] == "master_gradient_archive_grammar_registry_v1"
        # Per Catalog #327: the registry payload MUST NOT claim score authority.
        assert payload["score_claim_allowed"] is False
        assert payload["promotion_eligible"] is False
