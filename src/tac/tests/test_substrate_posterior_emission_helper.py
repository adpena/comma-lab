# SPDX-License-Identifier: MIT
"""Tests for tac.substrates._shared.posterior_emission_helper.

Per WAVE-1-POSTERIOR-EMISSION-CANONICAL-WIRE-IN charter 2026-05-26 +
audit roadmap commit ``e757bb74c`` META #1 CRITICAL finding closure.

Tests the canonical landing-time posterior emission helper that 8+ Path
3 substrates invoke to lift their L0/L1 signals into the cathedral
autopilot's 62 auto-discovered consumers via canonical posterior
surfaces.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.optimization.mlx_research_signal import (
    MLXResearchSignalError,
    append_manifest_row_to_jsonl,
)
from tac.provenance import InvalidProvenanceError
from tac.substrates._shared.posterior_emission_helper import (
    DEFAULT_MLX_RESEARCH_SIGNAL_MANIFEST_PATH,
    DEFAULT_MPS_RESEARCH_SIGNAL_MANIFEST_PATH,
    SubstrateLandingPosteriorAnchor,
    emit_substrate_landing_posterior_anchor,
    synthesize_substrate_archive_sha256,
)

# ─── Helper: produce canonical isolated tmp paths for the canonical
# posterior + canonical MLX manifest so tests don't touch live state.


def _isolated_paths(tmpdir: Path) -> tuple[Path, Path, Path]:
    return (
        tmpdir / "posterior.json",
        tmpdir / "posterior.lock",
        tmpdir / "manifest.jsonl",
    )


# ─── synthesize_substrate_archive_sha256 unit tests ────────────────────


class TestSynthesizeSubstrateArchiveSha256:
    def test_returns_64_char_lowercase_hex(self) -> None:
        sha = synthesize_substrate_archive_sha256("dreamer_v3_rssm")
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)

    def test_deterministic(self) -> None:
        a = synthesize_substrate_archive_sha256("z6_pc")
        b = synthesize_substrate_archive_sha256("z6_pc")
        assert a == b

    def test_distinct_per_substrate(self) -> None:
        a = synthesize_substrate_archive_sha256("dreamer_v3_rssm")
        b = synthesize_substrate_archive_sha256("z6_pc")
        assert a != b

    def test_distinct_per_salt(self) -> None:
        a = synthesize_substrate_archive_sha256("dreamer_v3_rssm", salt="salt_a")
        b = synthesize_substrate_archive_sha256("dreamer_v3_rssm", salt="salt_b")
        assert a != b

    def test_rejects_empty_substrate_id(self) -> None:
        with pytest.raises(ValueError, match="substrate_id"):
            synthesize_substrate_archive_sha256("")

    def test_rejects_whitespace_substrate_id(self) -> None:
        with pytest.raises(ValueError, match="substrate_id"):
            synthesize_substrate_archive_sha256("   ")


# ─── emit_substrate_landing_posterior_anchor — happy path + emission ──


class TestEmitSubstrateLandingPosteriorAnchorHappyPath:
    def test_emits_canonical_anchor_with_synthesized_sha(self, tmp_path: Path) -> None:
        post_p, lock_p, manifest_p = _isolated_paths(tmp_path)
        sha = synthesize_substrate_archive_sha256("dreamer_v3_rssm")
        anchor = emit_substrate_landing_posterior_anchor(
            substrate_id="dreamer_v3_rssm",
            archive_sha256=sha,
            archive_bytes=12345,
            source_path=str(tmp_path / "fixture.bin"),
            predicted_score=0.195,
            notes="L0 SCAFFOLD MLX landing per WAVE-1 wire-in 2026-05-26",
            posterior_path=post_p,
            posterior_lock_path=lock_p,
            manifest_path=manifest_p,
        )
        assert isinstance(anchor, SubstrateLandingPosteriorAnchor)
        assert anchor.substrate_id == "dreamer_v3_rssm"
        assert anchor.archive_sha256 == sha
        assert anchor.archive_bytes == 12345
        assert anchor.predicted_score == 0.195

    def test_posterior_refused_per_mlx_research_signal_non_promotable(self, tmp_path: Path) -> None:
        post_p, lock_p, manifest_p = _isolated_paths(tmp_path)
        anchor = emit_substrate_landing_posterior_anchor(
            substrate_id="dreamer_v3_rssm",
            archive_sha256=synthesize_substrate_archive_sha256("dreamer_v3_rssm"),
            archive_bytes=12345,
            source_path=str(tmp_path / "fixture.bin"),
            notes="L0 SCAFFOLD canonical landing anchor",
            posterior_path=post_p,
            posterior_lock_path=lock_p,
            manifest_path=manifest_p,
        )
        # Per CLAUDE.md local-substrate authority + custody validator:
        # [macOS-MLX research-signal] tag is NON_PROMOTABLE_TAGS so the
        # posterior REFUSES the anchor (advisory-grade refusal).
        assert anchor.posterior_update.accepted is False
        assert (
            "advisory" in anchor.posterior_update.refusal_reason.lower()
            or "non-authoritative" in anchor.posterior_update.refusal_reason.lower()
        )

    def test_posterior_state_bumps_refused_anchor_count(self, tmp_path: Path) -> None:
        post_p, lock_p, manifest_p = _isolated_paths(tmp_path)
        # Initial posterior: empty (zero refused)
        emit_substrate_landing_posterior_anchor(
            substrate_id="dreamer_v3_rssm",
            archive_sha256=synthesize_substrate_archive_sha256("dreamer_v3_rssm"),
            archive_bytes=12345,
            source_path=str(tmp_path / "fixture.bin"),
            notes="canonical landing anchor for refused-count test",
            posterior_path=post_p,
            posterior_lock_path=lock_p,
            manifest_path=manifest_p,
        )
        # Verify posterior recorded the refusal (refused_anchor_count > 0).
        posterior_data = json.loads(post_p.read_text())
        assert posterior_data["refused_anchor_count"] >= 1
        assert posterior_data["accepted_anchor_count"] == 0

    def test_manifest_row_emitted_to_jsonl(self, tmp_path: Path) -> None:
        post_p, lock_p, manifest_p = _isolated_paths(tmp_path)
        anchor = emit_substrate_landing_posterior_anchor(
            substrate_id="dreamer_v3_rssm",
            archive_sha256=synthesize_substrate_archive_sha256("dreamer_v3_rssm"),
            archive_bytes=12345,
            source_path=str(tmp_path / "fixture.bin"),
            notes="canonical landing anchor for manifest test",
            posterior_path=post_p,
            posterior_lock_path=lock_p,
            manifest_path=manifest_p,
        )
        # Manifest JSONL must exist with exactly one row.
        assert manifest_p.exists()
        lines = manifest_p.read_text().strip().split("\n")
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["substrate_id"] == "dreamer_v3_rssm"
        assert row["archive_sha256"] == anchor.archive_sha256
        assert row["archive_bytes"] == 12345
        # Canonical non-promotable markers (per Catalog #341):
        assert row["score_claim"] is False
        assert row["promotion_eligible"] is False
        assert row["ready_for_exact_eval_dispatch"] is False
        assert row["rank_or_kill_eligible"] is False
        assert row["promotable"] is False
        assert row["predicted_delta_adjustment"] == 0.0
        assert row["axis_tag"] == "[macOS-MLX research-signal]"
        assert row["evidence_tag"] == "[macOS-MLX research-signal]"
        assert row["evidence_grade"] == "macOS-MLX-research-signal"
        assert row["hardware_substrate"] == "macos_arm64_mlx"
        assert row["device"] == "mlx"

    def test_provenance_canonical_helper_invocation_present(self, tmp_path: Path) -> None:
        post_p, lock_p, manifest_p = _isolated_paths(tmp_path)
        anchor = emit_substrate_landing_posterior_anchor(
            substrate_id="dreamer_v3_rssm",
            archive_sha256=synthesize_substrate_archive_sha256("dreamer_v3_rssm"),
            archive_bytes=12345,
            source_path=str(tmp_path / "fixture.bin"),
            notes="canonical provenance test",
            posterior_path=post_p,
            posterior_lock_path=lock_p,
            manifest_path=manifest_p,
        )
        # Per Catalog #323 canonical Provenance:
        assert anchor.provenance.canonical_helper_invocation == (
            "tac.provenance.builders.build_provenance_for_macos_mlx_research_signal"
        )
        assert anchor.provenance.promotion_eligible is False
        assert anchor.provenance.score_claim_valid is False

    def test_extra_manifest_fields_threaded_through(self, tmp_path: Path) -> None:
        post_p, lock_p, manifest_p = _isolated_paths(tmp_path)
        emit_substrate_landing_posterior_anchor(
            substrate_id="z6_pc",
            archive_sha256=synthesize_substrate_archive_sha256("z6_pc"),
            archive_bytes=5000,
            source_path=str(tmp_path / "z6_fixture.bin"),
            notes="z6 extra fields test",
            posterior_path=post_p,
            posterior_lock_path=lock_p,
            manifest_path=manifest_p,
            extra_manifest_fields={
                "paradigm": "predictive_coding",
                "canonical_equation_id": "predictive_coding_residual_capacity_v1",
                "mlx_pytorch_decoder_parity_max_abs": 0.000011,
            },
        )
        row = json.loads(manifest_p.read_text().strip())
        assert row["paradigm"] == "predictive_coding"
        assert row["canonical_equation_id"] == "predictive_coding_residual_capacity_v1"
        assert row["mlx_pytorch_decoder_parity_max_abs"] == 0.000011

    def test_extra_manifest_fields_cannot_override_non_promotable_markers(self, tmp_path: Path) -> None:
        post_p, lock_p, manifest_p = _isolated_paths(tmp_path)
        emit_substrate_landing_posterior_anchor(
            substrate_id="dreamer_v3_rssm",
            archive_sha256=synthesize_substrate_archive_sha256("dreamer_v3_rssm"),
            archive_bytes=12345,
            source_path=str(tmp_path / "fixture.bin"),
            notes="extra fields cannot override canonical markers",
            posterior_path=post_p,
            posterior_lock_path=lock_p,
            manifest_path=manifest_p,
            extra_manifest_fields={
                "score_claim": True,  # MUST be ignored
                "promotion_eligible": True,  # MUST be ignored
                "rank_or_kill_eligible": True,  # MUST be ignored
                "ready_for_exact_eval_dispatch": True,  # MUST be ignored
                "custom_field": "kept",
            },
        )
        row = json.loads(manifest_p.read_text().strip())
        # Canonical non-promotable markers preserved
        assert row["score_claim"] is False
        assert row["promotion_eligible"] is False
        assert row["rank_or_kill_eligible"] is False
        assert row["ready_for_exact_eval_dispatch"] is False
        # Non-canonical extra field IS kept
        assert row["custom_field"] == "kept"

    def test_predicted_axis_components_threaded_through(self, tmp_path: Path) -> None:
        post_p, lock_p, manifest_p = _isolated_paths(tmp_path)
        emit_substrate_landing_posterior_anchor(
            substrate_id="z8_hpc",
            archive_sha256=synthesize_substrate_archive_sha256("z8_hpc"),
            archive_bytes=7000,
            source_path=str(tmp_path / "z8.bin"),
            predicted_d_seg=0.0010,
            predicted_d_pose=0.000025,
            notes="z8 per-axis prediction test",
            posterior_path=post_p,
            posterior_lock_path=lock_p,
            manifest_path=manifest_p,
        )
        row = json.loads(manifest_p.read_text().strip())
        assert row["predicted_d_seg"] == 0.0010
        assert row["predicted_d_pose"] == 0.000025


# ─── Validation / refusal tests ────────────────────────────────────────


class TestValidation:
    def test_rejects_empty_substrate_id(self, tmp_path: Path) -> None:
        post_p, lock_p, manifest_p = _isolated_paths(tmp_path)
        with pytest.raises(ValueError, match="substrate_id"):
            emit_substrate_landing_posterior_anchor(
                substrate_id="",
                archive_sha256="a" * 64,
                archive_bytes=1,
                source_path=str(tmp_path / "x.bin"),
                notes="canonical rationale",
                posterior_path=post_p,
                posterior_lock_path=lock_p,
                manifest_path=manifest_p,
            )

    def test_rejects_non_hex_sha256(self, tmp_path: Path) -> None:
        post_p, lock_p, manifest_p = _isolated_paths(tmp_path)
        with pytest.raises(ValueError, match="archive_sha256"):
            emit_substrate_landing_posterior_anchor(
                substrate_id="x",
                archive_sha256="not_hex_at_all_" + "g" * 49,
                archive_bytes=1,
                source_path=str(tmp_path / "x.bin"),
                notes="canonical rationale",
                posterior_path=post_p,
                posterior_lock_path=lock_p,
                manifest_path=manifest_p,
            )

    def test_rejects_short_sha256(self, tmp_path: Path) -> None:
        post_p, lock_p, manifest_p = _isolated_paths(tmp_path)
        with pytest.raises(ValueError, match="archive_sha256"):
            emit_substrate_landing_posterior_anchor(
                substrate_id="x",
                archive_sha256="abc123",
                archive_bytes=1,
                source_path=str(tmp_path / "x.bin"),
                notes="canonical rationale",
                posterior_path=post_p,
                posterior_lock_path=lock_p,
                manifest_path=manifest_p,
            )

    def test_rejects_non_str_sha256(self, tmp_path: Path) -> None:
        post_p, lock_p, manifest_p = _isolated_paths(tmp_path)
        with pytest.raises(TypeError, match="archive_sha256"):
            emit_substrate_landing_posterior_anchor(
                substrate_id="x",
                archive_sha256=12345,  # type: ignore[arg-type]
                archive_bytes=1,
                source_path=str(tmp_path / "x.bin"),
                notes="canonical rationale",
                posterior_path=post_p,
                posterior_lock_path=lock_p,
                manifest_path=manifest_p,
            )

    def test_rejects_non_positive_archive_bytes(self, tmp_path: Path) -> None:
        post_p, lock_p, manifest_p = _isolated_paths(tmp_path)
        with pytest.raises(ValueError, match="archive_bytes"):
            emit_substrate_landing_posterior_anchor(
                substrate_id="x",
                archive_sha256="a" * 64,
                archive_bytes=0,
                source_path=str(tmp_path / "x.bin"),
                notes="canonical rationale",
                posterior_path=post_p,
                posterior_lock_path=lock_p,
                manifest_path=manifest_p,
            )

    def test_rejects_negative_archive_bytes(self, tmp_path: Path) -> None:
        post_p, lock_p, manifest_p = _isolated_paths(tmp_path)
        with pytest.raises(ValueError, match="archive_bytes"):
            emit_substrate_landing_posterior_anchor(
                substrate_id="x",
                archive_sha256="a" * 64,
                archive_bytes=-1,
                source_path=str(tmp_path / "x.bin"),
                notes="canonical rationale",
                posterior_path=post_p,
                posterior_lock_path=lock_p,
                manifest_path=manifest_p,
            )

    def test_rejects_non_int_archive_bytes(self, tmp_path: Path) -> None:
        post_p, lock_p, manifest_p = _isolated_paths(tmp_path)
        with pytest.raises(TypeError, match="archive_bytes"):
            emit_substrate_landing_posterior_anchor(
                substrate_id="x",
                archive_sha256="a" * 64,
                archive_bytes=1.5,  # type: ignore[arg-type]
                source_path=str(tmp_path / "x.bin"),
                notes="canonical rationale",
                posterior_path=post_p,
                posterior_lock_path=lock_p,
                manifest_path=manifest_p,
            )

    def test_rejects_placeholder_rationale_literal(self, tmp_path: Path) -> None:
        post_p, lock_p, manifest_p = _isolated_paths(tmp_path)
        # Catalog #287 sister discipline: placeholder rationale literals
        # REJECTED so the helper's docstring example cannot self-waive.
        for placeholder in ("<rationale>", "<reason>", "<rationale_here>"):
            with pytest.raises(ValueError, match="placeholder"):
                emit_substrate_landing_posterior_anchor(
                    substrate_id="x",
                    archive_sha256=synthesize_substrate_archive_sha256("x"),
                    archive_bytes=1,
                    source_path=str(tmp_path / "x.bin"),
                    notes=placeholder,
                    posterior_path=post_p,
                    posterior_lock_path=lock_p,
                    manifest_path=manifest_p,
                )

    def test_rejects_too_short_rationale(self, tmp_path: Path) -> None:
        post_p, lock_p, manifest_p = _isolated_paths(tmp_path)
        with pytest.raises(ValueError, match="too short"):
            emit_substrate_landing_posterior_anchor(
                substrate_id="x",
                archive_sha256=synthesize_substrate_archive_sha256("x"),
                archive_bytes=1,
                source_path=str(tmp_path / "x.bin"),
                notes="ok",
                posterior_path=post_p,
                posterior_lock_path=lock_p,
                manifest_path=manifest_p,
            )

    def test_rejects_transient_tmp_source_path(self, tmp_path: Path) -> None:
        post_p, lock_p, manifest_p = _isolated_paths(tmp_path)
        # build_provenance_for_macos_mlx_research_signal refuses /tmp paths per
        # CLAUDE.md "Forbidden /tmp paths in any persisted artifact".
        with pytest.raises(InvalidProvenanceError):
            emit_substrate_landing_posterior_anchor(
                substrate_id="x",
                archive_sha256=synthesize_substrate_archive_sha256("x"),
                archive_bytes=1,
                source_path="/tmp/forbidden_transient.bin",
                notes="should refuse /tmp paths",
                posterior_path=post_p,
                posterior_lock_path=lock_p,
                manifest_path=manifest_p,
            )


# ─── SubstrateLandingPosteriorAnchor invariants ────────────────────────


class TestSubstrateLandingPosteriorAnchorInvariants:
    def _build_anchor(self, tmp_path: Path) -> SubstrateLandingPosteriorAnchor:
        post_p, lock_p, manifest_p = _isolated_paths(tmp_path)
        return emit_substrate_landing_posterior_anchor(
            substrate_id="dreamer_v3_rssm",
            archive_sha256=synthesize_substrate_archive_sha256("dreamer_v3_rssm"),
            archive_bytes=12345,
            source_path=str(tmp_path / "fixture.bin"),
            notes="canonical invariant test",
            posterior_path=post_p,
            posterior_lock_path=lock_p,
            manifest_path=manifest_p,
        )

    def test_score_claim_default_false(self, tmp_path: Path) -> None:
        anchor = self._build_anchor(tmp_path)
        assert anchor.score_claim is False

    def test_promotion_eligible_default_false(self, tmp_path: Path) -> None:
        anchor = self._build_anchor(tmp_path)
        assert anchor.promotion_eligible is False

    def test_ready_for_exact_eval_dispatch_default_false(self, tmp_path: Path) -> None:
        anchor = self._build_anchor(tmp_path)
        assert anchor.ready_for_exact_eval_dispatch is False

    def test_rank_or_kill_eligible_default_false(self, tmp_path: Path) -> None:
        anchor = self._build_anchor(tmp_path)
        assert anchor.rank_or_kill_eligible is False

    def test_non_authoritative_blockers_non_empty(self, tmp_path: Path) -> None:
        anchor = self._build_anchor(tmp_path)
        assert len(anchor.non_authoritative_blockers) >= 1
        # Canonical blocker tokens present
        joined = " ".join(anchor.non_authoritative_blockers)
        assert "macos" in joined.lower() or "mlx" in joined.lower()


# ─── Default manifest path canonical location ──────────────────────────


class TestDefaultManifestPath:
    def test_canonical_default_under_omx_state(self) -> None:
        assert ".omx/state/mlx_research_signal_manifest.jsonl" in str(DEFAULT_MLX_RESEARCH_SIGNAL_MANIFEST_PATH)

    def test_canonical_default_is_jsonl(self) -> None:
        assert DEFAULT_MLX_RESEARCH_SIGNAL_MANIFEST_PATH.suffix == ".jsonl"

    def test_legacy_mps_constant_aliases_mlx_path(self) -> None:
        assert DEFAULT_MPS_RESEARCH_SIGNAL_MANIFEST_PATH == DEFAULT_MLX_RESEARCH_SIGNAL_MANIFEST_PATH


class TestMLXResearchSignalManifestHelper:
    def test_rejects_authority_true_rows(self, tmp_path: Path) -> None:
        with pytest.raises(MLXResearchSignalError, match="cannot carry score authority"):
            append_manifest_row_to_jsonl(
                {
                    "substrate_id": "x",
                    "score_claim": True,
                },
                output_path=tmp_path / "mlx.jsonl",
            )


# ─── End-to-end: 8 substrate emissions land cleanly in isolation ───────


class TestPath3SubstrateBulkEmissions:
    """Verify all 8 Path 3 substrates can emit via the canonical helper.

    Per WAVE-1-POSTERIOR-EMISSION-CANONICAL-WIRE-IN charter: this
    simulates the audit's META #1 closure — ALL 8 substrates emit via
    one helper invocation. Tests use isolated tmp posterior + manifest
    paths so the live canonical state is not polluted.
    """

    PATH_3_SUBSTRATES: tuple[str, ...] = (
        "dreamer_v3_rssm",  # A
        "z7_mamba2_v2_fresh_substrate",  # B'
        "nscs06_v8_chroma_lut",  # C'
        "time_traveler_l5_z6",  # D
        "boost_nerv",  # E
        "z8_hierarchical_predictive_coding",  # F
        "nirvana_cascading_nerv",  # G
        "atw_v2_cooperative_receiver_v2",  # H
    )

    def test_all_8_path_3_substrates_emit_cleanly(self, tmp_path: Path) -> None:
        post_p, lock_p, manifest_p = _isolated_paths(tmp_path)
        anchors: list[SubstrateLandingPosteriorAnchor] = []
        for substrate_id in self.PATH_3_SUBSTRATES:
            sha = synthesize_substrate_archive_sha256(substrate_id)
            anchor = emit_substrate_landing_posterior_anchor(
                substrate_id=substrate_id,
                archive_sha256=sha,
                archive_bytes=10000,
                source_path=str(tmp_path / f"{substrate_id}_fixture.bin"),
                predicted_score=0.20,
                notes=f"L0 SCAFFOLD canonical landing anchor for {substrate_id}",
                posterior_path=post_p,
                posterior_lock_path=lock_p,
                manifest_path=manifest_p,
            )
            anchors.append(anchor)

        # All 8 anchors land
        assert len(anchors) == 8
        # All 8 are MLX research-signal anchors
        for a in anchors:
            assert a.evidence_tag == "[macOS-MLX research-signal]"
            assert a.hardware_substrate == "macos_arm64_mlx"
            assert a.posterior_update.accepted is False  # advisory-grade refusal

        # Manifest JSONL has exactly 8 rows
        lines = manifest_p.read_text().strip().split("\n")
        assert len(lines) == 8
        substrate_ids_in_manifest = {json.loads(line)["substrate_id"] for line in lines}
        assert substrate_ids_in_manifest == set(self.PATH_3_SUBSTRATES)

        # Posterior recorded all 8 refusals
        posterior_data = json.loads(post_p.read_text())
        assert posterior_data["refused_anchor_count"] >= 8

    def test_canonical_helper_is_idempotent_for_same_anchor(self, tmp_path: Path) -> None:
        """Per Catalog #128 + sister: emitting the same anchor twice is
        recorded both times in manifest JSONL (APPEND-ONLY) and is
        refused at the posterior surface (duplicate sha guard)."""
        post_p, lock_p, manifest_p = _isolated_paths(tmp_path)
        sha = synthesize_substrate_archive_sha256("dreamer_v3_rssm")
        for _ in range(2):
            emit_substrate_landing_posterior_anchor(
                substrate_id="dreamer_v3_rssm",
                archive_sha256=sha,
                archive_bytes=12345,
                source_path=str(tmp_path / "fixture.bin"),
                notes="canonical idempotence test",
                posterior_path=post_p,
                posterior_lock_path=lock_p,
                manifest_path=manifest_p,
            )
        # Manifest IS APPEND-ONLY: 2 rows
        lines = manifest_p.read_text().strip().split("\n")
        assert len(lines) == 2
        # Posterior: both refused (advisory-grade); refused_count >= 2
        posterior_data = json.loads(post_p.read_text())
        assert posterior_data["refused_anchor_count"] >= 2
