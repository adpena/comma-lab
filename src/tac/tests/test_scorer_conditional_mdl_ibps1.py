# SPDX-License-Identifier: MIT
"""Dedicated tests for IBPS1 wire-in in ScorerConditionalMDLEstimator.

Part B of lane ``lane_ibps1_canonical_surface_promotion_20260514`` (operator
decision #3 from ``feedback_c6_next_wave_landed_20260514.md``).

These tests verify that the canonical XRay primitive
:class:`tac.xray.mdl_scorer_conditional.ScorerConditionalMDLEstimator` (and
the underlying ``tac.analysis.scorer_conditional_mdl`` + parser-section
manifest dispatch in ``tac.analysis.hnerv_packet_sections``) recognizes
IBPS1 (C6 MDL-IBPS) archives by their ``b"IBPS"`` magic prefix and emits
the canonical 5-section breakdown (ibps1_header / encoder_blob /
decoder_blob / latent_blob / meta_blob) instead of the previous
``whole_blob`` fallback path.

The PARSER_IBPS1 dispatch + ``_parse_ibps1_sections`` helper live in
``src/tac/analysis/hnerv_packet_sections.py`` and consume the canonical
:func:`tac.substrates.c6_e4_mdl_ibps.archive.parse_ibps1_archive_bytes` +
:data:`IBPS1_SECTION_ROLES` surface — promoted in Part A of this same lane.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import pytest
import torch

from tac.substrates.c6_e4_mdl_ibps.archive import (
    IBPS1_SECTION_ROLES,
    pack_archive,
)


def _build_synthetic_ibps1_archive_zip(tmp_path: Path) -> Path:
    """Build a real IBPS1 archive.zip via pack_archive.

    Returns the path to the single-member zip suitable for the
    ScorerConditionalMDLEstimator / build_scorer_conditional_mdl_ablation
    intake.
    """
    torch.manual_seed(0)
    encoder_sd = {"layer.weight": torch.randn(8, 4, dtype=torch.float16)}
    decoder_sd = {"layer.weight": torch.randn(4, 8, dtype=torch.float16)}
    latents = torch.randn(10, 6, dtype=torch.float32)
    meta = {"beta_ib": 0.01, "decoder_channels": [16, 32]}
    archive_bytes = pack_archive(encoder_sd, decoder_sd, latents, meta)

    out = tmp_path / "ibps1_archive.zip"
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("0.bin", archive_bytes)
    return out


# --------------------------------------------------------------------------
# 1. parser_section_manifest IBPS1 dispatch
# --------------------------------------------------------------------------


def test_parser_section_manifest_dispatches_ibps1_by_magic(tmp_path: Path) -> None:
    """build_packet_section_manifest must auto-detect IBPS1 by ``b"IBPS"`` magic."""
    from tac.analysis.hnerv_packet_sections import (
        PARSER_AUTO,
        build_packet_section_manifest,
    )

    archive_path = _build_synthetic_ibps1_archive_zip(tmp_path)
    manifest = build_packet_section_manifest(
        archive_path, label="ibps1_synth", parser=PARSER_AUTO, repo_root=tmp_path
    )
    parser = manifest.get("parser")
    assert isinstance(parser, dict), parser
    assert parser.get("name") == "ibps1_mdl_ibps"


def test_parser_section_manifest_ibps1_emits_five_canonical_sections(
    tmp_path: Path,
) -> None:
    """The IBPS1 dispatch must emit all 5 canonical sections, not whole_blob."""
    from tac.analysis.hnerv_packet_sections import (
        PARSER_AUTO,
        build_packet_section_manifest,
    )

    archive_path = _build_synthetic_ibps1_archive_zip(tmp_path)
    manifest = build_packet_section_manifest(
        archive_path, label="ibps1_synth", parser=PARSER_AUTO, repo_root=tmp_path
    )
    sections = manifest.get("sections")
    assert isinstance(sections, list)
    section_names = {str(s["name"]) for s in sections}
    assert section_names == {
        "ibps1_header",
        "encoder_blob",
        "decoder_blob",
        "latent_blob",
        "meta_blob",
    }
    # Whole-blob fallback would emit a single "whole_blob" section
    assert "whole_blob" not in section_names


def test_parser_section_manifest_ibps1_section_roles_match_canonical_taxonomy(
    tmp_path: Path,
) -> None:
    """The dispatched sections must use the canonical IBPS1_SECTION_ROLES."""
    from tac.analysis.hnerv_packet_sections import (
        PARSER_AUTO,
        build_packet_section_manifest,
    )

    archive_path = _build_synthetic_ibps1_archive_zip(tmp_path)
    manifest = build_packet_section_manifest(
        archive_path, label="ibps1_synth", parser=PARSER_AUTO, repo_root=tmp_path
    )
    sections = manifest.get("sections")
    assert isinstance(sections, list)
    for sec in sections:
        name = str(sec["name"])
        role = str(sec["optimization_role"])
        assert role == IBPS1_SECTION_ROLES[name], (
            f"section {name!r} role {role!r} != canonical "
            f"{IBPS1_SECTION_ROLES[name]!r}"
        )


# --------------------------------------------------------------------------
# 2. build_scorer_conditional_mdl_ablation IBPS1 dispatch
# --------------------------------------------------------------------------


def test_build_mdl_ablation_includes_ibps1_sections(tmp_path: Path) -> None:
    """build_scorer_conditional_mdl_ablation must surface IBPS1 sections."""
    from tac.analysis.scorer_conditional_mdl import (
        ArchiveInput,
        build_scorer_conditional_mdl_ablation,
    )

    archive_path = _build_synthetic_ibps1_archive_zip(tmp_path)
    manifest = build_scorer_conditional_mdl_ablation(
        archives=[ArchiveInput(label="ibps1_synth", archive_path=archive_path)],
        repo_root=tmp_path,
    )
    archives = manifest.get("archives") or []
    assert len(archives) == 1
    entry = archives[0]
    sections = entry.get("sections") or []
    section_names = {str(s["name"]) for s in sections}
    assert section_names == set(IBPS1_SECTION_ROLES.keys())


def test_build_mdl_ablation_role_grouped_entropy_includes_ibps1(
    tmp_path: Path,
) -> None:
    """The role-grouped entropy layer must aggregate IBPS1 sections by role."""
    from tac.analysis.scorer_conditional_mdl import (
        ArchiveInput,
        build_scorer_conditional_mdl_ablation,
    )

    archive_path = _build_synthetic_ibps1_archive_zip(tmp_path)
    manifest = build_scorer_conditional_mdl_ablation(
        archives=[ArchiveInput(label="ibps1_synth", archive_path=archive_path)],
        repo_root=tmp_path,
    )
    layers = manifest.get("measurement_layers") or {}
    role_layer = layers.get("parser_role_conditioned") or {}
    role_groups = role_layer.get("groups") or []
    # IBPS1 sections occupy four role buckets: decoder_weight_stream,
    # latent_stream, control_or_metadata, and training_provenance_only.
    # Each must appear in role_groups.
    role_names = {str(g.get("group")) for g in role_groups}
    assert "decoder_weight_stream" in role_names
    assert "latent_stream" in role_names
    assert "control_or_metadata" in role_names
    assert "training_provenance_only" in role_names


def test_build_mdl_ablation_marks_ibps1_encoder_provenance_only(
    tmp_path: Path,
) -> None:
    """The encoder section must not be surfaced as score-affecting evidence."""
    from tac.analysis.scorer_conditional_mdl import (
        ArchiveInput,
        build_scorer_conditional_mdl_ablation,
    )

    archive_path = _build_synthetic_ibps1_archive_zip(tmp_path)
    manifest = build_scorer_conditional_mdl_ablation(
        archives=[ArchiveInput(label="ibps1_synth", archive_path=archive_path)],
        repo_root=tmp_path,
    )
    sections = manifest["archives"][0]["sections"]
    encoder = next(row for row in sections if row["name"] == "encoder_blob")
    decoder = next(row for row in sections if row["name"] == "decoder_blob")
    assert encoder["optimization_role"] == "training_provenance_only"
    assert encoder["runtime_effect"] == "training_provenance_only_not_score_affecting"
    assert encoder["score_affecting_at_inflate"] is False
    assert decoder["optimization_role"] == "decoder_weight_stream"
    assert decoder["score_affecting_at_inflate"] is True


# --------------------------------------------------------------------------
# 3. ScorerConditionalMDLEstimator XRay primitive end-to-end
# --------------------------------------------------------------------------


def test_xray_primitive_returns_ibps1_section_breakdown(tmp_path: Path) -> None:
    """ScorerConditionalMDLEstimator.compute() must surface IBPS1 sections."""
    from tac.xray.mdl_scorer_conditional import ScorerConditionalMDLEstimator

    archive_path = _build_synthetic_ibps1_archive_zip(tmp_path)
    est = ScorerConditionalMDLEstimator()
    result = est.compute(archive_path, label="ibps1_synth")
    val = result.primitive_value
    breakdown = val.per_section_breakdown
    section_names = {row[0] for row in breakdown}
    assert section_names == set(IBPS1_SECTION_ROLES.keys()), (
        f"XRay primitive returned sections {section_names!r}; expected "
        f"{set(IBPS1_SECTION_ROLES.keys())!r}"
    )


def test_xray_primitive_engages_wire_in_hooks(tmp_path: Path) -> None:
    """The XRay primitive result must declare wire-in hooks engaged."""
    from tac.xray.mdl_scorer_conditional import ScorerConditionalMDLEstimator

    archive_path = _build_synthetic_ibps1_archive_zip(tmp_path)
    est = ScorerConditionalMDLEstimator()
    result = est.compute(archive_path, label="ibps1_synth")
    # The primitive declares (continual_learning, probe_disambiguator,
    # cathedral_autopilot) as its canonical wire-in surface.
    engaged = set(result.wire_in_hooks_engaged)
    assert engaged, "wire_in_hooks_engaged must be nonempty"


def test_xray_primitive_mdl_density_in_unit_interval(tmp_path: Path) -> None:
    """The reported mdl_density must be a valid [0, 1] proxy."""
    from tac.xray.mdl_scorer_conditional import ScorerConditionalMDLEstimator

    archive_path = _build_synthetic_ibps1_archive_zip(tmp_path)
    est = ScorerConditionalMDLEstimator()
    result = est.compute(archive_path, label="ibps1_synth")
    val = result.primitive_value
    assert 0.0 <= val.mdl_density <= 1.0


# --------------------------------------------------------------------------
# 4. Tier-A invariance: corrupting IBPS1 magic falls through to error,
#    NOT a silent whole_blob fallback
# --------------------------------------------------------------------------


def test_corrupted_magic_does_not_silently_fall_back_to_whole_blob(
    tmp_path: Path,
) -> None:
    """If the IBPS1 magic is corrupted, the parser must fail rather than
    silently degrade to a whole_blob measurement.

    This protects the empirical anchor — a corrupted IBPS1 archive must
    NOT be silently mis-classified as some opaque single-section payload
    (which would inflate the structural-tier MDL density estimate).
    """
    from tac.analysis.hnerv_packet_sections import (
        PARSER_AUTO,
        HnervPacketSectionManifestError,
        build_packet_section_manifest,
    )

    archive_path = _build_synthetic_ibps1_archive_zip(tmp_path)
    # Corrupt the magic bytes inside the zip member
    raw = archive_path.read_bytes()
    # We need to rebuild the zip with a corrupted inner blob
    inner = zipfile.ZipFile(archive_path).read("0.bin")
    bad_inner = b"XXXX" + inner[4:]
    corrupted_zip = tmp_path / "ibps1_corrupted.zip"
    with zipfile.ZipFile(corrupted_zip, "w") as zf:
        zf.writestr("0.bin", bad_inner)

    # The corrupted archive has neither IBPS magic NOR the right size for
    # any other parser family — the inference must surface this as an error
    # rather than silently produce a whole_blob measurement.
    with pytest.raises(
        (HnervPacketSectionManifestError, Exception),
        match="(could not infer|IBPS1|parse failed|HNeRV)",
    ):
        build_packet_section_manifest(
            corrupted_zip, label="corrupted", parser=PARSER_AUTO, repo_root=tmp_path
        )
