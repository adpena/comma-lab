# SPDX-License-Identifier: MIT
"""Dedicated tests for D1/D4/DP1 wire-in in ScorerConditionalMDLEstimator.

Part B of lane ``lane_ibps1_parser_wave_p0_d1_d4_dp1_20260514`` (operator-routed
decision #1 from ``feedback_ibps1_canonical_surface_landed_20260514.md`` —
P0 follow-up: promote D1 / D4 / DP1 canonical parsers + wire them into
``hnerv_packet_sections.py`` so ``ScorerConditionalMDLEstimator`` can compute
section-aware MDL on these 3 across-class substrates).

Sister of ``test_scorer_conditional_mdl_ibps1.py`` (the IBPS1 reference).

These tests verify that the canonical XRay primitive
:class:`tac.xray.mdl_scorer_conditional.ScorerConditionalMDLEstimator` (and
the underlying ``tac.analysis.scorer_conditional_mdl`` + parser-section
manifest dispatch in ``tac.analysis.hnerv_packet_sections``) recognizes
D1POLY1 / WZF01 / DP1 archives by their magic prefixes and emits the
canonical section breakdown for each substrate.
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import pytest

from tac.substrates.d1_segnet_margin_polytope.archive import (
    D1POLY1_HEADER_FMT,
    D1POLY1_MAGIC,
    D1POLY1_SCHEMA_VERSION,
    D1POLY1_SECTION_ROLES,
)
from tac.substrates.d4_wyner_ziv_frame_0.archive import (
    BASE_SHA_HEX_LEN,
    WZF01_HEADER_FMT,
    WZF01_MAGIC,
    WZF01_SCHEMA_VERSION,
    WZF01_SECTION_ROLES,
)
from tac.substrates.pretrained_driving_prior.archive import (
    DP1_HEADER_FMT,
    DP1_MAGIC,
    DP1_SCHEMA_VERSION,
    DP1_SECTION_ROLES,
)


# --------------------------------------------------------------------------
# Synthetic archive builders (in-memory; no real training required)
# --------------------------------------------------------------------------


def _build_d1poly1_inner() -> bytes:
    margin_map_blob = b"\xaa" * 400
    polytope_blob = b"\xbb" * 200
    meta_blob = b'{"k":"v"}'
    base_id = b"a1"
    base_sha = b"0123456789abcdef"
    header = struct.pack(
        D1POLY1_HEADER_FMT,
        D1POLY1_MAGIC,
        D1POLY1_SCHEMA_VERSION,
        96, 128, 1.0, 0.5,
        len(base_id), len(base_sha),
        len(margin_map_blob), len(polytope_blob), len(meta_blob),
    )
    return header + base_id + base_sha + margin_map_blob + polytope_blob + meta_blob


def _build_wzf01_inner() -> bytes:
    base_sha = b"0" * 64
    base_bytes = b"\xff" * 500
    motion_blob = b"\xaa" * 1000
    residual_blob = b"\xbb" * 2000
    meta_blob = b'{"motion_mode":0}'
    header = struct.pack(
        WZF01_HEADER_FMT,
        WZF01_MAGIC, WZF01_SCHEMA_VERSION,
        600, 0, 0, 0, 12, 16,
        BASE_SHA_HEX_LEN,
        len(base_bytes), len(motion_blob), len(residual_blob), len(meta_blob),
    )
    return header + base_sha + base_bytes + motion_blob + residual_blob + meta_blob


def _build_dp1_inner() -> bytes:
    codebook_blob = b"\xaa" * 500
    renderer_blob = b"\xbb" * 2000
    residual_blob = b"\xcc" * 600
    meta_blob = b'{"k":"v"}'
    header = struct.pack(
        DP1_HEADER_FMT,
        DP1_MAGIC, DP1_SCHEMA_VERSION,
        600, 384, 512, 1,
        len(codebook_blob), len(renderer_blob), len(residual_blob), len(meta_blob),
    )
    return header + codebook_blob + renderer_blob + residual_blob + meta_blob


def _build_d1poly1_zip(tmp_path: Path) -> Path:
    out = tmp_path / "d1poly1_archive.zip"
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("0.bin", _build_d1poly1_inner())
    return out


def _build_wzf01_zip(tmp_path: Path) -> Path:
    out = tmp_path / "wzf01_archive.zip"
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("0.bin", _build_wzf01_inner())
    return out


def _build_dp1_zip(tmp_path: Path) -> Path:
    out = tmp_path / "dp1_archive.zip"
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("0.bin", _build_dp1_inner())
    return out


# ==========================================================================
# D1POLY1 wire-in
# ==========================================================================


class TestD1Poly1WireIn:
    def test_parser_section_manifest_dispatches_d1poly1_by_magic(
        self, tmp_path: Path
    ) -> None:
        """build_packet_section_manifest must auto-detect D1POLY1 by ``b"D1PY"`` magic."""
        from tac.analysis.hnerv_packet_sections import (
            PARSER_AUTO,
            build_packet_section_manifest,
        )

        archive_path = _build_d1poly1_zip(tmp_path)
        manifest = build_packet_section_manifest(
            archive_path,
            label="d1poly1_synth",
            parser=PARSER_AUTO,
            repo_root=tmp_path,
        )
        parser = manifest.get("parser")
        assert isinstance(parser, dict), parser
        assert parser.get("name") == "d1poly1_segnet_margin_polytope"

    def test_parser_section_manifest_d1poly1_emits_six_canonical_sections(
        self, tmp_path: Path
    ) -> None:
        """The D1POLY1 dispatch must emit all 6 canonical sections, not whole_blob."""
        from tac.analysis.hnerv_packet_sections import (
            PARSER_AUTO,
            build_packet_section_manifest,
        )

        archive_path = _build_d1poly1_zip(tmp_path)
        manifest = build_packet_section_manifest(
            archive_path,
            label="d1poly1_synth",
            parser=PARSER_AUTO,
            repo_root=tmp_path,
        )
        sections = manifest.get("sections")
        assert isinstance(sections, list)
        section_names = {str(s["name"]) for s in sections}
        assert section_names == set(D1POLY1_SECTION_ROLES.keys())
        assert "whole_blob" not in section_names

    def test_parser_section_manifest_d1poly1_section_roles_match_canonical(
        self, tmp_path: Path
    ) -> None:
        """The dispatched sections must use D1POLY1_SECTION_ROLES."""
        from tac.analysis.hnerv_packet_sections import (
            PARSER_AUTO,
            build_packet_section_manifest,
        )

        archive_path = _build_d1poly1_zip(tmp_path)
        manifest = build_packet_section_manifest(
            archive_path,
            label="d1poly1_synth",
            parser=PARSER_AUTO,
            repo_root=tmp_path,
        )
        sections = manifest.get("sections")
        assert isinstance(sections, list)
        for sec in sections:
            name = str(sec["name"])
            role = str(sec["optimization_role"])
            assert role == D1POLY1_SECTION_ROLES[name], (
                f"D1POLY1 section {name!r} role {role!r} != canonical "
                f"{D1POLY1_SECTION_ROLES[name]!r}"
            )

    def test_xray_primitive_returns_d1poly1_section_breakdown(
        self, tmp_path: Path
    ) -> None:
        """ScorerConditionalMDLEstimator.compute() must surface D1POLY1 sections."""
        from tac.xray.mdl_scorer_conditional import ScorerConditionalMDLEstimator

        archive_path = _build_d1poly1_zip(tmp_path)
        est = ScorerConditionalMDLEstimator()
        result = est.compute(archive_path, label="d1poly1_synth")
        breakdown = result.primitive_value.per_section_breakdown
        section_names = {row[0] for row in breakdown}
        assert section_names == set(D1POLY1_SECTION_ROLES.keys())

    def test_xray_primitive_d1poly1_mdl_density_in_unit_interval(
        self, tmp_path: Path
    ) -> None:
        """The reported mdl_density must be a valid [0, 1] proxy."""
        from tac.xray.mdl_scorer_conditional import ScorerConditionalMDLEstimator

        archive_path = _build_d1poly1_zip(tmp_path)
        est = ScorerConditionalMDLEstimator()
        result = est.compute(archive_path, label="d1poly1_synth")
        assert 0.0 <= result.primitive_value.mdl_density <= 1.0


# ==========================================================================
# WZF01 wire-in
# ==========================================================================


class TestWZF01WireIn:
    def test_parser_section_manifest_dispatches_wzf01_by_magic(
        self, tmp_path: Path
    ) -> None:
        """build_packet_section_manifest must auto-detect WZF01 by ``b"WZF\\x01"`` magic."""
        from tac.analysis.hnerv_packet_sections import (
            PARSER_AUTO,
            build_packet_section_manifest,
        )

        archive_path = _build_wzf01_zip(tmp_path)
        manifest = build_packet_section_manifest(
            archive_path,
            label="wzf01_synth",
            parser=PARSER_AUTO,
            repo_root=tmp_path,
        )
        parser = manifest.get("parser")
        assert isinstance(parser, dict), parser
        assert parser.get("name") == "wzf01_wyner_ziv_frame_0"

    def test_parser_section_manifest_wzf01_emits_six_canonical_sections(
        self, tmp_path: Path
    ) -> None:
        """The WZF01 dispatch must emit all 6 canonical sections."""
        from tac.analysis.hnerv_packet_sections import (
            PARSER_AUTO,
            build_packet_section_manifest,
        )

        archive_path = _build_wzf01_zip(tmp_path)
        manifest = build_packet_section_manifest(
            archive_path,
            label="wzf01_synth",
            parser=PARSER_AUTO,
            repo_root=tmp_path,
        )
        sections = manifest.get("sections")
        assert isinstance(sections, list)
        section_names = {str(s["name"]) for s in sections}
        assert section_names == set(WZF01_SECTION_ROLES.keys())

    def test_parser_section_manifest_wzf01_section_roles_match_canonical(
        self, tmp_path: Path
    ) -> None:
        """The dispatched sections must use WZF01_SECTION_ROLES."""
        from tac.analysis.hnerv_packet_sections import (
            PARSER_AUTO,
            build_packet_section_manifest,
        )

        archive_path = _build_wzf01_zip(tmp_path)
        manifest = build_packet_section_manifest(
            archive_path,
            label="wzf01_synth",
            parser=PARSER_AUTO,
            repo_root=tmp_path,
        )
        for sec in manifest["sections"]:
            name = str(sec["name"])
            assert sec["optimization_role"] == WZF01_SECTION_ROLES[name]

    def test_xray_primitive_returns_wzf01_section_breakdown(
        self, tmp_path: Path
    ) -> None:
        """ScorerConditionalMDLEstimator.compute() must surface WZF01 sections."""
        from tac.xray.mdl_scorer_conditional import ScorerConditionalMDLEstimator

        archive_path = _build_wzf01_zip(tmp_path)
        est = ScorerConditionalMDLEstimator()
        result = est.compute(archive_path, label="wzf01_synth")
        breakdown = result.primitive_value.per_section_breakdown
        section_names = {row[0] for row in breakdown}
        assert section_names == set(WZF01_SECTION_ROLES.keys())

    def test_xray_primitive_wzf01_engages_wire_in_hooks(
        self, tmp_path: Path
    ) -> None:
        """The XRay primitive result must declare wire-in hooks engaged."""
        from tac.xray.mdl_scorer_conditional import ScorerConditionalMDLEstimator

        archive_path = _build_wzf01_zip(tmp_path)
        est = ScorerConditionalMDLEstimator()
        result = est.compute(archive_path, label="wzf01_synth")
        engaged = set(result.wire_in_hooks_engaged)
        assert engaged, "wire_in_hooks_engaged must be nonempty"


# ==========================================================================
# DP1 wire-in
# ==========================================================================


class TestDP1WireIn:
    def test_parser_section_manifest_dispatches_dp1_by_magic(
        self, tmp_path: Path
    ) -> None:
        """build_packet_section_manifest must auto-detect DP1 by ``b"DP1\\x00"`` magic."""
        from tac.analysis.hnerv_packet_sections import (
            PARSER_AUTO,
            build_packet_section_manifest,
        )

        archive_path = _build_dp1_zip(tmp_path)
        manifest = build_packet_section_manifest(
            archive_path,
            label="dp1_synth",
            parser=PARSER_AUTO,
            repo_root=tmp_path,
        )
        parser = manifest.get("parser")
        assert isinstance(parser, dict), parser
        assert parser.get("name") == "dp1_pretrained_driving_prior"

    def test_parser_section_manifest_dp1_emits_five_canonical_sections(
        self, tmp_path: Path
    ) -> None:
        """The DP1 dispatch must emit all 5 canonical sections."""
        from tac.analysis.hnerv_packet_sections import (
            PARSER_AUTO,
            build_packet_section_manifest,
        )

        archive_path = _build_dp1_zip(tmp_path)
        manifest = build_packet_section_manifest(
            archive_path,
            label="dp1_synth",
            parser=PARSER_AUTO,
            repo_root=tmp_path,
        )
        sections = manifest.get("sections")
        assert isinstance(sections, list)
        section_names = {str(s["name"]) for s in sections}
        assert section_names == set(DP1_SECTION_ROLES.keys())

    def test_parser_section_manifest_dp1_section_roles_match_canonical(
        self, tmp_path: Path
    ) -> None:
        """The dispatched sections must use DP1_SECTION_ROLES."""
        from tac.analysis.hnerv_packet_sections import (
            PARSER_AUTO,
            build_packet_section_manifest,
        )

        archive_path = _build_dp1_zip(tmp_path)
        manifest = build_packet_section_manifest(
            archive_path,
            label="dp1_synth",
            parser=PARSER_AUTO,
            repo_root=tmp_path,
        )
        for sec in manifest["sections"]:
            name = str(sec["name"])
            assert sec["optimization_role"] == DP1_SECTION_ROLES[name]

    def test_xray_primitive_returns_dp1_section_breakdown(
        self, tmp_path: Path
    ) -> None:
        """ScorerConditionalMDLEstimator.compute() must surface DP1 sections."""
        from tac.xray.mdl_scorer_conditional import ScorerConditionalMDLEstimator

        archive_path = _build_dp1_zip(tmp_path)
        est = ScorerConditionalMDLEstimator()
        result = est.compute(archive_path, label="dp1_synth")
        breakdown = result.primitive_value.per_section_breakdown
        section_names = {row[0] for row in breakdown}
        assert section_names == set(DP1_SECTION_ROLES.keys())


# ==========================================================================
# Cross-substrate Tier-A invariance
# ==========================================================================


class TestCrossSubstrateInvariance:
    def test_corrupted_d1poly1_magic_does_not_silently_fall_back(
        self, tmp_path: Path
    ) -> None:
        """Corrupted D1POLY1 magic must surface as error, not silent whole_blob."""
        from tac.analysis.hnerv_packet_sections import (
            PARSER_AUTO,
            HnervPacketSectionManifestError,
            build_packet_section_manifest,
        )

        inner = _build_d1poly1_inner()
        # Corrupt magic AND make the size NOT match any known parser's
        # fixed-length window so all auto-detectors fail.
        bad_inner = b"XXXX" + inner[4:]
        corrupted_zip = tmp_path / "d1poly1_corrupted.zip"
        with zipfile.ZipFile(corrupted_zip, "w") as zf:
            zf.writestr("0.bin", bad_inner)

        with pytest.raises(
            (HnervPacketSectionManifestError, Exception),
            match="(could not infer|D1POLY1|parse failed|HNeRV)",
        ):
            build_packet_section_manifest(
                corrupted_zip,
                label="d1poly1_corrupted",
                parser=PARSER_AUTO,
                repo_root=tmp_path,
            )

    def test_parser_aliases_resolve_for_all_three_substrates(self) -> None:
        """Substrate-id aliases must route to the canonical parser names."""
        from tac.analysis.hnerv_packet_sections import (
            PARSER_ALIASES,
            PARSER_D1POLY1,
            PARSER_DP1,
            PARSER_WZF01,
        )

        assert PARSER_ALIASES["d1poly1"] == PARSER_D1POLY1
        assert PARSER_ALIASES["d1_segnet_margin_polytope"] == PARSER_D1POLY1
        assert PARSER_ALIASES["wzf01"] == PARSER_WZF01
        assert PARSER_ALIASES["d4_wyner_ziv_frame_0"] == PARSER_WZF01
        assert PARSER_ALIASES["dp1"] == PARSER_DP1
        assert PARSER_ALIASES["pretrained_driving_prior"] == PARSER_DP1

    def test_section_roles_use_canonical_taxonomy_for_all_three(self) -> None:
        """All three substrates' section roles must be in ROLE_WEIGHTS."""
        from tac.analysis.scorer_conditional_mdl import ROLE_WEIGHTS

        for name, roles in [
            ("D1POLY1", D1POLY1_SECTION_ROLES),
            ("WZF01", WZF01_SECTION_ROLES),
            ("DP1", DP1_SECTION_ROLES),
        ]:
            for section, role in roles.items():
                assert role in ROLE_WEIGHTS, (
                    f"{name}/{section}: role {role!r} not in canonical "
                    f"ROLE_WEIGHTS taxonomy"
                )
