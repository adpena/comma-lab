# SPDX-License-Identifier: MIT
"""Tests for Catalog #210 + #211 DP1 hardening preflight gates.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
non-negotiable: every bug class fix lands a paired STRICT preflight gate
with dedicated tests. These tests pin Catalog #210 (DP1 codebook provenance
metadata gate) and Catalog #211 (DP1 composition routes through canonical
helper) at STRICT @ 0 from byte one and cover positive / negative / waiver
behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_dp1_codebook_provenance_metadata_present,
    check_dp1_composition_routes_through_canonical_helper,
)


# =============================================================================
# Catalog #210 - DP1 codebook provenance metadata gate
# =============================================================================


def _write(tmp_path: Path, rel: str, body: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    return p


def test_check_210_live_repo_count_zero():
    """Live regression guard: the live repository MUST stay at zero
    violations because Catalog #210 is STRICT-from-byte-one."""
    v = check_dp1_codebook_provenance_metadata_present(strict=False)
    assert v == [], f"Catalog #210 live count is non-zero: {v[:3]}"


def test_check_210_positive_dp1_archive_missing_provenance(tmp_path: Path):
    """Caller imports DP1 pack_archive AND mentions DrivingPriorArchive
    but builds a meta dict without provenance — flagged."""
    body = '''
from tac.substrates.pretrained_driving_prior import pack_archive, DrivingPriorArchive

def build_my_dp1_archive(book, state_dict, residual):
    meta = {
        "residual_int8_scale": 64.0,
    }
    return pack_archive(
        book, state_dict, residual, meta,
        num_pairs=4, output_height=64, output_width=96, per_pair_bytes=8,
    )
'''
    _write(tmp_path, "src/tac/foo/dp1_packer.py", body)
    v = check_dp1_codebook_provenance_metadata_present(
        repo_root=tmp_path, strict=False
    )
    assert len(v) == 1
    assert "src/tac/foo/dp1_packer.py" in v[0]
    assert "license_tags" in v[0] or "Catalog #210" in v[0]


def test_check_210_negative_provenance_keys_present(tmp_path: Path):
    """Same fixture but with all 6 provenance keys in the meta dict — clean."""
    body = '''
from tac.substrates.pretrained_driving_prior import pack_archive, DrivingPriorArchive

def build_my_dp1_archive(book, state_dict, residual):
    meta = {
        "residual_int8_scale": 64.0,
        "license_tags": book.metadata["license_tags"],
        "dataset_provenance": book.metadata["dataset_provenance"],
        "distillation_version": book.metadata["distillation_version"],
        "random_seed": book.metadata["random_seed"],
        "basis_sha256": book.metadata["basis_sha256"],
        "num_frames_used": book.metadata["num_frames_used"],
    }
    return pack_archive(
        book, state_dict, residual, meta,
        num_pairs=4, output_height=64, output_width=96, per_pair_bytes=8,
    )
'''
    _write(tmp_path, "src/tac/foo/dp1_packer_clean.py", body)
    v = check_dp1_codebook_provenance_metadata_present(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_check_210_same_line_waiver_accepted(tmp_path: Path):
    """Same-line ``# DP1_PROVENANCE_OK:<rationale>`` waiver accepted."""
    body = '''
from tac.substrates.pretrained_driving_prior import pack_archive, DP1_MAGIC

def build_minimal(book):
    meta = {"residual_int8_scale": 64.0}
    return pack_archive(book, {}, b"", meta, num_pairs=0, output_height=0, output_width=0, per_pair_bytes=0)  # DP1_PROVENANCE_OK:test-fixture-no-real-distillation
'''
    _write(tmp_path, "src/tac/foo/dp1_waived.py", body)
    v = check_dp1_codebook_provenance_metadata_present(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_check_210_placeholder_reason_literal_rejected(tmp_path: Path):
    """Bare ``<reason>`` placeholder MUST NOT silently waive (security)."""
    body = '''
from tac.substrates.pretrained_driving_prior import pack_archive, DP1_MAGIC

def build_minimal(book):
    meta = {"residual_int8_scale": 64.0}
    return pack_archive(book, {}, b"", meta, num_pairs=0, output_height=0, output_width=0, per_pair_bytes=0)  # DP1_PROVENANCE_OK:<reason>
'''
    _write(tmp_path, "src/tac/foo/dp1_placeholder.py", body)
    v = check_dp1_codebook_provenance_metadata_present(
        repo_root=tmp_path, strict=False
    )
    assert len(v) == 1


def test_check_210_substrate_name_in_registry_not_flagged(tmp_path: Path):
    """False-positive guard: file mentions ``pretrained_driving_prior`` in a
    registry but doesn't import DP1's pack_archive and doesn't reference DP1
    archive tokens — must NOT flag."""
    body = '''
SUBSTRATES = ["a1", "pr101", "pretrained_driving_prior", "hdm8"]

def pack_archive(state, meta):
    # this is a DIFFERENT pack_archive, not DP1's
    return state + meta
'''
    _write(tmp_path, "src/tac/foo/other_substrate.py", body)
    v = check_dp1_codebook_provenance_metadata_present(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_check_210_strict_mode_raises_preflight_error(tmp_path: Path):
    body = '''
from tac.substrates.pretrained_driving_prior import pack_archive, DrivingPriorArchive

def build_my_dp1_archive(book, state_dict, residual):
    meta = {"residual_int8_scale": 64.0}
    return pack_archive(book, state_dict, residual, meta, num_pairs=4, output_height=64, output_width=96, per_pair_bytes=8)
'''
    _write(tmp_path, "src/tac/foo/dp1_strict.py", body)
    with pytest.raises(PreflightError, match="Catalog #210"):
        check_dp1_codebook_provenance_metadata_present(
            repo_root=tmp_path, strict=True
        )


def test_check_210_tests_dir_exempt(tmp_path: Path):
    """Test files are exempt from the scan (they exercise both positive
    and negative paths)."""
    body = '''
from tac.substrates.pretrained_driving_prior import pack_archive, DrivingPriorArchive

def test_dp1_archive_construction():
    meta = {"residual_int8_scale": 64.0}
    return pack_archive(None, {}, b"", meta, num_pairs=0, output_height=0, output_width=0, per_pair_bytes=0)
'''
    _write(tmp_path, "src/tac/foo/tests/test_dp1.py", body)
    v = check_dp1_codebook_provenance_metadata_present(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_check_210_canonical_implementation_self_exempt(tmp_path: Path):
    """The DP1 distillation module + codebook module + composition module
    + preflight.py are self-exempt by construction (they DEFINE the
    contract)."""
    # Re-running against the live repo proves the canonical files don't
    # self-flag.
    v = check_dp1_codebook_provenance_metadata_present(strict=False)
    assert v == []


def test_check_210_experiments_results_exempt(tmp_path: Path):
    """Generated build artifacts under experiments/results/ are out of scope."""
    body = '''
from tac.substrates.pretrained_driving_prior import pack_archive, DrivingPriorArchive

def build_my_dp1_archive(book, state_dict, residual):
    meta = {"residual_int8_scale": 64.0}
    return pack_archive(book, state_dict, residual, meta, num_pairs=4, output_height=64, output_width=96, per_pair_bytes=8)
'''
    _write(tmp_path, "experiments/results/lane_dp1_smoke/build.py", body)
    v = check_dp1_codebook_provenance_metadata_present(
        repo_root=tmp_path, strict=False
    )
    assert v == []


# =============================================================================
# Catalog #211 - DP1 composition routes through canonical helper
# =============================================================================


def test_check_211_live_repo_count_zero():
    """Live regression guard for Catalog #211."""
    v = check_dp1_composition_routes_through_canonical_helper(strict=False)
    assert v == [], f"Catalog #211 live count is non-zero: {v[:3]}"


def test_check_211_positive_hand_rolled_dpcomp_magic(tmp_path: Path):
    r"""File references ``b"DPC\x00"`` magic directly without routing through
    compose_with — flagged."""
    body = r'''
import struct

# Hand-rolled DP1 composition — silently bypasses the canonical helper.
def my_compose_dp1_a1(dp1_bytes, a1_bytes):
    header = struct.pack("<4sBI4s", b"DPC\x00", 1, len(dp1_bytes), b"A1\x00\x00")
    return header + dp1_bytes + a1_bytes
'''
    _write(tmp_path, "tools/hand_rolled_compose.py", body)
    v = check_dp1_composition_routes_through_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert len(v) >= 1
    assert "hand_rolled_compose.py" in v[0]
    assert "Catalog #211" in v[0]


def test_check_211_negative_compose_with_imported(tmp_path: Path):
    """File uses the canonical compose_with helper — clean."""
    body = r'''
from tac.substrates.pretrained_driving_prior import compose_with

def my_workflow(dp1_bytes, a1_bytes):
    return compose_with(dp1_bytes, a1_bytes, base_substrate="a1")
'''
    _write(tmp_path, "tools/canonical_compose.py", body)
    v = check_dp1_composition_routes_through_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_check_211_negative_decompose_imported(tmp_path: Path):
    """``decompose`` (the inverse helper) also satisfies the canonical
    routing — clean."""
    body = r'''
from tac.substrates.pretrained_driving_prior import decompose

def my_unwrap(composed_bytes):
    return decompose(composed_bytes)
'''
    _write(tmp_path, "tools/canonical_decompose.py", body)
    v = check_dp1_composition_routes_through_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_check_211_negative_verify_composition_imported(tmp_path: Path):
    """``verify_composition`` (the forensic surface) satisfies the canonical
    routing — clean."""
    body = r'''
from tac.substrates.pretrained_driving_prior import verify_composition

def my_audit(composed_bytes):
    report = verify_composition(composed_bytes, expected_base_substrate="a1")
    assert report["base_substrate"] == "a1"
    # Even mentioning the DP1 magic literal is fine because we route via verify_composition.
    print("DP1 magic check: b'DPC\\x00'")
'''
    _write(tmp_path, "tools/audit_composition.py", body)
    v = check_dp1_composition_routes_through_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_check_211_same_line_waiver_accepted(tmp_path: Path):
    r"""Same-line ``# DP1_COMPOSITION_OK:<rationale>`` waiver accepted for
    the rare legitimate manual byte path."""
    body = r'''
import struct

def emergency_recovery(dp1_bytes, a1_bytes):
    # Operator-reviewed manual recovery path; canonical compose_with
    # is broken for this offline replay.
    header = struct.pack("<4sBI4s", b"DPC\x00", 1, len(dp1_bytes), b"A1\x00\x00")  # DP1_COMPOSITION_OK:emergency-offline-replay-operator-approved
    return header + dp1_bytes + a1_bytes
'''
    _write(tmp_path, "tools/emergency_recovery.py", body)
    v = check_dp1_composition_routes_through_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_check_211_placeholder_reason_literal_rejected(tmp_path: Path):
    r"""Placeholder ``<reason>`` literal MUST NOT silently waive."""
    body = r'''
import struct

def my_fake_waived(dp1_bytes, a1_bytes):
    header = struct.pack("<4sBI4s", b"DPC\x00", 1, len(dp1_bytes), b"A1\x00\x00")  # DP1_COMPOSITION_OK:<reason>
    return header + dp1_bytes + a1_bytes
'''
    _write(tmp_path, "tools/fake_waived.py", body)
    v = check_dp1_composition_routes_through_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert len(v) >= 1


def test_check_211_archive_module_pattern_accepted(tmp_path: Path):
    r"""Files that look like the DP1 archive module (use DP1_MAGIC AND
    DP1_HEADER_FMT to pack) are heuristically accepted — that's Catalog #210
    territory, not composition."""
    body = r'''
import struct

DP1_MAGIC = b"DP1\x00"
DP1_HEADER_FMT = "<4sBHHHBIIII"

def pack_archive(...):
    header = struct.pack(DP1_HEADER_FMT, DP1_MAGIC, 1, ...)
'''
    _write(tmp_path, "src/tac/foo/my_archive_module.py", body)
    v = check_dp1_composition_routes_through_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_check_211_strict_mode_raises_preflight_error(tmp_path: Path):
    body = r'''
import struct

def my_compose(dp1_bytes, a1_bytes):
    header = struct.pack("<4sBI4s", b"DPC\x00", 1, len(dp1_bytes), b"A1\x00\x00")
    return header + dp1_bytes + a1_bytes
'''
    _write(tmp_path, "tools/strict_violator.py", body)
    with pytest.raises(PreflightError, match="Catalog #211"):
        check_dp1_composition_routes_through_canonical_helper(
            repo_root=tmp_path, strict=True
        )


def test_check_211_tests_dir_exempt(tmp_path: Path):
    """Test files are exempt by scope rule."""
    body = r'''
import struct

def test_hand_rolled_for_negative_check():
    header = struct.pack("<4sBI4s", b"DPC\x00", 1, 0, b"A1\x00\x00")
    assert len(header) == 13
'''
    _write(tmp_path, "src/tac/foo/tests/test_things.py", body)
    v = check_dp1_composition_routes_through_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_check_211_intake_clones_exempt(tmp_path: Path):
    """Vendored intake clones are exempt by path filter."""
    body = r'''
import struct

def public_pr_compose(dp1_bytes, a1_bytes):
    header = struct.pack("<4sBI4s", b"DPC\x00", 1, len(dp1_bytes), b"A1\x00\x00")
    return header + dp1_bytes + a1_bytes
'''
    _write(
        tmp_path,
        "experiments/results/public_pr_intake_foo/source/compose.py",
        body,
    )
    v = check_dp1_composition_routes_through_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_check_211_canonical_implementation_self_exempt():
    """The composition module + substrate __init__.py + preflight.py are
    self-exempt by construction."""
    v = check_dp1_composition_routes_through_canonical_helper(strict=False)
    assert v == []


def test_check_211_dp1_magic_alone_without_compose_handler_flagged(tmp_path: Path):
    r"""Mentioning DP1 magic in a way that suggests parsing/building a DP1
    archive but WITHOUT routing through canonical helpers or being a real
    archive module — flagged."""
    body = r'''
def my_byte_inspect(bytes_in):
    # Looking for DP1 archives smuggled into other byte streams
    if bytes_in.startswith(b"DP1\x00"):
        return bytes_in[28:]  # skip header
    return None
'''
    _write(tmp_path, "tools/byte_inspect.py", body)
    v = check_dp1_composition_routes_through_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert len(v) >= 1
