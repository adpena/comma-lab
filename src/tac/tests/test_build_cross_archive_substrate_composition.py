# SPDX-License-Identifier: MIT
"""Tests for T9 cross-archive substrate composition tool.

Council coverage:
- Section parsers correctly decode the canonical A1 / PR101 / PR103 byte layouts
- Composability matrix marks decoder/latent as substrate_tied (cannot
  cross-swap; per CLAUDE.md substrate-vs-codec meta-pattern)
- Composability matrix marks PR101↔A1 cross-format swaps as substrate_tied
- Inventory handles missing archives gracefully (returns available=False)
- Smoke composition selection prefers A1-anchored compatible candidates
- Build path raises on length-mismatch for fixed-length sections
- Build_manifest.json carries score_claim=False and lane_tag
- Default cost matrix construction works
"""

from __future__ import annotations

import importlib.util
import json
import struct
import zipfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module loading helper (the tool is in tools/, not in the package)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "build_cross_archive_substrate_composition.py"


def _load_tool_module():
    import sys
    name = "build_cross_archive_substrate_composition"
    spec = importlib.util.spec_from_file_location(name, str(TOOL_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # required so @dataclasses.dataclass can resolve __module__
    spec.loader.exec_module(mod)
    return mod


T9 = _load_tool_module()


# ---------------------------------------------------------------------------
# Layout parsers
# ---------------------------------------------------------------------------


def test_parse_hnerv_ft_microcodec_layout_returns_3_sections() -> None:
    blob = b"\x00" * 162_164 + b"\x01" * 15_387 + b"\x02" * 100  # 177_651 B
    sections = T9.parse_hnerv_ft_microcodec_layout(blob)

    assert [s.name for s in sections] == ["decoder", "latent", "sidecar"]
    assert sections[0].length == 162_164
    assert sections[1].length == 15_387
    assert sections[2].length == 100
    assert sections[0].offset == 0
    assert sections[1].offset == 162_164
    assert sections[2].offset == 162_164 + 15_387


def test_parse_hnerv_ft_microcodec_rejects_too_small() -> None:
    blob = b"\x00" * 100
    with pytest.raises(ValueError, match="too small"):
        T9.parse_hnerv_ft_microcodec_layout(blob)


def test_parse_a1_finetuned_layout_returns_4_sections_with_header() -> None:
    decoder_bytes = b"\x10" * 4_000
    latent_bytes = b"\x20" * 15_387
    sidecar_bytes = b"\x30" * 200
    section_total = 4 + len(decoder_bytes)  # header + decoder
    blob = (
        struct.pack("<I", section_total)
        + decoder_bytes
        + latent_bytes
        + sidecar_bytes
    )

    sections = T9.parse_a1_finetuned_layout(blob)

    names = [s.name for s in sections]
    assert names == ["a1_section_header", "decoder", "latent", "sidecar"]
    assert sections[0].length == 4
    assert sections[1].length == len(decoder_bytes)
    assert sections[2].length == 15_387
    assert sections[3].length == len(sidecar_bytes)


def test_parse_a1_finetuned_layout_rejects_truncated_header() -> None:
    blob = b"\x00\x01"
    with pytest.raises(ValueError, match="too short"):
        T9.parse_a1_finetuned_layout(blob)


def test_parse_a1_finetuned_layout_rejects_invalid_section_total() -> None:
    # section_total > len(blob) is an error.
    blob = struct.pack("<I", 999_999) + b"\x00" * 100
    with pytest.raises(ValueError, match="bad A1 decoder_section_total"):
        T9.parse_a1_finetuned_layout(blob)


def test_parse_hnerv_lc_ac_layout_returns_8_sections() -> None:
    head_len = 28 * 2 + 7097 + 895 + 153_856 + 28 * 4 + 15_537 + 15
    blob = b"\xaa" * (head_len + 50)  # 50-byte wrp tail
    sections = T9.parse_hnerv_lc_ac_layout(blob)

    names = [s.name for s in sections]
    assert names == [
        "scales", "br_concat", "hists", "merged_ac",
        "latent_meta", "latent_lo", "hi_hist", "wrp",
    ]
    assert sections[-1].length == 50  # variable wrp tail


def test_parse_unknown_layout_returns_single_section() -> None:
    blob = b"\xff" * 1024
    sections = T9.parse_unknown_layout(blob)
    assert len(sections) == 1
    assert sections[0].codec == "unknown"
    assert sections[0].length == 1024


# ---------------------------------------------------------------------------
# Composability matrix
# ---------------------------------------------------------------------------


def _fake_inv(name: str, layout: str, sections: list[tuple[str, int, int, str]]) -> dict:
    """Build an inventory dict by hand for matrix tests."""
    return {
        "name": name,
        "available": True,
        "section_layout": layout,
        "sections": [
            {
                "name": s[0],
                "offset": s[1],
                "length": s[2],
                "codec": s[3],
                "sha256": "deadbeef" * 8,
                "notes": "",
            }
            for s in sections
        ],
    }


def test_composability_matrix_marks_decoder_substrate_tied() -> None:
    a = _fake_inv("A", "L1", [("decoder", 0, 100, "c1"), ("sidecar", 100, 50, "c2")])
    b = _fake_inv("B", "L1", [("decoder", 0, 100, "c1"), ("sidecar", 100, 50, "c2")])

    matrix = T9.composability_matrix([a, b])

    decoder_cells = [c for c in matrix["cells"] if c["section"] == "decoder"]
    sidecar_cells = [c for c in matrix["cells"] if c["section"] == "sidecar"]

    assert all(c["verdict"] == "substrate_tied" for c in decoder_cells), (
        "decoder is co-trained; cross-substrate swap must be substrate_tied"
    )
    assert all(c["verdict"] == "compatible" for c in sidecar_cells), (
        "sidecar with same codec+length+layout is compatible"
    )


def test_composability_matrix_marks_codec_mismatch() -> None:
    a = _fake_inv("A", "L1", [("sidecar", 0, 100, "brotli")])
    b = _fake_inv("B", "L1", [("sidecar", 0, 100, "lzma")])

    matrix = T9.composability_matrix([a, b])
    sidecar_cells = [c for c in matrix["cells"] if c["section"] == "sidecar"]

    assert all(c["verdict"] == "codec_mismatch" for c in sidecar_cells)


def test_composability_matrix_marks_length_mismatch_for_fixed_sections() -> None:
    a = _fake_inv("A", "L1", [("scales", 0, 100, "raw")])
    b = _fake_inv("B", "L1", [("scales", 0, 200, "raw")])

    matrix = T9.composability_matrix([a, b])
    cells = [c for c in matrix["cells"] if c["section"] == "scales"]

    assert all(c["verdict"] == "length_mismatch" for c in cells)


def test_composability_matrix_marks_sidecar_compatible_despite_length_diff() -> None:
    """Sidecar is the variable-length tail in hnerv_ft_microcodec; length
    differences are EXPECTED and BYTE-LEVEL safe."""
    a = _fake_inv("A", "L1", [("sidecar", 0, 100, "brotli_per_pair_corrections")])
    b = _fake_inv("B", "L1", [("sidecar", 0, 200, "brotli_per_pair_corrections")])

    matrix = T9.composability_matrix([a, b])
    cells = [c for c in matrix["cells"] if c["section"] == "sidecar"]

    assert all(c["verdict"] == "compatible" for c in cells)


def test_composability_matrix_marks_cross_layout_substrate_tied() -> None:
    a = _fake_inv("A", "L1", [("decoder", 0, 100, "c1")])
    b = _fake_inv("B", "L2", [("decoder", 0, 100, "c1")])

    matrix = T9.composability_matrix([a, b])

    assert all(c["verdict"] == "substrate_tied" for c in matrix["cells"])


def test_composability_matrix_skips_unavailable_substrates() -> None:
    a = _fake_inv("A", "L1", [("sidecar", 0, 100, "c1")])
    b = {"name": "B", "available": False, "missing_path": "/nope"}

    matrix = T9.composability_matrix([a, b])

    assert matrix["cells"] == []


# ---------------------------------------------------------------------------
# Smoke composition selection + assembly
# ---------------------------------------------------------------------------


def test_select_smoke_composition_picks_anchor_sidecar_cell() -> None:
    """Both substrates must be exact-evidence eligible (contest-CUDA or
    contest-CPU GHA). Advisory public-comment rows are filtered out by
    the linter-hardened selection logic.
    """
    a = {
        "name": "A1", "available": True, "section_layout": "X",
        "sections": [{"name": "sidecar", "offset": 0, "length": 100, "codec": "c1",
                      "sha256": "a", "notes": ""}],
        "anchored_score": 0.19, "archive_relpath": "a.zip",
        "archive_sha256": "a", "member_name": "x",
        "exact_evidence_eligible": True,
    }
    b = {
        "name": "DONOR_EXACT", "available": True, "section_layout": "X",
        "sections": [{"name": "sidecar", "offset": 0, "length": 102, "codec": "c1",
                      "sha256": "b", "notes": ""}],
        "anchored_score": 0.19, "archive_relpath": "b.zip",
        "archive_sha256": "b", "member_name": "x",
        "exact_evidence_eligible": True,
    }
    matrix = T9.composability_matrix([a, b])

    pick = T9._select_smoke_composition([a, b], matrix)

    assert pick is not None
    assert pick["anchor"] == "A1"
    assert pick["donor"] == "DONOR_EXACT"
    assert pick["swap_section"] == "sidecar"


def test_select_smoke_composition_filters_advisory_donors() -> None:
    """Advisory donors (public PR host comments) MUST be excluded from the
    smoke build picker; only exact-CUDA / exact-CPU substrates are donors."""
    a = {
        "name": "A1", "available": True, "section_layout": "X",
        "sections": [{"name": "sidecar", "offset": 0, "length": 100, "codec": "c1",
                      "sha256": "a", "notes": ""}],
        "exact_evidence_eligible": True,
    }
    b = {
        "name": "PR101", "available": True, "section_layout": "X",
        "sections": [{"name": "sidecar", "offset": 0, "length": 102, "codec": "c1",
                      "sha256": "b", "notes": ""}],
        "exact_evidence_eligible": False,  # advisory only
    }
    matrix = T9.composability_matrix([a, b])
    pick = T9._select_smoke_composition([a, b], matrix)

    assert pick is None, (
        "Advisory donor PR101 should not seed a smoke composition"
    )


def test_select_smoke_composition_returns_none_when_no_compatible_cells() -> None:
    a = _fake_inv("A1", "L1", [("decoder", 0, 100, "c1")])
    b = _fake_inv("PR101", "L2", [("decoder", 0, 100, "c1")])

    matrix = T9.composability_matrix([a, b])
    pick = T9._select_smoke_composition([a, b], matrix)

    assert pick is None


# ---------------------------------------------------------------------------
# End-to-end via main() — inventory-only path (no --build)
# ---------------------------------------------------------------------------


def test_main_emits_manifest_and_inventory(tmp_path, monkeypatch) -> None:
    out = tmp_path / "t9_out"
    monkeypatch.setattr("sys.argv", [
        "build_cross_archive_substrate_composition.py",
        "--output-root", str(out),
    ])
    rc = T9.main()

    assert rc == 0
    assert (out / "build_manifest.json").exists()
    assert (out / "byte_layout_inventory.json").exists()
    assert (out / "composability_matrix.json").exists()
    assert (out / "rebuild_command.txt").exists()

    manifest = json.loads((out / "build_manifest.json").read_text())
    # Custody fields per CLAUDE.md gate B2 (no naked bytes).
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["archive_bound_candidate_contract_required"] is True
    assert manifest["lane_tag"] == "[T9-substrate-composition]"
    assert manifest["target_modes"] == ["contest_exact_eval"]
    # The dispatch_blockers / reactivation_criteria fields must always populate.
    assert (
        isinstance(manifest["dispatch_blockers"], list)
        and len(manifest["dispatch_blockers"]) > 0
    )
    assert (
        "reactivation_required_before_new_dispatch" in manifest["dispatch_blockers"]
    ), "preflight gate B6 requires this exact sentinel"
    assert (
        isinstance(manifest["reactivation_criteria"], list)
        and len(manifest["reactivation_criteria"]) > 0
    )


def test_smoke_inflate_pass_does_not_grant_exact_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "t9_smoke"
    registry = (
        T9.SubstrateEntry(
            name="A1",
            archive_relpath="unused_anchor.zip",
            submission_relpath="unused_submission",
            anchored_score=0.19,
            anchored_score_tag="[contest-CPU GHA]",
            member_name="x",
            section_layout="X",
        ),
        T9.SubstrateEntry(
            name="DONOR_EXACT",
            archive_relpath="unused_donor.zip",
            submission_relpath="unused_submission",
            anchored_score=0.18,
            anchored_score_tag="[contest-CPU GHA]",
            member_name="x",
            section_layout="X",
        ),
    )

    def fake_inventory(entry: object) -> dict:
        name = entry.name
        return {
            "name": name,
            "available": True,
            "section_layout": "X",
            "sections": [
                {
                    "name": "sidecar",
                    "offset": 0,
                    "length": 8 if name == "A1" else 10,
                    "codec": "brotli_per_pair_corrections",
                    "sha256": (name.encode().hex() * 8)[:64],
                    "notes": "",
                }
            ],
            "anchored_score": entry.anchored_score,
            "anchored_score_tag": entry.anchored_score_tag,
            "archive_relpath": entry.archive_relpath,
            "archive_sha256": ("a" if name == "A1" else "b") * 64,
            "member_name": "x",
            "member_size_bytes": 8 if name == "A1" else 10,
            "exact_evidence_eligible": True,
        }

    monkeypatch.setattr(T9, "SUBSTRATE_REGISTRY", registry)
    monkeypatch.setattr(T9, "inventory_substrate", fake_inventory)
    monkeypatch.setattr(T9, "_assemble_swapped_blob", lambda *args: b"payload")
    monkeypatch.setattr(T9, "_smoke_inflate", lambda *args: {"smoke_ok": True})
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_cross_archive_substrate_composition.py",
            "--output-root",
            str(out),
            "--build",
            "--smoke-inflate",
            "--gha-dispatch",
        ],
    )

    rc = T9.main()

    assert rc == 0
    manifest = json.loads((out / "build_manifest.json").read_text())
    assert manifest["smoke_inflate_passed"] is True
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["score_claim"] is False
    assert "proposed_gha_dispatch_command" not in manifest
    assert "blocked_gha_dispatch_command" in manifest
    assert manifest["gha_dispatch_blocked_reason"] == (
        T9.EXACT_DISPATCH_CONTRACT_BLOCKER
    )
    assert T9.EXACT_DISPATCH_CONTRACT_BLOCKER in manifest["dispatch_blockers"]
    assert T9.SMOKE_INFLATE_AUTHORITY_BLOCKER in manifest["dispatch_blockers"]


# ---------------------------------------------------------------------------
# Sha-256 helper / Section dataclass
# ---------------------------------------------------------------------------


def test_sha256_helper_is_deterministic() -> None:
    assert T9._sha256(b"abc") == T9._sha256(b"abc")
    assert T9._sha256(b"abc") != T9._sha256(b"abcd")


def test_is_exact_evidence_tag_classifies_correctly() -> None:
    assert T9._is_exact_evidence_tag("[contest-CUDA]") is True
    assert T9._is_exact_evidence_tag("[contest-CPU GHA]") is True
    assert T9._is_exact_evidence_tag("[contest-CPU]") is True
    assert T9._is_exact_evidence_tag("[advisory: public PR101 host-claimed]") is False
    assert T9._is_exact_evidence_tag("[macOS-CPU advisory only]") is False


def test_section_dataclass_has_all_fields() -> None:
    s = T9.Section("name", 10, 20, "codec", "sha", "notes")
    assert s.name == "name"
    assert s.offset == 10
    assert s.length == 20
    assert s.codec == "codec"
    assert s.sha256 == "sha"
    assert s.notes == "notes"


# ---------------------------------------------------------------------------
# Build path with synthetic substrates (full integration, no real archive)
# ---------------------------------------------------------------------------


def test_assemble_swapped_blob_replaces_sidecar_correctly(tmp_path) -> None:
    """Build a fake A1-style and PR101-style archive on disk, then call
    _assemble_swapped_blob to verify it produces the expected bytes."""
    # Anchor blob (PR101-shaped, not A1; we use parse_hnerv_ft_microcodec).
    anchor_blob = b"\xaa" * 162_164 + b"\xbb" * 15_387 + b"\xcc" * 50
    donor_blob = b"\xdd" * 162_164 + b"\xee" * 15_387 + b"\x11" * 80

    anchor_zip = tmp_path / "anchor.zip"
    donor_zip = tmp_path / "donor.zip"
    with zipfile.ZipFile(anchor_zip, "w") as z:
        z.writestr("x", anchor_blob)
    with zipfile.ZipFile(donor_zip, "w") as z:
        z.writestr("x", donor_blob)

    anchor_inv = {
        "name": "ANCHOR", "available": True,
        "archive_relpath": str(anchor_zip.relative_to(tmp_path)),
        "section_layout": "hnerv_ft_microcodec", "member_name": "x",
        "sections": [
            {"name": "decoder", "offset": 0, "length": 162_164, "codec": "c", "sha256": "a", "notes": ""},
            {"name": "latent", "offset": 162_164, "length": 15_387, "codec": "c", "sha256": "a", "notes": ""},
            {"name": "sidecar", "offset": 162_164 + 15_387, "length": 50, "codec": "c", "sha256": "a", "notes": ""},
        ],
    }
    donor_inv = {
        "name": "DONOR", "available": True,
        "archive_relpath": str(donor_zip.relative_to(tmp_path)),
        "section_layout": "hnerv_ft_microcodec", "member_name": "x",
        "sections": [
            {"name": "decoder", "offset": 0, "length": 162_164, "codec": "c", "sha256": "a", "notes": ""},
            {"name": "latent", "offset": 162_164, "length": 15_387, "codec": "c", "sha256": "a", "notes": ""},
            {"name": "sidecar", "offset": 162_164 + 15_387, "length": 80, "codec": "c", "sha256": "a", "notes": ""},
        ],
    }

    out = T9._assemble_swapped_blob(anchor_inv, donor_inv, "sidecar", tmp_path)

    # Decoder + latent come from anchor; sidecar comes from donor.
    assert out[:162_164] == b"\xaa" * 162_164
    assert out[162_164 : 162_164 + 15_387] == b"\xbb" * 15_387
    assert out[162_164 + 15_387 :] == b"\x11" * 80


def test_assemble_swapped_blob_rejects_length_mismatch_on_fixed_section(tmp_path) -> None:
    """Decoder is fixed-length; mismatched lengths must raise."""
    anchor_blob = b"\xaa" * 200
    donor_blob = b"\xdd" * 100
    anchor_zip = tmp_path / "anchor.zip"
    donor_zip = tmp_path / "donor.zip"
    with zipfile.ZipFile(anchor_zip, "w") as z:
        z.writestr("x", anchor_blob)
    with zipfile.ZipFile(donor_zip, "w") as z:
        z.writestr("x", donor_blob)

    anchor_inv = {
        "name": "A", "available": True,
        "archive_relpath": str(anchor_zip.relative_to(tmp_path)),
        "section_layout": "X", "member_name": "x",
        "sections": [{"name": "decoder", "offset": 0, "length": 200, "codec": "c", "sha256": "a", "notes": ""}],
    }
    donor_inv = {
        "name": "B", "available": True,
        "archive_relpath": str(donor_zip.relative_to(tmp_path)),
        "section_layout": "X", "member_name": "x",
        "sections": [{"name": "decoder", "offset": 0, "length": 100, "codec": "c", "sha256": "a", "notes": ""}],
    }

    with pytest.raises(RuntimeError, match="length mismatch"):
        T9._assemble_swapped_blob(anchor_inv, donor_inv, "decoder", tmp_path)


def test_assemble_archive_uses_deterministic_zip(tmp_path) -> None:
    """Per CLAUDE.md check_archive_builders_use_deterministic_zip, the
    archive must use ZipInfo with a fixed timestamp so identical inputs
    produce byte-identical outputs."""
    blob = b"hello world"
    out_a = tmp_path / "a.zip"
    out_b = tmp_path / "b.zip"
    T9._assemble_archive(blob, "x", out_a)
    T9._assemble_archive(blob, "x", out_b)

    assert out_a.read_bytes() == out_b.read_bytes(), (
        "Archive builder is non-deterministic"
    )
