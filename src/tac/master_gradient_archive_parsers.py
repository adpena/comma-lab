# SPDX-License-Identifier: MIT
"""Canonical archive grammar parser facade for master-gradient extraction.

[verified-against: .omx/research/comprehensive_analytical_surfaces_inventory_plus_synthesis_design_memo_20260518.md §2.3 "Per-archive parser modules needed"]
[verified-against: tools/extract_master_gradient.py (the canonical extractor; this facade is a thin import-stable namespace over its parser surface)]

This module is the canonical importable namespace for archive grammar parsers
referenced by the comprehensive analytical-surfaces inventory memo (TIER-1
op-routable #1). The inventory specifies 4 per-archive parser modules:

  1. ``tac.master_gradient_archive_parsers.a1_grammar_parser``         — A1 archive layout
  2. ``tac.master_gradient_archive_parsers.pr101_lc_v2_grammar_parser`` — PR101_lc_v2 / PR101 gold winner clone
  3. ``tac.master_gradient_archive_parsers.pr106_format0d_grammar_parser`` — PR106 format0d frontier
  4. ``tac.master_gradient_archive_parsers.pr107_apogee_grammar_parser`` — PR107 apogee baseline

Per the inventory: "Each parser implements the canonical
``(archive_path, codec_module) -> ArchiveLayout`` signature mirroring
``_Fec6ArchiveLayout``."

This facade exposes the EXISTING parser implementations already in
``tools/extract_master_gradient.py`` (parse_a1_archive_layout /
parse_pr101_lc_v2_archive_layout / parse_pr106_format0d_archive_layout /
parse_pr107_apogee_archive_layout) under a canonical
``tac.master_gradient_archive_parsers.*`` namespace so:

(a) Downstream consumers (cathedral autopilot ranker, frontier_scan,
    operator_briefing, audit tools) can import the parsers without taking
    a dependency on ``tools/*.py`` (per CLAUDE.md "tac stays clean" — thin
    CLIs may delegate to tac modules, NOT the other way around).

(b) The autograd extraction path in ``tools/extract_master_gradient.py``
    can register additional archive grammars (PR106 format0d Jacobian,
    PR107 apogee Jacobian, DP1 renderer/codebook/residual Jacobian) by
    populating the ``_SUPPORTED_PROJECTORS`` registry without breaking
    consumer imports.

(c) The facade preserves the projector contract per Catalog #327:
    parsers that have ``gradient_projection_supported=True`` emit
    anchor-eligible layouts; parsers with
    ``gradient_projection_supported=False`` are detection-only and emit
    fail-closed projection contracts.

Per CLAUDE.md "Catalog #327 master_gradient raw byte authority not landed":
this facade does NOT expose raw archive-byte / bit master-gradient APIs.
Score-lowering authority routes through typed ``CandidateModificationSpec``
+ ``grammar_aware_operator`` patterns in
``tac.master_gradient_operator_plan``.

Per CLAUDE.md "tac stays clean": this module is a 100% pure-Python facade.
Heavy dependencies (torch, brotli, the autograd extraction path itself) are
intentionally NOT imported at module load. Consumers wanting the autograd
extraction path import via ``tools/extract_master_gradient.py`` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.extract_master_gradient import ArchiveLayout, ArchiveProjectionContract  # noqa: F401

__all__ = [
    "ARCHIVE_GRAMMAR_REGISTRY",
    "a1_grammar_parser",
    "detect_archive_grammar_and_parse",
    "dp1_pretrained_driving_prior_grammar_parser",
    "fec6_fp11_selector_grammar_parser",
    "hnerv_lc_v2_grammar_parser",
    "is_anchor_emitting_grammar",
    "list_archive_grammar_contracts",
    "list_archive_grammar_names",
    "pr101_lc_v2_grammar_parser",
    "pr106_ff_packed_grammar_parser",
    "pr106_format0d_grammar_parser",
    "pr107_apogee_grammar_parser",
]


# --------------------------------------------------------------------------- #
# Canonical grammar registry                                                   #
# --------------------------------------------------------------------------- #

# The canonical mapping from per-archive grammar name to (parser function,
# anchor_emission_eligible) tuple. Anchor-emitting grammars are those that
# have a Jacobian projector wired in the extractor's _SUPPORTED_PROJECTORS
# dict. Detection-only grammars produce a layout for xray/routing but cannot
# emit master-gradient anchors per Catalog #327.

ARCHIVE_GRAMMAR_REGISTRY: tuple[tuple[str, bool], ...] = (
    # (grammar_name, anchor_emission_eligible)
    ("fec6_fp11_selector", True),
    ("a1_finetuned", True),
    ("pr101_lc_v2", True),
    ("pr106_format0d", False),
    ("pr106_ff_packed_hnerv", False),
    ("hnerv_lc_v2_length_prefixed", False),
    ("pr107_apogee_length_prefixed", False),
    ("dp1_pretrained_driving_prior", False),
)


def list_archive_grammar_names() -> tuple[str, ...]:
    """Return the canonical tuple of supported grammar names."""
    return tuple(name for name, _eligible in ARCHIVE_GRAMMAR_REGISTRY)


def is_anchor_emitting_grammar(grammar_name: str) -> bool:
    """Return True iff ``grammar_name`` has a Jacobian projector wired."""
    for name, eligible in ARCHIVE_GRAMMAR_REGISTRY:
        if name == grammar_name:
            return eligible
    return False


# --------------------------------------------------------------------------- #
# Lazy delegation to the canonical extractor parser surface                    #
# --------------------------------------------------------------------------- #


def _delegate(parser_name: str):
    """Return the named parser function from tools/extract_master_gradient.py.

    Lazy import to avoid heavy torch / brotli dependencies at module load.
    Per CLAUDE.md "Beauty, simplicity, and developer experience": the
    delegation pattern keeps this facade narrow (consumers can list/inspect
    the registry without paying the import cost of the extractor's torch
    dependencies).
    """
    # tools/extract_master_gradient.py inserts src/ into sys.path on import;
    # we follow the same pattern for symmetric callsite ergonomics.
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent.parent
    tools_dir = repo_root / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    try:
        import extract_master_gradient as _ext  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"cannot import canonical extractor parser surface: {exc}"
        ) from exc
    return getattr(_ext, parser_name)


def fec6_fp11_selector_grammar_parser(archive_bytes: bytes):
    """Parse fec6 FP11 selector wrapper around a PR101-like inner payload."""
    return _delegate("parse_fec6_fp11_selector_archive_layout")(archive_bytes)


def a1_grammar_parser(archive_bytes: bytes):
    """Parse A1 fine-tuned HNeRV layout (4-byte decoder-section header + PR101)."""
    return _delegate("parse_a1_archive_layout")(archive_bytes)


def pr101_lc_v2_grammar_parser(archive_bytes: bytes):
    """Parse PR101/HNeRV fixed decoder + latent + sidecar layout."""
    return _delegate("parse_pr101_lc_v2_archive_layout")(archive_bytes)


def pr106_format0d_grammar_parser(archive_bytes: bytes):
    """Parse PR106 format0d sidecar archive (frontier ``9cb989cef519...``)."""
    return _delegate("parse_pr106_format0d_archive_layout")(archive_bytes)


def pr106_ff_packed_grammar_parser(archive_bytes: bytes):
    """Parse public PR106's 0xff + uint24 packed HNeRV layout."""
    return _delegate("parse_pr106_ff_packed_archive_layout")(archive_bytes)


def hnerv_lc_v2_grammar_parser(archive_bytes: bytes):
    """Parse true hnerv_lc_v2 four-part length-prefixed layout."""
    return _delegate("parse_hnerv_lc_v2_archive_layout")(archive_bytes)


def pr107_apogee_grammar_parser(archive_bytes: bytes):
    """Parse PR107 Apogee three-part length-prefixed payload."""
    return _delegate("parse_pr107_apogee_archive_layout")(archive_bytes)


def dp1_pretrained_driving_prior_grammar_parser(archive_bytes: bytes):
    """Parse DP1 pre-trained-driving-prior archive sections."""
    return _delegate("parse_dp1_archive_layout")(archive_bytes)


def detect_archive_grammar_and_parse(archive_bytes: bytes):
    """Detect the archive grammar and return a typed ``(name, layout)`` tuple."""
    return _delegate("detect_archive_grammar_and_parse")(archive_bytes)


def list_archive_grammar_contracts():
    """Return the canonical operator-facing grammar registry payload."""
    return _delegate("list_archive_grammar_contracts")()
