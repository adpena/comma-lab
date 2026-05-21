# SPDX-License-Identifier: MIT
"""Tests for the ATW2 FULL candidate MVP-first phasing generator.

Cross-references OVERNIGHT-I 2026-05-21 + canonical generator at
``tools/generate_atw2_full_candidate_smoke.py``. Verifies the generator
produces an archive classified as ``full_candidate=True`` by the canonical
scanner at ``tools/scan_atw2_cdf_compaction_candidates.py``, and that the
canonical Provenance fields per Catalog #323 + slot 3-r7 reconciliation +
RATIFY-4 EXCLUDED context #6 are correctly populated.

These tests do NOT exercise the inflate-side compaction parity proof
(``prove_atw2_cdf_compaction_parity``) because that path is covered by the
existing ``test_cdf_dead_section.py`` suite and is orthogonal to the
classification-gate proof this generator addresses.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Import the canonical generator module dynamically because tools/ is not a
# package; mirrors the pattern used elsewhere in the repo for tools/ imports.
REPO_ROOT = Path(__file__).resolve().parents[5]
_GENERATOR_PATH = REPO_ROOT / "tools" / "generate_atw2_full_candidate_smoke.py"


def _load_generator_module():
    spec = importlib.util.spec_from_file_location(
        "_atw2_full_candidate_smoke_generator",
        _GENERATOR_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GEN = _load_generator_module()


def test_generator_produces_full_candidate_archive(tmp_path: Path) -> None:
    """The default 600-pair smoke produces an archive the canonical scanner
    classifies as ``full_candidate=True``."""
    result = GEN.generate_atw2_full_candidate_smoke(
        num_pairs=GEN.FULL_CANDIDATE_MIN_PAIRS,
        output_dir=tmp_path,
    )
    assert result.full_candidate_per_gate is True
    assert result.candidate_class == "full_candidate"
    assert result.num_pairs == 600
    assert result.full_candidate_min_pairs_threshold == 600


def test_generator_below_threshold_is_smoke_or_small(tmp_path: Path) -> None:
    """num_pairs below 600 produces ``smoke_or_small_candidate`` per the
    canonical scanner threshold."""
    result = GEN.generate_atw2_full_candidate_smoke(
        num_pairs=8,
        output_dir=tmp_path,
    )
    assert result.full_candidate_per_gate is False
    assert result.candidate_class == "smoke_or_small_candidate"
    assert result.num_pairs == 8


def test_generator_writes_bin_and_zip_artifacts(tmp_path: Path) -> None:
    """The generator writes both ``0.bin`` and ``archive.zip`` artifacts."""
    result = GEN.generate_atw2_full_candidate_smoke(
        num_pairs=600,
        output_dir=tmp_path,
    )
    bin_path = Path(result.archive_path)
    zip_path = Path(result.archive_zip_path)
    assert bin_path.exists()
    assert zip_path.exists()
    assert bin_path.stat().st_size == result.archive_bytes
    assert zip_path.stat().st_size == result.archive_zip_bytes


def test_generator_archive_parses_via_canonical_parser(tmp_path: Path) -> None:
    """The generated archive parses cleanly via the canonical
    ``parse_archive`` + ``analyze_atw2_cdf_section`` helpers."""
    from tac.substrates.atw_codec_v2 import ATW2_MAGIC, parse_archive
    from tac.substrates.atw_codec_v2.cdf_dead_section import (
        analyze_atw2_cdf_section,
    )

    result = GEN.generate_atw2_full_candidate_smoke(
        num_pairs=600,
        output_dir=tmp_path,
    )
    bin_bytes = Path(result.archive_path).read_bytes()
    assert bin_bytes.startswith(ATW2_MAGIC)
    parsed = parse_archive(bin_bytes)
    assert parsed.schema_version == 1
    assert int(parsed.latent_residual.shape[0]) == 600
    analysis = analyze_atw2_cdf_section(bin_bytes)
    assert analysis.cdf_bytes == result.cdf_bytes
    assert analysis.cdf_classes == result.cdf_classes
    assert analysis.cdf_symbols == result.cdf_symbols


def test_generator_canonical_provenance_per_catalog_323(tmp_path: Path) -> None:
    """The result manifest carries canonical Provenance fields per Catalog
    #323 + slot 3-r7 + RATIFY-4 EXCLUDED context #6."""
    result = GEN.generate_atw2_full_candidate_smoke(
        num_pairs=600,
        output_dir=tmp_path,
    )
    # Catalog #323 / #287 canonical non-promotable markers
    assert result.score_claim is False
    assert result.promotion_eligible is False
    assert result.ready_for_exact_eval_dispatch is False
    assert result.research_only is True
    assert result.evidence_grade == "predicted"
    # slot 3-r7 + RATIFY-4 binding
    assert result.removal_paradigm_reclassification_per_slot_3_r7 is True
    assert result.canonical_equation_26_excluded_context_per_ratify_4 == (
        "direct_byte_substitution_on_decode_opaque_raw_sections"
    )
    # Cite-chain to canonical source memos
    assert result.provenance_source_memo_slot_3_r7.endswith(
        "atw_v2_cdf_table_blob_reconciliation_codex_byte_mutation_smoke_"
        "falsified_20260521.md"
    )
    assert result.provenance_source_memo_ratify_4.endswith(
        "canonical_equation_26_excluded_context_decode_opaque_raw_sections_"
        "registration_landed_20260521.md"
    )


def test_generator_is_byte_deterministic_given_seed(tmp_path: Path) -> None:
    """Same seed produces byte-identical archives across runs."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    result_a = GEN.generate_atw2_full_candidate_smoke(
        num_pairs=600, output_dir=out_a, seed=20260521
    )
    result_b = GEN.generate_atw2_full_candidate_smoke(
        num_pairs=600, output_dir=out_b, seed=20260521
    )
    assert result_a.archive_sha256 == result_b.archive_sha256
    assert result_a.archive_zip_sha256 == result_b.archive_zip_sha256
    assert result_a.archive_bytes == result_b.archive_bytes
    assert result_a.cdf_offset == result_b.cdf_offset
    assert result_a.cdf_bytes == result_b.cdf_bytes
    assert result_a.cdf_classes == result_b.cdf_classes
    assert result_a.cdf_symbols == result_b.cdf_symbols


def test_generator_classified_by_canonical_scanner(tmp_path: Path) -> None:
    """The generated archive is classified ``full_candidate=True`` by the
    canonical scanner at ``tools/scan_atw2_cdf_compaction_candidates.py``.

    This is the end-to-end contract proof: the gate-resolution scenario.
    """
    # Import the canonical scanner the same way as the generator does.
    scanner_path = REPO_ROOT / "tools" / "scan_atw2_cdf_compaction_candidates.py"
    spec = importlib.util.spec_from_file_location(
        "_atw2_scanner_for_test", scanner_path
    )
    assert spec is not None and spec.loader is not None
    scanner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = scanner
    spec.loader.exec_module(scanner)

    # Generate the smoke artifact.
    GEN.generate_atw2_full_candidate_smoke(
        num_pairs=600, output_dir=tmp_path
    )

    # Run the canonical scanner against the tmp_path.
    report = scanner.scan_atw2_cdf_candidates(
        [tmp_path],
        member_names=["0.bin", "x"],
        max_archives=None,
    )
    assert report.candidates_found == 1
    candidate = report.candidates[0]
    assert candidate.full_candidate is True
    assert candidate.candidate_class == "full_candidate"
    assert candidate.num_pairs == 600
    assert candidate.score_claim is False
    assert candidate.promotion_eligible is False
    assert candidate.ready_for_exact_eval_dispatch is False


def test_generator_cli_writes_result_json(tmp_path: Path) -> None:
    """The CLI entry point writes a ``result.json`` manifest in ``output_dir``."""
    rc = GEN.main(
        [
            "--output-dir",
            str(tmp_path),
            "--num-pairs",
            "600",
        ]
    )
    assert rc == 0
    manifest = tmp_path / "result.json"
    assert manifest.exists()
    payload = json.loads(manifest.read_text())
    assert payload["full_candidate_per_gate"] is True
    assert payload["candidate_class"] == "full_candidate"
    assert payload["num_pairs"] == 600
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False


def test_generator_refuses_negative_or_zero_num_pairs(tmp_path: Path) -> None:
    """num_pairs <= 0 is rejected by the ATW2 codec config."""
    with pytest.raises((ValueError, RuntimeError, AssertionError, TypeError)):
        GEN.generate_atw2_full_candidate_smoke(
            num_pairs=0, output_dir=tmp_path
        )


def test_generator_byte_stability_across_runs_smoke_size(tmp_path: Path) -> None:
    """Smoke-size (num_pairs=8) is also byte-deterministic across runs."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    result_a = GEN.generate_atw2_full_candidate_smoke(
        num_pairs=8, output_dir=out_a, seed=20260521
    )
    result_b = GEN.generate_atw2_full_candidate_smoke(
        num_pairs=8, output_dir=out_b, seed=20260521
    )
    assert result_a.archive_sha256 == result_b.archive_sha256
    assert result_a.candidate_class == "smoke_or_small_candidate"
    assert result_a.full_candidate_per_gate is False


def test_full_candidate_min_pairs_threshold_matches_canonical_scanner() -> None:
    """The generator's FULL_CANDIDATE_MIN_PAIRS constant matches the canonical
    scanner's threshold to prevent drift between generator + classifier.

    Catalog #176 sister discipline: the generator and the scanner share the
    same threshold by importing the scanner's constant in the test.
    """
    scanner_path = REPO_ROOT / "tools" / "scan_atw2_cdf_compaction_candidates.py"
    spec = importlib.util.spec_from_file_location(
        "_atw2_scanner_threshold_check", scanner_path
    )
    assert spec is not None and spec.loader is not None
    scanner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = scanner
    spec.loader.exec_module(scanner)
    assert GEN.FULL_CANDIDATE_MIN_PAIRS == scanner.FULL_CANDIDATE_MIN_PAIRS == 600
