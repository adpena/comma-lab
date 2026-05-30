# SPDX-License-Identifier: MIT
"""Canonical tests for Z8 M11 L1 macOS-CPU end-to-end smoke runner.

Verifies:
  - Canonical 5-stage observability surface per Catalog #305.
  - Canonical Provenance non-promotable invariants per Catalog #192 + #323.
  - Canonical parser handles upstream/evaluate.py report.txt format.
  - Canonical contest 3-arg signature per Catalog #146 (inflate.sh).
  - Canonical PYTHONPATH self-containment per Catalog #295 (inflate.py).
  - Canonical Z8 M11 milestone status matches build_progress.py.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

from tac.substrates.z8_hierarchical_predictive_coding.m11_l1_macos_cpu_smoke import (
    CANONICAL_RAW_BYTES_PER_VIDEO,
    CANONICAL_UPSTREAM_EVALUATE_RELATIVE,
    CANONICAL_VIDEO_NAMES_RELATIVE,
    CANONICAL_VIDEO_PATH_RELATIVE,
    INFLATE_PY_TEMPLATE,
    INFLATE_SH_TEMPLATE,
    Z8M11L1SmokeResult,
    parse_evaluator_report,
    write_canonical_submission_packet,
)


def _baseline_kwargs() -> dict[str, Any]:
    """Canonical baseline kwargs that satisfy Z8M11L1SmokeResult invariants."""
    return dict(
        substrate_id="z8_hierarchical_predictive_coding",
        lane_id="lane_z8_m11_l1_test",
        schema_version="z8_m11_l1_macos_cpu_smoke_v1",
        git_head_sha="abc123",
        # M9 training
        training_pairs=4,
        training_epochs=5,
        training_resolution_hw=(32, 32),
        training_wall_clock_seconds=0.05,
        training_convergence_verdict="CONVERGED_MONOTONIC",
        training_final_total_loss=0.001,
        training_final_wyner_ziv_payload_bytes=43,
        # M9 archive
        archive_bytes_total=92408,
        archive_sha256="0" * 64,
        archive_emission_wall_clock_seconds=0.01,
        # M11 packet write
        packet_write_wall_clock_seconds=0.005,
        submission_dir_relative="experiments/results/test/submission",
        inflate_sh_path_relative="experiments/results/test/submission/inflate.sh",
        inflate_py_path_relative="experiments/results/test/submission/inflate.py",
        # M10 inflate
        inflate_wall_clock_seconds=12.0,
        inflate_raw_bytes_per_video=CANONICAL_RAW_BYTES_PER_VIDEO,
        inflate_total_videos=1,
        inflate_first_video_sha256_sample_first_4096_bytes="a" * 64,
        # evaluator stage
        evaluator_wall_clock_seconds=180.0,
        evaluator_posenet_distortion=0.123,
        evaluator_segnet_distortion=0.0456,
        evaluator_compression_rate=0.00246,
        evaluator_compressed_size_bytes=92408,
        evaluator_uncompressed_size_bytes=37_545_489,
        evaluator_final_score=5.74,
        evaluator_report_path_relative="experiments/results/test/submission/report.txt",
    )


# ---------- Canonical Provenance non-promotable invariants (Catalog #192 + #323)


class TestZ8M11L1SmokeResultProvenance:
    """Verify the result dataclass enforces non-promotable invariants."""

    def test_baseline_constructs_clean(self) -> None:
        result = Z8M11L1SmokeResult(**_baseline_kwargs())
        assert result.score_claim is False
        assert result.promotable is False
        assert result.ready_for_exact_eval_dispatch is False
        assert result.axis_tag == "[macOS-CPU advisory]"
        assert result.evidence_grade == "macOS-CPU-advisory"
        assert result.hardware_substrate == "macos_arm64"

    def test_rejects_score_claim_true(self) -> None:
        kw = _baseline_kwargs()
        kw["score_claim"] = True
        with pytest.raises(ValueError, match="score_claim=False"):
            Z8M11L1SmokeResult(**kw)

    def test_rejects_promotable_true(self) -> None:
        kw = _baseline_kwargs()
        kw["promotable"] = True
        with pytest.raises(ValueError, match="promotable=False"):
            Z8M11L1SmokeResult(**kw)

    def test_rejects_ready_for_exact_eval_dispatch_true(self) -> None:
        kw = _baseline_kwargs()
        kw["ready_for_exact_eval_dispatch"] = True
        with pytest.raises(ValueError, match="ready_for_exact_eval_dispatch=False"):
            Z8M11L1SmokeResult(**kw)


# ---------- Canonical M11 acceptance criteria invariants


class TestZ8M11L1SmokeResultAcceptanceCriteria:
    """Verify the result dataclass enforces build_progress.py M11 invariants."""

    def test_rejects_nan_final_score(self) -> None:
        kw = _baseline_kwargs()
        kw["evaluator_final_score"] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            Z8M11L1SmokeResult(**kw)

    def test_rejects_inf_final_score(self) -> None:
        kw = _baseline_kwargs()
        kw["evaluator_final_score"] = float("inf")
        with pytest.raises(ValueError, match="finite"):
            Z8M11L1SmokeResult(**kw)

    def test_rejects_negative_final_score(self) -> None:
        kw = _baseline_kwargs()
        kw["evaluator_final_score"] = -0.1
        with pytest.raises(ValueError, match="finite"):
            Z8M11L1SmokeResult(**kw)

    def test_rejects_huge_final_score(self) -> None:
        kw = _baseline_kwargs()
        kw["evaluator_final_score"] = 1500.0
        with pytest.raises(ValueError, match="finite"):
            Z8M11L1SmokeResult(**kw)

    def test_rejects_zero_archive_bytes(self) -> None:
        kw = _baseline_kwargs()
        kw["archive_bytes_total"] = 0
        with pytest.raises(ValueError, match="archive_bytes_total"):
            Z8M11L1SmokeResult(**kw)

    def test_rejects_wrong_inflate_raw_bytes(self) -> None:
        kw = _baseline_kwargs()
        kw["inflate_raw_bytes_per_video"] = 999
        with pytest.raises(ValueError, match="Catalog #367"):
            Z8M11L1SmokeResult(**kw)


# ---------- Canonical Catalog #305 observability surface


class TestZ8M11L1SmokeResultObservabilitySurface:
    """Verify as_dict() carries all 6 facets per Catalog #305."""

    def test_as_dict_canonical_keys(self) -> None:
        result = Z8M11L1SmokeResult(**_baseline_kwargs())
        d = result.as_dict()
        # Inspectable per layer (5 stages)
        assert "training_wall_clock_seconds" in d
        assert "archive_emission_wall_clock_seconds" in d
        assert "packet_write_wall_clock_seconds" in d
        assert "inflate_wall_clock_seconds" in d
        assert "evaluator_wall_clock_seconds" in d
        # Decomposable per signal (3 score components)
        assert "evaluator_posenet_distortion" in d
        assert "evaluator_segnet_distortion" in d
        assert "evaluator_compression_rate" in d
        assert "evaluator_final_score" in d
        # Diff-able across runs (sha256 anchors)
        assert "archive_sha256" in d
        assert "inflate_first_video_sha256_sample_first_4096_bytes" in d
        # Queryable post-hoc
        assert d["schema"] == "z8_m11_l1_macos_cpu_smoke_v1"
        # Cite-able
        assert "substrate_id" in d
        assert "lane_id" in d
        assert "git_head_sha" in d
        # Canonical Provenance
        assert d["score_claim"] is False
        assert d["promotable"] is False
        assert d["ready_for_exact_eval_dispatch"] is False
        assert d["axis_tag"] == "[macOS-CPU advisory]"
        assert d["hardware_substrate"] == "macos_arm64"

    def test_as_dict_json_round_trip(self) -> None:
        result = Z8M11L1SmokeResult(**_baseline_kwargs())
        d = result.as_dict()
        text = json.dumps(d, indent=2, sort_keys=True)
        reparsed = json.loads(text)
        assert reparsed["evaluator_final_score"] == result.evaluator_final_score
        assert reparsed["archive_sha256"] == result.archive_sha256


# ---------- Canonical upstream/evaluate.py report parser


class TestParseEvaluatorReport:
    """Verify the canonical parser handles upstream/evaluate.py report.txt."""

    def test_canonical_report_parses(self) -> None:
        sample = (
            "=== Evaluation config ===\n"
            "  device: cpu\n"
            "=== Evaluation results over 600 samples ===\n"
            "  Average PoseNet Distortion: 0.12345678\n"
            "  Average SegNet Distortion: 0.04567890\n"
            "  Submission file size: 92,408 bytes\n"
            "  Original uncompressed size: 37,545,489 bytes\n"
            "  Compression Rate: 0.00246123\n"
            "  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 5.74\n"
        )
        parsed = parse_evaluator_report(sample)
        assert parsed["posenet_distortion"] == pytest.approx(0.12345678)
        assert parsed["segnet_distortion"] == pytest.approx(0.04567890)
        assert parsed["compression_rate"] == pytest.approx(0.00246123)
        assert parsed["compressed_size_bytes"] == 92408
        assert parsed["uncompressed_size_bytes"] == 37_545_489
        assert parsed["final_score"] == pytest.approx(5.74)

    def test_scientific_notation_score_parses(self) -> None:
        sample = (
            "=== Evaluation results over 600 samples ===\n"
            "  Average PoseNet Distortion: 1.2e-3\n"
            "  Average SegNet Distortion: 4.5e-4\n"
            "  Submission file size: 1 bytes\n"
            "  Original uncompressed size: 1 bytes\n"
            "  Compression Rate: 1.0e-5\n"
            "  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 1.2e-1\n"
        )
        parsed = parse_evaluator_report(sample)
        assert parsed["posenet_distortion"] == pytest.approx(1.2e-3)
        assert parsed["segnet_distortion"] == pytest.approx(4.5e-4)
        assert parsed["compression_rate"] == pytest.approx(1.0e-5)
        assert parsed["final_score"] == pytest.approx(1.2e-1)

    def test_missing_posenet_raises(self) -> None:
        sample = (
            "Average SegNet Distortion: 0.04\n"
            "Submission file size: 1 bytes\n"
            "Original uncompressed size: 1 bytes\n"
            "Compression Rate: 0.0024\n"
            "Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 5.74\n"
        )
        with pytest.raises(ValueError, match="posenet_distortion"):
            parse_evaluator_report(sample)

    def test_missing_final_score_raises(self) -> None:
        sample = (
            "Average PoseNet Distortion: 0.12\n"
            "Average SegNet Distortion: 0.04\n"
            "Submission file size: 1 bytes\n"
            "Original uncompressed size: 1 bytes\n"
            "Compression Rate: 0.0024\n"
        )
        with pytest.raises(ValueError, match="final_score"):
            parse_evaluator_report(sample)


# ---------- Canonical contest packet writer (Catalog #146 + #295)


class TestWriteCanonicalSubmissionPacket:
    """Verify packet writer produces canonical contest packet."""

    def test_writes_canonical_packet(self, tmp_path: Path) -> None:
        archive_bytes = b"Z8HPC1_TEST_PAYLOAD_BYTES_HERE" * 64
        archive_zip, inflate_sh, inflate_py, archive_dir = (
            write_canonical_submission_packet(tmp_path, archive_bytes)
        )
        assert archive_zip.exists()
        assert inflate_sh.exists()
        assert inflate_py.exists()
        # Canonical archive.zip contains 0.bin per single-file grammar
        with zipfile.ZipFile(archive_zip, "r") as zf:
            names = zf.namelist()
            assert names == ["0.bin"]
            assert zf.read("0.bin") == archive_bytes

    def test_inflate_sh_canonical_3_arg_contract(self, tmp_path: Path) -> None:
        archive_bytes = b"test_archive"
        archive_zip, inflate_sh, inflate_py, _ = write_canonical_submission_packet(
            tmp_path, archive_bytes
        )
        sh_text = inflate_sh.read_text(encoding="utf-8")
        # Catalog #146 contract: $1=archive_dir, $2=output_dir, $3=file_list
        assert "ARCHIVE_DIR=\"$1\"" in sh_text
        assert "OUTPUT_DIR=\"$2\"" in sh_text
        assert "FILE_LIST=\"$3\"" in sh_text
        # set -euo pipefail per CLAUDE.md "Forbidden silent-skip cascades"
        assert "set -euo pipefail" in sh_text
        # inflate.sh routes to inflate.py (no scorer at inflate per CLAUDE.md
        # "strict-scorer-rule")
        assert "inflate.py" in sh_text
        # No forbidden scorer loads in inflate.sh itself
        assert "PoseNet" not in sh_text
        assert "SegNet" not in sh_text

    def test_inflate_py_pythonpath_self_contained(self, tmp_path: Path) -> None:
        archive_bytes = b"test_archive"
        _, _, inflate_py, _ = write_canonical_submission_packet(
            tmp_path, archive_bytes
        )
        py_text = inflate_py.read_text(encoding="utf-8")
        # Catalog #295 PYTHONPATH self-containment via canonical sys.path.insert
        # to local src/.
        assert "sys.path.insert" in py_text
        assert "src" in py_text
        # Catalog #205 canonical select_inflate_device inherited via M10
        # main_cli routing
        assert "main_cli" in py_text
        assert "z8_hierarchical_predictive_coding" in py_text

    def test_inflate_sh_executable(self, tmp_path: Path) -> None:
        archive_bytes = b"test_archive"
        _, inflate_sh, _, _ = write_canonical_submission_packet(
            tmp_path, archive_bytes
        )
        mode = inflate_sh.stat().st_mode
        # Owner-execute bit set
        assert mode & 0o100, f"inflate.sh must be executable; got mode {oct(mode)}"


# ---------- Canonical M11 build_progress.py milestone consistency


class TestZ8M11BuildProgressConsistency:
    """Verify M11 milestone in build_progress.py is canonical."""

    def test_m11_milestone_exists_in_tuple(self) -> None:
        from tac.substrates.z8_hierarchical_predictive_coding.build_progress import (
            Z8_PHASE_2_BUILD_MILESTONES,
        )
        m11_ids = [
            m.milestone_id
            for m in Z8_PHASE_2_BUILD_MILESTONES
            if m.milestone_id == "l1_macos_cpu_smoke_landed"
        ]
        assert len(m11_ids) == 1, "M11 milestone must exist exactly once"

    def test_m11_predecessor_chain_canonical(self) -> None:
        from tac.substrates.z8_hierarchical_predictive_coding.build_progress import (
            Z8_PHASE_2_BUILD_MILESTONES,
        )
        m11 = next(
            m
            for m in Z8_PHASE_2_BUILD_MILESTONES
            if m.milestone_id == "l1_macos_cpu_smoke_landed"
        )
        # Per build_progress.py M11 acceptance: predecessors are M10
        # (inflate_runtime_consumes_real_trained_weights).
        assert (
            "inflate_runtime_consumes_real_trained_weights"
            in m11.predecessor_milestone_ids
        )


# ---------- Canonical contest path constants


class TestCanonicalConstants:
    """Verify the canonical contest path + byte constants."""

    def test_canonical_video_path_constant(self) -> None:
        assert CANONICAL_VIDEO_PATH_RELATIVE == "upstream/videos/0.mkv"

    def test_canonical_video_names_constant(self) -> None:
        assert CANONICAL_VIDEO_NAMES_RELATIVE == "upstream/public_test_video_names.txt"

    def test_canonical_evaluator_constant(self) -> None:
        assert CANONICAL_UPSTREAM_EVALUATE_RELATIVE == "upstream/evaluate.py"

    def test_canonical_raw_bytes_per_video(self) -> None:
        # Catalog #367 contest contract: 1164 * 874 * 1200 * 3 = 3,662,409,600
        assert CANONICAL_RAW_BYTES_PER_VIDEO == 3_662_409_600
        assert CANONICAL_RAW_BYTES_PER_VIDEO == 1164 * 874 * 1200 * 3


# ---------- Catalog #287 docstring quality (no placeholder rationales)


class TestCatalog287DocstringQuality:
    """Verify no placeholder rationales in docstrings per Catalog #287."""

    def test_no_placeholder_rationales_in_inflate_templates(self) -> None:
        for template in (INFLATE_SH_TEMPLATE, INFLATE_PY_TEMPLATE):
            # Placeholder literals rejected per Catalog #287 sister discipline
            assert "<rationale>" not in template
            assert "<reason>" not in template
            assert "TBD" not in template
