# SPDX-License-Identifier: MIT
"""Focused tests for ``tac.master_gradient_wire_in``.

Cover the 4 wire-in surfaces (frontier_scan / probe_outcomes_ledger /
continual_learning / deploy.modal.call_id_ledger) plus the post-hoc query
helpers (query_anchors_with_master_gradient_coverage +
compute_master_gradient_wire_in_coverage).

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287 evidence
tags + Catalog #327 raw-byte-authority discipline: every test pins the
expected wire-in behavior to the canonical contract surface, not to a
specific numeric score / promotion verdict.

The fec6 archive sha256 ``f174192aeadf...`` is the live empirical anchor
known to exist in ``.omx/state/master_gradient_anchors.jsonl`` at landing
time per `feedback_master_gradient_canonical_helper_landed_with_cathedral_autopilot_wirein_20260517.md`.
Tests that depend on this anchor use it as a positive control; tests that
must be deterministic across future state mutation use temp ledgers.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tac.master_gradient_wire_in import (
    MASTER_GRADIENT_WIRE_IN_SCHEMA_VERSION,
    MasterGradientAnnotation,
    annotate_frontier_anchors_with_master_gradient_existence,
    annotate_posterior_row_with_master_gradient_anchor,
    compute_master_gradient_wire_in_coverage,
    query_anchors_with_master_gradient_coverage,
    register_dispatched_call_id_with_master_gradient_anchor,
    register_probe_outcome_with_master_gradient_anchor,
)

# Canonical empirical anchor known to exist in the live master_gradient_anchors
# ledger per `feedback_master_gradient_canonical_helper_landed_*` memo.
LIVE_FEC6_ARCHIVE_SHA = (
    "f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd"
)

# Synthetic sha known NOT to exist (zero-padded; never a real archive sha).
SYNTHETIC_MISSING_SHA = "0" * 64


def _write_synthetic_anchor_ledger(
    tmp_path: Path,
    archive_sha256: str,
    *,
    measurement_axis: str = "[contest-CUDA T4]",
    measurement_hardware: str = "linux_x86_64_t4",
) -> Path:
    """Write a tmp master_gradient_anchors.jsonl with one synthetic anchor."""
    ledger_path = tmp_path / "master_gradient_anchors.jsonl"
    row = {
        "schema_version": "master_gradient_anchor_v1",
        "archive_sha256": archive_sha256,
        "measurement_axis": measurement_axis,
        "measurement_hardware": measurement_hardware,
        "measurement_utc": "2026-05-18T00:00:00.000000Z",
        "measurement_call_id": "synthetic_test",
        "measurement_method": "test_fixture",
        "n_bytes": 1000,
        "operating_point": {"score": 0.5, "d_seg": 0.0, "d_pose": 0.0, "rate": 0.5},
        "pareto_facets": [],
        "rashomon_disagreement_score": None,
        "written_at_utc": "2026-05-18T00:00:00.000000+00:00",
        "written_host": "test_host",
        "written_pid": 0,
    }
    ledger_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    return ledger_path


# --------------------------------------------------------------------------- #
# MasterGradientAnnotation dataclass invariants                                #
# --------------------------------------------------------------------------- #


class TestMasterGradientAnnotationContract:
    def test_default_construction(self):
        anno = MasterGradientAnnotation(
            archive_sha256="abc",
            anchor_exists=False,
        )
        assert anno.archive_sha256 == "abc"
        assert anno.anchor_exists is False
        assert anno.measurement_axis is None
        assert anno.is_authoritative_axis is False

    def test_as_dict_contains_schema_version(self):
        anno = MasterGradientAnnotation(
            archive_sha256="abc", anchor_exists=True
        )
        payload = anno.as_dict()
        assert payload["schema"] == MASTER_GRADIENT_WIRE_IN_SCHEMA_VERSION

    def test_as_dict_is_json_serializable(self):
        anno = MasterGradientAnnotation(
            archive_sha256="abc",
            anchor_exists=True,
            measurement_axis="[contest-CPU]",
            measurement_hardware="linux_x86_64_cpu",
            is_authoritative_axis=True,
        )
        payload = anno.as_dict()
        json.dumps(payload)  # must not raise

    def test_frozen_immutability(self):
        anno = MasterGradientAnnotation(
            archive_sha256="abc", anchor_exists=False
        )
        with pytest.raises(FrozenInstanceError):
            anno.anchor_exists = True  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# WIRE-IN #1: frontier_scan annotation                                         #
# --------------------------------------------------------------------------- #


class TestFrontierAnchorAnnotation:
    def test_annotation_preserves_original_anchor_fields(self):
        from tac.frontier_scan import Anchor

        anchor = Anchor(
            score=0.19205,
            axis="contest_cpu",
            archive_sha256=SYNTHETIC_MISSING_SHA,
            hardware_substrate="linux_x86_64_cpu",
            source_path="test/path.md",
        )
        annotated = annotate_frontier_anchors_with_master_gradient_existence(
            [anchor]
        )
        assert len(annotated) == 1
        row = annotated[0]
        assert row["score"] == 0.19205
        assert row["axis"] == "contest_cpu"
        assert row["archive_sha256"] == SYNTHETIC_MISSING_SHA
        assert row["hardware_substrate"] == "linux_x86_64_cpu"
        assert row["source_path"] == "test/path.md"

    def test_annotation_adds_canonical_axis_and_is_qualifying(self):
        from tac.frontier_scan import Anchor

        anchor = Anchor(
            score=0.19205,
            axis="contest_cpu",
            archive_sha256=SYNTHETIC_MISSING_SHA,
            hardware_substrate="linux_x86_64_cpu",
            source_path="test/path.md",
        )
        annotated = annotate_frontier_anchors_with_master_gradient_existence(
            [anchor]
        )
        assert annotated[0]["canonical_axis"] == "contest_cpu"
        assert annotated[0]["is_qualifying"] is True

    def test_missing_anchor_resolves_to_anchor_exists_false(self, tmp_path):
        from tac.frontier_scan import Anchor

        anchor = Anchor(
            score=0.5,
            axis="contest_cpu",
            archive_sha256=SYNTHETIC_MISSING_SHA,
            hardware_substrate="linux_x86_64_cpu",
            source_path="test",
        )
        # Use empty ledger so the lookup misses
        ledger = tmp_path / "empty.jsonl"
        ledger.write_text("", encoding="utf-8")
        annotated = annotate_frontier_anchors_with_master_gradient_existence(
            [anchor], ledger_path=ledger
        )
        mg = annotated[0]["master_gradient_annotation"]
        assert mg["anchor_exists"] is False
        assert mg["measurement_axis"] is None
        assert mg["is_authoritative_axis"] is False

    def test_existing_anchor_resolves_with_axis_and_authority(self, tmp_path):
        from tac.frontier_scan import Anchor

        target_sha = "a" * 64
        ledger = _write_synthetic_anchor_ledger(
            tmp_path,
            target_sha,
            measurement_axis="[contest-CUDA T4]",
            measurement_hardware="linux_x86_64_t4",
        )
        anchor = Anchor(
            score=0.5,
            axis="contest_cuda",
            archive_sha256=target_sha,
            hardware_substrate="linux_x86_64_t4",
            source_path="test",
        )
        annotated = annotate_frontier_anchors_with_master_gradient_existence(
            [anchor], ledger_path=ledger
        )
        mg = annotated[0]["master_gradient_annotation"]
        assert mg["anchor_exists"] is True
        assert mg["measurement_axis"] == "[contest-CUDA T4]"
        assert mg["measurement_hardware"] == "linux_x86_64_t4"

    def test_empty_anchor_list_returns_empty_list(self):
        annotated = annotate_frontier_anchors_with_master_gradient_existence([])
        assert annotated == []

    def test_anchor_without_archive_sha_resolves_to_empty_annotation(self):
        from tac.frontier_scan import Anchor

        anchor = Anchor(
            score=0.5,
            axis="contest_cpu",
            archive_sha256="",
            hardware_substrate="linux_x86_64_cpu",
            source_path="test",
        )
        annotated = annotate_frontier_anchors_with_master_gradient_existence(
            [anchor]
        )
        mg = annotated[0]["master_gradient_annotation"]
        assert mg["anchor_exists"] is False


# --------------------------------------------------------------------------- #
# WIRE-IN #2: probe_outcomes_ledger wrapper                                    #
# --------------------------------------------------------------------------- #


class TestProbeOutcomesLedgerWrapper:
    def test_wrapper_threads_master_gradient_annotation_via_extra(self, tmp_path):
        ledger_path = tmp_path / "probe_outcomes.jsonl"
        lock_path = tmp_path / ".probe_outcomes.lock"
        # Use a synthetic mg ledger so the test is deterministic
        mg_ledger = _write_synthetic_anchor_ledger(tmp_path, "b" * 64)

        record = register_probe_outcome_with_master_gradient_anchor(
            probe_id="test_probe_id_wire_in_1",
            substrate="test_substrate",
            recipe_path=None,
            probe_kind="diagnostic_smoke",
            verdict="PROCEED",
            metric_name="test_metric",
            metric_value=0.5,
            archive_sha256="b" * 64,
            master_gradient_ledger_path=mg_ledger,
            probe_outcomes_ledger_path=ledger_path,
            probe_outcomes_lock_path=lock_path,
        )
        assert record["master_gradient_anchor_archive_sha256"] == "b" * 64
        assert record["master_gradient_anchor_present"] is True
        assert record["master_gradient_anchor_axis"] == "[contest-CUDA T4]"

    def test_wrapper_without_archive_sha_does_not_inject_annotation(
        self, tmp_path
    ):
        ledger_path = tmp_path / "probe_outcomes.jsonl"
        lock_path = tmp_path / ".probe_outcomes.lock"
        record = register_probe_outcome_with_master_gradient_anchor(
            probe_id="test_probe_id_wire_in_2",
            substrate="test_substrate",
            recipe_path=None,
            probe_kind="diagnostic_smoke",
            verdict="PROCEED",
            metric_name="test_metric",
            metric_value=0.5,
            # No archive_sha256 provided
            probe_outcomes_ledger_path=ledger_path,
            probe_outcomes_lock_path=lock_path,
        )
        # Annotation fields should NOT be in the record
        assert "master_gradient_anchor_archive_sha256" not in record

    def test_wrapper_preserves_canonical_probe_outcome_schema(self, tmp_path):
        """Annotation must NOT clobber canonical schema fields."""
        ledger_path = tmp_path / "probe_outcomes.jsonl"
        lock_path = tmp_path / ".probe_outcomes.lock"
        record = register_probe_outcome_with_master_gradient_anchor(
            probe_id="test_probe_id_wire_in_3",
            substrate="test_substrate",
            recipe_path="recipe.yaml",
            probe_kind="diagnostic_smoke",
            verdict="DEFER",
            metric_name="test_metric",
            metric_value=1.5,
            threshold=0.5,
            archive_sha256="c" * 64,
            probe_outcomes_ledger_path=ledger_path,
            probe_outcomes_lock_path=lock_path,
        )
        assert record["probe_id"] == "test_probe_id_wire_in_3"
        assert record["verdict"] == "DEFER"
        assert record["metric_value"] == 1.5
        assert record["blocker_status"] == "blocking"  # DEFER is a blocking verdict


# --------------------------------------------------------------------------- #
# WIRE-IN #3: continual_learning posterior row annotation                      #
# --------------------------------------------------------------------------- #


class TestPosteriorRowAnnotation:
    def test_annotation_does_not_mutate_input_row(self, tmp_path):
        row = {
            "archive_sha256": SYNTHETIC_MISSING_SHA,
            "score_value": 0.5,
            "axis": "contest_cpu",
        }
        ledger = tmp_path / "empty.jsonl"
        ledger.write_text("", encoding="utf-8")
        annotated = annotate_posterior_row_with_master_gradient_anchor(
            row, ledger_path=ledger
        )
        # Original row not modified
        assert "master_gradient_annotation" not in row
        # Annotated copy has the new key
        assert "master_gradient_annotation" in annotated
        # Original keys preserved
        assert annotated["score_value"] == 0.5
        assert annotated["axis"] == "contest_cpu"

    def test_annotation_with_existing_anchor(self, tmp_path):
        target_sha = "d" * 64
        ledger = _write_synthetic_anchor_ledger(tmp_path, target_sha)
        row = {"archive_sha256": target_sha, "score_value": 0.5}
        annotated = annotate_posterior_row_with_master_gradient_anchor(
            row, ledger_path=ledger
        )
        mg = annotated["master_gradient_annotation"]
        assert mg["anchor_exists"] is True
        assert mg["measurement_axis"] == "[contest-CUDA T4]"

    def test_annotation_without_archive_sha_resolves_to_empty(self):
        row = {"score_value": 0.5, "axis": "contest_cpu"}
        annotated = annotate_posterior_row_with_master_gradient_anchor(row)
        mg = annotated["master_gradient_annotation"]
        assert mg["anchor_exists"] is False

    def test_annotation_rejects_non_mapping_input(self):
        with pytest.raises(TypeError):
            annotate_posterior_row_with_master_gradient_anchor([1, 2, 3])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# WIRE-IN #4: deploy.modal.call_id_ledger wrapper                              #
# --------------------------------------------------------------------------- #


class TestCallIdLedgerWrapper:
    def test_wrapper_threads_master_gradient_annotation_via_extra(self, tmp_path):
        ledger_path = tmp_path / "call_id.jsonl"
        lock_path = tmp_path / ".call_id.lock"
        target_sha = "e" * 64
        mg_ledger = _write_synthetic_anchor_ledger(tmp_path, target_sha)

        record = register_dispatched_call_id_with_master_gradient_anchor(
            call_id="fc-test-wire-in-1",
            lane_id="lane_test_wire_in",
            label="smoke_test_wire_in",
            archive_sha256=target_sha,
            master_gradient_ledger_path=mg_ledger,
            call_id_ledger_path=ledger_path,
            call_id_lock_path=lock_path,
        )
        assert record["master_gradient_anchor_archive_sha256"] == target_sha
        assert record["master_gradient_anchor_present"] is True
        assert record["master_gradient_anchor_axis"] == "[contest-CUDA T4]"

    def test_wrapper_without_archive_sha_does_not_inject_annotation(
        self, tmp_path
    ):
        ledger_path = tmp_path / "call_id.jsonl"
        lock_path = tmp_path / ".call_id.lock"
        record = register_dispatched_call_id_with_master_gradient_anchor(
            call_id="fc-test-wire-in-2",
            lane_id="lane_test_wire_in",
            label="smoke_test_wire_in_no_archive",
            # No archive_sha256
            call_id_ledger_path=ledger_path,
            call_id_lock_path=lock_path,
        )
        assert "master_gradient_anchor_archive_sha256" not in record

    def test_wrapper_preserves_canonical_call_id_schema(self, tmp_path):
        ledger_path = tmp_path / "call_id.jsonl"
        lock_path = tmp_path / ".call_id.lock"
        record = register_dispatched_call_id_with_master_gradient_anchor(
            call_id="fc-test-wire-in-3",
            lane_id="lane_test_wire_in_schema",
            label="smoke_test_schema_preserved",
            archive_sha256="f" * 64,
            platform="modal",
            gpu="A100",
            expected_cost_usd=5.0,
            call_id_ledger_path=ledger_path,
            call_id_lock_path=lock_path,
        )
        assert record["call_id"] == "fc-test-wire-in-3"
        assert record["lane_id"] == "lane_test_wire_in_schema"
        assert record["platform"] == "modal"
        assert record["gpu"] == "A100"
        assert record["expected_cost_usd"] == 5.0


# --------------------------------------------------------------------------- #
# Post-hoc query helpers                                                       #
# --------------------------------------------------------------------------- #


class TestQueryAndCoverageHelpers:
    def test_query_normalizes_sha_to_lowercase(self, tmp_path):
        target_sha = "9876543210" + "a" * 54
        ledger = _write_synthetic_anchor_ledger(tmp_path, target_sha)
        # Pass uppercase variant
        upper_sha = target_sha.upper()
        result = query_anchors_with_master_gradient_coverage(
            [upper_sha], ledger_path=ledger
        )
        assert target_sha in result
        assert result[target_sha].anchor_exists is True

    def test_query_with_multiple_archives(self, tmp_path):
        existing_sha = "1" * 64
        ledger = _write_synthetic_anchor_ledger(tmp_path, existing_sha)
        missing_sha = "2" * 64
        result = query_anchors_with_master_gradient_coverage(
            [existing_sha, missing_sha], ledger_path=ledger
        )
        assert result[existing_sha].anchor_exists is True
        assert result[missing_sha].anchor_exists is False

    def test_query_rejects_empty_sha(self):
        with pytest.raises(ValueError):
            query_anchors_with_master_gradient_coverage([""])

    def test_compute_coverage_percentages(self, tmp_path):
        existing_sha = "3" * 64
        ledger = _write_synthetic_anchor_ledger(
            tmp_path,
            existing_sha,
            measurement_axis="[contest-CUDA T4]",
            measurement_hardware="linux_x86_64_t4",
        )
        cov = compute_master_gradient_wire_in_coverage(
            [existing_sha, "4" * 64, "5" * 64, "6" * 64], ledger_path=ledger
        )
        assert cov["total_archives"] == 4
        assert cov["archives_with_anchor"] == 1
        assert cov["archives_without_anchor"] == 3
        assert cov["coverage_pct"] == 25.0

    def test_compute_coverage_empty_list(self):
        cov = compute_master_gradient_wire_in_coverage([])
        assert cov["total_archives"] == 0
        assert cov["coverage_pct"] == 0.0
        assert cov["authoritative_coverage_pct"] == 0.0

    def test_compute_coverage_authoritative_distinguishes_advisory_from_contest(
        self, tmp_path
    ):
        # macOS-CPU advisory is NOT contest-authoritative per Catalog #127
        advisory_sha = "7" * 64
        ledger = _write_synthetic_anchor_ledger(
            tmp_path,
            advisory_sha,
            measurement_axis="[macOS-CPU advisory]",
            measurement_hardware="darwin_arm64_m5_max_macos_cpu_advisory",
        )
        cov = compute_master_gradient_wire_in_coverage(
            [advisory_sha], ledger_path=ledger
        )
        assert cov["coverage_pct"] == 100.0  # anchor exists
        assert cov["authoritative_coverage_pct"] == 0.0  # but is advisory, not authoritative

    def test_compute_coverage_payload_schema_version(self):
        cov = compute_master_gradient_wire_in_coverage([])
        assert cov["schema"] == MASTER_GRADIENT_WIRE_IN_SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# Live-state regression guards                                                 #
# --------------------------------------------------------------------------- #


class TestLiveStateRegressionGuards:
    def test_live_fec6_archive_anchor_exists(self):
        """Live anchor `f174192aeadf...` is the fec6 frontier archive
        per the inventory memo §A1.1. This regression guard pins the live
        ledger state at landing — if a future operator rotates the ledger
        (e.g. quarantine the macOS-CPU advisory row + add a Linux x86_64
        anchor), this test will surface the change as a signal-loss event
        per CLAUDE.md "no signal loss" discipline.
        """
        from tac.master_gradient import latest_anchor_for_archive

        # Use the live ledger (no path override).
        anchor = latest_anchor_for_archive(LIVE_FEC6_ARCHIVE_SHA)
        # At landing: the live anchor exists as [macOS-CPU advisory].
        # If the anchor disappears, this test fails loud per CLAUDE.md
        # "Bugs must be permanently fixed AND self-protected against".
        assert anchor is not None
        # Per CLAUDE.md "Apples-to-apples evidence discipline": the live
        # axis is macOS-CPU advisory at landing per the canonical ledger
        # state on disk.
        # NOTE: this assertion is permissive — any axis is accepted because
        # operator may rotate to Linux x86_64 [contest-CPU] without warning.
        assert anchor.get("measurement_axis") is not None
