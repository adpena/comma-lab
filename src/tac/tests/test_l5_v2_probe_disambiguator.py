# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys

from tac.optimization.l5_v2_probe_disambiguator import (
    L5V2_CANDIDATES,
    L5V2_PROBE_SCHEMA,
    L5V2_PROBE_TOOL_PATH,
    L5V2ProbeObservation,
    build_probe_template,
    evaluate_l5_v2_probe,
)


def _eligible(candidate_id: str, delta: float) -> L5V2ProbeObservation:
    return L5V2ProbeObservation(
        candidate_id=candidate_id,
        predicted_or_measured_delta=delta,
        evidence_grade="contest_cpu_cuda_paired_exact",
        exact_axes=("contest_cpu", "contest_cuda"),
        archive_sha256="a" * 64,
        runtime_tree_sha256="b" * 64,
        sideinfo_consumed=True,
        byte_closed_archive=True,
    )


def test_l5_v2_probe_template_is_fail_closed_and_complete() -> None:
    template = build_probe_template()

    assert template["schema"] == L5V2_PROBE_SCHEMA
    assert template["score_claim"] is False
    assert template["promotion_eligible"] is False
    assert template["ready_for_exact_eval_dispatch"] is False
    assert [row["candidate_id"] for row in template["observations"]] == list(
        L5V2_CANDIDATES
    )


def test_l5_v2_probe_fails_closed_without_observations() -> None:
    verdict = evaluate_l5_v2_probe(())

    assert verdict["architecture_lock_allowed"] is False
    assert verdict["selected_candidate_id"] is None
    assert "l5_v2_probe_observations_missing" in verdict["blockers"]
    assert "l5_v2_probe_candidate_coverage_incomplete" in verdict["blockers"]
    assert "l5_v2_probe_no_eligible_candidate" in verdict["blockers"]
    assert verdict["score_claim"] is False


def test_l5_v2_probe_selects_best_paired_exact_candidate() -> None:
    verdict = evaluate_l5_v2_probe(
        (
            _eligible("c1_world_model_foveation", -0.010),
            _eligible("z5_predictive_coding_world_model", -0.020),
            _eligible("time_traveler_l5_autonomy", -0.030),
        )
    )

    assert verdict["architecture_lock_allowed"] is True
    assert verdict["selected_candidate_id"] == "time_traveler_l5_autonomy"
    assert verdict["selected_delta"] == -0.030
    assert verdict["blockers"] == []


def test_l5_v2_probe_blocks_proxy_or_unconsumed_observations() -> None:
    proxy = L5V2ProbeObservation(
        candidate_id="time_traveler_l5_autonomy",
        predicted_or_measured_delta=-0.050,
        evidence_grade="macos_cpu_advisory",
        exact_axes=("contest_cpu",),
        archive_sha256="a" * 64,
        runtime_tree_sha256="b" * 64,
        sideinfo_consumed=False,
        byte_closed_archive=True,
    )

    verdict = evaluate_l5_v2_probe((proxy,))
    row = verdict["evaluated_observations"][0]

    assert verdict["architecture_lock_allowed"] is False
    assert "l5_v2_probe_paired_exact_axes_missing" in row["blockers"]
    assert "l5_v2_probe_sideinfo_consumption_missing" in row["blockers"]
    assert "l5_v2_probe_contest_evidence_grade_missing" in row["blockers"]


def test_l5_v2_probe_cli_emits_template() -> None:
    result = subprocess.run(
        [sys.executable, L5V2_PROBE_TOOL_PATH, "--emit-template"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["schema"] == L5V2_PROBE_SCHEMA
    assert [row["candidate_id"] for row in payload["observations"]] == list(
        L5V2_CANDIDATES
    )
