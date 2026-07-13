# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools/probe_segnet_exact_forward.py"
SPEC = importlib.util.spec_from_file_location("probe_segnet_exact_forward", TOOL)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_evenly_spaced_indices_has_finite_endpoints() -> None:
    assert probe.evenly_spaced_indices(64, 4) == (0, 21, 42, 63)
    assert probe.evenly_spaced_indices(64, 1) == (0,)


def test_parse_thread_candidates_includes_baseline_and_deduplicates() -> None:
    assert probe.parse_thread_candidates("4,1,4", 6) == (1, 4, 6)
    assert probe.parse_thread_candidates("auto", 6) == (1, 2, 3, 4, 5, 6)
    with pytest.raises(ValueError, match="positive integers"):
        probe.parse_thread_candidates("0,2", 6)


def test_select_thread_arm_rejects_flips_then_uses_median_and_thread_tiebreak() -> None:
    rows = [
        {"threads": 1, "argmax_flip_count": 2, "forward_ms_median": 1.0},
        {"threads": 2, "argmax_flip_count": 0, "forward_ms_median": 4.0},
        {"threads": 4, "argmax_flip_count": 0, "forward_ms_median": 4.0},
    ]
    assert probe.select_thread_arm(rows)["threads"] == 2


def test_select_thread_arm_fails_closed_when_every_candidate_flips() -> None:
    with pytest.raises(RuntimeError, match="no zero-flip"):
        probe.select_thread_arm(
            [{"threads": 1, "argmax_flip_count": 1, "forward_ms_median": 1.0}]
        )


def test_derive_economics_uses_matched_and_anchor_denominators() -> None:
    result = probe.derive_economics(
        cheap_ms=250.0,
        matched_control_ms=750.0,
        anchor_forward_ms=1650.0,
        anchor_backward_ms=450.0,
    )
    assert result["matched_forward_speedup_x"] == 3.0
    assert result["profile_anchor_scorer_slice_speedup_if_backward_unchanged_x"] == 3.0
    assert result["profile_anchor_yopo_ideal_speedup_if_backward_removed_x"] == 8.4


def test_admit_thread_arm_requires_distinct_exact_above_floor_controlled_arm() -> None:
    valid = {
        "selected_threads": 1,
        "baseline_threads": 6,
        "argmax_flip_count": 0,
        "cheap_median_ms": 250.0,
        "baseline_median_ms": 1_000.0,
        "composed_timing_noise_floor_ms": 500.0,
        "controls_passed": True,
    }
    assert probe.admit_thread_arm(**valid)

    for override in (
        {"selected_threads": 6},
        {"argmax_flip_count": 1},
        {"cheap_median_ms": 1_000.0},
        {"cheap_median_ms": 500.0, "composed_timing_noise_floor_ms": 500.0},
        {"controls_passed": False},
    ):
        assert not probe.admit_thread_arm(**(valid | override))


@pytest.mark.parametrize(
    "path",
    [Path("/tmp/r.json"), Path("/private/tmp/r.json"), Path("/private/var/folders/x/r.json")],
)
def test_validate_durable_output_refuses_transient_evidence(path: Path) -> None:
    with pytest.raises(ValueError, match="refusing transient"):
        probe.validate_durable_output(path)


def test_validate_durable_output_refuses_nonapproved_persistent_root() -> None:
    with pytest.raises(ValueError, match="approved Pact SSD"):
        probe.validate_durable_output(REPO_ROOT.parent / "outside-pact" / "r.json")


def _minimal_valid_report() -> dict:
    tool_sha = probe._sha256_file(TOOL)
    return {
        "verdict": "GO",
        "authority": {"score_claim": False, "pointer_moved": False},
        "control_law": {"selected_threads": 1, "baseline_threads": 6},
        "controls": {"passed": True},
        "measurement": {
            "n_real_pairs": 64,
            "total_argmax_pixels": 64 * probe.SEG_H * probe.SEG_W,
            "argmax_flip_count": 0,
            "argmax_flip_rate": 0.0,
            "argmax_bit_identical": True,
            "reference_argmax_sha256": "a" * 64,
            "candidate_argmax_sha256": "a" * 64,
            "baseline_forward": {
                "median_ms": 1_000.0,
                "p05_ms": 900.0,
                "p95_ms": 1_100.0,
            },
            "cheap_forward": {"median_ms": 250.0, "p05_ms": 225.0, "p95_ms": 275.0},
            "matched_speed_gap_ms": 750.0,
            "composed_timing_noise_floor_ms": 250.0,
            "speed_gap_exceeds_composed_floor": True,
        },
        "custody": {"tool_sha256": tool_sha},
    }


def test_validate_report_recomputes_persisted_admission_and_fields() -> None:
    report = _minimal_valid_report()
    probe.validate_report(report)

    mutations = (
        ("unconditional verdict", lambda row: row.__setitem__("verdict", "NO-GO")),
        (
            "same arm",
            lambda row: row["control_law"].__setitem__("selected_threads", 6),
        ),
        (
            "flip",
            lambda row: row["measurement"].__setitem__("argmax_flip_count", 1),
        ),
        (
            "below floor",
            lambda row: row["measurement"].__setitem__(
                "composed_timing_noise_floor_ms", 800.0
            ),
        ),
        ("failed controls", lambda row: row["controls"].__setitem__("passed", False)),
        (
            "wrong pixels",
            lambda row: row["measurement"].__setitem__("total_argmax_pixels", 1),
        ),
    )
    for _name, mutate in mutations:
        broken = deepcopy(report)
        mutate(broken)
        with pytest.raises(RuntimeError):
            probe.validate_report(broken)


def test_cli_requires_at_least_64_pairs() -> None:
    with pytest.raises(SystemExit):
        probe.parse_args(
            [
                "--raw",
                "experiments/results/x/0.raw",
                "--profile",
                "experiments/results/x/profile.json",
                "--n-pairs",
                "63",
                "--out",
                "experiments/results/x/r.json",
            ]
        )
