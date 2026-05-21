# SPDX-License-Identifier: MIT
"""Tests for the pair #4 procedural seed orthogonality smoke."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = (
    REPO_ROOT
    / "tools"
    / "run_magic_codec_pair_4_procedural_seed_orthogonality_smoke.py"
)


def _load_tool_module():
    spec = importlib.util.spec_from_file_location(
        "run_magic_codec_pair_4_procedural_seed_orthogonality_smoke",
        TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


SMOKE = _load_tool_module()


def test_tool_exports_pair4_surfaces():
    assert hasattr(SMOKE, "run_smoke")
    assert hasattr(SMOKE, "evaluate_seed_case")
    assert hasattr(SMOKE, "evaluate_ordered_seed")
    assert hasattr(SMOKE, "apply_ordering")
    assert hasattr(SMOKE, "deterministic_seed_bytes")
    assert hasattr(SMOKE, "ORDERINGS")
    assert SMOKE.PAIR_4_PREDICTED_DELTA_S == 0.0


def test_ordering_dimension_has_reversible_and_non_free_controls():
    reversible = [o.name for o in SMOKE.ORDERINGS if o.compliance == "reversible_free"]
    non_free = [o.name for o in SMOKE.ORDERINGS if o.compliance == "non_free_control"]
    assert reversible == [
        "identity",
        "reverse",
        "even_then_odd",
        "odd_then_even",
        "adjacent_pair_swap",
        "rotate_left_half",
    ]
    assert non_free == ["sorted_ascending", "sorted_descending"]


def test_apply_ordering_preserves_seed_length():
    seed = bytes(range(32))
    for ordering in SMOKE.ORDERINGS:
        ordered = SMOKE.apply_ordering(seed, ordering.name)
        assert isinstance(ordered, bytes)
        assert len(ordered) == len(seed)
    assert SMOKE.apply_ordering(seed, "reverse") == seed[::-1]
    assert SMOKE.apply_ordering(seed, "even_then_odd") == seed[0::2] + seed[1::2]


def test_canonical_uniform_seed_raw_dominates_reversible_orderings():
    seed = SMOKE.deterministic_seed_bytes(32, "unit_pair4_seed_len_32")
    case = SMOKE.evaluate_seed_case(
        seed_bytes=seed,
        seed_label="unit_pair4_seed_len_32",
        seed_class="canonical_uniform_seed",
    )
    assert case["seed_len"] == 32
    assert case["n_reversible_free_orderings"] == 6
    assert case["all_reversible_free_orderings_raw_seed_dominates"] is True
    assert case["min_reversible_free_best_nonraw_delta_vs_raw_bytes"] == 4
    verdict_rows = [
        r for r in case["ordering_rows"] if r["compliant_for_pair4_verdict"]
    ]
    assert len(verdict_rows) == 6
    assert all(r["best_codec_name"] == "raw_seed" for r in verdict_rows)
    assert all(r["best_nonraw_codec_name"] == "brotli_q11_seed_bytes" for r in verdict_rows)


def test_structured_controls_are_excluded_from_pair4_verdict_but_detect_compression():
    seed = SMOKE.structured_control_seed_bytes(64, "all_zero")
    case = SMOKE.evaluate_seed_case(
        seed_bytes=seed,
        seed_label="control_all_zero_len_64",
        seed_class="structured_negative_control",
    )
    assert case["all_reversible_free_orderings_raw_seed_dominates"] is None
    assert case["min_reversible_free_best_nonraw_delta_vs_raw_bytes"] is None
    assert not any(r["compliant_for_pair4_verdict"] for r in case["ordering_rows"])
    assert any(
        r["best_nonraw_delta_vs_raw_ordered_bytes"] < 0
        for r in case["ordering_rows"]
    )


def test_run_smoke_pair4_boundary_validated_on_canonical_seed_rows():
    result = SMOKE.run_smoke(seed_lengths=(16, 32), include_structured_controls=True)
    assert result["cascade_verdict"] == "PAIR_4_BOUNDARY_VALIDATED_RAW_SEED_DOMINATES"
    assert result["n_canonical_seed_cases"] == 2
    assert result["n_canonical_reversible_ordering_rows"] == 12
    assert result["n_canonical_reversible_ordering_rows_raw_seed_dominates"] == 12
    assert result["min_canonical_reversible_best_nonraw_delta_vs_raw_bytes"] == 4
    assert result["empirical_delta_s_pair_4"] == 0.0
    assert "selected raw-fallback delta" in result["empirical_delta_s_pair_4_interpretation"]
    assert result["best_nonraw_rate_regression_delta_s_pair_4"] > 0.0
    assert result["score_claim"] is False
    assert result["score_claim_valid"] is False
    assert result["promotion_eligible"] is False
    assert result["ready_for_exact_eval_dispatch"] is False
    assert result["rank_or_kill_eligible"] is False
    assert result["promotable"] is False


def test_main_writes_json_and_markdown(tmp_path):
    rc = SMOKE.main(
        [
            "--seed-lengths",
            "16",
            "32",
            "--output-dir",
            str(tmp_path),
            "--no-structured-controls",
        ]
    )
    assert rc == 0
    json_path = tmp_path / "smoke_result.json"
    md_path = tmp_path / "smoke_result.md"
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["cascade_verdict"] == "PAIR_4_BOUNDARY_VALIDATED_RAW_SEED_DOMINATES"
    assert payload["n_canonical_reversible_ordering_rows"] == 12
    assert payload["score_claim"] is False
    assert payload["best_nonraw_rate_regression_delta_s_pair_4"] > 0.0
    assert "ordering_dimension" in payload
    assert "codec_dimensions" in payload
    assert "raw seed is the rate floor" in md_path.read_text()


def test_main_rejects_non_positive_seed_length(tmp_path):
    rc = SMOKE.main(["--seed-lengths", "0", "--output-dir", str(tmp_path)])
    assert rc == 2
