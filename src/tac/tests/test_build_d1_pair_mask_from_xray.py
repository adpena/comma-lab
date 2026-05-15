# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tac.repo_io import read_json
from tac.tests.tool_loader import load_repo_tool

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_tool():
    return load_repo_tool(
        REPO_ROOT,
        "tools/build_d1_pair_mask_from_xray.py",
        "build_d1_pair_mask_from_xray_test",
    )


def _write_xray(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "pair_component_error_xray_v1",
                "evidence_grade": "diagnostic_pair_component_xray_cpu",
                "device": "cpu",
                "n_pairs": len(rows),
                "rows": rows,
                "score_claim": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_d1_pair_mask_from_xray_selects_only_improving_pairs(tmp_path: Path) -> None:
    module = _load_tool()
    baseline = tmp_path / "baseline.json"
    positive = tmp_path / "positive.json"
    negative = tmp_path / "negative.json"
    _write_xray(
        baseline,
        [
            {"pair_idx": 0, "pose_dist": 0.01, "seg_dist": 0.010},
            {"pair_idx": 1, "pose_dist": 0.01, "seg_dist": 0.010},
            {"pair_idx": 2, "pose_dist": 0.01, "seg_dist": 0.010},
        ],
    )
    _write_xray(
        positive,
        [
            {"pair_idx": 0, "pose_dist": 0.01, "seg_dist": 0.008},
            {"pair_idx": 1, "pose_dist": 0.01, "seg_dist": 0.012},
            {"pair_idx": 2, "pose_dist": 0.01, "seg_dist": 0.009},
        ],
    )
    _write_xray(
        negative,
        [
            {"pair_idx": 0, "pose_dist": 0.01, "seg_dist": 0.011},
            {"pair_idx": 1, "pose_dist": 0.01, "seg_dist": 0.007},
            {"pair_idx": 2, "pose_dist": 0.01, "seg_dist": 0.0105},
        ],
    )
    output = tmp_path / "mask.json"

    rc = module.main(
        [
            "--baseline-xray",
            str(baseline),
            "--positive-xray",
            str(positive),
            "--negative-xray",
            str(negative),
            "--improvement-guard",
            "0.02",
            "--evidence-axis",
            "local_cpu_xray",
            "--baseline-archive-bytes",
            "1000",
            "--candidate-archive-bytes",
            "1000",
            "--output-n-pairs",
            "5",
            "--output-json",
            str(output),
        ]
    )

    assert rc == 0
    payload = read_json(output)
    assert payload["pair_signs"] == [1, -1, 1, 0, 0]
    assert payload["measured_pairs"] == 3
    assert payload["active_pairs"] == 3
    assert payload["positive_pairs"] == 2
    assert payload["negative_pairs"] == 1
    assert payload["objective"] == "contest_score_linearized_at_baseline_mean_pose_v1"
    assert payload["selection_mode"] == "waterfill_prefix"
    assert payload["potential_pairs"] == 3
    assert payload["best_prefix_size"] == 3
    assert payload["best_component_prefix_size"] == 3
    assert payload["predicted_component_no_rate_delta"] < 0.0
    assert payload["predicted_total_delta_with_rate"] < 0.0
    assert payload["predicted_score_lowering_after_rate"] is True
    assert {row["selection_rank"] for row in payload["selected_pairs"]} == {1, 2, 3}
    assert payload["score_claim"] is False
    assert payload["evidence_axis_label"] == "[local-CPU xray]"
    assert len(payload["deterministic_provenance_sha256"]) == 64
    assert payload["compressed_rate_accounting"]["formula"] == "25 * byte_delta / 37545489"
    assert payload["compressed_rate_accounting"]["evidence_axis_label"] == (
        "[local-CPU xray]"
    )
    assert payload["pair_mask_custody"]["custody_scope"] == "custom_length_selector"
    assert payload["pair_mask_custody"]["packed_raw_bytes"] == 2
    assert payload["pair_mask_custody"]["score_bearing_runtime_keys"] == [
        "pair_mask_b85",
        "pair_mask_n",
    ]


def test_build_d1_pair_mask_from_xray_blocks_mask_when_rate_cost_dominates(
    tmp_path: Path,
) -> None:
    module = _load_tool()
    baseline = tmp_path / "baseline.json"
    positive = tmp_path / "positive.json"
    _write_xray(
        baseline,
        [
            {"pair_idx": 0, "pose_dist": 0.01, "seg_dist": 0.010},
            {"pair_idx": 1, "pose_dist": 0.01, "seg_dist": 0.010},
            {"pair_idx": 2, "pose_dist": 0.01, "seg_dist": 0.010},
        ],
    )
    _write_xray(
        positive,
        [
            {"pair_idx": 0, "pose_dist": 0.01, "seg_dist": 0.00999},
            {"pair_idx": 1, "pose_dist": 0.01, "seg_dist": 0.00999},
            {"pair_idx": 2, "pose_dist": 0.01, "seg_dist": 0.00999},
        ],
    )
    output = tmp_path / "mask.json"

    rc = module.main(
        [
            "--baseline-xray",
            str(baseline),
            "--positive-xray",
            str(positive),
            "--evidence-axis",
            "local_cpu_xray",
            "--incremental-rate-cost-bytes",
            "400000",
            "--incremental-baseline-label",
            "unit-test-d1-static-baseline",
            "--expected-pairs",
            "3",
            "--output-json",
            str(output),
        ]
    )

    assert rc == 0
    payload = read_json(output)
    assert payload["potential_pairs"] == 3
    assert payload["active_pairs"] == 0
    assert payload["best_prefix_size"] == 0
    assert payload["best_component_prefix_size"] == 3
    assert payload["best_component_no_rate_delta"] < 0.0
    assert payload["pair_signs"] == [0, 0, 0]
    assert payload["selected_pairs"] == []
    assert payload["predicted_component_no_rate_delta"] == 0.0
    assert payload["predicted_total_delta_with_rate"] == 0.0
    assert payload["predicted_score_lowering_after_rate"] is False
    assert (
        payload["pair_mask_custody"]["custody_scope"]
        == "full_measured_contest_selector"
    )
    assert payload["compressed_rate_accounting"]["source"] == (
        "same_family_incremental_selector_bytes"
    )


def test_build_d1_pair_mask_from_xray_blocks_a1_relative_full_rate_cost(
    tmp_path: Path,
) -> None:
    module = _load_tool()
    baseline = tmp_path / "baseline.json"
    positive = tmp_path / "positive.json"
    _write_xray(
        baseline,
        [
            {"pair_idx": 0, "pose_dist": 0.01, "seg_dist": 0.010},
            {"pair_idx": 1, "pose_dist": 0.01, "seg_dist": 0.010},
            {"pair_idx": 2, "pose_dist": 0.01, "seg_dist": 0.010},
        ],
    )
    _write_xray(
        positive,
        [
            {"pair_idx": 0, "pose_dist": 0.01, "seg_dist": 0.00999},
            {"pair_idx": 1, "pose_dist": 0.01, "seg_dist": 0.00999},
            {"pair_idx": 2, "pose_dist": 0.01, "seg_dist": 0.00999},
        ],
    )
    output = tmp_path / "mask.json"

    rc = module.main(
        [
            "--baseline-xray",
            str(baseline),
            "--positive-xray",
            str(positive),
            "--evidence-axis",
            "local_cpu_xray",
            "--baseline-archive-bytes",
            "178162",
            "--candidate-archive-bytes",
            "185593",
            "--expected-pairs",
            "3",
            "--output-json",
            str(output),
        ]
    )

    assert rc == 0
    payload = read_json(output)
    assert payload["evidence_axis"] == "local_cpu_xray"
    assert payload["rate_scope"] == "archive_delta"
    assert payload["archive_byte_delta"] == 7431
    assert payload["potential_pairs"] == 3
    assert payload["best_component_prefix_size"] == 3
    assert payload["best_component_no_rate_delta"] < 0.0
    assert payload["active_pairs"] == 0
    assert payload["pair_signs"] == [0, 0, 0]
    assert payload["predicted_score_lowering_after_rate"] is False


def test_build_d1_pair_mask_from_xray_rejects_axis_mismatch(tmp_path: Path) -> None:
    module = _load_tool()
    baseline = tmp_path / "baseline.json"
    positive = tmp_path / "positive.json"
    rows = [
        {"pair_idx": 0, "pose_dist": 0.01, "seg_dist": 0.010},
        {"pair_idx": 1, "pose_dist": 0.01, "seg_dist": 0.010},
    ]
    _write_xray(baseline, rows)
    _write_xray(positive, rows)
    output = tmp_path / "mask.json"

    try:
        module.main(
            [
                "--baseline-xray",
                str(baseline),
                "--positive-xray",
                str(positive),
                "--evidence-axis",
                "contest_cuda",
                "--baseline-archive-bytes",
                "1000",
                "--candidate-archive-bytes",
                "1001",
                "--expected-pairs",
                "2",
                "--output-json",
                str(output),
            ]
        )
    except SystemExit as exc:
        assert "does not match xray provenance axes" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("axis mismatch should exit")


def test_build_d1_pair_mask_from_xray_uses_candidate_manifest_rate(
    tmp_path: Path,
) -> None:
    module = _load_tool()
    baseline = tmp_path / "baseline.json"
    positive = tmp_path / "positive.json"
    _write_xray(
        baseline,
        [
            {"pair_idx": 0, "pose_dist": 0.01, "seg_dist": 0.010},
            {"pair_idx": 1, "pose_dist": 0.01, "seg_dist": 0.010},
        ],
    )
    _write_xray(
        positive,
        [
            {"pair_idx": 0, "pose_dist": 0.01, "seg_dist": 0.00999},
            {"pair_idx": 1, "pose_dist": 0.01, "seg_dist": 0.00999},
        ],
    )
    manifest = tmp_path / "candidate_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "candidate_id": "d1_pair_mask_unit",
                "archive_bytes": 185593,
                "archive_sha256": "a" * 64,
                "source_base_archive_bytes": 178162,
                "base_member_sha256": "b" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "mask.json"

    rc = module.main(
        [
            "--baseline-xray",
            str(baseline),
            "--positive-xray",
            str(positive),
            "--evidence-axis",
            "local_cpu_xray",
            "--rate-from-candidate-manifest",
            str(manifest),
            "--expected-pairs",
            "2",
            "--output-json",
            str(output),
        ]
    )

    assert rc == 0
    payload = read_json(output)
    assert payload["rate_scope"] == "archive_delta"
    assert payload["baseline_archive_bytes"] == 178162
    assert payload["candidate_archive_bytes"] == 185593
    assert payload["archive_byte_delta"] == 7431
    assert payload["rate_source_manifest"]["candidate_id"] == "d1_pair_mask_unit"
    assert payload["rate_source_manifest"]["rate_accounting_source"] == (
        "candidate_manifest_compressed_archive_bytes_v1"
    )
    assert payload["compressed_rate_accounting"]["source"] == "candidate_manifest"
    assert payload["compressed_rate_accounting"]["archive_byte_delta"] == 7431
    assert payload["active_pairs"] == 0


def test_build_d1_pair_mask_from_xray_rejects_manifest_without_source_base_bytes(
    tmp_path: Path,
) -> None:
    module = _load_tool()
    baseline = tmp_path / "baseline.json"
    positive = tmp_path / "positive.json"
    rows = [{"pair_idx": 0, "pose_dist": 0.01, "seg_dist": 0.010}]
    _write_xray(baseline, rows)
    _write_xray(positive, rows)
    manifest = tmp_path / "candidate_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "candidate_id": "d1_pair_mask_legacy",
                "archive_bytes": 185593,
                "base_member_bytes": 178162,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        module.main(
            [
                "--baseline-xray",
                str(baseline),
                "--positive-xray",
                str(positive),
                "--evidence-axis",
                "local_cpu_xray",
                "--rate-from-candidate-manifest",
                str(manifest),
                "--expected-pairs",
                "1",
                "--output-json",
                str(tmp_path / "mask.json"),
            ]
        )
    except ValueError as exc:
        assert "source_base_archive_bytes" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("manifest full-rate fallback should fail closed")


def test_build_d1_pair_mask_from_xray_rejects_nonfinite_component_rows(
    tmp_path: Path,
) -> None:
    module = _load_tool()
    baseline = tmp_path / "baseline.json"
    positive = tmp_path / "positive.json"
    _write_xray(
        baseline,
        [{"pair_idx": 0, "pose_dist": float("nan"), "seg_dist": 0.010}],
    )
    _write_xray(
        positive,
        [{"pair_idx": 0, "pose_dist": 0.01, "seg_dist": 0.009}],
    )

    try:
        module.main(
            [
                "--baseline-xray",
                str(baseline),
                "--positive-xray",
                str(positive),
                "--evidence-axis",
                "local_cpu_xray",
                "--baseline-archive-bytes",
                "1000",
                "--candidate-archive-bytes",
                "1001",
                "--expected-pairs",
                "1",
                "--output-json",
                str(tmp_path / "mask.json"),
            ]
        )
    except ValueError as exc:
        assert "pose_dist must be finite" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("nonfinite xray rows should fail closed")


def test_build_d1_pair_mask_from_xray_rejects_negative_archive_delta_without_rationale(
    tmp_path: Path,
) -> None:
    module = _load_tool()
    baseline = tmp_path / "baseline.json"
    positive = tmp_path / "positive.json"
    rows = [{"pair_idx": 0, "pose_dist": 0.01, "seg_dist": 0.010}]
    _write_xray(baseline, rows)
    _write_xray(positive, rows)
    output = tmp_path / "mask.json"

    try:
        module.main(
            [
                "--baseline-xray",
                str(baseline),
                "--positive-xray",
                str(positive),
                "--evidence-axis",
                "local_cpu_xray",
                "--baseline-archive-bytes",
                "1000",
                "--candidate-archive-bytes",
                "999",
                "--expected-pairs",
                "1",
                "--output-json",
                str(output),
            ]
        )
    except ValueError as exc:
        assert "negative archive deltas require" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("negative archive delta should fail without rationale")
