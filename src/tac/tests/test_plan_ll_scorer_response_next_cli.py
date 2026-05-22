from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _mlx_dataset() -> dict:
    false_authority = {
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    return {
        "schema": "scorer_response_dataset.v1",
        "summary": {"row_count": 1},
        "rows": [
            {
                "schema": "scorer_response_row.v1",
                "row_id": "mlx-row-1",
                "family": "mlx_scorer_response",
                "delta_vs_baseline_score": 1.0e-6,
                "scorer_delta_vs_baseline": 1.0e-6,
                "byte_budget_margin_vs_break_even": None,
                **false_authority,
            }
        ],
    }


def _failed_mlx_parity_sweep() -> dict:
    return {
        "schema_version": "mlx_scorer_torch_parity_sweep.v1",
        "run_id": "unit",
        "verdict": "FAIL_MLX_TORCH_SCORER_PARITY_SWEEP",
        "passed": False,
        "blockers": ["window_failed:index=39:pair_window=[156, 160]"],
        "evidence_grade": "macOS-MLX-research-signal",
        "evidence_tag": "[macOS-MLX research-signal]",
        "score_axis": "[macOS-MLX research-signal]",
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "device_type": "cpu",
        "window_count": 75,
        "covered_pair_window": [0, 300],
        "summary": {
            "passed_windows": 74,
            "failed_windows": 1,
            "posenet_output_abs_max": {"max": 7.62939453125e-06},
            "posenet_component_abs_max": {"max": 9.713470479344455e-12},
            "segnet_logit_abs_max": {"max": 0.0007913112640380859},
            "segnet_argmax_diff_pixels": {"max": 1.0},
            "segnet_argmax_diff_fraction": {"max": 1.2715657552083333e-06},
            "segnet_argmax_mismatch_pixels_total": 1,
        },
        "rows": [
            {
                "index": 39,
                "passed": False,
                "verdict": "FAIL_MLX_TORCH_SCORER_PARITY",
                "blockers": ["segnet_argmax_diff_pixels_exceeds_threshold:1>0"],
                "pair_window": [156, 160],
                "deltas": {
                    "posenet_output_abs_max": 7.62939453125e-06,
                    "posenet_component_abs_max": 9.713470479344455e-12,
                    "segnet_logit_abs_max": 0.0007913112640380859,
                    "segnet_argmax_diff_pixels": 1,
                    "segnet_argmax_diff_fraction": 1.2715657552083333e-06,
                },
            }
        ],
    }


def test_plan_ll_scorer_response_next_cli_accepts_null_byte_matrix(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    matrix_path = tmp_path / "null_byte_matrix.json"
    json_out = tmp_path / "plan.json"
    md_out = tmp_path / "plan.md"

    dataset_path.write_text(
        json.dumps({"schema": "scorer_response_dataset.v1", "summary": {}, "rows": []}),
        encoding="utf-8",
    )
    matrix_path.write_text(
        json.dumps(
            {
                "schema": "null_byte_master_gradient_probe_matrix_v1",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "axis_tag": "[predicted]",
                "n_anchors_probed_ok": 1,
                "top5_replacement_candidates": [
                    {
                        "substrate_label": "fec6",
                        "codec_family": "hnerv_family",
                        "scored_archive_sha256": "a" * 64,
                        "axis": "[contest-CUDA]",
                        "anchor_index": 1,
                        "n_null_bytes": 16292,
                        "null_fraction": 0.091,
                        "predicted_delta_s_per_seed_budget": {"K=16": -0.0108375},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_ll_scorer_response_next.py"),
            "--dataset",
            str(dataset_path),
            "--null-byte-matrix",
            str(matrix_path),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "ll_null_byte_procedural_codebook_candidates" in completed.stdout
    plan = json.loads(json_out.read_text(encoding="utf-8"))
    assert plan["probes"][0]["probe_id"] == "ll_null_byte_procedural_codebook_candidates"
    assert plan["probes"][0]["null_byte_priority_rows"][0]["priority_weight"] == 0.0108375
    assert "Null-Byte Matrix Priority" in md_out.read_text(encoding="utf-8")


def test_plan_ll_scorer_response_next_cli_accepts_legacy_null_byte_matrix_only_with_flag(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    matrix_path = tmp_path / "null_byte_matrix.json"
    json_out = tmp_path / "plan.json"

    dataset_path.write_text(
        json.dumps({"schema": "scorer_response_dataset.v1", "summary": {}, "rows": []}),
        encoding="utf-8",
    )
    matrix_path.write_text(
        json.dumps(
            {
                "schema": "null_byte_master_gradient_probe_matrix_v1",
                "score_claim": False,
                "promotable": False,
                "axis_tag": "[predicted]",
                "n_anchors_probed_ok": 1,
                "top5_replacement_candidates": [
                    {
                        "substrate_label": "legacy_fec6",
                        "codec_family": "hnerv_family",
                        "scored_archive_sha256": "a" * 64,
                        "axis": "[contest-CUDA]",
                        "anchor_index": 1,
                        "n_null_bytes": 16292,
                        "null_fraction": 0.091,
                        "predicted_delta_s_per_seed_budget": {"K=16": -0.0108375},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rejected = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_ll_scorer_response_next.py"),
            "--dataset",
            str(dataset_path),
            "--null-byte-matrix",
            str(matrix_path),
            "--json-out",
            str(json_out),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert rejected.returncode == 2
    assert "promotion_eligible must be explicit false" in rejected.stderr

    accepted = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_ll_scorer_response_next.py"),
            "--dataset",
            str(dataset_path),
            "--null-byte-matrix",
            str(matrix_path),
            "--allow-legacy-null-byte-matrix-missing-authority",
            "--json-out",
            str(json_out),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "ll_null_byte_procedural_codebook_candidates" in accepted.stdout
    plan = json.loads(json_out.read_text(encoding="utf-8"))
    assert set(
        plan["null_byte_priority_weights"]["legacy_missing_authority_fields_accepted"]
    ) == {
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "rank_or_kill_eligible",
    }


def test_plan_ll_scorer_response_next_cli_accepts_pair4_seed_boundary(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    boundary_path = tmp_path / "pair4_boundary.json"
    json_out = tmp_path / "plan.json"
    md_out = tmp_path / "plan.md"

    dataset_path.write_text(
        json.dumps({"schema": "scorer_response_dataset.v1", "summary": {}, "rows": []}),
        encoding="utf-8",
    )
    boundary_path.write_text(
        json.dumps(
            {
                "smoke_label": "wave_3_magic_codec_pair_4_procedural_seed_orthogonality_smoke",
                "smoke_pair_id": "pair_4_magic_codec_x_procedural_codebook_seed_bytes",
                "cascade_verdict": "PAIR_4_BOUNDARY_VALIDATED_RAW_SEED_DOMINATES",
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "n_canonical_reversible_ordering_rows": 30,
                "n_canonical_reversible_ordering_rows_raw_seed_dominates": 30,
                "min_canonical_reversible_best_nonraw_delta_vs_raw_bytes": 4,
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_ll_scorer_response_next.py"),
            "--dataset",
            str(dataset_path),
            "--magic-codec-seed-boundary-smoke",
            str(boundary_path),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "do_not_wrap_procedural_seed_bytes_with_magic_codec" in completed.stdout
    plan = json.loads(json_out.read_text(encoding="utf-8"))
    assert plan["magic_codec_seed_boundary"]["boundary_validated_raw_seed_dominates"] is True
    rules = {item["rule"] for item in plan["prohibitions"]}
    assert "do_not_wrap_procedural_seed_bytes_with_magic_codec" in rules
    assert "do_not_wrap_procedural_seed_bytes_with_magic_codec" in md_out.read_text(encoding="utf-8")


def test_plan_ll_scorer_response_next_cli_blocks_failed_mlx_parity_without_override(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    parity_path = tmp_path / "parity.json"
    json_out = tmp_path / "plan.json"
    md_out = tmp_path / "plan.md"
    dataset_path.write_text(json.dumps(_mlx_dataset()), encoding="utf-8")
    parity_path.write_text(json.dumps(_failed_mlx_parity_sweep()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_ll_scorer_response_next.py"),
            "--dataset",
            str(dataset_path),
            "--mlx-torch-parity-sweep",
            str(parity_path),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "do_not_use_mlx_rows_after_failed_strict_parity_sweep" in completed.stdout
    plan = json.loads(json_out.read_text(encoding="utf-8"))
    assert plan["mlx_torch_parity_sweep_gate"]["status"] == "blocked"
    assert plan["probes"][0]["probe_id"] == "ll_mlx_torch_parity_repair_or_override"
    assert "ll_mlx_cpu_stable_response_harvest" not in {
        probe["probe_id"] for probe in plan["probes"]
    }
    assert "MLX Torch Parity Gate" in md_out.read_text(encoding="utf-8")


def test_plan_ll_scorer_response_next_cli_allows_failed_mlx_parity_with_override(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    parity_path = tmp_path / "parity.json"
    json_out = tmp_path / "plan.json"
    md_out = tmp_path / "plan.md"
    dataset_path.write_text(json.dumps(_mlx_dataset()), encoding="utf-8")
    parity_path.write_text(json.dumps(_failed_mlx_parity_sweep()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_ll_scorer_response_next.py"),
            "--dataset",
            str(dataset_path),
            "--mlx-torch-parity-sweep",
            str(parity_path),
            "--allow-mlx-parity-research-signal-override",
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "ll_mlx_cpu_stable_response_harvest" in completed.stdout
    plan = json.loads(json_out.read_text(encoding="utf-8"))
    assert plan["mlx_torch_parity_sweep_gate"]["status"] == "research_signal_override"
    assert plan["probes"][0]["probe_id"] == "ll_mlx_cpu_stable_response_harvest"
    assert plan["probes"][0]["mlx_torch_parity_gate"]["score_claim"] is False
    assert "research_signal_override" in md_out.read_text(encoding="utf-8")
