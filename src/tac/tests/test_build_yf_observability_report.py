from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_PATH = REPO_ROOT / "experiments" / "build_yf_observability_report.py"


def _load_reporter():
    spec = importlib.util.spec_from_file_location("yf_observability_report_test", REPORT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_eval(
    path: Path,
    *,
    score: float,
    bytes_: int,
    seg: float,
    pose: float,
    gpu: str = "Tesla T4",
) -> Path:
    payload = {
        "archive_size_bytes": bytes_,
        "avg_posenet_dist": pose,
        "avg_segnet_dist": seg,
        "n_samples": 600,
        "provenance": {"archive_sha256": "a" * 64, "device": "cuda", "gpu_model": gpu},
        "score_pose_contribution": (10.0 * pose) ** 0.5,
        "score_rate_contribution": 25.0 * bytes_ / 37_545_489,
        "score_recomputed_from_components": score,
        "score_seg_contribution": 100.0 * seg,
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _write_profile(path: Path) -> Path:
    payload = {
        "aggregate_subspaces": {
            "classes": [
                {
                    "estimated_charged_bytes_sum": 90.0,
                    "estimated_lagrangian_net_proxy_sum": 0.09,
                    "estimated_marginal_score_saved_proxy_sum": 0.12,
                    "estimated_score_saved_per_charged_byte": 0.001333333333,
                    "hit_count": 7,
                    "key": "0",
                    "profile_hit_count": 2,
                    "residual_pixels_sum": 200,
                }
            ],
            "pairs": [
                {
                    "estimated_charged_bytes_sum": 48.0,
                    "estimated_lagrangian_net_proxy_sum": 0.11,
                    "estimated_marginal_score_saved_proxy_sum": 0.15,
                    "estimated_score_saved_per_charged_byte": 0.003125,
                    "hit_count": 6,
                    "key": "67",
                    "profile_hit_count": 2,
                    "residual_pixels_sum": 144,
                }
            ],
            "source_to_candidate": [
                {
                    "estimated_charged_bytes_sum": 48.0,
                    "estimated_lagrangian_net_proxy_sum": 0.11,
                    "estimated_marginal_score_saved_proxy_sum": 0.15,
                    "estimated_score_saved_per_charged_byte": 0.003125,
                    "hit_count": 6,
                    "key": "0->3",
                    "profile_hit_count": 2,
                    "residual_pixels_sum": 144,
                }
            ],
        },
        "evidence_grade": "planning_only",
        "fridrich_yousfi_signal_surface": {
            "low_dimensional_consensus": {
                "top_class_confusions": ["0->3"],
                "top_classes": ["0"],
                "top_pairs": [67],
            },
            "score_claim": False,
        },
        "profiles": [
            {
                "score_claim": False,
                "subspaces": {
                    "classes": [
                        {
                            "estimated_charged_bytes_sum": 30.0,
                            "estimated_lagrangian_net_proxy_sum": 0.01,
                            "estimated_marginal_score_saved_proxy_sum": 0.02,
                            "estimated_score_saved_per_charged_byte": 0.0006,
                            "hit_count": 3,
                            "key": "2",
                            "residual_pixels_sum": 99,
                        }
                    ],
                    "pairs": [
                        {
                            "estimated_charged_bytes_sum": 12.0,
                            "estimated_lagrangian_net_proxy_sum": 0.03,
                            "estimated_marginal_score_saved_proxy_sum": 0.04,
                            "estimated_score_saved_per_charged_byte": 0.003333333333,
                            "hit_count": 2,
                            "key": "69",
                            "residual_pixels_sum": 44,
                        }
                    ],
                    "source_to_candidate": [
                        {
                            "estimated_charged_bytes_sum": 12.0,
                            "estimated_lagrangian_net_proxy_sum": 0.03,
                            "estimated_marginal_score_saved_proxy_sum": 0.04,
                            "estimated_score_saved_per_charged_byte": 0.003333333333,
                            "hit_count": 2,
                            "key": "2->3",
                            "residual_pixels_sum": 44,
                        }
                    ],
                },
            }
        ],
        "schema": "atom_ledger_active_subspace_profile_v1",
        "score_claim": False,
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def test_observability_report_emits_control_plane_artifacts(tmp_path: Path) -> None:
    reporter = _load_reporter()
    baseline = _write_eval(tmp_path / "baseline.json", score=0.32, bytes_=276_000, seg=0.0006, pose=0.0005)
    collapse = _write_eval(tmp_path / "collapse.json", score=20.0, bytes_=250_000, seg=0.015, pose=50.0)
    profile = _write_profile(tmp_path / "profile.json")

    payload = reporter.build_report(
        eval_specs=[("baseline", baseline), ("collapse", collapse)],
        atom_profile_json=profile,
        output_dir=tmp_path / "report",
    )

    assert payload["schema"] == "yousfi_fridrich_observability_report_v1"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["target_gap_analysis"]["target_score"] == 0.3
    assert payload["target_gap_analysis"]["bytes_to_remove_if_distortion_unchanged"] > 0
    assert payload["atom_profile_summary"]["summary_source"] == "aggregate_subspaces"
    assert payload["atom_profile_summary"]["top_pairs"][0]["key"] == "67"
    assert payload["action_recommendations"][0]["action"] == "build_repaired_or_multimask_candidates"
    assert payload["exact_evals"][1]["fridrich_yousfi_signals"]["component_cliff"] is True
    assert payload["exact_evals"][1]["fridrich_yousfi_signals"]["byte_delta_vs_best"] == -26_000
    assert (tmp_path / "report" / "observability_report.json").exists()
    assert (tmp_path / "report" / "observability_report.md").read_text().startswith(
        "# Yousfi-Fridrich Observability Report"
    )
    assert "<svg" in (tmp_path / "report" / "score_breakdown.svg").read_text()
    assert "Need" in (tmp_path / "report" / "target_gap.svg").read_text()
    assert "pair 67" in (tmp_path / "report" / "top_pairs.svg").read_text()


def test_report_cross_checks_component_formula(tmp_path: Path) -> None:
    reporter = _load_reporter()
    eval_json = _write_eval(
        tmp_path / "eval.json",
        score=100.0 * 0.0006 + (10.0 * 0.0005) ** 0.5 + 25.0 * 276_000 / 37_545_489,
        bytes_=276_000,
        seg=0.0006,
        pose=0.0005,
    )

    payload = reporter.build_report(eval_specs=[("candidate", eval_json)], output_dir=tmp_path / "report")

    assert payload["exact_evals"][0]["score_formula_cross_check"]["matches_within_1e_5"] is True
