from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path("experiments/build_problem_space_manifest.py")
    spec = importlib.util.spec_from_file_location("build_problem_space_manifest", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frontier_uses_canonical_score_without_rounding_or_missing_promotion_false(tmp_path: Path) -> None:
    module = _load_module()
    frontier_json = tmp_path / "frontier.json"
    payload = {
        "archive_size_bytes": 236328,
        "avg_posenet_dist": 0.0001894,
        "avg_segnet_dist": 0.00057185,
        "canonical_score": 0.25806611029397786,
        "canonical_score_source": "contest_auth_eval_json",
        "n_samples": 600,
        "provenance": {
            "archive_sha256": "a" * 64,
            "device": "cuda",
            "gpu_model": "Tesla T4",
        },
        "score_recomputed_from_components": 0.25806611029397786,
    }
    frontier_json.write_text(json.dumps(payload), encoding="utf-8")

    manifest = module.build_manifest(
        frontier_json=frontier_json,
        pr85_profile=None,
        pr86_profile=None,
    )

    frontier = manifest["current_frontier"]
    assert frontier["score"] == 0.25806611029397786
    assert frontier["score_source"] == "contest_auth_eval_json"
    assert frontier["promotion_eligible"] is None
    assert (
        frontier["promotion_eligible_source"]
        == "absent_in_source_json_not_interpreted_as_false"
    )
    assert frontier["score_from_visible_components"] != frontier["score"]
    assert frontier["visible_component_score_delta_vs_canonical"] > 0


def test_equation_constraints_record_rust_first_native_codec_policy(tmp_path: Path) -> None:
    module = _load_module()
    frontier_json = tmp_path / "frontier.json"
    frontier_json.write_text(
        json.dumps(
            {
                "archive_size_bytes": 1,
                "avg_posenet_dist": 0.0,
                "avg_segnet_dist": 0.0,
                "n_samples": 600,
                "provenance": {
                    "archive_sha256": "b" * 64,
                    "device": "cuda",
                    "gpu_model": "Tesla T4",
                },
                "score_recomputed_from_components": 25.0 / 37_545_489,
            }
        ),
        encoding="utf-8",
    )

    manifest = module.build_manifest(
        frontier_json=frontier_json,
        pr85_profile=None,
        pr86_profile=None,
    )

    constraints = manifest["equation_system"]["constraints"]
    assert any("prefer Rust over C++" in row for row in constraints)
