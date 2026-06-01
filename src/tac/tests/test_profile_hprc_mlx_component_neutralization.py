from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "profile_hprc_mlx_component_neutralization.py"


def _load_tool_module():
    for path in (REPO, REPO / "tools"):
        path_s = str(path)
        if path_s not in sys.path:
            sys.path.insert(0, path_s)
    spec = spec_from_file_location("profile_hprc_mlx_component_neutralization", TOOL)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_full_video_mlx_gate_uses_rendered_pair_count_not_stale_partial_flag() -> None:
    module = _load_tool_module()
    assert (
        module._full_video_mlx_response_executed(  # pyright: ignore[reportPrivateUsage]
            cache_report={
                "raw_pair_count": 600,
                "cached_pair_count": 600,
                "local_acquisition_partial_raw": True,
            },
            max_pairs=600,
        )
        is True
    )
    assert (
        module._local_acquisition_partial_raw_blocker(  # pyright: ignore[reportPrivateUsage]
            {
                "raw_pair_count": 600,
                "cached_pair_count": 600,
                "local_acquisition_partial_raw": True,
            }
        )
        is False
    )
    assert (
        module._local_acquisition_partial_raw_blocker(  # pyright: ignore[reportPrivateUsage]
            {
                "raw_pair_count": 599,
                "cached_pair_count": 599,
                "local_acquisition_partial_raw": True,
            }
        )
        is True
    )


def test_shrink_backlog_prioritizes_distortion_nonworse_byte_cuts() -> None:
    module = _load_tool_module()
    backlog = module._build_shrink_backlog(  # pyright: ignore[reportPrivateUsage]
        {
            "schema": "profile.test",
            "section_value_rows": [
                {
                    "neutralized_section": "latents_rc",
                    "variant_id": "neutralize_latents_rc",
                    "marginal_status": "cut_candidate_value_below_rate_price",
                    "archive_bytes_removed_vs_baseline": 20_000,
                },
                {
                    "neutralized_section": "residual_rc",
                    "variant_id": "neutralize_residual_rc",
                    "marginal_status": "cut_candidate_distortion_nonworse",
                    "archive_bytes_removed_vs_baseline": 1_000_000,
                },
            ],
        }
    )
    assert backlog["rows"][0]["section"] == "residual_rc"
    assert (
        backlog["rows"][0]["next_materializer_task"]
        == "replace_raw_residual_grid_with_scorer_ranked_significance_and_learned_prior_coder"
    )


def test_shrink_backlog_emits_executable_pair_scoped_residual_candidate() -> None:
    module = _load_tool_module()
    backlog = module._build_shrink_backlog(  # pyright: ignore[reportPrivateUsage]
        {
            "schema": "profile.test",
            "archive_byte_ceiling": {"rate_price_score_per_byte": 0.01},
            "section_value_rows": [
                {
                    "neutralized_section": "residual_rc",
                    "variant_id": "residual_transform_threshold_abs_le_3",
                    "marginal_status": "cut_candidate_value_below_rate_price",
                    "archive_bytes_removed_vs_baseline": 300,
                    "delta_nonrate_score": 0.0,
                    "delta_total_mlx_score_advisory": 0.0,
                },
            ],
            "pair_value_rows": [
                {
                    "variant_id": "residual_transform_threshold_abs_le_3",
                    "pair_row": 0,
                    "delta_nonrate_score_pair_local": -0.1,
                },
                {
                    "variant_id": "residual_transform_threshold_abs_le_3",
                    "pair_row": 1,
                    "delta_nonrate_score_pair_local": 10.0,
                },
                {
                    "variant_id": "residual_transform_threshold_abs_le_3",
                    "pair_row": 2,
                    "delta_nonrate_score_pair_local": 0.0,
                },
            ],
        }
    )

    rows = backlog["pair_scoped_residual_candidate_rows"]
    assert len(rows) == 1
    assert rows[0]["residual_transform"] == "threshold_abs_le_pairs=3@0,2"
    assert rows[0]["selected_pair_count"] == 2
    assert rows[0]["protected_pair_count"] == 1
    assert rows[0]["score_claim"] is False


def test_hprc_profile_defaults_to_direct_cache_materialization() -> None:
    module = _load_tool_module()
    variant = module.VariantSpec(  # pyright: ignore[reportPrivateUsage]
        variant_id="baseline",
        neutralized_section=None,
        archive_zip_path=Path("candidate/archive.zip"),
        submission_dir=Path("candidate/submission"),
        hprc_bin_path=Path("candidate/0.bin"),
        archive_bytes=123,
        archive_sha256="abc",
        hprc_0bin_sha256="def",
        variant_dir=Path("candidate"),
    )

    direct_cmd = module._build_mlx_cache_materialization_command(  # pyright: ignore[reportPrivateUsage]
        tool=Path("tools/materialize_mlx_scorer_cache_from_submission.py"),
        variant=variant,
        cache_dir=Path("cache"),
        work_dir=Path("work"),
        report_output=Path("report.json"),
        max_pairs=600,
        inflate_timeout=1800,
        cache_materialization_mode="hprc-direct",
    )
    shell_cmd = module._build_mlx_cache_materialization_command(  # pyright: ignore[reportPrivateUsage]
        tool=Path("tools/materialize_mlx_scorer_cache_from_submission.py"),
        variant=variant,
        cache_dir=Path("cache"),
        work_dir=Path("work"),
        report_output=Path("report.json"),
        max_pairs=600,
        inflate_timeout=1800,
        cache_materialization_mode="shell-inflate",
    )

    assert "--hprc-direct-cache" in direct_cmd
    assert "--hprc-direct-cache" not in shell_cmd


def test_variant_slug_is_bounded_and_deterministic_for_pair_scoped_plans() -> None:
    module = _load_tool_module()
    raw = "threshold_abs_le_pairs=3@" + ",".join(str(i) for i in range(600))
    slug_a = module._variant_slug(raw)  # pyright: ignore[reportPrivateUsage]
    slug_b = module._variant_slug(raw)  # pyright: ignore[reportPrivateUsage]

    assert slug_a == slug_b
    assert len(slug_a) <= 80
    assert slug_a.startswith("threshold_abs_le_pairs_3")


def test_baseline_reuse_validates_identity_and_retargets_rate(tmp_path: Path) -> None:
    module = _load_tool_module()
    ref_cache = tmp_path / "reference_cache"
    ref_cache.mkdir()
    components = tmp_path / "components"
    components.mkdir()
    pose = components / "posenet_distortion.npy"
    seg = components / "segnet_distortion.npy"
    np.save(pose, np.asarray([0.25, 0.5], dtype=np.float32))
    np.save(seg, np.asarray([0.01, 0.02], dtype=np.float32))
    response = tmp_path / "baseline_response.json"
    response.write_text(
        json.dumps(
            {
                "archive_size_bytes": 100,
                "avg_posenet_dist": 0.25,
                "avg_segnet_dist": 0.01,
                "canonical_score": 1.0,
                "components": {
                    "artifacts": {
                        "posenet_distortion": {"path": pose.as_posix()},
                        "segnet_distortion": {"path": seg.as_posix()},
                    }
                },
                "n_samples": 2,
                "score_claim": False,
                "promotion_eligible": False,
            }
        ),
        encoding="utf-8",
    )
    cache_report = tmp_path / "baseline_cache_report.json"
    cache_report.write_text('{"raw_pair_count": 2, "cached_pair_count": 2}\n', encoding="utf-8")
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "max_pairs": 2,
                "reference_cache_dir": ref_cache.as_posix(),
                "variant_rows": [
                    {
                        "variant_id": "baseline",
                        "hprc_0bin_sha256": "abc",
                        "mlx_response": response.as_posix(),
                        "cache_report": cache_report.as_posix(),
                    }
                ],
                "cache_materialization_rows": {
                    "baseline": {"cache_dir": (tmp_path / "baseline_cache").as_posix()}
                },
            }
        ),
        encoding="utf-8",
    )
    variant = module.VariantSpec(  # pyright: ignore[reportPrivateUsage]
        variant_id="baseline",
        neutralized_section=None,
        archive_zip_path=tmp_path / "archive.zip",
        submission_dir=tmp_path / "submission",
        hprc_bin_path=tmp_path / "0.bin",
        archive_bytes=200,
        archive_sha256="zip-sha",
        hprc_0bin_sha256="abc",
        variant_dir=tmp_path,
    )

    reuse = module._prepare_baseline_reuse(  # pyright: ignore[reportPrivateUsage]
        profile_path=profile,
        baseline_variant=variant,
        reference_cache_dir=ref_cache,
        max_pairs=2,
    )
    payload = module._copy_reused_baseline_response(  # pyright: ignore[reportPrivateUsage]
        baseline_reuse=reuse,
        baseline_variant=variant,
        output_dir=tmp_path / "out",
    )

    assert payload["archive_size_bytes"] == 200
    assert payload["canonical_score_source"].endswith("retargeted_archive_bytes")
    assert payload["baseline_reuse"]["current_archive_sha256"] == "zip-sha"
    copied_pose = payload["components"]["artifacts"]["posenet_distortion"]["path"]
    assert Path(copied_pose).is_file()
