from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

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
