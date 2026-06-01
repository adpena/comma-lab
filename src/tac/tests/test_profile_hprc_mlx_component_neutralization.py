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
