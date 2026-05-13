from __future__ import annotations

import json

from tac.optimization.cooperative_receiver_campaigns import (
    build_campaign_queue,
    render_markdown,
    write_campaign_queue,
)
from tac.optimization.proxy_candidate_contract import validate_proxy_candidate


def test_campaign_queue_contains_four_team_convergence_and_blocks_dispatch() -> None:
    manifest = build_campaign_queue()

    assert manifest["schema"] == "tac_cooperative_receiver_campaign_queue_v1"
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["dispatch_ready_count"] == 0
    assert manifest["source_commits"] == [
        "1d62a114",
        "27cd8b41",
        "cbc6b48b",
        "d1fb9f6a",
        "fdfc347f",
        "local_dpw1_substrate",
    ]

    rows = manifest["top_k"]
    assert [row["campaign_id"] for row in rows[:7]] == [
        "darts_confirmed_time_traveler_config",
        "time_traveler_world_model_substrate",
        "sabor_boundary_only_renderer",
        "s2sbs_hf_byte_stuffing",
        "driving_prior_pretrained_renderer_2032",
        "driving_prior_world_model_substrate",
        "h15_coord_mlp_residual_sidecar_pr103_on_pr106",
    ]
    assert "tools/probe_driving_prior_readiness.py" in rows[4]["timing_smoke_command"]
    assert "driving_prior_world_model/tests" in rows[5]["timing_smoke_command"]
    assert "tools/probe_coord_mlp_residual_sidecar.py" in rows[6]["timing_smoke_command"]
    assert len(rows) == 14
    assert rows[-2]["campaign_id"] == "a1_plus_lapose_composition"
    assert rows[-1]["campaign_id"] == "a1_plus_wavelet_residual_retarget"
    assert all(validate_proxy_candidate(row) == [] for row in rows)
    assert all(row["ready_for_exact_eval_dispatch"] is False for row in rows)
    assert all(row["score_claim"] is False for row in rows)


def test_campaign_queue_markdown_and_writer_are_deterministic(tmp_path) -> None:
    json_path = tmp_path / "campaign_queue.json"
    md_path = tmp_path / "campaign_queue.md"

    manifest = write_campaign_queue(json_path, markdown_output=md_path, top_k=2)
    loaded = json.loads(json_path.read_text(encoding="utf-8"))

    assert loaded == manifest
    assert [row["rank_hint"] for row in loaded["top_k"]] == [1, 2]
    markdown = md_path.read_text(encoding="utf-8")
    assert markdown == render_markdown(manifest)
    assert "time_traveler_world_model_substrate" in markdown
    assert "score_claim: `false`" in markdown
