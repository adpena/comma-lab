from __future__ import annotations

import json

from tac.optimization.cooperative_receiver_integration import (
    build_integration_manifest,
    render_markdown,
    write_integration_manifest,
)
from tac.optimization.proxy_candidate_contract import validate_proxy_candidate


def test_integration_manifest_threads_campaigns_into_solver_surfaces() -> None:
    manifest = build_integration_manifest()

    assert manifest["schema"] == "tac_cooperative_receiver_solver_integration_v1"
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["campaign_count"] == 14

    autopilot_rows = manifest["autopilot_dispatch_hook"]["rows"]
    assert len(autopilot_rows) == 14
    assert all(validate_proxy_candidate(row) == [] for row in autopilot_rows)
    assert {
        row["source_campaign_id"]
        for row in autopilot_rows
        if row["source_campaign_id"]
        in {
            "time_traveler_world_model_substrate",
            "sabor_boundary_only_renderer",
            "s2sbs_hf_byte_stuffing",
            "h15_coord_mlp_residual_sidecar_pr103_on_pr106",
            "driving_prior_world_model_substrate",
            "a1_plus_lapose_composition",
            "a1_plus_wavelet_residual_retarget",
        }
    } == {
        "time_traveler_world_model_substrate",
        "sabor_boundary_only_renderer",
        "s2sbs_hf_byte_stuffing",
        "h15_coord_mlp_residual_sidecar_pr103_on_pr106",
        "driving_prior_world_model_substrate",
        "a1_plus_lapose_composition",
        "a1_plus_wavelet_residual_retarget",
    }

    meta = manifest["meta_lagrangian_hook"]
    assert len(meta["rows"]) == 14
    assert meta["score_claim"] is False
    assert all(row["proxy_row"] is True for row in meta["rows"])
    assert all(row["ready_for_exact_eval_dispatch"] is False for row in meta["rows"])

    pareto = manifest["pareto_constraint_hook"]
    assert pareto["binding"] is False
    assert len(pareto["rows"]) == 14
    assert all(row["pareto_eligible"] is False for row in pareto["rows"])

    continual = manifest["continual_learning_hook"]
    assert continual["posterior_update_allowed"] is False
    assert "byte_closed_archive_sha256" in continual["required_before_update"]


def test_integration_manifest_wires_xray_magic_codec_and_compiler_hooks() -> None:
    manifest = build_integration_manifest()

    xray_rows = manifest["xray_hook"]["cooperative_receiver_packet_grammars"]
    assert {
        "TT5L",
        "SBO1",
        "S2SB",
        "CMLR",
        "DPW1",
    }.issubset({row["magic_ascii"] for row in xray_rows})

    magic_codec_rows = manifest["magic_codec_hook"]["entries"]
    assert {"OWV2", "OWV3", "IMPS"}.issubset(
        {row["magic_ascii"] for row in magic_codec_rows}
    )

    compiler = manifest["packet_compiler_hook"]
    assert compiler["canonical_module"] == "tac.packet_compiler.deterministic_compiler"
    assert compiler["score_claim"] is False
    assert compiler["ready_for_exact_eval_dispatch"] is False
    assert compiler["required_order"] == [
        "representation",
        "prediction",
        "quantization",
        "hyperprior",
        "arithmetic",
        "pack",
    ]


def test_integration_writer_and_markdown_are_deterministic(tmp_path) -> None:
    json_path = tmp_path / "integration_manifest.json"
    md_path = tmp_path / "integration_manifest.md"

    manifest = write_integration_manifest(json_path, markdown_output=md_path)
    loaded = json.loads(json_path.read_text(encoding="utf-8"))

    assert loaded == manifest
    markdown = md_path.read_text(encoding="utf-8")
    assert markdown == render_markdown(manifest)
    assert "Cooperative-Receiver Solver Integration" in markdown
    assert "`TT5L`" in markdown
