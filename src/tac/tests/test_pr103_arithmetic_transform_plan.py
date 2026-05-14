# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import brotli
import numpy as np
import pytest

from tac.hnerv_lowlevel_packer import write_stored_single_member_zip
from tac.hnerv_pr103_lc_ac_schema import (
    Pr103LcAcLayout,
    encode_pr103_merged_ac_stream,
)
from tac.pr103_arithmetic_transform_plan import (
    BEAM_SCHEMA,
    CANDIDATE_SCHEMA,
    COORDINATE_SCHEMA,
    GLOBAL_COMBO_SCHEMA,
    PLAN_SCHEMA,
    RETARGET_SCHEMA,
    Pr103ArithmeticTransformPlanError,
    build_pr103_arithmetic_histogram_beam_probe,
    build_pr103_arithmetic_histogram_coordinate_probe,
    build_pr103_arithmetic_histogram_global_combo_probe,
    build_pr103_arithmetic_retarget_probe,
    build_pr103_arithmetic_transform_plan,
    materialize_pr103_arithmetic_histogram_candidate,
    optimize_pr103_histogram_frontier_combinations,
)
from tac.repo_io import sha256_bytes, write_json


def test_pr103_transform_plan_defaults_to_top_target_and_blocks_dispatch() -> None:
    manifest = _manifest()

    plan = build_pr103_arithmetic_transform_plan(schema_manifest=manifest)

    assert plan["schema"] == PLAN_SCHEMA
    assert plan["planning_only"] is True
    assert plan["score_claim"] is False
    assert plan["dispatch_attempted"] is False
    assert plan["ready_for_archive_preflight"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert plan["target_stream"]["label"] == "stem.weight"
    assert plan["byte_accounting"]["expected_savings_bytes_upper_bound"] == 46
    assert plan["byte_accounting"]["expected_rate_score_delta_upper_bound"] < 0
    assert plan["byte_accounting"]["estimate_is_score_claim"] is False
    assert "candidate_runtime_adapter_missing" in plan["readiness_blockers"]
    assert "exact_cuda_auth_eval_missing" in plan["dispatch_blockers"]


def test_pr103_transform_plan_selects_target_by_rank_or_label() -> None:
    manifest = _manifest()

    by_rank = build_pr103_arithmetic_transform_plan(
        schema_manifest=manifest,
        target_rank=2,
    )
    by_label = build_pr103_arithmetic_transform_plan(
        schema_manifest=manifest,
        target_label="blocks.1.weight",
    )

    assert by_rank["target_stream"]["label"] == "blocks.1.weight"
    assert by_label["target_stream"]["label"] == "blocks.1.weight"
    assert by_rank["target_stream"]["model_gap_bytes_estimate"] == 45
    assert by_label["target_selection"]["label"] == "blocks.1.weight"


def test_pr103_transform_plan_carries_schema_manifest_blockers() -> None:
    manifest = _manifest()
    manifest["ready_for_schema_review"] = False
    manifest["merged_arithmetic_stream"]["reencoded_byte_identical"] = False

    plan = build_pr103_arithmetic_transform_plan(schema_manifest=manifest)

    assert "source_schema_manifest_not_ready_for_schema_review" in plan["readiness_blockers"]
    assert "source_merged_stream_reencode_not_byte_identical" in plan["readiness_blockers"]
    assert plan["ready_for_archive_preflight"] is False


def test_pr103_transform_plan_rejects_unknown_target() -> None:
    with pytest.raises(Pr103ArithmeticTransformPlanError, match="target label not found"):
        build_pr103_arithmetic_transform_plan(
            schema_manifest=_manifest(),
            target_label="missing.weight",
        )


def test_pr103_transform_plan_reads_manifest_from_path(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_json(path, _manifest())

    plan = build_pr103_arithmetic_transform_plan(schema_manifest=path, repo_root=tmp_path)

    assert plan["source_schema_manifest"]["path"] == "manifest.json"
    assert plan["source_schema_manifest"]["sha256"]


def test_pr103_arithmetic_retarget_probe_reports_real_byte_delta_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = _probe_fixture(tmp_path)

    report = build_pr103_arithmetic_retarget_probe(
        schema_manifest=fixture["manifest"],
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
    )

    assert report["schema"] == RETARGET_SCHEMA
    assert report["score_claim"] is False
    assert report["dispatch_attempted"] is False
    assert report["ready_for_archive_preflight"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["target_stream"]["label"] == "fixture.weight0"
    assert report["source_roundtrip"]["byte_identical"] is True
    assert report["retargeted_merged_stream"]["retargeted_sha256"]
    assert report["retargeted_histogram"]["retargeted_raw_sha256"]
    assert report["byte_accounting"]["estimate_is_score_claim"] is False
    assert "candidate_runtime_adapter_missing" in report["readiness_blockers"]
    assert "exact_cuda_auth_eval_missing" in report["dispatch_blockers"]


def test_pr103_arithmetic_retarget_probe_detects_target_symbol_sha_mismatch(
    tmp_path: Path,
) -> None:
    fixture = _probe_fixture(tmp_path)
    fixture["manifest"]["next_arithmetic_schema_targets"][0][
        "decoded_symbols_sha256"
    ] = "f" * 64

    report = build_pr103_arithmetic_retarget_probe(
        schema_manifest=fixture["manifest"],
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
    )

    assert "target_decoded_symbols_sha_mismatch" in report["readiness_blockers"]
    assert report["ready_for_archive_preflight"] is False


def test_pr103_arithmetic_histogram_coordinate_probe_searches_changed_weights(
    tmp_path: Path,
) -> None:
    fixture = _probe_fixture(tmp_path)

    report = build_pr103_arithmetic_histogram_coordinate_probe(
        schema_manifest=fixture["manifest"],
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
        top_symbols=2,
        deltas=(-1, 1),
    )

    assert report["schema"] == COORDINATE_SCHEMA
    assert report["score_claim"] is False
    assert report["ready_for_archive_preflight"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["search_config"]["candidate_count"] > 0
    best = report["best_candidate"]
    assert best["new_weight"] != best["old_weight"]
    assert "merged_ac_delta" in best
    assert "histogram_brotli_delta" in best
    assert "candidate_runtime_adapter_missing" in report["readiness_blockers"]


def test_pr103_arithmetic_histogram_beam_probe_composes_coordinate_changes(
    tmp_path: Path,
) -> None:
    fixture = _probe_fixture(tmp_path)

    report = build_pr103_arithmetic_histogram_beam_probe(
        schema_manifest=fixture["manifest"],
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
        top_symbols=2,
        deltas=(-1, 1),
        rounds=2,
        beam_width=2,
    )

    assert report["schema"] == BEAM_SCHEMA
    assert report["score_claim"] is False
    assert report["ready_for_archive_preflight"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["search_config"]["evaluated_candidate_count"] > 0
    assert report["best_candidate"]["change_count"] >= 1
    assert report["best_candidate"]["moves"]
    assert "candidate_runtime_adapter_missing" in report["readiness_blockers"]


def test_pr103_arithmetic_histogram_candidate_materializes_byte_different_archive(
    tmp_path: Path,
) -> None:
    fixture = _probe_fixture(tmp_path)
    beam0 = build_pr103_arithmetic_histogram_beam_probe(
        schema_manifest=fixture["manifest"],
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
        target_label="fixture.weight0",
        top_symbols=2,
        deltas=(-1, 1),
        rounds=2,
        beam_width=2,
    )
    beam1 = build_pr103_arithmetic_histogram_beam_probe(
        schema_manifest=fixture["manifest"],
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
        target_label="fixture.weight1",
        top_symbols=2,
        deltas=(-1, 1),
        rounds=2,
        beam_width=2,
    )
    output_archive = tmp_path / "candidate.zip"

    report = materialize_pr103_arithmetic_histogram_candidate(
        schema_manifest=fixture["manifest"],
        beam_probe_reports=(beam0, beam1),
        output_archive=output_archive,
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
    )

    assert report["schema"] == CANDIDATE_SCHEMA
    assert output_archive.is_file()
    assert report["score_claim"] is False
    assert report["ready_for_archive_preflight"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["candidate_archive"]["sha256"]
    assert report["candidate_archive"]["member_sha256"] != report["source_archive"]["member_sha256"]
    assert report["byte_accounting"]["selection_mode"] == "greedy_per_stream_best"
    assert "source_probe_delta_sum" in report["byte_accounting"]
    assert "non_additivity_delta" in report["byte_accounting"]
    assert report["candidate_roundtrip"]["reencoded_byte_identical"] is True
    assert report["semantic_stream_parity"]["all_stream_symbol_sha_match"] is True
    assert "candidate_runtime_adapter_missing" in report["readiness_blockers"]
    changed_sections = {
        row["name"] for row in report["section_diffs"] if row["changed"] is True
    }
    assert "ac_histograms_brotli" in changed_sections


def test_pr103_arithmetic_histogram_global_combo_probe_rescores_full_sideband(
    tmp_path: Path,
) -> None:
    fixture = _probe_fixture(tmp_path)
    beam0 = build_pr103_arithmetic_histogram_beam_probe(
        schema_manifest=fixture["manifest"],
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
        target_label="fixture.weight0",
        top_symbols=2,
        deltas=(-1, 1),
        rounds=2,
        beam_width=3,
    )
    beam1 = build_pr103_arithmetic_histogram_beam_probe(
        schema_manifest=fixture["manifest"],
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
        target_label="fixture.weight1",
        top_symbols=2,
        deltas=(-1, 1),
        rounds=2,
        beam_width=3,
    )

    report = build_pr103_arithmetic_histogram_global_combo_probe(
        schema_manifest=fixture["manifest"],
        beam_probe_reports=(beam0, beam1),
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
        top_per_stream=3,
        beam_width=8,
    )

    assert report["schema"] == GLOBAL_COMBO_SCHEMA
    assert report["score_claim"] is False
    assert report["ready_for_archive_preflight"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["search_config"]["stream_count"] == 2
    assert "exact_merged_ac_delta_plus_exact_ac_histograms_brotli_delta" in report[
        "search_config"
    ]["objective"]
    assert "exact_latent_hi_histogram_brotli_delta" in report["search_config"]["objective"]
    best = report["best_candidate"]
    assert "selected_option_source_deltas" in best
    assert "source_probe_delta_sum" in best
    assert "non_additivity_delta" in best
    assert best["estimated_member_delta_if_runtime_adapter_supported"] == (
        best["source_probe_delta_sum"] + best["non_additivity_delta"]
    )
    assert "candidate_runtime_adapter_missing" in report["readiness_blockers"]


def test_pr103_arithmetic_histogram_global_combo_parallel_matches_serial(
    tmp_path: Path,
) -> None:
    fixture = _probe_fixture(tmp_path)
    beam0 = build_pr103_arithmetic_histogram_beam_probe(
        schema_manifest=fixture["manifest"],
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
        target_label="fixture.weight0",
        top_symbols=2,
        deltas=(-1, 1),
        rounds=2,
        beam_width=3,
    )
    beam1 = build_pr103_arithmetic_histogram_beam_probe(
        schema_manifest=fixture["manifest"],
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
        target_label="fixture.weight1",
        top_symbols=2,
        deltas=(-1, 1),
        rounds=2,
        beam_width=3,
    )

    serial = build_pr103_arithmetic_histogram_global_combo_probe(
        schema_manifest=fixture["manifest"],
        beam_probe_reports=(beam0, beam1),
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
        top_per_stream=3,
        beam_width=8,
        combo_workers=1,
    )
    parallel = build_pr103_arithmetic_histogram_global_combo_probe(
        schema_manifest=fixture["manifest"],
        beam_probe_reports=(beam0, beam1),
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
        top_per_stream=3,
        beam_width=8,
        combo_workers=2,
    )

    assert parallel["search_config"]["combo_workers"] == 2
    assert parallel["best_candidate"] == serial["best_candidate"]
    assert parallel["top_candidates"] == serial["top_candidates"]
    assert parallel["frontier_sizes"] == serial["frontier_sizes"]


def test_pr103_arithmetic_histogram_global_combo_process_pool_matches_serial() -> None:
    histograms = np.ones((1, 256), dtype=np.uint8)
    histograms[0, :3] = np.asarray([5, 7, 9], dtype=np.uint8)
    hi_histogram = np.asarray([3, 4, 5], dtype="<u2")
    source_symbol_streams = [
        np.asarray([0, 1, 2, 1, 0, 2, 2, 1], dtype=np.int32),
        np.asarray([0, 1, 2, 1, 0], dtype=np.int32),
    ]
    source_model_weights = [histograms[0].copy(), hi_histogram.copy()]
    source_merged = encode_pr103_merged_ac_stream(
        source_symbol_streams,
        source_model_weights,
    )
    source_histogram_blob = brotli.compress(histograms.tobytes(), quality=11)
    source_hi_histogram_blob = brotli.compress(hi_histogram.tobytes(), quality=11)
    options = [
        {
            "option_id": "fixture.weight0:source",
            "moves": [],
            "source_probe_delta": 0,
        }
    ]
    for index in range(512):
        options.append(
            {
                "option_id": f"fixture.weight0:candidate{index}",
                "moves": [
                    {
                        "symbol": 0,
                        "symbol_count": 8,
                        "delta": 1,
                        "old_weight": 5,
                        "new_weight": 6,
                    }
                ],
                "source_probe_delta": index,
            }
        )
    option_groups = [
        {
            "label": "fixture.weight0",
            "histogram_kind": "ac_histograms_brotli",
            "row_index": 0,
            "options": options,
        }
    ]

    serial = optimize_pr103_histogram_frontier_combinations(
        histograms=histograms,
        hi_histogram=hi_histogram,
        source_symbol_streams=source_symbol_streams,
        source_model_weights=source_model_weights,
        source_merged=source_merged,
        source_histogram_blob=source_histogram_blob,
        source_hi_histogram_blob=source_hi_histogram_blob,
        option_groups=option_groups,
        beam_width=16,
        combo_workers=1,
    )
    process_pool = optimize_pr103_histogram_frontier_combinations(
        histograms=histograms,
        hi_histogram=hi_histogram,
        source_symbol_streams=source_symbol_streams,
        source_model_weights=source_model_weights,
        source_merged=source_merged,
        source_histogram_blob=source_histogram_blob,
        source_hi_histogram_blob=source_hi_histogram_blob,
        option_groups=option_groups,
        beam_width=16,
        combo_workers=2,
    )

    assert process_pool["combo_workers"] == 2
    assert process_pool["evaluated_state_count"] == serial["evaluated_state_count"]
    assert process_pool["frontier_sizes"] == serial["frontier_sizes"]
    public_keys = (
        "estimated_member_delta_if_runtime_adapter_supported",
        "estimated_rate_score_delta_if_components_unchanged",
        "merged_ac_delta",
        "histogram_brotli_delta",
        "latent_hi_histogram_brotli_delta",
        "histogram_state_sha256",
        "selected_options",
        "source_probe_delta_sum",
        "non_additivity_delta",
    )
    assert [
        {key: state[key] for key in public_keys} for state in process_pool["states"]
    ] == [{key: state[key] for key in public_keys} for state in serial["states"]]


def test_pr103_arithmetic_histogram_global_combo_beats_greedy_per_stream_best(
    tmp_path: Path,
) -> None:
    fixture = _global_combo_synergy_fixture(tmp_path)
    source = _source_record_for_report(fixture)
    beam0 = _synthetic_beam_report(
        source=source,
        label="fixture.weight0",
        moves=[
            {
                "round": 1,
                "symbol": 0,
                "symbol_count": 2,
                "delta": -5,
                "old_weight": 8,
                "new_weight": 3,
            }
        ],
        source_probe_delta=0,
    )
    beam1 = _synthetic_beam_report(
        source=source,
        label="fixture.weight1",
        moves=[
            {
                "round": 1,
                "symbol": 13,
                "symbol_count": 1,
                "delta": -5,
                "old_weight": 18,
                "new_weight": 13,
            }
        ],
        source_probe_delta=-3,
    )

    greedy = materialize_pr103_arithmetic_histogram_candidate(
        schema_manifest=fixture["manifest"],
        beam_probe_reports=(beam0, beam1),
        output_archive=tmp_path / "greedy.zip",
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
    )
    combo = build_pr103_arithmetic_histogram_global_combo_probe(
        schema_manifest=fixture["manifest"],
        beam_probe_reports=(beam0, beam1),
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
        top_per_stream=5,
        beam_width=20,
    )
    global_candidate = materialize_pr103_arithmetic_histogram_candidate(
        schema_manifest=fixture["manifest"],
        beam_probe_reports=(beam0, beam1),
        global_combo_report=combo,
        output_archive=tmp_path / "global.zip",
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
    )

    assert greedy["byte_accounting"]["selection_mode"] == "greedy_per_stream_best"
    assert global_candidate["byte_accounting"]["selection_mode"] == "global_combo_best"
    assert combo["best_candidate"]["estimated_member_delta_if_runtime_adapter_supported"] < greedy[
        "byte_accounting"
    ]["payload_byte_delta"]
    assert global_candidate["byte_accounting"]["payload_byte_delta"] < greedy[
        "byte_accounting"
    ]["payload_byte_delta"]
    assert combo["best_candidate"]["selected_options"][0].endswith(":source")
    assert combo["best_candidate"]["score_claim"] is False
    assert global_candidate["score_claim"] is False
    assert "candidate_runtime_adapter_missing" in global_candidate["readiness_blockers"]


def test_pr103_arithmetic_histogram_global_combo_supports_latent_hi_frontier(
    tmp_path: Path,
) -> None:
    fixture = _latent_hi_fixture(tmp_path)
    fixture["manifest"]["next_arithmetic_schema_targets"].append(
        {
            "label": "latent_hi_bytes",
            "role": "latent_hi_stream",
            "schema_index": None,
            "symbol_count": int(fixture["hi_symbols"].size),
            "alphabet_size": int(fixture["hi_histogram"].size),
            "decoded_symbols_sha256": sha256_bytes(
                fixture["hi_symbols"].astype(np.uint16).tobytes()
            ),
            "model_gap_bytes_estimate": 0,
        }
    )
    source = _source_record_for_report(fixture)
    latent_report = _synthetic_beam_report(
        source=source,
        label="latent_hi_bytes",
        moves=[
            {
                "round": 1,
                "symbol": 0,
                "symbol_count": 1,
                "delta": -10,
                "old_weight": 41,
                "new_weight": 31,
            }
        ],
        source_probe_delta=-1,
    )

    combo = build_pr103_arithmetic_histogram_global_combo_probe(
        schema_manifest=fixture["manifest"],
        beam_probe_reports=(latent_report,),
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
        top_per_stream=1,
        beam_width=4,
    )
    candidate = materialize_pr103_arithmetic_histogram_candidate(
        schema_manifest=fixture["manifest"],
        beam_probe_reports=(latent_report,),
        global_combo_report=combo,
        output_archive=tmp_path / "latent-hi.zip",
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
    )

    assert combo["frontier_sizes"][0]["histogram_kind"] == "latent_hi_histogram_brotli"
    assert "latent_hi_histogram_brotli_delta" in combo["best_candidate"]
    assert combo["best_candidate"]["objective_components"][
        "latent_hi_histogram_brotli_delta"
    ] == combo["best_candidate"]["latent_hi_histogram_brotli_delta"]
    assert candidate["candidate_roundtrip"]["reencoded_byte_identical"] is True
    assert candidate["semantic_stream_parity"]["all_stream_symbol_sha_match"] is True
    changed_sections = {
        row["name"] for row in candidate["section_diffs"] if row["changed"] is True
    }
    assert "latent_hi_histogram_brotli" in changed_sections
    assert candidate["score_claim"] is False
    assert candidate["ready_for_exact_eval_dispatch"] is False


def test_pr103_arithmetic_histogram_candidate_can_materialize_global_combo(
    tmp_path: Path,
) -> None:
    fixture = _probe_fixture(tmp_path)
    beam0 = build_pr103_arithmetic_histogram_beam_probe(
        schema_manifest=fixture["manifest"],
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
        target_label="fixture.weight0",
        top_symbols=2,
        deltas=(-1, 1),
        rounds=2,
        beam_width=3,
    )
    beam1 = build_pr103_arithmetic_histogram_beam_probe(
        schema_manifest=fixture["manifest"],
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
        target_label="fixture.weight1",
        top_symbols=2,
        deltas=(-1, 1),
        rounds=2,
        beam_width=3,
    )
    combo = build_pr103_arithmetic_histogram_global_combo_probe(
        schema_manifest=fixture["manifest"],
        beam_probe_reports=(beam0, beam1),
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
        top_per_stream=3,
        beam_width=8,
    )
    if not combo["best_candidate"].get("moves_by_label"):
        pytest.skip("fixture global-combo best is source/noop")

    report = materialize_pr103_arithmetic_histogram_candidate(
        schema_manifest=fixture["manifest"],
        beam_probe_reports=(beam0, beam1),
        global_combo_report=combo,
        output_archive=tmp_path / "global_candidate.zip",
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
    )

    assert report["byte_accounting"]["selection_mode"] == "global_combo_best"
    assert report["byte_accounting"]["source_probe_delta_sum"] == combo["best_candidate"][
        "source_probe_delta_sum"
    ]
    assert report["candidate_roundtrip"]["reencoded_byte_identical"] is True
    assert report["semantic_stream_parity"]["all_stream_symbol_sha_match"] is True


def test_pr103_arithmetic_histogram_candidate_rejects_stale_beam_report(
    tmp_path: Path,
) -> None:
    fixture = _probe_fixture(tmp_path)
    beam = build_pr103_arithmetic_histogram_beam_probe(
        schema_manifest=fixture["manifest"],
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
        target_label="fixture.weight0",
        top_symbols=2,
        deltas=(-1, 1),
        rounds=1,
        beam_width=1,
    )
    beam["source_archive"]["member_sha256"] = "0" * 64

    with pytest.raises(Pr103ArithmeticTransformPlanError, match="member_sha256 mismatch"):
        materialize_pr103_arithmetic_histogram_candidate(
            schema_manifest=fixture["manifest"],
            beam_probe_reports=(beam,),
            output_archive=tmp_path / "candidate.zip",
            repo_root=tmp_path,
            layout=fixture["layout"],
            stream_specs=fixture["stream_specs"],
            hi_symbol_count=fixture["hi_symbol_count"],
        )


def _manifest() -> dict:
    return {
        "planning_only": True,
        "score_claim": False,
        "ready_for_schema_review": True,
        "source_archive": {
            "path": "experiments/results/pr103/archive.zip",
            "bytes": 178223,
            "sha256": "a" * 64,
            "member_name": "x",
            "member_bytes": 178111,
            "member_sha256": "b" * 64,
        },
        "merged_arithmetic_stream": {
            "source_bytes": 153856,
            "source_sha256": "c" * 64,
            "decoded_symbol_count": 237561,
            "decoder_maybe_exhausted": True,
            "reencoded_byte_identical": True,
        },
        "next_arithmetic_schema_targets": [
            {
                "label": "stem.weight",
                "role": "ac_weight_tensor",
                "schema_index": 0,
                "symbol_count": 48384,
                "alphabet_size": 256,
                "decoded_symbols_sha256": "d" * 64,
                "observed_entropy_bits_per_symbol": 5.22,
                "model_cross_entropy_bits_per_symbol": 5.23,
                "observed_entropy_bytes_floor": 31582,
                "model_cross_entropy_bytes_floor": 31627,
                "model_gap_bytes_estimate": 46,
                "required_next_artifact": "byte_different_archive_manifest",
            },
            {
                "label": "blocks.1.weight",
                "role": "ac_weight_tensor",
                "schema_index": 4,
                "symbol_count": 46656,
                "alphabet_size": 256,
                "decoded_symbols_sha256": "e" * 64,
                "observed_entropy_bits_per_symbol": 5.56,
                "model_cross_entropy_bits_per_symbol": 5.57,
                "observed_entropy_bytes_floor": 32434,
                "model_cross_entropy_bytes_floor": 32478,
                "model_gap_bytes_estimate": 45,
            },
        ],
    }


def _probe_fixture(tmp_path: Path) -> dict:
    histograms = np.ones((2, 256), dtype=np.uint8)
    histograms[0, :4] = np.asarray([2, 3, 5, 7], dtype=np.uint8)
    histograms[1, :4] = np.asarray([7, 5, 3, 2], dtype=np.uint8)
    hi_histogram = np.asarray([3, 1, 4], dtype="<u2")
    stream_specs = (
        ("fixture.weight0", 4, 0),
        ("fixture.weight1", 3, 1),
    )
    hi_symbol_count = 5
    symbol_streams = [
        np.asarray([0, 1, 2, 2], dtype=np.int32),
        np.asarray([0, 0, 1], dtype=np.int32),
        np.asarray([2, 0, 2, 1, 0], dtype=np.int32),
    ]
    merged_ac = encode_pr103_merged_ac_stream(
        symbol_streams,
        [histograms[0], histograms[1], hi_histogram],
    )
    scales = b"sc"
    non_ac = brotli.compress(b"non-ac-weights")
    hists = brotli.compress(histograms.tobytes())
    latent_meta = b"meta"
    low = brotli.compress(bytes([1, 2, 3, 4, 5]))
    hi_hist = brotli.compress(hi_histogram.tobytes())
    layout = Pr103LcAcLayout(
        scales_fp16=len(scales),
        non_ac_weights_brotli=len(non_ac),
        ac_histograms_brotli=len(hists),
        merged_range_coded_weights_and_hi_latents=len(merged_ac),
        latent_min_scale_fp16=len(latent_meta),
        latent_low_bytes_brotli=len(low),
        latent_hi_histogram_brotli=len(hi_hist),
    )
    payload = scales + non_ac + hists + merged_ac + latent_meta + low + hi_hist
    archive = tmp_path / "source.zip"
    write_stored_single_member_zip(archive, member_name="x", payload=payload)
    manifest = {
        "planning_only": True,
        "score_claim": False,
        "ready_for_schema_review": True,
        "source_archive": {
            "path": "source.zip",
            "bytes": archive.stat().st_size,
            "sha256": "a" * 64,
            "member_name": "x",
            "member_bytes": len(payload),
            "member_sha256": sha256_bytes(payload),
        },
        "merged_arithmetic_stream": {
            "source_bytes": len(merged_ac),
            "source_sha256": sha256_bytes(merged_ac),
            "decoded_symbol_count": 12,
            "decoder_maybe_exhausted": True,
            "reencoded_byte_identical": True,
        },
        "next_arithmetic_schema_targets": [
            {
                "label": "fixture.weight0",
                "role": "ac_weight_tensor",
                "schema_index": 0,
                "symbol_count": 4,
                "alphabet_size": 256,
                "decoded_symbols_sha256": sha256_bytes(
                    symbol_streams[0].astype(np.uint16).tobytes()
                ),
                "observed_entropy_bytes_floor": 1,
                "model_cross_entropy_bytes_floor": 2,
                "model_gap_bytes_estimate": 1,
            },
            {
                "label": "fixture.weight1",
                "role": "ac_weight_tensor",
                "schema_index": 1,
                "symbol_count": 3,
                "alphabet_size": 256,
                "decoded_symbols_sha256": sha256_bytes(
                    symbol_streams[1].astype(np.uint16).tobytes()
                ),
                "observed_entropy_bytes_floor": 1,
                "model_cross_entropy_bytes_floor": 2,
                "model_gap_bytes_estimate": 1,
            },
        ],
    }
    return {
        "archive": archive,
        "manifest": manifest,
        "layout": layout,
        "stream_specs": stream_specs,
        "hi_symbol_count": hi_symbol_count,
    }


def _global_combo_synergy_fixture(tmp_path: Path) -> dict:
    histograms = np.ones((2, 256), dtype=np.uint8)
    histograms[:, :32] = np.asarray(
        [
            [
                8,
                10,
                15,
                17,
                16,
                18,
                2,
                13,
                3,
                16,
                17,
                10,
                19,
                2,
                6,
                14,
                18,
                16,
                6,
                13,
                5,
                10,
                1,
                3,
                18,
                5,
                2,
                14,
                3,
                5,
                1,
                5,
            ],
            [
                17,
                4,
                9,
                10,
                4,
                16,
                6,
                3,
                5,
                13,
                9,
                7,
                13,
                18,
                10,
                14,
                18,
                10,
                9,
                10,
                6,
                12,
                6,
                12,
                10,
                19,
                8,
                6,
                15,
                14,
                15,
                10,
            ],
        ],
        dtype=np.uint8,
    )
    hi_histogram = np.asarray([1, 2, 3], dtype="<u2")
    stream_specs = (
        ("fixture.weight0", 20, 0),
        ("fixture.weight1", 20, 1),
    )
    hi_symbol_count = 5
    symbol_streams = [
        np.asarray(
            [17, 17, 29, 8, 26, 21, 0, 12, 27, 17, 1, 24, 23, 27, 5, 2, 27, 0, 17, 2],
            dtype=np.int32,
        ),
        np.asarray(
            [9, 15, 13, 12, 0, 0, 3, 0, 21, 16, 20, 8, 19, 24, 12, 14, 31, 25, 31, 12],
            dtype=np.int32,
        ),
        np.asarray([2, 2, 1, 2, 2], dtype=np.int32),
    ]
    merged_ac = encode_pr103_merged_ac_stream(
        symbol_streams,
        [histograms[0], histograms[1], hi_histogram],
    )
    scales = b"sc"
    non_ac = brotli.compress(b"non-ac-weights")
    hists = brotli.compress(histograms.tobytes(), quality=11)
    latent_meta = b"meta"
    low = brotli.compress(bytes([1, 2, 3, 4, 5]))
    hi_hist = brotli.compress(hi_histogram.tobytes(), quality=11)
    layout = Pr103LcAcLayout(
        scales_fp16=len(scales),
        non_ac_weights_brotli=len(non_ac),
        ac_histograms_brotli=len(hists),
        merged_range_coded_weights_and_hi_latents=len(merged_ac),
        latent_min_scale_fp16=len(latent_meta),
        latent_low_bytes_brotli=len(low),
        latent_hi_histogram_brotli=len(hi_hist),
    )
    payload = scales + non_ac + hists + merged_ac + latent_meta + low + hi_hist
    archive = tmp_path / "source.zip"
    write_stored_single_member_zip(archive, member_name="x", payload=payload)
    manifest = {
        "planning_only": True,
        "score_claim": False,
        "ready_for_schema_review": True,
        "source_archive": {
            "path": "source.zip",
            "bytes": archive.stat().st_size,
            "sha256": "a" * 64,
            "member_name": "x",
            "member_bytes": len(payload),
            "member_sha256": sha256_bytes(payload),
        },
        "merged_arithmetic_stream": {
            "source_bytes": len(merged_ac),
            "source_sha256": sha256_bytes(merged_ac),
            "decoded_symbol_count": sum(int(stream.size) for stream in symbol_streams),
            "decoder_maybe_exhausted": True,
            "reencoded_byte_identical": True,
        },
        "next_arithmetic_schema_targets": [
            {
                "label": "fixture.weight0",
                "role": "ac_weight_tensor",
                "schema_index": 0,
                "symbol_count": int(symbol_streams[0].size),
                "alphabet_size": 256,
                "decoded_symbols_sha256": sha256_bytes(
                    symbol_streams[0].astype(np.uint16).tobytes()
                ),
                "model_gap_bytes_estimate": 1,
            },
            {
                "label": "fixture.weight1",
                "role": "ac_weight_tensor",
                "schema_index": 1,
                "symbol_count": int(symbol_streams[1].size),
                "alphabet_size": 256,
                "decoded_symbols_sha256": sha256_bytes(
                    symbol_streams[1].astype(np.uint16).tobytes()
                ),
                "model_gap_bytes_estimate": 1,
            },
        ],
    }
    return {
        "archive": archive,
        "manifest": manifest,
        "layout": layout,
        "stream_specs": stream_specs,
        "hi_symbol_count": hi_symbol_count,
    }


def _latent_hi_fixture(tmp_path: Path) -> dict:
    histograms = np.ones((1, 256), dtype=np.uint8)
    histograms[0, :4] = np.asarray([2, 3, 5, 7], dtype=np.uint8)
    hi_histogram = np.asarray(
        [
            41,
            77,
            15,
            85,
            2,
            22,
            98,
            59,
            98,
            80,
            74,
            26,
            13,
            35,
            48,
            84,
            3,
            58,
            32,
            51,
            42,
            67,
            61,
            51,
            97,
            98,
            59,
            75,
            50,
            6,
            22,
            15,
            89,
            54,
            6,
            82,
            2,
            7,
            89,
            68,
            21,
            76,
            19,
            78,
            60,
            87,
            76,
            19,
            87,
            55,
            74,
            80,
            72,
            36,
            60,
            19,
            78,
            48,
            63,
            9,
            88,
            22,
            16,
            85,
        ],
        dtype="<u2",
    )
    hi_symbols = np.asarray(
        [
            42,
            55,
            53,
            56,
            19,
            30,
            39,
            17,
            58,
            0,
            53,
            41,
            16,
            46,
            26,
            53,
            63,
            18,
            30,
            13,
            44,
            40,
            54,
            51,
            62,
            61,
            57,
            9,
            2,
            30,
            22,
            57,
            51,
            27,
            36,
            37,
            57,
            1,
            31,
            43,
            28,
            58,
            60,
            52,
            29,
            56,
            4,
            42,
            17,
            15,
        ],
        dtype=np.int32,
    )
    stream_specs = (("fixture.weight0", 4, 0),)
    symbol_streams = [
        np.asarray([0, 1, 2, 2], dtype=np.int32),
        hi_symbols,
    ]
    merged_ac = encode_pr103_merged_ac_stream(
        symbol_streams,
        [histograms[0], hi_histogram],
    )
    scales = b"sc"
    non_ac = brotli.compress(b"non-ac-weights")
    hists = brotli.compress(histograms.tobytes(), quality=11)
    latent_meta = b"meta"
    low = brotli.compress(bytes([1, 2, 3, 4, 5]))
    hi_hist = brotli.compress(hi_histogram.tobytes(), quality=11)
    layout = Pr103LcAcLayout(
        scales_fp16=len(scales),
        non_ac_weights_brotli=len(non_ac),
        ac_histograms_brotli=len(hists),
        merged_range_coded_weights_and_hi_latents=len(merged_ac),
        latent_min_scale_fp16=len(latent_meta),
        latent_low_bytes_brotli=len(low),
        latent_hi_histogram_brotli=len(hi_hist),
    )
    payload = scales + non_ac + hists + merged_ac + latent_meta + low + hi_hist
    archive = tmp_path / "source.zip"
    write_stored_single_member_zip(archive, member_name="x", payload=payload)
    manifest = {
        "planning_only": True,
        "score_claim": False,
        "ready_for_schema_review": True,
        "source_archive": {
            "path": "source.zip",
            "bytes": archive.stat().st_size,
            "sha256": "a" * 64,
            "member_name": "x",
            "member_bytes": len(payload),
            "member_sha256": sha256_bytes(payload),
        },
        "merged_arithmetic_stream": {
            "source_bytes": len(merged_ac),
            "source_sha256": sha256_bytes(merged_ac),
            "decoded_symbol_count": sum(int(stream.size) for stream in symbol_streams),
            "decoder_maybe_exhausted": True,
            "reencoded_byte_identical": True,
        },
        "next_arithmetic_schema_targets": [
            {
                "label": "fixture.weight0",
                "role": "ac_weight_tensor",
                "schema_index": 0,
                "symbol_count": int(symbol_streams[0].size),
                "alphabet_size": 256,
                "decoded_symbols_sha256": sha256_bytes(
                    symbol_streams[0].astype(np.uint16).tobytes()
                ),
                "model_gap_bytes_estimate": 0,
            }
        ],
    }
    return {
        "archive": archive,
        "manifest": manifest,
        "layout": layout,
        "stream_specs": stream_specs,
        "hi_symbol_count": int(hi_symbols.size),
        "hi_histogram": hi_histogram,
        "hi_symbols": hi_symbols,
    }


def _source_record_for_report(fixture: dict) -> dict:
    archive = Path(fixture["archive"])
    payload = bytes(archive.read_bytes())
    import zipfile

    with zipfile.ZipFile(archive) as zf:
        member = zf.infolist()[0]
        member_payload = zf.read(member.filename)
    return {
        "path": archive.name,
        "bytes": archive.stat().st_size,
        "sha256": sha256_bytes(payload),
        "member_name": member.filename,
        "member_bytes": len(member_payload),
        "member_sha256": sha256_bytes(member_payload),
    }


def _synthetic_beam_report(
    *,
    source: dict,
    label: str,
    moves: list[dict],
    source_probe_delta: int,
) -> dict:
    return {
        "schema": BEAM_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "source_archive": source,
        "target_stream": {"label": label},
        "best_candidate": {
            "moves": moves,
            "estimated_member_delta_if_runtime_adapter_supported": source_probe_delta,
        },
        "top_candidates": [],
    }
