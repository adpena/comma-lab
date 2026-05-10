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
    PLAN_SCHEMA,
    RETARGET_SCHEMA,
    Pr103ArithmeticTransformPlanError,
    build_pr103_arithmetic_histogram_beam_probe,
    build_pr103_arithmetic_histogram_coordinate_probe,
    build_pr103_arithmetic_retarget_probe,
    build_pr103_arithmetic_transform_plan,
    materialize_pr103_arithmetic_histogram_candidate,
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
    assert report["candidate_roundtrip"]["reencoded_byte_identical"] is True
    assert "candidate_runtime_adapter_missing" in report["readiness_blockers"]
    changed_sections = {
        row["name"] for row in report["section_diffs"] if row["changed"] is True
    }
    assert "ac_histograms_brotli" in changed_sections


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
