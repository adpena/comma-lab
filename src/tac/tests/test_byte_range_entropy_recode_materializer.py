# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import brotli
import numpy as np

from tac.hnerv_lowlevel_packer import write_stored_single_member_zip
from tac.hnerv_pr103_lc_ac_schema import (
    Pr103LcAcLayout,
    encode_pr103_merged_ac_stream,
)
from tac.optimization.byte_range_entropy_recode_materializer import (
    CANDIDATE_SCHEMA,
    MATERIALIZER_ID,
    RECEIVER_CONTRACT_ID,
    RECEIVER_PROOF_SCHEMA,
    TARGET_KIND,
    build_byte_range_entropy_recode_plan,
    materialize_byte_range_entropy_recode_candidate,
    verify_byte_range_entropy_recode_receiver_contract,
)
from tac.pr103_arithmetic_transform_plan import (
    build_pr103_arithmetic_histogram_beam_probe,
)
from tac.repo_io import sha256_bytes


def test_byte_range_entropy_recode_plan_requires_runtime_proof() -> None:
    plan = build_byte_range_entropy_recode_plan(
        archive_member_name="x",
        archive_byte_range={
            "archive_member_name": "x",
            "section_name": "ac_histograms_brotli",
            "candidate_start": 2,
            "candidate_end": 10,
        },
    )

    assert plan["target_kind"] == TARGET_KIND
    assert plan["materializer_id"] == MATERIALIZER_ID
    assert plan["receiver_contract_satisfied"] is False
    assert "runtime_consumption_proof_missing" in plan["readiness_blockers"]
    assert plan["score_claim"] is False
    assert plan["promotion_eligible"] is False


def test_byte_range_entropy_recode_receiver_proof_accepts_strict_contract() -> None:
    proof = {
        "schema": RECEIVER_PROOF_SCHEMA,
        "ready_for_exact_eval_runtime": True,
        "archive_member_name": "x",
        "candidate_archive_sha256": "a" * 64,
        "candidate_member_sha256": "b" * 64,
        "archive_byte_ranges": [
            {
                "archive_member_name": "x",
                "section_name": "ac_histograms_brotli",
                "candidate_start": 2,
                "candidate_end": 10,
            }
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
    }

    verification = verify_byte_range_entropy_recode_receiver_contract(
        runtime_consumption_proof=proof,
        required_archive_member_name="x",
        required_candidate_archive_sha256="a" * 64,
        required_candidate_member_sha256="b" * 64,
    )

    assert verification["receiver_contract_id"] == RECEIVER_CONTRACT_ID
    assert verification["receiver_contract_satisfied"] is True
    assert verification["blockers"] == []
    assert verification["ready_for_exact_eval_dispatch"] is False


def test_pr103_backed_byte_range_entropy_materializer_emits_byte_closed_candidate(
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
        rounds=2,
        beam_width=2,
    )
    output_archive = tmp_path / "candidate.zip"

    report = materialize_byte_range_entropy_recode_candidate(
        schema_manifest=fixture["manifest"],
        beam_probe_reports=(beam,),
        output_archive=output_archive,
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
    )

    assert report["schema"] == CANDIDATE_SCHEMA
    assert report["materializer_id"] == MATERIALIZER_ID
    assert report["target_kind"] == TARGET_KIND
    assert output_archive.is_file()
    assert report["byte_closed_candidate_emitted"] is True
    assert report["candidate_archive"]["sha256"]
    assert report["archive_diff_manifest"]["candidate_non_noop"] is True
    assert report["receiver_contract_satisfied"] is False
    assert "runtime_consumption_proof_missing" in report["readiness_blockers"]
    assert "candidate_runtime_adapter_missing" in report["readiness_blockers"]
    assert "byte_range_entropy_recode_receiver_contract_not_satisfied" in report[
        "readiness_blockers"
    ]
    changed_sections = {row["section_name"] for row in report["archive_byte_ranges"]}
    assert "ac_histograms_brotli" in changed_sections
    assert all(
        row["candidate_end"] > row["candidate_start"]
        for row in report["archive_byte_ranges"]
    )
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


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
    low = brotli.compress(b"\x00" * 512)
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
