from __future__ import annotations

from pathlib import Path

import brotli
import numpy as np
import pytest

from tac.hnerv_lowlevel_packer import write_stored_single_member_zip
from tac.hnerv_pr103_lc_ac_schema import (
    Pr103LcAcLayout,
    build_pr103_lc_ac_schema_manifest,
    decode_pr103_merged_ac_stream,
    encode_pr103_merged_ac_stream,
    parse_pr103_lc_ac_payload,
)

pytest.importorskip("constriction")

REPO = Path(__file__).resolve().parents[3]


def test_pr103_merged_ac_stream_decodes_and_reencodes_byte_identically() -> None:
    fixture = _synthetic_lc_ac_fixture()

    summary = decode_pr103_merged_ac_stream(
        fixture["merged_ac"],
        fixture["histograms"],
        fixture["hi_histogram"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
    )

    assert summary["decoder_maybe_exhausted"] is True
    assert summary["reencoded_byte_identical"] is True
    assert summary["source_sha256"] == summary["reencoded_sha256"]
    assert summary["decoded_symbol_count"] == 12
    assert summary["blockers"] == []
    assert [row["label"] for row in summary["stream_rows"]] == [
        "fixture.weight0",
        "fixture.weight1",
        "latent_hi_bytes",
    ]
    assert all(row["score_claim"] is False for row in summary["stream_rows"])


def test_pr103_schema_manifest_is_fail_closed_without_candidate_or_exact_eval(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_lc_ac_fixture()
    archive = tmp_path / "source.zip"
    write_stored_single_member_zip(archive, member_name="x", payload=fixture["payload"])

    manifest = build_pr103_lc_ac_schema_manifest(
        source_archive=archive,
        source_label="fixture-pr103",
        repo_root=tmp_path,
        layout=fixture["layout"],
        stream_specs=fixture["stream_specs"],
        hi_symbol_count=fixture["hi_symbol_count"],
    )

    assert manifest["score_claim"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["ready_for_schema_review"] is True
    assert manifest["ready_for_archive_preflight"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["source_archive"]["sha256"]
    assert manifest["merged_arithmetic_stream"]["reencoded_byte_identical"] is True
    assert manifest["old_new_archive_sha256_pair"]["closed"] is False
    assert "candidate_archive_missing" in manifest["readiness_blockers"]
    assert "old_new_archive_sha256_pair_missing" in manifest["readiness_blockers"]
    assert "exact_cuda_auth_eval_missing" in manifest["dispatch_blockers"]
    assert manifest["next_arithmetic_schema_targets"]


def test_public_pr103_archive_schema_profile_if_available() -> None:
    archive = (
        REPO
        / "experiments/results/public_pr_intake_full/public_pr103_intake_20260505_auto/archive.zip"
    )
    adjudication = (
        REPO
        / "experiments/results/lightning_batch/"
        / "exact_eval_public_pr103_hnerv_lc_ac_t4_20260504T1245Z/adjudication.log"
    )
    replay_fidelity = (
        REPO
        / "experiments/results/public_hnerv_frontier_payload_profiles_20260504_codex/"
        / "pr103_replay_fidelity.json"
    )
    if not archive.exists() or not adjudication.exists() or not replay_fidelity.exists():
        pytest.skip("local PR103 public intake/eval artifacts are not present")

    manifest = build_pr103_lc_ac_schema_manifest(
        source_archive=archive,
        source_label="PR103 hnerv_lc_ac",
        exact_adjudication_log=adjudication,
        replay_fidelity_json=replay_fidelity,
        repo_root=REPO,
    )

    assert manifest["source_archive"]["bytes"] == 178223
    assert manifest["source_archive"]["sha256"] == (
        "31881b2d23d027e6619f2d8df2fe35d4d207d08882ec673d6c1b7ff119f18c30"
    )
    assert manifest["merged_arithmetic_stream"]["source_bytes"] == 153856
    assert manifest["merged_arithmetic_stream"]["decoded_symbol_count"] == 237561
    assert manifest["merged_arithmetic_stream"]["reencoded_byte_identical"] is True
    assert manifest["source_exact_adjudication"]["score_recomputed"] == pytest.approx(
        0.2277649714224471
    )
    assert "replay_fidelity:public_leaderboard_score_mismatch" in manifest[
        "readiness_blockers"
    ]
    assert manifest["ready_for_exact_eval_dispatch"] is False


def _synthetic_lc_ac_fixture() -> dict:
    histograms = np.asarray([[2, 3, 5], [5, 2, 1]], dtype=np.uint8)
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
    parsed = parse_pr103_lc_ac_payload(payload, layout=layout)
    assert parsed.section_bytes("merged_range_coded_weights_and_hi_latents") == merged_ac
    return {
        "payload": payload,
        "layout": layout,
        "merged_ac": merged_ac,
        "histograms": histograms,
        "hi_histogram": hi_histogram,
        "stream_specs": stream_specs,
        "hi_symbol_count": hi_symbol_count,
    }
