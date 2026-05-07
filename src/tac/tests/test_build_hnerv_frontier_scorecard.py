from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from experiments.build_hnerv_frontier_scorecard import (
    candidate_indexes,
    followup_targets,
    hidden_gem_byte_mass_ranking,
    inspect_single_member_archive,
    next_exact_evaluable_target,
    payload_equivalence_groups,
    payload_section_manifests,
    profile_indexes,
    render_markdown,
    row_from_eval,
    section_optimization_role,
    sha256_bytes,
)


def test_payload_sha_profile_fallback_tracks_byte_identical_repack(tmp_path: Path) -> None:
    payload = b"\xff\x08\x00\x00decoder!latent-sidecar"
    original_dir = tmp_path / "original"
    repack_dir = tmp_path / "repack"
    original_dir.mkdir()
    repack_dir.mkdir()
    original_info = _write_archive(original_dir / "archive.zip", "0.bin", payload)
    repack_info = _write_archive(repack_dir / "archive.zip", "x", payload)
    original_eval = _write_eval(original_dir, original_info)
    repack_eval = _write_eval(repack_dir, repack_info)
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(
        json.dumps(
            [
                {
                    **original_info,
                    "kind": "ff_packed_brotli_hnerv",
                    "sections": [
                        _section("decoder_packed_brotli", 4, 12, payload),
                        _section("latents_and_sidecar_brotli", 12, len(payload), payload),
                    ],
                }
            ]
        )
    )

    indexes = profile_indexes([profile_path])
    original_row = row_from_eval("PR106", original_eval, indexes)
    repack_row = row_from_eval("PR106x", repack_eval, indexes)

    assert original_row["profile_match_key"] == "archive_sha256"
    assert repack_row["profile_match_key"] == "member_sha256"
    assert repack_row["zip_member"] == "x"
    assert repack_row["profile_member_name"] == "0.bin"
    assert repack_row["payload_sha256"] == original_row["payload_sha256"]
    assert repack_row["largest_payload_section"]["name"] == "latents_and_sidecar_brotli"

    groups = payload_equivalence_groups([original_row, repack_row])
    assert groups == [
        {
            "payload_sha256": original_row["payload_sha256"],
            "labels": ["PR106", "PR106x"],
            "archive_byte_span": abs(original_row["archive_bytes"] - repack_row["archive_bytes"]),
            "same_seg_contribution": True,
            "same_pose_contribution": True,
            "archives": [
                {
                    "label": "PR106",
                    "archive_sha256": original_row["archive_sha256"],
                    "archive_bytes": original_row["archive_bytes"],
                    "zip_member": "0.bin",
                    "zip_overhead_bytes": original_row["zip_overhead_bytes"],
                    "profile_match_key": "archive_sha256",
                },
                {
                    "label": "PR106x",
                    "archive_sha256": repack_row["archive_sha256"],
                    "archive_bytes": repack_row["archive_bytes"],
                    "zip_member": "x",
                    "zip_overhead_bytes": repack_row["zip_overhead_bytes"],
                    "profile_match_key": "member_sha256",
                },
            ],
            "readiness": "byte-identical payload pair; use as repack custody/control only",
        }
    ]

    targets = followup_targets([repack_row])
    assert targets[0]["suggested_action"] == "latent/sidecar arithmetic-coding parity fixture"
    assert "Payload Follow-Up Targets" in render_markdown([original_row, repack_row])

    manifests = payload_section_manifests([original_row, repack_row])
    assert manifests[0]["dispatch_blockers"] == [
        "payload_section_manifest_is_byte_forensics_only",
        "requires_byte_different_candidate_archive",
        "requires_exact_cuda_auth_eval_on_candidate",
    ]
    assert manifests[0]["score_claim"] is False
    assert manifests[0]["sections"][0]["optimization_role"] == "decoder_weight_stream"
    assert manifests[0]["sections"][1]["optimization_role"] == "latent_stream"
    assert manifests[1]["profile_match_key"] == "member_sha256"
    assert "Payload Section Manifests" in render_markdown([original_row, repack_row])


def test_hidden_gem_ranking_prefers_current_frontier_before_larger_predecessor(
    tmp_path: Path,
) -> None:
    predecessor_payload = b"\xff\x0c\x00\x00bigger-dec!!latent"
    frontier_payload = b"\xff\x08\x00\x00decoder!latent"
    predecessor_dir = tmp_path / "predecessor"
    frontier_dir = tmp_path / "frontier"
    predecessor_dir.mkdir()
    frontier_dir.mkdir()
    predecessor_info = _write_archive(predecessor_dir / "archive.zip", "0.bin", predecessor_payload)
    frontier_info = _write_archive(frontier_dir / "archive.zip", "x", frontier_payload)
    predecessor_eval = _write_eval(predecessor_dir, predecessor_info, score=0.251)
    frontier_eval = _write_eval(frontier_dir, frontier_info, score=0.250)
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(
        json.dumps(
            [
                {
                    **predecessor_info,
                    "kind": "ff_packed_brotli_hnerv",
                    "sections": [
                        _section("decoder_packed_brotli", 4, 16, predecessor_payload),
                        _section("latents_and_sidecar_brotli", 16, len(predecessor_payload), predecessor_payload),
                    ],
                },
                {
                    **frontier_info,
                    "kind": "ff_packed_brotli_hnerv",
                    "sections": [
                        _section("decoder_packed_brotli", 4, 12, frontier_payload),
                        _section("latents_and_sidecar_brotli", 12, len(frontier_payload), frontier_payload),
                    ],
                },
            ]
        )
    )

    indexes = profile_indexes([profile_path])
    predecessor_row = row_from_eval("PR106", predecessor_eval, indexes)
    frontier_row = row_from_eval("PR106x-lowlevel-brotli", frontier_eval, indexes)

    ranking = hidden_gem_byte_mass_ranking([predecessor_row, frontier_row])
    target = next_exact_evaluable_target([predecessor_row, frontier_row])

    assert ranking[0]["label"] == "PR106x-lowlevel-brotli"
    assert ranking[0]["section"] == "decoder_packed_brotli"
    assert ranking[0]["priority"] == "current_frontier_primary"
    assert ranking[0]["score_gap_to_current_frontier"] == 0.0
    assert ranking[0]["score_claim"] is False
    assert ranking[0]["dispatch_attempted"] is False
    assert target == ranking[0]
    md = render_markdown([predecessor_row, frontier_row])
    assert "Next Exact-Evaluable Target" in md
    assert "Hidden-Gem Byte-Mass Ranking" in md
    assert "| PR106x-lowlevel-brotli | `decoder_packed_brotli` |" in md


def test_candidate_manifest_marks_lossless_brotli_repack_as_local_control(tmp_path: Path) -> None:
    payload = b"\xff\x08\x00\x00decoder!latent-sidecar"
    lane_dir = tmp_path / "candidate"
    lane_dir.mkdir()
    archive_info = _write_archive(lane_dir / "archive.zip", "x", payload)
    eval_path = _write_eval(lane_dir, archive_info)
    candidate_path = tmp_path / "candidate_result.json"
    candidate_path.write_text(
        json.dumps(
            {
                "candidate_archive_bytes": archive_info["archive_bytes"],
                "candidate_archive_sha256": archive_info["archive_sha256"],
                "candidate_payload_sha256": archive_info["member_sha256"],
                "source_archive_bytes": 200,
                "source_archive_sha256": "a" * 64,
                "source_label": "PR106x",
                "brotli_raw_equivalence": [
                    {
                        "section_name": "decoder_packed_brotli",
                        "raw_equal": True,
                        "raw_bytes": 8,
                        "source_raw_sha256": "b" * 64,
                        "candidate_raw_sha256": "b" * 64,
                    }
                ],
                "candidate_diff_audit": {
                    "score_claim": False,
                    "total_byte_delta": -151,
                    "sections": [],
                },
            },
            sort_keys=True,
        )
        + "\n"
    )

    row = row_from_eval("PR106x-lowlevel-brotli", eval_path, profile_indexes([]), candidate_indexes([candidate_path]))

    assert row["candidate_manifest_match_key"] == "candidate_archive_sha256"
    assert row["candidate_source_label"] == "PR106x"
    assert row["candidate_source_archive_sha256"] == "a" * 64
    assert row["raw_equivalence_closed"] is True
    assert row["frontier_scope"] == "exact_local_cuda_custody_lossless_repack_control"
    assert row["candidate_diff_audit"]["total_byte_delta"] == -151
    md = render_markdown([row])
    assert "lossless Brotli repack row is a local exact-custody" in md
    assert "categorical/range-coded HNeRV" in md


def test_section_optimization_role_is_stable_for_known_hnerv_sections() -> None:
    assert section_optimization_role("decoder_packed_brotli") == "decoder_weight_stream"
    assert section_optimization_role("latents_and_sidecar_brotli") == "latent_stream"
    assert section_optimization_role("ac_histograms_brotli") == "entropy_model_or_range_stream"
    assert section_optimization_role("latent_min_scale_fp16") == "control_or_metadata"
    assert section_optimization_role("opaque_single_payload") == "opaque_payload_stream"


def test_eval_sibling_archive_must_match_provenance(tmp_path: Path) -> None:
    lane_dir = tmp_path / "lane"
    lane_dir.mkdir()
    archive_info = _write_archive(lane_dir / "archive.zip", "x", b"payload")
    eval_path = _write_eval(lane_dir, {**archive_info, "archive_sha256": "bad"})

    with pytest.raises(ValueError, match="SHA mismatch"):
        row_from_eval("bad", eval_path, profile_indexes([]))


def test_exact_negative_adjudication_is_not_frontier_eligible(tmp_path: Path) -> None:
    lane_dir = tmp_path / "negative"
    lane_dir.mkdir()
    archive_info = _write_archive(lane_dir / "archive.zip", "x", b"payload")
    eval_path = _write_eval(
        lane_dir,
        archive_info,
        extra={
            "adjudication_provenance": {
                "lane_status": "REGRESSION_REVIEW_REQUIRED",
                "paper_claim_grade": "A-negative scoped forensic",
                "promotion_eligible": False,
                "regression_triggered": True,
            },
            "evidence_grade": "A-negative scoped forensic",
        },
    )

    row = row_from_eval("apogee_int4", eval_path, profile_indexes([]))

    assert row["evidence_grade"] == "A++"
    assert row["canonical_frontier_eligible"] is False
    assert "promotion_ineligible" in row["canonicality_blockers"]
    assert "regression_triggered" in row["canonicality_blockers"]
    assert "lane_status_REGRESSION_REVIEW_REQUIRED" in row["canonicality_blockers"]
    assert "canonical |" in render_markdown([row])
    assert "| apogee_int4 | A++ | no |" in render_markdown([row])


def _write_archive(path: Path, member_name: str, payload: bytes) -> dict[str, object]:
    info = ZipInfo(member_name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with ZipFile(path, "w") as zf:
        zf.writestr(info, payload)
    return inspect_single_member_archive(path)


def _write_eval(
    lane_dir: Path,
    archive_info: dict[str, object],
    *,
    score: float = 0.25,
    extra: dict[str, object] | None = None,
) -> Path:
    payload = {
        "archive_size_bytes": archive_info["archive_bytes"],
        "avg_posenet_dist": 0.001,
        "avg_segnet_dist": 0.002,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": archive_info["archive_sha256"],
            "device": "cuda",
            "gpu_model": "fixture T4",
            "gpu_t4_match": True,
            "inflate_runtime_manifest": {"runtime_tree_sha256": "runtime-fixture"},
        },
        "score_pose_contribution": 0.02,
        "score_rate_contribution": 0.03,
        "score_recomputed_from_components": score,
        "score_seg_contribution": 0.2,
    }
    if extra:
        payload.update(extra)
    path = lane_dir / "contest_auth_eval.adjudicated.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _section(name: str, start: int, end: int, payload: bytes) -> dict[str, object]:
    data = payload[start:end]
    return {
        "bytes": len(data),
        "end": end,
        "entropy_bits_per_byte": 7.0,
        "name": name,
        "sha256": sha256_bytes(data),
        "start": start,
    }
