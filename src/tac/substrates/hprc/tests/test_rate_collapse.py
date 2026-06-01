# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tac.repo_io import sha256_file
from tac.substrates.hprc.archive import HprcSectionKind, parse_hprc_packet
from tac.substrates.hprc.archive_candidate import export_hprc_archive_bytes
from tac.substrates.hprc.learned_receiver import (
    build_compact_receiver_packet_from_lowres_frames,
    decode_compact_receiver_packet,
    render_compact_receiver_frame_batch,
)
from tac.substrates.hprc.rate_collapse import (
    DEFAULT_RATE_COLLAPSE_SECTIONS,
    HPRC_RATE_COLLAPSE_EXACT_EXECUTION_SCHEMA,
    HPRC_RATE_COLLAPSE_REPORT_SCHEMA,
    ResidualTokenCollapseSpec,
    build_rate_collapse_exact_execution_report,
    collapse_residual_tokens_with_importance,
    parse_rate_collapse_sections,
    parse_residual_token_collapse_specs,
    rate_collapse_variant_groups,
    transcode_compact_receiver_importance_weighted_residual_tokens,
    transcode_compact_receiver_residual_tokens,
    transcode_compact_receiver_sections,
)

REPO = Path(__file__).resolve().parents[5]


def _load_rate_tool():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location(
        "transcode_hprc_compact_receiver_rate_collapse_test",
        REPO / "tools/transcode_hprc_compact_receiver_rate_collapse.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rate_tool = _load_rate_tool()


def _compressible_frames() -> np.ndarray:
    y = np.arange(12, dtype=np.float32)[:, None, None]
    x = np.arange(16, dtype=np.float32)[None, :, None]
    c = np.arange(3, dtype=np.float32)[None, None, :]
    frames = []
    for idx in range(6):
        frames.append((42.0 + x * 2.0 + y * 3.0 + c * 5.0 + idx).astype(np.float32))
    return np.stack(frames, axis=0)


def test_rate_collapse_defaults_cover_semantic_payload_mass() -> None:
    assert DEFAULT_RATE_COLLAPSE_SECTIONS == (
        HprcSectionKind.DECODER_QW,
        HprcSectionKind.LATENTS_RC,
        HprcSectionKind.SELECTORS_RC,
        HprcSectionKind.RESIDUAL_RC,
        HprcSectionKind.RECEIVER_STATE,
    )
    assert parse_rate_collapse_sections("residual,decoder,residual") == (
        HprcSectionKind.RESIDUAL_RC,
        HprcSectionKind.DECODER_QW,
    )
    groups = rate_collapse_variant_groups(DEFAULT_RATE_COLLAPSE_SECTIONS)
    assert groups[-1] == ("selected_sections", DEFAULT_RATE_COLLAPSE_SECTIONS)


def test_rate_collapse_transcode_is_lossless_for_pixels() -> None:
    packet = build_compact_receiver_packet_from_lowres_frames(
        _compressible_frames(),
        basis_count=3,
        residual_grid_h=3,
        residual_grid_w=4,
        source_manifest={"source": "unit_rate_collapse"},
    )
    base = decode_compact_receiver_packet(parse_hprc_packet(packet))
    base_render = render_compact_receiver_frame_batch(base, 0, 6, height=24, width=32)

    collapsed, rows = transcode_compact_receiver_sections(
        packet,
        sections=DEFAULT_RATE_COLLAPSE_SECTIONS,
        brotli_quality=11,
    )
    compact = decode_compact_receiver_packet(parse_hprc_packet(collapsed))
    rendered = render_compact_receiver_frame_batch(compact, 0, 6, height=24, width=32)

    assert any(row["accepted"] for row in rows)
    assert len(collapsed) < len(packet)
    np.testing.assert_array_equal(rendered, base_render)


def test_residual_token_collapse_is_receiver_decodable_and_records_damage() -> None:
    rng = np.random.default_rng(7)
    frames = rng.integers(0, 256, size=(6, 12, 16, 3), dtype=np.uint8).astype(np.float32)
    packet = build_compact_receiver_packet_from_lowres_frames(
        frames,
        basis_count=3,
        residual_grid_h=3,
        residual_grid_w=4,
        source_manifest={"source": "unit_residual_token_collapse"},
    )
    specs = parse_residual_token_collapse_specs("4:2 dz8_qd1")
    assert specs == (
        ResidualTokenCollapseSpec(deadzone=4, quant_divisor=2),
        ResidualTokenCollapseSpec(deadzone=8, quant_divisor=1),
    )

    collapsed, rows, metrics = transcode_compact_receiver_residual_tokens(
        packet,
        spec=specs[0],
        sections=DEFAULT_RATE_COLLAPSE_SECTIONS,
        brotli_quality=11,
    )

    compact = decode_compact_receiver_packet(parse_hprc_packet(collapsed))
    rendered = render_compact_receiver_frame_batch(compact, 0, 6, height=24, width=32)
    assert rendered.shape == (6, 24, 32, 3)
    assert any(row["accepted"] for row in rows)
    assert metrics["variant_id"] == "residual_tokens_dz4_qd2"
    assert metrics["tokens_changed"] > 0
    assert metrics["residual_q_mse"] > 0.0


def test_importance_weighted_residual_collapse_protects_high_importance_tokens() -> None:
    q = np.array([[[[13, 17, 25], [11, 17, 23]]]], dtype=np.int16)
    importance = np.array([[[0.0, 10.0]]], dtype=np.float32)

    collapsed, metrics = collapse_residual_tokens_with_importance(
        q,
        low_importance_spec=ResidualTokenCollapseSpec(deadzone=0, quant_divisor=6),
        high_importance_spec=ResidualTokenCollapseSpec(deadzone=0, quant_divisor=1),
        importance=importance,
        coarsen_quantile=0.50,
    )

    np.testing.assert_array_equal(collapsed[0, 0, 0], np.array([12, 18, 24], dtype=np.int16))
    np.testing.assert_array_equal(collapsed[0, 0, 1], q[0, 0, 1])
    assert metrics["importance_weighted"] is True
    assert metrics["coarsened_token_count"] == 3
    assert metrics["protected_token_count"] == 3


def test_importance_weighted_residual_collapse_can_confine_to_eligible_mask() -> None:
    q = np.array([[[[13, 17, 25], [11, 17, 23]]]], dtype=np.int16)
    importance = np.array([[[0.0, 1.0]]], dtype=np.float32)
    eligible_mask = np.array([[[True, False]]])

    collapsed, metrics = collapse_residual_tokens_with_importance(
        q,
        low_importance_spec=ResidualTokenCollapseSpec(deadzone=0, quant_divisor=6),
        high_importance_spec=ResidualTokenCollapseSpec(deadzone=0, quant_divisor=1),
        importance=importance,
        eligible_mask=eligible_mask,
        coarsen_quantile=1.0,
        selection_domain="eligible_low",
    )

    np.testing.assert_array_equal(collapsed[0, 0, 0], np.array([12, 18, 24], dtype=np.int16))
    np.testing.assert_array_equal(collapsed[0, 0, 1], q[0, 0, 1])
    assert metrics["selection_domain"] == "eligible_low"
    assert metrics["eligible_token_count"] == 3
    assert metrics["coarsened_token_count"] == 3


def test_importance_weighted_residual_transcode_is_receiver_decodable() -> None:
    rng = np.random.default_rng(13)
    frames = rng.integers(0, 256, size=(6, 12, 16, 3), dtype=np.uint8).astype(np.float32)
    packet = build_compact_receiver_packet_from_lowres_frames(
        frames,
        basis_count=3,
        residual_grid_h=3,
        residual_grid_w=4,
        source_manifest={"source": "unit_importance_weighted_residual_collapse"},
    )
    importance = np.ones((6, 3, 4), dtype=np.float32)
    importance[:2] = 0.0

    collapsed, rows, metrics = transcode_compact_receiver_importance_weighted_residual_tokens(
        packet,
        low_importance_spec=ResidualTokenCollapseSpec(deadzone=0, quant_divisor=6),
        high_importance_spec=ResidualTokenCollapseSpec(deadzone=0, quant_divisor=1),
        importance=importance,
        coarsen_quantile=0.34,
        sections=DEFAULT_RATE_COLLAPSE_SECTIONS,
        brotli_quality=11,
    )

    compact = decode_compact_receiver_packet(parse_hprc_packet(collapsed))
    rendered = render_compact_receiver_frame_batch(compact, 0, 6, height=24, width=32)
    assert rendered.shape == (6, 24, 32, 3)
    assert any(row["accepted"] for row in rows)
    assert metrics["variant_id"].startswith("residual_tokens_iw_")
    assert metrics["tokens_changed"] > 0
    assert 0.0 < metrics["coarsened_token_fraction"] < 1.0


def test_rate_collapse_cli_consumes_exact_bridge_without_reinterpreting_custody(
    tmp_path: Path,
) -> None:
    packet = build_compact_receiver_packet_from_lowres_frames(
        _compressible_frames(),
        basis_count=3,
        residual_grid_h=3,
        residual_grid_w=4,
        source_manifest={"source": "unit_rate_collapse_cli"},
    )
    source_dir = tmp_path / "source"
    archive_path, archive_sha, archive_bytes = export_hprc_archive_bytes(
        packet,
        source_dir,
        repo_root=REPO,
        emit_archive_bound_candidate_package=False,
    )
    source_bin = source_dir / "0.bin"
    bridge = {
        "schema": "hprc_incremental_exact_gate_bridge.v1",
        "ready_for_exact_eval_dispatch": True,
        "archive": {
            "path": archive_path.as_posix(),
            "sha256": archive_sha,
            "bytes": archive_bytes,
            "hprc_0bin_sha256": sha256_file(source_bin),
        },
        "archive_custody": {"verified": True},
        "hprc_0bin_custody": {"verified": True},
    }
    bridge_path = tmp_path / "bridge.json"
    bridge_path.write_text(json.dumps(bridge), encoding="utf-8")

    out_dir = tmp_path / "rate_collapse"
    rc = rate_tool.main(
        [
            "--exact-bridge",
            bridge_path.as_posix(),
            "--output-dir",
            out_dir.as_posix(),
            "--repo-root",
            REPO.as_posix(),
            "--skip-receiver-proof",
        ]
    )

    assert rc == 0
    report = json.loads((out_dir / "hprc_rate_collapse_report.json").read_text())
    assert report["schema"] == HPRC_RATE_COLLAPSE_REPORT_SCHEMA
    assert report["source_input"]["kind"] == "exact_bridge"
    assert report["source_input"]["custody_verified"] is True
    assert report["lossy_residual_collapse_enabled"] is False
    assert all(not row["lossy_residual_token_collapse"] for row in report["variants"])
    assert report["artifact"]["archive_bytes"] <= archive_bytes
    assert report["artifact"]["score_claim"] is False
    assert report["artifact"]["archive_bound_package_present"] is False
    assert report["artifact"]["lossy_residual_token_collapse"] is False
    assert report["artifact"]["residual_transform"] == "lossless_hprc_section_entropy_rate_collapse"
    assert report["artifact"]["losslessness_kind"] == "lossless_section_entropy_transcode"


def test_rate_collapse_cli_requires_target_or_flag_for_lossy_residual_candidates(
    tmp_path: Path,
) -> None:
    packet = build_compact_receiver_packet_from_lowres_frames(
        _compressible_frames(),
        basis_count=3,
        residual_grid_h=3,
        residual_grid_w=4,
        source_manifest={"source": "unit_rate_collapse_cli_lossy"},
    )
    source_dir = tmp_path / "source"
    archive_path, archive_sha, archive_bytes = export_hprc_archive_bytes(
        packet,
        source_dir,
        repo_root=REPO,
        emit_archive_bound_candidate_package=False,
    )
    source_bin = source_dir / "0.bin"
    bridge = {
        "schema": "hprc_incremental_exact_gate_bridge.v1",
        "ready_for_exact_eval_dispatch": True,
        "archive": {
            "path": archive_path.as_posix(),
            "sha256": archive_sha,
            "bytes": archive_bytes,
            "hprc_0bin_sha256": sha256_file(source_bin),
        },
        "archive_custody": {"verified": True},
        "hprc_0bin_custody": {"verified": True},
    }
    bridge_path = tmp_path / "bridge.json"
    bridge_path.write_text(json.dumps(bridge), encoding="utf-8")

    out_dir = tmp_path / "rate_collapse"
    rc = rate_tool.main(
        [
            "--exact-bridge",
            bridge_path.as_posix(),
            "--output-dir",
            out_dir.as_posix(),
            "--repo-root",
            REPO.as_posix(),
            "--skip-receiver-proof",
            "--enable-lossy-residual-collapse",
        ]
    )

    assert rc == 0
    report = json.loads((out_dir / "hprc_rate_collapse_report.json").read_text())
    assert report["lossy_residual_collapse_enabled"] is True
    assert any(row["lossy_residual_token_collapse"] for row in report["variants"])
    selected = next(row for row in report["variants"] if row["variant_id"] == report["best_variant_id"])
    assert report["artifact"]["lossy_residual_token_collapse"] == selected[
        "lossy_residual_token_collapse"
    ]
    assert report["artifact"]["residual_token_collapse"] == selected["residual_token_collapse"]
    assert report["artifact"]["losslessness_kind"] in {
        "lossless_section_entropy_transcode",
        "lossy_residual_token_collapse",
    }
    if selected["lossy_residual_token_collapse"]:
        assert report["artifact"]["residual_transform"].startswith(
            "hprc_lossy_residual_token_rate_collapse_"
        )


def test_rate_collapse_cli_builds_importance_weighted_variants_from_p19_p18_artifacts(
    tmp_path: Path,
) -> None:
    packet = build_compact_receiver_packet_from_lowres_frames(
        _compressible_frames(),
        basis_count=3,
        residual_grid_h=3,
        residual_grid_w=4,
        source_manifest={"source": "unit_rate_collapse_cli_importance"},
    )
    source_dir = tmp_path / "source"
    archive_path, archive_sha, archive_bytes = export_hprc_archive_bytes(
        packet,
        source_dir,
        repo_root=REPO,
        emit_archive_bound_candidate_package=False,
    )
    source_bin = source_dir / "0.bin"
    bridge = {
        "schema": "hprc_incremental_exact_gate_bridge.v1",
        "ready_for_exact_eval_dispatch": True,
        "archive": {
            "path": archive_path.as_posix(),
            "sha256": archive_sha,
            "bytes": archive_bytes,
            "hprc_0bin_sha256": sha256_file(source_bin),
        },
        "archive_custody": {"verified": True},
        "hprc_0bin_custody": {"verified": True},
    }
    bridge_path = tmp_path / "bridge.json"
    bridge_path.write_text(json.dumps(bridge), encoding="utf-8")
    p19_path = tmp_path / "p19.json"
    p19_path.write_text(
        json.dumps(
            {
                "schema": "p19_posenet_null_pair_detection.v1",
                "selected_pair_ids": [0],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    p18_path = tmp_path / "p18.json"
    p18_path.write_text(
        json.dumps(
            {
                "schema": "p18_segnet_region_waterfill.v1",
                "rows": [
                    {
                        "pair_id": 0,
                        "regions256": [
                            {"box": {"x0": 0.0, "y0": 0.0, "x1": 0.5, "y1": 0.5}}
                        ],
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "rate_collapse"

    rc = rate_tool.main(
        [
            "--exact-bridge",
            bridge_path.as_posix(),
            "--output-dir",
            out_dir.as_posix(),
            "--repo-root",
            REPO.as_posix(),
            "--skip-receiver-proof",
            "--target-rate-term",
            "0.30",
            "--residual-collapse-schedule",
            "dz0_qd6",
            "--importance-protected-spec",
            "dz0_qd1",
            "--importance-coarsen-quantile",
            "0.25",
            "--p19-posenet-null-pairs",
            p19_path.as_posix(),
            "--p18-segnet-region-waterfill",
            p18_path.as_posix(),
        ]
    )

    assert rc == 0
    report = json.loads((out_dir / "hprc_rate_collapse_report.json").read_text())
    weighted = [
        row
        for row in report["variants"]
        if row.get("importance_weighted_residual_token_collapse") is True
    ]
    assert weighted
    assert report["residual_importance_enabled"] is True
    assert report["residual_importance_source"]["kind"] == "p18_p19_scorer_region_artifacts"
    assert report["residual_importance_source"]["source_binding_status"] == "video_pair_count_compatible"
    assert weighted[0]["residual_token_collapse"]["coarsened_token_count"] > 0
    assert weighted[0]["residual_token_collapse"]["selection_domain"] == "global_weighted"


def test_rate_collapse_rejects_stale_p19_out_of_range_pair_artifact(tmp_path: Path) -> None:
    packet = build_compact_receiver_packet_from_lowres_frames(
        _compressible_frames(),
        basis_count=3,
        residual_grid_h=3,
        residual_grid_w=4,
        source_manifest={"source": "unit_rate_collapse_cli_stale_p19"},
    )
    p19_path = tmp_path / "p19_stale.json"
    p19_path.write_text(
        json.dumps(
            {
                "schema": "p19_posenet_null_pair_detection.v1",
                "n_pairs": 3,
                "selected_pair_ids": [99],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="out-of-range pair ids"):
        rate_tool._importance_from_p18_p19_artifacts(
            packet,
            p19_path=p19_path,
            p18_path=None,
            repo_root=REPO,
        )


def test_rate_collapse_exact_execution_report_feeds_existing_exact_gate(
    tmp_path: Path,
) -> None:
    packet = build_compact_receiver_packet_from_lowres_frames(
        _compressible_frames(),
        basis_count=3,
        residual_grid_h=3,
        residual_grid_w=4,
        source_manifest={"source": "unit_rate_collapse_exact_execution"},
    )
    source_dir = tmp_path / "source"
    archive_path, archive_sha, archive_bytes = export_hprc_archive_bytes(
        packet,
        source_dir,
        repo_root=REPO,
        emit_archive_bound_candidate_package=False,
    )
    source_bin = source_dir / "0.bin"
    bridge = {
        "schema": "hprc_incremental_exact_gate_bridge.v1",
        "candidate_id": "candidate-a",
        "candidate_variant_id": "variant-a",
        "ready_for_exact_eval_dispatch": True,
        "archive": {
            "path": archive_path.as_posix(),
            "sha256": archive_sha,
            "bytes": archive_bytes,
            "hprc_0bin_sha256": sha256_file(source_bin),
        },
        "archive_custody": {"verified": True},
        "hprc_0bin_custody": {"verified": True},
        "mlx_advisory_summary": {"delta_total_mlx_score_advisory": -0.5},
    }
    bridge_path = tmp_path / "bridge.json"
    bridge_path.write_text(json.dumps(bridge), encoding="utf-8")
    out_dir = tmp_path / "rate_collapse"
    rc = rate_tool.main(
        [
            "--exact-bridge",
            bridge_path.as_posix(),
            "--output-dir",
            out_dir.as_posix(),
            "--repo-root",
            REPO.as_posix(),
            "--disable-lossy-residual-collapse",
        ]
    )
    assert rc == 0

    exact_input = build_rate_collapse_exact_execution_report(
        rate_collapse_report_path=out_dir / "hprc_rate_collapse_report.json",
        source_exact_bridge_path=bridge_path,
        repo_root=REPO,
    )

    assert exact_input["schema"] == HPRC_RATE_COLLAPSE_EXACT_EXECUTION_SCHEMA
    assert exact_input["archive"]["bytes"] <= archive_bytes
    assert exact_input["receiver_proof_binding"]["receiver_contract_satisfied"] is True
    assert exact_input["incremental_summary"]["rate_collapse_archive_bytes_saved"] >= 0
    assert exact_input["residual_transform"] == "lossless_hprc_section_entropy_rate_collapse"
    assert exact_input["incremental_summary"]["rate_collapse_lossy_residual_token_collapse"] is False
    assert exact_input["score_claim"] is False
