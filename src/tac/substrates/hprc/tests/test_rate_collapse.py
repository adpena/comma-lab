# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

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
    parse_rate_collapse_sections,
    parse_residual_token_collapse_specs,
    rate_collapse_variant_groups,
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
