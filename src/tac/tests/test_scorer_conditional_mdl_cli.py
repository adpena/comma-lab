# SPDX-License-Identifier: MIT
"""CLI tests for Z1 scorer-conditional MDL operator surfaces."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load_tool(module_name: str, path: Path):
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _compute_tool():
    return _load_tool(
        "compute_scorer_conditional_mdl_ablation",
        REPO / "tools" / "compute_scorer_conditional_mdl_ablation.py",
    )


def _probe_tool():
    return _load_tool(
        "probe_zen_floor_disambiguator",
        REPO / "tools" / "probe_zen_floor_disambiguator.py",
    )


if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tac.packet_compiler.pr106_sidecar_packet import (  # noqa: E402
    PR106SidecarPacket,
    PR106_SIDECAR_FORMAT_BROTLI,
    emit_pr106_sidecar_packet,
)
from tac.analysis.hnerv_packet_sections import PARSER_IBPS1  # noqa: E402


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_pr106_fixture(path: Path) -> bytes:
    decoder = b"D" * 12
    tail = b"L" * 24
    payload = b"\xff" + len(decoder).to_bytes(3, "little") + decoder + tail
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", payload, compress_type=zipfile.ZIP_STORED)
    return payload


def _write_pr106_sidecar_fixture(path: Path) -> tuple[bytes, bytes]:
    decoder = b"D" * 12
    tail = b"L" * 24
    inner_payload = b"\xff" + len(decoder).to_bytes(3, "little") + decoder + tail
    wrapper_payload = emit_pr106_sidecar_packet(
        PR106SidecarPacket(
            format_id=PR106_SIDECAR_FORMAT_BROTLI,
            pr106_bytes=inner_payload,
            sidecar_payload=b"sidecar-bytes-that-must-not-be-section-sliced",
            framing_meta=None,
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", wrapper_payload, compress_type=zipfile.ZIP_STORED)
    return inner_payload, wrapper_payload


def _write_ibps1_fixture(path: Path) -> bytes:
    encoder = b"E" * 11
    decoder = b"D" * 17
    latent_dim = 4
    num_pairs = 3
    latent = b"Z" * (latent_dim * num_pairs)
    meta = b'{"beta_ib":0.1}'
    payload = (
        b"IBPS"
        + bytes([1])
        + latent_dim.to_bytes(2, "little")
        + num_pairs.to_bytes(2, "little")
        + len(encoder).to_bytes(4, "little")
        + len(decoder).to_bytes(4, "little")
        + len(latent).to_bytes(4, "little")
        + len(meta).to_bytes(4, "little")
        + encoder
        + decoder
        + latent
        + meta
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", payload, compress_type=zipfile.ZIP_STORED)
    return payload


def _write_eval_json(path: Path, archive: Path) -> None:
    archive_bytes = archive.read_bytes()
    path.write_text(
        json.dumps(
            {
                "score_axis": "contest_cuda",
                "avg_posenet_dist": 0.000034,
                "avg_segnet_dist": 0.00062,
                "archive_size_bytes": archive.stat().st_size,
                "archive": {
                    "archive_sha256": _sha256(archive_bytes),
                    "archive_size_bytes": archive.stat().st_size,
                },
                "score_recomputed_from_components": 0.206,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _run_compute(tmp_path: Path) -> tuple[Path, Path, dict]:
    archive = tmp_path / "pr106_fixture.zip"
    eval_json = tmp_path / "eval.json"
    out_json = tmp_path / "z1.json"
    out_md = tmp_path / "z1.md"
    _write_pr106_fixture(archive)
    _write_eval_json(eval_json, archive)
    rc = _compute_tool().main(
        [
            "--archive",
            f"pr106_fixture={archive}",
            "--eval-json",
            f"pr106_fixture={eval_json}",
            "--chunk-size",
            "8",
            "--repo-root",
            str(REPO),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ]
    )
    assert rc == 0
    return out_json, out_md, json.loads(out_json.read_text(encoding="utf-8"))


def test_compute_cli_writes_json_markdown_and_proxy_safe_invariants(tmp_path):
    out_json, out_md, payload = _run_compute(tmp_path)

    assert out_json.is_file()
    assert out_md.is_file()
    assert payload["schema"] == "tac_scorer_conditional_mdl_ablation_v1"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["dispatch_attempted"] is False
    assert payload["gpu_required"] is False
    assert payload["true_scorer_conditional_entropy_claim"] is False
    assert payload["archive_count"] == 1
    assert payload["archives"][0]["label"] == "pr106_fixture"
    assert payload["archives"][0]["scorer_feature_summary"]["score_axis"] == "contest_cuda"
    assert payload["archives"][0]["scorer_feature_summary"]["custody_match"] is True
    assert (
        payload["archives"][0]["scorer_feature_summary"]["custody_strength"]
        == "archive_sha256_and_bytes"
    )
    assert "scorer_feature_proxy_conditioned" in payload["measurement_layers"]
    assert payload["sensitivity_map"]["score_claim"] is False
    assert payload["allocator_hook"]["ready_for_exact_eval_dispatch"] is False
    assert payload["probe_disambiguator"]["tool_expected"] == "tools/probe_zen_floor_disambiguator.py"
    assert payload["autopilot_rows"][0]["score_claim"] is False
    assert payload["autopilot_rows"][0]["ready_for_exact_eval_dispatch"] is False
    assert "Decision-Z1" in json.dumps(payload["source_documents"])
    assert "score_claim: `false`" in out_md.read_text(encoding="utf-8")


def test_compute_cli_slices_pr106_sidecar_sections_from_inner_parser_payload(tmp_path):
    archive = tmp_path / "pr106_sidecar_fixture.zip"
    out_json = tmp_path / "z1_sidecar.json"
    out_md = tmp_path / "z1_sidecar.md"
    inner_payload, wrapper_payload = _write_pr106_sidecar_fixture(archive)

    rc = _compute_tool().main(
        [
            "--archive",
            f"pr106_sidecar_fixture={archive}",
            "--chunk-size",
            "8",
            "--repo-root",
            str(REPO),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ]
    )

    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    record = payload["archives"][0]
    assert record["member_sha256"] == _sha256(wrapper_payload)
    assert record["parser_input"]["kind"] == "pr106_sidecar_inner_payload"
    assert record["parser_input"]["sha256"] == _sha256(inner_payload)
    assert record["pr106_sidecar_wrapper"]["outer_member_sha256"] == _sha256(wrapper_payload)
    assert record["sections"][0]["name"] == "packed_header_ff_len24"
    assert record["sections"][0]["sha256"] == _sha256(inner_payload[:4])
    assert record["sections"][0]["sha256"] != _sha256(wrapper_payload[:4])


def test_compute_cli_parses_ibps1_sections_without_whole_blob(tmp_path):
    archive = tmp_path / "c6_ibps1_fixture.zip"
    out_json = tmp_path / "z1_ibps1.json"
    out_md = tmp_path / "z1_ibps1.md"
    inner_payload = _write_ibps1_fixture(archive)

    rc = _compute_tool().main(
        [
            "--archive",
            f"c6_fixture={archive},parser=ibps1",
            "--chunk-size",
            "8",
            "--repo-root",
            str(REPO),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ]
    )

    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    record = payload["archives"][0]
    assert record["parser"]["name"] == PARSER_IBPS1
    assert record["member_name"] == "0.bin"
    assert record["parser_input"]["kind"] == "member_payload"
    assert record["parser_input"]["sha256"] == _sha256(inner_payload)
    section_names = [section["name"] for section in record["sections"]]
    assert section_names == [
        "ibps1_header",
        "encoder_blob",
        "decoder_blob",
        "latent_blob",
        "meta_blob",
    ]
    assert "whole_blob" not in section_names
    assert payload["measurement_layers"]["parser_section_conditioned"]["group_count"] == 5


def test_compute_cli_infers_legacy_lightning_exact_eval_as_contest_cuda(tmp_path):
    archive = tmp_path / "pr106_fixture.zip"
    eval_json = tmp_path / "lightning_batch" / "exact_eval_fixture_t4" / "contest_auth_eval.json"
    out_json = tmp_path / "z1_lightning.json"
    out_md = tmp_path / "z1_lightning.md"
    _write_pr106_fixture(archive)
    archive_bytes = archive.read_bytes()
    eval_json.parent.mkdir(parents=True)
    eval_json.write_text(
        json.dumps(
            {
                "avg_posenet_dist": 0.000034,
                "avg_segnet_dist": 0.00062,
                "archive_size_bytes": archive.stat().st_size,
                "provenance": {
                    "archive_sha256": _sha256(archive_bytes),
                    "archive_size_bytes": archive.stat().st_size,
                },
                "score_recomputed_from_components": 0.206,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    rc = _compute_tool().main(
        [
            "--archive",
            f"pr106_fixture={archive}",
            "--eval-json",
            f"pr106_fixture={eval_json}",
            "--repo-root",
            str(REPO),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ]
    )

    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["archives"][0]["scorer_feature_summary"]["score_axis"] == "contest_cuda"


def test_compute_cli_eval_json_without_archive_sha_is_weak_custody(tmp_path):
    archive = tmp_path / "pr106_fixture.zip"
    eval_json = tmp_path / "eval_missing_sha.json"
    out_json = tmp_path / "z1_missing_eval_sha.json"
    out_md = tmp_path / "z1_missing_eval_sha.md"
    _write_pr106_fixture(archive)
    eval_json.write_text(
        json.dumps(
            {
                "score_axis": "contest_cuda",
                "avg_posenet_dist": 0.000034,
                "avg_segnet_dist": 0.00062,
                "archive_size_bytes": archive.stat().st_size,
                "score_recomputed_from_components": 0.206,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    rc = _compute_tool().main(
        [
            "--archive",
            f"pr106_fixture={archive}",
            "--eval-json",
            f"pr106_fixture={eval_json}",
            "--repo-root",
            str(REPO),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ]
    )

    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    summary = payload["archives"][0]["scorer_feature_summary"]
    assert summary["custody_match"] is False
    assert summary["custody_strength"] == "missing_archive_identity"
    assert "missing_eval_json_archive_sha256" in summary["custody_blockers"]
    assert payload["score_claim"] is False


def test_compute_cli_missing_archive_fails_closed_and_writes_error_artifacts(tmp_path):
    out_json = tmp_path / "z1_error.json"
    out_md = tmp_path / "z1_error.md"
    missing = tmp_path / "missing.zip"

    rc = _compute_tool().main(
        [
            "--archive",
            f"missing={missing}",
            "--repo-root",
            str(REPO),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ]
    )

    assert rc == 2
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["dispatch_attempted"] is False
    assert "archive missing" in payload["error"]["message"]
    assert "failed_closed" in out_md.read_text(encoding="utf-8")


def test_compute_cli_mismatched_eval_json_fails_closed(tmp_path):
    archive = tmp_path / "pr106_fixture.zip"
    eval_json = tmp_path / "wrong_eval.json"
    out_json = tmp_path / "z1_wrong_eval.json"
    out_md = tmp_path / "z1_wrong_eval.md"
    _write_pr106_fixture(archive)
    eval_json.write_text(
        json.dumps(
            {
                "score_axis": "contest_cuda",
                "avg_posenet_dist": 0.000034,
                "avg_segnet_dist": 0.00062,
                "archive_size_bytes": archive.stat().st_size + 1,
                "archive": {
                    "archive_sha256": "0" * 64,
                    "archive_size_bytes": archive.stat().st_size + 1,
                },
                "score_recomputed_from_components": 0.206,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    rc = _compute_tool().main(
        [
            "--archive",
            f"pr106_fixture={archive}",
            "--eval-json",
            f"pr106_fixture={eval_json}",
            "--repo-root",
            str(REPO),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ]
    )

    assert rc == 2
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert "archive sha mismatch" in payload["error"]["message"]
    assert "failed_closed" in out_md.read_text(encoding="utf-8")


def test_probe_cli_consumes_z1_output_as_proxy_planning_artifact(tmp_path):
    z1_json, _, _ = _run_compute(tmp_path)
    out_json = tmp_path / "probe.json"
    out_md = tmp_path / "probe.md"

    rc = _probe_tool().main(
        [
            "--z1-json",
            str(z1_json),
            "--repo-root",
            str(REPO),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ]
    )

    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["schema"] == "zen_floor_disambiguator_v1"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["planning_artifact_only"] is True
    assert payload["evidence_grade"] == "proxy_planning_only"
    assert payload["true_scorer_feature_bindings"]["available"] is False
    assert payload["probe_fields"]["static_vs_substrate_scope_arbitrated"] is True
    assert payload["probe_fields"]["proxy_only_unless_true_feature_bindings"] is True
    assert payload["verdict"]["selected_interpretation"].startswith("proxy_")
    assert payload["autopilot_rows"][0]["score_claim"] is False
    assert payload["autopilot_rows"][0]["ready_for_exact_eval_dispatch"] is False
    assert "selected_interpretation" in out_md.read_text(encoding="utf-8")


def test_probe_cli_accepts_true_feature_binding_followup_without_score_claim(tmp_path):
    z1_json, _, _ = _run_compute(tmp_path)
    binding = tmp_path / "binding.json"
    binding.write_text(
        json.dumps(
            {
                "schema": "byte_to_scorer_feature_binding_fixture_v1",
                "byte_to_scorer_feature_binding_ready": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    out_json = tmp_path / "probe_bound.json"
    out_md = tmp_path / "probe_bound.md"

    rc = _probe_tool().main(
        [
            "--z1-json",
            str(z1_json),
            "--feature-binding-json",
            str(binding),
            "--repo-root",
            str(REPO),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ]
    )

    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["planning_artifact_only"] is False
    assert payload["evidence_grade"] == "true_scorer_feature_bound_planning"
    assert payload["true_scorer_feature_bindings"]["available"] is True
    assert payload["verdict"]["true_scorer_feature_binding_available"] is True
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
