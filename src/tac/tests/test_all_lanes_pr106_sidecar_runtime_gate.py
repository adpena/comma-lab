from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
ALL_LANES = REPO / "tools" / "all_lanes_preflight.py"


def _load_all_lanes_module() -> Any:
    for path in (REPO, REPO / "src", REPO / "tools"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    spec = importlib.util.spec_from_file_location("all_lanes_pr106_sidecar_test", ALL_LANES)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pr106_sidecar_runtime_consumption_gate_passes_current_archives() -> None:
    module = _load_all_lanes_module()

    passed, output = module._run_pr106_sidecar_runtime_consumption_gate()

    assert passed is True
    assert "format_ids=0x01,0x02" in output
    assert "PacketIR identity parse-emit accounts for every payload byte" in output
    assert "runtime decodes/applies sidecar bytes" in output
    assert "runtime-consumption manifests intentionally remain non-promotable" in output
    assert "same-runtime full-frame parity manifest" in output
    assert "score_claim=false" in output
    assert "ready_for_exact_eval_dispatch=false" in output


def test_pr106_sidecar_runtime_consumption_gate_reports_valid_full_frame_parity_manifest(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load_all_lanes_module()
    manifest_path = tmp_path / "full_frame_parity.json"
    manifest_path.write_text(
        module.json.dumps(
            {
                "schema": "pr106_same_runtime_streaming_frame_parity_v1",
                "proof_scope": "same_runtime_streaming_full_frame_hash",
                "streaming_output_sha256_equal": True,
                "streaming_output_total_bytes_equal": True,
                "full_frame_inflate_output_parity_claim": True,
                "prefix_parity_claim": False,
                "device_axis_label": "local-cpu-streaming-runtime",
                "contest_axis_claim": False,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "source": {
                    "n_pairs_hashed": 600,
                    "total_frames": 1200,
                    "total_bytes": 3662409600,
                    "streaming_raw_sha256": "a" * 64,
                },
                "candidate": {
                    "n_pairs_hashed": 600,
                    "total_frames": 1200,
                    "total_bytes": 3662409600,
                    "streaming_raw_sha256": "a" * 64,
                },
            }
        )
    )
    monkeypatch.setattr(module, "PR106_R2_SAME_RUNTIME_FULL_FRAME_PARITY", manifest_path)

    failures, note = module._pr106_same_runtime_full_frame_parity_status()

    assert failures == []
    assert "manifest present" in note
    assert "contest_axis_claim=false" in note
    assert "score_claim=false" in note


def test_pr106_sidecar_runtime_consumption_gate_rejects_bad_full_frame_parity_manifest(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load_all_lanes_module()
    manifest_path = tmp_path / "full_frame_parity.json"
    manifest_path.write_text(
        module.json.dumps(
            {
                "schema": "pr106_same_runtime_streaming_frame_parity_v1",
                "proof_scope": "same_runtime_streaming_full_frame_hash",
                "streaming_output_sha256_equal": False,
                "streaming_output_total_bytes_equal": True,
                "full_frame_inflate_output_parity_claim": True,
                "prefix_parity_claim": False,
                "device_axis_label": "local-cpu-streaming-runtime",
                "contest_axis_claim": False,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "source": {
                    "n_pairs_hashed": 600,
                    "total_frames": 1200,
                    "total_bytes": 3662409600,
                    "streaming_raw_sha256": "a" * 64,
                },
                "candidate": {
                    "n_pairs_hashed": 600,
                    "total_frames": 1200,
                    "total_bytes": 3662409600,
                    "streaming_raw_sha256": "b" * 64,
                },
            }
        )
    )
    monkeypatch.setattr(module, "PR106_R2_SAME_RUNTIME_FULL_FRAME_PARITY", manifest_path)

    passed, output = module._run_pr106_sidecar_runtime_consumption_gate()

    assert passed is False
    assert "streaming_output_sha256_equal_drift" in output
    assert "streaming_raw_sha256_mismatch" in output


def test_pr106_sidecar_runtime_consumption_gate_rejects_promotable_manifest(monkeypatch) -> None:
    module = _load_all_lanes_module()

    def fake_proof(*, archive_path: Path, runtime_dir: Path) -> dict[str, object]:
        return {
            "schema": "pr106_sidecar_runtime_decode_consumption_proof_v1",
            "format_id": "0x01" if "pr101_grammar" not in str(runtime_dir) else "0x02",
            "payload_sha256_changed": True,
            "inner_pr106_payload_sha256_unchanged": True,
            "sidecar_payload_sha256_changed": True,
            "runtime_semantic_digest_changed": True,
            "runtime_corrected_latents_digest_changed": True,
            "runtime_sidecar_decode_consumption_claim": True,
            "runtime_sidecar_apply_consumption_claim": True,
            "full_frame_inflate_output_parity_claim": False,
            "score_claim": True,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "source_runtime_correction_digest": {
                "format_id": "0x01" if "pr101_grammar" not in str(runtime_dir) else "0x02",
                "n_pairs": 600,
                "source_latents_sha256": "a" * 64,
                "corrected_latents_sha256": "b" * 64,
                "combined_sha256": "c" * 64,
                "latents_changed_by_sidecar": True,
            },
            "mutated_runtime_correction_digest": {
                "format_id": "0x01" if "pr101_grammar" not in str(runtime_dir) else "0x02",
                "n_pairs": 600,
                "source_latents_sha256": "a" * 64,
                "corrected_latents_sha256": "d" * 64,
                "combined_sha256": "e" * 64,
                "latents_changed_by_sidecar": True,
            },
        }

    monkeypatch.setattr(
        module.sys.modules["tac.packet_compiler"],
        "prove_pr106_sidecar_runtime_decode_consumption",
        fake_proof,
    )

    passed, output = module._run_pr106_sidecar_runtime_consumption_gate()

    assert passed is False
    assert "score_claim_drift" in output


def test_pr106_sidecar_runtime_consumption_gate_rejects_packet_ir_identity_drift(
    monkeypatch,
) -> None:
    module = _load_all_lanes_module()
    packet_compiler = module.sys.modules["tac.packet_compiler"]

    def fake_identity(*, archive_path: Path) -> dict[str, object]:
        return {
            "schema": "pr106_sidecar_packet_ir_identity_proof_v1",
            "packet_ir_identity_passed": False,
            "packet": {
                "format_id": "0x01" if "pr101_grammar" not in str(archive_path) else "0x02",
                "packet_ir_consumed_byte_proof": {
                    "runtime_consumption_claim": False,
                    "all_payload_bytes_accounted": True,
                    "unconsumed_trailing_bytes": 0,
                    "section_gaps": [],
                    "score_affecting_section_names": ["pr106_payload", "sidecar_payload"],
                    "emitted_payload_bytes": 8,
                    "emitted_payload_sha256": "a" * 64,
                    "accounted_payload_bytes": 8,
                },
            },
            "emitted_payload": {
                "bytes": 8,
                "sha256": "a" * 64,
                "byte_identical_to_source_member": False,
            },
            "emitted_archive": {
                "byte_identical_to_source_archive": False,
            },
            "runtime_consumption_claim": False,
            "full_frame_inflate_output_parity_claim": False,
            "contest_axis_claim": False,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    monkeypatch.setattr(
        packet_compiler,
        "prove_pr106_sidecar_packet_ir_identity",
        fake_identity,
    )

    passed, output = module._run_pr106_sidecar_runtime_consumption_gate()

    assert passed is False
    assert "packet_ir_emit_payload_not_identity" in output
    assert "stored_zip_reemit_not_identity" in output
    assert "packet_ir_identity_not_passed" in output
