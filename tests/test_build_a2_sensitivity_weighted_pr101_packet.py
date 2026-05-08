from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "build_a2_sensitivity_weighted_pr101_packet.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("build_a2_sensitivity_weighted_pr101_packet", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_tiny_pr101_contract(monkeypatch, tool) -> None:
    monkeypatch.setattr(tool, "FIXED_STATE_SCHEMA", (("a", (4,)), ("b", (4,))))
    monkeypatch.setattr(tool, "DECODER_BLOB_LEN", 4)
    monkeypatch.setattr(tool, "LATENT_BLOB_LEN", 2)
    monkeypatch.setattr(tool, "DECODER_STORAGE_ORDER", (0, 1))
    monkeypatch.setattr(tool, "DECODER_STREAM_ENDS", (2,))
    monkeypatch.setattr(tool, "CONV4_STORAGE_PERMS", {})
    monkeypatch.setattr(tool, "DECODER_BYTE_MAPS", {})
    monkeypatch.setattr(tool, "pack_brotli_stream", lambda raw, quality=11: b"S" + raw)
    monkeypatch.setattr(tool, "_decode_decoder_blob_for_closure", lambda _blob: {"a": object(), "b": object()})


def _install_fake_encoder(monkeypatch, tool, *, reference_blob: bytes = b"DECO") -> None:
    def fake_encode_rounded_decoder_blob(*, state_dict_path, selected_ks, brotli_quality):
        del state_dict_path
        blob = reference_blob if selected_ks == [1, 1] else b"A2DC"
        return blob, {
            "brotli_quality": brotli_quality,
            "decoder_blob_bytes": len(blob),
            "decoder_blob_sha256": tool._sha256_bytes(blob),
            "rel_err_l1_quantized_proxy": 0.0 if selected_ks == [1, 1] else 0.125,
            "abs_err_sum": 0.0,
            "abs_orig_sum": 1.0,
            "tensor_rows": [],
        }

    monkeypatch.setattr(tool, "_encode_rounded_decoder_blob", fake_encode_rounded_decoder_blob)


def _write_zip(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("x", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr(info, payload)


def _write_runtime(path: Path, *, patchable: bool = True) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    inflate = path / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8")
    inflate.chmod(0o755)
    (path / "inflate.py").write_text("print('unused in unit test')\n", encoding="utf-8")
    codec = path / "src" / "codec.py"
    codec.parent.mkdir(parents=True, exist_ok=True)
    (path / "src" / "model.py").write_text(
        """class HNeRVDecoder:
    pass
""",
        encoding="utf-8",
    )
    if patchable:
        codec.write_text(
            """import torch

DECODER_BLOB_LEN = 4
LATENT_BLOB_LEN = 2
N_PAIRS = 1
LATENT_DIM = 1
BASE_CHANNELS = 1
EVAL_SIZE = (1, 1)


def decode_decoder_compact(decoder_blob):
    return {"decoder": torch.tensor([decoder_blob[0]], dtype=torch.float32)}


def decode_latents_compact(latent_blob):
    return torch.tensor([[latent_blob[0]]], dtype=torch.float32)


def apply_latent_sidecar(latents, sidecar_blob):
    return latents


def parse_archive(archive_bytes):
    decoder_blob = archive_bytes[:DECODER_BLOB_LEN]
    latent_blob = archive_bytes[DECODER_BLOB_LEN:DECODER_BLOB_LEN + LATENT_BLOB_LEN]
    sidecar_blob = archive_bytes[DECODER_BLOB_LEN + LATENT_BLOB_LEN:]
    if not decoder_blob or not latent_blob:
        raise ValueError("bad compact archive")
    meta = {
        "n_pairs": N_PAIRS,
        "latent_dim": LATENT_DIM,
        "base_channels": BASE_CHANNELS,
        "eval_size": list(EVAL_SIZE),
    }
    latents = apply_latent_sidecar(decode_latents_compact(latent_blob), sidecar_blob)
    return decode_decoder_compact(decoder_blob), latents, meta
""",
            encoding="utf-8",
        )
    else:
        codec.write_text("def unrelated():\n    return None\n", encoding="utf-8")
    return path


def _write_state_dict(path: Path) -> Path:
    torch.save(
        {
            "a": torch.tensor([1.0, 2.0, -3.0, 4.0]),
            "b": torch.tensor([-5.0, 6.0, 7.0, -8.0]),
        },
        path,
    )
    return path


def _write_a2_manifest(path: Path, selected_ks: list[int]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "phase_a2_sensitivity_weighted_lossy_coarsening.v1",
                "tool": "tools/sensitivity_weighted_lossy_coarsening.py",
                "status": "completed_local_diagnostic",
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "dispatch_blockers": [
                    "diagnostic_or_stub_sensitivity_map_not_score_authority",
                    "no_exact_cuda_auth_eval",
                ],
                "sensitivity_artifact": {
                    "allow_diagnostic_sensitivity": True,
                    "metadata_blockers": ["is_stub=true"],
                    "path": "stub.pt",
                    "status": "diagnostic_allowed",
                },
                "weighted_k_allocations": [
                    {
                        "joint_encoder_extras": {
                            "archive_overhead_bytes": 12,
                            "payload_brotli_bytes": 99,
                            "side_info_bytes": 2,
                        },
                        "rms_target": 0.05,
                        "rel_err": 0.0125,
                        "total_bytes": 123,
                        "selected_Ks": selected_ks,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_builds_byte_closed_packet_ladder_with_non_promotable_manifest(tmp_path: Path, monkeypatch) -> None:
    tool = _load_tool()
    _install_tiny_pr101_contract(monkeypatch, tool)
    _install_fake_encoder(monkeypatch, tool)
    source_archive = tmp_path / "source" / "archive.zip"
    _write_zip(source_archive, b"DECO" + b"LA" + b"SIDE")
    runtime = _write_runtime(tmp_path / "runtime")
    state_dict = _write_state_dict(tmp_path / "state.pt")
    a2_manifest = _write_a2_manifest(tmp_path / "a2.json", [2, 1])

    manifest = tool.build_packet_ladder(
        a2_manifest_path=a2_manifest,
        state_dict_path=state_dict,
        source_archive=source_archive,
        source_runtime_dir=runtime,
        output_dir=tmp_path / "out",
        recorded_at_utc=datetime(2026, 5, 8, 12, 0, tzinfo=UTC),
        force=False,
    )

    assert manifest["status"] == "completed_byte_closed_packet_ladder"
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    variant = manifest["variants"][0]
    assert manifest["packet_closure"]["state_dict_reproduces_source_decoder"]["passed"] is True
    assert variant["packet_closure"]["byte_closed_packet_built"] is True
    assert variant["packet_closure"]["runtime_consumes_changed_archive_bytes"] is True
    assert "no_byte_closed_runtime_packet_built" not in manifest["dispatch_blockers"]
    assert "packet_local_inflate_parity_not_run" in manifest["dispatch_blockers"]
    assert "packet_local_inflate_parity_not_run" not in manifest["packet_closure"]["cleared_blockers"]
    assert "diagnostic_or_stub_sensitivity_map_not_score_authority" in manifest["dispatch_blockers"]
    assert "diagnostic_or_stub_sensitivity_map_not_score_authority" in variant["dispatch_blockers"]
    assert "is_stub=true" in variant["dispatch_blockers"]
    assert variant["archive_member_manifest"]["layout_magic"] == "A2K1"
    assert variant["archive_member_manifest"]["decoder_len_field_matches_decoder_blob"] is True
    assert variant["runtime_packet"]["runtime_patch"]["codec_parse_archive_supports_a2_length_prefix"] is True
    assert variant["runtime_packet"]["runtime_checks"]["inflate_sh_bash_n"]["passed"] is True
    assert variant["runtime_packet"]["runtime_checks"]["packet_local_parse_smoke"]["passed"] is True
    assert variant["runtime_packet"]["report"]["relpath"] == "report.txt"
    assert (tmp_path / "out" / "variants" / variant["variant_id"] / "packet" / "report.txt").is_file()
    assert variant["proxy_vs_materialized"]["authoritative_bytes_field"] == "candidate_archive.bytes"
    assert variant["proxy_vs_materialized"]["reported_total_bytes"] == 123
    assert variant["score_claim"] is False
    assert variant["ready_for_exact_eval_dispatch"] is False

    candidate_archive = Path(variant["candidate_archive_relpath"])
    with zipfile.ZipFile(candidate_archive) as zf:
        payload = zf.read("x")
    assert payload.startswith(b"A2K1")
    assert variant["candidate_archive"]["members"][0]["sha256"] == tool._sha256_bytes(payload)
    patched_codec = Path(variant["runtime_packet"]["runtime_patch"]["codec_path"])
    assert "A2_LOSSY_COARSENING_MAGIC" in patched_codec.read_text(encoding="utf-8")


def test_noop_schedule_is_detected_and_non_promotable(tmp_path: Path, monkeypatch) -> None:
    tool = _load_tool()
    _install_tiny_pr101_contract(monkeypatch, tool)
    _install_fake_encoder(monkeypatch, tool)
    source_archive = tmp_path / "source" / "archive.zip"
    _write_zip(source_archive, b"DECO" + b"LA" + b"SIDE")
    runtime = _write_runtime(tmp_path / "runtime")
    state_dict = _write_state_dict(tmp_path / "state.pt")
    a2_manifest = _write_a2_manifest(tmp_path / "a2.json", [1, 1])

    manifest = tool.build_packet_ladder(
        a2_manifest_path=a2_manifest,
        state_dict_path=state_dict,
        source_archive=source_archive,
        source_runtime_dir=runtime,
        output_dir=tmp_path / "out",
        recorded_at_utc=datetime(2026, 5, 8, 12, 0, tzinfo=UTC),
    )

    variant = manifest["variants"][0]
    assert variant["noop_detection"]["is_noop"] is True
    assert "schedule_all_ones_no_semantic_coarsening" in variant["noop_detection"]["reasons"]
    assert "schedule_all_ones_no_semantic_coarsening" in variant["dispatch_blockers"]
    assert variant["score_affecting_payload_changed"] is False
    assert variant["score_claim"] is False
    assert variant["ready_for_exact_eval_dispatch"] is False


def test_blocked_manifest_names_missing_runtime_wire_contract(tmp_path: Path, monkeypatch) -> None:
    tool = _load_tool()
    _install_tiny_pr101_contract(monkeypatch, tool)
    _install_fake_encoder(monkeypatch, tool)
    source_archive = tmp_path / "source" / "archive.zip"
    _write_zip(source_archive, b"DECO" + b"LA" + b"SIDE")
    runtime = _write_runtime(tmp_path / "bad_runtime", patchable=False)
    state_dict = _write_state_dict(tmp_path / "state.pt")
    a2_manifest = _write_a2_manifest(tmp_path / "a2.json", [2, 1])
    blocked_json = tmp_path / "blocked.json"

    rc = tool.main(
        [
            "--a2-manifest",
            str(a2_manifest),
            "--state-dict",
            str(state_dict),
            "--source-archive",
            str(source_archive),
            "--source-runtime-dir",
            str(runtime),
            "--output-dir",
            str(tmp_path / "out"),
            "--json-out",
            str(blocked_json),
            "--now-utc",
            "2026-05-08T12:00:00Z",
        ]
    )

    assert rc == 2
    manifest = json.loads(blocked_json.read_text(encoding="utf-8"))
    assert manifest["status"] == "blocked_fail_closed"
    assert manifest["packet_closure"]["byte_closed_packet_ladder_built"] is False
    assert (
        "runtime_codec_parse_archive_patch_anchor_missing"
        in manifest["packet_closure"]["missing_wire_contracts"]
    )
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_run_inflate_parity_cli_uses_packet_subdir_archive(tmp_path: Path, monkeypatch) -> None:
    tool = _load_tool()
    _install_tiny_pr101_contract(monkeypatch, tool)
    _install_fake_encoder(monkeypatch, tool)
    source_archive = tmp_path / "source" / "archive.zip"
    _write_zip(source_archive, b"DECO" + b"LA" + b"SIDE")
    runtime = _write_runtime(tmp_path / "runtime")
    state_dict = _write_state_dict(tmp_path / "state.pt")
    a2_manifest = _write_a2_manifest(tmp_path / "a2.json", [2, 1])
    output_dir = tmp_path / "out"
    calls: list[tuple[Path, Path]] = []

    def fake_verify_inflate_parity(
        *,
        packet_dir,
        candidate_archive_path,
        source_archive_path,
        timeout_seconds,
    ):
        del source_archive_path, timeout_seconds
        calls.append((Path(packet_dir), Path(candidate_archive_path)))
        return {"passed": True, "cleared_blockers": ["packet_local_inflate_parity_not_run"]}

    monkeypatch.setattr(tool, "verify_inflate_parity", fake_verify_inflate_parity)

    rc = tool.main(
        [
            "--a2-manifest",
            str(a2_manifest),
            "--state-dict",
            str(state_dict),
            "--source-archive",
            str(source_archive),
            "--source-runtime-dir",
            str(runtime),
            "--output-dir",
            str(output_dir),
            "--variant-limit",
            "1",
            "--run-inflate-parity",
        ]
    )

    assert rc == 0
    manifest = json.loads((output_dir / "a2_packet_ladder_manifest.json").read_text())
    variant_id = manifest["variants"][0]["variant_id"]
    assert calls == [
        (
            output_dir / "variants" / variant_id / "packet",
            output_dir / "variants" / variant_id / "packet" / "archive.zip",
        )
    ]


def test_refuses_output_directory_overlapping_source_runtime(tmp_path: Path, monkeypatch) -> None:
    tool = _load_tool()
    _install_tiny_pr101_contract(monkeypatch, tool)
    _install_fake_encoder(monkeypatch, tool)
    source_archive = tmp_path / "source" / "archive.zip"
    _write_zip(source_archive, b"DECO" + b"LA" + b"SIDE")
    runtime = _write_runtime(tmp_path / "runtime")
    state_dict = _write_state_dict(tmp_path / "state.pt")
    a2_manifest = _write_a2_manifest(tmp_path / "a2.json", [2, 1])
    blocked_json = tmp_path / "blocked.json"

    rc = tool.main(
        [
            "--a2-manifest",
            str(a2_manifest),
            "--state-dict",
            str(state_dict),
            "--source-archive",
            str(source_archive),
            "--source-runtime-dir",
            str(runtime),
            "--output-dir",
            str(runtime / "nested_out"),
            "--json-out",
            str(blocked_json),
            "--force",
            "--now-utc",
            "2026-05-08T12:00:00Z",
        ]
    )

    assert rc == 2
    manifest = json.loads(blocked_json.read_text(encoding="utf-8"))
    assert "output_directory_overlaps_source_runtime_tree" in manifest["dispatch_blockers"]


def test_refuses_output_directory_that_would_delete_source_archive(tmp_path: Path, monkeypatch) -> None:
    tool = _load_tool()
    _install_tiny_pr101_contract(monkeypatch, tool)
    _install_fake_encoder(monkeypatch, tool)
    source_archive = tmp_path / "source" / "archive.zip"
    _write_zip(source_archive, b"DECO" + b"LA" + b"SIDE")
    runtime = _write_runtime(tmp_path / "runtime")
    state_dict = _write_state_dict(tmp_path / "state.pt")
    a2_manifest = _write_a2_manifest(tmp_path / "a2.json", [2, 1])
    blocked_json = tmp_path / "blocked.json"

    rc = tool.main(
        [
            "--a2-manifest",
            str(a2_manifest),
            "--state-dict",
            str(state_dict),
            "--source-archive",
            str(source_archive),
            "--source-runtime-dir",
            str(runtime),
            "--output-dir",
            str(source_archive.parent),
            "--json-out",
            str(blocked_json),
            "--force",
            "--now-utc",
            "2026-05-08T12:00:00Z",
        ]
    )

    assert rc == 2
    manifest = json.loads(blocked_json.read_text(encoding="utf-8"))
    assert "output_directory_contains_source_archive" in manifest["dispatch_blockers"]
    assert source_archive.is_file()


def test_refuses_unsafe_source_zip_member_name(tmp_path: Path, monkeypatch) -> None:
    tool = _load_tool()
    _install_tiny_pr101_contract(monkeypatch, tool)
    _install_fake_encoder(monkeypatch, tool)
    source_archive = tmp_path / "source" / "archive.zip"
    source_archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_archive, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("../x", b"DECO" + b"LA" + b"SIDE")
    runtime = _write_runtime(tmp_path / "runtime")
    state_dict = _write_state_dict(tmp_path / "state.pt")
    a2_manifest = _write_a2_manifest(tmp_path / "a2.json", [2, 1])
    blocked_json = tmp_path / "blocked.json"

    rc = tool.main(
        [
            "--a2-manifest",
            str(a2_manifest),
            "--state-dict",
            str(state_dict),
            "--source-archive",
            str(source_archive),
            "--source-runtime-dir",
            str(runtime),
            "--output-dir",
            str(tmp_path / "out"),
            "--json-out",
            str(blocked_json),
            "--now-utc",
            "2026-05-08T12:00:00Z",
        ]
    )

    assert rc == 2
    manifest = json.loads(blocked_json.read_text(encoding="utf-8"))
    assert "unsafe_zip_member_name" in manifest["dispatch_blockers"]


def test_refuses_state_dict_that_does_not_reproduce_source_decoder(tmp_path: Path, monkeypatch) -> None:
    tool = _load_tool()
    _install_tiny_pr101_contract(monkeypatch, tool)
    _install_fake_encoder(monkeypatch, tool, reference_blob=b"MISS")
    source_archive = tmp_path / "source" / "archive.zip"
    _write_zip(source_archive, b"DECO" + b"LA" + b"SIDE")
    runtime = _write_runtime(tmp_path / "runtime")
    state_dict = _write_state_dict(tmp_path / "state.pt")
    a2_manifest = _write_a2_manifest(tmp_path / "a2.json", [2, 1])
    blocked_json = tmp_path / "blocked.json"

    rc = tool.main(
        [
            "--a2-manifest",
            str(a2_manifest),
            "--state-dict",
            str(state_dict),
            "--source-archive",
            str(source_archive),
            "--source-runtime-dir",
            str(runtime),
            "--output-dir",
            str(tmp_path / "out"),
            "--json-out",
            str(blocked_json),
            "--now-utc",
            "2026-05-08T12:00:00Z",
        ]
    )

    assert rc == 2
    manifest = json.loads(blocked_json.read_text(encoding="utf-8"))
    assert "state_dict_does_not_reproduce_source_decoder_blob" in manifest["dispatch_blockers"]


def test_refuses_wrong_a2_manifest_schema(tmp_path: Path, monkeypatch) -> None:
    tool = _load_tool()
    _install_tiny_pr101_contract(monkeypatch, tool)
    _install_fake_encoder(monkeypatch, tool)
    source_archive = tmp_path / "source" / "archive.zip"
    _write_zip(source_archive, b"DECO" + b"LA" + b"SIDE")
    runtime = _write_runtime(tmp_path / "runtime")
    state_dict = _write_state_dict(tmp_path / "state.pt")
    a2_manifest = _write_a2_manifest(tmp_path / "a2.json", [2, 1])
    payload = json.loads(a2_manifest.read_text(encoding="utf-8"))
    payload["schema"] = "wrong"
    a2_manifest.write_text(json.dumps(payload), encoding="utf-8")
    blocked_json = tmp_path / "blocked.json"

    rc = tool.main(
        [
            "--a2-manifest",
            str(a2_manifest),
            "--state-dict",
            str(state_dict),
            "--source-archive",
            str(source_archive),
            "--source-runtime-dir",
            str(runtime),
            "--output-dir",
            str(tmp_path / "out"),
            "--json-out",
            str(blocked_json),
            "--now-utc",
            "2026-05-08T12:00:00Z",
        ]
    )

    assert rc == 2
    manifest = json.loads(blocked_json.read_text(encoding="utf-8"))
    assert "a2_manifest_wrong_schema" in manifest["dispatch_blockers"]


def test_refuses_authoritative_a2_manifest(tmp_path: Path, monkeypatch) -> None:
    tool = _load_tool()
    _install_tiny_pr101_contract(monkeypatch, tool)
    _install_fake_encoder(monkeypatch, tool)
    source_archive = tmp_path / "source" / "archive.zip"
    _write_zip(source_archive, b"DECO" + b"LA" + b"SIDE")
    runtime = _write_runtime(tmp_path / "runtime")
    state_dict = _write_state_dict(tmp_path / "state.pt")
    a2_manifest = _write_a2_manifest(tmp_path / "a2.json", [2, 1])
    payload = json.loads(a2_manifest.read_text(encoding="utf-8"))
    payload["score_claim"] = True
    a2_manifest.write_text(json.dumps(payload), encoding="utf-8")
    blocked_json = tmp_path / "blocked.json"

    rc = tool.main(
        [
            "--a2-manifest",
            str(a2_manifest),
            "--state-dict",
            str(state_dict),
            "--source-archive",
            str(source_archive),
            "--source-runtime-dir",
            str(runtime),
            "--output-dir",
            str(tmp_path / "out"),
            "--json-out",
            str(blocked_json),
            "--now-utc",
            "2026-05-08T12:00:00Z",
        ]
    )

    assert rc == 2
    manifest = json.loads(blocked_json.read_text(encoding="utf-8"))
    assert "a2_manifest_score_claim_not_false" in manifest["dispatch_blockers"]


def test_refuses_backslash_zip_member_name(tmp_path: Path, monkeypatch) -> None:
    tool = _load_tool()
    _install_tiny_pr101_contract(monkeypatch, tool)
    _install_fake_encoder(monkeypatch, tool)
    source_archive = tmp_path / "source" / "archive.zip"
    source_archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_archive, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("bad\\x", b"DECO" + b"LA" + b"SIDE")
    runtime = _write_runtime(tmp_path / "runtime")
    state_dict = _write_state_dict(tmp_path / "state.pt")
    a2_manifest = _write_a2_manifest(tmp_path / "a2.json", [2, 1])
    blocked_json = tmp_path / "blocked.json"

    rc = tool.main(
        [
            "--a2-manifest",
            str(a2_manifest),
            "--state-dict",
            str(state_dict),
            "--source-archive",
            str(source_archive),
            "--source-runtime-dir",
            str(runtime),
            "--output-dir",
            str(tmp_path / "out"),
            "--json-out",
            str(blocked_json),
            "--now-utc",
            "2026-05-08T12:00:00Z",
        ]
    )

    assert rc == 2
    manifest = json.loads(blocked_json.read_text(encoding="utf-8"))
    assert "unsafe_zip_member_name" in manifest["dispatch_blockers"]


def test_refuses_zip_local_central_header_mismatch(tmp_path: Path, monkeypatch) -> None:
    tool = _load_tool()
    _install_tiny_pr101_contract(monkeypatch, tool)
    _install_fake_encoder(monkeypatch, tool)
    source_archive = tmp_path / "source" / "archive.zip"
    _write_zip(source_archive, b"DECO" + b"LA" + b"SIDE")
    raw = bytearray(source_archive.read_bytes())
    header = raw.index(b"PK\x03\x04")
    name_len = int.from_bytes(raw[header + 26 : header + 28], "little")
    assert name_len == 1
    raw[header + 30 : header + 31] = b"y"
    source_archive.write_bytes(raw)
    runtime = _write_runtime(tmp_path / "runtime")
    state_dict = _write_state_dict(tmp_path / "state.pt")
    a2_manifest = _write_a2_manifest(tmp_path / "a2.json", [2, 1])
    blocked_json = tmp_path / "blocked.json"

    rc = tool.main(
        [
            "--a2-manifest",
            str(a2_manifest),
            "--state-dict",
            str(state_dict),
            "--source-archive",
            str(source_archive),
            "--source-runtime-dir",
            str(runtime),
            "--output-dir",
            str(tmp_path / "out"),
            "--json-out",
            str(blocked_json),
            "--now-utc",
            "2026-05-08T12:00:00Z",
        ]
    )

    assert rc == 2
    manifest = json.loads(blocked_json.read_text(encoding="utf-8"))
    assert "zip_local_header_name_mismatch" in manifest["dispatch_blockers"]
