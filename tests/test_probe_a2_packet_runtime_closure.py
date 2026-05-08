from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "probe_a2_packet_runtime_closure.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("probe_a2_packet_runtime_closure", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_zip(path: Path, payload: bytes, *, name: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr(info, payload)
    return path


def _write_packet_runtime(packet_dir: Path, *, ignore_decoder_bytes: bool = False) -> Path:
    src = packet_dir / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "model.py").write_text(
        """import torch
import torch.nn as nn


class HNeRVDecoder(nn.Module):
    def __init__(self, latent_dim=2, base_channels=1, eval_size=(1, 1)):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
""",
        encoding="utf-8",
    )
    state_expr = "torch.ones(1)" if ignore_decoder_bytes else "torch.tensor([decoder_blob[0]], dtype=torch.float32)"
    (src / "codec.py").write_text(
        """import torch
from model import HNeRVDecoder


def parse_archive(archive_bytes):
    if archive_bytes.startswith(b"A2K1"):
        if len(archive_bytes) < 8:
            raise ValueError("bad A2 header")
        decoder_len = int.from_bytes(archive_bytes[4:8], "little")
        decoder_blob = archive_bytes[8:8 + decoder_len]
    else:
        decoder_blob = archive_bytes[:4]
    if not decoder_blob:
        raise ValueError("bad compact archive")
    state = {"weight": STATE_EXPR}
    latents = torch.zeros(3, 2)
    meta = {"n_pairs": 3, "latent_dim": 2, "base_channels": 1, "eval_size": [1, 1]}
    return state, latents, meta
""".replace("STATE_EXPR", state_expr),
        encoding="utf-8",
    )
    return packet_dir


def _write_candidate_manifest(
    path: Path,
    *,
    tool,
    candidate_archive: Path,
    semantic_payload_changed: bool = True,
    blockers: list[str] | None = None,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "a2_packet_ladder_variant.v1",
                "status": "unit_test_candidate",
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "candidate_archive": {
                    "bytes": candidate_archive.stat().st_size,
                    "sha256": tool.sha256_file(candidate_archive),
                },
                "semantic_payload_changed": semantic_payload_changed,
                "dispatch_blockers": blockers
                or [
                    "diagnostic_or_stub_sensitivity_map_not_score_authority",
                    "is_stub=true",
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_probe_verifies_source_fallback_and_candidate_a2_runtime_closure(tmp_path: Path) -> None:
    tool = _load_tool()
    packet_dir = _write_packet_runtime(tmp_path / "packet")
    source_archive = _write_zip(tmp_path / "source.zip", b"SRC0")
    candidate_archive = _write_zip(tmp_path / "candidate.zip", b"A2K1" + (4).to_bytes(4, "little") + b"CAND")
    candidate_manifest = _write_candidate_manifest(
        packet_dir.parent / "candidate_manifest.json",
        tool=tool,
        candidate_archive=candidate_archive,
    )
    json_out = tmp_path / "probe.json"

    rc = tool.main(
        [
            "--packet-dir",
            str(packet_dir),
            "--source-archive",
            str(source_archive),
            "--candidate-archive",
            str(candidate_archive),
            "--candidate-manifest",
            str(candidate_manifest),
            "--json-out",
            str(json_out),
            "--now-utc",
            "2026-05-08T12:00:00Z",
        ]
    )

    assert rc == 0
    manifest = json.loads(json_out.read_text(encoding="utf-8"))
    assert manifest["status"] == "runtime_closure_verified_no_score"
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["runtime_closure"]["verified"] is True
    assert manifest["runtime_closure"]["decoded_decoder_state_changed"] is True
    assert manifest["runtime_closure"]["semantic_payload_changed"] is True
    assert manifest["runtime_closure"]["cleared_blockers"] == []
    assert "diagnostic_or_stub_sensitivity_map_not_score_authority" in manifest["dispatch_blockers"]
    assert "is_stub=true" in manifest["dispatch_blockers"]
    assert manifest["source_probe"]["archive_member_starts_with_a2_magic"] is False
    assert manifest["candidate_probe"]["archive_member_starts_with_a2_magic"] is True
    assert manifest["source_probe"]["decoder_state_sha256"] != manifest["candidate_probe"]["decoder_state_sha256"]
    assert manifest["candidate_probe"]["model_state_load_strict_passed"] is True
    assert manifest["candidate_probe"]["latents_shape"] == [3, 2]
    assert manifest["score_affecting_payload_changed"] is True
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_probe_marks_missing_candidate_manifest_non_promotable(tmp_path: Path) -> None:
    tool = _load_tool()
    packet_dir = _write_packet_runtime(tmp_path / "packet")
    source_archive = _write_zip(tmp_path / "source.zip", b"SRC0")
    candidate_archive = _write_zip(
        tmp_path / "candidate.zip",
        b"A2K1" + (4).to_bytes(4, "little") + b"CAND",
    )
    json_out = tmp_path / "probe.json"

    rc = tool.main(
        [
            "--packet-dir",
            str(packet_dir),
            "--source-archive",
            str(source_archive),
            "--candidate-archive",
            str(candidate_archive),
            "--json-out",
            str(json_out),
            "--now-utc",
            "2026-05-08T12:00:00Z",
        ]
    )

    assert rc == 0
    manifest = json.loads(json_out.read_text(encoding="utf-8"))
    assert manifest["candidate_manifest"]["present"] is False
    assert "candidate_manifest_missing_runtime_closure_not_promotable" in manifest["dispatch_blockers"]
    assert manifest["score_affecting_payload_changed"] is False


def test_probe_blocks_manifest_semantic_change_without_decoded_tensor_change(tmp_path: Path) -> None:
    tool = _load_tool()
    packet_dir = _write_packet_runtime(tmp_path / "packet", ignore_decoder_bytes=True)
    source_archive = _write_zip(tmp_path / "source.zip", b"SRC0")
    candidate_archive = _write_zip(
        tmp_path / "candidate.zip",
        b"A2K1" + (4).to_bytes(4, "little") + b"CAND",
    )
    candidate_manifest = _write_candidate_manifest(
        packet_dir.parent / "candidate_manifest.json",
        tool=tool,
        candidate_archive=candidate_archive,
        semantic_payload_changed=True,
    )
    json_out = tmp_path / "probe.json"

    rc = tool.main(
        [
            "--packet-dir",
            str(packet_dir),
            "--source-archive",
            str(source_archive),
            "--candidate-archive",
            str(candidate_archive),
            "--candidate-manifest",
            str(candidate_manifest),
            "--json-out",
            str(json_out),
            "--now-utc",
            "2026-05-08T12:00:00Z",
        ]
    )

    assert rc == 2
    manifest = json.loads(json_out.read_text(encoding="utf-8"))
    assert "semantic_payload_changed_but_decoded_decoder_state_unchanged" in manifest["dispatch_blockers"]


def test_probe_blocks_candidate_without_a2_magic(tmp_path: Path) -> None:
    tool = _load_tool()
    packet_dir = _write_packet_runtime(tmp_path / "packet")
    source_archive = _write_zip(tmp_path / "source.zip", b"SRC0")
    candidate_archive = _write_zip(tmp_path / "candidate.zip", b"CAND")
    json_out = tmp_path / "probe.json"

    rc = tool.main(
        [
            "--packet-dir",
            str(packet_dir),
            "--source-archive",
            str(source_archive),
            "--candidate-archive",
            str(candidate_archive),
            "--json-out",
            str(json_out),
            "--now-utc",
            "2026-05-08T12:00:00Z",
        ]
    )

    assert rc == 2
    manifest = json.loads(json_out.read_text(encoding="utf-8"))
    assert manifest["status"] == "blocked_fail_closed"
    assert "candidate_archive_missing_a2_magic" in manifest["dispatch_blockers"]
    assert manifest["score_claim"] is False


def test_probe_blocks_unsafe_member_name(tmp_path: Path) -> None:
    tool = _load_tool()
    packet_dir = _write_packet_runtime(tmp_path / "packet")
    source_archive = _write_zip(tmp_path / "source.zip", b"SRC0", name="../x")
    candidate_archive = _write_zip(tmp_path / "candidate.zip", b"A2K1" + (4).to_bytes(4, "little") + b"CAND")
    json_out = tmp_path / "probe.json"

    rc = tool.main(
        [
            "--packet-dir",
            str(packet_dir),
            "--source-archive",
            str(source_archive),
            "--candidate-archive",
            str(candidate_archive),
            "--json-out",
            str(json_out),
            "--now-utc",
            "2026-05-08T12:00:00Z",
        ]
    )

    assert rc == 2
    manifest = json.loads(json_out.read_text(encoding="utf-8"))
    assert "unsafe_zip_member_name" in manifest["dispatch_blockers"]
