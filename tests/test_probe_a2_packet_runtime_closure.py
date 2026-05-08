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


def _write_packet_runtime(packet_dir: Path) -> Path:
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
    state = {"weight": torch.ones(1)}
    latents = torch.zeros(3, 2)
    meta = {"n_pairs": 3, "latent_dim": 2, "base_channels": 1, "eval_size": [1, 1]}
    return state, latents, meta
""",
        encoding="utf-8",
    )
    return packet_dir


def test_probe_verifies_source_fallback_and_candidate_a2_runtime_closure(tmp_path: Path) -> None:
    tool = _load_tool()
    packet_dir = _write_packet_runtime(tmp_path / "packet")
    source_archive = _write_zip(tmp_path / "source.zip", b"SRC0")
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

    assert rc == 0
    manifest = json.loads(json_out.read_text(encoding="utf-8"))
    assert manifest["status"] == "runtime_closure_verified_no_score"
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["runtime_closure"]["verified"] is True
    assert manifest["runtime_closure"]["cleared_blockers"] == ["packet_local_inflate_parity_not_run"]
    assert manifest["source_probe"]["archive_member_starts_with_a2_magic"] is False
    assert manifest["candidate_probe"]["archive_member_starts_with_a2_magic"] is True
    assert manifest["candidate_probe"]["model_state_load_strict_passed"] is True
    assert manifest["candidate_probe"]["latents_shape"] == [3, 2]
    assert manifest["ready_for_exact_eval_dispatch"] is False


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
