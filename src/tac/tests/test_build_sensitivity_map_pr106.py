from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

from tac.sensitivity_map import load_sensitivity_map

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_sensitivity_map_pr106.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_sensitivity_map_pr106", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stub_design_mode_is_never_allowed_for_cuda() -> None:
    module = _load_module()

    assert module._allow_stub_design_mode("cuda", True) is False
    assert module._allow_stub_design_mode("cuda", False) is False
    assert module._allow_stub_design_mode("cpu", False) is False
    assert module._allow_stub_design_mode("cpu", True) is True


def test_design_mode_writes_sha_tied_pair_manifest_and_bindings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()

    class TinyDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = nn.Conv2d(2, 3, 1)

    monkeypatch.setattr(
        module,
        "_load_pr106_decoder",
        lambda state_dict_path, device: TinyDecoder().to(device),
    )

    state_dict = tmp_path / "state_dict.pt"
    state_dict.write_bytes(b"state-dict-bytes")
    latents = tmp_path / "latents.pt"
    torch.save(torch.zeros(5, 28), latents)
    source_archive = tmp_path / "archive.zip"
    source_archive.write_bytes(b"pr106-archive")
    archive_sha = hashlib.sha256(source_archive.read_bytes()).hexdigest()
    extract_metadata = tmp_path / "metadata.json"
    extract_metadata.write_text(
        json.dumps(
            {
                "archive_path": str(source_archive),
                "archive_sha256": archive_sha,
                "archive_size_bytes": source_archive.stat().st_size,
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "sensitivity_map.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_sensitivity_map_pr106.py",
            "--state-dict",
            str(state_dict),
            "--latents",
            str(latents),
            "--extract-metadata",
            str(extract_metadata),
            "--source-archive",
            str(source_archive),
            "--out",
            str(out),
            "--device",
            "cpu",
            "--n-pairs",
            "3",
            "--allow-stub-design-mode",
        ],
    )

    assert module.main() == 0

    pair_manifest = tmp_path / "sensitivity_map.pair_manifest.json"
    assert pair_manifest.is_file()
    manifest = json.loads(pair_manifest.read_text(encoding="utf-8"))
    assert manifest["n_pairs"] == 3
    assert manifest["pairs"][2]["frame_indices"] == [4, 5]
    assert manifest["source_bindings"]["source_archive_sha256"] == archive_sha

    _, metadata = load_sensitivity_map(out)
    state_sha = hashlib.sha256(state_dict.read_bytes()).hexdigest()
    latents_sha = hashlib.sha256(latents.read_bytes()).hexdigest()
    manifest_sha = module.sensitivity_manifest_sha256(manifest)
    assert metadata["is_stub"] is True
    assert metadata["source_archive_sha256"] == archive_sha
    assert metadata["source_archive_bytes"] == source_archive.stat().st_size
    assert metadata["state_dict_sha256"] == state_sha
    assert metadata["model_sha256"] == state_sha
    assert metadata["latents_sha256"] == latents_sha
    assert metadata["pair_manifest_sha256"] == manifest_sha
    assert metadata["sample_plan_sha256"] == manifest_sha
    assert "ts_utc" not in metadata
