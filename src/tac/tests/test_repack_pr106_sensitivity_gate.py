from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from tac.sensitivity_map import save_sensitivity_map

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "repack_pr106_with_water_filling.py"


def _load_repack_module(monkeypatch: pytest.MonkeyPatch):
    fake_codec = types.ModuleType("codec")
    fake_codec.encode_decoder = lambda sd: b"encoded"
    fake_codec.quantize_state_dict = lambda sd: sd
    monkeypatch.setitem(sys.modules, "codec", fake_codec)

    spec = importlib.util.spec_from_file_location("repack_pr106_gate_under_test", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repack_refuses_stub_sensitivity_before_archive_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_repack_module(monkeypatch)
    state_dict = tmp_path / "state_dict.pt"
    torch.save({"decoder.block.weight": torch.ones(2, 2, 1, 1)}, state_dict)
    sensitivity = tmp_path / "sensitivity_map.pt"
    save_sensitivity_map(
        sensitivity,
        {"decoder.block.weight": torch.ones(2)},
        metadata={"device": "cpu", "is_stub": True, "tag": "[stub-design-mode]"},
    )
    source_archive = tmp_path / "archive.zip"
    source_archive.write_bytes(b"not-read-before-gate")
    out_dir = tmp_path / "out"

    with pytest.raises(ValueError, match=r"refusing to build.*is_stub"):
        module.repack_pr106_with_water_filling(
            state_dict,
            sensitivity,
            source_archive,
            out_dir,
            verbose=False,
        )

    assert not (out_dir / "apogee_v2_archive.zip").exists()
