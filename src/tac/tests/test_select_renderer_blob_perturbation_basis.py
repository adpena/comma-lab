# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "select_renderer_blob_perturbation_basis.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "select_renderer_blob_perturbation_basis",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _asym_blob() -> bytes:
    header = {
        "version": 2,
        "layers": [
            {
                "name": "renderer.small",
                "shape": [2, 3, 1, 1],
                "bits": 8,
                "has_bias": False,
                "transposed": False,
                "is_linear": False,
            },
            {
                "name": "renderer.big",
                "shape": [4, 5, 2, 2],
                "bits": 8,
                "has_bias": False,
                "transposed": False,
                "is_linear": False,
            },
            {
                "name": "renderer.embedding",
                "shape": [3, 2],
                "bits": 8,
                "has_bias": False,
                "transposed": False,
                "is_embedding": True,
            },
        ],
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    out = bytearray(b"ASYM")
    out.extend(len(header_json).to_bytes(4, "little"))
    out.extend(header_json)

    def add_non_embedding(channels: int, fan_in: int, fill: int) -> None:
        blob = bytearray()
        for _ in range(channels):
            blob.extend(b"\x01\x3c")
            blob.extend(bytes([fill]) * fan_in)
        out.extend(len(blob).to_bytes(4, "little"))
        out.extend(blob)
        out.extend((0).to_bytes(4, "little"))

    add_non_embedding(channels=2, fan_in=3, fill=100)
    add_non_embedding(channels=4, fan_in=20, fill=120)
    emb = b"\x01\x3c" + bytes([110]) * 6
    out.extend(len(emb).to_bytes(4, "little"))
    out.extend(emb)
    return bytes(out)


def _asym_blob_with_transposed_largest() -> bytes:
    header = {
        "version": 2,
        "layers": [
            {
                "name": "renderer.transposed",
                "shape": [8, 4, 4, 4],
                "bits": 8,
                "has_bias": False,
                "transposed": True,
                "is_linear": False,
            },
            {
                "name": "renderer.conv",
                "shape": [3, 2, 1, 1],
                "bits": 8,
                "has_bias": False,
                "transposed": False,
                "is_linear": False,
            },
        ],
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    out = bytearray(b"ASYM")
    out.extend(len(header_json).to_bytes(4, "little"))
    out.extend(header_json)

    def add_blob(channels: int, fan_in: int, fill: int) -> None:
        blob = bytearray()
        for _ in range(channels):
            blob.extend(b"\x01\x3c")
            blob.extend(bytes([fill]) * fan_in)
        out.extend(len(blob).to_bytes(4, "little"))
        out.extend(blob)
        out.extend((0).to_bytes(4, "little"))

    add_blob(channels=4, fan_in=128, fill=100)
    add_blob(channels=3, fan_in=2, fill=120)
    return bytes(out)


def _write_archive(path: Path, renderer: bytes) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("renderer.bin", renderer)
        zf.writestr("masks.mkv", b"mask")
        zf.writestr("optimized_poses.bin", b"poses")
    return path


def test_parses_asym_regions_and_selects_largest_non_embedding_layer() -> None:
    module = _load_module()
    renderer = _asym_blob()

    header, atoms = module.select_basis_atoms(
        renderer,
        max_atoms=1,
        epsilons=[-1.0, 0.0, 1.0],
    )

    assert header["version"] == 2
    assert len(atoms) == 1
    atom = atoms[0]
    assert atom.member == "renderer.bin"
    assert atom.layer_name == "renderer.big"
    assert atom.blob_kind == "weight"
    assert atom.offset > 8 + len(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    assert renderer[atom.offset] == 120
    assert atom.margin_to_byte_range >= 1


def test_sensitivity_ranked_selection_prefers_high_response_channel() -> None:
    module = _load_module()
    renderer = _asym_blob()

    _, atoms = module.select_basis_atoms(
        renderer,
        max_atoms=2,
        epsilons=[-1.0, 0.0, 1.0],
        sensitivity_scores={
            "renderer.small.weight": [100.0, 0.0],
            "renderer.big.weight": [1.0, 2.0, 3.0, 4.0],
        },
        selection_mode="sensitivity-desc",
    )

    assert atoms[0].layer_name == "renderer.small"
    assert atoms[0].channel_index == 0
    assert atoms[0].sensitivity_score == 100.0
    assert atoms[0].selection_source == "sensitivity-desc"
    assert atoms[1].layer_name == "renderer.big"
    assert atoms[1].channel_index == 3


def test_transposed_payloads_are_excluded_by_default_for_map_compatibility() -> None:
    module = _load_module()
    renderer = _asym_blob_with_transposed_largest()

    _, atoms = module.select_basis_atoms(
        renderer,
        max_atoms=1,
        epsilons=[-1.0, 0.0, 1.0],
    )
    assert atoms[0].layer_name == "renderer.conv"

    _, forensic_atoms = module.select_basis_atoms(
        renderer,
        max_atoms=1,
        epsilons=[-1.0, 0.0, 1.0],
        include_transposed=True,
    )
    assert forensic_atoms[0].layer_name == "renderer.transposed"


def test_builds_basis_json_compatible_with_component_plan_loader(tmp_path: Path) -> None:
    module = _load_module()
    archive = _write_archive(tmp_path / "archive.zip", _asym_blob())
    output = tmp_path / "basis.json"

    summary = module.build_renderer_blob_perturbation_basis(
        archive=archive,
        output_json=output,
        epsilons=[-2.0, 0.0, 2.0],
        max_atoms=2,
        decode_verify=False,
    )

    payload = json.loads(output.read_text())
    assert payload["format"] == "perturbation_basis_v1"
    assert payload["extended_format"] == "renderer_blob_perturbation_basis_v1"
    assert payload["auth_eval_required"] == "cuda"
    assert payload["score_claim"] == "none"
    assert payload["atom_count"] == 2
    assert summary["atom_count"] == 2
    for atom in payload["atoms"]:
        assert atom["member"] == "renderer.bin"
        assert atom["offset"] >= 8
        assert atom["delta_per_epsilon"] == 1
        assert atom["margin_to_byte_range"] >= 2


def test_builds_sensitivity_ranked_basis_json_with_map_custody(tmp_path: Path) -> None:
    module = _load_module()
    archive = _write_archive(tmp_path / "archive.zip", _asym_blob())
    sensitivity_map = tmp_path / "combined_sensitivity_map.pt"
    output = tmp_path / "basis.json"
    torch.save(
        {
            "format": "tac_score_sensitivity_map_v1",
            "sensitivities": {
                "renderer.small.weight": torch.tensor([100.0, 0.0]),
                "renderer.big.weight": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            },
            "metadata": {"device": "cuda", "unit_test": True},
        },
        sensitivity_map,
    )

    summary = module.build_renderer_blob_perturbation_basis(
        archive=archive,
        output_json=output,
        epsilons=[-2.0, 0.0, 2.0],
        max_atoms=2,
        decode_verify=False,
        sensitivity_map=sensitivity_map,
        selection_mode="sensitivity-desc",
    )

    payload = json.loads(output.read_text())
    assert payload["selection_mode"] == "sensitivity-desc"
    assert payload["sensitivity_map"]["sha256"]
    assert payload["atoms"][0]["selection_source"] == "sensitivity-desc"
    assert payload["atoms"][0]["sensitivity_score"] == 100.0
    assert summary["atom_count"] == 2


def test_rejects_truncated_weight_blob() -> None:
    module = _load_module()
    renderer = bytearray(_asym_blob())
    del renderer[-3:]

    with pytest.raises(module.RendererBasisSelectionError, match="overruns|truncated"):
        module.parse_asym_payload_regions(bytes(renderer))


def test_cli_help_imports_without_scorer_dependencies() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--archive" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr
