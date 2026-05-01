from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


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
