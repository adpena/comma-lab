# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from tac.neural_weight_corpus import (
    CorpusManifestError,
    build_corpus_from_checkpoints,
    build_corpus_from_manifest,
    build_corpus_manifest,
    canonical_manifest_json,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _save_checkpoint(path: Path, tensors: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": tensors}, path)
    return path


def _tensor_entries(manifest: dict, checkpoint_name: str) -> dict[str, dict]:
    for file_entry in manifest["files"]:
        if Path(file_entry["path"]).name == checkpoint_name:
            return {entry["name"]: entry for entry in file_entry["tensors"]}
    raise AssertionError(f"checkpoint {checkpoint_name!r} not found in manifest")


def _normalized_blocks(tensor: torch.Tensor, block_size: int) -> torch.Tensor:
    flat = tensor.reshape(-1).float()
    blocks = flat[: (flat.numel() // block_size) * block_size].reshape(-1, block_size)
    scales = blocks.abs().amax(dim=1).clamp(min=1e-8)
    return blocks / scales.unsqueeze(1)


def test_corpus_manifest_generation_is_deterministic(tmp_path):
    a = _save_checkpoint(
        tmp_path / "b" / "renderer_b.pt",
        {"weight": torch.arange(16, dtype=torch.float32).reshape(4, 4)},
    )
    b = _save_checkpoint(
        tmp_path / "a" / "renderer_a.pt",
        {"weight": torch.arange(16, 32, dtype=torch.float32).reshape(4, 4)},
    )

    manifest1 = build_corpus_manifest(
        [a, b],
        block_size=4,
        max_files=10,
        max_blocks_per_ckpt=100,
        min_checkpoint_bytes=0,
        corpus_dir=tmp_path,
    )
    manifest2 = build_corpus_manifest(
        [b, a],
        block_size=4,
        max_files=10,
        max_blocks_per_ckpt=100,
        min_checkpoint_bytes=0,
        corpus_dir=tmp_path,
    )

    assert canonical_manifest_json(manifest1) == canonical_manifest_json(manifest2)
    assert [Path(f["path"]).name for f in manifest1["files"]] == [
        "renderer_a.pt",
        "renderer_b.pt",
    ]
    assert manifest1["selection"]["max_files"] == 10
    assert manifest1["selection"]["max_blocks_per_checkpoint"] == 100


def test_manifest_records_bias_int_tiny_exclusion_reasons(tmp_path):
    tiny_file = tmp_path / "00_tiny_file.pt"
    tiny_file.write_bytes(b"x")
    ckpt = _save_checkpoint(
        tmp_path / "01_model.pt",
        {
            "conv.weight": torch.randn(2, 4, 2, 2),
            "conv.bias": torch.randn(8),
            "running_idx": torch.arange(8, dtype=torch.long),
            "tiny.weight": torch.randn(1, 3),
        },
    )

    manifest = build_corpus_manifest(
        [tiny_file, ckpt],
        block_size=4,
        max_files=10,
        max_blocks_per_ckpt=100,
        min_checkpoint_bytes=2,
        corpus_dir=tmp_path,
    )

    file_reasons = {Path(f["path"]).name: f["exclusion_reason"] for f in manifest["files"]}
    assert file_reasons["00_tiny_file.pt"] == "checkpoint_too_small"

    entries = _tensor_entries(manifest, "01_model.pt")
    assert entries["conv.weight"]["selected"] is True
    assert entries["conv.bias"]["exclusion_reason"] == "bias_1d_small"
    assert entries["running_idx"]["exclusion_reason"] == "non_floating_tensor"
    assert entries["tiny.weight"]["exclusion_reason"] == "tensor_too_small"


def test_manifest_replay_uses_stable_file_tensor_and_block_order(tmp_path):
    a_first = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]],
        dtype=torch.float32,
    )
    a_second = torch.tensor(
        [[10.0, 20.0, 30.0, 40.0], [3.0, 6.0, 9.0, 12.0]],
        dtype=torch.float32,
    )
    z_only = torch.tensor(
        [[5.0, 10.0, 15.0, 20.0], [7.0, 14.0, 21.0, 28.0]],
        dtype=torch.float32,
    )
    z_path = _save_checkpoint(tmp_path / "z.pt", {"b.weight": z_only})
    a_path = _save_checkpoint(
        tmp_path / "a.pt",
        {"z.weight": a_second, "a.weight": a_first},
    )

    manifest = build_corpus_manifest(
        [z_path, a_path],
        block_size=4,
        max_files=None,
        max_blocks_per_ckpt=100,
        min_checkpoint_bytes=0,
        corpus_dir=tmp_path,
    )
    corpus = build_corpus_from_manifest(manifest)

    selected_names = [
        entry["name"]
        for file_entry in manifest["files"]
        for entry in file_entry["tensors"]
        if entry["selected"]
    ]
    assert selected_names == ["a.weight", "z.weight", "b.weight"]

    expected = torch.cat(
        [
            _normalized_blocks(a_first, 4),
            _normalized_blocks(a_second, 4),
            _normalized_blocks(z_only, 4),
        ],
        dim=0,
    )
    assert torch.allclose(corpus, expected)


def test_manifest_metadata_matches_actual_corpus_and_caps(tmp_path):
    ckpt = _save_checkpoint(
        tmp_path / "model.pt",
        {"weight": torch.arange(16, dtype=torch.float32).reshape(4, 4)},
    )
    extra = _save_checkpoint(
        tmp_path / "z_extra.pt",
        {"weight": torch.arange(16, 32, dtype=torch.float32).reshape(4, 4)},
    )

    manifest = build_corpus_manifest(
        [ckpt, extra],
        block_size=4,
        max_files=1,
        max_blocks_per_ckpt=3,
        min_checkpoint_bytes=0,
        corpus_dir=tmp_path,
    )
    corpus = build_corpus_from_manifest(manifest)
    legacy_corpus = build_corpus_from_checkpoints(
        [ckpt],
        block_size=4,
        max_blocks_per_ckpt=3,
    )

    file_entry = manifest["files"][0]
    capped_file_entry = manifest["files"][1]
    tensor_entry = file_entry["tensors"][0]
    assert manifest["totals"]["selected_blocks"] == corpus.shape[0] == 3
    assert manifest["selection"]["max_files"] == 1
    assert capped_file_entry["exclusion_reason"] == "max_files_cap"
    assert file_entry["selected_block_count"] == 3
    assert file_entry["cap_reached"] is True
    assert tensor_entry["block_count"] == 4
    assert tensor_entry["used_block_count"] == 3
    assert tensor_entry["corpus_block_start"] == 0
    assert tensor_entry["corpus_block_end"] == 3
    assert torch.allclose(corpus, legacy_corpus)


def test_empty_or_invalid_manifest_fails_closed(tmp_path):
    with pytest.raises(CorpusManifestError, match="empty J-NWC corpus"):
        build_corpus_manifest(
            [tmp_path / "missing.pt"],
            block_size=4,
            max_files=1,
            max_blocks_per_ckpt=10,
            min_checkpoint_bytes=0,
            corpus_dir=tmp_path,
        )


def test_manifest_replay_can_use_relocated_corpus_root(tmp_path):
    source_root = tmp_path / "source"
    relocated_root = tmp_path / "relocated"
    ckpt = _save_checkpoint(
        source_root / "nested" / "renderer.pt",
        {"weight": torch.arange(16, dtype=torch.float32).reshape(4, 4)},
    )
    manifest = build_corpus_manifest(
        [ckpt],
        block_size=4,
        max_files=10,
        max_blocks_per_ckpt=100,
        min_checkpoint_bytes=0,
        corpus_dir=source_root,
    )
    expected = build_corpus_from_manifest(manifest)

    shutil.copytree(source_root, relocated_root)
    ckpt.unlink()

    replayed = build_corpus_from_manifest(manifest, replay_root=relocated_root)
    assert torch.allclose(replayed, expected)


def test_train_cli_preserves_prebuilt_manifest_bytes_with_replay_root(tmp_path):
    source_root = tmp_path / "source"
    relocated_root = tmp_path / "relocated"
    ckpt = _save_checkpoint(
        source_root / "nested" / "renderer.pt",
        {"weight": torch.arange(16, dtype=torch.float32).reshape(4, 4)},
    )
    manifest = build_corpus_manifest(
        [ckpt],
        block_size=4,
        max_files=10,
        max_blocks_per_ckpt=100,
        min_checkpoint_bytes=0,
        corpus_dir=source_root,
    )
    prebuilt_manifest = tmp_path / "prebuilt_manifest.json"
    manifest_text = json.dumps(manifest, sort_keys=False, separators=(",", ":")) + "\n"
    prebuilt_manifest.write_text(manifest_text)

    shutil.copytree(source_root, relocated_root)
    ckpt.unlink()

    output = tmp_path / "codec.pt"
    manifest_out = tmp_path / "used_manifest.json"
    subprocess.run(
        [
            sys.executable,
            "experiments/train_neural_weight_codec.py",
            "--corpus-manifest",
            str(prebuilt_manifest),
            "--corpus-replay-root",
            str(relocated_root),
            "--manifest-out",
            str(manifest_out),
            "--output",
            str(output),
            "--num-steps",
            "1",
            "--batch-size",
            "2",
            "--lr",
            "1e-3",
            "--device",
            "cpu",
            "--block-size",
            "4",
            "--codebook-size",
            "2",
            "--latent-dim",
            "2",
            "--hidden",
            "4",
            "--seed",
            "1234",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert manifest_out.read_bytes() == prebuilt_manifest.read_bytes()
    payload = torch.load(output, map_location="cpu", weights_only=False)
    expected_sha = hashlib.sha256(prebuilt_manifest.read_bytes()).hexdigest()
    assert payload["corpus_manifest_sha256"] == expected_sha
    assert payload["corpus_replay_root"] == str(relocated_root)


def test_manifest_replay_rejects_unsafe_relocated_relative_paths(tmp_path):
    source_root = tmp_path / "source"
    relocated_root = tmp_path / "relocated"
    ckpt = _save_checkpoint(
        source_root / "nested" / "renderer.pt",
        {"weight": torch.arange(16, dtype=torch.float32).reshape(4, 4)},
    )
    manifest = build_corpus_manifest(
        [ckpt],
        block_size=4,
        max_files=10,
        max_blocks_per_ckpt=100,
        min_checkpoint_bytes=0,
        corpus_dir=source_root,
    )
    shutil.copytree(source_root, relocated_root)

    manifest["files"][0]["relative_path"] = "../source/nested/renderer.pt"
    with pytest.raises(CorpusManifestError, match="unsafe relative_path"):
        build_corpus_from_manifest(manifest, replay_root=relocated_root)

    manifest["files"][0]["relative_path"] = str(ckpt)
    with pytest.raises(CorpusManifestError, match="unsafe relative_path"):
        build_corpus_from_manifest(manifest, replay_root=relocated_root)


def test_manifest_replay_rejects_wrong_schema_for_direct_dict(tmp_path):
    ckpt = _save_checkpoint(
        tmp_path / "renderer.pt",
        {"weight": torch.arange(16, dtype=torch.float32).reshape(4, 4)},
    )
    manifest = build_corpus_manifest(
        [ckpt],
        block_size=4,
        max_files=10,
        max_blocks_per_ckpt=100,
        min_checkpoint_bytes=0,
        corpus_dir=tmp_path,
    )
    manifest["schema_version"] = 999

    with pytest.raises(CorpusManifestError, match="unsupported corpus manifest"):
        build_corpus_from_manifest(manifest)


def test_manifest_generation_excludes_paths_outside_declared_corpus_dir(tmp_path):
    corpus_root = tmp_path / "corpus"
    inside = _save_checkpoint(
        corpus_root / "inside.pt",
        {"weight": torch.arange(16, dtype=torch.float32).reshape(4, 4)},
    )
    outside = _save_checkpoint(
        tmp_path / "outside.pt",
        {"weight": torch.arange(16, 32, dtype=torch.float32).reshape(4, 4)},
    )

    manifest = build_corpus_manifest(
        [outside, inside],
        block_size=4,
        max_files=10,
        max_blocks_per_ckpt=100,
        min_checkpoint_bytes=0,
        corpus_dir=corpus_root,
    )

    by_name = {
        Path(file_entry["path"]).name: file_entry for file_entry in manifest["files"]
    }
    assert by_name["inside.pt"]["selected"] is True
    assert by_name["outside.pt"]["selected"] is False
    assert by_name["outside.pt"]["exclusion_reason"] == "outside_corpus_dir"


def test_manifest_generation_excludes_hidden_or_macos_relative_paths(tmp_path):
    visible = _save_checkpoint(
        tmp_path / "visible.pt",
        {"weight": torch.arange(16, dtype=torch.float32).reshape(4, 4)},
    )
    hidden = _save_checkpoint(
        tmp_path / ".cache" / "hidden.pt",
        {"weight": torch.arange(16, 32, dtype=torch.float32).reshape(4, 4)},
    )
    macos = _save_checkpoint(
        tmp_path / "__MACOSX" / "resource.pt",
        {"weight": torch.arange(32, 48, dtype=torch.float32).reshape(4, 4)},
    )

    manifest = build_corpus_manifest(
        [hidden, visible, macos],
        block_size=4,
        max_files=10,
        max_blocks_per_ckpt=100,
        min_checkpoint_bytes=0,
        corpus_dir=tmp_path,
    )

    by_relpath = {
        file_entry["relative_path"]: file_entry for file_entry in manifest["files"]
    }
    assert by_relpath["visible.pt"]["selected"] is True
    assert by_relpath[".cache/hidden.pt"]["selected"] is False
    assert (
        by_relpath[".cache/hidden.pt"]["exclusion_reason"] == "unsafe_relative_path"
    )
    assert by_relpath["__MACOSX/resource.pt"]["selected"] is False
    assert (
        by_relpath["__MACOSX/resource.pt"]["exclusion_reason"]
        == "unsafe_relative_path"
    )

    corpus = build_corpus_from_manifest(manifest)
    expected = _normalized_blocks(
        torch.arange(16, dtype=torch.float32).reshape(4, 4),
        4,
    )
    assert torch.allclose(corpus, expected)
