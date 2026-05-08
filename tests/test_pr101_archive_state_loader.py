from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_archive_state_loader import (  # noqa: E402
    Pr101ArchiveStateLoaderError,
    load_pr101_archive_state,
)
from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    decode_decoder_compact,
    encode_decoder_compact,
)


def _write_stored_zip(path: Path, member: str, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(member)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, data, compress_type=zipfile.ZIP_STORED)


def _zero_state_dict() -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros(shape, dtype=torch.float32)
        for name, shape in FIXED_STATE_SCHEMA
    }


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_load_pr101_archive_state_decodes_state_and_metadata(tmp_path: Path) -> None:
    decoder_blob = encode_decoder_compact(_zero_state_dict(), brotli_quality=0)
    latent_blob = b"latent"
    sidecar_blob = b"sidecar"
    archive = tmp_path / "archive.zip"
    _write_stored_zip(archive, "x", decoder_blob + latent_blob + sidecar_blob)

    loaded = load_pr101_archive_state(
        archive,
        decoder_blob_len=len(decoder_blob),
        latent_blob_len=len(latent_blob),
    )

    expected_state = decode_decoder_compact(decoder_blob)
    assert loaded.metadata["kind"] == "pr101_archive_decoder_blob"
    assert loaded.metadata["inner_member_name"] == "x"
    assert loaded.metadata["decoder_blob_bytes"] == len(decoder_blob)
    assert loaded.metadata["latent_blob_bytes"] == len(latent_blob)
    assert loaded.metadata["sidecar_blob_bytes"] == len(sidecar_blob)
    assert set(loaded.state_dict) == {name for name, _shape in FIXED_STATE_SCHEMA}
    for name, tensor in loaded.state_dict.items():
        assert torch.equal(tensor, expected_state[name])


def test_load_pr101_archive_state_fails_closed_on_wrong_member(tmp_path: Path) -> None:
    archive = tmp_path / "bad_member.zip"
    _write_stored_zip(archive, "not-x", b"anything")

    with pytest.raises(Pr101ArchiveStateLoaderError, match=r"expected \['x'\]"):
        load_pr101_archive_state(archive)


@pytest.mark.parametrize("member", [r"..\\x", "bad\x1fname"])
def test_load_pr101_archive_state_fails_closed_on_unsafe_member_name(
    tmp_path: Path,
    member: str,
) -> None:
    archive = tmp_path / "unsafe_member.zip"
    _write_stored_zip(archive, member, b"anything")

    with pytest.raises(Pr101ArchiveStateLoaderError, match="unsafe archive member name"):
        load_pr101_archive_state(archive, expected_member_name=member)


def test_load_pr101_archive_state_fails_closed_on_short_inner_member(tmp_path: Path) -> None:
    archive = tmp_path / "short.zip"
    _write_stored_zip(archive, "x", b"too-short")

    with pytest.raises(Pr101ArchiveStateLoaderError, match="required decoder\\+latent minimum"):
        load_pr101_archive_state(archive)


def test_load_pr101_archive_state_fails_closed_on_bad_decoder_blob(tmp_path: Path) -> None:
    archive = tmp_path / "bad_decoder.zip"
    _write_stored_zip(archive, "x", b"\x00\x01\x02\x03latent")

    with pytest.raises(Pr101ArchiveStateLoaderError, match="decoder_blob cannot be decoded"):
        load_pr101_archive_state(archive, decoder_blob_len=4, latent_blob_len=6)


def test_actual_public_pr101_archive_strict_loads_into_a1_decoder_when_present() -> None:
    archive = (
        REPO_ROOT
        / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip"
    )
    if not archive.is_file():
        pytest.skip("public PR101 archive fixture is not present in this checkout")

    train_mod = _load_module(
        REPO_ROOT / "experiments/train_score_gradient_pr101_finetune.py",
        "train_score_gradient_pr101_finetune_for_loader_test",
    )
    decoder = train_mod.HNeRVDecoder(latent_dim=28, base_channels=36, eval_size=(384, 512))
    metadata = train_mod.load_pr101_substrate(archive, decoder, smoke=False)

    assert metadata["archive_size_bytes"] == archive.stat().st_size
    assert metadata["decoder_blob_bytes"] == 162_164
    assert metadata["latent_blob_bytes"] == 15_387
    assert metadata["sidecar_blob_bytes"] == 607
    assert metadata["load_into_decoder"] == {
        "strict": True,
        "missing_keys": [],
        "unexpected_keys": [],
    }


def test_experiment_cli_materializes_state_dict_and_metadata_when_fixture_present(
    tmp_path: Path,
) -> None:
    archive = (
        REPO_ROOT
        / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip"
    )
    if not archive.is_file():
        pytest.skip("public PR101 archive fixture is not present in this checkout")

    cli_mod = _load_module(
        REPO_ROOT / "experiments/load_pr101_archive_to_state_dict.py",
        "load_pr101_archive_to_state_dict_for_test",
    )
    state_out = tmp_path / "pr101_decoder_state_dict.pt"
    metadata_out = tmp_path / "pr101_decoder_state_dict.metadata.json"

    assert cli_mod.main([
        "--archive",
        str(archive),
        "--output-state-dict",
        str(state_out),
        "--metadata-output",
        str(metadata_out),
    ]) == 0

    state = torch.load(state_out, map_location="cpu", weights_only=True)
    metadata = json.loads(metadata_out.read_text(encoding="utf-8"))
    assert set(state) == {name for name, _shape in FIXED_STATE_SCHEMA}
    assert metadata["archive_size_bytes"] == archive.stat().st_size
    assert metadata["state_dict_tensor_count"] == len(FIXED_STATE_SCHEMA)
