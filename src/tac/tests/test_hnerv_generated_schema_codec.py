from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from tac.hnerv_arch_schema import (
    HNeRVArchConfig,
    generate_hnerv_state_schema,
    schema_fingerprint,
    schema_to_jsonable,
)
from tac.hnerv_generated_schema_codec import (
    HNeRVGeneratedSchemaCodecError,
    decode_generated_schema_blob,
    encode_generated_schema_state_dict,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "materialize_hnerv_generated_schema_codec.py"


def _state_dict(config: HNeRVArchConfig) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(123)
    return {
        name: torch.randn(*shape, generator=gen, dtype=torch.float32) * 0.1
        for name, shape in generate_hnerv_state_schema(config)
    }


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "materialize_hnerv_generated_schema_codec",
        TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generated_schema_codec_is_deterministic_and_fail_closed() -> None:
    config = HNeRVArchConfig(latent_dim=3, base_channels=4, eval_size=(64, 64))
    schema = generate_hnerv_state_schema(config)
    state = _state_dict(config)

    first = encode_generated_schema_state_dict(state, schema=schema, brotli_quality=5)
    second = encode_generated_schema_state_dict(state, schema=schema, brotli_quality=5)
    decoded, decode_manifest = decode_generated_schema_blob(first.blob)

    assert first.blob == second.blob
    assert first.manifest["score_claim"] is False
    assert first.manifest["ready_for_exact_eval_dispatch"] is False
    assert first.manifest["blob_sha256"] == second.manifest["blob_sha256"]
    assert decode_manifest["schema_fingerprint"] == schema_fingerprint(schema)
    assert set(decoded) == set(state)
    assert tuple(decoded["stem.weight"].shape) == tuple(state["stem.weight"].shape)


def test_generated_schema_codec_rejects_shape_mismatch() -> None:
    config = HNeRVArchConfig(latent_dim=3, base_channels=4, eval_size=(64, 64))
    schema = generate_hnerv_state_schema(config)
    state = _state_dict(config)
    state["stem.weight"] = state["stem.weight"][:2]

    with pytest.raises(HNeRVGeneratedSchemaCodecError, match="shape"):
        encode_generated_schema_state_dict(state, schema=schema)


def test_generated_schema_codec_rejects_tampered_payload() -> None:
    config = HNeRVArchConfig(latent_dim=3, base_channels=4, eval_size=(64, 64))
    encoded = encode_generated_schema_state_dict(
        _state_dict(config),
        schema=generate_hnerv_state_schema(config),
        brotli_quality=5,
    )
    tampered = bytearray(encoded.blob)
    tampered[-1] ^= 0x01

    with pytest.raises(HNeRVGeneratedSchemaCodecError, match=r"sha256|truncated|decompress"):
        decode_generated_schema_blob(bytes(tampered))


def test_materialize_generated_schema_codec_cli(tmp_path: Path) -> None:
    tool = _load_tool()
    config = HNeRVArchConfig(latent_dim=3, base_channels=4, eval_size=(64, 64))
    schema = generate_hnerv_state_schema(config)
    state_path = tmp_path / "state.pt"
    schema_path = tmp_path / "schema.json"
    blob_path = tmp_path / "codec.hngs"
    manifest_path = tmp_path / "manifest.json"
    torch.save(_state_dict(config), state_path)
    schema_path.write_text(
        json.dumps({"state_schema": schema_to_jsonable(schema)}),
        encoding="utf-8",
    )

    assert tool.main([
        "--state-dict",
        str(state_path),
        "--schema-json",
        str(schema_path),
        "--output-blob",
        str(blob_path),
        "--output-manifest",
        str(manifest_path),
        "--brotli-quality",
        "5",
    ]) == 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert blob_path.is_file()
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["blob_bytes"] == blob_path.stat().st_size
    assert manifest["decode_roundtrip_schema_fingerprint"] == schema_fingerprint(schema)
