"""OWV3 sensitivity-weighted renderer archive tests."""
from __future__ import annotations

import json
import struct
import sys
from importlib import util as importlib_util
from pathlib import Path

import pytest
import torch

from tac.owv3_sensitivity_weighted import (
    OWV3_ARCHIVE_MAGIC,
    OWV3_ARCHIVE_VERSION,
    OWV3ArchiveError,
    decode_owv3_archive,
    enforce_owv3_byte_budget,
    encode_owv3_archive,
    inspect_owv3_archive,
    is_owv3_archive,
)
from tac.sensitivity_map import conv_weight_shapes


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _build_renderer():
    from tac.renderer import AsymmetricPairGenerator

    torch.manual_seed(1234)
    return AsymmetricPairGenerator(
        num_classes=5,
        embed_dim=6,
        base_ch=8,
        mid_ch=12,
        motion_hidden=8,
        depth=1,
        pose_dim=0,
        use_dilation=False,
    ).eval()


def _synthetic_sensitivity(model) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, channels in conv_weight_shapes(model).items():
        value = torch.full((channels,), 1e-6)
        if channels > 1:
            value[0] = 1e-2
        out[key] = value
    return out


def _all_protected_sensitivity(model) -> dict[str, torch.Tensor]:
    return {
        key: torch.full((channels,), 1e-2)
        for key, channels in conv_weight_shapes(model).items()
    }


def _parse_header(blob: bytes) -> dict:
    assert blob[:4] == OWV3_ARCHIVE_MAGIC
    offset = 4
    (header_len,) = struct.unpack("<I", blob[offset:offset + 4])
    offset += 4
    return json.loads(blob[offset:offset + header_len].decode("utf-8"))


def _load_inflate_renderer_module():
    target = _REPO_ROOT / "submissions" / "robust_current" / "inflate_renderer.py"
    assert target.exists(), f"missing inflate_renderer at {target}"
    spec = importlib_util.spec_from_file_location(
        "_owv3_test_inflate_renderer",
        str(target),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib_util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_owv3_magic_constants_and_registry() -> None:
    assert OWV3_ARCHIVE_MAGIC == b"OWV3"
    assert OWV3_ARCHIVE_VERSION == 1
    assert is_owv3_archive(b"OWV3\x00\x00")
    assert not is_owv3_archive(b"OWV2\x00\x00")

    from tac.codec_magic_registry import find_by_magic, sniff_codec

    entry = find_by_magic(b"OWV3")
    assert entry is not None
    assert entry.name == "Lane Ω-W-V3"
    assert "owv3_sensitivity_weighted" in entry.decode_module
    assert sniff_codec(b"OWV3payload") == entry


def test_owv3_mixed_channel_archive_round_trip_and_forward() -> None:
    model = _build_renderer()
    sensitivities = _synthetic_sensitivity(model)

    blob = encode_owv3_archive(
        model=model,
        sensitivities=sensitivities,
        bit_budget_ratio=0.7,
    )
    assert is_owv3_archive(blob)

    header = _parse_header(blob)
    assert header["byte_plan"]["fallback_action"] == "keep_asym"
    assert header["byte_plan"]["promotion_eligible"] is True
    mixed_layers = [l for l in header["layers"] if l["kind"] == "owv3_conv"]
    assert mixed_layers, "expected at least one sensitivity-weighted Conv2d"
    assert any(
        l["quant_indices"] and l["protected_indices"] for l in mixed_layers
    ), "expected a layer with both OWV2 and protected channels"
    assert all(l.get("protected_codec") == "asym" for l in mixed_layers if l["protected_indices"])
    assert header["byte_plan"]["action_counts"]["fp16_protect_channels"] == 0

    decoded = decode_owv3_archive(data=blob, device="cpu")
    assert not decoded.training
    assert set(decoded.state_dict()) == set(model.state_dict())
    for name, original in model.state_dict().items():
        restored = decoded.state_dict()[name]
        assert restored.shape == original.shape, f"{name}: shape drift"

    modules = dict(model.named_modules())
    decoded_modules = dict(decoded.named_modules())
    first_mixed = next(l for l in mixed_layers if l["protected_indices"])
    protected = torch.tensor(first_mixed["protected_indices"], dtype=torch.long)
    original_w = modules[first_mixed["name"]].weight.detach().cpu()[protected]
    restored_w = decoded_modules[first_mixed["name"]].weight.detach().cpu()[protected]
    max_abs = float(original_w.abs().max().item())
    max_err = float((original_w.float() - restored_w.float()).abs().max().item())
    assert max_err <= max(2e-3, max_abs * 0.02)

    mask_t = torch.randint(0, 5, (1, 16, 16), dtype=torch.long)
    mask_t1 = torch.randint(0, 5, (1, 16, 16), dtype=torch.long)
    with torch.no_grad():
        out = decoded(mask_t, mask_t1)
    assert out.shape == (1, 2, 16, 16, 3)


def test_owv3_default_protected_channels_are_asym_not_fp16() -> None:
    model = _build_renderer()
    blob = encode_owv3_archive(
        model=model,
        sensitivities=_synthetic_sensitivity(model),
        bit_budget_ratio=0.7,
    )
    inspection = inspect_owv3_archive(blob)
    header = inspection["header"]
    protected_layers = [
        l for l in header["layers"]
        if l["kind"] == "owv3_conv" and l["protected_indices"]
    ]
    assert protected_layers
    assert all(l["protected_codec"] == "asym" for l in protected_layers)
    assert header["byte_plan"]["action_counts"]["fp16_protect_channels"] == 0
    assert header["byte_plan"]["action_bytes"]["owv2_low_bit"] > 0
    assert header["byte_plan"]["action_bytes"]["asym"] > 0
    assert header["byte_plan"]["action_bytes"]["bias"] > 0
    assert header["byte_plan"]["fallback_reason_counts"]["convt_keep_asym_fallback"] > 0
    assert header["byte_plan"]["fallback_reason_bytes"]["convt_keep_asym_fallback"] > 0
    assert header["byte_plan"]["promotion_eligible"] is True


def test_owv3_all_protected_default_keeps_asym_not_fp16() -> None:
    model = _build_renderer()
    blob = encode_owv3_archive(
        model=model,
        sensitivities=_all_protected_sensitivity(model),
        bit_budget_ratio=0.7,
    )
    header = inspect_owv3_archive(blob)["header"]
    all_protected = [
        l for l in header["layers"]
        if l.get("fallback_reason") == "all_channels_protected"
    ]
    assert all_protected
    assert all(l["kind"] == "asym_conv" for l in all_protected)
    assert not [
        l for l in header["layers"]
        if l["kind"] == "fp16_conv" and l.get("fallback_reason") == "all_channels_protected"
    ]
    assert header["byte_plan"]["fallback_action"] == "keep_asym"
    assert header["byte_plan"]["fallback_reason_counts"]["all_channels_protected"] > 0
    assert header["byte_plan"]["fallback_reason_bytes"]["all_channels_protected"] > 0
    assert header["byte_plan"]["action_bytes"]["keep_asym"] > 0
    assert header["byte_plan"]["promotion_eligible"] is True


def test_owv3_diagnostic_fp16_fallback_is_explicit_smoke_only() -> None:
    model = _build_renderer()
    blob = encode_owv3_archive(
        model=model,
        sensitivities=_all_protected_sensitivity(model),
        bit_budget_ratio=0.7,
        fallback_action="diagnostic_fp16",
    )
    header = inspect_owv3_archive(blob)["header"]
    all_protected_fp16 = [
        l for l in header["layers"]
        if l["kind"] == "fp16_conv" and l.get("fallback_reason") == "all_channels_protected"
    ]
    assert all_protected_fp16
    assert header["byte_plan"]["fallback_action"] == "diagnostic_fp16"
    assert header["byte_plan"]["promotion_eligible"] is False
    assert header["byte_plan"]["action_counts"]["diagnostic_fp16_layers"] > 0
    assert header["byte_plan"]["action_bytes"]["diagnostic_fp16"] > 0
    assert header["byte_plan"]["fallback_reason_bytes"]["all_channels_protected"] > 0


def test_owv3_byte_budget_rejects_unjustified_size_regression() -> None:
    with pytest.raises(OWV3ArchiveError, match="exceeds"):
        enforce_owv3_byte_budget(
            candidate_bytes=1100,
            comparator_bytes=1000,
            candidate_label="test OWV3",
            comparator_label="test frontier",
        )
    report = enforce_owv3_byte_budget(
        candidate_bytes=1100,
        comparator_bytes=1000,
        allow_size_regression=True,
    )
    assert report["accepted"] is True
    justified = enforce_owv3_byte_budget(
        candidate_bytes=1100,
        comparator_bytes=1000,
        distortion_justification={"contest_auth_eval_json": "exact-cuda.json"},
    )
    assert justified["accepted"] is True


def test_owv3_requires_sensitivity_artifact() -> None:
    model = _build_renderer()
    with pytest.raises(OWV3ArchiveError, match="sensitiv"):
        encode_owv3_archive(model=model, sensitivities={})


def test_owv3_rejects_invalid_sensitivity_values() -> None:
    model = _build_renderer()
    sensitivities = _synthetic_sensitivity(model)
    first_key = next(iter(sensitivities))
    sensitivities[first_key][0] = float("nan")
    with pytest.raises(OWV3ArchiveError, match="NaN/Inf"):
        encode_owv3_archive(model=model, sensitivities=sensitivities)

    sensitivities = _synthetic_sensitivity(model)
    sensitivities[first_key][0] = -1.0
    with pytest.raises(OWV3ArchiveError, match="non-negative"):
        encode_owv3_archive(model=model, sensitivities=sensitivities)


def test_owv3_decode_rejects_bad_magic_version_and_trailing_bytes() -> None:
    with pytest.raises(OWV3ArchiveError, match="bad/missing magic"):
        decode_owv3_archive(data=b"FAKE\x00\x00\x00\x00", device="cpu")

    fake_header = json.dumps(
        {
            "version": 9999,
            "format": "owv3_sensitivity_weighted_renderer_archive_v1",
            "arch": {},
            "layers": [],
            "scalar_params": {},
            "body_len": 0,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    blob = (
        OWV3_ARCHIVE_MAGIC
        + struct.pack("<I", len(fake_header))
        + fake_header
        + struct.pack("<I", 0)
    )
    with pytest.raises(OWV3ArchiveError, match="unsupported version"):
        decode_owv3_archive(data=blob, device="cpu")

    good_header = json.dumps(
        {
            "version": OWV3_ARCHIVE_VERSION,
            "format": "owv3_sensitivity_weighted_renderer_archive_v1",
            "arch": {},
            "layers": [],
            "scalar_params": {},
            "body_len": 0,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    trailing = (
        OWV3_ARCHIVE_MAGIC
        + struct.pack("<I", len(good_header))
        + good_header
        + struct.pack("<I", 0)
        + b"x"
    )
    with pytest.raises(OWV3ArchiveError, match="declared body_len"):
        decode_owv3_archive(data=trailing, device="cpu")


def test_owv3_inflate_renderer_dispatch(tmp_path: Path) -> None:
    model = _build_renderer()
    blob = encode_owv3_archive(
        model=model,
        sensitivities=_synthetic_sensitivity(model),
        bit_budget_ratio=0.7,
    )
    renderer_path = tmp_path / "renderer.bin"
    renderer_path.write_bytes(blob)

    inflate_mod = _load_inflate_renderer_module()
    loaded = inflate_mod._load_renderer(str(renderer_path), device="cpu")
    assert not loaded.training
    assert set(loaded.state_dict()) == set(model.state_dict())


def test_nerv_mask_resolver_finds_nrv_sibling(tmp_path: Path) -> None:
    nrv = tmp_path / "masks.nrv"
    nrv.write_bytes(b"NRV1unit")

    inflate_mod = _load_inflate_renderer_module()
    assert inflate_mod._resolve_mask_path(tmp_path, "masks.mkv") == nrv
