# SPDX-License-Identifier: MIT
"""Catalog #91 + #139 + L0 SCAFFOLD contract for V4 (RLE selector)."""

from __future__ import annotations

import torch

from tac.substrates.pact_nerv_selector_v4.architecture import (
    PactNervSelectorV4Config,
    PactNervSelectorV4Substrate,
    RunLengthSelectorCoder,
)
from tac.substrates.pact_nerv_selector_v4.archive import (
    PSV4_HEADER_SIZE,
    PSV4_MAGIC,
    PSV4_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)


def _smoke_cfg() -> PactNervSelectorV4Config:
    return PactNervSelectorV4Config(
        latent_dim=8, embed_dim=24, initial_grid_h=3, initial_grid_w=4,
        decoder_channels=(20, 16, 12), sin_frequency=30.0,
        num_upsample_blocks=3, num_pairs=3, output_height=24, output_width=32,
        selector_palette_size=16,
    )


def _smoke_meta(cfg: PactNervSelectorV4Config) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim, "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "output_height": cfg.output_height, "output_width": cfg.output_width,
    }


def test_module_import_resolves_canonical_symbols() -> None:
    from tac.substrates import pact_nerv_selector_v4 as m
    for name in (
        "PactNervSelectorV4Config", "PactNervSelectorV4Substrate",
        "RunLengthSelectorCoder", "pack_archive", "parse_archive",
        "PactNervSelectorV4ScoreAwareLoss", "PactNervSelectorV4Archive",
    ):
        assert hasattr(m, name)


def test_substrate_forward_produces_unit_interval_rgb() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervSelectorV4Substrate(cfg).eval()
    idx = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1 = model(idx)
    assert rgb_0.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert float(rgb_0.min()) >= 0.0
    assert float(rgb_0.max()) <= 1.0


def test_rle_coder_encode_decode_roundtrip() -> None:
    """RLE encode/decode is invertible (Robinson-Cherry 1967 canonical)."""
    coder = RunLengthSelectorCoder(palette_size=16)
    syms = [0, 0, 0, 5, 5, 1, 0, 0, 0, 0, 0]
    encoded = coder.encode(syms)
    decoded = coder.decode(encoded)
    assert decoded == syms


def test_rle_coder_empty_stream() -> None:
    coder = RunLengthSelectorCoder(palette_size=16)
    assert coder.encode([]) == b""
    assert coder.decode(b"") == []


def test_rle_coder_long_run_uses_varint() -> None:
    """Long runs (>=128) use 2-byte varint encoding."""
    coder = RunLengthSelectorCoder(palette_size=16)
    syms = [3] * 200
    encoded = coder.encode(syms)
    # 1 byte value + 2 bytes varint(200) = 3 bytes total
    assert len(encoded) == 3
    decoded = coder.decode(encoded)
    assert decoded == syms


def test_rle_coder_rejects_invalid_symbols() -> None:
    coder = RunLengthSelectorCoder(palette_size=16)
    try:
        coder.encode([16])
    except ValueError as exc:
        assert "out of palette" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_rle_coder_rejects_palette_too_large() -> None:
    try:
        RunLengthSelectorCoder(palette_size=257)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for palette > 256")


def test_archive_pack_then_parse_roundtrip() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervSelectorV4Substrate(cfg)
    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k not in ("latents", "selectors")}
    latents = sd["latents"].clone()
    selector_bytes = b"\x00\x05\x05"
    blob = pack_archive(decoder_sd, latents, selector_bytes, _smoke_meta(cfg), palette_size=16)
    arc = parse_archive(blob)
    assert arc.schema_version == PSV4_SCHEMA_VERSION
    assert blob[:4] == PSV4_MAGIC
    assert arc.palette_size == 16
    assert arc.selector_bytes == selector_bytes


def test_archive_header_size_invariant_is_26_bytes() -> None:
    assert PSV4_HEADER_SIZE == 26


def test_byte_mutation_changes_archive_no_op_proof() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = PactNervSelectorV4Substrate(cfg).eval()
    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k not in ("latents", "selectors")}
    latents = sd["latents"].clone()
    blob_a = pack_archive(decoder_sd, latents, b"\x00\x05", _smoke_meta(cfg), palette_size=16)
    blob_b = pack_archive(decoder_sd, latents, b"\xff\x05", _smoke_meta(cfg), palette_size=16)
    assert blob_a != blob_b


def test_trainer_full_main_implemented_and_cuda_gated(tmp_path) -> None:
    """PACT-NERV-FULL-MAIN-CLUSTER-2 2026-05-27: _full_main IMPLEMENTED + CUDA-gated."""
    import importlib
    import inspect

    import pytest

    trainer = importlib.import_module("experiments.train_substrate_pact_nerv_selector_v4")
    src = inspect.getsource(trainer._full_main)
    assert "raise NotImplementedError" not in src
    assert "run_pact_nerv_score_aware_training" in src
    args = trainer._build_parser().parse_args(
        ["--output-dir", str(tmp_path / "out"), "--device", "cpu"]
    )
    with pytest.raises(SystemExit):
        trainer._full_main(args)


def test_trainer_routes_through_canonical_scorer_loss_helper() -> None:
    import inspect

    from tac.substrates.pact_nerv_selector_v4 import score_aware_loss as sal
    src = inspect.getsource(sal)
    assert "score_pair_components_dispatch" in src
    assert "tac.substrates.score_aware_common" in src


def test_trainer_patches_differentiable_eval_roundtrip_before_scorer() -> None:
    import inspect

    import experiments.train_substrate_pact_nerv_selector_v4 as trainer_module
    src = inspect.getsource(trainer_module._smoke_main)
    assert "patch_upstream_yuv6_globally" in src


def test_recipe_research_only_and_dispatch_disabled() -> None:
    from pathlib import Path

    import yaml  # type: ignore[import-untyped]
    recipe = yaml.safe_load(
        (Path(__file__).resolve().parents[5]
         / ".omx/operator_authorize_recipes/substrate_pact_nerv_selector_v4_modal_t4_dispatch.yaml"
        ).read_text(encoding="utf-8")
    )
    assert recipe["dispatch_enabled"] is False
    assert recipe["research_only"] is True


def test_driver_carries_canonical_nvml_block() -> None:
    from pathlib import Path
    txt = (
        Path(__file__).resolve().parents[5]
        / "scripts/remote_lane_substrate_pact_nerv_selector_v4.sh"
    ).read_text(encoding="utf-8")
    assert "DALI_DISABLE_NVML" in txt
    assert "CUBLAS_WORKSPACE_CONFIG" in txt
    assert "PYTORCH_CUDA_ALLOC_CONF" in txt


def test_inflate_py_loc_under_200() -> None:
    from pathlib import Path
    loc = len(
        (Path(__file__).resolve().parents[1] / "inflate.py")
        .read_text(encoding="utf-8").splitlines()
    )
    assert loc <= 200
