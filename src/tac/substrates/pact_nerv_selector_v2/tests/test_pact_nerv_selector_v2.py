# SPDX-License-Identifier: MIT
"""Catalog #91 ENCODE_INFLATE_ROUNDTRIP + #139 no_op_proof + L0 SCAFFOLD contract."""

from __future__ import annotations

import torch

from tac.substrates.pact_nerv_selector_v2.architecture import (
    FEC6_FIXED_K16_MODE_IDS,
    ArithmeticSelectorCoder,
    PactNervSelectorV2Config,
    PactNervSelectorV2Substrate,
)
from tac.substrates.pact_nerv_selector_v2.archive import (
    PSV2_HEADER_SIZE,
    PSV2_MAGIC,
    PSV2_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)


def _smoke_cfg() -> PactNervSelectorV2Config:
    return PactNervSelectorV2Config(
        latent_dim=8,
        embed_dim=24,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(20, 16, 12),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        num_pairs=3,
        output_height=24,
        output_width=32,
        selector_palette_size=16,
    )


def _smoke_meta(cfg: PactNervSelectorV2Config) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
    }


def test_module_import_resolves_canonical_symbols() -> None:
    from tac.substrates import pact_nerv_selector_v2 as m
    for name in (
        "PactNervSelectorV2Config",
        "PactNervSelectorV2Substrate",
        "ArithmeticSelectorCoder",
        "pack_archive",
        "parse_archive",
        "PactNervSelectorV2ScoreAwareLoss",
        "PactNervSelectorV2Archive",
    ):
        assert hasattr(m, name), f"missing canonical symbol: {name}"


def test_substrate_forward_produces_unit_interval_rgb() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervSelectorV2Substrate(cfg).eval()
    idx = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1 = model(idx)
    assert rgb_0.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert rgb_1.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert float(rgb_0.min()) >= 0.0
    assert float(rgb_0.max()) <= 1.0


def test_arithmetic_coder_encode_returns_bounded_bytes() -> None:
    """Witten 1987 arithmetic coder: |bitstream| <= H(X) + 2 bits per stream."""
    coder = ArithmeticSelectorCoder(palette_size=16)
    syms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    out = coder.encode(syms)
    assert isinstance(out, bytes)
    # Uniform 16-symbol stream of 16 syms = 16*4 = 64 bits = 8 bytes ideal;
    # arithmetic coder within 2 bits => <= 9 bytes worst case + flush bits.
    assert len(out) <= 16


def test_arithmetic_coder_roundtrip_selector_stream() -> None:
    coder = ArithmeticSelectorCoder(palette_size=16)
    syms = [0, 1, 2, 3, 15, 0, 7, 2, 2, 14, 0, 4, 6, 9, 10, 11]
    payload = coder.encode(syms)
    assert coder.decode(payload, symbol_count=len(syms)) == syms


def test_arithmetic_coder_rejects_invalid_symbols() -> None:
    coder = ArithmeticSelectorCoder(palette_size=16)
    try:
        coder.encode([16])
    except ValueError as exc:
        assert "out of palette" in str(exc)
    else:
        raise AssertionError("expected ValueError for symbol >= palette")


def test_arithmetic_coder_encoded_bit_length_matches_entropy() -> None:
    """Entropy estimate matches uniform-distribution H(X) = log2(N)."""
    coder = ArithmeticSelectorCoder(palette_size=16)
    bits = coder.encoded_bit_length([0, 1, 2, 3])
    assert bits >= 16  # 4 symbols * 4 bits/sym ideal; uniform-init
    assert bits <= 20


def test_archive_pack_then_parse_roundtrip() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervSelectorV2Substrate(cfg)
    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k not in ("latents", "selectors")}
    latents = sd["latents"].clone()
    selector_bytes = b"\x00\x01\x02\x03\x04"
    blob = pack_archive(
        decoder_sd, latents, selector_bytes, _smoke_meta(cfg), palette_size=16
    )
    arc = parse_archive(blob)
    assert arc.schema_version == PSV2_SCHEMA_VERSION
    assert blob[:4] == PSV2_MAGIC
    assert arc.palette_size == 16
    assert arc.selector_bytes == selector_bytes
    assert set(arc.decoder_state_dict.keys()) == set(decoder_sd.keys())


def test_inflate_consumes_selector_stream_and_changes_frame0(tmp_path) -> None:
    from tac.substrates.pact_nerv_selector_v2.archive_candidate import (
        pack_archive_from_exported_state_dict,
    )
    from tac.substrates.pact_nerv_selector_v2.inflate import inflate_one_video

    cfg = _smoke_cfg()
    torch.manual_seed(4)
    model = PactNervSelectorV2Substrate(cfg).eval()
    exported = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in model.state_dict().items()
        if name != "selectors"
    }
    none_blob = pack_archive_from_exported_state_dict(
        exported_state_dict=exported,
        cfg=cfg,
        selectors=torch.zeros(cfg.num_pairs, dtype=torch.long).numpy(),
    )
    selector = torch.zeros(cfg.num_pairs, dtype=torch.long)
    selector[0] = FEC6_FIXED_K16_MODE_IDS.index("frame0_luma_bias_+1")
    selected_blob = pack_archive_from_exported_state_dict(
        exported_state_dict=exported,
        cfg=cfg,
        selectors=selector.numpy(),
    )
    none_out = tmp_path / "none"
    selected_out = tmp_path / "selected"
    inflate_one_video(none_blob, none_out, device="cpu")
    inflate_one_video(selected_blob, selected_out, device="cpu")
    assert (none_out / "0.png").read_bytes() != (selected_out / "0.png").read_bytes()
    assert (none_out / "1.png").read_bytes() == (selected_out / "1.png").read_bytes()


def test_archive_header_size_invariant_is_26_bytes() -> None:
    assert PSV2_HEADER_SIZE == 26


def test_byte_mutation_changes_archive_no_op_proof() -> None:
    """Catalog #139: mutating selector bytes must change archive bytes."""
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = PactNervSelectorV2Substrate(cfg).eval()
    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k not in ("latents", "selectors")}
    latents = sd["latents"].clone()
    blob_a = pack_archive(
        decoder_sd, latents, b"\x00\x01\x02", _smoke_meta(cfg), palette_size=16
    )
    blob_b = pack_archive(
        decoder_sd, latents, b"\xff\x01\x02", _smoke_meta(cfg), palette_size=16
    )
    assert blob_a != blob_b, "no_op_proof: mutating selector bytes must change archive"


def test_trainer_full_main_implemented_and_cuda_gated(tmp_path) -> None:
    """PACT-NERV-FULL-MAIN-WAVE 2026-05-27: _full_main IMPLEMENTED + CUDA-gated."""
    import importlib
    import inspect

    import pytest

    trainer = importlib.import_module("experiments.train_substrate_pact_nerv_selector_v2")
    src = inspect.getsource(trainer._full_main)
    assert "raise NotImplementedError" not in src
    assert "run_pact_nerv_score_aware_training" in src
    args = trainer._build_parser().parse_args(
        ["--output-dir", str(tmp_path / "out"), "--device", "cpu"]
    )
    with pytest.raises(SystemExit):
        trainer._full_main(args)


def test_trainer_routes_through_canonical_scorer_loss_helper() -> None:
    """Catalog #164: score-aware path routes through canonical helper."""
    import inspect

    from tac.substrates.pact_nerv_selector_v2 import score_aware_loss as sal
    src = inspect.getsource(sal)
    assert "score_pair_components_dispatch" in src
    assert "tac.substrates.score_aware_common" in src


def test_trainer_patches_differentiable_eval_roundtrip_before_scorer() -> None:
    """Catalog #6: trainer MUST patch yuv6 BEFORE scorer load."""
    import inspect

    import experiments.train_substrate_pact_nerv_selector_v2 as trainer_module
    src = inspect.getsource(trainer_module._smoke_main)
    assert "patch_upstream_yuv6_globally" in src


def test_recipe_research_only_and_dispatch_disabled() -> None:
    """Catalog #240: recipe opts out of dispatch at L0 SCAFFOLD."""
    from pathlib import Path

    import yaml  # type: ignore[import-untyped]
    recipe_path = (
        Path(__file__).resolve().parents[5]
        / ".omx/operator_authorize_recipes/substrate_pact_nerv_selector_v2_modal_t4_dispatch.yaml"
    )
    assert recipe_path.exists(), f"recipe missing: {recipe_path}"
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    assert recipe["dispatch_enabled"] is False
    assert recipe["research_only"] is True


def test_driver_carries_canonical_nvml_block() -> None:
    """Catalog #244: driver carries the 3-export NVML block."""
    from pathlib import Path
    driver = (
        Path(__file__).resolve().parents[5]
        / "scripts/remote_lane_substrate_pact_nerv_selector_v2.sh"
    )
    assert driver.exists()
    txt = driver.read_text(encoding="utf-8")
    assert "DALI_DISABLE_NVML" in txt
    assert "CUBLAS_WORKSPACE_CONFIG" in txt
    assert "PYTORCH_CUDA_ALLOC_CONF" in txt


def test_inflate_py_loc_under_200_per_hnerv_parity_l4() -> None:
    """HNeRV parity L4: inflate runtime <= 200 LOC."""
    from pathlib import Path
    inflate_path = Path(__file__).resolve().parents[1] / "inflate.py"
    assert inflate_path.exists()
    loc = len(inflate_path.read_text(encoding="utf-8").splitlines())
    assert loc <= 200, f"inflate.py {loc} LOC exceeds HNeRV parity L4 ceiling"
