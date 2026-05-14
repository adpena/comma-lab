# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest
import torch

from tac.substrates.siren.activation_family import (
    ACTIVATION_FAMILY_IDS,
    SIREN_ACTIVATION_FAMILIES,
    activation_family_manifest,
    normalize_activation_family,
)
from tac.substrates.siren.architecture import SirenConfig, SirenSubstrate
from tac.substrates.siren.archive import pack_archive, parse_archive


def _cfg(activation_family: str) -> SirenConfig:
    return SirenConfig(
        hidden_dim=12,
        num_hidden_layers=3,
        first_omega=30.0,
        hidden_omega=1.0,
        num_pairs=2,
        output_height=6,
        output_width=8,
        activation_family=activation_family,
    )


def _meta(cfg: SirenConfig) -> dict[str, object]:
    return {
        "first_omega": cfg.first_omega,
        "hidden_omega": cfg.hidden_omega,
        "coord_dim": cfg.coord_dim,
        "output_dim": cfg.output_dim,
        "activation_family": cfg.activation_family,
        "wire_scale": cfg.wire_scale,
        "bacon_bandwidth_scale": cfg.bacon_bandwidth_scale,
    }


def test_activation_family_registry_names_all_comparison_modes() -> None:
    assert set(ACTIVATION_FAMILY_IDS) == {"siren", "finer", "wire", "bacon"}
    assert set(SIREN_ACTIVATION_FAMILIES) == set(ACTIVATION_FAMILY_IDS)
    manifest = activation_family_manifest()
    assert manifest["siren"]["full_paper_architecture"] is True
    assert manifest["wire"]["full_paper_architecture"] is False


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("siren", "siren"),
        ("naked-siren", "siren"),
        ("FINER-style", "finer"),
        ("wire_style", "wire"),
        ("band-limited", "bacon"),
    ],
)
def test_normalize_activation_family_accepts_aliases(alias: str, expected: str) -> None:
    assert normalize_activation_family(alias) == expected


def test_unknown_activation_family_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown SIREN activation family"):
        SirenConfig(activation_family="made_up")  # type: ignore[arg-type]


@pytest.mark.parametrize("activation_family", ACTIVATION_FAMILY_IDS)
def test_activation_family_forward_shape_and_archive_roundtrip(
    activation_family: str,
) -> None:
    torch.manual_seed(41)
    cfg = _cfg(activation_family)
    model = SirenSubstrate(cfg).eval()
    idx = torch.tensor([0, 1], dtype=torch.long)

    with torch.no_grad():
        rgb_0_a, rgb_1_a = model(idx)

    blob = pack_archive(
        model.runtime_state_dict_for_archive(),
        _meta(cfg),
        num_pairs=cfg.num_pairs,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
    )
    arc = parse_archive(blob)
    assert arc.meta["activation_family"] == activation_family

    rebuilt_cfg = SirenConfig(
        hidden_dim=arc.hidden_dim,
        num_hidden_layers=arc.num_hidden_layers,
        first_omega=float(arc.meta["first_omega"]),
        hidden_omega=float(arc.meta["hidden_omega"]),
        num_pairs=arc.num_pairs,
        output_height=arc.output_height,
        output_width=arc.output_width,
        activation_family=str(arc.meta["activation_family"]),
        wire_scale=float(arc.meta["wire_scale"]),
        bacon_bandwidth_scale=float(arc.meta["bacon_bandwidth_scale"]),
    )
    rebuilt = SirenSubstrate(rebuilt_cfg).eval()
    rebuilt.load_state_dict(arc.decoder_state_dict, strict=False)

    with torch.no_grad():
        rgb_0_b, rgb_1_b = rebuilt(idx)

    assert rgb_0_a.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert rgb_1_a.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert torch.allclose(rgb_0_a, rgb_0_b, atol=5e-2)
    assert torch.allclose(rgb_1_a, rgb_1_b, atol=5e-2)


@pytest.mark.parametrize("activation_family", ("finer", "wire", "bacon"))
def test_non_default_activation_family_changes_forward_function(
    activation_family: str,
) -> None:
    idx = torch.tensor([0, 1], dtype=torch.long)

    torch.manual_seed(7)
    base = SirenSubstrate(_cfg("siren")).eval()
    torch.manual_seed(7)
    variant = SirenSubstrate(_cfg(activation_family)).eval()

    with torch.no_grad():
        base_rgb, _ = base(idx)
        variant_rgb, _ = variant(idx)

    assert not torch.allclose(base_rgb, variant_rgb, atol=1e-5)


def test_pack_archive_rejects_unsupported_activation_family() -> None:
    torch.manual_seed(11)
    cfg = _cfg("siren")
    model = SirenSubstrate(cfg)
    bad_meta = _meta(cfg)
    bad_meta["activation_family"] = "unknown"

    with pytest.raises(ValueError, match="unknown SIREN activation family"):
        pack_archive(
            model.runtime_state_dict_for_archive(),
            bad_meta,
            num_pairs=cfg.num_pairs,
            hidden_dim=cfg.hidden_dim,
            num_hidden_layers=cfg.num_hidden_layers,
            output_height=cfg.output_height,
            output_width=cfg.output_width,
        )


def test_parse_archive_defaults_legacy_meta_to_siren() -> None:
    torch.manual_seed(13)
    cfg = _cfg("siren")
    model = SirenSubstrate(cfg)
    legacy_meta = {
        "first_omega": cfg.first_omega,
        "hidden_omega": cfg.hidden_omega,
        "coord_dim": cfg.coord_dim,
        "output_dim": cfg.output_dim,
    }

    blob = pack_archive(
        model.runtime_state_dict_for_archive(),
        legacy_meta,
        num_pairs=cfg.num_pairs,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
    )
    arc = parse_archive(blob)

    assert arc.meta["activation_family"] == "siren"
    assert arc.meta["wire_scale"] == 1.0
    assert arc.meta["bacon_bandwidth_scale"] == 1.0
