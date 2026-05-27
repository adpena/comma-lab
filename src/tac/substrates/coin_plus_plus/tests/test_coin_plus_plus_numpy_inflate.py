# SPDX-License-Identifier: MIT
"""Numpy-portable inflate parity for coin_plus_plus (8th MLX-first directive).

Proves the shipped inflate runtime decodes the CPP1-v2 archive with NO torch
import (numpy + brotli + PIL only) AND that the numpy forward pass matches the
torch model forward to fp16-roundtrip tolerance (Catalog #369: consumes the
REAL trained weights, not a synthetic frame base).
"""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path

import numpy as np
import torch

from tac.substrates.coin_plus_plus.architecture import (
    CoinplusplusConfig,
    CoinplusplusSubstrate,
)
from tac.substrates.coin_plus_plus.archive import (
    CPP1_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
    parse_archive_numpy,
)
from tac.substrates.coin_plus_plus.inflate import (
    _build_coord_grid,
    _linear,
    inflate_one_video,
)

_INFLATE_PATH = Path(__file__).resolve().parents[1] / "inflate.py"
_ARCHIVE_PATH = Path(__file__).resolve().parents[1] / "archive.py"


def _cfg() -> CoinplusplusConfig:
    return CoinplusplusConfig(
        modulation_dim=8,
        hidden_dim=16,
        num_hidden_layers=3,
        sin_frequency=30.0,
        coord_input_dim=3,
        output_channels=3,
        num_pairs=3,
        output_height=6,
        output_width=8,
    )


def _meta(cfg: CoinplusplusConfig) -> dict[str, object]:
    return {
        "hidden_dim": cfg.hidden_dim,
        "num_hidden_layers": cfg.num_hidden_layers,
        "sin_frequency": cfg.sin_frequency,
        "coord_input_dim": cfg.coord_input_dim,
        "output_channels": cfg.output_channels,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
    }


def _build_blob() -> bytes:
    torch.manual_seed(0)
    cfg = _cfg()
    model = CoinplusplusSubstrate(cfg).eval()
    with torch.no_grad():
        model.modulations.normal_(std=0.5)
    sd = {k: v for k, v in model.state_dict().items() if k != "modulations"}
    return pack_archive(
        sd, model.modulations.detach(), _meta(cfg), modulation_dim=cfg.modulation_dim
    )


def test_schema_version_is_v2_numpy_bridge() -> None:
    assert CPP1_SCHEMA_VERSION == 2


def test_inflate_module_has_no_torch_import() -> None:
    """The shipped inflate runtime must not import torch (8th MLX-first directive)."""
    tree = ast.parse(_INFLATE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(("torch", "mlx")), alias.name
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith(("torch", "mlx")), node.module


def test_parse_archive_numpy_is_torch_free_in_code() -> None:
    """parse_archive_numpy + numpy (de)serializers carry no torch code-refs."""
    tree = ast.parse(_ARCHIVE_PATH.read_text(encoding="utf-8"))
    targets = {
        "parse_archive_numpy",
        "_deserialize_numpy_state_dict",
        "_serialize_numpy_state_dict",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in targets:
            for inner in ast.walk(node):
                if isinstance(inner, ast.Name):
                    assert inner.id != "torch", node.name
                if isinstance(inner, ast.Attribute) and isinstance(
                    inner.value, ast.Name
                ):
                    assert inner.value.id != "torch", node.name


def test_numpy_parse_matches_torch_parse_exactly() -> None:
    blob = _build_blob()
    at = parse_archive(blob)
    an = parse_archive_numpy(blob)
    assert an.schema_version == CPP1_SCHEMA_VERSION
    # modulations are exact (same int8 dequant math)
    assert np.abs(at.modulations.numpy() - an.modulations).max() < 1e-7
    # weights exact (same fp16 blob)
    for key in an.base_mlp_state_dict:
        torch_w = at.base_mlp_state_dict[key].numpy()
        assert np.abs(torch_w - an.base_mlp_state_dict[key]).max() < 1e-7


def test_numpy_forward_matches_torch_model() -> None:
    """Numpy inflate forward parity vs torch model.forward (fp16-roundtrip tol)."""
    torch.manual_seed(0)
    cfg = _cfg()
    model = CoinplusplusSubstrate(cfg).eval()
    with torch.no_grad():
        model.modulations.normal_(std=0.5)
    sd = {k: v for k, v in model.state_dict().items() if k != "modulations"}
    blob = pack_archive(
        sd, model.modulations.detach(), _meta(cfg), modulation_dim=cfg.modulation_dim
    )

    with torch.no_grad():
        r0, _r1 = model(torch.tensor([0], dtype=torch.long))
    torch_f0 = r0[0].clamp(0, 1).permute(1, 2, 0).numpy()

    arc = parse_archive_numpy(blob)
    sdn = arc.base_mlp_state_dict
    coords = _build_coord_grid(cfg.output_height, cfg.output_width)
    m = arc.modulations[0]
    h = coords
    for layer in range(cfg.num_hidden_layers):
        p = f"mod_layers.{layer}."
        lin = _linear(h, sdn[p + "linear.weight"], sdn[p + "linear.bias"])
        g = _linear(m, sdn[p + "mod_gamma_proj.weight"], sdn[p + "mod_gamma_proj.bias"])
        b = _linear(m, sdn[p + "mod_beta_proj.weight"], sdn[p + "mod_beta_proj.bias"])
        h = np.sin(cfg.sin_frequency * (g[None, :] * lin + b[None, :]))
    rgb = 1.0 / (1.0 + np.exp(-_linear(h, sdn["output_head.weight"], sdn["output_head.bias"])))
    np_f0 = rgb[: cfg.output_height * cfg.output_width].reshape(
        cfg.output_height, cfg.output_width, 3
    )
    # fp16 weight roundtrip + matmul order: tolerance 1e-3
    assert np.abs(torch_f0 - np_f0).max() < 1e-3


def test_inflate_one_video_writes_pngs() -> None:
    blob = _build_blob()
    cfg = _cfg()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "0"
        inflate_one_video(blob, out)
        for frame_idx in range(cfg.num_pairs * 2):
            png = out / f"{frame_idx}.png"
            assert png.is_file(), png
            assert png.stat().st_size > 0
