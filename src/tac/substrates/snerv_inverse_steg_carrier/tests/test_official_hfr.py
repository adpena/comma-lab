# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tac.substrates.snerv_inverse_steg_carrier.official_hfr import (
    OFFICIAL_SNERV_HFR_SOURCE_CONTRACT,
    OFFICIAL_SNERV_HFR_SOURCE_SHA,
    SNERV_OFFICIAL_HFR_CONVBLOCK_NUMPY_PROOF,
    OfficialConv2dNchw,
    OfficialHfrConvBlock,
    OfficialHfrHeads,
    OfficialSnervHfrError,
    conv2d_nchw,
    conv2d_nchw_mlx,
    leaky_relu01,
)

DEFAULT_OFFICIAL_SNERV_REPO = Path(
    "/Volumes/VertigoDataTier/pact/experiments/results/"
    "oss_nerv_source_audit_20260602T113720Z/repos/SNeRV"
)


def test_official_hfr_package_exports_are_available() -> None:
    import tac.substrates.snerv_inverse_steg_carrier as snerv

    assert snerv.OFFICIAL_SNERV_HFR_SOURCE_SHA == OFFICIAL_SNERV_HFR_SOURCE_SHA
    assert snerv.OFFICIAL_SNERV_HFR_SOURCE_CONTRACT == OFFICIAL_SNERV_HFR_SOURCE_CONTRACT
    assert snerv.OfficialHfrConvBlock is OfficialHfrConvBlock
    assert snerv.OfficialHfrHeads is OfficialHfrHeads
    assert snerv.OfficialConv2dNchw is OfficialConv2dNchw
    assert snerv.SNERV_OFFICIAL_HFR_CONVBLOCK_NUMPY_PROOF == (
        SNERV_OFFICIAL_HFR_CONVBLOCK_NUMPY_PROOF
    )
    assert snerv.OFFICIAL_SNERV_HFR_FALSE_AUTHORITY["score_claim"] is False


def test_official_hfr_source_contract_is_pinned() -> None:
    repo = _official_repo()

    _assert_source_line(
        repo,
        "model/snerv.py",
        91,
        "idwt = IDWT(wave='haar', mode='periodization').cuda()",
    )
    _assert_source_line(
        repo,
        "model/snerv.py",
        62,
        "decoder_layer2 = ConvBlock(ngf1=new_ngf, ngf2=new_ngf, out=3, act='leaky01')",
    )
    _assert_source_line(
        repo,
        "model/snerv.py",
        63,
        "decoder_layer3 = ConvBlock(ngf1=new_ngf, ngf2=new_ngf, out=3, act='leaky01')",
    )
    _assert_source_line(
        repo,
        "model/snerv.py",
        64,
        "decoder_layer4 = ConvBlock(ngf1=new_ngf, ngf2=new_ngf, out=3, act='leaky01')",
    )
    _assert_source_line(repo, "model/snerv.py", 115, "HF_in = pyr_out")
    _assert_source_line(
        repo,
        "model/snerv.py",
        116,
        "lh_out = self.decoder[self.decoder_len](HF_in)",
    )
    _assert_source_line(
        repo,
        "model/snerv.py",
        117,
        "hl_out = self.decoder[self.decoder_len+1](HF_in)",
    )
    _assert_source_line(
        repo,
        "model/snerv.py",
        118,
        "hh_out = self.decoder[self.decoder_len+2](HF_in)",
    )
    _assert_source_line(
        repo,
        "model/snerv.py",
        120,
        "yh_out = torch.stack([lh_out, hl_out, hh_out], dim=2)",
    )
    _assert_source_line(repo, "model/snerv.py", 122, "img_out = idwt([yl_out, [yh_out]])")
    _assert_source_line(
        repo,
        "model/layers.py",
        144,
        "self.conv1 = nn.Conv2d(kargs['ngf1'], kargs['ngf2'], 1, 1, 0, bias=True)",
    )
    _assert_source_line(
        repo,
        "model/layers.py",
        145,
        "self.conv2 = nn.Conv2d(kargs['ngf2'], kargs['out'], 3, 1, 1)",
    )
    _assert_source_line(
        repo,
        "model/layers.py",
        148,
        "self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)",
    )
    _assert_source_line(
        repo,
        "model/layers.py",
        159,
        "x = self.act(self.norm(self.conv1(x)))",
    )
    _assert_source_line(repo, "model/layers.py", 160, "x = self.conv2(x)")


def test_official_hfr_convblock_shape_and_stack_contract() -> None:
    rng = np.random.default_rng(1)
    heads = OfficialHfrHeads(
        lh_head=_head(rng, in_ch=4, hidden_ch=5),
        hl_head=_head(rng, in_ch=4, hidden_ch=5),
        hh_head=_head(rng, in_ch=4, hidden_ch=5),
    )
    pyr_out = rng.standard_normal((2, 4, 6, 7))

    out = heads.forward(pyr_out)

    assert SNERV_OFFICIAL_HFR_CONVBLOCK_NUMPY_PROOF.endswith("numpy_nchw")
    assert out.lh.shape == (2, 3, 6, 7)
    assert out.hl.shape == (2, 3, 6, 7)
    assert out.hh.shape == (2, 3, 6, 7)
    assert out.yh_out.shape == (2, 3, 3, 6, 7)
    np.testing.assert_allclose(out.yh_out[:, :, 0], out.lh)
    np.testing.assert_allclose(out.yh_out[:, :, 1], out.hl)
    np.testing.assert_allclose(out.yh_out[:, :, 2], out.hh)
    payload = out.as_jsonable()
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_official_hfr_heads_forward_mlx_preserves_tensor_and_stack_contract() -> None:
    mx = pytest.importorskip("mlx.core")

    rng = np.random.default_rng(11)
    heads = OfficialHfrHeads(
        lh_head=_head(rng, in_ch=4, hidden_ch=5),
        hl_head=_head(rng, in_ch=4, hidden_ch=5),
        hh_head=_head(rng, in_ch=4, hidden_ch=5),
    )
    pyr_out = rng.standard_normal((2, 4, 5, 6)).astype(np.float32)

    expected = heads.forward(pyr_out)
    got = heads.forward_mlx(mx.array(pyr_out), accumulation_mode="optimized")

    assert tuple(int(v) for v in got.yh_out.shape) == (2, 3, 3, 5, 6)
    np.testing.assert_allclose(np.asarray(got.lh), expected.lh, atol=1e-4, rtol=2e-3)
    np.testing.assert_allclose(np.asarray(got.hl), expected.hl, atol=1e-4, rtol=2e-3)
    np.testing.assert_allclose(np.asarray(got.hh), expected.hh, atol=1e-4, rtol=2e-3)
    np.testing.assert_allclose(np.asarray(got.yh_out[:, :, 0]), np.asarray(got.lh))
    np.testing.assert_allclose(np.asarray(got.yh_out[:, :, 1]), np.asarray(got.hl))
    np.testing.assert_allclose(np.asarray(got.yh_out[:, :, 2]), np.asarray(got.hh))


def test_official_hfr_rejects_non_source_shapes() -> None:
    rng = np.random.default_rng(2)
    with pytest.raises(OfficialSnervHfrError, match="conv1 kernel must be 1x1"):
        OfficialHfrConvBlock(
            conv1=OfficialConv2dNchw(rng.standard_normal((5, 4, 3, 3)), padding=1),
            conv2=OfficialConv2dNchw(rng.standard_normal((3, 5, 3, 3)), padding=1),
        )
    with pytest.raises(OfficialSnervHfrError, match="head must output 3"):
        OfficialHfrConvBlock(
            conv1=OfficialConv2dNchw(rng.standard_normal((5, 4, 1, 1))),
            conv2=OfficialConv2dNchw(rng.standard_normal((2, 5, 3, 3)), padding=1),
        )
    heads = OfficialHfrHeads(
        lh_head=_head(rng, in_ch=4, hidden_ch=5),
        hl_head=_head(rng, in_ch=4, hidden_ch=5),
        hh_head=_head(rng, in_ch=4, hidden_ch=5),
    )
    with pytest.raises(OfficialSnervHfrError, match="pyr_out channels"):
        heads.forward(rng.standard_normal((1, 3, 4, 4)))


def test_conv2d_nchw_matches_small_manual_result() -> None:
    x = np.arange(1 * 1 * 3 * 3, dtype=np.float64).reshape(1, 1, 3, 3)
    weight = np.ones((1, 1, 2, 2), dtype=np.float64)

    out = conv2d_nchw(x, weight)

    expected = np.array([[[[8.0, 12.0], [20.0, 24.0]]]], dtype=np.float64)
    np.testing.assert_allclose(out, expected)


def test_official_hfr_numpy_matches_torch_convblock() -> None:
    torch = pytest.importorskip("torch")

    rng = np.random.default_rng(3)
    conv1_w = rng.standard_normal((5, 4, 1, 1)) * 0.1
    conv1_b = rng.standard_normal(5) * 0.1
    conv2_w = rng.standard_normal((3, 5, 3, 3)) * 0.1
    conv2_b = rng.standard_normal(3) * 0.1
    x = rng.standard_normal((2, 4, 5, 6))
    block = OfficialHfrConvBlock(
        conv1=OfficialConv2dNchw(conv1_w, conv1_b),
        conv2=OfficialConv2dNchw(conv2_w, conv2_b, padding=1),
    )

    conv1 = torch.nn.Conv2d(4, 5, 1, 1, 0, bias=True, dtype=torch.float64)
    conv2 = torch.nn.Conv2d(5, 3, 3, 1, 1, bias=True, dtype=torch.float64)
    with torch.no_grad():
        conv1.weight.copy_(torch.from_numpy(conv1_w))
        conv1.bias.copy_(torch.from_numpy(conv1_b))
        conv2.weight.copy_(torch.from_numpy(conv2_w))
        conv2.bias.copy_(torch.from_numpy(conv2_b))
    expected = conv2(torch.nn.LeakyReLU(negative_slope=0.1)(conv1(torch.from_numpy(x))))

    np.testing.assert_allclose(block.forward(x), expected.detach().numpy(), atol=1e-10)


def test_official_hfr_mlx_optimized_matches_numpy_convblock() -> None:
    mx = pytest.importorskip("mlx.core")

    rng = np.random.default_rng(4)
    block = _head(rng, in_ch=4, hidden_ch=5)
    x = rng.standard_normal((2, 4, 5, 6)).astype(np.float32)

    expected = block.forward(x)
    got = block.forward_mlx(mx.array(x), accumulation_mode="optimized")

    # Native MLX conv is the throughput path, not the fixed-order parity path;
    # the measured drift on Apple Metal is O(1e-5) for this official HFR block.
    np.testing.assert_allclose(np.asarray(got), expected, atol=1e-4, rtol=2e-3)


def test_official_hfr_mlx_default_is_fixed_reference_and_repeatable() -> None:
    mx = pytest.importorskip("mlx.core")

    rng = np.random.default_rng(45)
    block = _head(rng, in_ch=3, hidden_ch=4)
    x = rng.standard_normal((1, 3, 4, 4)).astype(np.float32)

    expected = block.forward(x)
    first = block.forward_mlx(mx.array(x))
    second = block.forward_mlx(mx.array(x))

    np.testing.assert_allclose(np.asarray(first), expected, atol=2e-5, rtol=2e-5)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def test_official_hfr_mlx_fixed_reference_matches_numpy_conv() -> None:
    mx = pytest.importorskip("mlx.core")

    rng = np.random.default_rng(5)
    x = rng.standard_normal((1, 2, 4, 5)).astype(np.float32)
    weight = (rng.standard_normal((3, 2, 3, 3)) * 0.05).astype(np.float32)
    bias = (rng.standard_normal(3) * 0.01).astype(np.float32)

    expected = conv2d_nchw(x, weight, bias=bias, padding=1)
    got = conv2d_nchw_mlx(
        mx.array(x),
        weight,
        bias=bias,
        padding=1,
        accumulation_mode="fixed_fp32",
    )

    np.testing.assert_allclose(np.asarray(got), expected, atol=2e-5, rtol=2e-5)


def test_leaky_relu01_matches_official_slope() -> None:
    x = np.array([-2.0, -0.5, 0.0, 3.0])
    np.testing.assert_allclose(leaky_relu01(x), np.array([-0.2, -0.05, 0.0, 3.0]))


def _head(
    rng: np.random.Generator,
    *,
    in_ch: int,
    hidden_ch: int,
) -> OfficialHfrConvBlock:
    return OfficialHfrConvBlock(
        conv1=OfficialConv2dNchw(
            rng.standard_normal((hidden_ch, in_ch, 1, 1)) * 0.05,
            rng.standard_normal(hidden_ch) * 0.01,
        ),
        conv2=OfficialConv2dNchw(
            rng.standard_normal((3, hidden_ch, 3, 3)) * 0.05,
            rng.standard_normal(3) * 0.01,
            padding=1,
        ),
    )


def _official_repo() -> Path:
    repo = Path(os.environ.get("PACT_SNERV_OFFICIAL_REPO", DEFAULT_OFFICIAL_SNERV_REPO))
    if not repo.exists():
        pytest.skip(f"official SNeRV checkout is absent: {repo}")
    result = subprocess.run(
        ["git", "-C", repo.as_posix(), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.stdout.strip() == OFFICIAL_SNERV_HFR_SOURCE_SHA
    return repo


def _assert_source_line(repo: Path, rel_path: str, line_no: int, snippet: str) -> None:
    path = repo / rel_path
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[line_no - 1].strip() == snippet
