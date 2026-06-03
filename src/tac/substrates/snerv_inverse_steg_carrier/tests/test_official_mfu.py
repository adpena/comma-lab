# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from tac.substrates.snerv_inverse_steg_carrier import (
    OFFICIAL_SNERV_MFU_NUMERIC_PARITY_BLOCKERS,
    OFFICIAL_SNERV_MFU_SOURCE,
    OFFICIAL_SNERV_T_MFU_SOURCE,
    SNERV_OFFICIAL_MFU_TORCH_NUMPY_MLX_PARITY_PROOF,
    ConvTranspose2dShapeSpec,
    OfficialConv2dNchw,
    OfficialConvTranspose2dNchw,
    OfficialResidualBlockNoBN,
    OfficialResidualBlocksWithInputConv,
    OfficialSnervMfu,
    OfficialSnervMfuError,
    OfficialSnervMfuSpec,
    TensorSpec,
    concat_nchw_arrays,
    concat_nchw_mlx,
    concat_nchw_specs,
    conv_transpose2d_nchw,
    conv_transpose2d_nchw_mlx,
)


def test_official_mfu_package_exports_are_available() -> None:
    import tac.substrates.snerv_inverse_steg_carrier as snerv

    assert snerv.OfficialSnervMfu is OfficialSnervMfu
    assert snerv.OfficialConvTranspose2dNchw is OfficialConvTranspose2dNchw
    assert snerv.OfficialResidualBlocksWithInputConv is OfficialResidualBlocksWithInputConv
    assert snerv.concat_nchw_arrays is concat_nchw_arrays
    assert snerv.concat_nchw_mlx is concat_nchw_mlx
    assert snerv.conv_transpose2d_nchw is conv_transpose2d_nchw
    assert snerv.conv_transpose2d_nchw_mlx is conv_transpose2d_nchw_mlx
    assert (
        snerv.SNERV_OFFICIAL_MFU_TORCH_NUMPY_MLX_PARITY_PROOF
        == SNERV_OFFICIAL_MFU_TORCH_NUMPY_MLX_PARITY_PROOF
    )


def test_official_mfu_shape_trace_matches_source_graph_contract() -> None:
    spec = OfficialSnervMfuSpec.from_official_lists(
        ngf_list=(64, 32, 16, 8),
        dec_strds=(2, 4, 2),
        num_blocks=2,
    )

    trace = spec.forward_shape(
        TensorSpec.from_shape((1, 32, 4, 5), name="low"),
        TensorSpec.from_shape((1, 16, 16, 20), name="mid"),
        TensorSpec.from_shape((1, 8, 32, 40), name="high"),
    )

    assert trace.schema == "official_snerv_mfu_shape_trace.v1"
    assert [node.name for node in trace.nodes] == [
        "up1",
        "cat_mid",
        "unet1",
        "unet1_up",
        "cat_high",
        "pyr_out",
    ]
    assert trace.output.nchw == (1, 8, 32, 40)
    assert trace.score_claim is False
    assert trace.ready_for_exact_eval_dispatch is False
    assert trace.rank_or_kill_eligible is False
    assert trace.numeric_parity_blockers == OFFICIAL_SNERV_MFU_NUMERIC_PARITY_BLOCKERS
    assert trace.parameter_shapes["decoder_len+3.weight"] == (32, 32, 4, 4)
    assert trace.parameter_shapes["decoder_len+4.main.0.weight"] == (16, 48, 3, 3)
    assert trace.parameter_shapes["decoder_len+6.main.1.1.conv2.weight"] == (8, 8, 3, 3)
    payload = trace.as_jsonable()
    assert payload["output_shape"] == [1, 8, 32, 40]
    assert payload["source"] == OFFICIAL_SNERV_MFU_SOURCE
    assert payload["promotion_eligible"] is False


def test_official_mfu_temporal_spec_pins_source_hardcoded_strides() -> None:
    spec = OfficialSnervMfuSpec.from_official_temporal_lists(
        ngf_list=(64, 32, 16, 8),
        dec_strds=(5, 4, 3, 7, 11),
        num_blocks=2,
    )

    trace = spec.forward_shape(
        TensorSpec.from_shape((1, 32, 4, 5), name="low"),
        TensorSpec.from_shape((1, 16, 8, 10), name="mid"),
        TensorSpec.from_shape((1, 8, 16, 20), name="high"),
    )

    assert spec.mid_stride == 2
    assert spec.high_stride == 2
    assert trace.source == OFFICIAL_SNERV_T_MFU_SOURCE
    assert trace.output.nchw == (1, 8, 16, 20)
    assert trace.parameter_shapes["decoder_len+3.weight"] == (32, 32, 2, 2)
    assert trace.parameter_shapes["decoder_len+5.weight"] == (16, 16, 2, 2)
    assert {node.source for node in trace.nodes if node.name != "unet1" and node.name != "pyr_out"} == {
        OFFICIAL_SNERV_T_MFU_SOURCE
    }
    assert trace.score_claim is False
    assert trace.ready_for_exact_eval_dispatch is False


def test_official_mfu_temporal_stride_control_is_distinct_from_base_snerv() -> None:
    ngf_list = (64, 32, 16, 8)
    dec_strds = (5, 4, 3, 7, 11)

    base = OfficialSnervMfuSpec.from_official_lists(
        ngf_list=ngf_list,
        dec_strds=dec_strds,
        num_blocks=1,
    )
    temporal = OfficialSnervMfuSpec.from_official_temporal_lists(
        ngf_list=ngf_list,
        dec_strds=dec_strds,
        num_blocks=1,
    )

    assert (base.mid_stride, base.high_stride) == (7, 11)
    assert base.source == OFFICIAL_SNERV_MFU_SOURCE
    assert (temporal.mid_stride, temporal.high_stride) == (2, 2)
    assert temporal.source == OFFICIAL_SNERV_T_MFU_SOURCE


def test_convtranspose_shape_spec_matches_torch_output_formula() -> None:
    spec = ConvTranspose2dShapeSpec(
        in_channels=3,
        out_channels=5,
        kernel_size=(4, 3),
        stride=(2, 3),
        padding=(1, 0),
        output_padding=(1, 0),
    )

    out = spec.forward_spec(TensorSpec.from_shape((2, 3, 7, 11)), name="up")

    assert out.nchw == (2, 5, 15, 33)
    assert spec.torch_weight_shape() == (3, 5, 4, 3)
    assert spec.torch_bias_shape() == (5,)


def test_conv_transpose2d_nchw_matches_torch_convtranspose2d() -> None:
    torch = pytest.importorskip("torch")

    rng = np.random.default_rng(11)
    x = rng.standard_normal((2, 3, 4, 5)).astype(np.float64)
    weight = (rng.standard_normal((3, 4, 3, 2)) * 0.05).astype(np.float64)
    bias = (rng.standard_normal(4) * 0.01).astype(np.float64)

    expected = torch.nn.functional.conv_transpose2d(
        torch.from_numpy(x),
        torch.from_numpy(weight),
        bias=torch.from_numpy(bias),
        stride=(2, 3),
        padding=(1, 0),
        output_padding=(1, 2),
    )
    got = conv_transpose2d_nchw(
        x,
        weight,
        bias=bias,
        stride=(2, 3),
        padding=(1, 0),
        output_padding=(1, 2),
    )
    module_got = OfficialConvTranspose2dNchw(
        weight,
        bias,
        stride=(2, 3),
        padding=(1, 0),
        output_padding=(1, 2),
    ).forward(x)

    np.testing.assert_allclose(got, expected.detach().numpy(), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(module_got, expected.detach().numpy(), atol=1e-12, rtol=1e-12)


def test_conv_transpose2d_mlx_modes_match_numpy_reference() -> None:
    mx = pytest.importorskip("mlx.core")

    rng = np.random.default_rng(111)
    x = rng.standard_normal((1, 2, 3, 4)).astype(np.float32)
    weight = (rng.standard_normal((2, 3, 2, 2)) * 0.05).astype(np.float32)
    bias = (rng.standard_normal(3) * 0.01).astype(np.float32)
    expected = conv_transpose2d_nchw(
        x,
        weight,
        bias=bias,
        stride=(2, 2),
        padding=(0, 0),
    )

    fixed = conv_transpose2d_nchw_mlx(
        mx.array(x),
        weight,
        bias=bias,
        stride=(2, 2),
        padding=(0, 0),
    )
    optimized = conv_transpose2d_nchw_mlx(
        mx.array(x),
        weight,
        bias=bias,
        stride=(2, 2),
        padding=(0, 0),
        accumulation_mode="optimized",
    )

    np.testing.assert_allclose(np.array(fixed), expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(np.array(optimized), expected, atol=5e-3, rtol=5e-3)


def test_official_mfu_residual_blocks_match_torch_source_block() -> None:
    torch = pytest.importorskip("torch")

    rng = np.random.default_rng(12)
    x = rng.standard_normal((1, 5, 4, 6)).astype(np.float64)
    input_w = (rng.standard_normal((3, 5, 3, 3)) * 0.04).astype(np.float64)
    input_b = (rng.standard_normal(3) * 0.01).astype(np.float64)
    conv1_w = (rng.standard_normal((3, 3, 3, 3)) * 0.04).astype(np.float64)
    conv1_b = (rng.standard_normal(3) * 0.01).astype(np.float64)
    conv2_w = (rng.standard_normal((3, 3, 3, 3)) * 0.04).astype(np.float64)
    conv2_b = (rng.standard_normal(3) * 0.01).astype(np.float64)
    block = OfficialResidualBlocksWithInputConv(
        input_conv=OfficialConv2dNchw(input_w, input_b, padding=1),
        residual_blocks=(
            OfficialResidualBlockNoBN(
                conv1=OfficialConv2dNchw(conv1_w, conv1_b, padding=1),
                conv2=OfficialConv2dNchw(conv2_w, conv2_b, padding=1),
            ),
        ),
    )

    input_conv = torch.nn.Conv2d(5, 3, 3, 1, 1, bias=True, dtype=torch.float64)
    rb_conv1 = torch.nn.Conv2d(3, 3, 3, 1, 1, bias=True, dtype=torch.float64)
    rb_conv2 = torch.nn.Conv2d(3, 3, 3, 1, 1, bias=True, dtype=torch.float64)
    with torch.no_grad():
        input_conv.weight.copy_(torch.from_numpy(input_w))
        input_conv.bias.copy_(torch.from_numpy(input_b))
        rb_conv1.weight.copy_(torch.from_numpy(conv1_w))
        rb_conv1.bias.copy_(torch.from_numpy(conv1_b))
        rb_conv2.weight.copy_(torch.from_numpy(conv2_w))
        rb_conv2.bias.copy_(torch.from_numpy(conv2_b))
    hidden = input_conv(torch.from_numpy(x))
    expected = hidden + rb_conv2(
        torch.nn.LeakyReLU(negative_slope=0.1)(rb_conv1(hidden))
    )

    np.testing.assert_allclose(block.forward(x), expected.detach().numpy(), atol=1e-12, rtol=1e-12)


def test_official_mfu_full_numpy_forward_matches_torch_graph() -> None:
    torch = pytest.importorskip("torch")

    rng = np.random.default_rng(13)
    spec = OfficialSnervMfuSpec(
        low_channels=2,
        mid_channels=3,
        high_channels=4,
        mid_stride=2,
        high_stride=2,
        num_blocks=1,
    )
    up_mid_w = (rng.standard_normal((2, 2, 2, 2)) * 0.04).astype(np.float64)
    up_mid_b = (rng.standard_normal(2) * 0.01).astype(np.float64)
    mid_input_w = (rng.standard_normal((3, 5, 3, 3)) * 0.04).astype(np.float64)
    mid_input_b = (rng.standard_normal(3) * 0.01).astype(np.float64)
    mid_conv1_w = (rng.standard_normal((3, 3, 3, 3)) * 0.04).astype(np.float64)
    mid_conv1_b = (rng.standard_normal(3) * 0.01).astype(np.float64)
    mid_conv2_w = (rng.standard_normal((3, 3, 3, 3)) * 0.04).astype(np.float64)
    mid_conv2_b = (rng.standard_normal(3) * 0.01).astype(np.float64)
    up_high_w = (rng.standard_normal((3, 3, 2, 2)) * 0.04).astype(np.float64)
    up_high_b = (rng.standard_normal(3) * 0.01).astype(np.float64)
    high_input_w = (rng.standard_normal((4, 7, 3, 3)) * 0.04).astype(np.float64)
    high_input_b = (rng.standard_normal(4) * 0.01).astype(np.float64)
    high_conv1_w = (rng.standard_normal((4, 4, 3, 3)) * 0.04).astype(np.float64)
    high_conv1_b = (rng.standard_normal(4) * 0.01).astype(np.float64)
    high_conv2_w = (rng.standard_normal((4, 4, 3, 3)) * 0.04).astype(np.float64)
    high_conv2_b = (rng.standard_normal(4) * 0.01).astype(np.float64)
    mfu = OfficialSnervMfu(
        spec=spec,
        upsample_mid=OfficialConvTranspose2dNchw(up_mid_w, up_mid_b, stride=2),
        rb_mid=OfficialResidualBlocksWithInputConv(
            input_conv=OfficialConv2dNchw(mid_input_w, mid_input_b, padding=1),
            residual_blocks=(
                OfficialResidualBlockNoBN(
                    conv1=OfficialConv2dNchw(mid_conv1_w, mid_conv1_b, padding=1),
                    conv2=OfficialConv2dNchw(mid_conv2_w, mid_conv2_b, padding=1),
                ),
            ),
        ),
        upsample_high=OfficialConvTranspose2dNchw(up_high_w, up_high_b, stride=2),
        rb_high=OfficialResidualBlocksWithInputConv(
            input_conv=OfficialConv2dNchw(high_input_w, high_input_b, padding=1),
            residual_blocks=(
                OfficialResidualBlockNoBN(
                    conv1=OfficialConv2dNchw(high_conv1_w, high_conv1_b, padding=1),
                    conv2=OfficialConv2dNchw(high_conv2_w, high_conv2_b, padding=1),
                ),
            ),
        ),
    )
    low = rng.standard_normal((1, 2, 2, 3)).astype(np.float64)
    skip_mid = rng.standard_normal((1, 3, 4, 6)).astype(np.float64)
    skip_high = rng.standard_normal((1, 4, 8, 12)).astype(np.float64)

    got = mfu.forward(low, skip_mid, skip_high)

    low_t = torch.from_numpy(low)
    skip_mid_t = torch.from_numpy(skip_mid)
    skip_high_t = torch.from_numpy(skip_high)
    up1 = torch.nn.functional.conv_transpose2d(
        low_t,
        torch.from_numpy(up_mid_w),
        bias=torch.from_numpy(up_mid_b),
        stride=2,
    )
    mid_hidden = torch.nn.functional.conv2d(
        torch.cat([up1, skip_mid_t], dim=1),
        torch.from_numpy(mid_input_w),
        bias=torch.from_numpy(mid_input_b),
        padding=1,
    )
    unet1 = mid_hidden + torch.nn.functional.conv2d(
        torch.nn.functional.leaky_relu(
            torch.nn.functional.conv2d(
                mid_hidden,
                torch.from_numpy(mid_conv1_w),
                bias=torch.from_numpy(mid_conv1_b),
                padding=1,
            ),
            negative_slope=0.1,
        ),
        torch.from_numpy(mid_conv2_w),
        bias=torch.from_numpy(mid_conv2_b),
        padding=1,
    )
    unet1_up = torch.nn.functional.conv_transpose2d(
        unet1,
        torch.from_numpy(up_high_w),
        bias=torch.from_numpy(up_high_b),
        stride=2,
    )
    high_hidden = torch.nn.functional.conv2d(
        torch.cat([unet1_up, skip_high_t], dim=1),
        torch.from_numpy(high_input_w),
        bias=torch.from_numpy(high_input_b),
        padding=1,
    )
    expected = high_hidden + torch.nn.functional.conv2d(
        torch.nn.functional.leaky_relu(
            torch.nn.functional.conv2d(
                high_hidden,
                torch.from_numpy(high_conv1_w),
                bias=torch.from_numpy(high_conv1_b),
                padding=1,
            ),
            negative_slope=0.1,
        ),
        torch.from_numpy(high_conv2_w),
        bias=torch.from_numpy(high_conv2_b),
        padding=1,
    )

    np.testing.assert_allclose(got.up1, up1.detach().numpy(), atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(got.unet1, unet1.detach().numpy(), atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(got.pyr_out, expected.detach().numpy(), atol=1e-11, rtol=1e-11)
    metadata = got.as_jsonable_metadata()
    assert metadata["score_claim"] is False
    assert metadata["rank_or_kill_eligible"] is False
    assert metadata["ready_for_exact_eval_dispatch"] is False
    assert metadata["source_forward_replay_authority"] is False


def test_official_mfu_mlx_forward_modes_match_numpy_reference() -> None:
    mx = pytest.importorskip("mlx.core")

    mfu, low, skip_mid, skip_high = _tiny_mfu_fixture(seed=113)
    expected = mfu.forward(low, skip_mid, skip_high)

    fixed = mfu.forward_mlx(mx.array(low), mx.array(skip_mid), mx.array(skip_high))
    optimized = mfu.forward_mlx(
        mx.array(low),
        mx.array(skip_mid),
        mx.array(skip_high),
        accumulation_mode="optimized",
    )

    np.testing.assert_allclose(np.array(fixed.up1), expected.up1, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(np.array(fixed.unet1), expected.unet1, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(
        np.array(fixed.pyr_out),
        expected.pyr_out,
        atol=5e-5,
        rtol=5e-5,
    )
    np.testing.assert_allclose(
        np.array(optimized.pyr_out),
        expected.pyr_out,
        atol=5e-3,
        rtol=5e-3,
    )


def test_official_mfu_rejects_shape_equivalent_non_source_upsamplers() -> None:
    rng = np.random.default_rng(14)
    spec = OfficialSnervMfuSpec(
        low_channels=2,
        mid_channels=3,
        high_channels=4,
        mid_stride=2,
        high_stride=2,
        num_blocks=1,
    )

    with pytest.raises(OfficialSnervMfuError, match="kernel=stride"):
        OfficialSnervMfu(
            spec=spec,
            upsample_mid=OfficialConvTranspose2dNchw(
                rng.standard_normal((2, 2, 3, 3)),
                stride=2,
                padding=1,
                output_padding=1,
            ),
            rb_mid=_rb(rng, in_ch=5, out_ch=3, blocks=1),
            upsample_high=OfficialConvTranspose2dNchw(
                rng.standard_normal((3, 3, 2, 2)),
                stride=2,
            ),
            rb_high=_rb(rng, in_ch=7, out_ch=4, blocks=1),
        )


def test_official_mfu_rejects_residual_block_count_mismatch() -> None:
    rng = np.random.default_rng(15)
    spec = OfficialSnervMfuSpec(
        low_channels=2,
        mid_channels=3,
        high_channels=4,
        mid_stride=2,
        high_stride=2,
        num_blocks=2,
    )

    with pytest.raises(OfficialSnervMfuError, match="block count mismatch"):
        OfficialSnervMfu(
            spec=spec,
            upsample_mid=OfficialConvTranspose2dNchw(
                rng.standard_normal((2, 2, 2, 2)),
                stride=2,
            ),
            rb_mid=_rb(rng, in_ch=5, out_ch=3, blocks=1),
            upsample_high=OfficialConvTranspose2dNchw(
                rng.standard_normal((3, 3, 2, 2)),
                stride=2,
            ),
            rb_high=_rb(rng, in_ch=7, out_ch=4, blocks=2),
        )


def test_official_mfu_rejects_skip_shape_and_channel_mismatches() -> None:
    spec = OfficialSnervMfuSpec(
        low_channels=4,
        mid_channels=3,
        high_channels=2,
        mid_stride=2,
        high_stride=2,
        num_blocks=1,
    )

    with pytest.raises(OfficialSnervMfuError, match="expected 4 input channels"):
        spec.forward_shape((1, 5, 4, 4), (1, 3, 8, 8), (1, 2, 16, 16))
    with pytest.raises(OfficialSnervMfuError, match="matching N/H/W"):
        concat_nchw_specs(
            (
                TensorSpec.from_shape((1, 4, 8, 8)),
                TensorSpec.from_shape((1, 3, 7, 8)),
            ),
            name="bad",
        )
    with pytest.raises(OfficialSnervMfuError, match="RB expected 7 input channels"):
        spec.forward_shape((1, 4, 4, 4), (1, 2, 8, 8), (1, 2, 16, 16))


def test_official_mfu_accepts_arrays_without_numeric_parity_claim() -> None:
    low = np.zeros((1, 4, 3, 3), dtype=np.float32)
    mid = np.zeros((1, 3, 6, 6), dtype=np.float32)
    high = np.zeros((1, 2, 12, 12), dtype=np.float32)
    spec = OfficialSnervMfuSpec(
        low_channels=4,
        mid_channels=3,
        high_channels=2,
        mid_stride=2,
        high_stride=2,
        num_blocks=0,
    )

    trace = spec.forward_shape(low, mid, high)

    assert trace.output.nchw == (1, 2, 12, 12)
    assert "official_weight_tensor_mapping_not_loaded" in trace.numeric_parity_blockers


def _conv(
    rng: np.random.Generator,
    *,
    out_ch: int,
    in_ch: int,
) -> OfficialConv2dNchw:
    return OfficialConv2dNchw(
        rng.standard_normal((out_ch, in_ch, 3, 3)) * 0.04,
        rng.standard_normal(out_ch) * 0.01,
        padding=1,
    )


def _rb(
    rng: np.random.Generator,
    *,
    in_ch: int,
    out_ch: int,
    blocks: int,
) -> OfficialResidualBlocksWithInputConv:
    return OfficialResidualBlocksWithInputConv(
        input_conv=_conv(rng, out_ch=out_ch, in_ch=in_ch),
        residual_blocks=tuple(
            OfficialResidualBlockNoBN(
                conv1=_conv(rng, out_ch=out_ch, in_ch=out_ch),
                conv2=_conv(rng, out_ch=out_ch, in_ch=out_ch),
            )
            for _ in range(int(blocks))
        ),
    )


def _tiny_mfu_fixture(
    *,
    seed: int,
) -> tuple[OfficialSnervMfu, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    spec = OfficialSnervMfuSpec(
        low_channels=2,
        mid_channels=3,
        high_channels=4,
        mid_stride=2,
        high_stride=2,
        num_blocks=1,
    )
    mfu = OfficialSnervMfu(
        spec=spec,
        upsample_mid=OfficialConvTranspose2dNchw(
            rng.standard_normal((2, 2, 2, 2)) * 0.04,
            rng.standard_normal(2) * 0.01,
            stride=2,
        ),
        rb_mid=_rb(rng, in_ch=5, out_ch=3, blocks=1),
        upsample_high=OfficialConvTranspose2dNchw(
            rng.standard_normal((3, 3, 2, 2)) * 0.04,
            rng.standard_normal(3) * 0.01,
            stride=2,
        ),
        rb_high=_rb(rng, in_ch=7, out_ch=4, blocks=1),
    )
    return (
        mfu,
        rng.standard_normal((1, 2, 2, 3)).astype(np.float32),
        rng.standard_normal((1, 3, 4, 6)).astype(np.float32),
        rng.standard_normal((1, 4, 8, 12)).astype(np.float32),
    )
