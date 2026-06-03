# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from tac.substrates.snerv_inverse_steg_carrier import (
    OFFICIAL_SNERV_MFU_NUMERIC_PARITY_BLOCKERS,
    ConvTranspose2dShapeSpec,
    OfficialSnervMfuError,
    OfficialSnervMfuSpec,
    TensorSpec,
    concat_nchw_specs,
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
    assert trace.numeric_parity_blockers == OFFICIAL_SNERV_MFU_NUMERIC_PARITY_BLOCKERS
    assert trace.parameter_shapes["decoder_len+3.weight"] == (32, 32, 4, 4)
    assert trace.parameter_shapes["decoder_len+4.main.0.weight"] == (16, 48, 3, 3)
    assert trace.parameter_shapes["decoder_len+6.main.1.1.conv2.weight"] == (8, 8, 3, 3)
    payload = trace.as_jsonable()
    assert payload["output_shape"] == [1, 8, 32, 40]
    assert payload["promotion_eligible"] is False


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
