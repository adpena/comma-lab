# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import tac.substrates.snerv_inverse_steg_carrier.official_tub as official_tub_mod
import tac.substrates.snerv_inverse_steg_carrier.official_tub_torch as official_tub_torch_mod
from tac.substrates.snerv_inverse_steg_carrier import (
    OFFICIAL_SNERV_T_SOURCE_SHA,
    OFFICIAL_SNERV_T_TUB_EVIDENCE_SCOPE,
    OFFICIAL_SNERV_T_TUB_SCHEMA,
    OFFICIAL_SNERV_T_TUB_SOURCE_FORWARD_BLOCKERS,
    OfficialTubError,
    official_output2_fusion_mlx,
    official_output2_fusion_numpy,
    official_output2_fusion_shape,
    official_output2_fusion_torch,
    prepare_official_tub_graph_inputs,
)

DEFAULT_OFFICIAL_SNERV_REPO = Path(
    "/Volumes/VertigoDataTier/pact/experiments/results/"
    "oss_nerv_source_audit_20260602T113720Z/repos/SNeRV"
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
    assert result.stdout.strip() == OFFICIAL_SNERV_T_SOURCE_SHA
    return repo


def _assert_source_line(repo: Path, rel_path: str, line_no: int, snippet: str) -> None:
    path = repo / rel_path
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[line_no - 1].strip() == snippet


def _haar_lf_chw(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float64)
    a = arr[:, 0::2, 0::2]
    b = arr[:, 0::2, 1::2]
    c = arr[:, 1::2, 0::2]
    d = arr[:, 1::2, 1::2]
    return (a + b + c + d) * 0.5


def test_official_snerv_t_tub_source_contract_is_pinned() -> None:
    repo = _official_repo()

    _assert_source_line(
        repo,
        "model/snerv_t.py",
        125,
        "yl, _ = DWT(J=1, wave='haar', mode='periodization').cuda()(torch.cat([input, input_p, input_n],0))",
    )
    _assert_source_line(
        repo,
        "model/snerv_t.py",
        126,
        "yl_norm = torch.as_tensor([yl.min(), yl.max()])",
    )
    _assert_source_line(
        repo,
        "model/snerv_t.py",
        127,
        "embed = (yl-yl_norm[0])/(yl_norm[1]-yl_norm[0]) ### normalize",
    )
    _assert_source_line(
        repo,
        "model/snerv_t.py",
        130,
        "embed_lv_p, embed_hv_p = DWT1D(J=1, wave='haar', mode='periodization').cuda()(torch.cat([embed[0:1], embed[1:2]],0).reshape(n,c,h*w).permute(2,1,0))",
    )
    _assert_source_line(
        repo,
        "model/snerv_t.py",
        131,
        "embed_lv_n, embed_hv_n = DWT1D(J=1, wave='haar', mode='periodization').cuda()(torch.cat([embed[0:1], embed[2:3]],0).reshape(n,c,h*w).permute(2,1,0))",
    )
    _assert_source_line(
        repo,
        "model/snerv_t.py",
        134,
        "embed_hv_p = self.encoder[1]((embed_lv_p.permute(2,1,0).reshape(1,c,h,w))/2)",
    )
    _assert_source_line(
        repo,
        "model/snerv_t.py",
        135,
        "embed_hv_n = self.encoder[2]((embed_lv_n.permute(2,1,0).reshape(1,c,h,w))/2)",
    )
    _assert_source_line(
        repo,
        "model/snerv_t.py",
        136,
        "img_embed = [embed_curr, torch.cat([embed_hv_p, embed_hv_n],1), yl_norm]",
    )
    _assert_source_line(
        repo,
        "model/snerv_t.py",
        148,
        "output_2 = self.decoder[self.decoder_len-1](torch.cat([img_embed[1][:,0:emb_ch,:,:], img_embed[1][:,emb_ch:,:,:]],0))",
    )
    _assert_source_line(
        repo,
        "model/snerv_t.py",
        150,
        "output_2 = output_2.view(n, -1, self.fc_h, self.fc_w, h, w).permute(0,1,4,2,5,3).reshape(n,-1,self.fc_h * h, self.fc_w * w)",
    )
    _assert_source_line(repo, "model/snerv_t.py", 161, "### MFU")
    _assert_source_line(
        repo,
        "model/snerv_t.py",
        162,
        "up1 = self.decoder[self.decoder_len+3](embed_list[-3])",
    )
    _assert_source_line(
        repo,
        "model/snerv_t.py",
        163,
        "unet1 = self.decoder[self.decoder_len+4](torch.cat([up1, embed_list[-2]], dim=1))",
    )
    _assert_source_line(
        repo,
        "model/snerv_t.py",
        165,
        "pyr_out = self.decoder[self.decoder_len+6](torch.cat([unet1_up, embed_list[-1]], dim=1))",
    )
    _assert_source_line(
        repo,
        "model/snerv_t.py",
        167,
        "img_yl = OutImg(self.head_layer(pyr_out), self.out_bias)",
    )
    _assert_source_line(
        repo,
        "model/snerv_t.py",
        168,
        "yl_out = img_yl * (img_embed[2][1]-img_embed[2][0]) + img_embed[2][0] ### needed",
    )
    _assert_source_line(repo, "model/snerv_t.py", 170, "### HFR")
    _assert_source_line(repo, "model/snerv_t.py", 171, "HF_in = pyr_out")
    _assert_source_line(
        repo,
        "model/snerv_t.py",
        176,
        "yh_out = torch.stack([lh_out, hl_out, hh_out], dim=2)",
    )
    _assert_source_line(
        repo,
        "model/snerv_t.py",
        178,
        "img_out = idwt([yl_out, [yh_out]])",
    )
    _assert_source_line(repo, "requirements.txt", 1, "torch==1.8.1")
    _assert_source_line(repo, "requirements.txt", 13, "pytorch-wavelets==1.3.0")
    _assert_source_line(repo, "requirements.txt", 14, "PyWavelets==1.4.1")


def test_official_tub_graph_inputs_match_haar_lowpass_contract() -> None:
    current = np.array(
        [[[0.0, 2.0, 4.0, 6.0], [8.0, 10.0, 12.0, 14.0]]],
        dtype=np.float64,
    )
    previous = current + 2.0
    next_frame = current + 4.0

    out = prepare_official_tub_graph_inputs(current, previous, next_frame)

    assert out.schema == OFFICIAL_SNERV_T_TUB_SCHEMA
    assert out.score_claim is False
    assert out.promotion_eligible is False
    expected_lf_current = np.array([[[[10.0, 18.0]]]], dtype=np.float64)
    expected_lf = np.stack(
        [
            _haar_lf_chw(current),
            _haar_lf_chw(previous),
            _haar_lf_chw(next_frame),
        ],
        axis=0,
    )
    expected_normalized = (expected_lf - expected_lf.min()) / (
        expected_lf.max() - expected_lf.min()
    )
    np.testing.assert_allclose(out.lf_triplet[0:1], expected_lf_current)
    np.testing.assert_allclose(out.lf_triplet, expected_lf)
    normalized = out.normalized_lf
    np.testing.assert_allclose(normalized, expected_normalized)
    np.testing.assert_allclose(out.current_lf, normalized[0:1])
    np.testing.assert_allclose(
        out.prev_lowpass_over_2,
        (normalized[0:1] + normalized[1:2]) / (2.0 * np.sqrt(2.0)),
    )
    np.testing.assert_allclose(
        out.next_lowpass_over_2,
        (normalized[0:1] + normalized[2:3]) / (2.0 * np.sqrt(2.0)),
    )
    metadata = out.as_jsonable_metadata()
    assert metadata["shape_metadata"]["temporal_encoder_input_count"] == 2
    assert metadata["source_equivalence_scope"] == OFFICIAL_SNERV_T_TUB_EVIDENCE_SCOPE
    assert metadata["primitive_numeric_graph_input_parity_proven"] is True
    assert metadata["source_forward_parity_proven"] is False
    assert metadata["full_tub_source_forward_parity_proven"] is False
    assert metadata["source_forward_replay_bound"] is False
    assert metadata["source_forward_replay_verified"] is False
    assert metadata["source_forward_replay_authority"] is False
    assert metadata["source_forward_blockers"] == list(
        OFFICIAL_SNERV_T_TUB_SOURCE_FORWARD_BLOCKERS
    )
    assert (
        "snerv_official_tub_normalized_lf_graph_inputs_not_full_source_forward_parity"
        in metadata["source_forward_blockers"]
    )
    assert metadata["score_claim"] is False
    assert metadata["promotion_eligible"] is False
    assert metadata["rank_or_kill_eligible"] is False
    assert metadata["ready_for_exact_eval_dispatch"] is False


def test_dwt1d_pair_reconstructs_current_and_previous_normalized_lf() -> None:
    current = np.arange(2 * 4 * 4, dtype=np.float64).reshape(2, 4, 4)
    previous = current + 3.0
    next_frame = current - 4.0

    out = prepare_official_tub_graph_inputs(current, previous, next_frame)

    prev_lowpass = out.prev_lowpass_over_2 * 2.0
    recovered_current = (prev_lowpass + out.prev_highpass) / np.sqrt(2.0)
    recovered_previous = (prev_lowpass - out.prev_highpass) / np.sqrt(2.0)

    np.testing.assert_allclose(recovered_current, out.normalized_lf[0:1])
    np.testing.assert_allclose(recovered_previous, out.normalized_lf[1:2])
    assert out.shape_metadata.source_frame_shape == (2, 4, 4)
    assert out.shape_metadata.temporal_encoder_input_shape == (1, 2, 2, 2)


def test_nchw_batch_one_inputs_match_chw_inputs() -> None:
    current = np.arange(16, dtype=np.float64).reshape(1, 4, 4)
    previous = current + 1.5
    next_frame = current - 2.5

    chw = prepare_official_tub_graph_inputs(current, previous, next_frame)
    nchw = prepare_official_tub_graph_inputs(
        current[np.newaxis, :, :, :],
        previous[np.newaxis, :, :, :],
        next_frame[np.newaxis, :, :, :],
    )

    np.testing.assert_allclose(nchw.normalized_lf, chw.normalized_lf)
    np.testing.assert_allclose(nchw.prev_lowpass_over_2, chw.prev_lowpass_over_2)
    assert nchw.as_jsonable_metadata()["promotion_eligible"] is False


def test_official_tub_graph_inputs_refuse_source_forward_overclaim() -> None:
    current = np.arange(16, dtype=np.float64).reshape(1, 4, 4)
    out = prepare_official_tub_graph_inputs(current, current + 1.0, current + 2.0)

    with pytest.raises(OfficialTubError, match="not full source-forward parity"):
        replace(out, source_forward_parity_proven=True)
    with pytest.raises(OfficialTubError, match="not full source-forward parity"):
        replace(out, full_tub_source_forward_parity_proven=True)
    with pytest.raises(OfficialTubError, match="not full source-forward parity"):
        replace(out, source_forward_replay_authority=True)
    with pytest.raises(OfficialTubError, match="unexpected source equivalence scope"):
        replace(out, source_equivalence_scope="full_source_forward_parity")
    with pytest.raises(OfficialTubError, match="must preserve source-forward blockers"):
        replace(
            out,
            source_forward_blockers=(
                "snerv_official_tub_encoder_decoder_weights_not_loaded",
            ),
        )


def test_official_tub_numpy_matches_torch_haar_and_temporal_algebra() -> None:
    torch = pytest.importorskip("torch")

    rng = np.random.default_rng(4)
    current = rng.standard_normal((2, 6, 8))
    previous = current * 0.75 + 0.5
    next_frame = current * -0.25 - 0.125

    out = prepare_official_tub_graph_inputs(current, previous, next_frame)

    frames = torch.tensor(
        np.stack([current, previous, next_frame], axis=0),
        dtype=torch.float64,
    )
    torch_lf = (
        frames[:, :, 0::2, 0::2]
        + frames[:, :, 0::2, 1::2]
        + frames[:, :, 1::2, 0::2]
        + frames[:, :, 1::2, 1::2]
    ) * 0.5
    torch_norm = (torch_lf - torch_lf.min()) / (torch_lf.max() - torch_lf.min())
    torch_prev_lowpass_over_2 = (torch_norm[0:1] + torch_norm[1:2]) / (
        2.0 * torch.sqrt(torch.tensor(2.0, dtype=torch.float64))
    )
    torch_next_lowpass_over_2 = (torch_norm[0:1] + torch_norm[2:3]) / (
        2.0 * torch.sqrt(torch.tensor(2.0, dtype=torch.float64))
    )

    np.testing.assert_allclose(out.lf_triplet, torch_lf.numpy(), atol=1e-12)
    np.testing.assert_allclose(out.normalized_lf, torch_norm.numpy(), atol=1e-12)
    np.testing.assert_allclose(
        out.prev_lowpass_over_2,
        torch_prev_lowpass_over_2.numpy(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        out.next_lowpass_over_2,
        torch_next_lowpass_over_2.numpy(),
        atol=1e-12,
    )


def test_official_tub_output2_fusion_shape_matches_source_split_concat_shuffle() -> None:
    shape = official_output2_fusion_shape(
        (1, 12, 4, 5),
        fc_hw=(2, 3),
        decoder_output_shape=(2, 18, 4, 5),
    )

    assert shape.emb_ch == 6
    assert shape.prev_half_shape == (1, 6, 4, 5)
    assert shape.next_half_shape == (1, 6, 4, 5)
    assert shape.decoder_input_shape == (2, 6, 4, 5)
    assert shape.fused_output2_shape == (2, 3, 8, 15)
    assert shape.as_jsonable()["fused_output2_shape"] == [2, 3, 8, 15]


def test_official_tub_output2_fusion_numpy_matches_source_split_concat_shuffle() -> None:
    temporal = np.arange(1 * 12 * 2 * 3, dtype=np.float64).reshape(1, 12, 2, 3)
    decoder_output = np.arange(2 * 18 * 2 * 3, dtype=np.float64).reshape(2, 18, 2, 3)

    out = official_output2_fusion_numpy(temporal, decoder_output, fc_hw=(2, 3))

    expected_decoder_input = np.concatenate([temporal[:, :6], temporal[:, 6:]], axis=0)
    expected_shuffled = (
        decoder_output.reshape(2, -1, 2, 3, 2, 3)
        .transpose(0, 1, 4, 2, 5, 3)
        .reshape(2, -1, 4, 9)
    )
    np.testing.assert_array_equal(out.prev_half, temporal[:, :6])
    np.testing.assert_array_equal(out.next_half, temporal[:, 6:])
    np.testing.assert_array_equal(out.decoder_input, expected_decoder_input)
    np.testing.assert_array_equal(out.output2_fused, expected_shuffled)
    metadata = out.as_jsonable_metadata()
    assert metadata["score_claim"] is False
    assert metadata["promotion_eligible"] is False
    assert metadata["ready_for_exact_eval_dispatch"] is False
    assert metadata["portable_output2_fusion_primitive_proven"] is True
    assert metadata["shape"]["decoder_input_shape"] == [2, 6, 2, 3]
    assert metadata["shape"]["fused_output2_shape"] == [2, 3, 4, 9]


def test_official_tub_frame_reconstruction_numpy_matches_hfr_idwt_source_path() -> None:
    img_yl = np.array([[[[0.25, 0.75], [0.5, 1.0]]]], dtype=np.float64)
    yl_norm = (2.0, 10.0)
    lh = np.array([[[[1.0, -2.0], [0.5, 1.5]]]], dtype=np.float64)
    hl = np.array([[[[0.25, 0.75], [-1.0, 2.0]]]], dtype=np.float64)
    hh = np.array([[[[-0.5, 1.0], [0.0, -1.5]]]], dtype=np.float64)
    yh_out = np.stack([lh, hl, hh], axis=2)

    out = official_tub_mod.official_tub_frame_reconstruction_numpy(
        img_yl,
        yh_out,
        yl_norm=yl_norm,
    )

    yl_out = img_yl * (yl_norm[1] - yl_norm[0]) + yl_norm[0]
    expected = np.empty((1, 1, 4, 4), dtype=np.float64)
    expected[:, :, 0::2, 0::2] = (yl_out + lh + hl + hh) * 0.5
    expected[:, :, 0::2, 1::2] = (yl_out + lh - hl - hh) * 0.5
    expected[:, :, 1::2, 0::2] = (yl_out - lh + hl - hh) * 0.5
    expected[:, :, 1::2, 1::2] = (yl_out - lh - hl + hh) * 0.5

    np.testing.assert_array_equal(out.yl_out, yl_out)
    np.testing.assert_array_equal(out.frame, expected)
    metadata = out.as_jsonable_metadata()
    assert metadata["source_contract"] == (
        official_tub_mod.OFFICIAL_SNERV_T_TUB_FRAME_RECONSTRUCTION_CONTRACT
    )
    assert metadata["portable_frame_reconstruction_primitive_proven"] is True
    assert metadata["source_forward_parity_proven"] is False
    assert metadata["full_tub_source_forward_parity_proven"] is False
    assert metadata["source_forward_replay_authority"] is False
    assert metadata["score_claim"] is False
    assert metadata["ready_for_exact_eval_dispatch"] is False
    assert metadata["output_shapes"]["frame"] == [1, 1, 4, 4]
    with pytest.raises(OfficialTubError, match="not score"):
        replace(out, score_claim=True)
    with pytest.raises(OfficialTubError, match="not score"):
        replace(out, source_forward_replay_authority=True)


def test_official_tub_output2_fusion_torch_matches_numpy_reference() -> None:
    torch = pytest.importorskip("torch")
    temporal = np.arange(1 * 12 * 2 * 3, dtype=np.float64).reshape(1, 12, 2, 3)
    decoder_output = np.arange(2 * 18 * 2 * 3, dtype=np.float64).reshape(2, 18, 2, 3)
    reference = official_output2_fusion_numpy(temporal, decoder_output, fc_hw=(2, 3))

    decoder_input, fused = official_output2_fusion_torch(
        torch.tensor(temporal, dtype=torch.float64),
        torch.tensor(decoder_output, dtype=torch.float64),
        fc_hw=(2, 3),
    )

    np.testing.assert_array_equal(decoder_input.detach().cpu().numpy(), reference.decoder_input)
    np.testing.assert_array_equal(fused.detach().cpu().numpy(), reference.output2_fused)


def test_official_tub_output2_fusion_mlx_matches_numpy_reference() -> None:
    mx = pytest.importorskip("mlx.core")
    temporal = np.arange(1 * 12 * 2 * 3, dtype=np.float32).reshape(1, 12, 2, 3)
    decoder_output = np.arange(2 * 18 * 2 * 3, dtype=np.float32).reshape(2, 18, 2, 3)
    reference = official_output2_fusion_numpy(temporal, decoder_output, fc_hw=(2, 3))

    decoder_input, fused = official_output2_fusion_mlx(
        mx.array(temporal),
        mx.array(decoder_output),
        fc_hw=(2, 3),
    )

    np.testing.assert_array_equal(np.asarray(decoder_input), reference.decoder_input)
    np.testing.assert_array_equal(np.asarray(fused), reference.output2_fused)


def test_official_tub_backend_bindings_share_one_fusion_core() -> None:
    assert "_apply_output2_fusion_backend" in official_output2_fusion_numpy.__code__.co_names
    assert (
        "_apply_output2_fusion_backend"
        in official_tub_mod.official_output2_fusion_mlx.__code__.co_names
    )
    assert (
        "_apply_output2_fusion_backend"
        in official_tub_torch_mod.official_output2_fusion_torch.__code__.co_names
    )


def test_prepare_can_attach_output2_shape_metadata() -> None:
    current = np.arange(16, dtype=np.float64).reshape(4, 4)
    out = prepare_official_tub_graph_inputs(
        current,
        current + 2.0,
        current - 3.0,
        temporal_encoder_output_shape=(1, 10, 2, 2),
        fc_hw=(2, 2),
        output2_decoder_output_shape=(2, 20, 3, 4),
    )

    output2 = out.shape_metadata.output2_fusion
    assert output2 is not None
    assert output2.emb_ch == 5
    assert output2.decoder_input_shape == (2, 5, 2, 2)
    assert output2.fused_output2_shape == (2, 5, 6, 8)


def test_official_tub_rejects_non_source_inputs() -> None:
    frame = np.zeros((1, 3, 4), dtype=np.float64)
    with pytest.raises(OfficialTubError, match="spatial dims must be even"):
        prepare_official_tub_graph_inputs(frame, frame, frame)
    with pytest.raises(OfficialTubError, match="requires non-constant LF"):
        prepare_official_tub_graph_inputs(
            np.zeros((1, 4, 4)),
            np.zeros((1, 4, 4)),
            np.zeros((1, 4, 4)),
        )
    with pytest.raises(OfficialTubError, match="even temporal channels"):
        official_output2_fusion_shape((1, 5, 2, 2))
    with pytest.raises(OfficialTubError, match="even temporal channels"):
        official_output2_fusion_numpy(
            np.zeros((1, 5, 2, 2)),
            np.zeros((2, 8, 2, 2)),
            fc_hw=(2, 2),
        )
    with pytest.raises(OfficialTubError, match="batch must be 2"):
        official_output2_fusion_numpy(
            np.zeros((1, 8, 2, 2)),
            np.zeros((1, 8, 2, 2)),
            fc_hw=(2, 2),
        )
    with pytest.raises(OfficialTubError, match="requires fc_hw"):
        official_output2_fusion_shape(
            (1, 8, 3, 5),
            decoder_output_shape=(2, 24, 4, 6),
        )
    with pytest.raises(OfficialTubError, match="divisible"):
        official_output2_fusion_shape(
            (1, 8, 3, 5),
            fc_hw=(2, 3),
            decoder_output_shape=(2, 25, 4, 6),
        )
    with pytest.raises(OfficialTubError, match="contains NaN or Inf"):
        official_output2_fusion_numpy(
            np.full((1, 8, 2, 2), np.nan),
            np.zeros((2, 8, 2, 2)),
            fc_hw=(2, 2),
        )
    with pytest.raises(OfficialTubError, match="shaped"):
        official_tub_mod.official_tub_frame_reconstruction_numpy(
            np.zeros((1, 1, 2, 2)),
            np.zeros((1, 1, 2, 2)),
            yl_norm=(0.0, 1.0),
        )
    with pytest.raises(OfficialTubError, match="exactly 3 HFR bands"):
        official_tub_mod.official_tub_frame_reconstruction_numpy(
            np.zeros((1, 1, 2, 2)),
            np.zeros((1, 1, 2, 2, 2)),
            yl_norm=(0.0, 1.0),
        )
    with pytest.raises(OfficialTubError, match="finite increasing"):
        official_tub_mod.official_tub_frame_reconstruction_numpy(
            np.zeros((1, 1, 2, 2)),
            np.zeros((1, 1, 3, 2, 2)),
            yl_norm=(1.0, 1.0),
        )
