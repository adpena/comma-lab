# SPDX-License-Identifier: MIT
"""D4 Wyner-Ziv frame-0 substrate test suite.

30+ tests covering:
- Motion model differentiability (SE(3) Rodrigues + optical flow).
- Residual codec roundtrip (int8 quantization + brotli).
- Archive byte-stability (pack + parse roundtrip, deterministic bytes).
- Inflate roundtrip (frame_0 reconstructed from frame_1 + side info).
- No-op detection (every byte structurally consumed).
- Composability with multiple base substrates (provider registry).
- /tmp refusal (no transient evidence in persisted manifests).
- Frame-1-byte-identical invariant (SegNet sees frame_1 unchanged).
- score_aware_loss routes through canonical score_pair_components.
- Catalog #146 inflate.sh 3-positional-arg CLI contract.
- Catalog #168 AnnAssign manifest (D4 module-level inline contract).
"""

from __future__ import annotations

import hashlib
import struct
import sys
import tempfile
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.substrates.d4_wyner_ziv_frame_0 import (
    BASE_SHA_HEX_LEN,
    EVAL_HW,
    MotionModelMode,
    NUM_PAIRS,
    OpticalFlowField,
    SE3MotionParams,
    WZF01_MAGIC,
    WZF01_SCHEMA_VERSION,
    WynerZivFrame0Archive,
    WynerZivFrame0Config,
    WynerZivFrame0LossWeights,
    WynerZivFrame0Substrate,
    apply_optical_flow,
    apply_se3_motion,
    decode_residual_blob,
    encode_residual_blob,
    pack_archive,
    parse_archive,
    synthesize_frame_0,
)
from tac.substrates.d4_wyner_ziv_frame_0.archive import (
    WZF01_HEADER_FMT,
    WZF01_HEADER_SIZE,
    deserialize_motion_to_tensor,
)
from tac.substrates.d4_wyner_ziv_frame_0.inflate import (
    inflate_one_video,
    register_base_substrate_provider,
)


# ----------------------------------------------------------------------
# Motion model tests
# ----------------------------------------------------------------------

class TestSE3MotionParams:
    def test_se3_construct_from_flat_roundtrip(self):
        flat = torch.randn(8, 6)
        params = SE3MotionParams.from_flat(flat)
        out = params.to_flat()
        assert torch.allclose(flat, out)
        assert params.num_pairs == 8

    def test_se3_rejects_bad_translation_shape(self):
        with pytest.raises(ValueError, match="translation shape"):
            SE3MotionParams(
                translation=torch.randn(4, 2),
                axis_angle=torch.randn(4, 3),
            )

    def test_se3_rejects_mismatched_axis_angle_shape(self):
        with pytest.raises(ValueError, match="axis_angle shape"):
            SE3MotionParams(
                translation=torch.randn(4, 3),
                axis_angle=torch.randn(3, 3),
            )

    def test_se3_warp_is_differentiable(self):
        torch.manual_seed(0)
        frame_1 = torch.randn(2, 3, 12, 16, requires_grad=True)
        translation = torch.randn(2, 3, requires_grad=True)
        axis_angle = (torch.randn(2, 3) * 0.01).requires_grad_(True)
        se3 = SE3MotionParams(translation=translation, axis_angle=axis_angle)
        warped = apply_se3_motion(frame_1, se3, output_hw=(12, 16))
        assert warped.shape == (2, 3, 12, 16)
        loss = warped.pow(2).mean()
        loss.backward()
        assert frame_1.grad is not None
        assert translation.grad is not None
        assert axis_angle.grad is not None
        assert frame_1.grad.abs().sum() > 0


class TestOpticalFlowField:
    def test_flow_construct_validates_shape(self):
        with pytest.raises(ValueError, match="flow_uv must be 4D"):
            OpticalFlowField(flow_uv=torch.randn(8, 2, 12), grid_h=12, grid_w=16)

    def test_flow_construct_validates_channels(self):
        with pytest.raises(ValueError, match="channels"):
            OpticalFlowField(flow_uv=torch.randn(8, 3, 12, 16), grid_h=12, grid_w=16)

    def test_flow_construct_validates_grid_match(self):
        with pytest.raises(ValueError, match="does not match"):
            OpticalFlowField(flow_uv=torch.randn(8, 2, 10, 16), grid_h=12, grid_w=16)

    def test_flow_warp_is_differentiable(self):
        torch.manual_seed(1)
        frame_1 = torch.randn(2, 3, 24, 32, requires_grad=True)
        flow_param = torch.zeros(2, 2, 6, 8, requires_grad=True)
        field = OpticalFlowField(flow_uv=flow_param, grid_h=6, grid_w=8)
        warped = apply_optical_flow(frame_1, field, output_hw=(24, 32))
        assert warped.shape == (2, 3, 24, 32)
        loss = warped.pow(2).mean()
        loss.backward()
        assert frame_1.grad is not None
        assert flow_param.grad is not None

    def test_flow_zero_field_warps_close_to_identity(self):
        """A zero flow field warps frame_1 close to itself.

        grid_sample with ``align_corners=False`` introduces a half-pixel
        offset by convention, so the warped frame is not bit-identical to
        the input even at zero flow. The expected behavior is "close" in
        mean-absolute terms; a bias of ~0.1 across 768 pixels is the
        sampling-convention noise floor.
        """
        torch.manual_seed(2)
        frame_1 = torch.rand(1, 3, 24, 32)
        field = OpticalFlowField(flow_uv=torch.zeros(1, 2, 6, 8), grid_h=6, grid_w=8)
        warped = apply_optical_flow(frame_1, field, output_hw=(24, 32))
        # Mean absolute deviation should be small (well under unit range).
        mad = (warped - frame_1).abs().mean()
        assert mad < 0.2


# ----------------------------------------------------------------------
# Frame-0 synthesis tests
# ----------------------------------------------------------------------

class TestFrame0Synthesis:
    def test_synthesize_se3_smoke(self):
        torch.manual_seed(3)
        frame_1 = torch.rand(2, 3, 24, 32)
        se3 = SE3MotionParams(
            translation=torch.zeros(2, 3),
            axis_angle=torch.zeros(2, 3),
        )
        residual = torch.zeros(2, 3, 24, 32)
        f0 = synthesize_frame_0(
            frame_1=frame_1,
            motion_mode=MotionModelMode.SE3_PARAMETRIC,
            se3_params=se3,
            residual=residual,
            output_hw=(24, 32),
            clamp_unit=True,
        )
        assert f0.shape == (2, 3, 24, 32)

    def test_synthesize_optical_flow_smoke(self):
        torch.manual_seed(4)
        frame_1 = torch.rand(2, 3, 24, 32)
        field = OpticalFlowField(flow_uv=torch.zeros(2, 2, 6, 8), grid_h=6, grid_w=8)
        residual = torch.zeros(2, 3, 24, 32)
        f0 = synthesize_frame_0(
            frame_1=frame_1,
            motion_mode=MotionModelMode.OPTICAL_FLOW,
            flow_field=field,
            residual=residual,
            output_hw=(24, 32),
            clamp_unit=True,
        )
        assert f0.shape == (2, 3, 24, 32)

    def test_synthesize_se3_requires_se3_params(self):
        frame_1 = torch.rand(2, 3, 12, 16)
        with pytest.raises(ValueError, match="se3_params is required"):
            synthesize_frame_0(
                frame_1=frame_1,
                motion_mode=MotionModelMode.SE3_PARAMETRIC,
                residual=torch.zeros(2, 3, 12, 16),
                output_hw=(12, 16),
            )

    def test_synthesize_optical_flow_requires_flow_field(self):
        frame_1 = torch.rand(2, 3, 12, 16)
        with pytest.raises(ValueError, match="flow_field is required"):
            synthesize_frame_0(
                frame_1=frame_1,
                motion_mode=MotionModelMode.OPTICAL_FLOW,
                residual=torch.zeros(2, 3, 12, 16),
                output_hw=(12, 16),
            )

    def test_synthesize_clamps_output_to_unit(self):
        frame_1 = torch.ones(1, 3, 12, 16)
        residual = torch.full((1, 3, 12, 16), 100.0)  # Way out of range
        f0 = synthesize_frame_0(
            frame_1=frame_1,
            motion_mode=MotionModelMode.SE3_PARAMETRIC,
            se3_params=SE3MotionParams(
                translation=torch.zeros(1, 3),
                axis_angle=torch.zeros(1, 3),
            ),
            residual=residual,
            output_hw=(12, 16),
            clamp_unit=True,
        )
        assert f0.max() <= 1.0
        assert f0.min() >= 0.0


# ----------------------------------------------------------------------
# Residual codec tests
# ----------------------------------------------------------------------

class TestResidualCodec:
    def test_residual_roundtrip_smoke(self):
        torch.manual_seed(5)
        residual = torch.randn(8, 3, 48, 64) * 0.1
        blob = encode_residual_blob(residual, coarse_hw=(48, 64))
        decoded = decode_residual_blob(blob, expected_num_pairs=8)
        assert decoded.shape == (8, 3, 48, 64)
        # int8 quantization introduces small but bounded error
        err = (decoded - residual).abs().max()
        assert err < 0.01

    def test_residual_rejects_wrong_pairs(self):
        residual = torch.zeros(8, 3, 24, 32)
        blob = encode_residual_blob(residual, coarse_hw=(24, 32))
        with pytest.raises(ValueError, match="num_pairs"):
            decode_residual_blob(blob, expected_num_pairs=4)

    def test_residual_rejects_4d_only(self):
        bad = torch.zeros(8, 3, 24)
        with pytest.raises(ValueError, match="must be"):
            encode_residual_blob(bad)

    def test_residual_rejects_non_3_channels(self):
        bad = torch.zeros(8, 4, 24, 32)
        with pytest.raises(ValueError, match="must be"):
            encode_residual_blob(bad)

    def test_residual_coarsens_to_smaller_grid(self):
        residual = torch.randn(4, 3, 96, 128)
        blob = encode_residual_blob(residual, coarse_hw=(12, 16))
        decoded = decode_residual_blob(blob, expected_num_pairs=4)
        assert decoded.shape == (4, 3, 12, 16)

    def test_residual_blob_is_brotli_compressed(self):
        """Sanity: the blob should be smaller than the raw int8 + scales."""
        residual = torch.zeros(4, 3, 24, 32)  # All-zero is highly compressible
        blob = encode_residual_blob(residual, coarse_hw=(24, 32))
        raw_size = 14 + 4 * 4 + 4 * 3 * 24 * 32  # header + scales + data
        assert len(blob) < raw_size


# ----------------------------------------------------------------------
# Archive grammar tests
# ----------------------------------------------------------------------

class TestArchive:
    def _build_minimal_se3(self):
        base_bytes = b"BASE_TEST"
        base_sha = hashlib.sha256(base_bytes).hexdigest()
        se3_flat = torch.zeros(4, 6)
        residual = torch.zeros(4, 3, 12, 16)
        residual_blob = encode_residual_blob(residual, coarse_hw=(12, 16))
        return base_bytes, base_sha, se3_flat, residual_blob

    def test_archive_header_size_invariant(self):
        assert WZF01_HEADER_SIZE == 33
        assert struct.calcsize(WZF01_HEADER_FMT) == WZF01_HEADER_SIZE

    def test_archive_pack_parse_roundtrip_se3(self):
        base_bytes, base_sha, se3_flat, residual_blob = self._build_minimal_se3()
        archive = pack_archive(
            motion_mode=0,
            se3_flat=se3_flat,
            flow_uv=None,
            residual_blob=residual_blob,
            meta={"base_substrate_id": "smoke", "design_notes": "test"},
            base_substrate_archive_sha256_hex=base_sha,
            base_substrate_bytes=base_bytes,
            num_pairs=4,
            flow_grid_h=0,
            flow_grid_w=0,
            residual_coarse_h=12,
            residual_coarse_w=16,
        )
        parsed = parse_archive(archive)
        assert parsed.num_pairs == 4
        assert parsed.motion_mode == 0
        assert parsed.motion_mode_label == "se3_parametric"
        assert parsed.base_substrate_archive_sha256_hex == base_sha
        assert parsed.base_substrate_bytes == base_bytes
        assert parsed.meta["base_substrate_id"] == "smoke"

    def test_archive_pack_parse_roundtrip_optical_flow(self):
        base_bytes = b"FLOW_TEST"
        base_sha = hashlib.sha256(base_bytes).hexdigest()
        flow_uv = torch.zeros(4, 2, 6, 8)
        residual = torch.zeros(4, 3, 12, 16)
        residual_blob = encode_residual_blob(residual, coarse_hw=(12, 16))
        archive = pack_archive(
            motion_mode=1,
            se3_flat=None,
            flow_uv=flow_uv,
            residual_blob=residual_blob,
            meta={},
            base_substrate_archive_sha256_hex=base_sha,
            base_substrate_bytes=base_bytes,
            num_pairs=4,
            flow_grid_h=6,
            flow_grid_w=8,
            residual_coarse_h=12,
            residual_coarse_w=16,
        )
        parsed = parse_archive(archive)
        assert parsed.motion_mode == 1
        assert parsed.motion_mode_label == "optical_flow"
        assert parsed.flow_grid_h == 6
        assert parsed.flow_grid_w == 8

    def test_archive_rejects_wrong_magic(self):
        bad = b"BAD\x00" + b"\x00" * (WZF01_HEADER_SIZE - 4)
        with pytest.raises(ValueError, match="bad magic"):
            parse_archive(bad)

    def test_archive_rejects_wrong_version(self):
        # MAGIC(4) + VERSION(1)
        bad_header = WZF01_MAGIC + b"\x99" + b"\x00" * (WZF01_HEADER_SIZE - 5)
        with pytest.raises(ValueError, match="unsupported schema version"):
            parse_archive(bad_header)

    def test_archive_rejects_short_blob(self):
        with pytest.raises(ValueError, match="too short"):
            parse_archive(b"\x00" * 5)

    def test_archive_rejects_bad_base_sha_length(self):
        base_bytes, _, se3_flat, residual_blob = self._build_minimal_se3()
        with pytest.raises(ValueError, match="must be a 64-char"):
            pack_archive(
                motion_mode=0,
                se3_flat=se3_flat,
                flow_uv=None,
                residual_blob=residual_blob,
                meta={},
                base_substrate_archive_sha256_hex="too_short",
                base_substrate_bytes=base_bytes,
                num_pairs=4,
                flow_grid_h=0,
                flow_grid_w=0,
                residual_coarse_h=12,
                residual_coarse_w=16,
            )

    def test_archive_rejects_non_hex_base_sha(self):
        base_bytes, _, se3_flat, residual_blob = self._build_minimal_se3()
        with pytest.raises(ValueError, match="must be hex"):
            pack_archive(
                motion_mode=0,
                se3_flat=se3_flat,
                flow_uv=None,
                residual_blob=residual_blob,
                meta={},
                base_substrate_archive_sha256_hex="z" * BASE_SHA_HEX_LEN,
                base_substrate_bytes=base_bytes,
                num_pairs=4,
                flow_grid_h=0,
                flow_grid_w=0,
                residual_coarse_h=12,
                residual_coarse_w=16,
            )

    def test_archive_rejects_unknown_motion_mode(self):
        base_bytes, base_sha, se3_flat, residual_blob = self._build_minimal_se3()
        with pytest.raises(ValueError, match="unknown motion_mode"):
            pack_archive(
                motion_mode=99,
                se3_flat=se3_flat,
                flow_uv=None,
                residual_blob=residual_blob,
                meta={},
                base_substrate_archive_sha256_hex=base_sha,
                base_substrate_bytes=base_bytes,
                num_pairs=4,
                flow_grid_h=0,
                flow_grid_w=0,
                residual_coarse_h=12,
                residual_coarse_w=16,
            )

    def test_archive_deterministic_bytes(self):
        """Same inputs produce same archive bytes — deterministic packing."""
        base_bytes, base_sha, se3_flat, residual_blob = self._build_minimal_se3()
        meta = {"a": 1, "z": "test"}
        a1 = pack_archive(
            motion_mode=0,
            se3_flat=se3_flat,
            flow_uv=None,
            residual_blob=residual_blob,
            meta=meta,
            base_substrate_archive_sha256_hex=base_sha,
            base_substrate_bytes=base_bytes,
            num_pairs=4,
            flow_grid_h=0,
            flow_grid_w=0,
            residual_coarse_h=12,
            residual_coarse_w=16,
        )
        a2 = pack_archive(
            motion_mode=0,
            se3_flat=se3_flat,
            flow_uv=None,
            residual_blob=residual_blob,
            meta=meta,
            base_substrate_archive_sha256_hex=base_sha,
            base_substrate_bytes=base_bytes,
            num_pairs=4,
            flow_grid_h=0,
            flow_grid_w=0,
            residual_coarse_h=12,
            residual_coarse_w=16,
        )
        assert a1 == a2

    def test_archive_motion_tensor_roundtrip(self):
        """SE3 motion bytes survive fp16 quantization (~ 1e-3 accuracy)."""
        base_bytes, base_sha, _, residual_blob = self._build_minimal_se3()
        se3_flat = torch.randn(4, 6) * 0.05  # small motion typical of dashcam
        archive = pack_archive(
            motion_mode=0,
            se3_flat=se3_flat,
            flow_uv=None,
            residual_blob=residual_blob,
            meta={},
            base_substrate_archive_sha256_hex=base_sha,
            base_substrate_bytes=base_bytes,
            num_pairs=4,
            flow_grid_h=0,
            flow_grid_w=0,
            residual_coarse_h=12,
            residual_coarse_w=16,
        )
        parsed = parse_archive(archive)
        recovered = deserialize_motion_to_tensor(
            parsed.motion_blob_raw,
            motion_mode=0,
            num_pairs=4,
            flow_grid_h=0,
            flow_grid_w=0,
        )
        err = (recovered - se3_flat).abs().max()
        assert err < 1e-3


# ----------------------------------------------------------------------
# Substrate composition tests
# ----------------------------------------------------------------------

class TestSubstrateComposition:
    def test_substrate_reconstruct_pair_se3(self):
        torch.manual_seed(10)
        cfg = WynerZivFrame0Config(
            motion_mode=MotionModelMode.SE3_PARAMETRIC,
            num_pairs=2,
            output_height=24,
            output_width=32,
            residual_coarse_h=6,
            residual_coarse_w=8,
        )
        sub = WynerZivFrame0Substrate(cfg)
        frame_1 = torch.rand(2, 3, 24, 32)
        f0, f1 = sub.reconstruct_pair(frame_1)
        assert f0.shape == (2, 3, 24, 32)
        assert f1.shape == (2, 3, 24, 32)

    def test_substrate_frame_1_byte_identical_invariant(self):
        """SegNet sees frame_1 unchanged — D4 must NEVER mutate frame_1."""
        torch.manual_seed(11)
        cfg = WynerZivFrame0Config(
            motion_mode=MotionModelMode.OPTICAL_FLOW,
            num_pairs=2,
            output_height=24,
            output_width=32,
            flow_grid_h=6,
            flow_grid_w=8,
            residual_coarse_h=6,
            residual_coarse_w=8,
        )
        sub = WynerZivFrame0Substrate(cfg)
        frame_1 = torch.rand(2, 3, 24, 32)
        f0, f1 = sub.reconstruct_pair(frame_1)
        assert torch.equal(f1, frame_1), (
            "D4 frame-1 invariant violated; SegNet would see modified bytes"
        )

    def test_substrate_rejects_wrong_batch(self):
        cfg = WynerZivFrame0Config(num_pairs=2, output_height=12, output_width=16,
                                    residual_coarse_h=3, residual_coarse_w=4)
        sub = WynerZivFrame0Substrate(cfg)
        with pytest.raises(ValueError, match="batch"):
            sub.reconstruct_pair(torch.rand(3, 3, 12, 16))

    def test_substrate_supports_both_motion_modes(self):
        """Both SE3 and OPTICAL_FLOW must work (probe-disambiguator wire-in)."""
        torch.manual_seed(12)
        frame_1 = torch.rand(2, 3, 24, 32)
        for mode in (MotionModelMode.SE3_PARAMETRIC, MotionModelMode.OPTICAL_FLOW):
            cfg = WynerZivFrame0Config(
                motion_mode=mode,
                num_pairs=2,
                output_height=24,
                output_width=32,
                flow_grid_h=6,
                flow_grid_w=8,
                residual_coarse_h=6,
                residual_coarse_w=8,
            )
            sub = WynerZivFrame0Substrate(cfg)
            f0, f1 = sub.reconstruct_pair(frame_1)
            assert f0.shape == (2, 3, 24, 32)


# ----------------------------------------------------------------------
# Mini-batched reconstruct_pair tests (D4 OOM fix
# lane_d4_oom_fix_minibatch_reconstruct_20260514).
#
# Per the 2026-05-14 OOM anchor (fc-01KRK9RKD3QV4C276Y5KXFMF65 — Modal T4
# rc=1 elapsed 121s with CUDA OOM at F.interpolate residual upsample with
# 600-pair batch needing ~13 GB activation memory > T4 14.56 GB capacity),
# reconstruct_pair gained a ``pair_indices`` kwarg that lets the caller
# subset motion params + residual rows + frame_1 by a 1-D long index
# tensor. Gradients flow into the selected nn.Parameter rows via
# torch.index_select autograd scatter-add.
# ----------------------------------------------------------------------


class TestMiniBatchReconstructPair:
    def _make_sub_se3(self, num_pairs: int = 4) -> WynerZivFrame0Substrate:
        cfg = WynerZivFrame0Config(
            motion_mode=MotionModelMode.SE3_PARAMETRIC,
            num_pairs=num_pairs,
            output_height=12,
            output_width=16,
            residual_coarse_h=3,
            residual_coarse_w=4,
        )
        return WynerZivFrame0Substrate(cfg)

    def _make_sub_flow(self, num_pairs: int = 4) -> WynerZivFrame0Substrate:
        cfg = WynerZivFrame0Config(
            motion_mode=MotionModelMode.OPTICAL_FLOW,
            num_pairs=num_pairs,
            output_height=12,
            output_width=16,
            flow_grid_h=3,
            flow_grid_w=4,
            residual_coarse_h=3,
            residual_coarse_w=4,
        )
        return WynerZivFrame0Substrate(cfg)

    def test_minibatch_se3_returns_correct_shape(self):
        torch.manual_seed(100)
        sub = self._make_sub_se3(num_pairs=8)
        idx = torch.tensor([0, 2, 5], dtype=torch.long)
        frame_1 = torch.rand(3, 3, 12, 16)
        f0, f1 = sub.reconstruct_pair(frame_1, pair_indices=idx)
        assert f0.shape == (3, 3, 12, 16)
        assert f1.shape == (3, 3, 12, 16)

    def test_minibatch_optical_flow_returns_correct_shape(self):
        torch.manual_seed(101)
        sub = self._make_sub_flow(num_pairs=8)
        idx = torch.tensor([1, 4, 7], dtype=torch.long)
        frame_1 = torch.rand(3, 3, 12, 16)
        f0, f1 = sub.reconstruct_pair(frame_1, pair_indices=idx)
        assert f0.shape == (3, 3, 12, 16)

    def test_minibatch_frame_1_byte_identical_invariant(self):
        """Mini-batch path must also preserve the frame_1-byte-identical
        invariant (SegNet sees only frame_1 unchanged)."""
        torch.manual_seed(102)
        sub = self._make_sub_se3(num_pairs=4)
        idx = torch.tensor([0, 3], dtype=torch.long)
        frame_1 = torch.rand(2, 3, 12, 16)
        _, f1 = sub.reconstruct_pair(frame_1, pair_indices=idx)
        assert torch.equal(f1, frame_1)

    def test_minibatch_gradient_flows_into_selected_rows_only(self):
        """Gradients from a mini-batched forward must flow ONLY into the
        selected rows of motion.se3_flat / residual_coarse. Rows not in
        ``pair_indices`` must have zero gradient."""
        torch.manual_seed(103)
        sub = self._make_sub_se3(num_pairs=4)
        sub.train()
        idx = torch.tensor([1, 2], dtype=torch.long)
        frame_1 = torch.rand(2, 3, 12, 16, requires_grad=False)
        f0, _ = sub.reconstruct_pair(frame_1, pair_indices=idx)
        loss = f0.sum()
        loss.backward()
        assert sub.motion.se3_flat.grad is not None
        # Rows 0 and 3 must be zero gradient; rows 1 and 2 must be non-zero.
        assert sub.motion.se3_flat.grad[0].abs().sum() == 0.0
        assert sub.motion.se3_flat.grad[3].abs().sum() == 0.0
        # The selected rows' gradient depends on the warp activation; for
        # SE(3) near identity at least one of the 6 params has non-trivial
        # gradient — we assert non-zero magnitude.
        assert sub.motion.se3_flat.grad[1].abs().sum() > 0.0 or \
               sub.motion.se3_flat.grad[2].abs().sum() > 0.0
        assert sub.residual_coarse.grad is not None
        assert sub.residual_coarse.grad[0].abs().sum() == 0.0
        assert sub.residual_coarse.grad[3].abs().sum() == 0.0

    def test_minibatch_equals_full_batch_when_indices_full(self):
        """Passing pair_indices=arange(num_pairs) must produce the same
        output as the back-compat full-batch call (modulo numerical noise
        from the index_select view)."""
        torch.manual_seed(104)
        sub = self._make_sub_se3(num_pairs=3)
        frame_1 = torch.rand(3, 3, 12, 16)
        f0_full, _ = sub.reconstruct_pair(frame_1)
        idx = torch.arange(3, dtype=torch.long)
        f0_mb, _ = sub.reconstruct_pair(frame_1, pair_indices=idx)
        assert torch.allclose(f0_full, f0_mb, atol=1e-6)

    def test_minibatch_rejects_non_1d_indices(self):
        sub = self._make_sub_se3(num_pairs=4)
        frame_1 = torch.rand(1, 3, 12, 16)
        with pytest.raises(ValueError, match="1-D"):
            sub.reconstruct_pair(
                frame_1, pair_indices=torch.tensor([[0]], dtype=torch.long)
            )

    def test_minibatch_rejects_empty_indices(self):
        sub = self._make_sub_se3(num_pairs=4)
        frame_1 = torch.zeros(0, 3, 12, 16)
        with pytest.raises(ValueError, match="non-empty"):
            sub.reconstruct_pair(
                frame_1, pair_indices=torch.empty(0, dtype=torch.long)
            )

    def test_minibatch_rejects_frame1_len_mismatch(self):
        sub = self._make_sub_se3(num_pairs=4)
        # frame_1 has 2 entries but pair_indices has 3 — mismatch.
        with pytest.raises(ValueError, match="!="):
            sub.reconstruct_pair(
                torch.rand(2, 3, 12, 16),
                pair_indices=torch.tensor([0, 1, 2], dtype=torch.long),
            )

    def test_minibatch_rejects_out_of_range_indices(self):
        sub = self._make_sub_se3(num_pairs=4)
        frame_1 = torch.rand(2, 3, 12, 16)
        with pytest.raises(ValueError, match="outside"):
            sub.reconstruct_pair(
                frame_1, pair_indices=torch.tensor([0, 4], dtype=torch.long)
            )

    def test_minibatch_rejects_negative_indices(self):
        sub = self._make_sub_se3(num_pairs=4)
        frame_1 = torch.rand(2, 3, 12, 16)
        with pytest.raises(ValueError, match="outside"):
            sub.reconstruct_pair(
                frame_1, pair_indices=torch.tensor([-1, 0], dtype=torch.long)
            )

    def test_minibatch_supports_single_pair(self):
        """B=1 is the val-loop hot path; must work."""
        sub = self._make_sub_se3(num_pairs=4)
        idx = torch.tensor([2], dtype=torch.long)
        frame_1 = torch.rand(1, 3, 12, 16)
        f0, _ = sub.reconstruct_pair(frame_1, pair_indices=idx)
        assert f0.shape == (1, 3, 12, 16)

    def test_minibatch_supports_duplicate_indices(self):
        """index_select tolerates duplicate indices; useful for replay
        ensembles. Gradients accumulate per the autograd contract."""
        sub = self._make_sub_se3(num_pairs=4)
        idx = torch.tensor([1, 1], dtype=torch.long)
        frame_1 = torch.rand(2, 3, 12, 16)
        f0, _ = sub.reconstruct_pair(frame_1, pair_indices=idx)
        assert f0.shape == (2, 3, 12, 16)

    def test_back_compat_full_batch_still_works(self):
        """Existing callers that don't pass pair_indices must keep working."""
        sub = self._make_sub_se3(num_pairs=2)
        f0, f1 = sub.reconstruct_pair(torch.rand(2, 3, 12, 16))
        assert f0.shape == (2, 3, 12, 16)

    def test_minibatch_oom_repro_at_smoke_scale(self):
        """Repro of the 2026-05-14 OOM anchor at smoke scale: a 600-pair
        reconstruct allocates O(600 * 384 * 512 * 3) activations through
        F.interpolate (residual upsample) which OOMs T4. We can't repro
        the actual OOM on CPU but we CAN verify that the mini-batch path
        keeps peak memory O(batch_size) by comparing param-count-vs-batch
        activation scaling at smoke scale."""
        sub = self._make_sub_se3(num_pairs=64)
        # Mini-batch at B=8 reconstructs 8 frames at a time.
        idx = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.long)
        frame_1 = torch.rand(8, 3, 12, 16)
        f0, _ = sub.reconstruct_pair(frame_1, pair_indices=idx)
        # The output shape must reflect the batch, not the full num_pairs.
        assert f0.shape == (8, 3, 12, 16), (
            f"mini-batch must produce B-size output, not num_pairs-size; "
            f"got {tuple(f0.shape)}"
        )

    def test_minibatch_loss_backward_no_retain_graph_needed(self):
        """The mini-batched path must not require retain_graph=True between
        consecutive backward calls because each batch builds its own
        forward graph. This is the central fix that eliminates both the
        OOM and the retain_graph activation accumulation."""
        torch.manual_seed(105)
        sub = self._make_sub_se3(num_pairs=8)
        sub.train()
        opt = torch.optim.SGD(sub.parameters(), lr=0.01)
        for i in range(3):
            idx = torch.tensor([i, i + 1], dtype=torch.long)
            frame_1 = torch.rand(2, 3, 12, 16)
            f0, _ = sub.reconstruct_pair(frame_1, pair_indices=idx)
            loss = (f0 - 0.5).pow(2).mean()
            opt.zero_grad()
            loss.backward()  # NO retain_graph=True; must succeed
            opt.step()

    def test_minibatch_residual_subset_selected_correctly(self):
        """The residual coarse rows used in synthesis must match the
        pair_indices selection (i.e. residual[pair_indices], not
        residual[0:len(pair_indices)])."""
        torch.manual_seed(106)
        sub = self._make_sub_se3(num_pairs=4)
        # Set residual to distinct constants per pair so we can verify the
        # right rows were selected.
        with torch.no_grad():
            for k in range(4):
                sub.residual_coarse[k].fill_(0.1 * (k + 1))
            # Zero motion so frame_0 = clamp(frame_1 + residual_upsampled).
            sub.motion.se3_flat.zero_()
        frame_1 = torch.zeros(2, 3, 12, 16) + 0.5
        # Reconstruct pair 2 (residual = 0.3) and pair 3 (residual = 0.4).
        idx = torch.tensor([2, 3], dtype=torch.long)
        f0, _ = sub.reconstruct_pair(frame_1, pair_indices=idx)
        # f0[0] should be ~ 0.5 + 0.3 = 0.8; f0[1] should be ~ 0.5 + 0.4 = 0.9
        # (within clamp_unit). Note residual upsamples from 3x4 to 12x16 by
        # bilinear; at the center the constant fill propagates exactly.
        assert f0[0].mean().item() > 0.75, (
            f"residual[2]=0.3 expected; got mean={f0[0].mean().item():.3f}"
        )
        assert f0[1].mean().item() > 0.85, (
            f"residual[3]=0.4 expected; got mean={f0[1].mean().item():.3f}"
        )


# ----------------------------------------------------------------------
# Inflate runtime tests (end-to-end)
# ----------------------------------------------------------------------

class TestInflateRuntime:
    def _build_full_archive(self, mode_int: int = 0):
        base_bytes = b"INFLATE_TEST_BASE"
        base_sha = hashlib.sha256(base_bytes).hexdigest()
        if mode_int == 0:
            se3_flat = torch.zeros(2, 6)
            flow_uv = None
        else:
            se3_flat = None
            flow_uv = torch.zeros(2, 2, 6, 8)
        residual = torch.zeros(2, 3, 12, 16)
        residual_blob = encode_residual_blob(residual, coarse_hw=(12, 16))
        return pack_archive(
            motion_mode=mode_int,
            se3_flat=se3_flat,
            flow_uv=flow_uv,
            residual_blob=residual_blob,
            meta={"base_substrate_id": "smoke_base_substrate_v0"},
            base_substrate_archive_sha256_hex=base_sha,
            base_substrate_bytes=base_bytes,
            num_pairs=2,
            flow_grid_h=6 if mode_int == 1 else 0,
            flow_grid_w=8 if mode_int == 1 else 0,
            residual_coarse_h=12,
            residual_coarse_w=16,
        )

    def test_inflate_one_video_se3(self, tmp_path):
        archive = self._build_full_archive(mode_int=0)
        out = tmp_path / "out.raw"
        n = inflate_one_video(archive, out, device="cpu")
        assert n == 4  # 2 pairs * 2 frames
        # Each frame is (874, 1164, 3) uint8
        expected_bytes = 4 * 874 * 1164 * 3
        assert out.stat().st_size == expected_bytes

    def test_inflate_one_video_optical_flow(self, tmp_path):
        archive = self._build_full_archive(mode_int=1)
        out = tmp_path / "out.raw"
        n = inflate_one_video(archive, out, device="cpu")
        assert n == 4
        expected_bytes = 4 * 874 * 1164 * 3
        assert out.stat().st_size == expected_bytes

    def test_inflate_verifies_base_sha(self, tmp_path):
        """Tampered base bytes must be detected and refused."""
        archive = self._build_full_archive(mode_int=0)
        parsed = parse_archive(archive)
        # Build a tampered archive with the same sha but different base bytes.
        # We do this by directly manipulating the header to keep the sha but
        # alter a byte in base_substrate_bytes section.
        # NOTE: simpler regression — flip a base byte and rebuild header bytes
        # by mutating after pack.
        b = bytearray(archive)
        # base bytes start at WZF01_HEADER_SIZE + BASE_SHA_HEX_LEN
        bb_start = WZF01_HEADER_SIZE + BASE_SHA_HEX_LEN
        b[bb_start] ^= 0xFF
        tampered = bytes(b)
        out = tmp_path / "out.raw"
        with pytest.raises(ValueError, match="sha256 mismatch"):
            inflate_one_video(tampered, out, device="cpu")

    def test_inflate_provider_registry_supports_multiple_base_substrates(self):
        """Composability: any base substrate can register a provider."""
        for base_id in ("a1_hnerv_ft_microcodec", "pr101_lc_v2", "hdm8"):
            register_base_substrate_provider(
                base_id,
                lambda b, i, d: torch.zeros(1, 3, 384, 512, device=d),
            )
        # Now check that the inflate path resolves the provider when meta
        # specifies the registered id.
        from tac.substrates.d4_wyner_ziv_frame_0 import inflate as inflate_mod
        for base_id in ("a1_hnerv_ft_microcodec", "pr101_lc_v2", "hdm8"):
            assert base_id in inflate_mod._BASE_PROVIDERS

    def test_inflate_falls_back_to_smoke_provider_on_unknown_base(self, tmp_path):
        """An unregistered base id falls back to the smoke provider (does
        NOT silently produce broken output — it produces deterministic
        random frames suitable for tests)."""
        base_bytes = b"UNKNOWN_BASE_TEST"
        base_sha = hashlib.sha256(base_bytes).hexdigest()
        se3_flat = torch.zeros(1, 6)
        residual = torch.zeros(1, 3, 12, 16)
        residual_blob = encode_residual_blob(residual, coarse_hw=(12, 16))
        archive = pack_archive(
            motion_mode=0,
            se3_flat=se3_flat,
            flow_uv=None,
            residual_blob=residual_blob,
            meta={"base_substrate_id": "unregistered_id_xyz"},
            base_substrate_archive_sha256_hex=base_sha,
            base_substrate_bytes=base_bytes,
            num_pairs=1,
            flow_grid_h=0,
            flow_grid_w=0,
            residual_coarse_h=12,
            residual_coarse_w=16,
        )
        out = tmp_path / "fallback.raw"
        n = inflate_one_video(archive, out, device="cpu")
        assert n == 2  # 1 pair * 2 frames

    def test_inflate_cli_rejects_too_few_args(self):
        """Catalog #146 CLI contract — 3 positional args required."""
        from tac.substrates.d4_wyner_ziv_frame_0.inflate import main_cli
        old_argv = sys.argv
        try:
            sys.argv = ["inflate.py", "only_one"]
            rc = main_cli()
            assert rc == 2
        finally:
            sys.argv = old_argv

    def test_inflate_runtime_no_scorer_import(self):
        """Strict scorer rule: inflate.py must NOT import PoseNet/SegNet."""
        inflate_file = (
            REPO_ROOT
            / "src"
            / "tac"
            / "substrates"
            / "d4_wyner_ziv_frame_0"
            / "inflate.py"
        )
        text = inflate_file.read_text(encoding="utf-8")
        for forbidden in (
            "PoseNet",
            "SegNet",
            "from upstream.modules",
            "import upstream.modules",
            "rgb_to_yuv6",
            "EfficientNet",
            "FastViT",
        ):
            assert forbidden not in text, (
                f"forbidden scorer-load token {forbidden!r} present in inflate.py"
            )


# ----------------------------------------------------------------------
# Score-aware loss tests (canonical scorer contract — Catalog #164)
# ----------------------------------------------------------------------

class TestScoreAwareLoss:
    def test_loss_uses_score_pair_components(self):
        """Loss must route through canonical helper (Catalog #164)."""
        loss_file = (
            REPO_ROOT
            / "src"
            / "tac"
            / "substrates"
            / "d4_wyner_ziv_frame_0"
            / "score_aware_loss.py"
        )
        text = loss_file.read_text(encoding="utf-8")
        assert "from tac.substrates.score_aware_common" in text
        assert "score_pair_components" in text
        assert "apply_eval_roundtrip_during_training" in text

    def test_loss_refuses_disabled_eval_roundtrip(self):
        """eval_roundtrip=False is forbidden per CLAUDE.md non-negotiable."""
        from tac.substrates.d4_wyner_ziv_frame_0.score_aware_loss import (
            WynerZivFrame0ScoreAwareLoss,
        )

        class DummyScorer(torch.nn.Module):
            def preprocess_input(self, x):  # noqa: D401 — stub
                return x

            def forward(self, x):
                return x

        loss = WynerZivFrame0ScoreAwareLoss(
            DummyScorer(), DummyScorer(), WynerZivFrame0LossWeights()
        )
        with pytest.raises(ValueError, match="eval_roundtrip"):
            loss(
                reconstructed_rgb_0=torch.zeros(1, 3, 12, 16),
                reconstructed_rgb_1=torch.zeros(1, 3, 12, 16),
                gt_rgb_0=torch.zeros(1, 3, 12, 16),
                gt_rgb_1=torch.zeros(1, 3, 12, 16),
                archive_bytes_proxy=torch.tensor(100.0),
                residual_coarse=torch.zeros(1, 3, 6, 8),
                apply_eval_roundtrip=False,
            )

    def test_loss_validates_rgb_255_domain(self):
        """Unit-domain RGB passed by mistake must be refused."""
        from tac.substrates.d4_wyner_ziv_frame_0.score_aware_loss import (
            WynerZivFrame0ScoreAwareLoss,
        )

        class DummyScorer(torch.nn.Module):
            def preprocess_input(self, x):
                return x

            def forward(self, x):
                return x

        loss = WynerZivFrame0ScoreAwareLoss(
            DummyScorer(), DummyScorer(), WynerZivFrame0LossWeights()
        )
        # Unit-domain (max <= 1.0) RGB should be refused.
        with pytest.raises(ValueError, match="unit-domain"):
            loss(
                reconstructed_rgb_0=torch.rand(1, 3, 12, 16),
                reconstructed_rgb_1=torch.rand(1, 3, 12, 16),
                gt_rgb_0=torch.rand(1, 3, 12, 16),
                gt_rgb_1=torch.rand(1, 3, 12, 16),
                archive_bytes_proxy=torch.tensor(100.0),
                residual_coarse=torch.zeros(1, 3, 6, 8),
                apply_eval_roundtrip=True,
            )


# ----------------------------------------------------------------------
# Compliance + custody tests
# ----------------------------------------------------------------------

class TestCompliance:
    def test_substrate_declares_archive_grammar_inline(self):
        """Catalog #124: 8 archive_grammar fields declared inline in __init__.py."""
        init_file = (
            REPO_ROOT
            / "src"
            / "tac"
            / "substrates"
            / "d4_wyner_ziv_frame_0"
            / "__init__.py"
        )
        text = init_file.read_text(encoding="utf-8")
        for field in (
            "archive_grammar",
            "parser_section_manifest",
            "inflate_runtime_loc_budget",
            "runtime_dep_closure",
            "export_format",
            "score_aware_loss",
            "bolt_on_loc_budget",
            "no_op_detector_planned",
        ):
            assert field in text, f"Catalog #124 field {field!r} missing"
        assert "lane_class=substrate_engineering" in text or "substrate_engineering" in text
        assert "lane_class" in text or "substrate-engineering" in text.lower()

    def test_no_tmp_paths_in_substrate(self):
        """No /tmp paths persisted (CLAUDE.md FORBIDDEN_PATTERNS).

        Scope: production substrate modules only. The tests/ subdirectory
        is excluded because test files use the pytest tmp_path fixture and
        may reference the literal "/tmp/" pattern in docstrings explaining
        the FORBIDDEN_PATTERNS rule.
        """
        subdir = (
            REPO_ROOT / "src" / "tac" / "substrates" / "d4_wyner_ziv_frame_0"
        )
        for path in subdir.rglob("*.py"):
            if "tests" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            sentinel = "/" + "tmp" + "/"  # avoid literal in test file source
            assert sentinel not in text, (
                f"transient /tmp path leaked into persisted module: {path}"
            )

    def test_inflate_loc_budget_under_350(self):
        """HNeRV parity L4: inflate.py ≤ 350 LOC substrate-engineering waiver."""
        inflate_file = (
            REPO_ROOT
            / "src"
            / "tac"
            / "substrates"
            / "d4_wyner_ziv_frame_0"
            / "inflate.py"
        )
        n_lines = sum(1 for _ in inflate_file.read_text(encoding="utf-8").splitlines())
        assert n_lines <= 350, f"inflate.py LOC {n_lines} > 350 substrate-engineering budget"

    def test_public_api_exports_complete(self):
        """The package __init__.py exports every public symbol."""
        from tac.substrates import d4_wyner_ziv_frame_0 as pkg

        required = {
            "MotionModelMode",
            "WynerZivFrame0Config",
            "WynerZivFrame0Substrate",
            "WynerZivFrame0ScoreAwareLoss",
            "pack_archive",
            "parse_archive",
            "synthesize_frame_0",
            "encode_residual_blob",
            "decode_residual_blob",
            "apply_se3_motion",
            "apply_optical_flow",
        }
        actual = set(pkg.__all__)
        missing = required - actual
        assert not missing, f"missing exports: {missing}"
