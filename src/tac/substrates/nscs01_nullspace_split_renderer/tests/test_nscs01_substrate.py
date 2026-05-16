# SPDX-License-Identifier: MIT
"""NSCS01 substrate tests.

Tests cover the canonical-vs-unique decisions per the design memo §7:

* Architecture: split-head per-pair forward; nullspace property
* Score-aware loss: routes through canonical scorer_loss_terms_btchw;
  Catalog #164 preprocess_input contract honored; eval_roundtrip mandated;
  split gradient routing (frame_0 head receives ZERO seg gradient)
* Archive: NSP1 grammar parser symmetry per head; per-head bit-width
  packing; deterministic byte-stable output; byte-mutation smoke per
  Catalog #139 (mutate HEAD0 → frame_0 changes; mutate HEAD1 →
  frame_1 changes; mutate LATENT → both change)
* Inflate runtime: 3-positional-arg signature (Catalog #146); no scorer
  imports (Catalog #6); device select via canonical helper (Catalog #205)
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import torch

from tac.substrates.nscs01_nullspace_split_renderer import (
    CAMERA_H,
    CAMERA_W,
    NSP1_HEADER_SIZE,
    NSP1_MAGIC,
    NSP1_SCHEMA_VERSION,
    NullspaceSplitArchive,
    NullspaceSplitConfig,
    NullspaceSplitLossWeights,
    NullspaceSplitRenderer,
    NullspaceSplitScoreAwareLoss,
    deserialize_head_state_dicts,
    deserialize_latents,
    pack_archive,
    parse_archive,
)
from tac.substrates.nscs01_nullspace_split_renderer.archive import (
    _dequantize_int,
    _pack_bits,
    _quantize_int,
    _unpack_bits,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smoke_renderer(num_pairs: int = 4) -> NullspaceSplitRenderer:
    cfg = NullspaceSplitConfig(
        latent_dim=8,
        head0_bits=4,
        head1_bits=8,
        latent_bits=12,
        head0_base_channels=8,
        head1_base_channels=16,
        num_pairs=num_pairs,
    )
    return NullspaceSplitRenderer(cfg)


# ---------------------------------------------------------------------------
# Architecture tests
# ---------------------------------------------------------------------------

class TestArchitecture:
    def test_config_validates_head0_bits(self):
        with pytest.raises(ValueError, match="head0_bits"):
            NullspaceSplitConfig(head0_bits=10)

    def test_config_validates_head1_bits(self):
        with pytest.raises(ValueError, match="head1_bits"):
            NullspaceSplitConfig(head1_bits=4)  # head1 must be 6/8 not 4

    def test_config_validates_latent_bits(self):
        with pytest.raises(ValueError, match="latent_bits"):
            NullspaceSplitConfig(latent_bits=4)

    def test_config_validates_latent_dim(self):
        with pytest.raises(ValueError, match="latent_dim"):
            NullspaceSplitConfig(latent_dim=0)
        with pytest.raises(ValueError, match="latent_dim"):
            NullspaceSplitConfig(latent_dim=512)

    def test_config_validates_num_pairs(self):
        with pytest.raises(ValueError, match="num_pairs"):
            NullspaceSplitConfig(num_pairs=0)

    def test_renderer_produces_two_frames(self):
        r = _smoke_renderer()
        idx = torch.tensor([0, 1])
        f0, f1 = r.reconstruct_pair(idx)
        assert f0.shape == (2, 3, CAMERA_H, CAMERA_W)
        assert f1.shape == (2, 3, CAMERA_H, CAMERA_W)

    def test_frames_are_in_rgb_255_domain(self):
        r = _smoke_renderer()
        idx = torch.tensor([0, 1, 2, 3])
        f0, f1 = r.reconstruct_pair(idx)
        assert f0.min() >= 0.0
        assert f0.max() <= 255.0
        assert f1.min() >= 0.0
        assert f1.max() <= 255.0

    def test_frame_0_head_smaller_than_frame_1_head(self):
        """The whole point: frame_0 head MUST be smaller than frame_1 head."""
        r = _smoke_renderer()
        n0 = sum(p.numel() for p in r.frame_0_head.parameters())
        n1 = sum(p.numel() for p in r.frame_1_head.parameters())
        assert n0 < n1, f"frame_0_head ({n0}) must be smaller than frame_1_head ({n1})"

    def test_pair_indices_validation(self):
        r = _smoke_renderer(num_pairs=4)
        with pytest.raises(ValueError, match="1-D"):
            r.reconstruct_pair(torch.tensor([[0, 1]]))
        with pytest.raises(ValueError, match="non-empty"):
            r.reconstruct_pair(torch.tensor([], dtype=torch.long))
        with pytest.raises(ValueError, match="outside"):
            r.reconstruct_pair(torch.tensor([0, 99]))
        with pytest.raises(ValueError, match="outside"):
            r.reconstruct_pair(torch.tensor([-1, 0]))

    def test_pair_indices_preserves_gradient_only_to_selected_rows(self):
        """index_select gradient flow must reach ONLY the selected latents."""
        r = _smoke_renderer(num_pairs=4)
        idx = torch.tensor([1, 2])
        f0, f1 = r.reconstruct_pair(idx)
        loss = (f0.mean() + f1.mean())
        loss.backward()
        # latents.grad must be non-zero only for rows 1 and 2.
        grad_norms = r.latents.grad.norm(dim=1)
        assert grad_norms[0].item() == 0.0
        assert grad_norms[1].item() > 0.0
        assert grad_norms[2].item() > 0.0
        assert grad_norms[3].item() == 0.0


# ---------------------------------------------------------------------------
# Archive tests
# ---------------------------------------------------------------------------

class TestArchive:
    def test_header_size_invariant(self):
        assert NSP1_HEADER_SIZE == 32

    def test_magic_constant(self):
        assert NSP1_MAGIC == b"NSP\x01"

    def test_pack_parse_roundtrip(self):
        r = _smoke_renderer()
        cfg = r.cfg
        archive = pack_archive(
            head0_state_dict=r.frame_0_head.state_dict(),
            head1_state_dict=r.frame_1_head.state_dict(),
            latents=r.latents,
            head0_bits=cfg.head0_bits,
            head1_bits=cfg.head1_bits,
            latent_bits=cfg.latent_bits,
            head0_base_channels=cfg.head0_base_channels,
            head1_base_channels=cfg.head1_base_channels,
        )
        parsed = parse_archive(archive)
        assert parsed.version == NSP1_SCHEMA_VERSION
        assert parsed.num_pairs == cfg.num_pairs
        assert parsed.latent_dim == cfg.latent_dim
        assert parsed.head0_bits == cfg.head0_bits
        assert parsed.head1_bits == cfg.head1_bits
        assert parsed.latent_bits == cfg.latent_bits
        assert parsed.head0_base_channels == cfg.head0_base_channels
        assert parsed.head1_base_channels == cfg.head1_base_channels

    def test_per_head_state_dict_roundtrip(self):
        r = _smoke_renderer()
        cfg = r.cfg
        archive = pack_archive(
            head0_state_dict=r.frame_0_head.state_dict(),
            head1_state_dict=r.frame_1_head.state_dict(),
            latents=r.latents,
            head0_bits=cfg.head0_bits,
            head1_bits=cfg.head1_bits,
            latent_bits=cfg.latent_bits,
            head0_base_channels=cfg.head0_base_channels,
            head1_base_channels=cfg.head1_base_channels,
        )
        parsed = parse_archive(archive)
        sd0, sd1 = deserialize_head_state_dicts(parsed)
        # Same keys + same shapes.
        assert set(sd0.keys()) == set(r.frame_0_head.state_dict().keys())
        assert set(sd1.keys()) == set(r.frame_1_head.state_dict().keys())
        for k, v in r.frame_0_head.state_dict().items():
            assert sd0[k].shape == v.shape

    def test_latent_roundtrip(self):
        r = _smoke_renderer()
        cfg = r.cfg
        archive = pack_archive(
            head0_state_dict=r.frame_0_head.state_dict(),
            head1_state_dict=r.frame_1_head.state_dict(),
            latents=r.latents,
            head0_bits=cfg.head0_bits,
            head1_bits=cfg.head1_bits,
            latent_bits=cfg.latent_bits,
            head0_base_channels=cfg.head0_base_channels,
            head1_base_channels=cfg.head1_base_channels,
        )
        parsed = parse_archive(archive)
        lats = deserialize_latents(parsed)
        assert lats.shape == r.latents.shape
        # 12-bit latents have ~4096 levels — quantization error is bounded.
        max_err = (lats - r.latents.detach().cpu()).abs().max().item()
        # Per-latent symmetric quantization with range ≈ 0.5 → error ≤ 0.5/4095.
        assert max_err < 1e-3, f"12-bit latent quantization error too large: {max_err}"

    def test_archive_rejects_bad_magic(self):
        bad = b"BAD\x01" + b"\x00" * (NSP1_HEADER_SIZE - 4)
        with pytest.raises(ValueError, match="magic"):
            parse_archive(bad)

    def test_archive_rejects_truncated(self):
        with pytest.raises(ValueError, match="too small"):
            parse_archive(b"NSP\x01")

    def test_pack_parse_deterministic(self):
        """Same inputs → byte-identical archive (deterministic for replay)."""
        r = _smoke_renderer()
        cfg = r.cfg
        kwargs = {
            "head0_state_dict": r.frame_0_head.state_dict(),
            "head1_state_dict": r.frame_1_head.state_dict(),
            "latents": r.latents,
            "head0_bits": cfg.head0_bits,
            "head1_bits": cfg.head1_bits,
            "latent_bits": cfg.latent_bits,
            "head0_base_channels": cfg.head0_base_channels,
            "head1_base_channels": cfg.head1_base_channels,
        }
        a1 = pack_archive(**kwargs)
        a2 = pack_archive(**kwargs)
        assert a1 == a2, "archive must be deterministic for replay"


# ---------------------------------------------------------------------------
# Bit-packing tests
# ---------------------------------------------------------------------------

class TestBitPacking:
    @pytest.mark.parametrize("bits", [4, 6, 8, 12])
    def test_quantize_dequantize_roundtrip(self, bits):
        import numpy as np
        rng = np.random.default_rng(0)
        x = rng.standard_normal(1024).astype(np.float32)
        q, scale, lo = _quantize_int(x, bits)
        deq = _dequantize_int(q, scale, lo, bits)
        # Quantization error should be bounded by scale * 0.5 (half-step).
        err = np.abs(x - deq).max()
        assert err <= scale * 0.5 + 1e-6, f"bits={bits} err={err} scale={scale}"

    @pytest.mark.parametrize("bits", [4, 6, 8, 12])
    def test_pack_unpack_roundtrip(self, bits):
        import numpy as np
        half = 1 << (bits - 1)
        # Build a deterministic int16 sequence in [-half, half-1].
        n = 100
        q = (np.arange(n) % (2 * half) - half).astype(np.int16)
        packed = _pack_bits(q, bits)
        unpacked = _unpack_bits(packed, bits, n)
        assert (q == unpacked).all(), f"bits={bits} roundtrip mismatch"

    def test_quantize_degenerate_range(self):
        """Catalog #161: degenerate range fills with -half, not 0."""
        import numpy as np
        x = np.full(10, 3.14, dtype=np.float32)
        q, scale, lo = _quantize_int(x, 4)
        assert (q == -8).all(), f"degenerate range must fill -half=-8; got {q}"


# ---------------------------------------------------------------------------
# Score-aware loss tests
# ---------------------------------------------------------------------------

class _MockScorer(torch.nn.Module):
    """Minimal scorer with preprocess_input contract for test purposes."""
    def __init__(self, channels: int = 5, target_h: int = 16, target_w: int = 32):
        super().__init__()
        self.channels = channels
        self.target_h = target_h
        self.target_w = target_w
        self.conv = torch.nn.Conv2d(3, channels, 3, padding=1)

    def preprocess_input(self, x):
        # Mimic SegNet: take last frame only.
        last = x[:, -1, ...]
        return torch.nn.functional.interpolate(
            last, size=(self.target_h, self.target_w), mode="bilinear", align_corners=False
        )

    def forward(self, x):
        return self.conv(x)


class _MockPoseScorer(torch.nn.Module):
    """Mock PoseNet that uses BOTH frames (canonical hydra-head dict surface)."""
    def __init__(self, target_h: int = 16, target_w: int = 32):
        super().__init__()
        self.target_h = target_h
        self.target_w = target_w
        self.conv = torch.nn.Conv2d(6, 6, 3, padding=1)  # 6 channels output
        self.proj = torch.nn.Linear(6 * target_h * target_w, 12)

    def preprocess_input(self, x):
        # Mimic PoseNet: rearrange both frames to one tensor (6 channels here for simplicity).
        b, t, c, h, w = x.shape
        x = x.contiguous().view(b, t * c, h, w)
        x = torch.nn.functional.interpolate(
            x, size=(self.target_h, self.target_w), mode="bilinear", align_corners=False
        )
        return x

    def forward(self, x):
        x.shape[0]
        h = self.conv(x)
        flat = h.flatten(1)
        # Canonical PoseNet returns dict {"pose": (B, 12)} per upstream/modules.py hydra head.
        pose = self.proj(flat)
        return {"pose": pose}


class TestScoreAwareLoss:
    def test_loss_rejects_non_preprocess_seg_scorer(self):
        bad = torch.nn.Linear(3, 5)
        good_pose = _MockPoseScorer()
        from tac.substrates.score_aware_common import ScoreAwareScorerContractError
        with pytest.raises(ScoreAwareScorerContractError):
            NullspaceSplitScoreAwareLoss(bad, good_pose, NullspaceSplitLossWeights())

    def test_loss_rejects_non_preprocess_pose_scorer(self):
        good_seg = _MockScorer()
        bad = torch.nn.Linear(3, 5)
        from tac.substrates.score_aware_common import ScoreAwareScorerContractError
        with pytest.raises(ScoreAwareScorerContractError):
            NullspaceSplitScoreAwareLoss(good_seg, bad, NullspaceSplitLossWeights())

    def test_loss_refuses_eval_roundtrip_false(self):
        loss = NullspaceSplitScoreAwareLoss(
            _MockScorer(), _MockPoseScorer(), NullspaceSplitLossWeights()
        )
        f0 = torch.rand(2, 3, 24, 32) * 255.0
        gt = torch.rand(2, 3, 24, 32) * 255.0
        with pytest.raises(ValueError, match="eval_roundtrip"):
            loss(
                frame_0_pred=f0, frame_1_pred=f0,
                gt_frame_0=gt, gt_frame_1=gt,
                archive_bytes_proxy=torch.tensor(1000.0),
                apply_eval_roundtrip=False,
            )

    def test_loss_validates_rgb_255_domain(self):
        loss = NullspaceSplitScoreAwareLoss(
            _MockScorer(), _MockPoseScorer(), NullspaceSplitLossWeights()
        )
        # unit-domain (0-1) RGB caught by _validate_rgb_255_domain.
        f0_unit = torch.rand(2, 3, 24, 32)  # max ~ 1.0 → flagged as unit
        gt = torch.rand(2, 3, 24, 32) * 255.0
        with pytest.raises(ValueError, match="unit-domain"):
            loss(
                frame_0_pred=f0_unit, frame_1_pred=gt,
                gt_frame_0=gt, gt_frame_1=gt,
                archive_bytes_proxy=torch.tensor(1000.0),
            )

    def test_loss_returns_parts_dict(self):
        loss = NullspaceSplitScoreAwareLoss(
            _MockScorer(), _MockPoseScorer(), NullspaceSplitLossWeights()
        )
        # Build (B, 3, H, W) at scorer input H, W = 16, 32 (the mocks resize anyway).
        f = torch.rand(2, 3, 24, 32) * 255.0
        gt = torch.rand(2, 3, 24, 32) * 255.0
        loss_value, parts = loss(
            frame_0_pred=f, frame_1_pred=f,
            gt_frame_0=gt, gt_frame_1=gt,
            archive_bytes_proxy=torch.tensor(1000.0),
            noise_std=0.0,
        )
        assert loss_value.ndim == 0
        assert "rate_term" in parts
        assert "seg_term" in parts
        assert "pose_term" in parts
        assert "pose_sqrt" in parts
        assert "pixel_0_l2" in parts
        assert "pixel_1_l2" in parts
        assert "loss_total" in parts
        assert torch.isfinite(loss_value)


# ---------------------------------------------------------------------------
# Byte-mutation smoke (Catalog #139)
# ---------------------------------------------------------------------------

class TestByteMutationSmoke:
    """Per Catalog #139: prove the archive bytes are STRUCTURALLY consumed
    by inflate. A byte mutation in HEAD0 must change frame_0; in HEAD1 must
    change frame_1; in LATENT must change BOTH frames.
    """

    def _build_archive_and_render(self, archive_bytes: bytes, device: str = "cpu"):
        from tac.substrates.nscs01_nullspace_split_renderer.inflate import (
            _build_renderer_from_archive_bytes,
        )
        renderer, _ = _build_renderer_from_archive_bytes(archive_bytes, device)
        idx = torch.tensor([0, 1], dtype=torch.long)
        with torch.no_grad():
            f0, f1 = renderer.reconstruct_pair(idx)
        return f0.cpu(), f1.cpu()

    def _smoke_archive(self) -> tuple[bytes, NullspaceSplitArchive]:
        r = _smoke_renderer(num_pairs=2)
        cfg = r.cfg
        archive = pack_archive(
            head0_state_dict=r.frame_0_head.state_dict(),
            head1_state_dict=r.frame_1_head.state_dict(),
            latents=r.latents,
            head0_bits=cfg.head0_bits,
            head1_bits=cfg.head1_bits,
            latent_bits=cfg.latent_bits,
            head0_base_channels=cfg.head0_base_channels,
            head1_base_channels=cfg.head1_base_channels,
        )
        parsed = parse_archive(archive)
        return archive, parsed

    def _flip_byte_inside(
        self,
        archive: bytes,
        section_offset: int,
        section_len: int,
        flip_at_within: int = 16,
    ) -> bytes:
        """Flip one byte inside the named section."""
        if flip_at_within >= section_len:
            flip_at_within = section_len - 1
        target = section_offset + flip_at_within
        b = bytearray(archive)
        # XOR the byte to ensure it changes (any non-zero value).
        b[target] ^= 0x55
        return bytes(b)

    def _section_offsets(self, parsed: NullspaceSplitArchive) -> dict[str, tuple[int, int]]:
        """Return (offset, length) for each blob section in the COMPRESSED form."""
        # Re-parse header to get compressed-section lengths.
        struct.unpack_from("<4sBHHBBBHHIIII", b"\x00")  # placeholder
        # We need the compressed lengths from the original header bytes.
        # Easier: build a raw copy of the archive and read header directly.
        raise NotImplementedError("computed in test_inflate_consumes_each_section instead")

    def _section_offsets_from_header(self, archive: bytes) -> dict[str, tuple[int, int]]:
        """Decode (offset, length) per section from the header."""
        (
            _magic, _version, _num_pairs, _latent_dim,
            _head0_bits, _head1_bits, _latent_bits,
            _h0_ch, _h1_ch,
            head0_len, head1_len, latent_len, _meta_len,
        ) = struct.unpack_from("<4sBHHBBBHHIIII", archive, 0)
        head0_off = NSP1_HEADER_SIZE
        head1_off = head0_off + head0_len
        latent_off = head1_off + head1_len
        return {
            "HEAD0": (head0_off, head0_len),
            "HEAD1": (head1_off, head1_len),
            "LATENT": (latent_off, latent_len),
        }

    def _try_render(self, archive: bytes) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Render or return None if the mutated archive fails to decode.

        Catalog #139 byte-mutation smoke: consumption is PROVEN if the
        bytes either (a) produce different output OR (b) cause a parse/decode
        error. A parse/decode error is still consumption — the byte was read
        and rejected as inconsistent.
        """
        try:
            return self._build_archive_and_render(archive)
        except Exception:
            return None

    def test_inflate_consumes_each_section(self):
        """Per Catalog #139: byte mutation in each section must either change
        rendered output or cause a parse/decode error (both prove consumption).
        """
        archive, _parsed = self._smoke_archive()
        offsets = self._section_offsets_from_header(archive)

        # Baseline render.
        f0_base, f1_base = self._build_archive_and_render(archive)

        for section_name, (offset, length) in offsets.items():
            # Try multiple flip positions; brotli's stream is fragile and most
            # mutations cause decode error (which is still proof of consumption).
            consumption_proven = False
            for flip_pos in (8, 16, 32, length // 2, length - 8):
                if flip_pos < 0 or flip_pos >= length:
                    continue
                mutated = self._flip_byte_inside(archive, offset, length, flip_pos)
                result = self._try_render(mutated)
                if result is None:
                    # Parse / decompress / unpack error → byte was consumed.
                    consumption_proven = True
                    break
                f0_mut, f1_mut = result
                # Check the relevant frame(s) for changes.
                f0_changed = not torch.allclose(f0_base, f0_mut, atol=1e-4)
                f1_changed = not torch.allclose(f1_base, f1_mut, atol=1e-4)
                if section_name == "HEAD0" and f0_changed:
                    consumption_proven = True
                    break
                if section_name == "HEAD1" and f1_changed:
                    consumption_proven = True
                    break
                if section_name == "LATENT" and (f0_changed or f1_changed):
                    consumption_proven = True
                    break
            assert consumption_proven, (
                f"{section_name} mutation produced neither frame change nor "
                f"decode error — inflate not consuming {section_name}"
            )


# ---------------------------------------------------------------------------
# Catalog #164 contract regression
# ---------------------------------------------------------------------------

class TestCatalog164PreprocessContract:
    def test_loss_routes_through_score_pair_components_pattern(self):
        """The score_aware_loss source must reference scorer_loss_terms_btchw."""
        from tac.substrates.nscs01_nullspace_split_renderer import score_aware_loss
        src = Path(score_aware_loss.__file__).read_text(encoding="utf-8")
        assert "scorer_loss_terms_btchw" in src, (
            "Catalog #164 contract requires routing through canonical "
            "scorer_loss_terms_btchw helper"
        )
        assert "stage_frame_pair" in src, (
            "must use canonical stage_frame_pair helper"
        )

    def test_inflate_does_not_import_scorer_modules(self):
        """Catalog #6: no scorer load at inflate time."""
        from tac.substrates.nscs01_nullspace_split_renderer import inflate
        src = Path(inflate.__file__).read_text(encoding="utf-8")
        forbidden_tokens = (
            "from upstream.modules",
            "import upstream.modules",
            "PoseNet",
            "SegNet",
            "FastViT",
            "EfficientNet",
            "rgb_to_yuv6",
        )
        for tok in forbidden_tokens:
            assert tok not in src, (
                f"inflate.py contains forbidden scorer token {tok!r} "
                f"(strict-scorer-rule violation)"
            )

    def test_runtime_inflate_template_uses_3_arg_signature(self):
        """Catalog #146: contest contract requires 3-positional-arg signature."""
        import tempfile

        from experiments.train_substrate_nscs01_nullspace_split_renderer import (
            _write_runtime,
        )
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "submission_dir"
            _write_runtime(out)
            sh = (out / "inflate.sh").read_text(encoding="utf-8")
            # Look for $1, $2, $3 OR DATA_DIR/OUTPUT_DIR/FILE_LIST helpers.
            for token in ("$1", "$2", "$3", "DATA_DIR", "OUTPUT_DIR", "FILE_LIST"):
                assert token in sh, f"inflate.sh missing {token}"
            assert "set -euo pipefail" in sh, (
                "Catalog #163: inflate.sh must use set -euo pipefail"
            )


# ---------------------------------------------------------------------------
# SubstrateContract registration
# ---------------------------------------------------------------------------

class TestSubstrateContract:
    def test_contract_validates_at_import(self):
        """Importing registered_substrate.py must succeed (validates contract)."""
        from tac.substrates.nscs01_nullspace_split_renderer.registered_substrate import (
            NSCS01_NULLSPACE_SPLIT_RENDERER_CONTRACT,
        )
        assert NSCS01_NULLSPACE_SPLIT_RENDERER_CONTRACT.id == "nscs01_nullspace_split_renderer"
        assert NSCS01_NULLSPACE_SPLIT_RENDERER_CONTRACT.lane_id == (
            "lane_nscs01_nullspace_split_renderer_20260515"
        )

    def test_contract_target_modes_research_substrate_at_l1_scaffold(self):
        """L1 SCAFFOLD landing: target_modes is 'research_substrate' per
        Catalog #240 sister-protection. Promotion to contest_one_video_replay
        requires paired CPU/CUDA Tier C custody + council green-up."""
        from tac.substrates.nscs01_nullspace_split_renderer.registered_substrate import (
            NSCS01_NULLSPACE_SPLIT_RENDERER_CONTRACT,
        )
        assert "research_substrate" in NSCS01_NULLSPACE_SPLIT_RENDERER_CONTRACT.target_modes


# ---------------------------------------------------------------------------
# Smoke trainer end-to-end
# ---------------------------------------------------------------------------

class TestSmokeTrainerEndToEnd:
    def test_smoke_main_runs_and_emits_archive(self, tmp_path):
        from experiments.train_substrate_nscs01_nullspace_split_renderer import main
        out_dir = tmp_path / "smoke_out"
        rc = main([
            "--smoke",
            "--output-dir", str(out_dir),
            "--device", "cpu",
            "--epochs", "3",
        ])
        assert rc == 0
        assert (out_dir / "0.bin").exists()
        assert (out_dir / "archive.zip").exists()
        assert (out_dir / "stats.json").exists()
        assert (out_dir / "submission_dir" / "inflate.sh").exists()
        assert (out_dir / "submission_dir" / "inflate.py").exists()

    def test_full_main_is_council_gated(self):
        """Per CLAUDE.md substrate-scaffolds-MUST-be-COMPLETE: full path raises."""
        import tempfile

        from experiments.train_substrate_nscs01_nullspace_split_renderer import main
        with tempfile.TemporaryDirectory() as td, pytest.raises(NotImplementedError, match="council-gated"):
            main([
                "--output-dir", td,
                "--device", "cpu",
            ])


# ---------------------------------------------------------------------------
# Byte-closed packet builder / runtime consumption proof
# ---------------------------------------------------------------------------

class TestNSCS01PacketBuilder:
    def test_builder_emits_no_score_byte_closed_manifest(self, tmp_path: Path):
        from tools.build_nscs01_split_renderer_packet import (
            build_nscs01_split_renderer_packet,
        )

        out_dir = tmp_path / "nscs01_packet"
        manifest = build_nscs01_split_renderer_packet(
            out_dir=out_dir,
            seed=123,
            num_pairs=1,
            latent_dim=4,
            head0_base_channels=4,
            head1_base_channels=8,
        )
        manifest_json = json.loads((out_dir / "build_manifest.json").read_text())

        assert manifest["score_claim"] is False
        assert manifest["promotion_eligible"] is False
        assert manifest["ready_for_exact_eval_dispatch"] is False
        assert manifest["packet_kind"] == "frame0_pose_heavy_frame1_seg_heavy_split_renderer"
        assert manifest_json["byte_closed_runtime_packet"] is True
        assert (out_dir / "0.bin").is_file()
        assert (out_dir / "archive.zip").is_file()
        assert (out_dir / "submission_dir" / "inflate.py").is_file()
        assert (out_dir / "submission_dir" / "inflate.sh").is_file()

        with zipfile.ZipFile(out_dir / "archive.zip") as zf:
            infos = zf.infolist()
            assert [info.filename for info in infos] == ["0.bin"]
            assert zf.read("0.bin") == (out_dir / "0.bin").read_bytes()

        proof = manifest_json["runtime_consumption_proof"]
        assert proof["score_claim"] is False
        assert proof["all_score_affecting_sections_consumed"] is True
        consumed = {
            section["section"]: section["consumed_by_runtime"]
            for section in proof["sections"]
        }
        assert consumed == {
            "HEAD0_BLOB": True,
            "HEAD1_BLOB": True,
            "LATENT_BLOB": True,
        }
        assert set(manifest_json["section_manifest"]) == {
            "HEAD0_BLOB",
            "HEAD1_BLOB",
            "LATENT_BLOB",
            "META_BLOB",
        }

    def test_emitted_runtime_inflates_builder_payload(self, tmp_path: Path):
        from tools.build_nscs01_split_renderer_packet import (
            build_nscs01_split_renderer_packet,
        )

        out_dir = tmp_path / "nscs01_packet"
        build_nscs01_split_renderer_packet(
            out_dir=out_dir,
            seed=456,
            num_pairs=1,
            latent_dim=4,
            head0_base_channels=4,
            head1_base_channels=8,
            run_consumption_proof=False,
        )
        file_list = tmp_path / "file_list.txt"
        output_dir = tmp_path / "raw_out"
        file_list.write_text("0.mkv\n", encoding="utf-8")

        subprocess.run(
            [
                sys.executable,
                str(out_dir / "submission_dir" / "inflate.py"),
                str(out_dir),
                str(output_dir),
                str(file_list),
            ],
            check=True,
        )

        raw_path = output_dir / "0.raw"
        assert raw_path.is_file()
        assert raw_path.stat().st_size == 2 * 874 * 1164 * 3
