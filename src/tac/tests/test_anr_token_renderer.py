"""Tests for src.tac.anr_token_renderer (ANR full substrate scaffold).

Covers: TokenRendererV62, ShrinkSingleNeRV, HPACMini, FiLMPortabilityGuard,
PPMd codec, train_step contract, export/parse roundtrip, archive grammar.
"""
from __future__ import annotations


import pytest
import torch
import torch.nn as nn

from src.tac.anr_token_renderer import (
    ANR_FORMAT_ID,
    ANR_FORMAT_VERSION,
    ANR_MAGIC,
    ANRTokenRendererConfig,
    ARCHIVE_GRAMMAR,
    CAMERA_H,
    CAMERA_W,
    FEAT_H,
    FEAT_W,
    FiLMPortabilityGuard,
    HPACMini,
    NUM_CLASSES,
    SEGNET_IN_H,
    SEGNET_IN_W,
    ShrinkSingleNeRV,
    TokenRendererV62,
    _eval_roundtrip_uint8_clamp,
    decode_hpac_weights_ppmd,
    encode_hpac_weights_ppmd,
    export_to_archive,
    parse_archive_sections,
    train_step,
)


# ── Constants ────────────────────────────────────────────────────────────


def test_constants_match_pr95():
    """Constants are byte-identical to PR95 inflate.py reference."""
    assert CAMERA_H == 874
    assert CAMERA_W == 1164
    assert SEGNET_IN_H == 384
    assert SEGNET_IN_W == 512
    assert FEAT_H == 6
    assert FEAT_W == 8
    assert NUM_CLASSES == 5
    assert ANR_MAGIC == b"ANRV"
    assert len(ANR_MAGIC) == 4
    assert ANR_FORMAT_ID == 0x50
    assert ANR_FORMAT_VERSION == 1


def test_archive_grammar_well_formed():
    """ARCHIVE_GRAMMAR is the parser-section manifest per HNeRV lesson 2."""
    assert ARCHIVE_GRAMMAR["format_id"] == ANR_FORMAT_ID
    assert ARCHIVE_GRAMMAR["format_version"] == ANR_FORMAT_VERSION
    assert ARCHIVE_GRAMMAR["magic"] == "ANRV"
    assert isinstance(ARCHIVE_GRAMMAR["sections"], list)
    assert len(ARCHIVE_GRAMMAR["sections"]) == 6  # header + 5 sections
    names = [s["name"] for s in ARCHIVE_GRAMMAR["sections"]]
    assert names == ["header", "meta", "master_state", "slave_state",
                     "hpac_state", "tokens"]
    # Header must be fixed size; the field list must match the unpack format.
    header = ARCHIVE_GRAMMAR["sections"][0]
    assert header["kind"] == "fixed_header"
    assert header["length"] == 16


# ── Config ───────────────────────────────────────────────────────────────


def test_config_default_post_init_ok():
    cfg = ANRTokenRendererConfig()
    assert cfg.num_pairs == 600
    assert cfg.num_classes == NUM_CLASSES
    assert cfg.d_film == 8
    assert cfg.slave_d_lat == 6
    assert cfg.slave_channels == (24, 16, 12, 8, 8, 6, 6)
    assert cfg.hpac_P == 32


def test_config_rejects_zero_pairs():
    with pytest.raises(ValueError, match="num_pairs must be positive"):
        ANRTokenRendererConfig(num_pairs=0)


def test_config_rejects_wrong_num_classes():
    with pytest.raises(ValueError, match="num_classes pinned at 5"):
        ANRTokenRendererConfig(num_classes=4)


def test_config_rejects_wrong_slave_channel_count():
    with pytest.raises(ValueError, match="slave_channels must have exactly 7"):
        ANRTokenRendererConfig(slave_channels=(24, 16, 12, 8, 8, 6))


def test_config_rejects_hpac_P_not_divisor():
    with pytest.raises(ValueError, match="hpac_P=.*must divide"):
        ANRTokenRendererConfig(hpac_P=7)


def test_config_rejects_bad_atol():
    with pytest.raises(ValueError, match="film_portability_atol must be positive"):
        ANRTokenRendererConfig(film_portability_atol=0.0)


def test_config_is_frozen():
    cfg = ANRTokenRendererConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.num_pairs = 100  # type: ignore[misc]


# ── TokenRendererV62 ─────────────────────────────────────────────────────


def test_master_forward_shape():
    master = TokenRendererV62(num_pairs=4)
    tokens = torch.zeros(2, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.long)
    idx = torch.tensor([0, 1], dtype=torch.long)
    out = master(tokens, idx)
    assert out.shape == (2, 3, CAMERA_H, CAMERA_W)


def test_master_forward_rejects_non_long_tokens():
    master = TokenRendererV62(num_pairs=4)
    tokens = torch.zeros(2, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.float32)
    idx = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(TypeError, match="tokens must be long"):
        master(tokens, idx)


def test_master_forward_rejects_non_long_idx():
    master = TokenRendererV62(num_pairs=4)
    tokens = torch.zeros(2, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.long)
    idx = torch.tensor([0, 1], dtype=torch.float32)
    with pytest.raises(TypeError, match="idx must be long"):
        master(tokens, idx)


def test_master_forward_rejects_wrong_token_dim():
    master = TokenRendererV62(num_pairs=4)
    tokens = torch.zeros(2, 5, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.long)
    idx = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(ValueError, match=r"tokens must be \(B, H, W\)"):
        master(tokens, idx)


def test_master_forward_rejects_oob_token_value():
    master = TokenRendererV62(num_pairs=4)
    tokens = torch.full((2, SEGNET_IN_H, SEGNET_IN_W), 5, dtype=torch.long)  # 5 = OOB
    idx = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(ValueError, match="tokens max"):
        master(tokens, idx)


def test_master_output_range_pixel():
    """Sigmoid * 255 keeps values in [0, 255]."""
    torch.manual_seed(0)
    master = TokenRendererV62(num_pairs=4)
    tokens = torch.randint(0, NUM_CLASSES, (2, SEGNET_IN_H, SEGNET_IN_W), dtype=torch.long)
    idx = torch.tensor([0, 1], dtype=torch.long)
    out = master(tokens, idx)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 255.0


def test_master_bake_film_table_idempotent():
    master = TokenRendererV62(num_pairs=4)
    master.bake_film_table()
    table1 = master._film_table.clone()
    master.bake_film_table()
    table2 = master._film_table.clone()
    assert torch.equal(table1, table2)


def test_master_baked_vs_live_film_equivalence():
    """After bake, forward with baked table == forward computed live (within fp).

    Sanity: when the same weights are used, the result is bit-identical because
    the bake op uses the same CPU FP32 path.
    """
    torch.manual_seed(0)
    master = TokenRendererV62(num_pairs=4)
    master.eval()
    tokens = torch.randint(0, NUM_CLASSES, (1, SEGNET_IN_H, SEGNET_IN_W), dtype=torch.long)
    idx = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        out_live = master(tokens, idx)
    master.bake_film_table()
    with torch.no_grad():
        out_baked = master(tokens, idx)
    # Same weights → tiny FP drift only.
    diff = (out_live - out_baked).abs().max().item()
    assert diff < 1e-3, f"baked vs live drift {diff:.3e} too large"


# ── ShrinkSingleNeRV ─────────────────────────────────────────────────────


def test_slave_forward_shape():
    slave = ShrinkSingleNeRV(num_pairs=4, d_lat=6, channels=(24, 16, 12, 8, 8, 6, 6))
    idx = torch.tensor([0, 1], dtype=torch.long)
    out = slave(idx)
    assert out.shape == (2, 3, CAMERA_H, CAMERA_W)


def test_slave_rejects_non_long_idx():
    slave = ShrinkSingleNeRV(num_pairs=4)
    idx = torch.tensor([0, 1], dtype=torch.float32)
    with pytest.raises(TypeError, match="idx must be long"):
        slave(idx)


def test_slave_rejects_wrong_channels():
    with pytest.raises(ValueError, match="channels must be 7 entries"):
        ShrinkSingleNeRV(num_pairs=4, channels=(24, 16, 12, 8, 8, 6))


def test_slave_output_range_pixel():
    torch.manual_seed(0)
    slave = ShrinkSingleNeRV(num_pairs=4)
    idx = torch.tensor([0, 1], dtype=torch.long)
    out = slave(idx)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 255.0


def test_slave_param_count_is_compact():
    """Slave should be small — substrate-engineering keeps decoder bytes minimal."""
    slave = ShrinkSingleNeRV(num_pairs=600)
    total = sum(p.numel() for p in slave.parameters())
    # Per PR95: ~10K params (most are codes + per_pair_bias embeddings).
    assert total < 30_000, f"slave param count {total} too large"


# ── HPACMini ─────────────────────────────────────────────────────────────


def test_hpac_forward_shape():
    hpac = HPACMini(num_pairs=4, P=32, delta=2, ch=64, d_film=32)
    tokens = torch.zeros(2, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.long)
    prev = torch.zeros(2, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.long)
    idx = torch.tensor([0, 1], dtype=torch.long)
    hpac.eval()
    with torch.no_grad():
        logits = hpac(tokens, idx, prev)
    assert logits.shape == (2, NUM_CLASSES, SEGNET_IN_H, SEGNET_IN_W)


def test_hpac_rejects_prev_shape_mismatch():
    hpac = HPACMini(num_pairs=4)
    tokens = torch.zeros(2, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.long)
    prev = torch.zeros(2, SEGNET_IN_H // 2, SEGNET_IN_W, dtype=torch.long)
    idx = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(ValueError, match="prev_tokens shape"):
        hpac(tokens, idx, prev)


def test_hpac_rejects_non_long_tokens():
    hpac = HPACMini(num_pairs=4)
    tokens = torch.zeros(2, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.float32)
    prev = torch.zeros(2, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.float32)
    idx = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(TypeError, match="tokens must be long"):
        hpac(tokens, idx, prev)


def test_hpac_with_spm():
    """SPM variant should still produce valid logits."""
    hpac = HPACMini(num_pairs=4, P=32, delta=2, ch=64, d_film=32, use_spm=True)
    tokens = torch.zeros(2, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.long)
    prev = torch.zeros(2, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.long)
    idx = torch.tensor([0, 1], dtype=torch.long)
    hpac.eval()
    with torch.no_grad():
        logits = hpac(tokens, idx, prev)
    assert logits.shape == (2, NUM_CLASSES, SEGNET_IN_H, SEGNET_IN_W)
    assert torch.isfinite(logits).all()


# ── FiLMPortabilityGuard ─────────────────────────────────────────────────


def test_portability_guard_pass_after_bake():
    master = TokenRendererV62(num_pairs=4)
    master.bake_film_table()
    guard = FiLMPortabilityGuard(atol=1e-5)
    guard.check(master)  # no raise


def test_portability_guard_raises_before_bake():
    master = TokenRendererV62(num_pairs=4)
    guard = FiLMPortabilityGuard()
    with pytest.raises(RuntimeError, match="before master.bake_film_table"):
        guard.check(master)


def test_portability_guard_detects_corruption():
    """If someone overwrites the baked table, the guard catches it."""
    master = TokenRendererV62(num_pairs=4)
    master.bake_film_table()
    master._film_table.add_(0.1)  # tamper
    guard = FiLMPortabilityGuard(atol=1e-5)
    with pytest.raises(RuntimeError, match="FiLM portability check failed"):
        guard.check(master)


def test_portability_guard_rejects_bad_atol():
    with pytest.raises(ValueError, match="atol must be positive"):
        FiLMPortabilityGuard(atol=0.0)


def test_portability_guard_detects_shape_mismatch():
    """Direct buffer-size attack: detected before the abs-diff comparison."""
    master = TokenRendererV62(num_pairs=4)
    master.bake_film_table()
    # Replace the buffer with a different-shape tensor (the .copy_ in bake_film_table
    # uses the buffer's storage, so we directly stub it).
    master._film_table = torch.zeros(8, 64)  # wrong shape
    guard = FiLMPortabilityGuard(atol=1e-5)
    with pytest.raises(RuntimeError, match="baked _film_table shape"):
        guard.check(master)


# ── PPMd codec ───────────────────────────────────────────────────────────


def test_ppmd_roundtrip_state_dict():
    sd = {
        "weight": torch.randn(8, 16),
        "bias": torch.zeros(8),
        "scale": torch.ones(8),
    }
    payload = encode_hpac_weights_ppmd(sd)
    assert isinstance(payload, bytes)
    decoded = decode_hpac_weights_ppmd(payload)
    assert set(decoded.keys()) == {"weight", "bias", "scale"}
    assert torch.allclose(decoded["weight"], sd["weight"])
    assert torch.allclose(decoded["bias"], sd["bias"])
    assert torch.allclose(decoded["scale"], sd["scale"])


def test_ppmd_rejects_bad_max_order():
    sd = {"w": torch.zeros(4)}
    with pytest.raises(ValueError, match=r"max_order must be in \[2, 16\]"):
        encode_hpac_weights_ppmd(sd, max_order=1)
    with pytest.raises(ValueError, match=r"max_order must be in \[2, 16\]"):
        encode_hpac_weights_ppmd(sd, max_order=17)


def test_ppmd_rejects_bad_mem_size():
    sd = {"w": torch.zeros(4)}
    with pytest.raises(ValueError, match="mem_size_mb"):
        encode_hpac_weights_ppmd(sd, mem_size_mb=0)


def test_ppmd_decode_rejects_non_bytes():
    with pytest.raises(TypeError, match="payload must be bytes-like"):
        decode_hpac_weights_ppmd("not bytes")  # type: ignore[arg-type]


def test_ppmd_decode_accepts_bytearray():
    sd = {"w": torch.zeros(4)}
    payload = encode_hpac_weights_ppmd(sd)
    decoded = decode_hpac_weights_ppmd(bytearray(payload))
    assert torch.equal(decoded["w"], sd["w"])


# ── Archive export / parse ───────────────────────────────────────────────


def _packed_hpac_state_dict() -> dict:
    """Mock packed HPAC state_dict matching the SCN-baked layout."""
    return {
        "conv_a.weight_q": torch.randint(-127, 128, (64, 7, 7, 7), dtype=torch.int8),
        "conv_a.weight_scale": torch.ones(64) * 0.01,
        "head.weight_q": torch.randint(-127, 128, (5, 64, 1, 1), dtype=torch.int8),
        "head.weight_scale": torch.ones(5) * 0.02,
    }


def test_export_archive_roundtrip():
    cfg = ANRTokenRendererConfig(num_pairs=4)
    master = TokenRendererV62(num_pairs=4, d_film=cfg.d_film)
    slave = ShrinkSingleNeRV(num_pairs=4, d_lat=cfg.slave_d_lat,
                              channels=cfg.slave_channels)
    master.bake_film_table()
    guard = FiLMPortabilityGuard(atol=1e-5)

    tokens_bin = b"\x01\x02\x03" * 100  # mock arithmetic-coded bytes

    blob, sha = export_to_archive(
        config=cfg,
        master=master,
        slave=slave,
        hpac_state_packed=_packed_hpac_state_dict(),
        tokens_bin=tokens_bin,
        portability_guard=guard,
    )
    assert isinstance(blob, bytes)
    assert len(sha) == 64

    sections = parse_archive_sections(blob)
    assert sections["_header"]["magic"] == ANR_MAGIC
    assert sections["_header"]["format_id"] == ANR_FORMAT_ID
    assert sections["_header"]["num_pairs"] == 4
    assert sections["tokens"] == tokens_bin


def test_export_archive_determinism():
    """Same config + weights → same archive bytes (no_op_detector input)."""
    cfg = ANRTokenRendererConfig(num_pairs=4)
    torch.manual_seed(123)
    master = TokenRendererV62(num_pairs=4, d_film=cfg.d_film)
    slave = ShrinkSingleNeRV(num_pairs=4, d_lat=cfg.slave_d_lat,
                              channels=cfg.slave_channels)
    master.bake_film_table()

    tokens_bin = b"x" * 50
    hpac_sd = _packed_hpac_state_dict()
    blob1, sha1 = export_to_archive(
        config=cfg, master=master, slave=slave,
        hpac_state_packed=hpac_sd, tokens_bin=tokens_bin,
    )
    blob2, sha2 = export_to_archive(
        config=cfg, master=master, slave=slave,
        hpac_state_packed=hpac_sd, tokens_bin=tokens_bin,
    )
    assert blob1 == blob2
    assert sha1 == sha2


def test_export_rejects_non_bytes_tokens():
    cfg = ANRTokenRendererConfig(num_pairs=4)
    master = TokenRendererV62(num_pairs=4)
    slave = ShrinkSingleNeRV(num_pairs=4)
    master.bake_film_table()
    with pytest.raises(TypeError, match="tokens_bin must be bytes-like"):
        export_to_archive(
            config=cfg, master=master, slave=slave,
            hpac_state_packed=_packed_hpac_state_dict(),
            tokens_bin="not bytes",  # type: ignore[arg-type]
        )


def test_parse_rejects_short_blob():
    with pytest.raises(ValueError, match="archive too short"):
        parse_archive_sections(b"\x00" * 8)


def test_parse_rejects_bad_magic():
    blob = bytearray(b"XXXX") + b"\x50\x00\x01\x00" + b"\x04\x00\x00\x00" + b"\x00" * 4
    blob += b"\x00\x00\x00\x00"  # zero-length first section so parse can advance
    blob_full = bytes(blob) + b"\x00\x00\x00\x00" * 4  # remaining zero sections
    with pytest.raises(ValueError, match="magic mismatch"):
        parse_archive_sections(blob_full)


def test_parse_rejects_bad_format_id():
    """Header with wrong format_id raises before section walk."""
    import struct as _struct
    bad_header = _struct.pack("<4sHHII", ANR_MAGIC, 0xFF, ANR_FORMAT_VERSION, 4, 0)
    blob = bad_header + b"\x00\x00\x00\x00" * 5
    with pytest.raises(ValueError, match="format_id mismatch"):
        parse_archive_sections(blob)


def test_parse_rejects_truncated_section():
    import struct as _struct
    header = _struct.pack(
        "<4sHHII", ANR_MAGIC, ANR_FORMAT_ID, ANR_FORMAT_VERSION, 4, 0,
    )
    # Claim a long section but provide no body
    blob = header + _struct.pack("<I", 1000) + b"\x00\x00"
    with pytest.raises(ValueError, match="archive truncated inside section"):
        parse_archive_sections(blob)


# ── Eval roundtrip helper ────────────────────────────────────────────────


def test_eval_roundtrip_clamps_and_rounds():
    rgb = torch.tensor([[-10.0, 0.0, 127.4, 127.6, 255.0, 300.0]])
    out = _eval_roundtrip_uint8_clamp(rgb)
    # Forward values are clamped + rounded; gradient flows via STE.
    assert torch.allclose(out, torch.tensor([[0.0, 0.0, 127.0, 128.0, 255.0, 255.0]]))


def test_eval_roundtrip_gradient_flows():
    """The STE allows backprop through the clamp."""
    rgb = torch.tensor([[100.0, 200.0]], requires_grad=True)
    out = _eval_roundtrip_uint8_clamp(rgb)
    out.sum().backward()
    assert rgb.grad is not None
    assert torch.allclose(rgb.grad, torch.tensor([[1.0, 1.0]]))


# ── train_step contract ─────────────────────────────────────────────────


class _MockSegNet(nn.Module):
    """Mock SegNet that mimics the contest scorer contract."""

    def __init__(self):
        super().__init__()
        # tiny linear so gradients can flow
        self.linear = nn.Linear(3 * 384 * 512, 5 * 384 * 512)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, 3, H, W). SegNet uses LAST frame; resize to (384, 512).
        last = x[:, -1, ...]  # (B, 3, H, W)
        from torch.nn import functional as F_
        return F_.interpolate(last, size=(384, 512), mode="bilinear",
                              align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return self.linear(x.reshape(B, -1)).reshape(B, 5, 384, 512)


class _MockPoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3 * 384 * 512 * 2, 6)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, 3, H, W). PoseNet uses both frames; resize each.
        from torch.nn import functional as F_
        B, F_pp, C, H, W = x.shape
        flat = x.reshape(B * F_pp, C, H, W)
        resized = F_.interpolate(flat, size=(384, 512), mode="bilinear",
                                  align_corners=False)
        return resized.reshape(B, F_pp, C, 384, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return self.linear(x.reshape(B, -1))


def test_train_step_rejects_eval_roundtrip_false():
    """CLAUDE.md non-negotiable: eval_roundtrip=False is forbidden."""
    cfg = ANRTokenRendererConfig(num_pairs=2)
    master = TokenRendererV62(num_pairs=2)
    slave = ShrinkSingleNeRV(num_pairs=2)
    tokens = torch.zeros(1, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.long)
    gt = torch.zeros(1, 2, 3, CAMERA_H, CAMERA_W)
    idx = torch.tensor([0], dtype=torch.long)
    with pytest.raises(ValueError, match="eval_roundtrip=False is forbidden"):
        train_step(
            master=master, slave=slave, pair_indices=idx, tokens=tokens,
            gt_pairs_uint8=gt, scorer_seg=_MockSegNet(), scorer_pose=_MockPoseNet(),
            seg_surrogate=lambda a, b: (a - b).pow(2).mean(),
            pose_surrogate=lambda a, b: (a - b).pow(2).mean(),
            lambda_seg=1.0, lambda_pose=1.0, eval_roundtrip=False,
        )


def test_train_step_rejects_wrong_token_shape():
    cfg = ANRTokenRendererConfig(num_pairs=2)
    master = TokenRendererV62(num_pairs=2)
    slave = ShrinkSingleNeRV(num_pairs=2)
    tokens = torch.zeros(1, 50, 50, dtype=torch.long)  # wrong
    gt = torch.zeros(1, 2, 3, CAMERA_H, CAMERA_W)
    idx = torch.tensor([0], dtype=torch.long)
    with pytest.raises(ValueError, match="tokens spatial shape must be"):
        train_step(
            master=master, slave=slave, pair_indices=idx, tokens=tokens,
            gt_pairs_uint8=gt, scorer_seg=_MockSegNet(), scorer_pose=_MockPoseNet(),
            seg_surrogate=lambda a, b: (a - b).pow(2).mean(),
            pose_surrogate=lambda a, b: (a - b).pow(2).mean(),
            lambda_seg=1.0, lambda_pose=1.0,
        )


def test_train_step_rejects_wrong_gt_shape():
    master = TokenRendererV62(num_pairs=2)
    slave = ShrinkSingleNeRV(num_pairs=2)
    tokens = torch.zeros(1, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.long)
    gt = torch.zeros(1, 2, 3, 100, 100)  # wrong
    idx = torch.tensor([0], dtype=torch.long)
    with pytest.raises(ValueError, match="gt_pairs_uint8 spatial shape"):
        train_step(
            master=master, slave=slave, pair_indices=idx, tokens=tokens,
            gt_pairs_uint8=gt, scorer_seg=_MockSegNet(), scorer_pose=_MockPoseNet(),
            seg_surrogate=lambda a, b: (a - b).pow(2).mean(),
            pose_surrogate=lambda a, b: (a - b).pow(2).mean(),
            lambda_seg=1.0, lambda_pose=1.0,
        )


def test_train_step_returns_loss_dict_with_grad():
    """train_step returns a dict; loss has grad_fn for backprop."""
    torch.manual_seed(0)
    master = TokenRendererV62(num_pairs=2)
    slave = ShrinkSingleNeRV(num_pairs=2)
    tokens = torch.zeros(1, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.long)
    gt = torch.full((1, 2, 3, CAMERA_H, CAMERA_W), 128.0)
    idx = torch.tensor([0], dtype=torch.long)
    result = train_step(
        master=master, slave=slave, pair_indices=idx, tokens=tokens,
        gt_pairs_uint8=gt, scorer_seg=_MockSegNet(), scorer_pose=_MockPoseNet(),
        seg_surrogate=lambda a, b: (a - b).pow(2).mean(),
        pose_surrogate=lambda a, b: (a - b).pow(2).mean(),
        lambda_seg=1.0, lambda_pose=1.0,
    )
    assert {"loss", "loss_seg", "loss_pose",
            "loss_seg_unweighted", "loss_pose_unweighted",
            "rendered_uint8_ste"}.issubset(result.keys())
    assert result["loss"].grad_fn is not None
    # The loss is differentiable wrt master + slave params
    result["loss"].backward()
    master_grad = next(p for p in master.parameters() if p.requires_grad).grad
    slave_grad = next(p for p in slave.parameters() if p.requires_grad).grad
    assert master_grad is not None
    assert slave_grad is not None


# ── Sanity: substrate-engineering size budget ───────────────────────────


def test_total_substrate_param_count_in_pr95_band():
    """Master + slave together stay near PR95's reported size band."""
    cfg = ANRTokenRendererConfig(num_pairs=600)
    master = TokenRendererV62(num_pairs=cfg.num_pairs, d_film=cfg.d_film)
    slave = ShrinkSingleNeRV(num_pairs=cfg.num_pairs,
                              d_lat=cfg.slave_d_lat,
                              channels=cfg.slave_channels)
    total = sum(p.numel() for p in master.parameters()) + sum(
        p.numel() for p in slave.parameters()
    )
    # Loose band: master + slave are both small renderers (~10-100K params each).
    assert 5_000 < total < 250_000, f"unexpected total param count: {total}"


def test_no_mps_fallback_default_in_config():
    """Forbidden pattern check: config defaults to cuda_required=True."""
    cfg = ANRTokenRendererConfig()
    assert cfg.cuda_required is True


# ── HNeRV-parity verification ───────────────────────────────────────────


def test_lesson5_full_rgb_renderer_not_mask_only():
    """HNeRV parity lesson 5: output is full RGB to camera resolution."""
    cfg = ANRTokenRendererConfig(num_pairs=2)
    master = TokenRendererV62(num_pairs=2)
    slave = ShrinkSingleNeRV(num_pairs=2)
    tokens = torch.zeros(1, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.long)
    idx = torch.tensor([0], dtype=torch.long)
    master_out = master(tokens, idx)
    slave_out = slave(idx)
    # Both heads must produce (B, 3, CAMERA_H, CAMERA_W); NOT mask-only logits.
    assert master_out.shape == (1, 3, CAMERA_H, CAMERA_W)
    assert slave_out.shape == (1, 3, CAMERA_H, CAMERA_W)
    # Output is uint8 range, not class logits range.
    assert master_out.max().item() <= 255.0
    assert slave_out.max().item() <= 255.0


def test_lesson3_monolithic_archive_single_blob():
    """HNeRV parity lesson 3: single monolithic bytes object, not multi-file."""
    cfg = ANRTokenRendererConfig(num_pairs=2)
    master = TokenRendererV62(num_pairs=2)
    slave = ShrinkSingleNeRV(num_pairs=2)
    master.bake_film_table()
    blob, _ = export_to_archive(
        config=cfg, master=master, slave=slave,
        hpac_state_packed=_packed_hpac_state_dict(),
        tokens_bin=b"\x00" * 10,
    )
    assert isinstance(blob, bytes)
    # All sections fit in a single contiguous bytes object.
    sections = parse_archive_sections(blob)
    assert "_header" in sections
    assert "tokens" in sections


def test_lesson11_no_op_detector_sha_uniqueness():
    """HNeRV parity lesson 11: differentially-changed inputs produce different SHAs."""
    cfg = ANRTokenRendererConfig(num_pairs=2)
    master = TokenRendererV62(num_pairs=2)
    slave = ShrinkSingleNeRV(num_pairs=2)
    master.bake_film_table()

    sd = _packed_hpac_state_dict()
    blob_a, sha_a = export_to_archive(
        config=cfg, master=master, slave=slave,
        hpac_state_packed=sd, tokens_bin=b"AAAA",
    )
    blob_b, sha_b = export_to_archive(
        config=cfg, master=master, slave=slave,
        hpac_state_packed=sd, tokens_bin=b"BBBB",
    )
    assert sha_a != sha_b, "different inputs must produce different archive sha"


def test_lesson8_eval_roundtrip_default_true():
    """HNeRV parity lesson 8: eval_roundtrip defaults True in train_step signature."""
    import inspect
    sig = inspect.signature(train_step)
    assert sig.parameters["eval_roundtrip"].default is True
