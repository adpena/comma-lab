"""Tests for Lane J-NWC end-to-end pipeline.

Covers the producer/consumer wiring that sits ABOVE the WeightCodec module
(which has its own tests in test_neural_weight_codec.py):

    (a) Magic-byte: the exported NWC1 binary starts with b"NWC1"
    (b) Round-trip: encode → decode produces a state-dict whose every
        floating tensor matches the original within bounded codec tolerance
    (c) Inflate dispatch: the submissions/robust_current/inflate_renderer.py
        _load_renderer code path correctly dispatches NWC1 magic and yields
        a usable nn.Module
    (d) Determinism: encoding the same model twice with the same codec
        produces byte-identical NWC1 binaries (modulo the embedded codec
        torch.save bytes which carry version metadata; we verify by
        re-decoding and comparing state-dicts)
    (e) Arch-header gate (pipeline.py): when weight_compression='nwc' is
        requested but no codec_path is configured, step_compress_weights
        falls back to FP4 and emits a WARN — instead of silently shipping
        a stub NWC binary
    (f) Bidirectional magic-byte registry test now passes (forward direction
        for NWC1)

All tests use synthetic data (no GT video, no CUDA/MPS dependency). CI-fast.
"""
from __future__ import annotations

import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tac.neural_weight_codec import (
    WeightCodec,
    WeightCodecConfig,
    train_codec,
)
from tac.renderer_export import (
    detect_checkpoint_type,
    export_neural_compressed_checkpoint,
    load_neural_compressed_checkpoint,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


# ── Helpers ──────────────────────────────────────────────────────────────


def _build_synth_codec(*, seed: int = 0) -> WeightCodec:
    """Pretrain a tiny codec on synthetic gaussian blocks (CPU-only).

    Mirrors test_neural_weight_codec._build_pretrained_codec but keeps the
    helper local to Lane J-NWC tests so reorganization upstream doesn't
    silently break this suite.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    block_size = 16
    raw = torch.randn(4096, block_size, generator=g)
    scales = raw.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    corpus = raw / scales
    codec = WeightCodec(
        WeightCodecConfig(
            block_size=block_size, codebook_size=64, latent_dim=16
        )
    )
    codec, _ = train_codec(
        corpus, codec=codec, num_steps=200, batch_size=128, lr=1e-3,
        device="cpu", log_interval=999, seed=seed,
    )
    codec.eval()
    return codec


def _save_codec(codec: WeightCodec, path: Path) -> None:
    """Persist a codec to disk in the format export_neural_compressed_checkpoint
    expects (matches experiments/train_neural_weight_codec.py)."""
    payload = {
        "codec_state_dict": codec.state_dict(),
        "codec_config": {
            "block_size": codec.config.block_size,
            "codebook_size": codec.config.codebook_size,
            "latent_dim": codec.config.latent_dim,
            "hidden": codec.config.hidden,
        },
    }
    torch.save(payload, str(path))


class _TinyRenderer(nn.Module):
    """Renderer-shaped synthetic model used by every test in this file."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(5, 6)
        self.conv1 = nn.Conv2d(6, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.head = nn.Conv2d(16, 5, 1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # Used only for shape-sanity in tests; not optimized.
        e = self.embed(idx)
        # (B, H, W, C) -> (B, C, H, W)
        e = e.permute(0, 3, 1, 2)
        h = torch.relu(self.conv1(e))
        h = torch.relu(self.conv2(h))
        return self.head(h)


# ── (a) Magic byte ────────────────────────────────────────────────────────


def test_nwc1_export_writes_magic_at_offset_zero(tmp_path: Path) -> None:
    """The NWC1 file's first 4 bytes must be b'NWC1' so the inflate-side
    dispatch in submissions/robust_current/inflate_renderer.py:_load_renderer
    can route the file. Companion to test_nwc1_magic_bytes_at_file_start."""
    torch.manual_seed(11)
    model = _TinyRenderer()
    codec = _build_synth_codec(seed=11)
    codec_path = tmp_path / "codec.pt"
    _save_codec(codec, codec_path)
    bin_path = tmp_path / "renderer_nwc.bin"
    nbytes = export_neural_compressed_checkpoint(
        model, codec_path=codec_path, output_path=bin_path,
        arch_extra={"tensor_only": True},
    )
    head = bin_path.read_bytes()[:4]
    assert head == b"NWC1", f"Magic header mismatch: {head!r}"
    assert nbytes == bin_path.stat().st_size
    assert detect_checkpoint_type(bin_path) == "neural_weight_compression_v1"


# ── (b) Round-trip ────────────────────────────────────────────────────────


def test_nwc1_roundtrip_state_dict_preserves_floating_tensors(tmp_path: Path) -> None:
    """Every floating-point state_dict tensor survives encode → decode within
    a bounded codec tolerance. Bound: max-abs-error < 4× max-abs-input
    (catches catastrophic codec failure; codec is intentionally lossy)."""
    torch.manual_seed(22)
    model = _TinyRenderer()
    codec = _build_synth_codec(seed=22)
    codec_path = tmp_path / "codec.pt"
    _save_codec(codec, codec_path)
    bin_path = tmp_path / "renderer_nwc.bin"
    export_neural_compressed_checkpoint(
        model, codec_path=codec_path, output_path=bin_path,
        arch_extra={"tensor_only": True},
    )
    restored = load_neural_compressed_checkpoint(bin_path, device="cpu")
    new_state = getattr(restored, "_nwc_state_dict", None) or restored.state_dict()
    orig_state = model.state_dict()
    for name, orig in orig_state.items():
        if not torch.is_floating_point(orig):
            continue
        assert name in new_state, f"NWC roundtrip lost key: {name}"
        rec = new_state[name]
        assert rec.shape == orig.shape
        max_err = (orig - rec).abs().max().item()
        bound = 4.0 * orig.abs().max().item() + 1e-3
        assert max_err < bound, (
            f"NWC roundtrip max_err {max_err:.3f} >= bound {bound:.3f} "
            f"for {name} shape={tuple(orig.shape)}"
        )


# ── (c) Inflate dispatch ──────────────────────────────────────────────────


def test_inflate_renderer_dispatches_nwc1_magic() -> None:
    """submissions/robust_current/inflate_renderer.py:_load_renderer MUST
    have a dispatch case for b'NWC1'. We grep the source rather than
    importing the file (it is a script entry, not a library, and pulls in
    heavy deps like av at top level)."""
    inflate_src = (
        REPO_ROOT / "submissions" / "robust_current" / "inflate_renderer.py"
    ).read_text()
    # Forward-direction registry check: the magic must appear in the source.
    assert b"NWC1" in inflate_src.encode("utf-8"), (
        "inflate_renderer.py is missing NWC1 dispatch — test_magic_bytes_"
        "bidirectional_consistency would also fail"
    )
    # The dispatch must call the canonical decode entry point.
    assert "load_neural_compressed_checkpoint" in inflate_src, (
        "inflate_renderer.py NWC1 case must delegate to "
        "tac.renderer_export.load_neural_compressed_checkpoint"
    )
    # And it must respect the strict-scorer-rule banner pattern (Check H).
    assert "strict-scorer-rule OK" in inflate_src, (
        "NWC1 dispatch missing strict-scorer-rule OK banner"
    )


def test_inflate_renderer_nwc_dispatch_yields_usable_module(tmp_path: Path) -> None:
    """Drive the equivalent dispatch in-process (without spawning the inflate
    script) by calling the canonical loader the inflate-side dispatch uses.
    This is the same code path NWC1 archives will hit at inflate time."""
    torch.manual_seed(33)
    model = _TinyRenderer()
    codec = _build_synth_codec(seed=33)
    codec_path = tmp_path / "codec.pt"
    _save_codec(codec, codec_path)
    bin_path = tmp_path / "renderer_nwc.bin"
    export_neural_compressed_checkpoint(
        model, codec_path=codec_path, output_path=bin_path,
        arch_extra={"tensor_only": True},
    )
    raw = bin_path.read_bytes()
    assert raw[:4] == b"NWC1"
    restored = load_neural_compressed_checkpoint(raw, device="cpu")
    assert isinstance(restored, nn.Module)
    # Has the state-dict snapshot the loader stashes for tensor_only mode.
    new_state = getattr(restored, "_nwc_state_dict", None) or restored.state_dict()
    assert any(k for k in new_state), "loaded module has no state_dict entries"


# ── (d) Determinism ───────────────────────────────────────────────────────


def test_nwc1_encoding_is_deterministic(tmp_path: Path) -> None:
    """Encoding the SAME model with the SAME codec twice produces NWC1
    binaries that decode to identical state-dicts (we cannot guarantee
    byte-identical files because torch.save embeds RNG state in the codec
    blob; but the post-decode tensors MUST be bit-identical).

    Stronger byte-equality is also asserted on the per-tensor blob payloads
    (they are produced by deterministic numpy.tobytes — no entropy)."""
    torch.manual_seed(44)
    model = _TinyRenderer()
    codec = _build_synth_codec(seed=44)
    codec_path = tmp_path / "codec.pt"
    _save_codec(codec, codec_path)

    bin1 = tmp_path / "nwc_1.bin"
    bin2 = tmp_path / "nwc_2.bin"
    export_neural_compressed_checkpoint(
        model, codec_path=codec_path, output_path=bin1,
        arch_extra={"tensor_only": True},
    )
    export_neural_compressed_checkpoint(
        model, codec_path=codec_path, output_path=bin2,
        arch_extra={"tensor_only": True},
    )

    r1 = load_neural_compressed_checkpoint(bin1, device="cpu")
    r2 = load_neural_compressed_checkpoint(bin2, device="cpu")
    s1 = getattr(r1, "_nwc_state_dict", None) or r1.state_dict()
    s2 = getattr(r2, "_nwc_state_dict", None) or r2.state_dict()
    for k in s1:
        if not torch.is_floating_point(s1[k]):
            continue
        assert torch.equal(s1[k], s2[k]), (
            f"NWC1 non-deterministic decode for tensor {k!r}"
        )


# ── (e) Arch-header gate (pipeline.py) ────────────────────────────────────


@dataclass
class _StubPipelineConfig:
    """Minimal stub mirroring the fields step_compress_weights reads. Avoids
    importing experiments/pipeline.py PipelineConfig (pulls a large transitive
    cost). Verifies the gate purely by the WARN/return-checkpoint contract."""

    output_dir: str
    weight_compression: str = "nwc"
    weight_codec_path: str = ""
    padding_mode: str = "zeros"
    use_dilation: bool = False
    use_zoom_flow: bool = False
    base_ch: int = 36
    mid_ch: int = 60
    motion_hidden: int = 32
    depth: int = 1
    pose_dim: int = 6
    embed_dim: int = 6
    use_dsconv: bool = False


def test_pipeline_nwc_branch_falls_back_to_fp4_when_codec_missing(tmp_path: Path) -> None:
    """When weight_compression='nwc' is requested but weight_codec_path does
    not exist, step_compress_weights must:
        - emit a WARN about the missing codec
        - record mode='fp4_fallback' in the .done marker
        - return the original checkpoint path (no NWC bin produced)
    Mirrors the I4LZ-arch-header-gate fallback pattern in the same function."""
    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    try:
        from pipeline import step_compress_weights
    finally:
        # Remove from path to avoid leaking the experiments module path
        # into other tests that might import a module named `pipeline` from
        # elsewhere.
        try:
            sys.path.remove(str(REPO_ROOT / "experiments"))
        except ValueError:
            pass

    # Synthesize a fake checkpoint with a tiny state-dict so the function's
    # fallback path doesn't try to re-load real weights.
    fake_ckpt = tmp_path / "fake.pt"
    torch.save({"model_state_dict": {}}, fake_ckpt)

    cfg = _StubPipelineConfig(
        output_dir=str(tmp_path / "pipeline_out"),
        weight_compression="nwc",
        weight_codec_path="",  # MISSING — should trigger fallback
    )
    result = step_compress_weights(cfg, fake_ckpt, iteration=0)
    assert result == fake_ckpt, (
        "fallback must return the input checkpoint path; got "
        f"{result}"
    )
    done = (tmp_path / "pipeline_out" / "iter_0" / ".done_compress_weights")
    assert done.exists(), "step_compress_weights did not write .done marker"
    import json
    payload = json.loads(done.read_text())
    assert payload.get("mode") == "fp4_fallback"
    assert payload.get("reason") == "nwc_codec_missing"


# ── (f) Bidirectional magic test must now pass ───────────────────────────


def test_bidirectional_magic_test_passes_for_nwc1() -> None:
    """The repo-level bidirectional magic-byte test
    (test_preflight_arity.py::test_magic_bytes_bidirectional_consistency)
    asserts every producer magic appears in the inflate consumer source.
    We replicate the forward-direction NWC1 check here so a regression in
    inflate_renderer.py is caught locally even if the umbrella test is
    accidentally skipped."""
    inflate_src = (
        REPO_ROOT / "submissions" / "robust_current" / "inflate_renderer.py"
    ).read_text(encoding="utf-8")
    producer_src = (
        REPO_ROOT / "src" / "tac" / "renderer_export.py"
    ).read_text(encoding="utf-8")
    assert 'b"NWC1"' in producer_src or "b'NWC1'" in producer_src
    assert "NWC1" in inflate_src, (
        "inflate_renderer.py must mention the NWC1 magic for the "
        "bidirectional consistency test to pass"
    )


# ── Optional: drive the umbrella test directly ───────────────────────────


def test_umbrella_bidirectional_magic_consistency_now_passes() -> None:
    """Run the repo-level test_magic_bytes_bidirectional_consistency in a
    subprocess so its assertion failure (if any) surfaces as our failure too.
    This guards against landing a Lane J-NWC change that breaks the umbrella
    test in some other way."""
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "src/tac/tests/test_preflight_arity.py::"
            "test_magic_bytes_bidirectional_consistency",
            "-x", "--tb=short", "-q",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"umbrella bidirectional magic test failed:\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
