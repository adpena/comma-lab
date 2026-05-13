"""Regression tests for qat_finetune.py FP4 export/load round-trip.

Companion to test_qat_finetune_loader.py (which covers .pt loader Bug 1).
This file covers Bug 2: load_asymmetric_checkpoint_fp4 must construct the
ACTUAL arch saved in the binary, not a hybrid arch from CLI defaults.

Bug 2 history (memory feedback_qat_finetune_chained_arch_bugs, 2026-04-27):

  qat_finetune.py:995 calls load_asymmetric_checkpoint_fp4 to verify the
  FP4 export round-trip. Before the fix, the loader routed through
  build_renderer(use_zoom_flow=False) which constructs a LEGACY
  PairGenerator + MotionPredictor with default output_channels=2 →
  motion.head shape [2, 32, 3, 3]. But the dilated-h64 baseline ships in
  ASYM/FP4A format with motion.head shape [6, 32, 3, 3] (AsymmetricPair-
  Generator(use_zoom_flow=False) sets motion_output_channels=6). The
  shape mismatch crashed at module.weight.copy_(flat.reshape(shape)) in
  renderer_export.py:1418 with `shape '[6, 32, 3, 3]' is invalid for
  input of size 576`.

The fix adds `pair_mode` to the header (via _infer_asymmetric_config) and
dispatches in the loader: pair_mode="asymmetric" → AsymmetricPairGenerator,
else → build_renderer (legacy PairGenerator path). Older binaries
(without pair_mode in header) default to "asymmetric" since every
existing ASYM/FP4A binary actually was AsymmetricPairGenerator.

Tests cover:
1. Full FP4 round-trip on the dilated-h64 arch (Bug 2 reproducer).
2. Output shapes match input arch after round-trip.
3. Forward pass produces the expected (B, 2, H, W, 3) HWC pair format.
4. pair_mode field is correctly populated in the header by the exporter.
5. Loader correctly dispatches to AsymmetricPairGenerator vs PairGenerator
   based on header's pair_mode.
6. Backward-compat: a header WITHOUT pair_mode (older binary) defaults
   to "asymmetric" — never to PairGenerator (which would silently corrupt
   shapes).
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
for sub in ("src", "upstream", "."):
    p = str(REPO / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from tac.renderer import AsymmetricPairGenerator, PairGenerator, build_renderer  # noqa: E402
from tac.renderer_export import (  # noqa: E402
    export_asymmetric_checkpoint,
    export_asymmetric_checkpoint_fp4,
    load_asymmetric_checkpoint,
    load_asymmetric_checkpoint_fp4,
)


# ── Architecture: matches the verified 0.9001 dilated-h64 baseline ─────
# (header inspected from submissions/baseline_dilated_h64_0_90/renderer.bin)
DILATED_H64_ARCH = dict(
    num_classes=5,
    embed_dim=6,
    base_ch=36,
    mid_ch=60,
    motion_hidden=32,
    depth=1,
    pose_dim=6,
    use_dsconv=False,
    padding_mode="zeros",
    use_dilation=False,
    use_zoom_flow=False,  # baseline has motion_output_channels=6
)


def _make_dilated_h64() -> AsymmetricPairGenerator:
    """Construct the dilated-h64 ASYM arch byte-for-byte from baseline."""
    return AsymmetricPairGenerator(**DILATED_H64_ARCH)


def _state_shapes(model: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def _smoke_forward(model: torch.nn.Module) -> tuple[int, ...]:
    """Run a tiny forward pass to verify the model is structurally sound.

    Returns the output shape for assertion. Uses small H, W to keep the
    test fast — 32x48 is divisible by stride patterns.
    """
    H, W = 32, 48
    mask_t = torch.randint(0, 5, (1, H, W), dtype=torch.long)
    mask_t1 = torch.randint(0, 5, (1, H, W), dtype=torch.long)
    pose = torch.zeros(1, 6) if DILATED_H64_ARCH["pose_dim"] == 6 else None
    model.eval()
    with torch.no_grad():
        # AsymmetricPairGenerator.forward signature: (mask_t, mask_t1, ...)
        out = model(mask_t, mask_t1, pose=pose) if pose is not None else model(mask_t, mask_t1)
    return tuple(out.shape)


# ── Tests ──────────────────────────────────────────────────────────────


def test_dilated_h64_fp4_roundtrip_preserves_arch():
    """Bug 2 reproducer: round-trip the dilated-h64 baseline arch through
    FP4 export → load. Before the fix, the loader built a hybrid arch
    (PairGenerator + MotionPredictor with output_channels=2) and crashed
    at flat.reshape(shape) trying to fit a [6, 32, 3, 3] blob into a
    [2, 32, 3, 3] module."""
    src = _make_dilated_h64()
    src.eval()

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        fp4_path = Path(f.name)
    try:
        nbytes = export_asymmetric_checkpoint_fp4(src, fp4_path)
        assert nbytes > 0
        loaded = load_asymmetric_checkpoint_fp4(fp4_path.read_bytes(), device="cpu")
    finally:
        fp4_path.unlink(missing_ok=True)

    # Same class — not a hybrid arch.
    assert isinstance(loaded, AsymmetricPairGenerator), (
        f"Loaded class {type(loaded).__name__} != AsymmetricPairGenerator. "
        f"Bug 2 regressed: loader built a hybrid arch instead of the "
        f"actual saved arch."
    )

    # Same state_dict shape map. ANY mismatch is a Bug 2 regression.
    src_shapes = _state_shapes(src)
    loaded_shapes = _state_shapes(loaded)
    assert src_shapes.keys() == loaded_shapes.keys(), (
        f"state_dict keys differ.\n"
        f"  only in src: {src_shapes.keys() - loaded_shapes.keys()}\n"
        f"  only in loaded: {loaded_shapes.keys() - src_shapes.keys()}"
    )
    mismatched = {
        k: (src_shapes[k], loaded_shapes[k])
        for k in src_shapes
        if src_shapes[k] != loaded_shapes[k]
    }
    assert not mismatched, (
        f"Shape mismatch on {len(mismatched)} keys after FP4 round-trip:\n"
        + "\n".join(f"  {k}: src={s} vs loaded={l}" for k, (s, l) in mismatched.items())
    )


def test_dilated_h64_fp4_roundtrip_motion_head_shape_6():
    """Sharper assertion of the exact field that crashed Lane B: motion.head
    bias must be shape [6] (AsymmetricPairGenerator with use_zoom_flow=False),
    NOT shape [2] (PairGenerator default MotionPredictor)."""
    src = _make_dilated_h64()
    assert src.motion.head.bias.shape == (6,), (
        "Test setup wrong: AsymmetricPairGenerator(use_zoom_flow=False) "
        "should have motion.head.bias shape [6]. Got "
        f"{tuple(src.motion.head.bias.shape)}."
    )

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        fp4_path = Path(f.name)
    try:
        export_asymmetric_checkpoint_fp4(src, fp4_path)
        loaded = load_asymmetric_checkpoint_fp4(fp4_path.read_bytes(), device="cpu")
    finally:
        fp4_path.unlink(missing_ok=True)

    assert loaded.motion.head.bias.shape == (6,), (
        f"Bug 2 regression: motion.head.bias loaded as "
        f"{tuple(loaded.motion.head.bias.shape)}, expected (6,). "
        "Loader is building PairGenerator instead of AsymmetricPair-"
        "Generator for use_zoom_flow=False."
    )


def test_dilated_h64_fp4_forward_pass_runs():
    """End-to-end: round-trip through FP4 and then run a forward pass.
    This catches dynamic shape errors that static state_dict comparisons
    miss (e.g. flow channel slicing in PairGenerator.forward)."""
    src = _make_dilated_h64()

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        fp4_path = Path(f.name)
    try:
        export_asymmetric_checkpoint_fp4(src, fp4_path)
        loaded = load_asymmetric_checkpoint_fp4(fp4_path.read_bytes(), device="cpu")
    finally:
        fp4_path.unlink(missing_ok=True)

    out_shape = _smoke_forward(loaded)
    # AsymmetricPairGenerator/PairGenerator output is HWC pair: (B, 2, H, W, 3)
    assert out_shape == (1, 2, 32, 48, 3), (
        f"Forward pass produced unexpected shape {out_shape}, "
        "expected (1, 2, 32, 48, 3) HWC pair format."
    )


def test_dilated_h64_asym_roundtrip_preserves_arch():
    """Same Bug 2 surface for the non-FP4 ASYM exporter/loader. Both
    code paths share _infer_asymmetric_config and must dispatch on
    pair_mode the same way."""
    src = _make_dilated_h64()

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        bin_path = Path(f.name)
    try:
        export_asymmetric_checkpoint(src, bin_path, default_bits=8)
        loaded = load_asymmetric_checkpoint(bin_path.read_bytes(), device="cpu")
    finally:
        bin_path.unlink(missing_ok=True)

    assert isinstance(loaded, AsymmetricPairGenerator)
    assert loaded.motion.head.bias.shape == (6,)


def test_pair_mode_field_present_in_fp4_header():
    """The exporter must write pair_mode into the header. Without it,
    older loaders fall back to 'asymmetric' (correct for AsymmetricPair-
    Generator, the only class actually exported today)."""
    src = _make_dilated_h64()

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        fp4_path = Path(f.name)
    try:
        export_asymmetric_checkpoint_fp4(src, fp4_path)
        data = fp4_path.read_bytes()
    finally:
        fp4_path.unlink(missing_ok=True)

    # Parse header: [magic 4B][header_len 4B][header JSON]
    assert data[:4] == b"FP4A"
    header_len = struct.unpack("<I", data[4:8])[0]
    header = json.loads(data[8:8 + header_len].decode("utf-8"))
    assert header["pair_mode"] == "asymmetric", (
        f"Exporter must write pair_mode='asymmetric' for "
        f"AsymmetricPairGenerator. Got {header.get('pair_mode')!r}."
    )
    # Sanity: arch fields needed for the loader are all present.
    for key in (
        "embed_dim", "base_ch", "mid_ch", "motion_hidden", "depth",
        "pose_dim", "use_dsconv", "padding_mode", "use_dilation",
        "use_zoom_flow", "num_classes", "max_flow_px", "max_residual",
        "flow_only", "output_channels",
    ):
        assert key in header, f"Header missing arch field {key!r}"


def test_pair_mode_field_present_in_asym_header():
    """Same check for the non-FP4 ASYM exporter."""
    src = _make_dilated_h64()

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        bin_path = Path(f.name)
    try:
        export_asymmetric_checkpoint(src, bin_path, default_bits=8)
        data = bin_path.read_bytes()
    finally:
        bin_path.unlink(missing_ok=True)

    assert data[:4] == b"ASYM"
    header_len = struct.unpack("<I", data[4:8])[0]
    header = json.loads(data[8:8 + header_len].decode("utf-8"))
    assert header["pair_mode"] == "asymmetric"


def test_legacy_pair_generator_pair_mode_label():
    """A vanilla PairGenerator (use_zoom_flow=False, build_renderer legacy
    path) must export pair_mode='pair_generator', not 'asymmetric'. Without
    this distinction the loader silently builds the wrong class — the
    latent-bug counterpart of Bug 2."""
    legacy = build_renderer(
        num_classes=5, embed_dim=6, base_ch=36, mid_ch=60,
        motion_hidden=32, depth=1, pose_dim=0,
        use_dsconv=False, padding_mode="zeros", use_dilation=False,
        use_zoom_flow=False,
    )
    assert isinstance(legacy, PairGenerator)
    assert not isinstance(legacy, AsymmetricPairGenerator)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        bin_path = Path(f.name)
    try:
        export_asymmetric_checkpoint(legacy, bin_path, default_bits=8)
        data = bin_path.read_bytes()
    finally:
        bin_path.unlink(missing_ok=True)

    header_len = struct.unpack("<I", data[4:8])[0]
    header = json.loads(data[8:8 + header_len].decode("utf-8"))
    assert header["pair_mode"] == "pair_generator", (
        f"Vanilla PairGenerator must export pair_mode='pair_generator'. "
        f"Got {header.get('pair_mode')!r}. If this is 'asymmetric' the "
        "loader will build AsymmetricPairGenerator → motion.head shape "
        "mismatch [2] vs [6]."
    )


def test_loader_dispatches_on_pair_mode_pair_generator():
    """With pair_mode='pair_generator' the loader must construct legacy
    PairGenerator (motion.head bias [2]), not AsymmetricPairGenerator
    (motion.head bias [6])."""
    legacy = build_renderer(
        num_classes=5, embed_dim=6, base_ch=36, mid_ch=60,
        motion_hidden=32, depth=1, pose_dim=0,
        use_dsconv=False, padding_mode="zeros", use_dilation=False,
        use_zoom_flow=False,
    )
    assert legacy.motion.head.bias.shape == (2,), (
        "Test setup wrong: PairGenerator default MotionPredictor should "
        "have motion.head.bias shape [2]."
    )

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        bin_path = Path(f.name)
    try:
        export_asymmetric_checkpoint(legacy, bin_path, default_bits=8)
        loaded = load_asymmetric_checkpoint(bin_path.read_bytes(), device="cpu")
    finally:
        bin_path.unlink(missing_ok=True)

    assert isinstance(loaded, PairGenerator)
    assert not isinstance(loaded, AsymmetricPairGenerator)
    assert loaded.motion.head.bias.shape == (2,)


def test_loader_backward_compat_missing_pair_mode_defaults_to_asymmetric():
    """Older binaries (pre-fix exporter) didn't carry pair_mode in the
    header. The loader must default to 'asymmetric' since every existing
    ASYM/FP4A binary actually was AsymmetricPairGenerator (the exporter
    hardcoded pair_mode='asymmetric' regardless of true class — a latent
    bug for PairGenerator that never fired because nobody exported one).

    This test forges a header without pair_mode and verifies the loader
    constructs AsymmetricPairGenerator. Critical for loading the
    submitted dilated-h64 baseline binaries that exist on disk today.
    """
    src = _make_dilated_h64()

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        fp4_path = Path(f.name)
    try:
        export_asymmetric_checkpoint_fp4(src, fp4_path)
        data = bytearray(fp4_path.read_bytes())
    finally:
        fp4_path.unlink(missing_ok=True)

    # Surgically strip pair_mode from the header to simulate an old binary.
    assert data[:4] == b"FP4A"
    header_len = struct.unpack("<I", data[4:8])[0]
    header = json.loads(data[8:8 + header_len].decode("utf-8"))
    assert header.get("pair_mode") == "asymmetric"  # baseline state
    header.pop("pair_mode")

    # Re-encode with the same blob layout but a stripped header.
    new_header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blobs_offset = 8 + header_len
    blobs = bytes(data[blobs_offset:])
    forged = bytearray()
    forged.extend(b"FP4A")
    forged.extend(struct.pack("<I", len(new_header_json)))
    forged.extend(new_header_json)
    forged.extend(blobs)

    loaded = load_asymmetric_checkpoint_fp4(bytes(forged), device="cpu")
    assert isinstance(loaded, AsymmetricPairGenerator), (
        "Backward-compat broken: loader must default to AsymmetricPair-"
        "Generator when header has no pair_mode field. Without this, "
        "every existing dilated-h64 baseline binary on disk would crash."
    )
    assert loaded.motion.head.bias.shape == (6,)


def test_use_zoom_flow_true_roundtrip():
    """Cover the use_zoom_flow=True branch: AsymmetricPairGenerator with
    motion_output_channels=4. Motion head bias must be shape [4] in
    both src and loaded."""
    arch = dict(DILATED_H64_ARCH)
    arch["use_zoom_flow"] = True
    src = AsymmetricPairGenerator(**arch)
    assert src.motion.head.bias.shape == (4,), (
        "Test setup: use_zoom_flow=True should give motion.head.bias [4]."
    )

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        fp4_path = Path(f.name)
    try:
        export_asymmetric_checkpoint_fp4(src, fp4_path)
        loaded = load_asymmetric_checkpoint_fp4(fp4_path.read_bytes(), device="cpu")
    finally:
        fp4_path.unlink(missing_ok=True)

    assert isinstance(loaded, AsymmetricPairGenerator)
    assert loaded.motion.head.bias.shape == (4,)
    assert loaded.use_zoom_flow is True


@pytest.mark.skipif(
    not (REPO / "submissions/baseline_dilated_h64_0_90/renderer.bin").exists(),
    reason="dilated-h64 baseline binary not present in this checkout",
)
def test_real_dilated_h64_baseline_fp4_roundtrip():
    """Integration test: load the actual on-disk dilated-h64 baseline
    .bin (verified 0.9001 score), re-export as FP4, and verify the
    round-trip preserves shape + forward pass works. This is the
    Lane F gate — if this fails, Lane F cannot launch."""
    baseline_path = REPO / "submissions/baseline_dilated_h64_0_90/renderer.bin"
    src = load_asymmetric_checkpoint(str(baseline_path), device="cpu")
    assert isinstance(src, AsymmetricPairGenerator)
    src_shapes = _state_shapes(src)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        fp4_path = Path(f.name)
    try:
        export_asymmetric_checkpoint_fp4(src, fp4_path)
        loaded = load_asymmetric_checkpoint_fp4(fp4_path.read_bytes(), device="cpu")
    finally:
        fp4_path.unlink(missing_ok=True)

    loaded_shapes = _state_shapes(loaded)
    assert src_shapes == loaded_shapes, (
        "Real baseline FP4 round-trip arch mismatch — Lane F is BLOCKED."
    )

    # Forward pass smoke. Use small H, W to keep test fast (the renderer
    # is fully convolutional so any divisible shape works).
    out_shape = _smoke_forward(loaded)
    assert out_shape == (1, 2, 32, 48, 3)


# ════════════════════════════════════════════════════════════════════════════
# Lane F regression tests (post-2026-04-27 forensic council)
# ════════════════════════════════════════════════════════════════════════════
#
# These tests pin the Bug 1 fix from findings.md "## 2026-04-27 Council audit:
# Lane F regression — bugged or dead?" — qat_finetune.py MUST have an explicit
# `--poses` CLI argument and MUST raise SystemExit (not silently warn) when
# pose_dim>0 but --poses is missing.
#
# History: Lane F v1 silently fell back to zero poses when --poses was not
# threaded, then deployed against real poses → +58% PoseNet regression
# misreported as "FP4 quantization is dead." The structural fix is the
# explicit --poses arg + raise. This is the test gate that prevents regression.


import subprocess
import sys


def test_qat_finetune_argparse_has_poses_flag():
    """The argparse parser of qat_finetune.py MUST register --poses.
    Mirrors the test pattern from test_train_renderer_auth_eval_wiring.py
    (CLAUDE.md NEVER invent CLI flags rule).
    """
    qat_path = REPO / "experiments" / "qat_finetune.py"
    assert qat_path.exists(), f"qat_finetune.py missing: {qat_path}"
    text = qat_path.read_text()
    assert '"--poses"' in text, (
        "qat_finetune.py argparse missing --poses flag. "
        "Required per findings.md 'Lane F regression' (2026-04-27). "
        "Bug 1 fix has regressed."
    )


def test_qat_finetune_argparse_has_poses_help_text_warning():
    """The --poses help text must mention that auto-discovery is intentionally
    NOT done (so a future agent reading the help doesn't reintroduce the bug)."""
    qat_path = REPO / "experiments" / "qat_finetune.py"
    text = qat_path.read_text()
    # The help text should be substantive (not a placeholder)
    assert "Auto-discovery" in text or "auto-discovery" in text, (
        "--poses help text should warn that auto-discovery is intentionally "
        "NOT done (per CLAUDE.md FORBIDDEN PATTERNS)."
    )
    assert "FORBIDDEN PATTERNS" in text or "explicit > implicit" in text, (
        "--poses help text should reference CLAUDE.md FORBIDDEN PATTERNS."
    )


def test_qat_finetune_raises_on_missing_poses_for_pose_dim_gt_0():
    """Subprocess test: qat_finetune.py MUST exit non-zero when checkpoint has
    pose_dim>0 and --poses is not provided. This is the falsifiable assertion
    that the silent-WARN fallback is gone.

    We use a real on-disk pose_dim>0 checkpoint (the dilated-h64 baseline),
    which is the same one Lane F v1 catastrophically miscompiled."""
    baseline = REPO / "submissions/baseline_dilated_h64_0_90/renderer.bin"
    if not baseline.exists():
        pytest.skip("dilated-h64 baseline binary not present in this checkout")
    qat_path = REPO / "experiments" / "qat_finetune.py"
    # Use --device cpu + --upstream a nonexistent path so we don't burn GPU
    # or require upstream/. We expect the script to fail at the --poses check
    # BEFORE it reaches scorer load (which is what would crash from the bad
    # upstream path). This is a mid-script-fail test, not an end-to-end test.
    result = subprocess.run(
        [
            sys.executable, str(qat_path),
            "--checkpoint", str(baseline),
            "--upstream", "/tmp/nonexistent_upstream_for_test",
            "--output-dir", "/tmp/qat_finetune_pose_test",
            "--device", "cpu",
            "--base-ch", "36", "--mid-ch", "60", "--pose-dim", "6",
            "--motion-hidden", "32", "--depth", "1", "--embed-dim", "6",
            "--use-zoom-flow", "--padding-mode", "zeros",
            "--skip-int8-warmup",
            "--fp4-epochs", "1",
            "--lr", "5e-5",
            "--batch-size", "1",
            # NOTE: deliberately NOT passing --poses to trigger the raise.
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode != 0, (
        f"qat_finetune.py exited 0 with pose_dim=6 but no --poses. "
        f"The silent-WARN fallback regressed.\nstdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    combined = result.stdout + result.stderr
    # The fatal message should mention --poses to give the operator a clear fix.
    assert "--poses" in combined, (
        f"Error message must mention --poses. Got:\n{combined[-2000:]}"
    )


def test_qat_finetune_no_silent_auto_discovery_pattern():
    """Final structural gate: scan the qat_finetune.py source for the
    silent-auto-discovery pattern using the meta-bug check. Must return
    0 violations after Bug 1 fix."""
    from tac.preflight import (
        _scan_python_for_silent_auto_discovery,
        REPO_ROOT,
    )
    qat_path = REPO_ROOT / "experiments" / "qat_finetune.py"
    v = _scan_python_for_silent_auto_discovery(qat_path, REPO_ROOT)
    assert v == [], (
        "qat_finetune.py has silent-auto-discovery violation(s). "
        "Bug 1 fix regressed. Findings:\n"
        + "\n".join(f"  • {x}" for x in v)
    )
