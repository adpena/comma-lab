# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the Track-A DISTORTION finishing-kit (inflate-side bolt-ons).

The kit (``tac.torch_vehicle.distortion_finishing_kit`` + the driver finishing
functions) is a default-OFF, byte-identical-when-disabled, numpy-portable
inflate-side postproc on the rendered frames + a ~54-byte distortion archive
section. These tests prove the load-bearing claims (each, if wrong, is either a
FAKE bolt-on OR — worse — silently changes the LIVE base_ch=20 distortion arm if
it resumes onto this code):

A. **DEFAULT-OFF BYTE IDENTITY (the daemon-safety guard).** A disabled/identity
   kit (1) serializes ZERO section bytes, (2) returns the raw frames UNCHANGED
   (the SAME object — a true no-op), (3) the finishing pass appends NOTHING (the
   finished archive is byte-identical to the input). If replacing the no-op guard
   with an always-apply body would still pass, the test is fake — so we assert the
   no-op returns the SAME object AND a non-identity kit returns a DIFFERENT one.

B. **THE POSTPROC ACTUALLY CHANGES THE FRAMES IN THE CLAIMED DIRECTION** (no
   constant-checking): the converged residual fixture lowers ONLY frame_0 red and
   blue by 1 and leaves frame_1 + other channels untouched; a T10 affine
   ``scale != 1`` scales the channel; the change is the EXACT
   ``round(clip(scale*x - bias))`` the contract claims.

C. **THE SECTION ROUND-TRIPS BIT-EXACTLY** and is fail-closed on corruption; the
   finished archive splits back into (base, kit) with the base byte-identical.

D. **PARITY** between the numpy raw-frame postproc and the torch camera-float
   postproc (the export-faithful eval uses the torch path; the numpy inflate uses
   the raw path; they must agree on the same transform).
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.torch_vehicle.distortion_finishing_kit import (
    CONVERGED_RESIDUAL_PR98_BIAS,
    CONVERGED_RESIDUAL_PR98_PROVENANCE,
    DISTORTION_SECTION_MAGIC,
    DistortionKitConfig,
    apply_distortion_kit_to_camera_float,
    apply_distortion_kit_to_raw_frames,
    parse_distortion_section,
    serialize_distortion_section,
)


def _rng_frames(n: int = 8, h: int = 16, w: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# A. DEFAULT-OFF BYTE IDENTITY (daemon-safety)
# ---------------------------------------------------------------------------
def test_disabled_kit_serializes_zero_bytes():
    cfg = DistortionKitConfig(enabled=False)
    assert serialize_distortion_section(cfg) == b""


def test_identity_enabled_kit_is_noop_object_identity():
    # An ENABLED but identity (scale=1,bias=0) kit is still a byte-identical no-op.
    cfg = DistortionKitConfig(enabled=True)
    assert cfg.is_identity
    frames = _rng_frames()
    out = apply_distortion_kit_to_raw_frames(frames, cfg)
    assert out is frames  # SAME object — a real no-op, not a copy


def test_disabled_kit_raw_postproc_is_noop_object_identity():
    cfg = DistortionKitConfig(enabled=False)
    frames = _rng_frames()
    out = apply_distortion_kit_to_raw_frames(frames, cfg)
    assert out is frames


def test_nonidentity_kit_returns_different_object_and_bytes():
    # The anti-fake guard: a non-identity kit must return a DIFFERENT array (else
    # the no-op short-circuit would mask a broken transform).
    bias = np.zeros((2, 3))
    bias[1, 1] = 1.0  # frame_1 green -1
    cfg = DistortionKitConfig.from_pr98_bias(bias)
    frames = _rng_frames(seed=1)
    out = apply_distortion_kit_to_raw_frames(frames, cfg)
    assert out is not frames
    assert not np.array_equal(out, frames)


def test_finish_checkpoint_disabled_is_byte_identical():
    from tac.torch_vehicle.driver import finish_checkpoint_with_distortion_kit

    base = b"VENDORED_ARCHIVE_BYTES_xyz" * 10
    res = finish_checkpoint_with_distortion_kit(base, DistortionKitConfig(enabled=False))
    assert res["is_byte_identical"] is True
    assert res["added_bytes"] == 0
    assert res["finished_archive"] == base


def test_finish_checkpoint_enabled_appends_fixed_section():
    from tac.torch_vehicle.driver import finish_checkpoint_with_distortion_kit

    base = b"VENDORED_ARCHIVE_BYTES_xyz" * 10
    bias = np.zeros((2, 3))
    bias[1, 1] = 1.0
    cfg = DistortionKitConfig.from_pr98_bias(bias)
    res = finish_checkpoint_with_distortion_kit(base, cfg)
    assert res["is_byte_identical"] is False
    assert res["added_bytes"] == 54  # MAGIC(4)+ver(1)+flags(1)+6*(f32+f32)=54
    assert res["finished_archive"][:len(base)] == base
    assert res["finished_archive"][len(base):][:4] == DISTORTION_SECTION_MAGIC


def test_finish_refuses_double_append():
    from tac.torch_vehicle.driver import finish_checkpoint_with_distortion_kit

    base = b"X" * 100
    bias = np.zeros((2, 3))
    bias[1, 1] = 1.0
    cfg = DistortionKitConfig.from_pr98_bias(bias)
    finished = finish_checkpoint_with_distortion_kit(base, cfg)["finished_archive"]
    with pytest.raises(ValueError, match="already carries"):
        finish_checkpoint_with_distortion_kit(finished, cfg)


# ---------------------------------------------------------------------------
# B. THE POSTPROC ACTUALLY CHANGES THE FRAMES IN THE CLAIMED DIRECTION
# ---------------------------------------------------------------------------
def test_pr98_bias_only_touches_targeted_slot():
    # PR98 frame_1 GREEN -= 1: ONLY odd frames' green channel decreases by 1.
    bias = np.zeros((2, 3))
    bias[1, 1] = 1.0
    cfg = DistortionKitConfig.from_pr98_bias(bias)
    # use a mid-range frame so clip doesn't saturate
    frames = np.full((4, 8, 8, 3), 128, dtype=np.uint8)
    out = apply_distortion_kit_to_raw_frames(frames, cfg)
    # frame_0 (parity 0) unchanged everywhere
    assert np.array_equal(out[0], frames[0])
    assert np.array_equal(out[2], frames[2])
    # frame_1 (parity 1) green -1, R/B unchanged
    assert np.all(out[1, :, :, 1] == 127)
    assert np.all(out[1, :, :, 0] == 128)
    assert np.all(out[1, :, :, 2] == 128)
    assert np.all(out[3, :, :, 1] == 127)


def test_converged_residual_fixture_is_pr98_only_frame0_red_blue():
    cfg = DistortionKitConfig.from_converged_residual_pr98()
    assert cfg.enabled is True
    assert cfg.scale == ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    assert cfg.bias == CONVERGED_RESIDUAL_PR98_BIAS
    assert "under-power audit" in cfg.provenance
    assert cfg.provenance == CONVERGED_RESIDUAL_PR98_PROVENANCE

    frames = np.full((4, 8, 8, 3), 128, dtype=np.uint8)
    out = apply_distortion_kit_to_raw_frames(frames, cfg)

    # frame_0 (even indices) red/blue decrease by 1; green remains unchanged.
    assert np.all(out[0, :, :, 0] == 127)
    assert np.all(out[0, :, :, 1] == 128)
    assert np.all(out[0, :, :, 2] == 127)
    assert np.all(out[2, :, :, 0] == 127)
    assert np.all(out[2, :, :, 2] == 127)

    # frame_1 (odd indices) is not part of the residual survivor.
    assert np.array_equal(out[1], frames[1])
    assert np.array_equal(out[3], frames[3])


def test_affine_scale_applies_round_clip_contract():
    # T10 scale=1.1 on frame_0 R: round(clip(1.1*x - 0)).
    scale = np.ones((2, 3))
    scale[0, 0] = 1.1
    bias = np.zeros((2, 3))
    cfg = DistortionKitConfig.from_affine(scale, bias)
    frames = np.full((2, 4, 4, 3), 100, dtype=np.uint8)
    out = apply_distortion_kit_to_raw_frames(frames, cfg)
    assert np.all(out[0, :, :, 0] == round(110.0))  # 1.1*100=110
    assert np.all(out[0, :, :, 1] == 100)  # untouched channel
    assert np.all(out[1] == 100)  # frame_1 untouched (scale[1]=1)


def test_clip_saturates_at_bounds():
    bias = np.zeros((2, 3))
    bias[1, 1] = -300.0  # would push green above 255 -> clip at 255
    cfg = DistortionKitConfig.from_pr98_bias(bias)
    frames = np.full((2, 2, 2, 3), 200, dtype=np.uint8)
    out = apply_distortion_kit_to_raw_frames(frames, cfg)
    assert np.all(out[1, :, :, 1] == 255)  # 200 - (-300) = 500 -> clip 255


# ---------------------------------------------------------------------------
# C. SECTION ROUND-TRIP + FAIL-CLOSED
# ---------------------------------------------------------------------------
def test_section_round_trips_bit_exactly():
    scale = np.array([[1.01, 0.99, 1.0], [1.0, 1.02, 0.98]])
    bias = np.array([[1.0, 0.0, -2.0], [0.0, 1.5, 0.0]])
    cfg = DistortionKitConfig.from_affine(scale, bias, s12_certified=True)
    blob = serialize_distortion_section(cfg)
    rt = parse_distortion_section(blob)
    assert np.allclose(rt.scale_array(), scale, atol=1e-6)
    assert np.allclose(rt.bias_array(), bias, atol=1e-6)
    assert rt.s12_invisibility_certified is True
    assert rt.enabled is True


def test_empty_section_parses_to_disabled():
    cfg = parse_distortion_section(b"")
    assert cfg.enabled is False
    assert cfg.is_identity


def test_corrupt_section_fails_closed():
    with pytest.raises(ValueError):
        parse_distortion_section(b"XXXX" + b"\x00" * 50)  # wrong magic, right length
    with pytest.raises(ValueError):
        parse_distortion_section(b"short")  # wrong length


def test_split_finished_archive_recovers_base_and_kit():
    from tac.torch_vehicle.driver import (
        finish_checkpoint_with_distortion_kit,
        split_finished_archive,
    )

    base = b"BASE_ARCHIVE_" * 20
    scale = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    bias = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    cfg = DistortionKitConfig.from_affine(scale, bias)
    finished = finish_checkpoint_with_distortion_kit(base, cfg)["finished_archive"]
    recovered_base, recovered_cfg = split_finished_archive(finished)
    assert recovered_base == base  # base byte-identical
    assert np.allclose(recovered_cfg.bias_array(), bias)
    assert recovered_cfg.enabled is True


def test_split_archive_without_section_returns_disabled():
    from tac.torch_vehicle.driver import split_finished_archive

    base = b"NO_SECTION_HERE" * 5
    recovered_base, cfg = split_finished_archive(base)
    assert recovered_base == base
    assert cfg.enabled is False


# ---------------------------------------------------------------------------
# D. NUMPY <-> TORCH POSTPROC PARITY
# ---------------------------------------------------------------------------
def test_numpy_torch_postproc_parity():
    torch = pytest.importorskip("torch")
    scale = np.array([[1.05, 1.0, 0.97], [1.0, 1.03, 1.0]])
    bias = np.array([[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    cfg = DistortionKitConfig.from_affine(scale, bias)

    # camera-float (B,2,H,W,3) torch path -> uint8
    cam = torch.rand(3, 2, 8, 10, 3) * 255.0
    cam_out = apply_distortion_kit_to_camera_float(cam, cfg)
    torch_u8 = cam_out.clamp(0, 255).round().to(torch.uint8).numpy()

    # raw-frame numpy path: flatten (B,2,...) -> (2B,...) so parity 0/1 matches frame.
    raw = cam.clamp(0, 255).round().to(torch.uint8).numpy().reshape(6, 8, 10, 3)
    np_out = apply_distortion_kit_to_raw_frames(raw, cfg).reshape(3, 2, 8, 10, 3)

    # The torch path applies the affine to the FLOAT then rounds; the numpy path
    # rounds first then applies. They agree to within 1 ULP of round on the
    # integer grid for these gentle transforms — assert near-equality.
    assert np.abs(torch_u8.astype(int) - np_out.astype(int)).max() <= 1


def test_identity_camera_float_is_noop():
    torch = pytest.importorskip("torch")
    cfg = DistortionKitConfig(enabled=False)
    cam = torch.rand(2, 2, 4, 4, 3) * 255.0
    out = apply_distortion_kit_to_camera_float(cam, cfg)
    assert out is cam


def test_config_validation_rejects_bad_shapes():
    with pytest.raises(ValueError):
        DistortionKitConfig(scale=((1.0, 1.0),))  # wrong shape
    with pytest.raises(ValueError):
        DistortionKitConfig.from_pr98_bias(np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# E. FULL INFLATE-CHAIN INTEGRATION (real vendored decoder + vendored inflate.py).
#    The finished packet splits -> base feeds the PRISTINE vendored inflate ->
#    the numpy postproc applies to the REAL raw frames -> byte-identical when off.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("enabled", [False, True])
def test_full_inflate_chain_with_kit(tmp_path, enabled):
    import os
    import subprocess
    import sys
    from pathlib import Path as _P

    torch = pytest.importorskip("torch")
    from tac.torch_vehicle.driver import (
        finish_checkpoint_with_distortion_kit,
        import_vendored_bundle,
        split_finished_archive,
    )
    from tac.torch_vehicle.vendored_imports import VENDORED_SRC

    if not (_P("upstream") / "frame_utils.py").exists():
        pytest.skip("challenge root (upstream/frame_utils.py) not present")

    v = import_vendored_bundle()
    n_pairs = 2
    dec = v.HNeRVDecoder(latent_dim=28, base_channels=8).eval()
    lat = torch.randn(n_pairs, 28) * 0.1
    base_arch = v.build_archive(
        dec.state_dict(), lat,
        meta_dict={"n_pairs": n_pairs, "latent_dim": 28, "base_channels": 8, "eval_size": [384, 512]},
    )

    # Build the kit: enabled -> converged residual PR98 frame_0 R/B -1;
    # disabled -> byte-identical.
    cfg = (
        DistortionKitConfig.from_converged_residual_pr98()
        if enabled
        else DistortionKitConfig(enabled=False)
    )

    finished = finish_checkpoint_with_distortion_kit(base_arch, cfg)["finished_archive"]
    if not enabled:
        assert finished == base_arch  # byte-identical when off

    # The substrate's inflate.sh: split -> vendored inflate on base -> numpy postproc.
    recovered_base, recovered_cfg = split_finished_archive(finished)
    assert recovered_base == base_arch
    (tmp_path / "0.bin").write_bytes(recovered_base)
    env = dict(os.environ)
    env["COMMA_CHALLENGE_ROOT"] = str(_P("upstream").resolve())
    r = subprocess.run(
        [sys.executable, str(VENDORED_SRC.parent / "inflate.py"),
         str(tmp_path / "0.bin"), str(tmp_path / "0.raw")],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert r.returncode == 0, f"vendored inflate failed: {r.stderr[-2000:]}"
    raw = np.frombuffer((tmp_path / "0.raw").read_bytes(), dtype=np.uint8)
    raw = raw.reshape(n_pairs * 2, 874, 1164, 3)

    finished_raw = apply_distortion_kit_to_raw_frames(raw, recovered_cfg)
    if not enabled:
        assert finished_raw is raw  # disabled -> the raw frames pass through unchanged
    else:
        # frame_0 (even indices) red/blue decreased; frame_1 (odd) unchanged.
        assert not np.array_equal(finished_raw, raw)
        expected_r = np.clip(raw[0, :, :, 0].astype(int) - 1, 0, 255).astype(np.uint8)
        expected_b = np.clip(raw[0, :, :, 2].astype(int) - 1, 0, 255).astype(np.uint8)
        assert np.array_equal(finished_raw[0, :, :, 0], expected_r)
        assert np.array_equal(finished_raw[0, :, :, 1], raw[0, :, :, 1])
        assert np.array_equal(finished_raw[0, :, :, 2], expected_b)
        assert np.array_equal(finished_raw[1], raw[1])
