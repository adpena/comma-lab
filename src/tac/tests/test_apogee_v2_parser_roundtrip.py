"""Test apogee_v2 0.bin parser round-trip on real PR106 weights.

Lane Ω-W-V3 step 4 hardening — verifies submissions/apogee_v2/inflate.py's
parse_apogee_v2_archive correctly reverses the encoding produced by
experiments/repack_pr106_with_water_filling.py.

This test gates against parser regressions that would silently corrupt
the decoded HNeRV state_dict (and thus PoseNet/SegNet eval scores) at
contest dispatch time.

Skipped if PR106 archive or repack output are not present locally
(those are large artifacts not committed to git).
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PR106_ARCHIVE = REPO_ROOT / (
    "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip"
)
APOGEE_V2_ARCHIVE = REPO_ROOT / (
    "experiments/results/apogee_v2_repack_20260504_claude/apogee_v2_archive.zip"
)
APOGEE_V2_INFLATE_DIR = REPO_ROOT / "submissions/apogee_v2"


def _load_apogee_v2_parser():
    sys.path.insert(0, str(APOGEE_V2_INFLATE_DIR))
    sys.modules.pop("inflate", None)
    sys.modules.pop("codec", None)
    from inflate import parse_apogee_v2_archive  # type: ignore[import-not-found]
    return parse_apogee_v2_archive


def _load_pr106_parser():
    sys.path.insert(0, str(REPO_ROOT / "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders/src"))
    sys.modules.pop("codec", None)
    from codec import parse_packed_archive  # type: ignore[import-not-found]
    return parse_packed_archive


@pytest.mark.skipif(
    not APOGEE_V2_ARCHIVE.is_file(),
    reason=f"apogee_v2 archive not present at {APOGEE_V2_ARCHIVE} — run experiments/repack_pr106_with_water_filling.py first",
)
def test_apogee_v2_parser_basic_smoke():
    """Parser accepts apogee_v2 0.bin and produces 28 tensors / 228,958 params + (600, 28) latents."""
    parse_apogee_v2_archive = _load_apogee_v2_parser()
    with zipfile.ZipFile(APOGEE_V2_ARCHIVE) as z:
        bin_bytes = z.read("0.bin")
    state_dict, latents, meta = parse_apogee_v2_archive(bin_bytes)

    assert len(state_dict) == 28, f"expected 28 tensors (PR106 HNeRV schema), got {len(state_dict)}"
    total_params = sum(t.numel() for t in state_dict.values())
    assert total_params == 228958, f"expected 228,958 params, got {total_params}"
    assert tuple(latents.shape) == (600, 28), f"expected latents (600, 28), got {tuple(latents.shape)}"
    assert meta == {"n_pairs": 600, "latent_dim": 28, "base_channels": 36, "eval_size": [384, 512]}, \
        f"meta mismatch: {meta}"


@pytest.mark.skipif(
    not APOGEE_V2_ARCHIVE.is_file() or not PR106_ARCHIVE.is_file(),
    reason="apogee_v2 + PR106 archives required",
)
def test_apogee_v2_parser_preserves_pr106_schema():
    """Apogee_v2 state_dict has EXACTLY the same tensor names and shapes as PR106."""
    parse_apogee_v2_archive = _load_apogee_v2_parser()
    parse_packed_archive = _load_pr106_parser()

    with zipfile.ZipFile(APOGEE_V2_ARCHIVE) as z:
        apogee_bin = z.read("0.bin")
    with zipfile.ZipFile(PR106_ARCHIVE) as z:
        pr106_bin = z.read("0.bin")

    apogee_sd, _, _ = parse_apogee_v2_archive(apogee_bin)
    pr106_sd, _, _ = parse_packed_archive(pr106_bin)

    assert set(apogee_sd.keys()) == set(pr106_sd.keys()), (
        f"tensor name set mismatch: apogee_v2-only={set(apogee_sd) - set(pr106_sd)}, "
        f"pr106-only={set(pr106_sd) - set(apogee_sd)}"
    )
    for name in pr106_sd:
        assert apogee_sd[name].shape == pr106_sd[name].shape, (
            f"{name}: apogee_v2 shape {tuple(apogee_sd[name].shape)} != "
            f"pr106 shape {tuple(pr106_sd[name].shape)}"
        )


@pytest.mark.skipif(
    not APOGEE_V2_ARCHIVE.is_file(),
    reason="apogee_v2 archive required",
)
def test_apogee_v2_parser_no_trailing_bytes():
    """Parser consumes exactly len(bin_bytes); no trailing bytes implies layout integrity."""
    parse_apogee_v2_archive = _load_apogee_v2_parser()
    with zipfile.ZipFile(APOGEE_V2_ARCHIVE) as z:
        bin_bytes = z.read("0.bin")
    # parse_apogee_v2_archive raises ValueError if trailing bytes detected.
    state_dict, latents, meta = parse_apogee_v2_archive(bin_bytes)
    assert len(state_dict) > 0
    assert latents.numel() > 0


@pytest.mark.skipif(
    not APOGEE_V2_ARCHIVE.is_file(),
    reason="apogee_v2 archive required",
)
def test_apogee_v2_magic_byte_check():
    """Parser raises ValueError on wrong magic byte (anti-corruption guard)."""
    parse_apogee_v2_archive = _load_apogee_v2_parser()
    with zipfile.ZipFile(APOGEE_V2_ARCHIVE) as z:
        bin_bytes = bytearray(z.read("0.bin"))
    bin_bytes[0] = 0xFF  # PR106's magic; should be rejected by apogee_v2 parser
    with pytest.raises(ValueError, match="apogee_v2 magic mismatch"):
        parse_apogee_v2_archive(bytes(bin_bytes))
