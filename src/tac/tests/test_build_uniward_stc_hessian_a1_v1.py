"""Tests for the A1 UNIWARD/Hessian byte-candidate builder."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "build_uniward_stc_hessian_a1_v1.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "build_uniward_stc_hessian_a1_v1_under_test",
        TOOL,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_allocate_bits_treats_source_decoder_size_as_eight_bit_baseline() -> None:
    mod = _load_tool()
    sd = {
        "low.weight": torch.ones(100),
        "high.weight": torch.ones(100) * 4,
    }
    fisher = {"low.weight": 1.0e-4, "high.weight": 1.0}

    no_cut = mod.allocate_bits_per_tensor(
        sd,
        fisher,
        target_decoder_bytes=1000,
        source_decoder_bytes=1000,
        floor_bits=4,
        ceiling_bits=8,
    )
    cut = mod.allocate_bits_per_tensor(
        sd,
        fisher,
        target_decoder_bytes=750,
        source_decoder_bytes=1000,
        floor_bits=4,
        ceiling_bits=8,
    )

    assert no_cut == {"low.weight": 8, "high.weight": 8}
    assert min(cut.values()) < 8
    assert cut["high.weight"] >= cut["low.weight"]


def test_allocate_bits_charges_large_tensors_by_numel() -> None:
    mod = _load_tool()
    sd = {
        "small_high.weight": torch.ones(10) * 4,
        "large_low.weight": torch.ones(1000),
    }
    fisher = {"small_high.weight": 1.0, "large_low.weight": 1.0e-4}

    bits = mod.allocate_bits_per_tensor(
        sd,
        fisher,
        target_decoder_bytes=650,
        source_decoder_bytes=1000,
        floor_bits=4,
        ceiling_bits=8,
    )
    charged_bits = sum(sd[name].numel() * bit for name, bit in bits.items())

    assert bits["small_high.weight"] >= bits["large_low.weight"]
    assert charged_bits <= sum(t.numel() for t in sd.values()) * 8
    assert charged_bits >= sum(t.numel() for t in sd.values()) * 4


def test_parse_manual_bit_overrides() -> None:
    mod = _load_tool()

    assert mod.parse_manual_bit_overrides(["blocks.3.weight=7"]) == {
        "blocks.3.weight": 7
    }


def test_parse_manual_bit_overrides_rejects_malformed_values() -> None:
    mod = _load_tool()

    try:
        mod.parse_manual_bit_overrides(["blocks.3.weight:7"])
    except ValueError as exc:
        assert "NAME=BITS" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("malformed override should fail")


def test_evaluate_cliff_zone_v1_blocks4_falls_in_cliff() -> None:
    """Track 4 v1 blocks4_7bit (-359 B at rms ≈ 1.84e-3) is in the cliff zone.

    Empirical anchor: blocks4_7bit lost +0.0058 score on contest-CPU GHA.
    The default gate (AND form) catches it because:
        bytes_saved = 359 < 1024 (= 1 KB) AND rms = 1.84e-3 > 1e-3
    """
    mod = _load_tool()
    ratio, in_cliff = mod.evaluate_cliff_zone(
        bytes_saved=359, rms_distortion=1.84e-3,
    )
    # ratio = (359/1024) / 1.84e-3 ≈ 191 KB/rms — below the
    # secondary floor as well as inside the primary AND gate.
    assert ratio > 0
    assert ratio < mod.CLIFF_ZONE_THRESHOLD_KB_PER_RMS_DEFAULT
    # AND form must fire: <1 KB saved AND >1e-3 rms.
    assert in_cliff is True


def test_evaluate_cliff_zone_marginal_savings_low_distortion_passes() -> None:
    """A candidate that saves <1KB at very low (rms <= 1e-3) distortion is NOT
    in the cliff zone — the rms axis is too small to do score damage so
    BOTH primary AND secondary criteria are gated by the rms floor.
    """
    mod = _load_tool()
    ratio, in_cliff = mod.evaluate_cliff_zone(
        bytes_saved=200, rms_distortion=5e-4,  # < threshold 1e-3
    )
    assert ratio > 0
    # rms 5e-4 NOT > 1e-3 → both primary and secondary gated off
    assert in_cliff is False


def test_evaluate_cliff_zone_high_savings_high_distortion_safe() -> None:
    """Large bytes savings (>>1KB) at moderate rms passes both gates IF the
    bytes-per-rms ratio is high enough.
    """
    mod = _load_tool()
    # 50 KB saved at 0.01 rms → ratio ≈ 4882 KB/rms, well above 1000 floor
    ratio, in_cliff = mod.evaluate_cliff_zone(
        bytes_saved=50_000, rms_distortion=0.01,
    )
    assert ratio > 1000.0
    assert in_cliff is False  # 50K B >= 1024 B (primary OFF) AND ratio high (secondary OFF)


def test_evaluate_cliff_zone_high_savings_at_high_rms_with_low_ratio_blocked() -> None:
    """Even large bytes savings at HIGH rms can fail if the ratio is too low.

    This is the secondary gate's job: catches "bytes_saved >= 1KB but the
    rms is so high that the bytes-per-rms ratio drops below the floor."
    """
    mod = _load_tool()
    # 5 KB saved at 0.05 rms → ratio = (5000/1024)/0.05 ≈ 97.7 KB/rms ≪ 1000
    ratio, in_cliff = mod.evaluate_cliff_zone(
        bytes_saved=5_000, rms_distortion=0.05,
    )
    assert ratio < 1000.0
    # Primary OFF (5K B >= 1 KB), but secondary fires (rms > 1e-3 AND ratio < 1000)
    assert in_cliff is True


def test_evaluate_cliff_zone_safe_candidate_passes() -> None:
    """A candidate with substantial bytes_saved at safe rms is NOT in cliff zone.

    Safe = (1) bytes_saved >= 1KB AND (2) ratio_kb_per_rms above secondary
    floor at the operating-point rms.
    """
    mod = _load_tool()
    # 50 KB saved at rms 0.5% → ratio = 9766 KB/rms, well above default 1000
    ratio, in_cliff = mod.evaluate_cliff_zone(
        bytes_saved=50_000, rms_distortion=5e-3,
    )
    assert in_cliff is False
    assert ratio > 1000.0


def test_evaluate_cliff_zone_no_savings_returns_inf_not_blocked() -> None:
    """Candidate that GAINED bytes is not in the savings cliff zone."""
    mod = _load_tool()
    ratio, in_cliff = mod.evaluate_cliff_zone(
        bytes_saved=-100, rms_distortion=5e-3,
    )
    assert in_cliff is False
    assert ratio == float("inf")


def test_evaluate_cliff_zone_zero_distortion_uses_eps_floor() -> None:
    """Zero distortion uses the eps floor; no ZeroDivisionError."""
    mod = _load_tool()
    ratio, in_cliff = mod.evaluate_cliff_zone(
        bytes_saved=100, rms_distortion=0.0,
    )
    # Finite ratio, very large (because eps is the denominator)
    import math as _math
    assert _math.isfinite(ratio)
    assert ratio > 0


def test_build_unknown_saliency_source_raises() -> None:
    """`build()` rejects unknown saliency-source strings."""
    mod = _load_tool()
    # We can probe build() argument-validation without touching disk by
    # asserting the early-fail path before src_archive is read.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        # Create a stub archive that is the actual A1 archive (we have it on disk)
        repo = Path(__file__).resolve().parents[3]
        a1_archive = (
            repo
            / "experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_"
              "20260509T012628Z_modal/harvested_artifacts/finetuned_archive/archive.zip"
        )
        if not a1_archive.is_file():
            return  # skip if A1 not present in this checkout
        out_dir = td_p / "out"
        try:
            mod.build(
                src_archive=a1_archive,
                source_submission_dir=None,
                out_dir=out_dir,
                target_bytes=178_000,
                floor_bits=4,
                ceiling_bits=8,
                require_sha=False,
                saliency_source="not_a_real_source",
            )
        except SystemExit as exc:
            assert "saliency-source" in str(exc) or "not_a_real_source" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("unknown saliency-source should fail")


def test_build_with_saliency_override_skips_compute() -> None:
    """`build(saliency_override=...)` uses the provided dict and skips
    both the autograd pass AND the mean(theta^2) compute path.
    """
    mod = _load_tool()
    import tempfile
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    a1_archive = (
        repo
        / "experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_"
          "20260509T012628Z_modal/harvested_artifacts/finetuned_archive/archive.zip"
    )
    if not a1_archive.is_file():
        return  # skip if A1 not present

    # Build a saliency override that names every state-dict key in the A1
    # decoder. Easiest way: load the decoder blob, decode it to a state_dict,
    # and stamp uniform 1.0 per key.
    import struct
    import zipfile as _zipfile
    import sys as _sys

    if str(repo / "src") not in _sys.path:
        _sys.path.insert(0, str(repo / "src"))
    from tac.pr101_split_brotli_codec import (  # noqa: E501  WEIGHT_SALIENCY_OK_ON_SCORE_AWARE: test fixture
        decode_decoder_compact, LATENT_BLOB_LEN,
    )

    with _zipfile.ZipFile(a1_archive) as zf:
        inner = zf.read("x")
    section_total = struct.unpack_from("<I", inner, 0)[0]
    decoder_blob = inner[4:section_total]
    sd = decode_decoder_compact(decoder_blob)
    saliency_override = {name: 1.0 + i * 0.1 for i, name in enumerate(sd)}

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "out_saliency_override"
        result = mod.build(
            src_archive=a1_archive,
            source_submission_dir=None,
            out_dir=out_dir,
            target_bytes=178_000,
            floor_bits=4,
            ceiling_bits=8,
            require_sha=False,
            saliency_override=saliency_override,
            # Cliff zone: at near-zero distortion the ratio is huge, so the
            # gate won't fire. Disable to be safe in case rms is computed >0.
            allow_cliff_zone=True,
            cliff_zone_override_operator="test_suite",
        )
        # Saliency source label must persist
        assert result.saliency_source == "mean_theta_squared"  # default param value
        # Bit-alloc records carry the OVERRIDE saliency, not mean(theta^2)
        observed_saliency = {r.name: r.fisher_proxy for r in result.bit_alloc}
        assert observed_saliency == saliency_override


def test_build_cliff_zone_blocks_unsafe_candidate() -> None:
    """Without --allow-cliff-zone, a low-savings/high-distortion candidate is refused."""
    mod = _load_tool()
    import tempfile
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    a1_archive = (
        repo
        / "experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_"
          "20260509T012628Z_modal/harvested_artifacts/finetuned_archive/archive.zip"
    )
    if not a1_archive.is_file():
        return  # skip if A1 not present

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "out_cliff"
        # Use an absurdly strict cliff threshold (1e30 KB·rms²) so EVERY
        # candidate that saves any bytes at any positive rms is refused.
        try:
            mod.build(
                src_archive=a1_archive,
                source_submission_dir=None,
                out_dir=out_dir,
                target_bytes=170_000,  # forces meaningful bit cuts -> rms > 0
                floor_bits=4,
                ceiling_bits=8,
                require_sha=False,
                cliff_zone_threshold_kb_rms_sq=1e30,
                allow_cliff_zone=False,
            )
        except SystemExit as exc:
            assert "cliff zone" in str(exc).lower()
        else:  # pragma: no cover
            raise AssertionError("cliff-zone gate should refuse this candidate")


def test_build_cliff_zone_override_requires_operator() -> None:
    """CLI gate: --allow-cliff-zone needs --cliff-zone-override-operator."""
    mod = _load_tool()
    try:
        mod.main([
            "--out-dir", "/tmp/whatever-test-not-actually-built",
            "--allow-cliff-zone",
        ])
    except SystemExit as exc:
        assert "cliff-zone-override-operator" in str(exc) or "operator" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("--allow-cliff-zone without operator should fail")
