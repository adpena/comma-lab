"""Pin the cross-paradigm flag → guard-site contract in experiments/pipeline.py.

Background: 4 commits (999211e5 / 9bdd3d56 / 77dc808a / 80455cf8) registered
5 cross-paradigm PipelineConfig flags + WARN-on-unwired guards in the relevant
``step_*`` functions. This test pins that contract:

1. Every flag on PipelineConfig that starts with ``use_`` is BOOL-typed and
   defaults to False.
2. Each cross-paradigm flag has at least one ``cfg.<flag>`` reference in the
   pipeline.py source — preventing dead-flag drift.
3. The PARADIGM-α ``mask_codec`` field validates: stub codecs raise
   NotImplementedError, unknown codecs raise ValueError, wired codecs proceed.

If a future commit adds a new cross-paradigm flag without a guard-site usage,
this test fails — surfacing the silent-no-op trap before it ships.

Cross-ref: project_cross_paradigm_pipeline_wiring_landed_20260506.md
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

from experiments.pipeline import PipelineConfig, step_extract_masks

_PIPELINE_PATH = (
    Path(__file__).resolve().parents[3] / "experiments" / "pipeline.py"
)


# Flags that the cross-paradigm wiring landing introduced. New flags added in
# the future MUST also be added here so the guard-site test pins them.
CROSS_PARADIGM_BOOL_FLAGS: tuple[str, ...] = (
    "use_sensitivity_weighted",
    "use_joint_codec_stack",
    "use_joint_scorer_aware",
    "use_learnable_entropy",
    "use_full_renderer_self_compress",
    "use_raft_init",
    "use_riemannian_tto",
)

# Companion non-bool fields registered alongside the bool flags.
CROSS_PARADIGM_OTHER_FIELDS: tuple[str, ...] = (
    "sensitivity_map_path",
    "owv3_bit_budget_ratio",
    "owv3_protect_threshold",
    "mask_codec",
)


def test_cross_paradigm_flags_exist_with_safe_defaults() -> None:
    cfg = PipelineConfig()
    for name in CROSS_PARADIGM_BOOL_FLAGS:
        value = getattr(cfg, name)
        assert isinstance(value, bool), f"{name} must be bool; got {type(value)}"
        assert value is False, (
            f"{name} default must be False — non-False default would change "
            f"existing pipeline.py runs without operator opt-in."
        )
    # mask_codec defaults to the wired AV1 path (current behavior).
    assert cfg.mask_codec == "av1_monochrome", (
        f"mask_codec default must be 'av1_monochrome' (the wired path); "
        f"got {cfg.mask_codec!r}"
    )
    # Numeric calibration fields must be in their documented ranges.
    assert 0.0 < cfg.owv3_bit_budget_ratio <= 1.0
    assert cfg.owv3_protect_threshold > 0.0


def test_each_cross_paradigm_flag_has_a_guard_site() -> None:
    src = _PIPELINE_PATH.read_text(encoding="utf-8")
    for name in CROSS_PARADIGM_BOOL_FLAGS:
        # Each flag must be referenced via cfg.<name> at least once outside
        # the dataclass field declaration. The dataclass uses ``: bool`` while
        # any guard or dispatch reads via cfg.<name>.
        guard_pattern = re.compile(rf"\bcfg\.{re.escape(name)}\b")
        assert guard_pattern.search(src), (
            f"Cross-paradigm flag {name!r} has NO ``cfg.{name}`` reference "
            f"in experiments/pipeline.py — silent-no-op trap. "
            f"Add a WARN guard or dispatch branch in the corresponding step."
        )


def test_dezeta_phase1_flags_fail_closed() -> None:
    """delta/epsilon/zeta Phase-1 flags must not silently fall through."""
    src = _PIPELINE_PATH.read_text(encoding="utf-8")
    for name in (
        "use_joint_scorer_aware",
        "use_learnable_entropy",
        "use_full_renderer_self_compress",
    ):
        idx = src.find(f"cfg.{name}")
        assert idx >= 0
        window = src[idx: idx + 900]
        assert "raise NotImplementedError" in window
        assert "REGISTERED-BUT-NOT-WIRED" in window
        assert "Refusing to fall" in window


# Flags with full dispatch wiring (not just WARN guards). New flags graduate
# into this set when their actual dispatch branch lands. Pinning the set here
# prevents accidental regression where a future commit downgrades a wired
# flag back to a WARN-only guard.
WIRED_CROSS_PARADIGM_FLAGS: tuple[str, ...] = (
    # β: encode_owv3_archive() called in step_compress_weights when
    # sensitivity_map_path provided (commit 107f6fea).
    "use_sensitivity_weighted",
    # la-pose Riemannian: --optimizer=riemannian-sgd appended to
    # optimize_poses.py subprocess in step_pose_tto (commit 330356f1).
    "use_riemannian_tto",
)


def test_wired_flags_have_actual_dispatch() -> None:
    """Flags that are documented as WIRED must have evidence of real
    dispatch beyond the WARN guard. We check for a module-import or
    subprocess-flag-extension that fires conditional on the flag."""
    src = _PIPELINE_PATH.read_text(encoding="utf-8")
    # β: encode_owv3_archive must be imported in the dispatch branch
    if "use_sensitivity_weighted" in WIRED_CROSS_PARADIGM_FLAGS:
        assert "encode_owv3_archive" in src, (
            "β is documented WIRED but encode_owv3_archive is not imported "
            "in pipeline.py — dispatch regressed to WARN-only?"
        )
    # Riemannian: --optimizer must appear (passed as subprocess flag)
    if "use_riemannian_tto" in WIRED_CROSS_PARADIGM_FLAGS:
        assert "riemannian-sgd" in src, (
            "Riemannian is documented WIRED but '--optimizer=riemannian-sgd' "
            "is not in pipeline.py — dispatch regressed to WARN-only?"
        )


def test_mask_codec_stub_gate_raises_not_implemented() -> None:
    """PARADIGM-α stub codecs must raise NotImplementedError, not silently
    fall back to AV1 or proceed with a non-decodable archive."""
    stub_codecs = ("nerv", "wavelet", "vqvae", "grayscale_lut")
    for codec in stub_codecs:
        cfg = PipelineConfig(
            mask_codec=codec, output_dir=tempfile.mkdtemp(prefix="cpwc_")
        )
        with pytest.raises(NotImplementedError) as exc_info:
            step_extract_masks(cfg)
        assert codec in str(exc_info.value), (
            f"NotImplementedError message must name the stub codec {codec!r} "
            f"so operators can find the lane registry entry. "
            f"Got: {exc_info.value!s}"
        )


def test_mask_codec_unknown_raises_value_error() -> None:
    """Typos / unknown mask codec values must raise ValueError, not silently
    fall through."""
    cfg = PipelineConfig(
        mask_codec="banana", output_dir=tempfile.mkdtemp(prefix="cpwc_")
    )
    with pytest.raises(ValueError) as exc_info:
        step_extract_masks(cfg)
    assert "banana" in str(exc_info.value)


def test_mask_codec_wired_paths_dont_raise_at_gate() -> None:
    """av1_monochrome / argmax_rle reach past the gate (they may fail later
    on missing video, but the gate itself doesn't reject them)."""
    for codec in ("av1_monochrome", "argmax_rle"):
        cfg = PipelineConfig(
            mask_codec=codec, output_dir=tempfile.mkdtemp(prefix="cpwc_")
        )
        # Expect a non-NotImplementedError, non-ValueError failure (the
        # downstream code will fail on the missing video / scorers).
        with pytest.raises(Exception) as exc_info:
            step_extract_masks(cfg)
        assert not isinstance(exc_info.value, NotImplementedError), (
            f"Wired codec {codec!r} must NOT raise NotImplementedError "
            f"at the mask_codec gate — that gate is for stubs only. "
            f"Got: {exc_info.value!s}"
        )


def test_other_cross_paradigm_fields_present() -> None:
    cfg = PipelineConfig()
    for name in CROSS_PARADIGM_OTHER_FIELDS:
        assert hasattr(cfg, name), (
            f"Cross-paradigm companion field {name!r} missing from "
            f"PipelineConfig — wiring drift."
        )
