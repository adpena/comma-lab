"""Regression tests for Codex 5-finding adversarial review fixes (2026-04-28).

Coverage:
  * F2 — qat_finetune.py threads --protected-pattern-set into QATConfig and
    bit_allocation_from_sensitivity FORCES protected layers to critical_bits.
  * F3 — swap_renderer_convs_with_self_compress accepts protected_patterns=
    as a REPLACEMENT (not extras), so segnet_prior produces a DIFFERENT
    protected layer set than posenet_prior on a real model swap.
  * F4 — get_supported_quantization_modes gates BF16 by CC >= 8.0 so T4
    (CC 7.5) and P100 (CC 6.0) do NOT advertise BF16 falsely.
  * F5 — contest_auth_eval refuses to start without config.env +
    PYTHON_INFLATE=renderer; launcher tarball includes .env files.

Each test pins a specific magnitude/value anchor per Round 26 convention
so a future regression cannot silently re-introduce the bug.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[3]


# ────────────────────────────────────────────────────────────────────────────
# F3 — swap_renderer_convs_with_self_compress: replacement vs additive
# ────────────────────────────────────────────────────────────────────────────


class _MiniRenderer(nn.Module):
    """Toy renderer with both PoseNet-prior and SegNet-prior named layers.

    Module names are chosen so the protection patterns from BOTH lists
    have at least one match — without this overlap, additive vs replacement
    would be indistinguishable (both reduce to "protect SegNet-only").
    """

    def __init__(self) -> None:
        super().__init__()
        # PoseNet-prior protected: renderer.head, motion.head, fuse_conv
        self.renderer_head = nn.Conv2d(8, 8, 1)  # match by suffix renderer.head
        self.motion_head = nn.Conv2d(8, 8, 1)    # matches motion.head suffix
        self.fuse_conv = nn.Conv2d(8, 8, 1)
        # SegNet-prior protected: out_conv, decode_head
        self.out_conv = nn.Conv2d(8, 8, 1)
        self.decode_head = nn.Conv2d(8, 8, 1)
        # Unprotected bulk layer
        self.bulk = nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.bulk(x)


# Names must match the suffix patterns in SC_PROTECTED_NAME_PATTERNS +
# SC_SEGNET_PROTECTED_NAME_PATTERNS. Both pattern lists use suffix matching
# (".pattern"), so module attributes must be named exactly with dots OR
# wrapped in a container module so the qualified name has the dot.
class _MiniRendererQualified(nn.Module):
    """Variant where qualified names ARE 'renderer.head' / 'motion.head'."""

    def __init__(self) -> None:
        super().__init__()
        self.renderer = nn.Module()
        self.renderer.head = nn.Conv2d(8, 8, 1)
        self.motion = nn.Module()
        self.motion.head = nn.Conv2d(8, 8, 1)
        self.fuse_conv = nn.Conv2d(8, 8, 1)
        self.out_conv = nn.Conv2d(8, 8, 1)
        self.decode_head = nn.Conv2d(8, 8, 1)
        self.bulk = nn.Conv2d(8, 8, 3, padding=1)


def _swap_and_get_protected(pattern_set_name: str | None) -> tuple[set[str], dict]:
    """Run swap and return (set_of_protected_names, full_diag)."""
    from tac.self_compress import (
        get_protected_patterns,
        swap_renderer_convs_with_self_compress,
    )
    model = _MiniRendererQualified()
    if pattern_set_name is None:
        diag = swap_renderer_convs_with_self_compress(model, init_bits=4.0)
    else:
        chosen = get_protected_patterns(pattern_set_name)
        diag = swap_renderer_convs_with_self_compress(
            model, init_bits=4.0, protected_patterns=tuple(chosen),
        )
    return set(diag["protected"]), diag


def test_f3_posenet_prior_protects_posenet_layers() -> None:
    """posenet_prior REPLACES default with PoseNet-prior list.

    Anchor: protected set must include {renderer.head, motion.head, fuse_conv}
    and EXCLUDE {out_conv, decode_head}.
    """
    protected, diag = _swap_and_get_protected("posenet_prior")
    assert "renderer.head" in protected, f"got {protected}"
    assert "motion.head" in protected, f"got {protected}"
    assert "fuse_conv" in protected, f"got {protected}"
    # SegNet-prior layers must NOT be protected when posenet_prior is chosen
    assert "out_conv" not in protected, f"posenet_prior wrongly protected out_conv: {protected}"
    assert "decode_head" not in protected, f"posenet_prior wrongly protected decode_head: {protected}"
    # F3 diagnostic surface: protected_patterns_used must be present
    assert "protected_patterns_used" in diag
    assert len(diag["protected_patterns_used"]) >= 3


def test_f3_segnet_prior_protects_segnet_layers_DISJOINT_from_posenet() -> None:
    """segnet_prior must produce a DISJOINT protected set from posenet_prior.

    This is THE behavioral test that differentiates additive (broken) from
    replacement (correct). Old code treated segnet_prior as
    extra_protected_patterns, so result was PoseNet ∪ SegNet. New code
    REPLACES, so result is SegNet only.

    Anchor: segnet_prior protected = {out_conv, decode_head} only;
    PoseNet-prior layers (renderer.head, motion.head, fuse_conv) must
    NOT be in the protected set.
    """
    posenet_protected, _ = _swap_and_get_protected("posenet_prior")
    segnet_protected, _ = _swap_and_get_protected("segnet_prior")

    # The KEY assertion (was broken before F3):
    overlap = posenet_protected & segnet_protected
    assert not overlap, (
        f"F3 REGRESSION: segnet_prior leaked PoseNet-prior protections. "
        f"Overlap: {overlap}. PoseNet={posenet_protected}, SegNet={segnet_protected}"
    )

    # Anchor: SegNet-prior MUST contain out_conv + decode_head
    assert "out_conv" in segnet_protected
    assert "decode_head" in segnet_protected
    # Anchor: SegNet-prior MUST NOT contain PoseNet-prior layers
    assert "renderer.head" not in segnet_protected
    assert "motion.head" not in segnet_protected
    assert "fuse_conv" not in segnet_protected


def test_f3_default_call_preserves_legacy_posenet_behavior() -> None:
    """When no pattern_set is given, behavior must match legacy default.

    Anchor: the protected set with no kwargs is identical to the
    posenet_prior set (the legacy default before Lane SG). This proves
    the F3 change is BACKWARD COMPATIBLE.
    """
    default_protected, _ = _swap_and_get_protected(None)
    posenet_protected, _ = _swap_and_get_protected("posenet_prior")
    assert default_protected == posenet_protected, (
        f"Default call must match posenet_prior. "
        f"default={default_protected}, posenet={posenet_protected}"
    )


def test_f3_extra_protected_still_extends_replacement() -> None:
    """extra_protected_patterns ADDS to whichever active list is chosen.

    Anchor: passing protected_patterns=['out_conv'] AND
    extra_protected_patterns=('bulk',) protects {out_conv, bulk}, NOT
    the legacy posenet list.
    """
    from tac.self_compress import swap_renderer_convs_with_self_compress
    model = _MiniRendererQualified()
    diag = swap_renderer_convs_with_self_compress(
        model,
        init_bits=4.0,
        protected_patterns=("out_conv",),
        extra_protected_patterns=("bulk",),
    )
    protected = set(diag["protected"])
    assert "out_conv" in protected
    assert "bulk" in protected
    # PoseNet-prior layers MUST NOT leak in
    assert "renderer.head" not in protected
    assert "motion.head" not in protected


# ────────────────────────────────────────────────────────────────────────────
# F2 — qat_finetune.py threads protected_pattern_set into bit allocation
# ────────────────────────────────────────────────────────────────────────────


def test_f2_qatconfig_has_protected_pattern_set_field() -> None:
    """QATConfig must expose protected_pattern_set so the flag has a home."""
    sys.path.insert(0, str(REPO / "experiments"))
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "qat_finetune_mod", REPO / "experiments" / "qat_finetune.py",
        )
        mod = importlib.util.module_from_spec(spec)
        # Don't actually exec full module (heavy imports); just inspect class
        # via source parsing.
        src = (REPO / "experiments" / "qat_finetune.py").read_text()
        assert "protected_pattern_set: str = \"posenet_prior\"" in src, (
            "F2 REGRESSION: QATConfig missing protected_pattern_set field"
        )
        # And the constructor MUST thread args.protected_pattern_set into it
        assert "protected_pattern_set=str(args.protected_pattern_set)" in src, (
            "F2 REGRESSION: QATConfig() construction not threading args.protected_pattern_set"
        )
        # And bit_allocation_from_sensitivity MUST receive cfg.protected_pattern_set
        assert "protected_pattern_set=cfg.protected_pattern_set" in src, (
            "F2 REGRESSION: bit_allocation_from_sensitivity call site not "
            "passing cfg.protected_pattern_set"
        )
    finally:
        sys.path.pop(0)


def test_f2_bit_allocation_forces_protected_to_critical_bits() -> None:
    """bit_allocation_from_sensitivity FORCES protected layers to critical_bits.

    This is the BEHAVIORAL test for F2 — without it, --protected-pattern-set
    would parse + log but produce identical allocations (the operator's
    Lane SG runs were byte-identical to legacy QAT before this fix).

    Anchor: a layer named 'renderer.head' (PoseNet-prior protected) gets
    critical_bits=16 even when the sensitivity profile claims it's the
    most FP4-tolerable layer in the model.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qat_finetune_mod", REPO / "experiments" / "qat_finetune.py",
    )
    qf = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module so dataclass(@) inside the
    # module can resolve cls.__module__ via sys.modules.get (Python 3.12).
    sys.modules["qat_finetune_mod"] = qf
    try:
        spec.loader.exec_module(qf)
    finally:
        # Don't pollute sys.modules for other tests
        pass

    # Build a tiny model with one PoseNet-protected layer + one bulk layer
    class _Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.renderer = nn.Module()
            self.renderer.head = nn.Conv2d(4, 4, 3, padding=1)
            self.bulk = nn.Conv2d(4, 4, 3, padding=1)
    model = _Toy()

    # Build sensitivity profile that claims renderer.head is BEST
    # (lowest sensitivity) — the budget would normally make it FP4.
    # But because it's PoseNet-prior protected, F2 should force it to
    # critical_bits (16).
    sensitivity_dict = {
        "delta": {
            "renderer.head": 0.001,  # Lowest sensitivity → would be FP4
            "bulk": 0.5,             # Highest sensitivity → would be FP16
        },
        "param_count": {
            "renderer.head": model.renderer.head.weight.numel(),
            "bulk": model.bulk.weight.numel(),
        },
    }

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(sensitivity_dict, f.name)
        sens_path = f.name

    # With posenet_prior, renderer.head MUST get critical_bits (16),
    # NOT bulk_bits (4) even though sensitivity profile said it's tolerable.
    allocation, _ = qf.bit_allocation_from_sensitivity(
        sens_path, model,
        target_rate=0.99,  # high budget → without F2, head would be FP4
        bulk_bits=4,
        critical_bits=16,
        protected_pattern_set="posenet_prior",
    )
    assert allocation["renderer.head.weight"] == 16, (
        f"F2 REGRESSION: posenet_prior did not force renderer.head to FP16. "
        f"Got {allocation['renderer.head.weight']} bits."
    )

    # With segnet_prior (DISJOINT from posenet_prior), renderer.head is
    # NO LONGER protected — and at high target_rate it should land at FP4.
    allocation_sg, _ = qf.bit_allocation_from_sensitivity(
        sens_path, model,
        target_rate=0.99,
        bulk_bits=4,
        critical_bits=16,
        protected_pattern_set="segnet_prior",
    )
    assert allocation_sg["renderer.head.weight"] == 4, (
        f"F2 REGRESSION: segnet_prior should NOT protect renderer.head "
        f"(it's PoseNet-prior). Got {allocation_sg['renderer.head.weight']} bits."
    )

    # The two allocations MUST DIFFER — that's the "behavioral test that
    # proves segnet_prior produces a DIFFERENT protected layer set" called
    # for in Codex finding.
    assert allocation != allocation_sg, (
        "F2 REGRESSION: posenet_prior and segnet_prior allocations are "
        "identical — the flag is being parsed but not applied."
    )


# ────────────────────────────────────────────────────────────────────────────
# F4 — BF16 hardware gating by CC >= 8.0
# ────────────────────────────────────────────────────────────────────────────


def test_f4_bf16_excluded_for_t4_cc75() -> None:
    """T4 (CC 7.5) MUST NOT advertise BF16.

    Anchor: with CC=(7, 5), 'bf16' is NOT in the supported set; 'fp16' IS;
    'int8' IS (CC >= 7.5).
    """
    from tac import quantization
    with patch.object(quantization.torch.cuda, "is_available", return_value=True), \
         patch.object(quantization.torch.cuda, "get_device_capability", return_value=(7, 5)):
        modes = quantization.get_supported_quantization_modes("cuda:0")
    assert "bf16" not in modes, f"F4 REGRESSION: T4 CC 7.5 wrongly advertises bf16. modes={modes}"
    assert "fp16" in modes
    assert "int8" in modes
    assert "fp8" not in modes
    assert "fp4" not in modes


def test_f4_bf16_excluded_for_p100_cc60() -> None:
    """P100 (CC 6.0) MUST NOT advertise BF16 OR INT8."""
    from tac import quantization
    with patch.object(quantization.torch.cuda, "is_available", return_value=True), \
         patch.object(quantization.torch.cuda, "get_device_capability", return_value=(6, 0)):
        modes = quantization.get_supported_quantization_modes("cuda:0")
    assert "bf16" not in modes, f"F4 REGRESSION: P100 CC 6.0 wrongly advertises bf16. modes={modes}"
    assert "fp16" in modes
    assert "int8" not in modes
    assert "fp8" not in modes
    assert "fp4" not in modes


def test_f4_bf16_included_for_a100_cc80() -> None:
    """A100 (CC 8.0) MUST advertise BF16 (but NOT FP8 — that's CC 8.9)."""
    from tac import quantization
    with patch.object(quantization.torch.cuda, "is_available", return_value=True), \
         patch.object(quantization.torch.cuda, "get_device_capability", return_value=(8, 0)):
        modes = quantization.get_supported_quantization_modes("cuda:0")
    assert "bf16" in modes, f"F4 REGRESSION: A100 CC 8.0 missing bf16. modes={modes}"
    assert "fp16" in modes
    assert "int8" in modes
    assert "fp8" not in modes


def test_f4_bf16_included_for_rtx4090_cc89() -> None:
    """RTX 4090 (CC 8.9) MUST advertise BF16 + FP8 (but NOT FP4 — CC 10.0)."""
    from tac import quantization
    with patch.object(quantization.torch.cuda, "is_available", return_value=True), \
         patch.object(quantization.torch.cuda, "get_device_capability", return_value=(8, 9)):
        modes = quantization.get_supported_quantization_modes("cuda:0")
    assert modes == {"fp16", "bf16", "int8", "fp8"}, f"F4 mode set wrong: {modes}"


def test_f4_assert_quantization_hardware_supported_rejects_bf16_on_t4() -> None:
    """fail-fast gate: BF16 on T4 must raise."""
    from tac import quantization
    with patch.object(quantization.torch.cuda, "is_available", return_value=True), \
         patch.object(quantization.torch.cuda, "get_device_capability", return_value=(7, 5)):
        with pytest.raises(ValueError, match="bf16.*not.*hardware-supported"):
            quantization.assert_quantization_hardware_supported("bf16", "cuda:0")


# ────────────────────────────────────────────────────────────────────────────
# F5 — contest_auth_eval refuses to start without config.env
# ────────────────────────────────────────────────────────────────────────────


def test_f5_launcher_includes_env_files() -> None:
    """launch_lane_on_vastai.py must include .env in the suffix list.

    Anchor: the literal `.env` substring MUST appear in the
    allowed_suffixes tuple of _enumerate_python_and_shell.
    """
    src = (REPO / "scripts" / "launch_lane_on_vastai.py").read_text()
    assert '".env"' in src, (
        "F5 REGRESSION: launch_lane_on_vastai.py does not include .env in "
        "tarball suffix list. config.env will not deploy → inflate.sh will "
        "fail at extracted/0.mkv."
    )
    # Verify the literal allowed_suffixes definition
    assert 'allowed_suffixes = (".py", ".sh", ".json", ".toml", ".md", ".txt", ".env")' in src, (
        "F5 REGRESSION: allowed_suffixes tuple shape changed; rewrite test."
    )


def test_f5_contest_auth_eval_guards_config_env_presence() -> None:
    """contest_auth_eval.py must hard-fail if config.env is missing.

    Anchor: the source file MUST contain the literal config.env-missing
    SystemExit so every lane that calls contest_auth_eval gets the
    canonical error instead of an opaque ffmpeg crash 200 lines later.
    """
    src = (REPO / "experiments" / "contest_auth_eval.py").read_text()
    assert "config.env" in src, "F5 REGRESSION: no config.env handling"
    assert "PYTHON_INFLATE=renderer" in src, (
        "F5 REGRESSION: contest_auth_eval does not check PYTHON_INFLATE=renderer"
    )
    # And it must use SystemExit (fail-fast) not a warning
    # (we look for SystemExit AND the FATAL: prefix that pairs with it)
    assert "FATAL:" in src and "SystemExit" in src, (
        "F5 REGRESSION: missing fail-fast SystemExit on missing config.env"
    )


def test_f5_robust_current_config_env_sets_python_inflate_renderer() -> None:
    """The actual config.env in the repo MUST set PYTHON_INFLATE=renderer.

    Anchor: this is the value the F5 guard checks for. If the repo file
    drifts to PYTHON_INFLATE=ffmpeg or unset, every Vast.ai eval breaks.
    """
    config = REPO / "submissions" / "robust_current" / "config.env"
    assert config.exists(), "robust_current/config.env missing"
    content = config.read_text()
    assert "PYTHON_INFLATE=renderer" in content, (
        f"F5 REGRESSION: config.env no longer sets PYTHON_INFLATE=renderer\n"
        f"Content:\n{content}"
    )


# ────────────────────────────────────────────────────────────────────────────
# F1 — vastai tracker file gitignored
# ────────────────────────────────────────────────────────────────────────────


def test_f1_gitignore_excludes_vastai_tracker() -> None:
    """vastai_active_instances.json must be gitignored (Codex F1).

    Anchor: the .gitignore literal MUST include the path so the tracker
    cannot be re-committed by accident.
    """
    gi = (REPO / ".gitignore").read_text()
    assert ".omx/state/vastai_active_instances.json" in gi, (
        "F1 REGRESSION: .gitignore does not exclude vastai tracker. "
        "75-record commits could leak again."
    )
