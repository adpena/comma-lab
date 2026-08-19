# SPDX-License-Identifier: MIT
"""The guard-then-destroy class: an fp32 positive floor narrowed to fp16.

A zero-scale guard written as ``clamp(min=1e-8)`` (or ``max(x, 1e-8)``) is
destroyed by a narrowing cast to fp16 applied afterwards, because fp16 cannot
represent 1e-8: **every fp32 value <= 2.980232e-08 casts to EXACTLY 0.0**.  The
guard therefore stores the very zero it exists to prevent, and the divide or
dequantize that follows produces NaN or inf -- silently, with no exception.

Measured breach (2026-08-19), reconstructed from the pre-fix source lines:

  * ``((maxs - mins).float() / 255.0).clamp(min=1e-8).to(torch.float16)`` on a
    constant latent table stored ``[0.0, 0.0, 0.0]`` and the next line's
    ``(latents - mins) / scales`` evaluated to ``[nan, nan, nan]``.
  * ``scale = max(max_abs, 1e-8) / 127.0`` is 7.874e-11 before the cast, so the
    stored fp16 scale was 0.0 and the decoder's ``1/scale`` was inf.  Worse, the
    guard is not even the widest hole: ANY tensor with ``max_abs <= 3.785e-06``
    produced a zero stored scale without the guard ever firing.

The cure is to re-apply the floor AFTER the cast, on the value actually stored
and read back, at fp16's smallest positive (subnormal) value 2**-24.

Both halves are tested here: the numeric contract (the exact breaching values
and the patched call sites) and a CLASS SWEEP over the repository that fails if
any new site narrows a guarded value to fp16 without re-flooring it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.fp16_floor_guard import scan_repo_for_fp16_destroyed_floors

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[3]

# fp16's smallest positive (subnormal) value.  Any fp32 value at or below half
# of this rounds to exactly 0.0 under round-to-nearest-even.
FP16_MIN_POSITIVE = 5.960464477539063e-08
FP16_ROUNDS_TO_ZERO_CEIL = 2.980232238769531e-08


# --------------------------------------------------------------------------
# 1. the numeric contract -- the exact values that breach
# --------------------------------------------------------------------------


def test_the_exact_breaching_values_round_to_zero_in_fp16():
    """The literal floors that were in the source, and what fp16 does to them."""
    for breaching in (1e-8, 1e-8 / 127.0, 1e-12, FP16_ROUNDS_TO_ZERO_CEIL):
        cast = float(torch.tensor([breaching], dtype=torch.float16))
        assert cast == 0.0, f"{breaching!r} must demonstrate the breach, got {cast!r}"

    # ... and the cure is representable exactly, so re-flooring is lossless.
    assert float(torch.tensor([FP16_MIN_POSITIVE], dtype=torch.float16)) == FP16_MIN_POSITIVE

    # The boundary is tight: one ulp above the ceiling already survives.
    just_above = float(torch.tensor([FP16_ROUNDS_TO_ZERO_CEIL * 1.001], dtype=torch.float16))
    assert just_above > 0.0


def test_pre_fix_expression_produced_nan_in_the_real_data_path():
    """Reconstructs the old variant-B line verbatim and shows the NaN it made."""
    latents = torch.full((4, 3), 0.25)
    mins = latents.min(dim=0).values.to(torch.float16)
    maxs = latents.max(dim=0).values.to(torch.float16)

    pre_fix = ((maxs - mins).float() / 255.0).clamp(min=1e-8).to(torch.float16)
    assert bool((pre_fix == 0).all()), "the guard must be shown destroyed"
    assert bool(torch.isnan((latents - mins.float()) / pre_fix.float()).any())

    post_fix = pre_fix.clamp(min=FP16_MIN_POSITIVE)
    assert bool((post_fix > 0).all())
    assert not bool(torch.isnan((latents - mins.float()) / post_fix.float()).any())


# --------------------------------------------------------------------------
# 2. the patched call sites actually behave
# --------------------------------------------------------------------------

_RENDERER_MODULES = (
    "tac.blocknerv_as_renderer",
    "tac.cnerv_as_renderer",
    "tac.dsnerv_as_renderer",
    "tac.e_nerv_as_renderer",
    "tac.ego_nerv_as_renderer",
    "tac.ffnerv_as_renderer",
    "tac.hinerv_as_renderer",
    "tac.lane_12_v2_nerv_as_renderer",
    "tac.mnerv_as_renderer",
    "tac.nervdc_as_renderer",
    "tac.tcnerv_as_renderer",
    "tac.vqvae_as_full_renderer",
)


@pytest.mark.parametrize("module_name", _RENDERER_MODULES)
def test_int8_scale_is_never_stored_as_zero(module_name):
    """The degenerate all-zero tensor is exactly the case the guard exists for."""
    import importlib

    mod = importlib.import_module(module_name)
    quantize = mod._quantize_per_tensor_int8_with_fp16_scale

    for tensor in (
        torch.zeros(8),  # guard fires
        torch.full((8,), 1e-6),  # guard never fires, yet pre-fix stored 0.0
        torch.full((8,), 3.0e-6),  # just under the max_abs <= 3.785e-06 hole
    ):
        _, scale = quantize(tensor)
        assert scale.dtype == torch.float16
        assert float(scale) > 0.0, f"{module_name} stored a zero scale for {tensor[0]!r}"
        assert not torch.isinf(1.0 / scale.float()).any()


@pytest.mark.parametrize("module_name", _RENDERER_MODULES)
def test_normal_weights_are_byte_identical_to_the_pre_fix_encoder(module_name):
    """The cure must not move archive bytes anywhere the encoder already worked."""
    import importlib

    mod = importlib.import_module(module_name)
    quantize = mod._quantize_per_tensor_int8_with_fp16_scale

    torch.manual_seed(0)
    tensor = torch.randn(256)
    codes, scale = quantize(tensor)

    max_abs = float(tensor.abs().max().item())
    pre_fix_scale_f32 = max(max_abs, 1e-8) / 127.0
    pre_fix_scale = torch.tensor([pre_fix_scale_f32], dtype=torch.float16)
    pre_fix_codes = (tensor / pre_fix_scale_f32).round().clamp(-128, 127).to(torch.int8)

    assert scale.numpy().tobytes() == pre_fix_scale.numpy().tobytes()
    assert bool((codes == pre_fix_codes).all())


def test_constant_latent_table_packs_without_nan():
    """Variant B, exercised through the real packer rather than a reconstruction."""
    from tac.ffnerv_as_renderer import _quantize_latent_table_uint8_delta_split

    blob = _quantize_latent_table_uint8_delta_split(torch.full((4, 3), 0.25))
    assert isinstance(blob, bytes) and len(blob) > 0


def test_pose_filler_preflight_survives_perfectly_constant_poses():
    """Variant B in tools/: every delta is 0, so the floor is the only thing
    standing between the divide and a NaN comparison."""
    import importlib.util
    import sys

    # Load by path rather than mutating sys.path, so this test cannot change
    # import resolution for anything that runs after it.
    tool = REPO_ROOT / "tools" / "pr101_pose_filler_stc_anchor.py"
    spec = importlib.util.spec_from_file_location("pr101_pose_filler_for_fp16_test", tool)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    assert mod._can_quantize(torch.zeros(600, 6)) is True


# --------------------------------------------------------------------------
# 3. CLASS SWEEP -- fail on re-introduction anywhere in the repo
#
# The predicate itself lives in ``tac.fp16_floor_guard`` (extracted by
# ``ddm_sp2`` 2026-08-19 when it wired the preflight gate this suite was
# owed). Keeping a second copy here would give ONE class TWO detectors that
# drift apart -- the split-bank failure -- so the local copy is deleted and
# both consumers import the shared module.
# --------------------------------------------------------------------------


def test_no_guarded_value_is_narrowed_to_fp16_without_being_re_floored():
    """A floor followed by an fp16 cast must be re-applied AFTER that cast.

    The predicate now lives in ``tac.fp16_floor_guard`` so that this regression
    sweep and the preflight gate (a Catalog #161 scope extension, wired by
    ``ddm_sp2`` 2026-08-19) are ONE detector rather than two that can drift --
    the split-bank failure. The behaviour is unchanged apart from one
    improvement the shared module adds: docstrings and comments are blanked
    before scanning, so a memo-quoting docstring no longer reads as a live
    violation (the blindness ``ddm_fx3`` named in Catalog #330).
    """
    violations, scanned = scan_repo_for_fp16_destroyed_floors(REPO_ROOT)

    assert scanned > 0, "class sweep found no guard-and-cast sites -- detector is vacuous"
    assert not violations, (
        "fp16 cast destroys a positive floor at "
        f"{len(violations)} site(s); re-apply the floor AFTER the cast at "
        f"{FP16_MIN_POSITIVE!r}:\n  " + "\n  ".join(violations)
    )
