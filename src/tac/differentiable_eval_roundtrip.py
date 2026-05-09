"""PR #95-faithful eval-roundtrip + autograd-preserving rgb_to_yuv6 for inner-loop training.

This module is the **canonical replication** of two findings from the binary
forensics dossier on Aaron's PR #95 / PR #106 ("belt_and_suspenders") submission
(see ``.omx/research/hnerv_leaderboard_binary_forensics_dossier_20260509.md``):

  Finding A — eval_roundtrip baked into the TRAINING inner loop
    Every inner training step takes the rendered RGB through the FULL contest
    eval roundtrip (384 → 874 bicubic-up → uint8 → 384 bilinear-down → clamp →
    STE-round) BEFORE the loss is computed, so the proxy gradient matches the
    contest-eval gradient at architecturally-correct precision. Without this,
    the proxy-eval gap is 2-11× on PoseNet (per CLAUDE.md
    ``feedback_proxy_auth_math_useless``).

  Finding B — autograd-preserving rgb_to_yuv6
    Upstream ``frame_utils.rgb_to_yuv6`` is decorated with ``@torch.no_grad()``
    AND uses in-place ``clamp_()`` calls. PoseNet's ``preprocess_input``
    delegates to it, so without a patch the pose loss gradient never reaches
    the renderer. Aaron's PR #95 quote: *"pose plateaued at 142 across 2500+
    epochs"* without the YUV monkey-patch.

The PR #95 source files are at::

  experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/
    source/submissions/belt_and_suspenders/src/data.py:51-81           (Finding B)
    source/submissions/belt_and_suspenders/src/stages/common.py:179-194 (Finding A)

These are READ-ONLY ORACLES per CLAUDE.md ``forbidden_in_place_edits_to_public_PR_intake_clones``.

Public API
----------

``apply_eval_roundtrip_during_training(rgb_tensor, *, simulate_uint8=True,
simulate_resize=True, ste_round=True, target_h=874, target_w=1164) -> Tensor``
    Inner-loop wrapper around the full contest eval roundtrip with the same
    upscale/quantize/downscale ladder PR #95 uses, with autograd preserved
    via straight-through-estimator on the round step.

``differentiable_rgb_to_yuv6(rgb_chw) -> Tensor``
    Autograd-friendly drop-in for ``frame_utils.rgb_to_yuv6``. Numerically
    equivalent (out-of-place ``clamp`` instead of in-place ``clamp_``, no
    ``@torch.no_grad`` decorator).

``patch_upstream_yuv6_globally() -> dict``
    Aaron's monkey-patch approach (Finding B style): overwrites
    ``frame_utils.rgb_to_yuv6`` AND ``modules.rgb_to_yuv6`` with the
    differentiable version. Returns a token used by ``unpatch_upstream_yuv6``.

``unpatch_upstream_yuv6(token) -> None``
    Restores upstream's original ``rgb_to_yuv6``. Use in test teardown.

``assert_yuv6_forward_equivalence_to_upstream(*, atol=1e-6) -> dict``
    Test-time golden-vector check that ``differentiable_rgb_to_yuv6`` matches
    upstream output bit-equivalence within ``atol``.

``yuv6_routing_mode`` enum: ``MONKEY_PATCH_GLOBAL`` (Aaron's path) or
``TAC_DIFFERENTIABLE_ROUTING`` (cleaner path). Exposes both per the
non-arbitrariness principle (``feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509``).
The ``tools/probe_yuv6_differentiability_disambiguator.py`` arbitrates.

Cross-references
----------------

CLAUDE.md non-negotiables:
  - "eval_roundtrip — NON-NEGOTIABLE, HIGHEST EMPHASIS"  (existing)
  - NEW: "Eval-roundtrip + autograd-YUV6 in training inner loop — NON-NEGOTIABLE"
    (proposed addition: ``.omx/research/CLAUDE_md_addition_eval_roundtrip_inner_loop_yuv6_20260509.md``)

Existing surfaces this module composes with:
  - ``tac.constrained_gen.rgb_to_yuv6`` — already byte-equivalent to upstream
  - ``tac.scorer.make_scorers_differentiable`` — closure-style monkey-patch on
    individual posenet/segnet instances (used by ``load_differentiable_scorers``)
  - ``tac.renderer.simulate_eval_roundtrip`` — older bilinear+bilinear+noise STE
    variant (kept for backward compatibility with score_gradient trainer)
  - ``tac.training._patch_scorers_for_training`` — internal helper inside the
    Trainer class (the differentiable-yuv6 closure is duplicated there)

Forensics provenance: subagent a30f2ade flagged this as the single biggest
"thing they did right that we didn't" from the PR #95 binary-forensics dossier.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from tac.quantization import Uint8STE


# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

#: Camera-native resolution (height, width) used by the contest eval roundtrip.
#: Matches PR #95 source: ``F.interpolate(flat, size=(874, 1164), mode='bicubic', ...)``.
CAMERA_HW: tuple[int, int] = (874, 1164)

#: Scorer-input resolution (height, width). Matches PR #95 source:
#: ``F.interpolate(up, size=(384, 512), mode='bilinear', ...)``.
SCORER_HW: tuple[int, int] = (384, 512)


# --------------------------------------------------------------------------- #
# Routing mode (non-arbitrariness: ship both, let probe arbitrate)             #
# --------------------------------------------------------------------------- #


class Yuv6RoutingMode(str, Enum):
    """How the trainer should achieve autograd-friendly rgb_to_yuv6.

    MONKEY_PATCH_GLOBAL
        Overwrite ``frame_utils.rgb_to_yuv6`` and ``modules.rgb_to_yuv6`` at
        import time (Aaron's PR #95 approach, see ``data.py:80-81``). Affects
        every consumer in the process and survives across calls. Highest
        empirical confidence (this is the verified-working PR #95 recipe).

    TAC_DIFFERENTIABLE_ROUTING
        Patch only the posenet/segnet instances passed to the trainer
        (``tac.scorer.make_scorers_differentiable``). Cleaner — does not mutate
        upstream module-level globals — but requires every scorer construction
        site to opt in. Architecturally preferred; equivalent in PoseNet
        gradient flow when consistently applied.

    AUTO
        Run ``tools/probe_yuv6_differentiability_disambiguator.py`` and select
        the mode whose pose gradient is non-zero on the calibration batch. If
        both pass (expected), default to MONKEY_PATCH_GLOBAL because it is the
        verified-working recipe with empirical PR #95 anchor.
    """

    MONKEY_PATCH_GLOBAL = "monkey_patch_global"
    TAC_DIFFERENTIABLE_ROUTING = "tac_differentiable_routing"
    AUTO = "auto"


# --------------------------------------------------------------------------- #
# Differentiable rgb_to_yuv6 (Finding B)                                       #
# --------------------------------------------------------------------------- #


def differentiable_rgb_to_yuv6(rgb_chw: torch.Tensor) -> torch.Tensor:
    """Differentiable BT.601 RGB -> YUV6 with 4:2:0 chroma subsampling.

    Numerically equivalent to upstream ``frame_utils.rgb_to_yuv6`` modulo:
      - no ``@torch.no_grad()`` decorator (gradients flow through),
      - out-of-place ``clamp`` instead of in-place ``clamp_`` (autograd-safe),
      - identical BT.601 coefficients (kYR=0.299, kYG=0.587, kYB=0.114),
      - identical 4:2:0 chroma subsampling (mean of 2x2 block).

    Args:
        rgb_chw: ``(..., 3, H, W)`` float tensor in ``[0, 255]``.

    Returns:
        ``(..., 6, H//2, W//2)`` tensor stacked as ``[y00, y10, y01, y11, U_sub, V_sub]``.

    Reference:
        BT.601 full-range YUV with 4:2:0 chroma subsampling. Validated against
        ``upstream/frame_utils.py::rgb_to_yuv6`` to 0 absolute error on random
        inputs (see :func:`assert_yuv6_forward_equivalence_to_upstream`).
    """
    if rgb_chw.dim() < 3:
        raise ValueError(
            f"differentiable_rgb_to_yuv6 requires (..., 3, H, W); got shape {tuple(rgb_chw.shape)}"
        )
    if rgb_chw.shape[-3] != 3:
        raise ValueError(
            f"differentiable_rgb_to_yuv6 expects 3 channels at dim -3; "
            f"got {rgb_chw.shape[-3]}"
        )
    H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
    H2, W2 = H // 2, W // 2
    rgb = rgb_chw[..., :, : 2 * H2, : 2 * W2]

    R = rgb[..., 0, :, :]
    G = rgb[..., 1, :, :]
    B = rgb[..., 2, :, :]

    # BT.601 full-range luma/chroma.
    Y = (R * 0.299 + G * 0.587 + B * 0.114).clamp(0.0, 255.0)
    U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
    V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)

    # 4:2:0 chroma subsampling: average of each 2x2 block.
    U_sub = (
        U[..., 0::2, 0::2] + U[..., 1::2, 0::2]
        + U[..., 0::2, 1::2] + U[..., 1::2, 1::2]
    ) * 0.25
    V_sub = (
        V[..., 0::2, 0::2] + V[..., 1::2, 0::2]
        + V[..., 0::2, 1::2] + V[..., 1::2, 1::2]
    ) * 0.25

    y00 = Y[..., 0::2, 0::2]
    y10 = Y[..., 1::2, 0::2]
    y01 = Y[..., 0::2, 1::2]
    y11 = Y[..., 1::2, 1::2]
    return torch.stack([y00, y10, y01, y11, U_sub, V_sub], dim=-3)


# --------------------------------------------------------------------------- #
# Eval-roundtrip baked into training inner loop (Finding A)                    #
# --------------------------------------------------------------------------- #


def apply_eval_roundtrip_during_training(
    rgb_tensor: torch.Tensor,
    *,
    simulate_uint8: bool = True,
    simulate_resize: bool = True,
    ste_round: bool = True,
    target_h: int = CAMERA_HW[0],
    target_w: int = CAMERA_HW[1],
) -> torch.Tensor:
    """Apply the contest eval roundtrip to a rendered RGB tensor with autograd preserved.

    PR #95 ``stages/common.py:178-185`` recipe::

        flat = decoded_pair.reshape(B * 2, 3, EVAL_SIZE[0], EVAL_SIZE[1])
        up = F.interpolate(flat, size=(874, 1164), mode='bicubic', align_corners=False)
        down = F.interpolate(up, size=(384, 512), mode='bilinear', align_corners=False)
        # ...
        decoded_bhwc = Uint8STE.apply(decoded_bhwc)

    Note PR #95 uses bicubic UP and bilinear DOWN (asymmetric), and applies the
    STE round AFTER the resize roundtrip in HWC layout. We replicate the same
    sequence in CHW layout (the layout the renderer emits before scorer preprocess).

    The STE step makes the gradient pass through the round-to-uint8 cliff
    (``round.detach()`` cancels the round forward, leaves identity backward).
    Without it, the renderer learns textures that survive in float-space but
    collapse under uint8 quantization at eval time (the classic proxy-auth
    drift class, CLAUDE.md ``feedback_proxy_auth_math_useless``).

    Args:
        rgb_tensor: ``(..., 3, H, W)`` float in ``[0, 255]`` at scorer resolution
            (typically ``H=384, W=512``). Layout: NCHW or BTCHW (any leading
            dims; the spatial roundtrip applies to last two dims).
        simulate_uint8: if False, skip the STE clamp+round (debug only).
        simulate_resize: if False, skip the bicubic-up + bilinear-down
            (debug only; produces invalid eval-faithful gradients).
        ste_round: if False, the STE round step is skipped (debug only;
            still does clamp).
        target_h: camera-native height (default 874 per contest evaluate.py).
        target_w: camera-native width (default 1164 per contest evaluate.py).

    Returns:
        Tensor of the same leading shape as input, spatial dims matching input.

    Raises:
        ValueError: on dtype/shape/range mismatch.
    """
    if rgb_tensor.dim() < 3:
        raise ValueError(
            f"apply_eval_roundtrip_during_training requires (..., 3, H, W); "
            f"got shape {tuple(rgb_tensor.shape)}"
        )
    if rgb_tensor.shape[-3] != 3:
        raise ValueError(
            f"apply_eval_roundtrip_during_training expects 3 channels at dim -3; "
            f"got {rgb_tensor.shape[-3]}"
        )
    if not rgb_tensor.is_floating_point():
        raise ValueError(
            f"apply_eval_roundtrip_during_training requires a float tensor; "
            f"got {rgb_tensor.dtype}"
        )
    orig_shape = rgb_tensor.shape
    orig_h, orig_w = orig_shape[-2], orig_shape[-1]

    # Collapse leading dims to (N, 3, H, W) for F.interpolate.
    flat = rgb_tensor.reshape(-1, 3, orig_h, orig_w)

    if simulate_resize:
        # PR #95 uses bicubic UP, bilinear DOWN. (Asymmetric on purpose; this
        # matches the contest evaluate.py ladder — see binary forensics.)
        up = F.interpolate(
            flat, size=(target_h, target_w), mode="bicubic", align_corners=False
        )
        down = F.interpolate(
            up, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
    else:
        down = flat

    if simulate_uint8:
        if ste_round:
            # Canonical STE: forward is exact uint8 clamp/round; backward is
            # identity inside range and zero outside saturation.
            out_flat = Uint8STE.apply(down)
        else:
            out_flat = down.clamp(0.0, 255.0)
    else:
        out_flat = down

    return out_flat.reshape(orig_shape)


# --------------------------------------------------------------------------- #
# Global monkey-patch (Aaron's path)                                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Yuv6PatchToken:
    """Opaque token used to revert a global rgb_to_yuv6 monkey-patch.

    Returned by :func:`patch_upstream_yuv6_globally`. Pass back into
    :func:`unpatch_upstream_yuv6` to restore the original implementations.
    """

    frame_utils_orig: Callable[..., torch.Tensor] | None
    modules_orig: Callable[..., torch.Tensor] | None
    frame_utils_was_patched: bool
    modules_was_patched: bool


def _resolve_upstream_modules() -> tuple[Any | None, Any | None]:
    """Locate (frame_utils, modules) imports from the upstream challenge repo.

    Tries import directly; if that fails, walks REPO_ROOT/upstream onto sys.path
    and retries. Returns ``(frame_utils, modules)``; either may be ``None`` if
    the import is unavailable (for tests that don't ship upstream).
    """
    try:
        frame_utils = importlib.import_module("frame_utils")
    except ImportError:
        # Try injecting the repo's upstream dir.
        repo_upstream = Path(__file__).resolve().parents[2] / "upstream"
        if repo_upstream.is_dir():
            if str(repo_upstream) not in sys.path:
                sys.path.insert(0, str(repo_upstream))
            try:
                frame_utils = importlib.import_module("frame_utils")
            except ImportError:
                frame_utils = None
        else:
            frame_utils = None
    try:
        modules = importlib.import_module("modules")
    except ImportError:
        modules = None
    return frame_utils, modules


def patch_upstream_yuv6_globally() -> Yuv6PatchToken:
    """Monkey-patch upstream ``frame_utils.rgb_to_yuv6`` AND ``modules.rgb_to_yuv6``.

    Mirrors PR #95 ``data.py:80-81``::

        frame_utils.rgb_to_yuv6 = _rgb_to_yuv6_differentiable
        modules.rgb_to_yuv6 = _rgb_to_yuv6_differentiable

    Both module-level references must be overwritten because ``modules.py``
    already imported ``rgb_to_yuv6`` from ``frame_utils`` at its own import
    time, so a single patch on ``frame_utils`` does NOT propagate to consumers
    that imported via ``from frame_utils import rgb_to_yuv6``.

    Idempotent: re-patching with an identical function is a no-op (the
    returned token will reflect the original state).

    Returns:
        :class:`Yuv6PatchToken` — pass to :func:`unpatch_upstream_yuv6` to revert.
    """
    frame_utils, modules = _resolve_upstream_modules()
    fu_orig = None
    md_orig = None
    fu_patched = False
    md_patched = False
    if frame_utils is not None and hasattr(frame_utils, "rgb_to_yuv6"):
        fu_orig = frame_utils.rgb_to_yuv6
        if fu_orig is not differentiable_rgb_to_yuv6:
            frame_utils.rgb_to_yuv6 = differentiable_rgb_to_yuv6
            fu_patched = True
    if modules is not None and hasattr(modules, "rgb_to_yuv6"):
        md_orig = modules.rgb_to_yuv6
        if md_orig is not differentiable_rgb_to_yuv6:
            modules.rgb_to_yuv6 = differentiable_rgb_to_yuv6
            md_patched = True
    return Yuv6PatchToken(
        frame_utils_orig=fu_orig,
        modules_orig=md_orig,
        frame_utils_was_patched=fu_patched,
        modules_was_patched=md_patched,
    )


def unpatch_upstream_yuv6(token: Yuv6PatchToken) -> None:
    """Restore the upstream ``rgb_to_yuv6`` references monkey-patched by
    :func:`patch_upstream_yuv6_globally`.

    Safe to call even if no patching occurred (the token records that).
    """
    frame_utils, modules = _resolve_upstream_modules()
    if token.frame_utils_was_patched and frame_utils is not None and token.frame_utils_orig is not None:
        frame_utils.rgb_to_yuv6 = token.frame_utils_orig
    if token.modules_was_patched and modules is not None and token.modules_orig is not None:
        modules.rgb_to_yuv6 = token.modules_orig


# --------------------------------------------------------------------------- #
# Forward-equivalence regression check                                         #
# --------------------------------------------------------------------------- #


def assert_yuv6_forward_equivalence_to_upstream(
    *,
    num_samples: int = 5,
    atol: float = 1e-6,
    seed: int = 20260509,
) -> dict[str, Any]:
    """Verify ``differentiable_rgb_to_yuv6`` matches upstream within ``atol``.

    Returns a dict with ``passed``, ``max_abs_error``, ``num_samples``, and
    per-sample ``details``. Raises ``AssertionError`` if the upstream module is
    importable AND any sample exceeds ``atol``.

    Args:
        num_samples: random RGB tensors to compare. Default 5.
        atol: absolute-error tolerance. Default 1e-6 (BT.601 coefficients are
            exact rationals; expected error is 0.0).
        seed: torch RNG seed for reproducibility.

    Raises:
        AssertionError: if the maximum observed error exceeds ``atol``.
        RuntimeError: if upstream ``frame_utils.rgb_to_yuv6`` cannot be imported.
    """
    frame_utils, _ = _resolve_upstream_modules()
    if frame_utils is None or not hasattr(frame_utils, "rgb_to_yuv6"):
        raise RuntimeError(
            "upstream frame_utils.rgb_to_yuv6 not importable; cannot validate "
            "forward equivalence. (This typically means the test environment "
            "is missing the comma_video_compression_challenge upstream tree.)"
        )
    upstream_yuv6 = frame_utils.rgb_to_yuv6

    g = torch.Generator()
    g.manual_seed(seed)

    details = []
    max_err = 0.0
    for _ in range(num_samples):
        rgb = torch.rand((1, 3, 128, 128), generator=g) * 255.0
        with torch.no_grad():
            upstream_out = upstream_yuv6(rgb)
        local_out = differentiable_rgb_to_yuv6(rgb)
        if local_out.shape != upstream_out.shape:
            details.append(float("inf"))
            max_err = float("inf")
            continue
        err = (local_out - upstream_out).abs().max().item()
        details.append(err)
        max_err = max(max_err, err)

    passed = max_err <= atol
    result = {
        "passed": passed,
        "max_abs_error": max_err,
        "num_samples": num_samples,
        "atol": atol,
        "details": details,
    }
    if not passed:
        raise AssertionError(
            f"differentiable_rgb_to_yuv6 forward-equivalence check FAILED: "
            f"max_abs_error={max_err:.3e} exceeds atol={atol:.3e}. Details: {details}"
        )
    return result


__all__ = [
    "CAMERA_HW",
    "SCORER_HW",
    "Yuv6RoutingMode",
    "Yuv6PatchToken",
    "differentiable_rgb_to_yuv6",
    "apply_eval_roundtrip_during_training",
    "patch_upstream_yuv6_globally",
    "unpatch_upstream_yuv6",
    "assert_yuv6_forward_equivalence_to_upstream",
]
