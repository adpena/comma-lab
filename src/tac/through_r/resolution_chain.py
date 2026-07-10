# SPDX-License-Identifier: MIT
"""resolution_chain — THE authoritative pinned R (round-trip) resolution chain.

Canonicalization of the single most-transposition-prone constant surface in the
campaign: the resolution chain that ``upstream/evaluate.py`` + ``upstream/modules.py``
put every scored frame through. Derived by READING the upstream source (verified, not
trusted-from-memory) and pinned here so every through-R harness reads ONE chain instead
of re-declaring it (the flip-resolution bug class the operator flagged:
*"Flip resolution must take upstream evaluate.py and modules.py into account and camera
res and all res used"*).

THE CHAIN (SegNet d_seg authority path, from the source):

    render-grid RGB (h, w, 3) float in [0,255]
      --[R first half]-->  bicubic UP to CAMERA (874, 1164) -> round -> clamp[0,255] -> uint8
      == the stored recon (``upstream/modules.py``: recon is uint8 @ camera 874x1164)
      --[R second half]--> SegNet.preprocess_input: pick last frame (``x[:, -1]``) +
                            bilinear DOWN to SegNet input (384, 512)
      --> SegNet forward -> argmax(dim=1) -> (384, 512) label map
      --> d_seg = mean_pixels(argmax != L*)

The R SECOND HALF (bilinear DOWN to 384x512) is DELEGATED to the real scorer's
``preprocess_input`` (``upstream/modules.py:107-109``), NOT re-implemented here — that
is deliberate: it removes the second place a (H,W) vs (W,H) transposition could be
introduced (there is exactly ONE resize we own, the bicubic UP). :func:`harness.measure_through_r`
composes this module's :func:`render_grid_to_camera_uint8` with the scorer's own preprocess.

THE TRANSPOSITION HAZARD (flagged, load-bearing):
  * The upstream CONSTANT NAMES are (W, H) tuples: ``camera_size = (1164, 874)``,
    ``segnet_model_input_size = (512, 384)`` (``upstream/frame_utils.py:11,13``).
  * ``torch.nn.functional.interpolate(size=...)`` and numpy array ``.shape`` are (H, W).
  So the UP resize target is ``size=(CAMERA_H=874, CAMERA_W=1164)`` and the scorer grid is
  ``(SEG_H=384, SEG_W=512)`` — NEVER the (W,H) tuples. This module keeps BOTH orderings
  explicit with distinct names (``*_SIZE_WH`` vs ``*_H``/``*_W``/``*_HW``) so a call site
  cannot silently cross them. The stored ``lstars`` / ``margins`` are (600, 384, 512) =
  (N, H, W) — the SEG_HW form — confirming the convention.

#149 INTENDED EXCEPTION (camera-res PLACEMENT, not a compare grid): some analytic carriers
place a field at CAMERA resolution before R (a render-time prior); that is a legal PLACEMENT
of a field into the render, distinct from the COMPARE grid (which is always SEG_HW after
argmax). #149 is a placement choice, not a transposition — documented here so an audit does
not mistake a legitimate camera-res field for a crossed resize.

REUSE-not-rederive: the pinned (W,H) constants are IMPORTED from the canonical
:mod:`tac.contest_eval_contract` (``CAMERA_SIZE_WH`` / ``SCORER_INPUT_SIZE_WH`` / ``SEQ_LEN``,
themselves source-verified against upstream snippets). The full-fused numpy R oracle
(``up->uint8@camera->down`` in one call, returning FLOAT scorer-res) is re-exported from
:mod:`tac.local_acceleration.metal_fused_r_operator` as :func:`contest_faithful_R_numpy`
for MLX-parity callers — it is NOT the d_seg path (which stops at camera-uint8 and lets the
real scorer preprocess bilinear-down).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from tac.contest_eval_contract import (
    CAMERA_SIZE_WH,  # (W, H) = (1164, 874)  -- source-verified against upstream
    SCORER_INPUT_SIZE_WH,  # (W, H) = (512, 384)
    SEQ_LEN,  # 2
)

# --- The (W, H) tuples (upstream CONSTANT-NAME convention; provenance/manifest strings). ---
# Kept as the exact objects the contract module holds so a drift there fails our tests too.
CAMERA_SIZE_WH = tuple(int(v) for v in CAMERA_SIZE_WH)  # (1164, 874)
SCORER_INPUT_SIZE_WH = tuple(int(v) for v in SCORER_INPUT_SIZE_WH)  # (512, 384)

# --- The (H, W) forms (numpy shape / torch interpolate convention; the RESIZE targets). ---
CAMERA_W, CAMERA_H = CAMERA_SIZE_WH  # W=1164, H=874
SEG_W, SEG_H = SCORER_INPUT_SIZE_WH  # W=512, H=384
CAMERA_HW: tuple[int, int] = (CAMERA_H, CAMERA_W)  # (874, 1164)
SEG_HW: tuple[int, int] = (SEG_H, SEG_W)  # (384, 512)

RGB_CHANNELS = 3

REPO_ROOT = Path(__file__).resolve().parents[3]
_UPSTREAM = REPO_ROOT / "upstream"


class ResolutionChainError(ValueError):
    """Raised when a constant fails verification against upstream (fail-closed)."""


def _ensure_upstream_on_path() -> None:
    p = str(_UPSTREAM)
    if p not in sys.path:
        sys.path.insert(0, p)


def read_upstream_constants() -> dict[str, Any]:
    """Read the raw resolution constants from ``upstream/frame_utils.py`` (source of truth).

    Returns the (W,H) tuples + ``seq_len`` EXACTLY as upstream declares them. Imports
    upstream (torch) lazily — call it in a verifier/test, not at module import.
    """

    _ensure_upstream_on_path()
    from frame_utils import (  # type: ignore  # upstream
        camera_size,
        segnet_model_input_size,
        seq_len,
    )

    return {
        "camera_size": tuple(int(v) for v in camera_size),
        "segnet_model_input_size": tuple(int(v) for v in segnet_model_input_size),
        "seq_len": int(seq_len),
    }


def verify_against_upstream() -> dict[str, Any]:
    """Assert the pinned constants EQUAL the live upstream values. Fail-closed.

    Cross-checks the (W,H) tuples + ``seq_len`` against ``upstream/frame_utils.py`` AND the
    sister :mod:`tac.contest_eval_contract`. Raises :class:`ResolutionChainError` on ANY
    mismatch (the whole point: if upstream is ever re-pinned, this harness must break loudly,
    not drift silently). Returns the upstream dict on success.
    """

    up = read_upstream_constants()
    if up["camera_size"] != CAMERA_SIZE_WH:
        raise ResolutionChainError(
            f"camera_size drift: pinned {CAMERA_SIZE_WH} != upstream {up['camera_size']}"
        )
    if up["segnet_model_input_size"] != SCORER_INPUT_SIZE_WH:
        raise ResolutionChainError(
            f"segnet_model_input_size drift: pinned {SCORER_INPUT_SIZE_WH} "
            f"!= upstream {up['segnet_model_input_size']}"
        )
    if up["seq_len"] != SEQ_LEN:
        raise ResolutionChainError(f"seq_len drift: pinned {SEQ_LEN} != upstream {up['seq_len']}")
    return up


def render_grid_to_camera_uint8(rgb_render_hwc: np.ndarray) -> np.ndarray:
    """R FIRST HALF: render-grid float RGB ``(h, w, 3)`` -> uint8 camera frame ``(874, 1164, 3)``.

    Bicubic UP to CAMERA_HW, then round + clamp[0,255] -> uint8. Op-for-op mirror of the
    campaign authority ``train_witness_realized_through_R_mlx._torch_R_to_camera_uint8`` and
    ``upstream/modules.py`` (the recon is a uint8 camera-res frame). The SECOND half of R
    (bilinear DOWN to SEG_HW) is done by the real ``SegNet.preprocess_input`` downstream, so
    this function deliberately STOPS at the uint8 camera frame.

    Input already at CAMERA_HW uint8 is returned unchanged (a candidate that is already the
    stored camera recon — the bicubic UP would be a same-size no-op). Any other shape is
    treated as a render grid and up-sampled.
    """

    import torch

    a = np.asarray(rgb_render_hwc)
    if a.ndim != 3 or a.shape[-1] != RGB_CHANNELS:
        raise ResolutionChainError(f"rgb_render_hwc must be (h, w, 3); got {a.shape}")
    if a.shape[0] == CAMERA_H and a.shape[1] == CAMERA_W and a.dtype == np.uint8:
        return np.ascontiguousarray(a)  # already the stored camera recon
    x = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].float()  # (1,3,h,w)
    with torch.inference_mode():
        up = torch.nn.functional.interpolate(
            x, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False
        )
        up = torch.clamp(torch.round(up), 0.0, 255.0)
    return up[0].permute(1, 2, 0).contiguous().numpy().astype(np.uint8)  # (CAMERA_H, CAMERA_W, 3)


def contest_faithful_R_numpy(x_nhwc: np.ndarray, *, ste_round: bool = True) -> np.ndarray:
    """FULL fused numpy R oracle (up->uint8@camera->down) -> FLOAT scorer-res ``(...,384,512,3)``.

    Re-export of :func:`tac.local_acceleration.metal_fused_r_operator.fused_r_forward_numpy`
    pinned to this chain's CAMERA_HW / SEG_HW. This is the MLX/Metal-parity reference (it
    folds the bilinear-down in), NOT the d_seg meter path — the meter stops at camera-uint8
    and delegates the down-resize to the real scorer's ``preprocess_input``. Provided so a
    caller can check the two are consistent (an optional parity anchor).
    """

    from tac.local_acceleration.metal_fused_r_operator import fused_r_forward_numpy

    return fused_r_forward_numpy(
        x_nhwc, camera_hw=CAMERA_HW, output_hw=SEG_HW, ste_round=bool(ste_round)
    )


def describe() -> dict[str, Any]:
    """Machine-readable provenance dump of the whole pinned chain (max-observability)."""

    return {
        "schema": "through_r_resolution_chain.v1",
        "seq_len": SEQ_LEN,
        "orderings": {
            "WH_convention": "upstream constant NAMES (camera_size / segnet_model_input_size)",
            "HW_convention": "numpy .shape + torch interpolate size (the RESIZE targets)",
            "camera_size_wh": list(CAMERA_SIZE_WH),
            "camera_hw": list(CAMERA_HW),
            "segnet_input_size_wh": list(SCORER_INPUT_SIZE_WH),
            "seg_hw": list(SEG_HW),
        },
        "chain": [
            "render-grid RGB (h,w,3) float[0,255]",
            f"R.up: bicubic -> CAMERA_HW={list(CAMERA_HW)} -> round -> clamp[0,255] -> uint8 (stored recon)",
            f"R.down (SegNet.preprocess_input): x[:,-1] last-frame + bilinear -> SEG_HW={list(SEG_HW)}",
            "SegNet forward -> argmax(dim=1) -> (384,512) label map",
            "d_seg = mean_pixels(argmax != L*)",
        ],
        "r_second_half_owner": "upstream/modules.py::SegNet.preprocess_input (delegated, not re-implemented)",
        "camera_res_placement_149": (
            "carriers MAY place a field at CAMERA res before R (render-time prior); that is a "
            "legal PLACEMENT, distinct from the COMPARE grid (always SEG_HW post-argmax). "
            "NOT a transposition."
        ),
        "transposition_hazard": (
            "WH tuples are provenance-only; resize targets are the HW forms. Crossing them "
            "(feeding (W,H) to interpolate) is the flip-resolution bug class this module extincts."
        ),
        "source_of_truth": {
            "camera_size": "upstream/frame_utils.py:11 (verify_against_upstream asserts parity)",
            "segnet_model_input_size": "upstream/frame_utils.py:13",
            "segnet_preprocess": "upstream/modules.py:107-109 (x[:,-1] + interpolate to (H,W))",
            "score_formula": "upstream/evaluate.py:92 (100*seg + sqrt(10*pose) + 25*rate)",
            "pinned_constants_reused_from": "tac.contest_eval_contract",
        },
    }


__all__ = [
    "CAMERA_H",
    "CAMERA_HW",
    "CAMERA_SIZE_WH",
    "CAMERA_W",
    "RGB_CHANNELS",
    "SCORER_INPUT_SIZE_WH",
    "SEG_H",
    "SEG_HW",
    "SEG_W",
    "SEQ_LEN",
    "ResolutionChainError",
    "contest_faithful_R_numpy",
    "describe",
    "read_upstream_constants",
    "render_grid_to_camera_uint8",
    "verify_against_upstream",
]
