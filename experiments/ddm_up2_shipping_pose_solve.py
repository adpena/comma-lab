"""ddm_up2 -- the first pose solve aimed at the object that actually ships.

Why this module exists
----------------------
Two paid pose refusals (``ddm_ps1u`` r2 at ``+1.686e-02`` S, ``ddm_t1h`` at
``+0.012557`` S) both minimised d_pose against a GT decoded by PyAV. ``ddm_up1``
then measured that the PyAV GT and the DALI GT differ by **23.74x** on identical
frames, and this module re-derives *structurally* -- not from that numeric
agreement -- which lineage the contest actually scores:

    upstream/evaluate.py:31-42   device.type == "cuda" -> DefaultDatasetClass = DaliVideoDataset
                                 else                  -> DefaultDatasetClass = AVVideoDataset
    upstream/evaluate.py:58      ds_gt = DefaultDatasetClass(...)
    upstream/frame_utils.py:113  DaliVideoDataset asserts device.type == 'cuda'
    upstream/frame_utils.py:188  AVVideoDataset  asserts device.type != 'cuda'

The binding is bijective and assert-enforced. A ``[contest-CUDA]`` row therefore
scores against DALI-lineage GT; a ``[contest-CPU]`` row scores against PyAV. They
are *different objectives*, not a hardware drift. The live pointer is a
contest-CUDA T4 row, so the DALI targets are the ones that ship, and every prior
solve was pointed at the other object.

The vehicle
-----------
Within a pair, frame 1 is the semantic master and frame 0 is the pose carrier::

    cpr1/inflate.py:312-328   frame 2p+1 <- SemanticTokenRenderer(tokens)      [seg]
    cpr1/inflate.py:335-352   frame 2p   <- einsum(coeff[p], basis) carrier    [pose]

``upstream/modules.py:108`` slices ``x[:, -1, ...]`` -- SegNet sees frame 1 ONLY.
So the carrier coefficients cannot move d_seg. The seg-hold here is *structural*,
proven by the slice, and is additionally measured as an argmax-identity control.

That leaves 12 free coefficients per pair against 6 pose equations per pair, with
every pair independent. This module solves those 600 independent 12-DOF problems
against the DALI targets, realises the solution on the shipped int12 lattice, and
re-measures through the same instrument.

Authority: frozen CPU-torch PoseNet, ``[macOS-CPU advisory]``. ``score_claim=false``,
``promotable=false``. Only ``upstream/evaluate.py`` on contest hardware is a score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
UPSTREAM = REPO / "upstream"

N_PAIRS_TOTAL = 600
POSE_DIMS = 6
CARRIER_DIM = 12
EVAL_H, EVAL_W = 384, 512
CAMERA_H, CAMERA_W = 874, 1164
CARRIER_AMPLITUDE = 64.0
COEFF_CODE_MIN, COEFF_CODE_MAX = -2048, 2047

DEFAULT_RUNTIME = Path(
    "/Volumes/APDataStore/pact/ddm_to1/generations/to1_tail_override_r1"
)
DEFAULT_RAW = Path(
    "/Volumes/APDataStore/pact/ddm_to1/advisory/attempt_0002/work/inflated/0.raw"
)
DEFAULT_DALI_GT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_dali.pt"
)
DEFAULT_AV_GT = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"

# The pointer body, re-read from ddm_to1's own receipts (never inferred).
POINTER_ARCHIVE_SHA256 = (
    "50e561454b23026d3870f056747e848a49bd5f2b1e23930155d1281aeee91927"
)
CPU_DECODE_RAW_SHA256 = (
    "ccbfa3327d0f2486f8a2d7970fe89c5d56302eb1e04714d05eabff52278f1f9d"
)
CUDA_DECODE_RAW_SHA256 = (
    "3c810cc4adff01a6783e8727f2cd7161e47d83693acc2aba7941b8ee7b115f6d"
)
# T4 row r1 (contest-CUDA, Tesla T4, n600) -- the row being improved on.
POINTER_D_POSE_T4 = 7.77e-06
POINTER_D_SEG_T4 = 0.00030309
POINTER_ARCHIVE_BYTES = 176_420
POINTER_SCORE = 0.15659459685822907
# 25 / 37_545_489 -- one archive byte in score units.
BYTE_TO_SCORE = 25.0 / 37_545_489.0
# upstream/evaluate.py:95 prints d_pose at 8 decimals; half-ULP of that report.
REPORT_HALF_ULP = 0.5e-8


class Up2Error(RuntimeError):
    """A ddm_up2 precondition failed. Always fail closed, never approximate."""


# --------------------------------------------------------------------------
# GT lineage -- the gate that ``ddm_pi2`` specified and nobody built.
# --------------------------------------------------------------------------

LINEAGE_DALI = "dali"
LINEAGE_AV_PYAV = "av_pyav"
#: Which GT lineage each score axis is actually scored against, derived from
#: ``upstream/evaluate.py:31-42`` + the two asserts in ``upstream/frame_utils.py``.
AXIS_GT_LINEAGE = {
    "contest_cuda": LINEAGE_DALI,
    "contest_cpu": LINEAGE_AV_PYAV,
    "macos_cpu_advisory": LINEAGE_AV_PYAV,
}


def required_lineage_for_axis(axis: str) -> str:
    """Return the GT lineage an axis must be scored against, or refuse."""
    try:
        return AXIS_GT_LINEAGE[axis]
    except KeyError as error:
        raise Up2Error(
            f"unknown score axis {axis!r}; known axes: {sorted(AXIS_GT_LINEAGE)}"
        ) from error


def verify_gt_lineage(*, axis: str, declared_lineage: str) -> dict[str, str]:
    """Fail closed unless the GT lineage matches what the axis actually scores.

    ``ddm_up1`` measured 23.74x between the two lineages on identical frames, and
    two paid rows were refused because a solve minimised against the wrong one.
    Any tool that consumes a GT cache to produce a pose verdict must call this.
    """
    required = required_lineage_for_axis(axis)
    if declared_lineage != required:
        raise Up2Error(
            f"GT lineage mismatch: axis {axis!r} is scored against {required!r} "
            f"but the supplied GT cache is {declared_lineage!r}. "
            "ddm_up1 measured these two lineages 23.74x apart on identical "
            "frames; solving against the wrong one has already bought two "
            "refused paid rows (ddm_ps1u r2, ddm_t1h)."
        )
    return {"axis": axis, "gt_lineage": declared_lineage, "status": "VERIFIED"}


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_gt_poses(path: Path) -> tuple[np.ndarray, str]:
    """Load GT pose targets together with their decode lineage.

    The lineage is *returned*, never inferred by the caller: an instrument that
    does not know which GT it holds is measuring a different object than the one
    that ships.
    """
    if not path.is_file():
        raise Up2Error(f"gt cache does not exist: {path}")
    if path.suffix == ".pt":
        import torch

        cache = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(cache, dict) or "pose" not in cache:
            raise Up2Error(f"gt .pt cache has no 'pose' key: {path}")
        poses = np.asarray(cache["pose"], dtype=np.float64)
        lineage = LINEAGE_DALI if "dali" in path.name.lower() else "unknown_pt"
    else:
        with np.load(path) as cache:
            if "gt_poses" not in cache.files:
                raise Up2Error(f"gt cache has no gt_poses: {sorted(cache.files)}")
            poses = np.asarray(cache["gt_poses"], dtype=np.float64)
        lineage = LINEAGE_AV_PYAV
    if poses.shape != (N_PAIRS_TOTAL, POSE_DIMS):
        raise Up2Error(
            f"gt poses have shape {poses.shape}, expected ({N_PAIRS_TOTAL}, {POSE_DIMS})"
        )
    return poses, lineage


# --------------------------------------------------------------------------
# Carrier state -- basis, coefficients, selector, straight from the archive.
# --------------------------------------------------------------------------


@dataclass
class CarrierState:
    """Everything frame 0 is made of, as the shipped receiver reconstructs it."""

    basis_raw: Any  # (12, 3, 24, 32) float32, pre-normalisation
    basis_norm: Any  # (12, 3, 384, 512) float32, post normalized_basis()
    coefficients: Any  # (600, 12) float32 -- codes * scales, post-compensation
    codes: Any  # (600, 12) int32 -- the realised lattice points
    coefficient_scales: Any  # (12,) float32
    selector_modes: tuple
    selector_choices: Any  # (600,) uint8
    has_compensation: bool
    renderer: Any


def load_carrier_state(runtime_dir: Path, *, verify_archive: bool = True) -> CarrierState:
    """Reconstruct the pose carrier exactly as ``f26_inflate`` does.

    Mirrors ``runtime/f26_inflate.py:450-476`` -- split selector, materialise the
    cpr1 carrier, unpack, then apply the compensation overlay to the codes.
    """
    import torch

    runtime_dir = runtime_dir.resolve()
    archive_path = runtime_dir / "archive.zip"
    if verify_archive:
        observed = _sha256_file(archive_path)
        if observed != POINTER_ARCHIVE_SHA256:
            raise Up2Error(
                f"archive.zip is not the pointer body: {observed} != {POINTER_ARCHIVE_SHA256}"
            )

    sys.path.insert(0, str(runtime_dir))
    sys.path.insert(0, str(runtime_dir / "cpr1"))
    try:
        import inflate as renderer_module  # type: ignore[import-not-found]
        from runtime.carrier_repack import (  # type: ignore[import-not-found]
            materialize_cpr1,
            split_frame0_selector_carrier,
        )
        from runtime.compensation_overlay import (  # type: ignore[import-not-found]
            apply_compensation_overlay,
        )
        from runtime.frame0_selector import decode_selector  # type: ignore[import-not-found]
        from runtime.residual_archive import (  # type: ignore[import-not-found]
            read_residual_archive,
        )
    finally:
        sys.path.pop(0)
        sys.path.pop(0)

    renderer = renderer_module
    parts = read_residual_archive(archive_path)
    carrier_blob, selector_blob = split_frame0_selector_carrier(parts.carrier_blob)
    canonical_carrier = materialize_cpr1(carrier_blob, renderer)

    semantic_width_marker = bytes(40_252)
    semantic_pose = (
        struct.pack("<II", len(semantic_width_marker), len(canonical_carrier))
        + semantic_width_marker
        + canonical_carrier
    )
    _, basis_raw, coefficients = renderer.unpack_semantic_pose(semantic_pose)

    basis_count = CARRIER_DIM * 3 * renderer.CARRIER_H * renderer.CARRIER_W
    _, _, coefficient_scales_array, encoded = renderer.decode_compact_carrier(
        canonical_carrier,
        basis_count=basis_count,
        frames=N_PAIRS_TOTAL,
        dimensions=CARRIER_DIM,
    )
    delta = (encoded.astype(np.int64) >> 1) ^ -(encoded.astype(np.int64) & 1)
    base_codes = np.cumsum(delta, axis=0) & 0xFFF
    base_codes = np.where(base_codes >= 0x800, base_codes - 0x1000, base_codes).astype(
        np.int32
    )
    coefficient_scales = torch.from_numpy(coefficient_scales_array)
    expected_base = torch.from_numpy(base_codes).float() * coefficient_scales[None]
    if not torch.equal(coefficients, expected_base):
        raise Up2Error("compensation base-code reconstruction differs from the carrier")

    codes = base_codes
    has_compensation = parts.compensation_blob is not None
    if has_compensation:
        codes = apply_compensation_overlay(base_codes, parts.compensation_blob)
        coefficients = torch.from_numpy(codes).float() * coefficient_scales[None]

    if selector_blob is None:
        selector_modes: tuple = ()
        selector_choices = np.zeros(N_PAIRS_TOTAL, dtype=np.uint8)
    else:
        selector_modes, selector_choices = decode_selector(selector_blob)

    basis_norm = renderer.normalized_basis(basis_raw.clone())
    return CarrierState(
        basis_raw=basis_raw,
        basis_norm=basis_norm,
        coefficients=coefficients,
        codes=np.asarray(codes, dtype=np.int32),
        coefficient_scales=coefficient_scales,
        selector_modes=selector_modes,
        selector_choices=np.asarray(selector_choices, dtype=np.uint8),
        has_compensation=has_compensation,
        renderer=renderer,
    )


# --------------------------------------------------------------------------
# The frame-0 forward model -- byte-exact replica of the shipped receiver.
# --------------------------------------------------------------------------


def _round_ste(tensor):
    """round() with a straight-through gradient.

    The receiver rounds twice (``cpr1/inflate.py:342`` and ``:348``). Both are
    step functions, so an exact gradient is zero almost everywhere. STE keeps the
    *forward* value bit-exact with the receiver while giving the solver a usable
    descent direction; every accepted step is then re-measured through the exact
    non-STE path, so the STE never enters a reported number.
    """
    return tensor + (tensor.round() - tensor).detach()


def render_frame0_float(
    coefficients,
    basis_norm,
    *,
    differentiable: bool = False,
):
    """Render frame 0 to float in [0,255], exactly as ``render_video`` does.

    Mirrors ``cpr1/inflate.py:338-349``. Returns (B, 3, 874, 1164).
    """
    import torch
    from torch.nn import functional

    rounder = _round_ste if differentiable else torch.round
    # A non-CPU device is used only for the differentiable proposal surface.
    # Realized scoring continues to call this with CPU tensors and the frozen
    # CPU scorer.  Moving the fixed basis here keeps the canonical receiver
    # arithmetic identical while making the requested gradient device real.
    basis = basis_norm.to(device=coefficients.device, dtype=coefficients.dtype)
    carrier = torch.einsum("bk,kchw->bchw", coefficients, basis)
    carrier = carrier / math.sqrt(CARRIER_DIM)
    low = rounder((127.5 + CARRIER_AMPLITUDE * carrier).clamp(0.0, 255.0))
    slave = functional.interpolate(
        low, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False
    )
    return rounder(slave.clamp(0.0, 255.0))


def apply_selector_float(frames_bchw, modes, choices_for_batch):
    """Apply the frame-0 selector pixel op in float, matching integer semantics.

    ``runtime/frame0_selector.py:113-144`` operates on uint8 BxHxWx3. Here the
    same arithmetic runs on float BxCxHxW so it stays differentiable; because
    every input is already an exact integer the two agree bit-for-bit.
    """
    import torch

    if not len(modes):
        return frames_bchw
    out = frames_bchw
    pieces = []
    for index in range(frames_bchw.shape[0]):
        mode = modes[int(choices_for_batch[index])]
        frame = out[index : index + 1]
        kind = mode.kind
        if kind == 0:  # IDENTITY
            pieces.append(frame)
            continue
        if kind == 5:  # ROLL: shift=(b, a) over axes (H, W) of BxHxWx3
            pieces.append(torch.roll(frame, shifts=(mode.b, mode.a), dims=(2, 3)))
            continue
        if kind == 6:  # TILE
            height, width = frame.shape[2], frame.shape[3]
            yy, xx = torch.meshgrid(
                torch.arange(height, device=frame.device),
                torch.arange(width, device=frame.device),
                indexing="ij",
            )
            if mode.a == 0:
                signs = ((yy + xx) & 1) * 2 - 1
            elif mode.a == 1:
                signs = (yy & 1) * 2 - 1
            elif mode.a == 2:
                signs = (xx & 1) * 2 - 1
            elif mode.a == 3:
                signs = (((yy >> 2) + (xx >> 2)) & 1) * 2 - 1
            else:
                raise Up2Error("invalid tile selector mode")
            delta = (signs * mode.b).to(frame.dtype)[None, None]
        elif kind == 3:  # LUMA
            delta = torch.as_tensor(float(mode.a), dtype=frame.dtype, device=frame.device)
        elif kind == 4:  # CHANNEL
            delta = torch.as_tensor(
                [float(mode.a), float(mode.b), float(mode.c)],
                dtype=frame.dtype,
                device=frame.device,
            ).view(1, 3, 1, 1)
        else:
            raise Up2Error(f"unsupported selector mode kind {kind}")
        pieces.append((frame + delta).clamp(0.0, 255.0))
    return torch.cat(pieces, dim=0)


def render_frame0(
    coefficients,
    state: CarrierState,
    pair_indices,
    *,
    differentiable: bool = False,
):
    """Full shipped frame-0 path: carrier render then selector override."""
    frames = render_frame0_float(
        coefficients, state.basis_norm, differentiable=differentiable
    )
    return apply_selector_float(
        frames, state.selector_modes, state.selector_choices[pair_indices]
    )


# --------------------------------------------------------------------------
# The instrument -- frozen CPU PoseNet on the shipped frames.
# --------------------------------------------------------------------------


def load_posenet(device: str = "cpu"):
    """Load frozen PoseNet on ``device``; score callers use the CPU default."""
    from safetensors.torch import load_file

    sys.path.insert(0, str(UPSTREAM))
    try:
        from modules import PoseNet, posenet_sd_path  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    network = PoseNet().eval().to(device)
    network.load_state_dict(load_file(posenet_sd_path, device="cpu"))
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return network


_GRADIENT_DEVICE = "cpu"
_GRADIENT_POSENET = None


def configure_gradient_device(device: str) -> dict[str, str]:
    """Select the proposal-gradient device without moving score authority.

    Realized candidates are still rescored by the CPU ``posenet`` passed to
    :func:`jacobian_and_residual`'s callers.  The optional accelerator is used
    only to form the local Jacobian that proposes lattice moves.
    """

    global _GRADIENT_DEVICE, _GRADIENT_POSENET
    _GRADIENT_DEVICE = device
    _GRADIENT_POSENET = None if device == "cpu" else load_posenet(device)
    return {"gradient_device": device, "score_device": "cpu"}


def enable_posenet_gradients() -> dict[str, Any]:
    """Make PoseNet's preprocess gradient-reachable, then prove it unchanged.

    ``upstream/frame_utils.py:50`` decorates ``rgb_to_yuv6`` with
    ``@torch.no_grad()``, which severs every gradient into the pose head. The
    canonical repo helper swaps in a differentiable implementation; this wrapper
    additionally asserts forward equivalence to upstream, so the patch cannot
    quietly change the number being measured.
    """
    from tac.differentiable_eval_roundtrip import (
        assert_yuv6_forward_equivalence_to_upstream,
        patch_upstream_yuv6_globally,
    )

    sys.path.insert(0, str(UPSTREAM))
    try:
        token = patch_upstream_yuv6_globally()
        equivalence = assert_yuv6_forward_equivalence_to_upstream()
    finally:
        sys.path.pop(0)
    if not equivalence.get("passed"):
        raise Up2Error(f"differentiable yuv6 is not forward-equivalent: {equivalence}")
    return {
        "frame_utils_patched": token.frame_utils_was_patched,
        "modules_patched": token.modules_was_patched,
        "max_abs_error": equivalence.get("max_abs_error"),
        "samples": equivalence.get("num_samples"),
    }


def pose_from_frames(posenet, frame0_bchw, frame1_bchw):
    """(B,3,H,W) x2 -> (B,6) pose. Differentiable when the inputs are."""
    import torch

    pair = torch.stack([frame0_bchw, frame1_bchw], dim=1)  # (B, 2, 3, H, W)
    prepared = posenet.preprocess_input(pair)
    return posenet(prepared)["pose"][..., :POSE_DIMS]


def open_raw(path: Path, *, verify_sha: bool = True):
    """Memmap the shipped decode of the pointer body."""
    if not path.is_file():
        raise Up2Error(f"raw decode does not exist: {path}")
    if verify_sha:
        observed = _sha256_file(path)
        if observed not in {CPU_DECODE_RAW_SHA256, CUDA_DECODE_RAW_SHA256}:
            raise Up2Error(
                "raw sha256 matches neither known decode of the pointer body: "
                + observed
            )
    return np.memmap(
        path,
        dtype=np.uint8,
        mode="r",
        shape=(2 * N_PAIRS_TOTAL, CAMERA_H, CAMERA_W, 3),
    )


def frames_to_bchw(frames_bhwc):
    """(B,H,W,3) uint8 -> (B,3,H,W) float32."""
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(frames_bhwc)).float()
    return tensor.permute(0, 3, 1, 2).contiguous()


# --------------------------------------------------------------------------
# Control: does the forward model reproduce the shipped bytes exactly?
# --------------------------------------------------------------------------


def validate_forward_model(
    state: CarrierState, raw, pair_indices: Sequence[int]
) -> dict[str, Any]:
    """Re-render frame 0 from the coefficients and diff against the shipped raw.

    This is the control that makes every later number mean something: if the
    model is byte-exact on the shipped coefficients, then a solved coefficient
    set renders what the receiver would render.
    """
    import torch

    indices = np.asarray(list(pair_indices), dtype=np.int64)
    with torch.inference_mode():
        rendered = render_frame0(
            state.coefficients[indices], state, indices, differentiable=False
        )
    rendered_u8 = rendered.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    shipped = np.asarray(raw[2 * indices])
    exact = bool(np.array_equal(rendered_u8, shipped))
    diff = np.abs(rendered_u8.astype(np.int16) - shipped.astype(np.int16))
    return {
        "pairs": indices.tolist(),
        "byte_exact": exact,
        "max_abs_delta": int(diff.max()),
        "mean_abs_delta": float(diff.mean()),
        "mismatched_pixels": int((diff != 0).sum()),
        "total_pixels": int(diff.size),
    }


# --------------------------------------------------------------------------
# The solve.
# --------------------------------------------------------------------------


def jacobian_and_residual(
    posenet,
    state: CarrierState,
    coefficients_batch,
    frame1_batch,
    targets_batch,
    pair_indices,
):
    """Per-pair Jacobian d(pose)/d(coeff) and pose residual.

    Pairs are independent, so summing pose component ``j`` over the batch and
    taking one backward pass yields ``d pose[p, j] / d coeff[p]`` for every p at
    once -- 6 backward passes for the whole batch, not 6 per pair.
    """
    import torch

    gradient_device = _GRADIENT_DEVICE
    gradient_net = posenet if gradient_device == "cpu" else _GRADIENT_POSENET
    if gradient_net is None:
        raise Up2Error("gradient device was selected without a scorer instance")
    coefficients = (
        coefficients_batch.clone()
        .detach()
        .to(gradient_device)
        .requires_grad_(True)
    )
    frame0 = render_frame0(coefficients, state, pair_indices, differentiable=True)
    pose = pose_from_frames(
        gradient_net,
        frame0,
        frame1_batch.to(gradient_device),
    )
    rows = []
    for component in range(POSE_DIMS):
        grad = torch.autograd.grad(
            pose[:, component].sum(), coefficients, retain_graph=component < POSE_DIMS - 1
        )[0]
        rows.append(grad)
    jacobian = torch.stack(rows, dim=1).detach().cpu()  # (B, 6, 12)
    pose_cpu = pose.detach().cpu()
    residual = pose_cpu - targets_batch.cpu()  # (B, 6)
    return jacobian, residual, pose_cpu


def measure_pose(
    posenet,
    state: CarrierState,
    coefficients,
    raw,
    targets,
    pair_indices: np.ndarray,
    *,
    batch_size: int = 8,
    coefficients_are_full: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact (non-STE) d_pose per pair. Returns (per-pair d_pose, pose vectors)."""
    import torch

    per_pair = np.zeros(len(pair_indices), dtype=np.float64)
    poses = np.zeros((len(pair_indices), POSE_DIMS), dtype=np.float64)
    for start in range(0, len(pair_indices), batch_size):
        chunk = pair_indices[start : start + batch_size]
        coeff = coefficients[chunk] if coefficients_are_full else coefficients[
            start : start + batch_size
        ]
        frame1 = frames_to_bchw(raw[2 * chunk + 1])
        with torch.inference_mode():
            frame0 = render_frame0(coeff, state, chunk, differentiable=False)
            pose = pose_from_frames(posenet, frame0, frame1)
        pose_np = pose.to(torch.float64).numpy()
        poses[start : start + len(chunk)] = pose_np
        per_pair[start : start + len(chunk)] = (
            (pose_np - targets[chunk]) ** 2
        ).mean(axis=1)
    return per_pair, poses


def realize_codes(coefficients, coefficient_scales) -> np.ndarray:
    """Project float coefficients onto the shipped signed-int12 lattice."""
    import torch

    scales = coefficient_scales[None]
    codes = torch.round(coefficients / scales)
    codes = codes.clamp(COEFF_CODE_MIN, COEFF_CODE_MAX)
    return codes.to(torch.int32).numpy()


def codes_to_coefficients(codes: np.ndarray, coefficient_scales):
    import torch

    return torch.from_numpy(np.asarray(codes, dtype=np.int32)).float() * (
        coefficient_scales[None]
    )


@dataclass
class SolveConfig:
    pairs: int = N_PAIRS_TOTAL
    batch_size: int = 8
    max_iterations: int = 0  # 0 = uncapped, run to measured convergence
    rel_improvement_floor: float = 1e-4
    lm_lambda: float = 1e-3
    lm_lambda_min: float = 1e-9
    lm_lambda_max: float = 1e6
    seed: int = 1234
    out_dir: Path = field(default_factory=lambda: Path("."))
    gt_cache: Path = DEFAULT_DALI_GT
    axis: str = "contest_cuda"
    runtime_dir: Path = DEFAULT_RUNTIME
    raw_path: Path = DEFAULT_RAW
    verify_sha: bool = True


def _score_from(d_pose: float, d_seg: float, archive_bytes: int) -> float:
    return 100.0 * d_seg + math.sqrt(10.0 * d_pose) + 25.0 * archive_bytes / 37_545_489.0


def pose_leg(d_pose: float) -> float:
    return math.sqrt(10.0 * d_pose)


def pose_report_bound(d_pose: float) -> float:
    """Half-ULP bound of the 8dp d_pose report, in score units.

    d(pose leg)/d(d_pose) = 5/sqrt(10*d_pose), so the bound GROWS as d_pose falls.
    Below ``resolvable_d_pose_floor()`` the report prints 0.00000000 and cannot
    resolve the improvement at all.
    """
    if d_pose <= 0.0:
        return pose_leg(REPORT_HALF_ULP)
    return 5.0 / math.sqrt(10.0 * d_pose) * REPORT_HALF_ULP


def resolvable_d_pose_floor() -> float:
    """Below this d_pose the T4 8dp report cannot resolve an improvement."""
    return REPORT_HALF_ULP


def conditioning_report(jacobian, residual, coefficient_scales) -> dict[str, Any]:
    """Why the carrier can or cannot reach its own pose residual.

    Returns, per pair, the singular values of d(pose)/d(coeff), the share of the
    residual lying in each left-singular direction, and the coefficient step (in
    int12 code units) an exact solve would demand. When the residual concentrates
    on the smallest singular direction the demanded step leaves the lattice, and
    no amount of solving inside this representation can reach it.
    """
    import torch

    jac = jacobian.double()
    res = residual.double()
    scales = coefficient_scales.double()
    rows = []
    for index in range(jac.shape[0]):
        left, svals, _ = torch.linalg.svd(jac[index], full_matrices=False)
        norm = res[index].norm().clamp_min(1e-30)
        share = (left.T @ res[index]).abs() / norm
        demanded = (share * norm / svals.clamp_min(1e-30))
        rows.append(
            {
                "singular_values": svals.tolist(),
                "residual_share_per_direction": share.tolist(),
                "residual_norm": float(norm),
                "condition_number": float(svals[0] / svals[-1].clamp_min(1e-30)),
                "demanded_coefficient_step": demanded.tolist(),
                "demanded_code_units_max": float(
                    (demanded.max() / scales.min()).item()
                ),
            }
        )
    return {"pairs": rows}


def candidate_codes_for_pair(codes_row: np.ndarray, offsets: Sequence[int]) -> tuple:
    """Single-coordinate lattice neighbours of one pair's 12 codes."""
    candidates = []
    labels = []
    for coordinate in range(CARRIER_DIM):
        for offset in offsets:
            trial = codes_row.copy()
            value = int(trial[coordinate]) + int(offset)
            if not COEFF_CODE_MIN <= value <= COEFF_CODE_MAX:
                continue
            trial[coordinate] = value
            candidates.append(trial)
            labels.append((coordinate, int(offset)))
    return np.stack(candidates) if candidates else np.zeros((0, CARRIER_DIM), np.int32), labels


def solve_pair_realized(
    posenet,
    state: CarrierState,
    raw,
    target_row: np.ndarray,
    pair: int,
    codes_row: np.ndarray,
    *,
    offsets: Sequence[int] = (-2, -1, 1, 2),
    max_passes: int = 0,
    eval_batch: int = 64,
) -> dict[str, Any]:
    """Uncapped greedy descent on the REALIZED objective for one pair.

    No surrogate: every candidate is rendered through the exact receiver path
    (real ``round``, real selector) and scored by the frozen CPU PoseNet, so an
    accepted step is a step that actually happened. ``max_passes=0`` means run
    until a full sweep finds no improving lattice neighbour -- the convergence
    proof, not an iteration cap.
    """
    import torch

    index = np.array([pair], dtype=np.int64)
    frame1 = frames_to_bchw(raw[2 * index + 1])

    def evaluate(code_block: np.ndarray) -> np.ndarray:
        if len(code_block) == 0:
            return np.zeros(0, dtype=np.float64)
        results = np.zeros(len(code_block), dtype=np.float64)
        for start in range(0, len(code_block), eval_batch):
            chunk = code_block[start : start + eval_batch]
            coefficients = codes_to_coefficients(chunk, state.coefficient_scales)
            indices = np.full(len(chunk), pair, dtype=np.int64)
            frames1 = frame1.expand(len(chunk), -1, -1, -1).contiguous()
            with torch.inference_mode():
                frame0 = render_frame0(coefficients, state, indices, differentiable=False)
                pose = pose_from_frames(posenet, frame0, frames1)
            results[start : start + len(chunk)] = (
                (pose.to(torch.float64).numpy() - target_row[None]) ** 2
            ).mean(axis=1)
        return results

    current = codes_row.copy()
    best = float(evaluate(current[None])[0])
    start_value = best
    history = [best]
    passes = 0
    evaluations = 1
    while True:
        passes += 1
        block, labels = candidate_codes_for_pair(current, offsets)
        values = evaluate(block)
        evaluations += len(block)
        if len(values) == 0:
            break
        winner = int(values.argmin())
        if values[winner] >= best:
            break  # convergence proof: no improving lattice neighbour exists
        best = float(values[winner])
        current = block[winner].copy()
        history.append(best)
        if max_passes and passes >= max_passes:
            break
    return {
        "pair": int(pair),
        "start_d_pose": start_value,
        "final_d_pose": best,
        "ratio": best / start_value if start_value > 0 else 1.0,
        "passes": passes,
        "evaluations": evaluations,
        "converged": not (max_passes and passes >= max_passes),
        "codes": current.astype(np.int32).tolist(),
        "changed_coordinates": int((current != codes_row).sum()),
        "history": history,
    }


# --------------------------------------------------------------------------
# Resumable driver. P0: every launch resumes from disk.
# --------------------------------------------------------------------------


def select_pairs(pairs: int, seed: int) -> np.ndarray:
    """Full field when ``pairs >= 600``, else a SEEDED RANDOM sample -- never a prefix.

    A contiguous prefix of this video is a different population, and the bias is
    worst on exactly this axis: pose prefixes measure 2.54-4.21x HARDER than the
    population because the first 120 pairs are the two hardest 60-pair blocks
    (``ddm_na2`` / ``ddm_bp2``). A prefix-based pose verdict is the canonical
    false-negative shape, so sub-n600 runs sample instead.
    """
    if pairs >= N_PAIRS_TOTAL:
        return np.arange(N_PAIRS_TOTAL, dtype=np.int64)
    if pairs < 1:
        raise Up2Error(f"pairs must be >= 1, got {pairs}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(N_PAIRS_TOTAL, size=pairs, replace=False)).astype(np.int64)


def _load_done(rows_path: Path) -> dict[int, dict[str, Any]]:
    done: dict[int, dict[str, Any]] = {}
    if rows_path.is_file():
        with rows_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[int(row["pair"])] = row
    return done


# The receiver's own limits on the sparse overlay section
# (runtime/compensation_overlay.py:52,58 -- count packs into 4 header bits and
# deltas into a 3-bit field). A 600-pair solve CANNOT ship through this section;
# only a 15-pair subset can, and that makes its byte cost exactly computable.
OVERLAY_MAX_PAIRS = 15
OVERLAY_MIN_DELTA = -3
OVERLAY_MAX_DELTA = 4


def overlay_payload_bytes(support_per_pair: Sequence[int]) -> int:
    """Exact encoded length of a compensation overlay, from the receiver's format.

    ``runtime/compensation_overlay.py:70-80``: 10 bits per pair index, 12 bits of
    support mask per pair, 3 bits per nonzero coordinate, then byte-aligned, with
    a 4-byte magic plus a packed version/count byte.
    """
    pairs = len(support_per_pair)
    if not 1 <= pairs <= OVERLAY_MAX_PAIRS:
        raise Up2Error(
            f"overlay carries {pairs} pairs; the receiver accepts 1..{OVERLAY_MAX_PAIRS}"
        )
    bits = pairs * 10 + pairs * CARRIER_DIM + int(sum(support_per_pair)) * 3
    return 4 + 1 + (bits + 7) // 8


def select_overlay_candidate(
    rows: Sequence[dict[str, Any]],
    base_codes: np.ndarray,
    *,
    max_pairs: int = OVERLAY_MAX_PAIRS,
    realized_gains: dict[int, float] | None = None,
) -> dict[str, Any]:
    """Pick the shippable 15-pair subset of a full solve, clipped to the section.

    d_pose is a mean over 600 pairs, so a pair's contribution to the score is its
    ABSOLUTE d_pose reduction. Rank by that, clip each coordinate delta into the
    receiver's ``[-3, 4]`` domain, and drop any pair left with empty support.

    Without ``realized_gains`` the ranking uses each pair's UNCLIPPED gain, which
    only pre-filters: clipping can destroy most of it. Pass ``realized_gains``
    (pair -> measured d_pose reduction of the CLIPPED codes) to rank on what
    actually ships. Either way the caller must re-measure the returned codes.
    """
    scored = []
    for row in rows:
        pair = int(row["pair"])
        delta = np.asarray(row["codes"], dtype=np.int32) - base_codes[pair]
        clipped = np.clip(delta, OVERLAY_MIN_DELTA, OVERLAY_MAX_DELTA)
        if not np.any(clipped):
            continue
        if realized_gains is not None:
            gain = float(realized_gains.get(pair, 0.0))
        else:
            gain = float(row["start_d_pose"]) - float(row["final_d_pose"])
        if gain <= 0:
            continue
        scored.append(
            {
                "pair": pair,
                "unclipped_gain": gain,
                "delta": clipped,
                "support": int(np.count_nonzero(clipped)),
                "clipped": bool(np.any(clipped != delta)),
            }
        )
    scored.sort(key=lambda item: item["unclipped_gain"], reverse=True)
    chosen = scored[:max_pairs]
    chosen.sort(key=lambda item: item["pair"])
    if not chosen:
        return {
            "pairs": [],
            "deltas": [],
            "support_per_pair": [],
            "any_clipped": False,
            "unclipped_gain_sum": 0.0,
            "payload_bytes": 0,
            "codes": base_codes.copy(),
        }
    codes = base_codes.copy()
    for item in chosen:
        codes[item["pair"]] = base_codes[item["pair"]] + item["delta"]
    return {
        "pairs": [item["pair"] for item in chosen],
        "deltas": [item["delta"].tolist() for item in chosen],
        "support_per_pair": [item["support"] for item in chosen],
        "any_clipped": bool(any(item["clipped"] for item in chosen)),
        "unclipped_gain_sum": float(sum(item["unclipped_gain"] for item in chosen)),
        "payload_bytes": overlay_payload_bytes([item["support"] for item in chosen]),
        "codes": codes,
    }


def run_solve(config: SolveConfig) -> dict[str, Any]:
    """Solve every pair against the shipping-lineage GT, resumably."""
    import torch

    torch.set_num_threads(16)
    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"

    patch = enable_posenet_gradients()
    state = load_carrier_state(config.runtime_dir, verify_archive=config.verify_sha)
    raw = open_raw(config.raw_path, verify_sha=config.verify_sha)
    targets, lineage = load_gt_poses(config.gt_cache)
    gate = verify_gt_lineage(axis=config.axis, declared_lineage=lineage)
    posenet = load_posenet()

    pairs = select_pairs(config.pairs, config.seed)
    done = _load_done(rows_path)
    started = time.time()
    with rows_path.open("a", encoding="utf-8") as stream:
        for position, pair in enumerate(pairs):
            if int(pair) in done:
                continue
            row = solve_pair_realized(
                posenet,
                state,
                raw,
                targets[int(pair)],
                int(pair),
                state.codes[int(pair)],
                max_passes=config.max_iterations,
            )
            done[int(pair)] = row
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if position % 10 == 0 or position == len(pairs) - 1:
                elapsed = time.time() - started
                remaining = len(pairs) - len(done)
                print(
                    f"[{len(done)}/{len(pairs)}] pair={int(pair)} "
                    f"ratio={row['ratio']:.4f} elapsed={elapsed / 60:.1f}m "
                    f"eta={elapsed / max(1, position + 1) * remaining / 60:.1f}m",
                    flush=True,
                )

    ordered = [done[int(p)] for p in pairs if int(p) in done]
    start_mean = float(np.mean([r["start_d_pose"] for r in ordered]))
    final_mean = float(np.mean([r["final_d_pose"] for r in ordered]))
    summary = {
        "schema": "ddm_up2_solve.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet]",
        "score_claim": False,
        "promotable": False,
        "gt_lineage_gate": gate,
        "gt_cache": str(config.gt_cache),
        "yuv6_patch": patch,
        "runtime_dir": str(config.runtime_dir),
        "raw_path": str(config.raw_path),
        "pairs": len(ordered),
        "start_d_pose_mean": start_mean,
        "final_d_pose_mean": final_mean,
        "d_pose_ratio": final_mean / start_mean if start_mean else 1.0,
        "pose_leg_start": pose_leg(start_mean),
        "pose_leg_final": pose_leg(final_mean),
        "delta_score_pose_only": pose_leg(final_mean) - pose_leg(start_mean),
        "pairs_improved": int(sum(1 for r in ordered if r["final_d_pose"] < r["start_d_pose"])),
        "total_changed_coordinates": int(sum(r["changed_coordinates"] for r in ordered)),
        "total_evaluations": int(sum(r["evaluations"] for r in ordered)),
        "all_converged": bool(all(r["converged"] for r in ordered)),
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def price_full_resolve_bytes(
    runtime_dir: Path, candidate_codes: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    """Measured archive-byte delta of a re-solved coefficient lattice.

    Reuses ``ddm_t1h``'s pricer, which models the counted CAP1 container exactly:
    everything but the Rice residual payload is independent of the coefficient
    codes, so ``carrier_bytes = fixed_prefix + ceil(rice_bits / 8)``. The pricer
    is required to reproduce the SHIPPED bit count from the shipped codes before
    any candidate number is returned -- otherwise the delta is unanchored.
    """
    sys.path.insert(0, str(REPO / "experiments"))
    try:
        import ddm_t1h_carrier_byte_pricer as pricer  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    (carrier_repack, _cap1, predictor, carrier_blob, _selector, info, model,
     base_codes, _comp) = pricer.load_shipped(runtime_dir / "archive.zip", runtime_dir)
    _, shipped_bits = pricer.rice_bits(base_codes, model, carrier_repack, predictor)
    if shipped_bits != int(info["rice_payload_bits"]):
        raise Up2Error(
            "byte pricer does not reproduce the shipped Rice payload "
            f"({shipped_bits} vs {info['rice_payload_bits']}); delta would be unanchored"
        )
    _, candidate_bits = pricer.rice_bits(
        np.asarray(candidate_codes, dtype=np.int32), model, carrier_repack, predictor
    )
    shipped_bytes = (shipped_bits + 7) // 8
    candidate_bytes = (candidate_bits + 7) // 8
    report = {
        "control_reproduces_shipped_payload": True,
        "rice_bits_shipped": int(shipped_bits),
        "rice_bits_candidate": int(candidate_bits),
        "rice_payload_bytes_shipped": int(shipped_bytes),
        "rice_payload_bytes_candidate": int(candidate_bytes),
        "delta_bytes": int(candidate_bytes - shipped_bytes),
        "changed_coordinates": int((np.asarray(candidate_codes) != base_codes).sum()),
    }
    # Returned separately, never inside the report: the report is JSON-dumped and
    # a numpy array in it would crash the dump at the worst possible moment.
    return report, base_codes.astype(np.int32)


def price_candidate(
    *,
    d_pose_start: float,
    d_pose_final: float,
    delta_bytes: int,
    d_seg_delta: float = 0.0,
) -> dict[str, Any]:
    """Net score delta with the report-resolution bound stated honestly.

    The 8dp d_pose report has a half-ULP of 5e-9, and its score consequence GROWS
    as d_pose falls. Bounds ADD across the two rows being differenced, so the
    quoted bound is the SUM, and the improvement is quoted as a multiple of it.
    """
    leg_start = pose_leg(d_pose_start)
    leg_final = pose_leg(d_pose_final)
    delta_pose = leg_final - leg_start
    delta_rate = delta_bytes * BYTE_TO_SCORE
    delta_seg = 100.0 * d_seg_delta
    net = delta_pose + delta_rate + delta_seg
    bound = pose_report_bound(d_pose_start) + pose_report_bound(d_pose_final)
    return {
        "d_pose_start": d_pose_start,
        "d_pose_final": d_pose_final,
        "pose_leg_start": leg_start,
        "pose_leg_final": leg_final,
        "delta_score_pose": delta_pose,
        "delta_bytes": int(delta_bytes),
        "delta_score_rate": delta_rate,
        "delta_score_seg": delta_seg,
        "net_delta_score": net,
        "summed_report_bound": bound,
        "net_over_bound": abs(net) / bound if bound else float("inf"),
        "resolvable_by_the_t4_report": abs(net) > bound,
        "d_pose_below_report_resolution": d_pose_final < resolvable_d_pose_floor(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="byte-exactness control on the forward model")
    validate.add_argument("--pairs", type=int, default=8)
    validate.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    validate.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    validate.add_argument("--out", type=Path, default=None)
    validate.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")

    solve = sub.add_parser("solve", help="uncapped realized pose solve on the shipping GT")
    solve.add_argument("--pairs", type=int, default=N_PAIRS_TOTAL)
    solve.add_argument("--max-iterations", type=int, default=0)
    solve.add_argument("--gt-cache", type=Path, default=DEFAULT_DALI_GT)
    solve.add_argument("--axis", type=str, default="contest_cuda")
    solve.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    solve.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    solve.add_argument("--out", type=Path, required=True)
    solve.add_argument("--seed", type=int, default=1234, help="sampling seed when pairs < 600")
    solve.add_argument("--no-verify-sha", action="store_true")
    solve.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from tac.semantic_pipeline.contracts import require_device

    args.device_binding = require_device(args.device).as_dict()
    args.device_binding.update(configure_gradient_device(args.device))
    if args.command == "validate":
        state = load_carrier_state(args.runtime_dir)
        raw = open_raw(args.raw, verify_sha=False)
        special = np.flatnonzero(state.selector_choices != 0).tolist()
        pairs = sorted(set(special + list(range(args.pairs))))
        report = validate_forward_model(state, raw, pairs)
        print(json.dumps(report, indent=2))
        if args.out:
            args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 0 if report["byte_exact"] else 1
    if args.command == "solve":
        config = SolveConfig(
            pairs=args.pairs,
            max_iterations=args.max_iterations,
            out_dir=args.out,
            gt_cache=args.gt_cache,
            axis=args.axis,
            runtime_dir=args.runtime_dir,
            raw_path=args.raw,
            seed=args.seed,
            verify_sha=not args.no_verify_sha,
        )
        summary = run_solve(config)
        print(json.dumps(summary, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
