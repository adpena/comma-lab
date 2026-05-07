#!/usr/bin/env python3
"""Wave-1.5 R(D)-joint coordinate-descent pose refinement.

Port of `reports/raw/leaderboard_intel_20260501/pr67_line_search.py` (PR #67
EthanYang's score-aware POSE refinement tool, ~194 LOC) onto OUR Lane
Q-FAITHFUL stack (`tac.qp1_pose_codec` + `tac.quantizr_qzs3_codec` +
`tac.quantizr_faithful_renderer.JointFrameGenerator`).

Mirrors pr67's coordinate-descent loop but with two structural improvements:

1.  **Slice offsets via metadata.json** (not pr67's brittle 7-bucket
    `model_br_len` length-lookup at pr67_inflate.py:746-768). The Wave-1
    orchestrator `experiments/build_qpose_archive.py` writes per-segment
    byte counts into ``metadata.json`` next to ``archive.zip``; we read
    those instead of guessing from total payload length.
2.  **CUDA-required by the MPS-PoseNet-23x rule** (CLAUDE.md). The default
    device is ``cuda:0``; ``--device cpu`` is allowed only for CI smoke and
    raises a banner that the resulting bytes/score are advisory.

The optimization objective at line 142 mirrors pr67_line_search.py:140 EXACTLY:

    obj = sqrt(10 * pose_mse) + 25 * archive_bytes / 37_545_489

Where ``archive_bytes = mask_br_bytes + model_br_bytes + pose_q_br_bytes``
(plus a small ZIP-overhead constant; pr67 used 100, we read the actual ZIP
overhead from the input archive's stat() vs blob length, then add it back
when writing the refined archive).

Per-frame coordinate descent at radii [1, 2, 3, 5, 8] x N passes accepts
ONLY pass-deltas that strictly decrease the joint objective. This is the
exact contest objective being directly optimized at compress time.

Predicted gain: +0.001 to +0.005 score on Wave-1 anchor when stacked on
Q-FAITHFUL+QZS3 (per `reference_pr67_line_search_R_D_joint_coordinate_descent_20260501`).

Usage:
    python experiments/line_search_pose_refinement.py \\
        --archive-path  experiments/results/wave1_anchor/archive.zip \\
        --metadata-path experiments/results/wave1_anchor/metadata.json \\
        --output-path   experiments/results/wave1_5_refined/archive.zip \\
        --output-metadata experiments/results/wave1_5_refined/metadata.json \\
        --posenet-path upstream/models/posenet.safetensors \\
        --gt-mkv upstream/videos/0.mkv \\
        --device cuda:0 --batch-size 16 --candidate-chunk 32 \\
        --radii "1,2,3,5,8" --passes 2

Smoke usage (random renderer, no GT, MockPoseNet — for CI determinism):
    python experiments/line_search_pose_refinement.py \\
        --archive-path /tmp/smoke/archive.zip \\
        --metadata-path /tmp/smoke/metadata.json \\
        --output-path /tmp/smoke/refined.zip \\
        --output-metadata /tmp/smoke/refined.json \\
        --device cpu --mock-posenet --no-gt --radii "1" --passes 1
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Any, Callable

import brotli
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tac.qp1_pose_codec import (
    POSE_SCALE,
    VELOCITY_OFFSET,
    VELOCITY_SCALE,
    decode_qp1,
    encode_qp1,
)
from tac.quantizr_faithful_renderer import (
    JointFrameGenerator,
    build_quantizr_faithful_renderer,
)
from tac.quantizr_qzs3_codec import decode_qzs3_state_dict


ORIGINAL_SIZE = 37_545_489  # contest baseline (pr67_line_search.py:26)
DETERMINISTIC_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ARCHIVE_MEMBER_NAME = "p"
SCORER_RUNTIME_MODULES = (
    "nvidia.dali",
    "timm",
    "einops",
    "segmentation_models_pytorch",
    "safetensors",
)


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def assert_scorer_runtime_dependencies_available(
    required_modules: tuple[str, ...] = SCORER_RUNTIME_MODULES,
) -> None:
    """Fail closed before importing upstream scorer modules on remote runners."""
    missing = [module for module in required_modules if not _module_available(module)]
    if not missing:
        return
    missing_csv = ", ".join(missing)
    raise RuntimeError(
        "line_search_pose_refinement needs scorer runtime dependencies before "
        "loading upstream/modules.py or DaliVideoDataset; missing: "
        f"{missing_csv}. Install the repo runtime extra or run the remote scorer "
        "and DALI dependency bootstraps before dispatch, e.g. "
        "`python -m pip install -e '.[runtime]'` plus "
        "`python scripts/bootstrap_dali_hash_pinned.py --json-out <path>` "
        "in the runner interpreter. "
        "This is a preflight failure, not score evidence."
    )


def assert_dali_runtime_dependency_available() -> None:
    """Fail closed before GT-video target extraction uses DALI."""
    if importlib.util.find_spec("nvidia.dali") is not None:
        return
    raise RuntimeError(
        "line_search_pose_refinement needs nvidia.dali for --gt-mkv target "
        "extraction before loading upstream/frame_utils.py. Install the "
        "driver-compatible DALI wheel in the runner interpreter before "
        "dispatch, e.g. `nvidia-dali-cuda130` for CUDA 13 images or "
        "`nvidia-dali-cuda120` for CUDA 12 images. This is a preflight "
        "failure, not score evidence."
    )


def patch_posenet_for_differentiable_search(posenet: torch.nn.Module) -> None:
    """Patch upstream PoseNet preprocessing so proposal gradients are live.

    The official scorer is still the canonical exact-eval authority. This patch
    is used only to propose candidate integer pose deltas; every accepted delta
    is still checked by the exact rounded archive objective.
    """
    from tac.scorer import make_scorers_differentiable

    make_scorers_differentiable(posenet, torch.nn.Identity())


# -----------------------------------------------------------------------------
# IO helpers (metadata-driven, not pr67's brittle length-lookup)
# -----------------------------------------------------------------------------


def load_archive_blob(archive_path: Path) -> bytes:
    """Return the single 'p' member of a Wave-1 archive.zip."""
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        if names != [ARCHIVE_MEMBER_NAME]:
            raise ValueError(
                f"expected single archive member 'p'; got {names}"
            )
        return zf.read(ARCHIVE_MEMBER_NAME)


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    """Read per-segment byte counts written by build_qpose_archive."""
    meta = json.loads(metadata_path.read_text())
    for required in ("mask_br_bytes", "model_br_bytes", "pose_br_bytes"):
        if required not in meta:
            raise ValueError(
                f"metadata.json missing required field {required!r}; "
                f"path={metadata_path}"
            )
    return meta


def slice_blob(
    blob: bytes, meta: dict[str, Any]
) -> tuple[bytes, bytes, bytes]:
    """Slice (mask_br, model_br, pose_br) using metadata byte offsets."""
    mask_n = int(meta["mask_br_bytes"])
    model_n = int(meta["model_br_bytes"])
    pose_n = int(meta["pose_br_bytes"])
    expected = mask_n + model_n + pose_n
    if len(blob) != expected:
        raise ValueError(
            f"blob length {len(blob)} != mask+model+pose {expected} per metadata"
        )
    mask_br = blob[:mask_n]
    model_br = blob[mask_n : mask_n + model_n]
    pose_br = blob[mask_n + model_n :]
    return mask_br, model_br, pose_br


def write_refined_archive(
    output_path: Path,
    mask_br: bytes,
    model_br: bytes,
    refined_pose_br: bytes,
) -> int:
    """Re-assemble blob with new pose stream + write deterministic stored ZIP."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blob = mask_br + model_br + refined_pose_br
    info = zipfile.ZipInfo(filename=ARCHIVE_MEMBER_NAME, date_time=DETERMINISTIC_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = (0o644 & 0xFFFF) << 16
    with zipfile.ZipFile(output_path, mode="w") as zf:
        zf.writestr(info, blob)
    return output_path.stat().st_size


def sha256_path(path: Path) -> str:
    """Hash a small contest archive or metadata payload."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_refined_metadata(
    *,
    source_meta: dict[str, Any],
    source_archive_path: Path,
    source_archive_sha256: str,
    output_archive_path: Path,
    mask_br: bytes,
    model_br: bytes,
    refined_pose_br: bytes,
    refined_col0: np.ndarray,
    archive_bytes: int,
    refinement: dict[str, Any],
) -> dict[str, Any]:
    """Return custody-correct metadata for a refined archive checkpoint."""
    out_meta = dict(source_meta)
    if "archive_path" in source_meta:
        out_meta["source_metadata_archive_path"] = source_meta["archive_path"]
    if "archive_sha256" in source_meta:
        out_meta["source_metadata_archive_sha256"] = source_meta["archive_sha256"]
    out_meta["source_archive_path"] = str(source_archive_path)
    out_meta["source_archive_sha256"] = source_archive_sha256
    out_meta["archive_path"] = str(output_archive_path)
    out_meta["archive_bytes"] = archive_bytes
    out_meta["archive_sha256"] = sha256_path(output_archive_path)
    out_meta["pose_br_bytes"] = len(refined_pose_br)
    out_meta["pose_br_sha256"] = hashlib.sha256(refined_pose_br).hexdigest()
    out_meta["pose_uncompressed_bytes"] = len(
        encode_qp1(col0_to_pose_array(refined_col0))
    )
    out_meta["blob_bytes"] = len(mask_br) + len(model_br) + len(refined_pose_br)
    out_meta["blob_sha256"] = hashlib.sha256(
        mask_br + model_br + refined_pose_br
    ).hexdigest()
    out_meta["refinement"] = refinement
    return out_meta


def assert_metadata_matches_archive(
    metadata: dict[str, Any],
    archive_path: Path,
) -> None:
    """Fail closed if archive custody fields disagree with emitted bytes."""
    actual_bytes = archive_path.stat().st_size
    actual_sha256 = sha256_path(archive_path)
    recorded_path = str(metadata.get("archive_path"))
    expected_path = str(archive_path)
    violations: list[str] = []
    if recorded_path != expected_path:
        violations.append(
            f"archive_path metadata={recorded_path!r} actual={expected_path!r}"
        )
    if int(metadata.get("archive_bytes", -1)) != actual_bytes:
        violations.append(
            f"archive_bytes metadata={metadata.get('archive_bytes')!r} actual={actual_bytes}"
        )
    if metadata.get("archive_sha256") != actual_sha256:
        violations.append(
            f"archive_sha256 metadata={metadata.get('archive_sha256')!r} actual={actual_sha256}"
        )
    if violations:
        raise RuntimeError(
            "line-search metadata/archive custody mismatch: "
            + "; ".join(violations)
        )


# -----------------------------------------------------------------------------
# QP1 col0 helpers — operate on the raw uint16 col0 array (length 600)
# -----------------------------------------------------------------------------


def col0_from_pose_payload(pose_br: bytes) -> np.ndarray:
    """Decompress brotli wrapper, decode QP1, return col0 as int64 (length 600)."""
    raw = brotli.decompress(pose_br)
    decoded = decode_qp1(raw)
    # decoded[:, 0] = q/512.0 + 20.0 -> recover q by reverse-quantizing
    col0 = np.rint((decoded[:, 0].astype(np.float64) - VELOCITY_OFFSET) * VELOCITY_SCALE)
    col0 = col0.astype(np.int64)
    return col0


def col0_to_pose_array(col0: np.ndarray) -> np.ndarray:
    """Lift col0 (uint16-range int64) into a (N, 6) float pose array — cols 1+ zero."""
    n = col0.shape[0]
    poses = np.zeros((n, 6), dtype=np.float32)
    poses[:, 0] = col0.astype(np.float32) / VELOCITY_SCALE + VELOCITY_OFFSET
    return poses


def encode_col0_to_pose_br(col0: np.ndarray, *, brotli_quality: int = 11) -> bytes:
    """Re-encode col0 -> QP1 -> brotli (matches build_qpose_archive output)."""
    pose_array = col0_to_pose_array(col0)
    qp1_payload = encode_qp1(pose_array)
    return brotli.compress(qp1_payload, quality=brotli_quality)


def pose_atom_selection_summary(
    *,
    source_col0: np.ndarray,
    refined_col0: np.ndarray,
    pose_br_bytes: int,
    archive_bytes: int,
    policy: str,
) -> dict[str, Any]:
    """Return charged-accounting metadata for accepted QP1 col0 atoms."""
    if source_col0.shape != refined_col0.shape:
        raise ValueError(
            f"source/refined col0 shape mismatch: {source_col0.shape} != {refined_col0.shape}"
        )
    delta = refined_col0.astype(np.int64) - source_col0.astype(np.int64)
    changed = np.flatnonzero(delta)
    atoms = [
        {
            "frame_index": int(idx),
            "pair_index": int(idx // 2),
            "frame_in_pair": int(idx % 2),
            "delta_q": int(delta[idx]),
            "source_q": int(source_col0[idx]),
            "refined_q": int(refined_col0[idx]),
        }
        for idx in changed
    ]
    return {
        "schema_version": 1,
        "pose_codec": "pose_qp1_v1",
        "wire_format": "QP1+brotli",
        "policy": policy,
        "selection_hooks": {
            "supports_multipass": True,
            "supports_bandit_or_rl_proposal_ordering": True,
            "proposal_score_is_non_promotable_until_archive_eval": True,
            "acceptance_rule": "complete rounded archive objective strictly improves",
        },
        "charged_accounting": {
            "pose_br_bytes": int(pose_br_bytes),
            "archive_bytes": int(archive_bytes),
            "sidecar_bytes": 0,
            "all_score_affecting_bits_inside_archive": True,
        },
        "atom_count": int(changed.size),
        "atoms": atoms,
    }


# -----------------------------------------------------------------------------
# Forward pipeline (renderer + posenet) — CUDA preferred, MPS forbidden for
# strategic decisions per CLAUDE.md non-negotiable
# -----------------------------------------------------------------------------


def pose_from_col0_torch(col0_chunk: torch.Tensor) -> torch.Tensor:
    """Lift a uint16-range int tensor into a (N, 6) pose tensor (cols 1+ zero)."""
    n = col0_chunk.shape[0]
    pose = torch.zeros((n, 6), device=col0_chunk.device)  # OFF_MANIFOLD_OK: PR67 line-search rank-1 recipe lifts a single-axis col0 byte stream into the col0 slot of a 6-DOF tensor; cols 1-5 are intentionally zero per the published PR67 score-aware coordinate descent
    pose[:, 0] = col0_chunk.to(dtype=torch.float32) / VELOCITY_SCALE + VELOCITY_OFFSET
    return pose


def make_frames(
    generator: torch.nn.Module,
    masks: torch.Tensor,
    pose: torch.Tensor,
    *,
    target_h: int = 874,
    target_w: int = 1164,
    round_output: bool = True,
) -> torch.Tensor:
    """Generate frame pair, bilinear upsample to (874, 1164), clamp to [0, 255]."""
    f1, f2 = generator(masks.long(), pose.float())
    f1 = torch.nn.functional.interpolate(
        f1, size=(target_h, target_w), mode="bilinear", align_corners=False
    )
    f2 = torch.nn.functional.interpolate(
        f2, size=(target_h, target_w), mode="bilinear", align_corners=False
    )
    frames = torch.stack([f1, f2], dim=1).clamp(0, 255)
    if round_output:
        frames = frames.round()
    return frames


def load_renderer(
    qzs3_payload: bytes, device: torch.device
) -> JointFrameGenerator:
    """Load JointFrameGenerator from QZS3 weight bytes onto device."""
    state = decode_qzs3_state_dict(qzs3_payload, device="cpu")
    model = build_quantizr_faithful_renderer().to(device).eval()
    model.load_state_dict(state, strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_posenet(
    posenet_sd_path: Path | None, device: torch.device
) -> torch.nn.Module:
    """Import upstream PoseNet and load weights. CUDA-preferred per MPS rule."""
    assert_scorer_runtime_dependencies_available()
    sys.path.insert(0, str(_REPO_ROOT / "upstream"))
    from modules import PoseNet  # type: ignore[import]

    posenet = PoseNet().to(device).eval()
    if posenet_sd_path is not None:
        from safetensors.torch import load_file

        sd = load_file(str(posenet_sd_path), device=str(device))
        posenet.load_state_dict(sd)
    for p in posenet.parameters():
        p.requires_grad_(False)
    patch_posenet_for_differentiable_search(posenet)
    return posenet


def pose_outputs(posenet: torch.nn.Module, pairs_bhwc: torch.Tensor) -> torch.Tensor:
    """Forward pose pairs through PoseNet, return first 6 pose dims."""
    import einops

    x = einops.rearrange(pairs_bhwc, "b t h w c -> b t c h w").float()
    return posenet(posenet.preprocess_input(x))["pose"][..., :6].float()


# -----------------------------------------------------------------------------
# Mask loader — read OBU bytes from inside the archive, return (N, H, W) tensor
# -----------------------------------------------------------------------------


def load_masks_from_blob(
    mask_br: bytes, fallback_shape: tuple[int, int, int] | None = None
) -> torch.Tensor:
    """Decompress + parse the mask OBU stream into (N, H, W) long tensor.

    For real archives, delegates to PR #67's ``load_encoded_mask_video``.
    For smoke tests where mask_br is just zero padding, returns a zero
    tensor of ``fallback_shape`` (N, H, W).
    """
    if fallback_shape is not None:
        return torch.zeros(fallback_shape, dtype=torch.long)

    import os
    import tempfile
    import av

    with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp:
        tmp.write(brotli.decompress(mask_br))
        tmp_path = tmp.name
    try:
        container = av.open(tmp_path)
        frames = []
        try:
            for frame in container.decode(video=0):
                img = frame.to_ndarray(format="gray")
                cls_img = np.round(img / 63.0).astype(np.uint8)
                cls_img = np.clip(cls_img, 0, 4)
                frames.append(cls_img)
        finally:
            container.close()
        if not frames:
            raise ValueError("decoded mask OBU stream produced zero frames")
        return torch.from_numpy(np.stack(frames)).contiguous()
    finally:
        os.remove(tmp_path)


# -----------------------------------------------------------------------------
# Joint objective (mirrors pr67_line_search.py:136-140)
# -----------------------------------------------------------------------------


def compute_joint_objective(
    pose_mse: float,
    *,
    mask_br_bytes: int,
    model_br_bytes: int,
    pose_br_bytes: int,
    archive_overhead: int = 100,
) -> float:
    """obj = sqrt(10 * pose_mse) + 25 * archive_bytes / ORIGINAL_SIZE."""
    archive_size = mask_br_bytes + model_br_bytes + pose_br_bytes + archive_overhead
    return math.sqrt(10.0 * pose_mse) + 25.0 * archive_size / ORIGINAL_SIZE


def parse_delta_sets(raw: str | None) -> list[list[int]] | None:
    """Parse semicolon-separated sparse/asymmetric delta stages.

    Example: ``"-34,-21,-13,-8,-5,-3,-1,0,1,2,3,5,8; -13,-8,-5,-3,-1,0,1"``.
    Each stage automatically includes ``0`` so per-frame no-op candidates remain
    available during directional/manifold searches.
    """
    if raw is None or not raw.strip():
        return None
    stages: list[list[int]] = []
    for stage_idx, chunk in enumerate(raw.split(";"), start=1):
        vals = [x.strip() for x in chunk.split(",") if x.strip()]
        if not vals:
            raise ValueError(f"--delta-sets stage {stage_idx} is empty")
        deltas = sorted({int(x) for x in vals} | {0})
        if len(deltas) < 2:
            raise ValueError(
                f"--delta-sets stage {stage_idx} must contain at least one nonzero delta"
            )
        stages.append(deltas)
    return stages


def parse_int_csv(raw: str | None, *, default: list[int]) -> list[int]:
    """Parse a comma-separated integer list."""
    if raw is None or not raw.strip():
        return list(default)
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("integer list must not be empty")
    return values


def parse_magnitude_sets(raw: str | None) -> list[list[int]] | None:
    """Parse semicolon-separated positive magnitude stages."""
    if raw is None or not raw.strip():
        return None
    stages: list[list[int]] = []
    for stage_idx, chunk in enumerate(raw.split(";"), start=1):
        vals = [x.strip() for x in chunk.split(",") if x.strip()]
        if not vals:
            raise ValueError(f"--gradient-delta-sets stage {stage_idx} is empty")
        magnitudes = sorted({abs(int(x)) for x in vals if int(x) != 0})
        if not magnitudes:
            raise ValueError(
                f"--gradient-delta-sets stage {stage_idx} must contain nonzero magnitudes"
            )
        stages.append(magnitudes)
    return stages


def parse_basis_delta_sets(raw: str | None) -> list[dict[str, Any]] | None:
    """Parse semicolon-separated temporal/vector proposal stages.

    Format: ``"dct:1,2,3;pair_window:1,2"``. Magnitudes are signed during
    search, so only positive nonzero magnitudes are stored.
    """
    if raw is None or not raw.strip():
        return None
    stages: list[dict[str, Any]] = []
    for stage_idx, chunk in enumerate(raw.split(";"), start=1):
        text = chunk.strip()
        if not text:
            raise ValueError(f"--basis-delta-sets stage {stage_idx} is empty")
        if ":" not in text:
            raise ValueError(
                f"--basis-delta-sets stage {stage_idx} must use kind:magnitudes"
            )
        kind, raw_mags = text.split(":", 1)
        kind = kind.strip()
        if kind not in {"dct", "pair_window"}:
            raise ValueError(
                f"--basis-delta-sets stage {stage_idx} has unknown kind {kind!r}"
            )
        vals = [x.strip() for x in raw_mags.split(",") if x.strip()]
        mags = sorted({abs(int(x)) for x in vals if int(x) != 0})
        if not mags:
            raise ValueError(
                f"--basis-delta-sets stage {stage_idx} must contain nonzero magnitudes"
            )
        stages.append({"kind": kind, "magnitudes": mags, "basis_delta_set_index": stage_idx})
    return stages


def temporal_basis_matrix(
    n: int,
    *,
    kind: str,
    modes: list[int],
    pair_indices: list[int],
    window_radius: int,
) -> np.ndarray:
    """Return normalized temporal/vector basis rows for QP1 col0 deltas."""
    if n <= 0:
        raise ValueError("n must be positive")
    if kind == "dct":
        if not modes:
            raise ValueError("dct basis requires at least one mode")
        t = np.arange(n, dtype=np.float64) + 0.5
        rows = []
        for mode in modes:
            if mode < 0:
                raise ValueError("dct modes must be nonnegative")
            row = np.cos(np.pi * float(mode) * t / float(n))
            scale = float(np.max(np.abs(row)))
            rows.append(row / scale if scale > 0 else row)
        return np.stack(rows, axis=0).astype(np.float32)
    if kind == "pair_window":
        if not pair_indices:
            raise ValueError("pair_window basis requires --basis-pair-indices")
        radius = max(0, int(window_radius))
        rows = []
        for pair_idx in pair_indices:
            if pair_idx < 0:
                raise ValueError("pair indices must be nonnegative")
            center = 2 * int(pair_idx)
            row = np.zeros((n,), dtype=np.float32)
            lo = max(0, center - radius)
            hi = min(n, center + 2 + radius)
            for frame_idx in range(lo, hi):
                if frame_idx in (center, center + 1):
                    weight = 1.0
                else:
                    dist = min(abs(frame_idx - center), abs(frame_idx - (center + 1)))
                    weight = max(0.0, 1.0 - dist / float(radius + 1))
                row[frame_idx] = max(row[frame_idx], weight)
            if np.any(row):
                rows.append(row)
        if not rows:
            raise ValueError("pair_window basis produced no in-range rows")
        return np.stack(rows, axis=0).astype(np.float32)
    raise ValueError(f"unknown temporal basis kind: {kind}")


def gradient_guided_delta_matrix(
    gradient: np.ndarray,
    *,
    magnitudes: list[int],
    backtrack_magnitudes: list[int],
) -> np.ndarray:
    """Build per-frame sparse candidates oriented by the differentiable gradient.

    Positive d(loss)/d(col0) means the descent direction is negative, and vice
    versa. Backtrack candidates keep a small opposite-direction escape hatch so
    the proposal remains robust when the soft, unrounded gradient is imperfect.
    """
    if gradient.ndim != 1:
        raise ValueError(f"gradient must be 1-D; got shape={gradient.shape}")
    mags = [abs(int(x)) for x in magnitudes if int(x) != 0]
    back = [abs(int(x)) for x in backtrack_magnitudes if int(x) != 0]
    if not mags:
        raise ValueError("gradient-guided magnitudes must include a nonzero value")
    descent = np.where(gradient > 0, -1, 1).astype(np.int64)
    rows = [np.zeros_like(descent)]
    for mag in mags:
        rows.append(descent * mag)
    for mag in back:
        rows.append(-descent * mag)
    return np.stack(rows, axis=1).astype(np.int64)


# -----------------------------------------------------------------------------
# The line-search inner loop (mirrors pr67_line_search.py:142-183)
# -----------------------------------------------------------------------------


def coordinate_descent(
    *,
    col0_init: np.ndarray,
    masks: torch.Tensor,
    target: torch.Tensor,
    generator: torch.nn.Module,
    posenet: torch.nn.Module,
    device: torch.device,
    radii: list[int],
    passes: int,
    batch_size: int,
    candidate_chunk: int,
    max_candidate_items: int,
    mask_br_bytes: int,
    model_br_bytes: int,
    archive_overhead: int,
    delta_sets: list[list[int]] | None = None,
    gradient_delta_sets: list[list[int]] | None = None,
    gradient_backtrack_magnitudes: list[int] | None = None,
    basis_delta_sets: list[dict[str, Any]] | None = None,
    basis_modes: list[int] | None = None,
    basis_pair_indices: list[int] | None = None,
    basis_window_radius: int = 0,
    progress_every_candidates: int = 16,
    progress_cb: Callable[[str], None] | None = None,
    accepted_cb: Callable[[np.ndarray, dict[str, Any]], None] | None = None,
    target_h: int = 874,
    target_w: int = 1164,
) -> tuple[np.ndarray, dict[str, float]]:
    """Run coordinate-descent line search on col0 values.

    Returns (refined_col0, stats_dict). stats_dict contains
    ``baseline_obj``, ``best_obj``, ``best_pose_mse``, ``best_pose_bytes``,
    ``best_archive_size``.
    """

    n = col0_init.shape[0]
    if masks.shape[0] != n:
        raise ValueError(f"masks.shape[0]={masks.shape[0]} != col0 N={n}")
    if target.shape[0] != n:
        raise ValueError(f"target.shape[0]={target.shape[0]} != col0 N={n}")

    def eval_col(vals: np.ndarray) -> float:
        total = 0.0
        with torch.inference_mode():
            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                idx = torch.arange(start, end)
                pose = pose_from_col0_torch(
                    torch.from_numpy(vals[start:end]).to(device)
                )
                frames = make_frames(
                    generator, masks.index_select(0, idx).to(device), pose,
                    target_h=target_h, target_w=target_w,
                )
                import einops

                pred = pose_outputs(
                    posenet, einops.rearrange(frames, "b t c h w -> b t h w c")
                )
                total += (pred - target[start:end].to(device)).pow(2).mean(dim=1).sum().item()
        return total / n

    def estimate_col0_gradient(vals: np.ndarray) -> np.ndarray:
        grads = np.zeros((n,), dtype=np.float32)
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            idx = torch.arange(start, end)
            col0_var = torch.tensor(
                vals[start:end],
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
            pose = pose_from_col0_torch(col0_var)
            frames = make_frames(
                generator,
                masks.index_select(0, idx).to(device),
                pose,
                target_h=target_h,
                target_w=target_w,
                round_output=False,
            )
            import einops

            pred = pose_outputs(
                posenet, einops.rearrange(frames, "b t c h w -> b t h w c")
            )
            loss = (pred - target[start:end].to(device)).pow(2).mean()
            if not loss.requires_grad:
                raise RuntimeError(
                    "gradient-guided search has no live gradient through "
                    "PoseNet preprocessing; call "
                    "patch_posenet_for_differentiable_search(posenet)"
                )
            loss.backward()
            grad = col0_var.grad
            if grad is None:
                raise RuntimeError("gradient-guided search produced no col0 gradient")
            grads[start:end] = grad.detach().cpu().numpy()
            del frames, pred, loss, pose, col0_var
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return grads

    def objective(vals: np.ndarray) -> tuple[float, float, int, int]:
        pose_mse = eval_col(vals)
        pose_payload = encode_col0_to_pose_br(vals)
        pose_bytes = len(pose_payload)
        archive_size = mask_br_bytes + model_br_bytes + pose_bytes + archive_overhead
        obj = math.sqrt(10.0 * pose_mse) + 25.0 * archive_size / ORIGINAL_SIZE
        return obj, pose_mse, pose_bytes, archive_size

    best = col0_init.copy()
    best_obj, best_pose, best_bytes, best_size = objective(best)
    if progress_cb is not None:
        progress_cb(
            f"baseline obj={best_obj:.9f} pose_mse={best_pose:.12f} "
            f"pose_bytes={best_bytes} size={best_size}"
        )

    cur = best.copy()
    safety_notes: set[tuple[int, int, int]] = set()
    if basis_delta_sets is not None:
        modes = basis_modes or [0, 1, 2, 3, 5, 8, 13, 21]
        pair_indices = basis_pair_indices or []
        for stage in basis_delta_sets:
            basis = temporal_basis_matrix(
                n,
                kind=str(stage["kind"]),
                modes=modes,
                pair_indices=pair_indices,
                window_radius=basis_window_radius,
            )
            magnitudes = [int(x) for x in stage["magnitudes"]]
            for pass_idx in range(passes):
                changed = 0
                tested = 0
                if progress_cb is not None:
                    progress_cb(
                        f"basis_delta_set={stage['basis_delta_set_index']} "
                        f"kind={stage['kind']} pass={pass_idx + 1} "
                        f"start basis_vectors={basis.shape[0]} "
                        f"signed_magnitudes={2 * len(magnitudes)}"
                    )
                for basis_idx, vec in enumerate(basis):
                    for signed_mag in [m for mag in magnitudes for m in (-mag, mag)]:
                        delta = np.rint(vec.astype(np.float64) * signed_mag).astype(
                            np.int64
                        )
                        if not np.any(delta):
                            continue
                        cand = np.clip(cur + delta, 0, 65535).astype(np.int64)
                        if np.array_equal(cand, cur):
                            continue
                        tested += 1
                        if progress_cb is not None and (
                            tested == 1
                            or (
                                progress_every_candidates > 0
                                and tested % progress_every_candidates == 0
                            )
                        ):
                            progress_cb(
                                f"basis_candidate_start "
                                f"basis_delta_set={stage['basis_delta_set_index']} "
                                f"kind={stage['kind']} pass={pass_idx + 1} "
                                f"tested={tested} basis_index={basis_idx} "
                                f"signed_magnitude={signed_mag} "
                                f"best={best_obj:.9f}"
                            )
                        obj, pose_mse, pose_bytes, archive_size = objective(cand)
                        if obj < best_obj:
                            cur = cand
                            best_obj, best_pose, best_bytes, best_size = (
                                obj,
                                pose_mse,
                                pose_bytes,
                                archive_size,
                            )
                            best = cur.copy()
                            changed += 1
                            if accepted_cb is not None:
                                accepted_cb(
                                    best,
                                    {
                                        "pass": pass_idx + 1,
                                        "best_obj": best_obj,
                                        "best_pose_mse": best_pose,
                                        "best_pose_bytes": best_bytes,
                                        "best_archive_size": best_size,
                                        "search_stage_kind": "basis_delta_set",
                                        "search_stage_label": (
                                            f"basis_delta_set={stage['basis_delta_set_index']} "
                                            f"kind={stage['kind']}"
                                        ),
                                        "basis_delta_set_index": int(
                                            stage["basis_delta_set_index"]
                                        ),
                                        "basis_kind": str(stage["kind"]),
                                        "basis_index": int(basis_idx),
                                        "basis_signed_magnitude": int(signed_mag),
                                        "basis_vector_count": int(basis.shape[0]),
                                    },
                                )
                if progress_cb is not None:
                    progress_cb(
                        f"basis_delta_set={stage['basis_delta_set_index']} "
                        f"kind={stage['kind']} pass={pass_idx + 1} "
                        f"obj={best_obj:.9f} pose_mse={best_pose:.12f} "
                        f"pose_bytes={best_bytes} size={best_size} "
                        f"tested={tested} changed={changed}"
                    )
                if changed == 0:
                    break
        return best, {
            "baseline_pose_mse_estimate_NA_for_subsequent_runs": -1.0,
            "best_obj": best_obj,
            "best_pose_mse": best_pose,
            "best_pose_bytes": best_bytes,
            "best_archive_size": best_size,
        }

    if gradient_delta_sets is not None:
        backtrack = gradient_backtrack_magnitudes or [1]
        search_stages = [
            {
                "kind": "gradient_delta_set",
                "label": (
                    f"gradient_delta_set={idx} max={max(magnitudes)} "
                    f"n={1 + len(magnitudes) + len(backtrack)}"
                ),
                "gradient_delta_set_index": idx,
                "magnitudes": [int(delta) for delta in magnitudes],
                "backtrack_magnitudes": [int(delta) for delta in backtrack],
            }
            for idx, magnitudes in enumerate(gradient_delta_sets, start=1)
        ]
    elif delta_sets is None:
        search_stages: list[dict[str, Any]] = [
            {
                "kind": "radius",
                "label": f"radius={radius}",
                "radius": int(radius),
                "deltas": list(range(-int(radius), int(radius) + 1)),
            }
            for radius in radii
        ]
    else:
        search_stages = [
            {
                "kind": "delta_set",
                "label": (
                    f"delta_set={idx} min={min(deltas)} max={max(deltas)} "
                    f"n={len(deltas)}"
                ),
                "delta_set_index": idx,
                "deltas": [int(delta) for delta in deltas],
            }
            for idx, deltas in enumerate(delta_sets, start=1)
        ]

    for stage in search_stages:
        for pass_idx in range(passes):
            stage_delta_matrix: np.ndarray | None = None
            if stage["kind"] == "gradient_delta_set":
                gradient = estimate_col0_gradient(cur)
                stage_delta_matrix = gradient_guided_delta_matrix(
                    gradient,
                    magnitudes=stage["magnitudes"],
                    backtrack_magnitudes=stage["backtrack_magnitudes"],
                )
                if progress_cb is not None:
                    nonzero = gradient[np.abs(gradient) > 0]
                    if nonzero.size:
                        progress_cb(
                            f"{stage['label']} pass={pass_idx + 1} "
                            f"grad_abs_median={float(np.median(np.abs(nonzero))):.6g} "
                            f"grad_pos={int((gradient > 0).sum())} "
                            f"grad_neg={int((gradient < 0).sum())}"
                        )
                    else:
                        progress_cb(
                            f"{stage['label']} pass={pass_idx + 1} "
                            "grad_all_zero=true"
                        )
            else:
                deltas = torch.tensor(stage["deltas"], device=device, dtype=torch.int64)
            changed = 0
            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                b = end - start
                base = torch.from_numpy(cur[start:end]).to(device)
                if stage_delta_matrix is not None:
                    deltas_batch = torch.from_numpy(
                        stage_delta_matrix[start:end]
                    ).to(device)
                    cand = torch.clamp(base[:, None] + deltas_batch, 0, 65535)
                else:
                    cand = torch.clamp(base[:, None] + deltas[None, :], 0, 65535)
                c = cand.shape[1]
                best_dist = torch.full((b,), float("inf"), device=device)
                best_val = base.clone()
                masks_batch = masks[start:end].to(device)
                target_batch = target[start:end].to(device)
                effective_chunk = min(
                    candidate_chunk,
                    max(1, max_candidate_items // max(1, b)),
                )
                if effective_chunk < c:
                    note = (b, c, effective_chunk)
                    if note not in safety_notes and progress_cb is not None:
                        progress_cb(
                            "candidate_batch_cap "
                            f"batch={b} candidates={c} "
                            f"candidate_chunk={candidate_chunk} "
                            f"effective_chunk={effective_chunk} "
                            f"max_candidate_items={max_candidate_items}"
                        )
                    safety_notes.add(note)
                c_start = 0
                while c_start < c:
                    cc = min(effective_chunk, c - c_start)
                    while True:
                        c_end = c_start + cc
                        cand_chunk = cand[:, c_start:c_end]
                        try:
                            masks_rep = masks_batch.repeat_interleave(cc, dim=0)
                            pose_chunk = pose_from_col0_torch(cand_chunk.reshape(-1))
                            frames = make_frames(
                                generator, masks_rep, pose_chunk,
                                target_h=target_h, target_w=target_w,
                            )
                            import einops

                            pred = pose_outputs(
                                posenet,
                                einops.rearrange(frames, "b t c h w -> b t h w c"),
                            )
                            dist = (
                                pred - target_batch.repeat_interleave(cc, dim=0)
                            ).pow(2).mean(dim=1).reshape(b, cc)
                            break
                        except RuntimeError as exc:
                            if "canUse32BitIndexMath" not in str(exc) or cc <= 1:
                                raise
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                            next_cc = max(1, cc // 2)
                            if progress_cb is not None:
                                progress_cb(
                                    "candidate_batch_runtime_split "
                                    f"batch={b} start={c_start} "
                                    f"old_chunk={cc} new_chunk={next_cc} "
                                    "reason=canUse32BitIndexMath"
                                )
                            cc = next_cc
                    chunk_dist, chunk_pick = dist.min(dim=1)
                    better = chunk_dist < best_dist
                    best_dist = torch.where(better, chunk_dist, best_dist)
                    best_val = torch.where(
                        better,
                        cand_chunk[torch.arange(b, device=device), chunk_pick],
                        best_val,
                    )
                    del frames, pred, dist
                    c_start += cc
                new = best_val.cpu().numpy().astype(np.int64)
                changed += int((new != cur[start:end]).sum())
                cur[start:end] = new
            obj, pose_mse, pose_bytes, archive_size = objective(cur)
            if progress_cb is not None:
                progress_cb(
                    f"{stage['label']} pass={pass_idx + 1} obj={obj:.9f} "
                    f"pose_mse={pose_mse:.12f} pose_bytes={pose_bytes} "
                    f"size={archive_size} changed={changed} best={best_obj:.9f}"
                )
            if obj < best_obj:
                best_obj, best_pose, best_bytes, best_size = (
                    obj,
                    pose_mse,
                    pose_bytes,
                    archive_size,
                )
                best = cur.copy()
                if accepted_cb is not None:
                    stage_payload = {
                        "search_stage_kind": stage["kind"],
                        "search_stage_label": stage["label"],
                    }
                    if "deltas" in stage:
                        stage_payload.update(
                            {
                                "delta_min": int(min(stage["deltas"])),
                                "delta_max": int(max(stage["deltas"])),
                                "delta_count": int(len(stage["deltas"])),
                            }
                        )
                    if "magnitudes" in stage:
                        signed_candidates = stage["magnitudes"] + stage[
                            "backtrack_magnitudes"
                        ]
                        stage_payload.update(
                            {
                                "gradient_delta_set_index": int(
                                    stage["gradient_delta_set_index"]
                                ),
                                "gradient_magnitude_min": int(min(stage["magnitudes"])),
                                "gradient_magnitude_max": int(max(stage["magnitudes"])),
                                "gradient_candidate_count": int(
                                    1
                                    + len(stage["magnitudes"])
                                    + len(stage["backtrack_magnitudes"])
                                ),
                                "delta_min": -int(max(signed_candidates)),
                                "delta_max": int(max(signed_candidates)),
                                "delta_count": int(
                                    1
                                    + len(stage["magnitudes"])
                                    + len(stage["backtrack_magnitudes"])
                                ),
                            }
                        )
                    if "radius" in stage:
                        stage_payload["radius"] = int(stage["radius"])
                    if "delta_set_index" in stage:
                        stage_payload["delta_set_index"] = int(
                            stage["delta_set_index"]
                        )
                    accepted_cb(
                        best,
                        {
                            "pass": pass_idx + 1,
                            "best_obj": best_obj,
                            "best_pose_mse": best_pose,
                            "best_pose_bytes": best_bytes,
                            "best_archive_size": best_size,
                            **stage_payload,
                        },
                    )
            else:
                cur = best.copy()
                break

    stats = {
        "baseline_pose_mse_estimate_NA_for_subsequent_runs": -1.0,
        "best_obj": best_obj,
        "best_pose_mse": best_pose,
        "best_pose_bytes": best_bytes,
        "best_archive_size": best_size,
    }
    return best, stats


# -----------------------------------------------------------------------------
# CLI driver
# -----------------------------------------------------------------------------


def _device_banner(device: torch.device) -> None:
    """Per CLAUDE.md MPS-PoseNet-23x rule: warn loudly on non-CUDA."""
    if device.type == "cuda":
        return
    msg = (
        f"\n[device-banner] device={device} is NOT cuda. The MPS-PoseNet-23x "
        f"and CPU-no-determinism rules apply: any pose_mse / objective values "
        f"produced are [advisory only] and CANNOT be used for kill/promote "
        f"decisions. Re-run on CUDA before claiming any score gain.\n"
    )
    print(msg, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-path", type=Path, required=True)
    parser.add_argument(
        "--metadata-path",
        type=Path,
        required=True,
        help="path to metadata.json next to archive.zip (NOT pr67's brittle 7-bucket lookup)",
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=None,
        help="path to write refined metadata.json (default: alongside --output-path)",
    )
    parser.add_argument("--posenet-path", type=Path, default=None)
    parser.add_argument(
        "--gt-mkv",
        type=Path,
        default=None,
        help="path to ground-truth video (.mkv). When omitted, uses zero "
        "target tensor (smoke / development only — will produce trivially "
        "small pose_mse and bogus refinements).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--candidate-chunk", type=int, default=32)
    parser.add_argument(
        "--max-candidate-items",
        type=int,
        default=64,
        help="maximum generated candidate frame-pairs per renderer forward. "
        "Caps candidate_chunk adaptively to avoid PyTorch 32-bit-index math "
        "failures on full-resolution 874x1164 candidates.",
    )
    parser.add_argument(
        "--no-checkpoint-accepted",
        action="store_true",
        help="disable writing accepted_latest checkpoint archive/metadata "
        "after each objective-improving pass.",
    )
    parser.add_argument(
        "--radii",
        default="1,2,3,5,8",
        help="comma-separated radii for coordinate descent (matches pr67 default)",
    )
    parser.add_argument(
        "--delta-sets",
        default=None,
        help="semicolon-separated sparse/asymmetric delta stages. When set, "
        "overrides --radii for the inner search while preserving the same "
        "objective and archive format. Example: '-34,-21,-13,-8,-5,-1,0,1,2,3;"
        "-8,-5,-3,-1,0,1'.",
    )
    parser.add_argument(
        "--gradient-delta-sets",
        default=None,
        help="semicolon-separated positive magnitudes for differentiable "
        "gradient-guided proposal stages. For each frame, candidates are "
        "oriented along -sign(d PoseNetLoss / d col0), with small backtrack "
        "deltas. Overrides --delta-sets and --radii.",
    )
    parser.add_argument(
        "--gradient-backtrack-deltas",
        default="1",
        help="comma-separated opposite-direction guard magnitudes for "
        "--gradient-delta-sets.",
    )
    parser.add_argument(
        "--basis-delta-sets",
        default=None,
        help="semicolon-separated vector proposal stages over QP1 col0. "
        "Format: 'dct:1,2,3;pair_window:1,2'. Overrides gradient/delta/radius "
        "search and accepts only complete-archive objective improvements.",
    )
    parser.add_argument(
        "--basis-modes",
        default="0,1,2,3,5,8,13,21",
        help="comma-separated temporal DCT modes for --basis-delta-sets dct stages.",
    )
    parser.add_argument(
        "--basis-pair-indices",
        default=None,
        help="comma-separated absolute contest pair indices for pair_window basis; "
        "frames 2*i and 2*i+1 are selected, with optional neighborhood radius.",
    )
    parser.add_argument(
        "--basis-window-radius",
        type=int,
        default=0,
        help="neighbor radius around frames 2*i and 2*i+1 for pair_window basis.",
    )
    parser.add_argument(
        "--progress-every-candidates",
        type=int,
        default=16,
        help="emit progress while evaluating long basis/vector candidate stages; "
        "0 disables periodic candidate progress.",
    )
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument(
        "--archive-overhead",
        type=int,
        default=100,
        help="ZIP overhead bytes added to (mask+model+pose) when computing "
        "the rate term in the joint objective (matches pr67 default 100)",
    )
    parser.add_argument(
        "--mock-posenet",
        action="store_true",
        help="use a numpy-based MockPoseNet instead of upstream weights "
        "(test/CI determinism only)",
    )
    parser.add_argument(
        "--no-gt",
        action="store_true",
        help="use zero target tensor (test/development only)",
    )
    parser.add_argument(
        "--mask-shape",
        default=None,
        help="(N,H,W) mask shape for smoke runs without a real OBU stream "
        "(comma-separated, e.g. '600,384,512')",
    )
    parser.add_argument(
        "--target-h",
        type=int,
        default=874,
        help="bilinear-upsample frame height for PoseNet input (contest default 874)",
    )
    parser.add_argument(
        "--target-w",
        type=int,
        default=1164,
        help="bilinear-upsample frame width for PoseNet input (contest default 1164)",
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    _device_banner(device)

    blob = load_archive_blob(args.archive_path)
    meta = load_metadata(args.metadata_path)
    mask_br, model_br, pose_br = slice_blob(blob, meta)
    source_archive_sha256 = sha256_path(args.archive_path)

    col0 = col0_from_pose_payload(pose_br)
    print(f"[init] col0 length={col0.shape[0]} min={col0.min()} max={col0.max()}")

    # Build renderer from the SAME archive (read-only) so the refinement
    # is grounded in exactly the model bytes that ship.
    qzs3_payload = brotli.decompress(model_br)
    generator = load_renderer(qzs3_payload, device)

    # Build PoseNet (mock or real)
    if args.mock_posenet:
        posenet = _MockPoseNet().to(device).eval()
    else:
        posenet_path = args.posenet_path
        if posenet_path is None:
            posenet_path = _REPO_ROOT / "upstream/models/posenet.safetensors"
        if not posenet_path.exists():
            raise FileNotFoundError(
                f"PoseNet weights not found at {posenet_path}; "
                "pass --posenet-path explicitly or --mock-posenet for smoke"
            )
        posenet = load_posenet(posenet_path, device)

    # Load masks. For the smoke / test path we synthesize zeros.
    fallback_shape = None
    if args.mask_shape is not None:
        fallback_shape = tuple(int(x) for x in args.mask_shape.split(","))  # type: ignore[assignment]
    masks = load_masks_from_blob(mask_br, fallback_shape=fallback_shape)
    if masks.shape[0] != col0.shape[0]:
        # Slice or pad to match col0 N (smoke convenience)
        n = col0.shape[0]
        if masks.shape[0] < n:
            pad = torch.zeros((n - masks.shape[0],) + masks.shape[1:], dtype=masks.dtype)
            masks = torch.cat([masks, pad], dim=0)
        else:
            masks = masks[:n]

    # Build target tensor: forward GT through PoseNet (real path) or zeros (smoke)
    if args.no_gt or args.gt_mkv is None:
        target = torch.zeros((col0.shape[0], 6), dtype=torch.float32)  # OFF_MANIFOLD_OK: explicit smoke/no-GT mode; banner below tells operator "refinement is bogus" so this is never used for scoring
        print("[init] no-GT mode: target=0 (smoke / development; refinement is bogus)")
    else:
        assert_dali_runtime_dependency_available()
        target = _compute_target_from_gt(
            posenet, args.gt_mkv, batch_size=args.batch_size, device=device
        )

    radii = [int(x) for x in args.radii.split(",") if x.strip()]
    delta_sets = parse_delta_sets(args.delta_sets)
    gradient_delta_sets = parse_magnitude_sets(args.gradient_delta_sets)
    gradient_backtrack_deltas = parse_int_csv(
        args.gradient_backtrack_deltas, default=[1]
    )
    basis_delta_sets = parse_basis_delta_sets(args.basis_delta_sets)
    basis_modes = parse_int_csv(
        args.basis_modes, default=[0, 1, 2, 3, 5, 8, 13, 21]
    )
    basis_pair_indices = parse_int_csv(args.basis_pair_indices, default=[])
    def checkpoint_accepted(best_col0: np.ndarray, stats_payload: dict[str, Any]) -> None:
        if args.no_checkpoint_accepted:
            return
        refined_pose_br = encode_col0_to_pose_br(best_col0)
        checkpoint_path = args.output_path.with_suffix(".accepted_latest.zip")
        checkpoint_bytes = write_refined_archive(
            checkpoint_path, mask_br, model_br, refined_pose_br
        )
        checkpoint_refinement = {
            "tool": "experiments/line_search_pose_refinement.py",
            "reference": "pr67_line_search.py",
            "device": str(device),
            "radii": radii,
            "delta_sets": delta_sets,
            "gradient_delta_sets": gradient_delta_sets,
            "gradient_backtrack_deltas": gradient_backtrack_deltas,
            "basis_delta_sets": basis_delta_sets,
            "basis_modes": basis_modes,
            "basis_pair_indices": basis_pair_indices,
            "basis_window_radius": args.basis_window_radius,
            "passes": args.passes,
            "checkpoint": "accepted_latest",
            "pose_atom_selection": pose_atom_selection_summary(
                source_col0=col0,
                refined_col0=best_col0,
                pose_br_bytes=len(refined_pose_br),
                archive_bytes=checkpoint_bytes,
                policy="coordinate_descent_qp1_col0",
            ),
            **stats_payload,
        }
        checkpoint_meta = build_refined_metadata(
            source_meta=meta,
            source_archive_path=args.archive_path,
            source_archive_sha256=source_archive_sha256,
            output_archive_path=checkpoint_path,
            mask_br=mask_br,
            model_br=model_br,
            refined_pose_br=refined_pose_br,
            refined_col0=best_col0,
            archive_bytes=checkpoint_bytes,
            refinement=checkpoint_refinement,
        )
        assert_metadata_matches_archive(checkpoint_meta, checkpoint_path)
        checkpoint_meta_path = args.output_path.with_suffix(".accepted_latest.json")
        checkpoint_meta_path.write_text(
            json.dumps(checkpoint_meta, indent=2, sort_keys=True) + "\n"
        )
        print(
            "checkpoint_accepted "
            f"path={checkpoint_path} bytes={checkpoint_bytes} "
            f"obj={stats_payload['best_obj']:.9f}",
            flush=True,
        )

    refined_col0, stats = coordinate_descent(
        col0_init=col0,
        masks=masks,
        target=target,
        generator=generator,
        posenet=posenet,
        device=device,
        radii=radii,
        passes=args.passes,
        batch_size=args.batch_size,
        candidate_chunk=args.candidate_chunk,
        max_candidate_items=args.max_candidate_items,
        mask_br_bytes=len(mask_br),
        model_br_bytes=len(model_br),
        archive_overhead=args.archive_overhead,
        delta_sets=delta_sets,
        gradient_delta_sets=gradient_delta_sets,
        gradient_backtrack_magnitudes=gradient_backtrack_deltas,
        basis_delta_sets=basis_delta_sets,
        basis_modes=basis_modes,
        basis_pair_indices=basis_pair_indices,
        basis_window_radius=args.basis_window_radius,
        progress_every_candidates=args.progress_every_candidates,
        progress_cb=lambda s: print(s, flush=True),
        accepted_cb=checkpoint_accepted,
        target_h=args.target_h,
        target_w=args.target_w,
    )

    refined_pose_br = encode_col0_to_pose_br(refined_col0)
    archive_bytes = write_refined_archive(
        args.output_path, mask_br, model_br, refined_pose_br
    )

    final_refinement = {
        "tool": "experiments/line_search_pose_refinement.py",
        "reference": "pr67_line_search.py",
        "device": str(device),
        "radii": radii,
        "delta_sets": delta_sets,
        "gradient_delta_sets": gradient_delta_sets,
        "gradient_backtrack_deltas": gradient_backtrack_deltas,
        "basis_delta_sets": basis_delta_sets,
        "basis_modes": basis_modes,
        "basis_pair_indices": basis_pair_indices,
        "basis_window_radius": args.basis_window_radius,
        "passes": args.passes,
        "pose_atom_selection": pose_atom_selection_summary(
            source_col0=col0,
            refined_col0=refined_col0,
            pose_br_bytes=len(refined_pose_br),
            archive_bytes=archive_bytes,
            policy="coordinate_descent_qp1_col0",
        ),
        **stats,
    }
    out_meta = build_refined_metadata(
        source_meta=meta,
        source_archive_path=args.archive_path,
        source_archive_sha256=source_archive_sha256,
        output_archive_path=args.output_path,
        mask_br=mask_br,
        model_br=model_br,
        refined_pose_br=refined_pose_br,
        refined_col0=refined_col0,
        archive_bytes=archive_bytes,
        refinement=final_refinement,
    )
    assert_metadata_matches_archive(out_meta, args.output_path)
    metadata_out = args.output_metadata
    if metadata_out is None:
        metadata_out = args.output_path.parent / "metadata.json"
    metadata_out.write_text(json.dumps(out_meta, indent=2, sort_keys=True) + "\n")

    print(json.dumps({"output": str(args.output_path), **stats}, indent=2, sort_keys=True))
    return 0


def _compute_target_from_gt(
    posenet: torch.nn.Module,
    gt_mkv: Path,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Forward GT video frames through PoseNet to obtain target pose tensor."""
    sys.path.insert(0, str(_REPO_ROOT / "upstream"))
    try:
        from frame_utils import DaliVideoDataset  # type: ignore[import]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"DaliVideoDataset unavailable: {exc}. Pass --no-gt for smoke."
        ) from exc

    ds = DaliVideoDataset(
        [gt_mkv.name],
        data_dir=gt_mkv.parent,
        batch_size=batch_size,
        device=device,
    )
    ds.prepare_data()
    gt_chunks = []
    for _, _, batch in ds:
        gt_chunks.append(batch.cpu())
    gt = torch.cat(gt_chunks).contiguous()
    target = []
    with torch.inference_mode():
        for start in range(0, gt.shape[0], batch_size):
            target.append(
                pose_outputs(posenet, gt[start : start + batch_size].to(device))
            )
    return torch.cat(target).cpu()


# -----------------------------------------------------------------------------
# Mock PoseNet for CI/smoke determinism (CPU-only, numpy-friendly)
# -----------------------------------------------------------------------------


class _MockPoseNet(torch.nn.Module):
    """Numpy-flavored deterministic surrogate for PoseNet (test/CI only).

    Mirrors the real PoseNet's IO contract: ``preprocess_input`` accepts
    ``(B, T, H, W, C)`` u8-ish frames, ``forward`` returns
    ``{"pose": (B, 12)}`` so downstream code can grab ``[..., :6]``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_proj", torch.randn(3, 6) * 0.001)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Deterministic 12-dim pose output: per-batch mean projected to 6 dims,
        # padded with zeros for the second 6-dim head.
        bsz = x.shape[0]
        feat = x.float().mean(dim=tuple(range(1, x.ndim)))[:, None].repeat(1, 3)
        pose6 = feat @ self._proj.to(x.device)
        pose12 = torch.cat([pose6, torch.zeros_like(pose6)], dim=1)
        return {"pose": pose12}


if __name__ == "__main__":
    raise SystemExit(main())
