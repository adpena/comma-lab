# SPDX-License-Identifier: MIT
"""Checkpoint verification utilities.

Prevents wrong-checkpoint bugs by running a quick sanity check before
any experiment that depends on a trained renderer.  A trained renderer
(auth=0.87) produces PoseNet MSE < 1.0 on any pair.  A smoke-test model
(5 epochs) produces PoseNet > 100.  The check takes < 5 seconds.

The canonical checkpoint lives at:
    experiments/results/v5_lagrangian_renderer/renderer_best.pt
    MD5: cff8dca4

History:
    2026-04-15 -- All Vast.ai experiments used a 5-epoch smoke model
    (MD5: a9aee326) instead of the auth=0.87 renderer.  This module
    exists to make that class of bug impossible going forward.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

CANONICAL_CHECKPOINT_DIR = "experiments/results/v5_lagrangian_renderer"
"""Default directory for the trained renderer checkpoint."""

CANONICAL_CHECKPOINT_MD5_PREFIX = "cff8dca4"
"""First 8 hex chars of the MD5 of the auth=0.87 renderer checkpoint."""

WRONG_CHECKPOINT_MD5_PREFIX = "a9aee326"
"""First 8 hex chars of the MD5 of the 5-epoch smoke-test model."""


def md5_prefix(path: str | Path, length: int = 8) -> str:
    """Compute the first *length* hex chars of a file's MD5 hash.

    Reads the file in 64 KiB chunks to handle large checkpoints without
    loading them entirely into memory.

    Args:
        path: Path to the file.
        length: Number of hex characters to return (default 8).

    Returns:
        Hex string of the first *length* characters of the MD5 digest.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:length]


def verify_checkpoint_identity(
    checkpoint_path: str | Path,
    expected_md5_prefix: str = CANONICAL_CHECKPOINT_MD5_PREFIX,
) -> str:
    """Verify a checkpoint file by MD5 prefix.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        expected_md5_prefix: Expected MD5 hex prefix (default: canonical
            auth=0.87 renderer).

    Returns:
        The actual MD5 prefix string.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        ValueError: If the MD5 prefix does not match, with a diagnostic
            message identifying the wrong checkpoint if it matches the
            known smoke-test model.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    actual = md5_prefix(path)

    if actual == expected_md5_prefix:
        return actual

    # Provide extra diagnostic if it matches the known bad checkpoint
    if actual == WRONG_CHECKPOINT_MD5_PREFIX:
        raise ValueError(
            f"WRONG CHECKPOINT: {path}\n"
            f"  MD5 prefix: {actual} (5-epoch smoke-test model)\n"
            f"  Expected:   {expected_md5_prefix} (auth=0.87 renderer)\n"
            f"  This is the bug that invalidated all Vast.ai experiments on 2026-04-15.\n"
            f"  Use the correct checkpoint from {CANONICAL_CHECKPOINT_DIR}/"
        )

    raise ValueError(
        f"UNKNOWN CHECKPOINT: {path}\n"
        f"  MD5 prefix: {actual}\n"
        f"  Expected:   {expected_md5_prefix} (auth=0.87 renderer)\n"
        f"  This does not match any known checkpoint. Verify provenance."
    )


def verify_checkpoint_quality(
    checkpoint_path: str | Path,
    upstream_path: str,
    device: str = "mps",
    max_expected_posenet: float = 1.0,
) -> dict:
    """Quick 2-pair sanity check that a renderer checkpoint is the trained model.

    Loads the renderer, generates 4 frames (2 pairs), computes proxy PoseNet.
    A trained renderer should produce PoseNet < 1.0.  A smoke-test model
    produces PoseNet > 100.  Takes < 5 seconds.

    This function requires torch and the tac library to be importable.  For
    a cheaper check that only reads the file on disk, use
    :func:`verify_checkpoint_identity` instead.

    Args:
        checkpoint_path: Path to renderer .pt checkpoint.
        upstream_path: Path to the upstream scorer repo.
        device: Computation device ('mps', 'cuda', 'cpu').
        max_expected_posenet: Maximum acceptable PoseNet MSE.  Defaults
            to 1.0 -- generous enough for any trained renderer, strict
            enough to catch smoke-test models (PoseNet > 100).

    Returns:
        Dict with 'posenet_mse', 'segnet_disagree', 'md5_prefix', and
        'checkpoint_path' keys.

    Raises:
        ValueError: If PoseNet MSE is NaN/Inf (corrupt checkpoint) or
            exceeds *max_expected_posenet* (undertrained checkpoint).
    """
    from pathlib import Path as _Path

    import torch as _torch

    from tac.data import load_gt_video
    from tac.scorer import compute_proxy_score, extract_gt_masks, load_differentiable_scorers

    upstream = _Path(upstream_path)
    video_path = str(upstream / "videos" / "0.mkv")
    dev = _torch.device(device)

    # Load scorers
    posenet, segnet = load_differentiable_scorers(upstream, device=device)

    # Load renderer
    # Import here to avoid circular imports at module level
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "renderer_tto",
        _Path(__file__).parent.parent.parent / "experiments" / "renderer_tto.py",
    )
    _renderer_tto = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_renderer_tto)

    renderer = _renderer_tto.load_renderer(checkpoint_path, dev)

    # Decode just 4 frames (2 pairs)
    gt_frames = load_gt_video(video_path, n_frames=4)
    masks = extract_gt_masks(gt_frames, segnet, dev)
    rendered = _renderer_tto.generate_renderer_frames(renderer, masks, dev)

    # Compute proxy score
    result = compute_proxy_score(rendered, gt_frames, posenet, segnet, dev)

    posenet_mse = result["pose"]
    md5 = md5_prefix(checkpoint_path)

    info = {
        "posenet_mse": posenet_mse,
        "segnet_disagree": result["seg"],
        "score": result["score"],
        "md5_prefix": md5,
        "checkpoint_path": str(checkpoint_path),
    }

    if math.isnan(posenet_mse) or math.isinf(posenet_mse):
        raise ValueError(
            f"CHECKPOINT SANITY CHECK FAILED -- NaN/Inf PoseNet\n"
            f"  PoseNet MSE: {posenet_mse}\n"
            f"  MD5 prefix:  {md5}\n"
            f"  Path:        {checkpoint_path}\n"
            f"  The checkpoint may be corrupt or produce degenerate output."
        )

    if posenet_mse > max_expected_posenet:
        raise ValueError(
            f"CHECKPOINT SANITY CHECK FAILED\n"
            f"  PoseNet MSE: {posenet_mse:.4f} (max allowed: {max_expected_posenet})\n"
            f"  MD5 prefix:  {md5}\n"
            f"  Path:        {checkpoint_path}\n"
            f"  A trained renderer produces PoseNet < 1.0.\n"
            f"  A smoke-test model produces PoseNet > 100.\n"
            f"  This checkpoint is NOT a properly trained renderer."
        )

    return info
