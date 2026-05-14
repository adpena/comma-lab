# SPDX-License-Identifier: MIT
"""Lane LR — Low-Rank pose adaptation for compress-time pose-space TTO.

Per memory `project_posenet_rank1_discovery`, the PoseNet Jacobian is rank ≈
1.008 with 99.8% variance in dim 0 — meaning the (N_pairs, 6) per-pair pose
tensor lives on a low-dimensional manifold. LoRA pose factorises that tensor
as

    poses = base + U @ V

where ``base`` is the frozen warm-start pose tensor (shape ``(N, 6)``),
``U`` is the per-pair coefficient matrix (shape ``(N, R)``), and ``V`` is the
shared low-rank basis (shape ``(R, 6)``). ``U`` and ``V`` are the only
trainable parameters.

Rate impact (for the standard 600-pair contest split):

  full-rank pose tensor:  600 × 6 × 4 = 14400 bytes (fp32) / 7200 (fp16)
  LoRA rank-1:            600 + 6 = 606 fp16 ≈ 1212 bytes
  LoRA rank-2:            1200 + 12 = 1212 fp16 ≈ 2424 bytes
  LoRA rank-3:            1800 + 18 = 1818 fp16 ≈ 3636 bytes

So rank-1 saves ~6 KB vs the fp16 baseline; in archive-rate score units
(25 * delta_bytes / ORIGINAL_VIDEO_BYTES = 25 * 6000 / 37545489 ≈ 0.004)
that's a -0.004 score band before any distortion change.

Storage format (LoRA-encoded ``optimized_poses.pt``):

The file is a torch.save() pickle whose top-level object is a ``dict``:

    {
        "format": "lora_pose_v1",     # sentinel
        "rank": int,                  # R ∈ {1, 2, 3, ...}
        "n_pairs": int,
        "pose_dim": int,              # always 6 currently
        "base": torch.Tensor,         # (N, pose_dim) float16 — frozen warm start
        "U": torch.Tensor,            # (N, R)        float16 — per-pair coeffs
        "V": torch.Tensor,            # (R, pose_dim) float16 — shared basis
    }

`tac.submission_archive.load_optimized_poses` detects this dict by the
"format" sentinel and returns the materialised ``base + U @ V`` tensor —
downstream consumers (inflate_renderer, contest_auth_eval) see a vanilla
(N, pose_dim) float32 tensor and need no LoRA awareness.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

LORA_FORMAT_SENTINEL_V1 = "lora_pose_v1"


class LoRAPose(nn.Module):
    """Low-rank parameterisation of a per-pair pose tensor.

    Args:
        base: (N, pose_dim) float tensor — frozen warm-start poses. Detached
            and registered as a buffer so optimiser state does NOT touch it.
        rank: rank of the U @ V factorisation. Must be >= 1.

    The forward pass returns ``base + U @ V`` (shape ``(N, pose_dim)``).
    """

    def __init__(self, base: torch.Tensor, rank: int) -> None:
        super().__init__()
        if base.ndim != 2:
            raise ValueError(
                f"LoRAPose: base must be 2-D (N, pose_dim), got shape "
                f"{tuple(base.shape)}"
            )
        if rank < 1:
            raise ValueError(f"LoRAPose: rank must be >= 1, got {rank}")
        n_pairs, pose_dim = base.shape
        if rank > min(n_pairs, pose_dim):
            # Mathematically OK but pointless — full-rank reachable already.
            # Don't crash, but warn loudly so the operator notices the
            # waste (R=6 on pose_dim=6 has zero rate benefit).
            print(
                f"[LoRAPose] WARNING: rank={rank} >= min(n_pairs={n_pairs}, "
                f"pose_dim={pose_dim}); LoRA is not reducing parameter count.",
                flush=True,
            )

        # Frozen base — NOT a parameter. Registered as a buffer so it
        # follows .to(device) / .cuda() without showing up in .parameters().
        self.register_buffer("base", base.detach().to(torch.float32).clone())
        self.rank = int(rank)
        self.n_pairs = int(n_pairs)
        self.pose_dim = int(pose_dim)

        # U: (N, R) — per-pair LoRA coefficients. Initialised to zero so the
        # forward pass starts at base (identity warm-start). LoRA convention
        # is one matrix zero, the other Kaiming; here we zero U because
        # downstream we want the optimiser to start at exactly the warm-start
        # poses (matches the full-rank ``conditioning[:, :pose_dim] = init``
        # path in optimize_poses.py).
        self.U = nn.Parameter(torch.zeros(n_pairs, rank, dtype=torch.float32))
        # V: (R, pose_dim) — shared basis. Kaiming-init scaled by 1/rank so
        # combined U @ V starts small even after a few gradient steps push
        # U away from zero.
        v = torch.empty(rank, pose_dim, dtype=torch.float32)
        nn.init.kaiming_uniform_(v, a=5.0 ** 0.5)
        self.V = nn.Parameter(v / max(rank, 1))

    def forward(self) -> torch.Tensor:
        """Return materialised poses ``(N, pose_dim)`` = base + U @ V."""
        return self.base + self.U @ self.V

    @property
    def trainable_params(self) -> int:
        """Number of trainable scalar parameters (= N*R + R*pose_dim)."""
        return self.U.numel() + self.V.numel()

    def archive_bytes_fp16(self) -> int:
        """Predicted on-disk byte cost when serialised via
        ``encode_lora_poses_dict`` and torch.save'd. Used by the launch
        script for rate-budget banners.

        Note: this is the *raw tensor* byte count for U + V + base. The
        actual pickle has a few hundred bytes of dict + header overhead,
        which is negligible at the rank-1 budget.
        """
        # 2 bytes/element fp16; base is part of the file (we keep it so
        # inflate-side reconstruction does not need a separate base file).
        return (
            self.U.numel() * 2
            + self.V.numel() * 2
            + self.base.numel() * 2
        )


def encode_lora_poses_dict(lora: LoRAPose) -> dict[str, Any]:
    """Serialise a trained LoRAPose into the canonical archive dict.

    Stored on disk via ``torch.save(encode_lora_poses_dict(lora), path)``.
    All tensors are cast to float16 (the same precision as the full-rank
    ``optimized_poses.bin`` pathway) for rate parity.
    """
    return {
        "format": LORA_FORMAT_SENTINEL_V1,
        "rank": int(lora.rank),
        "n_pairs": int(lora.n_pairs),
        "pose_dim": int(lora.pose_dim),
        "base": lora.base.detach().to(torch.float16).cpu(),
        "U": lora.U.detach().to(torch.float16).cpu(),
        "V": lora.V.detach().to(torch.float16).cpu(),
    }


def is_lora_poses_dict(obj: Any) -> bool:
    """Return True if ``obj`` looks like a LoRA-encoded pose pickle."""
    return (
        isinstance(obj, dict)
        and obj.get("format") == LORA_FORMAT_SENTINEL_V1
    )


def decode_lora_poses_dict(obj: dict[str, Any], pose_dim: int = 6) -> torch.Tensor:
    """Reconstruct the materialised ``(N, pose_dim)`` float32 pose tensor
    from the on-disk dict.

    Raises ``ValueError`` with a specific diagnostic on any field mismatch.
    """
    if not is_lora_poses_dict(obj):
        raise ValueError(
            f"decode_lora_poses_dict: not a LoRA-encoded pose dict "
            f"(got format={obj.get('format')!r}, expected "
            f"{LORA_FORMAT_SENTINEL_V1!r})"
        )
    for key in ("rank", "n_pairs", "pose_dim", "base", "U", "V"):
        if key not in obj:
            raise ValueError(
                f"decode_lora_poses_dict: missing required key {key!r} in "
                f"LoRA pose dict; have {sorted(obj.keys())}"
            )
    rank = int(obj["rank"])
    n_pairs = int(obj["n_pairs"])
    declared_pose_dim = int(obj["pose_dim"])
    if declared_pose_dim != pose_dim:
        raise ValueError(
            f"decode_lora_poses_dict: declared pose_dim={declared_pose_dim} "
            f"!= caller pose_dim={pose_dim}; this means the renderer's "
            f"FiLM input width does not match the pose factorisation."
        )
    base = obj["base"].to(torch.float32)
    U = obj["U"].to(torch.float32)
    V = obj["V"].to(torch.float32)
    if base.shape != (n_pairs, declared_pose_dim):
        raise ValueError(
            f"decode_lora_poses_dict: base shape {tuple(base.shape)} != "
            f"({n_pairs}, {declared_pose_dim})"
        )
    if U.shape != (n_pairs, rank):
        raise ValueError(
            f"decode_lora_poses_dict: U shape {tuple(U.shape)} != "
            f"({n_pairs}, {rank})"
        )
    if V.shape != (rank, declared_pose_dim):
        raise ValueError(
            f"decode_lora_poses_dict: V shape {tuple(V.shape)} != "
            f"({rank}, {declared_pose_dim})"
        )
    return base + U @ V


def save_lora_poses(lora: LoRAPose, path: Path | str) -> int:
    """Save a trained LoRAPose to disk via torch.save() and return file size.

    The saved file is a pickle whose top-level object is the dict produced
    by ``encode_lora_poses_dict``. ``load_optimized_poses`` detects this
    sentinel and reconstructs the materialised pose tensor transparently.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encode_lora_poses_dict(lora), str(path))
    return path.stat().st_size
