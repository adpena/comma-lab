# SPDX-License-Identifier: MIT
"""Lane LR-V2 — LEARNABLE-rank pose adaptation.

V1 oversight (memory ``project_posenet_rank1_discovery``): rank=1 was
hard-coded based on the 99.8%-variance heuristic. The OPTIMAL rank for
the contest score may be 1 OR 2 OR 3 — this is a learnable hyperparameter
that should not be set offline.

V2 fix: ``LearnableRankLoRAPose`` starts at ``max_rank`` (default 6) with
a per-rank gate (sigmoid of a learnable scalar). The gates are co-trained
with U + V via the same optimiser; ranks whose final gate value is below
``prune_threshold`` (default 0.1) are dropped before serialisation, giving
a data-driven effective rank.

Mathematically::

    poses = base + sum_r gate_r * (U[:, r:r+1] @ V[r:r+1, :])
          = base + (U * gate.unsqueeze(0)) @ V

where ``gate = sigmoid(logit_gate)`` and ``logit_gate`` is the learnable
``(max_rank,)`` parameter. After training we read the final gate values,
keep only the ranks where ``gate >= prune_threshold``, and serialise the
PRUNED U + V (so the on-disk archive carries only the effective rank).

Storage format (V2 ``optimized_poses.pt``)::

    {
        "format": "lora_pose_v2",            # sentinel — distinguishes from v1
        "rank": int,                          # PRUNED rank (effective)
        "max_rank": int,                      # original max_rank (for audit)
        "kept_indices": list[int],            # which gates survived pruning
        "n_pairs": int,
        "pose_dim": int,
        "base": torch.Tensor,                 # (N, pose_dim) fp16, frozen
        "U": torch.Tensor,                    # (N, kept_rank) fp16, GATED
        "V": torch.Tensor,                    # (kept_rank, pose_dim) fp16
        "final_gate_values": torch.Tensor,    # (max_rank,) fp16, audit only
    }

The reader reconstructs ``poses = base + U @ V`` exactly as for V1 —
the gate values were already absorbed into U at serialisation time, so
downstream consumers see a vanilla pose tensor and need no V2 awareness.

``tac.submission_archive.load_optimized_poses`` detects the format
sentinel and dispatches to ``decode_lora_v2_poses_dict``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

LORA_FORMAT_SENTINEL_V2 = "lora_pose_v2"

# Default prune threshold: ranks with gate < 0.1 contribute < 10% of their
# unpruned magnitude to the final pose tensor — empirically the breakpoint
# below which the rank is more noise than signal.
DEFAULT_PRUNE_THRESHOLD = 0.1


class LearnableRankLoRAPose(nn.Module):
    """LoRA pose factorisation with LEARNABLE per-rank gates.

    Parameters
    ----------
    base : torch.Tensor
        (N, pose_dim) frozen warm-start poses. Detached and registered as
        a buffer.
    max_rank : int
        Maximum rank explored during training. The effective rank after
        pruning will be in ``[1, max_rank]``. Default 6 (the pose_dim
        upper bound — full-rank reachable in principle).
    init_gate_logit : float
        Initial logit value for every gate. Sigmoid(0) = 0.5, so default
        starts every rank at half-on. Negative values bias toward sparsity.

    Forward returns ``base + (U * gate) @ V``. Gradients flow into
    ``U``, ``V``, AND ``logit_gate`` so the optimiser can drive unhelpful
    gates toward zero (effectively pruning that rank).
    """

    def __init__(
        self,
        base: torch.Tensor,
        max_rank: int = 6,
        init_gate_logit: float = 0.0,
    ) -> None:
        super().__init__()
        if base.ndim != 2:
            raise ValueError(
                f"LearnableRankLoRAPose: base must be 2-D (N, pose_dim), got "
                f"shape {tuple(base.shape)}"
            )
        if max_rank < 1:
            raise ValueError(
                f"LearnableRankLoRAPose: max_rank must be >= 1, got {max_rank}"
            )
        n_pairs, pose_dim = base.shape
        if max_rank > min(n_pairs, pose_dim) and max_rank > pose_dim:
            # Mathematically max_rank > pose_dim is wasteful (no extra
            # representational capacity); warn but don't crash so callers
            # exploring weird configs still work.
            print(
                f"[LearnableRankLoRAPose] WARNING: max_rank={max_rank} > "
                f"pose_dim={pose_dim}; ranks beyond pose_dim cannot add "
                f"representational capacity to a (N, pose_dim) tensor.",
                flush=True,
            )

        self.register_buffer(
            "base", base.detach().to(torch.float32).clone()
        )
        self.max_rank = int(max_rank)
        self.n_pairs = int(n_pairs)
        self.pose_dim = int(pose_dim)

        # U: zero-init so warm-start identity holds at step 0 regardless of
        # gate values (because U=0 → U@V=0 → forward()=base).
        self.U = nn.Parameter(
            torch.zeros(n_pairs, max_rank, dtype=torch.float32)
        )
        # V: Kaiming init / max_rank — same scaling as the V1 LoRAPose.
        v = torch.empty(max_rank, pose_dim, dtype=torch.float32)
        nn.init.kaiming_uniform_(v, a=5.0 ** 0.5)
        self.V = nn.Parameter(v / max(max_rank, 1))
        # Per-rank gate logits — start at init_gate_logit so sigmoid(.) is
        # the user-controlled initial gate value for every rank.
        self.logit_gate = nn.Parameter(
            torch.full((max_rank,), float(init_gate_logit), dtype=torch.float32)
        )

    @property
    def gate(self) -> torch.Tensor:
        """Sigmoid-squashed per-rank gate, ``(max_rank,) in (0, 1)``."""
        return torch.sigmoid(self.logit_gate)

    def forward(self) -> torch.Tensor:
        """Return materialised poses ``(N, pose_dim)``.

        Math: ``base + (U * gate) @ V`` where the gate broadcasts over the
        N axis. Equivalent to ``base + sum_r gate_r * U[:, r:r+1] @ V[r:r+1]``.
        """
        gated_U = self.U * self.gate.unsqueeze(0)  # (N, R)
        return self.base + gated_U @ self.V

    @property
    def trainable_params(self) -> int:
        """U.numel() + V.numel() + logit_gate.numel()."""
        return self.U.numel() + self.V.numel() + self.logit_gate.numel()

    def archive_bytes_fp16(
        self, kept_indices: list[int] | None = None
    ) -> int:
        """Predicted on-disk byte cost of the (post-prune) tensors.

        When ``kept_indices`` is None, predicts the un-pruned worst-case
        (all max_rank ranks survive). When supplied, predicts the actual
        cost given the prune decision.
        """
        if kept_indices is None:
            kept_rank = self.max_rank
        else:
            kept_rank = len(kept_indices)
        # base + U_pruned + V_pruned + gate audit
        return (
            self.base.numel() * 2
            + self.n_pairs * kept_rank * 2
            + kept_rank * self.pose_dim * 2
            + self.max_rank * 2  # final_gate_values for audit
        )

    def kept_indices(
        self, prune_threshold: float = DEFAULT_PRUNE_THRESHOLD
    ) -> list[int]:
        """Return the rank indices whose final gate value >= threshold.

        Empty result is replaced with ``[argmax(gate)]`` so the post-prune
        rank is never zero (a zero-rank LoRA degenerates to ``base + 0`` =
        the warm-start, which is identical to shipping no poses at all —
        not the operator's intent).
        """
        with torch.no_grad():
            g = self.gate.detach().cpu()
        keep = (g >= prune_threshold).nonzero(as_tuple=False).flatten().tolist()
        if not keep:
            # Degenerate: every gate pruned. Keep the strongest one so the
            # archive still encodes SOME LoRA delta (matches V1 rank-1 floor).
            keep = [int(g.argmax().item())]
        return keep


def encode_lora_v2_poses_dict(
    lora: LearnableRankLoRAPose,
    prune_threshold: float = DEFAULT_PRUNE_THRESHOLD,
) -> dict[str, Any]:
    """Serialise a trained LearnableRankLoRAPose with rank pruning applied.

    The on-disk U and V are GATED (gate values absorbed into U) and PRUNED
    (only kept ranks survive), so the reader's ``base + U @ V`` gives the
    same result as ``base + (U * gate) @ V`` on the full unpruned tensors.

    Parameters
    ----------
    lora : LearnableRankLoRAPose
        Trained module to serialise.
    prune_threshold : float
        Minimum gate value for a rank to survive pruning. Default 0.1.

    Returns
    -------
    dict
        On-disk schema (see module docstring). All tensors fp16.
    """
    kept_indices = lora.kept_indices(prune_threshold=prune_threshold)
    with torch.no_grad():
        gate = lora.gate.detach().cpu()              # (max_rank,) float
        # Absorb gate into U so the reader can do plain U@V.
        gated_U = (lora.U.detach().cpu() * gate.unsqueeze(0))
        # Index out kept ranks only.
        idx = torch.tensor(kept_indices, dtype=torch.long)
        U_kept = gated_U.index_select(1, idx)        # (N, kept_rank)
        V_kept = lora.V.detach().cpu().index_select(0, idx)  # (kept_rank, pose_dim)
        base_cpu = lora.base.detach().cpu()
    return {
        "format": LORA_FORMAT_SENTINEL_V2,
        "rank": len(kept_indices),
        "max_rank": int(lora.max_rank),
        "kept_indices": [int(i) for i in kept_indices],
        "n_pairs": int(lora.n_pairs),
        "pose_dim": int(lora.pose_dim),
        "base": base_cpu.to(torch.float16),
        "U": U_kept.to(torch.float16),
        "V": V_kept.to(torch.float16),
        "final_gate_values": gate.to(torch.float16),
    }


def is_lora_v2_poses_dict(obj: Any) -> bool:
    """Return True iff ``obj`` is a V2-encoded LoRA pose dict."""
    return (
        isinstance(obj, dict)
        and obj.get("format") == LORA_FORMAT_SENTINEL_V2
    )


def decode_lora_v2_poses_dict(
    obj: dict[str, Any], pose_dim: int = 6
) -> torch.Tensor:
    """Reconstruct ``(N, pose_dim)`` float32 poses from the V2 dict.

    Validates every shape and the format sentinel. Raises ``ValueError``
    with a specific diagnostic on any field mismatch.
    """
    if not is_lora_v2_poses_dict(obj):
        raise ValueError(
            f"decode_lora_v2_poses_dict: not a LoRA-V2 pose dict (got "
            f"format={obj.get('format')!r}, expected "
            f"{LORA_FORMAT_SENTINEL_V2!r})"
        )
    for key in (
        "rank", "max_rank", "kept_indices", "n_pairs", "pose_dim",
        "base", "U", "V", "final_gate_values",
    ):
        if key not in obj:
            raise ValueError(
                f"decode_lora_v2_poses_dict: missing required key {key!r}; "
                f"have {sorted(obj.keys())}"
            )
    rank = int(obj["rank"])
    max_rank = int(obj["max_rank"])
    n_pairs = int(obj["n_pairs"])
    declared_pose_dim = int(obj["pose_dim"])
    kept_indices = list(obj["kept_indices"])
    if declared_pose_dim != pose_dim:
        raise ValueError(
            f"decode_lora_v2_poses_dict: declared pose_dim={declared_pose_dim} "
            f"!= caller pose_dim={pose_dim}"
        )
    if len(kept_indices) != rank:
        raise ValueError(
            f"decode_lora_v2_poses_dict: kept_indices length "
            f"{len(kept_indices)} != rank {rank}"
        )
    if not (1 <= rank <= max_rank):
        raise ValueError(
            f"decode_lora_v2_poses_dict: rank {rank} not in [1, max_rank={max_rank}]"
        )
    base = obj["base"].to(torch.float32)
    U = obj["U"].to(torch.float32)
    V = obj["V"].to(torch.float32)
    if base.shape != (n_pairs, declared_pose_dim):
        raise ValueError(
            f"decode_lora_v2_poses_dict: base shape {tuple(base.shape)} != "
            f"({n_pairs}, {declared_pose_dim})"
        )
    if U.shape != (n_pairs, rank):
        raise ValueError(
            f"decode_lora_v2_poses_dict: U shape {tuple(U.shape)} != "
            f"({n_pairs}, {rank})"
        )
    if V.shape != (rank, declared_pose_dim):
        raise ValueError(
            f"decode_lora_v2_poses_dict: V shape {tuple(V.shape)} != "
            f"({rank}, {declared_pose_dim})"
        )
    return base + U @ V


def save_lora_v2_poses(
    lora: LearnableRankLoRAPose,
    path: Path | str,
    prune_threshold: float = DEFAULT_PRUNE_THRESHOLD,
) -> int:
    """Encode + persist a LearnableRankLoRAPose to disk and return file size.

    Equivalent to ``torch.save(encode_lora_v2_poses_dict(...), path)``;
    the dict's format sentinel is detected by ``load_optimized_poses``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = encode_lora_v2_poses_dict(lora, prune_threshold=prune_threshold)
    torch.save(obj, str(path))
    return path.stat().st_size
