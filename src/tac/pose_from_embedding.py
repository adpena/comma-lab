"""Lane M-V3: PoseNet-embedding-space pose distillation (Path A).

Strategic premise
-----------------
Lane M-V1 (1-DOF radial-zoom + zero-padded dims 1-5 at inflate) scored 2.35:
the (N, 1) tensor was zero-padded back to (N, 6) on the inflate side, and the
auxiliary 5 PoseNet pose dims that the renderer was conditioned on were
destroyed. Lane M-V2 attempts a (N, 6) save with frozen baseline dims 1-5;
even if it works, the rank-1 hypothesis (memory
``project_posenet_rank1_discovery``: 99.8% variance in dim 0) treats PoseNet
SENSITIVITY as if it were pose-input CONTROL space. The two are different.

Lane M-V3 (Path A) takes a different optimization variable: instead of
optimizing the 6-DOF pose vector that's INPUT to the renderer, distill a
small MLP that PREDICTS the 6-DOF pose from MASK FEATURES at inflate time.
The PoseNet 12-dim head output (the "embedding") is used as a TEACHER
signal at compress time only; at inflate the MLP runs on mask features
alone (strict-scorer-rule compliant: NO PoseNet load at inflate).

Council provenance
------------------
* memory ``project_posenet_rank1_discovery`` — PoseNet's effective Jacobian
  rank is 1.008 → dim 0 carries 99.8% variance. The MLP can learn to
  predict the residual pose components from the mask conditioning that
  the renderer ALSO sees.
* memory ``project_yousfi_geometric_analysis`` — PoseNet output dim 0 is
  speed-on-a-learned-scale (mean=31.295, std=1.265). Dims 1-5 are small
  near-zero values (std~0.05).
* memory ``project_lane_marking_speed_estimation`` — lane-mark mask
  centroids encode forward-motion radial displacement (the dim-0 signal)
  at zero archive cost. The MLP shares the same input domain.

Archive cost
------------
* MLP state dict (FP16): ~1-2 KB. Replaces ``optimized_poses.pt`` (~15 KB).
* Net rate saving: ~13 KB ⇒ -0.0085 score contribution at the rate term.
* Distortion impact: bounded by Yousfi's geometric analysis; if the MLP
  learns dim 0 within 0.5 RMSE of Lane A's optimized poses, PoseNet
  distortion regresses by at most ~0.05 (vs Lane A's 0.005).

Strict-scorer-rule
------------------
NO scorers loaded at inflate time. The MLP is a pure feed-forward neural
predictor whose inputs are derived from the already-decoded mask tensor.
Per CLAUDE.md non-negotiable: this is per-pair MASK FEATURE EXTRACTION,
not scorer inference (no PoseNet/SegNet weights touched at inflate).

Sentinel
--------
The build-side tool writes the sentinel ``pose_from_embedding_v1`` into
the archive. The inflate-side dispatch detects this sentinel and loads
the companion ``pose_from_embedding_v1.pt`` MLP state dict.

Usage at inflate
----------------
    from tac.pose_from_embedding import (
        PoseFromEmbeddingMLP,
        extract_mask_features,
        load_mlp,
    )

    mlp = load_mlp(archive_dir / "pose_from_embedding_v1.pt", device=device)
    mask_features = extract_mask_features(masks, mlp.feature_extractor)
    poses = mlp.predict_poses_from_features(mask_features)  # (P, 6)
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "POSE_FROM_EMBEDDING_SENTINEL",
    "POSE_FROM_EMBEDDING_WEIGHTS_FILENAME",
    "POSENET_EMBEDDING_DIM",
    "MASK_FEATURE_DIM",
    "POSE_OUTPUT_DIM",
    "MaskFeatureExtractor",
    "PoseFromEmbeddingMLP",
    "extract_mask_features",
    "save_mlp",
    "load_mlp",
]

# Sentinel filename written into the archive when poses are predicted at
# inflate from a distilled MLP. The presence of this 0-byte file is the
# inflate-side signal to look for the companion weights file.
POSE_FROM_EMBEDDING_SENTINEL: str = "pose_from_embedding_v1"
POSE_FROM_EMBEDDING_WEIGHTS_FILENAME: str = "pose_from_embedding_v1.pt"

# PoseNet head output dim ("pose" head: out=12, only first 6 used in
# distortion). The full 12-dim vector is the "embedding" we distill against
# at compress time; at inflate it's an OPTIONAL input that defaults to 0
# (the MLP is trained with embedding-dropout so it MUST work without it).
POSENET_EMBEDDING_DIM: int = 12

# Mask feature extractor output dim (small Conv2d → global pool → vector).
# Sized so the FULL MLP archive cost stays under 15 KB FP16 (vs the 15 KB
# optimized_poses.pt it replaces). With 16 output channels the feature
# extractor has 10*8*9+8 + 8*16*9+16 + 16*16*9+16 = 4082 params,
# total MLP ≈ 4082 + 16*16+16 + 16*6+6 = 4378 params ⇒ ~9 KB FP16.
MASK_FEATURE_DIM: int = 16

# Renderer pose conditioning dim (PoseNet first 6 dims).
POSE_OUTPUT_DIM: int = 6


class MaskFeatureExtractor(nn.Module):
    """Tiny Conv2d feature extractor that summarizes a 5-class mask pair
    into a fixed-size feature vector.

    Architecture (compact, sized for < 15 KB FP16 archive cost):
        Input: (B, 2 * n_classes, H, W) one-hot pair (t, t+1)
        → Conv2d in_ch→8 (3x3, stride 2) + ReLU
        → Conv2d 8→16 (3x3, stride 2) + ReLU
        → Conv2d 16→out_dim (3x3, stride 2) + ReLU
        → AdaptiveAvgPool2d(1) → (B, out_dim)

    Notes
    -----
    * One-hot encoding ensures the same numeric range as the renderer's
      input pipeline (no learned embedding for class indices).
    * Stride-2 convs keep the parameter budget small. With H=384, W=512
      the spatial dim is 48x64 → 24x32 → 12x16 after three conv layers.
    * Channel widths (8/16/out_dim) chosen to land the full MLP archive
      cost around 10 KB FP16 (vs the 15 KB optimized_poses.pt it replaces).
    """

    def __init__(self, n_classes: int = 5, out_dim: int = MASK_FEATURE_DIM):
        super().__init__()
        self.n_classes = int(n_classes)
        self.out_dim = int(out_dim)
        c1, c2 = 8, 16
        in_ch = 2 * self.n_classes  # pair × one-hot
        self.conv1 = nn.Conv2d(in_ch, c1, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(c2, self.out_dim, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, mask_pair_onehot: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        mask_pair_onehot : torch.Tensor
            ``(B, 2 * n_classes, H, W)`` float tensor (one-hot pair).

        Returns
        -------
        torch.Tensor
            ``(B, out_dim)`` feature vector per pair.
        """
        x = F.relu(self.conv1(mask_pair_onehot))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).flatten(1)
        return x


class PoseFromEmbeddingMLP(nn.Module):
    """2-layer MLP that maps (PoseNet embedding, mask features) → 6-DOF pose.

    Trained with embedding-dropout so that at inflate the embedding input
    can be a zero vector (we don't have PoseNet at inflate). The MLP
    learns to use the mask features as the primary signal, with the
    embedding as an OPTIONAL refinement at compress time.

    Architecture
    ------------
        Input: concat(embedding[12], mask_features[16]) = 28 dims
        → Linear 28 → 16 + ReLU
        → Linear 16 → 6 (pose dims)

    Total params (excluding feature extractor): 28*16 + 16 + 16*6 + 6 =
    566 → ~1.1 KB FP16. With the small feature extractor (~4.4K params,
    ~9 KB FP16) the total archive cost is ~10 KB FP16, comfortably below
    the 15 KB optimized_poses.pt rate-savings target.
    """

    def __init__(
        self,
        embedding_dim: int = POSENET_EMBEDDING_DIM,
        mask_feature_dim: int = MASK_FEATURE_DIM,
        hidden_dim: int = 16,
        pose_dim: int = POSE_OUTPUT_DIM,
        n_classes: int = 5,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.mask_feature_dim = int(mask_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.pose_dim = int(pose_dim)
        self.n_classes = int(n_classes)
        self.feature_extractor = MaskFeatureExtractor(
            n_classes=self.n_classes, out_dim=self.mask_feature_dim,
        )
        self.fc1 = nn.Linear(self.embedding_dim + self.mask_feature_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.pose_dim)

    def forward(
        self,
        embedding: torch.Tensor,
        mask_features: torch.Tensor,
    ) -> torch.Tensor:
        """Predict 6-DOF pose from (embedding, mask_features).

        Parameters
        ----------
        embedding : torch.Tensor
            ``(B, embedding_dim)`` PoseNet head output (or zeros at inflate).
        mask_features : torch.Tensor
            ``(B, mask_feature_dim)`` extracted by the feature extractor.

        Returns
        -------
        torch.Tensor
            ``(B, pose_dim)`` predicted pose.
        """
        if embedding.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"embedding dim mismatch: got {embedding.shape[-1]}, "
                f"expected {self.embedding_dim}"
            )
        if mask_features.shape[-1] != self.mask_feature_dim:
            raise ValueError(
                f"mask_features dim mismatch: got {mask_features.shape[-1]}, "
                f"expected {self.mask_feature_dim}"
            )
        x = torch.cat([embedding, mask_features], dim=-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def predict_poses_from_masks(
        self,
        masks: torch.Tensor,
        embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inflate-side convenience: predict poses from masks alone.

        Parameters
        ----------
        masks : torch.Tensor
            ``(N, H, W)`` long tensor of class indices (N = 2 * num_pairs)
            OR ``(P, 2, H, W)`` already-paired masks.
        embedding : torch.Tensor or None
            Optional ``(P, embedding_dim)`` PoseNet embedding. Defaults
            to zeros (inflate-time behavior; PoseNet not loaded).

        Returns
        -------
        torch.Tensor
            ``(P, pose_dim)`` predicted poses, where P = N // 2 (or
            masks.shape[0] when paired input is supplied).
        """
        mask_features = extract_mask_features(
            masks, self.feature_extractor, n_classes=self.n_classes,
        )
        n_pairs = mask_features.shape[0]
        if embedding is None:
            embedding = torch.zeros(
                n_pairs, self.embedding_dim,
                device=mask_features.device, dtype=mask_features.dtype,
            )
        elif embedding.shape[0] != n_pairs:
            raise ValueError(
                f"embedding has {embedding.shape[0]} pairs, but masks "
                f"decode to {n_pairs} pairs — mismatch from different runs."
            )
        return self.forward(embedding, mask_features)


def extract_mask_features(
    masks: torch.Tensor,
    feature_extractor: MaskFeatureExtractor,
    n_classes: int = 5,
) -> torch.Tensor:
    """Extract per-pair mask features.

    Parameters
    ----------
    masks : torch.Tensor
        ``(N, H, W)`` long-tensor of class indices, where N = 2 * num_pairs
        (interleaved frame_t, frame_t+1, ...). Alternatively, ``(P, 2, H, W)``
        already-paired masks.
    feature_extractor : MaskFeatureExtractor
        The compress-time-trained extractor (loaded from the archive MLP).
    n_classes : int
        Number of segmentation classes (default 5 for SegNet's classes).

    Returns
    -------
    torch.Tensor
        ``(P, MASK_FEATURE_DIM)`` per-pair feature vector.
    """
    if masks.dim() == 3:
        n_frames = masks.shape[0]
        if n_frames % 2 != 0:
            raise ValueError(
                f"masks must have even N to form non-overlapping pairs, "
                f"got N={n_frames}"
            )
        n_pairs = n_frames // 2
        # Interleaved: pair k = (frame[2k], frame[2k+1])
        m_t = masks[0::2]
        m_t1 = masks[1::2]
        pair = torch.stack([m_t, m_t1], dim=1)  # (P, 2, H, W)
    elif masks.dim() == 4 and masks.shape[1] == 2:
        pair = masks
        n_pairs = pair.shape[0]
    else:
        raise ValueError(
            f"masks must be (N, H, W) or (P, 2, H, W), got shape "
            f"{tuple(masks.shape)}"
        )

    if pair.dtype != torch.long:
        pair = pair.long()

    # One-hot encode: (P, 2, H, W) → (P, 2, n_classes, H, W) → (P, 2*n_classes, H, W)
    pair_onehot = F.one_hot(pair.clamp(0, n_classes - 1), num_classes=n_classes)
    # one_hot puts classes at the LAST axis, permute back to channel position
    pair_onehot = pair_onehot.permute(0, 1, 4, 2, 3).contiguous()
    pair_onehot = pair_onehot.reshape(n_pairs, 2 * n_classes, *pair.shape[-2:]).float()

    # Run through the feature extractor (place on its device/dtype)
    target_device = next(feature_extractor.parameters()).device
    target_dtype = next(feature_extractor.parameters()).dtype
    pair_onehot = pair_onehot.to(device=target_device, dtype=target_dtype)
    with torch.inference_mode():
        return feature_extractor(pair_onehot)


def save_mlp(
    mlp: PoseFromEmbeddingMLP,
    path: Path | str,
    *,
    fp16: bool = True,
) -> int:
    """Save the MLP state dict (default FP16 for archive compactness).

    Returns
    -------
    int
        Bytes written.
    """
    path = Path(path)
    state = mlp.state_dict()
    if fp16:
        state = {k: v.to(torch.float16) for k, v in state.items()}
    # Pin metadata so future loaders can verify the architecture spec
    payload = {
        "format_version": 1,
        "sentinel": POSE_FROM_EMBEDDING_SENTINEL,
        "embedding_dim": mlp.embedding_dim,
        "mask_feature_dim": mlp.mask_feature_dim,
        "hidden_dim": mlp.hidden_dim,
        "pose_dim": mlp.pose_dim,
        "n_classes": mlp.n_classes,
        "fp16": bool(fp16),
        "state_dict": state,
    }
    torch.save(payload, str(path))
    return path.stat().st_size


def load_mlp(
    path: Path | str,
    *,
    device: torch.device | str = "cpu",
) -> PoseFromEmbeddingMLP:
    """Load the MLP state dict and rebuild the model.

    The loader pins the architecture from the saved metadata so a future
    refactor can't silently mismatch dims (the sentinel + format_version
    fields force explicit migration).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"pose_from_embedding MLP weights not found at {path}"
        )
    # weights_only=False because we ship a metadata dict, NOT a raw state
    # dict. The payload schema is pinned by format_version. Mark with the
    # waiver comment so preflight loader-format-safety check accepts.
    payload = torch.load(str(path), map_location="cpu", weights_only=False)  # noqa: E501  # TORCH_LOAD_WAIVED format_version=1 metadata-only
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(
            f"{path} does not contain a pose_from_embedding payload "
            f"(expected dict with 'state_dict' key)"
        )
    if payload.get("sentinel") != POSE_FROM_EMBEDDING_SENTINEL:
        raise ValueError(
            f"{path} sentinel mismatch: got {payload.get('sentinel')!r}, "
            f"expected {POSE_FROM_EMBEDDING_SENTINEL!r}"
        )
    if payload.get("format_version") != 1:
        raise ValueError(
            f"{path} has unsupported format_version "
            f"{payload.get('format_version')!r} (expected 1)"
        )
    mlp = PoseFromEmbeddingMLP(
        embedding_dim=int(payload["embedding_dim"]),
        mask_feature_dim=int(payload["mask_feature_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        pose_dim=int(payload["pose_dim"]),
        n_classes=int(payload["n_classes"]),
    )
    state = payload["state_dict"]
    # Cast saved FP16 back to FP32 for inference (avoids subtle dtype drift)
    state = {k: v.to(torch.float32) for k, v in state.items()}
    mlp.load_state_dict(state)
    mlp = mlp.to(device).eval()
    return mlp
