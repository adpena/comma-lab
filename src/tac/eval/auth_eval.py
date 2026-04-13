"""Authoritative evaluation of a renderer checkpoint.

Platform-agnostic. Works on any machine with PyTorch + upstream scorer.
All platform wrappers (Modal, Lightning, Kaggle, CLI) import this module
and call AuthEvaluator.eval_checkpoint(). No eval logic lives elsewhere.

The scoring formula, pair construction, preprocessing, and distortion
computation match upstream evaluate.py exactly:

    score = 100 * segnet_dist + sqrt(10 * posenet_dist) + 25 * rate

Where:
    - segnet_dist = mean hard-argmax disagreement on LAST frame of each pair
    - posenet_dist = mean MSE on first 6 pose outputs over consecutive pairs
    - rate = archive_size / sum(file sizes in uncompressed videos dir)

Usage::

    from tac.eval import AuthEvaluator

    evaluator = AuthEvaluator(upstream_dir=Path("upstream"))
    result = evaluator.eval_checkpoint(Path("renderer.bin"), archive_size_bytes=204800)
    print(result)
"""

from __future__ import annotations

import enum
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — must match upstream exactly
# ---------------------------------------------------------------------------

OUT_W, OUT_H = 1164, 874
SEG_W, SEG_H = 512, 384
NUM_FRAMES = 1200

# Upstream evaluate.py computes this as:
#   sum(file.stat().st_size for file in args.uncompressed_dir.rglob('*') if file.is_file())
# For the single public test video (0.mkv), this is 37,545,489 bytes.
# We hardcode it as a fallback when the videos dir is not available.
FALLBACK_UNCOMPRESSED_SIZE = 37_545_489


class RendererMode(enum.Enum):
    """Generation strategy for the renderer checkpoint."""

    ASYMMETRIC = "asymmetric"
    """AsymmetricPairGenerator: generates frame pairs via warp."""

    DP_SIMS = "dp_sims"
    """DPSIMSRenderer: generates frames independently from masks."""


@dataclass
class AuthResult:
    """Complete result from an authoritative evaluation run.

    All fields needed for provenance, comparison, and reporting.
    """

    score: float
    pose_dist: float
    seg_dist: float
    rate: float

    # Component contributions to the score
    seg_contribution: float
    pose_contribution: float
    rate_contribution: float

    # Metadata
    n_pairs: int
    archive_size_bytes: int
    uncompressed_size_bytes: int
    checkpoint_path: str
    renderer_mode: str
    device: str

    # Timing breakdown (seconds)
    timing: dict[str, float] = field(default_factory=dict)

    # Optional extras (config, git sha, etc.)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON output."""
        return {
            "score": round(self.score, 6),
            "pose_dist": self.pose_dist,
            "seg_dist": self.seg_dist,
            "rate": self.rate,
            "seg_contribution": round(self.seg_contribution, 6),
            "pose_contribution": round(self.pose_contribution, 6),
            "rate_contribution": round(self.rate_contribution, 6),
            "n_pairs": self.n_pairs,
            "archive_size_bytes": self.archive_size_bytes,
            "uncompressed_size_bytes": self.uncompressed_size_bytes,
            "checkpoint_path": self.checkpoint_path,
            "renderer_mode": self.renderer_mode,
            "device": self.device,
            "timing": {k: round(v, 2) for k, v in self.timing.items()},
            "provenance": self.provenance,
        }

    def summary(self) -> str:
        """Human-readable one-line summary."""
        return (
            f"score={self.score:.4f}  "
            f"(seg={self.seg_contribution:.4f} + "
            f"pose={self.pose_contribution:.4f} + "
            f"rate={self.rate_contribution:.4f})  "
            f"n_pairs={self.n_pairs}  "
            f"mode={self.renderer_mode}"
        )


# ---------------------------------------------------------------------------
# AuthEvaluator
# ---------------------------------------------------------------------------


class AuthEvaluator:
    """Authoritative evaluation of a renderer checkpoint.

    Platform-agnostic. Handles both asymmetric (pair-wise) and dp_sims
    (independent) generation, both .pt and .bin checkpoint formats.

    Parameters
    ----------
    upstream_dir : Path
        Root of the upstream repo (contains modules.py, models/, videos/).
    device : str
        PyTorch device string. Defaults to auto-detection.
    """

    def __init__(
        self,
        upstream_dir: Path,
        device: str | None = None,
    ) -> None:
        self.upstream_dir = Path(upstream_dir).resolve()
        self.device = device or self._detect_device()

        # Lazily loaded
        self._posenet: nn.Module | None = None
        self._segnet: nn.Module | None = None

    # ------------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ------------------------------------------------------------------
    # Scorer loading
    # ------------------------------------------------------------------

    def load_scorers(self) -> None:
        """Load frozen PoseNet + SegNet from upstream.

        Idempotent — calling multiple times is safe.
        """
        if self._posenet is not None and self._segnet is not None:
            return

        t0 = time.monotonic()

        # Ensure upstream modules are importable
        upstream_str = str(self.upstream_dir)
        inserted = False
        if upstream_str not in sys.path:
            sys.path.insert(0, upstream_str)
            inserted = True

        try:
            from modules import PoseNet, SegNet
            from safetensors.torch import load_file

            models_dir = self.upstream_dir / "models"

            posenet = PoseNet().eval().to(self.device)
            segnet = SegNet().eval().to(self.device)

            posenet.load_state_dict(
                load_file(str(models_dir / "posenet.safetensors"), device=self.device)
            )
            segnet.load_state_dict(
                load_file(str(models_dir / "segnet.safetensors"), device=self.device)
            )

            for p in posenet.parameters():
                p.requires_grad = False
            for p in segnet.parameters():
                p.requires_grad = False

            self._posenet = posenet
            self._segnet = segnet

        finally:
            if inserted:
                try:
                    sys.path.remove(upstream_str)
                except ValueError:
                    pass

        elapsed = time.monotonic() - t0
        logger.info("Scorers loaded in %.1fs on %s", elapsed, self.device)

    @property
    def posenet(self) -> nn.Module:
        if self._posenet is None:
            raise RuntimeError("Call load_scorers() before accessing posenet")
        return self._posenet

    @property
    def segnet(self) -> nn.Module:
        if self._segnet is None:
            raise RuntimeError("Call load_scorers() before accessing segnet")
        return self._segnet

    # ------------------------------------------------------------------
    # GT video decoding
    # ------------------------------------------------------------------

    def decode_gt_video(
        self,
        video_name: str = "0.mkv",
    ) -> list[np.ndarray]:
        """Decode GT video via PyAV with BT.601 limited-range YUV->RGB.

        Returns list of (H, W, 3) uint8 ndarrays matching the scorer decode.
        """
        import av

        video_path = self.upstream_dir / "videos" / video_name
        if not video_path.exists():
            raise FileNotFoundError(f"GT video not found: {video_path}")

        t0 = time.monotonic()
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        frames: list[np.ndarray] = []

        for frame in container.decode(stream):
            rgb = _yuv420_to_rgb(frame)
            frames.append(rgb.numpy())

        container.close()
        elapsed = time.monotonic() - t0
        logger.info("Decoded %d GT frames from %s (%.1fs)", len(frames), video_path, elapsed)
        return frames

    # ------------------------------------------------------------------
    # Mask extraction
    # ------------------------------------------------------------------

    def extract_masks(
        self,
        gt_frames: list[np.ndarray],
        batch_size: int = 8,
    ) -> torch.Tensor:
        """Extract SegNet semantic masks from GT frames.

        Args:
            gt_frames: list of (H, W, 3) uint8 ndarrays
            batch_size: inference batch size

        Returns:
            (N, 384, 512) int8 tensor of class indices in [0, 4]
        """
        self.load_scorers()
        t0 = time.monotonic()
        N = len(gt_frames)
        masks_list: list[torch.Tensor] = []

        with torch.inference_mode():
            for i in range(0, N, batch_size):
                end = min(i + batch_size, N)
                batch_np = np.stack(gt_frames[i:end], axis=0)  # (B, H, W, 3)
                batch_t = (
                    torch.from_numpy(batch_np)
                    .float()
                    .permute(0, 3, 1, 2)
                    .to(self.device)
                )
                # SegNet.preprocess_input expects (B, T, C, H, W) — T=1 for mask extraction
                inp = batch_t.unsqueeze(1)  # (B, 1, C, H, W)
                seg_in = self.segnet.preprocess_input(inp)
                logits = self.segnet(seg_in)  # (B, 5, 384, 512)
                mask = logits.argmax(dim=1)  # (B, 384, 512)
                masks_list.append(mask.to(torch.int8).cpu())

        masks = torch.cat(masks_list, dim=0)
        elapsed = time.monotonic() - t0
        logger.info("Extracted %d masks (%.1fs)", masks.shape[0], elapsed)
        return masks

    # ------------------------------------------------------------------
    # Renderer loading
    # ------------------------------------------------------------------

    def load_renderer(self, checkpoint_path: Path) -> tuple[nn.Module, RendererMode]:
        """Load renderer from .pt or .bin checkpoint.

        Auto-detects format (ASYM binary, DPSM binary, PyTorch pickle)
        and renderer mode (asymmetric vs dp_sims).

        Returns:
            (model, mode) tuple
        """
        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        t0 = time.monotonic()
        raw_bytes = checkpoint_path.read_bytes()
        magic = raw_bytes[:4]

        if magic == b"ASYM":
            model = self._load_asym_bin(raw_bytes)
            mode = RendererMode.ASYMMETRIC
        elif magic == b"DPSM":
            model = self._load_dpsm_bin(raw_bytes)
            mode = RendererMode.DP_SIMS
        else:
            model, mode = self._load_pt_checkpoint(checkpoint_path)

        model = model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad = False

        n_params = sum(p.numel() for p in model.parameters())
        elapsed = time.monotonic() - t0
        logger.info(
            "Loaded renderer: %s mode, %s params, %.1fs",
            mode.value,
            f"{n_params:,}",
            elapsed,
        )
        return model, mode

    def _load_asym_bin(self, raw_bytes: bytes) -> nn.Module:
        """Load AsymmetricPairGenerator from ASYM binary format."""
        try:
            from tac.renderer_export import load_asymmetric_checkpoint

            return load_asymmetric_checkpoint(raw_bytes, device=self.device)
        except ImportError:
            # Inline fallback for environments without full tac
            return _inline_load_asym(raw_bytes, device=self.device)

    def _load_dpsm_bin(self, raw_bytes: bytes) -> nn.Module:
        """Load DPSIMSRenderer from DPSM binary format."""
        try:
            from tac.renderer_export import load_renderer_checkpoint

            return load_renderer_checkpoint(raw_bytes, device=self.device)
        except ImportError:
            raise RuntimeError(
                "DPSM .bin format requires tac.renderer_export. "
                "Install the tac package or use a .pt checkpoint."
            )

    def _load_pt_checkpoint(self, path: Path) -> tuple[nn.Module, RendererMode]:
        """Load renderer from a PyTorch pickle checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        config = ckpt.get("config", {})

        # Detect if this is an AsymmetricPairGenerator checkpoint
        raw_sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        has_motion = any(k.startswith("motion.") for k in raw_sd.keys())

        if has_motion:
            model = self._build_asymmetric_from_config(config)
            model.load_state_dict(raw_sd, strict=True)
            return model, RendererMode.ASYMMETRIC

        # DPSIMSRenderer — check for renderer. prefix (from PairGenerator wrapper)
        has_prefix = any(k.startswith("renderer.") for k in raw_sd.keys())
        if has_prefix:
            sd = {
                k[len("renderer."):]: v
                for k, v in raw_sd.items()
                if k.startswith("renderer.")
            }
        else:
            sd = raw_sd

        model = self._build_dpsims_from_config(config)
        model.load_state_dict(sd, strict=True)
        return model, RendererMode.DP_SIMS

    def _build_dpsims_from_config(self, config: dict) -> nn.Module:
        """Build DPSIMSRenderer from a config dict."""
        try:
            from tac.dp_sims_renderer import DPSIMSRenderer
        except ImportError:
            raise RuntimeError(
                "DPSIMSRenderer loading from .pt requires tac.dp_sims_renderer"
            )

        channels = config.get("channels", (256, 128, 64, 32))
        if isinstance(channels, list):
            channels = tuple(channels)

        return DPSIMSRenderer(
            num_classes=config.get("num_classes", 5),
            channels=channels,
            init_h=config.get("init_h", 24),
            init_w=config.get("init_w", 32),
            spade_hidden=config.get("spade_hidden", 64),
            noise_dim=config.get("noise_dim", 16),
            use_noise=config.get("use_noise", True),
        )

    def _build_asymmetric_from_config(self, config: dict) -> nn.Module:
        """Build AsymmetricPairGenerator from a config dict."""
        try:
            from tac.renderer import AsymmetricPairGenerator
        except ImportError:
            raise RuntimeError(
                "AsymmetricPairGenerator loading from .pt requires tac.renderer"
            )

        return AsymmetricPairGenerator(
            num_classes=config.get("num_classes", 5),
            embed_dim=config.get("embed_dim", 6),
            base_ch=config.get("base_ch", 36),
            mid_ch=config.get("mid_ch", 60),
            motion_hidden=config.get("motion_hidden", 32),
            depth=config.get("renderer_depth", config.get("depth", 1)),
            max_flow_px=config.get("max_flow_px", 20.0),
            max_residual=config.get("max_residual", 20.0),
            flow_only=config.get("flow_only", False),
        )

    # ------------------------------------------------------------------
    # Frame generation
    # ------------------------------------------------------------------

    def generate_frames(
        self,
        renderer: nn.Module,
        mode: RendererMode,
        masks: torch.Tensor,
        batch_size: int = 4,
    ) -> torch.Tensor:
        """Generate all frames from masks via the renderer.

        For dp_sims: generates each frame independently.
        For asymmetric: generates consecutive pairs via warp.

        Both paths upscale from SegNet resolution (384x512) to output
        resolution (874x1164) with bilinear interpolation.

        Args:
            renderer: loaded renderer model
            mode: ASYMMETRIC or DP_SIMS
            masks: (N, 384, 512) int8/long tensor
            batch_size: inference batch size

        Returns:
            (N, 3, 874, 1164) uint8 tensor of generated frames
        """
        t0 = time.monotonic()
        N = masks.shape[0]
        all_frames: list[torch.Tensor] = []

        # Deterministic seed for reproducible output (noise injectors)
        torch.manual_seed(42)

        with torch.inference_mode():
            if mode == RendererMode.DP_SIMS:
                all_frames = self._generate_dp_sims(renderer, masks, batch_size)
            elif mode == RendererMode.ASYMMETRIC:
                all_frames = self._generate_asymmetric(renderer, masks, batch_size)
            else:
                raise ValueError(f"Unknown renderer mode: {mode}")

        result = torch.cat(all_frames, dim=0)  # (N, 3, OUT_H, OUT_W) uint8
        elapsed = time.monotonic() - t0
        logger.info("Generated %d frames in %.1fs", result.shape[0], elapsed)
        return result

    def _generate_dp_sims(
        self,
        renderer: nn.Module,
        masks: torch.Tensor,
        batch_size: int,
    ) -> list[torch.Tensor]:
        """Generate frames independently via DPSIMSRenderer."""
        # Extract the actual renderer if wrapped in a PairGenerator
        renderer_module = getattr(renderer, "renderer", renderer)
        N = masks.shape[0]
        chunks: list[torch.Tensor] = []

        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            batch_masks = masks[i:end].to(device=self.device, dtype=torch.long)
            gen = renderer_module(batch_masks)  # (B, 3, 384, 512)
            gen_up = F.interpolate(
                gen, size=(OUT_H, OUT_W), mode="bilinear", align_corners=False
            )
            chunks.append(gen_up.round().clamp(0, 255).to(torch.uint8).cpu())

        return chunks

    def _generate_asymmetric(
        self,
        renderer: nn.Module,
        masks: torch.Tensor,
        batch_size: int,
    ) -> list[torch.Tensor]:
        """Generate frames via AsymmetricPairGenerator (consecutive pairs)."""
        N = masks.shape[0]
        # Pre-allocate frame buffer
        frame_buffer: list[torch.Tensor | None] = [None] * N
        pair_idx = 0

        while pair_idx < N - 1:
            batch_t_list: list[torch.Tensor] = []
            batch_t1_list: list[torch.Tensor] = []
            indices: list[int] = []

            batch_end = min(pair_idx + batch_size * 2, N - 1)
            for j in range(pair_idx, batch_end, 2):
                if j + 1 < N:
                    batch_t_list.append(masks[j])
                    batch_t1_list.append(masks[j + 1])
                    indices.append(j)

            if not batch_t_list:
                break

            masks_t = torch.stack(batch_t_list).to(device=self.device, dtype=torch.long)
            masks_t1 = torch.stack(batch_t1_list).to(device=self.device, dtype=torch.long)

            pairs = renderer(masks_t, masks_t1)  # (B, 2, H, W, 3) HWC

            for b, idx in enumerate(indices):
                for frame_pos in range(2):
                    frame_hwc = pairs[b, frame_pos]  # (H, W, 3)
                    frame_chw = frame_hwc.permute(2, 0, 1).unsqueeze(0)
                    frame_up = F.interpolate(
                        frame_chw, size=(OUT_H, OUT_W),
                        mode="bilinear", align_corners=False,
                    )
                    frame_buffer[idx + frame_pos] = (
                        frame_up.round().clamp(0, 255).to(torch.uint8).cpu()
                    )

            pair_idx += len(batch_t_list) * 2

        # Handle trailing odd frame
        if N % 2 != 0 and frame_buffer[N - 1] is None:
            last_mask = masks[N - 1:N].to(device=self.device, dtype=torch.long)
            renderer_sub = getattr(renderer, "renderer", renderer)
            frame = renderer_sub(last_mask)  # (1, 3, H, W)
            frame_up = F.interpolate(
                frame, size=(OUT_H, OUT_W), mode="bilinear", align_corners=False
            )
            frame_buffer[N - 1] = frame_up.round().clamp(0, 255).to(torch.uint8).cpu()

        # Validate no None entries
        missing = [i for i, f in enumerate(frame_buffer) if f is None]
        if missing:
            raise RuntimeError(f"Frame generation left gaps at indices: {missing[:10]}")

        return [f for f in frame_buffer if f is not None]

    # ------------------------------------------------------------------
    # Scoring — matches upstream evaluate.py exactly
    # ------------------------------------------------------------------

    def compute_score(
        self,
        generated_frames: torch.Tensor,
        gt_frames: list[np.ndarray],
        archive_size_bytes: int,
        batch_size: int = 16,
    ) -> AuthResult:
        """Compute the authoritative score.

        **Canonical scoring path.** This method is the single source of truth
        for pair construction, preprocessing, and distortion computation.
        The Modal ``DistortionNet`` path in ``modal_asymmetric_warp_deploy.py``
        uses ``DistortionNet.compute_distortion`` which builds pairs internally
        via the upstream ``seq_len=2`` batching. That path is a convenience
        wrapper; if the two ever disagree, THIS method governs because it
        replicates upstream ``evaluate.py`` pair construction explicitly.

        Scoring pipeline matches upstream evaluate.py:
        - Pairs are consecutive frames: (frame[i], frame[i+1])
        - PoseNet: MSE on first 6 pose outputs over both frames in pair
        - SegNet: hard-argmax disagreement on LAST frame of each pair
        - Rate: archive_size / uncompressed_size
        - Score: 100 * seg_dist + sqrt(10 * pose_dist) + 25 * rate

        Args:
            generated_frames: (N, 3, H, W) uint8 tensor
            gt_frames: list of (H, W, 3) uint8 ndarrays
            archive_size_bytes: size of archive.zip in bytes
            batch_size: scorer batch size (upstream default: 16)

        Returns:
            AuthResult with full score breakdown

        Raises:
            AssertionError: if generated and GT frame counts differ, which
                would cause the two scoring paths to diverge silently.
        """
        self.load_scorers()
        t0 = time.monotonic()

        N = generated_frames.shape[0]
        assert N == len(gt_frames), (
            f"Frame count mismatch: {N} generated vs {len(gt_frames)} GT"
        )

        # Compute uncompressed size
        uncompressed_size = self._compute_uncompressed_size()
        rate = archive_size_bytes / uncompressed_size

        # Convert GT to tensor: (N, 3, H, W) float
        gt_chw = torch.stack([
            torch.from_numpy(f).float().permute(2, 0, 1)
            for f in gt_frames
        ])  # (N, 3, H, W)

        gen_float = generated_frames.float()  # (N, 3, H, W)

        # Accumulate distortions over consecutive pairs
        total_pose = torch.zeros([], device=torch.device(self.device))
        total_seg = torch.zeros([], device=torch.device(self.device))
        n_pairs = 0

        with torch.inference_mode():
            # Process pairs in batches, matching upstream evaluate.py
            pair_indices = list(range(N - 1))

            for batch_start in range(0, len(pair_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(pair_indices))
                batch_idx = pair_indices[batch_start:batch_end]
                B = len(batch_idx)

                # Build (B, 2, C, H, W) pair tensors for PoseNet
                gen_t = gen_float[batch_idx].to(self.device)  # (B, 3, H, W)
                gen_t1 = gen_float[[i + 1 for i in batch_idx]].to(self.device)
                gt_t = gt_chw[batch_idx].to(self.device)
                gt_t1 = gt_chw[[i + 1 for i in batch_idx]].to(self.device)

                # --- PoseNet: MSE on first 6 outputs over the pair ---
                gen_pair = torch.stack([gen_t, gen_t1], dim=1)  # (B, 2, C, H, W)
                gt_pair = torch.stack([gt_t, gt_t1], dim=1)

                pose_in_gen = self.posenet.preprocess_input(gen_pair)
                pose_in_gt = self.posenet.preprocess_input(gt_pair)
                pose_out_gen = self.posenet(pose_in_gen)
                pose_out_gt = self.posenet(pose_in_gt)

                # Upstream: (out1[h.name][..., :h.out // 2] - out2[...]).pow(2).mean(...)
                # Head('pose', 32, 12) => out // 2 = 6
                p_gen = pose_out_gen["pose"][..., :6]
                p_gt = pose_out_gt["pose"][..., :6]
                pose_per_sample = (p_gen - p_gt).pow(2).mean(
                    dim=tuple(range(1, p_gen.ndim))
                )

                # --- SegNet: hard-argmax disagreement on LAST frame ---
                # Upstream: SegNet.preprocess_input uses x[:, -1, ...]
                gen_seg_in = gen_t1.unsqueeze(1)  # (B, 1, C, H, W) — last frame
                gt_seg_in = gt_t1.unsqueeze(1)

                seg_in_gen = self.segnet.preprocess_input(gen_seg_in)
                seg_in_gt = self.segnet.preprocess_input(gt_seg_in)
                seg_out_gen = self.segnet(seg_in_gen)
                seg_out_gt = self.segnet(seg_in_gt)

                diff = (seg_out_gen.argmax(dim=1) != seg_out_gt.argmax(dim=1)).float()
                seg_per_sample = diff.mean(dim=tuple(range(1, diff.ndim)))

                total_pose += pose_per_sample.sum()
                total_seg += seg_per_sample.sum()
                n_pairs += B

        avg_pose = (total_pose / n_pairs).item()
        avg_seg = (total_seg / n_pairs).item()

        # Score formula: exactly upstream evaluate.py line 92
        score = 100.0 * avg_seg + math.sqrt(10.0 * avg_pose) + 25.0 * rate

        elapsed = time.monotonic() - t0
        logger.info(
            "Scoring complete: score=%.4f (seg=%.6f, pose=%.6f, rate=%.8f) in %.1fs",
            score, avg_seg, avg_pose, rate, elapsed,
        )

        return AuthResult(
            score=score,
            pose_dist=avg_pose,
            seg_dist=avg_seg,
            rate=rate,
            seg_contribution=100.0 * avg_seg,
            pose_contribution=math.sqrt(10.0 * avg_pose),
            rate_contribution=25.0 * rate,
            n_pairs=n_pairs,
            archive_size_bytes=archive_size_bytes,
            uncompressed_size_bytes=uncompressed_size,
            checkpoint_path="",  # filled in by eval_checkpoint
            renderer_mode="",
            device=self.device,
        )

    def _compute_uncompressed_size(self) -> int:
        """Compute uncompressed video size matching upstream evaluate.py.

        Upstream: sum(file.stat().st_size for file in videos_dir.rglob('*') if file.is_file())
        Falls back to hardcoded value if videos dir is not available.
        """
        videos_dir = self.upstream_dir / "videos"
        if videos_dir.exists():
            total = sum(
                f.stat().st_size
                for f in videos_dir.rglob("*")
                if f.is_file()
            )
            if total > 0:
                return total

        logger.warning(
            "Videos dir not found at %s, using fallback uncompressed size %d",
            videos_dir,
            FALLBACK_UNCOMPRESSED_SIZE,
        )
        return FALLBACK_UNCOMPRESSED_SIZE

    # ------------------------------------------------------------------
    # Full eval pipeline
    # ------------------------------------------------------------------

    def eval_checkpoint(
        self,
        checkpoint_path: Path,
        archive_size_bytes: int | None = None,
        batch_size_masks: int = 8,
        batch_size_gen: int = 4,
        batch_size_score: int = 16,
        video_name: str = "0.mkv",
        provenance: dict[str, Any] | None = None,
    ) -> AuthResult:
        """Full authoritative evaluation pipeline.

        Steps:
            1. Load scorers (PoseNet + SegNet)
            2. Decode GT video via PyAV
            3. Extract semantic masks via SegNet
            4. Load renderer checkpoint (auto-detects format + mode)
            5. Generate all frames
            6. Score generated vs GT frames
            7. Return comprehensive result

        Args:
            checkpoint_path: path to .pt or .bin renderer checkpoint
            archive_size_bytes: archive.zip size. If None, uses checkpoint file size.
            batch_size_masks: batch size for mask extraction
            batch_size_gen: batch size for frame generation
            batch_size_score: batch size for scoring (upstream default: 16)
            video_name: GT video filename (default: "0.mkv")
            provenance: optional metadata dict (git sha, experiment tag, etc.)

        Returns:
            AuthResult with full score breakdown, timing, and provenance
        """
        checkpoint_path = Path(checkpoint_path).resolve()
        timing: dict[str, float] = {}
        t_total = time.monotonic()

        # 1. Load scorers
        t0 = time.monotonic()
        self.load_scorers()
        timing["load_scorers_s"] = time.monotonic() - t0

        # 2. Decode GT video
        t0 = time.monotonic()
        gt_frames = self.decode_gt_video(video_name)
        timing["decode_gt_s"] = time.monotonic() - t0

        # 3. Extract masks
        t0 = time.monotonic()
        masks = self.extract_masks(gt_frames, batch_size=batch_size_masks)
        timing["extract_masks_s"] = time.monotonic() - t0

        # 4. Load renderer
        t0 = time.monotonic()
        renderer, mode = self.load_renderer(checkpoint_path)
        timing["load_renderer_s"] = time.monotonic() - t0

        # 5. Generate frames
        t0 = time.monotonic()
        generated = self.generate_frames(
            renderer, mode, masks, batch_size=batch_size_gen
        )
        timing["generate_frames_s"] = time.monotonic() - t0

        # 6. Determine archive size
        if archive_size_bytes is None:
            archive_size_bytes = checkpoint_path.stat().st_size
            logger.info(
                "No archive_size_bytes provided, using checkpoint size: %d bytes",
                archive_size_bytes,
            )

        # 7. Score
        t0 = time.monotonic()
        result = self.compute_score(
            generated, gt_frames, archive_size_bytes,
            batch_size=batch_size_score,
        )
        timing["scoring_s"] = time.monotonic() - t0

        timing["total_s"] = time.monotonic() - t_total

        # Enrich result with metadata
        result.timing = timing
        result.checkpoint_path = str(checkpoint_path)
        result.renderer_mode = mode.value
        result.provenance = provenance or {}

        logger.info("Full eval complete: %s", result.summary())
        return result

    # ------------------------------------------------------------------
    # Results persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_results(result: AuthResult, output_path: Path) -> None:
        """Save results as JSON with full provenance."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
        logger.info("Results saved to %s", output_path)


# ---------------------------------------------------------------------------
# YUV->RGB decode — matches upstream frame_utils.py and tac.data exactly
# ---------------------------------------------------------------------------


def _yuv420_to_rgb(frame) -> torch.Tensor:
    """BT.601 limited-range YUV420->RGB. Returns (H, W, 3) uint8 tensor.

    Identical to tac.data.yuv420_to_rgb and inflate_renderer.yuv420_to_rgb.
    Inlined here so auth_eval has zero dependency on the rest of tac.data.
    """
    H, W = frame.height, frame.width
    y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(
        H, frame.planes[0].line_size
    )[:, :W]
    u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(
        H // 2, frame.planes[1].line_size
    )[:, : W // 2]
    v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(
        H // 2, frame.planes[2].line_size
    )[:, : W // 2]

    y_t = torch.from_numpy(y.copy()).float()
    u_t = torch.from_numpy(u.copy()).float().unsqueeze(0).unsqueeze(0)
    v_t = torch.from_numpy(v.copy()).float().unsqueeze(0).unsqueeze(0)
    u_up = F.interpolate(u_t, size=(H, W), mode="bilinear", align_corners=False).squeeze()
    v_up = F.interpolate(v_t, size=(H, W), mode="bilinear", align_corners=False).squeeze()

    yf = (y_t - 16.0) * (255.0 / 219.0)
    uf = (u_up - 128.0) * (255.0 / 224.0)
    vf = (v_up - 128.0) * (255.0 / 224.0)

    r = (yf + 1.402 * vf).clamp(0, 255)
    g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255)
    b = (yf + 1.772 * uf).clamp(0, 255)
    return torch.stack([r, g, b], dim=-1).round().to(torch.uint8)


# ---------------------------------------------------------------------------
# Inline ASYM .bin loader — standalone on scorer machines without full tac
# ---------------------------------------------------------------------------


def _inline_load_asym(raw_bytes: bytes, device: str = "cpu") -> nn.Module:
    """Inline ASYM .bin deserializer — no tac dependency beyond this file.

    Reads ASYM header -> parses JSON config -> reconstructs
    AsymmetricPairGenerator -> loads quantized weights from blobs.

    This is a copy of inflate_renderer._inline_load_asym, kept here so
    auth_eval can work on machines that lack the inflate_renderer module.
    """
    import struct

    offset = 0
    if raw_bytes[offset:offset + 4] != b"ASYM":
        raise ValueError(f"Not an ASYM binary (got {raw_bytes[:4]!r})")
    offset += 4

    header_len = struct.unpack("<I", raw_bytes[offset:offset + 4])[0]
    offset += 4
    header = json.loads(raw_bytes[offset:offset + header_len].decode("utf-8"))
    offset += header_len

    version = header.get("version", 0)
    if version != 2:
        raise ValueError(f"Unsupported ASYM export version {version} (expected 2)")

    # Import architecture classes — try tac first, fall back to inline
    try:
        from tac.renderer import AsymmetricPairGenerator
    except ImportError:
        raise RuntimeError(
            "ASYM .bin loading requires tac.renderer.AsymmetricPairGenerator. "
            "For fully standalone usage, use inflate_renderer.py instead."
        )

    model = AsymmetricPairGenerator(
        num_classes=header.get("num_classes", 5),
        embed_dim=header.get("embed_dim", 6),
        base_ch=header.get("base_ch", 36),
        mid_ch=header.get("mid_ch", 60),
        motion_hidden=header.get("motion_hidden", 32),
        depth=header.get("depth", 1),
        max_flow_px=header.get("max_flow_px", 20.0),
        max_residual=header.get("max_residual", 20.0),
        flow_only=header.get("flow_only", False),
    )

    # Build name -> module lookups
    embedding_lookup: dict[str, nn.Module] = {}
    conv_lookup: dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            embedding_lookup[name] = module
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            conv_lookup[name] = module

    # Restore weights from header-ordered layers
    for layer_meta in header["layers"]:
        name = layer_meta["name"]
        is_embedding = layer_meta.get("is_embedding", False)

        blob_len = struct.unpack("<I", raw_bytes[offset:offset + 4])[0]
        offset += 4
        weight_data = raw_bytes[offset:offset + blob_len]
        offset += blob_len

        if is_embedding:
            shape = layer_meta["shape"]
            bits = layer_meta["bits"]
            count = 1
            for s in shape:
                count *= s
            w_offset = 0
            scale = struct.unpack("<e", weight_data[w_offset:w_offset + 2])[0]
            w_offset += 2
            values, w_offset = _unpack_values(weight_data, w_offset, count, bits)
            emb_tensor = _dequantize_values(values, bits, scale).reshape(shape)
            with torch.no_grad():
                embedding_lookup[name].weight.copy_(emb_tensor)
            continue

        has_bias = layer_meta["has_bias"]
        bias_blob_len = struct.unpack("<I", raw_bytes[offset:offset + 4])[0]
        offset += 4
        bias_data = raw_bytes[offset:offset + bias_blob_len]
        offset += bias_blob_len

        module = conv_lookup[name]
        shape = layer_meta["shape"]
        transposed = layer_meta.get("transposed", False)
        bits = layer_meta["bits"]

        if transposed:
            C_out = shape[1]
            fan_in = shape[0] * shape[2] * shape[3]
            ch_shape = [shape[0]] + shape[2:]
        else:
            C_out = shape[0]
            fan_in = 1
            for s in shape[1:]:
                fan_in *= s
            ch_shape = shape[1:]

        with torch.no_grad():
            module.weight.zero_()
            if module.bias is not None:
                module.bias.zero_()

            w_offset = 0
            for ch_idx in range(C_out):
                scale = struct.unpack("<e", weight_data[w_offset:w_offset + 2])[0]
                w_offset += 2
                values, w_offset = _unpack_values(weight_data, w_offset, fan_in, bits)
                dequant = _dequantize_values(values, bits, scale)
                if transposed:
                    module.weight[:, ch_idx] = dequant.reshape(ch_shape)
                else:
                    module.weight[ch_idx] = dequant.reshape(ch_shape)

            if has_bias and bias_data:
                b_offset = 0
                for ch_idx in range(C_out):
                    scale_b = struct.unpack("<e", bias_data[b_offset:b_offset + 2])[0]
                    b_offset += 2
                    u_val = struct.unpack("<H", bias_data[b_offset:b_offset + 2])[0]
                    b_offset += 2
                    n_levels = 2 ** bits
                    half = n_levels // 2
                    q = u_val - half
                    module.bias[ch_idx] = q / max(half - 1, 1) * scale_b

    # Restore scalar parameters
    scalar_params = header.get("scalar_params", {})
    if scalar_params:
        param_dict = dict(model.named_parameters())
        with torch.no_grad():
            for pname, pval in scalar_params.items():
                if pname in param_dict:
                    param_dict[pname].fill_(pval)

    if offset != len(raw_bytes):
        raise ValueError(
            f"Trailing data: {len(raw_bytes) - offset} bytes unread (expected 0)"
        )

    model = model.to(device)
    model.eval()
    return model


def _unpack_values(
    data: bytes, offset: int, count: int, bits: int
) -> tuple[list[int], int]:
    """Unpack `count` values at `bits` per value from data starting at offset."""
    if bits == 8:
        values = [data[offset + i] for i in range(count)]
        return values, offset + count
    total_bits = count * bits
    total_bytes = (total_bits + 7) // 8
    if count > 10_000_000:
        raise ValueError(f"Implausible value count={count:,} — possible malformed .bin")
    raw = data[offset:offset + total_bytes]
    bit_buffer = int.from_bytes(bytes(raw), byteorder="little")
    mask = (1 << bits) - 1
    values = []
    for _ in range(count):
        values.append(bit_buffer & mask)
        bit_buffer >>= bits
    return values, offset + total_bytes


def _dequantize_values(
    values: list[int], bits: int, scale: float
) -> torch.Tensor:
    """Dequantize unsigned integer values back to float tensor."""
    bits = max(bits, 2)
    n_levels = 2 ** bits
    half = n_levels // 2
    return torch.tensor(
        [(v - half) / max(half - 1, 1) * scale for v in values],
        dtype=torch.float32,
    )
