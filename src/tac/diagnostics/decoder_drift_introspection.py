# SPDX-License-Identifier: MIT
"""DALI vs PyAV decoder-drift introspection.

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" and "MPS auth eval is
NOISE": this module quantifies the byte-level RGB divergence between DALI's
NVDEC-based decode (CUDA-only) and PyAV's libav-based decode (CPU/MPS), which
is the THIRD axis of CUDA-vs-CPU drift NOT covered by network-internals
introspection (af945f502) or FastViT precision-compounding theory (a22d581a).

Key insight: ``upstream/evaluate.py`` lines 31-42 select
``DefaultDatasetClass = DaliVideoDataset`` when ``device.type == 'cuda'`` and
``AVVideoDataset`` otherwise. The ground-truth tensor batch_gt is decoded
DIFFERENTLY between paths. PoseNet/SegNet then compute distortion against a
DIFFERENT GT tensor on each device — score drift can come from this BEFORE
any network kernel difference.

The upstream author EXPLICITLY designed AVVideoDataset to match NVDEC output
(see frame_utils.py:161 "matches nvdec output", :201 "matches nvdec"), but
exact byte-identity is NOT guaranteed: NVDEC uses fixed-point chroma
upsampling and YUV->RGB matrix; libav (and our F.interpolate(mode='bilinear')
+ float32 multiplications + .round().to(uint8)) uses different rounding.

Public API
----------
- :class:`DecoderDriftIntrospector`: runs the AV decode path locally, designs
  the DALI decode (does NOT run it; CUDA-only), and ingests pre-dumped DALI
  output.
- :class:`FrameByteFingerprint`: per-frame statistical fingerprint.
- :func:`quantify_drift`: compute L1/L2/max-abs/per-channel/histogram drift.
- :func:`lipschitz_pose_drift_prediction`: back-of-envelope mapping from
  per-pixel RGB drift to expected pose-component drift via a Lipschitz
  estimate.

All numerical outputs are tagged ``[diagnostic-not-score]`` per CLAUDE.md.
Local AV outputs are additionally tagged ``[macOS-CPU advisory only]`` per
the "MPS auth eval is NOISE" rule (the contest authoritative AV decode is
ubuntu-latest's ffmpeg + the pinned ``upstream/ffmpeg-new`` binary).
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# Repo-relative path to upstream so we can import frame_utils.yuv420_to_rgb
# (the EXACT helper used by AVVideoDataset; reusing it guarantees behavioral
# identity with the contest path).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_UPSTREAM = _REPO_ROOT / "upstream"


def _load_yuv420_to_rgb():
    """Lazily import upstream.frame_utils.yuv420_to_rgb without polluting
    sys.path globally."""
    if str(_UPSTREAM) not in sys.path:
        sys.path.insert(0, str(_UPSTREAM))
    from frame_utils import yuv420_to_rgb  # type: ignore[import-not-found]

    return yuv420_to_rgb


@dataclass
class FrameByteFingerprint:
    """Per-frame statistical fingerprint of a decoded RGB tensor."""

    frame_index: int
    height: int
    width: int
    rgb_sha256: str
    per_channel_mean: tuple[float, float, float]
    per_channel_std: tuple[float, float, float]
    pixel_count: int
    advisory_tag: str = "[diagnostic-not-score]"

    @classmethod
    def from_tensor(cls, frame_index: int, rgb: torch.Tensor) -> "FrameByteFingerprint":
        """Compute fingerprint of a single (H, W, 3) uint8 RGB tensor."""
        if rgb.dtype != torch.uint8:
            raise ValueError(f"expected uint8, got {rgb.dtype}")
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"expected (H, W, 3), got {tuple(rgb.shape)}")
        H, W, _ = rgb.shape
        rgb_bytes = rgb.contiguous().cpu().numpy().tobytes()
        return cls(
            frame_index=frame_index,
            height=H,
            width=W,
            rgb_sha256=hashlib.sha256(rgb_bytes).hexdigest()[:32],
            per_channel_mean=(
                float(rgb[..., 0].float().mean()),
                float(rgb[..., 1].float().mean()),
                float(rgb[..., 2].float().mean()),
            ),
            per_channel_std=(
                float(rgb[..., 0].float().std()),
                float(rgb[..., 1].float().std()),
                float(rgb[..., 2].float().std()),
            ),
            pixel_count=H * W,
        )


@dataclass
class DriftReport:
    """Statistical drift report between two decoded tensors."""

    n_frames: int
    height: int
    width: int
    l1_per_frame: list[float]
    l2_per_frame: list[float]
    max_abs_per_frame: list[int]
    mean_abs_per_channel: tuple[float, float, float]
    p99_abs_diff: int
    histogram_diff_signed: dict[int, int] = field(default_factory=dict)
    advisory_tag: str = "[diagnostic-not-score]"

    @property
    def l2_mean(self) -> float:
        return float(np.mean(self.l2_per_frame))

    @property
    def max_abs_global(self) -> int:
        return int(max(self.max_abs_per_frame))


def quantify_drift(
    av_rgb: torch.Tensor,
    dali_rgb: torch.Tensor,
    histogram_range: tuple[int, int] = (-10, 10),
) -> DriftReport:
    """Compute per-frame and global drift statistics between AV and DALI
    decoded RGB tensors.

    Parameters
    ----------
    av_rgb : torch.Tensor
        Shape (N, H, W, 3) uint8.
    dali_rgb : torch.Tensor
        Shape (N, H, W, 3) uint8.
    histogram_range : tuple[int, int]
        Inclusive range of signed per-pixel differences to bin (in uint8 LSB).

    Returns
    -------
    DriftReport
    """
    if av_rgb.shape != dali_rgb.shape:
        raise ValueError(
            f"shape mismatch: av_rgb {tuple(av_rgb.shape)} vs dali_rgb {tuple(dali_rgb.shape)}"
        )
    if av_rgb.dtype != torch.uint8 or dali_rgb.dtype != torch.uint8:
        raise ValueError(
            f"both inputs must be uint8; got av={av_rgb.dtype}, dali={dali_rgb.dtype}"
        )
    if av_rgb.ndim != 4 or av_rgb.shape[-1] != 3:
        raise ValueError(f"expected (N, H, W, 3), got {tuple(av_rgb.shape)}")

    diff_signed = av_rgb.to(torch.int16) - dali_rgb.to(torch.int16)  # signed
    diff_abs = diff_signed.abs()  # int16

    n_frames, H, W, _ = av_rgb.shape

    l1_per_frame = []
    l2_per_frame = []
    max_abs_per_frame = []
    for i in range(n_frames):
        l1_per_frame.append(float(diff_abs[i].sum()))
        l2_per_frame.append(float((diff_signed[i].float() ** 2).sum().sqrt()))
        max_abs_per_frame.append(int(diff_abs[i].max()))

    mean_abs_per_channel = (
        float(diff_abs[..., 0].float().mean()),
        float(diff_abs[..., 1].float().mean()),
        float(diff_abs[..., 2].float().mean()),
    )

    flat_abs = diff_abs.flatten().numpy()
    p99 = int(np.percentile(flat_abs, 99))

    # Build signed histogram for visual inspection (bias detection).
    lo, hi = histogram_range
    flat_signed = diff_signed.flatten().numpy()
    histogram = {}
    for v in range(lo, hi + 1):
        histogram[v] = int((flat_signed == v).sum())

    return DriftReport(
        n_frames=n_frames,
        height=H,
        width=W,
        l1_per_frame=l1_per_frame,
        l2_per_frame=l2_per_frame,
        max_abs_per_frame=max_abs_per_frame,
        mean_abs_per_channel=mean_abs_per_channel,
        p99_abs_diff=p99,
        histogram_diff_signed=histogram,
    )


def lipschitz_pose_drift_prediction(
    per_pixel_rgb_drift_lsb: float,
    pixel_count: int,
    posenet_input_normalize_std: float = 63.75,
    lipschitz_estimate: float = 1e-4,
) -> dict:
    """Map a per-pixel uint8 RGB drift assumption to an expected pose-component
    drift, given a Lipschitz estimate for FastViT-T12 + Hydra head.

    Per CLAUDE.md "Exact scorer architectures": PoseNet preprocessing applies
    rgb_to_yuv6 then normalize(mean=127.5, std=63.75). Effective input scale
    factor is 1/63.75 ≈ 0.0157 per uint8 LSB.

    Parameters
    ----------
    per_pixel_rgb_drift_lsb : float
        Assumed per-pixel max-abs drift between DALI and AV in uint8 LSB
        (typical literature: 1-3 LSB for FFmpeg-vs-NVDEC YUV->RGB on the same
        BT.601 limited-range matrix).
    pixel_count : int
        Number of RGB pixels per frame (H * W * 3).
    posenet_input_normalize_std : float
        Std used in PoseNet input normalization (default 63.75 per
        modules.py's FastViT preprocess).
    lipschitz_estimate : float
        Estimated Lipschitz constant of (input -> 6-dim pose) for FastViT-T12.
        Default 1e-4 per normalized RGB unit (back-of-envelope; refine via
        af945f502's PoseNet introspector + JVP probe).

    Returns
    -------
    dict with the chained derivation: per-pixel uint8 drift -> per-pixel
    normalized-input drift -> L2-norm of input perturbation -> pose drift
    via Lipschitz.
    """
    drift_normalized_per_pixel = per_pixel_rgb_drift_lsb / posenet_input_normalize_std
    # iid uniform[-d, +d] per pixel: variance = d^2 / 3
    # L2 norm of N iid samples each with variance v: sqrt(N * v)
    var_per_pixel = drift_normalized_per_pixel**2 / 3.0
    input_l2 = (pixel_count * var_per_pixel) ** 0.5
    # Lipschitz: |f(x+e) - f(x)| <= L * |e|
    pose_drift_predicted = lipschitz_estimate * input_l2
    # Observed at PR106 frontier: pose_avg_cuda ~ 1.7e-4, pose_avg_cpu ~ 3.4e-5
    # Gap ~ 1.4e-4 between the two paths.
    observed_gap_pr106 = 1.4e-4
    ratio = pose_drift_predicted / observed_gap_pr106 if observed_gap_pr106 > 0 else float("inf")

    if ratio > 0.7:
        verdict = "decoder-dominant"
    elif ratio < 0.3:
        verdict = "decoder-subdominant"
    else:
        verdict = "decoder-mixed"

    return dict(
        per_pixel_rgb_drift_lsb=per_pixel_rgb_drift_lsb,
        drift_normalized_per_pixel=drift_normalized_per_pixel,
        pixel_count=pixel_count,
        var_per_pixel=var_per_pixel,
        input_l2_normalized=input_l2,
        lipschitz_estimate=lipschitz_estimate,
        predicted_pose_component_drift=pose_drift_predicted,
        observed_pose_drift_gap_pr106=observed_gap_pr106,
        ratio_predicted_to_observed=ratio,
        verdict=verdict,
        advisory_tag="[diagnostic-not-score]",
        notes=(
            "iid-uniform model upper-bounds the structured drift. Real "
            "DALI-vs-AV drift is structured (chroma boundaries, edges); "
            "actual L2 may be 0.3-0.7x of this prediction."
        ),
    )


@dataclass
class DecoderDriftIntrospector:
    """High-level decoder-drift inspector.

    The AV decode path is locally runnable (PyAV is CPU-only). The DALI
    decode path requires CUDA + nvidia.dali, so this class designs the DALI
    invocation but does NOT run it; the operator dispatches a separate
    CUDA-capable script and feeds the dumped tensor back via ``ingest_dali_dump``.
    """

    advisory_tag: str = "[diagnostic-not-score]"
    macos_advisory_tag: str = "[macOS-CPU advisory only]"

    def decode_av(
        self, video_path: Path, frame_indices: Optional[list[int]] = None, max_frames: int = 10
    ) -> torch.Tensor:
        """Run AVVideoDataset's exact decode path.

        Imports upstream/frame_utils.py:yuv420_to_rgb to guarantee the EXACT
        byte-level path the contest evaluator uses on the CPU side.

        Returns
        -------
        torch.Tensor of shape (N, H, W, 3) uint8.
        """
        import av

        yuv420_to_rgb = _load_yuv420_to_rgb()

        target_indices = set(frame_indices) if frame_indices is not None else None

        fmt = "hevc" if Path(video_path).suffix == ".hevc" else None
        container = av.open(str(video_path), format=fmt)
        try:
            stream = container.streams.video[0]
            frames = []
            count = 0
            for i, frame in enumerate(container.decode(stream)):
                if target_indices is not None and i not in target_indices:
                    if i > max(target_indices):
                        break
                    continue
                frames.append(yuv420_to_rgb(frame))
                count += 1
                if target_indices is None and count >= max_frames:
                    break
        finally:
            container.close()

        return torch.stack(frames)

    def decode_dali_design(
        self, video_path: Path, frame_indices: Optional[list[int]] = None
    ) -> dict:
        """Describe (do NOT run) the DALI decode that would produce the
        CUDA-side ground-truth tensor.

        Returns a JSON-serializable dict that an operator can feed to a
        Linux+CUDA dispatch script (e.g.,
        ``experiments/dump_dali_decode.py`` on Lightning T4 g4dn.2xlarge).
        """
        return dict(
            advisory_tag=self.advisory_tag,
            video_path=str(video_path),
            frame_indices=frame_indices,
            requires=dict(
                cuda_capable=True,
                nvidia_dali=True,
                nvdec_supported_codec=True,
            ),
            pipeline=dict(
                op="fn.experimental.inputs.video",
                kwargs=dict(
                    name="inbuf",
                    sequence_length=2,
                    device="mixed",
                    no_copy=True,
                    blocking=False,
                    last_sequence_policy="pad",
                ),
                source_file="upstream/frame_utils.py:DaliVideoDataset",
                source_lines="L110-L157",
            ),
            yuv_to_rgb=dict(
                colorspace="BT.601 limited range",
                chroma_upsample="bilinear (NVDEC fixed-point)",
                output_dtype="uint8",
                note=(
                    "AVVideoDataset's yuv420_to_rgb() in frame_utils.py L159-183 "
                    "explicitly attempts to match this. Empirically verify "
                    "byte-identity with quantify_drift."
                ),
            ),
            dispatch_recipe=(
                "On a CUDA host with nvidia-dali installed:\n"
                "  python -c \"\n"
                "import sys; sys.path.insert(0, 'upstream'); \n"
                "import torch; from frame_utils import DaliVideoDataset; \n"
                "ds = DaliVideoDataset(['0.mkv'], data_dir='upstream/videos', "
                "batch_size=2, device=torch.device('cuda'), num_threads=2, seed=1234, "
                "prefetch_queue_depth=4); \n"
                "ds.prepare_data(); \n"
                "for path, idx, vid in ds: \n"
                "    torch.save(vid.cpu(), f'dali_{idx}.pt'); break\""
            ),
        )

    def ingest_dali_dump(self, dump_path: Path) -> torch.Tensor:
        """Load a DALI-decoded RGB tensor that was dumped on a CUDA host.

        Accepts .pt or .npy/.npz (with key 'frames'). Returns a (N, H, W, 3)
        uint8 tensor on CPU.
        """
        dump_path = Path(dump_path)
        if dump_path.suffix == ".pt":
            t = torch.load(dump_path, map_location="cpu", weights_only=True)
        elif dump_path.suffix in (".npy", ".npz"):
            arr = np.load(dump_path, allow_pickle=False)
            if dump_path.suffix == ".npz":
                arr = arr["frames"]
            t = torch.from_numpy(arr)
        else:
            raise ValueError(f"unsupported dump format: {dump_path.suffix}")

        # DALI's experimental.inputs.video typically returns (B, S, H, W, 3) uint8.
        # Unsqueeze/flatten to (N, H, W, 3) for fingerprint comparison.
        if t.ndim == 5:
            B, S, H, W, C = t.shape
            t = t.reshape(B * S, H, W, C)
        if t.dtype != torch.uint8:
            raise ValueError(
                f"DALI dump should be uint8 (NVDEC native); got {t.dtype}. "
                "If float/normalized, it has already been preprocessed and "
                "is no longer comparable to AV uint8 RGB."
            )
        return t

    def fingerprint(self, rgb_batch: torch.Tensor) -> list[FrameByteFingerprint]:
        """Compute per-frame fingerprints for a (N, H, W, 3) uint8 tensor."""
        return [FrameByteFingerprint.from_tensor(i, rgb_batch[i]) for i in range(rgb_batch.shape[0])]

    def quantify_drift(
        self,
        av_rgb: torch.Tensor,
        dali_rgb: torch.Tensor,
        histogram_range: tuple[int, int] = (-10, 10),
    ) -> DriftReport:
        """Convenience wrapper around module-level quantify_drift."""
        return quantify_drift(av_rgb, dali_rgb, histogram_range=histogram_range)


__all__ = [
    "DecoderDriftIntrospector",
    "DriftReport",
    "FrameByteFingerprint",
    "lipschitz_pose_drift_prediction",
    "quantify_drift",
]
