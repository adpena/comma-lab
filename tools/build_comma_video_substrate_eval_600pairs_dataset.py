# SPDX-License-Identifier: MIT
"""Build canonical HF dataset `adpena/comma-video-substrate-eval-600pairs`.

Decodes `upstream/videos/0.mkv` into 600 non-overlapping pairs (matching the
contest evaluator's `seq_len=2` batching per `upstream/evaluate.py`), runs
the contest SegNet + PoseNet on every pair, and uploads the labeled dataset
to the Hub. Once uploaded, every substrate trainer can pull from the
canonical dataset INSTEAD of decoding `upstream/videos/0.mkv` ad-hoc — which
saves ~5 min of CPU per Modal worker boot AND makes the substrate trainer
cross-platform (HF Jobs / Modal / Vast.ai / local M5 Max all consume the
same byte-identical dataset).

Schema (per row)
----------------
- image_frame_0 : PIL.Image RGB 384x512 (first frame of the pair)
- image_frame_1 : PIL.Image RGB 384x512 (second frame of the pair)
- segnet_mask_frame_1 : 5-class uint8 mask (192 x 256 = SegNet's argmax at
  its native model_input_size; matches `upstream.modules.SegNet.preprocess_input`)
- segnet_logits_frame_1 : (5, 192, 256) float16 logits (full logit surface
  for downstream distillation losses; ~376 KB per row)
- posenet_pose : (12,) float32 raw PoseNet head output (the upstream
  `compute_distortion` uses the first 6)
- video_name : str (the contest test video this pair came from)
- pair_idx : int (0..599)

Per CLAUDE.md:
  - "MPS auth eval is NOISE" — this dataset uses CUDA if available; falls
    back to CPU explicitly (we tag the upload's metadata with the source
    device so downstream consumers know the per-pair label provenance).
  - "Tag every reported score by axis" — the dataset card declares
    `[contest-CUDA]`, `[contest-CPU]`, or `[macOS-CPU advisory]` for the
    SegNet/PoseNet labels.
  - "Apples-to-apples evidence discipline" — we record the exact upstream
    modules.py SHA + safetensors SHAs + video SHA so future agents can
    verify byte-identity.

This is a CPU-capable build script (no GPU needed; SegNet+PoseNet inference
on 600 frame pairs takes ~20-30 min on CPU). The
heavy lifting is in pyav video decode + numpy/torch tensor conversion.

NOT a substrate trainer — this is a TOOL per Catalog #270 scope rule (tool
dispatches skip the substrate-only canonical dispatch optimization protocol
fields like min_smoke_gpu, autocast_fp16, etc.).

Usage
-----
Local dry-run (decode 4 pairs, no upload):

    .venv/bin/python tools/build_comma_video_substrate_eval_600pairs_dataset.py \\
        --max-pairs 4 --output-dir .omx/tmp/comma_video_dataset_smoke \\
        --no-upload

Full local build (no upload):

    .venv/bin/python tools/build_comma_video_substrate_eval_600pairs_dataset.py \\
        --output-dir experiments/results/comma_video_substrate_eval_600pairs \\
        --no-upload

Full build + upload to HF Hub (requires HF_TOKEN with write):

    .venv/bin/python tools/build_comma_video_substrate_eval_600pairs_dataset.py \\
        --output-dir experiments/results/comma_video_substrate_eval_600pairs \\
        --hf-repo adpena/comma-video-substrate-eval-600pairs

Canonical-vs-unique decision per layer
--------------------------------------
1. Video decode: ADOPT `pyav` (PR95 paradigm; non-DALI worker-compatible).
2. SegNet/PoseNet inference: ADOPT `upstream.modules.{SegNet, PoseNet}`
   (NO reimplementation — these are the contest scorers verbatim).
3. Mask compression: ADOPT uint8 argmax (matches SegNet.compute_distortion).
4. Logits storage: UNIQUE float16 (full logit surface for Hinton T=2.0
   distillation per CLAUDE.md "Quantizr intelligence" + future ATW V2-1
   per-pixel softmax sidecar).
5. Dataset format: ADOPT HF `datasets.Dataset.from_list` + `save_to_disk`
   directory (canonical local staging surface before HF Hub upload).
6. Upload: ADOPT `huggingface_hub.HfApi.upload_folder` (canonical per
   `huggingface-skills/hugging-face-datasets/SKILL.md`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# --- canonical 3-export Modal/CUDA env block per Catalog #244 ---
# (Tool dispatches are out-of-scope for Catalog #244 per the Catalog #270
# scope clarification, but we still set the envs as defense-in-depth since
# this tool MAY be invoked via Modal / Vast.ai workers in the future.)
os.environ.setdefault("DALI_DISABLE_NVML", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_VIDEOS_DIR = REPO_ROOT / "upstream" / "videos"
UPSTREAM_MODULES_PY = REPO_ROOT / "upstream" / "modules.py"
SEGNET_SAFETENSORS = REPO_ROOT / "upstream" / "models" / "segnet.safetensors"
POSENET_SAFETENSORS = REPO_ROOT / "upstream" / "models" / "posenet.safetensors"
PUBLIC_TEST_VIDEO_NAMES_TXT = REPO_ROOT / "upstream" / "public_test_video_names.txt"


CANONICAL_HF_REPO = "adpena/comma-video-substrate-eval-600pairs"
"""Canonical HF dataset identifier (per Insight 4)."""

CONTEST_SEQ_LEN = 2
"""upstream/evaluate.py uses seq_len=2 (non-overlapping pair batching)."""

EXPECTED_PAIRS_PER_VIDEO = 600
"""upstream/evaluate.py samples 600 non-overlapping pairs per video."""

CAMERA_RESOLUTION_HW = (384, 512)
"""Native camera resolution (H, W) per upstream.frame_utils.camera_size."""

SEGNET_MODEL_INPUT_HW = (192, 256)
"""SegNet's internal input resolution after preprocess_input (per upstream
modules.py:109: `segnet_model_input_size` reshape)."""


def resolve_public_video_path(
    video_name: str,
    videos_dir: Path = UPSTREAM_VIDEOS_DIR,
) -> Path:
    """Resolve upstream public-test video names with or without `.mkv`.

    The contest `public_test_video_names.txt` currently stores `0.mkv`; older
    scripts sometimes assumed bare stem `0`. Supporting both keeps this tool
    source-faithful without silently producing `0.mkv.mkv`.
    """
    clean_name = video_name.strip()
    if not clean_name:
        raise ValueError("public test video name is empty")
    video_name_path = Path(clean_name)
    if video_name_path.suffix == ".mkv":
        filename = video_name_path.name
    elif video_name_path.suffix == "":
        filename = f"{video_name_path.name}.mkv"
    else:
        raise ValueError(
            "public test video names must be bare stems or `.mkv` files, got "
            f"{video_name!r}"
        )
    return videos_dir / filename


# ---------------------------------------------------------------------------
# Provenance / cite-chain helpers (per Catalog #245 + observability surface)
# ---------------------------------------------------------------------------


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class BuildProvenance:
    """Apples-to-apples provenance per CLAUDE.md."""

    upstream_modules_py_sha256: str
    segnet_safetensors_sha256: str
    posenet_safetensors_sha256: str
    video_paths_to_sha256: dict[str, str] = field(default_factory=dict)
    device: str = "cpu"
    torch_version: str = ""
    n_pairs: int = 0
    seg_label_axis: str = "[contest-CUDA]"  # set per actual device used
    pose_label_axis: str = "[contest-CUDA]"

    def to_dict(self) -> dict:
        return {
            "upstream_modules_py_sha256": self.upstream_modules_py_sha256,
            "segnet_safetensors_sha256": self.segnet_safetensors_sha256,
            "posenet_safetensors_sha256": self.posenet_safetensors_sha256,
            "video_paths_to_sha256": dict(self.video_paths_to_sha256),
            "device": self.device,
            "torch_version": self.torch_version,
            "n_pairs": self.n_pairs,
            "seg_label_axis": self.seg_label_axis,
            "pose_label_axis": self.pose_label_axis,
        }


# ---------------------------------------------------------------------------
# Pyav video decode (canonical PR95 paradigm)
# ---------------------------------------------------------------------------


def decode_video_pairs_pyav(
    video_path: Path,
    n_pairs: int,
    target_hw: tuple[int, int] = CAMERA_RESOLUTION_HW,
) -> np.ndarray:
    """Decode a video into `n_pairs * 2` RGB frames at `target_hw`.

    Returns
    -------
    np.ndarray of shape (n_pairs, 2, H, W, 3) uint8.

    Notes
    -----
    Uses pyav (av) for non-DALI worker compatibility per CLAUDE.md
    "Forbidden re-implementing remote bootstrap inline" + HNeRV parity
    discipline lesson 9 (runtime closure).
    """
    try:
        import av
    except ImportError as exc:  # pragma: no cover - defensive
        raise ImportError(
            "decode_video_pairs_pyav requires `av`; install via "
            "`uv pip install av`"
        ) from exc

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    target_h, target_w = target_hw

    needed_frames = n_pairs * CONTEST_SEQ_LEN
    frames: list[np.ndarray] = []

    for frame in container.decode(stream):
        if len(frames) >= needed_frames:
            break
        img = frame.to_ndarray(format="rgb24")  # (H, W, 3)
        if img.shape[:2] != (target_h, target_w):
            # Decode-side resize via PIL bilinear (matches PR95 paradigm)
            from PIL import Image

            pil = Image.fromarray(img)
            pil = pil.resize((target_w, target_h), Image.BILINEAR)
            img = np.array(pil)
        frames.append(img)
    container.close()

    if len(frames) < needed_frames:
        raise RuntimeError(
            f"decode_video_pairs_pyav: needed {needed_frames} frames from "
            f"{video_path}, only got {len(frames)} (video too short?)"
        )

    arr = np.stack(frames[:needed_frames], axis=0)  # (n_pairs*2, H, W, 3)
    return arr.reshape(n_pairs, 2, target_h, target_w, 3)


# ---------------------------------------------------------------------------
# Resolve canonical device (CUDA if available, else MPS for local M5 Max
# advisory, else CPU; matches CLAUDE.md tagging rules)
# ---------------------------------------------------------------------------


def resolve_device(prefer_mps: bool = False) -> tuple[torch.device, str]:
    """Return (device, evidence_axis_tag) per CLAUDE.md device discipline."""
    if torch.cuda.is_available():
        return torch.device("cuda"), "[contest-CUDA]"
    if prefer_mps and getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available():
            # MPS proxy per CLAUDE.md "MPS auth eval is NOISE"; if
            # --prefer-mps is set the user is explicitly opting into advisory
            # labels that are never promotion evidence.
            return torch.device("mps"), "[MPS-PROXY advisory]"
    if platform.system() == "Darwin":
        return torch.device("cpu"), "[macOS-CPU advisory]"
    return torch.device("cpu"), "[contest-CPU]"


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------


def build_dataset(
    output_dir: Path,
    max_pairs: Optional[int] = None,
    prefer_mps: bool = False,
    verbose: bool = True,
) -> BuildProvenance:
    """Build the canonical dataset locally and write parquet + provenance.

    Parameters
    ----------
    output_dir : where to write the HF dataset dir + `provenance.json` + `README.md`.
    max_pairs : truncate to first N pairs (for smoke). None = all 600.
    prefer_mps : opt in to MPS proxy (tags labels as `[MPS-PROXY advisory]`;
        non-promotable per CLAUDE.md).

    Returns
    -------
    BuildProvenance with byte-identity manifest fields.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # PV-1: upstream artifacts present
    for required in (
        UPSTREAM_MODULES_PY,
        SEGNET_SAFETENSORS,
        POSENET_SAFETENSORS,
        PUBLIC_TEST_VIDEO_NAMES_TXT,
    ):
        if not required.exists():
            raise FileNotFoundError(
                f"Required upstream artifact missing: {required}. The dataset "
                "builder requires the pinned upstream snapshot to be intact "
                "per CLAUDE.md 'Non-Negotiable Upstream Rule'."
            )

    video_names = PUBLIC_TEST_VIDEO_NAMES_TXT.read_text().splitlines()
    if not video_names:
        raise RuntimeError(
            "public_test_video_names.txt is empty — cannot build dataset."
        )

    # Resolve canonical device
    device, axis_tag = resolve_device(prefer_mps=prefer_mps)
    if verbose:
        logger.info("Building dataset on device=%s (axis=%s)", device, axis_tag)

    # Lazy-import upstream modules (avoid pulling smp at module import time)
    sys.path.insert(0, str(REPO_ROOT / "upstream"))
    try:
        from modules import SegNet, PoseNet  # type: ignore
    finally:
        # We deliberately keep upstream on sys.path so subsequent calls
        # work; we just don't pollute the import cache further.
        pass

    from safetensors.torch import load_file

    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(str(SEGNET_SAFETENSORS), device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(str(POSENET_SAFETENSORS), device=str(device)))

    n_pairs = EXPECTED_PAIRS_PER_VIDEO if max_pairs is None else max_pairs

    # We currently only support video 0 (the contest's only public test video
    # per public_test_video_names.txt). Future expansion: iterate over all
    # video_names if/when comma releases additional public videos.
    video_name = video_names[0].strip()
    video_path = resolve_public_video_path(video_name)
    if not video_path.exists():
        raise FileNotFoundError(
            f"Video missing: {video_path} (expected from "
            f"public_test_video_names.txt entry '{video_name}')"
        )

    if verbose:
        logger.info("Decoding %d pairs from %s", n_pairs, video_path)
    pairs_arr = decode_video_pairs_pyav(video_path, n_pairs)
    # pairs_arr: (n_pairs, 2, H, W, 3) uint8

    rows: list[dict] = []
    with torch.inference_mode():
        for pair_idx in range(n_pairs):
            pair_np = pairs_arr[pair_idx]  # (2, H, W, 3)
            # Convert to (1, 2, 3, H, W) float for the contest scorers
            # (upstream's DistortionNet expects (B, T, H, W, C) with float
            # cast then rearrange — see modules.py:144)
            pair_t = (
                torch.from_numpy(pair_np)
                .to(device=device, dtype=torch.float32)
                .unsqueeze(0)
                .permute(0, 1, 4, 2, 3)
                .contiguous()
            )  # (1, 2, 3, H, W)

            # SegNet: input (B, T, 3, H, W) -> preprocess_input strips to
            # last frame (1, 3, H, W) -> resize to (192, 256) -> logits
            # (1, 5, 192, 256)
            seg_in = segnet.preprocess_input(pair_t)
            seg_logits = segnet(seg_in)  # (1, 5, 192, 256)
            seg_argmax = seg_logits.argmax(dim=1)  # (1, 192, 256) int64

            # PoseNet: input (B, T, 3, H, W) -> preprocess_input does
            # rgb_to_yuv6 + rearrange -> (1, 12, 192, 256) -> vision +
            # summarizer + hydra -> dict with 'pose' head (1, 12)
            pose_in = posenet.preprocess_input(pair_t)
            pose_out = posenet(pose_in)  # dict[str, (1, 12)]
            pose_vec = pose_out["pose"].squeeze(0)  # (12,)

            # Capture row
            rows.append(
                {
                    "image_frame_0": pair_np[0],  # uint8 (H, W, 3)
                    "image_frame_1": pair_np[1],  # uint8 (H, W, 3)
                    "segnet_mask_frame_1": (
                        seg_argmax.squeeze(0).to(torch.uint8).cpu().numpy()
                    ),  # uint8 (192, 256)
                    "segnet_logits_frame_1": (
                        seg_logits.squeeze(0).to(torch.float16).cpu().numpy()
                    ),  # float16 (5, 192, 256)
                    "posenet_pose": pose_vec.to(torch.float32).cpu().numpy(),
                    "video_name": video_name,
                    "pair_idx": pair_idx,
                }
            )
            if verbose and (pair_idx + 1) % 50 == 0:
                logger.info("  processed %d/%d pairs", pair_idx + 1, n_pairs)

    # Save as HF dataset
    if verbose:
        logger.info("Materializing HF dataset")
    from datasets import Dataset, Features, Image, Sequence, Value
    from PIL import Image as PILImage

    # Convert numpy arrays to PIL Images for the image columns
    materialized_rows = []
    for row in rows:
        materialized_rows.append(
            {
                "image_frame_0": PILImage.fromarray(row["image_frame_0"]),
                "image_frame_1": PILImage.fromarray(row["image_frame_1"]),
                "segnet_mask_frame_1": row["segnet_mask_frame_1"].flatten().tolist(),
                "segnet_logits_frame_1": row["segnet_logits_frame_1"].flatten().tolist(),
                "posenet_pose": row["posenet_pose"].tolist(),
                "video_name": row["video_name"],
                "pair_idx": int(row["pair_idx"]),
            }
        )

    features = Features(
        {
            "image_frame_0": Image(),
            "image_frame_1": Image(),
            "segnet_mask_frame_1": Sequence(Value("uint8")),
            "segnet_logits_frame_1": Sequence(Value("float16")),
            "posenet_pose": Sequence(Value("float32"), length=12),
            "video_name": Value("string"),
            "pair_idx": Value("int32"),
        }
    )
    ds = Dataset.from_list(materialized_rows, features=features)
    ds.save_to_disk(str(output_dir / "data"))

    # Provenance manifest
    provenance = BuildProvenance(
        upstream_modules_py_sha256=_sha256_of_file(UPSTREAM_MODULES_PY),
        segnet_safetensors_sha256=_sha256_of_file(SEGNET_SAFETENSORS),
        posenet_safetensors_sha256=_sha256_of_file(POSENET_SAFETENSORS),
        video_paths_to_sha256={video_name: _sha256_of_file(video_path)},
        device=str(device),
        torch_version=torch.__version__,
        n_pairs=n_pairs,
        seg_label_axis=axis_tag,
        pose_label_axis=axis_tag,
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance.to_dict(), indent=2, sort_keys=True)
    )

    # README.md (HF dataset card) per Catalog #305 observability surface
    readme = _build_dataset_card(provenance, n_pairs)
    (output_dir / "README.md").write_text(readme)

    if verbose:
        logger.info("Build complete: %d pairs at %s", n_pairs, output_dir)
    return provenance


def _build_dataset_card(provenance: BuildProvenance, n_pairs: int) -> str:
    """Generate the HF dataset card (README.md)."""
    return f"""---
license: mit
task_categories:
- image-segmentation
- image-classification
tags:
- comma-video-compression
- substrate-evaluation
- contest-cuda
size_categories:
- n<1K
---

# comma-video-substrate-eval-{n_pairs}pairs

Canonical evaluation dataset for the [comma.ai video compression challenge](https://comma.ai)
substrate-research lab at [github.com/adpena/comma-lab](https://github.com/adpena/comma-lab).

## Provenance

- `upstream/modules.py` sha256: `{provenance.upstream_modules_py_sha256}`
- `upstream/models/segnet.safetensors` sha256: `{provenance.segnet_safetensors_sha256}`
- `upstream/models/posenet.safetensors` sha256: `{provenance.posenet_safetensors_sha256}`
- inference device: `{provenance.device}`
- torch version: `{provenance.torch_version}`
- segnet label axis: `{provenance.seg_label_axis}`
- posenet label axis: `{provenance.pose_label_axis}`

## Schema

| field                 | shape           | dtype   | description                                                            |
|-----------------------|-----------------|---------|------------------------------------------------------------------------|
| image_frame_0         | (384, 512, 3)   | uint8   | first frame of pair                                                    |
| image_frame_1         | (384, 512, 3)   | uint8   | second frame of pair                                                   |
| segnet_mask_frame_1   | (192, 256)      | uint8   | 5-class argmax mask at SegNet's native model_input_size                |
| segnet_logits_frame_1 | (5, 192, 256)   | float16 | full logits (for Hinton T=2.0 distillation)                            |
| posenet_pose          | (12,)           | float32 | raw PoseNet head output (upstream `compute_distortion` uses first 6)   |
| video_name            | -               | str     | contest video this pair came from                                      |
| pair_idx              | -               | int32   | 0..599 (matches `upstream/evaluate.py` non-overlapping seq_len=2 batch)|

## Usage

```python
from datasets import load_dataset
ds = load_dataset("{CANONICAL_HF_REPO}", split="train")
print(ds[0]["posenet_pose"])  # (12,) raw PoseNet head output
```

## License

MIT — derivative of [`upstream/`](https://github.com/commaai/comma-video-compression-challenge)
pinned snapshot per CLAUDE.md "Non-Negotiable Upstream Rule".
"""


# ---------------------------------------------------------------------------
# HF Hub upload (canonical per huggingface-skills `hugging-face-datasets/SKILL.md`)
# ---------------------------------------------------------------------------


def upload_to_hf_hub(
    local_dir: Path,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
) -> str:
    """Upload the built dataset to HF Hub via `HfApi.upload_folder`."""
    try:
        from huggingface_hub import HfApi, get_token
    except ImportError as exc:  # pragma: no cover - defensive
        raise ImportError(
            "upload_to_hf_hub requires `huggingface_hub`; install via "
            "`uv pip install huggingface_hub`"
        ) from exc

    if token is None:
        token = get_token()
    if not token:
        raise RuntimeError(
            "No HF_TOKEN found. Either pass --token or run `huggingface-cli "
            "login` first. Token must have write permissions."
        )

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(local_dir),
        commit_message=(
            f"build_comma_video_substrate_eval_600pairs_dataset: "
            f"build from local CPU/CUDA"
        ),
    )
    return f"https://huggingface.co/datasets/{repo_id}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0] if __doc__ else "",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/comma_video_substrate_eval_600pairs"),
        help="Local directory for built HF dataset + provenance + README.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Truncate to first N pairs (smoke). Default 600 (all).",
    )
    parser.add_argument(
        "--prefer-mps",
        action="store_true",
        help=(
            "Opt into MPS proxy for local M5 Max (tags labels as "
            "[MPS-PROXY advisory] per CLAUDE.md 'MPS auth eval is NOISE'). "
            "Non-promotable. Default off."
        ),
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help=(
            "HF Hub dataset identifier (e.g. adpena/comma-video-substrate-eval-600pairs). "
            "If unset, equivalent to --no-upload."
        ),
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Build locally only; skip HF Hub upload.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF token (default: env HF_TOKEN or `huggingface-cli login` cache).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create as private dataset (default public).",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Verbose logging."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    provenance = build_dataset(
        output_dir=args.output_dir,
        max_pairs=args.max_pairs,
        prefer_mps=args.prefer_mps,
        verbose=args.verbose,
    )
    logger.info("Provenance: %s", provenance.to_dict())

    if args.no_upload or not args.hf_repo:
        logger.info("Skipping HF Hub upload (--no-upload or no --hf-repo set)")
        return 0

    url = upload_to_hf_hub(
        local_dir=args.output_dir,
        repo_id=args.hf_repo,
        token=args.token,
        private=args.private,
    )
    logger.info("Uploaded to %s", url)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
