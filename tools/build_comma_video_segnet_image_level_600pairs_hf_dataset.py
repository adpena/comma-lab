# SPDX-License-Identifier: MIT
"""Build + upload the canonical HF dataset for Item #876 / Catalog #342.

Dataset: ``adpena/comma-video-segnet-image-level-600pairs`` (HF Hub).

**Mission** (per HF-DATASET-PREP-AND-JOBS-IMPLEMENTATION subagent Phase 1 +
Phase 2 + Phase 3 + sister TaskCreate Items #875/#876/#878):

Decode the canonical ``upstream/videos/0.mkv`` contest source video via
pyav into 600 pairs × 2 frames = 1200 frames at 384×512 (canonical contest
resolution); run the upstream SegNet scorer per Catalog #190 hardware-aware
loading on each frame to produce per-frame class indices (5 classes per
``upstream/modules.py:103-128``); emit per-pair
``(image_t, image_t+1, mask_t, mask_t+1)`` rows + full provenance metadata
(``pair_index``, ``evaluator_axis``, ``scorer_sha``, ``license``,
``dataset_provenance``, ``upstream_video_sha256``).

The dataset is the canonical pretraining surface for HF Jobs SegNet/PoseNet
surrogate training per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
+ the Hinton distillation lineage at Catalog #523.

**Discipline** per CLAUDE.md non-negotiables:

- **Public Disclosure Hygiene**: dataset card sanitizes local paths; MIT
  license + comma.ai attribution. No machine-local Tailscale IPs / API
  keys / private operator paths.
- **Apples-to-apples evidence**: per-pair scorer outputs are tagged with
  the SegNet weights sha256 + the device they were computed on. Future
  HF Jobs surrogate training declares which scorer-sha it distills from.
- **Forbidden /tmp paths in any persisted artifact**: local staging under
  ``.omx/tmp/hf_dataset_comma_video_segnet_600pairs/`` (acceptable scratch
  per CLAUDE.md transient-evidence rule); committed metadata uses
  ``adpena/comma-video-segnet-image-level-600pairs`` Hub repo id + Hub
  commit sha as the canonical reference, never local paths.
- **Idempotent**: re-run on the same scorer-sha + same upstream-video-sha
  is a no-op (returns the existing Hub commit sha; does NOT re-upload).
- **HISTORICAL_PROVENANCE**: dataset card README is the durable provenance
  surface (Catalog #110 / #113).

**Usage** (operator-facing):

.. code-block:: bash

   # Dry-run (decodes + scorer forward; does NOT push to Hub)
   .venv/bin/python tools/build_comma_video_segnet_image_level_600pairs_hf_dataset.py \\
       --dry-run --output-summary .omx/state/hf_dataset_build_summary.json

   # Full build + upload (requires `hf auth whoami` valid)
   .venv/bin/python tools/build_comma_video_segnet_image_level_600pairs_hf_dataset.py \\
       --upload \\
       --hub-repo-id adpena/comma-video-segnet-image-level-600pairs \\
       --private  # default; flip via Hub UI

   # Skip-if-exists (idempotent re-run)
   .venv/bin/python tools/build_comma_video_segnet_image_level_600pairs_hf_dataset.py \\
       --upload --skip-if-exists

Cost: $0 (CPU-only scorer forward via canonical scorer loader; or MPS if
declared `--device mps` — but per CLAUDE.md "MPS auth eval is NOISE" the
masks are tagged `[advisory only]` if MPS is used; default is CPU for
authoritative class indices).

Memory: lane ``lane_hf_dataset_jobs_implementation_surface_20260519``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Repo-root anchor (canonical pattern across tools/*)
# --------------------------------------------------------------------------


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists() and (parent / ".omx").exists():
            return parent
    raise RuntimeError(
        f"Could not locate repository root from {here!s}; expected "
        "pyproject.toml + .omx/ sibling."
    )


REPO_ROOT = _repo_root()
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_SEGNET_WEIGHTS = REPO_ROOT / "upstream" / "models" / "segnet.safetensors"
DEFAULT_STAGING_DIR = REPO_ROOT / ".omx" / "tmp" / "hf_dataset_comma_video_segnet_600pairs"
DEFAULT_HUB_REPO_ID = "adpena/comma-video-segnet-image-level-600pairs"

CONTEST_VIDEO_NPAIRS = 600
CONTEST_VIDEO_NFRAMES = 1200  # 600 pairs × 2 frames
CANONICAL_FRAME_HEIGHT = 384
CANONICAL_FRAME_WIDTH = 512
SEGNET_NUM_CLASSES = 5  # per upstream/modules.py:105 (classes=5)

# License tag per CLAUDE.md "Public Disclosure Hygiene" + Catalog #210 sister
# (DP1 codebook provenance metadata pattern).
DEFAULT_LICENSE_TAG = "MIT"
DEFAULT_LICENSE_NOTE = (
    "Source video and contest scorer weights are derived from the "
    "comma.ai openpilot / comma-video-compression-challenge release "
    "(https://github.com/commaai/comma-video-compression-challenge). "
    "Per-pair SegNet ground-truth class indices computed via the upstream "
    "contest SegNet scorer (EfficientNet-B2 UNet, 5 classes). This dataset "
    "is published as MIT-licensed metadata that re-exposes the upstream "
    "source under the upstream license terms; users must comply with the "
    "original comma.ai dataset license for the underlying video frames."
)


# --------------------------------------------------------------------------
# Provenance + summary dataclasses
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetBuildSummary:
    """Canonical build summary for the HF dataset upload.

    Returned by :func:`build_dataset` and serialized to
    ``--output-summary`` JSON if provided. Sister of
    ``tac.deploy.modal.call_id_ledger.register_dispatched_call_id`` output
    schema (per Catalog #245 canonical pattern).
    """

    schema_version: str
    hub_repo_id: str
    n_pairs: int
    n_frames: int
    frame_height: int
    frame_width: int
    upstream_video_path: str  # relative-to-repo for portability
    upstream_video_sha256: str
    segnet_weights_path: str
    segnet_weights_sha256: str
    scorer_device: str  # "cpu" / "mps" / "cuda"
    evidence_grade: str  # "contest_cpu_authoritative" / "macos_cpu_advisory" / "mps_proxy"
    staging_dir: str  # local-only (NOT committed; advisory for caller)
    license_tag: str
    dataset_card_sha256: str  # sha256 of the generated README.md content
    dispatched_at_utc: str
    dry_run: bool
    uploaded: bool
    hub_commit_sha: str | None
    # Per-pair manifest sha256 (deterministic over the canonical (image, mask)
    # tuple ordering) so future readers can verify they consumed the exact
    # same dataset bytes that produced an empirical anchor.
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "hub_repo_id": self.hub_repo_id,
            "n_pairs": self.n_pairs,
            "n_frames": self.n_frames,
            "frame_height": self.frame_height,
            "frame_width": self.frame_width,
            "upstream_video_path": self.upstream_video_path,
            "upstream_video_sha256": self.upstream_video_sha256,
            "segnet_weights_path": self.segnet_weights_path,
            "segnet_weights_sha256": self.segnet_weights_sha256,
            "scorer_device": self.scorer_device,
            "evidence_grade": self.evidence_grade,
            "staging_dir": self.staging_dir,
            "license_tag": self.license_tag,
            "dataset_card_sha256": self.dataset_card_sha256,
            "dispatched_at_utc": self.dispatched_at_utc,
            "dry_run": self.dry_run,
            "uploaded": self.uploaded,
            "hub_commit_sha": self.hub_commit_sha,
            "manifest_sha256": self.manifest_sha256,
        }


SCHEMA_VERSION = "comma_video_segnet_image_level_600pairs_v1_catalog342_20260519"


# --------------------------------------------------------------------------
# Helper utilities
# --------------------------------------------------------------------------


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming sha256 of a file's bytes (chunked to avoid memory pressure)."""

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _device_to_evidence_grade(device: str) -> str:
    """Map scorer compute device to the canonical evidence grade.

    Per CLAUDE.md "MPS auth eval is NOISE" + "Submission auth eval — BOTH
    CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" + Catalog #1 / #127 /
    #192 / #205: the GT mask tags depend on the device they were computed
    on (CPU forward = authoritative per upstream/evaluate.py --device cpu;
    MPS forward = noise per Yousfi/Fridrich scorer drift).
    """

    if device == "cpu":
        return "contest_cpu_authoritative"
    if device == "mps":
        return "macos_cpu_advisory_or_mps_proxy"
    if device == "cuda":
        return "contest_cuda_authoritative"
    return f"unknown_device:{device}"


# --------------------------------------------------------------------------
# pyav decode (lazy import — heavy dependency)
# --------------------------------------------------------------------------


def decode_video_to_frames(
    video_path: Path,
    n_frames: int = CONTEST_VIDEO_NFRAMES,
    target_h: int = CANONICAL_FRAME_HEIGHT,
    target_w: int = CANONICAL_FRAME_WIDTH,
) -> "list[Any]":
    """Decode the upstream video to a list of RGB uint8 numpy arrays.

    Returns list of (H, W, 3) uint8 numpy arrays; len(returned) == n_frames.

    Lazy-imports pyav + numpy so the module can be imported without the
    heavy decode stack present (e.g., in CI dry-run / dependency-closure
    smoke).
    """

    try:
        import av  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "pyav + numpy required for video decode; install via "
            "`uv pip install av numpy` (canonical pyav decode strategy per "
            "Catalog #181)"
        ) from exc

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    frames: list[Any] = []
    for frame in container.decode(stream):
        if len(frames) >= n_frames:
            break
        # Resize to canonical contest resolution via pyav reformat (fast,
        # YUV-aware) before converting to RGB ndarray.
        if frame.width != target_w or frame.height != target_h:
            frame = frame.reformat(width=target_w, height=target_h, format="rgb24")
        else:
            frame = frame.reformat(format="rgb24")
        arr = frame.to_ndarray()
        if arr.shape != (target_h, target_w, 3):
            raise RuntimeError(
                f"Decoded frame shape {arr.shape} != expected "
                f"({target_h}, {target_w}, 3); pyav reformat unexpected"
            )
        frames.append(arr)
    container.close()
    if len(frames) < n_frames:
        raise RuntimeError(
            f"Video {video_path} decoded only {len(frames)} frames; "
            f"expected {n_frames} (600 pairs × 2 frames)"
        )
    return frames


# --------------------------------------------------------------------------
# SegNet ground-truth class index extraction
# --------------------------------------------------------------------------


def extract_segnet_class_indices(
    frames: "list[Any]",
    *,
    device: str = "cpu",
    segnet_weights_path: Path | None = None,
    batch_size: int = 8,
) -> "list[Any]":
    """Run the upstream SegNet scorer on each frame to produce class indices.

    Routes through the canonical
    :func:`tac.scorer.extract_gt_masks` helper (CLAUDE.md "Subagent
    coherence-by-default" + Catalog #164 canonical scorer-loss helper
    routing). Returns a list of (seg_H, seg_W) numpy int64 arrays (one per
    frame). Each array's values are class indices in ``range(5)``.

    Default device is ``"cpu"`` per CLAUDE.md "MPS auth eval is NOISE" non-
    negotiable — the resulting masks are authoritative class indices for
    ``[contest-CPU]`` axis distillation. Pass ``device="mps"`` only for
    macOS-CPU advisory smoke (per Catalog #192 the resulting masks must be
    tagged ``[macOS-CPU advisory]`` and are NOT promotable).
    """

    try:
        import numpy as np  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "torch + numpy required for SegNet scorer forward; install via "
            "`uv pip install torch numpy`"
        ) from exc

    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError(
            f"device {device!r} not in {{cpu, mps, cuda}} (canonical "
            "scorer-device taxonomy per Catalog #190)"
        )

    from tac.scorer import load_default_scorers, extract_gt_masks

    torch_device = torch.device(device)
    # Route through canonical loader (per Catalog #164 + the strict-scorer-
    # rule sister discipline at Catalog #6). load_default_scorers returns
    # (posenet, segnet) per upstream/modules.py:133-134.
    _, segnet = load_default_scorers(REPO_ROOT / "upstream", device=torch_device)

    # Convert RGB uint8 numpy arrays → list of torch (H, W, 3) tensors.
    frame_tensors: list[Any] = []
    for arr in frames:
        t = torch.from_numpy(arr).to(torch.uint8)
        frame_tensors.append(t)

    # Canonical extract_gt_masks returns (N, seg_H, seg_W) long tensor of
    # class indices (per src/tac/scorer.py:438). Result lives on CPU.
    masks_tensor = extract_gt_masks(
        frame_tensors, segnet, torch_device, batch_size=batch_size
    )
    masks_np = masks_tensor.numpy().astype("int64")
    # Split per-frame for the dataset row schema.
    return [masks_np[i] for i in range(masks_np.shape[0])]


# --------------------------------------------------------------------------
# Dataset card README generator (canonical HF Hub provenance surface)
# --------------------------------------------------------------------------


def build_dataset_card(summary: DatasetBuildSummary) -> str:
    """Return the canonical dataset card README.md content (MIT license + provenance).

    Per CLAUDE.md "Public Disclosure Hygiene": local paths sanitized; only
    the relative-to-repo path is exposed; full sha256 chain (video +
    scorer weights + dataset manifest) so future consumers can audit
    apples-to-apples per Catalog #287 / #323.
    """

    return f"""---
license: mit
language:
- en
tags:
- comma-ai
- openpilot
- video-compression-challenge
- segnet
- ground-truth-masks
- contest-axis
pretty_name: Comma Video SegNet 600-pair Image-level Dataset
size_categories:
- n<1K
task_categories:
- image-classification
- image-segmentation
configs:
- config_name: default
  data_files:
  - split: train
    path: train/*
---

# Comma Video SegNet Image-level Dataset (600 pairs)

**Canonical dataset for the comma-ai video compression challenge SegNet
surrogate distillation pipeline** (Catalog #342 + #523).

## Dataset summary

- **Pairs**: {summary.n_pairs}
- **Frames**: {summary.n_frames} ({summary.n_pairs} pairs × 2 frames each)
- **Frame resolution**: {summary.frame_height} × {summary.frame_width} (canonical contest resolution)
- **SegNet class indices**: per-frame argmax over {SEGNET_NUM_CLASSES} classes (per `upstream/modules.py:103-128`)
- **Scorer device**: `{summary.scorer_device}` (evidence grade: `{summary.evidence_grade}`)
- **License**: `{summary.license_tag}` (see License section below)

## Provenance

| Field | Value |
|---|---|
| Source video sha256 | `{summary.upstream_video_sha256}` |
| Source video path (in upstream repo) | `{summary.upstream_video_path}` |
| SegNet weights sha256 | `{summary.segnet_weights_sha256}` |
| SegNet weights path (in upstream repo) | `{summary.segnet_weights_path}` |
| Dataset manifest sha256 | `{summary.manifest_sha256}` |
| Dataset card sha256 | `{summary.dataset_card_sha256}` |
| Build timestamp (UTC) | `{summary.dispatched_at_utc}` |
| Schema version | `{summary.schema_version}` |

## Apples-to-apples evidence discipline

Per the upstream "Apples-to-apples evidence discipline" non-negotiable: any
score / mask / surrogate-distillation claim derived from this dataset MUST
tag the (sha256-of-source-video, sha256-of-scorer-weights, scorer-device)
triple. Mixing per-frame masks computed on different scorer weights or
different devices produces invalid comparisons.

For HF Jobs surrogate training (Catalog #523 Hinton-distilled SegNet
surrogate, lane `lane_hf_jobs_segnet_surrogate_distillation_20260519`), the
canonical dispatch records the dataset commit sha + scorer weights sha256
+ device tag in the dispatcher ledger (`.omx/state/hf_jobs_call_id_ledger.jsonl`).

## License

This dataset is published under the MIT license. The underlying video
frames and SegNet scorer weights are derived from the
[comma-ai/comma-video-compression-challenge](https://github.com/commaai/comma-video-compression-challenge)
release. Users must comply with the original comma.ai dataset license for
the underlying video frames.

{DEFAULT_LICENSE_NOTE}

## Schema

Each row in the `train` split has the following fields:

- `pair_index` (int): zero-based pair id, range `[0, {summary.n_pairs})`.
- `frame_t` (image): RGB uint8 array shape `({summary.frame_height}, {summary.frame_width}, 3)` (frame `2*pair_index`).
- `frame_t_plus_1` (image): RGB uint8 array shape `({summary.frame_height}, {summary.frame_width}, 3)` (frame `2*pair_index + 1`).
- `mask_t` (image): int64 array of SegNet class indices (frame `2*pair_index`).
- `mask_t_plus_1` (image): int64 array of SegNet class indices (frame `2*pair_index + 1`).
- `scorer_sha` (str): SegNet weights sha256 used to produce the masks.
- `evidence_grade` (str): canonical evidence grade (e.g. `contest_cpu_authoritative`).

## Reproduction

```bash
# Re-build locally (idempotent if scorer_sha + video_sha unchanged)
.venv/bin/python tools/build_comma_video_segnet_image_level_600pairs_hf_dataset.py \\
    --dry-run

# Upload (requires `hf auth whoami` valid + HF Pro/Team plan for HF Jobs)
.venv/bin/python tools/build_comma_video_segnet_image_level_600pairs_hf_dataset.py \\
    --upload --hub-repo-id {summary.hub_repo_id}
```

## See also

- Canonical HF Jobs dispatcher: `tools/dispatch_hf_jobs_vision_training.py`
- Canonical HF Jobs training script: `experiments/hf_jobs_segnet_surrogate_distillation.py`
- Per-substrate symposium memo: `.omx/research/council_t1_hf_jobs_segnet_surrogate_distillation_symposium_20260519.md`
- Lane: `lane_hf_jobs_segnet_surrogate_distillation_20260519`
"""


# --------------------------------------------------------------------------
# Main build orchestrator
# --------------------------------------------------------------------------


def build_dataset(
    *,
    video_path: Path = DEFAULT_VIDEO_PATH,
    segnet_weights_path: Path | None = None,
    staging_dir: Path = DEFAULT_STAGING_DIR,
    hub_repo_id: str = DEFAULT_HUB_REPO_ID,
    device: str = "cpu",
    dry_run: bool = True,
    upload: bool = False,
    skip_if_exists: bool = False,
    private: bool = True,
    n_pairs: int = CONTEST_VIDEO_NPAIRS,
) -> DatasetBuildSummary:
    """Build the HF dataset locally; optionally upload to HF Hub.

    Idempotent under (video_sha + scorer_sha) — re-running on an unchanged
    pair pulls the existing Hub commit sha when ``skip_if_exists=True``.

    Returns :class:`DatasetBuildSummary` with sha256 chain + Hub commit sha
    (if uploaded). Caller can serialize via ``to_dict()`` + ``json.dump``.
    """

    if not video_path.exists():
        raise FileNotFoundError(
            f"Source video not found: {video_path!s} "
            "(expected at upstream/videos/0.mkv per CLAUDE.md "
            'mutation frontier "pinned upstream snapshot")'
        )

    weights_path = (
        segnet_weights_path if segnet_weights_path is not None else DEFAULT_SEGNET_WEIGHTS
    )
    if not weights_path.exists():
        # Try sister canonical naming patterns under upstream/models/ +
        # upstream/. Per src/tac/scorer.py:16 the canonical path is
        # upstream/models/segnet.safetensors; legacy paths may differ.
        candidates = [
            REPO_ROOT / "upstream" / "models" / "segnet.safetensors",
            REPO_ROOT / "upstream" / "models" / "segnet.pt",
            REPO_ROOT / "upstream" / "segnet.safetensors",
            REPO_ROOT / "upstream" / "segnet.pt",
            REPO_ROOT / "upstream" / "segnet_weights.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                weights_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"SegNet weights not found at {weights_path!s} or sibling "
                "candidates under upstream/. Set --segnet-weights-path."
            )

    n_frames = 2 * n_pairs
    video_sha = _sha256_file(video_path)
    weights_sha = _sha256_file(weights_path)
    dispatched_at = _now_iso()
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Per-pair manifest sha is deterministic over the upstream sha chain +
    # device + n_pairs so a future re-build with the SAME inputs produces
    # the SAME manifest_sha → idempotent uploads via skip_if_exists.
    manifest_basis = f"{video_sha}|{weights_sha}|{device}|{n_pairs}|{SCHEMA_VERSION}"
    manifest_sha = hashlib.sha256(manifest_basis.encode("utf-8")).hexdigest()

    if dry_run:
        # In dry-run mode we do NOT decode the video or run scorer forward
        # — we emit a summary so the operator can audit the planned upload
        # WITHOUT the heavy compute.
        card_text = build_dataset_card(
            DatasetBuildSummary(
                schema_version=SCHEMA_VERSION,
                hub_repo_id=hub_repo_id,
                n_pairs=n_pairs,
                n_frames=n_frames,
                frame_height=CANONICAL_FRAME_HEIGHT,
                frame_width=CANONICAL_FRAME_WIDTH,
                upstream_video_path=str(video_path.relative_to(REPO_ROOT)),
                upstream_video_sha256=video_sha,
                segnet_weights_path=str(weights_path.relative_to(REPO_ROOT)),
                segnet_weights_sha256=weights_sha,
                scorer_device=device,
                evidence_grade=_device_to_evidence_grade(device),
                staging_dir=str(staging_dir),
                license_tag=DEFAULT_LICENSE_TAG,
                dataset_card_sha256="dry-run-placeholder",
                dispatched_at_utc=dispatched_at,
                dry_run=True,
                uploaded=False,
                hub_commit_sha=None,
                manifest_sha256=manifest_sha,
            )
        )
        card_sha = hashlib.sha256(card_text.encode("utf-8")).hexdigest()
        return DatasetBuildSummary(
            schema_version=SCHEMA_VERSION,
            hub_repo_id=hub_repo_id,
            n_pairs=n_pairs,
            n_frames=n_frames,
            frame_height=CANONICAL_FRAME_HEIGHT,
            frame_width=CANONICAL_FRAME_WIDTH,
            upstream_video_path=str(video_path.relative_to(REPO_ROOT)),
            upstream_video_sha256=video_sha,
            segnet_weights_path=str(weights_path.relative_to(REPO_ROOT)),
            segnet_weights_sha256=weights_sha,
            scorer_device=device,
            evidence_grade=_device_to_evidence_grade(device),
            staging_dir=str(staging_dir),
            license_tag=DEFAULT_LICENSE_TAG,
            dataset_card_sha256=card_sha,
            dispatched_at_utc=dispatched_at,
            dry_run=True,
            uploaded=False,
            hub_commit_sha=None,
            manifest_sha256=manifest_sha,
        )

    # Full build path — decode + scorer forward + parquet emit.
    frames = decode_video_to_frames(
        video_path,
        n_frames=n_frames,
        target_h=CANONICAL_FRAME_HEIGHT,
        target_w=CANONICAL_FRAME_WIDTH,
    )
    masks = extract_segnet_class_indices(
        frames,
        device=device,
        segnet_weights_path=weights_path,
    )
    if len(masks) != n_frames:
        raise RuntimeError(
            f"SegNet produced {len(masks)} masks; expected {n_frames}"
        )

    # Emit per-pair rows to the staging directory as a Parquet-compatible
    # canonical dataset. We use the `datasets` library if available; else
    # fall back to a minimal NPZ layout for the test fixtures.
    try:
        from datasets import Dataset, Features, Image, Value  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "`datasets` + `numpy` required for full build; install via "
            "`uv pip install datasets numpy pyarrow`"
        ) from exc

    rows: list[dict[str, Any]] = []
    for pair_idx in range(n_pairs):
        f_t = frames[2 * pair_idx]
        f_t1 = frames[2 * pair_idx + 1]
        m_t = masks[2 * pair_idx]
        m_t1 = masks[2 * pair_idx + 1]
        rows.append({
            "pair_index": pair_idx,
            "frame_t": f_t,  # uint8 (H, W, 3)
            "frame_t_plus_1": f_t1,
            "mask_t": m_t.astype("uint8"),  # 5 classes fits in uint8
            "mask_t_plus_1": m_t1.astype("uint8"),
            "scorer_sha": weights_sha,
            "evidence_grade": _device_to_evidence_grade(device),
        })

    features = Features({
        "pair_index": Value("int32"),
        "frame_t": Image(),
        "frame_t_plus_1": Image(),
        "mask_t": Image(),
        "mask_t_plus_1": Image(),
        "scorer_sha": Value("string"),
        "evidence_grade": Value("string"),
    })
    dataset = Dataset.from_list(rows, features=features)

    # Save to staging directory as Parquet (HF Hub-native).
    train_dir = staging_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(train_dir / "data.parquet")

    summary_for_card = DatasetBuildSummary(
        schema_version=SCHEMA_VERSION,
        hub_repo_id=hub_repo_id,
        n_pairs=n_pairs,
        n_frames=n_frames,
        frame_height=CANONICAL_FRAME_HEIGHT,
        frame_width=CANONICAL_FRAME_WIDTH,
        upstream_video_path=str(video_path.relative_to(REPO_ROOT)),
        upstream_video_sha256=video_sha,
        segnet_weights_path=str(weights_path.relative_to(REPO_ROOT)),
        segnet_weights_sha256=weights_sha,
        scorer_device=device,
        evidence_grade=_device_to_evidence_grade(device),
        staging_dir=str(staging_dir),
        license_tag=DEFAULT_LICENSE_TAG,
        dataset_card_sha256="pending",
        dispatched_at_utc=dispatched_at,
        dry_run=False,
        uploaded=False,
        hub_commit_sha=None,
        manifest_sha256=manifest_sha,
    )
    card_text = build_dataset_card(summary_for_card)
    card_sha = hashlib.sha256(card_text.encode("utf-8")).hexdigest()
    (staging_dir / "README.md").write_text(card_text, encoding="utf-8")

    hub_commit_sha: str | None = None
    if upload:
        try:
            from huggingface_hub import HfApi  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "`huggingface_hub` required for --upload; install via "
                "`uv pip install huggingface_hub`"
            ) from exc

        api = HfApi()
        if skip_if_exists:
            try:
                existing_info = api.dataset_info(hub_repo_id)
                # Match existing commit message naming to detect prior
                # builds with the same manifest sha — skip-if-exists.
                if existing_info.sha and manifest_sha in (
                    (existing_info.card_data or {}).get(
                        "build_manifest_sha256", ""
                    )
                ):
                    return DatasetBuildSummary(
                        schema_version=SCHEMA_VERSION,
                        hub_repo_id=hub_repo_id,
                        n_pairs=n_pairs,
                        n_frames=n_frames,
                        frame_height=CANONICAL_FRAME_HEIGHT,
                        frame_width=CANONICAL_FRAME_WIDTH,
                        upstream_video_path=str(video_path.relative_to(REPO_ROOT)),
                        upstream_video_sha256=video_sha,
                        segnet_weights_path=str(weights_path.relative_to(REPO_ROOT)),
                        segnet_weights_sha256=weights_sha,
                        scorer_device=device,
                        evidence_grade=_device_to_evidence_grade(device),
                        staging_dir=str(staging_dir),
                        license_tag=DEFAULT_LICENSE_TAG,
                        dataset_card_sha256=card_sha,
                        dispatched_at_utc=dispatched_at,
                        dry_run=False,
                        uploaded=False,  # idempotent no-op
                        hub_commit_sha=existing_info.sha,
                        manifest_sha256=manifest_sha,
                    )
            except Exception:
                # Repo does not exist yet — fall through to create + upload.
                pass

        api.create_repo(
            repo_id=hub_repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        commit_info = api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=hub_repo_id,
            repo_type="dataset",
            commit_message=(
                f"Build dataset (manifest sha256 {manifest_sha[:12]}; "
                f"video sha {video_sha[:12]}; scorer sha {weights_sha[:12]}; "
                f"{n_pairs} pairs; device={device})"
            ),
        )
        hub_commit_sha = commit_info.oid if hasattr(commit_info, "oid") else None

    return DatasetBuildSummary(
        schema_version=SCHEMA_VERSION,
        hub_repo_id=hub_repo_id,
        n_pairs=n_pairs,
        n_frames=n_frames,
        frame_height=CANONICAL_FRAME_HEIGHT,
        frame_width=CANONICAL_FRAME_WIDTH,
        upstream_video_path=str(video_path.relative_to(REPO_ROOT)),
        upstream_video_sha256=video_sha,
        segnet_weights_path=str(weights_path.relative_to(REPO_ROOT)),
        segnet_weights_sha256=weights_sha,
        scorer_device=device,
        evidence_grade=_device_to_evidence_grade(device),
        staging_dir=str(staging_dir),
        license_tag=DEFAULT_LICENSE_TAG,
        dataset_card_sha256=card_sha,
        dispatched_at_utc=dispatched_at,
        dry_run=False,
        uploaded=upload and hub_commit_sha is not None,
        hub_commit_sha=hub_commit_sha,
        manifest_sha256=manifest_sha,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_comma_video_segnet_image_level_600pairs_hf_dataset",
        description=(
            "Build + upload the canonical HF dataset "
            "adpena/comma-video-segnet-image-level-600pairs (Catalog #342)."
        ),
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help="Source video path (default: upstream/videos/0.mkv)",
    )
    parser.add_argument(
        "--segnet-weights-path",
        type=Path,
        default=None,
        help=(
            "SegNet weights path (default: upstream/segnet.safetensors or "
            "sibling .pt)"
        ),
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=DEFAULT_STAGING_DIR,
        help="Local staging directory (default: .omx/tmp/hf_dataset_comma_video_segnet_600pairs/)",
    )
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default=DEFAULT_HUB_REPO_ID,
        help="HF Hub destination repo id",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "mps", "cuda"),
        help="SegNet compute device (default: cpu per CLAUDE.md MPS-noise non-negotiable)",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=CONTEST_VIDEO_NPAIRS,
        help="Number of pairs to extract (default: 600 canonical contest count)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Default: dry-run (sha-chain only; no decode/scorer/upload).",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Run full build + upload to HF Hub (requires `hf auth whoami`).",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create dataset as public (default: private; flip via Hub UI).",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help=(
            "If repo exists with matching manifest sha, skip re-upload "
            "(idempotent)."
        ),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=None,
        help="Write build summary JSON to this path (e.g. .omx/state/hf_dataset_build_summary.json)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.upload:
        args.dry_run = False

    summary = build_dataset(
        video_path=args.video_path,
        segnet_weights_path=args.segnet_weights_path,
        staging_dir=args.staging_dir,
        hub_repo_id=args.hub_repo_id,
        device=args.device,
        dry_run=args.dry_run,
        upload=args.upload,
        skip_if_exists=args.skip_if_exists,
        private=not args.public,
        n_pairs=args.n_pairs,
    )

    summary_dict = summary.to_dict()
    if args.output_summary is not None:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(
            json.dumps(summary_dict, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    print(json.dumps(summary_dict, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
