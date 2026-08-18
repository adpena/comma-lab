"""Retain the per-pair distortion vectors the scorer computes and then throws away.

THE SIGNAL LOSS.  ``upstream/evaluate.py`` computes a per-pair PoseNet distortion of shape
``(B,)`` for every batch and immediately consumes it::

    posenet_dist, segnet_dist = distortion_net.compute_distortion(batch_gt, batch_comp)
    posenet_dists += posenet_dist.sum()          # <- the 600-vector dies on this line

600 numbers become one.  Every carrier arm that has argued about *where* pose cost lives has
had to argue from that single scalar, and at least three rows (``t1h`` pass-2, ``rr2``,
``#1054``) would have been readable as a per-pair map instead of a mystery.  This is the
ALWAYS-KEEP-THE-PAYLOAD law firing inside our own instrument.

THE HONEST COST — a correction to the premise this module was requested under.  The request
described this as "~14 KB and zero extra compute".  **It is not.**  The ``(B,)`` vector is
created and destroyed inside the upstream process; it never enters our address space, and
``upstream/`` is read-only.  The only legal retention is a SECOND pass that we own, which
re-runs PoseNet and SegNet over the same 600 pairs: measured **~40 s on contest-CUDA T4** and
**~214 s on contest-CPU** against the evaluate elapsed times in our harvested rows.  The
payload really is ~14 KB; the compute is not free.  Retention is therefore OPT-IN and default
OFF, and it runs strictly AFTER the scored result is in hand so it cannot perturb a number.

WHY THE SELF-CHECK IS THE POINT.  A second pass is only worth keeping if it is the SAME
computation.  So the retained vectors are reduced exactly as upstream reduces them — a plain
mean — and compared against the scalar upstream actually reported.  If the two disagree
beyond tolerance the retention is marked UNVERIFIED and says so in its manifest, because a
per-pair map that does not add up to the scored number is worse than no map at all.

AXIS.  Diagnostic.  This module never produces, adjusts, or implies a score.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = [
    "PerPairRetention",
    "RetentionVerification",
    "compute_per_pair_distortion",
    "retain_per_pair_distortion",
    "verify_reduction",
]

#: Relative tolerance for "our mean reproduces the reported scalar". fp32 accumulation over
#: 600 terms on a different device ordering will not be bit-identical; 1e-6 relative is tight
#: enough to catch a wrong loader and loose enough not to cry over summation order.
DEFAULT_REDUCTION_RTOL = 1e-6


@dataclass(frozen=True)
class RetentionVerification:
    """Does the retained per-pair vector reduce to the number the scorer reported?"""

    reported: float | None
    recomputed: float
    relative_error: float | None
    verified: bool
    note: str

    def to_dict(self) -> dict[str, object]:
        return {
            "reported": self.reported,
            "recomputed": self.recomputed,
            "relative_error": self.relative_error,
            "verified": self.verified,
            "note": self.note,
        }


@dataclass(frozen=True)
class PerPairRetention:
    """The retained payload and its receipt."""

    out_dir: Path
    pairs: int
    payload_paths: dict[str, str]
    payload_sha256: dict[str, str]
    payload_bytes: dict[str, int]
    pose_verification: RetentionVerification
    seg_verification: RetentionVerification

    @property
    def verified(self) -> bool:
        return self.pose_verification.verified and self.seg_verification.verified

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "pact.per_pair_distortion_retention.v1",
            "axis": "[diagnostic] per-pair distortion map; NEVER a score",
            "score_claim": False,
            "promotable": False,
            "out_dir": str(self.out_dir),
            "pairs": self.pairs,
            "payload_paths": self.payload_paths,
            "payload_sha256": self.payload_sha256,
            "payload_bytes": self.payload_bytes,
            "pose_verification": self.pose_verification.to_dict(),
            "seg_verification": self.seg_verification.to_dict(),
            "verified": self.verified,
        }


def verify_reduction(
    per_pair: Sequence[float],
    reported: float | None,
    label: str,
    rtol: float = DEFAULT_REDUCTION_RTOL,
) -> RetentionVerification:
    """Reduce exactly as upstream reduces, and compare against what upstream reported."""
    values = [float(value) for value in per_pair]
    if not values:
        return RetentionVerification(
            reported=reported,
            recomputed=float("nan"),
            relative_error=None,
            verified=False,
            note=f"{label}: no per-pair values were retained; nothing to verify",
        )
    recomputed = sum(values) / len(values)
    if reported is None:
        return RetentionVerification(
            reported=None,
            recomputed=recomputed,
            relative_error=None,
            verified=False,
            note=(
                f"{label}: UNVERIFIED — the scorer's reported scalar was not supplied, so this "
                "map is unanchored. Treat it as advisory shape only, never as levels."
            ),
        )
    denominator = abs(reported) if reported else 1.0
    relative_error = abs(recomputed - reported) / denominator
    verified = relative_error <= rtol
    note = (
        f"{label}: mean of {len(values)} retained pairs reproduces the reported scalar "
        f"(rel err {relative_error:.3e} <= {rtol:.0e})"
        if verified
        else (
            f"{label}: UNVERIFIED — mean of {len(values)} retained pairs is {recomputed!r} but the "
            f"scorer reported {reported!r} (rel err {relative_error:.3e} > {rtol:.0e}). The second "
            "pass is NOT reproducing the scored computation; do not read this map."
        )
    )
    return RetentionVerification(
        reported=reported,
        recomputed=recomputed,
        relative_error=relative_error,
        verified=verified,
        note=note,
    )


def _write_payload(out_dir: Path, name: str, array: Any) -> tuple[str, str, int]:
    import numpy as np

    path = out_dir / name
    np.save(path, np.asarray(array, dtype=np.float64))
    written = path if path.suffix == ".npy" else path.with_suffix(".npy")
    data = written.read_bytes()
    return str(written), hashlib.sha256(data).hexdigest(), len(data)


def retain_per_pair_distortion(
    out_dir: Path,
    per_pair_pose: Sequence[float],
    per_pair_seg: Sequence[float],
    reported_pose: float | None = None,
    reported_seg: float | None = None,
    pose_vectors: Any | None = None,
    rtol: float = DEFAULT_REDUCTION_RTOL,
) -> PerPairRetention:
    """Persist the per-pair vectors with their sha256s and their reduction check.

    Writing only the scalars while the vectors sat in memory is the measure-and-discard
    defect this module exists to end, so the payload is written before anything is reported.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    shas: dict[str, str] = {}
    sizes: dict[str, int] = {}
    payloads: list[tuple[str, Any]] = [
        ("per_pair_posenet_distortion.npy", per_pair_pose),
        ("per_pair_segnet_distortion.npy", per_pair_seg),
    ]
    if pose_vectors is not None:
        payloads.append(("per_pair_pose_vectors.npy", pose_vectors))
    for name, array in payloads:
        path, sha, size = _write_payload(out_dir, name, array)
        key = name.removesuffix(".npy")
        paths[key], shas[key], sizes[key] = path, sha, size

    retention = PerPairRetention(
        out_dir=out_dir,
        pairs=len(list(per_pair_pose)),
        payload_paths=paths,
        payload_sha256=shas,
        payload_bytes=sizes,
        pose_verification=verify_reduction(per_pair_pose, reported_pose, "posenet", rtol),
        seg_verification=verify_reduction(per_pair_seg, reported_seg, "segnet", rtol),
    )
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(retention.to_dict(), indent=2, sort_keys=True) + "\n")
    return retention


def compute_per_pair_distortion(
    upstream_dir: Path,
    submission_dir: Path,
    uncompressed_dir: Path,
    video_names_file: Path,
    device: str = "cpu",
    batch_size: int = 16,
    num_threads: int = 2,
    seed: int = 1234,
) -> tuple[list[float], list[float], Any]:
    """Re-run the scorer over the same pairs, keeping the per-pair vectors this time.

    This mirrors ``upstream/evaluate.py``'s loop deliberately, by IMPORTING upstream's own
    dataset and ``DistortionNet`` rather than reimplementing either: a reimplementation would
    be a different computation wearing the same name.  ``upstream/`` is not modified.

    It is a SECOND pass and it costs a second pass (~40 s T4 / ~214 s contest-CPU).  Callers
    must run it only after the scored result is already in hand.
    """
    import torch

    upstream_dir = Path(upstream_dir).resolve()
    if str(upstream_dir) not in sys.path:
        sys.path.insert(0, str(upstream_dir))

    from frame_utils import AVVideoDataset, TensorVideoDataset
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    torch_device = torch.device(device)
    distortion_net = DistortionNet(segnet_sd_path, posenet_sd_path).to(torch_device).eval()

    names = [line.strip() for line in Path(video_names_file).read_text().splitlines() if line.strip()]
    common = {
        "batch_size": batch_size,
        "device": torch_device,
        "num_threads": num_threads,
        "seed": seed,
    }
    ds_gt = AVVideoDataset(names, data_dir=Path(uncompressed_dir), **common)
    ds_gt.prepare_data()
    ds_comp = TensorVideoDataset(names, data_dir=Path(submission_dir) / "inflated", **common)
    ds_comp.prepare_data()

    dl_gt = torch.utils.data.DataLoader(ds_gt, batch_size=None, num_workers=0)
    dl_comp = torch.utils.data.DataLoader(ds_comp, batch_size=None, num_workers=0)

    per_pair_pose: list[float] = []
    per_pair_seg: list[float] = []
    with torch.inference_mode():
        for (_, _, batch_gt), (_, _, batch_comp) in zip(dl_gt, dl_comp, strict=False):
            batch_gt = batch_gt.to(torch_device)
            batch_comp = batch_comp.to(torch_device)
            pose_dist, seg_dist = distortion_net.compute_distortion(batch_gt, batch_comp)
            per_pair_pose.extend(float(value) for value in pose_dist.reshape(-1).cpu())
            per_pair_seg.extend(float(value) for value in seg_dist.reshape(-1).cpu())

    return per_pair_pose, per_pair_seg, None
