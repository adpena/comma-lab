#!/usr/bin/env python3
"""Exact per-sample component tracing for contest archives.

This is a diagnostic companion to ``experiments/contest_auth_eval.py``. It
does not modify the pinned upstream scorer. Instead, it mirrors the scorer's
DistortionNet loop and preserves the per-pair PoseNet and SegNet tensors that
``upstream/evaluate.py`` sums into averages.

The output is explicitly non-promotable by itself. It becomes scientifically
useful when its averages match a canonical CUDA ``contest_auth_eval.json`` for
the same archive bytes, after which the per-sample records can drive hard-pair
selection, Lagrangian water filling, and repair-atom allocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.contest_auth_eval import (
    _ensure_uv_available,
    _extract_archive,
    _run_inflate,
    _sha256,
    _validate_archive_members,
    _validate_uncompressed_dir,
)


SCHEMA_VERSION = 1
EXPECTED_CONTEST_SAMPLES = 600
REQUIRED_FFMPEG_SCALE_OPTIONS = (
    "in_range",
    "out_range",
    "in_color_matrix",
    "in_primaries",
    "in_transfer",
)
PARITY_FFMPEG_CANDIDATE_PATHS = (
    Path("/workspace/ffmpeg-btbn/bin/ffmpeg"),
    Path("/opt/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg"),
    Path("/usr/local/bin/ffmpeg-master"),
    Path("/usr/local/bin/ffmpeg-new"),
)


@dataclass(frozen=True)
class ComponentSample:
    pair_index: int
    video_name: str
    video_pair_index: int
    frame_start: int
    posenet_dist: float
    segnet_dist: float

    def to_json(self) -> dict[str, Any]:
        return {
            "pair_index": self.pair_index,
            "video_name": self.video_name,
            "video_pair_index": self.video_pair_index,
            "frame_start": self.frame_start,
            "frame_indices": [self.frame_start, self.frame_start + 1],
            "posenet_dist": self.posenet_dist,
            "segnet_dist": self.segnet_dist,
        }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _resolve_executable(candidate: str | Path) -> Path | None:
    raw = str(candidate)
    path = Path(raw)
    if path.is_file() and os.access(path, os.X_OK):
        return path.resolve()
    resolved = shutil.which(raw)
    if resolved:
        return Path(resolved).resolve()
    return None


def _ffmpeg_missing_scale_options(ffmpeg: Path) -> list[str]:
    try:
        proc = subprocess.run(
            [str(ffmpeg), "-hide_banner", "-h", "filter=scale"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        return [f"probe_failed:{exc!r}"]
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return [opt for opt in REQUIRED_FFMPEG_SCALE_OPTIONS if opt not in text]


def _ensure_parity_ffmpeg_env() -> Path:
    """Set FFMPEG_BIN to a parity-compatible binary before archive tracing.

    ``scripts/remote_archive_only_eval.sh`` self-bootstraps BtbN ffmpeg, but
    this diagnostic can also be run standalone.  Standalone archive traces must
    not silently fall back to a system ffmpeg missing the explicit color
    contract options used by ``submissions/robust_current/inflate.sh``.
    """
    explicit = os.environ.get("FFMPEG_BIN")
    if explicit:
        resolved = _resolve_executable(explicit)
        if resolved is None:
            raise RuntimeError(f"FFMPEG_BIN={explicit!r} is not executable")
        missing = _ffmpeg_missing_scale_options(resolved)
        if missing:
            raise RuntimeError(
                f"FFMPEG_BIN={resolved} is not parity-compatible; missing "
                f"scale option(s): {missing}"
            )
        os.environ["FFMPEG_BIN"] = str(resolved)
        return resolved

    candidates = list(PARITY_FFMPEG_CANDIDATE_PATHS)
    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        candidates.append(Path(path_ffmpeg))
    rejected: list[str] = []
    for candidate in candidates:
        resolved = _resolve_executable(candidate)
        if resolved is None:
            continue
        missing = _ffmpeg_missing_scale_options(resolved)
        if not missing:
            os.environ["FFMPEG_BIN"] = str(resolved)
            return resolved
        rejected.append(f"{resolved}: missing {missing}")

    details = "; ".join(rejected) if rejected else "no executable candidates found"
    raise RuntimeError(
        "contest_component_trace archive mode requires a parity-compatible "
        "ffmpeg with explicit scale color-contract options. Source "
        "scripts/remote_archive_only_eval.sh bootstrap, install BtbN ffmpeg, "
        f"or set FFMPEG_BIN. Rejected candidates: {details}"
    )


def _ensure_isolated_inflate_uv_env(work_dir: Path) -> Path:
    """Keep inflate-side ``uv run`` from mutating the repo/shared venv."""
    env = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if env:
        return Path(env).resolve()
    uv_env = (work_dir / "uv_project_env").resolve()
    os.environ["UV_PROJECT_ENVIRONMENT"] = str(uv_env)
    os.environ.setdefault("UV_LINK_MODE", "copy")
    return uv_env


def _score_components(
    *,
    avg_posenet_dist: float,
    avg_segnet_dist: float,
    archive_size_bytes: int,
    uncompressed_size_bytes: int,
) -> dict[str, float]:
    rate_unscaled = archive_size_bytes / uncompressed_size_bytes
    score_pose = math.sqrt(10.0 * avg_posenet_dist)
    score_seg = 100.0 * avg_segnet_dist
    score_rate = 25.0 * rate_unscaled
    return {
        "rate_unscaled": rate_unscaled,
        "score_pose_contribution": score_pose,
        "score_seg_contribution": score_seg,
        "score_rate_contribution": score_rate,
        "score_recomputed_from_components": score_pose + score_seg + score_rate,
    }


def _first_order_pose_contribution(posenet_dist: float, *, avg_posenet_dist: float, n: int) -> float:
    if avg_posenet_dist <= 0.0:
        return 0.0
    return (5.0 / math.sqrt(10.0 * avg_posenet_dist)) * (posenet_dist / n)


def _annotate_sample(
    sample: ComponentSample,
    *,
    avg_posenet_dist: float,
    n: int,
) -> dict[str, Any]:
    out = sample.to_json()
    score_seg = 100.0 * sample.segnet_dist / n
    score_pose = _first_order_pose_contribution(
        sample.posenet_dist,
        avg_posenet_dist=avg_posenet_dist,
        n=n,
    )
    out["score_seg_contribution_exact"] = score_seg
    out["score_pose_contribution_first_order"] = score_pose
    out["score_combined_contribution_first_order"] = score_seg + score_pose
    return out


def summarize_samples(
    samples: list[ComponentSample],
    *,
    archive_size_bytes: int,
    uncompressed_size_bytes: int,
    top_k: int,
    baseline_samples: list[ComponentSample] | None = None,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("component trace has no samples")
    n = len(samples)
    avg_pose = sum(s.posenet_dist for s in samples) / n
    avg_seg = sum(s.segnet_dist for s in samples) / n
    score = _score_components(
        avg_posenet_dist=avg_pose,
        avg_segnet_dist=avg_seg,
        archive_size_bytes=archive_size_bytes,
        uncompressed_size_bytes=uncompressed_size_bytes,
    )
    annotated = [
        _annotate_sample(sample, avg_posenet_dist=avg_pose, n=n)
        for sample in samples
    ]
    k = max(0, min(top_k, n))

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "score_claim": False,
        "evidence_grade": "diagnostic_component_trace",
        "promotion_policy": (
            "Non-promotable by itself; use only after averages match canonical "
            "CUDA contest_auth_eval.json for identical archive bytes."
        ),
        "n_samples": n,
        "expected_contest_samples": EXPECTED_CONTEST_SAMPLES,
        "avg_posenet_dist": avg_pose,
        "avg_segnet_dist": avg_seg,
        "archive_size_bytes": archive_size_bytes,
        "uncompressed_size_bytes": uncompressed_size_bytes,
        **score,
        "ranking_note": (
            "SegNet per-sample contribution is additive. PoseNet contribution "
            "uses the first-order derivative of sqrt(10*mean_pose) at the "
            "candidate mean, so it is a ranking signal rather than an exact "
            "additive decomposition."
        ),
        "top_pose_samples": sorted(
            annotated,
            key=lambda r: r["score_pose_contribution_first_order"],
            reverse=True,
        )[:k],
        "top_seg_samples": sorted(
            annotated,
            key=lambda r: r["score_seg_contribution_exact"],
            reverse=True,
        )[:k],
        "top_combined_samples": sorted(
            annotated,
            key=lambda r: r["score_combined_contribution_first_order"],
            reverse=True,
        )[:k],
        "samples": annotated,
    }

    if baseline_samples is not None:
        summary["delta_from_baseline"] = summarize_delta_from_baseline(
            samples,
            baseline_samples,
            candidate_avg_posenet_dist=avg_pose,
            top_k=k,
        )
    return summary


def summarize_delta_from_baseline(
    candidate_samples: list[ComponentSample],
    baseline_samples: list[ComponentSample],
    *,
    candidate_avg_posenet_dist: float,
    top_k: int,
) -> dict[str, Any]:
    baseline_by_pair = {sample.pair_index: sample for sample in baseline_samples}
    if len(baseline_by_pair) != len(baseline_samples):
        raise ValueError("baseline trace has duplicate pair_index entries")
    n = len(candidate_samples)
    deltas: list[dict[str, Any]] = []
    for sample in candidate_samples:
        baseline = baseline_by_pair.get(sample.pair_index)
        if baseline is None:
            raise ValueError(f"baseline trace missing pair_index={sample.pair_index}")
        delta_pose = sample.posenet_dist - baseline.posenet_dist
        delta_seg = sample.segnet_dist - baseline.segnet_dist
        score_seg_excess = 100.0 * delta_seg / n
        score_pose_excess = _first_order_pose_contribution(
            delta_pose,
            avg_posenet_dist=candidate_avg_posenet_dist,
            n=n,
        )
        deltas.append(
            {
                "pair_index": sample.pair_index,
                "video_name": sample.video_name,
                "video_pair_index": sample.video_pair_index,
                "frame_start": sample.frame_start,
                "delta_posenet_dist": delta_pose,
                "delta_segnet_dist": delta_seg,
                "score_seg_excess_exact": score_seg_excess,
                "score_pose_excess_first_order": score_pose_excess,
                "score_combined_excess_first_order": score_seg_excess + score_pose_excess,
                "candidate_posenet_dist": sample.posenet_dist,
                "candidate_segnet_dist": sample.segnet_dist,
                "baseline_posenet_dist": baseline.posenet_dist,
                "baseline_segnet_dist": baseline.segnet_dist,
            }
        )
    return {
        "baseline_n_samples": len(baseline_samples),
        "top_excess_pose_samples": sorted(
            deltas,
            key=lambda r: r["score_pose_excess_first_order"],
            reverse=True,
        )[:top_k],
        "top_excess_seg_samples": sorted(
            deltas,
            key=lambda r: r["score_seg_excess_exact"],
            reverse=True,
        )[:top_k],
        "top_excess_combined_samples": sorted(
            deltas,
            key=lambda r: r["score_combined_excess_first_order"],
            reverse=True,
        )[:top_k],
    }


def load_trace_samples(path: Path) -> list[ComponentSample]:
    payload = json.loads(path.read_text())
    samples = []
    for raw in payload.get("samples", []):
        samples.append(
            ComponentSample(
                pair_index=int(raw["pair_index"]),
                video_name=str(raw.get("video_name", "")),
                video_pair_index=int(raw.get("video_pair_index", raw["pair_index"])),
                frame_start=int(raw.get("frame_start", 2 * int(raw["pair_index"]))),
                posenet_dist=float(raw["posenet_dist"]),
                segnet_dist=float(raw["segnet_dist"]),
            )
        )
    return samples


def compare_to_contest_json(
    trace: dict[str, Any],
    contest_json_path: Path,
    *,
    avg_tolerance: float,
    score_tolerance: float,
) -> dict[str, Any]:
    contest = json.loads(contest_json_path.read_text())
    checks = {
        "n_samples": {
            "trace": trace["n_samples"],
            "contest": contest.get("n_samples"),
            "match": int(trace["n_samples"]) == int(contest.get("n_samples")),
        },
        "archive_size_bytes": {
            "trace": trace["archive_size_bytes"],
            "contest": contest.get("archive_size_bytes"),
            "match": int(trace["archive_size_bytes"]) == int(contest.get("archive_size_bytes")),
        },
    }
    for key in ("avg_posenet_dist", "avg_segnet_dist"):
        trace_val = float(trace[key])
        contest_val = float(contest[key])
        checks[key] = {
            "trace": trace_val,
            "contest": contest_val,
            "abs_diff": abs(trace_val - contest_val),
            "tolerance": avg_tolerance,
            "match": abs(trace_val - contest_val) <= avg_tolerance,
        }
    trace_score = float(trace["score_recomputed_from_components"])
    contest_score = float(contest.get("score_recomputed_from_components", contest.get("final_score")))
    checks["score_recomputed_from_components"] = {
        "trace": trace_score,
        "contest": contest_score,
        "abs_diff": abs(trace_score - contest_score),
        "tolerance": score_tolerance,
        "match": abs(trace_score - contest_score) <= score_tolerance,
    }
    return {
        "contest_auth_eval_json": str(contest_json_path),
        "contest_auth_eval_json_sha256": sha256_file(contest_json_path),
        "all_match": all(item["match"] for item in checks.values()),
        "checks": checks,
    }


def _import_upstream(upstream_dir: Path) -> dict[str, Any]:
    upstream_str = str(upstream_dir.resolve())
    if upstream_str not in sys.path:
        sys.path.insert(0, upstream_str)
    from frame_utils import (  # type: ignore[import-not-found]
        AVVideoDataset,
        DaliVideoDataset,
        TensorVideoDataset,
        camera_size,
        seq_len,
    )
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # type: ignore[import-not-found]

    return {
        "AVVideoDataset": AVVideoDataset,
        "DaliVideoDataset": DaliVideoDataset,
        "TensorVideoDataset": TensorVideoDataset,
        "camera_size": camera_size,
        "seq_len": seq_len,
        "DistortionNet": DistortionNet,
        "posenet_sd_path": posenet_sd_path,
        "segnet_sd_path": segnet_sd_path,
    }


def trace_submission_dir(
    *,
    submission_dir: Path,
    upstream_dir: Path,
    uncompressed_dir: Path,
    video_names_file: Path,
    device_name: str,
    batch_size: int,
    num_threads: int,
    seed: int,
    prefetch_queue_depth: int,
    top_k: int,
    baseline_trace_json: Path | None,
) -> dict[str, Any]:
    _validate_uncompressed_dir(uncompressed_dir, video_names_file)
    archive_path = submission_dir / "archive.zip"
    if not archive_path.exists():
        raise FileNotFoundError(f"submission_dir missing archive.zip: {archive_path}")

    upstream = _import_upstream(upstream_dir)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    if device.type == "cuda":
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        if device.index is None:
            device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        DefaultDatasetClass = upstream["DaliVideoDataset"]
    else:
        DefaultDatasetClass = upstream["AVVideoDataset"]

    distortion_net = upstream["DistortionNet"]().eval().to(device=device)
    distortion_net.load_state_dicts(upstream["posenet_sd_path"], upstream["segnet_sd_path"], device)

    test_video_names = [line.strip() for line in video_names_file.read_text().splitlines() if line.strip()]
    ds_gt = DefaultDatasetClass(
        test_video_names,
        data_dir=uncompressed_dir,
        batch_size=batch_size,
        device=device,
        num_threads=num_threads,
        seed=seed,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    ds_gt.prepare_data()
    dl_gt = torch.utils.data.DataLoader(ds_gt, batch_size=None, num_workers=0)

    ds_comp = upstream["TensorVideoDataset"](
        test_video_names,
        data_dir=submission_dir / "inflated",
        batch_size=batch_size,
        device=device,
        num_threads=num_threads,
        seed=seed,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    ds_comp.prepare_data()
    dl_comp = torch.utils.data.DataLoader(ds_comp, batch_size=None, num_workers=0)

    seq_len = int(upstream["seq_len"])
    camera_size = upstream["camera_size"]
    samples: list[ComponentSample] = []
    with torch.inference_mode():
        for (path_gt, batch_idx_gt, batch_gt), (path_comp, batch_idx_comp, batch_comp) in tqdm(zip(dl_gt, dl_comp)):
            batch_gt = batch_gt.to(device)
            batch_comp = batch_comp.to(device)
            expected_shape = [seq_len, camera_size[1], camera_size[0], 3]
            if list(batch_comp.shape)[1:] != expected_shape:
                raise RuntimeError(f"unexpected compressed batch shape: {batch_comp.shape}")
            if batch_gt.shape != batch_comp.shape:
                raise RuntimeError(f"ground truth and compressed batch shape mismatch: {batch_gt.shape} vs {batch_comp.shape}")
            posenet_dist, segnet_dist = distortion_net.compute_distortion(batch_gt, batch_comp)
            if posenet_dist.shape != (batch_gt.shape[0],) or segnet_dist.shape != (batch_gt.shape[0],):
                raise RuntimeError(f"unexpected distortion shapes: {posenet_dist.shape}, {segnet_dist.shape}")

            try:
                video_name = str(Path(path_gt).resolve().relative_to(uncompressed_dir.resolve()))
            except ValueError:
                video_name = Path(path_gt).name
            batch_start_pair = (int(batch_idx_gt) - 1) * batch_size
            pose_cpu = posenet_dist.detach().cpu().tolist()
            seg_cpu = segnet_dist.detach().cpu().tolist()
            for j, (pose_val, seg_val) in enumerate(zip(pose_cpu, seg_cpu)):
                video_pair_index = batch_start_pair + j
                samples.append(
                    ComponentSample(
                        pair_index=len(samples),
                        video_name=video_name,
                        video_pair_index=video_pair_index,
                        frame_start=video_pair_index * seq_len,
                        posenet_dist=float(pose_val),
                        segnet_dist=float(seg_val),
                    )
                )

    uncompressed_size = sum(file.stat().st_size for file in uncompressed_dir.rglob("*") if file.is_file())
    baseline_samples = load_trace_samples(baseline_trace_json) if baseline_trace_json else None
    summary = summarize_samples(
        samples,
        archive_size_bytes=archive_path.stat().st_size,
        uncompressed_size_bytes=uncompressed_size,
        top_k=top_k,
        baseline_samples=baseline_samples,
    )
    summary["trace_inputs"] = {
        "submission_dir": str(submission_dir),
        "archive_sha256": _sha256(archive_path, prefix=0),
        "upstream_dir": str(upstream_dir),
        "uncompressed_dir": str(uncompressed_dir),
        "video_names_file": str(video_names_file),
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_device_index": device.index if device.type == "cuda" else None,
        "cuda_device_name": (
            torch.cuda.get_device_name(device.index if device.index is not None else torch.cuda.current_device())
            if device.type == "cuda"
            else None
        ),
        "cuda_device_capability": (
            list(torch.cuda.get_device_capability(device.index if device.index is not None else torch.cuda.current_device()))
            if device.type == "cuda"
            else None
        ),
        "batch_size": batch_size,
        "num_threads": num_threads,
        "seed": seed,
        "prefetch_queue_depth": prefetch_queue_depth,
        "baseline_trace_json": str(baseline_trace_json) if baseline_trace_json else None,
    }
    return summary


def _validate_renderer_inflate_contract(inflate_sh: Path) -> None:
    config_env = inflate_sh.parent / "config.env"
    if inflate_sh.parent.name != "robust_current" and not config_env.exists():
        return
    if not config_env.exists():
        raise RuntimeError(
            f"FATAL: {config_env} missing; robust_current inflate.sh would use "
            "the wrong path for renderer archives."
        )
    if "PYTHON_INFLATE=renderer" not in config_env.read_text():
        raise RuntimeError(
            f"FATAL: {config_env} does not set PYTHON_INFLATE=renderer."
        )


def _prepare_archive_submission(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.archive is None:
        assert args.submission_dir is not None
        return args.submission_dir.resolve(), False
    if args.output_json is None and args.work_dir is None and not args.keep_work_dir:
        raise SystemExit(
            "--archive mode without --output-json requires --work-dir or --keep-work-dir "
            "so the diagnostic JSON is not deleted with the temporary work dir."
        )

    archive = args.archive.resolve()
    inflate_sh = args.inflate_sh.resolve()
    if not archive.exists():
        raise FileNotFoundError(f"--archive does not exist: {archive}")
    if not inflate_sh.exists():
        raise FileNotFoundError(f"--inflate-sh does not exist: {inflate_sh}")
    _validate_renderer_inflate_contract(inflate_sh)
    _ensure_uv_available()
    ffmpeg = _ensure_parity_ffmpeg_env()

    if args.work_dir:
        work_dir = args.work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="contest_component_trace_"))
        cleanup = not args.keep_work_dir

    uv_env = _ensure_isolated_inflate_uv_env(work_dir)
    archive_in_work = work_dir / "archive.zip"
    shutil.copy2(archive, archive_in_work)
    extracted = work_dir / "extracted"
    members = _extract_archive(archive_in_work, extracted)
    _validate_archive_members(members)
    _run_inflate(
        inflate_sh,
        extracted,
        work_dir / "inflated",
        args.video_names_file.resolve(),
        timeout=args.inflate_timeout,
    )
    (work_dir / "component_trace_runtime_env.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ffmpeg_bin": str(ffmpeg),
                "uv_project_environment": str(uv_env),
                "uv_link_mode": os.environ.get("UV_LINK_MODE"),
                "inflate_torch_spec": os.environ.get("INFLATE_TORCH_SPEC"),
                "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return work_dir, cleanup


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="Contest archive.zip to inflate and trace")
    source.add_argument("--submission-dir", type=Path, help="Existing contest-shaped dir with archive.zip and inflated/")
    parser.add_argument("--inflate-sh", type=Path, default=Path("submissions/robust_current/inflate.sh"))
    parser.add_argument("--upstream-dir", type=Path, default=Path("upstream"))
    parser.add_argument("--uncompressed-dir", type=Path, default=Path("upstream/videos"))
    parser.add_argument("--video-names-file", type=Path, default=Path("upstream/public_test_video_names.txt"))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--prefetch-queue-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--baseline-trace-json", type=Path, default=None)
    parser.add_argument("--contest-auth-eval-json", type=Path, default=None)
    parser.add_argument("--avg-tolerance", type=float, default=1e-5)
    parser.add_argument("--score-tolerance", type=float, default=1e-5)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--inflate-timeout", type=int, default=1800)
    args = parser.parse_args()

    upstream_dir = args.upstream_dir.resolve()
    if not (upstream_dir / "evaluate.py").exists():
        raise SystemExit(f"--upstream-dir missing evaluate.py: {upstream_dir}")
    video_names_file = args.video_names_file.resolve()
    if not video_names_file.exists():
        alt = upstream_dir / "public_test_video_names.txt"
        if alt.exists():
            video_names_file = alt
            args.video_names_file = alt
        else:
            raise SystemExit(f"--video-names-file does not exist: {video_names_file}")

    trace_started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    submission_dir, cleanup = _prepare_archive_submission(args)
    try:
        out_json = args.output_json.resolve() if args.output_json else submission_dir / "component_trace.json"
        trace = trace_submission_dir(
            submission_dir=submission_dir,
            upstream_dir=upstream_dir,
            uncompressed_dir=args.uncompressed_dir.resolve(),
            video_names_file=video_names_file,
            device_name=args.device,
            batch_size=args.batch_size,
            num_threads=args.num_threads,
            seed=args.seed,
            prefetch_queue_depth=args.prefetch_queue_depth,
            top_k=args.top_k,
            baseline_trace_json=args.baseline_trace_json.resolve() if args.baseline_trace_json else None,
        )
        trace["trace_started_at_utc"] = trace_started
        trace["trace_finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if args.contest_auth_eval_json:
            trace["contest_auth_eval_cross_check"] = compare_to_contest_json(
                trace,
                args.contest_auth_eval_json.resolve(),
                avg_tolerance=args.avg_tolerance,
                score_tolerance=args.score_tolerance,
            )
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(trace, indent=2) + "\n")
        print(f"RESULT_JSON: {json.dumps({'component_trace_json': str(out_json), 'n_samples': trace['n_samples'], 'avg_posenet_dist': trace['avg_posenet_dist'], 'avg_segnet_dist': trace['avg_segnet_dist'], 'score_recomputed_from_components': trace['score_recomputed_from_components']})}")
        print(f"component trace written: {out_json}")
        return 0
    finally:
        if cleanup:
            shutil.rmtree(submission_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
