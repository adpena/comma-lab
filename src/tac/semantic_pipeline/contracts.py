# SPDX-License-Identifier: MIT
"""Typed, payload-retaining stage contracts for the from-video pipeline."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

MIN_FREE_BYTES = int(1.5 * 1024**3)
PAIR_SEQUENCE_LENGTH = 2


class PipelineBlocked(RuntimeError):
    """A fail-closed pipeline boundary with an operator-actionable reason."""


@dataclasses.dataclass(frozen=True)
class ClipConfig:
    video: Path
    frame_count: int
    width: int
    height: int
    sequence_length: int
    pair_count: int
    trailing_frames: int
    first_frame_rgb_sha256: str

    def as_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["video"] = str(self.video)
        return result


@dataclasses.dataclass(frozen=True)
class DeviceBinding:
    requested: str
    torch_device: str
    available: bool
    tested_here: bool
    mps_patches: Mapping[str, bool] | None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class StageReceipt:
    stage: str
    status: str
    device: str
    inputs: tuple[Mapping[str, Any], ...]
    outputs: tuple[Mapping[str, Any], ...]
    elapsed_seconds: float
    resumed: bool
    resumable: bool
    seed: int
    config: Mapping[str, Any]
    non_negotiables: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    if temporary.exists():
        raise PipelineBlocked(f"stale atomic-write sibling exists: {temporary}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_copy(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if sha256_file(source) == sha256_file(destination):
            return file_fact(destination)
        raise PipelineBlocked(f"refusing to overwrite a different retained payload: {destination}")
    temporary = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
    if temporary.exists():
        raise PipelineBlocked(f"stale atomic-copy sibling exists: {temporary}")
    with source.open("rb") as src, temporary.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=8 << 20)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(temporary, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if sha256_file(source) != sha256_file(destination):
        raise PipelineBlocked(f"atomic copy changed bytes: {source} -> {destination}")
    return file_fact(destination)


def require_storage(root: Path, minimum_free_bytes: int = MIN_FREE_BYTES) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    if usage.free < minimum_free_bytes:
        raise PipelineBlocked(f"storage preflight refused {root}: {usage.free} free bytes, need {minimum_free_bytes}")
    probe = root / f".write_probe.{os.getpid()}"
    try:
        probe.write_bytes(b"ddm_fpc1\n")
        probe.unlink()
    except PermissionError as exc:
        raise PipelineBlocked(f"retention tier is not writable: {root}") from exc
    return {
        "root": str(root.resolve()),
        "free_bytes": usage.free,
        "minimum_free_bytes": minimum_free_bytes,
        "status": "PASS",
    }


def probe_clip(video: Path) -> ClipConfig:
    """Probe a real clip and route the decoded sample through the contest converter."""

    video = video.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    import av

    upstream_frame_utils = Path(__file__).resolve().parents[3] / "upstream" / "frame_utils.py"
    spec = importlib.util.spec_from_file_location("fpc1_upstream_frame_utils", upstream_frame_utils)
    if spec is None or spec.loader is None:
        raise PipelineBlocked(f"cannot load contest frame converter: {upstream_frame_utils}")
    frame_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frame_utils)

    frame_count = int(frame_utils.frame_count(str(video)))
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        try:
            frame = next(container.decode(stream))
        except StopIteration as exc:
            raise PipelineBlocked(f"video has no decodable frames: {video}") from exc
        rgb = frame_utils.yuv420_to_rgb(frame)
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        first_frame_sha = hashlib.sha256(memoryview(rgb.numpy()).cast("B")).hexdigest()
    if frame_count < PAIR_SEQUENCE_LENGTH:
        raise PipelineBlocked(f"video has only {frame_count} frames")
    return ClipConfig(
        video=video,
        frame_count=frame_count,
        width=width,
        height=height,
        sequence_length=PAIR_SEQUENCE_LENGTH,
        pair_count=frame_count // PAIR_SEQUENCE_LENGTH,
        trailing_frames=frame_count % PAIR_SEQUENCE_LENGTH,
        first_frame_rgb_sha256=first_frame_sha,
    )


def require_device(requested: str) -> DeviceBinding:
    """Bind exactly the requested torch device; never select a fallback."""

    if requested not in {"cpu", "mps", "cuda"}:
        raise ValueError(f"unsupported device {requested!r}")
    import torch

    patches: Mapping[str, bool] | None = None
    if requested == "cpu":
        available = True
    elif requested == "mps":
        available = bool(torch.backends.mps.is_available())
        if available:
            from tac.torch_mps_compat import patch_scorer_for_mps

            patches = patch_scorer_for_mps()
    else:
        available = bool(torch.cuda.is_available())
    if not available:
        raise PipelineBlocked(f"requested torch device is unavailable: {requested}")
    return DeviceBinding(
        requested=requested,
        torch_device=requested,
        available=True,
        tested_here=requested == "cpu",
        mps_patches=patches,
    )


def git_sha(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def stage_receipt_path(store: Path, ordinal: int, stage: str) -> Path:
    return store / "receipts" / f"{ordinal:02d}_{stage}.json"


def load_valid_receipt(
    receipt_path: Path,
    *,
    stage: str,
    expected_inputs: Sequence[Path],
    expected_outputs: Sequence[Path],
    device: str,
    seed: int,
    config: Mapping[str, Any],
    non_negotiables: Mapping[str, Any],
) -> StageReceipt | None:
    if not receipt_path.is_file():
        return None
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if payload.get("stage") != stage or payload.get("status") != "PASS":
        return None
    if (
        payload.get("device") != device
        or int(payload.get("seed", -1)) != seed
        or payload.get("config") != dict(config)
        or payload.get("non_negotiables") != dict(non_negotiables)
    ):
        return None
    expected = [file_fact(path) for path in expected_inputs]
    if payload.get("inputs") != expected:
        return None
    outputs = payload.get("outputs", [])
    if [Path(output["path"]).resolve() for output in outputs] != [path.resolve() for path in expected_outputs]:
        return None
    for output in outputs:
        path = Path(output["path"])
        if not path.is_file() or file_fact(path) != output:
            return None
    return StageReceipt(
        stage=payload["stage"],
        status=payload["status"],
        device=payload["device"],
        inputs=tuple(payload["inputs"]),
        outputs=tuple(payload["outputs"]),
        elapsed_seconds=float(payload["elapsed_seconds"]),
        resumed=True,
        resumable=bool(payload["resumable"]),
        seed=int(payload["seed"]),
        config=payload["config"],
        non_negotiables=payload["non_negotiables"],
    )


def run_payload_stage(
    *,
    store: Path,
    ordinal: int,
    stage: str,
    device: str,
    seed: int,
    inputs: Sequence[Path],
    outputs: Sequence[Path],
    config: Mapping[str, Any],
    non_negotiables: Mapping[str, Any],
    action: Callable[[], None],
    resume: bool,
) -> StageReceipt:
    """Run one stage or validate and reuse its byte-identical durable boundary."""

    receipt_path = stage_receipt_path(store, ordinal, stage)
    if resume:
        loaded = load_valid_receipt(
            receipt_path,
            stage=stage,
            expected_inputs=inputs,
            expected_outputs=outputs,
            device=device,
            seed=seed,
            config=config,
            non_negotiables=non_negotiables,
        )
        if loaded is not None:
            return loaded
    started = time.monotonic()
    action()
    missing = [path for path in outputs if not path.is_file()]
    if missing:
        raise PipelineBlocked(f"stage {stage} did not retain outputs: {missing}")
    receipt = StageReceipt(
        stage=stage,
        status="PASS",
        device=device,
        inputs=tuple(file_fact(path) for path in inputs),
        outputs=tuple(file_fact(path) for path in outputs),
        elapsed_seconds=time.monotonic() - started,
        resumed=False,
        resumable=True,
        seed=seed,
        config=dict(config),
        non_negotiables=dict(non_negotiables),
    )
    atomic_json(receipt_path, receipt.as_dict())
    return receipt


def host_provenance(repo: Path) -> dict[str, Any]:
    import torch

    return {
        "git_sha": git_sha(repo),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hostname": platform.node(),
    }
