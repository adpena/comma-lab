# SPDX-License-Identifier: MIT
"""Bounded, parameterized copy of the shipped CPR1 + F26 receiver.

The public submission remains immutable and pins one promoted archive.  This
module copies its receiver sources into the run store, records every copied
source by SHA-256 and byte count, and applies the smallest prefix-only repair:
the already-retained Python-decoder token checkpoint may be consumed for at
most eight pairs.  Full runs remain CUDA-only and use the unshortened Python
token decoder.  The contradiction in the shipped source is deliberate here:
the shipped file simultaneously rejects native HPAC and requires it for a
prefix.  Only the run-local copy is changed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import PipelineBlocked, atomic_json, file_fact, require_device

REPO = Path(__file__).resolve().parents[3]
SHIPPED = REPO / "submissions" / "semantic_joint_ctxmix"
TOKEN_FIELD = Path(
    "/Volumes/VertigoDataTier/pact/ddm_g8s_single_run_reproof/"
    "store_v2/retained/inputs/tokens.u8"
)
TOKEN_FIELD_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
TOKEN_FIELD_BYTES = 117_964_800
TOKEN_STREAM_SHA256 = "5601d6fd792c60c176e7cb7478e6033c4ed9a7e87404582340ed3f50ed60cfe3"
PAIR_BYTES = 384 * 512
RAW_PAIR_BYTES = 2 * 874 * 1164 * 3
MAX_CPU_PAIRS = 8


@dataclasses.dataclass(frozen=True)
class ReceiverRequest:
    archive: Path
    archive_sha256: str
    archive_bytes: int
    destination: Path
    runtime_root: Path
    checkpoint_dir: Path
    device: str
    pair_count: int

    def __post_init__(self) -> None:
        if self.pair_count < 1 or self.pair_count > 600:
            raise ValueError("receiver pair_count must be in [1, 600]")
        if self.device == "cpu" and self.pair_count > MAX_CPU_PAIRS:
            raise PipelineBlocked("pipeline CPU receiver is limited to at most 8 pairs")
        if self.device == "mps":
            raise PipelineBlocked("MPS is a gradient device, never a receiver authority")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            record = file_fact(path)
            record["relative_path"] = path.relative_to(root).as_posix()
            records.append(record)
    return records


def _replace_once(path: Path, old: str, new: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise PipelineBlocked(f"receiver-copy patch anchor count changed for {path}")
    before = file_fact(path)
    path.write_text(text.replace(old, new), encoding="utf-8")
    return {"before": before, "after": file_fact(path), "replacement_sha256": _sha256_bytes(new.encode())}


def materialize_runtime_copy(destination: Path) -> dict[str, Any]:
    """Copy the frozen shipped runtime and apply two prefix-only source edits."""

    destination = destination.resolve()
    manifest_path = destination / "PIPELINE_RECEIVER_COPY.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in prior["copied_sources_after_patch"]:
            path = destination / record["relative_path"]
            if file_fact(path)["sha256"] != record["sha256"]:
                raise PipelineBlocked("existing pipeline receiver copy drifted")
        return prior
    if destination.exists() and any(destination.iterdir()):
        raise PipelineBlocked(f"refusing nonempty receiver-copy destination: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    before: list[dict[str, Any]] = []
    for name in ("cpr1", "runtime"):
        source = SHIPPED / name
        source_records = _source_records(source)
        for record in source_records:
            record["relative_path"] = f"{name}/{record['relative_path']}"
        before.extend(source_records)
        shutil.copytree(
            source,
            destination / name,
            dirs_exist_ok=False,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    f26 = destination / "runtime" / "f26_inflate.py"
    prefix_patch = _replace_once(
        f26,
        '''    if pair_count != int(renderer.N) and token_decoder != "native-hpac":\n        raise InflationError("advisory prefix inflation requires the resumable native token path")\n''',
        '''    if pair_count != int(renderer.N) and token_decoder != "python":\n        raise InflationError("pipeline prefix inflation requires the retained Python-decoder checkpoint")\n''',
    )
    render_patch = _replace_once(
        f26,
        '''    if parallel_report is None:\n        if pair_count != int(renderer.N):\n            raise InflationError("prefix rendering requires F26_ADVISORY_RENDER_WORKERS")\n        renderer.render_video(semantic, basis, coefficients, tokens, partial, device)\n''',
        '''    if parallel_report is None:\n        renderer.render_video(semantic, basis, coefficients, tokens, partial, device, pair_count=pair_count)\n''',
    )
    renderer = destination / "cpr1" / "inflate.py"
    signature_patch = _replace_once(
        renderer,
        "def render_video(semantic, basis, coefficients, tokens, destination: Path, device):\n",
        "def render_video(semantic, basis, coefficients, tokens, destination: Path, device, pair_count=N):\n",
    )
    count_patch = _replace_once(
        renderer,
        "        shape=(N * 2, CAMERA_H, CAMERA_W, 3),\n",
        "        shape=(pair_count * 2, CAMERA_H, CAMERA_W, 3),\n",
    )
    loops_patch = _replace_once(
        renderer,
        "    for start in range(0, N, semantic_batch):\n        end = min(start + semantic_batch, N)\n",
        "    for start in range(0, pair_count, semantic_batch):\n        end = min(start + semantic_batch, pair_count)\n",
    )
    pose_loops_patch = _replace_once(
        renderer,
        "    for start in range(0, N, pose_batch):\n        end = min(start + pose_batch, N)\n",
        "    for start in range(0, pair_count, pose_batch):\n        end = min(start + pose_batch, pair_count)\n",
    )
    after = _source_records(destination)
    payload = {
        "schema": "ddm_fpc2_pipeline_receiver_copy.v1",
        "shipped_tree": str(SHIPPED.resolve()),
        "copied_sources_before_patch": before,
        "copied_sources_after_patch": after,
        "patches": [prefix_patch, render_patch, signature_patch, count_patch, loops_patch, pose_loops_patch],
        "prefix_contract": (
            "n<=8 only; Python token decoder identity is supplied by the retained, "
            "archive-bound cc10a7 token checkpoint; shipped sources are unchanged"
        ),
    }
    atomic_json(manifest_path, payload)
    return payload


def _import_copied_f26(runtime_root: Path):
    for name in list(sys.modules):
        if name == "runtime" or name.startswith("runtime.") or name == "_f26_renderer":
            sys.modules.pop(name, None)
    sys.path.insert(0, str(runtime_root))
    try:
        return importlib.import_module("runtime.f26_inflate")
    finally:
        sys.path.pop(0)


def _seed_prefix_checkpoint(f26, request: ReceiverRequest) -> dict[str, Any]:
    token_fact = file_fact(TOKEN_FIELD)
    if token_fact["sha256"] != TOKEN_FIELD_SHA256 or token_fact["bytes"] != TOKEN_FIELD_BYTES:
        raise PipelineBlocked("retained Python-decoder token field drifted")
    import torch

    renderer_dir = request.runtime_root / "cpr1"
    renderer = f26._load_renderer(renderer_dir)
    parts = f26.read_residual_archive(request.archive)
    token_stream_sha256 = hashlib.sha256(parts.token_stream).hexdigest()
    if token_stream_sha256 != TOKEN_STREAM_SHA256:
        raise PipelineBlocked(
            "retained Python token field is not proven for this archive token stream"
        )
    _, device_report = f26._configure_device(torch, request.device, 4 if request.device == "cpu" else None)
    fingerprint = f26._token_decoder_fingerprint(
        renderer=renderer,
        renderer_dir=renderer_dir,
        token_decoder="python",
        num_threads=4 if request.device == "cpu" else 0,
    )
    binding = f26._checkpoint_binding(
        archive_path=request.archive,
        parts=parts,
        renderer_dir=renderer_dir,
        device_report=device_report,
        pair_count=request.pair_count,
        token_decoder="python",
        token_decoder_fingerprint=fingerprint,
    )
    raw = np.memmap(TOKEN_FIELD, mode="r", dtype=np.uint8, shape=(600, 384, 512))
    tokens = torch.from_numpy(np.asarray(raw[: request.pair_count]).copy())  # SUBSET_SELECTION_OK:bounded plumbing smoke over the first pair_count pairs (receipt labels it a contiguous-prefix smoke, score_claim=false, never a verdict); population runs are chunked over all 600 pairs
    report = {
        "schema": "ddm_fpc2_retained_python_token_checkpoint.v1",
        "decoder": "python",
        "source": token_fact,
        "pair_count": request.pair_count,
        "archive_token_stream_sha256": token_stream_sha256,
    }
    return f26._write_token_checkpoint(
        request.checkpoint_dir,
        tokens,
        binding=binding,
        token_report=report,
    )


def inflate(request: ReceiverRequest) -> dict[str, Any]:
    """Inflate exactly the declared archive; never infer identity or device."""

    archive_fact = file_fact(request.archive)
    if archive_fact["sha256"] != request.archive_sha256 or archive_fact["bytes"] != request.archive_bytes:
        raise PipelineBlocked("receiver archive declaration does not match archive bytes")
    require_device(request.device)
    copy_manifest = materialize_runtime_copy(request.runtime_root)
    if request.destination.exists():
        receipt_path = request.destination.with_suffix(".receiver.json")
        if receipt_path.is_file():
            prior = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (
                prior.get("archive_declaration") == archive_fact
                and prior.get("raw_sha256") == file_fact(request.destination)["sha256"]
                and prior.get("raw_bytes") == request.pair_count * RAW_PAIR_BYTES
            ):
                return prior
        raise PipelineBlocked(f"refusing to overwrite receiver output: {request.destination}")
    f26 = _import_copied_f26(request.runtime_root)
    os.environ["F26_ADVISORY_PAIR_LIMIT"] = str(request.pair_count)
    os.environ["F26_TOKEN_DECODER"] = "python"
    request.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    receipt = request.checkpoint_dir / "tokens_cpu_stage_complete.json"
    if not receipt.exists():
        _seed_prefix_checkpoint(f26, request)
    report = f26.inflate_archive(
        request.archive,
        request.destination,
        renderer_dir=request.runtime_root / "cpr1",
        device_name=request.device,
        num_threads=4 if request.device == "cpu" else None,
        checkpoint_dir=request.checkpoint_dir,
    )
    expected_raw_bytes = request.pair_count * RAW_PAIR_BYTES
    if report["raw_bytes"] != expected_raw_bytes:
        raise PipelineBlocked("receiver output byte count differs from pair scope")
    report = {
        **report,
        "schema": "ddm_fpc2_pipeline_receiver.v1",
        "score_claim": False,
        "archive_declaration": archive_fact,
        "receiver_copy_manifest": file_fact(request.runtime_root / "PIPELINE_RECEIVER_COPY.json"),
        "shipped_source_tree": copy_manifest["shipped_tree"],
    }
    atomic_json(request.destination.with_suffix(".receiver.json"), report)
    return report


def subprocess_inflate(request: ReceiverRequest) -> dict[str, Any]:
    """Run a receiver in a fresh interpreter so copied module names cannot leak."""

    request_path = request.checkpoint_dir.parent / "receiver_request.json"
    atomic_json(request_path, {field.name: str(getattr(request, field.name)) if isinstance(getattr(request, field.name), Path) else getattr(request, field.name) for field in dataclasses.fields(request)})
    completed = subprocess.run(
        [sys.executable, "-m", "tac.semantic_pipeline.receiver", "--request", str(request_path)],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise PipelineBlocked(f"receiver subprocess failed: {completed.stderr[-2000:]}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise PipelineBlocked("receiver subprocess returned no receipt")
    return json.loads(lines[-1])


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.request.read_text(encoding="utf-8"))
    for key in ("archive", "destination", "runtime_root", "checkpoint_dir"):
        payload[key] = Path(payload[key])
    print(json.dumps(inflate(ReceiverRequest(**payload)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
