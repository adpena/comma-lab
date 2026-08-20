#!/usr/bin/env python3
"""Materialize the exact lossless ep725 xcodec recode and bounded replay proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.witness_dsl.ep725_levelset_predictor_adapter import (  # noqa: E402
    EP725_ARCHIVE_SHA256,
    EP725_CAMERA_HEIGHT,
    EP725_CAMERA_WIDTH,
    EP725_MEMBER_SHA256,
    EP725_RUNTIME_SHA256,
    EP725_SOURCE_DIRECTORY,
)
from tac.witness_dsl.ep725_lossless_xcodec_recode import (  # noqa: E402
    search_ep725_lossless_xcodec,
)

OUTPUT_ROOT: Final = (
    REPO / ".omx/research/original_taskspace_inverse_witness_codec_20260725" / "ep725_lossless_xcodec_recode_20260726"
)
ARCHIVE_NAME: Final = "ep725_lossless_xcodec_recode.not_a_candidate.zip"
RECEIPT_NAME: Final = "receipt.json"
POINTER_PATH: Final = REPO / ".omx/state/canonical_frontier_pointer.json"
DEFAULT_SCRATCH_ROOT: Final = Path("/Volumes/VertigoDataTier/pact/scratch")
MIN_SCRATCH_FREE_BYTES: Final = 64 * 1024 * 1024


class Ep725XCodecMaterializationError(RuntimeError):
    """Raised when materialization, storage, or runtime proof fails closed."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def _read_pointer() -> dict[str, Any]:
    raw = POINTER_PATH.read_bytes()
    parsed = json.loads(raw)
    effective = parsed.get("effective_frontier")
    if not isinstance(effective, dict):
        raise Ep725XCodecMaterializationError("canonical pointer has no effective_frontier object")
    return {
        "path": str(POINTER_PATH.relative_to(REPO)),
        "sha256": _sha256(raw),
        "effective_frontier": effective,
        "selection_role": "reporting snapshot only; recode selected solely by exact archive bytes",
    }


def _require_scratch_root(path: Path, *, allow_local_scratch: bool) -> Path:
    resolved = path.expanduser().resolve()
    approved = (
        Path("/Volumes/VertigoDataTier/pact").resolve(),
        Path("/Volumes/APDataStore/pact").resolve(),
    )
    if not allow_local_scratch and not any(resolved == root or root in resolved.parents for root in approved):
        raise Ep725XCodecMaterializationError(
            "scratch root must use the configured SSD waterfall; pass --allow-local-scratch explicitly"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(resolved).free
    if free < MIN_SCRATCH_FREE_BYTES:
        raise Ep725XCodecMaterializationError(
            f"scratch preflight has {free} free bytes, needs {MIN_SCRATCH_FREE_BYTES}"
        )
    return resolved


def _atomic_write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise Ep725XCodecMaterializationError(f"refusing to overwrite non-identical artifact {path}")
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _runtime_once(
    runtime: Path,
    member: bytes,
    scratch: Path,
    label: str,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    member_path = scratch / f"{label}.bin"
    raw_path = scratch / f"{label}.raw"
    member_path.write_bytes(member)
    environment = os.environ.copy()
    environment.update(
        {
            "INFLATE_MAX_PAIRS": "1",
            "INFLATE_WORKERS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        [sys.executable, str(runtime), str(member_path), str(raw_path)],
        cwd=str(runtime.parent),
        env=environment,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise Ep725XCodecMaterializationError(
            f"{label} bounded frozen-runtime replay failed rc={completed.returncode}; "
            f"stderr_sha256={_sha256(completed.stderr)}"
        )
    raw = raw_path.read_bytes()
    expected = 2 * EP725_CAMERA_HEIGHT * EP725_CAMERA_WIDTH * 3
    if len(raw) != expected:
        raise Ep725XCodecMaterializationError(f"{label} bounded runtime wrote {len(raw)} bytes, expected {expected}")
    return {
        "raw": raw,
        "raw_bytes": len(raw),
        "raw_sha256": _sha256(raw),
        "stdout_sha256": _sha256(completed.stdout),
        "stderr_sha256": _sha256(completed.stderr),
        "returncode": completed.returncode,
    }


def _bounded_runtime_equivalence(
    runtime: Path,
    source_member: bytes,
    selected_member: bytes,
    scratch_root: Path,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    if _sha256(runtime.read_bytes()) != EP725_RUNTIME_SHA256:
        raise Ep725XCodecMaterializationError("frozen ep725 runtime SHA-256 custody mismatch")
    with tempfile.TemporaryDirectory(prefix="ep725_xcodec_", dir=scratch_root) as temporary:
        scratch = Path(temporary)
        source = _runtime_once(runtime, source_member, scratch, "source", timeout_seconds=timeout_seconds)
        selected = _runtime_once(runtime, selected_member, scratch, "selected", timeout_seconds=timeout_seconds)
        equal = source.pop("raw") == selected.pop("raw")
    if not equal:
        raise Ep725XCodecMaterializationError("source and selected bounded frozen-runtime uint8 outputs differ")
    return {
        "axis": "[macOS-CPU frozen receiver structural proof]",
        "pair_ids": [0],
        "frames_compared": 2,
        "source": source,
        "selected": selected,
        "uint8_raw_equal": True,
        "scratch_auto_deleted": True,
        "full_n600_replay_owed": True,
    }


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def materialize(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    source_directory = args.source_directory.expanduser().resolve()
    source_archive_path = source_directory / "archive.zip"
    runtime_path = source_directory / "inflate.py"
    source_archive = source_archive_path.read_bytes()
    if _sha256(source_archive) != EP725_ARCHIVE_SHA256:
        raise Ep725XCodecMaterializationError("frozen ep725 source archive custody mismatch")
    scratch_root = _require_scratch_root(args.scratch_root, allow_local_scratch=args.allow_local_scratch)
    pointer_start = _read_pointer()
    result = search_ep725_lossless_xcodec(
        source_archive,
        expected_archive_sha256=EP725_ARCHIVE_SHA256,
        expected_member_sha256=EP725_MEMBER_SHA256,
    )
    if not result.selected.transformed or result.archive_delta_bytes >= 0:
        raise Ep725XCodecMaterializationError("exact search found no strictly smaller transformed ep725 archive")
    runtime = _bounded_runtime_equivalence(
        runtime_path,
        result.source.member_bytes,
        result.selected.member_bytes,
        scratch_root,
        timeout_seconds=args.timeout_seconds,
    )
    pointer_end = _read_pointer()
    if pointer_start["sha256"] != pointer_end["sha256"]:
        raise Ep725XCodecMaterializationError("canonical pointer changed during materialization; rerun")

    archive_path = args.output_directory / ARCHIVE_NAME
    receipt_path = args.output_directory / RECEIPT_NAME
    receipt = result.structural_receipt()
    receipt.update(
        {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "git_head": _git_head(),
            "source_paths": {
                "archive": str(source_archive_path),
                "runtime": str(runtime_path),
            },
            "runtime": {
                "path": str(runtime_path),
                "bytes": runtime_path.stat().st_size,
                "sha256": EP725_RUNTIME_SHA256,
                "bounded_equivalence": runtime,
            },
            "canonical_frontier_pointer": pointer_end,
            "artifact": {
                "path": str(archive_path.relative_to(REPO)),
                "classification": "not_a_candidate",
                "bytes": len(result.selected.archive_bytes),
                "sha256": _sha256(result.selected.archive_bytes),
            },
            "reproduction": {
                "argv": [str(value) for value in sys.argv],
                "python": sys.version,
                "implementation": {
                    "module_path": "src/tac/witness_dsl/ep725_lossless_xcodec_recode.py",
                    "module_sha256": _sha256((SRC / "tac/witness_dsl/ep725_lossless_xcodec_recode.py").read_bytes()),
                    "tool_path": "tools/materialize_ep725_lossless_xcodec_recode.py",
                    "tool_sha256": _sha256(Path(__file__).read_bytes()),
                },
            },
        }
    )
    _atomic_write_once(archive_path, result.selected.archive_bytes)
    receipt_bytes = _canonical_json(receipt)
    _atomic_write_once(receipt_path, receipt_bytes)
    if _sha256(archive_path.read_bytes()) != receipt["artifact"]["sha256"]:
        raise Ep725XCodecMaterializationError("durable archive re-read hash mismatch")
    if json.loads(receipt_path.read_bytes()) != receipt:
        raise Ep725XCodecMaterializationError("durable receipt canonical parse-back mismatch")
    return archive_path, receipt_path, receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-reviewed", action="store_true")
    parser.add_argument("--source-directory", type=Path, default=EP725_SOURCE_DIRECTORY)
    parser.add_argument("--output-directory", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    parser.add_argument("--allow-local-scratch", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.execute_reviewed:
        print(
            "REFUSE: materialization requires --execute-reviewed after source/spec review",
            file=sys.stderr,
        )
        return 2
    if not (0.0 < args.timeout_seconds <= 1_800.0):
        raise Ep725XCodecMaterializationError("timeout-seconds must be in (0,1800]")
    archive, receipt_path, receipt = materialize(args)
    print(
        json.dumps(
            {
                "archive": str(archive),
                "receipt": str(receipt_path),
                "archive_bytes": receipt["artifact"]["bytes"],
                "archive_sha256": receipt["artifact"]["sha256"],
                "archive_delta_bytes": receipt["exact_delta"]["archive_bytes"],
                "rate_score_delta": receipt["exact_delta"]["rate_score_units"],
                "bounded_runtime_equal": receipt["runtime"]["bounded_equivalence"]["uint8_raw_equal"],
                "candidate_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
