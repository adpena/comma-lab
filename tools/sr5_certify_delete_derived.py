#!/usr/bin/env python3
"""Certify-or-block deletion of DERIVED payload files on the SSD tiers (ddm_sr5).

The certify-or-block law in CLAUDE.md ("Local Disk, SSD Spill, Auto-Cleanup, And
Provenance") allows a destructive delete only for explicitly certified rebuildable
scratch, and only when a machine-readable record preserves deterministic
reproducibility.  This tool implements the narrowest honest version of that:

  a file is deletable only when it is a pure deterministic function of a SIBLING
  file that stays on disk, and that function reproduces the target's recorded
  sha256 EXACTLY, re-run here, per file, immediately before the delete.

Nothing is sampled.  Nothing is asserted.  Each deletion is preceded by four
independent bindings, all of which must hold or the file is REFUSED:

  1. the directory's RESULT.json manifest names the file with a sha256 and size;
  2. the file on disk hashes to that recorded sha256 (manifest binds THIS file);
  3. the retained source sibling exists and hashes to ITS recorded sha256;
  4. the rebuild, executed here from that source, serialises to the SAME sha256.

ALWAYS KEEP THE PAYLOAD is honored: the payload root (the source sibling) and
every scalar/manifest stay on disk, the deleted bytes remain bit-exactly
recoverable, and a DERIVED_DELETED.json resolver sidecar is written next to the
manifest naming the rebuild command for every removed file.

Usage (plan only, no mutation):
    .venv/bin/python tools/sr5_certify_delete_derived.py \
        --bank-root /Volumes/VertigoDataTier/pact/.../jacobian_bank \
        --payload-class frame0_receiver \
        --ledger .omx/research/ddm_sr5_certified_deletions_20260911.jsonl \
        --target-free-gib 61

Add --apply to actually delete.  Deletion stops as soon as the mount reaches
--target-free-gib (do not over-reclaim).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

SCHEMA = "ddm_sr5_certified_derived_deletion.v1"
SIDECAR_NAME = "DERIVED_DELETED.json"

# Paths that must never be mutated by this tool, whatever the caller passes.
# Live arms, sealed pointer trees, sr4's retained offload, and the two stores
# that are custody rather than scratch.
PROTECTED_SUBSTRINGS = (
    "/ddm_pc3_",
    "/ddm_ntb2_",
    "/ddm_mxo2",
    "/ddm_mxo3",
    "/ddm_rbf1",
    "/ddm_rlc5_cure_on_move43",
    "/ddm_sj1_pass6",
    "/ddm_rp1_round2",
    "/ddm_sr4_20260910/retained/ap_offload",
    "/public_datasets",
    "/cold_store",
    "/upstream/",
)


class CertifyError(RuntimeError):
    """Raised when the tool refuses to proceed."""


@dataclass(frozen=True)
class PayloadClass:
    """One derived filename and the retained sibling it is rebuilt from."""

    name: str
    target_filename: str
    source_filename: str
    rebuild_command: str
    rebuild_note: str
    rebuild: Callable[[Path], np.ndarray]


def _rebuild_frame0_receiver(source: Path) -> np.ndarray:
    """frame0_receiver == pose_input[:, 0] by construction.

    The producer (experiments/ddm_pk4_optimal_form_frame0_pose.py:535-538) builds
    ``inputs = np.stack((slaves, masters), axis=1)`` and saves ``slaves`` as
    frame0_receiver and ``inputs`` as pose_input, so axis-1 index 0 of pose_input
    IS the frame0_receiver array.
    """
    pose_input = np.load(source, allow_pickle=False, mmap_mode="r")
    if pose_input.ndim != 5 or pose_input.shape[1] != 2:
        raise CertifyError(f"unexpected pose_input shape {pose_input.shape} at {source}")
    return np.ascontiguousarray(pose_input[:, 0])


def _rebuild_pose_preprocessed_yuv6(source: Path) -> np.ndarray:
    """pose_preprocessed_yuv6 == PoseNet.preprocess_input(pose_input).

    ``preprocess_input`` reads no module state (upstream/modules.py:70-74), so it
    is called unbound: bilinear interpolate to the scorer input size then
    rgb_to_yuv6.  No network forward pass is executed and no weights are loaded.
    """
    import torch

    upstream = Path(__file__).resolve().parents[1] / "upstream"
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    from modules import PoseNet  # type: ignore[import-not-found]

    pose_input = np.load(source, allow_pickle=False)
    if pose_input.ndim != 5 or pose_input.shape[-1] != 3:
        raise CertifyError(f"unexpected pose_input shape {pose_input.shape} at {source}")
    tensor = torch.from_numpy(pose_input).permute(0, 1, 4, 2, 3).float()
    out = PoseNet.preprocess_input(None, tensor)
    return np.asarray(out.numpy(), dtype=np.float32)


PAYLOAD_CLASSES: dict[str, PayloadClass] = {
    "frame0_receiver": PayloadClass(
        name="frame0_receiver",
        target_filename="frame0_receiver.uint8.npy",
        source_filename="pose_input.uint8.npy",
        rebuild_command=(
            "numpy.save(out, numpy.ascontiguousarray("
            "numpy.load('pose_input.uint8.npy', allow_pickle=False)[:, 0]), allow_pickle=False)"
        ),
        rebuild_note=(
            "pure axis-1 slice of the retained sibling pose_input.uint8.npy; the producer "
            "experiments/ddm_pk4_optimal_form_frame0_pose.py:535-538 stacks the same array "
            "into pose_input, so no external input, archive, scorer or decode is required"
        ),
        rebuild=_rebuild_frame0_receiver,
    ),
    "pose_preprocessed_yuv6": PayloadClass(
        name="pose_preprocessed_yuv6",
        target_filename="pose_preprocessed_yuv6.float32.npy",
        source_filename="pose_input.uint8.npy",
        rebuild_command=(
            "numpy.save(out, numpy.asarray(upstream.modules.PoseNet.preprocess_input(None, "
            "torch.from_numpy(numpy.load('pose_input.uint8.npy', allow_pickle=False))"
            ".permute(0, 1, 4, 2, 3).float()).numpy(), dtype=numpy.float32), allow_pickle=False)"
        ),
        rebuild_note=(
            "deterministic upstream pose preprocess (bilinear interpolate + rgb_to_yuv6) of the "
            "retained sibling pose_input.uint8.npy; preprocess_input uses no module state, so no "
            "scorer forward pass and no weights are involved"
        ),
        rebuild=_rebuild_pose_preprocessed_yuv6,
    ),
}


def utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, bufsize: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(bufsize)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_npy_bytes(array: np.ndarray) -> tuple[str, int]:
    """Serialise an array exactly as numpy.save would and hash the file bytes."""
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    payload = buffer.getvalue()
    return hashlib.sha256(payload).hexdigest(), len(payload)


def df_free_bytes(mount: str) -> int:
    stat = os.statvfs(mount)
    return int(stat.f_bavail) * int(stat.f_frsize)


def df_snapshot(mount: str) -> dict[str, object]:
    out = subprocess.run(["df", "-k", mount], capture_output=True, text=True, check=True).stdout.strip()
    return {"mount": mount, "df_k": out, "free_bytes": df_free_bytes(mount)}


def assert_not_protected(path: Path) -> None:
    text = str(path.resolve())
    for needle in PROTECTED_SUBSTRINGS:
        if needle in text:
            raise CertifyError(f"protected path refused: {text} matches {needle!r}")


def append_ledger(ledger: Path, row: dict) -> None:
    """fcntl-locked, fsync'd append so a crash cannot lose a cert row."""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True) + "\n"
    with ledger.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_sidecar(directory: Path, rows: list[dict]) -> None:
    """Leave a resolver next to the manifest naming how to rebuild what was removed."""
    sidecar = directory / SIDECAR_NAME
    existing: list[dict] = []
    if sidecar.is_file():
        try:
            loaded = json.loads(sidecar.read_text())
            existing = list(loaded.get("deleted", []))
        except (json.JSONDecodeError, OSError):
            existing = []
    payload = {
        "schema": SCHEMA,
        "written_at_utc": utcnow(),
        "note": (
            "These files were deleted as certified-rebuildable derived payloads. "
            "Each is a deterministic function of the retained source sibling named "
            "below and reproduces the recorded sha256 exactly."
        ),
        "deleted": existing + rows,
    }
    tmp = sidecar.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
    os.replace(tmp, sidecar)


def require_record(manifest: dict, key: str, directory: Path, filename: str) -> dict:
    record = manifest.get(key)
    if not isinstance(record, dict):
        raise CertifyError(f"manifest has no dict record for {key!r}")
    for field in ("path", "bytes", "sha256"):
        if field not in record:
            raise CertifyError(f"manifest record {key!r} missing {field!r}")
    if Path(str(record["path"])).name != filename:
        raise CertifyError(f"manifest record {key!r} names {record['path']!r}, expected {filename}")
    if Path(str(record["path"])).parent != directory:
        raise CertifyError(f"manifest record {key!r} points outside {directory}")
    return record


def environment_pins() -> dict[str, str]:
    """Interpreter and library versions the rebuild command was exercised under."""
    pins = {"python": sys.version.split()[0], "numpy": np.__version__}
    try:
        import torch

        pins["torch"] = torch.__version__
    except ImportError:  # pragma: no cover - torch is only needed by one class
        pins["torch"] = "not-imported"
    return pins


def certify_one(
    directory: Path,
    payload: PayloadClass,
    tool_sha: str,
    ledger: Path,
    apply: bool,
    closed_store_evidence: str,
) -> dict:
    """Return a row describing the verdict for one directory. Deletes only on apply."""
    target = directory / payload.target_filename
    source = directory / payload.source_filename
    manifest_path = directory / "RESULT.json"
    row: dict[str, object] = {
        "schema": SCHEMA,
        "written_at_utc": utcnow(),
        "payload_class": payload.name,
        "original_path": str(target),
        "source_path": str(source),
        "closed_store_evidence": closed_store_evidence,
        "tool": "tools/sr5_certify_delete_derived.py",
        "tool_sha256": tool_sha,
        "rebuild_command": payload.rebuild_command,
        "rebuild_reason": payload.rebuild_note,
        "environment_pins": environment_pins(),
        "score_claim": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "verdict": "REFUSED",
    }
    try:
        assert_not_protected(target)
        if not target.is_file() or target.is_symlink():
            raise CertifyError("target missing or not a regular file")
        if not source.is_file() or source.is_symlink():
            raise CertifyError("retained source sibling missing or not a regular file")
        if not manifest_path.is_file():
            raise CertifyError("no RESULT.json manifest in the directory")

        manifest = json.loads(manifest_path.read_text())
        target_record = require_record(manifest, payload.name, directory, payload.target_filename)
        source_record = require_record(
            manifest, payload.source_filename.split(".")[0], directory, payload.source_filename
        )
        row["axis"] = manifest.get("axis")
        row["producer_schema"] = manifest.get("schema")
        row["manifest_path"] = str(manifest_path)
        row["manifest_sha256"] = str(target_record["sha256"])
        row["bytes"] = int(target.stat().st_size)
        row["source_manifest_sha256"] = str(source_record["sha256"])

        link_count = os.lstat(target).st_nlink
        row["st_nlink"] = int(link_count)
        if link_count != 1:
            raise CertifyError(f"st_nlink={link_count}: deleting would not free the blocks")
        if row["bytes"] != int(target_record["bytes"]):
            raise CertifyError("on-disk size differs from the manifest size")

        # Binding 2: the manifest binds THIS file.
        on_disk = sha256_file(target)
        row["on_disk_sha256"] = on_disk
        if on_disk != target_record["sha256"]:
            raise CertifyError("on-disk sha256 differs from the manifest sha256")

        # Binding 3: the retained rebuild input is itself the certified one.
        source_sha = sha256_file(source)
        row["source_on_disk_sha256"] = source_sha
        if source_sha != source_record["sha256"]:
            raise CertifyError("retained source sha256 differs from its manifest sha256")

        # Binding 4: the rebuild, executed now, reproduces the bytes exactly.
        rebuilt = payload.rebuild(source)
        rebuilt_sha, rebuilt_bytes = sha256_npy_bytes(rebuilt)
        row["rebuilt_sha256"] = rebuilt_sha
        row["rebuilt_bytes"] = rebuilt_bytes
        if rebuilt_sha != on_disk or rebuilt_bytes != row["bytes"]:
            raise CertifyError("rebuild did not reproduce the file byte-for-byte")

        row["verdict"] = "CERTIFIED_REBUILDABLE"
    except (CertifyError, OSError, ValueError, json.JSONDecodeError) as exc:
        row["refusal_reason"] = f"{type(exc).__name__}: {exc}"
        return row

    if not apply:
        row["action"] = "PLAN_ONLY_NO_MUTATION"
        return row

    # Cert row lands BEFORE the delete, so the record can never trail the mutation.
    cert_row = dict(row)
    cert_row["action"] = "DELETE_PENDING"
    append_ledger(ledger, cert_row)
    target.unlink()
    row["action"] = "DELETED"
    row["deleted_at_utc"] = utcnow()
    append_ledger(ledger, row)
    write_sidecar(
        directory,
        [
            {
                "deleted_path": str(target),
                "bytes": row["bytes"],
                "sha256": on_disk,
                "source_path": str(source),
                "source_sha256": source_sha,
                "rebuild_command": payload.rebuild_command,
                "deleted_at_utc": row["deleted_at_utc"],
            }
        ],
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", required=True)
    parser.add_argument("--payload-class", required=True, choices=sorted(PAYLOAD_CLASSES))
    parser.add_argument("--ledger", required=True)
    parser.add_argument(
        "--target-free-gib",
        type=float,
        required=True,
        help="Stop deleting as soon as the mount has at least this much free.",
    )
    parser.add_argument("--mount", default=None, help="Defaults to the bank-root's mount.")
    parser.add_argument(
        "--closed-store-evidence",
        required=True,
        help="Why this store is CLOSED (verdict/memo/claim row). Recorded on every cert row.",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = no cap on files considered.")
    parser.add_argument("--apply", action="store_true", help="Without this, census + verify only.")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    bank_root = Path(args.bank_root).resolve()
    if not bank_root.is_dir():
        print(f"REFUSE: bank-root is not a directory: {bank_root}", file=sys.stderr)
        return 2
    try:
        assert_not_protected(bank_root)
    except CertifyError as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 3

    payload = PAYLOAD_CLASSES[args.payload_class]
    ledger = Path(args.ledger).resolve()
    mount = args.mount or f"/Volumes/{bank_root.relative_to('/Volumes').parts[0]}"
    tool_sha = sha256_file(Path(__file__).resolve())

    targets: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(bank_root):
        if payload.target_filename in filenames:
            name = payload.target_filename
            if name.startswith("._"):
                continue
            targets.append(Path(dirpath) / name)
    targets.sort()
    if args.limit:
        targets = targets[: args.limit]

    before = df_snapshot(mount)
    target_free = int(args.target_free_gib * (1024**3))
    print(
        f"[sr5] class={payload.name} candidates={len(targets)} mount={mount} "
        f"free_before={before['free_bytes'] / 1024**3:.3f} GiB target={args.target_free_gib} GiB "
        f"apply={args.apply}"
    )

    certified = deleted = refused = 0
    freed = 0
    refusals: list[dict] = []
    for index, target in enumerate(targets, start=1):
        if args.apply and df_free_bytes(mount) >= target_free:
            print(f"[sr5] target reached after {deleted} deletions; stopping (no over-reclaim).")
            break
        row = certify_one(target.parent, payload, tool_sha, ledger, args.apply, args.closed_store_evidence)
        if row["verdict"] != "CERTIFIED_REBUILDABLE":
            refused += 1
            refusals.append(row)
            append_ledger(ledger, row)
            print(f"[sr5] REFUSED {target}: {row.get('refusal_reason')}", file=sys.stderr)
            continue
        certified += 1
        if row.get("action") == "DELETED":
            deleted += 1
            freed += int(row["bytes"])
        if index % args.progress_every == 0:
            print(
                f"[sr5] {index}/{len(targets)} certified={certified} deleted={deleted} "
                f"refused={refused} freed={freed / 1024**3:.3f} GiB "
                f"free_now={df_free_bytes(mount) / 1024**3:.3f} GiB",
                flush=True,
            )

    after = df_snapshot(mount)
    summary = {
        "schema": SCHEMA + ".summary",
        "written_at_utc": utcnow(),
        "payload_class": payload.name,
        "bank_root": str(bank_root),
        "closed_store_evidence": args.closed_store_evidence,
        "environment_pins": environment_pins(),
        "candidates": len(targets),
        "certified": certified,
        "deleted": deleted,
        "refused": refused,
        "freed_bytes_accounted": freed,
        "df_before": before,
        "df_after": after,
        "df_free_delta_bytes": int(after["free_bytes"]) - int(before["free_bytes"]),
        "apply": bool(args.apply),
        "tool_sha256": tool_sha,
        "refusal_reasons": sorted({str(r.get("refusal_reason")) for r in refusals}),
    }
    append_ledger(ledger, summary)
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0 if refused == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
