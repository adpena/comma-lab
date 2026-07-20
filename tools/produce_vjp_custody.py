#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Produce immutable per-pair frozen CPU-Torch Seg/Pose VJP sidecars."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for import_path in (REPO, SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from tac.optimization.vjp_custody import (  # noqa: E402
    ACTIVE_ARRANGEMENT,
    ACTIVE_ARRANGEMENTS,
    CAMERA_HW,
    EXPECTED_HASHES,
    MANIFEST_SCHEMA,
    NATIVE_REFRESH_ARRANGEMENT,
    RECEIVER_ARITHMETIC,
    REPRESENTATION,
    SCORER_HW,
    VJPCustodyError,
    atomic_json,
    canonical_json,
    compute_pair_derivatives,
    load_vjp_pair_row,
    recover_pair_sidecar_row,
    sha256_file,
    source_hashes,
    write_pair_sidecar,
)
from tools.measure_uint8_lattice_feasibility import (  # noqa: E402
    _stat_tree_snapshot,
    stored_npy_memmap,
)

DEFAULT_CACHE = Path("/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
DEFAULT_UPSTREAM = Path("/Users/adpena/Projects/pact/upstream")
DEFAULT_OUTPUT = Path("/Volumes/VertigoDataTier/pact/evidence/vjp_custody_20260719")
SECONDARY_OUTPUT = Path("/Volumes/APDataStore/pact/evidence/vjp_custody_20260719")
SACRED = Path("/Users/adpena/Projects/pact/experiments/results/levelset_n600_witness_20260717T113932Z")
MAX_PAIRS = 12
SEED = 20260719
BYTES_PER_PAIR_PREFLIGHT = 256 << 20
FIXED_FREE_SPACE_RESERVE = 1 << 30
REFUSAL_SCHEMA = "vjp_custody_pair_refusal.v1"
REFUSAL_VERDICT_SCOPE = (
    "this pair and frozen active arrangement only; not family or other-pair scope"
)


def _pair_ids(values: list[int]) -> list[int]:
    ids = [int(value) for value in values]
    if not ids or len(ids) > MAX_PAIRS or len(set(ids)) != len(ids):
        raise VJPCustodyError(f"producer requires 1..{MAX_PAIRS} unique pair ids")
    if any(pair_id < 0 or pair_id >= 600 for pair_id in ids):
        raise VJPCustodyError("producer pair ids must lie in [0,600)")
    return ids


def _enforce_output_tier(output: Path, pairs_remaining: int) -> Path:
    resolved = output.resolve()
    allowed = [root.resolve() for root in (DEFAULT_OUTPUT, SECONDARY_OUTPUT)]
    if not any(resolved == root or resolved.is_relative_to(root) for root in allowed):
        raise VJPCustodyError(
            "VJP bulk output must be under the dedicated VertigoDataTier evidence root "
            "or the declared APDataStore fallback"
        )
    selected_root = next(root for root in allowed if resolved == root or resolved.is_relative_to(root))
    ssd_tier = selected_root.parents[1]
    if not ssd_tier.exists():
        raise VJPCustodyError(f"selected SSD evidence tier is unavailable: {ssd_tier}")
    if selected_root == allowed[1] and allowed[0].parents[1].exists():
        raise VJPCustodyError(
            "APDataStore fallback is forbidden while the primary VertigoDataTier is available"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    required = FIXED_FREE_SPACE_RESERVE + pairs_remaining * BYTES_PER_PAIR_PREFLIGHT
    free = shutil.disk_usage(resolved).free
    if free < required:
        raise VJPCustodyError(
            f"SSD free-space preflight refused producer: free={free}, required={required}"
        )
    return resolved


def _load_cache(path: Path) -> dict[str, np.memmap]:
    fields = {
        key: stored_npy_memmap(path, key)
        for key in ("n_pairs", "gt_f0", "gt_f1", "lstars", "margins")
    }
    if int(np.asarray(fields["n_pairs"]).reshape(())) != 600:
        raise VJPCustodyError("only the real n600 cache is admissible")
    expected = {
        "gt_f0": (600, *CAMERA_HW, 3),
        "gt_f1": (600, *CAMERA_HW, 3),
        "lstars": (600, *SCORER_HW),
        "margins": (600, *SCORER_HW),
    }
    for key, shape in expected.items():
        if fields[key].shape != shape:
            raise VJPCustodyError(f"cache member {key} geometry mismatch")
    return fields


def _prepend_exact_import_root(root: Path) -> None:
    resolved = root.resolve()
    retained = []
    for entry in sys.path:
        try:
            if Path(entry or ".").resolve() == resolved:
                continue
        except (OSError, RuntimeError):
            pass
        retained.append(entry)
    sys.path[:] = [str(resolved), *retained]


def _load_scorers(upstream: Path, threads: int) -> tuple[Any, Any, Any]:
    _prepend_exact_import_root(upstream)
    import torch
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    torch.set_num_threads(threads)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    model = DistortionNet().eval().to("cpu")
    model.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    loaded_modules = Path(sys.modules["modules"].__file__).resolve()
    loaded_frame_utils = Path(sys.modules["frame_utils"].__file__).resolve()
    if loaded_modules != (upstream / "modules.py").resolve():
        raise VJPCustodyError(f"modules imported from wrong source: {loaded_modules}")
    if loaded_frame_utils != (upstream / "frame_utils.py").resolve():
        raise VJPCustodyError(f"frame_utils imported from wrong source: {loaded_frame_utils}")
    return model.segnet, model.posenet, torch


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"


def _manifest_base(
    *,
    pair_ids: list[int],
    cache: Path,
    upstream: Path,
    hashes: dict[str, str],
    output_dir: Path,
    winner_policy: str,
) -> dict[str, Any]:
    config = {
        "pair_ids": pair_ids,
        "cache": str(cache),
        "upstream": str(upstream),
        "output_dir": str(output_dir),
        "seed": SEED,
        "camera_hw": list(CAMERA_HW),
        "scorer_hw": list(SCORER_HW),
        "producer_sha256": sha256_file(Path(__file__).resolve()),
        "library_sha256": sha256_file(SRC / "tac/optimization/vjp_custody.py"),
        "winner_policy": winner_policy,
    }
    return {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": None,
        "pair_ids": pair_ids,
        "receiver_arithmetic": RECEIVER_ARITHMETIC,
        "active_arrangement": (
            NATIVE_REFRESH_ARRANGEMENT if winner_policy == "fresh-native" else ACTIVE_ARRANGEMENT
        ),
        "representation": REPRESENTATION,
        "source_hashes": hashes,
        "config": config,
        "config_sha256": hashlib.sha256(canonical_json(config)).hexdigest(),
        "authority": {
            "axis": f"[{platform.system()}-{platform.machine()} CPU advisory]",
            "score_claim": False,
            "promotion_eligible": False,
            "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        },
        "invocation": {
            "argv": [str(value) for value in sys.argv],
            "python": sys.version,
            "git_head": _git_head(),
            "pid": os.getpid(),
        },
        "reconstruction": {
            "source_members": ["gt_f0", "gt_f1", "lstars", "margins"],
            "cache_access": "ZIP_STORED member memmap; whole archive is never loaded",
            "atomicity": "same-directory temporary NPZ plus os.replace",
            "cleanup": "success-only temporary files; final evidence never auto-deleted",
        },
        "sidecars": [],
    }


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _write_immutable_json(path: Path, value: Any, *, resume: bool) -> str:
    """Atomically create JSON without ever replacing an existing artifact."""

    payload = canonical_json(value) + b"\n"
    digest = hashlib.sha256(payload).hexdigest()

    def accept_existing() -> str:
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise VJPCustodyError(f"cannot inspect immutable JSON artifact: {path}") from exc
        if resume and existing == payload:
            return digest
        qualifier = "non-byte-identical " if resume else ""
        raise VJPCustodyError(f"immutable {qualifier}JSON artifact already exists: {path}")

    if path.exists():
        return accept_existing()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            return accept_existing()
    finally:
        if tmp.exists():
            tmp.unlink()
    return digest


def _refusal_payload(pair_id: int, error: VJPCustodyError, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": REFUSAL_SCHEMA,
        "pair_id": int(pair_id),
        "error": str(error),
        "verdict_scope": REFUSAL_VERDICT_SCOPE,
        "source_hashes": manifest["source_hashes"],
        "config": manifest["config"],
        "config_sha256": manifest["config_sha256"],
        "authority": {
            "score_claim": False,
            "pointer": manifest["authority"]["pointer"],
            "pointer_moved": False,
        },
    }


def _record_pair_refusal(
    *,
    output_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    pair_id: int,
    error: VJPCustodyError,
    resume: bool,
) -> dict[str, Any]:
    refusal_path = output_dir / f"pair_{pair_id:04d}.vjp_refusal.json"
    payload = _refusal_payload(pair_id, error, manifest)
    digest = _write_immutable_json(refusal_path, payload, resume=resume)
    row = {
        "pair_id": int(pair_id),
        "path": str(refusal_path.resolve()),
        "bytes": refusal_path.stat().st_size,
        "sha256": digest,
        "verdict_scope": REFUSAL_VERDICT_SCOPE,
    }
    refusals = manifest.setdefault("refusals", [])
    if not isinstance(refusals, list):
        raise VJPCustodyError("producer manifest refusals must be a list")
    matches = [existing for existing in refusals if existing.get("pair_id") == pair_id]
    if matches:
        if len(matches) != 1 or matches[0] != row:
            raise VJPCustodyError(f"producer manifest has conflicting refusal for pair {pair_id}")
        return row
    refusals.append(row)
    manifest["completed_at_utc"] = None
    manifest.pop("manifest_content_sha256", None)
    atomic_json(manifest_path, manifest)
    return row


def _resume_manifest(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    try:
        existing = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise VJPCustodyError(f"cannot resume manifest: {path}") from exc
    immutable_keys = (
        "schema", "pair_ids", "receiver_arithmetic", "active_arrangement",
        "representation", "source_hashes", "config", "config_sha256",
    )
    for key in immutable_keys:
        if existing.get(key) != expected.get(key):
            raise VJPCustodyError(f"resume manifest {key} mismatch")
    rows = existing.get("sidecars")
    if not isinstance(rows, list):
        raise VJPCustodyError("resume manifest sidecars must be a list")
    expected_prefix = expected["pair_ids"][: len(rows)]
    if [row.get("pair_id") for row in rows] != expected_prefix:
        raise VJPCustodyError("resume manifest completed sidecars are not a pair-order prefix")
    for row in rows:
        load_vjp_pair_row(row, existing)
    return existing


def _recover_orphaned_sidecars(
    output_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    """Append valid prefix sidecars orphaned after their atomic rename."""

    pair_ids = manifest["pair_ids"]
    rows = manifest["sidecars"]
    while len(rows) < len(pair_ids):
        pair_id = int(pair_ids[len(rows)])
        sidecar = output_dir / f"pair_{pair_id:04d}.vjp.npz"
        if not sidecar.exists():
            break
        row = recover_pair_sidecar_row(sidecar, pair_id, manifest)
        rows.append(row)
        manifest["completed_at_utc"] = None
        manifest.pop("manifest_content_sha256", None)
        atomic_json(manifest_path, manifest)


def _source_manifest_path(path: Path) -> Path:
    resolved = path.resolve()
    return resolved / "manifest.json" if resolved.is_dir() else resolved


def _load_composition_source(path: Path) -> tuple[Path, dict[str, Any], str]:
    manifest_path = _source_manifest_path(path)
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise VJPCustodyError(f"cannot read composition source manifest: {manifest_path}") from exc
    expected_common = {
        "schema": MANIFEST_SCHEMA,
        "receiver_arithmetic": RECEIVER_ARITHMETIC,
        "representation": REPRESENTATION,
    }
    for key, expected in expected_common.items():
        if manifest.get(key) != expected:
            raise VJPCustodyError(f"composition source manifest {key} mismatch: {manifest_path}")
    if manifest.get("active_arrangement") not in ACTIVE_ARRANGEMENTS:
        raise VJPCustodyError(f"composition source active arrangement mismatch: {manifest_path}")
    if manifest.get("source_hashes") != EXPECTED_HASHES:
        raise VJPCustodyError(f"composition source frozen hashes mismatch: {manifest_path}")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise VJPCustodyError(f"composition source config is missing: {manifest_path}")
    if manifest.get("config_sha256") != _content_sha256(config):
        raise VJPCustodyError(f"composition source config hash mismatch: {manifest_path}")
    claimed_content_hash = manifest.get("manifest_content_sha256")
    if claimed_content_hash is not None:
        actual_content_hash = _content_sha256(
            {key: value for key, value in manifest.items() if key != "manifest_content_sha256"}
        )
        if claimed_content_hash != actual_content_hash:
            raise VJPCustodyError(f"composition source content hash mismatch: {manifest_path}")
    pair_ids = manifest.get("pair_ids")
    rows = manifest.get("sidecars")
    if (
        not isinstance(pair_ids, list)
        or not pair_ids
        or len(pair_ids) > MAX_PAIRS
        or len(set(pair_ids)) != len(pair_ids)
        or not isinstance(rows, list)
    ):
        raise VJPCustodyError(f"composition source pair coverage is malformed: {manifest_path}")
    row_ids = [row.get("pair_id") for row in rows if isinstance(row, dict)]
    if len(row_ids) != len(rows) or row_ids != pair_ids[: len(rows)]:
        raise VJPCustodyError(f"composition source sidecars are not a unique ordered prefix: {manifest_path}")
    authority = manifest.get("authority")
    if (
        not isinstance(authority, dict)
        or authority.get("score_claim") is not False
        or authority.get("promotion_eligible") is not False
        or authority.get("pointer") != "0.1910828242 [contest-CPU] UNMOVED"
    ):
        raise VJPCustodyError(f"composition source authority is not advisory: {manifest_path}")
    return manifest_path, manifest, sha256_file(manifest_path)


def _common_source_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in {"pair_ids", "output_dir", "producer_sha256"}
    }


def _validate_refusal_row(row: dict[str, Any], manifest: dict[str, Any]) -> int:
    try:
        pair_id = int(row["pair_id"])
        path = Path(row["path"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VJPCustodyError("composition source refusal row is malformed") from exc
    if not path.is_absolute() or not path.is_file():
        raise VJPCustodyError(f"pair {pair_id} refusal path custody failed")
    if row.get("sha256") != sha256_file(path) or row.get("bytes") != path.stat().st_size:
        raise VJPCustodyError(f"pair {pair_id} refusal hash/path custody failed")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise VJPCustodyError(f"cannot read pair {pair_id} refusal custody") from exc
    expected = {
        "schema": REFUSAL_SCHEMA,
        "pair_id": pair_id,
        "verdict_scope": REFUSAL_VERDICT_SCOPE,
        "source_hashes": manifest["source_hashes"],
        "config": manifest["config"],
        "config_sha256": manifest["config_sha256"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise VJPCustodyError(f"pair {pair_id} refusal {key} custody mismatch")
    authority = payload.get("authority")
    if (
        not isinstance(payload.get("error"), str)
        or not payload["error"]
        or not isinstance(authority, dict)
        or authority.get("score_claim") is not False
        or authority.get("pointer_moved") is not False
        or authority.get("pointer") != manifest["authority"]["pointer"]
    ):
        raise VJPCustodyError(f"pair {pair_id} refusal authority/error custody mismatch")
    if row.get("verdict_scope") != REFUSAL_VERDICT_SCOPE:
        raise VJPCustodyError(f"pair {pair_id} refusal row verdict scope mismatch")
    return pair_id


def compose_manifests(args: argparse.Namespace) -> dict[str, Any]:
    """Finalize a zero-copy manifest over validated existing pair sidecars."""

    pair_ids = _pair_ids(args.pair_indices)
    source_inputs = [Path(path) for path in args.compose_manifests]
    if not source_inputs:
        raise VJPCustodyError("composition requires at least one source manifest")
    output_dir = _enforce_output_tier(args.output_dir.resolve(), 0)
    destination = output_dir / "manifest.json"
    if destination.exists():
        raise VJPCustodyError(f"composition destination already exists: {destination}")
    sacred_before = _stat_tree_snapshot(SACRED)

    sources = [_load_composition_source(path) for path in source_inputs]
    source_paths = [path for path, _, _ in sources]
    if len(set(source_paths)) != len(source_paths):
        raise VJPCustodyError("composition source manifests must be unique")
    common_configs = [_common_source_config(manifest["config"]) for _, manifest, _ in sources]
    if any(config != common_configs[0] for config in common_configs[1:]):
        raise VJPCustodyError("composition source manifests have incompatible producer configs")

    selected_rows: dict[int, dict[str, Any]] = {}
    declared_ids: list[int] = []
    refused_ids: list[int] = []
    source_sidecar_snapshots: dict[Path, tuple[int, int, str]] = {}
    for _, manifest, _ in sources:
        for pair_id in manifest["pair_ids"]:
            if pair_id not in declared_ids:
                declared_ids.append(pair_id)
        refusals = manifest.get("refusals", [])
        if not isinstance(refusals, list):
            raise VJPCustodyError("composition source refusals must be a list")
        seen_refusals: set[int] = set()
        for refusal in refusals:
            if not isinstance(refusal, dict):
                raise VJPCustodyError("composition source refusal row is malformed")
            refused_id = _validate_refusal_row(refusal, manifest)
            if refused_id in seen_refusals or refused_id not in manifest["pair_ids"]:
                raise VJPCustodyError("composition source refusal coverage is invalid")
            seen_refusals.add(refused_id)
            if refused_id not in refused_ids:
                refused_ids.append(refused_id)
        for row in manifest["sidecars"]:
            pair_id = int(row["pair_id"])
            if pair_id in seen_refusals:
                raise VJPCustodyError(
                    f"pair {pair_id} cannot be both completed and refused in one source manifest"
                )
            if pair_id not in pair_ids:
                continue
            if pair_id in selected_rows:
                raise VJPCustodyError(f"selected pair {pair_id} appears in multiple source manifests")
            sidecar_path = Path(row["path"])
            if not sidecar_path.is_absolute():
                raise VJPCustodyError(f"selected pair {pair_id} sidecar path is not absolute")
            load_vjp_pair_row(row, manifest)
            selected_rows[pair_id] = dict(row)
            source_sidecar_snapshots[sidecar_path] = (
                sidecar_path.stat().st_mtime_ns,
                sidecar_path.stat().st_size,
                sha256_file(sidecar_path),
            )
    missing = [pair_id for pair_id in pair_ids if pair_id not in selected_rows]
    if missing:
        raise VJPCustodyError(f"composition lacks selected sidecar rows: {missing}")
    ordered_rows = [selected_rows[pair_id] for pair_id in pair_ids]
    omitted_ids = [pair_id for pair_id in declared_ids if pair_id not in pair_ids]
    source_records = [
        {"path": str(path), "sha256": digest}
        for path, _, digest in sources
    ]
    config = {
        "mode": "zero_copy_manifest_composition",
        "pair_ids": pair_ids,
        "output_dir": str(output_dir),
        "source_manifests": source_records,
        "source_common_config": common_configs[0],
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "pair_ids": pair_ids,
        "receiver_arithmetic": RECEIVER_ARITHMETIC,
        "active_arrangement": sources[0][1]["active_arrangement"],
        "representation": REPRESENTATION,
        "source_hashes": EXPECTED_HASHES,
        "config": config,
        "config_sha256": _content_sha256(config),
        "authority": {
            "axis": f"[{platform.system()}-{platform.machine()} CPU advisory]",
            "score_claim": False,
            "promotion_eligible": False,
            "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        },
        "composition": {
            "zero_copy": True,
            "requested_selected_pair_ids": pair_ids,
            "omitted_pair_ids": omitted_ids,
            "refused_pair_ids": refused_ids,
            "refusal_scope": REFUSAL_VERDICT_SCOPE,
            "refusals_are_family_negatives": False,
        },
        "source_manifests": source_records,
        "sidecars": ordered_rows,
    }
    manifest["manifest_content_sha256"] = _content_sha256(manifest)
    if _stat_tree_snapshot(SACRED) != sacred_before:
        raise VJPCustodyError("sacred result tree changed during VJP manifest composition")
    for path, snapshot in source_sidecar_snapshots.items():
        current = (path.stat().st_mtime_ns, path.stat().st_size, sha256_file(path))
        if current != snapshot:
            raise VJPCustodyError(f"source sidecar changed during composition: {path}")
    for path, _, digest in sources:
        if sha256_file(path) != digest:
            raise VJPCustodyError(f"source manifest changed during composition: {path}")
    _write_immutable_json(destination, manifest, resume=False)
    if _stat_tree_snapshot(SACRED) != sacred_before:
        raise VJPCustodyError("sacred result tree changed during VJP manifest composition")
    return manifest


def produce(args: argparse.Namespace) -> dict[str, Any]:
    pair_ids = _pair_ids(args.pair_indices)
    cache, upstream = args.cache.resolve(), args.upstream.resolve()
    provisional_output = args.output_dir.resolve()
    if (
        provisional_output == DEFAULT_OUTPUT.resolve()
        and not DEFAULT_OUTPUT.resolve().parents[1].exists()
        and SECONDARY_OUTPUT.resolve().parents[1].exists()
    ):
        provisional_output = SECONDARY_OUTPUT.resolve()
    manifest_path = provisional_output / "manifest.json"
    completed_count = 0
    if manifest_path.is_file():
        try:
            completed_count = len(json.loads(manifest_path.read_text()).get("sidecars", []))
        except (OSError, json.JSONDecodeError, AttributeError):
            completed_count = 0
    output_dir = _enforce_output_tier(provisional_output, max(0, len(pair_ids) - completed_count))
    manifest_path = output_dir / "manifest.json"
    sacred_before = _stat_tree_snapshot(SACRED)
    hashes = source_hashes(cache, upstream)
    expected = _manifest_base(
        pair_ids=pair_ids,
        cache=cache,
        upstream=upstream,
        hashes=hashes,
        output_dir=output_dir,
        winner_policy=args.winner_policy,
    )
    if manifest_path.exists():
        if not args.resume:
            raise VJPCustodyError(f"manifest exists; use --resume after validation: {manifest_path}")
        manifest = _resume_manifest(manifest_path, expected)
        _recover_orphaned_sidecars(output_dir, manifest_path, manifest)
    else:
        if args.resume:
            raise VJPCustodyError(f"--resume requested but manifest is absent: {manifest_path}")
        manifest = expected
        atomic_json(manifest_path, manifest)

    completed = {int(row["pair_id"]) for row in manifest["sidecars"]}
    fields = None
    segnet = posenet = torch = None
    if len(completed) != len(pair_ids):
        fields = _load_cache(cache)
        segnet, posenet, torch = _load_scorers(upstream, args.cpu_threads)
    for pair_id in pair_ids:
        if pair_id in completed:
            continue
        assert fields is not None and segnet is not None and posenet is not None and torch is not None
        try:
            arrays = compute_pair_derivatives(
                pair_id=pair_id,
                frame0=np.asarray(fields["gt_f0"][pair_id], dtype=np.uint8).copy(),
                frame1=np.asarray(fields["gt_f1"][pair_id], dtype=np.uint8).copy(),
                cached_winner=np.asarray(fields["lstars"][pair_id], dtype=np.int64).copy(),
                cached_margin=np.asarray(fields["margins"][pair_id], dtype=np.float32).copy(),
                segnet=segnet,
                posenet=posenet,
                torch=torch,
                refresh_native_winner=args.winner_policy == "fresh-native",
            )
        except VJPCustodyError as error:
            _record_pair_refusal(
                output_dir=output_dir,
                manifest_path=manifest_path,
                manifest=manifest,
                pair_id=pair_id,
                error=error,
                resume=args.resume,
            )
            if _stat_tree_snapshot(SACRED) != sacred_before:
                raise VJPCustodyError("sacred result tree changed during VJP production") from error
            raise
        sidecar = output_dir / f"pair_{pair_id:04d}.vjp.npz"
        row = write_pair_sidecar(sidecar, arrays, hashes)
        manifest["sidecars"].append(row)
        manifest["completed_at_utc"] = None
        atomic_json(manifest_path, manifest)
        if _stat_tree_snapshot(SACRED) != sacred_before:
            raise VJPCustodyError("sacred result tree changed during VJP production")

    manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
    manifest["manifest_content_sha256"] = hashlib.sha256(
        canonical_json({key: value for key, value in manifest.items() if key != "manifest_content_sha256"})
    ).hexdigest()
    atomic_json(manifest_path, manifest)
    if _stat_tree_snapshot(SACRED) != sacred_before:
        raise VJPCustodyError("sacred result tree changed during VJP production")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Produce 1..12 immutable real frozen CPU-Torch Seg/Pose VJP sidecars."
    )
    parser.add_argument("--pair-indices", nargs="+", type=int, required=True)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--compose-manifests",
        nargs="+",
        type=Path,
        help="Finalize a zero-copy manifest from existing complete or partial producer manifests.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--winner-policy",
        choices=("cached-verified", "fresh-native"),
        default="cached-verified",
        help="Use the cache-verified arrangement or explicitly refresh the native winner/rival chart.",
    )
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.cpu_threads <= 0:
        raise SystemExit("--cpu-threads must be positive")
    if args.compose_manifests and args.resume:
        raise SystemExit("--resume is incompatible with immutable --compose-manifests output")
    manifest = compose_manifests(args) if args.compose_manifests else produce(args)
    print(
        json.dumps(
            {
                "manifest": str(Path(manifest["config"]["output_dir"]) / "manifest.json"),
                "pair_count": len(manifest["sidecars"]),
                "schema": manifest["schema"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
