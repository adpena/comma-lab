#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Resumable real-n600 probe for the frozen-replay convex costate head.

The probe never mutates a source checkpoint or live run.  It renders one fixed
state for each of the 600 witness pairs, calls the exact CPU SegNet teacher once
per state, and discards raw training costates after writing objective-exact
ridge sufficient statistics.  Held-out labels are consumed once after fitting;
their reduced costate and renderer-VJP fidelity records are independently
resumable.  No MPS or evaluator-score claim is present.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import platform
import struct
import subprocess
import sys
import time
import types
import zipfile
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.scorer_surrogate.frozen_replay_convex_head import (  # noqa: E402
    AUTHORITY_SCOPE,
    FEATURE_NAMES,
    RESEARCH_ONLY,
    ReplayAssignment,
    StateSufficientStatistics,
    aggregate_sufficient_statistics,
    array_sha256,
    cache_exact_label_sufficient_statistics,
    derive_mission_verdict,
    deterministic_replay_assignments,
    fit_cached_convex_head,
    frozen_feature_matrix,
    predict_costate,
    sampled_costate_rows,
    teacher_call_accounting,
    vector_fidelity,
)
from tac.witness_dsl.frozen_replay_convex_head_policy import (  # noqa: E402
    FrozenReplayConvexHeadPolicy,
)

SCHEMA = "frozen_replay_convex_head_probe.v1"
LANE_ID = "lane_95kill_round2_frozen_replay_convex_head_20260713"
AXIS = "[macOS-CPU advisory; numpy-fp32 training-gradient evidence; no score authority]"

GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
SEGNET = REPO / "upstream/models/segnet.safetensors"
CHECKPOINT_DIR = REPO / "experiments/results/v9_cgauge_432_coherent_arm_20260711"
CHECKPOINTS = (
    ("v9_ep150_ema_best", CHECKPOINT_DIR / "levelset_witness_ema_BEST.npz", 150),
    ("v9_ep251_stage_octave1", CHECKPOINT_DIR / "levelset_ckpt_stageOctave1_ep251.npz", 251),
    ("v9_ep275_ema_final", CHECKPOINT_DIR / "levelset_witness_ema_mlx.npz", 275),
)

EXPECTED_INPUTS = {
    str(GT_CACHE.relative_to(REPO)): {
        "bytes": 5_078_017_610,
        "sha256": "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6",
    },
    str(SEGNET.relative_to(REPO)): {
        "bytes": 38_502_892,
        "sha256": "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6",
    },
    str(CHECKPOINTS[0][1].relative_to(REPO)): {
        "bytes": 379_776,
        "sha256": "2599ad8b396af2af220a3bdbeee2ade92f194771ae6ef01a6faa15d39333484c",
    },
    str(CHECKPOINTS[1][1].relative_to(REPO)): {
        "bytes": 380_136,
        "sha256": "c59cdec6eec16677c0a2eb5667979dd1c8f883bcd1cf5532302d67acd633c758",
    },
    str(CHECKPOINTS[2][1].relative_to(REPO)): {
        "bytes": 380_136,
        "sha256": "1676e4d45e180c7a28ec2ecce2b932d0e5087a2cfec2636ff2efe1673dbbcbf0",
    },
}

SOURCE_FILES = (
    "src/tac/scorer_surrogate/frozen_replay_convex_head.py",
    "src/tac/scorer_surrogate/amortized_onpolicy_costate.py",
    "src/tac/scorer_surrogate/onpolicy_costate.py",
    "src/tac/scorer_surrogate/onpolicy_matched_verdict.py",
    "src/tac/witness_dsl/frozen_replay_convex_head_policy.py",
    "src/tac/witness_dsl/onpolicy_scorer_surrogate_policy.py",
    "src/tac/canonical_equations/frozen_replay_convex_head_contraction_20260713.py",
    "tools/probe_frozen_replay_convex_head.py",
    "tools/probe_yopo_first_layer_costate.py",
    "tools/probe_onpolicy_costate_matched_window.py",
    "src/tac/boundary_math/segnet_gradient_replacement.py",
    "src/tac/boundary_math/seg_core.py",
    "src/tac/cuda_levelset_training.py",
    "src/tac/local_acceleration/torch_levelset_inflate.py",
    "src/tac/witness_annulus_metrics.py",
    "tools/dash_comb_probe_n600.py",
    "upstream/modules.py",
    ".omx/research/frozen_replay_convex_head_contraction_DAG_FEED_20260713.md",
)

SOURCE_AMENDMENT_ID = "fit-ratio-scale-floor-v1"
SOURCE_AMENDMENT_CHANGED_PATHS = frozenset(
    {
        "src/tac/scorer_surrogate/frozen_replay_convex_head.py",
        "tools/probe_frozen_replay_convex_head.py",
    }
)
SOURCE_AMENDMENT_OLD_SHA256 = {
    "src/tac/scorer_surrogate/frozen_replay_convex_head.py": (
        "74234d894552a64323411a8783026cd84b908b2814f7f7716ce05cf0cec73385"
    ),
    "tools/probe_frozen_replay_convex_head.py": (
        "8881c755d83088d878dc232f884bf36450425a1aa65f13b4b8e50a67c9c1079e"
    ),
}


class ProbeError(RuntimeError):
    """The measurement contract or its custody failed closed."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_bytes(path, _json_bytes(payload))


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Any) -> None:
    """Persist an event atomically before updating its human-readable JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    event_dir = path.with_name(f"{path.name}.events")
    event_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{time.time_ns():020d}-{os.getpid()}"
    event_path = event_dir / f"{stem}.json"
    suffix = 0
    while event_path.exists():
        suffix += 1
        event_path = event_dir / f"{stem}-{suffix}.json"
    _atomic_json(event_path, payload)
    with path.open("ab") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _event_ledger_rows(path: Path) -> list[dict[str, Any]]:
    event_dir = path.with_name(f"{path.name}.events")
    if not event_dir.is_dir():
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows: list[dict[str, Any]] = []
    for event_path in sorted(event_dir.glob("*.json")):
        try:
            row = json.loads(event_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ProbeError(f"atomic event ledger corruption at {event_path}") from exc
        if not isinstance(row, dict) or "event" not in row:
            raise ProbeError(f"invalid atomic event payload at {event_path}")
        rows.append(row)
    return rows


def _canonicalize_event_ledger(path: Path) -> list[dict[str, Any]]:
    rows = _event_ledger_rows(path)
    payload = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )
    _atomic_bytes(path, payload)
    return rows


def _load_tool_module(name: str, relative_path: str) -> Any:
    path = REPO / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProbeError(f"cannot import committed helper {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def _git_status() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"], cwd=REPO, check=True, text=True, capture_output=True
    )
    return result.stdout.splitlines()


def _runtime_custody(torch: Any) -> dict[str, Any]:
    cpu = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        text=True,
        capture_output=True,
        check=False,
    )
    environment_keys = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "PYTHONHASHSEED",
    )
    sealed_numerical_identity = {
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_brand": cpu.stdout.strip() if cpu.returncode == 0 else "UNAVAILABLE",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "environment": {key: os.environ.get(key) for key in environment_keys},
        "mps_used": False,
        "cuda_used": False,
    }
    return {
        "sealed_numerical_identity": sealed_numerical_identity,
        "invocation": {
            "argv": list(sys.argv),
            "executable": sys.executable,
            "cwd": str(Path.cwd()),
            "captured_at_utc": _utc_now(),
        },
    }


def _storage_preflight_custody(output_dir: Path) -> dict[str, Any]:
    path = REPO / ".omx/research/frozen_replay_convex_head_storage_preflight_20260713.json"
    if not path.is_file():
        raise ProbeError("missing durable storage preflight")
    payload = json.loads(path.read_text())
    if payload.get("blockers"):
        raise ProbeError(f"storage preflight remains blocked: {payload['blockers']}")
    if Path(payload.get("selected_workload_root", "")).resolve() != output_dir.resolve():
        raise ProbeError("storage preflight selected a different workload root")
    return {
        "path": str(path.relative_to(REPO)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "selected_tier": payload["selected_tier"],
        "selected_workload_root": payload["selected_workload_root"],
        "requested_bytes": payload["requested_bytes"],
        "usable_bytes": next(
            row["usable_bytes"]
            for row in payload["tiers"]
            if row["name"] == payload["selected_tier"]
        ),
        "explicit_local_opt_in": payload["operator_storage_policy"]["local_disk_enabled"],
        "blockers": [],
    }


def _load_cpu_segnet() -> Any:
    """Load only the exact teacher used by this formulation; PoseNet is out of scope."""

    from safetensors.torch import load_file

    upstream = str(REPO / "upstream")
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    from modules import SegNet

    segnet = SegNet().eval()
    segnet.load_state_dict(load_file(str(SEGNET), device="cpu"))
    for parameter in segnet.parameters():
        parameter.requires_grad_(False)
    return segnet


def _verify_input_custody() -> dict[str, dict[str, Any]]:
    measured: dict[str, dict[str, Any]] = {}
    for relative, expected in EXPECTED_INPUTS.items():
        path = REPO / relative
        if not path.is_file():
            raise ProbeError(f"missing sealed input: {relative}")
        observed = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        if observed != expected:
            raise ProbeError(f"sealed input custody drift for {relative}: {observed} != {expected}")
        measured[relative] = {**observed, "status": "MEASURED", "read_only": True}
    return measured


def _source_bundle(
    output_dir: Path, *, bundle_relative: Path = Path("source_bundle")
) -> dict[str, dict[str, Any]]:
    if bundle_relative.is_absolute() or ".." in bundle_relative.parts:
        raise ProbeError("source bundle path must remain relative to the run directory")
    bundle_dir = output_dir / bundle_relative
    manifest: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_FILES:
        source = REPO / relative
        if not source.is_file():
            raise ProbeError(f"missing source-custody file {relative}")
        destination = bundle_dir / relative
        payload = source.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if destination.exists() and (
            destination.stat().st_size != len(payload) or _sha256(destination) != digest
        ):
            raise ProbeError(f"existing source bundle drifted: {destination}")
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            _atomic_bytes(destination, payload)
        manifest[relative] = {
            "path": str(destination.relative_to(output_dir)),
            "bytes": len(payload),
            "sha256": digest,
        }
    return manifest


def _source_fingerprints() -> dict[str, dict[str, Any]]:
    measured: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_FILES:
        source = REPO / relative
        if not source.is_file():
            raise ProbeError(f"missing source-custody file {relative}")
        measured[relative] = {
            "bytes": source.stat().st_size,
            "sha256": _sha256(source),
        }
    return measured


def _custody_fingerprints(custody: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        relative: {"bytes": row["bytes"], "sha256": row["sha256"]}
        for relative, row in custody.items()
    }


def _verify_recovery_boundary(
    output_dir: Path, *, expected_train_pairs: set[int]
) -> dict[str, Any]:
    """Verify the immutable teacher/cache boundary before a source amendment."""

    stage_path = output_dir / "stage_train_cache_complete.json"
    if not stage_path.is_file():
        raise ProbeError("source amendment requires a completed training-cache stage")
    stage = json.loads(stage_path.read_text())
    if stage.get("state_count") != len(expected_train_pairs):
        raise ProbeError("source-amendment training-cache cardinality drift")
    if {int(value) for value in stage.get("records", {})} != expected_train_pairs:
        raise ProbeError("source-amendment training-cache pair coverage drift")
    record_manifest: list[dict[str, Any]] = []
    for pair_index in sorted(expected_train_pairs):
        custody = stage["records"][str(pair_index)]
        path = (output_dir / custody["path"]).resolve()
        try:
            path.relative_to(output_dir.resolve())
        except ValueError as exc:
            raise ProbeError("source-amendment cache record escapes run directory") from exc
        if (
            not path.is_file()
            or path.stat().st_size != custody["bytes"]
            or _sha256(path) != custody["sha256"]
        ):
            raise ProbeError(f"source-amendment cache record drift: pair {pair_index}")
        record_manifest.append(
            {
                "pair_index": pair_index,
                "path": custody["path"],
                "bytes": custody["bytes"],
                "sha256": custody["sha256"],
            }
        )

    event_dir = output_dir / "teacher_calls.jsonl.events"
    event_paths = sorted(event_dir.glob("*.json"))
    starts: set[int] = set()
    completions: set[int] = set()
    batch_completions = 0
    event_manifest: list[dict[str, Any]] = []
    for path in event_paths:
        row = json.loads(path.read_text())
        if row.get("stage") != "train_cache":
            # A crash-resume after amendment creation may already have a valid
            # held-out prefix.  The sealed recovery boundary is exactly the
            # immutable training-cache subset, so later-stage events neither
            # enlarge nor weaken it.
            continue
        if row.get("split", "train") != "train":
            raise ProbeError("training-cache source-amendment event has non-train split")
        event = row.get("event")
        if event == "exact_teacher_state_call_started":
            starts.add(int(row["pair_index"]))
        elif event == "exact_teacher_state_call_completed":
            completions.add(int(row["pair_index"]))
        elif event == "exact_teacher_batch_completed":
            if row.get("state_count") != 1:
                raise ProbeError("source-amendment boundary contains a non-unit teacher batch")
            batch_completions += 1
        else:
            raise ProbeError(f"unexpected teacher event at source-amendment boundary: {event}")
        event_manifest.append(
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if starts != expected_train_pairs or completions != expected_train_pairs:
        raise ProbeError("source-amendment teacher start/completion coverage drift")
    if batch_completions != len(expected_train_pairs):
        raise ProbeError("source-amendment teacher batch coverage drift")
    if len(event_manifest) != 3 * len(expected_train_pairs):
        raise ProbeError("source-amendment teacher event cardinality drift")

    weights_path = output_dir / "fit" / "convex_head_weights.npz"
    if not weights_path.is_file():
        raise ProbeError("source amendment requires preserved pre-manifest fit weights")
    return {
        "train_cache_stage": {
            "path": str(stage_path.relative_to(output_dir)),
            "bytes": stage_path.stat().st_size,
            "sha256": _sha256(stage_path),
            "record_count": len(record_manifest),
            "record_tree_sha256": hashlib.sha256(
                json.dumps(record_manifest, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "teacher_events": {
            "atomic_event_directory": str(event_dir.relative_to(output_dir)),
            "atomic_event_file_count": len(event_manifest),
            "atomic_event_tree_sha256": hashlib.sha256(
                json.dumps(event_manifest, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "unique_started_states": len(starts),
            "unique_completed_states": len(completions),
            "batch_completions": batch_completions,
            "teacher_calls_recomputed_by_amendment": 0,
        },
        "pre_manifest_fit_weights": {
            "path": str(weights_path.relative_to(output_dir)),
            "bytes": weights_path.stat().st_size,
            "sha256": _sha256(weights_path),
        },
    }


def _resolve_source_custody(
    output_dir: Path,
    *,
    prior_contract: dict[str, Any],
    requested_amendment: str | None,
    expected_train_pairs: set[int],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    """Resolve exact source bytes, allowing one explicit append-only verifier repair."""

    prior_sources = prior_contract["sources"]
    current_fingerprints = _source_fingerprints()
    if current_fingerprints == _custody_fingerprints(prior_sources):
        if requested_amendment is not None:
            raise ProbeError("source amendment requested but no source drift exists")
        return _source_bundle(output_dir), None

    amendment_path = output_dir / f"source_amendment_{SOURCE_AMENDMENT_ID}.json"
    if amendment_path.is_file():
        amendment = json.loads(amendment_path.read_text())
        if amendment.get("amendment_id") != SOURCE_AMENDMENT_ID:
            raise ProbeError("source-amendment identity drift")
        if amendment.get("old_sources") != prior_sources:
            raise ProbeError("source-amendment old-source custody drift")
        if current_fingerprints != _custody_fingerprints(amendment["new_sources"]):
            raise ProbeError("current sources drifted from the sealed source amendment")
        boundary = _verify_recovery_boundary(
            output_dir, expected_train_pairs=expected_train_pairs
        )
        if boundary != amendment.get("recovery_boundary"):
            raise ProbeError("source-amendment recovery boundary drifted")
        effective = _source_bundle(
            output_dir,
            bundle_relative=Path("source_bundle_amendments") / SOURCE_AMENDMENT_ID,
        )
        if effective != amendment["new_sources"]:
            raise ProbeError("source-amendment bundle custody drift")
        custody = {
            "path": str(amendment_path.relative_to(output_dir)),
            "bytes": amendment_path.stat().st_size,
            "sha256": _sha256(amendment_path),
            "amendment_id": SOURCE_AMENDMENT_ID,
        }
        return effective, custody

    if requested_amendment != SOURCE_AMENDMENT_ID:
        raise ProbeError(
            "resume source custody drift; the explicit verifier source amendment was not requested"
        )
    changed = {
        relative
        for relative in SOURCE_FILES
        if current_fingerprints[relative]
        != _custody_fingerprints(prior_sources)[relative]
    }
    if changed != SOURCE_AMENDMENT_CHANGED_PATHS:
        raise ProbeError(f"source amendment has an unapproved source delta: {sorted(changed)}")
    for relative, expected_sha256 in SOURCE_AMENDMENT_OLD_SHA256.items():
        if prior_sources[relative]["sha256"] != expected_sha256:
            raise ProbeError(f"source amendment old hash drift for {relative}")

    boundary = _verify_recovery_boundary(
        output_dir, expected_train_pairs=expected_train_pairs
    )
    effective = _source_bundle(
        output_dir,
        bundle_relative=Path("source_bundle_amendments") / SOURCE_AMENDMENT_ID,
    )
    delta = {
        relative: {
            "old": prior_sources[relative],
            "new": effective[relative],
        }
        for relative in sorted(changed)
    }
    amendment = {
        "schema": "frozen_replay_source_amendment.v1",
        "amendment_id": SOURCE_AMENDMENT_ID,
        "created_at_utc": _utc_now(),
        "reason": (
            "repair a dimensionally invalid absolute ratio-admission floor after the exact "
            "480-state cache was sealed; teacher, replay, feature chart, objective, optimizer, "
            "and fitted arrays are unchanged"
        ),
        "verdict_scope": "verifier numeric-floor implementation only; not the formulation",
        "old_sources": prior_sources,
        "new_sources": effective,
        "source_delta": delta,
        "recovery_boundary": boundary,
        "teacher_calls_reused": len(expected_train_pairs),
        "teacher_calls_recomputed": 0,
        "pointer_moved": False,
    }
    _atomic_json(amendment_path, amendment)
    custody = {
        "path": str(amendment_path.relative_to(output_dir)),
        "bytes": amendment_path.stat().st_size,
        "sha256": _sha256(amendment_path),
        "amendment_id": SOURCE_AMENDMENT_ID,
    }
    return effective, custody


def _write_cleanup_manifest(output_dir: Path, *, phase: str) -> dict[str, Any]:
    """Certify and remove only abandoned atomic-write scratch.

    Raw frames and costates are process-local and never written.  Every other
    artifact is small, durable evidence and is therefore preserved.
    """

    manifest_path = output_dir / "cleanup_manifest.json"
    prior_removed: list[dict[str, Any]] = []
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        prior_removed = list(prior.get("removed_scratch", []))
    removed: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob(".*.tmp.*")):
        if not path.is_file():
            continue
        prefix, separator, _pid = path.name[1:].partition(".tmp.")
        destination = path.with_name(prefix) if separator and prefix else None
        certification = {
            "original_path": str(path.relative_to(output_dir)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "cold_store_destination": None,
            "false_authority_score_flags": {
                "score_claim": False,
                "promotion_eligible": False,
            },
        }
        if destination is not None and not destination.exists():
            recoverable = False
            recovery_reason = ""
            try:
                if destination.suffix == ".json":
                    json.loads(path.read_text())
                    recoverable = True
                    recovery_reason = "complete JSON atomic payload"
                elif destination.suffix == ".jsonl":
                    for line in path.read_text().splitlines():
                        if line.strip():
                            json.loads(line)
                    recoverable = True
                    recovery_reason = "complete JSONL atomic payload"
                elif destination.suffix == ".npz":
                    with np.load(path, allow_pickle=False) as archive:
                        for key in archive.files:
                            np.asarray(archive[key])
                    recoverable = True
                    recovery_reason = "complete CRC-readable NPZ atomic payload"
                elif "source_bundle" in destination.parts:
                    relative = destination.relative_to(output_dir / "source_bundle")
                    source = REPO / relative
                    recoverable = source.is_file() and _sha256(source) == _sha256(path)
                    recovery_reason = "bytes match the still-present source file"
            except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile):
                recoverable = False
            if recoverable:
                os.replace(path, destination)
                certification.update(
                    {
                        "action": "lossless_atomic_recovery_to_destination",
                        "reason": recovery_reason,
                        "rebuildable": True,
                        "intended_destination": str(destination.relative_to(output_dir)),
                        "destination_bytes": destination.stat().st_size,
                        "destination_sha256": _sha256(destination),
                    }
                )
                removed.append(certification)
                continue
        if destination is None or not destination.is_file():
            quarantine_dir = output_dir / "recovery_quarantine"
            quarantine_dir.mkdir(parents=True, exist_ok=True)
            quarantine = quarantine_dir / f"{_sha256(path)}.bin"
            try:
                if quarantine.exists() and _sha256(quarantine) != _sha256(path):
                    raise ProbeError("recovery-quarantine SHA collision")
                if quarantine.exists():
                    path.unlink()
                else:
                    os.replace(path, quarantine)
                certification.update(
                    {
                        "action": "moved_losslessly_to_recovery_quarantine",
                        "reason": (
                            "atomic payload was incomplete or unrecognized; bytes preserved while "
                            "the sealed source/cache operation is retried"
                        ),
                        "rebuildable": True,
                        "intended_destination": (
                            None if destination is None else str(destination.relative_to(output_dir))
                        ),
                        "quarantine_path": str(quarantine.relative_to(output_dir)),
                        "retry_teacher_calls_are_conservatively_charged": True,
                    }
                )
                removed.append(certification)
                continue
            except OSError:
                pass
            certification.update(
                {
                    "action": "PRESERVED_BLOCKER",
                    "reason": "atomic destination absent; scratch may be the only durable copy",
                    "rebuildable": "UNPROVEN",
                    "intended_destination": (
                        None if destination is None else str(destination.relative_to(output_dir))
                    ),
                }
            )
            blockers.append(certification)
            continue
        certification.update(
            {
                "action": "deleted_certified_redundant_atomic_scratch",
                "reason": "intended destination exists and is content-addressed below",
                "rebuildable": True,
                "intended_destination": str(destination.relative_to(output_dir)),
                "destination_bytes": destination.stat().st_size,
                "destination_sha256": _sha256(destination),
            }
        )
        path.unlink()
        removed.append(certification)
    durable = [
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.name not in {".single_writer.lock", "cleanup_manifest.json"}
        and ".tmp." not in path.name
    ]
    manifest = {
        "schema": "frozen_replay_convex_head_cleanup.v1",
        "phase": phase,
        "updated_at_utc": _utc_now(),
        "root": str(output_dir),
        "raw_frame_tree_materialized": False,
        "raw_costate_tree_materialized": False,
        "large_artifacts_created": False,
        "preservation_policy": "all compact sufficient statistics, call logs, stage checkpoints, and receipts preserved",
        "automatic_cleanup": "certified abandoned atomic-write scratch only",
        "removed_scratch": prior_removed + removed,
        "preserved_scratch_blockers": blockers,
        "durable_file_count_excluding_manifest_and_lock": len(durable),
        "durable_bytes_excluding_manifest_and_lock": sum(path.stat().st_size for path in durable),
        "storage_preflight": ".omx/research/frozen_replay_convex_head_storage_preflight_20260713.json",
        "authority": AXIS,
    }
    _atomic_json(manifest_path, manifest)
    if blockers:
        raise ProbeError(
            "cleanup certify-or-block preserved atomic scratch without a verified destination"
        )
    return manifest


def _acquire_lock(output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ".single_writer.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise ProbeError(f"another writer owns {output_dir}") from exc
    return descriptor


def _release_lock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _completed_receipt_or_none(output_dir: Path, *, resume: bool) -> dict[str, Any] | None:
    """Return a hash-verified terminal receipt without mutating sacred bytes."""

    complete_path = output_dir / "complete.json"
    if not complete_path.is_file():
        return None
    if not resume:
        raise ProbeError(f"completed run directory is sacred: {output_dir}")
    completion = json.loads(complete_path.read_text())
    receipt_path = (output_dir / completion["receipt"]).resolve()
    try:
        receipt_path.relative_to(output_dir.resolve())
    except ValueError as exc:
        raise ProbeError("completed receipt path escapes the run directory") from exc
    if (
        not receipt_path.is_file()
        or receipt_path.stat().st_size != completion["bytes"]
        or _sha256(receipt_path) != completion["sha256"]
    ):
        raise ProbeError("completed run receipt custody drifted")
    receipt = json.loads(receipt_path.read_text())
    dependent_custodies = (
        receipt["initial_run_contract_custody"],
        receipt["invocation_custody"],
        receipt["cleanup_custody"],
        receipt["teacher_call_accounting"]["teacher_call_ledger"],
    )
    if receipt.get("source_amendment_custody") is not None:
        dependent_custodies = (*dependent_custodies, receipt["source_amendment_custody"])
    for custody in dependent_custodies:
        path = (output_dir / custody["path"]).resolve()
        try:
            path.relative_to(output_dir.resolve())
        except ValueError as exc:
            raise ProbeError("completed dependent custody path escapes the run directory") from exc
        if (
            not path.is_file()
            or path.stat().st_size != custody["bytes"]
            or _sha256(path) != custody["sha256"]
        ):
            raise ProbeError(f"completed dependent custody drifted: {custody['path']}")
        if "atomic_event_directory" in custody:
            event_dir = (output_dir / custody["atomic_event_directory"]).resolve()
            event_files = sorted(event_dir.glob("*.json"))
            manifest = [
                {
                    "path": str(event.relative_to(output_dir)),
                    "bytes": event.stat().st_size,
                    "sha256": _sha256(event),
                }
                for event in event_files
            ]
            tree_sha256 = hashlib.sha256(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if (
                len(event_files) != custody["atomic_event_file_count"]
                or tree_sha256 != custody["atomic_event_tree_sha256"]
            ):
                raise ProbeError(
                    f"completed atomic event-tree custody drifted: {custody['atomic_event_directory']}"
                )
    for source in receipt.get("effective_source_custody", {}).values():
        path = (output_dir / source["path"]).resolve()
        try:
            path.relative_to(output_dir.resolve())
        except ValueError as exc:
            raise ProbeError("completed effective-source path escapes the run directory") from exc
        if (
            not path.is_file()
            or path.stat().st_size != source["bytes"]
            or _sha256(path) != source["sha256"]
        ):
            raise ProbeError(f"completed effective-source custody drifted: {source['path']}")
    return receipt


def _stored_npy_memmap(npz_path: Path, member: str) -> np.memmap:
    """Memory-map one ZIP_STORED NPY member without materializing GT bulk."""

    with zipfile.ZipFile(npz_path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 0x1:
            raise ProbeError(f"{member} is not an unencrypted ZIP_STORED member")
        with npz_path.open("rb") as handle:
            handle.seek(info.header_offset)
            local = handle.read(30)
            if len(local) != 30:
                raise ProbeError(f"truncated local header for {member}")
            signature, *_fields, name_length, extra_length = struct.unpack("<IHHHHHIIIHH", local)
            if signature != 0x04034B50:
                raise ProbeError(f"bad local ZIP signature for {member}")
            handle.seek(name_length + extra_length, os.SEEK_CUR)
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version in {(2, 0), (3, 0)}:
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ProbeError(f"unsupported NPY version {version} for {member}")
            data_offset = handle.tell()
    order = "F" if fortran else "C"
    return np.memmap(npz_path, mode="r", dtype=dtype, shape=shape, offset=data_offset, order=order)


def _render_state_nchw(renderer: Any, pair_index: int) -> Any:
    """Settled non-gradient renderer plus the committed contest-R receiver."""

    import torch

    from tac.cuda_levelset_training import contest_r

    bulk, _lane = renderer.render_pair(pair_index)
    frame_nhwc = contest_r(torch.as_tensor(bulk, dtype=torch.float32).unsqueeze(0))
    return frame_nhwc.permute(0, 3, 1, 2).contiguous()


def _render_chart_for_pair(renderer: Any, theta: Any, pair_index: int) -> Any:
    """Pair-general form of task 455's committed differentiable chart."""

    import torch

    from tac.cuda_levelset_training import contest_r
    from tac.local_acceleration import torch_levelset_inflate as tli
    from tac.local_acceleration.torch_levelset_inflate import _torch_act

    renderer.code[2 * pair_index + 1] = theta.detach()
    features_np = (
        renderer._self_orient_native(pair_index) if renderer.m["self_orient"] else renderer.curv_n
    )
    features = torch.as_tensor(features_np, dtype=torch.float32)
    model, parameters = renderer.m, renderer.P
    hidden = tli.torch_in_proj_h0(parameters, features, model)
    film = (theta @ parameters["film.weight"].T + parameters["film.bias"]).reshape(
        renderer.nH, 2, renderer.hd
    )
    activation = (
        model["activation"],
        model["wire_w0"],
        model["wire_s0"],
        model["hosc_beta"],
        model["hosc_omega"],
    )
    for layer in range(renderer.nH):
        hidden = _torch_act(
            (hidden @ parameters[f"hidden.{layer}.weight"].T + parameters[f"hidden.{layer}.bias"])
            * (1.0 + film[layer, 0])
            + film[layer, 1],
            *activation,
        )
    phi = hidden @ parameters["out_sdf.weight"].T + parameters["out_sdf.bias"]
    texture = hidden @ parameters["out_tex.weight"].T + parameters["out_tex.bias"]
    logits = phi / float(model["softmax_temp"])
    logits = logits - logits.max(-1, keepdim=True).values
    weights = torch.exp(logits)
    weights = weights / weights.sum(-1, keepdim=True)
    rgb = torch.sigmoid(weights @ parameters["palette"] + texture) * 255.0
    if not model["chroma"]:
        luma = 0.299 * rgb[:, :1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        rgb = torch.cat((luma, luma, luma), dim=-1)
    return contest_r(rgb.reshape(1, model["render_h"], model["render_w"], 3))


def _checkpoint_parity(renderer: Any, pair_index: int) -> dict[str, Any]:
    settled = _render_state_nchw(renderer, pair_index).permute(0, 2, 3, 1)
    theta = renderer.code[2 * pair_index + 1].detach().clone().requires_grad_(True)
    chart = _render_chart_for_pair(renderer, theta, pair_index)
    difference = (chart.detach() - settled).abs()
    return {
        "pair_index": pair_index,
        "max_abs": float(difference.max().item()),
        "different_elements": int((difference != 0).sum().item()),
        "chart_requires_grad": bool(chart.requires_grad),
        "status": "MEASURED_PASS" if int((difference != 0).sum().item()) == 0 else "BLOCKED",
        "settled_renderer": "tools/dash_comb_probe_n600.py::Renderer",
        "receiver": "tac.cuda_levelset_training.contest_r",
    }


def _train_record_path(output_dir: Path, pair_index: int) -> Path:
    return output_dir / "train_cache" / f"pair_{pair_index:04d}.npz"


def _heldout_record_path(output_dir: Path, pair_index: int) -> Path:
    return output_dir / "heldout" / f"pair_{pair_index:04d}.json"


def _save_train_record(
    path: Path,
    assignment: ReplayAssignment,
    record: StateSufficientStatistics,
    *,
    frame_sha256: str,
    label_sha256: str,
    margin_sha256: str,
    teacher_metrics: dict[str, float],
    teacher_elapsed_seconds: float,
) -> None:
    _atomic_npz(
        path,
        gram=record.gram,
        rhs=record.rhs,
        target_square_sum=np.asarray(record.target_square_sum, dtype=np.float64),
        row_count=np.asarray(record.row_count, dtype=np.int64),
        feature_sha256=np.asarray(record.feature_sha256),
        target_sha256=np.asarray(record.target_sha256),
        pair_index=np.asarray(assignment.pair_index, dtype=np.int64),
        checkpoint_index=np.asarray(assignment.checkpoint_index, dtype=np.int64),
        checkpoint_name=np.asarray(assignment.checkpoint_name),
        split=np.asarray(assignment.split),
        frame_sha256=np.asarray(frame_sha256),
        label_sha256=np.asarray(label_sha256),
        margin_sha256=np.asarray(margin_sha256),
        teacher_ce=np.asarray(teacher_metrics["ce"], dtype=np.float64),
        teacher_dseg=np.asarray(teacher_metrics["dseg"], dtype=np.float64),
        teacher_elapsed_seconds=np.asarray(teacher_elapsed_seconds, dtype=np.float64),
    )


def _load_train_record(path: Path, assignment: ReplayAssignment) -> StateSufficientStatistics:
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["pair_index"]) != assignment.pair_index:
            raise ProbeError(f"pair drift in {path}")
        if str(archive["checkpoint_name"]) != assignment.checkpoint_name or str(archive["split"]) != "train":
            raise ProbeError(f"assignment drift in {path}")
        record = StateSufficientStatistics(
            gram=np.asarray(archive["gram"], dtype=np.float32),
            rhs=np.asarray(archive["rhs"], dtype=np.float32),
            target_square_sum=float(archive["target_square_sum"]),
            row_count=int(archive["row_count"]),
            feature_sha256=str(archive["feature_sha256"]),
            target_sha256=str(archive["target_sha256"]),
        )
    record.validate()
    return record


def _teacher_start(
    ledger: Path, assignment: ReplayAssignment, *, stage: str, batch_id: str
) -> None:
    _append_jsonl(
        ledger,
        {
            "event": "exact_teacher_state_call_started",
            "timestamp_utc": _utc_now(),
            "stage": stage,
            "batch_id": batch_id,
            **assignment.to_dict(),
        },
    )


def _teacher_complete(
    ledger: Path,
    assignment: ReplayAssignment,
    *,
    stage: str,
    batch_id: str,
    teacher_metrics: dict[str, float],
    elapsed_seconds: float,
) -> None:
    _append_jsonl(
        ledger,
        {
            "event": "exact_teacher_state_call_completed",
            "timestamp_utc": _utc_now(),
            "stage": stage,
            "batch_id": batch_id,
            "teacher_metrics": teacher_metrics,
            "batch_elapsed_seconds": elapsed_seconds,
            **assignment.to_dict(),
        },
    )


def _batched(values: Sequence[ReplayAssignment], size: int) -> Iterable[Sequence[ReplayAssignment]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _build_training_cache(
    *,
    output_dir: Path,
    assignments: Sequence[ReplayAssignment],
    labels: np.memmap,
    margins: np.memmap,
    policy: FrozenReplayConvexHeadPolicy,
    segnet: Any,
    yopo: Any,
) -> dict[str, Any]:
    import torch

    if policy.teacher_batch_size != 1:
        raise ProbeError(
            "committed exact teacher averages CE over batch; per-state exact-label parity requires batch size 1"
        )
    completed_path = output_dir / "stage_train_cache_complete.json"
    if completed_path.is_file():
        manifest = json.loads(completed_path.read_text())
        records = [
            _load_train_record(_train_record_path(output_dir, row.pair_index), row)
            for row in assignments
            if row.split == "train"
        ]
        if len(records) != policy.train_state_count:
            raise ProbeError("completed training stage lost cached states")
        for pair, custody in manifest["records"].items():
            path = output_dir / custody["path"]
            if (
                not path.is_file()
                or path.stat().st_size != custody["bytes"]
                or _sha256(path) != custody["sha256"]
            ):
                raise ProbeError(f"completed training cache drift at pair {pair}")
        return manifest

    ledger = output_dir / "teacher_calls.jsonl"
    parity_rows: list[dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0
    batch_invocations = 0
    for checkpoint_index, (checkpoint_name, checkpoint_path, _epoch) in enumerate(CHECKPOINTS):
        cohort = [
            row
            for row in assignments
            if row.split == "train" and row.checkpoint_index == checkpoint_index
        ]
        pending = [row for row in cohort if not _train_record_path(output_dir, row.pair_index).is_file()]
        cache_hits += len(cohort) - len(pending)
        cache_misses += len(pending)
        if not pending:
            continue
        renderer, code, model, _dash = yopo._load_renderer(checkpoint_path)
        if model["n_pairs"] != policy.n_pairs or code.shape[0] != 2 * policy.n_pairs:
            raise ProbeError(f"checkpoint {checkpoint_name} is not an n600 renderer")
        parity = _checkpoint_parity(renderer, checkpoint_index)
        if parity["status"] != "MEASURED_PASS":
            raise ProbeError(f"renderer parity failed for {checkpoint_name}: {parity}")
        parity_rows.append({"checkpoint_name": checkpoint_name, **parity})
        for batch_number, batch in enumerate(_batched(pending, policy.teacher_batch_size)):
            batch_id = f"train-c{checkpoint_index}-b{batch_number}-{int(time.time_ns())}"
            frames: list[Any] = []
            features: list[np.ndarray] = []
            label_arrays: list[np.ndarray] = []
            margin_arrays: list[np.ndarray] = []
            for assignment in batch:
                frame = _render_state_nchw(renderer, assignment.pair_index)
                label = np.array(labels[assignment.pair_index], dtype=np.int64, copy=True)
                margin = np.array(margins[assignment.pair_index], dtype=np.float32, copy=True)
                frames.append(frame)
                label_arrays.append(label)
                margin_arrays.append(margin)
                features.append(
                    frozen_feature_matrix(
                        frame.detach().cpu().numpy(),
                        label,
                        margin,
                        checkpoint_index=checkpoint_index,
                        checkpoint_count=policy.checkpoint_count,
                        stride=policy.train_lattice_stride,
                    )
                )
                _teacher_start(ledger, assignment, stage="train_cache", batch_id=batch_id)
            frame_batch = torch.cat(frames, dim=0)
            label_batch = torch.as_tensor(np.stack(label_arrays), dtype=torch.long)
            exact_batch, teacher_metrics, elapsed = yopo._capture_exact_teacher_costate(
                segnet=segnet, frame_nchw=frame_batch, labels=label_batch
            )
            batch_invocations += 1
            if exact_batch.shape[0] != len(batch):
                raise ProbeError("teacher batch cardinality drift")
            for offset, assignment in enumerate(batch):
                target = sampled_costate_rows(
                    exact_batch[offset : offset + 1].detach().cpu().numpy(),
                    stride=policy.train_lattice_stride,
                )
                record = cache_exact_label_sufficient_statistics(features[offset], target)
                _teacher_complete(
                    ledger,
                    assignment,
                    stage="train_cache",
                    batch_id=batch_id,
                    teacher_metrics=teacher_metrics,
                    elapsed_seconds=elapsed,
                )
                _save_train_record(
                    _train_record_path(output_dir, assignment.pair_index),
                    assignment,
                    record,
                    frame_sha256=array_sha256(frames[offset].detach().cpu().numpy()),
                    label_sha256=array_sha256(label_arrays[offset]),
                    margin_sha256=array_sha256(margin_arrays[offset]),
                    teacher_metrics=teacher_metrics,
                    teacher_elapsed_seconds=elapsed,
                )
            _append_jsonl(
                ledger,
                {
                    "event": "exact_teacher_batch_completed",
                    "timestamp_utc": _utc_now(),
                    "stage": "train_cache",
                    "batch_id": batch_id,
                    "state_count": len(batch),
                    "elapsed_seconds": elapsed,
                },
            )

    records = [
        _load_train_record(_train_record_path(output_dir, row.pair_index), row)
        for row in assignments
        if row.split == "train"
    ]
    if len(records) != policy.train_state_count:
        raise ProbeError("training cache did not reach its sealed state count")
    manifest = {
        "schema": "frozen_replay_train_cache_stage.v1",
        "completed_at_utc": _utc_now(),
        "state_count": len(records),
        "row_count": int(sum(row.row_count for row in records)),
        "feature_count": len(FEATURE_NAMES),
        "cache_hits_this_invocation": cache_hits,
        "cache_misses_this_invocation": cache_misses,
        "teacher_batch_invocations_this_invocation": batch_invocations,
        "renderer_parity": parity_rows,
        "records": {
            str(row.pair_index): {
                "path": str(_train_record_path(output_dir, row.pair_index).relative_to(output_dir)),
                "bytes": _train_record_path(output_dir, row.pair_index).stat().st_size,
                "sha256": _sha256(_train_record_path(output_dir, row.pair_index)),
            }
            for row in assignments
            if row.split == "train"
        },
    }
    _atomic_json(completed_path, manifest)
    return manifest


def _fit_stage(
    *, output_dir: Path, assignments: Sequence[ReplayAssignment], policy: FrozenReplayConvexHeadPolicy
) -> tuple[Any, dict[str, Any]]:
    completed_path = output_dir / "stage_fit_complete.json"
    if completed_path.is_file():
        manifest = json.loads(completed_path.read_text())
        weights_path = output_dir / manifest["weights"]["path"]
        if (
            not weights_path.is_file()
            or weights_path.stat().st_size != manifest["weights"]["bytes"]
            or _sha256(weights_path) != manifest["weights"]["sha256"]
        ):
            raise ProbeError("completed convex-head weights drifted")
        with np.load(weights_path, allow_pickle=False) as archive:
            weights = np.asarray(archive["weights"], dtype=np.float32)
            if tuple(str(value) for value in archive["feature_names"]) != FEATURE_NAMES:
                raise ProbeError("completed convex-head feature chart drifted")
        if array_sha256(weights) != manifest["weights"]["array_sha256"]:
            raise ProbeError("completed convex-head weight array drifted")
        return types.SimpleNamespace(weights=weights), manifest

    records = [
        _load_train_record(_train_record_path(output_dir, row.pair_index), row)
        for row in assignments
        if row.split == "train"
    ]
    aggregate = aggregate_sufficient_statistics(records)
    fit = fit_cached_convex_head(aggregate, epochs=policy.fit_epochs)
    weights_path = output_dir / "fit" / "convex_head_weights.npz"
    recovered_pre_manifest_weights = weights_path.is_file()
    if recovered_pre_manifest_weights:
        with np.load(weights_path, allow_pickle=False) as archive:
            if not np.array_equal(archive["weights"], fit.weights):
                raise ProbeError("pre-manifest convex-head weights disagree with re-derivation")
            if not np.array_equal(archive["optimum_weights"], fit.optimum_weights):
                raise ProbeError("pre-manifest optimum weights disagree with re-derivation")
            if tuple(str(value) for value in archive["feature_names"]) != FEATURE_NAMES:
                raise ProbeError("pre-manifest convex-head feature chart drifted")
            if str(archive["hessian_sha256"].item()) != fit.certificate.hessian_sha256:
                raise ProbeError("pre-manifest convex-head Hessian custody drifted")
    else:
        _atomic_npz(
            weights_path,
            weights=fit.weights,
            optimum_weights=fit.optimum_weights,
            feature_names=np.asarray(FEATURE_NAMES),
            hessian_sha256=np.asarray(fit.certificate.hessian_sha256),
        )
    observed_ratios = [
        float(row["parameter_contraction_ratio"])
        for row in fit.trace
        if row["parameter_contraction_ratio"] is not None
    ]
    if not observed_ratios and (
        fit.initial_parameter_error_norm > fit.parameter_ratio_numeric_floor
    ):
        raise ProbeError("executed fp32 contraction trace is missing above the scale-aware floor")
    parameter_ratio_max = max(observed_ratios) if observed_ratios else None
    ratio_slack = 64.0 * np.finfo(np.float64).eps
    if parameter_ratio_max is not None and (
        parameter_ratio_max > fit.certificate.contraction_gamma + ratio_slack
    ):
        raise ProbeError("executed fp32 iterate violated the derived contraction constant")
    observed_objective_ratios = [
        float(row["objective_gap_ratio"])
        for row in fit.trace
        if row["objective_gap_ratio"] is not None
    ]
    if not observed_objective_ratios and (
        fit.initial_objective_gap > fit.objective_ratio_numeric_floor
    ):
        raise ProbeError("executed fp32 objective trace is missing above the scale-aware floor")
    objective_ratio_max = max(observed_objective_ratios) if observed_objective_ratios else None
    if objective_ratio_max is not None and (
        objective_ratio_max > fit.certificate.contraction_gamma**2 + ratio_slack
    ):
        raise ProbeError("executed fp32 objective gap violated gamma squared")
    manifest = {
        "schema": "frozen_replay_convex_head_fit_stage.v1",
        "completed_at_utc": _utc_now(),
        "state_count": aggregate.state_count,
        "sampled_pixel_rows": aggregate.row_count,
        "epochs": policy.fit_epochs,
        "optimizer_steps": policy.fit_epochs,
        "effective_training_state_steps": policy.effective_training_state_steps,
        "certificate": fit.certificate.to_dict(),
        "initial_parameter_error_norm": fit.initial_parameter_error_norm,
        "parameter_ratio_numeric_floor": fit.parameter_ratio_numeric_floor,
        "max_observed_parameter_contraction_ratio_above_numeric_floor": parameter_ratio_max,
        "initial_objective_gap": fit.initial_objective_gap,
        "objective_ratio_numeric_floor": fit.objective_ratio_numeric_floor,
        "max_observed_objective_gap_ratio_above_numeric_floor": objective_ratio_max,
        "trace": list(fit.trace),
        "terminal_gradient_norm": fit.terminal_gradient_norm,
        "actual_parameter_residual": fit.actual_parameter_residual,
        "residual_parameter_bound": fit.residual_parameter_bound,
        "actual_prediction_rmse_residual": fit.actual_prediction_rmse_residual,
        "residual_prediction_rmse_bound": fit.residual_prediction_rmse_bound,
        "actual_objective_gap": fit.actual_objective_gap,
        "objective_gap_bound": fit.objective_gap_bound,
        "residual_bounds_validated": fit.residual_bounds_validated,
        "per_state_gradient_variance": fit.per_state_gradient_variance,
        "per_state_gradient_second_moment": fit.per_state_gradient_second_moment,
        "recovered_pre_manifest_weights_without_rewrite": recovered_pre_manifest_weights,
        "weights": {
            "path": str(weights_path.relative_to(output_dir)),
            "bytes": weights_path.stat().st_size,
            "sha256": _sha256(weights_path),
            "array_sha256": array_sha256(fit.weights),
        },
    }
    _atomic_json(completed_path, manifest)
    return fit, manifest


def _heldout_state(
    *,
    output_dir: Path,
    assignment: ReplayAssignment,
    labels: np.memmap,
    margins: np.memmap,
    policy: FrozenReplayConvexHeadPolicy,
    weights: np.ndarray,
    renderer: Any,
    segnet: Any,
    yopo: Any,
    matched: Any,
) -> dict[str, Any]:
    import torch

    ledger = output_dir / "teacher_calls.jsonl"
    batch_id = f"heldout-p{assignment.pair_index:04d}-{int(time.time_ns())}"
    label = np.array(labels[assignment.pair_index], dtype=np.int64, copy=True)
    margin = np.array(margins[assignment.pair_index], dtype=np.float32, copy=True)
    theta_exact = renderer.code[2 * assignment.pair_index + 1].detach().clone().requires_grad_(True)
    frame_nhwc = _render_chart_for_pair(renderer, theta_exact, assignment.pair_index)
    frame_nchw = frame_nhwc.permute(0, 3, 1, 2).contiguous()
    settled_frame_nchw = _render_state_nchw(renderer, assignment.pair_index)
    settled_difference = (frame_nchw.detach() - settled_frame_nchw).abs()
    settled_parity = {
        "equal": bool(torch.equal(frame_nchw.detach(), settled_frame_nchw)),
        "max_abs": float(settled_difference.max().item()),
        "different_elements": int((settled_difference != 0).sum().item()),
        "settled_renderer": "tools/dash_comb_probe_n600.py::Renderer",
        "receiver": "tac.cuda_levelset_training.contest_r",
    }
    if not settled_parity["equal"]:
        raise ProbeError(
            f"held-out chart/settled renderer drift at pair {assignment.pair_index}"
        )
    labels_t = torch.as_tensor(label[None], dtype=torch.long)
    _teacher_start(ledger, assignment, stage="heldout_validation", batch_id=batch_id)
    exact_costate, teacher_metrics, elapsed = yopo._capture_exact_teacher_costate(
        segnet=segnet, frame_nchw=frame_nchw, labels=labels_t
    )
    features = frozen_feature_matrix(
        frame_nchw.detach().cpu().numpy(),
        label,
        margin,
        checkpoint_index=assignment.checkpoint_index,
        checkpoint_count=policy.checkpoint_count,
        stride=1,
    )
    predicted_costate = predict_costate(
        features, weights, height=label.shape[0], width=label.shape[1]
    )
    predicted_t = torch.as_tensor(predicted_costate, dtype=torch.float32)
    costate_fidelity = vector_fidelity(
        exact_costate.detach().cpu().numpy(), predicted_costate
    )
    exact_gradient, exact_vjp_seconds = matched._renderer_gradient(
        frame_nhwc, theta_exact, exact_costate
    )
    theta_predicted = (
        renderer.code[2 * assignment.pair_index + 1].detach().clone().requires_grad_(True)
    )
    repeated_frame = _render_chart_for_pair(renderer, theta_predicted, assignment.pair_index)
    predicted_gradient, predicted_vjp_seconds = matched._renderer_gradient(
        repeated_frame, theta_predicted, predicted_t
    )
    repeat_equal = bool(torch.equal(frame_nhwc, repeated_frame))
    renderer_fidelity = vector_fidelity(
        exact_gradient.detach().cpu().numpy(), predicted_gradient.detach().cpu().numpy()
    )
    _teacher_complete(
        ledger,
        assignment,
        stage="heldout_validation",
        batch_id=batch_id,
        teacher_metrics=teacher_metrics,
        elapsed_seconds=elapsed,
    )
    if not repeat_equal:
        raise ProbeError(f"held-out render repeat drift at pair {assignment.pair_index}")
    _append_jsonl(
        ledger,
        {
            "event": "exact_teacher_batch_completed",
            "timestamp_utc": _utc_now(),
            "stage": "heldout_validation",
            "batch_id": batch_id,
            "state_count": 1,
            "elapsed_seconds": elapsed,
        },
    )
    result = {
        "schema": "frozen_replay_convex_head_heldout_state.v1",
        "completed_at_utc": _utc_now(),
        "assignment": assignment.to_dict(),
        "frame_sha256": array_sha256(frame_nchw.detach().cpu().numpy()),
        "label_sha256": array_sha256(label),
        "margin_sha256": array_sha256(margin),
        "exact_costate_sha256": array_sha256(exact_costate.detach().cpu().numpy()),
        "predicted_costate_sha256": array_sha256(predicted_costate),
        "teacher_metrics": teacher_metrics,
        "teacher_elapsed_seconds": elapsed,
        "costate_fidelity": costate_fidelity,
        "renderer_gradient_fidelity": renderer_fidelity,
        "exact_renderer_gradient": exact_gradient.detach().cpu().numpy().astype(float).tolist(),
        "predicted_renderer_gradient": predicted_gradient.detach().cpu().numpy().astype(float).tolist(),
        "exact_renderer_vjp_seconds": exact_vjp_seconds,
        "predicted_renderer_vjp_seconds": predicted_vjp_seconds,
        "deterministic_render_repeat_equal": repeat_equal,
        "settled_renderer_parity": settled_parity,
        "authority": AXIS,
    }
    _atomic_json(_heldout_record_path(output_dir, assignment.pair_index), result)
    return result


def _aggregate_fidelity(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    metrics = [row[key] for row in rows]
    dot = sum(float(row["dot"]) for row in metrics)
    reference_square = sum(float(row["reference_norm"]) ** 2 for row in metrics)
    candidate_square = sum(float(row["candidate_norm"]) ** 2 for row in metrics)
    delta_square = sum(
        (float(row["relative_l2_error"]) * float(row["reference_norm"])) ** 2
        for row in metrics
    )
    cosine = dot / np.sqrt(reference_square * candidate_square) if reference_square and candidate_square else None
    return {
        "state_count": len(rows),
        "compared_elements": sum(int(row["compared_elements"]) for row in metrics),
        "dot": dot,
        "cosine_similarity": cosine,
        "relative_l2_error": np.sqrt(delta_square / reference_square) if reference_square else None,
        "reference_norm": np.sqrt(reference_square),
        "candidate_norm": np.sqrt(candidate_square),
        "mean_per_state_cosine": float(
            np.mean([float(row["cosine_similarity"]) for row in metrics])
        ),
        "positive_dot_state_fraction": float(np.mean([float(row["dot"]) > 0.0 for row in metrics])),
        "reduction_dtype": "float64",
    }


def _heldout_stage(
    *,
    output_dir: Path,
    assignments: Sequence[ReplayAssignment],
    labels: np.memmap,
    margins: np.memmap,
    policy: FrozenReplayConvexHeadPolicy,
    weights: np.ndarray,
    segnet: Any,
    yopo: Any,
    matched: Any,
) -> dict[str, Any]:
    completed_path = output_dir / "stage_heldout_complete.json"
    if completed_path.is_file():
        manifest = json.loads(completed_path.read_text())
        if manifest["state_count"] != policy.heldout_state_count:
            raise ProbeError("completed held-out stage cardinality drifted")
        for pair, custody in manifest["records"].items():
            path = output_dir / custody["path"]
            if (
                not path.is_file()
                or path.stat().st_size != custody["bytes"]
                or _sha256(path) != custody["sha256"]
            ):
                raise ProbeError(f"completed held-out record drift at pair {pair}")
        return manifest

    cache_hits = 0
    cache_misses = 0
    for checkpoint_index, (checkpoint_name, checkpoint_path, _epoch) in enumerate(CHECKPOINTS):
        cohort = [
            row
            for row in assignments
            if row.split == "heldout" and row.checkpoint_index == checkpoint_index
        ]
        pending = [row for row in cohort if not _heldout_record_path(output_dir, row.pair_index).is_file()]
        cache_hits += len(cohort) - len(pending)
        cache_misses += len(pending)
        if not pending:
            continue
        renderer, code, model, _dash = yopo._load_renderer(checkpoint_path)
        if model["n_pairs"] != policy.n_pairs or code.shape[0] != 2 * policy.n_pairs:
            raise ProbeError(f"checkpoint {checkpoint_name} is not an n600 renderer")
        for assignment in pending:
            row = _heldout_state(
                output_dir=output_dir,
                assignment=assignment,
                labels=labels,
                margins=margins,
                policy=policy,
                weights=weights,
                renderer=renderer,
                segnet=segnet,
                yopo=yopo,
                matched=matched,
            )
            if not row["deterministic_render_repeat_equal"]:
                raise ProbeError(f"held-out render repeat drift at pair {assignment.pair_index}")
            if not row["settled_renderer_parity"]["equal"]:
                raise ProbeError(
                    f"held-out chart/settled renderer drift at pair {assignment.pair_index}"
                )

    rows = [
        json.loads(_heldout_record_path(output_dir, row.pair_index).read_text())
        for row in assignments
        if row.split == "heldout"
    ]
    if len(rows) != policy.heldout_state_count:
        raise ProbeError("held-out stage did not reach its sealed state count")
    for row in rows:
        pair_index = int(row["assignment"]["pair_index"])
        if not row.get("deterministic_render_repeat_equal"):
            raise ProbeError(f"cached held-out render repeat drift at pair {pair_index}")
        if not row.get("settled_renderer_parity", {}).get("equal"):
            raise ProbeError(f"cached held-out settled parity drift at pair {pair_index}")
    manifest = {
        "schema": "frozen_replay_convex_head_heldout_stage.v1",
        "completed_at_utc": _utc_now(),
        "state_count": len(rows),
        "cache_hits_this_invocation": cache_hits,
        "cache_misses_this_invocation": cache_misses,
        "costate_fidelity": _aggregate_fidelity(rows, "costate_fidelity"),
        "renderer_gradient_fidelity": _aggregate_fidelity(rows, "renderer_gradient_fidelity"),
        "settled_renderer_parity": {
            "states_checked": len(rows),
            "all_equal": all(row["settled_renderer_parity"]["equal"] for row in rows),
            "max_abs": max(row["settled_renderer_parity"]["max_abs"] for row in rows),
            "different_elements": sum(
                row["settled_renderer_parity"]["different_elements"] for row in rows
            ),
        },
        "records": {
            str(row["assignment"]["pair_index"]): {
                "path": str(
                    _heldout_record_path(output_dir, row["assignment"]["pair_index"]).relative_to(
                        output_dir
                    )
                ),
                "bytes": _heldout_record_path(
                    output_dir, row["assignment"]["pair_index"]
                ).stat().st_size,
                "sha256": _sha256(
                    _heldout_record_path(output_dir, row["assignment"]["pair_index"])
                ),
            }
            for row in rows
        },
    }
    _atomic_json(completed_path, manifest)
    return manifest


def _teacher_accounting(
    output_dir: Path,
    policy: FrozenReplayConvexHeadPolicy,
    assignments: Sequence[ReplayAssignment],
) -> dict[str, Any]:
    ledger_path = output_dir / "teacher_calls.jsonl"
    rows = _canonicalize_event_ledger(ledger_path)
    event_dir = ledger_path.with_name(f"{ledger_path.name}.events")
    event_files = sorted(event_dir.glob("*.json"))
    event_manifest = [
        {
            "path": str(path.relative_to(output_dir)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in event_files
    ]
    event_tree_sha256 = hashlib.sha256(
        json.dumps(event_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    starts = [row for row in rows if row["event"] == "exact_teacher_state_call_started"]
    completions = [row for row in rows if row["event"] == "exact_teacher_state_call_completed"]
    batches = [row for row in rows if row["event"] == "exact_teacher_batch_completed"]
    attempted_batch_ids = {row["batch_id"] for row in starts}
    required_states = {(row.split, row.pair_index) for row in assignments}
    attempted_states = {(row["split"], int(row["pair_index"])) for row in starts}
    completed_states = {(row["split"], int(row["pair_index"])) for row in completions}
    if attempted_states != required_states:
        missing = sorted(required_states - attempted_states)
        unexpected = sorted(attempted_states - required_states)
        raise ProbeError(
            f"teacher start-ledger state coverage drift: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    if completed_states != required_states:
        missing = sorted(required_states - completed_states)
        unexpected = sorted(completed_states - required_states)
        raise ProbeError(
            f"teacher completion-ledger state coverage drift: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    completed_keys = {
        (row["split"], int(row["pair_index"]), row["batch_id"]) for row in completions
    }
    pending_attempts = sum(
        (row["split"], int(row["pair_index"]), row["batch_id"]) not in completed_keys
        for row in starts
    )
    accounting = teacher_call_accounting(
        naive_teacher_calls=policy.effective_training_state_steps,
        fresh_anchor_samples=len(starts),
        paired_difference_samples=policy.effective_training_state_steps,
        exact_labels_per_difference=0,
        observed_teacher_forwards=len(starts),
    )
    accounting.update(
        {
            "teacher_call_unit": "one exact labeled state evaluation; batching does not alter this count",
            "effective_training_step_unit": "one cached labeled-state use in one fit epoch",
            "optimizer_steps_reported_separately": policy.fit_epochs,
            "train_cache_state_calls_started": sum(row["stage"] == "train_cache" for row in starts),
            "heldout_validation_state_calls_started": sum(
                row["stage"] == "heldout_validation" for row in starts
            ),
            "model_forward_backward_batch_invocations_completed": len(batches),
            "model_forward_backward_batch_invocations_conservative_upper_bound": len(
                attempted_batch_ids
            ),
            "completed_state_calls": len(completions),
            "required_unique_state_calls": len(required_states),
            "attempted_unique_state_calls": len(attempted_states),
            "completed_unique_state_calls": len(completed_states),
            "completed_unique_train_state_calls": sum(
                split == "train" for split, _pair_index in completed_states
            ),
            "completed_unique_heldout_state_calls": sum(
                split == "heldout" for split, _pair_index in completed_states
            ),
            "sealed_assignment_coverage": "PASS",
            "pending_or_crashed_attempts_conservatively_counted": pending_attempts,
            "resume_restore_teacher_calls": 0,
            "cached_same_state_gradient_difference_label_calls": 0,
            "label_cancellation_identity": "g_s(W)-g_s(V)=X_s'X_s(W-V)",
            "all_cache_build_and_validation_calls_included": True,
            "teacher_call_ledger": {
                "path": str(ledger_path.relative_to(output_dir)),
                "bytes": ledger_path.stat().st_size,
                "sha256": _sha256(ledger_path),
                "rows": len(rows),
                "atomic_event_directory": str(event_dir.relative_to(output_dir)),
                "atomic_event_file_count": len(event_files),
                "atomic_event_tree_sha256": event_tree_sha256,
            },
        }
    )
    return accounting


def run(
    *,
    output_dir: Path,
    resume: bool,
    teacher_batch_size: int,
    source_amendment: str | None = None,
) -> dict[str, Any]:
    completed = _completed_receipt_or_none(output_dir, resume=resume)
    if completed is not None:
        return completed

    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(455)
    torch.use_deterministic_algorithms(True)
    policy = FrozenReplayConvexHeadPolicy(teacher_batch_size=teacher_batch_size)
    contract = policy.compile_measurement_contract()
    checkpoint_names = tuple(row[0] for row in CHECKPOINTS)
    assignments = deterministic_replay_assignments(
        n_pairs=policy.n_pairs,
        checkpoint_names=checkpoint_names,
        holdout_period=policy.holdout_period,
        seed=policy.seed,
    )
    if sum(row.split == "train" for row in assignments) != policy.train_state_count:
        raise ProbeError("compiled train split disagrees with assignment")
    if sum(row.split == "heldout" for row in assignments) != policy.heldout_state_count:
        raise ProbeError("compiled held-out split disagrees with assignment")

    descriptor = _acquire_lock(output_dir)
    try:
        complete_path = output_dir / "complete.json"
        completed = _completed_receipt_or_none(output_dir, resume=resume)
        if completed is not None:
            return completed

        _write_cleanup_manifest(output_dir, phase="ARMED")
        config_path = output_dir / "run_contract.json"
        prior_contract: dict[str, Any] | None = None
        if config_path.exists():
            if not resume:
                raise ProbeError(f"{output_dir} already contains a run; pass --resume")
            prior_contract = json.loads(config_path.read_text())
            if prior_contract["compiled_policy"] != contract:
                raise ProbeError("resume policy drift")
        elif resume:
            raise ProbeError("--resume requested without a run_contract.json")

        input_custody = _verify_input_custody()
        runtime_custody = _runtime_custody(torch)
        storage_custody = _storage_preflight_custody(output_dir)
        if prior_contract is None:
            if source_amendment is not None:
                raise ProbeError("a source amendment is valid only for an existing resumable run")
            source_custody = _source_bundle(output_dir)
            source_amendment_custody = None
        else:
            source_custody, source_amendment_custody = _resolve_source_custody(
                output_dir,
                prior_contract=prior_contract,
                requested_amendment=source_amendment,
                expected_train_pairs={
                    row.pair_index for row in assignments if row.split == "train"
                },
            )
        run_contract = {
            "schema": "frozen_replay_convex_head_run_contract.v1",
            "created_or_verified_at_utc": _utc_now(),
            "lane_id": LANE_ID,
            "compiled_policy": contract,
            "inputs": input_custody,
            "sources": source_custody,
            "runtime": runtime_custody,
            "storage_preflight": storage_custody,
            "git_head_at_measurement": _git_head(),
            "git_status_at_measurement_start": _git_status(),
            "authority": AXIS,
            "source_runs_read_only": True,
            "live_run_touched": False,
            "paid_dispatch": False,
        }
        if prior_contract is not None:
            for key in ("compiled_policy", "inputs", "storage_preflight"):
                if prior_contract[key] != run_contract[key]:
                    raise ProbeError(f"resume custody drift in {key}")
            if (
                prior_contract["runtime"]["sealed_numerical_identity"]
                != run_contract["runtime"]["sealed_numerical_identity"]
            ):
                raise ProbeError("resume numerical-runtime identity drift")
            sealed_run_contract = prior_contract
        else:
            _atomic_json(config_path, run_contract)
            sealed_run_contract = run_contract

        initial_contract_custody = {
            "path": str(config_path.relative_to(output_dir)),
            "bytes": config_path.stat().st_size,
            "sha256": _sha256(config_path),
        }
        invocation_id = f"{time.time_ns()}-{os.getpid()}"
        invocation_ledger_path = output_dir / "invocations.jsonl"
        _append_jsonl(
            invocation_ledger_path,
            {
                "event": "probe_invocation_started",
                "invocation_id": invocation_id,
                "timestamp_utc": _utc_now(),
                "resume": resume,
                "runtime": runtime_custody,
                "git_head": run_contract["git_head_at_measurement"],
                "git_status": run_contract["git_status_at_measurement_start"],
                "initial_run_contract_sha256": initial_contract_custody["sha256"],
                "effective_source_custody": source_custody,
                "source_amendment_custody": source_amendment_custody,
            },
        )

        labels = _stored_npy_memmap(GT_CACHE, "lstars.npy")
        margins = _stored_npy_memmap(GT_CACHE, "margins.npy")
        if labels.shape != (600, 384, 512) or margins.shape != labels.shape:
            raise ProbeError("GT cache geometry drift")
        yopo = _load_tool_module(
            "_round2_committed_yopo", "tools/probe_yopo_first_layer_costate.py"
        )
        matched = _load_tool_module(
            "_round2_committed_matched", "tools/probe_onpolicy_costate_matched_window.py"
        )
        segnet = _load_cpu_segnet()

        train_stage = _build_training_cache(
            output_dir=output_dir,
            assignments=assignments,
            labels=labels,
            margins=margins,
            policy=policy,
            segnet=segnet,
            yopo=yopo,
        )
        fit, fit_stage = _fit_stage(
            output_dir=output_dir, assignments=assignments, policy=policy
        )
        heldout_stage = _heldout_stage(
            output_dir=output_dir,
            assignments=assignments,
            labels=labels,
            margins=margins,
            policy=policy,
            weights=fit.weights,
            segnet=segnet,
            yopo=yopo,
            matched=matched,
        )
        accounting = _teacher_accounting(output_dir, policy, assignments)
        cosine = heldout_stage["costate_fidelity"]["cosine_similarity"]
        if cosine is None:
            raise ProbeError("held-out costate cosine is undefined")
        verdict = derive_mission_verdict(
            heldout_costate_cosine=float(cosine),
            teacher_call_amortization_x=float(accounting["teacher_call_amortization_x"]),
        )
        _append_jsonl(
            invocation_ledger_path,
            {
                "event": "probe_invocation_completed",
                "invocation_id": invocation_id,
                "timestamp_utc": _utc_now(),
                "resume": resume,
                "verdict": verdict["verdict"],
                "teacher_call_ledger_sha256": accounting["teacher_call_ledger"]["sha256"],
                "stage_checkpoints": {
                    "train": _sha256(output_dir / "stage_train_cache_complete.json"),
                    "fit": _sha256(output_dir / "stage_fit_complete.json"),
                    "heldout": _sha256(output_dir / "stage_heldout_complete.json"),
                },
            },
        )
        invocation_rows = _canonicalize_event_ledger(invocation_ledger_path)
        invocation_event_dir = invocation_ledger_path.with_name(
            f"{invocation_ledger_path.name}.events"
        )
        invocation_event_manifest = [
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(invocation_event_dir.glob("*.json"))
        ]
        invocation_custody = {
            "path": str(invocation_ledger_path.relative_to(output_dir)),
            "bytes": invocation_ledger_path.stat().st_size,
            "sha256": _sha256(invocation_ledger_path),
            "rows": len(invocation_rows),
            "atomic_event_directory": str(invocation_event_dir.relative_to(output_dir)),
            "atomic_event_file_count": len(invocation_event_manifest),
            "atomic_event_tree_sha256": hashlib.sha256(
                json.dumps(
                    invocation_event_manifest, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
        }
        cleanup_manifest = _write_cleanup_manifest(output_dir, phase="COMPLETE_NO_BULK")
        cleanup_path = output_dir / "cleanup_manifest.json"
        cleanup_custody = {
            "path": str(cleanup_path.relative_to(output_dir)),
            "bytes": cleanup_path.stat().st_size,
            "sha256": _sha256(cleanup_path),
            "phase": cleanup_manifest["phase"],
            "removed_or_recovered_scratch_count": len(cleanup_manifest["removed_scratch"]),
            "preserved_scratch_blockers": cleanup_manifest["preserved_scratch_blockers"],
        }
        receipt = {
            "schema": SCHEMA,
            "completed_at_utc": _utc_now(),
            "verdict": verdict,
            "authority": {
                "axis": AXIS,
                "module_scope": AUTHORITY_SCOPE,
                "n600_real_states": True,
                "numpy_fp32_head_authority": True,
                "MPS_used": False,
                "score_claim": False,
                "promotion_eligible": False,
                "pointer_moved": False,
                "research_only": RESEARCH_ONLY,
            },
            "run_contract": sealed_run_contract,
            "initial_run_contract_custody": initial_contract_custody,
            "effective_source_custody": source_custody,
            "source_amendment_custody": source_amendment_custody,
            "invocation_custody": invocation_custody,
            "cleanup_custody": cleanup_custody,
            "replay": {
                "state_count": len(assignments),
                "unique_pair_count": len({row.pair_index for row in assignments}),
                "train_state_count": policy.train_state_count,
                "heldout_state_count": policy.heldout_state_count,
                "checkpoint_distribution": {
                    name: {
                        "epoch": epoch,
                        "train": sum(
                            row.checkpoint_name == name and row.split == "train"
                            for row in assignments
                        ),
                        "heldout": sum(
                            row.checkpoint_name == name and row.split == "heldout"
                            for row in assignments
                        ),
                    }
                    for name, _path, epoch in CHECKPOINTS
                },
                "fixed_distribution": True,
                "on_policy": False,
                "state_assignment_sha256": hashlib.sha256(
                    json.dumps(
                        [row.to_dict() for row in assignments],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            },
            "train_cache_stage": train_stage,
            "fit_stage": fit_stage,
            "heldout_stage": heldout_stage,
            "teacher_call_accounting": accounting,
            "triality": {
                "equation": "tac.canonical_equations.frozen_replay_convex_head_contraction_20260713",
                "dsl": "tac.witness_dsl.frozen_replay_convex_head_policy",
                "dag_feed": ".omx/research/frozen_replay_convex_head_contraction_DAG_FEED_20260713.md",
            },
            "reformulation_queue": (
                []
                if verdict["verdict"] == "GO"
                else [
                    {
                        "priority": 1,
                        "formulation": "frozen SegNet-stem features plus the same convex linear head",
                        "preserved_invariants": ["fixed replay", "cached exact labels", "convex head"],
                    },
                    {
                        "priority": 2,
                        "formulation": "fixed multiscale random/Fourier feature lift plus convex head",
                        "preserved_invariants": ["fixed operator", "explicit Hessian", "c_label=0"],
                    },
                    {
                        "priority": 3,
                        "formulation": "class-block convex heads with held-out renderer-VJP calibration",
                        "preserved_invariants": ["no nonlinear learner gap", "teacher-call custody"],
                    },
                ]
            ),
            "pointer_delta": "NONE; research-only local formulation evidence",
        }
        receipt_path = output_dir / "measurement_receipt.json"
        _atomic_json(receipt_path, receipt)
        _atomic_json(
            complete_path,
            {
                "schema": "frozen_replay_convex_head_completion.v1",
                "receipt": str(receipt_path.relative_to(output_dir)),
                "bytes": receipt_path.stat().st_size,
                "sha256": _sha256(receipt_path),
                "verdict": verdict["verdict"],
                "completed_at_utc": _utc_now(),
            },
        )
        return receipt
    finally:
        _release_lock(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--teacher-batch-size", type=int, default=1)
    parser.add_argument(
        "--source-amendment",
        choices=(SOURCE_AMENDMENT_ID,),
        help="Explicit append-only verifier-repair receipt used after a sealed stage boundary.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if REPO not in output_dir.parents:
        raise ProbeError("operator-facing evidence must be durable under the repository")
    receipt = run(
        output_dir=output_dir,
        resume=args.resume,
        teacher_batch_size=args.teacher_batch_size,
        source_amendment=args.source_amendment,
    )
    summary = {
        "verdict": receipt["verdict"]["verdict"],
        "heldout_costate_cosine": receipt["heldout_stage"]["costate_fidelity"][
            "cosine_similarity"
        ],
        "renderer_gradient_dot": receipt["heldout_stage"]["renderer_gradient_fidelity"]["dot"],
        "teacher_call_amortization_x": receipt["teacher_call_accounting"][
            "teacher_call_amortization_x"
        ],
        "contraction_gamma": receipt["fit_stage"]["certificate"]["contraction_gamma"],
        "receipt": str(args.output_dir / "measurement_receipt.json"),
    }
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
