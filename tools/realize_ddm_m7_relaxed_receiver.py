#!/usr/bin/env python3
"""Realize and advisory-score the frozen DDM M7 receiver member.

This tool is deliberately local-only and non-promotable.  It parses the exact
counted archive through ``tac.click_polish.FrozenPacket``, renders its native
integer latent table through ``tac.click_polish.Renderer``, and scores all 600
pairs with the frozen upstream CPU models in evaluator order.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
_DYNAMIC_TARGET_REPO_ROOT = REPO_ROOT
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.canonical_equations.ddm_m7_realization_transfer_20260723 import (  # noqa: E402
    DDM_M7_REALIZATION_TRANSFER_EQUATION_ID,
    ContestScoreTerms,
    ScoreGapDecomposition,
    contest_score_terms,
    realization_transfer_ratios,
    require_score_gap_closure,
    score_gap_decomposition,
)
from tac.canonical_frontier_pointer import CANONICAL_FRONTIER_POINTER_PATH  # noqa: E402
from tac.witness_dsl.dynamic_frontier_target import (  # noqa: E402
    DynamicFrontierTargetError,
    DynamicFrontierTargetSnapshot,
    load_dynamic_frontier_target,
    verify_dynamic_frontier_target_snapshot,
)

CONFIG_SCHEMA = "ddm_m7_relaxed_receiver_realize_config.v2"
CHECKPOINT_SCHEMA = "ddm_m7_relaxed_receiver_batch_checkpoint.v1"
RECEIPT_SCHEMA = "ddm_m7_relaxed_receiver_realization_receipt.v2"
REROUTE_AUDIT_SCHEMA = "ddm_m7_dynamic_frontier_reroute_audit.v1"
IDENTITY_SCHEMA = "ddm_m7_relaxed_receiver_run_identity.v1"
EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"
ROUTING_CANDIDATE = "BYTE-CLOSED_CANDIDATE_FOR_MODAL_EXACT_EVAL"
ROUTING_NOT_CANDIDATE = "REALIZED_SCORE_DID_NOT_CLEAR_STRICT_FORK"
ALLOWED_SSD_ROOTS = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)
MIN_OUTPUT_FREE_BYTES = 8 * 1024 * 1024
SHA256_LENGTH = 64

EXPECTED_CONSTANTS: dict[str, Any] = {
    "schema": CONFIG_SCHEMA,
    "expected_archive_bytes": 177169,
    "expected_archive_sha256": (
        "cb6cf0ba719a535bf8874b31675a4ec66a893423d320f1e4071a2012cd88a56f"
    ),
    "expected_member_name": "x",
    "n_pairs": 600,
    "batch_pairs": 16,
    "num_threads": 2,
    "seed": 1234,
    "reference_bytes": 37545489,
    "counterfactual_d_seg": 0.00015196,
    "counterfactual_d_pose": 0.00010184,
    "device": "cpu",
    "evidence_axis": EVIDENCE_AXIS,
    "score_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}
PATH_FIELDS = (
    "candidate_archive",
    "runtime_dir",
    "upstream_dir",
    "ssd_output_dir",
)
CONFIG_FIELDS = frozenset((*PATH_FIELDS, *EXPECTED_CONSTANTS))


class RealizationRefusal(RuntimeError):
    """Fail-closed refusal for invalid custody, resume, or measurement state."""


def _expected_dynamic_pointer_path() -> str:
    return os.path.abspath(
        os.fspath(_DYNAMIC_TARGET_REPO_ROOT / CANONICAL_FRONTIER_POINTER_PATH)
    )


def _require_dynamic_frontier_snapshot(
    snapshot: DynamicFrontierTargetSnapshot | None,
    *,
    now_utc_iso: str | None,
) -> DynamicFrontierTargetSnapshot:
    try:
        if snapshot is None:
            snapshot = load_dynamic_frontier_target(
                repo_root=_DYNAMIC_TARGET_REPO_ROOT,
                now_utc_iso=now_utc_iso,
            )
        if not isinstance(snapshot, DynamicFrontierTargetSnapshot):
            raise TypeError("frontier_snapshot must be a DynamicFrontierTargetSnapshot")
        if snapshot.pointer_path != _expected_dynamic_pointer_path():
            raise RealizationRefusal(
                "realization routing refuses a snapshot from a noncanonical pointer path"
            )
        return verify_dynamic_frontier_target_snapshot(
            snapshot,
            now_utc_iso=now_utc_iso,
        )
    except (DynamicFrontierTargetError, TypeError) as exc:
        raise RealizationRefusal(
            f"dynamic frontier target is not current and canonical: {exc}"
        ) from exc


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != SHA256_LENGTH:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _is_git_oid(value: Any) -> bool:
    if not isinstance(value, str) or len(value) not in (40, 64):
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


@dataclass(frozen=True)
class RealizeConfig:
    schema: str
    candidate_archive: str
    runtime_dir: str
    upstream_dir: str
    ssd_output_dir: str
    expected_archive_bytes: int
    expected_archive_sha256: str
    expected_member_name: str
    n_pairs: int
    batch_pairs: int
    num_threads: int
    seed: int
    reference_bytes: int
    counterfactual_d_seg: float
    counterfactual_d_pose: float
    device: str
    evidence_axis: str
    score_claim: bool
    promotion_eligible: bool
    ready_for_exact_eval_dispatch: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RealizeConfig:
        if not isinstance(raw, Mapping):
            raise RealizationRefusal("config must be one JSON object")
        keys = frozenset(raw)
        unknown = sorted(keys - CONFIG_FIELDS)
        missing = sorted(CONFIG_FIELDS - keys)
        if unknown:
            raise RealizationRefusal(f"unknown config keys: {unknown}")
        if missing:
            raise RealizationRefusal(f"missing config keys: {missing}")

        for field in PATH_FIELDS:
            value = raw[field]
            if not isinstance(value, str) or not value:
                raise RealizationRefusal(f"{field} must be a non-empty string")
            if not Path(value).is_absolute():
                raise RealizationRefusal(f"{field} must be an absolute path")

        for field, expected in EXPECTED_CONSTANTS.items():
            value = raw[field]
            if type(value) is not type(expected):
                raise RealizationRefusal(
                    f"{field} has type {type(value).__name__}; "
                    f"expected {type(expected).__name__}"
                )
            if value != expected:
                raise RealizationRefusal(
                    f"{field}={value!r} does not match sealed value {expected!r}"
                )
        return cls(**dict(raw))

    @classmethod
    def from_json_path(cls, path: Path) -> RealizeConfig:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RealizationRefusal(f"cannot load typed config {path}: {exc}") from exc
        return cls.from_mapping(raw)

    def canonical_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def canonical_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.canonical_mapping()))


@dataclass(frozen=True)
class ArchiveCustody:
    path: str
    archive_bytes: int
    archive_sha256: str
    member_name: str
    member_bytes: int
    member_sha256: str
    compression: str
    framing: str


@dataclass(frozen=True)
class BatchMeasurement:
    pair_start: int
    pair_end_exclusive: int
    pair_rows: tuple[dict[str, Any], ...]
    candidate_frames_sha256: str
    gt_frames_sha256: str
    candidate_shape: tuple[int, ...]
    gt_shape: tuple[int, ...]


def _require_regular_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise RealizationRefusal(f"missing {label}: {path}")


def _is_under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def validate_config_paths(config: RealizeConfig) -> None:
    candidate = Path(config.candidate_archive)
    runtime = Path(config.runtime_dir)
    upstream = Path(config.upstream_dir)
    output = Path(config.ssd_output_dir)
    _require_regular_file(candidate, label="candidate archive")
    if not runtime.is_dir():
        raise RealizationRefusal(f"missing runtime directory: {runtime}")
    if not upstream.is_dir():
        raise RealizationRefusal(f"missing upstream directory: {upstream}")

    runtime_required = (
        runtime / "inflate.py",
        runtime / "inflate.sh",
        runtime / "src" / "codec.py",
        runtime / "src" / "codec_ctx.py",
        runtime / "src" / "codec_sidecar.py",
        runtime / "src" / "fec10_hybrid_decoder.py",
        runtime / "src" / "frame_selector.py",
        runtime / "src" / "model.py",
    )
    upstream_required = (
        upstream / "evaluate.py",
        upstream / "frame_utils.py",
        upstream / "modules.py",
        upstream / "models" / "posenet.safetensors",
        upstream / "models" / "segnet.safetensors",
        upstream / "public_test_video_names.txt",
    )
    for path in (*runtime_required, *upstream_required):
        _require_regular_file(path, label="bound source/runtime file")

    output_resolved = output.resolve(strict=False)
    if not any(
        _is_under(output_resolved, root.resolve(strict=False))
        for root in ALLOWED_SSD_ROOTS
    ):
        raise RealizationRefusal(
            f"ssd_output_dir is not on an allowed SSD tier: {output_resolved}"
        )

    ancestor = output_resolved
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise RealizationRefusal(f"no existing output ancestor for {output_resolved}")
    free = shutil.disk_usage(ancestor).free
    if free < MIN_OUTPUT_FREE_BYTES:
        raise RealizationRefusal(
            f"insufficient output storage: {free} < {MIN_OUTPUT_FREE_BYTES} bytes"
        )


def validate_execution_host() -> None:
    if platform.system() != "Darwin":
        raise RealizationRefusal(
            f"{EVIDENCE_AXIS} requires macOS, found {platform.system()}"
        )


def validate_archive_contract(config: RealizeConfig) -> ArchiveCustody:
    archive_path = Path(config.candidate_archive)
    archive_payload = archive_path.read_bytes()
    archive_sha = _sha256_bytes(archive_payload)
    if len(archive_payload) != config.expected_archive_bytes:
        raise RealizationRefusal(
            f"archive byte mismatch: {len(archive_payload)} "
            f"!= {config.expected_archive_bytes}"
        )
    if archive_sha != config.expected_archive_sha256:
        raise RealizationRefusal(
            f"archive SHA-256 mismatch: {archive_sha} "
            f"!= {config.expected_archive_sha256}"
        )
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            infos = archive.infolist()
            if len(infos) != 1:
                raise RealizationRefusal(
                    f"archive must have exactly one member, found {len(infos)}"
                )
            info = infos[0]
            if info.filename != config.expected_member_name:
                raise RealizationRefusal(
                    f"archive member {info.filename!r} "
                    f"!= {config.expected_member_name!r}"
                )
            if info.compress_type != zipfile.ZIP_STORED:
                raise RealizationRefusal("archive member must use ZIP_STORED")
            member = archive.read(info)
    except (OSError, zipfile.BadZipFile) as exc:
        raise RealizationRefusal(f"invalid candidate ZIP: {exc}") from exc

    if len(member) < 16 or member[:4] != b"FP11":
        raise RealizationRefusal("candidate member is not FP11 framed")
    source_length = int.from_bytes(member[4:8], "little")
    source_end = 8 + source_length
    if source_end + 2 > len(member):
        raise RealizationRefusal("truncated FP11 source section")
    if member[8:12] != b"CTXR":
        raise RealizationRefusal("FP11 source is not CTXR framed")
    selector_length = int.from_bytes(member[source_end : source_end + 2], "little")
    if source_end + 2 + selector_length > len(member):
        raise RealizationRefusal("truncated FP11 selector section")
    return ArchiveCustody(
        path=str(archive_path.resolve()),
        archive_bytes=len(archive_payload),
        archive_sha256=archive_sha,
        member_name=config.expected_member_name,
        member_bytes=len(member),
        member_sha256=_sha256_bytes(member),
        compression="ZIP_STORED",
        framing="FP11->CTXR->latent+sidecar->FECa_selector(+optional_DQS1)",
    )


def require_roundtrip_identity(
    roundtrip: Mapping[str, Any],
    *,
    archive_custody: ArchiveCustody,
) -> None:
    required_legs = (
        "latent_raw_roundtrip_byte_exact",
        "member_byte_exact",
        "archive_byte_exact",
    )
    for leg in required_legs:
        if roundtrip.get(leg) is not True:
            raise RealizationRefusal(f"round-trip identity leg failed: {leg}")
    if roundtrip.get("archive_bytes") != archive_custody.archive_bytes:
        raise RealizationRefusal("round-trip archive byte count differs from custody")
    if roundtrip.get("archive_sha256") != archive_custody.archive_sha256:
        raise RealizationRefusal("round-trip archive SHA-256 differs from custody")
    for key, value in roundtrip.items():
        if key.endswith("_byte_exact") and value is not True:
            raise RealizationRefusal(f"round-trip identity leg failed: {key}")


def _manifest_entry(root: Path, path: Path) -> dict[str, Any]:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if not _is_under(resolved, resolved_root):
        raise RealizationRefusal(f"manifest path escapes root: {path}")
    _require_regular_file(resolved, label="manifest source")
    return {
        "path": resolved.relative_to(resolved_root).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _content_manifest(
    *,
    schema: str,
    root: Path,
    paths: Iterable[Path],
) -> dict[str, Any]:
    unique = {path.resolve(): path for path in paths}
    entries = sorted(
        (_manifest_entry(root, path) for path in unique.values()),
        key=lambda row: row["path"],
    )
    canonical_body = {"schema": schema, "files": entries}
    return {
        **canonical_body,
        "root": str(root.resolve()),
        "manifest_sha256": _sha256_bytes(_canonical_json_bytes(canonical_body)),
    }


def runtime_content_manifest(runtime_dir: Path) -> dict[str, Any]:
    runtime_paths = [
        runtime_dir / "inflate.py",
        runtime_dir / "inflate.sh",
        *sorted((runtime_dir / "src").rglob("*.py")),
    ]
    return _content_manifest(
        schema="ddm_m7_runtime_content_manifest.v1",
        root=runtime_dir,
        paths=runtime_paths,
    )


def upstream_content_manifest(upstream_dir: Path) -> dict[str, Any]:
    names_path = upstream_dir / "public_test_video_names.txt"
    video_names = [
        line.strip()
        for line in names_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not video_names:
        raise RealizationRefusal("public_test_video_names.txt is empty")
    video_paths = [upstream_dir / "videos" / name for name in video_names]
    paths = [
        upstream_dir / "evaluate.py",
        upstream_dir / "frame_utils.py",
        upstream_dir / "modules.py",
        upstream_dir / "models" / "posenet.safetensors",
        upstream_dir / "models" / "segnet.safetensors",
        names_path,
        *video_paths,
    ]
    return _content_manifest(
        schema="ddm_m7_upstream_content_manifest.v1",
        root=upstream_dir,
        paths=paths,
    )


def _git_commit_identity(config_path: Path) -> dict[str, Any]:
    resolved_config = config_path.resolve()
    if not _is_under(resolved_config, REPO_ROOT.resolve()):
        raise RealizationRefusal(
            f"typed config must be the repo-owned file, got {resolved_config}"
        )
    tracked_paths = (
        Path(__file__).resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        (
            SRC_ROOT
            / "tac"
            / "canonical_equations"
            / "ddm_m7_realization_transfer_20260723.py"
        )
        .resolve()
        .relative_to(REPO_ROOT.resolve())
        .as_posix(),
        resolved_config.relative_to(REPO_ROOT.resolve()).as_posix(),
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *tracked_paths],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RealizationRefusal(
            "implementation/config must be committed and clean before measurement: "
            f"{status.stdout.strip()}"
        )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if not _is_git_oid(value):
        raise RealizationRefusal(f"invalid git HEAD: {value!r}")
    return {"git_head": value, "scoped_implementation_clean": True}


def _cpu_model() -> str:
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or "unknown"


def environment_identity() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for package in (
        "numpy",
        "torch",
        "av",
        "safetensors",
        "timm",
        "segmentation-models-pytorch",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "MISSING"
    uname = platform.uname()
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "packages": packages,
        "host": {
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "processor": uname.processor,
            "node": uname.node,
        },
        "microarchitecture": {
            "machine": platform.machine(),
            "cpu_model": _cpu_model(),
        },
    }


def _implementation_source_hashes() -> dict[str, str]:
    equation_path = (
        SRC_ROOT
        / "tac"
        / "canonical_equations"
        / "ddm_m7_realization_transfer_20260723.py"
    )
    return {
        "tools/realize_ddm_m7_relaxed_receiver.py": _sha256_file(Path(__file__)),
        "src/tac/canonical_equations/ddm_m7_realization_transfer_20260723.py": (
            _sha256_file(equation_path)
        ),
    }


def build_run_identity(
    *,
    config: RealizeConfig,
    config_path: Path,
    archive: ArchiveCustody,
    runtime_manifest: Mapping[str, Any],
    upstream_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "schema": IDENTITY_SCHEMA,
        "config": config.canonical_mapping(),
        "config_sha256": config.canonical_sha256(),
        "config_file": {
            "path": str(config_path.resolve()),
            "sha256": _sha256_file(config_path),
        },
        "archive": dataclasses.asdict(archive),
        "runtime_manifest": dict(runtime_manifest),
        "upstream_manifest": dict(upstream_manifest),
        **_git_commit_identity(config_path),
        "implementation_source_hashes": _implementation_source_hashes(),
        "environment": environment_identity(),
    }
    return {
        "identity": identity,
        "identity_sha256": _sha256_bytes(_canonical_json_bytes(identity)),
    }


def _with_body_hash(payload: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    body = dict(payload)
    body[field] = _sha256_bytes(_canonical_json_bytes(body))
    return body


def _validate_body_hash(payload: Mapping[str, Any], *, field: str) -> None:
    recorded = payload.get(field)
    body = dict(payload)
    body.pop(field, None)
    actual = _sha256_bytes(_canonical_json_bytes(body))
    if recorded != actual:
        raise RealizationRefusal(f"{field} mismatch: {recorded!r} != {actual}")


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _canonical_json_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RealizationRefusal(
                f"immutable JSON exists with different bytes: {path}"
            )
        return

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise RealizationRefusal(
                    f"immutable JSON raced with different bytes: {path}"
                ) from None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_payload(
    *,
    identity_bundle: Mapping[str, Any],
    measurement: BatchMeasurement,
) -> dict[str, Any]:
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "identity": dict(identity_bundle["identity"]),
        "identity_sha256": identity_bundle["identity_sha256"],
        "pair_interval": {
            "start": measurement.pair_start,
            "end_exclusive": measurement.pair_end_exclusive,
        },
        "pair_rows": list(measurement.pair_rows),
        "candidate_frames": {
            "sha256": measurement.candidate_frames_sha256,
            "shape": list(measurement.candidate_shape),
            "dtype": "uint8",
        },
        "gt_frames": {
            "sha256": measurement.gt_frames_sha256,
            "shape": list(measurement.gt_shape),
            "dtype": "uint8",
        },
    }
    return _with_body_hash(payload, field="checkpoint_sha256")


def _checkpoint_path(
    checkpoint_dir: Path,
    *,
    pair_start: int,
    pair_end_exclusive: int,
) -> Path:
    return checkpoint_dir / (
        f"batch_{pair_start:04d}_{pair_end_exclusive - 1:04d}.json"
    )


def write_batch_checkpoint(
    *,
    checkpoint_dir: Path,
    identity_bundle: Mapping[str, Any],
    measurement: BatchMeasurement,
) -> Path:
    path = _checkpoint_path(
        checkpoint_dir,
        pair_start=measurement.pair_start,
        pair_end_exclusive=measurement.pair_end_exclusive,
    )
    _write_immutable_json(
        path,
        checkpoint_payload(
            identity_bundle=identity_bundle,
            measurement=measurement,
        ),
    )
    return path


def _validate_pair_rows(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    expected_pair_ids: Sequence[int],
) -> list[dict[str, Any]]:
    if len(pair_rows) != len(expected_pair_ids):
        raise RealizationRefusal("per-pair row count does not match interval")
    normalized: list[dict[str, Any]] = []
    for row, expected_pair in zip(pair_rows, expected_pair_ids, strict=True):
        if set(row) != {"pair_id", "d_seg", "d_pose"}:
            raise RealizationRefusal("per-pair row has unexpected fields")
        if row["pair_id"] != expected_pair:
            raise RealizationRefusal(
                f"pair order mismatch: {row['pair_id']} != {expected_pair}"
            )
        d_seg = float(row["d_seg"])
        d_pose = float(row["d_pose"])
        if not math.isfinite(d_seg) or d_seg < 0.0:
            raise RealizationRefusal("d_seg must be finite and non-negative")
        if not math.isfinite(d_pose) or d_pose < 0.0:
            raise RealizationRefusal("d_pose must be finite and non-negative")
        normalized.append(
            {"pair_id": expected_pair, "d_seg": d_seg, "d_pose": d_pose}
        )
    return normalized


def load_contiguous_checkpoints(
    *,
    checkpoint_dir: Path,
    identity_bundle: Mapping[str, Any],
    n_pairs: int,
    batch_pairs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not checkpoint_dir.exists():
        return [], []
    paths = sorted(checkpoint_dir.glob("*.json"))
    parsed: list[tuple[int, Path, dict[str, Any]]] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RealizationRefusal(f"invalid checkpoint {path}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
            raise RealizationRefusal(f"unexpected checkpoint schema: {path}")
        _validate_body_hash(payload, field="checkpoint_sha256")
        if payload.get("identity_sha256") != identity_bundle["identity_sha256"]:
            raise RealizationRefusal(f"stale checkpoint identity: {path}")
        if payload.get("identity") != identity_bundle["identity"]:
            raise RealizationRefusal(f"checkpoint identity content differs: {path}")
        interval = payload.get("pair_interval")
        if not isinstance(interval, dict):
            raise RealizationRefusal(f"checkpoint interval missing: {path}")
        start = interval.get("start")
        end = interval.get("end_exclusive")
        if type(start) is not int or type(end) is not int:
            raise RealizationRefusal(f"checkpoint interval is not integral: {path}")
        parsed.append((start, path, payload))

    parsed.sort(key=lambda item: (item[0], str(item[1])))
    expected_start = 0
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for start, path, payload in parsed:
        if start != expected_start:
            raise RealizationRefusal(
                f"resume is not one contiguous prefix: expected {expected_start}, "
                f"found {start} in {path}"
            )
        expected_end = min(start + batch_pairs, n_pairs)
        end = payload["pair_interval"]["end_exclusive"]
        if end != expected_end:
            raise RealizationRefusal(
                f"checkpoint interval {start}:{end} is not canonical "
                f"{start}:{expected_end}"
            )
        rows = _validate_pair_rows(
            payload.get("pair_rows", []),
            expected_pair_ids=list(range(start, end)),
        )
        for frame_key in ("candidate_frames", "gt_frames"):
            frame = payload.get(frame_key)
            if not isinstance(frame, dict) or not _is_sha256(frame.get("sha256")):
                raise RealizationRefusal(
                    f"{frame_key} content hash missing in {path}"
                )
        all_rows.extend(rows)
        summaries.append(
            {
                "path": str(path.resolve()),
                "pair_start": start,
                "pair_end_exclusive": end,
                "checkpoint_sha256": payload["checkpoint_sha256"],
            }
        )
        expected_start = end
    return all_rows, summaries


def _as_numpy(value: Any) -> Any:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _frame_sha256(frames: Any) -> str:
    import numpy as np

    contiguous = np.ascontiguousarray(_as_numpy(frames))
    return _sha256_bytes(memoryview(contiguous).cast("B"))


def _normalize_candidate_frames(candidate: Any, *, batch_size: int) -> Any:
    import numpy as np

    array = np.asarray(candidate)
    if array.ndim == 5 and array.shape[:2] == (batch_size, 2):
        return array
    if array.ndim == 4 and array.shape[0] == batch_size * 2:
        return array.reshape(batch_size, 2, *array.shape[1:])
    raise RealizationRefusal(
        f"renderer returned unexpected candidate shape {array.shape}"
    )


def measure_batch_stream(
    batches: Iterable[Any],
    *,
    n_pairs: int,
    batch_pairs: int,
    resume_rows: Sequence[Mapping[str, Any]],
    render_batch: Callable[[Sequence[int]], Any],
    score_batch: Callable[[Any, Any, Sequence[int]], tuple[Any, Any]],
    on_completed_batch: Callable[[BatchMeasurement], None],
) -> list[dict[str, Any]]:
    """Run the scorer-free injectable batch loop used by the real measurement."""

    prefix_ids = [row.get("pair_id") for row in resume_rows]
    if prefix_ids != list(range(len(resume_rows))):
        raise RealizationRefusal("resume rows are not a contiguous prefix")
    rows = [dict(row) for row in resume_rows]
    resume_count = len(rows)
    next_pair = 0

    for item in batches:
        gt_batch = item[2] if isinstance(item, tuple) and len(item) == 3 else item
        gt_array = _as_numpy(gt_batch)
        if gt_array.ndim < 1:
            raise RealizationRefusal("ground-truth batch has no batch dimension")
        batch_size = int(gt_array.shape[0])
        if next_pair >= n_pairs:
            raise RealizationRefusal("AVVideoDataset yielded more than n_pairs")
        expected_size = min(batch_pairs, n_pairs - next_pair)
        if batch_size != expected_size:
            raise RealizationRefusal(
                f"non-canonical batch size at pair {next_pair}: "
                f"{batch_size} != {expected_size}"
            )
        end = next_pair + batch_size
        pair_ids = list(range(next_pair, end))
        if end <= resume_count:
            next_pair = end
            continue
        if next_pair < resume_count:
            raise RealizationRefusal("resume prefix ends inside a canonical batch")

        candidate = _normalize_candidate_frames(
            render_batch(pair_ids),
            batch_size=batch_size,
        )
        if tuple(candidate.shape) != tuple(gt_array.shape):
            raise RealizationRefusal(
                f"candidate/GT shape mismatch: {candidate.shape} != {gt_array.shape}"
            )
        if str(candidate.dtype) != "uint8" or str(gt_array.dtype) != "uint8":
            raise RealizationRefusal("candidate and GT frames must be uint8")
        d_seg_raw, d_pose_raw = score_batch(gt_batch, candidate, pair_ids)
        d_seg = _as_numpy(d_seg_raw).reshape(-1)
        d_pose = _as_numpy(d_pose_raw).reshape(-1)
        if len(d_seg) != batch_size or len(d_pose) != batch_size:
            raise RealizationRefusal("scorer did not return one value per pair")
        pair_rows = _validate_pair_rows(
            [
                {
                    "pair_id": pair_id,
                    "d_seg": float(d_seg[offset]),
                    "d_pose": float(d_pose[offset]),
                }
                for offset, pair_id in enumerate(pair_ids)
            ],
            expected_pair_ids=pair_ids,
        )
        measurement = BatchMeasurement(
            pair_start=next_pair,
            pair_end_exclusive=end,
            pair_rows=tuple(pair_rows),
            candidate_frames_sha256=_frame_sha256(candidate),
            gt_frames_sha256=_frame_sha256(gt_array),
            candidate_shape=tuple(int(v) for v in candidate.shape),
            gt_shape=tuple(int(v) for v in gt_array.shape),
        )
        on_completed_batch(measurement)
        rows.extend(pair_rows)
        del candidate, d_seg, d_pose
        next_pair = end

    if next_pair != n_pairs:
        raise RealizationRefusal(
            f"AVVideoDataset yielded {next_pair} pairs; expected {n_pairs}"
        )
    aggregate_pair_rows(rows, n_pairs=n_pairs)
    return rows


def aggregate_pair_rows(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    n_pairs: int,
) -> dict[str, float]:
    ids = [row.get("pair_id") for row in pair_rows]
    expected = list(range(n_pairs))
    if ids != expected:
        duplicates = sorted(
            pair_id
            for pair_id in set(ids)
            if ids.count(pair_id) > 1
        )
        missing = sorted(set(expected) - set(ids))
        raise RealizationRefusal(
            f"pair IDs must be exactly 0..{n_pairs - 1} once each; "
            f"duplicates={duplicates}, missing={missing}"
        )
    normalized = _validate_pair_rows(pair_rows, expected_pair_ids=expected)
    return {
        "d_seg": math.fsum(row["d_seg"] for row in normalized) / n_pairs,
        "d_pose": math.fsum(row["d_pose"] for row in normalized) / n_pairs,
    }


def _terms_mapping(terms: ContestScoreTerms) -> dict[str, float]:
    return dataclasses.asdict(terms)


def _gap_mapping(gap: ScoreGapDecomposition) -> dict[str, float]:
    return dataclasses.asdict(gap)


def _validate_checkpoint_summaries(
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    n_pairs: int,
    batch_pairs: int,
) -> list[dict[str, Any]]:
    ordered = sorted(checkpoints, key=lambda row: row.get("pair_start", -1))
    expected_start = 0
    normalized: list[dict[str, Any]] = []
    for row in ordered:
        start = row.get("pair_start")
        end = row.get("pair_end_exclusive")
        digest = row.get("checkpoint_sha256")
        path = row.get("path")
        if start != expected_start:
            raise RealizationRefusal(
                f"checkpoint summaries are not contiguous at {expected_start}"
            )
        expected_end = min(expected_start + batch_pairs, n_pairs)
        if end != expected_end:
            raise RealizationRefusal(
                f"checkpoint summary ends at {end}; expected {expected_end}"
            )
        if not _is_sha256(digest) or not isinstance(path, str) or not path:
            raise RealizationRefusal("checkpoint summary custody is incomplete")
        normalized.append(dict(row))
        expected_start = expected_end
    if expected_start != n_pairs:
        raise RealizationRefusal(
            f"checkpoint summaries cover {expected_start} pairs; expected {n_pairs}"
        )
    return normalized


def build_final_receipt(
    *,
    config: RealizeConfig,
    identity_bundle: Mapping[str, Any],
    archive: ArchiveCustody,
    roundtrip: Mapping[str, Any],
    pair_rows: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[Mapping[str, Any]],
    frontier_snapshot: DynamicFrontierTargetSnapshot | None = None,
    now_utc_iso: str | None = None,
) -> dict[str, Any]:
    frontier = _require_dynamic_frontier_snapshot(
        frontier_snapshot,
        now_utc_iso=now_utc_iso,
    )
    aggregates = aggregate_pair_rows(pair_rows, n_pairs=config.n_pairs)
    normalized_checkpoints = _validate_checkpoint_summaries(
        checkpoints,
        n_pairs=config.n_pairs,
        batch_pairs=config.batch_pairs,
    )
    realized = contest_score_terms(
        d_seg=aggregates["d_seg"],
        d_pose=aggregates["d_pose"],
        archive_bytes=archive.archive_bytes,
        reference_bytes=config.reference_bytes,
    )
    counterfactual = contest_score_terms(
        d_seg=config.counterfactual_d_seg,
        d_pose=config.counterfactual_d_pose,
        archive_bytes=archive.archive_bytes,
        reference_bytes=config.reference_bytes,
    )
    gap = score_gap_decomposition(
        counterfactual=counterfactual,
        realized=realized,
    )
    require_score_gap_closure(gap)
    if gap.rate != 0.0:
        raise RealizationRefusal("counterfactual and realized rate terms differ")
    ratios = realization_transfer_ratios(
        counterfactual_d_seg=config.counterfactual_d_seg,
        counterfactual_d_pose=config.counterfactual_d_pose,
        realized_d_seg=aggregates["d_seg"],
        realized_d_pose=aggregates["d_pose"],
    )
    routing = (
        ROUTING_CANDIDATE
        if realized.total < frontier.target_score
        else ROUTING_NOT_CANDIDATE
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "equation_id": DDM_M7_REALIZATION_TRANSFER_EQUATION_ID,
        "identity": dict(identity_bundle["identity"]),
        "identity_sha256": identity_bundle["identity_sha256"],
        "archive": dataclasses.asdict(archive),
        "roundtrip": dict(roundtrip),
        "pair_count": config.n_pairs,
        "pair_ids_sha256": _sha256_bytes(
            _canonical_json_bytes([row["pair_id"] for row in pair_rows])
        ),
        "pair_rows_sha256": _sha256_bytes(_canonical_json_bytes(list(pair_rows))),
        "checkpoints": normalized_checkpoints,
        "realized": {
            "d_seg": aggregates["d_seg"],
            "d_pose": aggregates["d_pose"],
            "score_terms": _terms_mapping(realized),
        },
        "arithmetic_counterfactual": {
            "d_seg": config.counterfactual_d_seg,
            "d_pose": config.counterfactual_d_pose,
            "score_terms": _terms_mapping(counterfactual),
            "caveat": (
                "These solve-distortion values belonged to a different high-byte "
                "exact-C1 object and were never properties of this 177169-byte "
                "receiver member."
            ),
        },
        "realization_transfer_ratios": dataclasses.asdict(ratios),
        "counterfactual_to_realized_score_gap": _gap_mapping(gap),
        "dynamic_frontier_target": dataclasses.asdict(frontier),
        "comparisons": {
            "dynamic_frontier_target_score": frontier.target_score,
            "delta_vs_dynamic_frontier": realized.total - frontier.target_score,
            "counterfactual_score": counterfactual.total,
            "delta_vs_counterfactual": realized.total - counterfactual.total,
        },
        "routing_label": routing,
        "main_review_and_dispatch_required": routing == ROUTING_CANDIDATE,
        "evidence_axis": config.evidence_axis,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    receipt = _with_body_hash(receipt, field="receipt_sha256")
    verify_final_receipt(
        receipt,
        frontier_snapshot=frontier,
        now_utc_iso=now_utc_iso,
    )
    return receipt


def verify_final_receipt(
    receipt: Mapping[str, Any],
    *,
    frontier_snapshot: DynamicFrontierTargetSnapshot | None = None,
    now_utc_iso: str | None = None,
) -> None:
    frontier = _require_dynamic_frontier_snapshot(
        frontier_snapshot,
        now_utc_iso=now_utc_iso,
    )
    if receipt.get("dynamic_frontier_target") != dataclasses.asdict(frontier):
        raise RealizationRefusal(
            "final receipt does not bind the current canonical dynamic frontier"
        )
    _verify_final_receipt_structure(
        receipt,
        expected_target_score=frontier.target_score,
    )


def _verify_final_receipt_structure(
    receipt: Mapping[str, Any],
    *,
    expected_target_score: float,
) -> float:
    """Verify immutable receipt truth against one explicitly named threshold.

    This helper deliberately does not read live pointer state.  Historical
    receipt validation and current routing are separate operations: the former
    preserves what was measured, while the latter must reopen today's pointer.
    """

    if (
        isinstance(expected_target_score, bool)
        or not isinstance(expected_target_score, (int, float))
        or not math.isfinite(float(expected_target_score))
        or float(expected_target_score) <= 0.0
    ):
        raise RealizationRefusal("final receipt target score is invalid")
    target_score = float(expected_target_score)
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise RealizationRefusal("unexpected final receipt schema")
    _validate_body_hash(receipt, field="receipt_sha256")
    for field in (
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
    ):
        if receipt.get(field) is not False:
            raise RealizationRefusal(f"authority field must remain false: {field}")
    if receipt.get("evidence_axis") != EVIDENCE_AXIS:
        raise RealizationRefusal("final receipt evidence axis differs")
    embedded_frontier = receipt.get("dynamic_frontier_target")
    if not isinstance(embedded_frontier, Mapping):
        raise RealizationRefusal("final receipt has no embedded frontier snapshot")
    embedded_target = embedded_frontier.get("target_score")
    if (
        isinstance(embedded_target, bool)
        or not isinstance(embedded_target, (int, float))
        or not math.isfinite(float(embedded_target))
        or float(embedded_target) != target_score
    ):
        raise RealizationRefusal("embedded frontier target differs from the verification threshold")
    realized = receipt.get("realized", {}).get("score_terms", {})
    score = realized.get("total")
    threshold = receipt.get("comparisons", {}).get("dynamic_frontier_target_score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
    ):
        raise RealizationRefusal("final dynamic-frontier routing inputs are missing")
    if float(threshold) != target_score:
        raise RealizationRefusal(
            "final routing threshold differs from the embedded dynamic frontier"
        )
    score_value = float(score)
    comparisons = receipt.get("comparisons", {})
    recorded_delta = comparisons.get("delta_vs_dynamic_frontier")
    if (
        isinstance(recorded_delta, bool)
        or not isinstance(recorded_delta, (int, float))
        or not math.isfinite(float(recorded_delta))
        or not math.isclose(
            float(recorded_delta),
            score_value - target_score,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise RealizationRefusal("final dynamic-frontier delta does not close")
    expected_routing = (
        ROUTING_CANDIDATE if score_value < target_score else ROUTING_NOT_CANDIDATE
    )
    if receipt.get("routing_label") != expected_routing:
        raise RealizationRefusal("final routing label does not match realized score")
    if receipt.get("main_review_and_dispatch_required") is not (
        expected_routing == ROUTING_CANDIDATE
    ):
        raise RealizationRefusal("final review/dispatch flag does not match routing")
    gap_raw = receipt.get("counterfactual_to_realized_score_gap", {})
    try:
        gap = ScoreGapDecomposition(
            seg=float(gap_raw["seg"]),
            pose=float(gap_raw["pose"]),
            rate=float(gap_raw["rate"]),
            total=float(gap_raw["total"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RealizationRefusal("final score-gap decomposition is invalid") from exc
    try:
        require_score_gap_closure(gap)
    except ValueError as exc:
        raise RealizationRefusal(str(exc)) from exc
    if gap.rate != 0.0:
        raise RealizationRefusal("final rate gap must be exactly zero")
    return score_value


def verify_historical_final_receipt(receipt: Mapping[str, Any]) -> None:
    """Verify a stored receipt against its own immutable pointer snapshot.

    This establishes historical structural integrity only.  It does not say
    that the embedded threshold is current and must never steer a new action.
    """

    embedded = receipt.get("dynamic_frontier_target")
    if not isinstance(embedded, Mapping):
        raise RealizationRefusal("historical receipt has no embedded frontier snapshot")
    target_score = embedded.get("target_score")
    _verify_final_receipt_structure(
        receipt,
        expected_target_score=target_score,
    )


def reroute_final_receipt_against_current_frontier(
    receipt: Mapping[str, Any],
    *,
    frontier_snapshot: DynamicFrontierTargetSnapshot | None = None,
    now_utc_iso: str | None = None,
) -> dict[str, Any]:
    """Return a new current-pointer audit without mutating historical truth."""

    verify_historical_final_receipt(receipt)
    frontier = _require_dynamic_frontier_snapshot(
        frontier_snapshot,
        now_utc_iso=now_utc_iso,
    )
    realized_total = float(receipt["realized"]["score_terms"]["total"])
    current_routing = (
        ROUTING_CANDIDATE
        if realized_total < frontier.target_score
        else ROUTING_NOT_CANDIDATE
    )
    embedded = receipt["dynamic_frontier_target"]
    audit = {
        "schema": REROUTE_AUDIT_SCHEMA,
        "historical_receipt_sha256": receipt["receipt_sha256"],
        "historical_pointer_sha256": embedded["pointer_sha256"],
        "historical_target_score": embedded["target_score"],
        "current_dynamic_frontier_target": dataclasses.asdict(frontier),
        "current_target_score": frontier.target_score,
        "realized_total": realized_total,
        "delta_vs_current_frontier": realized_total - frontier.target_score,
        "current_routing_label": current_routing,
        "pointer_moved": embedded["pointer_sha256"] != frontier.pointer_sha256,
        "historical_structure_verified": True,
        "current_pointer_reopened_and_verified": True,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    return _with_body_hash(audit, field="audit_sha256")


def _require_module_under(module: Any, root: Path, *, label: str) -> None:
    module_path = Path(module.__file__).resolve()
    if not _is_under(module_path, root.resolve()):
        raise RealizationRefusal(
            f"{label} imported from stale path {module_path}, expected under {root}"
        )


def _load_upstream_modules(upstream_dir: Path) -> tuple[Any, Any]:
    upstream_text = str(upstream_dir.resolve())
    if upstream_text not in sys.path:
        sys.path.insert(0, upstream_text)
    frame_utils = importlib.import_module("frame_utils")
    modules = importlib.import_module("modules")
    _require_module_under(frame_utils, upstream_dir, label="frame_utils")
    _require_module_under(modules, upstream_dir, label="modules")
    return frame_utils, modules


class FrozenUpstreamScorer:
    def __init__(self, *, upstream_dir: Path) -> None:
        import torch

        _, modules = _load_upstream_modules(upstream_dir)
        self.torch = torch
        self.net = modules.DistortionNet().eval().to(device=torch.device("cpu"))
        self.net.load_state_dicts(
            upstream_dir / "models" / "posenet.safetensors",
            upstream_dir / "models" / "segnet.safetensors",
            torch.device("cpu"),
        )

    def score(
        self,
        gt_batch: Any,
        candidate_batch: Any,
        _pair_ids: Sequence[int],
    ) -> tuple[Any, Any]:
        torch = self.torch
        gt = (
            gt_batch.to(device="cpu")
            if hasattr(gt_batch, "to")
            else torch.from_numpy(_as_numpy(gt_batch)).to(device="cpu")
        )
        candidate = torch.from_numpy(_as_numpy(candidate_batch)).to(device="cpu")
        with torch.inference_mode():
            d_pose, d_seg = self.net.compute_distortion(gt, candidate)
        return d_seg.cpu().numpy(), d_pose.cpu().numpy()


def _configure_deterministic_cpu(config: RealizeConfig) -> None:
    import numpy as np
    import torch

    if config.device != "cpu":
        raise RealizationRefusal("only CPU Torch execution is permitted")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.num_threads)
    torch.use_deterministic_algorithms(True)


def _dataset(upstream_dir: Path, config: RealizeConfig) -> Any:
    import torch

    frame_utils, _ = _load_upstream_modules(upstream_dir)
    video_names = [
        line.strip()
        for line in (upstream_dir / "public_test_video_names.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    dataset = frame_utils.AVVideoDataset(
        video_names,
        data_dir=upstream_dir / "videos",
        batch_size=config.batch_pairs,
        device=torch.device("cpu"),
        num_threads=config.num_threads,
        seed=config.seed,
        prefetch_queue_depth=4,
    )
    dataset.prepare_data()
    return dataset


def run(config_path: Path) -> Path:
    config = RealizeConfig.from_json_path(config_path)
    validate_config_paths(config)
    validate_execution_host()
    archive = validate_archive_contract(config)
    runtime_manifest = runtime_content_manifest(Path(config.runtime_dir))
    upstream_manifest = upstream_content_manifest(Path(config.upstream_dir))
    identity_bundle = build_run_identity(
        config=config,
        config_path=config_path,
        archive=archive,
        runtime_manifest=runtime_manifest,
        upstream_manifest=upstream_manifest,
    )

    from tac.click_polish import FrozenPacket, Renderer

    packet = FrozenPacket.parse(config.candidate_archive, config.runtime_dir)
    for name in ("codec_ctx", "codec", "codec_sidecar", "inflate"):
        _require_module_under(
            getattr(packet.ns, name),
            Path(config.runtime_dir),
            label=f"runtime {name}",
        )
    if tuple(packet.Q0.shape) != (config.n_pairs, 28):
        raise RealizationRefusal(
            f"receiver-native Q0 shape {packet.Q0.shape} != ({config.n_pairs}, 28)"
        )
    if str(packet.Q0.dtype) != "uint8":
        raise RealizationRefusal(f"receiver-native Q0 dtype is {packet.Q0.dtype}")
    roundtrip = packet.verify_roundtrip()
    require_roundtrip_identity(roundtrip, archive_custody=archive)

    output_dir = Path(config.ssd_output_dir)
    checkpoint_dir = output_dir / "batch_checkpoints"
    resume_rows, checkpoint_summaries = load_contiguous_checkpoints(
        checkpoint_dir=checkpoint_dir,
        identity_bundle=identity_bundle,
        n_pairs=config.n_pairs,
        batch_pairs=config.batch_pairs,
    )

    _configure_deterministic_cpu(config)
    renderer = Renderer(packet, device="cpu", drop_sidecar=False)
    scorer = FrozenUpstreamScorer(upstream_dir=Path(config.upstream_dir))

    def render_batch(pair_ids: Sequence[int]) -> Any:
        return renderer.render(
            packet.Q0,
            pair_ids,
            batch_pairs=config.batch_pairs,
        )

    def on_completed_batch(measurement: BatchMeasurement) -> None:
        path = write_batch_checkpoint(
            checkpoint_dir=checkpoint_dir,
            identity_bundle=identity_bundle,
            measurement=measurement,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        checkpoint_summaries.append(
            {
                "path": str(path.resolve()),
                "pair_start": measurement.pair_start,
                "pair_end_exclusive": measurement.pair_end_exclusive,
                "checkpoint_sha256": payload["checkpoint_sha256"],
            }
        )

    pair_rows = measure_batch_stream(
        _dataset(Path(config.upstream_dir), config),
        n_pairs=config.n_pairs,
        batch_pairs=config.batch_pairs,
        resume_rows=resume_rows,
        render_batch=render_batch,
        score_batch=scorer.score,
        on_completed_batch=on_completed_batch,
    )
    receipt = build_final_receipt(
        config=config,
        identity_bundle=identity_bundle,
        archive=archive,
        roundtrip=roundtrip,
        pair_rows=pair_rows,
        checkpoints=checkpoint_summaries,
    )
    receipt_path = output_dir / "ddm_m7_relaxed_receiver_realization_receipt.json"
    _write_immutable_json(receipt_path, receipt)
    return receipt_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realize the frozen DDM M7 receiver member on local CPU."
    )
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt_path = run(args.config)
    except RealizationRefusal as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "routing_label": receipt["routing_label"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
