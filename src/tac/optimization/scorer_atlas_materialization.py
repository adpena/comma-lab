# SPDX-License-Identifier: MIT
"""Materialize the locked DDM AT1x scorer atlas without recomputing VJPs.

The functions in this module are deliberately split into small deterministic
stages.  They can certify a lock-selected environment, derive frozen-weight
closed forms, index the settled n600 VJP sidecars, compute float64 contractions,
and parse one explicitly supplied upstream calibration report.  Importing this
module never loads a scorer, touches the SSD, or runs an evaluator.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from tac.optimization.scorer_analytic_atlas import (
    AnalyticFactor,
    SourceHashStamp,
    build_factor,
    derive_batchnorm_expected_stats,
    derive_bn_silu_contrast,
    derive_kernel_dft_bank,
    derive_se_gate_closed_form,
)

SCHEMA = "ddm_at1x_atlas_materialization.v1"
ENVIRONMENT_SCHEMA = "ddm_at1x_locked_environment.v1"
TENSOR_INDEX_SCHEMA = "ddm_at1x_tensor_index.v1"
CONTRACTION_SCHEMA = "ddm_at1x_contraction_spectrum.v1"
CALIBRATION_SCHEMA = "ddm_at1x_locked_calibration.v1"
EVIDENCE_AXIS = "[macOS-CPU locked-env upstream frozen-harness advisory]"
SSD_ROOT = Path("/Volumes/VertigoDataTier/pact")
PAIR_COUNT = 600
SEG_BINDING = "segnet_head_rank4_linear_flipdist_v1"
DIRECTIVES = ("2026-07-19T19:42:07Z", "2026-07-19T19:48:01Z")
OBSERVED_E2 = {
    "total": 43.411509751432,
    "d_seg": 0.02861482,
    "d_pose": 162.58094788,
    "archive_bytes": 343466,
}
FROZEN_SCORER_D_SEG = 0.027470296224


class AtlasMaterializationError(ValueError):
    """A lock, custody, coverage, contraction, or receipt check failed closed."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise AtlasMaterializationError(f"required file is absent: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def require_ssd_environment(path: Path, *, ssd_root: Path = SSD_ROOT) -> Path:
    environment = path.resolve()
    root = ssd_root.resolve()
    if environment == root or not environment.is_relative_to(root):
        raise AtlasMaterializationError(f"environment must be a child of {root}, got {environment}")
    return environment


def storage_preflight(path: Path, *, required_free_bytes: int) -> dict[str, int | str | bool]:
    if isinstance(required_free_bytes, bool) or required_free_bytes <= 0:
        raise AtlasMaterializationError("required_free_bytes must be positive")
    anchor = path
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    usage = shutil.disk_usage(anchor)
    if usage.free < required_free_bytes:
        raise AtlasMaterializationError(f"insufficient storage: {usage.free} < {required_free_bytes}")
    return {
        "status": "PASS_STORAGE_PREFLIGHT",
        "path": str(path.resolve()),
        "required_free_bytes": required_free_bytes,
        "observed_free_bytes": usage.free,
        "passed": True,
    }


def certify_tree(path: Path) -> dict[str, Any]:
    """Hash a directory as ordered ``(relative path, bytes, sha256)`` rows."""

    root = path.resolve()
    if not root.is_dir():
        raise AtlasMaterializationError(f"tree does not exist: {root}")
    rows: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink():
            content = os.readlink(candidate).encode("utf-8")
            rows.append(
                {
                    "relative_path": relative,
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        elif candidate.is_file():
            rows.append(
                {
                    "relative_path": relative,
                    "bytes": candidate.stat().st_size,
                    "sha256": sha256_file(candidate),
                }
            )
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json_bytes(row))
        digest.update(b"\n")
    return {
        "root": str(root),
        "file_count": len(rows),
        "bytes": sum(int(row["bytes"]) for row in rows),
        "rows": rows,
        "tree_sha256": digest.hexdigest(),
    }


def version_set_sha256(versions: Mapping[str, str]) -> str:
    normalized = {str(name): str(version) for name, version in sorted(versions.items())}
    if not normalized or any(not name or not version for name, version in normalized.items()):
        raise AtlasMaterializationError("package version set must be nonempty")
    return payload_sha256(normalized)


def shared_receipt_contract(*, scoped_verdict: str) -> dict[str, Any]:
    return {
        "first_rung": True,
        "research_only": True,
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
        "scoped_verdict": scoped_verdict,
        "amplitude_factors": {
            "count": 0,
            "why": "no amplitude factor produced; no through-R/uint8 survival row exists",
        },
        "nonadditive_pools": {
            "summed_overlapping_factors": False,
            "policy": "overlapping factors are declared as pools and never summed",
        },
        "freshness_rule": "exact input-hash and version-set equality at consumption",
        "directive_consumption": [{"directive_timestamp": value, "status": "CONSUMED"} for value in DIRECTIVES],
        "storage_policy": "certify-or-block; preserve certified SSD bytes",
        "pointer": "UNCHANGED",
        "triality": {
            "dsl": "typed materialization CLI stages",
            "dag": "locked_env -> inventory -> factors + gaze -> calibration -> manifest",
            "equations": [SEG_BINDING],
        },
        "main_landing_review_required": True,
    }


def build_environment_receipt(
    *,
    environment: Path,
    upstream_root: Path,
    python_path: Path,
    uv_version: str,
    package_versions: Mapping[str, str],
    preflight: Mapping[str, Any],
    tree_certificate: Mapping[str, Any],
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    environment = require_ssd_environment(environment, ssd_root=ssd_root)
    upstream = upstream_root.resolve()
    python = python_path.absolute()
    if not python.is_relative_to(environment):
        raise AtlasMaterializationError("selected Python is outside the environment")
    if preflight.get("passed") is not True:
        raise AtlasMaterializationError("storage preflight did not pass")
    if tree_certificate.get("root") != str(environment):
        raise AtlasMaterializationError("tree certificate is for another environment")
    versions = dict(sorted((str(k), str(v)) for k, v in package_versions.items()))
    command = ["uv", "sync", "--frozen", "--group", "cpu", "--python", "3.11"]
    return {
        "schema": ENVIRONMENT_SCHEMA,
        **shared_receipt_contract(scoped_verdict="locked macOS-CPU library-source materialization only"),
        "environment": str(environment),
        "environment_disposition": {
            "reproducibly_rebuildable": True,
            "automatic_delete": False,
            "delete_policy": "NEVER_DELETE_ENVIRONMENT",
        },
        "sync": {
            "argv": command,
            "shell_command": "uv sync --frozen --group cpu --python 3.11",
            "cwd": str(upstream),
            "environment": {
                "UV_PROJECT_ENVIRONMENT": str(environment),
                "UV_LINK_MODE": "copy",
            },
            "uv_version": uv_version,
            "python": str(python),
        },
        "inputs": {
            "pyproject": file_identity(upstream / "pyproject.toml"),
            "uv_lock": file_identity(upstream / "uv.lock"),
        },
        "selected_observed_package_versions": versions,
        "package_version_set_sha256": version_set_sha256(versions),
        "storage_preflight": dict(preflight),
        "tree_certificate": dict(tree_certificate),
        "certify_or_block": "CERTIFIED_REBUILDABLE_KEEP_BYTES",
        "reconstruction_command": (
            f"cd {upstream} && UV_PROJECT_ENVIRONMENT={environment} "
            "UV_LINK_MODE=copy uv sync --frozen --group cpu --python 3.11"
        ),
    }


def unwrap_inventory(inventory: Mapping[str, Any]) -> Mapping[str, Any]:
    body = inventory.get("body", inventory)
    if not isinstance(body, Mapping):
        raise AtlasMaterializationError("inventory body is invalid")
    return body


def require_locked_inventory(inventory: Mapping[str, Any]) -> Mapping[str, Any]:
    body = unwrap_inventory(inventory)
    try:
        sources = body["source_strata"]["B_imported_library_sources"]
    except (KeyError, TypeError) as error:
        raise AtlasMaterializationError("inventory lacks library-source stratum") from error
    drift = sources.get("version_drift")
    gate = sources.get("binding_gate")
    if drift:
        raise AtlasMaterializationError(f"locked inventory has version_drift: {drift}")
    if gate != "PASS_LOCKED_LIBRARY_SOURCES_MATERIALIZED":
        raise AtlasMaterializationError("inventory did not PASS_LOCKED_LIBRARY_SOURCES_MATERIALIZED")
    return body


def wrap_receipt(body: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(body)
    return {"body": payload, "body_sha256": payload_sha256(payload)}


def write_immutable_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    """Atomically publish a stage receipt, accepting an identical resume."""

    payload = canonical_json_bytes(receipt) + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise AtlasMaterializationError(f"refusing to overwrite non-byte-identical receipt: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _provenanced_factor(
    factor: AnalyticFactor,
    *,
    network: str,
    layer_id: str,
    source_citation: Mapping[str, Any],
    checkpoint_sha256: str,
    locked_source_sha256: str,
    package_version_set_sha256: str,
    freshness_rule: str,
    consumer_status: str = "COUNTED_INERT",
) -> AnalyticFactor:
    payload = dict(factor.payload)
    payload["materialization_custody"] = {
        "FIRST-RUNG": True,
        "network": network,
        "layer_id": layer_id,
        "checkpoint_sha256": checkpoint_sha256,
        "locked_source_sha256": locked_source_sha256,
        "source_citation": dict(source_citation),
        "package_version_set_sha256": package_version_set_sha256,
        "freshness_rule": freshness_rule,
        "consumer_status": consumer_status,
    }
    return build_factor(
        factor_id=factor.factor_id,
        factor_kind=factor.factor_kind,
        status=factor.status,
        payload=payload,
        source_hashes=factor.source_hashes,
        pair_start=factor.pair_start,
        pair_stop=factor.pair_stop,
        network=network,
        layer_id=layer_id,
        consumer=factor.consumer,
        consumption_status=factor.consumption_status,
        nonadditive_pool_id=factor.nonadditive_pool_id,
        uint8_surviving_projection=factor.uint8_surviving_projection,
        tensor=factor.tensor,
    )


def source_citation_for_layer(inventory_body: Mapping[str, Any], *, network: str, layer_id: str) -> Mapping[str, Any]:
    modules = inventory_body["networks"][network]["modules"]
    matches = [row["source"] for row in modules if row["name"] == layer_id]
    if len(matches) != 1:
        raise AtlasMaterializationError(f"{network}:{layer_id}: expected one inventory source citation")
    citation = matches[0]
    required = {"path", "sha256", "line_start", "line_stop_inclusive"}
    if not required.issubset(citation):
        raise AtlasMaterializationError(f"{network}:{layer_id}: incomplete citation")
    return citation


def _weight_matrix(module: Any) -> tuple[np.ndarray, np.ndarray]:
    weight = module.weight.detach().cpu().numpy()
    bias_parameter = getattr(module, "bias", None)
    bias = (
        bias_parameter.detach().cpu().numpy()
        if bias_parameter is not None
        else np.zeros(weight.shape[0], dtype=weight.dtype)
    )
    if weight.ndim == 4 and weight.shape[-2:] == (1, 1):
        weight = weight[:, :, 0, 0]
    if weight.ndim != 2:
        raise AtlasMaterializationError("SE projection is not a linear/1x1 weight")
    return weight, bias


def _se_projections(module: Any) -> tuple[Any, Any]:
    names = (
        ("fc1", "fc2"),
        ("conv_reduce", "conv_expand"),
    )
    for first, second in names:
        if hasattr(module, first) and hasattr(module, second):
            return getattr(module, first), getattr(module, second)
    candidates = [
        child
        for child in module.modules()
        if child is not module and hasattr(child, "weight") and getattr(child.weight, "ndim", 0) in {2, 4}
    ]
    if len(candidates) >= 2:
        return candidates[0], candidates[-1]
    raise AtlasMaterializationError(f"{type(module).__name__}: cannot locate two SE projections")


def _actual_bn_silu_layers(network: Any) -> set[str]:
    result: set[str] = set()
    for parent_name, parent in network.named_modules():
        children = list(parent.named_children())
        for (left_name, left), (_, right) in itertools.pairwise(children):
            if "BatchNorm" in type(left).__name__ and type(right).__name__ == "SiLU":
                result.add(".".join(filter(None, (parent_name, left_name))))
        if "BatchNormAct" in type(parent).__name__:
            activation = getattr(parent, "act", getattr(parent, "activation", None))
            if activation is not None and type(activation).__name__ == "SiLU":
                result.add(parent_name)
    return result


def derive_network_closed_forms(
    *,
    networks: Mapping[str, Any],
    inventory: Mapping[str, Any],
    checkpoint_sha256s: Mapping[str, str],
    package_version_set_sha256: str,
    source_hashes: Mapping[str, Sequence[SourceHashStamp]],
) -> list[AnalyticFactor]:
    """Derive every requested frozen factor from already-loaded eval networks."""

    body = require_locked_inventory(inventory)
    factors: list[AnalyticFactor] = []
    freshness = "consume only while checkpoint, locked source, and version-set hashes match"
    for network_name, network in sorted(networks.items()):
        if network_name not in {"posenet", "segnet"}:
            raise AtlasMaterializationError(f"unsupported network: {network_name}")
        network.eval()
        bn_silu = _actual_bn_silu_layers(network)
        for layer_id, module in network.named_modules():
            if not layer_id:
                continue
            citation = source_citation_for_layer(body, network=network_name, layer_id=layer_id)
            common = {
                "network": network_name,
                "layer_id": layer_id,
                "source_citation": citation,
                "checkpoint_sha256": checkpoint_sha256s[network_name],
                "locked_source_sha256": citation["sha256"],
                "package_version_set_sha256": package_version_set_sha256,
                "freshness_rule": freshness,
            }
            factor_sources = (
                *source_hashes[network_name],
                SourceHashStamp(
                    source_id=f"{network_name}:{layer_id}:locked_source",
                    path=str(citation["path"]),
                    sha256=str(citation["sha256"]),
                    bytes=int(citation["bytes"]),
                    validity_horizon=freshness,
                ),
                SourceHashStamp(
                    source_id=f"{network_name}:{layer_id}:package_version_set",
                    path="package-version-set://locked-macos-cpu",
                    sha256=package_version_set_sha256,
                    bytes=64,
                    validity_horizon=freshness,
                ),
            )
            class_name = type(module).__name__
            if "BatchNorm" in class_name and getattr(module, "running_mean", None) is not None:
                channels = int(module.running_mean.numel())
                gamma = (
                    module.weight.detach().cpu().numpy()
                    if module.weight is not None
                    else np.ones(channels, dtype=np.float32)
                )
                beta = (
                    module.bias.detach().cpu().numpy()
                    if module.bias is not None
                    else np.zeros(channels, dtype=np.float32)
                )
                bn = derive_batchnorm_expected_stats(
                    layer_id=f"{network_name}:{layer_id}",
                    running_mean=module.running_mean.detach().cpu().numpy(),
                    running_variance=module.running_var.detach().cpu().numpy(),
                    gamma=gamma,
                    beta=beta,
                    epsilon=float(module.eps),
                    source_hashes=factor_sources,
                )
                bn = _provenanced_factor(bn, **common)
                factors.append(bn)
                if layer_id in bn_silu:
                    factors.append(
                        _provenanced_factor(
                            derive_bn_silu_contrast(
                                layer_id=f"{network_name}:{layer_id}",
                                bn_factor=bn,
                            ),
                            **common,
                        )
                    )
            if class_name in {"SEModule", "SqueezeExcite"}:
                reduce, expand = _se_projections(module)
                w1, b1 = _weight_matrix(reduce)
                w2, b2 = _weight_matrix(expand)
                activation: Literal["relu", "silu"] = "relu" if class_name == "SEModule" else "silu"
                factor = derive_se_gate_closed_form(
                    layer_id=f"{network_name}:{layer_id}",
                    reduce_weight=w1,
                    reduce_bias=b1,
                    expand_weight=w2,
                    expand_bias=b2,
                    activation=activation,
                    source_hashes=factor_sources,
                )
                factors.append(_provenanced_factor(factor, **common))
            weight = getattr(module, "weight", None)
            if class_name == "Conv2d" and weight is not None and weight.ndim == 4:
                factor = derive_kernel_dft_bank(
                    layer_id=f"{network_name}:{layer_id}",
                    kernels=weight.detach().cpu().numpy(),
                    source_hashes=factor_sources,
                )
                factors.append(_provenanced_factor(factor, **common))
    if not factors:
        raise AtlasMaterializationError("no locked closed forms were derived")
    return factors


def write_factor_shards(
    *,
    factors: Sequence[AnalyticFactor],
    shard_root: Path,
    reconstruction_command: str,
) -> dict[str, Any]:
    require_ssd_environment(shard_root)
    rows: list[dict[str, Any]] = []
    for index, factor in enumerate(sorted(factors, key=lambda row: row.factor_id)):
        filename = f"{index:06d}_{hashlib.sha256(factor.factor_id.encode()).hexdigest()[:16]}.json"
        path = shard_root / filename
        write_immutable_receipt(path, factor.to_dict())
        rows.append(
            {
                "factor_id": factor.factor_id,
                "factor_kind": factor.factor_kind,
                "network": factor.network,
                "layer_id": factor.layer_id,
                "content_sha256": factor.content_sha256,
                "shard": file_identity(path),
            }
        )
    return {
        "schema": "ddm_at1x_factor_shard_index.v1",
        **shared_receipt_contract(scoped_verdict="frozen-weight closed forms under the verified lock only"),
        "factor_count": len(rows),
        "factors": rows,
        "reconstruction_command": reconstruction_command,
        "certify_or_block": "CERTIFIED_REBUILDABLE_KEEP_BYTES",
    }


@dataclass(frozen=True)
class TensorIndexRow:
    pair_id: int
    tensor_name: str
    path: str
    archive_sha256: str
    tensor_sha256: str
    shape: tuple[int, ...]
    dtype: str
    version_stamp_id: str
    version_set_sha256: str

    def __post_init__(self) -> None:
        if not (0 <= self.pair_id < PAIR_COUNT):
            raise AtlasMaterializationError("tensor pair_id is out of range")
        if not self.tensor_name or not self.path:
            raise AtlasMaterializationError("tensor identity is incomplete")
        for name, value in (
            ("archive_sha256", self.archive_sha256),
            ("tensor_sha256", self.tensor_sha256),
            ("version_set_sha256", self.version_set_sha256),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise AtlasMaterializationError(f"{name} is not SHA-256")
        if not self.shape or any(value <= 0 for value in self.shape):
            raise AtlasMaterializationError("tensor shape is invalid")
        if not self.dtype or not self.version_stamp_id:
            raise AtlasMaterializationError("version stamp and dtype are required")

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["shape"] = list(self.shape)
        row["schema"] = TENSOR_INDEX_SCHEMA
        return row


def _tensor_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _manifest_sidecars(
    *,
    manifest_path: Path,
    expected_sha256: str,
    completed_pair_ids: Iterable[int],
) -> list[Mapping[str, Any]]:
    completed = {int(value) for value in completed_pair_ids}
    if sha256_file(manifest_path) != expected_sha256:
        raise AtlasMaterializationError(f"manifest hash mismatch: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [row for row in manifest.get("sidecars", []) if int(row["pair_id"]) in completed]
    observed = {int(row["pair_id"]) for row in rows}
    if observed != completed or len(rows) != len(completed):
        raise AtlasMaterializationError(
            f"manifest completed-sidecar mismatch: {manifest_path}; "
            f"expected={sorted(completed)}, observed={sorted(observed)}"
        )
    return rows


def _sidecars_from_campaign(campaign: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if campaign.get("status") != "COMPLETE_N600":
        raise AtlasMaterializationError("campaign is not COMPLETE_N600")
    require_exact_pair_coverage(campaign.get("final_completed_pair_ids", []))
    if campaign.get("refused_pair_ids"):
        raise AtlasMaterializationError("campaign has final refusals")
    sidecars: list[Mapping[str, Any]] = []
    for manifest_ref in campaign.get("source_manifests", []):
        sidecars.extend(
            _manifest_sidecars(
                manifest_path=Path(str(manifest_ref["path"])),
                expected_sha256=str(manifest_ref["sha256"]),
                completed_pair_ids=manifest_ref["completed_pair_ids"],
            )
        )
    for chunk in campaign.get("chunks", []):
        sidecars.extend(
            _manifest_sidecars(
                manifest_path=Path(str(chunk["path"])) / "manifest.json",
                expected_sha256=str(chunk["manifest_sha256"]),
                completed_pair_ids=chunk["completed_pair_ids"],
            )
        )
    return sidecars


def require_exact_pair_coverage(pair_ids: Iterable[int]) -> None:
    values = [int(value) for value in pair_ids]
    if len(values) != PAIR_COUNT or len(set(values)) != PAIR_COUNT:
        raise AtlasMaterializationError("pair coverage must contain 600 unique rows")
    if sorted(values) != list(range(PAIR_COUNT)):
        raise AtlasMaterializationError("pair coverage is not exact 0..599")


def validate_and_contract_sidecars(
    *,
    campaign: Mapping[str, Any],
    version_stamp_id: str,
    version_set_sha256: str,
    exact_v19_pair_ids: Iterable[int],
    checkpoint_dir: Path | None = None,
    process_pair_ids: Iterable[int] | None = None,
    verify_archive_hashes: bool = True,
) -> dict[str, Any]:
    """Validate settled sidecars and compute only the specified contractions."""

    sidecars = _sidecars_from_campaign(campaign)
    by_pair: dict[int, Mapping[str, Any]] = {}
    for sidecar in sidecars:
        pair_id = int(sidecar["pair_id"])
        if pair_id in by_pair:
            raise AtlasMaterializationError(f"duplicate sidecar for pair {pair_id}")
        by_pair[pair_id] = sidecar
    if set(by_pair) != set(range(PAIR_COUNT)):
        missing = sorted(set(range(PAIR_COUNT)) - set(by_pair))
        extra = sorted(set(by_pair) - set(range(PAIR_COUNT)))
        raise AtlasMaterializationError(f"sidecar coverage mismatch; missing={missing}, extra={extra}")

    selected_pair_ids = (
        list(range(PAIR_COUNT)) if process_pair_ids is None else [int(value) for value in process_pair_ids]
    )
    if (
        not selected_pair_ids
        or len(selected_pair_ids) != len(set(selected_pair_ids))
        or any(value < 0 or value >= PAIR_COUNT for value in selected_pair_ids)
    ):
        raise AtlasMaterializationError("selected gaze pair IDs must be unique and within 0..599")
    selected_pair_ids.sort()
    tensor_rows: list[dict[str, Any]] = []
    spectra: list[dict[str, Any]] = []
    required = (
        "pose_j_y",
        "pose_j_x",
        "seg_g_y",
        "seg_g_x",
        "head_pair_norms",
        "winner",
        "rival",
    )
    for pair_id in selected_pair_ids:
        sidecar = by_pair[pair_id]
        path = Path(str(sidecar["path"]))
        archive_sha = str(sidecar["sha256"])
        checkpoint_path = checkpoint_dir / f"pair_{pair_id:04d}.json" if checkpoint_dir is not None else None
        checkpoint_contract = {
            "pair_id": pair_id,
            "sidecar_path": str(path),
            "sidecar_sha256": archive_sha,
            "version_stamp_id": version_stamp_id,
            "version_set_sha256": version_set_sha256,
        }
        if checkpoint_path is not None and checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint.get("contract") != checkpoint_contract:
                raise AtlasMaterializationError(f"pair {pair_id}: stale or foreign gaze checkpoint")
            tensor_rows.extend(checkpoint["tensor_index_rows"])
            spectra.append(checkpoint["spectrum"])
            continue
        if verify_archive_hashes and sha256_file(path) != archive_sha:
            raise AtlasMaterializationError(f"sidecar archive hash mismatch: {path}")
        expected_hashes = sidecar.get("tensor_hashes", {})
        with np.load(path, allow_pickle=False) as tensors:
            absent = sorted(set(required) - set(tensors.files))
            if absent:
                raise AtlasMaterializationError(f"pair {pair_id}: missing tensors {absent}")
            loaded: dict[str, np.ndarray] = {}
            for name in required:
                array = tensors[name]
                tensor_sha = _tensor_sha256(array)
                if expected_hashes.get(name) != tensor_sha:
                    raise AtlasMaterializationError(f"pair {pair_id}:{name}: tensor hash mismatch")
                row = TensorIndexRow(
                    pair_id=pair_id,
                    tensor_name=name,
                    path=str(path),
                    archive_sha256=archive_sha,
                    tensor_sha256=tensor_sha,
                    shape=tuple(int(value) for value in array.shape),
                    dtype=str(array.dtype),
                    version_stamp_id=version_stamp_id,
                    version_set_sha256=version_set_sha256,
                )
                tensor_rows.append(row.to_dict())
                loaded[name] = array
            if int(np.asarray(tensors["pair_id"]).item()) != pair_id:
                raise AtlasMaterializationError(f"pair {pair_id}: embedded ID mismatch")
        spectrum = contraction_spectrum(
            pair_id=pair_id,
            pose_scorer_plane=loaded["pose_j_y"],
            pose_camera_input=loaded["pose_j_x"],
            seg_scorer_plane=loaded["seg_g_y"],
            seg_camera_input=loaded["seg_g_x"],
            head_pair_norms=loaded["head_pair_norms"],
            seg_binding=SEG_BINDING,
        )
        spectra.append(spectrum)
        if checkpoint_path is not None:
            write_immutable_receipt(
                checkpoint_path,
                {
                    "schema": "ddm_at1x_gaze_pair_checkpoint.v1",
                    "contract": checkpoint_contract,
                    "tensor_index_rows": tensor_rows[-len(required) :],
                    "spectrum": spectrum,
                },
            )
    full_coverage = selected_pair_ids == list(range(PAIR_COUNT))
    return {
        "schema": CONTRACTION_SCHEMA,
        **shared_receipt_contract(
            scoped_verdict=("settled n600 VJPs contracted at scorer_plane_y and camera_input_x only")
        ),
        "measured_relay_depths": ["scorer_plane_y", "camera_input_x"],
        "unmeasured_internal_layers_claimed": False,
        "hash_custody": {
            "archive_container_sha256": (
                "FRESHLY_RECOMPUTED" if verify_archive_hashes else "SETTLED_HASHED_CAMPAIGN_MANIFEST"
            ),
            "tensor_sha256": "FRESHLY_RECOMPUTED_FROM_DECOMPRESSED_ARRAYS",
        },
        "pair_count": len(selected_pair_ids),
        "selected_pair_ids": selected_pair_ids,
        "full_n600_coverage": full_coverage,
        "tensor_index_count": len(tensor_rows),
        "tensor_index_rows": tensor_rows,
        "pair_rows": spectra,
        "aggregates": aggregate_contractions(spectra) if full_coverage else None,
        "gaze_lambda_coverage": (
            classify_gaze_v19_coverage(exact_v19_pair_ids=exact_v19_pair_ids) if full_coverage else None
        ),
        "resume": {
            "unit": "pair",
            "checkpoint_count": (len(selected_pair_ids) if checkpoint_dir is not None else 0),
            "checkpoint_dir": (str(checkpoint_dir.resolve()) if checkpoint_dir is not None else None),
            "all_pair_checkpoints_preserved": checkpoint_dir is not None,
        },
    }


def _pose_gram_spectrum(array: np.ndarray, *, name: str) -> dict[str, Any]:
    values = np.asarray(array)
    if values.dtype != np.float32 or values.ndim < 2 or values.shape[0] != 6:
        raise AtlasMaterializationError(f"{name}: expected six fp32 Pose rows")
    rows = values.reshape(6, -1).astype(np.float64)
    nonzero = np.count_nonzero(rows, axis=1)
    if np.any(nonzero == 0):
        raise AtlasMaterializationError(f"{name}: every Pose row must be nonzero")
    gram = rows @ rows.T
    eigenvalues = np.linalg.eigvalsh(gram)
    return {
        "row_count": 6,
        "row_nonzero_counts": nonzero.astype(int).tolist(),
        "gram_shape": [6, 6],
        "eigenvalues_ascending": eigenvalues.tolist(),
        "trace": float(np.trace(gram)),
    }


def _seg_energy(array: np.ndarray, *, name: str) -> dict[str, Any]:
    values = np.asarray(array)
    if values.dtype != np.float32 or values.size == 0:
        raise AtlasMaterializationError(f"{name}: expected a nonempty fp32 Seg row")
    row = values.reshape(-1).astype(np.float64)
    energy = float(row @ row)
    if energy <= 0.0:
        raise AtlasMaterializationError(f"{name}: Seg row must be nonzero")
    return {
        "row_count": 1,
        "contracted_singular_energy": energy,
        "single_row_singular_value": math.sqrt(energy),
        "nonzero_count": int(np.count_nonzero(row)),
    }


def contraction_spectrum(
    *,
    pair_id: int,
    pose_scorer_plane: np.ndarray,
    pose_camera_input: np.ndarray,
    seg_scorer_plane: np.ndarray,
    seg_camera_input: np.ndarray,
    head_pair_norms: np.ndarray,
    seg_binding: str,
) -> dict[str, Any]:
    if seg_binding != SEG_BINDING:
        raise AtlasMaterializationError(f"Seg contraction must bind to {SEG_BINDING}")
    norms = np.asarray(head_pair_norms)
    if norms.dtype != np.float32 or norms.size == 0 or np.any(norms <= 0):
        raise AtlasMaterializationError("head-pair norms must be positive fp32")
    flat = norms.reshape(-1).astype(np.float64)
    return {
        "pair_id": pair_id,
        "pose": {
            "scorer_plane_y": _pose_gram_spectrum(pose_scorer_plane, name="scorer_plane_y"),
            "camera_input_x": _pose_gram_spectrum(pose_camera_input, name="camera_input_x"),
        },
        "seg": {
            "binding": seg_binding,
            "head_pullback_rank": 4,
            "scorer_plane_y": _seg_energy(seg_scorer_plane, name="scorer_plane_y"),
            "camera_input_x": _seg_energy(seg_camera_input, name="camera_input_x"),
            "head_pair_norm_distribution": {
                "count": int(flat.size),
                "min": float(flat.min()),
                "p50": float(np.quantile(flat, 0.5)),
                "p90": float(np.quantile(flat, 0.9)),
                "p99": float(np.quantile(flat, 0.99)),
                "max": float(flat.max()),
                "mean": float(flat.mean()),
            },
        },
    }


def aggregate_contractions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != PAIR_COUNT:
        raise AtlasMaterializationError("n600 aggregate requires exactly 600 rows")
    ids = [int(row["pair_id"]) for row in rows]
    if ids != list(range(PAIR_COUNT)):
        raise AtlasMaterializationError("aggregate rows must be ordered exact 0..599")

    def summary(values: Iterable[float]) -> dict[str, float]:
        array = np.asarray(list(values), dtype=np.float64)
        return {
            "min": float(array.min()),
            "mean": float(array.mean()),
            "p50": float(np.quantile(array, 0.5)),
            "p90": float(np.quantile(array, 0.9)),
            "max": float(array.max()),
        }

    return {
        "pair_count": PAIR_COUNT,
        "pose_trace": {
            depth: summary(row["pose"][depth]["trace"] for row in rows)
            for depth in ("scorer_plane_y", "camera_input_x")
        },
        "seg_contracted_singular_energy": {
            depth: summary(row["seg"][depth]["contracted_singular_energy"] for row in rows)
            for depth in ("scorer_plane_y", "camera_input_x")
        },
        "head_pair_norm_mean": summary(row["seg"]["head_pair_norm_distribution"]["mean"] for row in rows),
    }


def classify_gaze_v19_coverage(*, exact_v19_pair_ids: Iterable[int]) -> dict[str, Any]:
    exact = sorted(int(value) for value in exact_v19_pair_ids)
    if len(exact) != 8 or len(set(exact)) != 8 or any(value < 0 or value >= PAIR_COUNT for value in exact):
        raise AtlasMaterializationError("V19 exact joins must identify exactly 8 pairs")
    rows = [
        {
            "pair_id": pair_id,
            "classification": (
                "V19_EXACT_JOIN_AVAILABLE" if pair_id in set(exact) else "GAZE_MEASURED_V19_JOIN_OWED_COUNTED_INERT"
            ),
            "consumer": (
                "tac.ddm_costate_organ.build_live_ddm_costate"
                if pair_id in set(exact)
                else "receiver_closed_v19_evidence_join"
            ),
        }
        for pair_id in range(PAIR_COUNT)
    ]
    return {
        "pair_count": PAIR_COUNT,
        "v19_exact_join_count": 8,
        "gaze_measured_v19_join_owed_counted_inert": 592,
        "rows": rows,
    }


def parse_evaluate_report(value: str | Mapping[str, Any]) -> dict[str, float | int]:
    if isinstance(value, Mapping):
        source = value
    else:
        text = value.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping):
            source = parsed
        else:
            source = {}
            official_patterns = {
                "archive_bytes": r"(?im)Submission file size:\s*([\d,]+)\s*bytes",
                "d_seg": (
                    r"(?im)Average SegNet Distortion:\s*"
                    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)"
                ),
                "d_pose": (
                    r"(?im)Average PoseNet Distortion:\s*"
                    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)"
                ),
                "total": (
                    r"(?im)Final score:.*?=\s*"
                    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)\s*$"
                ),
            }
            for target, pattern in official_patterns.items():
                match = re.search(pattern, text)
                if match:
                    source[target] = match.group(1).replace(",", "")
            aliases = {
                "archive_bytes": ("archive_bytes", "archive size"),
                "d_seg": ("d_seg", "segmentation distortion"),
                "d_pose": ("d_pose", "pose distortion"),
                "total": ("total", "final_score", "score"),
            }
            for target, names in aliases.items():
                if target in source:
                    continue
                for name in names:
                    match = re.search(
                        rf"(?im)\b{re.escape(name)}\b\s*[:=]\s*"
                        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)",
                        text,
                    )
                    if match:
                        source[target] = match.group(1)
                        break
    aliases = {
        "archive_bytes": ("archive_bytes", "archive_size"),
        "d_seg": ("d_seg", "segmentation_distortion"),
        "d_pose": ("d_pose", "pose_distortion"),
        "total": ("total", "final_score", "score"),
    }
    result: dict[str, float | int] = {}
    for target, names in aliases.items():
        selected = next((source[name] for name in names if name in source), None)
        if selected is None and target == "total":
            continue
        if selected is None:
            raise AtlasMaterializationError(f"calibration report lacks {target}")
        number = float(str(selected).replace(",", ""))
        if not math.isfinite(number) or number < 0:
            raise AtlasMaterializationError(f"calibration {target} is invalid")
        result[target] = int(number) if target == "archive_bytes" else number
    return result


def score_formula(*, archive_bytes: int, d_seg: float, d_pose: float) -> float:
    return 100.0 * d_seg + math.sqrt(10.0 * d_pose) + 25.0 * archive_bytes / 37_545_489


def build_calibration_receipt(
    *,
    parsed: str | Mapping[str, Any],
    argv: Sequence[str],
    environment: Mapping[str, str],
    archive: Mapping[str, Any],
    runtime: Mapping[str, Any],
    upstream: Mapping[str, Any],
    stdout: Mapping[str, Any],
    stderr: Mapping[str, Any],
    report: Mapping[str, Any],
    wallclock_seconds: float,
) -> dict[str, Any]:
    if not argv or any(not str(value) for value in argv):
        raise AtlasMaterializationError("calibration requires exact nonempty argv")
    if not math.isfinite(wallclock_seconds) or wallclock_seconds <= 0:
        raise AtlasMaterializationError("calibration wallclock_seconds must be positive")
    measured = parse_evaluate_report(parsed)
    recomputed = score_formula(
        archive_bytes=int(measured["archive_bytes"]),
        d_seg=float(measured["d_seg"]),
        d_pose=float(measured["d_pose"]),
    )
    reported_total = measured.get("total")
    if reported_total is not None and not math.isclose(
        recomputed,
        float(reported_total),
        rel_tol=0,
        abs_tol=0.0050000001,
    ):
        raise AtlasMaterializationError(
            f"reported total disagrees with formula/rounding: {reported_total} vs {recomputed}"
        )
    observed_terms = {
        "seg": 100.0 * OBSERVED_E2["d_seg"],
        "pose": math.sqrt(10.0 * OBSERVED_E2["d_pose"]),
        "rate": 25.0 * OBSERVED_E2["archive_bytes"] / 37_545_489,
    }
    measured_terms = {
        "seg": 100.0 * float(measured["d_seg"]),
        "pose": math.sqrt(10.0 * float(measured["d_pose"])),
        "rate": 25.0 * int(measured["archive_bytes"]) / 37_545_489,
    }
    return {
        "schema": CALIBRATION_SCHEMA,
        **shared_receipt_contract(scoped_verdict="one locked-env upstream E2 frozen-harness advisory run"),
        "execution": {
            "argv": list(argv),
            "environment": dict(sorted(environment.items())),
            "wallclock_seconds": float(wallclock_seconds),
        },
        "custody": {
            "archive": dict(archive),
            "runtime": dict(runtime),
            "upstream": dict(upstream),
            "stdout": dict(stdout),
            "stderr": dict(stderr),
            "report": dict(report),
        },
        "measured": {
            "archive_bytes": int(measured["archive_bytes"]),
            "d_seg": float(measured["d_seg"]),
            "d_pose": float(measured["d_pose"]),
            "reported_total": (float(reported_total) if reported_total is not None else None),
            "formula_total": recomputed,
        },
        "reference_observed_environment": OBSERVED_E2,
        "frozen_scorer_realization": {
            "d_seg": FROZEN_SCORER_D_SEG,
            "signed_delta_to_upstream": OBSERVED_E2["d_seg"] - FROZEN_SCORER_D_SEG,
        },
        "signed_drift_locked_minus_observed": {
            "archive_bytes": int(measured["archive_bytes"]) - OBSERVED_E2["archive_bytes"],
            "d_seg": float(measured["d_seg"]) - OBSERVED_E2["d_seg"],
            "d_pose": float(measured["d_pose"]) - OBSERVED_E2["d_pose"],
            "total": recomputed - OBSERVED_E2["total"],
            "score_terms": {name: measured_terms[name] - observed_terms[name] for name in ("seg", "pose", "rate")},
        },
    }


def build_calibration_blocker_receipt(
    *,
    stderr_text: str,
    exit_code: int,
    argv: Sequence[str],
    environment: Mapping[str, str],
    archive: Mapping[str, Any],
    runtime: Mapping[str, Any],
    upstream: Mapping[str, Any],
    stdout: Mapping[str, Any],
    stderr: Mapping[str, Any],
    wallclock_seconds: float,
) -> dict[str, Any]:
    if exit_code == 0:
        raise AtlasMaterializationError("blocked calibration receipt requires a nonzero exit code")
    if not argv or any(not str(value) for value in argv):
        raise AtlasMaterializationError("calibration requires exact nonempty argv")
    if not math.isfinite(wallclock_seconds) or wallclock_seconds <= 0:
        raise AtlasMaterializationError("calibration wallclock_seconds must be positive")
    missing = re.search(
        r"ModuleNotFoundError:\s+No module named ['\"]([^'\"]+)['\"]",
        stderr_text,
    )
    dependency = missing.group(1) if missing else None
    return {
        "schema": "ddm_at1x_locked_calibration_blocker.v1",
        **shared_receipt_contract(scoped_verdict=("locked-env upstream E2 harness blocked before scorer evaluation")),
        "status": "BLOCKED_LOCKED_RUNTIME_DEPENDENCY",
        "execution": {
            "argv": list(argv),
            "environment": dict(sorted(environment.items())),
            "exit_code": exit_code,
            "wallclock_seconds": float(wallclock_seconds),
        },
        "custody": {
            "archive": dict(archive),
            "runtime": dict(runtime),
            "upstream": dict(upstream),
            "stdout": dict(stdout),
            "stderr": dict(stderr),
        },
        "blocker": {
            "class": "MISSING_RUNTIME_DEPENDENCY_IN_EXACT_UPSTREAM_LOCK",
            "module": dependency,
            "stderr_tail": stderr_text.splitlines()[-20:],
            "lock_contamination_refused": True,
        },
        "calibration": {
            "locked_environment_measurement": "NOT_MEASURED",
            "signed_drift_locked_minus_observed": None,
            "reference_observed_environment": OBSERVED_E2,
        },
        "frozen_scorer_realization": {
            "d_seg": FROZEN_SCORER_D_SEG,
            "observed_environment_d_seg": OBSERVED_E2["d_seg"],
            "signed_observed_minus_frozen": (OBSERVED_E2["d_seg"] - FROZEN_SCORER_D_SEG),
            "interpretation": (
                "priced observed-env realization gap only; not a measured locked-vs-observed library drift"
            ),
        },
    }


def build_atlas_manifest(
    *,
    environment_receipt: Mapping[str, Any],
    factor_index: Mapping[str, Any],
    contraction_atlas: Mapping[str, Any],
    calibration_receipt: Mapping[str, Any] | None,
    reconstruction_commands: Sequence[str],
) -> dict[str, Any]:
    counts = {
        "closed_forms": int(factor_index.get("factor_count", 0)),
        "gaze_pairs": int(contraction_atlas.get("pair_count", 0)),
        "jacobian_contraction_rows": len(contraction_atlas.get("pair_rows", [])),
    }
    if any(value <= 0 for value in counts.values()):
        raise AtlasMaterializationError(f"atlas counts must be nonzero: {counts}")
    layer_members: dict[str, list[str]] = {}
    for row in factor_index.get("factors", []):
        pool_id = f"scorer_layer::{row['network']}::{row['layer_id']}"
        layer_members.setdefault(pool_id, []).append(str(row["factor_id"]))
    overlapping_pools = [
        {
            "pool_id": pool_id,
            "member_count": len(members),
            "factor_ids": sorted(members),
            "aggregation": "NONADDITIVE_MAX_OR_MEASURED_JOINT_ONLY",
        }
        for pool_id, members in sorted(layer_members.items())
        if len(members) > 1
    ]
    return {
        "schema": SCHEMA,
        "first_rung": True,
        "research_only": True,
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
        "scoped_verdict": (
            "locked frozen-weight closed forms and settled n600 VJP contractions "
            "at scorer_plane_y and camera_input_x only"
        ),
        "counts": counts,
        "environment_receipt_sha256": payload_sha256(environment_receipt),
        "factor_index_sha256": payload_sha256(factor_index),
        "contraction_atlas_sha256": payload_sha256(contraction_atlas),
        "calibration_receipt_sha256": (
            payload_sha256(calibration_receipt) if calibration_receipt is not None else None
        ),
        "calibration_status": (
            calibration_receipt.get("status", "MEASURED") if calibration_receipt is not None else "ABSENT"
        ),
        "amplitude_factors": {
            "count": 0,
            "why": "no amplitude factor produced; no through-R/uint8 survival row exists",
            "through_r_uint8_survival_required_before_nonzero": True,
        },
        "frequency_dead_band": {
            "exact_dead_band_ids": [],
            "admission": "REFUSE_ZERO_BYTE_TRUNCATION",
        },
        "nonadditive_pools": {
            "policy": "overlapping factors are grouped and never arithmetically summed",
            "summed_overlapping_factors": False,
            "pool_count": len(overlapping_pools),
            "rows": overlapping_pools,
        },
        "freshness": {
            "rule": "exact input-hash and version-set equality at consumption",
            "stale_action": "FAIL_CLOSED_REDERIVE_DO_NOT_CONFIRM",
        },
        "directive_consumption": [
            {
                "directive_timestamp": timestamp,
                "status": "CONSUMED",
                "effect": (
                    "first-rung next measurement, nonadditive opportunity pools, "
                    "and counted-inert missing joins remain explicit"
                ),
            }
            for timestamp in DIRECTIVES
        ],
        "storage": {
            "policy": "certify-or-block; large shards stay on SSD",
            "environment_never_deleted": True,
            "reconstruction_commands": list(reconstruction_commands),
        },
        "pointer": "UNCHANGED",
        "triality": {
            "dsl": "materialization CLI stage graph and explicit typed arguments",
            "dag": "locked_env -> inventory -> closed_forms + n600_index -> calibration -> manifest",
            "equations": [
                SEG_BINDING,
                "pose_six_by_six_row_gram_eigenspectrum_v1",
                "ddm_at1x_two_relay_contracted_singular_energy_v1",
            ],
        },
        "main_landing_review_required": True,
    }


__all__ = [
    "CALIBRATION_SCHEMA",
    "CONTRACTION_SCHEMA",
    "ENVIRONMENT_SCHEMA",
    "EVIDENCE_AXIS",
    "OBSERVED_E2",
    "PAIR_COUNT",
    "SCHEMA",
    "SEG_BINDING",
    "SSD_ROOT",
    "AtlasMaterializationError",
    "TensorIndexRow",
    "aggregate_contractions",
    "build_atlas_manifest",
    "build_calibration_blocker_receipt",
    "build_calibration_receipt",
    "build_environment_receipt",
    "canonical_json_bytes",
    "certify_tree",
    "classify_gaze_v19_coverage",
    "contraction_spectrum",
    "derive_network_closed_forms",
    "file_identity",
    "parse_evaluate_report",
    "payload_sha256",
    "require_exact_pair_coverage",
    "require_locked_inventory",
    "require_ssd_environment",
    "score_formula",
    "sha256_file",
    "shared_receipt_contract",
    "storage_preflight",
    "validate_and_contract_sidecars",
    "version_set_sha256",
    "wrap_receipt",
    "write_factor_shards",
    "write_immutable_receipt",
]
