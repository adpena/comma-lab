#!/usr/bin/env python3
"""Run or hash-verify the contained Einstein--Kolmogorov xi bridge.

This wrapper intentionally delegates archive construction, parse-back, inflation,
bit-exact verification, and local scorer replay to
``tools.levelset_byte_close_and_eval.run``.  It adds cross-checkpoint custody and
containment; it does not duplicate the receiver or enable exact evaluation.
Diagnostic execution is capped at 24 pairs.  Full n600 execution is a distinct
typed mode and requires an explicit operator authorization reference.
"""

from __future__ import annotations

import argparse
import ast
import errno
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib import format as npformat

from tac.witness_dsl import einstein_kolmogorov_bridge_20260719 as config_contract
from tac.witness_dsl.einstein_kolmogorov_bridge_20260719 import EinsteinKolmogorovXiBridgeConfig

RECEIPT_SCHEMA = "einstein_kolmogorov_xi_bridge_receipt.v3"
HASH_RECEIPT_SCHEMA = "einstein_kolmogorov_xi_bridge_hash_receipt.v1"
FAILURE_MANIFEST_SCHEMA = "einstein_kolmogorov_xi_bridge_failure.v1"
OPERATOR_AUTHORIZATION_SCHEMA = "einstein_kolmogorov_xi_bridge_operator_authorization.v2"
BACKEND_RESUME_ABI_SCHEMA = "levelset_byte_close_resume_abi_verification.v1"
AUTHORITY = "[macOS-CPU advisory] diagnostic-only"
FULL_AUTHORITY = "[macOS-CPU advisory] governed-full n600 after executable resume verification; NON-PROMOTABLE"
R1_CALIBRATION_RECEIPT = "reports/r1_dxi_238/n600_shipdxi.json"
_COST_TOKEN_RE = re.compile(r"(?:^|[;\s])(?:cost=\$|cost_usd=)([0-9]+(?:\.[0-9]+)?)(?:[;\s]|$)")
_AUTHORIZATION_MAX_WINDOW = timedelta(hours=24)
_AUTHORIZATION_SIGNED_FIELDS = (
    "schema",
    "authorization_id",
    "approver",
    "authorized",
    "operator_directive_verbatim",
    "operator_directive_sha256",
    "issued_utc",
    "expires_utc",
    "lane_id",
    "execution_mode",
    "max_pairs",
    "max_cost_usd",
    "claim",
    "inputs",
    "backend_path",
)
_AUTHORIZATION_KEYS = frozenset((*_AUTHORIZATION_SIGNED_FIELDS, "signature_hex"))
_PLACEHOLDER_DIRECTIVE_TOKENS = frozenset({"todo", "tbd", "placeholder", "operator-go", "approved"})


class BridgeValidationError(ValueError):
    """Raised before any bulky output is created when custody is invalid."""


class PacketOutputFilesystemCustodyError(RuntimeError):
    """The configured SSD parent cannot safely accept a new packet directory."""


class BridgeAuthorizationError(RuntimeError):
    """A governed full run lacks explicit typed operator authorization."""


class BridgeStoragePreflightError(RuntimeError):
    """No canonical SSD tier can safely hold the declared workload."""


class BridgeResumabilityError(RuntimeError):
    """The governed full backend lacks a content-bound resume contract."""


class BridgeBackendExecutionError(RuntimeError):
    """The backend failed after it may have emitted certified partial output."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise BridgeValidationError(f"{label} is not a regular file: {path}")
    measured = _sha256_file(path)
    if measured != expected_sha256:
        raise BridgeValidationError(f"{label} SHA-256 mismatch: expected {expected_sha256}, measured {measured}")
    return {"path": str(path), "sha256": measured, "bytes": path.stat().st_size}


def _storage_roots() -> tuple[Path, ...]:
    """Return the canonical operator waterfall in priority order.

    The typed DSL owns the accepted path boundary so ``tac`` does not acquire a
    dependency on ``comma_lab``.  The order mirrors
    ``comma_lab.operator_storage_waterfall.DEFAULT_WORK_TIER_ORDER``.
    """

    configured = tuple(Path(root).resolve(strict=False) for root in config_contract._SSD_ROOTS)
    production_defaults = (
        Path("/Volumes/VertigoDataTier/pact"),
        Path("/Volumes/APDataStore/pact"),
    )
    # Tests and explicitly isolated callers may replace the typed boundary.  In
    # production, consume the canonical policy helper rather than duplicating it.
    if configured != production_defaults:
        return configured
    try:
        from comma_lab.operator_storage_waterfall import operator_work_tiers
    except ImportError:
        return configured
    return tuple(Path(tier.root).resolve(strict=False) for tier in operator_work_tiers())


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _committed_file_bytes(path: Path) -> bytes:
    """Read exactly the bytes committed at HEAD for a repo-contained path."""

    root = config_contract._REPO_ROOT.resolve(strict=False)
    try:
        relative = path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError as exc:
        raise BridgeAuthorizationError(f"operator authorization is outside repository root: {path}") from exc
    completed = subprocess.run(
        ["git", "-C", str(root), "show", f"HEAD:{relative}"],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise BridgeAuthorizationError(
            f"operator authorization must already be committed at HEAD; git could not read {relative}: {detail}"
        )
    return completed.stdout


def _canonical_operator_authorization_bytes(payload: dict[str, Any]) -> bytes:
    """Return the complete v2 authorization payload covered by Ed25519."""

    missing = [field for field in _AUTHORIZATION_SIGNED_FIELDS if field not in payload]
    if missing:
        raise BridgeAuthorizationError(f"operator authorization lacks signed fields: {missing}")
    signed = {field: payload[field] for field in _AUTHORIZATION_SIGNED_FIELDS}
    return json.dumps(
        signed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _parse_authorization_utc(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise BridgeAuthorizationError(f"operator authorization {field} must be a non-empty ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BridgeAuthorizationError(f"operator authorization {field} is not valid ISO-8601") from exc
    if parsed.tzinfo is None:
        raise BridgeAuthorizationError(f"operator authorization {field} must be timezone-aware")
    return parsed.astimezone(UTC)


def _verify_signed_operator_authorization(
    payload: Any,
    *,
    required_bindings: dict[str, Any],
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Verify identity, signature, scope, directive, and time for one GO.

    A committed memo establishes custody only. Execution authority requires a
    detached Ed25519 signature from an approver registered in the canonical
    Lane-C trust root; the matching private key remains outside the repository.
    """

    if not isinstance(payload, dict):
        raise BridgeAuthorizationError("operator authorization root must be a JSON object")
    if payload.get("schema") != OPERATOR_AUTHORIZATION_SCHEMA:
        raise BridgeAuthorizationError(
            "operator authorization schema v2 with Ed25519 signature is required; "
            "v1/self-authored authorization JSON is non-authorizing"
        )
    if set(payload) != _AUTHORIZATION_KEYS:
        missing = sorted(_AUTHORIZATION_KEYS - set(payload))
        extra = sorted(set(payload) - _AUTHORIZATION_KEYS)
        raise BridgeAuthorizationError(f"operator authorization key set mismatch: missing={missing}, extra={extra}")
    for field in (
        "authorization_id",
        "approver",
        "operator_directive_verbatim",
        "operator_directive_sha256",
        "issued_utc",
        "expires_utc",
        "signature_hex",
    ):
        if (
            not isinstance(payload[field], str)
            or not payload[field].strip()
            or payload[field] != payload[field].strip()
        ):
            raise BridgeAuthorizationError(f"operator authorization lacks canonical non-empty {field}")

    signature_hex = payload["signature_hex"]
    if len(signature_hex) != 128 or set(signature_hex) - set("0123456789abcdef"):
        raise BridgeAuthorizationError("operator authorization signature_hex must be 128 lowercase hex characters")
    try:
        from tac.lane_c_compliance import (
            TrustRootMalformed,
            TrustRootMissing,
            load_trust_root,
            trust_root_path,
        )

        trust_root = load_trust_root(config_contract._REPO_ROOT)
    except (TrustRootMissing, TrustRootMalformed) as exc:
        raise BridgeAuthorizationError(
            "operator authorization trust root is unavailable or malformed; refusing governed execution"
        ) from exc
    approver = payload["approver"]
    public_key = trust_root.get(approver)
    if public_key is None:
        raise BridgeAuthorizationError(
            f"operator authorization approver {approver!r} is not registered in "
            f"{trust_root_path(config_contract._REPO_ROOT)}"
        )
    try:
        public_key.verify(
            bytes.fromhex(signature_hex),
            _canonical_operator_authorization_bytes(payload),
        )
    except Exception as exc:
        raise BridgeAuthorizationError("operator authorization Ed25519 signature is invalid") from exc

    directive = payload["operator_directive_verbatim"]
    directive_digest = hashlib.sha256(directive.encode("utf-8")).hexdigest()
    if directive_digest != payload["operator_directive_sha256"]:
        raise BridgeAuthorizationError("operator authorization directive SHA-256 mismatch")
    if directive.casefold() in _PLACEHOLDER_DIRECTIVE_TOKENS:
        raise BridgeAuthorizationError("operator authorization directive is a placeholder, not verbatim authority")

    issued = _parse_authorization_utc(payload["issued_utc"], field="issued_utc")
    expires = _parse_authorization_utc(payload["expires_utc"], field="expires_utc")
    now = datetime.now(UTC) if now_utc is None else now_utc.astimezone(UTC)
    if expires <= issued or expires - issued > _AUTHORIZATION_MAX_WINDOW:
        raise BridgeAuthorizationError("operator authorization validity window must be positive and at most 24 hours")
    if not issued <= now <= expires:
        raise BridgeAuthorizationError("operator authorization is future-dated or expired")

    if any(payload.get(key) != value for key, value in required_bindings.items()):
        raise BridgeAuthorizationError(
            "operator authorization does not exactly bind lane, mode, n600 cost ceiling, "
            "governed claim, input hashes, and backend"
        )
    return {
        "authorization_id": payload["authorization_id"],
        "approver": approver,
        "operator_directive_sha256": directive_digest,
        "issued_utc": payload["issued_utc"],
        "expires_utc": payload["expires_utc"],
        "signature_hex": signature_hex,
    }


def _require_execution_authorization(
    config: EinsteinKolmogorovXiBridgeConfig,
) -> dict[str, Any]:
    if config.execution_mode != config_contract.GOVERNED_FULL_MODE:
        return {"required": False, "verified": False}
    if config.operator_authorization_path is None or config.operator_authorization_sha256 is None:
        raise BridgeAuthorizationError(
            "governed_full n600 execution refused: missing committed, hash-bound operator authorization"
        )
    path = Path(config.operator_authorization_path)
    auth_root = (config_contract._REPO_ROOT / ".omx" / "research" / "operator_authorizations").resolve(strict=False)
    if not _is_within(path.resolve(strict=False), auth_root) or path.suffix != ".json":
        raise BridgeAuthorizationError(
            "operator authorization must be JSON under canonical .omx/research/operator_authorizations"
        )
    try:
        working_bytes = path.read_bytes()
    except OSError as exc:
        raise BridgeAuthorizationError(f"cannot read operator authorization {path}: {exc}") from exc
    measured = hashlib.sha256(working_bytes).hexdigest()
    if measured != config.operator_authorization_sha256:
        raise BridgeAuthorizationError(
            "operator authorization SHA-256 mismatch: "
            f"expected {config.operator_authorization_sha256}, measured {measured}"
        )
    committed_bytes = _committed_file_bytes(path)
    committed_sha = hashlib.sha256(committed_bytes).hexdigest()
    if committed_bytes != working_bytes or committed_sha != measured:
        raise BridgeAuthorizationError("operator authorization working-tree bytes differ from committed HEAD bytes")
    try:
        payload = json.loads(working_bytes)
    except json.JSONDecodeError as exc:
        raise BridgeAuthorizationError(f"operator authorization is not valid JSON: {exc}") from exc
    required = {
        "schema": OPERATOR_AUTHORIZATION_SCHEMA,
        "authorized": True,
        "lane_id": config_contract.LANE_ID,
        "execution_mode": config_contract.GOVERNED_FULL_MODE,
        "max_pairs": config_contract.EXPECTED_N_PAIRS,
        "max_cost_usd": float(config.declared_max_cost_usd),
        "claim": {
            "instance_job_id": config.governed_claim_job_id,
            "platform": config.governed_claim_platform,
        },
        "inputs": {
            "generator_npz_sha256": config.generator_npz_sha256,
            "donor_r1_npz_sha256": config.donor_r1_npz_sha256,
            "gt_cache_sha256": config.gt_cache_sha256,
        },
        "backend_path": "tools/levelset_byte_close_and_eval.py",
    }
    verified = _verify_signed_operator_authorization(payload, required_bindings=required)
    return {
        "required": True,
        "verified": True,
        "path": str(path),
        "sha256": measured,
        "bytes": len(working_bytes),
        **verified,
        "bindings": required,
    }


def _require_governed_execution_context(
    config: EinsteinKolmogorovXiBridgeConfig,
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Require both the governed child marker and a current canonical claim."""

    if config.execution_mode != config_contract.GOVERNED_FULL_MODE:
        return {"required": False, "verified": False}
    from tac.admission_guard import GOVERNED_MARKER_ENV
    from tools import claim_lane_dispatch

    effective_env = os.environ if env is None else env
    if (effective_env.get(GOVERNED_MARKER_ENV, "") or "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise BridgeAuthorizationError(
            "governed_full n600 execution refused: missing active TAC_GOVERNED_ADMISSION marker; "
            "direct in-process/raw execution is forbidden"
        )
    claims_path = config_contract._REPO_ROOT / ".omx" / "state" / "active_lane_dispatch_claims.md"
    try:
        claims = claim_lane_dispatch._parse_claims(claims_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BridgeAuthorizationError(f"cannot read canonical governed claim ledger: {exc}") from exc
    latest = claim_lane_dispatch._latest_claims_by_job(claims).get(
        (config_contract.LANE_ID, str(config.governed_claim_job_id))
    )
    if latest is None or claim_lane_dispatch._is_terminal(latest.status):
        raise BridgeAuthorizationError("governed_full n600 execution refused: no active canonical lane dispatch claim")
    if latest.platform != config.governed_claim_platform:
        raise BridgeAuthorizationError("governed_full claim platform does not match the hash-bound authorization")
    claim_utc = claim_lane_dispatch._parse_utc(latest.timestamp_utc)
    if claim_utc is None or claim_lane_dispatch._claim_age_hours(claim_lane_dispatch._utc_now(), latest) > 24.0:
        raise BridgeAuthorizationError(
            "governed_full n600 execution refused: canonical claim is malformed or older than 24h"
        )
    cost_match = _COST_TOKEN_RE.search(latest.notes)
    if cost_match is None or float(cost_match.group(1)) != float(config.declared_max_cost_usd):
        raise BridgeAuthorizationError("governed_full claim notes must bind cost=$<usd> equal to declared_max_cost_usd")
    return {
        "required": True,
        "verified": True,
        "marker_env": GOVERNED_MARKER_ENV,
        "claim_ledger_path": str(claims_path),
        "claim": latest.__dict__,
    }


def _require_full_backend_resume_abi(config: EinsteinKolmogorovXiBridgeConfig) -> dict[str, Any]:
    """Execute the live backend's resume verifier; declarative receipts have no authority."""

    if config.execution_mode != config_contract.GOVERNED_FULL_MODE:
        return {"required": False, "verified": False}
    backend_path = config_contract._REPO_ROOT / "tools/levelset_byte_close_and_eval.py"
    try:
        source = backend_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(backend_path))
    except (OSError, SyntaxError) as exc:
        raise BridgeResumabilityError(f"cannot inspect live full backend resume ABI: {exc}") from exc
    exports_verifier = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "verify_resume_abi"
        for node in tree.body
    )
    legacy = {
        "provided": config.backend_resume_receipt_path is not None,
        "accepted_as_authority": False,
        "reason": "caller-authored declarative resume receipts cannot prove executable resume",
    }
    if not exports_verifier:
        raise BridgeResumabilityError(
            "governed_full n600 backend refused: live backend exports no executable "
            "verify_resume_abi; fabricated/declarative resume receipts are non-authorizing"
        )
    spec = importlib.util.spec_from_file_location("_ek_full_backend_resume_verifier", backend_path)
    if spec is None or spec.loader is None:
        raise BridgeResumabilityError("cannot load live full backend resume verifier")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        verification = module.verify_resume_abi()
    except BaseException as exc:
        raise BridgeResumabilityError(f"live backend resume ABI verification failed: {exc}") from exc
    required = {
        "schema": BACKEND_RESUME_ABI_SCHEMA,
        "verified": True,
        "checkpoint_roundtrip_executed": True,
        "resume_roundtrip_executed": True,
        "per_stage_checkpoints_preserved": True,
        "atomic_checkpoint_publish": True,
        "backend_sha256": _sha256_file(backend_path),
    }
    if not isinstance(verification, dict) or any(verification.get(key) != value for key, value in required.items()):
        raise BridgeResumabilityError(
            "live backend resume ABI verifier did not prove executable checkpoint+resume roundtrips"
        )
    return {
        "required": True,
        "verified": True,
        "backend_path": str(backend_path.resolve()),
        "verification": verification,
        "legacy_receipt": legacy,
    }


def _storage_waterfall_preflight(
    config: EinsteinKolmogorovXiBridgeConfig,
) -> dict[str, Any]:
    """Select the first available canonical SSD tier with sufficient capacity."""

    packet_dir = Path(config.packet_output_dir).resolve(strict=False)
    required = config.required_free_bytes
    tiers: list[dict[str, Any]] = []
    selected: Path | None = None
    for priority, root in enumerate(_storage_roots()):
        row: dict[str, Any] = {
            "priority": priority,
            "root": str(root),
            "available": root.is_dir(),
            "free_bytes": None,
            "qualifies": False,
        }
        if root.is_dir():
            try:
                free = int(shutil.disk_usage(root).free)
            except OSError as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            else:
                row["free_bytes"] = free
                row["qualifies"] = free >= required
                if selected is None and row["qualifies"]:
                    selected = root
        tiers.append(row)
    if selected is None:
        raise BridgeStoragePreflightError(
            "storage waterfall refused: no canonical SSD tier has "
            f"required_free_bytes={required}; local/tmp fallback is forbidden"
        )
    if not _is_within(packet_dir, selected):
        raise BridgeStoragePreflightError(
            "storage waterfall refused configured packet_output_dir: "
            f"selected preferred tier {selected}, configured {packet_dir}; "
            "local/tmp fallback and lower-priority bypass are forbidden"
        )
    return {
        "policy": "operator_storage_waterfall.v1",
        "tier_order": [str(root) for root in _storage_roots()],
        "selected_root": str(selected),
        "packet_output_dir": str(packet_dir),
        "derived_minimum_free_bytes": config.derived_minimum_free_bytes,
        "declared_minimum_free_bytes": config.declared_minimum_free_bytes,
        "required_free_bytes": required,
        "workload_pairs": config.max_pairs,
        "tiers": tiers,
        "local_fallback": False,
    }


def _npz_headers(path: Path) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    """Read NPY headers without inflating multi-gigabyte GT cache members."""
    headers: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if not member.endswith(".npy"):
                    continue
                with archive.open(member) as stream:
                    version = npformat.read_magic(stream)
                    if version == (1, 0):
                        shape, _fortran, dtype = npformat.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, _fortran, dtype = npformat.read_array_header_2_0(stream)
                    else:
                        shape, _fortran, dtype = npformat._read_array_header(stream, version)
                headers[Path(member).stem] = (tuple(int(x) for x in shape), np.dtype(dtype))
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise BridgeValidationError(f"invalid npz archive {path}: {exc}") from exc
    return headers


def _expect_shape(
    headers: dict[str, tuple[tuple[int, ...], np.dtype[Any]]],
    key: str,
    shape: tuple[int, ...],
    label: str,
) -> None:
    actual = headers.get(key)
    if actual is None:
        raise BridgeValidationError(f"{label} lacks required array {key!r}")
    if actual[0] != shape:
        raise BridgeValidationError(f"{label} {key!r} shape must be {shape}, got {actual[0]}")


def _validate_generator(path: Path) -> dict[str, Any]:
    headers = _npz_headers(path)
    code = headers.get("code")
    if code is None:
        raise BridgeValidationError("generator checkpoint lacks required array 'code'")
    if len(code[0]) != 2 or code[0][0] != 1200 or code[0][1] <= 0:
        raise BridgeValidationError(f"generator code shape must be (1200, positive_mod_dim), got {code[0]}")
    return {"code_shape": list(code[0]), "code_dtype": str(code[1]), "n_pairs": 600}


def _load_donor_xi(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    headers = _npz_headers(path)
    _expect_shape(headers, "pose_carrier.xi_stored", (600, 6), "donor R1 npz")
    _expect_shape(headers, "pose_carrier.dxi", (600, 6), "donor R1 npz")
    with np.load(path, allow_pickle=False) as donor:
        xi_stored = np.asarray(donor["pose_carrier.xi_stored"], dtype=np.float64)
        dxi = np.asarray(donor["pose_carrier.dxi"], dtype=np.float64)
    if not np.isfinite(xi_stored).all() or not np.isfinite(dxi).all():
        raise BridgeValidationError("donor xi_stored/dxi must contain only finite values")
    xi_effective = xi_stored + dxi
    report = {
        "formula": "pose_carrier.xi_stored + 1.0 * pose_carrier.dxi",
        "dxi_scale": 1.0,
        "shape": [600, 6],
        "dtype_passed_to_run": "float64",
        "xi_stored_sha256": hashlib.sha256(xi_stored.tobytes(order="C")).hexdigest(),
        "dxi_sha256": hashlib.sha256(dxi.tobytes(order="C")).hexdigest(),
        "xi_effective_sha256": hashlib.sha256(xi_effective.tobytes(order="C")).hexdigest(),
        "mean_abs_dxi": float(np.abs(dxi).mean()),
    }
    return xi_effective, report


def _r1_calibration_provenance(
    config: EinsteinKolmogorovXiBridgeConfig,
) -> dict[str, Any]:
    """Bind scalar metadata to the settled R1 byte-close receipt.

    With ``xi_override``, ``s_t`` and ``s_r`` no longer generate xi and are
    provenance-only. ``pitch`` remains computational because the receiver derives
    the ground-plane homography from the shipped xi and pitch.
    """

    path = config_contract._REPO_ROOT / R1_CALIBRATION_RECEIPT
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        measured = payload["pose_carrier"]["calibration"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise BridgeValidationError(f"cannot verify settled R1 calibration receipt {path}: {exc}") from exc
    expected = {
        "s_t": config.pose_carrier_s_t,
        "s_r": config.pose_carrier_s_r,
        "pitch": config.pose_carrier_pitch,
    }
    if measured != expected:
        raise BridgeValidationError(
            f"R1 calibration mismatch: typed={expected}, settled_receipt={measured}; refusing backend"
        )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "calibration": measured,
        "field_semantics_with_xi_override": {
            "s_t": "provenance_only; xi is supplied directly",
            "s_r": "provenance_only; xi is supplied directly",
            "pitch": "computational; ground-plane homography derives H from supplied xi",
        },
    }


def _validate_gt_cache(path: Path) -> dict[str, Any]:
    headers = _npz_headers(path)
    expected = {
        "gt_f0": (600, 874, 1164, 3),
        "gt_f1": (600, 874, 1164, 3),
        "lstars": (600, 384, 512),
        "margins": (600, 384, 512),
        "gt_poses": (600, 6),
    }
    for key, shape in expected.items():
        _expect_shape(headers, key, shape, "GT cache")
    with np.load(path, allow_pickle=False) as cache:
        if "n_pairs" not in cache.files or int(np.asarray(cache["n_pairs"]).item()) != 600:
            raise BridgeValidationError("GT cache n_pairs must be scalar 600")
    return {"n_pairs": 600, "shapes": {key: list(shape) for key, shape in expected.items()}}


def _refuse_existing_outputs(config: EinsteinKolmogorovXiBridgeConfig) -> None:
    packet_dir = Path(config.packet_output_dir)
    receipt = Path(config.result_json_path)
    failure_manifest = Path(str(config.failure_manifest_path))
    if packet_dir.exists():
        raise BridgeValidationError(f"packet_output_dir already exists; refusing overwrite: {packet_dir}")
    if receipt.exists():
        raise BridgeValidationError(f"result_json_path already exists; refusing overwrite: {receipt}")
    if failure_manifest.exists():
        raise BridgeValidationError(f"failure_manifest_path already exists; refusing overwrite: {failure_manifest}")
    if not packet_dir.parent.is_dir():
        raise BridgeValidationError(f"packet output parent does not exist: {packet_dir.parent}")
    if not receipt.parent.is_dir():
        raise BridgeValidationError(f"result JSON parent does not exist: {receipt.parent}")
    if not failure_manifest.parent.is_dir():
        raise BridgeValidationError(f"failure manifest parent does not exist: {failure_manifest.parent}")


def _refuse_existing_hash_receipt(config: EinsteinKolmogorovXiBridgeConfig) -> None:
    receipt = Path(config.result_json_path)
    if receipt.exists():
        raise BridgeValidationError(f"result_json_path already exists; refusing overwrite: {receipt}")
    if not receipt.parent.is_dir():
        raise BridgeValidationError(f"result JSON parent does not exist: {receipt.parent}")


def _preflight_packet_parent_writable(config: EinsteinKolmogorovXiBridgeConfig) -> None:
    """Prove exclusive create/write/fsync/unlink custody before loading checkpoints."""
    parent = Path(config.packet_output_dir).parent
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".einstein_kolmogorov_xi_bridge_preflight.",
            suffix=".tmp",
            dir=parent,
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(b"xi-bridge-custody-preflight\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.unlink()
        temporary = None
    except OSError as exc:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        if exc.errno in {errno.EACCES, errno.EPERM}:
            classification = "permission_denied"
        elif exc.errno == errno.EROFS:
            classification = "read_only_filesystem"
        else:
            classification = "filesystem_io_error"
        raise PacketOutputFilesystemCustodyError(
            "packet-output filesystem custody blocker: "
            f"classification={classification}; parent={parent}; "
            f"errno={exc.errno}; detail={exc.strerror or str(exc)}"
        ) from exc


def _run_levelset(**kwargs: Any) -> dict[str, Any]:
    from tools.levelset_byte_close_and_eval import run

    return run(**kwargs)


def _atomic_json_create_once(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise BridgeValidationError(f"result JSON appeared during run; refusing overwrite: {path}") from exc
        parent_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {
            "kind": "numpy.ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _certify_output_tree(path: Path) -> dict[str, Any]:
    """Hash every emitted byte and derive a stable tree digest without mutation."""

    resolved = path.resolve(strict=False)
    if resolved.is_file():
        return {
            "path": str(resolved),
            "kind": "file",
            "bytes": resolved.stat().st_size,
            "sha256": _sha256_file(resolved),
            "members": [],
        }
    if not resolved.is_dir():
        raise BridgeValidationError(f"cannot certify non-file output path: {resolved}")
    members: list[dict[str, Any]] = []
    tree_digest = hashlib.sha256()
    total_bytes = 0
    for member in sorted(item for item in resolved.rglob("*") if item.is_file()):
        relative = member.relative_to(resolved).as_posix()
        size = member.stat().st_size
        digest = _sha256_file(member)
        total_bytes += size
        members.append({"path": str(member), "relative_path": relative, "bytes": size, "sha256": digest})
        tree_digest.update(relative.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(str(size).encode("ascii"))
        tree_digest.update(b"\0")
        tree_digest.update(digest.encode("ascii"))
        tree_digest.update(b"\n")
    return {
        "path": str(resolved),
        "kind": "directory_tree",
        "bytes": total_bytes,
        "sha256": tree_digest.hexdigest(),
        "members": members,
    }


def _runtime_custody() -> dict[str, Any]:
    source = Path(__file__).resolve()
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "tool_path": str(source),
        "tool_sha256": _sha256_file(source),
    }


def _write_backend_failure_manifest(
    *,
    config: EinsteinKolmogorovXiBridgeConfig,
    exc: BaseException,
    run_kwargs: dict[str, Any],
    source_custody: dict[str, Any],
    storage_preflight: dict[str, Any],
) -> Path:
    packet_dir = Path(config.packet_output_dir)
    artifacts = [_certify_output_tree(packet_dir)] if packet_dir.exists() else []
    manifest_path = Path(str(config.failure_manifest_path))
    payload = {
        "schema": FAILURE_MANIFEST_SCHEMA,
        "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "BACKEND_FAILED_PARTIAL_OUTPUT_PRESERVED",
        "failure": {"type": type(exc).__name__, "message": str(exc)},
        "config": config.to_dict(),
        "config_fingerprint_sha256": config.fingerprint,
        "argv": _json_safe(run_kwargs),
        "source": source_custody,
        "runtime": _runtime_custody(),
        "storage_preflight": storage_preflight,
        "artifacts": artifacts,
        "rebuild_reason": (
            "backend did not complete; retained bytes are partial, rebuildable only from the "
            "recorded config, argv, source hashes, and runtime custody"
        ),
        "cleanup_performed": False,
        "uncertified_bytes_deleted": False,
    }
    _atomic_json_create_once(manifest_path, payload)
    return manifest_path


def execute(config: EinsteinKolmogorovXiBridgeConfig) -> dict[str, Any]:
    """Validate custody, invoke the existing receiver once, and write one receipt."""
    # Full-mode authority, governed admission, and the backend's executable resume
    # proof intentionally precede all output probes and all large input reads.
    authorization = _require_execution_authorization(config)
    governed_context = _require_governed_execution_context(config)
    resume_contract = _require_full_backend_resume_abi(config)
    _refuse_existing_outputs(config)
    storage_preflight = _storage_waterfall_preflight(config)
    _preflight_packet_parent_writable(config)
    generator_path = Path(config.generator_npz_path)
    generator_custody = _require_file(generator_path, config.generator_npz_sha256, "generator checkpoint npz")
    donor_custody = _require_file(Path(config.donor_r1_npz_path), config.donor_r1_npz_sha256, "donor R1 npz")
    gt_custody = _require_file(Path(config.gt_cache_path), config.gt_cache_sha256, "GT cache")
    r1_calibration = _r1_calibration_provenance(config)
    generator_custody["arrays"] = _validate_generator(generator_path)
    xi_effective, donor_math = _load_donor_xi(Path(config.donor_r1_npz_path))
    gt_custody["arrays"] = _validate_gt_cache(Path(config.gt_cache_path))

    run_kwargs: dict[str, Any] = {
        "ckpt_dir": Path(config.generator_checkpoint_dir),
        "npz_name": config.generator_npz_name,
        "max_pairs": config.max_pairs,
        "fold_pose_sidecar": False,
        "pose_sidecar_path": None,
        "gt_cache": config.gt_cache_path,
        "keep_packet": True,
        "packet_dir": Path(config.packet_output_dir),
        "skip_parity": False,
        "so_overrides": {"freq_across": 32.0, "freq_along": 4.0, "tau": 4.0, "iters": 4},
        "lane_render_band": False,
        "lane_band_cfg": None,
        "lane_rd": True,
        "lane_res": False,
        "pose_carrier": True,
        "pose_carrier_cfg": {
            "s_t": config.pose_carrier_s_t,
            "s_r": config.pose_carrier_s_r,
            "pitch": config.pose_carrier_pitch,
            "stride": 1,
            "downscale": 1,
            "mode": "store_nothing",
            "xi_coder": "delta_ar",
            "xi_q_levels": 4096,
        },
        "pose_carrier_xi_override": xi_effective,
        "phase_carrier": False,
        "phase_carrier_cfg": None,
        "dash_phase_carrier": False,
        "dash_phase_carrier_cfg": None,
        "cross_tensor_codec": False,
        "blind_coordinate_fill": False,
        "blind_coordinate_receipt": None,
        "verify_bit_exact": True,
        "bit_exact_pairs": 2,
        "bit_exact_strict": True,
        "run_exact_eval": False,
        "eval_device": "cpu",
        "uncompressed_dir": None,
        "video_names_file": None,
        "eval_timeout": 18000,
    }
    source_custody = {
        "generator": generator_custody,
        "donor_r1": donor_custody,
        "gt_cache": gt_custody,
        "r1_calibration_receipt": r1_calibration,
        "operator_authorization": authorization,
        "governed_execution_context": governed_context,
        "backend_resume_abi": resume_contract,
    }
    try:
        nested_report = _run_levelset(**run_kwargs)
    except BaseException as exc:
        manifest_path = _write_backend_failure_manifest(
            config=config,
            exc=exc,
            run_kwargs=run_kwargs,
            source_custody=source_custody,
            storage_preflight=storage_preflight,
        )
        raise BridgeBackendExecutionError(
            f"bridge backend failed; partial output preserved and certified at {manifest_path}"
        ) from exc
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "authority": (AUTHORITY if config.execution_mode == config_contract.DIAGNOSTIC_MODE else FULL_AUTHORITY),
        "execution_mode": config.execution_mode,
        "diagnostic_only": config.execution_mode == config_contract.DIAGNOSTIC_MODE,
        "promotion_claim": False,
        "cross_checkpoint_coadaptation": False,
        "pointer_moved": False,
        "pointer_status": "UNMOVED",
        "exact_eval_requested": False,
        "config": config.to_dict(),
        "config_fingerprint_sha256": config.fingerprint,
        "exact_bindings": {
            "generator_checkpoint_dir": config.generator_checkpoint_dir,
            "generator_npz_path": config.generator_npz_path,
            "generator_npz_name": config.generator_npz_name,
            "donor_formula": donor_math["formula"],
            "pose_carrier_mode": "store_nothing",
            "pose_carrier_calibration": {
                "s_t": config.pose_carrier_s_t,
                "s_r": config.pose_carrier_s_r,
                "pitch": config.pose_carrier_pitch,
                "source_receipt": R1_CALIBRATION_RECEIPT,
                "source_receipt_sha256": r1_calibration["sha256"],
                "field_semantics_with_xi_override": r1_calibration["field_semantics_with_xi_override"],
            },
            "xi_coder": "delta_ar",
            "xi_q_levels": 4096,
            "keep_packet": True,
            "verify_bit_exact": True,
            "bit_exact_pairs": 2,
            "run_exact_eval": False,
            "max_pairs": config.max_pairs,
        },
        "storage_preflight": storage_preflight,
        "custody": source_custody,
        "donor_math": donor_math,
        "nested_levelset_report": nested_report,
        "verdict_scope": (
            "diagnostic or governed local full-archive cross-check of a generator plus separately "
            "trained R1 xi; historical v2 declarative resume receipts carry no execution authority; "
            "not evidence that the checkpoints co-adapted and not promotion authority"
        ),
    }
    try:
        _atomic_json_create_once(Path(config.result_json_path), receipt)
    except BaseException as exc:
        manifest_path = _write_backend_failure_manifest(
            config=config,
            exc=exc,
            run_kwargs=run_kwargs,
            source_custody=source_custody,
            storage_preflight=storage_preflight,
        )
        raise BridgeBackendExecutionError(
            f"bridge post-backend receipt persistence failed; output preserved and certified at {manifest_path}"
        ) from exc
    return receipt


def verify_input_hashes(
    config: EinsteinKolmogorovXiBridgeConfig,
    *,
    command_argv: list[str],
) -> dict[str, Any]:
    """Read and hash bound inputs, then emit a command-bound durable receipt.

    This mode deliberately does not require full-run authorization, inspect output
    storage, load checkpoint arrays, or import/invoke the heavy receiver backend.
    """

    _refuse_existing_hash_receipt(config)
    custody = {
        "generator": _require_file(
            Path(config.generator_npz_path),
            config.generator_npz_sha256,
            "generator checkpoint npz",
        ),
        "donor_r1": _require_file(Path(config.donor_r1_npz_path), config.donor_r1_npz_sha256, "donor R1 npz"),
        "gt_cache": _require_file(Path(config.gt_cache_path), config.gt_cache_sha256, "GT cache"),
        "r1_calibration_receipt": _r1_calibration_provenance(config),
    }
    command_payload = {
        "argv": list(command_argv),
        "cwd": str(Path.cwd().resolve()),
        "config_fingerprint_sha256": config.fingerprint,
    }
    receipt = {
        "schema": HASH_RECEIPT_SCHEMA,
        "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "verify_input_hashes",
        "backend_started": False,
        "output_storage_probed": False,
        "authority": "read-only input custody; no score authority",
        "config": config.to_dict(),
        "command": command_payload,
        "command_sha256": hashlib.sha256(
            json.dumps(command_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "custody": custody,
        "runtime": _runtime_custody(),
    }
    _atomic_json_create_once(Path(config.result_json_path), receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="absolute typed config JSON path")
    parser.add_argument(
        "--verify-input-hashes",
        action="store_true",
        help="only verify configured input hashes and write a command-bound receipt",
    )
    args = parser.parse_args(argv)
    config = EinsteinKolmogorovXiBridgeConfig.load(args.config)
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    receipt = verify_input_hashes(config, command_argv=effective_argv) if args.verify_input_hashes else execute(config)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
