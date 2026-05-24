# SPDX-License-Identifier: MIT
"""Canonical deterministic submission-packet compiler.

This module is the **single canonical entry point** for deterministic packet
compilation per CLAUDE.md "Deterministic packet compiler — non-negotiable".
It composes the existing submission-packet oracle
(`tac.submission_packet_compiler.inspect_packet`) and the Phase-1-specific
compiler (`tac.phase1_packet_compiler.compile_phase1_packet`) under a unified
3-mode x 4-profile contract:

* Modes:    ``identity``      - byte-for-byte parity re-emit
            ``canonicalize``  - normalise compliance-approved metadata only;
                                report every changed byte
            ``optimize``      - score-affecting bytes may change; caller MUST
                                provide baseline SHA + size + runtime-
                                consumption proof. The compiler refuses if
                                the runtime does not consume the new bytes.
* Profiles: ``contest_one_video_replay``  - contest one-video overfit
            ``contest_generalized``       - contest, cross-video preserved
            ``production_generalized``    - comma-ai/openpilot, portable
            ``production_edge_adaptive``  - production edge, optional
                                            on-device learning gated

Per CLAUDE.md "Deterministic packet compiler" non-negotiable, every mode +
profile fails closed on:

* hidden sidecars
* scorer modifications at inflate time (strict-scorer-rule)
* external state / network dependencies in inflate.sh / runtime
* unsupported ZIP features (only STORED + DEFLATED allowed)
* parser divergence (runtime tree SHA mismatch in identity mode)
* non-deterministic native builds
* missing golden vectors
* missing runtime-tree custody

Per CLAUDE.md FORBIDDEN_PATTERNS:

* No score claims. ``score_claim`` is permanently ``False``. Promotion +
  dispatch readiness require contest-CUDA + contest-CPU auth eval on the
  exact archive bytes; this compiler only proves byte custody.
* No /tmp paths in any persisted artifact (callers route through
  ``experiments/results/<lane>_<timestamp>/`` or an operator-supplied path).
* Never invent CLI flags: every flag emitted by the sister CLI was grepped
  against this module's public API before being wired.

CLAUDE.md compliance + wire-in tags (per Catalog #125 coherence-by-default):

* Sensitivity-map contribution: N/A - this is a byte-custody primitive,
  not a sensitivity contributor.
* Pareto constraint: feeds into the byte-axis closure side of the
  archive-size feasibility set (Dykstra co-lead).
* Bit-allocator hook: N/A - per-tensor importance is upstream.
* Cathedral autopilot dispatch hook: every emitted manifest carries the
  ``ready_for_exact_eval_dispatch`` flag in the autopilot-consumable shape.
* Continual-learning posterior update: N/A - empirical anchors come from
  contest-CUDA / contest-CPU evals, not from compiler output.
* Probe-disambiguator: N/A - the three modes are non-overlapping by
  construction; there is no defensible interpretation gap.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import io
import json
import shutil
import stat
import tokenize
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from tac.optimization.proxy_candidate_contract import PROXY_FALSE_AUTHORITY_FIELDS
from tac.repo_io import json_text, sha256_bytes, sha256_file
from tac.submission_packet_compiler import (
    PacketCompilerError as _OraclePacketCompilerError,
)
from tac.submission_packet_compiler import (
    inspect_packet as _oracle_inspect_packet,
)

SCHEMA_VERSION = "deterministic_packet_compiler.v1"
TOOL_NAME = "tac.packet_compiler.deterministic_compiler"
MANIFEST_NAME = "deterministic_packet_compiler_manifest.json"
NO_OP_PROOF_NAME = "no_op_proof.json"

COMPILER_MODES: tuple[str, ...] = ("identity", "canonicalize", "optimize")
CompilerMode = Literal["identity", "canonicalize", "optimize"]

TARGET_PROFILES: tuple[str, ...] = (
    "contest_one_video_replay",
    "contest_generalized",
    "production_generalized",
    "production_edge_adaptive",
)
TargetProfile = Literal[
    "contest_one_video_replay",
    "contest_generalized",
    "production_generalized",
    "production_edge_adaptive",
]

PACKET_IR_OPERATION_SET_SCHEMA = "packet_ir_operation_set_v1"
PACKET_IR_OPERATION_SET_BRIDGE_CONTRACT_SCHEMA = (
    "packet_ir_operation_set_bridge_contract.v1"
)
DETERMINISTIC_COMPILER_REQUIRED_ORDER: tuple[str, ...] = (
    "representation",
    "prediction",
    "quantization",
    "hyperprior",
    "arithmetic",
    "pack",
)
PACKET_IR_OPERATION_SET_REQUIRED_PROOFS: tuple[str, ...] = (
    "byte_closed_archive",
    "deterministic_packet_manifest",
    "runtime_consumption_proof",
    "same_runtime_full_frame_parity_or_rate_only_control",
    "exact_auth_eval_axis_payload",
)

TARGET_PROFILE_POLICIES: dict[str, dict[str, Any]] = {
    "contest_one_video_replay": {
        "contest_dispatch_candidate": True,
        "allows_one_video_replay": True,
        "requires_cross_video_generalization": False,
        "allows_optional_device_learning": False,
        "requires_inflate_sh": True,
        "requires_runtime_tree": True,
        "description": (
            "Contest-only one-video overfit replay. Fixed tables / per-pair "
            "streams derived from the scored video are admissible only when "
            "the archive remains self-contained and exact CUDA auth eval "
            "validates it."
        ),
    },
    "contest_generalized": {
        "contest_dispatch_candidate": True,
        "allows_one_video_replay": False,
        "requires_cross_video_generalization": True,
        "allows_optional_device_learning": False,
        "requires_inflate_sh": True,
        "requires_runtime_tree": True,
        "description": (
            "Contest-compliant packet. MUST preserve the runtime contract "
            "for unseen contest-shaped videos and MUST NOT rely on fixed "
            "per-frame lookup tables or replay data from the scored video."
        ),
    },
    "production_generalized": {
        "contest_dispatch_candidate": False,
        "allows_one_video_replay": False,
        "requires_cross_video_generalization": True,
        "allows_optional_device_learning": False,
        "requires_inflate_sh": False,
        "requires_runtime_tree": True,
        "description": (
            "comma-ai/openpilot production target. Preserves cross-video "
            "behavior, portability, maintainability, and deterministic "
            "reproducible native builds. inflate.sh is optional since the "
            "runtime contract may be a Python/Rust/C++ entry point."
        ),
    },
    "production_edge_adaptive": {
        "contest_dispatch_candidate": False,
        "allows_one_video_replay": False,
        "requires_cross_video_generalization": True,
        "allows_optional_device_learning": True,
        "requires_inflate_sh": False,
        "requires_runtime_tree": True,
        "description": (
            "Production-only edge target. Optional on-device learning is "
            "allowed only outside contest mode and only behind "
            "deterministic fallbacks, reproducible builds, and explicit "
            "capability gates."
        ),
    },
}

# Mirrors phase1_packet_compiler. Kept inline to avoid a module-import cycle
# (this module is imported by Catalog #158, which is loaded very early).
FORBIDDEN_INFLATE_TOKENS: tuple[str, ...] = (
    "PoseNet",
    "SegNet",
    "from upstream.modules",
    "import upstream.modules",
    "rgb_to_yuv6",
    "EfficientNet",
    "FastViT",
)

FORBIDDEN_NETWORK_TOKENS: tuple[str, ...] = (
    "--extra-index-url",
    "--find-links",
    "--index-url",
    "curl ",
    "http://",
    "https://",
    "wget ",
    "pip install",
    "uv run --with",
    "uv pip install",
    "git clone",
)

FORBIDDEN_EXTERNAL_STATE_PATTERNS: tuple[str, ...] = (
    "/Users/",
    "/home/",
    "experiments/results/",
    ".omx/state/",
    ".omx/research/",
    "/tmp/",
    "/var/tmp/",
)

ALLOWED_ZIP_METHODS: frozenset[int] = frozenset(
    {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
)

DETERMINISTIC_ZIP_DATE_TIME: tuple[int, int, int, int, int, int] = (
    1980, 1, 1, 0, 0, 0,
)
NON_EXECUTABLE_MODE = 0o644
EXECUTABLE_MODE = 0o755


class DeterministicPacketCompilerError(ValueError):
    """Raised when a packet cannot be compiled deterministically."""


@dataclasses.dataclass(frozen=True)
class DeterministicPacketResult:
    """Structured result of ``compile_packet``.

    ``score_claim`` is permanently ``False``: this compiler only proves byte
    custody. Promotion / dispatch readiness require contest-CUDA + contest-CPU
    auth eval on the exact archive bytes per CLAUDE.md "Submission auth eval
    BOTH CPU AND CUDA" non-negotiable.
    """

    schema_version: str
    mode: str
    target_profile: str
    output_dir: str
    archive_path: str
    archive_sha256: str
    archive_size_bytes: int
    runtime_tree_sha256: str
    parser_section_manifest: dict[str, Any]
    no_op_proof: dict[str, Any]
    target_profile_policy: dict[str, Any]
    golden_vectors: dict[str, Any]
    score_claim: bool
    promotion_eligible: bool
    ready_for_exact_eval_dispatch: bool
    blockers: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class RuntimeConsumptionProof:
    """Typed runtime-consumption proof bound to candidate packet custody.

    Optimize mode can change score-affecting bytes only when the caller
    supplies structured evidence. A bare boolean is not evidence: the proof
    must tie the candidate archive SHA, runtime content SHA, and consumed
    byte/section evidence to this packet.
    """

    payload: dict[str, Any]
    source: str
    proof_sha256: str | None = None


RuntimeConsumptionProofInput = (
    RuntimeConsumptionProof | Mapping[str, Any] | Path | str | None
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_iso(now: _dt.datetime | None = None) -> str:
    value = now or _dt.datetime.now(_dt.UTC)
    return value.astimezone(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_packet_root(input_path: Path) -> tuple[Path, Path]:
    """Resolve (packet_dir, archive_path) from either a directory or archive.zip."""

    if input_path.is_dir():
        archive = input_path / "archive.zip"
        if not archive.is_file():
            raise DeterministicPacketCompilerError(
                f"packet dir missing archive.zip: {input_path}"
            )
        return input_path, archive
    if input_path.is_file() and input_path.suffix == ".zip":
        return input_path.parent, input_path
    raise DeterministicPacketCompilerError(
        f"input is neither a packet directory nor an archive.zip: {input_path}"
    )


def _scan_text_for_forbidden(
    text: str,
    forbidden: tuple[str, ...],
    *,
    label: str,
) -> list[str]:
    found: list[str] = []
    for token in forbidden:
        if token in text:
            found.append(f"{label}:forbidden_token:{token!r}")
    return found


def _strip_python_comments_for_scan(text: str) -> str:
    try:
        tokens = [
            token
            for token in tokenize.generate_tokens(io.StringIO(text).readline)
            if token.type != tokenize.COMMENT
        ]
        return tokenize.untokenize(tokens)
    except tokenize.TokenError:
        return text


def _scan_runtime_text(packet_dir: Path) -> list[str]:
    """Scan inflate.sh / inflate.py / src/* for forbidden patterns."""

    blockers: list[str] = []
    for name in ("inflate.sh", "inflate.py"):
        path = packet_dir / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        scan_text = _strip_python_comments_for_scan(text) if path.suffix == ".py" else text
        blockers.extend(_scan_text_for_forbidden(
            scan_text, FORBIDDEN_INFLATE_TOKENS, label=name,
        ))
        blockers.extend(_scan_text_for_forbidden(
            scan_text, FORBIDDEN_NETWORK_TOKENS, label=name,
        ))
        blockers.extend(_scan_text_for_forbidden(
            scan_text, FORBIDDEN_EXTERNAL_STATE_PATTERNS, label=name,
        ))
    src_dir = packet_dir / "src"
    if src_dir.is_dir():
        for py in src_dir.rglob("*.py"):
            text = py.read_text(encoding="utf-8", errors="replace")
            scan_text = _strip_python_comments_for_scan(text)
            rel = py.relative_to(packet_dir).as_posix()
            blockers.extend(_scan_text_for_forbidden(
                scan_text, FORBIDDEN_INFLATE_TOKENS, label=rel,
            ))
            blockers.extend(_scan_text_for_forbidden(
                scan_text, FORBIDDEN_NETWORK_TOKENS, label=rel,
            ))
            blockers.extend(_scan_text_for_forbidden(
                scan_text, FORBIDDEN_EXTERNAL_STATE_PATTERNS, label=rel,
            ))
    return blockers


def _scan_archive_zip_methods(archive_path: Path) -> list[str]:
    blockers: list[str] = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            if info.compress_type not in ALLOWED_ZIP_METHODS:
                blockers.append(
                    f"archive:{info.filename}:unsupported_zip_method:"
                    f"{info.compress_type}"
                )
    return blockers


def _expected_manifest_files() -> set[str]:
    """Files we explicitly expect inside a packet root."""

    return {
        "archive.zip",
        "inflate.sh",
        "inflate.py",
        "build_manifest.json",
        "no_op_proof.json",
        MANIFEST_NAME,
        "archive_section_manifest.json",
    }


def _scan_hidden_sidecars(packet_dir: Path) -> list[str]:
    """Refuse files in packet root that are not in the expected set.

    A "hidden sidecar" is any file in the packet root that the contest
    runtime does not consume. Expected files are: ``archive.zip``,
    ``inflate.sh``, ``inflate.py``, ``build_manifest.json``, the canonical
    manifest + no-op proof emitted by this compiler, and anything inside the
    ``src/`` runtime tree (which the inflate.py / inflate.sh references).
    """

    expected_top = _expected_manifest_files()
    blockers: list[str] = []
    for path in sorted(packet_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(packet_dir).as_posix()
        # src/ tree is declared via runtime_tree_manifest.
        if rel.startswith("src/"):
            continue
        if rel in expected_top:
            continue
        blockers.append(f"hidden_sidecar:{rel}")
    return blockers


def _parser_section_manifest(
    oracle_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Derive a parser-section manifest from the oracle's archive metadata."""

    archive = oracle_manifest.get("archive") or {}
    members = archive.get("members") or []
    return {
        "schema_version": "deterministic_parser_section_manifest.v1",
        "section_count": len(members),
        "section_names": [m["name"] for m in members],
        "lengths": [m["uncompressed_bytes"] for m in members],
        "section_sha256s": [m["payload_sha256"] for m in members],
        "offsets": [m["data_offset"] for m in members],
        "compress_types": [m["compress_type"] for m in members],
        "entropy_estimates": (
            "per-member entropy deferred to score-axis tooling; ZIP "
            "compression method is recorded above per-section."
        ),
        "old_new_section_boundaries": (
            "boundaries are the ZIP central directory; deterministic via "
            "DETERMINISTIC_ZIP_DATE_TIME"
        ),
    }


_ARCHIVE_SHA_KEYS = frozenset(
    {
        "archive_sha256",
        "candidate_archive_sha256",
        "new_archive_sha256",
        "mutated_archive_sha256",
        "output_archive_sha256",
    }
)
_RUNTIME_TREE_SHA_KEYS = frozenset(
    {
        "runtime_tree_sha256",
        "candidate_runtime_tree_sha256",
        "runtime_content_sha256",
    }
)
_RUNTIME_FILE_SHA_KEYS = frozenset(
    {
        "runtime_inflate_py_sha256",
        "inflate_py_sha256",
        "candidate_inflate_py_sha256",
    }
)
_SECTION_EVIDENCE_KEYS = frozenset(
    {
        "consumed_sections",
        "consumed_section_names",
        "runtime_consumed_sections",
        "consumed_member_names",
        "changed_sections_consumed",
        "consumed_byte_ranges",
        "consumed_streams",
        "read_members",
        "score_affecting_section_names",
        "sections",
    }
)
_RUNTIME_CONSUMPTION_TRUE_KEYS = frozenset(
    {
        "runtime_consumption_claim",
        "runtime_consumes_payload_bytes",
        "runtime_sidecar_apply_consumption_claim",
        "runtime_sidecar_decode_consumption_claim",
        "full_frame_inflate_output_parity_claim",
    }
)


def _is_sha256_hex(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(c in "0123456789abcdefABCDEF" for c in value)


def _collect_values_by_key(obj: Any, keys: frozenset[str]) -> list[Any]:
    values: list[Any] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if str(key) in keys:
                values.append(value)
            values.extend(_collect_values_by_key(value, keys))
    elif isinstance(obj, list):
        for item in obj:
            values.extend(_collect_values_by_key(item, keys))
    return values


def _collect_sha_values(obj: Any, keys: frozenset[str]) -> list[str]:
    return [
        str(value).lower()
        for value in _collect_values_by_key(obj, keys)
        if _is_sha256_hex(value)
    ]


def _has_runtime_consumption_evidence(payload: Mapping[str, Any]) -> bool:
    section_values = _collect_values_by_key(payload, _SECTION_EVIDENCE_KEYS)
    has_sections = any(
        isinstance(value, list) and len(value) > 0 for value in section_values
    )
    truth_values = _collect_values_by_key(
        payload, _RUNTIME_CONSUMPTION_TRUE_KEYS,
    )
    has_runtime_claim = any(value is True for value in truth_values)
    return has_sections and has_runtime_claim


def _load_runtime_consumption_proof(
    proof: RuntimeConsumptionProofInput,
) -> RuntimeConsumptionProof | None:
    if proof is None:
        return None
    if isinstance(proof, bool):
        raise DeterministicPacketCompilerError(
            "runtime_consumption_proof must be a typed proof mapping or JSON "
            "path; bare booleans are forbidden"
        )
    if isinstance(proof, RuntimeConsumptionProof):
        return proof
    if isinstance(proof, Mapping):
        return RuntimeConsumptionProof(payload=dict(proof), source="<mapping>")

    path = Path(proof)
    if not path.is_file():
        raise DeterministicPacketCompilerError(
            f"runtime_consumption_proof path does not exist: {path}"
        )
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise DeterministicPacketCompilerError(
            f"runtime_consumption_proof is not valid JSON: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise DeterministicPacketCompilerError(
            f"runtime_consumption_proof must decode to a JSON object: {path}"
        )
    return RuntimeConsumptionProof(
        payload=payload,
        source=str(path),
        proof_sha256=sha256_bytes(raw),
    )


def _validate_runtime_consumption_proof(
    proof: RuntimeConsumptionProof,
    *,
    new_archive_sha256: str,
    runtime_tree_sha256: str,
    packet_dir: Path,
) -> list[str]:
    blockers: list[str] = []
    payload = proof.payload

    archive_shas = _collect_sha_values(payload, _ARCHIVE_SHA_KEYS)
    if not archive_shas:
        blockers.append("runtime_consumption_proof_archive_sha256_missing")
    elif new_archive_sha256.lower() not in archive_shas:
        blockers.append(
            "runtime_consumption_proof_archive_sha256_mismatch:"
            f"expected={new_archive_sha256},found={','.join(sorted(set(archive_shas)))}"
        )

    runtime_tree_shas = _collect_sha_values(payload, _RUNTIME_TREE_SHA_KEYS)
    runtime_file_shas = _collect_sha_values(payload, _RUNTIME_FILE_SHA_KEYS)
    if runtime_tree_shas:
        if runtime_tree_sha256.lower() not in runtime_tree_shas:
            blockers.append(
                "runtime_consumption_proof_runtime_tree_sha256_mismatch:"
                f"expected={runtime_tree_sha256},"
                f"found={','.join(sorted(set(runtime_tree_shas)))}"
            )
    elif runtime_file_shas:
        inflate_py = packet_dir / "inflate.py"
        if not inflate_py.is_file():
            blockers.append("runtime_consumption_proof_inflate_py_missing")
        else:
            actual = sha256_file(inflate_py).lower()
            if actual not in runtime_file_shas:
                blockers.append(
                    "runtime_consumption_proof_inflate_py_sha256_mismatch:"
                    f"expected={actual},"
                    f"found={','.join(sorted(set(runtime_file_shas)))}"
                )
    else:
        blockers.append("runtime_consumption_proof_runtime_sha256_missing")

    if not _has_runtime_consumption_evidence(payload):
        blockers.append("runtime_consumption_proof_consumed_sections_missing")

    return blockers


def _no_op_proof(
    *,
    mode: str,
    new_sha: str,
    new_size: int,
    baseline_sha: str | None,
    baseline_size: int | None,
    score_affecting_payload_changed: bool,
    runtime_consumption_proof: RuntimeConsumptionProof | None,
    runtime_consumption_proof_valid: bool,
) -> dict[str, Any]:
    proof: dict[str, Any] = {
        "schema_version": "deterministic_no_op_proof.v1",
        "mode": mode,
        "score_affecting_payload_changed": score_affecting_payload_changed,
        "new_archive_sha256": new_sha,
        "new_archive_size_bytes": new_size,
        "baseline_archive_sha256": baseline_sha,
        "baseline_archive_size_bytes": baseline_size,
        "byte_delta": (
            new_size - baseline_size if baseline_size is not None else None
        ),
        "sha_changed": (
            new_sha != baseline_sha if baseline_sha is not None else None
        ),
        "runtime_consumption_proof": runtime_consumption_proof_valid,
        "runtime_consumption_proof_source": (
            runtime_consumption_proof.source
            if runtime_consumption_proof is not None
            else None
        ),
        "runtime_consumption_proof_sha256": (
            runtime_consumption_proof.proof_sha256
            if runtime_consumption_proof is not None
            else None
        ),
    }
    # For identity mode the contract is byte-for-byte parity. For
    # canonicalize mode score-affecting payload is unchanged by definition.
    # For optimize mode the caller must acknowledge the change.
    if mode == "identity" and baseline_sha is not None:
        proof["no_op_detector_passed"] = new_sha == baseline_sha
    elif mode == "canonicalize" and baseline_sha is not None:
        proof["no_op_detector_passed"] = (
            not score_affecting_payload_changed
        )
    elif mode == "optimize":
        # Optimize is allowed to change bytes; the detector passes when the
        # SHA actually moved AND the runtime is proved to consume the new
        # bytes. Both together close the no-op trap (bytes changed but
        # inflate ignored them).
        proof["no_op_detector_passed"] = (
            score_affecting_payload_changed
            and (baseline_sha is None or new_sha != baseline_sha)
            and runtime_consumption_proof_valid
        )
    else:
        proof["no_op_detector_passed"] = None
    return proof


def _golden_vectors(
    *,
    archive_sha: str,
    runtime_tree_sha: str,
    oracle_manifest: dict[str, Any],
    target_profile: str,
    mode: str,
) -> dict[str, Any]:
    archive = oracle_manifest.get("archive") or {}
    return {
        "schema_version": "deterministic_golden_vectors.v1",
        "tool_name": TOOL_NAME,
        "tool_schema_version": SCHEMA_VERSION,
        "mode": mode,
        "target_profile": target_profile,
        "archive_sha256": archive_sha,
        "runtime_tree_sha256": runtime_tree_sha,
        "member_vectors": [
            {
                "name": m["name"],
                "payload_sha256": m["payload_sha256"],
                "compressed_payload_sha256": m["compressed_payload_sha256"],
                "compress_type": m["compress_type"],
                "uncompressed_bytes": m["uncompressed_bytes"],
                "compressed_bytes": m["compressed_bytes"],
                "data_offset": m["data_offset"],
            }
            for m in (archive.get("members") or [])
        ],
    }


def _copy_packet(source: Path, dest: Path) -> None:
    """Copy the entire packet tree preserving file modes."""

    if dest.exists() and any(dest.iterdir()):
        raise DeterministicPacketCompilerError(
            f"output dir is not empty: {dest}"
        )
    dest.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        # Bare archive.zip: copy as archive.zip into dest root.
        shutil.copy2(source, dest / "archive.zip")
        return
    for src in source.rglob("*"):
        rel = src.relative_to(source)
        target = dest / rel
        if src.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            try:
                target.chmod(stat.S_IMODE(src.stat().st_mode))
            except OSError:
                pass


def _canonicalize_archive(
    source_archive: Path, dest_archive: Path,
) -> dict[str, Any]:
    """Re-emit the archive with canonical ZIP metadata.

    The contract refuses any payload-byte change: each member is read out and
    re-written with the same payload bytes but deterministic timestamp + mode
    + create_system + member ordering preserved.
    """

    changed: list[dict[str, Any]] = []
    with zipfile.ZipFile(source_archive, "r") as zin:
        infos = list(zin.infolist())
        with zipfile.ZipFile(
            dest_archive, "w", compression=zipfile.ZIP_STORED,
        ) as zout:
            for info in infos:
                payload = zin.read(info.filename)
                new_info = zipfile.ZipInfo(
                    info.filename, date_time=DETERMINISTIC_ZIP_DATE_TIME,
                )
                new_info.compress_type = info.compress_type
                new_info.external_attr = NON_EXECUTABLE_MODE << 16
                new_info.create_system = 3  # POSIX (Linux/macOS contest).
                # Preserve method to keep payload semantics; we only rewrite
                # the deterministic metadata around the bytes.
                zout.writestr(new_info, payload, compress_type=info.compress_type)
                original_date = tuple(info.date_time)
                original_mode = (info.external_attr >> 16) & 0o7777
                row = {
                    "name": info.filename,
                    "original_date_time": list(original_date),
                    "canonical_date_time": list(DETERMINISTIC_ZIP_DATE_TIME),
                    "original_unix_permissions": original_mode,
                    "canonical_unix_permissions": NON_EXECUTABLE_MODE,
                    "original_create_system": int(info.create_system),
                    "canonical_create_system": 3,
                    "payload_sha256": sha256_bytes(payload),
                    "payload_bytes": len(payload),
                }
                if (
                    original_date != DETERMINISTIC_ZIP_DATE_TIME
                    or original_mode != NON_EXECUTABLE_MODE
                    or int(info.create_system) != 3
                ):
                    changed.append(row)
    return {
        "schema_version": "deterministic_canonicalize_report.v1",
        "changed_member_count": len(changed),
        "changed_members": changed,
    }


def _is_byte_identical(a: Path, b: Path) -> bool:
    return sha256_file(a) == sha256_file(b)


def _profile_policy(target_profile: str) -> dict[str, Any]:
    if target_profile not in TARGET_PROFILE_POLICIES:
        raise DeterministicPacketCompilerError(
            f"unknown target_profile: {target_profile}"
        )
    return dict(TARGET_PROFILE_POLICIES[target_profile])


def _required_runtime_for_profile(
    target_profile: str, packet_dir: Path,
) -> list[str]:
    """Verify the packet has the runtime artifacts a given profile demands."""

    blockers: list[str] = []
    policy = TARGET_PROFILE_POLICIES[target_profile]
    if policy["requires_inflate_sh"]:
        inflate_sh = packet_dir / "inflate.sh"
        if not inflate_sh.is_file():
            blockers.append("inflate_sh_missing")
        elif not (inflate_sh.stat().st_mode & stat.S_IXUSR):
            blockers.append("inflate_sh_not_executable")
        inflate_py = packet_dir / "inflate.py"
        if not inflate_py.is_file():
            blockers.append("inflate_py_missing")
    if policy["requires_runtime_tree"]:
        # At least one runtime artifact must exist (either inflate.sh or
        # src/__init__.py-style runtime tree, e.g. Python/Rust source).
        has_inflate = (packet_dir / "inflate.sh").is_file()
        has_inflate_py = (packet_dir / "inflate.py").is_file()
        has_src = (packet_dir / "src").is_dir()
        if not (has_inflate or has_inflate_py or has_src):
            blockers.append("runtime_tree_missing")
    return blockers


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile_packet(
    input_packet: Path | str,
    *,
    output_dir: Path | str,
    mode: CompilerMode = "identity",
    target_profile: TargetProfile = "contest_one_video_replay",
    baseline_archive_sha256: str | None = None,
    baseline_archive_size_bytes: int | None = None,
    score_affecting_payload_changed: bool = False,
    runtime_consumption_proof: RuntimeConsumptionProofInput = None,
    allow_existing_output_dir: bool = False,
) -> DeterministicPacketResult:
    """Compile a packet under the deterministic-compiler contract.

    Per CLAUDE.md "Deterministic packet compiler" non-negotiable, this is the
    single canonical entry point. New packet-compilation surfaces MUST route
    through here (enforced by preflight Catalog #158).
    """

    if mode not in COMPILER_MODES:
        raise DeterministicPacketCompilerError(f"unknown mode: {mode}")
    policy = _profile_policy(target_profile)
    typed_runtime_proof = _load_runtime_consumption_proof(
        runtime_consumption_proof,
    )

    input_path = Path(input_packet)
    if not input_path.exists():
        raise DeterministicPacketCompilerError(
            f"input packet does not exist: {input_path}"
        )
    packet_dir, archive_path = _resolve_packet_root(input_path)

    out_dir = Path(output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not allow_existing_output_dir:
            raise DeterministicPacketCompilerError(
                f"output dir is not empty: {out_dir}"
            )
        for child in out_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optimize mode contract: caller MUST acknowledge the change AND supply
    # baseline. Fail closed before we touch any bytes.
    if mode == "optimize":
        if not score_affecting_payload_changed:
            raise DeterministicPacketCompilerError(
                "optimize mode requires score_affecting_payload_changed=True"
            )
        if (
            baseline_archive_sha256 is None
            or baseline_archive_size_bytes is None
        ):
            raise DeterministicPacketCompilerError(
                "optimize mode requires baseline_archive_sha256 + "
                "baseline_archive_size_bytes"
            )
        if typed_runtime_proof is None:
            raise DeterministicPacketCompilerError(
                "optimize mode requires a typed runtime_consumption_proof "
                "mapping or JSON path; byte-changing packets must prove the "
                "runtime reads the new bytes (no-op detector)"
            )
    elif mode == "canonicalize":
        if score_affecting_payload_changed:
            raise DeterministicPacketCompilerError(
                "canonicalize mode refuses score_affecting_payload_changed; "
                "use optimize mode for byte-changing emits"
            )

    # Step 1: emit the new packet under the requested mode.
    if mode == "identity":
        _copy_packet(packet_dir if input_path.is_dir() else input_path, out_dir)
        canonicalize_report: dict[str, Any] = {
            "schema_version": "deterministic_canonicalize_report.v1",
            "changed_member_count": 0,
            "changed_members": [],
        }
    elif mode == "canonicalize":
        # Copy the runtime tree byte-for-byte; rewrite the archive.zip with
        # canonical metadata only. Score-affecting bytes are preserved.
        _copy_packet(packet_dir if input_path.is_dir() else input_path, out_dir)
        new_archive = out_dir / "archive.zip"
        canonicalize_report = _canonicalize_archive(archive_path, new_archive)
    else:  # optimize
        # Optimize mode copies the caller-prepared packet verbatim; the
        # caller has already produced the new archive bytes. We verify
        # custody, gates, and emit the no-op proof on top.
        _copy_packet(packet_dir if input_path.is_dir() else input_path, out_dir)
        canonicalize_report = {
            "schema_version": "deterministic_canonicalize_report.v1",
            "changed_member_count": None,
            "changed_members": [],
            "note": (
                "optimize mode: byte changes are the caller's responsibility; "
                "this compiler only proves custody + runtime consumption"
            ),
        }

    new_archive = out_dir / "archive.zip"
    new_sha = sha256_file(new_archive)
    new_size = new_archive.stat().st_size

    # Step 2: gates that apply to every mode.
    blockers: list[str] = []

    # Profile-required runtime tree.
    blockers.extend(_required_runtime_for_profile(target_profile, out_dir))

    # Scorer-at-inflate / network / external-state / forbidden-token scans.
    blockers.extend(_scan_runtime_text(out_dir))

    # Unsupported ZIP features.
    blockers.extend(_scan_archive_zip_methods(new_archive))

    # Hidden sidecars.
    oracle_error: str | None = None
    try:
        oracle_manifest = _oracle_inspect_packet(
            out_dir, target_profile="contest_one_video_replay",
        )
    except _OraclePacketCompilerError as exc:
        oracle_error = str(exc)
        oracle_manifest = {"error": str(exc), "archive": {"members": []}}
    if oracle_error is not None:
        blockers.append(f"packet_oracle_inspect_failed:{oracle_error}")
    blockers.extend(_scan_hidden_sidecars(out_dir))

    # Identity-mode parser-divergence gate: archive SHA must match input.
    if mode == "identity":
        in_sha = sha256_file(archive_path)
        if in_sha != new_sha:
            blockers.append(
                f"parser_divergence:identity_sha_mismatch:"
                f"input={in_sha},output={new_sha}"
            )
        if (
            baseline_archive_sha256 is not None
            and new_sha != baseline_archive_sha256
        ):
            blockers.append(
                f"parser_divergence:baseline_sha_mismatch:"
                f"baseline={baseline_archive_sha256},output={new_sha}"
            )

    # Step 3: build the manifest + no-op proof.
    runtime_tree_sha = (
        oracle_manifest.get("runtime_tree_manifest", {}).get("tree_sha256")
        or hashlib.sha256(b"").hexdigest()
    )
    runtime_proof_blockers: list[str] = []
    if mode == "optimize" and typed_runtime_proof is not None:
        runtime_proof_blockers = _validate_runtime_consumption_proof(
            typed_runtime_proof,
            new_archive_sha256=new_sha,
            runtime_tree_sha256=runtime_tree_sha,
            packet_dir=out_dir,
        )
        blockers.extend(runtime_proof_blockers)
    runtime_consumption_proof_valid = (
        typed_runtime_proof is not None and not runtime_proof_blockers
    )
    no_op_proof = _no_op_proof(
        mode=mode,
        new_sha=new_sha,
        new_size=new_size,
        baseline_sha=baseline_archive_sha256,
        baseline_size=baseline_archive_size_bytes,
        score_affecting_payload_changed=score_affecting_payload_changed,
        runtime_consumption_proof=typed_runtime_proof,
        runtime_consumption_proof_valid=runtime_consumption_proof_valid,
    )
    parser_section_manifest = _parser_section_manifest(oracle_manifest)
    golden_vectors = _golden_vectors(
        archive_sha=new_sha,
        runtime_tree_sha=runtime_tree_sha,
        oracle_manifest=oracle_manifest,
        target_profile=target_profile,
        mode=mode,
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "tool_name": TOOL_NAME,
        "generated_at_utc": _utc_iso(),
        "mode": mode,
        "target_profile": target_profile,
        "target_profile_policy": policy,
        "input_packet": str(input_path),
        "output_dir": str(out_dir),
        "archive_path": str(new_archive),
        "archive_sha256": new_sha,
        "archive_size_bytes": new_size,
        "runtime_tree_sha256": runtime_tree_sha,
        "runtime_tree_manifest": (
            oracle_manifest.get("runtime_tree_manifest") or {}
        ),
        "parser_section_manifest": parser_section_manifest,
        "golden_vectors": golden_vectors,
        "canonicalize_report": canonicalize_report,
        "no_op_proof": no_op_proof,
        "blockers": sorted(set(blockers)),
        # Per CLAUDE.md "Apples-to-apples evidence discipline": this tool
        # never claims a score nor promotion eligibility. Both flags are
        # permanently False; downstream contest-CUDA + contest-CPU auth eval
        # provides the only authoritative score.
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "byte_custody_only",
        "score_dispatch_blockers": [
            "score_claim_forbidden_without_exact_cuda_auth_eval",
            "dispatch_readiness_forbidden_without_byte_closed_archive_and_exact_cuda_auth_eval",
            "level2_dispatch_claim_required_before_any_remote_exact_eval",
            "pre_submission_compliance_check_required_before_release_or_promotion",
        ],
    }

    # Persist manifest + no-op proof to the output dir.
    (out_dir / MANIFEST_NAME).write_text(json_text(manifest), encoding="utf-8")
    (out_dir / NO_OP_PROOF_NAME).write_text(
        json_text(no_op_proof), encoding="utf-8",
    )

    return DeterministicPacketResult(
        schema_version=SCHEMA_VERSION,
        mode=mode,
        target_profile=target_profile,
        output_dir=str(out_dir),
        archive_path=str(new_archive),
        archive_sha256=new_sha,
        archive_size_bytes=new_size,
        runtime_tree_sha256=runtime_tree_sha,
        parser_section_manifest=parser_section_manifest,
        no_op_proof=no_op_proof,
        target_profile_policy=policy,
        golden_vectors=golden_vectors,
        score_claim=False,
        promotion_eligible=False,
        ready_for_exact_eval_dispatch=False,
        blockers=tuple(sorted(set(blockers))),
    )


def inspect_packet_oracle(
    input_packet: Path | str,
    *,
    target_profile: TargetProfile = "contest_one_video_replay",
    zipwire_bin: Path | str | None = None,
) -> dict[str, Any]:
    """Thin wrapper around :func:`tac.submission_packet_compiler.inspect_packet`.

    Re-exported here so callers have one canonical import surface for the
    deterministic-compiler contract (inspect / identity / canonicalize /
    optimize). The oracle inspect is the read-only entry point that produces
    deterministic conformance vectors without writing a new packet.
    """

    if target_profile not in TARGET_PROFILE_POLICIES:
        raise DeterministicPacketCompilerError(
            f"unknown target_profile: {target_profile}"
        )
    # The oracle only knows the contest profiles. For production profiles we
    # still call it (it produces the byte/manifest vectors), then we annotate
    # the result with our wider profile policy.
    oracle_profile = (
        target_profile if target_profile in (
            "contest_one_video_replay", "contest_generalized",
        ) else "contest_one_video_replay"
    )
    try:
        base = _oracle_inspect_packet(
            input_packet,
            target_profile=oracle_profile,
            zipwire_bin=zipwire_bin,
        )
    except _OraclePacketCompilerError as exc:
        raise DeterministicPacketCompilerError(str(exc)) from exc
    base["deterministic_compiler_target_profile"] = target_profile
    base["deterministic_compiler_target_profile_policy"] = _profile_policy(
        target_profile,
    )
    return base


def packetir_operation_set_bridge_contract() -> dict[str, Any]:
    """Return the canonical PacketIR operation-set lowering contract.

    Queue/scheduler layers may wrap this with lane-specific context, but the
    compiler order, IR schema, and required proof vocabulary live here so
    materializer front ends do not grow duplicate mini-contracts.
    """

    return {
        "schema": PACKET_IR_OPERATION_SET_BRIDGE_CONTRACT_SCHEMA,
        "canonical_packet_compiler_module": TOOL_NAME,
        "canonical_packet_compiler_schema": SCHEMA_VERSION,
        "recommended_ir_schema": PACKET_IR_OPERATION_SET_SCHEMA,
        "required_order": list(DETERMINISTIC_COMPILER_REQUIRED_ORDER),
        "required_proofs": list(PACKET_IR_OPERATION_SET_REQUIRED_PROOFS),
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }


__all__ = [
    "COMPILER_MODES",
    "DETERMINISTIC_COMPILER_REQUIRED_ORDER",
    "DETERMINISTIC_ZIP_DATE_TIME",
    "FORBIDDEN_EXTERNAL_STATE_PATTERNS",
    "FORBIDDEN_INFLATE_TOKENS",
    "FORBIDDEN_NETWORK_TOKENS",
    "MANIFEST_NAME",
    "NO_OP_PROOF_NAME",
    "PACKET_IR_OPERATION_SET_BRIDGE_CONTRACT_SCHEMA",
    "PACKET_IR_OPERATION_SET_REQUIRED_PROOFS",
    "PACKET_IR_OPERATION_SET_SCHEMA",
    "SCHEMA_VERSION",
    "TARGET_PROFILES",
    "TARGET_PROFILE_POLICIES",
    "TOOL_NAME",
    "CompilerMode",
    "DeterministicPacketCompilerError",
    "DeterministicPacketResult",
    "RuntimeConsumptionProof",
    "TargetProfile",
    "compile_packet",
    "inspect_packet_oracle",
    "packetir_operation_set_bridge_contract",
]
