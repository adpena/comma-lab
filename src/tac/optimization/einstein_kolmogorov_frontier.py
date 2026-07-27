# SPDX-License-Identifier: MIT
"""Strict, fail-closed Einstein--Kolmogorov R-D frontier receipts.

This module deliberately compiles *custody*, not descriptions or proxies.  It
does not invoke an evaluator and cannot make a score claim from an incomplete
receipt.  Its only positive output is a same-artifact, receiver-closed n600
hard-score point with byte and provenance custody.

The runtime program is hash-bound for reproducibility but is not rate-priced:
generic ``inflate.py`` receiver code is free under the contest contract.  Only
video-derived/learned payload bytes inside the submitted archive are counted.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from tac.artifact_quarantine import DEFAULT_QUARANTINED, has_waiver, load_manifest, scan_text
from tac.exact_eval_custody import validate_exact_eval_evidence
from tac.optimization.s4_archive_composer import S4ArchiveError, parse_sections, section_map
from tac.repo_io import ArtifactWriteError, tree_sha256, write_bytes_artifact

SCHEMA = "einstein_kolmogorov_rd_frontier.v1"
TARGET_TOLERANCES = (0.000152, 0.000300, 0.000500, 0.000800)
REPRESENTATION_LEVELS = ("chart", "object", "event", "exception", "pixel")
CURRENT_SCOPE = (
    "current non-quarantined, explicitly supplied candidate receipts in this "
    "worktree and the R3/S4 custody surfaces available on 2026-07-21"
)
R3_DESCRIPTION_BYTES = 216_207
S4_SETTLED_BYTES = 451_191
S4_SETTLED_SHA256 = "d84f2fe053239d1542ba381420e9569d431ed2015e22e60e49ef48f1321696ed"
S4_SETTLED_RUNTIME_SHA256 = "eef055896474b8327baf57ace016c37fe651f4c22534e2442ebc44da8c3f40b0"
S4_SETTLED_STREAM_SHA256 = "01f4581354e108092010399c00dd6889286b87ad0525ea111d6f27f669d683f8"
S4_SETTLED_MEMBER_SHA256 = "595e69d41f96cc1a33ca7b58c0ed386549bfda6389a8176b24d7044d1f55955b"
MAX_RECEIPT_BYTES = 16 * 1024 * 1024
REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_POINTER_SOURCE = "reports/latest.md"
AUTHORITY_AXES = frozenset({"[contest-CPU]", "[contest-CUDA]"})
PARSEBACK_SCHEMA = "einstein_kolmogorov_parseback.v1"
EVALUATION_SCHEMA = "einstein_kolmogorov_official_eval.v1"
RUNTIME_MANIFEST_SCHEMA = "einstein_kolmogorov_runtime_manifest.v1"
GT_MANIFEST_SCHEMA = "einstein_kolmogorov_gt_manifest.v1"
U3_SECTION_RECEIPT_SCHEMA = "einstein_kolmogorov_section_receipt.v1"
EXTERNAL_EXECUTION_SCHEMA = "einstein_kolmogorov_external_execution.v1"
MAX_RUNTIME_TREE_ENTRIES = 4096
MAX_RUNTIME_TREE_FILE_BYTES = 64 * 1024 * 1024


class FrontierRefusal(ValueError):
    """A receipt has insufficient or contradictory authority."""


def canonical_json(value: Any) -> bytes:
    """Encode a custody object in the one receipt-canonical representation."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_json(path: Path) -> Mapping[str, Any]:
    """Load only bounded metadata receipts; dense arrays are never accepted."""

    if not path.is_file():
        raise FrontierRefusal(f"RECEIPT_PATH_MISSING:{path}")
    size = path.stat().st_size
    if size > MAX_RECEIPT_BYTES:
        raise FrontierRefusal(f"RECEIPT_METADATA_TOO_LARGE:{path}:{size}")
    try:
        with path.open("rb") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrontierRefusal(f"RECEIPT_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise FrontierRefusal(f"RECEIPT_ROOT_NOT_OBJECT:{path}")
    return payload


def _validate_file_ref(raw: Any, label: str) -> tuple[Path, int, str]:
    if not isinstance(raw, Mapping) or {"path", "bytes", "sha256"} - set(raw):
        raise FrontierRefusal(f"INVALID_{label}_FILE_REF")
    path = Path(str(raw["path"]))
    size = _nonnegative_int(raw["bytes"], f"{label}_bytes")
    digest = str(raw["sha256"])
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise FrontierRefusal(f"INVALID_{label}_SHA256")
    if not path.is_file():
        raise FrontierRefusal(f"MISSING_{label}_FILE")
    if path.stat().st_size != size or sha256_file(path) != digest:
        raise FrontierRefusal(f"MISMATCH_{label}_FILE_CUSTODY")
    return path, size, digest


def _quarantine_rows() -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = list(DEFAULT_QUARANTINED)
    manifest = load_manifest()
    if manifest is not None:
        known = {str(row.get("id")) for row in rows}
        rows.extend(row for row in manifest["quarantined"] if str(row.get("id")) not in known)
    return rows


def _quarantine_manifest_sha256() -> str:
    return hashlib.sha256(canonical_json(_quarantine_rows())).hexdigest()


def _quarantine_hits(value: Any) -> tuple[str, ...]:
    text = canonical_json(value).decode("utf-8")
    if has_waiver(text):
        return ("QUARANTINE_WAIVER_FORBIDDEN_ON_FRONTIER_INPUT",)
    return tuple(sorted({hit.identifier for hit in scan_text(text)}))


def _receipt_ref_payload(raw: Any, label: str) -> tuple[Mapping[str, Any], Path, str]:
    path, _, digest = _validate_file_ref(raw, label)
    payload = _bounded_json(path)
    hits = _quarantine_hits({"ref": raw, "payload": payload})
    if hits:
        raise FrontierRefusal(f"BLOCKED_QUARANTINE_{label}:{','.join(hits)}")
    return payload, path, digest


def _scan_bounded_text(path: Path, label: str) -> None:
    """Apply the canonical quarantine scanner to bounded referenced text.

    Scanning only a receipt wrapper is insufficient: a clean wrapper can point
    at a quarantined source/config/GT/runtime manifest.  Binary witness streams
    remain hash-bound but are not decoded as text here.
    """

    if path.stat().st_size > MAX_RECEIPT_BYTES:
        raise FrontierRefusal(f"{label.upper()}_TEXT_TOO_LARGE_FOR_QUARANTINE_SCAN")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise FrontierRefusal(f"{label.upper()}_TEXT_UNREADABLE") from exc
    if has_waiver(text):
        raise FrontierRefusal(f"QUARANTINE_WAIVER_FORBIDDEN_IN_{label.upper()}")
    hits = tuple(sorted({hit.identifier for hit in scan_text(text)}))
    if hits:
        raise FrontierRefusal(f"BLOCKED_QUARANTINE_{label.upper()}:{','.join(hits)}")


def _scanned_json_ref(raw: Any, label: str) -> tuple[Mapping[str, Any], Path, str]:
    """Validate, parse, and recursively quarantine-scan a JSON file reference."""

    path, _, digest = _validate_file_ref(raw, label)
    payload = _bounded_json(path)
    _scan_bounded_text(path, label)
    hits = _quarantine_hits({"ref": raw, "payload": payload})
    if hits:
        raise FrontierRefusal(f"BLOCKED_QUARANTINE_{label.upper()}:{','.join(hits)}")
    return payload, path, digest


def _runtime_tree_custody(runtime_manifest: Mapping[str, Any]) -> tuple[Path, str]:
    """Rederive the declared runtime tree from the actual durable directory."""

    root_text = runtime_manifest.get("runtime_root")
    if not isinstance(root_text, str) or not root_text.strip():
        raise FrontierRefusal("BLOCKED_RUNTIME_ROOT_MISSING")
    root = Path(root_text).resolve()
    if not root.is_dir() or root.is_symlink():
        raise FrontierRefusal("BLOCKED_RUNTIME_ROOT_NOT_DURABLE_DIRECTORY")
    for entries, child in enumerate(root.rglob("*"), start=1):
        if entries > MAX_RUNTIME_TREE_ENTRIES:
            raise FrontierRefusal("BLOCKED_RUNTIME_TREE_TOO_MANY_ENTRIES")
        if child.is_file() and not child.is_symlink() and child.stat().st_size > MAX_RUNTIME_TREE_FILE_BYTES:
            raise FrontierRefusal("BLOCKED_RUNTIME_TREE_FILE_TOO_LARGE")
        if not (child.is_file() or child.is_dir() or child.is_symlink()):
            raise FrontierRefusal("BLOCKED_RUNTIME_TREE_SPECIAL_FILE")
    actual = tree_sha256(root)
    declared = str(runtime_manifest.get("runtime_tree_sha256", ""))
    if declared != actual:
        raise FrontierRefusal("BLOCKED_RUNTIME_TREE_SHA_NOT_REDERIVED")
    return root, actual


def _inflated_manifest_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("payload")
    return nested if isinstance(nested, Mapping) else payload


def _validate_realized_output_manifest(
    manifest_path: Path,
    *,
    realized_path: Path,
    realized_bytes: int,
    realized_sha256: str,
    exact_raw_aggregate_sha256: str,
) -> str:
    """Hash every scored raw output and bind the official aggregate to it."""

    payload = _inflated_manifest_payload(_bounded_json(manifest_path))
    if payload.get("schema") != "contest_auth_eval_inflated_output_manifest_v1":
        raise FrontierRefusal("BLOCKED_INFLATED_OUTPUT_MANIFEST_SCHEMA")
    rows = payload.get("files")
    root_text = payload.get("inflated_dir")
    if not isinstance(rows, list) or not rows or not isinstance(root_text, str) or not root_text:
        raise FrontierRefusal("BLOCKED_INFLATED_OUTPUT_MANIFEST_SHAPE")
    if len(rows) > 1024:
        raise FrontierRefusal("BLOCKED_INFLATED_OUTPUT_MANIFEST_TOO_MANY_FILES")
    root = Path(root_text).resolve()
    if not root.is_dir():
        raise FrontierRefusal("BLOCKED_INFLATED_OUTPUT_ROOT_MISSING")
    canonical_rows: list[dict[str, object]] = []
    realized_seen = False
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("exists") is not True:
            raise FrontierRefusal(f"BLOCKED_INFLATED_OUTPUT_ROW_INVALID:{index}")
        relative = str(row.get("relative_path", ""))
        relative_path = Path(relative)
        if not relative or relative_path.is_absolute() or ".." in relative_path.parts:
            raise FrontierRefusal(f"BLOCKED_INFLATED_OUTPUT_RELATIVE_PATH:{index}")
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise FrontierRefusal(f"BLOCKED_INFLATED_OUTPUT_PATH_ESCAPE:{index}") from exc
        size = _nonnegative_int(row.get("bytes"), f"inflated_output_{index}_bytes")
        digest = str(row.get("sha256", ""))
        if not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest:
            raise FrontierRefusal(f"BLOCKED_INFLATED_OUTPUT_FILE_CUSTODY:{index}")
        canonical_rows.append({"relative_path": relative_path.as_posix(), "bytes": size, "sha256": digest})
        if path == realized_path.resolve():
            if size != realized_bytes or digest != realized_sha256:
                raise FrontierRefusal("BLOCKED_REALIZED_STREAM_MANIFEST_MISMATCH")
            realized_seen = True
    if not realized_seen:
        raise FrontierRefusal("BLOCKED_REALIZED_STREAM_NOT_IN_INFLATED_OUTPUT_MANIFEST")
    aggregate = hashlib.sha256(canonical_json({"files": canonical_rows})).hexdigest()
    if payload.get("aggregate_sha256") != aggregate or exact_raw_aggregate_sha256 != aggregate:
        raise FrontierRefusal("BLOCKED_INFLATED_OUTPUT_AGGREGATE_NOT_REDERIVED")
    if payload.get("raw_file_count") != len(canonical_rows):
        raise FrontierRefusal("BLOCKED_INFLATED_OUTPUT_FILE_COUNT_MISMATCH")
    if payload.get("total_bytes") != sum(int(row["bytes"]) for row in canonical_rows):
        raise FrontierRefusal("BLOCKED_INFLATED_OUTPUT_TOTAL_BYTES_MISMATCH")
    return aggregate


def _require_external_execution_attestation(
    raw: Any,
    *,
    artifact: Mapping[str, Any],
    runtime_tree_sha256: str,
    output_manifest_sha256: str,
    output_aggregate_sha256: str,
    evaluator_sha256: str,
    argv: tuple[str, ...],
) -> None:
    """Fail closed until a provider-reconciled attestation verifier exists."""

    if raw is None:
        raise FrontierRefusal("BLOCKED_EXTERNAL_EXECUTION_ATTESTATION_MISSING")
    receipt, _, _ = _receipt_ref_payload(raw, "external_execution_receipt")
    if receipt.get("schema") != EXTERNAL_EXECUTION_SCHEMA:
        raise FrontierRefusal("BLOCKED_EXTERNAL_EXECUTION_ATTESTATION_SCHEMA")
    expected = {
        "archive_sha256": artifact["sha256"],
        "archive_bytes": artifact["bytes"],
        "runtime_tree_sha256": runtime_tree_sha256,
        "output_manifest_sha256": output_manifest_sha256,
        "output_aggregate_sha256": output_aggregate_sha256,
        "evaluator_sha256": evaluator_sha256,
        "argv": list(argv),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise FrontierRefusal("BLOCKED_EXTERNAL_EXECUTION_ATTESTATION_BINDING")
    if (
        receipt.get("score_claim_valid") is not True
        or receipt.get("terminal") is not True
        or str(receipt.get("provider", "")) not in {"modal", "github_actions"}
        or not str(receipt.get("provider_job_id", "")).strip()
        or not str(receipt.get("hardware", "")).strip()
    ):
        raise FrontierRefusal("BLOCKED_EXTERNAL_EXECUTION_ATTESTATION_INCOMPLETE")
    # Local JSON and local ledgers cannot independently prove provider execution.
    # Positive admission remains closed until this call is replaced by a live,
    # provider-reconciled signature/status verifier.
    raise FrontierRefusal("BLOCKED_EXTERNAL_EXECUTION_ATTESTATION_UNVERIFIED")


def _canonical_pointer_from_source(raw: Any) -> tuple[str, Mapping[str, Any]]:
    """Rederive the contest-CPU pointer from the canonical scanner surface.

    ``reports/latest.md`` is the repository's current citation surface.  The
    path is fixed, the bytes are hash-bound into the receipt, and exactly one
    contest-CPU row must parse.  No score literal is embedded in this module.
    """

    if not isinstance(raw, Mapping) or str(raw.get("path")) != CANONICAL_POINTER_SOURCE:
        raise FrontierRefusal("BLOCKED_NONCANONICAL_POINTER_SOURCE")
    path = REPO_ROOT / CANONICAL_POINTER_SOURCE
    expected = {**dict(raw), "path": str(path)}
    _, size, digest = _validate_file_ref(expected, "canonical_pointer_source")
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^\| \*\*`\[contest-CPU Linux x86_64\]`\*\* \| \*\*([0-9]+\.[0-9]+)\*\* \|",
        re.MULTILINE,
    )
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise FrontierRefusal("BLOCKED_CANONICAL_POINTER_ROW_NOT_UNIQUE")
    score_text = matches[0]
    if not re.fullmatch(r"[0-9]+\.[0-9]{10}", score_text):
        raise FrontierRefusal("BLOCKED_CANONICAL_POINTER_PRECISION")
    return (
        f"{score_text} [contest-CPU] UNMOVED",
        {
            "path": CANONICAL_POINTER_SOURCE,
            "bytes": size,
            "sha256": digest,
            "axis": "[contest-CPU Linux x86_64]",
            "score": float(score_text),
        },
    )


def _report_values(path: Path) -> tuple[int, float, float, int]:
    if path.stat().st_size > MAX_RECEIPT_BYTES:
        raise FrontierRefusal("OFFICIAL_REPORT_TOO_LARGE")
    text = path.read_text(encoding="utf-8")
    patterns = {
        "sample_count": r"Evaluation results over\s+([0-9]+)\s+samples",
        "d_pose": r"Average PoseNet Distortion:\s*([0-9.eE+-]+)",
        "d_seg": r"Average SegNet Distortion:\s*([0-9.eE+-]+)",
        "archive_bytes": r"Submission file size:\s*([0-9,]+)\s+bytes",
    }
    values: dict[str, str] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match is None:
            raise FrontierRefusal(f"OFFICIAL_REPORT_MISSING_{key.upper()}")
        values[key] = match.group(1)
    return (
        _nonnegative_int(int(values["sample_count"]), "official_report_sample_count"),
        _finite(float(values["d_seg"]), "official_report_d_seg"),
        _finite(float(values["d_pose"]), "official_report_d_pose"),
        _nonnegative_int(int(values["archive_bytes"].replace(",", "")), "official_report_archive_bytes"),
    )


@dataclass(frozen=True)
class LevelAttribution:
    level: str
    bytes: int
    miss_mass: float

    def __post_init__(self) -> None:
        if self.level not in REPRESENTATION_LEVELS:
            raise FrontierRefusal(f"unknown representation level: {self.level!r}")
        if not isinstance(self.bytes, int) or isinstance(self.bytes, bool) or self.bytes < 0:
            raise FrontierRefusal("level bytes must be a non-negative integer")
        if not math.isfinite(self.miss_mass) or self.miss_mass < 0:
            raise FrontierRefusal("level miss_mass must be finite and non-negative")


@dataclass(frozen=True)
class CandidateCustody:
    candidate_id: str
    artifact_path: str
    artifact_bytes: int
    artifact_sha256: str
    counted_payload_bytes: int
    target_tolerance: float
    container_overhead_bytes: int
    runtime_overhead_bytes: int
    levels: tuple[LevelAttribution, ...]
    receiver_closed: bool | None
    parseback_double_decode_identical: bool | None
    d_seg: float | None
    d_pose: float | None
    score_artifact_sha256: str | None
    sample_count: int | None
    hard_score: bool | None
    realized_stream_hash: str | None
    evidence_axis: str | None
    source_evidence_axis: str | None
    quarantined_identifiers: tuple[str, ...]
    provenance: Mapping[str, Any]
    authority_bundle: Mapping[str, Any]
    raw: Mapping[str, Any]

    @property
    def counted_archive_bytes(self) -> int:
        """The exact, fully counted artifact size used by frontier selection."""

        return self.artifact_bytes


@dataclass(frozen=True)
class AuthorityEvidence:
    """Fields rederived from concrete parse-back and evaluator receipt bytes."""

    d_seg: float
    d_pose: float
    sample_count: int
    archive_bytes: int
    archive_sha256: str
    stream_sha256: str
    evidence_axis: str
    runtime_sha256: str
    runtime_tree_sha256: str
    interpreter_path: str
    interpreter_sha256: str
    interpreter_version: str
    evaluator_sha256: str
    source_sha256: str
    config_sha256: str
    gt_sha256: str
    gt_source_path: str
    gt_source_sha256: str
    seed: int
    argv: tuple[str, ...]
    receipt_paths: tuple[str, ...]


@dataclass(frozen=True)
class ToleranceVerdict:
    tolerance: float
    status: str
    verdict_scope: str
    candidate_id: str | None = None
    counted_archive_bytes: int | None = None
    d_seg: float | None = None
    d_pose: float | None = None
    candidate_sha256: str | None = None
    evidence_axis: str | None = None
    source_evidence_axis: str | None = None
    realized_stream_hash: str | None = None
    hard_score: bool | None = None
    provenance: Mapping[str, Any] | None = None
    blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class U3ReceiverTuplePreflight:
    status: str
    verdict_scope: str
    predicate_table: Mapping[str, bool]
    blockers: tuple[str, ...]
    execution_plan: Mapping[str, Any] | None


@dataclass(frozen=True)
class AggregateReceipt:
    schema: str
    tolerance_rows: tuple[ToleranceVerdict, ...]
    measured_frontier: tuple[Mapping[str, Any], ...]
    dominated_measured_candidates: tuple[Mapping[str, Any], ...]
    candidate_rejections: Mapping[str, tuple[str, ...]]
    candidates: tuple[Mapping[str, Any], ...]
    sibling_arms: Mapping[str, Any]
    quarantine_audit: Mapping[str, Any]
    u3_preflight: U3ReceiverTuplePreflight
    pointer: str
    pointer_source: Mapping[str, Any]
    authority_labels: Mapping[str, Any]
    compiler_source: Mapping[str, Any]
    source_input_manifests: tuple[Mapping[str, Any], ...]
    source_candidate_rows: tuple[Mapping[str, Any], ...]
    source_u3_row: Mapping[str, Any]
    source_sibling_arms: Mapping[str, Any]
    quarantine_manifest_sha256: str
    main_review_required: bool = True
    promotion_eligible: bool = False
    pointer_moved: bool = False
    receipt_sha256: str = ""

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["receipt_sha256"] = ""
        return value

    def with_hash(self) -> AggregateReceipt:
        return replace(self, receipt_sha256=hashlib.sha256(canonical_json(self.payload())).hexdigest())

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise FrontierRefusal(f"MISSING_{context}_{key}")
    return mapping[key]


def _compiler_source_custody() -> Mapping[str, Any]:
    path = Path(__file__).resolve()
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _bool(mapping: Mapping[str, Any], key: str, context: str) -> bool | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if not isinstance(value, bool):
        raise FrontierRefusal(f"INVALID_{context}_{key}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise FrontierRefusal(f"INVALID_{label}")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise FrontierRefusal(f"INVALID_{label}")
    return parsed


def _nonnegative_int(value: Any, label: str) -> int:
    """Accept only JSON integer values for byte and sample-count custody.

    ``int(value)`` is deliberately forbidden here: it turns booleans,
    fractional floats, and numeric strings into claimed measured quantities.
    """

    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FrontierRefusal(f"INVALID_{label}")
    return value


def _argv(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise FrontierRefusal(f"INVALID_{label}_ARGV")
    parsed = tuple(str(item) for item in value)
    if any(not item for item in parsed):
        raise FrontierRefusal(f"INVALID_{label}_ARGV")
    return parsed


def _require_executed_evaluator(argv: tuple[str, ...], evaluator_path: Path) -> None:
    """Bind the Python script position, not an arbitrary unused argv token."""

    if len(argv) < 2 or Path(argv[1]).resolve() != evaluator_path.resolve():
        raise FrontierRefusal("BLOCKED_EXECUTED_EVALUATOR_PATH_MISMATCH")


def _require_exact_eval_command_binding(
    command: Any,
    evaluation_argv: tuple[str, ...],
    evaluator_path: Path,
) -> None:
    """Cross-bind canonical exact-eval command text to the executed argv."""

    if isinstance(command, str):
        try:
            parsed = tuple(shlex.split(command))
        except ValueError as exc:
            raise FrontierRefusal("BLOCKED_EXACT_EVAL_COMMAND_PARSE") from exc
    elif isinstance(command, Sequence) and not isinstance(command, (str, bytes)):
        parsed = tuple(str(item) for item in command)
    else:
        raise FrontierRefusal("BLOCKED_EXACT_EVAL_COMMAND_MISSING")
    if parsed != evaluation_argv:
        raise FrontierRefusal("BLOCKED_EXACT_EVAL_COMMAND_ARGV_MISMATCH")
    _require_executed_evaluator(parsed, evaluator_path)


def _same_artifact(receipt: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    archive = _require(receipt, "archive", label)
    if not isinstance(archive, Mapping):
        raise FrontierRefusal(f"INVALID_{label}_ARCHIVE")
    _, size, digest = _validate_file_ref(archive, f"{label}_archive")
    if size != expected["bytes"] or digest != expected["sha256"] or str(archive["path"]) != str(expected["path"]):
        raise FrontierRefusal(f"BLOCKED_{label.upper()}_SPLIT_ARCHIVE")


def _validate_authority_bundle(raw: Mapping[str, Any], artifact: Mapping[str, Any]) -> AuthorityEvidence:
    """Rederive authority from receipt bytes and the files those receipts bind.

    Candidate booleans and hash strings are cross-checks only.  They cannot
    create authority without the parse-back streams and official report bytes.
    """

    authority = raw.get("authority_bundle")
    if not isinstance(authority, Mapping) or not authority:
        raise FrontierRefusal("BLOCKED_MISSING_AUTHORITY_BUNDLE")
    parseback, parseback_path, _ = _receipt_ref_payload(authority.get("parseback_receipt"), "parseback_receipt")
    evaluation, evaluation_path, _ = _receipt_ref_payload(authority.get("evaluation_receipt"), "evaluation_receipt")
    if parseback.get("schema") != PARSEBACK_SCHEMA:
        raise FrontierRefusal("BLOCKED_PARSEBACK_RECEIPT_SCHEMA")
    if evaluation.get("schema") != EVALUATION_SCHEMA:
        raise FrontierRefusal("BLOCKED_EVALUATION_RECEIPT_SCHEMA")
    _same_artifact(parseback, artifact, "parseback")
    _same_artifact(evaluation, artifact, "evaluation")

    _, first_bytes, first_sha = _validate_file_ref(parseback.get("first_stream"), "parseback_first_stream")
    _, second_bytes, second_sha = _validate_file_ref(parseback.get("second_stream"), "parseback_second_stream")
    if (
        parseback.get("receiver_closed") is not True
        or parseback.get("double_decode_identical") is not True
        or first_bytes != second_bytes
        or first_sha != second_sha
    ):
        raise FrontierRefusal("BLOCKED_PARSEBACK_DOUBLE_DECODE_NOT_BYTE_IDENTICAL")
    runtime_manifest, _, _ = _scanned_json_ref(parseback.get("runtime"), "parseback_runtime")
    if runtime_manifest.get("schema") != RUNTIME_MANIFEST_SCHEMA:
        raise FrontierRefusal("BLOCKED_RUNTIME_MANIFEST_SCHEMA")
    runtime_root, runtime_tree_sha = _runtime_tree_custody(runtime_manifest)
    runtime_entrypoint, _, runtime_entrypoint_sha = _validate_file_ref(
        runtime_manifest.get("runtime_entrypoint"), "runtime_entrypoint"
    )
    try:
        runtime_entrypoint.resolve().relative_to(runtime_root)
    except ValueError as exc:
        raise FrontierRefusal("BLOCKED_RUNTIME_ENTRYPOINT_OUTSIDE_REDERIVED_TREE") from exc
    _scan_bounded_text(runtime_entrypoint, "runtime_entrypoint")
    interpreter_path, _, interpreter_sha = _validate_file_ref(
        runtime_manifest.get("python_executable"), "runtime_python_executable"
    )
    interpreter_version = str(runtime_manifest.get("python_version", "")).strip()
    if not interpreter_version:
        raise FrontierRefusal("BLOCKED_RUNTIME_INTERPRETER_VERSION_MISSING")
    _argv(parseback.get("argv"), "parseback")

    realized_path, realized_bytes, realized_sha = _validate_file_ref(
        evaluation.get("realized_stream"), "evaluation_realized_stream"
    )
    if realized_bytes != first_bytes or realized_sha != first_sha:
        raise FrontierRefusal("BLOCKED_EVALUATION_SPLIT_REALIZED_STREAM")
    report_path, _, _ = _validate_file_ref(evaluation.get("official_report"), "evaluation_official_report")
    evaluator_path, _, evaluator_sha = _validate_file_ref(evaluation.get("evaluator"), "evaluation_evaluator")
    source_manifest, _, source_sha = _scanned_json_ref(evaluation.get("source_manifest"), "evaluation_source_manifest")
    config_manifest, _, config_sha = _scanned_json_ref(evaluation.get("config_manifest"), "evaluation_config_manifest")
    gt_manifest, _, gt_sha = _scanned_json_ref(evaluation.get("gt_manifest"), "evaluation_gt_manifest")
    _scan_bounded_text(report_path, "evaluation_official_report")
    _scan_bounded_text(evaluator_path, "evaluation_evaluator")
    if not source_manifest or not config_manifest:
        raise FrontierRefusal("BLOCKED_EMPTY_SOURCE_OR_CONFIG_MANIFEST")
    if gt_manifest.get("schema") != GT_MANIFEST_SCHEMA or gt_manifest.get("sample_count") != 600:
        raise FrontierRefusal("BLOCKED_GT_MANIFEST_SCHEMA_OR_SAMPLE_COUNT")
    gt_source_path, _, gt_source_sha = _validate_file_ref(gt_manifest.get("gt_source"), "evaluation_gt_source")
    if gt_source_path.stat().st_size <= MAX_RECEIPT_BYTES:
        _scan_bounded_text(gt_source_path, "evaluation_gt_source")
    canonical_evaluator = REPO_ROOT / "upstream" / "evaluate.py"
    if not canonical_evaluator.is_file() or evaluator_sha != sha256_file(canonical_evaluator):
        raise FrontierRefusal("BLOCKED_EVALUATOR_NOT_CANONICAL_UPSTREAM_BYTES")
    report_n, report_seg, report_pose, report_bytes = _report_values(report_path)
    sample_count = _nonnegative_int(evaluation.get("sample_count"), "evaluation_sample_count")
    archive_bytes = _nonnegative_int(artifact["bytes"], "authority_archive_bytes")
    d_seg = _finite(evaluation.get("d_seg"), "evaluation_d_seg")
    d_pose = _finite(evaluation.get("d_pose"), "evaluation_d_pose")
    if (report_n, report_seg, report_pose, report_bytes) != (sample_count, d_seg, d_pose, archive_bytes):
        raise FrontierRefusal("BLOCKED_OFFICIAL_REPORT_RECEIPT_MISMATCH")
    if sample_count != 600 or evaluation.get("hard_score") is not True:
        raise FrontierRefusal("BLOCKED_NON_N600_HARD_SCORE")
    axis = str(evaluation.get("evidence_axis", ""))
    if axis not in AUTHORITY_AXES:
        raise FrontierRefusal("BLOCKED_INVALID_EVIDENCE_AXIS")
    evaluation_argv = _argv(evaluation.get("argv"), "evaluation")
    if Path(evaluation_argv[0]).resolve() != interpreter_path.resolve():
        raise FrontierRefusal("BLOCKED_EVALUATOR_INTERPRETER_PATH_MISMATCH")
    _require_executed_evaluator(evaluation_argv, evaluator_path)
    seed = _nonnegative_int(evaluation.get("seed"), "evaluation_seed")
    exact = evaluation.get("exact_eval_evidence")
    if not isinstance(exact, Mapping):
        raise FrontierRefusal("BLOCKED_MISSING_CANONICAL_EXACT_EVAL_EVIDENCE")
    _require_exact_eval_command_binding(exact.get("auth_eval_command"), evaluation_argv, evaluator_path)
    semantic_axis = "contest_cpu" if axis == "[contest-CPU]" else "contest_cuda"
    validation = validate_exact_eval_evidence(
        exact,
        expected_axis=semantic_axis,
        expected_archive_sha256=str(artifact["sha256"]),
        expected_runtime_tree_sha256=runtime_tree_sha,
        require_artifact_path=True,
        require_hardware=True,
        require_auth_eval_command=True,
        require_log_path=True,
        require_devices=True,
        require_artifact_sha256=True,
        require_inflated_outputs_manifest=True,
        require_raw_output_aggregate_sha256=True,
        artifact_base_dir=evaluation_path.parent,
    )
    if validation.blockers:
        raise FrontierRefusal("BLOCKED_CANONICAL_EXACT_EVAL:" + ",".join(validation.blockers))
    if (
        validation.archive_bytes != archive_bytes
        or validation.n_samples != sample_count
        or validation.seg_dist != d_seg
        or validation.pose_dist != d_pose
    ):
        raise FrontierRefusal("BLOCKED_CANONICAL_EXACT_EVAL_METRIC_MISMATCH")
    exact_log = (evaluation_path.parent / str(exact.get("log_path", ""))).resolve()
    exact_artifact = (evaluation_path.parent / str(exact.get("artifact_path", ""))).resolve()
    if exact_log != report_path.resolve():
        raise FrontierRefusal("BLOCKED_EXACT_EVAL_LOG_NOT_OFFICIAL_REPORT")
    if exact_artifact != Path(str(artifact["path"])).resolve():
        raise FrontierRefusal("BLOCKED_EXACT_EVAL_SPLIT_ARTIFACT_PATH")
    inflated_manifest = (evaluation_path.parent / str(exact.get("inflated_outputs_manifest_path", ""))).resolve()
    _scan_bounded_text(inflated_manifest, "inflated_outputs_manifest")
    inflated_manifest_sha = sha256_file(inflated_manifest)
    expected_manifest_sha = str(
        exact.get("inflated_outputs_manifest_sha256") or exact.get("inflated_output_manifest_sha256") or ""
    )
    if inflated_manifest_sha != expected_manifest_sha:
        raise FrontierRefusal("BLOCKED_INFLATED_OUTPUT_MANIFEST_SHA_MISMATCH")
    output_aggregate_sha = _validate_realized_output_manifest(
        inflated_manifest,
        realized_path=realized_path,
        realized_bytes=realized_bytes,
        realized_sha256=realized_sha,
        exact_raw_aggregate_sha256=str(
            exact.get("raw_output_aggregate_sha256") or exact.get("inflated_output_aggregate_sha256") or ""
        ),
    )
    _require_external_execution_attestation(
        evaluation.get("external_execution_receipt"),
        artifact=artifact,
        runtime_tree_sha256=runtime_tree_sha,
        output_manifest_sha256=inflated_manifest_sha,
        output_aggregate_sha256=output_aggregate_sha,
        evaluator_sha256=evaluator_sha,
        argv=evaluation_argv,
    )
    return AuthorityEvidence(
        d_seg=d_seg,
        d_pose=d_pose,
        sample_count=sample_count,
        archive_bytes=archive_bytes,
        archive_sha256=str(artifact["sha256"]),
        stream_sha256=realized_sha,
        evidence_axis=axis,
        runtime_sha256=runtime_entrypoint_sha,
        runtime_tree_sha256=runtime_tree_sha,
        interpreter_path=str(interpreter_path),
        interpreter_sha256=interpreter_sha,
        interpreter_version=interpreter_version,
        evaluator_sha256=evaluator_sha,
        source_sha256=source_sha,
        config_sha256=config_sha,
        gt_sha256=gt_sha,
        gt_source_path=str(gt_source_path),
        gt_source_sha256=gt_source_sha,
        seed=seed,
        argv=evaluation_argv,
        receipt_paths=(str(parseback_path), str(evaluation_path)),
    )


def _extract_artifact(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    artifact = raw.get("artifact", raw)
    if not isinstance(artifact, Mapping):
        raise FrontierRefusal("INVALID_artifact")
    return artifact


def candidate_from_mapping(raw: Mapping[str, Any]) -> CandidateCustody:
    """Parse one external receipt without inventing missing custody fields."""

    if not isinstance(raw, Mapping):
        raise FrontierRefusal("INVALID_candidate")
    artifact = _extract_artifact(raw)
    candidate_id = str(_require(raw, "candidate_id", "candidate"))
    levels_raw = _require(raw, "levels", "candidate")
    if not isinstance(levels_raw, Sequence) or isinstance(levels_raw, (str, bytes)):
        raise FrontierRefusal("INVALID_candidate_levels")
    levels = tuple(
        LevelAttribution(
            level=str(_require(row, "level", "level")),
            bytes=_nonnegative_int(_require(row, "bytes", "level"), "level_bytes"),
            miss_mass=_finite(_require(row, "miss_mass", "level"), "level_miss_mass"),
        )
        for row in levels_raw
        if isinstance(row, Mapping)
    )
    if len(levels) != len(levels_raw) or len({row.level for row in levels}) != len(levels):
        raise FrontierRefusal("INVALID_candidate_levels")
    provenance = raw.get("provenance")
    if not isinstance(provenance, Mapping):
        provenance = {}
    score = raw.get("score")
    if score is not None and not isinstance(score, Mapping):
        raise FrontierRefusal("INVALID_score")
    score = {} if score is None else score
    quarantined = raw.get("quarantined_identifiers", ())
    if not isinstance(quarantined, Sequence) or isinstance(quarantined, (str, bytes)):
        raise FrontierRefusal("INVALID_quarantined_identifiers")
    authority_bundle = raw.get("authority_bundle")
    if authority_bundle is not None and not isinstance(authority_bundle, Mapping):
        raise FrontierRefusal("INVALID_authority_bundle")
    return CandidateCustody(
        candidate_id=candidate_id,
        artifact_path=str(_require(artifact, "path", "artifact")),
        artifact_bytes=_nonnegative_int(_require(artifact, "bytes", "artifact"), "artifact_bytes"),
        artifact_sha256=str(_require(artifact, "sha256", "artifact")),
        counted_payload_bytes=_nonnegative_int(
            _require(raw, "counted_payload_bytes", "candidate"), "candidate_counted_payload_bytes"
        ),
        target_tolerance=_finite(_require(raw, "target_tolerance", "candidate"), "target_tolerance"),
        container_overhead_bytes=_nonnegative_int(
            _require(raw, "container_overhead_bytes", "candidate"), "candidate_container_overhead_bytes"
        ),
        runtime_overhead_bytes=_nonnegative_int(
            _require(raw, "runtime_overhead_bytes", "candidate"), "candidate_runtime_overhead_bytes"
        ),
        levels=levels,
        receiver_closed=_bool(raw, "receiver_closed", "candidate"),
        parseback_double_decode_identical=_bool(raw, "parseback_double_decode_identical", "candidate"),
        d_seg=None if "d_seg" not in score else _finite(score["d_seg"], "score_d_seg"),
        d_pose=None if "d_pose" not in score else _finite(score["d_pose"], "score_d_pose"),
        score_artifact_sha256=None if "artifact_sha256" not in score else str(score["artifact_sha256"]),
        sample_count=None
        if "sample_count" not in score
        else _nonnegative_int(score["sample_count"], "score_sample_count"),
        hard_score=_bool(score, "hard_score", "score"),
        realized_stream_hash=None if "realized_stream_hash" not in score else str(score["realized_stream_hash"]),
        evidence_axis=None if "evidence_axis" not in raw else str(raw["evidence_axis"]),
        source_evidence_axis=None if "source_evidence_axis" not in raw else str(raw["source_evidence_axis"]),
        quarantined_identifiers=tuple(str(item) for item in quarantined),
        provenance=provenance,
        authority_bundle={} if authority_bundle is None else authority_bundle,
        raw=raw,
    )


def validate_candidate(candidate: CandidateCustody) -> tuple[str, ...]:
    """Return every concrete refusal reason; an empty tuple admits custody."""

    blockers: list[str] = []
    path = Path(candidate.artifact_path)
    if not path.is_file():
        blockers.append("BLOCKED_ARTIFACT_PATH_MISSING")
    elif path.stat().st_size != candidate.artifact_bytes or sha256_file(path) != candidate.artifact_sha256:
        blockers.append("BLOCKED_ARTIFACT_HASH_MISMATCH")
    else:
        try:
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                unsafe = any(
                    info.is_dir() or Path(info.filename).is_absolute() or ".." in Path(info.filename).parts
                    for info in infos
                )
                if not infos or unsafe or archive.testzip() is not None:
                    raise zipfile.BadZipFile("unsafe, empty, or corrupt archive")
        except zipfile.BadZipFile:
            blockers.append("BLOCKED_ARCHIVE_PARSEBACK_FAILED")
    if candidate.counted_payload_bytes < 0:
        blockers.append("BLOCKED_NEGATIVE_COUNTED_BYTES")
    if candidate.target_tolerance not in TARGET_TOLERANCES:
        blockers.append("BLOCKED_INVALID_TARGET_TOLERANCE")
    if candidate.container_overhead_bytes < 0 or candidate.runtime_overhead_bytes < 0:
        blockers.append("BLOCKED_NEGATIVE_OVERHEAD_BYTES")
    if candidate.runtime_overhead_bytes != 0:
        blockers.append("BLOCKED_VIDEO_AGNOSTIC_RUNTIME_MUST_BE_FREE")
    if sum(level.bytes for level in candidate.levels) != candidate.counted_payload_bytes:
        blockers.append("BLOCKED_LEVEL_BYTE_MISMATCH_OR_UNCLASSIFIED_BYTES")
    if candidate.counted_payload_bytes + candidate.container_overhead_bytes != candidate.counted_archive_bytes:
        blockers.append("BLOCKED_FULL_COUNTED_ARCHIVE_BYTE_MISMATCH")
    if candidate.receiver_closed is None:
        blockers.append("BLOCKED_MISSING_RECEIVER_CLOSED")
    elif not candidate.receiver_closed:
        blockers.append("BLOCKED_RECEIVER_NOT_CLOSED")
    if candidate.parseback_double_decode_identical is None:
        blockers.append("BLOCKED_MISSING_PARSEBACK_DOUBLE_DECODE")
    elif not candidate.parseback_double_decode_identical:
        blockers.append("BLOCKED_PARSEBACK_DOUBLE_DECODE_FAILED")
    if candidate.d_seg is None or candidate.d_pose is None:
        blockers.append("BLOCKED_MISSING_SAME_ARTIFACT_SCORE")
    if candidate.score_artifact_sha256 != candidate.artifact_sha256:
        blockers.append("BLOCKED_SPLIT_ARTIFACT_SCORE")
    if candidate.sample_count != 600:
        blockers.append("BLOCKED_NON_N600_HARD_SCORE")
    if candidate.hard_score is None:
        blockers.append("BLOCKED_MISSING_HARD_SCORE_CUSTODY")
    elif not candidate.hard_score:
        blockers.append("BLOCKED_PROXY_SCORE")
    if not candidate.realized_stream_hash:
        blockers.append("BLOCKED_MISSING_REALIZED_STREAM_HASH")
    if not candidate.evidence_axis:
        blockers.append("BLOCKED_MISSING_EVIDENCE_AXIS")
    elif candidate.evidence_axis not in AUTHORITY_AXES:
        blockers.append("BLOCKED_INVALID_EVIDENCE_AXIS")
    if not candidate.source_evidence_axis:
        blockers.append("BLOCKED_MISSING_SOURCE_EVIDENCE_AXIS")
    elif candidate.evidence_axis != candidate.source_evidence_axis:
        blockers.append("BLOCKED_EVIDENCE_AXIS_PROMOTION")
    if candidate.quarantined_identifiers:
        blockers.append("BLOCKED_QUARANTINED_IDENTIFIER")
    quarantine_hits = _quarantine_hits(candidate.raw)
    if quarantine_hits:
        blockers.append("BLOCKED_CANONICAL_QUARANTINE:" + ",".join(quarantine_hits))

    artifact = {
        "path": candidate.artifact_path,
        "bytes": candidate.artifact_bytes,
        "sha256": candidate.artifact_sha256,
    }
    try:
        authority = _validate_authority_bundle(candidate.raw, artifact)
    except FrontierRefusal as exc:
        blockers.append(str(exc))
        authority = None
    if authority is not None:
        if candidate.receiver_closed is not True or candidate.parseback_double_decode_identical is not True:
            blockers.append("BLOCKED_CANDIDATE_DISAGREES_WITH_PARSEBACK_RECEIPT")
        if (
            candidate.d_seg != authority.d_seg
            or candidate.d_pose != authority.d_pose
            or candidate.sample_count != authority.sample_count
            or candidate.score_artifact_sha256 != authority.archive_sha256
            or candidate.realized_stream_hash != authority.stream_sha256
            or candidate.hard_score is not True
        ):
            blockers.append("BLOCKED_CANDIDATE_DISAGREES_WITH_EVALUATION_RECEIPT")
        if (
            candidate.evidence_axis != authority.evidence_axis
            or candidate.source_evidence_axis != authority.evidence_axis
        ):
            blockers.append("BLOCKED_EVIDENCE_AXIS_PROMOTION")
        expected_provenance = {
            "source_hash": authority.source_sha256,
            "runtime_hash": authority.runtime_sha256,
            "evaluator_hash": authority.evaluator_sha256,
            "config_hash": authority.config_sha256,
            "seed": authority.seed,
            "gt_hash": authority.gt_sha256,
            "argv": list(authority.argv),
        }
        if dict(candidate.provenance) != expected_provenance:
            blockers.append("BLOCKED_PROVENANCE_NOT_REDERIVED_FROM_AUTHORITY_BYTES")
    return tuple(dict.fromkeys(blockers))


def _validate_u3_section_receipt(
    raw: Any,
    *,
    label: str,
    section_name: str,
    section_payload: bytes,
    archive_bytes: int,
    archive_sha256: str,
    member_sha256: str,
    build_receipt_sha256: str,
) -> None:
    """Require a producer receipt semantically bound to one archive section."""

    receipt, _, _ = _receipt_ref_payload(raw, label)
    if receipt.get("schema") != U3_SECTION_RECEIPT_SCHEMA:
        raise FrontierRefusal(f"BLOCKED_{label.upper()}_SCHEMA")
    archive = receipt.get("archive")
    if not isinstance(archive, Mapping) or (
        archive.get("bytes") != archive_bytes
        or archive.get("sha256") != archive_sha256
        or archive.get("member_sha256") != member_sha256
    ):
        raise FrontierRefusal(f"BLOCKED_{label.upper()}_ARCHIVE_BINDING")
    section = receipt.get("section")
    section_sha = hashlib.sha256(section_payload).hexdigest()
    if not isinstance(section, Mapping) or (
        section.get("name") != section_name
        or section.get("bytes") != len(section_payload)
        or section.get("sha256") != section_sha
    ):
        raise FrontierRefusal(f"BLOCKED_{label.upper()}_SECTION_BINDING")
    source_path, source_bytes, source_sha = _validate_file_ref(receipt.get("source_file"), f"{label}_source_file")
    if source_bytes != len(section_payload) or source_sha != section_sha or source_path.read_bytes() != section_payload:
        raise FrontierRefusal(f"BLOCKED_{label.upper()}_SOURCE_BYTES_BINDING")
    if receipt.get("build_receipt_sha256") != build_receipt_sha256:
        raise FrontierRefusal(f"BLOCKED_{label.upper()}_BUILD_RECEIPT_BINDING")
    source_commit = str(receipt.get("source_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40,64}", source_commit):
        raise FrontierRefusal(f"BLOCKED_{label.upper()}_SOURCE_COMMIT")
    _argv(receipt.get("argv"), label)


def preflight_u3(raw: Mapping[str, Any]) -> U3ReceiverTuplePreflight:
    """Validate the receiver tuple from archive bytes and bound receipts."""

    claimed = raw.get("counted_bytes")
    description = raw.get("description_only") is True or claimed == R3_DESCRIPTION_BYTES
    if description:
        return U3ReceiverTuplePreflight(
            status="BLOCKED_DESCRIPTION_ROW_NOT_RECEIVER_TUPLE",
            verdict_scope=CURRENT_SCOPE,
            predicate_table={"description_only_r3": True},
            blockers=(
                "BLOCKED_DESCRIPTION_ROW_NOT_RECEIVER_TUPLE",
                "R3_216207_EXCLUDES_CONCRETE_SEED_EVENT_SEMANTICS",
                "R3_HAS_NO_RECEIVER_CLOSED_STREAM_HASH_OR_MEASURED_POSE",
                f"RERUNNING_SETTLED_{S4_SETTLED_BYTES}_BYTE_S4_ARCHIVE_DOES_NOT_SATISFY_U3",
            ),
            execution_plan=None,
        )

    predicates: dict[str, bool] = {}
    blockers: list[str] = []
    archive = raw.get("archive")
    archive_path: Path | None = None
    archive_bytes: int | None = None
    archive_sha = ""
    member_sha = ""
    sections: Mapping[str, Any] = {}
    manifest: Mapping[str, Any] = {}
    try:
        archive_path, archive_bytes, archive_sha = _validate_file_ref(archive, "u3_archive")
        predicates["concrete_archive_artifact"] = True
        with zipfile.ZipFile(archive_path) as bundle:
            infos = bundle.infolist()
            if len(infos) != 1 or infos[0].filename != "0.bin" or infos[0].is_dir():
                raise FrontierRefusal("U3_ARCHIVE_NOT_EXACT_ONE_MEMBER_0_BIN")
            member = bundle.read(infos[0])
        member_sha = hashlib.sha256(member).hexdigest()
        parsed = parse_sections(member)
        if tuple(row.name for row in parsed) != (
            "manifest.json",
            "seed.ppcs",
            "base.pbase3",
            "causal.pcr3",
            "events.pce3",
            "components.pcomp3",
        ):
            raise FrontierRefusal("U3_SECTION_ORDER_MISMATCH")
        sections = section_map(member)
        loaded_manifest = json.loads(sections["manifest.json"].payload.decode("ascii"))
        if not isinstance(loaded_manifest, Mapping):
            raise FrontierRefusal("U3_MANIFEST_NOT_OBJECT")
        manifest = loaded_manifest
        runtime_node = manifest.get("runtime")
        if not isinstance(runtime_node, Mapping) or runtime_node.get("pair_count") != 600:
            raise FrontierRefusal("U3_MANIFEST_NOT_N600")
        quarantine_hits = _quarantine_hits(manifest)
        if quarantine_hits:
            raise FrontierRefusal("U3_MANIFEST_QUARANTINED:" + ",".join(quarantine_hits))
        predicates["archive_parseback_canonical"] = True
    except (
        AttributeError,
        FrontierRefusal,
        S4ArchiveError,
        zipfile.BadZipFile,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        predicates.setdefault("concrete_archive_artifact", False)
        predicates["archive_parseback_canonical"] = False
        blockers.append(f"BLOCKED_U3_ARCHIVE_PARSEBACK:{exc}")

    files = raw.get("files")
    files = files if isinstance(files, Mapping) else {}
    for name in ("seed.ppcs", "base.pbase3", "events.pce3", "components.pcomp3"):
        key = f"concrete_{name}"
        try:
            _, size, digest = _validate_file_ref(files.get(name), f"u3_{name}")
            section = sections[name]
            predicates[key] = size == len(section.payload) and digest == hashlib.sha256(section.payload).hexdigest()
        except (FrontierRefusal, KeyError):
            predicates[key] = False
        if not predicates[key]:
            blockers.append(f"BLOCKED_U3_{key.upper()}")

    build_receipt_sha = ""
    try:
        build_receipt, _, build_receipt_sha = _receipt_ref_payload(raw.get("build_receipt"), "u3_build_receipt")
        build_archive = build_receipt.get("archive")
        predicates["bound_build_receipt"] = bool(
            build_receipt.get("schema") == "s4_archive_build_receipt.v1"
            and isinstance(build_archive, Mapping)
            and build_archive.get("sha256") == archive_sha
            and build_archive.get("bytes") == archive_bytes
            and build_archive.get("member_sha256") == member_sha
        )
    except FrontierRefusal:
        predicates["bound_build_receipt"] = False
    if not predicates["bound_build_receipt"]:
        blockers.append("BLOCKED_U3_BOUND_BUILD_RECEIPT")

    for receipt_key, section_name in (
        ("base_receipt", "base.pbase3"),
        ("component_receipt", "components.pcomp3"),
    ):
        try:
            section = sections[section_name]
            _validate_u3_section_receipt(
                raw.get(receipt_key),
                label=f"u3_{receipt_key}",
                section_name=section_name,
                section_payload=section.payload,
                archive_bytes=_nonnegative_int(archive_bytes, "u3_archive_bytes"),
                archive_sha256=archive_sha,
                member_sha256=member_sha,
                build_receipt_sha256=build_receipt_sha,
            )
            predicates[f"bound_{receipt_key}"] = True
        except (FrontierRefusal, KeyError):
            predicates[f"bound_{receipt_key}"] = False
        if not predicates[f"bound_{receipt_key}"]:
            blockers.append(f"BLOCKED_U3_BOUND_{receipt_key.upper()}")

    authority: AuthorityEvidence | None = None
    if isinstance(archive, Mapping):
        try:
            authority = _validate_authority_bundle(raw, archive)
            predicates["authority_receipts_rederived"] = True
        except FrontierRefusal as exc:
            predicates["authority_receipts_rederived"] = False
            blockers.append(f"BLOCKED_U3_AUTHORITY:{exc}")
    else:
        predicates["authority_receipts_rederived"] = False
        blockers.append("BLOCKED_U3_AUTHORITY:MISSING_ARCHIVE")

    try:
        payload = _nonnegative_int(raw.get("counted_payload_bytes"), "u3_counted_payload_bytes")
        container = _nonnegative_int(raw.get("container_overhead_bytes"), "u3_container_overhead_bytes")
        runtime = _nonnegative_int(raw.get("runtime_overhead_bytes"), "u3_runtime_overhead_bytes")
        predicates["video_agnostic_runtime_is_free"] = runtime == 0
        predicates["archive_byte_accounting_closes"] = (
            archive_bytes is not None
            and _nonnegative_int(claimed, "u3_counted_bytes") == archive_bytes
            and payload + container == archive_bytes
        )
    except FrontierRefusal:
        predicates["video_agnostic_runtime_is_free"] = False
        predicates["archive_byte_accounting_closes"] = False
    if not predicates["archive_byte_accounting_closes"]:
        blockers.append("BLOCKED_U3_ARCHIVE_BYTE_ACCOUNTING_CLOSES")

    settled = (
        archive_sha == S4_SETTLED_SHA256
        and archive_bytes == S4_SETTLED_BYTES
        and member_sha == S4_SETTLED_MEMBER_SHA256
        and raw.get("runtime_hash") == S4_SETTLED_RUNTIME_SHA256
        and raw.get("realized_stream_hash") == S4_SETTLED_STREAM_SHA256
    )
    predicates["not_settled_s4_identity"] = not settled
    if settled:
        blockers.append("SETTLED_S4_REUSE_RECEIPT_ONLY_NO_RERUN")

    if authority is not None:
        try:
            interpreter_path, _, interpreter_sha = _validate_file_ref(raw.get("interpreter"), "u3_interpreter")
            predicates["interpreter_path_hash_bound"] = (
                str(interpreter_path) == authority.interpreter_path and interpreter_sha == authority.interpreter_sha256
            )
        except FrontierRefusal:
            predicates["interpreter_path_hash_bound"] = False
        predicates["interpreter_version_bound"] = raw.get("interpreter_version") == authority.interpreter_version
        try:
            gt_source_path, _, gt_source_sha = _validate_file_ref(raw.get("gt_source"), "u3_gt_source")
            predicates["gt_source_bound"] = (
                str(gt_source_path) == authority.gt_source_path and gt_source_sha == authority.gt_source_sha256
            )
        except FrontierRefusal:
            predicates["gt_source_bound"] = False
        predicates["same_archive_n600_score"] = (
            authority.archive_sha256 == archive_sha
            and authority.archive_bytes == archive_bytes
            and authority.sample_count == 600
        )
        predicates["same_realized_stream"] = raw.get("realized_stream_hash") == authority.stream_sha256
        predicates["official_argv_bound"] = raw.get("official_argv") == list(authority.argv)
        predicates["runtime_hash_bound"] = raw.get("runtime_hash") == authority.runtime_sha256
        predicates["evaluator_hash_bound"] = raw.get("evaluator_hash") == authority.evaluator_sha256
        predicates["seed_bound"] = raw.get("seed") == authority.seed
        predicates["gt_hash_bound"] = raw.get("gt_hash") == authority.gt_sha256
    else:
        for key in (
            "same_archive_n600_score",
            "same_realized_stream",
            "official_argv_bound",
            "runtime_hash_bound",
            "evaluator_hash_bound",
            "seed_bound",
            "gt_hash_bound",
            "interpreter_path_hash_bound",
            "interpreter_version_bound",
            "gt_source_bound",
        ):
            predicates[key] = False
    for key, passed in predicates.items():
        if not passed and key not in {"not_settled_s4_identity"}:
            blocker = f"BLOCKED_U3_{key.upper()}"
            if blocker not in blockers:
                blockers.append(blocker)
    if blockers:
        status = "SETTLED_S4_REUSE_RECEIPT_ONLY_NO_RERUN" if settled else "BLOCKED_U3_RECEIVER_TUPLE"
        return U3ReceiverTuplePreflight(
            status=status,
            verdict_scope=CURRENT_SCOPE,
            predicate_table=predicates,
            blockers=tuple(dict.fromkeys(blockers)),
            execution_plan=None,
        )
    return U3ReceiverTuplePreflight(
        status="U3_RECEIVER_TUPLE_READY",
        verdict_scope=CURRENT_SCOPE,
        predicate_table=predicates,
        blockers=(),
        execution_plan={
            "commands": [
                "tools/build_s4_archive_composer.py",
                "tools/measure_s4_archive_composer.py --advisory-eval",
            ],
            "requires_tuple_validation_before_execution": True,
            "settled_s4_identity_refused": True,
        },
    )


def _validated_sibling_arms(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise FrontierRefusal("INVALID_SIBLING_ARMS")
    normalized: dict[str, Any] = {}
    for arm, value in sorted(raw.items()):
        if not isinstance(value, Mapping):
            raise FrontierRefusal(f"INVALID_SIBLING_ARM:{arm}")
        _, size, digest = _validate_file_ref(value.get("receipt"), f"sibling_{arm}_receipt")
        hits = _quarantine_hits(value)
        if hits:
            raise FrontierRefusal(f"BLOCKED_SIBLING_QUARANTINE:{arm}:{','.join(hits)}")
        normalized[str(arm)] = {
            **dict(value),
            "receipt_custody_rederived": True,
            "receipt_bytes": size,
            "receipt_sha256": digest,
        }
    return normalized


def _validated_input_manifests(raw: Sequence[Mapping[str, Any]] | None) -> tuple[Mapping[str, Any], ...]:
    """Hash-bind the exact CLI JSON inputs, not only their decoded rows."""

    normalized: list[Mapping[str, Any]] = []
    for index, value in enumerate(() if raw is None else raw):
        if not isinstance(value, Mapping):
            raise FrontierRefusal(f"INVALID_INPUT_MANIFEST_REF:{index}")
        stored_path = Path(str(value.get("path", "")))
        actual_path = stored_path if stored_path.is_absolute() else REPO_ROOT / stored_path
        expected = {**dict(value), "path": str(actual_path)}
        _, size, digest = _validate_file_ref(expected, f"input_manifest_{index}")
        _scan_bounded_text(actual_path, f"input_manifest_{index}")
        normalized.append(
            {
                "path": str(stored_path),
                "bytes": size,
                "sha256": digest,
                "custody_rederived": True,
            }
        )
    return tuple(normalized)


def compile_frontier(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    u3_row: Mapping[str, Any] | None = None,
    sibling_arms: Mapping[str, Any] | None = None,
    input_manifests: Sequence[Mapping[str, Any]] | None = None,
    pointer_source: Mapping[str, Any],
) -> AggregateReceipt:
    """Compile the four exact tolerance rows from explicitly supplied receipts."""

    pointer, pointer_custody = _canonical_pointer_from_source(pointer_source)
    candidates: list[CandidateCustody] = []
    rejections: dict[str, tuple[str, ...]] = {}
    for row in candidate_rows:
        identifier = (
            str(row.get("candidate_id", "<missing-candidate-id>"))
            if isinstance(row, Mapping)
            else "<invalid-candidate>"
        )
        try:
            candidate = candidate_from_mapping(row)
            blockers = validate_candidate(candidate)
        except FrontierRefusal as exc:
            rejections[identifier] = (str(exc),)
            continue
        candidates.append(candidate)
        if any(existing.candidate_id == candidate.candidate_id for existing in candidates[:-1]):
            rejections[candidate.candidate_id] = ("BLOCKED_DUPLICATE_CANDIDATE_ID",)
            continue
        if blockers:
            rejections[candidate.candidate_id] = blockers
    admitted = [candidate for candidate in candidates if candidate.candidate_id not in rejections]
    rows: list[ToleranceVerdict] = []
    measured: list[dict[str, Any]] = []
    dominated: list[dict[str, Any]] = []
    for tolerance in TARGET_TOLERANCES:
        feasible = [candidate for candidate in admitted if candidate.d_seg is not None and candidate.d_seg <= tolerance]
        feasible.sort(
            key=lambda row: (
                row.counted_archive_bytes,
                row.d_seg,
                row.d_pose,
                row.artifact_sha256,
                row.candidate_id,
            )
        )
        if not feasible:
            rows.append(
                ToleranceVerdict(
                    tolerance,
                    "NO_FEASIBLE_CANDIDATE",
                    CURRENT_SCOPE,
                    blockers=("NO_ADMITTED_SAME_ARTIFACT_N600_RECEIVER_CLOSED_CANDIDATE",),
                )
            )
            continue
        winner = feasible[0]
        rows.append(
            ToleranceVerdict(
                tolerance=tolerance,
                status="MEASURED_FRONTIER_POINT",
                verdict_scope=CURRENT_SCOPE,
                candidate_id=winner.candidate_id,
                counted_archive_bytes=winner.counted_archive_bytes,
                d_seg=winner.d_seg,
                d_pose=winner.d_pose,
                candidate_sha256=winner.artifact_sha256,
                evidence_axis=winner.evidence_axis,
                source_evidence_axis=winner.source_evidence_axis,
                realized_stream_hash=winner.realized_stream_hash,
                hard_score=winner.hard_score,
                provenance=dict(winner.provenance),
            )
        )
        measured.append(
            {
                "tolerance": tolerance,
                "candidate_id": winner.candidate_id,
                "counted_archive_bytes": winner.counted_archive_bytes,
                "counted_payload_bytes": winner.counted_payload_bytes,
                "container_overhead_bytes": winner.container_overhead_bytes,
                "runtime_overhead_bytes": winner.runtime_overhead_bytes,
                "d_seg": winner.d_seg,
                "d_pose": winner.d_pose,
                "candidate_sha256": winner.artifact_sha256,
                "evidence_axis": winner.evidence_axis,
                "source_evidence_axis": winner.source_evidence_axis,
                "realized_stream_hash": winner.realized_stream_hash,
                "hard_score": winner.hard_score,
                "provenance": dict(winner.provenance),
                "authority_bundle": dict(winner.authority_bundle),
            }
        )
        dominated.extend(
            {
                "tolerance": tolerance,
                "candidate_id": row.candidate_id,
                "counted_archive_bytes": row.counted_archive_bytes,
                "counted_payload_bytes": row.counted_payload_bytes,
                "container_overhead_bytes": row.container_overhead_bytes,
                "runtime_overhead_bytes": row.runtime_overhead_bytes,
                "d_seg": row.d_seg,
                "d_pose": row.d_pose,
                "candidate_sha256": row.artifact_sha256,
                "evidence_axis": row.evidence_axis,
                "source_evidence_axis": row.source_evidence_axis,
                "realized_stream_hash": row.realized_stream_hash,
                "hard_score": row.hard_score,
                "provenance": dict(row.provenance),
                "authority_bundle": dict(row.authority_bundle),
            }
            for row in feasible[1:]
        )
    resolved_u3 = {"description_only": True, "counted_bytes": R3_DESCRIPTION_BYTES} if u3_row is None else dict(u3_row)
    normalized_siblings = _validated_sibling_arms(sibling_arms)
    receipt = AggregateReceipt(
        schema=SCHEMA,
        tolerance_rows=tuple(rows),
        measured_frontier=tuple(measured),
        dominated_measured_candidates=tuple(dominated),
        candidate_rejections=rejections,
        candidates=tuple(_candidate_summary(row) for row in candidates),
        sibling_arms=normalized_siblings,
        quarantine_audit={
            "blocked_candidate_ids": sorted(
                candidate_id
                for candidate_id, reasons in rejections.items()
                if any("QUARANTINE" in row for row in reasons)
            ),
            "canonical_manifest_sha256": _quarantine_manifest_sha256(),
            "candidate_supplied_empty_list_is_not_authority": True,
            "waivers_forbidden": True,
            "predicate": "canonical manifest scan over row plus authority receipt bytes",
        },
        u3_preflight=preflight_u3(resolved_u3),
        pointer=pointer,
        pointer_source=pointer_custody,
        authority_labels={
            "current_result": "receipt-byte-rederived custody compiler",
            "counted_bytes_scope": "complete archive.zip; contest-video-derived or learned payload only",
            "generic_runtime_code_rate_bytes": 0,
            "pointer_moved": False,
            "promotion_eligible": False,
        },
        compiler_source=_compiler_source_custody(),
        source_input_manifests=_validated_input_manifests(input_manifests),
        source_candidate_rows=tuple(dict(row) for row in candidate_rows),
        source_u3_row=resolved_u3,
        source_sibling_arms={} if sibling_arms is None else dict(sibling_arms),
        quarantine_manifest_sha256=_quarantine_manifest_sha256(),
    )
    return receipt.with_hash()


def _candidate_summary(candidate: CandidateCustody) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "artifact": {
            "path": candidate.artifact_path,
            "bytes": candidate.artifact_bytes,
            "sha256": candidate.artifact_sha256,
        },
        "counted_payload_bytes": candidate.counted_payload_bytes,
        "counted_archive_bytes": candidate.counted_archive_bytes,
        "target_tolerance": candidate.target_tolerance,
        "container_overhead_bytes": candidate.container_overhead_bytes,
        "runtime_overhead_bytes": candidate.runtime_overhead_bytes,
        "levels": [asdict(level) for level in candidate.levels],
        "evidence_axis": candidate.evidence_axis,
        "source_evidence_axis": candidate.source_evidence_axis,
        "realized_stream_hash": candidate.realized_stream_hash,
        "hard_score": candidate.hard_score,
        "provenance": dict(candidate.provenance),
        "authority_bundle": dict(candidate.authority_bundle),
        "quarantined_identifiers": list(candidate.quarantined_identifiers),
    }


def validate_receipt(value: Mapping[str, Any]) -> None:
    """Rederive the full receipt from its bound source rows and current bytes."""

    if value.get("schema") != SCHEMA:
        raise FrontierRefusal("INVALID_RECEIPT_SCHEMA")
    rows = value.get("tolerance_rows")
    if not isinstance(rows, list) or [row.get("tolerance") for row in rows] != list(TARGET_TOLERANCES):
        raise FrontierRefusal("INVALID_TOLERANCE_ROWS")
    received = value.get("receipt_sha256")
    canonical = dict(value)
    canonical["receipt_sha256"] = ""
    if not isinstance(received, str) or hashlib.sha256(canonical_json(canonical)).hexdigest() != received:
        raise FrontierRefusal("RECEIPT_SELF_HASH_MISMATCH")
    if (
        value.get("main_review_required") is not True
        or value.get("promotion_eligible") is not False
        or value.get("pointer_moved") is not False
    ):
        raise FrontierRefusal("RECEIPT_AUTHORITY_FLAGS_INVALID")
    source_pointer = value.get("pointer_source")
    if not isinstance(source_pointer, Mapping):
        raise FrontierRefusal("RECEIPT_POINTER_SOURCE_INVALID")
    pointer, pointer_custody = _canonical_pointer_from_source(source_pointer)
    if value.get("pointer") != pointer or dict(source_pointer) != dict(pointer_custody):
        raise FrontierRefusal("RECEIPT_POINTER_CUSTODY_MISMATCH")
    source_manifests = value.get("source_input_manifests")
    source_rows = value.get("source_candidate_rows")
    source_u3 = value.get("source_u3_row")
    source_siblings = value.get("source_sibling_arms")
    if (
        not isinstance(source_manifests, list)
        or not all(isinstance(row, Mapping) for row in source_manifests)
        or not isinstance(source_rows, list)
        or not all(isinstance(row, Mapping) for row in source_rows)
        or not isinstance(source_u3, Mapping)
        or not isinstance(source_siblings, Mapping)
    ):
        raise FrontierRefusal("RECEIPT_SOURCE_INPUTS_INVALID")
    rederived = compile_frontier(
        source_rows,
        u3_row=source_u3,
        sibling_arms=source_siblings,
        input_manifests=source_manifests,
        pointer_source=source_pointer,
    ).as_dict()
    if canonical_json(rederived) != canonical_json(value):
        raise FrontierRefusal("RECEIPT_REDERIVATION_MISMATCH")


def write_checkpoint(path: Path, receipt: AggregateReceipt) -> None:
    """Publish once through an exclusive final-name claim; never overwrite."""

    payload = canonical_json(receipt.as_dict()) + b"\n"
    if len(payload) > MAX_RECEIPT_BYTES:
        raise FrontierRefusal(f"FRONTIER_RECEIPT_TOO_LARGE:{len(payload)}>{MAX_RECEIPT_BYTES}")
    if path.is_file():
        if path.read_bytes() != payload:
            raise FrontierRefusal(f"INCOMPATIBLE_RESUME_OUTPUT: {path}")
        return
    try:
        write_bytes_artifact(path, payload, allow_overwrite=False)
    except ArtifactWriteError as exc:
        # A racing writer may have won the hard-link claim. Idempotent resume is
        # allowed only when its final bytes are exactly ours.
        if path.is_file() and path.read_bytes() == payload:
            return
        raise FrontierRefusal(f"INCOMPATIBLE_RESUME_OUTPUT: {path}") from exc
