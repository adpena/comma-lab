#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the strict V15-to-C1 conditional quotient profile over all 600 pairs.

This is an encoder-only profiler, not a candidate builder or evaluator.  The
single positional argument is a typed JSON config.  Current production use
strictly parses the counted V15 archive, renders bounded camera chunks through
``CarrierComposeReceiverV1.render_camera_pairs``, applies C0B's exact integer
resize, and compares them with the custody-bound C1 target chunks.  Fresh
batch-16 labels are reopened through the compile-ready teacher gate and are
used only for class-conditioned statistics.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, localcontext
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TOOLS_ROOT = REPO_ROOT / "tools"
for root in (SRC_ROOT, TOOLS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from build_c0b_semantic_quotient_archive import C1TargetTeacher  # noqa: E402

from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    CarrierComposeReceiverV1,
    receive_carrier_compose_archive,
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError  # noqa: E402
from tac.optimization.uint8_lattice_feasibility import (  # noqa: E402
    DisjointResizeOperator,
    Uint8LatticeError,
)
from tac.witness_control.taskspace_fresh_teacher_materializer_v1 import (  # noqa: E402
    FreshTeacherMaterializationError,
    load_compile_ready_materialization_receipt,
)
from tac.witness_dsl.c0b_semantic_quotient import (  # noqa: E402
    PlaneChunk,
    SemanticQuotientError,
    exact_resize_round_u8,
    sha256_file,
    storage_preflight,
    write_once_or_equal,
)
from tac.witness_dsl.taskspace_conditional_quotient_profiler_v1 import (  # noqa: E402
    EVIDENCE_AXIS,
    INPUT_BINDING_SCHEMA,
    PUBLIC_PAIR_COUNT,
    UPSTREAM_DEFAULT_BATCH_SIZE,
    ConditionalQuotientProfileConfigV1,
    ConditionalQuotientProfilerError,
    run_conditional_quotient_profile,
)

CLI_CONFIG_SCHEMA: Final = "tac.taskspace_conditional_quotient_profile_cli_config.v1"
TOOL_RECEIPT_SCHEMA: Final = "tac.taskspace_conditional_quotient_profile_tool_receipt.v2"
PREFLIGHT_RECEIPT_SCHEMA: Final = "tac.taskspace_conditional_quotient_profile_preflight.v2"
CONTEST_ARCHIVE_DENOMINATOR: Final = 37_545_489
CONTEST_RATE_NUMERATOR: Final = 25
C1_CHUNK_PAIRS: Final = 12
SUB_015_TARGET: Final = Decimal("0.15")
EXPECTED_SELECTION_RULE: Final = (
    "min(our_local_frontier_contest_cpu, our_local_frontier_contest_cuda, upstream_official_leaderboard.best_entry)"
)
FRESH_V15_DERIVATION_SCHEMA: Final = "tac.taskspace_fresh_v15_derivation_custody.v1"
FRESH_V15_RECEIPT_SCHEMA: Final = "ddm_v15_scorer_solved_template_receipt.v1"
FRESH_V15_RECEIVER_CHECKPOINT_SCHEMA: Final = "ddm_v15_receiver_closed_archive.v1"
V15_RECEIVER_SOURCE_PATH: Final = "src/tac/optimization/direct_description_carrier_compose.py"
FRESH_V15_IDENTITY_CHECKPOINT_SCHEMA: Final = "ddm_v15_full_p_camera_identity_batch.v1"
FRESH_V15_ARCHIVE_BYTES: Final = 133_941
FRESH_V15_ARCHIVE_SHA256: Final = "759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df"
FRESH_V15_IDENTITY_CHECKPOINT_COUNT: Final = 38


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ConditionalQuotientProfilerError("value is not canonical JSON") from exc


def _read_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConditionalQuotientProfilerError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ConditionalQuotientProfilerError(f"{label} is not a JSON object")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ConditionalQuotientProfilerError(f"{label} must be a lowercase SHA-256")
    return value


def _resolved_file(identity: Mapping[str, Any], *, label: str) -> Path:
    if set(identity) != {"path", "expected_sha256"}:
        raise ConditionalQuotientProfilerError(f"{label} identity keys differ")
    path = Path(str(identity["path"])).expanduser().resolve(strict=True)
    expected = _require_sha256(identity["expected_sha256"], label=f"{label}.expected_sha256")
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise ConditionalQuotientProfilerError(f"{label} path/type/SHA-256 differs")
    return path


def _resolved_directory(identity: Mapping[str, Any], *, label: str) -> Path:
    if set(identity) != {"path", "prepare_receipt_sha256"}:
        raise ConditionalQuotientProfilerError(f"{label} identity keys differ")
    root = Path(str(identity["path"])).expanduser().resolve(strict=True)
    expected = _require_sha256(
        identity["prepare_receipt_sha256"],
        label=f"{label}.prepare_receipt_sha256",
    )
    receipt = root / "prepare_receipt.json"
    if root.is_symlink() or not root.is_dir() or sha256_file(receipt) != expected:
        raise ConditionalQuotientProfilerError(f"{label} root/prepare receipt differs")
    return root


def _require_path_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ConditionalQuotientProfilerError(f"{label} must be a non-empty trimmed path")
    return value


def _validate_fresh_v15_derivation_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConditionalQuotientProfilerError("fresh V15 derivation custody must be an object")
    expected_keys = {
        "schema",
        "expected_run_id",
        "compile_receipt",
        "source_config",
        "adjacent_archive",
        "producer_sources",
        "receiver_checkpoint",
        "identity_checkpoints",
        "identity_digest_chain_sha256",
    }
    if set(value) != expected_keys or value.get("schema") != FRESH_V15_DERIVATION_SCHEMA:
        raise ConditionalQuotientProfilerError("fresh V15 derivation custody keys/schema differ")
    run_id = value.get("expected_run_id")
    if not isinstance(run_id, str) or len(run_id) < 8 or run_id.strip() != run_id:
        raise ConditionalQuotientProfilerError("fresh V15 expected run_id is invalid")
    compile_receipt = value.get("compile_receipt")
    if not isinstance(compile_receipt, dict) or set(compile_receipt) != {
        "path",
        "expected_sha256",
        "expected_schema",
    }:
        raise ConditionalQuotientProfilerError("fresh V15 compile receipt identity keys differ")
    _require_path_text(compile_receipt["path"], label="fresh V15 compile receipt path")
    _require_sha256(
        compile_receipt["expected_sha256"],
        label="fresh_v15.compile_receipt.expected_sha256",
    )
    if compile_receipt["expected_schema"] != FRESH_V15_RECEIPT_SCHEMA:
        raise ConditionalQuotientProfilerError("fresh V15 compile receipt schema binding differs")
    source_config = value.get("source_config")
    if not isinstance(source_config, dict) or set(source_config) != {
        "path",
        "expected_sha256",
        "expected_rfc8785_sha256",
    }:
        raise ConditionalQuotientProfilerError("fresh V15 source config identity keys differ")
    _require_path_text(source_config["path"], label="fresh V15 source config path")
    _require_sha256(
        source_config["expected_sha256"],
        label="fresh_v15.source_config.expected_sha256",
    )
    _require_sha256(
        source_config["expected_rfc8785_sha256"],
        label="fresh_v15.source_config.expected_rfc8785_sha256",
    )
    adjacent_archive = value.get("adjacent_archive")
    if not isinstance(adjacent_archive, dict) or set(adjacent_archive) != {
        "path",
        "expected_bytes",
        "expected_sha256",
    }:
        raise ConditionalQuotientProfilerError("fresh V15 adjacent archive identity keys differ")
    _require_path_text(adjacent_archive["path"], label="fresh V15 adjacent archive path")
    if adjacent_archive.get("expected_bytes") != FRESH_V15_ARCHIVE_BYTES:
        raise ConditionalQuotientProfilerError("fresh V15 adjacent archive byte binding differs")
    archive_sha = _require_sha256(
        adjacent_archive["expected_sha256"],
        label="fresh_v15.adjacent_archive.expected_sha256",
    )
    if archive_sha != FRESH_V15_ARCHIVE_SHA256:
        raise ConditionalQuotientProfilerError("fresh V15 adjacent archive SHA binding differs")
    producers = value.get("producer_sources")
    if not isinstance(producers, list) or not producers:
        raise ConditionalQuotientProfilerError("fresh V15 producer source identities are absent")
    producer_paths: list[str] = []
    for index, identity in enumerate(producers):
        if not isinstance(identity, dict) or set(identity) != {
            "path",
            "expected_bytes",
            "expected_sha256",
        }:
            raise ConditionalQuotientProfilerError(f"fresh V15 producer source {index} identity keys differ")
        producer_paths.append(_require_path_text(identity["path"], label=f"fresh V15 producer source {index} path"))
        if (
            isinstance(identity["expected_bytes"], bool)
            or not isinstance(identity["expected_bytes"], int)
            or identity["expected_bytes"] <= 0
        ):
            raise ConditionalQuotientProfilerError(f"fresh V15 producer source {index} bytes are invalid")
        _require_sha256(
            identity["expected_sha256"],
            label=f"fresh_v15.producer_sources[{index}].expected_sha256",
        )
    if len(set(producer_paths)) != len(producer_paths):
        raise ConditionalQuotientProfilerError("fresh V15 producer source paths are not unique")
    receiver_checkpoint = value.get("receiver_checkpoint")
    if not isinstance(receiver_checkpoint, dict) or set(receiver_checkpoint) != {
        "path",
        "expected_sha256",
    }:
        raise ConditionalQuotientProfilerError("fresh V15 receiver checkpoint identity keys differ")
    _require_path_text(receiver_checkpoint["path"], label="fresh V15 receiver checkpoint path")
    _require_sha256(
        receiver_checkpoint["expected_sha256"],
        label="fresh_v15.receiver_checkpoint.expected_sha256",
    )
    checkpoints = value.get("identity_checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != FRESH_V15_IDENTITY_CHECKPOINT_COUNT:
        raise ConditionalQuotientProfilerError("fresh V15 requires exactly 38 identity checkpoints")
    for index, identity in enumerate(checkpoints):
        start = index * UPSTREAM_DEFAULT_BATCH_SIZE
        stop = min(start + UPSTREAM_DEFAULT_BATCH_SIZE, PUBLIC_PAIR_COUNT)
        if not isinstance(identity, dict) or set(identity) != {
            "path",
            "expected_sha256",
            "local_pair_range",
        }:
            raise ConditionalQuotientProfilerError(f"fresh V15 identity checkpoint {index} identity keys differ")
        checkpoint_path = _require_path_text(
            identity["path"],
            label=f"fresh V15 identity checkpoint {index} path",
        )
        if Path(checkpoint_path).name != f"batch_{start:04d}_{stop:04d}.json":
            raise ConditionalQuotientProfilerError(f"fresh V15 identity checkpoint {index} filename/range differ")
        if identity["local_pair_range"] != [start, stop]:
            raise ConditionalQuotientProfilerError(f"fresh V15 identity checkpoint {index} ordered range differs")
        _require_sha256(
            identity["expected_sha256"],
            label=f"fresh_v15.identity_checkpoints[{index}].expected_sha256",
        )
    _require_sha256(
        value["identity_digest_chain_sha256"],
        label="fresh_v15.identity_digest_chain_sha256",
    )
    return value


def load_cli_config(path: Path) -> dict[str, Any]:
    """Parse the closed CLI config without opening its large scientific inputs."""

    value = _read_json_mapping(path, label="conditional quotient CLI config")
    expected_keys = {
        "schema",
        "profile",
        "work_root",
        "fresh_v15_derivation",
        "c1_root",
        "selected_plane_geometry_custody",
        "canonical_batch16_debt_receipt",
        "independent_batch16_replay_receipt",
        "fresh_teacher_receipt",
        "frontier_pointer",
    }
    if set(value) != expected_keys or value.get("schema") != CLI_CONFIG_SCHEMA:
        raise ConditionalQuotientProfilerError("conditional quotient CLI config keys/schema differ")
    if not isinstance(value["profile"], dict):
        raise ConditionalQuotientProfilerError("CLI profile field must be a typed object")
    profile = ConditionalQuotientProfileConfigV1.from_mapping(value["profile"])
    if profile.chunk_pairs != C1_CHUNK_PAIRS:
        raise ConditionalQuotientProfilerError("production C1 custody requires exact 12-pair chunks")
    _validate_fresh_v15_derivation_config(value["fresh_v15_derivation"])
    for field in (
        "selected_plane_geometry_custody",
        "canonical_batch16_debt_receipt",
        "independent_batch16_replay_receipt",
        "fresh_teacher_receipt",
        "frontier_pointer",
    ):
        identity = value[field]
        if not isinstance(identity, dict) or set(identity) != {"path", "expected_sha256"}:
            raise ConditionalQuotientProfilerError(f"CLI {field} identity keys differ")
        _require_sha256(identity["expected_sha256"], label=f"{field}.expected_sha256")
    if not isinstance(value["c1_root"], dict):
        raise ConditionalQuotientProfilerError("CLI c1_root identity must be an object")
    _resolved_directory(value["c1_root"], label="C1 target root")
    work_root = Path(str(value["work_root"])).expanduser()
    if not work_root.name or str(value["work_root"]).strip() != str(value["work_root"]):
        raise ConditionalQuotientProfilerError("CLI work_root must be a non-empty trimmed path")
    return value


def _strict_v15_receiver(path: Path) -> tuple[bytes, CarrierComposeReceiverV1]:
    payload = path.read_bytes()
    try:
        receiver = receive_carrier_compose_archive(payload, verify_member_effects=True)
    except DirectDescriptionError as exc:
        raise ConditionalQuotientProfilerError("V15 archive failed strict receiver parse") from exc
    if (
        receiver.archive != payload
        or int(receiver.z.n_pairs) != PUBLIC_PAIR_COUNT
        or int(receiver.predictor.source_pair_start) != 0
        or receiver.realization_profile is None
    ):
        raise ConditionalQuotientProfilerError("V15 receiver population/realization contract differs")
    return payload, receiver


def _frontier_score(pointer: Mapping[str, Any]) -> Decimal:
    effective = pointer.get("effective_frontier")
    if not isinstance(effective, Mapping):
        raise ConditionalQuotientProfilerError("frontier pointer lacks effective_frontier")
    score = effective.get("score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or float(score) <= 0.0
    ):
        raise ConditionalQuotientProfilerError("frontier pointer score is invalid")
    if effective.get("selection_rule") != EXPECTED_SELECTION_RULE:
        raise ConditionalQuotientProfilerError("frontier pointer selection rule differs")
    return Decimal(str(score))


def _receipt_self_hash(value: Mapping[str, Any], *, label: str) -> str:
    receipt_sha = _require_sha256(value.get("receipt_sha256"), label=f"{label}.receipt_sha256")
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if _sha256(_canonical_json(body)) != receipt_sha:
        raise ConditionalQuotientProfilerError(f"{label} internal receipt SHA-256 differs")
    return receipt_sha


def _largest_archive_below(
    *,
    target: Decimal,
    d_seg: Decimal,
    d_pose: Decimal,
) -> int:
    with localcontext() as context:
        context.prec = 80
        distortion = Decimal(100) * d_seg + (Decimal(10) * d_pose).sqrt()
        if target <= distortion:
            return 0
        exact_boundary = (target - distortion) * Decimal(CONTEST_ARCHIVE_DENOMINATOR) / Decimal(CONTEST_RATE_NUMERATOR)
        return max(0, int(exact_boundary.to_integral_value(rounding=ROUND_CEILING)) - 1)


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConditionalQuotientProfilerError("cannot resolve Git HEAD") from exc
    head = result.stdout.strip()
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise ConditionalQuotientProfilerError("Git HEAD is not a full lowercase commit")
    return head


def _module_source_path(module: str) -> Path | None:
    if module == "tac" or module.startswith("tac."):
        stem = SRC_ROOT.joinpath(*module.split("."))
    elif "." not in module:
        stem = TOOLS_ROOT / module
    else:
        return None
    module_path = stem.with_suffix(".py")
    if module_path.is_file():
        return module_path.resolve()
    package_path = stem / "__init__.py"
    if package_path.is_file():
        return package_path.resolve()
    return None


def _module_name_for_path(path: Path) -> tuple[str, tuple[str, ...]]:
    resolved = path.resolve()
    if resolved.is_relative_to(SRC_ROOT):
        relative = resolved.relative_to(SRC_ROOT)
    elif resolved.is_relative_to(TOOLS_ROOT):
        relative = resolved.relative_to(TOOLS_ROOT)
    else:
        raise ConditionalQuotientProfilerError("implementation source escaped repository source roots")
    if relative.name == "__init__.py":
        module_parts = relative.parts[:-1]
        package_parts = module_parts
    else:
        module_parts = (*relative.parts[:-1], relative.stem)
        package_parts = module_parts[:-1]
    return ".".join(module_parts), tuple(package_parts)


def _local_import_paths(path: Path) -> tuple[Path, ...]:
    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise ConditionalQuotientProfilerError(f"cannot parse implementation dependency: {path}") from exc
    _module_name, package_parts = _module_name_for_path(path)
    resolved: set[Path] = set()
    for node in ast.walk(tree):
        candidates: list[str] = []
        aliases: tuple[ast.alias, ...] = ()
        if isinstance(node, ast.Import):
            candidates.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            aliases = tuple(node.names)
            if node.level:
                if node.level > len(package_parts) + 1:
                    raise ConditionalQuotientProfilerError(f"relative import escapes local package: {path}")
                prefix = package_parts[: len(package_parts) - (node.level - 1)]
                suffix = tuple(node.module.split(".")) if node.module else ()
                base = ".".join((*prefix, *suffix))
            else:
                base = node.module or ""
            if base:
                candidates.append(base)
                candidates.extend(f"{base}.{alias.name}" for alias in aliases if alias.name != "*")
        for candidate in candidates:
            dependency = _module_source_path(candidate)
            if dependency is not None:
                resolved.add(dependency)
    return tuple(sorted(resolved))


def _implementation_sources() -> dict[str, Any]:
    pending = [
        Path(__file__).resolve(strict=True),
        (REPO_ROOT / "src/tac/witness_dsl/taskspace_conditional_quotient_profiler_v1.py").resolve(strict=True),
    ]
    paths: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in paths:
            continue
        paths.add(path)
        pending.extend(_local_import_paths(path))
    try:
        tracked_result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConditionalQuotientProfilerError("cannot enumerate tracked implementation closure") from exc
    tracked = {(REPO_ROOT / raw.decode()).resolve() for raw in tracked_result.stdout.split(b"\0") if raw}
    missing = sorted(path for path in paths if path not in tracked)
    if missing:
        names = ", ".join(path.relative_to(REPO_ROOT).as_posix() for path in missing)
        raise ConditionalQuotientProfilerError(f"implementation dependency closure contains untracked sources: {names}")
    return {
        path.relative_to(REPO_ROOT).as_posix(): {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths)
    }


@dataclass(frozen=True, slots=True)
class _FreshV15Derivation:
    receipt_path: Path
    receipt: Mapping[str, Any]
    source_config_path: Path
    archive_path: Path
    archive_payload: bytes
    custody: Mapping[str, Any]


def _resolve_bound_file(
    path_text: str,
    expected_sha256: str,
    *,
    label: str,
) -> Path:
    unresolved = Path(path_text).expanduser()
    path = unresolved.resolve(strict=True)
    if unresolved.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
        raise ConditionalQuotientProfilerError(f"{label} path/type/SHA-256 differs")
    return path


def _resolve_repo_recorded_file(path_text: str, *, label: str) -> Path:
    recorded = Path(path_text)
    unresolved = recorded if recorded.is_absolute() else REPO_ROOT / recorded
    try:
        path = unresolved.resolve(strict=True)
    except OSError as exc:
        raise ConditionalQuotientProfilerError(f"cannot resolve {label}: {path_text}") from exc
    if unresolved.is_symlink() or not path.is_file():
        raise ConditionalQuotientProfilerError(f"{label} is not a regular non-symlink file")
    return path


def _resolve_fresh_v15_derivation(value: Mapping[str, Any]) -> _FreshV15Derivation:
    derivation = _validate_fresh_v15_derivation_config(dict(value))
    compile_identity = derivation["compile_receipt"]
    receipt_path = _resolve_bound_file(
        compile_identity["path"],
        compile_identity["expected_sha256"],
        label="fresh V15 compile receipt",
    )
    receipt = _read_json_mapping(receipt_path, label="fresh V15 compile receipt")
    run_id = derivation["expected_run_id"]
    if (
        receipt.get("schema") != compile_identity["expected_schema"]
        or receipt.get("run_id") != run_id
        or receipt.get("research_only") is not True
        or receipt.get("execution_allowed") is not False
        or receipt.get("score_claim") is not False
        or receipt.get("promotion_eligible") is not False
        or receipt.get("pointer_moved") is not False
    ):
        raise ConditionalQuotientProfilerError("fresh V15 compile receipt schema/run/authority differs")

    config_identity = derivation["source_config"]
    source_config_path = _resolve_bound_file(
        config_identity["path"],
        config_identity["expected_sha256"],
        label="fresh V15 source config",
    )
    source_config = _read_json_mapping(source_config_path, label="fresh V15 source config")
    typed_config_sha = _sha256(rfc8785_canonicalize(source_config))
    if (
        typed_config_sha != config_identity["expected_rfc8785_sha256"]
        or typed_config_sha != receipt.get("typed_config_sha256")
        or rfc8785_canonicalize(source_config) != rfc8785_canonicalize(receipt.get("typed_config"))
        or source_config.get("run_id") != run_id
    ):
        raise ConditionalQuotientProfilerError("fresh V15 source/RFC8785 typed config custody differs")

    archive_identity = derivation["adjacent_archive"]
    archive_path = _resolve_bound_file(
        archive_identity["path"],
        archive_identity["expected_sha256"],
        label="fresh V15 adjacent archive",
    )
    if archive_path.parent != receipt_path.parent:
        raise ConditionalQuotientProfilerError(
            "fresh V15 archive is not adjacent to its compile receipt; historical-path fallback refused"
        )
    archive_payload = archive_path.read_bytes()
    if len(archive_payload) != archive_identity["expected_bytes"]:
        raise ConditionalQuotientProfilerError("fresh V15 adjacent archive byte count differs")

    selected_name = receipt.get("selected_candidate")
    ladder = receipt.get("solved_template_ladder")
    if not isinstance(selected_name, str) or not isinstance(ladder, list):
        raise ConditionalQuotientProfilerError("fresh V15 compile receipt lacks selected ladder custody")
    selected_rows = [row for row in ladder if isinstance(row, Mapping) and row.get("candidate") == selected_name]
    if len(selected_rows) != 1:
        raise ConditionalQuotientProfilerError("fresh V15 selected ladder row is not unique")
    selected = selected_rows[0]
    receiver_custody = selected.get("receiver_custody")
    if (
        selected.get("archive_bytes") != len(archive_payload)
        or selected.get("archive_sha256") != archive_identity["expected_sha256"]
        or selected.get("score_claim") is not False
        or not isinstance(receiver_custody, Mapping)
        or receiver_custody.get("archive_bytes") != len(archive_payload)
        or receiver_custody.get("archive_sha256") != archive_identity["expected_sha256"]
        or receiver_custody.get("score_claim") is not False
    ):
        raise ConditionalQuotientProfilerError("fresh V15 selected archive custody differs")

    producer_records = receipt.get("producer_custody")
    configured_producers = derivation["producer_sources"]
    if not isinstance(producer_records, list) or len(producer_records) != len(configured_producers):
        raise ConditionalQuotientProfilerError("fresh V15 producer custody count differs")
    live_producers: list[dict[str, Any]] = []
    for index, (configured, recorded) in enumerate(zip(configured_producers, producer_records, strict=True)):
        expected_record = {
            "path": configured["path"],
            "bytes": configured["expected_bytes"],
            "sha256": configured["expected_sha256"],
        }
        if recorded != expected_record:
            raise ConditionalQuotientProfilerError(f"fresh V15 producer source {index} receipt/config custody differs")
        producer_path = _resolve_repo_recorded_file(
            configured["path"],
            label=f"fresh V15 producer source {index}",
        )
        if (
            producer_path.stat().st_size != configured["expected_bytes"]
            or sha256_file(producer_path) != configured["expected_sha256"]
        ):
            raise ConditionalQuotientProfilerError(f"fresh V15 producer source {index} live bytes/SHA-256 differ")
        live_producers.append(
            {
                "path": configured["path"],
                "resolved_path": str(producer_path),
                "bytes": configured["expected_bytes"],
                "sha256": configured["expected_sha256"],
                "live_rehashed": True,
            }
        )

    receiver_identity = derivation["receiver_checkpoint"]
    receiver_path = _resolve_bound_file(
        receiver_identity["path"],
        receiver_identity["expected_sha256"],
        label="fresh V15 receiver checkpoint",
    )
    expected_receiver_path = receipt_path.parent / "stage_checkpoints" / "02_receiver_closed_archive.json"
    if receiver_path != expected_receiver_path.resolve(strict=True):
        raise ConditionalQuotientProfilerError("fresh V15 receiver checkpoint is outside the fresh run")
    receiver_checkpoint = _read_json_mapping(
        receiver_path,
        label="fresh V15 receiver checkpoint",
    )
    checkpoint_archive = receiver_checkpoint.get("archive")
    if (
        receiver_checkpoint.get("schema") != FRESH_V15_RECEIVER_CHECKPOINT_SCHEMA
        or receiver_checkpoint.get("typed_config_sha256") != typed_config_sha
        or receiver_checkpoint.get("score_claim") is not False
        or not isinstance(checkpoint_archive, Mapping)
        or checkpoint_archive.get("bytes") != len(archive_payload)
        or checkpoint_archive.get("sha256") != archive_identity["expected_sha256"]
    ):
        raise ConditionalQuotientProfilerError("fresh V15 receiver checkpoint contract differs")
    embedded_archive_path = _resolve_repo_recorded_file(
        str(checkpoint_archive.get("path")),
        label="fresh V15 receiver checkpoint archive",
    )
    if embedded_archive_path != archive_path:
        raise ConditionalQuotientProfilerError("fresh V15 receiver checkpoint names a different archive path")

    identity_summary = selected.get("full_p_camera_identity")
    if (
        not isinstance(identity_summary, Mapping)
        or identity_summary.get("pair_count") != PUBLIC_PAIR_COUNT
        or identity_summary.get("batch_count") != FRESH_V15_IDENTITY_CHECKPOINT_COUNT
        or identity_summary.get("batch_size") != UPSTREAM_DEFAULT_BATCH_SIZE
        or identity_summary.get("all_camera_bytes_identical") is not True
    ):
        raise ConditionalQuotientProfilerError("fresh V15 compile receipt identity summary differs")
    checkpoint_root = receipt_path.parent / "stage_checkpoints" / "full_p_camera_identity"
    expected_checkpoint_names = {Path(identity["path"]).name for identity in derivation["identity_checkpoints"]}
    actual_checkpoint_names = {
        entry.name for entry in checkpoint_root.iterdir() if entry.is_file() and not entry.is_symlink()
    }
    if actual_checkpoint_names != expected_checkpoint_names:
        raise ConditionalQuotientProfilerError(
            "fresh V15 identity checkpoint directory has missing or extra regular files"
        )
    checkpoint_rows: list[dict[str, Any]] = []
    digest_material: list[str] = []
    for index, identity in enumerate(derivation["identity_checkpoints"]):
        start, stop = identity["local_pair_range"]
        checkpoint_path = _resolve_bound_file(
            identity["path"],
            identity["expected_sha256"],
            label=f"fresh V15 identity checkpoint {index}",
        )
        expected_path = checkpoint_root / f"batch_{start:04d}_{stop:04d}.json"
        if checkpoint_path != expected_path.resolve(strict=True):
            raise ConditionalQuotientProfilerError(
                f"fresh V15 identity checkpoint {index} is outside the fresh ordered run"
            )
        row = _read_json_mapping(
            checkpoint_path,
            label=f"fresh V15 identity checkpoint {index}",
        )
        expected_row_keys = {
            "schema",
            "typed_config_sha256",
            "local_pair_range",
            "base_camera_sha256",
            "final_camera_sha256",
            "byte_identical",
            "camera_bytes_released_after_compare",
            "score_claim",
        }
        if (
            set(row) != expected_row_keys
            or row.get("schema") != FRESH_V15_IDENTITY_CHECKPOINT_SCHEMA
            or row.get("typed_config_sha256") != typed_config_sha
            or row.get("local_pair_range") != [start, stop]
            or row.get("byte_identical") is not True
            or row.get("camera_bytes_released_after_compare") is not True
            or row.get("score_claim") is not False
        ):
            raise ConditionalQuotientProfilerError(f"fresh V15 identity checkpoint {index} contract differs")
        base_digest = _require_sha256(
            row.get("base_camera_sha256"),
            label=f"fresh V15 identity checkpoint {index} base camera digest",
        )
        final_digest = _require_sha256(
            row.get("final_camera_sha256"),
            label=f"fresh V15 identity checkpoint {index} final camera digest",
        )
        if base_digest != final_digest:
            raise ConditionalQuotientProfilerError(f"fresh V15 identity checkpoint {index} camera digests differ")
        digest_material.append(base_digest + final_digest)
        checkpoint_rows.append(
            {
                "path": str(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
                "sha256": identity["expected_sha256"],
                "local_pair_range": [start, stop],
                "typed_config_sha256": typed_config_sha,
                "base_camera_sha256": base_digest,
                "final_camera_sha256": final_digest,
                "byte_identical": True,
                "score_claim": False,
            }
        )
    recomputed_chain = _sha256("".join(digest_material).encode("ascii"))
    if recomputed_chain != derivation["identity_digest_chain_sha256"] or recomputed_chain != identity_summary.get(
        "digest_chain_sha256"
    ):
        raise ConditionalQuotientProfilerError("fresh V15 38-checkpoint digest chain differs")

    custody = {
        "schema": FRESH_V15_DERIVATION_SCHEMA,
        "run_id": run_id,
        "derivation_proof_separate_from_archive_content_identity": True,
        "historical_path_fallback_allowed": False,
        "compile_receipt": {
            "path": str(receipt_path),
            "bytes": receipt_path.stat().st_size,
            "sha256": compile_identity["expected_sha256"],
            "schema": receipt["schema"],
            "run_id": receipt["run_id"],
        },
        "source_config": {
            "path": str(source_config_path),
            "bytes": source_config_path.stat().st_size,
            "sha256": config_identity["expected_sha256"],
            "rfc8785_sha256": typed_config_sha,
        },
        "adjacent_archive": {
            "path": str(archive_path),
            "bytes": len(archive_payload),
            "sha256": archive_identity["expected_sha256"],
            "content_identity_only": True,
        },
        "producer_sources": live_producers,
        "receiver_checkpoint": {
            "path": str(receiver_path),
            "bytes": receiver_path.stat().st_size,
            "sha256": receiver_identity["expected_sha256"],
            "schema": receiver_checkpoint["schema"],
            "typed_config_sha256": typed_config_sha,
            "archive_sha256": archive_identity["expected_sha256"],
            "score_claim": False,
        },
        "full_p_camera_identity": {
            "pair_count": PUBLIC_PAIR_COUNT,
            "batch_count": FRESH_V15_IDENTITY_CHECKPOINT_COUNT,
            "batch_size": UPSTREAM_DEFAULT_BATCH_SIZE,
            "typed_config_sha256": typed_config_sha,
            "ordered_checkpoints": checkpoint_rows,
            "receipt_digest_chain_sha256": identity_summary["digest_chain_sha256"],
            "recomputed_digest_chain_sha256": recomputed_chain,
            "digest_chain_matches_receipt": True,
            "all_camera_bytes_identical": True,
            "score_claim": False,
        },
    }
    return _FreshV15Derivation(
        receipt_path=receipt_path,
        receipt=receipt,
        source_config_path=source_config_path,
        archive_path=archive_path,
        archive_payload=archive_payload,
        custody=custody,
    )


def _build_input_binding(
    *,
    config: ConditionalQuotientProfileConfigV1,
    v15_path: Path,
    v15_payload: bytes,
    receiver: CarrierComposeReceiverV1,
    teacher: C1TargetTeacher,
    geometry_custody_path: Path,
    geometry_custody: Mapping[str, Any],
    fresh_receipt_path: Path,
    fresh_receipt: Mapping[str, Any],
    canonical_debt_path: Path,
    canonical_debt: Mapping[str, Any],
    corroboration_path: Path,
    corroboration: Mapping[str, Any],
    pointer_path: Path,
    pointer: Mapping[str, Any],
    implementation_sources: Mapping[str, Any],
    fresh_v15_derivation_custody: Mapping[str, Any],
) -> dict[str, Any]:
    v15_receiver_source = implementation_sources.get(V15_RECEIVER_SOURCE_PATH)
    if (
        not isinstance(v15_receiver_source, Mapping)
        or v15_receiver_source.get("path") != V15_RECEIVER_SOURCE_PATH
        or not isinstance(v15_receiver_source.get("sha256"), str)
    ):
        raise ConditionalQuotientProfilerError(
            "recursive implementation closure lacks the exact V15 receiver source identity"
        )
    artifacts = geometry_custody.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ConditionalQuotientProfilerError("selected-plane geometry custody lacks artifacts")
    ms1 = artifacts.get("ms1")
    v15 = artifacts.get("v15")
    economics = geometry_custody.get("economics")
    if not isinstance(ms1, Mapping) or not isinstance(v15, Mapping) or not isinstance(economics, Mapping):
        raise ConditionalQuotientProfilerError("selected-plane geometry custody shape differs")
    diagnostic = ms1.get("diagnostic")
    archive = v15.get("archive")
    if not isinstance(diagnostic, Mapping) or not isinstance(archive, Mapping):
        raise ConditionalQuotientProfilerError("selected-plane/V15 custody fields differ")
    origin_batch = diagnostic.get("batch_geometry")
    if isinstance(origin_batch, bool) or not isinstance(origin_batch, int):
        raise ConditionalQuotientProfilerError("selected-plane scorer batch geometry is absent")
    if (
        archive.get("sha256") != _sha256(v15_payload)
        or archive.get("bytes") != len(v15_payload)
        or receiver.custody.get("archive_sha256") != _sha256(v15_payload)
    ):
        raise ConditionalQuotientProfilerError("strict V15 receiver differs from geometry custody")

    target_labels = fresh_receipt.get("target_labels")
    if not isinstance(target_labels, Mapping):
        raise ConditionalQuotientProfilerError("fresh teacher receipt lacks target labels")
    if (
        fresh_receipt.get("scorer_pair_batch_size") != UPSTREAM_DEFAULT_BATCH_SIZE
        or target_labels.get("shape") != [config.pair_count, config.scorer_hw[0], config.scorer_hw[1]]
        or target_labels.get("dtype") != "uint8"
    ):
        raise ConditionalQuotientProfilerError("fresh teacher population/geometry differs")

    canonical_aggregate = canonical_debt.get("aggregate")
    canonical_config = canonical_debt.get("config")
    canonical_crosscheck = canonical_debt.get("contest_cpu_reference_same_decoded_raw")
    corroboration_distortion = corroboration.get("distortion")
    if (
        canonical_debt.get("schema") != "tac.coupled_witness_raw_debt.v2"
        or canonical_debt.get("axis") != "[Darwin-arm64 CPU advisory] NON-PROMOTABLE"
        or not isinstance(canonical_aggregate, Mapping)
        or canonical_aggregate.get("pair_count") != config.pair_count
        or not isinstance(canonical_config, Mapping)
        or canonical_config.get("scorer_batch_pairs") != UPSTREAM_DEFAULT_BATCH_SIZE
        or canonical_debt.get("score_claim") is not False
        or canonical_debt.get("promotion_eligible") is not False
        or canonical_debt.get("pointer_moved") is not False
        or not isinstance(canonical_crosscheck, Mapping)
        or canonical_crosscheck.get("evidence_grade") != "contest-CPU"
    ):
        raise ConditionalQuotientProfilerError("canonical batch-16 debt receipt contract differs")
    canonical_raw = canonical_config.get("raw")
    crosscheck_raw = canonical_crosscheck.get("decoded_raw")
    if (
        not isinstance(canonical_raw, Mapping)
        or not isinstance(crosscheck_raw, Mapping)
        or canonical_raw.get("sha256") != crosscheck_raw.get("sha256")
    ):
        raise ConditionalQuotientProfilerError("canonical batch-16 same-raw contest crosscheck differs")
    if (
        corroboration.get("schema") != "tac.taskspace_candidate_batch_replay_receipt.v1"
        or corroboration.get("evidence_axis") != "[macOS-CPU exact-upstream-mirror advisory]"
        or corroboration.get("pair_count") != config.pair_count
        or corroboration.get("batch_size") != UPSTREAM_DEFAULT_BATCH_SIZE
        or corroboration.get("batch_geometry_matches_upstream_default") is not True
        or corroboration.get("score_claim") is not False
        or corroboration.get("promotion_eligible") is not False
        or corroboration.get("pointer_mutation_allowed") is not False
        or not isinstance(corroboration_distortion, Mapping)
    ):
        raise ConditionalQuotientProfilerError("independent batch-16 corroboration contract differs")
    corroboration_raw = corroboration.get("candidate_raw")
    if not isinstance(corroboration_raw, Mapping) or corroboration_raw.get("sha256") != canonical_raw.get("sha256"):
        raise ConditionalQuotientProfilerError("independent batch-16 replay is not the same decoded raw")

    frontier = _frontier_score(pointer)
    d_seg = Decimal(str(canonical_aggregate.get("mean_d_seg")))
    d_pose = Decimal(str(canonical_aggregate.get("mean_d_pose")))
    corroboration_d_seg = Decimal(str(corroboration_distortion.get("d_seg")))
    corroboration_d_pose = Decimal(str(corroboration_distortion.get("d_pose")))
    with localcontext() as context:
        context.prec = 80
        canonical_distortion = Decimal(100) * d_seg + (Decimal(10) * d_pose).sqrt()
        corroboration_distortion_term = Decimal(100) * corroboration_d_seg + (Decimal(10) * corroboration_d_pose).sqrt()
        corroboration_minus_primary = corroboration_distortion_term - canonical_distortion
    base_bytes = len(v15_payload)
    frontier_ceiling = _largest_archive_below(
        target=frontier,
        d_seg=d_seg,
        d_pose=d_pose,
    )
    sub015_ceiling = _largest_archive_below(
        target=SUB_015_TARGET,
        d_seg=d_seg,
        d_pose=d_pose,
    )
    historical_frontier_ceiling = _largest_archive_below(
        target=frontier,
        d_seg=Decimal(str(economics.get("d_seg"))),
        d_pose=Decimal(str(economics.get("d_pose"))),
    )
    historical_sub015_ceiling = _largest_archive_below(
        target=SUB_015_TARGET,
        d_seg=Decimal(str(economics.get("d_seg"))),
        d_pose=Decimal(str(economics.get("d_pose"))),
    )
    binding = {
        "schema": INPUT_BINDING_SCHEMA,
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_mutation_allowed": False,
        "candidate_payload_allowed": False,
        "teacher_payload_serialized": False,
        "scorer_weights_present": False,
        "pair_count": config.pair_count,
        "scorer_hw": list(config.scorer_hw),
        "channels": config.channels,
        "v15_archive_path": str(v15_path),
        "v15_archive_bytes": len(v15_payload),
        "v15_archive_sha256": _sha256(v15_payload),
        "v15_strict_parse": True,
        "v15_current_receiver_source_sha256": v15_receiver_source["sha256"],
        "fresh_v15_derivation_custody": dict(fresh_v15_derivation_custody),
        "base_coordinate_transform": {
            "camera_hw": [874, 1164],
            "scorer_hw": list(config.scorer_hw),
            "method": "c0b_disjoint_factor2_exact_integer_resize_round_u8",
            "operator_source": "src/tac/optimization/uint8_lattice_feasibility.py",
        },
        "selected_plane_teacher_id": teacher.custody()["teacher_id"],
        "selected_plane_y0_sha256": teacher.custody()["y0_sha256"],
        "selected_plane_y1_sha256": teacher.custody()["y1_sha256"],
        "selected_plane_origin_scorer_batch_size": origin_batch,
        "selected_plane_geometry_custody": {
            "path": str(geometry_custody_path),
            "sha256": sha256_file(geometry_custody_path),
        },
        "fresh_teacher_scorer_batch_size": fresh_receipt["scorer_pair_batch_size"],
        "fresh_teacher_target_labels_path": target_labels["path"],
        "fresh_teacher_target_labels_sha256": target_labels["sha256"],
        "fresh_teacher_receipt": {
            "path": str(fresh_receipt_path),
            "sha256": sha256_file(fresh_receipt_path),
            "sealed_receipt_sha256": fresh_receipt["receipt_sha256"],
        },
        "upstream_default_scorer_batch_size": UPSTREAM_DEFAULT_BATCH_SIZE,
        "current_planning_scorer_batch_size": canonical_config["scorer_batch_pairs"],
        "current_planning_matches_upstream_batch_geometry": (
            canonical_config["scorer_batch_pairs"] == UPSTREAM_DEFAULT_BATCH_SIZE
        ),
        "canonical_batch16_debt_receipt": {
            "path": str(canonical_debt_path),
            "sha256": sha256_file(canonical_debt_path),
            "receipt_sha256": _receipt_self_hash(canonical_debt, label="canonical batch-16 debt"),
            "axis": canonical_debt["axis"],
            "batch_size": canonical_config["scorer_batch_pairs"],
            "decoded_raw_sha256": canonical_raw["sha256"],
            "contest_cpu_same_raw_crosscheck_sha256": canonical_crosscheck["sha256"],
            "authority": "PRIMARY_EXISTING_BATCH16_PLANNING_COORDINATE_NOT_SCORE_AUTHORITY",
            "score_claim": False,
        },
        "independent_batch16_replay_corroboration": {
            "path": str(corroboration_path),
            "sha256": sha256_file(corroboration_path),
            "receipt_sha256": _receipt_self_hash(corroboration, label="independent batch-16 replay"),
            "axis": corroboration["evidence_axis"],
            "batch_size": corroboration["batch_size"],
            "decoded_raw_sha256": corroboration_raw["sha256"],
            "d_seg": float(corroboration_d_seg),
            "d_pose": float(corroboration_d_pose),
            "distortion_minus_canonical_primary": float(corroboration_minus_primary),
            "authority": "INDEPENDENT_CORROBORATION_ONLY_NOT_PRIMARY_OR_SCORE_AUTHORITY",
            "score_claim": False,
        },
        "planning_coordinate_premise": (
            "PREEXISTING_CANONICAL_BATCH16_PRIMARY_G54_INDEPENDENT_CORROBORATION_NO_NOVELTY"
        ),
        "frontier_pointer": {
            "path": str(pointer_path),
            "sha256": sha256_file(pointer_path),
            "effective_frontier_score": float(frontier),
            "selection_rule": EXPECTED_SELECTION_RULE,
        },
        "current_batch16_planning_coordinate": {
            "effective_frontier_score": float(frontier),
            "d_seg": float(d_seg),
            "d_pose": float(d_pose),
            "largest_total_archive_bytes_below_effective_frontier": frontier_ceiling,
            "largest_total_archive_bytes_below_sub_0_15": sub015_ceiling,
            "base_archive_bytes": base_bytes,
            "headroom_bytes_to_effective_frontier": max(0, frontier_ceiling - base_bytes),
            "headroom_bytes_to_sub_0_15": max(0, sub015_ceiling - base_bytes),
            "authority": "canonical_batch16_planning_arithmetic_only_not_new_eval_or_frontier_authority",
            "score_claim": False,
        },
        "historical_ms1_batch32_counterfactual": {
            "effective_frontier_score": float(frontier),
            "d_seg": float(economics.get("d_seg")),
            "d_pose": float(economics.get("d_pose")),
            "largest_total_archive_bytes_below_effective_frontier": historical_frontier_ceiling,
            "largest_total_archive_bytes_below_sub_0_15": historical_sub015_ceiling,
            "base_archive_bytes": base_bytes,
            "headroom_bytes_to_effective_frontier": max(0, historical_frontier_ceiling - base_bytes),
            "headroom_bytes_to_sub_0_15": max(0, historical_sub015_ceiling - base_bytes),
            "authority": ("historical_batch32_coupled_score_arithmetic_only_not_eval_or_frontier_authority"),
            "score_claim": False,
        },
        "implementation_sources": dict(implementation_sources),
    }
    _canonical_json(binding)
    return binding


class _ProductionChunkLoader:
    def __init__(
        self,
        *,
        receiver: CarrierComposeReceiverV1,
        teacher: C1TargetTeacher,
        labels: np.memmap,
        config: ConditionalQuotientProfileConfigV1,
    ) -> None:
        self.receiver = receiver
        self.teacher_iterator: Iterator[PlaneChunk] = iter(teacher.chunks())
        self.next_teacher_index = 0
        self.labels = labels
        self.config = config
        try:
            self.operator = DisjointResizeOperator.build(
                camera_h=874,
                camera_w=1164,
                scorer_h=config.scorer_hw[0],
                scorer_w=config.scorer_hw[1],
            )
        except Uint8LatticeError as exc:
            raise ConditionalQuotientProfilerError("exact C0B resize geometry refused") from exc

    def _target(self, chunk_index: int) -> PlaneChunk:
        target: PlaneChunk | None = None
        while self.next_teacher_index <= chunk_index:
            try:
                target = next(self.teacher_iterator)
            except StopIteration as exc:
                raise ConditionalQuotientProfilerError("C1 teacher ended before requested chunk") from exc
            if target.chunk_index != self.next_teacher_index:
                raise ConditionalQuotientProfilerError("C1 teacher chunk chronology differs")
            self.next_teacher_index += 1
        if target is None or target.chunk_index != chunk_index:
            raise ConditionalQuotientProfilerError(
                "resumed C1 target loader cannot seek backwards; checkpoint chronology drifted"
            )
        return target

    def __call__(
        self,
        chunk_index: int,
        pair_ids: tuple[int, ...],
    ) -> tuple[PlaneChunk, PlaneChunk, np.ndarray]:
        target = self._target(chunk_index)
        if target.pair_ids != pair_ids:
            raise ConditionalQuotientProfilerError("C1 target pair range differs")
        try:
            camera = self.receiver.render_camera_pairs(pair_ids)
            y0 = np.stack([exact_resize_round_u8(self.operator, frame) for frame in camera[:, 0]])
            y1 = np.stack([exact_resize_round_u8(self.operator, frame) for frame in camera[:, 1]])
        except (DirectDescriptionError, SemanticQuotientError) as exc:
            raise ConditionalQuotientProfilerError("strict V15 base rendering failed") from exc
        base = PlaneChunk(chunk_index, pair_ids, y0, y1)
        label_chunk = np.ascontiguousarray(self.labels[list(pair_ids)])
        return base, target, label_chunk


@dataclass(frozen=True, slots=True)
class _PreparedInputs:
    config_path: Path
    config: ConditionalQuotientProfileConfigV1
    cli: Mapping[str, Any]
    v15_path: Path
    v15_payload: bytes
    receiver: CarrierComposeReceiverV1
    teacher: C1TargetTeacher
    fresh: Mapping[str, Any]
    binding: Mapping[str, Any]
    implementation_sources: Mapping[str, Any]
    config_sha256: str
    git_sha_start: str
    work_root: Path


def _prepare_inputs(config_path: Path) -> _PreparedInputs:
    resolved_config = config_path.expanduser().resolve(strict=True)
    config_sha256 = sha256_file(resolved_config)
    git_sha_start = _git_head()
    cli = load_cli_config(resolved_config)
    config = ConditionalQuotientProfileConfigV1.from_mapping(cli["profile"])
    if config.test_only_small_fixture:
        raise ConditionalQuotientProfilerError(
            "production CLI refuses test_only_small_fixture; synthetic fixtures use the core API"
        )
    fresh_v15_derivation = _resolve_fresh_v15_derivation(cli["fresh_v15_derivation"])
    v15_path = fresh_v15_derivation.archive_path
    geometry_path = _resolved_file(
        cli["selected_plane_geometry_custody"],
        label="selected-plane geometry custody",
    )
    canonical_debt_path = _resolved_file(
        cli["canonical_batch16_debt_receipt"],
        label="canonical batch-16 debt receipt",
    )
    corroboration_path = _resolved_file(
        cli["independent_batch16_replay_receipt"],
        label="independent batch-16 replay receipt",
    )
    fresh_path = _resolved_file(
        cli["fresh_teacher_receipt"],
        label="fresh teacher receipt",
    )
    pointer_path = _resolved_file(cli["frontier_pointer"], label="frontier pointer")
    c1_root = _resolved_directory(cli["c1_root"], label="C1 target root")
    geometry = _read_json_mapping(geometry_path, label="selected-plane geometry custody")
    canonical_debt = _read_json_mapping(canonical_debt_path, label="canonical batch-16 debt receipt")
    corroboration = _read_json_mapping(corroboration_path, label="independent batch-16 replay receipt")
    pointer = _read_json_mapping(pointer_path, label="frontier pointer")
    try:
        fresh = load_compile_ready_materialization_receipt(fresh_path)
    except FreshTeacherMaterializationError as exc:
        raise ConditionalQuotientProfilerError("fresh teacher compile-ready gate refused") from exc
    try:
        teacher = C1TargetTeacher(c1_root)
    except SemanticQuotientError as exc:
        raise ConditionalQuotientProfilerError("C1 selected-plane custody refused") from exc
    v15_payload, receiver = _strict_v15_receiver(v15_path)
    if v15_payload != fresh_v15_derivation.archive_payload:
        raise ConditionalQuotientProfilerError("fresh V15 archive bytes changed after derivation custody")
    sources_pre = _implementation_sources()
    binding = _build_input_binding(
        config=config,
        v15_path=v15_path,
        v15_payload=v15_payload,
        receiver=receiver,
        teacher=teacher,
        geometry_custody_path=geometry_path,
        geometry_custody=geometry,
        fresh_receipt_path=fresh_path,
        fresh_receipt=fresh,
        canonical_debt_path=canonical_debt_path,
        canonical_debt=canonical_debt,
        corroboration_path=corroboration_path,
        corroboration=corroboration,
        pointer_path=pointer_path,
        pointer=pointer,
        implementation_sources=sources_pre,
        fresh_v15_derivation_custody=fresh_v15_derivation.custody,
    )
    return _PreparedInputs(
        config_path=resolved_config,
        config=config,
        cli=cli,
        v15_path=v15_path,
        v15_payload=v15_payload,
        receiver=receiver,
        teacher=teacher,
        fresh=fresh,
        binding=binding,
        implementation_sources=sources_pre,
        config_sha256=config_sha256,
        git_sha_start=git_sha_start,
        work_root=Path(str(cli["work_root"])).expanduser().resolve(strict=False),
    )


def _require_prepared_sources_stable(
    prepared: _PreparedInputs,
) -> tuple[dict[str, Any], str]:
    if sha256_file(prepared.config_path) != prepared.config_sha256:
        raise ConditionalQuotientProfilerError("typed config changed after preparation")
    sources_post = _implementation_sources()
    if sources_post != prepared.implementation_sources:
        raise ConditionalQuotientProfilerError("complete implementation dependency closure changed")
    return sources_post, _git_head()


def preflight(config_path: Path) -> dict[str, Any]:
    """Strictly reopen all custody and seal readiness without rendering pairs."""

    prepared = _prepare_inputs(config_path)
    try:
        storage = storage_preflight(
            prepared.work_root,
            required_bytes=1 << 30,
            test_only_small_fixture=False,
            allow_local_storage=prepared.config.allow_local_storage,
        )
    except SemanticQuotientError as exc:
        raise ConditionalQuotientProfilerError("profile preflight storage gate refused") from exc
    sources_post, git_sha_end = _require_prepared_sources_stable(prepared)
    receipt = {
        "schema": PREFLIGHT_RECEIPT_SCHEMA,
        "config_path": str(prepared.config_path),
        "config_sha256": prepared.config_sha256,
        "git_sha_start": prepared.git_sha_start,
        "git_sha_end": git_sha_end,
        "git_head_stable": git_sha_end == prepared.git_sha_start,
        "git_head_drift_policy": (
            "diagnostic remains reproducible only when the complete local source "
            "dependency closure and typed config are byte-identical"
        ),
        "profile": prepared.config.as_mapping(),
        "input_binding": dict(prepared.binding),
        "storage_preflight": {
            key: storage[key]
            for key in (
                "schema",
                "selected_tier",
                "required_bytes",
                "passed",
                "test_only_small_fixture",
                "allow_local_storage",
            )
        },
        "strict_sources_reopened": True,
        "pair_rendering_started": False,
        "chunks_profiled": 0,
        "resumable": True,
        "per_stage_checkpoints": True,
        "full_n600_launch_authorized_by_this_receipt": False,
        "operator_or_governor_must_launch_explicitly": True,
        "frontier_feasibility_inference_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_mutation_performed": False,
        "launch_governance": {
            "status": "LAUNCH_NOT_PERFORMED",
            "dispatch_claim_bound": False,
            "governed_launcher_receipt_bound": False,
        },
        "implementation_sources": {
            "pre": dict(prepared.implementation_sources),
            "post": sources_post,
            "byte_identical": True,
        },
    }
    receipt["preflight_receipt_sha256"] = _sha256(_canonical_json(receipt))
    prepared.work_root.mkdir(parents=True, exist_ok=True)
    try:
        write_once_or_equal(
            prepared.work_root / "preflight_receipt.json",
            _canonical_json(receipt),
        )
    except SemanticQuotientError as exc:
        raise ConditionalQuotientProfilerError("profile preflight receipt write/resume failed") from exc
    return receipt


def run(config_path: Path) -> dict[str, Any]:
    prepared = _prepare_inputs(config_path)
    config = prepared.config
    fresh = prepared.fresh
    label_identity = fresh["target_labels"]
    labels = np.memmap(
        Path(label_identity["path"]),
        mode="r",
        dtype=np.uint8,
        shape=(config.pair_count, config.scorer_hw[0], config.scorer_hw[1]),
    )
    loader = _ProductionChunkLoader(
        receiver=prepared.receiver,
        teacher=prepared.teacher,
        labels=labels,
        config=config,
    )
    aggregate = run_conditional_quotient_profile(
        config=config,
        input_binding=prepared.binding,
        work_root=prepared.work_root,
        chunk_loader=loader,
    )
    sources_post, git_sha_end = _require_prepared_sources_stable(prepared)
    receipt = {
        "schema": TOOL_RECEIPT_SCHEMA,
        "config_path": str(prepared.config_path),
        "config_sha256": prepared.config_sha256,
        "git_sha_start": prepared.git_sha_start,
        "git_sha_end": git_sha_end,
        "git_head_stable": git_sha_end == prepared.git_sha_start,
        "git_head_drift_policy": (
            "diagnostic remains reproducible only when the complete local source "
            "dependency closure and typed config are byte-identical"
        ),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "implementation_sources": {
            "pre": dict(prepared.implementation_sources),
            "post": sources_post,
            "byte_identical": True,
        },
        "aggregate_path": str(prepared.work_root / "aggregate_receipt.json"),
        "aggregate_receipt_sha256": aggregate["aggregate_receipt_sha256"],
        "current_planning_matches_upstream_batch_geometry": prepared.binding[
            "current_planning_matches_upstream_batch_geometry"
        ],
        "frontier_feasibility_inference_allowed": False,
        "resumable": True,
        "per_stage_checkpoints": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_mutation_performed": False,
        "launch_governance": {
            "status": "NOT_CAPTURED_BY_PROFILER_V2",
            "dispatch_claim_bound": False,
            "governed_launcher_receipt_bound": False,
            "candidate_or_promotion_authority": False,
        },
    }
    receipt["tool_receipt_sha256"] = _sha256(_canonical_json(receipt))
    try:
        write_once_or_equal(
            prepared.work_root / "tool_receipt.json",
            _canonical_json(receipt),
        )
    except SemanticQuotientError as exc:
        raise ConditionalQuotientProfilerError("tool receipt write/resume failed") from exc
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help=f"typed {CLI_CONFIG_SCHEMA} JSON")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="strictly reopen custody and seal a zero-chunk readiness receipt",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        receipt = preflight(args.config) if args.preflight_only else run(args.config)
    except (
        ConditionalQuotientProfilerError,
        OSError,
        ValueError,
    ) as exc:
        raise SystemExit(f"REFUSE: {exc}") from exc
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
