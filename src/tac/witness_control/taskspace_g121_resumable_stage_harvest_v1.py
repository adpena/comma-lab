# SPDX-License-Identifier: MIT
"""Exhaustive, resumable harvest of immutable G111 semantic stages.

G121 is an encoder-side orchestration layer.  It never scores injected arrays,
never consumes a trainer BEST/PARETO pointer, and never grants contest score
authority.  Production accepts only recursively reopened G111 lineage nodes,
their immutable G112 partitions, and the minimum-safe G120-v2 physical
measurement/observation contract.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib
import inspect
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

from tac.witness_dsl.dynamic_frontier_target import (
    DynamicFrontierTargetError,
    DynamicFrontierTargetSnapshot,
    verify_dynamic_frontier_target_snapshot,
)

RETAINED_PREPOSE_SCHEMA: Final = "tac.g121_retained_prepose.v2"
STAGE_MEASUREMENT_SCHEMA: Final = "tac.g121_stage_measurement.v2"
COMPLETION_RECEIPT_SCHEMA: Final = "tac.g121_completion_receipt.v2"
SCHEDULING_HINT_SCHEMA: Final = "tac.g121_g105_public_wire_scheduling_hint.v2"
G120_REQUIRED_PRODUCTION_SCHEMA: Final = (
    "tac.g120_parsed_stage_production_authority.v2"
)
G120_REQUIRED_MEASUREMENT_SCHEMA: Final = "tac.g120_stage_measurement.v2"
G120_REQUIRED_OBSERVATION_SCHEMA: Final = "tac.g120_stage_observation.v2"
G120_REQUIRED_SCOPED_OBSTRUCTION_SCHEMA: Final = (
    "tac.g120_exact_distortion_obstruction.v1"
)
G120_REQUIRED_RUNNER: Final = "run_g120_parsed_stage_production_authority_v2"
G120_REQUIRED_MEASUREMENT_OPENER: Final = "open_g120_stage_measurement_v2"
G120_REQUIRED_OBSERVATION_OPENER: Final = "open_g120_stage_observation_v2"
G120_REQUIRED_SCOPED_OBSTRUCTION_OPENER: Final = (
    "open_g120_exact_distortion_obstruction_v1"
)
G120_REQUIRED_SCOPED_OBSTRUCTION_ERROR: Final = (
    "G120ExactDistortionObstruction"
)
RETAINED_PREPOSE_BASENAME: Final = "g121_retained_prepose.json"
STAGE_LEDGER_BASENAME: Final = "g121_stage_measurements.jsonl"
COMPLETION_RECEIPT_BASENAME: Final = "g121_completion_receipt.json"
SCHEDULING_HINT_BASENAME: Final = "g105_public_wire_best.json"
EXACT_SEG_PIXEL_DENOMINATOR: Final = 600 * 384 * 512
EXACT_OBSTRUCTION_RULE: Final = (
    "100*k*target_denominator < target_numerator*pixel_denominator"
)

RETAIN_POST_G105_POSE: Final = "RETAIN_POST_G105_POSE"
DEFER_G115_WIRE_QAT: Final = "DEFER_G115_WIRE_QAT"
PRUNE_EXACT_DISTORTION_OBSTRUCTION: Final = (
    "PRUNE_EXACT_DISTORTION_OBSTRUCTION"
)
BLOCKED_SCOPED: Final = "BLOCKED_SCOPED"
DISPOSITIONS: Final = frozenset(
    {
        RETAIN_POST_G105_POSE,
        DEFER_G115_WIRE_QAT,
        PRUNE_EXACT_DISTORTION_OBSTRUCTION,
        BLOCKED_SCOPED,
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STAGE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_UNSIGNED_DECIMAL = re.compile(r"^(0|[1-9][0-9]*)(?:\.[0-9]+)?$")
_PRESERVED_DEPLOY = re.compile(
    r"^levelset_ckpt_(?P<tag>stage[A-Za-z0-9_]+)_ep(?P<epoch>[0-9]+)\.npz$"
)
_PERIODIC_DEPLOY = re.compile(
    r"^levelset_periodic_ema_(?P<tag>stage[A-Za-z0-9_]+)_ep"
    r"(?P<epoch>[0-9]+)\.npz$"
)
_EPHEMERAL_ROOTS: Final = (
    Path("/tmp"),
    Path("/private/tmp"),
    Path("/var/tmp"),
)


class G121StageHarvestError(RuntimeError):
    """A physical custody, safe-authority, or exhaustive-harvest gate failed."""


@dataclass(frozen=True, slots=True)
class G121RetainedStageV2:
    """One exact, distortion-open G111 stage preserved for G119."""

    value: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return json.loads(json.dumps(self.value))


@dataclass(frozen=True, slots=True)
class G121RetainedPreposeV2:
    """Strictly reopened exhaustive retained-prepose population."""

    schema: str
    rows: tuple[G121RetainedStageV2, ...]
    completion_receipt: dict[str, object]
    exhaustive_enumeration_proven: bool
    manifest_path: Path
    manifest_sha256: str
    pointer_snapshot_identity_sha256: str
    postverified_pointer_identity_sha256: str
    live_target_score: float
    live_target_score_decimal: str
    live_target_numerator: int
    live_target_denominator: int


@dataclass(frozen=True, slots=True)
class G121StageHarvestResultV1:
    """Durable G121 reductions; never a score or promotion receipt."""

    retained_prepose_path: Path
    retained_prepose_sha256: str
    completion_receipt_path: Path
    completion_receipt_sha256: str
    stage_ledger_path: Path
    stage_ledger_sha256: str
    scheduling_hint_path: Path | None
    scheduling_hint_sha256: str | None
    discovered_stage_count: int
    accounted_stage_count: int
    retained_stage_count: int
    deferred_stage_count: int
    pruned_stage_count: int
    blocked_stage_count: int
    scorer_replay_count: int
    reused_measurement_count: int
    exhaustive_enumeration_proven: bool = True
    research_only: bool = True
    score_claim: bool = False
    evaluation_claim: bool = False
    promotion_eligible: bool = False
    pointer_moved: bool = False


@dataclass(frozen=True, slots=True)
class G121StageHarvestProgressV1:
    """Durable incremental work; deliberately carries no exhaustive manifest."""

    stage_ledger_path: Path
    stage_ledger_sha256: str
    discovered_stage_count: int
    accounted_stage_count: int
    scorer_replay_count: int
    reused_measurement_count: int
    exhaustive_enumeration_proven: bool = False
    producer_terminal_proven: bool = False
    research_only: bool = True
    score_claim: bool = False
    evaluation_claim: bool = False
    promotion_eligible: bool = False
    pointer_moved: bool = False


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise G121StageHarvestError(
            "G121 value is not finite canonical ASCII JSON"
        ) from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise G121StageHarvestError(f"{name} must be a lowercase SHA-256")
    return value


def _require_exact_dict(
    value: object,
    *,
    keys: set[str] | frozenset[str],
    name: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(keys):
        raise G121StageHarvestError(f"{name} has a noncanonical key set")
    return value


def _require_stage_tag(value: object) -> str:
    if type(value) is not str or _STAGE_TAG.fullmatch(value) is None:
        raise G121StageHarvestError("stage_tag is not canonical")
    return value


def _durable_directory(path: Path, *, name: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise G121StageHarvestError(f"{name} must be an absolute pathlib.Path")
    resolved = path.resolve(strict=False)
    if any(resolved == root or root in resolved.parents for root in _EPHEMERAL_ROOTS):
        raise G121StageHarvestError(f"{name} must not be temporary")
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise G121StageHarvestError(f"{name} must be a real directory")
    return path.resolve()


def _stable_regular_file(
    path: Path,
    *,
    name: str,
    expected_sha256: str | None = None,
) -> tuple[bytes, dict[str, object]]:
    if not isinstance(path, Path) or not path.is_absolute():
        raise G121StageHarvestError(f"{name} must be an absolute pathlib.Path")
    lexical = Path(os.path.abspath(os.fspath(path)))
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(lexical, flags)
    except OSError as exc:
        raise G121StageHarvestError(f"{name} cannot be opened physically") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size < 0:
            raise G121StageHarvestError(f"{name} must be a regular file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1 << 20))
            if not chunk:
                raise G121StageHarvestError(f"{name} truncated during reopen")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise G121StageHarvestError(f"{name} grew during reopen")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise G121StageHarvestError(f"{name} changed during reopen")
    try:
        named = lexical.stat(follow_symlinks=False)
    except OSError as exc:
        raise G121StageHarvestError(f"{name} disappeared during reopen") from exc
    if not stat.S_ISREG(named.st_mode) or identity != (
        named.st_dev,
        named.st_ino,
        named.st_size,
        named.st_mtime_ns,
        named.st_ctime_ns,
    ):
        raise G121StageHarvestError(f"{name} path identity changed")
    payload = b"".join(chunks)
    digest = _sha256(payload)
    if expected_sha256 is not None and digest != _require_sha256(
        expected_sha256,
        name=f"expected {name}",
    ):
        raise G121StageHarvestError(f"{name} SHA-256 differs")
    return payload, {
        "path": str(lexical.resolve()),
        "bytes": len(payload),
        "sha256": digest,
    }


def _binding_from_path(path: Path, *, name: str) -> dict[str, object]:
    return _stable_regular_file(path, name=name)[1]


def _open_binding(value: object, *, name: str) -> tuple[bytes, dict[str, object]]:
    binding = _require_exact_dict(
        value,
        keys={"path", "bytes", "sha256"},
        name=f"{name} binding",
    )
    if type(binding["bytes"]) is not int or binding["bytes"] < 0:
        raise G121StageHarvestError(f"{name} byte count is invalid")
    payload, reopened = _stable_regular_file(
        Path(str(binding["path"])),
        name=name,
        expected_sha256=_require_sha256(binding["sha256"], name=f"{name} SHA-256"),
    )
    if reopened != binding:
        raise G121StageHarvestError(f"{name} physical binding differs")
    return payload, reopened


def _seal(body: Mapping[str, object], *, key: str) -> dict[str, object]:
    if key in body:
        raise G121StageHarvestError(f"unsealed G121 body contains {key}")
    sealed = dict(body)
    sealed[key] = _sha256(_canonical_json(body))
    return sealed


def _atomic_write_once(path: Path, payload: bytes) -> None:
    if type(payload) is not bytes or not payload:
        raise G121StageHarvestError("atomic output payload must be nonempty bytes")
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise G121StageHarvestError(f"output is not a regular file: {path.name}")
        if path.read_bytes() == payload:
            return
        raise G121StageHarvestError(f"refusing to overwrite different {path.name}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            if path.is_symlink() or path.read_bytes() != payload:
                raise G121StageHarvestError(
                    f"concurrent output differs: {path.name}"
                ) from exc
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_replace(path: Path, payload: bytes) -> None:
    if type(payload) is not bytes or not payload:
        raise G121StageHarvestError("replacement payload must be nonempty bytes")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise G121StageHarvestError(f"refusing non-regular output: {path.name}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parse_json(payload: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G121StageHarvestError(f"{name} is not ASCII JSON") from exc
    if type(value) is not dict or _canonical_json(value) != payload:
        raise G121StageHarvestError(f"{name} is not canonical JSON")
    return value


def _exact_live_target(value: object) -> tuple[dict[str, Any], Fraction]:
    target = _require_exact_dict(
        value,
        keys={
            "score_decimal",
            "score_rational",
            "pointer_snapshot_identity_sha256",
            "postverified_pointer_identity_sha256",
        },
        name="live_target",
    )
    decimal_text = target["score_decimal"]
    rational = _require_exact_dict(
        target["score_rational"],
        keys={"numerator", "denominator"},
        name="live target rational",
    )
    if (
        type(decimal_text) is not str
        or _UNSIGNED_DECIMAL.fullmatch(decimal_text) is None
        or type(rational["numerator"]) is not int
        or type(rational["denominator"]) is not int
        or rational["numerator"] <= 0
        or rational["denominator"] <= 0
    ):
        raise G121StageHarvestError("live target decimal/rational is invalid")
    try:
        decimal_value = Decimal(decimal_text)
    except InvalidOperation as exc:
        raise G121StageHarvestError("live target decimal is invalid") from exc
    fraction = Fraction(rational["numerator"], rational["denominator"])
    if (
        not decimal_value.is_finite()
        or decimal_value <= 0
        or Fraction(decimal_value) != fraction
        or fraction.numerator != rational["numerator"]
        or fraction.denominator != rational["denominator"]
    ):
        raise G121StageHarvestError(
            "live target decimal/rational differs or is not reduced"
        )
    pointer_identity = _require_sha256(
        target["pointer_snapshot_identity_sha256"],
        name="pointer snapshot identity",
    )
    postverified_identity = _require_sha256(
        target["postverified_pointer_identity_sha256"],
        name="postverified pointer identity",
    )
    if pointer_identity != postverified_identity:
        raise G121StageHarvestError(
            "competitive pointer changed before observation publication"
        )
    return target, fraction


def _exact_public_wire_seg(
    value: object,
    *,
    measurement_identity_sha256: str | None = None,
) -> tuple[dict[str, Any], int]:
    public_wire = _require_exact_dict(
        value,
        keys={
            "disagreement_pixels",
            "pixel_denominator",
            "d_seg_rational",
            "d_seg_display_float",
            "measurement_identity_sha256",
        },
        name="public_wire_seg",
    )
    disagreements = public_wire["disagreement_pixels"]
    denominator = public_wire["pixel_denominator"]
    rational = _require_exact_dict(
        public_wire["d_seg_rational"],
        keys={"numerator", "denominator"},
        name="public-wire d_seg rational",
    )
    display = public_wire["d_seg_display_float"]
    identity = _require_sha256(
        public_wire["measurement_identity_sha256"],
        name="public-wire measurement identity",
    )
    if (
        type(disagreements) is not int
        or type(denominator) is not int
        or denominator != EXACT_SEG_PIXEL_DENOMINATOR
        or not 0 <= disagreements <= denominator
        or rational["numerator"] != disagreements
        or rational["denominator"] != denominator
        or type(display) is not float
        or not math.isfinite(display)
        or display != disagreements / denominator
        or (
            measurement_identity_sha256 is not None
            and identity != measurement_identity_sha256
        )
    ):
        raise G121StageHarvestError("public-wire exact coordinate differs")
    return public_wire, disagreements


def _validate_file_binding_shape(value: object, *, name: str) -> dict[str, Any]:
    binding = _require_exact_dict(
        value,
        keys={"path", "bytes", "sha256"},
        name=f"{name} binding",
    )
    if (
        type(binding["path"]) is not str
        or not Path(binding["path"]).is_absolute()
        or type(binding["bytes"]) is not int
        or binding["bytes"] <= 0
    ):
        raise G121StageHarvestError(f"{name} binding is invalid")
    _require_sha256(binding["sha256"], name=f"{name} SHA-256")
    return binding


def _validate_source_float(value: object) -> tuple[dict[str, Any], int | None]:
    source = _require_exact_dict(
        value,
        keys={
            "status",
            "disagreement_pixels",
            "pixel_denominator",
            "measurement_receipt",
        },
        name="source_float_seg",
    )
    status_value = source["status"]
    disagreements = source["disagreement_pixels"]
    denominator = source["pixel_denominator"]
    if denominator != EXACT_SEG_PIXEL_DENOMINATOR:
        raise G121StageHarvestError("source-float denominator differs")
    if status_value == "unmeasured":
        if disagreements is not None or source["measurement_receipt"] is not None:
            raise G121StageHarvestError("unmeasured source-float carries facts")
        return source, None
    if (
        status_value != "measured"
        or type(disagreements) is not int
        or not 0 <= disagreements <= denominator
    ):
        raise G121StageHarvestError("source-float measurement is invalid")
    _validate_file_binding_shape(
        source["measurement_receipt"],
        name="source-float measurement receipt",
    )
    return source, disagreements


def _validate_wire_regret(
    value: object,
    *,
    wire_disagreements: int,
    source_disagreements: int | None,
) -> dict[str, Any]:
    regret = _require_exact_dict(
        value,
        keys={"status", "disagreement_delta_pixels", "rational", "receipt"},
        name="wire_regret",
    )
    if regret["status"] == "unmeasured":
        if any(
            regret[key] is not None
            for key in ("disagreement_delta_pixels", "rational", "receipt")
        ):
            raise G121StageHarvestError("unmeasured wire regret carries facts")
        if source_disagreements is not None:
            raise G121StageHarvestError(
                "measured source-float requires measured wire regret"
            )
        return regret
    delta = regret["disagreement_delta_pixels"]
    rational = _require_exact_dict(
        regret["rational"],
        keys={"numerator", "denominator"},
        name="wire-regret rational",
    )
    if (
        regret["status"] != "measured"
        or source_disagreements is None
        or type(delta) is not int
        or delta != wire_disagreements - source_disagreements
        or rational["numerator"] != delta
        or rational["denominator"] != EXACT_SEG_PIXEL_DENOMINATOR
    ):
        raise G121StageHarvestError(
            "wire regret differs from exact disagreement counts"
        )
    # Negative regret is legal and must be preserved.
    _validate_file_binding_shape(regret["receipt"], name="wire-regret receipt")
    return regret


def _validate_g115_qat(value: object) -> dict[str, Any]:
    qat = _require_exact_dict(
        value,
        keys={
            "status",
            "terminal_stage_physical_identity_sha256",
            "disagreement_pixels",
            "pixel_denominator",
            "receipt",
        },
        name="g115_qat",
    )
    status_value = qat["status"]
    if status_value in {"not_required", "required_unmeasured"}:
        if (
            qat["terminal_stage_physical_identity_sha256"] is not None
            or qat["disagreement_pixels"] is not None
            or qat["receipt"] is not None
            or qat["pixel_denominator"] != EXACT_SEG_PIXEL_DENOMINATOR
        ):
            raise G121StageHarvestError("nonterminal G115-QAT carries custody")
    elif status_value == "terminal_stage_measured":
        _require_sha256(
            qat["terminal_stage_physical_identity_sha256"],
            name="G115 terminal-stage identity",
        )
        if (
            type(qat["disagreement_pixels"]) is not int
            or not 0
            <= qat["disagreement_pixels"]
            <= EXACT_SEG_PIXEL_DENOMINATOR
            or qat["pixel_denominator"] != EXACT_SEG_PIXEL_DENOMINATOR
        ):
            raise G121StageHarvestError(
                "terminal G115-QAT exact count differs"
            )
        _validate_file_binding_shape(qat["receipt"], name="G115-QAT receipt")
    else:
        raise G121StageHarvestError("G115-QAT status is invalid")
    return qat


def _exact_obstruction(
    *,
    wire_disagreements: int,
    target: Fraction,
    source_disagreements: int | None,
    g115_qat: Mapping[str, object],
) -> dict[str, object]:
    lhs = 100 * wire_disagreements * target.denominator
    rhs = target.numerator * EXACT_SEG_PIXEL_DENOMINATOR
    wire_open = lhs < rhs
    if wire_open:
        disposition = RETAIN_POST_G105_POSE
    elif g115_qat["status"] == "terminal_stage_measured":
        terminal_lhs = (
            100
            * int(g115_qat["disagreement_pixels"])
            * target.denominator
        )
        disposition = (
            RETAIN_POST_G105_POSE
            if terminal_lhs < rhs
            else PRUNE_EXACT_DISTORTION_OBSTRUCTION
        )
    elif source_disagreements is None:
        disposition = DEFER_G115_WIRE_QAT
    else:
        source_lhs = 100 * source_disagreements * target.denominator
        source_open = source_lhs < rhs
        if not source_open:
            disposition = PRUNE_EXACT_DISTORTION_OBSTRUCTION
        elif g115_qat["status"] != "terminal_stage_measured":
            disposition = DEFER_G115_WIRE_QAT
        else:  # Handled before source-float classification.
            raise AssertionError("terminal G115-QAT branch was not consumed")
    return {
        "rule": EXACT_OBSTRUCTION_RULE,
        "lhs": str(lhs),
        "rhs": str(rhs),
        "strict_distortion_open": wire_open,
        "disposition": disposition,
    }


def _require_safe_g120_v2() -> Any:
    """Resolve only the G122 minimum-safe G120-v2 production surface."""

    module = importlib.import_module(
        "tac.witness_dsl.g120_parsed_stage_production_authority_v2"
    )
    expected_schemas = {
        "PRODUCTION_SCHEMA": G120_REQUIRED_PRODUCTION_SCHEMA,
        "MEASUREMENT_SCHEMA": G120_REQUIRED_MEASUREMENT_SCHEMA,
        "OBSERVATION_SCHEMA": G120_REQUIRED_OBSERVATION_SCHEMA,
        "EXACT_DISTORTION_OBSTRUCTION_SCHEMA": (
            G120_REQUIRED_SCOPED_OBSTRUCTION_SCHEMA
        ),
    }
    if any(getattr(module, name, None) != value for name, value in expected_schemas.items()):
        raise G121StageHarvestError(
            "G121 production refuses unsafe G120 f928/v1; "
            "minimum-safe G120-v2 has not landed"
        )
    if getattr(module, "MINIMUM_SAFE_AUTHORITY_COMPLETE", None) is not True:
        raise G121StageHarvestError(
            "G120-v2 is present but has not sealed its minimum-safe "
            "measurement/observation implementation"
        )
    runner = getattr(module, G120_REQUIRED_RUNNER, None)
    if not callable(runner):
        raise G121StageHarvestError(
            f"G120-v2 lacks required runner {G120_REQUIRED_RUNNER}"
        )
    expected = {
        "repo_root",
        "g112_partition_receipt",
        "expected_g112_partition_receipt_sha256",
        "out_dir",
        "progress_dir",
        "measurement_cache_dir",
        "prior_measurement_receipt",
        "expected_prior_measurement_receipt_sha256",
    }
    if set(inspect.signature(runner).parameters) != expected:
        raise G121StageHarvestError("G120-v2 runner signature differs")
    for opener_name in (
        G120_REQUIRED_MEASUREMENT_OPENER,
        G120_REQUIRED_OBSERVATION_OPENER,
        G120_REQUIRED_SCOPED_OBSTRUCTION_OPENER,
    ):
        if not callable(getattr(module, opener_name, None)):
            raise G121StageHarvestError(
                f"G120-v2 lacks required physical opener {opener_name}"
            )
    obstruction_error = getattr(
        module,
        G120_REQUIRED_SCOPED_OBSTRUCTION_ERROR,
        None,
    )
    if (
        not isinstance(obstruction_error, type)
        or not issubclass(obstruction_error, Exception)
    ):
        raise G121StageHarvestError(
            "G120-v2 lacks its typed exact scoped-obstruction error"
        )
    return module


def _physical_stage_identity(value: object) -> tuple[dict[str, Any], str, str]:
    required = {
        "g112_partition_receipt",
        "g112_semantic_child",
        "g112_pose_initializer",
        "g111_deploy_checkpoint",
        "g111_full_state_resume_checkpoint",
        "g111_fresh_lineage_receipt",
        "g111_checkpoint_id_sha256",
        "g111_stage",
        "g111_epoch",
        "fresh_lineage_complete",
    }
    if type(value) is not dict or set(value) != required:
        raise G121StageHarvestError(
            "physical stage identity has a noncanonical key set"
        )
    if value["fresh_lineage_complete"] is not True:
        raise G121StageHarvestError("physical lineage is not complete")
    _require_sha256(
        value["g111_checkpoint_id_sha256"],
        name="G111 checkpoint identity",
    )
    if (
        type(value["g111_stage"]) is not str
        or not value["g111_stage"]
        or type(value["g111_epoch"]) is not int
        or value["g111_epoch"] < 0
    ):
        raise G121StageHarvestError("G111 stage/epoch coordinate is invalid")
    for key in (
        "g112_partition_receipt",
        "g112_semantic_child",
        "g112_pose_initializer",
        "g111_deploy_checkpoint",
        "g111_full_state_resume_checkpoint",
        "g111_fresh_lineage_receipt",
    ):
        section = value[key]
        if type(section) is not dict or not {"path", "bytes", "sha256"}.issubset(
            section
        ):
            raise G121StageHarvestError(f"physical stage lacks {key} binding")
        if (
            type(section["path"]) is not str
            or not Path(section["path"]).is_absolute()
            or type(section["bytes"]) is not int
            or section["bytes"] <= 0
        ):
            raise G121StageHarvestError(f"physical {key} binding is invalid")
        _require_sha256(section["sha256"], name=f"physical {key} SHA-256")
    identity_sha = _sha256(_canonical_json(value))
    pose_identity = _require_sha256(
        value["g112_pose_initializer"]["sha256"],
        name="G112 pose initializer",
    )
    return value, identity_sha, pose_identity


def _validate_alternatives(
    value: object,
    *,
    selected_archive: Mapping[str, object],
) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != 4 or any(
        type(row) is not dict for row in value
    ):
        raise G121StageHarvestError(
            "G121 requires the exact four semantic archive alternatives"
        )
    identities: set[str] = set()
    matrix: set[tuple[str, str]] = set()
    selected_sha = _require_sha256(
        selected_archive.get("sha256"),
        name="selected archive SHA-256",
    )
    selected_seen = False
    for row in value:
        identity_value = row.get("alternative_identity_sha256")
        if identity_value is None:
            identity_value = row.get("row_identity_sha256")
        identity = _require_sha256(
            identity_value,
            name="semantic alternative identity",
        )
        codec = row.get("y1_wire_codec")
        method = row.get("outer_zip_method")
        if (
            type(codec) is not str
            or type(method) is not str
            or identity in identities
            or (codec, method) in matrix
        ):
            raise G121StageHarvestError(
                "semantic alternatives are duplicate or malformed"
            )
        identities.add(identity)
        matrix.add((codec, method))
        archive = row.get("archive")
        if type(archive) is dict and archive.get("sha256") == selected_sha:
            selected_seen = True
    expected_matrix = {
        ("RAW_I16_LE", "STORE"),
        ("RAW_I16_LE", "DEFLATE"),
        ("DELTA_RICE_BEST_K", "STORE"),
        ("DELTA_RICE_BEST_K", "DEFLATE"),
    }
    if matrix != expected_matrix:
        raise G121StageHarvestError(
            "semantic alternatives are not the exact 2x2 wire matrix"
        )
    if not selected_seen:
        raise G121StageHarvestError(
            "selected archive is absent from the exact alternatives"
        )
    return value


def _validate_runtime_tree(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise G121StageHarvestError("public runtime tree is not an object")
    if set(value) == {
        "source_pre",
        "sealed_tree_sha256",
        "source_post",
        "toctou_closed_by",
    }:
        source_pre = value["source_pre"]
        source_post = value["source_post"]
        if (
            type(source_pre) is not dict
            or type(source_post) is not dict
            or source_pre != source_post
            or value["toctou_closed_by"]
            != "sealed_content_execution_plus_source_postverify"
        ):
            raise G121StageHarvestError(
                "public runtime source changed across sealed execution"
            )
        content_sha = value["sealed_tree_sha256"]
        if content_sha != source_pre.get("tree_sha256"):
            raise G121StageHarvestError(
                "sealed runtime content tree differs"
            )
    else:
        # Fixture-only compact representation accepted by the private seam.
        content_sha = value.get(
            "content_tree_sha256",
            value.get("tree_sha256"),
        )
    _require_sha256(content_sha, name="public runtime content-tree identity")
    return value


def _compile_completed_attempt(
    raw: Mapping[str, object],
    *,
    require_physical_files: bool,
) -> dict[str, object]:
    """Normalize one safe G120-v2 result and recompute its exact disposition."""

    required = {
        "stage_tag",
        "measurement_receipt",
        "measurement_identity_sha256",
        "public_wire_seg",
        "physical_stage_identity",
        "physical_stage_identity_sha256",
        "pose_initializer_identity_sha256",
        "selected_archive",
        "alternatives",
        "source_float_seg",
        "wire_regret",
        "g115_qat",
        "live_target",
        "prepose_obstruction",
        "observation_receipt",
        "observation_identity_sha256",
        "production_receipt",
        "public_runtime_tree",
    }
    if type(raw) is not dict or set(raw) != required:
        raise G121StageHarvestError(
            "G120-v2 G121 adapter has a noncanonical key set"
        )
    stage_tag = _require_stage_tag(raw["stage_tag"])
    measurement_identity = _require_sha256(
        raw["measurement_identity_sha256"],
        name="G120 measurement identity",
    )
    observation_identity = _require_sha256(
        raw["observation_identity_sha256"],
        name="G120 observation identity",
    )
    physical, physical_sha, pose_identity = _physical_stage_identity(
        raw["physical_stage_identity"]
    )
    if (
        raw["physical_stage_identity_sha256"] != physical_sha
        or raw["pose_initializer_identity_sha256"] != pose_identity
    ):
        raise G121StageHarvestError("G120 physical-stage identity differs")
    public_wire, wire_disagreements = _exact_public_wire_seg(
        raw["public_wire_seg"],
        measurement_identity_sha256=measurement_identity,
    )
    live_target, target_fraction = _exact_live_target(raw["live_target"])
    source_float, source_disagreements = _validate_source_float(
        raw["source_float_seg"]
    )
    wire_regret = _validate_wire_regret(
        raw["wire_regret"],
        wire_disagreements=wire_disagreements,
        source_disagreements=source_disagreements,
    )
    g115_qat = _validate_g115_qat(raw["g115_qat"])
    recomputed_obstruction = _exact_obstruction(
        wire_disagreements=wire_disagreements,
        target=target_fraction,
        source_disagreements=source_disagreements,
        g115_qat=g115_qat,
    )
    if raw["prepose_obstruction"] != recomputed_obstruction:
        raise G121StageHarvestError(
            "G120 obstruction differs from G121 exact recomputation"
        )
    selected_archive = _validate_file_binding_shape(
        raw["selected_archive"],
        name="selected archive",
    )
    alternatives = _validate_alternatives(
        raw["alternatives"],
        selected_archive=selected_archive,
    )
    public_runtime_tree = _validate_runtime_tree(raw["public_runtime_tree"])
    production_receipt = _validate_file_binding_shape(
        raw["production_receipt"],
        name="G120 production receipt",
    )
    measurement_receipt = _validate_file_binding_shape(
        raw["measurement_receipt"],
        name="G120 measurement receipt",
    )
    observation_receipt = _validate_file_binding_shape(
        raw["observation_receipt"],
        name="G120 observation receipt",
    )
    if require_physical_files:
        for name, binding in (
            ("G120 production receipt", production_receipt),
            ("G120 measurement receipt", measurement_receipt),
            ("G120 observation receipt", observation_receipt),
            ("selected archive", selected_archive),
        ):
            _open_binding(binding, name=name)
        for name in (
            "g112_partition_receipt",
            "g112_semantic_child",
            "g112_pose_initializer",
            "g111_deploy_checkpoint",
            "g111_full_state_resume_checkpoint",
            "g111_fresh_lineage_receipt",
        ):
            section = physical[name]
            basic = {
                key: section[key]
                for key in ("path", "bytes", "sha256")
            }
            _open_binding(basic, name=f"physical {name}")
    scientific = {
        "stage_tag": stage_tag,
        "physical_stage_identity_sha256": physical_sha,
        "measurement_identity_sha256": measurement_identity,
        "observation_identity_sha256": observation_identity,
        "public_wire_seg": public_wire,
        "live_target": live_target,
        "prepose_obstruction": recomputed_obstruction,
        "selected_archive_sha256": selected_archive["sha256"],
    }
    row_identity = _sha256(_canonical_json(scientific))
    body: dict[str, object] = {
        "schema": STAGE_MEASUREMENT_SCHEMA,
        "attempt_status": "COMPLETED",
        "stage_tag": stage_tag,
        "row_identity_sha256": row_identity,
        "measurement_identity_sha256": measurement_identity,
        "observation_identity_sha256": observation_identity,
        "physical_stage_identity": physical,
        "physical_stage_identity_sha256": physical_sha,
        "g120": {
            "production_receipt": production_receipt,
            "measurement_receipt": measurement_receipt,
            "observation_receipt": observation_receipt,
        },
        "public_wire_seg": public_wire,
        "live_target": live_target,
        "prepose_obstruction": recomputed_obstruction,
        "source_float_seg": source_float,
        "wire_regret": wire_regret,
        "g115_qat": g115_qat,
        "four_way_alternatives": alternatives,
        "selected_archive": selected_archive,
        "public_runtime_tree": public_runtime_tree,
        # Compatibility displays only; no decision may consume these aliases.
        "d_seg_wire": public_wire["d_seg_display_float"],
        "live_target_score": float(target_fraction),
        "retained_for_post_g105_pose": (
            recomputed_obstruction["disposition"]
            == RETAIN_POST_G105_POSE
        ),
        "pose_initializer_identity_sha256": pose_identity,
    }
    return _seal(body, key="attempt_identity_sha256")


def _validate_completed_attempt(
    value: object,
    *,
    require_physical_files: bool,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise G121StageHarvestError("G121 completed attempt is not an object")
    sealed = dict(value)
    attempt_identity = _require_sha256(
        sealed.pop("attempt_identity_sha256", None),
        name="G121 attempt identity",
    )
    if attempt_identity != _sha256(_canonical_json(sealed)):
        raise G121StageHarvestError("G121 attempt self-hash differs")
    adapter = {
        "stage_tag": sealed["stage_tag"],
        "measurement_receipt": sealed["g120"]["measurement_receipt"],
        "measurement_identity_sha256": sealed["measurement_identity_sha256"],
        "public_wire_seg": sealed["public_wire_seg"],
        "physical_stage_identity": sealed["physical_stage_identity"],
        "physical_stage_identity_sha256": sealed[
            "physical_stage_identity_sha256"
        ],
        "pose_initializer_identity_sha256": sealed[
            "pose_initializer_identity_sha256"
        ],
        "selected_archive": sealed["selected_archive"],
        "alternatives": sealed["four_way_alternatives"],
        "source_float_seg": sealed["source_float_seg"],
        "wire_regret": sealed["wire_regret"],
        "g115_qat": sealed["g115_qat"],
        "live_target": sealed["live_target"],
        "prepose_obstruction": sealed["prepose_obstruction"],
        "observation_receipt": sealed["g120"]["observation_receipt"],
        "observation_identity_sha256": sealed[
            "observation_identity_sha256"
        ],
        "production_receipt": sealed["g120"]["production_receipt"],
        "public_runtime_tree": sealed["public_runtime_tree"],
    }
    normalized = _compile_completed_attempt(
        adapter,
        require_physical_files=require_physical_files,
    )
    if normalized != value:
        raise G121StageHarvestError("G121 completed attempt derivation differs")
    return value


def _source_stage_identity(
    value: Mapping[str, object],
) -> tuple[dict[str, Any], str]:
    """Validate pre-G112 custody without fabricating downstream bindings."""

    required = {
        "g111_deploy_checkpoint",
        "g111_full_state_resume_checkpoint",
        "g111_fresh_lineage_receipt",
        "g111_checkpoint_id_sha256",
        "g111_stage",
        "g111_epoch",
        "fresh_lineage_complete",
    }
    if type(value) is not dict or set(value) != required:
        raise G121StageHarvestError(
            "source-stage identity has a noncanonical key set"
        )
    if value["fresh_lineage_complete"] is not True:
        raise G121StageHarvestError("source-stage lineage is incomplete")
    _require_sha256(
        value["g111_checkpoint_id_sha256"],
        name="source checkpoint identity",
    )
    if (
        type(value["g111_stage"]) is not str
        or not value["g111_stage"]
        or type(value["g111_epoch"]) is not int
        or value["g111_epoch"] < 0
    ):
        raise G121StageHarvestError("source stage/epoch is invalid")
    for key in (
        "g111_deploy_checkpoint",
        "g111_full_state_resume_checkpoint",
        "g111_fresh_lineage_receipt",
    ):
        _validate_file_binding_shape(value[key], name=key)
    return value, _sha256(_canonical_json(value))


def _source_from_completed_physical(
    physical: Mapping[str, object],
) -> dict[str, object]:
    return {
        key: physical[key]
        for key in (
            "g111_deploy_checkpoint",
            "g111_full_state_resume_checkpoint",
            "g111_fresh_lineage_receipt",
            "g111_checkpoint_id_sha256",
            "g111_stage",
            "g111_epoch",
            "fresh_lineage_complete",
        )
    }


def _compile_blocked_attempt(
    *,
    stage_tag: str,
    source_stage_identity: Mapping[str, object],
    live_target: Mapping[str, object],
    blocker_code: str,
    blocker_detail: str,
    g120_scoped_obstruction: Mapping[str, object] | None = None,
) -> dict[str, object]:
    stage_tag = _require_stage_tag(stage_tag)
    source, source_sha = _source_stage_identity(source_stage_identity)
    normalized_target, _target_fraction = _exact_live_target(live_target)
    if (
        type(blocker_code) is not str
        or not blocker_code
        or type(blocker_detail) is not str
        or not blocker_detail
    ):
        raise G121StageHarvestError("scoped blocker is malformed")
    body = {
        "schema": STAGE_MEASUREMENT_SCHEMA,
        "attempt_status": "BLOCKED",
        "stage_tag": stage_tag,
        "source_stage_identity": source,
        "source_stage_identity_sha256": source_sha,
        "live_target": normalized_target,
        "prepose_obstruction": {
            "disposition": BLOCKED_SCOPED,
        },
        "blocker": {
            "code": blocker_code,
            "detail": blocker_detail,
            "verdict_scope": "one_physical_stage_attempt",
        },
    }
    if g120_scoped_obstruction is not None:
        binding = dict(g120_scoped_obstruction)
        _validate_file_binding_shape(
            binding,
            name="G120 scoped obstruction",
        )
        body["g120_scoped_obstruction"] = binding
    return _seal(body, key="attempt_identity_sha256")


def _validate_blocked_attempt(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise G121StageHarvestError("G121 blocked attempt is not an object")
    expected = {
        "schema",
        "attempt_status",
        "attempt_identity_sha256",
        "stage_tag",
        "source_stage_identity",
        "source_stage_identity_sha256",
        "live_target",
        "prepose_obstruction",
        "blocker",
    }
    if set(value) not in (
        expected,
        expected | {"g120_scoped_obstruction"},
    ):
        raise G121StageHarvestError(
            "G121 blocked attempt has a noncanonical key set"
        )
    sealed = dict(value)
    identity = _require_sha256(
        sealed.pop("attempt_identity_sha256"),
        name="blocked attempt identity",
    )
    if (
        value["schema"] != STAGE_MEASUREMENT_SCHEMA
        or value["attempt_status"] != "BLOCKED"
        or identity != _sha256(_canonical_json(sealed))
        or value["prepose_obstruction"] != {"disposition": BLOCKED_SCOPED}
    ):
        raise G121StageHarvestError("G121 blocked attempt differs")
    source, source_sha = _source_stage_identity(
        value["source_stage_identity"]
    )
    if source is not value["source_stage_identity"] or (
        source_sha != value["source_stage_identity_sha256"]
    ):
        raise G121StageHarvestError("blocked source-stage identity differs")
    _exact_live_target(value["live_target"])
    blocker = _require_exact_dict(
        value["blocker"],
        keys={"code", "detail", "verdict_scope"},
        name="scoped blocker",
    )
    if (
        type(blocker["code"]) is not str
        or not blocker["code"]
        or type(blocker["detail"]) is not str
        or not blocker["detail"]
        or blocker["verdict_scope"] != "one_physical_stage_attempt"
    ):
        raise G121StageHarvestError("scoped blocker differs")
    scoped_binding = value.get("g120_scoped_obstruction")
    if scoped_binding is not None:
        _validate_file_binding_shape(
            scoped_binding,
            name="G120 scoped obstruction",
        )
        g120 = _require_safe_g120_v2()
        opened = getattr(
            g120,
            G120_REQUIRED_SCOPED_OBSTRUCTION_OPENER,
        )(
            Path(str(scoped_binding["path"])),
            expected_sha256=str(scoped_binding["sha256"]),
        )
        if (
            opened.receipt["live_target"] != value["live_target"]
            or _source_from_completed_physical(
                opened.receipt["physical_stage_identity"]
            )
            != value["source_stage_identity"]
        ):
            raise G121StageHarvestError(
                "G120 scoped obstruction differs from blocked stage custody"
            )
    return value


def _open_reusable_scoped_obstruction(
    *,
    g120_module: Any,
    prior: Mapping[str, object] | None,
    stage: Mapping[str, object],
) -> Any | None:
    """Reuse only an exact G120 blocker whose pointer snapshot is still live."""

    if (
        prior is None
        or prior.get("attempt_status") != "BLOCKED"
        or "g120_scoped_obstruction" not in prior
        or prior.get("source_stage_identity")
        != stage.get("source_stage_identity")
    ):
        return None
    binding = prior["g120_scoped_obstruction"]
    _validate_file_binding_shape(
        binding,
        name="reusable G120 scoped obstruction",
    )
    opened = getattr(
        g120_module,
        G120_REQUIRED_SCOPED_OBSTRUCTION_OPENER,
    )(
        Path(str(binding["path"])),
        expected_sha256=str(binding["sha256"]),
    )
    try:
        snapshot = DynamicFrontierTargetSnapshot(
            **opened.receipt["pointer_snapshot"]
        )
    except (KeyError, TypeError) as exc:
        raise G121StageHarvestError(
            "reusable G120 scoped obstruction has a malformed pointer snapshot"
        ) from exc
    try:
        verify_dynamic_frontier_target_snapshot(snapshot)
    except DynamicFrontierTargetError:
        return None
    if (
        opened.receipt["live_target"] != prior["live_target"]
        or _source_from_completed_physical(
            opened.receipt["physical_stage_identity"]
        )
        != prior["source_stage_identity"]
    ):
        raise G121StageHarvestError(
            "reusable G120 scoped obstruction differs from prior custody"
        )
    return opened


def _scoped_obstruction_scorer_replay_count(opened: Any) -> int:
    """Count one stage replay iff this obstruction made fresh scorer calls."""

    try:
        calls = opened.receipt["seg_scorer"][
            "fresh_direct_scorer_calls"
        ]
    except (AttributeError, KeyError, TypeError) as exc:
        raise G121StageHarvestError(
            "G120 scoped obstruction lacks exact scorer replay telemetry"
        ) from exc
    if type(calls) is not int or calls < 0:
        raise G121StageHarvestError(
            "G120 scoped obstruction scorer replay telemetry differs"
        )
    return int(calls > 0)


def _read_stage_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise G121StageHarvestError("stage ledger is not a regular file")
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise G121StageHarvestError("stage ledger has a torn final row")
    rows: list[dict[str, Any]] = []
    seen_attempts: dict[str, bytes] = {}
    for index, line in enumerate(payload.splitlines(keepends=True)):
        try:
            value = json.loads(line.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise G121StageHarvestError(
                f"stage ledger row {index} is corrupt"
            ) from exc
        if type(value) is not dict or _canonical_json(value) != line:
            raise G121StageHarvestError(
                f"stage ledger row {index} is noncanonical"
            )
        if value.get("attempt_status") == "COMPLETED":
            _validate_completed_attempt(value, require_physical_files=False)
        elif value.get("attempt_status") == "BLOCKED":
            _validate_blocked_attempt(value)
        else:
            raise G121StageHarvestError(
                f"stage ledger row {index} has an invalid status"
            )
        identity = _require_sha256(
            value.get("attempt_identity_sha256"),
            name="ledger attempt identity",
        )
        previous = seen_attempts.get(identity)
        if previous is not None and previous != line:
            raise G121StageHarvestError(
                "stage ledger has a conflicting duplicate identity"
            )
        if previous is None:
            seen_attempts[identity] = line
            rows.append(value)
    return rows


def _append_attempt(path: Path, row: Mapping[str, object]) -> bool:
    payload = _canonical_json(row)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing = _read_stage_ledger(path)
        identity = row["attempt_identity_sha256"]
        if any(item["attempt_identity_sha256"] == identity for item in existing):
            return False
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return True


def open_g121_retained_prepose_v1(
    path: Path,
    expected_sha256: str,
) -> G121RetainedPreposeV2:
    """Strictly reopen ``g121_retained_prepose.json`` by external SHA.

    The implementation body is filled below; the public opener deliberately
    checks the minimum-safe G120-v2 code surface before granting downstream
    production custody.
    """

    _require_safe_g120_v2()
    return _open_g121_retained_prepose_impl(
        path,
        expected_sha256=expected_sha256,
        require_physical=True,
    )


def harvest_g111_stages_v1(
    *,
    producer_run_dir: Path,
    expected_launch_manifest_sha256: str,
    output_dir: Path,
    progress_dir: Path,
) -> G121StageHarvestResultV1:
    """Finalize every preserved G111 stage after terminal-tip proof."""

    g120 = _require_safe_g120_v2()
    result = _harvest_g111_stages_impl(
        producer_run_dir=producer_run_dir,
        expected_launch_manifest_sha256=expected_launch_manifest_sha256,
        output_dir=output_dir,
        progress_dir=progress_dir,
        g120_module=g120,
        finalize=True,
    )
    if not isinstance(result, G121StageHarvestResultV1):
        raise AssertionError("final G121 harvest returned incremental progress")
    return result


def harvest_g111_available_stages_v1(
    *,
    producer_run_dir: Path,
    expected_launch_manifest_sha256: str,
    output_dir: Path,
    progress_dir: Path,
) -> G121StageHarvestProgressV1:
    """Measure the currently preserved stage set without claiming completion.

    This is the only production entrypoint intended while G111 is still live.
    It may append idempotent stage attempts, but it cannot emit either the
    exhaustive completion receipt or the retained-prepose manifest.
    """

    g120 = _require_safe_g120_v2()
    result = _harvest_g111_stages_impl(
        producer_run_dir=producer_run_dir,
        expected_launch_manifest_sha256=expected_launch_manifest_sha256,
        output_dir=output_dir,
        progress_dir=progress_dir,
        g120_module=g120,
        finalize=False,
    )
    if not isinstance(result, G121StageHarvestProgressV1):
        raise AssertionError("incremental G121 harvest returned a final result")
    return result


def _open_g121_retained_prepose_impl(
    path: Path,
    *,
    expected_sha256: str,
    require_physical: bool,
) -> G121RetainedPreposeV2:
    requested = Path(path).expanduser()
    if (
        not requested.is_absolute()
        or requested.name != RETAINED_PREPOSE_BASENAME
    ):
        raise G121StageHarvestError(
            f"G121 opener accepts only absolute {RETAINED_PREPOSE_BASENAME}"
        )
    payload, identity = _stable_regular_file(
        requested,
        name="G121 retained-prepose manifest",
        expected_sha256=expected_sha256,
    )
    manifest = _parse_json(payload, name="G121 retained-prepose manifest")
    exact_keys = {
        "schema",
        "status",
        "rows",
        "completion_receipt",
        "stage_ledger",
        "exhaustive_enumeration_proven",
        "pointer_snapshot_identity_sha256",
        "postverified_pointer_identity_sha256",
        "live_target",
        "live_target_score",
        "counts",
        "research_only",
        "score_claim",
        "evaluation_claim",
        "promotion_eligible",
        "pointer_moved",
        "manifest_identity_sha256",
    }
    if set(manifest) != exact_keys:
        raise G121StageHarvestError(
            "G121 retained-prepose manifest has a noncanonical key set"
        )
    unsealed = dict(manifest)
    sealed_identity = _require_sha256(
        unsealed.pop("manifest_identity_sha256"),
        name="G121 manifest identity",
    )
    if sealed_identity != _sha256(_canonical_json(unsealed)):
        raise G121StageHarvestError("G121 manifest self-hash differs")
    if (
        manifest["schema"] != RETAINED_PREPOSE_SCHEMA
        or manifest["status"] != "EXHAUSTIVE_STAGE_HARVEST_COMPLETE"
        or manifest["exhaustive_enumeration_proven"] is not True
        or manifest["research_only"] is not True
        or manifest["score_claim"] is not False
        or manifest["evaluation_claim"] is not False
        or manifest["promotion_eligible"] is not False
        or manifest["pointer_moved"] is not False
    ):
        raise G121StageHarvestError(
            "G121 manifest authority/exhaustiveness fields differ"
        )
    live_target, target_fraction = _exact_live_target(manifest["live_target"])
    if (
        manifest["pointer_snapshot_identity_sha256"]
        != live_target["pointer_snapshot_identity_sha256"]
        or manifest["postverified_pointer_identity_sha256"]
        != live_target["postverified_pointer_identity_sha256"]
        or type(manifest["live_target_score"]) is not float
        or manifest["live_target_score"] != float(target_fraction)
    ):
        raise G121StageHarvestError("G121 manifest target aliases differ")
    completion_payload, completion_binding = _open_binding(
        manifest["completion_receipt"],
        name="G121 completion receipt",
    )
    completion = _parse_json(
        completion_payload,
        name="G121 completion receipt",
    )
    _validate_completion_receipt(
        completion,
        live_target=live_target,
        manifest_rows=manifest["rows"],
    )
    ledger_payload, ledger_binding = _open_binding(
        manifest["stage_ledger"],
        name="G121 stage ledger",
    )
    if _sha256(ledger_payload) != ledger_binding["sha256"]:
        raise AssertionError("stable G121 ledger binding changed")
    ledger_rows = _read_stage_ledger(Path(str(ledger_binding["path"])))
    if (
        completion["stage_ledger"] != ledger_binding
        or completion["counts"] != manifest["counts"]
    ):
        raise G121StageHarvestError(
            "completion/manifest ledger or count binding differs"
        )
    by_attempt = {
        row["attempt_identity_sha256"]: row
        for row in ledger_rows
    }
    accounted_ids = completion["accounted_attempt_identity_sha256"]
    if any(identity_value not in by_attempt for identity_value in accounted_ids):
        raise G121StageHarvestError(
            "completion accounts an attempt absent from the ledger"
        )
    accounted_rows = [by_attempt[identity_value] for identity_value in accounted_ids]
    if any(row["live_target"] != live_target for row in accounted_rows):
        raise G121StageHarvestError(
            "completion accounts a different pointer observation"
        )
    eligible_identities = sorted(
        (
            row["physical_stage_identity_sha256"]
            if row["attempt_status"] == "COMPLETED"
            else row["source_stage_identity_sha256"]
        )
        for row in accounted_rows
    )
    if eligible_identities != completion[
        "eligible_physical_stage_identity_sha256"
    ]:
        raise G121StageHarvestError(
            "completion eligible/accounted stage sets differ"
        )
    rows_value = manifest["rows"]
    if type(rows_value) is not list:
        raise G121StageHarvestError("G121 retained rows are not a list")
    opened_rows: list[G121RetainedStageV2] = []
    seen_rows: set[str] = set()
    g120_module = _require_safe_g120_v2() if require_physical else None
    for raw_row in rows_value:
        row = _validate_completed_attempt(
            raw_row,
            require_physical_files=require_physical,
        )
        obstruction = row["prepose_obstruction"]
        if (
            obstruction["disposition"] != RETAIN_POST_G105_POSE
            or obstruction["strict_distortion_open"] is not True
            or row["live_target"] != live_target
            or row["retained_for_post_g105_pose"] is not True
        ):
            raise G121StageHarvestError(
                "retained manifest contains a non-retained stage"
            )
        attempt_identity = row["attempt_identity_sha256"]
        if (
            attempt_identity in seen_rows
            or by_attempt.get(attempt_identity) != row
        ):
            raise G121StageHarvestError(
                "retained row is duplicate or absent from append-only ledger"
            )
        seen_rows.add(attempt_identity)
        if require_physical:
            assert g120_module is not None
            measurement_binding = row["g120"]["measurement_receipt"]
            observation_binding = row["g120"]["observation_receipt"]
            reopened_measurement = getattr(
                g120_module,
                G120_REQUIRED_MEASUREMENT_OPENER,
            )(
                Path(str(measurement_binding["path"])),
                expected_sha256=str(measurement_binding["sha256"]),
            )
            reopened_observation = getattr(
                g120_module,
                G120_REQUIRED_OBSERVATION_OPENER,
            )(
                Path(str(observation_binding["path"])),
                expected_sha256=str(observation_binding["sha256"]),
            )
            if (
                reopened_measurement.measurement_identity_sha256
                != row["measurement_identity_sha256"]
                or reopened_observation.observation_identity_sha256
                != row["observation_identity_sha256"]
            ):
                raise G121StageHarvestError(
                    "G120 recursive receipt identity differs"
                )
            from tac.witness_control.taskspace_g112_exact_checkpoint_partition_v1 import (
                open_g112_partition_receipt,
            )

            g112 = row["physical_stage_identity"][
                "g112_partition_receipt"
            ]
            open_g112_partition_receipt(
                Path(str(g112["path"])),
                expected_sha256=str(g112["sha256"]),
            )
        opened_rows.append(G121RetainedStageV2(value=dict(row)))
    counts = _require_exact_dict(
        manifest["counts"],
        keys={"accounted", "blocked", "deferred", "discovered", "pruned", "retained"},
        name="G121 manifest counts",
    )
    if counts["retained"] != len(opened_rows):
        raise G121StageHarvestError("G121 retained count differs")
    return G121RetainedPreposeV2(
        schema=RETAINED_PREPOSE_SCHEMA,
        rows=tuple(opened_rows),
        completion_receipt=completion_binding,
        exhaustive_enumeration_proven=True,
        manifest_path=Path(str(identity["path"])),
        manifest_sha256=str(identity["sha256"]),
        pointer_snapshot_identity_sha256=str(
            live_target["pointer_snapshot_identity_sha256"]
        ),
        postverified_pointer_identity_sha256=str(
            live_target["postverified_pointer_identity_sha256"]
        ),
        live_target_score=float(target_fraction),
        live_target_score_decimal=str(live_target["score_decimal"]),
        live_target_numerator=target_fraction.numerator,
        live_target_denominator=target_fraction.denominator,
    )


def _validate_completion_receipt(
    value: object,
    *,
    live_target: Mapping[str, object],
    manifest_rows: object,
) -> dict[str, Any]:
    exact_keys = {
        "schema",
        "status",
        "launch_manifest",
        "stage_ledger",
        "live_target",
        "eligible_physical_stage_identity_sha256",
        "accounted_attempt_identity_sha256",
        "retained_attempt_identity_sha256",
        "counts",
        "all_eligible_stages_accounted",
        "exhaustive_enumeration_proven",
        "research_only",
        "score_claim",
        "evaluation_claim",
        "promotion_eligible",
        "pointer_moved",
        "completion_identity_sha256",
    }
    if type(value) is not dict or set(value) != exact_keys:
        raise G121StageHarvestError(
            "G121 completion receipt has a noncanonical key set"
        )
    unsealed = dict(value)
    identity = _require_sha256(
        unsealed.pop("completion_identity_sha256"),
        name="G121 completion identity",
    )
    if identity != _sha256(_canonical_json(unsealed)):
        raise G121StageHarvestError("G121 completion self-hash differs")
    if (
        value["schema"] != COMPLETION_RECEIPT_SCHEMA
        or value["status"] != "EXHAUSTIVE_STAGE_HARVEST_COMPLETE"
        or value["live_target"] != live_target
        or value["all_eligible_stages_accounted"] is not True
        or value["exhaustive_enumeration_proven"] is not True
        or value["research_only"] is not True
        or value["score_claim"] is not False
        or value["evaluation_claim"] is not False
        or value["promotion_eligible"] is not False
        or value["pointer_moved"] is not False
    ):
        raise G121StageHarvestError(
            "G121 completion authority/exhaustiveness fields differ"
        )
    _validate_file_binding_shape(
        value["launch_manifest"],
        name="launch manifest",
    )
    _validate_file_binding_shape(value["stage_ledger"], name="stage ledger")
    eligible = value["eligible_physical_stage_identity_sha256"]
    accounted = value["accounted_attempt_identity_sha256"]
    retained = value["retained_attempt_identity_sha256"]
    if (
        type(eligible) is not list
        or type(accounted) is not list
        or type(retained) is not list
        or eligible != sorted(set(eligible))
        or accounted != sorted(set(accounted))
        or retained != sorted(set(retained))
    ):
        raise G121StageHarvestError(
            "G121 completion identity sets are not sorted unique lists"
        )
    for name, values in (
        ("eligible stage", eligible),
        ("accounted attempt", accounted),
        ("retained attempt", retained),
    ):
        for item in values:
            _require_sha256(item, name=f"{name} identity")
    if type(manifest_rows) is not list or retained != sorted(
        row.get("attempt_identity_sha256")
        for row in manifest_rows
        if type(row) is dict
    ):
        raise G121StageHarvestError(
            "completion retained set differs from manifest rows"
        )
    counts = _require_exact_dict(
        value["counts"],
        keys={"accounted", "blocked", "deferred", "discovered", "pruned", "retained"},
        name="completion counts",
    )
    if (
        any(type(item) is not int or item < 0 for item in counts.values())
        or counts["discovered"] != len(eligible)
        or counts["accounted"] != len(accounted)
        or counts["retained"] != len(retained)
        or counts["accounted"] != counts["discovered"]
        or counts["accounted"]
        != counts["retained"]
        + counts["deferred"]
        + counts["pruned"]
        + counts["blocked"]
    ):
        raise G121StageHarvestError("G121 completion counts differ")
    return value


def _harvest_g111_stages_impl(
    *,
    producer_run_dir: Path,
    expected_launch_manifest_sha256: str,
    output_dir: Path,
    progress_dir: Path,
    g120_module: Any,
    finalize: bool,
) -> G121StageHarvestResultV1 | G121StageHarvestProgressV1:
    producer = _durable_directory(producer_run_dir, name="producer_run_dir")
    output = _durable_directory(output_dir, name="output_dir")
    progress = _durable_directory(progress_dir, name="progress_dir")
    if len({producer, output, progress}) != 3:
        raise G121StageHarvestError(
            "producer/output/progress directories must be distinct"
        )
    launch_binding, compile_hash = _open_governed_launch_manifest(
        producer,
        expected_sha256=expected_launch_manifest_sha256,
    )
    stages = _discover_physical_stages(
        producer,
        expected_current_launch_dsl_compile_hash=compile_hash,
    )
    if not stages:
        raise G121StageHarvestError(
            "no immutable fresh-lineage G111 stages were discovered"
        )
    terminal_binding: dict[str, object] | None = None
    if finalize:
        terminal_binding = _open_terminal_producer_result(
            producer,
            expected_current_launch_dsl_compile_hash=compile_hash,
            expected_final_checkpoint_id=str(
                stages[-1]["source_stage_identity"][
                    "g111_checkpoint_id_sha256"
                ]
            ),
        )
    ledger_path = output / STAGE_LEDGER_BASENAME
    existing = _read_stage_ledger(ledger_path)
    latest_by_checkpoint: dict[str, dict[str, Any]] = {}
    for row in existing:
        identity = (
            row["physical_stage_identity"]
            if row["attempt_status"] == "COMPLETED"
            else row["source_stage_identity"]
        )
        latest_by_checkpoint[
            identity["g111_checkpoint_id_sha256"]
        ] = row
    measurement_cache = _durable_directory(
        producer / "g121_measurement_cache",
        name="G121 measurement cache",
    )
    stage_rows: list[dict[str, Any]] = []
    scorer_replay_count = 0
    reused_measurement_count = 0
    pending_blockers: list[
        tuple[dict[str, Any], str, str]
    ] = []
    scoped_obstruction_error = getattr(
        g120_module,
        G120_REQUIRED_SCOPED_OBSTRUCTION_ERROR,
    )
    for stage in stages:
        checkpoint_id = stage["source_stage_identity"][
            "g111_checkpoint_id_sha256"
        ]
        prior = latest_by_checkpoint.get(checkpoint_id)
        reusable_obstruction = _open_reusable_scoped_obstruction(
            g120_module=g120_module,
            prior=prior,
            stage=stage,
        )
        if reusable_obstruction is not None:
            if prior is None:
                raise AssertionError(
                    "reusable obstruction exists without a prior row"
                )
            stage_rows.append(prior)
            continue
        try:
            g112_binding = _materialize_or_open_g112_stage(
                producer=producer,
                stage=stage,
            )
            prior_measurement: dict[str, Any] | None = None
            if prior is not None and prior.get("attempt_status") == "COMPLETED":
                prior_measurement = prior["g120"]["measurement_receipt"]
            stage_token = (
                f"{stage['stage_tag']}.{checkpoint_id[:16]}"
            )
            result = getattr(g120_module, G120_REQUIRED_RUNNER)(
                repo_root=Path(__file__).resolve().parents[3],
                g112_partition_receipt=Path(
                    str(g112_binding["path"])
                ),
                expected_g112_partition_receipt_sha256=str(
                    g112_binding["sha256"]
                ),
                out_dir=_durable_directory(
                    output / "g120" / stage_token,
                    name="G120 stage output",
                ),
                progress_dir=_durable_directory(
                    progress / "g120" / stage_token,
                    name="G120 stage progress",
                ),
                measurement_cache_dir=measurement_cache,
                prior_measurement_receipt=(
                    Path(str(prior_measurement["path"]))
                    if prior_measurement is not None
                    else None
                ),
                expected_prior_measurement_receipt_sha256=(
                    str(prior_measurement["sha256"])
                    if prior_measurement is not None
                    else None
                ),
            )
            adapter = result.to_g121_stage_measurement_v2()
            if type(adapter) is not dict:
                raise G121StageHarvestError(
                    "G120-v2 adapter did not return an exact dict"
                )
            adapter = dict(adapter)
            adapter["production_receipt"] = {
                "path": str(result.production_receipt_path),
                "bytes": result.production_receipt_path.stat().st_size,
                "sha256": result.production_receipt_sha256,
            }
            runtime = result.measurement.receipt.get("public_runtime")
            if type(runtime) is not dict:
                raise G121StageHarvestError(
                    "G120-v2 measurement lacks public runtime content tree"
                )
            adapter["public_runtime_tree"] = dict(runtime)
            public_wire = adapter.get("public_wire_seg")
            if type(public_wire) is not dict:
                raise G121StageHarvestError(
                    "G120-v2 adapter lacks public-wire Seg counts"
                )
            public_wire = dict(public_wire)
            public_wire.setdefault(
                "measurement_identity_sha256",
                adapter.get("measurement_identity_sha256"),
            )
            adapter["public_wire_seg"] = public_wire
            row = _compile_completed_attempt(
                adapter,
                require_physical_files=True,
            )
            if prior_measurement is None:
                scorer_replay_count += 1
            else:
                reused_measurement_count += 1
            _append_attempt(ledger_path, row)
            stage_rows.append(row)
        except scoped_obstruction_error as exc:
            opened = getattr(
                g120_module,
                G120_REQUIRED_SCOPED_OBSTRUCTION_OPENER,
            )(
                exc.receipt_path,
                expected_sha256=exc.receipt_sha256,
            )
            scoped_binding = {
                "path": str(opened.receipt_path),
                "bytes": opened.receipt_bytes,
                "sha256": opened.receipt_sha256,
            }
            row = _compile_blocked_attempt(
                stage_tag=str(stage["stage_tag"]),
                source_stage_identity=stage["source_stage_identity"],
                live_target=opened.receipt["live_target"],
                blocker_code=type(exc).__name__,
                blocker_detail=(
                    str(exc) or "exact scoped obstruction"
                ),
                g120_scoped_obstruction=scoped_binding,
            )
            _append_attempt(ledger_path, row)
            stage_rows.append(row)
            scorer_replay_count += (
                _scoped_obstruction_scorer_replay_count(opened)
            )
        except Exception as exc:
            pending_blockers.append(
                (
                    stage,
                    type(exc).__name__,
                    str(exc) or "stage attempt failed closed",
                )
            )
    if not stage_rows:
        raise G121StageHarvestError(
            "all physical stages blocked before an exact live-target "
            "observation could be established"
        )
    live_targets = {
        _canonical_json(row["live_target"])
        for row in stage_rows
    }
    if len(live_targets) != 1:
        raise G121StageHarvestError(
            "competitive pointer changed during harvest; rerun reuses "
            "external measurement receipts without scorer replay"
        )
    live_target = stage_rows[0]["live_target"]
    for stage, blocker_code, blocker_detail in pending_blockers:
        row = _compile_blocked_attempt(
            stage_tag=str(stage["stage_tag"]),
            source_stage_identity=stage["source_stage_identity"],
            live_target=live_target,
            blocker_code=blocker_code,
            blocker_detail=blocker_detail,
        )
        _append_attempt(ledger_path, row)
        stage_rows.append(row)
    if not finalize:
        ledger_binding = _binding_from_path(
            ledger_path,
            name="G121 incremental stage ledger",
        )
        return G121StageHarvestProgressV1(
            stage_ledger_path=ledger_path,
            stage_ledger_sha256=str(ledger_binding["sha256"]),
            discovered_stage_count=len(stages),
            accounted_stage_count=len(stage_rows),
            scorer_replay_count=scorer_replay_count,
            reused_measurement_count=reused_measurement_count,
        )
    stages_after = _discover_physical_stages(
        producer,
        expected_current_launch_dsl_compile_hash=compile_hash,
    )
    before_ids = [
        str(row["source_stage_identity"]["g111_checkpoint_id_sha256"])
        for row in stages
    ]
    after_ids = [
        str(row["source_stage_identity"]["g111_checkpoint_id_sha256"])
        for row in stages_after
    ]
    if before_ids != after_ids:
        raise G121StageHarvestError(
            "eligible preserved-stage census changed during final harvest; "
            "rerun reuses completed measurements"
        )
    reopened_terminal = _open_terminal_producer_result(
        producer,
        expected_current_launch_dsl_compile_hash=compile_hash,
        expected_final_checkpoint_id=after_ids[-1],
    )
    if reopened_terminal != terminal_binding:
        raise G121StageHarvestError(
            "producer terminal receipt changed during final harvest"
        )
    return _publish_reductions(
        output_dir=output,
        ledger_path=ledger_path,
        launch_binding=launch_binding,
        live_target=live_target,
        stage_rows=stage_rows,
        scorer_replay_count=scorer_replay_count,
        reused_measurement_count=reused_measurement_count,
    )


def _open_governed_launch_manifest(
    producer: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, object], str]:
    manifest_path = producer / "launch_manifest.json"
    payload, binding = _stable_regular_file(
        manifest_path,
        name="governed launch manifest",
        expected_sha256=expected_sha256,
    )
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G121StageHarvestError("launch manifest is corrupt") from exc
    required = {
        "schema",
        "config_family",
        "spec_id",
        "dsl_compile_hash",
        "launch_sh",
        "launch_sh_sha256",
        "dsl_provenance",
        "dsl_provenance_sha256",
        "resolved_launch_argv",
        "non_authoritative_context",
    }
    if type(manifest) is not dict or set(manifest) != required:
        raise G121StageHarvestError(
            "governed launch manifest key set differs"
        )
    if (
        manifest["schema"] != "witness_launch_manifest.v1"
        or manifest["config_family"] != "g111_batch16_v9_semantic_base"
        or manifest["spec_id"] != "g111_batch16_v9_semantic_base"
    ):
        raise G121StageHarvestError("launch manifest is not the G111 program")
    compile_hash = _require_sha256(
        manifest["dsl_compile_hash"],
        name="launch DSL compile hash",
    )
    launch_sh = producer / str(manifest["launch_sh"])
    provenance = producer / str(manifest["dsl_provenance"])
    for path, expected, name in (
        (launch_sh, manifest["launch_sh_sha256"], "launch.sh"),
        (
            provenance,
            manifest["dsl_provenance_sha256"],
            "DSL provenance",
        ),
    ):
        _stable_regular_file(
            path,
            name=name,
            expected_sha256=_require_sha256(expected, name=f"{name} SHA-256"),
        )
    from tac.v9_provenance_gates import verify_dsl_provenance_artifacts

    ok, detail = verify_dsl_provenance_artifacts(
        launch_sh,
        provenance_path=provenance,
        launch_manifest_path=manifest_path,
        expected_hash=compile_hash,
    )
    if not ok:
        raise G121StageHarvestError(
            f"governed launch provenance failed: {detail}"
        )
    _validate_g111_launch_argv(manifest["resolved_launch_argv"])
    return binding, compile_hash


def _validate_g111_launch_argv(value: object) -> None:
    if (
        type(value) is not list
        or not value
        or any(type(token) is not str or not token for token in value)
    ):
        raise G121StageHarvestError("resolved G111 argv is malformed")
    argv = list(value)
    forbidden = {
        "--verdict-pairs": "0",
    }
    required_values = {
        "--num-pairs": "600",
        "--verdict-batch": "16",
        "--activation": "hosc",
        "--mod-dim": "32",
        "--basis": "legacy_fourier_ab_control",
        "--render-h": "384",
        "--render-w": "512",
        "--render-aa": "none",
        "--pose-carrier-source": "generated_y1",
    }
    parsed: dict[str, str] = {}
    flags: set[str] = set()
    index = 0
    while index < len(argv):
        token = argv[index]
        if token.startswith("--"):
            if token in flags:
                raise G121StageHarvestError(
                    f"duplicate governed launch flag: {token}"
                )
            flags.add(token)
            if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                parsed[token] = argv[index + 1]
                index += 2
                continue
        index += 1
    for flag, expected in required_values.items():
        if parsed.get(flag) != expected:
            raise G121StageHarvestError(
                f"governed launch differs at {flag}"
            )
    if parsed.get("--verdict-pairs") != forbidden["--verdict-pairs"]:
        raise G121StageHarvestError("G111 verdict population is not all n600")
    for flag in ("--fresh-producer", "--no-self-orient"):
        if flag not in flags:
            raise G121StageHarvestError(f"governed launch lacks {flag}")
    if "--self-orient" in flags:
        raise G121StageHarvestError("governed launch enables self-orient")
    if "--training-target-capsule" not in parsed:
        raise G121StageHarvestError(
            "governed launch lacks physical G109 target capsule"
        )
    if "--training-target-capsule-sha256" not in parsed:
        raise G121StageHarvestError(
            "governed launch lacks external G109 capsule SHA"
        )
    _require_sha256(
        parsed["--training-target-capsule-sha256"],
        name="G109 capsule SHA-256",
    )


def _discover_physical_stages(
    producer: Path,
    *,
    expected_current_launch_dsl_compile_hash: str,
) -> list[dict[str, Any]]:
    lineage_dir = producer / "fresh_lineage"
    if lineage_dir.is_symlink() or not lineage_dir.is_dir():
        raise G121StageHarvestError("fresh_lineage directory is absent")
    receipt_paths = sorted(lineage_dir.glob("*.receipt.json"))
    if not receipt_paths:
        return []
    for receipt_path in receipt_paths:
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise G121StageHarvestError(
                "fresh-lineage enumeration found a non-regular receipt"
            )
    tip_payload, _tip_binding = _stable_regular_file(
        producer / "fresh_lineage_tip.json",
        name="fresh-lineage tip",
    )
    try:
        tip = json.loads(tip_payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G121StageHarvestError("fresh-lineage tip is corrupt") from exc
    tip_keys = {
        "schema",
        "receipt_path",
        "receipt_sha256",
        "receipt_bytes",
        "checkpoint_id_sha256",
        "root_sha256",
        "sequence_index",
        "epoch",
        "stage",
        "complete_trajectory_proven",
    }
    if (
        type(tip) is not dict
        or set(tip) != tip_keys
        or tip["schema"] != "tac.fresh_producer_lineage_tip.v1"
        or tip["complete_trajectory_proven"] is not True
    ):
        raise G121StageHarvestError("fresh-lineage tip schema or custody differs")
    from tac.witness_control.fresh_producer_lineage_v1 import (
        FreshProducerPhysicalCheckpointChainV1,
        open_fresh_physical_checkpoint_chain_v1,
    )

    tip_receipt = Path(str(tip["receipt_path"]))
    if not tip_receipt.is_absolute():
        raise G121StageHarvestError("fresh-lineage tip receipt path is not absolute")
    chain = open_fresh_physical_checkpoint_chain_v1(
        tip_receipt,
        expected_receipt_sha256=_require_sha256(
            tip["receipt_sha256"],
            name="fresh-lineage tip receipt SHA-256",
        ),
        expected_current_launch_dsl_compile_hash=(
            expected_current_launch_dsl_compile_hash
        ),
    )
    current = chain.current
    if (
        tip_receipt.resolve() != current.receipt_path.resolve()
        or tip["receipt_bytes"] != current.receipt_bytes
        or tip["checkpoint_id_sha256"] != current.pair.checkpoint_id_sha256
        or tip["root_sha256"] != chain.root_sha256
        or tip["sequence_index"] != current.sequence_index
        or tip["epoch"] != current.pair.epoch
        or tip["stage"] != current.pair.stage
    ):
        raise G121StageHarvestError(
            "fresh-lineage tip differs from its recursively reopened chain"
        )
    enumerated_receipts = {path.resolve() for path in receipt_paths}
    ancestry_receipts = {node.receipt_path.resolve() for node in chain.nodes}
    if enumerated_receipts != ancestry_receipts:
        raise G121StageHarvestError(
            "fresh-lineage receipt census is not the unique current-tip ancestry"
        )
    preserved = _preserved_stage_checkpoint_ids(producer, chain.nodes)

    rows: list[dict[str, Any]] = []
    for index, node in enumerate(chain.nodes):
        pair = node.pair
        if pair.checkpoint_id_sha256 not in preserved:
            continue
        prefix = FreshProducerPhysicalCheckpointChainV1(
            nodes=chain.nodes[: index + 1],
            current=node,
            root_sha256=chain.root_sha256,
            current_launch_dsl_compile_hash=(
                pair.current_launch_dsl_compile_hash
            ),
            complete_trajectory_proven=all(
                prefix_node.pair.complete_state_manifest_proven
                for prefix_node in chain.nodes[: index + 1]
            ),
        )
        receipt_binding = {
            "path": str(node.receipt_path),
            "bytes": node.receipt_bytes,
            "sha256": node.receipt_sha256,
        }
        source = {
            "g111_deploy_checkpoint": {
                "path": str(pair.deploy.path),
                "bytes": pair.deploy.bytes,
                "sha256": pair.deploy.sha256,
            },
            "g111_full_state_resume_checkpoint": {
                "path": str(pair.resume.path),
                "bytes": pair.resume.bytes,
                "sha256": pair.resume.sha256,
            },
            "g111_fresh_lineage_receipt": receipt_binding,
            "g111_checkpoint_id_sha256": pair.checkpoint_id_sha256,
            "g111_stage": pair.stage,
            "g111_epoch": pair.epoch,
            "fresh_lineage_complete": True,
        }
        rows.append(
            {
                "stage_tag": (
                    f"{preserved[pair.checkpoint_id_sha256]}.epoch_{pair.epoch}."
                    f"chk_{pair.checkpoint_id_sha256[:12]}"
                ),
                "source_stage_identity": source,
                "source_stage_identity_sha256": _sha256(
                    _canonical_json(source)
                ),
                "chain": prefix,
            }
        )
    return rows


def _preserved_stage_checkpoint_ids(
    producer: Path,
    nodes: Sequence[object],
) -> dict[str, str]:
    """Map physical nodes with a complete stage/final or periodic alias.

    The trainer writes each alias from the same captured arrays immediately
    after publishing the physical node.  Stage/final aliases are the historical
    deploy/resume pair plus native-v3 when the lineage node carries it. Periodic
    aliases are admitted only as the complete deploy/resume/native-v3 triplet.
    Every alias is reopened as a complete fresh-producer checkpoint and must
    recompute the exact current-ancestry physical checkpoint ID. The NPZ ZIP
    container bytes need not match because separate atomic ``np.savez`` calls
    can carry different container metadata.
    """

    from tac.witness_control.fresh_producer_lineage_v1 import (
        FreshProducerLineageV1Error,
        open_fresh_producer_checkpoint_pair_v1,
    )

    aliases: list[tuple[str, int, Path, Path, Path | None, str]] = []
    for deploy_path in sorted(producer.glob("levelset_ckpt_stage*_ep*.npz")):
        if deploy_path.is_symlink() or not deploy_path.is_file():
            raise G121StageHarvestError(
                "preserved deploy checkpoint is not a regular file"
            )
        match = _PRESERVED_DEPLOY.fullmatch(deploy_path.name)
        if match is None:
            continue
        stage_tag = match.group("tag")
        epoch = int(match.group("epoch"))
        aliases.append(
            (
                stage_tag,
                epoch,
                deploy_path,
                producer / f"levelset_resume_{stage_tag}_ep{epoch}.npz",
                None,
                "preserved stage",
            )
        )
    for (
        stage_tag,
        epoch,
        deploy_path,
        resume_path,
        native_path,
    ) in _complete_periodic_alias_triplets(producer):
        aliases.append(
            (
                f"{stage_tag}_periodic",
                epoch,
                deploy_path,
                resume_path,
                native_path,
                "periodic",
            )
        )

    mapped: dict[str, str] = {}
    for (
        stage_tag,
        epoch,
        deploy_path,
        resume_path,
        required_native_path,
        alias_kind,
    ) in aliases:
        _deploy_payload, deploy_binding = _stable_regular_file(
            deploy_path,
            name=f"{alias_kind} deploy checkpoint",
        )
        _resume_payload, resume_binding = _stable_regular_file(
            resume_path,
            name=f"{alias_kind} resume checkpoint",
        )
        candidates = [node for node in nodes if node.pair.epoch == epoch]
        native_path = required_native_path or (
            producer / f"levelset_g111_native_{stage_tag}_ep{epoch}.npz"
        )
        native_binding: dict[str, object] | None = None
        if required_native_path is not None or any(
            node.pair.native is not None for node in candidates
        ):
            _native_payload, native_binding = _stable_regular_file(
                native_path,
                name=f"{alias_kind} native-v3 checkpoint",
            )
        matched: list[object] = []
        for node in candidates:
            if (node.pair.native is None) != (native_binding is None):
                continue
            try:
                alias_pair = open_fresh_producer_checkpoint_pair_v1(
                    deploy_checkpoint=deploy_path,
                    expected_deploy_sha256=str(deploy_binding["sha256"]),
                    resume_checkpoint=resume_path,
                    expected_resume_sha256=str(resume_binding["sha256"]),
                    expected_current_launch_dsl_compile_hash=(
                        node.pair.current_launch_dsl_compile_hash
                    ),
                    native_checkpoint=(
                        None
                        if native_binding is None
                        else native_path
                    ),
                    expected_native_sha256=(
                        None
                        if native_binding is None
                        else str(native_binding["sha256"])
                    ),
                )
            except FreshProducerLineageV1Error:
                continue
            if (
                alias_pair.checkpoint_id_sha256
                == node.pair.checkpoint_id_sha256
            ):
                matched.append(node)
        if len(matched) != 1:
            raise G121StageHarvestError(
                "preserved stage alias does not uniquely reopen one current-ancestry "
                "physical checkpoint"
            )
        checkpoint_id = matched[0].pair.checkpoint_id_sha256
        prior = mapped.get(checkpoint_id)
        if prior is not None and prior != stage_tag:
            raise G121StageHarvestError(
                "two preserved stage aliases name the same physical state"
            )
        mapped[checkpoint_id] = stage_tag
    return mapped


def _complete_periodic_alias_triplets(
    producer: Path,
) -> tuple[tuple[str, int, Path, Path, Path], ...]:
    """Enumerate only complete, physical periodic native-v3 alias triplets.

    Checkpoint publication is sequential. A watcher can therefore observe the
    deploy alias before the resume or native alias exists. Such a partial
    coordinate is not evidence and remains invisible until all three regular,
    non-symlink files are present; the strict reopen above then rechecks every
    byte and its current-ancestry semantic checkpoint identity.
    """

    rows: list[tuple[str, int, Path, Path, Path]] = []
    for deploy_path in sorted(
        producer.glob("levelset_periodic_ema_stage*_ep*.npz")
    ):
        match = _PERIODIC_DEPLOY.fullmatch(deploy_path.name)
        if match is None:
            continue
        stage_tag = match.group("tag")
        epoch = int(match.group("epoch"))
        resume_path = (
            producer / f"levelset_periodic_resume_{stage_tag}_ep{epoch}.npz"
        )
        native_path = (
            producer
            / f"levelset_g111_native_{stage_tag}_periodic_ep{epoch}.npz"
        )
        triplet = (deploy_path, resume_path, native_path)
        try:
            modes = tuple(
                path.stat(follow_symlinks=False).st_mode for path in triplet
            )
        except OSError:
            continue
        if not all(stat.S_ISREG(mode) for mode in modes):
            continue
        rows.append(
            (stage_tag, epoch, deploy_path, resume_path, native_path)
        )
    return tuple(rows)


def _open_terminal_producer_result(
    producer: Path,
    *,
    expected_current_launch_dsl_compile_hash: str,
    expected_final_checkpoint_id: str,
) -> dict[str, object]:
    """Prove the trainer's terminal result names the current physical tip."""

    _require_sha256(
        expected_current_launch_dsl_compile_hash,
        name="terminal launch DSL compile hash",
    )
    expected_id = _require_sha256(
        expected_final_checkpoint_id,
        name="terminal checkpoint identity",
    )
    payload, binding = _stable_regular_file(
        producer / "levelset_train_result.json",
        name="G111 terminal train result",
    )
    try:
        result = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G121StageHarvestError("G111 terminal train result is corrupt") from exc
    stages = result.get("stage_checkpoints") if type(result) is dict else None
    if (
        type(result) is not dict
        or result.get("n_pairs") != 600
        or result.get("resumable") is not True
        or type(result.get("final_epoch")) is not int
        or type(stages) is not list
        or not stages
        or type(stages[-1]) is not dict
    ):
        raise G121StageHarvestError(
            "G111 terminal train result lacks n600 resumable final custody"
        )
    final = stages[-1]
    required = {
        "kind": "final",
        "fresh_lineage_complete_trajectory_proven": True,
        "fresh_lineage_checkpoint_id_sha256": expected_id,
    }
    if any(final.get(key) != value for key, value in required.items()):
        raise G121StageHarvestError(
            "G111 terminal result does not name the expected physical final node"
        )
    if final.get("epoch") != result["final_epoch"]:
        raise G121StageHarvestError(
            "G111 terminal result final epoch differs from its checkpoint"
        )
    receipt_path = Path(str(final.get("fresh_lineage_receipt", "")))
    receipt_sha = _require_sha256(
        final.get("fresh_lineage_receipt_sha256"),
        name="terminal fresh-lineage receipt SHA-256",
    )
    from tac.witness_control.fresh_producer_lineage_v1 import (
        open_fresh_physical_checkpoint_chain_v1,
    )

    terminal_chain = open_fresh_physical_checkpoint_chain_v1(
        receipt_path,
        expected_receipt_sha256=receipt_sha,
        expected_current_launch_dsl_compile_hash=(
            expected_current_launch_dsl_compile_hash
        ),
    )
    if terminal_chain.current.pair.checkpoint_id_sha256 != expected_id:
        raise G121StageHarvestError(
            "G111 terminal result receipt differs from the final checkpoint"
        )
    return binding


def _materialize_or_open_g112_stage(
    *,
    producer: Path,
    stage: Mapping[str, object],
) -> dict[str, object]:
    from tac.witness_control.taskspace_g112_exact_checkpoint_partition_v1 import (
        RECEIPT_NAME,
        materialize_g112_checkpoint_partition,
        open_g112_partition_receipt,
    )

    chain = stage["chain"]
    pair = chain.current.pair
    partition_root = _durable_directory(
        producer / "g121_stage_partitions",
        name="G121 stage partition root",
    )
    root = partition_root / pair.checkpoint_id_sha256
    receipt_path = root / RECEIPT_NAME
    if root.exists():
        if root.is_symlink() or not root.is_dir() or not receipt_path.is_file():
            raise G121StageHarvestError(
                "existing G112 stage partition is incomplete"
            )
        payload = receipt_path.read_bytes()
        receipt_sha = _sha256(payload)
        opened = open_g112_partition_receipt(
            receipt_path.resolve(),
            expected_sha256=receipt_sha,
        )
        return {
            "path": str(opened.receipt_path),
            "bytes": opened.receipt_bytes,
            "sha256": opened.receipt_sha256,
        }
    result = materialize_g112_checkpoint_partition(
        checkpoint=pair.deploy.path,
        expected_checkpoint_sha256=pair.deploy.sha256,
        resume_checkpoint=pair.resume.path,
        expected_resume_checkpoint_sha256=pair.resume.sha256,
        lineage_receipt=chain.current.receipt_path,
        expected_lineage_receipt_sha256=chain.current.receipt_sha256,
        expected_current_launch_dsl_compile_hash=(
            pair.current_launch_dsl_compile_hash
        ),
        output_root=root,
    )
    return {
        "path": str(result.receipt_path),
        "bytes": result.receipt_path.stat().st_size,
        "sha256": result.receipt_sha256,
    }


def _publish_reductions(
    *,
    output_dir: Path,
    ledger_path: Path,
    launch_binding: Mapping[str, object],
    live_target: Mapping[str, object],
    stage_rows: Sequence[Mapping[str, object]],
    scorer_replay_count: int,
    reused_measurement_count: int,
) -> G121StageHarvestResultV1:
    normalized_target, _target_fraction = _exact_live_target(live_target)
    if not stage_rows:
        raise G121StageHarvestError("cannot publish an empty stage harvest")
    by_physical: dict[str, Mapping[str, object]] = {}
    for row in stage_rows:
        if row["live_target"] != normalized_target:
            raise G121StageHarvestError(
                "stage reduction mixes competitive pointer views"
            )
        identity_sha = _require_sha256(
            (
                row["physical_stage_identity_sha256"]
                if row["attempt_status"] == "COMPLETED"
                else row["source_stage_identity_sha256"]
            ),
            name="eligible physical-stage identity",
        )
        if identity_sha in by_physical:
            raise G121StageHarvestError(
                "stage reduction contains duplicate physical identities"
            )
        by_physical[identity_sha] = row
    ordered = [
        by_physical[key]
        for key in sorted(by_physical)
    ]
    dispositions = [
        row["prepose_obstruction"]["disposition"]
        for row in ordered
    ]
    if any(item not in DISPOSITIONS for item in dispositions):
        raise G121StageHarvestError("stage reduction disposition is invalid")
    retained = [
        dict(row)
        for row in ordered
        if row["prepose_obstruction"]["disposition"]
        == RETAIN_POST_G105_POSE
    ]
    deferred = dispositions.count(DEFER_G115_WIRE_QAT)
    pruned = dispositions.count(PRUNE_EXACT_DISTORTION_OBSTRUCTION)
    blocked = dispositions.count(BLOCKED_SCOPED)
    counts = {
        "discovered": len(ordered),
        "accounted": len(ordered),
        "retained": len(retained),
        "deferred": deferred,
        "pruned": pruned,
        "blocked": blocked,
    }
    ledger_binding = _binding_from_path(
        ledger_path,
        name="G121 stage ledger",
    )
    completion_body = {
        "schema": COMPLETION_RECEIPT_SCHEMA,
        "status": "EXHAUSTIVE_STAGE_HARVEST_COMPLETE",
        "launch_manifest": dict(launch_binding),
        "stage_ledger": ledger_binding,
        "live_target": dict(normalized_target),
        "eligible_physical_stage_identity_sha256": sorted(by_physical),
        "accounted_attempt_identity_sha256": sorted(
            str(row["attempt_identity_sha256"]) for row in ordered
        ),
        "retained_attempt_identity_sha256": sorted(
            str(row["attempt_identity_sha256"]) for row in retained
        ),
        "counts": counts,
        "all_eligible_stages_accounted": True,
        "exhaustive_enumeration_proven": True,
        "research_only": True,
        "score_claim": False,
        "evaluation_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    completion = _seal(
        completion_body,
        key="completion_identity_sha256",
    )
    completion_path = output_dir / COMPLETION_RECEIPT_BASENAME
    _atomic_replace(completion_path, _canonical_json(completion))
    completion_binding = _binding_from_path(
        completion_path,
        name="G121 completion receipt",
    )
    scheduling_hint_path: Path | None = None
    scheduling_hint_sha: str | None = None
    if retained:
        best = min(
            retained,
            key=lambda row: (
                row["public_wire_seg"]["disagreement_pixels"],
                row["selected_archive"]["bytes"],
                row["stage_tag"],
                row["attempt_identity_sha256"],
            ),
        )
        hint = _seal(
            {
                "schema": SCHEDULING_HINT_SCHEMA,
                "status": "SCHEDULING_HINT_ONLY",
                "scheduling_hint_only": True,
                "pruning_authority": False,
                "score_authority": False,
                "selected_attempt_identity_sha256": best[
                    "attempt_identity_sha256"
                ],
                "selected_stage_tag": best["stage_tag"],
                "retained_population_count": len(retained),
                "research_only": True,
                "score_claim": False,
                "pointer_moved": False,
            },
            key="hint_identity_sha256",
        )
        scheduling_hint_path = output_dir / SCHEDULING_HINT_BASENAME
        _atomic_replace(scheduling_hint_path, _canonical_json(hint))
        scheduling_hint_sha = _binding_from_path(
            scheduling_hint_path,
            name="G121 scheduling hint",
        )["sha256"]
    manifest_body = {
        "schema": RETAINED_PREPOSE_SCHEMA,
        "status": "EXHAUSTIVE_STAGE_HARVEST_COMPLETE",
        "rows": retained,
        "completion_receipt": completion_binding,
        "stage_ledger": ledger_binding,
        "exhaustive_enumeration_proven": True,
        "pointer_snapshot_identity_sha256": normalized_target[
            "pointer_snapshot_identity_sha256"
        ],
        "postverified_pointer_identity_sha256": normalized_target[
            "postverified_pointer_identity_sha256"
        ],
        "live_target": dict(normalized_target),
        "live_target_score": float(
            Fraction(
                normalized_target["score_rational"]["numerator"],
                normalized_target["score_rational"]["denominator"],
            )
        ),
        "counts": counts,
        "research_only": True,
        "score_claim": False,
        "evaluation_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    manifest = _seal(manifest_body, key="manifest_identity_sha256")
    manifest_path = output_dir / RETAINED_PREPOSE_BASENAME
    _atomic_replace(manifest_path, _canonical_json(manifest))
    manifest_binding = _binding_from_path(
        manifest_path,
        name="G121 retained-prepose manifest",
    )
    return G121StageHarvestResultV1(
        retained_prepose_path=manifest_path,
        retained_prepose_sha256=str(manifest_binding["sha256"]),
        completion_receipt_path=completion_path,
        completion_receipt_sha256=str(completion_binding["sha256"]),
        stage_ledger_path=ledger_path,
        stage_ledger_sha256=str(ledger_binding["sha256"]),
        scheduling_hint_path=scheduling_hint_path,
        scheduling_hint_sha256=(
            str(scheduling_hint_sha)
            if scheduling_hint_sha is not None
            else None
        ),
        discovered_stage_count=counts["discovered"],
        accounted_stage_count=counts["accounted"],
        retained_stage_count=counts["retained"],
        deferred_stage_count=counts["deferred"],
        pruned_stage_count=counts["pruned"],
        blocked_stage_count=counts["blocked"],
        scorer_replay_count=scorer_replay_count,
        reused_measurement_count=reused_measurement_count,
    )


TestMeasurementProvider = Callable[
    [Mapping[str, object], Mapping[str, object] | None],
    tuple[dict[str, object], bool],
]


def _harvest_g111_stages_v1_test_only(
    *,
    stages: Sequence[Mapping[str, object]],
    live_target: Mapping[str, object],
    launch_manifest: Path,
    output_dir: Path,
    progress_dir: Path,
    provider: TestMeasurementProvider,
    injected_inputs_are_test_only: bool,
) -> G121StageHarvestResultV1:
    """Strict fixture seam; never reachable from the production API."""

    if injected_inputs_are_test_only is not True:
        raise G121StageHarvestError(
            "fixture injection requires injected_inputs_are_test_only=True"
        )
    del progress_dir
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    launch_binding = _binding_from_path(
        launch_manifest.resolve(),
        name="test-only launch manifest",
    )
    normalized_target, _fraction = _exact_live_target(live_target)
    ledger_path = output / STAGE_LEDGER_BASENAME
    prior_rows = _read_stage_ledger(ledger_path)
    latest_by_checkpoint: dict[str, dict[str, Any]] = {}
    for row in prior_rows:
        identity = (
            row["physical_stage_identity"]
            if row["attempt_status"] == "COMPLETED"
            else row["source_stage_identity"]
        )
        latest_by_checkpoint[
            identity["g111_checkpoint_id_sha256"]
        ] = row
    current_rows: list[dict[str, Any]] = []
    replay_count = 0
    reuse_count = 0
    seen_physical: set[str] = set()
    for stage in stages:
        stage_tag = _require_stage_tag(stage.get("stage_tag"))
        physical, physical_sha, _pose = _physical_stage_identity(
            stage.get("physical_stage_identity")
        )
        source = _source_from_completed_physical(physical)
        checkpoint_id = source["g111_checkpoint_id_sha256"]
        if physical_sha in seen_physical:
            raise G121StageHarvestError(
                "test-only discovery contains duplicate physical stages"
            )
        seen_physical.add(physical_sha)
        prior = latest_by_checkpoint.get(checkpoint_id)
        if prior is not None and prior["live_target"] == normalized_target:
            current_rows.append(prior)
            reuse_count += (
                1 if prior["attempt_status"] == "COMPLETED" else 0
            )
            continue
        try:
            raw, replayed = provider(stage, prior)
            if type(replayed) is not bool:
                raise G121StageHarvestError(
                    "test provider replay marker is not bool"
                )
            row = _compile_completed_attempt(
                raw,
                require_physical_files=False,
            )
            if (
                row["stage_tag"] != stage_tag
                or row["physical_stage_identity"] != physical
                or row["live_target"] != normalized_target
            ):
                raise G121StageHarvestError(
                    "test provider changed discovery/target custody"
                )
            replay_count += int(replayed)
            reuse_count += int(not replayed)
        except Exception as exc:
            row = _compile_blocked_attempt(
                stage_tag=stage_tag,
                source_stage_identity=source,
                live_target=normalized_target,
                blocker_code=type(exc).__name__,
                blocker_detail=str(exc) or "fixture stage blocked",
            )
        _append_attempt(ledger_path, row)
        current_rows.append(row)
    return _publish_reductions(
        output_dir=output,
        ledger_path=ledger_path,
        launch_binding=launch_binding,
        live_target=normalized_target,
        stage_rows=current_rows,
        scorer_replay_count=replay_count,
        reused_measurement_count=reuse_count,
    )


def _open_g121_retained_prepose_v1_test_only(
    path: Path,
    *,
    expected_sha256: str,
    injected_inputs_are_test_only: bool,
) -> G121RetainedPreposeV2:
    if injected_inputs_are_test_only is not True:
        raise G121StageHarvestError(
            "fixture opener requires injected_inputs_are_test_only=True"
        )
    return _open_g121_retained_prepose_impl(
        path,
        expected_sha256=expected_sha256,
        require_physical=False,
    )


__all__ = [
    "BLOCKED_SCOPED",
    "DEFER_G115_WIRE_QAT",
    "PRUNE_EXACT_DISTORTION_OBSTRUCTION",
    "RETAINED_PREPOSE_BASENAME",
    "RETAINED_PREPOSE_SCHEMA",
    "RETAIN_POST_G105_POSE",
    "G121RetainedPreposeV2",
    "G121RetainedStageV2",
    "G121StageHarvestError",
    "G121StageHarvestProgressV1",
    "G121StageHarvestResultV1",
    "harvest_g111_available_stages_v1",
    "harvest_g111_stages_v1",
    "open_g121_retained_prepose_v1",
]
