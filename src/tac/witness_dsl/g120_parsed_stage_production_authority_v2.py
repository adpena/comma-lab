# SPDX-License-Identifier: MIT
"""Immutable one-stage G120 measurement and exact live-pointer observation.

This v2 wrapper deliberately owns no cross-stage reduction.  A measurement is
pointer-independent and may be reused only through an externally supplied
physical receipt SHA-256.  An observation binds that immutable measurement to
the exact decimal/rational competitive target while the canonical pointer lock
is held.  G117 remains an engine only; its progress receipts are never accepted
as measurement authority.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import io
import json
import math
import os
import struct
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import numpy as np

from tac.canonical_frontier_pointer import CANONICAL_FRONTIER_POINTER_LOCK_PATH
from tac.witness_control.taskspace_g112_exact_checkpoint_partition_v1 import (
    open_g112_partition_receipt,
)
from tac.witness_control.taskspace_v9_training_target_capsule_v1 import (
    PRODUCTION_BATCH_PAIRS,
    PRODUCTION_PAIR_COUNT,
    PRODUCTION_SEG_HW,
    V9TrainingTargetCapsuleLoaderV1,
)
from tac.witness_dsl import g120_parsed_stage_production_authority_v1 as _v1
from tac.witness_dsl.dynamic_frontier_target import (
    DynamicFrontierTargetSnapshot,
    load_dynamic_frontier_target,
    verify_dynamic_frontier_target_snapshot,
)
from tac.witness_dsl.g111_parsed_g105_stage_selector_v1 import (
    VERDICT_BATCH_SIZES,
    G111ParsedG105ExactPrefixObstruction,
    compile_select_parsed_g105_stage_v1,
    open_g111_parsed_g105_exact_prefix_obstruction_v1,
)
from tac.witness_dsl.taskspace_g105_exact_v9_semantic_root_adapter_v1 import (
    Y1WireCodecV1,
)
from tac.witness_dsl.taskspace_g110_generic_two_layer_public_product_v1 import (
    PACKET_MEMBER,
    G110OuterZipMethodV1,
    parse_g110_counted_archive_variant,
    parse_g110_public_archive,
    parse_g110_two_layer_v1,
)

MEASUREMENT_SCHEMA: Final = "tac.g120_stage_measurement.v2"
OBSERVATION_SCHEMA: Final = "tac.g120_stage_observation.v2"
PRODUCTION_SCHEMA: Final = "tac.g120_parsed_stage_production_authority.v2"
EXACT_DISTORTION_OBSTRUCTION_SCHEMA: Final = (
    "tac.g120_exact_distortion_obstruction.v1"
)
SCHEMA: Final = PRODUCTION_SCHEMA
MINIMUM_SAFE_AUTHORITY_COMPLETE: Final = True
BATCH_SCHEMA: Final = "tac.g120_stage_measurement_batch.v2"
EVIDENCE_AXIS: Final = "[macOS-CPU exact parsed-public-wire producer screen]"
PIXEL_DENOMINATOR: Final = PRODUCTION_PAIR_COUNT * PRODUCTION_SEG_HW[0] * PRODUCTION_SEG_HW[1]

RETAIN_POST_G105_POSE: Final = "RETAIN_POST_G105_POSE"
DEFER_G115_WIRE_QAT: Final = "DEFER_G115_WIRE_QAT"
PRUNE_EXACT_DISTORTION_OBSTRUCTION: Final = "PRUNE_EXACT_DISTORTION_OBSTRUCTION"
BLOCKED_SCOPED: Final = "BLOCKED_SCOPED"
_LOWER_SHA256: Final = frozenset("0123456789abcdef")
_EXACT_DISTORTION_OBSTRUCTION_FIELDS: Final = frozenset(
    {
        "schema",
        "evidence_axis",
        "stage_tag",
        "obstruction_identity_sha256",
        "disposition",
        "verdict_scope",
        "engine_prefix_obstruction",
        "g112_partition_receipt",
        "physical_stage_identity",
        "physical_stage_identity_sha256",
        "g109_custody",
        "seg_scorer",
        "public_runtime",
        "live_target",
        "pointer_snapshot",
        "pointer_reverified_at_atomic_commit",
        "production_authority_closed",
        "semantic_only",
        "false_authority",
    }
)
_MEASUREMENT_FIELDS: Final = frozenset(
    {
        "schema",
        "evidence_axis",
        "stage_tag",
        "measurement_identity_sha256",
        "pointer_independent",
        "pair_count",
        "batch_sizes",
        "pixel_denominator",
        "g112_partition_receipt",
        "physical_stage_identity",
        "physical_stage_identity_sha256",
        "pose_initializer_identity_sha256",
        "g109_custody",
        "seg_scorer",
        "public_runtime",
        "alternatives",
        "selected_alternative_identity_sha256",
        "selected_archive",
        "public_wire_seg",
        "prediction_batches",
        "prediction_batch_receipt_chain_sha256",
        "source_float_seg",
        "wire_regret",
        "g115_qat",
        "repository_public_population_equal",
        "production_authority_closed",
        "semantic_only",
        "false_authority",
    }
)
_OBSERVATION_FIELDS: Final = frozenset(
    {
        "schema",
        "evidence_axis",
        "stage_tag",
        "observation_identity_sha256",
        "measurement_receipt",
        "measurement_identity_sha256",
        "public_wire_seg",
        "live_target",
        "prepose_obstruction",
        "source_float_seg",
        "wire_regret",
        "g115_qat",
        "pointer_snapshot",
        "pointer_reverified_at_atomic_commit",
        "semantic_only",
        "false_authority",
    }
)
_ALTERNATIVE_FIELDS: Final = frozenset(
    {
        "y1_wire_codec",
        "outer_zip_method",
        "disagreement_pixels",
        "pixel_denominator",
        "d_seg_rational",
        "d_seg_display_float",
        "semantic_action_display_float",
        "semantic_packet",
        "g110_product_packet",
        "archive",
        "g105_quantization_receipt_sha256",
        "scorer_y1_population_sha256",
        "camera_y1_population_sha256",
        "predicted_labels_population_sha256",
        "engine_progress_is_authority",
        "alternative_identity_sha256",
    }
)
_BATCH_FIELDS: Final = frozenset(
    {
        "schema",
        "measurement_execution_key_sha256",
        "batch_index",
        "pair_start",
        "pair_stop",
        "target_labels_batch_sha256",
        "scorer_y1_batch_sha256",
        "camera_y1_batch_sha256",
        "predicted_labels_batch_sha256",
        "prediction_file",
        "disagreement_pixels",
        "row_identity_sha256",
        "physical_receipt",
    }
)
_BATCH_RECEIPT_FIELDS: Final = _BATCH_FIELDS - {"physical_receipt"}


class G120ProductionAuthorityV2Error(RuntimeError):
    """Physical custody or exact decision formation failed closed."""


@dataclass(frozen=True, slots=True)
class G120ExactDistortionObstructionV1:
    """One pointer-atomic, custody-bound, engine-prefix scoped blocker."""

    receipt_path: Path
    receipt_sha256: str
    receipt_bytes: int
    receipt: dict[str, Any]

    @property
    def live_target(self) -> dict[str, Any]:
        return dict(self.receipt["live_target"])


class G120ExactDistortionObstruction(G120ProductionAuthorityV2Error):
    """A physical stage is scoped-blocked before a full public-wire screen."""

    def __init__(
        self,
        obstruction: G120ExactDistortionObstructionV1,
    ) -> None:
        self.obstruction = obstruction
        self.receipt_path = obstruction.receipt_path
        self.receipt_sha256 = obstruction.receipt_sha256
        self.receipt = obstruction.receipt
        self.live_target = obstruction.live_target
        super().__init__(
            "exact engine-prefix obstruction is BLOCKED_SCOPED pending "
            "sealed public-prefix and cross-wire equality: "
            f"{self.receipt_path} sha256={self.receipt_sha256}"
        )


def _sha256(payload: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


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
        raise G120ProductionAuthorityV2Error("G120-v2 value is not finite canonical ASCII JSON") from exc


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in _LOWER_SHA256 for character in value):
        raise G120ProductionAuthorityV2Error(f"{name} must be a lowercase SHA-256")
    return value


def _stable_file(path: Path, *, name: str) -> tuple[bytes, dict[str, Any]]:
    if not isinstance(path, Path) or not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise G120ProductionAuthorityV2Error(f"{name} must be an absolute regular non-symlink file")
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) or len(payload) != before.st_size:
        raise G120ProductionAuthorityV2Error(f"{name} changed during reopen")
    return payload, {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _reopen_binding(value: object, *, name: str) -> bytes:
    if type(value) is not dict or set(value) != {"path", "bytes", "sha256"}:
        raise G120ProductionAuthorityV2Error(f"{name} physical binding field census differs")
    expected_sha = _require_sha256(value.get("sha256"), name=f"{name} SHA-256")
    payload, observed = _stable_file(Path(str(value.get("path"))), name=name)
    if observed != value or observed["sha256"] != expected_sha:
        raise G120ProductionAuthorityV2Error(f"{name} physical identity differs")
    return payload


def _immutable_write(path: Path, payload: bytes) -> dict[str, Any]:
    try:
        _v1._atomic_write(path, payload)
        _payload, identity = _stable_file(path, name=path.name)
    except _v1.G120ProductionAuthorityError as exc:
        raise G120ProductionAuthorityV2Error(str(exc)) from exc
    return identity


def _durable_dir(path: Path, *, name: str) -> Path:
    try:
        return _v1._durable_dir(path, name=name)
    except _v1.G120ProductionAuthorityError as exc:
        raise G120ProductionAuthorityV2Error(str(exc)) from exc


def _ssd_cache_dir(path: Path) -> Path:
    try:
        return _v1._ssd_cache_dir(path)
    except _v1.G120ProductionAuthorityError as exc:
        raise G120ProductionAuthorityV2Error(str(exc)) from exc


def _receipt_identity(
    value: Mapping[str, Any],
    *,
    identity_field: str,
) -> str:
    body = dict(value)
    body.pop(identity_field, None)
    return _sha256(_canonical_json(body))


def _measurement_identity(value: Mapping[str, Any]) -> str:
    body = json.loads(json.dumps(value))
    body.pop("measurement_identity_sha256", None)
    public_wire = body.get("public_wire_seg")
    if isinstance(public_wire, dict):
        public_wire.pop("measurement_identity_sha256", None)
    return _sha256(_canonical_json(body))


def _open_canonical_receipt(
    path: Path,
    *,
    expected_sha256: str,
    schema: str,
    fields: frozenset[str],
    identity_field: str,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = _require_sha256(expected_sha256, name=f"expected {name}")
    raw, physical = _stable_file(path, name=name)
    if physical["sha256"] != expected:
        raise G120ProductionAuthorityV2Error(f"{name} differs from its externally expected SHA-256")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G120ProductionAuthorityV2Error(f"{name} is corrupt") from exc
    if (
        type(value) is not dict
        or set(value) != fields
        or value.get("schema") != schema
        or _canonical_json(value) != raw
        or value.get(identity_field)
        != (
            _measurement_identity(value)
            if schema == MEASUREMENT_SCHEMA
            else _receipt_identity(value, identity_field=identity_field)
        )
    ):
        raise G120ProductionAuthorityV2Error(f"{name} schema, field census, canonical bytes, or identity differs")
    return value, physical


def _decimal_fraction(value: Decimal) -> tuple[str, int, int]:
    if not value.is_finite() or value <= 0:
        raise G120ProductionAuthorityV2Error("live target Decimal must be positive and finite")
    numerator, denominator = value.as_integer_ratio()
    reduced = Fraction(numerator, denominator)
    return str(value), reduced.numerator, reduced.denominator


def _exact_target_from_snapshot(
    snapshot: DynamicFrontierTargetSnapshot,
) -> tuple[str, int, int]:
    """Recover the selected target from the pointer's exact lexical number."""

    pointer_path = Path(snapshot.pointer_path)
    raw, identity = _stable_file(pointer_path, name="canonical frontier pointer")
    if identity["bytes"] != snapshot.pointer_bytes or identity["sha256"] != snapshot.pointer_sha256:
        raise G120ProductionAuthorityV2Error("frontier pointer bytes differ from the loaded snapshot")
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            parse_float=Decimal,
            parse_int=int,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, InvalidOperation) as exc:
        raise G120ProductionAuthorityV2Error("frontier pointer has no exact lexical Decimal") from exc
    if type(decoded) is not dict:
        raise G120ProductionAuthorityV2Error("frontier pointer root is not an exact mapping")
    source = snapshot.selected_source
    if source in {
        "our_local_frontier_contest_cpu",
        "our_local_frontier_contest_cuda",
    }:
        row = decoded.get(source)
    elif source == "upstream_official_leaderboard":
        upstream = decoded.get("upstream_leaderboard_snapshot")
        row = upstream.get("best_entry") if isinstance(upstream, Mapping) else None
    else:
        # Older pointer readers may expose the constituent source text from a
        # local row.  Match only a unique custody-bearing row at the exact
        # recomputed float; never consume the serialized effective cache.
        candidates: list[Mapping[str, Any]] = []
        for key in (
            "our_local_frontier_contest_cpu",
            "our_local_frontier_contest_cuda",
        ):
            candidate = decoded.get(key)
            if isinstance(candidate, Mapping):
                candidates.append(candidate)
        upstream = decoded.get("upstream_leaderboard_snapshot")
        if isinstance(upstream, Mapping) and isinstance(upstream.get("best_entry"), Mapping):
            candidates.append(upstream["best_entry"])
        matched = [
            candidate
            for candidate in candidates
            if candidate.get("score") is not None and float(candidate["score"]) == snapshot.target_score
        ]
        if len(matched) != 1:
            raise G120ProductionAuthorityV2Error("cannot uniquely recover the exact selected pointer constituent")
        row = matched[0]
    if not isinstance(row, Mapping):
        raise G120ProductionAuthorityV2Error("selected pointer constituent is absent")
    raw_score = row.get("score")
    try:
        exact = raw_score if isinstance(raw_score, Decimal) else Decimal(raw_score) if type(raw_score) is int else None
    except InvalidOperation as exc:
        raise G120ProductionAuthorityV2Error("selected pointer score is not an exact JSON number") from exc
    if exact is None or float(exact) != snapshot.target_score:
        raise G120ProductionAuthorityV2Error("exact pointer score differs from the recomputed snapshot")
    return _decimal_fraction(exact)


def _validate_status_coordinates(
    *,
    source_float_seg: object,
    wire_regret: object,
    g115_qat: object,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if (
        type(source_float_seg) is not dict
        or set(source_float_seg)
        != {
            "status",
            "disagreement_pixels",
            "pixel_denominator",
            "measurement_receipt",
        }
        or source_float_seg.get("status") not in {"measured", "unmeasured"}
        or source_float_seg.get("pixel_denominator") != PIXEL_DENOMINATOR
    ):
        raise G120ProductionAuthorityV2Error("source-float Seg coordinate is malformed")
    if source_float_seg["status"] == "unmeasured":
        if (
            source_float_seg.get("disagreement_pixels") is not None
            or source_float_seg.get("measurement_receipt") is not None
        ):
            raise G120ProductionAuthorityV2Error("unmeasured source-float Seg coordinate differs")
    else:
        disagreement = source_float_seg.get("disagreement_pixels")
        if (
            type(disagreement) is not int
            or not 0 <= disagreement <= PIXEL_DENOMINATOR
            or type(source_float_seg.get("measurement_receipt")) is not dict
        ):
            raise G120ProductionAuthorityV2Error("measured source-float Seg coordinate differs")
        _reopen_binding(
            source_float_seg["measurement_receipt"],
            name="source-float measurement receipt",
        )

    if (
        type(wire_regret) is not dict
        or set(wire_regret)
        != {
            "status",
            "disagreement_delta_pixels",
            "rational",
            "receipt",
        }
        or wire_regret.get("status") not in {"measured", "unmeasured"}
    ):
        raise G120ProductionAuthorityV2Error("wire-regret coordinate is malformed")
    if wire_regret["status"] == "unmeasured":
        if any(wire_regret.get(key) is not None for key in ("disagreement_delta_pixels", "rational", "receipt")):
            raise G120ProductionAuthorityV2Error("unmeasured wire-regret coordinate differs")
        if source_float_seg["status"] == "measured":
            raise G120ProductionAuthorityV2Error("measured source-float requires exact measured wire regret")
    else:
        delta = wire_regret.get("disagreement_delta_pixels")
        rational = wire_regret.get("rational")
        if (
            source_float_seg["status"] != "measured"
            or type(delta) is not int
            or type(rational) is not dict
            or set(rational) != {"numerator", "denominator"}
            or rational.get("numerator") != delta
            or rational.get("denominator") != PIXEL_DENOMINATOR
            or type(wire_regret.get("receipt")) is not dict
        ):
            raise G120ProductionAuthorityV2Error("measured signed wire-regret coordinate differs")
        _reopen_binding(
            wire_regret["receipt"],
            name="wire-regret receipt",
        )

    if (
        type(g115_qat) is not dict
        or set(g115_qat)
        != {
            "status",
            "terminal_stage_physical_identity_sha256",
            "disagreement_pixels",
            "pixel_denominator",
            "receipt",
        }
        or g115_qat.get("status")
        not in {
            "terminal_stage_measured",
            "required_unmeasured",
            "not_required",
        }
    ):
        raise G120ProductionAuthorityV2Error("G115 QAT coordinate is malformed")
    if g115_qat["status"] == "terminal_stage_measured":
        terminal_disagreements = g115_qat.get("disagreement_pixels")
        if (
            _require_sha256(
                g115_qat.get("terminal_stage_physical_identity_sha256"),
                name="G115 terminal-stage identity",
            )
            is None
            or type(terminal_disagreements) is not int
            or not 0 <= terminal_disagreements <= PIXEL_DENOMINATOR
            or g115_qat.get("pixel_denominator") != PIXEL_DENOMINATOR
            or type(g115_qat.get("receipt")) is not dict
        ):
            raise G120ProductionAuthorityV2Error("terminal G115 QAT coordinate differs")
        _reopen_binding(g115_qat["receipt"], name="G115 QAT receipt")
    elif (
        g115_qat.get("terminal_stage_physical_identity_sha256") is not None
        or g115_qat.get("disagreement_pixels") is not None
        or g115_qat.get("pixel_denominator") != PIXEL_DENOMINATOR
        or g115_qat.get("receipt") is not None
    ):
        raise G120ProductionAuthorityV2Error("nonterminal G115 QAT coordinate carries custody")
    return source_float_seg, wire_regret, g115_qat


def exact_prepose_obstruction(
    *,
    disagreement_pixels: int,
    pixel_denominator: int,
    target_numerator: int,
    target_denominator: int,
    source_float_seg: dict[str, Any],
    wire_regret: dict[str, Any],
    g115_qat: dict[str, Any],
) -> dict[str, Any]:
    """Classify only the exact distortion obstruction, preserving QAT deferral."""

    if (
        type(disagreement_pixels) is not int
        or type(pixel_denominator) is not int
        or not 0 <= disagreement_pixels <= pixel_denominator
        or pixel_denominator <= 0
        or type(target_numerator) is not int
        or type(target_denominator) is not int
        or target_numerator <= 0
        or target_denominator <= 0
        or math.gcd(target_numerator, target_denominator) != 1
    ):
        raise G120ProductionAuthorityV2Error("exact obstruction operands are malformed")
    source_float_seg, wire_regret, g115_qat = _validate_status_coordinates(
        source_float_seg=source_float_seg,
        wire_regret=wire_regret,
        g115_qat=g115_qat,
    )
    if (
        wire_regret["status"] == "measured"
        and wire_regret["disagreement_delta_pixels"] != disagreement_pixels - source_float_seg["disagreement_pixels"]
    ):
        raise G120ProductionAuthorityV2Error("signed wire regret differs from exact source/wire counts")
    lhs = 100 * disagreement_pixels * target_denominator
    rhs = target_numerator * pixel_denominator
    wire_open = lhs < rhs
    if wire_open:
        disposition = RETAIN_POST_G105_POSE
    elif source_float_seg["status"] == "unmeasured":
        disposition = DEFER_G115_WIRE_QAT
    else:
        source_lhs = 100 * source_float_seg["disagreement_pixels"] * target_denominator
        if source_lhs >= rhs:
            disposition = PRUNE_EXACT_DISTORTION_OBSTRUCTION
        elif g115_qat["status"] != "terminal_stage_measured":
            disposition = DEFER_G115_WIRE_QAT
        else:
            terminal_lhs = 100 * g115_qat["disagreement_pixels"] * target_denominator
            disposition = RETAIN_POST_G105_POSE if terminal_lhs < rhs else PRUNE_EXACT_DISTORTION_OBSTRUCTION
    return {
        "rule": ("100*k*target_denominator < target_numerator*pixel_denominator"),
        "lhs": str(lhs),
        "rhs": str(rhs),
        "strict_distortion_open": wire_open,
        "disposition": disposition,
    }


def _capture_public_runtime(
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Read each shipped runtime member once into a sealed content snapshot."""

    root = repo_root / _v1.PUBLIC_RUNTIME_RELATIVE_ROOT
    if root.is_symlink() or not root.is_dir():
        raise G120ProductionAuthorityV2Error("shipped G110 public runtime root is absent or a symlink")
    observed: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise G120ProductionAuthorityV2Error("shipped public runtime contains a symlink")
        if path.is_dir():
            continue
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        observed.append(relative.as_posix())
    if tuple(observed) != _v1.PUBLIC_RUNTIME_EXPECTED_FILES:
        raise G120ProductionAuthorityV2Error("shipped public runtime file census differs")
    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    for relative in observed:
        payload, identity = _stable_file(
            (root / relative).resolve(),
            name=f"public runtime {relative}",
        )
        payloads[relative] = payload
        rows.append({"relative_path": relative, **identity})
    content_rows = [
        {
            "relative_path": row["relative_path"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in rows
    ]
    return {
        "root": str(root.resolve()),
        "files": rows,
        "tree_sha256": _sha256(_canonical_json(content_rows)),
        "physical_tree_identity_sha256": _sha256(_canonical_json(rows)),
    }, payloads


def _seal_public_runtime(
    *,
    progress_dir: Path,
    source_identity: Mapping[str, Any],
    payloads: Mapping[str, bytes],
) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    temporary = tempfile.TemporaryDirectory(
        prefix="g120-v2-sealed-runtime-",
        dir=progress_dir,
    )
    root = Path(temporary.name)
    for row in source_identity["files"]:
        relative = str(row["relative_path"])
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        written = _immutable_write(destination, payloads[relative])
        if written["bytes"] != row["bytes"] or written["sha256"] != row["sha256"]:
            temporary.cleanup()
            raise G120ProductionAuthorityV2Error("sealed public runtime copy differs from captured source bytes")
    return temporary, root


def _postverify_authority_sources(
    *,
    repo_root: Path,
    authority: Any,
    public_runtime_pre: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        closure = _v1._reopen_upstream_closure(authority.g109_custody["upstream_closure"])
        weights = _v1._regular_file_identity(
            Path(authority.g109_custody["segnet_weights"]["path"]),
            name="SegNet weights post-verification",
        )
    except (KeyError, _v1.G120ProductionAuthorityError) as exc:
        raise G120ProductionAuthorityV2Error("SegNet/upstream source changed during measurement") from exc
    if closure != authority.g109_custody["upstream_closure"] or weights != authority.g109_custody["segnet_weights"]:
        raise G120ProductionAuthorityV2Error("SegNet/upstream source changed during measurement")
    public_post, _unused = _capture_public_runtime(repo_root)
    if public_post != public_runtime_pre:
        raise G120ProductionAuthorityV2Error("public runtime changed during measurement")
    return {
        "upstream_closure_postverified": closure,
        "segnet_weights_postverified": weights,
        "public_runtime_postverified": public_post,
    }


def _build_exact_distortion_obstruction_receipt(
    *,
    engine_obstruction: Any,
    authority: Any,
    snapshot: DynamicFrontierTargetSnapshot,
    public_runtime_pre: Mapping[str, Any],
    postverified: Mapping[str, Any],
    scorer_calls: int,
) -> dict[str, Any]:
    """Bind an engine-prefix stop without promoting it to a public-wire prune."""

    score_decimal, target_numerator, target_denominator = (
        _exact_target_from_snapshot(snapshot)
    )
    pointer_identity = _v1.dynamic_snapshot_identity_sha256(snapshot)
    engine_receipt = engine_obstruction.receipt
    engine_target = engine_receipt["effective_frontier_target_exact"]
    progress_identity = engine_receipt["progress_identity"]
    if (
        engine_target
        != {
            "decimal": score_decimal,
            "numerator": target_numerator,
            "denominator": target_denominator,
        }
        or progress_identity["pointer_snapshot_identity_sha256"]
        != pointer_identity
        or progress_identity["source_checkpoint_identity_sha256"]
        != authority.physical_stage_identity_sha256
        or progress_identity["target_labels_sha256"]
        != _sha256(memoryview(authority.target_labels))
        or progress_identity["seg_scorer_identity_sha256"]
        != authority.seg_scorer_identity_sha256
        or progress_identity["pose_initializer_identity_sha256"]
        != authority.g112.initializer.checkpoint_sha256
        or progress_identity["pair_count"] != PRODUCTION_PAIR_COUNT
        or progress_identity["batch_sizes"] != list(VERDICT_BATCH_SIZES)
        or engine_receipt["stage_tag"] != authority.stage_tag
        or public_runtime_pre
        != postverified["public_runtime_postverified"]
    ):
        raise G120ProductionAuthorityV2Error(
            "engine-prefix obstruction differs from the physical stage, "
            "exact live target, or postverified public runtime"
        )
    receipt: dict[str, Any] = {
        "schema": EXACT_DISTORTION_OBSTRUCTION_SCHEMA,
        "evidence_axis": EVIDENCE_AXIS,
        "stage_tag": authority.stage_tag,
        "obstruction_identity_sha256": None,
        "disposition": BLOCKED_SCOPED,
        "verdict_scope": (
            "one_physical_stage_engine_prefix_without_"
            "sealed_public_prefix_or_cross_wire_equality"
        ),
        "engine_prefix_obstruction": {
            "path": str(engine_obstruction.receipt_path),
            "bytes": engine_obstruction.receipt_bytes,
            "sha256": engine_obstruction.receipt_sha256,
        },
        "g112_partition_receipt": dict(
            authority.physical_stage_identity["g112_partition_receipt"]
        ),
        "physical_stage_identity": authority.physical_stage_identity,
        "physical_stage_identity_sha256": (
            authority.physical_stage_identity_sha256
        ),
        "g109_custody": authority.g109_custody,
        "seg_scorer": {
            "identity_sha256": authority.seg_scorer_identity_sha256,
            "device": "cpu",
            "fresh_direct_scorer_calls": scorer_calls,
            "weights_pre": authority.g109_custody["segnet_weights"],
            "weights_post": postverified["segnet_weights_postverified"],
            "upstream_closure_pre": authority.g109_custody[
                "upstream_closure"
            ],
            "upstream_closure_post": postverified[
                "upstream_closure_postverified"
            ],
        },
        "public_runtime": {
            "source_pre": dict(public_runtime_pre),
            "sealed_tree_sha256": public_runtime_pre["tree_sha256"],
            "source_post": postverified["public_runtime_postverified"],
            "sealed_runtime_captured": True,
            "public_prefix_execution_performed": False,
            "public_prefix_equality": False,
            "cross_wire_prefix_equality": False,
        },
        "live_target": {
            "score_decimal": score_decimal,
            "score_rational": {
                "numerator": target_numerator,
                "denominator": target_denominator,
            },
            "pointer_snapshot_identity_sha256": pointer_identity,
            "postverified_pointer_identity_sha256": pointer_identity,
        },
        "pointer_snapshot": dataclasses.asdict(snapshot),
        "pointer_reverified_at_atomic_commit": True,
        "production_authority_closed": False,
        "semantic_only": True,
        "false_authority": {
            "contest_score_claim": False,
            "candidate_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
            "public_wire_prune_claim": False,
            "family_wide_claim": False,
        },
    }
    receipt["obstruction_identity_sha256"] = _receipt_identity(
        receipt,
        identity_field="obstruction_identity_sha256",
    )
    return receipt


def _commit_exact_distortion_obstruction(
    *,
    repo_root: Path,
    out_dir: Path,
    engine_obstruction: Any,
    authority: Any,
    snapshot: DynamicFrontierTargetSnapshot,
    public_runtime_pre: Mapping[str, Any],
    postverified: Mapping[str, Any],
    scorer_calls: int,
) -> G120ExactDistortionObstructionV1:
    receipt = _build_exact_distortion_obstruction_receipt(
        engine_obstruction=engine_obstruction,
        authority=authority,
        snapshot=snapshot,
        public_runtime_pre=public_runtime_pre,
        postverified=postverified,
        scorer_calls=scorer_calls,
    )
    path = out_dir / (
        f"{authority.stage_tag}."
        f"{receipt['obstruction_identity_sha256']}."
        "g120_exact_distortion_obstruction.v1.json"
    )
    lock_path = repo_root / CANONICAL_FRONTIER_POINTER_LOCK_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_SH)
        try:
            verify_dynamic_frontier_target_snapshot(snapshot)
            binding = _immutable_write(path, _canonical_json(receipt))
        except Exception as exc:
            if isinstance(exc, G120ProductionAuthorityV2Error):
                raise
            raise G120ProductionAuthorityV2Error(
                "frontier pointer changed before atomic obstruction commit"
            ) from exc
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(lock_fd)
    return open_g120_exact_distortion_obstruction_v1(
        path,
        expected_sha256=binding["sha256"],
    )


class _FreshPredictionBroker:
    """In-memory de-duplication around direct frozen-model inference only."""

    def __init__(self, scorer: Any) -> None:
        direct = getattr(scorer, "_predict", None)
        self._direct = direct if callable(direct) else scorer
        if not callable(self._direct):
            raise G120ProductionAuthorityV2Error("physical production authority has no direct SegNet predictor")
        self._by_camera_sha: dict[str, np.ndarray] = {}
        self.actual_scorer_calls = 0

    def __call__(self, camera_y1: np.ndarray) -> np.ndarray:
        raw = np.asarray(camera_y1)
        if (
            type(raw) is not np.ndarray
            or raw.dtype != np.uint8
            or raw.ndim != 4
            or raw.shape[1:] != (874, 1164, 3)
            or not raw.flags.c_contiguous
            or not 1 <= raw.shape[0] <= PRODUCTION_BATCH_PAIRS
        ):
            raise G120ProductionAuthorityV2Error("direct SegNet camera batch is malformed")
        key = _sha256(memoryview(raw))
        predicted = self._by_camera_sha.get(key)
        if predicted is None:
            predicted = self._direct(raw)
            self.actual_scorer_calls += 1
            expected = (raw.shape[0], *PRODUCTION_SEG_HW)
            if (
                type(predicted) is not np.ndarray
                or predicted.dtype != np.uint8
                or predicted.shape != expected
                or not predicted.flags.c_contiguous
                or np.any(predicted >= 5)
            ):
                raise G120ProductionAuthorityV2Error("direct frozen CPU SegNet returned malformed labels")
            predicted = np.ascontiguousarray(predicted, dtype=np.uint8)
            self._by_camera_sha[key] = predicted
        return predicted


def _alternative_identity_body(
    alternative: Any,
    *,
    semantic_packet: Mapping[str, Any],
    product_packet: Mapping[str, Any],
    archive: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "y1_wire_codec": alternative.y1_wire_codec.name,
        "outer_zip_method": alternative.outer_zip_method.name,
        "disagreement_pixels": alternative.disagreement_pixels,
        "pixel_denominator": PIXEL_DENOMINATOR,
        "d_seg_rational": {
            "numerator": alternative.disagreement_pixels,
            "denominator": PIXEL_DENOMINATOR,
        },
        "d_seg_display_float": alternative.d_seg,
        "semantic_action_display_float": alternative.semantic_action,
        "semantic_packet": dict(semantic_packet),
        "g110_product_packet": dict(product_packet),
        "archive": dict(archive),
        "g105_quantization_receipt_sha256": (alternative.g105_quantization_receipt_sha256),
        "scorer_y1_population_sha256": (alternative.scorer_y1_population_sha256),
        "camera_y1_population_sha256": (alternative.camera_y1_population_sha256),
        "predicted_labels_population_sha256": (alternative.predicted_labels_sha256),
        "engine_progress_is_authority": False,
    }


def _persist_alternatives(
    *,
    authority: Any,
    engine: Any,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matrix = {
        (Y1WireCodecV1.RAW_I16_LE, G110OuterZipMethodV1.STORE),
        (Y1WireCodecV1.RAW_I16_LE, G110OuterZipMethodV1.DEFLATE),
        (
            Y1WireCodecV1.DELTA_RICE_BEST_K,
            G110OuterZipMethodV1.STORE,
        ),
        (
            Y1WireCodecV1.DELTA_RICE_BEST_K,
            G110OuterZipMethodV1.DEFLATE,
        ),
    }
    alternatives = getattr(engine, "alternatives", None)
    if (
        type(alternatives) is not tuple
        or len(alternatives) != 4
        or {(item.y1_wire_codec, item.outer_zip_method) for item in alternatives} != matrix
        or not any(engine.selected is item for item in alternatives)
    ):
        raise G120ProductionAuthorityV2Error("G117 did not return the exact four-way archive matrix")
    prefix = f"{authority.stage_tag}.{authority.physical_stage_identity_sha256[:16]}"
    packet_bindings: dict[str, dict[str, Any]] = {}
    product_bindings: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    selected_row: dict[str, Any] | None = None
    for alternative in alternatives:
        codec = alternative.y1_wire_codec.name.lower()
        method = alternative.outer_zip_method.name.lower()
        semantic_binding = packet_bindings.get(codec)
        if semantic_binding is None:
            semantic_binding = _immutable_write(
                out_dir / f"{prefix}.{codec}.g105_packet.bin",
                alternative.semantic_packet,
            )
            packet_bindings[codec] = semantic_binding
        product_binding = product_bindings.get(codec)
        if product_binding is None:
            product_binding = _immutable_write(
                out_dir / f"{prefix}.{codec}.g110_packet.bin",
                alternative.product_packet,
            )
            product_bindings[codec] = product_binding
        archive_binding = _immutable_write(
            out_dir / f"{prefix}.{codec}.{method}.archive.zip",
            alternative.archive,
        )
        if (
            parse_g110_public_archive(alternative.archive) != alternative.product_packet
            or parse_g110_counted_archive_variant(
                alternative.archive,
                alternative.outer_zip_method,
            )
            != alternative.product_packet
        ):
            raise G120ProductionAuthorityV2Error("persisted alternative fails exact G110 parse-back")
        body = _alternative_identity_body(
            alternative,
            semantic_packet=semantic_binding,
            product_packet=product_binding,
            archive=archive_binding,
        )
        row = {
            **body,
            "alternative_identity_sha256": _sha256(_canonical_json(body)),
        }
        rows.append(row)
        if engine.selected is alternative:
            selected_row = row
    rows.sort(
        key=lambda row: (
            row["y1_wire_codec"],
            row["outer_zip_method"],
        )
    )
    if selected_row is None:
        raise AssertionError("selected alternative disappeared")
    return rows, selected_row


def _batch_chain(
    rows: Sequence[Mapping[str, Any]],
    field_name: str,
) -> str:
    return _sha256(
        _canonical_json(
            [
                {
                    "batch_index": row["batch_index"],
                    field_name: row[field_name],
                }
                for row in rows
            ]
        )
    )


def _load_public_plugins(
    *,
    sealed_root: Path,
    selected_archive: bytes,
    progress_dir: Path,
    runtime_tree_sha256: str,
) -> tuple[ModuleType, ModuleType, ModuleType, Any, bytes]:
    inflate = _v1._load_module(
        sealed_root / "inflate.py",
        module_name=f"_g120_v2_public_inflate_{runtime_tree_sha256}",
    )
    packet = parse_g110_public_archive(selected_archive)
    with tempfile.TemporaryDirectory(
        prefix="g120-v2-public-packet-",
        dir=progress_dir,
    ) as temporary:
        packet_root = Path(temporary)
        _immutable_write(packet_root / PACKET_MEMBER, packet)
        public_packet = inflate._load_packet(packet_root)
    if public_packet != packet:
        raise G120ProductionAuthorityV2Error("sealed public packet loader changed selected packet bytes")
    semantic_plugins = inflate._load_plugins(
        sealed_root / "semantic_variants",
        calls=inflate.SEMANTIC_PLUGIN_CALLS,
        expected=inflate.EXPECTED_SEMANTIC_PLUGINS,
    )
    frame0_plugins = inflate._load_plugins(
        sealed_root / "frame0_variants",
        calls=inflate.FRAME0_PLUGIN_CALLS,
        expected=inflate.EXPECTED_FRAME0_PLUGINS,
    )
    frame0_matches = [module for module in frame0_plugins.values() if module.accepts_packet(packet) is True]
    if len(frame0_matches) != 1:
        raise G120ProductionAuthorityV2Error("selected packet does not match exactly one sealed frame0 plugin")
    frame0 = frame0_matches[0]
    frame0_state = frame0.parse_packet(packet)
    semantic_packet = frame0.semantic_packet(frame0_state)
    semantic_matches = [
        module for module in semantic_plugins.values() if module.accepts_packet(semantic_packet) is True
    ]
    if len(semantic_matches) != 1:
        raise G120ProductionAuthorityV2Error("selected packet does not match exactly one sealed semantic plugin")
    semantic = semantic_matches[0]
    return inflate, semantic, frame0, frame0_state, semantic_packet


def _verify_selected_semantic_packet(
    semantic_packet: bytes,
    selected_row: Mapping[str, Any],
) -> None:
    if (
        type(semantic_packet) is not bytes
        or not semantic_packet
        or type(selected_row.get("semantic_packet")) is not dict
        or _sha256(semantic_packet) != selected_row["semantic_packet"].get("sha256")
    ):
        raise G120ProductionAuthorityV2Error("sealed public dispatch changed selected G105 semantic packet")


def _reopen_completed_prediction_batch(
    *,
    receipt_path: Path,
    prediction_path: Path,
    expected_execution_key_sha256: str,
    batch_index: int,
    pair_start: int,
    pair_stop: int,
    target_batch: np.ndarray,
    scorer_y1_batch_sha256: str,
    camera_y1_batch_sha256: str,
) -> dict[str, Any] | None:
    """Reopen one exact crash-complete batch before any scorer invocation."""

    if receipt_path.is_symlink():
        raise G120ProductionAuthorityV2Error(
            "prediction batch receipt must not be a symlink"
        )
    if not receipt_path.exists():
        return None
    raw, physical = _stable_file(
        receipt_path,
        name=f"prediction batch {batch_index} receipt",
    )
    try:
        row = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G120ProductionAuthorityV2Error(
            "prediction batch receipt is corrupt"
        ) from exc
    body = dict(row) if type(row) is dict else {}
    identity = body.pop("row_identity_sha256", None)
    expected_shape = (pair_stop - pair_start, *PRODUCTION_SEG_HW)
    if (
        type(row) is not dict
        or set(row) != _BATCH_RECEIPT_FIELDS
        or _canonical_json(row) != raw
        or row.get("schema") != BATCH_SCHEMA
        or row.get("measurement_execution_key_sha256")
        != expected_execution_key_sha256
        or row.get("batch_index") != batch_index
        or row.get("pair_start") != pair_start
        or row.get("pair_stop") != pair_stop
        or row.get("target_labels_batch_sha256")
        != _sha256(memoryview(target_batch))
        or row.get("scorer_y1_batch_sha256")
        != scorer_y1_batch_sha256
        or row.get("camera_y1_batch_sha256")
        != camera_y1_batch_sha256
        or identity != _sha256(_canonical_json(body))
    ):
        raise G120ProductionAuthorityV2Error(
            "prediction batch resume identity differs"
        )
    prediction = row.get("prediction_file")
    if (
        type(prediction) is not dict
        or prediction.get("path") != str(prediction_path.resolve())
    ):
        raise G120ProductionAuthorityV2Error(
            "prediction batch resume array path differs"
        )
    prediction_payload = _reopen_binding(
        prediction,
        name=f"prediction batch {batch_index} array",
    )
    try:
        predicted = np.load(
            io.BytesIO(prediction_payload),
            allow_pickle=False,
        )
    except (OSError, ValueError) as exc:
        raise G120ProductionAuthorityV2Error(
            "prediction batch resume array cannot be reopened"
        ) from exc
    if (
        type(predicted) is not np.ndarray
        or predicted.dtype != np.uint8
        or predicted.shape != expected_shape
        or not predicted.flags.c_contiguous
        or np.any(predicted >= 5)
    ):
        raise G120ProductionAuthorityV2Error(
            "prediction batch resume array geometry differs"
        )
    disagreement = int(np.count_nonzero(predicted != target_batch))
    if (
        row.get("predicted_labels_batch_sha256")
        != _sha256(memoryview(predicted))
        or row.get("disagreement_pixels") != disagreement
    ):
        raise G120ProductionAuthorityV2Error(
            "prediction batch resume array/count differs"
        )
    return {**row, "physical_receipt": physical}


def _persist_prediction_batch(
    *,
    receipt_path: Path,
    prediction_path: Path,
    execution_key_sha256: str,
    batch_index: int,
    pair_start: int,
    pair_stop: int,
    target_batch: np.ndarray,
    scorer_y1_batch_sha256: str,
    camera_y1_batch_sha256: str,
    predicted: np.ndarray,
) -> dict[str, Any]:
    """Atomically persist one production-shaped batch for exact later resume."""

    expected_shape = (pair_stop - pair_start, *PRODUCTION_SEG_HW)
    if (
        type(predicted) is not np.ndarray
        or predicted.dtype != np.uint8
        or predicted.shape != expected_shape
        or not predicted.flags.c_contiguous
        or np.any(predicted >= 5)
    ):
        raise G120ProductionAuthorityV2Error(
            "frozen CPU SegNet returned malformed public labels"
        )
    prediction_buffer = io.BytesIO()
    np.save(prediction_buffer, predicted, allow_pickle=False)
    prediction_binding = _immutable_write(
        prediction_path,
        prediction_buffer.getvalue(),
    )
    body = {
        "schema": BATCH_SCHEMA,
        "measurement_execution_key_sha256": execution_key_sha256,
        "batch_index": batch_index,
        "pair_start": pair_start,
        "pair_stop": pair_stop,
        "target_labels_batch_sha256": _sha256(memoryview(target_batch)),
        "scorer_y1_batch_sha256": scorer_y1_batch_sha256,
        "camera_y1_batch_sha256": camera_y1_batch_sha256,
        "predicted_labels_batch_sha256": _sha256(memoryview(predicted)),
        "prediction_file": prediction_binding,
        "disagreement_pixels": int(
            np.count_nonzero(predicted != target_batch)
        ),
    }
    row = {
        **body,
        "row_identity_sha256": _sha256(_canonical_json(body)),
    }
    receipt_binding = _immutable_write(
        receipt_path,
        _canonical_json(row),
    )
    return {**row, "physical_receipt": receipt_binding}


def _measure_public_surface_fresh(
    *,
    authority: Any,
    engine: Any,
    broker: _FreshPredictionBroker,
    sealed_root: Path,
    progress_dir: Path,
    measurement_cache_dir: Path,
    public_runtime_tree_sha256: str,
    selected_row: Mapping[str, Any],
) -> dict[str, Any]:
    selected = engine.selected
    inflate, semantic, frame0, frame0_state, semantic_packet = _load_public_plugins(
        sealed_root=sealed_root,
        selected_archive=selected.archive,
        progress_dir=progress_dir,
        runtime_tree_sha256=public_runtime_tree_sha256,
    )
    _verify_selected_semantic_packet(semantic_packet, selected_row)
    parsed = semantic.parse_packet(semantic_packet)
    execution_body = {
        "schema": BATCH_SCHEMA,
        "physical_stage_identity_sha256": (authority.physical_stage_identity_sha256),
        "selected_archive_sha256": selected_row["archive"]["sha256"],
        "archive_alternatives": sorted(
            [
                {
                    "y1_wire_codec": item.y1_wire_codec.name,
                    "outer_zip_method": item.outer_zip_method.name,
                    "archive_bytes": len(item.archive),
                    "archive_sha256": _sha256(item.archive),
                }
                for item in engine.alternatives
            ],
            key=lambda row: (
                row["y1_wire_codec"],
                row["outer_zip_method"],
            ),
        ),
        "target_labels_sha256": _sha256(memoryview(authority.target_labels)),
        "seg_scorer_identity_sha256": authority.seg_scorer_identity_sha256,
        "public_runtime_tree_sha256": public_runtime_tree_sha256,
        "pair_count": PRODUCTION_PAIR_COUNT,
        "batch_sizes": list(VERDICT_BATCH_SIZES),
        "frontier_pointer_intentionally_excluded": True,
    }
    execution_key = _sha256(_canonical_json(execution_body))
    cache_root = _durable_dir(
        measurement_cache_dir / execution_key,
        name="G120-v2 physical prediction bundle",
    )
    rows: list[dict[str, Any]] = []
    resumed_batch_count = 0
    final_y1_population = hashlib.sha256()
    pair_start = 0
    for batch_index, batch_size in enumerate(VERDICT_BATCH_SIZES):
        pair_stop = pair_start + batch_size
        scorer_digest = hashlib.sha256()
        camera_digest = hashlib.sha256()
        camera_frames: list[np.ndarray] = []
        for pair_id in range(pair_start, pair_stop):
            scorer_y1 = semantic.render_scorer_y1(parsed, pair_id)
            if (
                type(scorer_y1) is not np.ndarray
                or scorer_y1.dtype != np.uint8
                or scorer_y1.shape != (*PRODUCTION_SEG_HW, 3)
            ):
                raise G120ProductionAuthorityV2Error("sealed semantic plugin emitted malformed scorer Y1")
            scorer_y1 = np.ascontiguousarray(scorer_y1)
            final_y1_population.update(struct.pack(">H", pair_id))
            final_y1_population.update(memoryview(scorer_y1))
            camera_y1 = inflate._realize_factor2(scorer_y1)
            camera_y0 = frame0.render_camera_y0(
                frame0_state,
                pair_id,
                scorer_y1,
                camera_y1,
            )
            if (
                type(camera_y0) is not np.ndarray
                or camera_y0.dtype != np.uint8
                or not camera_y0.flags.c_contiguous
                or not np.array_equal(camera_y0, camera_y1)
            ):
                raise G120ProductionAuthorityV2Error("sealed rank-zero public Y0 differs from camera Y1")
            scorer_digest.update(memoryview(scorer_y1))
            camera_digest.update(memoryview(camera_y1))
            camera_frames.append(camera_y1)
        camera_batch = np.ascontiguousarray(
            np.stack(camera_frames),
            dtype=np.uint8,
        )
        target = authority.target_labels[pair_start:pair_stop]
        prediction_path = cache_root / (
            f"batch_{batch_index:03d}_{pair_start:03d}_{pair_stop:03d}.predicted_labels.npy"
        )
        receipt_path = cache_root / (
            f"batch_{batch_index:03d}_{pair_start:03d}_{pair_stop:03d}.receipt.json"
        )
        row = _reopen_completed_prediction_batch(
            receipt_path=receipt_path,
            prediction_path=prediction_path,
            expected_execution_key_sha256=execution_key,
            batch_index=batch_index,
            pair_start=pair_start,
            pair_stop=pair_stop,
            target_batch=target,
            scorer_y1_batch_sha256=scorer_digest.hexdigest(),
            camera_y1_batch_sha256=camera_digest.hexdigest(),
        )
        if row is None:
            predicted = broker(camera_batch)
            row = _persist_prediction_batch(
                receipt_path=receipt_path,
                prediction_path=prediction_path,
                execution_key_sha256=execution_key,
                batch_index=batch_index,
                pair_start=pair_start,
                pair_stop=pair_stop,
                target_batch=target,
                scorer_y1_batch_sha256=scorer_digest.hexdigest(),
                camera_y1_batch_sha256=camera_digest.hexdigest(),
                predicted=predicted,
            )
        else:
            resumed_batch_count += 1
        rows.append(row)
        pair_start = pair_stop
    frame0.verify_final_y1_population(
        frame0_state,
        final_y1_population.digest(),
    )
    if pair_start != PRODUCTION_PAIR_COUNT or len(rows) != len(VERDICT_BATCH_SIZES):
        raise AssertionError("public replay lost exact n600 chronology")
    scorer_population = _batch_chain(rows, "scorer_y1_batch_sha256")
    camera_population = _batch_chain(rows, "camera_y1_batch_sha256")
    predicted_population = _batch_chain(
        rows,
        "predicted_labels_batch_sha256",
    )
    if (
        scorer_population != selected.scorer_y1_population_sha256
        or camera_population != selected.camera_y1_population_sha256
        or predicted_population != selected.predicted_labels_sha256
    ):
        raise G120ProductionAuthorityV2Error("repository and sealed-public ordered populations differ")
    disagreements = sum(row["disagreement_pixels"] for row in rows)
    if disagreements != selected.disagreement_pixels or selected.d_seg != disagreements / PIXEL_DENOMINATOR:
        raise G120ProductionAuthorityV2Error("repository and sealed-public exact disagreement count differs")
    return {
        "execution_key_sha256": execution_key,
        "rows": rows,
        "batch_receipt_chain_sha256": _batch_chain(
            rows,
            "row_identity_sha256",
        ),
        "disagreement_pixels": disagreements,
        "scorer_y1_population_sha256": scorer_population,
        "camera_y1_population_sha256": camera_population,
        "predicted_labels_population_sha256": predicted_population,
        "fresh_measured_batch_count": len(rows) - resumed_batch_count,
        "resumed_physical_batch_count": resumed_batch_count,
    }


def _reopen_runtime_tree(value: object) -> dict[str, Any]:
    if (
        type(value) is not dict
        or set(value)
        != {
            "root",
            "files",
            "tree_sha256",
            "physical_tree_identity_sha256",
        }
        or type(value.get("files")) is not list
    ):
        raise G120ProductionAuthorityV2Error("public runtime identity field census differs")
    rows = value["files"]
    if [row.get("relative_path") for row in rows] != list(_v1.PUBLIC_RUNTIME_EXPECTED_FILES):
        raise G120ProductionAuthorityV2Error("public runtime identity file census differs")
    reopened: list[dict[str, Any]] = []
    for row in rows:
        if type(row) is not dict or set(row) != {
            "relative_path",
            "path",
            "bytes",
            "sha256",
        }:
            raise G120ProductionAuthorityV2Error("public runtime member identity differs")
        payload, identity = _stable_file(
            Path(row["path"]),
            name=f"public runtime {row['relative_path']}",
        )
        del payload
        observed = {"relative_path": row["relative_path"], **identity}
        if observed != row:
            raise G120ProductionAuthorityV2Error("public runtime member changed")
        reopened.append(observed)
    content_rows = [
        {
            "relative_path": row["relative_path"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in reopened
    ]
    observed_tree = {
        "root": value["root"],
        "files": reopened,
        "tree_sha256": _sha256(_canonical_json(content_rows)),
        "physical_tree_identity_sha256": _sha256(_canonical_json(reopened)),
    }
    if observed_tree != value:
        raise G120ProductionAuthorityV2Error("public runtime tree identity changed")
    return observed_tree


def _measurement_status_defaults() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        {
            "status": "unmeasured",
            "disagreement_pixels": None,
            "pixel_denominator": PIXEL_DENOMINATOR,
            "measurement_receipt": None,
        },
        {
            "status": "unmeasured",
            "disagreement_delta_pixels": None,
            "rational": None,
            "receipt": None,
        },
        {
            "status": "required_unmeasured",
            "terminal_stage_physical_identity_sha256": None,
            "disagreement_pixels": None,
            "pixel_denominator": PIXEL_DENOMINATOR,
            "receipt": None,
        },
    )


def _build_measurement_receipt(
    *,
    authority: Any,
    alternatives: list[dict[str, Any]],
    selected_row: Mapping[str, Any],
    public: Mapping[str, Any],
    public_runtime_pre: Mapping[str, Any],
    postverified: Mapping[str, Any],
    scorer_calls: int,
) -> dict[str, Any]:
    source_float, regret, qat = _measurement_status_defaults()
    public_wire = {
        "disagreement_pixels": public["disagreement_pixels"],
        "pixel_denominator": PIXEL_DENOMINATOR,
        "d_seg_rational": {
            "numerator": public["disagreement_pixels"],
            "denominator": PIXEL_DENOMINATOR,
        },
        "d_seg_display_float": (public["disagreement_pixels"] / PIXEL_DENOMINATOR),
        "measurement_identity_sha256": None,
    }
    receipt: dict[str, Any] = {
        "schema": MEASUREMENT_SCHEMA,
        "evidence_axis": EVIDENCE_AXIS,
        "stage_tag": authority.stage_tag,
        "measurement_identity_sha256": None,
        "pointer_independent": True,
        "pair_count": PRODUCTION_PAIR_COUNT,
        "batch_sizes": list(VERDICT_BATCH_SIZES),
        "pixel_denominator": PIXEL_DENOMINATOR,
        "g112_partition_receipt": dict(authority.physical_stage_identity["g112_partition_receipt"]),
        "physical_stage_identity": authority.physical_stage_identity,
        "physical_stage_identity_sha256": (authority.physical_stage_identity_sha256),
        "pose_initializer_identity_sha256": (authority.g112.initializer.checkpoint_sha256),
        "g109_custody": authority.g109_custody,
        "seg_scorer": {
            "identity_sha256": authority.seg_scorer_identity_sha256,
            "device": "cpu",
            "fresh_direct_scorer_calls": scorer_calls,
            "fresh_measured_batch_count": public[
                "fresh_measured_batch_count"
            ],
            "resumed_physical_batch_count": public[
                "resumed_physical_batch_count"
            ],
            "disk_prediction_cache_trusted": False,
            "weights_pre": authority.g109_custody["segnet_weights"],
            "weights_post": postverified["segnet_weights_postverified"],
            "upstream_closure_pre": authority.g109_custody["upstream_closure"],
            "upstream_closure_post": postverified["upstream_closure_postverified"],
        },
        "public_runtime": {
            "source_pre": dict(public_runtime_pre),
            "sealed_tree_sha256": public_runtime_pre["tree_sha256"],
            "source_post": postverified["public_runtime_postverified"],
            "toctou_closed_by": "sealed_content_execution_plus_source_postverify",
        },
        "alternatives": alternatives,
        "selected_alternative_identity_sha256": selected_row["alternative_identity_sha256"],
        "selected_archive": dict(selected_row["archive"]),
        "public_wire_seg": public_wire,
        "prediction_batches": list(public["rows"]),
        "prediction_batch_receipt_chain_sha256": public["batch_receipt_chain_sha256"],
        "source_float_seg": source_float,
        "wire_regret": regret,
        "g115_qat": qat,
        "repository_public_population_equal": True,
        "production_authority_closed": True,
        "semantic_only": True,
        "false_authority": {
            "contest_score_claim": False,
            "candidate_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
            "production_launch_performed": False,
            "g117_receipt_is_authority": False,
            "legacy_batch_cache_trusted": False,
        },
    }
    identity = _measurement_identity(receipt)
    receipt["measurement_identity_sha256"] = identity
    public_wire["measurement_identity_sha256"] = identity
    if _measurement_identity(receipt) != identity:
        raise AssertionError("measurement identity construction drifted")
    return receipt


def _validate_alternatives(
    rows: object,
    *,
    selected_identity: str,
) -> dict[str, Any]:
    if type(rows) is not list or len(rows) != 4:
        raise G120ProductionAuthorityV2Error("measurement lacks exact four-way alternatives")
    matrix: set[tuple[str, str]] = set()
    selected: dict[str, Any] | None = None
    keyed_rows: list[tuple[tuple[float, float, int, int, int, str], dict[str, Any]]] = []
    for row in rows:
        if type(row) is not dict or set(row) != _ALTERNATIVE_FIELDS:
            raise G120ProductionAuthorityV2Error("alternative row field census differs")
        identity = row.get("alternative_identity_sha256")
        body = dict(row)
        body.pop("alternative_identity_sha256", None)
        if identity != _sha256(_canonical_json(body)):
            raise G120ProductionAuthorityV2Error("alternative identity differs")
        codec_name = row.get("y1_wire_codec")
        method_name = row.get("outer_zip_method")
        try:
            Y1WireCodecV1[codec_name]
            method = G110OuterZipMethodV1[method_name]
        except (KeyError, TypeError) as exc:
            raise G120ProductionAuthorityV2Error("alternative typed codec/method differs") from exc
        matrix.add((codec_name, method_name))
        semantic_packet = _reopen_binding(
            row.get("semantic_packet"),
            name="alternative semantic packet",
        )
        product_packet = _reopen_binding(
            row.get("g110_product_packet"),
            name="alternative G110 product packet",
        )
        archive = _reopen_binding(
            row.get("archive"),
            name="alternative archive",
        )
        try:
            parsed_product = parse_g110_two_layer_v1(product_packet)
            semantic_program = _v1.g105_adapter.parse_packet(semantic_packet)
        except Exception as exc:
            raise G120ProductionAuthorityV2Error("alternative nested G105/G110 packet cannot be parsed") from exc
        if (
            parse_g110_public_archive(archive) != product_packet
            or parse_g110_counted_archive_variant(archive, method) != product_packet
            or parsed_product.packet != product_packet
            or parsed_product.semantic_packet != semantic_packet
            or _v1.g105_adapter.encode_packet(semantic_program) != semantic_packet
        ):
            raise G120ProductionAuthorityV2Error("alternative physical parse-back differs")
        k = row.get("disagreement_pixels")
        if (
            type(k) is not int
            or not 0 <= k <= PIXEL_DENOMINATOR
            or row.get("pixel_denominator") != PIXEL_DENOMINATOR
            or row.get("d_seg_rational") != {"numerator": k, "denominator": PIXEL_DENOMINATOR}
            or row.get("d_seg_display_float") != k / PIXEL_DENOMINATOR
            or row.get("semantic_action_display_float")
            != _v1.semantic_stage_action(
                d_seg=k / PIXEL_DENOMINATOR,
                archive_bytes=row["archive"]["bytes"],
            )
            or row.get("engine_progress_is_authority") is not False
        ):
            raise G120ProductionAuthorityV2Error("alternative exact Seg coordinate differs")
        for key in (
            "g105_quantization_receipt_sha256",
            "scorer_y1_population_sha256",
            "camera_y1_population_sha256",
            "predicted_labels_population_sha256",
        ):
            _require_sha256(row.get(key), name=f"alternative {key}")
        keyed_rows.append(
            (
                (
                    row["semantic_action_display_float"],
                    row["d_seg_display_float"],
                    row["archive"]["bytes"],
                    int(Y1WireCodecV1[codec_name]),
                    int(method),
                    row["archive"]["sha256"],
                ),
                row,
            )
        )
        if identity == selected_identity:
            selected = row
    expected_matrix = {
        (Y1WireCodecV1.RAW_I16_LE.name, G110OuterZipMethodV1.STORE.name),
        (
            Y1WireCodecV1.RAW_I16_LE.name,
            G110OuterZipMethodV1.DEFLATE.name,
        ),
        (
            Y1WireCodecV1.DELTA_RICE_BEST_K.name,
            G110OuterZipMethodV1.STORE.name,
        ),
        (
            Y1WireCodecV1.DELTA_RICE_BEST_K.name,
            G110OuterZipMethodV1.DEFLATE.name,
        ),
    }
    if matrix != expected_matrix or selected is None:
        raise G120ProductionAuthorityV2Error("alternative matrix or selected identity differs")
    if (
        len({row["disagreement_pixels"] for _key, row in keyed_rows}) != 1
        or len(
            {
                (
                    row["scorer_y1_population_sha256"],
                    row["camera_y1_population_sha256"],
                    row["predicted_labels_population_sha256"],
                )
                for _key, row in keyed_rows
            }
        )
        != 1
    ):
        raise G120ProductionAuthorityV2Error("four alternatives do not describe one exact scored object")
    canonical = min(keyed_rows, key=lambda item: item[0])[1]
    if canonical["alternative_identity_sha256"] != selected_identity:
        raise G120ProductionAuthorityV2Error("selected alternative is not the canonical exact G117 minimum")
    return selected


def _reopen_prediction_batches(
    rows: object,
    *,
    target_labels: np.ndarray,
    expected_execution_key_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    if type(rows) is not list or len(rows) != len(VERDICT_BATCH_SIZES):
        raise G120ProductionAuthorityV2Error("physical prediction batch census differs")
    reopened: list[dict[str, Any]] = []
    total = 0
    pair_start = 0
    for batch_index, batch_size in enumerate(VERDICT_BATCH_SIZES):
        pair_stop = pair_start + batch_size
        row = rows[batch_index]
        if type(row) is not dict or row.get("batch_index") != batch_index:
            raise G120ProductionAuthorityV2Error("prediction batch chronology differs")
        if set(row) != _BATCH_FIELDS:
            raise G120ProductionAuthorityV2Error("prediction batch field census differs")
        physical = row.get("physical_receipt")
        raw = _reopen_binding(
            physical,
            name=f"prediction batch {batch_index} receipt",
        )
        try:
            disk_row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise G120ProductionAuthorityV2Error("prediction batch receipt is corrupt") from exc
        embedded = dict(row)
        embedded.pop("physical_receipt", None)
        if disk_row != embedded or _canonical_json(disk_row) != raw:
            raise G120ProductionAuthorityV2Error("embedded/physical prediction batch receipt differs")
        body = dict(disk_row)
        identity = body.pop("row_identity_sha256", None)
        if identity != _sha256(_canonical_json(body)):
            raise G120ProductionAuthorityV2Error("prediction batch row identity differs")
        if (
            disk_row.get("pair_start") != pair_start
            or disk_row.get("pair_stop") != pair_stop
            or (
                expected_execution_key_sha256 is not None
                and disk_row.get("measurement_execution_key_sha256") != expected_execution_key_sha256
            )
        ):
            raise G120ProductionAuthorityV2Error("prediction batch pair span differs")
        prediction_payload = _reopen_binding(
            disk_row.get("prediction_file"),
            name=f"prediction batch {batch_index} array",
        )
        try:
            predicted = np.load(
                io.BytesIO(prediction_payload),
                allow_pickle=False,
            )
        except (OSError, ValueError) as exc:
            raise G120ProductionAuthorityV2Error("prediction array cannot be physically reopened") from exc
        expected_shape = (batch_size, *PRODUCTION_SEG_HW)
        target = target_labels[pair_start:pair_stop]
        if (
            type(predicted) is not np.ndarray
            or predicted.dtype != np.uint8
            or predicted.shape != expected_shape
            or not predicted.flags.c_contiguous
            or np.any(predicted >= 5)
        ):
            raise G120ProductionAuthorityV2Error("physical prediction array geometry differs")
        disagreement = int(np.count_nonzero(predicted != target))
        if (
            disk_row.get("predicted_labels_batch_sha256") != _sha256(memoryview(predicted))
            or disk_row.get("target_labels_batch_sha256") != _sha256(memoryview(target))
            or disk_row.get("disagreement_pixels") != disagreement
        ):
            raise G120ProductionAuthorityV2Error("physical prediction array/count differs")
        reopened.append(row)
        total += disagreement
        pair_start = pair_stop
    if pair_start != PRODUCTION_PAIR_COUNT:
        raise AssertionError("physical prediction batches lost n600")
    return reopened, total


def _build_observation_receipt(
    *,
    measurement: G120StageMeasurementV2,
    snapshot: DynamicFrontierTargetSnapshot,
) -> dict[str, Any]:
    score_decimal, numerator, denominator = _exact_target_from_snapshot(snapshot)
    pointer_identity = _v1.dynamic_snapshot_identity_sha256(snapshot)
    source_float = dict(measurement.receipt["source_float_seg"])
    regret = dict(measurement.receipt["wire_regret"])
    qat = dict(measurement.receipt["g115_qat"])
    k = measurement.receipt["public_wire_seg"]["disagreement_pixels"]
    wire_open = 100 * k * denominator < numerator * PIXEL_DENOMINATOR
    if qat["status"] != "terminal_stage_measured":
        qat = {
            "status": "not_required" if wire_open else "required_unmeasured",
            "terminal_stage_physical_identity_sha256": None,
            "disagreement_pixels": None,
            "pixel_denominator": PIXEL_DENOMINATOR,
            "receipt": None,
        }
    obstruction = exact_prepose_obstruction(
        disagreement_pixels=k,
        pixel_denominator=PIXEL_DENOMINATOR,
        target_numerator=numerator,
        target_denominator=denominator,
        source_float_seg=source_float,
        wire_regret=regret,
        g115_qat=qat,
    )
    receipt: dict[str, Any] = {
        "schema": OBSERVATION_SCHEMA,
        "evidence_axis": EVIDENCE_AXIS,
        "stage_tag": measurement.receipt["stage_tag"],
        "observation_identity_sha256": None,
        "measurement_receipt": {
            "path": str(measurement.receipt_path),
            "bytes": measurement.receipt_bytes,
            "sha256": measurement.receipt_sha256,
        },
        "measurement_identity_sha256": (measurement.measurement_identity_sha256),
        "public_wire_seg": dict(measurement.receipt["public_wire_seg"]),
        "live_target": {
            "score_decimal": score_decimal,
            "score_rational": {
                "numerator": numerator,
                "denominator": denominator,
            },
            "pointer_snapshot_identity_sha256": pointer_identity,
            "postverified_pointer_identity_sha256": pointer_identity,
        },
        "prepose_obstruction": obstruction,
        "source_float_seg": source_float,
        "wire_regret": regret,
        "g115_qat": qat,
        "pointer_snapshot": dataclasses.asdict(snapshot),
        "pointer_reverified_at_atomic_commit": True,
        "semantic_only": True,
        "false_authority": {
            "contest_score_claim": False,
            "candidate_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
        },
    }
    receipt["observation_identity_sha256"] = _receipt_identity(
        receipt,
        identity_field="observation_identity_sha256",
    )
    return receipt


@dataclass(frozen=True, slots=True)
class G120StageMeasurementV2:
    receipt_path: Path
    receipt_sha256: str
    receipt_bytes: int
    receipt: dict[str, Any]

    @property
    def schema(self) -> str:
        return MEASUREMENT_SCHEMA

    @property
    def measurement_identity_sha256(self) -> str:
        return str(self.receipt["measurement_identity_sha256"])

    @property
    def g121_stage_measurement(self) -> dict[str, Any]:
        return {
            "stage_tag": self.receipt["stage_tag"],
            "measurement_receipt": {
                "path": str(self.receipt_path),
                "bytes": self.receipt_bytes,
                "sha256": self.receipt_sha256,
            },
            "measurement_identity_sha256": self.measurement_identity_sha256,
            "public_wire_seg": dict(self.receipt["public_wire_seg"]),
            "physical_stage_identity": dict(self.receipt["physical_stage_identity"]),
            "physical_stage_identity_sha256": self.receipt["physical_stage_identity_sha256"],
            "pose_initializer_identity_sha256": self.receipt["pose_initializer_identity_sha256"],
            "selected_archive": dict(self.receipt["selected_archive"]),
            "alternatives": list(self.receipt["alternatives"]),
            "public_runtime_tree": dict(self.receipt["public_runtime"]),
            "source_float_seg": dict(self.receipt["source_float_seg"]),
            "wire_regret": dict(self.receipt["wire_regret"]),
            "g115_qat": dict(self.receipt["g115_qat"]),
        }


@dataclass(frozen=True, slots=True)
class G120StageObservationV2:
    receipt_path: Path
    receipt_sha256: str
    receipt_bytes: int
    receipt: dict[str, Any]

    @property
    def schema(self) -> str:
        return OBSERVATION_SCHEMA

    @property
    def observation_identity_sha256(self) -> str:
        return str(self.receipt["observation_identity_sha256"])


@dataclass(frozen=True, slots=True)
class G120ProductionStageResultV2:
    production_receipt_path: Path
    production_receipt_sha256: str
    measurement_receipt_path: Path
    measurement_receipt_sha256: str
    observation_receipt_path: Path
    observation_receipt_sha256: str
    measurement: G120StageMeasurementV2
    observation: G120StageObservationV2
    production_receipt: dict[str, Any]

    @property
    def g121_stage_measurement(self) -> dict[str, Any]:
        row = dict(self.measurement.g121_stage_measurement)
        row.update(
            {
                "live_target": dict(self.observation.receipt["live_target"]),
                "prepose_obstruction": dict(self.observation.receipt["prepose_obstruction"]),
                "observation_receipt": {
                    "path": str(self.observation_receipt_path),
                    "bytes": self.observation.receipt_bytes,
                    "sha256": self.observation_receipt_sha256,
                },
                "observation_identity_sha256": (self.observation.observation_identity_sha256),
            }
        )
        return row

    def to_g121_stage_measurement_v2(self) -> dict[str, Any]:
        return self.g121_stage_measurement


def run_g120_parsed_stage_production_authority_v2(
    *,
    repo_root: Path,
    g112_partition_receipt: Path,
    expected_g112_partition_receipt_sha256: str,
    out_dir: Path,
    progress_dir: Path,
    measurement_cache_dir: Path,
    prior_measurement_receipt: Path | None = None,
    expected_prior_measurement_receipt_sha256: str | None = None,
) -> G120ProductionStageResultV2:
    """Measure or physically reopen one exact stage and bind a live observation."""
    if not isinstance(repo_root, Path) or not repo_root.is_absolute():
        raise G120ProductionAuthorityV2Error("repo_root must be an absolute pathlib.Path")
    if (prior_measurement_receipt is None) is not (expected_prior_measurement_receipt_sha256 is None):
        raise G120ProductionAuthorityV2Error("prior measurement path and external SHA must be supplied together")
    expected_g112 = _require_sha256(
        expected_g112_partition_receipt_sha256,
        name="expected G112 partition receipt",
    )
    durable_out = _durable_dir(out_dir, name="out_dir")
    durable_progress = _durable_dir(progress_dir, name="progress_dir")
    cache_root = _ssd_cache_dir(measurement_cache_dir)
    snapshot = load_dynamic_frontier_target(repo_root=repo_root)
    pointer_identity = _v1.dynamic_snapshot_identity_sha256(snapshot)
    _exact_target_from_snapshot(snapshot)

    if prior_measurement_receipt is not None:
        measurement = open_g120_stage_measurement_v2(
            prior_measurement_receipt,
            expected_sha256=expected_prior_measurement_receipt_sha256,
        )
        expected_binding = measurement.receipt["g112_partition_receipt"]
        if (
            expected_binding["path"] != str(g112_partition_receipt.resolve())
            or expected_binding["sha256"] != expected_g112
        ):
            raise G120ProductionAuthorityV2Error("prior measurement belongs to a different physical G112 stage")
        expected_runtime_root = (repo_root / _v1.PUBLIC_RUNTIME_RELATIVE_ROOT).resolve()
        if Path(measurement.receipt["public_runtime"]["source_pre"]["root"]) != expected_runtime_root:
            raise G120ProductionAuthorityV2Error("prior measurement belongs to a different repository runtime")
    else:
        try:
            authority = _v1._open_production_authority(
                repo_root=repo_root,
                g112_partition_receipt=g112_partition_receipt,
                expected_g112_partition_receipt_sha256=expected_g112,
                measurement_cache_dir=cache_root,
            )
        except _v1.G120ProductionAuthorityError as exc:
            raise G120ProductionAuthorityV2Error(str(exc)) from exc
        public_pre, public_payloads = _capture_public_runtime(repo_root)
        if public_pre != authority.public_runtime_tree:
            raise G120ProductionAuthorityV2Error("independently captured public runtime differs from authority")
        sealed_temporary, sealed_root = _seal_public_runtime(
            progress_dir=durable_progress,
            source_identity=public_pre,
            payloads=public_payloads,
        )
        try:
            broker = _FreshPredictionBroker(authority.scorer)
            with tempfile.TemporaryDirectory(
                prefix="g120-v2-fresh-engine-",
                dir=durable_progress,
            ) as engine_temporary:
                engine_root = Path(engine_temporary)
                engine_out = _durable_dir(
                    engine_root / "out",
                    name="private G117 out directory",
                )
                engine_progress = _durable_dir(
                    durable_progress / "g117_engine_progress",
                    name="durable G117 progress directory",
                )
                try:
                    engine = compile_select_parsed_g105_stage_v1(
                        config=authority.config,
                        semantic_params=(
                            authority.g112.semantic_child.shared_params
                        ),
                        odd_y1=authority.g112.semantic_child.code_y1,
                        target_labels=authority.target_labels,
                        seg_argmax_batch_scorer=broker,
                        injected_inputs_are_test_only=True,
                        seg_scorer_identity_sha256=(
                            authority.seg_scorer_identity_sha256
                        ),
                        source_checkpoint_identity_sha256=(
                            authority.physical_stage_identity_sha256
                        ),
                        pose_initializer_identity_sha256=(
                            authority.g112.initializer.checkpoint_sha256
                        ),
                        effective_frontier_target=float(
                            snapshot.target_score
                        ),
                        pointer_snapshot_identity_sha256=pointer_identity,
                        out_dir=engine_out,
                        progress_dir=engine_progress,
                        stage_tag=authority.stage_tag,
                    )
                except G111ParsedG105ExactPrefixObstruction as exc:
                    engine_obstruction = (
                        open_g111_parsed_g105_exact_prefix_obstruction_v1(
                            exc.receipt_path,
                            expected_sha256=exc.receipt_sha256,
                        )
                    )
                    obstruction_postverified = (
                        _postverify_authority_sources(
                            repo_root=repo_root,
                            authority=authority,
                            public_runtime_pre=public_pre,
                        )
                    )
                    scoped = _commit_exact_distortion_obstruction(
                        repo_root=repo_root,
                        out_dir=durable_out,
                        engine_obstruction=engine_obstruction,
                        authority=authority,
                        snapshot=snapshot,
                        public_runtime_pre=public_pre,
                        postverified=obstruction_postverified,
                        scorer_calls=broker.actual_scorer_calls,
                    )
                    raise G120ExactDistortionObstruction(scoped) from exc
                try:
                    _v1._validate_engine_handoff(
                        authority=authority,
                        engine=engine,
                        pointer_identity_sha256=pointer_identity,
                    )
                except _v1.G120ProductionAuthorityError as exc:
                    raise G120ProductionAuthorityV2Error(str(exc)) from exc
                alternatives, selected_row = _persist_alternatives(
                    authority=authority,
                    engine=engine,
                    out_dir=durable_out,
                )
                public = _measure_public_surface_fresh(
                    authority=authority,
                    engine=engine,
                    broker=broker,
                    sealed_root=sealed_root,
                    progress_dir=durable_progress,
                    measurement_cache_dir=cache_root,
                    public_runtime_tree_sha256=public_pre["tree_sha256"],
                    selected_row=selected_row,
                )
            postverified = _postverify_authority_sources(
                repo_root=repo_root,
                authority=authority,
                public_runtime_pre=public_pre,
            )
            measurement_receipt = _build_measurement_receipt(
                authority=authority,
                alternatives=alternatives,
                selected_row=selected_row,
                public=public,
                public_runtime_pre=public_pre,
                postverified=postverified,
                scorer_calls=broker.actual_scorer_calls,
            )
            measurement_path = durable_out / (
                f"{authority.stage_tag}."
                f"{measurement_receipt['measurement_identity_sha256']}"
                ".g120_stage_measurement.v2.json"
            )
            measurement_binding = _immutable_write(
                measurement_path,
                _canonical_json(measurement_receipt),
            )
            measurement = open_g120_stage_measurement_v2(
                measurement_path,
                expected_sha256=measurement_binding["sha256"],
            )
        finally:
            sealed_temporary.cleanup()

    observation_receipt = _build_observation_receipt(
        measurement=measurement,
        snapshot=snapshot,
    )
    observation_path = durable_out / (
        f"{measurement.receipt['stage_tag']}."
        f"{measurement.measurement_identity_sha256}."
        f"{pointer_identity}.g120_stage_observation.v2.json"
    )
    lock_path = repo_root / CANONICAL_FRONTIER_POINTER_LOCK_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_SH)
        try:
            verify_dynamic_frontier_target_snapshot(snapshot)
            observation_binding = _immutable_write(
                observation_path,
                _canonical_json(observation_receipt),
            )
        except Exception as exc:
            if isinstance(exc, G120ProductionAuthorityV2Error):
                raise
            raise G120ProductionAuthorityV2Error("frontier pointer changed before atomic observation commit") from exc
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(lock_fd)
    observation = open_g120_stage_observation_v2(
        observation_path,
        expected_sha256=observation_binding["sha256"],
    )
    production: dict[str, Any] = {
        "schema": PRODUCTION_SCHEMA,
        "production_identity_sha256": None,
        "stage_tag": measurement.receipt["stage_tag"],
        "measurement_receipt": {
            "path": str(measurement.receipt_path),
            "bytes": measurement.receipt_bytes,
            "sha256": measurement.receipt_sha256,
        },
        "observation_receipt": {
            "path": str(observation.receipt_path),
            "bytes": observation.receipt_bytes,
            "sha256": observation.receipt_sha256,
        },
        "measurement_reused": prior_measurement_receipt is not None,
        "production_authority_closed": True,
        "semantic_only": True,
        "false_authority": {
            "contest_score_claim": False,
            "candidate_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
            "production_launch_performed": False,
        },
    }
    production["production_identity_sha256"] = _receipt_identity(
        production,
        identity_field="production_identity_sha256",
    )
    production_path = durable_out / (
        f"{measurement.receipt['stage_tag']}."
        f"{production['production_identity_sha256']}"
        ".g120_production_authority.v2.json"
    )
    production_binding = _immutable_write(
        production_path,
        _canonical_json(production),
    )
    return G120ProductionStageResultV2(
        production_receipt_path=production_path,
        production_receipt_sha256=production_binding["sha256"],
        measurement_receipt_path=measurement.receipt_path,
        measurement_receipt_sha256=measurement.receipt_sha256,
        observation_receipt_path=observation.receipt_path,
        observation_receipt_sha256=observation.receipt_sha256,
        measurement=measurement,
        observation=observation,
        production_receipt=production,
    )


def _reopen_g109_target_custody(
    value: object,
    *,
    g112: Any,
    context: str,
) -> tuple[dict[str, Any], np.ndarray]:
    """Reopen the one G109 target/scorer root shared by all G120 receipts."""

    if type(context) is not str or not context:
        raise G120ProductionAuthorityV2Error(
            "G109 custody context is malformed"
        )
    if (
        type(value) is not dict
        or set(value)
        != {
            "projection_sha256",
            "aggregate_receipt",
            "target_labels_sha256",
            "source_video",
            "segnet_weights",
            "upstream_closure",
            "scorer_runtime_identity_sha256",
        }
        or type(value.get("aggregate_receipt")) is not dict
    ):
        raise G120ProductionAuthorityV2Error(
            f"{context} lacks physical G109 custody"
        )
    g109 = value
    aggregate = g109["aggregate_receipt"]
    scalars = g112.semantic_child.g105_scalars
    projection_json = scalars.get(_v1.CHECKPOINT_PROJECTION_KEY)
    projection_sha = scalars.get(_v1.CHECKPOINT_PROJECTION_SHA_KEY)
    if not isinstance(projection_json, str) or not isinstance(
        projection_sha,
        str,
    ):
        raise G120ProductionAuthorityV2Error(
            f"{context} G112 semantic child lacks its G109 projection"
        )
    try:
        projection = _v1.reopen_v9_training_target_projection(
            projection_json=projection_json,
            expected_projection_sha256=projection_sha,
        )
    except Exception as exc:
        raise G120ProductionAuthorityV2Error(
            f"{context} G112 target projection cannot be reopened"
        ) from exc
    pair = g112.source_chain.current.pair
    if (
        projection_sha != g109["projection_sha256"]
        or projection_sha != g112.initializer.target_projection_sha256
        or projection_sha != pair.target_projection_sha256
        or projection.get("aggregate_receipt") != aggregate
    ):
        raise G120ProductionAuthorityV2Error(
            f"{context} G112/G111/G109 target projection differs"
        )
    loader = V9TrainingTargetCapsuleLoaderV1.open(
        Path(str(aggregate.get("path"))),
        expected_sha256=str(aggregate.get("sha256")),
    )
    target_labels = np.ascontiguousarray(
        loader.targets.seg_labels_u8,
        dtype=np.uint8,
    )
    if (
        loader.pair_count != PRODUCTION_PAIR_COUNT
        or loader.batch_pairs != PRODUCTION_BATCH_PAIRS
        or loader.seg_hw != PRODUCTION_SEG_HW
        or target_labels.shape
        != (PRODUCTION_PAIR_COUNT, *PRODUCTION_SEG_HW)
        or _sha256(memoryview(target_labels))
        != g109.get("target_labels_sha256")
    ):
        raise G120ProductionAuthorityV2Error(
            f"{context} physical G109 labels/geometry differ"
        )
    return g109, target_labels


def open_g120_exact_distortion_obstruction_v1(
    path: Path,
    *,
    expected_sha256: str,
) -> G120ExactDistortionObstructionV1:
    """Strictly reopen one custody-bound engine-prefix scoped blocker."""

    value, physical = _open_canonical_receipt(
        path,
        expected_sha256=expected_sha256,
        schema=EXACT_DISTORTION_OBSTRUCTION_SCHEMA,
        fields=_EXACT_DISTORTION_OBSTRUCTION_FIELDS,
        identity_field="obstruction_identity_sha256",
        name="G120 exact distortion obstruction",
    )
    engine_binding = value["engine_prefix_obstruction"]
    _reopen_binding(
        engine_binding,
        name="G111 exact-prefix obstruction",
    )
    engine = open_g111_parsed_g105_exact_prefix_obstruction_v1(
        Path(engine_binding["path"]),
        expected_sha256=engine_binding["sha256"],
    )
    g112_binding = value["g112_partition_receipt"]
    _reopen_binding(g112_binding, name="G112 partition receipt")
    g112 = open_g112_partition_receipt(
        Path(g112_binding["path"]),
        expected_sha256=g112_binding["sha256"],
    )
    physical_stage, physical_stage_sha, stage_tag = (
        _v1._physical_stage_identity(g112)
    )
    g109, target_labels = _reopen_g109_target_custody(
        value["g109_custody"],
        g112=g112,
        context="scoped obstruction",
    )
    scorer = value["seg_scorer"]
    progress_identity = engine.receipt["progress_identity"]
    live = value["live_target"]
    rational = live.get("score_rational") if type(live) is dict else None
    try:
        snapshot = DynamicFrontierTargetSnapshot(
            **value["pointer_snapshot"]
        )
    except (TypeError, KeyError) as exc:
        raise G120ProductionAuthorityV2Error(
            "G120 scoped obstruction pointer snapshot is malformed"
        ) from exc
    pointer_identity = _v1.dynamic_snapshot_identity_sha256(snapshot)
    public_runtime = value["public_runtime"]
    false_authority = value["false_authority"]
    if (
        value["evidence_axis"] != EVIDENCE_AXIS
        or value["disposition"] != BLOCKED_SCOPED
        or value["verdict_scope"]
        != (
            "one_physical_stage_engine_prefix_without_"
            "sealed_public_prefix_or_cross_wire_equality"
        )
        or value["stage_tag"] != stage_tag
        or engine.receipt["stage_tag"] != stage_tag
        or physical_stage != value["physical_stage_identity"]
        or physical_stage_sha != value["physical_stage_identity_sha256"]
        or progress_identity["source_checkpoint_identity_sha256"]
        != physical_stage_sha
        or progress_identity["target_labels_sha256"]
        != _sha256(memoryview(target_labels))
        or progress_identity["seg_scorer_identity_sha256"]
        != g109["scorer_runtime_identity_sha256"]
        or progress_identity["pose_initializer_identity_sha256"]
        != g112.initializer.checkpoint_sha256
        or progress_identity["pair_count"] != PRODUCTION_PAIR_COUNT
        or progress_identity["batch_sizes"] != list(VERDICT_BATCH_SIZES)
        or type(scorer) is not dict
        or set(scorer)
        != {
            "identity_sha256",
            "device",
            "fresh_direct_scorer_calls",
            "weights_pre",
            "weights_post",
            "upstream_closure_pre",
            "upstream_closure_post",
        }
        or scorer.get("identity_sha256")
        != g109["scorer_runtime_identity_sha256"]
        or scorer.get("device") != "cpu"
        or type(scorer.get("fresh_direct_scorer_calls")) is not int
        or scorer["fresh_direct_scorer_calls"] < 0
        or scorer["fresh_direct_scorer_calls"]
        > engine.receipt["completed_batch_count"]
        or scorer.get("weights_pre") != scorer.get("weights_post")
        or scorer.get("weights_pre") != g109["segnet_weights"]
        or scorer.get("upstream_closure_pre")
        != scorer.get("upstream_closure_post")
        or scorer.get("upstream_closure_pre") != g109["upstream_closure"]
        or type(live) is not dict
        or set(live)
        != {
            "score_decimal",
            "score_rational",
            "pointer_snapshot_identity_sha256",
            "postverified_pointer_identity_sha256",
        }
        or type(rational) is not dict
        or set(rational) != {"numerator", "denominator"}
        or live["score_decimal"]
        != engine.receipt["effective_frontier_target_exact"]["decimal"]
        or rational["numerator"]
        != engine.receipt["effective_frontier_target_exact"]["numerator"]
        or rational["denominator"]
        != engine.receipt["effective_frontier_target_exact"]["denominator"]
        or live["pointer_snapshot_identity_sha256"] != pointer_identity
        or live["postverified_pointer_identity_sha256"] != pointer_identity
        or engine.receipt["progress_identity"][
            "pointer_snapshot_identity_sha256"
        ]
        != pointer_identity
        or value["pointer_reverified_at_atomic_commit"] is not True
        or value["production_authority_closed"] is not False
        or value["semantic_only"] is not True
        or type(public_runtime) is not dict
        or set(public_runtime)
        != {
            "source_pre",
            "sealed_tree_sha256",
            "source_post",
            "sealed_runtime_captured",
            "public_prefix_execution_performed",
            "public_prefix_equality",
            "cross_wire_prefix_equality",
        }
        or _reopen_runtime_tree(public_runtime.get("source_pre"))
        != public_runtime.get("source_pre")
        or public_runtime.get("source_pre")
        != public_runtime.get("source_post")
        or public_runtime.get("sealed_tree_sha256")
        != public_runtime["source_pre"].get("tree_sha256")
        or public_runtime.get("sealed_runtime_captured") is not True
        or public_runtime.get("public_prefix_execution_performed") is not False
        or public_runtime.get("public_prefix_equality") is not False
        or public_runtime.get("cross_wire_prefix_equality") is not False
        or false_authority
        != {
            "contest_score_claim": False,
            "candidate_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
            "public_wire_prune_claim": False,
            "family_wide_claim": False,
        }
    ):
        raise G120ProductionAuthorityV2Error(
            "G120 scoped obstruction custody, target, or authority differs"
        )
    try:
        if (
            _v1._regular_file_identity(
                Path(scorer["weights_pre"]["path"]),
                name="scoped-obstruction SegNet weights",
            )
            != scorer["weights_pre"]
            or _v1._reopen_upstream_closure(
                scorer["upstream_closure_pre"]
            )
            != scorer["upstream_closure_pre"]
        ):
            raise G120ProductionAuthorityV2Error(
                "scoped-obstruction scorer files changed"
            )
    except _v1.G120ProductionAuthorityError as exc:
        raise G120ProductionAuthorityV2Error(str(exc)) from exc
    return G120ExactDistortionObstructionV1(
        receipt_path=path,
        receipt_sha256=physical["sha256"],
        receipt_bytes=physical["bytes"],
        receipt=value,
    )


def open_g120_stage_measurement_v2(
    path: Path,
    *,
    expected_sha256: str,
) -> G120StageMeasurementV2:
    """Strictly reopen one immutable pointer-independent measurement."""
    value, physical = _open_canonical_receipt(
        path,
        expected_sha256=expected_sha256,
        schema=MEASUREMENT_SCHEMA,
        fields=_MEASUREMENT_FIELDS,
        identity_field="measurement_identity_sha256",
        name="G120-v2 measurement receipt",
    )
    if (
        value["evidence_axis"] != EVIDENCE_AXIS
        or value["pointer_independent"] is not True
        or value["pair_count"] != PRODUCTION_PAIR_COUNT
        or value["batch_sizes"] != list(VERDICT_BATCH_SIZES)
        or value["pixel_denominator"] != PIXEL_DENOMINATOR
        or value["production_authority_closed"] is not True
        or value["repository_public_population_equal"] is not True
    ):
        raise G120ProductionAuthorityV2Error("measurement production geometry/status differs")
    g112_binding = value["g112_partition_receipt"]
    _reopen_binding(g112_binding, name="G112 partition receipt")
    g112 = open_g112_partition_receipt(
        Path(g112_binding["path"]),
        expected_sha256=g112_binding["sha256"],
    )
    physical_stage, physical_stage_sha, stage_tag = _v1._physical_stage_identity(g112)
    if (
        physical_stage != value["physical_stage_identity"]
        or physical_stage_sha != value["physical_stage_identity_sha256"]
        or stage_tag != value["stage_tag"]
        or g112.initializer.checkpoint_sha256 != value["pose_initializer_identity_sha256"]
    ):
        raise G120ProductionAuthorityV2Error("measurement G112/deploy/resume/lineage custody differs")
    g109, target_labels = _reopen_g109_target_custody(
        value["g109_custody"],
        g112=g112,
        context="measurement",
    )
    scorer = value["seg_scorer"]
    if (
        type(scorer) is not dict
        or set(scorer)
        != {
            "identity_sha256",
            "device",
            "fresh_direct_scorer_calls",
            "fresh_measured_batch_count",
            "resumed_physical_batch_count",
            "disk_prediction_cache_trusted",
            "weights_pre",
            "weights_post",
            "upstream_closure_pre",
            "upstream_closure_post",
        }
        or scorer.get("identity_sha256") != g109.get("scorer_runtime_identity_sha256")
        or scorer.get("device") != "cpu"
        or type(scorer.get("fresh_direct_scorer_calls")) is not int
        or scorer["fresh_direct_scorer_calls"] < 0
        or type(scorer.get("fresh_measured_batch_count")) is not int
        or scorer["fresh_measured_batch_count"] < 0
        or type(scorer.get("resumed_physical_batch_count")) is not int
        or scorer["resumed_physical_batch_count"] < 0
        or (
            scorer["fresh_measured_batch_count"]
            + scorer["resumed_physical_batch_count"]
            != len(VERDICT_BATCH_SIZES)
        )
        or scorer["fresh_direct_scorer_calls"]
        > scorer["fresh_measured_batch_count"]
        or (
            scorer["fresh_measured_batch_count"] > 0
            and scorer["fresh_direct_scorer_calls"] == 0
        )
        or scorer.get("disk_prediction_cache_trusted") is not False
        or scorer.get("weights_pre") != scorer.get("weights_post")
        or scorer.get("upstream_closure_pre") != scorer.get("upstream_closure_post")
    ):
        raise G120ProductionAuthorityV2Error("measurement scorer custody differs")
    try:
        if (
            _v1._regular_file_identity(
                Path(scorer["weights_pre"]["path"]),
                name="measurement SegNet weights",
            )
            != scorer["weights_pre"]
            or _v1._reopen_upstream_closure(scorer["upstream_closure_pre"]) != scorer["upstream_closure_pre"]
        ):
            raise G120ProductionAuthorityV2Error("measurement scorer files changed")
    except _v1.G120ProductionAuthorityError as exc:
        raise G120ProductionAuthorityV2Error(str(exc)) from exc
    public_runtime = value["public_runtime"]
    if (
        type(public_runtime) is not dict
        or set(public_runtime)
        != {
            "source_pre",
            "sealed_tree_sha256",
            "source_post",
            "toctou_closed_by",
        }
        or _reopen_runtime_tree(public_runtime["source_pre"]) != public_runtime["source_pre"]
        or public_runtime["source_pre"] != public_runtime["source_post"]
        or public_runtime["sealed_tree_sha256"] != public_runtime["source_pre"]["tree_sha256"]
        or public_runtime["toctou_closed_by"] != "sealed_content_execution_plus_source_postverify"
    ):
        raise G120ProductionAuthorityV2Error("measurement public-runtime TOCTOU closure differs")
    selected = _validate_alternatives(
        value["alternatives"],
        selected_identity=value["selected_alternative_identity_sha256"],
    )
    if value["selected_archive"] != selected["archive"]:
        raise G120ProductionAuthorityV2Error("measurement selected archive differs from four-way matrix")
    execution_body = {
        "schema": BATCH_SCHEMA,
        "physical_stage_identity_sha256": value["physical_stage_identity_sha256"],
        "selected_archive_sha256": selected["archive"]["sha256"],
        "archive_alternatives": sorted(
            [
                {
                    "y1_wire_codec": item["y1_wire_codec"],
                    "outer_zip_method": item["outer_zip_method"],
                    "archive_bytes": item["archive"]["bytes"],
                    "archive_sha256": item["archive"]["sha256"],
                }
                for item in value["alternatives"]
            ],
            key=lambda row: (
                row["y1_wire_codec"],
                row["outer_zip_method"],
            ),
        ),
        "target_labels_sha256": _sha256(memoryview(target_labels)),
        "seg_scorer_identity_sha256": value["seg_scorer"]["identity_sha256"],
        "public_runtime_tree_sha256": value["public_runtime"]["sealed_tree_sha256"],
        "pair_count": PRODUCTION_PAIR_COUNT,
        "batch_sizes": list(VERDICT_BATCH_SIZES),
        "frontier_pointer_intentionally_excluded": True,
    }
    batch_rows, disagreements = _reopen_prediction_batches(
        value["prediction_batches"],
        target_labels=target_labels,
        expected_execution_key_sha256=_sha256(_canonical_json(execution_body)),
    )
    if value["prediction_batch_receipt_chain_sha256"] != _batch_chain(
        batch_rows,
        "row_identity_sha256",
    ):
        raise G120ProductionAuthorityV2Error("measurement physical prediction receipt chain differs")
    if (
        any(item["disagreement_pixels"] != disagreements for item in value["alternatives"])
        or selected["scorer_y1_population_sha256"] != _batch_chain(batch_rows, "scorer_y1_batch_sha256")
        or selected["camera_y1_population_sha256"] != _batch_chain(batch_rows, "camera_y1_batch_sha256")
        or selected["predicted_labels_population_sha256"] != _batch_chain(batch_rows, "predicted_labels_batch_sha256")
    ):
        raise G120ProductionAuthorityV2Error("measurement repository/public population equality differs")
    public_wire = value["public_wire_seg"]
    if (
        type(public_wire) is not dict
        or set(public_wire)
        != {
            "disagreement_pixels",
            "pixel_denominator",
            "d_seg_rational",
            "d_seg_display_float",
            "measurement_identity_sha256",
        }
        or public_wire["disagreement_pixels"] != disagreements
        or public_wire["pixel_denominator"] != PIXEL_DENOMINATOR
        or public_wire["d_seg_rational"] != {"numerator": disagreements, "denominator": PIXEL_DENOMINATOR}
        or public_wire["d_seg_display_float"] != disagreements / PIXEL_DENOMINATOR
        or public_wire["measurement_identity_sha256"] != value["measurement_identity_sha256"]
        or selected["disagreement_pixels"] != disagreements
    ):
        raise G120ProductionAuthorityV2Error("measurement exact public-wire coordinate differs")
    _validate_status_coordinates(
        source_float_seg=value["source_float_seg"],
        wire_regret=value["wire_regret"],
        g115_qat=value["g115_qat"],
    )
    if value["semantic_only"] is not True or value["false_authority"] != {
        "contest_score_claim": False,
        "candidate_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "production_launch_performed": False,
        "g117_receipt_is_authority": False,
        "legacy_batch_cache_trusted": False,
    }:
        raise G120ProductionAuthorityV2Error("measurement false-authority flags differ")
    return G120StageMeasurementV2(
        receipt_path=path.resolve(),
        receipt_sha256=physical["sha256"],
        receipt_bytes=physical["bytes"],
        receipt=value,
    )


def open_g120_stage_observation_v2(
    path: Path,
    *,
    expected_sha256: str,
) -> G120StageObservationV2:
    """Strictly reopen one immutable exact-pointer observation."""
    value, physical = _open_canonical_receipt(
        path,
        expected_sha256=expected_sha256,
        schema=OBSERVATION_SCHEMA,
        fields=_OBSERVATION_FIELDS,
        identity_field="observation_identity_sha256",
        name="G120-v2 observation receipt",
    )
    measurement_binding = value["measurement_receipt"]
    _reopen_binding(
        measurement_binding,
        name="observation measurement receipt",
    )
    measurement = open_g120_stage_measurement_v2(
        Path(measurement_binding["path"]),
        expected_sha256=measurement_binding["sha256"],
    )
    if (
        value["stage_tag"] != measurement.receipt["stage_tag"]
        or value["measurement_identity_sha256"] != measurement.measurement_identity_sha256
        or value["public_wire_seg"] != measurement.receipt["public_wire_seg"]
    ):
        raise G120ProductionAuthorityV2Error("observation measurement binding differs")
    source_float, regret, qat = _validate_status_coordinates(
        source_float_seg=value["source_float_seg"],
        wire_regret=value["wire_regret"],
        g115_qat=value["g115_qat"],
    )
    live = value["live_target"]
    if (
        type(live) is not dict
        or set(live)
        != {
            "score_decimal",
            "score_rational",
            "pointer_snapshot_identity_sha256",
            "postverified_pointer_identity_sha256",
        }
        or type(live.get("score_decimal")) is not str
        or type(live.get("score_rational")) is not dict
        or set(live["score_rational"]) != {"numerator", "denominator"}
    ):
        raise G120ProductionAuthorityV2Error("observation exact live-target coordinate differs")
    try:
        exact_decimal = Decimal(live["score_decimal"])
        exact_fraction = Fraction(exact_decimal)
    except (InvalidOperation, ValueError, ZeroDivisionError) as exc:
        raise G120ProductionAuthorityV2Error("observation target Decimal is invalid") from exc
    if (
        not exact_decimal.is_finite()
        or exact_decimal <= 0
        or live["score_rational"]
        != {
            "numerator": exact_fraction.numerator,
            "denominator": exact_fraction.denominator,
        }
    ):
        raise G120ProductionAuthorityV2Error("observation target Decimal/rational differs")
    try:
        snapshot = DynamicFrontierTargetSnapshot(**value["pointer_snapshot"])
    except (TypeError, ValueError) as exc:
        raise G120ProductionAuthorityV2Error("observation pointer snapshot is malformed") from exc
    snapshot_identity = _v1.dynamic_snapshot_identity_sha256(snapshot)
    if (
        float(exact_decimal) != snapshot.target_score
        or live["pointer_snapshot_identity_sha256"] != snapshot_identity
        or live["postverified_pointer_identity_sha256"] != snapshot_identity
        or value["pointer_reverified_at_atomic_commit"] is not True
    ):
        raise G120ProductionAuthorityV2Error("observation pointer identity/exact target differs")
    expected_obstruction = exact_prepose_obstruction(
        disagreement_pixels=value["public_wire_seg"]["disagreement_pixels"],
        pixel_denominator=PIXEL_DENOMINATOR,
        target_numerator=exact_fraction.numerator,
        target_denominator=exact_fraction.denominator,
        source_float_seg=source_float,
        wire_regret=regret,
        g115_qat=qat,
    )
    if value["prepose_obstruction"] != expected_obstruction:
        raise G120ProductionAuthorityV2Error("observation exact obstruction/disposition differs")
    if value["semantic_only"] is not True or value["false_authority"] != {
        "contest_score_claim": False,
        "candidate_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }:
        raise G120ProductionAuthorityV2Error("observation false-authority flags differ")
    return G120StageObservationV2(
        receipt_path=path.resolve(),
        receipt_sha256=physical["sha256"],
        receipt_bytes=physical["bytes"],
        receipt=value,
    )


__all__ = [
    "BATCH_SCHEMA",
    "BLOCKED_SCOPED",
    "DEFER_G115_WIRE_QAT",
    "EVIDENCE_AXIS",
    "MEASUREMENT_SCHEMA",
    "MINIMUM_SAFE_AUTHORITY_COMPLETE",
    "OBSERVATION_SCHEMA",
    "PIXEL_DENOMINATOR",
    "PRODUCTION_SCHEMA",
    "PRUNE_EXACT_DISTORTION_OBSTRUCTION",
    "RETAIN_POST_G105_POSE",
    "G120ProductionAuthorityV2Error",
    "G120ProductionStageResultV2",
    "G120StageMeasurementV2",
    "G120StageObservationV2",
    "open_g120_stage_measurement_v2",
    "open_g120_stage_observation_v2",
    "run_g120_parsed_stage_production_authority_v2",
]
