# SPDX-License-Identifier: MIT
"""Fail-closed builder surfaces for the Task #603 direct-description minimizer.

This module does not launch an optimization.  It defines a counted-description
schema, a deterministic legacy-control repacker, typed DSL custody, exact cap
arithmetic, checkpoint serialization/restore audits, and fail-closed evidence
verifiers.  It does not implement the PRIMARY receiver, optimizer, or stage
continuation loop.  The current specification has ``execution_allowed=false``.

The first archive compiler deliberately reuses the already measured S4 carrier
container only to prove byte-exact legacy re-expression.  Its six section names
are opaque aliases and do not establish PRIMARY semantic ownership or receiver
consumption.  A later carrier may replace it only behind a new schema/version
and fresh source-bound receiver evidence.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, InvalidOperation, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from tac.artifact_quarantine import is_quarantined_archive_bytes, scan_text
from tac.optimization.s4_archive_composer import (
    SECTION_ORDER,
    SectionBytes,
    deterministic_archive,
    parse_sections,
    serialize_sections,
)

PRIMARY_SPEC_REL: Final = ".omx/research/direct_description_minimizer_PRIMARY_SPEC_20260721T214800Z.md"
PRIMARY_SPEC_SHA256: Final = "3ae7166e633d46e6341c3565e0fd2d475c501e6120fa68c419ce89e931f37aa9"
SOURCE_BYTES: Final = 37_545_489
SEED: Final = 1234
POINTER_SCORE_TEXT: Final = "0.1910828242"
STRICT_SCORE_TEXT: Final = "0.15"
RATE_LAMBDA: Final = Fraction(25, SOURCE_BYTES)
TOLERANCE_RUNG_TEXT: Final = ("0.000152", "0.000300", "0.000500", "0.000800")
EVIDENCE_ROOT: Final = "/Volumes/VertigoDataTier/pact/evidence/ddm_builder_20260721"
OWNER_BUNDLE_REL: Final = (
    ".omx/research/direct_description_minimizer_owner_bundle_603_20260721T225631Z.json"
)
S4_BASELINE_ARCHIVE: Final = (
    "/Volumes/VertigoDataTier/pact/evidence/s4_composer_20260721/canonical_s4_20260721/archive.zip"
)
S4_BASELINE_BYTES: Final = 451_191
S4_BASELINE_SHA256: Final = "d84f2fe053239d1542ba381420e9569d431ed2015e22e60e49ef48f1321696ed"
_REPO_ROOT: Final = Path(__file__).resolve().parents[3]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_STREAM_TO_SECTION: Final = {
    "entropy_state": "manifest.json",
    "xi_curve_knots": "seed.ppcs",
    "static_ground_coefficients": "base.pbase3",
    "pose6_dxi_residuals": "causal.pcr3",
    "sparse_events": "events.pce3",
    "exceptions": "components.pcomp3",
}
_SECTION_TO_STREAM: Final = {value: key for key, value in _STREAM_TO_SECTION.items()}
_SECTION_CODEC: Final = {
    "manifest.json": "raw",
    "seed.ppcs": "raw",
    "base.pbase3": "mixed",
    "causal.pcr3": "raw",
    "events.pce3": "lzma1_raw_1MiB",
    "components.pcomp3": "zlib9",
}
_APPLICABLE_STRATA: Final = {
    "MyCar": frozenset({"cell_interior", "boundary_codim1"}),
    "Undrivable": frozenset({"cell_interior", "boundary_codim1"}),
    "Road": frozenset({"cell_interior", "boundary_codim1"}),
    "Lane": frozenset({"cell_interior", "boundary_codim1"}),
    "Movable": frozenset({"boundary_codim1", "movable_track", "critical_event"}),
}


class DirectDescriptionError(ValueError):
    """A typed description, custody record, or launch precondition is invalid."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DirectDescriptionError(f"{field} must be a lowercase SHA-256")
    return value


def _require_exact_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DirectDescriptionError(f"{field} must be a nonnegative exact integer")
    return value


def _json_string(value: str) -> str:
    if not isinstance(value, str):
        raise DirectDescriptionError("RFC 8785 object keys and strings must be strings")
    try:
        value.encode("utf-8", "strict")
        value.encode("utf-16-be", "strict")
    except UnicodeEncodeError as exc:
        raise DirectDescriptionError("RFC 8785 refuses lone Unicode surrogates") from exc
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _ecmascript_number(value: int | float) -> str:
    """Render an I-JSON number using ECMAScript/JCS exponent thresholds.

    CPython and ECMAScript both use shortest round-tripping binary64 digits.
    Their presentation thresholds differ, so the shortest CPython digit string
    is normalized here to ECMAScript's fixed/scientific boundary.
    """

    if isinstance(value, bool):
        raise DirectDescriptionError("booleans are not JCS numbers")
    if isinstance(value, int):
        if abs(value) > 9_007_199_254_740_991:
            raise DirectDescriptionError("JCS integer exceeds the I-JSON exact range")
        return str(value)
    if not isinstance(value, float) or not math.isfinite(value):
        raise DirectDescriptionError("JCS numbers must be finite binary64 values")
    if value == 0.0:
        return "0"
    sign = "-" if value < 0 else ""
    text = repr(abs(value)).lower()
    mantissa, marker, exponent_text = text.partition("e")
    exponent = int(exponent_text) if marker else 0
    integer, dot, fraction = mantissa.partition(".")
    if not marker and dot and fraction == "0":
        return sign + integer
    digits = (integer + (fraction if dot else "")).lstrip("0")
    if not digits:
        return "0"
    decimal_position = len(integer) + exponent
    leading_removed = len(integer + (fraction if dot else "")) - len((integer + (fraction if dot else "")).lstrip("0"))
    decimal_position -= leading_removed
    if -6 <= decimal_position - 1 < 21:
        if decimal_position <= 0:
            return sign + "0." + "0" * (-decimal_position) + digits
        if decimal_position >= len(digits):
            return sign + digits + "0" * (decimal_position - len(digits))
        return sign + digits[:decimal_position] + "." + digits[decimal_position:]
    scientific_exponent = decimal_position - 1
    coefficient = digits[0] + (("." + digits[1:]) if len(digits) > 1 else "")
    exponent_sign = "+" if scientific_exponent >= 0 else ""
    return f"{sign}{coefficient}e{exponent_sign}{scientific_exponent}"


def rfc8785_canonicalize(value: Any) -> bytes:
    """Return RFC 8785 JSON Canonicalization Scheme UTF-8 bytes.

    The accepted domain is I-JSON: string keys, valid Unicode, finite binary64
    numbers, and exact integers no larger than 2**53-1.  Rejecting a wider
    Python domain prevents silent numeric coercion before custody hashing.
    """

    def render(item: Any) -> str:
        if item is None:
            return "null"
        if item is True:
            return "true"
        if item is False:
            return "false"
        if isinstance(item, (int, float)):
            return _ecmascript_number(item)
        if isinstance(item, str):
            return _json_string(item)
        if isinstance(item, (list, tuple)):
            return "[" + ",".join(render(element) for element in item) + "]"
        if isinstance(item, Mapping):
            keys = list(item)
            if any(not isinstance(key, str) for key in keys):
                raise DirectDescriptionError("JCS object keys must be strings")
            try:
                ordered = sorted(keys, key=lambda key: key.encode("utf-16-be", "strict"))
            except UnicodeEncodeError as exc:
                raise DirectDescriptionError("JCS refuses lone Unicode surrogates") from exc
            return "{" + ",".join(_json_string(key) + ":" + render(item[key]) for key in ordered) + "}"
        raise DirectDescriptionError(f"value of type {type(item).__name__} is not I-JSON")

    return render(value).encode("utf-8")


def _duplicate_refusing_json(payload: bytes) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DirectDescriptionError(f"duplicate JSON object name: {key!r}")
            result[key] = value
        return result

    try:
        text = payload.decode("utf-8", "strict")
        return json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                DirectDescriptionError(f"non-finite JSON number {token!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DirectDescriptionError("payload is not strict UTF-8 JSON") from exc


class CountedDescriptionStreamV1(BaseModel):
    """One exact counted byte stream owned by a direct-description field."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    payload: bytes
    codec: StrictStr
    decoded_bytes: StrictInt = Field(ge=0)
    semantic_status: Literal["LEGACY_OPAQUE_SECTION_REEXPRESSION"] = (
        "LEGACY_OPAQUE_SECTION_REEXPRESSION"
    )

    @model_validator(mode="after")
    def _coherent(self) -> CountedDescriptionStreamV1:
        if self.payload and self.decoded_bytes == 0:
            raise ValueError("a nonempty counted stream cannot claim zero decoded bytes")
        return self


class DirectDescriptionZV1(BaseModel):
    """The complete counted archive description; unknown fields are forbidden."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    static_ground_coefficients: CountedDescriptionStreamV1
    xi_curve_knots: CountedDescriptionStreamV1
    pose6_dxi_residuals: CountedDescriptionStreamV1
    sparse_events: CountedDescriptionStreamV1
    entropy_state: CountedDescriptionStreamV1
    exceptions: CountedDescriptionStreamV1

    @model_validator(mode="after")
    def _registered_codecs(self) -> DirectDescriptionZV1:
        for stream_name, section_name in _STREAM_TO_SECTION.items():
            stream = getattr(self, stream_name)
            expected = _SECTION_CODEC[section_name]
            if stream.codec != expected:
                raise ValueError(f"{stream_name} codec {stream.codec!r} != registered {expected!r}")
        return self

    def to_s4_sections(self) -> tuple[SectionBytes, ...]:
        """Map six legacy opaque aliases onto the settled S4 section registry."""

        by_section: dict[str, SectionBytes] = {}
        for stream_name, section_name in _STREAM_TO_SECTION.items():
            stream = getattr(self, stream_name)
            by_section[section_name] = SectionBytes(
                name=section_name,
                payload=stream.payload,
                codec=stream.codec,
                decoded_bytes=stream.decoded_bytes,
            )
        return tuple(by_section[name] for name in SECTION_ORDER)

    @classmethod
    def from_s4_sections(cls, sections: Sequence[SectionBytes]) -> DirectDescriptionZV1:
        rows = tuple(sections)
        if tuple(row.name for row in rows) != SECTION_ORDER:
            raise DirectDescriptionError("S4 sections are missing or reordered")
        values: dict[str, CountedDescriptionStreamV1] = {}
        for row in rows:
            values[_SECTION_TO_STREAM[row.name]] = CountedDescriptionStreamV1(
                payload=row.payload,
                codec=row.codec,
                decoded_bytes=row.decoded_bytes,
                semantic_status="LEGACY_OPAQUE_SECTION_REEXPRESSION",
            )
        return cls(**values)

    def stream_ledger(self) -> list[dict[str, Any]]:
        """Exact payload ledger; these bytes are not mislabelled as ZIP bytes."""

        rows: list[dict[str, Any]] = []
        for stream_name, section_name in _STREAM_TO_SECTION.items():
            stream = getattr(self, stream_name)
            rows.append(
                {
                    "stream": stream_name,
                    "carrier_section": section_name,
                    "encoded_payload_bytes": len(stream.payload),
                    "decoded_bytes": stream.decoded_bytes,
                    "codec": stream.codec,
                    "sha256": _sha256(stream.payload),
                    "semantic_status": stream.semantic_status,
                    "claim_kind": "MEASURED_LEGACY_OPAQUE_SECTION_NOT_PRIMARY_OWNER_SEMANTICS",
                }
            )
        return rows


@dataclass(frozen=True, slots=True)
class DirectArchiveBuildResult:
    """Exact result of compiling one ``z`` through canonical ``A``."""

    archive: bytes
    member: bytes
    z: DirectDescriptionZV1

    def custody(self) -> dict[str, Any]:
        stream_payload_bytes = sum(row["encoded_payload_bytes"] for row in self.z.stream_ledger())
        return {
            "schema": "direct_description_archive_build.v1",
            "compiler": "s4_archive_composer.v1",
            "archive_bytes": len(self.archive),
            "archive_sha256": _sha256(self.archive),
            "member_bytes": len(self.member),
            "member_sha256": _sha256(self.member),
            "stream_payload_bytes": stream_payload_bytes,
            "container_framing_bytes": len(self.member) - stream_payload_bytes,
            "archive_bytes_source": "len(exact_A_of_z_bytes)",
            "stream_ledger": self.z.stream_ledger(),
            "receiver_consumption_verified": False,
            "receiver_consumption_blocker": (
                "S4 receiver rejects nonempty pose6/dxi and legacy sections do not prove "
                "the six PRIMARY semantic owners"
            ),
        }


def compile_direct_description_archive(z: DirectDescriptionZV1) -> DirectArchiveBuildResult:
    """Compile ``z`` to exact deterministic container bytes without receiver authority."""

    if not isinstance(z, DirectDescriptionZV1):
        raise DirectDescriptionError("z must be DirectDescriptionZV1")
    member = serialize_sections(z.to_s4_sections())
    with tempfile.TemporaryDirectory(prefix="ddm_A_of_z_") as temporary:
        path = Path(temporary) / "archive.zip"
        deterministic_archive(path, member)
        archive = path.read_bytes()
    return DirectArchiveBuildResult(archive=archive, member=member, z=z)


def _read_canonical_archive(archive: bytes | Path) -> tuple[bytes, str]:
    if isinstance(archive, Path):
        payload = _read_regular_file_once(archive)
        hits = [*is_quarantined_archive_bytes(payload), *scan_text(str(archive))]
        if hits:
            identifiers = ", ".join(sorted({hit.identifier for hit in hits}))
            raise DirectDescriptionError(
                "PRIMARY archive consumption refuses quarantine hits even when the "
                f"signal-only environment waiver is set: {identifiers}"
            )
        return payload, str(archive)
    if not isinstance(archive, bytes) or not archive:
        raise DirectDescriptionError("archive must be nonempty exact bytes or a Path")
    byte_hits = is_quarantined_archive_bytes(archive)
    if byte_hits:
        identifiers = ", ".join(hit.identifier for hit in byte_hits)
        raise DirectDescriptionError(
            "PRIMARY archive byte consumption refuses quarantine hits: " + identifiers
        )
    return archive, "fresh_compiler_output_bytes"


def parse_direct_description_archive(archive: bytes | Path) -> DirectArchiveBuildResult:
    """Reopen, strictly parse, and canonical-reencode one exact archive.

    Path inputs pass the non-waivable PRIMARY quarantine check.  Byte inputs are
    intended only for the immediate output of :func:`compile_direct_description_archive`.
    Canonical recompile equality rejects trailing ZIP bytes and metadata drift.
    """

    archive_bytes, _source = _read_canonical_archive(archive)
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as handle:
            members = handle.infolist()
            if len(members) != 1 or members[0].filename != "0.bin":
                raise DirectDescriptionError("archive must contain exactly canonical 0.bin")
            member = handle.read("0.bin")
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        raise DirectDescriptionError("archive ZIP is malformed") from exc
    sections = parse_sections(member)
    z = DirectDescriptionZV1.from_s4_sections(sections)
    rebuilt = compile_direct_description_archive(z)
    if rebuilt.member != member:
        raise DirectDescriptionError("archive member parse/re-encode identity failed")
    if rebuilt.archive != archive_bytes:
        raise DirectDescriptionError("archive ZIP is not the canonical A(z) encoding")
    return rebuilt


def prove_baseline_reexpression(archive_path: Path = Path(S4_BASELINE_ARCHIVE)) -> dict[str, Any]:
    """Measure whether the settled S4 control is byte-exact under the typed ``A``."""

    parsed = parse_direct_description_archive(Path(archive_path))
    custody = parsed.custody()
    expected_bytes_match = len(parsed.archive) == S4_BASELINE_BYTES
    expected_sha_match = _sha256(parsed.archive) == S4_BASELINE_SHA256
    return {
        "schema": "direct_description_z0_reexpression.v1",
        "source_archive_path": str(archive_path),
        "source_archive_bytes": len(parsed.archive),
        "source_archive_sha256": _sha256(parsed.archive),
        "parse_reencode_archive_byte_exact": True,
        "settled_control_bytes_match": expected_bytes_match,
        "settled_control_sha256_match": expected_sha_match,
        "verdict": "PASS_BYTE_EXACT" if expected_bytes_match and expected_sha_match else "EXACT_DELTA",
        "archive_custody": custody,
        "legacy_opaque_alias_mapping": dict(_STREAM_TO_SECTION),
        "primary_owner_semantics_proven": False,
        "primary_owner_semantics_blocker": (
            "manifest/seed/causal S4 sections are not entropy/xi/Pose6 live-owner proofs"
        ),
    }


class DirectDescriptionOpsGrammarMinimizerV1(BaseModel):
    """Locked declarative contract for a future PRIMARY minimizer implementation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DirectDescriptionOpsGrammarMinimizerV1"] = Field(
        default="DirectDescriptionOpsGrammarMinimizerV1",
        alias="schema",
        serialization_alias="schema",
    )
    objective_domain: Literal["exact_contest_score"] = "exact_contest_score"
    decision_variable: Literal["complete_counted_archive_description"] = "complete_counted_archive_description"
    grammar: Literal["static_ground_coefficients_plus_xi_curve_plus_sparse_events"] = (
        "static_ground_coefficients_plus_xi_curve_plus_sparse_events"
    )
    carrier_source: Literal["governed_v8_v9_503_reuse"] = "governed_v8_v9_503_reuse"
    rate_source: Literal["exact_len_of_final_archive_zip_A_of_z"] = "exact_len_of_final_archive_zip_A_of_z"
    receiver_domain: Literal["integer_uint8_R"] = "integer_uint8_R"
    post_hoc_composition: Literal["forbidden"] = "forbidden"
    pose_owner: Literal["counted_pose6_dxi_residuals_inside_xi_grammar"] = (
        "counted_pose6_dxi_residuals_inside_xi_grammar"
    )
    pairwise_pose_targets_compose_to_absolute: Literal["forbidden_without_coordinate_receipt"] = (
        "forbidden_without_coordinate_receipt"
    )
    solve_order: Literal["seg_cells_then_pose_tube_within_seg_feasible_polytope"] = (
        "seg_cells_then_pose_tube_within_seg_feasible_polytope"
    )
    admission_authority: Literal["hard_integer_uint8_receiver_oracle"] = "hard_integer_uint8_receiver_oracle"
    step_metric_readback: tuple[StrictStr, ...] = (
        "euclidean_cosine",
        "fisher_cosine",
        "relative_norm_ratio",
        "extensible_additional_metrics",
    )
    seed: StrictInt = SEED
    n64_role: Literal["deterministic_receiver_and_custody_smoke_only"] = "deterministic_receiver_and_custody_smoke_only"
    n600_required_for_admission: Literal[True] = True
    all_stage_checkpoints_preserved: Literal[True] = True
    execution_allowed: Literal[False] = False
    research_only: Literal[True] = True
    evidence_root: Literal[EVIDENCE_ROOT] = EVIDENCE_ROOT
    tolerance_rungs: tuple[Decimal, ...] = tuple(Decimal(value) for value in TOLERANCE_RUNG_TEXT)
    allocation_axis_order: tuple[StrictStr, ...] = (
        "class",
        "canonical_stratum",
        "temporal_segment_or_event_window",
        "class_pair_boundary",
        "frequency_or_scale",
    )
    rate_lambda_numerator: Literal[25] = 25
    rate_lambda_denominator: Literal[SOURCE_BYTES] = SOURCE_BYTES
    governed_launcher: Literal["tools/launch_witness_run.py"] = "tools/launch_witness_run.py"
    memory_preflight: Literal["tools/witness_memory_preflight.py"] = "tools/witness_memory_preflight.py"
    full_precision_target_receipt_required: Literal[True] = True
    operator_go_required: Literal[True] = True

    @model_validator(mode="after")
    def _sealed_values(self) -> DirectDescriptionOpsGrammarMinimizerV1:
        if self.seed != SEED or isinstance(self.seed, bool):
            raise ValueError(f"seed must be the exact integer {SEED}")
        expected_rungs = tuple(Decimal(value) for value in TOLERANCE_RUNG_TEXT)
        if self.tolerance_rungs != expected_rungs:
            raise ValueError(f"tolerance_rungs must be exactly {TOLERANCE_RUNG_TEXT!r}")
        expected_axes = (
            "class",
            "canonical_stratum",
            "temporal_segment_or_event_window",
            "class_pair_boundary",
            "frequency_or_scale",
        )
        if self.allocation_axis_order != expected_axes:
            raise ValueError("allocation_axis_order is a sealed recursive custody tree")
        if self.step_metric_readback != (
            "euclidean_cosine",
            "fisher_cosine",
            "relative_norm_ratio",
            "extensible_additional_metrics",
        ):
            raise ValueError("dual-metric readback fields are sealed and extensible")
        return self


class DirectDescriptionWitnessProgramV1(BaseModel):
    """Canonical DDM compile target; it never routes through a level-set trainer."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DirectDescriptionWitnessProgramV1"] = Field(
        default="DirectDescriptionWitnessProgramV1",
        alias="schema",
        serialization_alias="schema",
    )
    consumer: Literal["tools/run_direct_description_minimizer.py"] = (
        "tools/run_direct_description_minimizer.py"
    )
    owner_manifest_path: Literal[OWNER_BUNDLE_REL] = OWNER_BUNDLE_REL
    mode: Literal["preflight"] = "preflight"
    execution_allowed: Literal[False] = False

    def compile_consumer_argv(self) -> tuple[str, ...]:
        argv = (
            "/usr/bin/env",
            "python3",
            self.consumer,
            "--owner-manifest",
            self.owner_manifest_path,
            "--mode",
            self.mode,
            "--execution-allowed",
            "false",
        )
        build_direct_description_arg_parser().parse_args(list(argv[3:]))
        return argv


class DirectDescriptionTypedConfigV1(BaseModel):
    """Typed config and Lever custody for the direct-description consumer itself."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DirectDescriptionTypedConfigV1"] = Field(
        default="DirectDescriptionTypedConfigV1",
        alias="schema",
        serialization_alias="schema",
    )
    name: Literal["direct_description_ops_grammar_minimizer_v1"] = (
        "direct_description_ops_grammar_minimizer_v1"
    )
    owner: DirectDescriptionOpsGrammarMinimizerV1
    owner_manifest_path: Literal[OWNER_BUNDLE_REL] = OWNER_BUNDLE_REL
    custody_levers: tuple[dict[str, Any], ...] = ()

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))

    def to_program(self) -> DirectDescriptionWitnessProgramV1:
        return DirectDescriptionWitnessProgramV1(owner_manifest_path=self.owner_manifest_path)

    def program_manifest(self) -> dict[str, Any]:
        program = self.to_program()
        argv = program.compile_consumer_argv()
        return {
            "schema": "direct_description_program_manifest.v1",
            "compile_target": program.schema_,
            "typed_config_hash": self.typed_config_hash(),
            "consumer_argv": list(argv),
            "consumer_argv_sha256": _sha256("\0".join(argv).encode("utf-8")),
            "execution_allowed": False,
        }


def _strip_volatile(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _strip_volatile(child)
            for key, child in value.items()
            if key not in {"compiled_at_utc", "resolved_at_utc", "resolved_at"}
        }
    if isinstance(value, list):
        return [_strip_volatile(child) for child in value]
    return value


def build_direct_description_owner() -> dict[str, Any]:
    """Compile the locked owner through a DDM WitnessProgram/Lever/LawRef target."""

    from tac.witness_dsl.lawref import (
        LADDER_HARDCODED_WAIVER,
        InputRef,
        LawRef,
        lawref_to_declaration,
    )
    from tac.witness_dsl.typed_config import TypedLever

    owner = DirectDescriptionOpsGrammarMinimizerV1()
    pre = DirectDescriptionTypedConfigV1(owner=owner)
    pre_hash = pre.typed_config_hash()
    pre_argv = pre.to_program().compile_consumer_argv()
    seed_ref = LawRef(
        equation_id="dsl_custodied_scalar_identity_v1",
        inputs={
            "value": InputRef.literal(
                SEED,
                "PRIMARY spec exact deterministic seed; identity custody only, no derivation claim",
            )
        },
        ladder_class=LADDER_HARDCODED_WAIVER,
        fallback=SEED,
        fallback_waiver_reason=("operator-sealed PRIMARY seed; rederive only when a superseding spec changes it"),
    )
    from tac.witness_dsl.lawref import resolve_flag_dict_constants

    resolved_seed, seed_constant_manifest = resolve_flag_dict_constants({"--seed": seed_ref})
    if resolved_seed != {"--seed": SEED}:
        raise DirectDescriptionError("seed LawRef identity compiler changed the sealed value")
    owner_lever = TypedLever(
        name="direct_description_ops_grammar_minimizer_owner",
        overrides={"--seed": SEED},
        notes="custody attached last; argv-value-neutral; execution_allowed:false",
        lawrefs={"--seed": seed_ref},
        lawref_declarations={"--seed": lawref_to_declaration(seed_ref)},
        constant_manifest=seed_constant_manifest,
        runtime_receipt_schemas={"--seed": "direct_description_runtime_consumption.v1"},
        policy_contracts={"direct_description_owner": owner.model_dump(mode="json", by_alias=True)},
    )
    lever_manifest = owner_lever.model_dump(mode="json")
    post = pre.model_copy(update={"custody_levers": (lever_manifest,)})
    post_argv = post.to_program().compile_consumer_argv()
    if post_argv != pre_argv:
        raise DirectDescriptionError("custody attachment changed compiled argv")
    post_hash = post.typed_config_hash()
    if pre_hash == post_hash:
        raise DirectDescriptionError("pre/post custody hashes unexpectedly alias")
    program_manifest = post.program_manifest()
    if program_manifest["typed_config_hash"] != post_hash:
        raise DirectDescriptionError("standard typed_config_hash is not the post-custody hash")
    bundle: dict[str, Any] = {
        "schema": "direct_description_typed_owner_bundle.v1",
        "compile_target": "direct_description_minimizer.v1",
        "owner": owner.model_dump(mode="json", by_alias=True),
        "typed_config": post.model_dump(mode="json", by_alias=True),
        "pre_custody_typed_config_hash": pre_hash,
        "typed_config_hash": post_hash,
        "program_manifest": program_manifest,
        # Durable owner bytes must reproduce across invocations; resolution time
        # is operational telemetry, not semantic compiler input.
        "custody_constants_manifest": _strip_volatile(seed_constant_manifest),
        "custody_argv_byte_identical": True,
        "custody_argv": list(post_argv),
        "consumer_argv": list(post_argv),
        "execution_allowed": False,
        "launch_readiness": "BLOCKED_RECEIVER_AND_RUNNER_NOT_IMPLEMENTED",
    }
    semantic = _strip_volatile(bundle)
    bundle["dsl_compile_hash"] = _sha256(rfc8785_canonicalize(semantic))
    return bundle


def build_direct_description_arg_parser() -> argparse.ArgumentParser:
    """The real consumer parser used by both compiler and CLI entry point."""

    parser = argparse.ArgumentParser(description="Task #603 direct-description PRIMARY consumer")
    parser.add_argument("--owner-manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "optimize"), required=True)
    parser.add_argument("--execution-allowed", choices=("false", "true"), required=True)
    parser.add_argument("--operator-go", type=Path)
    parser.add_argument("--resume-from", type=Path)
    return parser


def numpy_reference_rank(candidates: Sequence[Mapping[str, Any]], *, seed: int = SEED) -> list[dict[str, Any]]:
    """Deterministically rank complete same-artifact tuples by the exact score.

    This is a small NumPy reference, not an authority evaluation.  Candidates
    must already contain realized-through-R ``d_seg``, ``d_pose``, and exact
    ``archive_bytes`` from their own ``A(z)``.  Seeded random values break exact
    score ties reproducibly without changing any score.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DirectDescriptionError("seed must be a nonnegative exact integer")
    rng = np.random.default_rng(seed)
    ranked: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        identifier = candidate.get("candidate_id")
        if not isinstance(identifier, str) or not identifier:
            raise DirectDescriptionError(f"candidate {index} lacks candidate_id")
        d_seg = candidate.get("d_seg")
        d_pose = candidate.get("d_pose")
        archive_bytes = candidate.get("archive_bytes")
        if (
            isinstance(d_seg, bool)
            or not isinstance(d_seg, (int, float))
            or not math.isfinite(float(d_seg))
            or float(d_seg) < 0
        ):
            raise DirectDescriptionError(f"candidate {identifier} has invalid d_seg")
        if (
            isinstance(d_pose, bool)
            or not isinstance(d_pose, (int, float))
            or not math.isfinite(float(d_pose))
            or float(d_pose) < 0
        ):
            raise DirectDescriptionError(f"candidate {identifier} has invalid d_pose")
        _require_exact_nonnegative_int(archive_bytes, f"{identifier}.archive_bytes")
        score = (
            np.float64(100.0) * np.float64(d_seg)
            + np.sqrt(np.float64(10.0) * np.float64(d_pose))
            + np.float64(25.0) * np.float64(archive_bytes) / np.float64(SOURCE_BYTES)
        )
        ranked.append(
            {
                "candidate_id": identifier,
                "d_seg": float(d_seg),
                "d_pose": float(d_pose),
                "archive_bytes": archive_bytes,
                "score": float(score),
                "tie_break": float(rng.random()),
                "axis": "numpy-fp64 deterministic reference; not contest authority",
            }
        )
    return sorted(ranked, key=lambda row: (row["score"], row["tie_break"], row["candidate_id"]))


def derive_ceil_minus_one_caps(target_receipt_path: Path, expected_sha256: str) -> dict[str, Any]:
    """Derive binding integer caps from a SHA-bound full-precision receipt.

    Decimal strings are mandatory.  JSON floats and planning-only receipts are
    refused so displayed rounded values cannot leak into a launch config.
    """

    expected_sha256 = _require_sha256(expected_sha256, "expected_sha256")
    path = Path(target_receipt_path)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise DirectDescriptionError(f"full-precision target receipt is unreadable: {path}") from exc
    observed_sha256 = _sha256(payload)
    if observed_sha256 != expected_sha256:
        raise DirectDescriptionError("full-precision target receipt SHA-256 mismatch")
    receipt = _duplicate_refusing_json(payload)
    if not isinstance(receipt, dict):
        raise DirectDescriptionError("full-precision target receipt must be an object")
    if rfc8785_canonicalize(receipt) + b"\n" != payload:
        raise DirectDescriptionError("full-precision target receipt must be canonical JCS plus one LF")
    if receipt.get("schema") != "direct_description_full_precision_target.v1":
        raise DirectDescriptionError("full-precision target receipt schema mismatch")
    if receipt.get("authority") != "official_frozen_evaluator_solved_target":
        raise DirectDescriptionError("target receipt is not an official frozen-evaluator authority row")
    if receipt.get("hardware_axis") not in {"[contest-CPU]", "[contest-CUDA]"}:
        raise DirectDescriptionError("target receipt has no contest authority axis")
    if receipt.get("launch_config_admissible") is not True or receipt.get("planning_only") is True:
        raise DirectDescriptionError("planning/display target values cannot enter launch config")
    if receipt.get("pairs") != 600 or isinstance(receipt.get("pairs"), bool):
        raise DirectDescriptionError("full-precision target receipt must bind pairs=600")
    d_seg_text = receipt.get("solved_d_seg")
    d_pose_text = receipt.get("solved_d_pose")
    if not isinstance(d_seg_text, str) or not isinstance(d_pose_text, str):
        raise DirectDescriptionError("solved_d_seg and solved_d_pose must be full-precision strings")
    prohibited_display_pair = ("0.00015196", "0.00010184")
    if (d_seg_text, d_pose_text) == prohibited_display_pair:
        raise DirectDescriptionError("display-rounded planning targets cannot enter launch config")
    for name, text_value in (("solved_d_seg", d_seg_text), ("solved_d_pose", d_pose_text)):
        coefficient = text_value.lower().split("e", 1)[0]
        significant_digits = coefficient.replace("+", "").replace("-", "").replace(".", "").lstrip("0")
        if len(significant_digits) < 17:
            raise DirectDescriptionError(f"{name} does not preserve a full binary64-decimal authority value")
    evaluator_path = Path(str(receipt.get("evaluator_path")))
    scorer_runtime_path = Path(str(receipt.get("scorer_runtime_path")))
    derivation_path = Path(str(receipt.get("target_derivation_receipt_path")))
    evaluator_sha = _require_sha256(receipt.get("evaluator_sha256"), "evaluator_sha256")
    scorer_sha = _require_sha256(receipt.get("scorer_runtime_sha256"), "scorer_runtime_sha256")
    derivation_sha = _require_sha256(
        receipt.get("target_derivation_receipt_sha256"), "target_derivation_receipt_sha256"
    )
    if _sha256(_read_regular_file_once(evaluator_path)) != evaluator_sha:
        raise DirectDescriptionError("target receipt evaluator custody mismatch")
    if _sha256(_read_regular_file_once(scorer_runtime_path)) != scorer_sha:
        raise DirectDescriptionError("target receipt scorer-runtime custody mismatch")
    derivation_payload = _read_regular_file_once(derivation_path)
    if _sha256(derivation_payload) != derivation_sha:
        raise DirectDescriptionError("target derivation receipt SHA-256 mismatch")
    derivation = _duplicate_refusing_json(derivation_payload[:-1] if derivation_payload.endswith(b"\n") else derivation_payload)
    if not isinstance(derivation, Mapping) or derivation.get("schema") != "direct_description_solved_target_derivation.v1":
        raise DirectDescriptionError("target derivation receipt schema mismatch")
    if rfc8785_canonicalize(derivation) + b"\n" != derivation_payload:
        raise DirectDescriptionError("target derivation receipt is not canonical JCS plus one LF")
    required_derivation = {
        "pairs": 600,
        "through_R": True,
        "exact_evaluator_called": True,
        "evaluator_sha256": evaluator_sha,
        "scorer_runtime_sha256": scorer_sha,
        "solved_d_seg": d_seg_text,
        "solved_d_pose": d_pose_text,
    }
    if any(derivation.get(key) != value for key, value in required_derivation.items()):
        raise DirectDescriptionError("target derivation receipt does not bind the official solved tuple")
    try:
        d_seg = Decimal(d_seg_text)
        d_pose = Decimal(d_pose_text)
    except InvalidOperation as exc:
        raise DirectDescriptionError("solved target decimal string is malformed") from exc
    if not d_seg.is_finite() or not d_pose.is_finite() or d_seg < 0 or d_pose < 0:
        raise DirectDescriptionError("solved target distortions must be finite and nonnegative")
    with localcontext() as context:
        context.prec = 100
        nonrate = Decimal(100) * d_seg + (Decimal(10) * d_pose).sqrt()

        def cap(score_text: str) -> tuple[int, Decimal]:
            continuous = (Decimal(score_text) - nonrate) * Decimal(SOURCE_BYTES) / Decimal(25)
            if continuous <= 0:
                raise DirectDescriptionError("solved target leaves no positive archive budget")
            integer_cap = int(continuous.to_integral_value(rounding=ROUND_CEILING)) - 1
            return integer_cap, continuous

        pointer_cap, pointer_continuous = cap(POINTER_SCORE_TEXT)
        strict_cap, strict_continuous = cap(STRICT_SCORE_TEXT)
    return {
        "schema": "direct_description_ceil_minus_one_caps.v1",
        "full_precision_target_receipt_path": str(path),
        "full_precision_target_receipt_sha256": observed_sha256,
        "solved_d_seg": d_seg_text,
        "solved_d_pose": d_pose_text,
        "pointer_score": POINTER_SCORE_TEXT,
        "pointer_cap_bytes": pointer_cap,
        "pointer_continuous_bytes": format(pointer_continuous, "f"),
        "pointer_cap_formula": "ceil_minus_one",
        "strict_score": STRICT_SCORE_TEXT,
        "strict_0_15_cap_bytes": strict_cap,
        "strict_continuous_bytes": format(strict_continuous, "f"),
        "strict_cap_formula": "ceil_minus_one",
        "strict_cap_role": "stretch_only",
    }


class ToleranceAllocationNodeV1(BaseModel):
    """One uniquely owned node in the preregistered recursive allocation tree."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    node_id: StrictStr
    parent_id: StrictStr | None = None
    axis: Literal[
        "class",
        "canonical_stratum",
        "temporal_segment_or_event_window",
        "class_pair_boundary",
        "frequency_or_scale",
    ]
    label: StrictStr
    chart: StrictStr
    tolerance: Decimal
    archive_bytes: StrictInt = Field(ge=0)
    d_seg_contribution: Decimal = Field(ge=0)
    marginal_gain_numerator: StrictInt = Field(ge=0)
    marginal_gain_denominator: StrictInt = Field(gt=0)
    quantization_floor: Decimal = Field(ge=0)
    receipt_sha256: StrictStr

    @model_validator(mode="after")
    def _node_custody(self) -> ToleranceAllocationNodeV1:
        if _SAFE_COMPONENT_RE.fullmatch(self.node_id) is None:
            raise ValueError("node_id must be a safe nonempty identifier")
        if self.parent_id is not None and _SAFE_COMPONENT_RE.fullmatch(self.parent_id) is None:
            raise ValueError("parent_id must be a safe identifier or null")
        if not self.label or not self.chart:
            raise ValueError("label and chart must be nonempty")
        if self.tolerance not in {Decimal(value) for value in TOLERANCE_RUNG_TEXT}:
            raise ValueError("node tolerance is not on the preregistered ladder")
        if not self.d_seg_contribution.is_finite() or not self.quantization_floor.is_finite():
            raise ValueError("allocation node decimals must be finite")
        _require_sha256(self.receipt_sha256, "receipt_sha256")
        return self


def verify_allocation_tree(nodes: Sequence[ToleranceAllocationNodeV1]) -> dict[str, Any]:
    """Verify the recursive tree's structure and exact common-lambda conditions.

    This does not claim final-ZIP byte attribution.  The current S4 carrier is
    one monolithic DEFLATE member, so receiver-rate custody remains a separate
    red gate even when this mathematical tree is structurally valid.
    """

    rows = tuple(nodes)
    if not rows or any(not isinstance(row, ToleranceAllocationNodeV1) for row in rows):
        raise DirectDescriptionError("allocation tree must contain typed nodes")
    by_id = {row.node_id: row for row in rows}
    if len(by_id) != len(rows):
        raise DirectDescriptionError("allocation node IDs must be unique")
    axis_order = DirectDescriptionOpsGrammarMinimizerV1().allocation_axis_order
    axis_index = {axis: index for index, axis in enumerate(axis_order)}
    class_roots = [row for row in rows if row.axis == "class"]
    expected_classes = {"MyCar", "Undrivable", "Road", "Lane", "Movable"}
    if {row.label for row in class_roots} != expected_classes or any(row.parent_id is not None for row in class_roots):
        raise DirectDescriptionError("allocation tree requires exactly five root class owners")
    children: dict[str, list[ToleranceAllocationNodeV1]] = {key: [] for key in by_id}
    for row in rows:
        if row.axis == "class":
            continue
        if row.parent_id not in by_id:
            raise DirectDescriptionError(f"allocation node {row.node_id} has an unknown parent")
        parent = by_id[row.parent_id]
        if axis_index[row.axis] != axis_index[parent.axis] + 1:
            raise DirectDescriptionError("allocation tree skipped or reordered a custody axis")
        children[parent.node_id].append(row)
    visited: set[str] = set()

    def visit(node_id: str, active: set[str]) -> None:
        if node_id in active:
            raise DirectDescriptionError("allocation tree contains a cycle")
        if node_id in visited:
            return
        active.add(node_id)
        for child in children[node_id]:
            visit(child.node_id, active)
        active.remove(node_id)
        visited.add(node_id)

    for root in class_roots:
        visit(root.node_id, set())
    if visited != set(by_id):
        raise DirectDescriptionError("allocation tree contains unreachable nodes")
    leaves = [row for row in rows if not children[row.node_id]]
    if any(row.axis != "frequency_or_scale" for row in leaves):
        raise DirectDescriptionError(
            "allocation tree leaves must reach frequency_or_scale; gate exclusions belong in receipts"
        )
    for root in class_roots:
        strata = {
            child.label
            for child in children[root.node_id]
            if child.axis == "canonical_stratum"
        }
        if strata != _APPLICABLE_STRATA[root.label]:
            raise DirectDescriptionError(
                f"allocation tree has incomplete canonical strata for {root.label}: {sorted(strata)}"
            )
    for row in rows:
        owner = row
        while owner.parent_id is not None:
            owner = by_id[owner.parent_id]
        if owner.label in {"Road", "Lane"} and row.chart not in {
            "corrected_xi",
            "image_edge_fallback",
        }:
            raise DirectDescriptionError("Road/Lane allocation uses an ungoverned coordinate chart")
    for parent_id, child_rows in children.items():
        if not child_rows:
            continue
        parent = by_id[parent_id]
        if sum(child.archive_bytes for child in child_rows) != parent.archive_bytes:
            raise DirectDescriptionError(f"allocation bytes do not reconcile under {parent_id}")
        if sum((child.d_seg_contribution for child in child_rows), Decimal(0)) != (parent.d_seg_contribution):
            raise DirectDescriptionError(f"d_seg contribution does not reconcile under {parent_id}")
    residuals: list[Fraction] = []
    for row in rows:
        residual = abs(
            Fraction(row.marginal_gain_numerator, row.marginal_gain_denominator) - RATE_LAMBDA
        )
        floor = Fraction(row.quantization_floor)
        if residual > floor:
            raise DirectDescriptionError(
                f"allocation marginal residual exceeds quantization floor at {row.node_id}"
            )
        residuals.append(residual)
    maximum = max(residuals)
    return {
        "schema": "direct_description_allocation_tree_verification.v1",
        "node_count": len(rows),
        "class_roots": sorted(expected_classes),
        "archive_bytes_unique_home_total": sum(row.archive_bytes for row in class_roots),
        "d_seg_contribution_total": format(sum((row.d_seg_contribution for row in class_roots), Decimal(0)), "f"),
        "common_lambda_exact": f"{RATE_LAMBDA.numerator}/{RATE_LAMBDA.denominator}",
        "maximum_absolute_marginal_residual_exact": (f"{maximum.numerator}/{maximum.denominator}"),
        "acyclic": True,
        "axis_exhaustive": True,
        "structurally_valid": True,
        "verified": False,
        "admission_blocker": "FINAL_ZIP_RECEIVER_RATE_CUSTODY_UNAVAILABLE_FOR_MONOLITHIC_DEFLATE",
    }


class ChargedFreePartitionRowV1(BaseModel):
    """One STEP-0 component classified before description optimization."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    component: StrictStr
    disposition: Literal["FREE", "NULL", "COUNTED"]
    justification: StrictStr
    bytes_if_counted: StrictInt | None = Field(default=None, ge=0)
    receipt_sha256: StrictStr
    video_derived: StrictBool
    generic_decoder_logic_only: StrictBool

    @model_validator(mode="after")
    def _partition_law(self) -> ChargedFreePartitionRowV1:
        if not self.component or not self.justification:
            raise ValueError("partition component and justification must be nonempty")
        _require_sha256(self.receipt_sha256, "receipt_sha256")
        if self.disposition == "COUNTED":
            if self.bytes_if_counted is None or not self.video_derived:
                raise ValueError("COUNTED residue requires exact bytes and video-derived custody")
            if self.generic_decoder_logic_only:
                raise ValueError("COUNTED residue cannot be labelled generic free logic")
        else:
            if self.bytes_if_counted not in (None, 0):
                raise ValueError("FREE/NULL components cannot claim counted bytes")
            if self.disposition == "FREE" and (self.video_derived or not self.generic_decoder_logic_only):
                raise ValueError("FREE requires generic, non-video-specific decoder logic")
            if self.disposition == "NULL" and self.generic_decoder_logic_only:
                raise ValueError("NULL is scorer-invisible omission, not decoder free code")
        return self


def verify_charged_free_partition(rows: Sequence[ChargedFreePartitionRowV1]) -> dict[str, Any]:
    """Verify the FREE/NULL/COUNTED decision-variable partition without fake savings."""

    values = tuple(rows)
    if not values or any(not isinstance(row, ChargedFreePartitionRowV1) for row in values):
        raise DirectDescriptionError("charged/free partition requires typed rows")
    components = [row.component for row in values]
    if len(set(components)) != len(components):
        raise DirectDescriptionError("charged/free partition components must be unique")
    required_counted = set(_STREAM_TO_SECTION)
    present_counted = {row.component for row in values if row.disposition == "COUNTED"}
    if not required_counted <= present_counted:
        raise DirectDescriptionError("charged/free partition omits one or more complete-description owners")
    counted_bytes = sum(int(row.bytes_if_counted or 0) for row in values if row.disposition == "COUNTED")
    return {
        "schema": "direct_description_charged_free_partition_verification.v1",
        "structurally_valid": True,
        "verified": False,
        "counts": {
            disposition: sum(row.disposition == disposition for row in values)
            for disposition in ("FREE", "NULL", "COUNTED")
        },
        "counted_description_payload_bytes": counted_bytes,
        "counted_description_payload_is_archive_bytes": False,
        "null_byte_savings": None,
        "null_byte_savings_blocker": (
            "geometric nullity is not subtractable until parser-consumed A(z) byte delta is measured"
        ),
        "admission_blocker": "SOURCE_BOUND_COMPONENT_RECEIPTS_NOT_REOPENED_BY_THIS_SCHEMA_CHECK",
    }


class DescriptionStepMetricTelemetryV1(BaseModel):
    """Per-step dual-metric readback; metrics steer but never admit."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    step: StrictInt = Field(ge=0)
    tolerance_rung: StrictStr
    euclidean_cosine: float
    fisher_cosine: float
    relative_norm_ratio: float = Field(ge=0)
    additional_metrics: dict[StrictStr, float] = Field(default_factory=dict)
    exact_secant_direction_sha256: StrictStr
    hard_receiver_admitted: StrictBool

    @model_validator(mode="after")
    def _finite_metrics(self) -> DescriptionStepMetricTelemetryV1:
        if self.tolerance_rung not in TOLERANCE_RUNG_TEXT:
            raise ValueError("metric row tolerance is not preregistered")
        if not math.isfinite(self.euclidean_cosine) or not -1 <= self.euclidean_cosine <= 1:
            raise ValueError("Euclidean cosine must be finite in [-1,1]")
        if not math.isfinite(self.fisher_cosine) or not -1 <= self.fisher_cosine <= 1:
            raise ValueError("Fisher cosine must be finite in [-1,1]")
        if not math.isfinite(self.relative_norm_ratio):
            raise ValueError("relative norm ratio must be finite")
        for name, value in self.additional_metrics.items():
            if not name or not math.isfinite(value):
                raise ValueError("additional metric names/values must be nonempty and finite")
        _require_sha256(self.exact_secant_direction_sha256, "exact_secant_direction_sha256")
        return self


class MeasurementRungRowV1(BaseModel):
    """One same-artifact tolerance row with explicit conditioning scope."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tolerance: StrictStr
    archive_path: StrictStr
    archive_bytes: StrictInt = Field(ge=0)
    archive_sha256: StrictStr
    d_seg: float = Field(ge=0)
    d_pose: float = Field(ge=0)
    pairs: Literal[600] = 600
    generator_lineage: StrictStr
    seg_constraints_solved_first: Literal[True] = True
    pose_solved_within_seg_feasible_polytope: Literal[True] = True
    receipt_path: StrictStr
    receipt_sha256: StrictStr

    @model_validator(mode="after")
    def _same_artifact_rung(self) -> MeasurementRungRowV1:
        if self.tolerance not in TOLERANCE_RUNG_TEXT:
            raise ValueError("measurement tolerance is not preregistered")
        if not math.isfinite(self.d_seg) or not math.isfinite(self.d_pose):
            raise ValueError("rung distortions must be finite")
        if Decimal(str(self.d_seg)) > Decimal(self.tolerance):
            raise ValueError("measured d_seg exceeds its declared tolerance rung")
        if not self.generator_lineage:
            raise ValueError("generator lineage is required")
        _require_sha256(self.archive_sha256, "archive_sha256")
        _require_sha256(self.receipt_sha256, "receipt_sha256")
        parsed = parse_direct_description_archive(Path(self.archive_path))
        if len(parsed.archive) != self.archive_bytes or _sha256(parsed.archive) != self.archive_sha256:
            raise ValueError("measurement rung does not bind its exact canonical A(z) archive")
        receipt_payload = _read_regular_file_once(Path(self.receipt_path))
        if _sha256(receipt_payload) != self.receipt_sha256:
            raise ValueError("measurement rung receipt SHA-256 mismatch")
        if not receipt_payload.endswith(b"\n"):
            raise ValueError("measurement rung receipt must end in one LF")
        receipt = _duplicate_refusing_json(receipt_payload[:-1])
        if not isinstance(receipt, Mapping) or receipt.get("schema") != "direct_description_n600_measurement.v1":
            raise ValueError("measurement rung receipt schema mismatch")
        if rfc8785_canonicalize(receipt) + b"\n" != receipt_payload:
            raise ValueError("measurement rung receipt must be canonical JCS")
        required = {
            "pairs": 600,
            "archive_bytes": self.archive_bytes,
            "archive_sha256": self.archive_sha256,
            "d_seg": self.d_seg,
            "d_pose": self.d_pose,
            "generator_lineage": self.generator_lineage,
            "through_R": True,
            "exact_evaluator_called": True,
            "seg_constraints_solved_first": True,
            "pose_solved_within_seg_feasible_polytope": True,
        }
        if any(receipt.get(key) != value for key, value in required.items()):
            raise ValueError("measurement rung receipt does not bind the row")
        if receipt.get("hardware_axis") not in {"[contest-CPU]", "[contest-CUDA]"}:
            raise ValueError("measurement rung receipt lacks a contest authority axis")
        raw_paths = receipt.get("raw_decode_paths_x2")
        raw_hashes = receipt.get("raw_decode_sha256_x2")
        if (
            not isinstance(raw_paths, list)
            or len(raw_paths) != 2
            or not isinstance(raw_hashes, list)
            or len(raw_hashes) != 2
            or raw_hashes[0] != raw_hashes[1]
        ):
            raise ValueError("measurement rung requires deterministic double-decode custody")
        raw_payloads = _read_distinct_double_decode(raw_paths)
        for raw_payload, raw_hash in zip(raw_payloads, raw_hashes, strict=True):
            _require_sha256(raw_hash, "raw_decode_sha256")
            if _sha256(raw_payload) != raw_hash:
                raise ValueError("measurement rung raw decode hash mismatch")
        return self


def verify_measurement_ladder(rows: Sequence[MeasurementRungRowV1]) -> dict[str, Any]:
    """Validate four source-bound row receipts without minting authority locally."""

    values = tuple(rows)
    if len(values) != len(TOLERANCE_RUNG_TEXT) or any(not isinstance(row, MeasurementRungRowV1) for row in values):
        raise DirectDescriptionError("measurement ladder requires exactly four typed n600 rows")
    by_tolerance = {row.tolerance: row for row in values}
    if set(by_tolerance) != set(TOLERANCE_RUNG_TEXT):
        raise DirectDescriptionError("measurement ladder has missing or duplicate tolerances")
    scores_by_archive: dict[str, tuple[float, float]] = {}
    for row in values:
        score_pair = (row.d_seg, row.d_pose)
        prior = scores_by_archive.setdefault(row.archive_sha256, score_pair)
        if prior != score_pair:
            raise DirectDescriptionError("one deterministic archive cannot carry conflicting score tuples")
    scored = numpy_reference_rank(
        [
            {
                "candidate_id": tolerance,
                "archive_bytes": by_tolerance[tolerance].archive_bytes,
                "d_seg": by_tolerance[tolerance].d_seg,
                "d_pose": by_tolerance[tolerance].d_pose,
            }
            for tolerance in TOLERANCE_RUNG_TEXT
        ]
    )
    return {
        "schema": "direct_description_measurement_ladder_verification.v1",
        "structurally_valid": True,
        "verified": False,
        "admission_blocker": "EXTERNAL_OFFICIAL_MEASUREMENT_ATTESTATION_NOT_VERIFIED_BY_LOCAL_BUILDER",
        "solve_order": "seg_cells_then_pose_tube_within_seg_feasible_polytope",
        "rows": [
            {
                "tolerance": tolerance,
                "archive_bytes": by_tolerance[tolerance].archive_bytes,
                "d_seg": by_tolerance[tolerance].d_seg,
                "d_pose": by_tolerance[tolerance].d_pose,
                "score": next(row["score"] for row in scored if row["candidate_id"] == tolerance),
            }
            for tolerance in TOLERANCE_RUNG_TEXT
        ],
        "minimum_score_tolerance": scored[0]["candidate_id"],
    }


class DirectDescriptionStageCheckpointV1(BaseModel):
    """Complete immutable serialization schema for a future stage runner."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DirectDescriptionStageCheckpointV1"] = Field(
        default="DirectDescriptionStageCheckpointV1",
        alias="schema",
        serialization_alias="schema",
    )
    config_sha256: StrictStr
    dsl_compile_hash: StrictStr
    run_id: StrictStr
    stage_name: StrictStr
    stage_phase: Literal["static_grammar", "n64_custody", "tolerance", "n600", "admission"]
    stage_index: StrictInt = Field(ge=0)
    epoch: StrictInt = Field(ge=0)
    global_step: StrictInt = Field(ge=0)
    next_pair: StrictInt = Field(ge=0, le=600)
    active_tolerance: StrictStr
    active_rate_rung: StrictInt = Field(ge=0)
    z_member_b64: StrictStr
    z_member_sha256: StrictStr
    entropy_state: dict[str, Any]
    archive_compiler_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    ema_shadow: dict[str, Any]
    rng_state: dict[str, Any]
    allocation_pool_states: dict[str, Any]
    hard_oracle_receipt_chain: tuple[dict[str, Any], ...]
    step_metric_telemetry: tuple[DescriptionStepMetricTelemetryV1, ...]
    seg_constraints_solved_for_active_tolerance: StrictBool
    pose_solved_within_seg_feasible_polytope: StrictBool
    best_archive_b64: StrictStr
    best_archive_sha256: StrictStr
    best_archive_bytes: StrictInt = Field(ge=0)
    trigger_certificate_state: dict[str, Any]
    config: dict[str, Any]
    argv: tuple[StrictStr, ...]
    argv_sha256: StrictStr

    @model_validator(mode="after")
    def _complete_and_bound(self) -> DirectDescriptionStageCheckpointV1:
        for field in (
            "config_sha256",
            "dsl_compile_hash",
            "z_member_sha256",
            "best_archive_sha256",
            "argv_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if _SAFE_COMPONENT_RE.fullmatch(self.run_id) is None:
            raise ValueError("run_id is not filename safe")
        if _SAFE_COMPONENT_RE.fullmatch(self.stage_name) is None:
            raise ValueError("stage_name is not filename safe")
        if self.active_tolerance not in TOLERANCE_RUNG_TEXT:
            raise ValueError("active_tolerance is not a preregistered rung")
        if self.stage_phase in {"tolerance", "n600", "admission"} and (
            not self.seg_constraints_solved_for_active_tolerance or not self.pose_solved_within_seg_feasible_polytope
        ):
            raise ValueError("scored stages require cells-then-pose-tube solve ordering")
        if not self.argv:
            raise ValueError("argv must be preserved and nonempty")
        if _sha256(rfc8785_canonicalize(self.config)) != self.config_sha256:
            raise ValueError("config SHA-256 mismatch")
        if self.config.get("dsl_compile_hash") != self.dsl_compile_hash:
            raise ValueError("checkpoint config does not bind its DSL compile hash")
        if _sha256("\0".join(self.argv).encode("utf-8")) != self.argv_sha256:
            raise ValueError("argv SHA-256 mismatch")
        try:
            z_member = base64.b64decode(self.z_member_b64, validate=True)
            best_archive = base64.b64decode(self.best_archive_b64, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("checkpoint byte state is not canonical base64") from exc
        if base64.b64encode(z_member).decode("ascii") != self.z_member_b64:
            raise ValueError("z_member_b64 is not canonical")
        if base64.b64encode(best_archive).decode("ascii") != self.best_archive_b64:
            raise ValueError("best_archive_b64 is not canonical")
        if _sha256(z_member) != self.z_member_sha256:
            raise ValueError("z member SHA-256 mismatch")
        parse_sections(z_member)
        if len(best_archive) != self.best_archive_bytes:
            raise ValueError("best archive byte count mismatch")
        if _sha256(best_archive) != self.best_archive_sha256:
            raise ValueError("best archive SHA-256 mismatch")
        parsed = parse_direct_description_archive(best_archive)
        if parsed.member != z_member:
            raise ValueError("checkpoint z and best archive describe different candidates")
        for field in (
            "entropy_state",
            "archive_compiler_state",
            "optimizer_state",
            "ema_shadow",
            "rng_state",
            "allocation_pool_states",
            "trigger_certificate_state",
            "config",
        ):
            value = getattr(self, field)
            if not isinstance(value, dict) or not value:
                raise ValueError(f"{field} must be a nonempty preserved mapping")
            rfc8785_canonicalize(value)
        rfc8785_canonicalize(list(self.hard_oracle_receipt_chain))
        rfc8785_canonicalize([row.model_dump(mode="json") for row in self.step_metric_telemetry])
        return self

    def to_bytes(self) -> bytes:
        body = self.model_dump(mode="json", by_alias=True)
        body_bytes = rfc8785_canonicalize(body)
        return rfc8785_canonicalize({"body": body, "body_sha256": _sha256(body_bytes)})

    @classmethod
    def from_bytes(cls, payload: bytes) -> DirectDescriptionStageCheckpointV1:
        document = _duplicate_refusing_json(payload)
        if not isinstance(document, dict) or set(document) != {"body", "body_sha256"}:
            raise DirectDescriptionError("checkpoint envelope fields mismatch")
        if rfc8785_canonicalize(document) != payload:
            raise DirectDescriptionError("checkpoint bytes are not canonical JCS")
        body = document["body"]
        if not isinstance(body, dict):
            raise DirectDescriptionError("checkpoint body must be an object")
        expected = _require_sha256(document["body_sha256"], "body_sha256")
        if _sha256(rfc8785_canonicalize(body)) != expected:
            raise DirectDescriptionError("checkpoint body hash mismatch")
        body = dict(body)
        for field in ("hard_oracle_receipt_chain", "step_metric_telemetry", "argv"):
            if isinstance(body.get(field), list):
                body[field] = tuple(body[field])
        return cls.model_validate(body)

    def filename(self) -> str:
        return (
            f"{self.run_id}__ddm_stage{self.stage_index:03d}_{self.stage_name}"
            f"_ep{self.epoch:06d}_step{self.global_step:012d}.json"
        )

    def write_new(self, directory: Path) -> Path:
        """Atomically publish one distinct, immutable stage checkpoint."""

        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        return _publish_new_bytes(root / self.filename(), self.to_bytes())


def load_stage_checkpoint(
    path: Path,
    *,
    expected_config_sha256: str,
    expected_dsl_compile_hash: str,
    expected_argv: Sequence[str],
) -> DirectDescriptionStageCheckpointV1:
    """Safely reopen checkpoint bytes and audit exact program-identity bindings."""

    expected_config_sha256 = _require_sha256(expected_config_sha256, "expected_config_sha256")
    expected_dsl_compile_hash = _require_sha256(
        expected_dsl_compile_hash, "expected_dsl_compile_hash"
    )
    payload = _read_regular_file_once(Path(path))
    checkpoint = DirectDescriptionStageCheckpointV1.from_bytes(payload)
    argv = tuple(expected_argv)
    if not argv:
        raise DirectDescriptionError("expected resume argv must be nonempty")
    if checkpoint.config_sha256 != expected_config_sha256:
        raise DirectDescriptionError("resume checkpoint config differs from the governed run")
    if checkpoint.dsl_compile_hash != expected_dsl_compile_hash:
        raise DirectDescriptionError("resume checkpoint DSL compile differs from the governed run")
    if checkpoint.argv != argv:
        raise DirectDescriptionError("resume checkpoint argv differs from the governed run")
    return checkpoint


def _publish_new_bytes(path: Path, payload: bytes) -> Path:
    """fsync temporary bytes and atomically hard-link them without overwrite."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.lstat()
    except FileNotFoundError:
        pass
    else:
        raise DirectDescriptionError(f"immutable output already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise DirectDescriptionError(f"immutable output raced into existence: {path}") from exc
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def storage_preflight(
    expected_bulk_bytes: int,
    *,
    reserve_bytes: int = 10 * 1024**3,
    tiers: Sequence[Path] = (
        Path("/Volumes/VertigoDataTier/pact"),
        Path("/Volumes/APDataStore/pact"),
    ),
) -> dict[str, Any]:
    """Select the first SSD tier with measured capacity or return authoritative REFUSE."""

    _require_exact_nonnegative_int(expected_bulk_bytes, "expected_bulk_bytes")
    _require_exact_nonnegative_int(reserve_bytes, "reserve_bytes")
    rows: list[dict[str, Any]] = []
    selected: Path | None = None
    for tier in tiers:
        root = Path(tier)
        if not root.is_dir():
            rows.append({"tier": str(root), "exists": False, "admissible": False})
            continue
        usage = shutil.disk_usage(root)
        admissible = usage.free >= expected_bulk_bytes + reserve_bytes
        rows.append(
            {
                "tier": str(root),
                "exists": True,
                "free_bytes": usage.free,
                "required_bytes_including_reserve": expected_bulk_bytes + reserve_bytes,
                "admissible": admissible,
            }
        )
        if selected is None and admissible:
            selected = root
    return {
        "schema": "direct_description_storage_preflight.v1",
        "expected_bulk_bytes": expected_bulk_bytes,
        "reserve_bytes": reserve_bytes,
        "waterfall": rows,
        "selected_tier": None if selected is None else str(selected),
        "evidence_root": None if selected is None else str(selected / "evidence/ddm_builder_20260721"),
        "cleanup_policy": "certify_or_block; success_only_scratch_delete; bulk_cold_store_first",
        "outcome": "ADMIT" if selected is not None else "REFUSE",
    }


def _fraction_from_fields(row: Mapping[str, Any], prefix: str) -> Fraction:
    numerator = _require_exact_nonnegative_int(row.get(f"{prefix}_numerator"), f"{prefix}_numerator")
    denominator = _require_exact_nonnegative_int(row.get(f"{prefix}_denominator"), f"{prefix}_denominator")
    if denominator == 0:
        raise DirectDescriptionError(f"{prefix}_denominator must be positive")
    return Fraction(numerator, denominator)


def verify_completion_certificate(
    certificate: Mapping[str, Any],
    *,
    expected_preregistration_sha256: str,
    expected_grammar_manifest_sha256: str,
    expected_independent_audit_sha256: str,
) -> dict[str, Any]:
    """Independently verify declared-grammar stationarity or finite exhaustion.

    The expected preregistration hash is external input.  Consequently a
    post-run certificate cannot make a new grammar/pool/stopping rule appear
    preregistered merely by rehashing its own fields.
    """

    expected_preregistration_sha256 = _require_sha256(
        expected_preregistration_sha256, "expected_preregistration_sha256"
    )
    expected_grammar_manifest_sha256 = _require_sha256(
        expected_grammar_manifest_sha256, "expected_grammar_manifest_sha256"
    )
    expected_independent_audit_sha256 = _require_sha256(
        expected_independent_audit_sha256, "expected_independent_audit_sha256"
    )
    if not isinstance(certificate, Mapping):
        raise DirectDescriptionError("completion certificate must be an object")
    if certificate.get("schema") != "DirectGrammarCompletionCertificateV1":
        raise DirectDescriptionError("completion certificate schema mismatch")
    preregistration = certificate.get("preregistration")
    if not isinstance(preregistration, Mapping):
        raise DirectDescriptionError("completion certificate lacks preregistration")
    preregistration_sha256 = _sha256(rfc8785_canonicalize(preregistration))
    if preregistration_sha256 != expected_preregistration_sha256:
        raise DirectDescriptionError("completion preregistration differs from external hash")
    if certificate.get("preregistration_sha256") != preregistration_sha256:
        raise DirectDescriptionError("completion certificate preregistration hash mismatch")
    if preregistration.get("grammar_manifest_sha256") != expected_grammar_manifest_sha256:
        raise DirectDescriptionError("completion certificate binds the wrong grammar")
    mode = preregistration.get("mode")
    if mode not in {"finite_enumeration", "stationarity_exhaustion"}:
        raise DirectDescriptionError("completion mode is unknown")
    stopping_rule = preregistration.get("stopping_rule")
    if mode == "finite_enumeration" and stopping_rule != "all_candidates_enumerated":
        raise DirectDescriptionError("finite enumeration has the wrong stopping rule")
    if mode == "stationarity_exhaustion" and stopping_rule != "all_pool_kkt_or_gate_excluded":
        raise DirectDescriptionError("stationarity certificate has the wrong stopping rule")
    if preregistration.get("pairs") != 600 or isinstance(preregistration.get("pairs"), bool):
        raise DirectDescriptionError("completion certificate must bind n600")
    for field in ("max_steps_per_pool", "wallclock_budget_seconds"):
        if _require_exact_nonnegative_int(preregistration.get(field), field) == 0:
            raise DirectDescriptionError(f"completion preregistration {field} must be positive")
    _require_exact_nonnegative_int(preregistration.get("max_restarts"), "max_restarts")
    if preregistration.get("hard_oracle_admission_rule") != (
        "same_pool_realized_score_gain_gt_25_over_37545489_per_byte"
    ):
        raise DirectDescriptionError("completion preregistration lacks the hard-oracle admission rule")
    pools = preregistration.get("pools")
    interpretations = preregistration.get("candidate_interpretations")
    if (
        not isinstance(pools, list)
        or not pools
        or any(not isinstance(item, str) or not item for item in pools)
        or len(set(pools)) != len(pools)
    ):
        raise DirectDescriptionError("completion pools must be unique nonempty strings")
    if (
        not isinstance(interpretations, list)
        or not interpretations
        or any(not isinstance(item, str) or not item for item in interpretations)
        or len(set(interpretations)) != len(interpretations)
    ):
        raise DirectDescriptionError("candidate interpretations must be unique nonempty strings")
    threshold = _fraction_from_fields(preregistration, "kkt_threshold")
    if certificate.get("optimizer_health") != "HEALTHY":
        raise DirectDescriptionError("optimizer is not healthy")
    if certificate.get("restart_exhausted") is not False:
        raise DirectDescriptionError("restart exhaustion is not completion")
    if certificate.get("budget_exhausted") is not False:
        raise DirectDescriptionError("budget exhaustion is not completion")
    if certificate.get("verdict_scope") != "FORMULATION_DECLARED_ANALYTIC_OPS_GRAMMAR":
        raise DirectDescriptionError("completion scope overclaims beyond the declared formulation")
    rows = certificate.get("rows")
    if not isinstance(rows, list):
        raise DirectDescriptionError("completion rows must be a list")
    if certificate.get("rows_sha256") != _sha256(rfc8785_canonicalize(rows)):
        raise DirectDescriptionError("completion row evidence hash mismatch")
    expected_keys = {(pool, interpretation) for pool in pools for interpretation in interpretations}
    observed: set[tuple[str, str]] = set()
    maximum_residual = Fraction(0)
    searched = 0
    excluded = 0
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise DirectDescriptionError(f"completion row {index} is not an object")
        key = (row.get("pool"), row.get("interpretation"))
        if key not in expected_keys or key in observed:
            raise DirectDescriptionError("completion rows are duplicated or out of preregistration")
        observed.add(key)  # type: ignore[arg-type]
        terminal_sha = _require_sha256(
            row.get("terminal_evidence_sha256"), "terminal_evidence_sha256"
        )
        terminal = _load_canonical_json_file(
            Path(str(row.get("terminal_evidence_path"))),
            terminal_sha,
            "direct_description_terminal_pool_evidence.v1",
        )
        terminal_bindings = {
            "grammar_manifest_sha256": expected_grammar_manifest_sha256,
            "preregistration_sha256": expected_preregistration_sha256,
            "pool": key[0],
            "interpretation": key[1],
            "pairs": 600,
        }
        if any(terminal.get(name) != value for name, value in terminal_bindings.items()):
            raise DirectDescriptionError("terminal evidence does not bind its pool and grammar")
        status = row.get("status")
        if status == "gate_excluded":
            gate_sha = _require_sha256(row.get("gate_receipt_sha256"), "gate_receipt_sha256")
            gate = _load_canonical_json_file(
                Path(str(row.get("gate_receipt_path"))),
                gate_sha,
                "direct_description_gate_exclusion.v1",
            )
            gate_bindings = {
                **terminal_bindings,
                "outcome": "EXCLUDED_BY_PREREGISTERED_GATE",
                "terminal_evidence_sha256": terminal_sha,
            }
            if any(gate.get(name) != value for name, value in gate_bindings.items()):
                raise DirectDescriptionError("gate receipt does not bind its exclusion")
            if row.get("gradient_terms") not in (None, []):
                raise DirectDescriptionError("gate-excluded row must not carry searched gradients")
            if terminal.get("status") != "gate_excluded":
                raise DirectDescriptionError("terminal evidence status disagrees with gate exclusion")
            excluded += 1
            continue
        if status != "searched":
            raise DirectDescriptionError("completion row status must be searched or gate_excluded")
        terms = row.get("gradient_terms")
        if not isinstance(terms, list) or not terms:
            raise DirectDescriptionError("searched completion row lacks raw gradient terms")
        coordinate_sums: dict[str, Fraction] = {}
        for term in terms:
            if not isinstance(term, Mapping):
                raise DirectDescriptionError("gradient term must be an object")
            coordinate = term.get("coordinate")
            if not isinstance(coordinate, str) or not coordinate:
                raise DirectDescriptionError("gradient term requires a named governed coordinate")
            numerator = term.get("numerator")
            denominator = term.get("denominator")
            if isinstance(numerator, bool) or not isinstance(numerator, int):
                raise DirectDescriptionError("gradient numerator must be an exact signed integer")
            denominator = _require_exact_nonnegative_int(denominator, "gradient denominator")
            if denominator == 0:
                raise DirectDescriptionError("gradient denominator must be positive")
            coordinate_sums[coordinate] = coordinate_sums.get(coordinate, Fraction(0)) + Fraction(
                numerator, denominator
            )
        residual = max(abs(value) for value in coordinate_sums.values())
        claimed = _fraction_from_fields(row, "claimed_kkt_residual")
        if claimed != residual:
            raise DirectDescriptionError("claimed KKT residual is not independently reproducible")
        if residual > threshold:
            raise DirectDescriptionError("searched pool exceeds preregistered KKT threshold")
        if mode == "finite_enumeration":
            candidate_count = _require_exact_nonnegative_int(row.get("candidate_count"), "candidate_count")
            enumerated_count = _require_exact_nonnegative_int(row.get("enumerated_count"), "enumerated_count")
            if candidate_count == 0 or candidate_count != enumerated_count:
                raise DirectDescriptionError("finite candidate pool was not exhausted")
        terminal_terms = terminal.get("gradient_terms")
        if terminal.get("status") != "searched" or terminal_terms != terms:
            raise DirectDescriptionError("terminal evidence does not preserve raw coordinate gradients")
        maximum_residual = max(maximum_residual, residual)
        searched += 1
    if observed != expected_keys:
        raise DirectDescriptionError("completion certificate leaves pools/interpretations uncovered")
    if searched == 0:
        raise DirectDescriptionError("empty/all-gate-excluded search cannot certify completion")
    audit_sha = _require_sha256(
        certificate.get("independent_audit_receipt_sha256"),
        "independent_audit_receipt_sha256",
    )
    if audit_sha != expected_independent_audit_sha256:
        raise DirectDescriptionError("completion audit differs from the external expected SHA-256")
    audit = _load_canonical_json_file(
        Path(str(certificate.get("independent_audit_receipt_path"))),
        audit_sha,
        "direct_description_completion_independent_audit.v1",
    )
    audit_bindings = {
        "preregistration_sha256": expected_preregistration_sha256,
        "grammar_manifest_sha256": expected_grammar_manifest_sha256,
        "rows_sha256": certificate.get("rows_sha256"),
        "pairs": 600,
        "source_bound_replay": True,
        "auditor_independent": True,
        "outcome": "VERIFIED_DECLARED_FORMULATION_COMPLETION",
    }
    if any(audit.get(name) != value for name, value in audit_bindings.items()):
        raise DirectDescriptionError("independent completion audit does not bind the certificate")
    return {
        "schema": "direct_grammar_completion_verification.v1",
        "verified": True,
        "mode": mode,
        "pair_count": 600,
        "row_count": len(rows),
        "searched_count": searched,
        "gate_excluded_count": excluded,
        "maximum_kkt_residual_exact": (f"{maximum_residual.numerator}/{maximum_residual.denominator}"),
        "verdict_scope": "FORMULATION_DECLARED_ANALYTIC_OPS_GRAMMAR",
    }


def optimizer_admission_status(
    certificate: Mapping[str, Any] | None,
    *,
    expected_preregistration_sha256: str,
    expected_grammar_manifest_sha256: str,
    expected_independent_audit_sha256: str,
) -> str:
    """Return the only status allowed when completion cannot be independently certified."""

    try:
        if certificate is None:
            raise DirectDescriptionError("missing completion certificate")
        verify_completion_certificate(
            certificate,
            expected_preregistration_sha256=expected_preregistration_sha256,
            expected_grammar_manifest_sha256=expected_grammar_manifest_sha256,
            expected_independent_audit_sha256=expected_independent_audit_sha256,
        )
    except (DirectDescriptionError, ValueError, TypeError):
        return "OPTIMIZER_NO_ADMISSION"
    return "COMPLETION_CERTIFIED"


_FAILURE_REQUIRED_FIELDS: Final = frozenset(
    {
        "schema",
        "verdict_token",
        "verdict_scope",
        "primary_spec_sha256",
        "git_sha",
        "seed",
        "run_id",
        "hardware_axis",
        "full_precision_target_receipt_path",
        "full_precision_target_receipt_sha256",
        "solved_d_seg",
        "solved_d_pose",
        "pointer_cap_bytes",
        "pointer_cap_formula",
        "strict_0_15_cap_bytes",
        "strict_cap_role",
        "grammar_manifest_sha256",
        "grammar_manifest_path",
        "live_owner_receipt_manifest_sha256",
        "live_owner_receipt_manifest_path",
        "archive_path",
        "archive_sha256",
        "archive_bytes",
        "parseback_sha256",
        "canonical_reencode_sha256",
        "raw_decode_sha256_x2",
        "raw_decode_paths_x2",
        "evaluator_path",
        "evaluator_sha256",
        "scorer_runtime_path",
        "scorer_runtime_sha256",
        "measurement_receipt_path",
        "measurement_receipt_sha256",
        "measured_d_seg",
        "measured_d_pose",
        "measured_score",
        "pairs",
        "optimizer_health",
        "completion_certificate",
        "completion_preregistration_sha256",
        "completion_independent_audit_sha256",
        "failed_receiver_predicates",
        "all_readiness_predicates",
        "readiness_evidence_manifest_path",
        "readiness_evidence_manifest_sha256",
        "external_attestation_path",
        "external_attestation_sha256",
    }
)


def _validate_failure_body(receipt: Mapping[str, Any], *, expected_completion_preregistration_sha256: str) -> None:
    missing = sorted(_FAILURE_REQUIRED_FIELDS - set(receipt))
    if missing:
        raise DirectDescriptionError(f"failure receipt lacks required fields: {missing}")
    if "receipt_file_sha256" in receipt:
        raise DirectDescriptionError("receipt_file_sha256 belongs only in the immutable sidecar")
    if receipt.get("schema") != "DirectGrammarReceiverReachabilityFailureReceiptV1":
        raise DirectDescriptionError("failure receipt schema mismatch")
    if receipt.get("verdict_token") != "DIRECT_GRAMMAR_RECEIVER_REACHABILITY_FAILURE":
        raise DirectDescriptionError("failure receipt token mismatch")
    if receipt.get("verdict_scope") != "FORMULATION_DECLARED_ANALYTIC_OPS_GRAMMAR":
        raise DirectDescriptionError("failure receipt scope overclaims")
    if receipt.get("primary_spec_sha256") != PRIMARY_SPEC_SHA256:
        raise DirectDescriptionError("failure receipt binds the wrong PRIMARY spec")
    if receipt.get("seed") != SEED or isinstance(receipt.get("seed"), bool):
        raise DirectDescriptionError("failure receipt seed mismatch")
    if not isinstance(receipt.get("git_sha"), str) or re.fullmatch(r"[0-9a-f]{40}", receipt["git_sha"]) is None:
        raise DirectDescriptionError("failure receipt requires a full lowercase git SHA-1")
    if receipt.get("hardware_axis") not in {"[contest-CPU]", "[contest-CUDA]"}:
        raise DirectDescriptionError("failure receipt requires an exact contest authority axis")
    if receipt.get("pairs") != 600 or isinstance(receipt.get("pairs"), bool):
        raise DirectDescriptionError("failure receipt is not n600")
    if receipt.get("optimizer_health") != "HEALTHY":
        raise DirectDescriptionError("unhealthy optimizer cannot emit reachability failure")
    if receipt.get("pointer_cap_formula") != "ceil_minus_one":
        raise DirectDescriptionError("pointer cap formula mismatch")
    if receipt.get("strict_cap_role") != "stretch_only":
        raise DirectDescriptionError("strict cap cannot replace the pointer fallback ceiling")
    for field in (
        "primary_spec_sha256",
        "full_precision_target_receipt_sha256",
        "grammar_manifest_sha256",
        "live_owner_receipt_manifest_sha256",
        "archive_sha256",
        "parseback_sha256",
        "canonical_reencode_sha256",
        "evaluator_sha256",
        "scorer_runtime_sha256",
        "completion_preregistration_sha256",
        "completion_independent_audit_sha256",
        "measurement_receipt_sha256",
        "readiness_evidence_manifest_sha256",
        "external_attestation_sha256",
    ):
        _require_sha256(receipt.get(field), field)
    raw_hashes = receipt.get("raw_decode_sha256_x2")
    if not isinstance(raw_hashes, list) or len(raw_hashes) != 2 or raw_hashes[0] != raw_hashes[1]:
        raise DirectDescriptionError("double decode hashes must be two identical SHA-256 values")
    _require_sha256(raw_hashes[0], "raw_decode_sha256_x2")
    raw_paths = receipt.get("raw_decode_paths_x2")
    if not isinstance(raw_paths, list) or len(raw_paths) != 2:
        raise DirectDescriptionError("double decode custody requires two exact paths")
    for raw_payload in _read_distinct_double_decode(raw_paths):
        if _sha256(raw_payload) != raw_hashes[0]:
            raise DirectDescriptionError("failure receipt raw decode bytes do not match their SHA-256")
    archive_path = Path(str(receipt.get("archive_path")))
    parsed = parse_direct_description_archive(archive_path)
    if receipt.get("archive_bytes") != len(parsed.archive):
        raise DirectDescriptionError("failure receipt archive_bytes is not len(A(z))")
    if receipt.get("archive_sha256") != _sha256(parsed.archive):
        raise DirectDescriptionError("failure receipt archive SHA-256 mismatch")
    if receipt.get("parseback_sha256") != _sha256(parsed.archive):
        raise DirectDescriptionError("parseback SHA-256 must bind the exact reopened A(z) bytes")
    if receipt.get("canonical_reencode_sha256") != _sha256(parsed.archive):
        raise DirectDescriptionError("canonical re-encode SHA-256 must bind exact A(z) bytes")
    target_path = Path(str(receipt.get("full_precision_target_receipt_path")))
    derived_caps = derive_ceil_minus_one_caps(
        target_path,
        receipt["full_precision_target_receipt_sha256"],
    )
    if (
        receipt.get("solved_d_seg") != derived_caps["solved_d_seg"]
        or receipt.get("solved_d_pose") != derived_caps["solved_d_pose"]
    ):
        raise DirectDescriptionError("failure receipt solved target differs from its SHA-bound receipt")
    pointer_cap = _require_exact_nonnegative_int(receipt.get("pointer_cap_bytes"), "pointer_cap_bytes")
    strict_cap = _require_exact_nonnegative_int(receipt.get("strict_0_15_cap_bytes"), "strict_0_15_cap_bytes")
    if strict_cap >= pointer_cap:
        raise DirectDescriptionError("strict stretch cap must be below the pointer cap")
    if pointer_cap != derived_caps["pointer_cap_bytes"] or strict_cap != derived_caps["strict_0_15_cap_bytes"]:
        raise DirectDescriptionError("failure receipt integer caps were not rederived from full precision")
    if len(parsed.archive) > pointer_cap:
        raise DirectDescriptionError("byte excess cannot trigger fallback #366")
    failed_predicates = receipt.get("failed_receiver_predicates")
    if not isinstance(failed_predicates, list) or not failed_predicates:
        raise DirectDescriptionError("reachability failure requires measured failed predicates")
    readiness = receipt.get("all_readiness_predicates")
    if not isinstance(readiness, Mapping) or not readiness or any(value is not True for value in readiness.values()):
        raise DirectDescriptionError("every fallback readiness predicate must be green")
    required_readiness = {
        "grammar",
        "archive_parse_reencode",
        "charged_free",
        "quarantine",
        "deterministic_decode",
        "storage",
        "resume",
        "evaluator",
        "external_attestation",
        "live_owners",
        "pose_owner",
        "completion",
        "byte_cap",
        "receiver_boundary",
    }
    if set(readiness) != required_readiness:
        raise DirectDescriptionError("fallback readiness predicate inventory is incomplete or invented")
    expected_completion_preregistration_sha256 = _require_sha256(
        expected_completion_preregistration_sha256,
        "expected_completion_preregistration_sha256",
    )
    if receipt.get("completion_preregistration_sha256") != (expected_completion_preregistration_sha256):
        raise DirectDescriptionError("failure receipt completion preregistration mismatch")
    completion_verification = verify_completion_certificate(
        receipt["completion_certificate"],
        expected_preregistration_sha256=expected_completion_preregistration_sha256,
        expected_grammar_manifest_sha256=receipt["grammar_manifest_sha256"],
        expected_independent_audit_sha256=receipt["completion_independent_audit_sha256"],
    )
    if completion_verification["searched_count"] == 0:
        raise DirectDescriptionError("empty search cannot trigger fallback")
    for field in ("measured_d_seg", "measured_d_pose", "measured_score"):
        value = receipt.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise DirectDescriptionError(f"{field} must be a finite JSON number")
    recomputed_score = (
        100.0 * float(receipt["measured_d_seg"])
        + math.sqrt(10.0 * float(receipt["measured_d_pose"]))
        + 25.0 * len(parsed.archive) / SOURCE_BYTES
    )
    if not math.isclose(recomputed_score, float(receipt["measured_score"]), rel_tol=0, abs_tol=1e-12):
        raise DirectDescriptionError("failure receipt measured_score does not match exact objective")
    solved_d_seg = Decimal(receipt["solved_d_seg"])
    solved_d_pose = Decimal(receipt["solved_d_pose"])
    if (
        Decimal(str(receipt["measured_d_seg"])) <= solved_d_seg
        and Decimal(str(receipt["measured_d_pose"])) <= solved_d_pose
    ):
        raise DirectDescriptionError("qualifying solved tuple is PRIMARY success, not reachability failure")
    evaluator_path = Path(str(receipt.get("evaluator_path")))
    try:
        evaluator_bytes = evaluator_path.read_bytes()
    except OSError as exc:
        raise DirectDescriptionError("failure receipt evaluator path is unreadable") from exc
    if _sha256(evaluator_bytes) != receipt.get("evaluator_sha256"):
        raise DirectDescriptionError("failure receipt evaluator file hash mismatch")
    if _sha256(_read_regular_file_once(Path(str(receipt["scorer_runtime_path"])))) != receipt.get(
        "scorer_runtime_sha256"
    ):
        raise DirectDescriptionError("failure receipt scorer-runtime file hash mismatch")
    grammar_manifest = _load_canonical_json_file(
        Path(str(receipt["grammar_manifest_path"])),
        receipt["grammar_manifest_sha256"],
        "direct_description_grammar_manifest.v1",
    )
    if grammar_manifest.get("run_id") != receipt.get("run_id"):
        raise DirectDescriptionError("grammar manifest does not bind the failure run")
    owner_manifest = _load_canonical_json_file(
        Path(str(receipt["live_owner_receipt_manifest_path"])),
        receipt["live_owner_receipt_manifest_sha256"],
        "direct_description_live_owner_manifest.v1",
    )
    expected_owners = set(_STREAM_TO_SECTION) | {
        "pixel",
        "class",
        "boundary",
        "frame",
        "pair",
        "epoch",
        "chroma",
        "scale",
        "frequency",
    }
    owner_rows = owner_manifest.get("owner_rows")
    if not isinstance(owner_rows, list) or {row.get("owner") for row in owner_rows if isinstance(row, Mapping)} != expected_owners:
        raise DirectDescriptionError("live-owner manifest has incomplete v8/v9/DDM ownership")
    for row in owner_rows:
        if not isinstance(row, Mapping) or any(
            row.get(field) is not True
            for field in ("config_consumed", "output_mutated", "archive_byte_effect_measured")
        ):
            raise DirectDescriptionError("live-owner row lacks consumption/mutation/rate proof")
        evidence_sha = _require_sha256(row.get("receipt_sha256"), "owner receipt SHA-256")
        _load_canonical_json_file(
            Path(str(row.get("receipt_path"))),
            evidence_sha,
            "direct_description_live_owner_receipt.v1",
        )
    measurement = _load_canonical_json_file(
        Path(str(receipt["measurement_receipt_path"])),
        receipt["measurement_receipt_sha256"],
        "direct_description_n600_measurement.v1",
    )
    measurement_bindings = {
        "archive_sha256": receipt["archive_sha256"],
        "archive_bytes": receipt["archive_bytes"],
        "raw_decode_sha256_x2": receipt["raw_decode_sha256_x2"],
        "evaluator_sha256": receipt["evaluator_sha256"],
        "scorer_runtime_sha256": receipt["scorer_runtime_sha256"],
        "d_seg": receipt["measured_d_seg"],
        "d_pose": receipt["measured_d_pose"],
        "score": receipt["measured_score"],
        "pairs": 600,
        "hardware_axis": receipt["hardware_axis"],
        "through_R": True,
        "exact_evaluator_called": True,
    }
    if any(measurement.get(name) != value for name, value in measurement_bindings.items()):
        raise DirectDescriptionError("n600 measurement receipt does not bind the failure tuple")
    expected_failed: set[str] = set()
    if Decimal(str(receipt["measured_d_seg"])) > solved_d_seg:
        expected_failed.add("solved_seg_cell_tolerance")
    if Decimal(str(receipt["measured_d_pose"])) > solved_d_pose:
        expected_failed.add("solved_pose_tube_tolerance")
    if set(failed_predicates) != expected_failed:
        raise DirectDescriptionError("failed receiver predicates do not match the measured solved-target miss")
    readiness_manifest = _load_canonical_json_file(
        Path(str(receipt["readiness_evidence_manifest_path"])),
        receipt["readiness_evidence_manifest_sha256"],
        "direct_description_readiness_evidence_manifest.v1",
    )
    evidence_rows = readiness_manifest.get("predicates")
    if not isinstance(evidence_rows, list) or {row.get("predicate") for row in evidence_rows if isinstance(row, Mapping)} != required_readiness:
        raise DirectDescriptionError("readiness evidence manifest inventory mismatch")
    for row in evidence_rows:
        if not isinstance(row, Mapping) or row.get("outcome") != "GREEN":
            raise DirectDescriptionError("readiness evidence row is not green")
        evidence_sha = _require_sha256(row.get("evidence_sha256"), "readiness evidence SHA-256")
        evidence = _load_canonical_json_file(
            Path(str(row.get("evidence_path"))),
            evidence_sha,
            "direct_description_readiness_evidence.v1",
        )
        if (
            evidence.get("predicate") != row.get("predicate")
            or evidence.get("outcome") != "GREEN"
            or evidence.get("archive_sha256") != receipt["archive_sha256"]
            or evidence.get("run_id") != receipt["run_id"]
        ):
            raise DirectDescriptionError("readiness evidence does not bind this run/archive")
    _verify_external_failure_attestation(receipt)


def seal_failure_receipt(
    receipt_without_payload_hash: Mapping[str, Any],
    *,
    expected_completion_preregistration_sha256: str,
) -> dict[str, Any]:
    """Validate and add the non-self-referential RFC 8785 payload hash."""

    if "receipt_payload_sha256" in receipt_without_payload_hash:
        raise DirectDescriptionError("payload hash field must be absent while sealing")
    body = dict(receipt_without_payload_hash)
    _validate_failure_body(
        body,
        expected_completion_preregistration_sha256=expected_completion_preregistration_sha256,
    )
    payload_sha256 = _sha256(rfc8785_canonicalize(body))
    return {**body, "receipt_payload_sha256": payload_sha256}


def write_failure_receipt(
    path: Path,
    sealed_receipt: Mapping[str, Any],
    *,
    expected_completion_preregistration_sha256: str,
) -> dict[str, Any]:
    """Write immutable canonical receipt + exact-file-hash sidecar."""

    receipt = dict(sealed_receipt)
    payload_hash = _require_sha256(receipt.get("receipt_payload_sha256"), "receipt_payload_sha256")
    body = dict(receipt)
    del body["receipt_payload_sha256"]
    _validate_failure_body(
        body,
        expected_completion_preregistration_sha256=expected_completion_preregistration_sha256,
    )
    if _sha256(rfc8785_canonicalize(body)) != payload_hash:
        raise DirectDescriptionError("sealed failure receipt payload hash mismatch")
    exact_file_bytes = rfc8785_canonicalize(receipt) + b"\n"
    path = _publish_new_bytes(Path(path), exact_file_bytes)
    file_sha256 = _sha256(exact_file_bytes)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_bytes = f"{file_sha256}  {path.name}\n".encode("ascii")
    _publish_new_bytes(sidecar, sidecar_bytes)
    return {
        "receipt_path": str(path),
        "receipt_payload_sha256": payload_hash,
        "receipt_file_sha256": file_sha256,
        "sidecar_path": str(sidecar),
    }


def _read_regular_file_with_identity(path: Path) -> tuple[bytes, tuple[int, int]]:
    """Safely read one regular file and return its opened device/inode identity."""

    try:
        before = path.lstat()
    except OSError as exc:
        raise DirectDescriptionError(f"receipt file cannot be inspected: {path}") from exc
    if not path.is_file() or path.is_symlink():
        raise DirectDescriptionError("receipt must be a non-symlink regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DirectDescriptionError("receipt safe-open failed") from exc
    try:
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise DirectDescriptionError("receipt changed during safe-open")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks), (after.st_dev, after.st_ino)
    finally:
        os.close(descriptor)


def _read_regular_file_once(path: Path) -> bytes:
    return _read_regular_file_with_identity(path)[0]


def _read_distinct_double_decode(paths: Sequence[Any]) -> tuple[bytes, bytes]:
    """Open two decode artifacts and reject path or inode aliasing."""

    if len(paths) != 2:
        raise DirectDescriptionError("double decode custody requires exactly two paths")
    path_objects = tuple(Path(str(value)) for value in paths)
    normalized = tuple(os.path.normcase(os.path.abspath(path)) for path in path_objects)
    if normalized[0] == normalized[1]:
        raise DirectDescriptionError("double decode custody requires distinct paths")
    opened = tuple(_read_regular_file_with_identity(path) for path in path_objects)
    if opened[0][1] == opened[1][1]:
        raise DirectDescriptionError("double decode custody requires distinct file identities")
    return opened[0][0], opened[1][0]


def _load_canonical_json_file(path: Path, expected_sha256: str, expected_schema: str) -> dict[str, Any]:
    """Safely load one exact SHA-bound JCS+LF evidence artifact."""

    expected_sha256 = _require_sha256(expected_sha256, "evidence file SHA-256")
    payload = _read_regular_file_once(Path(path))
    if _sha256(payload) != expected_sha256:
        raise DirectDescriptionError(f"evidence file SHA-256 mismatch: {path}")
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise DirectDescriptionError(f"evidence file must end in exactly one LF: {path}")
    document = _duplicate_refusing_json(payload[:-1])
    if not isinstance(document, dict):
        raise DirectDescriptionError(f"evidence file must be a JSON object: {path}")
    if rfc8785_canonicalize(document) + b"\n" != payload:
        raise DirectDescriptionError(f"evidence file is not canonical JCS: {path}")
    if document.get("schema") != expected_schema:
        raise DirectDescriptionError(f"evidence file schema mismatch: {path}")
    return document


def _load_attestor_trust_root() -> Mapping[str, Any]:
    """Load the repository's externally managed Ed25519 approver registry."""

    from tac.lane_c_compliance import load_trust_root

    return load_trust_root(_REPO_ROOT)


def _verify_external_failure_attestation(receipt: Mapping[str, Any]) -> None:
    """Require a two-party signature over every non-attestation failure field."""

    attestation = _load_canonical_json_file(
        Path(str(receipt.get("external_attestation_path"))),
        str(receipt.get("external_attestation_sha256")),
        "direct_description_external_attestation.v1",
    )
    expected_keys = {
        "schema",
        "scope",
        "subject_sha256",
        "approver",
        "signed_at_utc",
        "signature_hex",
    }
    if set(attestation) != expected_keys:
        raise DirectDescriptionError("external failure attestation key set mismatch")
    unsigned_subject = dict(receipt)
    del unsigned_subject["external_attestation_path"]
    del unsigned_subject["external_attestation_sha256"]
    subject_sha = _sha256(rfc8785_canonicalize(unsigned_subject))
    if (
        attestation.get("scope") != "DIRECT_GRAMMAR_RECEIVER_REACHABILITY_FAILURE"
        or attestation.get("subject_sha256") != subject_sha
    ):
        raise DirectDescriptionError("external attestation does not bind the failure body")
    approver = attestation.get("approver")
    if not isinstance(approver, str) or not approver:
        raise DirectDescriptionError("external attestation approver is missing")
    signature_hex = attestation.get("signature_hex")
    if (
        not isinstance(signature_hex, str)
        or len(signature_hex) != 128
        or set(signature_hex) - set("0123456789abcdef")
    ):
        raise DirectDescriptionError("external attestation signature is not lowercase Ed25519 hex")
    public_key = _load_attestor_trust_root().get(approver)
    if public_key is None:
        raise DirectDescriptionError("external attestation approver is not in the trust root")
    signed_payload = dict(attestation)
    del signed_payload["signature_hex"]
    try:
        public_key.verify(bytes.fromhex(signature_hex), rfc8785_canonicalize(signed_payload))
    except Exception as exc:
        raise DirectDescriptionError("external failure attestation signature is invalid") from exc


def verify_failure_receipt(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_payload_sha256: str,
    expected_completion_preregistration_sha256: str,
) -> dict[str, Any]:
    """Verify exact file bytes first, then JCS payload hash and full semantics."""

    expected_file_sha256 = _require_sha256(expected_file_sha256, "expected_file_sha256")
    expected_payload_sha256 = _require_sha256(expected_payload_sha256, "expected_payload_sha256")
    payload = _read_regular_file_once(Path(path))
    if _sha256(payload) != expected_file_sha256:
        raise DirectDescriptionError("failure receipt exact-file SHA-256 mismatch")
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise DirectDescriptionError("failure receipt must end in exactly one LF")
    document = _duplicate_refusing_json(payload[:-1])
    if not isinstance(document, dict):
        raise DirectDescriptionError("failure receipt must be an object")
    if rfc8785_canonicalize(document) + b"\n" != payload:
        raise DirectDescriptionError("failure receipt file is not RFC 8785 canonical")
    observed_payload_hash = _require_sha256(document.get("receipt_payload_sha256"), "receipt_payload_sha256")
    if observed_payload_hash != expected_payload_sha256:
        raise DirectDescriptionError("failure receipt GO payload hash mismatch")
    body = dict(document)
    del body["receipt_payload_sha256"]
    if _sha256(rfc8785_canonicalize(body)) != observed_payload_hash:
        raise DirectDescriptionError("failure receipt reconstructed payload hash mismatch")
    _validate_failure_body(
        body,
        expected_completion_preregistration_sha256=expected_completion_preregistration_sha256,
    )
    sidecar = Path(path).with_suffix(Path(path).suffix + ".sha256")
    expected_sidecar = f"{expected_file_sha256}  {Path(path).name}\n".encode("ascii")
    if _read_regular_file_once(sidecar) != expected_sidecar:
        raise DirectDescriptionError("failure receipt immutable sidecar mismatch")
    return {
        "schema": "direct_grammar_failure_receipt_verification.v1",
        "verified": True,
        "receipt_file_sha256": expected_file_sha256,
        "receipt_payload_sha256": expected_payload_sha256,
        "verdict_token": document["verdict_token"],
        "verdict_scope": document["verdict_scope"],
    }


def validate_receiver_rate_custody(path: Path) -> dict[str, Any]:
    """Validate a candidate's parser-consumed unique-home final-ZIP attribution.

    This is intentionally stronger than a payload/component sum.  Byte ranges
    must partition the exact canonical archive file, every nonempty owner must
    have one real consumer and output mutation, and n64/n600 receipts must bind
    the same archive.  The current z0 control cannot satisfy this candidate
    contract because Deflate9 does not expose per-section final-ZIP ranges.
    """

    receipt_path = Path(path)
    try:
        encoded = receipt_path.read_bytes()
    except OSError as exc:
        raise DirectDescriptionError(f"receiver-rate custody is unreadable: {receipt_path}") from exc
    receipt = _duplicate_refusing_json(encoded)
    if not isinstance(receipt, dict):
        raise DirectDescriptionError("receiver-rate custody must be an object")
    if receipt.get("schema") != "direct_description_receiver_rate_custody.v1":
        raise DirectDescriptionError("receiver-rate custody schema mismatch")
    if receipt.get("candidate_role") != "fresh_primary_candidate":
        raise DirectDescriptionError("controls/retired archives cannot mint candidate custody")
    # V1 attempted to assign byte intervals inside one globally DEFLATE-compressed
    # member to semantic classes.  Those intervals are not local final-ZIP homes:
    # changing one decoded stream can perturb the remainder of the Deflate bitstream.
    # Consequently no V1 receipt can establish unique-home receiver rate, no
    # matter how many hash-shaped metadata fields it carries.
    raise DirectDescriptionError(
        "UNSUPPORTED_NONLOCAL_DEFLATE_ATTRIBUTION: receiver-rate custody v1 is "
        "non-authorizing; use a versioned independently framed carrier with "
        "source-bound parser/mutation receipts"
    )


def build_launch_readiness(
    owner_bundle: Mapping[str, Any],
    *,
    storage_receipt: Mapping[str, Any],
    full_precision_caps: Mapping[str, Any] | None = None,
    live_owner_receipts_green: bool = False,
    n64_custody_green: bool = False,
    n600_archive_green: bool = False,
    receiver_rate_custody_green: bool = False,
    memory_preflight_outcome: str = "REFUSE",
    governor_outcome: str = "REFUSE",
    operator_go: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the sealed DRAFT ticket; current spec structurally cannot launch."""

    if not isinstance(owner_bundle, Mapping) or owner_bundle.get("schema") != (
        "direct_description_typed_owner_bundle.v1"
    ):
        raise DirectDescriptionError("launch ticket requires the typed owner bundle")
    recorded_compile_hash = _require_sha256(owner_bundle.get("dsl_compile_hash"), "dsl_compile_hash")
    unhashed = dict(owner_bundle)
    del unhashed["dsl_compile_hash"]
    if _sha256(rfc8785_canonicalize(_strip_volatile(unhashed))) != recorded_compile_hash:
        raise DirectDescriptionError("owner bundle DSL compile hash mismatch")
    typed_config = DirectDescriptionTypedConfigV1.model_validate_json(
        json.dumps(owner_bundle.get("typed_config"), separators=(",", ":"), ensure_ascii=False)
    )
    if typed_config.typed_config_hash() != owner_bundle.get("typed_config_hash"):
        raise DirectDescriptionError("owner bundle typed-config hash mismatch")
    if typed_config.owner.model_dump(mode="json", by_alias=True) != owner_bundle.get("owner"):
        raise DirectDescriptionError("owner bundle typed config and owner disagree")
    consumer_argv = owner_bundle.get("consumer_argv")
    expected_argv = list(typed_config.to_program().compile_consumer_argv())
    if not isinstance(consumer_argv, list) or consumer_argv != expected_argv:
        raise DirectDescriptionError("owner bundle consumer argv was not emitted by the DDM compiler")
    if owner_bundle.get("custody_argv") != expected_argv:
        raise DirectDescriptionError("owner bundle post-custody argv differs from the DDM compiler")
    program_manifest = typed_config.program_manifest()
    if owner_bundle.get("program_manifest") != program_manifest:
        raise DirectDescriptionError("owner bundle program manifest is stale or invented")
    if len(consumer_argv) < 4:
        raise DirectDescriptionError("owner bundle consumer argv is missing")
    parsed_argv = build_direct_description_arg_parser().parse_args(consumer_argv[3:])
    if parsed_argv.execution_allowed != "false":
        raise DirectDescriptionError("current PRIMARY consumer argv must compile execution_allowed=false")
    blockers: list[str] = []
    if owner_bundle.get("execution_allowed") is not False:
        raise DirectDescriptionError("current PRIMARY owner must seal execution_allowed=false")
    blockers.append("PRIMARY_SPEC_EXECUTION_ALLOWED_FALSE")
    if storage_receipt.get("outcome") != "ADMIT":
        blockers.append("STORAGE_PREFLIGHT_REFUSE")
    if full_precision_caps is None:
        blockers.append("FULL_PRECISION_SHA_BOUND_TARGET_RECEIPT_MISSING")
    if not live_owner_receipts_green:
        blockers.append("LIVE_V8_V9_OWNER_RECEIPTS_MISSING")
    if not n64_custody_green:
        blockers.append("N64_DETERMINISTIC_CUSTODY_SMOKE_MISSING")
    if not n600_archive_green:
        blockers.append("N600_SAME_ARTIFACT_ARCHIVE_CLOSURE_MISSING")
    if not receiver_rate_custody_green:
        blockers.append("PER_STRATUM_RECEIVER_RATE_CUSTODY_MISSING")
    blockers.append("FRESH_V3_FAMILY_POSE_IN_OBJECTIVE_RUNG_ZERO_MISSING")
    blockers.append("FOUR_RUNG_CELLS_THEN_POSE_MEASUREMENT_LADDER_MISSING")
    blockers.append("POSE_CONSUMING_INTEGER_UINT8_RECEIVER_NOT_IMPLEMENTED")
    blockers.append("DDM_OPTIMIZER_AND_STAGE_CONTINUATION_RUNNER_NOT_IMPLEMENTED")
    blockers.append("MONOLITHIC_DEFLATE_UNIQUE_HOME_RATE_ATTRIBUTION_UNSUPPORTED")
    blockers.append("CANONICAL_TYPED_COMPILER_INTEGRATION_MISSING")
    blockers.append("GOVERNED_LAUNCHER_AND_MEMORY_PREFLIGHT_ADAPTER_NOT_IMPLEMENTED")
    blockers.append("CANONICAL_RESUME_REGISTRY_AND_CHECKPOINT_CADENCE_NOT_IMPLEMENTED")
    blockers.append("OPERATIONAL_CLEANUP_COLD_STORE_HOOK_NOT_IMPLEMENTED")
    blockers.append("CONTEST_CPU_AND_CUDA_REPLAYS_MISSING")
    if memory_preflight_outcome != "ADMIT":
        blockers.append("WITNESS_MEMORY_PREFLIGHT_REFUSE")
    if governor_outcome != "ADMIT":
        blockers.append("SYSTEM_MEMORY_GOVERNOR_REFUSE")
    if operator_go is None:
        blockers.append("SEPARATE_SHA_BOUND_OPERATOR_GO_MISSING")
    else:
        blockers.append("PRIMARY_SPEC_MUST_BE_SUPERSEDED_BEFORE_OPERATOR_GO")
    return {
        "schema": "direct_description_launch_ticket.v1",
        "status": "DRAFT_DO_NOT_FIRE",
        "execution_allowed": False,
        "research_only": True,
        "dsl_compile_hash": recorded_compile_hash,
        "typed_config_hash": owner_bundle.get("typed_config_hash"),
        "consumer_argv": consumer_argv,
        "governed_launcher": "tools/launch_witness_run.py",
        "memory_preflight": "tools/witness_memory_preflight.py",
        "memory_preflight_outcome": memory_preflight_outcome,
        "governor_outcome": governor_outcome,
        "storage_preflight": dict(storage_receipt),
        "full_precision_caps": None if full_precision_caps is None else dict(full_precision_caps),
        "blockers": blockers,
        "launch_ready": False,
        "spawn_permitted": False,
    }


__all__ = [
    "ChargedFreePartitionRowV1",
    "CountedDescriptionStreamV1",
    "DescriptionStepMetricTelemetryV1",
    "DirectArchiveBuildResult",
    "DirectDescriptionError",
    "DirectDescriptionOpsGrammarMinimizerV1",
    "DirectDescriptionStageCheckpointV1",
    "DirectDescriptionTypedConfigV1",
    "DirectDescriptionWitnessProgramV1",
    "DirectDescriptionZV1",
    "MeasurementRungRowV1",
    "ToleranceAllocationNodeV1",
    "build_direct_description_arg_parser",
    "build_direct_description_owner",
    "build_launch_readiness",
    "compile_direct_description_archive",
    "derive_ceil_minus_one_caps",
    "load_stage_checkpoint",
    "numpy_reference_rank",
    "optimizer_admission_status",
    "parse_direct_description_archive",
    "prove_baseline_reexpression",
    "rfc8785_canonicalize",
    "seal_failure_receipt",
    "storage_preflight",
    "validate_receiver_rate_custody",
    "verify_allocation_tree",
    "verify_charged_free_partition",
    "verify_completion_certificate",
    "verify_failure_receipt",
    "verify_measurement_ladder",
    "write_failure_receipt",
]
