# SPDX-License-Identifier: MIT
"""Public-shaped G17 receiver operation for ``Y0 | exact-decoded Y1``.

This module adds one audited receiver operation, not another selected-solution
state or physical-byte ontology.  Counted operand custody is supplied by the
canonical :class:`G17CompilerPlacementManifestV1`; the operation consumes that
manifest's exact member bytes and produces one camera-resolution uint8 Y0
without mutating the caller's already-decoded Y1.

The learned quotient callable follows the load-bearing G49 receiver contract:
the packet does not select an import, the caller binds a real callable plus its
exact Python-source identity, inputs are immutable, execution is repeated, and
counted bytes must be receiver-live under mutation.  There is deliberately no
built-in transform, dense-plane fallback, scorer, teacher, or target path.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import marshal
import textwrap
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import FunctionType, MappingProxyType, ModuleType
from typing import Final, Literal, TypeAlias

import numpy as np

import tac.witness_dsl.taskspace_selected_preimage_program_v1 as _selected_preimage_program
from tac.witness_dsl.taskspace_selected_solution_compiler import (
    G17ArtifactClassV1,
    G17ChronologicalPosePreimageV1,
    G17CompilerPlacementManifestV1,
    G17CompilerPlacementRecordV1,
    G17LogicalOwnershipKindV1,
    G17PlacementClassV1,
    G17RuntimeDependencyFileV1,
    G17RuntimeFileScopeV1,
    G17ScientificRoleV1,
    G17SemanticStreamRoleV1,
)

CAMERA_HEIGHT: Final = 874
CAMERA_WIDTH: Final = 1164
CHANNELS: Final = 3
CAMERA_FRAME_SHAPE: Final = (CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS)
CAMERA_FRAME_BYTES: Final = CAMERA_HEIGHT * CAMERA_WIDTH * CHANNELS

OPERATION_ID: Final = "G17_CONDITIONAL_FULLFRAME_Y0_GIVEN_EXACT_DECODED_Y1_V1"
RECEIVER_CONSUMER_ID: Final = "tac.witness_dsl.taskspace_conditional_fullframe_receiver_operation_v1"
COUNTED_PAYLOAD_CLASS: Final = "CONDITIONAL_Y0_GIVEN_EXACT_DECODED_Y1_VIDEO_OPERAND"
PARAMETER_SPELLING_FORMAT: Final = "taskspace-selected-preimage-learned-quotient-v1"
GENERIC_SOURCE_PROVENANCE_BLOCKER: Final = "G17_CONDITIONAL_DECODER_GENERIC_SOURCE_PLACEMENT_OWED"
RUNTIME_GRAPH_LINK_BLOCKER: Final = "G17_CONDITIONAL_DECODER_RUNTIME_GRAPH_LINK_OWED"
ARCHIVE_RANGE_LINK_BLOCKER: Final = "G17_CONDITIONAL_LOGICAL_OPERAND_TO_ARCHIVE_RANGE_LINK_OWED"
INPUT_CONTRACT_ID: Final = "exact-decoded-camera-Y1:uint8[874,1164,3]:immutable"
OUTPUT_CONTRACT_ID: Final = "conditional-camera-Y0:uint8[874,1164,3]"

_SHA256_HEX = frozenset("0123456789abcdef")
_ALLOWED_MODULE_ROOTS: Final = frozenset(
    {
        "hashlib",
        "math",
        "numpy",
        "struct",
        "zlib",
    }
)
_ALLOWED_BUILTINS: Final = frozenset(
    {
        "ValueError",
        "bool",
        "bytearray",
        "bytes",
        "enumerate",
        "float",
        "int",
        "len",
        "max",
        "memoryview",
        "min",
        "range",
        "round",
        "slice",
        "sum",
        "tuple",
        "zip",
    }
)
_FORBIDDEN_DYNAMIC_NAMES: Final = frozenset(
    {
        "__import__",
        "compile",
        "eval",
        "exec",
        "getattr",
        "globals",
        "input",
        "locals",
        "open",
        "setattr",
        "vars",
    }
)
_FORBIDDEN_IO_ATTRIBUTES: Final = frozenset(
    {
        "fromfile",
        "getenv",
        "load",
        "loads",
        "memmap",
        "open",
        "read",
        "read_bytes",
        "read_text",
        "save",
        "savetxt",
        "tofile",
        "urlopen",
    }
)
_FORBIDDEN_DATA_NAME_FRAGMENTS: Final = (
    "direct_plane",
    "ground_truth",
    "gt_bank",
    "hidden_plane",
    "label_bank",
    "scorer",
    "target_bank",
    "teacher",
)


class G17ConditionalFullFrameReceiverError(ValueError):
    """The conditional receiver operation failed a source or execution guard."""


G17ConditionalY0LearnedQuotientDecoderV1: TypeAlias = Callable[
    [bytes, int, np.ndarray],
    np.ndarray,
]


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return _sha256(memoryview(np.ascontiguousarray(value)).cast("B"))


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise G17ConditionalFullFrameReceiverError(
            "receiver source closure is not finite canonical ASCII JSON"
        ) from exc


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in _SHA256_HEX for character in value):
        raise G17ConditionalFullFrameReceiverError(f"{name} must be canonical lowercase SHA-256")
    return value


def _source_path(value: object) -> Path:
    try:
        source = Path(str(inspect.getsourcefile(value) or "")).resolve()
    except (TypeError, OSError) as exc:
        raise G17ConditionalFullFrameReceiverError(
            "learned quotient decoder is not backed by exact Python source bytes"
        ) from exc
    if source.suffix != ".py" or not source.is_file():
        raise G17ConditionalFullFrameReceiverError(
            "learned quotient decoder is not backed by exact Python source bytes"
        )
    return source


def learned_quotient_decoder_source_sha256(
    decoder: G17ConditionalY0LearnedQuotientDecoderV1,
) -> str:
    """Return the exact source-file identity used by the G49 callable contract."""

    if not callable(decoder):
        raise G17ConditionalFullFrameReceiverError("learned quotient decoder must be a real callable")
    return _sha256(_source_path(decoder).read_bytes())


def _forbidden_data_name(name: str) -> bool:
    lowered = name.lower()
    return any(fragment in lowered for fragment in _FORBIDDEN_DATA_NAME_FRAGMENTS)


def _audit_function_ast(function: FunctionType) -> frozenset[str]:
    try:
        source = textwrap.dedent(inspect.getsource(function))
        tree = ast.parse(source)
    except (OSError, TypeError, IndentationError, SyntaxError) as exc:
        raise G17ConditionalFullFrameReceiverError("learned quotient decoder source cannot be audited") from exc
    attribute_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise G17ConditionalFullFrameReceiverError(
                "learned quotient decoder uses a local import outside its audited module-global allowlist"
            )
        if isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_DYNAMIC_NAMES:
                raise G17ConditionalFullFrameReceiverError(
                    "learned quotient decoder uses dynamic or external input access"
                )
            if _forbidden_data_name(node.id):
                raise G17ConditionalFullFrameReceiverError(
                    "learned quotient decoder names scorer/teacher/target/direct-plane state"
                )
        elif isinstance(node, ast.Attribute):
            attribute_names.add(node.attr)
            if node.attr in _FORBIDDEN_IO_ATTRIBUTES:
                raise G17ConditionalFullFrameReceiverError(
                    "learned quotient decoder attempts an external data read/write"
                )
            if _forbidden_data_name(node.attr):
                raise G17ConditionalFullFrameReceiverError(
                    "learned quotient decoder reaches scorer/teacher/target/direct-plane state"
                )
        elif isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            raw = node.value.encode("utf-8") if isinstance(node.value, str) else node.value
            try:
                literal = raw.decode("ascii").lower()
            except UnicodeDecodeError:
                literal = ""
            if any(fragment in literal for fragment in _FORBIDDEN_DATA_NAME_FRAGMENTS):
                raise G17ConditionalFullFrameReceiverError(
                    "learned quotient decoder embeds a forbidden data-class literal"
                )
            if len(literal) == 64 and all(character in _SHA256_HEX for character in literal):
                raise G17ConditionalFullFrameReceiverError(
                    "learned quotient decoder embeds an uncounted artifact identity"
                )
            if len(raw) > 1024:
                raise G17ConditionalFullFrameReceiverError(
                    "learned quotient decoder embeds an oversized literal payload"
                )
        elif (isinstance(node, (ast.List, ast.Tuple, ast.Set)) and len(node.elts) > 16) or (
            isinstance(node, ast.Dict) and len(node.keys) > 16
        ):
            raise G17ConditionalFullFrameReceiverError("learned quotient decoder embeds an oversized literal table")
    return frozenset(attribute_names)


def _audit_decoder_callable(
    decoder: G17ConditionalY0LearnedQuotientDecoderV1,
) -> tuple[str, str]:
    """Bind source and reject hidden/default/global video-specific inputs."""

    if type(decoder) is not FunctionType:
        raise G17ConditionalFullFrameReceiverError("learned quotient decoder must be one inspectable Python function")
    source_path = _source_path(decoder)
    source_bytes = source_path.read_bytes()
    source_sha256 = _sha256(source_bytes)
    pending = [decoder]
    seen: set[int] = set()
    audited_rows: list[tuple[str, str]] = []
    module_names: set[str] = set()
    while pending:
        function = pending.pop()
        if id(function) in seen:
            continue
        seen.add(id(function))
        if function.__closure__ is not None or function.__defaults__ is not None or function.__kwdefaults__ is not None:
            raise G17ConditionalFullFrameReceiverError("learned quotient decoder carries hidden closure/default values")
        if _source_path(function) != source_path:
            raise G17ConditionalFullFrameReceiverError(
                "learned quotient decoder helper escaped its exact source closure"
            )
        attribute_names = _audit_function_ast(function)
        closure = inspect.getclosurevars(function)
        if closure.nonlocals or (closure.unbound - attribute_names):
            raise G17ConditionalFullFrameReceiverError("learned quotient decoder has unresolved or nonlocal inputs")
        for name, _value in closure.builtins.items():
            if name not in _ALLOWED_BUILTINS:
                raise G17ConditionalFullFrameReceiverError(f"learned quotient decoder uses unapproved builtin {name!r}")
            if _forbidden_data_name(name):
                raise G17ConditionalFullFrameReceiverError("learned quotient decoder builtin names forbidden data")
        for name, value in closure.globals.items():
            if _forbidden_data_name(name):
                raise G17ConditionalFullFrameReceiverError("learned quotient decoder reaches forbidden global data")
            if isinstance(value, ModuleType):
                module_root = value.__name__.partition(".")[0]
                if module_root not in _ALLOWED_MODULE_ROOTS:
                    raise G17ConditionalFullFrameReceiverError(
                        f"learned quotient decoder imports unapproved module {value.__name__!r}"
                    )
                module_names.add(value.__name__)
            elif type(value) is FunctionType:
                pending.append(value)
            else:
                raise G17ConditionalFullFrameReceiverError("learned quotient decoder reaches an uncounted global value")
        audited_rows.append(
            (
                function.__qualname__,
                _sha256(marshal.dumps(function.__code__)),
            )
        )
    closure_payload = {
        "schema": "tac.g17_conditional_fullframe_callable_source_closure.v1",
        "source_file_sha256": source_sha256,
        "functions": sorted(audited_rows),
        "modules": sorted(module_names),
        "external_value_inputs": 0,
    }
    return source_sha256, _sha256(_canonical_json(closure_payload))


def _require_exact_y1(value: np.ndarray) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.uint8
        or value.shape != CAMERA_FRAME_SHAPE
        or not value.flags.c_contiguous
    ):
        raise G17ConditionalFullFrameReceiverError("Y1 must be exact C-contiguous uint8 camera frame [874,1164,3]")
    return value


def _require_exact_y0(value: object) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.uint8
        or value.shape != CAMERA_FRAME_SHAPE
        or not value.flags.c_contiguous
    ):
        raise G17ConditionalFullFrameReceiverError(
            "learned quotient decoder must return C-contiguous uint8 Y0 [874,1164,3]"
        )
    result = value.copy(order="C")
    result.setflags(write=False)
    return result


def _require_conditional_pose_manifest(
    manifest: G17CompilerPlacementManifestV1,
    *,
    conditional_pose_owner_id: str,
    source_pair_id: int,
) -> tuple[bytes, str, str, str]:
    if type(manifest) is not G17CompilerPlacementManifestV1:
        raise G17ConditionalFullFrameReceiverError("operation requires the canonical G17 compiler-placement manifest")
    if (
        type(conditional_pose_owner_id) is not str
        or not conditional_pose_owner_id
        or not conditional_pose_owner_id.isascii()
    ):
        raise G17ConditionalFullFrameReceiverError("conditional pose owner ID must be nonempty ASCII")
    if type(source_pair_id) is not int or not 0 <= source_pair_id < 600:
        raise G17ConditionalFullFrameReceiverError(
            "conditional operation source pair ID must be an exact n600 coordinate"
        )
    owner_records: tuple[G17CompilerPlacementRecordV1, ...] = tuple(
        row for row in manifest.records if row.logical_owner.owner_id == conditional_pose_owner_id
    )
    if not owner_records:
        raise G17ConditionalFullFrameReceiverError(
            "conditional pose owner is absent from the canonical placement manifest"
        )
    owner = owner_records[0].logical_owner
    if any(row.logical_owner is not owner for row in owner_records):
        raise G17ConditionalFullFrameReceiverError("conditional pose owner ID aliases multiple logical objects")
    if (
        owner.ownership_kind is not G17LogicalOwnershipKindV1.CHRONOLOGICAL_POSE
        or type(owner.value) is not G17ChronologicalPosePreimageV1
        or owner.parameter_spelling is None
        or owner.parameter_spelling.exact_parameter_bytes != owner.value.exact_bytes
        or owner.parameter_spelling.spelling_format != PARAMETER_SPELLING_FORMAT
    ):
        raise G17ConditionalFullFrameReceiverError(
            "conditional pose owner lacks exact learned-quotient parameter spelling"
        )
    operand = owner.value.exact_bytes
    if not operand:
        raise G17ConditionalFullFrameReceiverError("conditional pose owner has an empty counted operand")
    # This exact G49 parser closes the compact latent/parameter header, scalar
    # accounting, internal hashes, direct-plane structural bound, and EOF.  It
    # is the V1 anti-dense boundary; raw length alone is not used as a proof.
    try:
        learned_payload = _selected_preimage_program._parse_learned_payload(operand)
    except _selected_preimage_program.TaskspaceSelectedPreimageProgramError as exc:
        raise G17ConditionalFullFrameReceiverError(
            "conditional operand is not an exact compact learned-quotient payload"
        ) from exc
    if not any(start <= source_pair_id < stop for start, stop in learned_payload.active_pair_ranges):
        raise G17ConditionalFullFrameReceiverError(
            "conditional learned quotient does not address the exact source pair"
        )

    groups_by_id = {group.group_id: group for group in manifest.coding_groups}
    selected_group_ids: set[str] = set()
    for row in owner_records:
        if row.physical_coding_group_id is None:
            continue
        if (
            row.placement_class is not G17PlacementClassV1.COUNTED_VIDEO_STATISTIC
            or row.artifact_class is not G17ArtifactClassV1.IRREDUCIBLE_VIDEO_SPECIFIC_STATISTIC
            or row.video_specific_derivation is not True
            or row.packaged_inside_archive is not False
            or row.target_selected_constant is not False
            or row.payload_class != COUNTED_PAYLOAD_CLASS
        ):
            raise G17ConditionalFullFrameReceiverError(
                "conditional operand placement is not a counted irreducible video statistic"
            )
        if (
            row.scientific_role is not G17ScientificRoleV1.POSE_TRANSPORT_FRAME0
            or row.semantic_role is not G17SemanticStreamRoleV1.FIBER
        ):
            raise G17ConditionalFullFrameReceiverError(
                "conditional pose owner is not exact Y0-given-Y1 fiber incidence"
            )
        group = groups_by_id.get(row.physical_coding_group_id)
        if (
            group is None
            or group.receiver_consumer != RECEIVER_CONSUMER_ID
            or group.receiver_operation != OPERATION_ID
            or conditional_pose_owner_id not in group.logical_owner_ids
        ):
            raise G17ConditionalFullFrameReceiverError("conditional pose incidence names a foreign receiver operation")
        selected_group_ids.add(group.group_id)
    if not selected_group_ids:
        raise G17ConditionalFullFrameReceiverError("conditional pose owner has no counted physical coding group")
    if len(selected_group_ids) != 1:
        raise G17ConditionalFullFrameReceiverError(
            "G17_CONDITIONAL_OPERAND_TO_PHYSICAL_GROUP_SPAN_LINKER_OWED: "
            "V1 requires one exact possibly-many-to-many group for this operand"
        )
    # The monolithic member may contain unrelated P/G/Y1/common state.  The
    # conditional operand is the owner's exact spelling, not the whole member.
    if manifest.exact_member_bytes.count(operand) != 1:
        raise G17ConditionalFullFrameReceiverError(
            "conditional operand spelling is absent or ambiguous in the exact member"
        )
    return (
        operand,
        next(iter(selected_group_ids)),
        learned_payload.decoder_contract_id,
        learned_payload.decoder_implementation_source_sha256,
    )


def _invoke_decoder_once(
    decoder: G17ConditionalY0LearnedQuotientDecoderV1,
    *,
    operand: bytes,
    source_pair_id: int,
    exact_y1: np.ndarray,
) -> np.ndarray:
    immutable_y1 = exact_y1.copy(order="C")
    immutable_y1.setflags(write=False)
    before = _array_sha256(immutable_y1)
    try:
        output = decoder(operand, source_pair_id, immutable_y1)
    except Exception as exc:
        if _array_sha256(immutable_y1) != before:
            raise G17ConditionalFullFrameReceiverError(
                "learned quotient decoder mutated immutable Y1 before refusing"
            ) from exc
        raise
    if _array_sha256(immutable_y1) != before:
        raise G17ConditionalFullFrameReceiverError("learned quotient decoder mutated immutable Y1")
    return _require_exact_y0(output)


def _invoke_decoder_twice(
    decoder: G17ConditionalY0LearnedQuotientDecoderV1,
    *,
    operand: bytes,
    source_pair_id: int,
    exact_y1: np.ndarray,
) -> np.ndarray:
    first = _invoke_decoder_once(
        decoder,
        operand=operand,
        source_pair_id=source_pair_id,
        exact_y1=exact_y1,
    )
    second = _invoke_decoder_once(
        decoder,
        operand=operand,
        source_pair_id=source_pair_id,
        exact_y1=exact_y1,
    )
    if not np.array_equal(first, second):
        raise G17ConditionalFullFrameReceiverError("learned quotient decoder failed deterministic double execution")
    return first


def _valid_learned_operand_mutations(
    operand: bytes,
) -> tuple[tuple[int, bytes], ...]:
    """Rebuild parse-valid latent/parameter mutations under the exact G49 wire."""

    parsed = _selected_preimage_program._parse_learned_payload(operand)
    _magic, header_bytes = _selected_preimage_program._LEARNED_HEADER.unpack_from(operand)
    latent_start = _selected_preimage_program._LEARNED_HEADER.size + header_bytes
    parameter_start = latent_start + len(parsed.latent_payload)
    positions = tuple(
        sorted(
            {
                latent_start,
                latent_start + len(parsed.latent_payload) - 1,
                parameter_start,
                parameter_start + len(parsed.parameter_payload) - 1,
            }
        )
    )
    mutations: list[tuple[int, bytes]] = []
    for position in positions:
        latent = bytearray(parsed.latent_payload)
        parameters = bytearray(parsed.parameter_payload)
        if position < parameter_start:
            latent[position - latent_start] ^= 1
        else:
            parameters[position - parameter_start] ^= 1
        factor = _selected_preimage_program.build_learned_irreducible_quotient_factor(
            section_id="conditional-y0-valid-mutation",
            source_pair_start=parsed.active_pair_ranges[0][0],
            source_pair_stop_exclusive=parsed.active_pair_ranges[-1][1],
            decoder_contract_id=parsed.decoder_contract_id,
            decoder_implementation_source_sha256=parsed.decoder_implementation_source_sha256,
            model_family_id=parsed.model_family_id,
            latent_codec_id=parsed.latent_codec_id,
            parameter_codec_id=parsed.parameter_codec_id,
            latent_dtype=parsed.latent_dtype,
            parameter_dtype=parsed.parameter_dtype,
            latent_payload=bytes(latent),
            parameter_payload=bytes(parameters),
            source_receipt_sha256=_sha256(operand),
            active_pair_ranges=parsed.active_pair_ranges,
        )
        mutations.append((position, factor.payload))
    return tuple(mutations)


@dataclass(frozen=True, slots=True)
class G17ConditionalFullFrameReceiverReceiptV1:
    """Dense-free execution evidence for one conditional full-frame operation.

    ``logical_group_operand_liveness_results`` is derived from strict mutation
    of the uniquely spelled learned operand owned by the named physical group.
    It is not a claim that arbitrary raw ZIP-range mutation was reparsed into
    the same logical operand; archive-range linkage remains an unconditional
    admission blocker, and the many-group span linker remains fail-closed.
    """

    operation_id: str
    conditional_pose_owner_id: str
    source_pair_id: int
    decoder_contract_id: str
    decoder_implementation_source_sha256: str
    decoder_source_closure_sha256: str
    decoder_runtime_dependency_identity_sha256: str
    placement_manifest_sha256: str
    counted_archive_sha256: str
    counted_archive_bytes: int
    counted_operand_sha256: str
    counted_operand_bytes: int
    exact_decoded_y1_sha256: str
    conditional_y0_sha256: str
    logical_group_operand_liveness_results: tuple[tuple[str, Literal["CHANGED", "REFUSED"]], ...]
    operand_mutation_results: tuple[tuple[int, Literal["CHANGED", "REFUSED"]], ...]
    deterministic_double_decode: Literal[True]
    exact_decoded_y1_immutable: Literal[True]
    hidden_or_direct_plane_inputs: Literal[0]
    scorer_teacher_target_inputs: Literal[0]
    external_unbound_value_inputs: Literal[0]
    decoder_runtime_dependency_source_bound: Literal[True]
    public_archive_admission_blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.operation_id != OPERATION_ID
            or not self.conditional_pose_owner_id
            or not self.conditional_pose_owner_id.isascii()
            or type(self.source_pair_id) is not int
            or not 0 <= self.source_pair_id < 600
            or not self.decoder_contract_id
        ):
            raise G17ConditionalFullFrameReceiverError("conditional receiver receipt operation identity is invalid")
        for name in (
            "decoder_implementation_source_sha256",
            "decoder_source_closure_sha256",
            "decoder_runtime_dependency_identity_sha256",
            "placement_manifest_sha256",
            "counted_archive_sha256",
            "counted_operand_sha256",
            "exact_decoded_y1_sha256",
            "conditional_y0_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        if (
            type(self.counted_archive_bytes) is not int
            or self.counted_archive_bytes < 1
            or type(self.counted_operand_bytes) is not int
            or self.counted_operand_bytes < 1
        ):
            raise G17ConditionalFullFrameReceiverError("conditional receiver receipt byte accounting is invalid")
        if not self.logical_group_operand_liveness_results or not self.operand_mutation_results:
            raise G17ConditionalFullFrameReceiverError("conditional receiver receipt lacks mutation liveness")
        if any(
            status not in {"CHANGED", "REFUSED"}
            for _, status in self.logical_group_operand_liveness_results + self.operand_mutation_results
        ):
            raise G17ConditionalFullFrameReceiverError("conditional receiver receipt mutation status is invalid")
        if (
            self.deterministic_double_decode is not True
            or self.exact_decoded_y1_immutable is not True
            or self.hidden_or_direct_plane_inputs != 0
            or self.scorer_teacher_target_inputs != 0
            or self.external_unbound_value_inputs != 0
            or self.decoder_runtime_dependency_source_bound is not True
            or self.public_archive_admission_blockers
            != (
                GENERIC_SOURCE_PROVENANCE_BLOCKER,
                RUNTIME_GRAPH_LINK_BLOCKER,
                ARCHIVE_RANGE_LINK_BLOCKER,
            )
        ):
            raise G17ConditionalFullFrameReceiverError("conditional receiver receipt weakens a fail-closed invariant")


@dataclass(frozen=True, slots=True)
class G17ConditionalFullFrameReceiverResultV1:
    y0: np.ndarray = field(repr=False)
    receipt: G17ConditionalFullFrameReceiverReceiptV1

    def __post_init__(self) -> None:
        object.__setattr__(self, "y0", _require_exact_y0(self.y0))
        if type(self.receipt) is not G17ConditionalFullFrameReceiverReceiptV1:
            raise G17ConditionalFullFrameReceiverError("conditional receiver result lacks its exact execution receipt")
        if _array_sha256(self.y0) != self.receipt.conditional_y0_sha256:
            raise G17ConditionalFullFrameReceiverError("conditional Y0 differs from its execution receipt")


@dataclass(frozen=True, slots=True)
class G17ConditionalFullFrameY0GivenY1OperationV1:
    """Exact callable and counted-custody binding for the registered operation."""

    placement_manifest: G17CompilerPlacementManifestV1
    conditional_pose_owner_id: str
    source_pair_id: int
    decoder_contract_id: str
    decoder_implementation_source_sha256: str
    decoder_runtime_dependency: G17RuntimeDependencyFileV1
    learned_quotient_decoder: G17ConditionalY0LearnedQuotientDecoderV1 = field(repr=False)
    operation_id: str = field(default=OPERATION_ID, init=False)
    _decoder_source_closure_sha256: str = field(init=False, repr=False)
    _decoder_runtime_dependency_identity_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.decoder_contract_id) is not str or not self.decoder_contract_id:
            raise G17ConditionalFullFrameReceiverError("learned quotient decoder contract ID must be nonempty")
        declared_source = _require_sha256(
            self.decoder_implementation_source_sha256,
            name="decoder_implementation_source_sha256",
        )
        actual_source, source_closure = _audit_decoder_callable(self.learned_quotient_decoder)
        decoder_source_bytes = _source_path(self.learned_quotient_decoder).read_bytes()
        if (
            type(self.decoder_runtime_dependency) is not G17RuntimeDependencyFileV1
            or self.decoder_runtime_dependency.scope is not G17RuntimeFileScopeV1.SUBMISSION_RUNTIME_DEPENDENCY
            or self.decoder_runtime_dependency.exact_file_bytes != decoder_source_bytes
            or self.decoder_runtime_dependency.content_sha256 != actual_source
        ):
            raise G17ConditionalFullFrameReceiverError(
                "decoder source lacks exact SUBMISSION_RUNTIME_DEPENDENCY custody"
            )
        (
            _operand,
            _group_id,
            payload_contract_id,
            payload_source_sha256,
        ) = _require_conditional_pose_manifest(
            self.placement_manifest,
            conditional_pose_owner_id=self.conditional_pose_owner_id,
            source_pair_id=self.source_pair_id,
        )
        if actual_source != declared_source:
            raise G17ConditionalFullFrameReceiverError(
                "learned quotient decoder source differs from its bound identity"
            )
        if payload_contract_id != self.decoder_contract_id or payload_source_sha256 != declared_source:
            raise G17ConditionalFullFrameReceiverError(
                "learned quotient payload contract/source differs from the bound callable"
            )
        object.__setattr__(self, "_decoder_source_closure_sha256", source_closure)
        object.__setattr__(
            self,
            "_decoder_runtime_dependency_identity_sha256",
            self.decoder_runtime_dependency.identity_sha256,
        )

    def execute(
        self,
        exact_decoded_y1: np.ndarray,
    ) -> G17ConditionalFullFrameReceiverResultV1:
        y1 = _require_exact_y1(exact_decoded_y1)
        caller_y1_sha256 = _array_sha256(y1)
        actual_source, source_closure = _audit_decoder_callable(self.learned_quotient_decoder)
        decoder_source_bytes = _source_path(self.learned_quotient_decoder).read_bytes()
        if (
            actual_source != self.decoder_implementation_source_sha256
            or source_closure != self._decoder_source_closure_sha256
        ):
            raise G17ConditionalFullFrameReceiverError("learned quotient decoder source closure drifted after binding")
        if (
            self.decoder_runtime_dependency.exact_file_bytes != decoder_source_bytes
            or self.decoder_runtime_dependency.identity_sha256 != self._decoder_runtime_dependency_identity_sha256
        ):
            raise G17ConditionalFullFrameReceiverError("decoder runtime dependency custody drifted after binding")
        (
            operand,
            selected_group_id,
            payload_contract_id,
            payload_source_sha256,
        ) = _require_conditional_pose_manifest(
            self.placement_manifest,
            conditional_pose_owner_id=self.conditional_pose_owner_id,
            source_pair_id=self.source_pair_id,
        )
        if payload_contract_id != self.decoder_contract_id or payload_source_sha256 != actual_source:
            raise G17ConditionalFullFrameReceiverError("conditional operand callable custody drifted after binding")
        baseline_y0 = _invoke_decoder_twice(
            self.learned_quotient_decoder,
            operand=operand,
            source_pair_id=self.source_pair_id,
            exact_y1=y1,
        )
        if _array_sha256(y1) != caller_y1_sha256:
            raise G17ConditionalFullFrameReceiverError("conditional receiver mutated caller-owned exact Y1")

        archive = self.placement_manifest.exact_archive_bytes
        operand_results: list[tuple[int, Literal["CHANGED", "REFUSED"]]] = []
        for index, mutated_operand in _valid_learned_operand_mutations(operand):
            try:
                parsed_mutation = _selected_preimage_program._parse_learned_payload(mutated_operand)
                if (
                    parsed_mutation.decoder_contract_id != self.decoder_contract_id
                    or parsed_mutation.decoder_implementation_source_sha256 != actual_source
                ):
                    raise G17ConditionalFullFrameReceiverError("mutated operand changed callable custody")
                changed_y0 = _invoke_decoder_twice(
                    self.learned_quotient_decoder,
                    operand=mutated_operand,
                    source_pair_id=self.source_pair_id,
                    exact_y1=y1,
                )
            except Exception:
                operand_results.append((index, "REFUSED"))
            else:
                if np.array_equal(changed_y0, baseline_y0):
                    raise G17ConditionalFullFrameReceiverError(f"sampled counted operand byte {index} is receiver-dead")
                operand_results.append((index, "CHANGED"))
        group_status: Literal["CHANGED", "REFUSED"] = (
            "CHANGED" if any(status == "CHANGED" for _, status in operand_results) else "REFUSED"
        )
        group_results = ((selected_group_id, group_status),)
        if _array_sha256(y1) != caller_y1_sha256:
            raise G17ConditionalFullFrameReceiverError("mutation liveness execution mutated caller-owned exact Y1")
        receipt = G17ConditionalFullFrameReceiverReceiptV1(
            operation_id=OPERATION_ID,
            conditional_pose_owner_id=self.conditional_pose_owner_id,
            source_pair_id=self.source_pair_id,
            decoder_contract_id=self.decoder_contract_id,
            decoder_implementation_source_sha256=actual_source,
            decoder_source_closure_sha256=source_closure,
            decoder_runtime_dependency_identity_sha256=(self.decoder_runtime_dependency.identity_sha256),
            placement_manifest_sha256=self.placement_manifest.manifest_sha256,
            counted_archive_sha256=_sha256(archive),
            counted_archive_bytes=len(archive),
            counted_operand_sha256=_sha256(operand),
            counted_operand_bytes=len(operand),
            exact_decoded_y1_sha256=caller_y1_sha256,
            conditional_y0_sha256=_array_sha256(baseline_y0),
            logical_group_operand_liveness_results=group_results,
            operand_mutation_results=tuple(operand_results),
            deterministic_double_decode=True,
            exact_decoded_y1_immutable=True,
            hidden_or_direct_plane_inputs=0,
            scorer_teacher_target_inputs=0,
            external_unbound_value_inputs=0,
            decoder_runtime_dependency_source_bound=True,
            public_archive_admission_blockers=(
                GENERIC_SOURCE_PROVENANCE_BLOCKER,
                RUNTIME_GRAPH_LINK_BLOCKER,
                ARCHIVE_RANGE_LINK_BLOCKER,
            ),
        )
        return G17ConditionalFullFrameReceiverResultV1(
            y0=baseline_y0,
            receipt=receipt,
        )


@dataclass(frozen=True, slots=True)
class G17PublicReceiverOperationRegistryEntryV1:
    operation_id: str
    receiver_consumer: str
    input_contract_id: str
    output_contract_id: str
    executor: Callable[
        [G17ConditionalFullFrameY0GivenY1OperationV1, np.ndarray],
        G17ConditionalFullFrameReceiverResultV1,
    ] = field(repr=False)

    def __post_init__(self) -> None:
        if (
            self.operation_id != OPERATION_ID
            or self.receiver_consumer != RECEIVER_CONSUMER_ID
            or self.input_contract_id != INPUT_CONTRACT_ID
            or self.output_contract_id != OUTPUT_CONTRACT_ID
            or not callable(self.executor)
        ):
            raise G17ConditionalFullFrameReceiverError(
                "public receiver operation registry entry is not the closed G64 operation"
            )

    @property
    def implementation_source_sha256(self) -> str:
        return learned_quotient_decoder_source_sha256(self.executor)


def _execute_registered_operation(
    binding: G17ConditionalFullFrameY0GivenY1OperationV1,
    exact_decoded_y1: np.ndarray,
) -> G17ConditionalFullFrameReceiverResultV1:
    if type(binding) is not G17ConditionalFullFrameY0GivenY1OperationV1:
        raise G17ConditionalFullFrameReceiverError(
            "registry execution requires the exact conditional operation binding"
        )
    return binding.execute(exact_decoded_y1)


_REGISTRY_ENTRY: Final = G17PublicReceiverOperationRegistryEntryV1(
    operation_id=OPERATION_ID,
    receiver_consumer=RECEIVER_CONSUMER_ID,
    input_contract_id=INPUT_CONTRACT_ID,
    output_contract_id=OUTPUT_CONTRACT_ID,
    executor=_execute_registered_operation,
)

G17_PUBLIC_RECEIVER_OPERATION_REGISTRY_V1: Final[Mapping[str, G17PublicReceiverOperationRegistryEntryV1]] = (
    MappingProxyType({OPERATION_ID: _REGISTRY_ENTRY})
)


def execute_g17_public_receiver_operation(
    operation_id: str,
    binding: G17ConditionalFullFrameY0GivenY1OperationV1,
    exact_decoded_y1: np.ndarray,
) -> G17ConditionalFullFrameReceiverResultV1:
    """Execute one closed public operation; unknown strings never dispatch."""

    if type(operation_id) is not str:
        raise G17ConditionalFullFrameReceiverError("public receiver operation ID must be exact text")
    entry = G17_PUBLIC_RECEIVER_OPERATION_REGISTRY_V1.get(operation_id)
    if entry is None:
        raise G17ConditionalFullFrameReceiverError("unknown public receiver operation is not dispatchable")
    if binding.operation_id != operation_id:
        raise G17ConditionalFullFrameReceiverError("operation binding and registry dispatch ID differ")
    return entry.executor(binding, exact_decoded_y1)


def conditional_fullframe_receiver_source_closure() -> dict[str, object]:
    """Hash the exact local source closure required by a future public inflate."""

    source_paths = (
        Path(__file__).resolve(),
        _source_path(_selected_preimage_program._parse_learned_payload),
        _source_path(G17CompilerPlacementManifestV1),
    )
    rows = tuple(
        {
            "path": path.name,
            "sha256": _sha256(path.read_bytes()),
        }
        for path in source_paths
    )
    closure_payload = {
        "schema": "tac.g17_conditional_fullframe_receiver_source_closure.v1",
        "files": rows,
        "external_reads": [],
        "scorer_teacher_target_dependencies": [],
    }
    return {
        **closure_payload,
        "closure_sha256": _sha256(_canonical_json(closure_payload)),
    }


__all__ = [
    "ARCHIVE_RANGE_LINK_BLOCKER",
    "CAMERA_FRAME_BYTES",
    "CAMERA_FRAME_SHAPE",
    "COUNTED_PAYLOAD_CLASS",
    "G17_PUBLIC_RECEIVER_OPERATION_REGISTRY_V1",
    "INPUT_CONTRACT_ID",
    "OPERATION_ID",
    "OUTPUT_CONTRACT_ID",
    "RECEIVER_CONSUMER_ID",
    "RUNTIME_GRAPH_LINK_BLOCKER",
    "G17ConditionalFullFrameReceiverError",
    "G17ConditionalFullFrameReceiverReceiptV1",
    "G17ConditionalFullFrameReceiverResultV1",
    "G17ConditionalFullFrameY0GivenY1OperationV1",
    "G17ConditionalY0LearnedQuotientDecoderV1",
    "G17PublicReceiverOperationRegistryEntryV1",
    "conditional_fullframe_receiver_source_closure",
    "execute_g17_public_receiver_operation",
    "learned_quotient_decoder_source_sha256",
]
