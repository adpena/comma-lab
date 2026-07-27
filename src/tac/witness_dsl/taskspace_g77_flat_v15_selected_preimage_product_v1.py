# SPDX-License-Identifier: MIT
"""Flat semantic-P plus additive TSPPV2 product archive.

The V15 semantic program is already a canonical ZIP.  Nesting those complete
ZIP bytes inside the older monolithic G17 P section is both redundant and
rejected by that container.  This additive research product copies each exact
top-level semantic member payload into one outer ZIP and appends exactly one
counted TSPPV2 member.  It exhaustively races STORE/DEFLATE across all six
top-level members; nested ``predictor.zip`` stays opaque because recursively
flattening it is larger.

The receiver removes the TSPPV2 member and reconstructs the original canonical
STORE semantic ZIP from payload bytes plus fixed generic ZIP constants.  No
outer-product ``ZipInfo`` survives that boundary.  The same reconstruction API
works after an ordinary filesystem extraction, which models public ``unzip``.
It requires byte identity with the P SHA/length bound by TSPPV2 before opening
G74.  Semantic P and counted A are therefore distinct logical objects while
the archive stores the semantic member bytes only once.

This module closes a local flat product/demux receiver.  It is not wired into a
submission ``inflate.py``/``inflate.sh`` and makes no public-runtime,
candidate, score, Pose, or n600 claim.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal

from tac.witness_dsl.taskspace_g74_v15_roleaware_overlay_decoder_v1 import (
    V15RoleAwareOverlayDecoderV1,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v2 import (
    COMPACT_RECEIVER_PACKET_BLOCKER,
    PUBLIC_INFLATE_INTEGRATION_BLOCKER,
    BoundV15RoleAwareSelectedPreimageDecoderV2,
    SelectedPreimageDecodedPairV2,
    TaskspaceSelectedPreimageProgramV2,
    TaskspaceSelectedPreimageProgramV2Error,
    decode_selected_preimage_pair_v2,
    encode_selected_preimage_program_v2,
    parse_selected_preimage_program_v2,
)

TSPPV2_MEMBER_NAME: Final = "taskspace/selected_preimage_program.tsppv2"
SEMANTIC_MEMBER_NAMES: Final = (
    "manifest.json",
    "predictor.zip",
    "predict/movable_polygon_worldsheet.g1s",
    "render/receiver_realization.ddrp",
    "render/scorer_solved_templates.ddst",
)
CANONICAL_ZIP_DATE_TIME: Final = (1980, 1, 1, 0, 0, 0)
CANONICAL_EXTERNAL_ATTR: Final = 0o100644 << 16
PRODUCT_RECEIVER_ID: Final = "tac.g77.flat_v15_p_plus_tsppv2_demux.v1"
PRODUCT_COMPRESSION_POLICY_ID: Final = "EXHAUSTIVE_ALL_TOP_LEVEL_MEMBER_STORE_DEFLATE_RACE_V1"
FINAL_MULTI_ACTUATOR_DEMUX_BLOCKER: Final = "FINAL_ORDERED_Y1_THEN_Y0_MULTI_ACTUATOR_DEMUX_ABI_OWED"
OPEN_PRODUCT_BLOCKERS: Final = (
    PUBLIC_INFLATE_INTEGRATION_BLOCKER,
    "CROSS_HOST_TORCH_FLOAT32_DETERMINISM_OR_FIXED_CAMERA_BYTES_OWED",
    COMPACT_RECEIVER_PACKET_BLOCKER,
    FINAL_MULTI_ACTUATOR_DEMUX_BLOCKER,
)


class G77FlatV15SelectedPreimageProductError(ValueError):
    """Flat product bytes, semantic reconstruction, or decode failed closed."""


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _safe_member_name(name: object) -> str:
    if type(name) is not str or not name or "\\" in name:
        raise G77FlatV15SelectedPreimageProductError("ZIP member name must be nonempty canonical POSIX text")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise G77FlatV15SelectedPreimageProductError("ZIP member name is absolute, empty, or traversing")
    if name.endswith("/"):
        raise G77FlatV15SelectedPreimageProductError("flat product admits files only")
    return name


def _clone_zip_info(source: zipfile.ZipInfo) -> zipfile.ZipInfo:
    """Clone every metadata field that affects the current canonical STORE ZIP."""

    result = zipfile.ZipInfo(
        filename=_safe_member_name(source.filename),
        date_time=source.date_time,
    )
    for field_name in (
        "compress_type",
        "comment",
        "extra",
        "internal_attr",
        "external_attr",
        "create_system",
        "create_version",
        "extract_version",
        "flag_bits",
        "volume",
    ):
        setattr(result, field_name, getattr(source, field_name))
    return result


def _canonical_member_info(
    filename: str,
    *,
    compress_type: int,
) -> zipfile.ZipInfo:
    if filename not in (*SEMANTIC_MEMBER_NAMES, TSPPV2_MEMBER_NAME):
        raise G77FlatV15SelectedPreimageProductError("member is outside the closed G77 research product directory")
    if compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise G77FlatV15SelectedPreimageProductError("G77 admits STORE or DEFLATE only")
    result = zipfile.ZipInfo(
        filename=filename,
        date_time=CANONICAL_ZIP_DATE_TIME,
    )
    result.compress_type = compress_type
    result.create_system = 3
    result.create_version = 20
    result.extract_version = 20
    result.external_attr = CANONICAL_EXTERNAL_ATTR
    result.flag_bits = 0
    result.comment = b""
    result.extra = b""
    result.internal_attr = 0
    result.volume = 0
    return result


def _canonical_program_info(
    compress_type: int = zipfile.ZIP_DEFLATED,
) -> zipfile.ZipInfo:
    return _canonical_member_info(
        TSPPV2_MEMBER_NAME,
        compress_type=compress_type,
    )


def _canonical_semantic_info(name: str) -> zipfile.ZipInfo:
    if name not in SEMANTIC_MEMBER_NAMES:
        raise G77FlatV15SelectedPreimageProductError("semantic member name differs from canonical V15 P")
    return _canonical_member_info(name, compress_type=zipfile.ZIP_STORED)


def _require_metadata_equal(
    info: zipfile.ZipInfo,
    expected: zipfile.ZipInfo,
    *,
    label: str,
) -> None:
    for field_name in (
        "filename",
        "date_time",
        "compress_type",
        "comment",
        "extra",
        "internal_attr",
        "external_attr",
        "create_system",
        "create_version",
        "extract_version",
        "flag_bits",
        "volume",
    ):
        if getattr(info, field_name) != getattr(expected, field_name):
            raise G77FlatV15SelectedPreimageProductError(f"{label} metadata differs from canonical G77/V15 constants")


def _require_member_info(
    info: zipfile.ZipInfo,
    *,
    maximum_member_bytes: int,
    archive_kind: Literal["semantic_p", "flat_product"],
) -> None:
    name = _safe_member_name(info.filename)
    if info.flag_bits & 0x1:
        raise G77FlatV15SelectedPreimageProductError("encrypted ZIP member is forbidden")
    if info.file_size < 1 or info.file_size > maximum_member_bytes:
        raise G77FlatV15SelectedPreimageProductError("ZIP member size is empty or exceeds caller ceiling")
    if info.compress_type == zipfile.ZIP_STORED and info.compress_size != info.file_size:
        raise G77FlatV15SelectedPreimageProductError("STORE member compressed/file sizes differ")
    if archive_kind == "semantic_p":
        _require_metadata_equal(
            info,
            _canonical_semantic_info(name),
            label="semantic P member",
        )
        return
    if name == TSPPV2_MEMBER_NAME:
        expected = _canonical_program_info(info.compress_type)
    elif name in SEMANTIC_MEMBER_NAMES:
        expected = _canonical_member_info(name, compress_type=info.compress_type)
    else:
        raise G77FlatV15SelectedPreimageProductError(
            "flat product member differs from the closed G77 research directory"
        )
    _require_metadata_equal(info, expected, label="flat product member")


def _read_zip(
    payload: bytes,
    *,
    maximum_archive_bytes: int,
    maximum_member_bytes: int,
    archive_kind: Literal["semantic_p", "flat_product"],
) -> tuple[bytes, tuple[tuple[zipfile.ZipInfo, bytes], ...]]:
    if type(payload) is not bytes or not payload:
        raise G77FlatV15SelectedPreimageProductError("archive must be nonempty exact bytes")
    if len(payload) > maximum_archive_bytes:
        raise G77FlatV15SelectedPreimageProductError("archive exceeds caller byte ceiling")
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            if archive.comment:
                raise G77FlatV15SelectedPreimageProductError("G77 v1 requires an empty ZIP comment")
            infos = archive.infolist()
            names = tuple(info.filename for info in infos)
            if not infos or len(names) != len(set(names)):
                raise G77FlatV15SelectedPreimageProductError("ZIP member directory is empty or duplicated")
            rows: list[tuple[zipfile.ZipInfo, bytes]] = []
            for info in infos:
                _require_member_info(
                    info,
                    maximum_member_bytes=maximum_member_bytes,
                    archive_kind=archive_kind,
                )
                rows.append((info, archive.read(info)))
    except G77FlatV15SelectedPreimageProductError:
        raise
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        raise G77FlatV15SelectedPreimageProductError("strict ZIP reader refused archive") from exc
    return payload, tuple(rows)


def _write_zip(rows: tuple[tuple[zipfile.ZipInfo, bytes], ...]) -> bytes:
    if not rows:
        raise G77FlatV15SelectedPreimageProductError("cannot write an empty ZIP")
    output = io.BytesIO()
    try:
        with zipfile.ZipFile(output, "w") as archive:
            archive.comment = b""
            for info, payload in rows:
                archive.writestr(
                    _clone_zip_info(info),
                    payload,
                    compress_type=info.compress_type,
                    compresslevel=9,
                )
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        raise G77FlatV15SelectedPreimageProductError("deterministic ZIP construction failed") from exc
    return output.getvalue()


def _semantic_payload_rows(
    rows: tuple[tuple[zipfile.ZipInfo, bytes], ...],
) -> tuple[tuple[zipfile.ZipInfo, bytes], ...]:
    names = tuple(info.filename for info, _ in rows)
    if names != SEMANTIC_MEMBER_NAMES:
        raise G77FlatV15SelectedPreimageProductError("semantic members/order differ from canonical top-level V15 P")
    return tuple(
        (_canonical_semantic_info(name), payload)
        for name, (_, payload) in zip(SEMANTIC_MEMBER_NAMES, rows, strict=True)
    )


def _reconstruct_semantic_p_from_payloads(
    payloads: dict[str, bytes],
    *,
    maximum_member_bytes: int,
) -> bytes:
    if set(payloads) != set(SEMANTIC_MEMBER_NAMES):
        raise G77FlatV15SelectedPreimageProductError("extracted semantic payload directory is incomplete or has extras")
    rows: list[tuple[zipfile.ZipInfo, bytes]] = []
    for name in SEMANTIC_MEMBER_NAMES:
        payload = payloads[name]
        if type(payload) is not bytes or not payload or len(payload) > maximum_member_bytes:
            raise G77FlatV15SelectedPreimageProductError(
                "extracted semantic payload is empty or exceeds caller ceiling"
            )
        rows.append((_canonical_semantic_info(name), payload))
    return _write_zip(tuple(rows))


def reconstruct_semantic_p_from_flat_product(
    product_archive: bytes,
    *,
    maximum_archive_bytes: int,
    maximum_member_bytes: int,
) -> bytes:
    """Remove the final TSPPV2 member and reconstruct exact semantic P."""

    _, rows = _read_zip(
        product_archive,
        maximum_archive_bytes=maximum_archive_bytes,
        maximum_member_bytes=maximum_member_bytes,
        archive_kind="flat_product",
    )
    names = tuple(info.filename for info, _ in rows)
    if names != (*SEMANTIC_MEMBER_NAMES, TSPPV2_MEMBER_NAME):
        raise G77FlatV15SelectedPreimageProductError("flat product directory/order differs from G77 research product")
    return _reconstruct_semantic_p_from_payloads(
        {info.filename: payload for info, payload in rows[:-1]},
        maximum_member_bytes=maximum_member_bytes,
    )


@dataclass(frozen=True, slots=True)
class ParsedExtractedG77FlatV15SelectedPreimageProductV1:
    """Typed public-unzip demux retaining both reconstructed P and counted A."""

    semantic_p_archive: bytes
    program_packet: bytes
    program: TaskspaceSelectedPreimageProgramV2
    extracted_file_names: tuple[str, ...]
    public_unzip_reconstruction_closed: Literal[True] = True
    public_inflate_integration_closed: Literal[False] = False
    terminal_multi_actuator_demux_closed: Literal[False] = False
    research_only: Literal[True] = True
    score_claim: Literal[False] = False

    def __post_init__(self) -> None:
        if (
            type(self.semantic_p_archive) is not bytes
            or not self.semantic_p_archive
            or type(self.program_packet) is not bytes
            or not self.program_packet
            or type(self.program) is not TaskspaceSelectedPreimageProgramV2
            or encode_selected_preimage_program_v2(self.program) != self.program_packet
            or self.extracted_file_names != (*SEMANTIC_MEMBER_NAMES, TSPPV2_MEMBER_NAME)
        ):
            raise G77FlatV15SelectedPreimageProductError("extracted product lost canonical P/A custody")
        identity = self.program.semantic_program_identity
        if (
            len(self.semantic_p_archive) != identity.compiled_semantic_archive_bytes
            or _sha256(self.semantic_p_archive) != identity.compiled_semantic_archive_sha256
            or self.public_unzip_reconstruction_closed is not True
            or self.public_inflate_integration_closed is not False
            or self.terminal_multi_actuator_demux_closed is not False
            or self.research_only is not True
            or self.score_claim is not False
        ):
            raise G77FlatV15SelectedPreimageProductError("extracted product identity/truth labels became permissive")

    @property
    def open_product_blockers(self) -> tuple[str, ...]:
        return OPEN_PRODUCT_BLOCKERS

    def open_bound_decoder(
        self,
        *,
        verify_member_effects: bool = True,
    ) -> BoundV15RoleAwareSelectedPreimageDecoderV2:
        try:
            overlay = V15RoleAwareOverlayDecoderV1.open(
                self.semantic_p_archive,
                expected_archive_bytes=len(self.semantic_p_archive),
                expected_archive_sha256=_sha256(self.semantic_p_archive),
                verify_member_effects=verify_member_effects,
            )
            return BoundV15RoleAwareSelectedPreimageDecoderV2(
                semantic_identity=self.program.semantic_program_identity,
                target_custody_identity=self.program.target_custody_identity,
                decoder_identity=self.program.decoder_identity,
                overlay_decoder=overlay,
            )
        except Exception as exc:
            raise G77FlatV15SelectedPreimageProductError(
                "extracted product failed exact P/source/runtime reopen"
            ) from exc

    def decode_pair(
        self,
        pair_index: int,
        *,
        verify_member_effects: bool = True,
    ) -> SelectedPreimageDecodedPairV2:
        try:
            return decode_selected_preimage_pair_v2(
                self.program,
                pair_index,
                self.open_bound_decoder(verify_member_effects=verify_member_effects),
            )
        except TaskspaceSelectedPreimageProgramV2Error as exc:
            raise G77FlatV15SelectedPreimageProductError("extracted product TSPPV2 decode failed") from exc


def parse_extracted_g77_flat_v15_selected_preimage_product(
    product_directory: Path,
    *,
    maximum_member_bytes: int,
    maximum_program_bytes: int,
) -> ParsedExtractedG77FlatV15SelectedPreimageProductV1:
    """Demux P and counted A after ordinary extraction, without ``ZipInfo``."""

    if not isinstance(product_directory, Path) or not product_directory.is_dir():
        raise G77FlatV15SelectedPreimageProductError("extracted product root must be an existing exact Path directory")
    expected = {*SEMANTIC_MEMBER_NAMES, TSPPV2_MEMBER_NAME}
    observed: dict[str, bytes] = {}
    for path in product_directory.rglob("*"):
        if path.is_symlink():
            raise G77FlatV15SelectedPreimageProductError("extracted product cannot contain symlinks")
        if not path.is_file():
            continue
        name = path.relative_to(product_directory).as_posix()
        if name not in expected or name in observed:
            raise G77FlatV15SelectedPreimageProductError("extracted product file directory is unknown or duplicated")
        payload = path.read_bytes()
        ceiling = maximum_program_bytes if name == TSPPV2_MEMBER_NAME else maximum_member_bytes
        if not payload or len(payload) > ceiling:
            raise G77FlatV15SelectedPreimageProductError("extracted product file is empty or exceeds caller ceiling")
        observed[name] = payload
    if set(observed) != expected:
        raise G77FlatV15SelectedPreimageProductError("extracted product file directory is incomplete")
    try:
        program = parse_selected_preimage_program_v2(
            observed[TSPPV2_MEMBER_NAME],
            maximum_packet_bytes=maximum_program_bytes,
        )
    except TaskspaceSelectedPreimageProgramV2Error as exc:
        raise G77FlatV15SelectedPreimageProductError("extracted TSPPV2 file failed strict parser") from exc
    semantic = _reconstruct_semantic_p_from_payloads(
        {name: observed[name] for name in SEMANTIC_MEMBER_NAMES},
        maximum_member_bytes=maximum_member_bytes,
    )
    identity = program.semantic_program_identity
    if (
        len(semantic) != identity.compiled_semantic_archive_bytes
        or _sha256(semantic) != identity.compiled_semantic_archive_sha256
    ):
        raise G77FlatV15SelectedPreimageProductError("extracted payload reconstruction differs from TSPPV2 semantic P")
    return ParsedExtractedG77FlatV15SelectedPreimageProductV1(
        semantic_p_archive=semantic,
        program_packet=observed[TSPPV2_MEMBER_NAME],
        program=program,
        extracted_file_names=(*SEMANTIC_MEMBER_NAMES, TSPPV2_MEMBER_NAME),
    )


def reconstruct_semantic_p_from_extracted_product(
    product_directory: Path,
    *,
    maximum_member_bytes: int,
    maximum_program_bytes: int,
) -> bytes:
    """Compatibility helper returning P from the typed extracted demux."""

    return parse_extracted_g77_flat_v15_selected_preimage_product(
        product_directory,
        maximum_member_bytes=maximum_member_bytes,
        maximum_program_bytes=maximum_program_bytes,
    ).semantic_p_archive


@dataclass(frozen=True, slots=True)
class ParsedG77FlatV15SelectedPreimageProductV1:
    """Strict demux of exact flat bytes into reconstructed P and counted A."""

    archive_bytes: bytes
    semantic_p_archive: bytes
    program_packet: bytes
    program: TaskspaceSelectedPreimageProgramV2
    semantic_member_names: tuple[str, ...]
    semantic_compression_bits: str
    program_compression_bit: str
    actuator_member_slots: tuple[str, ...] = (TSPPV2_MEMBER_NAME,)
    product_receiver_id: Literal["tac.g77.flat_v15_p_plus_tsppv2_demux.v1"] = PRODUCT_RECEIVER_ID
    compression_policy_id: Literal["EXHAUSTIVE_ALL_TOP_LEVEL_MEMBER_STORE_DEFLATE_RACE_V1"] = (
        PRODUCT_COMPRESSION_POLICY_ID
    )
    semantic_p_occurrences: Literal[1] = 1
    tsppv2_occurrences: Literal[1] = 1
    nested_complete_semantic_zip_stored: Literal[False] = False
    public_unzip_reconstruction_closed: Literal[True] = True
    public_inflate_integration_closed: Literal[False] = False
    terminal_multi_actuator_demux_closed: Literal[False] = False
    research_only: Literal[True] = True
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False

    def __post_init__(self) -> None:
        if (
            type(self.archive_bytes) is not bytes
            or not self.archive_bytes
            or type(self.semantic_p_archive) is not bytes
            or not self.semantic_p_archive
            or type(self.program_packet) is not bytes
            or not self.program_packet
        ):
            raise G77FlatV15SelectedPreimageProductError("parsed product lost exact byte custody")
        if type(self.program) is not TaskspaceSelectedPreimageProgramV2:
            raise G77FlatV15SelectedPreimageProductError("parsed product program changed exact type")
        if encode_selected_preimage_program_v2(self.program) != self.program_packet:
            raise G77FlatV15SelectedPreimageProductError("parsed product program changed on re-encode")
        identity = self.program.semantic_program_identity
        if (
            len(self.semantic_p_archive) != identity.compiled_semantic_archive_bytes
            or _sha256(self.semantic_p_archive) != identity.compiled_semantic_archive_sha256
        ):
            raise G77FlatV15SelectedPreimageProductError("reconstructed semantic P differs from TSPPV2 identity")
        if self.semantic_member_names != SEMANTIC_MEMBER_NAMES:
            raise G77FlatV15SelectedPreimageProductError("semantic member directory differs from canonical V15 P")
        if (
            type(self.semantic_compression_bits) is not str
            or len(self.semantic_compression_bits) != len(SEMANTIC_MEMBER_NAMES)
            or any(bit not in "01" for bit in self.semantic_compression_bits)
            or self.program_compression_bit not in {"0", "1"}
            or self.actuator_member_slots != (TSPPV2_MEMBER_NAME,)
        ):
            raise G77FlatV15SelectedPreimageProductError("research product compression/actuator slots are invalid")
        truth = (
            self.product_receiver_id == PRODUCT_RECEIVER_ID
            and self.compression_policy_id == PRODUCT_COMPRESSION_POLICY_ID
            and self.semantic_p_occurrences == 1
            and self.tsppv2_occurrences == 1
            and self.nested_complete_semantic_zip_stored is False
            and self.public_unzip_reconstruction_closed is True
            and self.public_inflate_integration_closed is False
            and self.terminal_multi_actuator_demux_closed is False
            and self.research_only is True
            and self.candidate_claim is False
            and self.score_claim is False
        )
        if not truth:
            raise G77FlatV15SelectedPreimageProductError("flat product truth labels became permissive")

    @property
    def archive_sha256(self) -> str:
        return _sha256(self.archive_bytes)

    @property
    def semantic_p_sha256(self) -> str:
        return _sha256(self.semantic_p_archive)

    @property
    def program_packet_sha256(self) -> str:
        return _sha256(self.program_packet)

    @property
    def open_product_blockers(self) -> tuple[str, ...]:
        return OPEN_PRODUCT_BLOCKERS

    def open_bound_decoder(
        self,
        *,
        verify_member_effects: bool = True,
    ) -> BoundV15RoleAwareSelectedPreimageDecoderV2:
        try:
            overlay = V15RoleAwareOverlayDecoderV1.open(
                self.semantic_p_archive,
                expected_archive_bytes=len(self.semantic_p_archive),
                expected_archive_sha256=self.semantic_p_sha256,
                verify_member_effects=verify_member_effects,
            )
        except Exception as exc:
            raise G77FlatV15SelectedPreimageProductError("reconstructed semantic P failed exact G74 reopen") from exc
        try:
            return BoundV15RoleAwareSelectedPreimageDecoderV2(
                semantic_identity=self.program.semantic_program_identity,
                target_custody_identity=self.program.target_custody_identity,
                decoder_identity=self.program.decoder_identity,
                overlay_decoder=overlay,
            )
        except TaskspaceSelectedPreimageProgramV2Error as exc:
            raise G77FlatV15SelectedPreimageProductError("flat product source/runtime binding failed") from exc

    def decode_pair(
        self,
        pair_index: int,
        *,
        verify_member_effects: bool = True,
    ) -> SelectedPreimageDecodedPairV2:
        decoder = self.open_bound_decoder(verify_member_effects=verify_member_effects)
        try:
            return decode_selected_preimage_pair_v2(
                self.program,
                pair_index,
                decoder,
            )
        except TaskspaceSelectedPreimageProgramV2Error as exc:
            raise G77FlatV15SelectedPreimageProductError("flat product TSPPV2 decode failed") from exc


def parse_g77_flat_v15_selected_preimage_product(
    archive_bytes: bytes,
    *,
    maximum_archive_bytes: int,
    maximum_member_bytes: int,
    maximum_program_bytes: int,
) -> ParsedG77FlatV15SelectedPreimageProductV1:
    """Strictly demux, reconstruct exact P, and parse exact counted TSPPV2."""

    payload, rows = _read_zip(
        archive_bytes,
        maximum_archive_bytes=maximum_archive_bytes,
        maximum_member_bytes=maximum_member_bytes,
        archive_kind="flat_product",
    )
    names = tuple(info.filename for info, _ in rows)
    if names != (*SEMANTIC_MEMBER_NAMES, TSPPV2_MEMBER_NAME):
        raise G77FlatV15SelectedPreimageProductError("flat product directory/order differs from G77 research product")
    program_packet = rows[-1][1]
    try:
        program = parse_selected_preimage_program_v2(
            program_packet,
            maximum_packet_bytes=maximum_program_bytes,
        )
    except TaskspaceSelectedPreimageProgramV2Error as exc:
        raise G77FlatV15SelectedPreimageProductError("flat product TSPPV2 member failed strict parser") from exc
    semantic = _reconstruct_semantic_p_from_payloads(
        {info.filename: member_payload for info, member_payload in rows[:-1]},
        maximum_member_bytes=maximum_member_bytes,
    )
    compression_bits = "".join("1" if info.compress_type == zipfile.ZIP_DEFLATED else "0" for info, _ in rows[:-1])
    program_compression_bit = "1" if rows[-1][0].compress_type == zipfile.ZIP_DEFLATED else "0"
    parsed = ParsedG77FlatV15SelectedPreimageProductV1(
        archive_bytes=payload,
        semantic_p_archive=semantic,
        program_packet=program_packet,
        program=program,
        semantic_member_names=SEMANTIC_MEMBER_NAMES,
        semantic_compression_bits=compression_bits,
        program_compression_bit=program_compression_bit,
    )
    if (
        reconstruct_semantic_p_from_flat_product(
            payload,
            maximum_archive_bytes=maximum_archive_bytes,
            maximum_member_bytes=maximum_member_bytes,
        )
        != semantic
    ):
        raise G77FlatV15SelectedPreimageProductError("flat semantic reconstruction is nondeterministic")
    return parsed


@dataclass(frozen=True, slots=True)
class G77FlatV15SelectedPreimageProductBuildV1:
    archive_bytes: bytes
    parsed: ParsedG77FlatV15SelectedPreimageProductV1
    complete_compression_bits: str
    zlib_runtime_version: str = zlib.ZLIB_RUNTIME_VERSION
    method_profiles_evaluated: Literal[64] = 64
    deflate_level: Literal[9] = 9

    def __post_init__(self) -> None:
        if (
            type(self.archive_bytes) is not bytes
            or type(self.parsed) is not ParsedG77FlatV15SelectedPreimageProductV1
            or self.parsed.archive_bytes != self.archive_bytes
            or self.complete_compression_bits
            != self.parsed.semantic_compression_bits + self.parsed.program_compression_bit
            or type(self.zlib_runtime_version) is not str
            or not self.zlib_runtime_version
            or self.method_profiles_evaluated != 64
            or self.deflate_level != 9
        ):
            raise G77FlatV15SelectedPreimageProductError("flat product build lost exact race/parse-back custody")


def _optimized_flat_product_archive(
    semantic_rows: tuple[tuple[zipfile.ZipInfo, bytes], ...],
    program_packet: bytes,
) -> tuple[bytes, str]:
    """Exhaustively choose STORE/DEFLATE for all six top-level members."""

    canonical_rows = _semantic_payload_rows(semantic_rows)
    best: tuple[int, str, bytes] | None = None
    member_count = len(canonical_rows) + 1
    for mask in range(1 << member_count):
        bits = "".join("1" if mask & (1 << index) else "0" for index in range(member_count))
        candidate_rows = (
            *(
                (
                    _canonical_member_info(
                        info.filename,
                        compress_type=(zipfile.ZIP_DEFLATED if bits[index] == "1" else zipfile.ZIP_STORED),
                    ),
                    payload,
                )
                for index, (info, payload) in enumerate(canonical_rows)
            ),
            (
                _canonical_program_info(zipfile.ZIP_DEFLATED if bits[-1] == "1" else zipfile.ZIP_STORED),
                program_packet,
            ),
        )
        candidate = _write_zip(candidate_rows)
        row = (len(candidate), bits, candidate)
        if best is None or row[:2] < best[:2]:
            best = row
    if best is None:  # pragma: no cover - five canonical members are mandatory
        raise G77FlatV15SelectedPreimageProductError("compression method race produced no product")
    return best[2], best[1]


def build_g77_flat_v15_selected_preimage_product(
    *,
    semantic_p_archive: bytes,
    program_packet: bytes,
    maximum_archive_bytes: int,
    maximum_member_bytes: int,
    maximum_program_bytes: int,
) -> G77FlatV15SelectedPreimageProductBuildV1:
    """Flatten exact P members, append TSPPV2 once, and double reopen."""

    semantic_payload, semantic_rows = _read_zip(
        semantic_p_archive,
        maximum_archive_bytes=maximum_archive_bytes,
        maximum_member_bytes=maximum_member_bytes,
        archive_kind="semantic_p",
    )
    canonical_semantic_rows = _semantic_payload_rows(semantic_rows)
    if _write_zip(canonical_semantic_rows) != semantic_payload:
        raise G77FlatV15SelectedPreimageProductError("semantic P is not byte-identically reconstructable from members")
    try:
        program = parse_selected_preimage_program_v2(
            program_packet,
            maximum_packet_bytes=maximum_program_bytes,
        )
    except TaskspaceSelectedPreimageProgramV2Error as exc:
        raise G77FlatV15SelectedPreimageProductError("builder TSPPV2 packet failed strict parser") from exc
    if program.semantic_program_identity.compiled_semantic_archive_sha256 != _sha256(
        semantic_payload
    ) or program.semantic_program_identity.compiled_semantic_archive_bytes != len(semantic_payload):
        raise G77FlatV15SelectedPreimageProductError("TSPPV2 semantic identity differs from exact P input")
    archive_bytes, compression_bits = _optimized_flat_product_archive(
        canonical_semantic_rows,
        program_packet,
    )
    first = parse_g77_flat_v15_selected_preimage_product(
        archive_bytes,
        maximum_archive_bytes=maximum_archive_bytes,
        maximum_member_bytes=maximum_member_bytes,
        maximum_program_bytes=maximum_program_bytes,
    )
    second = parse_g77_flat_v15_selected_preimage_product(
        archive_bytes,
        maximum_archive_bytes=maximum_archive_bytes,
        maximum_member_bytes=maximum_member_bytes,
        maximum_program_bytes=maximum_program_bytes,
    )
    if first != second or first.semantic_p_archive != semantic_payload:
        raise G77FlatV15SelectedPreimageProductError("flat product deterministic double parse-back failed")
    if first.semantic_compression_bits + first.program_compression_bit != compression_bits:
        raise G77FlatV15SelectedPreimageProductError("compression race result differs from strict product parse")
    return G77FlatV15SelectedPreimageProductBuildV1(
        archive_bytes=archive_bytes,
        parsed=first,
        complete_compression_bits=compression_bits,
    )


__all__ = [
    "FINAL_MULTI_ACTUATOR_DEMUX_BLOCKER",
    "OPEN_PRODUCT_BLOCKERS",
    "PRODUCT_COMPRESSION_POLICY_ID",
    "PRODUCT_RECEIVER_ID",
    "SEMANTIC_MEMBER_NAMES",
    "TSPPV2_MEMBER_NAME",
    "G77FlatV15SelectedPreimageProductBuildV1",
    "G77FlatV15SelectedPreimageProductError",
    "ParsedExtractedG77FlatV15SelectedPreimageProductV1",
    "ParsedG77FlatV15SelectedPreimageProductV1",
    "build_g77_flat_v15_selected_preimage_product",
    "parse_extracted_g77_flat_v15_selected_preimage_product",
    "parse_g77_flat_v15_selected_preimage_product",
    "reconstruct_semantic_p_from_extracted_product",
    "reconstruct_semantic_p_from_flat_product",
]
