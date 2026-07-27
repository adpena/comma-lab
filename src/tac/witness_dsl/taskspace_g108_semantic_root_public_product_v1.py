# SPDX-License-Identifier: MIT
"""Deterministic public product envelope for ``SemanticRootY1V1``.

This module owns only the receiver-facing archive envelope and the exact
callable *shape* required by the G102 state machine.  It deliberately does not
pretend that a fresh source compiler, source-lineage manifest, G17 placement,
or post-R evaluator closure exists.  Those producer-side calls fail closed.

The archive contains only the counted, self-describing semantic-root packet.
Generic reconstruction lives in the separately shipped public runtime under
``submissions/robust_current/g108_semantic_root_receiver``.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from tac.witness_dsl.taskspace_pfree_semantic_root_v1 import (
    GeneratorArchitectureV1,
    SemanticRootY1V1,
    encode_semantic_root_y1_v1,
)
from tac.witness_dsl.taskspace_pfree_semantic_root_v1 import (
    parse_semantic_root_y1_v1 as _parse_wire,
)

ARCHIVE_SCHEMA: Final = "tac.semantic_root_y1.public_archive.v1"
CAPABILITY_INTERFACE_ID: Final = "tac.semantic_root_y1.compiler_receiver.v2"
PACKET_MEMBER: Final = "semantic_root_y1_v1.bin"
SEMANTIC_VARIANT_ID: Final = "tac.semantic_root_y1.original_coordinr_film_mlp.v1"
FRAME0_VARIANT_ID: Final = "tac.semantic_root_y0.duplicate_y1_research_only.v1"
PUBLIC_RUNTIME_RELATIVE_ROOT: Final = "submissions/robust_current/g108_semantic_root_receiver"
PAIR_COUNT: Final = 600
FRAME_COUNT: Final = 1200
CAMERA_HEIGHT: Final = 874
CAMERA_WIDTH: Final = 1164
CHANNELS: Final = 3
SOURCE_COMPILER_BLOCKER: Final = "G108_FRESH_SOURCE_COMPILER_LINEAGE_AND_EXACT_POST_R_CLOSURE_OWED"
MAX_ARCHIVE_BYTES: Final = 2_100_000

_ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)
_HEX64: Final = frozenset("0123456789abcdef")


class G108PublicProductError(ValueError):
    """The public archive or receiver-product contract failed closed."""


class G108SourceClosureOwed(RuntimeError):
    """A producer-side G102 call was attempted before source closure."""


@dataclass(frozen=True, slots=True)
class ParsedSemanticRootY1V1:
    """G102-compatible parsed handle with exact byte re-emission."""

    root: SemanticRootY1V1
    packet: bytes

    def to_bytes(self) -> bytes:
        return self.packet


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    import json

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _zip_member(name: str, payload: bytes) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.flag_bits = 0
    return info, payload


def _runtime_tree_sha256() -> str:
    """Match the exact G102 public-runtime tree identity algorithm."""

    repo_root = Path(__file__).resolve().parents[3]
    root = repo_root / PUBLIC_RUNTIME_RELATIVE_ROOT
    if root.is_symlink() or not root.is_dir():
        return "0" * 64
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or path.is_dir():
            continue
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        payload = path.read_bytes()
        records.append(
            {
                "path": str(Path(PUBLIC_RUNTIME_RELATIVE_ROOT) / relative),
                "bytes": len(payload),
                "sha256": _sha256(payload),
                "role": "PUBLIC_RUNTIME_SOURCE",
                "candidate_dependency": False,
                "packaged_in_archive": False,
                "video_derived": False,
            }
        )
    return _sha256(_canonical_json(records)) if records else "0" * 64


def parse_semantic_root_y1_v1(packet: bytes) -> ParsedSemanticRootY1V1:
    """Strictly parse the committed G103 wire and retain exact re-emission."""

    if type(packet) is not bytes:
        raise G108PublicProductError("semantic-root packet must be exact bytes")
    root = _parse_wire(packet)
    if encode_semantic_root_y1_v1(root) != packet:
        raise G108PublicProductError("semantic-root packet changed under exact re-emission")
    if root.shared_generator.architecture is not GeneratorArchitectureV1.ORIGINAL_COORDINR_FILM_MLP_V1:
        raise G108PublicProductError("semantic-root packet requires a different public runtime variant")
    return ParsedSemanticRootY1V1(root=root, packet=packet)


def build_semantic_root_y1_v1_public_archive(packet: bytes) -> bytes:
    """Build the exact deterministic public archive consumed after extraction."""

    parsed = parse_semantic_root_y1_v1(packet)
    if parsed.to_bytes() != packet:
        raise AssertionError("internal semantic-root parse-back drifted")
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=False,
    ) as archive:
        info, payload = _zip_member(PACKET_MEMBER, packet)
        archive.writestr(
            info,
            payload,
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        )
    result = stream.getvalue()
    if not result or len(result) > MAX_ARCHIVE_BYTES:
        raise G108PublicProductError("public archive exceeds the closed semantic-root envelope cap")
    if parse_semantic_root_y1_v1_public_archive(result) != packet:
        raise AssertionError("internal public archive parse-back changed packet bytes")
    return result


def parse_semantic_root_y1_v1_public_archive(archive_bytes: bytes) -> bytes:
    """Safe-parse the exact one-member archive and return the counted packet."""

    if type(archive_bytes) is not bytes or not archive_bytes:
        raise G108PublicProductError("public archive must be nonempty exact bytes")
    if len(archive_bytes) > MAX_ARCHIVE_BYTES:
        raise G108PublicProductError("public archive exceeds the closed envelope cap")
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if names != [PACKET_MEMBER]:
                raise G108PublicProductError("public archive member set/order differs")
            info = infos[0]
            mode = (info.external_attr >> 16) & 0o170000
            if info.is_dir() or info.flag_bits & 0x1 or mode == 0o120000 or info.compress_type != zipfile.ZIP_DEFLATED:
                raise G108PublicProductError("public archive contains an unsafe/noncanonical member")
            packet = archive.read(info)
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise G108PublicProductError("public archive cannot be decoded") from exc
    parse_semantic_root_y1_v1(packet)
    return packet


def semantic_root_y1_v1_capability() -> dict[str, object]:
    """Expose the exact G102 key set while keeping producer authority false."""

    return {
        "interface_id": CAPABILITY_INTERFACE_ID,
        "producer_identity": "receiver_only_semantic_root_y1_v1",
        "own_lineage": False,
        "p_free": True,
        "full_population_n600": True,
        "label_topology_is_one_factor": True,
        "label_mask_palette_only": False,
        "scorer_native_rgb_appearance": True,
        "chroma_gauge": True,
        "parallax_gauge": True,
        "irreducible_rgb_quotient_seam": True,
        "exact_post_r_seg_closure": False,
        "exact_post_r_pose_closure": False,
        "teacher_quarantined": True,
        "scorer_free_receiver": True,
        "public_codec_section_sha256": _runtime_tree_sha256(),
    }


def _source_owed(*_args: object, **_kwargs: object) -> bytes:
    raise G108SourceClosureOwed(SOURCE_COMPILER_BLOCKER)


compile_semantic_root_y1_v1_stage = _source_owed
semantic_root_y1_v1_source_lineage_manifest = _source_owed
semantic_root_y1_v1_g17_whole_object_state = _source_owed


__all__ = [
    "ARCHIVE_SCHEMA",
    "CAPABILITY_INTERFACE_ID",
    "FRAME0_VARIANT_ID",
    "PACKET_MEMBER",
    "PUBLIC_RUNTIME_RELATIVE_ROOT",
    "SEMANTIC_VARIANT_ID",
    "SOURCE_COMPILER_BLOCKER",
    "G108PublicProductError",
    "G108SourceClosureOwed",
    "build_semantic_root_y1_v1_public_archive",
    "compile_semantic_root_y1_v1_stage",
    "parse_semantic_root_y1_v1",
    "parse_semantic_root_y1_v1_public_archive",
    "semantic_root_y1_v1_capability",
    "semantic_root_y1_v1_g17_whole_object_state",
    "semantic_root_y1_v1_source_lineage_manifest",
]
