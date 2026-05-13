"""SE4R scorer-conditional range-coder archive-section format.

The SE4R archive section is a self-describing binary blob with the layout:

    +---------+------+-------+----------+-----------+-------------+----------+----------+--------+
    | MAGIC   | VER  | FLAGS | SIDE_DIM | ALPHABET  | N_SYMBOLS   | META_LEN | METADATA | PAYLOAD|
    | (4 B)   | (1B) | (1B)  | (2 B BE) | (4 B BE)  | (4 B BE)    | (4 B BE) | JSON     |        |
    +---------+------+-------+----------+-----------+-------------+----------+----------+--------+

* ``MAGIC`` — ``b"SE4R"`` (4 bytes), see ``SE4R_MAGIC``.
* ``VER``   — format version byte; current is ``JSCC_FORMAT_VERSION = 2``.
* ``FLAGS`` — custody flags. Bit 0 = model embedded/charged. Bit 1 =
  side-state reconstruction embedded/charged. Bit 2 = caller requested an
  exact-eval-ready section. The manifest still keeps dispatch authority false.
* ``SIDE_DIM`` — uint16 big-endian; the model's side-state dimensionality.
* ``ALPHABET`` — uint32 big-endian; alphabet size K.
* ``N_SYMBOLS`` — uint32 big-endian; number of encoded symbols.
* ``META_LEN`` — uint32 big-endian; length of UTF-8 JSON custody metadata.
* ``METADATA`` — canonical JSON with proxy-only score authority and embedded
  model / side-state reconstruction contracts.
* ``PAYLOAD`` — the range-coded byte stream.

NOTE: this primitive is proxy-only unless its integration archive charges and
embeds both the conditional probability model and the deterministic side-state
reconstruction contract. Even then, this section manifest is not score
authority: exact eval packet paths remain the only route to dispatch readiness.

Cross-references
----------------
- Sister archive format for unconditional arithmetic coding:
  ``tac.packet_compiler.pr103_arithmetic_coding`` (different magic, different
  layout — JSCC is intentionally a separate primitive).
- Sister magic-codec wrapper grammar:
  ``tac.packet_compiler.magic_codec``.

Lane: ``lane_implement_iglt_ternary_jscc_kc3_canonical_20260513``.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

from tac.codec.jscc.entropy_coder import (
    JSCC_FORMAT_VERSION,
    JSCC_MAGIC,
    SE4R_MAGIC,
)

__all__ = [
    "JSCC_PROXY_EVIDENCE_GRADE",
    "JSCCArchiveSection",
    "JSCCCustodyContract",
    "JSCCSectionManifest",
    "parse_jscc_section",
    "serialize_jscc_section",
]

_FLAG_MODEL_EMBEDDED_CHARGED = 1 << 0
_FLAG_SIDE_STATE_EMBEDDED_CHARGED = 1 << 1
_FLAG_EXACT_EVAL_READY_REQUESTED = 1 << 2

JSCC_PROXY_EVIDENCE_GRADE = "proxy_only_jscc_section"


@dataclass(frozen=True)
class JSCCCustodyContract:
    """Embedded/charged custody contract for a JSCC dependency blob."""

    embedded: bool
    charged_bytes: int
    sha256: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.charged_bytes < 0:
            raise ValueError("charged_bytes must be non-negative")
        if self.sha256 is not None:
            if len(self.sha256) != 64:
                raise ValueError("sha256 must be 64 hex characters")
            try:
                int(self.sha256, 16)
            except ValueError as exc:
                raise ValueError("sha256 must be lowercase hex") from exc

    @property
    def charged_and_embedded(self) -> bool:
        return self.embedded and self.charged_bytes > 0

    def to_json(self) -> dict[str, object]:
        return {
            "embedded": bool(self.embedded),
            "charged_bytes": int(self.charged_bytes),
            "sha256": self.sha256,
            "description": self.description,
            "charged_and_embedded": self.charged_and_embedded,
        }


@dataclass(frozen=True)
class JSCCSectionManifest:
    """Typed metadata for a JSCC archive section.

    Used by the packet_compiler grammar / parser_section_manifest pipeline
    (per CLAUDE.md HNeRV parity discipline lesson 3 — every monolithic
    archive must declare fixed offsets).

    Attributes:
        magic: ``SE4R_MAGIC`` (4 bytes).
        version: format-version byte.
        flags: custody bit flags stored in the binary header.
        side_dim: model's side-state dimensionality.
        alphabet_size: alphabet size K.
        n_symbols: number of encoded symbols.
        custody_metadata_offset: byte offset where custody metadata starts.
        custody_metadata_length: byte length of custody metadata.
        payload_offset: byte offset where the encoded payload starts within
            this section.
        payload_length: byte length of the encoded payload.
        total_section_bytes: header_bytes + payload_length.
        custody_metadata: JSON-safe proxy/custody metadata.
        evidence_grade: proxy-only evidence grade.
        proxy: fixed true for this primitive.
        proxy_only: fixed true for this primitive.
        score_claim: fixed false.
        promotion_eligible: fixed false.
        ready_for_exact_eval_dispatch: fixed false.
    """

    magic: bytes
    version: int
    flags: int
    side_dim: int
    alphabet_size: int
    n_symbols: int
    custody_metadata_offset: int
    custody_metadata_length: int
    payload_offset: int
    payload_length: int
    total_section_bytes: int
    custody_metadata: dict[str, object]
    evidence_grade: str
    proxy: bool
    proxy_only: bool
    score_claim: bool
    promotion_eligible: bool
    ready_for_exact_eval_dispatch: bool


@dataclass(frozen=True)
class JSCCArchiveSection:
    """A parsed JSCC archive section (header + payload).

    Attributes:
        manifest: typed metadata.
        payload: the raw range-coded byte stream.
    """

    manifest: JSCCSectionManifest
    payload: bytes


# Fixed header layout, big-endian:
#   magic[4] || version[1] || flags[1] || side_dim[2] ||
#   alphabet[4] || n_symbols[4] || custody_metadata_len[4]
# = 20 bytes header
_HEADER_STRUCT = struct.Struct(">4sBBHIII")
HEADER_BYTES: int = _HEADER_STRUCT.size  # 20


def _normalize_contract(
    contract: JSCCCustodyContract | dict[str, Any] | None,
) -> JSCCCustodyContract | None:
    if contract is None:
        return None
    if isinstance(contract, JSCCCustodyContract):
        return contract
    return JSCCCustodyContract(
        embedded=bool(contract.get("embedded", False)),
        charged_bytes=int(contract.get("charged_bytes", 0)),
        sha256=contract.get("sha256"),  # type: ignore[arg-type]
        description=str(contract.get("description", "")),
    )


def _contract_json(contract: JSCCCustodyContract | None) -> dict[str, object]:
    if contract is None:
        return {
            "embedded": False,
            "charged_bytes": 0,
            "sha256": None,
            "description": "",
            "charged_and_embedded": False,
        }
    return contract.to_json()


def _build_custody_metadata(
    *,
    exact_eval_ready_requested: bool,
    model_contract: JSCCCustodyContract | None,
    side_state_contract: JSCCCustodyContract | None,
) -> dict[str, object]:
    model_ready = bool(model_contract and model_contract.charged_and_embedded)
    side_ready = bool(side_state_contract and side_state_contract.charged_and_embedded)
    contract_complete = model_ready and side_ready
    return {
        "schema_version": 1,
        "format_family": "se4r_scorer_conditional_range_coder",
        "legacy_huffman_magic": "JSCC",
        "section_magic": SE4R_MAGIC.decode("ascii"),
        "evidence_grade": JSCC_PROXY_EVIDENCE_GRADE,
        "proxy": True,
        "proxy_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "exact_eval_ready_requested": bool(exact_eval_ready_requested),
        "exact_eval_blocker": (
            "section_manifest_is_proxy_only_until_full_archive_exact_eval"
        ),
        "model_contract": _contract_json(model_contract),
        "side_state_reconstruction_contract": _contract_json(side_state_contract),
        "embedded_side_contract_complete": contract_complete,
        "dispatch_authority": "none",
    }


def serialize_jscc_section(
    payload: bytes,
    side_dim: int,
    alphabet_size: int,
    n_symbols: int,
    version: int = JSCC_FORMAT_VERSION,
    *,
    model_contract: JSCCCustodyContract | dict[str, Any] | None = None,
    side_state_contract: JSCCCustodyContract | dict[str, Any] | None = None,
    ready_for_exact_eval_dispatch: bool = False,
) -> bytes:
    """Wrap a JSCC payload in the archive-section header.

    Args:
        payload: range-coded byte stream from
            ``encode_jscc_stream`` / ``ScorerConditionalEntropyCoder.encode``.
        side_dim: model's side-state dimensionality.
        alphabet_size: alphabet size K.
        n_symbols: number of encoded symbols.
        version: format version. Default is current.
        model_contract: optional embedded/charged model custody contract.
        side_state_contract: optional embedded/charged side-state
            reconstruction custody contract.
        ready_for_exact_eval_dispatch: caller's requested dispatch readiness.
            A true request fails closed unless both dependency contracts are
            embedded and charged. The serialized manifest still records
            ``ready_for_exact_eval_dispatch=false`` because this section is not
            exact-eval score authority.

    Returns:
        Header + payload bytes (length = HEADER_BYTES + len(payload)).

    Raises:
        ValueError: on out-of-range inputs.
    """
    if not (0 <= version <= 255):
        raise ValueError(f"version must fit in uint8, got {version}")
    if not (0 <= side_dim <= 0xFFFF):
        raise ValueError(f"side_dim must fit in uint16, got {side_dim}")
    if not (0 <= alphabet_size <= 0xFFFFFFFF):
        raise ValueError(
            f"alphabet_size must fit in uint32, got {alphabet_size}"
        )
    if not (0 <= n_symbols <= 0xFFFFFFFF):
        raise ValueError(
            f"n_symbols must fit in uint32, got {n_symbols}"
        )
    if alphabet_size < 2:
        raise ValueError(f"alphabet_size must be >= 2, got {alphabet_size}")
    model = _normalize_contract(model_contract)
    side_state = _normalize_contract(side_state_contract)
    if ready_for_exact_eval_dispatch:
        if not (model and model.charged_and_embedded):
            raise ValueError(
                "exact-eval-ready JSCC section requires embedded charged "
                "model_contract"
            )
        if not (side_state and side_state.charged_and_embedded):
            raise ValueError(
                "exact-eval-ready JSCC section requires embedded charged "
                "side_state_contract"
            )
    flags = 0
    if model and model.charged_and_embedded:
        flags |= _FLAG_MODEL_EMBEDDED_CHARGED
    if side_state and side_state.charged_and_embedded:
        flags |= _FLAG_SIDE_STATE_EMBEDDED_CHARGED
    if ready_for_exact_eval_dispatch:
        flags |= _FLAG_EXACT_EVAL_READY_REQUESTED
    custody_metadata = _build_custody_metadata(
        exact_eval_ready_requested=ready_for_exact_eval_dispatch,
        model_contract=model,
        side_state_contract=side_state,
    )
    metadata = json.dumps(
        custody_metadata, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    header = _HEADER_STRUCT.pack(
        JSCC_MAGIC, version, flags, side_dim, alphabet_size, n_symbols, len(metadata)
    )
    return header + metadata + payload


def parse_jscc_section(section_bytes: bytes) -> JSCCArchiveSection:
    """Parse a JSCC archive section.

    Args:
        section_bytes: header + payload.

    Returns:
        ``JSCCArchiveSection`` with typed manifest + raw payload.

    Raises:
        ValueError: on magic / version / length mismatches.
    """
    if len(section_bytes) < HEADER_BYTES:
        raise ValueError(
            f"section too short ({len(section_bytes)} < {HEADER_BYTES})"
        )
    (
        magic,
        version,
        flags,
        side_dim,
        alphabet_size,
        n_symbols,
        metadata_length,
    ) = _HEADER_STRUCT.unpack(section_bytes[:HEADER_BYTES])
    if magic != JSCC_MAGIC:
        raise ValueError(
            f"bad magic: expected {JSCC_MAGIC!r}, got {magic!r}"
        )
    if version != JSCC_FORMAT_VERSION:
        raise ValueError(
            f"unsupported JSCC version {version}; this build supports "
            f"version {JSCC_FORMAT_VERSION}"
        )
    if len(section_bytes) < HEADER_BYTES + metadata_length:
        raise ValueError(
            f"section too short for custody metadata "
            f"({len(section_bytes)} < {HEADER_BYTES + metadata_length})"
        )
    metadata_bytes = section_bytes[HEADER_BYTES : HEADER_BYTES + metadata_length]
    try:
        custody_metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid JSCC custody metadata JSON") from exc
    if not isinstance(custody_metadata, dict):
        raise ValueError("JSCC custody metadata must be a JSON object")
    for field, expected in (
        ("score_claim", False),
        ("promotion_eligible", False),
        ("ready_for_exact_eval_dispatch", False),
        ("proxy", True),
        ("proxy_only", True),
    ):
        if custody_metadata.get(field) is not expected:
            raise ValueError(f"JSCC custody metadata {field} must be {expected}")
    payload = section_bytes[HEADER_BYTES + metadata_length :]
    manifest = JSCCSectionManifest(
        magic=JSCC_MAGIC,
        version=int(version),
        flags=int(flags),
        side_dim=int(side_dim),
        alphabet_size=int(alphabet_size),
        n_symbols=int(n_symbols),
        custody_metadata_offset=HEADER_BYTES,
        custody_metadata_length=int(metadata_length),
        payload_offset=HEADER_BYTES + int(metadata_length),
        payload_length=len(payload),
        total_section_bytes=len(section_bytes),
        custody_metadata=dict(custody_metadata),
        evidence_grade=str(
            custody_metadata.get("evidence_grade", JSCC_PROXY_EVIDENCE_GRADE)
        ),
        proxy=True,
        proxy_only=True,
        score_claim=False,
        promotion_eligible=False,
        ready_for_exact_eval_dispatch=False,
    )
    return JSCCArchiveSection(manifest=manifest, payload=payload)
