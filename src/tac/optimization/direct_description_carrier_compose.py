# SPDX-License-Identifier: MIT
"""Receiver-closed V9 carrier composition over a custodied DDM predictor.

The counted predictor already owns the five class-carrier payloads and the one
Pose6 stream.  This module adds only a strict outer grammar and the solved G2CS1
chart-symbol refinement surface.  Refinement changes Lane chart coefficients
before generic region-coherent rasterization; pixel-coordinate/value patches
are deliberately not part of this grammar.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from tac.optimization.direct_description_entropy_priced_member import (
    COMPOSED_ROLE_ORDER,
    ComposedStructuredMemberReceiverV1,
    StructuredRoleLayerV1,
    _sha256,
    _zip_stored,
    parse_structured_member_archive,
    receive_structured_member_archive,
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_entropy_streams import parse_entropy_chart_archive
from tac.optimization.direct_description_minimizer import SEED, DirectDescriptionError, _require_sha256
from tac.optimization.predictor_upgrade_xi_chart import (
    LaneCoefficientDelta,
    decode_lane_coefficient_deltas,
    encode_lane_coefficient_deltas,
)

CONFIG_SCHEMA: Final = "DirectDescriptionV9CarrierComposeConfigV1"
ARCHIVE_SCHEMA: Final = "direct_description_v9_carrier_compose_archive.v1"
RECEIVER_SCHEMA: Final = "direct_description_v9_carrier_compose_receiver.v1"
RESULT_SCHEMA: Final = "direct_description_v9_carrier_compose_receipt.v1"
MAGIC: Final = "DDV9C1"
EVIDENCE_AXIS: Final = "[macOS-CPU frozen-scorer advisory]"
CLASS_ORDER: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
ROLE_CLASS_IDS: Final = {
    "Road": 0,
    "Lane": 1,
    "UndrivableBoundary": 2,
    "Movable": 3,
    "MyCar": 4,
}
CORRECTION_MEMBER: Final = "correction/lane_chart_symbols.g2cs"


class DirectDescriptionV9CarrierComposeConfigV1(BaseModel):
    """Typed local-only measurement contract for one bridge window."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DirectDescriptionV9CarrierComposeConfigV1"] = Field(
        default=CONFIG_SCHEMA, alias="schema", serialization_alias="schema"
    )
    run_id: Literal["ddm_v9_carrier_compose_seed1234"] = "ddm_v9_carrier_compose_seed1234"
    seed: Literal[1234] = SEED
    pair_start: Literal[344, 448]
    pair_count: Literal[64, 256]
    v6_receipt_path: StrictStr
    v6_receipt_sha256: StrictStr
    predictor_archive_path: StrictStr
    predictor_archive_sha256: StrictStr
    upstream_root: StrictStr
    scorer_batch_size: Literal[16, 32] = 16
    scorer_threads: StrictInt = Field(ge=1, le=16)
    correction_policy: Literal["g2cs1_chart_coefficients_only_fisher_margin_ranked_no_pixel_residual"] = (
        "g2cs1_chart_coefficients_only_fisher_margin_ranked_no_pixel_residual"
    )
    correction_symbols: tuple[tuple[StrictInt, StrictInt, StrictInt, float], ...] = ()
    checkpoint_policy: Literal["atomic_preserve_build_then_measure"] = "atomic_preserve_build_then_measure"
    rate_authority: Literal["exact_len_receiver_closed_v9_zip"] = "exact_len_receiver_closed_v9_zip"
    class_order: tuple[StrictStr, ...] = CLASS_ORDER
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False
    d_seg_claim: Literal[False] = False
    d_pose_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DirectDescriptionV9CarrierComposeConfigV1:
        for name in ("v6_receipt_sha256", "predictor_archive_sha256"):
            _require_sha256(getattr(self, name), name)
        if (self.pair_start, self.pair_count) not in {(448, 64), (344, 256)}:
            raise ValueError("bridge windows must be exactly [448,512) or [344,600)")
        if not Path(self.upstream_root).is_absolute():
            raise ValueError("upstream_root must be absolute scorer custody")
        if self.class_order != CLASS_ORDER:
            raise ValueError(f"class_order must be canonical {CLASS_ORDER!r}")
        symbols = tuple(LaneCoefficientDelta(*row) for row in self.correction_symbols)
        if tuple((s.pair_index, s.line_index, s.coefficient_index) for s in symbols) != tuple(
            sorted({(s.pair_index, s.line_index, s.coefficient_index) for s in symbols})
        ):
            raise ValueError("correction symbols must be sorted and address-unique")
        return self

    def symbols(self) -> tuple[LaneCoefficientDelta, ...]:
        return tuple(LaneCoefficientDelta(*row) for row in self.correction_symbols)

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))


def _manifest_for(
    predictor_archive: bytes,
    predictor: ComposedStructuredMemberReceiverV1,
    correction_payload: bytes,
) -> dict[str, Any]:
    return {
        "schema": ARCHIVE_SCHEMA,
        "magic": MAGIC,
        "pair_count": predictor.z.n_pairs,
        "source_pair_start": predictor.source_pair_start,
        "class_order": list(CLASS_ORDER),
        "role_order": list(COMPOSED_ROLE_ORDER),
        "role_class_ids": ROLE_CLASS_IDS,
        "predictor": {"bytes": len(predictor_archive), "sha256": _sha256(predictor_archive)},
        "correction": {
            "member": CORRECTION_MEMBER if correction_payload else None,
            "bytes": len(correction_payload),
            "sha256": _sha256(correction_payload),
            "symbol_count": len(decode_lane_coefficient_deltas(correction_payload)),
            "policy": "G2CS1 counted Lane coefficient deltas before region-coherent rerasterization",
            "pixel_coordinate_or_rgb_patch_present": False,
            "admission": "nonempty only after caller-owned hard-oracle selection",
        },
        "merge_diff_correct": {
            "merge": "five nested semantic carrier masks in canonical role order",
            "diff": "G2CS1 addresses chart coefficients, never pixels",
            "correct": "generic Lane chart rerasterization then canonical layer merge",
        },
        "xi_pose6": {
            "home": "predictor.zip::chart.zip::ddm_chart_v3/05_pose6_pair_codes.bin",
            "ownership": "inherited sole counted Pose6 owner; no duplicate stream",
        },
        "scorer_weights_present": False,
        "ground_truth_argmax_present": False,
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
    }


def compile_carrier_compose_archive(
    predictor_archive: bytes,
    symbols: Sequence[LaneCoefficientDelta] = (),
) -> tuple[bytes, tuple[dict[str, Any], ...]]:
    """Compile a byte-canonical outer archive around the five-carrier predictor."""

    predictor = receive_structured_member_archive(predictor_archive)
    if not isinstance(predictor, ComposedStructuredMemberReceiverV1):
        raise DirectDescriptionError("v9 carrier composition requires the composed structured predictor")
    if [layer.role for layer in predictor.layers] != list(COMPOSED_ROLE_ORDER):
        raise DirectDescriptionError("predictor role order differs from the canonical merge order")
    if {layer.role: layer.class_id for layer in predictor.layers} != ROLE_CLASS_IDS:
        raise DirectDescriptionError("predictor role/class self-detection differs from canonical IDs")
    if any(
        symbol.pair_index < predictor.source_pair_start
        or symbol.pair_index >= predictor.source_pair_start + predictor.z.n_pairs
        for symbol in symbols
    ):
        raise DirectDescriptionError("G2CS1 symbol is outside the nested predictor source window")
    correction_payload = encode_lane_coefficient_deltas(tuple(symbols))
    members = {
        "manifest.json": rfc8785_canonicalize(_manifest_for(predictor_archive, predictor, correction_payload)),
        "predictor.zip": predictor_archive,
    }
    if correction_payload:
        members[CORRECTION_MEMBER] = correction_payload
    first = _zip_stored(members)
    second = _zip_stored(members)
    if first != second:
        raise DirectDescriptionError("v9 carrier compiler is nondeterministic")
    parsed, homes = parse_carrier_compose_archive(first)
    if parsed != members or _zip_stored(parsed) != first:
        raise DirectDescriptionError("v9 carrier archive parse/re-encode identity failed")
    return first, homes


def parse_carrier_compose_archive(archive: bytes) -> tuple[dict[str, bytes], tuple[dict[str, Any], ...]]:
    """Strictly parse the v9 outer ZIP and close exact unique-home accounting."""

    try:
        with zipfile.ZipFile(io.BytesIO(archive), "r") as reader:
            infos = reader.infolist()
            expected_prefix = ["manifest.json", "predictor.zip"]
            if [row.filename for row in infos[:2]] != expected_prefix or len(infos) not in (2, 3):
                raise DirectDescriptionError("v9 carrier archive member order/cardinality is invalid")
            if any(
                row.is_dir()
                or row.compress_type != zipfile.ZIP_STORED
                or row.date_time != (1980, 1, 1, 0, 0, 0)
                or row.filename.startswith("/")
                or ".." in Path(row.filename).parts
                for row in infos
            ):
                raise DirectDescriptionError("v9 carrier archive metadata is noncanonical")
            members = {row.filename: reader.read(row) for row in infos}
            start_dir = reader.start_dir
    except DirectDescriptionError:
        raise
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise DirectDescriptionError("v9 carrier archive ZIP is malformed") from exc
    try:
        manifest = json.loads(members["manifest.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DirectDescriptionError("v9 carrier manifest is malformed") from exc
    correction = members.get(CORRECTION_MEMBER, b"")
    invalid = (
        rfc8785_canonicalize(manifest) != members["manifest.json"]
        or manifest.get("schema") != ARCHIVE_SCHEMA
        or manifest.get("magic") != MAGIC
        or manifest.get("class_order") != list(CLASS_ORDER)
        or manifest.get("role_order") != list(COMPOSED_ROLE_ORDER)
        or manifest.get("role_class_ids") != ROLE_CLASS_IDS
        or manifest.get("predictor")
        != {"bytes": len(members["predictor.zip"]), "sha256": _sha256(members["predictor.zip"])}
        or manifest.get("correction", {}).get("member") != (CORRECTION_MEMBER if correction else None)
        or manifest.get("correction", {}).get("bytes") != len(correction)
        or manifest.get("correction", {}).get("sha256") != _sha256(correction)
        or set(members) != {"manifest.json", "predictor.zip", *([CORRECTION_MEMBER] if correction else [])}
    )
    if invalid:
        raise DirectDescriptionError("v9 carrier manifest identity/custody is invalid")
    symbols = decode_lane_coefficient_deltas(correction)
    if manifest["correction"]["symbol_count"] != len(symbols):
        raise DirectDescriptionError("v9 chart-symbol count differs after parse-back")
    predictor = receive_structured_member_archive(members["predictor.zip"])
    if not isinstance(predictor, ComposedStructuredMemberReceiverV1):
        raise DirectDescriptionError("v9 nested predictor is not composed")
    if predictor.z.n_pairs != manifest["pair_count"] or predictor.source_pair_start != manifest["source_pair_start"]:
        raise DirectDescriptionError("v9 nested predictor window differs from manifest")
    if _zip_stored(members) != archive:
        raise DirectDescriptionError("v9 carrier archive is not byte-canonical")
    homes: list[dict[str, Any]] = []
    for index, info in enumerate(infos):
        next_offset = infos[index + 1].header_offset if index + 1 < len(infos) else start_dir
        homes.append(
            {
                "name": info.filename,
                "payload_bytes": info.file_size,
                "zip_home_bytes": next_offset - info.header_offset,
                "payload_sha256": _sha256(members[info.filename]),
            }
        )
    homes.append(
        {
            "name": "__central_directory_and_eocd__",
            "payload_bytes": 0,
            "zip_home_bytes": len(archive) - sum(row["zip_home_bytes"] for row in homes),
        }
    )
    if sum(row["zip_home_bytes"] for row in homes) != len(archive):
        raise DirectDescriptionError("v9 carrier unique-home accounting does not close")
    return members, tuple(homes)


def _apply_chart_symbols(
    layers: Sequence[StructuredRoleLayerV1],
    symbols: Sequence[LaneCoefficientDelta],
) -> tuple[StructuredRoleLayerV1, ...]:
    copied = list(layers)
    lane_index = next((index for index, layer in enumerate(copied) if layer.role == "Lane"), None)
    if lane_index is None or copied[lane_index].lane_lines is None:
        raise DirectDescriptionError("v9 receiver lacks a decoded Lane chart")
    lines = [[np.asarray(value, dtype=np.float64).copy() for value in pair] for pair in copied[lane_index].lane_lines]
    for symbol in symbols:
        if symbol.pair_index >= len(lines) or symbol.line_index >= len(lines[symbol.pair_index]):
            raise DirectDescriptionError("G2CS1 address is absent from nested Lane chart")
        vector = lines[symbol.pair_index][symbol.line_index]
        if symbol.coefficient_index >= min(4, vector.size):
            raise DirectDescriptionError("G2CS1 correction must address a Lane centerline coefficient")
        vector[symbol.coefficient_index] += symbol.coefficient_delta
        if not np.isfinite(vector).all():
            raise DirectDescriptionError("G2CS1 application produced nonfinite Lane coefficients")
    copied[lane_index] = replace(
        copied[lane_index],
        lane_lines=tuple(tuple(value for value in pair) for pair in lines),
    )
    return tuple(copied)


@dataclass(frozen=True, slots=True)
class CarrierComposeReceiverV1:
    archive: bytes
    predictor: ComposedStructuredMemberReceiverV1
    layers: tuple[StructuredRoleLayerV1, ...]
    symbols: tuple[LaneCoefficientDelta, ...]
    custody: Mapping[str, Any]

    @property
    def z(self) -> Any:
        return self.predictor.z

    @property
    def pose6_codes(self) -> np.ndarray:
        return self.predictor.pose6_codes

    def render_pairs(self, pair_ids: Sequence[int]) -> np.ndarray:
        indexes = tuple(int(value) for value in pair_ids)
        if any(value < 0 or value >= self.z.n_pairs for value in indexes):
            raise DirectDescriptionError("v9 receiver pair ID is outside its local window")
        output = self.predictor.baseline.render_pairs(indexes)
        for layer in self.layers:
            for local_index, pair_id in enumerate(indexes):
                source_pair_id = self.predictor.source_pair_start + pair_id
                mask = layer.mask(
                    local_pair_id=pair_id,
                    source_pair_id=source_pair_id,
                    camera=self.predictor.camera,
                )
                output[local_index, 0, mask] = layer.paint_rgb_u8
                output[local_index, 1, mask] = layer.paint_rgb_u8
        return np.ascontiguousarray(output)


def receive_carrier_compose_archive(archive: bytes) -> CarrierComposeReceiverV1:
    members, homes = parse_carrier_compose_archive(archive)
    predictor = receive_structured_member_archive(members["predictor.zip"])
    if not isinstance(predictor, ComposedStructuredMemberReceiverV1):
        raise DirectDescriptionError("v9 nested predictor changed type after strict parse")
    symbols = decode_lane_coefficient_deltas(members.get(CORRECTION_MEMBER, b""))
    if any(
        symbol.pair_index < predictor.source_pair_start
        or symbol.pair_index >= predictor.source_pair_start + predictor.z.n_pairs
        for symbol in symbols
    ):
        raise DirectDescriptionError("G2CS1 symbol is outside the nested predictor source window")
    layers = _apply_chart_symbols(predictor.layers, symbols)
    first = CarrierComposeReceiverV1(
        archive=archive,
        predictor=predictor,
        layers=layers,
        symbols=symbols,
        custody={},
    )
    for symbol in symbols:
        local_pair_id = symbol.pair_index - predictor.source_pair_start
        isolated = CarrierComposeReceiverV1(
            archive=archive,
            predictor=predictor,
            layers=_apply_chart_symbols(predictor.layers, (symbol,)),
            symbols=(symbol,),
            custody={},
        )
        if np.array_equal(
            predictor.render_pairs((local_pair_id,)),
            isolated.render_pairs((local_pair_id,)),
        ):
            raise DirectDescriptionError("G2CS1 symbol is a receiver-output no-op")
    probes = tuple(sorted({0, predictor.z.n_pairs - 1}))
    a = first.render_pairs(probes)
    b = first.render_pairs(probes)
    if not np.array_equal(a, b):
        raise DirectDescriptionError("v9 carrier receiver replay is nondeterministic")
    custody = {
        "schema": RECEIVER_SCHEMA,
        "archive_bytes": len(archive),
        "archive_sha256": _sha256(archive),
        "member_homes": list(homes),
        "all_archive_bytes_have_one_home": sum(row["zip_home_bytes"] for row in homes) == len(archive),
        "all_five_roles_consumed": [layer.role for layer in layers] == list(COMPOSED_ROLE_ORDER),
        "chart_symbol_count": len(symbols),
        "chart_symbol_parse_reencode_identical": encode_lane_coefficient_deltas(symbols)
        == members.get(CORRECTION_MEMBER, b""),
        "region_coherent_chart_rerasterization": True,
        "pixel_coordinate_or_rgb_patch_present": False,
        "nested_pose6_owner_reused": True,
        "deterministic_probe_replay": True,
        "scorer_weights_present": False,
        "ground_truth_argmax_present": False,
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
    }
    return replace(first, custody=custody)


def recursive_carrier_byte_rows(archive: bytes) -> list[dict[str, Any]]:
    """Attribute nested semantic payload homes without double counting bytes."""

    members, outer_homes = parse_carrier_compose_archive(archive)
    predictor_members, predictor_homes = parse_structured_member_archive(members["predictor.zip"])
    home_by_name = {row["name"]: row for row in predictor_homes}
    chart = parse_entropy_chart_archive(predictor_members["chart.zip"])
    pose_row = next(row for row in chart.stream_byte_rows() if row["stream"] == "pose6_pair_codes")
    groups = {
        "Road": ("structure/road_pxq1_mask.br", "structure/road_events.lz", "structure/road_components.br"),
        "Lane": ("structure/lane_lbnd2.lz", "structure/lane_events.lz", "structure/lane_components.br"),
        "Undrivable": ("structure/undrivable_events.lz", "structure/undrivable_components.br"),
        "Movable": ("structure/movable_events.lz",),
        "MyCar": ("structure/mycar_static_hood.br",),
    }
    rows = [
        {
            "stratum": name,
            "nested_members": list(names),
            "nested_unique_home_bytes": sum(int(home_by_name[item]["zip_home_bytes"]) for item in names),
            "byte_authority": "exact nested ZIP home bytes; part of predictor.zip outer home",
        }
        for name, names in groups.items()
    ]
    rows.append(
        {
            "stratum": "xi/Pose6",
            "nested_members": ["chart.zip::ddm_chart_v3/05_pose6_pair_codes.bin"],
            "nested_unique_home_bytes": int(pose_row["unique_final_zip_home_bytes"]),
            "byte_authority": "exact nested entropy-chart ZIP home bytes; sole Pose6 owner",
        }
    )
    correction_home = next((row for row in outer_homes if row["name"] == CORRECTION_MEMBER), None)
    rows.append(
        {
            "stratum": "chart_symbol_refinement",
            "nested_members": [] if correction_home is None else [CORRECTION_MEMBER],
            "nested_unique_home_bytes": 0 if correction_home is None else int(correction_home["zip_home_bytes"]),
            "byte_authority": "exact outer ZIP home bytes",
        }
    )
    return rows


def prove_carrier_archive_fail_closed(archive: bytes) -> dict[str, Any]:
    """Sample every outer home: a mutation must refuse or alter decoded RGB."""

    baseline = receive_carrier_compose_archive(archive)
    probe_ids = tuple(sorted({0, baseline.z.n_pairs - 1}))
    digest = hashlib.sha256(baseline.render_pairs(probe_ids).tobytes()).hexdigest()
    _members, homes = parse_carrier_compose_archive(archive)
    positions: list[int] = []
    with zipfile.ZipFile(io.BytesIO(archive), "r") as reader:
        infos = reader.infolist()
        for info in infos:
            if info.file_size:
                positions.append(
                    info.header_offset + 30 + len(info.filename.encode()) + len(info.extra) + info.file_size // 2
                )
    refused = changed = 0
    for position in positions:
        altered = bytearray(archive)
        altered[position] ^= 1
        try:
            candidate = receive_carrier_compose_archive(bytes(altered))
        except (DirectDescriptionError, OSError, ValueError, zipfile.BadZipFile):
            refused += 1
            continue
        candidate_digest = hashlib.sha256(candidate.render_pairs(probe_ids).tobytes()).hexdigest()
        if candidate_digest == digest:
            raise DirectDescriptionError("sampled archive mutation was accepted as a receiver no-op")
        changed += 1
    return {
        "sampled_member_payload_homes": len(positions),
        "refused": refused,
        "changed_decode": changed,
        "all_samples_refused_or_changed_decode": refused + changed == len(positions),
        "unique_home_coverage_bytes": sum(row["zip_home_bytes"] for row in homes),
    }


__all__ = [
    "ARCHIVE_SCHEMA",
    "RESULT_SCHEMA",
    "CarrierComposeReceiverV1",
    "DirectDescriptionV9CarrierComposeConfigV1",
    "compile_carrier_compose_archive",
    "parse_carrier_compose_archive",
    "prove_carrier_archive_fail_closed",
    "receive_carrier_compose_archive",
    "recursive_carrier_byte_rows",
]
