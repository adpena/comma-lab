# SPDX-License-Identifier: MIT
"""Strict post-solve common exact-R receiver for DDM Probe B.

The v12 correction pool is a V11 post-solve archive, while the camera-resolution
realization profile and scorer-solved template bank were first counted in v15.
This module combines those already-counted *receiver* fields without importing
the v13 G1/PREDICT payload.  The nested v12 archive stays byte-for-byte intact;
the two realization payloads are explicit outer-ZIP members and therefore take
part in every equal-byte comparison.

No scorer, gradient, target label, or pixel-coordinate table is present at
decode.  Parse-back is strict and deterministic.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    import numpy as np

from tac.optimization.direct_description_carrier_compose import (
    ARCHIVE_SCHEMA_V3,
    EVIDENCE_AXIS,
    CarrierComposeReceiverV1,
    ReceiverRealizationProfileV1,
    ScorerSolvedTemplateBankV1,
    _decode_realization_profile,
    _encode_realization_profile,
    decode_scorer_solved_template_bank,
    encode_scorer_solved_template_bank,
    parse_carrier_compose_archive,
    receive_carrier_compose_archive,
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_entropy_priced_member import _sha256, _zip_stored
from tac.optimization.direct_description_minimizer import DirectDescriptionError

ARCHIVE_SCHEMA: Final = "ddm_v18_common_exact_r_master_archive.v1"
RECEIVER_SCHEMA: Final = "ddm_v18_common_exact_r_master_receiver.v1"
MANIFEST_MEMBER: Final = "manifest.json"
BASE_MEMBER: Final = "base/ddm_v12_postsolve.zip"
PROFILE_MEMBER: Final = "render/receiver_realization.ddrp"
TEMPLATE_MEMBER: Final = "render/scorer_solved_templates.ddst"
_MEMBER_ORDER: Final = (MANIFEST_MEMBER, BASE_MEMBER, PROFILE_MEMBER, TEMPLATE_MEMBER)


def _validate_postsolve_base(
    receiver: CarrierComposeReceiverV1,
    *,
    manifest_schema: str,
) -> None:
    if (
        receiver.worldsheet_g1_mask is not None
        or receiver.worldsheet_tracks
        or receiver.worldsheet_knots
        or receiver.lane_programs
        or receiver.lane_knots
    ):
        raise DirectDescriptionError("v18 common master refuses every PREDICT production")
    if receiver.realization_profile is not None or receiver.scorer_solved_templates is not None:
        raise DirectDescriptionError("v18 common master base already owns realization payloads")
    if not (
        manifest_schema == ARCHIVE_SCHEMA_V3
        or receiver.symbols
        or receiver.boundary_shearlets
        or receiver.island_shapes
    ):
        raise DirectDescriptionError("v18 common master requires a V11 post-solve correction base")


def compile_common_exact_r_master(
    base_archive: bytes,
    realization_profile: ReceiverRealizationProfileV1,
    template_bank: ScorerSolvedTemplateBankV1,
) -> bytes:
    """Count and wrap one v12 post-solve archive under the current exact-R receiver."""

    base = bytes(base_archive)
    base_members, _base_homes = parse_carrier_compose_archive(base)
    _validate_postsolve_base(
        receive_carrier_compose_archive(base),
        manifest_schema=str(json.loads(base_members[MANIFEST_MEMBER])["schema"]),
    )
    profile_payload = _encode_realization_profile(realization_profile)
    template_payload = encode_scorer_solved_template_bank(template_bank)
    if not profile_payload or not template_payload:
        raise DirectDescriptionError("v18 common master realization payloads must be nonempty")
    manifest = {
        "schema": ARCHIVE_SCHEMA,
        "base": {
            "member": BASE_MEMBER,
            "bytes": len(base),
            "sha256": _sha256(base),
            "vocabulary": "V11_POSTSOLVE_ONLY",
        },
        "realization_profile": {
            "member": PROFILE_MEMBER,
            "bytes": len(profile_payload),
            "sha256": _sha256(profile_payload),
        },
        "scorer_solved_templates": {
            "member": TEMPLATE_MEMBER,
            "bytes": len(template_payload),
            "sha256": _sha256(template_payload),
            "record_count": len(template_bank.templates),
        },
        "decode_boundary": "v12_postsolve_masks_then_camera_uint8_paint_then_evaluator_R",
        "predict_productions_present": False,
        "scorer_present_at_decode": False,
        "ground_truth_argmax_present_at_decode": False,
        "pixel_coordinate_table_present_at_decode": False,
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
    }
    members = {
        MANIFEST_MEMBER: rfc8785_canonicalize(manifest),
        BASE_MEMBER: base,
        PROFILE_MEMBER: profile_payload,
        TEMPLATE_MEMBER: template_payload,
    }
    first = _zip_stored(members)
    second = _zip_stored(members)
    if first != second:
        raise DirectDescriptionError("v18 common-master compiler is nondeterministic")
    parsed, _homes = parse_common_exact_r_master(first)
    if parsed != members or _zip_stored(parsed) != first:
        raise DirectDescriptionError("v18 common-master parse/re-encode differs")
    return first


def parse_common_exact_r_master(
    archive: bytes,
) -> tuple[dict[str, bytes], tuple[dict[str, Any], ...]]:
    try:
        with zipfile.ZipFile(io.BytesIO(archive), "r") as reader:
            infos = reader.infolist()
            if tuple(row.filename for row in infos) != _MEMBER_ORDER:
                raise DirectDescriptionError("v18 common-master member order differs")
            if any(
                row.is_dir()
                or row.compress_type != zipfile.ZIP_STORED
                or row.date_time != (1980, 1, 1, 0, 0, 0)
                or row.filename.startswith("/")
                or ".." in Path(row.filename).parts
                for row in infos
            ):
                raise DirectDescriptionError("v18 common-master ZIP metadata is noncanonical")
            members = {row.filename: reader.read(row) for row in infos}
            start_dir = reader.start_dir
    except DirectDescriptionError:
        raise
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise DirectDescriptionError("v18 common-master ZIP is malformed") from exc
    try:
        manifest = json.loads(members[MANIFEST_MEMBER])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DirectDescriptionError("v18 common-master manifest is malformed") from exc
    profile = _decode_realization_profile(members[PROFILE_MEMBER])
    bank = decode_scorer_solved_template_bank(members[TEMPLATE_MEMBER])
    invalid = (
        rfc8785_canonicalize(manifest) != members[MANIFEST_MEMBER]
        or manifest.get("schema") != ARCHIVE_SCHEMA
        or manifest.get("base")
        != {
            "member": BASE_MEMBER,
            "bytes": len(members[BASE_MEMBER]),
            "sha256": _sha256(members[BASE_MEMBER]),
            "vocabulary": "V11_POSTSOLVE_ONLY",
        }
        or manifest.get("realization_profile")
        != {
            "member": PROFILE_MEMBER,
            "bytes": len(members[PROFILE_MEMBER]),
            "sha256": _sha256(members[PROFILE_MEMBER]),
        }
        or bank is None
        or manifest.get("scorer_solved_templates")
        != {
            "member": TEMPLATE_MEMBER,
            "bytes": len(members[TEMPLATE_MEMBER]),
            "sha256": _sha256(members[TEMPLATE_MEMBER]),
            "record_count": 0 if bank is None else len(bank.templates),
        }
        or profile is None
        or manifest.get("decode_boundary")
        != "v12_postsolve_masks_then_camera_uint8_paint_then_evaluator_R"
        or manifest.get("predict_productions_present") is not False
        or manifest.get("scorer_present_at_decode") is not False
        or manifest.get("ground_truth_argmax_present_at_decode") is not False
        or manifest.get("pixel_coordinate_table_present_at_decode") is not False
        or manifest.get("score_claim") is not False
        or manifest.get("evidence_axis") != EVIDENCE_AXIS
    )
    if invalid:
        raise DirectDescriptionError("v18 common-master manifest custody differs")
    base_members, _base_homes = parse_carrier_compose_archive(members[BASE_MEMBER])
    _validate_postsolve_base(
        receive_carrier_compose_archive(members[BASE_MEMBER]),
        manifest_schema=str(json.loads(base_members[MANIFEST_MEMBER])["schema"]),
    )
    homes = []
    for info in infos:
        next_offset = (
            info.header_offset
            + 30
            + len(info.filename.encode("utf-8"))
            + len(info.extra)
            + info.compress_size
        )
        homes.append(
            {
                "name": info.filename,
                "payload_bytes": info.file_size,
                "zip_home_bytes": next_offset - info.header_offset,
                "sha256": _sha256(members[info.filename]),
            }
        )
    if sum(int(row["zip_home_bytes"]) for row in homes) + len(archive) - start_dir != len(archive):
        raise DirectDescriptionError("v18 common-master ZIP home accounting does not close")
    return members, tuple(homes)


@dataclass(frozen=True, slots=True)
class CommonExactRReceiverV1:
    archive: bytes
    base: CarrierComposeReceiverV1
    custody: Mapping[str, Any]

    @property
    def z(self) -> Any:
        return self.base.z

    @property
    def predictor(self) -> Any:
        return self.base.predictor

    @property
    def scorer_solved_templates(self) -> ScorerSolvedTemplateBankV1:
        bank = self.base.scorer_solved_templates
        if bank is None:  # guarded by parse; retained for type narrowing
            raise DirectDescriptionError("v18 common-master template bank disappeared")
        return bank

    def render_camera_pairs(self, pair_ids: Sequence[int]) -> np.ndarray:
        return self.base.render_camera_pairs(pair_ids)

    def template_camera_masks(self, pair_ids: Sequence[int], template: Any) -> np.ndarray:
        return self.base.template_camera_masks(pair_ids, template)


def receive_common_exact_r_master(archive: bytes) -> CommonExactRReceiverV1:
    members, homes = parse_common_exact_r_master(archive)
    profile = _decode_realization_profile(members[PROFILE_MEMBER])
    bank = decode_scorer_solved_template_bank(members[TEMPLATE_MEMBER])
    if profile is None or bank is None:
        raise DirectDescriptionError("v18 common-master realization payload disappeared")
    postsolve = receive_carrier_compose_archive(members[BASE_MEMBER])
    base = replace(
        postsolve,
        archive=bytes(archive),
        realization_profile=profile,
        scorer_solved_templates=bank,
        custody={},
    )
    custody = {
        "schema": RECEIVER_SCHEMA,
        "archive_bytes": len(archive),
        "archive_sha256": _sha256(archive),
        "base_archive_bytes": len(members[BASE_MEMBER]),
        "base_archive_sha256": _sha256(members[BASE_MEMBER]),
        "profile_bytes": len(members[PROFILE_MEMBER]),
        "profile_sha256": _sha256(members[PROFILE_MEMBER]),
        "template_bytes": len(members[TEMPLATE_MEMBER]),
        "template_sha256": _sha256(members[TEMPLATE_MEMBER]),
        "template_count": len(bank.templates),
        "postsolve_only": True,
        "predict_productions_present": False,
        "zip_homes_close": sum(int(row["zip_home_bytes"]) for row in homes) < len(archive),
        "scorer_weights_present": False,
        "ground_truth_argmax_present": False,
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
    }
    return CommonExactRReceiverV1(bytes(archive), base, custody)


def common_exact_r_byte_rows(archive: bytes) -> list[dict[str, Any]]:
    _members, homes = parse_common_exact_r_master(archive)
    strata = {
        MANIFEST_MEMBER: "outer_manifest",
        BASE_MEMBER: "v12_postsolve_correction_base",
        PROFILE_MEMBER: "receiver_realization_profile",
        TEMPLATE_MEMBER: "shared_scorer_solved_templates",
    }
    return [{"stratum": strata[row["name"]], **row} for row in homes]


__all__ = [
    "ARCHIVE_SCHEMA",
    "BASE_MEMBER",
    "CommonExactRReceiverV1",
    "common_exact_r_byte_rows",
    "compile_common_exact_r_master",
    "parse_common_exact_r_master",
    "receive_common_exact_r_master",
]
