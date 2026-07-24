# SPDX-License-Identifier: MIT
"""Receiver-closed WS1 warm-start archives for DDM joint descent.

WS1 priced its two post-receiver transforms as the exact nested V19C receiver
bytes plus one self-framing payload.  This module makes that accounting an
executable archive contract: the archive is the canonical nested receiver
followed by exactly one parse-backable WS1 payload.  No scorer state, labels,
or target pixels are present at decode.

The flat suffix is intentional.  Both payload codecs are self-framing, so a
second ZIP manifest would add container bytes that the preregistered WS1 rows
did not price.  Parse-back still fails closed on the nested receiver, suffix
kind, payload geometry, and byte-identical re-emission.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np

from tac.optimization.ddm_hood_static_reassert import (
    expand_support_to_camera,
    reassert_frame1,
)
from tac.optimization.ddm_realized_flip_menu import (
    CAMERA_HW,
    SEG_HW,
    apply_local_statistics,
    apply_temporal_affine,
    decode_local_statistics,
    decode_temporal_affine,
)
from tac.optimization.direct_description_carrier_compose import (
    CLASS_ORDER,
    REALIZATION_PAINT_ORDER,
    ROLE_CLASS_IDS,
    CarrierComposeReceiverV1,
    receive_carrier_compose_archive,
)
from tac.optimization.direct_description_coupled_margin import (
    BASE_MEMBER as COUPLED_BASE_MEMBER,
)
from tac.optimization.direct_description_coupled_margin import (
    PROGRAM_MEMBER as COUPLED_PROGRAM_MEMBER,
)
from tac.optimization.direct_description_coupled_margin import (
    compile_coupled_margin_archive,
    decode_coupled_margin_program,
    parse_coupled_margin_archive,
)
from tac.optimization.direct_description_entropy_priced_member import _sha256
from tac.optimization.direct_description_minimizer import DirectDescriptionError
from tac.optimization.direct_description_preuint8_channel import (
    BASE_MEMBER as PREUINT8_BASE_MEMBER,
)
from tac.optimization.direct_description_preuint8_channel import (
    PROGRAM_MEMBER as PREUINT8_PROGRAM_MEMBER,
)
from tac.optimization.direct_description_preuint8_channel import (
    PreUint8Q8ReceiverV1,
    compile_preuint8_q8_archive,
    decode_preuint8_q8_program,
    parse_preuint8_q8_archive,
    receive_preuint8_q8_archive,
)

SCHEMA: Final = "ddm_ws1_receiver_closed_warm_start.v1"
W_SEG: Final = "W_seg"
W_JOINT: Final = "W_joint"
W_SEG_CANDIDATE_ID: Final = "temporal_affine_16knot_frame1_seglex96_hood_masked"
W_JOINT_CANDIDATE_ID: Final = "statistics_hard_analytic_composed_frame1"
W_SEG_PAYLOAD_BYTES: Final = 204
W_JOINT_PAYLOAD_BYTES: Final = 974
W_SEG_MAGIC: Final = b"DDMTA1\0\0"
W_JOINT_MAGIC: Final = b"DDMLS1\0\0"
# The WS1 receipt proves that the decoder-derived static-hood classifier
# selected MyCar/class 4.  The class is a transform-law field, not GT state.
W_SEG_HOOD_CLASS: Final = 4


def _palette(receiver: PreUint8Q8ReceiverV1) -> np.ndarray:
    profile = receiver.base.base.realization_profile
    if profile is None:
        raise DirectDescriptionError("WS1 nested receiver lacks a realization profile")
    role_for_class = {
        "Road": "Road",
        "Lane": "Lane",
        "Undrivable": "UndrivableBoundary",
        "Movable": "Movable",
        "MyCar": "MyCar",
    }
    return np.stack(
        [
            profile.colour_for(role_for_class[name])
            for name in CLASS_ORDER
        ],
        axis=0,
    ).astype(np.uint8)


def _semantic_cells(
    receiver: PreUint8Q8ReceiverV1,
    pair_ids: Sequence[int],
    camera: np.ndarray,
    palette: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay MENU1's scorer-free semantic ownership/fallback construction."""

    indexes = tuple(int(value) for value in pair_ids)
    carrier = receiver.base.base
    layer_by_role = {row.role: row for row in carrier.layers}
    semantic = np.full((len(indexes), *SEG_HW), -1, dtype=np.int16)
    owned = np.zeros_like(semantic, dtype=bool)
    for role in REALIZATION_PAINT_ORDER:
        layer = layer_by_role[role]
        for local, pair_id in enumerate(indexes):
            mask = carrier._mask_for_layer(
                layer,
                pair_id,
                replace_g1_movable=True,
            )
            semantic[local, mask] = ROLE_CLASS_IDS[role]
            owned[local, mask] = True
    row_centres = (
        (
            np.arange(SEG_HW[0], dtype=np.int64) * CAMERA_HW[0]
            + CAMERA_HW[0] // 2
        )
        // SEG_HW[0]
    ).clip(0, CAMERA_HW[0] - 1)
    col_centres = (
        (
            np.arange(SEG_HW[1], dtype=np.int64) * CAMERA_HW[1]
            + CAMERA_HW[1] // 2
        )
        // SEG_HW[1]
    ).clip(0, CAMERA_HW[1] - 1)
    sampled = camera[:, 1][
        :, row_centres[:, None], col_centres[None, :]
    ]
    distance = np.square(
        sampled[..., None, :].astype(np.int16)
        - palette[None, None, None].astype(np.int16)
    ).sum(axis=-1)
    fallback = np.argmin(distance, axis=-1).astype(np.int16)
    semantic[~owned] = fallback[~owned]
    if semantic.min() < 0 or semantic.max() >= len(CLASS_ORDER):
        raise DirectDescriptionError("WS1 semantic map escaped the five-class receiver")
    return semantic.astype(np.uint8), owned


def _geometry_statistics_camera(
    *,
    base_camera: np.ndarray,
    semantic: np.ndarray,
    owned: np.ndarray,
    palette: np.ndarray,
    statistics_payload: bytes,
) -> np.ndarray:
    """Replay MENU1's hard-interior/analytic-boundary statistics receiver."""

    # W_joint alone owns the SciPy-backed continuous-paint path.  W_seg/IC2
    # decode must not import it or declare SciPy as a runtime dependency.
    from tac.optimization.ddm_continuous_paint_ceiling import (
        render_analytic_coverage_blend,
        render_hard_camera_placement,
        resample_fields_at_pixel_centres,
        signed_distance_fields,
    )

    fields = signed_distance_fields(semantic)
    camera_fields = resample_fields_at_pixel_centres(fields)
    hard = render_hard_camera_placement(camera_fields, palette)
    analytic = render_analytic_coverage_blend(
        camera_fields,
        palette,
        softness=1.0,
    )
    ordered = np.partition(camera_fields, -2, axis=-1)
    interior = (ordered[..., -1] - ordered[..., -2]) >= 1.0
    geometry = analytic.copy()
    geometry[interior] = hard[interior]
    ys = (
        np.arange(CAMERA_HW[0]) * SEG_HW[0] // CAMERA_HW[0]
    ).clip(0, SEG_HW[0] - 1)
    xs = (
        np.arange(CAMERA_HW[1]) * SEG_HW[1] // CAMERA_HW[1]
    ).clip(0, SEG_HW[1] - 1)
    owner_camera = owned[:, ys[:, None], xs[None, :]]
    result = base_camera.copy()
    result[:, 1][owner_camera] = geometry[owner_camera]
    return apply_local_statistics(result, semantic, statistics_payload)


@dataclass(frozen=True, slots=True)
class WS1WarmStartArchiveV1:
    """Parsed archive plus the nested programs needed for J5 rewrapping."""

    archive: bytes
    candidate: Literal["W_seg", "W_joint"]
    candidate_id: str
    base_archive: bytes
    payload: bytes
    preuint8_program: Any
    coupled_program: Any
    carrier_archive: bytes
    custody: Mapping[str, Any]

    def exact_reemit(self) -> bytes:
        emitted = compile_ws1_warm_start_archive(
            self.base_archive,
            candidate=self.candidate,
            payload=self.payload,
        )
        if emitted != self.archive:
            raise DirectDescriptionError("WS1 parse/re-emit changed archive bytes")
        return emitted

    def rewrap_carrier(self, carrier_archive: bytes) -> bytes:
        """Preserve both nested programs and the WS1 suffix around new J5 bytes."""

        carrier = bytes(carrier_archive)
        receive_carrier_compose_archive(carrier, verify_member_effects=False)
        coupled = compile_coupled_margin_archive(
            carrier,
            self.coupled_program,
            verify_base_member_effects=False,
        )
        preuint8 = compile_preuint8_q8_archive(
            coupled,
            self.preuint8_program,
            verify_base_member_effects=False,
        )
        return compile_ws1_warm_start_archive(
            preuint8,
            candidate=self.candidate,
            payload=self.payload,
        )


@dataclass(frozen=True, slots=True)
class WS1WarmStartReceiverV1:
    archive: bytes
    parsed: WS1WarmStartArchiveV1
    base: PreUint8Q8ReceiverV1

    @property
    def carrier(self) -> CarrierComposeReceiverV1:
        return self.base.base.base

    @property
    def predictor(self) -> Any:
        return self.carrier.predictor

    @property
    def z(self) -> Any:
        return self.carrier.z

    @property
    def pose6_codes(self) -> np.ndarray:
        return self.carrier.pose6_codes

    @property
    def scorer_solved_templates(self) -> Any:
        return self.carrier.scorer_solved_templates

    @property
    def realization_profile(self) -> Any:
        return self.carrier.realization_profile

    @property
    def layers(self) -> Any:
        return self.carrier.layers

    @property
    def realization_static_rule_codes(self) -> Any:
        return self.carrier.realization_static_rule_codes

    @property
    def realization_static_rule_id(self) -> Any:
        return self.carrier.realization_static_rule_id

    def _mask_for_layer(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self.carrier._mask_for_layer(*args, **kwargs)

    def template_camera_masks(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self.carrier.template_camera_masks(*args, **kwargs)

    def render_camera_pairs(self, pair_ids: Sequence[int]) -> np.ndarray:
        indexes = tuple(int(value) for value in pair_ids)
        base_camera = self.base.render_camera_pairs(indexes)
        palette = _palette(self.base)
        semantic, owned = _semantic_cells(
            self.base,
            indexes,
            base_camera,
            palette,
        )
        if self.parsed.candidate == W_JOINT:
            return _geometry_statistics_camera(
                base_camera=base_camera,
                semantic=semantic,
                owned=owned,
                palette=palette,
                statistics_payload=self.parsed.payload,
            )
        transformed = apply_temporal_affine(
            base_camera,
            pair_ids=indexes,
            pair_count=600,
            payload=self.parsed.payload,
        )
        support = expand_support_to_camera(
            (semantic == W_SEG_HOOD_CLASS) & owned,
            batch_size=len(indexes),
            camera_hw=CAMERA_HW,
        )
        return reassert_frame1(
            winner_camera=transformed,
            base_camera=base_camera,
            camera_support=support,
        )


def _validate_payload(
    candidate: Literal["W_seg", "W_joint"],
    payload: bytes,
) -> str:
    if candidate == W_SEG:
        if len(payload) != W_SEG_PAYLOAD_BYTES or not payload.startswith(W_SEG_MAGIC):
            raise DirectDescriptionError("W_seg temporal payload custody differs")
        decode_temporal_affine(payload)
        return W_SEG_CANDIDATE_ID
    if candidate == W_JOINT:
        if len(payload) != W_JOINT_PAYLOAD_BYTES or not payload.startswith(W_JOINT_MAGIC):
            raise DirectDescriptionError("W_joint statistics payload custody differs")
        decode_local_statistics(payload)
        return W_JOINT_CANDIDATE_ID
    raise DirectDescriptionError(f"unknown WS1 warm-start candidate: {candidate!r}")


def compile_ws1_warm_start_archive(
    base_archive: bytes,
    *,
    candidate: Literal["W_seg", "W_joint"],
    payload: bytes,
) -> bytes:
    """Compile exact base bytes plus one canonical self-framing WS1 suffix."""

    base = bytes(base_archive)
    receive_preuint8_q8_archive(base, verify_base_member_effects=False)
    _validate_payload(candidate, bytes(payload))
    archive = base + bytes(payload)
    parsed = parse_ws1_warm_start_archive(archive)
    if parsed.candidate != candidate or parsed.base_archive != base or parsed.payload != payload:
        raise DirectDescriptionError("WS1 archive compiler parse-back differs")
    return archive


def parse_ws1_warm_start_archive(archive: bytes) -> WS1WarmStartArchiveV1:
    value = bytes(archive)
    matches: list[tuple[Literal["W_seg", "W_joint"], int]] = []
    if len(value) > W_SEG_PAYLOAD_BYTES and value[-W_SEG_PAYLOAD_BYTES:].startswith(W_SEG_MAGIC):
        matches.append((W_SEG, W_SEG_PAYLOAD_BYTES))
    if len(value) > W_JOINT_PAYLOAD_BYTES and value[-W_JOINT_PAYLOAD_BYTES:].startswith(W_JOINT_MAGIC):
        matches.append((W_JOINT, W_JOINT_PAYLOAD_BYTES))
    if len(matches) != 1:
        raise DirectDescriptionError("WS1 archive lacks one unambiguous self-framing suffix")
    candidate, payload_bytes = matches[0]
    base = value[:-payload_bytes]
    payload = value[-payload_bytes:]
    candidate_id = _validate_payload(candidate, payload)
    preuint8_members, preuint8_homes = parse_preuint8_q8_archive(base)
    preuint8_program = decode_preuint8_q8_program(
        preuint8_members[PREUINT8_PROGRAM_MEMBER]
    )
    coupled_archive = preuint8_members[PREUINT8_BASE_MEMBER]
    coupled_members, coupled_homes = parse_coupled_margin_archive(coupled_archive)
    coupled_program = decode_coupled_margin_program(
        coupled_members[COUPLED_PROGRAM_MEMBER]
    )
    carrier_archive = coupled_members[COUPLED_BASE_MEMBER]
    receive_carrier_compose_archive(carrier_archive, verify_member_effects=False)
    custody = {
        "schema": SCHEMA,
        "candidate": candidate,
        "candidate_id": candidate_id,
        "archive_bytes": len(value),
        "archive_sha256": _sha256(value),
        "base_archive_bytes": len(base),
        "base_archive_sha256": _sha256(base),
        "payload_bytes": len(payload),
        "payload_sha256": _sha256(payload),
        "preuint8_member_homes": list(preuint8_homes),
        "coupled_member_homes": list(coupled_homes),
        "carrier_archive_bytes": len(carrier_archive),
        "carrier_archive_sha256": _sha256(carrier_archive),
        "decoder_derived_hood_class": (
            W_SEG_HOOD_CLASS if candidate == W_SEG else None
        ),
        "scorer_weights_present": False,
        "ground_truth_argmax_present": False,
        "score_claim": False,
    }
    return WS1WarmStartArchiveV1(
        archive=value,
        candidate=candidate,
        candidate_id=candidate_id,
        base_archive=base,
        payload=payload,
        preuint8_program=preuint8_program,
        coupled_program=coupled_program,
        carrier_archive=carrier_archive,
        custody=custody,
    )


def receive_ws1_warm_start_archive(archive: bytes) -> WS1WarmStartReceiverV1:
    parsed = parse_ws1_warm_start_archive(archive)
    parsed.exact_reemit()
    return WS1WarmStartReceiverV1(
        archive=bytes(archive),
        parsed=parsed,
        base=receive_preuint8_q8_archive(
            parsed.base_archive,
            verify_base_member_effects=False,
        ),
    )


def receive_joint_descent_archive(
    archive: bytes,
    *,
    verify_member_effects: bool = True,
) -> CarrierComposeReceiverV1 | WS1WarmStartReceiverV1:
    """Receive either the sealed V15 carrier or one receiver-closed WS1 state."""

    try:
        return receive_carrier_compose_archive(
            archive,
            verify_member_effects=verify_member_effects,
        )
    except DirectDescriptionError as carrier_error:
        try:
            return receive_ws1_warm_start_archive(archive)
        except DirectDescriptionError as ws1_error:
            raise DirectDescriptionError(
                "joint-descent archive is neither a carrier nor WS1 state: "
                f"carrier={carrier_error}; ws1={ws1_error}"
            ) from ws1_error


__all__ = [
    "SCHEMA",
    "W_JOINT",
    "W_JOINT_CANDIDATE_ID",
    "W_SEG",
    "W_SEG_CANDIDATE_ID",
    "WS1WarmStartArchiveV1",
    "WS1WarmStartReceiverV1",
    "compile_ws1_warm_start_archive",
    "parse_ws1_warm_start_archive",
    "receive_joint_descent_archive",
    "receive_ws1_warm_start_archive",
]
