"""Typed parser for PR79/qpose14-r55 SegAction packed payloads.

The public PR79 family stores a single ZIP member named ``p`` whose wire order
is:

``mask_br | model_br | seg_tile_actions_br | pose_q_br``

Some public revisions add a small ``P3`` header; older revisions infer the
split from byte lengths. This module keeps that inference centralized and
validated so search/build scripts do not hand-slice by unchecked constants.
"""
from __future__ import annotations

import hashlib
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

import brotli

from tac.submission_archive import validate_zip_member_infos, write_deterministic_zip_member

PR79_MASK_BR_BYTES = 219_472
PR79_MODEL_BR_CANDIDATE_BYTES = (
    55_756,
    55_757,
    55_914,
    56_034,
    56_093,
    56_221,
    57_031,
    57_053,
    57_757,
    60_880,
    61_147,
)
PR79_POSE_BR_CANDIDATE_BYTES = (898, 899)
PR79_MODEL_RAW_MAGICS = (b"QZS3", b"QZS2", b"QZS1", b"QZC1", b"QZC2", b"QZC3")
PR79_ACTION_RAW_MAGICS = (b"SG2", b"TG1")
PR79_POSE_RAW_MAGIC = b"QP1"


@dataclass(frozen=True)
class Pr79PayloadPart:
    name: str
    compressed: bytes
    raw: bytes
    raw_magic: bytes

    @property
    def compressed_bytes(self) -> int:
        return len(self.compressed)

    @property
    def raw_bytes(self) -> int:
        return len(self.raw)

    @property
    def compressed_sha256(self) -> str:
        return hashlib.sha256(self.compressed).hexdigest()

    @property
    def raw_sha256(self) -> str:
        return hashlib.sha256(self.raw).hexdigest()

    def summary(self) -> dict[str, object]:
        return {
            "compressed_bytes": self.compressed_bytes,
            "compressed_sha256": self.compressed_sha256,
            "name": self.name,
            "raw_bytes": self.raw_bytes,
            "raw_magic": self.raw_magic.hex(),
            "raw_sha256": self.raw_sha256,
        }


@dataclass(frozen=True)
class Pr79SegactionPayload:
    payload: bytes
    header: str
    mask: Pr79PayloadPart
    model: Pr79PayloadPart
    actions: Pr79PayloadPart
    pose: Pr79PayloadPart

    @property
    def payload_bytes(self) -> int:
        return len(self.payload)

    @property
    def payload_sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    @property
    def mask_br(self) -> bytes:
        return self.mask.compressed

    @property
    def model_br(self) -> bytes:
        return self.model.compressed

    @property
    def actions_br(self) -> bytes:
        return self.actions.compressed

    @property
    def pose_br(self) -> bytes:
        return self.pose.compressed

    def action_free_archive_bytes(self, *, zip_single_member_overhead: int = 100) -> int:
        """Return the public-script archive-byte estimate with actions removed.

        Public optimizers use this as a proxy rate term. It remains a proxy;
        exact ZIP bytes must come from the archive builder.
        """

        return len(self.payload) + zip_single_member_overhead - len(self.actions.compressed)

    def replace_actions(self, actions_br: bytes, *, preserve_header: bool = True) -> bytes:
        """Return a new ``p`` payload with a validated replacement action stream."""

        actions = _part("seg_tile_actions.br", actions_br, _validate_actions_raw)
        if preserve_header and self.header == "P3":
            return (
                b"P3"
                + struct.pack("<IHH", len(self.mask_br), len(self.model_br), len(actions.compressed))
                + self.mask_br
                + self.model_br
                + actions.compressed
                + self.pose_br
            )
        return self.mask_br + self.model_br + actions.compressed + self.pose_br

    def summary(self) -> dict[str, object]:
        return {
            "header": self.header,
            "payload_bytes": self.payload_bytes,
            "payload_sha256": self.payload_sha256,
            "parts": {
                "actions": self.actions.summary(),
                "mask": self.mask.summary(),
                "model": self.model.summary(),
                "pose": self.pose.summary(),
            },
            "slice_contract": {
                "actions_bytes": self.actions.compressed_bytes,
                "mask_bytes": self.mask.compressed_bytes,
                "model_bytes": self.model.compressed_bytes,
                "pose_bytes": self.pose.compressed_bytes,
            },
        }


def read_pr79_archive_payload(archive_path: str | Path) -> bytes:
    path = Path(archive_path)
    with zipfile.ZipFile(path, "r") as zf:
        validate_zip_member_infos(zf.infolist())
        names = zf.namelist()
        if names.count("p") != 1:
            raise ValueError(f"{path}: expected exactly one 'p' member, got {names.count('p')}")
        return zf.read("p")


def parse_pr79_archive(archive_path: str | Path) -> Pr79SegactionPayload:
    return parse_pr79_payload_bytes(read_pr79_archive_payload(archive_path))


def parse_pr79_payload_bytes(payload: bytes) -> Pr79SegactionPayload:
    if payload.startswith(b"P3"):
        return _parse_p3(payload)
    if payload.startswith(b"P2"):
        raise ValueError("P2 payload has no SegAction stream; PR79 parser requires actions")
    return _parse_legacy(payload)


def write_pr79_single_member_archive(path: str | Path, payload: bytes) -> dict[str, object]:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    parse_pr79_payload_bytes(payload)
    with zipfile.ZipFile(out, "w") as zf:
        write_deterministic_zip_member(zf, "p", payload, compress_type=zipfile.ZIP_STORED)
    return {
        "archive_bytes": out.stat().st_size,
        "archive_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _parse_p3(payload: bytes) -> Pr79SegactionPayload:
    if len(payload) < 10:
        raise ValueError("P3 payload too short")
    mask_len, model_len, actions_len = struct.unpack_from("<IHH", payload, 2)
    cursor = 10
    end_mask = cursor + mask_len
    end_model = end_mask + model_len
    end_actions = end_model + actions_len
    if end_actions >= len(payload):
        raise ValueError(
            "P3 length table leaves no pose payload: "
            f"payload={len(payload)} mask={mask_len} model={model_len} actions={actions_len}"
        )
    return _build_payload(
        payload,
        header="P3",
        mask_br=payload[cursor:end_mask],
        model_br=payload[end_mask:end_model],
        actions_br=payload[end_model:end_actions],
        pose_br=payload[end_actions:],
    )


def _parse_legacy(payload: bytes) -> Pr79SegactionPayload:
    matches: list[Pr79SegactionPayload] = []
    for model_len in PR79_MODEL_BR_CANDIDATE_BYTES:
        for pose_len in PR79_POSE_BR_CANDIDATE_BYTES:
            actions_len = len(payload) - PR79_MASK_BR_BYTES - model_len - pose_len
            if actions_len <= 0:
                continue
            try:
                matches.append(
                    _build_payload(
                        payload,
                        header="legacy",
                        mask_br=payload[:PR79_MASK_BR_BYTES],
                        model_br=payload[PR79_MASK_BR_BYTES : PR79_MASK_BR_BYTES + model_len],
                        actions_br=payload[
                            PR79_MASK_BR_BYTES + model_len : len(payload) - pose_len
                        ],
                        pose_br=payload[len(payload) - pose_len :],
                    )
                )
            except ValueError:
                continue
    if not matches:
        raise ValueError(f"no valid PR79 legacy split for payload bytes={len(payload)}")
    if len(matches) > 1:
        contracts = [
            (
                m.model.compressed_bytes,
                m.actions.compressed_bytes,
                m.pose.compressed_bytes,
                m.model.raw_magic,
                m.actions.raw_magic,
                m.pose.raw_magic,
            )
            for m in matches
        ]
        raise ValueError(f"ambiguous PR79 legacy split for payload bytes={len(payload)}: {contracts}")
    return matches[0]


def _build_payload(
    payload: bytes,
    *,
    header: str,
    mask_br: bytes,
    model_br: bytes,
    actions_br: bytes,
    pose_br: bytes,
) -> Pr79SegactionPayload:
    parts = {
        "mask": _part("mask.obu.br", mask_br, _validate_mask_raw),
        "model": _part("model.pt.br", model_br, _validate_model_raw),
        "actions": _part("seg_tile_actions.br", actions_br, _validate_actions_raw),
        "pose": _part("pose_q.br", pose_br, _validate_pose_raw),
    }
    return Pr79SegactionPayload(
        payload=payload,
        header=header,
        mask=parts["mask"],
        model=parts["model"],
        actions=parts["actions"],
        pose=parts["pose"],
    )


def _part(name: str, data: bytes, validator) -> Pr79PayloadPart:
    try:
        raw = brotli.decompress(data)
    except brotli.error as exc:
        raise ValueError(f"{name}: Brotli decompression failed") from exc
    magic = validator(raw)
    return Pr79PayloadPart(name=name, compressed=data, raw=raw, raw_magic=magic)


def _validate_mask_raw(raw: bytes) -> bytes:
    if not raw:
        raise ValueError("mask.obu.br: empty decompressed payload")
    return raw[:4]


def _validate_model_raw(raw: bytes) -> bytes:
    magic = raw[:4]
    if magic not in PR79_MODEL_RAW_MAGICS:
        raise ValueError(f"model.pt.br: unsupported decompressed magic {magic!r}")
    return magic


def _validate_actions_raw(raw: bytes) -> bytes:
    if raw.startswith(b"TG1"):
        if len(raw) < 5:
            raise ValueError("seg_tile_actions.br: TG1 header is truncated")
        body = raw[5:]
        if not body:
            raise ValueError("seg_tile_actions.br: TG1 body is empty")
        if body.startswith(b"SG2"):
            return raw[:3]
        if len(body) % 4 == 0 or len(body) % 5 == 0:
            return raw[:3]
        raise ValueError("seg_tile_actions.br: TG1 body is not an action record stream")
    if raw.startswith(b"SG2"):
        return raw[:3]
    if raw and (len(raw) % 4 == 0 or len(raw) % 5 == 0):
        return b"records"
    raise ValueError(f"seg_tile_actions.br: unsupported raw payload bytes={len(raw)}")


def _validate_pose_raw(raw: bytes) -> bytes:
    if raw.startswith(PR79_POSE_RAW_MAGIC):
        return raw[:3]
    if len(raw) % 12 == 0 and raw:
        return b"u16x6"
    raise ValueError(f"pose_q.br: unsupported raw payload bytes={len(raw)}")
