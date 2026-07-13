# SPDX-License-Identifier: MIT
"""JRD prefix adaptation for PR110-lineage HNeRV decoder coefficients.

This module keeps the existing FP11/CTXR/FECa/DQS1 grammar intact.  It decodes
the 28 signed int8 decoder tensors with the submission's own codec, replaces
exactly one tensor in q-space, re-encodes the decoder section with the same
``codec_ctx`` range coder, and splices every non-decoder section verbatim.

The affine uint8 latents, fp16 scales, selector, sidecar, and DQS1 tail are not
zero-centred signed-int8 coefficient tensors.  They are therefore deliberately
outside this adapter rather than being silently reinterpreted to fit the JRD
oracle's input contract.
"""

from __future__ import annotations

import dataclasses
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tac import click_polish


@dataclasses.dataclass(frozen=True, slots=True)
class Pr110DecoderCoefficientSection:
    """One signed-int8 decoder tensor in PR110 storage order."""

    name: str
    storage_position: int
    storage_index: int
    stream_index: int
    count: int
    tensor_shape: tuple[int, ...]
    storage_shape: tuple[int, ...]
    byte_map: str


def _zigzag_encode_i8(values: np.ndarray) -> np.ndarray:
    q = values.astype(np.int16, copy=False)
    encoded = np.where(q >= 0, 2 * q, -2 * q - 1)
    return encoded.astype(np.uint8)


def encode_mapped_i8(values: np.ndarray | Sequence[int], byte_map: str) -> np.ndarray:
    """Inverse of the submission runtime's ``decode_mapped_u8`` function."""

    q = np.asarray(values)
    if q.dtype != np.int8:
        raise TypeError(f"mapped decoder coefficients must be int8; got {q.dtype}")
    if byte_map == "zig":
        return _zigzag_encode_i8(q)
    if byte_map == "negzig":
        negated = (-q.astype(np.int16)).astype(np.int8)
        return _zigzag_encode_i8(negated)
    if byte_map == "off":
        return (q.astype(np.int16) + 128).astype(np.uint8)
    if byte_map == "twos":
        return q.view(np.uint8).copy()
    raise ValueError(f"unknown decoder byte map: {byte_map!r}")


def _stream_index(storage_position: int, stream_ends: Sequence[int]) -> int:
    for index, end in enumerate(stream_ends):
        if storage_position < int(end):
            return index
    raise ValueError(f"storage position {storage_position} is outside decoder streams")


class Pr110CoefficientPacket:
    """Byte-closed decoder-coefficient view of one PR110-lineage archive."""

    def __init__(self, archive_path: str | Path, submission_dir: str | Path):
        self.archive_path = Path(archive_path)
        self.submission_dir = Path(submission_dir)
        self.packet = click_polish.FrozenPacket.parse(self.archive_path, self.submission_dir)
        self.ns = self.packet.ns
        self.raw_streams = self.ns.codec_ctx.decode_decoder_section(self.packet.dec_sec)
        self.stored_weights, self.scale_bytes = self.ns.codec_ctx._split_streams_to_tensors(
            self.raw_streams
        )
        self.sections = self._derive_sections()
        self._validate_runtime_inverse()

    def _derive_sections(self) -> tuple[Pr110DecoderCoefficientSection, ...]:
        probe = self.ns.inflate.HNeRVDecoder(
            latent_dim=self.ns.codec.LATENT_DIM,
            base_channels=self.ns.codec.BASE_CHANNELS,
            eval_size=tuple(self.ns.codec.EVAL_SIZE),
        )
        state_items = list(probe.state_dict().items())
        schema = tuple(self.ns.codec_ctx.TENSOR_SCHEMA)
        order = tuple(self.ns.codec.DECODER_STORAGE_ORDER)
        stream_ends = tuple(self.ns.codec_ctx.STREAM_ENDS)
        if len(schema) != len(order) or len(schema) != len(self.stored_weights):
            raise ValueError("runtime decoder schema/order/weight counts disagree")

        sections: list[Pr110DecoderCoefficientSection] = []
        for position, ((schema_index, schema_count), storage_index) in enumerate(
            zip(schema, order, strict=True)
        ):
            if int(schema_index) != int(storage_index):
                raise ValueError(
                    f"runtime tensor schema disagrees at position {position}: "
                    f"{schema_index} != {storage_index}"
                )
            name, tensor = state_items[int(storage_index)]
            shape = tuple(int(dim) for dim in tensor.shape)
            if int(tensor.numel()) != int(schema_count):
                raise ValueError(f"runtime tensor count disagrees for {name}")
            storage_perm = self.ns.codec.CONV4_STORAGE_PERMS.get(int(storage_index))
            storage_shape = (
                tuple(shape[axis] for axis in storage_perm)
                if storage_perm is not None
                else shape
            )
            sections.append(
                Pr110DecoderCoefficientSection(
                    name=str(name),
                    storage_position=position,
                    storage_index=int(storage_index),
                    stream_index=_stream_index(position, stream_ends),
                    count=int(schema_count),
                    tensor_shape=shape,
                    storage_shape=storage_shape,
                    byte_map=str(
                        self.ns.codec.DECODER_BYTE_MAPS.get(int(storage_index), "zig")
                    ),
                )
            )
        if len({section.name for section in sections}) != len(sections):
            raise ValueError("runtime decoder state contains duplicate tensor names")
        return tuple(sections)

    def _validate_runtime_inverse(self) -> None:
        for section in self.sections:
            stored = self.stored_weights[section.storage_position]
            decoded = self.ns.codec.decode_mapped_u8(stored, section.byte_map)
            encoded = encode_mapped_i8(decoded, section.byte_map)
            if not np.array_equal(encoded, stored):
                raise ValueError(f"byte-map inverse is not exact for {section.name}")

    def section_by_name(self, name: str) -> Pr110DecoderCoefficientSection:
        matches = [section for section in self.sections if section.name == name]
        if len(matches) != 1:
            raise KeyError(f"unknown or ambiguous decoder section {name!r}")
        return matches[0]

    def read_section(self, section: Pr110DecoderCoefficientSection) -> np.ndarray:
        self._validate_section_identity(section)
        stored = self.stored_weights[section.storage_position]
        decoded = self.ns.codec.decode_mapped_u8(stored, section.byte_map)
        return np.ascontiguousarray(decoded.reshape(section.storage_shape))

    def repack_archive(
        self,
        section: Pr110DecoderCoefficientSection,
        replacement: np.ndarray | Sequence[int],
    ) -> bytes:
        """Replace one q-tensor and return a deterministic byte-closed ZIP."""

        return self.repack_archive_replacements({section.name: replacement})

    def repack_archive_replacements(
        self,
        replacements: dict[str, np.ndarray | Sequence[int]],
    ) -> bytes:
        """Replace named q-tensors and return one deterministic byte-closed ZIP."""

        if not replacements:
            raise ValueError("at least one decoder coefficient replacement is required")

        weights = [weight.copy() for weight in self.stored_weights]
        for name, replacement in replacements.items():
            section = self.section_by_name(name)
            q = np.asarray(replacement)
            if q.dtype != np.int8:
                raise TypeError(f"replacement for {name} must be int8; got {q.dtype}")
            if tuple(q.shape) != section.storage_shape:
                raise ValueError(
                    f"replacement shape {tuple(q.shape)} != {section.storage_shape} "
                    f"for {section.name}"
                )
            weights[section.storage_position] = encode_mapped_i8(
                np.ascontiguousarray(q).reshape(-1), section.byte_map
            )
        raw_streams = self.ns.codec_ctx._tensors_to_streams(weights, self.scale_bytes)
        decoder_section = self.ns.codec_ctx.encode_decoder_section(raw_streams)
        member = self._repack_member(decoder_section)
        return click_polish._deterministic_zip(member)

    def no_op_archive(self) -> bytes:
        """Re-encode the unchanged decoder and prove complete archive identity."""

        decoder_section = self.ns.codec_ctx.encode_decoder_section(self.raw_streams)
        return click_polish._deterministic_zip(self._repack_member(decoder_section))

    def _repack_member(self, decoder_section: bytes) -> bytes:
        latent_section = self.packet._original_lat_sec()
        sidecar = self.packet.sidecar
        ctxr = (
            b"CTXR"
            + bytes([self.packet.ctxr_version])
            + len(decoder_section).to_bytes(3, "little")
            + len(latent_section).to_bytes(3, "little")
            + len(sidecar).to_bytes(3, "little")
            + decoder_section
            + latent_section
            + sidecar
        )
        return (
            b"FP11"
            + struct.pack("<I", len(ctxr))
            + ctxr
            + struct.pack("<H", len(self.packet.sel_bytes))
            + self.packet.sel_bytes
            + self.packet.dqs1_tail
        )

    def _validate_section_identity(self, section: Pr110DecoderCoefficientSection) -> None:
        if not isinstance(section, Pr110DecoderCoefficientSection):
            raise TypeError("section must be a Pr110DecoderCoefficientSection")
        canonical = self.section_by_name(section.name)
        if canonical != section:
            raise ValueError(f"section metadata changed for {section.name}")

    def custody(self) -> dict[str, Any]:
        """Return the exact grammar surface changed and deliberately preserved."""

        return {
            "archive_path": str(self.archive_path),
            "decoder_section_count": len(self.sections),
            "decoder_signed_int8_coefficients": sum(s.count for s in self.sections),
            "eligible_surface": "28 decoder signed-int8 q-tensors",
            "semantic_values_preserved_exactly_through_decoder_reencode": [
                "28 fp16 per-tensor scale values",
            ],
            "archive_sections_spliced_verbatim": [
                "affine uint8 latent codes and fp16 min/scale header",
                "latent sidecar",
                "FECa selector",
                "DQS1 tail",
            ],
        }


__all__ = [
    "Pr110CoefficientPacket",
    "Pr110DecoderCoefficientSection",
    "encode_mapped_i8",
]
