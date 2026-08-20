# SPDX-License-Identifier: MIT

from __future__ import annotations

import io
import zipfile

import pytest

from tac.optimization.arith_selfcomp_rate_coders import (
    encode_bellard_class_mixing,
    encode_g4_decoder_context,
    encode_willems_ctw,
)
from tac.optimization.ddm_wf7_seven_home_stream_waterfill import (
    CODEC_NAMES,
    WF7Error,
    restore_candidate,
    serialize_candidate,
)


def _state() -> bytes:
    output = io.BytesIO()
    names = (
        "manifest.json",
        "predictor.zip",
        "predict/movable_polygon_worldsheet.g1s",
        "render/receiver_realization.ddrp",
        "render/scorer_solved_templates.ddst",
        "predict/lane_periodic_programs.ddlp",
    )
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for index, name in enumerate(names):
            archive.writestr(name, bytes((index + 1,)) * (index + 3))
    return output.getvalue()


def _homes(state: bytes) -> tuple[bytes, ...]:
    with zipfile.ZipFile(io.BytesIO(state), "r") as archive:
        infos = archive.infolist()
        stops = [info.header_offset for info in infos[1:]] + [archive.start_dir]
        rows = tuple(state[info.header_offset : stop] for info, stop in zip(infos, stops, strict=True))
        return (*rows, state[archive.start_dir :])


def test_mixed_candidate_round_trips_exact_state() -> None:
    state = _state()
    homes = _homes(state)
    codecs = (
        "RAW_CURRENT",
        "G4_FREE_DECODER_CONTEXT",
        "WILLEMS_CTW",
        "BELLARD_CLASS_MIXING",
        "RAW_CURRENT",
        "G4_FREE_DECODER_CONTEXT",
        "BELLARD_CLASS_MIXING",
    )
    encoders = {
        "RAW_CURRENT": bytes,
        "G4_FREE_DECODER_CONTEXT": encode_g4_decoder_context,
        "WILLEMS_CTW": encode_willems_ctw,
        "BELLARD_CLASS_MIXING": encode_bellard_class_mixing,
    }
    frames = tuple(encoders[codec](home) for codec, home in zip(codecs, homes, strict=True))
    candidate = serialize_candidate(codecs, frames)
    restored, receipt = restore_candidate(candidate, require_sealed_identity=False)
    assert restored == state
    assert receipt["exact_parseback"] is True
    assert receipt["codec_names"] == list(codecs)
    assert receipt["directory_bytes"] < 40


def test_candidate_refuses_noncanonical_or_unsupported_directory() -> None:
    state = _state()
    homes = _homes(state)
    candidate = serialize_candidate(("RAW_CURRENT",) * 7, homes)
    with pytest.raises(WF7Error, match="magic/version"):
        restore_candidate(b"FAIL" + candidate[4:], require_sealed_identity=False)
    corrupted = bytearray(candidate)
    corrupted[7] |= 0xE0
    with pytest.raises(WF7Error, match=r"reserved bits|unsupported codec"):
        restore_candidate(bytes(corrupted), require_sealed_identity=False)


def test_codec_registry_fits_three_bit_directory() -> None:
    assert len(CODEC_NAMES) == 5
    assert len(CODEC_NAMES) <= 2**3
