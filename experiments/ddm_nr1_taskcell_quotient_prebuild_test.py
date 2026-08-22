from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import tac.optimization.nr1_taskcell_quotient as nr1


def fixture() -> tuple[np.ndarray, dict[nr1.Section, bytes], bytes, np.ndarray]:
    tokens = np.zeros((4, 8, 8), dtype=np.uint8)
    tokens[:, 0:4, 4:8] = 1
    tokens[1:, 4:8, 0:4] = 2
    tokens[2:, 4:8, 4:8] = 3
    tokens[1, 1, 1] = 2
    tokens[2, 1, 2] = 3
    tokens[3, 1, 1] = 4
    raw, base = nr1.encode_raw_sections(
        tokens,
        tile_height=4,
        tile_width=4,
        codebook_size=6,
    )
    packet = nr1.build_packet(raw, *tokens.shape)
    return tokens, raw, packet, base


def rebuild_with_mutation(
    raw: dict[nr1.Section, bytes],
    section: nr1.Section,
    mutate: callable,
    shape: tuple[int, int, int],
) -> bytes:
    changed = dict(raw)
    changed[section] = mutate(bytearray(raw[section]))
    return nr1.build_packet(changed, *shape)


def test_canonical_packet_exact_once_and_physical_ownership() -> None:
    tokens, raw, packet, _ = fixture()
    parsed = nr1.parse_packet(packet)
    assert tuple(section.name for section in parsed.sections) == tuple(nr1.Section)
    assert nr1.build_packet(
        {section.name: section.raw for section in parsed.sections},
        *tokens.shape,
        selected={
            section.name: nr1.CodedCandidate(section.coder, section.coded)
            for section in parsed.sections
        },
    ) == packet
    decoded = nr1.decode_packet(packet)
    decoded.trace.require_exact_once()
    assert decoded.tokens.shape == tokens.shape
    assert decoded.tokens.dtype == np.uint8
    assert np.count_nonzero(decoded.tokens != tokens) > 0
    attribution = nr1.physical_attribution(packet)
    assert sum(end - start for start, end in attribution.values()) == len(packet)
    assert list(attribution) == list(nr1.Section)
    assert tuple(raw) == tuple(nr1.Section)


def test_absent_and_explicit_inactive_are_exact_base_identity() -> None:
    tokens, _, _, _ = fixture()
    base = bytes(range(255))
    explicit = nr1.build_explicit_inactive(base, *tokens.shape)
    assert nr1.receive_inactive_or_base(None, base, *tokens.shape) is base
    assert nr1.receive_inactive_or_base(explicit, base, *tokens.shape) is base
    with pytest.raises(nr1.NR1FormatError):
        nr1.receive_inactive_or_base(explicit, base + b"x", *tokens.shape)
    with pytest.raises(nr1.NR1FormatError):
        nr1.parse_packet(explicit)


def test_active_empty_unknown_reordered_truncated_and_trailing_refuse() -> None:
    tokens, raw, packet, _ = fixture()
    empty = dict(raw)
    empty[nr1.Section.QEVENT] = b""
    with pytest.raises(nr1.NR1FormatError):
        nr1.build_packet(empty, *tokens.shape)

    corruptions = []
    bad_magic = bytearray(packet)
    bad_magic[0] ^= 1
    corruptions.append(bytes(bad_magic))
    bad_version = bytearray(packet)
    bad_version[4] = nr1.VERSION + 1
    corruptions.append(bytes(bad_version))
    bad_mode = bytearray(packet)
    bad_mode[5] = 9
    corruptions.append(bytes(bad_mode))
    unknown = bytearray(packet)
    unknown[nr1._OUTER.size : nr1._OUTER.size + 8] = b"UNKNOWN\x00"
    corruptions.append(bytes(unknown))
    reordered = bytearray(packet)
    reordered[nr1._OUTER.size : nr1._OUTER.size + 8] = b"QCTX\x00\x00\x00\x00"
    corruptions.append(bytes(reordered))
    unknown_coder = bytearray(packet)
    unknown_coder[nr1._OUTER.size + 8] = 255
    corruptions.append(bytes(unknown_coder))
    corruptions.append(packet + b"trailing")
    for corrupted in corruptions:
        with pytest.raises(nr1.NR1FormatError):
            nr1.parse_packet(corrupted)

    parsed = nr1.parse_packet(packet)
    cuts = [nr1._OUTER.size - 1]
    for section in parsed.sections:
        cuts.extend([section.header_start + nr1._SECTION.size - 1, section.payload_end - 1])
    for cut in cuts:
        with pytest.raises(nr1.NR1FormatError):
            nr1.parse_packet(packet[:cut])


def test_parse_valid_mutation_makes_every_surface_live() -> None:
    tokens, raw, packet, base = fixture()
    reference = nr1.decode_packet(packet).tokens

    def mutate_qparam(payload: bytearray) -> bytes:
        payload[nr1._QPARAM.size] = 1
        return bytes(payload)

    def mutate_qctx(payload: bytearray) -> bytes:
        payload[nr1._QCTX.size] = (payload[nr1._QCTX.size] + 1) % 5
        return bytes(payload)

    def mutate_qpair(payload: bytearray) -> bytes:
        payload[nr1._QPAIR.size] = 6
        return bytes(payload)

    def mutate_qevent(payload: bytearray) -> bytes:
        old = payload[-1]
        delta, _ = nr1._decode_uleb(payload, nr1._QEVENT.size)
        base_value = int(base.reshape(-1)[delta])
        payload[-1] = next(value for value in range(5) if value not in {old, base_value})
        return bytes(payload)

    for section, mutator in (
        (nr1.Section.QPARAM, mutate_qparam),
        (nr1.Section.QCTX, mutate_qctx),
        (nr1.Section.QPAIR, mutate_qpair),
        (nr1.Section.QEVENT, mutate_qevent),
    ):
        candidate = rebuild_with_mutation(raw, section, mutator, tokens.shape)
        try:
            changed = nr1.decode_packet(candidate).tokens
        except nr1.NR1FormatError:
            continue
        assert not np.array_equal(changed, reference), section


@pytest.mark.parametrize(
    "consumer",
    ["_consume_qparam", "_consume_qctx", "_consume_qpair", "_consume_qevent"],
)
def test_inert_consumer_replacement_refuses(monkeypatch: pytest.MonkeyPatch, consumer: str) -> None:
    _, _, packet, _ = fixture()
    monkeypatch.setattr(nr1, consumer, lambda *args, **kwargs: None)
    with pytest.raises((nr1.NR1FormatError, TypeError)):
        nr1.decode_packet(packet)


def test_deterministic_real_coder_race_and_fresh_process(tmp_path: Path) -> None:
    tokens, raw, packet, _ = fixture()
    for section_raw in raw.values():
        first = nr1.coder_candidates(section_raw)
        second = nr1.coder_candidates(section_raw)
        assert {candidate.coder for candidate in first} == set(nr1.Coder)
        assert [(row.coder, row.payload) for row in first] == [
            (row.coder, row.payload) for row in second
        ]
    assert nr1.build_packet(raw, *tokens.shape) == packet
    packet_path = tmp_path / "p"
    packet_path.write_bytes(packet)
    staged_module = tmp_path / "nr1_taskcell_quotient.py"
    staged_module.write_bytes(Path(nr1.__file__).read_bytes())
    code = """
import hashlib
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[2])
from nr1_taskcell_quotient import decode_packet
p = Path(__import__('sys').argv[1]).read_bytes()
print(hashlib.sha256(decode_packet(p).tokens.tobytes()).hexdigest())
"""
    environment = {
        "PATH": os.environ["PATH"],
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    output = subprocess.check_output(
        [sys.executable, "-I", "-c", code, str(packet_path), str(tmp_path)],
        env=environment,
        text=True,
    ).strip()
    expected = hashlib.sha256(nr1.decode_packet(packet).tokens.tobytes()).hexdigest()
    assert output == expected
