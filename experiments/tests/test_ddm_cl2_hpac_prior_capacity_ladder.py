"""Guards for the ddm_cl2 pricing runner's pure pieces.

The exact prices are MEASURED by the runner on the shipped fs2 tree; nothing here
restates them.  These pin (a) the RX1 header rewrite that carries a new HPAC section,
(b) the two-pin inflate.py patch (jf2 #1237: a half-updated pin), and (c) cl1's
adjacent-rung break-even arithmetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments import ddm_cl2_hpac_prior_capacity_ladder as cl2
from experiments import ddm_jg2_tail_reencode as jg2


def _member(hpac: bytes, semantic: bytes, carrier: bytes, tail: bytes) -> bytes:
    header = jg2.RX1_HEADER.pack(b"RX1M", 1, 2, 3, 0, len(hpac), len(semantic), len(carrier))
    return header + hpac + semantic + carrier + tail


def test_replace_hpac_section_rewrites_only_the_hpac_length_and_bytes() -> None:
    member = _member(b"OLDHPAC", b"SEM", b"CARR", b"TAIL")
    replaced = cl2.replace_hpac_section(member, b"NEWMODEL!!")
    sections = jg2.split_member(replaced)
    assert sections["hpac"] == b"NEWMODEL!!"
    assert sections["semantic"] == b"SEM"
    assert sections["carrier"] == b"CARR"
    assert sections["tail"] == b"TAIL"
    magic, version, a, b, reserved, hpac, semantic, carrier = jg2.RX1_HEADER.unpack(sections["header"])
    assert (magic, version, a, b, reserved) == (b"RX1M", 1, 2, 3, 0)
    assert (hpac, semantic, carrier) == (10, 3, 4)


def test_replace_hpac_section_refuses_a_section_the_header_cannot_carry() -> None:
    member = _member(b"OLD", b"", b"", b"")
    with pytest.raises(cl2.Cl2Error, match="uint16"):
        cl2.replace_hpac_section(member, b"x" * 65_536)


def test_patch_inflate_pins_rewrites_both_pins_exactly_once(tmp_path: Path) -> None:
    text = f'ARCHIVE_SHA256 = "{cl2.FS2_ARCHIVE_SHA256}"\nARCHIVE_BYTES = {cl2.FS2_ARCHIVE_BYTES:_}\n'
    (tmp_path / "inflate.py").write_text(text, encoding="utf-8")
    fact = cl2.patch_inflate_pins(tmp_path, "ab" * 32, 123_456)
    patched = (tmp_path / "inflate.py").read_text(encoding="utf-8")
    assert patched.count('ARCHIVE_SHA256 = "' + "ab" * 32 + '"') == 1
    assert patched.count("ARCHIVE_BYTES = 123_456") == 1
    assert cl2.FS2_ARCHIVE_SHA256 not in patched
    assert fact["bytes"] == len(patched.encode("utf-8"))
    # Idempotent: a second call on the patched copy is a no-op, not a refusal.
    cl2.patch_inflate_pins(tmp_path, "ab" * 32, 123_456)


def test_patch_inflate_pins_refuses_an_ambiguous_or_foreign_pin(tmp_path: Path) -> None:
    (tmp_path / "inflate.py").write_text('ARCHIVE_SHA256 = "deadbeef"\nARCHIVE_BYTES = 1\n', encoding="utf-8")
    with pytest.raises(cl2.Cl2Error, match="absent or ambiguous"):
        cl2.patch_inflate_pins(tmp_path, "ab" * 32, 2)


def test_adjacent_slope_is_cl1_break_even() -> None:
    left = {"model_packed_bytes": 13_500, "stream_bytes": 113_400}
    # +1,000 B of model repaid by -2,500 B of stream: slope -2.5 < -1 -> pays.
    pays = cl2.adjacent_slope("a", left, "b", {"model_packed_bytes": 14_500, "stream_bytes": 110_900})
    assert pays["slope_stream_per_model"] == -2.5 and pays["pays"] is True and pays["delta_joint_bytes"] == -1_500
    # +1,000 B of model repaid by only -600 B of stream: slope -0.6 >= -1 -> does not pay.
    fails = cl2.adjacent_slope("a", left, "b", {"model_packed_bytes": 14_500, "stream_bytes": 112_800})
    assert fails["slope_stream_per_model"] == -0.6 and fails["pays"] is False
    # Model did not grow: the rung pays only if the joint fell.
    flat = cl2.adjacent_slope("a", left, "b", {"model_packed_bytes": 13_500, "stream_bytes": 113_300})
    assert flat["slope_stream_per_model"] is None and flat["pays"] is True
    shrink = cl2.adjacent_slope("a", left, "b", {"model_packed_bytes": 13_400, "stream_bytes": 113_600})
    assert shrink["pays"] is False


def test_shipped_arithmetic_matches_the_charter() -> None:
    assert cl2.SHIPPED_JOINT_BYTES == 126_926
    demand = cl2.FS2_ARCHIVE_BYTES - cl2.RATE_CORNER_ARCHIVE_BYTES
    assert demand == pytest.approx(cl2.DEMAND_BYTES, abs=0.05)
