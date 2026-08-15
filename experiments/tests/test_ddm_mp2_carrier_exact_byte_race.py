# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util
from pathlib import Path

import brotli
import pytest

MODULE_PATH = Path(__file__).parents[1] / "ddm_mp2_carrier_exact_byte_race.py"
SPEC = importlib.util.spec_from_file_location("ddm_mp2_carrier_exact_byte_race", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
RACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RACE)


def test_exact_race_retains_all_payloads_and_ties_best(tmp_path: Path) -> None:
    raw = bytes(range(256)) * 86 + bytes(range(203))
    assert len(raw) == 22_219
    source = tmp_path / "carrier.raw.bin"
    incumbent = tmp_path / "carrier.br"
    source.write_bytes(raw)
    incumbent.write_bytes(brotli.compress(raw, quality=11))
    result = RACE.run_race(source, incumbent, tmp_path / "output")
    assert result["candidate_denominator"] == 12
    assert result["candidate_complete"] == 12
    assert result["all_decode_exact"] is True
    assert result["best"]["delta_bytes_vs_incumbent"] == 0
    for row in result["rows"]:
        assert Path(row["payload"]["path"]).is_file()
        assert Path(row["repeat"]["path"]).read_bytes() == Path(row["payload"]["path"]).read_bytes()


def test_exact_race_refuses_wrong_source_size(tmp_path: Path) -> None:
    source = tmp_path / "carrier.raw.bin"
    incumbent = tmp_path / "carrier.br"
    source.write_bytes(b"wrong")
    incumbent.write_bytes(brotli.compress(b"wrong", quality=11))
    with pytest.raises(RACE.CarrierRaceRefusal, match="22,219-byte"):
        RACE.run_race(source, incumbent, tmp_path / "output")


def test_exact_race_refuses_incumbent_decode_mismatch(tmp_path: Path) -> None:
    raw = b"a" * 22_219
    source = tmp_path / "carrier.raw.bin"
    incumbent = tmp_path / "carrier.br"
    source.write_bytes(raw)
    incumbent.write_bytes(brotli.compress(b"b" * 22_219, quality=11))
    with pytest.raises(RACE.CarrierRaceRefusal, match="does not decode"):
        RACE.run_race(source, incumbent, tmp_path / "output")
