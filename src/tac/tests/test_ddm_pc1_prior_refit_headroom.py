"""Guards for ddm_pc1's prior-refit headroom probe.

The probe exists because a mis-named plane substituted silently for the coded token
field and read as 150x the true staleness.  These tests pin the two guards that catch
that class, plus the arithmetic the verdict rests on.
"""

from __future__ import annotations

import hashlib
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.ddm_pc1_prior_refit_headroom import (  # noqa: E402
    CALIBRATION,
    FIELD_SHAPE,
    HeadroomError,
    archive_sections,
    calibration_slope,
    headroom,
    load_field,
    prior_identity,
    split_member,
)

PAYLOAD_HEADER = struct.Struct("<4sBBBBHHH")


def _plane(fill: int = 0) -> np.ndarray:
    return np.full(FIELD_SHAPE, fill, dtype=np.uint8)


def _sha(plane: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(plane).tobytes()).hexdigest()


def _write_u8(tmp_path: Path, plane: np.ndarray, name: str = "field.u8") -> Path:
    path = tmp_path / name
    path.write_bytes(np.ascontiguousarray(plane).tobytes())
    return path


def _member(prior: bytes, renderer: bytes = b"r", carrier: bytes = b"c", tail: bytes = b"t") -> bytes:
    header = PAYLOAD_HEADER.pack(b"RX1M", 1, 2, 0, 250, len(prior), len(renderer), len(carrier))
    return header + prior + renderer + carrier + tail


def _archive(tmp_path: Path, name: str, member: bytes) -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("p", member)
    return path


# --- guard 2: the plane must be the bytes the caller declared -----------------------


def test_load_field_accepts_a_plane_whose_sha_matches(tmp_path):
    plane = _plane(3)
    field = load_field("live", _write_u8(tmp_path, plane), _sha(plane))
    assert field.sha256 == _sha(plane)
    assert field.plane.shape == FIELD_SHAPE


def test_load_field_refuses_a_substituted_plane(tmp_path):
    """The confound this module was written for: right shape, right dtype, wrong plane."""
    declared = _sha(_plane(1))
    path = _write_u8(tmp_path, _plane(2))
    with pytest.raises(HeadroomError, match="refusing"):
        load_field("live", path, declared)


def test_load_field_refuses_wrong_geometry(tmp_path):
    path = tmp_path / "short.u8"
    path.write_bytes(b"\x00" * 128)
    with pytest.raises(HeadroomError, match="bytes"):
        load_field("live", path, "0" * 64)


def test_load_field_refuses_a_symbol_outside_the_alphabet(tmp_path):
    plane = _plane(0)
    plane[0, 0, 0] = 7
    with pytest.raises(HeadroomError, match="alphabet"):
        load_field("live", _write_u8(tmp_path, plane), _sha(plane))


def test_load_field_reads_npy_and_npz(tmp_path):
    plane = _plane(2)
    npy = tmp_path / "f.npy"
    np.save(npy, plane)
    assert load_field("live", npy, _sha(plane)).sha256 == _sha(plane)
    npz = tmp_path / "f.npz"
    np.savez(npz, **{str(i): plane[i] for i in range(FIELD_SHAPE[0])})
    assert load_field("live", npz, _sha(plane)).sha256 == _sha(plane)


# --- guard 1: the shipped prior is named by byte identity, never by a memo ----------


def test_split_member_refuses_foreign_magic():
    bad = PAYLOAD_HEADER.pack(b"XXXX", 1, 2, 0, 0, 0, 0, 0)
    with pytest.raises(HeadroomError, match="magic"):
        split_member(bad)


def test_archive_sections_round_trips_the_four_sections(tmp_path):
    member = _member(b"prior-bytes", b"rend", b"carr", b"tail-bytes")
    sections = archive_sections(_archive(tmp_path, "a.zip", member))
    assert sections["prior"] == b"prior-bytes"
    assert sections["tail"] == b"tail-bytes"


def test_prior_identity_names_only_the_byte_identical_ancestor(tmp_path):
    live = _archive(tmp_path, "live.zip", _member(b"AAAA"))
    same = _archive(tmp_path, "same.zip", _member(b"AAAA", tail=b"different-tail"))
    other = _archive(tmp_path, "other.zip", _member(b"BBBB"))
    identity = prior_identity(live, {"same": same, "other": other, "absent": tmp_path / "nope.zip"})
    assert identity["produced_by"] == ["same"]
    assert identity["ancestors"]["other"]["identical_to_live"] is False
    assert identity["ancestors"]["absent"] == {"present": False}


# --- the verdict arithmetic ---------------------------------------------------------


def test_calibration_slope_is_the_single_measured_conversion():
    slope = calibration_slope()
    assert slope["n"] == 1
    assert slope["bytes_per_site"] == pytest.approx(-887 / 11128)
    assert slope["zero_drift_rows"] == [37]


def test_calibration_rows_are_the_two_byte_closed_instances():
    assert {row["sites"] for row in CALIBRATION} == {0, 11128}
    assert {row["delta_bytes"] for row in CALIBRATION} == {37, -887}


def test_headroom_counts_sites_and_refuses_a_rung_below_the_bar(tmp_path):
    fit = _plane(0)
    live = _plane(0)
    live[0, 0, :100] = 1
    fit_field = load_field("fit", _write_u8(tmp_path, fit, "a.u8"), _sha(fit))
    live_field = load_field("live", _write_u8(tmp_path, live, "b.u8"), _sha(live))
    result = headroom(fit_field, live_field, fire_bar_bytes=-25.0)
    assert result["drifted_sites"] == 100
    assert result["verdict"] == "REFIT_NOT_OWED"


def test_headroom_owes_a_rung_when_the_drift_is_large(tmp_path):
    fit = _plane(0)
    live = _plane(0)
    live[:, :, :] = 1
    live[0, 0, 0] = 0
    fit_field = load_field("fit", _write_u8(tmp_path, fit, "a.u8"), _sha(fit))
    live_field = load_field("live", _write_u8(tmp_path, live, "b.u8"), _sha(live))
    result = headroom(fit_field, live_field, fire_bar_bytes=-25.0)
    assert result["drifted_sites"] > 11128
    assert result["verdict"] == "REFIT_OWED"


def test_headroom_reports_zero_drift_for_an_identical_field(tmp_path):
    plane = _plane(4)
    field = load_field("fit", _write_u8(tmp_path, plane), _sha(plane))
    result = headroom(field, field, fire_bar_bytes=-25.0)
    assert result["drifted_sites"] == 0
    assert result["predicted_delta_bytes_slope_only"] == 0.0
    assert result["verdict"] == "REFIT_NOT_OWED"


def test_headroom_is_labelled_derived_never_a_price(tmp_path):
    plane = _plane(1)
    field = load_field("fit", _write_u8(tmp_path, plane), _sha(plane))
    result = headroom(field, field, fire_bar_bytes=-25.0)
    assert "DERIVED" in result["label"]
    assert "priced" in result["label"]
