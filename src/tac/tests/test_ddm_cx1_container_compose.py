# SPDX-License-Identifier: MIT
"""Tests for the ddm_cx1 container composition encoder and its parity verifier.

Written so that a plausible no-op FAILS them.  The encoder's whole job is to
change bytes and NOTHING else, so most of these assert that a field which was
silently dropped, silently rounded, or silently migrated is refused rather than
absorbed — the ways a lossless claim turns into a fake one.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXPERIMENTS = _REPO_ROOT / "experiments"
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))


def _load(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cx1 = _load("cx1_build_ix2_container_archive", _REPO_ROOT / "tools" / "cx1_build_ix2_container_archive.py")
parity = _load("cx1_verify_frame_parity", _REPO_ROOT / "tools" / "cx1_verify_frame_parity.py")

from tac.optimization.ddm_ix2_archive_container import (  # noqa: E402
    parse_payload,
    unpack_config_section,
)
from tac.optimization.ddm_tr1_runtime import (  # noqa: E402
    RENDERER_FRAME_MAGIC,
    RENDERER_RAW_HEADER,
    _encode_brotli_frame,
)
from tac.optimization.pfs1_warp_receiver import ST_GRID as _PFS1_ST_GRID  # noqa: E402

# ANCHORED, not copied (ddm_qd1 2026-08-03).  This module previously carried its
# own literal copy of the vendored grid, so it CERTIFIED ITSELF: had
# ``pfs1_warp_receiver.ST_GRID`` drifted, every assertion here would still have
# passed against the stale local literal while the receiver shipped something
# else.  That matters because ddm_cx1 measured this exact constant to be the
# rule-118 discriminator -- vendored-generic on dc1_fold, FITTED and
# video-derived on pj2 -- so a silent drift is a compliance question, not a
# style question.
#
# The receiver-side copies are deliberately NOT de-duplicated: pfs1_warp_receiver
# documents itself as needing "NO tac dependency" because it is vendored whole
# into the shipping decode path, and collapsing those copies would break that
# self-containment.  Tests do not ship, so the verifier -- and only the verifier
# -- anchors to the canonical constant.
VENDORED_ST_GRID = tuple(float(v) for v in _PFS1_ST_GRID)
FITTED_ST_GRID = [0.06, 0.065, 0.07, 0.075, 0.08, 0.09, 0.1, 0.11, 0.12, 0.14, 0.16]
BETA_MAGS = [-7.5, -3.5, -2.5, -1.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 4.5]
POLICY = "warp_two_plane_static_photo_beta_v4d"
TR1_METADATA = {"schema": "ddm_tr1_four_section_packet.v1", "section_order": ["tokens"]}
POSE_STUB = b'{"inert":true}'


def _stub_receiver() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        FRAME0_POLICY=POLICY,
        IX2_TR1_METADATA=TR1_METADATA,
        IX2_POSE_STUB=POSE_STUB,
        IX2_JOINT_ORDER=("config", "renderer", "selector", "pose_warp"),
        ST_GRID=VENDORED_ST_GRID,
    )


def _renderer_section(mask_count: int = 40, float_count: int = 6) -> bytes:
    bits = (np.arange(mask_count) % 3 == 0).astype(np.uint8)
    floats = np.linspace(-1.0, 1.0, float_count).astype(">f2").tobytes()
    raw = (
        RENDERER_RAW_HEADER.pack(mask_count, float_count)
        + np.packbits(bits, bitorder="big").tobytes()
        + floats
    )
    return _encode_brotli_frame(RENDERER_FRAME_MAGIC, raw)


def _legacy(**overrides) -> cx1.LegacyArchive:
    manifest = {
        "frame0_policy": POLICY,
        "tr1_metadata": TR1_METADATA,
        "pose_dim0_offset": 32.1875,
        "rs_beta_mags": BETA_MAGS,
        "st_grid": FITTED_ST_GRID,
        "tokens_sha256": "0" * 64,
        "tr1_packet_sha256": "1" * 64,
        "renderer_sha256": "2" * 64,
        "selector_sha256": "3" * 64,
        "pose_stub_sha256": "4" * 64,
        "pose_warp_sha256": "5" * 64,
    }
    manifest.update(overrides.pop("manifest", {}))
    fields = {
        "manifest": manifest,
        "tokens": b"tokens",
        "renderer": _renderer_section(),
        "selector": b"selector-bytes" * 3,
        "pose_stub": POSE_STUB,
        "pose_warp": b"pose-warp-bytes" * 7,
        "zip_bytes": 1000,
    }
    fields.update(overrides)
    return cx1.LegacyArchive(**fields)


def _codes(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.integers(0, 16, size=(5, 3, 4, 2), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# migrated-constant guards                                                     #
# --------------------------------------------------------------------------- #


def test_build_refuses_a_frame0_policy_the_receiver_does_not_carry() -> None:
    """A migrated constant is only generic-in-fact when it equals the archive's."""

    legacy = _legacy(manifest={"frame0_policy": "some_other_policy"})
    with pytest.raises(cx1.CX1BuildError, match="frame0_policy"):
        cx1.build_container(
            legacy,
            vendored_st_grid=VENDORED_ST_GRID,
            token_codes=_codes(),
            receiver=_stub_receiver(),
        )


def test_build_refuses_divergent_tr1_metadata() -> None:
    legacy = _legacy(manifest={"tr1_metadata": {"schema": "something.else"}})
    with pytest.raises(cx1.CX1BuildError, match="tr1_metadata"):
        cx1.build_container(
            legacy,
            vendored_st_grid=VENDORED_ST_GRID,
            token_codes=_codes(),
            receiver=_stub_receiver(),
        )


def test_build_refuses_a_pose_stub_that_is_not_the_migrated_constant() -> None:
    legacy = _legacy(pose_stub=b'{"inert":false}')
    with pytest.raises(cx1.CX1BuildError, match="pose_stub"):
        cx1.build_container(
            legacy,
            vendored_st_grid=VENDORED_ST_GRID,
            token_codes=_codes(),
            receiver=_stub_receiver(),
        )


def test_build_refuses_a_joint_order_the_receiver_does_not_expect() -> None:
    receiver = _stub_receiver()
    receiver.IX2_JOINT_ORDER = ("config", "renderer", "selector")
    with pytest.raises(cx1.CX1BuildError, match="joint order"):
        cx1.build_container(
            _legacy(),
            vendored_st_grid=VENDORED_ST_GRID,
            token_codes=_codes(),
            receiver=receiver,
        )


# --------------------------------------------------------------------------- #
# rule-118 custody                                                             #
# --------------------------------------------------------------------------- #


def test_fitted_st_grid_is_counted_and_vendored_st_grid_is_migrated() -> None:
    """The (field, archive) pair property, exercised on both sides."""

    fitted_payload, fitted_acc = cx1.build_container(
        _legacy(),
        vendored_st_grid=VENDORED_ST_GRID,
        token_codes=_codes(),
        receiver=_stub_receiver(),
    )
    assert fitted_acc.custody["st_grid"] == "VIDEO_DERIVED"
    assert "st_grid" not in fitted_acc.migrated
    _, sections = parse_payload(fitted_payload)
    assert unpack_config_section(sections[0])[2] == tuple(FITTED_ST_GRID)

    generic_payload, generic_acc = cx1.build_container(
        _legacy(manifest={"st_grid": list(VENDORED_ST_GRID)}),
        vendored_st_grid=VENDORED_ST_GRID,
        token_codes=_codes(),
        receiver=_stub_receiver(),
    )
    assert generic_acc.custody["st_grid"] == "GENERIC"
    assert "st_grid" in generic_acc.migrated
    _, generic_sections = parse_payload(generic_payload)
    assert unpack_config_section(generic_sections[0])[2] is None
    # counting the fitted table must COST bytes; a "free" fitted migration is the fake
    assert fitted_acc.counted_config_bytes > generic_acc.counted_config_bytes


def test_fitted_beta_codebook_is_always_counted() -> None:
    """``rs_beta_mags`` is derived from this clip's solve on every v4d archive."""

    payload, accounting = cx1.build_container(
        _legacy(),
        vendored_st_grid=VENDORED_ST_GRID,
        token_codes=_codes(),
        receiver=_stub_receiver(),
    )
    assert accounting.custody["rs_beta_mags"] == "VIDEO_DERIVED"
    assert "rs_beta_mags" not in accounting.migrated
    _, sections = parse_payload(payload)
    assert list(unpack_config_section(sections[0])[1]) == BETA_MAGS


def test_all_six_manifest_hashes_are_deleted_not_migrated() -> None:
    """Two of the six were measured WRONG and none was ever read by a decode step."""

    _, accounting = cx1.build_container(
        _legacy(),
        vendored_st_grid=VENDORED_ST_GRID,
        token_codes=_codes(),
        receiver=_stub_receiver(),
    )
    hashes = {k for k in accounting.deleted if k.endswith("_sha256")}
    assert hashes == {
        "tokens_sha256",
        "tr1_packet_sha256",
        "renderer_sha256",
        "selector_sha256",
        "pose_stub_sha256",
        "pose_warp_sha256",
    }
    assert not any(k.endswith("_sha256") for k in accounting.migrated)


# --------------------------------------------------------------------------- #
# losslessness                                                                 #
# --------------------------------------------------------------------------- #


def test_every_counted_field_reconstructs_from_the_container_bytes_alone() -> None:
    legacy = _legacy()
    codes = _codes()
    receiver = _stub_receiver()
    payload, _ = cx1.build_container(
        legacy, vendored_st_grid=VENDORED_ST_GRID, token_codes=codes, receiver=receiver
    )
    checks = cx1.verify_container_bytes(payload, legacy, codes, receiver)
    assert all(checks.values()), checks
    assert set(checks) == {
        "tokens_bit_identical",
        "renderer_mask_bit_identical",
        "renderer_floats_bit_identical",
        "selector_bit_identical",
        "pose_warp_bit_identical",
        "dim0_offset_exact",
        "beta_mags_exact",
        "st_grid_exact",
    }


def test_verification_fails_when_a_section_is_corrupted() -> None:
    """The verifier must be able to return the NEGATIVE, or it proves nothing."""

    legacy = _legacy()
    codes = _codes()
    receiver = _stub_receiver()
    payload, _ = cx1.build_container(
        legacy, vendored_st_grid=VENDORED_ST_GRID, token_codes=codes, receiver=receiver
    )
    tampered = cx1.verify_container_bytes(
        payload, legacy, codes, receiver
    )
    assert all(tampered.values())
    other = _legacy(pose_warp=b"different-pose-warp-bytes")
    checks = cx1.verify_container_bytes(payload, other, codes, receiver)
    assert checks["pose_warp_bit_identical"] is False


def test_renderer_field_split_roundtrips() -> None:
    section = _renderer_section(mask_count=57, float_count=8)
    bits, floats = cx1.split_renderer_fields(section)
    assert bits.size == 57
    assert len(floats) == 16
    rebuilt = _encode_brotli_frame(
        RENDERER_FRAME_MAGIC,
        RENDERER_RAW_HEADER.pack(57, 8)
        + np.packbits(bits, bitorder="big").tobytes()
        + floats,
    )
    assert rebuilt == section


def test_read_legacy_archive_refuses_a_foreign_member_set(tmp_path: Path) -> None:
    path = tmp_path / "wrong.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps({}))
        archive.writestr("state/tokens.dr7t", b"x")
    with pytest.raises(cx1.CX1BuildError, match="6-member"):
        cx1.read_legacy_archive(path)


# --------------------------------------------------------------------------- #
# the parity verifier's own honesty                                            #
# --------------------------------------------------------------------------- #


def test_packet_equal_reports_every_field_and_never_returns_none() -> None:
    """A check that can only answer 'yes' or 'unknown' is the vacuity trap."""

    left = types.SimpleNamespace(
        metadata={"a": 1},
        selector={"b": 2},
        section_payloads=(b"x", b"y"),
        token_codes=np.arange(6),
        pose_stub_consumed=True,
        masks=(np.ones(3),),
        gains=(np.zeros(2),),
        biases=(np.full(2, 0.5),),
    )
    right = types.SimpleNamespace(**vars(left))
    verdict = parity._packet_equal(left, right)
    assert set(verdict) == {
        "metadata",
        "selector",
        "section_payloads",
        "token_codes",
        "pose_stub_consumed",
        "masks",
        "gains",
        "biases",
    }
    assert all(v is True for v in verdict.values())
    assert not any(v is None for v in verdict.values())

    right.token_codes = np.arange(6) + 1
    assert parity._packet_equal(left, right)["token_codes"] is False


def test_packet_equal_raises_on_a_missing_field_rather_than_reporting_unknown() -> None:
    left = types.SimpleNamespace(metadata={}, selector={})
    with pytest.raises(AttributeError):
        parity._packet_equal(left, left)


def test_state_equal_compares_arrays_elementwise() -> None:
    left = types.SimpleNamespace(
        n_pairs=2,
        dim0_offset=1.0,
        beta_mags=(0.0,),
        st_vals=np.array([0.1, 0.2]),
        p_best=np.zeros((2, 6)),
        st_idx=np.array([0, 1]),
        sel=np.array([0, 1]),
        ab=np.ones((2, 2)),
        beta_idx=np.array([0, 0]),
    )
    right = types.SimpleNamespace(**vars(left))
    assert all(parity._state_equal(left, right).values())
    right.p_best = np.ones((2, 6))
    assert parity._state_equal(left, right)["p_best"] is False
