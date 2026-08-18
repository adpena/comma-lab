"""Tests for the ddm_sz1 semantic-metadata byte-split transform and receiver patch.

The load-bearing tests are the ones that would catch a SHIPPING failure:

* the un-split must invert the split EXACTLY (a wrong inverse corrupts every weight);
* the code the RECEIVER actually runs -- rendered from the frozen profile -- must invert
  the code the ENCODER actually runs (encoder/receiver desynchronisation is the failure
  mode that a decoder, which takes no arguments, cannot detect at runtime);
* the patch must fail CLOSED when its anchor is absent (a receiver patch that silently
  does nothing would ship an archive no decoder can read);
* ``reserved == 0`` must remain byte-identity, and unknown reserved bits must refuse.

Several tests deliberately assert that the transform is NOT the identity, so the suite
cannot pass against a no-op implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# The module under test uses TOP-LEVEL sibling imports because that is what works when
# the real chain runs `python experiments/<script>.py` (experiments/ becomes sys.path[0]).
# pytest's pythonpath is [".", "src"], so without this bootstrap the bare import here
# resolves only when some OTHER experiments test happens to be collected first -- a
# collection-order dependency that passes in a full run and fails standalone.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ddm_sz1_semantic_metadata_split as sz1


def make_body(length: int = sz1.WANS_BODY_BYTES, seed: int = 7) -> bytes:
    return np.random.default_rng(seed).integers(0, 256, length, dtype=np.uint8).tobytes()


# --- the transform --------------------------------------------------------------------


@pytest.mark.parametrize("profile", list(sz1.PROFILES.values()))
def test_unsplit_inverts_split_exactly(profile: sz1.SplitProfile) -> None:
    body = make_body()
    assert sz1.unsplit_region(sz1.split_region(body, profile), profile) == body


@pytest.mark.parametrize("profile", list(sz1.PROFILES.values()))
def test_split_preserves_length_and_is_not_identity(profile: sz1.SplitProfile) -> None:
    body = make_body()
    split = sz1.split_region(body, profile)
    assert len(split) == len(body)
    assert split != body, "a split that is the identity would be an inert transform"


@pytest.mark.parametrize("profile", list(sz1.PROFILES.values()))
def test_split_touches_only_its_region(profile: sz1.SplitProfile) -> None:
    body = make_body()
    split = sz1.split_region(body, profile)
    start, end = profile.offset, profile.offset + profile.length
    assert split[:start] == body[:start]
    assert split[end:] == body[end:]


def test_split_groups_the_two_byte_planes() -> None:
    """High (odd-index) bytes first, then low (even-index) bytes -- the fp16 exponent
    plane is what gains the run structure the live coder is paid for (#859)."""
    profile = sz1.SplitProfile("t", 0, 8, "test")
    body = bytes([0, 1, 2, 3, 4, 5, 6, 7])
    assert sz1.split_region(body, profile) == bytes([1, 3, 5, 7, 0, 2, 4, 6])
    assert sz1.unsplit_region(bytes([1, 3, 5, 7, 0, 2, 4, 6]), profile) == body


def test_split_is_a_permutation_of_the_region() -> None:
    body = make_body()
    profile = sz1.SHIPPED_PROFILE
    start, end = profile.offset, profile.offset + profile.length
    assert sorted(sz1.split_region(body, profile)[start:end]) == sorted(body[start:end])


def test_odd_offset_profile_round_trips() -> None:
    """FX2_R5C sits at an odd offset and straddles the metadata boundary; a straddling
    region must still restore byte-exactly, because the un-split is a pure permutation."""
    body = make_body()
    assert sz1.FX2_R5C.offset % 2 == 1
    assert sz1.unsplit_region(sz1.split_region(body, sz1.FX2_R5C), sz1.FX2_R5C) == body


# --- profile validation ---------------------------------------------------------------


def test_profile_rejects_odd_length_negative_offset_and_overrun() -> None:
    with pytest.raises(sz1.SemanticSplitError):
        sz1.SplitProfile("odd", 0, 7, "x").validate()
    with pytest.raises(sz1.SemanticSplitError):
        sz1.SplitProfile("neg", -2, 8, "x").validate()
    with pytest.raises(sz1.SemanticSplitError):
        sz1.SplitProfile("over", 0, sz1.WANS_BODY_BYTES + 2, "x").validate()
    with pytest.raises(sz1.SemanticSplitError):
        sz1.SplitProfile("empty", 0, 0, "x").validate()


def test_derived_profile_is_taken_from_the_format_not_fitted() -> None:
    assert sz1.DERIVED.offset == sz1.F12_OFFSET_TABLE_BYTES
    assert sz1.DERIVED.length == sz1.F12_FIXED_METADATA_BYTES
    assert sz1.SHIPPED_PROFILE is sz1.DERIVED
    assert sz1.DERIVED.offset + sz1.DERIVED.length <= sz1.WANS_BODY_BYTES


# --- the RX1M reserved byte -----------------------------------------------------------


def test_reserved_zero_is_inactive_and_unknown_bits_refuse() -> None:
    assert sz1.semantic_split_active(0) is False
    assert sz1.semantic_split_active(sz1.RX1_RESERVED_SEMANTIC_SPLIT) is True
    for bad in (0x02, 0x80, 0xFF):
        with pytest.raises(sz1.SemanticSplitError):
            sz1.semantic_split_active(bad)


def test_set_rx1_reserved_changes_one_byte_only() -> None:
    model = sz1.RX1_MODEL_HEADER.pack(sz1.RX1_MAGIC, 1, 2, 0, 0, 10, 20, 30) + b"payload"
    updated = sz1.set_rx1_reserved(model, sz1.RX1_RESERVED_SEMANTIC_SPLIT)
    assert len(updated) == len(model)
    assert sum(a != b for a, b in zip(model, updated, strict=True)) == 1
    assert sz1.read_rx1_header(updated)[3] == sz1.RX1_RESERVED_SEMANTIC_SPLIT
    assert sz1.read_rx1_header(updated)[4:] == sz1.read_rx1_header(model)[4:]
    with pytest.raises(sz1.SemanticSplitError):
        sz1.set_rx1_reserved(model, 0x04)


def test_read_rx1_header_rejects_a_foreign_container() -> None:
    with pytest.raises(sz1.SemanticSplitError):
        sz1.read_rx1_header(b"NOPE" + bytes(20))
    with pytest.raises(sz1.SemanticSplitError):
        sz1.read_rx1_header(b"RX1M")


# --- the receiver patch ---------------------------------------------------------------

FAKE_RECEIVER = (
    "import numpy as np\n"
    "class ResidualArchiveError(ValueError):\n    pass\n"
    "WANS_BODY_BYTES = 36040\n"
    "def _decode(reserved, semantic_body):\n"
    + sz1.HEADER_ANCHOR
    + '        raise ResidualArchiveError("invalid RX1 model metadata")\n'
    + sz1.RECEIVER_ANCHOR
    + "        semantic = semantic_body\n"
    "    return semantic\n"
)


def test_patch_inserts_both_anchors() -> None:
    patched = sz1.patch_receiver_source(FAKE_RECEIVER)
    assert "DDM_SZ1_SEMANTIC_METADATA_SPLIT_V1" in patched
    assert "_sz1_unsplit_semantic" in patched
    assert "reserved != 0" not in patched, "the strict reserved check must be widened"
    assert "SZ1_RESERVED_KNOWN_BITS" in patched


def test_patch_fails_closed_on_missing_or_duplicate_anchor() -> None:
    with pytest.raises(sz1.SemanticSplitError):
        sz1.patch_receiver_source("def nothing():\n    pass\n")
    with pytest.raises(sz1.SemanticSplitError):
        sz1.patch_receiver_source(FAKE_RECEIVER + FAKE_RECEIVER)
    with pytest.raises(sz1.SemanticSplitError):
        sz1.patch_receiver_source(sz1.patch_receiver_source(FAKE_RECEIVER))


@pytest.mark.parametrize("profile", list(sz1.PROFILES.values()))
def test_rendered_receiver_inverts_the_encoder(profile: sz1.SplitProfile) -> None:
    """Anti-desynchronisation: execute the code the receiver will actually run.

    A decoder takes no arguments, so an encoder/receiver constant mismatch cannot be
    detected at decode time -- it silently corrupts every weight.  This test compiles
    the rendered helper and checks it inverts the encoder's own split.
    """
    namespace: dict[str, object] = {
        "np": np,
        "ResidualArchiveError": type("ResidualArchiveError", (ValueError,), {}),
    }
    exec(compile(sz1.receiver_helper_source(profile), "<receiver>", "exec"), namespace)
    assert namespace["SZ1_SPLIT_OFFSET"] == profile.offset
    assert namespace["SZ1_SPLIT_LENGTH"] == profile.length
    body = make_body()
    unsplit = namespace["_sz1_unsplit_semantic"]
    assert unsplit(sz1.split_region(body, profile)) == body
    assert unsplit(sz1.split_region(body, profile)) != sz1.split_region(body, profile)


def test_rendered_receiver_rejects_a_short_body() -> None:
    namespace: dict[str, object] = {
        "np": np,
        "ResidualArchiveError": type("ResidualArchiveError", (ValueError,), {}),
    }
    exec(compile(sz1.receiver_helper_source(), "<receiver>", "exec"), namespace)
    with pytest.raises(ValueError):
        namespace["_sz1_unsplit_semantic"](b"\x00" * 8)


def test_receiver_helper_bakes_the_frozen_profile() -> None:
    rendered = sz1.receiver_helper_source(sz1.TUNED)
    assert f"SZ1_SPLIT_OFFSET = {sz1.TUNED.offset}" in rendered
    assert f"SZ1_SPLIT_LENGTH = {sz1.TUNED.length}" in rendered


def test_profiles_are_distinct_offsets() -> None:
    offsets = [profile.offset for profile in sz1.PROFILES.values()]
    assert len(set(offsets)) == len(offsets)
