"""The rc64 TWO-ROLE recipe contract.

Two distinct C bodies wear the file name ``rc64_backend.c``:

  ENCODER role   12,222 B  5c75e2c7...  encoder + decoder; the build stage appends the
                                        checkpoint/resume extension and compiles it.
  SHIPPED role    5,638 B  05839d14...  decoder ONLY; the member the archive carries at
                                        ``runtime/entropy/rc64_backend.c``.

``ddm_pq2_compress_e2e.RR4_RECIPE["rc64_source_sha256"]`` pins the ENCODER role, and that
pin is CORRECT -- the file exists on the VertigoDataTier pr135 intake tree and ddm_fx2's
byte-close used it successfully on 2026-08-17.  ddm_ma1's memo nonetheless reported the pin
unclearable after a scan keyed on the file name ``rc64_backend.c`` returned only the shipped
decoder body.  Nothing named the role, so the wrong body was measured against the pin.
These tests hold the distinction in place:

* the shipped body exports no encoder symbol, so it can never satisfy the encoder pin;
* a recipe may pin BOTH roles, and a wrong sha on EITHER is refused;
* rr4's recipe keeps its original input set exactly (no silent behaviour change).

The sha/behaviour facts are asserted against synthesised files, so the suite needs no
attached custody volume.  The two real bodies are checked only when they are present.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.ddm_pq2_compress_e2e import (
    RECEIVER_RECIPE_KEYS,
    RECIPE_KEYS,
    RR4_RECIPE,
    input_spec,
    load_recipe,
    verify_inputs,
)

ENCODER_ROLE_SHA = "5c75e2c70b89f148bc9d117d4dbd39a24dfb2e72ec41b0a7e9b9cf490ca07ee6"
ENCODER_ROLE_BYTES = 12_222
SHIPPED_ROLE_SHA = "05839d1416e68a49c8022d0cccb1581c3e4338fb14c867fc6c116e203c412996"
SHIPPED_ROLE_BYTES = 5_638

CUSTODY = Path("/Volumes/APDataStore/pact/ddm_rc1x/retained")
ENCODER_BODY = CUSTODY / "rc64_backend_encoder_role.c"
SHIPPED_BODY = CUSTODY / "rc64_backend_shipped_receiver_role.c"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fake_inputs(tmp_path: Path, recipe: dict[str, object]) -> dict[str, Path]:
    """Build a directory tree whose members hash to whatever the recipe declares."""
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "archive.zip").write_bytes(b"")
    hm1 = tmp_path / "hm1"
    hm1.mkdir()
    (hm1 / "group_index.u8").write_bytes(b"")
    tokens = tmp_path / "tokens.u8"
    tokens.write_bytes(b"")
    rc64 = tmp_path / "rc64_backend.c"
    rc64.write_bytes(b"")
    resolved = {
        "prepared_dir": prepared,
        "hm1_dir": hm1,
        "tokens_file": tokens,
        "rc64_source": rc64,
    }
    if recipe.get("rc64_shipped_member_sha256"):
        shipped = tmp_path / "shipped_rc64_backend.c"
        shipped.write_bytes(b"")
        resolved["rc64_shipped_member"] = shipped
    return resolved


def _empty_sha() -> str:
    return _sha(b"")


# --- the schema -------------------------------------------------------------


def test_receiver_key_is_optional_and_not_part_of_the_required_set() -> None:
    assert "rc64_shipped_member_sha256" not in RECIPE_KEYS
    assert RECEIVER_RECIPE_KEYS == ("rc64_shipped_member_sha256",)


def test_rr4_recipe_pins_the_encoder_role_and_keeps_its_original_inputs() -> None:
    assert RR4_RECIPE["rc64_source_sha256"] == ENCODER_ROLE_SHA
    assert "rc64_shipped_member_sha256" not in RR4_RECIPE
    assert sorted(input_spec(dict(RR4_RECIPE))) == [
        "hm1_dir",
        "prepared_dir",
        "rc64_source",
        "tokens_file",
    ]


def test_declaring_the_receiver_key_adds_exactly_one_verified_input(tmp_path: Path) -> None:
    recipe = dict(RR4_RECIPE)
    recipe["rc64_shipped_member_sha256"] = SHIPPED_ROLE_SHA
    spec = input_spec(recipe)
    assert "rc64_shipped_member" in spec
    assert spec["rc64_shipped_member"]["sha256"] == SHIPPED_ROLE_SHA
    assert spec["rc64_shipped_member"]["env"] == "TAC_PQ2_RC64_SHIPPED_MEMBER"
    assert spec["rc64_source"]["sha256"] == ENCODER_ROLE_SHA


def test_load_recipe_accepts_the_receiver_key_from_json(tmp_path: Path) -> None:
    recipe = dict(RR4_RECIPE)
    recipe["rc64_shipped_member_sha256"] = SHIPPED_ROLE_SHA
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps({"recipe": recipe}))
    assert load_recipe(path)["rc64_shipped_member_sha256"] == SHIPPED_ROLE_SHA


def test_load_recipe_still_refuses_a_genuinely_unknown_key(tmp_path: Path) -> None:
    recipe = dict(RR4_RECIPE)
    recipe["rc64_backend_sha256"] = SHIPPED_ROLE_SHA  # plausible-looking, still wrong
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps({"recipe": recipe}))
    with pytest.raises(SystemExit, match="unknown key"):
        load_recipe(path)


@pytest.mark.parametrize("blank", ["", None, "not-a-sha", "ABC" * 21 + "D", 12345])
def test_declared_but_unusable_receiver_pin_refuses_rather_than_skipping(blank: object) -> None:
    """VACUITY MUST NOT PASS.

    A declared-but-blank pin would otherwise produce a verification that silently checks
    nothing -- an instrument that reads green because it read nothing at all.
    """
    recipe = dict(RR4_RECIPE)
    recipe["rc64_shipped_member_sha256"] = blank
    with pytest.raises(SystemExit, match="not a 64-character lowercase hex"):
        input_spec(recipe)


def test_uppercase_sha_is_refused_because_pins_are_compared_lowercase() -> None:
    recipe = dict(RR4_RECIPE)
    recipe["rc64_shipped_member_sha256"] = SHIPPED_ROLE_SHA.upper()
    with pytest.raises(SystemExit, match="not a 64-character lowercase hex"):
        input_spec(recipe)


# --- the fail-closed verification (the positive controls) -------------------


def test_wrong_encoder_source_sha_is_refused(tmp_path: Path) -> None:
    """A source that is not the pinned encoder body must never reach the compiler."""
    recipe = dict(RR4_RECIPE)
    empty = _empty_sha()
    recipe.update(
        base_archive_sha256=empty,
        decoded_field_sha256=empty,
        rc64_source_sha256=ENCODER_ROLE_SHA,  # the file on disk is empty -> mismatch
    )
    spec = input_spec(recipe)
    resolved = _fake_inputs(tmp_path, recipe)
    with pytest.raises(SystemExit, match="rc64_source: sha256"):
        verify_inputs(spec, resolved)


def test_wrong_shipped_member_sha_is_refused(tmp_path: Path) -> None:
    """Pinning the receiver role is only worth anything if a mismatch fails closed."""
    recipe = dict(RR4_RECIPE)
    empty = _empty_sha()
    recipe.update(
        base_archive_sha256=empty,
        decoded_field_sha256=empty,
        rc64_source_sha256=empty,
        rc64_shipped_member_sha256=SHIPPED_ROLE_SHA,  # on-disk file is empty -> mismatch
    )
    spec = input_spec(recipe)
    resolved = _fake_inputs(tmp_path, recipe)
    with pytest.raises(SystemExit, match="rc64_shipped_member: sha256"):
        verify_inputs(spec, resolved)


def test_correct_shas_pass_and_report_both_roles(tmp_path: Path) -> None:
    """The negative control: when every sha agrees, verification returns both roles."""
    recipe = dict(RR4_RECIPE)
    empty = _empty_sha()
    recipe.update(
        base_archive_sha256=empty,
        decoded_field_sha256=empty,
        rc64_source_sha256=empty,
        rc64_shipped_member_sha256=empty,
    )
    spec = input_spec(recipe)
    resolved = _fake_inputs(tmp_path, recipe)
    manifest = verify_inputs(spec, resolved)
    by_name = {row["input"]: row for row in manifest}
    assert by_name["rc64_source"]["sha256_matches"] is True
    assert by_name["rc64_shipped_member"]["sha256_matches"] is True
    assert "ENCODER-ROLE" in by_name["rc64_source"]["role"]
    assert "SHIPPED" in by_name["rc64_shipped_member"]["role"]


# --- the structural fact the two arms missed --------------------------------


@pytest.mark.skipif(not SHIPPED_BODY.is_file(), reason="rc1x custody volume not attached")
def test_shipped_body_exports_no_encoder_symbol() -> None:
    """This is WHY the shipped member can never satisfy the encoder pin."""
    text = SHIPPED_BODY.read_text()
    assert SHIPPED_BODY.stat().st_size == SHIPPED_ROLE_BYTES
    assert _sha(SHIPPED_BODY.read_bytes()) == SHIPPED_ROLE_SHA
    for symbol in (
        "rc64_encoder_create",
        "rc64_encoder_encode",
        "rc64_encoder_finish",
        "rc64_encoder_data",
    ):
        assert symbol not in text, f"shipped decoder-only body unexpectedly exports {symbol}"
    assert "rc64_decoder_decode_probabilities" in text


@pytest.mark.skipif(not ENCODER_BODY.is_file(), reason="rc1x custody volume not attached")
def test_encoder_body_exports_both_halves() -> None:
    text = ENCODER_BODY.read_text()
    assert ENCODER_BODY.stat().st_size == ENCODER_ROLE_BYTES
    assert _sha(ENCODER_BODY.read_bytes()) == ENCODER_ROLE_SHA
    assert "rc64_encoder_encode" in text
    assert "rc64_decoder_decode_probabilities" in text


@pytest.mark.skipif(
    not (ENCODER_BODY.is_file() and SHIPPED_BODY.is_file()),
    reason="rc1x custody volume not attached",
)
def test_the_two_roles_are_different_files() -> None:
    assert ENCODER_BODY.read_bytes() != SHIPPED_BODY.read_bytes()
    assert ENCODER_ROLE_SHA != SHIPPED_ROLE_SHA
