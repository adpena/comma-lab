# SPDX-License-Identifier: MIT
"""Tests for ``ddm_to1`` -- the ma1 tail-override on the live ck2 pointer body.

TWO TIERS, ON PURPOSE.  The rewrite tests are pure functions over strings and run
everywhere.  The custody tests read the pinned archives from the SSD tier and are
skipped when it is not mounted -- so this module reports the DENOMINATOR
(``test_custody_tier_denominator_is_reported``) rather than going silently green
on a machine where nothing was checked.  A skipped gate that looks like a pass is
the vacuity-equals-pass failure.
"""

from __future__ import annotations

import ast

import pytest

from experiments import ddm_to1_tail_override as to1

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

_CUSTODY = (
    to1.CK2_ARCHIVE.is_file()
    and to1.MA1_ARCHIVE.is_file()
    and to1.FX2_D1_ARCHIVE.is_file()
)
_needs_custody = pytest.mark.skipif(
    not _CUSTODY, reason="SSD custody tier not mounted; pinned archives unavailable"
)


# --- pure functions: always run ------------------------------------------------------


def test_rewrite_removes_every_repo_absolute_import() -> None:
    staged = to1.rewrite_corrector_to_relative_imports(
        to1.MA1_CORRECTOR_SOURCE.read_text()
    )
    code = [
        line
        for line in staged.splitlines()
        if line.startswith(("from ", "import ")) or "__import__" in line
    ]
    assert code, "the rewrite produced a module with no imports at all"
    assert not any("experiments." in line for line in code), code


def test_rewrite_output_parses_and_keeps_the_drop_in_name() -> None:
    staged = to1.rewrite_corrector_to_relative_imports(
        to1.MA1_CORRECTOR_SOURCE.read_text()
    )
    tree = ast.parse(staged)
    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    # residual_archive.py does `from .free_corrector import FreeCorrector`.
    assert "FreeCorrector" in classes
    assert "Ma1WithinMissCorrector" in classes


def test_rewrite_binds_shipped_config_relatively() -> None:
    staged = to1.rewrite_corrector_to_relative_imports(
        to1.MA1_CORRECTOR_SOURCE.read_text()
    )
    assert "from .fx2_model_axis_corrector import SHIPPED_CONFIG" in staged
    assert "__import__(" not in staged


def test_rewrite_fails_closed_when_an_expected_import_is_absent() -> None:
    # A silently-missed rewrite is what ships a tree that cannot decode.
    with pytest.raises(to1.To1Error, match="expected import not found"):
        to1.rewrite_corrector_to_relative_imports("import numpy as np\n")


def test_rewrite_fails_closed_when_the_dynamic_import_is_absent() -> None:
    source = to1.MA1_CORRECTOR_SOURCE.read_text().replace(to1._DYNAMIC_IMPORT, "")
    with pytest.raises(to1.To1Error, match="dynamic SHIPPED_CONFIG import"):
        to1.rewrite_corrector_to_relative_imports(source)


def test_s_per_byte_matches_the_upstream_rate_term() -> None:
    assert to1.RATE_DENOMINATOR == 37_545_489
    assert to1.S_PER_BYTE == pytest.approx(25.0 / 37_545_489)  # noqa: SIM300


def test_expected_delta_is_the_pre_registered_ma1_projection() -> None:
    # ma1 measured -104.584 B of code length and pre-registered -105 B of archive.
    assert to1.EXPECTED_TAIL_DELTA_BYTES == -105


# --- custody tier --------------------------------------------------------------------


def test_custody_tier_denominator_is_reported() -> None:
    """Name what was and was not checked, so a skip cannot read as a pass."""
    present = {
        "ck2": to1.CK2_ARCHIVE.is_file(),
        "ma1": to1.MA1_ARCHIVE.is_file(),
        "fx2_d1": to1.FX2_D1_ARCHIVE.is_file(),
    }
    checked = sum(present.values())
    assert checked in (0, 3), (
        f"custody tier is PARTIALLY mounted ({checked}/3 archives): {present}. "
        "A partial mount would let some gates run and others skip."
    )


@_needs_custody
def test_tail_identity_gate_proves_ck2_borrows_the_fx2_tail() -> None:
    gate = to1.tail_identity_gate()
    assert gate["verdict"] == "IDENTITY"
    # The whole claim: the object ma1 re-encoded IS the object ck2 ships.
    assert gate["ck2_tail_sha256"] == gate["fx2_tail_sha256"]
    assert gate["tail_delta_bytes"] == to1.EXPECTED_TAIL_DELTA_BYTES
    assert gate["ck2_tail_bytes"] - gate["ma1_tail_bytes"] == 105


@_needs_custody
def test_identity_control_off_reproduces_the_pointer_byte_identically() -> None:
    control = to1.identity_control_off()
    assert control["archive_sha256"] == to1.CK2_ARCHIVE_SHA256
    assert control["archive_bytes"] == to1.CK2_ARCHIVE_BYTES


@_needs_custody
def test_splice_is_tail_only_and_prices_as_pure_rate(tmp_path) -> None:
    built = to1.build_override(tmp_path / "candidate.zip")
    assert built["archive_bytes"] == to1.CK2_ARCHIVE_BYTES - 105
    assert built["determinism_repeat_byte_identical"] is True
    price = built["price"]
    assert price["delta_bytes_vs_pointer"] == -105
    assert price["dS_seg"] == 0.0 and price["dS_pose"] == 0.0
    assert price["net_dS"] == pytest.approx(-105 * to1.S_PER_BYTE, rel=1e-12)
    # comfortably past the -3.5e-6 admit bar
    assert price["net_dS"] < -3.5e-06


@_needs_custody
def test_splice_preserves_every_non_tail_section(tmp_path) -> None:
    from experiments.ddm_sa3_rebase_sz1 import sections

    built = to1.build_override(tmp_path / "candidate.zip")
    new = sections(tmp_path / "candidate.zip")
    base = sections(to1.CK2_ARCHIVE)
    for field in ("hpac", "semantic", "carrier", "reserved", "magic", "version"):
        assert new[field] == base[field], f"{field} moved on a tail-only edit"
    assert new["tail"] != base["tail"]
    assert built["tail_bytes"] == len(new["tail"])


@_needs_custody
def test_splice_gate_parser_agrees_with_the_composer_parser() -> None:
    """The gate's in-memory parser must be the composer's, not a lookalike."""
    from experiments.ddm_sa3_rebase_sz1 import sections

    parsed = to1.read_member_sections(to1.CK2_ARCHIVE.read_bytes())
    reference = sections(to1.CK2_ARCHIVE)
    for field in ("magic", "version", "codec", "table_mode", "reserved",
                  "hpac", "semantic", "carrier", "tail"):
        assert parsed[field] == reference[field], field


@_needs_custody
def test_splice_gate_has_discriminating_power() -> None:
    """A mutated section must parse DIFFERENTLY, or the gate could not catch it.

    The gate this replaced compared a slice against itself and was true by
    construction. This asserts the replacement can actually see a change.
    """
    base = to1.read_member_sections(to1.CK2_ARCHIVE.read_bytes())
    member = bytearray(to1.read_member(to1.CK2_ARCHIVE))
    # flip one bit inside the semantic section (past the 14-byte header + hpac)
    target = to1.RX1_HEADER.size + len(base["hpac"]) + 1
    member[target] ^= 0x01
    mutated = to1.read_member_sections(to1.deterministic_zip(bytes(member)))
    assert mutated["semantic"] != base["semantic"]
    assert mutated["hpac"] == base["hpac"]
    assert mutated["tail"] == base["tail"]


@_needs_custody
def test_composer_hook_defaults_to_the_verbatim_borrow() -> None:
    """The override must be opt-in: an omitted argument keeps sz1's tail."""
    import inspect

    from experiments.ddm_sa3_rebase_sz1 import build_candidate

    signature = inspect.signature(build_candidate)
    assert signature.parameters["tail_override"].default is None
