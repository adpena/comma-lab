# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.canonical_equations.ddm_rg3_residual_family_productions_20260724 import (
    CLASS_BIRTH,
    FINER_EVENT,
    FISHER_STRATUM,
    select_rg3_residual_address,
)


def test_class_birth_derives_coarse_and_fine_band_with_one_magnitude() -> None:
    support = [0] * 384
    support[129] = 3
    support[144] = 7
    assert select_rg3_residual_address(support, family=CLASS_BIRTH) == {
        "row_band": 2,
        "fine_band": 1,
        "signed_magnitude_alphabet": (1,),
        "fisher_margin_field_shipped": False,
    }


def test_finer_event_uses_receiver_mass_in_bound_rg2_band() -> None:
    support = [0] * 384
    support[96] = 5
    assert select_rg3_residual_address(
        support,
        family=FINER_EVENT,
        row_band=1,
    )["fine_band"] == 2


def test_fisher_family_uses_weighted_mass_and_never_ships_field() -> None:
    support = [0] * 384
    weighted = [0.0] * 384
    support[4] = 10
    weighted[48] = 1.5
    result = select_rg3_residual_address(
        support,
        family=FISHER_STRATUM,
        row_band=0,
        fisher_weighted_support_mass_by_row=weighted,
    )
    assert result["fine_band"] == 3
    assert result["signed_magnitude_alphabet"] == (1, 2)
    assert result["fisher_margin_field_shipped"] is False


def test_fisher_family_fails_closed_without_weighted_mass() -> None:
    with pytest.raises(ValueError, match="requires weighted"):
        select_rg3_residual_address(
            [1] * 384,
            family=FISHER_STRATUM,
            row_band=0,
        )
