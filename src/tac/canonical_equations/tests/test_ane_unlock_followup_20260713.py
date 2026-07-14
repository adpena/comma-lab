from __future__ import annotations

import pytest

from tac.canonical_equations.ane_unlock_followup_20260713 import (
    account_weight_fit,
    admit_concurrency,
    batch_seconds_per_pair,
    forward_only_amdahl_speedup,
)


def test_concurrency_strict_five_percent_and_placement_are_separate() -> None:
    row = admit_concurrency(
        teacher_solo_s=1.0,
        teacher_concurrent_s=1.049,
        mlx_solo_s=2.0,
        mlx_concurrent_s=2.098,
        placement_proved=False,
    )
    assert row.timing_accept is True
    assert row.architecture_accept is False
    boundary = admit_concurrency(
        teacher_solo_s=1.0,
        teacher_concurrent_s=1.05,
        mlx_solo_s=1.0,
        mlx_concurrent_s=1.0,
        placement_proved=True,
    )
    assert boundary.timing_accept is False


def test_batch_and_weight_fit_laws_refuse_authority_upgrade() -> None:
    assert batch_seconds_per_pair(batch_seconds=0.08, batch_size=8) == pytest.approx(0.01)
    row = account_weight_fit(16 * 2**20, payload_evidence="MEASURED_PACKAGE_WEIGHT_BLOB_BYTES")
    assert row.clears_cliff is True
    assert row.derived_headroom_bytes == 16 * 2**20
    assert row.actual_ane_sram_residency == "UNKNOWN_NOT_MEASURED"


def test_amdahl_is_parameterized() -> None:
    assert forward_only_amdahl_speedup(forward_share=0.5, forward_speedup=2.0) == pytest.approx(4 / 3)
    with pytest.raises(ValueError):
        forward_only_amdahl_speedup(forward_share=1.1, forward_speedup=2.0)
