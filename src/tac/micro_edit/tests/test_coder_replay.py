# SPDX-License-Identifier: MIT
"""Tests for the decode-identical coder replay.

The heavy positive controls (full n600 replays against the shipped cross-entropy
and the shipped corrected code length) are OPT-IN via ``TAC_ME1_HEAVY=1`` and also
require the retained assets to be mounted. Both were MEASURED exact to 0.000000 on
2026-08-17; the constants they check are recorded in the module docstring of
``tac.micro_edit.coder_replay``.

They are opt-in rather than default because each one memory-maps 1.18 GB of logits
plus 118 MB of tokens from an external SSD and takes ~20 s. Run inside the shared
pre-commit hook that cost a hard ``Bus error`` and would have made EVERY agent's
commit in this repo slower and flakier -- a correctness test that destabilises the
commit path for everyone is a net loss, however good the test is. Reproduce with::

    TAC_ME1_HEAVY=1 .venv/bin/python -m pytest src/tac/micro_edit/tests/ -q

The always-on tests cover the arithmetic invariant that explains every negative
this arm measured: a weighted arithmetic mean of odds multipliers is bounded by
the multipliers it mixes, so it can never beat the best of them.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tac.micro_edit.coder_replay import (
    EXPECTED_UNCORRECTED_CROSS_ENTROPY_BYTES,
    HPAC_LOGIT_PRECISION,
    NUM_CLASSES,
    PLANE,
    ReplayAssets,
    ReplayResult,
)

ASSETS = ReplayAssets(
    logits_i16=Path(
        "/Volumes/APDataStore/pact/ddm_hm1_20260816/retained/base_logits_int16_n600.i16"
    ),
    tokens_u8=Path(
        "/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/work_r2/inflated/"
        ".f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
    ),
    boundary_u8=Path(
        "/Volumes/APDataStore/pact/ddm_hm1_20260816/retained/boundary_bucket_n600.u8"
    ),
    group_index_u8=Path(
        "/Volumes/APDataStore/pact/ddm_hm1_20260816/retained/group_index.u8"
    ),
    table_values_npy=Path("/Volumes/APDataStore/pact/ddm_me1/table_values.npy"),
)

_HAVE_ASSETS = all(
    Path(p).exists()
    for p in (
        ASSETS.logits_i16,
        ASSETS.tokens_u8,
        ASSETS.boundary_u8,
        ASSETS.group_index_u8,
        ASSETS.table_values_npy,
    )
)
_HEAVY_ENABLED = os.environ.get("TAC_ME1_HEAVY") == "1"
requires_assets = pytest.mark.skipif(
    not (_HAVE_ASSETS and _HEAVY_ENABLED),
    reason="heavy replay controls are opt-in: set TAC_ME1_HEAVY=1 with assets mounted",
)


def test_plane_matches_the_contest_seg_denominator() -> None:
    """The token field must be exactly the SegNet argmax cell count per frame."""
    from tac.contest_oracle.constants import (
        CONTEST_NUM_PAIRS,
        CONTEST_PER_ARCHIVE_PIXEL_CELLS,
    )

    assert PLANE * CONTEST_NUM_PAIRS == CONTEST_PER_ARCHIVE_PIXEL_CELLS
    assert NUM_CLASSES == 5
    assert HPAC_LOGIT_PRECISION == 8


def test_replay_result_reports_bytes_and_deltas() -> None:
    a = ReplayResult("a", 8000.0, 2, 10, np.array([4000.0, 4000.0]))
    b = ReplayResult("b", 8800.0, 2, 10, np.array([4400.0, 4400.0]))
    assert a.code_bytes == 1000.0
    assert b.delta_bytes_vs(a) == 100.0


def test_assets_validate_refuses_missing_paths(tmp_path: Path) -> None:
    bad = ReplayAssets(
        logits_i16=tmp_path / "nope.i16",
        tokens_u8=tmp_path / "nope.u8",
        boundary_u8=tmp_path / "nope.u8",
        group_index_u8=tmp_path / "nope.u8",
        table_values_npy=tmp_path / "nope.npy",
    )
    with pytest.raises(FileNotFoundError):
        bad.validate()


def test_weighted_arithmetic_mean_cannot_beat_its_best_member() -> None:
    """The invariant that explains every mixture negative this arm measured.

    A count/quality-weighted arithmetic mean of odds multipliers is bounded below by
    the smallest member and above by the largest. So when one model is already the
    best available, blending it with weaker ones moves the estimate AWAY from that
    model's opinion and can only cost bits. Beating a strong single model needs a
    combination rule that SHARPENS (a product / logistic mix), not one that averages.
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        multipliers = rng.uniform(0.0625, 16.0, size=5)
        weights = rng.uniform(0.0, 10.0, size=5)
        if weights.sum() == 0:
            continue
        blended = float((weights * multipliers).sum() / weights.sum())
        assert multipliers.min() - 1e-12 <= blended <= multipliers.max() + 1e-12


def test_zero_weight_family_cannot_move_the_blend() -> None:
    """A cold family must ABSTAIN, never vote for 1.0 and drag the mixture."""
    multipliers = np.array([3.0, 1.0])
    weights = np.array([5.0, 0.0])
    blended = float((weights * multipliers).sum() / weights.sum())
    assert blended == 3.0


@requires_assets
def test_uncorrected_replay_reproduces_shipped_cross_entropy() -> None:
    """POSITIVE CONTROL 1 -- full n600, no corrector. MEASURED exact 2026-08-17."""
    from tac.micro_edit.coder_replay import replay_code_length

    result = replay_code_length(ASSETS, label="uncorrected")
    assert abs(result.code_bytes - EXPECTED_UNCORRECTED_CROSS_ENTROPY_BYTES) < 1e-3


@requires_assets
def test_live_corrector_replay_reproduces_shipped_code_length() -> None:
    """POSITIVE CONTROL 2 -- the shipped rr4 law. MEASURED exact 2026-08-17."""
    from tac.micro_edit.coder_replay import load_corrector_module, replay_code_length

    module = load_corrector_module(Path("experiments/ddm_rr4_free_corrector_v2.py"))
    result = replay_code_length(
        ASSETS, label="live", corrector_factory=lambda plane: module.FreeCorrector(plane)
    )
    assert abs(result.code_bits - 884_090.2210952122) < 1e-6


@requires_assets
def test_single_family_mixture_is_bit_identical_to_the_shipped_law() -> None:
    """The mixture refactor must reduce EXACTLY to the law it generalises."""
    from experiments.ddm_me1_mixed_context_corrector import (
        MixedContextCorrector,
        default_families,
    )
    from tac.micro_edit.coder_replay import replay_code_length

    shipped_only = [f for f in default_families() if f[0] == "shipped_joint"]
    result = replay_code_length(
        ASSETS,
        label="identity",
        corrector_factory=lambda plane: MixedContextCorrector(plane, families=shipped_only),
    )
    assert abs(result.code_bits - 884_090.2210952122) < 1e-6
