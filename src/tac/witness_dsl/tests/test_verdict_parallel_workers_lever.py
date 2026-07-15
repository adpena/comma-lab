# SPDX-License-Identifier: MIT
"""#509 burn-down 2: VerdictParallelWorkers DSL Lever custody tests.

The lever composes ``--verdict-parallel-workers N`` (chunk-parallel ADVISORY CPU verdict;
bit-identical values, verdict-wall sec lever). Trainer default is 0 (sequential,
byte-identical); the lever exists so the flag is DSL-held (never hand-typed) and the
paired OFF arm is compose-nothing.
"""
from __future__ import annotations

import pytest

from tac.witness_dsl.curriculum_dsl import VerdictParallelWorkers


def test_lever_name_and_flag():
    lv = VerdictParallelWorkers(4)
    assert lv.name == "verdict_parallel_workers"
    assert lv.overrides == {"--verdict-parallel-workers": 4}


def test_default_workers_derived_from_headroom_law():
    """workers=None (default) derives the count from the registered law's measured
    constants — DERIVED value-provenance rung, never a hand count. The derived value
    is host-headroom-dependent but always inside [2, SIZED_WORKERS]."""
    from tac.canonical_equations.verdict_parallel_workers_speedup_20260715 import (
        SIZED_WORKERS,
        derived_verdict_workers,
    )

    w = VerdictParallelWorkers().overrides["--verdict-parallel-workers"]
    assert 2 <= w <= SIZED_WORKERS
    assert w == derived_verdict_workers()  # same law, same live headroom read


def test_derived_verdict_workers_pure_ladder():
    """Deterministic given available_gib: caps at the measured ladder top (w=8; no
    efficiency evidence beyond it), floors at 2, scales with the 2.9 GiB/worker
    measured marginal below the knee."""
    from tac.canonical_equations.verdict_parallel_workers_speedup_20260715 import (
        derived_verdict_workers,
    )

    assert derived_verdict_workers(available_gib=1000.0) == 8   # abundant -> ladder top
    assert derived_verdict_workers(available_gib=64.0) == 8     # burn-down host class
    assert derived_verdict_workers(available_gib=0.0) == 2      # no headroom -> floor
    # knee arithmetic: 0.70*avail - base(10.66) budget // 2.9 per worker
    assert derived_verdict_workers(available_gib=30.0) == 3   # (21.0-10.66)//2.9 = 3
    assert derived_verdict_workers(available_gib=40.0) == 5   # (28.0-10.66)//2.9 = 5


@pytest.mark.parametrize("w", [2, 3, 8, 16])
def test_worker_counts_pass_through(w):
    assert VerdictParallelWorkers(w).overrides["--verdict-parallel-workers"] == w


@pytest.mark.parametrize("w", [-1, 0, 1])
def test_noop_workers_refused(w):
    # off-is-orphan: an inert composed lever is orphaned signal — refuse, don't no-op.
    with pytest.raises(ValueError, match="workers >= 2"):
        VerdictParallelWorkers(w)


def test_trainer_flag_exists():
    """Never-invent-flags: the emitted flag must exist in the levelset trainer argparse."""
    import re
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4]
    src = (repo / "experiments/train_levelset_witness_realized_through_R_mlx.py").read_text()
    assert re.search(r"add_argument\(\s*\"--verdict-parallel-workers\"", src)
