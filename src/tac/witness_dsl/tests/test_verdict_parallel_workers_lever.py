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


def test_default_workers():
    assert VerdictParallelWorkers().overrides["--verdict-parallel-workers"] == 4


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
