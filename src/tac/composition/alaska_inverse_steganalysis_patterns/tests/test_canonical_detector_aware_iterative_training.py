# SPDX-License-Identifier: MIT
"""Tests for canonical ALASKA detector-aware iterative training config."""

from __future__ import annotations

import pytest

from tac.composition.alaska_inverse_steganalysis_patterns import (
    DetectorAwareIterativeTrainingStrategy,
    DetectorAwareTrainingConfig,
    DetectorAwareTrainingError,
)
from tac.composition.alaska_inverse_steganalysis_patterns.canonical_detector_aware_iterative_training import (
    CANONICAL_ALASKA_LR_SCHEDULE,
    CANONICAL_ALASKA_MAX_ITER,
)


def test_canonical_lr_schedule_verbatim_yousfi() -> None:
    """Verbatim Yousfi 2019 ALASKA-#1: boundaries=[20000,100000], values=[1e-4,1e-3,1e-4]."""
    assert CANONICAL_ALASKA_LR_SCHEDULE == (
        (0, 1e-4),
        (20_000, 1e-3),
        (100_000, 1e-4),
    )


def test_canonical_max_iter_verbatim_yousfi() -> None:
    """Verbatim Yousfi 2019: max_iter = 200000."""
    assert CANONICAL_ALASKA_MAX_ITER == 200_000


def test_strategy_class_exists() -> None:
    """Marker class for downstream consumers to import."""
    assert DetectorAwareIterativeTrainingStrategy is not None


def test_default_config_uses_canonical_lr_schedule() -> None:
    cfg = DetectorAwareTrainingConfig()
    assert cfg.lr_schedule == CANONICAL_ALASKA_LR_SCHEDULE
    assert cfg.max_iter == CANONICAL_ALASKA_MAX_ITER


def test_default_optimizer_adamax() -> None:
    cfg = DetectorAwareTrainingConfig()
    assert cfg.optimizer_class == "Adamax"


def test_default_ema_decay_comma_canonical() -> None:
    """COMMA adaptation per CLAUDE.md "EMA - NON-NEGOTIABLE": decay=0.997."""
    cfg = DetectorAwareTrainingConfig()
    assert cfg.ema_decay == 0.997


def test_default_eval_roundtrip_active() -> None:
    """COMMA adaptation per CLAUDE.md "eval_roundtrip - NON-NEGOTIABLE": True."""
    cfg = DetectorAwareTrainingConfig()
    assert cfg.eval_roundtrip_active is True


def test_default_pair_constraint_batch_active() -> None:
    """The pair-constraint batching pattern MUST be on by default."""
    cfg = DetectorAwareTrainingConfig()
    assert cfg.pair_constraint_batch is True


def test_default_multi_scheme_prior_active() -> None:
    """The multi-scheme Dirichlet prior MUST be on by default."""
    cfg = DetectorAwareTrainingConfig()
    assert cfg.multi_scheme_prior_active is True


def test_lr_at_step_canonical_warmup_phase() -> None:
    """Step 0..19999 -> LR 1e-4."""
    cfg = DetectorAwareTrainingConfig()
    assert cfg.lr_at_step(0) == 1e-4
    assert cfg.lr_at_step(19_999) == 1e-4


def test_lr_at_step_canonical_main_phase() -> None:
    """Step 20000..99999 -> LR 1e-3."""
    cfg = DetectorAwareTrainingConfig()
    assert cfg.lr_at_step(20_000) == 1e-3
    assert cfg.lr_at_step(50_000) == 1e-3
    assert cfg.lr_at_step(99_999) == 1e-3


def test_lr_at_step_canonical_fine_tune_phase() -> None:
    """Step >= 100000 -> LR 1e-4."""
    cfg = DetectorAwareTrainingConfig()
    assert cfg.lr_at_step(100_000) == 1e-4
    assert cfg.lr_at_step(150_000) == 1e-4
    assert cfg.lr_at_step(200_000) == 1e-4


def test_lr_at_step_rejects_negative() -> None:
    cfg = DetectorAwareTrainingConfig()
    with pytest.raises(DetectorAwareTrainingError, match="must be >= 0"):
        cfg.lr_at_step(-1)


def test_config_rejects_bad_max_iter() -> None:
    with pytest.raises(DetectorAwareTrainingError, match="must be >= 1"):
        DetectorAwareTrainingConfig(max_iter=0)


def test_config_rejects_bad_optimizer() -> None:
    with pytest.raises(DetectorAwareTrainingError, match="must be one of"):
        DetectorAwareTrainingConfig(optimizer_class="MyAdam")


def test_config_rejects_bad_ema_decay() -> None:
    with pytest.raises(DetectorAwareTrainingError, match=r"must be in \(0, 1\)"):
        DetectorAwareTrainingConfig(ema_decay=1.0)


def test_config_rejects_non_monotonic_schedule() -> None:
    with pytest.raises(DetectorAwareTrainingError, match="strictly increasing"):
        DetectorAwareTrainingConfig(
            lr_schedule=((0, 1e-4), (100, 1e-3), (50, 1e-4))
        )


def test_config_rejects_negative_step() -> None:
    with pytest.raises(DetectorAwareTrainingError, match="must be >= 0"):
        DetectorAwareTrainingConfig(lr_schedule=((-1, 1e-4),))


def test_config_rejects_non_positive_lr() -> None:
    with pytest.raises(DetectorAwareTrainingError, match="must be > 0"):
        DetectorAwareTrainingConfig(lr_schedule=((0, 0.0),))


def test_config_substantively_distinct_3_phases() -> None:
    """Slot EEE substantive-distinctness: the 3-phase LR schedule MUST
    produce distinct LRs at the canonical boundaries (1e-4 / 1e-3 / 1e-4
    is two distinct values, not one)."""
    cfg = DetectorAwareTrainingConfig()
    lrs = {cfg.lr_at_step(0), cfg.lr_at_step(50_000), cfg.lr_at_step(150_000)}
    assert len(lrs) == 2  # canonical 1e-4 / 1e-3 / 1e-4 = 2 distinct values
    assert 1e-4 in lrs
    assert 1e-3 in lrs


def test_warm_start_branch_optional() -> None:
    """Warm-start checkpoint branch is optional (cold-start = None)."""
    cfg_warm = DetectorAwareTrainingConfig(warm_start_checkpoint_branch="YCrCb")
    cfg_cold = DetectorAwareTrainingConfig(warm_start_checkpoint_branch=None)
    assert cfg_warm.warm_start_checkpoint_branch == "YCrCb"
    assert cfg_cold.warm_start_checkpoint_branch is None
