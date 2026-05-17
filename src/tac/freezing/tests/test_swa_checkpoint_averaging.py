# SPDX-License-Identifier: MIT
"""Tests for ``tac.freezing.swa_checkpoint_averaging``."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tac.freezing.swa_checkpoint_averaging import (
    SWACheckpointAverager,
    SWACheckpointReport,
)


def test_initial_state_empty():
    """A freshly-constructed averager has snapshot_count=0."""
    swa = SWACheckpointAverager()
    assert swa.snapshot_count == 0


def test_apply_without_snapshots_raises():
    """Calling ``averaged_state_dict()`` with no snapshots raises."""
    swa = SWACheckpointAverager()
    m = nn.Linear(4, 2)
    with pytest.raises(RuntimeError):
        swa.apply_to(m)


def test_single_snapshot_equals_model():
    """A single snapshot returns the model's exact state."""
    torch.manual_seed(0)
    m = nn.Linear(4, 2)
    swa = SWACheckpointAverager()
    swa.update(m)
    avg = swa.averaged_state_dict()
    for k, v in m.state_dict().items():
        assert torch.allclose(v, avg[k], atol=1e-6)


def test_two_snapshots_average_correctly():
    """Two snapshots produce the arithmetic mean."""
    m = nn.Linear(4, 2)
    with torch.no_grad():
        m.weight.fill_(1.0)
        m.bias.fill_(0.0)
    swa = SWACheckpointAverager()
    swa.update(m)
    with torch.no_grad():
        m.weight.fill_(3.0)
        m.bias.fill_(2.0)
    swa.update(m)
    avg = swa.averaged_state_dict()
    # Mean of 1.0 and 3.0 is 2.0; mean of 0.0 and 2.0 is 1.0.
    assert torch.allclose(avg["weight"], torch.full_like(m.weight, 2.0), atol=1e-6)
    assert torch.allclose(avg["bias"], torch.full_like(m.bias, 1.0), atol=1e-6)


def test_running_average_three_snapshots():
    """Running-average algorithm produces the correct mean across N snapshots."""
    m = nn.Linear(2, 2)
    swa = SWACheckpointAverager()
    values = [1.0, 2.0, 6.0]
    for v in values:
        with torch.no_grad():
            m.weight.fill_(v)
            m.bias.fill_(v)
        swa.update(m)
    avg = swa.averaged_state_dict()
    mean = sum(values) / len(values)
    assert torch.allclose(avg["weight"], torch.full_like(m.weight, mean), atol=1e-6)


def test_apply_to_model_loads_averaged_state():
    """``apply_to`` loads the averaged weights into a model."""
    m1 = nn.Linear(2, 2)
    m2 = nn.Linear(2, 2)
    with torch.no_grad():
        m1.weight.fill_(1.0)
        m1.bias.fill_(1.0)
        m2.weight.fill_(3.0)
        m2.bias.fill_(3.0)
    swa = SWACheckpointAverager()
    swa.update(m1)
    swa.update(m2)
    target = nn.Linear(2, 2)
    swa.apply_to(target)
    assert torch.allclose(target.weight, torch.full_like(target.weight, 2.0), atol=1e-6)


def test_report_carries_snapshot_count():
    """Reports correctly track snapshot count."""
    m = nn.Linear(4, 2)
    swa = SWACheckpointAverager()
    swa.update(m)
    swa.update(m)
    swa.update(m)
    report = swa.report()
    assert isinstance(report, SWACheckpointReport)
    assert report.snapshot_count == 3


def test_non_floating_buffers_preserved():
    """Non-floating buffers (e.g. BN num_batches_tracked) get last-write-wins semantics."""
    # BatchNorm has a long-tracked count of int type.
    m = nn.BatchNorm1d(4)
    swa = SWACheckpointAverager()
    # Simulate two forward passes that increment num_batches_tracked.
    m(torch.randn(2, 4))
    swa.update(m)
    m(torch.randn(2, 4))
    swa.update(m)
    avg = swa.averaged_state_dict()
    # The int buffer "num_batches_tracked" must be present.
    assert "num_batches_tracked" in avg
    # And it's the latest non-float value (2 forward passes).
    assert int(avg["num_batches_tracked"]) == 2


def test_averaging_does_not_mutate_source_model():
    """SWA snapshots do not mutate the source model's state."""
    m = nn.Linear(4, 2)
    initial = {k: v.clone() for k, v in m.state_dict().items()}
    swa = SWACheckpointAverager()
    swa.update(m)
    # Source unchanged.
    for k, v in m.state_dict().items():
        assert torch.equal(v, initial[k])


def test_chained_use_as_apply_returns_report():
    """``apply_to`` returns a typed report."""
    m = nn.Linear(2, 2)
    swa = SWACheckpointAverager()
    swa.update(m)
    swa.update(m)
    rep = swa.apply_to(m)
    assert isinstance(rep, SWACheckpointReport)
    assert rep.snapshot_count == 2
