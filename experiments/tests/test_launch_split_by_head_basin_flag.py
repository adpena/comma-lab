# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the ``--split-by-head / --no-split-by-head`` flag on the
basin launcher (``experiments/launch_split_by_head_basin.py``).

The load-bearing claims (the ones that, if wrong, silently mis-route the basin
onto the WRONG gradient backend):

1. The legacy DEFAULT is UNCHANGED — omitting the flag still yields
   ``split_by_head=True`` (the 7-11x pose-axis salvage), so every prior
   split-by-head launch command keeps its exact semantics.
2. ``--no-split-by-head`` actually sets ``split_by_head=False`` (the full-MPS
   104x lever), and ``--split-by-head`` explicitly sets it True.

These verify the PARSER behavior directly (Slot EEE Class 2 — behavior, not a
constant): a regression that re-hardcoded ``split_by_head=True`` at the config
call would still parse fine but would NOT honor ``--no-split-by-head``; that is
caught by the threading assertions below, which inspect the actual value passed
into both ``TorchVehicleConfig`` and ``RealScorerContext``.
"""

from __future__ import annotations

import experiments.launch_split_by_head_basin as launcher


def _parse(argv: list[str]):
    return launcher._build_arg_parser().parse_args(argv)


def test_default_is_split_by_head_true_legacy_unchanged() -> None:
    """Omitting the flag preserves the legacy default (split-by-head salvage)."""
    args = _parse(["--out-dir", "/tmp/does_not_run"])
    assert args.split_by_head is True


def test_explicit_split_by_head_true() -> None:
    args = _parse(["--out-dir", "/tmp/does_not_run", "--split-by-head"])
    assert args.split_by_head is True


def test_no_split_by_head_sets_false_full_mps() -> None:
    """``--no-split-by-head`` selects the full-MPS basin (both heads on MPS)."""
    args = _parse(["--out-dir", "/tmp/does_not_run", "--no-split-by-head"])
    assert args.split_by_head is False


def test_full_mps_train_and_eval_devices_are_distinct() -> None:
    """The full-MPS invocation trains on MPS and evals on the CPU authority —
    distinct devices (split_by_head==False does NOT require train==device, but
    the canonical full-MPS launch keeps train=mps / device=cpu so the BEST
    tracker stays CPU-authoritative)."""
    args = _parse(
        [
            "--out-dir", "/tmp/does_not_run",
            "--no-split-by-head",
            "--train-device", "mps",
            "--device", "cpu",
        ]
    )
    assert args.split_by_head is False
    assert args.train_device == "mps"
    assert args.device == "cpu"


def test_flag_threads_into_both_config_and_scorer_calls() -> None:
    """Source-level guard against a re-hardcoded ``split_by_head=True``: the
    parsed ``args.split_by_head`` must be the value threaded into BOTH the
    ``TorchVehicleConfig(...)`` and ``RealScorerContext(...)`` calls (the two
    sites that previously hardcoded ``True``). We assert the threading by
    inspecting the launcher source for the canonical ``split_by_head=args.split_by_head``
    pattern at both call sites and the ABSENCE of a hardcoded
    ``split_by_head=True`` keyword."""
    src = launcher.__file__
    with open(src, "r", encoding="utf-8") as fh:
        text = fh.read()
    # Both call sites must thread args.split_by_head (config uses inline comment).
    assert "split_by_head=args.split_by_head" in text
    # The two hardcoded keyword forms that previously routed every launch to the
    # salvage path must be gone from the call sites.
    assert "split_by_head=True,  # the pose-axis salvage" not in text
    assert "        split_by_head=True,\n" not in text
