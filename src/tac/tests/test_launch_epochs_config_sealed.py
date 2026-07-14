"""NEW-1 (seal v7 round-2 docket): the launcher must not trample config-sealed epochs.

Bug class: ``tools/launch_witness_run.py`` hardcoded ``--epochs default=1000`` and threaded it
into every named-config derive call, silently overriding the derive functions' OWN sealed
defaults (crucible_v6/v7 seal 3000). The rc=8 wall-clock gate DERIVES its budget from whatever
epochs it is handed, so a forgotten ``--epochs 3000`` compiled a 2.48d budget instead of the
sealed 7.427d and PASSED every gate (orchestrator dry-run, 2026-07-08).

Fix under test: ``--epochs`` defaults to ``None``; ``derive_named_config(epochs=None)`` omits
the kwarg so each config family's signature default — the single source of truth — applies.
An explicit feasible value still overrides (with a loud provenance note at launch time).
An explicit value shorter than an enabled curriculum's stage caps now fails at config
construction; it is not a valid override merely because argparse accepted the integer.
"""
import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "tools"))

import launch_witness_run as L  # noqa: E402

_GT = str(_REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz")


def test_crucible_v6_none_epochs_uses_sealed_3000():
    cfg = L.derive_named_config("crucible_v6", _GT, num_pairs=8, epochs=None, overfit=True)
    assert int(cfg.epochs) == 3000


def test_crucible_v7_none_epochs_uses_sealed_3000():
    cfg = L.derive_named_config("crucible_v7", _GT, num_pairs=8, epochs=None, overfit=True)
    assert int(cfg.epochs) == 3000


def test_proven_base_none_epochs_uses_sealed_1000():
    cfg = L.derive_named_config("proven_base", _GT, num_pairs=8, epochs=None, overfit=True)
    assert int(cfg.epochs) == 1000


def test_feasible_explicit_epochs_still_overrides():
    cfg = L.derive_named_config("crucible_v6", _GT, num_pairs=8, epochs=1000, overfit=True)
    assert int(cfg.epochs) == 1000


def test_impossible_explicit_epochs_refuses_before_launch():
    with pytest.raises(ValueError, match="EPOCH-BUDGET FEASIBILITY"):
        L.derive_named_config("crucible_v6", _GT, num_pairs=8, epochs=42, overfit=True)


def test_argparse_epochs_defaults_to_none():
    """The launcher parser must default --epochs to None (config-sealed resolution), and still
    accept an explicit value."""
    import argparse

    src = (pathlib.Path(L.__file__)).read_text()
    assert '"--epochs", type=int, default=None' in src, (
        "--epochs must default to None so config-sealed defaults apply (NEW-1)")
    # And no residual hardcoded default=1000 on the epochs arg anywhere.
    assert '"--epochs", type=int, default=1000' not in src

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    assert ap.parse_args([]).epochs is None
    assert ap.parse_args(["--epochs", "3000"]).epochs == 3000


def test_unknown_config_still_raises():
    """The fail-LOUD unknown-name behavior (seal v7 r1 BLOCKER fix) survives the epochs change."""
    with pytest.raises(ValueError, match="unknown config name"):
        L.derive_named_config("not_a_config", _GT, num_pairs=8, epochs=None, overfit=True)
