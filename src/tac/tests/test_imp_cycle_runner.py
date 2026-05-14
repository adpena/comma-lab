# SPDX-License-Identifier: MIT
"""Lane 17 — IMP 10-cycle orchestrator tests.

Per Phase 3 Lane 17 spec (memory project_phases_2_3_4_*):
- Pruning mask determinism (same input → same mask)
- Weight count decreases monotonically across cycles
- Sparsity tracker reports honestly

All claims tagged [synthetic].

CLAUDE.md non-negotiables verified:
- No scorer load (synthetic train_step_fn)
- No silent defaults (every arg required)
- No GPU
- Deterministic CPU-only
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Allow imports of experiments.* without installing
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from experiments.imp_cycle_runner import (  # noqa: E402
    ImpRunResult,
    _write_manifest,
    run_imp_cycles,
)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic model + no-op train_step_fn
# ─────────────────────────────────────────────────────────────────────────────


def _make_tiny_renderer(seed: int = 2026) -> nn.Module:
    """Tiny conv renderer for fast cycle tests."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 5, kernel_size=1),
    )


def _no_op_train_step(model: nn.Module, cycle_idx: int) -> float:
    """Train-step callback that returns a deterministic synthetic loss.

    Does NOT actually train — Lane 17 scaffold tests verify the orchestrator
    plumbing, not the training itself.
    """
    return 0.5 - 0.05 * float(cycle_idx)  # decreasing synthetic loss


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: weight count decreases monotonically across cycles (sparsity tracker)
# ─────────────────────────────────────────────────────────────────────────────


def test_sparsity_monotone_across_cycles_synthetic() -> None:
    """[synthetic] Sparsity must monotonically increase across IMP cycles."""
    model = _make_tiny_renderer()
    result = run_imp_cycles(
        model=model,
        num_cycles=4,
        sparsity_increment=0.20,
        train_step_fn=_no_op_train_step,
    )
    assert isinstance(result, ImpRunResult)
    assert result.monotone_sparsity is True
    assert len(result.cycle_results) == 4
    sparsities = [cr.sparsity_after_prune for cr in result.cycle_results]
    # Each cycle's sparsity should be >= previous
    for prev, curr in zip(sparsities[:-1], sparsities[1:]):
        assert curr >= prev - 1e-9, f"sparsity went down: {prev} → {curr}"
    # Final cycle's sparsity should be close to expected (1 - 0.8^4 = 0.5904)
    assert sparsities[-1] == pytest.approx(0.5904, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: weight_count_kept decreases each cycle
# ─────────────────────────────────────────────────────────────────────────────


def test_weight_count_kept_decreases_each_cycle_synthetic() -> None:
    """[synthetic] weight_count_kept must non-increase across cycles."""
    model = _make_tiny_renderer()
    result = run_imp_cycles(
        model=model,
        num_cycles=3,
        sparsity_increment=0.30,
        train_step_fn=_no_op_train_step,
    )
    kept = [cr.weight_count_kept for cr in result.cycle_results]
    for prev, curr in zip(kept[:-1], kept[1:]):
        assert curr <= prev, f"weight_count_kept went UP: {prev} → {curr}"
    # weight_count_total stays constant
    totals = [cr.weight_count_total for cr in result.cycle_results]
    assert all(t == totals[0] for t in totals)


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: pruning mask determinism — same model + same seed → same final mask
# ─────────────────────────────────────────────────────────────────────────────


def test_pruning_mask_deterministic_synthetic() -> None:
    """[synthetic] Two runs with same seeded model produce same final masks."""
    model_a = _make_tiny_renderer(seed=2026)
    model_b = _make_tiny_renderer(seed=2026)
    result_a = run_imp_cycles(
        model=model_a,
        num_cycles=3,
        sparsity_increment=0.20,
        train_step_fn=_no_op_train_step,
    )
    result_b = run_imp_cycles(
        model=model_b,
        num_cycles=3,
        sparsity_increment=0.20,
        train_step_fn=_no_op_train_step,
    )
    assert set(result_a.final_state.mask.keys()) == set(result_b.final_state.mask.keys())
    for k in result_a.final_state.mask:
        ma = result_a.final_state.mask[k]
        mb = result_b.final_state.mask[k]
        assert ma.shape == mb.shape
        assert torch.equal(ma, mb), f"mask diverged at key {k!r}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: no-rewind path runs without error
# ─────────────────────────────────────────────────────────────────────────────


def test_no_rewind_path_runs_synthetic() -> None:
    """[synthetic] rewind_after_prune=False still produces valid CycleResults."""
    model = _make_tiny_renderer()
    result = run_imp_cycles(
        model=model,
        num_cycles=2,
        sparsity_increment=0.20,
        train_step_fn=_no_op_train_step,
        rewind_after_prune=False,
    )
    assert len(result.cycle_results) == 2
    assert result.monotone_sparsity is True


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: no silent defaults — orchestrator rejects None args
# ─────────────────────────────────────────────────────────────────────────────


def test_no_silent_defaults_synthetic() -> None:
    """[synthetic] All required args must be provided (Check 81 STRICT)."""
    model = _make_tiny_renderer()
    with pytest.raises(ValueError, match="model is required"):
        run_imp_cycles(
            model=None,
            num_cycles=3,
            sparsity_increment=0.20,
            train_step_fn=_no_op_train_step,
        )
    with pytest.raises(ValueError, match="num_cycles must be"):
        run_imp_cycles(
            model=model,
            num_cycles=0,
            sparsity_increment=0.20,
            train_step_fn=_no_op_train_step,
        )
    with pytest.raises(ValueError, match="sparsity_increment must be"):
        run_imp_cycles(
            model=model,
            num_cycles=3,
            sparsity_increment=1.5,
            train_step_fn=_no_op_train_step,
        )
    with pytest.raises(ValueError, match="train_step_fn is required"):
        run_imp_cycles(
            model=model,
            num_cycles=3,
            sparsity_increment=0.20,
            train_step_fn=None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: CLI raises NotImplementedError (no default model loader)
# ─────────────────────────────────────────────────────────────────────────────


def test_cli_raises_not_implemented_synthetic(tmp_path: Path) -> None:
    """[synthetic] CLI main raises NotImplementedError (Check 81: no silent default)."""
    from experiments.imp_cycle_runner import main
    with pytest.raises(NotImplementedError, match="SCAFFOLD"):
        main(
            [
                "--num-cycles",
                "3",
                "--sparsity-increment",
                "0.20",
                "--manifest-path",
                str(tmp_path / "m.json"),
            ]
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: _write_manifest produces valid JSON
# ─────────────────────────────────────────────────────────────────────────────


def test_write_manifest_produces_valid_json_synthetic(tmp_path: Path) -> None:
    """[synthetic] _write_manifest emits valid JSON with all expected fields."""
    import json

    model = _make_tiny_renderer()
    result = run_imp_cycles(
        model=model,
        num_cycles=2,
        sparsity_increment=0.20,
        train_step_fn=_no_op_train_step,
    )
    manifest_path = tmp_path / "imp_manifest.json"
    _write_manifest(result, manifest_path)
    payload = json.loads(manifest_path.read_text())
    assert "cycle_results" in payload
    assert "final_cycle_count" in payload
    assert "final_sparsity_target" in payload
    assert "monotone_sparsity" in payload
    assert len(payload["cycle_results"]) == 2
    assert payload["final_cycle_count"] == 2
    assert payload["monotone_sparsity"] is True
