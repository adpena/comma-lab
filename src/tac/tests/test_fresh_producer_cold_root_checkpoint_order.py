from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TRAINER = REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"


def _run_train_source() -> str:
    source = TRAINER.read_text()
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "run_train"
    )
    segment = ast.get_source_segment(source, node)
    assert segment is not None
    return segment


def test_fresh_cold_root_follows_every_captured_loop_state_and_precedes_training() -> None:
    source = _run_train_source()
    cold_root = source.index('stage_tag="stageColdRoot"')

    required_predecessors = (
        "recent_losses: list[float] = []",
        "hardness_prob = None",
        '_resume_registry.register("rng_streams"',
        '_resume_registry.register("closed_loop"',
        '_resume_registry.register("tau_advance"',
        '_resume_registry.register("evt_curriculum"',
        '_resume_registry.register("birth_completion"',
    )
    for marker in required_predecessors:
        assert source.index(marker) < cold_root, marker

    assert cold_root < source.index(
        "for ep in range(start_epoch, args.epochs + 1):"
    )
    assert source.count('stage_tag="stageColdRoot"') == 1
