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
        '_live: dict[str, Any] = {"ep_acc": 0',
    )
    for marker in required_predecessors:
        assert source.index(marker) < cold_root, marker

    assert cold_root < source.index(
        "for ep in range(start_epoch, args.epochs + 1):"
    )
    assert source.count('stage_tag="stageColdRoot"') == 1


def test_direct_checkpoint_closure_loads_have_prior_textual_bindings() -> None:
    """Diagnostic for direct loads, not a transitive definite-assignment proof."""

    source = TRAINER.read_text()
    tree = ast.parse(source)
    run_train = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "run_train"
    )
    checkpoint = next(
        item
        for item in run_train.body
        if isinstance(item, ast.FunctionDef) and item.name == "_do_checkpoint"
    )
    cold_root_lines = [
        node.lineno
        for node in ast.walk(run_train)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name)
            and target.id == "_cold_root_checkpoint"
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
    ]
    assert len(cold_root_lines) == 1
    cold_root_line = cold_root_lines[0]

    first_outer_binding: dict[str, int] = {}

    def record(name: str, line: int) -> None:
        first_outer_binding[name] = min(first_outer_binding.get(name, line), line)

    def visit_outer(node: ast.AST) -> None:
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        ):
            if hasattr(node, "name"):
                record(node.name, node.lineno)
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                record(alias.asname or alias.name.split(".", maxsplit=1)[0], node.lineno)
            return
        if isinstance(node, ast.Name) and isinstance(
            node.ctx, (ast.Store, ast.Param)
        ):
            record(node.id, node.lineno)
        for child in ast.iter_child_nodes(node):
            visit_outer(child)

    for statement in run_train.body:
        visit_outer(statement)
    for argument in (
        *run_train.args.posonlyargs,
        *run_train.args.args,
        *run_train.args.kwonlyargs,
    ):
        record(argument.arg, run_train.lineno)
    if run_train.args.vararg is not None:
        record(run_train.args.vararg.arg, run_train.lineno)
    if run_train.args.kwarg is not None:
        record(run_train.args.kwarg.arg, run_train.lineno)

    checkpoint_loads = {
        node.id
        for node in ast.walk(checkpoint)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    late_dependencies = {
        name: first_outer_binding[name]
        for name in checkpoint_loads
        if first_outer_binding.get(name, 0) > cold_root_line
    }
    assert late_dependencies == {}


def test_live_binding_unconditionally_dominates_cold_root() -> None:
    source = TRAINER.read_text()
    tree = ast.parse(source)
    run_train = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "run_train"
    )
    live_bindings = [
        item
        for item in run_train.body
        if isinstance(item, ast.AnnAssign)
        and isinstance(item.target, ast.Name)
        and item.target.id == "_live"
    ]
    assert len(live_bindings) == 1
    cold_root_if = next(
        item
        for item in run_train.body
        if isinstance(item, ast.If)
        and any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_cold_root_checkpoint"
                for target in node.targets
            )
            for node in ast.walk(item)
        )
    )
    assert live_bindings[0].lineno < cold_root_if.lineno


def test_cold_root_causal_boundary_cannot_claim_weights_stepped() -> None:
    """Every checkpoint writer that records a causal boundary with a
    boundary kind must refuse weights_stepped on cold_root. Selected by
    behavior (calls _record_causal_boundary AND handles
    causal_boundary_kind), not by source position: _do_checkpoint is now a
    G111-barrier wrapper delegating to _do_checkpoint_impl, so a positional
    window anchored at the wrapper misses the guard."""

    source = TRAINER.read_text()
    tree = ast.parse(source)
    run_train = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "run_train"
    )
    writers = []
    for item in run_train.body:
        if not isinstance(item, ast.FunctionDef):
            continue
        if item.name == "_record_causal_boundary":
            continue
        segment = ast.get_source_segment(source, item)
        if segment is None:
            continue
        if (
            "_record_causal_boundary(" in segment
            and "causal_boundary_kind" in segment
        ):
            writers.append((item.name, segment))
    assert writers, (
        "no nested run_train function both calls _record_causal_boundary and"
        " handles causal_boundary_kind — the checkpoint writer moved; rebind"
    )
    for name, segment in writers:
        assert 'if causal_boundary_kind == "cold_root"' in segment, name
        assert "weights_stepped=(\n                False" in segment, name
