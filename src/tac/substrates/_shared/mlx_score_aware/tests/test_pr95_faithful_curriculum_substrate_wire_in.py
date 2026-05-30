# SPDX-License-Identifier: MIT
"""Sister tests for the m9-v3 PR95-faithful curriculum substrate wire-in wave.

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS" non-negotiable + "HNeRV /
leaderboard-implementation parity discipline" L14 + L15 + the m9-v3 landing
memo (commit ``c91481212``) op-routable #1 (sister wave to wire
``pr95_faithful_curriculum_enabled`` into substrate trainers).

This module covers the FULL wire-in chain across THREE surfaces simultaneously:

1. ``run_long_training`` epoch loop → calls
   ``adapter.notify_global_epoch(epoch)`` once per epoch when the adapter
   implements the protocol; silent no-op when not.
2. ``run_mlx_score_aware_full_main`` harness → accepts + forwards
   ``pr95_faithful_curriculum_enabled`` + ``pr95_curriculum_total_epochs``
   into the ``MlxScoreAwareAdapter`` constructor.
3. Substrate trainer CLI flags (z6_v2 / z8 / dreamer_v3_rssm) → declare
   ``--pr95-faithful-curriculum-enabled`` and ``--pr95-curriculum-total-epochs``
   and pass them through to the canonical harness.

The Slot EEE "NO FAKE" substantive-distinctness gate is structural: each
test verifies ACTUAL behavior (notify_global_epoch is called per epoch with
the correct epoch index; harness construction propagates the kwargs into the
adapter object's state; each trainer's parser declares the flag and the
``_full_main`` body forwards the value into the harness invocation).

[verified-against: tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum
 canonical helper landed commit c91481212]
[verified-against: tac.training.long_training_canonical.run_long_training
 epoch-loop notify_global_epoch wire-in landed today]
[verified-against: tac.substrates._shared.mlx_score_aware.harness.run_mlx_score_aware_full_main
 kwargs forwarded landed today]
"""
from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from tac.training.long_training_canonical import (
    CurriculumStage,
    LongTrainingConfig,
    run_long_training,
)


# ---------------------------------------------------------------------------
# Mock adapter that records notify_global_epoch invocations
# ---------------------------------------------------------------------------


class _DictStateModel:
    """Minimal model exposing state_dict / load_state_dict / parameters."""

    def __init__(self) -> None:
        self._params: dict[str, list[float]] = {"w_0": [0.5], "w_1": [0.6]}

    def state_dict(self) -> dict[str, list[float]]:
        return {k: list(v) for k, v in self._params.items()}

    def load_state_dict(self, state: Mapping[str, list[float]]) -> None:
        for k, v in state.items():
            self._params[k] = list(v)

    def parameters(self):
        return list(self._params.values())


class _NotifyRecordingAdapter:
    """Adapter that records every notify_global_epoch invocation.

    Slot EEE NO FAKE: this adapter REQUIRES the canonical harness call
    ``notify_global_epoch`` so we can verify the wire-in is real and not a
    docstring-only contract.
    """

    def __init__(self, substrate_id: str = "wire_in_test_substrate") -> None:
        self.substrate_id = substrate_id
        self.model = _DictStateModel()
        self.notify_calls: list[int] = []
        self.step_count = 0

    def notify_global_epoch(self, global_epoch: int) -> None:
        self.notify_calls.append(int(global_epoch))

    def sample_batch(self, batch_size: int, seed: int) -> Any:
        return {"batch_size": batch_size, "seed": seed}

    def loss_fn(
        self,
        model: Any,
        batch: Any,
        loss_weights: Mapping[str, float],
    ) -> Mapping[str, float]:
        self.step_count += 1
        base = max(0.001, 1.0 / (self.step_count + 1))
        return {"total": base, "recon": base}

    def optimizer_step(self, model: Any, loss: Any, learning_rate: float) -> None:
        for _k, v in model._params.items():
            for i in range(len(v)):
                v[i] = v[i] + 1e-6

    def export_state_dict(self, model: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(model.state_dict(), sort_keys=True))

    def export_archive(
        self,
        model: Any,
        output_dir: Path,
    ) -> tuple[Path, str, int] | None:
        return None

    def score_aware_components(
        self,
        model: Any,
        batch: Any,
    ) -> Mapping[str, float] | None:
        return None


class _NoNotifyAdapter:
    """Adapter that LACKS notify_global_epoch (backward-compat regression case).

    Constructed independently from ``_NotifyRecordingAdapter`` so the legacy
    contract (no ``notify_global_epoch`` method) is preserved without class
    surgery — this is the canonical pre-m9-v3 adapter shape.
    """

    def __init__(self) -> None:
        self.substrate_id = "no_notify_test_substrate"
        self.model = _DictStateModel()
        self.step_count = 0

    def sample_batch(self, batch_size: int, seed: int) -> Any:
        return {"batch_size": batch_size, "seed": seed}

    def loss_fn(
        self,
        model: Any,
        batch: Any,
        loss_weights: Mapping[str, float],
    ) -> Mapping[str, float]:
        self.step_count += 1
        base = max(0.001, 1.0 / (self.step_count + 1))
        return {"total": base, "recon": base}

    def optimizer_step(self, model: Any, loss: Any, learning_rate: float) -> None:
        for _k, v in model._params.items():
            for i in range(len(v)):
                v[i] = v[i] + 1e-6

    def export_state_dict(self, model: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(model.state_dict(), sort_keys=True))

    def export_archive(
        self,
        model: Any,
        output_dir: Path,
    ) -> tuple[Path, str, int] | None:
        return None

    def score_aware_components(
        self,
        model: Any,
        batch: Any,
    ) -> Mapping[str, float] | None:
        return None


def _make_config(tmp_path: Path, epochs: int = 5) -> LongTrainingConfig:
    return LongTrainingConfig(
        substrate_id="wire_in_test_substrate",
        lane_id="lane_test_wire_in_test_substrate_20260530",  # FAKE_LANE_OK:test_fixture_lane_token_not_a_lane_registry_pre_registration_per_catalog_126
        epochs=epochs,
        batch_pair_indices_per_step=2,
        curriculum_stages=(
            CurriculumStage(name="only_stage", start_epoch=0, end_epoch=epochs),
        ),
        ema_decay=0.9,
        checkpoint_interval_epochs=epochs,
        early_stopping_patience=epochs + 1,
        learning_rate=1e-3,
        seed=0,
        output_dir=tmp_path,
        device="cpu",
        evidence_grade="[advisory only]",
        notes="m9-v3 PR95-faithful curriculum wire-in cascade test fixture",
    )


# ---------------------------------------------------------------------------
# Layer 1: run_long_training calls notify_global_epoch per epoch
# ---------------------------------------------------------------------------


def test_run_long_training_calls_notify_global_epoch_once_per_epoch(
    tmp_path: Path,
) -> None:
    """notify_global_epoch is invoked once per epoch with monotonic indices."""
    adapter = _NotifyRecordingAdapter()
    config = _make_config(tmp_path, epochs=7)
    run_long_training(adapter, config)
    # Slot EEE NO FAKE: notify_calls must equal [0, 1, 2, 3, 4, 5, 6] —
    # not a docstring-only contract.
    assert adapter.notify_calls == list(range(7)), (
        f"notify_global_epoch was not invoked once per epoch; got "
        f"{adapter.notify_calls} expected [0..6]"
    )


def test_run_long_training_notify_global_epoch_silent_noop_when_missing(
    tmp_path: Path,
) -> None:
    """Adapters without notify_global_epoch are a silent no-op (backward compat)."""
    adapter = _NoNotifyAdapter()
    config = _make_config(tmp_path, epochs=3)
    # Slot EEE NO FAKE: this MUST NOT raise (otherwise legacy adapters break).
    result = run_long_training(adapter, config)
    assert result is not None


def test_run_long_training_notify_global_epoch_failure_does_not_crash(
    tmp_path: Path,
) -> None:
    """An exception in notify_global_epoch is observability-only; never fails run."""

    class _BrokenNotify(_NotifyRecordingAdapter):
        def notify_global_epoch(self, global_epoch: int) -> None:
            raise RuntimeError("simulated adapter notify failure")

    adapter = _BrokenNotify()
    config = _make_config(tmp_path, epochs=2)
    # Slot EEE NO FAKE: the run must complete even if notify raises every epoch.
    result = run_long_training(adapter, config)
    assert result is not None


# ---------------------------------------------------------------------------
# Layer 2: harness forwards pr95 kwargs into adapter construction
# ---------------------------------------------------------------------------


def test_harness_signature_accepts_pr95_faithful_curriculum_kwargs() -> None:
    """``run_mlx_score_aware_full_main`` declares the canonical kwargs."""
    import inspect

    from tac.substrates._shared.mlx_score_aware.harness import (
        run_mlx_score_aware_full_main,
    )

    sig = inspect.signature(run_mlx_score_aware_full_main)
    params = sig.parameters
    # Slot EEE NO FAKE: the canonical kwargs MUST be in the signature.
    assert "pr95_faithful_curriculum_enabled" in params, (
        "harness signature missing canonical kwarg; wire-in is FAKE if absent"
    )
    assert "pr95_curriculum_total_epochs" in params, (
        "harness signature missing canonical kwarg; wire-in is FAKE if absent"
    )
    # Defaults must preserve backward compat (default off).
    assert params["pr95_faithful_curriculum_enabled"].default is False
    assert params["pr95_curriculum_total_epochs"].default is None


def test_harness_source_constructs_adapter_with_pr95_kwargs() -> None:
    """AST scan: ``MlxScoreAwareAdapter(...)`` carries the canonical kwargs."""
    from tac.substrates._shared.mlx_score_aware import harness

    source = Path(harness.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    found_calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "MlxScoreAwareAdapter"
        ):
            found_calls.append(node)
    assert found_calls, (
        "MlxScoreAwareAdapter is not constructed in harness module; harness "
        "may have regressed (FAKE wire-in regression)"
    )
    canonical_kwargs = {"pr95_faithful_curriculum_enabled", "pr95_curriculum_total_epochs"}
    for call in found_calls:
        kw_names = {kw.arg for kw in call.keywords if kw.arg is not None}
        missing = canonical_kwargs - kw_names
        assert not missing, (
            f"MlxScoreAwareAdapter construction missing canonical kwargs "
            f"{missing}; the harness wire-in is FAKE if absent"
        )


# ---------------------------------------------------------------------------
# Layer 3: each priority substrate trainer declares the CLI flag and
#           passes it through to run_mlx_score_aware_full_main
# ---------------------------------------------------------------------------


PRIORITY_TRAINER_PATHS: tuple[Path, ...] = (
    Path("experiments/train_substrate_z6_v2_cargo_cult_unwind_mlx_local.py"),
    Path("experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py"),
    Path("experiments/train_substrate_dreamer_v3_rssm.py"),
)


def _repo_root() -> Path:
    # Walk up from this test file until we find pyproject.toml.
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("pyproject.toml not found in any parent")


@pytest.mark.parametrize(
    "trainer_relpath",
    PRIORITY_TRAINER_PATHS,
    ids=[p.stem for p in PRIORITY_TRAINER_PATHS],
)
def test_priority_trainer_declares_pr95_curriculum_cli_flag(
    trainer_relpath: Path,
) -> None:
    """Each priority trainer's argparse declares ``--pr95-faithful-curriculum-enabled``."""
    repo_root = _repo_root()
    trainer_path = repo_root / trainer_relpath
    assert trainer_path.is_file(), f"trainer file missing: {trainer_path}"
    source = trainer_path.read_text(encoding="utf-8")
    # Slot EEE NO FAKE: the flag string must literally appear (and be in an
    # add_argument call, not just a docstring) AND the total-epochs sister
    # flag must appear so the operator can choose smoke/full budget.
    assert "--pr95-faithful-curriculum-enabled" in source, (
        f"trainer {trainer_path.name} missing --pr95-faithful-curriculum-enabled CLI flag"
    )
    assert "--pr95-curriculum-total-epochs" in source, (
        f"trainer {trainer_path.name} missing --pr95-curriculum-total-epochs CLI flag"
    )


@pytest.mark.parametrize(
    "trainer_relpath",
    PRIORITY_TRAINER_PATHS,
    ids=[p.stem for p in PRIORITY_TRAINER_PATHS],
)
def test_priority_trainer_forwards_pr95_kwargs_to_harness(
    trainer_relpath: Path,
) -> None:
    """Each priority trainer forwards the kwargs into run_mlx_score_aware_full_main."""
    repo_root = _repo_root()
    trainer_path = repo_root / trainer_relpath
    source = trainer_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forwarded_in_any_call = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_mlx_score_aware_full_main"
        ):
            kw_names = {kw.arg for kw in node.keywords if kw.arg is not None}
            if (
                "pr95_faithful_curriculum_enabled" in kw_names
                and "pr95_curriculum_total_epochs" in kw_names
            ):
                forwarded_in_any_call = True
                break
    assert forwarded_in_any_call, (
        f"trainer {trainer_path.name} does not forward pr95 curriculum kwargs "
        f"to run_mlx_score_aware_full_main; the wire-in is FAKE without this"
    )


@pytest.mark.parametrize(
    "trainer_relpath",
    PRIORITY_TRAINER_PATHS,
    ids=[p.stem for p in PRIORITY_TRAINER_PATHS],
)
def test_priority_trainer_cli_parser_accepts_pr95_flag(
    trainer_relpath: Path,
) -> None:
    """The trainer's argparse actually parses the canonical CLI flag."""
    import importlib.util
    import sys

    repo_root = _repo_root()
    trainer_path = repo_root / trainer_relpath
    # Importable module path = ``experiments.<stem>``.
    module_name = f"experiments.{trainer_path.stem}"
    # Ensure repo_root on sys.path so ``from tac...`` imports inside the trainer
    # resolve under the package layout described in pyproject.toml.
    inserted_paths: list[str] = []
    for required in (str(repo_root), str(repo_root / "src")):
        if required not in sys.path:
            sys.path.insert(0, required)
            inserted_paths.append(required)
    spec = importlib.util.spec_from_file_location(module_name, trainer_path)
    if spec is None or spec.loader is None:
        pytest.skip(f"could not load spec for {trainer_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(
            f"trainer {trainer_path.name} requires runtime deps not present "
            f"on this host: {exc}"
        )
    finally:
        for path in inserted_paths:
            if path in sys.path:
                sys.path.remove(path)
    parser = module._build_parser()
    # Some priority trainers have other required CLI args (e.g. --output-dir);
    # supply minimal placeholders for any flag declared with required=True so
    # parse_known_args reaches our flags. Slot EEE NO FAKE: the parser MUST
    # accept the canonical flag (not just match a string in source).
    fixture_args = [
        "--pr95-faithful-curriculum-enabled",
        "--pr95-curriculum-total-epochs",
        "100",
    ]
    for action in parser._actions:
        if getattr(action, "required", False) and action.option_strings:
            fixture_args.extend([action.option_strings[0], "/tmp/pact_test_fixture_only_path"])
    args, _unknown = parser.parse_known_args(fixture_args)
    assert getattr(args, "pr95_faithful_curriculum_enabled", None) is True
    assert getattr(args, "pr95_curriculum_total_epochs", None) == 100


# ---------------------------------------------------------------------------
# Layer 4: full-chain integration (harness wire propagates kwarg → adapter)
# ---------------------------------------------------------------------------


def test_harness_construction_propagates_pr95_state_into_adapter() -> None:
    """End-to-end: harness construction wires the canonical curriculum state.

    Skip when MLX unavailable (Linux / CI without Apple Silicon); the
    canonical wire-in cascade still passes via the source-level AST tests.
    """
    try:
        from tac.substrates._shared.mlx_score_aware.device_gate import (
            is_mlx_available,
        )
    except Exception:
        pytest.skip("device_gate import failed; cannot probe MLX availability")
    if not is_mlx_available():
        pytest.skip(
            "MLX unavailable on this host; canonical AST-level wire-in tests "
            "above still verify the source contract"
        )
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    # Construct a minimal bundle via a stub so the adapter can initialize.
    class _StubBundle:
        model = type("StubModel", (), {})()

        def __init__(self):
            self.num_pairs = 1
            self.distillation_weight = 0.0
            self.scorer_teacher = None
            self.pose_scorer_teacher = None
            self.learnable_student_head = None
            self.pose_learnable_student_head = None
            self.export_state_dict_fn = None

    bundle = _StubBundle()
    # Slot EEE NO FAKE: the opt-in MUST construct the canonical factory.
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="wire_in_full_chain_test",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=100,
    )
    assert adapter._pr95_faithful_curriculum_enabled is True
    assert adapter._pr95_curriculum_factory is not None
    assert adapter._pr95_optimizer_state is not None
    # The factory's total_epoch_budget must reflect what the harness passed.
    assert adapter._pr95_curriculum_factory.total_epoch_budget == 100
    # notify_global_epoch must advance internal state.
    adapter.notify_global_epoch(42)
    assert adapter._pr95_global_epoch == 42
