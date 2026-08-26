from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO / "tools/probe_onpolicy_costate_matched_window.py"


def _load_tool():
    name = "_test_probe_onpolicy_costate_matched_window"
    spec = importlib.util.spec_from_file_location(name, TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_tool()


def _checkpoint(anchor_frame: torch.Tensor, anchor_costate: torch.Tensor, *, sequence: int = 0):
    return probe.build_probe_checkpoint(
        run_contract_sha256="a" * 64,
        stage="collection",
        next_step=sequence,
        anchor_frame=anchor_frame,
        anchor_costate=anchor_costate,
        state={"sequence": sequence},
    )


def test_exact_branch_schedule_is_applied_identically_to_both_branches() -> None:
    exact_theta = torch.tensor([3.0, 4.0])
    exact_gradient = torch.tensor([1.0, 0.0])
    surrogate_gradient = torch.tensor([0.0, 2.0])
    norm = probe.derive_common_step_norm(exact_theta, step_fraction=0.1)
    record = probe.matched_schedule_record(0, norm)

    exact_candidate = probe.candidate_at_common_norm(exact_theta, exact_gradient, record["exact_step_norm"])
    surrogate_candidate = probe.candidate_at_common_norm(exact_theta, surrogate_gradient, record["surrogate_step_norm"])

    assert record["selected_by"] == "exact_branch"
    assert record["identical_norm_predicate"]
    assert record["exact_step_norm"] == record["surrogate_step_norm"] == pytest.approx(0.5)
    assert torch.linalg.vector_norm(exact_candidate - exact_theta).item() == pytest.approx(norm)
    assert torch.linalg.vector_norm(surrogate_candidate - exact_theta).item() == pytest.approx(norm)


def test_joint_descent_predicate_requires_both_exact_hard_metrics() -> None:
    current = {"ce": 1.0, "d_seg": 0.2, "d_pose": 3.0}
    passing = {"ce": 0.9, "d_seg": 0.2, "d_pose": 2.9}
    pose_failure = {"ce": 0.9, "d_seg": 0.19, "d_pose": 3.1}
    assert all(probe.joint_descent_predicates(current, passing).values())
    assert not all(probe.joint_descent_predicates(current, pose_failure).values())


def test_resume_restores_anchor_tensors_without_teacher_callback(tmp_path: Path) -> None:
    store = probe.TwoSlotCheckpointStore(tmp_path)
    anchor_frame = torch.arange(18, dtype=torch.float32).reshape(1, 3, 2, 3)
    anchor_costate = -anchor_frame / anchor_frame.numel()
    record = store.save_slot(_checkpoint(anchor_frame, anchor_costate, sequence=7), stage="collection", sequence=7)

    # The restore API deliberately has no teacher/provider argument.  A resume
    # therefore cannot hide an unscheduled exact call behind dependency injection.
    restored = probe.restore_probe_checkpoint(store, Path(record.path), expected_run_contract_sha256="a" * 64)

    torch.testing.assert_close(restored["anchor_frame"], anchor_frame, rtol=0.0, atol=0.0)
    torch.testing.assert_close(restored["anchor_costate"], anchor_costate, rtol=0.0, atol=0.0)
    assert restored["next_step"] == 7


def test_timing_ledger_keeps_measured_surfaces_and_operator_reference_disjoint() -> None:
    ledger = probe.TimingLedger()
    ledger.add("exact_forward_only", 1.0)
    ledger.add("exact_costate_forward_backward", 2.0)
    ledger.add("anchor_fit", 3.0)
    ledger.add("surrogate_inference", 0.25)
    ledger.add("renderer_vjp_exact_control", 0.5)
    ledger.add("candidate_update_exact_control", 0.01)
    ledger.add("candidate_update_surrogate_target", 0.02)
    ledger.add("exact_window_operational_step", 2.5)
    ledger.add("surrogate_window_operational_step", 0.75)
    summary = ledger.summary()

    assert summary["measured"]["exact_forward_only"]["mean_seconds"] == 1.0
    assert summary["measured"]["exact_costate_forward_backward"]["mean_seconds"] == 2.0
    assert summary["measured"]["anchor_fit"]["mean_seconds"] == 3.0
    assert summary["measured"]["surrogate_inference"]["mean_seconds"] == 0.25
    assert summary["measured"]["renderer_vjp_exact_control"]["mean_seconds"] == 0.5
    assert summary["measured"]["candidate_update_exact_control"]["mean_seconds"] == 0.01
    assert summary["measured"]["candidate_update_surrogate_target"]["mean_seconds"] == 0.02
    assert summary["measured"]["exact_window_operational_step"]["mean_seconds"] == 2.5
    assert summary["measured"]["surrogate_window_operational_step"]["mean_seconds"] == 0.75
    assert summary["operator_supplied_reference"]["exact_forward_ms"] == 1656.0
    assert not summary["operator_supplied_reference"]["included_in_measured_samples"]


def test_complete_operational_step_rejects_asymmetric_missing_component() -> None:
    assert probe.complete_operational_step_seconds(2.0, 0.125) == pytest.approx(2.125)
    with pytest.raises(probe.ProbeError, match="finite and nonnegative"):
        probe.complete_operational_step_seconds(2.0, -0.125)


def test_both_matched_loops_include_candidate_update_in_complete_step_timer() -> None:
    tree = ast.parse(TOOL_PATH.read_text(encoding="utf-8"))
    assignments = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target = node.target
        else:
            continue
        call = node.value
        if not isinstance(target, ast.Name) or target.id != "operational_elapsed":
            continue
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        if call.func.id == "complete_operational_step_seconds":
            assignments.append(call)

    assert len(assignments) == 2
    assert [
        [ast.unparse(argument) for argument in call.args] for call in assignments
    ] == [
        ["provider_path_elapsed", "candidate_elapsed"],
        ["provider_path_elapsed", "candidate_elapsed"],
    ]


def test_surrogate_cache_warmup_is_accounted_before_operational_timing() -> None:
    source = Path(probe.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    surrogate_window = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "surrogate_window"
    )
    loop = next(node for node in ast.walk(surrogate_window) if isinstance(node, ast.For))
    statements = [ast.unparse(statement) for statement in loop.body]
    warmup_call = next(
        index
        for index, statement in enumerate(statements)
        if "timing.add" in statement and "surrogate_matched_warmup_exact_forward" in statement
    )
    warmup_accounting = next(
        index
        for index, statement in enumerate(statements)
        if "teacher_counts" in statement
        and "timing_only_segnet_forwards" in statement
        and "+= 1" in statement
    )
    operational_start = next(
        index
        for index, statement in enumerate(statements)
        if statement == "operational_started = time.perf_counter()"
    )
    assert any("with torch.no_grad():" in statement and "segnet(" in statement for statement in statements)
    assert warmup_call < warmup_accounting < operational_start


def test_blocked_isolated_timing_reason_does_not_reintroduce_terminal_floor() -> None:
    source = Path(probe.__file__).read_text(encoding="utf-8")
    assert "measured prefix left one or more isolated matched timing surfaces empty" in source
    assert "terminal floor left one or more isolated matched timing surfaces empty" not in source


def test_two_slot_atomic_custody_and_preserved_stage_checkpoints(tmp_path: Path) -> None:
    store = probe.TwoSlotCheckpointStore(tmp_path)
    frame = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    costate = torch.ones_like(frame)
    first = store.save_slot(_checkpoint(frame, costate, sequence=0), stage="collection", sequence=0)
    second = store.save_slot(_checkpoint(frame, costate, sequence=1), stage="collection", sequence=1)
    third = store.save_slot(_checkpoint(frame, costate, sequence=2), stage="collection", sequence=2)

    assert Path(first.path) == Path(third.path)
    assert Path(second.path) != Path(third.path)
    latest = json.loads((tmp_path / "checkpoint_latest.json").read_text(encoding="utf-8"))
    assert latest["sequence"] == 2
    assert latest["sha256"] == third.sha256
    loaded = store.load(Path(third.path), expected_run_contract_sha256="a" * 64)
    assert loaded["state"]["sequence"] == 2

    preserved0 = store.preserve_stage(_checkpoint(frame, costate, sequence=2), stage="collection", sequence=2)
    preserved1 = store.preserve_stage(
        probe.build_probe_checkpoint(
            run_contract_sha256="a" * 64,
            stage="matched_window",
            next_step=5,
            anchor_frame=frame,
            anchor_costate=costate,
            state={"sequence": 5},
        ),
        stage="matched_window",
        sequence=5,
    )
    assert Path(preserved0.path).is_file() and Path(preserved1.path).is_file()
    assert Path(preserved0.path) != Path(preserved1.path)
    with pytest.raises(probe.ProbeError, match="overwrite preserved"):
        store.preserve_stage(_checkpoint(frame, costate, sequence=2), stage="collection", sequence=2)

    # A torn/corrupt slot cannot pass the byte sidecar.
    with Path(third.path).open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(probe.ProbeError, match="byte custody mismatch"):
        store.load(Path(third.path), expected_run_contract_sha256="a" * 64)


def test_defaults_follow_named_derivations_and_every_tunable_has_a_cli_surface() -> None:
    args = probe._parser().parse_args(
        [
            "--output-dir",
            "/tmp/out",
            "--storage-plan",
            "/tmp/storage.json",
            "--step-fraction",
            "0.01",
            "--learning-rate",
            "0.002",
        ]
    )
    assert args.target_teacher_skip_fraction == 0.95
    assert args.window_steps == 5
    assert args.collection_steps == args.window_steps
    assert args.branch_kernel_sizes == (3, 5)
    assert args.hidden_channels == 16
    assert args.ema_decay == pytest.approx(0.8)
    assert args.optimizer_steps_per_label == len(args.branch_kernel_sizes)


def test_typed_contract_allows_decisive_k20_horizon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / "experiments" / "results" / "out"
    output_dir.mkdir(parents=True)
    storage = tmp_path / "storage.json"
    storage.write_text("{}", encoding="utf-8")
    args = probe._parser().parse_args(
        [
            "--output-dir",
            str(output_dir),
            "--storage-plan",
            str(storage),
            "--window-steps",
            "20",
            "--step-fraction",
            "0.01",
            "--learning-rate",
            "0.002",
        ]
    )
    config = probe.MatchedProbeConfig(
        regime=args.regime,
        seed=args.seed,
        collection_steps=args.collection_steps,
        window_steps=args.window_steps,
        optimizer_steps_per_label=args.optimizer_steps_per_label,
        step_fraction=args.step_fraction,
        learning_rate=args.learning_rate,
        ema_decay=args.ema_decay,
        hidden_channels=args.hidden_channels,
        branch_kernel_sizes=tuple(args.branch_kernel_sizes),
        target_teacher_skip_fraction=args.target_teacher_skip_fraction,
    )
    monkeypatch.setattr(
        probe,
        "_validate_storage_plan",
        lambda *_args: {
            "selected_workload_root": str(output_dir),
            "requested_bytes": probe.DERIVED_MIN_STORAGE_BYTES,
        },
    )
    monkeypatch.setattr(probe, "_source_custody", lambda: {})
    monkeypatch.setattr(probe, "_input_custody", lambda _config: {})
    contract = probe._run_contract(args, config)
    assert contract["payload"]["config"]["window_steps"] == config.anchor_cadence == 20


def test_line_search_exhaustion_is_blocked_not_terminal_floor() -> None:
    assert probe.classify_exact_window_completion(
        certified_zero_renderer_gradient=False,
        observed_updates=0,
        requested_updates=20,
    ) == "LINE_SEARCH_BLOCKED_AFTER_MEASURED_PREFIX"
    assert probe.classify_exact_window_completion(
        certified_zero_renderer_gradient=False,
        observed_updates=3,
        requested_updates=20,
    ) == "LINE_SEARCH_BLOCKED_AFTER_MEASURED_PREFIX"


def test_only_exact_zero_renderer_gradient_is_terminal_floor() -> None:
    assert probe.classify_exact_window_completion(
        certified_zero_renderer_gradient=True,
        observed_updates=0,
        requested_updates=20,
    ) == "CERTIFIED_ZERO_RENDERER_GRADIENT"


def test_source_bundle_is_bound_to_run_contract_custody(tmp_path: Path) -> None:
    bundle = probe._materialize_source_bundle(tmp_path)
    expected = probe._source_custody()
    probe._verify_source_bundle(tmp_path, bundle, expected)
    first = next(iter(bundle.values()))
    (tmp_path / first["path"]).write_bytes(b"tamper")
    with pytest.raises(probe.ProbeError, match="byte custody"):
        probe._verify_source_bundle(tmp_path, bundle, expected)
