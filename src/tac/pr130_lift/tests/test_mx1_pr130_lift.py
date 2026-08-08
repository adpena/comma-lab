from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from tac.pr130_lift.mlx_semantic_renderer import (
    MlxSemanticConfig,
    curriculum_loss_mlx,
    mlx_device_probe,
)

LIFTED = Path(__file__).resolve().parents[1] / "lifted"


def _load_lifted_semantic_module():
    sys.path.insert(0, str(LIFTED))
    spec = importlib.util.spec_from_file_location(
        "mx1_lifted_semantic_renderer_oracle",
        LIFTED / "semantic_renderer_oracle.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lifted_torch_forward_and_curriculum_are_real_behavior() -> None:
    lifted = _load_lifted_semantic_module()
    torch.manual_seed(7)
    model = lifted.SemanticTokenRenderer(width=8, blocks=2, frame_dim=4, num_pairs=6, num_tokens=5)
    tokens = torch.randint(0, 5, (2, 12, 16), dtype=torch.long)
    pair_idx = torch.tensor([0, 3], dtype=torch.long)
    out = model(tokens, pair_idx)
    assert tuple(out.shape) == (2, 3, 12, 16)
    assert float(out.detach().min()) >= 0.0
    assert float(out.detach().max()) <= 255.0
    logits = torch.randn(2, 5, 12, 16)
    loss, phase = lifted.curriculum_loss(logits, tokens, step=0, total_steps=20, ce_fraction=0.5, softplus_fraction=0.8)
    assert phase == "ce"
    assert loss.requires_grad is False
    assert float(loss) > 0.0


def test_torch_reference_deterministic_same_seed() -> None:
    lifted = _load_lifted_semantic_module()
    tokens = torch.randint(0, 5, (2, 8, 8), dtype=torch.long)
    pair_idx = torch.tensor([1, 5], dtype=torch.long)
    torch.manual_seed(11)
    a = lifted.SemanticTokenRenderer(width=8, blocks=2, frame_dim=4, num_pairs=8)
    out_a = a(tokens, pair_idx)
    torch.manual_seed(11)
    b = lifted.SemanticTokenRenderer(width=8, blocks=2, frame_dim=4, num_pairs=8)
    out_b = b(tokens, pair_idx)
    assert torch.equal(out_a, out_b)


def test_mlx_port_imports_without_eager_mlx_runtime() -> None:
    cfg = MlxSemanticConfig(width=96, blocks=4)
    assert cfg.width == 96
    assert cfg.blocks == 4
    assert callable(curriculum_loss_mlx)


def test_mlx_device_probe_is_fail_closed_or_available() -> None:
    probe = mlx_device_probe(device="cpu")
    assert probe["status"] in {"available", "blocked"}
    if probe["status"] == "blocked":
        assert probe["error"]


def test_m1_journal_resume_rewinds_active_view_without_deleting_tail(tmp_path: Path) -> None:
    from experiments.ddm_mx1_pr130_semantic_renderer import (
        M1_EVAL_JOURNAL_SCHEMA,
        _append_jsonl_durable,
        _read_active_m1_eval_rows,
    )

    journal = tmp_path / "eval.jsonl"
    _append_jsonl_durable(
        journal,
        {
            "schema": M1_EVAL_JOURNAL_SCHEMA,
            "row_kind": "segment_start",
            "resume_step": 0,
        },
    )
    for step, value in ((50, 0.2), (100, 0.3)):
        _append_jsonl_durable(
            journal,
            {
                "schema": M1_EVAL_JOURNAL_SCHEMA,
                "row_kind": "eval",
                "step": step,
                "objective_S": value,
            },
        )
    _append_jsonl_durable(
        journal,
        {
            "schema": M1_EVAL_JOURNAL_SCHEMA,
            "row_kind": "segment_start",
            "resume_step": 50,
        },
    )
    _append_jsonl_durable(
        journal,
        {
            "schema": M1_EVAL_JOURNAL_SCHEMA,
            "row_kind": "eval",
            "step": 100,
            "objective_S": 0.1,
        },
    )

    active = _read_active_m1_eval_rows(journal)
    assert [(row["step"], row["objective_S"]) for row in active] == [(50, 0.2), (100, 0.1)]
    assert len(journal.read_text(encoding="utf-8").splitlines()) == 5

    incomplete = tmp_path / "incomplete.jsonl"
    incomplete.write_text(
        json.dumps(
            {
                "schema": M1_EVAL_JOURNAL_SCHEMA,
                "row_kind": "segment_start",
                "resume_step": 0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not a complete JSONL row"):
        _read_active_m1_eval_rows(incomplete)


def test_m1_resume_cosine_holds_terminal_lr_without_horizon_jump() -> None:
    from experiments.ddm_mx1_pr130_semantic_renderer import _m1_cosine_lr

    terminal = _m1_cosine_lr(2e-7, 3249, 3250)
    assert terminal == pytest.approx(2e-9)
    assert _m1_cosine_lr(2e-7, 3250, 3250) == terminal
    assert _m1_cosine_lr(2e-7, 6499, 3250) == terminal
    assert _m1_cosine_lr(2e-7, 3250, 6500) > terminal * 50.0


def _cpu_verdict_payload(d_seg: float) -> dict:
    return {
        "schema": "ddm_mx1_torch_verdict.v1",
        "status": "passed",
        "aggregate_d_seg": d_seg,
        "pair_ids": [1, 2],
    }


def test_m1_schedule_selection_admits_one_flip_and_refuses_no_gain(tmp_path: Path) -> None:
    from experiments.ddm_mx1_pr130_semantic_renderer import run_m1_schedule_selection

    ticket = tmp_path / "ticket.json"
    ticket.write_text(
        json.dumps(
            {
                "stop_policy": {"predicate": {"one_sample_flip_S": 4.0e-6}},
            }
        ),
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(_cpu_verdict_payload(0.001)), encoding="utf-8")
    candidate.write_text(json.dumps(_cpu_verdict_payload(0.001 - 4.1e-8)), encoding="utf-8")
    args = Namespace(
        launch_ticket_path=ticket,
        init=baseline,
        resume_from=candidate,
    )
    assert run_m1_schedule_selection(args)["status"] == "passed"
    candidate.write_text(json.dumps(_cpu_verdict_payload(0.001)), encoding="utf-8")
    assert run_m1_schedule_selection(args)["status"] == "refused"


def test_m1_tail_average_materializes_mean_and_refuses_key_drift(tmp_path: Path) -> None:
    from experiments.ddm_mx1_pr130_semantic_renderer import _write_tail_average_npz

    members = []
    for index, value in enumerate((1.0, 3.0)):
        path = tmp_path / f"member{index}.npz"
        np.savez(
            path,
            **{
                "param::weight": np.asarray([value], dtype=np.float32),
                "meta::step": np.asarray([index], dtype=np.int64),
                "meta::extra_json": np.frombuffer(b"{}", dtype=np.uint8),
            },
        )
        members.append(path)
    out = _write_tail_average_npz(members, tmp_path / "average.npz", selection_extra={"k": 2})
    with np.load(out, allow_pickle=False) as payload:
        assert float(payload["param::weight"][0]) == pytest.approx(2.0)
        assert json.loads(bytes(payload["meta::extra_json"]).decode("utf-8"))["k"] == 2
    drift = tmp_path / "drift.npz"
    np.savez(drift, **{"param::other": np.asarray([1.0], dtype=np.float32)})
    with pytest.raises(ValueError, match="key sets differ"):
        _write_tail_average_npz([members[0], drift], tmp_path / "bad.npz", selection_extra={})


def test_m1_ema_update_moves_shadow_and_refuses_tree_drift() -> None:
    from experiments.ddm_mx1_pr130_semantic_renderer import (
        _derive_m1_ema_policy,
        _update_m1_ema_flat,
    )

    policy = {
        "executor": {
            "ema": {
                "updates_per_run": 3250,
                "warmup_fraction": 2000 / 3250,
                "derived_decay": 0.999,
            }
        }
    }
    assert _derive_m1_ema_policy(policy)["derived_decay"] == pytest.approx(0.999)
    policy["executor"]["ema"]["derived_decay"] = 0.997
    with pytest.raises(ValueError, match="drifted"):
        _derive_m1_ema_policy(policy)

    updated = _update_m1_ema_flat(
        {"weight": np.asarray([0.0], dtype=np.float32)},
        {"weight": np.asarray([2.0], dtype=np.float32)},
        decay=0.75,
    )
    assert float(updated["weight"][0]) == pytest.approx(0.5)
    assert float(updated["weight"][0]) != 2.0
    with pytest.raises(ValueError, match="parameter set drifted"):
        _update_m1_ema_flat(
            {"weight": np.asarray([0.0])},
            {"other": np.asarray([2.0])},
            decay=0.75,
        )


def test_m1_controlled_train_receipts_a_safe_run_wall_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.ddm_mx1_pr130_semantic_renderer as driver

    journal = tmp_path / "eval.jsonl"
    status = tmp_path / "safe_status.json"
    terminal = tmp_path / "terminal.json"
    ticket_path = tmp_path / "ticket.json"
    one_flip = 100.0 / float(120 * 384 * 512)
    for index in range(5):
        if index == 0:
            driver._append_jsonl_durable(
                journal,
                {
                    "schema": driver.M1_EVAL_JOURNAL_SCHEMA,
                    "row_kind": "segment_start",
                    "resume_step": 0,
                },
            )
        driver._append_jsonl_durable(
            journal,
            {
                "schema": driver.M1_EVAL_JOURNAL_SCHEMA,
                "row_kind": "eval",
                "step": 50 * (index + 1),
                "objective_S": 0.1,
                "loss": 1.0,
                "weights_stepped": 50 * (index + 1),
                "accepted_batch_fraction": 1.0,
            },
        )
    status.write_text(
        json.dumps(
            {
                "schema": "safe_run_status_receipt.v1",
                "status": "timeout",
                "timeout_s": 100.0,
                "elapsed_s": 100.1,
            }
        ),
        encoding="utf-8",
    )
    ticket = {
        "child": [".venv/bin/python", "tools/safe_run.py", "--", "/usr/bin/true"],
        "resume_wrapper": ["resume"],
        "stop_policy": {
            "predicate": {
                "N": 120,
                "H": 384,
                "W": 512,
                "one_sample_flip_S": one_flip,
                "eval_every_steps": 50,
                "marginal_bar_S_per_step": 8.477105034722223e-8,
                "min_eval_rows": 5,
                "window_rows": 5,
                "creep_eps_dseg_per_eval": 1e-6,
                "sustained_erosion_windows": 3,
            },
            "executor": {
                "child_argv_keys": ["child"],
                "event_free_horizon_evals": 5,
                "journal_path": str(journal),
                "decision_path": str(tmp_path / "decisions.jsonl"),
                "resume_argv_key": "resume_wrapper",
                "safety_bound_steps_by_key": {"child": 3250},
                "same_object_cpu_selection": {},
                "controller_routes": {
                    "wrapper": {
                        "child_argv_key": "child",
                        "safe_run_status_receipt_path": str(status),
                        "terminal_receipt_path": str(terminal),
                    }
                },
            },
        },
    }
    ticket_path.write_text(json.dumps(ticket), encoding="utf-8")
    monkeypatch.setattr(driver.subprocess, "run", lambda *args, **kwargs: Namespace(returncode=124))
    result = driver.run_m1_controlled_train(
        Namespace(
            launch_ticket_path=ticket_path,
            fire_argv_key="wrapper",
        )
    )
    receipt = json.loads(terminal.read_text(encoding="utf-8"))
    assert result["status"] == "passed"
    assert receipt["terminal_mode"] == "wall_clock_cap"
    assert receipt["action"] == "QUEUE_RESUME"
    assert receipt["cap_stop_receipt"]["bound_kind"] == "wall_clock_seconds"


def test_live_m1_ticket_binds_fresh_resume_controller_and_status_receipts() -> None:
    ticket_path = Path(".omx/research/ddm_m1_20260808/launch_ticket_v5_event_driven.json")
    ticket = json.loads(ticket_path.read_text(encoding="utf-8"))
    executor = ticket["stop_policy"]["executor"]
    assert set(executor["child_argv_keys"]) == {
        "argv_m1_n120_cap_saturated",
        "argv_m1_n120_cap_saturated_resume",
    }
    for key in executor["child_argv_keys"]:
        argv = ticket[key]
        assert "--status-receipt" in argv
        assert "--fire-argv-key" in argv
        assert argv[argv.index("--fire-argv-key") + 1] == key
    assert ticket["argv_m1_controller_fresh"][2:4] == ["--mode", "controlled-train"]
    assert ticket["argv_m1_controller_resume"][2:4] == ["--mode", "controlled-train"]
