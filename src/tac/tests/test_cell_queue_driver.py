"""Tests for ``tools/cell_queue_driver.py`` -- the governed ordered-cell queue driver.

Coverage: queue-spec parsing and its two structural refusals (duplicate cell_id, duplicate
done-receipt -- the ng1/ng2 collision), falsifier parsing, the SEAL LAW check in both directions
(content-identical + re-rooted passes; content drift and un-re-rooted paths refuse -- the defect
that killed ng2's launch in 4 s), launcher-argv binding, milestone reads against a named control,
pre-registered falsifier evaluation at every state, the composed per-cell verdict, planning with
admission wired in, and the CLI including ``--dry-run`` (which must never claim or launch).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO / "tools" / "cell_queue_driver.py"

_spec = importlib.util.spec_from_file_location("cell_queue_driver_under_test", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
q = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = q
_spec.loader.exec_module(q)


# ── fixtures ────────────────────────────────────────────────────────────────────────────────────


def _pin(sha: str = "a" * 64, size: int = 10, path: str = "/tree/x.py") -> dict:
    return {"sha256": sha, "bytes": size, "path": path}


def _cell_payload(root: Path, name: str = "cell_a", **overrides) -> dict:
    sealed = root / f"{name}.sealed.json"
    authorized = root / f"{name}.authorized.json"
    out_dir = root / "launch" / name
    payload = {
        "cell_id": name,
        "sealed_config": str(sealed),
        "sealed_tree": str(root / "tree"),
        "authorized_config": str(authorized),
        "done_receipt": f"{name}_DONE.json",
        "scorer_lane_prefix": f"{name}_scorer",
        "metal_lane_prefix": f"{name}_metal",
        "measured_peak_rss_gib": 4.0,
        "control_run_dir": None,
        "control_label": None,
        "milestones": [1000, 2000],
        "falsifiers": [
            {
                "name": "flip_too_high_at_2k",
                "at_step": 2000,
                "metric": "objective.seg_expected_flip_realized",
                "op": "gt",
                "threshold": 0.5,
            }
        ],
        "notes": "test cell",
        "launcher_argv": [
            "python", "launch_detached_process.py",
            "--output-dir", str(out_dir),
            "--done-receipt", f"{name}_DONE.json",
            "--", "python", "trainer.py", "run-config", str(authorized),
        ],
    }
    payload.update(overrides)
    return payload


def _write_spec(root: Path, cells: list[dict]) -> Path:
    path = root / "queue.json"
    path.write_text(
        json.dumps({"schema": q.QUEUE_SPEC_SCHEMA, "cells": cells}, indent=2), encoding="utf-8"
    )
    return path


def _write_sealed(root: Path, name: str, pins: dict) -> Path:
    path = root / f"{name}.sealed.json"
    path.write_text(
        json.dumps({"cell_id": name, "total_steps": 5000, "source_pins": pins}, indent=2),
        encoding="utf-8",
    )
    return path


def _history(run_dir: Path, rows: list[dict]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "history.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _cell(root: Path, name: str = "cell_a", **overrides) -> q.QueuedCell:
    return q.QueuedCell.from_mapping(_cell_payload(root, name, **overrides))


# ── queue spec ──────────────────────────────────────────────────────────────────────────────────


class TestQueueSpec:
    def test_loads_an_ordered_queue(self, tmp_path):
        spec = _write_spec(tmp_path, [_cell_payload(tmp_path, "a"), _cell_payload(tmp_path, "b")])
        cells = q.load_queue_spec(spec)
        assert [c.cell_id for c in cells] == ["a", "b"]
        assert cells[0].milestones == (1000, 2000)
        assert cells[0].falsifiers[0].threshold == 0.5

    def test_refuses_duplicate_done_receipt(self, tmp_path):
        """The ng1/ng2 collision: the launcher refuses to overwrite an existing receipt."""
        spec = _write_spec(
            tmp_path,
            [
                _cell_payload(tmp_path, "a", done_receipt="DONE.json"),
                _cell_payload(tmp_path, "b", done_receipt="DONE.json"),
            ],
        )
        with pytest.raises(q.QueueRefusal) as exc:
            q.load_queue_spec(spec)
        assert exc.value.reason == "DUPLICATE_DONE_RECEIPT"

    def test_refuses_duplicate_cell_id(self, tmp_path):
        spec = _write_spec(
            tmp_path,
            [
                _cell_payload(tmp_path, "a"),
                _cell_payload(tmp_path, "a", done_receipt="other.json"),
            ],
        )
        with pytest.raises(q.QueueRefusal) as exc:
            q.load_queue_spec(spec)
        assert exc.value.reason == "DUPLICATE_CELL_ID"

    def test_refuses_wrong_schema(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"schema": "other.v1", "cells": []}), encoding="utf-8")
        with pytest.raises(q.QueueRefusal) as exc:
            q.load_queue_spec(path)
        assert exc.value.reason == "QUEUE_SPEC_SCHEMA"

    def test_refuses_empty_queue(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"schema": q.QUEUE_SPEC_SCHEMA, "cells": []}), encoding="utf-8")
        with pytest.raises(q.QueueRefusal) as exc:
            q.load_queue_spec(path)
        assert exc.value.reason == "QUEUE_SPEC_EMPTY"

    def test_refuses_unreadable_spec(self, tmp_path):
        with pytest.raises(q.QueueRefusal) as exc:
            q.load_queue_spec(tmp_path / "absent.json")
        assert exc.value.reason == "QUEUE_SPEC_UNREADABLE"

    def test_refuses_missing_required_field(self, tmp_path):
        payload = _cell_payload(tmp_path, "a")
        del payload["measured_peak_rss_gib"]
        spec = _write_spec(tmp_path, [payload])
        with pytest.raises(q.QueueRefusal) as exc:
            q.load_queue_spec(spec)
        assert exc.value.reason == "CELL_SPEC_SHAPE"

    def test_refuses_unknown_falsifier_op(self, tmp_path):
        payload = _cell_payload(tmp_path, "a")
        payload["falsifiers"][0]["op"] = "approximately"
        spec = _write_spec(tmp_path, [payload])
        with pytest.raises(q.QueueRefusal) as exc:
            q.load_queue_spec(spec)
        assert exc.value.reason == "FALSIFIER_OP"

    def test_refuses_malformed_falsifier(self, tmp_path):
        payload = _cell_payload(tmp_path, "a")
        del payload["falsifiers"][0]["threshold"]
        spec = _write_spec(tmp_path, [payload])
        with pytest.raises(q.QueueRefusal) as exc:
            q.load_queue_spec(spec)
        assert exc.value.reason == "FALSIFIER_SHAPE"

    def test_falsifier_describe_is_readable(self, tmp_path):
        cell = _cell(tmp_path)
        assert "flip_too_high_at_2k" in cell.falsifiers[0].describe()
        assert "@2000" in cell.falsifiers[0].describe()


# ── the SEAL LAW ────────────────────────────────────────────────────────────────────────────────


class TestSealVerification:
    def _prepare(self, tmp_path, config_pins, tree_pins, monkeypatch):
        (tmp_path / "tree").mkdir(exist_ok=True)
        _write_sealed(tmp_path, "cell_a", config_pins)
        monkeypatch.setattr(q.reseal, "verify_inputs_inside", lambda tree: tree_pins)
        return _cell(tmp_path)

    def test_passes_when_content_identical_and_rerooted(self, tmp_path, monkeypatch):
        pins = {"x": _pin(path="/tree/x.py")}
        cell = self._prepare(tmp_path, pins, {"x": _pin(path="/tree/x.py")}, monkeypatch)
        report = q.verify_sealed_config_in_its_tree(cell)
        assert report["content_identical"] is True
        assert report["paths_rooted_in_firing_tree"] is True
        assert report["pins_total"] == 1

    def test_refuses_paths_not_rerooted(self, tmp_path, monkeypatch):
        """THE ng2 DEFECT: every sha and byte identical, only the paths differ."""
        cell = self._prepare(
            tmp_path,
            {"x": _pin(path="/Users/adpena/Projects/pact/x.py")},
            {"x": _pin(path="/sealed/tree/x.py")},
            monkeypatch,
        )
        with pytest.raises(q.QueueRefusal) as exc:
            q.verify_sealed_config_in_its_tree(cell)
        assert exc.value.reason == "PIN_PATHS_NOT_REROOTED"
        assert "ddm_reseal_pins_inside_sealed_tree" in str(exc.value)

    def test_refuses_content_drift(self, tmp_path, monkeypatch):
        cell = self._prepare(
            tmp_path, {"x": _pin(sha="a" * 64)}, {"x": _pin(sha="b" * 64)}, monkeypatch
        )
        with pytest.raises(q.QueueRefusal) as exc:
            q.verify_sealed_config_in_its_tree(cell)
        assert exc.value.reason == "PIN_CONTENT_DRIFT"

    def test_refuses_pin_key_set_mismatch(self, tmp_path, monkeypatch):
        cell = self._prepare(tmp_path, {"x": _pin()}, {"y": _pin()}, monkeypatch)
        with pytest.raises(q.QueueRefusal) as exc:
            q.verify_sealed_config_in_its_tree(cell)
        assert exc.value.reason == "PIN_KEY_SET_MISMATCH"

    def test_refuses_missing_sealed_config(self, tmp_path, monkeypatch):
        (tmp_path / "tree").mkdir(exist_ok=True)
        with pytest.raises(q.QueueRefusal) as exc:
            q.verify_sealed_config_in_its_tree(_cell(tmp_path))
        assert exc.value.reason == "SEALED_CONFIG_MISSING"

    def test_refuses_missing_sealed_tree(self, tmp_path):
        _write_sealed(tmp_path, "cell_a", {"x": _pin()})
        with pytest.raises(q.QueueRefusal) as exc:
            q.verify_sealed_config_in_its_tree(_cell(tmp_path))
        assert exc.value.reason == "SEALED_TREE_MISSING"

    def test_refuses_config_without_pins(self, tmp_path, monkeypatch):
        (tmp_path / "tree").mkdir(exist_ok=True)
        (tmp_path / "cell_a.sealed.json").write_text(json.dumps({"cell_id": "cell_a"}))
        with pytest.raises(q.QueueRefusal) as exc:
            q.verify_sealed_config_in_its_tree(_cell(tmp_path))
        assert exc.value.reason == "SEALED_CONFIG_NO_PINS"

    def test_surfaces_verify_inputs_failure(self, tmp_path, monkeypatch):
        (tmp_path / "tree").mkdir(exist_ok=True)
        _write_sealed(tmp_path, "cell_a", {"x": _pin()})

        def _boom(tree):
            raise q.reseal.ResealError("no interpreter")

        monkeypatch.setattr(q.reseal, "verify_inputs_inside", _boom)
        with pytest.raises(q.QueueRefusal) as exc:
            q.verify_sealed_config_in_its_tree(_cell(tmp_path))
        assert exc.value.reason == "SEALED_TREE_VERIFY_INPUTS"


# ── launcher argv ───────────────────────────────────────────────────────────────────────────────


class TestLauncherArgv:
    def test_accepts_a_bound_argv(self, tmp_path):
        report = q.verify_launcher_argv(_cell(tmp_path))
        assert report["done_receipt"] == "cell_a_DONE.json"
        assert report["output_dir"].endswith("launch/cell_a")

    def test_refuses_missing_done_receipt_flag(self, tmp_path):
        payload = _cell_payload(tmp_path, "a")
        payload["launcher_argv"] = ["python", "x.py", "--output-dir", str(tmp_path)]
        with pytest.raises(q.QueueRefusal) as exc:
            q.verify_launcher_argv(q.QueuedCell.from_mapping(payload))
        assert exc.value.reason == "ARGV_NO_DONE_RECEIPT"

    def test_refuses_receipt_name_disagreement(self, tmp_path):
        payload = _cell_payload(tmp_path, "a")
        payload["done_receipt"] = "declared.json"
        with pytest.raises(q.QueueRefusal) as exc:
            q.verify_launcher_argv(q.QueuedCell.from_mapping(payload))
        assert exc.value.reason == "ARGV_DONE_RECEIPT_MISMATCH"

    def test_refuses_argv_not_naming_the_authorized_config(self, tmp_path):
        payload = _cell_payload(tmp_path, "a")
        payload["authorized_config"] = str(tmp_path / "elsewhere.json")
        with pytest.raises(q.QueueRefusal) as exc:
            q.verify_launcher_argv(q.QueuedCell.from_mapping(payload))
        assert exc.value.reason == "ARGV_CONFIG_MISMATCH"


# ── milestone reads + falsifiers ────────────────────────────────────────────────────────────────


class TestReads:
    def test_dotted_lookup(self):
        assert q._dotted({"a": {"b": 1}}, "a.b") == 1
        assert q._dotted({"a": {"b": 1}}, "a.missing") is None
        assert q._dotted({"a": 1}, "a.b") is None

    def test_read_history_at_step_takes_the_last_row_at_or_below(self, tmp_path):
        run_dir = tmp_path / "run"
        _history(run_dir, [{"completed_steps": s, "objective": {"m": s / 10}} for s in (1, 2, 3)])
        assert q.read_history_at_step(run_dir, 2)["completed_steps"] == 2
        assert q.read_history_at_step(run_dir, 0) is None

    def test_read_history_returns_none_when_the_step_is_not_reached(self, tmp_path):
        """REGRESSION (caught on the LIVE ng3 run, 2026-09-04): no answering an unreached milestone.

        ng3 stood at 741 steps and the step-5000 falsifier came back EVALUATED/fired=false off
        step-741 data -- a FALSE SURVIVED for a kill condition the cell had had no chance to trip.
        """
        run_dir = tmp_path / "run"
        _history(run_dir, [{"completed_steps": s, "objective": {"m": 0.1}} for s in (1, 2, 3)])
        assert q.read_history_at_step(run_dir, 99) is None
        assert q.read_history_at_step(run_dir, 3)["completed_steps"] == 3

    def test_falsifier_beyond_current_progress_is_pending_not_survived(self, tmp_path):
        """End-to-end shape of the same bug: the verdict must be PENDING, never SURVIVED."""
        run_dir = tmp_path / "run"
        _history(
            run_dir,
            [{"completed_steps": s, "objective": {"seg_expected_flip_realized": 0.003}}
             for s in (740, 741)],
        )
        cell = _cell(tmp_path)  # falsifier at_step=2000
        results = q.evaluate_falsifiers(cell, run_dir)
        assert results[0]["status"] == "NOT_YET_REACHED"
        assert results[0]["fired"] is None
        assert q.cell_verdict(cell, run_dir, [])["verdict"] == "PENDING"

    def test_read_history_missing_file(self, tmp_path):
        assert q.read_history_at_step(tmp_path / "absent", 10) is None

    def test_read_history_tolerates_garbage_lines(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "history.jsonl").write_text(
            "not json\n" + json.dumps({"completed_steps": 5, "objective": {"m": 1.0}}) + "\n",
            encoding="utf-8",
        )
        assert q.read_history_at_step(run_dir, 5)["completed_steps"] == 5

    def test_milestone_reads_against_a_named_control(self, tmp_path):
        treatment = tmp_path / "t"
        control = tmp_path / "c"
        _history(treatment, [{"completed_steps": 2000, "objective": {"m": 0.40}}])
        _history(control, [{"completed_steps": 2000, "objective": {"m": 0.50}}])
        cell = _cell(
            tmp_path, control_run_dir=str(control), control_label="cold control", milestones=[2000]
        )
        rows = q.milestone_reads(cell, treatment, ["objective.m"])
        assert len(rows) == 1
        row = rows[0]
        assert row["comparable"] is True
        assert row["control_label"] == "cold control"
        assert row["metrics"]["objective.m"]["delta_vs_control"] == pytest.approx(-0.10)

    def test_milestone_without_control_is_labelled_not_comparable(self, tmp_path):
        treatment = tmp_path / "t"
        _history(treatment, [{"completed_steps": 1000, "objective": {"m": 0.4}}])
        rows = q.milestone_reads(_cell(tmp_path, milestones=[1000]), treatment, ["objective.m"])
        assert rows[0]["comparable"] is False
        assert rows[0]["metrics"]["objective.m"]["control"] is None

    def test_milestone_not_yet_reached(self, tmp_path):
        """A milestone beyond the run's progress is NOT reached, and reports no treatment value.

        This assertion previously read ``is True`` -- it had codified the false-survived bug
        (a row existed at/below 5000, so the read "succeeded" off step-10 data). The live ng3
        integration run exposed it; the correct semantics are asserted here.
        """
        treatment = tmp_path / "t"
        _history(treatment, [{"completed_steps": 10, "objective": {"m": 0.4}}])
        rows = q.milestone_reads(_cell(tmp_path, milestones=[5000]), treatment, ["objective.m"])
        assert rows[0]["treatment_reached"] is False
        assert rows[0]["metrics"]["objective.m"]["treatment"] is None
        # A milestone the run HAS passed still reads normally.
        rows = q.milestone_reads(_cell(tmp_path, milestones=[10]), treatment, ["objective.m"])
        assert rows[0]["treatment_reached"] is True
        assert rows[0]["metrics"]["objective.m"]["treatment"] == 0.4


class TestFalsifiers:
    def _run(self, tmp_path, value: float | None, step: int = 2000) -> Path:
        run_dir = tmp_path / "run"
        row: dict = {"completed_steps": step, "objective": {}}
        if value is not None:
            row["objective"]["seg_expected_flip_realized"] = value
        _history(run_dir, [row])
        return run_dir

    def test_falsifier_fires_above_threshold(self, tmp_path):
        results = q.evaluate_falsifiers(_cell(tmp_path), self._run(tmp_path, 0.6))
        assert results[0]["fired"] is True
        assert results[0]["observed"] == 0.6
        assert results[0]["status"] == "EVALUATED"

    def test_falsifier_survives_below_threshold(self, tmp_path):
        results = q.evaluate_falsifiers(_cell(tmp_path), self._run(tmp_path, 0.4))
        assert results[0]["fired"] is False

    def test_falsifier_pending_when_step_not_reached(self, tmp_path):
        results = q.evaluate_falsifiers(_cell(tmp_path), self._run(tmp_path, 0.4, step=9999))
        assert results[0]["fired"] is None
        assert results[0]["status"] == "NOT_YET_REACHED"

    def test_falsifier_metric_absent(self, tmp_path):
        results = q.evaluate_falsifiers(_cell(tmp_path), self._run(tmp_path, None))
        assert results[0]["fired"] is None
        assert results[0]["status"] == "METRIC_ABSENT"

    def test_all_ops_evaluate(self, tmp_path):
        for op, value, expected in (
            ("lt", 0.4, True), ("lt", 0.6, False),
            ("le", 0.5, True), ("gt", 0.6, True), ("ge", 0.5, True),
        ):
            payload = _cell_payload(tmp_path, "a")
            payload["falsifiers"][0]["op"] = op
            cell = q.QueuedCell.from_mapping(payload)
            got = q.evaluate_falsifiers(cell, self._run(tmp_path, value))[0]["fired"]
            assert got is expected, f"op={op} value={value}"

    def test_verdict_falsified(self, tmp_path):
        verdict = q.cell_verdict(_cell(tmp_path), self._run(tmp_path, 0.6), ["objective.m"])
        assert verdict["verdict"] == "FALSIFIED"
        assert verdict["falsifiers_fired"] == ["flip_too_high_at_2k"]
        assert verdict["score_claim"] is False

    def test_verdict_survived(self, tmp_path):
        assert q.cell_verdict(_cell(tmp_path), self._run(tmp_path, 0.4), [])["verdict"] == "SURVIVED"

    def test_verdict_pending(self, tmp_path):
        run_dir = self._run(tmp_path, 0.4, step=9999)
        assert q.cell_verdict(_cell(tmp_path), run_dir, [])["verdict"] == "PENDING"

    def test_verdict_without_falsifiers_is_labelled(self, tmp_path):
        cell = _cell(tmp_path, falsifiers=[])
        verdict = q.cell_verdict(cell, self._run(tmp_path, 0.4), [])
        assert verdict["verdict"] == "NO_FALSIFIERS_DECLARED"

    def test_verdict_is_json_serializable(self, tmp_path):
        json.dumps(q.cell_verdict(_cell(tmp_path), self._run(tmp_path, 0.4), ["objective.m"]))


# ── planning + CLI ──────────────────────────────────────────────────────────────────────────────


def _admit_always(monkeypatch, admits: bool = True) -> None:
    """Stub the admission surface so planning tests never touch live machine state.

    ``discover_live_cells`` is stubbed too: ``plan_queue`` takes ONE live-fleet snapshot for the
    whole plan, and without this stub every planning test would walk the real SSD tiers.
    """
    monkeypatch.setattr(q.admission, "discover_live_cells", lambda *a, **kw: [])

    def _decide(candidate_peak_gib, **kw):
        mem = q.admission.MemoryVerdict(
            admits=admits, reclaimable_gib=100.0, committed_gib=1.0,
            candidate_peak_gib=candidate_peak_gib, live_unrealized_gib=0.0, margin_gib=16.0,
            required_gib=candidate_peak_gib + 16.0, headroom_gib=10.0, ceiling_gib=116.0,
            ceiling_headroom_gib=10.0, basis="stub", measurable=True,
            naive_shell_reclaimable_gib=None, reasons=("stub",),
        )
        thr = q.admission.ThroughputVerdict(
            admits=True, concurrency_observed=None, total_steps_per_min=None,
            serial_baseline_steps_per_min=None, ratio=None,
            evidence="NO_CONCURRENT_OBSERVATION", reasons=("stub",),
        )
        return q.admission.AdmissionDecision(
            verdict="ADMIT" if admits else "REFUSE", memory=mem, throughput=thr,
            live_cells=(), decided_utc="2026-09-04T00:00:00Z",
        )

    monkeypatch.setattr(q.admission, "decide_admission", _decide)


class TestPlanning:
    def test_ready_cell_is_planned_and_named_next(self, tmp_path, monkeypatch):
        (tmp_path / "tree").mkdir(exist_ok=True)
        _write_sealed(tmp_path, "cell_a", {"x": _pin()})
        monkeypatch.setattr(q.reseal, "verify_inputs_inside", lambda tree: {"x": _pin()})
        _admit_always(monkeypatch)
        plan = q.plan_queue([_cell(tmp_path)], reserve_bytes=0)
        assert plan["cells_ready"] == 1
        assert plan["next_cell_id"] == "cell_a"
        assert plan["cells"][0]["blockers"] == []
        assert plan["actuation"] == "PLAN_ONLY"

    def test_admission_refusal_blocks_the_cell(self, tmp_path, monkeypatch):
        (tmp_path / "tree").mkdir(exist_ok=True)
        _write_sealed(tmp_path, "cell_a", {"x": _pin()})
        monkeypatch.setattr(q.reseal, "verify_inputs_inside", lambda tree: {"x": _pin()})
        _admit_always(monkeypatch, admits=False)
        plan = q.plan_queue([_cell(tmp_path)], reserve_bytes=0)
        assert plan["cells_ready"] == 0
        assert "admission:REFUSE" in plan["cells"][0]["blockers"]

    def test_seal_blocker_is_reported_not_raised(self, tmp_path, monkeypatch):
        """A blocked cell must degrade into a plan entry, never crash the whole queue."""
        (tmp_path / "tree").mkdir(exist_ok=True)
        _write_sealed(tmp_path, "cell_a", {"x": _pin(path="/working/tree/x.py")})
        monkeypatch.setattr(
            q.reseal, "verify_inputs_inside", lambda tree: {"x": _pin(path="/sealed/x.py")}
        )
        _admit_always(monkeypatch)
        plan = q.plan_queue([_cell(tmp_path)], reserve_bytes=0)
        assert "seal:PIN_PATHS_NOT_REROOTED" in plan["cells"][0]["blockers"]
        assert plan["cells"][0]["ready"] is False

    def test_skip_seal_verify_bypasses_the_tree_call(self, tmp_path, monkeypatch):
        def _boom(tree):
            raise AssertionError("verify_inputs_inside must not be called")

        monkeypatch.setattr(q.reseal, "verify_inputs_inside", _boom)
        _admit_always(monkeypatch)
        plan = q.plan_queue([_cell(tmp_path)], reserve_bytes=0, verify_seal=False)
        assert plan["cells"][0]["seal"] == {"skipped": "verify_seal=False"}

    def test_plan_is_json_serializable(self, tmp_path, monkeypatch):
        _admit_always(monkeypatch)
        json.dumps(q.plan_queue([_cell(tmp_path)], reserve_bytes=0, verify_seal=False), default=str)

    def test_live_fleet_is_discovered_once_for_the_whole_plan(self, tmp_path, monkeypatch):
        """One snapshot per plan: cheaper, and every cell is judged against the SAME machine."""
        _admit_always(monkeypatch)
        calls: list[int] = []
        monkeypatch.setattr(
            q.admission, "discover_live_cells", lambda *a, **kw: calls.append(1) or []
        )
        cells = [_cell(tmp_path, "a"), _cell(tmp_path, "b"), _cell(tmp_path, "c")]
        q.plan_queue(cells, reserve_bytes=0, verify_seal=False)
        assert len(calls) == 1


class TestCLI:
    def _ready_queue(self, tmp_path, monkeypatch) -> Path:
        (tmp_path / "tree").mkdir(exist_ok=True)
        _write_sealed(tmp_path, "cell_a", {"x": _pin()})
        monkeypatch.setattr(q.reseal, "verify_inputs_inside", lambda tree: {"x": _pin()})
        _admit_always(monkeypatch)
        return _write_spec(tmp_path, [_cell_payload(tmp_path, "cell_a")])

    def test_plan_returns_zero_with_a_ready_cell(self, tmp_path, monkeypatch, capsys):
        spec = self._ready_queue(tmp_path, monkeypatch)
        assert q.main(["plan", "--queue", str(spec), "--reserve-bytes", "0"]) == 0
        assert json.loads(capsys.readouterr().out)["next_cell_id"] == "cell_a"

    def test_plan_returns_two_when_nothing_ready(self, tmp_path, monkeypatch, capsys):
        spec = self._ready_queue(tmp_path, monkeypatch)
        _admit_always(monkeypatch, admits=False)
        assert q.main(["plan", "--queue", str(spec), "--reserve-bytes", "0"]) == 2
        capsys.readouterr()

    def test_dry_run_never_claims_or_launches(self, tmp_path, monkeypatch, capsys):
        spec = self._ready_queue(tmp_path, monkeypatch)

        def _forbidden(*a, **kw):
            raise AssertionError("--dry-run must not claim or launch")

        monkeypatch.setattr(q, "fire_cell", _forbidden)
        monkeypatch.setattr(q, "place_claims", _forbidden)
        monkeypatch.setattr(q.subprocess, "run", _forbidden)
        rc = q.main(["run", "--dry-run", "--queue", str(spec), "--reserve-bytes", "0"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["dry_run"] is True
        assert payload["would_fire"] == "cell_a"

    def test_run_refuses_when_no_cell_is_ready(self, tmp_path, monkeypatch, capsys):
        spec = self._ready_queue(tmp_path, monkeypatch)
        _admit_always(monkeypatch, admits=False)
        monkeypatch.setattr(
            q, "fire_cell", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not fire"))
        )
        assert q.main(["run", "--queue", str(spec), "--reserve-bytes", "0"]) == 2
        assert json.loads(capsys.readouterr().out)["reason"] == "NO_READY_CELL"

    def test_verdict_command(self, tmp_path, monkeypatch, capsys):
        spec = self._ready_queue(tmp_path, monkeypatch)
        run_dir = tmp_path / "run"
        _history(run_dir, [{"completed_steps": 2000, "objective": {"seg_expected_flip_realized": 0.6}}])
        rc = q.main(
            [
                "verdict", "--queue", str(spec), "--cell-id", "cell_a",
                "--run-dir", str(run_dir), "--metric", "objective.seg_expected_flip_realized",
            ]
        )
        assert rc == 0
        assert json.loads(capsys.readouterr().out)["verdict"] == "FALSIFIED"

    def test_verdict_unknown_cell(self, tmp_path, monkeypatch, capsys):
        spec = self._ready_queue(tmp_path, monkeypatch)
        rc = q.main(["verdict", "--queue", str(spec), "--cell-id", "nope", "--run-dir", str(tmp_path)])
        assert rc == 2
        assert json.loads(capsys.readouterr().out)["reason"] == "UNKNOWN_CELL"

    def test_refusal_is_emitted_not_raised(self, tmp_path, capsys):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"schema": "other.v1", "cells": []}), encoding="utf-8")
        assert q.main(["plan", "--queue", str(path)]) == 2
        assert json.loads(capsys.readouterr().out)["status"] == "REFUSED"


class TestClaimIds:
    def test_claim_ids_are_day_suffixed(self, tmp_path):
        scorer, metal = q.claim_ids(_cell(tmp_path), day="20260904")
        assert scorer == "cell_a_scorer_20260904"
        assert metal == "cell_a_metal_20260904"


class TestPhantomClaimDiscipline:
    """A failure AFTER the claims are placed must report the orphan, never leave it silent."""

    def _fire(self, tmp_path, monkeypatch, *, launch_rc: int, authorize_raises: bool):
        monkeypatch.setattr(q, "place_claims", lambda cell, **kw: ("s_id", "m_id"))
        if authorize_raises:
            monkeypatch.setattr(
                q,
                "authorize",
                lambda *a: (_ for _ in ()).throw(q.QueueRefusal("X", "authorize blew up")),
            )
        else:
            monkeypatch.setattr(q, "authorize", lambda *a: {"path": "auth"})

        class _Completed:
            returncode = launch_rc
            stdout = "out"
            stderr = "err"

        monkeypatch.setattr(q.subprocess, "run", lambda *a, **kw: _Completed())
        return q.fire_cell(
            _cell(tmp_path),
            day="20260904",
            claims_path=tmp_path / "claims.md",
            ttl_hours=8.0,
            agent="test",
        )

    def test_launch_failure_reports_the_placed_claims(self, tmp_path, monkeypatch):
        with pytest.raises(q.QueueRefusal) as exc:
            self._fire(tmp_path, monkeypatch, launch_rc=2, authorize_raises=False)
        assert exc.value.reason == "LAUNCH_FAILED"
        assert exc.value.detail["placed_claims"] == {"scorer": "s_id", "metal": "m_id"}
        assert exc.value.detail["claims_need_terminal_row"] is True

    def test_authorize_failure_reports_the_placed_claims(self, tmp_path, monkeypatch):
        with pytest.raises(q.QueueRefusal) as exc:
            self._fire(tmp_path, monkeypatch, launch_rc=0, authorize_raises=True)
        assert exc.value.reason == "AUTHORIZE_FAILED_AFTER_CLAIMS"
        assert exc.value.detail["placed_claims"]["metal"] == "m_id"
        assert exc.value.detail["claims_need_terminal_row"] is True

    def test_successful_fire_returns_the_claim_ids(self, tmp_path, monkeypatch):
        receipt = self._fire(tmp_path, monkeypatch, launch_rc=0, authorize_raises=False)
        assert receipt["scorer_claim_id"] == "s_id"
        assert receipt["metal_claim_id"] == "m_id"
        assert receipt["cell_id"] == "cell_a"


class TestReuseNotFork:
    """The driver must EXTEND the existing surfaces, never fork a parallel launcher."""

    def test_imports_the_chain_driver_primitives(self):
        assert q.chain.authorized_config is not None
        assert q.chain.write_or_verify_authorized is not None
        assert q.chain.storage_preflight is not None

    def test_imports_the_admission_governor(self):
        assert q.admission.decide_admission is not None

    def test_imports_the_canonical_reseal_helper(self):
        assert q.reseal.verify_inputs_inside is not None

    def test_does_not_define_its_own_launcher(self):
        """Launching must go through the canonical detached launcher in the spec's argv."""
        source = _MODULE_PATH.read_text(encoding="utf-8")
        assert "start_new_session" not in source
        assert "os.fork" not in source
        assert "nohup" not in source

    def test_authorize_delegates_to_the_chain_driver(self, tmp_path, monkeypatch):
        _write_sealed(tmp_path, "cell_a", {"x": _pin()})
        seen: dict = {}

        def _authorized_config(sealed, scorer, metal):
            seen["args"] = (scorer, metal)
            return {"ok": True}

        monkeypatch.setattr(q.chain, "authorized_config", _authorized_config)
        monkeypatch.setattr(q.chain, "write_or_verify_authorized", lambda p, e: {"path": str(p)})
        q.authorize(_cell(tmp_path), "s1", "m1")
        assert seen["args"] == ("s1", "m1")
