"""Tests for tools/costate_digest.py section_telemetry_binding — the #404 P0 wire-in.

The digest surfaces the binding-vs-inert lever readback (amber clip rate / chroma share /
pose-gate liveness / EMA-lag / D27b terminal-band) for the live run. Read-only + score-neutral
=> defaults ON. Fail-open: a broken analyzer must never break a session; absent telemetry omits
the row."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import costate_digest as cd  # noqa: E402


def _write_run_log(run_dir: Path, *, n_verdict: int = 12) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"stage": "witness_stability_resolved", "grad_clip": 0.5, "per_group_grad_clip": True}]
    for i in range(n_verdict):
        ep = 25 * (i + 1)
        rows.append({"stage": "loss_terms", "ep": ep, "accum_batch": 0,
                     "terms": {"seg": 0.5, "pose": 0.1, "chroma_boundary": 0.0},
                     "total": 1.0, "gnorm": 0.1})
        rows.append({"stage": "verdict", "epoch": ep, "d_seg": 0.01, "d_pose": 0.001,
                     "implied_S": 0.4})
        rows.append({"stage": "jacobian_basin", "epoch": ep, "median_sigma_min": 0.01,
                     "sigma_min_plateau_est": 0.01, "would_have_fired": False})
    (run_dir / "run.log").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_section_present_on_telemetry(tmp_path) -> None:
    _write_run_log(tmp_path)
    line, data = cd.section_telemetry_binding(tmp_path)
    assert line is not None and line.startswith("telem-binding:")
    assert "amber=INERT_NEVER_BINDS" in line
    assert data is not None
    assert data["pose_gate"]["verdict"] == "OK"
    assert data["terminal_band"]["d27b_ready"] is False


def test_section_omitted_when_run_dir_none() -> None:
    assert cd.section_telemetry_binding(None) == (None, None)


def test_section_omitted_on_missing_run_log(tmp_path) -> None:
    assert cd.section_telemetry_binding(tmp_path) == (None, None)


def test_section_fail_open_on_analyzer_exception(tmp_path, monkeypatch) -> None:
    _write_run_log(tmp_path)
    from tac.witness_control import telemetry_binding as tb

    def boom(*_a, **_k):  # pragma: no cover - trivial
        raise RuntimeError("synthetic")

    monkeypatch.setattr(tb, "audit_rows", boom)
    assert cd.section_telemetry_binding(tmp_path) == (None, None)


def test_build_digest_never_raises_with_telemetry_binding_wired() -> None:
    lines, data = cd.build_digest()
    assert isinstance(lines, list) and lines
    assert "telemetry_binding" in data  # key present even when None (no live run)


def test_section_arm_next_if_resumed_reads_schema_rows(tmp_path) -> None:
    path = tmp_path / "next.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema": "codex_arm_queue.next_if_resumed.v1",
                "name": "au1",
                "provenance": "positive-control",
                "source_path": ".omx/research/ddm_au1_20260805/AU1_RECEIPT.md",
                "line_start": 48,
            }
        )
        + "\n"
        + json.dumps({"schema": "unrelated", "name": "ignored"})
        + "\n",
        encoding="utf-8",
    )

    line, data = cd.section_arm_next_if_resumed(path)

    assert line.startswith("arm-next-if-resumed: 1 plan row(s)")
    assert "positive-control=1" in line
    assert data is not None
    assert data["rows"] == 1
    assert data["latest"][0]["name"] == "au1"


def _queue_module():
    import importlib.util
    from pathlib import Path

    tool = Path(__file__).resolve().parents[3] / "tools" / "codex_arm_queue.py"
    spec = importlib.util.spec_from_file_location("_q_for_costate_test", tool)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_section_arm_next_if_resumed_hides_superseded_rows_and_says_so(tmp_path) -> None:
    """MUTATION test on the live consumer: plant a retraction, prove the digest stops
    serving the row; remove the retraction, prove it comes back.

    The stale-bar hazard ddm_fb1 priced (+0.002337165 S, 233.7x the 1e-5 naming bar)
    was loaded HERE -- this is the surface a resuming arm actually reads."""
    q = _queue_module()
    path = tmp_path / "next.jsonl"
    plan = {
        "schema": "codex_arm_queue.next_if_resumed.v1",
        "row_id": "row-under-test",
        "name": "rx1",
        "provenance": "harvested-final",
        "source_path": ".omx/research/arm_final_messages/rx1.md",
        "line_start": 17,
        "text": "fire trigger: a retained archive below 186,269 B",
    }
    path.write_text(json.dumps(plan) + "\n", encoding="utf-8")

    line, data = cd.section_arm_next_if_resumed(path)
    assert data is not None and data["rows"] == 1
    assert data["retraction_debt"]["filter_available"] is True
    assert data["retraction_debt"]["superseded"] == 0

    q.retract_next_if_resumed_row(
        "row-under-test",
        reason="the 186,269 B bar sits 3,510 B above the live 182,759 B shipping archive",
        citation=".omx/research/ddm_fb1_stale_bar_rebase_and_bank_union_20260816.md",
        retracted_by="ddm_sc3",
        path=path,
    )

    line, data = cd.section_arm_next_if_resumed(path)
    assert data is not None
    assert data["rows"] == 0, "a superseded fire order must not be served as live"
    assert data["latest"] == []
    assert data["retraction_debt"]["superseded"] == 1
    assert "retracted: 1 superseded (hidden)" in line, "the drop must be reported, never silent"
    assert data["retraction_debt"]["reasons"], "the reason must be surfaced on request"

    path.write_text(json.dumps(plan) + "\n", encoding="utf-8")  # CONTROL
    _line, data = cd.section_arm_next_if_resumed(path)
    assert data is not None and data["rows"] == 1


def test_section_arm_next_if_resumed_amend_required_row_stays_served(tmp_path) -> None:
    q = _queue_module()
    path = tmp_path / "next.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema": "codex_arm_queue.next_if_resumed.v1",
                "row_id": "amend-row",
                "name": "wd2",
                "provenance": "harvested-final",
                "source_path": ".omx/research/arm_final_messages/wd2.md",
                "line_start": 17,
                "text": "three follow-ons, one of which quotes a 15,157 B cut",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    q.retract_next_if_resumed_row(
        "amend-row",
        reason="one clause quotes a 15,157 B cut computed off the superseded e480b v2 base",
        citation=".omx/research/ddm_fb1_stale_bar_rebase_and_bank_union_20260816.md",
        retracted_by="ddm_sc3",
        disposition=q.RETRACTION_AMEND_REQUIRED,
        path=path,
    )
    line, data = cd.section_arm_next_if_resumed(path)
    assert data is not None
    assert data["rows"] == 1, "AMEND_REQUIRED must not suppress the live clauses"
    assert data["retraction_debt"]["amend_required"] == 1
    assert "1 amend-required" in line
