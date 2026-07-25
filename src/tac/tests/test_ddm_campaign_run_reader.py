# SPDX-License-Identifier: MIT
"""Parser/discovery tests for ``tac.ddm_campaign_run_reader`` (#366 CAMPAIGN tab).

Fixtures are REAL rows copied from the live attempt-5 run dir
``ddm_366_campaign_v5_cured_20260725T062259Z`` (telemetry step schema
``ddm_joint_descent_full_run_step.v1``, verdict schema
``ddm_joint_descent_chunked_stage_verdict.v1``, receipt schema
``ddm_joint_descent_full_run_receipt.v1``) — never synthetic-shaped guesses.
Gitignored artifact classes (run.log, *.npz checkpoints, run.pid) are synthesized
per-test into tmp_path so the committed fixture tree stays JSON-only.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tac.ddm_campaign_run_reader import (
    ADVISORY_BATCH_LOCAL,
    SEALED_STEP_SECONDS,
    SNAPSHOT_SCHEMA,
    CampaignRunReader,
    is_campaign_run_dir,
    newest_campaign_run_dir,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ddm_campaign_run"


def _make_run(tmp_path: Path, name: str = "ddm_366_campaign_test_20260725T000000Z") -> Path:
    run = tmp_path / name
    shutil.copytree(FIXTURES, run)
    (run / "run.pid").write_text("99999999")  # not a live pid
    (run / "run.log").write_text('{"stage":"fixture"}\n')
    ck = run / "checkpoints"
    ck.mkdir()
    (ck / "01_residual_bucket_realized_acceptance_accepted_global000001.npz").write_bytes(b"x")
    (ck / "01_residual_bucket_realized_acceptance_accepted_global000002.npz").write_bytes(b"x")
    return run


# ---- discovery ---------------------------------------------------------------

def test_is_campaign_run_dir_by_name_glob(tmp_path):
    run = _make_run(tmp_path)
    assert is_campaign_run_dir(run)


def test_is_campaign_run_dir_structural_marker_without_name(tmp_path):
    # a run-dir NAME is never a discovery contract: run_identity.json schema admits it
    run = _make_run(tmp_path, name="renamed_arm_without_family_prefix")
    assert is_campaign_run_dir(run)


def test_not_a_run_dir(tmp_path):
    plain = tmp_path / "not_a_run"
    plain.mkdir()
    assert not is_campaign_run_dir(plain)
    assert newest_campaign_run_dir([tmp_path / "missing_root"]) is None


def test_newest_run_dir_prefers_freshest_signal(tmp_path):
    import os
    old = _make_run(tmp_path, name="ddm_366_campaign_old_20260724T000000Z")
    new = _make_run(tmp_path, name="ddm_366_campaign_new_20260725T000000Z")
    past = 1_700_000_000.0
    for d in (old, new):
        for p in [d, *d.rglob("*")]:
            os.utime(p, (past, past))
    # freshen ONE signal file in `new` — discovery must follow it
    os.utime(new / "run.log", None)
    assert newest_campaign_run_dir([tmp_path]) == new


# ---- snapshot parsing ----------------------------------------------------------

@pytest.fixture()
def snap_and_reader(tmp_path):
    run = _make_run(tmp_path)
    reader = CampaignRunReader(roots=[tmp_path])
    return reader.snapshot(), reader, run


def test_snapshot_shape_and_honesty_labels(snap_and_reader):
    snap, _, run = snap_and_reader
    assert snap["ok"] is True
    assert snap["schema"] == SNAPSHOT_SCHEMA
    assert snap["score_claim"] is False
    assert snap["advisory_batch_local_label"] == ADVISORY_BATCH_LOCAL
    assert snap["run_name"] == run.name


def test_snapshot_steps_parsed_sorted_batch_local(snap_and_reader):
    snap, _, _ = snap_and_reader
    steps = snap["steps"]
    assert [s["global_step"] for s in steps] == [49, 50]
    last = steps[-1]
    # real values from the copied step000050.json row
    assert last["stage_id"] == "01_residual_bucket_realized_acceptance"
    assert last["d_seg_initial"] == pytest.approx(0.0688985213637352)
    assert last["d_seg_final"] == pytest.approx(0.06640625)
    assert last["d_pose_final"] == pytest.approx(15.288575172424316)
    assert last["seconds"] == pytest.approx(147.05009958310984)
    assert last["gradient_norm"] == pytest.approx(0.6680812601521519)
    assert last["proposal_source"] == "local_exact_gradient"
    assert last["pair_ids"] == [192, 193, 194, 195]


def test_snapshot_verdicts_kinds_and_per_class(snap_and_reader):
    snap, _, _ = snap_and_reader
    verdicts = snap["verdicts"]
    kinds = {v["kind"] for v in verdicts}
    assert kinds == {"baseline", "stage_verdict", "warm_start_proposal"}
    stage = [v for v in verdicts if v["kind"] == "stage_verdict"][-1]
    assert stage["global_step"] == 50
    assert stage["parameter_shadow"] == "ema"
    assert stage["d_seg"] == pytest.approx(0.07051923116048177)
    assert stage["target_d_seg"] == pytest.approx(0.020602722168)
    assert stage["realized_stage_decision"] == "BLOCKED_REALIZED_DSEG_REGRESSION"
    # canonical comma10k per-class breakdown survives slimming
    assert set(stage["per_class"]) == {"Road", "Lane", "Undrivable", "Movable", "MyCar"}
    assert stage["per_class"]["Lane"]["d_seg"] == pytest.approx(0.5344369489704462)
    # engage state rides the verdict
    assert stage["engage"]["classification"] == "INSUFFICIENT_EXACT_VERDICTS"
    assert stage["engage"]["exact_verdict_steps"] == [0, 1, 50]


def test_snapshot_geometry_checkpoints_cadence(snap_and_reader):
    snap, _, _ = snap_and_reader
    assert snap["geometry_events_count"] == 1
    assert snap["geometry_events_tail"][0]["status"] == "cured"
    assert snap["checkpoints"]["count"] == 2
    cad = snap["cadence"]
    assert cad["sealed_step_seconds"] == SEALED_STEP_SECONDS
    assert cad["measured_n"] == 2
    assert cad["measured_last_s"] == pytest.approx(147.05009958310984)


def test_snapshot_schedule_targets_from_receipt(snap_and_reader):
    snap, _, _ = snap_and_reader
    stages = snap["schedule"]["stages"]
    assert [s["stage_id"] for s in stages] == [
        "01_residual_bucket_realized_acceptance",
        "02_lane_production_joint_descent",
        "03_pose_coupled_finish",
    ]
    assert stages[0]["target_d_seg"] == pytest.approx(0.020602722168)
    assert stages[1]["target_d_seg"] == pytest.approx(0.013735148112)
    assert stages[2]["target_d_seg"] == pytest.approx(0.006867574056)
    assert stages[2]["target_d_pose"] == pytest.approx(163.06116431842463)
    rec = snap["receipt"]
    assert rec["present"] is True
    assert rec["verdict"] == "BLOCKED_REALIZED_DSEG_REGRESSION"
    assert rec["pointer_moved"] is False


def test_snapshot_status_pid_dead_and_ended(snap_and_reader):
    snap, _, _ = snap_and_reader
    st = snap["status"]
    assert st["pid"] == 99999999
    assert st["pid_alive"] is False
    assert st["ended"] is True  # receipt present
    assert st["global_step"] == 50
    assert st["stage_index"] == 0
    assert st["last_telemetry_age_s"] is not None
    assert st["run_log_age_s"] is not None


def test_incremental_reparse_only_changed_files(snap_and_reader, tmp_path):
    snap1, reader, run = snap_and_reader
    # unchanged second snapshot: identical step rows (mtime-gated cache hit)
    snap2 = reader.snapshot()
    assert snap2["steps"] == snap1["steps"]
    # mutate one telemetry file -> only that row updates
    p = run / "telemetry" / "step000050.json"
    row = json.loads(p.read_text())
    row["final"]["d_seg"] = 0.001
    p.write_text(json.dumps(row))
    import os
    st = p.stat()
    os.utime(p, (st.st_atime + 5, st.st_mtime + 5))
    snap3 = reader.snapshot()
    assert snap3["steps"][-1]["d_seg_final"] == pytest.approx(0.001)
    assert snap3["steps"][0] == snap1["steps"][0]


def test_new_run_dir_resets_reader(tmp_path):
    old = _make_run(tmp_path, name="ddm_366_campaign_a_20260725T000000Z")
    reader = CampaignRunReader(roots=[tmp_path])
    assert reader.snapshot()["run_name"] == old.name
    import os
    past = 1_700_000_000.0
    for p in [old, *old.rglob("*")]:
        os.utime(p, (past, past))
    new = _make_run(tmp_path, name="ddm_366_campaign_b_20260725T010000Z")
    snap = reader.snapshot()
    assert snap["run_name"] == new.name  # auto-follows the freshest run, no repoint
