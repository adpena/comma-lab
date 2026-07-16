# SPDX-License-Identifier: MIT
"""FEED-resume-observability-harden (2026-07-15): fast, $0, seconds-scale resume-epoch
OBSERVABILITY round-trip — the fidelity-vs-observability distinction made machine-verifiable.

Context: the #507 dry-start gate false-negatived (dry_start_report.json pass2 had
``resume_model_source: true`` but ``resume_start_epoch: null``) because the trainer restores the
epoch position bit-faithfully (``__resume_epoch`` stored + restored) yet never EMITTED it on the
launcher's parse surface (run.log JSONL) — the only prior emission lived inside the conditional C16
seed-anneal WARN. Resume FIDELITY was never broken; resume OBSERVABILITY was.

These tests chain the REAL shipped functions (no stubs of the work): the trainer's
``_build_resume_state_arrays`` -> ``_atomic_savez`` -> ``_load_resume_state`` ->
``_resolve_weights_only_warm_start`` -> ``_emit_resume_start_epoch_row`` (checkpoint -> restore ->
emit), then the launcher's ``parse_dry_start_run_metrics`` -> ``dry_start_resume_ok`` (parse ->
gate) on the ACTUAL emitted bytes. The only thing not exercised is the multi-hour training loop
between checkpoint and crash — that is the heavy governed 2x~9000s dry-start (operator-GO).

NO-FAKE: every assertion is on real function BEHAVIOR (arrays written and re-read from disk, the
printed JSON row re-parsed by the real launcher parser), not on constants/markers. The single
source-structure check at the bottom pins the CALL SITE only (anti-rot), after the callee's
behavior has been proven above.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
for _p in (str(_REPO / "tools"), str(_REPO / "src"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_TRAINER = _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"
_HAS_MLX = importlib.util.find_spec("mlx") is not None


@pytest.fixture(scope="module")
def T():
    if not _HAS_MLX:
        pytest.skip("mlx unavailable — trainer module import needs it")
    spec = importlib.util.spec_from_file_location("_twr_resume_obs", _TRAINER)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _tiny_args() -> SimpleNamespace:
    """Minimal args carrying every attribute _build_resume_state_arrays reads DIRECTLY (the rest
    default via getattr). Curriculum stage-position fields set to distinctive values so the
    round-trip assertion is non-vacuous."""
    return SimpleNamespace(
        n_hidden=2, hidden_dim=8, mod_dim=4, self_orient=False,
        n_dir_freqs=2, freq_across=1.0, freq_along=1.0, reorient_every=0, w_pose=0.0,
        # curriculum / stage-position cfg (persisted -> must round-trip)
        lane_band_start_epoch=311, pose_finish_start_epoch=123,
        seg_temporal_screw_start_epoch=77,
    )


def _tiny_state() -> tuple[dict, dict, dict]:
    rng = np.random.default_rng(0)
    live = {"w": rng.standard_normal((3, 3)).astype(np.float32)}
    ema = {"w": rng.standard_normal((3, 3)).astype(np.float32)}
    opt = {"w.m": rng.standard_normal((3, 3)).astype(np.float32)}
    return live, ema, opt


# ── the boot -> checkpoint -> resume -> emit chain, on the REAL shipped functions ──────────────
def test_checkpoint_restore_emit_roundtrip(T, tmp_path, capsys):
    live, ema, opt = _tiny_state()
    args = _tiny_args()
    ckpt_epoch = 7

    # "boot -> checkpoint": the real sidecar builder + the real atomic writer.
    arrays = T._build_resume_state_arrays(live, ema, opt, args=args, epoch=ckpt_epoch, in_feat=4)
    sidecar = tmp_path / "levelset_resume_state.npz"
    T._atomic_savez(sidecar, arrays)

    # "crash -> resume": the real loader restores the epoch position bit-faithfully.
    rs = T._load_resume_state(sidecar)
    assert rs["epoch"] == ckpt_epoch                      # fidelity (was NEVER broken)
    assert np.allclose(rs["live"]["w"], live["w"])
    assert np.allclose(rs["ema"]["w"], ema["w"])
    # curriculum stage-position round-trips (the "stage-position matches" leg).
    assert int(rs["cfg"]["__cfg_lane_band_start_epoch"]) == 311
    assert int(rs["cfg"]["__cfg_pose_finish_start_epoch"]) == 123
    assert int(rs["cfg"]["__cfg_seg_temporal_screw_start_epoch"]) == 77

    # start-epoch resolution: the trainer's continuation convention (no warm-start).
    ws = T._resolve_weights_only_warm_start(
        rs, warm_start_weights_only=False, warm_start_epoch=-1,
        ckpt_start_epoch=int(rs["epoch"]) + 1)
    assert ws["start_epoch"] == ckpt_epoch + 1

    # the NEW unconditional emission (the observability fix): printed AND returned.
    row = T._emit_resume_start_epoch_row(
        ws["start_epoch"], int(rs["epoch"]), resume_from=str(sidecar),
        warm_start_override=bool(ws["start_epoch"] != int(rs["epoch"]) + 1))
    printed = capsys.readouterr().out.strip().splitlines()[-1]
    emitted = json.loads(printed)
    assert emitted == row
    assert emitted["stage"] == "resume_start_epoch"
    assert emitted["resume_start_epoch"] == ckpt_epoch + 1   # restored epoch IS emitted
    assert emitted["resume_ckpt_epoch"] == ckpt_epoch        # ... AND equals the checkpoint epoch
    assert emitted["warm_start_override"] is False

    # ── launcher side: the ACTUAL emitted bytes must parse + pass the TIGHTENED gate ──────────
    import launch_witness_run as L
    p1_log = tmp_path / "p1_run.log"
    p1_log.write_text("\n".join([
        json.dumps({"stage": "loss_terms", "ep": ckpt_epoch, "total": 1.0}),
        json.dumps({"stage": "checkpoint", "kind": "intra_stage", "epoch": ckpt_epoch,
                    "resume_latest": "levelset_resume_state.npz"}),
    ]))
    p2_log = tmp_path / "p2_run.log"
    p2_log.write_text("\n".join([
        json.dumps({"stage": "resume_model_source", "resume_model_from": "live"}),
        printed,                                             # the trainer's real emitted row
        json.dumps({"stage": "loss_terms", "ep": ckpt_epoch + 1, "total": 0.9}),
    ]))
    p1 = L.parse_dry_start_run_metrics(p1_log)
    p2 = L.parse_dry_start_run_metrics(p2_log)
    assert p1["last_ckpt_epoch"] == ckpt_epoch
    assert p2["resume_start_epoch"] == ckpt_epoch + 1
    assert p2["resume_ckpt_epoch"] == ckpt_epoch
    assert L.dry_start_resume_ok(p2, p1) is True

    # negative control: a resume against the WRONG pass-1 checkpoint epoch must FAIL-CLOSED.
    assert L.dry_start_resume_ok(p2, {"last_ckpt_epoch": ckpt_epoch - 1}) is False
    # negative control: the pre-fix silent failure (null resume_start_epoch) must FAIL-CLOSED,
    # never ride the old weak ">= 1 or falsy->0" path.
    assert L.dry_start_resume_ok(
        {**p2, "resume_start_epoch": None, "resume_ckpt_epoch": None}, p1) is False


def test_warm_start_override_is_flagged(T, capsys):
    # A warm-start epoch override breaks the ckpt+1 convention — the row must SAY so (the
    # dry-start gate then fail-closes on rule 3, since a dry-start never warm-starts).
    row = T._emit_resume_start_epoch_row(126, 130, resume_from="x", warm_start_override=True)
    emitted = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert emitted["warm_start_override"] is True and row["warm_start_override"] is True
    import launch_witness_run as L
    p2 = {"resume_model_source": True, "resume_start_epoch": 126, "resume_ckpt_epoch": 130,
          "epochs_completed": 127}
    assert L.dry_start_resume_ok(p2) is False


# ── anti-rot: the call site (callee behavior proven above; this pins only that main() emits) ───
def test_trainer_main_calls_the_emit_helper_in_resume_path():
    src = _TRAINER.read_text()
    # def + >=1 call site
    assert src.count("_emit_resume_start_epoch_row(") >= 2
    # the call is anchored immediately after the start-epoch assignment in the resume block —
    # UNCONDITIONAL (not inside the C16 seed-anneal WARN, the pre-fix false-negative source).
    anchor = src.index('start_epoch = _ws["start_epoch"]')
    call = src.index("_emit_resume_start_epoch_row(", anchor)
    between = src[anchor:call]
    assert "if " not in between.replace("# ", ""), (
        "the resume_start_epoch emission must be UNCONDITIONAL in the resume path; a conditional "
        "emission is the exact silent-observability failure this row extincts")
