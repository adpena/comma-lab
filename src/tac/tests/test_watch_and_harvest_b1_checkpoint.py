# SPDX-License-Identifier: MIT
"""NO-FAKE tests for tools/watch_and_harvest_b1_checkpoint.py.

These tests verify ACTUAL behavior, not constants:

* idempotency: a present result JSON short-circuits the watch loop;
* fail-closed: a stale heartbeat with no eligible checkpoint returns rc=3;
* fail-closed: max-wait elapsed returns rc=3;
* memory gate: harvest_once waits (returns None + status) when free < min;
* checkpoint selection: earliest ep>=target with an existing EMA state file;
* final fallback: selects the final checkpoint when no ep>=target exists;
* heartbeat parsing: ISO line age + TRAIN_EXIT detection;
* the export->B2 chain wires the real B2 command and writes the result JSON
  (B2 mocked so the test is hermetic);
* a REAL checkpoint state round-trips through import and exports a single-member
  ``x`` (HIV1) backend-only archive (MLX; skipped if MLX unavailable);
* the detached launch command is the canonical nohup pattern.

Each test would FAIL if the function body were replaced by a constant-return
stub (the Slot EEE NO-FAKE discipline).
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_TOOL_PATH = REPO_ROOT / "tools" / "watch_and_harvest_b1_checkpoint.py"


def _load_tool():
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    spec = importlib.util.spec_from_file_location(
        "watch_and_harvest_b1_checkpoint", _TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["watch_and_harvest_b1_checkpoint"] = mod  # so dataclasses resolve
    spec.loader.exec_module(mod)
    return mod


WH = _load_tool()


# The harvester (correctly) refuses persisted artifacts under the system /tmp
# tree per CLAUDE.md disk-hygiene. pytest's default ``tmp_path`` lives under
# ``/private/tmp`` on macOS, which the guard rejects. So these tests use a
# repo-local ``.omx/tmp`` scratch base (the canonical allowed scratch) instead
# of ``tmp_path`` for any path handed to the harvester. This keeps the real-CLI
# guard strict while the tests remain hermetic + flag-free.
_SCRATCH_BASE = REPO_ROOT / ".omx" / "tmp" / "pytest_harvest_scratch"


@pytest.fixture
def tmp_path(request):
    import shutil
    import uuid

    d = _SCRATCH_BASE / f"{request.node.name}_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Fixtures: a fake checkpoint dir + heartbeat.
# ---------------------------------------------------------------------------


def _write_checkpoint(
    checkpoint_dir: Path,
    *,
    epoch: int,
    role: str = "periodic",
    loss: float = 1.0,
    write_ema_state: bool = True,
    metric_value: float | None = None,
    metric_mode: str = "min",
) -> Path:
    """Write a canonical checkpoint meta + (optionally) its .npsd state files."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = f"epoch{epoch:06d}_20260609T000000Z"
    if role == "final":
        stem = f"final_{stem}"
    elif role == "best":
        stem = f"best_{stem}"
    live_path = checkpoint_dir / f"{stem}.live.state"
    ema_path = checkpoint_dir / f"{stem}.ema_shadow.state"
    meta_path = checkpoint_dir / f"{stem}.meta.json"
    # The adapter emits ``<path>.npsd``; mirror that.
    if write_ema_state:
        (ema_path.with_suffix(ema_path.suffix + ".npsd")).write_bytes(b"FAKE_NPSD")
        (live_path.with_suffix(live_path.suffix + ".npsd")).write_bytes(b"FAKE_NPSD")
    meta = {
        "schema_version": "long_training_canonical_checkpoint.v1",
        "substrate_id": "hi_nerv_mlx_local",
        "lane_id": "lane_hi_nerv_mlx_score_aware_local_20260602",
        "curriculum_hash": "0" * 64,
        "global_epoch": int(epoch),
        "loss": float(loss),
        "checkpoint_selection_metric_key": "total",
        "checkpoint_selection_metric_value": (
            float(metric_value) if metric_value is not None else float(loss)
        ),
        "checkpoint_selection_metric_mode": metric_mode,
        "wall_clock_seconds": 100.0,
        "is_final": role == "final",
        "is_best": role == "best",
        "checkpoint_role": role,
        "live_state_path": str(live_path),
        "ema_shadow_state_path": str(ema_path),
        "captured_at_utc": "2026-06-09T00:00:00Z",
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta_path


def _write_heartbeat(path: Path, *, age_seconds: float, train_exit: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = time.gmtime(time.time() - age_seconds)
    line = time.strftime("%Y-%m-%dT%H:%M:%SZ", ts) + " pid=99999 run=test"
    body = line + "\n"
    if train_exit:
        body += time.strftime("%Y-%m-%dT%H:%M:%SZ", ts) + " TRAIN_EXIT rc=0 run=test\n"
    path.write_text(body, encoding="utf-8")


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "b1_run"
    d.mkdir()
    return d


@pytest.fixture
def paths(run_dir: Path):
    hb = run_dir / "heartbeat.log"
    return WH.resolve_paths(run_dir, target_epoch=250, heartbeat_path=hb)


# ---------------------------------------------------------------------------
# 1. Checkpoint selection behavior.
# ---------------------------------------------------------------------------


def test_select_checkpoint_picks_earliest_eligible_above_target(paths):
    _write_checkpoint(paths.checkpoint_dir, epoch=250)
    _write_checkpoint(paths.checkpoint_dir, epoch=500)
    cks = WH.list_checkpoints(paths.checkpoint_dir)
    target = WH.select_target_checkpoint(cks, target_epoch=250)
    assert target is not None
    assert target.global_epoch == 250  # earliest >= target, not the newest


def test_select_checkpoint_skips_checkpoint_with_missing_ema_state(paths):
    # ep250 meta exists but its EMA state file is missing -> not eligible.
    _write_checkpoint(paths.checkpoint_dir, epoch=250, write_ema_state=False)
    _write_checkpoint(paths.checkpoint_dir, epoch=500, write_ema_state=True)
    cks = WH.list_checkpoints(paths.checkpoint_dir)
    target = WH.select_target_checkpoint(cks, target_epoch=250)
    assert target is not None
    assert target.global_epoch == 500  # 250 skipped (no state on disk)


def test_select_checkpoint_returns_none_when_below_target_and_no_fallback(paths):
    _write_checkpoint(paths.checkpoint_dir, epoch=100)
    cks = WH.list_checkpoints(paths.checkpoint_dir)
    assert (
        WH.select_target_checkpoint(cks, target_epoch=250, allow_final_fallback=False)
        is None
    )


def test_select_checkpoint_final_fallback(paths):
    _write_checkpoint(paths.checkpoint_dir, epoch=100)  # below target
    _write_checkpoint(paths.checkpoint_dir, epoch=120, role="final")
    cks = WH.list_checkpoints(paths.checkpoint_dir)
    target = WH.select_target_checkpoint(cks, target_epoch=250, allow_final_fallback=True)
    assert target is not None
    assert target.is_final and target.global_epoch == 120


def test_list_checkpoints_ignores_non_canonical_meta(paths):
    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (paths.checkpoint_dir / "junk.meta.json").write_text('{"schema_version":"other"}')
    _write_checkpoint(paths.checkpoint_dir, epoch=250)
    cks = WH.list_checkpoints(paths.checkpoint_dir)
    assert len(cks) == 1 and cks[0].global_epoch == 250


# ---------------------------------------------------------------------------
# 1b. OFF-BY-ONE FIX (2026-06-09): the 250th-epoch interval-boundary checkpoint
# is saved as ``epoch000249`` with ``global_epoch=249`` (0-indexed). The
# diverging pilot's harvester targeted ``--target-epoch 250`` with a strict
# ``>= 250`` match and the harvest NEVER fired. The fix matches
# ``>= target_epoch - 1``. These regression tests reproduce the EXACT killed-run
# naming so the bug cannot return.
# ---------------------------------------------------------------------------


def test_select_checkpoint_matches_epoch000249_for_target_250(paths):
    """REGRESSION: the killed run's epoch000249 (global_epoch=249) MUST match.

    Reproduces the exact failure: the 250th-epoch checkpoint is named
    ``epoch000249`` with ``global_epoch=249``; targeting 250 must select it
    (>= target_epoch - 1 == 249), not return None.
    """
    meta_path = _write_checkpoint(paths.checkpoint_dir, epoch=249)
    # Confirm the killed-run naming is reproduced exactly.
    assert "epoch000249" in meta_path.name
    cks = WH.list_checkpoints(paths.checkpoint_dir)
    assert cks[0].global_epoch == 249
    target = WH.select_target_checkpoint(cks, target_epoch=250)
    assert target is not None, (
        "epoch000249 (global_epoch=249) MUST be selected for target_epoch=250 "
        "(the off-by-one bug returned None and the harvest never fired)"
    )
    assert target.global_epoch == 249


def test_select_checkpoint_off_by_one_picks_boundary_not_next_interval(paths):
    """The interval-boundary checkpoint (249) wins over the next one (499)."""
    _write_checkpoint(paths.checkpoint_dir, epoch=249)  # 250th-epoch boundary
    _write_checkpoint(paths.checkpoint_dir, epoch=499)  # 500th-epoch boundary
    cks = WH.list_checkpoints(paths.checkpoint_dir)
    target = WH.select_target_checkpoint(cks, target_epoch=250)
    assert target is not None
    assert target.global_epoch == 249  # earliest >= 249, not 499


def test_select_checkpoint_still_excludes_genuinely_early_checkpoints(paths):
    """A checkpoint well below the boundary (ep=200) is NOT matched for tgt 250.

    Guards against the fix over-matching: -1 tolerance is for the 0-indexed
    boundary only; a 200-epoch checkpoint (global_epoch=200 < 249) is excluded.
    """
    _write_checkpoint(paths.checkpoint_dir, epoch=200)
    cks = WH.list_checkpoints(paths.checkpoint_dir)
    assert (
        WH.select_target_checkpoint(cks, target_epoch=250, allow_final_fallback=False)
        is None
    )


def test_select_checkpoint_target_epoch_one_clamps_threshold_floor(paths):
    """target_epoch<=1 clamps the threshold to 0 (never negative)."""
    _write_checkpoint(paths.checkpoint_dir, epoch=0)
    cks = WH.list_checkpoints(paths.checkpoint_dir)
    target = WH.select_target_checkpoint(cks, target_epoch=1)
    assert target is not None and target.global_epoch == 0


# ---------------------------------------------------------------------------
# 1c. SELF-BOOTSTRAP (2026-06-09): the diverging pilot's detached harvester
# failed with ``ModuleNotFoundError: No module named 'experiments'`` because it
# ran without PYTHONPATH and did not insert REPO_ROOT on sys.path. The export
# path does an in-process ``import experiments.train_substrate_hi_nerv_mlx_local``,
# so the harvest NEVER produced an archive. The module now self-bootstraps.
# ---------------------------------------------------------------------------


def test_harvester_inserts_repo_root_on_sys_path() -> None:
    """REGRESSION: loading the harvester puts REPO_ROOT + src on sys.path so the
    in-process ``import experiments...`` export path resolves without PYTHONPATH."""
    assert str(REPO_ROOT) in sys.path, (
        "harvester must insert REPO_ROOT on sys.path so import experiments works "
        "(the diverging pilot's harvest died on ModuleNotFoundError: experiments)"
    )
    assert str(REPO_ROOT / "src") in sys.path


def test_harvester_export_path_imports_experiments_in_subprocess() -> None:
    """NO-FAKE: a fresh python (no PYTHONPATH, neutral cwd) that loads the
    harvester module can then ``import experiments...`` — proving the bootstrap
    fixes the killed-run failure end-to-end, not just in this test process."""
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('wh', {str(_TOOL_PATH)!r})\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['wh']=m\n"
        "spec.loader.exec_module(m)\n"
        "import experiments.train_substrate_hi_nerv_mlx_local\n"
        "print('OK')\n"
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)  # reproduce the killed-run no-PYTHONPATH context
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(os.sep)),  # neutral cwd (root), not the repo
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"harvester self-bootstrap failed: rc={proc.returncode}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    assert "OK" in proc.stdout


# ---------------------------------------------------------------------------
# 2. Heartbeat parsing.
# ---------------------------------------------------------------------------


def test_heartbeat_age_parses_iso_line(paths):
    _write_heartbeat(paths.heartbeat_path, age_seconds=30)
    age = WH.heartbeat_age_seconds(paths.heartbeat_path, now=time.time())
    assert age is not None and 25 <= age <= 45


def test_heartbeat_age_none_when_missing(tmp_path: Path):
    assert WH.heartbeat_age_seconds(tmp_path / "nope.log") is None


def test_heartbeat_train_exit_detected(paths):
    _write_heartbeat(paths.heartbeat_path, age_seconds=30, train_exit=True)
    assert WH.heartbeat_reports_train_exit(paths.heartbeat_path) is True


# ---------------------------------------------------------------------------
# 3. Idempotency.
# ---------------------------------------------------------------------------


def test_watch_loop_idempotent_when_result_exists(paths):
    paths.result_json.parent.mkdir(parents=True, exist_ok=True)
    paths.result_json.write_text('{"schema":"b1_checkpoint_harvest_result.v1"}')
    calls = {"sleep": 0}

    rc = WH.watch_loop(
        paths,
        target_epoch=250,
        sleep_fn=lambda s: calls.__setitem__("sleep", calls["sleep"] + 1),
        export_fn=lambda *a, **k: pytest.fail("export must NOT run when result exists"),
        b2_runner=lambda cmd: pytest.fail("B2 must NOT run when result exists"),
    )
    assert rc == 0
    assert calls["sleep"] == 0  # never slept, never harvested
    status = json.loads(paths.status_json.read_text())
    assert status["state"] == "idempotent_result_exists"


# ---------------------------------------------------------------------------
# 4. Fail-closed paths.
# ---------------------------------------------------------------------------


def test_watch_loop_fail_closed_on_stale_heartbeat_no_checkpoint(paths):
    # Heartbeat is stale (10 min) and no checkpoint -> rc=3, failure status.
    _write_heartbeat(paths.heartbeat_path, age_seconds=10 * 60)
    rc = WH.watch_loop(
        paths,
        target_epoch=250,
        heartbeat_stale_seconds=7 * 60,
        sleep_fn=lambda s: None,
    )
    assert rc == 3
    status = json.loads(paths.status_json.read_text())
    assert status["state"] == "failed_heartbeat_stale"
    assert status["detail"]["heartbeat_age_seconds"] >= 7 * 60


def test_watch_loop_fail_closed_on_max_wait_elapsed(paths):
    # Fresh heartbeat but no checkpoint ever; max_wait elapses quickly.
    _write_heartbeat(paths.heartbeat_path, age_seconds=5)
    fake_now = {"t": 0.0}

    def now_fn() -> float:
        return fake_now["t"]

    def sleep_fn(_s: float) -> None:
        fake_now["t"] += 1000.0  # jump time forward each tick

    rc = WH.watch_loop(
        paths,
        target_epoch=250,
        heartbeat_stale_seconds=7 * 60,
        max_wait_seconds=10.0,
        sleep_fn=sleep_fn,
        now_fn=now_fn,
    )
    assert rc == 3
    status = json.loads(paths.status_json.read_text())
    assert status["state"] == "failed_max_wait_elapsed"


def test_watch_loop_does_not_fail_when_train_exited(paths):
    # TRAIN_EXIT present + a final checkpoint -> should harvest, not fail-close.
    _write_heartbeat(paths.heartbeat_path, age_seconds=30 * 60, train_exit=True)
    _write_checkpoint(paths.checkpoint_dir, epoch=120, role="final")

    def fake_export(checkpoint, out_dir, **kwargs):
        return WH.ExportResult(
            archive_path=out_dir / "archive.zip",
            archive_sha256="deadbeef",
            archive_bytes=1234,
            param_count=WH.EXPECTED_PARAM_COUNT,
            zip_members=["x"],
            member_magic="HIV1",
            load_info={"param_tensors_restored": 27},
        )

    def fake_b2(cmd):
        # Simulate B2 writing its out-row.
        out_row = Path(cmd[cmd.index("--out-row") + 1])
        out_row.parent.mkdir(parents=True, exist_ok=True)
        out_row.write_text(json.dumps({"final_score": 0.42, "verdict": "ok"}))
        return 0

    rc = WH.watch_loop(
        paths,
        target_epoch=250,
        heartbeat_stale_seconds=7 * 60,
        sleep_fn=lambda s: None,
        export_fn=fake_export,
        b2_runner=fake_b2,
    )
    assert rc == 0
    assert paths.result_json.is_file()


# ---------------------------------------------------------------------------
# 5. Memory gate.
# ---------------------------------------------------------------------------


def test_harvest_once_waits_when_memory_low(paths):
    _write_checkpoint(paths.checkpoint_dir, epoch=250)
    result = WH.harvest_once(
        paths,
        target_epoch=250,
        min_free_gb=15.0,
        free_memory_fn=lambda: 5.0,  # below threshold
        export_fn=lambda *a, **k: pytest.fail("export must NOT run when memory low"),
        b2_runner=lambda cmd: pytest.fail("B2 must NOT run when memory low"),
    )
    assert result is None
    status = json.loads(paths.status_json.read_text())
    assert status["state"] == "waiting_for_memory"
    assert status["detail"]["free_gb"] == 5.0


def test_harvest_once_proceeds_when_memory_sufficient(paths):
    _write_checkpoint(paths.checkpoint_dir, epoch=250)

    def fake_export(checkpoint, out_dir, **kwargs):
        assert checkpoint.global_epoch == 250
        return WH.ExportResult(
            archive_path=out_dir / "archive.zip",
            archive_sha256="abc123",
            archive_bytes=265288,
            param_count=WH.EXPECTED_PARAM_COUNT,
            zip_members=["x"],
            member_magic="HIV1",
            load_info={"param_tensors_restored": 27},
        )

    captured = {}

    def fake_b2(cmd):
        captured["cmd"] = cmd
        out_row = Path(cmd[cmd.index("--out-row") + 1])
        out_row.parent.mkdir(parents=True, exist_ok=True)
        out_row.write_text(json.dumps({"final_score": 0.18, "verdict": "full"}))
        return 0

    result = WH.harvest_once(
        paths,
        target_epoch=250,
        min_free_gb=15.0,
        free_memory_fn=lambda: 60.0,
        export_fn=fake_export,
        b2_runner=fake_b2,
    )
    assert result is not None
    assert result["first_exact_score_advisory"] == 0.18
    assert result["evidence_grade"] == "[macOS-CPU advisory]"
    assert result["authoritative"] is False
    # The B2 command is the real bridge command, device cpu, no --keep-work-dir.
    cmd = captured["cmd"]
    assert "run_hi_nerv_backend_only_b2_exact_eval.py" in " ".join(cmd)
    assert "--device" in cmd and cmd[cmd.index("--device") + 1] == "cpu"
    assert "--keep-work-dir" not in cmd  # disk hygiene


# ---------------------------------------------------------------------------
# 6. B2 command construction + tmp refusal.
# ---------------------------------------------------------------------------


def test_build_b2_command_is_turnkey(tmp_path: Path):
    cmd = WH.build_b2_command(
        tmp_path / "archive.zip",
        tmp_path / "out.json",
        tmp_path / "work",
        device="cpu",
    )
    s = " ".join(cmd)
    assert "run_hi_nerv_backend_only_b2_exact_eval.py" in s
    assert "--archive" in cmd and "--out-row" in cmd and "--work-root" in cmd
    assert "--device" in cmd and cmd[cmd.index("--device") + 1] == "cpu"


def test_refuse_tmp_path_in_work_root():
    with pytest.raises(ValueError, match="transient"):
        WH.run_b2_eval(
            Path("/Volumes/VertigoDataTier/pact/x/archive.zip"),
            Path("/Volumes/VertigoDataTier/pact/x/out.json"),
            Path("/tmp/b2work"),
            runner=lambda cmd: 0,
        )


def test_resolve_paths_refuses_tmp_run_dir():
    with pytest.raises(ValueError, match="transient"):
        WH.resolve_paths(Path("/tmp/b1_run"), target_epoch=250)


# ---------------------------------------------------------------------------
# 7. Detached launch command (canonical nohup pattern).
# ---------------------------------------------------------------------------


def test_detached_launch_command_is_canonical_nohup_pattern(tmp_path: Path):
    cmd = WH.build_detached_launch_command(
        ["/venv/python", "/tools/x.py", "--run-dir", "/ssd/run"],
        tmp_path / "watcher.log",
    )
    assert cmd.startswith("nohup bash -c ")
    assert "< /dev/null" in cmd
    assert "disown" in cmd
    assert "tee" in cmd


def test_detached_launch_command_refuses_tmp_log():
    with pytest.raises(ValueError, match="transient"):
        WH.build_detached_launch_command(["x"], Path("/tmp/watcher.log"))


# ---------------------------------------------------------------------------
# 8. Result payload carries false-authority + advisory tags.
# ---------------------------------------------------------------------------


def test_result_payload_marks_non_promotable_and_advisory(paths):
    ck = WH.CheckpointRef(
        meta_path=paths.checkpoint_dir / "m.meta.json",
        global_epoch=250,
        role="periodic",
        loss=1.0,
        ema_state_path=paths.checkpoint_dir / "e.ema_shadow.state",
        live_state_path=paths.checkpoint_dir / "e.live.state",
        selection_metric_key="total",
        selection_metric_value=1.0,
        is_final=False,
    )
    export = WH.ExportResult(
        archive_path=paths.export_dir / "archive.zip",
        archive_sha256="sha",
        archive_bytes=265288,
        param_count=WH.EXPECTED_PARAM_COUNT,
        zip_members=["x"],
        member_magic="HIV1",
        load_info={},
    )
    payload = WH.write_result(
        paths.result_json,
        target_epoch=250,
        checkpoint=ck,
        export=export,
        b2_summary={"b2_result": {"final_score": 0.19, "verdict": "full"}},
    )
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["promotable"] is False
    assert payload["evidence_grade"] == "[macOS-CPU advisory]"
    assert payload["first_exact_score_advisory"] == 0.19
    # The written file matches.
    on_disk = json.loads(paths.result_json.read_text())
    assert on_disk["first_exact_score_advisory"] == 0.19


# ---------------------------------------------------------------------------
# 9. REAL export round-trip (MLX) — the NO-FAKE end-to-end proof.
# ---------------------------------------------------------------------------


def _mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _mlx_available(), reason="MLX not available")
def test_real_export_from_checkpoint_emits_single_member_x_hiv1(tmp_path: Path):
    """Build a REAL tiny model, save its state as .npsd, then harvest-export it.

    This proves the standalone export path actually loads checkpoint weights and
    emits a contest-shaped single-member ``x`` (HIV1) archive — it would fail if
    export_backend_only_archive_from_checkpoint returned a stub.
    """
    import argparse

    import mlx.core as mx
    import numpy as np

    import experiments.train_substrate_hi_nerv_mlx_local as trainer
    from tac.substrates._shared.numpy_portable_inflate import pack_state_dict_numpy
    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    # Use a tiny num_pairs to keep the test fast (architecture overrides via the
    # public harvester arch-override path).
    arch = {"num_pairs": 4}

    # Build a model with the SAME arch the harvester will rebuild, and save its
    # EMA state to the .npsd the harvester reads.
    ns = argparse.Namespace(**{**WH.PILOT_ARCH_DEFAULTS, **arch})
    cfg = trainer._config_from_args(ns)
    model = HinervSubstrateMLX(cfg)
    mx.eval(model.parameters())

    flat: dict[str, np.ndarray] = {}

    def _flatten(prefix: str, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _flatten(f"{prefix}.{i}" if prefix else str(i), v)
        elif hasattr(obj, "shape"):
            flat[prefix] = np.asarray(obj)

    _flatten("", model.parameters())
    blob = pack_state_dict_numpy(flat, dtype="fp32")

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    ema_state = ckpt_dir / "epoch000004_20260609T000000Z.ema_shadow.state"
    (ema_state.with_suffix(ema_state.suffix + ".npsd")).write_bytes(blob)
    live_state = ckpt_dir / "epoch000004_20260609T000000Z.live.state"
    (live_state.with_suffix(live_state.suffix + ".npsd")).write_bytes(blob)

    ck = WH.CheckpointRef(
        meta_path=ckpt_dir / "m.meta.json",
        global_epoch=4,
        role="periodic",
        loss=1.0,
        ema_state_path=ema_state,
        live_state_path=live_state,
        selection_metric_key="total",
        selection_metric_value=1.0,
        is_final=False,
    )

    out_dir = tmp_path / "export"
    # NB: param-count guard expects the FULL pilot config (228903). With a tiny
    # num_pairs override the count differs, so this test exercises the export
    # mechanics by passing the matching arch override + relaxing the guard via
    # the same arch the state was built with. We assert the round-trip + member.
    result = WH.export_backend_only_archive_from_checkpoint(
        ck, out_dir, arch=arch, hard_byte_ceiling=WH.PILOT_HARD_BYTE_CEILING
    )
    assert result.zip_members == ["x"]
    assert result.member_magic == "HIV1"
    assert result.archive_bytes > 0
    assert result.load_info["param_tensors_restored"] > 0
