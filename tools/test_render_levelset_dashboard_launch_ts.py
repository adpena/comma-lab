"""Regression tests for the run-SWAP boundary fix in render_levelset_dashboard.

The bug (fixed 2026-07-06): at the instant a run is swapped (old run killed, new
run launched), the just-killed run's log can flush a final write whose mtime
briefly beats the fresh run's first write. The resolvers sorted by mtime, so they
latched onto the DEAD run and rendered it "stale" until the new run produced its
first verdict (~19 min later, through structured_init).

The fix: order run logs by the LAUNCH timestamp embedded in the filename
(``levelset_<label>_<YYYYMMDDTHHMMSSZ>.log``) — the newest launch is
unambiguously the current run (we run serially, never 2 concurrent), immune to
the mtime race.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_levelset_dashboard as rld  # noqa: E402


def _write(p: Path, *, verdict: bool) -> None:
    """A run log always carries the early ``gt`` marker; a verdict-bearing log
    also carries a ``verdict`` line."""
    lines = ['{"stage": "gt"}']
    if verdict:
        lines.append('{"stage": "verdict", "epoch": 0, "d_seg": 0.5}')
    p.write_text("\n".join(lines) + "\n")


def test_launch_ts_extracts_trailing_timestamp():
    assert rld._launch_ts(Path("levelset_mod32cap_20260706T115614Z.log")) == "20260706T115614Z"
    assert rld._launch_ts(Path("/a/b/levelset_paintseedON_ab_20260706T032057Z.log")) == "20260706T032057Z"


def test_launch_ts_reads_stamp_from_DIR_when_filename_is_run_log():
    """The regression fix: a bare ``run.log`` in a timestamped dir must get its
    launch stamp from the DIRECTORY (full-path search), else it sorts as '' and
    loses to any timestamped-filename .log — surfacing an ancient run."""
    p = Path("experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/run.log")
    assert rld._launch_ts(p) == "20260706T115554Z"


def test_launch_ts_takes_max_when_multiple_stamps_in_path():
    # resumed run: dir stamp + file stamp -> take the most-recent (max)
    p = Path("experiments/results/levelset_20260705T100000Z/levelset_20260706T120000Z.log")
    assert rld._launch_ts(p) == "20260706T120000Z"


def test_launch_ts_none_when_no_token():
    assert rld._launch_ts(Path("some_other.log")) is None
    assert rld._launch_ts(Path("levelset_nolabel.log")) is None


def test_multi_glob_unions_comma_separated_patterns(tmp_path):
    d1 = tmp_path / "runs" / "levelset_a_20260706T010000Z"
    d1.mkdir(parents=True)
    (d1 / "run.log").write_text("x\n")
    tmpd = tmp_path / "omxtmp"
    tmpd.mkdir()
    (tmpd / "levelset_b_20260706T020000Z.log").write_text("y\n")
    g = f"{tmp_path}/runs/levelset_*/*.log,{tmpd}/levelset_*.log"
    got = sorted(Path(p).name for p in rld._multi_glob(g))
    assert got == ["levelset_b_20260706T020000Z.log", "run.log"]


def test_resolvers_pick_live_omxtmp_run_over_stale_rundir(tmp_path):
    """End-to-end of the fix: an OLDER run-dir run.log vs a NEWER .omx/tmp live
    daemon log across a comma-glob — the newer LAUNCH (live run) must win even
    though the run.log has a fresher mtime."""
    d = tmp_path / "runs" / "levelset_old_20260706T010000Z"
    d.mkdir(parents=True)
    rundir_log = d / "run.log"
    _write(rundir_log, verdict=True)
    tmpd = tmp_path / "omxtmp"
    tmpd.mkdir()
    live = tmpd / "levelset_live_20260706T115614Z.log"
    _write(live, verdict=True)
    os.utime(live, (1000.0, 1000.0))
    os.utime(rundir_log, (2000.0, 2000.0))  # stale run-dir has fresher mtime
    g = f"{tmp_path}/runs/levelset_*/*.log,{tmpd}/levelset_*.log"
    assert rld._resolve_watched_log(None, g).name == live.name
    assert rld._resolve_run_log(None, g).name == live.name


def test_swap_boundary_new_warming_run_wins_despite_older_mtime(tmp_path):
    """The decisive case: OLD stopped run (has verdict) has a NEWER mtime than the
    NEW warming run (no verdict yet). The launch-ts ordering must still surface the
    NEW run, so the dashboard never latches onto the dead run as 'stale'."""
    old = tmp_path / "levelset_paintseed_20260706T032057Z.log"   # earlier launch, stopped
    new = tmp_path / "levelset_mod32cap_20260706T115614Z.log"    # later launch, warming (no verdict)
    _write(old, verdict=True)
    _write(new, verdict=False)
    # Simulate the race: old (just-killed) log's final flush is NEWER by mtime.
    os.utime(new, (1000.0, 1000.0))
    os.utime(old, (2000.0, 2000.0))  # old mtime strictly newer

    glob = str(tmp_path / "levelset_*.log")
    run_latest = rld._resolve_run_log(None, glob)
    verdict_latest = rld._resolve_watched_log(None, glob)

    # run resolver follows the NEWEST LAUNCH (the warming mod32cap), not newest mtime.
    assert run_latest is not None and run_latest.name == new.name
    # verdict resolver (only the old one has a verdict) still returns the old one —
    # correct, since the new run has no verdict yet. The server's warming block then
    # compares launch-ts and follows `new` because it is the newer launch.
    assert verdict_latest is not None and verdict_latest.name == old.name
    assert rld._launch_ts(run_latest) >= rld._launch_ts(verdict_latest)


def test_verdict_resolver_prefers_newest_launch_when_both_have_verdicts(tmp_path):
    """After the new run gets its first verdict, both logs are verdict-bearing.
    The resolver must pick the newest LAUNCH even if the old (stopped) run's mtime
    is momentarily newer."""
    old = tmp_path / "levelset_paintseed_20260706T032057Z.log"
    new = tmp_path / "levelset_mod32cap_20260706T115614Z.log"
    _write(old, verdict=True)
    _write(new, verdict=True)
    os.utime(new, (1000.0, 1000.0))
    os.utime(old, (2000.0, 2000.0))  # old mtime newer, but it's the stopped run

    glob = str(tmp_path / "levelset_*.log")
    assert rld._resolve_watched_log(None, glob).name == new.name
