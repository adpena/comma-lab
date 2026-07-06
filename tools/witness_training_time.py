#!/usr/bin/env python3
"""Measure + compare witness (#205) training WALL-CLOCK across runs/arms.

Training time is a first-class campaign metric: as we turn on more levers
(micro-batch-pairs, the derived curriculum, …) we want to see the s/epoch and
total-wall delta next to the d_seg delta, per arm. This reads a run's log (no
launches, read-only) and reports:

  * seconds/epoch  — from the verdict timestamp deltas (the honest available
    signal; it includes any async-verdict overlap, labelled as such)
  * total wall-clock (verdict-span) and epochs completed
  * best d_seg so far
  * the speed-relevant config knobs that are ON/OFF (grouped-backward,
    micro-batch-pairs vs serial accum, mlx device, mem-probe, num-pairs,
    render size, mod/hidden dims) + a short config hash

Pass one or more run dirs and/or log files to COMPARE them side by side; with
no args it picks the newest levelset run. Every parsed run is also appended to
``.omx/state/witness_training_time.jsonl`` so the campaign accumulates a
speed ledger (per "Results must become system intelligence").

Usage
-----
    tools/witness_training_time.py                     # newest run
    tools/witness_training_time.py --all               # all recent runs, compared
    tools/witness_training_time.py <run_dir> <log> ... # explicit set, compared
    tools/witness_training_time.py --json              # machine-readable
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_RESULTS = _REPO / "experiments" / "results"
_TMP = _REPO / ".omx" / "tmp"
_LEDGER = _REPO / ".omx" / "state" / "witness_training_time.jsonl"

# speed-relevant knobs to surface (flag -> short label); presence/absence matters
_PERF_FLAGS = [
    ("micro-batch-pairs", r"--micro-batch-pairs\s+(\d+)"),
    ("accum-pairs", r"--accum-pairs\s+(\d+)"),
    ("num-pairs", r"--num-pairs\s+(\d+)"),
    ("mlx-device", r"--mlx-device\s+(\w+)"),
    ("render-h", r"--render-h\s+(\d+)"),
    ("mod-dim", r"--mod-dim\s+(\d+)"),
    ("hidden-dim", r"--hidden-dim\s+(\d+)"),
    ("eikonal-weight", r"--eikonal-weight\s+([\d.]+)"),
]


def _parse_ts(s: str) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _find_log_for_dir(run_dir: Path) -> Path | None:
    """A run dir logs either to <dir>/run.log or to .omx/tmp/levelset_<label>_*.log."""
    inline = run_dir / "run.log"
    if inline.exists():
        return inline
    # label = the run-dir name with the levelset_n600_witness_ prefix stripped,
    # minus its trailing timestamp; match the closest .omx/tmp log by that label.
    name = run_dir.name
    m = re.match(r"levelset_n600_witness_(.*)_(\d{8}T\d{6}Z)$", name)
    if not m or not _TMP.is_dir():
        return None
    label, stamp = m.group(1), m.group(2)
    cands = sorted(_TMP.glob(f"levelset_{label}_*.log"))
    if not cands:
        return None
    # pick the one whose timestamp is closest to the run-dir stamp
    want = _parse_ts(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}Z")

    def _key(p: Path) -> float:
        lm = re.search(r"_(\d{8}T\d{6}Z)\.log$", p.name)
        if not lm or want is None:
            return 0.0
        s = lm.group(1)
        t = _parse_ts(f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[9:11]}:{s[11:13]}:{s[13:15]}Z")
        return abs((t - want).total_seconds()) if t else 1e18

    return min(cands, key=_key)


def _find_launch_for_log(log: Path) -> Path | None:
    """A log in .omx/tmp maps back to a run dir's launch.sh by label."""
    if (log.parent / "launch.sh").exists():       # log lives inside the run dir
        return log.parent / "launch.sh"
    m = re.match(r"levelset_(.*)_(\d{8}T\d{6}Z)\.log$", log.name)
    if not m:
        return None
    label = m.group(1)
    cands = sorted(_RESULTS.glob(f"levelset_n600_witness_{label}_*"))
    for d in reversed(cands):                      # newest matching dir first
        if (d / "launch.sh").exists():
            return d / "launch.sh"
    return None


def resolve_pair(arg: Path) -> tuple[Path | None, Path | None]:
    """Given a run dir OR a log file, return (log, launch.sh)."""
    if arg.is_dir():
        return _find_log_for_dir(arg), (arg / "launch.sh" if (arg / "launch.sh").exists() else None)
    if arg.is_file():
        return arg, _find_launch_for_log(arg)
    return None, None


def parse_flags(launch: Path | None) -> dict:
    flags: dict = {"grouped_backward": None, "mem_probe": None}
    if launch is None or not launch.exists():
        return flags
    text = launch.read_text()
    flags["grouped_backward"] = "1" if "TAC_MLX_CUSTOM_GROUPED_BACKWARD=1" in text else "0"
    flags["mem_probe"] = "1" if "TAC_MEM_PROBE=1" in text else "0"
    for label, pat in _PERF_FLAGS:
        m = re.search(pat, text)
        flags[label] = m.group(1) if m else None      # None = flag absent
    return flags


def parse_log(log: Path) -> dict:
    verdicts: list[tuple[int, dt.datetime, float]] = []
    max_epoch = 0
    best_dseg = None
    last_alive = None
    for line in log.read_text().splitlines():
        if '"stage"' not in line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        st = d.get("stage")
        if st == "loss_terms":
            max_epoch = max(max_epoch, int(d.get("ep", 0)))
        elif st == "verdict":
            ts = _parse_ts(d.get("ts", ""))
            if ts is not None:
                verdicts.append((int(d["epoch"]), ts, float(d.get("d_seg", "nan"))))
                last_alive = ts
        elif st == "checkpoint" and d.get("kind") == "best":
            dv = float(d.get("d_seg", "nan"))
            best_dseg = dv if best_dseg is None else min(best_dseg, dv)

    # s/epoch from consecutive verdict deltas (the honest available signal)
    rates = []
    for (ea, ta, _), (eb, tb, _) in zip(verdicts, verdicts[1:]):
        de = eb - ea
        if de > 0:
            rates.append((tb - ta).total_seconds() / de)
    s_per_epoch = statistics.median(rates) if rates else None
    total_wall = ((verdicts[-1][1] - verdicts[0][1]).total_seconds()
                  if len(verdicts) >= 2 else None)
    if best_dseg is None and verdicts:
        best_dseg = min(v[2] for v in verdicts if v[2] == v[2])  # skip nan

    return {
        "epochs_seen": max_epoch,
        "verdict_count": len(verdicts),
        "s_per_epoch": s_per_epoch,
        "total_wall_span_s": total_wall,
        "best_dseg": best_dseg,
    }


def config_hash(flags: dict) -> str:
    payload = json.dumps({k: flags.get(k) for k in sorted(flags)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def analyze(arg: Path) -> dict | None:
    log, launch = resolve_pair(arg)
    if log is None:
        print(f"  ! {arg}: no log found", file=sys.stderr)
        return None
    flags = parse_flags(launch)
    m = parse_log(log)
    on = [f"{k}={v}" for k, v in flags.items() if v not in (None, "0")]
    row = {
        "run": (launch.parent.name if launch else log.stem),
        "log": str(log),
        "config_hash": config_hash(flags),
        **m,
        "flags_on": on,
        "flags": flags,
    }
    return row


def _fmt(x, unit="", nd=1):
    return "—" if x is None else (f"{x:.{nd}f}{unit}")


def print_table(rows: list[dict]) -> None:
    hdr = f"{'run':<44} {'ep':>5} {'s/epoch':>9} {'wall':>8} {'best d_seg':>11}  perf-on"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        wall = None if r["total_wall_span_s"] is None else r["total_wall_span_s"] / 3600
        perf = []
        f = r["flags"]
        perf.append("grpBWD" if f.get("grouped_backward") == "1" else "noGrpBWD")
        perf.append(f"micro={f['micro-batch-pairs']}" if f.get("micro-batch-pairs")
                    else f"accum={f.get('accum-pairs') or '?'}(serial)")
        if f.get("mem_probe") == "1":
            perf.append("mem-probe")
        if f.get("eikonal-weight") not in (None, "0"):
            perf.append(f"eik={f['eikonal-weight']}")
        print(f"{r['run'][:44]:<44} {r['epochs_seen']:>5} "
              f"{_fmt(r['s_per_epoch'],'s'):>9} {_fmt(wall,'h',2):>8} "
              f"{_fmt(r['best_dseg'],'',6):>11}  {' '.join(perf)}")


def append_ledger(rows: list[dict]) -> None:
    _LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with _LEDGER.open("a") as fh:
        for r in rows:
            fh.write(json.dumps({k: r[k] for k in
                                 ("run", "config_hash", "epochs_seen", "s_per_epoch",
                                  "total_wall_span_s", "best_dseg", "flags_on")}) + "\n")


def _run_stamp(p: Path) -> str:
    """The embedded YYYYmmddTHHMMSSZ (chronological key; labels can vary in length)."""
    m = re.search(r"(\d{8}T\d{6}Z)", p.name)
    return m.group(1) if m else ""


def newest_run() -> Path | None:
    cands = sorted(_RESULTS.glob("levelset_n600_witness_*"), key=_run_stamp)
    return cands[-1] if cands else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="run dirs and/or log files to compare")
    ap.add_argument("--all", action="store_true", help="analyze all recent levelset runs")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--no-ledger", action="store_true", help="do not append to the speed ledger")
    args = ap.parse_args()

    targets: list[Path]
    if args.all:
        targets = sorted(_RESULTS.glob("levelset_n600_witness_*"),
                         key=lambda p: p.name)
    elif args.paths:
        targets = [Path(p) for p in args.paths]
    else:
        nr = newest_run()
        if nr is None:
            print("no levelset runs found", file=sys.stderr)
            return 1
        targets = [nr]

    rows = [r for r in (analyze(t) for t in targets) if r is not None]
    if not rows:
        return 1

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print_table(rows)
        print("\n(s/epoch = between-verdict delta, includes async-verdict overlap; "
              "'best d_seg' is the min so far)")

    if not args.no_ledger:
        append_ledger(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
