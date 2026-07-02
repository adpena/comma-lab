#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Canonical ONE-COMMAND launcher for a level-set witness run — the automated
value-generator for launch + observability (operator 2026-06-30: "That needs to
happen automatically in the future" + the guiding principle "build the automated
value-generator, not ad-hoc").

It folds the whole error-prone hand-assembly into ONE reusable, flag-validated,
no-silent-failure path:

  (a) DERIVE the config from the clip's GT cache (tac.witness_autoconfig) and
      FLAG-VALIDATE every emitted flag against the trainer's REAL argparse
      (never-invent-a-flag — NO-FAKE);
  (b) WRITE the command into ``<out_dir>/launch.sh`` (a SCRIPT — so the daemon
      cmd is a clean ``["bash", launch.sh]`` argv with NO word-split fragility,
      the exact class of bug that collapsed a whole command into argv[0]);
  (c) LAUNCH durably via tools/spawn_durable_daemon.py, which now AUTO-VERIFIES
      the child survived exec (dead launch -> nonzero + detailed debug);
  (d) VERIFY the perf-env line ``custom_grouped_backward active=true`` is in the
      log (the ~17x fast path; an unset env is a silent slow-run footgun);
  (e) CONFIRM the dashboard is up — once up it AUTO-TRACKS this (and every future)
      run every refresh tick, so no manual repoint/reload is ever needed for a
      new run (only NEW dashboard CODE needs a zero-downtime reload).

Determinism/resumability/observability are preserved: the emitted command carries
``--seed``, ``--ckpt-every``, ``--stage-checkpoints`` (resumable per-stage), and
the run is rendered live by the dashboard.

means != ends: this LAUNCHES an advisory [macOS-MLX] run. Only a byte-closed exact
n600 row < 0.19110 moves the pointer. Use ``--dry-run`` to emit + validate + write
launch.sh WITHOUT spawning (CPU-only, GPU-free, safe).

Usage:
    .venv/bin/python tools/launch_witness_run.py \\
        --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \\
        --num-pairs 600 --epochs 1000              # real launch
    .venv/bin/python tools/launch_witness_run.py ... --dry-run   # emit+validate only
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

from tac import witness_autoconfig as wac  # noqa: E402

_TRAINER = _REPO / "experiments/train_levelset_witness_realized_through_R_mlx.py"


# ───────────────────────── never-invent-a-flag guard ─────────────────────────
def real_trainer_flags() -> frozenset[str]:
    """The SET of real ``--flag`` names parsed from the trainer's argparse."""
    return frozenset(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', _TRAINER.read_text()))


def validate_emitted_flags(cfg, out_dir: str) -> tuple[bool, list[tuple[str, bool]]]:
    """Validate every emitted flag against the real argparse. Returns
    ``(all_pass, [(flag, ok), ...])``."""
    real = real_trainer_flags()
    results = [(flag, flag in real) for flag, _ in cfg.to_trainer_flags(out_dir)]
    return all(ok for _, ok in results), results


# ───────────────────────── launch.sh (no word-split fragility) ─────────────────────────
def build_launch_sh(cfg, out_dir: str, repo_root: Path | None = None) -> str:
    """Render the launch.sh body. The trainer command goes into a SCRIPT so the
    daemon cmd is ``bash launch.sh`` (2 clean tokens) — never a space-bearing
    single argv[0]. Includes the perf-env prefix (TAC_MLX_CUSTOM_GROUPED_BACKWARD=1)
    via cfg.to_command(perf_env=True)."""
    repo = str(repo_root or _REPO)
    cmd = cfg.to_command(out_dir, perf_env=True)
    return (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f"cd {repo}\n"
        f"{cmd}\n"
    )


def write_launch_sh(cfg, out_dir: Path, repo_root: Path | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    launch = out_dir / "launch.sh"
    launch.write_text(build_launch_sh(cfg, str(out_dir), repo_root))
    launch.chmod(0o755)
    return launch


# ───────────────────────── perf-env verification ─────────────────────────
def verify_perf_env(run_log: Path, timeout_s: float = 30.0, poll_s: float = 1.0) -> tuple[str, str | None]:
    """Wait (bounded) for the trainer's ``{"stage": "custom_grouped_backward", ...}``
    line and report whether the ~17x fast path is active. Returns
    ``("active"|"inactive"|"not_seen", raw_line|None)``. An unset env -> "inactive"
    (a silent slow-run footgun the launch gate must catch)."""
    deadline = time.time() + timeout_s
    run_log = Path(run_log)
    while time.time() < deadline:
        try:
            text = run_log.read_text(errors="replace")
        except OSError:
            text = ""
        for line in text.splitlines():
            if '"stage": "custom_grouped_backward"' in line or '"stage":"custom_grouped_backward"' in line:
                try:
                    active = bool(json.loads(line.strip()).get("active"))
                except Exception:
                    active = '"active": true' in line or '"active":true' in line
                return ("active" if active else "inactive", line.strip())
        time.sleep(poll_s)
    return ("not_seen", None)


# ───────────────────────── dashboard ensure ─────────────────────────
def _healthz(port: int, timeout: float = 3.0) -> int | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=timeout) as r:
            return r.getcode()
    except Exception:
        return None


def ensure_dashboard(port: int) -> bool:
    """Confirm the dashboard is serving. Once up it AUTO-TRACKS this run (the
    resolver re-resolves the newest run every refresh tick) — no manual repoint.
    Returns True iff healthz==200; otherwise prints an actionable command (does NOT
    auto-start, to stay contained)."""
    code = _healthz(port)
    if code == 200:
        print(f"[launch-witness] dashboard :{port} is UP (healthz 200) — it AUTO-TRACKS "
              f"this run every refresh tick (no manual repoint/reload needed).")
        return True
    print(f"[launch-witness] WARNING: dashboard :{port} NOT serving (healthz={code}). "
          f"This run is launched + durable regardless; to observe it, start/reload the "
          f"dashboard:\n    .venv/bin/python tools/dashboard_reload.py --port {port} --tau 300 --l7 600",
          file=sys.stderr)
    return False


# ───────────────────────── throughput gate (compute pass) ─────────────────────────
def _emitted_flag_names(cfg, out_dir: str) -> set[str]:
    return {flag for flag, _ in cfg.to_trainer_flags(out_dir)}


def _run_throughput_gate(cfg, out_dir, *, threshold_ms: float | None) -> int:
    """Pre-spawn SegNet fwd+bwd throughput assertion (the ~17x fast-path gate) + the
    conditional --compile-step bit-identical requirement. Returns 0 (proceed) / nonzero
    (REFUSE). NEVER blocks on unavailability (measured-slow only)."""
    try:
        from tac.local_acceleration.scorer_throughput_gate import (
            ABS_THRESHOLD_MS,
            evaluate_throughput,
        )
    except Exception as exc:  # helper import failure must not block the launch
        print(f"[launch-witness] WARNING: throughput gate unavailable (import: {exc}); "
              f"proceeding (perf-env log check still runs post-spawn).", file=sys.stderr)
        return 0
    thr = float(threshold_ms) if threshold_ms is not None else ABS_THRESHOLD_MS
    print(f"# throughput gate: measuring SegNet fwd+bwd (B=8, custom fast path) — REFUSE if median "
          f"> {thr:.0f}ms (measured ON~396 / OFF~6713)")
    verdict = evaluate_throughput(abs_threshold_ms=thr)
    if verdict.status == "fast":
        print(f"[launch-witness] throughput OK: SegNet fwd+bwd {verdict.segnet_fwd_bwd_ms:.1f}ms "
              f"<= {thr:.0f}ms (custom-grouped-backward fast path ACTIVE).")
    elif verdict.status == "unavailable":
        print(f"[launch-witness] WARNING: throughput gate could not measure "
              f"({verdict.reason}); proceeding (perf-env log check still runs post-spawn).",
              file=sys.stderr)
    else:  # slow
        print(f"[launch-witness] ERROR: REFUSING to launch — {verdict.reason}", file=sys.stderr)
        return 3
    # sub-part 3: --compile-step (or any --compile* flag) requires bit-identical compiled step.
    compile_flags = {f for f in _emitted_flag_names(cfg, str(out_dir)) if f.startswith("--compile")}
    if compile_flags:
        print(f"[launch-witness] NOTE: {sorted(compile_flags)} emitted — the trainer MUST assert "
              f"assert_compile_bit_identical at construction (compiled step == uncompiled + "
              f"deterministic). See tac.local_acceleration.scorer_throughput_gate."
              f"assert_compile_step_bit_identical.")
    return 0


# ───────────────────────── main ─────────────────────────
def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-cache", required=True, help="path to the clip's GT cache (e.g. .../gt_n600.npz)")
    ap.add_argument("--num-pairs", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--aggressive", action="store_true",
                    help="overfit=False: aggressive Whitney-floor mod-dim (rate-saving)")
    ap.add_argument("--config", default=None,
                    choices=["proven_base", "all_levers", "sealed_205"],
                    help="canonical named config resolved from the triality (tac.witness_autoconfig): "
                    "proven_base (attribution-clean baseline; the default when neither --config nor "
                    "--all-levers is given), all_levers (== --all-levers), or sealed_205 (the #205 "
                    "Phase-3 SEALED capstone argv — the deep-math-OPTIMAL all-levers base + the 4 "
                    "SEALED deltas mod-dim 32 / adam-beta2 0.999 / w-pose 1.0 + pose-carrier table; "
                    "reproduces the sealed §7 launch.sh BYTE-IDENTICALLY modulo --out-dir).")
    ap.add_argument("--all-levers", action="store_true",
                    help="emit the deep-math-OPTIMAL all-levers from-scratch config (#205 artifact); "
                    "equivalent to --config all_levers. "
                    "--render-aa none + analytic coverage-integrated lane-render-band (Wave D AA "
                    "correction; supersample DISQUALIFIED: hurts -49%% + decode over budget) + "
                    "persistence/topology loss + "
                    "island-birth amplification + annealed hosc 1->4 + l7 DEMOTED + verdict-pairs 0 + "
                    "mod-dim 19 (Whitney floor) + adam-beta2 0.9999999. Default OFF => attribution-clean "
                    "proven_base baseline.")
    ap.add_argument("--out-dir", default=None,
                    help="run out-dir (default: experiments/results/levelset_n<N>_witness_<utc>)")
    ap.add_argument("--label", default=None, help="daemon label (default: derived from out-dir)")
    ap.add_argument("--rss-cap-mb", type=int, default=90000,
                    help="per-arm RSS cap (MiB) for safe_run layer-3 (default 90000)")
    ap.add_argument("--min-free-gb", type=float, default=10.0,
                    help="OOM launch-preflight free-memory floor (default 10; operator-relaxed)")
    ap.add_argument("--verify-s", type=float, default=4.0,
                    help="seconds spawn_durable_daemon verifies the child survived exec")
    ap.add_argument("--perf-env-timeout-s", type=float, default=45.0,
                    help="seconds to wait for the custom_grouped_backward perf line")
    ap.add_argument("--skip-throughput-gate", action="store_true",
                    help="skip the pre-spawn SegNet fwd+bwd throughput micro-bench (the ~17x fast-path "
                    "assertion). Default runs it (a measured gate, not a flag-grep).")
    ap.add_argument("--throughput-threshold-ms", type=float, default=None,
                    help="override the SegNet fwd+bwd median ms gate (default 700; measured ON~396 / "
                    "OFF~6713). >threshold => REFUSE (custom-grouped-backward fast path not active).")
    ap.add_argument("--dashboard-port", type=int, default=8790)
    ap.add_argument("--no-dashboard", action="store_true", help="skip the dashboard up-check")
    ap.add_argument("--dry-run", action="store_true",
                    help="emit + flag-validate + write launch.sh, but DO NOT spawn (CPU-only, safe)")
    args = ap.parse_args(argv)

    overfit = not args.aggressive

    # Resolve the canonical named config (triality selector). Backward-compat: --all-levers is an
    # alias for --config all_levers; the historical default (neither given) is proven_base.
    config = args.config
    if config is None:
        config = "all_levers" if args.all_levers else "proven_base"
    elif args.all_levers and config != "all_levers":
        print(f"[launch-witness] ERROR: --config {config} conflicts with --all-levers "
              f"(--all-levers == --config all_levers); pass exactly one.", file=sys.stderr)
        return 2

    if config == "sealed_205":
        # The #205 P3 SEALED capstone config fixes its own knobs (mod-dim 32 etc.); overfit N/A.
        cfg = wac.derive_sealed_205_config(args.gt_cache, num_pairs=args.num_pairs,
                                           epochs=args.epochs)
    else:
        cfg = wac.derive_config(args.gt_cache, num_pairs=args.num_pairs,
                                overfit=overfit, epochs=args.epochs,
                                all_levers=(config == "all_levers"))

    out_dir = Path(args.out_dir) if args.out_dir else (
        _REPO / "experiments" / "results" / f"levelset_n{args.num_pairs}_witness_{_utc()}")
    label = args.label or f"levelset_witness_{out_dir.name}"

    print(f"# launch_witness_run {wac.ADVISORY_TAG}  pointer 0.19110 UNMOVED")
    print(f"# clip={args.gt_cache} num_pairs={args.num_pairs} epochs={args.epochs} "
          f"overfit={overfit} config={config}")
    print(f"# out_dir={out_dir}")
    if not (_REPO / args.gt_cache).exists() and not Path(args.gt_cache).exists():
        print(f"# NOTE: gt-cache {args.gt_cache} not found on disk -> gt regen required at launch",
              file=sys.stderr)

    # (a) FLAG-VALIDATE (never-invent-a-flag, NO-FAKE) — refuse before writing anything.
    all_pass, results = validate_emitted_flags(cfg, str(out_dir))
    n = len(results)
    n_ok = sum(1 for _, ok in results if ok)
    print(f"# flag validation: {n_ok}/{n} flags exist in the trainer argparse")
    if not all_pass:
        bad = [f for f, ok in results if not ok]
        print(f"[launch-witness] ERROR: REFUSING to launch — emitted invented flag(s) "
              f"not in the trainer argparse: {bad}", file=sys.stderr)
        return 2

    # (b) WRITE launch.sh (script-based command — no word-split fragility).
    launch_sh = write_launch_sh(cfg, out_dir)
    run_log = out_dir / "run.log"
    print(f"# wrote {launch_sh}")

    if args.dry_run:
        print("# DRY-RUN: launch.sh written + flags validated; NOT spawning. "
              "Re-run without --dry-run to launch.")
        return 0

    # (b2) THROUGHPUT GATE (compute pass): a MEASURED SegNet fwd+bwd micro-bench (B=8), NOT a flag-grep.
    # REFUSE if the custom-grouped-backward ~17x fast path is not actually active on this machine
    # (median > threshold => the ~6713ms reference accumulator, not the ~396ms fast path). Unavailable
    # (no MLX/scorer/GPU) => WARN, never block. Sub-part 3: if a --compile* flag is emitted, the compiled
    # step must be bit-identical (asserted at trainer construction; the gate flags the requirement here).
    if not args.skip_throughput_gate:
        gate_rc = _run_throughput_gate(cfg, out_dir, threshold_ms=args.throughput_threshold_ms)
        if gate_rc != 0:
            return gate_rc

    # (c) LAUNCH durably (spawn_durable_daemon auto-verifies the child survived exec).
    import spawn_durable_daemon as sdd  # late import: heavy-ish + only needed for real launch
    spawn_argv = [
        "--log", str(run_log), "--label", label,
        "--rss-cap-mb", str(int(args.rss_cap_mb)),
        "--min-free-gb", str(float(args.min_free_gb)),
        "--verify-s", str(float(args.verify_s)),
        "--", "bash", str(launch_sh),
    ]
    rc = sdd.main(spawn_argv)
    if rc != 0:
        print(f"[launch-witness] ERROR: durable launch FAILED (spawn rc={rc}); see the "
              f"detailed debug above and the log: {run_log}", file=sys.stderr)
        return rc

    # (d) VERIFY the perf-env fast path (loud warning on the silent-slow footgun).
    status, line = verify_perf_env(run_log, timeout_s=args.perf_env_timeout_s)
    if status == "active":
        print(f"[launch-witness] perf-env OK: custom_grouped_backward ACTIVE (~17x fast path).")
    elif status == "inactive":
        print(f"[launch-witness] WARNING: custom_grouped_backward is INACTIVE — the run will be "
              f"~17x SLOW (TAC_MLX_CUSTOM_GROUPED_BACKWARD unset/disabled). line: {line}",
              file=sys.stderr)
    else:
        print(f"[launch-witness] WARNING: could not confirm the custom_grouped_backward perf line "
              f"within {args.perf_env_timeout_s:.0f}s (run is alive; check {run_log}).",
              file=sys.stderr)

    # (e) CONFIRM dashboard observability (auto-tracks this run once up).
    if not args.no_dashboard:
        ensure_dashboard(args.dashboard_port)

    print(f"[launch-witness] LAUNCHED + VERIFIED: label={label} log={run_log}")
    print(f"  stop:   .venv/bin/python tools/spawn_durable_daemon.py --stop {label}")
    print(f"  status: .venv/bin/python tools/spawn_durable_daemon.py --status")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
