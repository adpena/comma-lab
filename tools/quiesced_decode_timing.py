#!/usr/bin/env python
"""Quiesced decode timing: run a timing producer under a measured concurrency sampler,
assemble the local receipt, write the calibration, and build the decode_wall_clock leg.

Subcommands (each writes one JSON and prints its path):
  run        launch the producer command; writes CONCURRENCY.json + producer stdout
  assemble   producer receipt + CONCURRENCY.json -> decode_wall_clock.local.v1 receipt
  calibrate  local receipt + T4 receipt -> decode_wall_clock.calibration.v1
  leg        local + calibration (+ candidate T4 receipt) -> validated measured leg,
             optionally copied as the sidecar beside a seal

The rule that decides "competing" lives in tac.decode_timing_concurrency and is recorded in
every receipt. Nothing here retypes a timing number: wall seconds come from the producer,
T4 seconds from the Modal receipt, digests from the runtime on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(1, str(REPO))  # tac.decode_wall_clock imports experiments.contest_auth_eval for the T4 digest

from tac.decode_timing_concurrency import (  # noqa: E402
    ConcurrencyRule,
    assemble_local_receipt,
    run_producer,
    write_calibration,
)
from tac.decode_wall_clock import build_decode_wall_clock, validate_decode_wall_clock  # noqa: E402


def _save(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    print(path)
    return path


def cmd_run(args: argparse.Namespace) -> int:
    out = Path(args.out_dir).resolve()
    extra = {} if args.aggregate_cap_pcpu is None else {"aggregate_cap_pcpu": args.aggregate_cap_pcpu}
    rule = ConcurrencyRule(threshold_pcpu=args.threshold_pcpu, visible_pcpu=args.visible_pcpu,
                           ancestor_cap_pcpu=args.ancestor_cap_pcpu, interval_seconds=args.interval_seconds, **extra)
    command = shlex.split(args.producer)
    env = dict(os.environ)
    for item in args.env:
        key, _, value = item.partition("=")
        env[key] = value
    code, summary = run_producer(command, cwd=Path(args.cwd).resolve(), env=env,
                                 stdout_path=out / "producer.stdout.log", rule=rule,
                                 settle_seconds=args.settle_seconds, paused_pids=args.pause_pid,
                                 timeout_seconds=args.timeout_seconds, monitor_command=list(sys.argv))
    summary["producer_command"] = command
    summary["producer_returncode"] = code
    _save(out / "CONCURRENCY.json", summary)
    print(f"producer rc={code} competing_process_count={summary['competing_process_count']} "
          f"quiesced={summary['quiesced']} samples={summary['sample_count']}")
    return 0 if code == 0 else 3


def cmd_assemble(args: argparse.Namespace) -> int:
    producer_path = Path(args.producer_receipt).resolve()
    concurrency_path = Path(args.concurrency).resolve()
    doc = assemble_local_receipt(json.loads(producer_path.read_text()), json.loads(concurrency_path.read_text()),
                                 producer_receipt_path=producer_path, concurrency_receipt_path=concurrency_path,
                                 stage_checkpoint_dir=Path(args.stage_checkpoints) if args.stage_checkpoints else None,
                                 rule=ConcurrencyRule(threshold_pcpu=args.threshold_pcpu, impact_tolerance=args.impact_tolerance,
                                                      stage_tolerance=args.stage_tolerance))
    _save(Path(args.out), doc)
    block = doc["concurrency"]
    print(f"margin_time_basis={doc['margin_time_basis']} wall_seconds={doc['wall_seconds']} "
          f"frames={len(doc['frames'])} competing={block['competing_process_count']} "
          f"pcpu_competing={block.get('pcpu_competing_process_count')} "
          f"stage_max_deviation={block.get('stage_rates', {}).get('max_deviation')}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    doc = write_calibration(name=args.name, local_receipt_path=Path(args.local), t4_receipt_path=Path(args.t4_receipt))
    _save(Path(args.out), doc)
    print(f"cpu_to_t4_ratio={doc['cpu_to_t4_ratio']} local_600_seconds={doc['local_600_seconds']} "
          f"measured_t4_seconds={doc['measured_t4_seconds']}")
    return 0


def cmd_leg(args: argparse.Namespace) -> int:
    runtime_dir, archive = Path(args.runtime_dir).resolve(), Path(args.archive).resolve()
    leg = build_decode_wall_clock(local_receipt_path=Path(args.local), calibration_receipt_path=Path(args.calibration),
                                  runtime_dir=runtime_dir, archive_path=archive, enforce_margin=not args.no_enforce,
                                  candidate_t4_receipt_path=Path(args.candidate_t4_receipt) if args.candidate_t4_receipt else None)
    problems, observed = validate_decode_wall_clock(leg, runtime_dir=runtime_dir, archive_path=archive)
    leg_path = _save(Path(args.out), leg)
    print(f"projected_t4_decode_seconds={leg['projected_t4_decode_seconds']:.3f} observed={observed} "
          f"problems={problems or 'none'}")
    if problems:
        return 4
    if args.sidecar_for:
        seal = Path(args.sidecar_for).resolve()
        if not seal.is_file():
            raise SystemExit(f"seal missing: {seal}")
        sidecar = seal.with_name(seal.name + ".decode_wall_clock.json")
        shutil.copyfile(leg_path, sidecar)
        print(sidecar)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a producer under the concurrency sampler")
    run.add_argument("--producer", required=True, help="producer argv as one shell-quoted string")
    run.add_argument("--cwd", default=str(REPO))
    run.add_argument("--out-dir", required=True)
    run.add_argument("--env", action="append", default=[], help="KEY=VALUE for the producer (repeatable)")
    run.add_argument("--pause-pid", action="append", type=int, default=[], help="SIGSTOP this pid for the window")
    run.add_argument("--settle-seconds", type=float, default=600.0)
    run.add_argument("--timeout-seconds", type=float, default=3000.0)
    run.add_argument("--interval-seconds", type=float, default=20.0)
    run.add_argument("--threshold-pcpu", type=float, default=100.0, help="sampling-time competitor threshold; the verdict re-derives from the >= 5 %% visible list")
    run.add_argument("--visible-pcpu", type=float, default=5.0)
    run.add_argument("--ancestor-cap-pcpu", type=float, default=100.0)
    run.add_argument("--aggregate-cap-pcpu", type=float, default=None, help="default 100*(ncpu-4)/2")
    run.set_defaults(func=cmd_run)

    assemble = sub.add_parser("assemble", help="producer receipt + CONCURRENCY.json -> local.v1 receipt")
    assemble.add_argument("--producer-receipt", required=True)
    assemble.add_argument("--concurrency", required=True)
    assemble.add_argument("--out", required=True)
    assemble.add_argument("--stage-checkpoints", default=None,
                          help="producer's frame_checkpoints dir (stage_NNNN.npz); enables the impact-weighted count")
    assemble.add_argument("--stage-tolerance", type=float, default=0.05, help="per-stage slowdown vs median that is LISTED")
    assemble.add_argument("--impact-tolerance", type=float, default=0.01, help="total stage excess / wall that REFUSES")
    assemble.add_argument("--threshold-pcpu", type=float, default=100.0, help="competing %%CPU outside the instrumented stages")
    assemble.set_defaults(func=cmd_assemble)

    calibrate = sub.add_parser("calibrate", help="local receipt + T4 receipt -> calibration.v1")
    calibrate.add_argument("--name", required=True)
    calibrate.add_argument("--local", required=True)
    calibrate.add_argument("--t4-receipt", required=True)
    calibrate.add_argument("--out", required=True)
    calibrate.set_defaults(func=cmd_calibrate)

    leg = sub.add_parser("leg", help="build + validate the measured decode_wall_clock leg")
    leg.add_argument("--local", required=True)
    leg.add_argument("--calibration", required=True)
    leg.add_argument("--runtime-dir", required=True)
    leg.add_argument("--archive", required=True)
    leg.add_argument("--candidate-t4-receipt", default=None)
    leg.add_argument("--out", required=True)
    leg.add_argument("--sidecar-for", default=None, help="seal JSON to place the leg beside as <seal>.decode_wall_clock.json")
    leg.add_argument("--no-enforce", action="store_true", help="record problems instead of refusing (never for a seal)")
    leg.set_defaults(func=cmd_leg)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
