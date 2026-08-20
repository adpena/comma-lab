#!/usr/bin/env python3
"""fire_local_advisory.py — the ONE deterministic path from a candidate runtime
to a LOCAL CPU advisory n600 row.

Sister of tools/fire_modal_auth_eval.py (the paid-axis firer). Local advisory
launches were hand-assembled argv until 2026-08-18; each launch re-rolled the
dice on env completeness and the dice came up wrong twice:

  * ck1 attempt_0001 omitted the pyshim PATH prefix -> the sz1-lineage
    inflate.sh's bare `python` probe fell into its uv bootstrap branch ->
    `uv pip install --python python` -> rc=2 at t=0.0s (#929 class, locally).
  * V7's advisory omitted PYTHONDONTWRITEBYTECODE=1 -> __pycache__ written
    into the ExFAT upstream mirror -> AppleDouble .pyc sidecar -> compliance
    hasher refused every subsequent advisory at t=5s (#1122 class).

The cure is structural: this tool CARRIES the env. The detector zeroes on it —
with every advisory launched here, neither class can recur.

Env contract (composed, never hand-typed):
  * PATH is prefixed with a freshly GENERATED exec-wrapper `python` shim
    resolving to the repo venv interpreter. Generated per-launch into the
    attempt dir from sys.executable — never a stale copy, never a symlink
    (a symlink named `python` breaks venv detection; the shim law).
  * PYTHONDONTWRITEBYTECODE=1.

Runtime-knob passthrough (`--env KEY=VALUE`, repeatable). Some candidate
runtimes read their own decode-path knobs from the environment — e.g.
`F26_TOKEN_DECODER=native-hpac` selects the ddm_wc2c split token decoder. Those
have to reach the DETACHED child, and before 2026-08-20 this tool carried only
the two keys above, so the only way to vary a knob was to hand-assemble the
launcher argv — which is the error factory this tool exists to replace.

The passthrough may NOT set a carried key. Letting `--env` overwrite PATH or
PYTHONDONTWRITEBYTECODE would reopen exactly the two failure classes documented
above, so a collision is a REFUSAL rather than a precedence rule: the detector
zeroes on the cure. Passthrough keys are recorded separately in the launch
manifest, because the env IS the instrument and a receipt that cannot say which
decode path was requested cannot grade the row.

The shipped runtime tree is NEVER modified — inflate.sh keeps its bare
`python` (correct in the contest container); the shim is the host adaptation.

Launch mechanics: tools/safe_run.py (RSS/walltime caps + status receipt)
inside tools/launch_detached_process.py (detached, provenance manifest,
done-receipt -> monitor pickup). The argv mirrors the proven keep01/V-series
advisory template (ddm_sa1 launch manifests).
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_UPSTREAM_MIRROR = Path("/Volumes/APDataStore/pact/upstream_eval_mirror_20260815")
# The proven PATH tail from the successful keep01/V-series advisory launches.
PATH_TAIL = (
    f"{Path.home()}/.local/bin:/opt/homebrew/bin:/usr/local/bin:"
    "/usr/bin:/bin:/usr/sbin:/sbin"
)


def _refuse(msg: str) -> int:
    print(f"FATAL: {msg}", file=sys.stderr)
    return 2


# The keys this tool CARRIES. A passthrough may never set one of them; see the
# module docstring for why that is a refusal and not a precedence rule.
CARRIED_ENV_KEYS = frozenset({"PATH", "PYTHONDONTWRITEBYTECODE"})


def parse_passthrough_env(pairs: list[str]) -> dict[str, str]:
    """Parse repeatable ``KEY=VALUE`` runtime knobs, fail-closed.

    Refuses a missing ``=``, an empty key, a duplicate key (which would make the
    effective value depend on argv order), and any key this tool carries.
    """
    out: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"--env expects KEY=VALUE, got {pair!r}")
        if key in CARRIED_ENV_KEYS:
            raise ValueError(
                f"--env may not set {key}: this tool carries it, and overriding it "
                "reopens the ck1/V7 failure classes it exists to close"
            )
        if key in out:
            raise ValueError(f"--env {key} given twice; the effective value would be argv-order dependent")
        out[key] = value
    return out


def generate_pyshim(shim_dir: Path, interpreter: Path) -> Path:
    """Write an exec-wrapper `python` shim (NEVER a symlink) into shim_dir.

    `interpreter` must be the UNRESOLVED venv path (e.g. .venv/bin/python).
    Resolving it would follow the venv symlink to the base interpreter and
    silently lose site-packages — the trap the self-test below catches: the
    shim must land in the SAME environment (sys.prefix) as the launcher.
    """
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "python"
    shim.write_text(f'#!/bin/sh\nexec "{interpreter}" "$@"\n')
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    probe = subprocess.run(
        [str(shim), "-c", "import sys; print(sys.prefix)"],
        capture_output=True, text=True, timeout=30,
    )
    if probe.returncode != 0 or Path(probe.stdout.strip()) != Path(sys.prefix):
        raise RuntimeError(
            f"pyshim self-test failed (environment mismatch): rc={probe.returncode} "
            f"shim prefix={probe.stdout.strip()!r} launcher prefix={sys.prefix!r} "
            f"err={probe.stderr!r}"
        )
    return shim


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runtime-dir", required=True,
                   help="the staged runtime tree (inflate.sh + inflate.py + archive)")
    p.add_argument("--archive", default=None,
                   help="default: <runtime-dir>/archive.zip")
    p.add_argument("--attempt-dir", required=True,
                   help="fresh output dir for this attempt (work/, receipts, launch/)")
    p.add_argument("--label", required=True,
                   help="short label: safe_run label AND the done-receipt name")
    p.add_argument("--upstream-dir", default=str(DEFAULT_UPSTREAM_MIRROR))
    p.add_argument("--rss-mb", type=int, default=32768)
    p.add_argument("--timeout", type=int, default=21600)
    p.add_argument("--projected-gib", type=int, default=12)
    p.add_argument("--inflate-timeout", type=int, default=5400)
    p.add_argument("--evaluate-timeout", type=int, default=14400)
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="runtime knob passed to the detached child (repeatable); "
                        "may not set PATH or PYTHONDONTWRITEBYTECODE")
    p.add_argument("--receipt-supersede", action="store_true",
                   help="tombstone an unconsumed prior done-receipt of the same name")
    p.add_argument("--dry-run", action="store_true",
                   help="print the composed env + argv, launch nothing")
    args = p.parse_args()

    try:
        passthrough = parse_passthrough_env(args.env)
    except ValueError as error:
        return _refuse(str(error))

    runtime = Path(args.runtime_dir).resolve()
    archive = Path(args.archive).resolve() if args.archive else runtime / "archive.zip"
    attempt = Path(args.attempt_dir).resolve()
    upstream = Path(args.upstream_dir).resolve()
    inflate_sh = runtime / "inflate.sh"
    video_names = upstream / "public_test_video_names.txt"

    for what, path in (("runtime dir", runtime), ("archive", archive),
                       ("inflate.sh", inflate_sh), ("upstream mirror", upstream),
                       ("video names file", video_names)):
        if not path.exists():
            return _refuse(f"{what} missing: {path}")
    if attempt.exists() and any(attempt.iterdir()):
        return _refuse(f"attempt dir not empty (mint a fresh one): {attempt}")
    interpreter = Path(sys.executable)  # UNRESOLVED: preserve the venv (see generate_pyshim)
    if args.dry_run:
        # A dry run is an observation, not an attempt.  Exercise the shim
        # self-test in success-only scratch, but point the printed plan at the
        # path a real launch would create.  In particular, do not create the
        # attempt dir, pyshim, or launch manifest (rv15 F14).
        with tempfile.TemporaryDirectory(prefix="fire_local_advisory_dryrun_") as scratch:
            generate_pyshim(Path(scratch) / "pyshim", interpreter)
        shim = attempt / "pyshim" / "python"
    else:
        attempt.mkdir(parents=True, exist_ok=True)
        shim = generate_pyshim(attempt / "pyshim", interpreter)

    env_pairs = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": f"{shim.parent}:{PATH_TAIL}",
    }
    # Carried keys are written last so no ordering accident can let a
    # passthrough shadow one; parse_passthrough_env has already refused any.
    env_pairs = {**passthrough, **env_pairs}
    inner = [
        str(interpreter), "tools/safe_run.py",
        "--rss-mb", str(args.rss_mb), "--timeout", str(args.timeout),
        "--poll", "5", "--label", args.label,
        "--projected-gib", str(args.projected_gib),
        "--child-pidfile", str(attempt / "safe_child.pid"),
        "--status-receipt", str(attempt / "safe_run_status.json"),
        "--",
        str(interpreter), "experiments/contest_auth_eval.py",
        "--archive", str(archive),
        "--inflate-sh", str(inflate_sh),
        "--upstream-dir", str(upstream),
        "--video-names-file", str(video_names),
        "--device", "cpu", "--inflate-device", "cpu",
        "--work-dir", str(attempt / "work"), "--keep-work-dir",
        "--json-out", str(attempt / "contest_auth_eval.json"),
        "--inflate-timeout", str(args.inflate_timeout),
        "--evaluate-timeout", str(args.evaluate_timeout),
    ]
    launcher = [
        str(interpreter), "tools/launch_detached_process.py",
        "--output-dir", str(attempt / "launch"),
        "--cwd", str(REPO_ROOT),
        "--purpose", f"local CPU advisory n600 [{args.label}] via fire_local_advisory (canonical env)",
        "--authority", "advisory; paid-axis fires gate on this row",
        "--done-receipt", args.label,
    ]
    for k, v in env_pairs.items():
        launcher += ["--env", f"{k}={v}"]
    if args.receipt_supersede:
        launcher.append("--receipt-supersede")
    launcher.append("--")
    launcher += inner

    manifest = {
        "schema": "fire_local_advisory.v1",
        "runtime_dir": str(runtime),
        "archive": str(archive),
        "attempt_dir": str(attempt),
        "label": args.label,
        "env": env_pairs,
        "passthrough_env": passthrough,
        "pyshim": str(shim),
        "pyshim_interpreter": str(interpreter),
        "argv": launcher,
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=1))
        return 0
    (attempt / "ADVISORY_LAUNCH.json").write_text(json.dumps(manifest, indent=1))

    proc = subprocess.run(launcher, cwd=REPO_ROOT, env={**os.environ, **env_pairs})
    if proc.returncode != 0:
        return _refuse(f"detached launcher failed rc={proc.returncode}")
    print(f"LAUNCHED: {args.label} attempt={attempt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
