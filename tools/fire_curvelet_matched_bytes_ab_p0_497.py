#!/usr/bin/env python3
"""p0_497 QUEUED matched-COUNTED-BYTES curvelet-vs-Fourier n600 A/B — governed fire script.

PREPARED_NOT_FIRED. This script exists so the decisive A/B fires the moment the live c2
run (`experiments/results/levelset_n600_witness_20260717T113932Z`) completes, with ZERO
signal loss and ZERO admission overrides. It NEVER auto-fires: CONTAINMENT — heavy local
launches require explicit operator GO (`--operator-go`), and even then each arm runs its
governed `tools/launch_witness_run.py --dry-run` first and refuses on any governor /
storage / seal / DSL refusal. A refusal is information, not an obstacle.

The two arms (both compile from the SAME sealed named config; the ONLY intended delta is
the basis lever — the treatment additionally forces `--render-aa none` because scalar
Fourier IPE is fail-closed for the literal family per the crux SPEC):

  control:   --config v9_cgauge_ideal_mod32                                  (legacy Fourier)
  treatment: --config v9_cgauge_ideal_mod32 --dsl-lever LiteralPolarCurveletBasis

Memory: the two arms each need ~63-75 GiB — they are fired SEQUENTIALLY (one per
invocation, `--arm control|treatment`), never concurrently with each other or with c2.

After BOTH arms complete + are independently byte-closed, the equal-byte law runs via
`tools/curvelet_equal_byte_ab_receipt.py match` (deterministic ZIP_STORED rate_match.bin
padding to EXACT equal final ZIP bytes) -> inflate both matched archives -> official
scorer batch 32, n600, declared contest axis -> `finalize` emits the only admissible
transfer verdict (`curvelet_equal_archive_transfer_v1`; pointer_authorized=false).

Pointer discipline: 0.19108 UNMOVED until an exact `upstream/evaluate.py` row says
otherwise; this harness never writes the pointer.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
C2_RUN_DIR_NAME = "levelset_n600_witness_20260717T113932Z"
QUIESCENT_SECONDS = 30 * 60  # newest c2 run-dir mtime must be at least this old

ARMS = {
    "control": {
        "out_dir": "/Volumes/VertigoDataTier/pact/experiments/results/curvelet_matched_bytes_ab_control_n600_seed0",
        "label": "p0497-matched-bytes-control-n600",
        "extra": [],
        "purpose": "p0_497 matched-bytes A/B: equal-config legacy-Fourier control; operator-GO only",
    },
    "treatment": {
        "out_dir": "/Volumes/VertigoDataTier/pact/experiments/results/curvelet_matched_bytes_ab_literal_n600_seed0",
        "label": "p0497-matched-bytes-literal-n600",
        "extra": ["--dsl-lever", "LiteralPolarCurveletBasis"],
        "purpose": "p0_497 matched-bytes A/B: literal polar-curvelet + native-orient treatment; operator-GO only",
    },
}


def _c2_status() -> dict:
    """The live c2 run must be COMPLETE (or absent) before either arm fires."""
    candidates = [
        REPO / "experiments" / "results" / C2_RUN_DIR_NAME,
        Path("/Users/adpena/Projects/pact/experiments/results") / C2_RUN_DIR_NAME,
    ]
    run_dir = next((c for c in candidates if c.exists()), None)
    if run_dir is None:
        return {"present": False, "quiescent": True, "note": "c2 run dir not found (absent or archived)"}
    newest = max((p.stat().st_mtime for p in run_dir.rglob("*") if p.is_file()), default=0.0)
    age = time.time() - newest
    # trainer liveness: any process whose cmdline mentions the run dir (psutil-first per
    # the launcher-buffered-log lesson: never judge by log tails).
    alive = []
    try:
        import psutil  # type: ignore

        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cl = " ".join(proc.info.get("cmdline") or [])
            except Exception:
                continue
            if C2_RUN_DIR_NAME in cl:
                alive.append(proc.info["pid"])
    except ImportError:
        pass  # age-based check still applies; psutil is in the project venv
    return {
        "present": True,
        "run_dir": str(run_dir),
        "newest_file_age_s": int(age),
        "trainer_pids_alive": alive,
        "quiescent": (not alive) and age >= QUIESCENT_SECONDS,
    }


def _launch_argv(arm: str, dry_run: bool) -> list[str]:
    spec = ARMS[arm]
    argv = [
        sys.executable, str(REPO / "tools" / "launch_witness_run.py"),
        "--gt-cache", "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
        "--num-pairs", "600",
        "--config", "v9_cgauge_ideal_mod32",
        *spec["extra"],
        "--out-dir", spec["out_dir"],
        "--label", spec["label"],
        "--purpose", spec["purpose"],
    ]
    if dry_run:
        argv.append("--dry-run")
    return argv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=("control", "treatment"), required=True)
    ap.add_argument("--operator-go", action="store_true",
                    help="REQUIRED to fire. Without it this script only reports gate status + dry-run.")
    ap.add_argument("--skip-c2-gate", action="store_true",
                    help="Explicit operator override of the c2-completion gate (NOT the admission gate).")
    args = ap.parse_args()

    c2 = _c2_status()
    print(json.dumps({"stage": "c2_gate", **c2}, indent=2), flush=True)
    if not c2["quiescent"] and not args.skip_c2_gate:
        print("REFUSE: live c2 run is not complete/quiescent. The A/B is QUEUED behind it "
              "(memory: two n600 arms cannot coexist with c2 under the admission gate).",
              flush=True)
        return 3

    dry = _launch_argv(args.arm, dry_run=True)
    print(json.dumps({"stage": "dry_run", "argv": dry}), flush=True)
    rc = subprocess.run(dry, cwd=str(REPO)).returncode
    if rc != 0:
        print(f"REFUSE: governed dry-run refused rc={rc} (governor/storage/seal/DSL). "
              "That refusal is the finding; fix or wait, never override.", flush=True)
        return rc

    if not args.operator_go:
        print("PREPARED_NOT_FIRED: dry-run PASSED. Re-run with --operator-go to fire this arm "
              "(CONTAINMENT: heavy local launch requires explicit operator GO).", flush=True)
        return 0

    fire = _launch_argv(args.arm, dry_run=False)
    print(json.dumps({"stage": "fire", "argv": fire}), flush=True)
    return subprocess.run(fire, cwd=str(REPO)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
