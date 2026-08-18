#!/usr/bin/env python3
"""fire_modal_auth_eval.py — the ONE deterministic path from candidate runtime to T4 row.

Operator binding 2026-08-17: "Every mistake due to doing things by hand should be
fixed and deterministic and no longer ad hoc or manual." This tool replaces the
hand-assembled Modal auth-eval dispatch, which failed FOUR separate ways on the
rr4 fire of 2026-08-17, each failure a paid round-trip or a dead watcher:

  F1  AppleDouble litter (._*, .DS_Store) on an ExFAT runtime tree tripped the
      hidden-file validator remotely.                     -> stage SANITIZE
  F2  A payload-custody dir (retained/) inside the tree tripped the secret-name
      validator remotely.                                 -> stage VALIDATE (local, $0)
  F3  --expected-runtime-tree-sha256 was omitted by hand. -> pinned 'auto' always
  F4  A phantom-active Modal claim (call long completed) blocked the single-flight
      guard; the terminal-close needed args discovered by argparse error.
                                                          -> stage CLAIMS
  F5  The harvest poller was hand-armed with INVENTED flags and died instantly,
      leaving the paid call unwatched; the prior watcher was a background bash
      loop the harness SIGURG-kills at ~3 min.            -> stage ARM-POLLER
                                                            (detached, real argparse)

Sister incident (rr2, same day): a HAND-ASSEMBLED staged_* tree missing the
receiver's decoder guard was fired instead of the proved candidate_runtime —
fired bytes were never the proved bytes (S 27.83 vs projected 0.1585). This tool
therefore takes the runtime dir ITSELF, computes every sha from the actual bytes
(no hand-typed pins), and refuses on any mismatch with an optional external pin.

Stages (all local until DISPATCH):
  1 SANITIZE   strip macOS metadata litter (._* / .DS_Store) — filesystem
               artifacts, never receiver content. Anything else hidden refuses.
  2 VALIDATE   walk the tree with the SAME validate_runtime_upload_file the
               remote path uses, so every refusal surfaces locally in one pass.
               Secret-name hits print the exact relocation command and refuse
               (payload custody moves stay a deliberate act, never automatic).
  3 PIN        archive sha256/size computed from the file; optional
               --require-archive-sha asserts against a sealed fire-order pin.
  4 CLAIMS     reconcile; auto-terminal-close ONLY the exact provable-phantom
               condition (active Modal claim while the call-id ledger has zero
               live rows), recording the automation in the claim notes.
  5 DISPATCH   the proven `modal run --detach ...::main` invocation with a fixed
               flag template (never hand-typed), tree sha pinned 'auto'.
  6 ARM-POLLER read call_id from the spawn record (refuse-loud if absent) and
               arm tools/modal_harvest_poller.py DETACHED via
               tools/launch_detached_process.py with a done-receipt.
  7 MANIFEST   write FIRE_MANIFEST.json (stages, shas, claim actions, poller).

Contest-axis note: F26_TOKEN_DECODER-style env overrides are deliberately NOT
exposed — --inflate-env demotes the row to a diagnostic axis; decoder guards
belong IN the fired tree (the rr4 lesson).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tac.candidate_seal import (  # noqa: E402
    MISMATCH,
    PIN_ABSENT,
    check_pin_consistency,
    repin_receiver,
)
from tac.deploy.modal.auth_eval import validate_runtime_upload_file  # noqa: E402

VENV_PY = str(REPO / ".venv" / "bin" / "python")
MODAL_BIN = str(REPO / ".venv" / "bin" / "modal")
LITTER_BASENAMES = {".DS_Store"}
LITTER_PREFIXES = ("._",)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitize_litter(runtime_dir: Path) -> list[str]:
    """Delete macOS metadata litter. Returns relative paths removed."""

    removed: list[str] = []
    for path in sorted(runtime_dir.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        if name in LITTER_BASENAMES or any(name.startswith(p) for p in LITTER_PREFIXES):
            rel = str(path.relative_to(runtime_dir))
            path.unlink()
            removed.append(rel)
    return removed


def validate_tree(runtime_dir: Path) -> list[str]:
    """Run the remote upload validators locally. Returns refusal messages."""

    refusals: list[str] = []
    for path in sorted(runtime_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(runtime_dir))
        try:
            validate_runtime_upload_file(path, rel)
        except ValueError as exc:
            refusals.append(str(exc))
    return refusals


def reconcile_claims(auto_close_agent: str, dry_run: bool) -> dict:
    """Reconcile; terminal-close the provable-phantom condition only."""

    proc = subprocess.run(
        [VENV_PY, str(REPO / "tools" / "claim_lane_dispatch.py"), "reconcile"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    out = proc.stdout + proc.stderr
    action: dict = {"reconcile_output_tail": out.strip().splitlines()[-4:], "closed": []}
    # Provable phantom: "0 live ledger call_id(s)" alongside >=1 active claim row.
    if "0 live ledger call_id" not in out:
        return action
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("claim: lane="):
            continue
        fields = dict(
            part.split("=", 1) for part in line.removeprefix("claim: ").split() if "=" in part
        )
        lane, job = fields.get("lane"), fields.get("job")
        if not lane or not job:
            continue
        if dry_run:
            action["closed"].append({"lane": lane, "job": job, "dry_run": True})
            continue
        close = subprocess.run(
            [
                VENV_PY,
                str(REPO / "tools" / "claim_lane_dispatch.py"),
                "claim",
                "--lane-id", lane,
                "--platform", "modal",
                "--instance-job-id", job,
                "--agent", auto_close_agent,
                "--force",
                "--status", "stale_superseded_reconciled_no_live_call",
                "--notes",
                "Auto terminal-close by tools/fire_modal_auth_eval.py: reconcile "
                "reported an active Modal claim with ZERO live ledger call_ids "
                "(the provable-phantom condition). Deterministic per operator "
                "2026-08-17 no-manual binding.",
            ],
            capture_output=True,
            text=True,
            cwd=REPO,
        )
        action["closed"].append({"lane": lane, "job": job, "rc": close.returncode})
    return action


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runtime-dir", required=True)
    ap.add_argument("--archive", default=None, help="default: <runtime-dir>/archive.zip")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--lane-id", required=True)
    ap.add_argument("--instance-job-id", required=True)
    ap.add_argument("--claim-agent", default="MAIN")
    ap.add_argument("--single-axis-waiver-reason", default="")
    ap.add_argument("--pair-group-id", default="")
    ap.add_argument("--require-archive-sha", default="", help="sealed fire-order pin; mismatch refuses")
    ap.add_argument(
        "--repin-receiver",
        action="store_true",
        help="re-pin the staged receiver's ARCHIVE_SHA256/ARCHIVE_BYTES from the staged archive "
        "(the compose-time staging step; without it a mismatched tree refuses)",
    )
    ap.add_argument("--poller-deadline-s", type=float, default=2400.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runtime_dir = Path(args.runtime_dir).resolve()
    archive = Path(args.archive) if args.archive else runtime_dir / "archive.zip"
    # The dispatch subprocess runs with cwd=REPO, so a relative --output-dir must
    # resolve against REPO here too or the spawn-record check reads the wrong dir.
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    manifest: dict = {"schema": "fire_modal_auth_eval.v1", "runtime_dir": str(runtime_dir)}

    if not runtime_dir.is_dir():
        print(f"FATAL: runtime dir missing: {runtime_dir}")
        return 2
    if not (runtime_dir / "inflate.sh").is_file():
        print(f"FATAL: {runtime_dir}/inflate.sh missing — not a receiver runtime tree")
        return 2
    if not archive.is_file():
        print(f"FATAL: archive missing: {archive}")
        return 2

    manifest["stage1_sanitized"] = sanitize_litter(runtime_dir)
    if manifest["stage1_sanitized"]:
        print(f"SANITIZE: removed {len(manifest['stage1_sanitized'])} metadata-litter file(s)")

    refusals = validate_tree(runtime_dir)
    manifest["stage2_refusals"] = refusals
    if refusals:
        print("VALIDATE: the remote upload path WOULD refuse this tree. Fix locally first:")
        for r in refusals:
            print(f"  - {r}")
        print(
            "Payload-custody dirs (e.g. retained/) must be RELOCATED, never deleted "
            "(ALWAYS KEEP THE PAYLOAD): mv <dir> <runtime-dir>_retained_custody/"
        )
        return 3

    archive_sha = _sha256(archive)
    archive_bytes = archive.stat().st_size
    manifest["stage3_archive"] = {"path": str(archive), "sha256": archive_sha, "bytes": archive_bytes}
    print(f"PIN: archive {archive_bytes} B sha {archive_sha[:16]}…")
    if args.require_archive_sha and args.require_archive_sha.lower() != archive_sha:
        print(
            "FATAL: archive sha does not match the sealed fire-order pin "
            f"(pinned {args.require_archive_sha[:16]}…, actual {archive_sha[:16]}…)"
        )
        return 4

    # Stage 3b SEAL — archive and runtime are ONE sealed object (the r3 law). The staged
    # receiver's own pin must name the staged archive's exact bytes. Nothing here is compared
    # against a remembered candidate: both sides are measured from this tree, at this moment,
    # so it holds for rr4, fx1, sa1 and every successor without editing a literal.
    seal = check_pin_consistency(runtime_dir, archive_path=archive)
    if seal.verdict == MISMATCH and args.repin_receiver:
        repin = repin_receiver(runtime_dir, archive_path=archive)
        manifest["stage3b_repin"] = repin.to_dict()
        print(
            f"SEAL: re-pinned {seal.receiver_path.name} "
            f"{str(repin.old_sha256)[:16]}…/{repin.old_bytes:,} B -> "
            f"{repin.new_sha256[:16]}…/{repin.new_bytes:,} B"
        )
        seal = check_pin_consistency(runtime_dir, archive_path=archive)
    manifest["stage3b_seal"] = seal.to_dict()
    print(f"SEAL: {seal.summary()}")
    if seal.verdict == MISMATCH:
        print(
            "FATAL: the staged receiver pins a DIFFERENT candidate than the staged archive. "
            "This tree would be refused remotely at decode time, after the meter started. "
            "Re-run with --repin-receiver to re-pin at compose time, or stage the archive "
            "this receiver names."
        )
        return 6
    if seal.verdict == PIN_ABSENT:
        print(
            "WARNING: this receiver carries no archive pin, so the seal is weaker than the "
            "rr4 lineage's and the pin check is vacuous for this tree. Proceeding, loudly."
        )

    manifest["stage4_claims"] = reconcile_claims(args.claim_agent, args.dry_run)

    cmd = [
        MODAL_BIN, "run", "--detach", "experiments/modal_auth_eval.py::main",
        "--archive", str(archive),
        "--submission-dir", str(runtime_dir),
        "--inflate-sh", "inflate.sh",
        "--expected-archive-sha256", archive_sha,
        "--expected-runtime-tree-sha256", "auto",
        "--output-dir", str(out_dir),
        "--detach", "--provider-detach-ack",
        "--lane-id", args.lane_id,
        "--instance-job-id", args.instance_job_id,
        "--claim-agent", args.claim_agent,
    ]
    if args.single_axis_waiver_reason:
        cmd += ["--single-axis-waiver-reason", args.single_axis_waiver_reason]
    if args.pair_group_id:
        cmd += ["--pair-group-id", args.pair_group_id]
    manifest["stage5_dispatch_argv"] = cmd
    if args.dry_run:
        print("DRY-RUN: would dispatch:\n  " + " ".join(cmd))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "FIRE_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
        return 0

    disp = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    sys.stdout.write(disp.stdout[-2000:])
    sys.stderr.write(disp.stderr[-2000:])
    spawn_path = out_dir / "modal_auth_eval_spawn.json"
    if not spawn_path.is_file():
        print("FATAL: dispatch produced no spawn record — the fire DID NOT take. Do not assume a call exists.")
        return 5
    call_id = json.loads(spawn_path.read_text()).get("call_id", "")
    if not call_id:
        print("FATAL: spawn record carries no call_id — refuse-loud, investigate before refiring.")
        return 5
    manifest["stage5_call_id"] = call_id
    print(f"DISPATCHED: call {call_id}")

    receipt = f"{args.instance_job_id}_harvest"
    arm = subprocess.run(
        [
            VENV_PY, str(REPO / "tools" / "launch_detached_process.py"),
            "--output-dir", str(out_dir),
            "--purpose", f"auto-armed harvest poller for {call_id} (fire_modal_auth_eval)",
            "--done-receipt", receipt,
            "--", VENV_PY, str(REPO / "tools" / "modal_harvest_poller.py"),
            "--call-id", call_id,
            "--output-dir", str(out_dir),
            "--deadline-s", str(args.poller_deadline_s),
        ],
        cwd=REPO, capture_output=True, text=True,
    )
    try:
        arm_info = json.loads(arm.stdout)
        manifest["stage6_poller"] = {"pid": arm_info.get("pid"), "done_receipt": receipt}
        print(f"POLLER ARMED: pid {arm_info.get('pid')} receipt {receipt}")
    except (json.JSONDecodeError, ValueError):
        manifest["stage6_poller"] = {"error": arm.stdout[-400:] + arm.stderr[-400:]}
        print("WARNING: poller arm output unparsable — VERIFY THE WATCHER before ending the turn.")

    (out_dir / "FIRE_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"MANIFEST: {out_dir / 'FIRE_MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
