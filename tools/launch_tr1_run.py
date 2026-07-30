"""Governed launcher for the tr1 partition-renderer trainer (the tb1-owed D1 item).

The canonical witness launcher (tools/launch_witness_run.py) hardcodes its
_TRAINER to the levelset entry point; tb1's T1/T2 windows therefore ran detached
with sealed tickets but WITHOUT a governor. This launcher closes that gap for
tr1 with the governor duties that bind (a REFUSE is information, not an obstacle):

  G1 SEAL FRESHNESS (never fire a stale seal): recompile the ticket's argv from
     the CURRENT DSL factories + trainer argparse and REFUSE on any drift between
     the sealed argv and the recompiled argv (out-dir/resume-from excepted only
     when --allow-outdir-drift, for resumable window restarts).
  G2 IMPORT CUSTODY (shared-venv hijack guard): resolve tac + the trainer from
     the launch cwd tree; REFUSE if tac.__file__ escapes it.
  G3 MEMORY PREFLIGHT at the real config: free-RAM floor vs the MEASURED tb1 T2
     peak-RSS receipt (12.8 GiB at n600/D16/c4/w24 chunk32) x safety factor; the
     projection is anchor-based (never a guess) and REFUSES below floor.
  G4 SCORER SLOT: REFUSE if another tr1/levelset/witness trainer is running
     (ONE n600 job at a time).
  G5 DETACHED + RECEIPTED: launches via tools/launch_detached_process.py and
     writes launch_receipt.json (ticket hash, git head, argv, pid) in the run dir.

score_claim=false; the launcher grants NO authority — downstream artifacts decide.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MEASURED_T2_PEAK_RSS_GIB = 12.8  # tb1 memo T2: /usr/bin/time -l at the real config
SAFETY_FACTOR = 2.0
SLOT_PATTERNS = ("train_tr1_partition_renderer", "train_levelset_witness",
                 "train_witness_realized")


def refuse(msg: str) -> int:
    print(f"REFUSE: {msg}", file=sys.stderr)
    return 4


def venv_custody_gate0(cwd: Path) -> str | None:
    """G0 SHARED-VENV HIJACK GUARD (sg1 §10 owed guard): before ANY dispatch, prove the
    shared .venv is not editable-installed to a codex_worktrees / linked-worktree ``src``
    (the eg1 hijack class — 2nd recurrence). Prefers the canonical tools/check_venv_src_custody.py
    (MAIN is landing it); falls back to an inline scan of the venv's ``.pth``/``__editable__``
    entries for out-of-tree targets. Returns a REFUSE message or None (clean)."""
    checker = cwd / "tools" / "check_venv_src_custody.py"
    if checker.is_file():
        r = subprocess.run([str(cwd / ".venv" / "bin" / "python"), str(checker)],
                           cwd=cwd, capture_output=True, text=True)
        if r.returncode != 0:
            return f"venv custody gate0 (tools/check_venv_src_custody.py rc={r.returncode}): " \
                   f"{(r.stderr or r.stdout).strip()[:400]}"
        return None
    # inline fallback: scan the venv site-packages for editable *.pth targets pointing OUT of cwd.
    bad: list[str] = []
    for pth in (cwd / ".venv").glob("lib/python*/site-packages/*.pth"):
        try:
            for ln in pth.read_text(errors="ignore").splitlines():
                ln = ln.strip()
                if not ln or ln.startswith("import "):
                    continue
                if "codex_worktrees" in ln or (
                        "/src" in ln and str(cwd) not in ln and Path(ln).is_absolute()):
                    bad.append(f"{pth.name}: {ln}")
        except Exception:
            continue
    # editable finder modules (uv/setuptools) can also embed an out-of-tree path.
    for ed in (cwd / ".venv").glob("lib/python*/site-packages/__editable__*.py"):
        txt = ed.read_text(errors="ignore")
        if "codex_worktrees" in txt:
            bad.append(f"{ed.name}: codex_worktrees target")
    if bad:
        return "venv custody gate0 (inline .pth scan) — shared-venv HIJACK to out-of-tree src:\n" \
               + "\n".join(bad[:5])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticket", required=True, type=Path,
                    help="sealed ticket JSON (schema ddm_tb1_tr1_sealed_ticket.v1)")
    ap.add_argument("--cwd", type=Path, default=Path.cwd(),
                    help="repo tree to launch from (the T3 long burn fires from MAIN)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--resume-from", type=Path, default=None)
    ap.add_argument("--allow-outdir-drift", action="store_true",
                    help="permit sealed out-dir/resume-from to differ (window restarts)")
    ap.add_argument("--purpose", default="tr1 governed window")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cwd = args.cwd.resolve()

    # G0 — shared-venv hijack guard (runs BEFORE any import from cwd/src below).
    g0 = venv_custody_gate0(cwd)
    if g0 is not None:
        return refuse(g0)

    ticket = json.loads(args.ticket.read_text())
    if ticket.get("schema") != "ddm_tb1_tr1_sealed_ticket.v1":
        return refuse(f"unknown ticket schema {ticket.get('schema')!r}")

    # G1 — seal freshness: recompiled hash must match the sealed levers/argv.
    sys.path.insert(0, str(cwd / "src"))
    from tac.witness_dsl.spec_tr1_renderer_20260728 import (  # noqa: E402
        TR1RendererProgramV1, trainer_declared_flags)
    from tac.witness_dsl.curriculum_dsl import Lever  # noqa: E402
    levers = tuple(Lever(name=d["name"], overrides=dict(d["overrides"]),
                         notes=d.get("notes", "")) for d in ticket["levers"])
    sealed_argv = list(ticket["argv"])

    def _get(flag: str, argv: list[str]) -> str | None:
        return argv[argv.index(flag) + 1] if flag in argv else None

    prog = TR1RendererProgramV1(
        levers=levers,
        num_pairs=int(_get("--num-pairs", sealed_argv) or 600),
        out_dir=str(args.out_dir),
        seed=int(_get("--seed", sealed_argv) or 0),
        gt_cache=_get("--gt-cache", sealed_argv),
        resume_from=str(args.resume_from) if args.resume_from else None,
        full_confirm="--full-confirm" in sealed_argv)
    recompiled = prog.compile_trainer_argv()  # fail-closed never-invent-flags

    def _norm(argv: list[str]) -> list[str]:
        drop_next = False
        out = []
        for i, a in enumerate(argv):
            if drop_next:
                drop_next = False
                continue
            if a in ("--out-dir", "--resume-from") and args.allow_outdir_drift:
                drop_next = True
                continue
            out.append(a)
        return out

    if _norm(recompiled) != _norm(sealed_argv):
        return refuse("STALE SEAL: recompiled argv drifts from sealed argv\n"
                      f"  sealed:     {_norm(sealed_argv)}\n"
                      f"  recompiled: {_norm(recompiled)}")
    declared = trainer_declared_flags(cwd / ticket["trainer"])
    emitted = {a for a in sealed_argv if a.startswith("--")}
    if not emitted <= declared:
        return refuse(f"sealed argv emits undeclared flags: {sorted(emitted - declared)}")

    # G2 — import custody.
    import tac  # noqa: E402
    tac_file = Path(tac.__file__).resolve()
    if cwd not in tac_file.parents:
        return refuse(f"tac import custody escape: {tac_file} not under {cwd}")

    # G3 — memory preflight (anchor-based).
    page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 16384
    try:
        vmout = subprocess.run(["vm_stat"], capture_output=True, text=True,
                               check=True).stdout
        free_pages = 0
        for ln in vmout.splitlines():
            if ln.startswith(("Pages free", "Pages inactive", "Pages purgeable")):
                free_pages += int(ln.split(":")[1].strip().rstrip("."))
        page = 16384 if "page size of 16384" in vmout else page
        free_gib = free_pages * page / 2**30
    except Exception as exc:
        return refuse(f"memory preflight failed to measure free RAM: {exc}")
    need = MEASURED_T2_PEAK_RSS_GIB * SAFETY_FACTOR
    if free_gib < need:
        return refuse(f"memory preflight: free+reclaimable {free_gib:.1f} GiB < "
                      f"{need:.1f} GiB (= measured T2 peak {MEASURED_T2_PEAK_RSS_GIB} "
                      f"GiB x {SAFETY_FACTOR})")

    # G4 — scorer slot.
    ps = subprocess.run(["ps", "-axo", "pid,command"], capture_output=True,
                        text=True).stdout
    holders = [ln for ln in ps.splitlines()
               if any(p in ln for p in SLOT_PATTERNS) and "launch_tr1_run" not in ln]
    if holders:
        return refuse("scorer slot busy (ONE n600 job at a time):\n" +
                      "\n".join(holders[:3]))

    # G5 — detached launch + receipt.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "-C", str(cwd), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(cwd), "status", "--short"],
                           capture_output=True, text=True).stdout.strip()
    launch_argv = list(sealed_argv)
    launch_argv[launch_argv.index("--out-dir") + 1] = str(args.out_dir)
    if args.resume_from is not None:
        if "--resume-from" in launch_argv:
            launch_argv[launch_argv.index("--resume-from") + 1] = str(args.resume_from)
        else:
            launch_argv += ["--resume-from", str(args.resume_from)]
    receipt = {
        "schema": "ddm_lv1_tr1_governed_launch_receipt.v1",
        "ticket_path": str(args.ticket), "ticket_hash": ticket.get("ticket_hash"),
        "sealed_sha256": ticket.get("sealed_sha256"),
        "git_head": head, "git_dirty_files": dirty.splitlines(),
        "cwd": str(cwd), "argv": launch_argv,
        "gates": {"venv_custody_gate0": "PASS", "seal_freshness": "PASS",
                  "import_custody": str(tac_file),
                  "memory_free_gib": round(free_gib, 1),
                  "memory_floor_gib": round(need, 1), "scorer_slot": "FREE"},
        "score_claim": False, "launched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                            time.gmtime()),
    }
    if args.dry_run:
        receipt["dry_run"] = True
        (args.out_dir / "launch_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps(receipt["gates"], indent=2))
        print("DRY-RUN OK (all gates pass; not launched)")
        return 0
    env_py = str(cwd / ".venv" / "bin" / "python")
    cmd = [env_py, "tools/launch_detached_process.py",
           "--output-dir", str(args.out_dir), "--cwd", str(cwd),
           "--purpose", args.purpose,
           "--authority", "advisory macOS-CPU/MLX; score_claim=false",
           "--env", f"PYTHONPATH={cwd}/src:{cwd}:{cwd}/upstream",
           "--env", "TAC_MLX_CUSTOM_GROUPED_BACKWARD=1",
           "--", env_py] + launch_argv
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print(out.stdout.strip())
    if out.returncode != 0:
        return refuse(f"detached launcher failed rc={out.returncode}: {out.stderr}")
    try:
        receipt["detached"] = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        receipt["detached_raw"] = out.stdout[-500:]
    (args.out_dir / "launch_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("GOVERNED LAUNCH OK; receipt:", args.out_dir / "launch_receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
