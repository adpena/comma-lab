# SPDX-License-Identifier: MIT
"""Canonical READ-ONLY contract for #366 DDM joint-descent CAMPAIGN run dirs.

Single source of truth for discovering the LATEST ``ddm_366_campaign_*`` run dir
and parsing its emitted artifacts into a compact dashboard snapshot. Mirrors the
role ``tac.witness_run_artifacts`` plays for witness runs (the run-artifact drift
class fix): every consumer (dashboard CAMPAIGN tab, check-ins, digests) reads THIS
module instead of re-spelling filenames/globs.

Producer (verified against live run dirs 2026-07-25, attempt-5
``ddm_366_campaign_v5_cured_20260725T062259Z``):
``tools/launch_ddm_joint_descent.py --full-run`` writes into ``--out-dir``:

  * ``telemetry/step%06d.json``            — ``ddm_joint_descent_full_run_step.v1``
      per-step BATCH-LOCAL (4-pair) initial/final {d_seg,d_pose,joint} + seconds +
      gradient_norm + proposal + pose_finish_engage_state. **ADVISORY_BATCH_LOCAL:
      never conflate with n600.**
  * ``telemetry/geometry_step*.json``      — geometry projection/cure events
      (``ddm_joint_descent_geometry_projection_event.v1``)
  * ``verdicts/*_n600.json``               — EXACT n600 realized verdicts
      (``ddm_joint_descent_chunked_stage_verdict.v1``): d_seg/d_pose, per_class,
      parameter_shadow (ema|live), realized_stage_decision, targets, references.
  * ``checkpoints/*.npz``                  — accepted-step checkpoints (count =
      accepted-step count; ckpt-every-accepted-step per the j9 cure).
  * ``run.log``                            — launcher stdout; final line set includes
      the ``ddm_joint_descent_full_run_receipt.v1`` row; also mirrored to
      ``full_run_receipt.json`` in the run dir.
  * ``run.pid`` / ``launch_manifest.json`` / ``run_identity.json`` — provenance.

READ-ONLY LAW: this module NEVER writes into a run dir.

Discovery LAW (dashboard-refresh latest-run law,
``dashboard_refresh_shows_latest_run_zero_manual_20260711``): the latest run dir is
re-derived per snapshot by freshest-signal mtime — a run-dir NAME is never a
discovery contract, so the structural marker (``launch_manifest.json`` whose argv
invokes ``launch_ddm_joint_descent.py --full-run``) also admits renamed campaign
dirs, while EXCLUDING the ws*/smoke/preflight probe dirs that share the same
run-identity schema.

Authority labels: everything here is advisory / research telemetry.
``score_claim=false`` — the n600 verdicts are ``[macOS-CPU frozen-scorer advisory]``
rows and the batch-local step metrics are ADVISORY_BATCH_LOCAL. Nothing in this
snapshot is a contest score.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from tac import process_liveness

SNAPSHOT_SCHEMA = "ddm_campaign_dashboard_snapshot.v1"
STEP_SCHEMA = "ddm_joint_descent_full_run_step.v1"
VERDICT_SCHEMA = "ddm_joint_descent_chunked_stage_verdict.v1"
RECEIPT_SCHEMA = "ddm_joint_descent_full_run_receipt.v1"

#: fast-path name family (the structural marker below is the durable contract)
CAMPAIGN_RUN_DIR_GLOB = "ddm_366_campaign_*"

#: the launcher process signature for the psutil liveness scan
LAUNCHER_SIG = "launch_ddm_joint_descent.py"

#: Sealed schedule budget: the j9-cured attempt-5 seal priced the full run at an
#: "honest 39.4h schedule" over 450 max steps (3 stages x 150) = ~312 s/step
#: (launch_manifest.purpose, ticket ddm_j9_366_geometry_escape_cure_20260725).
#: Rendered as the cadence REFERENCE line, never as a measured number.
SEALED_STEP_SECONDS = 312.0

#: BATCH-LOCAL honesty label, rendered verbatim by consumers next to step metrics.
ADVISORY_BATCH_LOCAL = ("ADVISORY_BATCH_LOCAL — per-step metrics are over the "
                        "step's own 4-pair batch, never n600")

_ENV_ROOTS = "DDM_CAMPAIGN_RESULTS_ROOTS"  # comma-separated override
_DEFAULT_ROOTS = (
    "/Volumes/VertigoDataTier/pact/experiments/results",  # canonical SSD tier
    "/Volumes/APDataStore/pact/experiments/results",      # second SSD tier
    "experiments/results",                                # local fallback
)

# per-class display order (canonical comma10k order — MEASURED, CLAUDE.md)
CLASS_ORDER = ("Road", "Lane", "Undrivable", "Movable", "MyCar")


def results_roots() -> list[Path]:
    """Candidate results roots (env override first), existing dirs only."""
    raw = os.environ.get(_ENV_ROOTS, "")
    roots = [Path(r) for r in raw.split(",") if r.strip()] if raw else [
        Path(r) for r in _DEFAULT_ROOTS]
    return [r for r in roots if r.is_dir()]


def is_campaign_run_dir(path: Path | str) -> bool:
    """True if ``path`` is a #366 CAMPAIGN (full-run) dir — name glob OR structural marker.

    Structural marker = a run PROPERTY, not a name: the launch manifest's argv
    invokes the joint-descent launcher with ``--full-run``. A bare run_identity
    schema match is deliberately NOT sufficient — the ws*/smoke/preflight probe
    dirs share that identity schema (census 2026-07-25: 39 matches under the SSD
    results root) and must never be mistaken for the campaign."""
    p = Path(path)
    if not p.is_dir():
        return False
    if p.match(CAMPAIGN_RUN_DIR_GLOB):
        return True
    manifest = _load_json(p / "launch_manifest.json")
    if not manifest:
        return False
    argv = [str(a) for a in (manifest.get("argv") or [])]
    return "--full-run" in argv and any(LAUNCHER_SIG in a for a in argv)


def _freshness_key(run_dir: Path) -> float:
    """Freshest-signal mtime for ranking (cheap: a few stats, no full listing)."""
    candidates = [run_dir, run_dir / "run.log", run_dir / "telemetry",
                  run_dir / "verdicts", run_dir / "full_run_receipt.json"]
    mtimes = []
    for c in candidates:
        try:
            mtimes.append(c.stat().st_mtime)
        except OSError:
            pass
    return max(mtimes) if mtimes else 0.0


def newest_campaign_run_dir(roots: list[Path] | None = None) -> Path | None:
    """Latest campaign run dir across ``roots`` by freshest-signal mtime."""
    dirs: list[Path] = []
    for root in (roots if roots is not None else results_roots()):
        try:
            dirs.extend(d for d in root.iterdir() if is_campaign_run_dir(d))
        except OSError:
            continue
    if not dirs:
        return None
    return max(dirs, key=_freshness_key)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text())
        return obj if isinstance(obj, dict) else None
    except (OSError, ValueError):
        return None


def _slim_step(row: dict[str, Any], mtime: float) -> dict[str, Any]:
    ini = row.get("initial") or {}
    fin = row.get("final") or {}
    engage = row.get("pose_finish_engage_state") or {}
    return {
        "global_step": row.get("global_step"),
        "stage_index": row.get("stage_index"),
        "stage_step": row.get("stage_step"),
        "stage_id": row.get("stage_id"),
        "seconds": row.get("seconds"),
        "gradient_norm": row.get("gradient_norm"),
        "d_seg_initial": ini.get("d_seg"), "d_seg_final": fin.get("d_seg"),
        "d_pose_initial": ini.get("d_pose"), "d_pose_final": fin.get("d_pose"),
        "joint_initial": ini.get("joint_objective_no_rate"),
        "joint_final": fin.get("joint_objective_no_rate"),
        "proposal_source": row.get("proposal_source"),
        "proposal_multiplier": row.get("proposal_multiplier"),
        "pair_ids": row.get("pair_ids"),
        "active_groups": row.get("active_groups"),
        "pose_objective_weight": row.get("pose_objective_weight"),
        "lr_rewarmup_factor": row.get("lr_rewarmup_factor"),
        "realized_boundary_crossed": row.get("realized_boundary_crossed"),
        "engage_classification": engage.get("classification"),
        "ts": mtime,
    }


def _slim_verdict(row: dict[str, Any], name: str, mtime: float) -> dict[str, Any]:
    engage = row.get("pose_finish_engage_state") or {}
    per_class = {}
    for cls, d in (row.get("per_class") or {}).items():
        if isinstance(d, dict):
            per_class[cls] = {"d_seg": d.get("d_seg"), "errors": d.get("errors"),
                              "sites": d.get("sites")}
    if name.startswith("stage00") or name.startswith("baseline"):
        kind = "baseline"
    elif name.startswith("warm_start"):
        kind = "warm_start_proposal"
    else:
        kind = "stage_verdict"
    return {
        "file": name,
        "kind": kind,
        "global_step": row.get("global_step"),
        "stage_step": row.get("stage_step"),
        "stage_id": row.get("stage_id"),
        "d_seg": row.get("d_seg"), "d_pose": row.get("d_pose"),
        "per_class": per_class,
        "parameter_shadow": row.get("parameter_shadow"),
        "realized_stage_decision": row.get("realized_stage_decision"),
        "reference_d_seg": row.get("reference_d_seg"),
        "reference_d_pose": row.get("reference_d_pose"),
        "target_d_seg": row.get("target_d_seg"),
        "target_d_pose": row.get("target_d_pose"),
        "archive_bytes": row.get("archive_bytes"),
        "advisory_action": row.get("advisory_action"),
        "evidence_axis": row.get("evidence_axis"),
        "num_pairs": row.get("num_pairs"),
        "elapsed_seconds": row.get("elapsed_seconds"),
        "engage": {
            "classification": engage.get("classification"),
            "engaged_global_step": engage.get("engaged_global_step"),
            "exact_d_seg": engage.get("exact_d_seg"),
            "exact_verdict_steps": engage.get("exact_verdict_steps"),
            "latest_relative_slope": engage.get("latest_relative_slope"),
            "strict_seg_admissions": engage.get("strict_seg_admissions"),
        },
        "ts": mtime,
    }


def _slim_schedule(sched: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(sched, dict):
        return None
    stages = []
    for st in sched.get("stages") or []:
        if isinstance(st, dict):
            stages.append({
                "stage_id": st.get("stage_id"),
                "target_d_seg": st.get("target_d_seg"),
                "target_d_pose": st.get("target_d_pose"),
                "maximum_steps": st.get("maximum_steps"),
                "verdict_interval_steps": st.get("verdict_interval_steps"),
                "active_groups": st.get("active_groups"),
            })
    return {
        "stages": stages,
        "train_batch": sched.get("train_batch"),
        "learning_rate": sched.get("learning_rate"),
        "checkpoint_interval_steps": sched.get("checkpoint_interval_steps"),
        "pose_finish_engage": sched.get("pose_finish_engage"),
    }


def _pid_status(run_dir: Path) -> dict[str, Any]:
    """Launcher liveness: run.pid check + psutil cmdline scan (fail-open)."""
    out: dict[str, Any] = {"pid": None, "pid_alive": False, "launcher_procs": []}
    pid_file = run_dir / "run.pid"
    try:
        out["pid"] = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        pass
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        pid = out["pid"]
        if pid:
            try:
                proc = psutil.Process(pid)
                cmd = " ".join(proc.cmdline())
                out["pid_alive"] = proc.is_running() and LAUNCHER_SIG in cmd
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                out["pid_alive"] = False
        try:
            for proc in psutil.process_iter(["pid", "cmdline"]):
                cmd = " ".join(proc.info.get("cmdline") or [])
                if LAUNCHER_SIG in cmd:
                    out["launcher_procs"].append(proc.info["pid"])
        except Exception:  # a liveness scan must never crash the snapshot
            pass
    else:  # canonical-liveness fallback: existence only (cannot verify the cmdline)
        pid = out["pid"]
        if pid:
            # tac.process_liveness instead of a bare kill(pid,0).  This brings
            # the no-psutil branch INTO line with the psutil branch above on the
            # leg they can both answer: psutil treats ZombieProcess as not
            # running, and so does this now (the bare probe called a zombie
            # alive forever).  PermissionError still differs by design -- psutil
            # returns False on AccessDenied because it could not read the
            # cmdline to match LAUNCHER_SIG, which is a different question from
            # "does the pid exist"; this branch only ever claimed existence.
            out["pid_alive"] = (
                process_liveness.pid_state(pid) == process_liveness.ALIVE
            )
    return out


class CampaignRunReader:
    """Incremental, mtime-gated reader. Cheap per poll: dir listings + stat;
    parses only NEW/CHANGED files; caches slim rows across snapshots."""

    def __init__(self, roots: list[Path] | None = None) -> None:
        self._roots = roots
        self._run_dir: Path | None = None
        self._file_sig: dict[str, tuple[float, int]] = {}
        self._steps: dict[str, dict[str, Any]] = {}
        self._verdicts: dict[str, dict[str, Any]] = {}
        self._geometry_events: dict[str, dict[str, Any]] = {}
        self._receipt_slim: dict[str, Any] | None = None
        self._ticket_schedule: dict[str, Any] | None = None

    def _reset(self) -> None:
        self._file_sig.clear()
        self._steps.clear()
        self._verdicts.clear()
        self._geometry_events.clear()
        self._receipt_slim = None
        self._ticket_schedule = None

    def _changed(self, path: Path) -> bool:
        try:
            st = path.stat()
        except OSError:
            return False
        sig = (st.st_mtime, st.st_size)
        key = str(path)
        if self._file_sig.get(key) == sig:
            return False
        self._file_sig[key] = sig
        return True

    # -- sources ---------------------------------------------------------------
    def _scan_telemetry(self, run_dir: Path) -> None:
        tdir = run_dir / "telemetry"
        if not tdir.is_dir():
            return
        for p in tdir.iterdir():
            if p.suffix != ".json" or not self._changed(p):
                continue
            row = _load_json(p)
            if row is None:
                continue
            mtime = self._file_sig[str(p)][0]
            if row.get("schema") == STEP_SCHEMA or row.get("event") == "full_run_step":
                self._steps[p.name] = _slim_step(row, mtime)
            else:  # geometry projection / cure events (any sibling event schema)
                self._geometry_events[p.name] = {
                    "file": p.name,
                    "event": row.get("event"),
                    "global_step": row.get("global_step"),
                    "status": row.get("status"),
                    "reason": row.get("reason"),
                    "track_index": row.get("track_index"),
                    "ts": mtime,
                }

    def _scan_verdicts(self, run_dir: Path) -> None:
        vdir = run_dir / "verdicts"
        if not vdir.is_dir():
            return
        for p in vdir.iterdir():
            if p.suffix != ".json" or not self._changed(p):
                continue
            row = _load_json(p)
            if row is None:
                continue
            self._verdicts[p.name] = _slim_verdict(
                row, p.name, self._file_sig[str(p)][0])

    def _scan_receipt(self, run_dir: Path) -> None:
        rp = run_dir / "full_run_receipt.json"
        if not rp.is_file() or not self._changed(rp):
            return
        row = _load_json(rp)
        if row is None:
            return
        secs = [s for s in (row.get("step_seconds") or []) if isinstance(s, (int, float))]
        self._receipt_slim = {
            "present": True,
            "schema": row.get("schema"),
            "written_at_utc": row.get("written_at_utc"),
            "verdict": row.get("verdict"),
            "campaign_blocker": row.get("campaign_blocker"),
            "latest_realized_stage_decision": row.get("latest_realized_stage_decision"),
            "pointer": row.get("pointer"),
            "pointer_moved": row.get("pointer_moved"),
            "pose_finish_engaged": row.get("pose_finish_engaged"),
            "global_step": row.get("global_step"),
            "stage_index": row.get("stage_index"),
            "stage_step": row.get("stage_step"),
            "elapsed_seconds": row.get("elapsed_seconds"),
            "telemetry_rows": row.get("telemetry_rows"),
            "hardware_axis": row.get("hardware_axis"),
            "research_only": row.get("research_only"),
            "schedule": _slim_schedule(row.get("schedule")),
            "step_seconds_n": len(secs),
            "step_seconds_median": statistics.median(secs) if secs else None,
            "step_seconds_mean": statistics.fmean(secs) if secs else None,
        }

    def _scan_ticket_schedule(self, run_dir: Path) -> None:
        """LIVE runs have no receipt yet — stage targets come from the sealed
        ticket referenced by launch_manifest argv (``--ticket <path>``)."""
        if self._ticket_schedule is not None:
            return
        manifest = _load_json(run_dir / "launch_manifest.json")
        if not manifest:
            return
        argv = manifest.get("argv") or []
        ticket_path: Path | None = None
        for i, a in enumerate(argv):
            if a == "--ticket" and i + 1 < len(argv):
                tp = Path(argv[i + 1])
                if not tp.is_absolute():
                    tp = Path(manifest.get("cwd") or ".") / tp
                ticket_path = tp
                break
        if ticket_path is None or not ticket_path.is_file():
            return
        ticket = _load_json(ticket_path)
        if not ticket:
            return
        sched = ((ticket.get("semantic_program") or {}).get("full_run_schedule"))
        self._ticket_schedule = _slim_schedule(sched)

    # -- snapshot ----------------------------------------------------------------
    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        run_dir = newest_campaign_run_dir(self._roots)
        if run_dir is None:
            return {"ok": False, "schema": SNAPSHOT_SCHEMA,
                    "reason": "no campaign run dir found",
                    "roots": [str(r) for r in (self._roots or results_roots())]}
        if self._run_dir != run_dir:
            self._reset()
            self._run_dir = run_dir

        self._scan_telemetry(run_dir)
        self._scan_verdicts(run_dir)
        self._scan_receipt(run_dir)
        self._scan_ticket_schedule(run_dir)

        steps = sorted(self._steps.values(),
                       key=lambda r: (r.get("global_step") or 0))
        verdicts = sorted(self._verdicts.values(),
                          key=lambda r: (r.get("ts") or 0.0,
                                         r.get("global_step") or 0))
        geometry = sorted(self._geometry_events.values(),
                          key=lambda r: (r.get("ts") or 0.0))

        # checkpoints: count only (stat-level; the npz bytes are never read)
        ckpt_dir = run_dir / "checkpoints"
        ckpt_names: list[str] = []
        if ckpt_dir.is_dir():
            try:
                ckpt_names = sorted(p.name for p in ckpt_dir.iterdir()
                                    if p.suffix == ".npz")
            except OSError:
                pass
        ckpt_latest_age = None
        if ckpt_names:
            try:
                ckpt_latest_age = max(
                    0.0, now - (ckpt_dir / ckpt_names[-1]).stat().st_mtime)
            except OSError:
                pass

        def _age(ts: float | None) -> float | None:
            return max(0.0, now - ts) if ts else None

        log_mtime = None
        try:
            log_mtime = (run_dir / "run.log").stat().st_mtime
        except OSError:
            pass

        last_step = steps[-1] if steps else None
        last_verdict = verdicts[-1] if verdicts else None
        secs = [s.get("seconds") for s in steps
                if isinstance(s.get("seconds"), (int, float))]

        status = _pid_status(run_dir)
        status.update({
            "receipt_present": self._receipt_slim is not None,
            "ended": self._receipt_slim is not None,
            "last_telemetry_age_s": _age(last_step["ts"]) if last_step else None,
            "last_verdict_age_s": _age(last_verdict["ts"]) if last_verdict else None,
            "run_log_age_s": _age(log_mtime),
            "global_step": (last_step or {}).get("global_step"),
            "stage_index": (last_step or {}).get("stage_index"),
            "stage_step": (last_step or {}).get("stage_step"),
            "stage_id": (last_step or {}).get("stage_id"),
        })

        schedule = ((self._receipt_slim or {}).get("schedule")
                    or self._ticket_schedule)

        return {
            "ok": True,
            "schema": SNAPSHOT_SCHEMA,
            "score_claim": False,
            "advisory_batch_local_label": ADVISORY_BATCH_LOCAL,
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "generated_at": now,
            "status": status,
            "steps": steps,
            "verdicts": verdicts,
            "geometry_events_count": len(geometry),
            "geometry_events_tail": geometry[-8:],
            "checkpoints": {"count": len(ckpt_names),
                            "latest": ckpt_names[-1] if ckpt_names else None,
                            "latest_age_s": ckpt_latest_age},
            "cadence": {
                "sealed_step_seconds": SEALED_STEP_SECONDS,
                "sealed_source": "j9 sealed 39.4h schedule / 450 max steps",
                "measured_median_s": statistics.median(secs) if secs else None,
                "measured_mean_s": statistics.fmean(secs) if secs else None,
                "measured_last_s": secs[-1] if secs else None,
                "measured_n": len(secs),
            },
            "schedule": schedule,
            "receipt": self._receipt_slim or {"present": False},
            "class_order": list(CLASS_ORDER),
        }
