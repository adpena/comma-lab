#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""One-command status reader for a B1 HiNeRV-MLX pilot run (READ-ONLY).

Prints a compact, operator-facing status for a detached B1 pilot:

  * heartbeat age + alive?  (parsed from .omx/tmp/heartbeat_b1_<run_id>.log)
  * latest epoch / stage_name / latest loss / learning rate / EMA drift
  * seconds/epoch estimate + ETA to the next checkpoint + ETA to run end
  * latest checkpoint (epoch + role) and best-by-selection-metric checkpoint
  * gate signals from the launch manifest: sidecar_exported, pay_rent_gate,
    stage8 muon status, param_count, measurement_axis
  * harvest status (if the harvester has written its status JSON)

It reads ONLY the run dir's ``telemetry.jsonl`` + heartbeat + launch manifest +
``checkpoints/`` + harvester status JSON. It imports NO training modules and
never mutates the pilot. This is the durable sister of the harvester: a human
or watchdog can ask "where is the pilot?" in one command without a Claude
session.

Usage::

    .venv/bin/python tools/read_b1_pilot_status.py \
        --run-dir /Volumes/VertigoDataTier/pact/b1_229k_pilot_20260609T055851Z

    .venv/bin/python tools/read_b1_pilot_status.py --run-dir <dir> --json
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHECKPOINT_INTERVAL = 250


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_jsonl_tail(path: Path, max_rows: int = 4000) -> list[dict[str, Any]]:
    """Read JSONL rows (best-effort; tolerant of partial trailing lines)."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except OSError:
        return rows
    if len(rows) > max_rows:
        return rows[-max_rows:]
    return rows


def heartbeat_status(heartbeat_path: Path, *, stale_seconds: float = 7 * 60) -> dict[str, Any]:
    """Return {age_seconds, alive, last_line, train_exited, last_pid}."""
    out: dict[str, Any] = {
        "heartbeat_path": str(heartbeat_path),
        "exists": heartbeat_path.is_file(),
        "age_seconds": None,
        "alive": False,
        "train_exited": False,
        "last_line": None,
        "last_pid": None,
    }
    if not heartbeat_path.is_file():
        return out
    try:
        text = heartbeat_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return out
    out["last_line"] = lines[-1]
    out["train_exited"] = any("TRAIN_EXIT" in ln for ln in lines)
    now_ts = time.time()
    last_ts: float | None = None
    for ln in reversed(lines):
        token = ln.split(" ", 1)[0]
        try:
            dt = datetime.strptime(token, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        except ValueError:
            continue
        last_ts = dt.timestamp()
        break
        # find pid token
    for ln in reversed(lines):
        for tok in ln.split():
            if tok.startswith("pid="):
                out["last_pid"] = tok[4:]
                break
        if out["last_pid"]:
            break
    if last_ts is None:
        try:
            last_ts = heartbeat_path.stat().st_mtime
        except OSError:
            return out
    age = max(0.0, now_ts - last_ts)
    out["age_seconds"] = round(age, 1)
    out["alive"] = age <= stale_seconds and not out["train_exited"]
    return out


def _find(d: Any, keys: set[str]) -> dict[str, Any]:
    found: dict[str, Any] = {}

    def walk(o: Any, prefix: str = "") -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                if k in keys and not isinstance(v, (dict, list)):
                    found.setdefault(k, v)
                if isinstance(v, dict):
                    walk(v, prefix + k + ".")

    walk(d)
    return found


def telemetry_status(telemetry_path: Path) -> dict[str, Any]:
    rows = _read_jsonl_tail(telemetry_path)
    if not rows:
        return {"telemetry_path": str(telemetry_path), "rows": 0}
    first, last = rows[0], rows[-1]
    interest = {
        "epoch",
        "global_epoch",
        "stage_name",
        "stage_id",
        "loss",
        "learning_rate",
        "ema_drift_l2",
        "muon_active",
        "sidecar_exported",
        "pay_rent_gate_active",
        "wall_clock_seconds",
    }
    last_fields = _find(last, interest)
    first_fields = _find(first, interest)

    # seconds/epoch estimate from wall_clock + epoch span.
    sec_per_epoch = None
    try:
        e0 = float(first_fields.get("epoch", first.get("epoch")))
        e1 = float(last_fields.get("epoch", last.get("epoch")))
        w0 = float(first_fields.get("wall_clock_seconds", first.get("wall_clock_seconds", 0.0)))
        w1 = float(last_fields.get("wall_clock_seconds", last.get("wall_clock_seconds", 0.0)))
        if e1 > e0 and w1 > w0:
            sec_per_epoch = (w1 - w0) / (e1 - e0)
    except (TypeError, ValueError):
        sec_per_epoch = None

    return {
        "telemetry_path": str(telemetry_path),
        "rows": len(rows),
        "latest_epoch": last_fields.get("epoch", last.get("epoch")),
        "stage_name": last_fields.get("stage_name"),
        "stage_id": last_fields.get("stage_id"),
        "latest_loss": last_fields.get("loss"),
        "learning_rate": last_fields.get("learning_rate"),
        "ema_drift_l2": last_fields.get("ema_drift_l2"),
        "muon_active": last_fields.get("muon_active"),
        "sidecar_exported_telemetry": last_fields.get("sidecar_exported"),
        "pay_rent_gate_active_telemetry": last_fields.get("pay_rent_gate_active"),
        "latest_wall_clock_seconds": last_fields.get("wall_clock_seconds"),
        "seconds_per_epoch_estimate": (round(sec_per_epoch, 2) if sec_per_epoch else None),
    }


def checkpoint_status(checkpoint_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "checkpoint_dir": str(checkpoint_dir),
        "exists": checkpoint_dir.is_dir(),
        "count": 0,
        "latest": None,
        "best": None,
        "epochs": [],
    }
    if not checkpoint_dir.is_dir():
        return out
    metas: list[dict[str, Any]] = []
    for meta_path in sorted(checkpoint_dir.glob("*.meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(meta, Mapping):
            continue
        if meta.get("schema_version") != "long_training_canonical_checkpoint.v1":
            continue
        metas.append(dict(meta))
    out["count"] = len(metas)
    out["epochs"] = sorted({int(m.get("global_epoch", -1)) for m in metas})
    if not metas:
        return out
    latest = max(metas, key=lambda m: int(m.get("global_epoch", -1)))
    out["latest"] = {
        "global_epoch": int(latest.get("global_epoch", -1)),
        "checkpoint_role": latest.get("checkpoint_role"),
        "loss": latest.get("loss"),
        "captured_at_utc": latest.get("captured_at_utc"),
    }
    # best by selection metric (mode min/max).
    def metric_val(m: Mapping[str, Any]) -> float:
        return float(m.get("checkpoint_selection_metric_value", m.get("loss", float("inf"))))

    modes = {str(m.get("checkpoint_selection_metric_mode", "min")) for m in metas}
    mode = "min" if "min" in modes or not modes else next(iter(modes))
    best = (min if mode == "min" else max)(metas, key=metric_val)
    out["best"] = {
        "global_epoch": int(best.get("global_epoch", -1)),
        "checkpoint_role": best.get("checkpoint_role"),
        "checkpoint_selection_metric_key": best.get("checkpoint_selection_metric_key"),
        "checkpoint_selection_metric_value": best.get("checkpoint_selection_metric_value"),
        "checkpoint_selection_metric_mode": mode,
    }
    return out


def manifest_status(run_dir: Path) -> dict[str, Any]:
    """Read the launch manifest's gate/observability fields, if present."""
    candidates = sorted(run_dir.glob("b1_launch_manifest_*.json"))
    if not candidates:
        return {"launch_manifest": None}
    path = candidates[0]
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"launch_manifest": str(path), "read_error": True}
    keys = [
        "param_count",
        "parity_target",
        "sidecar_export_enabled",
        "pay_rent_gate_active",
        "stage8_use_muon_flag",
        "stage8_muon_status",
        "measurement_axis",
        "checkpoint_cadence_epochs",
        "total_curriculum_epochs",
        "muon_param_count",
        "adamw_param_count",
    ]
    total = (
        d.get("total_curriculum_epochs")
        or d.get("pr95_curriculum_total_epochs")
        or d.get("research_curriculum_total_epochs")
    )
    return {
        "launch_manifest": str(path),
        **{k: d.get(k) for k in keys if k in d},
        "total_research_curriculum_epochs": total,
    }


def scaled_curriculum_status(
    total_epochs: int | None, current_epoch: Any
) -> dict[str, Any]:
    """ACTUAL scaled PR95 stage for ``current_epoch`` — the SOURCE OF TRUTH.

    Closes the manifest fidelity gap: the launch manifest prints CANONICAL stage
    boundaries (ep0..3000..29650), but the trainer proportionally SCALES the 8 stages
    to the total budget (PR95FaithfulCurriculumFactory).  This recomputes the scaled
    boundaries so 'what stage / on track for Muon' is one command, never a manual
    derivation (the false-alarm that cost a 6-step investigation).
    """

    if not total_epochs or current_epoch is None:
        return {"scaled_curriculum_available": False, "reason": "missing total_epochs or epoch"}
    try:
        from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
            PR95FaithfulCurriculumFactory,
        )

        factory = PR95FaithfulCurriculumFactory(total_epoch_budget=int(total_epochs))
        ep = int(current_epoch)
        boundaries = [list(b) for b in factory.stage_epoch_boundaries]
        cur = factory.current_stage_index(ep)
        verdict = factory.current_stage_verdict(ep)
        muon_start = next((s for (idx, s, _e) in boundaries if idx == 8), None)
        return {
            "scaled_curriculum_available": True,
            "total_epochs": int(total_epochs),
            "current_stage_index": cur,
            "current_loss_family": getattr(verdict, "loss_family", None)
            or getattr(verdict, "stage_module", None),
            "muon_active_now": cur == 8,
            "muon_starts_epoch": muon_start,
            "epochs_to_muon": (
                muon_start - ep if (muon_start is not None and ep < muon_start) else 0
            ),
            "stage_boundaries_scaled": boundaries,
        }
    except Exception as exc:
        return {"scaled_curriculum_available": False, "reason": repr(exc)}


def _research_total_from_launch_script(run_dir: Path) -> int | None:
    """Parse the ACTUAL research-curriculum total from the launch script (ground truth).

    The manifest records the CANONICAL 29650; the launch command carries the real
    reduced total (--research-curriculum-total-epochs N). Returns N, or None."""

    import re

    for name in ("launch_b1_pilot.sh", "launch.sh"):
        p = run_dir / name
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        m = re.search(r"--research-curriculum-total-epochs[=\s]+(\d+)", text)
        if m:
            return int(m.group(1))
    return None


def harvest_status(run_dir: Path) -> dict[str, Any]:
    """Read the harvester's status + result JSONs, if present."""
    out: dict[str, Any] = {"harvest_status_files": [], "harvest_result_files": []}
    for p in sorted(run_dir.glob("harvest_status_ep*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out["harvest_status_files"].append(
                {
                    "path": str(p),
                    "state": d.get("state"),
                    "target_epoch": d.get("target_epoch"),
                    "updated_utc": d.get("updated_utc"),
                }
            )
        except (OSError, json.JSONDecodeError):
            continue
    for p in sorted(run_dir.glob("hi_nerv_backend_only_ep*_exact_eval.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out["harvest_result_files"].append(
                {
                    "path": str(p),
                    "first_exact_score_advisory": d.get("first_exact_score_advisory"),
                    "evidence_grade": d.get("evidence_grade"),
                    "target_epoch": d.get("target_epoch"),
                }
            )
        except (OSError, json.JSONDecodeError):
            continue
    return out


def build_status(
    run_dir: Path,
    *,
    heartbeat_path: Path | None = None,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    stale_seconds: float = 7 * 60,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    run_id = run_dir.name
    if heartbeat_path is None:
        heartbeat_path = REPO_ROOT / ".omx" / "tmp" / f"heartbeat_b1_{run_id}.log"
    hb = heartbeat_status(heartbeat_path, stale_seconds=stale_seconds)
    tele = telemetry_status(run_dir / "telemetry.jsonl")
    ckpt = checkpoint_status(run_dir / "checkpoints")
    manifest = manifest_status(run_dir)
    harv = harvest_status(run_dir)
    # The manifest records the CANONICAL total (29650); the ACTUAL research total comes
    # from the launch command (--research-curriculum-total-epochs). Prefer the ground truth.
    actual_total = _research_total_from_launch_script(run_dir) or manifest.get(
        "total_research_curriculum_epochs"
    )
    scaled = scaled_curriculum_status(actual_total, tele.get("latest_epoch"))

    # ETA computations.
    sec_per_epoch = tele.get("seconds_per_epoch_estimate")
    latest_epoch = tele.get("latest_epoch")
    eta: dict[str, Any] = {}
    if sec_per_epoch and isinstance(latest_epoch, (int, float)):
        interval = int(manifest.get("checkpoint_cadence_epochs") or checkpoint_interval)
        next_ckpt_epoch = (int(latest_epoch) // interval + 1) * interval
        epochs_to_next = next_ckpt_epoch - int(latest_epoch)
        eta["seconds_per_epoch"] = sec_per_epoch
        eta["next_checkpoint_epoch"] = next_ckpt_epoch
        eta["eta_to_next_checkpoint_seconds"] = round(epochs_to_next * sec_per_epoch, 1)
        eta["eta_to_next_checkpoint_minutes"] = round(epochs_to_next * sec_per_epoch / 60.0, 1)

    return {
        "schema": "b1_pilot_status.v1",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "read_at_utc": _utc_now(),
        "heartbeat": hb,
        "telemetry": tele,
        "checkpoints": ckpt,
        "manifest": manifest,
        "scaled_curriculum": scaled,
        "eta": eta,
        "harvest": harv,
    }


def render_human(status: Mapping[str, Any]) -> str:
    hb = status.get("heartbeat", {})
    tele = status.get("telemetry", {})
    ckpt = status.get("checkpoints", {})
    man = status.get("manifest", {})
    eta = status.get("eta", {})
    harv = status.get("harvest", {})
    lines: list[str] = []
    lines.append(f"=== B1 pilot status: {status.get('run_id')} ===")
    alive = "ALIVE" if hb.get("alive") else ("TRAIN_EXIT" if hb.get("train_exited") else "STALE/DEAD")
    lines.append(
        f"heartbeat: {alive}  age={hb.get('age_seconds')}s  pid={hb.get('last_pid')}"
    )
    lines.append(
        f"epoch={tele.get('latest_epoch')}  stage={tele.get('stage_name')}  "
        f"loss={tele.get('latest_loss')}  lr={tele.get('learning_rate')}  "
        f"ema_drift_l2={tele.get('ema_drift_l2')}"
    )
    lines.append(
        f"sec/epoch~{tele.get('seconds_per_epoch_estimate')}  "
        f"muon_active(telemetry)={tele.get('muon_active')}  "
        f"sidecar_exported(telemetry)={tele.get('sidecar_exported_telemetry')}"
    )
    lines.append(
        f"checkpoints: count={ckpt.get('count')} epochs={ckpt.get('epochs')}"
    )
    if ckpt.get("latest"):
        latest = ckpt["latest"]
        lines.append(
            f"  latest ckpt: ep{latest.get('global_epoch')} role={latest.get('checkpoint_role')} "
            f"loss={latest.get('loss')}"
        )
    if ckpt.get("best"):
        best = ckpt["best"]
        lines.append(
            f"  best ckpt: ep{best.get('global_epoch')} "
            f"{best.get('checkpoint_selection_metric_key')}="
            f"{best.get('checkpoint_selection_metric_value')} ({best.get('checkpoint_selection_metric_mode')})"
        )
    if eta:
        lines.append(
            f"ETA: next ckpt ep{eta.get('next_checkpoint_epoch')} in "
            f"~{eta.get('eta_to_next_checkpoint_minutes')} min"
        )
    lines.append(
        f"manifest gates: param={man.get('param_count')}/{man.get('parity_target')}  "
        f"sidecar_export_enabled={man.get('sidecar_export_enabled')}  "
        f"pay_rent_gate={man.get('pay_rent_gate_active')}  "
        f"stage8_muon={man.get('stage8_muon_status')}  axis={man.get('measurement_axis')}"
    )
    sc = status.get("scaled_curriculum", {})
    if sc.get("scaled_curriculum_available"):
        if sc.get("muon_active_now"):
            muon_phase = "MUON ACTIVE"
        else:
            muon_phase = (
                f"pre-muon; muon@ep{sc.get('muon_starts_epoch')} "
                f"(in {sc.get('epochs_to_muon')}ep)"
            )
        lines.append(
            f"SCALED curriculum (source-of-truth, not manifest canonical): "
            f"stage {sc.get('current_stage_index')}/8 ({sc.get('current_loss_family')})  "
            f"{muon_phase}"
        )
    res_files = harv.get("harvest_result_files") or []
    stat_files = harv.get("harvest_status_files") or []
    if res_files:
        for r in res_files:
            lines.append(
                f"HARVEST RESULT: ep{r.get('target_epoch')} "
                f"first_exact_score_advisory={r.get('first_exact_score_advisory')} "
                f"{r.get('evidence_grade')}"
            )
    elif stat_files:
        for s in stat_files:
            lines.append(
                f"harvest: ep{s.get('target_epoch')} state={s.get('state')} "
                f"({s.get('updated_utc')})"
            )
    else:
        lines.append("harvest: no harvester status/result yet")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--heartbeat-path", default=None)
    ap.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    ap.add_argument("--stale-seconds", type=float, default=7 * 60)
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    status = build_status(
        Path(args.run_dir).expanduser(),
        heartbeat_path=(Path(args.heartbeat_path) if args.heartbeat_path else None),
        checkpoint_interval=args.checkpoint_interval,
        stale_seconds=args.stale_seconds,
    )
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print(render_human(status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
