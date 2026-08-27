# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""Schema-driven RUN INTROSPECTION for the live dashboard LIVE tab (task #352).

Operator directive 2026-07-08: the LIVE tab must render the run's ACTUAL schedule,
curriculum, controller, and telemetry through the run's OWN artifacts — never the
legacy epoch-scripted ``tau/l7`` lens (the exact PR95-era assumption the operator
just banned from configs). This module is the ONE run-introspection layer the tab
renders WHATEVER it exposes: as the DSL evolves (event-triggered curriculum,
per-class-λ costate homotopy, τ warm-restart cycles) new stage/event kinds are
ADDITIVE here and the renderer needs no rewrite.

Every field sources a REAL artifact the run emits (NO-FAKE — a panel is absent, never
fabricated, when its artifact is absent):

  * ``launch.sh``               -> the run's flag-validated trainer argv (curves + schedule)
  * ``tac.witness_dsl.schedule_readback.read_schedule`` -> event/fixed stage map (the DSL SoT)
  * ``constants_manifest.json`` (#351)  -> LawRef-resolved constants + value-provenance ladder
  * ``costate_shadow.jsonl`` (#247)     -> costate controller SENSE state (λ traces, duty queue)
  * ``run.log`` ``{"stage":"verdict"}`` -> confound-immune LIVENESS row (accepted-frac / stepped / skip)
  * ``run.log`` ``{"stage":"mem_probe"}`` (#329) -> rss / mlx memory telemetry
  * ``run.log`` transition/switch rows   -> fired curriculum EVENT markers

Classification (DESIGN LAW #1): each schedule element is one of
  EVENT-TRIGGERED  (sensor named + live arm state: pending/fired, watching cap)
  DERIVED          (a LawRef ``equation_id`` + resolved value from the constants manifest)
  FIXED/CAP        (an epoch literal, tagged).

Dependency-light BY CONTRACT: stdlib + ``tac.witness_dsl.schedule_readback`` +
``tac.witness_run_artifacts`` only (no numpy/mlx/torch; both are pure-stdlib themselves)
— imported every dashboard refresh tick. Curve math (τ/β/LR) is a
FAITHFUL pure-python port of the trainer's own ``_softmax_temp_for_epoch`` /
``_hosc_beta_for_epoch`` / ``_lr_scheduled_for_epoch`` (BIT-shape identical), so a
"planned" curve is a real derivation, not a decorative sketch.

Incremental: the growing ``run.log`` is read via a BOUNDED tail (last ~256 KB), never a
full re-parse per tick. Every entry point is FAIL-OPEN (returns ``None``; never raises)
because the consumer is a load-bearing multi-day daemon.

Authority: OBSERVABILITY (score-neutral, read-only). The frontier pointer (contest-CPU
0.19110) is UNMOVED by anything here.
"""
from __future__ import annotations

import datetime
import json
import math
from pathlib import Path
from typing import Any

# The DSL schedule read-back is the single source of truth for the stage map + the
# run's typed trainer args. Fail-open: a broken/absent tac install must never kill the
# daemon — introspect_run degrades every DSL-derived facet to None.
try:
    from tac.witness_dsl.schedule_readback import (
        build_real_trainer_parser,
        read_schedule,
        trainer_argv_from_launch_sh,
    )
except Exception:  # pragma: no cover - import guard
    build_real_trainer_parser = None  # type: ignore
    read_schedule = None  # type: ignore
    trainer_argv_from_launch_sh = None  # type: ignore

# Canonical witness run-artifact CONTRACT (single source of truth for run filenames).
# Fail-open with a literal fallback so a broken tac install never kills the daemon.
try:
    from tac import witness_run_artifacts as _wra
except Exception:  # pragma: no cover - import guard
    class _wra:  # type: ignore[no-redef]
        COSTATE_JSONL = "costate_shadow.jsonl"

__all__ = [
    "introspect_run",
    "trainer_args_from_run",
    "planned_curves",
    "read_constants_manifest",
    "read_controller",
    "read_liveness_row",
    "read_mem_probes",
    "read_events",
]

# value-provenance ladder (CLAUDE.md "VALUE-PROVENANCE LADDER"): highest trust first.
# A ladder_class maps to a display TIER + human label so the constants table can rank
# and colour rows by how the value was justified (derived-live > ... > hardcoded).
_LADDER_TIER = {
    "derived_live": (0, "derived (live)"),
    "derived_at_config": (1, "derived (at config)"),
    "measured_anchor": (2, "measured anchor"),
    "hardcoded_with_waiver": (3, "hardcoded (waiver)"),
    "hardcoded": (4, "hardcoded"),
}
# run.log stage rows that are genuine curriculum/optimizer TRANSITION events FIRED during
# training — distinct from the one-time setup-lever config rows (those live in the config
# panel). This distinction IS the story: an event that FIRED vs a scripted epoch cap.
_EVENT_STAGES = {
    "curriculum_transition_fired": "curriculum transition fired",
    "curriculum_transition": "curriculum transition",
    "muon_finisher_switch": "Muon finisher engaged",
    "stage_transition_reset_moments": "optimizer moments reset",
    "rollback": "rollback to best",
}
_TAIL_BYTES = 262_144  # bounded run.log tail (last ~256 KB) — incremental, O(1) in file size


def _utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _age_s(path: Path) -> float | None:
    try:
        import time

        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return None


def _tail_text(path: Path, max_bytes: int = _TAIL_BYTES) -> str:
    """The last ``max_bytes`` of a (possibly huge, growing) log — bounded read from the
    end so a multi-day file never costs O(size) per tick. Returns '' on any error."""
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
                fh.readline()  # drop the partial first line after the seek
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _iter_stage_rows(text: str):
    """Yield parsed ``{"stage": ...}`` JSON objects from log text (skips non-JSON)."""
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{") or '"stage"' not in line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if isinstance(d, dict) and "stage" in d:
            yield d


def _run_log(run_dir: Path) -> Path | None:
    """The run's primary log: ``run.log`` preferred, else the newest ``*.log``."""
    cand = run_dir / "run.log"
    if cand.is_file():
        return cand
    logs = sorted(run_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


# ─────────────────────────── trainer args (typed, faithful) ───────────────────────────
def trainer_args_from_run(run_dir: Path) -> Any | None:
    """The run's typed trainer namespace, parsed from ``launch.sh`` through the trainer's
    REAL argparse (never-invent-flags). None on any failure (missing launch.sh / bad tac
    install / a SystemExit from a value the real parser rejects)."""
    if trainer_argv_from_launch_sh is None or build_real_trainer_parser is None:
        return None
    launch = run_dir / "launch.sh"
    if not launch.is_file():
        return None
    try:
        argv = trainer_argv_from_launch_sh(launch.read_text(errors="replace"))
        if not argv:
            return None
        ns, _unknown = build_real_trainer_parser().parse_known_args(argv)
        return ns
    except (SystemExit, Exception):  # noqa: BLE001 — fail-open by contract
        return None


# ─────────────────────────── planned curves (faithful ports) ───────────────────────────
def _tau_at(ep: int, a: Any) -> float:
    """Faithful pure-python port of the trainer's ``_softmax_temp_for_epoch``."""
    ae = int(getattr(a, "anneal_epochs", None) or getattr(a, "epochs", 1))
    prog = (ep - 1) / max(ae - 1, 1)
    shape = str(getattr(a, "tau_anneal_shape", "cosine"))
    start = float(a.softmax_temp_start)
    end = float(a.softmax_temp_end)
    if shape == "geometric" and start > 0 and end > 0:
        return float(start * (end / start) ** prog)
    if shape == "cosine_hold":
        hf = float(getattr(a, "tau_hold_frac", 1.0))
        if hf < 1.0:
            if prog >= hf:
                return end
            prog = prog / hf
    return float(end + 0.5 * (start - end) * (1 + math.cos(math.pi * prog)))


def _beta_at(ep: int, a: Any) -> float:
    """Faithful port of the trainer's ``_hosc_beta_for_epoch``."""
    ae = int(getattr(a, "anneal_epochs", None) or getattr(a, "epochs", 1))
    prog = (ep - 1) / max(ae - 1, 1)
    b0, b1 = float(a.hosc_beta), float(a.hosc_beta_end)
    if str(getattr(a, "hosc_beta_anneal", "linear")) == "cosine":
        return float(b1 + 0.5 * (b0 - b1) * (1 + math.cos(math.pi * prog)))
    return float(b0 + (b1 - b0) * prog)


def _lr_at(ep: int, a: Any) -> float:
    """Faithful port of the trainer's ``_lr_scheduled_for_epoch`` (base AdamW schedule)."""
    warm = int(getattr(a, "warmup_epochs", 1))
    if ep <= warm:
        return float(a.lr) * ep / max(warm, 1)
    lae = int(getattr(a, "lr_anneal_epochs", None) or getattr(a, "anneal_epochs", None)
              or getattr(a, "epochs", 1))
    prog = (ep - warm) / max(lae - warm, 1)
    hf = float(getattr(a, "lr_hold_frac", 1.0))
    if hf < 1.0:
        if prog >= hf:
            return float(a.lr_end)
        prog = prog / hf
    return float(a.lr_end + 0.5 * (a.lr - a.lr_end) * (1 + math.cos(math.pi * prog)))


def planned_curves(run_dir: Path, n: int = 60) -> dict | None:
    """PLANNED τ / β / base-LR curves, sampled from the run's typed flags via the faithful
    trainer ports above. Each curve carries ``points`` [[epoch,value]...], endpoints, the
    shape name, and any Muon-freeze / hold boundary so the renderer can annotate. None when
    launch.sh is unreadable. The τ/β finisher-freeze (the Muon stage freezes the schedule at
    its muon-start value) is reflected by HOLDING those two curves past ``muon_start``."""
    a = trainer_args_from_run(run_dir)
    if a is None:
        return None
    epochs = int(getattr(a, "epochs", 0) or 0)
    if epochs <= 1:
        return None
    muon = getattr(a, "muon_start_epoch", None)
    muon = int(muon) if isinstance(muon, (int, float)) and muon and muon < epochs else None
    xs = sorted({1, *(1 + round(i * (epochs - 1) / (n - 1)) for i in range(n)), epochs})

    def _curve(fn, freeze_at_muon: bool) -> list[list[float]]:
        pts = []
        for ep in xs:
            e = ep
            if freeze_at_muon and muon is not None and ep > muon:
                e = muon  # finisher freezes the schedule at the muon-start value
            pts.append([ep, round(fn(e, a), 6)])
        return pts

    tau_hold = None
    if str(getattr(a, "tau_anneal_shape", "")) == "cosine_hold":
        hf = float(getattr(a, "tau_hold_frac", 1.0))
        ae = int(getattr(a, "anneal_epochs", None) or epochs)
        if hf < 1.0:
            tau_hold = 1 + round(hf * (ae - 1))  # epoch τ reaches the floor + holds
    curves = {
        "tau": {
            "name": "softmax τ", "shape": str(getattr(a, "tau_anneal_shape", "cosine")),
            "start": float(a.softmax_temp_start), "end": float(a.softmax_temp_end),
            "denom": int(getattr(a, "anneal_epochs", None) or epochs),
            "hold_epoch": tau_hold, "muon_freeze": muon, "unit": "τ",
            "points": _curve(_tau_at, True),
            "note": "planned · Muon freezes τ at the muon-start value" if muon else "planned",
        },
        "beta": {
            "name": "hosc β", "shape": str(getattr(a, "hosc_beta_anneal", "linear")),
            "start": float(a.hosc_beta), "end": float(a.hosc_beta_end),
            "denom": int(getattr(a, "anneal_epochs", None) or epochs),
            "muon_freeze": muon, "unit": "β",
            "points": _curve(_beta_at, True),
            "note": "planned · Muon freezes β at the muon-start value" if muon else "planned",
        },
        "lr": {
            "name": "base LR", "shape": "cosine",
            "start": float(a.lr), "end": float(a.lr_end),
            "denom": int(getattr(a, "lr_anneal_epochs", None) or getattr(a, "anneal_epochs", None) or epochs),
            "warmup": int(getattr(a, "warmup_epochs", 1)), "muon_freeze": muon, "unit": "lr",
            "points": _curve(_lr_at, False),
            "note": ("planned · AdamW base schedule; Muon params use --muon-lr after ep%d" % muon)
            if muon else "planned",
        },
    }
    return {"epochs": epochs, "muon_start": muon, "curves": curves}


# ─────────────────────────── constants manifest (#351) ───────────────────────────
def read_constants_manifest(run_dir: Path) -> dict | None:
    """The LawRef-compiled ``constants_manifest.json`` (#351) as a ranked display table:
    each constant carries value, value-provenance ladder tier+label, equation_id, the
    resolving anchor's sha256+source, and the human provenance note. None when absent."""
    path = run_dir / "constants_manifest.json"
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(errors="replace"))
    except Exception:
        return None
    consts = raw.get("constants")
    if not isinstance(consts, dict):
        return None
    rows = []
    for name, c in consts.items():
        if not isinstance(c, dict):
            continue
        lc = str(c.get("ladder_class") or "hardcoded")
        tier, label = _LADDER_TIER.get(lc, (4, lc))
        inputs = c.get("inputs") or []
        anchor_sha = anchor_src = prov = None
        if isinstance(inputs, list):
            for inp in inputs:
                if not isinstance(inp, dict):
                    continue
                if inp.get("sha256") and anchor_sha is None:
                    anchor_sha = inp.get("sha256")
                    anchor_src = inp.get("source")
                if prov is None and inp.get("provenance"):
                    prov = inp.get("provenance")
        rows.append({
            "name": name, "value": c.get("value"), "ladder_class": lc,
            "ladder_tier": tier, "ladder_label": label,
            "equation_id": c.get("equation_id"), "fallback_used": bool(c.get("fallback_used")),
            "anchor_sha": (anchor_sha[:12] if isinstance(anchor_sha, str) else None),
            "anchor_source": anchor_src, "provenance": prov,
            "warnings": c.get("warnings") or [],
        })
    rows.sort(key=lambda r: (r["ladder_tier"], str(r["name"])))
    return {
        "schema": raw.get("schema"), "config_family": raw.get("config_family"),
        "generated_at": raw.get("generated_at"), "count": len(rows), "rows": rows,
    }


# ─────────────────────────── controller / costate (#247) ───────────────────────────
def _last_jsonl_row(path: Path) -> dict | None:
    """The last valid JSON object of a (possibly multi-MB) JSONL — bounded tail read."""
    text = _tail_text(path)
    last = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                last = obj
    return last


def read_controller(run_dir: Path) -> dict | None:
    """The costate controller SENSE state from ``costate_shadow.jsonl`` last row (#247): the
    per-λ costate traces (value/band/status/method/units), classification, top recommendation,
    duty-to-measure queue counts, probe queue, per-axis EV producer signals, pointer, ts.
    READ-ONLY / advisory (CONTAINMENT — the dashboard never actuates). None when absent."""
    path = run_dir / _wra.COSTATE_JSONL
    if not path.is_file():
        return None
    row = _last_jsonl_row(path)
    if not row:
        return None
    try:
        costates = []
        for c in row.get("costates") or []:
            if not isinstance(c, dict):
                continue
            costates.append({
                "name": c.get("name"), "value": c.get("value"), "band": c.get("band"),
                "status": c.get("status"), "units": c.get("units"), "method": c.get("method"),
            })
        recs = row.get("recommendations") or []
        rec = None
        if recs and isinstance(recs[0], dict):
            r0 = recs[0]
            rec = {"action": r0.get("action"), "predicted_dS": r0.get("predicted_dS"),
                   "horizon_epochs": r0.get("horizon_epochs")}
        duty = row.get("duty_to_measure")
        duty_owed = len(duty) if isinstance(duty, list) else None
        duty_nf = (sum(1 for d in duty if isinstance(d, dict) and d.get("state") == "never-fired")
                   if isinstance(duty, list) else None)
        duty_ranked = []
        for d in (row.get("duty_ranked") or [])[:6]:
            if isinstance(d, dict):
                duty_ranked.append({"lever": d.get("candidate_lever"),
                                    "cost_epochs": d.get("measurement_cost_epochs")})
        producer = None
        for p in row.get("producer_signals") or []:
            if isinstance(p, dict) and p.get("producer") == "sensitivity_map.axis_weights":
                sig = p.get("signal")
                if isinstance(sig, dict):
                    producer = sig
                break
        probe_q = row.get("probe_queue")
        cls = row.get("classification")
        factor = row.get("factorized_adjoint")
        factor_summary = None
        if isinstance(factor, dict):
            fac = factor.get("factorization") or {}
            exact = fac.get("exact") or {}
            derived = fac.get("derived") or {}
            learned = factor.get("learned_residual") or {}
            decision = factor.get("decision") or {}
            factor_summary = {
                "architecture": factor.get("architecture"),
                "admission": factor.get("admission"),
                "head_rank": exact.get("head_rank"),
                "zero_weight_camera_frac": exact.get("certified_zero_weight_camera_frac"),
                "road_lane_lambda_ratio": derived.get(
                    "road_lane_gain_only_lambda_ratio_vs_other_median"),
                "learned_parameters": learned.get("n_parameters"),
                "amplitude_gate": learned.get("amplitude_gate"),
                "predicted_dS": decision.get("predicted_dS"),
                "predicted_dS_band": decision.get("predicted_dS_band"),
                "why": decision.get("why"),
                "confidence": ((factor.get("recommendation_candidate") or {}).get(
                    "confidence") or factor.get("validation_scope")),
            }
        event_rows = row.get("event_advisories")
        return {
            "ok": True, "ts": row.get("ts"), "epoch": row.get("epoch"),
            "age_s": _age_s(path), "actuation": row.get("actuation"),
            "classification": (str(cls).upper() if cls else None),
            "n_verdicts": (row.get("state") or {}).get("n_verdicts"),
            "pointer": row.get("pointer"), "axis": row.get("axis"),
            "costates": costates, "rec": rec,
            "duty_owed": duty_owed, "duty_never_fired": duty_nf, "duty_ranked": duty_ranked,
            "probe_queue": (len(probe_q) if isinstance(probe_q, list) else None),
            "axis_ev": producer,
            "factorized_adjoint": factor_summary,
            "event_advisories": (event_rows if isinstance(event_rows, list) else []),
        }
    except Exception:
        return None


# ─────────────────────────── liveness row (confound-immune) ───────────────────────────
def read_liveness_row(run_dir: Path) -> dict | None:
    """The confound-immune LIVENESS signals from the LAST ``{"stage":"verdict"}`` row: the
    accepted-batch fraction, weights_stepped, skip counts, frozen_epoch, ep_loss (CLAUDE.md
    "Confound self-protection" — a frozen run must LOOK frozen). Alarms flag a spike-deadlock
    (low accepted-frac) or a frozen epoch. None until the first verdict (early warming state)."""
    log = _run_log(run_dir)
    if log is None:
        return None
    last = None
    for d in _iter_stage_rows(_tail_text(log)):
        if d.get("stage") == "verdict" and "epoch" in d:
            last = d
    if last is None:
        return None
    acc = last.get("accepted_frac")
    frozen_ep = last.get("frozen_epoch")  # truthy => the spike-guard froze this epoch's weights
    ep_loss = last.get("ep_loss")
    alarms = []
    try:
        if frozen_ep:  # bool True or a non-zero epoch — a bare False means NOT frozen
            alarms.append("frozen_epoch")
        if ep_loss is not None and float(ep_loss) == 0.0:
            alarms.append("ep_loss_zero")
        if acc is not None and float(acc) < 0.5:
            alarms.append("low_accepted_frac")
    except (TypeError, ValueError):
        pass
    return {
        "epoch": last.get("epoch"), "accepted_frac": acc,
        "accepted_batches": last.get("accepted_batches"),
        "skipped_batches": last.get("skipped_batches"),
        "weights_stepped": last.get("weights_stepped"),
        "frozen_epoch": frozen_ep, "ep_loss": ep_loss,
        "d_seg": last.get("d_seg"), "d_pose": last.get("d_pose"),
        "seg_form": last.get("seg_form"), "ts": last.get("ts"),
        "alarms": alarms,
    }


# ─────────────────────────── mem_probe telemetry (#329) ───────────────────────────
def read_mem_probes(run_dir: Path) -> dict | None:
    """The ``{"stage":"mem_probe"}`` RSS / MLX telemetry rows (#329) from the bounded tail:
    [{phase, rss_gib, mlx_active_gib, mlx_cache_gib?}]. None when the run emits none."""
    log = _run_log(run_dir)
    if log is None:
        return None
    rows = []
    for d in _iter_stage_rows(_tail_text(log)):
        if d.get("stage") != "mem_probe":
            continue
        rows.append({
            "phase": d.get("phase"), "rss_gib": d.get("rss_gib"),
            "mlx_active_gib": d.get("mlx_active_gib"), "mlx_cache_gib": d.get("mlx_cache_gib"),
        })
    if not rows:
        return None
    peak = max((r["rss_gib"] for r in rows if isinstance(r.get("rss_gib"), (int, float))),
               default=None)
    # time-ordered rss/mlx series for a sparkline (a growing run emits many rows -> cap the
    # rendered window to the most recent, keep the tail-wide peak); ``latest`` is the newest row.
    total = len(rows)
    recent = rows[-64:]
    series = [[i, r["rss_gib"]] for i, r in enumerate(recent)
             if isinstance(r.get("rss_gib"), (int, float))]
    mlx_series = [[i, r["mlx_active_gib"]] for i, r in enumerate(recent)
                 if isinstance(r.get("mlx_active_gib"), (int, float))]
    return {"rows": recent, "series": series, "mlx_series": mlx_series,
            "peak_rss_gib": peak, "count": total, "latest": rows[-1]}


# ─────────────────────────── mod-dim dynamics telemetry (2026-07-08) ───────────────────────────
def read_mod_dim_dynamics(run_dir: Path) -> dict | None:
    """The ``{"stage":"mod_dim_dynamics"}`` latent-table telemetry (score-neutral) from the bounded
    tail: the LATEST spectrum summary (effective_rank / k90 / k99 / spectral entropy / tau / seg_form
    + k90 truncate-bytes estimate) plus a small effective-rank-vs-epoch series for a sparkline (does the
    rank track the anneal octaves? does it saturate before dash birth?). Presence-gated (None when the
    run emits none => additive over pre-2026-07-08 run dirs); NEVER raises. Error rows are surfaced too
    so a fail-open telemetry hiccup is visible rather than silently dropped."""
    log = _run_log(run_dir)
    if log is None:
        return None
    rows = []
    for d in _iter_stage_rows(_tail_text(log)):
        if d.get("stage") != "mod_dim_dynamics":
            continue
        if "error" in d:
            rows.append({"epoch": d.get("epoch"), "seg_form": d.get("seg_form"),
                         "error": d.get("error")})
            continue
        spec = d.get("spectrum") or {}
        rows.append({
            "epoch": d.get("epoch"), "seg_form": d.get("seg_form"), "tau": d.get("tau"),
            "mod_dim": d.get("mod_dim"),
            "effective_rank": spec.get("effective_rank"), "k90": spec.get("k90"),
            "k99": spec.get("k99"), "spectral_entropy_norm": spec.get("spectral_entropy_norm"),
            "k90_truncate_bytes_estimate": d.get("k90_truncate_bytes_estimate"),
            "code_bytes_full": d.get("code_bytes_full"),
            "latent_xi_cca_max": (d.get("latent_xi_cca") or {}).get("max"),
        })
    if not rows:
        return None
    recent = rows[-64:]
    eff_series = [[i, r["effective_rank"]] for i, r in enumerate(recent)
                  if isinstance(r.get("effective_rank"), (int, float))]
    k90_series = [[i, r["k90"]] for i, r in enumerate(recent)
                  if isinstance(r.get("k90"), (int, float))]
    return {"rows": recent, "effective_rank_series": eff_series, "k90_series": k90_series,
            "count": len(rows), "latest": rows[-1]}


# ─────────────────────────── fired curriculum events ───────────────────────────
def read_events(run_dir: Path) -> list[dict] | None:
    """Fired curriculum / optimizer TRANSITION events from the bounded tail — each a distinct
    marker {stage, label, epoch}. Distinct from epoch-cap boundaries (that distinction IS the
    story: an event fired vs a scripted boundary). None when none present."""
    log = _run_log(run_dir)
    if log is None:
        return None
    events = []
    for d in _iter_stage_rows(_tail_text(log)):
        s = d.get("stage")
        if s in _EVENT_STAGES:
            events.append({"stage": s, "label": _EVENT_STAGES[s], "epoch": d.get("epoch")})
    return events or None


# ─────────────────────────── schedule (classified) ───────────────────────────
def _schedule_payload(run_dir: Path, log_paths, const_names: set[str]) -> dict | None:
    """The DSL-derived stage map (event/fixed), each element tagged with a display CLASS
    (event / fixed) + live status. None when the DSL read-back is unavailable."""
    if read_schedule is None:
        return None
    try:
        rb = read_schedule(run_dir, log_paths=log_paths)
    except Exception:
        return None
    if rb is None or not getattr(rb, "ok", False):
        reason = getattr(rb, "reason", "read-back unavailable") if rb else "read-back unavailable"
        return {"ok": False, "reason": reason}
    stages = []
    for st in rb.stages:
        d = st.to_dict()
        d["klass"] = "event" if d.get("mode") == "event" else "fixed"
        stages.append(d)
    return {
        "ok": True, "source": rb.source, "epochs": rb.epochs, "eval_every": rb.eval_every,
        "event_triggered": rb.event_triggered, "stages": stages,
        "n_derived_constants": len(const_names),
    }


# ─────────────────────────── top-level ───────────────────────────
def introspect_run(run_dir, log_paths=None) -> dict:
    """The full schema-driven introspection payload for one run dir. Every sub-key is None
    when its source artifact is absent (graceful degradation over pre-v6 run dirs). NEVER
    raises. ``ok`` is False only when ``run_dir`` does not resolve to a directory."""
    out: dict = {
        "ok": False, "run_dir": (str(run_dir) if run_dir else None),
        "generated_at": _utc(), "schedule": None, "constants": None, "controller": None,
        "liveness": None, "mem": None, "events": None, "curves": None,
        "mod_dim_dynamics": None,
    }
    if not run_dir:
        return out
    rd = Path(run_dir)
    if not rd.is_dir():
        return out
    out["ok"] = True
    lp = log_paths if log_paths else None
    try:
        out["constants"] = read_constants_manifest(rd)
    except Exception:
        out["constants"] = None
    const_names = set()
    if out["constants"]:
        const_names = {r["name"] for r in out["constants"].get("rows", [])}
    for key, fn in (
        ("schedule", lambda: _schedule_payload(rd, lp, const_names)),
        ("controller", lambda: read_controller(rd)),
        ("liveness", lambda: read_liveness_row(rd)),
        ("mem", lambda: read_mem_probes(rd)),
        ("events", lambda: read_events(rd)),
        ("curves", lambda: planned_curves(rd)),
        ("mod_dim_dynamics", lambda: read_mod_dim_dynamics(rd)),
    ):
        try:
            out[key] = fn()
        except Exception:
            out[key] = None
    return out


if __name__ == "__main__":  # tiny manual smoke: python tools/witness_run_introspect.py <run_dir>
    import sys

    rd = sys.argv[1] if len(sys.argv) > 1 else "."
    print(json.dumps(introspect_run(rd, log_paths=[str(Path(rd) / "run.log")]), indent=2)[:4000])
