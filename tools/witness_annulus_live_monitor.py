#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""LIVE, NON-INVASIVE annulus-convergence MONITOR for a running level-set witness run.

Gives the operator live telemetry on the codim-1 boundary ANNULUS plus a human-readable
"what is happening and WHY" narration -- WITHOUT touching, signalling, or editing the live
training process. READ-ONLY on the run's checkpoints + stdout log. numpy-fp32 authority;
every row is tagged ``[macOS-numpy advisory . NON-PROMOTABLE]``; the frontier pointer
(0.19110) is UNMOVED (this is SENSE-layer observability for task #333, NOT a score).

WHAT IT DOES per tick (or one ``--once`` smoke):
  1. Parse the run's stdout log (``verdict`` + ``loss_terms`` rows) -- FREE, no render. This
     is the "what is happening" mechanics: d_seg/d_pose trajectory, ep_loss, accepted_frac,
     weights_stepped, spike_skipped, gnorm, hosc_beta anneal, softmax_temp, seg_form stage.
  2. Render the newest preserved checkpoint(s) through the REAL render-through-R + REAL
     frozen CPU-torch SegNet path by REUSING ``tools/witness_annulus_convergence.py`` (which
     itself reuses ``witness_per_stage_annulus_attribution``). We snapshot each checkpoint to
     an epoch-stamped, immutable copy first (no torn reads of the live-updated BEST) and
     subprocess the renderer at a small ``--pairs`` (default 16, MEMORY-SAFE, released on
     child exit). Cached maps accumulate across ticks -> live dV/dEpoch.
  3. Append one machine-readable JSONL row to ``<run>/annulus_live.jsonl``.
  4. Emit a WHY NARRATION block (deep-math interpretation, not just numbers) to
     ``<run>/annulus_live_narration.txt`` (overwrite-latest) AND print it.

NARRATION RULES (the deep-math "why"):
  * boundary-jitter vs structural-miss (annulus_flip_mass_share high AND interior_flip small
    => boundary jitter on the separatrix, partition topology correct; interior rising =>
    STRUCTURAL region miss, investigate).
  * per-class stuck boundary (name the class with the highest per-class annulus flip-frac;
    canonical comma10k order 0=Road 1=Lane 2=Undrivable 3=Movable 4=MyCar).
  * convergence direction (sign of dV/dEpoch on annulus_flip_frac: <0 tightening, ~0 flicker
    floor, >0 WIDENING regression).
  * stage transition (seg_form change => ep_loss scale change is the loss-FORM change, not a
    real d_seg move; watch the next verdict).
  * training health (gnorm range, spike_skipped rate, accepted_frac, hosc_beta anneal,
    softmax_temp; frozen_epoch / ep_loss==0 / accepted_frac==0 = spike-guard-deadlock).
  * margin convergence (rising annulus witness-margin p10/p50 = real convergence).

The narration classifiers are pure module-level functions (unit-tested without rendering).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.witness_annulus_metrics import ADVISORY, CLASS_NAMES  # noqa: E402

# ---------------------------------------------------------------------------
# NARRATION THRESHOLDS (documented; overridable in the pure functions for tests).
# ---------------------------------------------------------------------------
MASS_SHARE_HI = 0.9            # annulus_flip_mass_share above this => boundary-dominated.
INTERIOR_FLIP_LO = 1e-3        # interior_flip_frac below this => no structural miss.
CONVERGENCE_DEADBAND = 1e-6    # |dV/dEpoch| below this => plateaued (flip-frac scale ~1e-5/ep).
SPIKE_SKIP_CEILING = 0.30      # recent spike_skipped fraction above this => spike storm.
HOSC_BETA_START = 1.0
HOSC_BETA_END = 4.0


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


# ===========================================================================
# LOG PARSING (pure; free -- no render).
# ===========================================================================
def parse_log_rows(text: str) -> tuple[list[dict], list[dict]]:
    """Extract ``verdict`` and ``loss_terms`` JSON rows from a run's stdout log text.

    Each relevant line is a standalone JSON object carrying a ``stage`` field. Non-JSON /
    other-stage lines are ignored. Returns ``(verdicts, loss_terms)`` in file order.
    """
    verdicts: list[dict] = []
    loss_terms: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        i = line.find("{")
        if i < 0:
            continue
        frag = line[i:]
        try:
            obj = json.loads(frag)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        stage = obj.get("stage")
        if stage == "verdict":
            verdicts.append(obj)
        elif stage == "loss_terms":
            loss_terms.append(obj)
    return verdicts, loss_terms


def d_seg_trajectory(verdicts: list[dict]) -> list[dict]:
    """Compact per-verdict trajectory rows (the "what is happening" mechanics)."""
    out: list[dict] = []
    for v in verdicts:
        out.append({
            "epoch": v.get("epoch"),
            "seg_form": v.get("seg_form"),
            "d_seg": v.get("d_seg"),
            "d_pose": v.get("d_pose"),
            "ep_loss": v.get("ep_loss"),
            "accepted_frac": v.get("accepted_frac"),
            "weights_stepped": v.get("weights_stepped"),
            "spike_skipped": v.get("skipped_batches"),
            "frozen_epoch": v.get("frozen_epoch"),
        })
    return out


# ===========================================================================
# NARRATION CLASSIFIERS (pure; unit-tested without rendering).
# ===========================================================================
def classify_residual(
    annulus_flip_mass_share: float,
    interior_flip_frac: float,
    mass_hi: float = MASS_SHARE_HI,
    interior_lo: float = INTERIOR_FLIP_LO,
) -> tuple[str, str]:
    """boundary-jitter vs structural-miss classification of the d_seg residual.

    Returns ``(label, text)`` where label in {"boundary_jitter","structural_miss",
    "mixed"}.
    """
    if interior_flip_frac is None:
        interior_flip_frac = 0.0
    if annulus_flip_mass_share is None:
        annulus_flip_mass_share = 0.0
    if interior_flip_frac >= interior_lo:
        return (
            "structural_miss",
            f"STRUCTURAL miss appearing in class interior (interior_flip_frac="
            f"{interior_flip_frac:.2e} >= {interior_lo:.0e}) -- a region is being mislabeled "
            f"wholesale, not boundary jitter; investigate which class/region.",
        )
    if annulus_flip_mass_share >= mass_hi:
        return (
            "boundary_jitter",
            f"residual is BOUNDARY JITTER on the codim-1 separatrix "
            f"(annulus holds {100 * annulus_flip_mass_share:.1f}% of flips, interior_flip_frac="
            f"{interior_flip_frac:.2e}) -- the partition TOPOLOGY is correct; the witness is "
            f"placing the boundary ~1px off.",
        )
    return (
        "mixed",
        f"residual is MIXED: annulus holds {100 * annulus_flip_mass_share:.1f}% of flips "
        f"(< {100 * mass_hi:.0f}%) with interior_flip_frac={interior_flip_frac:.2e} "
        f"(< {interior_lo:.0e}) -- mostly boundary, some flip mass leaking off the annulus.",
    )


def dominant_stuck_class(per_class: dict) -> tuple[int, str, float]:
    """Return (class_idx, class_name, flip_frac) for the highest per-class annulus flip-frac.

    ``per_class`` maps class idx (int or str after a JSON round-trip) -> flip frac. Empty
    input -> (-1, "none", 0.0).
    """
    if not per_class:
        return (-1, "none", 0.0)
    best_idx, best_val = -1, -1.0
    for k, v in per_class.items():
        idx = int(k)
        val = float(v)
        if val > best_val:
            best_idx, best_val = idx, val
    if best_idx < 0:
        return (-1, "none", 0.0)
    return (best_idx, CLASS_NAMES.get(best_idx, f"cls{best_idx}"), best_val)


def convergence_direction(rate: float, deadband: float = CONVERGENCE_DEADBAND) -> tuple[str, str]:
    """Sign of dV/dEpoch on annulus flip-frac -> convergence-direction label + text."""
    if rate is None:
        return ("unknown", "convergence direction UNKNOWN (need >=2 checkpoints for dV/dEpoch).")
    if rate < -abs(deadband):
        return (
            "tightening",
            f"annulus STILL TIGHTENING (dV/dEpoch={rate:+.2e} < 0): the viscous level-set flow "
            f"is shrinking the boundary band.",
        )
    if rate > abs(deadband):
        return (
            "widening",
            f"annulus WIDENING (dV/dEpoch={rate:+.2e} > 0): REGRESSION -- the boundary band is "
            f"growing; flag it.",
        )
    return (
        "plateau",
        f"annulus flip-frac PLATEAUED at the flicker floor (dV/dEpoch={rate:+.2e} ~ 0).",
    )


def detect_stage_transition(verdicts: list[dict]) -> dict | None:
    """Detect a seg_form change between the two most-recent verdicts.

    Returns ``{"from","to","epoch"}`` on a transition, else None.
    """
    forms = [(v.get("epoch"), v.get("seg_form")) for v in verdicts if v.get("seg_form") is not None]
    if len(forms) < 2:
        return None
    (_ep_prev, f_prev), (ep_cur, f_cur) = forms[-2], forms[-1]
    if f_prev != f_cur:
        return {"from": f_prev, "to": f_cur, "epoch": ep_cur}
    return None


def training_health(
    loss_terms_recent: list[dict],
    latest_verdict: dict | None,
    spike_ceiling: float = SPIKE_SKIP_CEILING,
) -> tuple[dict, str, bool]:
    """Summarize training-health mechanics from recent loss_terms + the latest verdict.

    Spike-guard-deadlock signature (memory
    ``spike_guard_median_freeze_deadlock_ep_loss_zero_signature``) is read from the VERDICT
    row (frozen_epoch / ep_loss==0 / accepted_frac==0) -- NOT from the per-accum-batch
    loss_terms ``accepted_frac`` (which is a running fraction, legitimately 0.0 at
    accum_batch 0). Returns ``(health_dict, narration, is_deadlock)``.
    """
    gnorms = [float(r["gnorm"]) for r in loss_terms_recent if r.get("gnorm") is not None]
    spike_flags = [bool(r.get("spike_skipped")) for r in loss_terms_recent if "spike_skipped" in r]
    betas = [float(r["hosc_beta"]) for r in loss_terms_recent if r.get("hosc_beta") is not None]
    temps = [float(r["softmax_temp"]) for r in loss_terms_recent if r.get("softmax_temp") is not None]

    spike_rate = (sum(spike_flags) / len(spike_flags)) if spike_flags else 0.0
    beta_cur = betas[-1] if betas else None
    beta_pct = (
        max(0.0, min(1.0, (beta_cur - HOSC_BETA_START) / (HOSC_BETA_END - HOSC_BETA_START)))
        if beta_cur is not None else None
    )
    temp_cur = temps[-1] if temps else None

    # Deadlock signature from the VERDICT row.
    frozen = bool(latest_verdict.get("frozen_epoch")) if latest_verdict else False
    ep_loss = latest_verdict.get("ep_loss") if latest_verdict else None
    v_accepted = latest_verdict.get("accepted_frac") if latest_verdict else None
    ep_loss_zero = (ep_loss is not None and float(ep_loss) == 0.0)
    accepted_zero = (v_accepted is not None and float(v_accepted) == 0.0)
    spike_storm = spike_rate > spike_ceiling
    is_deadlock = bool(frozen or ep_loss_zero or accepted_zero or spike_storm)

    health = {
        "gnorm_min": (min(gnorms) if gnorms else None),
        "gnorm_max": (max(gnorms) if gnorms else None),
        "spike_skipped_rate": spike_rate,
        "hosc_beta": beta_cur,
        "hosc_beta_anneal_pct": beta_pct,
        "softmax_temp": temp_cur,
        "verdict_accepted_frac": v_accepted,
        "verdict_frozen_epoch": frozen,
        "verdict_ep_loss": ep_loss,
        "is_deadlock": is_deadlock,
    }

    if is_deadlock:
        why = []
        if frozen:
            why.append("frozen_epoch=true")
        if ep_loss_zero:
            why.append("ep_loss==0")
        if accepted_zero:
            why.append("verdict accepted_frac==0")
        if spike_storm:
            why.append(f"spike_skipped_rate={spike_rate:.2f} > {spike_ceiling:.2f}")
        text = (
            "TRAINING-HEALTH ALERT -- SPIKE-GUARD DEADLOCK signature (" + ", ".join(why) + "): "
            "the median-freeze reference window cannot re-arm after a level shift; verdicts "
            "off a frozen run are NON-load-bearing. VERIFY training is actually stepping."
        )
    else:
        parts = []
        if gnorms:
            parts.append(f"gnorm in [{min(gnorms):.2f}, {max(gnorms):.2f}]")
        if beta_cur is not None:
            parts.append(f"hosc_beta={beta_cur:.3f} ({100 * beta_pct:.0f}% annealed {HOSC_BETA_START}->{HOSC_BETA_END})")
        if temp_cur is not None:
            parts.append(f"softmax_temp={temp_cur:.4f}")
        if v_accepted is not None:
            parts.append(f"verdict accepted_frac={float(v_accepted):.2f}")
        parts.append(f"spike_skipped_rate={spike_rate:.2f}")
        text = "training HEALTHY: " + ", ".join(parts) + "."
    return health, text, is_deadlock


def margin_convergence_text(margin_p10_curve: list, margin_p50_curve: list) -> str:
    """Interpret the annulus witness-margin p10/p50 trend across checkpoints."""
    def _finite(xs):
        return [float(x) for x in xs if x is not None and float(x) == float(x)]  # drop NaN
    p10 = _finite(margin_p10_curve or [])
    p50 = _finite(margin_p50_curve or [])
    if len(p50) < 2:
        return (f"annulus witness-margin p10={p10[-1]:.3f} p50={p50[-1]:.3f} (single checkpoint; "
                f"need >=2 for a trend)." if (p10 and p50) else "annulus witness-margin: insufficient data.")
    d10 = p10[-1] - p10[0] if len(p10) >= 2 else 0.0
    d50 = p50[-1] - p50[0]
    if d50 > 0 and d10 >= 0:
        return (f"annulus witness-margin RISING (p10 {p10[0]:.3f}->{p10[-1]:.3f}, "
                f"p50 {p50[0]:.3f}->{p50[-1]:.3f}): the witness is pushing its OWN margin away "
                f"from the flip threshold -- real convergence, not cosmetic argmax luck.")
    if d50 < 0:
        return (f"annulus witness-margin FALLING (p50 {p50[0]:.3f}->{p50[-1]:.3f}): the witness "
                f"margin is eroding toward the flip threshold -- watch for rising flips.")
    return (f"annulus witness-margin ~flat (p50 {p50[0]:.3f}->{p50[-1]:.3f}).")


def build_narration(
    run_label: str,
    epoch,
    seg_form: str | None,
    latest_metrics: dict | None,
    rates: dict | None,
    verdicts: list[dict],
    health_text: str,
    margin_text: str,
    stage_transition: dict | None,
    verdict_pairs: int,
    advisory_subset: bool,
) -> str:
    """Assemble the full human-readable WHY narration block."""
    lines: list[str] = []
    lines.append(f"=== WITNESS ANNULUS LIVE NARRATION  {ADVISORY} ===")
    lines.append(f"run: {run_label}  |  latest epoch: {epoch}  |  seg_form: {seg_form}")
    lines.append(f"annulus rendered on {verdict_pairs} verdict pairs"
                 + (" (ADVISORY STRIDED SUBSET)" if advisory_subset else " (full)"))
    lines.append("")

    if latest_metrics is not None:
        thr = latest_metrics.get("threshold", {})
        d_seg = latest_metrics.get("overall_d_seg")
        mass = thr.get("annulus_flip_mass_share")
        interior = thr.get("interior_flip_frac")
        annf = thr.get("annulus_flip_frac")
        lines.append(f"[d_seg] overall={d_seg:.6f}  annulus_flip_frac={annf:.5f}  "
                     f"interior_flip_frac={interior:.2e}  annulus_mass_share={100 * mass:.1f}%")
        r_label, r_text = classify_residual(mass, interior)
        lines.append(f"  WHY (residual character): {r_text}")
        idx, name, val = dominant_stuck_class(thr.get("per_class_annulus_flip_frac", {}))
        lines.append(f"  WHY (stuck boundary): {name}(cls{idx}) is the dominant stuck boundary "
                     f"(annulus flip-frac {val:.4f})"
                     + (" -- the known thin-dash long-tail." if idx == 1 else "."))
    else:
        lines.append("[d_seg] no annulus render available this tick (checkpoint pending / skipped).")

    if rates is not None:
        c_label, c_text = convergence_direction(rates.get("annulus_flip_frac_rate"))
        lines.append(f"  WHY (convergence): {c_text}")
    lines.append(f"  WHY (margin): {margin_text}")
    lines.append("")

    if stage_transition is not None:
        lines.append(
            f"[STAGE TRANSITION] {stage_transition['from']} -> {stage_transition['to']} at "
            f"ep{stage_transition['epoch']}: the ep_loss scale change is the loss-FORM change, "
            f"NOT a real d_seg move; watch the NEXT verdict for the real effect.")
        lines.append("")

    lines.append(f"[training health] {health_text}")

    # last few verdict d_seg values for context.
    tail = [v for v in verdicts if v.get("d_seg") is not None][-4:]
    if tail:
        traj = "  ".join(f"ep{v.get('epoch')}={float(v['d_seg']):.6f}" for v in tail)
        lines.append(f"[recent d_seg verdicts] {traj}")

    lines.append("")
    lines.append(f"pointer 0.19110 UNMOVED. advisory telemetry only -- no score claim. ({_utc()})")
    return "\n".join(lines)


# ===========================================================================
# CHECKPOINT SNAPSHOT + RENDER (subprocess; memory-isolated).
# ===========================================================================
def _epoch_of(path: Path, best_json: dict | None) -> int:
    """Cheap epoch guess for snapshot naming (true epoch comes from cfg in the renderer)."""
    import re
    if best_json and path.name == best_json.get("path") and best_json.get("epoch") is not None:
        return int(best_json["epoch"])
    m = re.search(r"ep(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def snapshot_checkpoints(run_dir: Path, snap_dir: Path, log) -> list[tuple[str, Path]]:
    """Copy the newest stage ckpt(s) + BEST to epoch-stamped immutable snapshots (no torn reads).

    Returns ``[(NAME, snapshot_path), ...]`` for the annulus renderer's explicit --ckpt set.
    """
    snap_dir.mkdir(parents=True, exist_ok=True)
    best_json = None
    bj = run_dir / "levelset_best.json"
    if bj.exists():
        try:
            best_json = json.loads(bj.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            best_json = None

    srcs: list[Path] = []
    for pat in ("levelset_ckpt_stage*.npz", "levelset_ema_stage*.npz", "levelset_witness_ema_BEST.npz"):
        srcs.extend(sorted(run_dir.glob(pat)))
    # de-dup preserving order.
    seen: dict[Path, None] = {}
    for p in srcs:
        seen.setdefault(p.resolve(), None)

    out: list[tuple[str, Path]] = []
    for p in seen:
        ep = _epoch_of(p, best_json)
        if "BEST" in p.name:
            label = f"BEST_ep{ep}"
        else:
            import re
            m = re.search(r"stage([A-Za-z0-9]+)", p.stem)
            label = f"{m.group(1)}_ep{ep}" if m else f"{p.stem}_ep{ep}"
        dst = snap_dir / f"{label}.npz"
        if not dst.exists():
            tmp = snap_dir / f".{label}.tmp.npz"
            try:
                shutil.copy2(p, tmp)
                os.replace(tmp, dst)
                log(f"[snapshot] {p.name} -> {dst.name}")
            except OSError as e:
                log(f"[snapshot] skip {p.name}: {e}")
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
                continue
        out.append((label, dst))
    return out


def run_annulus_render(
    snapshots: list[tuple[str, Path]],
    gt_cache: Path,
    num_pairs: int,
    pairs: int,
    maps_dir: Path,
    threads: int,
    log,
) -> dict | None:
    """Subprocess ``tools/witness_annulus_convergence.py`` on the snapshots (memory-isolated).

    Returns the parsed ``annulus_convergence.json`` summary dict, or None on failure. The
    convergence tool caches per-name maps in ``maps_dir`` so unchanged epochs are NOT
    re-rendered -> the series accumulates across ticks (live dV/dEpoch).
    """
    if not snapshots:
        log("[render] no snapshots to render.")
        return None
    maps_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(REPO / "tools/witness_annulus_convergence.py"),
        "--gt-cache", str(gt_cache),
        "--num-pairs", str(num_pairs),
        "--pairs", str(pairs),
        "--threads", str(threads),
        "--out-dir", str(maps_dir),
    ]
    for name, path in snapshots:
        cmd += ["--ckpt", f"{name}={path}"]
    log(f"[render] subprocess: pairs={pairs} ckpts={[n for n, _ in snapshots]}")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        log("[render] TIMEOUT (>3600s); skipping this tick's render.")
        return None
    if proc.returncode != 0:
        log(f"[render] renderer rc={proc.returncode}; stderr tail:\n{proc.stderr[-1200:]}")
        return None
    summary_path = maps_dir / "annulus_convergence.json"
    if not summary_path.exists():
        log("[render] renderer produced no summary json.")
        return None
    log(f"[render] done in {time.time() - t0:.1f}s -> {summary_path.name}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


# ===========================================================================
# ORCHESTRATION.
# ===========================================================================
def discover_log(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.exists() else None
    cands = sorted((REPO / ".omx/tmp").glob("levelset_mod32cap_*.log"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def latest_verdict(verdicts: list[dict]) -> dict | None:
    return verdicts[-1] if verdicts else None


def one_tick(args, log) -> dict:
    """Run one monitor pass: parse log, render annulus, emit JSONL row + narration."""
    run_dir = args.run_dir
    log_path = discover_log(args.log)
    verdicts: list[dict] = []
    loss_terms: list[dict] = []
    if log_path and log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        verdicts, loss_terms = parse_log_rows(text)
        log(f"[log] {log_path.name}: {len(verdicts)} verdict rows, {len(loss_terms)} loss_terms rows")
    else:
        log("[log] no log found; mechanics narration will be empty.")

    lv = latest_verdict(verdicts)
    recent_lt = loss_terms[-args.loss_terms_window:] if loss_terms else []
    health, health_text, is_deadlock = training_health(recent_lt, lv)
    stage_transition = detect_stage_transition(verdicts)

    # render the annulus (memory-isolated subprocess).
    snap_dir = run_dir / "annulus_live_snapshots"
    maps_dir = run_dir / "annulus_live_maps"
    snapshots = snapshot_checkpoints(run_dir, snap_dir, log)
    summary = run_annulus_render(
        snapshots, args.gt_cache, args.num_pairs, args.pairs, maps_dir, args.threads, log)

    latest_metrics = None
    rates = None
    margin_p10 = margin_p50 = []
    verdict_pairs = 0
    advisory_subset = False
    if summary is not None:
        ckpts = summary.get("checkpoints", [])
        if ckpts:
            latest_metrics = max(ckpts, key=lambda c: float(c["epoch"]))["metrics"]
        rates = summary.get("convergence", {}).get("threshold")
        if rates:
            margin_p10 = rates.get("margin_p10_curve", [])
            margin_p50 = rates.get("margin_p50_curve", [])
        verdict_pairs = int(summary.get("verdict_pairs", 0))
        advisory_subset = bool(summary.get("advisory_subset", False))

    margin_text = margin_convergence_text(margin_p10, margin_p50)

    epoch = lv.get("epoch") if lv else (
        max((float(c["epoch"]) for c in summary["checkpoints"]), default=None) if summary else None)
    seg_form = lv.get("seg_form") if lv else None

    narration = build_narration(
        run_label=run_dir.name, epoch=epoch, seg_form=seg_form,
        latest_metrics=latest_metrics, rates=rates, verdicts=verdicts,
        health_text=health_text, margin_text=margin_text,
        stage_transition=stage_transition, verdict_pairs=verdict_pairs,
        advisory_subset=advisory_subset)

    # emit narration (overwrite latest) + print.
    narr_path = run_dir / "annulus_live_narration.txt"
    narr_path.write_text(narration + "\n", encoding="utf-8")
    print("\n" + narration + "\n", flush=True)

    # build + append the JSONL row.
    thr = (latest_metrics or {}).get("threshold", {}) if latest_metrics else {}
    row = {
        "advisory": ADVISORY,
        "ts": _utc(),
        "run_label": run_dir.name,
        "epoch": epoch,
        "seg_form": seg_form,
        "verdict_pairs": verdict_pairs,
        "advisory_subset": advisory_subset,
        "annulus": {
            "overall_d_seg": (latest_metrics or {}).get("overall_d_seg") if latest_metrics else None,
            "annulus_flip_frac": thr.get("annulus_flip_frac"),
            "interior_flip_frac": thr.get("interior_flip_frac"),
            "annulus_flip_mass_share": thr.get("annulus_flip_mass_share"),
            "annulus_area_frac": thr.get("annulus_area_frac"),
            "per_class_annulus_flip_frac": thr.get("per_class_annulus_flip_frac"),
            "margin_p10": (thr.get("annulus_margin", {}) or {}).get("p10"),
            "margin_p50": (thr.get("annulus_margin", {}) or {}).get("p50"),
            "gibbs_ring_proxy": thr.get("gibbs_ring_proxy"),
        },
        "convergence_rates": ({
            "annulus_flip_frac_rate": rates.get("annulus_flip_frac_rate"),
            "interior_flip_frac_rate": rates.get("interior_flip_frac_rate"),
            "overall_d_seg_rate": rates.get("overall_d_seg_rate"),
            "per_class_annulus_flip_frac_rate": rates.get("per_class_annulus_flip_frac_rate"),
            "n_checkpoints": len(rates.get("epochs", [])),
        } if rates else None),
        "mechanics": {
            "latest_verdict": (d_seg_trajectory([lv])[0] if lv else None),
            "health": health,
            "stage_transition": stage_transition,
        },
        "narration_classes": {
            "residual": (classify_residual(thr.get("annulus_flip_mass_share"),
                                           thr.get("interior_flip_frac"))[0] if latest_metrics else None),
            "convergence": (convergence_direction(rates.get("annulus_flip_frac_rate"))[0] if rates else None),
            "dominant_stuck_class": (dominant_stuck_class(thr.get("per_class_annulus_flip_frac", {}))[1]
                                     if latest_metrics else None),
            "is_deadlock": is_deadlock,
        },
    }
    jsonl_path = run_dir / "annulus_live.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=float) + "\n")
    log(f"[emit] appended row to {jsonl_path.name} (epoch={epoch})")
    print(json.dumps(row, indent=2, default=float), flush=True)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="the live run dir (READ-ONLY on its checkpoints/log).")
    ap.add_argument("--log", type=Path, default=None,
                    help="explicit stdout log; default auto-discovers newest .omx/tmp/levelset_mod32cap_*.log.")
    ap.add_argument("--gt-cache", type=Path,
                    default=REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--num-pairs", type=int, default=600)
    ap.add_argument("--pairs", type=int, default=16,
                    help="verdict pairs to render (MEMORY-SAFE small subset). default 16.")
    ap.add_argument("--threads", type=int, default=4,
                    help="OMP/torch threads for the render child (keep modest; do NOT starve the live run).")
    ap.add_argument("--loss-terms-window", type=int, default=50,
                    help="how many recent loss_terms rows to summarize for training health.")
    ap.add_argument("--once", action="store_true",
                    help="single pass then exit (the bounded smoke). Default is also single-pass; "
                         "continuous looping is the launcher's job, not this script's.")
    args = ap.parse_args(argv)

    def log(msg):
        print(f"[{_utc()}] {msg}", flush=True)

    if not args.run_dir.exists():
        raise SystemExit(f"run-dir does not exist: {args.run_dir}")
    log(f"annulus live monitor tick on {args.run_dir.name} (pairs={args.pairs}) {ADVISORY}")
    one_tick(args, log)
    log("tick complete.")


if __name__ == "__main__":
    main()
