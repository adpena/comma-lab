#!/usr/bin/env python
"""Witness-launch peak-RSS memory preflight (#205 OOM self-protection).

The #205 n600 launch OOM'd (safe_run killed it at 90 GB, peak_rss=90300 MiB) BEFORE its first
checkpoint. Root cause (MEASURED 2026-07-02, ledger
``.omx/research/n205_oom_fix_and_relaunch_*.md``): the resident self-orient ``cf_mx_cache``
(~0.072 GiB/pair -> ~41 GiB @ n600) PLUS a transient ~66 GiB spike from the CPU-scorer verdict
running EfficientNet-B2/FastViT-T12 over ONE 600-wide torch batch. The launcher only throughput-gated
at B=8; it never PROJECTED peak RSS at the real config, so the OOM config passed the gate.

This module projects peak RSS from the EMITTED launch flags using constants MEASURED in that ledger's
isolated micro-probes, and refuses (fail-closed) a config whose projected peak exceeds a
control-plane-safe fraction of total RAM (default 70 %, leaving headroom for the OS + control-plane +
parallel agents). It is a CONSERVATIVE projection (over-estimates -> fail-closed), NOT the definitive
guard: safe_run's ``--rss-cap-mb`` remains the runtime backstop. Its point is to REFUSE a known-OOM
config (e.g. ``--verdict-batch 0`` at n600, or n so large the resident cf_mx_cache alone busts RAM)
at launch time, before a multi-hour init burns to a guaranteed OOM.

MEASURED calibration constants (all from the 2026-07-02 ledger; ``experiments/results/
n205_oom_probe_*/verdict_mem_microprobe.py`` + the trainer TAC_MEM_PROBE rows):
  * cf_mx_cache active: n64 -> 4.62 GiB (0.072/pair), n300 -> 21.04 GiB (0.070/pair) @ 384x512,
    in_feat~=88, self-orient.  (self-orient OFF => a single shared tensor, ~one pair.)
  * verdict transient (peak over baseline): unchunked N=600 -> +66.2 GiB; chunked vbatch in {8,32}
    -> +5.6 GiB floor. => ~0.11 GiB/pair marginal, ~6 GiB floor.
  * gt uint8 keyframes: 2 frames x 874x1164x3 / pair.

C4 FIX (pre-launch SEAL review 2026-07-04, .omx/research/fresh_run_config_adversarial_review_
20260704.md): the ``--launch-sh`` path was in_feat-BLIND — it parsed neither the curvelet-bank
flags (``--bank-n-scales`` etc.) nor ``--n-dir-freqs``/``--max-bank-freq``, and silently ignored an
``--in-feat`` override, so a bank-6 config (measured in_feat 176 -> cf_mx_cache 86.4 GiB -> true
peak 110.81 GiB REFUSE) was handed a FALSE "SAFE 67.6" — the exact surrogate-green class this tool
exists to extinct. The fix derives ``in_feat`` from the parsed launch flags EXACTLY the way the
trainer does (see :func:`derive_in_feat_from_flags`) and honors an explicit ``--in-feat`` override
alongside ``--launch-sh``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# ── MEASURED constants (2026-07-02 ledger) ─────────────────────────────────────────────────────
CF_PER_PAIR_GIB_REF = 0.072          # cf_mx_cache active per pair @ REF_PIXELS, REF_IN_FEAT, self-orient
REF_PIXELS = 384 * 512
REF_IN_FEAT = 88
GT_PER_PAIR_GIB = 2 * 874 * 1164 * 3 / (1024.0 ** 3)   # uint8, 2 frames/pair (~0.00568)
VERDICT_PER_PAIR_GIB = 0.11          # marginal fp32-cast + EfficientNet/FastViT activations per batched pair
VERDICT_FLOOR_GIB = 6.0              # measured chunked floor (vbatch 8/32 both ~5.6)
FIXED_OVERHEAD_GIB = 15.0            # python + torch scorers + MLX buffer pool + lstar/numpy caches (conservative)
DEFAULT_SAFE_FRAC = 0.70             # refuse above this fraction of total RAM (control-plane + coexistence headroom)
DEFAULT_VERDICT_BATCH = 32           # trainer default when --verdict-batch is not emitted


@dataclass(frozen=True)
class MemoryProjection:
    num_pairs: int
    render_h: int
    render_w: int
    in_feat: int
    self_orient: bool
    verdict_batch: int
    cf_cache_gib: float
    gt_gib: float
    verdict_transient_gib: float
    fixed_overhead_gib: float
    projected_peak_gib: float
    total_ram_gib: float
    safe_frac: float
    safe_ceiling_gib: float
    safe: bool
    reason: str


def project_peak_rss_gib(
    *,
    num_pairs: int,
    render_h: int = 384,
    render_w: int = 512,
    in_feat: int = REF_IN_FEAT,
    self_orient: bool = True,
    verdict_batch: int = DEFAULT_VERDICT_BATCH,
    total_ram_gib: float | None = None,
    safe_frac: float = DEFAULT_SAFE_FRAC,
) -> MemoryProjection:
    """Project peak RSS (GiB) for a witness launch config and decide launch-safety.

    Pure + deterministic (unit-testable). ``verdict_batch <= 0`` => the unchunked (pre-fix) N-wide
    batch => the full ~0.11 GiB/pair verdict spike (this is what makes the projection REFUSE the
    #205-original OOM config)."""
    pix_ratio = (render_h * render_w) / float(REF_PIXELS)
    feat_ratio = in_feat / float(REF_IN_FEAT)
    per_pair = CF_PER_PAIR_GIB_REF * pix_ratio * feat_ratio
    # self-orient => per-pair cache of P; else a single shared tensor (~one pair).
    cf_cache = per_pair * num_pairs if self_orient else per_pair
    gt = GT_PER_PAIR_GIB * num_pairs
    eff_batch = num_pairs if (verdict_batch is None or verdict_batch <= 0) else min(int(verdict_batch), num_pairs)
    verdict = max(VERDICT_FLOOR_GIB, VERDICT_PER_PAIR_GIB * eff_batch)
    peak = FIXED_OVERHEAD_GIB + cf_cache + gt + verdict

    if total_ram_gib is None:
        total_ram_gib = _total_ram_gib()
    ceiling = safe_frac * total_ram_gib
    safe = peak <= ceiling
    if safe:
        reason = (f"projected peak {peak:.1f} GiB <= safe ceiling {ceiling:.1f} GiB "
                  f"({safe_frac:.0%} of {total_ram_gib:.0f} GiB)")
    else:
        reason = (f"projected peak {peak:.1f} GiB EXCEEDS safe ceiling {ceiling:.1f} GiB "
                  f"({safe_frac:.0%} of {total_ram_gib:.0f} GiB) — would OOM / starve control-plane. "
                  f"Reduce --num-pairs, ensure --verdict-batch>0 (got {verdict_batch}), or free RAM.")
    return MemoryProjection(
        num_pairs=num_pairs, render_h=render_h, render_w=render_w, in_feat=in_feat,
        self_orient=self_orient, verdict_batch=int(verdict_batch or 0),
        cf_cache_gib=round(cf_cache, 2), gt_gib=round(gt, 2),
        verdict_transient_gib=round(verdict, 2), fixed_overhead_gib=FIXED_OVERHEAD_GIB,
        projected_peak_gib=round(peak, 2), total_ram_gib=round(total_ram_gib, 1),
        safe_frac=safe_frac, safe_ceiling_gib=round(ceiling, 1), safe=safe, reason=reason)


def _total_ram_gib() -> float:
    try:
        import psutil

        return float(psutil.virtual_memory().total) / (1024.0 ** 3)
    except Exception:
        try:
            return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024.0 ** 3)
        except Exception:
            return 128.0  # last-resort assumption for this fleet's M5 Max


_FLAG_INT = {"--num-pairs", "--render-h", "--render-w", "--verdict-batch", "--mod-dim", "--hidden-dim",
             # C4 fix: the front-end flags in_feat is derived from (trainer argparse:4785-4799).
             "--bank-n-scales", "--bank-n-orient0", "--bank-n-iso", "--n-dir-freqs"}
_FLAG_FLOAT = {"--bank-f0", "--bank-base", "--max-bank-freq"}


def parse_launch_flags(text: str) -> dict:
    """Extract the memory-relevant flags from a launch.sh body (or an argv string). Robust to the
    ``\\``-continued multiline launch.sh the launcher writes. Missing flags fall back to defaults."""
    out: dict = {}
    toks = re.split(r"\s+", text.replace("\\\n", " ").strip())
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in _FLAG_INT and i + 1 < len(toks):
            try:
                out[t.lstrip("-").replace("-", "_")] = int(toks[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        if t in _FLAG_FLOAT and i + 1 < len(toks):
            try:
                out[t.lstrip("-").replace("-", "_")] = float(toks[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        if t == "--self-orient":
            out["self_orient"] = True
        if t == "--no-self-orient":
            out["self_orient"] = False
        i += 1
    return out


# ── C4 fix: in_feat derivation from the launch flags (the trainer's EXACT arithmetic) ───────────
# Trainer front-end defaults, cited from the trainer argparse
# (experiments/train_levelset_witness_realized_through_R_mlx.py:4785-4799):
#   --bank-n-scales 4  --bank-n-orient0 6  --bank-f0 2.0  --bank-base 2.0  --bank-n-iso 4
#   --max-bank-freq None  --self-orient False  --n-dir-freqs 6
_TRAINER_BANK_DEFAULTS: dict = {
    "bank_n_scales": 4, "bank_n_orient0": 6, "bank_f0": 2.0, "bank_base": 2.0,
    "bank_n_iso": 4, "max_bank_freq": None, "n_dir_freqs": 6,
}


def curvelet_bank_cols(*, n_scales: int, n_orient0: int, f0: float, base: float, n_iso: int,
                       max_freq: float | None) -> int:
    """Column count of the generic curvelet/shearlet bank — an EXACT replication of
    ``curvelet_directional_B`` (src/tac/boundary_math/lever_b_levelset_generator.py:136-173, the
    function the trainer calls at train_levelset_witness_realized_through_R_mlx.py:1491-1499).

    Deliberately replicated (NOT imported — per the C4 fix contract this tool stays standalone and
    must not import the trainer; the bank helper's arithmetic is frozen here with the same dtype
    dance): columns are built as float32 vectors, the ``max_freq`` cap computes float64 norms OF the
    float32-rounded components, ``keep = norms <= max_freq + 1e-6``. The float32 rounding is
    LOAD-BEARING at the cap boundary: bank-6 @ max_freq 64 keeps 84 of 88 atoms (4 of the f=64
    atoms round a hair above 64+1e-6), which is exactly the measured in_feat 176 = 2*84 + 8.
    """
    import numpy as np  # lazy: keep bare-module import light; .venv (the only launch env) has numpy

    cols: list = []
    for j in range(int(n_scales)):
        f_j = float(f0) * (float(base) ** j)
        l_j = int(n_orient0) * (2 ** (j // 2))  # parabolic curvelet doubling
        for l in range(l_j):  # noqa: E741 — mirrors the bank helper's loop variable
            theta = np.pi * l / l_j
            cols.append(np.array([f_j * np.cos(theta), f_j * np.sin(theta)], dtype=np.float32))
    for i in range(int(n_iso)):
        theta = np.pi * i / max(int(n_iso), 1)
        f_low = float(f0) * 0.5
        cols.append(np.array([f_low * np.cos(theta), f_low * np.sin(theta)], dtype=np.float32))
    stacked = np.stack(cols, axis=1).astype(np.float32)  # (2, n_feats)
    if max_freq is not None:
        norms = np.sqrt((stacked.astype(np.float64) ** 2).sum(axis=0))
        keep = norms <= float(max_freq) + 1e-6
        if not keep.any():  # never empty the bank — keep the single lowest-freq atom
            keep = norms <= float(norms.min()) + 1e-6
        stacked = stacked[:, keep]
    return int(stacked.shape[1])


def derive_in_feat_from_flags(flags: dict) -> int:
    """Derive ``in_feat`` from parsed launch flags EXACTLY the way the trainer derives it
    (train_levelset_witness_realized_through_R_mlx.py:1491-1513):

        B = curvelet_directional_B(bank, max_freq)        # cols atoms
        in_feat = curvelet_feats(...).shape[1]            # = 2 * cols  (sin + cos)
        if self_orient: in_feat += 4 * n_dir_freqs        # dir_w

    Missing flags fall back to the trainer argparse defaults (``_TRAINER_BANK_DEFAULTS``;
    self_orient default False)."""
    d = _TRAINER_BANK_DEFAULTS
    cols = curvelet_bank_cols(
        n_scales=int(flags.get("bank_n_scales", d["bank_n_scales"])),
        n_orient0=int(flags.get("bank_n_orient0", d["bank_n_orient0"])),
        f0=float(flags.get("bank_f0", d["bank_f0"])),
        base=float(flags.get("bank_base", d["bank_base"])),
        n_iso=int(flags.get("bank_n_iso", d["bank_n_iso"])),
        max_freq=flags.get("max_bank_freq", d["max_bank_freq"]),
    )
    in_feat = 2 * cols
    if bool(flags.get("self_orient", False)):
        in_feat += 4 * int(flags.get("n_dir_freqs", d["n_dir_freqs"]))
    return int(in_feat)


def project_from_launch_sh(path: Path, *, safe_frac: float = DEFAULT_SAFE_FRAC,
                           total_ram_gib: float | None = None,
                           in_feat_override: int | None = None) -> MemoryProjection:
    text = Path(path).read_text()
    flags = parse_launch_flags(text)
    self_orient = bool(flags.get("self_orient", "--self-orient" in text))
    flags["self_orient"] = self_orient
    # C4 fix: derive in_feat from the emitted bank/dir flags (the FALSE-SAFE hole: bank-6 =>
    # in_feat 176 => cf_mx_cache x2 => 110.81 GiB REFUSE, previously projected 67.6 SAFE).
    in_feat = int(in_feat_override) if in_feat_override is not None else derive_in_feat_from_flags(flags)
    return project_peak_rss_gib(
        num_pairs=int(flags.get("num_pairs", 600)),
        render_h=int(flags.get("render_h", 384)),
        render_w=int(flags.get("render_w", 512)),
        in_feat=in_feat,
        self_orient=self_orient,
        verdict_batch=int(flags.get("verdict_batch", DEFAULT_VERDICT_BATCH)),
        safe_frac=safe_frac, total_ram_gib=total_ram_gib)


# ── SYSTEM-aware admission (P0 crash prevention) ────────────────────────────────────────────────
# The per-run projection above is a SINGLE-JOB peak vs a fraction of RAM. It is BLIND to what else is
# already running — two runs each individually under the fraction can still SUM over total RAM (the
# 2026-07-02 crash). ``system_aware_admission`` composes this projection's peak with the live
# system-wide used RAM + all active jobs' remaining growth via the system_memory_governor, so the
# launcher can HARD-REFUSE the SUM-over-RAM case. The per-run projection remains a fast local floor.
def system_aware_admission(projected_peak_gib: float, *, exclude_pid: int | None = None):
    """Return the governor's live ``LiveAdmissionContext`` for a job whose projected peak is
    ``projected_peak_gib`` (imports the governor lazily; raises if it is unavailable so the caller can
    decide fail-open vs fail-closed)."""
    import system_memory_governor as gov  # tools/ is on sys.path (same dir)

    return gov.live_admission_decision(projected_new_gib=float(projected_peak_gib), exclude_pid=exclude_pid)


# ── SELF-CALIBRATING MARGIN LEDGER (BUILD #294 piece D) ─────────────────────────────────────────
# The projection is a MODEL; the ledger measures the model. Every gated launch appends
# {run_dir, projected_peak, config_hash, ts}; ``--reconcile <run_dir>`` appends the MEASURED
# actual peak + residual; ``calibrated_margin()`` returns the measured p95 |projected - actual|
# over the ledger (falling back to the policy's ASSUMED ~10 GiB spike/model-error margin when
# < 3 reconciled rows exist — labeled, never silently invented). fcntl-locked JSONL append,
# mirroring the canonical .omx/state store pattern (Catalog #128/#131 sister discipline).
_REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = _REPO_ROOT / ".omx" / "state" / "memory_projection_ledger.jsonl"
BLACKBOX_PATH = _REPO_ROOT / ".omx" / "state" / "memory_blackbox.jsonl"
ASSUMED_MARGIN_GIB = 10.0   # operator memory-policy leg 2 (spike/model-error) — the <3-row fallback
_MIN_ROWS_FOR_CALIBRATION = 3


def config_hash_from_launch_sh(path: Path) -> str:
    """Stable 16-hex config hash = sha256 of the launch.sh bytes."""
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def append_ledger_row(row: dict, ledger_path: Path | None = None) -> dict:
    """fcntl-LOCK_EX append of one JSON line (canonical .omx/state JSONL store pattern)."""
    import datetime as _dt
    import fcntl

    row = dict(row)
    row.setdefault("ts", _dt.datetime.now(_dt.UTC).isoformat())
    ledger_path = Path(ledger_path if ledger_path is not None else LEDGER_PATH)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return row


def read_ledger_rows(ledger_path: Path | None = None) -> list[dict]:
    p = Path(ledger_path if ledger_path is not None else LEDGER_PATH)
    if not p.exists():
        return []
    rows: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def record_projection(run_dir: Path, launch_sh: Path, proj: MemoryProjection,
                      *, note: str = "", ledger_path: Path | None = None) -> dict:
    """Append the projection row the launch gate computed ({run_dir, projected_peak, config_hash, ts})."""
    return append_ledger_row({
        "event": "projection",
        "run_dir": str(run_dir),
        "projected_peak_gib": float(proj.projected_peak_gib),
        "config_hash": config_hash_from_launch_sh(launch_sh),
        "num_pairs": proj.num_pairs, "in_feat": proj.in_feat,
        "verdict_batch": proj.verdict_batch, "self_orient": proj.self_orient,
        "safe_frac": proj.safe_frac, "safe": proj.safe,
        "note": note,
    }, ledger_path)


_SAFE_RUN_PEAK_RE = re.compile(r"peak_rss=(\d+(?:\.\d+)?)MiB")
_SAFE_RUN_JSON_PEAK_RE = re.compile(r'"peak_rss_mib"\s*:\s*(\d+(?:\.\d+)?)')


def actual_peak_from_run_log(text: str) -> tuple[float, str] | None:
    """Parse the run's actual peak RSS (GiB) from safe_run's exit telemetry in run.log:
    the ``SAFE_RUN ... peak_rss=NNNNMiB`` detail line or the ``"peak_rss_mib": N`` JSON row.
    Present only after the wrapped process EXITS (safe_run reports peak at exit). Returns
    ``(peak_gib, source)`` or None."""
    peaks = [float(m) for m in _SAFE_RUN_PEAK_RE.findall(text)]
    peaks += [float(m) for m in _SAFE_RUN_JSON_PEAK_RE.findall(text)]
    if not peaks:
        return None
    return (max(peaks) / 1024.0, "run_log_safe_run_peak")


# UNITS (measured, load-bearing — #205 memory mine §1, n205_memory_behavior_mine_20260704.md):
# the governor/blackbox tracked field NAMED ``current_rss_gib`` is actually KiB/1e6 units
# (memory_guard.group_rss_gb returns sum(rss_kb)/1e6): 1 unit = 1.024e9 B ~= 0.9537 true GiB
# (cross-validated vs ps: 65.27 units == 62.25 true GiB). The blackbox SYSTEM fields are true GiB.
# Underlying rename/fix is owned upstream (memory_guard) — until then every consumer converts.
TRACKED_RSS_UNIT_TO_GIB = 1e6 * 1024 / (1024.0 ** 3)   # = 0.95367431640625


def actual_peak_from_blackbox(run_dir: Path, blackbox_path: Path | None = None,
                              ) -> tuple[float, str] | None:
    """Fallback actual-peak source for a STILL-RUNNING (or telemetry-less) run: max group RSS over
    the governor black-box samples whose tracked label references this run dir's name — scanning
    the LIVE file AND its rotated archives (.omx/state/archive/), converted from tracked units to
    TRUE GiB (see TRACKED_RSS_UNIT_TO_GIB). This is a MEASURED lower bound on the true peak (2s
    sampling; the run may still grow). Returns ``(peak_gib, source)`` or None."""
    p = Path(blackbox_path if blackbox_path is not None else BLACKBOX_PATH)
    candidates = [p] if p.exists() else []
    archive_dir = p.parent / "archive"
    if archive_dir.is_dir():
        candidates += sorted(archive_dir.glob(p.stem + "_*" + p.suffix))
    if not candidates:
        return None
    name = Path(run_dir).name
    peak_gib = 0.0
    found = False
    for fp in candidates:
        try:
            f = open(fp, encoding="utf-8")  # noqa: SIM115 - closed by the with below; open guarded for OSError
        except OSError:
            continue
        with f:
            for line in f:
                if name not in line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # MIXED-ERA rows (GB-F1 units fix, BUILD #298): rows written after the governor's
                # units-at-read fix carry ``tracked_units: "true_gib"`` and need NO correction;
                # legacy rows are KiB/1e6 units and get the x0.9537 correction. Convert PER ROW
                # before taking the max so old+new archives reconcile coherently.
                factor = 1.0 if row.get("tracked_units") == "true_gib" else TRACKED_RSS_UNIT_TO_GIB
                for t in row.get("tracked", []):
                    if name in str(t.get("label", "")):
                        found = True
                        r = float(t.get("current_rss_gib", 0.0)) * factor
                        if r > peak_gib:
                            peak_gib = r
    if not found:
        return None
    return (peak_gib, "blackbox_tracked_max_units_corrected")


def reconcile_run_dir(run_dir: Path, *, ledger_path: Path | None = None,
                      blackbox_path: Path | None = None,
                      actual_override: tuple[float, str] | None = None) -> dict:
    """Measure the run's actual peak (run.log safe_run telemetry, else black-box tracked max) and
    append the residual row. ``actual_override=(gib, source)`` records an actual peak MEASURED
    elsewhere (e.g. the #205 mine's 46,618-sample units-corrected 67.68 — memo
    n205_memory_behavior_mine_20260704.md §4); the source cite is mandatory + non-placeholder.
    Raises RuntimeError when no measured actual-peak source exists (never invents a number) or
    when no projection row exists for the run_dir."""
    run_dir = Path(run_dir)
    if actual_override is not None:
        src = str(actual_override[1]).strip().lower()
        if not src or src in {"<source>", "tbd", "todo", "placeholder", "n/a"}:
            raise RuntimeError("actual_override requires a real measurement-source cite")
    projections = [r for r in read_ledger_rows(ledger_path)
                   if r.get("event") == "projection" and r.get("run_dir") == str(run_dir)]
    if not projections:
        raise RuntimeError(f"no projection ledger row for run_dir {run_dir} — append one first "
                           f"(the launch gate does this automatically; --record-projection for backfill)")
    proj_row = projections[-1]

    run_log = run_dir / "run.log"
    actual: tuple[float, str] | None = actual_override
    if actual is None and run_log.exists():
        actual = actual_peak_from_run_log(run_log.read_text(errors="replace"))
    still_running = actual is None
    if actual is None:
        actual = actual_peak_from_blackbox(run_dir, blackbox_path)
    if actual is None:
        raise RuntimeError(
            f"no MEASURED actual-peak source for {run_dir}: run.log has no safe_run peak telemetry "
            f"(run still alive?) and the black-box has no tracked samples for this label. "
            f"Refusing to invent a number.")
    actual_gib, source = actual
    projected = float(proj_row["projected_peak_gib"])
    return append_ledger_row({
        "event": "reconcile",
        "run_dir": str(run_dir),
        "config_hash": proj_row.get("config_hash"),
        "projected_peak_gib": projected,
        "actual_peak_gib": round(actual_gib, 3),
        "residual_gib": round(projected - actual_gib, 3),   # + => over-projection (conservative side)
        "actual_source": source,
        "in_progress": bool(still_running and source.startswith("blackbox")),
    }, ledger_path)


def calibrated_margin(ledger_path: Path | None = None) -> tuple[float, str]:
    """Measured p95 of |projected - actual| over the ledger's reconcile rows. Falls back to the
    ASSUMED policy margin (10 GiB) when < 3 reconciled rows exist — labeled, never silent."""
    ledger_path = Path(ledger_path if ledger_path is not None else LEDGER_PATH)
    residuals = sorted(abs(float(r["residual_gib"])) for r in read_ledger_rows(ledger_path)
                       if r.get("event") == "reconcile" and r.get("residual_gib") is not None)
    n = len(residuals)
    if n < _MIN_ROWS_FOR_CALIBRATION:
        return (ASSUMED_MARGIN_GIB,
                f"assumed_default_insufficient_rows(n={n}<{_MIN_ROWS_FOR_CALIBRATION}) "
                f"[policy leg-2 spike/model-error margin]")
    import math

    idx = max(0, min(n - 1, math.ceil(0.95 * n) - 1))
    return (residuals[idx], f"measured_p95_over_{n}_reconciled_rows")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Witness-launch peak-RSS memory preflight (#205 OOM self-protection).")
    ap.add_argument("--launch-sh", type=str, help="path to an emitted launch.sh to project")
    ap.add_argument("--num-pairs", type=int, default=600)
    ap.add_argument("--render-h", type=int, default=384)
    ap.add_argument("--render-w", type=int, default=512)
    ap.add_argument("--in-feat", type=int, default=None,
                    help="override the coord-feature width (default: derived from the launch.sh "
                    "bank/dir flags on the --launch-sh path, else the measured reference "
                    f"{REF_IN_FEAT}). Honored on BOTH paths (C4 fix).")
    ap.add_argument("--verdict-batch", type=int, default=DEFAULT_VERDICT_BATCH)
    ap.add_argument("--no-self-orient", action="store_true")
    ap.add_argument("--safe-frac", type=float, default=DEFAULT_SAFE_FRAC)
    ap.add_argument("--total-ram-gib", type=float, default=None)
    ap.add_argument("--strict", action="store_true", help="exit rc=3 when projected peak is unsafe")
    ap.add_argument("--system-aware", action="store_true",
                    help="ALSO run the live SYSTEM admission gate (this run's peak + current system-wide "
                         "used + active jobs' growth vs the adaptive ceiling); with --strict, exit rc=4 "
                         "when the SUM over RAM would exceed the ceiling")
    # ── ledger modes (BUILD #294 piece D) ──
    ap.add_argument("--record-projection", action="store_true",
                    help="append the projection to the margin ledger (requires --launch-sh + --run-dir)")
    ap.add_argument("--run-dir", type=str, default=None,
                    help="run directory for --record-projection")
    ap.add_argument("--projected-peak-gib", type=float, default=None,
                    help="with --record-projection: record THIS projected peak instead of the computed "
                         "one (backfill of a historically-recorded projection, e.g. the registry value)")
    ap.add_argument("--reconcile", type=str, default=None, metavar="RUN_DIR",
                    help="measure the run's actual peak (run.log safe_run telemetry, else black-box "
                         "tracked max, units-corrected) and append the residual row; prints the row")
    ap.add_argument("--reconcile-actual-gib", type=float, default=None,
                    help="with --reconcile: record THIS measured actual peak (true GiB) from an "
                         "external measurement (requires --reconcile-actual-source cite)")
    ap.add_argument("--reconcile-actual-source", type=str, default=None,
                    help="mandatory measurement-source cite for --reconcile-actual-gib")
    ap.add_argument("--calibrated-margin", action="store_true",
                    help="print the measured p95 |projected-actual| margin over the ledger "
                         "(falls back to the assumed 10 GiB when <3 reconciled rows; labeled)")
    args = ap.parse_args(argv)

    if args.reconcile:
        override = None
        if args.reconcile_actual_gib is not None:
            override = (float(args.reconcile_actual_gib), str(args.reconcile_actual_source or ""))
        try:
            row = reconcile_run_dir(Path(args.reconcile), actual_override=override)
        except RuntimeError as exc:
            print(f"[witness-mem-preflight] RECONCILE ERROR: {exc}", file=sys.stderr)
            return 5
        print(f"[witness-mem-preflight] reconciled: {json.dumps(row, sort_keys=True)}")
        margin, label = calibrated_margin()
        print(f"[witness-mem-preflight] calibrated margin: {margin:.2f} GiB ({label})")
        return 0

    if args.calibrated_margin:
        margin, label = calibrated_margin()
        print(f"[witness-mem-preflight] calibrated margin: {margin:.2f} GiB ({label})")
        return 0

    if args.launch_sh:
        proj = project_from_launch_sh(Path(args.launch_sh), safe_frac=args.safe_frac,
                                      total_ram_gib=args.total_ram_gib,
                                      in_feat_override=args.in_feat)
    else:
        proj = project_peak_rss_gib(
            num_pairs=args.num_pairs, render_h=args.render_h, render_w=args.render_w,
            in_feat=(REF_IN_FEAT if args.in_feat is None else args.in_feat),
            self_orient=not args.no_self_orient,
            verdict_batch=args.verdict_batch, safe_frac=args.safe_frac, total_ram_gib=args.total_ram_gib)

    if args.record_projection:
        if not (args.launch_sh and args.run_dir):
            print("[witness-mem-preflight] ERROR: --record-projection requires --launch-sh + --run-dir",
                  file=sys.stderr)
            return 5
        if args.projected_peak_gib is not None:
            # Backfill path: record the HISTORICALLY-recorded projection (e.g. the daemon-registry
            # value) rather than the freshly computed one, so residuals measure the model AS GATED.
            import dataclasses

            proj = dataclasses.replace(proj, projected_peak_gib=float(args.projected_peak_gib))
        row = record_projection(Path(args.run_dir), Path(args.launch_sh), proj, note="cli")
        print(f"[witness-mem-preflight] projection recorded: {json.dumps(row, sort_keys=True)}")

    tag = "SAFE" if proj.safe else "REFUSE"
    print(f"[witness-mem-preflight] {tag}: {proj.reason}")
    print(f"  breakdown (GiB): fixed={proj.fixed_overhead_gib} + cf_mx_cache={proj.cf_cache_gib} "
          f"+ gt={proj.gt_gib} + verdict={proj.verdict_transient_gib} = peak {proj.projected_peak_gib}")
    print(f"  config: num_pairs={proj.num_pairs} render={proj.render_h}x{proj.render_w} "
          f"in_feat={proj.in_feat} self_orient={proj.self_orient} verdict_batch={proj.verdict_batch}")

    admit_refused = False
    if args.system_aware:
        try:
            ctx = system_aware_admission(proj.projected_peak_gib)
            d = ctx.decision
            atag = "ADMIT" if d.admit else "REFUSE"
            print(f"[system-admission] {atag}: {d.reason}")
            print(f"  system: total={ctx.snapshot.total_gib:.0f}GiB used={ctx.snapshot.used_gib:.1f}GiB "
                  f"available={ctx.snapshot.available_gib:.1f}GiB ceiling={ctx.ceiling.adaptive_ceiling_gib:.1f}GiB "
                  f"budget={ctx.ceiling.training_budget_gib:.1f}GiB active_jobs={len(ctx.active_jobs)}")
            admit_refused = not d.admit
        except Exception as exc:  # governor unavailable => fail-open on the SYSTEM axis (per-run still gates)
            print(f"[system-admission] WARNING: unavailable ({type(exc).__name__}: {exc}); "
                  f"per-run projection still applies.")

    if args.strict:
        if not proj.safe:
            return 3
        if admit_refused:
            return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
