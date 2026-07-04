#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Memory WATERFILL config solver (BUILD #294) — safely maximize throughput inside the envelope.

Operator policy 2026-07-04 (memory ``operator-memory-policy-sole-workload-no-artificial-ceiling``):
sole-workload 128 GB box; the constraint is NEVER-CRASH physics (control-plane floor >= 10 GiB +
~10 GiB spike/model-error margin + no swap-thrash) => single-workload envelope ``safe_frac ~ 0.85``.
Spare RAM buys THROUGHPUT + CACHES, never capacity (capacity stays geometry-driven, mod-19 Whitney).

This solver turns that policy into a config: given the envelope + the MEASURED per-knob curve
points, it exhaustively enumerates the (micro_batch B, verdict_batch) grid — a TINY discrete space,
so exhaustive is the honest choice (no fake sophistication) — projects every candidate's peak RSS
through the REAL preflight projection function (``tools/witness_memory_preflight.project_peak_rss_gib``,
IMPORTED not forked), and returns the argmax-throughput candidate whose projected peak fits the
envelope, alongside the full frontier table.

NO-FAKE discipline on the two knobs:

* **micro_batch B** — curve points are INPUT DATA (the #261 measured rows ship as the provenance-cited
  default set: B1 2.23 s / 5907 MiB and B4 2.19 s / 5918 MiB, n=8, 6 ep, CONTENTION-CONFOUNDED with
  #205 on the Apple GPU; DAG FEED-03g 2026-07-03). The solver treats the knob as MEASURED only when
  uncontended points at the TARGET n-pairs scale exist with >= 2 distinct B values. Until then the
  knob is **UNMEASURED-EXCLUDED** (B pinned to 1 = the byte-identical serial path) and the result
  says so — it never invents a curve. (#293's batched-seed-co-grad RSS delta is the named pending
  measurement.)
* **verdict_batch** — memory uses the preflight's MEASURED verdict-spike model (floor 6 GiB /
  0.11 GiB/pair, 2026-07-02 ledger). Throughput uses the **verdict-wall ∝ 1/batch approximation,
  labeled [modeled]** everywhere it appears: it assumes per-chunk overhead dominates the chunked
  CPU-scorer verdict wall, is anchored to the MEASURED #205 wall at vbatch 32 (2062–2439 s per 25-ep
  eval window), and is UNVALIDATED at other batch sizes — replace it with measured points when they
  exist. Under ``--async-hidden`` (the M2 decide-on-previous landing, DAG FEED-04o) only the wall
  EXPOSED beyond the train window costs throughput: ``exposed(vb) = max(0, wall(vb) - train_window)``.

$0, pure + unit-testable; the only I/O is reading an optional ``--launch-sh`` and printing.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import witness_memory_preflight as wmp  # noqa: E402  (the REAL projection fn — imported, not forked)

MODELED_TAG = "[modeled]"   # every ∝1/batch-derived number carries this label (NO-FAKE)
MEASURED_TAG = "[measured]"

# ── MEASURED anchors (inputs with provenance; NOT fabricated curves) ────────────────────────────
# #205 verdict wall at vbatch 32, per 25-epoch eval window (run.log verdict_async_done rows,
# 2026-07-03/04): 2062–2439 s. Conservative throughput accounting uses the WORST wall.
MEASURED_VERDICT_WALL_REF_S = 2439.0
MEASURED_VERDICT_WALL_REF_VB = 32
# The 25-ep TRAIN window itself (what the async-hidden verdict must overhang to cost anything):
# #205 mine, 505 epochs / 28.29 h wall ⟹ 28.29*3600 / (505/25) ≈ 5,042 s per 25-ep window (duty
# 47% verdict-active ✓ consistent: 2439/5042 ≈ 0.48). WF-F1 (throughput review 2026-07-04): the
# prior value 2062.0 mislabeled the verdict-wall MINIMUM as the train window, which manufactured
# a phantom ×1.183 speedup at vb=64. Corrected: exposed(vb=32)=max(0,2439−5042)=0 already, so
# vb≥32 are all throughput-EQUIVALENT under async-hide — vb=64 is chosen as never-slower + deeper
# hide margin, NOT as a speedup. Do not cite a vb-64 speedup from this model.
MEASURED_TRAIN_WINDOW_S = 5042.0

# #261 micro-batch measured points (DAG FEED-03g, 2026-07-03): n=8, 6 epochs, GPU shared with #205
# (contention-confounded — B1 inflated 2.23 s vs the prior clean ~1.56 s). These are the DEFAULT
# input points; they are honest data AND insufficient for an n600 decision — see assess_micro_batch.
@dataclass(frozen=True)
class CurvePoint:
    knob: str            # "micro_batch" | "verdict_batch"
    value: int
    step_s: float | None = None       # measured median step wall (s)
    rss_mib: float | None = None      # measured peak RSS (MiB) at the point's n_pairs
    n_pairs: int | None = None        # scale the point was measured at
    contended: bool = False           # True => GPU shared with another job (throughput masked)
    source: str = ""                  # provenance cite


# #205 memory mine (n205_memory_behavior_mine_20260704.md §2/§4, 46,618 blackbox samples): the
# REALIZED per-window verdict spike at vb=32 is +11.9..+12.9 GiB (mean +12.3) — an immediate
# +4.5-5.4 GiB verdict-thread step PLUS ~+7 GiB MLX-pool climb under concurrent training. The
# preflight's +5.6 GiB component matches only the STEP; its NET projection was accurate at vb=32
# (67.61 vs measured 67.68) ONLY via compensating over-reads in the baseline terms. Therefore the
# solver NEVER scales verdict-batch memory from the +5.6 component: it re-bases on the realized
# +12.3 anchor (conservative — over-projects at the anchor by the baseline over-read, fail-closed)
# and labels every vb != 32 extrapolation [modeled] until measured.
REALIZED_VERDICT_SPIKE_AT_REF_GIB = 12.3      # MEASURED at vb=32 (mine §4 verdict_spike_total_gib)
REALIZED_SPIKE_REF_VB = 32


def realized_verdict_spike_gib(vb: int, *, ref_spike_gib: float = REALIZED_VERDICT_SPIKE_AT_REF_GIB,
                               ref_vb: int = REALIZED_SPIKE_REF_VB) -> float:
    """Realized (concurrency-inclusive) verdict spike. At the anchor: MEASURED. Away from it:
    [modeled] — the measured +0.11 GiB/pair marginal extrapolates UPWARD only; below the anchor the
    spike is floored at the measured value (the ~+7 GiB pool climb is not assumed to shrink)."""
    if vb <= 0:
        raise ValueError("vb must be > 0")
    return max(float(ref_spike_gib),
               float(ref_spike_gib) + wmp.VERDICT_PER_PAIR_GIB * (float(vb) - float(ref_vb)))


DEFAULT_MICRO_BATCH_POINTS: tuple[CurvePoint, ...] = (
    CurvePoint(knob="micro_batch", value=1, step_s=2.23, rss_mib=5907.0, n_pairs=8, contended=True,
               source="#261 B1 n=8 6ep 2026-07-03 (DAG FEED-03g; contention-confounded)"),
    CurvePoint(knob="micro_batch", value=4, step_s=2.19, rss_mib=5918.0, n_pairs=8, contended=True,
               source="#261 B4 n=8 6ep 2026-07-03 (DAG FEED-03g; contention-confounded)"),
)

DEFAULT_VERDICT_BATCH_GRID: tuple[int, ...] = (8, 16, 32, 64, 128, 256)


# ── knob measurement assessment (UNMEASURED => excluded, never invented) ────────────────────────
@dataclass(frozen=True)
class KnobStatus:
    name: str
    measured: bool
    grid: tuple[int, ...]      # the values the solver may search over
    reason: str


def assess_micro_batch(points: tuple[CurvePoint, ...] | list[CurvePoint],
                       *, target_n_pairs: int) -> KnobStatus:
    """MEASURED iff >=2 distinct-B UNCONTENDED points exist AT the target n-pairs scale (both a
    throughput curve AND an RSS delta at the real scale are required to size B against the
    envelope). Anything less => UNMEASURED: pin B=1 (the byte-identical serial path) and say why."""
    pts = [p for p in points if p.knob == "micro_batch"]
    usable = [p for p in pts if (not p.contended) and p.n_pairs == target_n_pairs
              and p.step_s is not None and p.rss_mib is not None]
    distinct = sorted({p.value for p in usable})
    if len(distinct) >= 2:
        return KnobStatus(
            name="micro_batch", measured=True, grid=tuple(distinct),
            reason=f"{len(usable)} uncontended points at n={target_n_pairs} "
                   f"(B in {distinct}) {MEASURED_TAG}")
    why: list[str] = []
    if not pts:
        why.append("no curve points supplied")
    else:
        scales = sorted({p.n_pairs for p in pts})
        if all(p.contended for p in pts):
            why.append("all points contention-confounded (#261 B1~B4 1.02x under #205 contention)")
        if target_n_pairs not in scales:
            why.append(f"no points at target n={target_n_pairs} (points at n={scales}; "
                       f"n600 RSS delta pending #293)")
    return KnobStatus(
        name="micro_batch", measured=False, grid=(1,),
        reason="UNMEASURED — excluded (B pinned to 1, the byte-identical serial path): "
               + "; ".join(why))


def assess_verdict_batch(grid: tuple[int, ...]) -> KnobStatus:
    """Memory side is MEASURED (preflight spike model, 2026-07-02 ledger). Throughput side is the
    ∝1/batch approximation anchored at the measured vb=32 wall — labeled [modeled]."""
    return KnobStatus(
        name="verdict_batch", measured=True, grid=tuple(sorted({int(v) for v in grid if v > 0})),
        reason=f"mem {MEASURED_TAG} (verdict-spike model: floor {wmp.VERDICT_FLOOR_GIB} GiB, "
               f"{wmp.VERDICT_PER_PAIR_GIB} GiB/pair); wall ∝1/batch {MODELED_TAG} anchored at "
               f"vb={MEASURED_VERDICT_WALL_REF_VB} wall {MEASURED_VERDICT_WALL_REF_S:.0f}s")


# ── throughput model (verdict_batch) — every ∝1/vb number is [modeled] ──────────────────────────
def verdict_wall_s(vb: int, *, ref_wall_s: float = MEASURED_VERDICT_WALL_REF_S,
                   ref_vb: int = MEASURED_VERDICT_WALL_REF_VB) -> float:
    """Chunked-verdict wall per eval window. ∝ 1/batch approximation — [modeled]: measured ONLY at
    the anchor (vb=ref_vb); assumes per-chunk overhead dominates."""
    if vb <= 0:
        raise ValueError("vb must be > 0 (0 = the unchunked OOM path the preflight refuses)")
    return float(ref_wall_s) * (float(ref_vb) / float(vb))


def exposed_verdict_wait_s(vb: int, *, train_window_s: float = MEASURED_TRAIN_WINDOW_S,
                           async_hidden: bool = True,
                           ref_wall_s: float = MEASURED_VERDICT_WALL_REF_S,
                           ref_vb: int = MEASURED_VERDICT_WALL_REF_VB) -> float:
    """Wall-clock the verdict EXPOSES per eval window. async_hidden=True models the M2
    decide-on-previous landing (join the PREVIOUS window's verdict => only the overhang beyond the
    train window stalls); False models the synchronous join (pre-M2: +wall(vb) per window — at the
    measured anchors 2439/5042 ≈ +48%, NOT the earlier "~2x" claim; WF-F1 correction)."""
    w = verdict_wall_s(vb, ref_wall_s=ref_wall_s, ref_vb=ref_vb)
    return max(0.0, w - float(train_window_s)) if async_hidden else w


def throughput_multiplier(vb: int, *, baseline_vb: int = MEASURED_VERDICT_WALL_REF_VB,
                          train_window_s: float = MEASURED_TRAIN_WINDOW_S,
                          async_hidden: bool = True,
                          ref_wall_s: float = MEASURED_VERDICT_WALL_REF_S,
                          ref_vb: int = MEASURED_VERDICT_WALL_REF_VB) -> float:
    """Run-throughput multiplier of vb RELATIVE to baseline_vb (>1 = faster). [modeled] via
    exposed_verdict_wait_s; the train window itself is the measured 25-ep wall."""
    t_base = float(train_window_s) + exposed_verdict_wait_s(
        baseline_vb, train_window_s=train_window_s, async_hidden=async_hidden,
        ref_wall_s=ref_wall_s, ref_vb=ref_vb)
    t = float(train_window_s) + exposed_verdict_wait_s(
        vb, train_window_s=train_window_s, async_hidden=async_hidden,
        ref_wall_s=ref_wall_s, ref_vb=ref_vb)
    return t_base / t


# ── the solver ───────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Candidate:
    micro_batch: int
    verdict_batch: int
    projected_peak_gib: float          # the REAL preflight net projection
    adjusted_peak_gib: float           # net - preflight-spike-component + REALIZED spike (mine §4)
    realized_spike_gib: float          # measured 12.3 @32; [modeled] extrapolation elsewhere
    cf_cache_gib: float
    gt_gib: float
    verdict_transient_gib: float
    fixed_overhead_gib: float
    safe: bool                         # preflight SAFE  AND  adjusted peak <= envelope (conjunction)
    throughput_multiplier: float
    throughput_label: str      # [modeled] / [measured]
    reason: str


@dataclass(frozen=True)
class WaterfillResult:
    best: Candidate | None
    candidates: tuple[Candidate, ...]
    knob_status: tuple[KnobStatus, ...]
    total_ram_gib: float
    safe_frac: float
    envelope_gib: float                 # safe_frac * total_ram
    control_plane_floor_gib: float
    floor_ok: bool                      # total - envelope >= floor (never-crash physics leg 1)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_json(self) -> dict:
        return {
            "best": asdict(self.best) if self.best else None,
            "candidates": [asdict(c) for c in self.candidates],
            "knob_status": [asdict(k) for k in self.knob_status],
            "total_ram_gib": self.total_ram_gib, "safe_frac": self.safe_frac,
            "envelope_gib": round(self.envelope_gib, 2),
            "control_plane_floor_gib": self.control_plane_floor_gib,
            "floor_ok": self.floor_ok, "notes": list(self.notes),
        }


def solve_waterfill(
    *,
    num_pairs: int,
    total_ram_gib: float,
    safe_frac: float = 0.85,
    control_plane_floor_gib: float = 10.0,
    render_h: int = 384,
    render_w: int = 512,
    in_feat: int = wmp.REF_IN_FEAT,
    self_orient: bool = True,
    verdict_batch_grid: tuple[int, ...] = DEFAULT_VERDICT_BATCH_GRID,
    micro_batch_points: tuple[CurvePoint, ...] = DEFAULT_MICRO_BATCH_POINTS,
    train_window_s: float = MEASURED_TRAIN_WINDOW_S,
    verdict_wall_ref_s: float = MEASURED_VERDICT_WALL_REF_S,
    verdict_wall_ref_vb: int = MEASURED_VERDICT_WALL_REF_VB,
    async_hidden: bool = True,
) -> WaterfillResult:
    """Exhaustive argmax-throughput over the (B, verdict_batch) grid s.t. projected peak <= envelope.

    PURE + deterministic. Safety verdict per candidate comes from the REAL preflight projection
    (``wmp.project_peak_rss_gib`` — imported). Tie-break among equal-throughput safe candidates:
    LOWEST projected peak, then LOWEST verdict_batch (safety-first waterfill: spend memory only
    where it buys throughput)."""
    mb_status = assess_micro_batch(micro_batch_points, target_n_pairs=num_pairs)
    vb_status = assess_verdict_batch(verdict_batch_grid)
    notes: list[str] = []
    if not mb_status.measured:
        notes.append(f"micro_batch knob: {mb_status.reason}")
    notes.append(
        "verdict-spike calibration (mine 2026-07-04): the preflight's +5.6 component matches only "
        "the verdict-thread STEP; realized per-window spike is +12.3 GiB mean under concurrent "
        "training. The NET preflight was accurate at vb=32 (67.61 vs 67.68 measured) only via "
        "compensating baseline over-reads — safety therefore requires BOTH the preflight verdict "
        "AND the realized-spike-adjusted peak inside the envelope; adjusted values away from vb=32 "
        f"are {MODELED_TAG} until measured")
    envelope = float(safe_frac) * float(total_ram_gib)
    floor_ok = (float(total_ram_gib) - envelope) >= float(control_plane_floor_gib)
    if not floor_ok:
        notes.append(
            f"ENVELOPE INVALID: total {total_ram_gib:.1f} - envelope {envelope:.1f} = "
            f"{total_ram_gib - envelope:.1f} GiB < control-plane floor {control_plane_floor_gib:.1f} "
            f"GiB (never-crash physics leg 1) — refusing all candidates")

    cands: list[Candidate] = []
    for b in mb_status.grid:
        for vb in vb_status.grid:
            proj = wmp.project_peak_rss_gib(
                num_pairs=num_pairs, render_h=render_h, render_w=render_w, in_feat=in_feat,
                self_orient=self_orient, verdict_batch=vb,
                total_ram_gib=total_ram_gib, safe_frac=safe_frac)
            # NOTE: no micro-batch RSS term is added — B is only searchable when its RSS delta at
            # the target scale is MEASURED (assess_micro_batch); with B pinned to 1 the projection
            # is exactly the preflight's serial-path model. UNMEASURED => never invent.
            spike = realized_verdict_spike_gib(vb)
            adjusted = round(proj.projected_peak_gib - proj.verdict_transient_gib + spike, 2)
            mult = throughput_multiplier(
                vb, baseline_vb=verdict_wall_ref_vb, train_window_s=train_window_s,
                async_hidden=async_hidden, ref_wall_s=verdict_wall_ref_s,
                ref_vb=verdict_wall_ref_vb)
            safe = bool(proj.safe) and floor_ok and (adjusted <= envelope)
            cands.append(Candidate(
                micro_batch=b, verdict_batch=vb,
                projected_peak_gib=proj.projected_peak_gib, adjusted_peak_gib=adjusted,
                realized_spike_gib=round(spike, 2),
                cf_cache_gib=proj.cf_cache_gib,
                gt_gib=proj.gt_gib, verdict_transient_gib=proj.verdict_transient_gib,
                fixed_overhead_gib=proj.fixed_overhead_gib, safe=safe,
                throughput_multiplier=round(mult, 4), throughput_label=MODELED_TAG,
                reason=(proj.reason if floor_ok else "envelope violates control-plane floor")
                       + (f" | adjusted (realized-spike) peak {adjusted} GiB"
                          f"{' EXCEEDS envelope' if adjusted > envelope else ''}")))

    safe_cands = [c for c in cands if c.safe]
    best = None
    if safe_cands:
        best = min(safe_cands, key=lambda c: (-c.throughput_multiplier,
                                              c.adjusted_peak_gib, c.verdict_batch, c.micro_batch))
    return WaterfillResult(
        best=best, candidates=tuple(cands), knob_status=(mb_status, vb_status),
        total_ram_gib=float(total_ram_gib), safe_frac=float(safe_frac), envelope_gib=envelope,
        control_plane_floor_gib=float(control_plane_floor_gib), floor_ok=floor_ok,
        notes=tuple(notes))


def format_frontier_table(res: WaterfillResult) -> str:
    """Human-readable frontier table (every candidate: mem breakdown, throughput mult, verdict)."""
    lines = [
        f"WATERFILL frontier — envelope {res.envelope_gib:.1f} GiB "
        f"({res.safe_frac:.0%} of {res.total_ram_gib:.0f} GiB; control-plane floor "
        f"{res.control_plane_floor_gib:.0f} GiB {'OK' if res.floor_ok else 'VIOLATED'})",
    ]
    for k in res.knob_status:
        lines.append(f"  knob {k.name}: grid={list(k.grid)}  {k.reason}")
    lines.append("  (adj GiB = preflight net - its spike component + REALIZED +12.3 GiB spike "
                 "anchor, mine 2026-07-04; conservative, [modeled] away from vb=32)")
    hdr = (f"  {'B':>3} {'vbatch':>6} {'peak GiB':>9} {'adj GiB':>8} {'spike':>6} {'cf':>6} "
           f"{'gt':>5} {'fixed':>5} {'thpt x':>7} {'':10} verdict")
    lines.append(hdr)
    for c in res.candidates:
        mark = "SAFE  " if c.safe else "REFUSE"
        star = " <== BEST" if (res.best is not None and c == res.best) else ""
        lines.append(
            f"  {c.micro_batch:>3} {c.verdict_batch:>6} {c.projected_peak_gib:>9.2f} "
            f"{c.adjusted_peak_gib:>8.2f} {c.realized_spike_gib:>6.2f} {c.cf_cache_gib:>6.2f} "
            f"{c.gt_gib:>5.2f} {c.fixed_overhead_gib:>5.1f} {c.throughput_multiplier:>7.3f} "
            f"{c.throughput_label:>10} {mark}{star}")
    if res.best is None:
        lines.append("  NO SAFE CANDIDATE inside the envelope.")
    else:
        lines.append(
            f"  BEST: micro_batch={res.best.micro_batch} verdict_batch={res.best.verdict_batch} "
            f"peak {res.best.projected_peak_gib:.2f} / adjusted {res.best.adjusted_peak_gib:.2f} GiB "
            f"<= {res.envelope_gib:.1f} GiB, throughput x{res.best.throughput_multiplier:.3f} "
            f"{res.best.throughput_label}")
    for n in res.notes:
        lines.append(f"  NOTE: {n}")
    return "\n".join(lines)


def load_points_json(path: Path) -> tuple[CurvePoint, ...]:
    """Load measured curve points from a JSON file: a list of CurvePoint dicts."""
    rows = json.loads(Path(path).read_text())
    return tuple(CurvePoint(**r) for r in rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Memory waterfill config solver (BUILD #294).")
    ap.add_argument("--launch-sh", type=str, default=None,
                    help="parse the base config (num-pairs / bank flags -> in_feat / self-orient) "
                         "from an emitted launch.sh")
    ap.add_argument("--num-pairs", type=int, default=600)
    ap.add_argument("--in-feat", type=int, default=None)
    ap.add_argument("--no-self-orient", action="store_true")
    ap.add_argument("--render-h", type=int, default=384)
    ap.add_argument("--render-w", type=int, default=512)
    ap.add_argument("--total-ram-gib", type=float, default=None)
    ap.add_argument("--safe-frac", type=float, default=0.85,
                    help="single-workload envelope fraction (operator policy 2026-07-04; keep 0.70 "
                         "under admitted concurrency)")
    ap.add_argument("--control-plane-floor-gib", type=float, default=10.0)
    ap.add_argument("--verdict-batch-grid", type=str,
                    default=",".join(str(v) for v in DEFAULT_VERDICT_BATCH_GRID))
    ap.add_argument("--micro-batch-points-json", type=str, default=None,
                    help="JSON file of measured micro-batch CurvePoint rows (default: the #261 "
                         "measured set, which is honest but insufficient => knob excluded)")
    ap.add_argument("--train-window-s", type=float, default=MEASURED_TRAIN_WINDOW_S)
    ap.add_argument("--verdict-wall-ref-s", type=float, default=MEASURED_VERDICT_WALL_REF_S)
    ap.add_argument("--verdict-wall-ref-vb", type=int, default=MEASURED_VERDICT_WALL_REF_VB)
    ap.add_argument("--sync-verdict", action="store_true",
                    help="model the SYNCHRONOUS verdict join (pre-M2); default models "
                         "decide-on-previous (async-hidden)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    num_pairs, in_feat, self_orient = args.num_pairs, args.in_feat, (not args.no_self_orient)
    render_h, render_w = args.render_h, args.render_w
    if args.launch_sh:
        text = Path(args.launch_sh).read_text()
        flags = wmp.parse_launch_flags(text)
        flags["self_orient"] = bool(flags.get("self_orient", "--self-orient" in text))
        num_pairs = int(flags.get("num_pairs", num_pairs))
        render_h = int(flags.get("render_h", render_h))
        render_w = int(flags.get("render_w", render_w))
        self_orient = flags["self_orient"]
        if in_feat is None:
            in_feat = wmp.derive_in_feat_from_flags(flags)
    if in_feat is None:
        in_feat = wmp.REF_IN_FEAT

    total_ram = args.total_ram_gib if args.total_ram_gib is not None else wmp._total_ram_gib()
    grid = tuple(int(v) for v in str(args.verdict_batch_grid).split(",") if v.strip())
    points = (load_points_json(Path(args.micro_batch_points_json))
              if args.micro_batch_points_json else DEFAULT_MICRO_BATCH_POINTS)

    res = solve_waterfill(
        num_pairs=num_pairs, total_ram_gib=total_ram, safe_frac=args.safe_frac,
        control_plane_floor_gib=args.control_plane_floor_gib,
        render_h=render_h, render_w=render_w, in_feat=int(in_feat), self_orient=self_orient,
        verdict_batch_grid=grid, micro_batch_points=points,
        train_window_s=args.train_window_s, verdict_wall_ref_s=args.verdict_wall_ref_s,
        verdict_wall_ref_vb=args.verdict_wall_ref_vb, async_hidden=(not args.sync_verdict))
    if args.json:
        print(json.dumps(res.to_json(), indent=2))
    else:
        print(f"# config: num_pairs={num_pairs} in_feat={in_feat} self_orient={self_orient} "
              f"render={render_h}x{render_w}")
        print(format_frontier_table(res))
    return 0 if res.best is not None else 3


if __name__ == "__main__":
    raise SystemExit(main())
