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
"""
from __future__ import annotations

import argparse
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
        import psutil  # noqa: PLC0415

        return float(psutil.virtual_memory().total) / (1024.0 ** 3)
    except Exception:
        try:
            return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024.0 ** 3)
        except Exception:
            return 128.0  # last-resort assumption for this fleet's M5 Max


_FLAG_INT = {"--num-pairs", "--render-h", "--render-w", "--verdict-batch", "--mod-dim", "--hidden-dim"}


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
        if t == "--self-orient":
            out["self_orient"] = True
        if t == "--no-self-orient":
            out["self_orient"] = False
        i += 1
    return out


def project_from_launch_sh(path: Path, *, safe_frac: float = DEFAULT_SAFE_FRAC,
                           total_ram_gib: float | None = None) -> MemoryProjection:
    flags = parse_launch_flags(Path(path).read_text())
    return project_peak_rss_gib(
        num_pairs=int(flags.get("num_pairs", 600)),
        render_h=int(flags.get("render_h", 384)),
        render_w=int(flags.get("render_w", 512)),
        self_orient=bool(flags.get("self_orient", "--self-orient" in Path(path).read_text())),
        verdict_batch=int(flags.get("verdict_batch", DEFAULT_VERDICT_BATCH)),
        safe_frac=safe_frac, total_ram_gib=total_ram_gib)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Witness-launch peak-RSS memory preflight (#205 OOM self-protection).")
    ap.add_argument("--launch-sh", type=str, help="path to an emitted launch.sh to project")
    ap.add_argument("--num-pairs", type=int, default=600)
    ap.add_argument("--render-h", type=int, default=384)
    ap.add_argument("--render-w", type=int, default=512)
    ap.add_argument("--in-feat", type=int, default=REF_IN_FEAT)
    ap.add_argument("--verdict-batch", type=int, default=DEFAULT_VERDICT_BATCH)
    ap.add_argument("--no-self-orient", action="store_true")
    ap.add_argument("--safe-frac", type=float, default=DEFAULT_SAFE_FRAC)
    ap.add_argument("--total-ram-gib", type=float, default=None)
    ap.add_argument("--strict", action="store_true", help="exit rc=3 when projected peak is unsafe")
    args = ap.parse_args(argv)

    if args.launch_sh:
        proj = project_from_launch_sh(Path(args.launch_sh), safe_frac=args.safe_frac,
                                      total_ram_gib=args.total_ram_gib)
    else:
        proj = project_peak_rss_gib(
            num_pairs=args.num_pairs, render_h=args.render_h, render_w=args.render_w,
            in_feat=args.in_feat, self_orient=not args.no_self_orient,
            verdict_batch=args.verdict_batch, safe_frac=args.safe_frac, total_ram_gib=args.total_ram_gib)

    tag = "SAFE" if proj.safe else "REFUSE"
    print(f"[witness-mem-preflight] {tag}: {proj.reason}")
    print(f"  breakdown (GiB): fixed={proj.fixed_overhead_gib} + cf_mx_cache={proj.cf_cache_gib} "
          f"+ gt={proj.gt_gib} + verdict={proj.verdict_transient_gib} = peak {proj.projected_peak_gib}")
    print(f"  config: num_pairs={proj.num_pairs} render={proj.render_h}x{proj.render_w} "
          f"self_orient={proj.self_orient} verdict_batch={proj.verdict_batch}")
    if not proj.safe and args.strict:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
