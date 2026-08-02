#!/usr/bin/env python
"""ddm_wd1 — the correction stream has a MINIMUM SCALE, and QA03 ran 78x below it.

Scorer-free. Reads ddm_dc1's measured n600 label+position price curve and answers the one
question neither ba31 nor dc1 asked: *what fraction of the residual must a correction stream
address before it pays for itself at all?*

Both prior arms priced the correction at a single density -- the full residual (f = 1.0). That
is the CHEAPEST point on the curve, because measured B/flip falls monotonically with support
density (1.72 B/flip at rho=2.2e-4 down to 0.33 at rho=2.2e-2). A stream that corrects only a
fraction ``f`` of the residual codes a support at density ``f*rho0``, where coding is DEARER per
flip. So the single-point number is a best case, and the decision-relevant quantity is the
break-even fraction ``f*`` at which the per-flip price crosses the water level.

Consequence, and the reason this is worth a receipt: QA03 addressed 1,866 of 508,639 flips
(0.367% of the residual). That is ~0.013x the break-even fraction. QA03's negative verdict was
therefore taken at a scale where NO correction stream can pay, independently of solver quality
or the cap-censoring ddm_dc1 separately found. The two censorings are distinct and compound.

Definitions (all from registered constants, none remembered):
    N       = 600*384*512 frames*pixels          -- the n600 scorer geometry
    S/flip  = 100 / N                            -- seg term, d_seg = flips/N
    S/byte  = 25 / 37_545_489                    -- rate term denominator
    water   = (S/flip)/(S/byte)                  -- bytes a flip is worth conceding

NON-PROMOTABLE: [macOS-CPU advisory], score_claim=false. This is arithmetic over an existing
measured receipt; it fires no scorer and ships no bytes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Registered scorer geometry. Kept as literals ONLY because they are the contest's own
# constants (upstream/evaluate.py), re-derived below into `water` and cross-checked against the
# registered WATER_B_PER_FLIP so a typo cannot pass silently.
N_PIXELS = 600 * 384 * 512
RATE_DENOMINATOR_BYTES = 37_545_489
REGISTERED_WATER_B_PER_FLIP = 1.2731082153320312

DEFAULT_PRICE_RECEIPT = Path(".omx/research/ddm_dc1_label_price_n600_20260801.json")
# Bases are MEASURED d_seg values with named provenance, not tuned constants:
#   ja1_v4c_live_seg 0.00431179 -- .omx/state/main_hot_state.md POINTER_LINE, "seg 0.4311790
#     (100*0.00431179) UNCHANGED from v4d". This is the LIVE seg term: the pw1 move was on the
#     pose axis only, which is why a seg-axis correction composes onto the live pointer.
#   burn_ep854 0.003943024 -- ddm_ba31_negative_surfaces_20260731.md sec B.3, "burn-4 ep854 base".
# Override with --base NAME=D_SEG when a newer measured base lands; do not edit these in place.
DEFAULT_BASES = {"ja1_v4c_live_seg": 0.00431179, "burn_ep854": 0.003943024}


def water_level() -> float:
    """Bytes-per-flip at which conceding a flip and coding it cost the same S."""
    return (100.0 / N_PIXELS) / (25.0 / RATE_DENOMINATOR_BYTES)


def _log_linear(xs: list[float], ys: list[float], x: float) -> float | None:
    """Interpolate y at x, linear in log(x). Returns None OFF-GRID -- never extrapolates.

    Refusing to extrapolate is the point: the interesting fractions (QA03's 0.367%) fall below
    the measured grid, and an extrapolated price there would be a manufactured number.
    """
    lx = [math.log(v) for v in xs]
    lo = math.log(x)
    if lo < lx[0] or lo > lx[-1]:
        return None
    for i in range(len(lx) - 1):
        if lx[i] <= lo <= lx[i + 1]:
            t = (lo - lx[i]) / (lx[i + 1] - lx[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return None


def load_price_curve(path: Path) -> tuple[list[float], list[float]]:
    """(densities, total B/flip) from dc1's receipt, sorted ascending by density."""
    payload = json.loads(path.read_text())
    rows = sorted(payload["rows"], key=lambda r: r["density"])
    return ([r["density"] for r in rows], [r["b_per_err_total_best"] for r in rows])


def net_s_at_fraction(frac: float, base_d_seg: float, xs: list[float],
                      ys: list[float]) -> dict | None:
    """Net S of correcting `frac` of the residual at `base_d_seg`. None if OFF-GRID.

    Sign convention: NEGATIVE net is a WIN (rate paid minus seg gained).
    """
    density = frac * base_d_seg
    b_per_flip = _log_linear(xs, ys, density)
    if b_per_flip is None:
        return None
    flips = frac * base_d_seg * N_PIXELS
    rate_cost = b_per_flip * flips * (25.0 / RATE_DENOMINATOR_BYTES)
    seg_gain = flips * (100.0 / N_PIXELS)
    return {
        "fraction": frac,
        "density": density,
        "flips": flips,
        "b_per_flip": b_per_flip,
        "x_water": b_per_flip / water_level(),
        "rate_cost_s": rate_cost,
        "seg_gain_s": seg_gain,
        "net_s": rate_cost - seg_gain,
    }


def break_even_fraction(base_d_seg: float, xs: list[float], ys: list[float]) -> float | None:
    """Smallest fraction whose per-flip price is at or under water. None if never on-grid.

    Bisection is valid because the measured price curve is monotone decreasing in density over
    the whole grid; the caller asserts that monotonicity rather than assuming it.
    """
    water = water_level()
    lo, hi = xs[0] / base_d_seg, 1.0
    price_at_full = _log_linear(xs, ys, hi * base_d_seg)
    # None means the base itself is off-grid (a base denser or sparser than anything measured);
    # `> water` at f=1.0 means even the full residual never gets under water. Both are honest
    # "no answer", and the None check must come FIRST or the comparison raises TypeError.
    if lo > 1.0 or price_at_full is None or price_at_full > water:
        return None
    for _ in range(100):
        mid = (lo + hi) / 2.0
        price = _log_linear(xs, ys, mid * base_d_seg)
        if price is None or price > water:
            lo = mid
        else:
            hi = mid
    return hi


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--price-receipt", type=Path, default=DEFAULT_PRICE_RECEIPT,
                    help="ddm_dc1 label+position price receipt (n600)")
    ap.add_argument("--out", type=Path, default=None, help="write JSON receipt here")
    ap.add_argument("--qa03-flips", type=int, default=1866,
                    help="realized flips of the QA03 solve (its receipt: 1866)")
    ap.add_argument("--base", action="append", metavar="NAME=D_SEG", default=None,
                    help="override the measured bases, e.g. --base burn_ep900=0.00381. "
                         "Repeatable; replaces DEFAULT_BASES entirely when given.")
    args = ap.parse_args(argv)

    bases = dict(DEFAULT_BASES)
    if args.base:
        bases = {}
        for spec in args.base:
            if "=" not in spec:
                raise SystemExit(f"--base expects NAME=D_SEG, got {spec!r}")
            name, _, raw = spec.partition("=")
            d_seg = float(raw)
            if not 0.0 < d_seg < 1.0:
                raise SystemExit(f"--base {name}: d_seg must be in (0,1), got {d_seg!r}")
            bases[name.strip()] = d_seg

    xs, ys = load_price_curve(args.price_receipt)
    water = water_level()

    # Self-check: the derived water level must reproduce the registered constant. If it does
    # not, a geometry constant is wrong and every number below is void.
    if abs(water - REGISTERED_WATER_B_PER_FLIP) > 1e-9:
        raise SystemExit(f"water {water!r} != registered {REGISTERED_WATER_B_PER_FLIP!r}")
    # The break-even bisection is only valid on a monotone curve; assert, do not assume.
    if any(ys[i] <= ys[i + 1] for i in range(len(ys) - 1)):
        raise SystemExit("price curve is not monotone decreasing in density; bisection invalid")

    grid = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    receipt = {
        "schema": "ddm_wd1_correction_scale_threshold.v1",
        "evidence_axis": "[macOS-CPU advisory] NON-PROMOTABLE",
        "score_claim": False,
        "promotion_eligible": False,
        "source_price_receipt": str(args.price_receipt),
        "water_b_per_flip": water,
        "grid_density_min": xs[0],
        "grid_density_max": xs[-1],
        "note": ("net_s < 0 is a WIN. Fractions whose support density falls below the measured "
                 "grid are reported OFF-GRID and are NOT extrapolated."),
        "bases": {},
    }

    for name, d_seg in bases.items():
        f_star = break_even_fraction(d_seg, xs, ys)
        curve = [row for f in grid if (row := net_s_at_fraction(f, d_seg, xs, ys))]
        full = net_s_at_fraction(1.0, d_seg, xs, ys)
        qa03_frac = args.qa03_flips / (d_seg * N_PIXELS)
        receipt["bases"][name] = {
            "d_seg": d_seg,
            "residual_flips": d_seg * N_PIXELS,
            "break_even_fraction": f_star,
            "break_even_density": (f_star * d_seg) if f_star else None,
            "full_residual_net_s": full["net_s"] if full else None,
            "qa03_realized_fraction": qa03_frac,
            "qa03_fraction_over_break_even": (qa03_frac / f_star) if f_star else None,
            "curve": curve,
        }
        print(f"\n=== {name}: d_seg={d_seg} residual={d_seg * N_PIXELS:,.0f} flips")
        print(f"    break-even fraction f* = {f_star:.4f} "
              f"(density {f_star * d_seg:.4e}); below it NO stream pays")
        print(f"    full-residual net = {full['net_s']:+.6f} S "
              f"at {full['b_per_flip']:.4f} B/flip ({full['x_water']:.3f}x water)")
        print(f"    QA03 realized {qa03_frac * 100:.4f}% of residual "
              f"= {qa03_frac / f_star:.4f}x the break-even fraction")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=1))
        print(f"\nreceipt -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
