# Doubling the aligned window buys 1.21× more depth, not 2× — and the descent has an interior best

**Status:** MEASURED (2026-08-17, MAIN, $0 local Metal). `EF6000`, 6,000 steps, lr 2e-5,
`--ce-fraction 0.0 --softplus-fraction 0.0`, seed 20260715, `--eval-every 200`, rc=0, 2,042 s.
**Axis:** `[macOS-MPS training-signal]`, `quantized_exact_seg` on the trainer's own instrument.
**No score claim. Frontier untouched.** Payload `/Volumes/APDataStore/pact/ddm_ce1/EF6000/`.

## ANSWER FIRST

Init = 2.861616347e-04 = **33,757 flips**.

| arm | steps | endpoint vs init | best vs init | `best_step` |
|---|---:|---:|---:|---:|
| `EF3000` | 3,000 | −2,286 | −2,286 | **3000 (AT the cap)** |
| **`EF6000`** | **6,000** | **−2,620** | **−2,755** | **5200 (INTERIOR)** |

1. **The descent reproduces and deepens.** `improved_over_init = True`, verdict PASS. The
   aligned-objective regime crossing its own initialization is now measured twice.
2. **Doubling the window bought 1.205×, not 2×.** −2,755 vs −2,286 for 2× the compute.
   Sublinear in steps — the descent is real and decelerating.
3. **The best is INTERIOR, which is the qualitative change.** EF3000 stopped *at* its cap
   still falling; EF6000 found its minimum at step 5,200 and did not improve on it over the
   remaining 800 steps. That is the first evidence this regime has a floor it approaches
   rather than a cap it is cut off at.
4. **The tail oscillates wider than the noise floor.** Adjacent evals in the last 1,000 steps
   swing 5,200→5,400 by **+1,290 flips** — 2.1× the measured A/A floor (605). So `best` is
   partly selection over that scatter. **The endpoint (−2,620) is the honest single number;
   the best (−2,755) is the honest upper bound on what a best-checkpoint selector would keep.**

## Scope, stated plainly

**verdict_scope: INSTANCE.** One seed (20260715), one config, the trainer's advisory
`quantized_exact_seg` on the burn-2/PR130-lineage QAT base. **NOT** the `hv1` frontier vehicle,
**NOT** byte-closed, **NOT** exact-eval'd. In S units the best is **−2.335e-3 seg** — 24.3% of
the −0.0095973 gap *if* it transferred and byte-closed, and **neither is measured**. The claim
is narrow: *this regime descends, deepens sublinearly, and has an interior minimum.*

## The wall-clock law, corrected per-curriculum

EF3000 and EF6000 differ **only in step count** — both ran exactly 30 evals (3000/100 = 6000/200),
so the two-point secant isolates the per-step rate cleanly:

```
r_expected_flip = (2042 − 1445) / (6000 − 3000) = 597 / 3000 = 0.199 s/step
fixed + 30 evals = 1445 − 3000(0.199)          = 848 s
```

versus the mixture-fit law (`F=122.29 · r=0.22267 · e=25.400`, which predicts
`F + 30e = 884.29`). So on this curriculum **`r` is 0.199, 10.6% cheaper per step than the
mixture rate**, and the fixed+eval block is 848 s vs 884 s (−4.1%).

This confirms this morning's residual as mechanism, not noise: the law's `r` was fit across a
curriculum MIXTURE, and `expected_flip` is cheaper per step than the `ce`/`softplus` phases it
was averaged with. The predicted-vs-measured error grew monotonically with step count
(−2.7% at 600 · −6.9% at 3,000 · **−8.0% at 6,000**) exactly as a per-step-rate error would.

**Honest limit:** two points cannot separate `F` from `e`. Only their SUM (848 s) is measured
here; quoting a per-curriculum `F` or `e` separately would be unlicensed.

## What this does NOT settle

- **Transfer.** Every number here is on the burn-2 base through the trainer's own advisory
  instrument. Whether an aligned-objective descent produces a byte-closeable candidate on the
  `hv1` frontier vehicle is a **different measurement** and it is where the S actually lives.
- **The floor's identity.** The interior best at 5,200 is consistent with approaching a floor,
  but one seed cannot distinguish "floor" from "this seed's basin." The second seed is owed.
- **Whether 6,000 is the right length.** The best moved from the cap (EF3000) to the interior
  (EF6000), so somewhere between 3,000 and 6,000 the window stopped being the binding
  constraint. Nothing here locates that point.

## NEXT

1. **The drain fires first** — `--film-row-dropout 0.077` is LIVE on the EF3000 config
   (FRD077, launch 149), single-variable against EF3000's measured −2,286. That is the P0.
2. **Second seed** on EF3000 (~24 min) — both the crossing and the interior-best rest on n=1.
3. **The transfer measurement**, which is the real question.
4. **`cos(sign g)` for unmeasured objectives** — 0.6185 is the best of three, not a ceiling.
