# ddm_rv17 — WAVE 3, ROUND 1b: two HIGH findings in the fs3 instruments; counter 0/3

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [source + receipt review, scorer-free]` ·
`score_claim: false` · cost $0 · items 2 and 5 of the wave-3 scope. Item 1/3/6 arm still in flight.

## THE ANSWER, FIRST

**Counter 0/3 — two HIGH, six MED, four LOW.** I verified the sharpest myself before adopting it.

**RV17-W3-F2 (HIGH) — a shipped receipt's price-provenance field names a price its own arithmetic
contradicts, next to a sub-0.15 flag.** Verified from disk:

```
/projection/rate_source            = jg2_prior_4.1379_bits_per_token
/projection/tokens_projected_n600  = 15240.0
/projection/rate_bytes_projected   = 5062.144112478031
  → IMPLIED bits/token             = 2.657293497363796      ← the calibration, not 4.1379
/projection/projected_S            = 0.1466302378922147
/projection/clears_sub_015         = True
```

The disarm rebinds the price to 2.657293 and all four of its assertions are numeric — **none touches
the label**. So the receipt computes at the new price and reports the old one, and the mislabel sits
directly beside `clears_sub_015: True` and a 0.1466 projected S. The field's own docstring says it
exists *"because a modelled rate leg and a measured one are different claims"* — which is precisely
the distinction the disarm silently falsifies.

**RV17-W3-F3 (HIGH) — CONTROL B and the realized `+300/+116` are entailed by CONTROL A. They are one
check, not three.** My arm reconstructed jg3's argmin offline and showed the prereg, the chosen
config, and the realized totals are all the *same replay* of the *same rule* over the *same retained
sweep table*, at two prices. Given A, B is forced; given B, `realized == predicted` is arithmetic.

**This is my own wave-2 law arriving from the other direction.** *Three checks that share an
instrument are one check* — I derived it from my own round-8 failure, and it convicts fs3's control
structure on the same standard. The controls demonstrate **reproducibility**, not **correctness**:
they would pass identically if the census rule were wrong, provided it were wrongly applied both
times. The half of the memo's claim they support is *"the price-transfer is wrong"*; the half they
cannot test is *"the census is right."*

**And F8 supplies the alternative reading the controls cannot exclude:** the 2.6573 calibration is an
**average over a 569-token removal**; the 5.9467 is a **marginal over a 300-token addition**. Under a
context-modelled coder those are different quantities — and this campaign already carries two laws
for that shape (direction-dependence at 0.93×/0.09×, union ≠ sum-of-legs at 3.705×). So the same
numbers also read as *the census priced with an average where a marginal was required* — which
implicates the **rule**, not the transfer. I am not asserting that reading; I am recording that the
memo's is not uniquely supported, and that the controls by construction cannot distinguish them.

---

## THE REMAINING ROWS

| id | sev | finding |
|---|---|---|
| F3 | MED | `disarm()` asserts the dict it just wrote, not the value the computation uses — a synthetic repro made it write a spurious key, report `DISARMED_AND_PROVED`, and still run at the old price. Live count 0 (jg3's param is genuinely kw-only), but the **proof cannot detect its own failure** — the named unasserted-repin class |
| F4 | MED | the AST audit's binary classification misfiles five import-time binding forms (module constant, lambda default, class attribute, dataclass `field`, `functools.partial`) as "cured by reassignment", and none trips the refusal gate — so the docstring's *"cannot slip past this shim silently"* is false. Live count 0 |
| F5 | MED | CONTROL A passes **vacuously** for any pair absent from jg3's retained shards (`len(shared) == len(want_sweep)` with both zero). **27 of 600 pairs** are in that zone — real exposure, not hypothetical. Live count 0 this run |
| F6 | MED | the rate baseline is an **unreceipted CLI literal** (`113_847`) while the receipt the tool loads carries its own `token_stream_bytes_base = 109696` under a near-identical name. A **0.08% baseline error flips the refusal into a pass** |
| F7 | MED | `carrier_MEASURED_leg2` is a linear extrapolation (45 B over 454 pairs, re-multiplied by 38) labelled MEASURED, while its siblings are honestly `pose_DERIVED` and `rate_MEASURED_real_reencode` |
| F9–F12 | LOW | `known={"project"}` matches bare function names · `run_price` raises on the format string at zero tokens · `drift_flag` is computed but non-blocking (`clears_admission_bar` could read true beside a 124% flagged drift) · `JG3_SHARDS` is un-overridable and un-sha'd, last-wins on duplicate pairs |

## WHAT VERIFIED CLEAN — and it is substantial

**The P0 payload law holds.** No measure-and-discard in either instrument: the compose tool
materialises the field and persists it (`np.savez_compressed`), immediately sha'd and sized; every
re-screen shard's edit npz is sha'd and sized into `edit_sources`. **No bare excepts anywhere** —
both tools define a fail-closed `RuntimeError` subclass and raise it, and `run_price` refuses
outright when `delta_trustworthy` is false.

**The rebind does reach the computation in the real run** — confirmed end-to-end by the receipt's
implied 2.657293. The charter's suspected "explicit-pass defeats the rebind" path is **not present**.
**CONTROL A is not vacuous in this run** (all 38 present, zero supersets, zero mismatches). **The
composed field did not silently grow** (455 == 455, no admitted pair outside the jg5 subset).
**`--pose-leg-s` is `required=True`** — the pose leg cannot be silently zeroed. And the disarm *does*
carry one genuine functional post-assertion: `break_even_yield()` is actually called after the rebind
and checked to 1e-12.

**My arm disclosed its own instrument sharing**, unprompted: F1's three sub-measurements read the
same two JSON families through the same argmin reimplementation, and it reported them as **one**
finding rather than three. That is the wave-2 law operating inside the review rather than only on it.

---

## COUNTER

**0 / 3.** Items 1, 3, 6 remain with the second arm.

The shape so far inverts wave 2's. There the measurements were sound and the generalizations were
not. Here the two HIGH findings are both about **what a control or a label can license**: a
provenance field that names the wrong price beside a sub-0.15 flag, and a control triad that is one
observation wearing three hats. fs3's arithmetic may well survive intact — the second arm will say —
but its *evidentiary structure* claims more than it can carry, and it does so in the two places a
reader trusts most: the receipt's own labels, and the word "control."

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_wave3_round1b_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]** — gen6 frozen, #1111 operator-HELD.
