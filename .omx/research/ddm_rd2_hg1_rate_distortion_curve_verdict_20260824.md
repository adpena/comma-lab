# ddm_rd2 — the HG1/born-small R-D curve is CLOSED with a measured interior point: the curve cannot reach, and pose is 81% of why

**Status:** VERDICT (MAIN-authored from receipts; the supervising arm died to the Opus weekly limit mid-memo — measurements complete, seven positive controls PASSED, one serialization bug at line 545 `int(healed.sum())` after all results had landed).
**verdict_scope:** FAMILY — born-small/HG1 base + explicit correction payloads, three orderings including the ORACLE upper bound, 21 ladder points coded, ONE budget point scored n600.
**axis:** ladder/coding legs `[macOS-CPU scorer-free EXACT byte measurement]` · scored row `[env-mismatch advisory, macOS-CPU, PyAV GT lineage]` — `score_claim=false`, all authority flags false in the receipt.
**Receipts:** `/Volumes/APDataStore/pact/ddm_rd2_hg1_rate_distortion_curve/phaseA2/run.log` (ladder + controls) · `rows/budget_k311571/contest_auth_eval.json` (the scored point) · instrument commit `caf5c0d36a`.

## 1. What was measured

HG1's exactness residual (1,334,939 corrections, shipped 359,280 B coded) was treated as a
**continuous knob** for the first time — the standing open question in
[[dx2-block-ceilings-are-measured-and-sum-to-5-percent]]. Three value-orderings × 11 fractions
were coded with real coders (21 distinct rows), then the byte-cap budget point was solved per
ordering and ONE candidate was scored end-to-end through the real receiver at n600.

**Controls (all PASSED, from run.log):** mirrored constants vs the receiver's own table ·
full-digest custody on both payloads · declared order `tile64_time` AND content sorted in it ·
full-set re-encode byte-identical to shipped raw · coder race reproduces the shipped 359,280 B ·
flip counts reproduce bo2 EXACTLY (base 40,981 / cand 1,527,551 / new 1,504,691 / healed 18,121) ·
**the SHIPPED receiver accepts the 311,571-correction budget payload and applies exactly 311,571
corrections** (receiver-realizable, not an arithmetic exercise).

## 2. The budget point, scored (the decisive row)

At the cap, the best ordering (`oracle_tile`, an ORACLE — not realizable) buys k=311,571
corrections for 36,812 B → container ≈ 101,128 + 36,812 = **137,940 B — FITS under 137,986**.
Scored through the real receiver at n600:

| quantity | measured | needed for sub-0.12 at 137,940 B | over |
|---|---:|---:|---:|
| d_seg | 0.00857012 → seg S 0.857012 | — | — |
| d_pose | 1.33455658 → pose S 3.653158 | — | — |
| **distortion S** | **4.510170** | ≤ 0.028163 | **160.1×** |
| vs merely beating dx2 | 4.510170 | ≤ 0.056378 | 80.0× |

(The receipt's `archive_size_bytes: 180368` is the dx2-sized harness placeholder; the composed
`final_score 4.63` in the receipt therefore uses the WRONG rate leg for this candidate — the
honest rate is 25·137,940/37,545,489 = 0.091837. The distortion legs are the measurement.)

**Decomposition — pose is 81.0% of the shortfall and seg corrections cannot touch it.**
Born-small's d_pose 1.33456 is ~**10,183×** the absolute pose budget (d_pose may rise ≤1.25e-4
total, `af1`). Even PERFECT seg repair leaves pose S 3.653 = 129.7× over. **Sixth independent
confirmation of the pose-budget law.**

**Repair efficiency resolved my pre-registered caution in the pessimistic direction.** bo2's
no-corrections distortion excess over dx2 ≈ 5.13108; with the full budget ≈ 4.48205 → S-repair
**12.65%** vs flip-count repair 18.11%. ΔS-weighted repair UNDERPERFORMS flip count: the
orderings optimize flips-per-byte, not S-per-byte, and reach zero pose.

**Lineage caveat, priced:** CPU/PyAV pose can inflate vs DALI/T4 (MC36 measured 21×). Granting
the FULL 21×: pose contribution → ~0.797, still ~28× over; seg 0.857 alone (lineage-robust,
fork 1.43×) exceeds the budget >21×. **The verdict is robust across the axis caveat.**

## 3. What survives regardless of the verdict (banked coding law)

Tile-ordered corrections cost **3.5–5× less per correction** than pixel-ordered
(0.559–0.968 b/corr vs 2.153–3.462; naive scan worst at 2.47–4.62) — a reusable fact about this
residual's spatial coherence. Budget-search rows: oracle_tile k=311,571 @36,812 B ·
oracle_pixel k=131,316 @36,828 B · scan_prefix k=74,061 @36,856 B.

## 4. GESTALT-DELTA

The two-object table in [[dx2-block-ceilings-are-measured-and-sum-to-5-percent]] gains a
**measured interior point**: the family line between dx2 and born-small does NOT bend toward
feasibility — pose stays catastrophic along the whole curve because every correction budget is a
seg-only actuator. Closed as a CURVE, not a point. The only unmeasured object-level candidates
on disk are now `ddm_rj1`'s three receiver-closed precompensation forms (NO-VERDICT, #1224).

Sisters: [[perfect_localization_is_worthless_the_address_is_the_tax_20260824]] (the pose budget) ·
[[the-demand-has-two-readings-distortion-is-worth-42235-bytes]] · `ddm_bo2` (the endpoint this
curve interpolates from).
