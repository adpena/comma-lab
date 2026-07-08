# CRUCIBLE v7 LAUNCH PACKAGE — the operator-facing launch record (SEAL v7.3 round-2)

STORES CONSULTED: ORCHESTRATION_LEDGER (operator decisions verbatim) · SYNTHESIS_seal_v73_round2 + all 4 round-2 lens reports · crucible_v73_compile memo · run-1 run.log (measured cadence + verdicts) · canonical equations (safe_compile_hosc_device_bitidentity_v1, tail_stop_forfeit_floor) · DSL lever registry · DAG FEED-v7seal · memories L25/L68/L79.

- **UTC:** 20260708 · **Authority:** `[macOS advisory]` $0, NO launch — this is the launch-decision
  record a reviewer/operator consults BEFORE approving the run. Live run + pid 63069 UNTOUCHED.
  Pointer contest-CPU **0.19110 UNMOVED** — everything here is APPARATUS/MEANS. The END is the
  byte-closed n600 exact row < 0.19110 from `upstream/evaluate.py` AFTER the run.
- **Config:** `crucible_v7` (`tac.witness_autoconfig.compile_crucible_v7_config`); the per-delta
  evidence is `.omx/research/crucible_v73_compile_20260708.md`.

## 1. MODE DECISION — EVENT (operator override, verbatim)

The round-1 seal recorded THREE independent Opus convergences on **CLOCK for run-1** (unify-L_τ is the
one load-bearing continuation variable; EVENT mode couples 3 schedules to a never-run sensor and
CONFOUNDS the attribution). The operator OVERRODE that recommendation:

> **Operator, 2026-07-08 08:45 (ORCHESTRATION_LEDGER, verbatim):** *"We want to transition to event
> based now and accept the risk, this is a new baseline, not clean but we are choosing to make a leap
> forward and accept the related uncertainty"* + *"Your rec regarding the basis is approved"*.

**Risk framing (BINDING — no reader may mis-grade the run):**
- v7 is a **NEW BASELINE, not an A/B arm.** The operator KNOWINGLY trades clean single-variable
  attribution for the leap. v7-vs-run-1 differences are the **COMPOSED stack**, NOT an isolated
  unify-L_τ measurement.
- Attribution is via **per-stage checkpoints** (every curriculum/phase boundary saves a byte-close-
  loadable ckpt) + **would-fire telemetry** (`cap_fired_before_event` rows = a sensor that never
  triggered → falsification signal) ONLY.
- The deep-math BLOCKER that hid UNDER the mode question — `hosc_beta_end=10.0` freezing β≈10 under
  the EVENT octave-fraction driver (forbidden tanh-saturation) — is **FIXED (round-2 A1)**:
  event-mode endpoint re-derived to the control's frozen β(726)≈**3.177**. Event mode is now coherent.

### Clock-revert recipe (TWO tokens, not one — deep-math MAJOR-2)

Reverting to clock is **NOT** a one-token flip (the code comment's "byte-identical to the incumbent"
was FALSE): v7 flipped BOTH the mode AND the shape off the v6 incumbent. To restore the v6 turnpike:

```
--tau-advance-mode clock   AND   --tau-anneal-shape cosine_hold
```

Flipping ONLY the mode leaves `--tau-anneal-shape geometric`, a THIRD schedule: geometric-clock reaches
τ=0.31 only at the denominator end (`τ(726)=1·0.31^(725/2999)≈0.75`), so the Muon phase runs at τ≈0.75
with NO pre-Muon τ*=0.31 turnpike — the exact wrong-schedule class round-1's BLOCKER-1 fix killed.
**Verified geometric-hold alternative:** if the trainer's geometric path is confirmed to honor
`--tau-hold-frac` (holds τ at the floor for the last fraction), geometric-clock WOULD floor by ep600 —
CONFIRM + test that before offering geometric-clock as "byte-identical"; else it is a two-token revert.

## 2. WATCH-LIST — the pre-registered run signals (primary → secondary)

| priority | signal | pre-registered threshold / response |
|---|---|---|
| **PRIMARY (blocker)** | **Road (class 0) per-class d_seg / flip-rate** | Road is ~2/3 of composite d_seg; run-1 ep125 **0.398** and BARELY moving (0.424→0.398 over 75 ep). v7's entire Road bet rides ONE new mechanism (the −48% directional basis + geometric anneal) with no contingency. **THRESHOLD: if Road flip-rate stays > 0.30 at ep200**, do a per-class d_seg decomposition BEFORE any "basis helps" claim, and fire the registered `road_boundary_fallback` lever (Road↔Undrivable margin term / Menon Road-offset audit / earlier chroma-boundary) rather than a cold restart. |
| **PRIMARY** | **per-class d_seg spectrum vs goal-anchors** (run-1 ep125, canonical order Road0/Lane1/Undriv2/Movable3/MyCar4) | Road **0.398** (dominant) · Lane **0.039** vs analytic-band anchor **0.00087** (~45×; but lane is BAND-carried under lane_offloaded — watch the COMPOSED band+learned lane, NOT the learned render reproducing the dash comb) · Undriv **0.074** · Movable **0.0069** · MyCar **0.0028**. **THRESHOLD: any non-Road class climbing past its ep125 anchor by >2× at equal epoch** → per-class decomposition BEFORE attributing to the basis change. |
| **PRIMARY (blocker)** | **d_pose (verdict + byte-close 3-arm)** | POSE ALONE BLOCKS sub-0.19: run-1 d_pose **1.90** ⇒ contribution √(10·1.90) ≈ **4.35** of implied_S **17.4** — no d_seg win crosses 0.19 while pose sits here; need ~**3e-5**-scale (≈0.018 contribution). d_pose OPEN + UNMEASURED on this vehicle (ancestor 3.4e-5 does NOT transfer). Early verdict-d_pose RISE while training-pose falls = EMA-shadow lag (CONFIRMED run-1 ep50: 9.58→3.59), NOT composition failure — check EMA lag FIRST. **THRESHOLD: no order-of-magnitude pose descent by mid-run (~ep1500)** → escalate; also watch the byte-close 3-arm (ema/live/polyak) d_pose AT EXPORT (the B1 arm-selection consumer) and flag if the winning arm's export d_pose does not track the verdict trajectory. |
| **PRIMARY** | **island-birth (part_frac, born-empty tail)** | The born-empty erasure-tail classes (lane[1], movable[3]) must NUCLEATE: run-1 is POSITIVE (islands born; d_seg 1.0→0.039 lane / 1.0→0.0069 movable via the island/persistence + λ-homotopy suite). v7 must NOT regress this. **THRESHOLD: lane/movable part_frac still ≈0 (unborn) past the `LadderIslandHomotopy` arm birth schedule** → the birth machinery failed; inspect the per-class-λ radii + seed-anneal BEFORE any "islands don't help" claim. |
| secondary | **rate (blob_bytes)** | Archive rate term: run-1 blob ~**89–90 KB** (implied rate ≈ 25·90e3/37.5e6 ≈ **0.060**). **THRESHOLD: blob_bytes trending UP past ~95 KB with no matching d_seg/d_pose gain** → byte budget spent without task return; audit the payload. |
| secondary | **Road↔Lane separatrix jitter** (surface-1/2 M1 counter-arm) | If the lane_offloaded basis + analytic band UNDER-serve lane OR jitter the binding Road↔Lane boundary, switch to the registered `lane_carried_basis_regime` counter-arm (freq_along≈26 + restore lane to `--persistence-classes`). The two regimes are MUTUALLY EXCLUSIVE; v7 commits to lane_offloaded. |
| secondary | **island-amplify surface-3 gating window [0, 350]** (DM-MINOR-2, recorded-not-silent) | The per-class-λ island-amplify (`LadderIslandHomotopy`, surface-3) self-gates on λ_lane FALLING, which requires the analytic band to composite (run.log: band `start_epoch: 350`). So over epochs **[0, 350]** the homotopy GROWS lane islands under the lane_offloaded basis — the PRE-EXISTING round-2 M1 residual, BOUNDED to that window and arguably correct (birth born-empty lane early → hand to the band at ep350 → homotopy self-de-emphasizes as λ_lane drops). NO code change. Already watched via the `lane_carried` counter-arm + the jitter row above. **THRESHOLD: Road↔Lane jitter rises specifically inside [0,350]** → the early-window island growth is the cause → consider moving the band-composite start (350) earlier. NOTE (r4 O-2): [0,350] is run-1's CLOCK window; v7 is EVENT-mode — its window is event-determined by `lane_nucleus` with cap 500, so read the v7 window as [0, lane-fire ≤500], not the literal 350. |
| secondary | **event would-fire telemetry** | Every `cap_fired_before_event` row = the sensor never triggered → the fixed-epoch backstop cap fired = falsification signal (mode-decision risk realized). Muon must fire on `powerlaw_meat` (gated on nucleation-complete), lane-band on `lane_nucleus`, chroma on `annulus_plateau`. |
| liveness | **spike-guard / ep_loss** | `ep_loss==0.0` + spike-guard median-freeze = the frozen-run ALARM signature (accepted-only median can't re-arm). Verify training-is-happening (accepted-frac > floor, weights_stepped) before narrating any plateau. |

**verdict_scope:** every threshold above is an INSTANCE/FORMULATION-level watch signal on THIS run — a tripped threshold triggers a decomposition or a registered counter-arm, NOT a paradigm/family kill. Holistic (operator 2026-07-08): the facets couple — Road d_seg, lane-birth, pose, and rate are read TOGETHER (a pose descent that costs blob_bytes, or a Road gain that comes from lane jitter, is a net-negative even if one facet improves).

## 3. THE ROUND-2 FIXES FOLDED INTO THIS CANDIDATE (config/math)

- **A1 (BLOCKER):** event-mode `--hosc-beta-end` 10.0 → 3.177 (control frozen β(726); ≤4.0 divergence bound).
- **A2:** budget anchor 3.62 → 3.39 (startup-amortized 3000-ep cadence; refuse a TRUE ~15% gate); budget 8.673 → 8.122 d. **(v7.4 r3 DM-MINOR-1 re-anchor: 3.39 → 3.47 on the wider ep25→125 window; budget 8.122 → 8.31 d; strictly more conservative.)**
- **A3:** Polyak degenerate start = epochs+1 (genuinely inert); non-degenerate start 2545 → 2546 (averages exactly 455 ep).
- **A5 (M1):** `--persistence-classes` 'auto' → '3' (movable only; lane rides the analytic band under lane_offloaded); counter-arm registered.
- **A6 (M2):** Road per-class d_seg named the PRIMARY run signal (above) + `road_boundary_fallback` registered duty-to-measure.
- **A7 (R3):** `--per-group-grad-clip` ON (bounds the ep1 gnorm_hijack seg-starvation risk).
- **A8:** tail-stop s*=ν·forfeit reactivation extended (rebalanced-basis ν is stale).

Builder B owns the code-side round-2 fixes (byte-close polyak arm B1 · D16 pool LOUD fallback B2 ·
fire sensor-data epoch B3 · perf-env token-boundary B4 · pre-fire manifest window B5 · closed-loop
guard-test decoupling B6). This package is the config/math + decision-record half.

Pointer 0.19110 UNMOVED — this is a MEANS. The run + byte-close are the next units.
