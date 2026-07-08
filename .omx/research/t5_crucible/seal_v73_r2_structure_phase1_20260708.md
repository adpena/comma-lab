# SEAL v7.3 · Round-2 · STRUCTURE lens · PHASE-1 BLIND derivation

**Seat:** STRUCTURE (anti-cargo-cult). **Phase:** 1 (BLIND — pre-memo/synthesis).
**Pointer 0.19110 UNMOVED — everything here is MEANS.** Only a byte-closed n600 exact row
`< 0.19110` from `upstream/evaluate.py` (contest-CPU/CUDA, NEVER MPS) moves it.

## STORES CONSULTED (allowlist ONLY — blinding proof)
- `CLAUDE.md` (in-context)
- `docs/operating_manual_craft_handoff.md`
- `experiments/train_levelset_witness_realized_through_R_mlx.py` (argparse + mechanism grep — CODE)
- `src/tac/witness_control/{polyak_finisher,powerlaw_exit,event_wirings,tail_cycles}.py` (CODE, read)
- `src/tac/witness_control/*` (file inventory only for the rest)
- `src/tac/witness_dsl/{curriculum_dsl}.py` (lever-factory grep + directional_basis / ladder_island_homotopy / seg_form_unify_tau defs — CODE)
- `experiments/results/levelset_n600_crucible_v6_run1_20260708T095730Z/run.log` (MEASURED telemetry, read-only)

**NOT consulted (phase-1 discipline):** any `.omx/research/t5_crucible/*` memo, the v7.3 compile/synthesis,
`witness_autoconfig`, `launch.sh`. I did read the v6-run1 schedule facts that appear IN run.log's own WARN
rows (epochs 3000, lr-anneal 1000, muon 726, l7 3000, softmax-temp-end→turnpike) because they are telemetry;
I treat them as "the incumbent v6-run1 numbers" and derive v7 independently, not from them.

---

## D0. Ground truth I derive FROM (MEASURED, v6-run1)

d_seg trajectory (verdict rows), and its per-class decomposition, are the whole basis:

| ep | d_seg | Δ/25ep | Road(0) | Lane(1) | Undriv(2) | Movable(3) | Hood(4) | flip-share Road / Undriv |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0  | 0.745 | — | 0.997 | 1.00 | 1.00 | 1.00 | 0.00 | 0.31 / 0.66 |
| 25 | 0.177 | −0.568 | 0.436 | 0.095 | 0.149 | 0.016 | 0.003 | 0.573 / 0.419 |
| 50 | 0.155 | −0.022 | 0.424 | 0.063 | 0.111 | 0.013 | 0.003 | 0.636 / 0.355 |
| 75 | 0.142 | −0.013 | 0.417 | 0.047 | 0.088 | 0.013 | 0.003 | 0.683 / 0.308 |
| 100| 0.136 | −0.006 | 0.400 | 0.041 | 0.085 | 0.009 | 0.003 | 0.682 / 0.310 |

**Five load-bearing MEASURED facts (each drives a structural claim below):**

1. **The CE descent is already power-law-flat by ep100.** Slope collapses −0.568 → −0.006 over four
   windows; ~0.00024 d_seg/ep and falling. Goal needs d_seg ~0.001 (S<0.15); we are at 0.136 = **~136×
   above goal**. The coarse partition forms in the first ~25 epochs; everything after is the long tail.
2. **Road (class 0, bulk) is the binding residual, overwhelmingly.** flip-share Road 0.68 + Undriv 0.31 =
   **99% of all flip mass**; Road alone contributes `0.68·0.136 ≈ 0.092` of the total d_seg. Road's own
   flip RATE is stuck at 0.40 (was 0.44 at ep25 — barely moving).
3. **The rare/island classes are a rounding error in the SCORE.** Lane(1) contributes `~0.002·0.136 ≈
   2.7e-4`; Movable(3) and Hood(4) near zero. Even driving lane+movable to 0 saves `< 3e-4` of d_seg.
4. **d_pose is NOT solved on this vehicle.** verdict d_pose bounces 7.38 → 9.58 (ep25, EMA-shadow lag)
   → 3.59 → 1.86 (ep100), and the pose-carrier s_t=0.044 fit gives d_pose 2.562 (√(10·2.562)≈5.06,
   catastrophic). This is nowhere near the borrowed RGB-ancestor 3.4e-5 — do NOT cite that number here.
5. **Hood(4) is already solved** (d_seg 0.003, IoU 0.993 from structured-init) — the static core.
   part_frac at init: Road 0.248 / Lane 0.0 / Undriv 0.498 / Movable 0.0 / Hood 0.254. Lane & Movable
   are BORN-EMPTY at init (part_frac 0.0) — deliberate (structured_init lane_px=0).

**The annulus fact (CLAUDE.md + telemetry):** ~97% of d_seg is boundary-jitter in a ~4.7%-area annulus,
NOT region-miss. So Road's 0.40 is jitter on the Road↔Undrivable (horizon) and Road↔Lane separatrices —
it is a BOUNDARY problem on the two big-area classes, not a bulk-interior region error.

---

## D1. Stage topology I would build (from level-set/annealing first principles)

The witness is ONE variational level-set flow; the "curriculum" IS temperature annealing = coarse→fine
scale. From-scratch topology:

```
[form]  L_τ at τ≈1 (=CE)  ── coarse partition, ~25 ep does 75% of the work
   │
[sharpen]  τ anneals τ→τ* (turnpike floor), CONTINUOUS, no discrete switch
   │        ↑ this is where the long tail lives; the binding class (Road boundary) is worked here
   │
   ●  ── EVENT: tau-descent power-law meat exhausted (per-BINDING-class, gated on nucleation-complete)
   │
[finish]  Muon orthogonalized descent at the τ* turnpike (constant τ)
   │
[tail_k]  K warm-restart cycles mining the 1/t tail at τ* (constant-τ turnpike dwell)
   │
[export]  EMA shadow  +  Polyak/Ruppert uniform tail mean  (BOTH candidates; byte-close picks)
```

**Two discipline calls my derivation makes here, independently:**

- **l7 (L∞ terminal sharpen) should be OFF / demoted.** CLAUDE.md flags l7 as a MEASURED DEFECT (L∞
  sharpening inside a viscosity/smoothing flow decouples d_seg; only the smooth stage RAISES d_seg). A
  from-scratch level-set design does NOT bolt an L∞ term onto a viscous mean-curvature flow. So the
  topology has NO separate l7 stage. (If the incumbent sets `l7-start ≈ epochs`, that is l7-effectively-
  off, which is the same thing achieved by placement rather than by omission — acceptable but I'd omit it.)
- **The CE→tau_softplus discrete stage switch should be DISSOLVED into one continuous L_τ homotopy.**
  `L_τ = τ·logsumexp(φ/τ) − φ_y` recovers CE at τ=1 and max-margin at τ→τ*. A discrete `_seg_form_for_epoch`
  dispatch at a fixed ep300 is a PR95-inherited stage bone; the level-set flow has no privileged switch
  epoch. This is the SCALE=annealing=curriculum unification applied honestly. (I derived this before
  seeing that a `seg_form_unify_tau` lever exists — it does, in curriculum_dsl; my blind prescription
  matches it.)

**Where τ-start-of-sharpen belongs:** the measured CE plateau is at **~ep75–100**, not ep300. A fixed
`tau-softplus-start=300` is LATE by ~200 epochs — it burns budget holding τ=1 after the coarse partition
has already stopped improving. A from-scratch design fires the sharpen on the plateau EVENT (see D3), not
a fixed epoch. (Consistent with the operator's DECIDED event mode.)

---

## D2. Finishing-window treatment (the item the prompt flags — Polyak placement)

**Polyak/Ruppert uniform tail mean is correct at a constant-τ turnpike** (the sealed case τ_0=τ_end=0.31):
the iterates ORBIT the basin center; an exponential EMA carries orbit phase at horizon 1/(1−decay), a
uniform mean averages the orbit to O(1/√n) — the strictly better basin-CENTER estimate. It NEVER replaces
the EMA shadow; it is an ADDITIONAL export candidate the byte-close picks. All of that is right.

**THE STRUCTURAL TENSION my blind derivation hits (prompt item a):** the Polyak window is defined as
`start_epoch` (a fixed config integer) + `window = frac · stage_window` (frac 0.1–0.3 of a FIXED stage
window). But Muon — and therefore the turnpike the Polyak mean is supposed to average — fires on the
**powerlaw_meat EVENT**, whose epoch is NOT known at config time. This is a fixed-epoch window keyed to an
event-timed regime. Failure modes:

- **Polyak start too early (< actual muon-fire):** the uniform mean folds in PRE-turnpike, still-DESCENDING
  iterates → biases the exported candidate toward stale, higher-d_seg weights. A uniform mean is
  unforgiving of a descending prefix (unlike EMA, which forgets it).
- **Polyak start too late (fixed epoch after a late event):** tiny n, variance not averaged out — the whole
  point (O(1/√n)) is lost.

**Correct structure:** the Polyak accumulation window must be **anchored to the muon-fire event**, i.e.
`polyak_start = muon_fired_epoch + dwell_settle`, with the fixed `--polyak-finisher-start-epoch` demoted to
a BACKSTOP (the same event/backstop-cap pattern the three schedule transitions already use). The window
should begin only once the constant-τ turnpike is actually engaged. A fixed start-epoch chosen "conserva-
tively early" is the worse of the two failure modes. **PREDICTION for phase-2:** if v7.3 sets a fixed
`polyak-finisher-start-epoch` that is NOT tied to the muon event, that is at least a REVISE (bias risk),
possibly MAJOR if the fixed epoch precedes the expected event.

**Muon placement:** the incumbent v6-run1 had muon(726) < l7(3000), which the trainer itself WARNs is
weaker-d_seg (orthogonalized finisher on a not-yet-formed partition). With l7 dissolved/off (D1), Muon is
correctly the FINAL descent (after τ has annealed to the turnpike) — good, as long as the powerlaw_meat
gate's nucleation-complete positive-control prevents an island-birth transient from being misread as
tau-descent exhaustion (which would fire Muon before the partition is sharp). I'd keep that positive
control MANDATORY, not optional.

---

## D3. Sensor→transition graph I would build

Event-driven (operator DECIDED), numeric `--<x>-start-epoch` = fail-safe BACKSTOP CAP emitting a LOUD
`cap_fired_before_event` row (a firing cap is falsification signal — the sensor never triggered). The
transitions a from-scratch design needs:

1. **τ-sharpen start ← CE/coarse-partition plateau** (per-class or aggregate d_seg plateau). *[The measured
   plateau at ~ep75–100 is the anchor; do NOT hold τ=1 to a fixed ep300.]*
2. **Muon start ← tau-descent power-law meat exhausted**, per BINDING class (powerlaw_meat picks the
   max-remaining-meat class → naturally tracks Road), gated on **nucleation-complete** (islands' anneal
   done) so an island transient can't trip it.
3. **tail_k cycle exits ← per-cycle power-law meat exit**, dwell-floored (a cycle < 3/ν ≈ 237 ep measures
   its own transient), cycle-cap 387 ep, net-ΔS PowerPlay stop.
4. **Polyak accumulation start ← muon-fire event** (D2), not a fixed epoch.

Two SPECIFIC transitions a from-scratch design ALSO wants (rare-class hygiene), matching what I found in
the mechanism code:
5. **lane-band start ← lane critical-nucleus born+formed** (part_frac>0 AND within-flip≤thresh).
6. **seg-chroma-boundary start ← annulus_frac plateau** (a FORMED boundary chroma can then sharpen).

**Gap my blind derivation flags:** there is NO sensor/transition that watches the ROAD (bulk-binding)
class SPECIFICALLY — items 5–6 are rare-class/boundary. Item 2's binding-class picker covers it *for the
Muon gate*, but there is no ROAD-targeted schedule action. See D5/D6.

---

## D4. Budget form

Event-driven ⇒ the budget is a set of CAPS, not a schedule. `epochs` is a total ceiling; each transition's
`--<x>-start-epoch` is a backstop; `k_max = floor(budget / cycle_floor≈387)` bounds the tail cycles. This
is correct: extra budget should EXTEND THE TAIL at τ* (turnpike), not stretch the transients. The one thing
to verify: the total budget must leave room for enough tail cycles to matter — if the caps consume most of
the budget before muon fires, k_max→0 and the tail (the actual meat-miner) never runs. **PREDICTION:** the
budget/cap arithmetic should show ≥2–3 tail cycles reachable after the expected muon-fire; if not, REVISE.

---

## D5. Lever composition — what I'd have ON, and the redundancy I predict

**PRIMARY (must be ON — they touch the BINDING Road boundary):**
- **all-class DIRECTIONAL basis** (freq-along/across two-regime allocation + self-orient). Measured −48%
  d_seg, all-class ⇒ includes Road↔Undrivable/Road↔Lane. Basis-match is PRIOR to capacity. **This is the
  #1 lever and it is Road-relevant** — it must be ON with a live (non-dead) warm-start window.
- **eikonal/viscous margin + length** (C3-normalized) — generic boundary regularity, all-class.
- **seg-chroma-boundary** at the fragile annulus — all-class boundary SHARPENER; but see timing below.
- **Menon logit-adjust on the LOSS surface only** (deployed argmax reads RAW logits) — fine, it re-weights
  toward rare classes in the loss but does not corrupt the verdict.

**SECONDARY (rare-class polish — cap the budget spent here):**
- persistence[1,3] + island_amplify[1,3] + island_seed[1,3] + lane_render_band[1] + lane_prior +
  ladder island homotopy[1,3].

**The redundancy/antagonism I predict (prompt item b):** THREE mechanisms all chase the same thin LANE:
1. `lane_prior_phi1` / `lane_render_band` — analytic openpilot centerline composited PRE-R (measured
   band-vs-GT-lane recall 0.55);
2. `n323_ladder_island_homotopy` — a per-class-λ-gated island-BIRTH LOSS pushing the LEARNED render to
   grow lane islands;
3. `directional_basis` freq-along reallocation — frees/rebalances learned Fourier capacity toward the
   lane tangent when the analytic band "offloads" the lane class.

If the analytic band already composites lane authority at render time, then (2) and (3) are partly
redundant with it — AND potentially ANTAGONISTIC: the verdict reads the witness-alone LEARNED render;
if the LADDER loss + freed basis capacity over-drive lane in the learned render while the analytic band
ALSO fires, the composed lane can over-fire (false lane pixels) → and since lane is only ~2.7e-4 of d_seg,
any Road-boundary capacity spent servicing this triple-driven lane is capacity NOT spent on the 0.092
Road residual. **This is a real "two levers touch lane capacity" pair (LADDER × basis-rebalance
lane_offloaded), plus a third (analytic band). Phase-2 must check whether v7.3 composes all three without a
guard, and whether the freed basis capacity is redirected to ROAD or just re-spent on lane.**

**Correctly-OFF levers (memory-confirmed INERT):** `LEVER4_uniward` / `margin_saliency` msal_uni (measured
at chance vs through-R reachability). Keep OFF. verdict_scope: formulation — the msal_uni TEXTURE-PROXY formulation only (prior measured finding, L76/#268: the proxy is at chance vs through-R reachability); NOT the margin-saliency family (the exact S_R reachability weight #268 is the built successor).

---

## D6. The bulk-class (Road) structural gap — the core question

**Every SPECIFIC seg lever in the suite is boundary/rare-class-focused; the binding residual is Road
(68% of flips, 0.092 of d_seg, flip-rate stuck at 0.40).** The GENERIC levers (directional basis, eikonal,
chroma-boundary) DO touch the Road boundary — so Road is not un-addressed — but NOTHING is Road-PRIORITIZED,
and two design choices actively point AWAY from Road:

- **Menon logit-adjust** adds `τ·log(prior)` offsets `[−1.46, −5.14, −0.70, −4.39, −1.37]` — it BOOSTS rare
  classes in the loss, i.e. it de-emphasizes the majority Road in the training gradient. Defensible for
  recall on thin classes, but it is spending loss-gradient AWAY from the 68%-binding class.
- **The entire island/persistence/ladder budget** targets classes [1,3] (≈0.3% of flip mass combined).

**From-scratch prescription:** the #1 structural priority is crushing Road-boundary jitter. Concretely
that means: (a) the directional basis MUST be oriented to the Road↔Undrivable (horizon) + Road↔Lane tangent
field, not just the lane islands; (b) seg-chroma-boundary should engage EARLY (the Road boundary is binding
from ep~50, not ep300 — a start_epoch of 300 is ~250 ep late for the binding class); (c) the powerlaw_meat
Muon gate binding-class picker should be VERIFIED to select Road (it will, if per-class telemetry is fed);
(d) consider whether the Menon offset for Road (−1.37) is under-serving the binding class relative to its
score weight. **A composition that is ALL boundary-island levers with no Road-first treatment is optimizing
the 0.3% while the 92% sits.** This is the single highest-value structural question for phase-2 item (d),
and my blind read is: v7.3 likely inherits the rare-class lever focus and does NOT have a Road-first lever —
if so, MAJOR (not blocker, because generic levers do touch Road, but the priority is inverted).

---

## D7. Off-lever queue (what a from-scratch design would toggle vs the likely incumbent)

| lever | my derivation | rationale |
|---|---|---|
| directional_basis (freq-along fix) | **ON, Road-oriented** | measured −48%, all-class, touches binding Road boundary |
| seg-chroma-boundary | ON but **EARLIER** (~ep50–100, not 300) | Road boundary binding from the start |
| Polyak finisher | ON but **event-anchored start** (D2) | fixed-epoch start biases the uniform mean |
| l7 terminal | **OFF** | measured DEFECT in a viscous flow |
| LEVER4_uniward / msal_uni | **OFF** | measured INERT (at chance) |
| Menon logit-adjust | ON (loss-only) but **audit Road offset** | de-emphasizes the 68%-binding class |
| LADDER × analytic-band × basis lane | **guard against triple-drive** (D5) | redundant/antagonistic on lane, which is 2.7e-4 |
| a ROAD-first boundary lever | **WANTED, likely MISSING** | the binding residual has no dedicated treatment |

---

## Phase-1 verdict (pre-diff)

**verdict:** the derived optimal v7 topology is CE→continuous-τ-sharpen→Muon(on powerlaw_meat event)→tail_k
at τ*→(EMA + Polyak export), l7 off, event-driven with backstop caps. This is a sound, level-set-native
shape. **THREE structural risks a from-scratch eye flags in advance of the diff:** (1) Polyak window keyed
to a fixed epoch rather than the muon EVENT [expect REVISE/MAJOR]; (2) lane serviced by three
redundant/antagonistic mechanisms while it is 2.7e-4 of d_seg [expect MAJOR on antagonism]; (3) **the
composition is boundary/rare-class-focused while Road (68% of flips) has no Road-first lever** [expect the
decisive MAJOR — item (d)].

**verdict_scope:** FORMULATION-level (the v7.3 composition/schedule as drafted); NOT a
PARADIGM/FAMILY verdict — the witness paradigm and event mode are sound and DECIDED. Diff follows in phase-2.
Pointer 0.19110 UNMOVED — MEANS.
