# ddm_lp1 — THE LANE PROGRAM (task #934)

**Operator directive 2026-08-03:** *"The Lane guard and Island stuff can work with proper engineering, we
shouldn't just give up so easily."* The directive is upheld on both legs. Nothing here is a kill. Both legs
found the cure was **built but mis-defaulted**, and both defects are in the *governance* layer, not the physics.

**Axis:** `[macOS-CPU advisory]` throughout. `score_claim=false`, `promotable=false`. Own-vehicle frontier
**S = 0.7910689 @ 353,805 B** (pu2) — **UNMOVED** by this arm. This landing is apparatus + two refutations.

STORES CONSULTED: `.omx/research/ddm_bs2_*`, `ddm_qa92_*`, `ddm_xp1_*`, `ddm_p4x_*`, `ddm_sq1_*`, `ddm_gt2_*`,
`ddm_cg1_*.json/.jsonl`, `msdf_lane_carrier_probe_*`, `CANONICAL_RESEARCH_INDEX_20260629`;
`/Volumes/VertigoDataTier/pact/ddm_b4s_20260731/window_0{1,2,3}/telemetry.jsonl`,
`ddm_qa92_20260731/qa92_verdict.json`, `ddm_xp1_20260731/xp1_verdict.json`,
`ddm_p4x_20260803/p4x_connectivity_control.json`; `src/tac/optimization/{lane_guard,ddm_lp2_e1_seeding_harness,
ddm_lp2_birth_completion}.py`, `experiments/train_tr1_partition_renderer_mlx.py`,
`src/tac/witness_dsl/curriculum_dsl.py`, `src/tac/boundary_math/island_protection.py`.

---

## §1 LEG A — the ratchet is NOT inert. My charter's premise is REFUTED. (landed `d4a4b3c541`)

My charter, inheriting bs2, said the ratchet *"has NEVER run inside a real trainer … and on the REAL burn-4
series it stays at λ=0."* I replayed **bs2's own shipped code** on **bs2's own cited telemetry** and it does not.

n = **64** `lane_guard` rows, **selection_mode = ALL rows** in windows 01–03, ep644→945 (the whole burn-4 gate
population, not a subset).

| replay | λ>0 | max λ | max g |
|---|---|---|---|
| LEGACY constant budget (**positive control**) | 0/64 | 0.0 | **−0.003452** |
| RATCHET, shipped default horizon | **3/64** | **0.119376** | **+0.000951** |

The legacy control reproduces bs2's published numbers **exactly** (max g −0.003452, min −0.053665, 64/64), so
the harness is faithful; the ratchet row is the new fact. Engagement is at gates **17–19**, *outside* the
10-gate warmup, on a real +0.002006 rise, decaying back to 0 when descent resumes.

**Second refuted claim:** bs2's §4 justified λ=0 as *"that series improved monotonically."* It does not —
**22 of 63** steps are rises, max single rise +0.005185.

### §1.1 ROOT CAUSE — an unladdered governance knob (the m51 class)

`derive_deadband_k` prices the condition *"noise alone must not move the dual by more than one
`lambda_step_cap` **over the horizon**."* That horizon is a **multiple-comparisons burden**, and a
multiple-comparisons correction must be paid against the comparisons the run **will** make. The shipped
default set horizon = gates **elapsed**, under-pricing the burden at every gate before the last (k = 1.909 at
gate 17 vs 2.270 at the true total).

| horizon | λ>0 | max λ | max g |
|---|---|---|---|
| elapsed (old default) | 3/64 | 0.119376 | +0.000951 |
| 16 | 4/64 | 0.126228 | +0.000955 |
| 32 | 2/64 | 0.045200 | +0.000372 |
| **64 (the run's true total)** | **0/64** | 0.0 | −0.000285 |
| 128 | 0/64 | 0.0 | −0.000429 |

One knob spans fully-inert ↔ engaging and **nobody derived it**. bs2's own `--help` already read *"pass the
run's planned gate count for a stationary deadband"* — the correct value was **documented in the help string
and not made the default.** That is the unladdered-knob defect exactly: we ladder constants, not control knobs.

Note also that at the true horizon the margin is **−0.000285 S = 0.20 σ**. The guard is on a knife-edge, not
in comfortable slack.

### §1.2 THE INVERSION — I corrected my own reading mid-measurement

My first draft was going to claim 0.029133 S of "real give-back the guard is blind to." That is **wrong** and I
withdraw it. Against the series' own iid-noise null (MC n=20000, σ_diff from the shipped estimator):

| | observed | null mean | null p5–p95 | percentile |
|---|---|---|---|---|
| sum of rises | **+0.029133** | +0.050097 | +0.035449 … +0.065819 | **0.7** |
| n rises | **22** | 31.5 | 25 … 38 | **0.6** |

Burn-4 contains **less** give-back than noise alone. So the 3 engagements were **FALSE POSITIVES**, and only
1 of 22 rises even reaches the deadband. **bs2's verdict (λ=0 is correct here) was right; its stated reason and
its shipped default were both wrong.** [verdict_scope: **INSTANCE** — this defect is proven on the burn-4
series and on the shipped-code path; it does not bound how the ratchet behaves on an eroding series, where
bs2's positive control still measures detection from +0.005 S.]

### §1.3 THE FIX (derived, not picked)

`derive_planned_gate_horizon(epochs, gate_every)` counts the trainer's **own** gate predicate
`(epoch+1) % gate_every == 0 or epoch == epochs-1` (= `ceil(epochs/gate_every)`, fails safe to
`N_GATES_TO_ENGAGE_DEFAULT`). The trainer resolves `--lane-guard-ratchet-horizon 0` to that total and logs
`ratchet_horizon_provenance` + `ratchet_horizon_source`. Operator override still wins.

4 tests, including a **mutation check** verified by actually deleting the wire and watching it go red. 50
lane_guard tests green (71 with e1); ruff F clean; **0 ruff errors added**. No reproducibility break — the
ratchet is default-OFF and has never run in a real trainer, so no sealed ticket or landed run is affected.

### §1.4 WHY I DID NOT SPEND THE TRAINING WINDOW THE CHARTER ASKED FOR

The charter's A/B was *"does λ ever leave 0 in a real trainer?"* That question is now **answered for $0, with a
refutation**, on real trainer telemetry. Re-buying an answered question is means-hoarding. Worse: run *before*
this fix, the A/B would have measured a guard with a known false-positive defect and attributed the difference
to the ratchet.

**Fire order (not a deferral).** The window fires when a **real** Lane rise appears — pre-registered trigger:
`inertness_alarm` clears **and** g > 0 for ≥2 consecutive gates at the derived horizon on a live run. Until a
series exhibits give-back above the deadband, both A/B arms are arithmetically identical and the window buys
nothing. The guard now **self-reports** that state every gate, so the trigger is observable, not remembered.

---

## §2 LEG B — the charter's premise is REFUTED. The composite birth path is measured NET-NEGATIVE.

My charter said: *"The physics is NOT the blocker: QA92 measured erased super-nucleus Lane components
composited onto control_tail frames … DO recover through R → uint8 → SegNet (recovery fractions O, F)."*

That cites the **numerator and omits the denominator.** QA92's own verdict (`qa92_verdict.json`, n = **600
pairs = the whole corpus**, not a subset):

| quantity | oracle (GT-RGB) | flat prototype |
|---|---|---|
| recovery fraction | O = **0.40732** | F = **0.19394** |
| recovered | +0.01706 S | +0.00812 S |
| **collateral** | **+0.31698 S** | **+0.23300 S** |
| **JOINT ΔS** | **+0.29992 S WORSE** | **+0.22487 S WORSE** |

Collateral is **18.6× the recovery.** The identity-fill control is **bit-identical** (max |Δd_seg| = 0.0), so
**100% of that collateral is receiver physics**, not a compositing artifact: the frozen SegNet's ~85 px r50 ERF
makes localized paint a *neighborhood* perturbation. Even a **perfect GT-RGB oracle at camera res with correct
AA** leaves 59% of the pool unrecovered and loses 0.3 S doing it.

Independently, sq1 (n = **32**, selection_mode = **stratified systematic** on flips-sorted order; m88 ratio
0.997329): pasting **true camera pixels** gives `eta_net` = **−3.7640**, **0/32** pairs positive, flips
amplified **4.26×** (27,055 → 115,273). And sq1 localises the loss precisely — S1 paint / S2 R·D / S3 uint8 are
**EXACT (max abs error 0.0, all 32 pairs)**, so **AA coverage, camera-res placement, and the uint8 amplitude
floor are all measured NON-BINDING**, and 100% of the debt is S4, the frozen net's regional response.

**So building the seed/amplify path as compositing would have been building a measured-dead thing.** That is
the "built-instead-of-paid" poison inverted, and the charter would have funded it. [verdict_scope:
**FORMULATION** — this kills *compositing/painting* pixels into erased components as the birth mechanism on
this vehicle. It does **not** kill Lane birth, and §2.1 is the measured escape.]

### §2.1 THE ESCAPE IS MEASURED, AND IT IS THE OPPOSITE OF WHAT THE CHARTER SPECIFIED

The same sq1 probe measures the cure: **margin-optimal prototype colours SOLVED from the frozen head** move the
**identical band, addresses and bytes** from **−3.7640 → +0.7895**. A sign flip from the same geometry. This is
memory m95 exactly: *pixel TRUTH −3.764 vs pixel SOLVED +0.7895 ⇒ CONTENT-vs-SOLVE, not granularity.*

And e1's docstring already states the structural property to carry: it **re-renders from seeds rather than
compositing** — which is why it stays on the manifold.

**Corrected birth-path spec (supersedes the charter's "seed/amplify composite"):**

1. **Address** — erased super-nucleus components (`>5 px`, `<50%` Lane-classified), the p4x/QA91 grammar.
2. **Seed** — set the *token/renderer* state, never the pixels; `eased_island_masks` is the named, **verified
   importable** integration point (signature confirmed; plain module, no trainer port needed).
3. **Colour by SOLVE, never by truth** — prototypes solved against the frozen head (the S4 cure), not GT RGB.
4. **Re-render, never composite** — the e1 property; the render is what keeps it on-manifold.
5. **Price jointly** — recovery **and** collateral in one ΔS, against the live best. QA92's failure was
   invisible to a recovery-only read.

### §2.2 A THIRD CORRECTION — to my own charter's ceiling

My charter said the honest Lane ANNIHILATE ceiling is *"0.044175 S = 7.137% of gap."* At source
(`p4x_connectivity_control.json`) **0.044175 S is Lane + Movable combined.** Lane alone, 8-connected, is
**0.037276 S = 6.022% of gap** (gap denominator 0.6189279). The Lane-only target is ~16% smaller than the
charter states, and p4x's own §7 notes it is a 100%-capture bound that *"bounds the row, it does not predict
it."*

### §2.3 WHAT IS ALREADY BUILT AND HAS NEVER FIRED (the operator's premise, confirmed)

| surface | state | wired to |
|---|---|---|
| `--existence-hinge-weight` (p4x #920) | **built in TR1**, 6 config fields, default 0.0 = OFF, 31 tests | TR1 — **never fired**, no scorer A/B exists |
| `eased_island_masks` | built, **verified importable** into TR1 | nothing in TR1 |
| `SeedIslandEased` / `SeedIslandBirth` (#323) | built DSL levers | flags exist **only** in the RETIRED levelset trainer, **0×** in TR1 |
| `ddm_lp2_e1_seeding_harness` | built, oracle+solver **injected** (stub pattern) | tests only — no trainer |
| `birth_completion` gate key (#802) | built **and wired** | `supervise_ddm_{r1c,b4s}` — the one live one |

Island/birth flags in TR1 = **0 occurrences**, independently re-verified. The operator is right that nothing
was refuted: every one of these is a built-but-never-fired cure, and the top row is a **live, already-wired
objective term that needs no new code to test.**

---

## §3 DISPOSITIONS — every row exits owned

| # | item | disposition |
|---|---|---|
| 1 | Ratchet horizon defect | **FIRED** — fixed, tested, mutation-checked, landed `d4a4b3c541` |
| 2 | Leg-A training window | **QUEUED-WITH-FIRE-ORDER** — trigger in §1.4; guard self-reports the trigger |
| 3 | Composite seed/amplify path | **FOLDED as a spec** (§2.1), not built — measured net-negative as specified |
| 4 | Existence hinge never fired | **QUEUED-WITH-FIRE-ORDER**, owner `ddm_lp1` successor — highest-value Leg-B row: built, wired, 31 tests, needs only a bounded A/B at `--existence-hinge-weight > 0` |
| 5 | `eased_island_masks` → TR1 | **QUEUED** behind #4 — seeding is worthless until the objective term is measured |
| 6 | `cg1r` "~2.5 px" | **FOLDED as prose-level** — no JSON field backs it; do not cite as receipt-grade |
| 7 | `cg1r` memo filename mismatch | **QUEUED** — cited as `ddm_cg1r_*`, resolves to `ddm_cg1_*.jsonl` |
| 8 | ±1.0 token clip literal (#933) | **NOT FOLDED** — my Leg-B wiring never touched the token path; left owned by #933 |

## §4 WHAT I REFUTE IN MY OWN CHARTER

1. *"on the REAL burn-4 series it stays at λ=0"* — **false** at the shipped default (3/64). §1.
2. *"that series improved monotonically"* (bs2 §4) — **false**, 22/63 rises. §1.
3. *"The physics is NOT the blocker … components DO recover"* — **false as stated**; recovery is real but
   collateral is 18.6× larger and the joint move is +0.29992 S **worse**. §2.
4. *"honest ANNIHILATE ceiling 0.044175 S = 7.137% of gap"* for Lane — that is **Lane+Movable**; Lane alone is
   **0.037276 S = 6.022%**. §2.2.
5. *"cheapest decisive fire"* = the training window — **false**; the decisive fire was a $0 replay, and running
   the window first would have measured a defective guard.

Three of the five are cases where a **numerator was quoted without its denominator** — the same shape as
`a_delta_without_its_baseline_is_unanchored`. Recovery without collateral, engagement without the null,
Lane-ceiling without the class split.
