# ddm_pt2 — porting old flags and levers to the TR1 frontier

**Date:** 2026-08-03 · **Arm:** `ddm_pt2` · **Axis:** apparatus + reachability. **NOT a score claim.**
**Scorer-free, $0** (`ddm_pu2` holds the scorer slot; no scorer forward was performed).
**Pointer UNMOVED:** `0.1910828242` [contest-CPU custody]. Own-vehicle live best `0.7910689` unmoved.

Operator directive 2026-08-03 verbatim: *"We can port old flags and levers to our latest frontier."*

Predecessor: `.omx/research/ddm_lr2_lever_instrument_repair_20260803.md` (`7f6301ed20`) — its named
highest-value owed row (§1 below) is paid here. Consumed, not re-run: `ddm_fh1` (#805 force
adaptation), `ddm_vh1` (#798), `ddm_sb2` (#819), `ddm_gd1` (#817), `ddm_rz1`/`ddm_ra1` (TR1 has no
SDF / level-set field / coverage integral / continuous-geometry rasterization), `ddm_tp2`
(2026-08-02, the port precedent this arm follows).

---

## ANSWER FIRST

**The TR1 trainer imports the SAME `make_loss_fn` the RETIRED levelset trainer uses, and was
passing 5 of its 18 parameters. Four fully-implemented seg forces were therefore already sitting
inside TR1's own loss graph, unreachable for exactly one reason: TR1's argparse never declared
their flags.** That is the port. Nothing about the forces was re-implemented — which is what makes
them class (a) and not a rename.

Landed, all $0, all with executed controls:

1. **§1 — the factory↔instance JOIN** (the lr2 prerequisite). **41 of 43** live instance names now
   resolve to a UNIQUE factory, **0 ambiguous**, yielding **29 distinct factories** the live
   vehicle actually launched — the activation history the ledger structurally could not give. The
   2 that do not resolve are **real config-orphans**, and the join surfaces them instead of hiding
   them under a fallback.
2. **§2 — the PORT.** Six flags threaded onto TR1 (`--seg-focal-gamma`, `--fisher-density-weight`,
   `--fisher-density-source`, `--head-natural-grad`, `--head-natural-grad-eps`,
   `--tau-softplus-tau`) + **4 DSL `Lever` factories** in a module that declares its trainer.
   Args-only ⇒ `config_hash` flag-invariant ⇒ the sealed lineage is byte-untouched.
3. **§3 — behavioural proof on REAL cached SegNet logits + REAL GT.** **7 of 7** checks pass,
   including a **mutation control that fails as required**, and one leg **MEASURED VACUOUS** and
   reported as such rather than scored.
4. **§4 — the a/b/c classification** of the retired lever population, with its denominators.

**The scope correction I owe up front, because it bounds everything else.** MEASURED: the RETIRED
trainers declare **478** flags, TR1 declares **73**, and they share **15**. Of the **141**
retired-bound lever factories, exactly **3** have all their flags already on TR1 — and two of those
three are `--ema-decay`. **127 have none.** So "port the old levers" is not a bulk-migration job:
the overwhelming majority of the old surface was built for a vehicle whose architecture TR1 does
not have. The value is in the small set where the MECHANISM genuinely transfers, and the four here
transfer because they are literally the same function call.

**Honest bound on what this is worth.** These four are now REACHABLE and RACEABLE. **None has been
raced.** No row here claims a d_seg effect, and the pointer did not move. Seg is 64.9% of the
remaining gap and this arm made four seg-force actuators available to it; converting that into a
lower exact score is the next unit's job, not a result of this one.

---

## §1 — THE PREREQUISITE: the factory↔instance name join

### 1.1 Why it had to come first

`ddm_lr2` §2.4 MEASURED that the ledger keys on **factory** names (`lever_token_grid`) while a
sealed ticket records the constructed **instance** name (`tr1_token_grid_D16_c4`), built by
f-strings parameterised by the factory's own arguments — **0 of 43 joinable**. Wiring
`record_activation` into `tools/launch_tr1_run.py` first would pile rows under keys nothing joins
to and leave `never_fired()` reporting every factory as never-fired forever: a second cross-store
contradiction wearing the costume of a fix. lr2 declined to fake it; this pays it.

### 1.2 Mechanism (static; no imports, no execution)

`src/tac/witness_dsl/lever_name_join.py` AST-scans every factory for the `name=` argument of every
`Lever(...)` it constructs. MEASURED shapes in the live tree:

| shape | count | handling |
|---|---:|---|
| `ast.Constant` | **151** | LITERAL name |
| `ast.JoinedStr` | **22** | TEMPLATE: interpolations → `(?:.+)`, literals `re.escape`d |
| `f"...".rstrip("0")` | (in the 22) | **unwrapped** to the JoinedStr — without this `lever_ema_decay`'s two real launched names do not resolve (MEASURED before/after) |
| neither | **4 factories** | statically unjoinable; ENUMERATED by `dynamic_name_factories()` so the denominator is visible |

**Literal-first precedence is load-bearing, not cosmetic.** `lever_seg_physics` emits
`f"tr1_seg_{form_start}"`, whose regex also matches the LITERAL `tr1_seg_margin_weight` from
`lever_seg_margin_weight`. Literals resolve first; residual overlap is reported `AMBIGUOUS`, never
collapsed to a first match.

### 1.3 MEASURED result — the live vehicle's activation history

```
receipts 32 · tickets_read 32 · instances_seen 43
instances_joined_to_factory 41 · ambiguous 0 · unresolved 2
distinct_factories_launched 29
roots_scanned [/Volumes/VertigoDataTier/pact, experiments/results]
roots_unavailable [/Volumes/APDataStore/pact]
```

`roots_unavailable` is reported, not assumed empty — an unmounted SSD must never be
indistinguishable from a clean tree. The counts are therefore a **lower bound** on launch history.

### 1.4 The two unresolved names are a FINDING, not a join failure

| instance | provenance | verdict |
|---|---|---|
| `tr1_coupling_field_only` | built by a bare `Lever(...)` at `experiments/ddm_pa1r_seal_and_tickets.py:111` | **config-orphan**: a hand-constructed lever that bypassed the DSL and was sealed into a live ticket — exactly what "the DSL HOLDS every designed lever" forbids |
| `qa86_live_config_pin` | `git grep` over `src tools experiments` finds **no** construction site | **provenance absent from the repo**; scope of that negative is stated, not universal |

A resolver with a fallback would have swallowed both. Surfacing them is the point.

### 1.5 Controls — EXECUTED

* **Positive round-trip** (6 factories incl. the `.rstrip` shape): construct a real `Lever`,
  resolve its name, land back on ITS factory. Fails on a template bug, a precedence bug, or a
  missing unwrap.
* **Negative**: `zz_definitely_not_a_lever`, `""`, and a bare FACTORY name all resolve to
  `UNRESOLVED_NOT_A_FACTORY`.
* **Ambiguity**: synthetic overlapping forms return `AMBIGUOUS` with BOTH refs — asserted on
  constructed inputs so the test is about the resolver, not about today's tree happening to be clean.
* **Package-wide self-resolution census, with its denominator**: **151** literal names, **147**
  self-resolve uniquely. The 4 failing rows are **2 NAMES emitted by two different config-compilers
  each** (`c2_component_wallclock_telemetry`, `c2_speed_stack`, from both
  `compile_c2_surgical_warm_launch_config` and `compile_v9c3_duty_ab_config`) — a genuine
  duplicate-name defect in the retired tree, correctly reported AMBIGUOUS. The exact allowlist is
  pinned so a NEW collision fails the test.

### 1.6 The limitation I did not paper over

Template holes are `.+` (they must be: real instance names carry underscores INSIDE a hole —
`tr1_token_temporal_shared_base`, `tr1_token_init_solve_project`). A trailing hole is therefore
unanchored, so a **synthetic name no factory emits** (`tr1_token_grid_D16_c4_NOPE_`) still matches
`lever_token_grid`. Operationally the join only ever sees names that WERE emitted, and both real
orphans in §1.4 correctly failed to match — but the over-match is real and is recorded here rather
than left for someone to find. **`record_activation` is NOT wired into the live launcher by this
arm**; the join is the prerequisite, and wiring + backfilling the 32 receipts is the next row.

---

## §2 — THE PORT

### 2.1 The measured opening

TR1 (`experiments/train_tr1_partition_renderer_mlx.py:2333`) calls `make_loss_fn` — imported from
`experiments/train_witness_realized_through_R_mlx.py`, the RETIRED BASE trainer. It passed
`adapter, SEG_H, SEG_W, score_domain, seg_loss, margin_weighted, margin_weight_temp, render_fn`:
**5 of 18 parameters**. The unpassed ones include four complete, documented, default-off,
byte-identical-when-off seg forces.

### 2.2 Why these four and not `margin_weight_fn` — the check that decides a real port from a dead flag

Focal and the Fisher density fold into **`seg_pixel_w`**; natural-gradient rewrites **`seg_logits`**.
Both surfaces are read by **every** seg-loss branch before the mean. By contrast `margin_weight_fn`
is read only under `apply_mw`, and the `tau_softplus` branch contains **no `apply_mw` guard at all**
— and `tau_softplus` is the form the live burn lineage occupied for **~100% of its epochs**
(MEASURED, `ddm_tp2` row 2: b4s window_01..03 + r1c window_01; `MARGIN_WEIGHTED_HONORING_SEG_FORMS`).
**Porting `margin_weight_fn` would have shipped a flag that cannot act on the vehicle we ship** —
the exact genus this trainer already refuses. It is classified (b), not (a), below.

### 2.3 What landed

| flag | old lever ported | surface it acts on | default |
|---|---|---|---|
| `--seg-focal-gamma` | `SegFocalGamma` | `seg_pixel_w` (all forms) | `0.0` = OFF |
| `--fisher-density-weight` / `--fisher-density-source` | `FisherDensityWeight` | `seg_pixel_w` (all forms) | `0.0` / `model` |
| `--head-natural-grad` / `--head-natural-grad-eps` | `HeadNaturalGradient` | `seg_logits` transform (all forms) | `off` / `1e-3` |
| `--tau-softplus-tau` | the live seg form's own scalar | `tau_softplus` branch | `0.3` (== the inherited default) |

DSL factories (`src/tac/witness_dsl/pt2_ported_levers_20260803.py`, `TRAINER_RELPATH` declared per
lr2 §1): `lever_seg_focal_gamma`, `lever_fisher_density`, `lever_head_natural_grad`,
`lever_tau_softplus_tau`. Each is default-OFF and score-affecting, so each is a **duty-to-measure
row** the moment the module exists — `package_known_levers()` AST-scans the package, so they enter
`never_fired()` / `duty_to_measure()` automatically. **No parallel registry.** Census moved
**141/39 → 141/43** (total 184); `verdict_relevant_undeclared` stays **0**.

### 2.4 Two design decisions, with their costs stated

**(a) Args-only, never `TR1Config`.** `canonical_json()` is `asdict(self)`, so a new config field
moves `config_hash` for EVERY run including the off case, breaking the sealed lineage's
flag-invariance. This follows the `--persist-optimizer-state` / `--telemetry-v9-port` precedent set
in this same file. **The cost, stated not hidden:** two runs differing only in these flags share a
`config_hash`. Mitigation: a `ported_loss_forces` telemetry row is emitted **unconditionally**
(score-neutral observability is not gate-able) carrying every value plus an explicit `active` list,
so the run record distinguishes them.

**(b) A fail-closed gate guard.** `assert_ported_force_scalars_have_their_gate` refuses
`--fisher-density-source` without `--fisher-density-weight > 0`, and `--head-natural-grad-eps`
without `--head-natural-grad on` — the same silent-no-op genus this file already refuses for
`--margin-weighted-loss` and `--seg-spike-downweight`. Both directions are unit-tested.

### 2.5 `tau_softplus_tau` is the row I did not expect to find

The live vehicle runs `tau_softplus` for ~100% of its epochs, and **the one scalar shaping that
loss was unreachable** — TR1 took `make_loss_fn`'s default `0.3` with no way to set it. That is a
borrowed constant governing the live objective, never derived for this vehicle. The lever's rung
says so (`INHERITED-DEFAULT-BEING-PROMOTED-TO-RACED`); §3 shows it has a real, monotone geometric
effect, so the sweep is a genuine duty-to-measure slot rather than a formality.

---

## §3 — BEHAVIOURAL PROOF (real inputs, executed)

`experiments/ddm_pt2_ported_force_behavioural_proof.py` →
`reports/ddm_pt2/behavioural_proof.json`. Inputs are a **REAL cached SegNet logit field**
(`/Volumes/VertigoDataTier/pact/ddm_b2b_qa75_field_20260730/pair-000000.npz`, `(5,384,512)` f16)
and the **REAL cached GT** argmax + margin (`experiments/results/mlx_fleet_gt_cache/gt_n96.npz`).
**No scorer forward is performed.** Each force is checked against a PREDICTED closed-form
consequence, not merely "the number moved" — a wrong implementation also moves it.

| force | prediction (falsifiable) | MEASURED | verdict |
|---|---|---:|---|
| focal | stop-grad + mean-1 ⇒ weight ratio is EXACTLY `((1-p1)/(1-p2))^γ` | rel err **1.73e-05** (pred 5.828168 / obs 5.828067) | **HOLDS** |
| focal — **mutation control** | replace the map with all-ones ⇒ the SAME assertion must FAIL | rel err **0.828** ≫ 1e-4 | **FAILS AS REQUIRED** |
| Fisher density | `w == (1/2)sech²(m/2)` renormalized, on the real GT margin, at λ=1 `source=gt` | rel err **4.57e-07** | **EXACT LAW HOLDS** |
| Fisher vs focal (real leg) | both up-weight the small-margin separatrix ⇒ positive rank correlation | Spearman **+0.9157**, n=28,087 | **HOLDS** |
| natural grad | forward EXACTLY identity **and** backward CHANGED (both required) | fwd max-abs-diff **0.0**; grad max-abs-diff **6.10e-06**; cosine **0.99360** | **HOLDS** |
| τ | smaller τ ⇒ MORE gradient mass in the separatrix band (\|m\|<1) — monotone | share **1.000 → 0.9986 → 0.9584 → 0.7782 → 0.5007** over τ = 0.05…1.0 | **MONOTONE** |

**7 of 7 checks pass.** The `note` in each row records that forward-identity alone would be
satisfied by a no-op, and that a changed backward alone would not be the claimed preconditioner —
both legs are required precisely so the check can fail.

### 3.1 The leg I had to correct — MEASURED VACUOUS, reported as such

My first draft asserted the documented focal-vs-Fisher **disagreement** on confidently-wrong pixels
(low `p_y` ∧ large `|m|`). It returned `n = 0`. That is not a failure and not a pass — it is a
**vacuous scope**, the genus that surfaced seven times across this campaign today.

MEASURED denominator on this field: **0 of 196,608** pixels qualify; **min `p_y` over the whole
frame is 0.3377**; of the **189** argmax-wrong pixels (0.0961%) the **median `|margin|` is 0.043**.
The b2b field is a near-perfect solve (d_seg ~1.52e-4 by construction), so **every error in it is
marginal and the disagreement regime is absent from the input**. The claim is therefore split: the
REAL leg (agreement in the regime that exists, ρ=+0.9157) is measured; the REGIME leg is reported
`VACUOUS` with its reason and its denominator, plus a labelled FUNCTION-DOMAIN evaluation of the two
closed forms showing where they diverge. **Settling the composition question needs a real
mid-training TR1 logit field — a scorer pass this arm does not hold. Named as owed, not implied.**

---

## §4 — THE a/b/c CLASSIFICATION

### 4.1 Hard measurements (these are the load-bearing numbers)

| quantity | value | route |
|---|---:|---|
| flags declared by the RETIRED trainers (levelset + base) | **478** | AST `add_argument` scan |
| flags declared by the LIVE TR1 trainer (post-port) | **73** | same scan |
| intersection | **15** | set ∩ |
| retired-only (no TR1 surface) | **463** | set − |
| retired-bound lever factories | **141** | `package_lever_factories()` |
| …with ALL flags already on TR1 | **3** | 2× `--ema-decay` + `SegSpikeReweight` (tp2's completed port) |
| …with SOME | **2** | both are whole-config compilers, not levers |
| …with NONE | **127** | |
| …with no flags at all | **9** | |

### 4.2 The three classes

**(a) PORTS CLEANLY — same mechanism, TR1 has the surface. LANDED: 4.**

| lever | TR1 form | behavioural proof | registry row |
|---|---|---|---|
| `SegFocalGamma` | `--seg-focal-gamma` → `focal_gamma` | closed-form ratio, rel err 1.73e-05 (+ mutation control) | `lever_seg_focal_gamma`, default-OFF, never-fired |
| `FisherDensityWeight` | `--fisher-density-weight` / `--fisher-density-source` | exact sech² law, rel err 4.57e-07 | `lever_fisher_density`, default-OFF, never-fired |
| `HeadNaturalGradient` | `--head-natural-grad` / `-eps` | fwd diff 0.0 ∧ grad diff 6.10e-06 | `lever_head_natural_grad`, default-OFF, never-fired |
| (`tau_softplus` scalar) | `--tau-softplus-tau` | separatrix share monotone 1.000→0.5007 | `lever_tau_softplus_tau`, default = inherited 0.3 |

Already ported before this arm, recorded for completeness: `SegSpikeReweight` (`ddm_tp2` row 3,
2026-08-02) and `EmaDecayCalibrated`/`DerivedEmaDecay` (`--ema-decay`).

**(b) NEEDS RE-DERIVATION — intent survives, mechanism assumed retired structure.**
The named, already-done work is `ddm_fh1` (#805): **5** v8/v9/v10 forces adapted to TR1 and landed
as DESIGNED-STUB `Lever` factories with per-force adapted forms and falsifiers
(`TieLocusEdgeWeighted`, `MarginSatisficeCap`, `XiAdvectedTokenBase`, `BirthPlateauKneeConjunct`,
`ErfBirthContextCoadapt`), plus `ph3_s10` ×2 and `ax1` ×1. **I did not re-derive these** — the
charter said consume them, and their debt is a TRAINER BUILD, not a DSL gap: MEASURED, none of
their 8 flags exists on either trainer. `lr2` re-homed them to TR1 so the debt is now correctly
attributed; building them is the owner's job, gated by the fh1 falsifiers.
Also class (b), and the one I explicitly declined: **`margin_weight_fn`** — the flag would compile
but is INERT on the live seg form (§2.2). Porting it needs a form that honours `apply_mw`, i.e. a
loss-form change, which is a raced decision and not a port.

**(c) STRUCTURALLY INAPPLICABLE — no TR1 surface exists and none should be invented.**
`verdict_scope: FAMILY — the named missing surface`, per `ddm_rz1`/`ddm_ra1`: TR1 is
`tokens → CNN → np.repeat×2 → sigmoid×255`, with no SDF, no level-set field, no coverage integral,
no continuous-geometry rasterization, no coordinate-INR, no Fourier/curvelet/shearlet bank, no
FiLM, no Muon, and no pose in the training loop (`compute_pose=False`; there is no `--w-pose`).

*The grouping below is a **regex heuristic over flag names, first-match-wins** — an exposition aid,
not a measured taxonomy. The measured facts are §4.1; the per-factory assignment is reproducible
from the script in this memo's commit.*

| family (heuristic) | n | missing TR1 surface |
|---|---:|---|
| optimizer / schedule / stage-transition (Muon, rewarmup, Polyak, tail, lr) | 27 | multi-stage curriculum + Muon; TR1 has one Adam and a `ce→tau_softplus` switch |
| basis / INR (Fourier, curvelet, shearlet, SIREN, FreSh, self-orient, FiLM) | 24 | a coordinate-INR with a frequency bank; TR1 has a CNN over a token grid |
| level-set / eikonal / SDF / geometry rasterization / topology | 22 | a continuous field to take an eikonal of, and a rasterizer |
| curriculum / event graph | 20 | the 8-stage event graph |
| verdict / telemetry / compute plumbing | 17 | mostly a different verdict harness |
| pose (training loop) | 12 | pose is not in TR1's loop at all |
| seg per-pixel / per-logit force | 7 | **this is where the (a) rows came from** |
| unclassified by the heuristic | 3 | = the 3 already-ported rows, a consistency check |
| **total (retired-bound, ≥1 flag)** | **132** | |

**(c) is a first-class result.** The honest headline of §4 is that ~96% of the old flag surface
does not transfer, and saying so is worth more than forty asserted ports.

---

## §5 — ADVERSARIAL SELF-REVIEW (round 1; findings reset the counter)

**Counter: 1 clean pass. Round 1 produced a finding and reset it. This memo is PROVISIONAL on the
review axis** and says so rather than claiming a seal.

### Round 1 — FINDING (counter → 0), caught by my own test
My first draft named the focal lever `tr1_seg_focal_gamma{γ}`, which the round-trip test resolved
as **AMBIGUOUS** against `lever_seg_physics`'s `f"tr1_seg_{form}"` family. I had landed a new lever
inside another factory's name namespace. **Fix:** renamed to `tr1_focal_gamma{γ}`; the collision
comment is in the factory; the round-trip test is the standing guard; and I added the package-wide
self-resolution census (§1.5) which turned up the **2 pre-existing duplicate names** as a bonus.
*The join caught the join's own author. That is the instrument working.*

### Round 1 — SECOND FINDING: my own control was vacuous
The focal-vs-Fisher disagreement check returned `n=0` and my code scored it `false`. Reporting a
vacuous scope as a failure is the same error as reporting it as a pass. **Fix:** §3.1 — split into
a non-vacuous REAL leg and a REGIME leg declared VACUOUS with its denominator and reason.

### Round 2 — CLEAN (1 of 3)
Re-derived every load-bearing number from primaries. `ruff --select F` clean on all 5 touched
files. `28/28` of this arm's tests pass with **no skips** (the trainer-guard test originally
SKIPPED on an import failure — a control that does not run — and was rewritten to compile only the
guard's own function def, so it always executes).

### Test state, with its denominator
**28/28** of this arm's tests. **92/93** across `test_lever_registry.py` +
`test_build_completeness_grades.py` + `test_ddm_lr2_lever_instrument_repair.py`. The one failure is
**PRE-EXISTING and proven not mine**: `test_332_coverage_rose_from_deorphaning` asserts `stale == []`
and `stale` is the 3 `--integer-plane-emitter-*` flags. **My diff contains zero occurrences of that
string** (`git diff | grep -c` → 0), it is the same 3 flags `ddm_lr2` §6 recorded and proved at
HEAD, and `ddm_la1` independently measured `stale=3` before either arm ran.

### Assumption-challenge

| # | assumption | status | if violated |
|---|---|---|---|
| 1 | TR1 really calls the same `make_loss_fn` | **VERIFIED_VIA_SOURCE_INSPECTION** (single call site, import traced, AST-asserted in a test) | the whole (a) class collapses — this is the load-bearing one |
| 2 | the forces are byte-identical when off | **VERIFIED_VIA_SOURCE_INSPECTION** (each is inside a `> 0.0` / `is not None` guard; defaults reproduce the pre-port call exactly) | the port is not free and the sealed lineage is disturbed |
| 3 | the cached b2b logits are representative of what the live loss sees | **REFUTED IN PART, and I say so** — MEASURED min `p_y` 0.3377 over **0 of 196,608** confidently-wrong pixels. **This is a VACUITY finding with its denominator stated, NOT a magnitude dismissal:** the confidently-wrong population on this field is *empty*, so the composition leg has **no support to measure on**, and no claim about its size is made in either direction. It is a SOLVE field, not a mid-training field. The closed-form checks are input-independent; the *regime* conclusions are not, which is exactly why §3.1 declares its leg vacuous. **The unpriced quantity is named, not dismissed:** allocator COMPOSITION on a mid-training field is **UNMEASURED**, and its relative significance is unknown against the live denominator (gap to the PR130 bar **0.6189279**; **1% = 0.0061893 S = 7,301 flips**; seg leg **0.4015190 = 64.9%** of that gap). Resolving measurement: re-run §3.1 on a mid-training logit field, where a non-empty confidently-wrong population exists. Verdict scope: **INSTANCE (this cached field)** | conclusions about allocator COMPOSITION do not transfer; the closed-form proofs do |
| 4 | args-only threading is the right byte-identity tradeoff | **VERIFIED_VIA_SOURCE_INSPECTION** + precedent (`ddm_op2`, same file, same day) | `config_hash` would move for every run; the telemetry row is the stated mitigation |
| 5 | the join's `.+` hole is safe in practice | **PARTIAL** — over-match on synthetic never-emitted names is MEASURED and recorded (§1.6); both real orphans correctly failed to match | a mis-attributed activation row; bounded by the ambiguity report, not eliminated |
| 6 | these four levers will lower d_seg | **ASSUMED_AWAITING_VERIFICATION — and NOT claimed anywhere** | nothing in this memo depends on it; they are reachable, not proven useful |

**The strongest challenge I cannot close:** is this on the critical path to a lower exact score?
**Not directly.** It is reachability. Its claim is that four seg-force actuators that were already
inside the live loss graph could not be turned on, on the axis that is 64.9% of the remaining gap —
and now can be, with provenance, falsifiers, and duty-to-measure rows. **The pointer is UNMOVED at
`0.1910828242`.**

---

## NEXT-IF-RESUMED

1. **RACE the four.** Each has a pre-registered falsifier in its factory docstring. Highest prior:
   `--tau-softplus-tau` (it governs ~100% of the live lineage's epochs and has never been swept on
   this vehicle) and `--seg-focal-gamma` (concentration, against `ddm_pc2`'s measured ~0.058%
   interior flip share). `--head-natural-grad` carries an own-optimum caveat: re-tune lr per arm or
   a null result is measuring lr, not the metric.
2. **Wire `record_activation` into `tools/launch_tr1_run.py` THROUGH the join**, then backfill the
   32 receipts. The join now exists; lr2's ordering constraint is satisfied.
3. **The composition question** (focal × Fisher on confidently-wrong pixels) needs a real
   mid-training TR1 logit field. Owed, named, scorer-gated.
4. **Two config-orphans to adjudicate:** `tr1_coupling_field_only` (inline `Lever` at
   `experiments/ddm_pa1r_seal_and_tickets.py:111`) and `qa86_live_config_pin` (no repo provenance).
5. **Two duplicate lever names** (`c2_component_wallclock_telemetry`, `c2_speed_stack`) emitted by
   two compilers each — retired-tree hygiene, recorded not touched.
6. **Un-taken, named rather than implied:** I did not build any of the 8 fh1/ph3/ax1 DESIGNED-STUBs
   (that is a trainer BUILD, gated by their own falsifiers); I did not sweep the 127 no-surface
   factories individually beyond the family grouping; I did not port `margin_weight_fn` and §2.2
   says why; I did not re-open `ddm_lr2`'s or `ddm_la1`'s closed rows.
