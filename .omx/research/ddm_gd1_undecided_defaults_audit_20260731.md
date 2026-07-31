---
schema: ddm_gd1_undecided_defaults_audit.v1
date_utc: 2026-07-31
arm: ddm_gd1_undecided_defaults (task #817, Opus 5)
lane_id: "lane_ddm_gd1_undecided_defaults_20260731"
research_only: true
score_claim: false
promotion_eligible: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
axis: "[macOS-CPU advisory] — every measurement here is scorer-FREE (frozen GT cache + pure
  arithmetic); NO SegNet/PoseNet forward, NO MLX, NO Metal, NO paid dispatch, $0"
consumes: [ddm_gc14 (the five instances + the boundary-step verdict), ddm_gc15 #816 (the reset
  operator + the DEPENDENCY-default subclass — owns rows 4/5), ddm_dg1/us1 (#812 rate
  denominator, catalog #407), ddm_gd1 generic-default census (QA82, the SISTER axis — objects),
  train_tr1_partition_renderer_mlx.py @ e922da7a92, train_witness_realized_through_R_mlx.py,
  lever_registry/activation_ledger/costate_digest/confound_gates/v9_provenance_gates/lawref,
  ema_decay_run_geometry_v1 + its p0 ledger row, arbitrariness_extinction_audit_20260518.jsonl]
consumers: [MAIN (burn-4 window boundary + burn-2 config window), the costate organ duty queue,
  whoever lands the Decision Provenance Register]
tokens: [p0-ledger-ok]
---

# ddm_gd1 — undecided defaults: the load-bearing CHOICES nobody made

## §0 POINTER HONESTY FIRST

**The exact frontier did NOT move. `0.1910828242 [contest-CPU]` is UNMOVED.** This unit is an
AUDIT plus one scorer-free measured fix-design: MEANS, not the end. Every number is
`[macOS-CPU advisory]`, `score_claim=false`. Labels on every claim: **MEASURED** (artifact
attached) / **DERIVED** (algebra shown) / **INFERRED** / **ASSUMED**.

## §1 METHOD + MULTI-PASS RECALL (what each round actually recalled)

**Round 1 (memory index + research listing).** Surfaced the single most important fact of this
unit: **a sibling arm already ran today** — `.omx/research/ddm_gd1_generic_default_census_20260731.md`
(QA82, Fable, 34 KB, sealed at round 3). It swept the *generic-default* class: textbook OBJECTS,
STRUCTURES and OPERATORS in the live path (activation, squash, quantizer, padding, upsample, warp
kernel, scan order, coder context, pose chart). It executed one race (Hilbert scan order → raster
WINS, the law's first surviving default). **I did not re-derive one row of it.** Rows T7 (optimizer),
T8 (margin temp), T18 (pair order), T19 (w_rate) and S12 (gate-set geometry) are directly relevant to
my charter and are cited, not re-minted.

**Round 2 (gc14 + source).** Grepped `ddm_gc14_first_descent_20260731.md` for the five seeded
instances and traced each to code: `resolve_gate_ids` / `realized_gate` (trainer 899–966),
`derive_ema_decay` + `total_updates` (342–380, 1372), `--gate-every` (1216, 1898),
`_torch_R_to_camera_uint8` (witness 1611), `full_confirm` (2118–2150). gc14's own new-law line is
the seed of this memo: *"constants-are-poison applies to an estimator's WINDOW LENGTH and REFERENCE
FRAME, not only to its threshold."*

**Round 3 (apparatus, delegated fan-out).** Traced `ema_decay_run_geometry_v1` end-to-end
(law module → `eval_ema_decay_run_geometry` → `EmaDecayCalibrated` Lever → the bijection gate) and
inventoried the CHOICE-tracking surface. **Recalled a prior orphan:**
`.omx/state/arbitrariness_extinction_audit_20260518.jsonl` — 52 rows of "is this value arbitrary and
what retires it", dated 2026-05-18, **wired into no gate and no consumer**. That is the failure mode
my §7 proposal must not repeat, and it is why §7 leads with the gate and the duty row rather than
with the schema.

**Scope honesty.** This is an AUDIT, sealed at one round of self-attack (§4.4 records what that round
caught in my own code). It is not claimed exhaustive; §3 names what I did not reach.

## §2 THE CLASS — what an "undecided default" is, and why the ladder cannot see it

The value-provenance ladder types **VALUES**: every semantic flag must resolve to a LawRef with one
of four rungs (`measured_anchor | derived_at_config | derived_live | hardcoded_waiver`,
`v9_provenance_gates.py:66`). It is a real, well-built surface. It structurally **cannot reach** the
class this unit sweeps, for three reasons — each verified, not asserted:

1. **A choice usually has no literal.** It is expressed as CODE:
   `np.mean(dsegs)` (trainer:921), `np.random.default_rng(0)` (trainer:963),
   `(epoch+1) % cfg.gate_every == 0` (trainer:1898), `torch.round(up)` (witness:1623),
   `optimizer = optim.Adam(...)` reconstructed unconditionally at every resume (trainer:1543).
   The bijection gate walks flags↔Levers; **a choice with no flag is invisible to it by construction.**
2. **A formula's certificate says nothing about the binding of its free variables.** `ema_decay` IS
   derived — `d = 1 − 2/(φU)`, LawRef-resolved, two validated anchors. Nobody ever decided **what
   `U` means under windowed execution**. The trainer binds it to `args.epochs`, the *cumulative*
   target passed at each resume, so `U` grows every window and the shadow time-constant lengthens
   166 → 202 → 236 epochs (MEASURED, gc14). **A DERIVED constant can still be undecided.**
3. **The ladder cannot express an ESTIMAND mismatch.** No rung token can say "this 36-pair mean is
   being reported as a stand-in for a 600-pair mean." The harm is not a wrong value; it is a
   *different quantity than the one the decision needs*. Two live rows already prove the gate
   rubber-stamps choices: `--seed-anneal-shape: "linear"` and `--containment-mode: "shield"` both
   carry rung `derived_at_config` in `spec_v9_cgauge.py` with **no choice-set, no eliminated
   alternatives, no falsifier** — the rung token is asserted in a free-text `note`
   (MEASURED, source inspection).

**Honesty about the ladder's current force:** `check_config_flag_provenance_bijection_complete` is
wired **WARN-ONLY** with **3862 live violations on main** (`preflight.py:6716-6721`, verbatim
reason). So today the ladder is a *design*, not an enforced boundary — a fact that must not be
laundered when citing it.

**Working definition adopted here.** An *undecided default* is a load-bearing **CHOICE** — a SUBSET,
an ORDER, a CADENCE, a MODE, a REFERENCE FRAME, a WINDOW, or an ESTIMAND — that (a) steers a decision
(a gate, an alarm, a stop rule, a verdict, a promotion), and (b) has no recorded decision: no
derivation, no race receipt, no typed governance owner. Its inverse is a **standing PROHIBITION**
whose PRECONDITION may have moved with nobody watching.

## §3 THE RANKED TABLE

Ranked by `blast-radius × silence × (cheapness of the deciding measurement)`. "Who chose it" is the
charter's key column: **NEVER-DECIDED** = no artifact anywhere records a decision; **INHERITED** =
carried in from another vehicle/context; **DERIVED/RACED** = has a receipt.

| # | surface | current value | kind | WHO CHOSE IT | blast radius | the $0 deciding measurement | expected gain | fire |
|---|---|---|---|---|---|---|---|---|
| **1** | **A1 gate estimator** `realized_gate` | unweighted mean over 36 = block(447-450) + 32 SRS | ESTIMAND + SUBSET | **NEVER-DECIDED** (`np.mean(dsegs)`, trainer:921; geometry typed "DERIVED — fd2 instrument" in the QA82 census, but the *reduction* was never typed) | **every** continuation / alarm / stage-exit / stop decision in the campaign reads it | DONE — §4 (scorer-free, receipt attached) | removes 29-39% of the design error EXACTLY; cuts sensitivity to block drift **16.7×** | **now** (log) / **next window boundary** (HT key) |
| **2** | `full_confirm` per-pair vector | `all_dsegs` computed for 600 pairs then **discarded** to mean+max (trainer:2141-2147); gate logs mean + per-pair MAX only (921-922) | OBSERVABILITY (score-neutral) | **NEVER-DECIDED** | the instrument's own error is **unmeasurable from its own logs** — this is why the bias drifted unseen | none needed; it is a 2-line write of data already in memory | unlocks #1, #10, per-pair drift typing, Neyman design — all at **zero** extra scorer cost | **now** |
| **3** | `ema_decay`'s `U` under windowing | `U = args.epochs × steps` = **cumulative** target (trainer:1372) | REFERENCE FRAME inside a DERIVED law | **DERIVED-formula / NEVER-DECIDED-binding** | shadow time-constant 166→202→236 ep (MEASURED gc14) = the lengthening half of the unintended SWA; the gate reads `gate_params="ema_shadow"`, so **the instrument's own basis drifts** | none — read the two candidate bindings off the law (campaign-total vs per-window-remaining) and pick one | removes a confound that is currently **entangled with the descent being measured** | **next window boundary** ($0 config) |
| **4** | optimizer state at window boundary | `opt_state_flat={}` at every `save_checkpoint`; `optim.Adam(...)` fresh at every resume (trainer:1543) | REFERENCE FRAME (reset-to-zero) | **NEVER-DECIDED** (a persistence omission, not a choice) | +9.6% / +22.8% ep_loss spike per boundary; gc14's "descent is a boundary step" verdict | **OWNED BY ddm_gc15 (#816) — DO NOT DUPLICATE.** gc15 closed the mechanism: `mlx.optimizers.Adam` defaults `bias_correction=False` (never overridden), so zeroed moments give η(1)=3.16 peaking at **η(12)=6.57**; one-field falsifier `bias_correction=True` ⇒ η≡1 | gc15's, not mine | gc15 |
| **5** | **window length** (~140 ep chunks) | chosen for supervisor governance | WINDOW | **NEVER-DECIDED — never even labelled** (gc14: *"a silent, unpriced hyperparameter for the entire campaign"*) | if a boundary is worth −1.1e-4 realized d_seg for ~4 s of restart, the chunking policy IS an optimizer | **DEMOTED by gc15**: the per-restart impulse is now DERIVED and CONSTANT, so a cadence A/B mostly re-measures a derived quantity — it becomes the CONTROL arm | gc15's | gc15 |
| **5b** | **DEPENDENCY defaults never explicitly passed** | `Adam(bias_correction=False)` (gc15); `torch.round` = half-to-even (row 8); `default_rng` bit generator; `zlib` level; unpassed `interpolate` kwargs | MODE (library default) | **NEVER-DECIDED — chosen by a library author for a different problem** | gc15: this is where "the load-bearing one was hiding"; gc15 records it as a **standing-law violation** — `v←0` + no bias correction makes the first step `3.16·lr·sign(g)`, a metric-free generic step, forbidden by `generic_basis_metric_never_optimal` absent derivation or race | $0 AST rule: flag every load-bearing library call whose consequential keyword is **not explicitly passed** (see §7 vii) — it would have caught `bias_correction` and it catches row 8 | closes the subclass **neither census covered** (gc15's NAMED $0 GAP: "gd1's QA82 census covered OUR defaults; it did not cover our DEPENDENCIES' defaults") | detector |
| **6** | estimator window `n_points=5` | hard-coded | WINDOW | **NEVER-DECIDED** | aliases a ~30-gate oscillation: Lane 5-gate `t=+5.15` vs full-window `t=−1.50` vs 38-gate `t=−0.69` (MEASURED gc14) — a **sign flip** | derive the window from the series' own autocorrelation length; $0 on logged telemetry | kills a sign-flipping artifact in the watch predicates | with gc14 R2 |
| **7** | gate cadence `--gate-every 5` | bare argparse default (trainer:1216) | CADENCE | **NEVER-DECIDED** | sets the sampling rate of every series #6 then fits; 5 gates × 5 ep = the 25-epoch aliasing window | $0: recompute existing gate series at every subsampling and check verdict stability | decision-quality; couples #6 and #7 into one derivation | with #6 |
| **8** | uint8 rounding mode | `torch.clamp(torch.round(up),0,255)` — round-half-to-even (witness:1623) | MODE | **NEVER-DECIDED** (generic) | the final quantization of OUR OWN output; **entirely ours to choose** — nothing upstream constrains how we pick the uint8 value from our float render | scorer-free bound available: flip-prone mass at τ=0.25 is **0.701%** of pixels vs current d_seg ≈0.41% — the target population is commensurate, not negligible (MEASURED, §4 receipt) | zero-byte, zero-rate lever; **magnitude UNKNOWN** | post-burn, realized-through-R only (see §8) |
| **9** | gate seed | `default_rng(0)`, hardcoded, **independent of `cfg.seed`** (trainer:963) | SUBSET | **NEVER-DECIDED** | good for A/B comparability, but the gate set has **never been resampled**, so its sampling error has never been observed once | with #2: recompute HT over K=8 alternative gate seeds from the already-computed per-pair vector — the empirical distribution of gate error, free | converts "how lucky was seed 0" from unknown to measured | with #2 |
| **10** | n600 anchor placement | `full_confirm` fires only at `stop_reason ∈ {epochs_complete, max_wall_minutes, basin_entry_handoff}` (trainer:2118) | CADENCE | **NEVER-DECIDED** | **every n600 anchor is taken at a window boundary** — i.e. exactly where the #4 restart transient lives. Bias drift and boundary step are **confounded by anchor placement** | $0 once #2 lands (the bias is then computed at each anchor for free); a mid-window anchor costs one n600 verdict, not $0 | de-confounds the only authority series we have | after #2 |
| 11 | `check_..._provenance_bijection_complete` | WARN-ONLY, 3862 live | (the ladder itself) | typed + owned (`preflight.py:6716`) | the enforcement everyone cites is not enforcing | — | none; **listed so no one cites it as a live boundary** | MAIN |
| **12** | B2 pose-never-trained vs `pose-in-burn REQUIRED` | **both live, in different stores** (§5b) | PROHIBITION | rule = DERIVED (5-formulation photometric wall); its **precondition is contested in the record** and unowned | the pose axis is 71.4% of the 2.08 gap in the QA43 row's own arithmetic | $0 — the reconciliation is a **read**: charter row vs SPEC vs ps1 vs QA43 stage-1a, adjudicate scope | removes a directional error already in circulation (MAIN's relay had the trigger **inverted**) | **now** (adjudication, not a race) |
| 13 | B1 "one full-n600 scorer job at a time" | claim-based prose in `current_focus.md`, no lock found | PROHIBITION | **UNVERIFIED-BY-ME** — no measured justification located | measurement throughput is the campaign's real bottleneck; wrong either way is expensive (doubling vs a #205-class OOM) | re-derive the admission projection at the real config via the existing memory-preflight | **owed** — see §5b | after a real preflight receipt |
| — | optimizer / margin temp / w_rate / pair order | Adam 2e-3 fixed / 1.0 / 0.05 / seeded shuffle | MODE, constants, ORDER | **already TRACKED — QA82 census T7/T8/T19/T18** | — | — | **NOT re-minted here** — see §5 correction | census queue |

**Not reached (named, not swept):** precision choices inside the render/loss path beyond the
documented MLX-GPU-vs-fp32 verdict split; the supervisor's own predicates outside the trainer;
`verdict_chunk=120` as a determinism-vs-memory choice; the alarm thresholds A1_SMOOTH_DROP_REL /
A1_REALIZED_DROP_REL (typed DERIVED-PROVISIONAL in the census, and gc14's R2 already owns their
re-calibration).

## §4 A1 — DESIGNED AND BUILT

### 4.1 The defect, derived exactly

`resolve_gate_ids(600)` (trainer:958-966) returns `BLOCK(447,448,449,450) + 32 SRS-without-
replacement from the other 596`, and `realized_gate` reduces it with `float(np.mean(dsegs))`
(trainer:921). That is an **unweighted mean over a non-probability sample**. The 4 block pairs carry
weight `4/36 = 11.11%` in the estimator but `4/600 = 0.667%` in the quantity it stands in for — a
**16.67× over-weight** (DERIVED; unit-tested).

With B = block, S = the SRS draw, O = the 596 off-block pairs, N = 600:

```
X_unweighted − X_pop  =  (4/36 − 4/600)·(X̄_B − X̄_S)      ← BLOCK OVER-WEIGHT
                                                            deterministic, removable EXACTLY
                      +  (596/600)·(X̄_S − X̄_O)            ← SRS SAMPLING ERROR
                                                            mean-zero over the seed, FIXED at seed 0
```

Coefficients `0.1044444…` and `0.9933333` (DERIVED; the two terms are asserted to sum to the total
error on random data in `test_decomposition_terms_sum_to_total_error_on_random_data`).

**Why this is the mechanism of the measured drift.** A change δ in the block's mean moves the live
estimator by `(4/36)·δ = 0.111·δ`; under Horvitz-Thompson weights it moves it by `(4/600)·δ =
0.00667·δ`. The block is a **contiguous 4-pair cluster** — the correlated kind whose mean drifts
together. The estimator therefore **amplifies block drift 16.7×**, and gc14 measured exactly a drift:
bias `−1.10e-6` (w01) → `+1.52e-5` (w02), overstating the window_02 descent by **7.2%**.

### 4.2 MEASURED, scorer-free — is the block unrepresentative?

`tools/gd1_gate_estimator_audit.py`, receipt
`.omx/research/ddm_gd1_gate_estimator_audit_20260731.json` (peak RSS 1415.6 MiB, no scorer, no MLX).
Per-pair proxies from the frozen GT cache: `margins` (the SegNet margin field — CLAUDE.md records it
as the Fisher surrogate at Pearson 0.978), plus 4-neighbour inter-class edge density and per-class
composition from `lstars`.

| proxy | population | block (447-450) vs pop | gate unweighted vs pop | removable by HT | SRS SE |
|---|---:|---:|---:|---:|---:|
| flip-prone mass τ=0.25 | 0.007011 | **+8.3%** | **+1.91%** | 39.4% | 1.91% |
| flip-prone mass τ=0.5 | 0.013824 | **+6.3%** | **+1.80%** | 29.3% | 1.86% |
| flip-prone mass τ=1.0 | 0.026702 | **+6.9%** | **+1.77%** | 33.9% | 1.80% |
| flip-prone mass τ=2.0 | 0.048332 | **+6.4%** | **+1.51%** | 37.6% | 1.84% |
| inter-class edge density | 0.021628 | −2.6% | +1.64% | 19.0% | 1.38% |
| **Lane pixel fraction** | 0.005855 | **−16.2%** | +3.34% | 29.0% | 3.88% |
| mean margin | 5.6116 | +1.3% | +0.13% | 91.5% | 0.31% |

**Three MEASURED findings:**

1. **The block is systematically harder than the population on every flip-prone-mass proxy
   (+6.3% to +8.3%)** — and it is the thing carrying 16.7× its population weight. Sign agrees with
   gc14's measured d_seg overstatement. *Scope: the SIGN transfers (fp-mass drives d_seg — 100% of
   flips live in the small-margin annulus, QA74); the MAGNITUDE does not, and is not claimed.*
2. **The gate set is compositionally unrepresentative in a class-specific way: Lane fraction −16.2%
   in the block, and the gate over-reads Lane by +3.34% overall.** The per-class watches (Lane b0,
   `lane_erased`, UNDRIV erosion) are computed on this same 36-pair set via `topology_per_class`
   (trainer:927). **This is a second, independent consequence of the same undecided default, and it
   has never been priced.**
3. **Level vs difference.** For a FIXED gate set the level error is a near-constant offset that
   cancels in window-over-window differences — which is *why* gc14 was right to use the series for
   shape and refuse it for magnitude. The residual in differences is exactly the bias DRIFT, which
   gc14 measured at 1.63e-5 over one window against a 2.10e-4 n600 descent = **7.8% of the signal**.

### 4.3 The fix, built

**Landed** (scorer-free, no burn dependency, 11 tests passing, ruff clean, two review passes):

- `src/tac/optimization/ddm_gd1_gate_estimator.py`
  - `GateDesign` — the sampling design, validated fail-closed.
  - `horvitz_thompson_mean(design, {pair_id: value})` — **the drop-in replacement for
    `float(np.mean(dsegs))`. Same 36 renders, different weights, zero extra scorer cost.**
  - `anchored_mean(ht_now, ht_at_anchor, n600_at_anchor)` — residual correction against the last
    n600 anchor; the caller must carry the anchor epoch (freshness at consumption).
  - `bias_decomposition` / `srs_standard_error` — the audit + the instrument's own noise floor.
  - `neyman_stratified_gate` — the re-DESIGN.
- `tools/gd1_gate_estimator_audit.py` — reads the LIVE geometry by importing the trainer (never a
  hand-copy; the trainer's module-level imports are stdlib+numpy, so this cannot start a scorer).
- `src/tac/tests/test_gd1_gate_estimator.py` — 11 behaviour tests, including a **drift guard** that
  re-derives the design from `resolve_gate_ids(600)` and a **400-draw unbiasedness test** showing HT
  bias < 15% of the unweighted bias on a deliberately-hard block.

**The trainer patch is NOT landed here — deliberately.** The burn is live and run dirs are sacred;
changing an instrument mid-campaign is itself a comparability confound. The patch is
**additive-only** and belongs at a window boundary, in this order:

```python
# trainer realized_gate(), after `dsegs, realized = cpu_verdict_d_seg_argmax_batch(...)`:
row["realized_gate_dseg_per_pair"] = [float(d) for d in dsegs]           # (a) OBSERVABILITY, now
row["realized_gate_dseg_mean_ht"] = horvitz_thompson_mean(              # (b) NEW KEY, never a replacement
    _GD1_DESIGN, {int(i): float(d) for i, d in zip(gate_ids, dsegs)})
# trainer full_confirm block, alongside realized_dseg_mean:
receipt["full_confirm"]["realized_dseg_per_pair"] = [float(d) for d in all_dsegs]   # (c) free
```

(a) and (c) are pure observability — they cannot change a weight, a byte, d_seg or d_pose, so per the
"off is a tracked queue" rule there was never a safety reason to discard them. (b) adds a key and
leaves `realized_gate_dseg_mean` untouched, so **the historical series stays comparable** while the
unbiased one starts accumulating. **Ordering matters: (a)+(c) first — they are what make everything
else measurable rather than argued.**

**Deferred to a campaign boundary, pre-registered, NOT recommended mid-burn:** the Neyman
re-design. Measured prediction on the τ=1.0 proxy at the *same* 32-pair budget: variance ratio
**0.128**, i.e. **−64.3% SE** (allocation `[8,3,4,17]` over 4 equal-count sensitivity strata,
candidate ids in the receipt). This is a DESIGN prediction on a proxy, not a d_seg measurement, and
it **changes the gate SET**, which breaks comparability with the campaign's own gate history.
Re-weighting does not.

### 4.4 Attacking my own work (§6 of the operating manual)

The self-review round found a real defect **in my own code**: the audit tool's docstring claimed
"peak RSS stays ~O(chunk)". False — an `NpzFile` member is fully materialised on first access, so
`margins` (472 MB) + `lstars` (944 MB) dominate; chunking bounds only the transient intermediates.
Fixed, and the tool now **measures and reports** peak RSS (1415.6 MiB) rather than claiming a bound.
That is the docstring-overstatement forbidden pattern, caught in the one artifact this unit built.

Remaining honest limits: the HT fix removes the *deterministic* term exactly, not the fixed-seed SRS
term — for that you need #2 + #10 (anchoring). And the whole §4.2 measurement is on **proxies**; the
realized per-pair d_seg split is not computable from existing telemetry, which is finding #2.

## §5 AUDIT OF MAIN'S SEED — corrections

**A1 — CONFIRMED and sharpened.** MAIN ranked it highest; agreed, and the mechanism is now derived
(16.67× block over-weight) and the block's unrepresentativeness measured. One correction to the
standing record: the QA82 census types row **S12 gate-set geometry as "DERIVED (fd2 instrument
geometry)"**. That typing is right about the *geometry* and wrong about the *estimator*: a
deliberately-chosen instrument block is a perfectly good design, but reducing it with an unweighted
mean silently converts a designed instrument into a biased estimator of a different quantity. **The
census row should be amended to DERIVED-geometry / NEVER-DECIDED-reduction.**

**A2 optimizer — CORRECTED. This is not undiscovered; it is unenforced.** The QA82 census row **T7**
already types it: `plain Adam, lr 2e-3 fixed, no schedule — TRACKED-NEVER-FIRED`, with the lever
`update_rms_matched_optimizer_race` recorded as *logged-not-enforced* (#685 px1) and Muon typed
ancestor-lesson-only per L18. Minting a new "race the optimizer" item would be exactly the
rediscovery this unit exists to prevent. **The real finding is the meta one:** a lever can be
correctly registered, correctly typed, and still never fire — which is precisely the
`never-fired` state the activation ledger already models and the costate duty queue already ranks.
The gap is that nothing *forces* a never-fired high-blast-radius row to be drained. gc14's own
evidence that the optimizer's STATE is where the gains live (#4 above) is the argument for promoting
it in that queue — not for a new row.

**A3 uint8 rounding — CONFIRMED, and it is genuinely ours.** `torch.clamp(torch.round(up),0,255)`
at camera resolution (witness:1623). Nothing upstream constrains how we choose the uint8 value from
our own float render; the scorer reads whatever uint8 we emit. So margin-aware rounding is legal and
zero-byte. I did **not** verify MAIN's relayed `#532` figure ("Δ 62.74 vs 1.7e-13") — cited as
**UNVERIFIED-BY-ME**. What I can offer scorer-free is a bound on the target population: flip-prone
mass at τ=0.25 is **0.701%** of pixels while current d_seg is ≈0.41% — commensurate, so the lever is
not dismissible on magnitude grounds. See §8 for the prior-law prediction that constrains how it must
be evaluated.

**B1 / B2 — see §5b.**

**The "also sweep" list — two items are already census rows and are NOT re-minted:** loss-term weight
values (**T19** `w_rate=0.05`, no provenance rung, re-derivation from the exact `25/37,545,489`
exchange rate already queued as QA86; **T8** margin temp 1.0, sweep queued) and pair order (**T18**,
held on a pools-slot argument with a named reopen trigger). Base init is census **T11/T12** (RACED /
DERIVED). Per-window EMA re-derive is my #3 and is the sharpest item on that list.

### §5b B1 and B2 — the inverse class (PROHIBITIONS whose precondition may have moved)

**B2 — CORRECTED, and the precondition has ALREADY MOVED.** MAIN relayed the trigger as
"*>~600 B/pair ⇒ pose-in-burn REQUIRED*". **The recorded logic is the inverse.** Verbatim, from the
QA43 row of `ddm_burn4_charter_skeleton_20260731.md:75`:

> "re-adjudicates v10 row-12 pose-in-burn (**currently REQUIRED** at INSTANCE(uniform-≤4KB) scope —
> do NOT pre-flip). Falsifier: realized price > ~600 B/pair → burn strictly cheaper, **REQUIRED
> stands**"

So (a) `pose-in-burn = REQUIRED` is the **standing registered verdict**, not the flip target;
(b) `>600 B/pair` **CONFIRMS** it; (c) it is a price coming in **cheap** (the row's target is
~120 B/pair, which would give `ΔS ≈ −1.10 ≈ 53% of the gap`) that would relax it. Anyone acting on
the relayed direction would have read the trigger exactly backwards.

Two further recorded facts sharpen this into the strongest PROHIBITION finding of the sweep:

- **The terminal-solve base is MEASURED WALLED.** `ddm_ps1_pose_stage_20260730.md` (#791): on the
  B-control post-burn seg-native parent the post-hoc geometric solver is L68-walled — n600 stub
  160.10 → warp 27.82 → solved 20.41, while the same solver reproduces 0.172 on the pose-legible
  pb1 base as a CONTROL, so the ~120× blowup is **purely f1 photometric illegibility**. Verdict:
  "QA43 tail refinement is **moot on a walled base**… the family stays LIVE for a pose-CONDITIONED
  base. **Reopen when a burn conditions pose in-loop.**"
- **QA43 stage-1a then FIRED positive at ~0 bytes:** two-plane per-class warp, full tail-112,
  95/112 wins, composed pose axis 1.4881 → 0.9127 = **−0.5754 S measured** at ≤7.3 KB marginal.
  Its own routing line reads: "v10 row-12 pose-in-burn: pressure **REDUCED** by −0.407 measured at
  ~0 bytes; verdict still INSTANCE-scoped, **re-adjudicate at full-112**."

**The finding is therefore not "nobody computes bytes-per-pair."** It is worse and more interesting:
**a `REQUIRED` verdict and a `NEVER` rule are both live, in different stores, and nothing
reconciles them.** The charter ledger types pose-in-burn REQUIRED; the SPEC/#383 discipline says pose
is never trained; ps1 says the terminal-solve base is walled and names the reopen condition; QA43
says re-adjudicate at full-112. Every one of those is individually well-scoped and honestly labelled
(INSTANCE scopes throughout — this is not sloppiness). **What is missing is the reconciliation
surface**, which is exactly what §7(vi) `kind=PROHIBITION` with `precondition` +
`precondition_watch` exists to be. I am NOT declaring #383 dead: its own measured basis (the
post-hoc/stored family died across 5 formulations on a photometric wall) is intact and is on the
PRESERVE list. I am saying its **precondition is contested in the record and nobody owns the
adjudication.**

**B1 — partially verified; I could not close it scorer-free within this unit.** What I could confirm
myself: the serialization is expressed as **claim-based prose, not a lock** — `current_focus.md`
carries "owns the scorer slot" language per-arm (e.g. `:311` "Rung 1 LIVE (task #803, ddm_r1c, owns
the scorer slot)"), and my own charter inherited it as "window_03 owns the single n600 slot until
~18:40Z". A convention enforced by charter text is exactly the class of rule whose precondition is
never re-checked, because there is no code to fail. **I did not locate a measured justification for
the ONE-at-a-time bound, and I did not verify the relayed "83.8 GiB free vs 25.6 GiB floor" receipt.
Both are marked UNVERIFIED-BY-ME and remain owed** — flagged rather than asserted, because
"measurement throughput is the campaign's bottleneck" makes this the row where a wrong answer is
most expensive in either direction (a straight doubling of learning rate if it fits; an OOM that
kills a burn with no checkpoint if it does not — the #205 receipt).

**The structural point, independent of both:** B1 and B2 are the same object as §3 rows 1-10 with
the sign flipped. A standing "never do X" is a decision too, and it decays the same way — not by
being wrong when made, but by having a PRECONDITION nobody watches. **A registered trigger with no
watcher is strictly worse than no trigger, because it manufactures the belief that the condition is
being monitored** — and B2 shows the second-order version: a trigger that IS registered, HAS fired
in the record, and still has no one whose job is to act on it.

## §6 PRESERVE LIST — what survives, with its measured basis

An audit that reopens everything is as useless as one that reopens nothing. These **SURVIVE** and
must not be reopened by this unit or its successors:

| rule | measured / structural basis | status |
|---|---|---|
| **MPS is never a score authority** | 23× PoseNet drift, 2× SegNet, 2.5× final score, measured side-by-side on a pinned archive; 95.5% of a selector's argmin picks corrupted | **PRESERVE — unconditional** |
| **Never auto-delete upstream strays** | pinned upstream is immutable by directive; the #812/#407 fix is fail-closed guard + warn-only detect, explicitly NEVER auto-delete | **PRESERVE — unconditional** |
| **`REVIEW_GATE_OVERRIDE` forbidden on `.py`** | the gate is the pre-ship bug filter; override on code is how bugs ship | **PRESERVE — unconditional** |
| **No old lineage (HNeRV/PR95/110/128) as vehicles/carriers/calibration** | operator-set 2026-07-23; lessons-only | **PRESERVE — operator-set** |
| **No co-author trailer** | operator-set; all commits are the operator's | **PRESERVE — operator-set** |
| **Governed launcher / DSL-hash / memory-preflight P0** | the #205 OOM receipt: a config that passed the full seal and the B=8 throughput gate still OOM'd at 90 GB with no checkpoint | **PRESERVE — P0** |
| **n600-or-it-is-not-evidence** | this unit's §4.2 is a *fresh receipt* for it: the 36-pair instrument is measurably non-representative. gc14's authority rule (n600 binds, 36-pair for shape only) was **correct** | **PRESERVE — strengthened** |
| `|net betti0| ≤ 10` pre-authorization bound | gc14 endorses it explicitly as the **model** of a correctly-typed class-4 governance knob: owner named (MAIN), re-derivation trigger named (λ_undriv via cg1 #809) | **PRESERVE — and use as the template** |
| Raster wire order for SMEVR | MEASURED, QA82 §5: raster wins everywhere; Hilbert +452 B on the context-matched comparison, +5 KB under brotli/lzma | **PRESERVE — closed with a receipt** |
| bicubic-up in the R chain | RACED (QA79): −8.4e-5 d_seg vs bilinear at zero bytes | **PRESERVE — raced** |
| `class_values` canonical comma10k order | MEASURED; luma-sorting is wrong and has bitten 3× | **PRESERVE — unconditional** |
| Pose is a terminal solve, seg frozen first | #383 conditioning gate + the measured death of the post-hoc/stored family across 5 formulations (photometric wall) | **PRESERVE the RULE — watch the trigger (§5b)** |

The last row is the honest shape of this whole unit: *preserving a rule and watching its precondition
are different jobs, and only one of them is currently being done.*

## §7 THE DETECTOR — the Decision Provenance Register (DPR)

**One sentence:** the value-provenance ladder types **VALUES**; the DPR types **ESTIMANDS, SETS,
ORDERS, CADENCES, MODES, WINDOWS and REFERENCE FRAMES** — the things a formula's certificate never
touches.

**Recall first, so this does not become the 53rd orphan.**
`.omx/state/arbitrariness_extinction_audit_20260518.jsonl` already tried this: 52 rows of
`{value_id, is_arbitrary, resolution_path, predicted_ev_delta_s, blocking_dependencies, …}`. It died
for two reasons, both avoidable: it was **value-scoped** (so it could not express the A1 estimand
mismatch) and it was **wired to no gate and no consumer** (so nothing ever read it). Therefore the
DPR below leads with the gate and the duty row; the schema is last.

**(i) The gate — a sibling in the existing `confound_gates.py` idiom, not a new layer.**
`check_load_bearing_choices_are_decided(repo_root=None, *, strict=False, verbose=True) -> list[str]`,
returning violation strings, ending in `_finish(...)`, honouring a same-line
`# CHOICE_PROVENANCE_OK:<rationale>` waiver with the existing non-placeholder rationale check, added
to the `CONFOUND_GATES` tuple **WARN-ONLY** per strict-flip atomicity. It refuses a choice that is
(a) reached from a decision path, (b) `decided_by == NEVER_DECIDED`, and (c) has no bias-audit
receipt. Its natural strict-flip sibling is `v9_provenance_gates.py:1324` — the single line that
today accepts any of four rungs for any flag *regardless of whether the value is a number or an
enum/mode/cadence*.

**(ii) Auto-discovery, AST-derived — this is what makes it not a hand-typed registry.** Mirror
`lever_registry.completeness()`: an AST pass over the trainer/witness/decode modules emitting
`unregistered_choices` from choice-shaped call sites —
`np.mean(...)`/`.mean()` over a subset-named variable → **ESTIMAND/SUBSET**;
`default_rng(<literal>)` / `seed(<literal>)` → **SUBSET/ORDER**;
`% <literal> == 0` inside a loop → **CADENCE**;
`round/rint/floor/ceil` in a verdict or export path → **MODE**;
a literal-window slice (`[-N:]`, `n_points=`) → **WINDOW**;
re-construction of stateful objects after `load_checkpoint` → **REFERENCE FRAME**.
**(vii) — the rule the sister arm's find demands:** for a curated set of load-bearing library
constructors and numerics calls (optimizers, `round/rint`, `interpolate`, RNG constructors,
compressors), flag every **consequential keyword that is not explicitly passed**, so the *dependency's*
default is registered as OUR choice. This is the rule that would have caught
`mlx.optimizers.Adam(bias_correction=False)` (gc15 #816 — the load-bearing one, hiding in a
dependency) and it independently catches §3 row 8 (`torch.round` = half-to-even). It also produces
benign hits — e.g. `interpolate(..., antialias=False)` on an UPsample, which a human clears in
seconds — and that is the correct failure direction for a detector.
Every one of §3 rows 1-10 is a hit for exactly one of these seven patterns — the rule set was derived
*from* the found instances, and its first job is to reproduce them. `emit_stub_choice(site)` mirrors
`emit_stub_lever` so completing the register is review-and-accept, never hand-typing.

**(iii) The duty row — reuse, do not rebuild.** Each registered choice gets an
`activation_ledger`-shaped row `{choice_id, kind, decided_by, last_bias_audit_utc, blast_radius}`
so `duty_to_measure_ranked()` ranks NEVER-DECIDED × high-blast-radius into the costate DECIDE queue
and `costate_digest` NAGS. The `*=never-fired` marker generalises to `*=never-decided` unchanged.

**(iv) The carrier already exists.** `Lever.policy_contracts` is an **argv-inert, sha-pinned,
non-scalar custody channel that already survives the Lever→TypedLever boundary** (its one live user,
`IntegerPlaneEmitter`, compiles a policy into `{policy_sha256, basis, mode}`). A `ChoiceRef` rides
that channel. No new plumbing.

**(v) The one genuinely new field: `estimand`.** REQUIRED for `kind ∈ {SUBSET, WINDOW, ESTIMAND}`.
It names the population or quantity the choice stands in for, and it obliges a `bias_audit` — the $0
check that bounds the gap between what is computed and what is reported. A1 is the worked example:
estimand = "mean realized d_seg over all 600 pairs", bias_audit = `bias_decomposition` against the
n600 anchor. **No existing field can express this, which is why A1 was invisible for the whole
campaign.**

**(vi) The inverse class in the same register.** `kind=PROHIBITION` with two required fields:
`precondition` (the measured fact that made the rule right) and `precondition_watch` (the thing that
re-checks it, with a staleness clock). The gate fails closed on a PROHIBITION whose precondition has
no watcher. B1 and B2 are the seed rows; the `|net betti0| ≤ 10` bound is the passing example to
calibrate against.

**Why this is the right shape (from the ema_decay asymmetry the operator asked me to explain).**
`ema_decay` got a law for two reasons and neither was structural: an operator P0 pushed it
(`p0_ema_calibration_20260717`), and **a free anchor existed** — a closed-form geometric identity
cross-checkable against an already-registered executable (`ema_warmup_updates(0.997) == 667`).
Everything else falls through to the honest class-4 backfill via `dsl_custodied_scalar_identity_v1`.
**There is no queue anywhere that says "derive this next."** So the asymmetry is:
`operator attention × free-anchor availability`. The DPR fixes exactly that by ranking on
blast-radius and by making "is there a free anchor?" an explicit field — and A1 is the proof the
question pays: **the anchor was free the whole time** (the n600 per-pair vector was being computed
and thrown away at every window boundary) and nobody was pointed at the question.

**Not built this unit, and deliberately so.** A register with no gate is the 2026-05-18 orphan; a
gate landed without its auto-discovery is a hand-typed registry beside the DSL, which the triality
discipline forbids. Both halves belong in one landing. This is a specification, labelled as such.

## §8 PRIOR-LAW PREDICTION LINES (pre-registered, before any of the above fires)

- **constants-are-poison** → predicts #3 and #6: a certificate on the formula does not certify the
  binding of its free variables (`U`) or the length of an estimator's window (`n_points`). **Both
  confirmed by source inspection.** gc14's extension of the law to WINDOW and REFERENCE FRAME is
  what this memo generalises to SUBSET, ORDER, CADENCE, MODE and ESTIMAND.
- **generic-basis-metric-never-optimal** → predicts that the *unweighted mean* and *simple random
  sampling* are controls, never optima, exactly as cosine/Fourier/Euclid were. **Confirmed** (§4.2):
  SRS is beaten by Neyman allocation by a predicted −64.3% SE at identical budget. **Independently
  corroborated by gc15 the same day** on a different surface: the zeroed-moment first step is
  `3.16·lr·sign(g)` — a uniform-magnitude, metric-free step that "arrived as a library default and
  was never derived or raced." Law record note: QA82 §5 found raster SURVIVING as the first raced
  default; this unit adds a *falsified* one — the ledger stays honest in both directions.
- **alarm-predicates-are-per-vehicle-calibration-objects** → predicts finding #2 in §4.2: the
  per-class watches read a Lane-poor sample (−16.2% in the block). **Confirmed structurally**; the
  realized effect on the UNDRIV/Lane predicates is NOT measured and is owed.
- **staleness-is-a-named-confound** → predicts that estimator bias is a staleness axis, and that
  `anchored_mean` must carry its anchor epoch. Built that way.
- **ERF-collateral (fp1+QA92, 07-31)** → constrains **#8**: post-hoc injection on textured renders is
  net-worse because an ~85 px ERF re-reads the perturbation. A ±0.5-LSB rounding nudge is ~1/255 of
  the amplitude that law was measured on, so it does not obviously kill the lever — but it predicts
  the falsifier precisely: **per-pixel greedy margin-aware rounding will not compose**, because each
  nudge is re-read across the ERF. Margin-aware rounding must be evaluated **realized-through-R at
  n600**, never per-pixel-greedily. Pre-registered before the race.
- **conditional-validity re-grade** → the QA82 census row S12 ("DERIVED — fd2 instrument geometry")
  was valid for the geometry and does not cover the reduction; re-graded in §5, not overturned.

## §9 TRIALITY / verdict scope / STORES CONSULTED

- **DAG:** FEED-gd1 appended to `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
  **DSL:** no new lever this unit — the A1 fix is an estimator + a telemetry key, not a config knob;
  the DPR §7 is a *specification* for a `confound_gates` sibling and is explicitly not half-wired.
  **equations:** no new law; this unit APPLIES constants-are-poison and generic-basis-never-optimal
  and proposes their extension from VALUES to CHOICES. `[p0-ledger-ok]`.
- **verdict_scope:** §4.1 and §4.3 are **DERIVED** (algebra, unit-tested) and INSTANCE-scoped to this
  gate design. §4.2 is **MEASURED** on scorer-free proxies from the frozen GT cache — sign transfers
  to d_seg, magnitude does not. §3's "WHO CHOSE IT" column is **VERIFIED_VIA_SOURCE_INSPECTION**
  except where it cites gc14/QA82 (VERIFIED_VIA_EMPIRICAL_ANCHOR / prior receipt). §7 is a
  **specification**, unbuilt. §8's ERF prediction is **INFERRED** and pre-registered.
- **Relative significance:** no ΔS is claimed by this unit. The A1 fix buys **decision quality**, and
  its size is stated relative to the signal it corrupts: the instrument's bias drift is **7.8% of one
  window's realized descent** (gc14 MEASURED), and the estimator amplifies the drifting component
  **16.7×**. Rows 3-7 are confound removal, not S. Row 8 is the only row with a plausible direct ΔS
  and its magnitude is **UNKNOWN**, bounded only by a commensurate target population (0.701% of
  pixels vs d_seg ≈0.41%).
- **STORES CONSULTED:** CLAUDE.md (NO-FAKE supreme rule, THE GOAL, off-is-a-tracked-queue, confound
  self-protection 3-layer, allergic-to-non-n600, strict-flip atomicity, forbidden patterns);
  `docs/operating_manual_craft_handoff.md` (§3 risk ranking, §4 re-derive, §5 labels, §6 attack your
  own conclusion — which caught §4.4); MEMORY index (constants-are-poison, generic-triple law,
  staleness, ERF-collateral, verdict-scope ladder, no-old-lineage, deferral-scatter);
  `ddm_gc14_first_descent_20260731.md`; **FEED-gc15 (#816) in the canonical DAG** — recalled on the
  third pass and the reason rows 4/5 are ceded rather than re-minted;
  `ddm_gd1_generic_default_census_20260731.md` (the sister
  axis); `ddm_dg1_rate_denominator_guard_20260731.md`; `ddm_us1_upstream_reread_20260731.md`;
  `ddm_qa43_two_plane_parallax_20260729.md`; trainer/witness/coder sources (file:line per row);
  `lever_registry.py`, `activation_ledger.py`, `costate_digest.py`, `confound_gates.py`,
  `v9_provenance_gates.py`, `lawref.py`, `spec_v9_cgauge.py`,
  `canonical_equations/ema_decay_run_geometry_20260717.py`, `constants_telemetry_build_wave_20260715.py`,
  `.omx/state/arbitrariness_extinction_audit_20260518.jsonl`, `.omx/state/operator_p0_ledger.jsonl`.
