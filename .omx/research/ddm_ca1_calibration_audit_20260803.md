# ddm_ca1 — calibration audit of the live TR1 path: is a derivation RUNNING, and was it sized for the job it is doing?

**Arm:** `ddm_ca1`. **Date:** 2026-08-03. **Operator directive:** *"Ensure that all calibration is
optimized and informed and not naive or toy or generic basis."*

**No scorer slot taken** (`ddm_pu2` holds it). Every measurement below is static/AST/arithmetic and
`$0`. Every owed measurement is **named, not run**. **Pointer UNMOVED.**

**Baseline for every ΔS in this memo:** live best **S = 0.7910689** @ **353,805 B** (the `pu2` pose-tail
rung on top of `cx1`), seg leg **0.4311790**, gap to the PR130 bar (**0.172141**) = **0.6189279**,
`W = 1.273108215332031` B/flip. *(`cx1` proper is a different rung: 353,808 B, S = 0.8264972. I had
these conflated in my stub; corrected.)*

---

## 0. HEADLINE

| | |
|---|---|
| **The live TR1 path is in GOOD calibration shape, and that is the primary result.** | Of 6 derivation-named functions defined on the 22 live TR1 files, **6/6 are called in production**. **Zero** frozen-output candidates. The decode closure's constants (`_R7_SMEVR_MAX_LEVELS=16`, `BETA_MAX_DOUBLINGS=17`, `_EXHAUSTIVE_CAP=400000`, `CANARY_MAX_ABS_ERR=1.2e-05`, `MS8_ARCHIVE_DELTA_BYTES=51`) carry **explicit, checkable derivations in-line**. Several are exemplary. |
| **But my own detector is too weak to make that a strong claim — and the coordinator's own case proves it.** | `derive_margin_floor` has `prod_calls=1` and would be scored CLEAN by the zero-callsite test. Refined: of **463** scalar-returning derivations repo-wide, **57** have zero prod callsites, **329 (71%)** are called **only inside their own defining module** — the `derive_margin_floor` signature — and only **77 (17%)** are consumed by another module. |
| **Three armed landmines, none of them a live defect today.** | A **200,000-byte archive ceiling** that computes `maximum_allowed_added = 0` at the live 353,805 B; a **rANS 12-bit precision** with no derivation sitting where a coder race will read it as a result; a **pose wall threshold sized for a 3.27× looser target** than the bar we now chase. |
| **The operator's third word lands literally, once.** | `--distill-temp 2.0` is tagged in-repo as *"Quantizr/PR95 T=2.0 provenance rung"* — a **banned-lineage borrowed constant**, inert today because its lever defaults OFF. |

---

## 1. THE FAILURE MODE, CORRECTED MID-AUDIT

My charter's canonical instance was `margin_floor = 0.1` as *"right floor, wrong job."* **`ddm_rt2`
withdrew that finding** (`34011da8c1`): re-measured against the **Lane-restricted** margin field (not
the global one), the derived floor is **0.104748 vs shipped 0.1 = 1.047×**, and two independent routes
agree (fp32 cross-hardware drift ~0.096; Lane-margin p10 0.1047). **The constant is fine.** A clean row
is a result, and this one is reported as one.

The replacement instance is sharper and it reorders this audit:

> **A derivation exists in-repo, is documented, and is NOT CALLED — its output was frozen into a
> literal.** The value is right *today*, so nothing fails and no test goes red. If the data moved,
> nobody would find out. **Such a constant passes every provenance check while running none.**

So the audit asks two questions per row, **in this order**:

1. **Does a derivation exist in-repo, and is it CALLED at runtime — or was its output frozen?**
2. **Sized for WHAT job — and is that still the job it is doing?**

Question 1 is primary because a frozen-output constant *has* a derivation and therefore satisfies the
value-provenance ladder's rung test by inspection.

### 1.1 Two corrections to the relayed instance (recompute, never re-type)

- **Path.** `derive_margin_floor` is at **`src/tac/optimization/lane_guard.py:547`**, not
  `src/tac/witness_control/lane_guard.py:547` — that file **does not exist**
  (`ls` → `No such file or directory`). The line number is right; the package is not.
- **"Not called" is too strong.** `derive_margin_floor` has exactly **one** prod callsite —
  `lane_guard.py:810`, inside its own module — reached only when the lane guard is enabled, and
  `--lane-guard-margin-floor-weight` defaults to **0.0 ⇒ OFF**. The accurate statement is: *a live
  derivation, behind a default-off lever, in a different module from the one that hardcodes the
  constant. The two surfaces never meet.* That is the `default_off_is_orphaned_signal` class
  compounded with the frozen-output class, not either alone.

---

## 2. METHOD + DENOMINATORS (empty scope is VACUOUS, never PASS)

### 2.1 The drowning mechanism, measured

`rt2`'s "no matches" and my own first `grep -r` were both **killed at 120 s before reaching `src/`**.
The cause is measurable:

```
all .py under src+tools+experiments                        62,999
  excluding experiments/results/**  (vendored source_bundle copies)   13,868
  => vendored duplicates                                   49,131  = 78.0%
```

**78% of the `.py` surface is vendored copies of itself.** Any unbounded content grep spends its whole
budget there. Every sweep below is **bounded by construction** and states its exclusions.

### 2.2 Instrument A — calibration-knob extractor (AST)

Emits `CONST` / `KWARG` / `ARGPARSE` / `Field` numeric defaults whose *name* matches a calibration
token (floor/cap/thresh/tol/weight/temp/rate/margin/quant/budget/window/…), excluding structural
shapes, and attaches the nearest justification text (same-line comment → up to 4 preceding comment
lines → enclosing docstring) so question 2 can be asked per row.

```
files named 22 · found 22 · missing 0 · knob rows emitted 108
```

### 2.3 Instrument B — frozen-derivation detector (AST, two passes)

Pass 1 collects functions named `derive_*|_derive_*|compute_*|*_derived|*_from_*`; pass 2 counts their
callsites, split test vs non-test.

```
files in scope 10,342 · parsed 10,342 · parse errors 0
excluded path parts: experiments/results/, /.venv/, /node_modules/, upstream/
derivation-named functions   2,472
  scalar-returning             463
    (a) ZERO prod callsites     57   [FROZEN]
    (b) prod calls ONLY in own module  329  (71%)   <- the derive_margin_floor signature
    (c) consumed by >=1 other module    77  (17%)
```

**Instrument disposition (deliberately NOT landed as code).** Both instruments are specified above in
enough detail to rebuild — name-match patterns, excluded path parts, and the denominators each stage
must report. They are **not** landed as `tools/*.py` because the durable surface this audit needs is
the **name join** (O1), and that join is already being built by `ddm_pt2 §1`; landing a second,
parallel detector beside it would be the `built_new_machinery_instead_of_paying_identified_debt` bug
this repo has already named. This memo is therefore a **dated bridge artifact** whose named consumer
is `pt2`'s join. No evidence path in this memo points at scratch storage.

### 2.4 The honest limit of instrument B — stated against my own result

**329 is an UPPER BOUND on the class, not a defect count.** Many self-contained derivations are
legitimately internal helpers whose own module *is* the consumer (`derive_lambda_step_cap`,
`derive_noise_floor`, `derive_deadband_k` — all correctly consumed by `lane_guard` itself). The actual
defect is narrower: **derivation in module A, constant hardcoded in module B, A and B never meet.**
Detecting *that* requires a **name join** from derivation-name → constant-name, which instrument B
does not have. `ddm_pt2 §1` is building exactly that join for levers (it measured 0 of 43 joinable);
**the same join is the missing detector here.** Named, not built — building a second parallel join
would be the `built_new_machinery_instead_of_paying_identified_debt` bug.

Consequence to carry: **my live-path result of "0 frozen" is bounded by an instrument that would have
scored the coordinator's own canonical case CLEAN.** It is evidence, not proof.

### 2.5 Scope of "the live TR1 path" — corrected mid-audit

My first file list was **wrong** and I re-ran both sweeps. Corrections, from an independent call-graph
trace cross-checked against `.omx/research/ddm_cu1_frontier_custody_20260803.md`:

- `launch_ddm_joint_descent.py` + `direct_description_joint_descent.py` are a **different vehicle**
  (describe/joint-descent) — an active research surface, but they do **not** produce the archive.
  `rt2`'s `margin_floor` lives there, not on the TR1 build path.
- `direct_description_minimizer.py` + `direct_description_carrier_compose.py` are **dormant and
  BLOCKED** (fail their own tests, per `ddm_cu1_consolidation_disposition_20260803.md:31`).
- I had **missed the entire 6-file decode closure**, including `experiments/ddm_r7_token_coder.py` —
  the token codebook that carries **99% of the archive bytes**.

The corrected list is **22 files**: the 6-file decode closure + 13 builders/solvers + the trainer,
launcher and its DSL spec.

---

## 3. THE TABLE

`Q1` = does a derivation exist / is it called. `Q2` = sized for what job / still that job.
**Ranked by (distance from re-derived value) × (downstream sensitivity)** — a 59× ratio on a term
contributing 7e-06 ranks below a 1.0× ratio on a term that gates the rate axis.

| # | constant | value | read-site | rung | Q1: derivation exists / called? | Q2: sized for WHAT job | still that job? | re-derived / owed |
|---|---|---|---|---|---|---|---|---|
| **1** | `total_archive_ceiling_bytes` | `Literal[200000]` | `direct_description_carrier_compose.py:1334,:1441` → enforced `run_ddm_v9_carrier_compose.py:2181,:2371,:2513,:3036` | **frozen literal, pydantic `Literal` — not overridable** | none exists | the **~52.5 KB-base era** (`added_budget_bytes` max 147,456 ⇒ implied base ≤ **52,544 B**) | **NO — era gap 6.73×** | at live 353,805 B: `maximum_allowed_added = max(0, 200000−353805) = **0**`. **Owed:** delete or re-derive from the live base before the composer is ever re-fired |
| **2** | `ANS_SCALE_BITS` | `12` | `repair_entropy_coder_runtime_adapters.py:33`; imported by `ddm_r7_token_coder.py:50-57` | **bare literal, ZERO justification text** | none exists | **nothing stated.** 12 is the textbook/reference rANS probability precision (`ANS_TOTAL_FREQ = 1<<12 = 4096`) | **N/A — not auto-selected** (`AUTO_CODECS = ("smevr","brotli11")`) | **Owed:** derive from the token-symbol distribution's entropy *before* any rANS-vs-smevr race, else the race measures the constant (see §4.2) |
| **3** | `thr_wall` | `0.00025` | `ddm_p3v2_optimal_form_pose_resolve.py:606` | MEASURED-ANCHOR-shaped, **target-conditional** | derivation is inline: `sqrt(10·0.00025) = 0.050000` ✓ recomputed | a **pose contribution of 0.05** | **NO** — the PR130 bar's pose contribution is **0.015268**; the wall sits **3.275×** above the bar, **10.72×** looser in `d_pose` units | re-derived at the bar: `d_pose = 0.015268²/10 = **2.3311e-05**. **Owed:** confirm the wall's consumer wants "binding at 0.05" and not "binding at the bar" |
| **4** | `--distill-temp` | `2.0` | `train_tr1_partition_renderer_mlx.py:2103`, `spec_tr1_renderer_20260728.py:359` | **BORROWED — self-tagged** *"Quantizr/PR95 T=2.0 provenance rung"* | none; a lineage citation is not a derivation | **PR95's vehicle** | **banned as calibration** (HNeRV/PR95/110/128/130 = lessons-only). Inert today: `distill_weight` defaults **0.0 ⇒ OFF** | **Owed:** derivation-or-race before the distill lever is ever fired. Not urgent; the lever is off |
| **5** | `MS8_ARCHIVE_DELTA_BYTES` | `51` | `dc1_menu_sweep.py:89` | **MEASURED-ANCHOR, config-conditional** | measured: `360,374 − 360,323` | the **ms8 archive** (360,374 B) | **UNVERIFIED at live** — the live archive is **353,805 B**, a different container (`cx1` ix2 single-member) | **Owed:** re-measure the s_t index-stream delta on the ix2 container, or state it as ms8-scoped |
| **6** | `--lr` · `--gate-every` | `0.002` · `5` | `train_tr1_partition_renderer_mlx.py:1967,:1970` | **bare argparse defaults, no justification text** | none | nothing stated | — | **ALREADY OWNED — `ddm_gd1` §3 rows 7 + QA82 census T7. Cite, do not duplicate.** |
| **7** | `_EXHAUSTIVE_CAP` | `400000` | `ms8_st_codebook_race.py:230` | **DERIVED** ✓ | inline: `C(21,10) = 352,716` ⇒ every codebook size on a 21-point support is a **certified global optimum** | a **21-point s_t support** | **YES today** (live `ST_GRID` = 11 entries) — but the guarantee is **silent-conditional**: `C(23,11) = 1,352,078` **breaches** the cap | headroom `400000−352716 = 47,284`. **Owed (cheap):** assert `comb(len(support), len(support)//2) <= _EXHAUSTIVE_CAP` so the certificate fails loudly instead of degrading to a truncated search |
| **8** | `margin_floor` | `0.1` | `direct_description_joint_descent.py:2281` | **derivation EXISTS, output frozen** | **`derive_margin_floor` exists** (`optimization/lane_guard.py:547`) and is called at **one** site (`:810`) — **in a different module, behind a default-OFF lever** | fp32 cross-hardware drift guard (~0.096) | **YES — WITHDRAWN as a defect.** Lane-restricted p10 = **0.104748 = 1.047×**; two independent routes agree | **CLEAN.** The hinge **WEIGHT** (`0.05`) is the real lever — **owned by `ddm_mg1`** |
| **9** | `_R7_SMEVR_MAX_LEVELS` · `R7_MAX_LEVELS` | `16` · `16` | `ddm_tr1_runtime.py:83`, `ms8_st_codebook_race.py:309` | **hard boundary, correctly labelled** | n/a — it is a coder constraint, and both sites say so | the `_codes` admissible range `2 ≤ levels ≤ 16` | **YES** | **CLEAN — exemplary.** ms8 states it verbatim: *"a hard boundary of the design space, not a tuning choice, and it is reported rather than silently…"* |
| **10** | `BETA_MAX_DOUBLINGS` | `17` | `ddm_v4d_resolve.py:92` | **DERIVED** ✓ | inline: `ceil(log2(65504/0.5)) = 17` (fp16 max / `BETA_STEP0`) | fp16 dynamic range | **YES** | **CLEAN — exemplary.** The derivation is re-checkable from the literal in one line |
| **11** | `CANARY_MAX_ABS_ERR` | `1.2e-05` | `dc1_menu_sweep.py:79` | **MEASURED** ✓ | ms8's instrument floor **on this exact vehicle** | *"a canary above this means the harness is not reproducing the shipped decode and NO verdict from it is admissible"* | **YES** | **CLEAN — exemplary.** A measured instrument floor wired as a fail-closed admissibility gate |
| **12** | `GATE_D_POSE_V4B` | `0.06365131` | `ddm_v4c_resolve.py:83` → `:392` | **MEASURED-ANCHOR, observability only** | measured v4b gate mean (`evaluate.py` rc=0) | the **v4b** era | **N/A** — only 2 sites; `:392` writes it into a receipt as `v4b_gate_d_pose_ref`. It is **provenance, not control**, and the `_ref` suffix says so | **CLEAN.** Flagged then cleared — an era-stale *number* that is not an era-stale *gate* |
| **13** | `derive_ema_decay` | LawRef-resolved | `train_tr1_partition_renderer_mlx.py:380`, `spec_tr1_renderer_20260728.py:185` | **DERIVED-LIVE** ✓ | LawRef `ema_decay_run_geometry_v1`, called `1p/5t` | decay from **run geometry**, not a flat constant | **YES** | **CLEAN — this is the pattern the other rows should imitate.** The 0.997 legacy value survives only as a labelled fallback |
| **14** | `--token-quant-levels` | `16` | `train_tr1_partition_renderer_mlx.py:1925` | pinned at the row-9 coder ceiling | n/a | the coder's admissible max | **YES** | **CLEAN.** "16" here is not a calibration choice; it is the boundary of row 9 |

---

## 4. THE THREE ARMED LANDMINES, in detail

### 4.1 The 200,000-byte ceiling produces a SILENT zero budget (rank 1)

```python
# tools/run_ddm_v9_carrier_compose.py:2371
maximum_allowed_added = max(0, config.total_archive_ceiling_bytes - len(base_archive))
for requested_budget in config.added_budget_bytes:
    effective_budget = min(requested_budget, maximum_allowed_added)
```

At the live base of **353,805 B** this is **exactly 0**. Every ladder rung collapses to
`effective_budget = 0`; the composer selects only states with `exact_added_bytes <= 0` and admits
**nothing**. It does not raise. It does not refuse. It produces a run that reads as *"the ladder
flattened."*

Worse, the surface's own falsifier consumes that flatness:

```python
# :2513
plateau_falsifier = (... and maximum_budget_envelope_bytes == config.total_archive_ceiling_bytes
                     and byte_ceiling_nonbinding and flattened and ...)
```

A **starved** run and a **genuinely plateaued** run emit the same symbol. That is the
`a probe that cannot return the negative` genus, with the extra twist that the probe is the
*falsifier* — so the failure direction is toward a false KILL.

**Liveness, stated honestly:** `direct_description_carrier_compose.py` is **dormant and BLOCKED**
(fails its own tests) and last moved 2026-07-22, so **this is not a live defect**. It is armed for
whoever re-fires the composer against the current archive. `total_archive_ceiling_bytes` is a pydantic
`Literal[200000]` — it **cannot be overridden by config**; it must be edited.

### 4.2 `ANS_SCALE_BITS = 12` is a race handicap, not a live defect (rank 2)

The token coder that carries **99% of 353,805 B** imports the rANS prototype
(`ddm_r7_token_coder.py:50-57`), and that prototype quantizes all symbol probabilities to
`ANS_TOTAL_FREQ = 1 << 12 = 4096` with **no derivation anywhere in the file**. 12 is the value that
appears in the standard rANS reference implementations — a **generic basis**, in the operator's exact
sense.

**Scope, checked before claiming:** `AUTO_CODECS = ("smevr", "brotli11")` — **rANS is not in the
auto-selected set**, so it is not on the shipped decode. The live coder is `smevr`.

**Why it still matters:** coder races are a live activity (`ddm_cc2_coder_races.py`,
`ddm_cc3_mixed_coder_receiver.py`). A race that pits rANS against `smevr` at an **underived 12-bit
probability precision** measures the constant, not the codec — and returns *"rANS loses"* as if it
were a codec verdict. **Owed, and cheap:** derive the precision from the token-symbol distribution's
entropy (the quantization error floor is `~2^-SCALE_BITS` per symbol against a support already bounded
by `levels ≤ 16`) before any rANS row is admitted.

### 4.3 `thr_wall` is sized 3.27× above the bar it must clear (rank 3)

`thr_wall = 0.00025` with the inline note *"contribution 0.05 (the BINDING FORMULATION-scope wall
threshold)"*. Recomputed: `sqrt(10 × 0.00025) = 0.050000` ✓ — the derivation is sound and the note is
accurate.

But **0.05 is not the target any more**. The PR130 bar decomposes as seg `0.02966` + pose `0.015268` +
rate `0.127214` = `0.172142` (vs stated `0.172141`, +1e-6 rounding ✓). A wall declared at a pose
contribution of **0.05** sits **3.275×** above the bar's pose term; in `d_pose` units the wall is
**10.72×** looser than `2.3311e-05`.

**This is the charter's target class exactly** — a *correct derivation* whose *target input* is stale.
Nothing is wrong with the arithmetic; the question is whether the consumer wants "binding at 0.05"
(an absolute formulation-scope wall, which may be the intent) or "binding at the bar." **I did not
determine which**, so this is an owed adjudication, not a defect claim.

---

## 5. CLEAN ROWS — the primary result

Rows 8–14 are clean, and three are worth imitating:

- **`derive_ema_decay`** — a LawRef evaluated at config time from run geometry, with the old flat
  constant demoted to a labelled fallback. This is the shape every row in §3 should converge to.
- **`BETA_MAX_DOUBLINGS = 17`** — `ceil(log2(65504/0.5))`. The literal is re-derivable from its own
  comment in one line, by anyone, without running anything.
- **`CANARY_MAX_ABS_ERR = 1.2e-05`** — a measured instrument floor wired as a **fail-closed
  admissibility gate**: *"a canary above this means the harness is not reproducing the shipped decode
  and NO verdict from it is admissible."* Calibration that protects the instrument, not just the model.

And the honest headline for the operator's question: **on the live TR1 path, the calibration is
mostly derived, mostly documented in-line, and mostly re-checkable.** The `ms8`/`dc1`/`v4d` solver
constants in particular carry their arithmetic with them. That is not a naive surface.

---

## 6. OWED MEASUREMENTS (named, none run — `pu2` holds the scorer)

| # | owed | cost | falsifier / stop condition |
|---|---|---|---|
| O1 | **The name join** derivation-name ↔ constant-name, which converts §2.3's `329` upper bound into a defect count | `$0`, static | **Do not build a second join** — `ddm_pt2 §1` is building it for levers. Extend that one or record the blocker |
| O2 | Re-derive or delete `total_archive_ceiling_bytes` before the composer is re-fired | `$0`, one edit | KILL the concern if the composer is formally retired instead |
| O3 | Derive rANS `SCALE_BITS` from the token-symbol entropy | `$0`, static + existing coder-race harness | KILL if the coder race is scoped to never admit rANS |
| O4 | Adjudicate `thr_wall`'s target: absolute 0.05 wall, or bar-relative `2.3311e-05`? | `$0`, a read of the consumer | KILL if the consumer's wall is deliberately formulation-scoped |
| O5 | Re-measure `MS8_ARCHIVE_DELTA_BYTES` on the ix2 container, or re-scope it to ms8 | one byte-close, no scorer | KILL if `dc1_menu_sweep` is never run against an ix2-container base |
| O6 | Guard `_EXHAUSTIVE_CAP`'s certificate: `assert comb(n, n//2) <= _EXHAUSTIVE_CAP` | `$0`, 1 line + 1 test | none — this is a fail-loud conversion, not a value change |

**No ΔS is claimed for any of these.** None of rows 1–7 is on the shipped decode's numeric path today,
so **none of them can move `S = 0.7910689` by construction**. Their value is that each one currently
sits where a *future measurement* would silently read it as a result.

---

## 7. ROUND-1 ADVERSARIAL SELF-REVIEW

**My pre-registered likeliest failure was:** *grading a constant "derived" because a derivation exists
somewhere, without checking the derivation's assumptions still hold at today's operating point.*

It **fired**, twice, and both times against me:

1. **Instrument B is too weak for its own headline.** "0 frozen candidates on the live TR1 path" is
   produced by a test that scores the coordinator's canonical case **CLEAN** (`prod_calls=1`). I
   report the result *and* its blindness (§2.4), and the refined 329/463 figure that shows the real
   scale. **The clean live-path result is evidence, not proof.**
2. **My file list was wrong and I audited two dormant, BLOCKED modules as if they were live** — while
   **missing the entire decode closure**, including the coder carrying 99% of the bytes. Corrected in
   §2.5 and both sweeps re-run. Had I not re-scoped, the 200,000-byte ceiling would have been my
   rank-1 *live* finding rather than a correctly-scoped dormant landmine.

**A third failure I have NOT closed:** instrument A is **name-matched**. A calibration constant whose
identifier contains none of my knob tokens is invisible to it. I did not measure that miss rate, so
**§3 is a lower bound on the knob population, not a census.** Stating the 108-row count without this
caveat would be the same vacuity I criticise in §2.1.

**Negative-existence statements in this memo are scoped, not absolute:** every "no derivation exists"
in §3 means *did not find one in the 10,342-file bounded scope of §2.3, by name-match on the
derivation-name patterns listed there* — not that none exists.

---

## 8. VERDICT SCOPES

- `verdict_scope: INSTANCE — the 22-file live TR1 list of §2.5`: **6/6 derivation-named functions are
  called in production; zero frozen-output candidates.** Bounded by instrument B's power (§2.4).
- `verdict_scope: FORMULATION — scalar-returning derivations, 10,342-file bounded scope`: **329 of 463
  (71%) are never consumed outside their defining module.** This is an **upper bound on the
  frozen-output class**, not a defect count; the name join (O1) is what converts it.
- `verdict_scope: INSTANCE — direct_description_carrier_compose.py + run_ddm_v9_carrier_compose.py`:
  `total_archive_ceiling_bytes = 200000` yields `maximum_allowed_added = 0` at the live base and is
  consumed by the surface's own `plateau_falsifier`. **DORMANT + BLOCKED — armed, not firing.**
- `verdict_scope: INSTANCE — repair_entropy_coder_runtime_adapters.py:33`: `ANS_SCALE_BITS = 12` is
  underived and generic. **NOT on the shipped decode** (`AUTO_CODECS = ("smevr","brotli11")`).
- `verdict_scope: INSTANCE — ddm_p3v2_optimal_form_pose_resolve.py:606`: `thr_wall` is correctly
  derived for a **0.05** pose contribution, which is **3.275×** the PR130 bar's `0.015268`. Whether
  that is stale or deliberate is **UNRESOLVED**, not refuted.
- `verdict_scope: INSTANCE — direct_description_joint_descent.py:2281`: `margin_floor = 0.1` is
  **CLEAN** (1.047× the Lane-restricted p10; two independent routes agree). The hinge **weight** is
  the live lever and is **`ddm_mg1`'s**.

---

## 9. WHAT I DID NOT DO

- **No scorer slot, no n600 pass, no training run, no paid dispatch.** All six owed items are named.
- **I did not open the retired-vehicle surfaces** (`train_levelset_witness_realized_through_R_mlx.py`,
  `launch_witness_run.py`) — `ddm_pt2`'s scope.
- **I did not re-run `ddm_gd1`'s sweep.** Its rows (gate estimator ESTIMAND, `ema_decay`'s `U` binding,
  optimizer state at window boundary, `--gate-every 5`, `--lr`, uint8 rounding mode, gate seed) are
  the **CHOICE** class — subset/order/cadence/mode/reference-frame/window/estimand — which the value
  ladder structurally cannot see. **That axis is orthogonal to this one and is consumed, not
  duplicated.** `A1_SMOOTH_DROP_REL` / `A1_REALIZED_DROP_REL` are explicitly owned by `gc14 R2`.
- **I did not touch** the files of `pt2`, `mg1`, `dd1`, `hs1`, `as1`, `ss1`, `wf2`, `rs2`, `ph4`.
- **Task-ledger note (not a finding of mine, a re-confirmation):** of the prior-art ids in my charter
  (`#847`, `#874`, `#817`, `#686`, `#888`, `#875`), **only `#817` resolves** in
  `.omx/state/canonical_task_status.jsonl`. The others are in the harness TaskList, a **different
  store**. I consumed `#817` (= `ddm_gd1`) by content and cite the rest by content, never by bare id.
