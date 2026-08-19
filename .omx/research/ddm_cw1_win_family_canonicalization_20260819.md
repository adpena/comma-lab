# ddm_cw1 — the WIN-FAMILY CANONICALIZATION WAVE (task #1143)

**Operator directive 2026-08-19, verbatim:** *"all of the win classes and families are
generating signal we can use to create more canonical and standardized and master and
comprehensive versions including using neural and geometric means and more"*.

**Axis** `[macOS-CPU exact byte / advisory]`. **No score is produced or implied.**
`score_claim=false`, `promotable=false`. Only `upstream/evaluate.py` on contest hardware
is a score. The own-vehicle frontier is **UNMOVED** by this arm — this is apparatus, and
saying otherwise would be the means-as-ends violation.

---

## 0. STORES CONSULTED

| Store | What I read | What it changed |
|---|---|---|
| `src/tac/` module tree | `gt_lineage.py`, `gt_lineage_registry.json`, `contest_score.py`, `canonical_equations/`, `witness_dsl/curriculum_dsl.py` | **Decisive.** Two canonical surfaces already existed (§2.1) and one intended wiring was a category error (§6). |
| `.omx/research/` (corpus sweep) | 24 arm memos + receipts across up2/up3/jg1/jg2/me1/tq1/qs5/ck2/to1/fx1/ma1/pi2/ra3/na10/gl1 | Every measured constant in this memo. |
| `experiments/` sibling sources | 7,990 lines across 11 arm scripts — read the CODE, not the memos | Found the up2 filename-lineage defect (§2.1) that no memo records. |
| `.omx/state/canonical_task_status.jsonl` | grepped for 1103 / 969 / 1142 / 1143 | **Zero rows.** Ledger was DOCUMENT-ABANDONED 2026-07-01; numeric IDs live only in the DAG + memo prose. Charter's `#1103` (me1) and `#969` (tq1) are **unverifiable as ledger rows**; I located those arms by name instead and say so rather than pretend the citation resolved. |
| `.omx/state/canonical_equations_registry.jsonl` | checked for existing win-family laws | None of the three laws in §5 was registered. |
| MEMORY.md + CLAUDE.md | anti-orphan (#936), vacuity==pass, cross-regime constant transfer, prefix bias, denominator/falsifier vacuity | Each became an enforced behaviour, not a comment (§3). |

---

## 1. SIBLING INVENTORY — what was rebuilt, and how many times

Read from source. Line counts are the arm scripts as they stand.

| Family | Siblings (LOC) | Shared mechanism | Divergence that is REAL (kept) |
|---|---|---|---|
| **F1** realized-acceptance lattice descent | `ddm_up2` (1,126) · `ddm_jg1` (711) · `ddm_tq1` (1,782) · `ddm_me1` (145) | propose → apply → **REAL decode** → accept iff realized joint ΔS < 0; stop = no-improving-neighbour sweep | up2 moves 12 int12 coefficients; jg1 rewrites GT-labelled disks in a 384×512 token map. Move sets are family math, not boilerplate. |
| **F2** terminal joint compile | `ddm_jg1` S2 · `ddm_jg2` · `ddm_qs5` (1,057) | seg edit → carrier re-solve vs **edited** frames → compensation → rate re-encode → container search | qs5's Schur compensation is its own algebra; only its GT-lineage declaration is canonicalised. |
| **F3** container / re-encode | `ddm_up3` (863+284) · `ddm_ck2` (293) · `ddm_to1` (607) | encoder-only knobs searched at the **archive** layer, identity-controlled | ck2 acts on the semantic body, to1 on the tail, up3 on the carrier. Same knob space, different sections. |
| **F4** model-axis recoder | `ddm_fx1` (785) · `ddm_ma1` (337) | probability model changes; payload and container fixed | fx1's dyadic-radical mixer and ma1's KT miss-ratio are different estimators. Untouched. |
| **F5** local authority instruments | `ddm_up2` pose gate · `ddm_jg1` seg gate | frozen CPU scorer + GT lineage + score arithmetic + seeded pair draw | None. This one was pure duplication. |

**Measured worth of the families** (all `[contest-CUDA T4 n600]` unless noted):
F1 → up2 d_pose 7.769484e-06 → 7.649247e-06 at **ΔB = 0**, 429 pairs improved / **0 worsened**;
tq1 phase-B ΔS −1.88043296e-04 at Δbytes +1.
F3 → ck2 **−657 B** + to1 **−105 B** = **−762 B = −5.074e-04 S** at zero distortion.
F4 → fx1 **−560 B** code length (+127.3 s decode vs 297.7 s headroom); ma1 **−100.4 B**.
F5 → up2 pose gate **0.99993×** of T4; jg1 seg gate **0.99995×**.

---

## 2. THE TWO FINDINGS THAT CHANGED THE DESIGN

### 2.1 `tac.gt_lineage` already existed — and neither live arm consumes it

`src/tac/gt_lineage.py` landed **2026-08-16** (`ddm_gl1`) with a **content-addressed**
registry after gl1 censused 68 reachable GT files / 48 distinct sha256 (16 DALI, 39 PyAV,
13 UNKNOWN) and found **seven files named `gt_argmax_n600.npy` spanning three distinct
sha256 and BOTH lineages**.

`ddm_up2`, written **2026-08-19**, resolves lineage by filename substring:

```python
lineage = LINEAGE_DALI if "dali" in path.name.lower() else "unknown_pt"
```

That is exactly the laundering gl1 was built to refuse — a name extending one verified
file's reputation to six unverified ones. `ddm_jg1` imports up2 and inherits it.

This is the **#936 orphan pattern live**: the canonical cure existed for three days and
the two arms that most needed it each wrote their own weaker version. It is why F5 is
built as a **composition of** `tac.gt_lineage` rather than a fourth implementation, and
why the parity suite asserts the up2 defect string is still present — so that if up2 is
fixed, the test fails and the rationale gets re-examined instead of silently rotting.

### 2.2 `tac.contest_score` already owned the score arithmetic

`seg_term` / `pose_term` / `rate_term` / `compute_contest_score` were already canonical,
including a guard against the `×25` slip a subagent made on 2026-06-23. F5 **delegates**
to them and adds only what was missing. Nothing was re-derived.

---

## 3. WHAT LANDED

| Family | Module | Tests | What it standardises |
|---|---|---|---|
| **F5** | `src/tac/local_contest_instruments.py` | 45 + 33 parity | axis↔lineage binding (DERIVED from `evaluate.py:31-42` + the two `frame_utils` asserts) · PyAV pose-absolute **refusal** · `InstrumentReceipt` with `score_claim`/`promotable` as `init=False` · seeded-random pair draw · cached CPU scorers |
| **F1** | `src/tac/win_families/realized_acceptance.py` | 65 (with rankers) | the descent loop, the convergence proof, per-coordinate atomic checkpoint/resume, and the free labelled corpus |
| **F1 rankers** | `src/tac/win_families/proposal_rankers.py` | *(in the 65)* | geometric + neural proposal ordering, with the never-accepts boundary enforced |
| **F3** | `src/tac/win_families/container_optimizer.py` | 37 | sealed option space · tie-to-incumbent · parse-back-before-win · archive-vs-payload attribution |
| **F2** | `src/tac/win_families/terminal_compile.py` | 59 (with F4) | stage input/output contracts · **staleness at consumption** · GT-lineage gate on compensation · measured-vs-modelled rate leg |
| **F4** | `src/tac/win_families/model_axis.py` | *(in the 59)* | deflated-reservoir accounting · calibration-with-provenance · ceiling-vs-realized separation |

**239 tests, all passing.** Every module's central claim is an **executed** control, not an
assertion about one.

### The memory-derived behaviours that are now code, not comments

* **vacuity ≠ pass** — `DescentReport.converged` returns **False** on an empty run
  (`all([])` is `True`, so a run that measured nothing would have reported itself
  converged). `CompilePipeline.assert_fresh` refuses when a stage never ran.
* **cross-regime constant transfer** — `model_axis.Calibration` requires a `source` AND a
  `regime`, and refuses to apply outside that regime without an explicit
  `allow_cross_regime=True`. This is ma1's own scar: at `mc=32` an inherited constant
  produced a **FALSE SATURATION VERDICT**.
* **prefix bias inverts by axis** — `select_pairs` never returns `[0..n)`; pose prefixes
  measure 2.54–4.21× harder, seg prefixes 0.95–0.97× easier.
* **the denominator can be vacuous** — `MarginSaliencyRanker` refuses without an incumbent
  to diff against, because otherwise every proposal scores the field minimum and the
  "geometry" is a constant wearing a geometry's name. *(This was a real defect in my own
  first draft, caught in review pass 1.)*
* **modelled ≠ measured** — `RateLeg.assert_measured()` refuses a modelled rate on the
  certifying path (§4.2).

---

## 4. CANONICAL-vs-UNIQUE DECISION PER FAMILY

Per UNIQUE-AND-COMPLETE-PER-METHOD: the **loop, the accounting, and the refusals** are
canonicalised; **every family's math is left alone**.

| Layer | Decision | Rationale |
|---|---|---|
| GT lineage resolution | **ADOPT_CANONICAL** (`tac.gt_lineage`) | Content-addressed and already measured-correct (§2.1). Rebuilding it is the defect. |
| Score arithmetic | **ADOPT_CANONICAL** (`tac.contest_score`) | Already canonical, already guarded. F5 delegates. |
| Axis↔lineage binding | **NEW** | Nothing owned it. DERIVED from upstream's assert-enforced bijection. |
| Pose-absolute policy | **NEW** | Nothing owned it. Two paid refusals (`ddm_ps1u` r2 +1.686e-02 S, `ddm_t1h` +0.012557 S) were bought by its absence. |
| Descent loop / stop rule | **CANONICALISE** | Identical across four arms; the divergence is the move set, which stays injected. |
| Proposal move sets | **FORK_PRINCIPLED** | up2's int12 neighbours and jg1's GT-labelled disks are different mathematics. Only the generic integer-lattice generator is shared. |
| Objective units | **CANONICALISE to SCORE UNITS** | up2 descends d_pose, jg1 counts flips. Incomparable until both are priced in S — which is what makes "accept iff realized ΔS < 0" one rule instead of four. |
| Coder / mixer / context model | **FORK_PRINCIPLED** | fx1's dyadic-radical mixer and ma1's KT ratio are family math. F4 canonicalises only the accounting both got burned by. |
| Compensation algebra | **FORK_PRINCIPLED** | qs5's Schur solve is its own. Only its lineage declaration is canonicalised. |

### 4.0 A defect in my own first draft, caught by an existing gate

My first F3/F4 draft **hand-rolled the rate term** (`25 * bytes / 37_545_489` with a local
denominator) while this memo claimed full delegation to `tac.contest_score`. Catalog #391
(`check_no_hand_rolled_contest_score`) flagged 8 lines across my files. Two were real
arithmetic; six were docstring prose that happened to match the text scan.

I fixed the **code** rather than waiving the gate: `bytes_to_score` and
`projected_score_delta` now call `contest_score.rate_term` (carrying the sign outside the
magnitude, since `rate_term` takes a non-negative size), and both modules re-export
`UNCOMPRESSED_SIZE_BYTES` instead of redeclaring it. **cw1 violations: 8 → 0.** The
consumer proof still reproduces `7ce46fd7…` at −48 B afterwards, so the delegation is
byte-neutral.

This is worth recording because it is the exact failure the wave exists to prevent — a
memo asserting canonical adoption while the code quietly forked — and it was caught by
apparatus, not by me.

### 4.1 Where the canonical is STRICTER than the sibling

* **F3 measures `len(archive.zip)` directly.** up3's loop minimises the *carrier stream*
  length. On this body the two agree (every other section is fixed), but the stream is a
  proxy and the archive is the counted quantity.
* **F3 seals the option space.** up3 named the laundering risk in its own memo and relied
  on discipline; `ContainerSpace.seal_digest` makes growing the space visible.

### 4.2 Where the canonical carries a CORRECTION to a sibling

* **jg1's rate leg was MODELLED.** jg1 modelled **+4.718 bits/changed token**; jg2 then
  measured **+30 B at 4.1379 bits/token**. That moved the headline from −0.0104 to an
  honest gap of 0.006526. `RateLeg` refuses a modelled leg on the certifying path.
* **qs5's `GT_POSE` is the PyAV table.** The shipped object is scored on the CUDA axis
  against DALI. `CompilePipeline` refuses a PyAV-fed compensation on a CUDA target.

---

## 5. THE NEURAL + GEOMETRIC PROPOSAL LAYER

**The boundary that makes a learned ranker admissible:** a ranker **orders and truncates**;
it **never accepts**. Acceptance stays realized-only inside the engine, which re-scores
whatever survives the ranker through the real decode. The worst a bad ranker can do is
waste realizations or reach a different local optimum. This is proved by an executed
**negative control**: an adversarial ranker that puts the *worst* candidate first still
reaches the exact optimum.

Truncation is the one honest caveat — with `top_k` set, "no improving neighbour" becomes
"no improving neighbour among the k kept", so `RankerConfig.convergence_proof_weakened`
carries that downstream.

| Ranker | Kind | State |
|---|---|---|
| `IdentityRanker` | control | **WIRED.** Every ranker claim is measured against it; it is also the only config where the convergence proof is unweakened by construction. |
| `MarginSaliencyRanker` | geometric | **WIRED.** Ranks by the minimum scorer margin over the cells a proposal changes — the margin field IS the Fisher surrogate (Pearson 0.978). Minimum, not mean: one near-flipping cell is the opportunity and a mean would hide it. |
| `JacobianConditioningRanker` | geometric | **WIRED.** Orders coefficient moves by ‖column‖ of d(objective)/d(state) — up2's `conditioning_report` read as an ordering. Recovers the moved slot from the generator's own `dN+M` label. |
| `LearnedRanker` | neural | **INTERFACE ONLY, and it refuses.** Constructing one without a model raises. |

**Training is out of scope for this arm and is stated, not stubbed.** No model is fitted
and no ranking quality is claimed for the learned path. What IS built is the corpus
emitter: every accept **and reject** is an `AcceptanceEvent`, and
`training_table_from_events` turns a run's log into `(features, labels)` where the label is
the realized ΔS. The realizations already happened to satisfy the acceptance rule, so the
corpus costs nothing extra — a measured descent emits ~66 labelled examples per coordinate
in the smoke, of which the large majority are rejects (a ranker trained only on accepts
never learns what a bad move looks like).

---

## 6. REGISTRATION

**Three canonical equations registered** and queryable via
`tools/list_canonical_equations.py --json`:

| equation_id | Law | Anchors | max residual |
|---|---|---|---|
| `cw1_realized_acceptance_monotonicity_v1` | realized-only acceptance ⇒ **zero** worsened coordinates | up2 n600 (429 improved / 0 worsened) · tq1 phase-B (8 of 12, ΔS −1.88e-04) | 0.0 |
| `cw1_gt_lineage_additive_pose_offset_v1` | `d_pose_pyav = d_pose_dali + C`, **C = 1.4061e-04** on `0.mkv` | ra3 closure (6.88e-06 + C = 1.474900e-04 vs 1.4747e-04, 0.014%) · up2 finalize cross-price | 8.67e-08 |
| `cw1_container_archive_vs_payload_delta_v1` | archive ΔB − payload ΔB = container term | up3 (+7 bits → +48 B, term 47 B) · **this arm's consumer proof** (−48 B, sha match) | 0 |

Each carries its own excluded-domain list. The pose-offset law explicitly **excludes**
per-pair use (the per-pair ratio `C/d_dali` spans 0.887–1,627, so 19.09× is a population
median, not a conversion) and **excludes** any other source video.

**DSL Lever: deliberately NOT created, and here is why.** `tac.witness_dsl.Lever` is a set
of *witness-trainer flag overrides* plus `epochs_delta`. **No win family exposes a
witness-trainer flag** — all five act downstream of training, on the archive, decode, and
coder surfaces. A Lever whose overrides no trainer consumes is precisely the inert-flag /
config-orphan class the DSL gates exist to refuse, so creating one would be fake wiring.
The F3 swept knobs are held by `ContainerSpace`, which is family-native and carries a seal
digest — a stronger provenance device than an override dict for this surface. **If MAIN
disagrees, this is the one place to push back.**

---

## 7. CONSUMER WIRING — the anti-orphan proof

### 7.1 F3: byte-identical, on the real body

`experiments/ddm_cw1_container_consumer_proof.py` drives up3's **actual** build path
through the canonical optimizer. `ddm_up3`'s file is READ, never modified; the per-config
drive rebinds `up3.CONTAINER_OPTIONS` in-process for one compile and restores it.

Receipt: `.omx/research/ddm_cw1_container_consumer_proof_20260819.json`

```
identity control     PASS   shipped codes + shipped config -> sha 50e56145... @ 176,420 B
determinism control  PASS   double compile byte-identical
search               8 declared configs, 8 admissible (all parse back exactly)
  incumbent  (interleave ON,  q11/lgwin24)  176,468 B
  winner     (interleave OFF, q10/lgwin16)  176,420 B   sha 7ce46fd7...
  delta                                        -48 B  =  -3.196123e-05 S
```

The winner is **byte-identical to the shipped thirteenth-move pointer archive**.

One thing this surfaced that up3's memo does not record: **configs 5 and 6 TIE at
176,420 B**, and only the lower-index one reproduces the shipped bytes. The deterministic
tie-break is load-bearing, not decorative.

### 7.2 F5: proven drop-in for both live arms

`src/tac/tests/test_win_family_f5_arm_parity.py` (33 tests) imports `ddm_up2` and
`ddm_jg1` and checks the canonical returns the **same answers on the same inputs**:
score arithmetic across 3 rows (abs 1e-15), `pose_leg`, `pose_report_bound` (abs 1e-18),
`resolvable_d_pose_floor`, the per-byte rate, `select_pairs` across 6 sizes × 3 seeds,
`d_seg_per_pair`, and both arms' exchange rates (`S_PER_SEG_CELL`, `S_PER_ARCHIVE_BYTE`).

Adoption cannot be proved by migrating the arms' imports today — they are in flight. It is
proved the other way round, and any future divergence now fails a test instead of forking
silently.

### 7.3 F1 / F2 / F4

Wired to their laws and to F5, and exercised by executed controls, but **not yet migrated
into a live arm** — the arms that would consume them are the ones I may not edit. These are
the adoption rows in §8. I am labelling this plainly rather than counting a test suite as a
consumer.

---

## 8. ADOPTION ROWS FOR MAIN

Ordered by value. None is urgent; all are boundary-time choices.

| # | Row | Why | Risk |
|---|---|---|---|
| **A1** | Point `ddm_up2.load_gt_poses` at `tac.gt_lineage.assert_gt_lineage` | Removes the filename-based lineage inference (§2.1). Highest-value single edit in this memo. | Low — the DALI cache must be in the registry; if it is not, that absence is itself the finding. |
| **A2** | Re-export F5 from the arm scripts (`from tac.local_contest_instruments import ...`) | Kills the duplication; parity is already proven (§7.2). | Very low — byte-for-byte equivalent by test. |
| **A3** | Route `ddm_jg2`'s chain through `CompilePipeline` | Gets staleness-at-consumption on the sub-0.15 chain, where a stale carrier is the live risk. | Low, but jg2 is in flight — boundary only. |
| **A4** | Re-run the ck2 / to1 container wins through `search_container_space` | Two more sealed receipts; would confirm the −657 B and −105 B under the canonical measurement (archive, not stream). | Low, $0, local. |
| **A5** | Have `ddm_qs5` declare its compensation lineage | The PyAV `GT_POSE` defect becomes a refusal instead of a footnote. | Low — may refuse immediately, which is the point. |
| **A6** | Fit the `LearnedRanker` from banked `AcceptanceEvent` corpora | Needs runs emitting events first (A2/A3 enable it). Advisory-only by construction. | Low — a bad ranker cannot corrupt the state, only waste realizations. |
| **A7** | Reconcile the abandoned numeric task ledger | `#1103` / `#969` did not resolve; numeric IDs live only in DAG prose. | Not mine to decide. |

---

## 9. HONEST LIMITS

1. **The frontier did not move.** This is apparatus. Its value is entirely in what the next
   arm does not have to rebuild.
2. **F1, F2 and F4 have no live consumer yet** (§7.3). Only F3 and F5 have real ones.
3. **The learned ranker is untrained** and refuses to construct without a model. No ranking
   quality is claimed for it.
4. **`#1103` and `#969` did not resolve** to ledger rows. I located `ddm_me1` and `ddm_tq1`
   by name; those two rows in the sibling table rest on the memos, not on the cited IDs.
5. **The synthetic-objective controls are synthetic.** The engine is proven to reach a known
   optimum on a toy objective and to be safe under an adversarial ranker; it has **not** yet
   been run against a real scorer inside this canonical form. F3's consumer proof IS on the
   real body and real bytes; F1's is not yet.
6. **`ma1` carries a SUPERSESSION block** appended by `ddm_rv14f` on 2026-08-19. I read the
   original; F4's accounting surface does not depend on ma1's headline number, only on its
   false-saturation mechanism, which the supersession does not touch. A reader building on
   ma1's −100.4 B should read the supersession first.

---

## 10. RECEIPTS

Commits (this arm, all via the serializer, no AI attribution):

| Commit | Contents |
|---|---|
| `b7f9701323` | F5 instruments + F1 engine + ranker layer (+110 tests) |
| `e14db28ace` | F3 optimizer + consumer proof (+37 tests) |
| `1327ea85dc` | F2 pipeline + F4 reservoir (+59 tests) |
| `3e53e09c72` | 3 canonical equations + F5 arm parity (+33 tests) |

Artifacts: `.omx/research/ddm_cw1_container_consumer_proof_20260819.json` ·
3 rows in `.omx/state/canonical_equations_registry.jsonl`.

**Own-vehicle frontier: S 0.15652626435208142 @ 176,420 B, archive `7ce46fd7…`
`[contest-CUDA T4 n600]` — UNMOVED by this arm.**
