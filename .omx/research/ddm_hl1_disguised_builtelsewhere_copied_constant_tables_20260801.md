# ddm_hl1 (#847 family) — the DISGUISED built-elsewhere-unwired form: constants copied out of their canonical producer

**Scorer-FREE (0 SegNet/PoseNet forwards). $0. APPARATUS + CUSTODY — no score claim.**
Pointer `0.1910828242` [contest-CPU] UNMOVED. All rows `[macOS-CPU advisory]`,
`score_claim=false`, `promotable=false`.

**STORES CONSULTED:** `tools/corpus_query.py "hardcoded constant copied from measured
artifact value provenance ladder canonical equation"` — loaded
`.omx/research/lawref_constant_compiler_351_20260708.md` (#351 LawRef constant compiler),
`.omx/research/hardcoded_run_constants_sweep_20260707.md` (#340, the sister run-constant
sweep — confirmed DISJOINT: it types RUN properties, not measured quantities),
`.omx/research/ddm_gd1_undecided_defaults_audit_20260731.md` (the ladder types VALUES),
`.omx/research/t5_crucible2/SPEC_v752_20260709.md` §E (bare-literal provenance = a bug
class, grading OWED), `.omx/research/ddm_sb2_complete_the_stubs_20260731.md` (the seed).
Deliberately NOT loaded: the burn-4/window_03 run state — this arm touches no run.

---

## §0 HEADLINE (answer first)

**1. The named instance is REAL and is now RESOLVED — and it had NOT drifted.**
`src/tac/optimization/lane_guard.py:64-65` held the ten frozen-SegNet head-pair normals as
bare float literals. Re-derived at source: they match
`tac.canonical_equations.segnet_head_rank4_flipdist_20260715:HEAD_PAIR_NORMS` **exactly**
(set equality, both tuples). So this was not yet a wrong number — it was an unlinked one,
which is the whole point: a copied literal LOOKS wired, passes review, carries a plausible
comment citing the memo it came from, and cannot track its source. The fix resolves both
tuples from the producer and is **byte-identical**: ratio `1.1960730088495577`, tuples
identical, `dw_norm` default `4.007` unchanged, all asserted by test.

**2. A copied TABLE is detectable; a copied VALUE is not.** The first sweep cut
(">=3 decimal digits") returned **2,220 collisions, essentially all noise** — `0.999` is
every Adam beta2 in the repo, not a copy. The discriminating signature is
**>=3 DISTINCT values of ONE canonical constant in a file that never names the producer**.
That took 253 unlinked (file, constant) groups down to **3**, and `lane_guard` ranked
**first of 224 files**, independently reproducing the seed's claim by mechanism rather than
by citation.

**3. The gate I built to catch the class had the exact bug the class is made of.**
My first cut matched exclusion markers against the ABSOLUTE path, so any checkout under a
directory named `tests`/`test_*` — including every pytest tmpdir — skipped **every file**
and reported the same clean symbol as a genuinely clean tree. It was caught only because
the positive controls RAN and failed. This is the vacuity class
(`vacuity_is_indistinguishable_from_pass`) reproduced inside its own countermeasure, one
day after that memory was written. Fixed, and asserted against by a named regression test.

**4. The module I extended is itself unwired — and preflight cannot take it.**
`tac.run_constant_gates` has had **ZERO consumers outside its own tests since 2026-07-07**;
its docstring deferred the wire-in to "once the sibling's preflight.py lands" and that never
happened (25 days). MEASURED today: `preflight_all(check_codebase=True)` costs **28.1 s
against the 30.0 s `DEFAULT_PREFLIGHT_CLI_TIMEOUT_S` — 94 % of budget**, and the new check
costs 4.4 s. **No new gate can be wired into `preflight_all` right now.** That is a
cross-finding for every other arm on this P0. The debt is registered machine-readably
rather than left as prose (§4).

---

## §1 PROVENANCE

| item | value |
|---|---|
| venv | `/Users/adpena/Projects/pact/src/tac/__init__.py` — hijack check **CLEAN** |
| scorer jobs | **0** |
| tests | 27 new, all passing; 79 passing across the touched suites; ruff `--select F` clean |
| pointer | **`0.1910828242` [contest-CPU] UNMOVED** |

**RE-DERIVED at source (not taken from the seed):** the `lane_guard:64-65` literals; the
canonical `HEAD_PAIR_NORMS` and `HEAD_CENTERED_SINGVALS`; the raw artifact
`experiments/results/segnet_fractal_20260715/stage_a.json`; `LANE_BUDGET_S_UNITS` against
`xp1_verdict.json`; the ckpt sha256 the docstring cites; the `preflight_all` wall clock and
budget; the `run_constant_gates` consumer count.
**NOT verified (labelled):** `EROSION_S_MEASURED = 0.00151` — see §3.
**ASSUMED:** that `.omx/research/segnet_recursive_fractal_factorization_20260715.md` §2 is
the memo the docstring means; I did not re-read the memo, I went to the artifact it cites.

---

## §2 THE THREE SITES (the sweep result, per-site)

Denominator of the sweep: **233 canonical-equation modules parsed → 754 module-level float
literals → 450 with >=3 significant digits → 425 distinctive (single owning module)**;
scanned **5,794 live `.py` files** under `src/tac` + `tools` + `experiments` (excluding
tests, `experiments/results/`, intake clones, vendored trees) containing **59,992 float
literals**. Unlinked (file, constant) groups: **253**. Copied TABLES (>=3 distinct): **3**.

| # | site | what was copied | live? | drifted? | direction of error |
|---|---|---|---|---|---|
| 1 | `src/tac/optimization/lane_guard.py:64-65` | all 10 `HEAD_PAIR_NORMS` | **YES** — tr1 trainer under `if cfg.lane_guard` (`train_tr1_partition_renderer_mlx.py:1865`) | **NO** — exact set match | **CONSERVATIVE.** Derived ratio 1.1960730088495577 vs full-precision-artifact 1.196155096510409: literal is **LOW by 6.9e-5 relative**. The ratio scales Lane protection weight, so the 3-decimal rounding **under-protects**; it under-states an intervention, never over-states a counted quantity. **FIXED** (byte-identically — the conservative bias is preserved deliberately, see §3). |
| 2 | `tools/pool_channel_jacobian_rd_harness.py:39` | 4 of 5 `HEAD_CENTERED_SINGVALS` as `SETTLED_SINGULAR_VALUES` | tool; gate-like | **NO** — equals canonical`[:4]` | **FAIL-CLOSED / SAFE.** Used as an ASSERTION TARGET (`verify_settled_singular_values(..., atol=5e-4)` at `:462`), so a re-measured head that drifted would REFUSE, not silently miscompute. Max literal-vs-artifact error **3.762e-4 against atol 5e-4 = 25 % headroom** — the tolerance was evidently sized for 3-decimal rounding. Residual risk is someone widening `atol`, not the copy itself. **REPORTED, not fixed** (other arm's file). |
| 3 | `tools/costate_organ_v2_backtest.py:110-113` | 3 of 5 `MEASURED_TOP_MARGINALS_EP900` | standalone backtest, **no consumer found** | **NO** | **CONSERVATIVE, partial.** A 3-of-5 copy (`chroma_boundary`, `thin_lane` absent); the file's own comment says unmapped variants "remain 0/UNIDENTIFIABLE", so the partial copy under-counts. A partial copy is exactly how a table drifts silently. **REPORTED, not fixed.** |

Sites 2 and 3 are **REPORTED, not fixed**, following the ddm_sb2 precedent: they are
chartered files owned by other arms, and refusing the tree for debt this arm does not own
inverts the atomicity rule's intent. Both are named in the gate's live count.

---

## §3 A SECOND, DIFFERENT GRADE IN THE SAME FILE (the sweep is blind to it)

`lane_guard.py` carries two more measured literals that have **no canonical-equation home
at all**, so the sweep cannot see them by construction. Both re-derived by hand:

| constant | citation | verification today |
|---|---|---|
| `LANE_BUDGET_S_UNITS = 0.12589` (`:59`) | `xp1_verdict.json` `base_per_class_S_units[1]` | **MATCHES EXACTLY (diff 0.0).** The cited `ckpt_sha256` also matches. But the artifact lives at `/Volumes/VertigoDataTier/pact/ddm_xp1_20260731/` — **outside the repo, on an external volume**. No gate can check it and it can vanish; it is unverifiable on any other host. |
| `EROSION_S_MEASURED = 0.00151` (`:68`) | docstring says "xp1 (task #808 brief)" | **NOT VERIFIABLE.** The value does not appear anywhere in `xp1_verdict.json` (all 23 keys enumerated). The citation is prose, not a machine-readable artifact. Ladder class 4 with **no typed `HardcodedWaiverCustody`**. |

This is a distinct grade from the copied table — **artifact-cited literal whose artifact is
out-of-repo or prose-only** — and it is strictly harder to detect, because there is no
canonical constant to collide with. **Verdict scope: INSTANCE** (two constants in one
file). I did not sweep for it: doing so honestly requires indexing measured artifacts, not
canonical modules, and the artifact set is unbounded and partly off-volume. Naming the
boundary is the honest deliverable; the sweep's denominator above is the canonical-equation
surface **only**.

**On the residual rounding in site 1:** the fix keeps the canonical module's 3-decimal
values rather than the full-precision artifact, because the module IS the registered
source of truth and matching it preserves byte-identity. The 6.9e-5 conservative gap is
therefore RECORDED, not closed. **Re-derivation trigger:** close it to full precision the
next time the lane guard is re-tuned, as a measured A/B — never as a silent edit, since it
moves a shipped weight.

---

## §4 WHAT WAS BUILT

### The fix — `src/tac/optimization/lane_guard.py`

Imports `HEAD_PAIR_NORMS` from the canonical producer and derives both tuples by
descending sort (order pinned so the derived mean is bit-reproducible). Adds a **fail-closed
shape canary**: if the producer ever reports other than 10 pairs / 4 Lane pairs, the module
REFUSES at import rather than silently averaging the wrong set. Marginal import cost +121 ms
/ +271 modules, paid only inside `if cfg.lane_guard` — negligible against a multi-hour run.

### The guard — `check_no_canonical_equation_constant_copied_as_literal`

A **scope extension of the existing #340 `run_constant_gates` surface**, deliberately NOT a
new catalog number (post-#400 Catalog #299 consolidation rule). Same module, same waiver
grammar (`# CANONICAL_CONSTANT_COPY_OK:<rationale>`, placeholder rationales rejected), new
pattern **P5**. **LIVE COUNT 2, WARN-ONLY. Flip condition: live count 0**, i.e. sites 2 and
3 each gain a link or a real waiver.

Two thresholds, both MEASURED rather than chosen:

* **>=3 distinct values of one constant** — the copied-table signature. Single-value
  collisions: 347 (noise). Tables: 3.
* **>=4 significant digits** to be indexed — a measurement, not a rounded convention.
  Measured on the live tree: at 4 every real finding is identical while the candidate index
  drops 425→278 and the files needing an AST parse drop from 61 % to 9 %; **at 5 both
  remaining findings are LOST**, so 4 is the boundary, not a round number.

Wall clock **4.4 s** (from 25 s: excluded a vendored `.venv`/site-packages tree the first
cut was reading, and prefiltered). A richer prefilter that applied the table threshold at
the text level was built and **MEASURED SLOWER by >10x** (`finditer` must walk every match
of short digit sequences across ~100 MB) — reverted, and the reason is recorded in the
docstring so it is not rebuilt.

### 27 tests — behaviour, not constants

Positive controls (including **the real pre-fix `lane_guard` shape rebuilt against the REAL
canonical constant** — a detector that cannot reproduce its own founding case is not a
detector), the scientific-notation spelling class (`0.0004203` is written `4.203e-4`, which
a repr-substring prefilter would silently miss), negative controls, waiver respect,
placeholder-waiver rejection, the absolute-vs-relative-path vacuity regression, and a
mutation guard asserting `describe()` quotes THIS finding (a canned marker string fails).

### Registered debt

`.omx/state/required_component_ledger.jsonl` ← `RunConstantGatesConsumerWiring`, fire order
2, **owner UNASSIGNED — needs an owner in the next dispatch**, with the measured budget
blocker and an explicit clearing trigger.

---

## §5 CROSS-FINDINGS

* **To every arm on this P0:** `preflight_all(check_codebase=True)` = **28.1 s / 30.0 s
  budget (94 %)**. Any new gate wired there pushes it over. Measure before wiring.
* **To the ledger owner:** `ddm_sb2` §2 names a fifth grade **BUILT-ELSEWHERE-UNWIRED-HERE**
  as "the grade nothing detects", but `activation_ledger.VALID_BUILD_GRADES` does not carry
  it — the taxonomy in the memo and the taxonomy in the code disagree, so the grade cannot
  be recorded even when correctly diagnosed. Registered under `built-never-fired` with the
  real grade in `notes`.
* **Pre-existing, not mine:** `preflight_all` raises `CodebaseDriftError` on this tree over
  ~18 `experiments/launch_*` ad-hoc launchers. Confirmed present independent of these edits.

---

## §6 VERDICT SCOPES

* Copied canonical-constant tables in `src/tac` + `tools`: **swept, 3 found, denominator
  declared**. Verdict scope **FORMULATION** — this covers constants owned by a canonical
  equation module; it says nothing about constants copied from artifacts with no canonical
  home (§3), which is a strictly larger and partly off-volume surface.
* "No other copied tables exist": **NOT claimed.** The claim is "no other copied table was
  found in 5,794 files / 59,992 float literals against 425 distinctive canonical values,"
  which is a scoped negative, not a negative-existence claim.
* `experiments/` is out of the gate's standing scope (a trainer is the DSL's compile target,
  per the module's own P1-P4 rationale) — but the one-off sweep **did** include it and found
  nothing there.

## §7 POINTER HONESTY

**`0.1910828242` [contest-CPU] UNMOVED.** Nothing here lowered the exact score. This is
MEANS. The justification is specific: a literal copied out of a measured artifact is the
form of orphaned signal that *survives review*, and the one live instance sits in a guard
that shapes per-pixel Lane loss weight on the tr1 trainer.
