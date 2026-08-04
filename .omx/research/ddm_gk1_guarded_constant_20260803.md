# ddm_gk1 — GuardedConstant: a constant that resolves at runtime and REFUSES misuse

**Arm:** `ddm_gk1` (third custodian; two predecessors lost to session limits) · **Date:** 2026-08-03
**Axis:** apparatus. `score_claim=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`.
**Pointer:** UNMOVED (0.1910828242). This is MEANS, not the END (CLAUDE.md THE GOAL means/ends firewall).

## Operator directive (verbatim, 2026-08-03)

> "We could build a new class that resolves to a constant at runtime that includes
> protection against being used where inappropriate."
> "To help protect against confounding and also not fail silently."

## §0 — Status board

| step | item | state |
|---|---|---|
| 1 | survey the existing surfaces (#351 LawRef, value-provenance ladder) | DONE |
| 2 | locate + re-derive the three measured instances | DONE |
| 3 | the class (`tac.witness_dsl.guarded_constant`) | DONE |
| 4 | four positive controls | DONE — 29/29 green |
| 5 | migrate `margin_floor` (byte-identical, PROVEN) | DONE |
| 6 | the gate (landing 2, P6, warn-only ratchet pin) | DONE |
| 7 | **P6 anti-vacuity + its 14 tests** | DONE (this custodian) |

## §1 — Corrections to the brief (MEASURED, source inspection)

1. `derive_margin_floor` is at **`src/tac/optimization/lane_guard.py:547`**, not
   `src/tac/witness_control/lane_guard.py:547`. There is no `witness_control/lane_guard.py`.
2. The frozen literal was **`src/tac/optimization/direct_description_joint_descent.py:2281`**,
   `margin_floor: float = 0.1` on `DirectDescriptionJointDescentMLXModule.__init__`.

## §2 — Why the parent surface (#351 LawRef) does not already cover this

`src/tac/witness_dsl/lawref.py` gives the 4-rung ladder, `config_tags` conditionality, artifact
`sha256` integrity, artifact-mtime `max_staleness_days`, and a fail-closed resolve. What it does
**not** carry — exactly the three instances measured 2026-08-03:

| axis | LawRef today | the instance it misses |
|---|---|---|
| STALENESS | artifact **mtime** only | a **derivation that exists in-repo and is never called** has no artifact to age |
| DOMAIN | `config_tags: str->str` equality | the **input distribution** a value was derived over (GT-Lane-restricted vs GLOBAL margins) |
| UNITS / ROLE | *(no field at all)* | B/flip **exchange rate** used as a mechanism **cost**; bits/flip vs bits/**band-pixel** |

`CanonicalEquation` already declares `domain_of_validity` / `units_in` / `units_out` — but as
free-form prose enforced by nothing. This arm makes those three fields **executable and refusing**.

## §3 — What landed

`src/tac/witness_dsl/guarded_constant.py` — the class. A `GuardedConstant` **composes** a `LawRef`
(#351 stays the SoT for value + provenance; REUSE-NOT-REBUILD, task #533) and adds the guard axes:

- `Units` — dimension as a base-token→exponent map. `bits/flip != bits/band_pixel` by construction.
  Deliberately **not** a conversion table: bits→bytes must be visible at the call site.
- `role` ∈ {exchange_rate, mechanism_cost, threshold, scale, budget, count, fraction}, with the
  incompatibility REASON carried into the refusal message.
- `DomainSpec` — two lines of defence: declared-id equality (cheap, always available) and a
  statistical witness. `DomainCheck.max_margin_separator` **DERIVES** the band as the log-space
  max-margin separator between two MEASURED statistics (`G = sqrt(out/in)`), so no band is chosen.
- `DerivationRef` — the LIVE resolver. `invocation_required=True` means resolving without invoking
  it REFUSES: the exact "documented derivation that never runs" instance.
- `Corroboration` — independent routes are REPRESENTED, not resolved away (the L7 fp32
  cross-hardware bound 0.096 and the Lane p10 0.104748 agree at 1.091×; both survive).
- `resolve()` fail-closed (write path) vs `audit()` LOUD (read path); `STATUS_VACUOUS` with a
  `checks_run / checks_declared` denominator; deterministic `resolution_hash`; and migration
  honesty (`byte_identical_to_incumbent` / `behaviour_change`).

`guarded_constant_registry.py` — two declarations: `seg_margin_hinge_floor` and
`flip_byte_exchange_rate_W`. Every figure is quoted from an artifact or re-derived in tests.

**Migration (VERIFIED live, not asserted):** `margin_floor` default is now
`MARGIN_FLOOR_INCUMBENT`. Re-derived this session via `inspect.signature`:
default `0.1`, registry incumbent `0.1`, `MARGIN_FLOOR_MIGRATION_IS_BYTE_IDENTICAL=True`.
**Zero shipped values changed.**

`src/tac/run_constant_gates.py` — gate **P6**
(`check_no_frozen_literal_where_guarded_derivation_declared`), warn-only, pinned into the Gate #38
ratchet already wired in `tools/all_lanes_preflight.py`. **Declaration-driven by design**: it scans
only for `<name> = <value>` where BOTH the identifier and the exact value are declared by a
registry entry. (`ddm_gd5`/task #864 BUILT the auto-derived "is this reachable?" variant and
REFUTED it — the predicate fires on 1229 of 3251 modules.) Same-line waiver
`# GUARDED_CONSTANT_OK:<rationale>`, placeholder rationales rejected.

## §4 — The defect this custodian found and fixed (the headline)

**P6 was pinned at live count 0 with ZERO tests.** A gate at zero that has never been shown firing
is indistinguishable from a gate that *cannot* fire. Three concrete holes, all now closed:

1. **The docstring was a comment-only contract.** `_guarded_literal_targets` said import failures
   "degrade to an EMPTY target set, and callers report that as a denominator of zero rather than
   as a clean scan (vacuity != pass)" — **no caller reported anything**. A broken registry import
   yielded `{}` → 0 findings → "at baseline" → **PASS, forever, silently**. That is precisely the
   genus this arm exists to extinct, reproduced inside the guard itself.
   *Fix:* `GuardedConstantScanScope` (declared constants, declared site names, files scanned,
   files exempt, `registry_import_ok`) with `is_vacuous`; the ratchet now prints the denominator on
   every run and **REFUSES an empty scope** (`ok=False`), and `strict=True` raises on it too.
2. **`mentions-it == guarded-by-it`.** The "already routed" exemption was a substring test on raw
   file TEXT, so a *comment* naming the registry silenced the gate for that whole file.
   *Fix:* exemption is now an AST import check (`_module_is_imported`), with a regression test.
3. **The repo was RED.** The predecessor added the P6 key to `RUN_CONSTANT_RATCHET_BASELINE` but
   left `test_ratchet_baseline_keys_match_the_checks_it_runs` pinned to two keys.
   *Fix:* pin updated deliberately (that assertion is the reason the key set is pinned at all).

**Live P6 scope, MEASURED at HEAD:** 1 declared constant → 1 site name; **5,068 files scanned**
(`src/tac` + `tools`, excluding `_intake_`, `/tests/`, `/witness_dsl/`, `test_*`); **2 exempt as
already routed** — `direct_description_joint_descent.py` (the migrated instance, correct) and
`run_constant_gates.py` itself (the gate imports `REGISTRY`, so **the gate exempts itself**; benign
today since it holds no declared literal, recorded here because it is a real blind spot).

## §5 — Tests

| file | count | note |
|---|---|---|
| `src/tac/witness_dsl/tests/test_guarded_constant.py` | 29 | 4 positive controls + 4 not-vacuous mirrors |
| `src/tac/tests/test_run_constant_gates.py` | 40 (was 26) | 14 new P6 tests |

Every positive control has a **mirror** asserting the guard *passes* the correct case — without
it, "refuses the bad case" is satisfied by a guard that refuses everything. The domain fixtures are
quantile-matched reconstructions of the MEASURED ddm_rt2 percentile knots; a first attempt shaped
only to match p5 was REFUSED by the guard (its p10 was 0.0740, not the measured 0.1047), which is
itself evidence the guard reads the distribution and not the label.

New anti-vacuity controls: `test_p6_empty_scope_is_vacuous_not_pass` (monkeypatches the registry
import to fail and asserts the ratchet REFUSES), `test_p6_merely_mentioning_the_registry_does_not_exempt_a_file`,
`test_p6_scope_reports_its_denominator_on_the_live_repo`, `test_p6_canonical_instance_stays_migrated`.

## §6 — Adoption map: the next 5, ranked by blast radius (probability × blast × SILENCE)

Survey MEASURED this session. **None of the five is in a file that imports the registry — all five
are unguarded.** Sibling arm `ddm_ca1` owns the repo-wide audit
(`.omx/research/ddm_ca1_calibration_audit_20260803.md:153-155` independently reaches the same
verdicts for rows 1–3); this is the *adoption* view, not a duplicate audit.

| # | constant | sites | provenance in repo | frame it needs | P6-reachable today? |
|---|---|---|---|---|---|
| 1 | `total_archive_ceiling_bytes: Literal[200000]` | `direct_description_carrier_compose.py:1334,:1441` + 4 enforcement sites in `tools/run_ddm_v9_carrier_compose.py` + 6 config JSONs | **NONE EXISTS** (read ±100 lines of both class bodies: no comment, no docstring, no derivation) | `role=budget`, `units=bytes`, domain = the archive ERA it was set in | **YES** (src/tac) |
| 2 | `thr_wall = 2.5e-4` | `experiments/ddm_p3v2_optimal_form_pose_resolve.py:606` (+comment), `ddm_p3v2_finalize_from_cache.py:110` (**no** comment) | arithmetic inline and CORRECT; the *target* is charter-sourced | `role=threshold`, `units=dimensionless (d_pose)`, domain = the operating point it was set at | **NO** — `experiments/` is outside `_P5_SCANNED_SUBTREES` |
| 3 | `W` = 1.2731082153320312 | 4 derived defs + 3 exact retypes + 6 truncated `1.2731` retypes = **13 sites** | exact closed form, derived in 2 src modules | `role=exchange_rate` (already declared); needs `literal_site_names` | **NO** — see structural note below |
| 4 | `ANS_SCALE_BITS = 12` | `repair_entropy_coder_runtime_adapters.py:33`, 3 consumers | **none** — bare literal, zero justification text | `role=scale`, `units=bits`, domain = the token distribution it was chosen for | **YES** (src/tac) |
| 5 | the "57.2 %" denominator pair | **ZERO .py hits** | memos + JSON receipts only | — | **NO — out of this class's reach** |

**#1 is ranked first on SILENCE.** `Literal[200000]` is a pydantic type annotation, so it is
*non-overridable* — any other value is a validation error, not a tunable. A ceiling that refuses
every candidate makes the search report "no candidate", which reads as a **plateau**, not as a
misconfiguration. Zero provenance anywhere. The only contextual clue is the sibling field two lines
up (`added_budget_bytes` max 147,456), implying a base archive ≤ 52,544 B — an era where 200 kB was
generous. The charter's "era gap 6.73×" is **charter-supplied and NOT re-derived by me**; what I can
state is that the live-best own-vehicle archive is ~360,309 B (dc1_fold), i.e. **1.80× above this
hard ceiling**, so the ceiling cannot admit the current frontier at all.

**#2's frame debt, DERIVED (not recalled).** The inline arithmetic is correct — I recomputed
`sqrt(10 · 2.5e-4) = 0.05` exactly. The defect is the TARGET, not the algebra: 0.05 is the wall's
d_pose *contribution* target, while the PR130 pose contribution is **0.015268**, so
`0.05 / 0.015268 = 3.275×`. The wall is **3.275× looser than the bar we must beat** — a config can
pass the wall and still be 3.275× short. This is the charter's figure, and it **reproduces**. Both
sites are function-local literals in `experiments/`; site B carries no provenance comment at all.
Adoption requires either moving the constant into `src/tac` or extending `_P5_SCANNED_SUBTREES`.

**#3 exposes a structural limit of P6, stated honestly.** P6's target filter requires
`invocation_required=True` on a `DerivationRef`. `W` has no derivation — it is an *exact closed
form* — so declaring `literal_site_names` on `W_EXCHANGE_RATE` would **not** bring its 13 sites into
P6's scope as the gate is built. W needs a *different* guard (recompute-the-closed-form), or P6's
filter must widen to "declares a derivation OR an exact closed form". Also MEASURED and unexplained:
`src/tac/tests/test_residual_flip_sidecar_pareto.py:195` asserts against **1.2742**, a *different*
value, at 0.01 tolerance — that discrepancy is unresolved and is flagged, not fixed, here.

**#5 is the honest negative: this class cannot reach it.** A `GuardedConstant` guards a value a
*program resolves*. The 57.2 % lives only in prose and receipts, so there is no consumption site to
guard. Its cure is the memo-side price law (`wf2`), not this class.

## §7 — Refutations of my own charter (reported per §7 of the operating manual)

1. **"78 % of repo .py files are vendored copies under `experiments/results/`" — REFUTED.**
   MEASURED: `experiments/results/` holds 49,131 .py and `experiments/archive/` 9, together
   **7.2 %** of the 677,958 .py under the repo root. The real bulk is agent worktrees:
   `.claude/worktrees/` (288,243) and `.omx/tmp/codex_worktrees/` (~285k), plus `workspace/`
   (28,127) and `.venv_executorch_spike/` (15,069). Cross-check: `git ls-files '*.py'` = 10,676
   tracked, of which only **14** live under results/archive. Anyone bounding a scan by
   `experiments/results` alone is still scanning ~10× the real tree.
2. **"the 57.2 % denominator pair (odd-holdout 18,401 B vs n600 36,798 B)" — the PAIRING is wrong.**
   18,401.206 / 36,797.668 = **0.50006**: those two are the *same* quantity at 300 vs 600 frames, a
   frame-count artifact, not the 57.2 %. The 57.2 % is `1 − 7867/18401 = 0.5725`, measured on the
   odd holdout. **The denominator swap the charter is chasing is real** and sits at
   `ddm_sx2_...md:180` — the fraction measured on the *odd holdout* (18,401 B) is applied to the
   *n600* marginal (36,798 B) to produce 21,048 B and a 430× exchange rate. The residual 7,867 B is
   in **no committed receipt**; that round's artifact is not in the repo.
3. **"validated the detector at P6 live count = 1 with ZERO false positives" — true but incomplete.**
   The count was 1 *pre-migration*; the same landing migrated it, so HEAD is 0. A live count of 0
   with no positive control is not a validated detector — see §4. Now it is.

## §8 — What is owed / not done

- **P6 strict-flip** is NOT taken. It stays warn-only with a ratchet pin at 0; the ratchet already
  refuses any NEW frozen literal, which is the strict surface for this class.
- **The gate exempts itself** (§4). Low blast radius today; unfixed.
- **`experiments/` is outside P6 scope**, so adoption row #2 cannot be gated without a scope change.
- **The 1.2742-vs-1.2731 discrepancy** (§6 #3) is flagged, not resolved.
- **No constant from §6 has been migrated.** The charter asked for a MAP, and this is the map;
  each row is a separate landing with its own byte-identity proof.
- Pointer UNMOVED. Nothing here is a score claim.

---

# §9 — FOURTH CUSTODIAN (2026-08-03, post-limit resume). The gate now FIRES.

**Correction to my own framing first:** the class, the registry, the migration and the 29 class
tests were already LANDED by the previous custodian in commit `02994479ef`. I re-derived them from
scratch before checking `git status` and briefly believed I had authored them. I had not. The
working tree matched HEAD byte-for-byte for all four files. **Verified, not assumed** — per the
coordinator's absorption warning, and it is exactly why that warning exists.

This segment's contribution is the half §8 left owed: **making the gate fire at commit**, plus
three defects found by applying two of today's sibling laws to my own work.

## 9.1 The placement defect — the gate was decoration (CRITICAL)

P6 was pinned into the Gate #38 ratchet in `tools/all_lanes_preflight.py`. That is a
**pre-dispatch** surface, not a commit surface. MEASURED by `ddm_ss1`: `--no-codebase` — the
pre-commit hook's default — examines **0 of 27** preflight gates, which is why the two STRICT
kill-verdict gates (Catalog #307/#308) have **never executed at commit**. A gate that only lives
there cannot refuse the commit that introduces the bug.

**Fixed:** P6 is now **step 1e of `tools/preflight_hook.py`**, over the **staged diff's ADDED
lines**, placed *before* `run_preflight()` — which early-returns on failure and would otherwise
skip it (the same ordering bug the hook's own step-1b comment records). Fail-OPEN and LOUD if the
guard itself breaks, matching siblings 1b/1c/1d. New:
`scan_staged_for_guarded_constant_frozen_literals`, which **reuses** `ss1`'s
`tac.subset_selection_gate.added_lines` rather than reimplementing diff parsing, and returns
`(violations, unexamined)` so a caller can never report clean over files it could not read.
Ordering is asserted by `test_p6_is_wired_into_the_hook_that_actually_fires_not_only_preflight_all`.

## 9.2 The gauge defect — the gate measured its own knob (closes §8's "exempts itself")

**sm1's law:** *ask what the gate reads if the cure is applied and nothing else changes; if it
reads "cured", it measures the knob.*

The previous custodian had already hardened the "already routed" exemption from a raw-TEXT
substring test to an AST import check (a real fix, §4.2). But the exemption itself does not survive
the gauge question: **adding an import is the cosmetic cure**, and it silenced the gate for the
whole file while the frozen literal sat untouched.

The exemption was also **redundant** — the scan only flags `ast.Constant`, so a genuinely migrated
site (whose default is a `Name`) is already invisible. **Removed entirely.** Consequences:

* A file that imports the registry **and still hardcodes the value** is now caught.
* §8's recorded blind spot — *"the gate exempts itself"*, because `run_constant_gates.py` imports
  `REGISTRY` — is **CLOSED**. That item can come off the owed list.
* The scope field is renamed `files_exempt_already_routed` → `files_also_importing_registry`
  (COUNTED, not exempted), so the denominator line no longer claims an exemption that is gone.
* `test_p6_file_that_really_imports_the_registry_is_exempt` **asserted the defective behaviour**.
  It is INVERTED, deliberately and with the reason recorded in its docstring, to
  `test_p6_importing_the_registry_does_NOT_launder_a_frozen_literal`, with a new sibling
  `test_p6_a_genuinely_migrated_site_is_invisible` proving the real cure still silences it.
  Inverting a passing test is exactly the deliberate act that must be argued for, so it is argued
  for here and in the test body.

**Live count unchanged at 0 with the exemption gone** — which is the substantive point: the
migration is real (the site is a `Name`), not import-laundered.

## 9.3 The message defect — "Fix: consume X" named an X that does not exist

`describe()` built the fix target by upper-casing the `constant_id`, emitting
`guarded_constant_registry.SEG_MARGIN_HINGE_FLOOR`. **The real attribute is `MARGIN_FLOOR`.** An
unresolvable symbol in the fix line violates the preflight-message non-negotiable (name the rule,
the value, AND the fix). Now derived from the registry module's own namespace and asserted by
`test_p6_refusal_message_names_a_registry_symbol_that_actually_exists`, which checks `hasattr`.

## 9.4 MUTATION CHECK (si1's law) — the gate can go RED

*A green that cannot go red is the same bug wearing a lab coat.* Reverted the migration in place
(`margin_floor: float = _MARGIN_FLOOR_DEFAULT` → `= 0.1`), re-ran: live count **1**. Restored:
**0**. Patched and restored in-process — **no `git stash`** was used on the shared worktree at any
point (per the `ks1` near-miss).

## 9.5 A duplicate I created and then removed

I wrote `src/tac/tests/test_guarded_constant_gate.py` (17 tests) before discovering the previous
custodian's **14 P6 tests already in `src/tac/tests/test_run_constant_gates.py`**. Ten of mine were
duplicates. **Deleted the file**; folded only the four genuinely-new tests into the existing
module (task #533, REUSE-NOT-REBUILD — which applies to test surfaces too). Suite: **40 → 45
passing**. Class suite unchanged at 29. Ruff clean on every touched file.

## 9.6 Would this have surfaced `#933` (the ±1.0 token range)? **No.**

`#933` is a `±1.0` token-range literal pinning **33.3% of token mass**, with no flag, no menu, no
DSL lever. P6 would not have found it, for two structural reasons worth stating plainly:

1. P6 is **declaration-driven** — it fires only on constants already declared in the registry with
   `literal_site_names`. A value nobody declared is invisible to it. That is the deliberate price
   of avoiding `ddm_gd5`/#864's REFUTED auto-detector (import-reachability fires on 1229 of 3251
   modules), and the printed denominator exists so the blindness is visible rather than implied.
2. More fundamentally, ±1.0 is **not instance (a)**. Instance (a) is *"a derivation exists and is
   not called."* For ±1.0 **there is no derivation at all** — it is the *unladdered governance
   knob* class (`m51` / task #847): a value that CONTROLS an outcome and was never put on the
   ladder. This is the same structural limit §6 #3 already recorded for `W`.

**What would find it:** not a literal scanner but a **consumption-side coverage sweep** —
enumerate the values the receiver/objective actually READS at decode time and check each has a
ladder rung. That is `ddm_ca1`'s axis (463 scalar-returning derivations, 57 zero-callsite, 329
self-module-only), not this one's.

**What this arm does contribute:** `GuardedConstant` is the correct *destination* once ±1.0 is
found — `role=scale`, `units` in token units, `domain` = the token distribution it was fitted on.
The moment it is declared with `literal_site_names=("token_range",)`, P6 refuses any NEW
re-freezing of it across `src/tac` + `tools`. **The class holds it; the gate keeps it held;
neither finds it.**

## 9.7 State after this segment

| item | state |
|---|---|
| P6 live count | **0** over **5,078** files scanned; 1 file also imports the registry (counted, not exempted) |
| P6 surface | pre-commit hook step 1e (staged ADDED lines) **+** Gate #38 ratchet (repo-wide debt view) |
| tests | `test_run_constant_gates.py` **45** (was 40) · `test_guarded_constant.py` **29** |
| ruff | clean on every touched file (`--select F` and full) |
| §8 owed item "the gate exempts itself" | **CLOSED** (9.2) |
| §8 owed items: strict-flip, `experiments/` scope, 1.2742-vs-1.2731, no §6 constant migrated | **still owed, unchanged** |
| pointer | **UNMOVED.** Own-vehicle frontier **S = 0.7910689 @ 353,805 B** `[macOS-CPU advisory]`, gap 0.6189279. No score claim. |

