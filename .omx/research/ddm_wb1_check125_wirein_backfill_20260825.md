# ddm_wb1 — Catalog #125 wire-in declaration backfill: 124 → 0, gate re-flipped STRICT

**Date:** 2026-08-25 · **Arm:** wb1 (task ledger #1277) · **Cost:** $0 (memo + gate work; no dispatch)
**Gate:** `tac.preflight.check_subagent_landing_has_solver_wire_in` (Catalog #125)
**Outcome:** live count **124 → 0**; both call sites re-flipped `strict=True`.
**Pointer:** UNMOVED. This is apparatus work, not goal progress. No score claim.

---

## 1. Result

| Stage | Live memos | Cleared | Mechanism |
|---|---:|---:|---|
| Start (warn-only, commit `ab3e29900b`) | 124 | — | 318 missing (memo, hook) pairs |
| After detector cure 1 | 62 | 62 | multi-word hook labels + em/en-dash separator |
| After detector cure 2 | 59 | 3 | status slot vs downstream qualifier |
| After detector cure 3 | 57 | 2 | one-line bare-ACTIVE convention |
| After per-memo backfill | **0** | 57 | APPEND-ONLY declaration blocks |

**Half the debt was a detector defect, not memo debt.** 67 of 124 memos already carried
substantive wire-in declarations the scanner could not read.

## 2. Population decomposition (measured before any edit)

The 124 memos held **318** missing (memo, hook) pairs. Per hook:

| Hook | Missing | Alias line present in memo | No alias line |
|---|---:|---:|---:|
| continual_learning | 119 | 101 | 18 |
| bit_allocator | 54 | 29 | 25 |
| pareto | 48 | 23 | 25 |
| probe_disambiguator | 36 | 11 | 25 |
| sensitivity_map | 33 | 4 | 29 |
| cathedral_autopilot | 28 | 4 | 24 |

`continual_learning` was missing in **119 of 124** memos, and 63 memos were missing
*only* that hook. A single-hook miss at 96% prevalence is a detector signature, not an
era-wide discipline failure. Reading the alias lines confirmed it: memos said
`* **Hook #5 continual-learning posterior**: ACTIVE via canonical equation registration`
and the gate refused them.

Refined failure-kind classification of all 318 pairs:

| Kind | Pairs | Disposition |
|---|---:|---|
| ABSENT (no alias line at all) | 146 | genuine backfill |
| BARE_NA (`N/A` with no rationale) | 40 | genuine backfill — CLAUDE.md requires a rationale |
| NARRATIVE (alias only in prose) | 41 | genuine backfill |
| DETECTOR (declaration present, refused) | 87 → 91 | detector cures |
| NEG_ACK (memo states the hook is unwired) | 1 | correctly refused; backfilled honestly |

## 3. Detector cures — root cause and controls

### Root cause

The six CLAUDE.md hook labels are **multi-word**: "Sensitivity-map CONTRIBUTION",
"Pareto CONSTRAINT", "Bit-allocator HOOK", "Cathedral autopilot DISPATCH hook",
"Continual-learning POSTERIOR update", "Probe-disambiguator". `_memo_declares_hook`
matched the short alias, then required a declaration marker in the text immediately
following it. When a memo wrote the full label, the marker sat *after* the trailing
label noun. The trailing nouns of hooks #1–#4 (`contribution` / `constraint` / `hook` /
`dispatch`) happened to already be in the marker keyword set, so they passed. Hook #5's
trailing noun — `posterior` — was not. That asymmetry is the whole 101-memo signature.

### Cure 1 — label continuation + dash separator

Skip up to three canonical label-continuation tokens (closed set, derived from the hook
labels) before the marker test, and accept an em/en dash as a declaration separator
(`1. **Sensitivity-map** — ACTIVE: …`). ASCII hyphen is deliberately **excluded** —
it would false-accept `sensitivity-map-based`.

The continuation strip is **additive, never destructive**. A first draft consumed the
token and immediately broke the table form `| #2 Pareto constraint | ACTIVE | … |`,
which passes through the *unstripped* window via the `constraint` keyword. Measured as a
regression (124 → 119 with new failures appearing); restructured to evaluate both
windows. A test pins the table form.

### Cure 2 — status slot vs downstream qualifier

A negative token means "this hook is unwired" only when it occupies the **status slot**.
Three corpus lines said `hook #2 Pareto constraint = ACTIVE via Dykstra-feasibility check
+ canonical equation candidate registration DEFERRED` — a declaration whose evidence
mentions a deferral. The cure accepts a positive status word (`active|wired|registered`,
word-boundary matched) only within the first 40 chars of the window and only when it
precedes the negative token. `hook #2 Pareto constraint = DEFERRED` still fails.

The `reactivation` trap is pinned by test: `bit-allocator: reactivation criteria missing`
must not read `active` inside `reactivation`.

### Cure 3 — one-line bare-ACTIVE convention

`**6-hook wire-in (Catalog #125):** #1 sensitivity-map ACTIVE (…); #2 Pareto ACTIVE (…)`
writes the status with no punctuation at all. An **anchored** `active` at the window
start is accepted. `registered` and `wired` are deliberately **not** accepted bare —
they read as narrative verbs ("the bit-allocator registered five tensors") and would
widen the gate into prose.

### Cure 4 — PROPOSED, MEASURED, AND REJECTED

Some memos hard-wrap a declaration across two lines:

```
- hook #3 bit-allocator
= N/A; hook #4 cathedral autopilot dispatch = ACTIVE (…)
```

The obvious cure — evaluate each line joined with its successor — was measured on the
real corpus and **manufactures 49 false accepts across 27 memos**. A bare
`- **hook #3 bit-allocator**: N/A` joined with the next bullet
`- **hook #4 cathedral autopilot dispatch**: **ACTIVE** (…)` makes the *next hook's text*
read as bit-allocator's missing rationale. It would have laundered 27 memos green
without a single rationale being written. **Rejected.** A design-guard test
(`test_line_wrap_join_is_deliberately_not_implemented`) pins the refusal with the
measurement in its docstring, so a future agent does not re-propose it.

### Controls (both directions, all EXECUTED)

`src/tac/tests/test_check_subagent_landing_has_solver_wire_in.py`, 53 → 86 tests, all pass.

| Control | Direction | Result |
|---|---|---|
| 5 verbatim corpus hook-#5 forms | positive | accepted post-cure |
| em-dash form for #1 and #6 | positive | accepted post-cure |
| table form `\| Pareto constraint \| ACTIVE \|` | positive (regression) | still accepted |
| one-line bare-ACTIVE, verbatim corpus line | positive | accepted post-cure |
| 3 corpus `ACTIVE … DEFERRED` lines | positive | accepted post-cure |
| narrative "posterior rows accumulate over waves" | negative | still refused |
| bare `N/A` with no rationale | negative | still refused |
| `— not wired` / `: deferred` / `= missing` / `: unwired` / `: TODO` | negative | still refused |
| `reactivation criteria missing` (the `active` trap) | negative | still refused |
| `sensitivity-map-based heuristic` (hyphen trap) | negative | still refused |
| `registered` / bare narrative verbs | negative | still refused |
| bare N/A + next bullet (the rejected cure-4 shape) | negative | still refused |
| end-to-end: memo in corpus convention | positive | 0 violations |
| end-to-end: memo that only narrates the hooks | negative | 1 violation, names the hooks |

**Cures bind:** with `_WIRE_IN_LABEL_CONTINUATION_TOKENS` and `_WIRE_IN_DASH_SEPARATORS`
neutered, all four cure-1 positives return `False`; with `_WIRE_IN_POSITIVE_STATUS_RE`
neutered, the cure-2 positive returns `False`. Executed and recorded.

**No regression:** after every cure the violation set was verified a strict **subset** of
the original 124. The cures only ADD acceptance paths; no previously-passing memo can fail.

## 4. Per-memo backfill — 57 memos, 222 pairs

Each memo was adjudicated from its **own body** (title, apparatus-mutation section,
lane/anchor/probe-outcome rows, and what it actually changed). No blanket stamping.

| Disposition | Pairs | Notes |
|---|---:|---|
| ACTIVE — transcribed from the memo body | 62 | the memo already documents the surface |
| N/A with a memo-specific rationale | 160 | the memo honestly does not create the surface |
| research_only opt-out | 0 | not used; every memo got explicit per-hook lines |
| same-line waiver | 0 | the gate has no waiver token; none invented |

ACTIVE transcriptions by hook: continual_learning 28, probe_disambiguator 17,
bit_allocator 8, sensitivity_map 5, pareto 3, cathedral_autopilot 1.

**31 of 57** memos carried at least one genuinely-ACTIVE surface; **26 of 57** were
honestly N/A across every missing hook (audits, STAND_DOWNs, recovery passes, and
apparatus landings that create no solver surface — most of them against retired
lineages, per the no-old-lineage ban).

Examples of the adjudication, not templates:

- `feedback_z8_detail_entropy_headroom_report_landed_20260531.md` — 5 of 6 ACTIVE. The
  report measures 414,720 detail coefficients on a real archive and finds the quantize +
  zero-RLE codec cuts detail bytes 90.4% at Δ=0.015625. That *is* a bit-allocator
  finding, a two-sided Pareto statement (the detail blob is the lever AND lossless
  byteshuffle is dead), and a per-coefficient sensitivity statement.
- `feedback_slot_qq_…byte_mutation_smoke…md` — bit_allocator ACTIVE: measured per-archive
  null-byte counts that correct the sister estimate by 25× (16,909 predicted vs 665 real).
- `feedback_slot_mm_null_byte_probe_matrix…md` — bit_allocator ACTIVE **with its
  falsification recorded on the same line**, because the sister smoke overturned it.
- `feedback_wave_4_z7_mamba_2_dao_gu_fidelity_audit_landed_20260529.md` — 1 ACTIVE
  (5th anchor on a canonical equation), 5 honest N/A: a math-fidelity audit against a
  published reference moves no bytes and produces no sensitivity signal.
- `feedback_stub_audit_and_fix_wave_landed_20260527.md` — bit_allocator N/A **because the
  memo itself found the `feeds bit_allocator (TODO follow-on)` stub and deliberately left
  it unedited as sister territory**. Naming an unbuilt follow-on is not a wire-in.

### APPEND-ONLY verification

Per Catalog #110/#113, nothing above the appended heading was modified. Verified
mechanically against a pre-edit copy of the whole memory directory: all **57** originals
are exact byte prefixes of their new contents, and **zero** untouched memos changed.

## 5. Strict re-flip

Both call sites in `src/tac/preflight.py` flipped `strict=False` → `strict=True`:

- `preflight_all` (line ~5095), carrying the full burn-down note including the rejected
  cure-4 measurement.
- the dev-scope gate table (line ~8573).

`check_subagent_landing_has_solver_wire_in(strict=True)` returns `[]` against the live
memory directory (759 post-cutover memos scanned, 76 research-only opt-outs, 0 missing).
103 tests across the three related suites pass.

## 6. Honest residue

- **None on the burn-down.** Live count is 0 and both call sites are strict.
- **Scope limit worth naming:** the gate reads an operator-local memory directory outside
  the repo, so these 57 appended blocks are **not committable** and are not covered by
  git history. The strict flip therefore protects *future* landings; the backfilled state
  lives only on this machine (and in the pre-edit copy under this session's scratchpad).
  A CI runner without that directory takes the documented GitHub-Actions skip path.
- **The rejected cure-4 population is real.** Some historical memos genuinely hard-wrap
  their declarations. Those memos are now green because they received an appended block,
  not because the wrap was parsed. If a future agent wants wrapped declarations parsed,
  the honest route is a join that re-checks the N/A-rationale rule per hook, not a raw
  line join — and the design-guard test says so.
- **`research_only=true` was not used anywhere.** Several of these memos would have
  qualified, but an explicit per-hook line preserves more signal than a blanket opt-out.

## 7. Cross-references

- CLAUDE.md "Subagent coherence-by-default" → "Mandatory wire-in for every landing" (the contract).
- CLAUDE.md "Bugs must be permanently fixed AND self-protected against" (the 2-landing pattern; the detector cures ship with their controls).
- CLAUDE.md "Strict-flip atomicity rule" (flip rides live-count 0).
- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE (the backfill discipline).
- Catalog #287 placeholder-rationale rejection (every rationale is substantive).
- `.omx/research/catalog125_external_memory_wire_in_backfill_20260515.md` — the 2026-05-15 precedent (25 memos).
- Sister backfills this session: #287 (162 → 0), #300 (22 → 0), #305 (30 → 0).

## Commits

| Commit | Content |
|---|---|
| `c570ce03e4` | detector cure 1 + 16 controls (124 → 62) |
| `344c4b32fc` | detector cure 2 + 10 controls (62 → 59) |
| `d2105a64e9` | detector cure 3 + 7 controls incl. the cure-4 design guard (59 → 57) |
| *(this commit)* | strict re-flip at both call sites + this ledger |
