# Relative-significance FOLD into the costate duty-to-measure RANKER (2026-07-08)

**Operator directive (2026-07-08/09):** *"the costate controller is central to continual learning."*
Permanent structural fix for the recurring magnitude-dismissal bug
(`relative-not-absolute-significance-near-goal-dont-orphan-small-deltaS`): make the APPARATUS compute
fraction-of-remaining-descent so no significant-near-goal lever is orphaned by eyeball.

**Pointer 0.19110 UNMOVED** — apparatus, score-neutral (read / rank / log only → no byte, d_seg, or
d_pose effect → observability-defaults-ON per the "off is a tracked queue" reconciliation). Nothing
launched; live #205 run untouched.

## The recurring bug this extincts

`activation_ledger.duty_to_measure()` ranked the owed lever queue by **state-then-alphabetical** — it
had NO value axis, and levers carried NO ΔS estimate anywhere. So "which owed lever matters most" fell
back to the operator's eyeball, which keeps anchoring on **absolute** ΔS. The re-audit
(`.omx/research/relative_significance_reaudit_20260708.md`) proved the cost: #169 horizon-margin
(ΔS 0.012–0.024) was dismissed as "weak" because it is only ~9% of the full score 0.19110 — yet it is
**~44% of the remaining descent** to sub-0.15. The re-audit named the exact fold; this landing builds it.

## What landed

### 1. The metric
`relative_significance(est, s_current, s_target=0.15) = est / (s_current − s_target)` — the fraction of
the REMAINING descent the lever buys. Key property (the bug's correction): for a FIXED `est`, rel-sig
**RISES** as `s_current → s_target` — a small absolute ΔS becomes MORE significant near the goal, not
negligible. Also exposes the operator's trigger framing `rel_sig_dseg = Δd_seg / 0.0009` (Δd_seg =
ΔS/100 on the d_seg axis) for readability. `s_current` is read from the LIVE pointer
(`read_pointer_s()` → `.omx/state/canonical_frontier_pointer.json`), **never hardcoded**; `s_target`
defaults to 0.15 (THE GOAL) but is a parameter everywhere.

MEASURED flip (the #169 case): `0.018 / 0.19110 = 9.4%` (the eyeball framing that orphaned it) vs
`0.018 / (0.19110 − 0.15) = 43.8%` of remaining descent — ~4.6× reweight.

### 2. The store
`.omx/state/lever_relative_significance.jsonl` — canonical APPEND-ONLY, fcntl-locked, latest-row-wins
per lever (mirrors the `.omx/state/*.jsonl` discipline). Row:
`{lever, est_delta_s, delta_s_label (MEASURED|ESTIMATED|UNMEASURED), source_anchor, axis (d_seg|d_pose|rate), notes, agent, ts}`.
Writer `record_relative_significance` is NO-FAKE-guarded: `source_anchor` required, negative ΔS rejected,
label/axis validated, only `UNMEASURED` may carry a `None` est (a registered duty-to-ESTIMATE marker).

**Seeded** from the re-audit RE-OPEN table (6 rows), each with its source anchor:

| lever key | est ΔS | label | axis | rel-sig (pointer 0.19110→0.15) |
|---|---|---|---|---|
| `d_seg_aware_taper_121` | 0.030 | ESTIMATED | d_seg | **73.0%** |
| `horizon_weighted_margin_169` | 0.018 | MEASURED | d_seg | **43.8%** |
| `StepNativeActivation` (registered) | 0.013 | MEASURED | d_seg | **31.6%** |
| `latent_table_truncate_d18_k90` | 0.001 | ESTIMATED | rate | 2.4% |
| `mod32_neutrality_19_ab` | 0.0005 | ESTIMATED | rate | 1.2% |
| `seg_down_weight_274` | — | UNMEASURED | d_seg | duty-to-ESTIMATE |

### 3. The ranker
`duty_to_measure_ranked(s_current=None, s_target=0.15, *, known, path, sig_path, pointer_path)` in
`activation_ledger.py` — joins `duty_to_measure()` (registered owed levers) with the significance
store, computes rel_sig, returns rows sorted by rel_sig DESC. Store findings that are NOT yet registered
levers are **included** (an orphan is often a *missing wire* — `horizon_weighted_margin_169` is an
un-built trainer flag, a duty-to-BUILD); registered owed levers with no ΔS row are surfaced as a
duty-to-ESTIMATE queue (an un-estimated lever is itself orphaned signal). Ties/unknowns break by
est_delta_s then name — the eyeball is removed from the loop. When the pointer is unreadable it degrades
gracefully to est-DESC ordering (still value-ranked, `rel_sig=None`).

### 4. The surface
`tools/costate_digest.py::section_duty_to_measure()` now renders the ranked queue with **% of remaining
descent** next to each lever (markers `*`=never-fired registered · `~`=unbuilt finding · `?`=est-owed),
anchored to the live pointer. The SessionStart digest + witness check-in now lead every session with the
highest-relative-value owed levers — the continual-learning payoff: the recovered signal auto-surfaces
forever. Real digest line:

```
duty-to-measure (51 owed; ranked by % of remaining descent, pointer 0.19110→0.15;
  *=never-fired ~=unbuilt ?=est-owed): d_seg_aware_taper_121~ 73%,
  horizon_weighted_margin_169~ 43.8%, StepNativeActivation* 31.6%,
  latent_table_truncate_d18_k90~ 2.4%, mod32_neutrality_19_ab~ 1.2%,
  AACoverageRender? ?% (+50 more)
```

## Tests
`src/tac/tests/test_relative_significance_ranking.py` (20): metric near-goal amplification (the exact
#169 flip) + monotone-toward-goal + None-guards; store roundtrip/latest-wins + corrupt-line skip +
validation; `read_pointer_s` from json + missing; ranked reads pointer NOT hardcoded; pointer-unavailable
degrades to est order; ranking by rel_sig NOT name; estimated outrank unknowns; s_target parameterized;
unbuilt-finding included as missing-wire; rel_sig_dseg only for d_seg axis; measured lever dropped from
duty; digest renders % on the seeded store; build_digest never raises. Regression: `test_activation_ledger`
+ `test_costate_digest_corpus_recall` GREEN (44 total). Ruff clean on all three files.

## Sibling coordination
`tools/magnitude_dismissal_detector.py` (FEED-magdismiss, sibling agent) = the **DETECTOR** half —
catches bad dismissals in commits. This = the **RANKER** half — prevents orphaning by surfacing relative
significance. Distinct files; two halves of the relative-significance immune system.

## Triality
- **DAG leg:** FEED-relsigfold in `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- **DSL/apparatus leg:** the ranker + store in `tac.witness_dsl.activation_ledger`, consumed by
  `tools/costate_digest.py` (the #247 costate SENSE layer — central to continual learning).
- **Equations leg:** none registered — the store CONSUMES existing measured anchors; the rel-sig metric
  is a definitional ratio (ΔS / remaining-gap), not a new empirical law.

**Pointer 0.19110 UNMOVED.**
