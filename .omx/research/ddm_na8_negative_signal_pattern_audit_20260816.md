# na8 — negative/mixed-signal audit: the patterns we had NOT named

`date_utc: 2026-08-16` · `owner: MAIN` · `verdict_scope: apparatus (repo-wide, measured)`
`axis: [ledger census, MEASURED]` · `score_claim: false` · `promotable: false`

**Operator directive 2026-08-16:** *"Audit and use all negative and mixed signal and identify
patterns including those we've missed for further optimization efforts."*

The emphasis is **"including those we've missed."** We already carry ~24 named negative-signal
laws (MEMORY.md ☠️ rows: constants-are-poison · cross-regime constant transfer · prefix bias
sign-inverts-by-axis · wrong-object floors · level errors · vacuity==PASS · same-defect
negatives masquerading as family convergence · stale headlines over corrected bodies · …).
Re-listing those is not the ask. This audit went to the **ledger** and measured.

---

## 1. THE NEW PATTERN — expiry is wired as AMNESTY, never as a NAG

**MEASURED over `.omx/state/probe_outcomes.jsonl`, 728 rows / 436 distinct probe_ids:**

| verdict | n | carries a NAMED blocker | has `next_action` |
|---|---:|---:|---:|
| PROCEED | 353 | — | — |
| **DEFER** | **230** | **65 (28.3%)** | 226 |
| **PARTIAL** | **90** | **28 (31.1%)** | 90 |
| **KILL** | **25** | **6 (24.0%)** | 25 |

345 of 728 rows (47.4%) are non-PROCEED. CLAUDE.md's anti-signal-loss rule says DEFER is a
**forbidden resting state** without a *named, measured* blocker. 100% of these rows carry a
measured token (numbers are everywhere); only **28.3%** name what actually blocks them.

That is the smaller finding. The larger one is the mechanism:

**`expires_at_utc` had exactly ONE reader in the entire repo.** `query_blocking_outcomes`
(`src/tac/probe_outcomes_ledger.py:1557`), and it reads expiry like this:

```python
if isinstance(expires_at, str) and expires_at <= now_iso:
    continue  # outcome has aged past the staleness window
```

Expiry does exactly one thing: it makes a **blocking** row stop blocking. It never makes a
**stale deferral** surface for re-adjudication. **A deferral crossing its own re-examination
date becomes MORE invisible, not less.** The apparatus writes down the precise date at which a
row should be looked at again, and then uses that date only to stop mentioning it.

**Measured consequence (survives the supersession check — these are LATEST-per-probe rows, not
superseded by anything):**

- **148 expired deferrals** live right now (111 DEFER + 37 PARTIAL)
- longest stale **74 days**; median ~52 days; **106 of them past 30 days**
- probe_kinds include real score-relevant work: `wyner_ziv_hoist_deliverability_archive_member`
  (5), `paired_cuda_ratification` (3), `per_pair_dominant_segnet_argmax_reducer`,
  `local_macos_cpu_advisory_residual_codec_smoke` (2)
- sample head, with its own recorded next_action:
  `harvest_e7_vq_k_sweep_1_t4_oom_20260519` → *"Re-dispatch on A100-80GB OR enable
  autocast_fp16+reduce batch"* — a fully specified, never-fired action, 74 days cold.

**Why this is a genuinely new name.** It is the DEFERRAL-surface instance of the CLAUDE.md law
*"'off' is a tracked queue, never a forgotten default"* — but it is **strictly worse** than a
plain default-off, because the timestamp EXISTS and is read in the **wrong direction**. It is
also the sister of #936's *write-only API* (verdict.v1: 486 producers, 0 readers) and of the
#404 telemetry-without-consumer genus. Same shape, third surface, never named here.

**CURE LANDED (two-landing, structural half):** `query_expired_deferrals()` in
`src/tac/probe_outcomes_ledger.py` (commit `17eebd418b`) — the missing reader, ranked
longest-stale-first, carrying `days_expired`. Four both-direction controls executed and
passing: fires on the 148 (positive); returns **0** when `now` is rewound before every expiry
(negative — it can distinguish); verdict-purity 0 violations; rows without an expiry are NOT
invented into the set. It deliberately does **not** auto-resurface: surfacing is a consumer
decision, and inventing a staleness clock where none was written would be its own defect.

---

## 2. TWO MORE PATTERNS VISIBLE IN THE CORPUS, NOT YET IN THE NAMED SET

**(a) FITTED corrections don't transfer; SOLVED ones do.** pk3 (23/23 in-sample → 0/23 LOO) and
pk4 (all 3 rungs LOPO-positive in the modeled space, heldout NEGATIVE-or-zero in reality, on a
deterministic instrument with repeat-noise 0.0) are the same fact twice. Meanwhile qs5's Schur
compensation — an **exact solve**, not a fit — worked and was explicitly recorded as
"UNAFFECTED" by pk4's ceiling. The named laws cover *borrowed constants* and *cross-regime
transfer*; they do not name **fit-vs-solve generalization asymmetry**. Consequence for routing:
a family whose mechanism is *fitting per-pair coefficients from Jacobians* should be priced with
a heldout gate from the first rung, not after.

**(b) SUPPORT-SET CEILING — more pixels on the same support recover nothing.** qs4 netted 17
flips from 100 px; qs5 netted 17 from 132 px; connective restoration recovered **zero**. The
binding constraint was the support's topology, not the edit budget. This is distinct from a cap
artifact (#874) and from under-convergence (#850/#935): those are *instrument* limits, this is a
*structural* one, and the tell is a net result that is invariant to enlarging the actuator.

---

## 3. WHAT THIS BUYS THE OPTIMIZATION EFFORT

The 148 expired deferrals are not a hygiene backlog — several are score-relevant work with a
written next_action and a measured blocker that may since have cleared. The optimization value
is **re-adjudication under changed preconditions**, which is exactly the rv1/na6/na7 protocol
that has repeatedly converted stale negatives into live candidates. The reader is now the join
that protocol was missing.

**Remaining owed (named, not silently dropped):**
1. A consumer that ranks the 148 by `days_expired × score-relevance` and re-adjudicates the head
   (MAIN, next $0 slot). The reader exists; nothing calls it yet.
2. The 71.7% of DEFER rows with no named blocker: either name the blocker or re-verdict. This is
   a compliance debt against a CLAUDE.md rule, measured for the first time here.
3. Fold (a) and (b) into the named-pattern set if they survive one more instance each.

**Honest limits.** The 148 is a census of ONE ledger; memos and task rows carry deferrals this
file never sees (#880's join defect: the follow-on detector's corpus is memos, the backlog is
task rows). The compliance percentages use a lexical blocker-detector, so they are a
*lower bound on naming quality*, not a proof of intent.
