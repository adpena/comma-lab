# CHARTER — ddm_qs2_compensation_rate_rung (2026-08-13)

REOPEN THE QS1 COUPLED FAMILY ON ITS MEASURED REOPEN CONDITIONS
(`.omx/research/ddm_qs1_dual_axis_verdict_20260813.md`). The mechanism is PROVEN at the
exact instrument (pose leakage +1.13e-7 S; seg −32 net flips); the candidate lost ONLY on
rate (+77 B for 6 pairs ≈ 12.8 B/pair). **The measured breakeven law: 0.785 realized
flips per compensation byte; qs1 delivered 0.416.** Close that 1.9× gap. Scorer-free
local build + solve; NO Modal fire (MAIN fires); sealed dual-axis fire-order output.

**PRIOR-LAW PREDICTION (m38):** admissibility ⇔ realized flips/B > 0.785. **OPERATOR
STEER (2026-08-13, binding): the ~17–20% realization efficiency and 12.8 B/pair coding
cost are FIRST-PASS INSTANCE numbers — IMPROVABLE, never constants** (realization-gap-is-
fixable · first-attempt ≠ family verdict · constants-are-poison). Use measured efficiency
only as the honest PRIOR for screens while you RAISE it. Three levers compose:
efficiency ×2.5 alone clears breakeven at qs1's own bytes; coding ≤6.8 B/pair alone
reaches ~breakeven; both together give real margin. Honest outcomes: a realized win
(a FLOOR MOVE) or measured ceilings on ALL THREE levers. Both are rows.

**OPERATOR DOCTRINES BINDING:** "no naive or toy or generic basis ever" (real archive
coder prices, calibrated screens, per-pair exact decomposition from RETAINED fields —
never re-derived) · byte-closed-row cadence · "as much as possible locally" (T4 only for
the ~$0.16 dual-axis verdict).

## OPTIMAL FORM
Rate-rung arm on a PROVEN engine. Reference forms (ADAPT, provenance-pin path+sha):
- qs1 engine + closure: `1c5557096d` + the import fix on main; the 14 retained Q3
  diagnostic pairs; the compiled-candidate workspace at
  `/Volumes/VertigoDataTier/pact/ddm_qs1_20260813/`.
- Per-pair verdict fields: volume `comma-ddm-js1b-argmax-retained/
  ddm_qs1_dual_axis_20260813_r2` (seg field + pose 6-vectors + pair_error_rms) — decompose
  WHICH pairs delivered the −32 flips and which pixels reverted (the ~80% realization
  loss) BEFORE proposing anything new.
- Proposal source: the js6 bank (200 rows) re-screened with the calibrated efficiency.
- Instrument: the js6b dual-axis worker (unchanged). MECHANISM changes = TOY-BRACKET.

## THE WORK
1. **Per-pair post-mortem** from retained fields: realized flips, changed pixels, and
   coded bytes PER PAIR. Identify the waterfill order (marginal flips/B per pair) and
   where the 189→32 realization loss lives (which edits reverted through uint8/decode).
1b. **RAISE the realization efficiency (co-equal lever, operator steer)**: from the
   post-mortem, classify the reverted edits (sub-quantum amplitude vs the js5 uint8
   quantum floor · AA/resize washout · tie-margin failures) and ENGINEER surviving edits
   per the hr1/rvs1 realization-survival playbook (amplitude above the quantum floor,
   margin-targeted placement). Report the efficiency CURVE per engineering variant —
   17–20% is the naive first-pass floor, not the family's number.
2. **Compensation coding race** (real coder, honest byte counts): (a) coarser dc0
   quantization along the measured pose-null slack — derive the coarsest step whose
   residual pose leakage stays ≤ ~2e-7 S per candidate; (b) shared codebook / joint
   coding across pairs; (c) drop the expensive-cancellation tail (keep only pairs whose
   cancellation survives coarse coding). Target ≤6.8 B/pair; report the measured
   bytes-vs-cancellation curve whatever it shows.
3. **Waterfilled compile**: select pairs by calibrated marginal flips/B > 0.785, compile
   ONE candidate through the HP3/RC64 closure (count every byte), retain everything.
4. **Output**: candidate archive + adapted runtime (po1 pin) + sealed dual-axis
   fire-order (fresh run-id, exact argv for the js6b worker, ~$0.16; NOTE: the worker
   SELF-CLAIMS its lane — the fire-order must NOT instruct MAIN to pre-claim). Admission
   pre-encoded per hv1: net realized complete-S ΔS < 0 on matched instruments. If the
   coding race cannot reach breakeven: sealed NO_FIRE_ORDER with the measured flips/B
   asymptote — that curve is the deliverable.

## OUTPUT
`.omx/research/ddm_qs2_compensation_rate_rung_20260813.md` + code/tests + retained store
(`/Volumes/VertigoDataTier/pact/ddm_qs2_20260813/`) + the sealed (no-)fire-order. Commit
via `tools/subagent_commit_serializer.py` (post-edit shas, `[no-triality] [p0-ledger-ok]`).
End with NEXT_IF_RESUMED + LIVE-HYPOTHESES + DEAD-ENDS.
