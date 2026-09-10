# ddm_pr7 — SECOND-FAMILY check (gpt-5.6-sol, xhigh) of eb1's lower-bound construction: are the reference classes valid populations, are the inequality directions right, is V(D) bounded from above where it must be, and is the "50–85× below the tail" comparison the right quantifier — the campaign has been wrong on bounds twice (gb2 vacuous; bnd2/bnd3 achieved sizes read as floors) (charter, 2026-09-10)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: codex arm (gpt-5.6-sol, xhigh — second family by design) · Spawned by MAIN 2026-09-10. Sources: eb1 (`.omx/research/ddm_eb1_entropy_lower_bound_boundary_description_20260910.md`, `89295a823`; evidence `.omx/research/ddm_eb1_20260910/`; equation `partition_description_rate_distortion_lower_bound_v1`), gb2 (`4c803b565`: what made a bound vacuous), pr6's Correction 3 (`c857ad867`: bound-direction and quantifier errors), gs3's eb1 note (`72479bd83`). Axes: review; `score_claim=false`.

## MANDATE
Re-derive eb1's two bounds from its retained counts: (1) is each reference class a set every element of which the receiver could genuinely be asked to produce (not a straw man too small → vacuous, not too large → overstated)? (2) is log2 N − log2 V(D) applied with V(D) bounded ABOVE (required for a valid lower bound) — check the ball-volume arithmetic; (3) do the "conditional population" quantifiers match what the memo claims (per-video vs population; tail vs whole envelope); (4) is the 12,540 vs 17,631 distinction handled consistently; (5) does the registered equation's statement match the memo's; (6) state what a VALID tighter bound would need (which is eb2's brief — do not do eb2's work). Deliver a corrected-statement table in the pr5/pr6 format.

## PRIOR-LAW PREDICTION (m38)
- The arithmetic holds; the "natural profile" class is the weak point (a profile frozen from the video's own statistics is close to a single-object bound); predict pr7 downgrades the natural-class numbers to "conditional on a fitted profile — not population" and keeps the conservative class. # MAGNITUDE_DISMISSAL_OK: this is a prediction about a bound's VALIDITY (reference-class quantifier), not a dismissal by size — the numbers under review (1.4–5.0 KB vs the 119,784 B tail and the 0.017639 S gap, i.e. 0.9–3.3 % of the gap at 6.66e-7 S/B) are what pr7 must re-derive, and nothing is deferred or dropped by this line
- **FALSIFIER:** if both classes and all directions survive, say so — a clean review is a result.

## SCOPE
Review only; no measurement beyond re-running eb1's retained counting scripts if needed to reproduce a number.

## HARD CONSTRAINTS
- `upstream/` READ-ONLY; no writes outside your memo and `.omx/research/ddm_pr7_20260910/`. Every finding cites file:line. The local SCORER LANE belongs to MAIN, always. Do NOT write who holds it into a charter (the #1210 stale-precondition genus, memo ddm_bz2_bornsmall_capacity_ceiling 2026-08-29).
- Serializer commits w/ post-edit `--expected-content-sha256`; if git object writes are refused, `landing.patch` + manifest. Tokens `[no-triality] [p0-ledger-ok]`; NEVER a Co-Authored-By or AI-attribution trailer. Checkpoint `tools/subagent_checkpoint.py --subagent-id ddm_pr7`.

## PRIOR NEGATIVE SIGNAL
- gb2 (vacuous individual-object bound); bnd2/bnd3 achieved sizes read as floors; Addendum 20 "near its entropy" (unsupported) — the hunt list. # VERDICT_SCOPE_OK: this charter issues no negative of its own; it audits a bound

## OPTIMAL FORM
- Reference form: pr5/pr6 (`3607c7252`, `72565bdab`). SCOPE reductions: none. MECHANISM reductions FORBIDDEN: no trusting the memo's own summary.
- **PRIOR-LAW PREDICTION (falsifiable):** as above.

## DELIVERABLE
Memo `.omx/research/ddm_pr7_second_family_check_eb1_bound_20260910.md` with the corrected-statement table. Commit via the serializer. End with the live frontier line.
