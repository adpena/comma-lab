# DDM GC1 canonical-equations note

`research_only=true` · `execution_allowed=false` · `score_claim=false` ·
`main_review_required=true`

This note composes existing verified laws and records one build-time
acquisition rule. It registers no empirical score claim and introduces no
numeric constant.

## E1 — exact admission action

For a receiver-closed typed description \(q\),

\[
S(q)=100d_{\rm seg}(\mathcal R D(q))
 +\sqrt{10d_{\rm pose}(\mathcal R D(q))}
 +{25B_{\rm counted}(q)\over 37{,}545{,}489}.
\]

Candidate \(q'\) is admitted from accepted state \(q_k\) iff:

\[
S(q_k)-S(q')>0.
\]

Epistemic status:
`VERIFIED_VIA_SOURCE_INSPECTION` for the functional implementation and
`VERIFIED_VIA_EMPIRICAL_ANCHOR` for its established component inputs. Exact
receiver closure remains a per-candidate obligation.

Existing law/contract:
`ddm_score_quotient_functional_v1` and
`objective_is_min_S_over_solution_set_not_box_or_point`.

## E2 — two-part counted-length closure

\[
\begin{aligned}
B_{\rm counted}(q)=&
B_{\rm named\ base}+B_{\rm framing}+B_\theta+B_z+B_{25}+B_\epsilon\\
&+B_{\rm selected\ program\ branch}
 +B_{\rm learned\ dictionary/codebook}.
\end{aligned}
\]

The last two terms are zero only when the branch and dictionary are fixed
video-independently. A generic interpreter may be free; a choice selected
after examining the video is counted. This is a charge-boundary refinement for
MAIN review, not a measured byte row.

Empirical verification status: `VERIFIED_VIA_SOURCE_INSPECTION` for the
generic-program/content contract; exact selector accounting in DC1 is
`ASSUMED_AWAITING_VERIFICATION` until OP-GC1-2 lands.

## E3 — exact-S and compression-progress acquisition order

For measured positive work \(w(q_k,q')\), define:

\[
g_S(q')={S(q_k)-S(q')\over w(q_k,q')},
\qquad
g_L(q')={B_{\rm counted}(q_k)-B_{\rm counted}(q')\over w(q_k,q')}.
\]

The #688 acquisition state retains candidates nondominated in
\((g_S,g_L)\); a stable typed-coordinate ID is the deterministic final
tie-break. There is no weighted scalarization and no borrowed coefficient.
Exact E1 admission remains the only actuator gate.

Equation ID proposed for MAIN review:
`ddm_event_exact_s_description_progress_pareto_acquisition_v1`.

Empirical verification status:
`INFERRED_FROM_DOMAIN_LITERATURE`. It is permitted only as typed,
default-inert acquisition telemetry until a DDM receipt measures predictive
value. It cannot validate a score claim.

## E4 — CONNECTION conditional-codelength discriminator

For an eligible same-bucket consecutive solved-record pair,

\[
\Delta B_{\rm connection}
=B_{\rm static}
-\left(B_{\rm history\ program/state}+B_{\rm exact\ residual}\right).
\]

Every term is an exact emitted byte count under deterministic parse-back.
Evaluation is leave-one-pair-out. Positive held-out
\(\Delta B_{\rm connection}\) supports missing program structure at the tested
formulation; non-positive value is an instance result, not a CONNECTION-family
negative.

Equation ID proposed for MAIN review:
`ddm_connection_heldout_conditional_codelength_v1`.

Empirical verification status: `ASSUMED_AWAITING_VERIFICATION`; DM1 currently
has zero eligible comparators.

## E5 — semantic-to-realization receipt remains a vector

Until dm2 lands, the semantic-to-realization object is:

\[
\left(
B_{\rm semantic},
B_{\rm realized},
\Delta S_{\rm collateral},
\Delta d_{\rm pose},
\text{row status}
\right)
=
\left(1569,\mathrm{NULL},\mathrm{NULL},\mathrm{NULL},\mathrm{PENDING}\right)
\]

at aggregate scope. The 1,569-byte first coordinate is measured by DM1. The
other coordinates are intentionally NULL, not zero. They must not be collapsed
into one ratio until dm2 defines and MAIN reviews the exact row/joint byte
allocation and collateral accounting.

Empirical verification status:
`VERIFIED_VIA_EMPIRICAL_ANCHOR` for \(B_{\rm semantic}\);
`ASSUMED_AWAITING_VERIFICATION` for every dm2 coordinate.

## Triality references

- DSL consumer: proposed `DDMEventContinuationV1`.
- DAG consumer:
  `FEED-DDM-GC1-SCHMIDHUBER-SYMPOSIUM-20260724`.
- Council authority:
  `feedback_grand_council_ddm_capstone_composition_schmidhuber_lead_20260724.md`.
- Pointer:
  `0.1910828242 [contest-CPU] UNMOVED`.

STORES CONSULTED: (appended by MAIN at custody per the recall-before-decide hook; mirrors the
council memo's own §STORES CONSULTED for this same deliberation) — CLAUDE.md/AGENTS.md/operating
manual · canonical DAG (FEED-603 rows incl. is1/sched1/dm1/dc1 MAIN-review + 159x/economics/
objective) · the five SHA-bound input findings memos (is1, sched1, dm1, dc1, px1) · canonical
equations registry (curriculum/rewarmup/EMA/reverse-waterfill LawRefs) · council posterior +
canonical roster helper · lane registry + subagent checkpoints · task ledger (#688/#689 metadata)
· dm2 charter (pending input, not guessed) · both directive inboxes. Memories: proactive-recall
of the already-settled table entries cited inline; no store claimed beyond those actually read.
