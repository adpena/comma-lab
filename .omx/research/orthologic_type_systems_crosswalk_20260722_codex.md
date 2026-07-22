# Orthologic Type Systems crosswalk (#615)

`research_only=true` · `$0` · `score_claim=false` · `promotion_eligible=false` ·
`paid_dispatch=false` · `MAIN landing review required`

**Paper:** Simon Guilloud and Viktor Kunčak, *Orthologic Type Systems*,
[arXiv:2507.10482v1](https://arxiv.org/abs/2507.10482) (30 pages, submitted
2025-07-14). The fetched PDF SHA-256 is
`e173d3114e8519134a2fb0fef60b5ce590f018c7113230f51bc0021bf02e9b5d`.

**Frontier pointer:** `0.1910828242 [contest-CPU Linux x86_64]` — **unchanged by
construction**. No scorer, archive, launch, paid dispatch, or pointer mutation was
performed.

## Outcome first

The paper supplies no adoptable v10 lever under the inherited #588/#554/#483
pays-rent bar. Its strongest result is a quadratic smallest-syntax normalizer for
free ortholattice terms, not a general quotient selector and not a minimum-byte
coder. Pact's live gauge representatives already use problem-specific linear or
max-plus projections, strict native-float32 parse-back, and a same-coder byte
comparison. The scorer-cell and DSL touchpoints do not instantiate the paper's
ortholattice assumptions.

| Rank | Proposed transfer | Verdict | Why | Named consumer | First measurement if ADOPT |
|---:|---|---|---|---|---|
| 1 | Polynomial canonical-form normalization -> PDW2/#519/#581 representatives | **ALREADY-BETTER** | OL+ minimizes syntax-tree nodes under free-ortholattice equivalence. Pact must preserve native-float32 ties and minimize encoded bytes under a declared coder. On the existing strict-prototype comparator, equal 142-byte packets encode to Aurenhammer `134 B`, tropical-principal `137 B`, and zero-sum `131 B` with Brotli q11, so “principal/canonical” is not “shortest.” | Existing `segnet_head_affine_gauge_quotient_v1`, PDW2, and `serialize_affine_cell_candidate_same_coder`; no new consumer | **N/A — no ADOPT.** Retain exact same-coder and receiver-identity measurements. |
| 2 | Non-distributive ortholattice laws -> argmax/tropical/polytope geometry | **N-A** | Frozen-head cells are checked as halfspace/argmax membership. Set union/intersection is distributive; restricting to convex polytopes removes closure under union and gives no natural orthocomplement. A convex-hull “join” would be non-distributive but would add label-invalid points, and no live computation uses it. | None | **N/A — no ADOPT.** |
| 3 | `O(n^2(1+m))` subtyping under assumptions -> #332 typed-config subsumption | **N-A** | #332 is a closed-world bijection from validated DSL ownership/LawRef custody to exact argv bytes. It has no union/intersection/negation type expressions, subtyping axiom store, or proof-search bottleneck. OL+'s open-world entailment would not replace exact flag ownership or compile hashes. | None | **N/A — no ADOPT.** |
| 4 | Partial cut elimination + Horn saturation -> DAG/LawRef apparatus | **N-A** | The paper's subterm property makes OL+ proof search finite; Pact's current dependency questions are ordinary typed graph reachability and exact source/hash checks. Encoding them as orthologic would add a theory without changing a result, computation, or forced generalization. | None; keep existing graph/registry validators | **N/A — no ADOPT.** |

There are **zero ADOPT rows**, so no new measurement, DSL lever, canonical equation,
allocator row, posterior update, or dispatch action is authorized.

## What the paper actually proves

The normalizer is narrower and more structured than the abstract suggests:

1. `delta` computes a pseudo-negation-normal form. Negation crosses joins/meets by
   De Morgan laws but stops above variables and function symbols; complemented
   function symbols receive opposite variance.
2. `beta` collapses complement-containing joins/meets to top/bottom using BL+
   entailment.
3. `zeta` removes locally reducible join/meet structure.
4. `eta` prunes each flattened join/meet to an entailment antichain with a stable
   tie order.

The resulting composition `eta(zeta(beta(delta(S))))` is a normal-form function in
the paper's Definition 6.2 sense: it preserves OL+ equivalence, maps equivalent
inputs to the same syntax tree, and returns a smallest tree in the equivalence
class. The `O(n^2)` bound is for normalization **without background axioms**. It
uses pairwise entailment among input subterms; it is not a confluence theorem for
arbitrary rewrite systems and not a bit-length or compressed-size optimum.

For assumption-sensitive subtyping, the proof system eliminates unrestricted cut
but retains an axiom-specific `AxiomCut`. The resulting subterm property bounds the
candidate sequents quadratically. Deduction-rule instances become Horn clauses, and
unit propagation decides the query in `O(n^2(1+|A|))`, where `A` is the finite set
of ground assumptions. Function symbols are only invariant, monotone, or antitone
in declared arguments; stronger constructor-distribution/conjunctivity laws are
deliberately excluded.

These are genuine results. Their hypotheses simply do not match the proposed live
v10 computations.

## Touchpoint 1 — canonical representatives and gauges

**Verdict: ALREADY-BETTER for the live rate/receiver objective.**

The reusable-looking part of Definition 6.2 is the three-obligation contract:

```text
equivalence preservation:       q(N(x)) = q(x)
class constancy / canonicity:   x ~ y  =>  N(x) = N(y)
minimality:                     no equivalent y has lower declared cost
```

Pact already keeps these notions separate instead of laundering one into another:

- `segnet_head_affine_gauge_quotient_v1` proves the shared-affine gauge and labels
  its 20/19-scalar forms as upper constructions, not information minima.
- PDW2 performs strict encode/decode/re-encode, reconstructs a deterministic
  zero-sum common-affine gauge, and tests the frame-195 native-float32 tie. This is
  stronger receiver arithmetic custody than an abstract term normalizer provides.
- The #581 prerequisite comparator runs Aurenhammer, tropical-residuation, and
  zero-sum gauges through the identical PDW2 serializer and Brotli q11. All three
  preserve the same five strict prototype cells, while coded lengths differ
  `134/137/131 B`. This directly demonstrates why syntactic or order-principal
  normality cannot stand in for archive-rate minimality.
- #519's common-logit/nullspace quotient and PDW2's shared-affine quotient are
  finite-dimensional linear-algebraic invariances. Their natural selectors are
  projections or reference gauges, not OL+ term reductions.

The paper therefore strengthens terminology discipline, not the solver: call a
representation **canonical** only relative to a specified equivalence and selector;
call it **minimal** only relative to a specified cost with a proof or exhaustive
comparison. That distinction is already explicit in the live laws and receipts, so
registering a duplicate equation would add no purchase.

**Scoped reopen condition:** revisit only if a future consumer's state is genuinely
a free BL+/OL+ term, its equivalence is exactly the paper's equational theory, and
tree-node minimality is the consumer's declared cost. An arbitrary gauge orbit,
float32 receiver, or compressed packet does not qualify.

## Touchpoint 2 — non-distributive decision geometry

**Verdict: N-A.**

There is a concrete place where non-distributivity *could* change a computation:
if a solver forced convex decision sets to stay convex, used intersection as meet,
and used convex hull as join, then

```text
C intersect hull(A union B)
```

can strictly contain

```text
hull((C intersect A) union (C intersect B)).
```

The paper's interval counterexample makes the difference explicit: with
`C=[1,3]`, `A=[0,2]`, and `B=[4,5]`, the left side is `[1,3]` while the right
side is `[1,2]`. But this is not a live Pact operation.
`DirectDescriptionPolytopeMembership` compares decoded pixels' hard argmax labels
to target labels; PDW2 uses explicit affine-score differences; the bounded uint8
preimage solver uses exact arithmetic constraints. None forms a convex-hull join or
applies a distributive rewrite.

The alternative carrier choices do not rescue the mapping:

- arbitrary subsets under union/intersection form a distributive lattice;
- convex polyhedra are not closed under set union or complement;
- an argmax partition's cell family is not closed under arbitrary joins/meets;
- max-plus addition distributes over max, but the exact receiver also contains
  nonmonotone rounding, uint8 bounds, argmax ties, and scorer evaluation.

Thus there is no ortholattice, orthocomplement, or failed distributive identity to
feed a named live computation. The distinction changes no current result.

**Scoped reopen condition:** only if a future solver introduces an explicit
join/meet/complement algebra over scorer regions. Its first test must exhibit a
five-element nondistributive sublattice on exact receiver cells and show a different
hard-oracle decision from the current halfspace/set formulation.

## Touchpoint 3 — typed-config subsumption

**Verdict: N-A.**

Catalog #332's live problem was custody, not logical expressiveness. The measured
repair made every emitted flag owned by the typed DSL, attached real LawRefs where
they exist, attached explicit class-4 waivers otherwise, preserved exact argv bytes,
and bound launcher/governor execution to compile hashes. Config schemas are small,
closed, nominal records with validation; they do not expose OL+ type terms or a
subtyping judgement under ground assumptions.

The paper's algorithm would therefore solve a problem the DSL does not have while
omitting the exact properties #332 needs: unique flag ownership, value provenance,
argv ordering/identity, source hashes, and refusal on undeclared fields. No
implementation or benchmark is warranted.

## Surprise audit — partial cut, Horn clauses, and open-world safety

The full paper adds three ideas not visible in the abstract alone:

1. **Partial, not total, cut elimination.** Assumptions remain localized in
   `AxiomCut`, which is enough for the subterm property.
2. **One saturation can answer multiple queries.** The finite Horn dependency graph
   can share structure across entailment checks.
3. **Free-algebra/open-world validity.** Accepted inequalities remain valid when new
   constructors/types are introduced, because only declared monotonicity is used.

None clears the local counterfactual-novelty test. Pact's DAG/LawRef validation is
already graph-shaped, but ordinary reachability and exact-hash/schema checks answer
its current questions. The witness DSL intentionally fails closed on new flags; an
open-world subtype theorem would not authorize a new actuator. Finally, the exact
receiver path is not monotone because of rounding, ties, and byte coding, so the
paper's monotone-constructor discipline cannot replace the existing
monotone/nonmonotone decomposition.

## Reference mine

Ranked follow-ups from the paper's bibliography:

1. **Guilloud et al., “Formula Normalizations in Verification” (CAV 2023), ref.
   [16].** Closest follow-up if Pact ever needs a proof-producing, externally checked
   normalizer. It underlies the minimal-form machinery and is more relevant than a
   generic orthologic survey.
2. **Guilloud and Kunčak, “Orthologic with Axioms” (POPL 2024), ref. [21].** The
   direct source for assumption-sensitive proof theory and the cut-elimination
   argument. Follow only if a live consumer acquires real OL/BL terms plus ground
   assumptions.
3. **Guilloud and Pit-Claudel, “Verified and Optimized Implementation of Orthologic
   Proof Search” (CAV 2025), ref. [22].** Best executable/reference implementation
   and Rocq-custody lead. Follow if, and only if, the reopen conditions above hold.
4. **Freese, Jezek, and Nation, *Free Lattices* (1995), ref. [14], plus Whitman
   (1941), ref. [48].** The source of the free-lattice antichain characterization;
   valuable for proof reconstruction, not a current byte or receiver lever.
5. **Dowling and Gallier, Horn satisfiability (1984), ref. [12].** No follow-up
   needed for Pact: linear-time Horn propagation is standard and current graph
   reachability is simpler.

This reference mine authorizes no separate research arm. Items 1--3 are conditional
reopen pointers, not queued work.

## Triality / quadrality disposition

- **DAG:** companion feed
  `.omx/research/orthologic_type_systems_crosswalk_DAG_FEED_20260722_codex.md`.
- **DSL:** explicit N/A; no lever or config mutation because there is no ADOPT row.
- **Equations:** **NO-NEW-LAW.** The relevant established laws are
  `segnet_head_affine_gauge_quotient_v1`,
  `segnet_head_rank4_linear_flipdist_v1`, and
  `bounded_uint8_resize_preimage_cell_feasibility_v1`. Orthologic supplies no new
  callable, domain extension, or measured anchor. A duplicate “canonical normalizer”
  equation would overstate applicability and create naming drift.
- **Tasks/ledgers:** this memo contains a TaskUpdate-ready summary; MAIN decides
  whether and how to mirror #615. No canonical task or shared posterior was mutated
  from the isolated branch.
- **Sensitivity/Pareto/allocator/autopilot:** no receiver-closed score/byte marginal
  exists; all remain unchanged.

## TaskUpdate-ready summary

```text
Task #615 COMPLETE (research-only): deep-read all 30 pages of arXiv:2507.10482v1
(PDF sha256 e173d3114e8519134a2fb0fef60b5ce590f018c7113230f51bc0021bf02e9b5d).
Ranked verdicts: canonical normalization ALREADY-BETTER; ortholattice decision
geometry N-A; #332 subtyping transfer N-A; partial-cut/Horn apparatus transfer N-A.
Zero ADOPT rows, therefore no new law/DSL/measurement/dispatch. Key falsifier:
existing equal-raw-length same-coder representatives encode to 134/137/131 B, so
OL+ syntactic or principal normality does not imply minimum archive bytes. Pointer
0.1910828242 [contest-CPU Linux x86_64] unchanged by construction. MAIN review and
explicit merge required.
```

## Stores consulted

- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`
- paper PDF in full, plus its bibliography and appendix
- `.omx/research/shalizi_ct_paysrent_audit_20260720T184053Z.md`
- `.omx/research/categorical_spectrum_crosswalk_20260719_codex.md`
- `.omx/research/bousfield_deep_read_20260713.md`
- `.omx/research/pdw2_gauge_packet_probe_20260719_codex.md`
- `.omx/research/prereq_surfaces_flush_20260720T171050Z.md` and its same-coder
  receipt
- `.omx/research/regmax_family_probes_20260720T162719Z.md`
- `.omx/research/nielsen_hsg_crosswalk_20260721T223231Z.md`
- `.omx/research/catalog332_flag_custody_backfill_20260717.md`
- `src/tac/boundary_math/power_diagram_witness.py`
- `src/tac/boundary_math/prereq_surfaces.py`
- `src/tac/optimization/direct_description_polytope_membership.py`
- `src/tac/canonical_equations/seg_rate_breakeven_and_head_gauge_laws_20260719.py`
- `reports/latest.md`, lane/subagent/task/equation registries, and both delegation
  inboxes

The memo follows the operating manual: result first, paper claims re-derived from
the primary artifact, evidence labels remain attached, and each negative is scoped
to the proposed transfer rather than the orthologic family.

## MAIN landing requirement

This isolated branch is not repository authority. MAIN must review the complete
base-to-head diff, confirm the same-coder receipt and live consumer mappings still
hold on main, verify that no new OL+/BL+ consumer landed concurrently, and merge
explicitly. MAIN should preserve the zero-ADOPT/no-new-law disposition unless that
review supplies a named changed computation and its first measurement.
