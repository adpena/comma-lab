`PRICING-CLOSED` at FORMULATION scope: every tested full categorical GOP field with one global translation per frame failed the byte gate.

| GOP L | Certified mismatches | Packet bytes | Optimistic total bytes |
|---:|---:|---:|---:|
| 2 | 154,586 | 157,124 | 188,477.8 |
| 3 | 162,633 | 137,832 | 171,526.7 |
| 5 | 208,152 | 78,428 | 125,364.1 |
| 10 | 273,402 | 31,896 | 97,813.4 |
| 20 | 409,920 | 14,820 | 120,450.4 |
| 50 | 661,753 | 5,280 | 184,168.7 |

L=10 was closest but remained 12,793.4 B above the 85,020 B replacement cap. The prior L10 ≥400,000 prediction was falsified, but no build trigger fired.

Deliverables:

- [Research memo](/Users/adpena/Projects/pact/.omx/research/ddm_gf3_time_conditioned_factorized_field_closed_form_pricing_20260908.md)
- [Retained result](/Volumes/VertigoDataTier/pact/ddm_gf3_time_conditioned_pricing/certified_globality_v2/RESULT.json), SHA-256 `5a539a7f…8737e0c1`
- [Custody manifest](/Volumes/VertigoDataTier/pact/ddm_gf3_time_conditioned_pricing/certified_globality_v2/MANIFEST.json), SHA-256 `d6e166bb…57822df`
- Code and memo commit: `14a07fa42`
- Canonical-equation anchor fallback: [bundle](/Volumes/APDataStore/pact/ddm_gf3/receipts/commit_serializer_fallbacks/20260909T150531.764344Z-55702/intended-commit.bundle), fallback commit `6df2c2f5c58c3140674d85dc36b37348d40af7b0`; the sandbox denied the final Git object write.
- Verification: 23 focused tests passed; two review passes completed. The broader development preflight was killed before completion on both attempts, so it is not claimed as passed.

No scorer, training, Modal, MPS, submission, or pointer mutation was performed.

Current own frontier: `S 0.13885056455024844 @ 181,414 B [contest-CUDA T4 n600]`.

## NEXT_IF_RESUMED

- `BUNDLE_READY_MAIN_MUST_LAND` — owner: MAIN; consumer store: `.omx/state/canonical_equations_registry.jsonl`; fire trigger: Git object writes become available, then land the retained equation-anchor bundle.
- `QUEUED-CONDITIONAL` — owner: MAIN; consumer store: `/Volumes/VertigoDataTier/pact/ddm_gf3_time_conditioned_pricing/certified_globality_v2/`; fire trigger: an outside-scope representation supplies a certified n600 bound and receiver-closed packet meeting either `≤71,404.5 B with ≤46,804 mismatches` or `packet + physical residual ≤85,020 B`.

## LIVE-HYPOTHESES

- Time-conditioned low-rank factors may avoid repeatedly storing full categorical fields because temporal scene evolution is structured and smooth.
- Non-rigid warps or parametric boundary atoms may repair many correlated sites per byte, escaping the generic `0.2909 B/mismatch` charge.
- A receiver-valid learned residual probability model may compress the structured residual more efficiently than the generic coders tested here.

## DEAD-ENDS

- The tested `L ∈ {2,3,5,10,20,50}` full-field GOP formulation with global integer translations and generic residual pricing is closed.
- The initial cellwise translation relaxation is valid but too weak for adjudication; the certified pairing/histogram-TV bound supersedes it.
- Coordinate descent cannot be cited as an optimum; it supplies only an upper bound.
- The prior `L10 ≥400,000` numerical hypothesis is false.
- Calling a generic coded stream “RC64” without the receiver’s probability model is not the same physical object and must not be retried.