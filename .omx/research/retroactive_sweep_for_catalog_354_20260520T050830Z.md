# Retroactive sweep for Catalog #354 — master-gradient exploit consumer bundle

Per CLAUDE.md "Meta-bug class catalog" Catalog #348 event-driven sweep
discipline. Every new STRICT preflight gate landing requires this 4-field
contract: bug-class symptom signature, pre-fix window, historical
KILL/DEFER/FALSIFY search results, and per-finding RE-EVAL-priority
assignment.

## 1. Bug-class symptom signature

Catalog #354 detects: "one or more of the 8 required master-gradient
exploit consumers (exploits 2-9) is missing from `src/tac/cathedral_consumers/`
OR fails the canonical `CathedralConsumerContract`".

The orphan-signal symptom: a cathedral autopilot ranker invocation
silently misses an exploit's contribution because the consumer package
is absent (silent skip) OR present-but-non-compliant (silent skip via
auto-discovery filter). The ranker's predicted_delta computation
proceeds without the missing exploit's adjustment; downstream dispatch
decisions are made on incomplete signal.

This is the canonical orphan-signal failure mode per CLAUDE.md
"Subagent coherence-by-default" non-negotiable. Sister of Catalog #335
(canonical contract surface) at the bundle-completeness sub-surface.

## 2. Pre-fix window

Pre-RESPAWN-MG-7-BUNDLE (i.e. before 2026-05-20T04:52Z when slot
`slot_mg_7_bundle_exploits_2_thru_9` landed): the 8 exploit consumers
did not exist. The producer surface `tac.master_gradient_comparison.
multi_granularity` (sister MG-3, ~1272 LOC) was emitted by sister
recovery wave at approximately 2026-05-20T04:37Z but had NO consumer
packages routing its 9 helpers through the cathedral autopilot.

The pre-fix window is the entire interval from sister MG-3's emission
of the producer surface to RESPAWN-MG-7-BUNDLE's emission of the 8
consumers (~15 minutes wall-clock). During this interval the producer
surface was DARK to the cathedral ranker — no autopilot dispatch could
have consumed any of exploits 2-9.

## 3. Historical KILL / DEFER / FALSIFY search

Searched memory + research artifacts for any prior verdict that
falsified, killed, or deferred a master-gradient exploit consumer
(exploits 2-9). Specifically searched for:

- `KILL` / `FALSIFIED` / `DEFER` / `RETIRED` verdicts mentioning
  "master gradient" / "exploit #2" through "exploit #9" /
  "score-weighted reconstruction" / "per-class chroma" /
  "top-k byte sensitivity" / "information-theoretic floor" /
  "bit-level sensitivity" / "per-pair clustering".
- `M_contest` / `M_archive` / `M_inflated` related kill verdicts.

Result: NO prior KILL or FALSIFIED verdicts found. The 8 exploit
consumers are NEW design landings; this catalog gate is the first
structural protection for the bundle.

The closest sister anchors:
- Catalog #318 master-gradient raw-byte-authority guard (extincts the
  "raw bit-flip FD over ZIP packet" forbidden pattern; this gate's
  consumers RESPECT #318 by only consuming chain-rule-derived M_archive).
- NSCS06 v6→v7 (44% contest-CUDA improvement; cited by exploit #5
  per-class chroma consumer as DESIGN-TIME empirical anchor; NOT a
  contest-CUDA score claim for the consumer's output).

Neither sister anchor is a kill or defer of an exploit consumer; both
are confirmatory empirical evidence for the 8-consumer bundle's
design direction.

## 4. Per-finding RE-EVAL priority assignment

Since NO prior KILL / FALSIFIED / DEFERRED verdicts apply to the 8
exploit consumers, there are NO findings requiring RE-EVAL.

For completeness, the 8 consumers' adoption status at landing:

| Exploit | Consumer | Status at landing | Re-eval priority |
|---|---|---|---|
| #2 | score_weighted_reconstruction_error_consumer | NEW (warn-only inside consumer; routing-style returns 0.0 adjustment) | N/A (new) |
| #3 | top_k_byte_sensitivity_consumer | NEW | N/A (new) |
| #4 | bottom_k_free_entropy_byte_consumer | NEW | N/A (new) |
| #5 | per_segnet_class_chroma_consumer | NEW (cites NSCS06 v6→v7 anchor) | N/A (new) |
| #6 | substrate_fit_diagnostic_consumer | NEW | N/A (new) |
| #7 | information_theoretic_floor_consumer | NEW (cites Cramer-Rao + Shannon R(D)) | N/A (new) |
| #8 | bit_level_score_critical_bits_consumer | NEW (derives bit-level from byte-level via 8x expansion per Catalog #318) | N/A (new) |
| #9 | per_pair_gradient_clustering_consumer | NEW | N/A (new) |

## Sister coordination

Live count at strict-flip: 0. All 8 consumers land in the same commit
batch as Catalog #354 per CLAUDE.md "Strict-flip atomicity rule".

Sister gates verified for cross-cutting consistency:
- Catalog #335 (canonical consumer contract): all 8 consumers pass
  `validate_consumer_module` empirically.
- Catalog #287 (placeholder-rationale rejection): the file-level waiver
  token `MASTER_GRADIENT_EXPLOIT_CONSUMER_BUNDLE_WAIVED` rejects
  `<rationale>` / `<reason>` literals.
- Catalog #318 (master-gradient raw-byte authority): consumer #3 + #4
  + #8 all respect chain-rule discipline (no raw bit-flip FD).
- Catalog #341 (canonical-routing markers): every consumer's
  `consume_candidate` returns canonical non-promotable markers
  (`predicted_delta_adjustment=0.0` + `promotable=False` +
  `axis_tag="[predicted]"`) verified by end-to-end test
  `test_master_gradient_exploits_end_to_end.py::test_all_8_consumers_consume_candidate_returns_canonical_markers`.

## Operator-routable next steps

The 8 consumers are dispatch-GUIDANCE-only at landing (per Catalog
#341 non-promotable markers). Empirical promotion to contest-CUDA
score signal requires:

1. Paired contest-CUDA dispatch comparing consumer-guided dispatch vs
   baseline; the consumer's contribution must measurably move score.
2. Per-substrate symposium per Catalog #325 if the consumer's output
   is to be promoted from advisory to authoritative.

Until then, the consumers feed the autopilot ranker as observability +
diagnostic surfaces; the cathedral autopilot's adjustment cascade
remains structurally non-promotable until paired-axis evidence lands.


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:retroactive-sweep-Catalog-354-master-gradient-exploit-consumer-bundle-trigger-tokens-describe-swept-bug-class-not-new-equation -->
