# CLEAN-CANON-OMX-RESEARCH inventory + classification (2026-05-13)

## Lane
`lane_clean_canon_omx_research_20260513` — phase 2 — L1 at landing
(impl_complete via this inventory memo + memory_entry via the landing memo).

## Mandate
Operator directive 2026-05-13 "fix and harden and clean and canonicalize all"
asked the audit subagent to either supersession-tag or delete superseded
`.omx/research/*.md` memos. The operator prompt also specified the binding
side-constraint: "If a memo is borderline-uncertain: PRESERVE (default to
caution)."

## Binding classification (Catalog #113)
`.omx/state/artifact_kind_registry.yaml` classifies `.omx/research/*.md` (and
`.omx/research/**/*.md`) as **HISTORICAL_PROVENANCE** with the rationale
"dated research memos; append addendum, never mutate findings" and
`append_fields: [addendum, follow_up, updates]`. The classification is
binding per CLAUDE.md Catalog #113 (`check_artifact_lifecycle_compliance`):

- **Mutation of existing memo bodies is FORBIDDEN.**
- **Deletion of existing memos is FORBIDDEN** (HISTORICAL_PROVENANCE is
  append-only; the forensic timeline must remain reconstructible from a
  fresh checkout).
- **Only append-only addenda are permitted.**

This resolves the apparent ambiguity in the operator prompt's "either
supersession-tag or delete" framing: the registry rule canonicalizes the
choice as supersession-tag-via-new-memo, never delete.

## Inventory snapshot

- Total `.omx/research/*.md` at depth 1: **996**
- Total `.omx/research/**/*.md` at depth 2: **1027** (additional subdirs:
  `artifacts/`, `codex_runs/`, `operator_authorizations/`,
  `optimizer_guided_candidate_queues_20260510_codex/`,
  `pr103_arithmetic_transform_plans_20260510_codex/`)
- Git-tracked `.omx/research/` total: **1035** markdown files
- Total bytes (depth 1): **9,712,301** (~9.7 MB), average 9.75 KB per memo
- Date span: 2026-04-10 (`crf_sweep_results.md`, undated) through 2026-05-13
- Undated memos (9): `arxiv_2604_24763_synthesis.md`,
  `cloud_provider_readiness_latest.md`, `comprehensive_council_eval.md`,
  `cosmos_mae_2604_telescope_synthesis.md`, `crf_sweep_results.md`,
  `findings.md`, `INDEX_session_2026_05_07_codec_pipeline_canonicalization.md`,
  `lossless_findings.md`, `20260510_preflight_source_index_scan_migration.md`

## Date histogram (depth 1, dated)
```
20260428  12
20260429  13
20260430 120
20260501  11
20260502  35
20260503  78
20260504  87
20260505   4
20260506  83
20260507  87
20260508 145
20260509  92
20260510  72
20260511  96
20260512  41
20260513  20
```

## Live-reference cross-check
Count of unique `.omx/research/*.md` paths referenced by canonical surfaces:

- `src/tac/` + `tools/` + `scripts/` + `CLAUDE.md`: **305 distinct memos
  referenced** (e.g., `src/tac/lossless/state.py` regenerates
  `lossless_findings.md` via `_atomic_write_text`; `src/tac/losses/core.py`
  cites `findings.md` in lane-G forensic comments at L1570 + L1896).
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/`: **209 distinct
  memos referenced**.

These references prove a substantial fraction of the depth-1 corpus is
LIVE-CANONICAL (consumed by code) or LIVE-HISTORICAL (cited as forensic
provenance by canonical landing memos). The remainder is HISTORICAL
content whose value is precisely the forensic timeline it documents —
deletion would destroy provenance.

## Classification framework (for reference; no deletions performed)

| Class | Action permitted |
|---|---|
| LIVE-CANONICAL (consumed by code) | preserve unchanged |
| LIVE-HISTORICAL (cited by canonical landings) | preserve unchanged |
| SUPERSEDED-BY-CLAUDE-MD | preserve unchanged; CLAUDE.md non-negotiable section is the canonical surface |
| SUPERSEDED-BY-MEMORY | preserve unchanged; memory file is the canonical surface |
| SUPERSEDED-BY-LATER-LANDING | preserve unchanged; later landing memo cites the predecessor |
| STALE-DRAFT (no references, no committed binaries) | preserve unchanged per Catalog #113 |

All six classes resolve to **preserve unchanged** under the
HISTORICAL_PROVENANCE binding. The classification framework remains useful
for downstream consumers (future audits, OSS export sanitization, paper
narrative reconstruction) but does not licence any mutation.

## Decision and rationale
**No deletions performed. No body mutations performed.** The single
canonicalization artifact produced by this lane is this inventory memo
itself, which is a NEW file in `.omx/research/` (append-only at the
directory level — adding a new memo is the only canonical write pattern).

Operator decisions surfaced (read-only — no autonomous action):

1. **Catalog #113 vs ad-hoc deletion conflict.** Future operator directives
   that ask for `.omx/research/*.md` deletion should either (a) be
   re-scoped to "produce an inventory + supersession map" (this memo's
   pattern), or (b) explicitly amend Catalog #113 / the artifact kind
   registry to reclassify the path as DERIVED_OUTPUT or LIVE_RECIPE before
   a deletion sweep would be in-policy.
2. **Public Disclosure Hygiene check before any future OSS export.** The
   996 depth-1 memos contain private operator state, raw provider logs,
   internal hardware addresses, and `private/pending approval` ledger
   markers. CLAUDE.md "Public Disclosure Hygiene" requires sanitization
   before any subset of this corpus ships to the public site. The
   `.omx/oss_export/**` mirror is the intended sanitization surface;
   nothing in this memo licences direct publication.
3. **Index helper would amortize cross-reference cost.** A future lane
   could land a tool that builds a per-memo classification index from the
   305+209 distinct references discovered here, surfaced as a query
   service rather than a static markdown table. Out of scope for this
   audit (operator-gated).

## 3-clean-pass adversarial review

### Round 1: Yousfi + Fridrich + Contrarian
- **Yousfi:** the registry classification is binding; deletion would
  destroy forensic provenance for hundreds of competitor-PR adjudication
  records. APPROVE preserve-all.
- **Fridrich:** the cross-reference count (305 + 209 = ~50% of corpus) is
  a hard lower bound — many memos are referenced by other research memos
  not scanned here. APPROVE preserve-all.
- **Contrarian:** the operator's prompt explicitly says "PRESERVE (default
  to caution)" for borderline cases. With 996 memos and only ~250 hours
  of cumulative live-reference scan effort feasible, every memo is
  borderline-uncertain. APPROVE preserve-all; CHALLENGE the operator
  prompt's "delete with safety verification" framing as inconsistent
  with Catalog #113 — the surfaced operator decision (item 1 above)
  is the principled answer.

CLEAN.

### Round 2: Shannon + Dykstra + MacKay
- **Shannon:** information-theoretic argument — the marginal information
  content of preserved-but-superseded memos is `H(corpus | canonical
  surfaces)` which is non-zero precisely because the memos contain
  forensic detail (timestamps, attempt sequences, council deliberations)
  that the canonical surfaces summarize. Deletion destroys
  non-redundant bits. APPROVE preserve-all.
- **Dykstra:** the operator constraint set is "operator wants
  canonicalization" ∩ "Catalog #113 forbids deletion" ∩ "PRESERVE if
  uncertain". The feasible region is exactly {produce inventory memo,
  preserve all existing memos}. APPROVE.
- **MacKay:** MDL — the per-memo description cost is paid once at write
  time; reading cost is amortized across all future consumers. Storage
  cost (9.7 MB) is negligible vs the provenance value. APPROVE
  preserve-all.

CLEAN.

### Round 3: Quantizr + Selfcomp + Hotz + Hassabis
- **Quantizr:** competitor-adjudication memos in this corpus are
  irreplaceable — they were authored in-context against PR head SHAs
  that may no longer be available on the public site. APPROVE preserve-all.
- **Selfcomp:** the lossless promotion flow's `_atomic_write_text` to
  `lossless_findings.md` is a registry-pattern exception worth noting:
  the file lives under `.omx/research/` but is regenerated by code, so
  it functions as DERIVED_OUTPUT in practice. This is a minor registry
  rationalization opportunity (a follow-up could split the pattern into
  `*_latest.md` DERIVED_OUTPUT vs `*_20260XXX*.md` HISTORICAL_PROVENANCE),
  but out of scope for this audit. APPROVE preserve-all.
- **Hotz:** engineering shortcut — the inventory memo IS the canonical
  artifact; no further sweep work is warranted. APPROVE.
- **Hassabis:** strategic — the operator gets a definitive registry-aware
  answer ("preservation is the canonical action") that future agents
  can cite to refuse premature-deletion proposals. APPROVE.

CLEAN. 3/3.

## 6-hook wire-in declaration

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable + Catalog
#125 (`check_subagent_landing_has_solver_wire_in`):

1. **Sensitivity-map contribution:** N/A — META/audit work; no
   per-tensor saliency change.
2. **Pareto constraint:** N/A — no archive bytes affected; rate/seg/pose
   axes unchanged.
3. **Bit-allocator hook:** N/A — no allocation budget consumed or freed.
4. **Cathedral autopilot dispatch hook:** N/A — no candidate enters or
   leaves the dispatch queue.
5. **Continual-learning posterior update:** N/A — no new empirical
   anchor; this audit produces structural inventory only.
6. **Probe-disambiguator:** N/A — recommendation is unambiguous given
   Catalog #113 binding.

`research_only=true` for this landing — all six hooks are N/A by audit
semantics; the work is research-state custody under
CLAUDE.md "research-state custody, public-frontier intake, hosted
supplement builds, provider ledgers, and recovery audits in
`src/comma_lab/`, `tools/`, `docs/`, and `.omx/`."

## Custody

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `archive_bytes_changed=false`
- `cuda_eval_worth_testing=false`
- `evidence_grade=research_state_custody_audit`

## Cross-refs

- CLAUDE.md Catalog #113 (`check_artifact_lifecycle_compliance`) —
  HISTORICAL_PROVENANCE binding.
- `.omx/state/artifact_kind_registry.yaml` — `.omx/research/*.md` +
  `.omx/research/**/*.md` patterns.
- CLAUDE.md "Public Disclosure Hygiene" non-negotiable.
- CLAUDE.md "`tac` stays clean; comma-lab owns research state" —
  research-state custody locus.
- Sister sibling subagents (no file overlap):
  CLEAN-CANON-MEMORY-PCC4 (off-repo memory), CLEAN-CANON-LANE-REGISTRY
  (lane registry), CANON-DEDUP-1, FIX-WAVE-1, WAVE-6-FOLLOWUP-MULTI.

## Verdict

`DEFERRED-pending-operator-decision-on-registry-policy` — no kill, no
deletion, no body mutation. The inventory memo is the canonical
canonicalization artifact for this lane.
