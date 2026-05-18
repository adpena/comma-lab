# Codex routing directive: inflate.py OP-1/2/5 + Wyner-Ziv compliance defense + master-gradient extension
# Date: 2026-05-18
# Originating session: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
# Per AGENTS.md "Role Division — Claude designs / Codex executes" + CLAUDE.md "Subagent coherence-by-default" inter-agent directive pattern

## CANONICAL POINTERS (read FIRST, do NOT hardcode their state below)

Read these IN ORDER before any work:

1. `/Users/adpena/Projects/pact/CLAUDE.md` (full; honor every NON-NEGOTIABLE marker)
2. `/Users/adpena/Projects/pact/AGENTS.md` (full; Claude×Codex feedback loop semantics)
3. `.omx/research/grand_council_symposium_inflate_py_extreme_compression_20260518.md` (the parent T3 symposium; PROCEED_WITH_REVISIONS verdict)
4. `.omx/research/canonical_upstream_pr_review_procedural_generation_compliance_20260518.md` (contest-compliance verdict — empirically VERIFIED 4 PR precedents + maintainer CI rejection)
5. `.omx/research/inflate_py_extreme_compression_symposium_directive_20260518.md` (earlier operator-routed directive)
6. `.omx/state/lane_registry.json` (in-flight collisions — declare your lane up-front)
7. `.omx/state/cost_band_posterior.jsonl` (recent posterior; reweight if needed)
8. `.omx/state/probe_outcomes.jsonl` (predecessor probe verdicts; Catalog #313 consultation mandatory)

## OPERATOR APPROVAL CONTEXT

Operator verbatim 2026-05-18 (across this session):
> *"approved, continue with all"*
> *"if our implementation is different and contest compliant I am all for it"*
> *"the per pair master gradient is far from fully exploited and utilized and wired and integrated and fleshed out"*
> *"i think the procedural generation is actually different if we're generating from a hash seed or something else like that or some weights"*
> *"maybe all of this can be combined and integrated for optimal and synergy and extreme optimization and compression and signal density"*

## CONTEST-COMPLIANCE VERDICT (just-landed, 2026-05-18)

| Direction | Compliance | Precedent | Notes |
|---|---|---|---|
| Pre-baked Comma2k19/ImageNet constants in inflate.py | **NON-COMPLIANT** | 4 PR rejections (#36/#38/#68/#69/#78/#87) + maintainer CI bot rejection on PR #69 (houdini) | EXCLUDED from scope |
| Hash-seed PRNG codebook generation (seed inside archive.zip) | **STRUCTURALLY COMPLIANT** | NOT FOUND in precedent (unexplored frontier) | seed bytes IN archive.zip count toward rate; codebook computed at inflate time |
| Weight-derived codebook (derive from renderer.bin SHA) | **COMPLIANT-WITH-CAVEATS** | NOT FOUND | requires frozen renderer.bin SHA + Catalog #272 byte-mutation smoke |
| Master-gradient null-space exploitation | **STRUCTURALLY COMPLIANT** | NOT FOUND | reduces bytes INSIDE archive.zip; no external state |
| OP-1/OP-2/OP-5 reviewability discipline | n/a (no score impact) | n/a | reviewability + LOC budget value only |

**Compliance line clarified by maintainer precedent**: *"must not import external state that wasn't shipped in archive.zip."*

## WORK ITEMS (priority order; execute sequentially or batch as appropriate)

### ITEM 1 — DEFENSIVE: `DeliverabilityProof.contest_compliance_rationale` field (~1-2h, $0)

Per contest-compliance op-routable #1 (HIGHEST EV defensive).

Add new field to `src/tac/wyner_ziv_deliverability/contract.py::DeliverabilityProof` dataclass:

```python
contest_compliance_rationale: str  # MANDATORY non-empty
contest_compliance_citation_chain: tuple[str, ...]  # MUST cite at least one of: archive.zip seed inclusion / weight-derived (no new bytes) / null-space (in-archive reduction) / reviewability-only (no score impact)
```

Update:
- `__post_init__` validation: empty string REJECTED
- `tac.provenance.validate_provenance` integration (cross-reference Catalog #323)
- `tools/audit_provenance_compliance.py` adds new audit category `MISSING_CONTEST_COMPLIANCE_RATIONALE`
- All sister callers of `build_deliverability_proof_*` updated to pass rationale string
- Tests: ~15 dedicated tests in `src/tac/tests/test_wyner_ziv_deliverability_proof_builder.py` extending existing test cohort

Protects $50-200 of downstream substrate engineering by preempting editorial exclusion before paid dispatch fires.

### ITEM 2 — DEFENSIVE: extend `ProvenanceKind` enum (Catalog #323 sister) (~3-5h, $0)

Per contest-compliance op-routable #3 (DEFENSIVE META-meta umbrella).

Edit `src/tac/provenance/contract.py::ProvenanceKind` enum to add:

```python
PROCEDURAL_GENERATION_FROM_ARCHIVE_SEED = "procedural_generation_from_archive_seed"
WEIGHT_DERIVED_CODEBOOK = "weight_derived_codebook"
FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD = "forbidden_out_of_archive_payload"
```

Update:
- `src/tac/provenance/validator.py::audit_score_claim_dict` — the new FORBIDDEN kind triggers PRE-DISPATCH refusal per Catalog #270 dispatch optimization protocol
- `src/tac/provenance/builders.py` — single-line builders for each new kind
- Cross-reference Catalog #313 probe-outcomes-ledger: a FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD verdict registers as blocking probe outcome with 365-day expiry (long stale window since maintainer rules are stable)
- Tests: ~20 dedicated tests in `src/tac/tests/test_provenance_contract.py` + `test_provenance_validator.py` extending existing cohorts
- Per Catalog #186: claim new Catalog # via `tools/claim_catalog_number.py claim --commit-via-serializer --reason "ProvenanceKind enum extension for contest-compliance META-meta umbrella"` — do NOT pre-claim; let the canonical tool serialize the claim

META-meta umbrella structural protection: every future canonical helper that registers any output Provenance is automatically classified per these 3 kinds at validation time. Catches contest-compliance bugs at the artifact-row surface BEFORE dispatch.

### ITEM 3 — HIGHEST EV per TIER-1: extend `tools/extract_master_gradient.py` (~2-4h, $0)

Per TIER-1 wave (a45889b7) finding 2026-05-18: current extractor is fec6-codec-specific (`parse_fec6_archive_layout`); A1 baseline FAILS at `brotli.error: decoder failed`; PR101_lc_v2 / PR106 format0d / PR107 apogee NOT materialized on disk in production-size form.

Extend `tools/extract_master_gradient.py` with:

```python
# New parsers (one per frontier archive grammar):
def parse_a1_archive_layout(archive_bytes: bytes) -> ArchiveLayout: ...
def parse_pr101_lc_v2_archive_layout(archive_bytes: bytes) -> ArchiveLayout: ...
def parse_pr106_format0d_archive_layout(archive_bytes: bytes) -> ArchiveLayout: ...
def parse_pr107_apogee_archive_layout(archive_bytes: bytes) -> ArchiveLayout: ...

# Dispatch helper:
def detect_archive_grammar_and_parse(archive_bytes: bytes) -> tuple[str, ArchiveLayout]:
    """Try each parser in order; return (grammar_name, layout) on first success.
    Raises ArchiveGrammarUnknownError if none match."""
```

For each parser:
- Read the actual archive structure (consult `experiments/results/public_pr*_intake_*` for reverse-engineering data; consult `submissions/<sub>/inflate.py` for the canonical parse-side reference)
- Materialize per-pair fp64 master gradient via the existing `extract_per_pair_gradient_*` pipeline
- Test on a real production-size archive sample for each grammar (declare `[empirical:<artifact path>]` per Catalog #287)
- DO NOT modify the canonical fec6 path (it's the empirical baseline; preserve regression-test coverage)

Unblocks: Catalog #319 v2 cascade across full substrate registry + empirical α-orthogonality matrix computation + master-gradient null-space exploitation roadmap (sister synthesis subagent ac5921b2 in flight).

### ITEM 4 — OP-1+OP-2+OP-5 reviewability batch (~6h editor, $0)

Per parent T3 symposium's PROCEED_WITH_REVISIONS op-routables.

**OP-1**: `src/tac/substrates/_shared/inflate_runtime_extensions.py` (~200 LOC + tests)
- Extract common inflate runtime patterns from the 5 over-200-LOC inflate.py files (hdm8_film_grain 730 / pr103_pr106_final_runtime 532 / pr106_yshift 240 / nscs03 226 / pr106_lapose 214)
- Canonical helper functions: `_inflate_loop_per_video(file_list, archive_dir, output_dir, render_fn)`, `_load_per_substrate_state_dict(archive_dir, sha256)`, etc.
- DO NOT add hash-seed / weight-derived primitives yet (those wait for synthesis subagent verdict + contest-compliance integration)
- Idempotent + byte-identical-output preserving per Catalog #105/#139/#272

**OP-2**: `tools/audit_inflate_py_loc_budget.py` (~250 LOC + tests)
- Per-substrate LOC audit + technique-applicability classification
- HNeRV parity L4 100-LOC default + 200-LOC waiver enforcement
- Operator-facing CLI: `--summary` human-readable, `--json` machine-readable
- Sister to `tools/audit_provenance_compliance.py` pattern

**OP-5**: STRICT preflight gate `check_submission_inflate_py_under_loc_budget`
- Per Catalog #186: claim new Catalog # via `tools/claim_catalog_number.py claim --commit-via-serializer --reason "OP-5 inflate.py LOC budget STRICT gate per T3 symposium"`. The symposium PRE-CLAIMED Catalog #327 but it COLLIDES with Codex's just-landed Catalog #327 (`check_master_gradient_raw_byte_authority_not_landed`) — DO NOT use #327; let the canonical serializer assign the next available #
- Refuses any `submissions/*/inflate.py` >200 LOC without `# INFLATE_LOC_WAIVER:<rationale>` (placeholder rationales rejected per the gate's own discipline)
- Refuses any >100 LOC without `# INFLATE_LOC_DEFAULT_BUDGET_WAIVED:<rationale>`
- Initial wire-in WARN-ONLY per CLAUDE.md "Strict-flip atomicity rule" (live count will be ~25 of 33 inflate.py files at landing); strict-flip pending OP-1 canonical-helper-extraction sweep reduces violations to 0
- Add CLAUDE.md catalog row per Catalog #176 (`check_strict_preflight_callsites_have_claude_md_catalog_row`)

### ITEM 5 — DEFERRED-PENDING-SISTER-SYNTHESIS: `tac.procedural_codebook_generator` canonical helper

Per contest-compliance op-routable #2 (FRONTIER-BREAKING). DO NOT execute this item yet — wait for sister synthesis subagent `ac5921b2` to land its design memo at `.omx/research/comprehensive_analytical_surfaces_inventory_plus_synthesis_design_memo_20260518.md`. The synthesis memo's design will scope this helper's API + integration with master-gradient null-space exploitation + Wyner-Ziv Tier-2 deliverability + the unified-action S_total Lagrangian.

When that memo lands, a follow-up directive will route ITEM 5 (estimated ~6-10h editor + $5-15 dispatch on NSCS06 v8 chroma LUT to validate; expected aggregate ΔS ~−0.013 across 5 substrates).

## DISCIPLINE REMINDERS (Catalog # references)

- Per Catalog #117 + #157 + #174: every commit MUST use `tools/subagent_commit_serializer.py --message "<one-liner>" --files <files> --expected-content-sha256 <file>=<POST_EDIT_sha>` (compute sha AFTER edits in working tree, NOT HEAD)
- Per Catalog #206: checkpoint every ~10 tool uses via `tools/subagent_checkpoint.py --subagent-id <CODEX_SESSION_ID> --step <N> --status in_progress --files-touched <...> --next-action <...>`
- Per Catalog #229: premise-verification BEFORE edit (especially for the archive-grammar parsers — verify the canonical reference points actually exist)
- Per Catalog #287: every empirical claim MUST carry `[empirical:<artifact path>]` tag
- Per Catalog #292: per-deliberation explicit assumption-statement discipline (every architectural decision states "the shared assumption I am operating within is X")
- Per Catalog #295: inflate.py must work with empty PYTHONPATH (Codex's OP-1 helper extraction MUST preserve this)
- Per Catalog #313: probe-outcomes ledger consultation MANDATORY before any dispatch
- Per Catalog #314: declare your `files_touched` at session start to prevent absorption-pattern collisions with in-flight subagents (currently in-flight: `ac5921b2` — synthesis design memo, scope is `.omx/research/comprehensive_analytical_surfaces_inventory_plus_synthesis_design_memo_20260518.md` + memory file only; YOUR scope is the source files listed in ITEMS 1-4 above)
- Per Catalog #325: per-substrate symposium discipline — these are TOOL-class deliverables not substrate-class; no per-substrate symposium needed (per Catalog #270 scope)
- Per Catalog #523: HF Jobs work is sister-owned (do not touch L2 Hinton SegNet surrogate work mid-flight)

## CODEX FEEDBACK LOOP (continual-learning artifacts)

Per AGENTS.md "Continual Learning Feedback Loop" pattern:
- Write your session summary to `.omx/research/codex_session_summary_<topic>_<utc>_codex.md`
- Write any adversarial findings to `.omx/research/codex_findings_<topic>_<utc>_codex.md`
- Both are picked up by Claude at next pre-flight scan per the canonical-pointer rule

## ACKNOWLEDGEMENT

Acknowledge receipt of this directive in your next checkpoint via `tools/subagent_checkpoint.py --notes "incorporated codex_routing_directive_inflate_py_plus_wyner_ziv_compliance_plus_master_gradient_extension_20260518"`.

— Main-Claude (relayed on behalf of operator)
