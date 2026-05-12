# Catalog drift audit — CLAUDE.md documentation vs `src/tac/preflight.py` wired strictness — 2026-05-12

**Scope.** Audit every catalog entry in CLAUDE.md "Meta-bug class catalog
(strict-mode preflight)" against the actual `strict=` parameter used in
`src/tac/preflight.py` orchestrator callsites. The W/I/A pass flagged this drift
class after observing CLAUDE.md described Catalog #125 + #126 as
"warn-only initially" while `preflight.py:1894 / :1902` were `strict=True`
(flipped 2026-05-09 with the strict-flip atomicity rule landing).

**Method.** Read-only static analysis. Pulled every `^([0-9]+|[A-Z])\. \`check_*\``
catalog entry from CLAUDE.md (89 entries total spanning #1-#11, #15-#19,
#A-#L, #90-#98, #99-#115, #117-#119, #123-#128, #130-#148, #150-#154, #156-#157).
For each, scanned `src/tac/preflight.py` for every callsite
`<check_name>(...strict=True|False...)` with up-to-15-line forward-lookahead to
handle multi-line `lambda: check_X(strict=..., verbose=...)` patterns. When no
explicit `strict=` kwarg was present at a callsite, fell back to the function's
default-argument value. CLAUDE.md claim classifier distinguishes
`strict-flip PENDING` (still warn-only) from `strict-flipped` (active strict),
and credits checks #1-#11 to the bold-section header
`**Strict (in preflight_all() — fail-loud):**`.

**No code or documentation is modified by this audit.** Recommendations only.

## Headline numbers

| Bucket | Count |
|---|---:|
| Total catalog entries | 89 |
| OK (claim matches code) | 66 |
| DRIFT — claim says warn-only, code is `strict=True` | 16 |
| DRIFT — claim says strict, code is `strict=False` | 1 |
| AMBIGUOUS (no strict-status language in claim; code is `strict=True`) | 5 |
| DEFINED_BUT_NOT_INVOKED (function exists, not wired into orchestrator) | 1 |
| **Total drift surface** | **23** |

## Drift table — claim says WARN-ONLY but code is STRICT (16 entries)

These entries either ratchet'd to STRICT in code without updating CLAUDE.md, or
the section-level "warn-only initially" bold header is now stale.

| Catalog # | Check | CLAUDE.md claim phrase | Code wire-in | Recommended fix |
|---|---|---|---|---|
| #B | `check_vastai_create_writes_tracker` | "Live count: 2" implies warn-only | `strict=True` | Update entry: "Live count: 0 → STRICT" |
| #C | `check_subagent_prompts_no_cpu_fallback` | "Live count: 1" implies warn-only | `strict=True` | Update entry: "Live count: 0 → STRICT" |
| #D | `check_scores_have_lane_tag` | "Live count: 20" implies warn-only | `strict=True` | Update entry: "Live count: 0 → STRICT" |
| #F | `check_halfframe_archive_uses_trained_profile` | "Live count: 2" implies warn-only | `strict=True` | Update entry: "Live count: 0 → STRICT" |
| #G | `check_profile_keys_have_resolvers` | "Live count: 91 (real cleanup target)" | `strict=True` | Update entry: "Live count: 0 → STRICT" (after cleanup) |
| #I | `check_test_files_imports_resolve` | "Live count: 25" implies warn-only | `strict=True` | Update entry: "Live count: 0 → STRICT" |
| #K | `check_uniward_delta_has_attestation_gate` | "Live count: 6" implies warn-only | `strict=True` | Update entry: "Live count: 0 → STRICT" |
| #L | `check_remote_scripts_write_provenance` | "Live count: 5 (Lanes A/B/D/G)" | `strict=True` | Update entry: "Live count: 0 → STRICT" |
| #123 | `check_no_weight_domain_saliency_on_score_gradient_substrate` | "warn-only initially; ... strict-flip after stability period" | `strict=True` | Update entry: "STRICT-FLIPPED 2026-05-09" |
| #124 | `check_representation_lane_has_archive_grammar_at_design_time` | "Currently warn-only per operator approval 2026-05-09; strict-flip after Phase 2 lanes (T1/T6/T10/T15/T17/T18) backfill the 8 fields." | `strict=True` | Verify operator decision: backfill complete + strict-flipped, or revert to warn-only |
| #125 | `check_subagent_landing_has_solver_wire_in` | "Held warn-only initially per operator approval 2026-05-09; strict-flip after legacy-memo backfill drives live count to 0." | `strict=True` | Update entry: "STRICT-FLIPPED" (drove live count to 0) |
| #126 | `check_lane_pre_registered_before_work_starts` | "Held warn-only initially per operator approval 2026-05-09; strict-flip after legacy-commit backfill drives live count to 0." | `strict=True` | Update entry: "STRICT-FLIPPED" |
| #127 | `check_authoritative_tag_requires_custody_metadata` | "Held warn-only initially per directive; strict-flip after live count drives to 0." | `strict=True` | Update entry: "STRICT-FLIPPED at 0" |
| #128 | `check_continual_learning_writes_use_lock` | "Held warn-only initially per directive; strict-flip after live count drives to 0." | `strict=True` | Update entry: "STRICT-FLIPPED at 0" |
| #130 | `check_no_tag_only_custody_validation` | "Held warn-only initially" | `strict=True` | Update entry: "STRICT-FLIPPED at 0" |
| #131 | `check_no_bare_writes_to_shared_state` | "Held warn-only initially" | `strict=True` | Update entry: "STRICT-FLIPPED at 0" |

Cross-reference: section-level bold header
`**Additive 2026-04-27 (12 new, NOT YET wired into preflight_all()):**`
is also stale — most A–L entries are now wired strict=True.

## Drift table — claim says STRICT but code is WARN-ONLY (1 entry)

| Catalog # | Check | CLAUDE.md claim phrase | Code wire-in | Recommended fix |
|---|---|---|---|---|
| #119 | `check_subagent_commits_have_co_author_trailer` | "Held warn-only initially; flip to STRICT after legacy-commit allowlist baseline is populated (~16 known legacy commits)." | `strict=False` (line 1817 + 3531-area) | **Claim is actually warn-only-consistent** — the embedded "Held warn-only initially" phrase exists. False-positive: audit's claim classifier ranked the trailing `strict_claimed` phrase higher than the "Held warn-only" phrase. **No drift — entry is correct.** |

So the actual count of "claim-says-strict-but-code-warn" drifts is **0** after re-read. (Audit v3's per-entry false positive is a tooling artifact; the actual CLAUDE.md text is consistent.)

## Ambiguous — code is STRICT but claim text doesn't explicitly state strictness (5 entries)

The `**Codex R5-r6 (warn-only initially, owned by codex-fix subagent):**` section
header is stale — its 5 entries (#15-#19) are now wired `strict=True` in code.

| Catalog # | Check | Code wire-in | Recommended fix |
|---|---|---|---|
| #15 | `check_no_brittle_six_line_waiver_lookback` | `strict=True` | Move out of "warn-only initially" header into the strict block |
| #16 | `check_kl_distill_uses_roundtripped_frames` | `strict=True` | Same |
| #17 | `check_eval_roundtrip_gate_called_after_output_dir_resolution` | `strict=True` | Same |
| #18 | `check_nvdec_probe_has_error_classification` | `strict=True` | Same |
| #19 | `check_archive_builders_use_deterministic_zip` | `strict=True` | Same |

## DEFINED_BUT_NOT_INVOKED (1 entry)

| Catalog # | Check | Status | Recommended fix |
|---|---|---|---|
| #145 | `check_preflight_cli_default_scope_is_all` | Function defined at `preflight.py:37766` but never invoked from any orchestrator (`preflight_all()` / `preflight_codebase()` / etc.). | Either wire it into `preflight_all()` strict=True (matches CLAUDE.md claim), or remove from catalog table. |

## Drift class root-cause analysis

Of the 22 real drift entries above:

1. **Stale section headers (10 entries: A–L block + R5-r6 block).** When checks
   were migrated/strict-flipped, the *individual entry text* was rotated forward
   but the *bold section header* still says "NOT YET wired" / "warn-only
   initially". The header is the load-bearing context; readers see it before
   the per-entry text. **Single-line header edit fixes 10 entries at once.**
2. **"Held warn-only initially; strict-flip after live count drives to 0"
   entries (6 entries: #123, #125, #126, #127, #128, #130, #131).** Strict-flip
   happened (per the strict-flip atomicity rule) but the entry text was not
   updated to "STRICT-FLIPPED" / "STRICT @ 0". **One-token edit per entry.**
3. **"Live count: N" rows that ratchet'd to 0 (8 entries in #A–#L).**
   Stale live-count counts that should now read "Live count: 0 → STRICT".
4. **Unwired function (#145).** Real wire-in gap. Operator decision needed.

## Recommendations — operator decision surfaced

The audit recommends three mutually-exclusive paths. Per CLAUDE.md "Multiple
contenders → multiple paths" non-negotiable, this audit RECOMMENDS but does
NOT apply edits inline.

### Path A — Bulk text-fix the drift in CLAUDE.md (manual or scripted)

Pros: closes 22 stale entries in one commit. Restores trust that the catalog
table is the canonical strictness ledger.
Cons: no permanent protection. The next strict-flip without text update
re-introduces the drift class.

Estimated effort: ~30 minutes (mechanical: 10 section-header edits + 12 entry
text rotations + 1 #145 wire-in decision).

### Path B — Land Catalog #XXX `check_claude_md_catalog_text_matches_preflight_strict_value`

A new STRICT preflight check that parses CLAUDE.md catalog entries, finds the
matching `check_*` function in `preflight.py`, extracts the orchestrator
callsite `strict=` value (or function default), and refuses any CLAUDE.md state
where the entry text claim contradicts the code wire-in.

Pros: permanent self-protection per CLAUDE.md "Bugs must be permanently fixed
AND self-protected against" non-negotiable. Future strict-flips that forget to
update the catalog text fail STRICT preflight at commit time.
Cons: requires the audit's claim classifier to live in production code (the
NLP problem of "warn-only initially" vs "warn-only initially; ... strict-flip"
is non-trivial and false-positive prone). Best landed warn-only initially and
strict-flipped after the Path A bulk-fix lands. Expected violations after
Path A: 0. Suggested next catalog number: **#158** (or whichever
`tools/claim_catalog_number.py claim` returns at land-time).

Estimated effort: 1-2 hours (claim classifier + 25 tests + STRICT wire-in).

### Path C — Both A and B (recommended)

Land Path A (bulk text-fix) first. Once CLAUDE.md catalog table is canonical,
land Path B as a strict-from-byte-one gate to prevent re-drift. This matches
the "Strict-flip atomicity rule" non-negotiable: every fix lands its
self-protection gate in the same commit-batch.

Estimated effort: 2-3 hours total.

## Test / verification footprint

This audit reads:

- `/Users/adpena/Projects/pact/CLAUDE.md` (read-only)
- `/Users/adpena/Projects/pact/src/tac/preflight.py` (read-only)

This audit writes:

- `/Users/adpena/Projects/pact/.omx/research/catalog_drift_documentation_vs_code_audit_20260512.md` (this file)
- `/Users/adpena/Projects/pact/.omx/tmp/catalog_drift_audit_rows_v3.txt` (full machine-readable rows, gitignored)

No `experiments/results/`, no provider artifacts, no GPU spend.

## Cross-references

- CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
  non-negotiable
- CLAUDE.md "Strict-flip atomicity rule" — "If the fix subagent achieves live
  count = 0 in the same landing, the strict-flip should land in the SAME
  commit-batch."
- Catalog #118 `check_claude_md_catalog_no_duplicate_numbers` — adjacent
  self-protection on the catalog table (duplicate-number drift)
- Catalog #145 `check_preflight_cli_default_scope_is_all` — defined but not
  invoked; operator decision needed

## Wire-in declaration (per Catalog #125)

This audit is a **research-only landing**. Sensitivity-map / Pareto / bit-
allocator / autopilot dispatch / continual-learning / probe-disambiguator
hooks are N/A: the audit is a META-meta documentation-vs-code consistency
check, not a score-affecting solver primitive.

`research_only=true` per the wire-in opt-out.
