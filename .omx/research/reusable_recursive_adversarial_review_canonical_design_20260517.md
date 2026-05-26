# Reusable recursive-adversarial-review canonical design — 2026-05-17

**Status:** PARTIAL IMPLEMENTATION — operator-approved 2026-05-17 verbatim *"Should we make that kind of analysis reproducible for other designs for whatever? Proceed with path B and continue fixing and hardening all as well regardless of severity"*
**Provenance:** generated as direct response to the Round-1 adversarial fresh-eyes review of the master-gradient campaign that surfaced 4 CRITICAL + 3 Medium + 2 Low findings
**Lane:** `lane_reusable_recursive_adversarial_review_canonical_20260517` (pre-register at L0 per Catalog #126)
**Mirror pattern:** Catalog #245 4-layer canonical ledger pattern (canonical helper + CLI + STRICT gate + operator_authorize wire-in)

## §0 — Why a reusable canonical helper

The Round-1 adversarial review of `.omx/research/grand_reunion_t4_symposium_orthogonal_optimization_master_gradient_20260517.md` + `.omx/research/cpu_frontier_master_gradient_campaign_plan_20260517.md` + `docs/pr_writeups/cpu_frontier_fec6_20260517.md` was performed AD-HOC in-context. The findings are persistent (4 CRITICAL block paid dispatch); the artifacts are persistent (3 documents under `.omx/research/` + `docs/`); the clean-pass-counter is **NOT persistent** — if the operator-authorize wrapper fires tomorrow with no memory of this review, the campaign would proceed and burn the $200 budget against unaddressed CRITICAL findings.

This is the same class of "ad-hoc evidence with no machine-readable persistence" that Catalog #316 (frontier scan) extincted for canonical-state best-anchor citations and Catalog #245 (Modal call_id ledger) extincted for dispatch evidence. The canonical 4-layer pattern is:

| layer | role | example artifact |
|---|---|---|
| 1. Canonical helper module | fcntl-locked JSONL append-only persistence + query API | `src/tac/recursive_adversarial_review.py` |
| 2. Operator-facing CLI | `--scope-paths`/`--round`/`--rotation` interface | `tools/run_recursive_adversarial_review.py` |
| 3. STRICT preflight gate | refuses paid dispatch when scope-paths' clean-pass counter < 3 | `check_recursive_review_clean_pass_counter_gate` (Catalog # claim via canonical) |
| 4. operator_authorize wire-in | runtime check between Catalog #243 + #271 + #313 and dispatch fire | `tools/operator_authorize.py::_check_recursive_review_clean_pass` |

The helper applies to ANY artifact bundle that gates a costly action — not just the master-gradient campaign. Examples that benefit:
- Substrate design memos (per Catalog #290 + #294 + #303 + #305) gating paid trainer dispatch
- Council deliberation memos (Catalog #300) gating subsequent T-tier decisions
- PR writeups gating contest PR submission
- OSS release manifests gating publication
- Codec grammar changes gating archive rebuild
- Any bundle of `.md` / `.py` / `.yaml` artifacts whose joint correctness is auditable but currently lives only in conversation memory

## §1 — Schema

### `RecursiveReviewRound`

```python
@dataclass(frozen=True)
class RecursiveReviewRound:
    """One round of adversarial review on a versioned artifact bundle.

    Persisted append-only to .omx/state/recursive_review_rounds.jsonl
    per Catalog #128/#131/#245 sister discipline.
    """
    review_id: str                       # uuid-12 — uniquely identifies the round
    bundle_id: str                       # stable id for the artifact bundle being reviewed
                                         # (sha256 of sorted-joined scope_paths bytes)
    scope_paths: tuple[str, ...]         # sorted tuple of relative repo paths
    scope_content_sha256: str            # sha256 of concatenated file contents at review time
    round_number: int                    # 1, 2, 3, ...
    council_rotation: str                # named rotation (e.g., "Z", "skunkworks_sextet", "fresh_eyes_30days")
    council_attendees: tuple[str, ...]   # explicit member list per CLAUDE.md "Council conduct"
    findings: tuple[ReviewFinding, ...]  # all findings in this round
    verdict: str                         # one of PROCEED / PROCEED_WITH_REVISIONS / DEFER / KILL_CANDIDATE
    counter_before: int                  # clean-pass counter value at round start
    counter_after: int                   # 0 if any non-CONFIRMS finding; else counter_before + 1
    reviewed_at_utc: str
    reviewer_agent: str                  # "claude-in-context" / "codex-companion" / "operator-manual"
    related_round_ids: tuple[str, ...]   # prior rounds on same bundle_id (for the cite-chain)
```

### `ReviewFinding`

```python
@dataclass(frozen=True)
class ReviewFinding:
    finding_id: str          # e.g., "C-1", "M-5", "L-8" — locally unique within a round
    axis: str                # one of 8 axes per CLAUDE.md "Recursive adversarial review protocol"
    severity: str            # CRITICAL / MEDIUM / LOW / CONFIRMS
    member: str              # which council member surfaced this
    description: str         # detailed description; the empirical/structural argument
    recommended_fix: str     # the specific change that closes this finding
```

### 8 canonical axes (per CLAUDE.md item 8)

| axis | what it asks |
|---|---|
| `call_sites` | trace actual call sites, not just function signatures |
| `phase_interactions` | check phase-by-phase interactions |
| `resume_scenarios` | verify behavior under crash + resume |
| `edge_cases` | mental-execute `--batch-size 1` / `--rho-max 0` / empty input |
| `default_overrides` | check default arguments that callers might override |
| `comments_vs_code` | verify comments match code (per "Comment-only contracts" rule) |
| `phase_gate_thresholds` | phase-gate phase-sensitive thresholds |
| `assumption_challenge` | what shared assumption is this work operating within; would violating it unlock breakthrough? |

The 8th axis is non-negotiable per CLAUDE.md "Recursive adversarial review protocol" item 8 (added 2026-05-15).

## §2 — Clean-pass-counter semantics

Per CLAUDE.md "Recursive adversarial review protocol — close paths" (post R12+R13):

1. **Counter-advance SEAL** (canonical): `counter_after = counter_before + 1` ⇔ round has ZERO findings of severity ∈ {CRITICAL, MEDIUM, LOW}. CONFIRMS-only findings do NOT block.
2. **Counter-reset**: `counter_after = 0` ⇔ round has ≥1 non-CONFIRMS finding.
3. **SEAL threshold**: `counter_after >= 3` ⇒ bundle is REVIEWED-CLEAN, can be cited as such, and the gating action (paid dispatch / PR submission / publication) is unblocked.
4. **Operator-declared SEAL (D-1 conservative)**: per the same section, can close via {external-adversary unanimous SEAL + Contrarian SUPER-VETO + 7-day cool-down + operator-explicit-invoke}.

## §3 — Bundle identity

A bundle is identified by `bundle_id = sha256(sorted(scope_paths))[:16]`. Adding/removing files from a bundle creates a NEW bundle_id; the counter does NOT transfer. This is intentional — adding a new file is a material scope change that re-triggers review.

The `scope_content_sha256` separately captures the bundle's CONTENT at review time. If the bundle's content changes (any file edited) but the path list is unchanged, the bundle_id is preserved but `counter_after` resets to 0 on the next round (the content delta IS a finding — "the artifact changed without re-review"). This is operationally important: the operator cannot edit an artifact mid-review-cycle and inherit the clean-pass counter.

## §4 — Persistence + concurrency

`.omx/state/recursive_review_rounds.jsonl` is fcntl-locked JSONL append-only per Catalog #128/#131/#245 sister discipline:

- Writer: `tac.recursive_adversarial_review.append_round_locked(record)` acquires `LOCK_EX` on `.omx/state/.recursive_review.lock`, writes the row atomically via `.tmp.<uuid>` + `os.replace`, releases.
- Strict loader: `tac.recursive_adversarial_review.load_rounds_strict(path)` raises `RecursiveReviewLedgerCorruptError` on JSON parse failure per Catalog #138; quarantines corrupt files to `<path>.corrupt.<utc>`.
- Query helpers: `query_rounds_by_bundle_id`, `latest_round_by_bundle_id`, `clean_pass_counter_for_bundle`, `query_unresolved_critical_findings`.
- 4-process concurrent-append stress: tested via `src/tac/tests/test_recursive_adversarial_review.py` 4-proc spawn pool per Catalog #245 pattern.

## §5 — STRICT preflight gate

**`check_recursive_review_clean_pass_counter_gate`** (new Catalog # via canonical claim) refuses any operator-authorize dispatch when:

1. The dispatch's recipe references a bundle (e.g., a substrate dispatch references a design memo bundle that needs review), AND
2. The bundle's latest `counter_after` < 3, AND
3. The bundle has NO `# RECURSIVE_REVIEW_COUNTER_BYPASS_OK:<rationale>` waiver in the recipe YAML.

The recipe `.omx/operator_authorize_recipes/*.yaml` adds an optional `recursive_review_bundle_id: <id>` field. When present, the gate consults the ledger. When absent, the gate is silent (the dispatch is independent of any reviewed bundle).

Same-line rationale-only waivers per Catalog #211 sister discipline; placeholder `<rationale>` literal REJECTED.

## §6 — operator_authorize.py wire-in

Insertion point: between Catalog #313 (predecessor-probe-outcome check) and the dispatch fire (just before `_dispatch_modal` / `_dispatch_vastai` / `_dispatch_lightning`). New function `_check_recursive_review_clean_pass(recipe)` per the same paired-env-bypass pattern as Catalog #199 + #202 + #243 + #271 + #313:

```python
def _check_recursive_review_clean_pass(
    recipe: OperatorAuthorizeRecipe,
    *,
    repo_root: Path,
) -> None:
    """Per Catalog #317 STRICT discipline: refuse dispatch on
    unreviewed-or-failing-review bundles unless paired-env bypass."""
    bundle_id = recipe.recursive_review_bundle_id
    if bundle_id is None:
        return  # recipe declares no bundle; gate is silent
    counter = clean_pass_counter_for_bundle(bundle_id, repo_root=repo_root)
    if counter >= 3:
        return  # SEALED; dispatch may fire
    if _recursive_review_bypass_active():
        _print_loud_bypass_banner(bundle_id, counter)
        return
    raise SystemExit(
        f"[operator-authorize] FATAL: recursive review bundle {bundle_id} "
        f"has clean-pass counter {counter}/3; refusing paid dispatch per "
        f"Catalog #317. Run `tools/run_recursive_adversarial_review.py "
        f"--bundle-id {bundle_id} --round {counter + 1} --rotation <name>` "
        f"to add a clean round, OR set OPERATOR_AUTHORIZE_RECURSIVE_REVIEW_BYPASS_VERDICT=1 "
        f"+ OPERATOR_AUTHORIZE_RECURSIVE_REVIEW_BYPASS_RATIONALE=<text>."
    )
```

## §7 — CLI

`tools/run_recursive_adversarial_review.py` — operator-runnable entry point:

```
usage: run_recursive_adversarial_review.py [-h]
    [--scope-paths PATH [PATH ...]]
    [--bundle-id ID]
    [--round N]
    [--rotation NAME]
    [--reviewer-agent NAME]
    [--findings-jsonl PATH | --findings-from-stdin | --start-bundle-only]
    [--output PATH]
    [--json] [--strict]

Modes:
  --start-bundle-only        Register a new bundle from --scope-paths and exit (counter 0/3)
  --findings-from-stdin      Read structured findings JSONL from stdin, register round, persist
  --findings-jsonl PATH      Same but from file
  (no mode flag)             Interactive Q+A — operator answers per-axis prompts, helper builds findings

Outputs:
  --output PATH              Where to write the round record (default: stdout JSON)
  --strict                   Exit 1 if counter_after < 3 after this round
```

Operator workflow:

```bash
# Step 1: register the bundle (idempotent)
.venv/bin/python tools/run_recursive_adversarial_review.py \
  --scope-paths \
    .omx/research/grand_reunion_t4_symposium_orthogonal_optimization_master_gradient_20260517.md \
    .omx/research/cpu_frontier_master_gradient_campaign_plan_20260517.md \
    docs/pr_writeups/cpu_frontier_fec6_20260517.md \
  --start-bundle-only

# Step 2: run rounds (one per review pass)
.venv/bin/python tools/run_recursive_adversarial_review.py \
  --bundle-id <returned-id> \
  --round 1 \
  --rotation Z \
  --findings-jsonl /tmp/round_1_findings.jsonl

# Step 3: query counter
.venv/bin/python tools/run_recursive_adversarial_review.py \
  --bundle-id <id> --query-counter --json
```

## §8 — Council rotation registry

A "rotation" is a named tuple of council members from CLAUDE.md "Grand Council (advisory)" + the canonical sextet pact. Recommended canonical rotations:

| rotation name | members | best for |
|---|---|---|
| `skunkworks_sextet` | Shannon LEAD / Dykstra CO-LEAD / Yousfi / Fridrich / Contrarian / Assumption-Adversary | core sextet pact per CLAUDE.md "Council conduct" |
| `Z_fresh_eyes` | Karpathy / van den Oord / Boyd / Contrarian-skeptical / Assumption-Adversary / post-mortem-30d / newcomer-no-context | Round 1 fresh review of a council-deliberation output |
| `Y_engineering_red` | Carmack / Hotz / Quantizr / Selfcomp / Tao / Karpathy / Contrarian | Round 2 engineering-red-team |
| `X_theoretical_floor` | Shannon / MacKay / Ballé / Tishby (memorial) / Wyner / Atick / Redlich / Rao / Ballard | Round 3 theoretical-grounding check |
| `A_substrate_specialist` | (specialist seat for the substrate's paradigm) + sextet pact | substrate-specific design reviews |
| `B_oss_release` | Karpathy / Hassabis / Quantizr / Contrarian / Assumption-Adversary / newcomer | OSS release prep / PR submission review |

The CLI's `--rotation` accepts any of these names OR a comma-separated explicit list. Stored in the ledger as the `council_rotation` field.

## §9 — How the master-gradient campaign uses it (concrete worked example)

1. **Register the bundle:**
   ```bash
   .venv/bin/python tools/run_recursive_adversarial_review.py \
     --scope-paths \
       .omx/research/grand_reunion_t4_symposium_orthogonal_optimization_master_gradient_20260517.md \
       .omx/research/cpu_frontier_master_gradient_campaign_plan_20260517.md \
       docs/pr_writeups/cpu_frontier_fec6_20260517.md \
     --start-bundle-only
   # Returns bundle_id: <hex16>
   ```
2. **Round 1 (just completed in-context):** 4 CRITICAL + 3 Medium + 2 Low findings; verdict PROCEED_WITH_REVISIONS; counter_after = 0.
3. **Apply revisions (Phase B of campaign):** revise op-routable #1 to autograd, update writeup math, etc.
4. **Round 2 (with rotation `Y_engineering_red`):** if all CRITICAL findings closed; counter_after = 1.
5. **Round 3 (with rotation `X_theoretical_floor`):** if clean; counter_after = 2.
6. **Round 4 (with rotation `skunkworks_sextet` for final SEAL):** if clean; counter_after = 3 ⇒ SEALED ⇒ campaign Wave 1 may fire.
7. **Recipes for Waves 1-4 dispatch:** declare `recursive_review_bundle_id: <id>` in YAML; Catalog #317 STRICT preflight refuses dispatch if counter < 3.

## §10 — Reproducibility for ANY design

The same workflow applies to:

- A new substrate design memo bundle (e.g., `.omx/research/<substrate>_design_<date>.md` + `experiments/train_substrate_<id>.py` + `.omx/operator_authorize_recipes/<id>.yaml`): register as bundle, review, gate dispatch.
- An OSS release manifest bundle: register, review, gate publication.
- A codec grammar change bundle (e.g., `src/tac/packet_compiler/*.py` + `submissions/<sub>/inflate.py` + golden vectors): register, review, gate archive rebuild.
- A council deliberation memo bundle (per Catalog #300): register, review, gate downstream T-tier decisions.

Each bundle gets its own `bundle_id`; each gets its own clean-pass counter; each can be gated independently. The CLI + helper + STRICT preflight scale to arbitrary number of bundles.

## §11 — Op-routables (this design's own)

1. **(0.5 hour build, $0):** Land `src/tac/recursive_adversarial_review.py` canonical helper + `src/tac/tests/test_recursive_adversarial_review.py` dedicated tests + 4-proc spawn-pool stress
2. **(0.3 hour build, $0):** Land `tools/run_recursive_adversarial_review.py` CLI
3. **(0.2 hour build, $0):** Land `check_recursive_review_clean_pass_counter_gate` STRICT preflight + new Catalog # via canonical claim
4. **(0.1 hour build, $0):** Wire into `tools/operator_authorize.py::_check_recursive_review_clean_pass`
5. **(0.5 hour build, $0):** Backfill the master-gradient campaign bundle's Round 1 findings into the ledger so the in-context review I just performed becomes persistent + machine-readable

Total ~1.6 hours operator + agent wall-clock, all $0.

## §11.1 — 2026-05-17 Codex hardening pass

Codex reviewed the WIP implementation before landing and found one dispatch-
blocking correctness issue in the draft helper:

- **Finding:** `bundle_id` is path-based by design, but the draft clean-pass
  lookup carried the counter forward even when the bundle's file contents
  changed. That contradicted §3: "If the bundle's content changes ... the
  counter_after resets to 0 on the next round." In the worst case, a design
  bundle could receive two clean passes, be edited materially, receive one more
  clean pass, and incorrectly seal at `3/3`.
- **Fix landed in helper:** `clean_pass_counter_for_bundle(...)` now accepts
  `scope_content_sha256`; when supplied and it differs from the latest round's
  content hash, it returns `0`.
- **Fix landed at append boundary:** `append_round_locked(...)` recomputes the
  expected content-aware counter under the fcntl lock and rejects records whose
  `counter_before` would inherit stale clean passes after content changes.
- **Fix landed in CLI:** `tools/run_recursive_adversarial_review.py` passes
  the computed content hash for `--start-bundle-only`, `--query-counter` when
  `--scope-paths` is supplied, and new-round appends.
- **Tests:** `src/tac/tests/test_recursive_adversarial_review.py` now includes
  content-change reset tests and direct-append stale-counter rejection.

Implementation status after this pass:

- Op-routable #1: **LANDED except 4-process stress**, with 30 focused tests.
- Op-routable #2: **LANDED MVP CLI**, with temp-ledger smoke proof.
- Op-routable #3: **PENDING** — STRICT preflight gate not yet wired.
- Op-routable #4: **PENDING** — `tools/operator_authorize.py` runtime gate not
  yet wired.
- Op-routable #5: **PENDING** — master-gradient Round 1 findings not yet
  backfilled to the canonical ledger. The canonical ledger path is currently
  ignored by `.gitignore`, so any no-signal-loss backfill must either adjust
  tracking deliberately or mirror a durable summary in `.omx/research/`.

Authority after this pass:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`

## §12 — Cross-references

- Catalog #245 — Modal call_id ledger (the canonical 4-layer pattern this design mirrors)
- Catalog #316 — frontier scan (sister "canonical-state vs ad-hoc citation" extinction)
- Catalog #300 — council deliberation v2 frontmatter (sister continual-learning surface)
- Catalog #291 + #292 — META-ASSUMPTION + per-deliberation assumption surfacing
- Catalog #313 — predecessor-probe-outcome ledger (sister "gate dispatch on prior verdict" pattern)
- CLAUDE.md "Recursive adversarial review protocol" — the canonical 8-axis discipline this helper operationalizes
- CLAUDE.md "Recursive adversarial review protocol — close paths" — the SEAL semantics this helper enforces


# OBSERVABILITY_SURFACE_SECTION_WAIVED:historical_design_memo_predates_catalog_305_section_header_requirement_or_is_namespace_design_not_substrate_specific_observability_per_catalog_110_113_HISTORICAL_PROVENANCE_APPEND_ONLY_discipline_canonical_clearance_per_comprehensive_bug_audit_cascade_20260526
