# Phase 10 — PR111-candidate end-to-end dry-run validation LANDED 2026-05-27

**Lane**: `lane_phase_10_pr111_candidate_dry_run_validation_20260527` L1 (impl_complete + memory_entry)

**Subagent**: `phase_10_pr111_candidate_dry_run_validation_35F1E11A` (parent session
`b74f6039-6caf-44f2-a2c3-cd8156acd447`)

**Mission contribution per Catalog #300**: `apparatus_maintenance` /
`frontier_protecting`. First $0 end-to-end exercise of the just-closed
canonical-submission-pipeline 7/7-layer architecture (Phases 2-9 LANDED today)
via the canonical `tools/operator_pr_submission_full_lifecycle.py --dry-run`
against the V14-V2 frontier-crossing PR111 candidate. NO paid GPU; NO `gh`
commands (operator-gated). Surfaces the exact per-layer blockers + the
Catalog #370 strict-flip blocker chain.

## TL;DR verdict

The canonical Phase 9 CLI runs end-to-end and **the first 3 surfaces PASS**
(attribution self-lint + Layer 0 compression-pipeline with
`--skip-protocol-verification` + predecessor parse). It then **halts at Layer 1
(archive grammar) with a real, substantive blocker**: the V14-V2 candidate's
archive uses ZIP member `x` (the PR101/DQS1-grammar lineage convention), which
the canonical `discover_section_specs_from_archive` helper classifies as
`derived_is_monolithic=False` because the member name is not the canonical
`0.bin`. The CLI then surfaces `monolithic_single_file=False requires non-None
multi_file_justification per HNeRV parity L3` and returns **exit 5 (CLI-ERROR)**.

This is the canonical-pipeline's first real end-to-end finding: **the canonical
Layer 1 does not yet handle the PR101/DQS1-grammar single-file `x`-member
convention**, even though the V14-V2 candidate IS a genuine single-file archive
(one member, 178446 bytes). The candidate itself is byte-clean, paired-CUDA +
paired-CPU verified, sole-author + attribution-clean, and IS the current
canonical CPU frontier.

## The V14-V2 PR111 candidate (verified on disk)

| Field | Value |
|---|---|
| Archive sha256 | `0a3abfe645c4fac0df9ea89237f25dd9bfc6b2471b897c36d7437795d27d1403` |
| Archive bytes | 178546 |
| ZIP member(s) | single member `x` (178446 bytes) |
| inflate.sh | 818 bytes (canonical 3-arg contract) |
| inflate.py | 627 LOC (over the HNeRV parity L4 ≤200 budget) |
| Work dir | `experiments/results/v14_v2_dqs1_plus_fec10_substituted_20260526T023000Z/` |
| submission_dir | `<work dir>/submission_dir/` (archive.zip + inflate.sh + inflate.py + src/ + encoder/) |

Both paired axes already measured on disk (Modal, upstream `evaluate.py` on the
exact archive bytes):

| Axis | score_recomputed_from_components | evidence_grade | hardware | score_claim | passed |
|---|---|---|---|---|---|
| `[contest-CPU]` | **0.19202062679074616** | contest-CPU | linux_x86_64 | true | true |
| `[contest-CUDA T4]` | **0.22618311337661345** | contest-CUDA | linux_x86_64_t4 | true | — |

The CUDA result carries `final_score: 0.23` (display-rounded), `provenance_device: cuda`,
`scorer_device: cuda`, `gpu_t4_match`, and expected sha/size both `0a3abfe6.../178546`.
The CPU result carries `promotion_eligible: false` (CPU-axis-advisory-until-1:1
discipline per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA").

## Per-layer dry-run verdict table

`tools/operator_pr_submission_full_lifecycle.py --dry-run --skip-protocol-verification`
(scratch output to `.omx/tmp/phase10_dry_run/`; cleaned post-run; NO `submissions/`
pollution; NO sidecars landed because Layer 1 halted before the Layer 2 builder ran):

| Layer | Phase | Surface | Dry-run verdict | Detail |
|---|---|---|---|---|
| pre | — | predecessor parse | **PASS** | `@SajayR:56:HNeRV_substrate @AaronLeslie138:95:fec_curriculum` parsed cleanly |
| pre | — | attribution self-lint | **PASS** | ZERO forbidden tokens in generated attribution-chain markdown |
| 0 | 2 | compression_pipeline | **PASS** (with `--skip-protocol-verification`) | substrate_id resolved (`pr101_lc_v2_clone_enhanced_curriculum` placeholder trainer/recipe) |
| 1 | 3 | archive_grammar | **FAIL** (exit 5) | `monolithic_single_file=False requires non-None multi_file_justification per HNeRV parity L3` |
| 2 | 4 | builder | NOT REACHED | (halted at Layer 1) |
| 3 | 5 | linter | NOT REACHED | (would flag inflate.py 627 LOC > 200 budget + 4 advisory emdashes if PR body copied from reports/) |
| 4 | 6 | compliance | NOT REACHED | |
| 5 | 7 | paired_auth_eval | NOT REACHED | (paired CPU + CUDA already on disk; would be PAIRED_PASS once threaded) |
| 6 | 8 | Catalog #370 gate | NOT REACHED | |

**Exit code: 5 (CLI-ERROR)** — Layer 1 `ArchiveGrammarError`.

## Root-cause diagnosis (empirically confirmed)

`build_archive_grammar_from_compression_pipeline_result` is called by the CLI
with `monolithic_single_file=True` (Phase 9 source line). When `section_specs`
is None (always, for the CLI), the helper auto-derives from the actual archive
via `discover_section_specs_from_archive(archive_abs)`:

```
derived_specs, derived_is_monolithic = discover_section_specs_from_archive(archive_abs)
...
if monolithic_single_file and not derived_is_monolithic:
    monolithic_single_file = False   # <-- flips True -> False
```

Empirical confirmation:

```
discover_section_specs_from_archive(<V14-V2 archive>) ->
  derived_is_monolithic = False
  num section_specs = 1
  section: x  member: x
```

The helper treats `derived_is_monolithic=True` ONLY when the single member is
named `CANONICAL_MONOLITHIC_MEMBER_NAME = "0.bin"`. The V14-V2 archive uses
member `x` (the PR101 frame-exploit-selector grammar convention inherited from
DQS1 rank021). So `monolithic_single_file` flips to False, but the CLI never
threads a `multi_file_justification`, so `ArchiveGrammarManifest.__post_init__`
raises at the `monolithic_single_file=False requires non-None
multi_file_justification` invariant.

**This is a single-file archive misclassified as multi-file purely because of
the member-name convention.** It is NOT a defect in the V14-V2 candidate (which
is a legitimate single-`x`-member archive); it is a canonical-pipeline gap: the
canonical Layer 1 does not yet recognize the PR101/DQS1-grammar `x`-member
single-file convention as monolithic.

## Apples-to-apples + Catalog #343 frontier-pointer reconciliation

The V14-V2 candidate IS the canonical CPU frontier:

| | Canonical frontier pointer | V14-V2 candidate | Match |
|---|---|---|---|
| CPU sha | `0a3abfe645c4fac0...` | `0a3abfe645c4fac0...` | **YES** |
| CPU score | 0.19202062679074616 `[contest-CPU]` | 0.19202062679074616 `[contest-CPU]` | **YES** (exact) |
| CPU hardware | linux_x86_64_cpu | linux_x86_64 | YES |
| CUDA frontier | PR106 format0d `9cb989cef519...` @ 0.20533002902019143 `[contest-CUDA T4]` | candidate 0.22618311337661345 `[contest-CUDA T4]` | candidate is NOT CUDA frontier (expected) |

The candidate targets the CPU ranking axis (the contest leaderboard ranks by
CPU per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"). On CUDA it scores
0.22618 vs the PR106 format0d CUDA frontier 0.20533 — i.e., the candidate is NOT
the CUDA frontier. This is the correct axis-targeting: the FEC10 substitution is
a pure CPU-axis rate-axis improvement (-7.66e-6 [contest-CPU] frontier-crossing)
and the canonical CPU frontier pointer already reflects it. The CUDA axis is a
separate evidence space per CLAUDE.md "Apples-to-apples evidence discipline"
rule 2 (CPU and CUDA are separate evidence spaces; do not infer one from the
other).

## Attribution + apples-to-apples discipline verification (point C)

The PR111 candidate landing report
`reports/pr111_candidate_cascade_a_fec10_substitution_onto_dqs1_frontier_landed_20260526.md`:

| Discipline | Result |
|---|---|
| Forbidden tokens (Claude/Anthropic/Co-Authored/claude.com/anthropic.com) | **0** (clean) |
| First-person-plural (we/our/us/we're/we've/we'll/we'd) | **0** (clean) |
| Author line | **Alejandro Peña <adpena@gmail.com>** (sole-author) ✓ |
| Axis labels on score claims (Catalog #343) | 4 (`[contest-CPU]` + `[contest-CUDA T4]` on every score) ✓ |
| Emdash (U+2014) | 4 (in prose: title + 3 lineage lines) — **advisory only** |

The CLI's own generated attribution-chain markdown (built from the
`--predecessors` specs) passed the self-lint (`attribution_self_lint: ok`) — it
contains no forbidden token, no first-person-plural, and no emdash.

The 4 emdashes are in the `reports/` landing report (NOT a PR-facing surface in
the submission_dir). The CLI's Layer 3 linter scans the actual generated PR body
(`PR_BODY.md` etc.), not the reports/ landing memo. The advisory note: if the
operator copy-pastes the reports/ landing report into the PR body, the 4
emdashes WOULD trip the Phase 5 linter `_EMDASH` check and must be replaced with
` - ` or `:` first. The CLI's generated attribution placeholder is emdash-clean.

## Catalog #370 strict-flip blocker chain

Catalog #370 (`check_no_pr_submission_without_canonical_compliance_verdict`)
current LIVE_COUNT = **4** (the exact WARN-ONLY baseline from the Phase 8
landing): `submissions/a1/` + `submissions/pr106_latent_sidecar_r2/` +
`submissions/pr106_latent_sidecar_r2_pr101_grammar/` + `submissions/robust_current/`
— each has a PR-facing `README.md` without the canonical 4-verdict chain.

The V14-V2 candidate does NOT currently contribute to the live count because its
`submission_dir` lives under `experiments/results/` (the gate scans
`submissions/*` only) AND has NO PR-facing artifact (no `PR_BODY.md`; no
PR-facing `README.md`). It also has NONE of the 4 canonical verdict sidecars yet
(`submission_bundle_result.json` / `lint_verdict.json` / `compliance_verdict.json`
/ `paired_auth_eval_verdict.json`).

**What the V14-V2 PR111 candidate needs to produce all 4 canonical verdicts:**

| Verdict | $0-achievable in dry-run? | Blocker |
|---|---|---|
| Phase 4 `SubmissionBundleResult` (overall_pass=true) | YES (once Layer 1 passes) | **BLOCKED on Layer 1 archive-grammar `x`-member gap** |
| Phase 5 `LintVerdict` (overall_clean=true) | YES (once Layer 2 emits a PR body) | inflate.py 627 LOC > 200 budget needs `--inflate-py-loc-waiver-rationale`; emdash-clean PR body |
| Phase 6 `ComplianceVerdict` (overall_clean=true) | PARTIAL | structural checks $0; D3 hosting (`gh release`) + D5 (`gh pr create`) artifacts operator-gated |
| Phase 7 `PairedAuthEvalVerdict` (verdict=PAIRED_PASS) | **Already on disk** | paired CPU 0.19202 + CUDA 0.22618 verified; needs threading into the canonical `paired_auth_eval_verdict.json` / `dual_eval_adjudicated.json` sidecar (Catalog #370 accepts both filenames) |

**Strict-flip blocker chain (the operator-gated actions remaining before
Catalog #370 can flip strict):**

1. **(canonical-pipeline gap, $0)** Layer 1 archive-grammar must recognize the
   PR101/DQS1-grammar single-`x`-member convention as monolithic OR the CLI must
   thread a `multi_file_justification` for the PR101-grammar lineage (e.g.
   `"PR101 frame-exploit-selector grammar uses single ZIP member 'x' per the
   DQS1 rank021 frontier substrate convention; single-file archive"`). This is a
   sister-routable apparatus fix (NOT operator-gated; NOT in my scope — I am
   submission-pipeline-validation, orthogonal to substrate-grammar work).
2. **($0)** Once Layer 1 passes, run the CLI `--dry-run` end-to-end against an
   `--output-dir submissions/pr111_v14_v2_fec10/` to emit the 4 verdict sidecars
   + the canonical PR body; thread `--inflate-py-loc-waiver-rationale` for the
   627-LOC inflate.py.
3. **($0)** Thread the existing paired CPU + CUDA results into the Phase 7
   `paired_auth_eval_verdict.json` sidecar (the candidate already has both axes
   on disk; this is a wire-in, not a paid dispatch).
4. **(operator-gated)** D3 hosting: `gh release create` on the
   `adpena/comma_video_compression_challenge` fork (operator chooses hosting).
5. **(operator-gated)** D5 submission: `gh pr create` to
   `commaai/comma_video_compression_challenge` (CLAUDE.md "Executing actions
   with care").
6. **(strict-flip prerequisite)** The 4 baseline `submissions/*/` (a1 +
   pr106_latent_sidecar_r2 + pr106_latent_sidecar_r2_pr101_grammar +
   robust_current) each need either the 4-verdict chain backfilled OR a
   `# PR_SUBMISSION_NO_CANONICAL_COMPLIANCE_OK:<rationale>` waiver OR PR-facing
   sentinel-token stripping (per the Phase 8 landing's operator-routable
   resolution). Catalog #370 strict-flips to live count 0 only after this
   backfill.

**The $0-achievable subset (Layers 0-4 + 6) is BLOCKED at Layer 1 by the
canonical-pipeline `x`-member gap.** The operator-gated subset is Layer 5
paired-CUDA (already on disk, so $0 to wire in) + the `gh` host/submit commands.

## Verdict + operator-routable next steps

**LIFECYCLE VERDICT: NAMED-BLOCKER (exit 5, Layer 1 archive-grammar
`x`-member-convention gap).**

The canonical Phase 9 CLI exercises end-to-end correctly: it parses
predecessors, self-lints attribution, runs Layer 0, and then halts cleanly at
the first real blocker (Layer 1) with an actionable error message and the
correct exit-code routing (5 = CLI-ERROR per the 9th-directive taxonomy). The
12th canonicalization × standardization × ease-of-contest-compliance trinity is
empirically validated: the single-command default-path WORKS — it found the
exact gap the V14-V2 candidate hits, at $0, in one invocation.

**Operator-routable next (priority order):**

1. **(sister-routable apparatus fix, $0, NOT my scope)** Extend the canonical
   Layer 1 `discover_section_specs_from_archive` to recognize the PR101/DQS1
   single-`x`-member grammar as monolithic (add `x` to the monolithic
   member-name allow-set OR introduce a PR101-grammar-lineage classification),
   OR add a `--multi-file-justification` CLI flag to
   `tools/operator_pr_submission_full_lifecycle.py` so PR101-grammar candidates
   can pass the canonical Layer 1.
2. **($0, post-Layer-1-fix)** Re-run the CLI `--dry-run --output-dir
   submissions/pr111_v14_v2_fec10/ --inflate-py-loc-waiver-rationale
   "<rationale>"` end-to-end; expect the 4 verdict sidecars + a PR body + exit 4
   (OPERATOR-GATED) once Layer 5 paired results are threaded.
3. **($0)** Wire the existing paired CPU 0.19202 + CUDA 0.22618 into the Phase 7
   `paired_auth_eval_verdict.json` (or copy the medal-class `dual_eval_adjudicated.json`
   pattern Catalog #370 accepts).
4. **(operator-gated)** D3 host (`gh release create`) + D5 submit
   (`gh pr create`) per CLAUDE.md "Executing actions with care".
5. **(strict-flip)** Backfill the 4 baseline `submissions/*/` to drive
   Catalog #370 live count to 0, then strict-flip.

**Canonical single-command future:** once the Layer 1 `x`-member gap is closed,
the entire V14-V2 PR111 submission collapses to ONE CLI invocation reaching
exit 4 OPERATOR-GATED, emitting the 4 verdict sidecars + PR body + the
operator-gated `gh` commands — per the Phase 9 `full_lifecycle_cli_consolidation_savings_v1`
canonical-equation candidate (~180x wall-clock collapse vs the 2026-05-19 PR101
~3h × 4-subagent manual anti-pattern).

## Canonical equation #344 promotion status (NOT YET)

The Phase 9 landing memo declared 6 FORMALIZATION_PENDING canonical-equation
candidates (`compression_pipeline` + `archive_grammar` + `submission_bundle` +
`submission_linter` + `pr_submission_compliance_gate` +
`full_lifecycle_cli_consolidation_savings_v1`) that promote to REGISTERED at the
first PACKET-CLEAN end-to-end regression. **This dry-run did NOT land PACKET-CLEAN
(exit 5, Layer 1 blocker), so the 6 equations remain FORMALIZATION_PENDING.**
They promote when a PR111 candidate (V14-V2 or a sister) reaches exit 4
OPERATOR-GATED end-to-end after the Layer 1 `x`-member gap is closed. This is
the correct gating per Catalog #344 — no equation promotes on an incomplete
end-to-end run.

## Canonical-vs-unique decision per layer

| Decision | Choice | Rationale |
|---|---|---|
| Use canonical Phase 9 CLI (not hand-edit submission_dir) | ADOPT_CANONICAL | the dry-run IS the canonical proof per the 12th-directive |
| Scratch output to `.omx/tmp/` not `submissions/` | ADOPT_CANONICAL | avoid `submissions/` pollution + Catalog #370 false-positive during validation |
| `--skip-protocol-verification` at Layer 0 | ADOPT (dry-prep) | V14-V2 is a substitution lane with no trainer/recipe pair; the placeholder trainer is for Layer 0 type-satisfaction only; the candidate's real evidence is the paired auth eval already on disk |
| Report the Layer 1 gap as an apparatus finding (not a candidate defect) | FORK_BECAUSE_PRINCIPLED | the candidate is byte-clean + frontier-anchored; the gap is in the canonical pipeline's member-name convention |

## 9-dimension success checklist evidence

1. **UNIQUENESS** — first $0 end-to-end exercise of the closed 7/7 pipeline.
2. **BEAUTY + ELEGANCE** — single CLI invocation surfaces the exact blocker.
3. **DISTINCTNESS** — validation-only; orthogonal to Sister D substrate work.
4. **RIGOR** — empirically confirmed the `x`-member root cause via
   `discover_section_specs_from_archive`; verified both paired axes on disk;
   verified attribution + apples-to-apples + frontier-pointer reconciliation.
5. **OPTIMIZATION PER TECHNIQUE** — CLI short-circuits at first failing layer;
   no wasted layers run.
6. **STACK-OF-STACKS COMPOSABILITY** — exercises all 7 layers' composition.
7. **DETERMINISTIC REPRODUCIBILITY** — exit 5 deterministic; archive sha + paired
   scores byte-stable.
8. **EXTREME OPTIMIZATION + PERFORMANCE** — $0; ~1s per dry-run; NO paid GPU.
9. **OPTIMAL MINIMAL CONTEST SCORE** — does NOT directly lower score;
   frontier-protecting validation that unblocks the canonical PR111 lifecycle.

## Cargo-cult audit per assumption

| Assumption | HARD-EARNED / CARGO-CULTED | Unwind |
|---|---|---|
| The V14-V2 candidate is PR-ready end-to-end | CARGO-CULTED (falsified) — Layer 1 `x`-member gap blocks the canonical pipeline | the candidate IS frontier-anchored + paired-verified; the gap is in the canonical Layer 1 |
| Layer 0 needs a real trainer/recipe for a substitution lane | HARD-EARNED — `build_compression_pipeline` requires a trainer/recipe pair; substitution lanes have none; `--skip-protocol-verification` + placeholder is the dry-prep path | a future substitution-lane mode could bypass Layer 0 trainer requirement |
| `0.bin` is the only monolithic member name | CARGO-CULTED — PR101/DQS1 grammar uses `x`; single-`x`-member IS monolithic | extend `discover_section_specs_from_archive` to recognize PR101-grammar single-member convention |
| The 6 canonical equations promote on any end-to-end run | CARGO-CULTED — they promote on PACKET-CLEAN (exit 4) only | gating is correct per Catalog #344; this run was exit 5 |

## Observability surface

| Facet | Implementation |
|---|---|
| Inspectable per layer | CLI `--json` report `layers.<layer_key>.{ok, error/substrate_id/...}` |
| Decomposable per signal | per-layer verdict + per-layer error; archive member breakdown; per-axis paired scores |
| Diff-able across runs | exit code 5 deterministic; archive sha byte-stable |
| Queryable post-hoc | this memo + the candidate's on-disk paired auth eval JSONs + frontier pointer |
| Cite-able | archive sha `0a3abfe6...`; CPU/CUDA scores axis-labeled; lane id; Layer 1 error string |
| Counterfactual-able | adding a `multi_file_justification` OR recognizing `x` as monolithic immediately changes Layer 1 from FAIL to PASS |

## 6-hook wire-in declaration per Catalog #125

- **Hook #1 SENSITIVITY_MAP** — N/A (validation run; no signal contribution)
- **Hook #2 PARETO_CONSTRAINT** — N/A
- **Hook #3 BIT_ALLOCATOR** — N/A
- **Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH** — **ACTIVE** (this memo's per-layer
  blocker table + the Layer 1 `x`-member gap is consumable by the
  `pr_submission_compliance_consumer` per Catalog #335 to weight PR111-readiness)
- **Hook #5 CONTINUAL_LEARNING_POSTERIOR** — **ACTIVE** (the exit-5 NAMED-BLOCKER
  verdict is the first empirical anchor for the Phase 9
  `full_lifecycle_cli_consolidation_savings_v1` equation; it records that the
  first end-to-end run was blocked at Layer 1, deferring the 6-equation
  promotion to a future PACKET-CLEAN run)
- **Hook #6 PROBE_DISAMBIGUATOR** — **ACTIVE** (the NAMED-BLOCKER-at-Layer-1
  verdict IS the canonical disambiguator: the V14-V2 candidate's blocker is
  apparatus-side, NOT candidate-side; the candidate is frontier-anchored +
  paired-verified)

## Sister coordination (Catalog #230 ownership map)

In-flight at landing (per `.omx/state/subagent_progress.jsonl` audit): Sister D
V15 EXTREMA cross-surface (`aebe0652dc7e63ed1`) owns the grayscale_lut + VQ-VAE
indices_blob substrate surfaces — SISTER-DISJOINT (this subagent is
submission-pipeline validation; ZERO overlap with substrate-grammar work).

Phase 10 owns: THIS memo (NEW). ZERO file collision. The dry-run wrote ONLY to
`.omx/tmp/phase10_dry_run/` (scratch, cleaned) + `.omx/state/subagent_progress.jsonl`
(checkpoints, Catalog #131-exempt). NO `submissions/` pollution. NO paid GPU. NO
`gh` execution. No new catalog claim (Phase 10 is a validation run, not a STRICT
gate).

## Cross-references

- Phase 9 CLI landing: `.omx/research/phase_9_operator_pr_submission_full_lifecycle_cli_landed_20260527.md`
- Phase 8 STRICT gate Catalog #370 landing: `.omx/research/phase_8_strict_gate_catalog_370_canonical_submission_compliance_landed_20260526.md`
- V14-V2 candidate report: `reports/pr111_candidate_cascade_a_fec10_substitution_onto_dqs1_frontier_landed_20260526.md`
- V14-V2 landing memo: `.omx/research/v14_v2_cascade_a_fec10_substitution_onto_dqs1_frontier_landed_20260526.md`
- T3 grand council V14-V2 PROMOTE-RATIFIED verdict: `.omx/research/t3_grand_council_negative_results_falsifications_bad_scores_comprehensive_review_landed_20260526.md`
- Canonical frontier pointer: `.omx/state/canonical_frontier_pointer.json`
- User attribution memory: `~/.claude/projects/-Users-adpena-Projects-pact/memory/user_pr_attribution.md`

**VERDICT: PHASE_10_PR111_CANDIDATE_DRY_RUN_VALIDATION_LANDED — canonical 7/7
pipeline exercised end-to-end at $0; NAMED-BLOCKER at Layer 1 (PR101/DQS1
`x`-member-convention gap); candidate is frontier-anchored + paired-verified +
attribution-clean; Catalog #370 strict-flip + 6-equation promotion deferred to a
future PACKET-CLEAN run after the Layer 1 gap is closed.**

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
