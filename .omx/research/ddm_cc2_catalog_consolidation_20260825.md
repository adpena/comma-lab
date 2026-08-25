# DDM-CC2 catalog consolidation review

**Date:** 2026-08-25  
**Harness task:** #1272  
**Snapshot:** `d3b84a54693bf54b6673c5246f519ab958c3f277`  
**Axis:** `[macOS-CPU static/source/git-history audit; scorer-free]`  
**Disposition:** recommendation-only; no catalog, preflight, `CLAUDE.md`, or upstream mutation  
**Pointer:** UNMOVED. This arm measured apparatus structure, not a candidate archive or score.

The live catalog has **285 numeric rows, 285 unique numeric IDs, 284 unique row names, and six
rows above #400**. All six are covered by the file-level quota waiver in `CLAUDE.md:2`. No row
meets the retirement rule. Five of the six over-quota rows can be folded into three existing
mechanism families without deleting their detectors; #406 should remain separately waived until
its claimed #332/#351 host identity is repaired or a valid numbered host is selected. This review
therefore recommends consolidation, not retirement and not a higher quota.

## §1. Census

### §1.1 Population structure first

The overage is **six distinct detector facts**, not a large fan-out population:

| population layer | measured denominator | decomposition |
|---|---:|---|
| Numeric catalog rows | 285 | 285 unique numbers; one repeated function identity |
| Rows above #400 | 6 | #401, #402, #404, #406, #407, #408 |
| Actionable umbrella families | 3 | confound integrity (3 overage rows), process-tree lifecycle (1), score-denominator authority (1) |
| Singleton overage facts | 1 | #406 DSL compile binding |
| Literal `Catalog #N` markers in `src/tac/preflight.py` | 2,411 occurrences / 263 unique IDs | reference fan-out, not catalog claims |

The six overage rows are not copies of one template. However, five already share a runtime or
registry mechanism with earlier gates. The concentration is therefore at the **family** layer, not
the individual finding layer. Counting every literal marker as a gate would repeat the #821 error:
markers #405, #417, and #513 are comments or umbrella references, while charter examples #812 and
#842 are task/incident IDs. The authoritative claimed-gate denominator remains the numeric catalog
rows in `docs/meta_bug_class_catalog.md`.

### §1.2 Authority crosswalk

| authority | measured result | evidence |
|---|---|---|
| `docs/meta_bug_class_catalog.md` | 285 numeric rows; max #408; six rows above #400 | overage rows at lines 650, 652, 654, 656, 658, 660 |
| Number state | next available number is 409 | `.omx/state/next_catalog_number.txt:1` |
| Claim audit | 490 JSONL rows; claimed history reaches #408 | `.omx/state/catalog-claim.log:483-490` |
| Preflight markers | 2,411 occurrences; 263 unique; literal max #513 | `src/tac/preflight.py`; unique markers above 400 are 405, 406, 407, 408, 417, 513 |
| Quota waiver | PRESENT in the required first 200 lines and names all six overage gates | `CLAUDE.md:2` |
| Live integrity checks | quota 0 findings; duplicate-number 0; strict-callsite-row 0 | direct read-only invocation on this snapshot |

The counter and catalog are consistent as different objects: the counter records the next claim,
not the number of live rows. Commits `abe58c7d2f`, `76401e87fa`, `fa5a671330`, `e66f225934`, and
`e4ab0626da` advance the state through the overage period. Gaps #403 and #405 and the absence of a
standalone #406 claim-log row do not create duplicate claims; #406 arrived with a state jump in
`fa5a671330`.

The preflight-marker maximum is **not** the catalog maximum. `src/tac/preflight.py:91469` uses
"Catalog #405" in a serializer-discipline comment, `src/tac/preflight.py:73624` references #417 in
an anti-fake explanation, and `src/tac/preflight.py:8218-8221` uses #513 as an umbrella reference.
None has a numeric authority row. The claimed catalog maximum is therefore #408, not #513.

## §2. Orphan and duplicate cross-check

### §2.1 Strict callsites under Catalog #176

`_check_176_collect_strict_callsites` measured **268 strict callsites** in `preflight_all()`:

| typed outcome | count | evidence |
|---|---:|---|
| Exact numbered catalog row | 184 | numeric `check_*` rows joined by function name |
| Legacy allowlist | 76 | `_CHECK_176_LEGACY_ALLOWLIST`, `src/tac/preflight.py:54268-54373` |
| Same-line `CLAUDE_MD_ENTRY_OK` waiver | 8 | callsite lines 5421, 8069, 8071, 8175, 8176, 8215, 8218, 8221 |
| Uncovered strict callsite | **0** | live `check_strict_preflight_callsites_have_claude_md_catalog_row` result |

The eight waived names are `check_no_unwaived_pyppmd_imports`,
`check_v9_fake_claim_guards`, `check_evidence_authority_claims_are_custodied`,
`check_no_reasoning_echo_instructions`, `check_subagent_contract_module_integrity`,
`check_worker_target_venv_dependency_closure_is_sealed`,
`check_modal_dispatch_claim_guard_precedes_write`, and
`check_modal_dual_ledger_matching_is_call_id_first`.

### §2.2 Catalog rows without a direct `preflight_all()` call

Ten numeric rows have no direct AST call node:

| typed outcome | rows | evidence and disposition |
|---|---|---|
| Registry-mediated, not orphaned | #397, #398, #399, #400, #401, #402, #404 | `CONFOUND_GATES` at `src/tac/confound_gates.py:3523` and loop at `src/tac/preflight.py:8048` |
| Deliberately standalone gate | #328 | function at `src/tac/preflight.py:73614`; not wired into `preflight_all()` |
| Numeric non-gate change rows | #225, #329 | catalog lines 328 and 464 describe a tool vocabulary change and a provenance enum extension, not `check_*` callables |

This produces **zero proven runtime orphans** in the Catalog #176 strict scope. It also exposes a
schema problem: #225 and #329 consume gate numbers despite not being gates, and #328 is a numbered
gate outside the canonical orchestrator. Those are consolidation debt, but this review does not
pretend they are uncovered strict callsites.

### §2.3 Duplicate and identity defects

| defect | count | evidence | recommendation |
|---|---:|---|---|
| Duplicate numeric IDs | **0** | 285 rows / 285 unique IDs | none |
| Duplicate row function identity | **1** | #203 at catalog line 296 and #224 at line 326 both name `check_modal_training_image_includes_hard_runtime_deps` | fold #224 text into #203 |
| Missing numbered host identity | **2 references** | catalog header line 12 claims a #351 scope extension, and #406 cites #332; neither #332 nor #351 has a numeric catalog row | do not use either as a consolidation host until repaired |
| Wrong sister cross-reference | **1** | #407 at catalog line 658 says `tac.contest_score` is #168, but #168 at line 242 is the Assign/AnnAssign AST gate; the contest-score gate is #391 at line 628 | correct to #391 in any adjudicated landing |

The #332/#351 issue is not counted as two new gate orphans: their associated callsites are covered
by same-line #176 waivers or other rows. It is an **operator-facing catalog identity gap**, which is
exactly why #406 cannot honestly be declared consolidated into those numbers today.

## §3. Consolidation families

| family | member gates and surfaces | shared bug-class signature | proposed umbrella shape | coverage-loss risk |
|---|---|---|---|---|
| C1: training/verdict confound integrity | #397-#402 plus #404; overage members #401, #402, #404 | a measurement or verdict can look valid while the training state, sample denominator, telemetry liveness, or decision denominator is invalid | Expand numbered #397 into a registry umbrella that retains every predicate and a per-member strictness map; keep `CONFOUND_GATES` as the enumeration SoT | **Medium:** a single Boolean strictness would wrongly strict-block the live #402/#404 backlogs or weaken strict #401; severity must remain per predicate |
| C2: process-tree lifecycle | #389 detached wrapper, #408 timed synchronous wrapper, unnumbered retry/descendant check | the parent lifecycle action does not control or observe the real descendant process group | Replace the two direct strict calls with one numbered #389 umbrella callable that invokes the existing detector bodies and preserves both positive controls | **Medium:** an over-broad process regex would lose #408's function-scoped argv resolution; detectors must remain separate internally |
| C3: contest score and denominator authority | #268 unit calibration, #391 no hand-rolled score, #407 dynamic upstream-video denominator | score arithmetic is trusted while its rate unit, canonical helper, or live denominator inventory is wrong | Extend numbered #391 as the catalog umbrella; retain #407's warn-only inventory checker and the fail-closed `rate_term` guard as separate enforcement layers | **Low:** catalog folding is safe only if the hard runtime guard remains; do not turn the dotfile warning into broad strict preflight |
| S1: DSL compile provenance | #406 plus unnumbered/#176-waived #332/#351 apparatus | launch admission is not bound to the typed compiler, LawRefs, and exact argv | **No safe numbered host identified.** Keep #406 and its current waiver until the #332/#351 catalog identity gap is adjudicated | **High:** folding into generic score provenance would blur launch admission and could hide a fail-open path |
| E1: Modal runtime dependency row duplication | #203 and #224 | one callable, with #224 explicitly an extension of #203 | Move #224's DALI/NVML extension text into #203 and remove the duplicate row identity | **Low:** no callable or strictness change |

No scope reduction is proposed inside any detector. The consolidation unit is the catalog identity
and, where safe, the orchestrator wrapper. Detector algorithms, target globs, waiver rules,
positive controls, and current strict/warn modes remain unchanged.

## §4. Retirement candidates

**Retirement candidates: NONE.** The repository has no centralized cumulative gate-fire ledger, so
the exact since-landing count is `NOT-MEASURED` unless a durable receipt documents a fire. Current
finding counts are fresh on this snapshot; they are not misrepresented as cumulative history.

| gate | landing | cumulative fires since landing | fresh current findings | structural-extinction test | retirement verdict |
|---|---|---:|---:|---|---|
| #401 | `221670d74ad` family landing; strict flip `e8e2ed751a` | NOT-MEASURED | 0 | default is cured but regressible; positive control was expanded in `ddm_gc16...:144` | NOT ELIGIBLE |
| #402 | `221670d74ad` family landing | NOT-MEASURED | **1** at `experiments/train_levelset_witness_realized_through_R_mlx.py:11758` | live violation proves class not extinct | NOT ELIGIBLE |
| #404 | `43b2108177` / catalog row `66355e1e94` | NOT-MEASURED; one Stop-hook correction documented | **15** (report cap) | `ddm_pz1_pose_axis_cx1_base_20260803.md:28-51` records a real relative-significance correction | NOT ELIGIBLE |
| #406 | `fa5a671330` | **at least 2 documented; exact cumulative NOT-MEASURED** | 0 | `catalog406_332_backfill_current_measurement_20260717.md:8-15` found two launcher defects; later fixed in `c75cdca704` | NOT ELIGIBLE |
| #407 | `e66f225934` / row `4a5504fe51` | NOT-MEASURED | 0 | live tree is clean, but macOS dotfiles are a recurring external state and hard `rate_term` enforcement remains load-bearing | NOT ELIGIBLE |
| #408 | `fc1bc17f90` | **at least 1 documented; exact cumulative NOT-MEASURED** | 0 | `591e412319` migrated `experiments/ddm_jo2_receiver_close.py:1081` after a real post-landing fire | NOT ELIGIBLE |

The annual-audit rule requires both zero findings and structural extinction. A current zero is not
enough. #402 and #404 are currently nonzero, #406 and #408 have documented post-landing catches,
and #401/#407 protect state that can recur.

## §5. Ranked recommendation table

| rank | affected row | recommendation | exact Catalog #299-compliant landing shape |
|---:|---|---|---|
| 1 | #408 | **CONSOLIDATE-INTO-#389** | In one replacement commit, rename/extend #389 to a process-tree-lifecycle umbrella callable, retain #389 and #408 detector bodies and tests, preserve strict mode for both, add the `ddm_jo2` regression as a positive control, replace both direct orchestrator calls with the umbrella call, and delete the #408 row. This is #299 replacement path (c), not retirement. |
| 2 | #401 | **CONSOLIDATE-INTO-#397 umbrella** | Extend #397's row into the confound-registry catalog identity; retain #401 predicate, strict membership, tests, and live positive control. Delete only #401's separate row in the same replacement commit. |
| 3 | #402 | **CONSOLIDATE-INTO-#397 umbrella** | Same landing as #401, but preserve #402 as warn-only until its current finding is cured. The umbrella must carry a per-member severity map. Delete only the redundant #402 row, not the detector. |
| 4 | #404 | **CONSOLIDATE-INTO-#397 umbrella** | Same registry landing; preserve the static detector, Stop-hook mechanism, max-report behavior, and warn-only backlog. Add an explicit measurement-interpretation member label and delete only the #404 row. |
| 5 | #407 | **CONSOLIDATE-INTO-#391** | Extend #391's catalog text to cover canonical formula plus live denominator inventory, correct the false #168 cross-reference, retain the fail-closed `rate_term` guard and warn-only #407 checker, retag its callsite as a #391 scope extension, and delete the #407 row in the same commit. |
| 6 | #406 | **KEEP** | Keep #406 and the existing `CLAUDE.md:2` waiver. Do not claim `CONSOLIDATE-INTO-#332/#351` until one of those identities is a real numbered catalog host or the operator selects another semantically exact host. Re-run the launch-sink census before any later replacement. |
| 7 | #224 | **CONSOLIDATE-INTO-#203** | Docs-only catalog cleanup: append #224's runtime-env scope to #203 and remove the duplicate #224 row. No callable, callsite, strictness, or test change. |

`QUOTA-WAIVER-NEEDED`: **none for the current six**, because `CLAUDE.md:2` already provides an
operator-specific waiver naming all six and its rationale is substantive. Keep that waiver until
the corresponding replacement landings are complete. After the five recommended overage folds,
the numeric claim state must still remain 409; consolidation does not rewrite history or authorize
a new #409 gate. Future strict work should use an existing umbrella or satisfy #299 replacement or
fresh operator-waiver rules.

## §6. Prior-law prediction verdict

**Verdict: PARTIAL HIT, with the universal form falsified.**

- The population is small: six overage rows, within the predicted ceiling of six families.
- Five of six rows fit three mechanism-coherent families, each with at least three sister surfaces.
- The falsifier does not fire: the overage is not predominantly singleton; only #406 is a singleton
  under the authoritative numeric catalog.
- The stronger wording that *every* overage row would have a safe existing umbrella is false.
  #406's apparent #332/#351 sisters do not have numbered authority rows, and using generic score
  provenance as a host would reduce mechanism fidelity.

So the #821/#1085 concentration law is useful for routing this population, but it is not a licence
to force a semantically weak umbrella around the last singleton.

## RECALL EVIDENCE

### Sources and queries

- Full top-level research corpus searches: `rg -n -i "catalog consolidation|Catalog #299|quota|past-#400|retire.*gate|gate.*retire|fire history|scope extension" .omx/research` and narrower searches for `#401`, `#402`, `#404`, `#406`, `#407`, `#408`, `#821`, `#1085`, and `#1149`.
- Reference census form recovered from commit `57c87898c2`:
  `.omx/research/ddm_ca1_20260805/CA1_RECEIPT.md`.
- Prior population warnings: `.omx/research/ddm_lg2_arity_mismatch_three_rows_20260802.md`,
  `.omx/research/ddm_hd1_na9_hazard_hardening_20260818.md`, and
  `.omx/research/ddm_sp3_preflight_debt_consolidation_20260819.md`.
- Post-landing evidence: `.omx/research/catalog406_332_backfill_current_measurement_20260717.md`,
  `.omx/research/ddm_dg1_rate_denominator_guard_20260731.md`,
  `.omx/research/ddm_gc16_dev_gate_denominator_and_control_ratchet_20260801.md`,
  `.omx/research/ddm_pz1_pose_axis_cx1_base_20260803.md`, and
  `.omx/research/ddm_kg1_killclass_backlog_20260821.md`.
- Canonical equations: `.venv/bin/python tools/list_canonical_equations.py --json`, then searches
  over equation ID/name/summary for `catalog`, `quota`, `preflight`, and `consolidation`.
- Research index/DAG/task surfaces: content searches of `.omx/research/CANONICAL_RESEARCH_INDEX*`,
  `sub015_DAG_*`, `.omx/state/main_hot_state.md`, and `.omx/state/codex_arm_queue.jsonl` for the
  same terms and task #1272.
- Git history: bounded `git log` / `git show` searches for the six overage rows, their landing
  commits, catalog counter transitions, and post-landing fixes.

### What was found beyond the charter seeds

1. #408 really fired after landing; `591e412319` is not hypothetical retirement resistance.
2. #406 had two documented findings on 2026-07-17 even though its current count is zero.
3. #402 is currently nonzero and #404 returns its 15-row report cap.
4. #203/#224 duplicate one callable identity.
5. #332/#351 are cited as catalog hosts but have no numeric rows in the authority document.
6. #407's #168 sister citation is wrong; the intended contest-score identity is #391.
7. The literal-marker and task-ID examples far above 400 do not enlarge the claimed-gate
   population.

### What recall changed

The CA1 receipt changed the plan from a prose review to an explicit denominator plus typed-outcome
join. The #821/#1085 evidence forced registry/template fan-out measurement before ranking. The
#406/#408 receipts eliminated retirement. The missing #332/#351 rows prevented a false
`CONSOLIDATE-INTO` claim for #406, while the #391 discovery created a safer host for #407. No
canonical equation specific to catalog consolidation was found in the searched equation metadata,
and no additional load-bearing catalog-consolidation decision was found in the bounded research
index/DAG scope.

## Boundaries

- No scorer, evaluator, Modal, GPU, or paid dispatch ran.
- `upstream/` was read-only and unchanged.
- No payload was materialized; the payload-retention rule was not triggered.
- No cumulative fire telemetry store was found, so unproved histories are labeled
  `NOT-MEASURED`.
- No recommendation here authorizes edits to the catalog, preflight, or waiver. MAIN and the
  operator own adjudication.
- Concurrent edits to four unrelated `.omx/research/*.md` files were preserved and excluded from
  this arm.

## LIVE-HYPOTHESES

- A per-member-severity registry umbrella can remove #401/#402/#404 catalog duplication without
  changing any detector behavior, because all three already execute through `CONFOUND_GATES`.
- A single #389 process-lifecycle wrapper can preserve both strict detectors and reduce direct
  orchestration surface, because #389 and #408 differ in detection but share the process-group
  invariant.
- The #332/#351 identity gap may be extraction drift rather than absent doctrine, because both
  names are referenced by live preflight comments and a catalog-header scope extension.

## DEAD-ENDS

- **RETIRE any overage row:** closed; current findings or documented post-landing catches defeat the
  required zero-fire plus structural-extinction test.
- **Treat literal #513/#417 or task #812/#842 as claimed gates:** closed; none is a numeric authority
  row.
- **Consolidate #406 into #332/#351 now:** closed on the current authority document; neither is a
  numbered host, so the disposition would be a fake catalog claim.
- **Use #168 as the score umbrella for #407:** closed; #168 is the Assign/AnnAssign AST gate. #391 is
  the actual contest-score authority row.

## NEXT_IF_RESUMED

- **QUEUED FOR OPERATOR ADJUDICATION** — owner: MAIN + operator; consumer store:
  `docs/meta_bug_class_catalog.md` and `src/tac/preflight.py`; fire trigger: explicit approval of the
  C1 #397 confound-registry replacement landing.
- **QUEUED FOR OPERATOR ADJUDICATION** — owner: MAIN + operator; consumer store:
  `docs/meta_bug_class_catalog.md` and `src/tac/preflight.py`; fire trigger: explicit approval of the
  C2 #389 process-lifecycle replacement landing.
- **QUEUED FOR OPERATOR ADJUDICATION** — owner: MAIN + operator; consumer store:
  `docs/meta_bug_class_catalog.md`, `src/tac/contest_score.py`, and `src/tac/preflight.py`; fire
  trigger: explicit approval of the C3 #391 score-authority scope extension.
- **QUEUED FOR IDENTITY REPAIR** — owner: MAIN + catalog custodian; consumer store:
  `docs/meta_bug_class_catalog.md`; fire trigger: operator chooses whether #332/#351 are restored
  numbered identities or #406 remains permanently separately waived.

**Own-vehicle frontier: gb1 — S 0.14811799921260607 @ 180,215 B [contest-CUDA T4, n600]; pointer UNMOVED by DDM-CC2.**
