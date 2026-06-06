# Audit Decision Matrix: wf_81fb975b-c30

- Created UTC: 2026-06-06T21:34:07Z
- Authority scope: audit_decision_only_no_score_authority
- Overall decision: PARTIAL_MATRIX_BLOCKS_C_F_E
- Source workflow status: completed_but_partially_rate_limited (12/18 parsed audit rows)
- Score authority: none minted here; exact archive bytes plus upstream evaluator remain the only score authority.

## Canonical Result Inspection

- No workflow-local result/decision/matrix/summary/verdict file was present under the subagent workflow directory by filename scan.
- Outer workflow result record found: `/Users/adpena/.claude/projects/-Users-adpena-Projects-pact/89ff112f-013d-43b5-b949-2a6d43b650c3/workflows/wf_81fb975b-c30.json`
- The outer workflow record is copied into the bridge packet as raw workflow evidence, but it is partial and is not a clearance decision.

## Launch Gate Rule

No audited swarm output may feed v6 or the launch DAG unless its row below says `clear` or `remediated`, and the usual canonical gate for that family still passes. This matrix does not bypass source-forward, receiver-closed, exact replay, or custody gates.

## Target Decisions

| Target | Parsed lenses | Severity counts | Decision | v6/launch DAG consumption | Required action |
|---|---:|---|---|---|---|
| A-survival | 3/3 | H0 M0 L5 | clear | allowed_after_normal_canonical_gates | None from this audit. LOW notes are nonblocking and no score authority is minted. |
| B-composite | 3/3 | H0 M3 L4 | remediated | allowed_after_remediation_commit_and_normal_canonical_gates | Require current tree or descendant containing commit 2ef6c747a; keep report-count inaccuracies out of authority claims. |
| C-miner | 3/3 | H0 M1 L4 | blocked_pending_remediation | blocked | Before C-miner evidence can feed v6 or the launch DAG, the launch gate must require coverage_row.passed is true and validate accepted/diversity threshold fields with negative tests for failed or contradictory coverage rows. |
| D-action-effect | 3/3 | H0 M2 L3 | remediated | allowed_for_current_production_consumers_after_remediation_and_normal_canonical_gates | Require current tree or descendant containing b38e800f1; do not treat batch-local exact_nonrate as contest score authority. |
| F-pose-v5 | 0/3 | H0 M0 L0 | blocked_incomplete_audit | blocked | Re-run no_fake, claims_vs_reality, and integration_custody audits for commits 0b9a1ec99 and 68d382131. Verify the SSD training_artifact.json fields, the memo match, the post-47dfdd4c4 runner tests, and the claimed pre-existing failure attribution. |
| E-commutator | 0/3 | H0 M0 L0 | blocked_incomplete_audit | blocked | Re-run no_fake, claims_vs_reality, and integration_custody audits for commits 76b228749 and 729b1f2a9. Hand-check commutator arithmetic, basis unification, authority mismatch refusal, absent pair-row queueing, CLI fixture path, and no fabricated synergy values. |

## Medium Findings And Remediation

### B-composite
- [claims_vs_reality] Agent reported '4 new behavioral tests' but commit 2ce14c114 introduced 5 new tests; mutation-failing tests are #2 and #5 (not 'tests 1 and 4')
- [claims_vs_reality] Test-count claims '19 prior pass' and '208 hi_nerv tests green' are both inaccurate (actual: 18 prior, 216 green)
- [claims_vs_reality] Undisclosed latent state-leak in B's commit: frame1-safety RuntimeError path did NOT restore head_rgb_0 before raising; fixed by separate follow-up commit 2ef6c747a
- Remediation evidence: `2ef6c747a` Restore HiNeRV compensation state before safety raise (present_in_current_history)
- Decision: remediated

### C-miner
- [claims_vs_reality] Launch gate consumes coverage SCHEMA PRESENCE only, not coverage substance (passed/diversity counts ignored)
- Blockers: launch_gate_consumes_representative_coverage_schema_presence_not_substance
- Decision: blocked_pending_remediation

### D-action-effect
- [integration_custody] from_hinerv_birth_receipt reads exact_nonrate keys the real renderer never emits (synthetic-fixture-passed-as-real test masks a schema mismatch)
- [claims_vs_reality] from_hinerv_birth_receipt distortion aliases match NO production receipt producer; 'real schema roundtrips' test uses synthetic bare-key exact_nonrate (synthetic-fixture-as-real)
- Remediation evidence: `b38e800f1` Harden NeRV launch evidence gates (present_in_current_history)
- Decision: remediated

### F-pose-v5
- Blockers: all_three_F_audit_lenses_rate_limited_before_final_json
- Decision: blocked_incomplete_audit

### E-commutator
- Blockers: all_three_E_audit_lenses_rate_limited_before_final_json
- Decision: blocked_incomplete_audit

## Incomplete Audit Slots

- Transient retry error: journal.jsonl:22 was an API socket error for D-action-effect claims_vs_reality; the retry later parsed successfully and D has 3/3 lenses.
- journal.jsonl:30 - You've hit your weekly limit · resets Jun 9 at 12pm (America/Chicago)
- journal.jsonl:31 - You've hit your weekly limit · resets Jun 9 at 12pm (America/Chicago)
- journal.jsonl:32 - You've hit your weekly limit · resets Jun 9 at 12pm (America/Chicago)
- journal.jsonl:36 - You've hit your weekly limit · resets Jun 9 at 12pm (America/Chicago)
- journal.jsonl:37 - You've hit your weekly limit · resets Jun 9 at 12pm (America/Chicago)
- journal.jsonl:38 - You've hit your weekly limit · resets Jun 9 at 12pm (America/Chicago)

## Source Hashes

- Workflow record SHA-256: `e6b8963ec2fe99f779a6535ab794fbde5ce2bfe8984987f569e7a4bf9cf80915`
- Journal SHA-256: `14d85a43b1545d9bce07693c1fb3590896fb9320e5f32bae5d743b1356eab541`
