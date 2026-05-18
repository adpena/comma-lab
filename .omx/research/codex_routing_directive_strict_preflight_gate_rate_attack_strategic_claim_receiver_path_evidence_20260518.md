# Codex routing directive: STRICT preflight gate for rate-attack strategic-claim receiver-path-evidence
# Date: 2026-05-18
# Authority: META-audit (commit e86ca6d0c) §2 recommendation + Codex F1 finding (commit 35b06f9ec) + ADVERSARIAL paradigm challenger (commit 4c6e46bfa) + cargo-cult burn-down supplement
# Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable: EVERY adversarial-review finding MUST be addressed with TWO landings — the fix AND a STRICT preflight check
# Per operator standing directive "burn down all cargo culted" + "all operator decisions approved"

## CANONICAL POINTERS

1. `/Users/adpena/Projects/pact/CLAUDE.md` (FULL; especially "Bugs must be permanently fixed AND self-protected against" + Catalog #229 premise verification + Catalog #287 evidence tags + Catalog #6 strict-scorer-rule)
2. `/Users/adpena/Projects/pact/AGENTS.md`
3. `.omx/research/meta_audit_conflate_declarative_with_physical_error_pattern_12_claim_self_audit_20260518.md` (commit e86ca6d0c; THE META-audit; §2 recommended this gate)
4. `.omx/research/rate_attack_legal_receiver_path_audit_codex_f1_finding_relay_20260518.md` (commit 35b06f9ec; canonical legal-receiver-path classification taxonomy)
5. `.omx/research/cargo_cult_burn_down_supplement_extending_meta_audit_across_session_20260518.md` (cargo-cult burn-down supplement)
6. `src/tac/preflight.py` (canonical strict preflight surface)
7. `tools/claim_catalog_number.py` (transactional catalog # claim per Catalog #186)

## STRATEGIC FRAMING (the bug class this extincts)

**Bug class**: CONFLATE_DECLARATIVE_PROPERTY_WITH_PHYSICAL_IMPLEMENTATION — claiming a rate-attack vector / exploit / framing "works" based on a DECLARATIVE PROPERTY (e.g., "dim 7-12 are unscored") WITHOUT verifying the PHYSICAL IMPLEMENTATION (where information lives + what receiver path is needed + whether receiver path is legal per strict-scorer-rule + L4 LOC budget).

**Empirical anchors** (META-audit 12-claim self-audit):
- F1 (Codex caught): "dim 7-12 are free byte channel in archive" → false; collapsed to A2
- F3-F6 (my F1-audit caught): same pattern at deeper layers; all need scorer-load at inflate
- A1 original framing (operator corrected): "scorer-feature-space encoding requires full PoseNet → STRICT_SCORER_RULE_VIOLATION" — too conservative; A1-SPECIALIZED is RECLAIMABLE_VIA_PACKET_COMPILER per Codex's Section 0 correction
- 12 total instances in META-audit

**Cost of not extincting**: every rate-attack strategic-claim memo without receiver-path evidence becomes a CARGO-CULTED design artifact that wastes downstream engineering resources OR worse, leads to invalid contest submissions.

## WHAT CODEX BUILDS

### Phase 1: Claim Catalog # transactionally

```bash
.venv/bin/python tools/claim_catalog_number.py claim --commit-via-serializer --reason "STRICT preflight gate: rate-attack strategic-claim receiver-path-evidence per META-audit §2"
```

Returns next-available catalog # (e.g., #335 estimated based on Multi-loop F's transactional claim of #332).

### Phase 2: STRICT preflight check at `src/tac/preflight.py`

```python
def check_rate_attack_strategic_claim_has_receiver_path_evidence(
    *, strict: bool = False, verbose: bool = False
) -> list[str]:
    """Catalog #<N>: refuses strategic-claim-relay artifacts in
    `.omx/research/rate_attack_*.md` + `.omx/research/codex_routing_directive_rate_attack_*.md`
    + `.omx/research/*rate_attack*.md` + `.omx/research/*scorer_blind*.md`
    that make exploit claims (matching patterns like "encode in X", "exploit X",
    "free bytes in X", "scorer-blind to X") without adjacent:

    1. WHERE evidence: which artifact bytes / which scorer-internal state /
       which decoder-side info (citing source like upstream/modules.py:NN)
    2. RECEIVER PATH evidence: one of
       - NO_RECEIVER_NEEDED (scorer IS the receiver via scorer-blind input)
       - LEGAL_RECEIVER_IN_BUDGET (with LOC + dep count vs HNeRV parity L4 ≤200 LOC + ≤2 deps)
       - RECLAIMABLE_VIA_PACKET_COMPILER (A1-SPECIALIZED; with binary size target +
         contest_one_video_replay mode citation)
       - STRICT_SCORER_RULE_VIOLATION (with reactivation criteria per Catalog #325)
    3. CARGO-CULT classification per Catalog #303: HARD-EARNED-VERIFIED
       (with source-trace) / CARGO-CULTED-PENDING-EMPIRICAL (with test plan)

    Acceptance:
    - same-line waiver `# RATE_ATTACK_CLAIM_PATH_PENDING_OK:<rationale>` for
      in-progress claims being designed (placeholder `<rationale>` literal rejected
      so the gate's docstring example cannot self-waive)
    - file-level waiver `# RATE_ATTACK_CLAIM_PATH_AUDIT_DEFERRED_OK:<rationale>`
      in first 30 lines for audit-deferred artifacts

    Excluded path markers: `experiments/results/` + `_intake_` + `.omx/oss_export/`
    + `vendored/` + `/tests/` + `test_*` files.

    Sister of Catalog #287 (empirical claims have evidence tag) applied
    specifically to rate-attack strategic claims. Sister of Catalog #6
    (strict-scorer-rule) applied at the claim-relay surface.
    """
```

### Phase 3: Wire into `preflight_all()`

Initial wire-in WARN-ONLY per CLAUDE.md "Strict-flip atomicity rule" — backfill the 12+ self-audit instances + any other surfaced cases.

### Phase 4: CLAUDE.md catalog row

```
N. `check_rate_attack_strategic_claim_has_receiver_path_evidence` — META-audit §2 self-protection 2026-05-18. Refuses strategic-claim-relay artifacts in `.omx/research/rate_attack_*.md` + `.omx/research/codex_routing_directive_rate_attack_*.md` that make exploit claims without adjacent (a) WHERE evidence (source-trace) (b) RECEIVER PATH evidence (NO_RECEIVER_NEEDED / LEGAL_RECEIVER_IN_BUDGET / RECLAIMABLE_VIA_PACKET_COMPILER / STRICT_SCORER_RULE_VIOLATION) (c) CARGO-CULT classification per Catalog #303. Bug class anchor: F1 (Codex F1 finding relay commit 35b06f9ec) + 12 self-audit instances per META-audit (commit e86ca6d0c). Sister of Catalog #287 (empirical claims have evidence tag) applied to strategic claims. Sister of Catalog #6 (strict-scorer-rule) applied at the claim-relay surface. Initial wire-in WARN-ONLY per "Strict-flip atomicity rule"; live count at landing: backfill pending. Memory: `feedback_strict_preflight_gate_rate_attack_strategic_claim_receiver_path_evidence_landed_20260518.md`. Lane: `lane_strict_preflight_gate_rate_attack_strategic_claim_receiver_path_evidence_20260518` L1.
```

### Phase 5: Dedicated tests

`src/tac/tests/test_check_<N>_rate_attack_strategic_claim_receiver_path_evidence.py`:
- Test positive: synthetic memo with exploit claim missing receiver-path evidence → flagged
- Test negative: synthetic memo with NO_RECEIVER_NEEDED + WHERE evidence + HARD-EARNED-VERIFIED → not flagged
- Test waiver semantics: same-line + file-level + placeholder rejection
- Test exempt markers
- Test live-repo regression guard

## DISCIPLINE

- Catalog #229 premise verification BEFORE editing
- Catalog #117/#157/#174 commit serializer with POST-EDIT sha
- Catalog #186 catalog # claim transactional
- Catalog #206 checkpoint discipline
- Catalog #131 fcntl-locked writes (N/A for this gate; pure source-text scan)
- Catalog #314 absorption avoidance: scope is `src/tac/preflight.py` + `src/tac/tests/test_check_<N>_*.py` + CLAUDE.md row + memory entry
- Catalog #174 + #185 META-meta protections enforced via sister gates

## EXIT CRITERIA

- [ ] Catalog #N claimed via canonical helper
- [ ] `check_rate_attack_strategic_claim_has_receiver_path_evidence` lands in `src/tac/preflight.py`
- [ ] Wire into `preflight_all()` WARN-ONLY initial wire-in
- [ ] CLAUDE.md catalog row appended
- [ ] Dedicated tests pass
- [ ] Memory entry lands per Catalog #229+#287 evidence-tag discipline
- [ ] codex_persistent_session_state row appended
- [ ] canonical_task_status row updated 'completed'
- [ ] Lane `lane_strict_preflight_gate_rate_attack_strategic_claim_receiver_path_evidence_20260518` L1

## OPERATOR-FACING NOTE

This gate structurally extincts the CONFLATE_DECLARATIVE_PROPERTY_WITH_PHYSICAL_IMPLEMENTATION bug class at the rate-attack strategic-claim surface. After this lands, any future Claude rate-attack research subagent CANNOT relay a strategic claim without receiver-path-evidence + WHERE-source-trace + CARGO-CULT classification. Catalog #229 premise-verification-before-edit is applied PROSPECTIVELY (not retroactively) at the claim-relay surface.

Sister of CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable: every adversarial-review finding MUST be addressed with the fix AND a STRICT preflight check. Codex's F1 finding (the fix being the F-category collapse to A2 + A1-SPECIALIZED reclamation) gets its sister structural protection via THIS gate.

— Main-Claude 2026-05-18 (META-audit §2 routing per operator "burn down all cargo culted" + "all operator decisions approved")
