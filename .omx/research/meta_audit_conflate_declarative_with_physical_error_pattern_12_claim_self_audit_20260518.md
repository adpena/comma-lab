# META-audit: CONFLATE_DECLARATIVE_WITH_PHYSICAL error pattern — 12-claim self-audit
# Date: 2026-05-18
# Operator question 2026-05-18 verbatim: "have we made similar interpretation or translation or strategy errors or mistakes or errors anywhere else in the past few days like you almost did with A2 binary distillation? all of your recommended next steps are approved"
# Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable + "META-ASSUMPTION ADVERSARIAL REVIEW" Catalog #291 + Catalog #303 cargo-cult audit + Catalog #229 premise-verification-before-edit

## CANONICAL POINTERS

1. `/Users/adpena/Projects/pact/CLAUDE.md` (FULL; especially "Bugs must be permanently fixed AND self-protected against" + "META-ASSUMPTION ADVERSARIAL REVIEW" Catalog #291 + Catalog #229 + Catalog #303 + Catalog #6 strict-scorer-rule + HNeRV parity L4 inflate.py LOC budget)
2. `/Users/adpena/Projects/pact/AGENTS.md`
3. `.omx/research/rate_attack_legal_receiver_path_audit_codex_f1_finding_relay_20260518.md` (commit 35b06f9ec; the F1-finding audit that surfaced the error)
4. `.omx/research/adversarial_rate_attack_paradigm_challenger_20260518.md` (commit 4c6e46bfa; ADVERSARIAL caught 3 of the 12 claims)
5. `.omx/research/wiring_integration_orphan_audit_post_12_landings_20260518.md` (commit b1aae8536; sister META-audit at the wire-in surface)

## THE ERROR PATTERN

**CONFLATE_DECLARATIVE_WITH_PHYSICAL**: claiming a vector / exploit / framing "works" or "achieves X" based on a DECLARATIVE PROPERTY (e.g., "dim 7-12 are unscored") WITHOUT verifying the PHYSICAL IMPLEMENTATION (e.g., "where do those dim 7-12 bytes live in archive.zip? what's the legal receiver path? does the receiver path bust the LOC budget or violate strict-scorer-rule?").

Codex caught the F1 instance via Catalog #229 source-verification discipline applied AFTER my claim. The pattern is structural — I generated the brainstorm aggressively (good) but didn't pre-verify each claim against strict-scorer-rule + L4 LOC budget + composition_alpha sub-additive default BEFORE relaying them as canonical.

## SECTION 1 — 12-CLAIM SELF-AUDIT (errors from past few days)

### CONFLATE_DECLARATIVE_WITH_PHYSICAL family (4 claims; same pattern as F1)

#### Claim 1 — F1 "dim 7-12 are free byte channel in archive"
- **What I claimed**: PoseNet dim 7-12 are unscored → free byte channel in archive
- **Why it was wrong**: dim 7-12 are OUTPUTS of PoseNet's forward pass on RGB inputs; they don't exist in archive bytes
- **Caught by**: Codex (operator-relayed 2026-05-18)
- **Resolution**: F1 collapses to A2 (Adversarial steganography on scorer blind-spots); the actual exploit is RGB perturbations that PoseNet first-6-dim is invariant to
- **Status**: ✓ RESOLVED via `rate_attack_legal_receiver_path_audit_codex_f1_finding_relay_20260518.md`

#### Claim 2 — F3-F6 "encode in PoseNet vision(2048) / summary(512) / ResBlock / Hydra trunk"
- **What I claimed**: encode information in scorer intermediate features
- **Why it was wrong**: same as F1 at deeper layers; all require scorer-load at inflate to invert
- **Caught by**: my own F1-audit (sister analysis)
- **Resolution**: STRICT_SCORER_RULE_VIOLATION at full-scorer-weight scale; RECLAIMABLE via binary distillation per A1-distillation subagent (acb1a03a8e4234022 in flight) + operator's "we can get the stuff extremely small" confirmation
- **Status**: ⚠️ RECLAIMABILITY under empirical test

#### Claim 3 — A1 original framing "scorer-feature-space encoding requires full PoseNet → STRICT_SCORER_RULE_VIOLATION"
- **What I claimed**: A1 is forbidden; ~73MB rate hit
- **Why it was wrong** (operator-corrected): assumed full PoseNet weights; didn't consider distillation/specialization/Zig/sparseness/ablation/etc.
- **Caught by**: operator via "should we be able to engineer this into an extreme small and optimized binary using something like zig?" + "we can get the stuff extremely small"
- **Resolution**: A1 REOPENED for binary-distillation viability test; A1-distillation subagent (acb1a03a8e4234022) producing canonical design memo
- **Status**: ⚠️ RECLAIMABILITY under empirical test

#### Claim 4 — META-category "STRUCTURAL INFORMATION NOT SHIPPED" unification
- **What I claimed**: cooperative-receiver + deterministic-optimizer + rate-attack lineages all converge on ONE meta-paradigm
- **Why it was wrong (partially)**: 3 distinct sub-paradigms hidden under one label per Tao's mathematical critique; mathematical formulations DIFFER across the 3
- **Caught by**: Tao (ADVERSARIAL §3 binding critique)
- **Resolution**: META-paradigm unification is STRUCTURALLY USEFUL as a coordination framework but MATHEMATICALLY CONFLATED; needs more rigorous treatment per Tao
- **Status**: ⚠️ NEEDS RE-FRAMING by SYNTHESIS-V2 (a18c228872a761bdb in flight)

### NUMERICAL / PREDICTION CONFLATIONS (3 claims; caught by ADVERSARIAL)

#### Claim 5 — "43 vectors" cardinality
- **What I claimed**: 43 distinct rate-attack vectors organized into 8 sub-categories
- **Why it was wrong (partially)**: numerically arbitrary per Contrarian; after F-category collapse to A2 it's ~36 distinct; further reductions possible after legal-receiver-path audit
- **Caught by**: Contrarian (ADVERSARIAL §3)
- **Resolution**: cardinality is incidental; what matters is per-vector empirical viability; SYNTHESIS-V2 will produce reconciled count
- **Status**: ⚠️ NEEDS RE-COUNT post-SYNTHESIS-V2

#### Claim 6 — "TOP-5 aggregate predicted band [0.152, 0.180]"
- **What I claimed**: TOP-5 vectors compose to aggregate ΔS [-0.040, -0.012]
- **Why it was wrong**: assumed composition_alpha ADDITIVE when Catalog #322 empirical default is SUB-ADDITIVE per 9×9 matrix evidence
- **Caught by**: Boyd (ADVERSARIAL §3 composition-infeasibility critique)
- **Resolution**: predicted band requires Dykstra-feasibility check per Catalog #296; aggregate prediction was likely OVER-OPTIMISTIC
- **Status**: ⚠️ NEEDS REVISED BAND with empirical composition_alpha measurement

#### Claim 7 — Cross-stack synthesis "cheap-probe family composes to [0.182, 0.189]"
- **What I claimed**: synthesis OP-1+OP-2+OP-6+OP-7+OP-10 composes to frontier displacement [0.182, 0.189]
- **Why it was wrong**: same as #6 at the cross-stack synthesis layer
- **Caught by**: Boyd's critique generalizes
- **Resolution**: SUB-ADDITIVE composition default per Catalog #322; needs empirical verification
- **Status**: ⚠️ NEEDS REVISED BAND

### SCOPE / FRAMING OVERSTATEMENTS (5 claims; my own pattern)

#### Claim 8 — G1 "frontier displacement via re-rank existing PUBLIC dual-eval data"
- **What I claimed**: G1 can move the frontier via re-ranking existing PR101+102+103+106+107 dual-eval data
- **Why it was partially wrong**: re-ranking PUBLIC archives doesn't move the frontier (PUBLIC archives have fixed scores; we don't control them); G1's actual exploit is choosing which of OUR submission candidates to submit by CPU not CUDA
- **Caught by**: self-audit now (post-operator's question)
- **Resolution**: G1's real value is in OUR submission choice; we'd need multiple competitive candidates where best-CPU ≠ best-CUDA; need to verify whether this condition holds for current local candidates
- **Status**: ⚠️ NEEDS CLARIFICATION in synthesis; G1 routing directive 8ebea02ef should be revised to focus on OUR candidates not PUBLIC

#### Claim 9 — "14+ strategic landings today"
- **What I claimed**: 14+ landings in this session segment
- **Why it was partially wrong**: liberal counting; some are routing directives (research artifacts), some are design memos (research artifacts), some are audits (research artifacts); few are actual implementation
- **Caught by**: wiring/integration/orphan audit (commit b1aae8536) — 48/96 cells PLANNED_BUT_UNROUTED
- **Resolution**: distinguish DESIGN-LANDINGS vs IMPLEMENTATION-LANDINGS; audit is correct
- **Status**: ✓ ACKNOWLEDGED; future counts should distinguish

#### Claim 10 — E-category "NVDEC / NVJPEG free hardware decode on T4"
- **What I claimed**: T4 hardware decodes AV1/HEVC/JPEG for FREE in compute
- **Why it was partially wrong**: didn't verify inflate.py LOC + dep budget per HNeRV parity L4; NVDEC via PyAV/ffmpeg may bust ≤2 dep budget
- **Caught by**: self-audit during F1-audit (table classified some as LEGAL_RECEIVER_OVER_BUDGET)
- **Resolution**: per-vector budget audit needed; some E-category vectors may need binary distillation OR bust budget
- **Status**: ⚠️ NEEDS PER-VECTOR BUDGET AUDIT

#### Claim 11 — D-category "YUV-native skip RGB conversion"
- **What I claimed**: encode in YUV space to skip RGB conversion at inflate
- **Why it was wrong**: scorer takes RGB; conversion happens at inflate either way; "skip" was misleading (we'd just MOVE the conversion to a different layer of the inflate.py code)
- **Caught by**: self-audit now
- **Resolution**: YUV-native is still LEGAL_RECEIVER_IN_BUDGET but framing was misleading; the actual benefit is encoding efficiency in YUV space (better entropy for chroma at lower bit depth) not "skipping" conversion
- **Status**: ⚠️ NEEDS FRAMING CLARIFICATION

#### Claim 12 — "Hotz binding directive PROCEED IMMEDIATELY $0/~50 LOC"
- **What I claimed**: PRIMARY's G1 design says $0 cost, ~50 LOC, this session
- **Why it was partially wrong**: relayed PRIMARY's optimistic estimate without independent verification; the actual LOC may be higher; the empirical $0 cost claim depends on G1 actually achieving frontier displacement (Claim 8 issue)
- **Caught by**: self-audit now (G1 is more nuanced than "PROCEED IMMEDIATELY")
- **Resolution**: G1 routing directive 8ebea02ef should fire as DIAGNOSTIC + INFORMATIONAL not as frontier-displacement-guaranteed; Codex executes; verdict informs next-step
- **Status**: ⚠️ NEEDS EXPECTATION RESET

## SECTION 2 — MITIGATION (what's already in place vs what needs to be added)

### Already in place (canonical structural protection)

1. **Catalog #229 premise-verification-before-edit** — Codex applied this to catch F1; my future strategic claims should pre-verify against this
2. **Catalog #292 per-deliberation assumption surfacing** — ADVERSARIAL's quartet (Tao+Carmack+Hotz+Boyd) applied this to catch claims 4-6
3. **Catalog #325 per-substrate symposium discipline** — requires PROCEED-unconditional before paid empirical
4. **Catalog #303 cargo-cult audit per assumption** — HARD-EARNED vs CARGO-CULTED classification
5. **Catalog #6 strict-scorer-rule** — bounded the false A1 framing (intent vs letter analysis in A1-distillation subagent)
6. **Catalog #322 composition_alpha sub-additive default** — should have prevented claims 6-7
7. **HNeRV parity L4 inflate.py LOC budget** — should have prevented claim 10
8. **Catalog #324 predicted-band post-training Tier-C validation** — should have prevented claims 6-7

**The discipline EXISTS; I didn't APPLY it consistently.** The structural fix is making the discipline routine.

### What needs to be added (per "Bugs must be permanently fixed AND self-protected against" non-negotiable)

A new STRICT preflight gate that REFUSES strategic-claim-relay artifacts without explicit receiver-path-feasibility evidence:

**Proposed Catalog # (next available)**: `check_rate_attack_strategic_claim_has_receiver_path_evidence`
- Scans `.omx/research/*_design_*.md` + `.omx/research/codex_routing_directive_*.md` + `.omx/research/rate_attack_*.md` for rate-attack-vector claims
- For each claim (matching patterns like "encode in X", "exploit X", "free bytes in X"), requires adjacent:
  - **WHERE evidence**: which artifact bytes / which scorer-internal state / which decoder-side info
  - **RECEIVER PATH evidence**: NO_RECEIVER_NEEDED / LEGAL_RECEIVER_IN_BUDGET (with LOC + dep count) / STRICT_SCORER_RULE_VIOLATION (with reactivation criteria) / RECLAIMABLE_VIA_DISTILLATION (with size target)
  - **CARGO-CULT classification**: HARD-EARNED-VERIFIED (with source-trace) / CARGO-CULTED-PENDING-EMPIRICAL (with test plan)
- Same-line waiver `# RATE_ATTACK_CLAIM_PATH_PENDING_OK:<rationale>` for in-progress claims being designed

This gate would have CAUGHT my F1 + F3-F6 + A1 claims at landing time (not after Codex's source-verification). The pattern matches Catalog #287 evidence-tag discipline applied to strategic-claim relay.

## SECTION 3 — OPERATOR'S "WE CAN GET THE STUFF EXTREMELY SMALL" IMPLICATIONS

If binary distillation reliably achieves <1-10 KB receivers, the following vectors are RECLAIMABLE from STRICT_SCORER_RULE_VIOLATION:

| Vector | Original status | Distillation target |
|---|---|---|
| A1 scorer-feature-space encoding | STRICT_SCORER_RULE_VIOLATION | <10 KB per A1-distillation subagent |
| F3 PoseNet vision(2048) inverter | STRICT_SCORER_RULE_VIOLATION | <5 KB (smaller manifold than full PoseNet) |
| F4 PoseNet summary(512) bottleneck inverter | STRICT_SCORER_RULE_VIOLATION | <2 KB (smallest bottleneck) |
| F5 PoseNet ResBlock output inverter | STRICT_SCORER_RULE_VIOLATION | <5 KB |
| F6 Hydra trunk-vs-head inverter | STRICT_SCORER_RULE_VIOLATION | <10 KB (trunk is larger) |
| Other "encode in scorer intermediate" vectors | STRICT_SCORER_RULE_VIOLATION | depends on intermediate layer |

The A1-distillation subagent (acb1a03a8e4234022) is currently focused on A1 specifically. Scope extension to systematically re-examine F3-F6 + others is the next logical subagent when slots free.

## SECTION 4 — RECOMMENDATIONS

### Immediate (main thread; no slot consumed)

1. ✓ This META-audit memo lands (you're reading it)
2. Write Codex routing directive for `check_rate_attack_strategic_claim_has_receiver_path_evidence` STRICT preflight gate (Section 2 above)

### Next slot to free (subagent)

3. SYSTEMATIC RECLAIMABILITY RE-EXAMINATION subagent — extends A1-distillation subagent's scope to systematically re-examine ALL STRICT_SCORER_RULE_VIOLATION vectors per the binary-distillation framework

### Continuous (apparatus)

4. Apply Catalog #229 + Catalog #303 + Catalog #6 + L4 LOC budget PROSPECTIVELY (not retroactively) to all rate-attack strategic-claim relay
5. ADVERSARIAL council pattern (Tao+Carmack+Hotz+Boyd quartet) becomes standard sister discipline for all rate-attack research subagents

## 6-HOOK WIRE-IN DECLARATION per Catalog #125

1. Sensitivity-map contribution: N/A (META-audit memo)
2. Pareto constraint: N/A
3. Bit-allocator hook: N/A
4. Cathedral autopilot dispatch hook: **ACTIVE** — autopilot ranker should consume RECLAIMABLE vs STRICT_SCORER_RULE_VIOLATION verdicts per this audit
5. Continual-learning posterior update: **ACTIVE** via council deliberation anchor when SYNTHESIS-V2 cites this audit
6. Probe-disambiguator: **ACTIVE** — this audit IS the canonical disambiguator between "declarative property" (claims) vs "physical implementation" (legal-receiver-path evidence)

## CROSS-REFERENCES

- Codex F1 finding relay (commit 35b06f9ec; this audit's source)
- ADVERSARIAL paradigm challenger (commit 4c6e46bfa; caught claims 4-6)
- PRIMARY rate-attack research (commit ~2cae89a87 estimated)
- Supplement (commit d43ecddb0; per-axis matrix)
- A1-binary-distillation subagent (acb1a03a8e4234022 in flight)
- SYNTHESIS-V2 subagent (a18c228872a761bdb in flight)
- Wiring/integration/orphan audit (commit b1aae8536; caught claim 9)
- CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable (canonical reference)
- CLAUDE.md "META-ASSUMPTION ADVERSARIAL REVIEW" Catalog #291 (canonical reference)

— Main-Claude 2026-05-18 (META-audit per operator question "have we made similar interpretation or translation or strategy errors or mistakes or errors anywhere else in the past few days like you almost did with A2 binary distillation?" + "we can get the stuff extremely small" confirmation REOPENS reclaimable vectors)
