# CLAUDE.md Amendment Proposal — Quantizr Canonical 3-Stack — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Date**: 2026-05-29T07:54Z
**Status**: DEFERRED-to-operator-decision per Slot CC STRATEGIC RESET #1 (cap=1 apparatus_maintenance-per-turn for ONE WEEK)
**Lane**: `lane_slot_ee_quantizr_catalog_325_audit_20260529`

## Source

Slot EE Quantizr canonical 3-stack audit (`.omx/research/quantizr_canonical_3_stack_audit_design_20260529.md`) empirically established that 99 of 100 substrate trainers at `experiments/train_substrate_*.py` omit at least one Quantizr-canonical primitive (EMA decay 0.997 + KL distill T=2.0 SegNet + eval_roundtrip=True per CLAUDE.md non-negotiables) despite the operator's Slot CC T3 grand-council Quantizr binding revision #5 verbatim *"Every PR-95-PARITY SUBSTRATE BUILD MUST include EMA decay 0.997 + KL distill T=2.0 + eval_roundtrip=True per CLAUDE.md non-negotiables. 0.196-0.199 cluster includes substrates that VIOLATE these silently. Catalog #325 6-step symposium MUST audit substrate trainer for all 3 canonical primitives before PROCEED."*

The existing CLAUDE.md "EMA - NON-NEGOTIABLE, HIGHEST EMPHASIS" + "eval_roundtrip - NON-NEGOTIABLE, HIGHEST EMPHASIS" + "Quantizr intelligence" sections cover each primitive individually. This amendment proposes a sister "Quantizr canonical 3-stack - NON-NEGOTIABLE, HIGHEST EMPHASIS" section codifying the COMPOSITE binding requirement so per-substrate Catalog #325 6-step symposia inherit the canonical contract at the structural source-text surface.

## Proposed CLAUDE.md amendment text

Insert AFTER existing "EMA - NON-NEGOTIABLE, HIGHEST EMPHASIS" section, BEFORE existing "MPS auth eval is NOISE" section:

```markdown
## Quantizr canonical 3-stack — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Source**: Slot CC T3 grand-council strategic-reprioritization Quantizr binding revision #5 (2026-05-29) verbatim *"Every PR-95-PARITY SUBSTRATE BUILD MUST include EMA decay 0.997 + KL distill T=2.0 + eval_roundtrip=True per CLAUDE.md non-negotiables. 0.196-0.199 cluster includes substrates that VIOLATE these silently."* + Slot EE audit (`feedback_slot_ee_quantizr_catalog_325_audit_ema_kl_eval_roundtrip_per_slot_cc_dissent_landed_20260529.md`) empirical finding 99/100 substrate trainers omit at least one primitive.

**Every substrate trainer at `experiments/train_substrate_*.py` (excluding `research_only=true` opt-outs per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable) MUST honor all 3 Quantizr-canonical primitives**:

1. **EMA decay 0.997** per `tac.training.EMA` canonical helper (Council D EMA audit 2026-04-29 + Quantizr 0.33 [contest-CUDA] anchor); apply at eval-time only with snapshot+restore per Catalog #88 + sister discipline; save EMA shadow as inference checkpoint
2. **KL distill T=2.0 SegNet** per Hinton-Vinyals-Dean 2014 canonical (Quantizr canonical SegNet distillation; PR100 hnerv_lc_v2 canonical reference); ALONGSIDE standard loss NOT as primary loss per CLAUDE.md "Critical lessons — DO NOT repeat these mistakes"
3. **eval_roundtrip=True in training inner loop** per `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training` canonical helper (PR #95 Finding A + Finding B anchors; without it proxy-auth gap is 2-11x on PoseNet per CLAUDE.md "eval_roundtrip - NON-NEGOTIABLE")

**Substrate-optimal-engineering fork acceptance**:

- Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode": a substrate trainer MAY FORK any of the 3 primitives per Catalog #290 canonical-vs-unique decision per layer
- Fork acceptance REQUIRES: (a) explicit same-line `# QUANTIZR_3_STACK_FORK_OK:<rationale>` waiver on the missing-primitive line per Catalog #287 placeholder-rationale-rejection sister discipline; (b) Catalog #325 6-step symposium content surfacing the per-primitive canonical-vs-unique decision per layer per Catalog #290; (c) per-substrate cargo-cult audit per Catalog #303 classifying the FORK as HARD-EARNED-from-substrate-empirical OR CARGO-CULTED-inherited-from-non-Quantizr-substrate

**Concrete enforcement**:

- Strict preflight gate candidate `check_substrate_trainer_honors_quantizr_canonical_3_stack` (NEW Catalog # claim DEFERRED per Slot CC STRATEGIC RESET #1 cap=1 apparatus_maintenance-per-turn for ONE WEEK; the gate would refuse substrate trainers without 3-stack adherence + waiver discipline)
- Canonical equation `quantizr_canonical_ema_kl_eval_roundtrip_stack_savings_v1` registered per Catalog #344 (predicted -0.005 to -0.020 ΔS per missing primitive; 0 empirical anchors at registration; first anchor pending Class A/D Catalog #325 6-step symposium)
- Canonical anti-pattern `substrate_trainer_missing_quantizr_canonical_3_stack_ema_kl_eval_roundtrip_v1` registered per Catalog #344 sister (severity HIGH; paradigm DISCIPLINE_RIGOR_LOSS)

**Cross-references**:

- CLAUDE.md "EMA - NON-NEGOTIABLE, HIGHEST EMPHASIS" (Quantizr decay 0.997 source)
- CLAUDE.md "eval_roundtrip - NON-NEGOTIABLE, HIGHEST EMPHASIS" (PR #95 inner-loop source)
- CLAUDE.md "Quantizr intelligence - verified competitive data (2026-04-21)" (KL T=2.0 SegNet distill source)
- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L7 (bind-all-ingredients per substrate)
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" (per-substrate canonical-vs-fork decision per Catalog #290)
- CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" (Catalog #325 6-step symposium)
- Council D EMA audit `.omx/research/council_ema_audit_20260429.md`
- Slot EE audit `.omx/research/quantizr_canonical_3_stack_audit_design_20260529.md`
- Slot CC T3 grand-council source directive
```

## Rationale for DEFERRAL

Per Slot CC STRATEGIC RESET #1: cap=1 apparatus_maintenance-per-turn for ONE WEEK; ZERO new Catalog # gates unless EXTINCT canonical anti-pattern AND ENABLE frontier_breaking; operator-frontier-override required for any new apparatus_maintenance landing.

This CLAUDE.md amendment is `apparatus_maintenance` mission-contribution per Catalog #300 (codifies existing canonical disciplines into a sister composite section). Per the strategic reset directive, the amendment SHOULD be considered when:

1. Class A wavelet + Class D Wyner-Ziv Catalog #325 6-step symposia land
2. First empirical anchor against canonical equation `quantizr_canonical_ema_kl_eval_roundtrip_stack_savings_v1` lands (Class A or Class D paired CPU+CUDA RATIFICATION)
3. Operator approves the amendment per Catalog #299 quota brake operator-explicit waiver

## Operator-routable next-steps

When the operator chooses to ADOPT the amendment:

1. Insert the proposed text AFTER existing CLAUDE.md "EMA - NON-NEGOTIABLE" section
2. Claim new Catalog # via `tools/claim_catalog_number.py claim --commit-via-serializer --reason "Slot EE Quantizr canonical 3-stack STRICT preflight gate per CLAUDE.md amendment"`
3. Implement strict preflight gate `check_substrate_trainer_honors_quantizr_canonical_3_stack` in `src/tac/preflight.py` per canonical 2-landing pattern (canonical helper + STRICT gate)
4. Backfill the 99/100 substrate trainers via per-substrate Catalog #325 6-step symposium ADOPT vs FORK decision per Catalog #290

## Lane

`lane_slot_ee_quantizr_catalog_325_audit_20260529` L1 (CLAUDE.md amendment proposal — DEFERRED-to-operator-decision)
