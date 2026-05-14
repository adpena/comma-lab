# All design decisions through Grand Council — directive 2026-05-14

**Operator directive verbatim 2026-05-14**: *"ask the grand council to weigh in on all design decisions"*

**Tag**: `research_only=true`; canonical persistence handoff per CLAUDE.md "Subagent coherence-by-default" mandatory pre-read of `.omx/research/*_directive_*` files dated within last 24 hours. **EVERY future subagent MUST honor.**

## Codification

This directive REINFORCES + EXPANDS CLAUDE.md "Design decisions — non-negotiable":

> *"Always consult the skunkworks council (Yousfi + Fridrich + Hotz + Quantizr + Contrarian) before implementing any change that affects training behavior, loss functions, architecture configuration, interpolation methods, boundary values, optimization strategy, or any other design tradeoff."*

The operator's reinforcement adds:

1. **EVERY design decision** must go through the grand council (inner ten + Time-Traveler peer + grand bench consulted on demand). NOT just training-behavior changes — ALL tradeoff decisions.
2. **No unilateral architectural choices** by subagent landings — even high-priority ones. Op-routable surfaces are the canonical signaling channel; council verdict is the canonical decision channel.
3. **Active design decisions** must be enumerated + scheduled into council reviews (the omnibus review pattern), not handled ad-hoc as they arise.
4. **The council's job is binding** — per CLAUDE.md "Design decisions — non-negotiable" + "Subagent coherence-by-default", the council ledger IS the design decision. No further operator confirmation required for decisions reaching consensus (per operator standing pre-approval).

## What qualifies as a "design decision"

Per CLAUDE.md "Design decisions — non-negotiable" + this directive:

- **Architectural choices**: bicubic vs bilinear / GRU vs LSTM vs Transformer / depth / width / activation
- **Loss function choices**: rate-distortion vs cooperative-receiver / KL weight / Lagrangian λ
- **Constraint boundaries**: epoch count / batch size / max-pairs / smoke band / full band
- **Optimization strategy**: AdamW vs Muon / scheduler / EMA decay
- **Compositional ordering**: Z3 → Z4 → Z5 vs partial parallel
- **Substrate routing**: C1 wholesale into Z5 vs independent dispatches
- **Strict-flip atomicity**: when to flip a STRICT preflight gate from warn-only to STRICT
- **Default-value tradeoffs**: F3 GTScorerCache default True vs False
- **Architectural refactors**: if/elif → registry pattern / canonical surface promotion
- **Substrate-class promotion criteria**: what evidence promotes a lane from L1 to L2
- **Provider routing**: Modal vs Vast.ai vs Lightning per dispatch class
- **OSS release timing**: announcement coordination with contest-CUDA events
- **Multi-stage curriculum design**: PR95 Phase 2-4 / Z3 v1 byte-identical strategy / Phase 3 multi-stage

## What does NOT qualify (no council required)

- **Clear bugs**: crashes, wrong formulas, missing imports, dead code — fix immediately
- **Premise verification**: subagent verifying its own prompt's assertions (per the prompt-premise pattern)
- **Routine operational tasks**: lane registration / checkpoint writes / canonical commits
- **Mechanical reproducer execution**: running an existing probe / harvest / ablation tool
- **Sister-subagent ownership respect**: not editing files owned by other subagents

If ambiguous → it IS a design decision; ask the council.

## How to apply (operationally)

For every future subagent landing that surfaces operator-routable decisions:

1. **Classify each op-routable**: clear-bug-fix? premise-verification? mechanical-routine? OR design-decision?
2. **Design decisions queue into the council omnibus** — accumulate in `.omx/state/pending_council_design_decisions.jsonl` (one row per decision)
3. **Periodic council omnibus reviews** sweep the queue (~120-180 min, $0 deliberation, 11-voice rotation, binding verdicts per decision)
4. **Single design decision can also trigger an ad-hoc council** if it's blocking high-EV dispatch (e.g., C1 contest-scale dispatch route)

## The 12+ active design decisions enumerated at this directive's commit

Per session accumulation (sourced from subagent op-routables):

1. **10-grammar registry refactor** in `tac.analysis.hnerv_packet_sections` (IBPS1-PARSER-WAVE-P0 op-routable #5)
2. **F3 GTScorerCache canonical default flip** to True (F3 GTScorerCache landing op-routable #5)
3. **Z3 v1 byte-identical strategy** + v2 prioritization (Z3-FULL-MAIN op-routable #5)
4. **Z3 → Z4 → Z5 staircase ordering** (TIME-TRAVELER landing op-routable #2)
5. **C1 ↔ Z5 wholesale routing** decision (C1-PROBE-V2 op-routable #4; partially covered by C1-COUNCIL-RECONVENE in flight)
6. **STRICT gate strict-flip timing** for Catalog #227/#228/#229/#230/#231 (CATALOG-227-231-WAVE in flight)
7. **MDL Tier C extension** to PR106_latent_sidecar + other substrates (MDL-ABLATION-TIER-C op-routable #3)
8. **Substrate-class promotion criteria** (what evidence promotes lane L1 → L2) — C6 5ep architectural-ACROSS-CLASS finding raises this
9. **Provider routing**: Modal vs Vast.ai vs Lightning for upcoming dispatcher wave per cost-band posterior
10. **Phase B-2 / Wave 2 sweep design** (6-7 substrates @ $6-21 cap $15) — ORCHESTRATOR-5 pending
11. **DARTS-SuperNet C7 strategic dispatch** ($100-300 Tier 3)
12. **OSS announcement timing** (alignment with contest-CUDA event?)
13. **F3 backport sister-trainer wave**: vq_vae flag-declare + PDP substrate-side wire-in (F3-BACKPORT-WAVE-V2 op-routable #3+#4)
14. **MDL ablation Tier C → Tier D / Tier E?** (future extension pattern)
15. **Sister-substrate parser P2/P3 batch sequencing** (IBPS1-PARSER-WAVE-P0 op-routable #2+#3)
16. **Push v0.2.0-rc1 vs hold for v0.2.0 stable** (OSS-RELEASE op-routable #5)

The C1-COUNCIL-RECONVENE subagent (`a7f33c89` — wait, that's OSS-PUBLIC-PUSH; correct: `aadaca94`) currently handles #5 (C1↔Z5 routing). The omnibus sweep handles the remaining 15.

## Cross-refs

- CLAUDE.md "Design decisions — non-negotiable" (the canonical doctrine this directive reinforces)
- CLAUDE.md "Council non-conservatism enforcement" (the council operates per inner-ten + Time-Traveler peer non-conservative charter)
- CLAUDE.md "Subagent coherence-by-default" (this directive's structural propagation mechanism)
- CLAUDE.md "KILL/FALSIFIED memo structural requirements" (council ledger format)
- `.omx/research/holistic_engineering_picture_seven_factor_directive_20260514.md` (the 7-factor frame the council applies to each decision)
- `.omx/research/grand_council_tiered_parallel_plan_full_authority_20260514.md` (prior council standing authority + tiered plan)
- `.omx/research/grand_council_c1_world_model_adversarial_review_20260514.md` (prior C1 council)

## Effective immediately

In-flight subagents (AUTOPILOT-TIER-C-INTEGRATION / CATALOG-227-231-WAVE / SISTER-PARSER-P1-WAVE / C1-COUNCIL-RECONVENE / OSS-PUBLIC-PUSH) MUST queue any new design-decision op-routables into `.omx/state/pending_council_design_decisions.jsonl` rather than declaring unilateral verdicts. Future subagents inherit via mandatory `.omx/research/*_directive_*` last-24-hours pre-read.

Tagged `research_only=true`. NO score claims. NO GPU spend.
