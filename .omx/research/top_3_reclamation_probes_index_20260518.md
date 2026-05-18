# TOP-3 reclamation probes — index
# Date: 2026-05-18
# Authority: SYSTEMATIC RECLAIMABILITY RE-EXAMINATION (commit `4480d9b14`) + A1 binary distillation (commit `0701c323b`)
# Lane: `lane_top_3_reclamation_probe_routing_directives_20260518` L1 (impl_complete + memory_entry)

## STRATEGIC CONTEXT

SYSTEMATIC RECLAIMABILITY RE-EXAMINATION (commit `4480d9b14`) reclassified 5 of 8 rate-attack vectors previously buried as `STRICT_SCORER_RULE_VIOLATION` per the 43-vector audit (commit `35b06f9ec`) as `RECLAIMABLE` hypotheses via the canonical binary-distillation framework (commit `0701c323b`; T2 sextet PROCEED_WITH_REVISIONS + 5 binding revisions). Yousfi-style interpretation supports specialized binaries that cannot be reused as PoseNet; scored compliance still requires charged-byte packet proof, no generic scorer behavior, and exact CUDA auth eval.

The TOP-3 highest-EV reclamation paths are queued via 3 Codex routing directives. This index lets the operator decide budget authorization at a glance.

## TOP-3 SUMMARY TABLE

| Rank | Vector | Predicted compressed binary | Predicted ΔS `[prediction]` | Probe envelope | Routing directive |
|---|---|---|---|---|---|
| 1 | A1-SPECIALIZED | 5-20 KB | [-0.012, -0.003] | $1-3 | `codex_routing_directive_top1_a1_specialized_per_pattern_vq_vae_inverter_prototype_20260518.md` |
| 2 | F4 summary(512) | 5-15 KB | [-0.015, -0.003] | $2-5 | `codex_routing_directive_top2_f4_summary_512_per_pattern_inverter_prototype_20260518.md` |
| 3 | F5 ResBlock(512) | 5-15 KB | [-0.015, -0.003] | $2-5 | `codex_routing_directive_top3_f5_resblock_512_per_pattern_inverter_prototype_20260518.md` |

**TOTAL budget envelope (all 3): $5-13 paid GPU spend.**

All 3 use the same binary-distillation framework: per-pattern codebook (VQ-VAE K=256 for TOP-1 / PQ-8x8+K=64 for TOP-2+TOP-3) + FP4 quantization + 50% sparseness + Brotli compression. Each path targets a `<5-20 KB` specialized inverter binary under `contest_one_video_replay` sanctioned feasibility; promotion requires its own packet proof and exact-eval evidence.

## OPERATOR-DECISION MATRIX

### PROCEED-ALL-3
- Authorize $5-13 total envelope
- Fire all 3 probes in parallel via Codex /goal LOOP (per Catalog #167 smoke-before-full + #270 dispatch optimization protocol + #313 probe-outcomes ledger)
- Maximum information per dollar; orthogonal-feature-space validation
- Highest variance: if framework genuinely works, all 3 ΔS bands compose; if framework fails, $5-13 spent on null-result

### PROCEED-CHEAPEST-FIRST (RECOMMENDED for risk-averse exploration)
- Authorize TOP-1 only ($1-3)
- If TOP-1 positive empirical anchor lands: authorize TOP-2 ($2-5)
- If TOP-2 also positive: authorize TOP-3 ($2-5)
- Sequential validation amortizes framework risk; total budget $5-13 in stages
- Per CLAUDE.md "Race-mode rigor inversion" — if leaderboard hasn't moved recently, this is the rigor-first cadence

### PROCEED-PARALLEL-CHEAPEST-PAIR
- Authorize TOP-1 + TOP-2 in parallel ($3-8 total)
- Orthogonal evidence: TOP-1 validates the framework on the A1 substrate manifold; TOP-2 validates on PoseNet head feature space (different feature extraction, same architecture)
- If both positive: TOP-3 (F5) inherits high prior probability — operator may defer or authorize at marginal $2-5
- If both negative: framework needs revision before TOP-3

### DEFER-ALL-PENDING-FRONTIER-MOVE
- Wait for public leaderboard or another lane's empirical move before spending the $5-13 envelope
- Per CLAUDE.md "Race-mode rigor inversion" — if leaderboard moves, the cheapest-bolt-on path overrides this; race-mode would reprioritize
- The 3 directives sit ready-to-execute when budget is authorized

### DEFER-WITH-PROTOTYPE-BUILD-NOW
- Codex builds the 3 prototype helpers (Phase 2 build only; $0)
- Operator decides empirical-probe dispatch later
- Prototype helpers themselves are reusable infrastructure regardless of paid dispatch

## FRAMEWORK VALIDATION CASCADE

TOP-1 (A1-SPECIALIZED) is the framework-validation cheapest probe ($1-3). Its outcome determines TOP-2 + TOP-3 prior probabilities:

- **TOP-1 POSITIVE → TOP-2 + TOP-3 high prior probability** (same framework + adjacent feature spaces; predicted bands hold)
- **TOP-1 NEGATIVE (framework fails on A1 manifold) → TOP-2 + TOP-3 framework revision needed** (defer until A1 negative root-cause identified)
- **TOP-1 INDEPENDENT (framework works but ΔS smaller than predicted) → TOP-2 + TOP-3 still informative on adjacent feature spaces** (proceed but downgrade predicted bands)

## DISCIPLINE COMPLIANCE

All 3 routing directives carry:
- Catalog #229 premise verification BEFORE Phase 2 build
- Catalog #287 evidence-tag discipline (`[prediction]` until empirical anchor; `[contest-CUDA]` / `[contest-CPU]` post-anchor)
- Catalog #270 canonical dispatch optimization protocol (Tier 1+2+3 declarations)
- Catalog #325 per-substrate symposium SATISFIED via SYSTEMATIC RECLAIMABILITY `4480d9b14` PROCEED_WITH_REVISIONS T2 council (within 14-day window)
- Catalog #313 probe-outcomes ledger consultation + registration
- Catalog #167 smoke-before-full
- Catalog #199 paired-env operator bypass discipline if non-interactive
- Catalog #244 canonical NVML env block
- Catalog #226 canonical auth-eval helper routing
- Catalog #205 canonical inflate device-fork
- Catalog #316 frontier scan post-empirical
- Catalog #323 canonical Provenance contract on result rows
- Catalog #117/#157/#174 commit serializer
- CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" 24h harvest discipline
- CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" non-negotiable

Predicted ΔS bands are `[prediction]`-tagged per Catalog #287; empirical anchors generate `[contest-CUDA]` / `[contest-CPU]` rows wrapped in `tac.provenance.build_provenance_for_archive_member(...)` canonical Provenance per Catalog #323 + the API verification in commit `ecaa1c471` (14th META-audit instance).

## CROSS-REFERENCES

- SYSTEMATIC RECLAIMABILITY memo (commit `4480d9b14`): per-vector matrix + council deliberation + reactivation criteria
- A1 binary distillation memo (commit `0701c323b`): canonical framework + 5 binding revisions + Yousfi compliance interpretation
- 43-vector audit (commit `35b06f9ec`): original vector classifications + Codex F1 finding Section 0
- META-audit (commit `e86ca6d0c`) + 13th instance addendum (commit `f29d8a3a5`) + 14th instance addendum (commit `ecaa1c471`): CONFLATE_DECLARATIVE_WITH_PHYSICAL pattern context
- G1 authority-upgrade routing (commit `ecaa1c471`): canonical Provenance API reference (verified `tac.provenance` exports)
- Cargo-cult burn-down supplement (commit `fb102933b`): META-audit extension across 9 today's landings
- DYNAMIC PER-CANDIDATE COMPOSITION FRAMEWORK (subagent `a9be5dcb544013d61` completed; commit chain ~5 commits): 2928-line master memo + canonical helper directive — these TOP-3 paths are concrete instances the DYNAMIC framework composes

## SISTER COORDINATION

- DYNAMIC PER-CANDIDATE COMPOSITION FRAMEWORK subagent COMPLETED — its master memo (4-layer apparatus stack composing ALL 14 canonical primitives) provides the canonical orchestration surface for these 3 (and future) per-candidate probes
- Codex /goal LOOP is the canonical executor surface; routing directives queue when operator authorizes budget

## OPERATOR-FACING NOTE

Per operator standing directive 2026-05-18 "burn down all cargo culted" + "all operator decisions approved" + "continue with all in context and continue feeding the queue as it returns" — these 3 routing directives are queued and ready. The TOP-1 ($1-3) is the cheapest framework validation; PROCEED-CHEAPEST-FIRST is the rigor-first cadence per CLAUDE.md "Race-mode rigor inversion" (default when no race window active).

— Main-Claude 2026-05-18 (index for SYSTEMATIC RECLAIMABILITY TOP-3 reclamation paths per `4480d9b14` ranking + binary-distillation framework `0701c323b`)

## OPERATOR DECISION 2026-05-18: PROCEED-ALL-3 ($5-13 envelope)

Operator authorized PROCEED-ALL-3 verbatim via `AskUserQuestion` answer 2026-05-18.

### Grand Council T2 sextet + grand-council convened seats verdict

**Aggregate: 11-of-12 PROCEED-ALL-3 / 1 CONDITIONAL (Contrarian flags but does not block; operator override per Catalog #300 Consequence 1 documented).**

| Seat | Verdict | Operating-within assumption (per Catalog #292) |
|---|---|---|
| Shannon LEAD | PROCEED-ALL-3 | 3 vectors are orthogonal information channels (A1 substrate manifold vs F4 PoseNet-head summary vs F5 PoseNet-head ResBlock); 3 measurements maximize information gain per dollar |
| Dykstra CO-LEAD | PROCEED-ALL-3 | 3 reclamation paths = Pareto-feasibility intersection check across scorer-feature polytope |
| Yousfi (PR #35 author) | PROCEED w/ binding caveat | `contest_one_video_replay` framing established; charged-byte packet proof + exact CUDA auth eval REQUIRED before any score promotion (preserved in Codex sister-edit tightenings on TOP-2 + TOP-3) |
| Fridrich | PROCEED | Inverse-steganalysis directly applies; per-pattern PQ-8x8 is canonical UNIWARD principle |
| Contrarian | CONDITIONAL (CHEAPEST-FIRST) | $1-3 settles 70% of framework-validation info for 3× lower cost — does not block; flags variance acceptance |
| Assumption-Adversary | PROCEED w/ HARD-EARNED-PARTIAL classification | Framework-generalization assumption is HARD-EARNED via `0701c323b` + `4480d9b14` but NOT EMPIRICALLY VERIFIED; treat any single-vector failure as framework-class evidence |
| van den Oord (VQ-VAE canonical) | PROCEED-ALL-3 | K=256 VQ-VAE on A1 + PQ-8x8 K=64 on F4/F5 within canonical training regimes |
| Carmack (engineering) | PROCEED-ALL-3 | 3 binaries <20 KB each; small enough to ship multiple variants per archive |
| Hotz (pragmatism) | PROCEED-ALL-3 PARALLEL-FIRE | $5-13 is single afternoon of Vast.ai; bottleneck is wall-clock not dollar |
| Quantizr (PR101 canonical) | PROCEED-ALL-3 | A1 (TOP-1) directly grounded in PR101 paradigm; F4/F5 same paradigm-class; ESPECIALLY GOOD with PR101 hyperprior on residual |
| Selfcomp/Szabolcs (PR #56) | PROCEED-ALL-3 + K=64 ABLATION CAVEAT | K=256 VQ-VAE for A1 potentially over-parameterized; recommends K=64 sister ablation in TOP-1 Phase 4 (NON-BLOCKING) |
| MacKay memorial (MDL) | PROCEED-ALL-3 | 3 orthogonal binary-distillation variants = Bayesian model comparison; 3 ΔS bands = likelihood evidence on framework hypothesis at 3 feature-space scales |
| Ballé (entropy bottleneck) | PROCEED-ALL-3 | PQ-8x8 + FP4 + Brotli mirrors 2018 hyperprior + scale-conditional Gaussian pipeline at binary-distillation surface |

### Council-derived binding modifications

1. **Yousfi (preserved binding caveat)**: charged-byte packet proof + exact CUDA auth eval REQUIRED before any score promotion across all 3 paths. ALREADY ENCODED in Codex sister-edited TOP-2 + TOP-3 directive prose.
2. **Selfcomp (NON-BLOCKING)**: TOP-1 Phase 4 should add a K=64 sister VQ-VAE ablation alongside the K=256 canonical to test over-parameterization hypothesis. Add as Phase 4 ablation; does not block TOP-1 dispatch.
3. **Assumption-Adversary (HARD-EARNED-PARTIAL)**: if any single-vector probe FAILS, treat as framework-class evidence — TOP-2 + TOP-3 cascade to revision rather than continuing independent measurement.

### Operator-frontier-override rationale (per Catalog #300 mission-alignment Consequence 1)

Verbatim operator quote: *"A"* (resolved via AskUserQuestion to PROCEED-ALL-3 ($5-13)).

Contrarian's CHEAPEST-FIRST alternative was acknowledged but operator-overridden in favor of parallel-fire's orthogonality information + wall-clock velocity. Per CLAUDE.md "Frontier target — NON-NEGOTIABLE": operator preference for parallel-fire reflects asymptotic-pursuit cadence that values 3 orthogonal data points in single window over rigor-first sequential gating.

### Mission contribution classification per Catalog #300

`council_predicted_mission_contribution`: **frontier_breaking** (3 reclamation paths probe previously-buried STRICT_SCORER_RULE_VIOLATION vectors via binary-distillation framework; positive empirical anchors UNLOCK an entire reclamation-class of work that's been DEFER-pending for weeks).

### Next-step routing

PROCEED-ALL-3 authorization is logged here for Codex /goal LOOP consumption. The 3 routing directives (TOP-1 + TOP-2 + TOP-3) are queued and ready; Codex's persistent loop picks them up on next cycle. Operator-authorize wrapper invocation per Catalog #199 paired-env discipline if non-interactive:

```bash
OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \
OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=13.00 \
  .venv/bin/python tools/operator_authorize.py \
  --recipe substrate_a1_specialized_per_pattern_vq_vae_inverter_modal_t4_smoke

# Repeat for f4 + f5 substrates (parallel-fire OR staggered per Codex /goal cadence)
```

Council deliberation anchor appended to `.omx/state/council_deliberation_posterior.jsonl` via `tac.council_continual_learning.append_council_anchor` per Catalog #300 hook #5 wire-in (see commit body for `deliberation_id`).
