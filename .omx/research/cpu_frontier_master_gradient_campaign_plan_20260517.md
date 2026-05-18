# CPU Frontier Campaign Plan — fec6 baseline 0.19205 → predicted 0.16-0.17 [contest-CPU]

**Status:** DRAFT — operator review pending
**Source:** ratified by T4 Grand Reunion Symposium 2026-05-17 (`.omx/research/grand_reunion_t4_symposium_orthogonal_optimization_master_gradient_20260517.md`); operator-frontier-override per CLAUDE.md "Mission alignment" Consequence 1
**Baseline:** `0.19205 [contest-CPU GHA Linux x86_64]` (archive `6bae0201`, lane `pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean`)
**Target:** `0.16-0.17 [contest-CPU GHA Linux x86_64]` after 5 PR submissions over 5-10 days
**Budget envelope:** $120-170 GPU + $0 build time
**Mission contribution:** frontier_breaking
**Lane index:** `lane_cpu_frontier_master_gradient_campaign_20260517` (parent; per-step lanes registered below)

## §0 — Executive summary

The campaign operationalizes the symposium's 7 op-routables into 4 sequential dispatch waves. Each wave produces a paired-axis submission (per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"), a master-gradient anchor (per the new Phase-7 lens), and a promotion verdict (per Catalog #233 4-gate canonical). The continual-learning loop closes after every wave: anchors feed the autopilot's Rashomon ensemble; the ensemble's consensus reranks the next wave's candidates; the autopilot's master_gradient_lens (op-routable #3) consumes the updated posterior.

**Predicted score trajectory** — ALL rows below row 0 carry `[predicted, council-consensus]` axis tag per CLAUDE.md FORBIDDEN_PATTERN "docstring-overstatement-trap" + Round-1 review C-3 finding. **These are NOT empirical scores and NOT promotable as such.** Only row 0 is `[contest-CPU empirical]`.

| step | submission | predicted [contest-CPU] | axis tag | confidence | cost (revised per M-5 + M-6) |
|---|---|---|---|---|---|
| 0 (baseline) | fec6 PR (this PR) | **0.19205** | **`[contest-CPU empirical]`** | — | already paid |
| 1 | + master-gradient + writeup fix | 0.19205 (no score change; enabler) | `[no score change]` | high | $0.50-5 (per §3.2 revised methodology) |
| 2 | + SABOR SegNet-only sister | 0.182-0.187 | `[predicted, council-consensus]` | medium | $50-80 |
| 3 | + L5 Wyner-Ziv pose deltas | 0.174-0.182 | `[predicted, council-consensus]` | medium-high | $40-60 |
| 4 | + U-DIE-KL substrate-wide loss | 0.165-0.175 | `[predicted, council-consensus]` | medium | $60-120 |
| 5 (stretch) | + cross-paradigm format0d-CPU-sister | 0.155-0.170 | `[predicted, council-consensus]` | low-medium | $80-150 |
| **+ master-gradient re-measurement per wave** | (3 waves × $5-15) | — | — | — | $15-45 |
| **Total realistic envelope** | | | | | **$245-460** (revised from $135-185) |

**Per CLAUDE.md "Apples-to-apples evidence discipline":** none of rows 1-5 may be promoted, ranked, or cited as authoritative score until the corresponding wave's paired-axis empirical anchor lands and supersedes the predicted row.

## §1 — Wave 1 (FOUNDATION; days 1-2; $5-15)

**Purpose:** establish the master-gradient anchor + autopilot lens + writeup correction. No score change. Enables every subsequent wave.

### §1.1 op-routable #1 — Materialize per-byte master gradient

**Lane:** `lane_master_gradient_materialization_fec6_20260518`
**Pre-registration:** `python tools/lane_maturity.py add-lane lane_master_gradient_materialization_fec6_20260518 --name "Master gradient materialization on fec6 archive" --phase 1`
**Pre-launch gates:**
1. `tools/canonical_dispatch_optimization_protocol.py --trainer experiments/build_master_gradient_finite_difference.py --recipe .omx/operator_authorize_recipes/master_gradient_fec6_modal_cpu_dispatch.yaml --json` returns `overall_pass=true`
2. `tools/local_pre_deploy_check.py --strict` exits 0
3. `tools/check_predecessor_probe_outcome.py --substrate fec6 --json` returns no blocking predecessor (Catalog #313 sister)
4. `tools/run_codex_review_for_dispatch.py` returns verdict `approve` or `advisory` (Catalog #271)

**Dispatch command (canonical operator-authorize entry):**

```bash
.venv/bin/python tools/operator_authorize.py \
  --recipe .omx/operator_authorize_recipes/master_gradient_fec6_modal_cpu_dispatch.yaml \
  --estimated-cost-usd 12 \
  --session-budget-usd 200 \
  --confirmed-via-session-directive
```

**Build prerequisites (TODO before dispatch):**
- `experiments/build_master_gradient_finite_difference.py` — new tool; takes archive sha + axis; emits `(N_bytes, 3)` float32 npy by parallel finite-difference probing
- `.omx/operator_authorize_recipes/master_gradient_fec6_modal_cpu_dispatch.yaml` — declares `min_vram_gb: 0` (CPU only), `target_modes: [research_substrate, contest_one_video_replay]`, `canary_status: independent_substrate`
- `tac.master_gradient_ledger` — canonical fcntl-locked JSONL helper at `.omx/state/master_gradient_anchors.jsonl` (mirror of `tac.deploy.modal.call_id_ledger` per Catalog #245 4-layer pattern; new Catalog # via `tools/claim_catalog_number.py claim --commit-via-serializer`)

**Harvest:**
- `tac.deploy.modal.call_id_ledger` records dispatch + outcome
- `tac.master_gradient_ledger.append_anchor(...)` records the `MasterGradient` row
- Output artifact: `.omx/state/master_gradient_fec6_6bae0201_20260518.npy` shape `(178517, 3)` dtype float32

**Promotion criteria (per Catalog #233 4-gate):**
- Smoke green: rc=0 + master_gradient.npy materialized + sha256 matches expected layout
- Tier C MDL density: N/A (this is an artifact, not a substrate)
- Auth-eval anchor: N/A (rate-limited by finite-difference probe completion)
- Custody validated: archive_sha256 matches, axis tagged `[contest-CPU]`, hardware substrate `linux_x86_64_modal_cpu`

### §1.2 op-routable #3 — Phase-7 master_gradient lens in tac.autopilot_rudin_daubechies

**Lane:** `lane_master_gradient_lens_phase_7_20260518`
**Cost:** $0 (build only; no dispatch)
**Files to create:**
- `src/tac/autopilot_rudin_daubechies/master_gradient_lens.py` — Phase-7 lens module mirroring `phase_1_slim_risk_scorer.py` skeleton; consumes `MasterGradient` objects from `tac.master_gradient_ledger`; emits per-candidate `predicted_delta_s` projections
- `src/tac/autopilot_rudin_daubechies/tests/test_phase_7_master_gradient_lens.py` — ~20 tests covering candidate ranking, Rashomon-disagreement-queue surfacing, axis-localness, operating-point staleness handling
- `src/tac/preflight.py::check_master_gradient_lens_canonical_use` — STRICT preflight (new Catalog # via canonical claim) refusing direct construction outside canonical helpers
- Update `tools/cathedral_autopilot_autonomous_loop.py::rerank_candidates` to consume the lens (single line: `lens_registry.add(MasterGradientLens.from_canonical_ledger())`)

**Promotion criteria:** all gates per Catalog #233 N/A (this is infrastructure); STRICT preflight + dedicated tests green is the structural acceptance.

### §1.3 op-routable #5 — Quantizr 5-stage staircase canonical helper

**Lane:** `lane_quantizr_5_stage_staircase_canonical_20260518`
**Cost:** $0 (build only)
**Files to create:**
- `src/tac/training/quantizr_5_stage_staircase.py` — `QuantizrFiveStageStaircase` dataclass + helper functions; the 5 stages: (1) anchor (pixel-loss only, EMA active, BN trainable), (2) finetune (+ SegNet KL distill), (3) joint (+ PoseNet loss), (4) QAT (BN frozen, FP4 fakequant), (5) final (all but pose-axis frozen)
- `src/tac/training/tests/test_quantizr_5_stage_staircase.py` — ~15 tests covering stage transitions, EMA snapshot/restore, BN freeze invariants, transition criteria
- Hotz's revision honored: this is the Quantizr-specific staircase only; generalization to other substrates deferred until a second substrate demonstrably needs a different schedule

**Promotion criteria:** STRICT preflight + dedicated tests + adoption by one substrate trainer (default candidate: `nscs01_nullspace_split_renderer`).

### §1.4 op-routable #7 — Writeup math correction

**Lane:** `lane_pr_writeup_math_correction_20260517`
**Cost:** $0
**Edit `docs/pr_writeups/cpu_frontier_fec6_20260517.md`:**
- §1 table row 2: change marginal column from `5/√(10·d_pose) ≈ 922` to `5/√(10·d_pose) ≈ 292`
- §1 paragraph after table: change "9.22× the SegNet marginal and 1.4 × 10⁹×" to "2.92× the SegNet marginal and 4.4 × 10⁸×"
- §1 next paragraph: recompute the per-1000-bytes ΔS example (`9.2e-4` becomes ~`2.9e-4`)
- Add a parenthetical citation: `(symposium memo §1 op-routable #7)`

**Promotion criteria:** edit lands + writeup self-consistent.

### §1.5 Wave 1 gates (all must pass before Wave 2)

- master_gradient.npy materialized + master_gradient_ledger anchor written
- master_gradient lens module landed + STRICT preflight green + dedicated tests pass
- quantizr_5_stage_staircase helper landed + dedicated tests pass
- writeup correction landed
- Catalog #316 frontier scan still shows 0.19205 as canonical [contest-CPU] best (no regression)
- 1 council deliberation entry persisted via `tac.council_continual_learning.append_council_anchor` for the symposium (CLAUDE.md "Council hierarchy" wire-in rule)

## §1.6 — Catalog #315 procedural blocker resolution (Round-1 review C-4)

The T4 symposium council_verdict is `PROCEED_WITH_REVISIONS`. Per Catalog #315 (`check_substrate_at_optimal_form_before_paid_dispatch`), substrate L1+ lanes whose latest council deliberation is `PROCEED_WITH_REVISIONS` with no chronologically-later PROCEED-unconditional anchor are STRICTLY refused for paid dispatch UNLESS one of 5 opt-outs applies. The symposium's operator-frontier-override per Catalog #300 mission-alignment binding directive Consequence 1 is documented in the memo's `council_override_invoked: true` frontmatter — but this PROMOTES the revisions to advisory at the symposium deliberation surface, NOT at the per-substrate dispatch surface. Catalog #315 fires per substrate.

**Per-substrate opt-out plan for each Wave 2/3/4 dispatch:**

| wave | substrate-lane | Catalog #315 opt-out | rationale |
|---|---|---|---|
| 2.1 | `lane_pr101_fec6_plus_sabor_disjoint_20260518` | (a) sister T2 council PROCEED-unconditional | new substrate-lane; spawn focused T2 deliberation on SABOR-boundary-disjoint design before dispatch |
| 2.2 | `lane_l5_wyner_ziv_fec6_20260518` | (a) sister T2 council PROCEED-unconditional | L5 already has prior PROCEED-unconditional sextet anchor per `feedback_l5_staircase_v2_and_adversarial_apparatus_structural_fixes_landed_20260515`; reaffirm at T2 for the fec6-specific bolt-on |
| 3 | `lane_u_die_kl_fec6_adoption_20260518` | (a) sister T2 council PROCEED-unconditional | U-DIE-KL adoption is operating-point-shift; T2 must adjudicate the new loss-function design |
| 4 (stretch) | `lane_a1_x_format0d_cpu_axis_sister_20260520` | (b) `research_only=true` initially, then promote via T2 council | format0d-CPU-sister is research-grade until first paired-axis empirical anchor; ratchet to dispatch-eligible after Wave 2 anchors validate the orthogonality assumption |

**Per-wave T2 council cadence budget impact (Catalog #300 mission-alignment):** Wave 2 + 3 = 3 T2 councils per CLAUDE.md "Council hierarchy: 4-tier protocol" T2 cadence budget ≤3/day, ≤90/30d. We are within budget; no over-cadence alert fires. Wave 4 T2 deferred until anchor-validated.

**Sister artifact:** each T2 council deliberation lands a v2-frontmatter memo per Catalog #300 + emits a continual-learning anchor via `tac.council_continual_learning.append_council_anchor`. The anchor's `deferred_substrate_id` field MUST match the lane registry's `substrate_alias` for that lane so Catalog #315's family-join lookup succeeds (per `feedback_meta_framing_correction_optimal_form_before_paid_dispatch_landed_20260517`).

**Alternative path if T2 council cadence is exceeded OR if operator declines per-substrate deliberation:** tag affected lane(s) `lane_class=substrate_engineering` OR add explicit `# OPTIMAL_FORM_DISPATCH_OK:<rationale>` waiver in lane registry evidence. Both are Catalog #315 opt-outs (c) and (e) respectively per the 5-cascade.

## §1.7 — The 8 master-gradient uses (operator question 2026-05-17)

Per symposium memo §3.6, the master gradient is a first-class TRAINING + DESIGN signal (not just an autopilot ranker input). Eight uses, with this campaign's wire-in ownership:

| # | use | this campaign owns? | wire-in surface |
|---|---|---|---|
| 1 | Score-aware loss term at byte-grain | YES (Wave 1.5; follow-on) | `tac.losses.master_gradient_term` |
| 2 | Per-pixel/per-byte attention reweighting | PARTIAL (consumed by Wave 3 U-DIE-KL) | `tac.losses.u_die_kl` |
| 3 | Bit allocator hook (Catalog #125 hook #3) | NO (Lane Ω-W-V3 owns; consume the gradient) | `tac.optimization.bit_allocator` |
| 4 | Architecture search / design discriminator | YES (Wave 4 substrate selection) | autopilot composition matrix |
| 5 | Score-aware QAT FP4 codebook | NO (QAT pipeline integration; deferred) | `tac.quantization.lsq_step_size` |
| 6 | Pareto facets feed Dykstra (Catalog #296) | YES (Wave 2 stacked-combination cross-term test) | `tac.optimization.dykstra_feasibility` |
| 7 | Continual-learning posterior for autopilot | YES (Phase 1 Wave 1; op-routable #3 Phase-7 lens) | `tac.autopilot_rudin_daubechies.master_gradient_lens` |
| 8 | *Magic codec*[^magic-codec-cp] per-stream selection | YES (Wave 4 format0d-CPU-sister; magic-codec score-aware mode) | `tac.packet_compiler.score_aware_selector` |

[^magic-codec-cp]: *Magic codec* = internal nickname for the per-stream optimal-entropy-coder auto-selector in `tac.packet_compiler.*`. See the PR writeup's Glossary §12.1 for the full definition.

Use #7 is the FOUNDATIONAL one — Wave 1 lands it and unlocks the others. Uses #1 + #2 land in Wave 1.5 + Wave 3 respectively as wire-ins to substrate trainers. Use #8 lands in Wave 4 as the magic-codec score-aware mode (replaces pure rate-minimization with score-aware-rate-minimization).

## §2 — Wave 2 (ORTHOGONAL EXPLOITS; days 3-5; $70)

**Purpose:** layer the first byte-disjoint cross-product exploits (SegNet-only + Wyner-Ziv rate-only) on top of fec6 PoseNet-only. Predicted: 0.174-0.187 [contest-CPU].

### §2.1 op-routable #2 — fec6 + SABOR sister submission

**Lane:** `lane_pr101_fec6_plus_sabor_disjoint_20260518`
**Pre-registration:** add-lane with `phase=2`, `target_modes=[contest_one_video_replay]`, `canary_status=post_canary_dependent`, `canary_dependency=lane_pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean`

**Design notes:**
- SABOR (Score-Aware Boundary Optimized Renderer) per `tac.symposium_impls.*` + the aerospace stealth memo §5.2 + the SegNet stride-2 stem blindspot analysis
- Archive grammar: add new section `boundary_pixels.bin` (~1-2 KB) carrying per-pair coordinates + RGB values for 3-5% argmax-boundary frame_1 pixels
- Inflate path: after fec6 frame_0 modifications complete, decode boundary_pixels.bin and overlay onto frame_1 RGB
- **Disjoint-byte guarantee:** boundary_pixels.bin is a SegNet-only Venn region byte stream because (a) it only modifies frame_1 (PoseNet sees these but only at the boundary pixels which are below FastViT's effective receptive field per the aerospace memo) and (b) the SegNet argmax decision flips on exactly these boundary pixels
- Adopted as new Phase-7 lens consumer: query master_gradient_lens before dispatch for predicted ΔS

**Dispatch command:**

```bash
.venv/bin/python tools/operator_authorize.py \
  --recipe .omx/operator_authorize_recipes/substrate_pr101_fec6_plus_sabor_modal_a100_dispatch.yaml \
  --estimated-cost-usd 40 \
  --session-budget-usd 200 \
  --confirmed-via-session-directive
```

**Build prerequisites:**
- `experiments/train_substrate_pr101_fec6_plus_sabor.py` — substrate trainer (uses `QuantizrFiveStageStaircase` from §1.3)
- `.omx/operator_authorize_recipes/substrate_pr101_fec6_plus_sabor_modal_a100_dispatch.yaml` — declares full Tier-1/2/3 schema per Catalog #270 + min_smoke_gpu A100 + canonical NVML block per Catalog #244
- `tools/build_pr101_fec6_plus_sabor_packet.py` — packet builder; emits archive with disjoint sections per Catalog #167 sister gate
- SABOR boundary classifier — vendored from `tac.symposium_impls.sabor_renderer_atick_redlich`; needs `_full_main` lift (currently scaffold)
- `submissions/pr101_fec6_plus_sabor/inflate.py` ≤200 LOC per HNeRV parity L4 budget

**Promotion criteria per Catalog #233 4-gate:**
1. Smoke green: rc=0 + auth-eval JSON parseable + final_score parsed
2. Tier C MDL density: measure on smoke archive; require density < 0.95 (not fully class-saturated)
3. 100ep CPU auth-eval anchor: byte-deterministic archive sha across re-runs; final_score < 0.190
4. Custody validated: `[contest-CPU GHA Linux x86_64]` axis, archive_sha256 matches

**Pre-PR commit gates:** Catalog #316 frontier-regression block compares against 0.19205; PASS required to submit.

**Predicted ΔS via master_gradient_lens:** -0.005 to -0.010 (lens query against op-routable #1 anchor).

### §2.2 op-routable #6 — L5 Time-Traveler Wyner-Ziv pose deltas on fec6

**Lane:** `lane_l5_wyner_ziv_fec6_20260518`
**Pre-registration:** add-lane with `phase=2`, `canary_status=post_canary_dependent`, `canary_dependency=lane_pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean`
**Cost:** $30

**Design notes:**
- Replace `poses.bin` (~4,800 bytes qpose14) with `poses_wyner_ziv.bin` (~1,500-2,000 bytes): per-pair pose RESIDUAL against an ego-motion predictor from the previous pair's frames
- Decoder reconstructs pose from `pose_residual + ego_motion_predictor(prev_frames_decoded)`
- Per Wyner 1976: information-theoretically lossless when the side info I_dec(X) is sufficient
- **Disjoint-byte guarantee:** the pose VALUES are unchanged (no SegNet/PoseNet effect); only the ENCODING is reduced (rate axis only)
- Architecture class-shift candidate per Z6/Z7/Z8 predictive-coding framework (per `feedback_six_meta_pattern_strict_gates_d_e_f_g_h_i_landed_20260516`)

**Dispatch command:** (operator_authorize.py invocation against `substrate_pr101_fec6_plus_l5_wyner_ziv_modal_a100_dispatch.yaml`)

**Build prerequisites:**
- `experiments/train_substrate_pr101_fec6_plus_l5_wyner_ziv.py` — trainer
- Recipe + lane driver + packet builder per same pattern as §2.1
- L5 trainer dependencies vendored from `src/tac/optimization/l5_*.py`
- Ego-motion predictor architecture per Rao+Ballard 1999 — small (~10K params) hierarchical Bayesian predictor

**Promotion criteria:** same 4-gate per Catalog #233.

**Predicted ΔS:** -0.008 to -0.015 (lens query).

### §2.3 Wave 2 stacked combination decision

Per the symposium's Assumption-Adversary verdict + Contrarian dissent: cross-product orthogonality is FIRST-ORDER true when byte-disjoint, but the empirical cross-term must be measured before claiming additive ΔS. Strategy:

1. Dispatch SABOR-only submission (§2.1) and L5-only submission (§2.2) in parallel as INDEPENDENT submissions
2. Harvest both; if both beat 0.19205 individually, dispatch a third submission stacking BOTH on fec6
3. Compare stacked-ΔS to (fec6_baseline_ΔS + sabor_ΔS + l5_ΔS); if within ±0.002, orthogonality holds; if not, the cross-term is real and needs separate measurement

**Cost reservation:** $40 (SABOR) + $30 (L5) + $40 (stacked combination) = $110 IF orthogonality holds; otherwise $70 + $30 cross-term measurement = $100. Plan for $110.

### §2.4 Wave 2 gates

- SABOR submission lands with paired-axis auth-eval JSON; final_score < 0.190 [contest-CPU]
- L5 submission lands with paired-axis auth-eval JSON; final_score < 0.185 [contest-CPU]
- Stacked combination (or cross-term measurement) decision data in `.omx/state/master_gradient_ledger.jsonl`
- Catalog #316 frontier scan updates to reflect new canonical best [contest-CPU]
- 2 new master-gradient anchors persisted

## §3 — Wave 3 (SUBSTRATE-WIDE LOSS ADOPTION; days 6-8; $60)

**Purpose:** U-DIE-KL substrate-wide loss adoption shifts the OPERATING POINT (second-order effect, not first-order Venn-region addition). Predicted: 0.165-0.175 [contest-CPU].

### §3.1 op-routable #4 — U-DIE-KL substrate-wide loss in fec6 trainer

**Lane:** `lane_u_die_kl_fec6_adoption_20260518`
**Pre-registration:** add-lane with `phase=3`, `canary_status=post_canary_dependent`, `canary_dependency=lane_pr101_lc_v2_clone`
**Cost:** $30-60 (Modal A100 retrain)

**Design notes:**
- U-DIE-KL = UNIWARD per-pixel weighting × Tishby Information Bottleneck KL loss (per `src/tac/losses/u_die_kl.py` + the Tishby-Zaslavsky 2015 framework)
- Adoption: replace fec6 trainer's score-aware loss (currently `100·d_seg + √(10·d_pose)`) with the U-DIE-KL loss that automatically per-pixel-weights by SegNet+PoseNet attention
- Trainer continues to use the master gradient as the per-byte cost function; U-DIE-KL is the per-pixel cost function (different grain)
- Expected effect: PoseNet `d_pose` drops from 2.943e-5 to ~5e-6 (5× reduction); SegNet `d_seg` drops from 5.6e-4 to ~3e-4 (1.9× reduction); operating point shifts; master gradient must be re-measured at the new operating point

**Promotion criteria:** Catalog #233 4-gate + Tier C density measurement (per Catalog #227) + class-shift evidence (token in lane notes per Catalog #311)

**Predicted ΔS:** -0.005 to -0.020 (lens query at fec6 operating point; actual delta likely larger at the shifted operating point)

### §3.2 Wave 3 gates

- U-DIE-KL fec6 submission lands with paired-axis auth-eval JSON; final_score < 0.18 [contest-CPU]
- Master gradient re-measurement triggered at new operating point ($5-15 dispatch sister)
- Tier C density measurement on the new archive ($0.30 CPU smoke)
- Catalog #316 frontier scan updates

## §4 — Wave 4 (STRETCH: CROSS-PARADIGM + CLASS-SHIFT; days 9-12; $50)

**Purpose:** explicit class-shift architecture exploration. Predicted: 0.155-0.170 [contest-CPU]. Stretch goal because class-shift architectures historically falsify-at-implementation per the operator-mandated CLAUDE.md "Substrate MUST be at OPTIMAL FORM" non-negotiable.

### §4.1 op-routable (deferred from symposium §5.4) — Cross-paradigm A1 ⊕ format0d-CPU-axis-sister + *magic-codec*[^magic-codec-cp-4] score-aware mode

**Lane:** `lane_a1_x_format0d_cpu_axis_sister_20260520`
**Pre-registration:** add-lane with `phase=4`, `canary_status=post_canary_dependent`, `canary_dependency=lane_pr101_fec6_plus_sabor_disjoint_20260518`

[^magic-codec-cp-4]: *Magic codec* = internal nickname; see PR writeup Glossary §12.1.

**Design notes (TWO LAYERS — corrected per operator 2026-05-17):**
- A1 (`87ec7ca5...`, 0.19285 [contest-CPU]) is the PR101-grammar minimalist baseline — separate from fec6 lineage
- **Layer 1 (wire-format grammar):** format0d-CPU-axis-sister = format0d additive correction stream with scales fitted against CPU PoseNet's gradient signature (not CUDA as in the original format0d)
- **Layer 2 (*magic-codec* per-stream entropy coder):** re-fit the per-stream codec selection table for the CPU PoseNet's byte-pattern preferences AND consume the master gradient (use #8) to prefer codecs whose byte patterns align with low-`|G|` regions. The original format0d's magic-codec selections were CUDA-fitted; CPU-axis requires fresh per-stream selection.
- Cross-product: A1 archive + format0d-CPU-sister Layer 1 + magic-codec-CPU-score-aware Layer 2 additive corrections; predicted to be sub-additive (factor 0.6-0.8 of the linear prediction) per Catalog #227 composition_alpha framework

**Promotion criteria:** Catalog #233 4-gate + sub-additivity composition_alpha measurement (per Catalog #227); only ratchet to submission if composition_alpha > 0.5

**Predicted ΔS:** -0.020 to -0.040 (relative to A1's 0.19285 baseline); if combined with fec6+SABOR+L5+U-DIE-KL, predicted final 0.155-0.170 [contest-CPU]

### §4.2 Alternate Wave 4: Z6/Z7/Z8 predictive-coding substrate

If §4.1 is rejected by codex pre-dispatch review or by the autopilot's master_gradient_lens (low EIG/$ verdict), the alternate Wave 4 candidate is one of the Z6/Z7/Z8 ego-motion-conditioned predictive-coding substrates (per `feedback_six_meta_pattern_strict_gates_d_e_f_g_h_i_landed_20260516`). These are architecture class-shift candidates with explicit cooperative-receiver framing.

## §5 — Budget envelope + risk register

### §5.1 Budget breakdown — REVISED per Round-1 review M-5 + M-6

Original estimates were systematically too low (post-mortem-30d + Karpathy findings). The table below reflects realistic build + dispatch + master-gradient-remeasurement costs:

| wave | activities | cost (revised) | notes |
|---|---|---|---|
| 1 | op-routable #1 (autograd + top-K discrete sister) + #3 + #5 + #7 | $0.50-5 | only #1 paid; revised methodology per §3.2 |
| 2 | op-routable #2 (SABOR; build $50-80) + #6 (L5; $40-60) + stacked-combination cross-term | $90-170 | per M-5 corrected build costs |
| 2.5 | master-gradient re-measurement at new operating point | $5-15 | per M-6 added; required per §3.5 |
| 3 | op-routable #4 (U-DIE-KL; $60-120 + retrain) | $60-120 | per M-5 corrected |
| 3.5 | master-gradient re-measurement | $5-15 | per M-6 added |
| 4 (stretch) | cross-paradigm format0d-CPU-sister ($80-150) | $80-150 | per M-5 corrected |
| 4.5 | master-gradient re-measurement | $5-15 | per M-6 added |
| **Total (full campaign with Wave 4)** | | **$245-490** | |
| **Total (core, no Wave 4)** | | **$160-325** | |

**Envelope must increase from $200 to $400 minimum for full campaign; $300 if Wave 4 is deferred.** Per CLAUDE.md "Frontier target — NON-NEGOTIABLE": deferral order if budget is constrained is Wave 4 → Wave 3 → Wave 2.5 re-measurement → Wave 2 stacked combination → Wave 2 individual submissions → Wave 1.

### §5.2 Risk register

| risk | likelihood | mitigation |
|---|---|---|
| Master gradient finite-difference probe takes longer than $15 | medium | sequential probe chunking with per-chunk cost cap; abort + escalate at $20 spend |
| SABOR boundary classification has unmeasured PoseNet effect | medium | per-pair `d_pose` measurement on smoke archive before full dispatch; abort if `d_pose` regression > 5% |
| L5 ego-motion predictor doesn't converge in budget | low | predictor is small (~10K params); cosine LR + EMA per Quantizr discipline |
| U-DIE-KL adoption regresses SegNet (over-weights PoseNet attention) | medium-high | smoke-before-full dispatch with operator_authorize Catalog #167; abort if smoke d_seg regresses > 10% |
| Cross-product orthogonality breaks (cross-term > ±0.002) | medium | measure cross-term explicitly; if real, partition campaign into byte-disjoint single-exploit submissions |
| CPU/CUDA bifurcation flips on a sub (e.g., SABOR works CPU but regresses CUDA) | low | acceptable — campaign optimizes CPU axis explicitly per operator directive; CUDA regression noted in writeup, not blocking |
| 5/√(10·d_pose) marginal correction surfaces other math errors in the writeup | low | sister codex adversarial review on the writeup after correction lands |
| Modal call_id ledger or master_gradient_ledger corrupts | very low | both helpers use fcntl-locked JSONL with strict-load + quarantine per Catalog #128/#131/#138/#245 |

### §5.3 Off-ramps

- **Hard off-ramp at $200 total spend:** stop all dispatch; consolidate best CPU score; submit
- **Off-ramp after Wave 2 if no submission beats 0.19:** falsify cross-product orthogonality assumption; re-deliberate at T3 council
- **Off-ramp after Wave 3 if U-DIE-KL regresses:** abandon U-DIE-KL adoption; proceed direct to Wave 4 stretch
- **Hard PR deadline override:** if contest deadline approaches, submit the best of {Wave 1-paid, Wave 2 individual, Wave 2 stacked, Wave 3, Wave 4 alternatives} and ship; per CLAUDE.md "Frontier target" submission-escrow discipline

## §6 — Continuous wire-ins (cathedral autopilot)

After every wave's dispatch outcome:

1. Modal call_id ledger records dispatched + harvested events (Catalog #245)
2. Continual-learning posterior receives the auth-eval anchor (Catalog #128)
3. Master_gradient_ledger receives the new gradient anchor (after Wave 1; consumes finite-difference output)
4. Council deliberation posterior receives the auth-eval-result-review verdict (Catalog #300; T2 council per cadence budget)
5. Cathedral autopilot ranker reads all 7 surfaces; re-ranks remaining campaign candidates per the Rashomon ensemble's consensus
6. Catalog #316 frontier scan updates `reports/latest.md` FRONTIER section
7. Pre-submission compliance check (per `scripts/pre_submission_compliance_check.py --contest-final`) refuses any candidate that regresses against the canonical best per Catalog #316

## §7 — Definition of done

The campaign is COMPLETE when ANY of:

1. Frontier reaches **0.165 [contest-CPU]** (operator-mandated target)
2. Budget exhausted ($200 reached) AND best frontier is below 0.180 [contest-CPU]
3. Contest submission deadline reached AND best submission deposited via PR
4. Three consecutive waves produce no improvement > 0.005 (diminishing returns; per CLAUDE.md "Forbidden premature KILL" + per Z1 within-class-trap analysis)

Output artifacts:
- `experiments/results/<campaign-final-archive>/` — final submission packet
- `.omx/state/master_gradient_anchors.jsonl` — 4+ master gradient anchors
- `.omx/state/council_deliberation_posterior.jsonl` — 4+ T2 council deliberations
- `.omx/state/continual_learning_posterior.jsonl` — 4+ paired-axis auth-eval anchors
- `docs/pr_writeups/cpu_frontier_<final_archive>.md` — updated writeup with final score
- `reports/latest.md` — FRONTIER section updated to final canonical best
- `feedback_cpu_frontier_master_gradient_campaign_landed_20260527.md` — campaign retrospective memory entry

## §8 — Cross-references

- `.omx/research/grand_reunion_t4_symposium_orthogonal_optimization_master_gradient_20260517.md` — the symposium memo this plan operationalizes
- `docs/pr_writeups/cpu_frontier_fec6_20260517.md` — the current PR writeup (will be amended per op-routable #7)
- `.omx/research/full_problem_space_reverse_engineering_cpu_gpu_both_20260517.md` — full problem-space analysis
- `.omx/research/alien_tech_reverse_engineering_pr106_format0_family_20260517.md` — alien-tech reference
- `feedback_permanent_fix_frontier_signal_loss_landed_20260517` — Catalog #316 canonical state infrastructure
- `feedback_canonical_dispatch_optimization_protocol_landed_20260515` — Catalog #270 dispatch protocol
- `feedback_rudin_daubechies_autopilot_full_implementation_landed_20260515` — Phase 1-6 lens architecture
- `feedback_modal_call_id_ledger_canonical_landed_20260515` — Catalog #245 4-layer canonical ledger pattern (template for master_gradient_ledger)
- `feedback_council_hierarchy_v2_landed_20260516` + `feedback_mission_alignment_followon_catalog_300_extension_landed_20260516` — Catalog #300 v2 frontmatter + mission-alignment fields

## §9 — Decisions awaiting operator approval

1. **Ratify campaign plan as-is** (Wave 1 starts immediately) — or amend
2. **Approve $200 envelope** — or constrain
3. **Approve master_gradient_ledger Catalog # claim** — auto-claim via `tools/claim_catalog_number.py --commit-via-serializer` is the canonical pattern
4. **Approve Phase-7 lens module wire-in to cathedral autopilot** — single-line autopilot edit; reviewable in 30 seconds
5. **Approve op-routable #7 writeup correction** before opening the contest PR — recommend YES
6. **Approve push of OSS changes to comma-lab/main + tac/main** before opening the contest PR — operator's prior directive was to have main as sole source of truth; this campaign's outputs should land there incrementally

After operator approval, Wave 1 fires within 30 minutes.
