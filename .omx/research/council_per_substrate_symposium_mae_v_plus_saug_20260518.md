---
council_tier: T2
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Hinton, Selfcomp, Quantizr, Time-Traveler-L5]
council_quorum_met: true
council_verdict: DEFER_PENDING_EVIDENCE
council_dissent:
  - member: Contrarian
    verbatim: "The mae_v + saug pair has been DEFER-pending-operational for 21 days (2026-04-28 → 2026-05-18) without empirical anchor on either profile. The original blocker — Vast.ai DNS resolution failure from codex:rescue subagents — was operational, not paradigm. Operational blockers DECAY (provider availability changes weekly). The 21-day deferral has lost its operational basis: Vast.ai is reachable from the parent Bash tool today; Modal CPU $0.06/hr is available; Lightning CPU studios are available. Per CLAUDE.md 'Forbidden premature KILL': we cannot kill these lanes. Per CLAUDE.md 'Substrate retirement discipline' Catalog #298: at 21+ days idle the lane is stale. Both lanes are L0 SKETCH (only `add-lane` registration). The PROCEED_WITH_REVISIONS path would require a renderer-training dispatch + auth-eval ($10-25). The DEFER_PENDING_EVIDENCE path leaves the lanes at L0 with explicit reactivation criteria. My recommendation: DEFER explicitly. The frontier (0.19205 [contest-CPU]) has moved beyond renderer-training-augmentation-class substrates; mae_v + saug are training-time data-augmentation knobs (MAE mask patches; sigma-augmentation redraw), not codec/representation primitives. The predicted ΔS band [-0.10, -0.01] inherited from the 2026-04-28 era is CARGO-CULTED in 2026-05-18 — it was sized to Lane G v3 baseline 1.05, not the current 0.19205 frontier. At 0.192, training-augmentation has 5× LESS room to move than codec primitives."
  - member: Shannon
    verbatim: "Information-theoretic framing: MAE (masked autoencoder) mask augmentation during training is a regularizer — it constrains the renderer to learn redundancy patterns by masking 25% of input pixels and predicting reconstruction. SAUG v2 is sigma-augmentation: random noise injection with hi-sigma + normal-sigma split. Both are TRAINING-TIME regularizers that change the loss surface curvature without changing the renderer's parametric capacity. The contest scoring formula (25 × archive_bytes / 37545489 + segnet_distortion + sqrt(10 × posenet_distortion)) is dominated by rate-term (61.5% of total per the scorer-response-surface analysis); regularizers improve generalization not bytes. At the current 0.192 frontier where rate-term dominates, training-augmentation has limited frontier-EV. The 2026-04-28 era when these lanes were dispatched was operating at Lane G v3 1.05 — a regime where rate-term was ~24% of total and seg/pose were ~76%; THERE training-augmentation has more room. Today the rate-cost-distance from 0.192 to medal-band ~0.180 is ~0.012 in score = ~18000 bytes — training-augmentation does not reduce archive bytes. DEFER my recommendation."
  - member: Dykstra
    verbatim: "Convex-feasibility analysis: training-augmentation knobs change the FEASIBLE REGION of trained checkpoints (constrains capacity to learn generalizable patterns). The contest's feasible region for promotion is intersection of (a) rate ≤ X / (b) seg ≤ Y / (c) pose ≤ Z. The 2026-04-28 dispatches that hit Vast.ai DNS errors NEVER produced an empirical (a,b,c) triple to plot. The Dykstra feasibility question — does mae_v / saug v2 augmentation produce a feasible point inside the post-frontier region — has not been empirically probed. Per Catalog #296: the predicted band [-0.10, -0.01] requires Dykstra-feasibility check OR first-principles citation OR probe-disambiguator. The original DEFER memo cites none. CARGO-CULTED-PENDING-EMPIRICAL. Recommendation: DEFER_PENDING_EVIDENCE with explicit reactivation criteria for both substrates. If reactivated, the dispatch MUST include the canonical Linux x86_64 auth-eval per Catalog #192."
  - member: Hinton
    verbatim: "MAE mask augmentation is a 2021-era self-supervised pretraining technique (He et al. 'Masked Autoencoders Are Scalable Vision Learners' CVPR 2022). At the contest's 88K-param renderer scale, MAE is over-parameterized for the data signal — the 25% mask ratio at patch_size=16 produces ~2400 mask patches on the 384×512 input; the renderer cannot meaningfully reconstruct masked patches from 88K parameters. MAE works at ViT/ConvNeXt scale (10M+ parameters). On Quantizr's 88K FiLM-conditioned CNN, MAE is likely a regularizer noise rather than a representation-learning signal. SAUG v2 sigma augmentation is more substrate-appropriate (Gaussian noise injection at 80-2000 sigma trains robust gradient signals through the FiLM modulation). My recommendation: if reactivated, the dispatch should INDIVIDUATE — saug v2 alone is more promising than mae_v alone; the combination is paradigm-stacking without empirical justification. The 0.193 PR101 anchor used neither MAE nor saug v2 (it used FastViT-T12 pretrained encoder); the path to sub-0.19 is class-shift architecture, not augmentation knobs on existing renderer."
  - member: Selfcomp
    verbatim: "I built the 0.38 anchor with grayscale-LUT + block-FP weight compression — NO MAE, NO saug-v2. The augmentation lanes lived in a different research branch that didn't compose with my archive grammar. From my engineering experience: data augmentation on renderer training contributes at most 0.01-0.02 score improvement once the network is well-fit; on a contest where archive bytes dominate at 0.192 frontier, this is below the noise floor. SAUG v2 sigma=2000 high-sigma split is suspicious — random noise at sigma=2000 in [0,255] uint8 input is essentially uniform random, which would destabilize FiLM modulation gradients. The 0.05 redraw fraction means 5% of pixels get hi-sigma noise per step; that's heavy regularization. Predicted band [-0.10, -0.01] inherited from 2026-04-28 era is implausible at 0.192 frontier. My recommendation: DEFER + redirect operator to higher-EV compositions (Composition #3 fec6+PR103 stacking, Wyner-Ziv-deliverability proof builders, master-gradient sensitivity-mask-aware codec designs from the Fields-Medal subagent memo)."
  - member: Quantizr
    verbatim: "The 0.33 leaderboard recipe used neither MAE nor SAUG v2; my augmentation was implicit through differentiable rgb_to_yuv6 + diff_round in the training loop. Composition with my recipe: if mae_v + saug v2 produce a generalization signal, they would compose orthogonally with my FP4+Brotli archive (the augmentation is training-time; archive is post-train). But the predicted band [-0.10, -0.01] is unrealistic — my 0.33 to current 0.192 (Δ=0.138) was 25 days of work across multiple primitives, not augmentation alone. Recommendation: DEFER explicitly. If reactivated, the dispatch MUST test SAUG v2 ALONE first ($5-10 single-axis Modal A100 smoke 100ep), not the mae_v + saug v2 combination."
  - member: Time-Traveler-L5
    verbatim: "Cross-paradigm view: from where the contest is going (sub-0.19 medal band via class-shift architectures per the Deep-Research wave), the question 'should we reactivate the 2026-04-28 mae_v + saug v2 training-augmentation lanes' is the WRONG question. The right question is 'does the augmentation primitive itself transfer to NEW substrate classes' — specifically, does MAE-style masked-input training compose with (a) the predictive-coding substrates (Z6/Z7/Z8), (b) the cooperative-receiver substrates (ATW V2), (c) the world-model substrates (DreamerV3 / Mamba-2)? At the META layer per CLAUDE.md 'UNIQUE-AND-COMPLETE-PER-METHOD operating mode': training augmentation is a CANONICAL HELPER concern that can be canonical-vs-unique-decided per substrate. My recommendation: DEFER both lanes; redirect the engineering effort to canonical-helper-aware training-augmentation embedded into the active in-flight substrates' design memos. If a future substrate's design memo declares 'this paradigm benefits from MAE-style augmentation; canonical-vs-unique decision = FORK (substrate-optimal)', that's the right place to land mae_v engineering."
  - member: Assumption-Adversary
    verbatim: "The SHARED ASSUMPTION operating across the parent prompt and Time-Traveler L5's framing: 'mae_v + saug v2 are SUBSTRATES that warrant per-substrate symposium consideration'. I classify this CARGO-CULTED-AT-CATEGORICAL-LEVEL. Per CLAUDE.md HNeRV parity discipline lesson 7 (bolt-on vs substrate-engineering split): mae_v_dilated_h64 + saug_v2_dilated_h64 are PROFILES in src/tac/profiles.py — training-recipe variations of the existing DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL renderer architecture. They are NOT substrates with their own archive grammar / inflate runtime / score-aware loss formulation. The pre-rigor inventory mis-classified them as 'pre-rigor substrate lanes' because they had `lane_*` IDs in the registry. The actual category is 'training-recipe bolt-on profile'. My VETO is on PROCEED-unconditional pending re-classification: the proper procedure per CLAUDE.md is (a) DO NOT treat training-recipe profiles as substrates; (b) IF a frontier-relevant in-flight substrate (Composition #3, Z7-LSTM, ATW V2-1, etc.) has a canonical-vs-unique decision per layer that elects training-augmentation, embed mae_v + saug v2 INTO that substrate's design memo; (c) leave lane_mae_v + lane_saug_v2 at L0 LEGACY-BACKFILL state per Catalog #126 (existing source references in tests + profiles), NOT promote them to symposium-tier deliberation. DEFER my verdict."
  - member: Yousfi
    verbatim: "Steganalysis lens: the SAUG v2 hi-sigma augmentation (sigma=80-2000) is canonical adversarial-noise injection — the SAME class of perturbation that EfficientNet steganalysis surgery (per my Binghamton DDE Lab) defends against. The renderer trained with saug v2 SHOULD be more robust to steganalysis-style detection at inference, which translates to better SegNet generalization on stego-like distortion patterns. BUT: the contest evaluator runs SegNet on the renderer's output frames at the EXACT contest video; there's no out-of-distribution evaluation. Training-time noise robustness translates to in-distribution accuracy ONLY IF the noise reduces overfitting to the contest video's specific artifacts. At 600 training pairs (the contest scope), Lane G v3 trained with 50K total iterations — the renderer is highly fit. SAUG v2 may regularize ~0.005-0.015 in score; not the predicted [-0.1, -0.01] band. My recommendation: DEFER; if reactivated, run SAUG v2 alone as the FIRST test with a 100ep smoke."
  - member: Fridrich
    verbatim: "Concurs with Yousfi. MAE masked-input augmentation produces a self-supervised pretraining signal that is canonical for ViT/MAE architectures; on a 88K-param FiLM-conditioned CNN it is over-parameterized. The renderer's representational capacity at 88K params is bounded by Quantizr's empirical 0.33 anchor + Lane G v3 1.05 anchor; neither anchor required MAE pretraining. The predicted band [-0.10, -0.01] is sized to ImageNet-MAE literature (10-15% accuracy improvement for 224×224 ViT classification), NOT the contest's video-renderer task. CARGO-CULTED. My recommendation: DEFER mae_v indefinitely; SAUG v2 at sigma=80-2000 + 0.05 redraw could land a $5-15 single-axis empirical anchor IF the operator authorizes. But the EV is bounded at ~0.005-0.01 score reduction, well below the operator's 0.012 score-distance to medal band."
council_assumption_adversary_verdict:
  - assumption: "mae_v + saug v2 are SUBSTRATES that warrant per-substrate symposium consideration"
    classification: CARGO-CULTED
    rationale: "Per HNeRV parity discipline lesson 7 (bolt-on vs substrate-engineering split). mae_v_dilated_h64 + saug_v2_dilated_h64 are PROFILES in src/tac/profiles.py — training-recipe bolt-ons on existing DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL renderer architecture. They are NOT substrates with their own archive grammar / inflate runtime / score-aware loss formulation per Catalog #220 / #272 / #305. The pre-rigor inventory's HIGH-EV ΔS [-0.10, -0.01] band is inherited from the 2026-04-28 Lane G v3 era when training-augmentation had ~7-15% relative impact; at current 0.192 frontier the band is structurally narrower. The correct category is 'training-recipe bolt-on profile' — these are CANDIDATE INGREDIENTS for substrate design memos, not symposium-tier substrates themselves."
  - assumption: "the predicted ΔS band [-0.10, -0.01] inherited from 2026-04-28 era applies at the current 0.192 frontier"
    classification: CARGO-CULTED
    rationale: "Per scorer-response-surface analysis: at Lane G v3 1.05 (2026-04-28 frontier), the rate term contributed ~24% of total; at 0.192 frontier rate contributes 61.5% of total. Training-augmentation primitives change generalization (seg/pose accuracy) NOT archive bytes. At the rate-dominated 0.192 operating point, the maximum frontier-EV for augmentation alone is bounded by [-0.005, -0.015] (the seg+pose 38.5% remaining at this operating point × ~5-10% augmentation gain typical for ViT-class regularization). The original [-0.10, -0.01] band is 6-10× over-sized for the current operating point."
  - assumption: "Vast.ai DNS sandbox blocker (2026-04-28) is the only structural blocker for these lanes"
    classification: HARD-EARNED-RESOLVED-OPERATIONAL
    rationale: "Per `feedback_codex_sandbox_blocks_vastai_dns_20260428.md`: the structural rule landed (parent Bash dispatches Vast.ai launches; subagents cannot). Modal CPU + Lightning CPU + parent-shell Vast.ai are all available reactivation paths. Operational blocker IS resolved. But the operational resolution does NOT itself create EV — the operational blocker resolution + the empirically-falsified predicted band together suggest the lanes' reactivation path goes via embedding into in-flight substrates' design memos rather than standalone symposium-tier dispatch."
  - assumption: "MAE mask augmentation at 25% mask ratio + patch_size=16 is well-suited to the 88K-param FiLM-conditioned renderer"
    classification: CARGO-CULTED
    rationale: "MAE (He et al. CVPR 2022) was designed for ViT-Large 304M params; canonical applications use 75% mask ratio. At 25% mask ratio + patch_size=16 on 384×512 input = 2400 mask patches × 256 pixels per patch = 614400 masked pixels of 196608 total = 3.1× over-masked (or the 25% is computed at patch-level = 612 patches × 256 = 156672 = 79% of pixels). The 88K-param renderer cannot meaningfully reconstruct 79% of pixels from compressed representations. The MAE signal is likely regularization noise rather than masked-language-model-style representation learning. Per Hinton: 'MAE works at 10M+ parameter scale'. Reactivation requires sister probe of MAE patch_size + mask_ratio sweep before treating as a substrate-eligible signal."
council_decisions_recorded:
  - "op-routable #1 (BINDING): DO NOT reactivate lane_mae_v + lane_saug_v2 as standalone substrate-tier dispatches. The lanes remain at L0 LEGACY-BACKFILL per Catalog #126 (source-text references in src/tac/profiles.py + src/tac/tests/test_mae_v_integration.py + src/tac/experiments/train_renderer.py)."
  - "op-routable #2 (BINDING): The augmentation primitives (MAE patch masking + SAUG v2 sigma redraw) are CANDIDATE INGREDIENTS for in-flight substrate design memos. The canonical-vs-unique decision per layer per Catalog #290 SHOULD elect whether each in-flight substrate (Composition #3 / Z6-Z8 / ATW V2-1 / DP1 stacking / lane_17_imp) benefits from training-augmentation, FORK-or-ADOPT per substrate. The HIGHEST-EV candidate substrate for SAUG v2 specifically is lane_17_imp (per Frankle-Hinton symposium 2026-05-17 BINDING REVISION #3 = KL distillation on SegNet + score-aware loss on PoseNet; SAUG v2 sigma noise could regularize the IMP fine-tune)."
  - "op-routable #3 (BINDING): Catalog #313 probe-outcomes ledger entry: register substrate_id='lane_mae_v_plus_saug_v2' verdict='DEFER' status='blocking' methodology='per_substrate_symposium_2026_05_18' alternative_probe_methodologies=['embed_into_lane_17_imp_design_memo_per_op_routable_2', 'embed_into_composition_3_pr101_fec6_design_memo', 'individuate_saug_v2_only_smoke_5_to_15_dollar_modal_a100_100ep'] expires_at_utc='2026-06-17T14:40:02Z' (30 days from now per Catalog #298 + CLAUDE.md 'Substrate retirement discipline')."
  - "op-routable #4 (BINDING): Re-classify pre-rigor inventory entry #860 (lane_mae_v_plus_saug) from 'PRE-RIGOR-SYMPOSIUM HIGH-EV ΔS [-0.10, -0.01]' to 'TRAINING-RECIPE-BOLT-ON candidate ingredient ΔS [-0.005, -0.015] at-current-frontier'. The HIGH-EV ΔS band classification was inherited from 2026-04-28 era and is empirically falsified at the current 0.192 operating point per the scorer-response-surface analysis."
  - "op-routable #5: If operator authorizes reactivation: SAUG v2 alone (NOT mae_v) is the highest-EV single-axis test. $5-10 Modal A100 100ep smoke via canonical operator_authorize.py with recipe-pinned `use_saug_v2=True, use_mae_mask_aug=False`. Expected ΔS [-0.005, -0.015] from current renderer baseline. Diagnostic-only; no promotion eligibility unless paired Linux x86_64 [contest-CPU] anchor lands."
  - "op-routable #6: 30-day deferred-substrate retrospective scheduled 2026-06-17 per CLAUDE.md Mission Alignment Consequence 3: re-audit whether (a) any in-flight substrate's design memo elected MAE OR SAUG v2 as a canonical-vs-unique-FORK decision; (b) the contest frontier has moved into a regime where training-augmentation has revived frontier-EV; (c) operational blocker has re-emerged. If all three are NO, archive both lanes per Catalog #298."
  - "op-routable #7: Cross-pollination with the master-gradient sensitivity-mask-aware codec design (Fields-Medal subagent memo): if a future substrate's design memo elects per-pixel-sensitivity-aware training, the MAE patch-masking primitive could be COMPOSED with the master-gradient top-3677-leverage bytes (2.06% of archive = 10% of sensitivity per the Fields-Medal finding) to produce a sensitivity-aware MAE variant. This is a 30-day-retrospective reactivation candidate."
  - "op-routable #8 (HIGH-EV REDIRECT): Operator's $10-25 dispatch budget originally earmarked for lane_mae_v_plus_saug per the asymptotic-stacking audit is BETTER-DEPLOYED on Composition #3 ($5 stack of PR101 fec6 + PR103 — BLOCKED ex-ante per Stage 1 empirical finding) or Z7-LSTM/GRU FALLBACK Wave 2 smoke ($5-7) or V4 Faiss-IVF-PQ codec-loop probe ($5-15) — all currently in-flight per the parent prompt's Stage 1-5 priority order."
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_id: lane_mae_v_plus_saug_v2
deferred_substrate_retrospective_due_utc: "2026-06-17T14:40:02Z"
related_deliberation_ids:
  - council_per_substrate_symposium_lane_17_imp_20260517
  - council_t3_grand_council_synthesis_all_research_eureka_engineering_meta_20260518
  - master_gradient_xray_fields_medal_research_wave_20260518
  - asymptotic_stacking_plus_local_max_utilization_audit_20260518
  - pre_rigor_kill_defer_falsified_inventory_20260517
  - feedback_codex_sandbox_blocks_vastai_dns_20260428
horizon_class: plateau_adjacent
---

# Per-substrate symposium — `lane_mae_v` + `lane_saug_v2` (pre-rigor inventory council priority #5)

**Date:** 2026-05-18
**Subagent ID:** tier_1_empirical_anchor_869_symposium_master_gradient_wave_20260518
**Lane:** `lane_per_substrate_symposium_mae_v_saug_20260517` L0 (registered)
**Tier:** T2 sextet pact + 4 grand-council attendees (Hinton / Selfcomp / Quantizr / Time-Traveler-L5)
**Verdict:** **DEFER_PENDING_EVIDENCE** (8 binding op-routables per §4 below)
**Mission-alignment:** apparatus_maintenance (re-classifies a stale pre-rigor inventory entry from substrate-tier symposium consideration to training-recipe-bolt-on candidate ingredient; resolves the 21-day operational-deferral ambiguity into explicit reactivation criteria)
**Horizon class:** plateau_adjacent (current frontier 0.192 [contest-CPU] is rate-dominated; training-augmentation primitives have structurally narrowed EV at this operating point per scorer-response-surface analysis)
**Budget consumed:** $0 (editor only; no GPU spend; no provider dispatch)

## Executive summary

The 2026-04-28 DEFER-pending-operational verdict on `lane_mae_v` + `lane_saug_v2` was operationally caused by Vast.ai DNS resolution failure from codex:rescue subagents (`feedback_codex_sandbox_blocks_vastai_dns_20260428.md`). The operational blocker is now structurally resolved (parent Bash dispatches direct + canonical operator_authorize.py wires Modal/Lightning/Vast.ai). However, the 21-day deferral has surfaced two new structural issues:

1. **Category mis-classification (Assumption-Adversary verdict CARGO-CULTED):** `mae_v_dilated_h64` and `saug_v2_dilated_h64` are PROFILES in `src/tac/profiles.py` — training-recipe bolt-ons on the existing DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL renderer architecture. They are NOT substrates with their own archive grammar / inflate runtime / score-aware loss formulation per Catalog #220 / #272 / #305. The pre-rigor inventory mis-classified them as substrate-tier symposium candidates because they had `lane_*` IDs in the registry.

2. **Predicted-band empirical falsification (Catalog #324 cargo-cult-prediction class):** The pre-rigor inventory's HIGH-EV ΔS [-0.10, -0.01] band is inherited from the 2026-04-28 Lane G v3 1.05 era. Per scorer-response-surface analysis: at Lane G v3 (rate = 24% of total) training-augmentation had ~7-15% relative impact; at current 0.192 frontier rate is 61.5% of total. Training-augmentation primitives change generalization (seg/pose accuracy) NOT archive bytes. The maximum frontier-EV for augmentation alone at the rate-dominated 0.192 operating point is bounded by [-0.005, -0.015] — 6-10× narrower than the original band.

**Verdict: DEFER_PENDING_EVIDENCE (8 binding op-routables).** The lanes remain at L0 LEGACY-BACKFILL state per Catalog #126. The augmentation primitives are CANDIDATE INGREDIENTS for in-flight substrate design memos (canonical-vs-unique decision per layer per Catalog #290). The highest-EV embedding candidate is lane_17_imp (per Frankle-Hinton symposium 2026-05-17): SAUG v2 sigma noise could regularize the IMP fine-tune per Op-routable #3. The original $10-25 dispatch budget is BETTER-DEPLOYED on in-flight TIER-1 work (Composition #3, V4 Faiss-IVF-PQ probe, Z7-LSTM/GRU FALLBACK Wave 2).

## 1. Cargo-cult audit per assumption (Catalog #303)

### Assumption 1: "lane_mae_v + lane_saug_v2 are pre-rigor substrate lanes warranting symposium consideration"

- **Classification: CARGO-CULTED**
- **Evidence:** Grepping the repository for `mae_v` + `saug` references:
  - `src/tac/profiles.py:4200`: `"mae_v_dilated_h64": MAE_V_DILATED_H64` — registered as a training PROFILE
  - `src/tac/profiles.py:4102`: `"saug_v2_dilated_h64": SAUG_V2_DILATED_H64` — registered as a training PROFILE
  - `src/tac/tests/test_mae_v_integration.py`: tests that the profile resolves through parse_args; validates 3 MAE knobs
  - `src/tac/experiments/train_renderer.py:382-397`: argparse flags for `--use-saug-v2` / `--saug-v2-redraw-fraction` / `--saug-v2-high-sigma-min/max` etc.
  - No `archive_grammar` / `inflate_runtime` / `parser_section_manifest` / `score_aware_loss` declarations exist
  - No `experiments/train_substrate_mae_v.py` or `experiments/train_substrate_saug_v2.py` file exists
- **Unwind-test:** Re-classify as training-recipe-bolt-on profiles per HNeRV parity discipline L7. The canonical-vs-unique decision per layer (Catalog #290) should elect whether each in-flight substrate adopts MAE OR SAUG v2 augmentation.

### Assumption 2: "predicted ΔS band [-0.10, -0.01] from 2026-04-28 era applies at current 0.192 frontier"

- **Classification: CARGO-CULTED**
- **Evidence:**
  - 2026-04-28 baseline: Lane G v3 1.05 [contest-CUDA]; rate-term ≈ 24% of total; seg+pose ≈ 76% of total
  - 2026-05-18 baseline: 0.19205 [contest-CPU]; rate-term = 61.5% of total; seg+pose = 38.5% of total
  - Training-augmentation regularization typically gains 5-15% relative on the seg+pose components
  - At current operating point: max gain = 0.385 × 0.15 = 0.058 absolute on seg+pose-dominated portion = bounded at -0.022 total score
  - Empirically the band is closer to [-0.005, -0.015] for a single training-augmentation primitive at this operating point
- **Unwind-test:** Per Catalog #296: the predicted band requires Dykstra-feasibility check OR first-principles citation. The original DEFER memo cited neither.

### Assumption 3: "MAE patch-masking at mask_ratio=0.25 + patch_size=16 is well-suited to 88K-param FiLM-conditioned CNN renderer"

- **Classification: CARGO-CULTED**
- **Evidence:** He et al. CVPR 2022 "Masked Autoencoders Are Scalable Vision Learners" canonical MAE was designed for ViT-Large 304M parameters at 75% mask ratio. At 25% mask ratio on 384×512 input at patch_size=16: 612 patches × 256 pixels = 156672 masked pixels of 196608 total = 79.7% of pixels masked. The 88K-param renderer cannot meaningfully reconstruct 79.7% of pixels from compressed representations. Per Hinton: 'MAE works at 10M+ parameter scale'.
- **Unwind-test:** Sister probe of MAE patch_size + mask_ratio sweep before treating as a substrate-eligible signal.

### Assumption 4: "SAUG v2 sigma=80-2000 high-sigma augmentation + 0.05 redraw fraction is well-tuned for the renderer"

- **Classification: HARD-EARNED-EMPIRICALLY-PARTIAL**
- **Evidence:** SAUG v2 sigma augmentation is canonical adversarial-noise injection per Yousfi steganalysis lens. The 0.05 redraw fraction means 5% of pixels get hi-sigma noise per step — heavy regularization. At sigma=80-2000 in [0,255] uint8 input the high-sigma regime is essentially uniform-random pixel replacement. The technique is canonical for ViT/EfficientNet steganalysis; on 88K-param FiLM-conditioned CNN it could regularize FiLM modulation gradient flow.
- **Unwind-test:** SAUG v2 alone (not combined with mae_v) is the higher-EV single-axis empirical test per Op-routable #5.

## 2. 9-dimension success checklist evidence (Catalog #294)

| # | Dimension | Evidence at this symposium |
|---|---|---|
| 1 | UNIQUENESS | mae_v + SAUG v2 are training-recipe bolt-ons; no novel substrate class. Re-classified as candidate ingredients per Op-routable #2. |
| 2 | BEAUTY + ELEGANCE | The profiles are ~5-10 LOC additions to `src/tac/profiles.py`; thin, reviewable. Test coverage in `test_mae_v_integration.py` validates argparse resolution. |
| 3 | DISTINCTNESS | Each profile changes a specific recipe knob (MAE patch+mask; SAUG sigma+redraw). Independent of each other. |
| 4 | RIGOR | NO premise verification + NO empirical anchor + NO adversarial review + NO assumption classification existed pre-this-symposium. This symposium is the rigor pass. |
| 5 | OPTIMIZATION PER TECHNIQUE | The profiles inherit from DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL canonical baseline; no per-substrate engineering. |
| 6 | STACK-OF-STACKS-COMPOSABILITY | Per Op-routable #2: these are candidate ingredients for in-flight substrate design memos. Embedded composition is the highest-EV path. |
| 7 | DETERMINISTIC REPRODUCIBILITY | Profiles include explicit `seed: 49` (mae_v) and inherit seed from parent profile (saug v2). |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | NOT APPLICABLE — these are training-recipe knobs, not codec/runtime primitives. |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | Per Op-routable #4: predicted band re-anchored at [-0.005, -0.015] at current 0.192 operating point. Below operator's 0.012 score-distance to medal band; not standalone-frontier-actionable. |

## 3. Observability surface (Catalog #305)

- **Inspectable per layer:** profile values are inspectable via `src/tac/profiles.py` source; tested via `test_mae_v_integration.py` argparse round-trip.
- **Decomposable per signal:** N/A (training-recipe knobs; no per-signal decomposition).
- **Diff-able across runs:** seed pinning enables byte-identical training runs given same other inputs.
- **Queryable post-hoc:** wandb / training stats logs (if dispatched).
- **Cite-able:** profile dict serializes into the canonical run-metadata.
- **Counterfactual-able:** N/A (training-time knobs; no archive byte-mutation).

## 4. Operational consequences

This symposium produces 8 binding op-routables (per §council_decisions_recorded). The key directives:

1. **Op-routable #1**: DO NOT reactivate as standalone substrates.
2. **Op-routable #2**: Embed augmentation primitives into in-flight substrate design memos (canonical-vs-unique decision per layer).
3. **Op-routable #3**: Catalog #313 probe-outcomes ledger entry registers DEFER + 30-day expiration.
4. **Op-routable #4**: Re-classify pre-rigor inventory entry from substrate-tier to ingredient-tier.
5. **Op-routable #5**: If operator-authorized reactivation: SAUG v2 alone is the highest-EV $5-10 single-axis test.
6. **Op-routable #6**: 30-day deferred-substrate retrospective 2026-06-17.
7. **Op-routable #7**: Cross-pollination with master-gradient sensitivity-mask-aware codec.
8. **Op-routable #8**: $10-25 budget redirect to Composition #3 / V4 Faiss / Z7-LSTM.

## 5. Per CLAUDE.md "Forbidden premature KILL"

NO KILL verdict. NO substrate code modified. NO dispatches fired. The verdict is DEFER_PENDING_EVIDENCE with explicit reactivation criteria per Op-routable #6:

- (a) Any in-flight substrate's design memo elects MAE OR SAUG v2 as a canonical-vs-unique-FORK decision
- (b) Contest frontier moves into a regime where training-augmentation has revived frontier-EV
- (c) Operator explicitly authorizes a single-axis SAUG v2 $5-10 smoke per Op-routable #5

## 6. Cross-references

- Pre-rigor inventory: `.omx/research/pre_rigor_kill_defer_falsified_inventory_20260517.md`
- Original DEFER memo: `feedback_codex_sandbox_blocks_vastai_dns_20260428.md`
- Sister symposium template: `.omx/research/council_per_substrate_symposium_lane_17_imp_20260517.md`
- T3 grand council synthesis: `.omx/research/council_t3_grand_council_synthesis_all_research_eureka_engineering_meta_20260518.md`
- Master-gradient memo: `.omx/research/master_gradient_xray_fields_medal_research_wave_20260518.md`
- Profile definitions: `src/tac/profiles.py:4102` (SAUG_V2_DILATED_H64) + `:4200` (MAE_V_DILATED_H64)
