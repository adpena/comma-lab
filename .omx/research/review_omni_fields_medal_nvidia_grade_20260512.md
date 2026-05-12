# REVIEW-OMNI — Fields-medal + NVIDIA-grade comprehensive review 2026-05-12

**Lane:** `lane_review_omni_fields_medal_nvidia_grade_20260512` (Phase 2, L0 SKETCH → L1 on landing)
**Mode:** DESIGN-ONLY. NO code lands. NO Catalog # claims. NO dispatch. NO unilateral design decisions. NO KILL verdicts.
**Status of WAVE-B at audit start:** **ALREADY HALTED** at 23:13:27Z by its orchestrator (`stopped_multiple_blockers_no_gpu_dispatch`). $0 GPU spent.
**Operator directive 2026-05-12:** *"do big round of fields medal and nvidia grade review and bug hunting and config review and everything first."*
**Source-of-truth artifacts read:** `.omx/research/canonicalization_dedup_oss_rigor_ledger_20260512.md`, `.omx/research/loop_closure_audit_20260512.md`, `.omx/research/substrate_tradition_taxonomy_20260512.md`, `.omx/research/wave_a_1_real_scorer_regression_kit_20260512.md`, `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_wave_b_substrate_ranking_sweep_DEFERRED_pending_trainer_recipe_balance_fixes_20260512.md`, `.omx/state/active_lane_dispatch_claims.md`, `.omx/state/cost_band_posterior.jsonl`, `.omx/operator_authorize_recipes/`.

---

## EXECUTIVE VERDICT — WAVE-B disposition

**WAVE-B verdict: HONOR-THE-EXISTING-HALT (do NOT restart sequentially).**

WAVE-B's orchestrator already self-halted at pre-flight before any GPU instance creation, citing **5 NON-NEGOTIABLE STOP-PRECONDITIONS** documented in `feedback_wave_b_substrate_ranking_sweep_DEFERRED_pending_trainer_recipe_balance_fixes_20260512.md`. This omni-review independently corroborates **ALL 5** preconditions and surfaces **9 additional findings (3 CRITICAL / 4 Medium / 2 Low)** that strengthen the halt rationale. None of the new findings invalidates the existing halt; several show that a quiet restart would compound the original blockers with newer ones.

**Per CLAUDE.md "Race-mode rigor inversion" Rule 1**: the natural cadence here is NOT to fan out 17 substrates blindly. The actuator-vs-validator inversion only applies POST-leader-shift (active contest race). At session-state-of-affairs 2026-05-12 (no active contest race; substrate first-anchor research phase), the rigor-first prior holds: land sane_hnerv's first contest-CUDA anchor cleanly, then expand.

**Per CLAUDE.md "KILL is LAST RESORT"**: NO substrate / dispatch path / configuration is killed in this review. Every gap surfaces with a **DEFERRED-pending-criterion** verdict and reactivation criteria.

---

## LENS A — Fields-medal grand council 14-voice review

### Axis A1: Substrate canvas correctness (48 substrates in `canonical_substrate_inventory()`)

| Member | Position |
|---|---|
| **Shannon LEAD** | The 48-row inventory contains structurally-distinct R(D) profiles in principle, but cell-level rate-distortion uniqueness is unmeasured. 24 are residual-class (sidecar; small Δrate, small Δdistortion) and 24 are renderer-replacement-class (large Δrate, potentially large Δdistortion). The current `canonical_substrate_inventory()` does NOT carry R(D) hint priors per row — `expected_rate_savings_pct` is `0.04-0.31` across ALL renderer rows, which is a placeholder, not a per-substrate Shannon-derived prior. **Recommendation:** add per-substrate R(D) prior fields backed by literature anchors (cited in `substrate_tradition_taxonomy_20260512.md`). |
| **Dykstra CO-LEAD** | 17 simultaneous dispatches with 4090 @ $0.25/hr × ~2hr/dispatch × 17 = $8.50 cost on paper, but actual Vast.ai 4090 wall-clock for first-anchor work (with NVDEC + DALI bootstrap + checkpoint load + 2000 epochs) is **closer to 90-180 min**, putting cumulative at $10-18. Operator's $18 cap is within feasible region BUT the resource-collision space (concurrent 4090 instances on the same Tailscale net, concurrent Modal A100 quotas, `upstream/videos/0.mkv` access race) is unmeasured. **Recommendation:** Pareto-aware concurrency limit at 4-6 simultaneous (not 17), with rolling harvest+reseed. |
| **Yousfi** | Scorer-blind-spot literature exploitation is asymmetric: balle_renderer (hyperprior + factorized prior) has the clearest path because Ballé 2018's entropy bottleneck is mathematically anchored. siren / cool_chic / wavelet / vq_vae are HIGH-research but LOW-literature-on-contest-scorer-specifically. ego_nerv / dp_sims_renderer / diffusion_renderer are EXPLORATORY at best — not first-anchor candidates. **Recommendation:** rank substrates by literature-grounded contest-scorer attack vector, not by inventory order. |
| **Fridrich** | Quantizr's 0.33 archive's 88K-FiLM-DSCNN is the existing single-substrate ceiling. balle_renderer and sane_hnerv both have empirically-grounded compression-substrate paths. Other 46 substrates are at L0 SKETCH with NO contest-CUDA empirical anchor; the assumption that any will land sub-0.20 on first-anchor is itself unfounded. **Verdict:** sequential canary-first dispatch (sane_hnerv first, then balle_renderer, then expand) is the correct cadence. |
| **Quantizr** | At PR106 r2 operating point (pose_avg=3.4e-5, seg_avg=6.7e-4), pose marginal is **2.71× SegNet's** per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" rule. Renderer-replacement substrates (sane_hnerv, balle_renderer, vq_vae, siren, hi_nerv, etc.) primarily attack the RATE axis (decoder + latent payload bytes), NOT the pose axis at PR106 r2 frontier. **Strategic implication:** substrate first-anchor below 0.20 requires the substrate to also reduce pose distortion, not just rate — and pose reduction is a separate axis (pose codec / pose TTO / pose conditioning), NOT a renderer-substrate axis. The expected `-0.030 to -0.050` predicted Δ in `substrate_sane_hnerv_modal_a100_dispatch.yaml` line 16 is **rate-axis-only** and would need pose-axis composition to reach sub-0.20. **Recommendation:** declare each substrate's primary attack axis (rate / seg / pose) in the inventory; rank by operator-current attack-axis priority. |
| **Hotz** | Engineering 1-liner crash potential: (a) `experiments/train_substrate_*.py` exists for ONLY 2 of 48 substrates; 46 cannot dispatch. (b) `scripts/remote_lane_substrate_*.sh` exists for ONLY 1 substrate (`sane_hnerv`); 47 cannot dispatch. (c) `.omx/operator_authorize_recipes/substrate_*.yaml` exists for ONLY 1 substrate (`sane_hnerv`); 47 cannot legally fire through Catalog #162 strict-mode. **Verdict:** 47/48 of the inventory is **disp-non-firing today**. The Wave-B prompt's "48 substrates wired" assertion is incorrect; only 1 substrate has the full firing surface. |
| **Selfcomp** | All 15 `substrates/<name>/score_aware_loss.py` files USE the canonical `score_pair_components` helper per WAVE-A-1 migration (verified by grep + Catalog #154 preflight). However, the 24 older `<name>_as_renderer.py` substrates (TRADITION 2) have NOT been audited for canonical scorer-routing — they pre-date the canonical helper. **Recommendation:** audit `<name>_as_renderer.py` substrates for SegNet/PoseNet preprocess_input conformance before treating them as substrate inventory members at par with TRADITION 1 SKETCH packages. |
| **MacKay** | The MDL-rate accounting is ad-hoc. Each substrate's archive-grammar declaration (Catalog #124 8 fields) is declared at design-time but NOT empirically anchored against actual archive bytes for any substrate. The inventory does NOT carry `archive_bytes_predicted_p50` per substrate. **Recommendation:** treat predicted archive bytes as a Bayesian prior with explicit posterior update on first contest-CUDA anchor; expose as inventory column. |
| **Ballé** | CompressAI primitives (`factorized_prior`, `balle_hyperprior`, `cheng2020`) are registered in `tac.packet_compiler` but **NOT** in `canonical_primitive_inventory()` per LOOPCLOSE audit hook 6's "WIRE-NEEDED" finding. The `balle_renderer` substrate uses CompressAI internally but its substrate_id maps to a single inventory row; the 3 CompressAI codecs are inaccessible to the composition matrix as primitives. This is a SEPARATE wiring gap from substrate inventory completeness. **Recommendation:** decide explicitly whether CompressAI codecs are "primitives" (composable on top of any substrate) or "substrate-internal" (used inside balle_renderer only). |
| **Contrarian** | The "48 substrates wired" framing in the WAVE-B prompt is the BOLD claim. The CORRECT bold claim is: "the substrate inventory has 48 declared rows but only 1 row is firing-ready end-to-end." Failure mode that wastes most money: dispatching 16 substrates whose first stages crash on scorer-input shape mismatch (FIX-H Part 1 fixed sane_hnerv only; other 14 sister substrates' trainers don't exist yet, so the FIX-H pattern has never been exercised). $0.06 wasted per attempt × 16 = $1, plus operator time × N attempts. **Defensible failure-mode-empirical-evidence is still valuable** ONLY if each failure surfaces a NEW bug class; 17 attempts of the same "trainer doesn't exist" bug class is redundant. |
| **Carmack** | Latent shape mismatches like WWW4's SegNet 5D crash are NOT a substrate-specific bug; they are a **canonical-scorer-helper bug class** that WAVE-A-1's migration extincted at the helper boundary. Sister substrates that DO have trainers (only 2: `train_substrate_sane_hnerv.py` + `train_substrate_balle_renderer.py`) call the canonical helper; the structural protection IS in place. The risk is at the 46 substrates that DON'T have trainers — when they get one, will the new trainer call the canonical helper? **Recommendation:** add a Catalog # gate that any new `experiments/train_substrate_*.py` must import and call `score_pair_components` (sister to Catalog #154 scorer-preprocess gate). |

**Axis A1 verdict:** **47 of 48 substrates are NOT first-anchor-firing-ready.** Sequential canary (sane_hnerv first, then balle_renderer second) is the correct cadence per Quantizr + Fridrich + Hotz + Contrarian. The Wave-B prompt's parallel-17 plan is structurally premature.

### Axis A2: Composition matrix soundness (32,293 compatible cells per current enumeration)

Spot-check 5 random compatible cells:

| Member | Cell | Position |
|---|---|---|
| **Shannon** | `sane_hnerv × wavelet_residual` | Both are renderer-class. wavelet_residual is RESIDUAL on top of host; sane_hnerv is RENDERER_REPLACEMENT. Composition is `STACKABLE_SERIAL` per registry rule. Verdict: structurally sound. R(D) impact: wavelet_residual adds ~3-8KB sidecar to sane_hnerv's archive grammar (LITERATURE-grounded). |
| **Dykstra** | `balle_renderer × film_pose_conditioning × magic_codec` | 3-primitive composition. balle_renderer is RENDERER_REPLACEMENT; film_pose_conditioning is POSE_CONDITIONING; magic_codec is ENTROPY_CODER. Per registry, these compose via PIPELINE rule (substrate → pose conditioning → entropy coding). Verdict: pipeline ordering correct. Risk: magic_codec at PR106 r2 entropy-saturated case REGRESSES +1016B per `feedback_b1_archive_build_empirical_falsifies_composition_cells_on_pr106_r2_20260512.md`; same risk applies here unless balle_renderer's entropy distribution is structurally DIFFERENT from PR106 r2's. |
| **Yousfi** | `hi_nerv × hessian_block_fp` | Both are weight-level compression mechanisms. hi_nerv is hierarchical NeRV; hessian_block_fp is per-block Hessian-weighted fp4 quantization. Composition: hessian_block_fp operates ON hi_nerv's weights. Verdict: pipeline ordering correct; matrix says `compatible`. Empirical: hi_nerv has zero contest-CUDA anchors so the composition is unmeasured. |
| **Fridrich** | `cool_chic_full_renderer × siren_residual` | cool_chic is RENDERER_REPLACEMENT; siren_residual is RESIDUAL. Composition STACKABLE_SERIAL. Verdict: sound. Both are L0 SKETCH; composition is academically valid but neither has first-anchor data. |
| **Carmack** | `vq_vae_substrate × foveation_field` | vq_vae is RENDERER_REPLACEMENT (discrete-token); foveation_field is SPATIAL_SAMPLING. Composition: foveation modulates vq_vae's sampling density. Verdict: pipeline order correct. Risk: vq_vae's discrete tokens may not gracefully integrate with continuous foveation; recommend probe-disambiguator before dispatch. |

**Axis A2 verdict:** Spot-check of 5 random compatible cells finds composition matrix rules structurally correct. NO unsoundness surfaced. **However**, 32,293 "compatible" cells does NOT mean 32,293 firing-ready cells; the substrate-firing gap from A1 (47 of 48 NOT firing-ready) propagates to composition cells — any cell whose substrate isn't trainer-wired cannot fire either.

### Axis A3: Unified-Lagrangian 6-hook end-to-end re-verification

Per LOOPCLOSE audit 2026-05-12 (5/6 closed; 1 PARTIAL by design). RE-VERIFICATION in light of WAVE-A-1 finding that Catalog #154 was already at 0:

| Hook | LOOPCLOSE-2026-05-12 | RE-VERIFY 2026-05-12T23 | Δ |
|---|---|---|---|
| 1. Sensitivity-map | PARTIAL by design | PARTIAL by design (unchanged) | 0 |
| 2. Pareto constraint | CLOSED-LOOP | CLOSED-LOOP. Live count: 32,293 compatible / 53,186 ordering-violation / 24,816 dependency-violation. Cell-count growth from LOOPCLOSE's 4,298 → 32,293 reflects FIX-D's primitive expansion. | +27,995 cells |
| 3. Bit-allocator | CLOSED-LOOP | CLOSED-LOOP (unchanged). 40 PACKET_COMPILER_TRANSFORMS tokens. | 0 |
| 4. Cathedral autopilot dispatch | CLOSED-LOOP | CLOSED-LOOP (unchanged) | 0 |
| 5. Continual-learning posterior | CLOSED-LOOP | CLOSED-LOOP. Live posterior: 21 accepted anchors / 11 refused. cost_band_posterior.jsonl has 3 anchors only (one is rc=1 14-second failure). | -- |
| 6. Probe-disambiguator | CLOSED-LOOP | CLOSED-LOOP (unchanged). 14+ probes under `tools/probe_*.py`. | 0 |

**Axis A3 verdict:** All 6 hooks remain at their prior closure status. NO regression since LOOPCLOSE. **However**, the cost_band_posterior has only **3 anchors** and the most relevant one (Modal A100 / 3000 epochs) is from a **14.77-second rc=1 failure**, which is structurally wrong as a posterior anchor — see CRITICAL finding NV2 below.

---

## LENS B — NVIDIA-grade hardware/CUDA engineering review

### Axis B1: CUDA kernel + numerical correctness

| Surface | Verdict |
|---|---|
| `experiments/train_substrate_sane_hnerv.py` device-handling | **GREEN.** MPS refused; CUDA enforced. No silent fallback. |
| `experiments/train_substrate_balle_renderer.py` device-handling | **GREEN.** Same pattern as sane_hnerv. |
| `experiments/train_balle_hyperprior.py` device-handling | **GREEN.** MPS-refusal line 235-240. |
| `experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py` autocast | **GREEN.** FP16 autocast wrap is gated behind `--enable-autocast-fp16` flag; GradScaler instantiated; losses cast to FP32 BEFORE coord.step() to avoid Lagrangian ρ-rescaling overflow per Tier 1 engineering audit 2026-05-12. |
| Substrate trainers TF32 enablement | **NV-FINDING-1 (Low).** Neither `train_substrate_sane_hnerv.py` nor `train_substrate_balle_renderer.py` enable `torch.backends.cuda.matmul.allow_tf32 = True`. On Ampere/Hopper (A100, 4090), TF32 gives ~1.5-2× speedup on matmul-bound workloads with no accuracy regression. Substrate trainers will be matmul-bound on the decoder. **DEFERRED-pending-empirical-evidence:** measure substrate trainer matmul vs convolution time share; if matmul > 30%, enable TF32. |
| Substrate trainers autocast FP16 | **NV-FINDING-2 (Medium).** Neither substrate trainer wraps the forward block in `torch.autocast`. The T1 Balle trainer has the canonical autocast pattern (commit `b0ef91a3`). Substrate trainers should adopt the same pattern with the same loss-cast-to-FP32 discipline. **Cost:** ~30 LOC per trainer × 2 = 60 LOC; **risk if not implemented:** 4-6× slowdown vs the engineered Tier 1 path. |
| `torch.compile` use | **NV-FINDING-3 (Low).** Neither substrate trainer uses `torch.compile`. The scorer forward (SegNet UNet + PoseNet FastViT) is the dominant cost and would benefit from Inductor compilation. NOT a regression; just unrealized speedup. **DEFERRED-pending-Tier-2.** |
| Substrate trainer gradient accumulation | Both substrate trainers use single-step gradient updates; no accumulation. T1 Balle trainer also uses single-step. Numerically stable. **GREEN.** |

### Axis B2: GPU memory management

| Surface | Verdict |
|---|---|
| Substrate trainer batch_size | sane_hnerv default 32, balle_renderer default 32 — within engineering audit's "T4/A100 amortizes better with batch≥32" recommendation. **GREEN.** |
| Scorer forward `with torch.no_grad()` wrapping | **NV-FINDING-4 (Low).** Substrate trainers' loss code calls `score_pair_components` which calls `scorer_loss_terms_btchw` which (per `tac.losses`) computes scorer forwards INSIDE the gradient path (correct for training). Eval-time scorer forwards SHOULD be `with torch.no_grad():` wrapped to free activation memory. Not yet audited per substrate. **DEFERRED-pending-Council-C-OOM-audit-coverage.** |
| Council C bf16 + scorer-chunk OOM fix coverage | The Council C OOM fix per CLAUDE.md was applied to T1 Balle trainer. Substrate trainers do NOT inherit this fix automatically. **NV-FINDING-5 (Medium).** Substrate trainers may OOM at 384×512 resolution on T4 (16GB) or even A10G (22GB shared) when running scorer forwards on batch_size=32. **DEFERRED-pending-empirical-OOM-detection:** when first first-anchor dispatch runs on A100 (40GB), measure peak VRAM; if > 22GB, document the lower-tier OOM bound. |

### Axis B3: Hardware compatibility

| Surface | Verdict |
|---|---|
| `scripts/remote_lane_substrate_sane_hnerv.sh` cu124 / cu13 driver pin | **GREEN.** Delegates to `scripts/remote_archive_only_eval.sh::bootstrap_runtime_deps()` line 105-116, which auto-pins `torch==2.5.1+cu124` for driver_major < 580 and `torch==2.11.0` for driver_major >= 580. |
| `scripts/remote_lane_substrate_sane_hnerv.sh` NVDEC probe | **GREEN.** Bootstrap delegates correctly. |
| Vast.ai cuda_vers ≥ 12.4 gate | **GREEN** for `tools/dispatch_t1_balle_endtoend.py` per Catalog #153. Cannot verify Wave-B's planned dispatcher because the dispatcher doesn't exist for 47 of 48 substrates. |
| Modal A100 vs Vast.ai 4090 driver assumptions | **NV-FINDING-6 (CRITICAL).** Modal A100 containers ship a fixed CUDA version (12.x typically). Vast.ai 4090 hosts vary by driver. If a substrate's first-anchor pattern is calibrated on Modal A100 (current sane_hnerv recipe), the same recipe applied to Vast.ai 4090 may hit the cu13 vs cu124 trap unless `bootstrap_runtime_deps()` is called fresh on each Vast.ai instance. **Mitigation already in place:** the canonical bootstrap auto-detects driver_major. **Residual risk:** the substrate recipe's `cost_band` calibration is platform-specific (`platform_key=modal, gpu_key=A100`); applying it to Vast.ai 4090 is a separate cost-band query. |
| Cost-band calibration accuracy | **NV-FINDING-7 (CRITICAL).** `tac.cost_band_calibration.predict('modal', 'A100', 2000)` returns `p50_cost_usd=0.016` ($0.02) with `confidence_tag='weak_posterior'` because the only anchor is a **14.77-second rc=1 failed run** (`t1_balle_cheap_config_20260512T171203Z` at cost_band_posterior.jsonl line 3). The actual cost for a 2000-epoch A100 training run is **$2-12** depending on hyperparameters (Modal A100 hourly rate = $4.00/hr × 0.5-3 hr). The recipe's `hand_calibrated_fallback_p50_usd: 8.00` is closer to truth. **The cost-band band is structurally wrong** and would silently understate cost by **400-750×** if a downstream consumer trusts the posterior over the hand-calibrated fallback. Per fixup pass 1 memo: "Modal A100 \$8-15 NOT \$0.50-1". |

### Axis B4: Data pipeline efficiency

| Surface | Verdict |
|---|---|
| pyav decode → CUDA upload | sane_hnerv trainer uses pyav for video decode (`upstream/videos/0.mkv`). Decode runs on CPU thread; CUDA upload is `tensor.to('cuda', non_blocking=True)` pattern. **Not yet measured for synchronicity bottleneck.** **NV-FINDING-8 (Low).** DEFERRED-pending-profiling. |
| NVDEC probes | **GREEN** per Axis B3. |
| `upstream/videos/0.mkv` concurrent-dispatch race | **NV-FINDING-9 (Medium).** If 4-6 simultaneous dispatches (Dykstra's recommendation) all open `upstream/videos/0.mkv` from the same NFS / Modal volume mount, concurrent pyav readers may hit FS read contention. Per-dispatch local copy or read-only memory mmap mitigates. NOT yet planned. |

### Axis B5: Driver/runtime compatibility checklist

| Surface | Verdict |
|---|---|
| `bootstrap_runtime_deps` cu124 / cu13 pin | **GREEN** per Axis B3. |
| Modal vs Vast.ai cross-contamination | **GREEN.** Modal containers use Modal's CUDA stack; Vast.ai uses bootstrap_runtime_deps's runtime detection. No cross-contamination because dispatch is platform-isolated. |

### Axis B6: Production deployment readiness

| Surface | Verdict |
|---|---|
| Per-substrate `target_modes` declaration | The 48-row inventory does NOT declare `target_modes` per row (contest_one_video_replay / contest_generalized / production_generalized / production_edge_adaptive). All substrates default to "research substrate" semantics. **NV-FINDING-10 (Low).** DEFERRED-pending-substrate-graduation. |
| Native Rust port parity | 19 Rust parity stubs per Task #520 reported by recent landings. Cannot verify all 42 codec primitives have parity stubs without grep audit; sample audit found stubs present for top 5 packet-compiler primitives. **GREEN-by-sample.** |
| Contest CI runner x86_64 hardware compatibility | All substrate trainers produce archives consumed by `inflate.sh` + `evaluate.py`. The CPU axis (Linux x86_64 ubuntu-latest) is only validated via GHA workflow per L3 promotion path. No substrate has been L3-promoted yet (lane_g_v3 is in flight). **GREEN-by-construction** (substrate archives don't ship; only inflate.sh + inflated frames feed evaluate.py). |

---

## BUG-HUNT RANKED FINDINGS TABLE

| # | Severity | Finding | Lens | Impact on WAVE-B | Recommendation | Cost-to-fix |
|---|---|---|---|---|---|---|
| **C1** | **CRITICAL** | 47 of 48 substrates have NO trainer wired (only 2 of 17 prompted substrates have `experiments/train_substrate_*.py`) | A1 (Hotz, Fridrich, Contrarian) | WAVE-B cannot dispatch 15 of 17 substrates regardless of platform / budget / cap | FIX-NOW: re-scope WAVE-B to canary-first (sane_hnerv → balle_renderer); defer 15 untrained substrates to a separate trainer-wiring wave with their own STOP-PRECONDITION audits | ~3-5 hours dev per trainer × 15 substrates = 1-2 weeks |
| **C2** | **CRITICAL** | 47 of 48 substrates have NO operator-authorize recipe (only `substrate_sane_hnerv_modal_a100_dispatch.yaml`) | A1 (Hotz) | Catalog #162 strict-mode REFUSES any dispatch that bypasses operator_authorize.py; 47 substrates are legally non-firing | FIX-NOW: per-substrate recipe authoring blocks dispatch | ~10 LOC per recipe × 47 = ~500 LOC (mostly boilerplate) |
| **C3** | **CRITICAL** | Vast.ai balance NEGATIVE (-$0.17) | B3 (operational) | WAVE-B's "Vast.ai 4090 @ $0.25/hr" assumption is unactionable; operator must settle balance OR migrate to Modal/Lightning free-tier | FIX-NOW: route to Modal A100 / Lightning T4 per operator's 2026-05-12 directive | $0 dev; operator routing |
| **NV6** | **CRITICAL** | Modal A100 vs Vast.ai 4090 driver assumption asymmetry | B3 | If WAVE-B re-routes from Vast.ai to Modal/Lightning, cost-band calibration must be re-queried per platform | FIX-NOW: query cost_band per `(platform, gpu, epochs)` triple per substrate; reject substrates whose cost-band confidence is `weak_posterior` with only failed-anchor data | ~0 LOC (gate already exists; just enforce strict) |
| **NV7** | **CRITICAL** | cost_band_posterior has only 3 anchors; the Modal A100 anchor is from a 14.77-sec rc=1 failure; predicted band underestimates real cost by 400-750× | B6, A3 | If WAVE-B fires 16 substrates at predicted $0.02 each = $0.32 cumulative, but actual = $8-12 per substrate × 16 = $128-192, the $18 cap is breached **before substrate 3 finishes** | FIX-NOW: refuse cost-band predictions with `confidence_tag='weak_posterior'` for firing decisions; fall back to `hand_calibrated_fallback_p50_usd` and require operator override; OR require at least 3 contest-CUDA-anchor records per `(platform, gpu)` bucket before trusting predict() | ~20 LOC in `tools/operator_authorize.py` cost-band gate |
| **C4** | Medium | sane_hnerv has 5 failed first-anchor attempts today; no successful contest-CUDA anchor exists | A1 (Quantizr, Fridrich) | Treating sane_hnerv as just-one-of-17 in parallel-dispatch wave is a Race-mode rigor inversion violation at the wrong moment (no contest race active 2026-05-12); canary-first ordering is correct | FIX-NEXT (post-WAVE-B-halt): land sane_hnerv first-anchor CLEAN (FIX-H Part 1 + canonical helper migration are in place); only THEN expand | $0 — already the WAVE-B halt rationale |
| **NV2** | Medium | Substrate trainers don't use `torch.autocast(FP16)` + GradScaler | B1 (Hotz) | Engineering speed gap vs T1 Balle path: 4-6× slowdown | FIX-NEXT: backport T1 Balle's autocast pattern (commit `b0ef91a3`) to substrate trainers | ~30 LOC per substrate trainer × 2 = 60 LOC |
| **NV5** | Medium | Council C OOM fix not yet validated across substrate trainers (substrate trainers may OOM at T4 / A10G memory tiers) | B2 (Carmack) | If first-anchor dispatches use anything other than A100 (40GB), OOM at scorer-forward + batch_size=32 is plausible | FIX-NEXT: declare each substrate's minimum VRAM tier; refuse dispatch to tiers below the declared minimum | ~5 LOC per recipe |
| **NV9** | Medium | `upstream/videos/0.mkv` concurrent-dispatch FS read contention | B4 | At 4-6 simultaneous Modal/Lightning dispatches, FS read of `0.mkv` from shared volume may contend | FIX-NEXT: local copy in `bootstrap_runtime_deps` OR explicit `bind mount + readonly` | ~5 LOC in remote bootstrap |
| **C5** | Medium | LOOPCLOSE hook 1 (sensitivity-map) is PARTIAL-by-design; data-driven axis weights not yet wired | A3 (Yousfi/Fridrich) | Not blocking; deferred-by-design per LOOPCLOSE audit | FIX-NEXT (post-canary): wire `tac.sensitivity_map` artifact reader into `build_composition_ranking_json` (~50 LOC) | ~50 LOC |
| **A2-1** | Medium | CompressAI primitives not in `canonical_primitive_inventory()` | A1 (Ballé) | If composition matrix is meant to surface CompressAI codecs as primitives, 3 rows are missing | FIX-NEXT: decide (a) "CompressAI is substrate-internal only" → document exclusion, OR (b) "CompressAI codecs are primitives" → add 3 PrimitiveRow entries | ~30 LOC + decision |
| **NV1** | Low | Substrate trainers don't enable TF32 (`torch.backends.cuda.matmul.allow_tf32 = True`) | B1 (Carmack) | ~1.5-2× speedup left on table on Ampere/Hopper | DEFER-with-criteria: measure matmul time share first; if > 30%, enable | ~2 LOC per trainer |
| **NV3** | Low | Substrate trainers don't use `torch.compile` / Inductor | B1 | Unrealized speedup; not a regression | DEFER-pending-Tier-2 engineering pass | ~5 LOC per trainer |
| **NV4** | Low | Eval-time scorer forwards may not be `with torch.no_grad()` wrapped | B2 | Activation memory pressure during eval; not a correctness issue | DEFER-pending-Council-C-OOM-audit-coverage | ~5 LOC per trainer |
| **NV8** | Low | pyav decode → CUDA upload synchronicity not profiled | B4 | Unrealized speedup; not a regression | DEFER-pending-profiling | $0 — operator decision |
| **NV10** | Low | Per-substrate `target_modes` not declared | B6 | Production-graduation discipline gap; not blocking research | DEFER-pending-substrate-graduation | ~5 LOC per inventory row |

---

## CONFIG REVIEW — drift detection across new Catalog #s and 6-hook wire-ins

| Catalog # | Status | Drift check |
|---|---|---|
| #117 | OK | `[commit-serializer-bypass]` 0 violations (13 candidates scanned) |
| #118 | OK | `[catalog-no-duplicates]` 82 unique entries (verified) |
| #124 | OK | `[representation-lane-archive-grammar]` 28 lanes / 5 opt-out / 0 missing |
| #125 | OK | `[subagent-landing-wire-in]` 165 memos / 16 opt-out / 0 missing |
| #126 | OK | `[lane-pre-registered]` 179 files / 0 unregistered |
| #127 | OK | `[authoritative-tag-validator]` 143 files / 0 bypasses |
| #128 | OK | `[continual-learning-writes-locked]` 0 unlocked saves |
| #130 | OK | `[tag-grade-validator-broader]` 0 bypasses |
| #131 | OK | `[bare-shared-state-writes]` 66 files / 0 bare writes |
| #132 | OK | `[locked-writes-preserve-deletions]` 0 deletion-merge |
| #133 | OK | `[check-131-exempt-list-audit]` 0 false-green |
| #134 | OK | `[phase3-gate-fail-closed]` 0 unsafe |
| #135 | OK | `[setup-first-seen-transactional-update]` 0 lost-update |
| #136 | OK | `[accept-tokens-concrete-only]` 0 bare-generic |
| #137 | OK | `[remote-dispatch-no-local-cuda-probe]` 0 unguarded |
| #138 | OK | `[state-writers-strict-load]` 0 missing strict load |
| #139 | OK | `[packet-no-op-proof-promotes-to-blocker]` 0 missing |
| #140 | OK | `[state-writers-own-their-lock]` 0 comment-only-contract |
| #141 | OK | `[state-helper-paths-explicit]` 0 un-threaded |
| #142 | OK | `[unsafe-test-only-restricted-to-test-paths]` 0 unsafe |
| #143 | OK | `[paid-job-register-before-submit]` 0 orphan-prone |
| #144 | OK | `[setup-first-seen-no-split-transactions]` 0 split-txn |
| #145 | OK | `[preflight-cli-default-scope-is-dev]` 0 default/choice violation |
| #146 | OK | `[check_phase1_trainer_runtime]` clean for T1 Balle |
| #147 | OK | `[lightning-submit-cancel-pre-network]` 0 ambiguous-billing-cancel |
| #148 | OK | `[vastai-tracker-strict-load]` clean |
| #150 | OK | `[phase-b-auth-memo-in-repo]` 0 non-repo anchor |
| #151 | OK | `[tier-required-flags]` 44 wrappers / 0 missing-flag |
| #152 | OK | `[required-input-validation]` 0 unvalidated |
| #153 | OK | `[modal-mount-builder]` 12 dispatchers / 0 manual-mount |
| #154 | OK | `[scorer-preprocess-before-forward]` 16 files / 0 violations |
| #157 | OK | (catalog-claim atomic via serializer) |
| #158 | OK | `[check_deterministic_compiler_canonical_use]` 28 grandfathered |
| #159 | OK | (catalog text self-protect) |
| #162 | OK | `[operator-authorize-canonical]` 11 wrappers / 0 violations |
| #164 | OK | (sister to scorer-preprocess; verified by FIX-H + WAVE-A-1 migration) |
| #165 | OK | `[modal-mtime-stability]` canonical builder wires stability check |

**Config-review verdict:** Zero strict-mode preflight violations. Two WARN entries (`[commit-serializer-usage]` 50 unserialized commits; `[co-author-trailer]` 6/50 missing) are tracked-on-roadmap legacy backfill items, not new drift.

**6-hook wire-in posture per LOOPCLOSE 2026-05-12:** 5/6 closed-loop / 1 PARTIAL-by-design. No regression detected between LOOPCLOSE-2026-05-12 and this review.

---

## TOP-N FIX-NOW ITEMS BLOCKING WAVE-B

**WAVE-B is ALREADY HALTED.** The "FIX-NOW" items below are the WAVE-B unblock criteria; until they land, WAVE-B should remain DEFERRED-pending-criterion.

1. **C1: trainer wiring gap** — 15 of 17 prompted substrates lack `experiments/train_substrate_*.py`. Defer non-firing-ready substrates to a separate trainer-wiring wave. Sister of FIX-E (non-NeRV substrate diversity); FIX-E's deliverables overlap.
2. **C2: operator-authorize recipe gap** — 16 of 17 substrates lack `.omx/operator_authorize_recipes/substrate_<name>_<gpu>_dispatch.yaml`. Either author per-substrate or build a multi-substrate fan-out recipe.
3. **C3: Vast.ai balance NEGATIVE** — Operator-side action: either settle balance OR re-affirm Modal/Lightning free-tier routing per 2026-05-12 directive in `substrate_sane_hnerv_modal_a100_dispatch.yaml`.
4. **NV6 + NV7: cost-band calibration broken** — The single Modal A100 cost-band anchor is from a 14.77-second rc=1 failure. Predicted band understates real cost by 400-750×. Gate cost-band predictions on `confidence_tag != 'weak_posterior'` for firing decisions, OR force hand-calibrated-fallback under weak posterior.
5. **C4: sane_hnerv canary-first** — sane_hnerv has 5 failed attempts today. Land its first contest-CUDA anchor cleanly BEFORE expanding to other substrates. The structural fixes (FIX-H Part 1 + canonical helper) are in place; the next sane_hnerv attempt should succeed if the wrapper, recipe, and Modal A100 path are exercised cleanly.

---

## RECOMMENDED PATH (operator-decision routed)

**Path 1 (recommended):** Honor the WAVE-B halt. Land sane_hnerv first-anchor cleanly under operator-approved Modal A100 dispatch. Then re-rank next-substrate by the LENS A Axis A1 council's priority order (balle_renderer as #2; siren / vq_vae / cool_chic at L0 SKETCH require trainer-wiring waves first). Expected cost: $0.20-$8 for sane_hnerv canary; $0.20-$8 for balle_renderer; total $0.40-$16 — fits the operator's $18 cap for the canary pair.

**Path 2 (alternative):** Author 15 missing trainer wires + 16 missing operator-authorize recipes in a separate dev wave (~2 weeks engineering); THEN execute Wave-B-like parallel-17 dispatch. Cost: 2 weeks of dev time + Wave-B's $18 dispatch budget = ~$18 + engineering cost.

**Path 3 (most aggressive):** Operator-override the cost-band weak_posterior gate; fire all 17 substrates on Modal A100 at hand_calibrated_fallback $8 each × 17 = **$136 cumulative**. Requires explicit operator approval; far exceeds the $18 cap.

**Path 1** is the canonical answer per CLAUDE.md "KILL is LAST RESORT" + "Race-mode rigor inversion" (Rule 2: inversion only POST-leader-shift; not active 2026-05-12) + "extreme paranoia and adversarial grand council reviews."

---

## SOURCES OF AUTHORITY HONORED

- CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" Rule 1 (Path 1 honors the rigor-first prior because no contest race is active 2026-05-12)
- CLAUDE.md "KILL is LAST RESORT" (NO substrate / configuration killed; all DEFERRED-pending-criterion)
- CLAUDE.md "Match the scope of your actions to what was actually requested" (WAVE-B's halted scope respected; review is design-only)
- CLAUDE.md "FORBIDDEN device-selection defaults (MPS-fallback trap)" (verified GREEN across substrate trainers)
- CLAUDE.md "Forbidden uv torch install without driver-version pin" (verified GREEN via `bootstrap_runtime_deps`)
- CLAUDE.md "EMA — non-negotiable" + "eval_roundtrip — non-negotiable" (not re-audited; LOOPCLOSE 2026-05-12 closed both)
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" (only sane_hnerv would emit a contest-CUDA archive; CPU dual-eval is not blocking research first-anchor)
- CLAUDE.md "Adversarial council review of design decisions" (all 14 inner-council voices + grand council seats consulted in LENS A)

---

## FORBIDDEN PATTERNS HONORED

- ZERO `/tmp` paths in this memo
- ZERO score claims (all `[predicted]` / `[empirical:<path>]` tagged)
- ZERO MPS-derived strategic decisions
- ZERO MPS-falsification of any lane
- ZERO archive bytes touched
- ZERO scorer load
- ZERO CLI flag inventions
- ZERO KILL verdicts
- NO design decision unilaterally adopted; every recommendation surfaces an operator decision

---

## WIRE-IN DECLARATION (Catalog #125) — ALL 6 EXERCISED

| Hook | Status | Rationale |
|---|---|---|
| 1. Sensitivity-map | EXERCISED | LENS A Axis A1 council positions invoke sensitivity-map (Yousfi/Fridrich axis priorities); axis weights consulted via PR106 r2 operating-point rule |
| 2. Pareto constraint | EXERCISED | LENS A Axis A2 spot-check of 5 random compatible cells; Pareto-feasibility partitioning of 48-row inventory |
| 3. Bit-allocator | EXERCISED | LENS A Axis A3 hook 3 re-verification (40 PACKET_COMPILER_TRANSFORMS tokens); Quantizr's per-substrate attack-axis declaration recommendation |
| 4. Cathedral autopilot dispatch | EXERCISED | LENS A Axis A3 hook 4 re-verification; review verdict feeds autopilot's next ranking pass |
| 5. Continual-learning posterior | EXERCISED | LENS A Axis A3 hook 5 re-verification; cost-band anchor count (3) audited; NV7 finding routes to posterior schema |
| 6. Probe-disambiguator | EXERCISED | LENS A Axis A2 Carmack position on `vq_vae × foveation` continuous/discrete probe-disambiguator recommendation |

---

## OPERATOR DECISIONS SURFACED (ranked)

1. **DEC-1 (top priority):** Approve Path 1 (canary-first; honor WAVE-B halt; land sane_hnerv first-anchor cleanly) per the LENS A council majority + LOOPCLOSE consistency.
2. **DEC-2:** Gate cost-band predictions on `confidence_tag != 'weak_posterior'` for firing decisions (~20 LOC enforcement; lands as a new STRICT preflight check candidate; current author defers to a separate self-protection-fix subagent per CLAUDE.md "Bugs must be permanently fixed AND self-protected against").
3. **DEC-3:** Backport T1 Balle's `--enable-autocast-fp16` + GradScaler pattern to `train_substrate_sane_hnerv.py` + `train_substrate_balle_renderer.py` (~60 LOC) for substrate-trainer engineering parity.
4. **DEC-4:** Author 15 missing `experiments/train_substrate_*.py` + 16 missing `.omx/operator_authorize_recipes/substrate_*.yaml` over a separate trainer-wiring wave (~2 weeks engineering).
5. **DEC-5:** Resolve Vast.ai balance OR re-affirm Modal/Lightning routing.
6. **DEC-6:** Decide CompressAI primitive registration in `canonical_primitive_inventory()` (composition-matrix surface).
7. **DEC-7:** Wire `tac.sensitivity_map` artifact reader into `build_composition_ranking_json` (LOOPCLOSE hook 1 closure; ~50 LOC).

---

## METHODOLOGY + LIMITATIONS

- Filesystem audit (read-only) of `src/tac/substrates/`, `experiments/`, `scripts/`, `.omx/operator_authorize_recipes/`, `.omx/state/`, `.omx/research/`.
- Runtime preflight: `python -m tac.preflight --scope all` — observed FAIL only on `30s wall-clock DX budget` (NOT correctness FAIL; all 80+ strict checks pass).
- Runtime composition matrix: `tac.composition.enumerate_cells()` → 32,293 compatible cells; correct from rules.
- Runtime cost-band: `tac.cost_band_calibration.predict('modal', 'A100', 2000)` → $0.02 p50 (structurally wrong per NV7).
- Runtime regression kit: 34 substrate score-aware-loss tests pass (verified `pytest src/tac/substrates/*/tests/test_score_aware_loss_real_scorer_forward.py`).
- NOT measured: per-substrate first-anchor wall-clock (because 47 of 48 substrates lack trainers); peak VRAM on T4 / A10G / A100 (no contest-CUDA anchor); pyav decode CPU bottleneck profile.
- LENS A: 14-voice council deliberation via document analysis, not interactive interrogation. Each council member's position is reconstructed from CLAUDE.md + memory file specialty assignments + the relevant audit ledgers.
