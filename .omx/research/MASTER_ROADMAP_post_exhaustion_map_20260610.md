# MASTER ROADMAP — post-exhaustion-map (the aimed-retraining campaign era)

UTC 2026-06-10 · claude (`findings_audit_roadmap_20260610`) · operator directive
2026-06-10: *"audit all of the findings and results from the past couple days, then
evaluate the roadmap and next steps exploiting ALL information and code and everything
we have now."* Supersedes the framing of `MASTER_ROADMAP_v3_to_theoretical_floor_20260609.md`
(v1/v2/v3 PRESERVED, append-only per Catalog #110/#113); this is the post-frozen-bytes-exhaustion
re-prioritization. `[macOS-CPU advisory]` synthesis; no score claim, no dispatch, `promotable=false`.

Frontier (pointer, never hardcoded — `tools/refresh_canonical_frontier.py`):
**contest-CPU 0.19198275** (178,495 B, `lane_pr110pp_r3_candidate_cpu`, sha `1ccae18d…`) /
**contest-CUDA 0.20533003** (186,876 B, `lane_pr106_format0d_latent_score_table`, sha `9cb989ce…`).
Score law (frozen authority): `S = 100·d_seg + √(10·d_pose) + 25·B/N`, N=37,545,489.
RACE_MODE_ACTIVE.flag EXISTS (dated 2026-05-14; currency re-verify is op-routable #4 of the T4 council).

---

## 0. THE CONVERGENT STATE (what the past two days proved)

The 2026-06-09/10 wave attacked the 0.19199 frontier on **every frozen-bytes axis** and the
**alternate-vehicle axis**, and they converge on ONE conclusion:

> **The 0.19199 frontier is a near-globally-optimal MEMORIZED single-video renderer with NO
> redundant precision. Every frozen-bytes attack — selector, decoder, latent, sidecar, seg-repair —
> is DEFER-pending-research (Pareto-vertex saturated). The ONE remaining door is a FULL score-aware
> training campaign aimed by our measurement stack, OR a faithful HF-mechanism vehicle that escapes
> the spectral-bias mean-field. Both are training campaigns, not byte transforms.**

The byte budget that bounds everything (frontier member-x `[decoder 162,127 | latent 15,387 |
sidecar 879]`, +selector/dqs1 tail):
- **decoder = 90.9% of bytes, 99.98% of |grad|** — INT8 + brotli at 98.6% of iid Shannon. Coding
  exhausted; only lever is distortion; coarsening kills d_seg ~10× the rate gain (FALSIFIED-AT-IMPL ×2:
  naive uniform AND grid-LSQ retrain).
- **latent = 8.6% of bytes, 4.42% of score-weighted sensitivity** — LZMA at ~3.4% of iid floor;
  2nd-order re-prediction WORSE; sidecar pays rent ~7×. FALSIFIED-AT-IMPL.
- **sidecar/selector = 0.5% of bytes** — selector at entropy floor (recode = sha-identical no-op);
  per-pair pose lever EXHAUSTED (593/600 already argmin; max win −2.58e-6, below contest precision).
- **seg-repair pool = 0.056 score (29% of total), FULLY MAPPED** (66,039 flips, 91% at margin<0.5),
  but a frame-1 correction sidecar is INFORMATION-THEORETICALLY incapable of clearing THE LAW
  (1.525 B/flip position-only floor > 1.27 B/flip break-even).

**The deep finding (system intelligence):** on a memorized renderer, the only carrier whose cost is
NOT proportional to error-count is the decoder/latent axis paid in EXISTING bytes — i.e. **a better
reconstruction fixes flips for free**. That is a *training* lever, not a *byte-transform* lever. The
roadmap therefore pivots from "shave the frozen frontier" to "re-synthesize a cheaper/sharper carrier
aimed by the measurement stack we just built."

### The aiming surfaces we now own (all landed, all reusable)
1. **Flip map** (`/Volumes/VertigoDataTier/pact/frontier_seg_repair_pool_*/flip_map_full/`): 66,039
   frontier seg flips, 91% margin<0.5, road/horizon band rows 171–292, the EXACT seg-axis sensitivity.
2. **Per-tensor + per-pair/per-dim sensitivity maps** (`per_tensor_sensitivity_map.json`,
   `latent_sensitivity_map.json`): decoder 99.98% of |grad|; latent freest pairs/dims ranked.
3. **Evaluator invisibility basis** (`tac.null_space_exploiter` + `evaluator_invisibility_basis`):
   CERTIFIED 22.7% zero-weight + 80.67% full resize-null per channel, residual==0.0.
4. **Resize-null preimage compiler** (`tac.optimization.resize_null_preimage`): frees 10.3% (source)
   to 19.5% (rendered) of coded frame bytes at certified zero distortion — for FRAME-storing carriers.
5. **Frame-1 joint-safe cone** (`frame1_joint_safe_cone`) + **Class-2/3 atom generators**: per-pixel
   seg-safe-pose budget (open cone) + seg-repair direction (margin-normal); pose binds 73% (49% usable).
6. **Spectral atlas v2** (`scorer_spectral_sensitivity_v2`): measured scorer-sensitivity peak at
   **w_equiv≈294** (luma, vertical, low-band) — the DERIVED learned-ω replacement for the w=30 trap.
7. **B2 gradient atlas**: pose Y-luma fraction 0.964, near-full-rank INPUT sensitivity (dim_95pct=243/288).
8. **Composition algebra coherence law**: compose in DISTORTION domain (never score-deltas); pose
   concavity super-additive, seg receptive-field-coupled; cone/margin/bytes as debited ledgers.
9. **THE LAW waterfiller** (`lf_payload_rate_distortion` #46) + **ActionEffect IR** (`action_effect` #36)
   + **evaluator_action_waterfill** admission — the one currency every candidate flows through.

---

## 1. THE EXPLOITATION ROADMAP — ranked candidate moves

Ranking key: `EV = |predicted ΔS lower bound (with derivation)| / (cost × time × risk)`, with a
hard gate on FALSIFIABILITY and CLAUDE.md campaign discipline (lane / timing smoke / cost model /
byte-closed export plan / stop-continue thresholds). All ΔS predictions are THEORETICAL ESTIMATES from
the score law + the landed measurement stack; the exact paired-axis eval is the only authority.

### RANK 1 — THE AIMED SCORE-AWARE RETRAINING CAMPAIGN (prompt candidate (a))
**`lane_aimed_score_aware_retrain_20260610`** — free-weight, score-aware, PR95 8-stage curriculum base,
aimed by surfaces (1)(2)(5)(6)(7), targeting a SMALLER decoder that re-memorizes the video at fewer bytes.

- **Mechanism / why it's the only structural door.** The decoder verdict's reactivation #3 + the latent
  verdict's reactivation #1/#2 + the seg-repair verdict's reactivation all name the SAME thing: the
  frontier is rate-bound (62%, 0.119 of 0.192) and the decoder bytes ARE the rate floor. The frozen
  decoder cannot be coarsened (no redundant precision), but a FRESH decoder trained score-aware at a
  smaller channel width can re-reach the same evaluator cells at fewer bytes — paying the d_seg/d_pose
  in the training loss, not a sidecar. This is the "better reconstruction fixes flips for free" lever
  the seg-repair memo proved is the ONLY non-error-count-proportional carrier.
- **The aiming (what makes this NOT a blind PR95 rerun).** Wire the landed surfaces into the loss:
  (i) **margin-weighted seg loss** `w ∝ exp(−margin/τ)` from the flip map / spectral atlas (concentrate
  capacity on the 91% margin<0.5 boundary pixels — the 29% seg pool); (ii) **Y-dominant pose loss**
  (B2: 0.964 luma) on the `both_opposite` inter-frame incidence (the pose signal is inter-frame, Y, low-band);
  (iii) **learned per-scale ω initialized at w_equiv≈294** (spectral atlas) replacing the w=30 spectral-bias
  trap; (iv) **PR101 L27 correction sidecar** as the rent-paying fine-tune (latent verdict: pays 7×);
  (v) **resize-null preimage postprocess on the emitted frames** (#49) to drop the 10–19% scorer-invisible
  coded bytes — but ONLY if the carrier stores frames/residuals (HNeRV stores weights, so this applies to
  any frame-pixel sidecar, not the decoder blob).
- **The smaller-arch hypothesis (the rate lever).** PR95/our-frontier decoder is ~229K params → 162 KB.
  The MASTER_ROADMAP_v3 floor analysis: with d_seg≈small + d_pose≈ε, S≈rate, and aggressive entropy
  coding (L20–L32) on a SMALLER decoder targets B~100–150 KB → S~0.15–0.17. The smaller arch is the
  rate bet; the aiming surfaces are the d_seg/d_pose protection that lets the smaller arch not collapse.
- **Predicted ΔS band (with derivation).** Two regimes:
  - *Conservative (same arch, better aiming):* the frontier d_seg=5.6e-4, d_pose=2.9e-5 are already low;
    aiming can at best halve the seg pool (0.056→0.028) IF the smaller arch holds rate. But same-arch =
    same bytes, so the gain is pure distortion: ΔS ≈ −0.028 (seg) + ε ≈ **−0.02 to −0.03** → S~0.16–0.17.
    This is the realistic, derivable band.
  - *Aggressive (smaller arch + L20–L32):* if a 100–150 KB decoder re-reaches evaluator cells, rate
    drops 0.119→0.067–0.10, ΔS ≈ **−0.02 to −0.05** → S~0.14–0.17. Compounds with the seg aiming.
  - **Honest framing per the T4 council:** this is frontier_PROTECTING/incremental until an exact
    sub-0.192 lands. The breakthrough (sub-0.15) requires the smaller arch to NOT collapse — unproven.
- **Cost / time.** The descent-proof smoke (16-pair, ~300ep, MLX-local) = $0, ~30 min. The staged full
  run (8-stage curriculum, real PR95-class config) = a multi-hour MLX-local run + ONE paired
  CPU+CUDA exact eval (~$0.6) per checkpoint of interest. Total campaign: $2–5, 1–3 days wall.
- **Dependencies / reused code (NAMED).** `experiments/train_substrate_*` PR95-family trainer +
  `tac.differentiable_eval_roundtrip` (mandatory) + the F1 `bilinear_skip_residual_canonical` +
  `terminal_hf_refine_canonical` kernels + the flip map + `per_tensor_sensitivity_map.json` +
  `scorer_spectral_sensitivity_v2` (w_equiv) + B2 atlas (Y-fraction) + `pr101_split_brotli_codec`
  (archive grammar) + `resize_null_preimage` (frame-pixel sidecar postprocess) + the THE LAW waterfiller.
- **FALSIFIABLE KILL CRITERIA (pre-registered, gate the spend).**
  1. **Descent-proof smoke gate (BLOCKING, $0):** train aimed-HiNeRV 16-pair/~300ep, exact eval. If
     d_seg does NOT DESCEND below the B1-R2 flat 0.50 (i.e. the renderer still mean-fields), the
     aiming did not fix the NON-LEARNING — escalate to RANK 2 (faithful HF vehicle) or RANK 5
     (the binding constraint is architecture, not objective). **This is the B1-R2 lesson encoded.**
  2. **Same-arch sub-floor gate:** if the aimed same-arch run cannot drive exact S below the frontier
     0.19198 on a paired CPU eval, the aiming surfaces add no exact-axis value over the incumbent FECa
     selection — re-route to RANK 3 (CUDA axis) or RANK 6 (pose-output-entropy breakthrough).
  3. **Smaller-arch collapse gate:** if the 100–150 KB decoder's d_seg rises faster than the rate
     drops (the decoder-verdict knife-edge re-manifests), the smaller arch is FALSIFIED — the rate
     lever requires the evaluator-inverse direct grammar (Phase 3), not a smaller neural decoder.
- **Council status:** the T4 grand council (`feedback_grand_council_symposium_all_results_roadmap_20260609.md`)
  ALREADY ratified this as PROCEED_WITH_REVISIONS with the descent-proof smoke gate + "fix BOTH B1 bugs
  (RGB/Y anchor + real PR95 config)." This roadmap promotes it to RANK 1 with the full aiming wire-in.

### RANK 2 — FAITHFUL HF-MECHANISM VEHICLE: patch `pact_nerv_vq` with the F1 bilinear-skip (prompt candidate, alternate-vehicle)
**`lane_pact_nerv_vq_pr95_hf_patch_20260610`** — the cheapest path to ONE byte-closed evaluator-cell
carrier that escapes the spectral-bias mean-field (per `reference_carrier_comparison_20260609.md`).

- **Mechanism.** Our three vehicles (hi_nerv/snerv/pact_nerv_vq) all share Shared Mistake A (skip-free
  PixelShuffle+sin30 decoder, no HF path → mean-field → d_seg≈0.5). `pact_nerv_vq` ALONE already has
  the complete byte-closed archive→inflate path AND the correct frozen-scorer/no-recon-base objective —
  the ONLY missing column is the ~15-LOC F1 bilinear-skip+refine HF kernel (already landed, shared).
- **Why it's RANK 2 not RANK 1.** It is a parallel BET on the alternate vehicle reaching parity; RANK 1
  improves the PROVEN frontier lineage. If RANK 1's descent-proof smoke FAILS (architecture is the bug),
  RANK 2 becomes RANK 1 (a clean HF vehicle sidesteps whatever harness bug the frontier lineage carries).
- **Predicted ΔS band.** N/A as a frontier delta YET — this is a vehicle reaching its FIRST byte-closed
  exact score (target d_seg < ~0.2, then drive toward ~0.19 with the curriculum). It is a CEILING bet,
  not an incremental delta. Predicted: a working carrier in the 0.2–0.5 range that DESCENDS (the proof
  the frontier lineage's mean-field is escapable), then the RANK 1 aiming + L20–L32 apply to it too.
- **Cost / time.** F1 kernel wire-in ~15–30 LOC + parity test = $0, hours. Descent smoke + first exact
  eval = $0.6, 1 day. Behind RANK 1 in priority but PARALLEL-DISPATCHABLE (different vehicle, no conflict).
- **Dependencies / reused code.** `tac.framework_agnostic.canonical_kernels.{bilinear_skip_residual_canonical,
  terminal_hf_refine_canonical}` + `pact_nerv_vq.{architecture,archive,inflate,score_aware_loss}` (all
  landed) + the F1 recon-fit ablation result (gates RANK 1 vs RANK 2 per the comparison memo Pre-step 0).
- **FALSIFIABLE KILL CRITERIA.** Per the comparison memo §4 gap-closure: if skip-ON + correct-objective
  keeps d_seg ~0.5, H1 is falsified for this carrier → the binding constraint is the missing coordinate
  PE (add Fourier-feature PE input, the 2×2 {skip}×{PE} ablation) → if THAT stalls, vendor a clean
  PR95-HNeRV MLX port (the safest-fidelity fallback, the comparison memo's RANK 2).

### RANK 3 — THE CUDA AXIS (prompt candidate (c)) — paired-axis discipline
**`lane_cuda_axis_frozen_pool_20260610`** — we have ONLY attacked contest-CPU; the CUDA frontier is
0.20533 (186,876 B, pr106), a DIFFERENT, WORSE archive than the CPU frontier (0.19198, 178,495 B).

- **Mechanism / the open question.** The CPU and CUDA frontiers are DIFFERENT archives. The CPU
  frontier (0.19198) has NEVER been evaluated on CUDA; the CUDA frontier (0.20533) is a stale pr106
  archive. **Is the CPU frontier archive ALSO better on CUDA?** If the 0.19198 archive scores < 0.20533
  on CUDA, the CUDA frontier is a free promotion (no new bytes). And: is there a cheaper pool on the
  CUDA axis that the CPU-only attacks missed (CUDA−CPU gap is empirical per-archive, PR102 showed +0.033)?
- **Predicted ΔS band.** CUDA-frontier improvement up to **−0.013** (0.20533→0.19198 IF the CPU archive
  transfers) from a single eval, $0 bytes. This is the cheapest possible win in the whole roadmap (one
  paired eval, no training, no byte transform) — but it improves the CUDA axis, not the CPU leaderboard
  axis (which is the ranking axis). Derivation: the CPU archive is 8,381 B smaller AND scores 0.013 lower
  on CPU; if the CUDA−CPU gap is similar magnitude on both archives, the CPU archive dominates on CUDA too.
- **Cost / time.** ONE CUDA (T4-equivalent) exact eval of the 0.19198 CPU archive = ~$0.3, hours.
  Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA," this is REQUIRED before ANY submission
  regardless — so it is a no-regret prerequisite, not a speculative spend.
- **Dependencies / reused code.** `experiments/contest_auth_eval.py --device cuda` + the modal CUDA
  dispatch (`modal_auth_eval` CPU+CUDA path, fixed `61f6dfd74`) + `tools/plan_dual_device_auth_eval.py`.
- **FALSIFIABLE KILL CRITERIA.** If the 0.19198 CPU archive scores WORSE on CUDA than the pr106 0.20533,
  the axes genuinely diverge and the CUDA frontier needs its own attack (the frozen-axis verdicts were
  CPU-only; a CUDA-axis seg/pose pool map would be the reactivation). If it scores BETTER, update the
  CUDA pointer (free win) and the CUDA axis inherits the same Pareto-vertex exhaustion as CPU.

### RANK 4 — RICHER FRAME-0 / FRAME-1 VOCABULARY beyond the 16 modes (prompt candidate (d))
**`lane_richer_perpair_vocabulary_20260610`** — R3 exhausted the EXISTING K16 frame-0 menu only; the
per-pair selector lever is "lever-exhausted" w.r.t. THAT vocabulary, not all vocabularies.

- **Mechanism.** R3's verdict: the per-pair selector is saturated for the K16 frame-0 menu (593/600
  already argmin). But the Class-2/3 frame-1 atom generators (landed) are a RICHER vocabulary: frame-1
  seg-safe pose atoms (open cone) + frame-1 seg-repair atoms (margin-normal). The Class-3 headline
  accepted 14/26 atoms at net-negative LAW ΔS (advisory). These were NEVER ratified on-host (R1 lesson:
  off-host ranking mis-ranks; the on-host R3-style run with noise-floor tie law is the admission authority).
- **Predicted ΔS band.** SMALL. The seg-repair verdict proved a frame-1 SIDECAR cannot clear THE LAW
  at pool scale (1.525 B/flip floor). The 14 accepted Class-3 atoms are advisory and the composition
  algebra warns the shared-carrier amortization is defeated by receptive-field coupling (composed
  worsening measured). Realistic band: **−1e-5 to +1e-3** — likely below contest precision like R3's
  −2.58e-6. This is a "confirm the bound on-host" move, not a frontier breaker.
- **Cost / time.** The Class-3 host-ranking packet is ALREADY BUILT (14 candidates). One R3-style
  on-host ranking run = ~$0.3. But the seg-repair information-theoretic floor makes this LOW-EV.
- **FALSIFIABLE KILL CRITERIA.** If the on-host Class-3 ratification confirms net-positive ΔS (the
  sidecar-floor bound), the per-pair frame-1 vocabulary is DEFER-confirmed and the only remaining
  per-pair lever is a vocabulary whose errors are BLOCKY (position entropy < 1 B/flip) — which the
  flip map shows the frontier does NOT have (1.1 flips/4×4 block, scattered). Routes to RANK 1.

### RANK 5 — THE EVALUATOR-INVERSE DIRECT GRAMMAR (the floor path, V3 Phase 3)
**`lane_evaluator_inverse_direct_grammar_20260610`** — replace/augment the neural decoder with the
DIRECT skeleton (entropy-coded seg-argmax maps + pose trajectory + minimal RGB carrier).

- **Mechanism.** The MASTER_ROADMAP_v3 floor analysis: the theoretical floor (~0.02–0.05) is reachable
  ONLY by dropping the neural decoder for the direct grammar — the decoder weights ARE the rate floor.
  The invisibility basis + resize-null preimage + read-surface atoms (`scorer_read_surface_atoms.py`)
  are the primitives. The seg rate floor (full argmax storage) = 424,722 B → 0.283 (WORSE than frontier),
  so naive skeleton storage LOSES — it needs amortization (a tiny decoder that GENERATES the skeleton).
- **Predicted ΔS band.** Theoretical −0.10 to −0.16 (S~0.03–0.08) IF the direct grammar amortizes. But
  this is the HIGHEST-RISK, longest-path move: no vehicle has produced a descending exact score yet, and
  the B0.5 codec budget proved naive storage loses. Band is SPECULATIVE until a working amortizer exists.
- **Cost / time.** WEEKS (a new vehicle class). Per the T4 council + the SEAL discipline: the live loop
  waits for a working Phase-1 base (RANK 1 or RANK 2). The read-surface atoms feed Phase 4 NOW as
  primitives, not the live loop.
- **FALSIFIABLE KILL CRITERIA.** If a minimal direct-grammar prototype's amortized bytes (decoder +
  skeleton sidecar) exceed the 178 KB frontier at equal d_seg/d_pose, the direct grammar is dominated
  by the neural decoder at THIS video's entropy — DEFER to a smaller neural decoder (RANK 1's smaller arch).

### RANK 6 — THE POSE-OUTPUT-ENTROPY BREAKTHROUGH PROBE (the Assumption-Adversary's open lever)
**`lane_pose_output_entropy_probe_20260610`** — $0 measurement: is a cheaper-than-HNeRV pose carrier possible?

- **Mechanism.** The T4 council's Assumption-Adversary flagged that B2 measured INPUT-sensitivity rank
  (near-full-rank), NOT carrier COST. The pose OUTPUT is 6 smooth numbers/pair (600×6, a low-entropy
  ego-motion trajectory). Near-full-rank input-sensitivity means the inverse problem is WELL-CONDITIONED
  (easy to hit a target pose) — which HELPS a cheap carrier. The unexploited question: what is the
  ENTROPY of the 600×6 pose trajectory, and can a tiny pose-trajectory codec + a seg-only renderer beat
  the dense HNeRV's 178 KB?
- **Predicted ΔS band.** This is a PROBE, not a candidate — it RESOLVES whether RANK 1's smaller-arch
  bet or a RANK 5 pose-carrier split is the right rate lever. If pose trajectory entropy is ~1–3 KB
  (smooth driving), it confirms the pose term is nearly free and the decoder bytes are ALL seg-carrier —
  redirecting RANK 1's capacity allocation. ΔS impact is INDIRECT (it aims the other ranks).
- **Cost / time.** $0, hours (measure the 600×6 pose-output trajectory entropy + the inverse-conditioning
  number from the B2 JtJ spectrum we already have). It is the T4 council's op-routable #3.
- **FALSIFIABLE KILL CRITERIA.** If the pose trajectory entropy is HIGH (not a smooth low-dim manifold),
  the dense-Y-carrier is HARD-EARNED (not cargo-culted) and RANK 1 proceeds with the dense decoder. If
  LOW, a pose-carrier split becomes a RANK 5 sub-lane.

---

## 2. THE RANKED EXECUTION ORDER (race-mode actionable, parallel-dispatch first)

Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" (RACE_MODE_ACTIVE.flag EXISTS):

1. **RANK 3 CUDA-axis eval FIRST** ($0.3, hours, no-regret) — evaluate the 0.19198 CPU archive on CUDA.
   Free CUDA-frontier promotion if it transfers; required pre-submission gate regardless.
2. **RANK 6 pose-output-entropy probe** ($0, hours) — aims RANK 1's capacity allocation. PARALLEL with #1.
3. **RANK 1 descent-proof smoke** (BLOCKING gate, $0, ~30 min) — aimed-HiNeRV 16-pair/300ep. The gate
   that decides RANK 1 vs RANK 2. PARALLEL with #1/#2.
4. **RANK 2 F1 HF patch** (parallel vehicle, $0 wire-in + $0.6 first exact) — gated on the F1 recon-fit
   ablation; the fallback if RANK 1's smoke fails.
5. **RANK 1 staged full campaign** (IFF the smoke descends) — the multi-hour aimed score-aware run +
   paired CPU+CUDA eval. The primary frontier bet.
6. **RANK 4 Class-3 on-host ratification** (LOW-EV, $0.3) — confirm the sidecar-floor bound; deprioritized.
7. **RANK 5 direct-grammar** — research primitives feed Phase 4; live loop waits for a Phase-1 base.

---

## 3. CROSS-CUTTING DISCIPLINE (unchanged, binding)

THE LAW (admit iff exact ΔS<0); compose in DISTORTION domain (composition algebra); dual CPU(Linux
x86_64)+CUDA(T4) authority (NEVER MPS/macOS-CPU for score); resize-null preimage on any frame-pixel
sidecar; the descent-proof smoke gate before any long run (the B1-R2 lesson); serializer +
--expected-content-sha256; SSD disk hygiene, no /tmp; NO FAKE; Forbidden premature KILL (every frozen-axis
verdict is DEFER, not KILL); submit only if exact dual-axis S beats the public frontier.

## 4. WIRE-IN (Catalog #125)
- Hook #1 sensitivity-map: the flip map + per-tensor/per-pair maps + spectral atlas ARE the aiming
  surfaces RANK 1 consumes.
- Hook #2 Pareto: every frozen-axis verdict confirmed the frontier on its Pareto vertex; RANK 1/2 are
  the only moves off it (re-synthesis, not byte transform).
- Hook #3 bit-allocator: RANK 1's margin-weighted-seg + Y-dominant-pose + w_equiv-ω is the allocator.
- Hook #4 cathedral-autopilot: RANK 3's CUDA eval + RANK 1's checkpoints are the dispatch surface.
- Hook #5 continual-learning: the 4 frozen-axis exact rows (R3 selector, 3 decoder, 1 latent) reseed
  the V3 ΔS-judge that frozen-bytes is exhausted; RANK 1's exact rows are the next anchors.
- Hook #6 probe-disambiguator: RANK 6 (pose-output-entropy) + RANK 1's descent smoke ARE the
  disambiguators (cheaper-carrier-possible? + objective-vs-architecture?).

## 5. Cross-refs (the audited evidence base)
`pr110pp_r3_onhost_selector_verdict_20260610.md` · `frontier_decoder_axis_waterfill_verdict_20260610.md` ·
`frontier_decoder_qat_recovery_verdict_20260610.md` · `frontier_latent_axis_waterfill_verdict_20260610.md` ·
`frontier_seg_repair_pool_verdict_20260610.md` · `snerv_branch_b_round2_verdict_20260610.md` ·
`reference_carrier_comparison_20260609.md` · `evaluator_invisibility_basis_landed_20260610.md` ·
`resize_null_preimage_compiler_landed_20260610.md` · `pr110pp_frame1_class23_generators_landed_20260610.md` ·
`composition_algebra_coherence_law_20260610.md` · `feedback_scorer_spectral_atlas_v2_*_20260609.md` ·
`feedback_grand_council_symposium_all_results_roadmap_20260609.md` (the T4 ratification) ·
`MASTER_ROADMAP_v3_to_theoretical_floor_20260609.md` (the floor analysis this re-prioritizes) ·
`evaluator_inverse_orphan_inventory_20260609.md` + `dedup_consolidation_audit_20260610.md` (the code map).
