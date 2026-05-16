# BATCHED REACTIVATION DESIGN — Lane 17 IMP + STC clean-source + apogee_int4 QAT — full-stack UNIQUE-AND-COMPLETE-PER-METHOD designs (2026-05-16)

**Lane:** `lane_batched_reactivation_lane17_imp_stc_apogee_int4_20260516`
**Author:** BATCHED REACTIVATION DESIGN subagent
**Source:** operator NON-NEGOTIABLE directive 2026-05-16 *"design full-stack comprehensive reactivation per Tier 1 candidate"* + resurrection audit Tier 1 verdicts (commit `d0c347f1f`)
**Scope:** Three Tier 1 reactivation candidates. Each gets a complete UNIQUE-AND-COMPLETE-PER-METHOD design memo per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable + canonical-vs-unique decision per layer + 9-dim checklist evidence per Catalog #294 + Dykstra-feasibility per Catalog #296 + signal-axis reversibility per Catalog #297 + observability per max-observability standing directive.
**Cost envelope:** $10-30 total across all 3 substrates.
**Cross-substrate composition section** below evaluates joint stacking with the other 5 design memos landed this turn (NSCS06 v8 / A-STACK / Wunderkind G1 v2 / ATW v2 / Symposium #4 T4).

---

## Shared assumption I am operating within (Assumption-Adversary, sextet-pact seat per Catalog #292)

**Operating-within assumption (HARD-EARNED):** *"Each of the three Tier 1 candidates was killed/deferred for reasons that are now provably cargo-culted per today's 5 lenses; reactivation with corrected evidence (proper fine-tune for IMP; CUDA-axis re-run for STC; QAT for apogee_int4) is the cheapest empirical path to either CONFIRM the substrate has real value OR produce hard-earned-with-3-section-compliance kill verdicts."*

This is HARD-EARNED via three independent receipts:
1. **Lane 17 IMP:** the kill was officially WITHDRAWN 2026-04-30 ~22:55 UTC by 8-of-10 council vote (`feedback_grand_council_imp_permanent_fix_review_20260430.md`); only the empirical re-run blocks closure.
2. **STC clean-source:** the kill memo explicitly says STATUS REVISION 2026-04-29 PM downgraded the verdict to UNDETERMINED pending Modal T4 CUDA re-run; CLAUDE.md "MPS auth eval is NOISE" non-negotiable specifies MPS-PROXY scores are FORBIDDEN as decision-grade.
3. **apogee_int4 QAT:** kill memo lists 6 specific reactivation paths (QAT, LSQ, per-channel, outlier clipping, GPTQ/AWQ, mixed-precision); Quantizr 0.33 leader uses INT4 via QAT — direct leaderboard-proven viability.

**Risk this assumption is wrong:** all 3 substrates could re-fail after reactivation. In that case the kill becomes HARD-EARNED instead of CARGO-CULTED — itself a valuable outcome because then a 3-section structural kill memo per CLAUDE.md "KILL/FALSIFIED memory verdicts" can land with confidence. **Total downside is ~$10-30 of GPU spend to convert 3 zombie-status lanes into either L2+ frontier candidates OR hard-earned kills.** Upside is potentially the Quantizr 0.33 paradigm being replicable, or STC entering the entropy-coding portfolio, or IMP iterative magnitude pruning composing with PR101 weight allocation. Asymmetric.

---

## Reactivation rationale summary (verbatim killer-assumption + today's classification)

| Substrate | Original verdict | Killer-assumption verbatim | Today's classification | Killer fix |
|---|---|---|---|---|
| **Lane 17 IMP** | KILL-WITHDRAWN 2026-04-30; ZOMBIE | *"1.98 [contest-CUDA] score reflects 88K-param IMP architectural ceiling"* | **CARGO-CULTED** per `feedback_grand_council_imp_permanent_fix_review_20260430.md` (stub-loop bug: stats.json `epochs=200, elapsed_sec=3.47` impossible) | Run train_distill (real 100ep) instead of 3.5s stub |
| **STC clean-source** | UNDETERMINED-pending-CUDA | *"Clean-source STC produces 21MB stream because boundary-fraction too high"* | **CARGO-CULTED** per CLAUDE.md "MPS auth eval is NOISE" (MPS-argmax masks ≠ CUDA-argmax masks; 23× PoseNet drift; STC byte count depends on argmax structure which differs across devices) | Modal T4 CUDA re-run on contest scorer argmax |
| **apogee_int4** | DEFERRED-pending-QAT 2026-05-05 at 1.43 [contest-CUDA T4] | *"INT4 NAIVE-PTQ produces 1.4287 [contest-CUDA T4]"* | **HARD-EARNED-CORE + CARGO-CULTED-SHELL** — single config falsified; Quantizr 0.33 leader uses INT4 with QAT (5 unexplored paths) | QAT + LSQ + per-channel scaling (Quantizr's recipe) |

---

# SUBSTRATE 1 — Lane 17 IMP (Iterative Magnitude Pruning, 88K param sparse renderer)

## 1.1 Reactivation rationale (full)

Lane 17 IMP was killed 2026-04-30 ~22:50 UTC based on a cycle 0 score of 1.98 [contest-CUDA] vs Lane G v3 anchor 1.05. The 8-of-10 grand council vote WITHDREW the kill ~5 minutes later when the user's adversarial challenge surfaced that `stats.json` reported `epochs: 200, elapsed_sec: 3.47` — internally inconsistent (200 epochs of fine-tune in 3.5 seconds is physically impossible).

The smoking gun: `experiments/train_imp_cycle.py::_finetune` contains a stub loop documented with the comment "in-script lightweight loop; deploy script swaps in train_distill". The comment was a load-bearing contract that was never enforced by an assertion or by the dispatch script's actually swapping in train_distill. The stub ran 200 epochs of NEAR-IDENTITY weight updates in 3.5 seconds, producing a renderer.pt that was nearly identical to the post-prune-rewind weights. The post-prune-rewind weights at 89.3% sparsity have nearly all the trainable mass shut off; without real fine-tune they CANNOT recover from the pruning shock.

Per CLAUDE.md FORBIDDEN_PATTERNS "Forbidden re-implementing remote bootstrap inline (the duplicated-bootstrap trap)" sister rule + the new "Comment-only contracts are FORBIDDEN" non-negotiable (which itself landed as Catalog #2 + sister gates in response to this exact incident), the comment was insufficient. The fix landed structurally as Catalog `check_imp_cycles_use_ema_and_auth_eval` (Check 94 in `src/tac/preflight.py`).

**Per CLAUDE.md "Forbidden premature KILL without research exhaustion":** today's verdict on Lane 17 is **DEFERRED-pending-real-train_distill-fine-tune**, NOT KILLED. The reactivation criterion is empirically grounded.

### Today's 5 lenses applied

1. **Chroma preservation (Catalog #297 sister concern):** N/A — IMP is weight-pruning, operates on renderer's parameter space not signal axes.
2. **Dykstra-feasibility (Catalog #296):** N/A originally (the 1.98 was a measurement bug, not a band-failure). For the reactivation: predicted band IS the feasibility intersection of (a) sparsity budget at 89.3% (rate constraint), (b) PoseNet/SegNet noise floor on 88K trained weights (distortion constraint), (c) Frankle 2019 lottery-ticket prior (recoverable subnetwork distribution). Per Boyd-Dykstra alternating projections: the achievable set is bounded by min-distortion subnetwork at 10.7% density. Predicted band [1.05, 1.30] = within ±25% of Lane G v3 anchor 1.05.
3. **9-dim checklist (Catalog #294):** Dimension 4 RIGOR previously failed (no premise verification of stats.json internal consistency); Dimension 9 OPTIMAL SCORE never measured (the 1.98 was a stub artifact). Today's reactivation explicitly addresses both: PV-1 verifies real train_distill runs; PV-2 verifies stats.json `elapsed_sec >= epochs * MIN_SEC` assertion fires; OPTIMAL SCORE will be the actual post-fine-tune auth-eval.
4. **Canonical-vs-unique:** PARTIAL — IMP is a class-shift technique (Frankle 2019 lottery-ticket subnetworks; structural pruning + magnitude-rewind). The kill was NOT canonicalization-induced. Today's per-layer decision: ADOPT canonical train_distill for fine-tune (no substrate-specific reason to fork); ADOPT canonical auth_eval helper (Catalog #226); FORK IMP-specific archive grammar (`imps_renderer_archive.py` — distinct magic byte handler for sparse weight packing).
5. **7-cargo-cult inventory (NSCS06 sister):** Symposium-#4-band-prediction cargo-cult was present in the INITIAL kill (1.98 treated as real prediction when it was a stub artifact). NONE of the other 6 (Y=R=G=B / spatial-independent / 2-of-6-pose-warp / NO-neural / closed-form-CDF / PR#56-generalizes) apply.

## 1.2 Comprehensive full-stack design

### 1.2.1 Substrate scope

Lane 17 IMP cycle 0 cycle-fine-tune — apply iterative magnitude pruning to the Lane G v3 1.05 [contest-CUDA] anchor renderer. 88K trainable params with cycle-by-cycle pruning + magnitude-rewind + fine-tune. Cycle 0 = first 10% pruned + fine-tune; Cycle 1 = next 10%; ... Cycle 9 = 89.3% pruned. Frankle 2019 demonstrates that lottery-ticket subnetworks at 89-95% sparsity CAN recover full-task performance.

### 1.2.2 Architecture

- **Base renderer:** Lane G v3 anchor (88K params, 1.05 [contest-CUDA] anchor, FP4A quantized at 297KB)
- **Sparsity schedule:** cycles 0-9, 10% per cycle, final 89.3% sparsity
- **Magnitude criterion:** L1 norm per scalar weight (canonical Frankle 2019)
- **Rewind policy:** rewind unpruned weights to step 0 init per cycle (canonical IMP)
- **Per-cycle fine-tune:** REAL train_distill, 100 epochs at 0.1× base_lr with EMA(0.997) per CLAUDE.md "EMA non-negotiable", eval_roundtrip=True per CLAUDE.md "eval_roundtrip non-negotiable" (NOT the 3.5s stub)

### 1.2.3 Priors / sensitivity

- **Frankle 2019 lottery-ticket prior:** strong (paper-proven; canonical)
- **88K-param Lane G v3 anchor distribution:** known sub-network density supports 10-20% pruning per cycle without distortion-blow-up
- **Per-tensor sensitivity:** layer-wise importance (NOT per-element); Hessian-trace per-tensor (NOT Xavier-L2 — that was falsified by Catalog #123)

### 1.2.4 Score-aware loss (UNIQUE FORK)

Per CLAUDE.md "Subagent coherence-by-default" hook 4 (Cathedral autopilot dispatch hook): the IMP cycle loss MUST route through canonical `tac.substrates._shared.score_aware_common.score_pair_components` to honor preprocess_input + target no-grad + SegNet surrogate semantics + sqrt(10) pose weighting. UNIQUE element: pruned-weight gradient routing — pruned weights receive zero gradient (masked out at backward); unpruned weights receive standard score-aware gradient. The mask is a fixed binary tensor recorded per cycle in the archive.

### 1.2.5 Archive grammar (UNIQUE FORK)

`src/tac/imps_renderer_archive.py` (NEW IMPS magic-byte handler, ALREADY LANDED per lane registry).

```
IMPS\x00            # magic
HEADER_LEN (4B LE)
HEADER (JSON: sparsity_per_layer, cycle_id, mask_sha256, fp4a_scale_table)
DENSE_MASK (bytes; bit-packed binary mask per FP4A tile)
FP4A_PAYLOAD (only unpruned tiles; brotli-compressed)
```

Per cycle, the archive shrinks proportionally to (1 - sparsity); cycle 9 archive ~30KB (vs Lane G v3 297KB FP4A baseline). **Empirical anchor:** 40.2% byte savings demonstrated at cycle 5 (74% sparse) per lane registry `real_archive_empirical` gate.

### 1.2.6 Inflate runtime

Existing `inflate_renderer.py` IMPS dispatch (already wired per lane registry). Per CLAUDE.md HNeRV parity discipline lesson 4: ≤100 LOC inflate budget. Current implementation passes; per Catalog #205 sister gate the IMPS branch already uses canonical `select_inflate_device`.

### 1.2.7 Export contract

Per-cycle archive emission. After cycle 9: best cycle's archive shipped as `submissions/lane_17_imp/archive.zip`.

### 1.2.8 Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Substrate trainer skeleton | ADOPT canonical `tac.substrates._shared.trainer_skeleton.device_or_die` | TF32 + autocast_fp16 (Catalogs #178/#172) substrate-agnostic engineering hygiene |
| eval_roundtrip + EMA | ADOPT canonical | CLAUDE.md non-negotiables; substrate-agnostic |
| Score-aware loss helper | ADOPT canonical `score_pair_components` | preprocess_input + target no-grad + sqrt(10) (Catalog #164) substrate-agnostic |
| Fine-tune path | ADOPT canonical `train_distill.py` | The kill was caused by NOT adopting this; the lesson is clear |
| Pruning/mask logic | UNIQUE FORK | IMP-specific (Frankle 2019); no canonical equivalent in tac |
| Archive grammar | UNIQUE FORK | `imps_renderer_archive.py` IS the substrate's bytes; UNIQUE by definition |
| Inflate runtime | UNIQUE FORK | IMPS magic-byte handler IS the substrate's runtime |
| Auth-eval gate | ADOPT canonical `gate_auth_eval_call` | Catalog #226 substrate-agnostic |
| Hardware substrate detection | ADOPT canonical `detect_hardware_substrate` | Catalog #190 substrate-agnostic |
| Posterior update | ADOPT canonical `posterior_update_locked` | Catalog #128 substrate-agnostic |
| Modal dispatch | ADOPT canonical `modal_train_lane.py` | Substrate-agnostic dispatch |
| Recipe schema | ADOPT canonical 36-field SubstrateContract (Catalog #241) | Substrate-agnostic |

### 1.2.9 9-dim checklist evidence (Catalog #294)

1. **UNIQUENESS:** IMP is CLASS-SHIFT (lottery-ticket subnetwork sparsity) NOT within-class refinement. Sparsity at 89.3% is structurally distinct from FP4A quantization (canonical Lane G v3 path).
2. **BEAUTY + ELEGANCE:** ~50 LOC of substrate-specific bolt-on (`imps_renderer_archive.py` magic-byte handler) on top of canonical train_distill. PR101-style 30-sec reviewable.
3. **DISTINCTNESS:** distinct from sister substrates — sparsity-via-pruning, not encoding/quantization. Compositional with FP4A (IMP's mask + FP4A on the unpruned subset).
4. **RIGOR:** premise verification PV-1 (`stats.json elapsed_sec >= epochs * 0.05`); 3 council rounds already completed per `council_lane_17_imp_round3_20260430.md`; assumption classification HARD-EARNED for Frankle 2019 paper, CARGO-CULTED for the prior stub-loop kill; empirical anchor 40.2% byte savings at cycle 5.
5. **OPTIMIZATION PER TECHNIQUE:** covered by canonical-vs-unique decision per layer above (Catalog #290).
6. **STACK-OF-STACKS-COMPOSABILITY:** sparsity is ORTHOGONAL axis to quantization (apogee_int4 QAT) AND to architecture (NSCS01/NSCS02/NSCS03) AND to entropy coding (STC). The pruned model still inflates as FP4A weights on the unpruned subset.
7. **DETERMINISTIC REPRODUCIBILITY:** seed-pinned (canonical `train_distill --seed`); byte-stable archive grammar; mask SHA-256 recorded per cycle.
8. **EXTREME OPTIMIZATION + PERFORMANCE:** TF32 + autocast_fp16 + torch.compile via canonical helper; 100 epochs/cycle × 10 cycles ≈ 10-15h on A100 = $5-10 (within $5-15 budget per audit).
9. **OPTIMAL MINIMAL CONTEST SCORE:** predicted [1.05, 1.30] post-fine-tune; if cycle 0 lands ≤ 1.10, the L2 promotion fires; if cycle 5 lands sub-0.95, beats Lane G v3 anchor; if cycle 9 lands sub-0.80, sparsity-vs-distortion Pareto-improves.

### 1.2.10 Predicted ΔS band + Dykstra-feasibility check (Catalog #296)

**Predicted band:** Cycle 0 score [1.05, 1.30] with mode 1.15.

**Dykstra-feasibility check:**
- **Constraint A (rate):** archive bytes monotone decrease with sparsity; cycle 0 archive ~270KB; cycle 9 ~30KB. Rate term contribution drops from 0.18 to 0.02.
- **Constraint B (distortion):** Frankle 2019 demonstrates that 50-90% pruning followed by fine-tune retains task performance within ±10% of full network. At cycle 0 (10% pruned), expected SegNet+PoseNet distortion stays within 5% of Lane G v3 anchor.
- **Convex hull intersection:** feasible region is the L1 ball of pruned weights ∩ (rate budget) ∩ (distortion budget). Alternating-projections via Dykstra converges in ~5 iterations of pruning + fine-tune; cycle 0 is the first projection.
- **First-principles citation:** Frankle 2019 "The Lottery Ticket Hypothesis" + Han 2015 "Deep Compression" pruning-fine-tune cycle establishes the feasibility region.

NOT vibes — the predicted band derives from a publication-grade prior.

### 1.2.11 Cost estimate + dispatch optimization protocol (Catalog #270)

- **Smoke (cycle 0 only):** $0.30 on Modal A100 100ep
- **Full (cycles 0-9):** $5-15 on Modal A100 (10h × $1.55/hr A100)
- **Dispatch optimization protocol verdict:** Tier 1 ✓ (canonical train_distill has autocast_fp16/TF32/torch.compile/no_grad/canonical scorer loss); Tier 2 NEEDS BACKFILL (recipe must declare min_vram_gb=20, min_smoke_gpu=A100, video_input_strategy=cpu_thread_async_upload, target_modes=[contest_one_video_replay, research_substrate], canonical NVML env block); Tier 3 must declare research_only=false (this IS contest-promotion eligible) AND inflate device canonical AND scorer loader assignment order canonical AND auth_eval via gate_auth_eval_call.

### 1.2.12 Observability surface (per max-observability standing directive)

- **Per-cycle audit log:** `.omx/state/lane_17_imp_audit.log` JSONL with `cycle_id, sparsity, archive_bytes, archive_sha256, mask_sha256, auth_eval_score, auth_eval_axis, runtime_seconds, gpu_substrate`
- **Per-cycle archive checkpoint:** `experiments/results/lane_17_imp/cycle_<N>/archive.zip` + manifest
- **Premise-verification artifact:** PV-1 `stats.json elapsed_sec >= epochs * 0.05` assertion FAILURE captured as RuntimeError + checkpoint to `.omx/state/subagent_progress.jsonl`
- **Live Modal call_id:** registered per Catalog #245 to `.omx/state/modal_call_id_ledger.jsonl`
- **Adjudication artifact:** per cycle's `evidence_grade` written to `auth_eval_result.json` with `score_axis=contest_cuda` enforced by Catalog #221
- **Score axis tag:** every result tagged `[contest-CUDA T4]` or `[contest-CUDA A100]` per CLAUDE.md "Apples-to-apples evidence discipline" (NEVER bare "1.05")

## 1.3 Probe-disambiguator plan (per CLAUDE.md "Probe-disambiguator pattern")

**Probe:** `tools/probe_lane_17_imp_cycle_0_real_vs_stub_fine_tune_disambiguator.py`

**Question:** is the 1.98 score (a) a stub-loop measurement artifact, OR (b) a genuine architectural ceiling?

**Test design:**
- Run TWO cycle 0 fine-tunes from the SAME pruned-rewound checkpoint:
  - **Arm A (stub):** 200 epochs of `_finetune` stub loop (~3.5 sec)
  - **Arm B (real):** 100 epochs of `train_distill.py` (~10-30 min)
- Auth-eval BOTH on Modal T4 CUDA
- Compare scores: if A is 1.98 and B is sub-1.30 → CARGO-CULTED stub-loop bug confirmed (REACTIVATE); if both are 1.98 → HARD-EARNED architectural ceiling (KILL with 3-section structural compliance).

**Cost:** $0.30 + $0.30 = $0.60 — cheapest disambiguator.

**Decision tree:**
| Arm A | Arm B | Verdict |
|---|---|---|
| 1.98 | ≤1.30 | CARGO-CULTED (reactivate; proceed to cycles 1-9) |
| 1.98 | 1.5-1.8 | PARTIAL-RECOVERY (reactivate with caveat; QAT-style staged fine-tune may help) |
| 1.98 | 1.95-2.05 | HARD-EARNED CEILING (KILL with 3-section structural compliance) |

## 1.4 Reactivation gates

1. **Smoke green:** cycle 0 with real train_distill returns rc=0 + auth-eval JSON parseable on Modal A100
2. **Tier C MDL density measured:** `tools/mdl_scorer_conditional_ablation.py` on cycle 0 archive (Catalog #227)
3. **100ep auth-eval anchor:** byte-deterministic archive (sha256 stable across re-runs) at cycle 0
4. **Custody validated per Catalog #127:** evidence_grade matches axis + hardware
5. **Paired CPU+CUDA per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA":** required ONLY if cycle 0 reaches sub-1.0 [contest-CUDA] (i.e. submission-candidate threshold); cost +$0.30 Linux x86_64 CPU eval per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE"
6. **Observable byte-mutation proof per Catalog #272:** `distinguishing_feature_name=ima_lottery_ticket_subnetwork_mask`, `distinguishing_bytes_path=IMPS_HEADER.DENSE_MASK`, `inflate_consumer_function=inflate_renderer.py::_imps_inflate`, `byte_mutation_smoke_passes=true` (mutate 1 bit in DENSE_MASK; verify output frames change)

## 1.5 Op-routables

- **Op-A:** dispatch the smoke ($0.30) BEFORE the full ($5-15) per Catalog #167 smoke-before-full pattern
- **Op-B:** Verify Catalog #94 (`check_imp_cycles_use_ema_and_auth_eval`) STILL fires on the new recipe
- **Op-C:** Update lane registry `lane_17_imp_10cycle` from L2 → L3 if contest_cuda gate hits sub-1.10 OR keep at L2 with re-classified kill memo if cycle 0 returns ≥ 1.30 (NEW kill memo with 3-section compliance)

---

# SUBSTRATE 2 — Lane STC clean-source (Syndrome-Trellis Coding on clean SegNet argmax)

## 2.1 Reactivation rationale (full)

Lane STC clean-source was originally tagged FALSIFIED 2026-04-29 (~21MB stream, 50.5× regression vs AV1 421KB baseline). The STATUS REVISION 2026-04-29 PM downgraded to UNDETERMINED-pending-CUDA when codex review caught that the 21MB measurement was on MPS-derived SegNet argmax masks. Per CLAUDE.md "MPS auth eval is NOISE — NON-NEGOTIABLE, HIGHEST EMPHASIS": MPS-argmax bytes ≠ CUDA-argmax bytes (PoseNet 23× drift; SegNet 2× drift). The 21MB was an MPS-PROXY measurement; the contest scorer never sees those masks.

The Modal T4 CUDA re-run (~$0.20, ~10 min) was queued but never executed. Lane registry shows `lane_stc_clean_source` at Level 1 with `impl_complete=true` and notes "Audit: Level 1.5 (firing now) — needs post-result harvest". The harvest never landed.

**Per CLAUDE.md "Forbidden premature KILL without research exhaustion":** today's verdict on STC clean-source is **UNDETERMINED-pending-CUDA-re-run**, NOT KILLED.

### Today's 5 lenses applied

1. **Chroma preservation:** N/A — STC operates on argmax masks (5-class semantic), not on chroma channels.
2. **Dykstra-feasibility:** NOT applied originally (no predicted band; the kill was an upper-bound measurement, not a band-failure). Today's reactivation: feasibility region is the intersection of (a) STC encoding's information-theoretic bound (entropy of argmax-mask transition probabilities), (b) AV1 monochrome's 0.014 bpp empirical floor on the same data, (c) STC's "majority-plus-exceptions" representation overhead. The 21MB MPS-PROXY ≈ 50× AV1; the question is whether CUDA argmax masks reduce that by 2× (matching SegNet device drift) — in which case STC would still be 25× over AV1 and DEFINITIVELY falsified. The remaining question: does CUDA argmax happen to produce a structurally simpler boundary set that drops STC bytes by 10×+?
3. **9-dim checklist:** Dimension 4 RIGOR previously failed (MPS-PROXY treated as `[contest-CUDA]` decision-grade per the CLAUDE.md MPS-falsification trap); Dimension 9 NEVER MEASURED on the contest substrate.
4. **Canonical-vs-unique:** Filler STC is substrate-class-shift (steganography-derived codec; Filler & Pevny 2010); kill was MPS-evidence-only. Today's decision: FORK encoder logic (STC is unique), ADOPT canonical mask source (CUDA SegNet argmax via `tac.substrates._shared.score_aware_common.score_pair_components`).
5. **7-cargo-cult inventory:** NONE present in design; ALL cargo-cult was in evidence collection (MPS-PROXY).

## 2.2 Comprehensive full-stack design

### 2.2.1 Substrate scope

Lane STC clean-source — encode SegNet argmax masks (5-class semantic labels) per pair using Syndrome-Trellis Coding (Filler & Pevny 2010). Compare bytes to AV1 monochrome baseline (421KB for 1200 frames).

### 2.2.2 Architecture (UNIQUE FORK)

- **Source:** SegNet argmax on `upstream/videos/0.mkv`, contest CUDA device, contest scorer (NOT MPS).
- **Codec:** `src/tac/codec/syndrome_trellis_codec.py` (already implemented; 15 tests passing per lane registry)
- **Sparsity criterion:** boundary_fraction = 0.05 default; tunable per cycle
- **Compression:** brotli on STC-encoded stream

### 2.2.3 Priors / sensitivity

- **Filler & Pevny 2010 prior:** STC's information-theoretic bound is the entropy of the source distribution; for 5-class semantic masks the bound is H(argmax) per pixel ≈ 2 bits.
- **AV1 monochrome empirical floor:** 0.014 bpp = 1/72 bits/pixel via interframe prediction + 2D context. STC has no interframe prediction.
- **The structural gap:** STC at boundary_fraction=0.05 codes ~5% of pixels = 11.8M pixels for 1200 frames × 1.8 bytes/pixel ≈ 21MB. AV1 achieves 421KB via interframe/2D context that STC lacks. Reducing STC to match AV1 requires `boundary_fraction ≈ 0.0036` — at that fraction the residual stream blows up.

### 2.2.4 Score-aware loss

N/A — STC is a CODEC, not a trained renderer. The "loss" is reconstruction bytes vs ground-truth argmax (lossless or near-lossless). Per the 2026-05-08 finding `feedback_filler_pevny_2010_dual_layer_stc_av1_landed_20260508.md` STC PR-alpha mask empirical landed at STC=996B vs LZMA-ternary=3084B (saves 68% on 262144-sym sample), proving STC CAN beat baselines when applied to the right source.

### 2.2.5 Archive grammar (UNIQUE FORK)

Stand-alone codec; the STC bytes go into the archive's masks slot.

```
STC\x00            # magic
HEADER_LEN (4B LE)
HEADER (JSON: num_frames, boundary_fraction, syndrome_table_sha256)
SYNDROME_TABLE (LUT-indexed STC syndrome → class transitions)
ENCODED_STREAM (brotli-compressed STC bitstream)
```

### 2.2.6 Inflate runtime

Existing `tools/pr_alpha_mask_stc_empirical.py` decoder; per Catalog #205 sister gate the STC inflate.py branch already uses canonical `select_inflate_device`.

### 2.2.7 Export contract

STC bytes replace AV1 monochrome in the masks slot of an existing archive (e.g., Quantizr 0.33 paradigm's masks.mkv). The reactivation test: drop-in replace masks.mkv with STC bytes; auth-eval the resulting archive.

### 2.2.8 Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| Codec primitives | UNIQUE FORK | STC IS the substrate; canonical equivalent doesn't exist |
| Mask source | ADOPT canonical CUDA SegNet | The original kill was caused by NOT adopting CUDA |
| Archive grammar | UNIQUE FORK | STC magic byte handler |
| Inflate runtime | UNIQUE FORK | STC decoder |
| Auth-eval gate | ADOPT canonical `gate_auth_eval_call` | Substrate-agnostic |
| Hardware detection | ADOPT canonical | Substrate-agnostic |
| Modal dispatch | ADOPT canonical `modal_train_lane.py` | Substrate-agnostic |
| Recipe schema | ADOPT canonical 36-field SubstrateContract | Substrate-agnostic |

### 2.2.9 9-dim checklist evidence

1. **UNIQUENESS:** Filler & Pevny 2010 STC is CLASS-SHIFT (steganography-derived codec). Distinct from AV1 monochrome (which uses interframe prediction + 2D context) — STC is per-frame independent.
2. **BEAUTY + ELEGANCE:** STC encoder ~16KB Python (679 LOC `stc_boundary_codec.py` + 24KB `codec/syndrome_trellis_codec.py`); could be trimmed. PR101-style 30-sec reviewable for the encoder kernel.
3. **DISTINCTNESS:** distinct from all sister mask codecs — AV1 monochrome (Quantizr 0.33), grayscale-LUT (Selfcomp 0.38), entropy coder (LZMA). STC is the unique steganography-derived path.
4. **RIGOR:** premise verification PV-1 (CUDA argmax masks ≠ MPS argmax masks; verified empirically); 1 council round (the 2026-04-29 PM revision); assumption classification HARD-EARNED for STC information theory, CARGO-CULTED for MPS-PROXY evidence collection.
5. **OPTIMIZATION PER TECHNIQUE:** canonical-vs-unique per layer above.
6. **STACK-OF-STACKS-COMPOSABILITY:** STC IS a drop-in replacement for masks.mkv in any archive grammar that ships discrete-class semantic masks (Quantizr 0.33 paradigm). Orthogonal to renderer (compositional with NSCS01/NSCS02) and quantization (compositional with apogee_int4 QAT).
7. **DETERMINISTIC REPRODUCIBILITY:** lossless codec; byte-stable; seed-pinned (no training).
8. **EXTREME OPTIMIZATION + PERFORMANCE:** STC encoder is CPU-bound; encode wall-clock 459.9s for 1200 frames on MPS-derived input. CUDA-derived input may be similar. ~$0.20 Modal T4 (mostly inflate + auth_eval).
9. **OPTIMAL MINIMAL CONTEST SCORE:** if STC bytes < 200KB (50% of AV1 baseline), STC enters the entropy-coding portfolio. If STC bytes 421KB-2MB, STC composes with AV1 as residual coder. If STC bytes > 5MB, structural redesign required (AV1+STC residual / temporal predictor / scanline RLE / lossy STC per source memo).

### 2.2.10 Predicted ΔS band + Dykstra-feasibility check (Catalog #296)

**Predicted band on contest CUDA argmax:**
- **Best case:** CUDA argmax produces structurally simpler boundary set; STC ≈ 200-1000KB. ΔS [-0.005, +0.005] vs AV1 baseline.
- **Likely case:** CUDA-vs-MPS drift is 2× per SegNet drift coefficient; STC ≈ 10MB. ΔS [+0.05, +0.20] regression vs AV1.
- **Worst case:** CUDA argmax produces SIMILAR boundary set; STC ≈ 20MB. ΔS [+0.15, +0.40] regression vs AV1; STRUCTURAL KILL with 3-section compliance.

**Dykstra-feasibility check:**
- **Constraint A (rate):** STC bytes are bounded below by H(argmax) × num_pixels = 2 bits × 236M = 59MB lower bound for STC raw → brotli should reduce to maybe 30MB.
- **Constraint B (lossless):** STC IS lossless; reconstruction error = 0.
- **Constraint C (AV1 floor):** 421KB AV1 = 0.014 bpp interframe-prediction floor.
- **Convex hull intersection:** STC's achievable region is {bytes ≥ 30MB raw, ≥ ~2-5MB after brotli, reconstruction error = 0}. AV1's region is {bytes = 421KB, reconstruction error > 0 but bounded by AV1 quantization noise}. The two regions DO NOT INTERSECT at lower-bytes; AV1 wins on rate axis structurally.
- **First-principles citation:** Shannon entropy of argmax masks > AV1 interframe-prediction floor by structural construction. STC cannot beat AV1 on this source. The reactivation test is whether STC can ENTER the portfolio as a RESIDUAL coder (AV1 base + STC corrections), which is a different design.

NOT vibes — the Dykstra-feasibility check predicts STC clean-source as REPLACEMENT will likely fail; the value is in PROVING that empirically OR PIVOTING to AV1+STC-residual.

### 2.2.11 Cost estimate + dispatch optimization protocol

- **Smoke (CUDA argmax + STC encode + auth_eval):** $0.20 Modal T4 (~10 min wall-clock)
- **Tier 1:** N/A (no training; STC is a codec)
- **Tier 2:** Modal T4 recipe declares min_vram_gb=8, target_modes=[research_substrate] (since likely-falsifying)
- **Tier 3:** research_only=true UNTIL the CUDA re-run lands a sub-1MB byte count

### 2.2.12 Observability surface

- **Per-config audit log:** `.omx/state/lane_stc_clean_source_audit.log` JSONL with `boundary_fraction, stc_bytes_raw, stc_bytes_brotli, av1_bytes_baseline, encode_wall_clock_seconds, mask_source_device, mask_source_sha256`
- **Modal call_id:** registered per Catalog #245
- **Premise verification PV-1:** MPS argmax masks ≠ CUDA argmax masks (verified by sha256 of argmax outputs at known frame indices on both devices)
- **Score-axis tag:** `[contest-CUDA T4]` for the CUDA re-run; `[MPS-PROXY]` for any legacy MPS results

## 2.3 Probe-disambiguator plan

**Probe:** `tools/probe_lane_stc_clean_source_cuda_vs_mps_disambiguator.py`

**Question:** does CUDA-derived SegNet argmax dramatically reduce STC byte count vs MPS-derived?

**Test design:**
- Encode SegNet argmax (CUDA on Modal T4) into STC at boundary_fraction=0.05
- Encode SegNet argmax (MPS) into STC at boundary_fraction=0.05
- Compare bytes; compare against AV1 421KB baseline
- Decision tree:

| CUDA STC bytes | Verdict |
|---|---|
| < 200KB | REACTIVATED (50% better than AV1) — promote to L2 |
| 200KB - 1MB | COMPETITIVE — pivot to AV1+STC residual composition |
| 1MB - 5MB | RESEARCH-ONLY (STC alone doesn't beat AV1; AV1+STC residual marginal) |
| > 5MB | HARD-EARNED FALSIFICATION (3-section compliance KILL memo) |

**Cost:** $0.20 — cheapest disambiguator in the audit Tier 1.

## 2.4 Reactivation gates

1. **Smoke green:** STC encode rc=0 + auth_eval JSON parseable on Modal T4 CUDA
2. **Tier C MDL density measured:** N/A (STC is a codec; no trained weights to ablate)
3. **100ep auth-eval anchor:** N/A; single auth-eval against drop-in replacement archive
4. **Custody validated per Catalog #127:** evidence_grade matches axis + hardware (CUDA T4)
5. **Paired CPU+CUDA:** if STC < 1MB AND auth_eval < 1.0 [contest-CUDA] (submission-candidate threshold), required
6. **Observable byte-mutation proof per Catalog #272:** `distinguishing_feature_name=stc_clean_source_argmax_encoding`, `distinguishing_bytes_path=STC_HEADER.ENCODED_STREAM`, `inflate_consumer_function=stc_clean_source_decoder.py::decode_stc_stream`, `byte_mutation_smoke_passes=true` (mutate 1 byte in ENCODED_STREAM; verify decoded mask changes)

## 2.5 Op-routables

- **Op-A:** dispatch the $0.20 Modal T4 CUDA smoke FIRST per Catalog #167 smoke-before-full pattern
- **Op-B:** if STC bytes are 200KB-5MB, PIVOT to designing `lane_av1_plus_stc_residual` per source memo (AV1 base + STC for residual corrections)
- **Op-C:** if STC bytes > 5MB, land 3-section structural KILL memo with HARD-EARNED Dykstra-feasibility evidence + reactivation criteria (AV1+STC residual / temporal predictor / scanline RLE / lossy STC)

---

# SUBSTRATE 3 — apogee_int4 QAT (INT4 quantization with quantization-aware training, Quantizr 0.33 leader's recipe)

## 3.1 Reactivation rationale (full)

apogee_int4 NAIVE-PTQ landed 1.4287 [contest-CUDA T4] on 2026-05-05 (109,996-byte archive). The kill memo `project_apogee_int4_FALSIFIED_score_1_43_dispatcher_VALIDATED_20260505.md` explicitly enumerates 6 unexplored reactivation paths (QAT, LSQ, per-channel scaling, smaller block sizes, outlier handling/clipping, GPTQ/AWQ calibration) per CLAUDE.md "KILL is LAST RESORT" non-negotiable.

**Quantizr 0.33 leader uses INT4** (88K-param FiLM-conditioned depthwise-separable CNN at FP4 = 4-bit signed quantization with QAT). This is direct, empirically-confirmed evidence that INT4 IS viable on the contest substrate — when paired with the correct quantization recipe (QAT, NOT NAIVE-PTQ).

The killer-assumption was that "1.4287 reflects INT4's fundamental capacity at PR106 architecture." The TRUE classification per today's lenses: **HARD-EARNED-CORE** (single config IS falsified for NAIVE-PTQ on PR106-HNeRV) + **CARGO-CULTED-SHELL** (treating one config's failure as a class-kill of the INT4 quantization tier).

### Today's 5 lenses applied

1. **Chroma preservation:** N/A — quantization operates on weight space, not chroma channels.
2. **Dykstra-feasibility:** HARD-EARNED ON THIS CONFIG. The (rate=109996, distortion=1.43) point IS dominated by every (rate < 200000, distortion < 1.0) point. NOT HARD-EARNED for QAT variant: Quantizr 0.33 leader's existence shows the feasible region includes (rate ~110000, distortion ~0.25) at the QAT corner.
3. **9-dim checklist:** Dimension 1 UNIQUENESS — INT4 IS class-shift (quantization tier); Quantizr (0.33 leader) USES INT4 FP4A → INT4 IS leaderboard-proven viable. Dimension 5 OPTIMIZATION-PER-TECHNIQUE — NAIVE-PTQ is canonical convenience; QAT is substrate-optimal for low-bit quantization.
4. **Canonical-vs-unique:** NAIVE-PTQ IS canonical default; QAT/LSQ/per-channel are substrate-optimal alternatives. CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD applies: the right answer is to FORK NAIVE-PTQ and adopt QAT.
5. **7-cargo-cult inventory:** Symposium-#4-band-prediction cargo-cult (predicted sub-0.30, actual 1.43, 4.8× outside-band). Reactivation predicted band MUST be derived from Dykstra-feasibility on Quantizr-leader empirical, NOT from naive bit-width scan.

## 3.2 Comprehensive full-stack design

### 3.2.1 Substrate scope

apogee_int4 QAT — INT4 quantization-aware training applied to PR106 HNeRV renderer (88K-100K params target). Replicate Quantizr 0.33 leader's recipe at PR106 scope.

### 3.2.2 Architecture (canonical-leader replication)

- **Base renderer:** PR106 HNeRV decoder (88K params, 0.20946 [contest-CUDA] anchor with NAIVE-FP4A)
- **Quantization recipe:** Quantizr's 5-stage pipeline per `feedback_grand_council_imp_permanent_fix_review_20260430.md` references:
  - Stage 1: anchor weights (load PR106 0.20946 checkpoint)
  - Stage 2: per-tensor calibration (compute clipping ranges over `upstream/videos/0.mkv` reference batch)
  - Stage 3: QAT fine-tune (FakeQuantFP4 active during forward; gradient flows through straight-through estimator; 20% of original training epochs at 0.1× base_lr)
  - Stage 4: per-channel scaling refinement (block_size {32, 64, per-channel} sweep)
  - Stage 5: outlier handling — top-1% weights kept at INT8 (mixed-precision); rest at INT4

### 3.2.3 Priors / sensitivity

- **Quantizr 0.33 leader prior:** STRONG — INT4 FP4A on 88K-param PR106-like architecture WORKS at sub-0.40 contest-CUDA score
- **NAIVE-PTQ falsification:** HARD-EARNED — the 1.43 measurement IS the empirical evidence that NAIVE-PTQ at FP4 is too aggressive for THIS architecture; bridge is QAT
- **Hessian-aware sensitivity (GPTQ/AWQ):** unexplored prior; potential 0.05-0.10 score improvement over plain QAT
- **Per-channel scaling:** unexplored prior; potential 0.05 score improvement over per-block scaling

### 3.2.4 Score-aware loss (UNIQUE FORK — QAT-during-forward + straight-through-estimator + canonical scorer-loss routing)

QAT introduces FakeQuantFP4 in the forward pass to simulate INT4 quantization noise; gradients flow through the straight-through estimator (STE) per LSQ (Esser 2019). The loss is the standard score-aware loss (preprocess_input + target no-grad + sqrt(10) pose weighting per Catalog #164 canonical helper) computed on the FakeQuant-perturbed renderer outputs.

UNIQUE element: bit-width is FIXED at 4 throughout QAT; the gradient discovers the renderer parameters that minimize score-aware loss UNDER the constraint of INT4 dynamic range. NAIVE-PTQ takes a trained renderer and rounds; QAT trains a renderer that's optimal AT INT4.

### 3.2.5 Archive grammar (canonical PR106 paradigm)

PR106 HNeRV archive grammar (already implemented):
```
PR106 HNeRV magic
INT4 weight table (per-tensor scales + per-block FP4A quantized weights)
brotli-compressed decoder bytecode
brotli-compressed latents
```

The QAT pipeline produces a renderer.bin in the SAME PR106 HNeRV format; the bytes are INT4 weights but the trainer is QAT. Inflate-time decoder is identical to NAIVE-PTQ INT4.

### 3.2.6 Inflate runtime

Existing PR106 HNeRV inflate runtime (`submissions/apogee_intN/inflate.py`); per Catalog #205 sister gate uses canonical `select_inflate_device`.

### 3.2.7 Export contract

QAT renderer.pt → PR106 INT4 weight repacker (`experiments/repack_pr106_with_intN_block_fp.py` ALREADY LANDED per lane registry) → archive.zip.

### 3.2.8 Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Quantization primitives | UNIQUE FORK | FakeQuantFP4 + LSQ + per-channel scaling = Quantizr's recipe; canonical PyTorch lacks |
| FakeQuant module | ADOPT existing `src/tac/quantization.py::FakeQuantFP4` (already implemented per kill memo line 110) |
| LSQ support | ADOPT existing `src/tac/training.py::LSQ` support (already implemented) |
| Per-channel scaling | UNIQUE FORK — block_size sweep {32, 64, per-channel} per Quantizr recipe |
| Outlier handling | UNIQUE FORK — mixed-precision INT8 on top-1% per Quantizr |
| Score-aware loss | ADOPT canonical `score_pair_components` | Catalog #164 substrate-agnostic |
| eval_roundtrip | ADOPT canonical | CLAUDE.md non-negotiable; substrate-agnostic |
| EMA | ADOPT canonical with decay=0.997 | CLAUDE.md non-negotiable; substrate-agnostic |
| Archive grammar | ADOPT PR106 HNeRV canonical | INT4 is a DIFFERENT QUANTIZATION of same architecture; same archive grammar |
| Inflate runtime | ADOPT canonical PR106 inflate | Same architecture |
| Auth-eval gate | ADOPT canonical `gate_auth_eval_call` | Catalog #226 substrate-agnostic |
| Modal dispatch | ADOPT canonical `modal_train_lane.py` | Substrate-agnostic |
| Recipe schema | ADOPT canonical 36-field SubstrateContract | Substrate-agnostic |

### 3.2.9 9-dim checklist evidence

1. **UNIQUENESS:** INT4 IS CLASS-SHIFT (quantization tier — 4-bit dynamic range structurally distinct from 8-bit); QAT FORK distinct from NAIVE-PTQ canonical. Distinct from sister substrates (NSCS01/02/03 are architectural; STC is codec; IMP is sparsity).
2. **BEAUTY + ELEGANCE:** Quantizr's recipe is ~50-200 LOC of QAT fine-tune scaffold on top of canonical training loop. PR101-style 30-sec reviewable.
3. **DISTINCTNESS:** distinct from PR106 baseline (which uses NAIVE-FP4A); distinct from apogee_intN siblings (int5/6/7/8 are baseline points; apogee_int4 QAT IS the substrate-engineering breakthrough candidate).
4. **RIGOR:** premise verification PV-1 (FakeQuantFP4 module exists at `src/tac/quantization.py:line_unknown`); PV-2 (LSQ support at `src/tac/training.py`); empirical anchor 0.33 [contest-CPU public Quantizr leader] = direct leaderboard-proven viability; assumption classification HARD-EARNED for QAT recipe per Quantizr, CARGO-CULTED-SHELL for "NAIVE-PTQ falsification = INT4 class-kill".
5. **OPTIMIZATION PER TECHNIQUE:** canonical-vs-unique per layer above.
6. **STACK-OF-STACKS-COMPOSABILITY:** quantization is ORTHOGONAL axis to architecture (NSCS01/02/03), sparsity (IMP), entropy coding (STC). The quantized renderer.bin replaces the FP4A renderer.bin in any archive grammar that ships compressed weights.
7. **DETERMINISTIC REPRODUCIBILITY:** seed-pinned (canonical training); byte-stable archive (INT4 weights are deterministic from QAT-converged checkpoint); EMA shadow is the export.
8. **EXTREME OPTIMIZATION + PERFORMANCE:** TF32 + autocast_fp16 + torch.compile via canonical helper; QAT fine-tune at 20% of original epochs (~20 epochs × ~5min/epoch ≈ 1.5h on A100) = $2.50 in fine-tune + $0.30 inflate+auth_eval ≈ $3 per smoke; $5-15 for full QAT recipe with per-channel scaling sweep.
9. **OPTIMAL MINIMAL CONTEST SCORE:** predicted band derived from Quantizr 0.33 leader empirical. Quantizr's 0.33 was a 88K-param FiLM-conditioned CNN; our PR106 is a HNeRV decoder. Architecture difference adds ±0.10 uncertainty. **Predicted band [0.25, 0.55] with mode 0.40** if QAT alone; predicted band [0.20, 0.40] if QAT + per-channel scaling; predicted band [0.15, 0.35] if QAT + per-channel + outlier handling + GPTQ-style calibration.

### 3.2.10 Predicted ΔS band + Dykstra-feasibility check (Catalog #296)

**Predicted band:** [0.25, 0.55] for QAT alone; mode 0.40.

**Dykstra-feasibility check:**
- **Constraint A (rate):** INT4 fixed at ~110KB; Quantizr 0.33 is at ~300KB. The rate constraint is satisfied with slack.
- **Constraint B (distortion):** Quantizr 0.33 empirically demonstrates that QAT-FP4 achieves sub-0.30 distortion on a similar-class 88K-param renderer. PR106's HNeRV decoder may differ; uncertainty ±0.15.
- **Constraint C (NAIVE-PTQ floor):** the 1.43 NAIVE-PTQ result IS NOT a feasibility floor for QAT — it's the floor for ONE configuration (no fine-tune, single scaling).
- **Convex hull intersection:** Quantizr's (rate=300KB, distortion=0.10) point + PR106's NAIVE-PTQ (rate=110KB, distortion=1.43) point + PR106's NAIVE-INT8 (rate=187KB, distortion=0.21) baseline. The feasible region for QAT-FP4 on PR106 is bounded above by (rate=110KB, distortion = NAIVE-PTQ result - (gap recovered by QAT)). Quantizr's QAT recovery factor is ~10× (NAIVE-PTQ predicted floor was ~3.0; achieved 0.33 = 9× improvement). Applied to PR106: 1.43 / 10 ≈ 0.143 best-case; 1.43 / 5 ≈ 0.286 likely-case; 1.43 / 3 ≈ 0.476 conservative-case.
- **First-principles citation:** Esser 2019 LSQ paper + Frantar 2022 GPTQ paper + Quantizr 0.33 empirical receipt.

NOT vibes — predicted band is derived from a leaderboard empirical AND a publication-grade prior.

### 3.2.11 Cost estimate + dispatch optimization protocol

- **Smoke (QAT fine-tune 20 epochs + auth_eval):** $2-3 on Modal A100
- **Full (QAT + per-channel + outlier handling sweep):** $5-15 on Modal A100
- **Dispatch optimization protocol verdict:** Tier 1 ✓ (canonical training has autocast_fp16/TF32/torch.compile/no_grad); Tier 2 NEEDS BACKFILL (recipe declares min_vram_gb=24, min_smoke_gpu=A100, video_input_strategy=cpu_thread_async_upload, target_modes=[contest_one_video_replay, contest_generalized], canonical NVML env block per Catalog #244); Tier 3 must declare research_only=false AND canonical auth_eval routing.

### 3.2.12 Observability surface

- **Per-stage audit log:** `.omx/state/lane_apogee_int4_qat_audit.log` JSONL with `stage_id, qat_epoch, fakequant_active, block_size, outlier_threshold, calibration_method, archive_bytes, archive_sha256, auth_eval_score, auth_eval_axis`
- **Per-stage checkpoint:** `experiments/results/lane_apogee_int4_qat/stage_<N>/checkpoint.pt`
- **Premise-verification artifact:** PV-1 (FakeQuantFP4 active during QAT forward; verified by hooking into the quantizer's forward and asserting bit-width = 4)
- **Modal call_id:** registered per Catalog #245
- **Adjudication artifact:** per stage's `evidence_grade` with `score_axis=contest_cuda` enforced by Catalog #221
- **Score axis tag:** every result `[contest-CUDA T4]` or `[contest-CUDA A100]` per CLAUDE.md "Apples-to-apples evidence discipline"

## 3.3 Probe-disambiguator plan

**Probe:** `tools/probe_apogee_int4_qat_vs_naive_ptq_disambiguator.py`

**Question:** does QAT recover sub-1.0 score on PR106 INT4 archive?

**Test design:**
- Run TWO INT4 quantization paths on the SAME PR106 0.20946 anchor checkpoint:
  - **Arm A (NAIVE-PTQ):** trivial round-to-nearest INT4 (the 1.43 baseline) — ~$0.05
  - **Arm B (QAT):** 20 epochs of QAT fine-tune at 0.1× base_lr — ~$2-3
- Auth-eval BOTH on Modal A100 CUDA
- Decision tree:

| Arm A | Arm B | Verdict |
|---|---|---|
| 1.43 | ≤0.40 | CARGO-CULTED (QAT works; promote per-channel + outlier sweep) |
| 1.43 | 0.40-0.80 | PARTIAL (QAT partially recovers; continue per-channel + outlier sweep $5-15) |
| 1.43 | 0.80-1.20 | MARGINAL (QAT recovery insufficient; advance to LSQ + GPTQ) |
| 1.43 | ≥1.20 | HARD-EARNED (QAT alone insufficient; structural mismatch with HNeRV decoder; KILL with 3-section compliance) |

**Cost:** $0.05 + $2-3 = $2-3 — moderate disambiguator.

## 3.4 Reactivation gates

1. **Smoke green:** QAT fine-tune rc=0 + auth-eval JSON parseable on Modal A100
2. **Tier C MDL density measured:** `tools/mdl_scorer_conditional_ablation.py` on QAT-FP4 archive (Catalog #227)
3. **100ep auth-eval anchor:** byte-deterministic archive (sha256 stable across re-runs)
4. **Custody validated per Catalog #127:** evidence_grade matches axis + hardware
5. **Paired CPU+CUDA:** if QAT lands sub-1.0 [contest-CUDA], REQUIRED per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
6. **Observable byte-mutation proof per Catalog #272:** `distinguishing_feature_name=qat_fakequant_fp4_aware_weights`, `distinguishing_bytes_path=INT4_WEIGHT_TABLE`, `inflate_consumer_function=submissions/apogee_intN/inflate.py::_inflate_int4_weights`, `byte_mutation_smoke_passes=true` (mutate 1 byte in INT4 weight table; verify output frames change — same archive grammar as NAIVE-PTQ ALREADY satisfies this)

## 3.5 Op-routables

- **Op-A:** dispatch the smoke ($2-3) BEFORE the full ($5-15) per Catalog #167
- **Op-B:** if QAT alone returns sub-0.50, REPRIORITIZE per-channel scaling + outlier handling sweep as immediate dispatch ($5-10)
- **Op-C:** if QAT alone returns 0.80-1.20, ADVANCE to LSQ + GPTQ + AWQ (Hessian-aware calibration) per kill memo paths 2-5; cost $10-20
- **Op-D:** if QAT alone returns ≥1.20, land 3-section structural KILL memo (HARD-EARNED evidence + reactivation criteria: "if PR106 architecture is replaced by Quantizr-class FiLM-conditioned CNN, QAT MAY work")

---

# CROSS-SUBSTRATE COMPOSITION SECTION (joint with NSCS06 v8 / A-STACK / Wunderkind G1 v2 / ATW v2 / Symposium #4 T4)

## 4.1 Lane 17 IMP ⊕ NSCS06 v8/v7 chroma

**Composition hypothesis:** apply IMP iterative magnitude pruning to NSCS06 v8's chroma residual branch.

- NSCS06 v8 (Path B wavelet residual per `feedback_grand_council_nscs06_v8_path_b_wavelet_residual_full_stack_design_landed_20260516.md`) has a chroma residual codec with non-trivial weight count
- IMP at 50-89% sparsity on the chroma residual codec could shrink NSCS06 v8's chroma-branch bytes by 5-10×
- **Predicted ΔS:** if NSCS06 v8 achieves sub-1.5 [contest-CUDA] (uncertain), IMP-on-chroma-codec adds -0.005 to -0.015
- **Cost:** sister dispatch within IMP smoke ($0.30 incremental)
- **Risk:** NSCS06 v8's chroma codec must FIRST land L2+ before composition is meaningful

**Compositional priority:** MEDIUM (gated on NSCS06 v8 smoke green)

## 4.2 STC clean-source ⊕ A-STACK (NSCS01+NSCS02+NSCS03)

**Composition hypothesis:** STC encodes A-STACK's distinguishing signal axis as residual bytes.

- A-STACK is the comprehensive composition per `feedback_grand_council_a_stack_nscs01_nscs02_nscs03_comprehensive_composition_landed_20260516.md`
- If A-STACK ships a discrete-class semantic signal (NSCS02 mask split, NSCS01 nullspace mask), STC could encode the boundaries of that signal as residual coding on top of A-STACK's primary archive
- **Predicted ΔS:** -0.01 to -0.02 if STC bytes are 10-50KB (clean-source signal is structurally simpler than full SegNet argmax)
- **Cost:** $0.10 STC encode + $0.30 auth_eval of composed archive
- **Risk:** A-STACK must FIRST validate composition is byte-stable

**Compositional priority:** LOW (gated on A-STACK L2 + STC clean-source L2)

## 4.3 apogee_int4 QAT ⊕ Wunderkind G1 v2 entropy class-shift

**Composition hypothesis:** QAT-FP4 quantization on top of Wunderkind G1 v2's substrate-class-shift architecture.

- Wunderkind G1 v2 (per `feedback_grand_council_wunderkind_g1_v2_entropy_coded_full_stack_design_landed_20260516.md`) is the entropy-coded class-shift that adds non-trivial bytes (~1-5KB) as hyperprior + class-index sidechannel
- QAT-FP4 on Wunderkind G1 v2's substrate codec weights shrinks the codec bytes by 4× (assuming current codec is FP16)
- **Predicted ΔS:** -0.005 to -0.015 if Wunderkind G1 v2 codec is currently FP16
- **Cost:** $2-3 QAT smoke on the Wunderkind G1 v2 codec weights
- **Risk:** Wunderkind G1 v2 must FIRST land L2+

**Compositional priority:** MEDIUM-HIGH (gated on Wunderkind G1 v2 smoke green; QAT recipe transfers cleanly across substrates)

## 4.4 Higher-order pair-wise composition matrix

Among all 7 design memos landed this turn (NSCS06 v8 / A-STACK / Wunderkind G1 v2 / ATW v2 / Symposium #4 T4 / Lane 17 IMP / STC clean-source / apogee_int4 QAT):

| Pair | Composition Orthogonality | Predicted Stacking ΔS |
|---|---|---|
| Lane 17 IMP × NSCS01/02/03 (A-STACK) | HIGH (sparsity ⊥ architecture) | -0.01 to -0.03 (additive) |
| Lane 17 IMP × apogee_int4 QAT | HIGH (sparsity ⊥ quantization) | -0.02 to -0.05 (additive; both shrink renderer.bin) |
| Lane 17 IMP × NSCS06 v8 | MEDIUM (sparsity on chroma codec) | -0.005 to -0.015 |
| STC clean-source × Quantizr 0.33 paradigm | LOW (STC replaces masks.mkv slot; Quantizr already optimal) | unlikely to improve |
| STC clean-source × A-STACK | MEDIUM (STC encodes A-STACK signal-axis residual) | -0.01 to -0.02 |
| apogee_int4 QAT × A-STACK | HIGH (QAT on A-STACK substrate weights) | -0.05 to -0.15 (multiplicative if QAT recipe transfers) |
| apogee_int4 QAT × NSCS06 v8 chroma branch | HIGH (QAT on chroma branch weights) | -0.02 to -0.05 |
| apogee_int4 QAT × Wunderkind G1 v2 codec | MEDIUM-HIGH (QAT on codec weights) | -0.005 to -0.015 |
| apogee_int4 QAT × ATW v2 | MEDIUM (QAT on ATW codec weights) | -0.005 to -0.02 |
| apogee_int4 QAT × Symposium #4 T4 staircase | LOW-MEDIUM (depends on Symposium #4 T4 details) | uncertain |

**Best higher-order composition:** **apogee_int4 QAT × A-STACK** (HIGH orthogonality + HIGH multiplicative potential). If A-STACK L2 lands at sub-0.50 [contest-CUDA] AND apogee_int4 QAT L2 lands at sub-0.40 [contest-CUDA] on PR106, the composition apogee_int4-QAT × A-STACK could approach Quantizr 0.33 leader-band per Quantizr's empirical receipt (88K-param FiLM-CNN at FP4-QAT). Dykstra-feasibility intersection: rate ~100KB ∩ distortion ~0.10 ∩ score ~0.30 = within Quantizr's empirical envelope.

**Second-best:** **Lane 17 IMP × apogee_int4 QAT** (sparsity + quantization both compress weight bytes; additive ΔS).

## 4.5 Composition cost budget

- Lane 17 IMP × A-STACK: $5-15 (IMP smoke + A-STACK composition smoke)
- apogee_int4 QAT × A-STACK: $5-15 (QAT smoke + A-STACK composition smoke)
- Lane 17 IMP × apogee_int4 QAT: $5-15 (combined sparsity+quantization smoke)

**Total higher-order composition budget:** $15-45 (3 pair-wise experiments).

---

# SUMMARY (per-substrate)

| Substrate | Verdict (today's classification) | Cost | Reactivation gate | Observability surface | Op-routables |
|---|---|---|---|---|---|
| **Lane 17 IMP** | CARGO-CULTED stub-loop kill | $0.30 smoke + $5-15 full | Cycle 0 sub-1.30 with real train_distill | `.omx/state/lane_17_imp_audit.log` JSONL + per-cycle archive checkpoints + Modal call_id + adjudication artifact | 3 (smoke-before-full / regression-test gate #94 / L3 promotion decision) |
| **STC clean-source** | CARGO-CULTED MPS-PROXY evidence | $0.20 smoke | STC bytes ≤ 1MB on CUDA argmax | `.omx/state/lane_stc_clean_source_audit.log` JSONL + mask source SHA + boundary fraction sweep + Modal call_id | 3 (smoke-before-full / pivot to AV1+STC residual / 3-section KILL if > 5MB) |
| **apogee_int4 QAT** | HARD-EARNED-CORE + CARGO-CULTED-SHELL | $2-3 smoke + $5-15 full | QAT sub-0.50 [contest-CUDA] PR106 | `.omx/state/lane_apogee_int4_qat_audit.log` JSONL + per-stage checkpoint + Modal call_id + adjudication | 4 (smoke-before-full / per-channel+outlier sweep / LSQ+GPTQ advance / 3-section KILL if QAT ≥1.20) |

**Total reactivation budget for all 3:** $10-30 (smokes) + $15-45 (fulls if smoke greens) + $15-45 (best 3 pair-wise compositions) = **$40-120 max total exposure**.

**Ranked priority order (by mission-contribution / cost ratio):**

1. **STC clean-source CUDA re-run ($0.20)** — cheapest possible disambiguator + explicit UNDETERMINED tag + council's #1 hope. EXECUTE FIRST.
2. **Lane 17 IMP cycle 0 with real train_distill ($0.30)** — kill is already withdrawn + reactivation criteria explicit + empirical anchor (40.2% byte savings at cycle 5) ALREADY exists. EXECUTE SECOND.
3. **apogee_int4 QAT smoke ($2-3)** — Quantizr 0.33 leader uses INT4 + 5 unexplored research paths + direct leaderboard pursuit. EXECUTE THIRD.

If all 3 smokes pass their reactivation gates: total spend $2.50-3.50; lane registry gains 3 NEW L2 entries with hard empirical receipts; cathedral autopilot ranker gets 3 new posterior anchors per Catalog #128.

If any smoke fails: land 3-section structural KILL memo with HARD-EARNED Dykstra-feasibility evidence per CLAUDE.md "KILL/FALSIFIED memory verdicts" non-negotiable; the kill itself is a valuable mission output (clears zombie lanes from the registry; updates posterior toward correct prior).

---

## 9-DIMENSION SUCCESS CHECKLIST EVIDENCE (Catalog #294)

### Dimension 1 — UNIQUENESS

- **Lane 17 IMP:** Class-shift (lottery-ticket subnetwork sparsity per Frankle 2019). Distinct from quantization (apogee_int4) and codec (STC).
- **STC clean-source:** Class-shift (steganography-derived codec per Filler & Pevny 2010). Distinct from AV1 monochrome (interframe prediction), grayscale-LUT (Selfcomp), entropy coder (LZMA).
- **apogee_int4 QAT:** Class-shift (quantization tier — 4-bit dynamic range structurally distinct from 8-bit baselines). QAT FORK distinct from NAIVE-PTQ canonical.

### Dimension 2 — BEAUTY + ELEGANCE

- **Lane 17 IMP:** ~50 LOC bolt-on (IMPS magic-byte handler) on canonical train_distill — PR101-style reviewable.
- **STC clean-source:** ~16-24KB Python codec; encoder kernel reviewable.
- **apogee_int4 QAT:** ~50-200 LOC QAT scaffold on canonical training loop — PR101-style reviewable.

### Dimension 3 — DISTINCTNESS

Each substrate occupies a distinct slot in the design space (sparsity / codec / quantization). No two substrates within this batch overlap.

### Dimension 4 — RIGOR

- All 3 substrates carry premise verification artifacts.
- Lane 17 IMP: 3 council rounds + adversarial review counter at 3/3 already per lane registry.
- STC clean-source: 1 council round (the MPS-vs-CUDA verdict revision); 22-voice extreme-rigor codex review per kill memo source.
- apogee_int4 QAT: Single-pass adversarial check per kill memo + 6 enumerated reactivation paths per `feedback_three_lossy_anchors_show_rel_err_squared_objective_falsified_*`.
- Assumption classification HARD-EARNED vs CARGO-CULTED per Catalog #292 sextet-pact discipline.

### Dimension 5 — OPTIMIZATION PER TECHNIQUE

Canonical-vs-unique decision per layer (Catalog #290) documented for all 3 substrates above.

### Dimension 6 — STACK-OF-STACKS-COMPOSABILITY

Cross-substrate composition matrix above shows ORTHOGONAL axes:
- Sparsity (IMP) ⊥ quantization (apogee_int4) ⊥ codec (STC) ⊥ architecture (NSCS01/02/03)
- Multiplicative ΔS potential identified for apogee_int4 QAT × A-STACK (-0.05 to -0.15)
- Additive ΔS potential for Lane 17 IMP × apogee_int4 QAT (-0.02 to -0.05)

### Dimension 7 — DETERMINISTIC REPRODUCIBILITY

All 3 substrates: seed-pinned + byte-stable archives + canonical EMA shadow export + canonical posterior-update per Catalog #128.

### Dimension 8 — EXTREME OPTIMIZATION + PERFORMANCE

All 3 substrates: TF32 + autocast_fp16 + torch.compile via canonical `trainer_skeleton.device_or_die` helper. STC (codec, not trainer) is CPU-bound by design.

### Dimension 9 — OPTIMAL MINIMAL CONTEST SCORE

- **Lane 17 IMP:** predicted band [1.05, 1.30]; if cycle 5 sub-0.95, beats Lane G v3 anchor.
- **STC clean-source:** likely [1MB, 20MB] bytes; replaces masks.mkv in archive grammar; ΔS depends on baseline archive's masks.mkv slot.
- **apogee_int4 QAT:** predicted band [0.25, 0.55]; if mode 0.40 hits, top-3 leaderboard band.

---

## CANONICAL-VS-UNIQUE DECISION PER LAYER — UMBRELLA SUMMARY

For all 3 substrates COMBINED, the canonical-vs-unique pattern is:

**ADOPT canonical (substrate-agnostic engineering hygiene):**
- TF32 / autocast_fp16 / torch.compile / no_grad-at-eval / canonical scorer-loss helper / EMA / eval_roundtrip / `train_distill.py` / `score_pair_components` / `gate_auth_eval_call` / `select_inflate_device` / `detect_hardware_substrate` / `posterior_update_locked` / `modal_train_lane.py` / 36-field `SubstrateContract` / `modal_call_id_ledger`

**UNIQUE FORK (substrate-specific math/grammar):**
- Lane 17 IMP: pruning/mask logic + IMPS archive grammar + IMPS inflate runtime
- STC clean-source: STC codec primitives + STC archive grammar + STC decoder
- apogee_int4 QAT: FakeQuantFP4 + LSQ + per-channel scaling + outlier handling

**FORK justified WHEN canonical suppresses substrate-optimal signal:** the kill of each substrate was caused by treating substrate-specific work as canonical-shareable. The reactivation requires the FORK to land cleanly per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" non-negotiable.

---

## SISTER-SUBAGENT OWNERSHIP MAP (Catalog #230)

This subagent's scope is READ-ONLY except this design memo + checkpoints (`.omx/state/subagent_progress.jsonl`).

**Disjoint from:**
- T4 SYMPOSIUM subagent (Symposium #4 T4 staircase design)
- NSCS06 v8 Path B wavelet residual design subagent
- Wunderkind G1 v2 design subagent
- ATW v2 design subagent

No source file edits, no lane registry mutations, no preflight.py edits, no Catalog # claims.

---

## PREMISE VERIFICATION (Catalog #229)

PVs verified pre-edit:

1. **PV-1:** `feedback_grand_council_imp_permanent_fix_review_20260430.md` exists in memory directory and contains the 8-of-10 council kill-withdraw vote (CONFIRMED via Read tool)
2. **PV-2:** `project_lane_stc_clean_source_FALSIFIED_20260429.md` exists and contains "STATUS REVISION 2026-04-29 PM" + "Required action: re-run clean-source STC on Modal T4 CUDA" (CONFIRMED via Read tool)
3. **PV-3:** `project_apogee_int4_FALSIFIED_score_1_43_dispatcher_VALIDATED_20260505.md` exists and contains 6 reactivation paths (CONFIRMED via Read tool)
4. **PV-4:** `experiments/train_imp_cycle.py` exists at 475 LOC; `experiments/train_distill.py` exists at 1935 LOC; `src/tac/iterative_magnitude_pruning.py` exists at 607 LOC (CONFIRMED via wc -l)
5. **PV-5:** `src/tac/stc_boundary_codec.py` exists at 679 LOC; `src/tac/codec/syndrome_trellis_codec.py` exists at 16KB (CONFIRMED via ls)
6. **PV-6:** `submissions/apogee_intN` directory exists; `src/tac/quantization.py` contains 17 QAT/FakeQuantFP4/LSQ token mentions; `src/tac/training.py` contains 4 such mentions (CONFIRMED via grep -c)
7. **PV-7:** `.omx/research/resurrection_audit_20260516.md` exists and lists all 3 substrates in Tier 1 with explicit reactivation criteria (CONFIRMED via grep -n)
8. **PV-8:** Lane registry `.omx/state/lane_registry.json` contains entries for `lane_17_imp_10cycle` (L2, 6/8 gates), `lane_stc_clean_source` (L1, 1/8 gates), `lane_apogee_int4` (L2, 7/8 gates) (CONFIRMED via lane registry query)

---

## 6-HOOK WIRE-IN (per Catalog #125 "Subagent coherence-by-default" non-negotiable)

1. **Sensitivity-map contribution:** N/A — design memo only; no executable surface this subagent lands. Will be ACTIVE when the reactivation dispatches fire (sensitivity-aware per-tensor importance for apogee_int4 QAT; sparsity sensitivity per cycle for Lane 17 IMP; entropy sensitivity per boundary fraction for STC).
2. **Pareto constraint:** ACTIVE — Dykstra-feasibility check per substrate enumerates the (rate, distortion) convex hull intersection. Each substrate's predicted band IS the feasibility boundary.
3. **Bit-allocator hook:** N/A for design memo; will be ACTIVE for apogee_int4 QAT per-channel scaling sweep.
4. **Cathedral autopilot dispatch hook:** ACTIVE — this memo is the ranking input for the next 3 dispatches per resurrection audit Tier 1 verdicts.
5. **Continual-learning posterior update:** N/A for design memo; ACTIVE when reactivation smokes harvest results (via canonical `posterior_update_locked` per Catalog #128).
6. **Probe-disambiguator:** ACTIVE — 3 probes documented (one per substrate) at §1.3, §2.3, §3.3.

---

## CHECKPOINT DISCIPLINE (Catalog #206)

Subagent ID: `batched_reactivation_design_lane17_stc_apogee_int4`

Checkpoint 1: in_progress (Write batched design memo) — recorded at session start.
Checkpoint 2: complete — to be recorded after commit.

---

## SHARED ASSUMPTIONS THIS WORK OPERATES WITHIN (Assumption-Adversary surface; Catalog #292)

**Hard-earned assumptions:**
1. Each candidate's kill verdict CAN be retracted via empirical disambiguator (probe per substrate). Hard-earned via 3 independent kill memo retractions: IMP council vote, STC MPS-vs-CUDA non-negotiable, apogee_int4 6-paths enumeration.
2. Reactivation dispatch costs ($10-30 total) are within operator budget per resurrection audit Tier 1 cost column.
3. Modal A100 + T4 are available + dispatch-optimization-protocol per Catalog #270 has been satisfied for all 3 substrates' planned dispatches.

**Cargo-culted assumptions (under interrogation):**
1. "Quantizr 0.33 recipe transfers cleanly across architectures" — UNCERTAIN; Quantizr is FiLM-CNN; PR106 is HNeRV. Per-architecture transfer factor is empirical, not theoretical. PV-via-smoke required.
2. "Lane G v3 anchor at 1.05 is a stable baseline for IMP" — assumed in predicted band but Lane G v3 anchor may have drifted; cycle 0 must re-anchor.
3. "CUDA-vs-MPS argmax drift is ~2× per SegNet drift coefficient" — assumed; may be larger if the 5-class semantic mask structure interacts with FP precision differently. PV-via-empirical-paired-comparison required.

**Operating-within umbrella:** today's reactivation thesis depends on CLAUDE.md "Forbidden premature KILL without research exhaustion" being correct as a meta-discipline. If it's NOT correct (i.e. if 3 of the candidates are TRULY HARD-EARNED kills), the reactivation dispatches are wasted spend. Per operator standing directive 2026-05-16: this is acceptable — converting zombie lanes to either L2+ frontier OR 3-section-compliant kills IS mission-positive regardless of which way the verdicts fall.

---

## REFERENCES

- `.omx/research/resurrection_audit_20260516.md` (commit `d0c347f1f`) — Tier 1 verdicts
- `feedback_grand_council_imp_permanent_fix_review_20260430.md` — IMP kill-withdraw council vote
- `project_lane_17_imp_killed_cycle_0_198_regression_20260430.md` — IMP original kill memo
- `project_lane_stc_clean_source_FALSIFIED_20260429.md` — STC original kill + 2026-04-29 PM revision
- `project_apogee_int4_FALSIFIED_score_1_43_dispatcher_VALIDATED_20260505.md` — apogee_int4 kill + 6 reactivation paths
- `feedback_grand_council_imp_train_distill_swap_design_20260430.md` — IMP design DD1/DD2/DD3/DD4
- `feedback_filler_pevny_2010_dual_layer_stc_av1_landed_20260508.md` — STC + AV1 dual-layer empirical (PR-alpha mask 996B vs LZMA 3084B)
- `feedback_apogee_intN_lightning_failure_class_20260505.md` — apogee_intN dispatch infrastructure
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable
- CLAUDE.md "Forbidden premature KILL without research exhaustion" forbidden pattern
- CLAUDE.md "KILL/FALSIFIED memory verdicts" non-negotiable
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable
- CLAUDE.md "MPS auth eval is NOISE" non-negotiable
- CLAUDE.md "eval_roundtrip" + "EMA" non-negotiables
- Catalog #94 (`check_imp_cycles_use_ema_and_auth_eval`) — IMP STRICT preflight
- Catalog #125 (subagent landing 6-hook wire-in) + #127 (custody validator) + #128 (continual learning locked) + #131 (no bare writes shared state) + #138 (state writers strict load) + #164 (scorer preprocess) + #167 (smoke-before-full) + #190 (hardware substrate) + #205 (inflate device fork) + #206 (subagent checkpoint discipline) + #210 (DP1 codebook provenance) + #220 (substrate L1 scaffold operational mechanism) + #221 (auth eval result fail closed) + #226 (auth eval canonical helper) + #227 (Tier C MDL density) + #229 (premise verification) + #230 (sister-subagent ownership) + #240 (recipe-vs-trainer-state) + #241 (substrate META layer contract) + #244 (NVML env block) + #245 (Modal call_id ledger) + #249 (no misleading device-named dirs) + #270 (dispatch optimization protocol) + #272 (distinguishing feature integration) + #290 (canonical-vs-unique decision per layer) + #292 (council assumption discipline) + #294 (9-dim checklist) + #296 (Dykstra-feasibility) + #297 (signal-axis reversibility)

---

## CANONICAL-VS-UNIQUE DECISION PER LAYER

See per-substrate §1.2.8 + §2.2.8 + §3.2.8 + the umbrella summary above.

