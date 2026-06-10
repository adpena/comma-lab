<!-- SPDX-License-Identifier: MIT -->
# Full-stack correctness audit + findings-trust re-categorization — Task #81

**UTC:** 2026-06-10T20:01:15Z · **Subagent:** `task81_fullstack_audit` · **Mode:** forensic audit + 1 test file landed.
**Authority:** every numeric below is `[macOS-CPU advisory]` / `[macOS-MLX research-signal]` numerical-parity or
config inspection. NO contest score, NO MPS (CPU torch + CPU MLX), GT-free (synthetic latents). `$0` spend, no
dispatch. `promotable=false`, `score_claim=false`, `mechanism_update_eligible=true`,
`score_roadmap_update_eligible=false`.

---

## LEAD ANSWERS (the three the task demands)

1. **Is our PixelShuffle / decoder stack CORRECT? — YES.** The MLX training kernel
   `pr95_hnerv_mlx.pixel_shuffle_2x_nhwc` is **BIT-EXACT (0.0 absolute drift)** vs `torch.nn.PixelShuffle(2)`
   across 4 shapes; bilinear-2x matches `F.interpolate(align_corners=False)` to **~1e-7**; and the FULL
   `HNeRVDecoderMLX` (PixelShuffle + bilinear-skip + sin + terminal HF-refine + 6-stage 6×8→384×512 cascade,
   PR95 channel taper `[36,36,36,27,20,18,18]`) matches a from-scratch `nn.Module` reference **end-to-end to
   rel ~2e-7** (abs ~3e-5 on a 0..255 output = fp32 conv-accumulation epsilon, NOT a structural divergence).
   **The decoder math is not the bug.** Pinned by 21 new tests
   (`src/tac/tests/test_hnerv_decoder_nn_pixelshuffle_parity.py`).

2. **Does Quantizr's 88K success DISPROVE the pose-capacity-wall (#74)? — YES, the WALL AS STATED IS FALSE; but
   #74's *measurement* is correct for the case it actually tested.** Quantizr (PR #55, `0.33`) held
   **d_pose = 0.00051010 (IN the tube)** with an **88K-param** depthwise-separable FiLM CNN — SMALLER than #74's
   80-120kb student. The #74 wall ("a small net at 384×512 physically cannot hold the pose tube → needs
   near-frontier capacity") is FALSIFIED by counter-example. **#74 is SUSPECT and must be re-validated** — its
   verdict is sound *for distill-onto-teacher-frames where pose is recovered from pixels*, but FALSE as a
   general capacity claim, because Quantizr **does not recover pose from pixels — he STORES the 6-dim GT pose
   explicitly (`pose.npy.br`) and FiLM-injects it**, so the net never has to reproduce frames to ±5/255.

3. **Which findings are SUSPECT (ran on the buggy/unaudited training stack) and must be re-validated on #76's
   fixed loop?** The four that touched OUR score-aware MLX training loop and/or the distill-from-frame pose
   model: **#74 (pose-capacity-wall — SUSPECT, Quantizr disproves), #62 (d_seg wall — SUSPECT, same loop),
   #68 reactivation predictions (DESIGN, consume the #76 result), and the whole NeRV-fleet d_seg≈0.5 plateau
   (SUSPECT — shared-harness M-loss).** The frozen-frontier-bytes findings (#64/#69/#71/#72/#73/#54) are
   **PROVEN-FRONTIER (TRUST)** — they ran on PR95's released archive bytes through the #75-verified exact eval,
   not through our training loop. Detail in §3.

---

## §1 PIXELSHUFFLE / DECODER — bit-exact verdict (audit point 1)

| kernel | reference | drift | verdict |
|---|---|---|---|
| MLX `pixel_shuffle_2x_nhwc` | `torch.nn.PixelShuffle(2)` (the actual nn module, NCHW) | **0.0 absolute** | BIT-EXACT |
| MLX `bilinear_resize2x` (align_corners=False) | `F.interpolate(scale_factor=2, mode=bilinear, align_corners=False)` | ~1.2e-7 | fp32-exact |
| `bilinear_skip_residual_canonical` | `sin(w*(shuffled+identity))` numpy/torch/mlx | <1e-5 | parity (already tested) |
| `terminal_hf_refine_canonical` | `h + 0.1*sin(refine)` | <1e-6 | parity (already tested) |
| **FULL `HNeRVDecoderMLX`** | from-scratch `nn.Module` (same weights) | abs 3e-5 / **rel 2e-7** on 0..255 | **CORRECT** |

**The channel convention is correct.** The MLX kernel uses the channel-FIRST reshape `(B,H,W,out_C,2,2)` +
transpose `(0,1,4,2,5,3)` — this is exactly `nn.PixelShuffle`'s interleave. The FORBIDDEN channel-LAST layout
(the historical FIX-WAVE-R1/R1' bug that caused 2.4–3.8 drift on sister substrates) is NOT present; a dedicated
test (`test_channel_first_convention_pins_against_channel_last_drift`) proves the kernel does NOT match the
channel-last layout. The sin/skip/refine COMPOSITIONS all match. **No divergence found.** The
`pr95_hnerv_mlx.pixel_shuffle_2x_nhwc` docstring's "0.0 drift vs nn.PixelShuffle" claim was previously an
EXTERNAL anchor (sister D=Z6); it is now an in-tree regression guard.

**One ordering nuance (not a bug, a config faithfulness gap — see §2):** the skip-FREE `_UpBlockMLX` path is
`PixelShuffle(sin(w·conv))` (sin INSIDE shuffle); the skip-ON path and the PR95 reference are
`sin(w·(PixelShuffle(conv)+identity))` (sin AFTER shuffle+add). Both are internally correct; they are simply
two different decoder forms. The skip-ON path is the PR95 one.

---

## §2 CONFIG DEFECT LIST (audit point 2) — beyond #75's three

Evidence: `HinervConfig` (`src/tac/substrates/hi_nerv/architecture.py`), the shared harness
`src/tac/substrates/_shared/mlx_score_aware/bundle.py`, and the B1 inert run's OWN committed config snapshot
(`/Volumes/VertigoDataTier/pact/b1_229k_clean_.../training_artifact.json`).

| # | defect | location | evidence | severity |
|---|---|---|---|---|
| C1 | **`use_bilinear_skip = False` default** (M-arch: skip-free mean-field carrier) | `architecture.py:154` | default cfg; B1 used it | HIGH (the #68 M-arch root) |
| C2 | **`sin_frequency = 30.0` default** (SIREN w=30 spectral-bias trap on a skip-free feature map) | `architecture.py:121` | canonical-kernel docstring flags w=30 as the trap | HIGH |
| C3 | **`use_hierarchical_feature_grid = False` + `use_convnext_blocks = False`** (HiNeRV's defining grid-PE + ConvNeXt OFF → it's vanilla NeRV mislabeled HiNeRV) | `architecture.py:136,139` | default cfg | MEDIUM |
| C4 | **ALL scorer distillation weights default 0.0** in the shared harness | `bundle.py:519,527,540,541` | `distillation_weight=0.0`, `pose_distillation_weight=0.0`, `segnet_direct_live_distillation_weight=0.0`, `pose_direct_live_distillation_weight=0.0` | HIGH (the #75 M-loss root, confirmed at source) |
| C5 | **B1 ran 1 recon-MSE stage, not the 8-stage curriculum** | `training_artifact.json::config_snapshot` | `curriculum_stages=[1 stage]` name `..._score_aware_full` `loss_weights={"recon":1.0}`; `score_aware_loss_kwargs={}` | HIGH — corrects #75's "(d) B1 ran the score-aware curriculum" |
| C6 | **recon weight pinned 1.0 + dual-ascent slowly RAMPS the scorer weight from 0.0** | B1 telemetry `loss_components` | `active_loss_weight__recon=1.0` always; `dual_ascent_effective_loss_weight__..segnet..` 0.0→0.0027→0.07→0.37→1.0→1.6→2.4→2.98; pose 0.0→6.0 | HIGH — recon dominance + late scorer ramp |
| C7 | **AdamW (not Muon) in stages 1–7 + global-norm grad-clip to 1.0 fires 100% of steps** | `training_artifact.json` + telemetry | `optimizer_class=adamw`, `muon_active=false`, `pr95_muon_policy=faithful_stage8_only`; `grad_clip_applied=True` every sampled epoch; grad_norm 5.3e4→6.8e6 → scale ~1.5e-7 (#77) | HIGH (the #77 root) |
| C8 | **skip/refine path CANNOT export to an archive** — `use_bilinear_skip=True` raises `NotImplementedError` in the export layout | `mlx_renderer.py:7456-7462` | the M-arch fix is research-only recon-fit until export+oracle-parity lands | HIGH — a REAL wall on the #76 promotion path |
| C9 | **two non-equivalent scorer-weighting paths** (`score_aware_loss_kwargs` distillation_weight vs the `dual_ascent_*` Lagrangian) | bundle + harness | B1: `score_aware_loss_kwargs={}` yet dual-ascent applied seg weight | MEDIUM — confusing dual-control; the kwargs path was inert, the dual-ascent path ramped |

**The B1 inert mechanism, fully reconciled (corrects/extends #75):** the seg loss WAS in the loop (telemetry
`loss_form` cycles `ce_seg → tau_softplus → smooth_disagreement → l7_softplus`; the `dual_ascent` ramps the seg
effective weight to ~3.0 by ep3000), so #75's "scorer weight hardwired 0.0" is too strong — it is the
`bundle.py` *kwargs* path that defaults 0.0 (C4), while a SEPARATE dual-ascent path applied a slowly-ramping
weight. BUT the **seg metric never descends** (1.59 → 0.41 transient → 0.50 → 0.65 → 0.84, drifts UP at the
end) while recon=1.0 dominates and the grad is ill-conditioned (clip 100%, AdamW). So the NET effect IS the
#75 inert loop — the score-aware gradient never reduces argmax disagreement — but the precise cause is the
**recon-dominant + late-ramp-scorer + AdamW-clip-pathology + skip-free-arch** quartet, not a single
hardwired-0.0 weight. Pose DOES descend (271 → 3.84) because the pose target is a regression that AdamW can
chase; seg (argmax) does not, for the #68 reasons (mean-field collapse + margin-surrogate vs hard-argmax).

---

## §3 FINDINGS-TRUST TABLE (audit point 3)

Classification rule: **PROVEN-FRONTIER (TRUST)** = ran on the real frontier/PR95 ARCHIVE BYTES through the
#75-verified exact-eval (which reproduces PR95's 0.19871 bit-exact). **SUSPECT (our-stack)** = its conclusion
depended on OUR score-aware MLX training loop (the inert C4–C9 quartet) or the distill-from-frame pose model.

| finding | what it ran on | classification | re-validate on #76? |
|---|---|---|---|
| **#64** lossless stack | frozen frontier archive bytes, exact eval | **PROVEN-FRONTIER (TRUST)** | NO — frozen-byte fact, loop-independent |
| **#69** whole-tensor re-quant | frozen frontier int8 weights, 600-pair exact_evaluate | **PROVEN-FRONTIER (TRUST)** | NO — but note its reactivation (#78 QAT-in-loop) DOES need the fixed loop |
| **#71** Q* structural compression | frozen frontier weights, per-tensor exact-scorer ablation | **PROVEN-FRONTIER (TRUST)** | NO — post-hoc-on-frozen-weights is closed; its reactivation = score-domain RETRAIN, which needs #76 |
| **#72** lever-D margin residual | frozen frontier, exact_pair_scorer | **PROVEN-FRONTIER (TRUST)** | NO — receptive-field collateral is a real geometric fact |
| **#73** Dykstra legal-frame feasibility | exact_pair_scorer on the frontier manifold | **PROVEN-FRONTIER (TRUST)** | NO — feasibility geometry is loop-independent |
| **#54** cross-pair pose corrector | frontier FEC6 selector, exact_pair_scorer | **PROVEN-FRONTIER (TRUST)** | NO — selector saturation is a frozen-frontier fact |
| **#74** distill to smaller student | OUR torch KD trainer, distill-onto-teacher-FRAMES | **SUSPECT (partial)** | **YES** — the pose-capacity-WALL is FALSE (Quantizr); but #74's *trainer* did NOT use the inert MLX harness (it wired w_seg 0.5/w_pose 0.1 directly, torch) → the distill *measurement* is real for the case tested. The SUSPECT part is the GENERALIZATION to "all small nets" — re-validate with explicit-pose-storage (§4) |
| **#62** d_seg wall (small fresh-init conv) | OUR score-aware training (argmax-CE-on-GT) | **SUSPECT** | **YES** — same M-loss/M-arch/optimizer pathology; Quantizr's 88K holds d_seg 6.1e-4 |
| **B1 / hi_nerv d_seg≈0.50** | OUR inert MLX loop (C4–C9) | **SUSPECT** | **YES** — the canonical #76 target |
| **NeRV-fleet d_seg 0.5–0.71** (pact_nerv_vq, snerv path-B, etc.) | OUR shared MLX harness | **SUSPECT** | **YES** — shared-harness M-loss; #68 Tier-1 reactivation list |
| **#75** "our eval reproduces PR95 0.19871 bit-exact" | PR95 released bytes through exact eval | **PROVEN (TRUST)** | n/a — this is the trust anchor itself |

**Honest caveat on #74's authority:** #74's trainer (`tools/distill_smaller_student_from_frontier_teacher.py`)
is torch-based and wires nonzero scorer weights directly, so it is NOT the inert MLX harness — its
seg-KL-crushes-to-0.04 evidence is real. What is SUSPECT is the conclusion "the pose tube needs near-frontier
CAPACITY," because Quantizr proves an 88K net holds the tube **when pose is carried as explicit side-info**.
#74 tested only the "recover-pose-from-distilled-pixels" regime and correctly found IT fails; it
over-generalized to a capacity wall. Reactivation = §4.

---

## §4 QUANTIZR COUNTER-EVIDENCE on the pose-capacity-wall (audit point 4) — recipe MINED

**Source:** contest PR #55 (`Quantizr/Jimmy`, "quantizr (0.33)", head `e0b643b0`). Kept EXTERNAL/historical
(`[external:PR#55]`) — the contest evolved to ~0.19; 0.33 is NOT our score.

**Report (600 samples):** SegNet 0.00061113 · **PoseNet 0.00051010** · 299,970 bytes · **score 0.33**. Model
"just 88k params and 64kb when compressed."

**The architecture (`compress.py`, `JointFrameGenerator`):** NOT a pure HNeRV frame decoder. It is a
**conditional decoder whose INPUTS carry the per-pair signal explicitly**:
- `forward(mask2, pose6)` — takes the decoded MASK (from `mask.obu.br`, AV1) AND the **6-dim GT pose vector**
  (from `pose.npy.br`, brotli) as conditioning inputs.
- `SharedMaskDecoder` trunk (depthwise-separable `SepConv`/`SepResBlock`, GroupNorm, coord-grid input) →
  shared feature.
- `frame1_head` = `FiLMSepResBlock` **FiLM-conditioned on `pose_mlp(pose6)`** — the GT pose vector directly
  modulates frame1 so PoseNet reads the correct pose. `frame2_head` is a static head (no pose FiLM).
- Pose is EXTRACTED from the GT video via `posenet(...)["pose"][...,:6]` and **STORED** (600 pairs × 6 floats,
  brotli-compressed `pose.npy.br`).
- FP4 block-quant (block=32, codebook `pos_levels`, scales fp16) → 64kb decoder.
- **5-stage pipeline** (ANCHOR → FINETUNE → JOINT → QAT → final) with per-stage freeze control
  (`apply_freeze_state`); AdamW betas (0.9, 0.99); **EMA decay 0.99**; **`kl_on_logits` T=2.0** SegNet
  distillation; **eval-roundtrip** (interp-up → clamp/round → interp-down) baked into training.

**Why this disproves the capacity wall:** #74's tube probe perturbed the TEACHER's pixels and found d_pose
needs frame-RMSE < 3 (±5/255) — true *if you must recover pose from pixels*. Quantizr **never recovers pose
from pixels**: he carries the 6-dim pose as explicit side-info (~14.4 KB raw, brotli-compressed) and
FiLM-injects it, so the 88K net only has to render a frame whose PoseNet readout matches a pose it is HANDED.
That is a fundamentally cheaper rate/distortion split than the HNeRV-frame-decoder pose tube #73/#74 measured.
**The binding constraint is NOT decoder capacity — it is the pose REPRESENTATION (carry-explicitly-and-FiLM
vs reconstruct-from-pixels).**

**The #74 reactivation path #1 ("pose-frame decoupling: distill frame1 + carry frame0 as residual") is
exactly the right instinct — and Quantizr is the existence proof it works.** Generalize it: carry the 6-dim
pose explicitly + FiLM-condition (Quantizr's mechanism) instead of forcing the net to encode pose in pixels.

**Mined recipe (for #76/#78), kept EXTERNAL:** FiLM-on-pose conditioning (cheapest pose carrier) ·
depthwise-separable convs (88K params holds both terms) · `kl_on_logits(T=2.0)` SegNet distill · EMA 0.99 ·
eval-roundtrip in the inner loop · 5-stage freeze curriculum · FP4 block-32 quant · mask carried as AV1 (only
600 odd-frame masks) · pose carried as 6-float/pair side-info. Artifacts cached at
`.omx/tmp/quantizr_compress.py` + `.omx/tmp/quantizr_inflate.py` (rebuildable from PR#55; safe to delete).

---

## §5 WIRE-IN (Catalog #125)

1. **sensitivity-map — ACTIVE:** new prior — the d_seg≈0.5 plateau is a TRAINING-LOOP + ARCH-DEFAULT bug
   (C4–C9), the decoder MATH is bit-exact correct; the aiming surface is the loop fix, not the kernels.
2. **Pareto — ACTIVE:** new constraint — pose can be carried as explicit 6-float side-info + FiLM (Quantizr)
   → the pose tube is NOT a decoder-capacity wall; the rate cost is ~the brotli of 3600 floats, not
   near-frontier capacity. Re-opens #74 path-1 / #54 region-allocator at small bytes.
3. **bit-allocator — N/A:** audit, no archive emitted.
4. **cathedral autopilot — N/A:** research/advisory, non-promotable.
5. **continual-learning — ACTIVE:** reseed the V3 judge with (a) the decoder is PROVEN-correct (do NOT chase
   PixelShuffle bugs); (b) #74's pose-capacity-WALL is FALSIFIED by Quantizr's 88K → re-classify #74 from
   "capacity wall" to "pose-representation choice"; (c) the findings-trust split (frozen-byte = TRUST,
   our-training-loop = SUSPECT-pending-#76); (d) C8 export-blocker (the skip/refine fix can't archive yet).
6. **probe-disambiguator — RESOLVED:** "Is the d_seg plateau a PixelShuffle/decoder bug?" → **NO (decoder
   bit-exact).** "Does Quantizr's 88K disprove the pose-capacity-wall?" → **YES.** "Which findings are
   SUSPECT?" → #74/#62/B1/NeRV-fleet (our loop); #64/#69/#71/#72/#73/#54 = TRUST (frozen bytes).

---

## §6 NO-FAKE attestation + tests

- PixelShuffle drift `0.0` and full-decoder `rel 2e-7` are REAL `np.max(np.abs(mlx − torch_nn))` measurements,
  not derivations. The torch reference is a from-scratch `nn.Module` loaded with the SAME MLX weights
  (NHWC→NCHW transposed), so a structural bug would show drift of 2–200 (channel/shuffle/skip), not 1e-7.
- The config defects (C1–C9) are quoted from source line-cites + the B1 run's OWN committed config/telemetry
  (not paraphrase). C5 (1 recon stage) is a verbatim `curriculum_stages` read that CORRECTS #75's launch-manifest
  claim — flagged honestly, not hidden.
- Quantizr's 0.33 / d_pose 0.00051 is his PR#55 report verbatim, kept EXTERNAL.
- 21 behavior tests land (`src/tac/tests/test_hnerv_decoder_nn_pixelshuffle_parity.py`): bit-exact
  nn.PixelShuffle parity, bilinear parity, full-decoder end-to-end parity (×3 latents), RGB-range,
  skip-structure, channel-convention guard, + 3 audit-finding regression guards (skip-free default, scorer-
  weight-0.0 default, w=30-vs-w=1 skip ordering). `[macOS-CPU advisory]`; review-gate marked reviewed.

## CROSS-REFERENCES
`pr95_elephant_audit_20260610T185556Z` (#75 — corrected re C5/C9: B1's score-aware-loss-kwargs were empty +
dual-ascent ramped seg weight; net effect still inert) · `nerv_fleet_reactivation_and_arch_selection_20260610T192434Z`
(#68 — M-loss + M-arch confirmed at source: `bundle.py:519/527/540/541` + `architecture.py:121/136/139/154`) ·
`tilde_optimizers_for_inert_loop_20260610T193200Z` (#77 — AdamW + clip-100% confirmed in B1 artifact) ·
`distillation_smaller_student_20260610T191237Z` (#74 — pose-capacity-WALL FALSIFIED by Quantizr; its torch
trainer was NOT the inert MLX harness, so its distill measurement is real for the recover-from-pixels case) ·
`frontier_pointer_move_ledger_20260610` (the 7 no-moves: #64/#69/#71/#72/#73/#54 = PROVEN-FRONTIER TRUST) ·
`src/tac/local_acceleration/pr95_hnerv_mlx.py` (HNeRVDecoderMLX + canonical PixelShuffle) ·
`src/tac/substrates/hi_nerv/{architecture.py,mlx_renderer.py}` (the default config + skip-free UpBlock + C8
export-blocker) · `src/tac/substrates/_shared/mlx_score_aware/bundle.py` (scorer-weight 0.0 defaults).
**External:** Quantizr PR#55 `[external:github.com/commaai/comma_video_compression_challenge/pull/55]`.
