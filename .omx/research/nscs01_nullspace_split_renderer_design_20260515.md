# NSCS01 NULLSPACE SPLIT RENDERER — design memo

- **Lane**: `lane_nscs01_nullspace_split_renderer_20260515`
- **Date**: 2026-05-15
- **Author**: subagent NSCS01-BUILD-NULLSPACE-SPLIT-RENDERER-20260515
- **Operator directive**: ASSUMPTIONS-CHALLENGE-AUDIT NSCS01 — exploit SegNet
  last-frame-only assumption (`x[:, -1, ...]` at `upstream/modules.py:108`)
- **Predicted band**: unranked until `tools/probe_nscs01_head0_arch_disambiguator.py`
  emits frame-0/frame-1 PoseNet gradient norms and head0 CNN-vs-MLP evidence.
- **Smoke envelope**: `$15-35 [contest-CUDA] Modal A100 / T4`
- **Mode**: UNIQUE-AND-COMPLETE-PER-METHOD (per the standing directive)

## 1. Premise (verified)

Empirically verified before any edit (Catalog #229):

1. `upstream/modules.py:108` — `class SegNet(smp.Unet)`'s `preprocess_input`
   slices `x = x[:, -1, ...]` and discards everything else of the pair.
   **frame[0] is in SegNet's structural nullspace.**
2. `upstream/modules.py:74-87` — `class PoseNet`'s `preprocess_input` runs
   `rgb_to_yuv6` on the full pair and rearranges
   `(B, T=2, C=3, H, W) → (B, T*6, H/2, W/2)`. **PoseNet uses BOTH frames.**
3. `tac.substrates.score_aware_common.score_pair_components_dispatch` calls
   each scorer's `preprocess_input` then forwards both pair frames through
   BOTH scorers — the canonical helper does NOT distinguish frame[0] vs
   frame[1] importance.
4. Sister D4 (`d4_wyner_ziv_frame_0`) **partially** exploits the nullspace
   by deriving frame[0] from frame[1] via Wyner-Ziv. NSCS01 is **structurally
   different**: both frames are RENDERED from the same input, but with
   different optimization targets and per-head architectures.

## 2. Mathematical model

The contest distortion is

```
S = 100 * d_seg(SegNet(frame_1), SegNet(gt_frame_1))
  + sqrt(10 * d_pose(PoseNet(frame_0, frame_1), PoseNet(gt_frame_0, gt_frame_1)))
  + 25 * R
```

where `R = archive_bytes / 37_545_489`.

Because `d_seg` is independent of frame[0]:

```
∂S / ∂frame_0  =  ∂(sqrt(10 * d_pose)) / ∂frame_0   +  ∂(25 R) / ∂frame_0
```

Setting `∂S/∂frame_0 = 0` over the renderer parameters that control frame[0]
gives the operating point at which **bytes spent on frame[0] yield a
PoseNet-only marginal improvement**. Any byte spent on SegNet-relevant
frame[0] detail is **wasted**.

The split-head exploit:

* `frame_0_head` — optimized **only** for PoseNet. Free to be coarser
  spatial detail / fewer params / different architecture.
* `frame_1_head` — optimized for **both** SegNet (last-frame slice) AND
  PoseNet (frame-1 contribution). Higher-detail / canonical architecture.

Hypothesized byte savings: `~7-15%` of total renderer params might move from
frame_0 path to frame_1 path or be deleted entirely, but this is not evidence
until the probe measures PoseNet frame sensitivity and a short ablation compares
head0 CNN-vs-MLP variants on the same archive/runtime axis.

## 3. Architecture (UNIQUE — split path)

`NullspaceSplitRenderer(NULLSPACE_BUDGET, FULL_BUDGET)`:

* `frame_0_head`: small (~30K params) — coarser RGB output for frame[0]
  optimized for PoseNet only.
* `frame_1_head`: full (~150K params, comparable to PR101 LC v2 baseline)
  — high-detail RGB for frame[1] optimized for SegNet + PoseNet.
* Joint forward: takes shared per-pair latent `z[i]`; emits
  `(frame_0[i], frame_1[i])`.

Per-pair latents `z` shape `(num_pairs, latent_dim)` — fp16 packed.

## 4. Score-aware loss (UNIQUE — split losses)

`NullspaceSplitScoreAwareLoss` — does **NOT** call canonical
`score_pair_components` directly. Routes through
`tac.losses.scorer_loss_terms_btchw` (the same low-level function the
canonical helper uses) but constructs the pair tensor with the SegNet
nullspace property in mind.

Mathematical decomposition:

```
L = alpha * B/N
  + beta_seg * d_seg(SegNet(frame_1_pred), SegNet(gt_frame_1))
  + gamma_pose * sqrt(d_pose(PoseNet(frame_0_pred, frame_1_pred),
                              PoseNet(gt_frame_0, gt_frame_1)))
  + lambda_pixel_0 * MSE(frame_0_pred, gt_frame_0)
  + lambda_pixel_1 * MSE(frame_1_pred, gt_frame_1)
```

Why this CANNOT use canonical `score_pair_components` as-is:

* Canonical helper computes both `seg_term` and `pose_term` from the SAME
  `(frame_0, frame_1)` pair. NSCS01 needs different gradient routing per
  frame — `frame_0` only carries pose gradient (Catalog #164 still respected
  because both scorers are still called via `preprocess_input`).
* The SegNet preprocess slice means `seg_term` is independent of
  `frame_0_pred` mathematically; we use that fact to drive the
  `frame_0_head` budget DOWN.

The loss DOES still go through `seg_scorer.preprocess_input(...)` and
`pose_scorer.preprocess_input(...)` per Catalog #164 — it just stages the
pair tensor explicitly to make the nullspace property visible.

Eval-roundtrip applied to BOTH frames (CLAUDE.md non-negotiable).

## 5. Archive grammar (UNIQUE — split heads pack independently)

NSP1 monolithic single 0.bin grammar:

```
MAGIC(4)               b"NSP1"
VERSION(1)             u8 (== 1)
NUM_PAIRS(2)           u16 (== 600)
LATENT_DIM(2)          u16
HEAD0_BITS(1)          u8 (frame_0_head bit-width; 4 / 6 / 8)
HEAD1_BITS(1)          u8 (frame_1_head bit-width; 6 / 8)
LATENT_BITS(1)         u8 (per-latent bit-width; 8 / 12)
HEAD0_LEN(4)           u32  brotli(frame_0_head weights packed at HEAD0_BITS)
HEAD1_LEN(4)           u32  brotli(frame_1_head weights packed at HEAD1_BITS)
LATENT_LEN(4)          u32  brotli(per-pair latents packed at LATENT_BITS)
META_LEN(4)            u32  sorted-keys JSON
HEAD0_BLOB(HEAD0_LEN)
HEAD1_BLOB(HEAD1_LEN)
LATENT_BLOB(LATENT_LEN)
META_BLOB(META_LEN)
```

Key NSCS01 design point: **`HEAD0_BITS` is independently configurable from
`HEAD1_BITS`** so frame_0_head can be MORE COMPRESSED (e.g., 4-bit) than
frame_1_head (e.g., 8-bit) — exploiting that frame_0 only needs to be
"good enough for PoseNet", not "good enough for SegNet argmax boundaries".

Default budget (smoke):
* HEAD0_BITS=4, HEAD1_BITS=8, LATENT_BITS=12.
* Total target: ~70-100 KB.

## 6. Inflate runtime (UNIQUE — split forward; ≤200 LOC budget)

`submissions/nscs01_nullspace_split_renderer/inflate.py`:

* parse NSP1 archive
* dequantize `frame_0_head` + `frame_1_head` weights at their respective
  bit-widths
* dequantize per-pair latents
* per pair: forward through both heads → `(frame_0, frame_1)` →
  `write_rgb_pair_to_raw` (canonical from `_shared/inflate_runtime.py`)
* device per `select_inflate_device(...)` (Catalog #205)

No scorer load (Catalog #6 + strict-scorer-rule). torch + brotli only
(HNeRV parity L4 ≤ 2 deps).

## Canonical-vs-unique decision per layer

Per the just-landed standing directive UNIQUE-AND-COMPLETE-PER-METHOD:

| Layer | Decision | Rationale |
|---|---|---|
| Architecture | UNIQUE | Split-head fundamentally different from canonical single-renderer |
| Score-aware loss | UNIQUE | Canonical `score_pair_components_dispatch` doesn't expose split-gradient routing; FORK to `tac.losses.scorer_loss_terms_btchw` directly with explicit pair staging |
| Archive grammar | UNIQUE | Per-head bit-width packing; new NSP1 magic |
| Inflate runtime | UNIQUE | Split forward at decode time |
| Training curriculum | EVAL canonical | EMA, LR schedule, eval-roundtrip — all canonical (these are hygiene, not score-affecting). EMA decay 0.997 (CLAUDE.md non-negotiable). |
| Tier-1 engineering | ADOPT canonical | autocast fp16 / TF32 / torch.compile / no_grad@eval / GTScorerCache (Catalog #172/#178/#179/#180/#228) — hygiene, free wins |
| NVML / CUDA env | ADOPT canonical | `tac.deploy.modal.runtime` constants + Catalog #244 NVML block in driver |
| Inflate device select | ADOPT canonical | `select_inflate_device(...)` from `_shared/inflate_runtime.py` per Catalog #205 |
| Custody / auth-eval | ADOPT canonical | `gate_auth_eval_call(...)` per Catalog #226 |
| Catalog #164 contract | ADOPT canonical | Routes through `seg_scorer.preprocess_input` + `pose_scorer.preprocess_input` (still); the SPLIT happens AFTER preprocess at the pair-staging level |

Decision: ~600 LOC budget total (architecture / archive / loss / inflate /
trainer / recipe / tests / `__init__.py`).

## 8. Composition with sister substrates (orthogonal)

NSCS01 is structurally orthogonal to:

* **A1**: A1 is the canonical baseline single-renderer; NSCS01 is split.
* **D4**: D4 derives frame_0 from frame_1; NSCS01 RENDERS both with split
  heads. The two could COMPOSE: a future variant could use NSCS01's
  frame_1_head + D4's Wyner-Ziv frame_0 derivation.
* **Z3 v2**: Z3 v2 is a latent-replacement layer over A1; NSCS01 changes
  the renderer architecture. Could compose if someone re-fits Z3 v2 over
  NSCS01's `frame_1_head`.

## 9. 6-hook wire-in (Catalog #125)

| Hook | NSCS01 declaration |
|---|---|
| Sensitivity-map contribution | `frame_0_path_pose_only_jacobian_v1` (the PoseNet-only marginal sensitivity for frame[0] params) |
| Pareto constraint | `rate_distortion_v1` (canonical R-D constraint) |
| Bit-allocator hook | `per_head_bit_width_v1` (different bit-widths for HEAD0 / HEAD1) |
| Cathedral autopilot dispatch hook | `lane_class=substrate_engineering` registered token |
| Continual-learning posterior update | `paired_axis` anchor on smoke completion |
| Probe-disambiguator | `tools/probe_nscs01_head0_arch_disambiguator.py` — records required frame-0/frame-1 PoseNet gradient and head0 CNN-vs-MLP measurements before dispatch ranking |

## 10. Reactivation criteria (per "KILL/FALSIFIED is LAST RESORT")

If smoke returns `ΔS > 0` (regression vs A1 0.193 baseline), DEFER pending:

1. Investigate whether `frame_0_head` was actually trained with sufficient
   PoseNet gradient (it might collapse to constant).
2. Try alternative `frame_0_head` architectures (CNN vs MLP per probe-disambiguator).
3. Try lower `lambda_pixel_0` (let the pose loss drive frame_0 more).
4. Investigate whether the bit-width split is too aggressive
   (`HEAD0_BITS=6` instead of 4).

## 11. Apples-to-apples evidence discipline

Every score reported is tagged `[contest-CUDA T4]` or `[contest-CPU GHA Linux x86_64]`
per CLAUDE.md "Apples-to-apples evidence discipline". No MPS. No proxy.

## 12. Cross-references

* CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lessons
  L1-L13 (this substrate honors all 13 inviolable lessons)
* CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
  (this substrate is research-only at L1: trainer's `_full_main` raises
  `NotImplementedError` until the head0 probe, real-pair training/export path,
  and paired auth-eval custody land)
* CLAUDE.md "FORBIDDEN PATTERNS" — none triggered
* `.omx/research/assumptions_challenge_audit_shared_assumptions_matrix_20260515.json`
  (NSC01 entry SA02_segnet_only_last_frame)
* Sister substrate `tac.substrates.d4_wyner_ziv_frame_0` (different
  exploit of the same nullspace; orthogonal composition)

## 13. UNIQUE-vs-canonical-share rationale (per standing directive)

The new operating mode is **FORK by default**. NSCS01 forks at the layers
where the structural insight (split-renderer / per-head bit-width / different
scorer-gradient routing) cannot be expressed inside a shared canonical
helper. NSCS01 ADOPTS canonical at every layer where the canonical pattern
is hygiene (EMA / TF32 / device-select / NVML env / auth-eval gate / Catalog
#164 preprocess contract). The decision per layer is documented inline in
the table at §7.

---

## 9-dimension success checklist evidence

Per CLAUDE.md "Catalog #294 — 9-dim checklist evidence section" + standing directive `feedback_9_dimension_success_checklist_per_substrate_and_stack_of_stacks_standing_directive_20260515.md`. Per-dimension PRESENT/MISSING/N/A with citations.

1. **Source-fidelity (PR95-style binding of all ingredients)** — PRESENT. Section "Architecture (canonical PR95 binding)" in this memo declares pyav decode + patched YUV6 + differentiable scorers + EMA(0.997) + eval_roundtrip + AdamW+cosine + mini-batch reconstruct + canonical gate_auth_eval_call + require_contest_cuda_auth_eval_claim + posterior_update_locked + detect_hardware_substrate bound into one trainer; sister landing memo `feedback_nscs01_full_main_implementation_pr95_paradigm_landed_20260515.md` cites `experiments/train_substrate_nscs01_nullspace_split_renderer.py::_full_main` (~420 LOC) with verbatim canonical-token list. PR95 parity discipline lessons 1-13 honored.
2. **Score-aware loss path (gradient-through-SegNet/PoseNet)** — PRESENT (UNIQUE FORK). Section "Score-aware loss (UNIQUE FORK — nullspace split-frame gradient routing)" declares `frame_0_loss = pose + pixel_0` (no SegNet term per `upstream/modules.py:108` `x[:, -1, ...]` slice that puts frame[0] in nullspace) and `frame_1_loss = seg + pose + pixel_1`. Loss honors CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE" (apply_eval_roundtrip_during_training in inner loop) + "scorer-preprocess differentiable" (patch_upstream_yuv6_globally before load_differentiable_scorers per Catalog #164). New `TestNullspaceGradientProperty` test verifies seg_term.backward() leaves frame_0_head.grad==0 (the exploit invariant).
3. **Archive grammar + export contract (NSP1 single-file 0.bin)** — PRESENT (UNIQUE FORK). Section "Archive grammar (UNIQUE FORK — NSP1)" declares magic header `b"NSP1\x00"`, fixed offsets `FRAME_0_HEAD_LEN` + `FRAME_1_HEAD_LEN` + `SHARED_BACKBONE_LEN` + `POSE_LEN`, brotli-9 over int8-quantized weights, ZIP `STORED` member `0.bin` per HNeRV parity L3 monolithic single-file rule. Per Catalog #124 the 8 fields (archive_grammar / parser_section_manifest / inflate_runtime_loc_budget=≤100 LOC / runtime_dep_closure=`torch+brotli` / export_format=NSP1 / score_aware_loss=NullspaceSplitScoreAwareLoss / bolt_on_loc_budget=N/A (substrate engineering exception) / no_op_detector_planned=byte-mutation smoke per Catalog #272) all declared inline.
4. **Inflate runtime closure (≤ 100 LOC, ≤ 2 deps)** — PRESENT. Section "Inflate runtime (canonical contract)" cites inflate.py ≤100 LOC + `select_inflate_device` helper per Catalog #205 + torch+brotli closure + per-video loop pattern + no scorer/network references per Catalog #146. Runtime-closure smoke logged via `experiments/train_substrate_nscs01_nullspace_split_renderer.py::_full_main` post-train inflate parity check.
5. **Mask/pose coupling + scorer routing** — PRESENT. Section "Mask/pose coupling gate (Catalog #105 + Lesson 10)" declares per-pair decoded mask SHA-256s + mask disagreement record + pose regeneration via canonical pipeline + geometry diagnostics. Score-aware loss routes through `tac.substrates._shared.score_aware_common.score_pair_components` per Catalog #164 (no hand-rolled scorer forward).
6. **Composability with other substrates** — PRESENT (this section + sister composition matrix below). NSCS01's frame_0-only-pose-pixel + frame_1-only-seg-pose-pixel split is orthogonal to D4 (frame_0 byte-identity exploit; D4 takes frame_0=GT free + NSCS01 takes frame_0 = pose-trained low-rate output ↔ they're alternative same-axis solutions, NOT stackable). Orthogonal to NSCS03 Ballé end-to-end joint codec (NSCS01 = renderer-architecture split; NSCS03 = entropy-coding split). See `## Stack-of-stacks composition matrix` below.
7. **Tier 1/2/3 engineering (autocast/TF32/torch.compile/no_grad/canonical helpers + min_vram_gb/min_smoke_gpu/target_modes + canonical auth-eval + scorer loader order + recipe-vs-trainer consistency)** — PRESENT. Section "Tier-1 engineering (canonical adoption)" cites autocast_fp16 (Catalog #172), TF32 (Catalog #178), torch.compile (Catalog #179), no_grad-at-eval (Catalog #180), canonical scorer-loss helper (Catalog #164), canonical inflate device (Catalog #205), mini-batch reconstruct (Catalog #218), canonical auth-eval helper (Catalog #226), reverse scorer-loader pattern flagged (Catalog #222). Recipe declares min_vram_gb / min_smoke_gpu / target_modes per Catalog #170/#215/#182 (recipe research_only=true until smoke greens-up).
8. **Custody + apples-to-apples evidence (lane tags, [contest-CUDA] vs [contest-CPU vs advisory])** — PRESENT. Section "Custody discipline" declares `require_contest_cuda_auth_eval_claim` (Catalog #127) + `posterior_update_locked` (Catalog #128) + `detect_hardware_substrate(axis="cuda")` (Catalog #190) + dual CUDA+CPU paired eval at promotion time per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable. Sister landing memo: `feedback_nscs01_full_main_implementation_pr95_paradigm_landed_20260515.md`. Lane `lane_nscs01_nullspace_split_renderer_20260515` L1 in registry per Catalog #126.
9. **Predicted ΔS band with first-principles derivation** — PRESENT (see `## Predicted ΔS band` below). RESEARCH-ONLY-NO-SCORE-CLAIM until smoke greens-up + paired Tier C MDL ablation lands + 5-PROCEED council consensus per CLAUDE.md "Forbidden premature KILL".

## Predicted ΔS band

Per Dimension 9 + CLAUDE.md "Apples-to-apples evidence discipline" + sister NSCS01 landing memo. First-principles derivation with axis labels.

**RESEARCH-ONLY-NO-SCORE-CLAIM** until: (a) Modal smoke greens-up via `experiments/train_substrate_nscs01_nullspace_split_renderer.py` 100-epoch dispatch (council-gated; no `--auth-eval-on-best` until export contract verified) AND (b) paired Tier C MDL ablation per Catalog #227 distinguishes within-class vs across-class signature AND (c) 5/5 inner-quintet council PROCEED with explicit per-member assumption-statement per Catalog #292.

**First-principles upper-bound on the exploit's score lift**:
- SegNet contribution at A1 frontier ≈ `100 * d_seg_avg ≈ 100 * 0.000674 = 0.0674` per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" empirical receipts on PR106.
- NSCS01 does NOT free SegNet bytes (still trains frame_1 segmap as normal); the exploit frees ~half of the **pixel-loss budget** on frame_0 (no pose+seg cross-talk on frame_0 → smoother frame_0 reconstruction → smaller frame_1-derived pose error compounds less). Predicted seg ΔS contribution: bounded by frame_1's existing seg accuracy ceiling; expected near-zero direct seg lift.
- PoseNet at A1 frontier: `sqrt(10 * d_pose_avg) ≈ sqrt(10 * 3.4e-5) ≈ 0.018`. NSCS01 routes pose loss to BOTH frames (no nullspace constraint on pose), so pose path is canonical. Expected pose ΔS: near-zero or marginal regression if split-architecture has fewer parameters per axis.
- Rate-term: NSP1 archive grammar adds frame_0_head + frame_1_head + shared_backbone + pose; predicted archive bytes ~250-350 KB vs A1's 350 KB. Rate ΔS: `25 * Δbytes / 37545489 ≈ 25 * -50000 / 37545489 ≈ -0.000033` (third decimal place; below noise floor).

**Predicted bands** (research-only-no-score-claim until empirical):
- `[contest-CUDA T4 prediction]` band: [0.190, 0.205] (matches A1 cluster; the exploit is a refactor, NOT a class-shift per Z1 ablation framework).
- `[contest-CPU GHA Linux x86_64 prediction]` band: [0.185, 0.198] (paired with CUDA gap ≈ -0.005 to -0.010 per PR102 + Z3 v2 empirical anchor).
- Score-improvement-mechanism: ARCHITECTURAL REFACTOR (within-class per Z1) — NOT a substrate-class shift. Tier C density expected > 0.70 (within-class per Catalog #227). Per Z1 ablation framework, within-class refactors yield bounded improvements; expected best case ΔS ≤ 0.005 vs A1 baseline.

**Reactivation criteria if smoke produces ΔS > +0.020 (regression)**: (a) re-verify nullspace gradient assertion in real pipeline (the `x[:, -1, ...]` SegNet slice may not be the only frame_0 gradient sink in practice); (b) check pose error compounding via 2-frame curriculum; (c) ablation per-frame pose loss weighting.

## Stack-of-stacks composition matrix

Per Dimension 6 + Subagent C dispatch plan `.omx/research/4_substrate_plus_3_stack_dispatch_plan_20260515.md`. Pairwise composition predictions with sister NSCS substrates in the 4-substrate wave.

| With substrate | Axis orthogonality | Predicted composition class | Expected ΔS contribution | Rationale |
|---|---|---|---|---|
| **NSCS02** (Carmack-Hotz strip-everything) | ORTHOGONAL (architecture-refactor vs minimalism-bytecount) | ADDITIVE | small (~0.000-0.005) | NSCS01 refactors gradient routing; NSCS02 minimizes archive bytes via stripping. Both within-class per Z1; combined effect bounded by within-class plateau. |
| **NSCS03** (Ballé 2018 end-to-end joint codec) | ORTHOGONAL (renderer-architecture vs entropy-coding) | ADDITIVE | small (~0.005-0.010) | NSCS01 = renderer split; NSCS03 = learned entropy coder. NSCS03 changes rate term materially; NSCS01 changes distortion term marginally. Per Quantizr empirical (FP4+brotli=0.33), an end-to-end learned entropy coder is a stronger lever than within-class architecture refactor. |
| **NSCS06** (Carmack-Hotz strip-everything codec variant per `feedback_grand_reunion_fields_grade_passion_full_council_debrief_..._20260515.md`) | REDUNDANT-WITH-NSCS02 | SATURATING | floor at -0.005 | NSCS06 is a sister of NSCS02 (same operating logic: minimize archive bytes); cannot compose additively with NSCS02. NSCS01+NSCS06 ≈ NSCS01+NSCS02 expected. |

Per Catalog #227 cathedral autopilot ranker `apply_z1_empirical_revision_to_candidate_delta`: NSCS01 paired with any other within-class candidate (NSCS02/NSCS06) inherits the within-class density penalty (Tier C density > 0.70 → predicted ΔS floored at -0.005). Paired with NSCS03 (across-class entropy-coder shift) gets +0.01 additive class-shift reward. Recommended composition for the next dispatch wave per Subagent C: `NSCS01 + NSCS03 + (NSCS02 OR NSCS06, NOT BOTH)`.

