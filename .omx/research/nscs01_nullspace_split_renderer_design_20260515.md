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
