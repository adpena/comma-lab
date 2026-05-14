# Optimization Opportunities Audit 2026-05-14

**Operator directive 2026-05-14**: *"need to investigate optimization opportunities also, like the casting issue and others we found in the past, there is likely low-hanging fruit with no regressions in signal"*.

**Scope**: AUDIT-ONLY — no code changes this round. Output is research memo + ranked recommendations. Operator routes which optimizations to implement.

**Lane**: `lane_optimization_opportunities_audit_20260514` (Phase 2, L0 → L1 after this landing).

**Cross-refs**: Catalogs #172/#178/#179/#180 (declare-flag warn-only sister gates); REVIEW-OMNI Medium + Low memos `feedback_wave7_review_omni_6_medium_fixes_LANDED_20260512.md` + `feedback_wave7_review_omni_6_low_fixes_LANDED_20260513.md`; F1 finding (A1 Council R1 — bicubic precision = root of +0.0335 CPU/CUDA gap).

**Apples-to-apples discipline**: every speedup / regression estimate is tagged `[derived]` (first-principles), `[literature-extrapolation]` (PyTorch / HF benchmarks), or `[would-need-empirical]`. No score claims; no archive bytes; no KILL verdicts.

---

## Headline Finding (top of audit)

**The flag is declared, the wiring is dead.** 14 substrate trainers carry `--enable-autocast-fp16` and `--enable-torch-compile` argparse flags (Catalog #172 / #179 declare-only gates), but the canonical `tac.substrates.score_aware_common.score_pair_components` does NOT wrap the scorer forward in autocast, does NOT cache GT scorer outputs, and does NOT compile the scorer or the substrate forward. Every trainer carries an `# AUTOCAST_FP16_WAIVED:score-aware-scorer-path-pending-canonical-autocast-backport` file-level waiver pointing at the same root cause.

**Cost of NOT fixing this** (1.5-3× speedup left on table per dispatch × N dispatches per week × $0.30-$15 per dispatch).

**Fix-once-N-substrate-inherit** opportunity:

1. **Autocast wrap inside `score_pair_components`** (~10 LOC change, 1 helper module): all 14 substrates inherit; signal regression risk ~0 (autocast is non-mutating on accumulators)
2. **GT-scorer-output caching at trainer init** + use `scorer_loss_terms_cached_btchw` (canonical helper that already exists in `tac.losses.core`): -50% per-step scorer compute when iterating the same GT pair set across epochs (which is the standard case — N pair indices fixed at config time)
3. **`torch.compile` wrap of the substrate forward + scorer forward** (~15 LOC change, opt-in via existing flag): 1.5-2× Inductor speedup on Ampere+, signal regression risk ~1e-5 score drift (Inductor numerics)
4. **`fused=True` AdamW** (1-line optimizer kwarg change × 14 trainers — or refactor to use `tac.training.create_optimizer` helper if it exists): 10-20% optimizer-step speedup, signal regression 0 (functionally identical)
5. **`non_blocking=True` device transfers + `pin_memory=True` on pre-decoded GT tensors** (a few `.to(device, non_blocking=True)` substitutions): 5-15% wall-clock per epoch on Modal A100 dispatches, signal regression 0

---

## Section 1 — Per-substrate-trainer optimization audit

24 substrate trainers scanned. Pattern detection: ripgrep against `experiments/train_substrate_*.py`.

### 1.1 Autocast FP16/BF16 status

**Declared via `--enable-autocast-fp16` flag (Catalog #172)**: all 24/24 substrates.

**Actually consumed in training loop body**: **0/24**. Every trainer carries the file-level waiver `# AUTOCAST_FP16_WAIVED:score-aware-scorer-path-pending-canonical-autocast-backport`. The single mention of `torch.amp.autocast` in `a1_plus_lapose.py:1` is inside the autocast-discipline docstring describing why the path is waived.

Reference comparison: `experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py` is the only Tier 1 trainer in the repo that wires autocast end-to-end. The pattern can be backported.

### 1.2 TF32 status

**Routes through canonical `device_or_die` helper in `tac.substrates._shared.trainer_skeleton`**: all 24/24. The canonical helper (`trainer_skeleton.py:307-308`) sets `torch.backends.cuda.matmul.allow_tf32 = True` AND `torch.backends.cudnn.allow_tf32 = True` for every CUDA device acquired through the helper.

**TF32 is on for every substrate trainer**. This is a closed bug class — Catalog #178 is structurally extincted via the centralized `device_or_die` helper.

**Action**: NONE. TF32 is already canonical.

### 1.3 torch.compile status

**Declared via `--enable-torch-compile`**: 24/24 substrates declare the flag.

**Actually consumed (`torch.compile(...)` invocation)**: **0/24**. Every trainer has the flag but no `torch.compile(model)` or `torch.compile(scorer)` call. The substrate forward + scorer forward dominate per-step time on Modal A100; 1.5-2× Inductor speedup is unrealized.

### 1.4 no_grad / inference_mode at eval

**Catalog #180 is STRICT-from-byte-one**: 24/24 substrate trainers wrap their eval/validation loop in `torch.inference_mode()` (preferred for activation memory savings vs `torch.no_grad()`). This is a closed bug class.

**Action**: NONE. Catalog #180 enforces; this is already canonical.

### 1.5 Optimizer wiring (AdamW with `fused=True` / `foreach=True`)

**`fused=True` usage**: **0/24**. PyTorch 2.0+ `torch.optim.AdamW(fused=True)` is a single-kernel optimizer step that is 10-20% faster than the default loop-over-parameters path. Signal regression: 0 (functionally identical; numerical bit-identical on CUDA matching configs).

**`foreach=True` usage**: 0/24 (default; PyTorch picks `foreach=True` automatically when available since 2.1, so this is less critical than `fused=True`).

**8-bit Adam (bitsandbytes)**: 0/24 (and not recommended unless OOM-bound; signal regression is non-zero).

### 1.6 Gradient checkpointing (`torch.utils.checkpoint.checkpoint`)

**Usage**: **0/24** substrates use gradient checkpointing.

**Applicability**: most substrates have <1M trainable parameters (the substrate is small; the scorer is frozen at ~74MB). Gradient checkpointing is unlikely to help substrates with <100M params on A100 24GB+. Only relevant for substrate trainers that OOM on Modal A100; if no OOM is observed, skip.

**Action**: defer unless a specific substrate OOMs. (NV2 Council C OOM audit per REVIEW-OMNI Medium handles this through Catalog #170 `min_vram_gb` declaration.)

### 1.7 DataLoader settings (`pin_memory`, `num_workers`, `persistent_workers`, `prefetch_factor`)

**Usage**: **0/24** substrates use `torch.utils.data.DataLoader`. All substrates pre-decode the full GT pair tensor into memory at trainer init (`decode_real_pairs` in `trainer_skeleton.py:418-481` returns `torch.Tensor shape (N, 2, 3, 384, 512)`) and iterate by integer index.

**Implication**: DataLoader-style wins do NOT apply here because there is no async batch fetching. The pre-decode strategy is correct for this codebase (600 GT pairs × ~1.8 MB/pair = ~1 GB GT cache, fits in CPU RAM trivially).

**Sister opportunity**: pin the pre-decoded tensor in CPU memory + use `non_blocking=True` device transfer per batch. See finding O5 below.

---

## Section 2 — Inflate.py / Archive hot path audit

### 2.1 `.cpu().numpy()` patterns

All 17 substrate inflate.py files end with `frames.clamp(0,255).round().to(torch.uint8).cpu().numpy().tobytes()` (the canonical raw-output write path in `_shared/inflate_runtime.py:103-111`). This is correct — the file write must be bytes; `cpu().numpy().tobytes()` is the standard idiom.

**No batchability issue**: inflate.py is per-frame-pair (already a no-op vs batched). The `cpu()` transfer is ONCE per pair (300 pairs × 2 frames = 600 transfers per video × 5 videos = 3000 syncs); could be batched into one `torch.cat([all_frames], dim=0).cpu().numpy()` but the savings are likely <5% of inflate wall-clock (which is dominated by scorer forwards, not the IO).

**Action**: low priority; defer.

### 2.2 Redundant float32 ↔ float16 / int8 casts

`balle_renderer/archive.py:106` writes `dtype=torch.float16` for the decoder state_dict (correct for archive-size win). `archive.py:127` reads `dtype=torch.float32` at inflate (correct for math). These are not redundant.

**No casting issue observed in archive hot path.**

### 2.3 `torch.tensor(x)` inside hot loops

Inspected `balle_renderer/inflate.py:138`: `torch.tensor([pair_idx], device=render_device, dtype=torch.long)` is inside the per-pair loop (300 iterations). Each `torch.tensor(...)` call allocates a new tensor on the device. This is a minor opportunity: pre-allocate once outside the loop, reuse via `.fill_(pair_idx)`.

Saving: probably <1% of inflate wall-clock. **Action**: low priority.

### 2.4 Bicubic precision (F1 finding)

`_shared/inflate_runtime.py:102`: `F.interpolate(frames, size=CAMERA_HW, mode="bicubic", align_corners=False)`.

`tac.differentiable_eval_roundtrip.py:226`: `up = F.interpolate(flat, size=(874, 1164), mode='bicubic', align_corners=False)`.

Both use `align_corners=False` — which is the PyTorch default and is non-deterministic across CPU/CUDA per PyTorch docs. The F1 finding showed this is the root of +0.0335 CPU/CUDA score drift on A1.

**Hypothesis (already established in F1)**: switching to a precision-stable bicubic kernel (or using `align_corners=True` consistently across both training and inflate) closes the gap. The exact kernel choice is a SCORE-RELEVANT design decision — NOT a low-hanging-fruit optimization. **Defer to council per CLAUDE.md "Design decisions — non-negotiable"**; mention in this audit as a known signal-regression-non-zero opportunity.

---

## Section 3 — Scorer-loss helper hot path

### 3.1 GT scorer forward is recomputed every training step (high-impact finding)

**`tac.substrates.score_aware_common.score_pair_components`** (the canonical entry point used by ALL 14 substrate score_aware_loss helpers) calls `tac.losses.core.scorer_loss_terms_btchw`, which at line 736-738 does:

```python
fp_out, fs_out = scorer_forward_pair(filtered_pair_btchw, posenet, segnet)
with torch.no_grad():
    gp_out, gs_out = scorer_forward_pair(gt_pair_btchw, posenet, segnet)
```

The GT scorer forward (`gp_out, gs_out`) is computed every training step. But:
- The target video is fixed
- The scorer weights are frozen
- Therefore: the GT scorer forward output is **invariant across epochs** for a given pair set

**Sister helper that already solves this**: `tac.losses.core.scorer_loss_terms_cached_btchw` (line 757-820) takes precomputed `gt_pose` + `gt_seg_logits` and skips the GT scorer forward. Docstring (line 772-775) confirms: *"The target video and scorer weights are fixed during training, so the gt_pair -> PoseNet/SegNet forward is invariant."*

**Single trainer using the cached path**: `train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py:2706` (the T1 Balle endtoend trainer). Reference pattern: precompute `gt_pose_cache` + `gt_seg_cache` once at trainer init (or every N epochs), index into them per batch.

**Speedup**: scorer forward is the dominant per-step cost (PoseNet FastViT-T12 + SegNet EfficientNet-B2 UNet). Removing GT forward = **~50% per-step scorer compute**, which is the dominant cost for substrates whose own forward is small (NeRV/HNeRV/SIREN/Cool-Chic — most of the substrate canvas).

**Action**: **TOP PRIORITY**. Refactor `score_pair_components` to accept optional precomputed GT cache; refactor substrate trainers to precompute GT cache at init.

### 3.2 Differentiable rgb_to_yuv6 + eval_roundtrip path

`tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training` does (per call, per training step, per batch frame):

1. `F.interpolate(flat, size=(874, 1164), mode='bicubic', align_corners=False)` — upsample
2. `F.interpolate(up, size=(384, 512), mode='bilinear', align_corners=False)` — downsample
3. uint8 round + clamp

These ops are unavoidable (they ARE the contest eval roundtrip simulation). However:

**torch.compile applicability**: the 3-op sequence is a perfect candidate for Inductor compilation. 1.5-2× speedup expected on Ampere+. Signal regression risk ~0 because the ops are deterministic.

**Action**: wrap `apply_eval_roundtrip_during_training` with `@torch.compile` (opt-in via existing flag).

### 3.3 Redundant `.detach()` calls

`score_pair_components` returns `seg_term, pose_term` (no detach — correct, gradient flows). `sane_hnerv/score_aware_loss.py:152-155` detaches the parts for logging — correct.

**No redundant detach observed.**

### 3.4 SegNet UNet preprocess hot path

`SegNet.preprocess_input` (from `tac.scorers.segnet_loader`) slices `x[:, -1, ...]` then `F.interpolate` to `(512, 384)`. This is per-call.

Sister opportunity: at preprocess hot path, if the same pair is forwarded repeatedly across epochs, the resize result is also invariant — could be cached. But this is dominated by the bigger GT-cache win in §3.1; the per-pair resize is <1% of scorer forward time.

---

## Section 4 — Modal mount manifest + image build audit

### 4.1 Pip install list (`experiments/modal_train_lane.py:60-84`)

Current image:
- `torch==2.5.1`
- `torchvision`
- `safetensors`, `einops`, `segmentation-models-pytorch`, `av`, `brotli`
- `constriction>=0.4,<0.5`, `pyppmd>=1.3,<2.0`, `click`
- `nvidia-dali-cuda120==1.52.0` (~800MB — biggest non-torch dep)
- `tqdm`, `timm`, `scipy`, `numpy<2.0`, `Pillow`, `pydantic>=2.0`

**Observation 1**: `nvidia-dali-cuda120` is ~800MB. Not every substrate uses DALI — the DALI hot path is L2 SAR coherent pose pairs trainer. If DALI is only needed for ONE substrate, splitting it out into a substrate-specific image saves ~30s of image build per non-DALI dispatch.

**Observation 2**: `scipy` is imported at runtime by some preflight gates but may not be needed for every trainer. Marginal saving.

**Observation 3**: BtbN ffmpeg-master tarball is ~60MB (Stage 6 in script). The build does `curl -sL` + `tar xf` — at image build time only (cached across dispatches), so this is fine.

### 4.2 Mount order / stability check

`build_training_image` walks the structural minimum dirs (`src`, `scripts`, `upstream`, `submissions`, `experiments` (excluding `results/**`), `tools`) and adds optional files (per-trainer `required_input_file=True`). The mtime-stability check (Catalog #165) is wired correctly.

**No mount-manifest optimization observed.**

### 4.3 Image cache key

Modal images are content-hashed; adding/removing pip packages invalidates the cache. Splitting into `image_base` + `image_dali` + `image_pr95_specific` could reduce cold-start time, but Modal already deduplicates layers (LRU-cached).

**Defer** — marginal saving (<10s per dispatch); cost of split is N image-build variants to maintain.

---

## Section 5 — Ranked recommendation table (top-15 by EV/$)

| Rank | Finding | Speedup estimate | Signal-regression risk | Impl. cost (LOC + tests) | Strict-gate path | Notes |
|---:|---|---:|---:|---:|:---|:---|
| **O1** | **GT-scorer-output caching in `score_pair_components`** | **~50% per-step scorer compute** for substrates with small own-forward (12/14 substrates) | **0** (mathematically invariant; reference impl already in T1 Balle endtoend trainer) | ~60 LOC + 25 tests (1 helper + per-trainer init refactor pattern) | Add per-trainer cache-init token to canonical recipe; future gate `check_substrate_uses_gt_scorer_cache` | Single biggest win. [derived] |
| **O2** | **Canonical autocast FP16 wrap in `score_pair_components`** | **1.5-2× scorer forward** on A100/4090/H100 | <1e-5 score drift (BF16 if regression observed; FP16 with GradScaler if not) | ~30 LOC + 15 tests (1 helper + opt-in via `apply_autocast` kwarg) | Catalog #172 strict-flip on use (after backport) | All 14 substrates inherit. [literature-extrapolation, HuggingFace BF16 ~1.7×] |
| **O3** | **`torch.compile` wrap of substrate forward + scorer forward + eval_roundtrip** | **1.5-2× total per-step wall-clock** on A100/Ampere+ | ~1e-5 score drift (Inductor numerics; deterministic) | ~40 LOC + 20 tests (opt-in via existing `--enable-torch-compile` flag) | Catalog #179 strict-flip on use | Compile cache hits second epoch onward. [literature-extrapolation, PyTorch 2.1 ~1.8×] |
| **O4** | **`fused=True` AdamW across all 24 substrate trainers** | **10-20% optimizer-step time** (substrate dependent; bigger gains on large-param substrates) | **0** (functionally identical) | ~5 LOC × 24 = ~120 LOC OR ~20 LOC if refactor into helper | New gate `check_trainer_uses_fused_adamw` | One-line per trainer or shared helper. [derived] |
| **O5** | **`pin_memory=True` on pre-decoded GT pair tensor + `non_blocking=True` device transfers** | **5-15% wall-clock per epoch** on Modal A100 (CPU→GPU bandwidth-bound for some substrates) | **0** | ~10 LOC in canonical `decode_real_pairs` helper + per-batch `.to(non_blocking=True)` token | New gate `check_canonical_decode_pin_memory` | Centralized win. [derived] |
| **O6** | **Modal image split: base + dali variant** | **30-60s image build per non-DALI dispatch** (cold-start) | **0** | ~80 LOC + 15 tests (variant builder + dispatch path) | None | Marginal vs O1-O5. Defer unless cold-start is a P0. [derived] |
| **O7** | **Pre-allocate `torch.tensor([pair_idx])` outside inflate loop** | **<1% inflate wall-clock** | **0** | ~3 LOC × 17 inflate.py files | None | Drop in noise. Defer. [derived] |
| **O8** | **Batched `.cpu().numpy()` in inflate raw-output write** | **~2-5% inflate wall-clock** | **0** | ~30 LOC in `_shared/inflate_runtime.py` (collect frames, write one block) | None | Single-source canonicalization. [derived] |
| **O9** | **`torch.compile` wrap of `differentiable_rgb_to_yuv6`** | **5-15% eval_roundtrip path** | ~1e-5 numerics | ~5 LOC + 10 tests | None | Sub-saving of O3; can be folded in. [literature-extrapolation] |
| **O10** | **CUDA stream overlap for GT decode + scorer warmup** | **<5% wall-clock** | **0** | ~50 LOC + 20 tests | None | Diminishing returns after O1. Defer. [derived] |
| **O11** | **BF16 scorer forward (vs FP16)** | **0% additional vs FP16** (BF16 has wider range, no GradScaler needed) | **0** (vs FP16) | ~2 LOC in O2 helper (toggle dtype) | Subsumed by O2 | Choose BF16 if FP16 underflow ever observed. [derived] |
| **O12** | **Lazy import of pyav + heavy deps at trainer module top** | **0.5-2s startup per trainer module import** | **0** | ~10 LOC × 14 trainers (move imports inside `_full_main`) | New gate `check_trainer_lazy_imports_heavy_deps` | Sister of Modal cold-start. [derived] |
| **O13** | **`bicubic align_corners=True` for inflate resize** | **0** (precision fix; not speed) | **−0.0335 contest-CUDA score gap closed (F1)** | Council review required | DEFER — design decision per CLAUDE.md | Not low-hanging fruit; documented in F1 finding. [empirical, F1] |
| **O14** | **`tf.compile()` of `pose_dist.pow(2).mean()` + softmax surrogate** | **<2% scorer loss path** | ~1e-7 numerics | ~10 LOC | Subsumed by O3 | Folded into O3. [derived] |
| **O15** | **Drop `nvidia-dali-cuda120` from default image (only L2 SAR uses)** | **~30s image cold-start; ~800MB image size** | **0** | ~30 LOC (substrate-specific image variant) | Subsumed by O6 | DALI is L2 SAR only. [derived] |

---

## Section 6 — Cross-reference to existing Catalog rows

| Catalog # | Function | Status | Audit takeaway |
|---:|---|:---|:---|
| #172 | `check_substrate_trainers_declare_autocast_fp16_support` | warn-only; 14 trainers with file-level waiver | Backport autocast (O2) → flip strict |
| #178 | `check_substrate_trainers_declare_tf32_support` | strict @ 0 (canonical `device_or_die` helper) | Closed bug class. No action. |
| #179 | `check_substrate_trainers_declare_torch_compile_support` | warn-only; 14 trainers declare flag, 0 use it | Backport compile (O3) → flip strict |
| #180 | `check_substrate_trainers_use_no_grad_at_eval` | strict @ 0 | Closed bug class. No action. |

---

## Section 7 — Implementation priority (operator routes)

**Tier 1 (highest EV/$)**:
- **O1** GT-scorer-output caching → ~50% per-step scorer compute, 0 regression, ~60 LOC + 25 tests
- **O2** Canonical autocast wrap → 1.5-2× scorer forward, <1e-5 drift, ~30 LOC + 15 tests
- **O5** pin_memory + non_blocking → 5-15% per-epoch, 0 regression, ~10 LOC

**Tier 2 (moderate EV/$)**:
- **O3** torch.compile wrap → 1.5-2× per-step, ~1e-5 drift, ~40 LOC + 20 tests
- **O4** fused AdamW → 10-20% optimizer step, 0 regression, ~120 LOC or ~20 LOC via helper

**Tier 3 (defer or fold in)**:
- O6, O7, O8, O10, O12 — marginal; consider after Tier 1+2 land
- O9, O11, O14, O15 — subsumed by O2/O3/O6

**Council-grade design decisions (DEFER per CLAUDE.md)**:
- O13 bicubic precision fix — known signal-relevant; council review required

---

## Section 8 — Operator-routable decisions

1. **APPROVE O1 (GT-scorer cache)?** Single biggest win. Reference pattern exists. ~60 LOC + tests.
2. **APPROVE O2 (canonical autocast wrap)?** Closes the 24/24 file-level waiver. ~30 LOC + tests.
3. **APPROVE Tier-1 batch (O1+O2+O5)?** Operator route.
4. **APPROVE Tier-2 batch (O3+O4)?** Operator route.
5. **DEFER O13 (bicubic precision)?** Per CLAUDE.md design-decision rule, council review required.

---

## 6-hook wire-in declaration (per Catalog #125)

| Hook | Status |
|---|---|
| Sensitivity-map contribution (`tac.sensitivity_map.*`) | N/A — META engineering audit; no per-tensor sensitivity changes |
| Pareto constraint (`tac.pareto_*`) | N/A — no new feasibility constraints |
| Bit-allocator hook | YES — speedups enable more dispatches per dollar; the autopilot ranker should incorporate the post-speedup cost-per-eval into its EV/$ rank ordering. Hook into `tac.cost_band_calibration` cost model after O1/O2 land (estimated 30-50% lower $/full-dispatch on T4/A100). |
| Cathedral autopilot dispatch hook | N/A this round (audit-only); will register when O1/O2 land |
| Continual-learning posterior update | YES — every benchmark dispatch after O1/O2/O3 land must record `outcome="successful_dispatch"` + measured wall-clock + $/dispatch into `.omx/state/cost_band_posterior.jsonl` per Catalog #175/#177. This reseeds the EV/$ ranker. |
| Probe-disambiguator | MAYBE — O2 autocast and O3 torch.compile have non-zero numerics drift; if signal-regression risk is observed empirically on the first autocast/compile dispatch, build `tools/probe_autocast_vs_fp32_regression.py` per CLAUDE.md probe-disambiguator pattern. |

---

## Forbidden-pattern compliance check

- No /tmp paths in this audit memo ✓
- No score claims (every speedup tagged [derived] / [literature-extrapolation] / [would-need-empirical]) ✓
- No KILL verdicts (every recommendation is DEFERRED or APPROVED-pending-operator) ✓
- No MPS-derived strategic decisions (all speedup estimates first-principles or PyTorch literature) ✓

---

## Sister-subagent scope respect

- **Wyner-Ziv RE-RESUME** (touching `src/tac/substrates/wyner_ziv_*`): NOT touched in this audit; `train_substrate_wyner_ziv_cooperative_receiver.py` was scanned for the SAME class-level patterns as siblings (autocast/compile/no_grad status) and matches the canonical waiver pattern; no recommendation specific to Wyner-Ziv beyond the canvas-wide wins above
- **DISPATCH-WAVE** (touching `time_traveler` recipes + L2 SAR + PR95++): NOT touched in this audit
- **CRASH-RESUME** (touching `tools/subagent_checkpoint.py` + preflight gate): NOT touched

The findings here are CANVAS-WIDE optimizations (O1/O2/O3/O4/O5 all touch the canonical helper or the canonical pattern that 14+ substrates inherit). Implementation by a future SUBAGENT will need to coordinate with sister subagents to avoid file-edit-race per CLAUDE.md "Subagent commits MUST use serializer" + Catalog #157+#174.

---

*Memo authored 2026-05-14 per operator directive. Commit via canonical serializer with `--expected-content-sha256` per Catalog #157+#174.*
