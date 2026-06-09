# Reference-carrier comparison — patch-our-sketch vs vendor-PR95-HNeRV vs promote-SNeRV vs promote-pact_nerv_vq

Date: 2026-06-09
Author: Claude Tranche-4 / Subagent-A reference-fidelity comparison analyst (READ-ONLY; no carrier/upstream edits, no
training launched, no commit to any carrier source). Decision + recommendation memo.

Axis discipline: EVERY numeric here is tagged. The only score authority is exact `upstream/evaluate.py` on
`[contest-CPU]` (Linux x86_64) or `[contest-CUDA]` (T4). All fidelity verdicts in this memo are **mechanism-only**
/ `[macOS-CPU advisory]` — NO score claims, NO promotion, NO rank/kill. PSNR is `[advisory only]` and is never a
score; `PSNR != d_seg` (the lab's documented lesson). Per CLAUDE.md "Forbidden premature KILL": nothing here KILLS
any paradigm; it classifies the *current carrier implementations* at the implementation level (Catalog #307
IMPLEMENTATION-LEVEL) and ranks the cheapest path to ONE mechanism-complete evaluator-cell carrier.

Status: `research_only=true` / `mechanism_update_eligible`. This memo directs the next build; it does NOT touch the
score roadmap (only a contest-axis `exact_evaluate` row does).

Sisters (read in full; this memo verifies their claims at file:line rather than trusting them):
`deep_hinerv_snerv_fidelity_review_vs_evaluate_py_20260609.md` (HiNeRV manifest, H1-H5/F1-F6) ·
`snerv_all_vehicles_fidelity_review_vs_evaluate_py_20260609.md` (cross-vehicle Mistake A+B) ·
`b1_f1_bilinear_skip_canonical_primitive_landed_20260609.md` (the F1 landing this memo's recommendation builds on).

---

## 0. Executive summary (the operator's question, answered)

The operator's fleet-wide meta-bug — **"vehicle names have been stronger than vehicle implementations"** — is
confirmed at the byte level. The winning reference (`hnerv_muon`, PR95) is **863 LOC across 7 source files**, fully
mechanism-bound and reviewable in 30 seconds per file. Our three candidate carriers are:

- **hi_nerv**: `mlx_renderer.py` alone is **388.9 KB** (~7,400 lines), `birth_survival.py` 140 KB,
  `archive_candidate.py` 132 KB, `short_scorer_readiness.py` 107 KB. The mechanism that matters (the decoder) is
  buried under target-region-birth/survival scaffolds. Verified: the default decoder is a **skip-free
  PixelShuffle+SIREN(w=30) NeRV** with HiNeRV's two defining ideas (grid PE, ConvNeXt) **OFF by default**.
- **snerv_inverse_steg_carrier**: `mlx_native_train_export.py` is **583 KB**, `archive.py` 228 KB. BUT the
  architecture is **genuinely faithful** (real orthonormal DWT + exact adjoint; real official MFU/HFR/TUB conv
  blocks with torch/numpy/MLX parity proofs). Its inflate.py is **scorer-free AND torch-free (numpy-portable)**.
- **pact_nerv_vq**: the **smallest + cleanest** of the three (`architecture.py` 10 KB, `archive.py` 11.5 KB,
  `inflate.py` 3 KB). The VQ machinery is **genuine** (real STE + EMA + commitment), the archive grammar is
  **complete** (PR95-style monolithic 0.bin, 4 length-prefixed sections), the inflate path is **complete + byte-
  consuming** (indices->codebook lookup), and its PyTorch loss has the **correct objective shape** (frozen-scorer,
  no recon-MSE base). Its ONE fatal gap is the skip-free decoder.

### THE SINGLE RANKED PRIMARY RECOMMENDATION

> **(a) PATCH/MATURE our carrier to PR95-parity — but execute it on `pact_nerv_vq`, NOT on `hi_nerv`.**

Rationale (one line): `pact_nerv_vq` is the ONLY candidate that already has a complete + byte-consuming +
numpy-reviewable archive->inflate path AND the correct (frozen-scorer, no-recon-base) objective shape — so the only
missing mechanism is the F1 bilinear-skip+refine HF path, which is a ~15-LOC drop-in of an already-landed shared
canonical kernel, reaching a byte-closed evaluator-cell carrier in the fewest steps with the least MLX-port-parity
risk and the smallest review surface.

This is a REFINEMENT of the in-flight F1 work, not a contradiction: F1 landed the shared HF-residual canonical
kernel + wired it into `hi_nerv`'s MLX *recon-fit probe* (export fail-closed). The recommendation is to wire that
SAME canonical kernel into the carrier that already byte-closes (`pact_nerv_vq`), so the HF fix lands on an
end-to-end-scored vehicle instead of a research-only probe whose export+oracle parity is still unbuilt.

**Fallback / parallel:** **(c) promote SNeRV** is the strongest *architectural* answer to the spectral-bias root
cause (it generates HF instead of mean-fielding) and its inflate is already numpy-portable — but it is gated behind
the MFU/HFR/TUB source-forward export binding (a real, named, multi-step blocker). Run it as the parallel campaign,
behind `pact_nerv_vq`. **(b) vendor a fresh PR95-HNeRV MLX port** is the *safest fidelity* path (the reference is
863 LOC and fully reproducible) and is the correct move IF the F1 recon-fit ablation (running now) shows skip-on
does NOT break the mean-field on our MLX stack (i.e. our MLX harness has a deeper bug than the missing skip).

Ordering rationale across all four, by time-to-byte-closed-evaluator-cell-carrier (cheapest first):
**(a) pact_nerv_vq+skip  <  (b) vendor PR95  <  (c) SNeRV  <  (a') hi_nerv+skip** (hi_nerv ranked LAST among the
patch options — see §3.4: its export+oracle-parity for the skip is unbuilt and its decoder is buried under 388 KB
of birth/survival scaffold).

---

## 1. The mechanism-parity matrix (verified at file:line)

Columns are the mechanisms that determine evaluator-cell fidelity. Verdicts: **MATCH** (genuine, present, on) ·
**OFF** (present in code but gated off by default) · **MISSING** (absent) · **APPROX** (present but a degenerate or
divergent variant) · **CLAIMED** (docstring claims it; code lacks it — Catalog #307 documentation-fake).

Reference rows (PR95-HNeRV verified at `experiments/results/public_pr_archive_kaggle_mirror/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/src/model.py:42-54`,
`codec.py`, `train.py`, `stages/`; paper rows from sister memos' web-verified summaries arXiv 2306.09818 / 2501.01681
/ vdO 1711.00937).

| carrier | bilinear-skip per block | terminal HF refine | grid/coord PE | upsample op | ConvNeXt | per-pair latent geom | content-adaptive embed | codebook/VQ STE+EMA | score-aware objective ACTIVE | archive+inflate+exact-eval integrated |
|---|---|---|---|---|---|---|---|---|---|---|
| **PR95-HNeRV** (winner, ref) | **MATCH** `model.py:47-50` | **MATCH** `model.py:51` (dilated-conv `+0.1*sin(refine)`) | MISSING (no PE; pure latent) | PixelShuffle(2) `model.py:32,49` | MISSING | single 28-d/pair -> 2 frames `model.py:14` | MISSING (learned per-pair latent) | n/a | **MATCH** (frozen-SegNet margin `100*seg`, NO recon base; curriculum `losses.py`) | **MATCH** (codec.py monolithic 0.bin + inflate.py + 50h reproducible curriculum) |
| **paper HiNeRV** (arXiv 2306.09818) | MATCH (hierarchical skip) | n/a (refine-equiv via grid) | **MATCH (defining)** multi-res local grid PE | **bilinear-up REPLACES PixelShuffle (defining)** | **MATCH (defining)** | per-frame embed | **MATCH** content-adaptive | n/a | n/a (paper = recon PSNR) | n/a |
| **paper HNeRV** (Chen 2023) | MATCH (HNeRV blocks) | partial | content embed (not grid PE) | sub-pixel/PixelShuffle | redesigned blocks | content-adaptive embed | **MATCH (defining)** | n/a | n/a | n/a |
| **VQ-NeRV** (codebook) | varies | varies | varies | varies | varies | per-frame | partial | **MATCH (defining)** codebook on shallow/inter-frame feats | n/a | n/a |
| **our hi_nerv** | **OFF->probe** `architecture.py:344-345` skip-free; F1 flag `use_bilinear_skip=False` default `:154`; wired ONLY in MLX `mlx_renderer.py:635,7397` (export fail-closed `:7456-7458`); PyTorch oracle still skip-free | OFF (same F1 flag; MLX-only) | **OFF** `use_hierarchical_feature_grid=False` `:136` (machinery real `:224-282`, `official_core_forward_parity_proven=False :92`) | PixelShuffle(2) `:342` + terminal `F.interpolate` global resize `:524-530` | **OFF** `use_convnext_blocks=False` `:139` (machinery real `:297-325`) | 3-scale 16+20+24=60-d/pair, injected at 3 depths `:498-522` (APPROX: richer, more latent bytes) | MISSING | n/a | **DIVERGES** (trained path = shared MLX harness recon-MSE base; sister memo §1 trace d_seg=0.5075 `[macOS-MLX research-signal]`) | PARTIAL (inflate.py exists `inflate.py`; archive grammar exists; BUT skip-ON export fail-closed; buried under 388KB birth/survival) |
| **our snerv** | n/a (DWT split) MFU has **skip-concat carries** `official_mfu.py:480-514` (MATCH for path B) | n/a (HFR generates HF) | n/a (DWT coords) | ConvTranspose2d (MFU) `official_mfu.py` | n/a | per-frame LF code `carrier.py:736-755` | n/a (stores LF) | n/a | **AVAILABLE but OFF** (`segnet/pose_direct_live_*` knobs exist; ep22399 ran `observed_segnet_distillation_weight=None` recon-only, sister memo §0) | **inflate numpy-portable** `inflate.py:6` BUT path-B export **BLOCKED** `carrier.py:315` (`snerv_official_mfu_hfr_tub_source_forward_replay_missing`) |
| **our pact_nerv_vq** | **MISSING** `architecture.py:77-78` (`_DsUpBlock`=`shuffle(act(dsc(x)))`) | MISSING | MISSING | PixelShuffle(2) `:75` (depthwise-sep conv) | MISSING | single 24-d/pair -> VQ `:185-193` | MISSING (VQ on latent) | **MATCH (genuine)** STE `:147` + EMA+Laplace `:150-166` + commitment `:144` | **MATCH** (PyTorch `score_aware_loss.py:84-109`: `100*seg + gamma*sqrt(pose) + commitment`, NO recon base, eval_roundtrip mandatory `:72-76`, frozen-scorer dispatch `:84`) | **MATCH (complete + byte-consuming)** monolithic 0.bin 4 sections `archive.py:14-17` + inflate consumes indices->codebook `inflate.py:49-58` |

### 1.1 The single decisive headline from the matrix

Every one of our carriers shares **Shared Mistake A** (skip-free PixelShuffle+sin30 decoder, no HF residual path)
EXCEPT where F1 just landed it into hi_nerv's MLX probe. AND the two carriers that train at scale through the shared
MLX harness (hi_nerv, snerv) also share **Shared Mistake B** (recon-MSE base objective rewards the mean-field).

The crucial asymmetry the matrix exposes: **`pact_nerv_vq` already does NOT have Mistake B** (its PyTorch loss is
the correct frozen-scorer/no-recon-base shape) AND already has a complete byte-closed archive->inflate path. It has
exactly ONE missing column among the four "fidelity-critical" ones (skip / objective / archive-integrated /
genuine-mechanism): the bilinear-skip. That is why it is the cheapest patch target.

---

## 2. The size/complexity headline (the operator's meta-bug, quantified)

| carrier | core mechanism file | size | reviewable in 30s? | mechanism buried? |
|---|---|---|---|---|
| **PR95-HNeRV** (ref) | `model.py` + `codec.py` + `train.py` + 8 `stages/*.py` | **863 LOC total** | YES (each file 1-7 KB) | NO — every line is mechanism |
| our hi_nerv | `mlx_renderer.py` | **388.9 KB (~7,400 LOC)** | NO | YES — decoder buried under birth/survival/scorer-readiness scaffolds (`birth_survival.py` 140 KB, `archive_candidate.py` 132 KB, `short_scorer_readiness.py` 107 KB, `target_region_birth.py` 44 KB) |
| our snerv | `mlx_native_train_export.py` | **583 KB** | NO | PARTIAL — real MFU/HFR/TUB primitives are clean (`official_mfu.py` 44 KB, etc.) but the train/export harness is enormous (`archive.py` 228 KB, `advisory.py` 47 KB) |
| our pact_nerv_vq | `architecture.py` | **10.1 KB (288 LOC)** | YES | NO — closest in spirit to the PR95 reference's reviewability |

This table is the operator's thesis made concrete: the carrier whose NAME most strongly evokes the paper (hi_nerv,
named after HiNeRV; snerv, named after SNeRV) is the one most buried under scaffold; the carrier with the most modest
name (`pact_nerv_vq`) is the one closest to the winning reference's mechanism-bound minimalism.

Lane-registry corroboration (`.omx/state/lane_registry.json`, 1731 lanes): the `hi_nerv_*` and `snerv_*` lane
families have **dozens of L0/L1 variants** (`hi_nerv_target_region_birth_v30`..`v38`, `lane_snerv_*` x ~20) — name
proliferation without a single carrier reaching a byte-closed evaluator-cell score. This is the 18-shared-assumption
plateau manifesting as substrate-count sprawl.

---

## 3. The ranked decision, with full rationale per option

Decision criteria (all four scored): LOC-to-mechanism-complete, MLX-port-parity risk, time-to-mechanism-complete,
time-to-exact-eval-integrated, and (the decisive one) **which reaches a real byte-closed evaluator-cell carrier
FASTEST**. Every estimate is an engineering estimate, not a measurement.

### 3.1 RANK 1 (PRIMARY) — (a') Patch `pact_nerv_vq` to PR95-parity

- **LOC to mechanism-complete:** ~15-30 LOC. Wire the already-landed `bilinear_skip_residual_canonical` +
  `terminal_hf_refine_canonical` (`tac.framework_agnostic.canonical_kernels:652,720` — genuine numpy+MLX+torch+
  tinygrad, fail-closed on shape mismatch) into `_DsUpBlock.forward` (a 1x1 channel-match skip) + a terminal refine
  conv before the RGB heads. The kernel is shared, so this is a drop-in, not a re-derivation.
- **MLX-port-parity risk:** LOW-MEDIUM. `pact_nerv_vq` HAS an MLX renderer (`mlx_renderer.py` 27 KB — the smallest
  MLX renderer of the three) and a PyTorch oracle (`architecture.py`). The skip must land in BOTH and pass parity;
  but the small surface (288-LOC oracle) makes the parity test tractable. Contrast hi_nerv (7,400-LOC MLX renderer).
- **time-to-mechanism-complete:** SHORTEST. Archive grammar (`archive.py` monolithic 0.bin) + inflate (`inflate.py`
  byte-consuming indices->codebook) + objective shape (frozen-scorer, no recon base) are ALL already done. Only the
  decoder HF path is missing.
- **time-to-exact-eval-integrated:** SHORTEST. The inflate path already byte-closes; adding the skip changes decoder
  *weights* (more params), not the archive *grammar* (the skip is a few extra conv tensors in the same INT8+brotli
  decoder blob). So the existing parse_archive/inflate contract extends naturally.
- **reaches a byte-closed evaluator-cell carrier:** FASTEST. This is the only path where every non-HF piece is
  already integrated and the objective is already correct.
- **WHY pact_nerv_vq over hi_nerv for "patch our sketch":** the prompt framed "patch our sketch" as hi_nerv, but the
  cross-vehicle audit + my verification show `pact_nerv_vq` is strictly the better patch target: it lacks Mistake B
  (hi_nerv has it), it has a complete byte-closed inflate (hi_nerv's skip-ON export is fail-closed), and its mechanism
  is reviewable (hi_nerv's is buried). The VQ codebook is a genuine bonus byte-story (L21/L24/L25 PR95 lessons) that
  comes for free.
- **risk this DOESN'T work:** if the F1 recon-fit ablation (running now at
  `/Volumes/VertigoDataTier/pact/recon_fit_f1_skipON_w30_20260609T191128Z`) shows the skip does NOT break the
  mean-field on our MLX stack, then the binding constraint is deeper (grid PE / objective / a harness bug) and the
  patch must escalate to also turn on coordinate PE (pact_nerv_vq lacks PE entirely) or switch to (b)/(c). The skip
  is necessary; the ablation tells us if it's sufficient.

### 3.2 RANK 2 (FALLBACK / SAFEST FIDELITY) — (b) Vendor a faithful PR95-HNeRV MLX port from the intake model.py

- **LOC:** ~863 LOC reference is the spec; an MLX port of `model.py` (54 LOC decoder) + `codec.py` (180 LOC, mostly
  numpy/brotli — portable as-is) + a short score-aware curriculum is **~300-500 new LOC** (the decoder + a thin
  curriculum; codec.py is already numpy and needs no port). This is the UNIQUE-AND-COMPLETE-PER-METHOD ideal: a
  single coherent packet reviewable in 30 seconds, no birth/survival scaffold.
- **MLX-port-parity risk:** MEDIUM. The decoder is tiny (6 stages, sin, PixelShuffle, bilinear-skip, refine) and the
  F1 canonical kernel already provides the skip+refine math with cross-backend parity. The main risk is the
  PixelShuffle + bilinear-interpolate MLX numerics matching the torch oracle (a known, bounded parity surface).
- **time-to-mechanism-complete:** MEDIUM. Building a clean carrier from scratch is more LOC than the pact_nerv_vq
  patch, but every LOC is mechanism (no scaffold to fight). codec.py ports for free.
- **time-to-exact-eval-integrated:** MEDIUM. codec.py's `build_archive`/`parse_archive` is a complete, bit-exact-
  verified monolithic 0.bin grammar — vendoring it gives a byte-closed archive immediately; the inflate is the
  decoder forward + the 25-line raw writer.
- **reaches a byte-closed carrier:** SECOND fastest, and HIGHEST fidelity-confidence (it IS the thing that scored
  ~0.193 `[contest-CUDA historical]`). This is the right move if we want a *known-good* baseline carrier to anchor
  the whole fleet, independent of whether our existing carriers have latent harness bugs.
- **WHY not RANK 1:** it is more new LOC than the pact_nerv_vq patch and re-implements an archive grammar we already
  have a working analog of. But it is the safest if the ablation says our MLX stack is the problem. Strongly consider
  promoting this to RANK 1 if the F1 ablation comes back flat.

### 3.3 RANK 3 (PARALLEL CAMPAIGN) — (c) Promote SNeRV after the Mistake-B + export fixes

- **LOC to mechanism-complete:** the architecture is DONE (real DWT + adjoint, real conv MFU/HFR/TUB with parity
  proofs). The missing work is (i) config-only: turn ON `segnet/pose_direct_live_*` scorer weights + anneal recon +
  force `skip_high_mode='full'` (Mistake-B fix, ~0 LOC, the sister memo's G1); (ii) the export binding: close
  `snerv_official_mfu_hfr_tub_source_forward_replay_missing` (`carrier.py:315`) + the MFU numeric-parity blockers —
  a real, multi-step, NON-trivial source-forward/export task (AGENTS.md names this the SNeRV hard blocker).
- **MLX-port-parity risk:** the parity proofs already exist for the primitives, but the *full renderer* source-forward
  replay is unproven — that is the blocker, and it is the riskiest of the four.
- **time-to-mechanism-complete:** MEDIUM-LONG (the export binding is the long pole).
- **time-to-exact-eval-integrated:** LONG (export-blocked today; inflate.py is numpy-portable and ready, but path-B
  weights can't yet be exported into it).
- **reaches a byte-closed carrier:** SLOWEST of the viable three, BUT it is the only one that attacks the
  spectral-bias root cause *architecturally* (generates HF instead of relying on a skip to carry it) AND has the best
  byte story (store-LF/generate-HF) AND restores the two-frame pose signal via TUB. **Highest ceiling, longest path.**
- **WHY parallel not primary:** it is the right long-term carrier but the export blocker makes it the wrong *first*
  byte-closed score. Run G1 (the config-only Mistake-B fix) as a cheap recon/scorer smoke in parallel to confirm the
  faithful architecture escapes the mean-field; if it does, prioritize the export binding.

### 3.4 RANK 4 (NOT RECOMMENDED as the primary patch) — (a) Patch `hi_nerv` itself

- The F1 skip is wired into hi_nerv's MLX recon-fit probe ONLY; **export is fail-closed when skip is ON**
  (`mlx_renderer.py:7456-7458`), the **PyTorch oracle `_UpBlock` is still skip-free** (`architecture.py:344-345`), and
  the trained path carries **Mistake B** (recon-MSE base via the shared MLX harness). So reaching a byte-closed
  evaluator-cell carrier from hi_nerv requires: (i) lift the export fail-closed + build the skip export layout; (ii)
  bring the PyTorch oracle to parity; (iii) fix the objective (drop recon base); (iv) all while the decoder is buried
  under 388 KB of birth/survival/scorer-readiness scaffold that must not regress. That is strictly MORE work than the
  pact_nerv_vq patch for the SAME HF fix.
- hi_nerv remains the right home for the **F1 recon-fit ablation** (it's already running there) — that ablation is
  the disambiguator that informs RANK 1 vs RANK 2. But hi_nerv is the wrong carrier to *promote* to the byte-closed
  score; its value right now is diagnostic, not deliverable.

---

## 4. Gap-closure checklist for the RECOMMENDED path (pact_nerv_vq + PR95 HF mechanism)

Ordered; each step has a falsifiable mechanism-test. All local MLX/CPU, `[macOS-MLX research-signal]` / mechanism-only
unless the final exact-eval step. A downstream completion subagent can execute this without re-deriving. Per
"Forbidden premature KILL" these are research steps; promotion needs a byte-closed archive + paired CPU/CUDA.

**Pre-step 0 (BLOCKING gate, ~$0, do FIRST):** read the F1 recon-fit ablation result when
`/Volumes/VertigoDataTier/pact/recon_fit_f1_skipON_w30_*` and the w1 arm finish (currently only `source.raw`
written; no `recon_fit_f1_*.json` yet). **Mechanism-test:** if skip-ON breaks PSNR 21.74 -> >28 dB on hi_nerv's MLX
stack, the skip is the binding fix -> proceed with this checklist. If skip-ON stays ~21.7 (H1 falsified on our
stack), the binding constraint is grid-PE or a harness bug -> escalate to RANK 2 (vendor PR95) instead, because a
clean-room port avoids whatever latent bug our MLX harness carries. **Do not skip this gate** — it decides RANK 1 vs
RANK 2.

1. **Wire the F1 bilinear-skip into `pact_nerv_vq._DsUpBlock` (PyTorch oracle + MLX renderer).**
   - Change `_DsUpBlock.forward` to `bilinear_skip_residual_canonical(shuffle(act(dsc(x))), skip_1x1(bilinear_2x(x)),
     sin_frequency=w)`; add a `skip_1x1` 1x1 conv for channel-match (in!=out). Gate behind a config flag default OFF
     (zero-regression per F1's pattern). Replicate to `mlx_renderer.py`.
   - **Mechanism-test:** OFF path byte-identical (param count unchanged); ON path forwards + param count rises + init
     output std jumps (the F1 landing measured 0.0001 -> 0.0084, 84x, on hi_nerv — expect a comparable variance
     injection here). Cross-backend parity test (numpy vs MLX vs torch) on the skip composition.

2. **Add the terminal `x + 0.1*sin(refine(x))` dilated-conv refine before the RGB heads** (PyTorch oracle + MLX).
   - Use `terminal_hf_refine_canonical(h, refine_conv(h), scale=0.1)`. PR95 uses a 2-conv dilated refine
     (`model.py:35-38`).
   - **Mechanism-test:** refine OFF byte-identical; ON adds the documented param delta + a measurable HF-energy
     increase in the rendered frame's high-pass spectrum.

3. **Lower the SIREN frequency for the skip-fed sin from w=30 to w~1 (PR95-implicit).**
   - The F1 ablation's Arm-B tests exactly this: w=30 on a skip-free feature map is a spectral-bias trap (sister memo
     H4); once the skip provides a coherent carrier inside the sin, PR95 uses w~1. Sweep {1,6,30} if the ablation is
     ambiguous.
   - **Mechanism-test:** at fixed (skip-ON) architecture, w~1 recon PSNR >= w=30 recon PSNR on the single video. If
     w=30 wins with the skip, keep it (HARD-EARNED); the prediction is w~1 wins.

4. **Train the SAME short curriculum with the skip ON, scored through the LIVE DistortionNet** (NOT PSNR).
   - `pact_nerv_vq`'s PyTorch loss is already the correct shape (frozen-scorer `100*seg + gamma*sqrt(pose) +
     commitment`, no recon base, eval_roundtrip mandatory). Confirm the MLX training path uses THIS objective and NOT
     the shared MLX harness recon-MSE base (the sister memo flags `lane_pact_nerv_vq_l1_long_run_mlx_local` may route
     through the shared harness — **verify + fork if so**; this is the Mistake-B guard).
   - **Mechanism-test:** live MLX render scored through the exact DistortionNet -> **d_seg drops from the ~0.5
     skip-free baseline toward <0.1 and d_pose co-improves** within the same epoch budget. If d_seg stays ~0.5 with
     skip+correct-objective, H1 is falsified for this carrier and the binding constraint is the missing coordinate PE
     (pact_nerv_vq has none) -> add a coordinate/Fourier-feature PE input (step 6).

5. **Extend the archive grammar + inflate to carry the new skip/refine conv tensors; prove byte-closure.**
   - The skip+refine add conv weights to the decoder state_dict; `archive.py`'s INT8+brotli decoder blob already
     serializes arbitrary state_dict tensors, and `inflate.py` already `load_state_dict`s — so the grammar extends
     without a new section. Verify round-trip bit-exactness (PR95 codec.py does this; mirror the test).
   - **Mechanism-test:** `parse_archive(build_archive(...))` reproduces the decoder state_dict bit-exactly; the
     no-op detector confirms the new bytes are CONSUMED at inflate (decoded frames change when a skip weight is
     perturbed). This is the Catalog #105/#139/#220 operational-consumption proof.

6. **(Conditional, only if step 4's d_seg stalls) Add a coordinate/grid positional encoding input.**
   - pact_nerv_vq has NO PE (single latent -> decoder). If the skip alone doesn't break the mean-field, inject a
     coordinate PE (or the HiNeRV-style local feature grid) so the decoder has location-specific HF. This is the
     2x2 {skip off/on} x {PE off/on} ablation the sister memos propose.
   - **Mechanism-test:** PE-on adds a further d_seg reduction (HiNeRV's own ablation = +2.5 dB PSNR). If skip-on
     alone fixed it, PE is optional polish.

7. **(Promotion gate, NOT done by the patch) byte-closed archive -> paired `[contest-CPU]` + `[contest-CUDA]`.**
   - Only after steps 1-5 give a live-MLX d_seg < ~0.2 do we build a real archive and run the exact paired eval per
     CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA". Until then every number is `[macOS-MLX research-signal]`.

**Sequencing note (architecture-first per AGENTS.md "Evaluator-Equivalent Witness Compiler"):** steps 1-4 are the
mechanism; step 5 is the byte-closure; steps 6-7 are conditional/promotion. Do NOT branch to codebook-size sweeps,
PR110++ stacking, or optimizer exotica until step 4 produces a live d_seg that breaks the mean-field — none of our
carriers has reached evaluator fidelity yet, so HF-mechanism completion strictly precedes every byte-shaving lane.

---

## 5. Honest scope / limits

- **Verified by reading source (file:line cited in §1):** PR95 decoder (`model.py:42-54`), PR95 codec
  (`codec.py:30-181`, bit-exact monolithic 0.bin), PR95 8-stage curriculum (`train.py` + `stages/stage1..8`); our
  hi_nerv skip-free PyTorch `_UpBlock` (`architecture.py:344-345`) + F1 flag default-OFF (`:154`) + grid-PE/ConvNeXt
  OFF (`:136,139`) + the false-authority split (`:82-98`); our pact_nerv_vq genuine VQ (STE `:147`, EMA `:150-166`,
  commitment `:144`) + correct objective shape (`score_aware_loss.py:84-109`, no recon base, eval_roundtrip
  mandatory) + complete byte-consuming inflate (`inflate.py:49-58`) + complete archive grammar (`archive.py:14-17`);
  our snerv genuine DWT + conv MFU/HFR/TUB primitives (`official_mfu.py:480-514`) + numpy-portable inflate
  (`inflate.py:6`) + the export blocker (`carrier.py:315`); the F1 canonical kernels are genuine numpy+MLX+torch+
  tinygrad fail-closed (`canonical_kernels.py:652-768`); F1 skip wired ONLY into hi_nerv MLX (`mlx_renderer.py:635,
  7388-7397`), export fail-closed when ON (`:7456-7458`), NOT into pact_nerv_vq or snerv (grep confirmed empty).
- **Verified by file system:** F1 recon-fit ablation IN PROGRESS at
  `/Volumes/VertigoDataTier/pact/recon_fit_f1_skipON_w30_20260609T191128Z` (only `source.raw` 3.49 GB + a 97-byte
  stdout.log written so far; no `recon_fit_f1_*.json` result yet — the ablation has NOT completed, so its verdict is
  not yet available; another subagent owns it). Lane registry (`.omx/state/lane_registry.json`, 1731 lanes) shows
  dozens of L0/L1 hi_nerv_*/snerv_* variants and no byte-closed evaluator-cell score for any of the three.
- **Verified by reading `upstream/evaluate.py`:** the contest law `score = 100*segnet_dist +
  sqrt(posenet_dist*10) + 25*rate` (`evaluate.py:92`), rate = compressed_size / summed-uncompressed-size
  (`:63-65`), SegNet/PoseNet via `DistortionNet.compute_distortion` on both frames (`:79`).
- **INFERRED (engineering estimates, NOT measurements):** all LOC-to-complete + time-to-complete estimates in §3;
  the prediction that the pact_nerv_vq+skip path reaches a byte-closed carrier fastest (a sequencing argument from
  what is already integrated, not a measured wall-clock); the d_seg < 0.1 predictions in §4 (falsifiable hypotheses
  carried from the sister memos' F1/G2 mechanism arguments, not scored here).
- **NOT done (out of scope for a read-only comparison):** no training, no GPU, no paid dispatch, no edits to any
  carrier source or `upstream/`, no byte-closed archive built, no `[contest-CPU]`/`[contest-CUDA]` score produced.
  Every fidelity verdict is a *mechanism* verdict.
- **Authority:** everything here is `[macOS-CPU advisory]` / `mechanism_update_eligible`. It routes the next build;
  it does NOT promote, rank, kill, or close any lane (CLAUDE.md "Meta-Lagrangian/Pareto solver" + "Forbidden
  premature KILL"). The recommendation is a build-priority recommendation, not a score claim.

### One-line decision

**Patch `pact_nerv_vq` with the already-landed F1 bilinear-skip + refine HF kernel** (cheapest path to ONE
byte-closed evaluator-cell carrier: it alone already has the complete archive->inflate path AND the correct
frozen-scorer/no-recon-base objective, so only the ~15-LOC HF mechanism is missing) — gated on the in-flight F1
recon-fit ablation, with **vendor-a-clean-PR95-HNeRV-MLX-port** as the fallback if that ablation shows our MLX stack
has a deeper bug than the missing skip, and **SNeRV** as the parallel architectural campaign behind the export
binding.
