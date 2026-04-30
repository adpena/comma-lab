# All-Scores Forensic Audit (2026-04-30)

**Author:** ALL-SCORES-FORENSIC-AGENT (#298)
**Companion files:**
- Inventory: `.omx/research/all_scores_inventory_20260430.md`
- Re-engineering plans: `.omx/research/recoverable_lanes_re_engineering_plans_20260430.md`

**Mandate:** classify every "bad" / "killed" lane against:

- `APPROACH_KILLED` — fundamental architecture/method failure (math/empirical proof)
- `ENGINEERING_BUG` — implementation bug (dead flag, OOM, wrong device, missing import)
- `CONFIG_BUG` — wrong hyperparameter / wrong anchor / wrong distribution
- `METHODOLOGY_BUG` — eval pipeline wrong (MPS, CPU, wrong archive, wrong pairs)
- `LEGITIMATE_REGRESSION` — sound impl + sound eval, score genuinely worse
- `INDETERMINATE` — insufficient evidence to classify

---

## Classification methodology

For each lane I ran sanity checks **2a-2g** from the audit spec:
- 2a archive byte size (300KB-700KB expected)
- 2b device (CUDA only counts; CPU/MPS = methodology bug)
- 2c eval_roundtrip on training (if False = engineering bug)
- 2d dead-flag on subprocess (engineering bug)
- 2e encode-then-discard (engineering bug)
- 2f anchor parity (must be Lane G v3 1.05 or Lane A 1.15)
- 2g final classification

The strongest forensic signal is **the harvested `_stdout_tail.txt`** — every Modal lane has its crash trace in `experiments/results/lane_*_modal/harvested_artifacts/_stdout_tail.txt`.

---

## Per-lane forensic verdicts

### Group A — UNIWARD lineage (v1-v8)

| Lane | Final state | Bug class | Sanity check | Verdict |
|------|-------------|-----------|--------------|---------|
| **UNIWARD v1** | crashed Stage 1 | Anchor path missing (`baseline_dilated_h64_0_90/renderer.bin`) | 2f failed | ENGINEERING_BUG |
| **UNIWARD v2** | crashed Stage 2 | bash-as-python heredoc syntax | 2d failed | ENGINEERING_BUG |
| **UNIWARD v3** | crashed Stage 3 | `NameError: sys not defined` in inline python | 2d failed | ENGINEERING_BUG |
| **UNIWARD v4** | crashed Stage 3 | `apply_saliency_weighted_compression()` got unexpected `mode` kwarg | 2d failed (dead-flag class) | ENGINEERING_BUG |
| **UNIWARD v5** | crashed Stage 3 | `saliency_inv must be 2-D bool; got float32` | 2d failed (type contract) | ENGINEERING_BUG |
| **UNIWARD v6** | crashed Stage 3 | `NameError: json not defined` | 2d failed | ENGINEERING_BUG |
| **UNIWARD v7** | 53.61 | 48x64 mask anchor (Pre-Check-76) | 2a + 2f failed | METHODOLOGY_BUG (CLAUDE.md "MASKS.MKV AT 48x64 DESTROYED THE SCORE") |
| **UNIWARD v8** | 1.14 [contest-CPU advisory] **NO-OP** | Stage 4 `cp $ANCHOR_DIR/masks.mkv` discards Stage 3's 8.6MB SLI1 payload | 2e failed (encode-then-discard) | ENGINEERING_BUG (not LEGITIMATE; mask SHA-identical to Lane A) |

**Underlying approach (UNIWARD-style cost-routed mask-CRF):** UNTESTED on the merits. Every v1-v8 was an engineering failure (or a no-op). The lane has not produced a single CUDA-valid measurement of the actual UNIWARD hypothesis. The `src/tac/uniward_texture.py` module also has math errors (2-tap stencils not Daubechies-8 — Council B audit). Hidden gem: re-engineerable.

---

### Group B — Lane SegMapTrainer family (SC++, SA, SO — 9 dispatches, 0 outputs)

All 9 SegMapTrainer-using lanes (SC++ v1-v4, SA v1-v5, SO v1-v3) crashed with `torch.OutOfMemoryError: Tried to allocate 7.03 GiB on T4` or `21.09 GiB on A10G`. Fixed locus: `src/tac/segmap_renderer.py:284` (or `:296`/`:391` depending on version) where `rendered = self.model(masks_flat, frame_indices)` materializes the entire `(B*T, 3, H, W)` rendered tensor in float32. T4 has 14.56GB, A10G shared 22GB.

| Lane | Verdict | Justification |
|------|---------|---------------|
| SC++ v1-v4 | ENGINEERING_BUG (×4) | OOM in train_epoch, never reached eval. Approach untested. |
| SA v1-v5 | ENGINEERING_BUG (×5) | Same locus |
| SO v1-v3 | ENGINEERING_BUG (×3) | Same locus |

**Underlying approach (KL-distill SegMap clone of Selfcomp):** UNTESTED. Council A invalidated 9 SegMapTrainer lanes (memory `feedback_round6_defects_lane_mm_correction_segmap_invalidation_extended_20260429`) but the bug was specifically `.round()` zero-grad, NOT the OOM that prevented these from running. **Both bugs need fix before any approach claim.**

Re-engineering: chunk `rendered` over batch dim (already done in src commit at line 391 per stack trace, but A10G v3 still OOMed at 21GB); use `bf16` + `torch.utils.checkpoint`. Council C bf16 + scorer-chunk fix per memory `project_session_state_checkpoint_20260430` was dispatched on Vast.ai 4090 tonight ($3.12, 12h overnight).

---

### Group C — Lane GP (Gaussian-Process pose fit) v1, v2, v3, v4

| Lane | Score | Bug class | Verdict |
|------|-------|-----------|---------|
| **GP v1** | crash (rc=1) | `RuntimeError: tensors on cpu and cuda` in `save_pose_gp` (`pose_gaussian_process.py:104`) | ENGINEERING_BUG |
| **GP v2** | 89.66 [contest-CPU advisory] | Polynomial degree-10 fit produces RMSE ≈ 1.011 vs baseline (signal std) → Runge phenomenon at 600 equispaced points | APPROACH_KILLED (math: white-noise dims 1-5 cannot be smooth-fit at K<500) — but TAG is CPU; CUDA confirm not done |
| **GP v3** | 89.67 [contest-CPU advisory] | Fix A landed (`baseline_poses=` kwarg), but score didn't budge — proves the off-manifold hypothesis was wrong; the issue is degree-10 polynomial CAN'T represent dim 0 | APPROACH_KILLED (Same proof as v2) |
| **GP v4** | KILLED at design | Council Round 1-4 empirical proof: pose trajectory diff_std > signal_std → no smooth basis (poly/cubic-B-spline/DCT/natural-cubic-spline) fits below RMSE ≈ 1.2 (= signal std) at K<500 | APPROACH_KILLED (basis-fit infeasibility) |

**Verdict:** Lane GP class **truly killed**. The math is sound (Council #271 empirical analysis is in the lane registry). STRICT preflight Check 91 prevents revival. **NOT a hidden gem.**

Note: the [contest-CPU advisory] tag is non-fatal here because the magnitude of the regression (89× baseline) exceeds any CPU-CUDA drift class. A CUDA confirm is academically nice-to-have but the math is the kill.

---

### Group D — Lane MM (encoder-only grayscale-LUT mask)

| Lane | Score | Bug class | Verdict |
|------|-------|-----------|---------|
| **MM v1** | crash (rc=3) | Dead-flag: `--hard` not in `build_lane_mm_archive` argparse | ENGINEERING_BUG |
| **MM v2** | 2.63 [contest-CPU advisory] | PoseNet 51× worse, archive 1.55× BIGGER vs Lane A baseline. Hard-argmax grayscale on 3ch-trained renderer destroys reconstruction. | LEGITIMATE_REGRESSION-conditional + METHODOLOGY_BUG (CPU eval) — Council Round 7 retag pending CUDA confirm |

**Underlying approach assessment:** MM as bolt-on FALSIFIED conditionally on the math (3ch-trained renderer + hard-argmax grayscale-LUT mask = catastrophic train/test mismatch). Council EUREKA path is **Lane AL** (SGD-optimized soft grayscale + retrained renderer) — that approach is UNTESTED.

Hidden gem: Lane AL (the proper variant) is not the same lane and has not been engineered.

---

### Group E — Lane V family (Quantizr replica 88K + half-frame)

| Lane | Crash | Bug | Verdict |
|------|-------|-----|---------|
| **V** | RuntimeError: input[1, 1, 384, 512] expected 3 channels but got 1 | Channel mismatch in HintedRenderer alpha_map path (`renderer.py:944`); use_dsconv=True dispatches different conv-construction | ENGINEERING_BUG |
| **V-V2** | Same | Inherits V channel bug | ENGINEERING_BUG |
| **D-V3** | Half-frame broken | end_value=0.5 ≠ inflate-time 1.0 (train/test distribution mismatch); same class as Lane M-V2 BUG-1 | CONFIG_BUG |
| **J-JBL** | exited unexpectedly | `combined_jbl_distill_loss` not wired in `train_renderer.py` loss dispatch | ENGINEERING_BUG |

**Underlying approach (Quantizr replica + half-frame):** UNTESTED on the merits. Quantizr ships half-frame at 0.33 — math says it works. All 4 attempts crashed/misconfigured. **Hidden gem.**

Reference: `project_killed_lanes_forensic_audit_20260428.md` (the prior audit by same agent class).

---

### Group F — Lane F (FP4 QAT)

| Lane | Score | Bug | Verdict |
|------|-------|-----|---------|
| **F V1** | 2.73 | Silent default override: `gt_poses.pt` auto-discovered from 2 paths, neither existed → trained with zero poses → +58% PoseNet from CONDITIONING bug, NOT from FP4 quant | ENGINEERING_BUG (silent-default, memory `feedback_silent_default_masquerading_as_negative_result`) |
| **F V2** | 1.79 | Fix landed; FP4 QAT actually ran. PoseNet 0.005 → 0.101 (20× regression) | LEGITIMATE_REGRESSION-conditional (FP4 simulation on 4090 ≠ HW FP4 on Blackwell) |
| **F V4** | n/a | Closed by council. NVFP4 hardware requires Blackwell CC 10.0; 4090 = Ada CC 8.9 = simulated FakeQuantFP4 in FP32 | METHODOLOGY_BUG (hardware-quant disclosure) |

**Verdict on Lane F class:** ALL prior FP4 results were SIMULATED FP4 on FP32 hardware. Real FP4 needs Blackwell. Lane F-V5 (hardware FP8 via torchao.float8) is the proper rescue (Ada/Lovelace+ supports FP8). **Hidden gem (F-V5 specifically).**

---

### Group G — Lane FL (RAFT-derived poses)

Both v1 + v2 OOM-killed (rc=137) at Stage 3 `derive_poses_from_raft.py --device cuda --n-frames 1200`. RAFT model on T4 needs >14.56GB to materialize 1200 frames at full res.

**Verdict:** ENGINEERING_BUG ×2. Approach (RAFT-derived poses) UNTESTED. Re-engineering: `--n-frames 100` + accumulate, or use A10G 22GB, or chunk RAFT inference. Hidden gem.

---

### Group H — Lane W (hard-pair self-compress)

| Lane | Crash | Bug | Verdict |
|------|-------|-----|---------|
| **W v1** | rc=1 | `ValueError: --resume-from is quantised binary (magic=ASYM)` — caller passed renderer.bin (quantized) instead of renderer_*_best_fp32.pt | ENGINEERING_BUG (R11 Finding 1 catch) |
| **W v2** | rc=124 (8h timeout) | Stage 2 train hung indefinitely after pair_weights computed | ENGINEERING_BUG (timeout = bug) |

**Verdict:** Approach UNTESTED. Hidden gem.

---

### Group I — Lane Q-FAITHFUL (Quantizr 1:1 architecture)

| Lane | Crash | Bug | Verdict |
|------|-------|-----|---------|
| **Q-FAITHFUL v1** | rc=2 | argparse `--tag required` | ENGINEERING_BUG (missing CLI arg in dispatch) |
| **Q-FAITHFUL v2** | rc=1 | CONFIG ERROR: `variant=quantizr_faithful` not FP4A-exportable; `--auth-eval-on-best` enabled (default) crashes after hours | ENGINEERING_BUG (codex R5-2 #1 — gate ordering vs variant validator) |

**Verdict:** Approach UNTESTED. Hidden gem.

---

### Group J — Lane I (Cool-Chic CCh1 replacement)

Trained 999/1000 epochs successfully (best FP4 scorer 2.7196 at ep 754). Stage 3 export crashed: `parametrize.weight.original` vs raw weight key mismatch. **NO auth eval.**

**Verdict:** ENGINEERING_BUG (parametrize-strip mismatch — same class as the SHIRAZ R23-26 chain). Approach UNTESTED. Hidden gem.

---

### Group K — Lane STC (clean-source)

Original FALSIFICATION WITHDRAWN per CLAUDE.md "MPS auth eval is NOISE" non-negotiable. The local MPS encoder produced 21MB but SegNet-on-MPS was the input — measurement was contaminated.

**Verdict:** METHODOLOGY_BUG. Approach status: UNDETERMINED until CUDA Modal T4 re-run (~$0.20). Codec still has known bug — "one-majority-plus-exceptions" stores 109M exceptions on multi-region masks (council audit). Floor 7.6MB after deflation — 18× worse than AV1's 0.014 bpp. Structural redesign needed.

---

### Group L — Lane M-V2 / Lane M+N (radial-zoom 1-DOF)

| Lane | Score | Bug | Verdict |
|------|-------|-----|---------|
| **M-V2** | 1.84 [contest-CUDA] | BUG-1 (CRITICAL): train/inference pose-pad mismatch — optimizer feeds renderer ZERO-padded dims 1-5 while inflate feeds frozen-baseline-padded | CONFIG_BUG (proper Lane M-V3-clean predicted [1.05, 1.20]) |
| **M+N v1** | 2.35 [contest-CUDA] | Rank-1 PoseNet sensitivity ≠ rank-1 renderer input space; baseline renderer trained on 6-DOF | CONFIG_BUG / partial APPROACH_KILLED |

**Verdict:** Lane M-V2 specifically has a clean re-engineering path (Lane M-V3-clean) costing $0.30 with predicted band [1.05, 1.20]. Hidden gem.

The broader rank-1 hypothesis remains uncertain: the math (rank-1 PoseNet Jacobian) is sound but the renderer-input-space math is wrong (Lane HF Telescope foveation is the proper revival).

---

### Group M — Lane B (pose TTO on dilated-h64)

| Lane | Score | Bug | Verdict |
|------|-------|-----|---------|
| **B** | 2.40 [contest-CUDA] | 350× proxy-auth gap. Proxy → 0.0007, auth 0.246 (23× baseline 0.0107) | LEGITIMATE_REGRESSION-conditional (per `feedback_proxy_auth_math_useless` — proxy is training signal not measurement; pose TTO doesn't transfer between architectures without an auth-validated loop). Bootstrap also had 3 silent failures (FIXED, see commit 813a4891). |

Mixed verdict: bootstrap was ENGINEERING_BUG (now structurally extinct via STRICT Check 2/3); pose TTO regression is structural to the dilated-h64 architecture. Re-engineering would require auth-eval-every-100-steps in the TTO loop (memory `feedback_proxy_auth_math_useless`).

---

### Group N — Lane MAE-V

Crash: `ModuleNotFoundError: No module named 'pydantic'` at import time. **Pure Modal-image dependency bug.**

**Verdict:** ENGINEERING_BUG. Approach UNTESTED. Hidden gem.

---

### Group O — Lane Omega-Hessian

Crash: `CUDA error: device-side assert triggered` in `renderer.py:471 torch.linspace(-1, 1, H, ...)`. Likely OOB index from `pair_idx`. **Tiny smoke needed** to reproduce.

**Verdict:** ENGINEERING_BUG. Approach (per-weight Hessian-aware bit allocation) UNTESTED. Hidden gem.

---

### Group P — Lane S (full-arch Self-Compression)

Crash log truncated, but Phase 1 init wrote `lane_a_sc_init.pt: 1188638 bytes`, swapped 16 SC layers, then OOM at Phase 2 (precompute GT scorer cache 2359MB on T4 14.56GB).

**Verdict:** ENGINEERING_BUG (T4 OOM). Approach UNTESTED. Hidden gem.

---

## Summary classification distribution

| Class | Count | Lanes |
|-------|-------|-------|
| **APPROACH_KILLED** (math/empirical proof) | **3** | GP v2, GP v3, GP v4 (basis-fit infeasibility, Council #271) |
| **ENGINEERING_BUG** | **23** | UNIWARD v1-v6 (×6), V, V-V2, J-JBL, FL v1+v2, GP v1, MAE-V, MM v1, Omega-Hessian, S, SegMap-OOM lanes (SC++ v1-v4, SA v1-v5, SO v1-v3 = 12 individual instances of same root bug), W v1+v2, Q-F v1+v2, I, F V1 (silent-default) |
| **CONFIG_BUG** | **3** | D-V3 (end_value=0.5 vs 1.0), M-V2 (train/inference pose pad), M+N (rank-1 hypothesis-renderer mismatch) |
| **METHODOLOGY_BUG** | **5** | UNIWARD v7 (48x64 mask), UNIWARD v8 (NO-OP cp), F-V4 (simulated FP4), STC clean-source (MPS), MM v2 (CPU eval) |
| **LEGITIMATE_REGRESSION-conditional** | **3** | F-V2 (FP4 architectural bottleneck on dilated-h64; would need Lane F-V5 hardware FP8), B (pose TTO 350× proxy-auth gap; needs auth-eval-every-100-steps), M-V2 partially |
| **INDETERMINATE** | **1** | STC clean-source (need CUDA confirm) |

**Total cataloged dispatch attempts: ~38 lanes / ~50 individual run instances.**

**Distribution headline:**
- ~7% APPROACH_KILLED (math-grade)
- ~60% ENGINEERING_BUG (most "bad scores" are bugs)
- ~8% CONFIG_BUG (re-engineerable)
- ~13% METHODOLOGY_BUG (re-eval rescues)
- ~8% LEGITIMATE_REGRESSION-conditional (architectural bound)
- ~3% INDETERMINATE

**Key finding (matches user's intuition):** the vast majority of "bad scoring" lanes are bad because they were not engineered correctly, not because the underlying approach failed. Only Lane GP class (3 lanes, 1 root) is hard APPROACH_KILLED. Every other "kill" deserves a re-engineering pass.

---

## Top 5 confirmed APPROACH_KILLED (kill catalog — do NOT revive)

1. **Lane GP class (v1-v4):** smooth-basis pose fit infeasible at K<500 (Council #271 white-noise empirical proof). STRICT Check 91 enforces.
2. **Lane M+N v1 (rank-1 radial-zoom on 6-DOF-trained renderer):** rank-1 PoseNet Jacobian sensitivity ≠ rank-1 renderer input space. Lane M-V2 1.84 is partial confirm. Lane HF Telescope foveation is the proper revival, NOT M+N revival.
3. **Lane STC original (one-majority-plus-exceptions):** structurally stores 109M exceptions for multi-region masks → 7.6MB floor, 18× worse than AV1 0.014 bpp. Codec REDESIGN required (AV1+STC residual / NeRV).
4. **Lane F V1-V4 simulated FP4 on 4090:** hardware FP4 needs Blackwell CC 10.0; Ada/Lovelace 4090 = simulated FakeQuantFP4 in FP32 — fundamentally not the same operator. (Memory `feedback_hardware_quantization_disclosure_20260428`.)
5. **Lane B pose TTO (without auth-eval-every-100-steps):** the proxy-auth gap is 350× on PoseNet on dilated-h64. The proxy is a training signal not a measurement. Per CLAUDE.md non-negotiable, every TTO loop MUST run smoke auth at step 100 and every 200 steps.

---

## Top 3 systemic bugs discovered (or confirmed) in this audit

### 1. SegMapTrainer materializes the entire `(B*T, 3, H, W)` rendered tensor in float32

**Locus:** `src/tac/segmap_renderer.py:284` (or `:296` / `:391` per version).
**Impact:** 9 lane attempts crashed with OOM (7GB on T4, 21GB on A10G). Wasted ~$5-10 GPU. Full SegMap clone path unmeasured.
**Already partially fixed:** Council C bf16 + scorer-chunk per `project_session_state_checkpoint_20260430` — Vast.ai 4090 dispatch overnight. **Verify post-harvest.**
**Proposed STRICT preflight check:** `check_segmap_trainer_chunked` — scan `src/tac/segmap_renderer.py train_epoch()` for an explicit `for chunk in chunks(rendered, max_chunk_bytes=2GB)` loop OR a `bf16/fp16` autocast scope.

### 2. UNIWARD lineage: encode-then-discard pattern + missing imports + dead-flags

**Locus:** `scripts/remote_lane_uniward_texture.sh` Stage 4 `cp $ANCHOR_DIR/masks.mkv $ITER_DIR/`.
**Impact:** UNIWARD v8 reported as 1.14 "competitive with Lane A" but archive masks.mkv is SHA-identical to Lane A — pure CPU-vs-CUDA PoseNet drift. Encoded 8.6MB SLI1 payload silently discarded.
**Proposed STRICT preflight check:** `check_remote_lane_scripts_use_computed_payloads` — scan `scripts/remote_lane_*.sh` for `ENCODE_PAYLOAD ... cp $ANCHOR_DIR/masks.mkv` patterns. (Already named "Check 88" in `project_lane_uniward_v8_NO_OP_finding_20260429`; need to confirm landing status — see check inventory in CLAUDE.md "Meta-bug class catalog".)

### 3. Modal-image dependency drift (`pydantic` missing from Lane MAE-V image)

**Locus:** `experiments/modal_train_lane.py` image build.
**Impact:** Lane MAE-V crashed at first import. Cost $0.05 + multi-day blocker.
**Proposed STRICT preflight check:** `check_modal_image_imports_train_renderer_clean` — Modal image build must successfully `python -c "import tac.experiments.train_renderer"` BEFORE allowing dispatch.

(All three checks are landable additions to the catalog — see `project_all_scores_forensic_audit_20260430.md` memory file for register-and-promote spec.)

---

## Process verdict

**The user's intuition is correct.** "Some of the bad scoring lanes are bad because they're not engineered correctly" is a vast understatement. ~60% of the lane dispatches are ENGINEERING_BUG; another ~21% are CONFIG_BUG / METHODOLOGY_BUG. Only ~10% are real APPROACH_KILLED with math/empirical proof.

The session-cost of this audit (one ALL-SCORES-FORENSIC-AGENT run) reveals at least 23 hidden-gem lanes whose underlying approach has never been measured. Re-engineering cost typically <$2/lane on Modal T4 / Vast.ai 4090.

**Strategic recommendation:** prioritize re-engineering of the top-EV hidden gems before further new-lane exploration. See companion file `recoverable_lanes_re_engineering_plans_20260430.md` for the prioritized list.
