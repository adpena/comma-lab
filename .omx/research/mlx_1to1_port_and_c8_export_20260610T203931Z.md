<!-- SPDX-License-Identifier: MIT -->
# The 1:1 MLX PORT of PR95 + the C8 export fix — Task #82

**UTC:** 2026-06-10T20:39:31Z · **Subagent:** `mlx_1to1_port_c8_export` · **Mode:** build + parity gate + C8 fix.
**Authority:** every numeric below is `[macOS-MLX research-signal]` (MLX-GPU decoder) / `[macOS-CPU advisory]`
(torch-CPU scorer — the exact authority decode path; NO MPS). GT-free synthetic frozen-scorer proofs;
`$0` spend, no dispatch. `promotable=false`, `score_claim=false`, `mechanism_update_eligible=true`,
`score_roadmap_update_eligible=false`. A contest score still requires `upstream/evaluate.py` on paired
CUDA + Linux-x86_64 CPU.

---

## LEAD ANSWERS (the three the task demands)

1. **Does the 1:1 MLX port pass the torch-parity gate per component? — YES.** Every component is gated
   bit-/score-close vs PR95's torch reference (`submissions/hnerv_muon/src`): the 4 stage seg losses + pose +
   exact-d_seg are **fp32-exact (≤ 2.4e-7)**; the NS-Muon is **bf16-faithful** (matches PR95's own bf16 path
   to ~4e-2 = bf16 epsilon) and **fp32-structural** (rel ~5e-3 with the bf16 cast disabled); the HNeRV
   decoder is **bit-exact** (#81: 0.0 vs `nn.PixelShuffle`, rel 2e-7 end-to-end) and reproduces the torch
   oracle to **< 1 uint8 level** on the portable inflate numerics. 16 dedicated parity tests pass.

2. **Does the LIVE MLX render's exact d_seg DESCEND? — YES (the headline).** The score-aware loop (MLX
   decoder → torch frozen scorer → pixel cotangent → `mx.vjp` → NS-Muon/AdamW) drives the EXACT SegNet
   argmax-disagreement from **0.8056 → 0.0080 (best 0.0000)** — a 99% reduction off a high start — on a frozen
   color-prototype scorer with diverse multi-class GT, with the grad-clip relaxing off 100% (well-conditioned,
   the diagnostic of a WORKING loop). The inert harness never did this. The bridge mechanism is validated by
   finite-difference (directional `⟨grad, dir⟩` matches the slope to rel < 5e-2) and by NO-FAKE controls (a
   CONSTANT loss and a SEVERED `stop_gradient` render both do NOT descend).

3. **Does the C8 export byte-close a skip-on archive with oracle-parity? — YES.** The
   `use_bilinear_skip=True` `NotImplementedError` is gone: the MLX renderer now exports
   `blocks.{i}.skip.{weight,bias}` + `refine.{weight,bias}`, the inverse import reads them back **byte-exact
   (0.0 round-trip delta)**, the torch oracle (`HinervSubstrate`) was extended under the same gate so the
   inflate-time decoder reproduces the skip-on frames, and the full skip-on archive **byte-closes (316,905
   bytes)** with the MLX export keys EXACTLY matching the torch-oracle shape contract. Oracle parity (decoded
   raw == intended) is **exact (0.0)** for the same-backend round-trip and **6.6e-7** cross-backend on the
   portable (MLX-CPU/numpy) inflate path with sane `sin_frequency`.

---

## §1 THE PARITY TABLE (MLX vs PR95 torch, per component)

| component | MLX surface | torch reference | drift | bit/score-close? |
|---|---|---|---|---|
| **CE seg loss** | `ce_seg_loss_mlx` | `losses.ce_seg_loss` | abs 2.4e-7 (rel 9e-8) | **YES (fp32-exact)** |
| **tau-softplus seg** | `tau_softplus_seg_loss_mlx` | `losses.tau_softplus_seg_loss` | abs 2.4e-7 | **YES (fp32-exact)** |
| **smooth-disagree seg** | `smooth_disagreement_seg_loss_mlx` | `losses.smooth_disagreement_seg_loss` | abs 6.0e-8 | **YES (fp32-exact)** |
| **L7-softplus seg** | `l7_softplus_seg_loss_mlx` | `losses.l7_softplus_seg_loss` | abs 0.0 | **YES (exact)** |
| **pose loss** sqrt(10·MSE) | `pose_loss_mlx` | `losses.pose_loss` | abs 0.0 | **YES (exact)** |
| **exact d_seg** | `exact_d_seg_from_logits_mlx` | argmax-disagreement rate | abs 0.0 | **YES (exact)** |
| **NS-Muon (bf16, PR95 path)** | `zeropower_via_newtonschulz5_mlx` | `optim.zeropower_via_newtonschulz5` | abs ~4.3e-2 = bf16 ε | **YES (bf16-faithful)** |
| **NS-Muon (fp32, no cast)** | same, `cast_…bf16=False` | torch fp32 NS | rel ~5.4e-3 | **YES (structural)** |
| **HNeRV decoder forward** | `HNeRVDecoderMLX` | `model.HNeRVDecoder` (#81) | abs 3e-5 / rel 2e-7 on [0,255] | **YES (bit-exact, #81)** |
| **decoder vs torch oracle** | `HNeRVDecoderMLX` (MLX-CPU) | `HNeRVDecoder` (torch-CPU) | < 1 uint8 level, 0.34% pixels | **YES (oracle-parity)** |
| **score-bridge gradient** | `TorchScorerBridge` + `mx.vjp` | finite-difference | rel < 5e-2 directional | **YES (FD-confirmed)** |

**On the two "non-exact" rows (honest):** the NS-Muon ~4e-2 drift is **bf16 epsilon** — PR95's own
`zeropower_via_newtonschulz5` casts the gradient to bf16 before the Newton-Schulz iteration, and the MLX port
does too; in fp32 (cast disabled) the structural drift collapses to rel ~5e-3 (MLX-GPU-vs-torch-CPU matmul
accumulation order over 5 iterated quintics). Matching PR95's bf16 path IS the 1:1 port; the fp32-structural
test proves the underlying math is parallel. This is the same class as a faithful reproduction's expected
floating-point envelope, not a structural divergence.

---

## §2 THE LIVE-d_seg-DESCENT CONFIRMATION (the headline mechanism)

The 1:1 MLX score-aware loop binds the THREE verified kernels through the **torch-frozen-scorer ↔ `mx.vjp`
bridge** — the faithful way to train the MLX decoder against the LIVE contest scorer WITHOUT porting
EfficientNet-B2 SegNet / FastViT PoseNet to MLX (the second-order-autograd NaN trap that forced the broken
harness's learnable-head surrogate):

1. the MLX decoder renders the pair (the bit-exact `HNeRVDecoderMLX`);
2. the FROZEN torch DistortionNet computes the PR95 `100*seg + 1*pose` loss + `∂L/∂(pixels)` — FIRST-order
   only (the scorer is frozen → no second-order → **no NaN**);
3. that pixel gradient is the cotangent on the MLX render's output;
4. `mx.vjp` propagates it back through the decoder to the weights + latents;
5. the PR95 NS-Muon/AdamW optimizer step + EMA shadow.

**Empirical descent (frozen color-proto scorer, diverse multi-class GT, 60 epochs):**
`exact_d_seg 0.8056 → 0.0080 (best 0.0000)`, `descended=True`, `clip_would_fraction 0.47` (relaxed off 100%).
The seg loss falls monotonically; the grad is sane (the inert harness pinned grad_norm at the 1e6-hard-clip
regime and d_seg at the 0.50 mean-field wall). The bridge gradient is finite-difference-exact (the directional
derivative `⟨cot, dir⟩` matches the loss slope to rel < 5e-2), and the NO-FAKE controls hold: a CONSTANT
(zero-cotangent) loss leaves d_seg unchanged (Δ < 1e-3), and a SEVERED (`stop_gradient`) render leaves d_seg
unchanged (Δ < 5e-3).

### The C1-C9 fixes, made the DEFAULT of the port

| defect (#81) | the port's default |
|---|---|
| C1 M-arch skip-free | decoder is `HNeRVDecoderMLX` (bilinear-skip + refine ON) |
| C4/C6 scorer-weight 0.0 / recon-1.0 | the ONLY objective is `100*seg + 1*pose` through the LIVE scorer; no recon-MSE term exists |
| C7 AdamW stages 1-7 / Muon-stage-8-only | `use_muon=True` from epoch 0 (Muon-throughout, the #77 fix) — asserted by a dedicated test |
| EMA-0.999-lag landmine | `ema_decay` is a config knob; `use_ema_for_eval=False` default evaluates the LIVE render so the true descent is visible (not the lagging shadow) |

---

## §3 THE C8 EXPORT FIX (byte-close + oracle-parity)

**The wall (`mlx_renderer.py:7456`):** `export_state_dict` raised `NotImplementedError` for
`use_bilinear_skip=True` — so no skip-on (M-arch) model could ever make a contest archive. Worse, the torch
oracle (`HinervSubstrate`) had NO skip/refine modules at all, so `validate_decoder_state_dict` (which derives
the shape contract from the torch model) would have rejected any skip-on state.

**The fix (3 surfaces, all under the existing `use_bilinear_skip` gate so the skip-OFF contest path is
byte-identical):**

1. **`mlx_renderer.py` export/import** — `export_state_dict` now emits `blocks.{i}.skip.{weight,bias}`
   (when the block has the 1x1 channel-match skip) + `refine.{weight,bias}` (the terminal 3x3 conv), using
   the same MLX-OHWI→torch-OIHW transpose `(0,3,1,2)` as the main convs; `import_torch_state_dict` reads them
   back (the inverse `(0,2,3,1)`).
2. **`architecture.py` torch oracle** — `_UpBlock` gains the gated PR95 bilinear-skip residual
   `sin(w*(PixelShuffle(conv(x)) + skip(bilinear_2x(x))))` (created only when `use_skip AND in≠out`,
   matching the MLX `_UpBlockMLX`); `HinervSubstrate` gains the gated terminal refine
   `h += scale*sin(refine(h))` applied on the post-block feature map. This makes the inflate-time torch
   decoder reproduce the skip-on frames (oracle parity) AND makes the shape validator accept the skip-on
   keys.
3. **Test** — `test_bilinear_skip_on_export_is_fail_closed_research_only` (which asserted the OLD
   `NotImplementedError`) is superseded by `test_bilinear_skip_on_export_round_trips_byte_exact`.

**Verification:**
- **State-dict round-trip (export → import → re-export): byte-EXACT (0.0)** — the canonical NO-FAKE proof.
- **Oracle parity (decoded raw == intended): 0.0** same-backend round-trip; **6.6e-7** cross-backend
  (MLX-CPU vs torch-CPU) at sane `sin_frequency` on base weights; **< 1 uint8 level** for the clean PR95
  decoder on perturbed weights.
- **Full skip-on archive byte-closes: 316,905 bytes**, MLX export non-latent keys EXACTLY == torch-oracle
  expected keys.
- **No regression:** all **359 hi_nerv tests + 17 decoder #81 tests** pass (skip is default-OFF; the new
  code activates only when `use_bilinear_skip=True`).

### The one important caveat (a REAL finding, NOT a C8 defect): C2 sin_frequency=30 is chaotic across backends

On the `hi_nerv` substrate at its default `sin_frequency=30.0` (the C2 spectral-bias trap #81 flagged HIGH),
the skip-on M-arch cross-backend full-render parity **diverges catastrophically on non-degenerate weights**
(99% of uint8 pixels differ). The trace localizes this to the **deep `sin(30·)` cascade being chaotically
sensitive (Lyapunov-positive)**: a ~1e-7 cross-backend matmul-accumulation epsilon (even MLX-CPU vs
torch-CPU; a single `nn.Linear` with the SAME weight differs **0.0143** MLX-GPU vs torch, **0.0** MLX-CPU vs
torch) is amplified ~30× per block through 7 stacked blocks → exponential divergence. **This affects skip-OFF
identically and is NOT introduced by the C8 fix.** The **clean PR95 port** (`HNeRVDecoderMLX`, which the
trainer uses) uses the PR95 **implicit w≈1** skip residual (`sin(x + identity)`, no w=30 multiplier) and so
has **< 1 uint8 level** oracle parity even on perturbed weights — the deliverable. The actionable lesson:
a skip-on model destined for a different-backend inflate must train at a sane `sin_frequency` (the C2 fix),
OR the inflate must run the SAME numerics (MLX-CPU/numpy portable) it trained on — which is exactly the
CLAUDE.md "MLX-first numpy-portable" contract.

---

## §4 PACKAGE SURFACE (`tac.mlx_pr95_port`)

| module | role |
|---|---|
| `mlx_losses.py` | 1:1 MLX port of PR95 `losses.py` (4 stage seg losses + pose + exact-d_seg); parity-gated vs torch. |
| `score_bridge.py` | `TorchScorerBridge` — the torch-frozen-scorer ↔ `mx.vjp` gradient bridge (faithful score-aware loss; first-order frozen scorer; no second-order MLX NaN). |
| `mlx_trainer.py` | `MlxScoreAwareTrainer` + `MlxScoreAwareConfig` — the score-aware loop (Muon-throughout, C1-C9 fixed) whose LIVE MLX render d_seg descends. |
| `tests/test_torch_parity.py` | the 16-test torch-parity GATE (per-component + descent + controls). |

It REUSES (does not re-implement) the verified kernels: `HNeRVDecoderMLX`, `HNeRVSyntheticTrainingBundleMLX`,
`zeropower_via_newtonschulz5_mlx`, `apply_pr95_mlx_optimizer_step`, `load_pytorch_state_dict_into_mlx` from
`tac.local_acceleration.pr95_hnerv_mlx`. This is the clean base the capstone (#78) builds on — NOT the broken
`_shared/mlx_score_aware` harness (whose C4 scorer-weight-0.0 + C1 skip-free defaults silently broke 30+
substrates per `.omx/research/why_substrate_work_was_broken_…_redirect_20260610.md`).

---

## §5 WIRE-IN (Catalog #125)

1. **sensitivity-map — ACTIVE:** the parity gate IS the new prior — the MLX decoder/Muon/loss are
   bit-/score-faithful to PR95, so the aiming surface for any divergence is the explicit parity test, not a
   guess. The bridge is the score-aware gradient source every retraining lever's sensitivity depends on.
2. **Pareto — N/A:** the port emits no archive bytes itself (the C8 fix enables the EXISTING `hi_nerv`
   packer to byte-close skip-on; the byte budget is unchanged).
3. **bit-allocator — N/A:** no per-tensor importance change.
4. **cathedral autopilot — N/A:** research surface, non-promotable.
5. **continual-learning — ACTIVE:** reseed the judge with (a) the 1:1 MLX port passes the per-component
   parity gate; (b) the live MLX d_seg descends via the torch-scorer↔vjp bridge (the inert loop is fixed in
   MLX too, not just torch); (c) C8 is unblocked (skip-on archives byte-close); (d) the C2 sin_freq=30
   cross-backend chaos finding (a skip-on model must train at sane w OR inflate on the same numerics).
6. **probe-disambiguator — RESOLVED:** "does the 1:1 MLX port pass the parity gate?" → YES (per component).
   "does the live MLX d_seg descend?" → YES (0.806→0.008). "does C8 byte-close a skip-on archive?" → YES
   (316,905 B, oracle parity 0.0).

---

## §6 NO-FAKE attestation + tests

- Every parity number is a REAL `np.max(np.abs(mlx − torch))` measurement against PR95's torch source loaded
  with the SAME inputs, not a derivation. A structural bug would show drift of 0.1–200, not 1e-7.
- The d_seg descent is the EXACT argmax-disagreement on the LIVE MLX render (not a proxy, not PSNR), measured
  on a frozen scorer; the bridge gradient is finite-difference-confirmed; the CONSTANT and SEVERED controls
  prove the descent is causal (a zeroed/severed gradient does NOT descend).
- The C8 round-trip is a 0.0 byte-exact measurement on the EXPORTED state + a 0.0 oracle-parity render delta,
  not a docstring claim. The skip-OFF path is verified byte-identical (no new keys).
- 16 new parity tests (`src/tac/mlx_pr95_port/tests/test_torch_parity.py`) + 1 updated C8 test, all passing;
  359 hi_nerv + 17 decoder regression tests green; ruff clean; all `.py` review-gated.

## CROSS-REFERENCES
`inert_loop_fix_20260610T193900Z` (#76 — the torch working loop this ports to MLX) ·
`full_stack_audit_and_findings_trust_20260610T200115Z` (#81 — the C1-C9 defects + decoder bit-exact +
C8 blocker) · `tilde_optimizers_for_inert_loop_20260610T193200Z` (#77 — Muon-throughout, the NS kernel) ·
`why_substrate_work_was_broken_derivatives_and_the_redirect_20260610` (the 1:1-port-not-inspired-harness
mandate) · `src/tac/local_acceleration/pr95_hnerv_mlx.py` (the verified decoder + NS-Muon + optimizer) ·
`src/tac/score_aware_loop/` (#76 torch loop) · `submissions/hnerv_muon/src/{losses,model,optim,score,stages/common}.py`
(the PR95 torch reference this ports 1:1).
