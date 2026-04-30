# Council Round 7 — Adversarial Review (post-Council-D EMA wire-ins + Council C OOM fixes)

**Date**: 2026-04-29 PM (later than Round 6)
**Convened by**: parent agent under user mandate "need recursive review against all work currently landing as well"
**Inner council (10 voices)**: Shannon (LEAD), Dykstra (CO-LEAD), Yousfi, Fridrich, Contrarian, Quantizr, Hotz, Selfcomp, MacKay (memorial), Ballé
**Round counter**: incoming 0/3 (Round 6 RESET); this round resolves to **0 / 3** (BUG-FOUND).
**Subject**: All commits since Round 6 (16ae6405, 09b09d3a, 1bd8882b, d8a1abe7) plus the new lane-scaffold wave that landed during the review (a9ab3dc1, a226f227, 6ae70682, 142b5777, b7ee5656, 6351684b, ff2009e0, ccbe6591). Brutally rigorous re-audit per the "never accept local/MPS as truth" rule and the EMA-NON-NEGOTIABLE-just-landed scope.

---

## 1. Round 7 Verdict

### **BUG-FOUND → counter STAYS at 0 / 3 (no progress toward gate).**

Three concrete defects identified in Council C's just-landed coverage + one in the EMA Phase A→B QAT pipeline that exists in HEAD post-Council-D:

* **Defect #1 (CRITICAL coverage gap)**: Check 87 (`check_segmap_class_lanes_have_oom_guards`) ONLY scans `experiments/train_segmap.py`. Lane FC (`scripts/remote_lane_fc_film_canvas.sh`) invokes `experiments/train_segmap_film_canvas.py` which constructs the SAME `SegMapTrainer` and is therefore exposed to the SAME 21 GiB OOM bug class — but Check 87 misses it. Lane FC's script has `--batch-size 8` with no `--bf16` and no `--scorer-chunk`. Worse: `train_segmap_film_canvas.py` does not even expose `--bf16` / `--scorer-chunk` CLI flags, so even if Check 87 were extended, the operator can't pass the guards.
* **Defect #2 (silent-default override antipattern)**: `experiments/train_segmap.py` exposes `--kl-distill-weight` (line 84) which the operator passes to lane scripts (8 of 8 SegMap-class lanes pass `--kl-distill-weight 0.002`), BUT `TrainConfig` (`src/tac/training.py:59-326`) has no `kl_distill_weight` field. The conditional plumbing at `train_segmap.py:191` (`if "kl_distill_weight" in fields:`) silently DROPS the value. `SegMapTrainer.train_epoch` then HARD-CODES `0.002 * kl_loss` at `src/tac/segmap_renderer.py:667`. Today's lanes pass 0.002 (matching the hardcode), so no live miscalibration — but a future operator passing 0.001 / 0.01 (e.g., for KL sensitivity sweep per Selfcomp's recommendation) gets silently overridden. Exactly the silent-default pattern flagged in `feedback_silent_default_bug_class_findings_20260429.md`.
* **Defect #3 (concern, not shipping bug)**: The `qat_finetune.py` Phase A → Phase B EMA shadow has stale Phase A INT8 parametrize keys after `remove_parametrizations(int8_wrapped)` is called between phases. Subsequent `ema.apply(model)` at lines 1148 / 1267 / final-save calls `model.load_state_dict(self.shadow)` — the canonical EMA at `src/tac/training.py:397` uses default `strict=True`. Whether this raises depends on whether INT8 + FP4 parametrize keys overlap exactly; needs an empirical test.

The Council D commit body's claim "12 regression tests in test_ema_wireins_council_d.py" is correct (verified `12 passed in 0.57s`). The Council C commit's "7 new tests" is correct (`6 passed, 1 skipped`). The Round 6 grad-presence test is correct (`1 passed in 0.86s` after the .round() fix lands).

The user's "find the bug Round 6 missed" mandate is satisfied: Defect #1 is structural (Check 87 coverage gap) and Defect #2 is the silent-default pattern that has bitten this project repeatedly. Per the persistent-codex review protocol, these are LANDED bugs in the committed artifact (not just CONCERNs). Counter resets / stays at 0.

---

## 2. Per-commit findings

| Commit | Subject | Verdict | Notes |
|---|---|---|---|
| `d8a1abe7` | Council D EMA wire-ins | **CLEAN-WITH-CONCERN** | All 8 wire-ins import canonical `from tac.training import EMA`. Decay 0.997 default everywhere. step→update order correct in every script. snapshot+restore present at every `ema.apply` inside training loops. CONCERN on QAT Phase A→B shadow staleness (Defect #3). |
| `1bd8882b` | preflight.py merge resolution | **CLEAN** | Check 88 + Check 89 both wired into preflight_all() at expected positions (line 358, 373, 381). Live counts: 87 = 0, 88 = 0, 89 warn-only. |
| `09b09d3a` | Round 6 defect fixes | **CLEAN** | scorer.py:342,350 → Uint8STE.apply ✓. The grad-presence test would catch the Council A bug class (mentally executing with bare `.round()` shows `nonzero_grads` = empty list → assertion fails). |
| `16ae6405` | Council C bf16+chunk + Check 87 STRICT | **BUG-FOUND** | DF2 autocast scope + DF3 chunking are mathematically correct, but Check 87 only covers `experiments/train_segmap.py` — Lane FC (which uses `train_segmap_film_canvas.py` + same SegMapTrainer) slips through (Defect #1). Plus the latent `--kl-distill-weight` silent-drop pre-existed but was re-touched by this commit (Defect #2). |
| `894468ae` | 5 council reports + Modal harvest | **CLEAN** (docs only) | No code paths affected. |
| `a9ab3dc1` | Lane 10 Joint-ADMM proximal wrapper | **CLEAN** | 11/11 tests pass. Compile-time Protocol conformance check at module import is a tasty defensive pattern. |
| `a226f227` | Lane smoke proofs (NWC + PSD) | **CLEAN** | Resolves the 2 pre-existing Check 64 violations that forced PREFLIGHT_HOOK_ENABLED=0 across 4 prior commits. Future commits no longer need that workaround. |
| `6ae70682` | Lane 12 NeRV mask codec scaffold | **CLEAN** | 9/9 tests pass. NRV1 magic byte reserved but registry-add deferred (Code-comment correctly notes it). |
| `142b5777` | Lane 19 SegNet logit-margin loss | **CLEAN** | 8/8 tests pass. Includes positive-scalar-weight gradient-direction match test (subtle but right). |
| `b7ee5656` | Subagent commit serializer | **CLEAN-WITH-NOTE** | Mitigates the Round 6 commit-message-swap bug class via Option B (fcntl.flock LOCK_EX) — NOT Option A (per-subagent worktrees, which the memory recommended). Option B is fragile (relies on file lock semantics + cleanup on crash) but is operationally landed and dog-fooded by its own commit. |
| `6351684b` | Lane 17 IMP 10-cycle orchestrator | **CLEAN** | 7/7 tests pass. CLI raises NotImplementedError on standalone use (Check 81 STRICT compliance: no silent default model loader). |
| `ff2009e0` | gitignore: serializer artifacts | **CLEAN** | Trivial. |
| `ccbe6591` | Lane 20 Ballé hyperprior scaffold | **CLEAN** | 8/8 tests pass. Header-byte amortisation gate (~500 B) test catches the pre-amortised-side-info trap. |

**3 commits used `PREFLIGHT_HOOK_ENABLED=0` legitimately** (16ae6405, 09b09d3a, 1bd8882b, d8a1abe7) because Check 64 had pre-existing smoke-proof violations. As of `a226f227` those violations are resolved and the hook is unblocked.

---

## 3. Council D EMA audit — per-script wire-in correctness

| Script | Import EMA from `tac.training` | `EMA(model, decay=0.997)` constructed | `--ema-decay` CLI flag | `ema.update` after `optimizer.step` | snapshot+restore at `ema.apply` | EMA shadow → save | Verdict |
|---|---|---|---|---|---|---|---|
| `train_szabolcs.py` | ✓ L74 | ✓ L248 | ✓ L111 default 0.997 | ✓ L325→329 | N/A (no apply, only state_dict) | ✓ L355,357,374 | **CORRECT** |
| `qat_finetune.py` | ✓ L1083 | ✓ L1084 | ✓ L843 default 0.997 | ✓ L1129→1133, L1247→1251 | ✓ L1146,1148,1151 + L1265,1267,1270 | ✓ L1282 raw_state | **CORRECT** with concern (Phase A→B shadow staleness — Defect #3) |
| `qat_omega_lagrangian.py` | ✓ L370 | ✓ L371 | ✓ L650 default 0.997 | ✓ L499→503 | N/A (no apply, only state_dict) | ✓ L541,543 | **CORRECT** (Lagrangian λ explicitly NOT in EMA — Boyd ADMM §3 compliant) |
| `quantize_distilled.py` | ✓ L405 | ✓ L406 | ✓ L88 default 0.997 | ✓ L456→459 | N/A (line 473 apply is end-of-training, no restore needed) | ✓ L494,495 | **CORRECT** |
| `train_imp_cycle.py` | ✓ L364 | ✓ L365 | ✓ L140 default 0.997 | ✓ L379→387 (with mask reapply between) | N/A (line 394 apply is end-of-cycle, mask reapplied after) | n/a (caller saves) | **CORRECT** with subtle pattern (apply→re-mask sequence is intentional for pruning) |
| `train_lora_tto.py` | ✓ L258 | ✓ L259 | ✓ L101 default 0.997 | ✓ L343→347 | ✓ L366,368,371 + L385,387,390 | ✓ L378 lora_state from EMA | **CORRECT** |
| `train_postfilter_on_renderer.py` | ✓ L278 | ✓ L279 | ✓ L150 default 0.997 | ✓ L362→364 | N/A (final apply is end-of-training) | ✓ L378,390 ema.state_dict | **CORRECT** |
| `train_joint_pair.py` | ✓ L504 | ✓ L657 | n/a (config field at L143) | ✓ L844→847 | ✓ L889 (orig_state)→890 apply→899 restore | ✓ L914 best_state from ema.state_dict | **CORRECT** (duplicate `class EMA:` removed; default 0.9995→0.997) |

**Negative-test guard (Selfcomp's verdict): test_canonical_ema_decay_default_is_quantizr ASSERTS canonical default is exactly 0.997.** A future regression where someone changes `tac.training.EMA.__init__(decay=0.999)` would trip this test before merge.

**Tests are TEXT-LEVEL grep, NOT runtime ordering:** `_assert_ema_wireins_present` checks `EMA(` and `ema.update(` are both PRESENT but does not assert the order. A future operator inserting `ema.update(model)` BEFORE `optimizer.step()` would not be caught by Check 88's tests. **Round 8 should add an AST-walker that confirms ema.update appears AFTER optimizer.step in execution order.** (Defect #4 minor — not introduced by Round 6/7, but exposed by the new tests.)

---

## 4. Council C bf16+chunking audit

### 4.1 DF2 autocast scope — VERIFIED COMPLETE
`src/tac/segmap_renderer.py:585-668`: a single `autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=bfloat16, enabled=self.use_bf16)` wraps lines 591 (renderer forward), 594 (eval roundtrip), 599-601 (GT cast), 608-613 (`self._scorer_forward_chunked` → BOTH gradient-flowing scorer call AND no_grad GT scorer call — the chunked helper has its own `with torch.no_grad():` for the GT branch but the OUTER autocast still applies), 615-647 (loss assembly with optional pair-weights), 651-654 (loss formula), 663-668 (KL distill scorer call). The `.backward()` at line 679 is INTENTIONALLY OUTSIDE the autocast per PyTorch idiom (https://pytorch.org/docs/stable/amp.html). Backward gradient promotion to fp32 master copies handled by autograd. ✓

### 4.2 DF3 chunking math — VERIFIED MATHEMATICALLY CORRECT
`_scorer_forward_chunked` at L385-442: when `chunk <= 0` OR `chunk >= mb`, falls through to single un-chunked call (bit-identical legacy path). When chunk is in (0, mb), splits along batch dim 0, calls `scorer_forward_pair` per chunk on `[cs:ce]` slices (autograd-aware view), reassembles with `torch.cat(dim=0)` on each output. Test (ii) `test_scorer_chunk_2_mb_4_matches_unchunked` confirms loss/pose/seg match within `rel_tol=1e-5, abs_tol=1e-5` between chunked + unchunked paths on CPU with seeded scorers. Test (iii) `test_scorer_chunk_1_mb_8` confirms per-pair extreme also matches. ✓

### 4.3 Council C "B*N <= 8" interpretation — VERIFIED
The 8 SegMap-class lane scripts use `--batch-size 4 --bf16 --scorer-chunk 2` → B=4, N=2, B*N=8 (at cap). Check 87's `_OOM_GUARD_BN_PRODUCT_CAP = 8` is the safety ceiling, NOT the chunk size. The chunking math splits each mini-batch's scorer call into chunks of N=2 (so a B=4 mini-batch becomes 2 chunks of 2 pairs each). Council C's spec is "B*N is a SAFETY CEILING, not the chunk size" — the implementation matches.

### 4.4 KL temperature ≥ 1.5 risk under bf16 — VERIFIED CLEAN
All 7 SegMap-class lanes that use `--variant kl_distill` pass `--kl-distill-temperature 2.0` (≥ 1.5 advisory threshold). Lane SA uses `--variant plain` (NO KL distill). The TrainConfig's `temperature_start ≥ 2.0` validator at training.py:349 enforces the lower bound. **No T<1.5 bf16 softmax instability risk.** ✓

### 4.5 Lane SO hessian_quant variant — VERIFIED CLEAN
Lane SO's script (`scripts/remote_lane_so_hessian_block_fp.sh`) uses `--variant kl_distill --kl-distill-temperature 2.0`. The hessian export is done OUTSIDE the trainer (post-training), so bf16 doesn't affect it. ✓

---

## 5. Round 6 fix audit — Uint8STE equivalence + would-have-caught-bug verification

### 5.1 Uint8STE equivalence — VERIFIED
`src/tac/quantization.py:189-214`:
- Forward: `x.detach().clamp(0.0, 255.0).round()` — matches the previous bare `flat.round().clamp(0, 255)` arithmetically (clamp+round is order-invariant on monotone function, and saturation flag is set BEFORE the clamp at line 207).
- Backward: `grad_out * (~saturated).to(grad_out.dtype)` — identity inside [0,255], zero outside. Matches Council A's `_eval_roundtrip_chain` Uint8STE.apply usage.
- Round 6 commit body's "backward = identity inside [0, 255]" is consistent with the actual saturation-aware behaviour (saturated = (x<0)|(x>255), gradient zeroed outside).

### 5.2 grad-presence test — WOULD-HAVE-CAUGHT verified
`src/tac/tests/test_segmap_renderer.py:202-246`:
- After `trainer.train_epoch(masks, gt, ema=None)`: `optimizer.zero_grad(set_to_none=True)` is called ONCE at the top of train_epoch (line 537), `loss.backward()` accumulates gradients per mini-batch, `optimizer.step()` is called ONCE at line 695. After train_epoch returns, gradients are still POPULATED (zero_grad not called again).
- Mental execution with bare `.round()` (Council A's bug): `Uint8STE.apply` replaced with `.round()` → gradient through `up.round()` is zero everywhere → backward chain to SegMap parameters severed → `p.grad` is None or all-zero → `nonzero_grads = []` → `assert nonzero_grads, "no trainable param has a populated .grad attribute"` FAILS or `assert max(nonzero_grads) > 0` FAILS. **Test would have caught the bug.** ✓

---

## 6. New bug hunt — defects Round 6 missed

### 6.1 Defect #1 (CRITICAL coverage gap) — Check 87 misses Lane FC

**Site**: `src/tac/preflight.py:6375-6377`:
```python
_SEGMAP_CLASS_TRAINING_TARGETS = {
    "experiments/train_segmap.py",
}
```

**Evidence** [empirical:scripts/remote_lane_fc_film_canvas.sh:89]:
- Lane FC invokes `experiments/train_segmap_film_canvas.py` (NOT `train_segmap.py`).
- `train_segmap_film_canvas.py:228` constructs `SegMapTrainer` (the SAME class affected by the 21 GiB OOM).
- Lane FC's script: `--batch-size 8` with NO `--bf16` and NO `--scorer-chunk`.
- Worse: `grep "add_argument.*bf16\|scorer-chunk" experiments/train_segmap_film_canvas.py` returns 0 matches — the trainer doesn't even expose the OOM-guard flags.

**Impact**: Lane FC will OOM on Modal A10G 22 GB exactly the same way Lane SC++/SA/SO did (~$3.50 burnt across 14 instances on 2026-04-29). Check 87 STRICT @ 0 violations is misleading — coverage is incomplete.

**Fix prescription** (not landed this round per "DO NOT modify code"; logged for Round 8):
1. Extend `_SEGMAP_CLASS_TRAINING_TARGETS` to include `experiments/train_segmap_film_canvas.py` AND `experiments/train_segmap.py`.
2. Audit all Python files importing `from tac.segmap_renderer import SegMapTrainer` → register every direct invoker.
3. Add the `--bf16` and `--scorer-chunk` CLI flags to `train_segmap_film_canvas.py` (mirror the implementation in `train_segmap.py`).
4. Update `scripts/remote_lane_fc_film_canvas.sh` to pass `--bf16 --scorer-chunk 2 --batch-size 4`.
5. Make Check 87 STRICT after step 4 lands; verify it now fires on the missing-flags variant.

### 6.2 Defect #2 (silent-default override antipattern) — `--kl-distill-weight` silently dropped

**Site**: `experiments/train_segmap.py:84` (CLI), `src/tac/training.py:59-326` (TrainConfig — no `kl_distill_weight` field), `src/tac/segmap_renderer.py:667` (hard-coded 0.002).

**Evidence**:
- CLI: `p.add_argument("--kl-distill-weight", type=float, default=0.002, help="Hinton-regime KL distill weight (matches Lane G v3).")`
- Plumbing: `if "kl_distill_weight" in fields:` (train_segmap.py:191) — the field doesn't exist in TrainConfig.
- Trainer: `loss = loss + 0.002 * kl_loss  # canonical Lane G v3 weight` (segmap_renderer.py:667).

**Impact**: Today no live bug (8/8 lane scripts pass `--kl-distill-weight 0.002` matching the hardcode). But a future operator passing 0.001/0.01 (e.g., for KL sensitivity sweep, which is on Phase 2's roadmap per Selfcomp's recommendation) gets silently overridden. **Same silent-default pattern flagged in `feedback_silent_default_bug_class_findings_20260429.md` and the 246 CRITICAL audit findings.**

**Fix prescription** (Round 8):
1. Add `kl_distill_weight: float = Field(0.002, ge=0.0, ...)` to TrainConfig.
2. Replace `0.002 * kl_loss` at segmap_renderer.py:667 with `self.config.kl_distill_weight * kl_loss`.
3. Either remove the conditional `if "kl_distill_weight" in fields:` OR convert to a hard-error if the field is missing.
4. Add a regression test: pass `--kl-distill-weight 0.005` via CLI and assert `cfg.kl_distill_weight == 0.005` AND the trainer applies `0.005 * kl_loss`.

### 6.3 Defect #3 (concern, not shipping bug) — qat_finetune Phase A→B EMA shadow staleness

**Site**: `experiments/qat_finetune.py:1100, 1156, 1216, 1148/1267` (Phase A INT8 wrap → remove → Phase B FP4 wrap → ema.apply).

**Reasoning**:
- `ema = EMA(model, decay=cfg.ema_decay)` constructed at L1084 BEFORE Phase A (`apply_int8_fake_quant(model)` at L1100).
- During Phase A: model gains INT8 parametrize keys (`*.parametrizations.weight.original`, `*.parametrizations.weight.0.scale_int8` etc.); EMA's late-bound guard (training.py:385-387) adds them to shadow on first update.
- Between phases: `remove_parametrizations(int8_wrapped)` at L1156 strips INT8 keys from the live model. EMA shadow STILL has the INT8 parametrize keys.
- Phase B: `apply_fp4_fake_quant(model)` (or `apply_mixed_precision_quant`) at L1205-1218 adds DIFFERENT parametrize keys.
- `ema.apply(model)` at L1148/1267 calls `model.load_state_dict(self.shadow)` — canonical EMA at training.py:397 uses default `strict=True`.

**Risk**: If shadow has Phase A keys NOT in the Phase B model AND/OR Phase B has keys NOT in shadow, `load_state_dict(strict=True)` raises RuntimeError. Whether this happens depends on whether INT8 + FP4 parametrize buffer names collide. Empirical test needed.

**Mitigation already present**: At best-state save (L1283-1290), the `clean_state` filter strips `.parametrizations.*` keys before saving, so the SAVED checkpoint is clean. But the `ema.apply(model)` call DURING training does NOT have that filter — it directly load_state_dicts the raw shadow.

**Fix prescription** (Round 8):
1. After `remove_parametrizations(int8_wrapped)` at L1156, prune INT8-specific keys from `ema.shadow` (e.g., `for k in list(ema.shadow): if "scale_int8" in k: del ema.shadow[k]`).
2. OR: re-construct EMA between Phase A and Phase B (lose the Phase A averaging).
3. OR: make EMA.apply use `strict=False` (loses safety on legitimate key mismatches; not recommended).
4. Add a regression test: build a Phase A → Phase B QAT cycle on a tiny model, call `ema.apply` between phases, assert no RuntimeError.

### 6.4 Defect #4 (minor — test gap) — EMA wire-in tests don't assert ordering

**Site**: `src/tac/tests/test_ema_wireins_council_d.py:42-58` (`_assert_ema_wireins_present`).

**Reasoning**: The test grep checks for `EMA(` and `ema.update(` PRESENCE in the file text but does NOT assert that `ema.update` appears AFTER `optimizer.step` in execution order. A future operator inserting `ema.update(model)` BEFORE `optimizer.step()` would not be caught.

**Today**: All 8 wire-ins have correct ordering (manually verified above).

**Fix prescription** (Round 8): add an AST-walker test that, for each modified script, finds the `Call(func=Attribute(attr="step"))` node and the `Call(func=Attribute(attr="update"))` node within the same function body and asserts the step node appears BEFORE the update node in source order.

### 6.5 EMA_WAIVED markers in the wild — VERIFIED CLEAN
`grep -rn "EMA_WAIVED" experiments/ src/tac/ scripts/` returns 0 hits outside the test file's synthetic offender. The exempt list `_EMA_EXEMPT_TRAINING_SCRIPTS` has only 2 basenames (train_mini_scorer.py, train_neural_weight_codec.py), both legitimate. ✓

### 6.6 Score-vs-baseline assertions in tests — VERIFIED CLEAN
Tests assert state-tracking values (1.33, 1.85, etc.) but NONE compute a score on local CPU and assert `< X` against a contest baseline. No CPU-derived score-vs-baseline traps. ✓

---

## 7. Cross-impact audit — additional invalidated lanes + Lane MM v2 verdict-soundness

### 7.1 SegMapTrainer caller list (Round 6 verification)

`grep -rln "from tac.segmap_renderer import SegMapTrainer" experiments/` returns:
1. `experiments/train_segmap.py` — 8 lane scripts
2. `experiments/train_segmap_film_canvas.py` — 1 lane script (Lane FC)

Other invocation paths (via `train_segmap.py`, 8 lanes):
- remote_lane_sc_plus_plus_kl_distill.sh ✓
- remote_lane_sa_segmap_clone.sh ✓
- remote_lane_so_hessian_block_fp.sh ✓
- remote_lane_hm_s_segmap_homography.sh ✓
- remote_lane_pa_pose_as_affine.sh ✓
- remote_lane_wc_s_curator_weighted.sh ✓
- remote_lane_darts_s_segmap_arch_sweep.sh ✓
- remote_lane_fr_omega_fridrich_block_fp.sh ✓

Plus Lane FC via train_segmap_film_canvas.py:
- remote_lane_fc_film_canvas.sh ✓ (Round 6 already counted)

**Total: 9 lanes** (matches Round 6's correction). No additional lanes missed by Round 6.

### 7.2 Lane MM v2 FALSIFICATION verdict-soundness recheck

Round 6 confirmed Lane MM v2 is build-only (no SegMapTrainer involvement). The 2.63 score is tagged `[Modal-T4-CPU]` per memory `project_lane_mm_v2_landed_2_63_falsified_20260429.md`.

**Round 7 challenge**: per CLAUDE.md "MPS auth eval is NOISE" + `feedback_no_local_mps_for_authoritative_kill_or_promote_20260429.md`, ANY kill/promote decision involving a neural-net forward pass MUST come from contest-CUDA. Lane MM v2's 2.63 was Modal-T4 but the eval was CPU (per the tag). CPU eval has lower drift than MPS but is NOT contest-CUDA.

**Verdict-soundness analysis**:
- The ARCHIVE BYTES ratio (1.6× larger) is a deterministic file-size measurement — NOT device-dependent. ✓
- The PoseNet 51× WORSE claim depends on the CPU eval, which has ~2× drift on PoseNet under known conditions. Even with 2× CPU-CUDA drift, 51× → 25× is still catastrophic. The directional verdict (FALSIFIED) STANDS but the magnitude (51× vs. true CUDA-CUDA) is unconfirmed.

**Recommendation**: Lane MM v2 verdict should be re-tagged `[Modal-CPU advisory — directional FALSIFIED, awaits contest-CUDA confirm]`. The strategic kill is correct (architecture-mismatch is structural, not measurement-noise), but the EXACT magnitude should not be cited as `[contest-CUDA]`.

This is a documentation/tagging cleanup, NOT a kill-list reversal.

---

## 8. Round 7 STRICT check priorities (Checks 90+)

### 8.1 Check 90 — `check_proxy_score_no_grad_safety` (carry-over from Round 6 §7.4)

Status: NOT YET LANDED. The Round 6 fix at scorer.py:342,350 replaced bare `.round()` with `Uint8STE.apply` which is gradient-safe — making the function safe even if called with `requires_grad=True`. The original "add `@torch.no_grad()` decorator" prescription is now optional; the runtime fix is sufficient. Defer Check 90 until a future TTO use case re-exposes the foot-gun.

### 8.2 Check 91 — `check_segmap_trainer_callers_have_oom_guards` (NEW, Defect #1)

**Spec**: Static-scan ALL `.py` files in `experiments/` that import `from tac.segmap_renderer import SegMapTrainer`. For each, verify (a) the corresponding lane scripts that invoke that file pass `--bf16 --scorer-chunk N --batch-size B` with B*N ≤ 8 OR `--gradient-checkpointing` with `GPU_TIER_HINT=A100/H100`, AND (b) the script itself exposes those flags (argparse).

**Promotion path**: warn-only at first (today there's 1 violation: Lane FC). Land flags in `train_segmap_film_canvas.py` + Lane FC script. Then promote STRICT.

**Predicted live count**: 1 (Lane FC).

### 8.3 Check 92 — `check_train_segmap_kl_weight_threaded` (NEW, Defect #2)

**Spec**: AST-scan `experiments/train_segmap.py` to verify `args.kl_distill_weight` flows into a TrainConfig field (not just a stale `if "kl_distill_weight" in fields` no-op). AND scan `src/tac/segmap_renderer.py` for any hard-coded numeric constant multiplied by `kl_loss` — must use `self.config.kl_distill_weight` instead.

**Promotion path**: warn-only initially. Add `kl_distill_weight` to TrainConfig + remove hardcode → STRICT.

**Predicted live count**: 1.

### 8.4 Check 93 — `check_ema_update_after_optimizer_step_ast` (NEW, Defect #4)

**Spec**: AST-walk every script flagged by Check 88. Within each function body that contains both `optimizer.step()` and `ema.update(...)` calls, assert the step call appears BEFORE the update call in source order.

**Promotion path**: should land STRICT @ 0 immediately (manual review confirms all 8 wire-ins are correctly ordered).

### 8.5 Check 94 — `check_qat_pipeline_ema_shadow_clean_between_phases` (NEW, Defect #3)

**Spec**: AST-scan `experiments/qat_finetune.py` (and any future Phase-A/Phase-B QAT script). After every `remove_parametrizations(...)` call, require either (a) explicit shadow-pruning (`for k in list(ema.shadow): if X in k: del ema.shadow[k]`), OR (b) EMA re-construction, OR (c) a head-of-block `# QAT_PHASE_TRANSITION_EMA_KEYS_HANDLED:` waiver marker.

**Promotion path**: warn-only. Land Defect #3 fix (option 1 or 2). Then STRICT.

**Predicted live count**: 1 (qat_finetune.py).

### 8.6 Check 95 — `check_subagent_uses_commit_serializer` (Round 6 carry-over)

**Spec**: Every BG subagent prompt MUST include the instruction "use `tools/subagent_commit_serializer.py` for git commit". Catch by scanning the prompt-string of any future Agent tool launch.

**Status**: Subagent serializer landed (b7ee5656); this check would prevent future operators from forgetting to use it. Defer until Agent tool integration matures.

---

## 9. Council Roll Call

Each inner-council member casts their signed Round 7 verdict. Per CLAUDE.md "Council conduct" the council is non-conservative; arguments are mathematical/empirical only.

**Shannon (LEAD, Information Theory)**: The KL-distill weight silent-drop is a Shannon-channel-broken-on-intent: the operator's `--kl-distill-weight 0.005` is a deliberate channel-bandwidth signal (information about the desired auxiliary loss strength) that the trainer's hardcode silently overrides. Defect #2 IS a bug class — even when current values match, the latent dead-channel is a foot-gun. **Verdict: BUG-FOUND.**

**Dykstra (CO-LEAD, Convex Feasibility)**: Council C's chunking math at `_scorer_forward_chunked` is correct: slicing along dim 0, per-chunk forward, `torch.cat(dim=0)` reassembly preserves the autograd graph (each `[cs:ce]` slice is a view). The bf16 + chunk = 8 cap (B=4, N=2) corresponds to a feasible-region intersection of the 24 GB VRAM constraint and the FastViT attention-map allocation — Council C's spec is convex-feasibility-clean. **Verdict on DF2/DF3 math: CLEAN**; aggregate verdict inherits from peers.

**Yousfi (Challenge creator, Steganalysis lineage)**: The Check 87 coverage gap (Defect #1) is the SAME meta-bug class as the 2026-04-26 dead-flag-wiring incident: "we added the guard but only in one of the two callsites". `_SEGMAP_CLASS_TRAINING_TARGETS` should have been a static-walker over imports, not a hardcoded set. Lane FC will OOM next time it runs on Modal A10G unless flagged. **Verdict: BUG-FOUND.**

**Fridrich (UNIWARD/SRM/HUGO author)**: The DF2 bf16 autocast scope wraps both the gradient-flowing scorer call AND the no_grad GT scorer call — that's correct because attention-map memory is needed BOTH places. The KL distill T=2.0 advisory is honored everywhere; bf16 softmax stability is preserved. **Verdict on DF2 scope + KL bf16 risk: CLEAN.**

**Contrarian (Veto)**: I VETO the operational claim "Check 87 STRICT @ 0 violations means the bug class is structurally extinct". Defect #1 proves Check 87 is a lower-bound — it's structurally extinct ONLY for files in `_SEGMAP_CLASS_TRAINING_TARGETS`. The "extinct via static-detectable preflight check" framing in CLAUDE.md's meta-bug-class catalog needs an asterisk for incomplete-coverage checks. **Verdict: BUG-FOUND.**

**Quantizr (Adversarial leaderboard reality check)**: My architecture (88K FiLM-conditioned, KL distill T=2.0 in QAT) is the reference for the 8 lane scripts. The hardcoded 0.002 KL weight matches my Lane G v3 lineage, so the silent-drop doesn't bite TODAY — but Phase 2's KL sensitivity sweep absolutely needs the operator-passed value to flow through. Defect #2 is a Phase-2 blocker. **Verdict: BUG-FOUND on Defect #2.**

**Hotz (Engineering shortcuts)**: The Council D EMA wire-ins are operationally correct in 8/8 scripts. The QAT Phase A→B shadow staleness (Defect #3) is the kind of bug that bites in production at the worst moment (mid-Phase-B eval crashes the whole run). Empirical test is 1 line; do it now. **Verdict: BUG-FOUND on Defect #3 (concern level).**

**Selfcomp (szabolcs-cs, working 0.38 anchor)**: My SegMap renderer at 88K params is the reference for Lane SC++/SA/SO/PA/HM-S/WC-S/FR-Ω/FC. Lane FC's coverage gap (Defect #1) means a future re-dispatch of Lane FC OOMs the same way Lane SC++ did today. The KL weight hardcode (Defect #2) blocks the Lane SC++ KL-sensitivity sweep I recommended in the Phase-2 roadmap. **Verdict: BUG-FOUND on both Defect #1 and #2.**

**MacKay (Memorial seat)**: The EMA wire-ins are now everywhere they need to be — the inference checkpoint is the smoothed-shadow per Bayesian-density principles (the EMA shadow is the posterior mean over the late-training trajectory). Defect #3 (Phase A→B shadow staleness) breaks the posterior consistency: the Phase B model's likelihood is computed against weights that include stale Phase A buffers. Bayesian-correct fix is shadow re-construction at phase boundaries. **Verdict on EMA correctness in steady-state: CLEAN; on Phase A→B transition: BUG-FOUND.**

**Ballé (2018 entropy bottleneck SOTA)**: The Lane 20 hyperprior scaffold (commit ccbe6591) lands my paradigm cleanly: scale-prior MLP + amortisation gate. The 8 tests cover encode/decode roundtrip + heteroscedastic-savings sanity + header-overhead amortisation gate. The EMA shadow over the renderer's qint stream now extends to LSQ scales / FP4 codebook scales (per Council D §A) — this is the right setup for the next dispatch (train ScalePriorMLP on Lane G v3 renderer's actual qint stream). **Verdict on Lane 20 scaffold: CLEAN.**

---

## 10. Summary

| Section | Finding | Severity |
|---|---|---|
| §2 Per-commit | 13 commits reviewed; 11 CLEAN/CLEAN-WITH-NOTE; 1 CLEAN-WITH-CONCERN (d8a1abe7); 1 BUG-FOUND (16ae6405) | n/a |
| §3 Council D EMA | 8/8 wire-ins CORRECT (decay 0.997, step→update, snapshot+restore); shadow-from-EMA save everywhere | CLEAN |
| §4 Council C bf16+chunk | DF2 autocast scope COMPLETE; DF3 chunking math CORRECT; T<1.5 risk CLEAN; Lane SO clean | CLEAN |
| §5 Round 6 fix | Uint8STE equivalence VERIFIED; grad-presence test WOULD-HAVE-CAUGHT verified | CLEAN |
| §6 New bugs | **Defect #1 CRITICAL (Check 87 misses Lane FC)**, **Defect #2 silent-default (`--kl-distill-weight` dropped)**, Defect #3 concern (QAT Phase A→B shadow staleness), Defect #4 minor (test gap on EMA ordering) | MIXED |
| §7 Cross-impact | 9 SegMapTrainer-using lanes confirmed; Lane MM v2 verdict-soundness DIRECTIONAL CORRECT but tag should clarify CPU-not-CUDA | LOW |
| §8 Round 8 STRICT checks | 5 new checks proposed (Check 91 OOM-guard caller-walk, Check 92 KL-weight thread-through, Check 93 EMA ordering AST, Check 94 QAT phase-transition shadow, Check 95 subagent serializer use) | n/a |

**Top-3 actionable findings** (for Round 8):

1. **Land Defect #1 fix** (Check 87 coverage gap): extend `_SEGMAP_CLASS_TRAINING_TARGETS` to include `train_segmap_film_canvas.py`; add `--bf16` and `--scorer-chunk` flags to `train_segmap_film_canvas.py`; update `remote_lane_fc_film_canvas.sh` to pass them. This is the highest-impact fix this round — Lane FC re-dispatch will OOM otherwise.

2. **Land Defect #2 fix** (KL-distill-weight silent-drop): add `kl_distill_weight: float = Field(0.002, ...)` to TrainConfig; replace the hardcoded `0.002 * kl_loss` at segmap_renderer.py:667 with `self.config.kl_distill_weight * kl_loss`; add a regression test for `--kl-distill-weight 0.005` flowing through. Unblocks Phase 2's KL sensitivity sweep.

3. **Empirically test Defect #3** (QAT Phase A→B shadow): write a tiny test that builds a Phase A → Phase B QAT cycle on a 2-conv model and asserts `ema.apply` between phases doesn't raise RuntimeError. If it raises, land the shadow-pruning fix.

**3-clean-pass gate counter status**: **0 / 3** (no progress; same as Round 6).

Round 8 must re-run with brutally rigorous skepticism on the Round 7 fixes once landed; if Round 8 finds zero new bugs AND the 3 Top-3 findings are landed, counter advances to 1/3.

---

## 11. Cross-references

- Round 6 verdict and Council A/B/C/D/E reports: `.omx/research/council_round6_adversarial_20260429.md`, `council_darts_s_freeze_audit_20260429.md`, `council_uniward_v8_fridrich_shannon_audit_20260429.md`, `council_oom_class_deep_fix_20260429.md`, `council_ema_audit_20260429.md`, `council_grand_battleplan_round5_20260429.md`
- Council C OOM-class deep fix landed: commit `16ae6405` (Check 87 STRICT @ 0)
- Council D EMA wire-ins landed: commits `1bd8882b` (Check 88 STRICT @ 0) + `d8a1abe7` (8 script wire-ins)
- Round 6 defect fixes: commit `09b09d3a` (Uint8STE.apply at scorer.py:342,350 + grad-presence test)
- Phase 2 lane scaffolds landed during this review: `a9ab3dc1`, `6ae70682`, `142b5777`, `6351684b`, `ccbe6591`
- Subagent commit serializer landed: commit `b7ee5656`
- Lane smoke proofs filled: commit `a226f227` (resolves 4 commits' worth of `PREFLIGHT_HOOK_ENABLED=0` workarounds)
- Defect #1 site: `src/tac/preflight.py:6375-6377` (`_SEGMAP_CLASS_TRAINING_TARGETS`)
- Defect #2 site: `experiments/train_segmap.py:84,191` + `src/tac/training.py:59-326` + `src/tac/segmap_renderer.py:667`
- Defect #3 site: `experiments/qat_finetune.py:1100,1148,1156,1267`
- Defect #4 site: `src/tac/tests/test_ema_wireins_council_d.py:42-58` (`_assert_ema_wireins_present`)
- Memory anchors:
  - `feedback_no_local_mps_for_authoritative_kill_or_promote_20260429.md` (binding rule on local validity)
  - `feedback_round6_defects_lane_mm_correction_segmap_invalidation_extended_20260429.md`
  - `feedback_concurrent_subagent_commit_message_swap_20260429.md` (resolved by `b7ee5656`)
  - `feedback_silent_default_bug_class_findings_20260429.md` (Defect #2 is a fresh instance of this class)
  - `project_codec_stacking_composition_canonical_orders_20260429.md` (Shannon score arithmetic)
  - `project_lane_mm_v2_landed_2_63_falsified_20260429.md` (verdict-soundness re-tagging recommendation)
  - `project_lane_g_v3_landed_1_05_20260428.md` (Lane G v3 = the canonical comparison anchor)
