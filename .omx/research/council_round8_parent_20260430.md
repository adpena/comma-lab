# Round 8 Adversarial Review — Parent Direct (subagent #265 still in flight)

**Date**: 2026-04-30
**Reviewer**: Parent agent (direct, per user "do round 8 now" mandate)
**Counter at start**: 0/3 (Round 7 reset)
**Counter at end**: see verdict below

## Cumulative landings since Round 7 (commits)

| SHA | Subject | What |
|---|---|---|
| `cb3a7647` | Round 7 Defect #2: --kl-distill-weight operator-controllable | TrainConfig + plumbing + 5/5 tests |
| `edb9561e` | Round 7 Defect #3: rebuild EMA after Phase A→B QAT (Option A) | qat_finetune.py + 3/3 tests |
| `45a51c67` | Round 7 Defect #4: AST-level EMA ordering + decay assertions | 14/14 tests |
| `a00da459` | Round 7 §7.2 retag: Lane MM v2 [Modal-T4-CPU] → [contest-CPU advisory] | docs only |
| `22a2bcd2` | Lane Ω-W-V2 Council F SAFE-LOCAL real-archive validation (+ Defect #1 swept) | 9/9 tests + Check 87 extension |
| `17e9bb6e` | Lane Joint-ADMM Council F SAFE-LOCAL 4-stream non-convex | 12/12 tests |
| **`b860710c`** | **Subagent serializer V2: per-invocation temp git index** | **Eliminates staging-race** |

## Adversarial verification

### A. Preflight passes naturally (no workaround)
```
.venv/bin/python -c "from tac.preflight import preflight_all; preflight_all(check_codebase=True, verbose=False); print('OK')"
→ PREFLIGHT OK — all 88 STRICT checks pass
```
✅ **CLEAN**

### B. Round 6/7/F test suites all pass
```
pytest src/tac/tests/test_segmap_renderer.py + test_kl_distill_weight_plumbed.py
     + test_qat_phase_a_to_b_ema_rebuild.py + test_ema_wireins_council_d.py
     + test_omega_w_v2_real_archive.py + test_joint_admm_4stream_nonconvex.py
→ 56 passed in 5.40s
```
✅ **CLEAN**

### C. No new uses of forbidden workarounds in this loop's commits
```
grep "PREFLIGHT_HOOK_ENABLED=0\|REVIEW_GATE_OVERRIDE=1" --include="*.py" --include="*.sh" -l
→ Only tools/preflight_hook.py + tools/review_tracker.py + tools/review_gate_hook.py + CLAUDE.md
  (the legitimate hook impls + the docs themselves)
```
✅ **CLEAN** — no source / commit message uses the workarounds.

### D. Eat-own-dog-food on serializer V2 fix
Commit `b860710c` was itself produced by the new serializer:
```
.omx/state/commit-serializer.log:
  outcome=committed, head_after=b860710c,
  temp_index=/Users/adpena/Projects/pact/.omx/state/.subagent-temp-index-95923-1777526731041,
  commit_seconds=146.286
```
Cleanup verified: `ls .omx/state/.subagent-temp-index-* → no matches found`.
✅ **CLEAN** — temp index path created + used + cleaned up.

### E. Lane MM v2 retag propagated correctly
Memory file + 2 council reports updated to `[contest-CPU advisory]`. Verdict (FALSIFIED) directionally stands. No lingering `[Modal-T4-CPU advisory]` in active workflow files.
✅ **CLEAN**

## Adversarial bug hunt

### F1. Round 7's Defect #1 (Check 87 coverage gap) was swept into commit 22a2bcd2 — verify content correctness
- `git show 22a2bcd2 --stat`: confirmed Check 87 extension to scan train_segmap_film_canvas.py, plus the Council F validation.
- File presence verified by direct read of preflight.py.
- Defect #1 fix is semantically correct in HEAD even though attribution is shuffled.
**Verdict**: CONCERN — attribution shuffled but content correct. Documented in subagent #264 report as known forensic note. The serializer V2 (b860710c) prevents this going forward. ✅ **DOWNGRADE TO INFORMATIONAL**

### F2. Defect #2 fix — does it actually break anything?
- `kl_distill_weight: float = Field(0.002, ge=0.0)` added to TrainConfig.
- All 8 lane scripts pass `--kl-distill-weight 0.002` — no behavior change for default value.
- `0.002 * kl_loss` → `self.config.kl_distill_weight * kl_loss` is functionally identical when kl_distill_weight=0.002.
- Test verifies kl_distill_weight=0 produces different loss than kl_distill_weight=10.
**Verdict**: ✅ **CLEAN**

### F3. Defect #3 fix — Option A rebuild EMA after Phase B FP4 wrap
- Test `test_qat_phase_a_to_b_ema_rebuild.py` regression-proves the OLD path raises RuntimeError.
- Option A creates fresh EMA after Phase B wrap → no orphan keys.
- Side effect: Phase A's EMA history is discarded at Phase B boundary. Per Quantizr's 5-stage paradigm, each stage has its own EMA; this is correct.
**Verdict**: ✅ **CLEAN**

### F4. Defect #4 AST tests — adversarial check
- `test_ema_update_called_after_optimizer_step_via_ast` uses real AST line-number ordering check.
- A hypothetical bypass: define `optimizer.step()` inside a helper method that ema.update is also called from in the wrong order. Test would NOT catch this nested case.
- But this would be a deliberate obfuscation — the rule covers training-loop top-level call site which is the canonical pattern.
**Verdict**: ⚠️ **MINOR** — test catches the canonical bug class but not deliberate obfuscation. Acceptable.

### F5. Council F Ω-W-V2 validation — the 40.98% claim
- V1 raw qint estimate computed correctly (sum of byte_count for each tensor).
- V2 encoded byte count includes header overhead (verified per-tensor).
- L1 round-trip tolerance: `≤ max_abs/4 = 3-bit floor algebra` per the codec design.
**Verdict**: ✅ **CLEAN**

### F6. Council F Joint-ADMM — corner-pinned optimum issue
- Per the council report: "all 4 streams pinned to local boundaries → 0 strictly-interior streams → KKT residual sub-prediction skipped (vacuous)."
- The test PASSES via Path (a): convergence + budget feasibility (sum=700.0 EXACTLY at budget).
- ADVERSARIAL: a STRONGER test would force interior optimum. But the test as-written DOES catch silent-non-feasibility (the bug it gates against).
**Verdict**: ⚠️ **MINOR** — test gates against the documented bug class but doesn't fully exercise the non-convex KKT theory. Phase 2 Lane 10 V2 dispatch is unblocked but the test could be strengthened in Round 9.

### F7. Hunt for new bugs
- `grep -rln "SegMapTrainer\|SegMap(" experiments/ src/tac/ scripts/remote_lane_*.sh`: same 9-lane list as Round 6 + Lane FC (defect #1 closed). No 10th lane.
- New code in commits since Round 7: all has tests, all preflight-clean.
**Verdict**: ✅ **NO NEW BUGS FOUND**

### F8. Lane MM v2 CUDA confirm priority
Per Round 7's note: "Verdict needs CUDA confirm to be promoted from FALSIFIED-on-CPU to FALSIFIED-on-CUDA. Cost: ~$0.50 Vast.ai 4090."
- Should this $0.50 spend gate the $3 SegMap re-train wave? **NO.**
- Rationale: Lane MM v2 is BUILD-only (no SegMapTrainer). Its FALSIFICATION is documentation, not blocking any active research direction. The $0.50 can run in parallel with the $3 dispatch. Council F's dispatch order (HM-S → FR-Ω) doesn't require Lane MM v2 confirmation.
**Verdict**: defer Lane MM v2 CUDA confirm; not on critical path for #3.

## Round 8 Verdict

**CLEAN** (no new bugs found that bite the narrowed dispatch). Counter advances **0/3 → 1/3**.

| Defect class | Status |
|---|---|
| `.round()` zero-grad | ✅ EXTINCT (Check 86 STRICT) |
| OOM 21GB | ✅ EXTINCT (Check 87 STRICT) |
| Missing EMA | ✅ EXTINCT (Check 88 STRICT) |
| Concurrent commit-message swap | ✅ EXTINCT (file-lock serializer) |
| Concurrent commit staging-race | ✅ EXTINCT (temp-index serializer V2) |
| UNIWARD-NO-OP | ⚠️ Check 89 warn-only, not biting today's narrowed plan |
| Display-bug masking NaN | ✅ EXTINCT (Check 85 STRICT) |
| Silent default override | ✅ EXTINCT (Check 81 STRICT + Defect #2 fix) |
| Callsite contracts | ✅ EXTINCT (Check 82 STRICT) |
| MPS-derived strategic decisions | ✅ EXTINCT (Check 83 STRICT) |

## Counter status: 1/3 — needs 2 more clean rounds before #3 approval

## #3 approval-readiness summary for the user

**Approved by Council F for dispatch**:
- Lane HM-S ($1.50, 6h Vast.ai 4090, predicted [0.32, 0.45], geometric orthogonal to KL-distill)
- Lane FR-Ω ($1.50, 6h Vast.ai 4090, predicted [0.27, 0.45], Hessian block-FP canonical Selfcomp ingredient)

**Total committed cost**: $3.00. Sequencing per Council F: HM-S first as band-calibration anchor.

**Local validations DONE** (Council F Path C):
- ✅ Lane Ω-W-V2 real-archive: 40.98% byte savings on Lane G v3 renderer.bin (central in [20%, 60%])
- ✅ Lane Joint-ADMM 4-stream non-convex: convergence + budget feasibility, Lane 10 V2 unblocked

**Phase 2 acceleration scaffolds DONE**:
- ✅ Lane 10 (Joint-ADMM water-fill V2 wrap), Lane 12 (NeRV), Lane 17 (IMP), Lane 19 (logit-margin), Lane 20 (Ballé)

## Council Roll Call — Round 8 (parent-direct)

- **Shannon** (LEAD, R(D) floors): Council F's per-lane EV analysis is mathematically sound. The narrowed $3 dispatch is the right move; HM-S band-calibration first.
- **Dykstra** (CO-LEAD, convex feasibility): the 3-clean-pass gate is at 1/3, which is mathematically meaningful — counter must reach 3/3 for full confidence. Round 9 + Round 10 still required.
- **Yousfi** (challenge designer): bf16 + scorer-chunk fix means SegMap-class lanes can finally produce gradient-flowing training. The 9-lane invalidation should be FULLY re-runnable now (3 will go on 4090 today, the others later).
- **Fridrich** (steganography): UNIWARD v8 NO-OP discovery + Council F's narrowing of stale lanes is the right kind of skepticism. Continue.
- **Contrarian** (veto power): every score band remains [prediction] until HM-S lands [contest-CUDA]. Do not promote anything else until that anchor.
- **Quantizr** (competitive intel): the 5-stage paradigm with EMA at every stage is now correctly wired (Defect #3 fix). QAT path is sound.
- **Hotz** (engineering instinct): the staging-race fix is the right structural fix. Eat-own-dog-food verification is the gold standard.
- **Selfcomp** (collaborative anchor): SegMap-class lanes finally have proper bf16+scorer-chunk. My 0.38 paradigm is reproducible from this codebase now.
- **MacKay** (info theory + Bayesian inference): Council F's [contest-CPU advisory] retag for Lane MM v2 is the right epistemic discipline. CPU is still drift-prone vs CUDA.
- **Ballé** (entropy bottleneck): Lane 20 hyperprior scaffold landed. Phase 3 dispatch ready when prerequisite Lane Ω-W-V2 CUDA validation lands.

## Top 3 actionable findings for Round 9

1. **Strengthen Joint-ADMM test** with INTERIOR-optimum 4-stream problem (Round 9 priority — current corner-pinned test is necessary but not sufficient for KKT theory verification).
2. **Spawn Round 9 review** to advance counter to 2/3.
3. **Optional $0.50 Lane MM v2 CUDA confirm** can run in parallel with HM-S dispatch — not gating.

## Should the user approve #3 yet?

**Conditional YES**:
- Counter is at 1/3 (was 0/3 at Round 7).
- All bugs in scope for #3 narrowed plan are fixed.
- Phase 2 scaffolds + permanent fixes + Council F local validations all landed clean.
- HM-S is the band-calibration anchor — its $1.50 spend is the cheapest possible CUDA validation.

**Conservative recommendation**: spawn Round 9 BEFORE dispatch (counter to 2/3) for full 3-clean-pass discipline. Round 9 should focus on the 2 minor findings above (F4 obfuscation gap + F6 ADMM interior test) plus any new commits.

**Aggressive recommendation**: dispatch HM-S now in parallel with Round 9. If HM-S lands within Council F's predicted band [0.32, 0.45], it validates the entire stack-fixing exercise of today. If not, Round 9's findings might explain why.

User's call.
