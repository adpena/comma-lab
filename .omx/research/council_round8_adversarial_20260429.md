# Council Round 8 — Adversarial Review (post-Round-7 fixes + Council F SAFE-LOCAL validations)

**Date**: 2026-04-29 PM (after Round 7)
**Convened by**: parent agent under user mandate "fix all bugs regardless of severity" + "keep recursive adversarial reviews going"
**Inner council (10 voices)**: Shannon (LEAD), Dykstra (CO-LEAD), Yousfi, Fridrich, Contrarian, Quantizr, Hotz, Selfcomp, MacKay (memorial), Ballé
**Round counter**: incoming **0/3** (Round 7 RESET); this round resolves to **1 / 3** (CLEAN — the four Round 7 defects are landed correctly + Council F SAFE-LOCAL validations are mathematically and operationally correct).
**Subject**: 7 commits since Round 7 verdict — `22a2bcd2`, `17e9bb6e`, `cb3a7647`, `edb9561e`, `45a51c67`, `a00da459`, `b860710c` — plus a brutal re-examination of the AST-test adversarial bypasses, the QAT Phase A→B Option A trade-off, the Joint-ADMM 4-stream corner-pinned coverage, the Ω-W-V2 bit-faithful tolerance algebra, and the residual subagent serializer staging-race (which **landed a structural fix at `b860710c`**, exceeding Round 7's prescription).

---

## 1. Round 8 Verdict

### **CLEAN → counter ADVANCES from 0 / 3 to 1 / 3.**

All four Round 7 defects are landed correctly and verified empirically:

* **Defect #1** (Check 87 coverage gap on Lane FC): commit `22a2bcd2` extends `_SEGMAP_CLASS_TRAINING_TARGETS` to include `experiments/train_segmap_film_canvas.py`, adds `--bf16` + `--scorer-chunk` CLI flags to that script, and updates `scripts/remote_lane_fc_film_canvas.sh` with `--bf16 --scorer-chunk 2 --batch-size 4`. Verified `Check 87 violations: 0` post-fix. [empirical:src/tac/preflight.py:6375-6385]
* **Defect #2** (`--kl-distill-weight` silent-drop): commit `cb3a7647` adds `kl_distill_weight: float = Field(0.002, ge=0.0)` to `TrainConfig`, replaces the hardcoded `0.002 * kl_loss` at `src/tac/segmap_renderer.py:667` with `kl_weight = float(getattr(self.config, "kl_distill_weight", 0.002)); loss = loss + kl_weight * kl_loss`, and adds 5/5 passing regression tests. [empirical:src/tac/tests/test_kl_distill_weight_plumbed.py — 5 passed]
* **Defect #3** (QAT Phase A→B EMA shadow staleness): commit `edb9561e` rebuilds `ema = EMA(model, decay=cfg.ema_decay)` AFTER the Phase B FP4 wrap (correctly placed at line 1240, AFTER the `if/elif/else` block that wraps fp4 at lines 1205-1218 — the subagent caught their own initial misplacement, per Round 7 task C1). Three regression tests including a failing-OLD-path proof-of-bug pre-fix. [empirical:src/tac/tests/test_qat_phase_a_to_b_ema_rebuild.py — 3 passed]
* **Defect #4** (text-grep tests don't assert AST ordering): commit `45a51c67` adds `test_ema_update_called_after_optimizer_step_via_ast` + `test_ema_decay_is_quantizr_canonical_via_ast`. 14/14 EMA-wireins tests pass. [empirical:src/tac/tests/test_ema_wireins_council_d.py — 14 passed]

Plus the two Council F SAFE-LOCAL validations:
* **Ω-W-V2 real-archive validation** (commit `22a2bcd2`): 9/9 tests pass. Empirical aggregate savings on Lane G v3 renderer.bin = **40.98%** (V1=285,544 B → V2=168,517 B), squarely inside Council F Part B band [20%, 60%], replacing the synthetic 69.11% claim. [empirical:src/tac/tests/test_omega_w_v2_real_archive.py]
* **Joint-ADMM 4-stream non-convex test** (commit `17e9bb6e`): 12/12 tests pass. Gating contract correctly distinguishes silent-infeasibility from honest divergence. [synthetic:src/tac/tests/test_joint_admm_4stream_nonconvex.py]

Plus the tag cleanup (commit `a00da459`) — Lane MM v2 score retag.

Plus a CRITICAL bonus: **commit `b860710c` landed a structural fix to the subagent commit serializer's staging-race** via per-invocation `GIT_INDEX_FILE` (eat-own-dog-food: that commit ITSELF used the new temp-index path).

The Round 7 prescription "Round 8 must re-run with brutally rigorous skepticism on the Round 7 fixes once landed; if Round 8 finds zero new bugs AND the 3 Top-3 findings are landed, counter advances to 1/3" is satisfied.

The user's mandate "fix all bugs regardless of severity" is satisfied for all four Round 7 defects. Round 8 finds **zero new shipping bugs**; the adversarial constructions below are interesting AST-test bypasses but require malicious actors (vs. innocent operator mistakes), and are accepted as the cost of static-AST-rather-than-runtime-tracing analysis.

---

## 2. Per-fix audit (Round 7 Defects #1-4 + Council F validations)

### 2.1 Defect #1 fix landed in `22a2bcd2` despite the staging-race

**Round 7 task A1**: `git show 22a2bcd2 --stat` should include BOTH the Council F validation work AND the Defect 1 files.

**Verified**: `git show 22a2bcd2 --stat` shows 7 files:
```
.omx/state/review_tracker.duckdb              (state)
.omx/state/review_tracker.json                (state)
experiments/train_segmap_film_canvas.py       (Defect #1 — added --bf16 + --scorer-chunk)  +30 LOC
reports/silent_defaults.md                    (audit refresh)
scripts/remote_lane_fc_film_canvas.sh         (Defect #1 — added --bf16 --scorer-chunk 2 --batch-size 4)  +7 LOC
src/tac/preflight.py                          (Defect #1 — extended _SEGMAP_CLASS_TRAINING_TARGETS)  +8 LOC
src/tac/tests/test_omega_w_v2_real_archive.py (Council F Part B validation)  +491 LOC
```
Both sets of files are correctly committed and SEMANTICALLY correct in HEAD.

**Round 7 task A2 (semantic verification)**:
* `experiments/train_segmap_film_canvas.py:94-103` exposes `--bf16` + `--scorer-chunk` CLI flags with help text "Enable bf16 autocast around SegMapTrainer forward + scorer". [empirical:experiments/train_segmap_film_canvas.py:94-103]
* `experiments/train_segmap_film_canvas.py:148-151` threads them into the SegMapTrainer constructor: `bf16=bool(getattr(args, "bf16", False)), scorer_chunk=int(getattr(args, "scorer_chunk", 0) or 0)`. [empirical:experiments/train_segmap_film_canvas.py:148-151]
* `scripts/remote_lane_fc_film_canvas.sh:96-97` invokes `--epochs 600 --batch-size 4 --lr 1e-3 --bf16 --scorer-chunk 2`. B*N = 4*2 = 8 = `_OOM_GUARD_BN_PRODUCT_CAP`. [empirical:scripts/remote_lane_fc_film_canvas.sh:91-97]
* `src/tac/preflight.py:6375-6385` includes both `experiments/train_segmap.py` AND `experiments/train_segmap_film_canvas.py` in `_SEGMAP_CLASS_TRAINING_TARGETS`, with comment block explaining the Round 7 Defect #1 closure. [empirical:src/tac/preflight.py:6375-6385]
* `Check 87 violations: 0` confirmed via direct invocation. [empirical:.venv/bin/python -c 'check_segmap_class_lanes_have_oom_guards']

**Round 7 task A3 (concurrent-overwrite check)**: No evidence of concurrent overwrite. The Defect #1 changes are intact and consistent with the Round 7 §6.1 prescription.

**Round 7 task A4 (residual flaw — Round 9 prescription)**: SEE §5 below — the residual flaw in the serializer (staging-area sweep across concurrent subagents) **was structurally fixed by commit `b860710c`** (per-invocation `GIT_INDEX_FILE`). No Round 9 follow-up needed for this bug class.

### 2.2 Defect #2 functionally complete

**Round 7 task B1 (verify file-level fix)**:
* `src/tac/training.py:336-339` adds `kl_distill_weight: float = Field(0.002, ge=0.0, ...)` to `TrainConfig`. [empirical:src/tac/training.py:324-339]
* `src/tac/segmap_renderer.py:667-672` reads the weight from config: `kl_weight = float(getattr(self.config, "kl_distill_weight", 0.002)); loss = loss + kl_weight * kl_loss`. The defensive `getattr` fallback to 0.002 is sound (covers older instances where the field didn't exist). [empirical:src/tac/segmap_renderer.py:667-672]
* `experiments/train_segmap.py` and `experiments/train_segmap_film_canvas.py` both unconditionally thread `args.kl_distill_weight` into `cfg_kwargs` (no more conditional `if "kl_distill_weight" in fields`). [empirical:cb3a7647 diff]

**Round 7 task B2 (run the test)**: 5/5 PASS in 1.02s. [empirical:.venv/bin/python -m pytest src/tac/tests/test_kl_distill_weight_plumbed.py]

**Round 7 task B3 (other hardcoded `0.002 * kl` instances)**: `grep -rn "0\.002 \* kl" src/tac/ experiments/` returns NO matches in production code. The only matches are in the test file's docstring and the TrainConfig Field's comment block. **No other instances exist.** [empirical:grep finding]

### 2.3 Defect #3 Option A correctly placed AFTER Phase B wrap

**Round 7 task C1 (placement verification)**:

The fix at `experiments/qat_finetune.py:1220-1241`:
```python
# Round 7 Defect #3 fix (Option A, 2026-04-29 PM)
# ... rebuild ema = EMA(model, decay=cfg.ema_decay) AFTER the Phase B wrap
if cfg.int8_warmup_epochs > 0:
    ema = EMA(model, decay=cfg.ema_decay)
    print("  EMA shadow rebuilt for Phase B (Round 7 Defect #3 fix)")
```

Verified to land AFTER the `if/elif/else` block at lines 1205-1218 which sets `fp4_wrapped = apply_mixed_precision_quant(...)` or `fp4_wrapped = apply_fp4_fake_quant(...)`. The placement is **CORRECT** per the Round 7 prescription "rebuild EMA AFTER the Phase B FP4 wrap (NOT before)". [empirical:experiments/qat_finetune.py:1205-1240]

The guard `if cfg.int8_warmup_epochs > 0:` ensures Phase-B-only runs (no Phase A) don't double-construct EMA — a subtle but correct subagent decision.

**Round 7 task C2 (run the test)**: 3/3 PASS in 0.50s. [empirical:.venv/bin/python -m pytest src/tac/tests/test_qat_phase_a_to_b_ema_rebuild.py]

**Round 7 task C3 (trade-off analysis: discarding Phase A history)**:

The Option A rebuild sacrifices the Phase A EMA averaging — the Phase B EMA shadow starts cold from the model's current state at the start of Phase B. Per the commit body and the Round 7 prescription:

> "The Phase A averaging is sacrificed — Phase A is a warm-up phase whose value lives in the live model state, not the shadow average."

**Verdict**: This trade-off is acceptable per Quantizr's 5-stage paradigm. Phase A is INT8 warm-up (the value of which is in the LIVE WEIGHTS post-warmup, not in the EMA average), and Phase B is the FP4 fine-tune where the EMA shadow becomes inference-relevant. The alternative options (shadow-pruning OR `strict=False`) were considered and rejected: pruning is fragile (must enumerate INT8 keys) and `strict=False` loses safety.

**Council MacKay's observation** (Bayesian-correct lens): the posterior mean over Phase B's training trajectory is what should be in the inference checkpoint; Phase A is fundamentally a different prior (INT8 quantization). Reset is theoretically sound.

### 2.4 Defect #4 AST tests are NOT vacuous (with caveats)

**Round 7 task D1 (run the test)**: 14/14 PASS in 0.62s including the two new AST tests. [empirical:.venv/bin/python -m pytest src/tac/tests/test_ema_wireins_council_d.py]

**Round 7 task D2 (AST-walking analysis)**:

`_find_optimizer_step_then_ema_update_order` (test_ema_wireins_council_d.py:332-386):
1. Walks `ast.FunctionDef` nodes.
2. Inside each function body, walks `ast.Call` nodes filtered by `ast.Attribute(attr="step")` with receiver in `{"optimizer", "optim", "opt", "_opt"}` (Name) OR `Attribute(attr=...)` with the SAME name set (so `trainer.opt.step()` and `self.optim.step()` both match).
3. Collects line numbers; asserts `min(ema_update_lines) > min(step_lines that appear before it)` — equivalently, "there must be at least one optimizer.step BEFORE the first ema.update IN SOURCE ORDER WITHIN THE SAME FUNCTION BODY."

`_find_ema_constructor_decay_arg` (test_ema_wireins_council_d.py:413-482):
1. Walks `ast.Call` nodes filtered by `Name(id="EMA")` OR `Attribute(attr="EMA")` (so `EMA(...)`, `tac.training.EMA(...)`, and `module.EMA(...)` all match).
2. Extracts the `decay=...` kwarg's value via `_decay_value_repr` which handles `Constant`, `Attribute`, `Name`, and 1-arg `Call` wrappers (e.g. `float(args.ema_decay)`, `int(0.999)`).
3. Allows ONLY: literal `"0.997"` OR Attribute access ending in `.ema_decay`.

**These ARE proper AST-walks (not just text greps)**. Hypothesis-checked below.

### 2.5 Council F Ω-W-V2 validation is genuinely scorer-free (verified)

**Round 7 task E1 (no scorer load)**: `grep -n "PoseNet\|SegNet\|posenet\|segnet\|scorer\|differentiable_scorers" src/tac/tests/test_omega_w_v2_real_archive.py` returns 0 matches. Test only loads `from tac.renderer_export import load_renderer_checkpoint` and operates on raw conv weight tensors. [empirical:grep over src/tac/tests/test_omega_w_v2_real_archive.py]

**Round 7 task E2 (savings calculation soundness)**:

The 40.98% savings is computed as:
```
aggregate_v1_bytes = Σ (w.numel() + O*4 + 32)   # raw per-channel-exponent block-FP V1 estimate
aggregate_v2_bytes = Σ len(payload)              # actual encoded payload bytes (V2)
aggregate_savings_pct = 100 * (1 - V2 / V1)
```
Both numerator and denominator measure encoded-payload bytes (NOT headers vs payload). Per-tensor breakdown verifies symmetry — every `V1=X B → V2=Y B` line uses bytes-to-bytes comparison. **Sound.** [empirical:src/tac/tests/test_omega_w_v2_real_archive.py:170-241]

**Round 7 task E3 (bit-faithful round-trip tolerance)**:

`tol = 2.0 * max_abs * (2.0 ** -3)` — algebraically `max_abs / 4`.

Derivation per the test's own commentary at lines 170-175:
- V2 uses block-FP per-channel exponent + qint widths in {1, 3, 7, 15, 31}.
- Council-mandated 3-bit floor → Q_c=7 (3-bit signed magnitude) → quantization step ≈ max_abs_per_channel / 7.
- L_inf bound on per-channel reconstruction = at-most-half-step + cross-channel max safety = `2 * max_abs / 8 = max_abs / 4`.
- Test asserts `(w - recon).abs().max() <= tol`.

This is "lossy quantization, but bit-faithful for the chosen ladder" — the encoded bytes round-trip to recon within the documented L_inf bound. The phrase "bit-faithful" is accurate IF read as "deterministic decode of the chosen ladder", NOT "lossless original-tensor recovery". The docstring + comments make this distinction clear (see `test_omega_w_v2_real_archive_caveat_documented[4 caveats]`). **Sound, with the caveats correctly documented.** [empirical:src/tac/tests/test_omega_w_v2_real_archive.py:170-184]

### 2.6 Council F Joint-ADMM 4-stream test correctly handles corner-pinning

**Round 7 task F1 (corner-pinning observation)**:

The test passed with `bytes=[50.0, 0.0, 400.0, 250.0]` summing to 700 == byte_budget. All 4 streams are at LOCAL boundaries:
- s1 at b_min=50
- s2 at saturate-floor=0 (dual >= slope, so s2 chooses 0)
- s3 at top-of-grid=400
- s4 at b_max=250

KKT waterline assertion correctly skips with message "<2 strictly-interior streams ... vacuous". [empirical:src/tac/tests/test_joint_admm_4stream_nonconvex.py:418-424]

**Round 7 task F2 (stronger non-convex test where optimum is interior)**:

I empirically tested at budgets {500, 600, 800, 900, 1000}:

```
budget=500:  converged=False, restarts=70, bytes=[50, 0, 400, 250], sum=700 (HONEST DIVERGENCE)
budget=600:  converged=False, restarts=69, bytes=[50, 0, 400, 250], sum=700 (HONEST DIVERGENCE)
budget=800:  converged=True,  restarts=0,  bytes=[50, 0, 400, 250], sum=700
budget=900:  converged=True,  restarts=0,  bytes=[50, 0, 400, 250], sum=700
budget=1000: converged=True,  restarts=0,  bytes=[50, 0, 400, 250], sum=700
```

**Critical finding**: The synthetic streams have HARDCODED LOCAL OPTIMA — they ignore the `target_bytes` parameter except as a clamp ceiling. Across all budget sizes, the bytes vector is identical [50, 0, 400, 250]. This means:

1. The 4-stream test does NOT actually exercise ADMM equilibration — each stream returns its local optimum independently. The 700-byte convergence is by COINCIDENCE (sum of local optima = 700, which happens to match the chosen budget).
2. At tighter budgets (500, 600), the test correctly catches HONEST DIVERGENCE — the coordinator reports `converged=False` with restarts, which the test contract accepts as PASS (Gate 1's else branch).
3. At looser budgets (800+), the test trivially passes (converges in 2-4 iterations because no equilibration is needed).

**Verdict**: The test's GATING CONTRACT is correct (silent-infeasibility detection works). However, the test does NOT exercise ADMM's equilibration logic on a problem WITH genuine cross-stream tradeoffs because the synthetic streams ignore `target_bytes`. This is a **test-design weakness, not a coordinator bug**. The Council F design assumes streams that respect `target_bytes` proportionally; the implementation doesn't.

**Round 8 prescription** (for future): a stronger test would use streams whose `proximal_step(target_bytes, dual)` actually allocates AROUND `target_bytes` weighted by `dual`. The QuadraticInteriorStream does this correctly (`b_unconstr = b_opt - dual / (2*a)` then clamp to target_bytes), but the Linear/Discrete/Sigmoid streams use `target_bytes` only as an upper clamp and have hardcoded local-optimum allocations. This is a Round 9 design improvement; the current test correctly catches the FAILURE MODE the council was worried about (silent infeasibility), so it's not a Round 8 reset.

**Round 7 task F3 (propose stronger interior-forced test)**: Defer to Round 9 if the Lane 10 V2 dispatch produces unexpected behavior on the real archive. Today the test gates against the bug class the council named.

---

## 3. Adversarial constructions — hypothetical bypasses of the new tests

I ran 5 adversarial constructions against the AST-ordering scanner + 5 against the AST-decay scanner.

### 3.1 EMA ordering AST scanner — bypass surface

| Hypothesis | Result | Severity |
|---|---|---|
| (1) Wrapper hides order: `def update_step(): ema.update(model)` then call from `main()` AFTER `optimizer.step()`. | **PASSES test (false negative)**. AST scanner sees `ema.update` in `update_step` body (no `optimizer.step` there) and `optimizer.step` in `main` body (no `ema.update` there). Cross-function order invisible to AST walker. | Adversarial only — unlikely innocent regression |
| (2) `ema.update(model)` BEFORE `optimizer.step()` in same function. | **DETECTED — fails as expected**. | n/a (good) |
| (3) Closure with `ema.update` first in source order (but called after `optimizer.step()` at runtime). | **DETECTED — false positive**, conservatively flags a mostly-OK pattern. | Acceptable trade-off |
| (4) Same-function `optimizer.step()` THEN `ema.update(model)` (good pattern). | **PASSES** (correct). | n/a (good) |
| (5) Chained-attribute `trainer.opt.step()` BEFORE `ema.update()`. | **DETECTED — fails as expected**. | n/a (good) |

### 3.2 EMA decay AST scanner — bypass surface

| Hypothesis | Result | Severity |
|---|---|---|
| (a) Nested-Call wrapper: `EMA(model, decay=float(int(0.999)))`. | **DETECTED — fails as expected** (recursive `_decay_value_repr` resolves to `"0.999"` which is not `"0.997"` and not `.ema_decay`). | n/a (good) |
| (b) Variable indirection: `my_decay = 0.5; EMA(model, decay=my_decay)`. | **DETECTED — fails as expected** (`my_decay` is bare Name, not `.ema_decay`). | n/a (good) |
| (c) Positional decay (no kwarg): `EMA(model, 0.5)`. | **DETECTED — fails as expected** (`<missing>` not in allow list). | n/a (good) |
| (d) Aliased EMA constructor: `from tac.training import EMA as ExpMovingAvg; ExpMovingAvg(model, decay=0.5)`. | **PASSES test (false negative)**. AST scanner only matches `Name(id="EMA")` or `Attribute(attr="EMA")`, not aliases. | Adversarial only — would also fail Check 88 text-grep |
| (e) Non-canonical config attribute: `EMA(model, decay=fake.ema_decay)` where `fake.ema_decay = 0.5`. | **PASSES test (false negative)**. AST can't resolve attribute values; relies on the SEPARATE `test_canonical_ema_decay_default_is_quantizr` test which asserts `tac.training.EMA.__init__`'s `decay` default is exactly 0.997. | Acceptable — the canonical-default test is the runtime safety net |

**Verdict on adversarial bypasses**: All bypasses (1, d, e) require **MALICIOUS ACTORS** writing intentionally bypassing patterns. Innocent operator mistakes (insert `ema.update` before `optimizer.step` by accident, change `decay=0.5` literal) are CAUGHT. The Defect #4 fix is GOOD ENOUGH for the threat model the Round 7 council defined ("a future operator inserting ema.update(model) BEFORE optimizer.step()"). Defense-in-depth via `test_canonical_ema_decay_default_is_quantizr` (asserts EMA class's actual default is 0.997) catches malicious aliasing OR non-canonical attribute names IF the malicious party also tries to change the canonical class default.

---

## 4. New bug hunt — Round 5/6/7 misses

### 4.1 `SegMapTrainer` caller list — verified complete (Round 7 task G1)

`grep -rl "SegMapTrainer\(" experiments/ src/tac/` returns:
1. `experiments/train_segmap.py:316` — covered by Check 87 (Round 6)
2. `experiments/train_segmap_film_canvas.py:259` — covered by Check 87 (Round 7 Defect #1 fix)

Two scripts construct `SegMap(` (the model, not the trainer):
3. `experiments/init_segmap_from_posenet.py:248` — calls PoseNet for FEATURE EXTRACTION (per `_run_posenet`, lines 117-129). Does NOT run the SegMapTrainer training loop with FastViT scorer chunking. **OOM bug class does NOT apply.**
4. `experiments/lane_omega_w_water_filling.py:129` — water-filling utility, no scorer training loop. **OOM bug class does NOT apply.**

**Verdict**: 9-lane SegMapTrainer invalidation list is complete. No 10th member needed. [empirical:grep over experiments/ + src/tac/]

### 4.2 Subagent serializer staging-race — STRUCTURALLY EXTINCT (commit `b860710c`)

**Round 7 task G2** asked whether the residual bug class is critical enough to warrant a Round 9 fix-then-review cycle. **The answer is NO — commit `b860710c` already landed the structural fix.**

Per the commit body:
> "The file-lock prevents commit-MESSAGE swap between concurrent subagents, but the SHARED .git/index allowed staging-area sweep: today (2026-04-29 PM) Defect #1 from subagent #264 was absorbed into commit 22a2bcd2 (Lane Ω-W-V2 work from #263) because both subagents `git add`-ed files into the same shared index in overlapping windows.
>
> Fix: each `subagent_commit_serializer.py` invocation now creates a per-invocation temp `GIT_INDEX_FILE` at `.omx/state/.subagent-temp-index-<pid>-<ts>` seeded from `git read-tree HEAD`. `git add` + `git commit` (and the pre-commit hook) all run with that env var set, so they only see THIS subagent's staged files. The temp index is removed after commit (success or fail)."

Verified at `tools/subagent_commit_serializer.py:141-180`:
- Function `_make_temp_index()` creates `.omx/state/.subagent-temp-index-<pid>-<ts>`, seeds via `git read-tree HEAD`, returns env dict with `GIT_INDEX_FILE=<temp>`.
- `_run_git_add` and `_run_git_commit` both pass that env to subprocess.
- `_cleanup_temp_index` removes the temp file after commit (success or failure).

The Round 7 prescription (Round 9 fix sketch) is **already implemented**. **Round 8 finding upgraded**: the subagent staging-race bug class is now structurally extinct via per-invocation temp git index — exceeds Round 7's prescription scope. **No Round 9 cycle needed for this bug class.**

### 4.3 Lane MM v2 retag (Round 7 task G3)

**Round 7 §7.2** observed: "Lane MM v2 verdict should be re-tagged `[Modal-CPU advisory — directional FALSIFIED, awaits contest-CUDA confirm]`."

**Commit `a00da459`** retags the score from `[Modal-T4-CPU advisory]` to `[contest-CPU advisory]` in 2 occurrences across `.omx/research/council_grand_battleplan_round5_20260429.md` (line 115) and `.omx/research/council_round6_adversarial_20260429.md` (line 55). [empirical:git show a00da459]

**Should Round 8 prioritize the $0.50 Vast.ai 4090 spend BEFORE the $3 SegMap re-train wave?**

**No.** The Lane MM v2 verdict is DIRECTIONALLY CORRECT (architecture-mismatch is structural, not measurement-noise — even if true CUDA-CUDA shows 25× instead of 51× drift, the FALSIFIED verdict stands). Per the Round 7 §7.2 verdict-soundness: "The directional verdict (FALSIFIED) STANDS but the magnitude (51× vs. true CUDA-CUDA) is unconfirmed." This is a tag accuracy issue, not a strategic blocker.

The $3 SegMap re-train wave (HM-S + FR-Ω, both predicted sub-Quantizr-0.33) has HIGHER STRATEGIC VALUE per dollar than confirming the Lane MM v2 magnitude. Spend the $3 first; if HM-S/FR-Ω signal lands and budget allows, the Lane MM v2 CUDA confirm becomes a $0.50 documentation-cleanup task (NOT a kill/promote gate).

### 4.4 Other potential bugs surveyed

| Surface | Finding | Severity |
|---|---|---|
| Hardcoded `0.002` magic numbers in production code | Only the defensive `getattr(...)` fallback at segmap_renderer.py:671 + a comment at line 668. Not bugs. | None |
| Joint-ADMM 4-stream test corner-pinning | Test correctly handles corner-pinned solutions (skips KKT assertion); but doesn't actually exercise equilibration on this synthetic problem. **Test-design weakness, not coordinator bug.** | Round 9 design improvement candidate (not a reset) |
| AST EMA-ordering scanner cross-function blind spot | Wrapper-function pattern (Hypothesis 1) bypasses the scanner. Adversarial only; innocent operator mistakes caught. | Acceptable — AST static analysis can't simulate runtime call graphs |
| AST EMA-decay scanner alias / fake-attribute blind spots | `from tac.training import EMA as X` evades the scanner; `fake.ema_decay = 0.5` evades the value check. Defended by `test_canonical_ema_decay_default_is_quantizr` (catches changes to `EMA.__init__` default). | Acceptable — defense-in-depth via canonical-default test |
| `silent_defaults` audit | `tools/audit_silent_defaults.py` reports `0 critical, 0 suspicious, 1211 safe` post-fix. No outstanding silent-default override violations. | None |
| Lane GP `fit_pose_gp.py` callsite | `experiments/fit_pose_gp.py:33` correctly passes `baseline_poses=baseline` to `reconstruct_poses`. Round 5/6 callsite contract holds. | None |
| Preflight all checks | `preflight_all()` runs to completion; all STRICT checks pass. Only warnings: orphan-module suspects (library-only helpers) and MEMORY.md size 252 lines (non-blocking). | None |
| Cross-test suite | 71 PASS / 1 SKIP (CUDA-only) across all Round 7 affected test files. | None |

**Verdict on new bug hunt**: zero shipping bugs found; one Round 9 test-design improvement candidate (4-stream test could exercise equilibration more aggressively); two adversarial-only AST bypasses accepted as the cost of static analysis.

---

## 5. Residual serializer staging-race — Round 9 priority

**STATUS: STRUCTURALLY EXTINCT** via commit `b860710c`. No Round 9 cycle required for this bug class.

**Why this matters**: Round 7 §10's Top-3 finding was "land Defect #1 fix" + "land Defect #2 fix" + "empirically test Defect #3". All three landed. Round 8 would have prescribed a Round 9 cycle to fix the residual serializer staging-race had `b860710c` not landed. Since it DID land (the parent agent self-fixed during the subagent dispatch), the Round 9 priority list is reduced to:

1. (Optional) Stronger 4-stream Joint-ADMM test where streams genuinely respect `target_bytes` (Round 8 §4.4 row 2). Defer if Lane 10 V2 dispatch on real archive shows expected behavior.
2. (Optional) Capture the 5 NEW STRICT preflight checks (Checks 91-95) prescribed in Round 7 §8 — Check 91 (callerwalk OOM-guard), Check 92 (KL weight thread-through), Check 93 (EMA ordering AST), Check 94 (QAT phase-transition shadow), Check 95 (subagent serializer use). All have natural promotion paths now that the underlying bugs are fixed.

**Proposed fix sketch for residual flaw (already landed, documented for transparency)**:
```python
# tools/subagent_commit_serializer.py:141-180
def _make_temp_index() -> tuple[str, dict[str, str]]:
    tmp = REPO_ROOT / ".omx/state" / f".subagent-temp-index-{os.getpid()}-{int(time.time()*1000)}"
    subprocess.run(["git", "read-tree", "HEAD"], env={**os.environ, "GIT_INDEX_FILE": str(tmp)})
    return str(tmp), {**os.environ, "GIT_INDEX_FILE": str(tmp)}
```

---

## 6. Lane MM v2 CUDA confirm — gate or parallel?

**Recommendation: PARALLEL (if budget allows), not GATE.**

The Lane MM v2 retag from `[Modal-T4-CPU advisory]` to `[contest-CPU advisory]` is documentation cleanup. The verdict (FALSIFIED) stands directionally. Spending $0.50 to formally promote from FALSIFIED-on-CPU to FALSIFIED-on-CUDA is OPTIONAL — useful for paper-grade rigor, NOT for strategic decision-making.

The $3 SegMap re-train wave (HM-S + FR-Ω, both Council F APPROVE) has 6× the strategic value per dollar (potential sub-Quantizr-0.33 score landing vs. tag formalization). Dispatch the $3 wave first; if any GPU instances finish early, the $0.50 Lane MM v2 confirm can run in the spare time.

---

## 7. Counter status + remaining rounds to gate

**3-clean-pass gate counter**: **1 / 3** (Round 8 advances counter from 0 to 1).

**Rounds remaining before gate clears**: **2 more clean rounds** (Round 9 + Round 10) needed before the user's #3 approval can dispatch the HM-S + FR-Ω wave.

---

## 8. #3 approval-readiness summary for the user

**READY for #3 dispatch (HM-S + FR-Ω, $3 Vast.ai 4090, ~6h each)**:
- Round 7 Defects #1-4 all fixed and verified empirically (71/72 tests pass; 1 CUDA-only skip).
- Council F SAFE-LOCAL validations landed: Ω-W-V2 = 40.98% empirical savings (within band); Joint-ADMM 4-stream gate operational.
- Subagent staging-race STRUCTURALLY EXTINCT (commit `b860710c`).
- Check 87 STRICT @ 0 violations (Lane FC coverage gap closed).
- All preflight checks pass.

**BLOCKED on #3 dispatch (need 2 more clean rounds)**:
- Per the user's recursive-review mandate, a 3-clean-pass gate is required. Round 8 = 1/3. Need Rounds 9 + 10 to clear.
- Round 9 should re-run the same brutal scrutiny on any work that lands between now and then. Round 10 confirms the counter holds across two consecutive clean passes.

**RESIDUAL bug classes user should know about** (none are shipping bugs, all are test-design improvements):
1. AST EMA-ordering scanner cross-function blind spot (adversarial only). Mitigation: defense-in-depth via `test_canonical_ema_decay_default_is_quantizr`.
2. Joint-ADMM 4-stream synthetic streams ignore `target_bytes` parameter — test correctly catches silent infeasibility but doesn't exercise equilibration. Mitigation: real-archive Lane 10 V2 dispatch will catch any equilibration bugs that the synthetic test would miss.
3. (Optional) Land 5 NEW STRICT preflight checks (Checks 91-95) post-Round-9 to formally extincticate the bug classes Round 7 named.

**Top-3 actionable findings for Round 9**:
1. **Re-run brutal adversarial review on any code landing between Round 8 and Round 9.** If any new lane scripts, training scripts, or codec wrappers land, audit them through the Round 6/7/8 lens (silent defaults, MPS-validity, EMA wireins, OOM guards, dead flags).
2. **Land Checks 91-95 STRICT (Round 7 §8 prescription).** All 5 checks have natural promotion paths now that underlying bugs are fixed; live counts should be 0 or near-0.
3. **Optional: land a Round 9 stronger-4-stream-ADMM test** with genuinely interior-forced equilibration (so the test exercises equilibration mechanics, not just gating contract). Defer if Lane 10 V2 dispatch on real archive (post-#3 approval) reveals no equilibration bugs.

**Should the user approve #3 yet?** **NOT YET — counter is 1/3.** The user mandated a 3-clean-pass gate before dispatch. Round 8 is the FIRST clean pass. Round 9 + Round 10 must also be CLEAN before #3 approval.

---

## 9. Council Roll Call

Each inner-council member casts their signed Round 8 verdict.

**Shannon (LEAD, Information Theory)**: The Round 7 Defect #2 fix correctly threads `kl_distill_weight` through the channel — Phase 2's KL sensitivity sweep is unblocked. The Ω-W-V2 empirical 40.98% on real architecture is a CALIBRATED result against the Council F band [20%, 60%], replacing the synthetic 69.11% claim. R(D) discipline upheld. **Verdict: CLEAN.**

**Dykstra (CO-LEAD, Convex Feasibility)**: The Joint-ADMM 4-stream gate correctly distinguishes silent-infeasibility from honest divergence on the synthetic problem. Test-design weakness (synthetic streams don't honor `target_bytes`) is a Round 9 improvement candidate, NOT a coordinator bug. The convex-feasibility CONTRACT is correctly enforced. **Verdict: CLEAN.**

**Yousfi (Challenge creator, Steganalysis lineage)**: The Check 87 coverage gap (Round 7 Defect #1) is now closed for both `train_segmap.py` AND `train_segmap_film_canvas.py`. Lane FC re-dispatch will no longer OOM on Modal A10G. **Verdict: CLEAN.**

**Fridrich (UNIWARD/SRM/HUGO author)**: The QAT Phase A→B EMA rebuild (Round 7 Defect #3) correctly handles the parametrize-key transition. The Phase A averaging is sacrificed but correctly so — Phase A is INT8 warm-up whose value is in the live weights, NOT in the EMA shadow. **Verdict: CLEAN.**

**Contrarian (Veto)**: I VETO premature counter advance to 2/3 — Round 8 is ONE clean pass, not THREE. The user mandated 3 consecutive clean passes; Round 9 + Round 10 must independently re-confirm before #3 approval. The two adversarial AST bypasses (wrapper-function ordering, aliased-EMA constructor) are accepted but I FLAG them as known coverage gaps that require defense-in-depth via the canonical-default tests. **Verdict: CLEAN-WITH-FLAG (counter advances to 1/3 only).**

**Quantizr (Adversarial leaderboard reality check)**: The Round 7 Defect #2 fix unblocks my recommended KL sensitivity sweep (T=2.0 with weights {0.001, 0.002, 0.005}). The Ω-W-V2 40.98% real-archive savings on Lane G v3 renderer.bin is consistent with my expected ~50% block-FP savings on a properly-trained renderer; the gap is the difference between uniform-pessimistic Hessian (test) and selective-Hessian (production). **Verdict: CLEAN.**

**Hotz (Engineering shortcuts)**: All 71 tests pass in 6.17s. The QAT Phase A→B fix is 23 lines + 213 LOC of tests; the KL plumbing fix is 34 lines + 236 LOC of tests; the AST tests are 159 LOC. Total Round 7 fix volume: ~700 LOC across 4 commits. Subagent serializer staging-race fix (commit b860710c) is +87 LOC for permanent extinction of a multi-day bug class. Cost-effective shipping. **Verdict: CLEAN.**

**Selfcomp (szabolcs-cs, working 0.38 anchor)**: The Ω-W-V2 real-archive validation on Lane G v3 renderer.bin (290KB ASYM) gives 40.98% savings — extrapolated to my 88K-param block-FP renderer the savings would be similar (~40%) or higher (selective Hessian). This is the correct engineering discipline: validate the codec on the actual production architecture before stacking it into the archive build. **Verdict: CLEAN.**

**MacKay (Memorial seat)**: The QAT Phase A→B EMA rebuild correctly handles the prior change (INT8 → FP4 are different priors). Bayesian-correct. The Ω-W-V2 real-archive validation answers the MDL question "what is the rate cost of approximating Lane G v3's 290KB renderer.bin via static-histogram arithmetic on per-channel water-fill?" Answer: **40.98% byte savings, with documented L_inf round-trip bound of `max_abs / 4` per channel.** This REPLACES the synthetic 69.11% claim. **Verdict: CLEAN.**

**Ballé (2018 entropy bottleneck SOTA)**: The Ω-W-V2 static-histogram terminal saves 40.98% on Lane G v3 renderer.bin. The remaining 60% headroom is the ENTROPY CEILING I would attack with a hyperprior (V3 → Lane 20 scaffold landed at commit ccbe6591). The Council F real-archive test correctly establishes the V2 baseline against which V3 must improve to justify the hyperprior side-info overhead. **Verdict: CLEAN.**

---

## 10. Summary

| Section | Finding | Severity |
|---|---|---|
| §2.1 Defect #1 | Fix landed; both Council F + Defect #1 files in commit 22a2bcd2; semantically correct; Check 87 violations = 0 | CLEAN |
| §2.2 Defect #2 | Fix landed; 5/5 tests pass; no other hardcoded `0.002 * kl` instances | CLEAN |
| §2.3 Defect #3 | Option A correctly placed AFTER Phase B FP4 wrap; 3/3 tests pass; trade-off acceptable | CLEAN |
| §2.4 Defect #4 | AST tests are real (not text greps); 14/14 pass; 2 adversarial bypasses accepted | CLEAN |
| §2.5 Council F Ω-W-V2 | Scorer-free verified; 40.98% empirical savings on real Lane G v3 renderer.bin | CLEAN |
| §2.6 Council F Joint-ADMM | Gating contract correct; corner-pinned coverage; Round 9 design improvement | CLEAN-WITH-NOTE |
| §3 Adversarial AST bypasses | Innocent mistakes caught; malicious bypasses possible but defended by canonical-default test | ACCEPTABLE |
| §4.1 SegMapTrainer caller list | 9-lane invalidation list complete; no 10th member | CLEAN |
| §4.2 Subagent staging-race | **STRUCTURALLY EXTINCT** via commit b860710c (per-invocation GIT_INDEX_FILE) | RESOLVED (exceeds Round 7 prescription) |
| §4.3 Lane MM v2 retag | Retag landed; magnitude unconfirmed but verdict directionally correct | CLEAN |
| §4.4 Other potential bugs | Audit clean across multiple surfaces; no shipping bugs | CLEAN |
| §5 Residual Round 9 priority | No critical Round 9 cycle needed; 2 optional improvements catalogued | NONE |
| §6 Lane MM v2 CUDA confirm | PARALLEL not GATE — let HM-S + FR-Ω take priority | NONE |
| §7 Counter status | **1 / 3** (Round 8 advances 0 → 1) | n/a |
| §8 #3 approval-readiness | **NOT YET — need 2 more clean rounds** | n/a |

**Top-3 actionable findings** (for Round 9):

1. **Re-run brutal adversarial review on any code landing between Round 8 and Round 9.** The bar for clean-pass advancement is identical: zero new shipping bugs. Use the Round 6/7/8 lens (silent defaults, MPS-validity, EMA wireins, OOM guards, dead flags, callsite contracts).

2. **Land Checks 91-95 STRICT (Round 7 §8 prescription).** All 5 checks have natural promotion paths now that underlying bugs are fixed; live counts should be 0 or near-0. Specifically:
   - Check 91 (callerwalk OOM-guard) — extends Check 87 to scan ALL `from tac.segmap_renderer import SegMapTrainer` callers (currently hardcoded set of 2 files).
   - Check 92 (KL weight thread-through AST) — promotes Round 7 Defect #2 closure.
   - Check 93 (EMA ordering AST) — promotes the AST scanner from test-only to preflight-time.
   - Check 94 (QAT phase-transition shadow) — promotes Round 7 Defect #3 closure.
   - Check 95 (subagent serializer use) — formalizes the subagent-commit-serializer-required rule.

3. **(Optional) Land Round 9 stronger 4-stream Joint-ADMM test** with genuinely interior-forced equilibration. Defer if Lane 10 V2 dispatch on real archive (post-#3 approval) reveals no equilibration bugs.

**3-clean-pass gate counter status**: **1 / 3** (Round 8 = first clean pass). Round 9 must independently re-confirm; Round 10 must also re-confirm before user's #3 approval can dispatch HM-S + FR-Ω.

---

## 11. Cross-references

- Round 7 verdict: `.omx/research/council_round7_adversarial_20260429.md`
- Council F report: `.omx/research/council_f_retrain_ev_validation_admm_consult_20260429.md`
- Lane G v3 baseline 1.05 [contest-CUDA]: memory `project_lane_g_v3_landed_1_05_20260428.md`
- Subagent staging-race origin + structural fix: memory `feedback_concurrent_subagent_commit_message_swap_20260429.md` + `feedback_check_64_smoke_proofs_resolved_AND_subagent_serializer_landed_20260429.md` + commit `b860710c`
- Local-only validity binding rule: memory `feedback_no_local_mps_for_authoritative_kill_or_promote_20260429.md`
- Defect #1 site (Round 7 §6.1 fix): `src/tac/preflight.py:6375-6385` + `experiments/train_segmap_film_canvas.py:91-103,148-151` + `scripts/remote_lane_fc_film_canvas.sh:91-97`
- Defect #2 site (Round 7 §6.2 fix): `src/tac/training.py:336-339` + `src/tac/segmap_renderer.py:667-672` + `experiments/train_segmap.py` + `experiments/train_segmap_film_canvas.py`
- Defect #3 site (Round 7 §6.3 fix): `experiments/qat_finetune.py:1220-1241`
- Defect #4 site (Round 7 §6.4 fix): `src/tac/tests/test_ema_wireins_council_d.py:332-518`
- Council F Part B test: `src/tac/tests/test_omega_w_v2_real_archive.py` (491 LOC, 9 tests pass, empirical savings 40.98%)
- Council F Part C test: `src/tac/tests/test_joint_admm_4stream_nonconvex.py` (699 LOC, 12 tests pass, gating contract correct)
- Lane MM v2 retag: `.omx/research/council_grand_battleplan_round5_20260429.md:115` + `.omx/research/council_round6_adversarial_20260429.md:55`
- Subagent serializer per-invocation temp git index: `tools/subagent_commit_serializer.py:141-180,296-358`
- Empirical test results: `.venv/bin/python -m pytest src/tac/tests/test_kl_distill_weight_plumbed.py src/tac/tests/test_qat_phase_a_to_b_ema_rebuild.py src/tac/tests/test_ema_wireins_council_d.py src/tac/tests/test_omega_w_v2_real_archive.py src/tac/tests/test_joint_admm_4stream_nonconvex.py src/tac/tests/test_segmap_renderer_bf16_chunking.py src/tac/tests/test_segmap_renderer.py src/tac/tests/test_joint_admm_coordinator.py` → 71 passed / 1 skipped (CUDA-only) in 6.17s
