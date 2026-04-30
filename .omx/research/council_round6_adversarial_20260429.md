# Council Round 6 — Adversarial Review (post-Check-86 landing)

**Date**: 2026-04-29 PM
**Convened by**: parent agent under user mandate "keep the recursive adversarial reviews going; remember that local only and MPS is notoriously broken and inaccurate; make all permanent fixes and fix all bugs regardless of severity"
**Inner council (10 voices)**: Shannon (LEAD), Dykstra (CO-LEAD), Yousfi, Fridrich, Contrarian, Quantizr, Hotz, Selfcomp, MacKay (memorial), Ballé
**Round counter**: incoming +1 (Round 5 = CLEAN-WITH-CONCERNS); this round resolves to 0/3 (BUG-FOUND).
**Subject**: Council A's just-landed Check 86 fix (commit `d9d7ca43`) + Council E battleplan recommendations + sister-council reports B/C/D, with brutally rigorous re-audit per the user's "never accept local/MPS as truth" rule.

---

## 1. Round 6 Verdict

### **BUG-FOUND → counter RESETS to 0 / 3 toward gate.**

Three concrete defects were identified in this round (one in the just-landed Check-86 fix's commit *body*, one in the original Check-86 fix's *cross-impact attribution*, one in the *whitelist's API contract*). None of them re-introduce the gradient-killing bug at runtime — the corrected `_eval_roundtrip_chain` is empirically gradient-flowing — but the **commit message is materially incorrect**, the **whitelist is documentation-only enforcement on a function that lacks a `@torch.no_grad()` guard**, and the **manual-STE regex has a verified multi-line false-positive risk**. Per the persistent-codex review protocol, these are LANDED bugs in the committed artifact (not just CONCERNs about untested empirical claims), so the counter resets.

The user's mandate "fix all bugs regardless of severity" forces the reset; under the looser "no critical bugs found" reading the verdict would be CLEAN-WITH-CONCERNS.

---

## 2. Council A's fix audit — structurally complete?

**Verdict**: **PARTIALLY COMPLETE — fix is correct at the runtime layer but commit-body cross-impact attribution is wrong.**

### 2.1 Runtime correctness — VERIFIED CLEAN

`/Users/adpena/Projects/pact/src/tac/segmap_renderer.py:280-293` (the patched `_eval_roundtrip_chain`):
- Line 281-289: replaces `up.clamp(0, 255).round()` with `Uint8STE.apply(up)` ✓
- Line 290: noise injection at `noise_std > 0` is preserved (Hotz STE-leak fix intact) ✓
- Line 292-293: `back = F.interpolate(...)` followed by `return back.clamp(0, 255).reshape(...)` — `back.clamp` has identity gradient inside [0, 255], no `.round()` here so no gradient kill ✓

Independent empirical verification (this session):
```
$ .venv/bin/python -c "import torch; x = torch.tensor([1.3], requires_grad=True); x.round().backward(); print(x.grad)"
tensor([0.])
```
Confirms Council A's empirical claim. Bare `.round()` IS zero-gradient.

`/Users/adpena/Projects/pact/src/tac/optimize_grayscale_canvas.py:282-315` (Lane AL `_eval_roundtrip_with_noise`):
- Line 304-310: same Uint8STE.apply replacement, applied AFTER noise injection (line 299-303). This is the canonical AV1-quantization-noise-then-uint8-clip pattern ✓
- Note: Council E found this Lane AL site in addition to the Lane SegMap site. Both fixes landed together.

### 2.2 Other gradient-killing patterns downstream — NONE FOUND

Audited `src/tac/segmap_renderer.py` for:
- `.detach()` outside the no-grad export path: only `seg_argmax_gt = gt_seg_out.argmax(dim=1)` inside `with torch.no_grad():` block (correct — GT targets must be detached). [empirical:src/tac/segmap_renderer.py:501]
- `.argmax()` on the gradient-flowing branch: NONE on `seg_logits_pred`. CE loss receives raw logits. ✓
- `back.clamp(0, 255)` at line 293: clamp has identity grad in interior, hard-zero grad at boundary — but values are forward-clipped to [0, 255] AFTER bicubic upsample-then-downsample so saturation should be uncommon. Not a bug. ✓

### 2.3 Commit-body cross-impact attribution — INCORRECT (Defect #1)

The commit message (`d9d7ca43`) states:
> "CROSS-IMPACT (HUGE): all prior runs through SegMapTrainer.train_epoch INVALIDATED. Lane SC++/SA-v2/SO/MM v2 all suspect."

**Lane MM v2 was NOT invalidated by Check 86.** Verified by reading `experiments/build_lane_mm_archive.py`: the script takes the Lane A archive (renderer.bin + masks.mkv + optimized_poses.pt) and re-encodes ONLY masks.mkv as grayscale.mkv. The renderer is NOT trained — it is the pre-trained Lane A renderer (which uses `train_distill.py` + `Uint8STE.apply` correctly per Lane G v3 commit lineage). Therefore Lane MM v2's score 2.63 `[Modal-T4-CPU advisory]` IS a real architectural-mismatch FALSIFICATION (3-channel-trained renderer + grayscale-LUT encoding), unaffected by the SegMapTrainer bug.

Affected lanes (verified via `grep -rln SegMapTrainer scripts/`):
- ✓ `scripts/remote_lane_sc_plus_plus_kl_distill.sh` (Lane SC++) — INVALIDATED
- ✓ `scripts/remote_lane_sa_segmap_clone.sh` (Lane SA) — INVALIDATED
- ✓ `scripts/remote_lane_so_hessian_block_fp.sh` (Lane SO) — INVALIDATED
- ✓ `scripts/remote_lane_wc_s_curator_weighted.sh` (Lane WC-S) — INVALIDATED (not in Council A's list but uses SegMapTrainer)
- ✓ `scripts/remote_lane_pa_pose_as_affine.sh` (Lane PA) — INVALIDATED (not in list, uses SegMapTrainer)
- ✓ `scripts/remote_lane_hm_s_segmap_homography.sh` (Lane HM-S) — INVALIDATED
- ✓ `scripts/remote_lane_fr_omega_fridrich_block_fp.sh` (Lane FR-Ω) — INVALIDATED
- ✓ `scripts/remote_lane_fc_film_canvas.sh` (Lane FC) — INVALIDATED
- ✓ `scripts/remote_lane_darts_s_segmap_arch_sweep.sh` (Lane DARTS-S) — INVALIDATED
- ✗ `scripts/remote_lane_mm_grayscale_lut.sh` (Lane MM v2) — **NOT invalidated; build-only path**

**Council A undercounted by ~5 affected lanes AND included one (MM v2) that is unaffected.** The commit body needs a follow-up correction note recording the actual impact set. [empirical:scripts/remote_lane_*.sh + experiments/build_lane_mm_archive.py]

---

## 3. Lane MM v2 FALSIFICATION revisit

**Verdict**: **VERDICT STANDS — Lane MM v2's 2.63 IS a genuine FALSIFICATION, unaffected by Check 86.**

Verification path (per Round 6 mandate task B):

1. Read `experiments/build_lane_mm_archive.py:1-80` — it's an **archive transformation script**, not a trainer. It decodes Lane A's `masks.mkv` to class IDs, re-encodes them as grayscale.mkv via the Selfcomp class-target LUT `[0, 255, 64, 192, 128]`, and re-bundles the ZIP with Lane A's untouched renderer.bin + optimized_poses.pt.
2. `grep SegMapTrainer experiments/build_lane_mm_archive.py` → 0 matches.
3. `grep train_segmap experiments/build_lane_mm_archive.py` → 0 matches.
4. The renderer used = Lane A's renderer.bin = produced by `train_distill.py` (NOT SegMapTrainer). `train_distill.py` does NOT route through `_eval_roundtrip_chain` (the bare-round buggy site); it has its own roundtrip implementation that already used `Uint8STE`. Verified at Lane G v3 commit lineage.
5. The 2.63 score is therefore measuring the architectural mismatch between a 3-channel-RGB-trained Lane A renderer and a grayscale-LUT mask encoding — exactly what `project_lane_mm_v2_landed_2_63_falsified_20260429.md` already records.

**Implication**: Lane MM v2's KILL verdict in Council E's battleplan (Section 3.2) is correct and based on a valid measurement. Lane AL / Lane SC++ remain the alive descendants (they fine-tune the renderer for the grayscale path).

**Implication for Council A**: the commit body needs amendment to remove the "MM v2 invalidated" claim. (Not amended in this round per the "DO NOT modify code" mandate; logged as Defect #1 above.)

---

## 4. Local-only validation scrutiny (Council E's 3 dispatch recs)

**Verdict**: **2 of 3 Council E local-only dispatches are SCORE-FREE (VALID locally); 1 of 3 has been promoted correctly.**

| # | Dispatch | Touches neural net forward? | Local validity | Notes |
|---|---|---|---|---|
| 1 | Lane UNIWARD v8 CUDA-confirm ($0.50 Vast.ai) | YES (PoseNet + SegNet on archive bytes) | **CUDA REQUIRED** ✓ Council E correctly dispatched to Vast.ai | Per Check 83 STRICT, the score must come from contest-CUDA. Council E correctly does NOT propose local validation here. |
| 2 | Lane Ω-W-V2 real-archive empirical (local-only, $0) | NO — pure tensor encode/decode | **VALID LOCALLY** ✓ | Verified by `grep -E "compute_proxy_score|posenet|segnet|F.interpolate|requires_grad" src/tac/water_filling_codec_v2.py` → 0 matches. Pure water-fill bit allocation + arithmetic coding on a real `state_dict.values()` iteration. The MEASUREMENT is a byte count + round-trip max-abs-error check; both deterministic across CPU / MPS / CUDA. |
| 3 | Lane Joint-ADMM 4-stream non-convex (local-only, $0) | NO — pure proximal optimization on synthetic R(D) functions | **VALID LOCALLY** ✓ | Verified by `grep -E "compute_proxy_score|posenet|segnet|F.interpolate" src/tac/joint_admm_coordinator.py` → 0 matches. Coordinator is strict-scorer-rule compliant by construction (per Council E Section 2.3 "no SegNet/PoseNet inside coordinator"). KKT residual is a math check on the dual variables. |

**Score-claim smuggling check**: I scanned each of Council E's three dispatch lines (Sections 3.1 #1-3) for any score being PROMOTED (e.g., "this proves Lane Ω-W-V2 saves 60% on real archive"). Council E does NOT promote a score from #2 or #3 — both are gated as "validate before promotion" measurements. Council E DOES promote Lane UNIWARD v8 CUDA-confirm to "promote `[Modal-T4-CPU advisory]` to `[contest-CUDA]` if within 0.05" — this is correctly CUDA-dispatched.

**Round 6 finding on local-only mandate**: the user's caveat "local only and MPS is notoriously broken and inaccurate" applies to **any measurement that depends on a neural-net forward pass**. Pure codec round-trip + pure ADMM math on cached R(D) functions are bit-deterministic on CPU/MPS/CUDA and may be validated locally. Council E's 3 dispatches respect this dividing line correctly. **No score-claim trap detected.**

---

## 5. Bug Round 5 missed — concrete file:line

**Verdict**: **Defect #2 found — `tac/scorer.py:compute_proxy_score` is whitelisted for Check 86 with the comment "read-only" but the function lacks an `@torch.no_grad()` decorator that would enforce that contract.**

File: `/Users/adpena/Projects/pact/src/tac/scorer.py:274-388`

Concrete observations:
- Function signature: `def compute_proxy_score(frames: torch.Tensor, gt_frames, posenet, segnet, device, rate=0.0, batch_size=16, eval_roundtrip=True) -> dict`
- Line 274: NO `@torch.no_grad()` decorator on the function.
- Lines 342, 350: bare `cand_chw.round().clamp(0, 255)` and `flat.round().clamp(0, 255)` — the same gradient-killing pattern Council A just fixed in `_eval_roundtrip_chain`.
- Line 357: `with torch.no_grad():` block wraps ONLY the scorer forward (lines 358-372). The round/clamp at lines 342/350 is OUTSIDE that block.

Current safety net: every CALLER of `compute_proxy_score` (verified via `grep -rn "compute_proxy_score" src/ experiments/`) uses it for read-only metric computation. None passes a `frames` tensor with `requires_grad=True`. So at the moment, the bare `.round()` is harmless because no caller relies on gradient flow.

**The defect is in the API contract**: the whitelist comment in `preflight.py:6188` says "compute_proxy_score read-only", but the function does not enforce read-only. A future caller (e.g. a TTO loop that wants a proxy score gradient for the renderer) could pass a `requires_grad=True` tensor and silently get zero gradients — exactly the bug class Check 86 is designed to prevent — and Check 86's whitelist would mask the regression.

**Fix prescription** (not landed this round per "DO NOT modify code" mandate; logged for Round 7):
1. Add `@torch.no_grad()` decorator to `compute_proxy_score` (1 line).
2. OR: replace bare `.round()` at lines 342, 350 with `Uint8STE.apply(...)` (so the function is gradient-safe even if called with grad tensors).
3. OR: remove `compute_proxy_score` from `_BARE_ROUND_READONLY_FILES` whitelist and require either (1) or (2).

Recommended: **option 1** (smallest change, makes the contract explicit, prevents future foot-gun).

[empirical:src/tac/scorer.py:274-388 + grep of all callers]

### 5.1 Other audit results (clean)

- `experiments/pair_difficulty_map.py:186, 196` — bare `.round().clamp(0, 255)`. Verified: file is a pre-computation TOOL, not in any training path. Function `compute_pair_difficulty` does not flow gradient through these tensors. Whitelist justified.
- `experiments/profile_fp4_layer_sensitivity.py:194` — bare `.round().clamp(0, 255).to(uint8).float()`. Verified: profiling tool, no gradient flow. Whitelist justified.
- `src/tac/forensics.py:465` — bare `.round().clamp(0, 255)`. Verified: explicit "no STE — analysis" comment in surrounding code per the whitelist comment. Whitelist justified.
- `src/tac/training.py:1001, 1492, 2125` — bare `.round().clamp(0, 255).to(uint8).float()`. Verified all three sites are inside `with torch.no_grad():` blocks (line 997 wraps line 1001; corresponding wrappers for 1492, 2125 verified). Correct pattern, no bug.
- `src/tac/constrained_gen.py` — many `.round().clamp(0.0, 255.0)` sites. Sampled 6 of them: all use either `frames.detach().round().clamp(...)` (manual detach pattern) or are inside `torch.no_grad()` blocks or are post-optimization export. No bug found.
- `src/tac/renderer.py:1884` — `up_quantized = up + (up.round().clamp(0, 255) - up).detach()`. This is the canonical manual STE pattern (forward = `up.round().clamp`, backward = identity via `.detach()` of the `(round - up)` correction). Correct. Check 86's `_MANUAL_STE_RE` matches this pattern on the same line. ✓
- `src/tac/preprocessor.py:143` — `rounded = rounded + (rounded.round() - rounded).detach()`. Same canonical pattern. ✓
- `src/tac/tto.py:246` — same canonical pattern. ✓
- `src/tac/water_filling_codec_v2.py:243`, `block_fp_codec.py:508` — `(wc / scale).round().clamp(-Qc, Qc)` inside codec encode functions. These ARE intended to be lossy (codec rounding), but they're inside encode-only paths (no gradient required at encode time). Acceptable.

---

## 6. Check 86 whitelist audit

**Verdict**: **Whitelist is operationally correct TODAY but documentation-only enforcement; one site (compute_proxy_score) lacks a runtime guard.**

| File in `_BARE_ROUND_READONLY_FILES` | `@torch.no_grad()` decorator? | Inside `with torch.no_grad():` block? | All inputs `.detach()`-ed? | Verdict |
|---|---|---|---|---|
| `src/tac/forensics.py:465` (`up_quantised = up.round().clamp(0, 255)`) | function `_run_eval_chain` lacks decorator; verified caller uses inputs that are not differentiable (post-decode uint8 frames cast to float) | NOT inside no-grad | YES — function receives uint8 frames | **Justified** by data shape (uint8 origin, no grad chain) |
| `src/tac/scorer.py:342, 350` (`cand_chw.round().clamp(0, 255)`) | NO decorator (line 274) | NOT inside no-grad block (the no-grad starts at line 357 AFTER the rounds) | NOT enforced — caller could pass `requires_grad=True` | **DEFECT — whitelist is a foot-gun (see §5)** |
| `experiments/pair_difficulty_map.py:186, 196` | function lacks decorator; verified data shape is uint8 GT frames + uint8-cast renderer outputs in a measurement loop | NOT inside no-grad explicitly | inputs come from disk/uint8 cast, no grad chain | **Justified by data shape** |
| `experiments/profile_fp4_layer_sensitivity.py:194` | function `profile_layer_sensitivity` lacks decorator; verified the `cam` tensor is post-render uint8 | NOT inside no-grad explicitly | inputs are post-render uint8 | **Justified by data shape** |

**Manual STE regex false-positive risk** (Check 86's `_MANUAL_STE_RE`):
- Pattern: `r"\.detach\(\s*\).*\.round\(\)|\.round\(\).*\.detach\(\s*\)"` — REQUIRES `.detach()` and `.round()` on the SAME line.
- Probed (this session) with a multi-line manual STE pattern (`fwd = (x.round().clamp(0, 255)\n        - x).detach() + x`): the line containing `.round()` has NO `.detach()` and would be flagged. The line containing `.detach()` has NO `.round()` and is fine.
- **Result**: a FUTURE manual-STE site formatted across two lines (e.g., for line-length wrapping) would be a FALSE POSITIVE (Check 86 would block a correct STE pattern).
- **Today**: searched `src/tac/` for multi-line manual STE patterns inside roundtrip-named functions — NONE FOUND. The hypothetical FP is not realized in code today.
- **Future hardening**: Check 86 should AST-walk the binary expression tree (find `.detach()` calls inside an enclosing expression that also calls `.round()`) instead of regex-on-line. Logged as Round 7 follow-up.

**Multi-line `.detach()` evasion risk for the BUG direction** (i.e., a real bug masked by detach on adjacent line): less concerning. The check would still flag the `.round()` line as a violation (no detach on same line); a manual review would confirm.

---

## 7. Round 7 STRICT check priorities (Checks 87 / 88 / 89 specs validated)

### 7.1 Check 87 — `check_phase15_lanes_have_real_archive_validation` (Council E proposal)

**Spec validation**: Council E proposes scanning `src/tac/*_codec*.py` and `src/tac/*_v2.py` for public `encode_*` / `decode_*` functions and requiring at least one companion test file matching `tests/test_*real_archive*.py` OR `tests/test_*on_lane_g_v3*.py`.

**Round 6 validation**:
- Predicted live count: 4-6 violations (Lane Ω-W-V2, PD-V2, J-NWC).
- I confirmed by glob: `tests/test_water_filling_codec_v2.py` and `tests/test_pose_delta_codec_v2.py` exist but use SYNTHETIC tensors only.
- Promotion path (warn-only → fix → STRICT) is correct.
- **Spec gap**: also need to scan `src/tac/joint_admm_*.py` (the Joint-ADMM coordinator + proximal codec wrappers) since they have similar `encode_*` semantics (project a stream onto a codec). Suggest broadening the glob to `src/tac/*proximal*.py` and `src/tac/joint_admm_*.py`.
- **Approved with one spec amendment.**

### 7.2 Check 88 — `check_training_paths_use_ema_correctly` (Council D proposal)

**Spec validation**: Council D proposes scanning `experiments/train_*.py` for trainers (`optimizer.step()` present) and requiring (a) `EMA(model, decay=...)` construction, (b) `ema.update(model)` call, (c) `ema.state_dict()` in checkpoint save.

**Round 6 validation**:
- Council D's EMA inventory found 5 correct paths + 8 missing paths (per `council_ema_audit_20260429.md` §2-§3). Spec implementation is sound — regex + AST hybrid.
- The bidirectional companion check (`check_ema_shadow_no_back_shadow`) is also valuable — catches the antipattern Council D explicitly named in §3 (calling `ema.apply(model)` inside `train_epoch` shadows live weights and kills learning).
- **Predicted live count**: 8 violations (Council D enumerated them in §3.1-§3.5).
- Promotion path (warn-only → fix 8 → STRICT) is correct.
- **Spec gap**: `experiments/train_imp_cycle.py` is an OUTER LOOP not a trainer (it calls `train_renderer.py` per cycle); the EMA missing flag from Council D applies to the cycle-level EMA over checkpoints, NOT optimizer-step EMA. The check should distinguish "outer-loop trainer" (Council D wants this too) from "inner-loop trainer" (the regex-target).
- **Approved with one spec amendment** — add a separate detector for IMP-cycle-level EMA.

### 7.3 Check 89 — `check_remote_lane_scripts_use_computed_payloads` (Council B proposal)

**Spec validation**: Council B proposes scanning `scripts/remote_lane_*.sh` for the pattern:
```
ENCODE_PAYLOAD ($PAYLOAD_FILE)
...
cp $ANCHOR_DIR/masks.mkv $ITER_DIR/    # ← THIS DISCARDS THE PAYLOAD
```
If a script computes a payload but the archive build doesn't include it, fail.

**Round 6 validation**:
- This catches the EXACT bug Council B identified in `project_lane_uniward_v8_NO_OP_finding_20260429.md`: the UNIWARD v8 lane computed an 8.6MB SLI1 payload then `cp $ANCHOR_DIR/masks.mkv $ITER_DIR/` overwrote with the anchor's unmodified masks. The "lane" was a PURE NO-OP.
- **Predicted live count**: 1 violation (`scripts/remote_lane_uniward.sh` v8). May increase if other remote-lane scripts have the same encode-then-discard pattern; recommend a sweep before flipping STRICT.
- **Spec gap**: the regex must allow LEGITIMATE cases where the payload is CONSUMED by a downstream tool (not just `cp`'d into the archive). Detection should match: (a) compute payload at line N, (b) NEVER reference the payload variable in any subsequent zip/tar/cp INTO the archive. Pure file-existence check is insufficient.
- **Approved with one spec amendment** — track payload-variable USAGE in subsequent ZIP/TAR/cp ops; only fail if computed-but-unused.

### 7.4 New check candidates from Round 6 findings

**Check 90 — `check_proxy_score_no_grad_safety`** (this round's Defect #2):
- AST-scan `src/tac/scorer.py` and any future `compute_*_score` function for the pattern: bare `.round()` outside a `with torch.no_grad():` block and without `@torch.no_grad()` decorator.
- Would have caught `compute_proxy_score` lines 342, 350 today.
- Predicted live count: 1 (compute_proxy_score) — fixable in 1 line (add decorator).
- Promotion: land warn-only at 1 → add `@torch.no_grad()` decorator → flip STRICT.

**Check 91 — `check_commit_body_attribution_matches_diff`** (Round 6 Defect #1):
- Scan the latest commit's body for lane-name claims (e.g., "Lane MM v2 invalidated") and cross-reference against the actual diff via `git diff-tree HEAD --name-only`.
- A claim like "Lane MM v2 invalidated by SegMapTrainer fix" requires the diff to actually touch a file in the Lane MM v2 train path. If the lane name doesn't appear in the diff's docstrings, raise a warning.
- This is a softer check (can have false positives on legitimate cross-impact claims), so warn-only ONLY — don't flip STRICT.
- Predicted live count: 1 (today's commit `d9d7ca43` falsely claims MM v2 invalidation).

---

## 8. Council Roll Call

Each inner-council member casts their signed Round 6 verdict (1-2 sentences). Per CLAUDE.md "Council conduct" the council is non-conservative; arguments are mathematical/empirical only.

**Shannon (LEAD, Information Theory)**: The bare `.round()` zero-gradient is a Shannon-channel-broken-then-restored situation: Check 86's whitelist correctly identifies channels where the rounding is on a no-grad payload (information has already been quantized to uint8 + sent to disk). The SCORER channel at lines 342, 350 has no such guarantee — the rounding is on a tensor that COULD be a gradient channel. The whitelist is documentation-only; promote to runtime guard via `@torch.no_grad()`. **Verdict: BUG-FOUND.**

**Dykstra (CO-LEAD, Convex Feasibility)**: Council E's local-only dispatches #2 (Lane Ω-W-V2 real-archive) and #3 (Joint-ADMM 4-stream KKT) are pure convex-feasibility checks on cached R(D) functions; no neural-net forward in the dependency chain. Local validity is preserved per the narrow-exception rule in `feedback_no_local_mps_for_authoritative_kill_or_promote_20260429.md`. **Verdict on local-only validity: CLEAN**; aggregate verdict inherits from peers' bug-finds.

**Yousfi (Challenge creator, Steganalysis lineage)**: The Lane MM v2 attribution error in the commit body is a meta-bug class — claims of "X invalidated" must be substantiated by the actual diff touching X's training path. This is the same dead-flag-wiring class as the 2026-04-26 incident: a claim made without grep confirmation. Check 91 candidate (Section 7.4) addresses this. **Verdict: BUG-FOUND on commit body attribution.**

**Fridrich (UNIWARD/SRM/HUGO author)**: Lane UNIWARD v8 is a NO-OP per Council B's verified SHA-equality finding; the proper rebuild (V9 with Daubechies-8 + SLI1 inflate decoder) is the next move. Round 6 does not change that strategic verdict. The proposed Check 89 (Council B's `check_remote_lane_scripts_use_computed_payloads`) prevents recurrence. **Verdict: CLEAN on Council B's analysis; APPROVED Check 89.**

**Contrarian (Veto)**: I VETO the Round 5 commit message claim "Lane MM v2 all suspect" — Lane MM v2 IS a real falsification, and conflating it with the SegMapTrainer freeze cohort dilutes the strategic record. I VETO any future commit that attaches scorer-derived score claims to a `compute_proxy_score` call without verifying the function is no-grad-protected. I do NOT veto the runtime fix at segmap_renderer.py:281 — that is correct. **Verdict: BUG-FOUND.**

**Quantizr (Adversarial leaderboard reality check)**: The DARTS-S V1 freeze is a textbook example of "the scorer-rendering simulation must preserve gradient end-to-end". My architecture (88K FiLM-conditioned) doesn't touch this path because I trained via `train_distill.py` (which used `Uint8STE` correctly per Lane G v3 lineage). Check 86 prevents the entire bug class for ALL future SegMap variants. **Verdict on Check 86: APPROVED**; aggregate verdict: BUG-FOUND on whitelist API (Defect #2).

**Hotz (Engineering shortcuts)**: The single-line fix `Uint8STE.apply(up)` saves 5 hours of GPU per future Lane DARTS-S clone. The whitelist is fine for today but MUST become a runtime guard (`@torch.no_grad()` decorator on `compute_proxy_score`) before the next training run. **Verdict: BUG-FOUND, fix prescription is 1 line.**

**Selfcomp (szabolcs-cs, working 0.38 anchor)**: My SegMap renderer at 88K params is one of the architectures that flows through SegMapTrainer; the Check 86 fix should be re-confirmed by an empirical training-step test that asserts `model.parameters()[0].grad.abs().max() > 0` after one optimizer step. The current `test_segmap_trainer_train_epoch_loss_finite` test only asserts `pre_param != post_param` which passes vacuously via weight-decay shrinkage. **Verdict: BUG-FOUND on test coverage (vacuous test); fix prescription is to add a gradient-presence assertion to the existing test.**

**MacKay (Memorial seat, Information Theory + Bayesian Inference + Learning Algorithms)**: The bare `.round()` bug is a degenerate case of the Shannon-Bayesian principle "no posterior update can occur if the likelihood gradient is zero almost everywhere." AdamW's weight-decay shrinkage observed by Council A is the prior pulling the posterior toward zero in the absence of data signal — exactly the failure mode predicted by zero-likelihood-gradient. Check 86 restores the likelihood gradient. **Verdict on Check 86: APPROVED.** Aggregate inherits from peers.

**Ballé (2018 entropy bottleneck SOTA)**: The Council E local-only dispatch #2 (Lane Ω-W-V2 real-archive) is the right next step for my hyperprior-amortization analysis — once we have a real Lane G v3 weight distribution, we can compare static-histogram (Ω-W-V2) vs hyperprior-conditional (future Lane 20) entropy on the same payload. The synthetic 69.11% number is misleading; the real-archive number (predicted [40%, 65%]) is what matters. Local-only is valid because the measurement is byte-count, not score. **Verdict on Council E #2 dispatch: APPROVED.**

---

## 9. Summary

| Section | Finding | Severity |
|---|---|---|
| §2 Council A audit | Runtime fix CORRECT; commit body MM v2 attribution wrong (**Defect #1**) | LOW (cosmetic / strategic record) |
| §3 Lane MM v2 revisit | FALSIFICATION verdict STANDS (lane is build-only, not trained via SegMapTrainer) | INFO |
| §4 Local-only validity | Council E's 3 dispatches respect the dividing line; Lane Ω-W-V2 + Joint-ADMM are score-free; Lane UNIWARD v8 correctly CUDA-dispatched | CLEAN |
| §5 Bug Round 5 missed | `compute_proxy_score` lacks `@torch.no_grad()` decorator (**Defect #2**) | MEDIUM (foot-gun for future TTO) |
| §6 Check 86 whitelist | 3 of 4 whitelist sites justified by data shape; 1 (compute_proxy_score) needs runtime guard. `_MANUAL_STE_RE` has multi-line FP risk (no live violations) (**Defect #3 minor**) | LOW (no live false positives) |
| §7 Round 7 checks | Checks 87 / 88 / 89 specs validated with minor amendments; 2 new check candidates (Check 90 for proxy_score no-grad safety; Check 91 for commit-body attribution) | n/a |

**Top-3 actionable findings** (for Round 7):

1. **Add `@torch.no_grad()` decorator to `src/tac/scorer.py:compute_proxy_score`** (1 line). Removes the foot-gun the Check 86 whitelist masks. Permanent fix.
2. **Amend the test `test_segmap_trainer_train_epoch_loss_finite`** (per Selfcomp's verdict): add an assertion `assert any(p.grad.abs().max() > 0 for p in model.parameters() if p.requires_grad)` after the first optimizer step. The current test passes vacuously via weight-decay shrinkage and would have missed Council A's bug.
3. **Land Check 89** (`check_remote_lane_scripts_use_computed_payloads`) STRICT after fixing the 1 known violation (`scripts/remote_lane_uniward.sh` v8 NO-OP). This is the highest-EV preflight check this week — catches the entire encode-then-discard bug class permanently.

**3-clean-pass gate counter status**: **0 / 3** (RESET).

Round 7 must re-run with brutally rigorous skepticism on the Round 6 fixes once landed; if Round 7 finds zero new bugs AND the 3 Top-3 findings are landed, counter advances to 1/3.

---

## 10. Cross-references

- Round 5 verdict and Council E battleplan: `.omx/research/council_grand_battleplan_round5_20260429.md`
- Council A DARTS-S freeze: `.omx/research/council_darts_s_freeze_audit_20260429.md`
- Council B UNIWARD v8 NO-OP: `.omx/research/council_uniward_v8_fridrich_shannon_audit_20260429.md`
- Council C OOM-class deep fix: `.omx/research/council_oom_class_deep_fix_20260429.md`
- Council D EMA audit: `.omx/research/council_ema_audit_20260429.md`
- Just-landed commit: `d9d7ca43` (Check 86 STRICT + 2 .round() fixes)
- Fix sites verified: `src/tac/segmap_renderer.py:280-293`, `src/tac/optimize_grayscale_canvas.py:282-315`
- Check 86 implementation: `src/tac/preflight.py:6150-6293`
- Defect site (foot-gun): `src/tac/scorer.py:274-388` (`compute_proxy_score`)
- Memory anchors:
  - `feedback_no_local_mps_for_authoritative_kill_or_promote_20260429.md` (binding rule on local validity)
  - `feedback_check_85_metric_key_display_bug_landed_20260429.md` (Check 85 lineage)
  - `feedback_modal_spawn_result_cache_pattern_20260429.md` (Modal spawn() harvest rule)
  - `feedback_concurrent_subagent_commit_message_swap_20260429.md` (commit attribution antipattern)
  - `project_lane_uniward_v8_NO_OP_finding_20260429.md` (Council B finding)
  - `project_lane_mm_v2_landed_2_63_falsified_20260429.md` (MM v2 falsification — verified to STAND in this round)
  - `project_lane_g_v3_landed_1_05_20260428.md` (Lane G v3 baseline used as comparison)
