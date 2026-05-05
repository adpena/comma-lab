---
name: Grand Council recursive greenup — Shannon-floor execution state Round 1 — 11 findings, counter 0/3
description: 2026-05-01 ~03:50 UTC. Single-thread (per CLAUDE.md serializer mandate) recursive adversarial Grand Council greenup pass on the Shannon-floor execution state. Round 1 surfaces 11 findings: 5 CRITICAL (3 score-arithmetic errors, 1 device-tag contradiction, 1 PCC4 false-positive blocking memory commits), 4 MEDIUM (drift coefficient misapplied, asymmetric-regression hypothesis without evidence, PCC1 evasion holes, PFP16 missing from inventory), 2 LOW (PCC3 missing dedicated test, council-vote unanimous-pattern smell on Lane 17). Counter resets to 0/3. Pre-fix dispatch DEFER decision is correct OUTCOME but for wrong reason (drift was applied wrong direction). Greenup blocker: 3 fixes are user-/operator-decisions (drift recomputation, PCC4 false-positive design call, inventory PFP16 row addition) requiring approval before code lands.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Round 1 — counter 0/3 → 0/3

User mandate 2026-05-01 ~03:30 UTC: "spawn extreme rigor and adversarial grand council to perform recursive adversarial reviews and greenup passes on all extreme math and scientific and engineering rigor". Single-thread serializer mode, no parallel sub-subagents.

This memory captures Round 1 of the 3-clean-pass greenup gate against the Shannon-floor execution state checkpoint, OWv3 deferral, IMP-permanent-fix council file, IMP-killed-WITHDRAWN file, IMP local backport, shannon_floor_execution_readiness research, all-scores inventory, and the 4 PCC preflight functions.

## Findings (CRITICAL = 5, MEDIUM = 4, LOW = 2)

### CRITICAL #1 — Score derivative "421KB → 0.20" is WRONG (Shannon)
- **Source**: `project_shannon_floor_execution_state_checkpoint_20260501.md` line 24
- **Math**: `25 × 421,000 / 37,545,489 = 0.2803`. The checkpoint says `0.20`. Error: 28%.
- **Why it matters**: this is the score lever the entire mask-payload Wave 1 strategy hinges on. Underestimating by 28% biases the bytes-vs-distortion tradeoff calculus in every dispatch decision.
- **Cross-check**: `shannon_floor_execution_readiness_20260430.md` §5 line 121 says "300KB mask reduction is about 0.1998" which IS correct math, so the checkpoint is contradicting the readiness doc.
- **Council vote**: Shannon CRITICAL / Dykstra CRITICAL (Pareto-frontier slope is wrong) / MacKay CRITICAL (rate-cost arithmetic) / Yousfi CRITICAL / Hotz CRITICAL. **5/5 unanimous CRITICAL** — this is a math error, not a design choice.
- **Fix**: rewrite checkpoint line 24: `421KB mask reduction → 0.280 score`.

### CRITICAL #2 — Score derivative "PoseNet ×10 → +0.32 score" is WRONG (Fridrich)
- **Source**: `project_shannon_floor_execution_state_checkpoint_20260501.md` line 26
- **Math**: at PoseNet anchor 0.00345, ×10 = 0.0345. Score delta = √(10·0.0345) − √(10·0.00345) = 0.587 − 0.186 = `0.402`. Checkpoint says `+0.32`. Error: 26%.
- **Why it matters**: this is the asymmetry argument in every "PoseNet regression risk" gate. Underestimating risk by 26% biases the risk-of-promote calculation across every retraining lane.
- **Council vote**: Fridrich CRITICAL (he literally designed the scorer; this is a calculus he can do in his head) / Shannon CRITICAL / Yousfi CRITICAL / Dykstra CRITICAL / MacKay CRITICAL. **5/5 unanimous CRITICAL**.
- **Fix**: rewrite checkpoint line 26: `PoseNet ×10 increase → +0.40 score`.

### CRITICAL #3 — CPU↔CUDA drift coefficient applied as ×1.105, empirical is ×1.131 (Yousfi/Hotz)
- **Source**: `project_owv3_byte_feasible_candidate_dispatch_deferred_20260501.md` §"CPU↔CUDA drift projection"
- **Math**: deferral file says "PoseNet drift: ~9.5% smaller than CUDA" → multiplier 1.105. But empirical ratio of Lane G v3 anchor: Vast.ai-CUDA PoseNet 0.00345458 / Modal-CPU PoseNet 0.00305354 = `1.1313`. The deferral file applied 1.105 (≠ 1.131). Off by 2.5pp.
- **Recompute with correct drift**: PoseNet CUDA-pred = 0.00309012 × 1.131 = 0.003495. Score = 100·0.00400950 + √(10·0.003495) + 25·686557/37545489 = `1.0450`. Not 1.0429.
- **Implication**: OWv3 candidate is actually +0.001 ABOVE the frontier (slightly worse), not −0.0011 below. **The DEFER decision is right** (the candidate is no better than frontier, possibly worse) **but the rationale is wrong** (it's not "below noise floor" — the candidate is genuinely on the wrong side of frontier).
- **Council vote**: Yousfi CRITICAL / Hotz CRITICAL (this is the kind of arithmetic error that causes wasted dispatch decisions) / Shannon CRITICAL (R(D) inference is invalid) / Dykstra CRITICAL / Fridrich CRITICAL. **5/5 unanimous CRITICAL**.
- **Fix**: recompute drift factor in `all_scores_inventory_20260430.md` §B; update deferral file's CUDA-pred score to 1.0450; flag DEFER reason as "predicted slightly ABOVE frontier" not "below noise floor".

### CRITICAL #4 — Modal eval is CPU but inventory labels it `[Modal-T4-CUDA]` (Yousfi)
- **Source**: `all_scores_inventory_20260430.md` Section E line 102: "Modal T4 (CUDA) | Lane G v3 | 1.04 | `experiments/results/modal_auth_eval_9b20bdfca246.json`"
- **Evidence**: `experiments/modal_auth_eval.py:164` literally passes `"--device", "cpu"` to `evaluate.py`. Modal *runs on T4 hardware* (gpu="T4" line 61), but evaluate.py is forced to CPU device. **This is a `[Modal-T4-CPU advisory]` result, NOT `[Modal-T4-CUDA]`.**
- **Cascade**: this means the "Modal pipeline trusted within 0.01 noise floor" claim in `feedback_modal_pipeline_trusted_lane_g_v3_1_04_20260429` is comparing Modal-CPU 1.04 to Vast.ai-CUDA 1.05 — that 0.01 gap IS the CPU↔CUDA drift, NOT a noise floor.
- **Council vote**: Yousfi CRITICAL (mislabeling kill/promote-grade evidence is the exact CLAUDE.md non-negotiable) / Selfcomp CRITICAL (the all-scores table's Section A "kill/promote-valid set" claim depends on this row being CUDA) / Contrarian CRITICAL (the unanimous-trust of Modal pipeline is the council-conduct red flag) / Shannon CRITICAL / Hotz CRITICAL. **5/5 unanimous CRITICAL**.
- **Fix**: re-label Section E row to `[Modal-T4-CPU advisory]`; update the cross-platform confirmation table to acknowledge the 0.01 difference is CPU↔CUDA drift; update CLAUDE.md note that "Modal canonical for >2h training" should NOT be read as "Modal-T4 eval is CUDA-grade".

### CRITICAL #5 — PCC4 STRICT false-positive blocks shannon_floor_execution_state_checkpoint commit (Contrarian/Selfcomp)
- **Source**: `src/tac/preflight.py:18602` — `_PCC4_KILL_BODY_LITERALS = ("VERDICT: KILL", "FALSIFIED", "RETIRED",)`. The substring `RETIRED` appears once in the checkpoint file, in the Wave 1 status table row referring to Lane 12 NeRV α-redesign as a previously-retired sub-lane.
- **Reproduction**: `python -c "from src.tac.preflight import check_kill_memory_files_have_council_review; print(check_kill_memory_files_have_council_review(strict=False, verbose=True))"` — flags the just-landed checkpoint with 4 missing sections.
- **Why it matters**: the PCC4 design intent is to catch FILES THAT ARE THEMSELVES KILL VERDICTS. A status-table row CITING a previously-retired lane is NOT the same thing. The greedy substring match makes any future status doc that quotes a kill memory line into a PCC4 violation.
- **Council vote**: Contrarian CRITICAL (this just-landed STRICT check is structurally over-broad) / Selfcomp CRITICAL (it blocks the very memory file documenting the council's own work) / Hotz MEDIUM (the fix is straightforward — title-prefix gate or length-of-context check) / Yousfi MEDIUM / Shannon MEDIUM. **3 CRITICAL / 2 MEDIUM**.
- **Fix options**:
  - (A) Tighten body-literal trigger: require the literal to be in a TITLE (frontmatter `name:` line) OR appear in a `## Verdict` / `## Status` heading, not in a table cell.
  - (B) Add a "self-citation" auto-pass: if `RETIRED` / `FALSIFIED` appear ONLY inside markdown table rows (lines with leading `|`), skip the file.
  - (C) Require the trigger literal to appear OUTSIDE markdown table syntax (lines without leading whitespace+`|`).
  - **Council recommendation**: Option C — simplest, smallest semantic change. Combined with fix to Option A in a follow-up if Option C still over-fires.
- **Blocker**: the checkpoint can't be committed under STRICT until this is fixed (or the checkpoint adds a `COUNCIL_REVIEW_SKIPPED_USER_OVERRIDE:` line, which is itself a CLAUDE.md non-negotiable violation since the file isn't a kill verdict).

### MEDIUM #6 — PFP16 A++ frontier baseline missing from all_scores_inventory.md Section A (Yousfi/Fridrich)
- **Source**: `all_scores_inventory_20260430.md` Section A is a 12-row table; Lane G v3 PFP16 A++ (1.044, the *current frontier*) does NOT appear. Lane G v3 (1.05) is row 1.
- **Why it matters**: the readiness doc and checkpoint both call PFP16 A++ "the controlling deploy baseline" but the canonical authoritative-scores table doesn't list it. Future agents reading the inventory will not see the frontier.
- **Council vote**: Yousfi MEDIUM / Fridrich MEDIUM / Selfcomp MEDIUM / MacKay LOW (this is a documentation completeness issue, not a math/strategy failure). **3 MEDIUM / 1 LOW**.
- **Fix**: add a row 0 (or insert row 1 with PFP16 A++) to Section A of `all_scores_inventory_20260430.md`.

### MEDIUM #7 — Drift coefficient misapplied to candidate that is itself the same source class (Shannon)
- **Source**: `project_owv3_byte_feasible_candidate_dispatch_deferred_20260501.md`
- **Issue**: the drift correction applied to OWv3 candidate is derived from the difference between Modal-CPU Lane G v3 and Vast.ai-CUDA Lane G v3. But the OWv3 candidate is **also** evaluated on Modal-CPU. The correction takes a CPU result and "predicts" what CUDA would say. That is structurally legitimate IF (a) the drift coefficient is constant across renderers (which it isn't — drift varies by checkpoint/architecture/distribution), and (b) the magnitude of the drift is empirically calibrated (only ONE anchor pair, n=1). Single-point calibration with no error bars is not a "prediction"; it's a guess with overconfidence.
- **Council vote**: Shannon MEDIUM / Dykstra MEDIUM / MacKay MEDIUM (no Bayesian prior; treating point estimate as ground truth) / Quantizr MEDIUM (similar archs may drift differently). **4 MEDIUM**.
- **Fix**: tag the CUDA-pred score as `[contest-CUDA single-point drift extrapolation, ±?]` not `[contest-CUDA prediction]`. Acknowledge n=1 calibration. If a 2nd CPU↔CUDA pair exists for any other lane, fold it in.

### MEDIUM #8 — Lane 17 asymmetric regression hypothesis vote is on UNVERIFIED hypothesis (Contrarian)
- **Source**: `project_lane_17_imp_killed_cycle_0_198_regression_20260430.md` §"Why the asymmetric regression supports..."
- **Issue**: "Hypothesis 1 (stub-bug, motion.head more weight-perturbation sensitive than seg.head) — council 8/10 favored". But the file admits "Hypothesis 2 has no supporting evidence — we don't have a converged sparse cycle-0 to test it". Hypothesis 1 also doesn't have direct evidence — only a Frankle 2019 reference (which is about general lottery-ticket theory, not THIS architecture). An 8/10 council vote on a hypothesis with NO direct evidence is the unanimous-vote-is-suspicious pattern from CLAUDE.md "Council conduct".
- **Council vote**: Contrarian MEDIUM / Hotz MEDIUM (Frankle is a prior, not evidence) / Dykstra MEDIUM (we should defer hypothesis-selection until cycle 0 with proper train_distill lands) / Quantizr LOW. **3 MEDIUM / 1 LOW**.
- **Fix**: weaken the file's claim to "hypothesis 1 is more parsimonious (Frankle prior); hypothesis 2 cannot be ruled out until proper cycle 0 lands".

### MEDIUM #9 — PCC1 evasion: `python -m experiments.train_distill` not detected (Hotz/Quantizr)
- **Source**: `src/tac/preflight.py:_scan_imp_dispatcher_for_train_distill_swap` regex requires `<runner> -u <target>` where target is the file path, not module path.
- **Test**: a dispatcher with `"$PYBIN" -m experiments.train_distill --resume foo` (legitimate) FAILS PCC1 (false positive). A dispatcher with `python experiments/train_distill.py` (no `-u`) ALSO fails (correctly catches missing `-u`, but motivates an arms race).
- **Council vote**: Hotz MEDIUM / Quantizr MEDIUM / Selfcomp LOW / Ballé LOW. **2 MEDIUM / 2 LOW**.
- **Fix**: extend regex to also match `<runner> [-u] -m experiments.train_distill` AND `<runner> [-u] experiments/train_distill.py`. Document the contract.

### LOW #10 — PCC3 has no dedicated test file (Selfcomp)
- **Source**: `src/tac/tests/` has `test_no_comment_only_contracts.py` (PCC2), `test_check_pcc4_kill_memory_council_review.py` (PCC4), `test_preflight_imp_dispatch_train_distill.py` (PCC1). **PCC3 (`check_stats_json_internal_consistency`) has no dedicated test file.**
- **Council vote**: Selfcomp LOW / Ballé LOW. **2 LOW**.
- **Fix**: add `test_check_stats_json_internal_consistency.py` with positive (waiver works), negative (missing assertion gets caught), and inter-function-waiver tests.

### LOW #11 — Score "100KB → 0.0666" is correct but 7,439 byte → 0.005 has 1pp imprecision (Shannon)
- **Source**: checkpoint line 25 — "7,439 byte PFP16 win → 0.005 score". Math: 25·7439/37545489 = 0.00495. Rounding to 0.005 is fine; flagging only because the same checkpoint's nearby line 24 errors are off by 28% — wanted to double-check this one.
- **Council vote**: Shannon LOW (acceptable rounding). **1 LOW**.
- **Fix**: none required (acceptable rounding).

## Top 3 most-impactful findings

1. **CRITICAL #4** — Modal labeling drift contaminates the entire all-scores authoritative table. Affects Section A row 1 (Lane G v3 cross-platform claim), affects every downstream "Modal noise floor" claim, affects the OWv3 deferral.
2. **CRITICAL #5** — PCC4 false-positive is a self-blocking STRICT check that prevents the very file documenting it from being committed. This is gate-stacking that must be fixed BEFORE any greenup pass can land.
3. **CRITICAL #1+#2 (joint)** — Score derivative arithmetic errors (28% on rate, 26% on PoseNet ×10) skew every Wave 1 dispatch ranking by misweighting the underlying score levers.

## Counter logic

Round 1 found 11 findings (5 CRITICAL + 4 MEDIUM + 2 LOW). **Counter resets to 0/3.**

Round 2 cannot run until Round 1 fixes land (or are explicitly deferred with operator approval).

## Blockers requiring operator action

1. **PCC4 false-positive fix** — needs design call: Option C (skip table-row matches) is council recommendation. Once landed, the checkpoint can be committed without `COUNCIL_REVIEW_SKIPPED_USER_OVERRIDE:`.
2. **Drift coefficient correction** — `all_scores_inventory_20260430.md` and `project_owv3_byte_feasible_candidate_dispatch_deferred_20260501.md` need a coordinated rewrite: drift is `×1.131`, not `×1.105`; OWv3 candidate is `1.045` not `1.043`; rationale shifts from "below noise floor" to "predicted above frontier".
3. **Modal CPU-vs-CUDA labeling correction** — `all_scores_inventory_20260430.md` Section E and Section A cross-platform note need a re-label. This is documentation only but affects strategic interpretation.
4. **Score derivative arithmetic fix** — checkpoint lines 24/26 need recomputation. Pure documentation fix, no GPU spend.

## What I CAN fix without operator approval

- **PCC1 evasion hole** (MEDIUM #9): extend regex to cover `python -m` invocation pattern.
- **PCC3 missing test** (LOW #10): add dedicated test file.
- **Lane 17 hypothesis weakening** (MEDIUM #8): tone down the 8/10 vote claim to "hypothesis 1 is more parsimonious".
- **Score derivative arithmetic fix in checkpoint** (CRITICAL #1+#2): pure math correction.

## Recommendation

**HALT — escalate to operator** for findings #4 (Modal labeling), #5 (PCC4 false-positive), #6 (PFP16 inventory row), #7 (drift n=1 caveat). These are documentation/policy decisions that benefit from operator visibility before landing.

For the other 6 findings (#1, #2, #8, #9, #10, #11), proceed with single-thread serial fixes, then re-run Round 2.

**3-clean-pass gate status: 0/3.** Cannot continue passes until at least the CRITICAL findings are addressed.

## Cross-refs

- `project_shannon_floor_execution_state_checkpoint_20260501.md` (under review; CRITICAL #1, #2 fixes target this file)
- `project_owv3_byte_feasible_candidate_dispatch_deferred_20260501.md` (under review; CRITICAL #3, MEDIUM #7 fixes target this file)
- `all_scores_inventory_20260430.md` (under review; CRITICAL #4, MEDIUM #6 fixes target this file)
- `feedback_grand_council_imp_permanent_fix_review_20260430.md` (under review; PCC1+2+3+4 design vote tallies — design intent is sound, implementation has the bugs caught here)
- `project_lane_17_imp_killed_cycle_0_198_regression_20260430.md` (under review; MEDIUM #8 fix targets this file)
- `feedback_imp_local_backport_landed_20260430.md` (under review; nothing major found beyond cross-references)
- `.omx/research/shannon_floor_execution_readiness_20260430.md` (under review; cross-checks against checkpoint reveal CRITICAL #1 contradiction)
- `src/tac/preflight.py` (under review; CRITICAL #5, MEDIUM #9, LOW #10 target this file)

## Process note

Per CLAUDE.md "Council conduct — non-negotiable", two unanimous council votes here (8/10 on Lane 17 withdrawal, 5/5 on the unanimous CRITICAL findings) trigger contrarian self-skeptic check. The 5/5 votes here ARE legitimate (math errors are not opinions; a 100KB→0.0666 calculation has a single right answer). The 8/10 vote on Lane 17 hypothesis IS the suspicious-unanimity pattern flagged in MEDIUM #8.

## Internal consistency check

This memory file is itself a kill-adjacent record (it discusses kill literals + cites RETIRED/FALSIFIED) so PCC4 will scan it. To pass:

- 1. **Grand Council header present**: Yes (every CRITICAL/MEDIUM finding has a "Council vote" line).
- 2. **Inner council members named with rationale**: Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Quantizr, Hotz, Selfcomp, MacKay, Ballé all appear with rationale across findings #1-#11. Count ≥5.
- 3. **Each finding has a vote-count tally**: every finding lists count (5/5 unanimous CRITICAL, 3 CRITICAL/2 MEDIUM, etc.).
- 4. **Math claims recomputed independently**: each score-derivative finding has the recomputed value (`25 × 421000 / 37545489 = 0.2803` etc.) verified via `bc -l`.
- 5. **Cross-references to source files**: every finding cites the file path + line being challenged.

## What would change my mind

This greenup-pass record can be retracted IF:

1. The Modal eval is verified to be running CUDA (not CPU) by inspecting a fresh provenance dump from `experiments/modal_auth_eval.py` — i.e. CRITICAL #4 is wrong about the CPU labeling. (Currently the script literally passes `--device cpu` at line 164, so this is unlikely.)
2. The score derivative values in the checkpoint were computed against a different convention (e.g. `bytes / 1e6` instead of `bytes / 37545489`) and the math errors are actually a unit-mismatch by the council not the checkpoint author. (Spot-checked: 100KB → 0.0666 IS correct under the canonical formula, so the convention is the right one. Errors stand.)
3. A 2nd Modal CPU↔CUDA pair lands and the empirical drift is closer to ×1.105 than ×1.131 — would require operator dispatch to verify (~$0.50 Vast.ai 4090).
4. The PCC4 false-positive fix (CRITICAL #5) lands and the shannon_floor checkpoint commits clean — would confirm the Option C fix was sufficient.
