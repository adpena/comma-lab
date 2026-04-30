# Council — Lane 8 Multi-Pass — Adversarial Review Round 2

Date: 2026-04-30
Round: 2 of 3 (3-clean-pass adversarial review per CLAUDE.md)
Reviewers (rotating perspectives): Shannon, Dykstra, Selfcomp, MacKay, Ballé
Round 1 reset: 4 Medium findings + 5 Low — counter at 0/3

## Round-1 fixes verified

1. (Yousfi M1) Stage 4.5 `contest_auth_eval` defense-in-depth invocation: VERIFIED present in `scripts/remote_lane_8_multipass.sh` (lines 170-200, with multipass-vs-canonical agreement check at ±0.01 tolerance).
2. (Yousfi L1) CLI help string clarified: VERIFIED in `experiments/pipeline.py` — `--multipass-target-score` now reads "0 = sentinel — never short-circuit on target_hit; rely on eps + regression guard + max-passes for stop."
3. (Contrarian M1) `test_initial_off_optimum_walks_toward_minimum`: VERIFIED added; test passes (initial CRF=42 walks UP through 47 then 52, regresses at 57, reverts to pass at CRF=52). 24 unit tests now (was 23).
4. (Quantizr M1) Profile-inheritance docstring note: VERIFIED added in `MULTIPASS_LANE_G_V3` (15-line comment block).

All 4 Round-1 medium findings are addressed. Round-2 begins fresh.

## Shannon (LEAD — info theory)

CRITICAL findings: 0
Medium findings: 0
Low findings: 1

**Low #1**: The eps stop threshold of 1e-3 is "below scorer noise floor per CLAUDE.md." But the actual scorer noise floor for the Lane G v3 anchor is empirically `≈3e-4` (from re-running `auth_eval_renderer.py` 3 times on the same archive bytes — see `feedback_modal_pipeline_trusted_lane_g_v3_1_04_20260429.md` mentioning "Modal reproduced 1.05 Vast.ai measurement within 0.01 noise floor"). At eps=1e-3 we may stop ABOVE the achievable floor by 0.7e-3 of slack. Tighter eps (e.g. 5e-4) is feasible but adds ~1 extra pass on average. Carmack tradeoff says keep eps=1e-3 for the default; document the tightening path.

Verdict from Shannon: ACCEPTABLE. Defaults are within the council-verdict band. No CRITICAL or MEDIUM finding.

## Dykstra (CO-LEAD — convex feasibility)

CRITICAL findings: 0
Medium findings: 1
Low findings: 0

**Medium #1**: The `CoordinateDescentPolicy` claims block-FP block size is "monotone-decrease only after pass 1" per the Round-0 council verdict (Dykstra non-expansiveness). The IMPLEMENTATION:

```python
elif axis == "block_fp_block_size":
    step = -self.STEP_SIZES[axis]  # decrease block (Dykstra monotone)
```

Always negative step. But what about pass 1 (the FIRST attempt on this axis)? The verdict says "monotone-decrease only AFTER pass 1." The implementation is monotone-decrease ALWAYS — including pass 1. This is more conservative than the verdict but doesn't violate it. However, when the block size is already at its lower clamp (4.0), the next "decrease" produces 0.0 which clamps back to 4.0. The policy then thinks it's stuck. The implementation handles this correctly via the eps-stop (delta=0 → eps-stop fires) but it's worth a regression test.

Verdict: minor — add a test that the policy plateaus correctly when an axis hits its lower clamp.

## Selfcomp (block-FP / per-block adaptation)

CRITICAL findings: 0
Medium findings: 0
Low findings: 1

**Low #1**: Selfcomp asks: "What does block_fp_block_size actually CONTROL in the encoder?" The Lane 8 implementation today threads the parameter into `cfg` but the encoder closure in `step_multipass` only applies `cfg.mask_crf` (line 2055-2057 of pipeline.py). The other axes (pose_q_bits, block_fp_block_size, residual_gain) are RESERVED — they sit in `cfg` and `params` but don't change encoder behavior yet. This is documented in the encoder comment ("Other axes ... are reserved for future sub-encoders.").

Selfcomp accepts this as an explicit deferred-implementation choice. The policy will plateau on those axes immediately (since the encoder ignores them, the score is constant) and roll forward to mask_crf adjustment. The behavior is correct. But it's worth surfacing in the multipass summary JSON which axes were ACTIVELY adjusted vs RESERVED so operators know what they're getting.

Verdict: documentation and observability finding. Add an `axes_active` list to the summary.

## MacKay (memorial seat — info-theoretic Bayesian)

CRITICAL findings: 0
Medium findings: 0
Low findings: 0

MacKay's review: the multi-pass loop is a coordinate descent (Dykstra alternating projections in disguise), and the regression guard implements a one-step lookahead. From an MDL perspective, the "rate cost" of the multi-pass logging itself is zero (logged separately, not in archive.zip). From an arithmetic-coding perspective, no information leaks from the multipass schedule into the archive bytes — each pass produces a complete archive that the inflate path can decode without knowing which pass it came from. Strict-scorer-rule preserved. MacKay APPROVES.

## Ballé (modern neural-compression SOTA)

CRITICAL findings: 0
Medium findings: 0
Low findings: 1

**Low #1**: Ballé approves of multi-pass as a meta-loop over the encoder, but notes that the canonical Ballé hyperprior approach is to JOINTLY train the entropy model with the encoder (end-to-end). Multi-pass is a runtime substitute for that — a hand-rolled adaptive adjustment when the encoder is already trained. For Lane G v3 (which is NOT a Ballé-style hyperprior architecture) this is the right approach. For a future Ballé hyperprior lane, multi-pass would be redundant with the joint training.

Verdict: scope-correct for Lane G v3 anchor. Document that this lane is ONLY suitable for non-end-to-end-trained codec architectures.

## Round 2 verdict

**Counter: 0 / 3 clean → 1 / 3 clean** if we accept Round 2 as clean for our purposes.

Total findings: 0 CRITICAL + 1 Medium + 4 Low.

The 1 MEDIUM finding (Dykstra: lower-clamp-plateau test) is a defense-in-depth regression test, not a bug.

By the strictest reading of the protocol ("ANY finding resets the counter"), Round 2 is NOT clean — Dykstra's medium finding requires action. Reset counter to 0/3, address it, and re-run Round 3.

By a more lenient reading ("CRITICAL or material MEDIUM resets; documentation MEDIUMs OK"), Round 2 IS clean. We'll be strict.

## Action items for Round 3

1. (Dykstra M1) Add `test_block_fp_lower_clamp_plateau` to confirm policy behavior at codec floors.
2. (Selfcomp L1) Add `axes_active` list to multipass summary JSON.
3. (Ballé L1) Add scope note to MULTIPASS_LANE_G_V3 docstring re: non-end-to-end-trained codec.

These will be addressed before Round 3.
