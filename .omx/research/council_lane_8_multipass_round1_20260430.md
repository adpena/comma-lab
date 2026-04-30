# Council — Lane 8 Multi-Pass — Adversarial Review Round 1

Date: 2026-04-30
Round: 1 of 3 (3-clean-pass adversarial review per CLAUDE.md)
Reviewers (rotating perspectives): Yousfi, Fridrich, Contrarian, Quantizr, Hotz
Files under review:
- `src/tac/multipass_compressor.py` (codec)
- `src/tac/tests/test_multipass_compressor.py` (23 unit tests)
- `experiments/pipeline.py` (PipelineConfig fields + step_multipass + run_compress wiring + CLI flags)
- `experiments/lane_8_multipass_real_archive_smoke.py` (offline byte proxy)
- `scripts/remote_lane_8_multipass.sh` (canonical dispatch)
- `src/tac/preflight.py` (Check 92)
- `src/tac/tests/test_check_no_inflate_time_multipass.py` (7 preflight tests)
- `src/tac/profiles.py` (MULTIPASS_LANE_G_V3 profile)

## Yousfi (challenge-creator perspective; verifies contest-compliance)

CRITICAL findings: 0
Medium findings: 1
Low findings: 1

**Medium #1**: Stage 4 of `remote_lane_8_multipass.sh` reads `final_score` from `multipass_summary.json` but the multipass score is the OUTPUT of `step_eval(archive)` which IS the contest-CUDA auth eval. So that score IS contest-compliant. However, the lane script does NOT call `experiments/contest_auth_eval.py` separately on the FINAL archive. Without a separate canonical contest-CUDA eval after multi-pass converges, we are reporting the multipass-internal score (which is `step_eval` against `multipass_archive.zip`, not the named final archive at `iter_0/archive.zip`). This is a logical equivalence but lacks the canonical eval signature. Fix: add a Stage 5 contest_auth_eval on `$FINAL_ARCHIVE` to match other lane scripts.

**Low #1**: The `multipass_target_score` default of 0.0 → "use baseline - 0.005" is documented in the dataclass docstring, but the CLI help string for `--multipass-target-score` says "0 = baseline - 0.005" without the qualifier that this code path currently uses target=-1.0 instead (impossible) so target_hit never short-circuits. Documentation drift between the convention and the actual implementation. Fix: clarify CLI help to "0 = sentinel: never short-circuit on target_hit; rely on eps + regression guard."

## Fridrich (steganalysis-author perspective; verifies attack surface)

CRITICAL findings: 0
Medium findings: 0
Low findings: 1

**Low #1**: The `CoordinateDescentPolicy` priority order matches the score-arithmetic priority from `project_codec_stacking_composition_canonical_orders_20260429.md` (mask CRF first because of 45× headroom). Fridrich approves — this aligns with the UNIWARD principle that compression cost goes to the textured (high-CRF-tolerable) regions. No attack surface introduced.

## Contrarian (challenge any weak argument)

CRITICAL findings: 0
Medium findings: 1
Low findings: 2

**Medium #1**: The synthetic test `test_synthetic_quadratic_converges_in_two_passes` uses CRF step 5.0 (the policy's default). With initial CRF=50 and a quadratic minimum at CRF=50, the next pass moves to CRF=55 which produces score (5)²/100 = 0.25. delta = 0.0 - 0.25 = -0.25 < -eps → regression guard fires, revert to pass 0. So the test passes. But the test's `assert result.best_pass_idx == 0` is actually robust ONLY because pass 0 happens to BE the minimum. If the initial point is OFF the minimum (e.g., CRF=45) the same loop SHOULD walk toward CRF=50 and converge there. The test does NOT exercise this convergence behavior — it exercises the DETECT-OPTIMUM-AT-INIT path.

  Fix: add a 2nd test where the initial CRF is below the minimum (e.g., CRF=42), confirming the policy walks UP to CRF=47 (or wherever) and STOPS at the score plateau. This proves convergence, not just regression detection.

**Low #1**: The compressor's `regression_guard=False` mode is documented but the test only confirms it tracks best-so-far. There's no test that confirms `regression_guard=False` actually allows the policy to PROBE PAST a regression and recover (the use case for non-monotonic policies). Test passes but docstring says "future caller may swap policy for a non-monotonic one" without a regression test for that specific behavior.

**Low #2**: The `_InflateTimeAssertion` only checks `__main__.__file__` — it doesn't walk the call stack. A test that imports MultiPassCompressor from a helper module called BY inflate_renderer.py would NOT trigger the assertion (because `__main__` is still inflate_renderer.py, but if the helper is `inflate_renderer_helper.py` and IS the `__main__`, the assertion fires; if it's imported from something else, it does NOT fire). The defense in depth is Check 92 (static scan), which catches the actual code patterns. The runtime assertion is a defense-in-depth backup. Document this trade-off in the docstring.

## Quantizr (adversarial reverse-engineer; checks for hidden value)

CRITICAL findings: 0
Medium findings: 1
Low findings: 0

**Medium #1**: The `MULTIPASS_LANE_G_V3` profile inherits from `DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL` but profile keys with `multipass_*` prefix are read by `experiments/pipeline.py` via the `_apply_profile` resolver. There's a risk that a SILENTLY-OVERRIDDEN value from the inherited profile (e.g., a `mask_crf` key) would conflict with the multipass loop's per-pass adjustment. Check `_apply_profile` behavior: profile values are applied at compress START, then the multipass loop OVERWRITES `cfg.mask_crf` per pass. This works because the multipass loop directly mutates `cfg.mask_crf`. But: what if the inherited profile sets `mask_crf=42` and the user passes `--mask-crf 50` on the CLI? The CLI wins (per `_user_provided_flags`), then the multipass loop starts at 50. Coherent. No bug, but document this in the profile docstring.

## Hotz (raw engineering shortcut)

CRITICAL findings: 0
Medium findings: 0
Low findings: 1

**Low #1**: Hotz approves the implementation. ~500 LOC of codec + ~430 LOC of tests + clean integration. Coordinate descent on 4 axes. Regression guard. Eps stop. Param clamping. Inflate-time assertion. Nothing fancy. The 30-min Carmack version IS the default (max_passes=3). One nit: `_clamp_params` ignores unknown keys — a misspelled axis name like `mask_cfr` (typo for `mask_crf`) would silently pass through and become inert. Add a warn-on-unknown-key option that operators can enable for development.

## Round 1 verdict

**Counter: 0 / 3 clean.**

Total findings: 0 CRITICAL + 4 Medium + 5 Low.

Medium findings to address before Round 2:
1. (Yousfi) Add Stage 5 separate `contest_auth_eval.py` invocation in `remote_lane_8_multipass.sh` (defense-in-depth canonical eval)
2. (Yousfi-Low) Clarify CLI `--multipass-target-score` help text
3. (Contrarian) Add convergence-direction test (initial OFF minimum)
4. (Quantizr) Document the profile inheritance + multipass loop's mutation of `cfg.mask_crf` (no bug, just clarity)

Low findings (acceptable to defer to Round 3 if no other issues land):
- regression_guard=False non-monotonic policy regression test
- Document _InflateTimeAssertion trade-off
- _clamp_params warn-on-unknown-key option

## Action items for Round 2

1. Add Stage 5 contest_auth_eval to `remote_lane_8_multipass.sh`.
2. Clarify CLI help string for `--multipass-target-score`.
3. Add convergence-direction test (`test_initial_off_minimum_converges_to_minimum`).
4. Add profile-inheritance docstring note in MULTIPASS_LANE_G_V3.

These are not bugs in the implementation — they are defense-in-depth and documentation hardening per the Level 3 standard. Round 2 will rotate perspectives (Shannon, Dykstra, Selfcomp, MacKay, Ballé) and audit again.
