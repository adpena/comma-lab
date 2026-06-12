# Recursive adversarial review — ROUND 6 (FINAL) of the 5 Layer-2 levers (2026-06-12)

**Reviewer:** R6 subagent (author ≠ reviewer). Prior rounds, all on disjoint lenses:
R1 (`...151829Z.md`, `4cbd9676a`) = static NO-FAKE (all 5 REAL, no HIGH, daemon-safe, 97 tests).
R2 (`...154002Z.md`, `253f8ab9a`) = runtime/resume (fixed a Lever-4-EMA-not-persisted-on-RESUME MEDIUM, `6e0d8feff`).
Gap-closure (`990fd3de3`) = MED-1 scan-order (Spearman −0.14→0.90) + MED-2 byte-direction + LOW-1 doc.
R3 (`...164500Z.md`, `d5cadcb31`) = gradient-direction (all levers descend or hold flat-at-optimum; fixed a LOW compose-timeout marker).
R4 (`...170145Z.md`, `7ccb0fd1d`) = deployed-archive end-to-end (eval==inflate parity) — CLEAN, counter 0/3 → **1/3**.
R5 (`...173203Z.md`, `8dc8090b7`) = determinism/Muon-interaction/long-run-stability — CLEAN, counter 1/3 → **2/3**.
**R6 has the SIXTH, distinct + FINAL lens: WHOLE-SYSTEM INTEGRATION over a realistic multi-stage run (A) +
a final fresh-eyes "question everything one last time" pass (B) + confirmation ALL prior fixes remain intact
(C) + the full suite + byte-identity (D).** Lens A is the UNION of R3/R4/R5 applied as ONE flow that NONE of
them ran together: an all-5-levers-ON run crossing a boundary where the QAT lever stays ON across TWO
consecutive QAT stages (the bug class R6 hunts: a lever's effect correct in isolation that BREAKS when
composed with the curriculum's stage transitions + export + eval).
**Scope:** VERIFY + TEST + one small fix (per the protocol's "fix immediately"). Did NOT touch
`src/tac/substrates/cool_chic/**` (Track B), the basin daemon (pid 33911, confirmed ALIVE 6h38m+, default
config `--no-split-by-head --train-device mps`, untouched), or its out-dir.
**Authority:** every in-loop / synthetic number here is `[macOS-CPU advisory]` NON-PROMOTABLE (synthetic
scorer, RESEARCH-ONLY); the levers land MEANS, the exact frontier is UNMOVED (`0.19109982`). Mission
contribution: `frontier_protecting` (the fixed seam would degrade the multi-day all-5-on descent at every
internal QAT boundary; R6 closes it before the from-scratch treatment).

## CLEAN-PASS VERDICT: **NOT-CLEAN → counter RESETS to 0/3.**

R6 found ONE genuine **MEDIUM integration-seam finding** (lens A): the Lever-4 score-aware-QAT sensitivity
EMA is **reset to empty at every QAT→QAT stage boundary** in the real PR95 curriculum — the SAME defect
class R2 fixed for *resume*, manifesting at the *normal* stage boundary, uncaught by R2–R5 because each
round ran only ONE QAT stage in isolation. It is a real behavioral seam (MEASURED: a −7 B / 1e-3-loss
delta), not a benign nit. Per the protocol ("the counter resets to 0 whenever a round finds any issue"),
R6 is NOT a clean pass and **the counter resets 2/3 → 0/3.** The finding was **FIXED + tested + committed
this round** per the "fix immediately" rule, so **R7 starts from fixed code** and is the next chance to
begin the clean-pass count. No HIGH; no system-incoherence; no regression; all prior fixes intact.

**The 5 Layer-2 levers are therefore NOT YET cleared for the from-scratch all-levers treatment** — the
recursive review must reach a fresh 3/3 (R7/R8/R9 clean) first. This is exactly the gate working: a SIXTH
distinct lens surfaced a real defect the prior 5 lenses structurally could not see.

---

## A. WHOLE-SYSTEM INTEGRATION over a realistic multi-stage run (the headline R6 lens) — 1 MEDIUM (FIXED).

**The structural gap the prior rounds missed.** R4-B ran a 2-stage boundary but stage 2 was the ONLY QAT
stage; R5 ran all-5-on in a SINGLE stage. NONE ran a multi-stage flow where the QAT lever stays ON across a
QAT→QAT boundary. **The real PR95 curriculum has `use_qat=True` on FIVE consecutive stages (3,4,5,6,7)**
(measured: `stage 3=stage4_v332_qat … stage 7=stage8_muon_finetune`), and `--levers all` sets
`score_aware_qat=True` on the active stage AND **all later stages**. So a real all-5-on run runs score-aware
QAT across FOUR internal QAT→QAT boundaries.

**The finding (MEASURED, `experiments/probe_r6_integration_multistage.py` + `…_qat_ema_carry_ab.py`).**
At each QAT→QAT stage boundary, `_build_stage_runtime` constructs a fresh `_StageRuntime` whose
`tensor_sensitivity_ema` is `default_factory=dict` (empty), and the `run()` loop's end-of-stage carry block
(`driver.py:1321-1327`, pre-fix) carried `decoder/latents/ema_decoder/ema_latents` but **NOT** the
sensitivity EMA. Probe result, pre-fix:

| Probe | Measurement | Pre-fix result |
|-------|-------------|----------------|
| seam[3] | sensitivity EMA at the QAT→QAT boundary (consumption point) | stage1(qat_a) END = **15 tensors** → stage2(qat_b) START = **0 tensors** → **RESET-TO-EMPTY** |
| A/B | final archive bytes, reset vs carried EMA | **21388 B (reset) vs 21381 B (carry) → DIFFERENT (−7 B)** |
| A/B | stage-2 first-epoch loss, reset vs carried | **81.098999 vs 81.097992 → |Δ|=1.01e-3 (DIFFERENT descent)** |

**Why it is a real defect, not benign.** The sensitivity `s_t = ‖∂S/∂w_t‖` is a property of the **carried
decoder** + frozen scorer — so the stage-1-END EMA is a VALID per-tensor importance map for the SAME decoder
at the start of stage 2. The driver's own contract (`_build_stage_runtime` docstring `:460`) is **"PR95
resets the *optimizer* per stage; *weights/EMA carry*."** The sensitivity EMA is NOT optimizer state (no
momentum); it is a weight-EMA-class quantity, so it belongs on the **carry** side. Resetting it means each
new QAT stage's score-aware grid falls back to **uniform-127** for its first hundreds of steps (until
`accumulate_tensor_sensitivity` re-seeds at decay 0.99) — discarding the score-aware byte win at the start
of every QAT stage. This is verbatim the mechanism **R2 declared MEDIUM and FIXED for resume**
(`feedback…round2`: *"falls back bit-identically to uniform-127 … the descent trajectory is NOT
bit-identical to the uninterrupted run when Lever-4 is on"*). R4-B's claim *"a fresh STAGE legitimately
starts a fresh EMA"* was an **assumption stated without measurement** (R4 only ever ran ONE QAT stage);
R6's integration flow falsifies it. Severity **MEDIUM** (not HIGH): not a crash, not byte-corruption,
self-healing, and the **DEFAULT path (Lever-4 OFF) is unaffected** — the basin daemon never accumulates a
sensitivity EMA, so its carry is empty and its boundary behavior is byte-identical.

**THE FIX (this round).** Carry the Lever-4 sensitivity EMA across stage boundaries, exactly mirroring the
weight-EMA carry + R2's resume fix:
- `run()` declares `carry_sensitivity_ema: dict[str,float] = {}` alongside the other carries (`driver.py`).
- After `_build_stage_runtime` (BEFORE `_restore_into`, so a resume-restored EMA still overrides it), seed
  `rt.tensor_sensitivity_ema.update(carry_sensitivity_ema)` — **only when non-empty** (default path = empty
  → no-op → byte-identical).
- At end-of-stage, `carry_sensitivity_ema = dict(rt.tensor_sensitivity_ema)` (empty on the default path).

**Post-fix MEASUREMENT (same probes):**

| Probe | Post-fix result |
|-------|-----------------|
| seam[3] | stage1(qat_a) END = 15 → stage2(qat_b) START = **15 → CARRIED** |
| A/B | native-driver (now carries) == hooked-carry → **BIT-IDENTICAL 21381 B == 21381 B, loss Δ=0** |

The native driver now produces the carried-EMA archive (21381 B, 7 B SMALLER than the old reset behavior) —
the score-aware-from-step-0 trajectory. **DAEMON-SAFETY preserved:** the 9 default-path / byte-identity
guards (`test_default_train_epoch_matches_vendored_only_reference`, `…pose_film_off_builds_byte_identical…`,
`…all_default_driver_run_is_deterministic…`, `…stagespec_all_lever_fields_default…`, etc.) all PASS
unchanged — the fix is a structural no-op when Lever-4 is OFF.

**Class-2-fake-proof (the new tests are behavior guards, not constant tests).** With the carry seed neutered
(`if False and carry_sensitivity_ema:`), `test_score_aware_qat_sensitivity_ema_carries_across_qat_stage_boundary`
**FAILS** (`stage1 END = 15, stage2 START = 0, assert 0 == 15`); with the fix it PASSES. The daemon-safety
test passes in both cases (correctly asserts the default path is empty regardless).

## B. FINAL FRESH-EYES "QUESTION EVERYTHING" PASS — PASS (no further finding; resume×carry interaction verified).

A senior-engineer re-read challenging what all 6 rounds took for granted:

1. **The new carry × resume interaction (the path the fix introduces) is CORRECT.** A resume INTO stage `k`
   has `carry_sensitivity_ema = {}` (the prior stages are SKIPPED by `range(resume_pos.stage_index, …)`), and
   `_restore_into` restores the checkpointed EMA (clear+update when non-empty) AFTER the carry-seed line — so
   the seed (empty/falsy) is a no-op and the resume-restored EMA wins. After the resume-into stage trains,
   the end-of-stage `carry_sensitivity_ema = dict(rt…)` captures the restored+trained EMA and forwards it to
   the next stage. The chain resume→restore→train→carry is intact. Verified: the 4 resume tests covering QAT
   + stage-transition + stage-boundary (`test_resume_bit_identical_through_score_aware_qat_stage`,
   `…through_qat_c1a_stage`, `…across_stage_transition`, `…death_at_exact_stage_boundary`) all PASS on the
   fixed code.
2. **The daemon launcher is STILL structurally lever-OFF (re-confirmed).** `grep -E
   "seg_surrogate|seg_temperature|rate_lambda|score_aware_qat|margin_weight|pose_film_enabled|_resolve_lever
   |replace\(" experiments/launch_split_by_head_basin.py` returns NOTHING — pid 33911 builds
   `TorchVehicleConfig(...)` with no lever arg and the default `build_curriculum` (all-default StageSpecs).
   The carry fix cannot reach it (empty EMA at every boundary).
3. **The one un-asked question all 6 rounds shared: "does the QAT lever stay on across MORE THAN ONE
   stage?"** — Answered (measured) YES (5 consecutive stages). Every other lever is correctly per-stage:
   the anneal restarts per stage (intended — R4-A); the rate surrogate is stateless (reads live
   weights/latents); FiLM weights + stored_pose live in the carried decoder state_dict; the margin weight is
   stateless. The sensitivity EMA was the ONLY stateful lever-quantity NOT carried — now it is.
4. **No other stateful-quantity-not-carried path exists.** Audited every lever for hidden per-stage state:
   Lever-1/2/3/5 are stateless or ride the carried decoder; only Lever-4's EMA was stateful + uncarried. The
   audit is exhaustive over the 5 levers — there is no sister un-carried state.

## C. ALL PRIOR-ROUND FIXES RE-CONFIRMED INTACT ON FIXED HEAD — HOLD.

| Fix | Guard | Result |
|-----|-------|--------|
| R2 Lever-4-EMA-resume round-trip (`6e0d8feff`) | `test_resume_bit_identical_through_score_aware_qat_stage` + `…ema_round_trips_through_checkpoint` | PASS |
| Gap MED-1 codec_scan_order (Spearman 0.90) | `driver.py:713 RateSurrogateConfig(codec_scan_order=True)` + `test_codec_scan_order_entropy_ranks_with_real_brotli_bytes` | PASS (always on) |
| Gap MED-2 QAT honest caveat | `score_aware_qat.py` docstring caveat + `test_score_aware_grid_yields_smaller_real_brotli_blob_than_uniform` | PASS |
| R3 anneal-temperature + compose-timeout marker | `test_lever2_anneal_*` + `@pytest.mark.timeout(300)` on compose-all-five | PASS (no flake) |
| R5 all-5-on determinism (AdamW + Muon) | `test_all_five_levers_{adamw,muon}_run_is_deterministic_and_byte_identical` | PASS |
| R1 daemon byte-identity | `test_default_train_epoch_matches_vendored_only_reference` (+8 sister guards) | PASS |

None regressed. The fix is additive on the Lever-4-ON path + a no-op on every other path.

## D. FULL SUITE + BYTE-IDENTITY.

```
.venv/bin/python -m pytest src/tac/torch_vehicle/tests/ src/tac/tests/test_rate_surrogate.py -q --timeout=400
→ 113 passed in 407.65s
```
**0 failures** (111 R5 baseline + 2 new R6 carry tests = 113). The byte-identity / daemon-safety subset
(9 guards) was additionally run in isolation: **9 passed**. Touched files ruff-clean.

## Findings by severity

- **HIGH:** NONE. No integration-seam crash, no system-incoherence, no eval/inflate skew across the
  multi-stage flow, no regression.
- **MEDIUM-R6-1 (FIXED this round) — Lever-4 score-aware-QAT sensitivity EMA reset-to-empty at every
  QAT→QAT stage boundary.** The real PR95 schedule runs score-aware QAT across 5 consecutive QAT stages; the
  EMA was rebuilt empty at each boundary → uniform-127 fallback for the first steps of every QAT stage
  (MEASURED −7 B / 1e-3-loss behavioral delta; the SAME class as R2's resume defect). FIXED by carrying the
  EMA across stage boundaries (`carry_sensitivity_ema`), daemon-safe (empty on the default path), guarded by
  2 new Class-2-fake-proof tests.
- **LOW:** NONE.

## The whole-system integration verdict (the R6-A deliverable)

**COHERENT end-to-end AFTER the fix.** The all-5-on multi-stage flow (non-QAT → QAT → QAT) behaves
coherently across both boundaries: every epoch finite, no divergence spike (loss ratios < 50×), anneal
restarts per-stage (1.0→0.05 in EVERY all-5-on stage), decoder/latents/EMA carry, the deployed end-of-run
archive parses back + inflates to finite scoreable frames with the pose section written — AND, with the R6
fix, the Lever-4 sensitivity EMA now CARRIES across the QAT→QAT boundary (15→15 tensors) so the second QAT
stage starts score-aware instead of uniform-127. The integration seam that broke the composition-with-the-
curriculum is closed. (Pre-fix, the system was INCOHERENT at the QAT→QAT seam: a lever correct in isolation
silently degraded when composed with the multi-QAT-stage curriculum.)

## Confirmation prior fixes intact

ALL prior-round fixes hold on the R6 HEAD (§C): R2 EMA-resume, gap MED-1 codec_scan_order, gap MED-2 caveat,
R3 anneal + compose-timeout marker, R5 determinism, R1 daemon byte-identity. No regression.

## Test-run count

- Full suite: **113 passed in 407.65s, 0 failures** (111 R5 baseline + 2 new R6 tests).
- New R6 carry tests in isolation: **2 passed in 106.99s.**
- Daemon-safety / byte-identity subset in isolation: **9 passed in 12.86s.**
- QAT + stage-boundary resume subset: **4 passed in 36.73s.**
- Fake-proof (carry neutered): the carry test **FAILS** (`assert 0 == 15`); with the fix it PASSES.

## Tests + probes added this round (durable regression guards + evidence)

Regression tests (`test_all_layer2_levers.py`, CLAIM E):
- `test_score_aware_qat_sensitivity_ema_carries_across_qat_stage_boundary` — the carry guard (FAILS if the
  EMA is lost at the QAT→QAT boundary; Class-2-fake-proof).
- `test_default_qat_path_carries_empty_ema_across_boundary` — the daemon-safety guard (the carry is empty on
  the non-score-aware-QAT path).

Probes (durable evidence artifacts):
- `experiments/probe_r6_integration_multistage.py` — the whole-system multi-QAT-stage integration probe
  (struct + [1] finite/no-divergence + [2] deployed-archive valid/inflatable + [3] the seam measurement).
- `experiments/probe_r6_qat_ema_carry_ab.py` — the A/B that MEASURED the seam magnitude (reset vs carry).

All ruff-clean.

## Fix committed this round

- `torch_vehicle: carry Lever-4 score-aware-QAT sensitivity EMA across stage boundaries` (driver.py 3-part
  carry + 2 new tests; sha in the session report). Byte-identity of the default/daemon path preserved (the
  9 guards pass); the fix is additive on the Lever-4-ON path only.

## Wire-in / provenance

6-hook (Catalog #125): all N/A — this is a review-round memo + an integration-seam fix + 2 regression tests
+ 2 $0 evidence probes (no new score-claim surface; the levers' own hooks are in the landing memo). Mission
contribution: `frontier_protecting` (closes a seam that would degrade the multi-day all-5-on descent at
every internal QAT boundary; the END remains a lower exact score, frontier UNMOVED `0.19109982`). Authority:
all numbers `[macOS-CPU advisory]` synthetic-scorer NON-PROMOTABLE. No GPU launched, no daemon touched (pid
33911 ALIVE 6h38m+ + untouched), no Cool-Chic touched.

**VERDICT: NOT-CLEAN (1 MEDIUM found + FIXED) → counter RESETS 2/3 → 0/3.** The 5 Layer-2 levers are **NOT
cleared** for the from-scratch all-levers treatment — a fresh 3-consecutive-clean-pass run (R7/R8/R9) is
required first. R7 starts from this fixed code. The next lens should be a SEVENTH distinct surface (e.g. a
real-scorer paired smoke on a tiny real-video slice to close the synthetic-scorer gap; or the
codec-grammar parse-back under the carried-EMA score-aware grid at a maximally-coarse operating point) —
NOT a re-run of the now-fixed multi-stage seam.
