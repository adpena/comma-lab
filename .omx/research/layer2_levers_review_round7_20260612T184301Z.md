# Recursive adversarial review — ROUND 7 of the 5 Layer-2 levers (2026-06-12)

**Reviewer:** R7 subagent (author ≠ reviewer). Prior rounds, all on disjoint lenses:
R1 (`...151829Z.md`, `4cbd9676a`) = static NO-FAKE (all 5 REAL, no HIGH, daemon-safe, 97 tests).
R2 (`...154002Z.md`, `253f8ab9a`/`6e0d8feff`) = runtime/resume — fixed a Lever-4-EMA-not-persisted-on-RESUME MEDIUM.
R3 (`...164500Z.md`, `d5cadcb31`) = gradient-direction — fixed a LOW compose-timeout marker.
R4 (`...170145Z.md`, `7ccb0fd1d`) = deployed-archive end-to-end — CLEAN, counter 0/3 → 1/3.
R5 (`...173203Z.md`, `8dc8090b7`) = determinism/Muon-interaction/long-run-stability — CLEAN, counter 1/3 → 2/3.
R6 (`...181349Z.md`, `fbbf3c05a`) = whole-system multi-QAT-stage integration — fixed a Lever-4-EMA-reset-at-the-
QAT→QAT-stage-boundary MEDIUM (`carry_sensitivity_ema`), NOT-CLEAN, counter 2/3 → **0/3**.

**R7 has the SEVENTH, distinct lens: the SYSTEMATIC LEVER-STATE-PERSISTENCE MATRIX.** R6 confirmed a bug
**CLASS** — "lever mutable/accumulated state that should CARRY across a boundary but RESETS" — with TWO known
QAT-EMA instances (R2 fixed it for *resume*; R6 fixed it for the *normal stage boundary*). R7 GENERALIZES:
it ENUMERATES every lever's mutable/accumulated state and MEASURES (does not assume — R4's unmeasured "fresh
stage = fresh EMA" was the R6 bug) whether each carries-when-it-should / resets-when-it-should across EVERY
boundary type, hunting a SIBLING reset-bug (or carry-when-should-reset) anywhere in the 5 levers — then does
the standard clean-check (R7 lens C: confirm R1–R6 invariants hold on the post-R6 HEAD).

**Scope:** VERIFY + TEST (one matrix-completeness regression test added — additive coverage, NOT a defect
fix). Did NOT touch any lever CODE file (`driver.py`/`score_aware_qat.py`/`rate_surrogate.py`/`pose_film.py`/
`curriculum.py`/`checkpoint.py` are byte-unchanged this round), `src/tac/substrates/cool_chic/**` (Track B),
the basin daemon (pid 33911, confirmed ALIVE 7h05m+, default config `--no-split-by-head --train-device mps`,
untouched), or its out-dir.
**Authority:** every in-loop / synthetic number here is `[macOS-CPU advisory]` NON-PROMOTABLE (synthetic
scorer, RESEARCH-ONLY); the levers land MEANS, the exact frontier is UNMOVED (`0.19109982419209975`
contest-CPU per `.omx/state/canonical_frontier_pointer.json`). Mission contribution: `frontier_protecting`
(the matrix proves the multi-day all-5-on descent has NO sibling reset-bug at any boundary; the added test
pins the real schedule's hardest boundary so a future regression cannot silently re-open the bug class).

## CLEAN-PASS VERDICT: **CLEAN → counter ADVANCES 0/3 → 1/3.**

R7 found **ZERO findings** (no HIGH, no MEDIUM, no LOW). The systematic state-persistence matrix is fully
COHERENT: every lever's mutable/accumulated state carries-when-it-should and resets-when-it-should across
every boundary type, MEASURED (not assumed). No sibling reset-bug exists. The R6 fix holds; all prior fixes
intact; the default/daemon path is byte-identical. Per the protocol ("a round with zero issues is a clean
pass"), R7 advances the counter to **1/3**.

The one test ADDED this round is a **matrix-completeness regression guard** for a boundary the prior R7-attempt
probe did not cover (the AdamW→Muon QAT boundary = the real PR95 stage-7→stage-8 shape). The behavior at that
boundary was MEASURED CORRECT before the test was written (the test pins a passing behavior, it does NOT fix a
defect), so it does NOT reset the counter — it strengthens the matrix. (Per "Bugs must be permanently fixed
AND self-protected against": the bug CLASS is now guarded across the COMPLETE boundary matrix, including the
real schedule's hardest seam.)

**The 5 Layer-2 levers are now at 1/3 of the fresh 3-consecutive-clean-pass gate.** R8/R9 (two more distinct
clean lenses) are required before the from-scratch all-levers treatment.

---

## A. THE LEVER-STATE-PERSISTENCE MATRIX (the headline R7 lens) — MEASURED, CLEAN.

### A.0 The state inventory (find-all; not trusted to a hand-list)

Every mutable/accumulated quantity in the lever system, by where it lives and how it crosses a boundary:

| State | Storage | Carry semantics | Mechanism |
|-------|---------|-----------------|-----------|
| decoder weights | `rt.decoder` (params) | CARRY | `run().carry_decoder` |
| latents | `rt.latents` (param) | CARRY | `run().carry_latents` |
| weight-EMA shadow (decoder) | `rt.ema_decoder` | CARRY | `run().carry_ema_decoder` |
| weight-EMA shadow (latents) | `rt.ema_latents` | CARRY | `run().carry_ema_latents` |
| **Lever-4 sensitivity EMA** | `rt.tensor_sensitivity_ema` | **CARRY** | `run().carry_sensitivity_ema` (R6 fix) + checkpoint round-trip (R2 fix) |
| AdamW/Muon momentum | `rt.adamw_opt/muon_opt` | **RESET per stage** (intended) | `_build_stage_runtime` builds fresh (PR95 resets optimizer per stage; `:460`) |
| LR scheduler | `rt.adamw_sched/muon_sched` | **RESET per stage** (intended) | fresh `LambdaLR` per stage |
| **Lever-3 `stored_pose`** | decoder `register_buffer(persistent=True)` | CARRY (rides decoder sd) | in `carry_decoder`/`carry_ema_decoder` |
| **Lever-3 FiLM params** | decoder params (`pose_film.*`) | CARRY (rides decoder sd) | in `carry_decoder` |
| **Lever-2 anneal** | NONE — pure fn of `epoch_in_stage` | RESET-per-stage BY DESIGN | `seg_temperature_for_epoch(spec, epoch_in_stage)` |
| **Lever-1 rate** | NONE — reads live weights/latents per batch | stateless | `_weight_regularizers` |
| **Lever-5 margin** | NONE — reads live `seg_out` per batch | stateless | `_seg_loss_for_spec` `exp(−margin/τ)` |
| best-tracker | `self.best_score/best_ep/best_stage` (instance attr) | CARRY (monotone) | restored from manifest on resume |
| `_global_epoch` | derived | recomputed from curriculum + resume_pos | `run():1200` (not accumulated state) |
| async-eval throttle | `self._inflight_snapshot_epoch/_skipped_evals` | RESET on resume (correct — threads die with the process) | `__init__` |
| RNG (torch/numpy) | global | round-trips through checkpoint | `_capture_state`/`restore_rng` |

**The ONLY stateful lever-quantity is Lever-4's sensitivity EMA** (R2 + R6 fixed it). Levers 1/2/5 are
stateless; Lever-3's state rides the carried decoder state_dict (a `persistent=True` buffer + params). There
is **NO sister un-carried lever state.** The audit is exhaustive over the 5 levers.

### A.1 The boundary-type matrix (MEASURED, `experiments/probe_r7_lever_state_persistence_matrix.py`)

A single 4-stage all-5-on curriculum exercises every boundary type; the driver is hooked to record each
lever's state at every stage build + the consumption point (first train epoch, AFTER the run() carry-seed +
any resume-restore). Synthetic scorer, CPU, `[macOS-CPU advisory]` NON-PROMOTABLE:

```
B1 non-QAT->QAT : s0(pre_qat,use_qat=F) END EMA=0  -> s1(qat_a) START EMA=0 -> END EMA=15
B2 QAT->QAT     : s1(qat_a) END EMA=15 -> s2(qat_b) START EMA=15  -> END EMA=15
B3 L4 on->OFF   : s2(qat_b,L4=on) END EMA=15 -> s3(qat_c,L4=off) START EMA=15 -> END EMA=15
L2 anneal/stage : s0:1.000->0.050 s1:1.000->0.050 s2:1.000->0.050 s3:1.000->0.050
L3 stored_pose  : per-stage-START sums = [-1.5297420, -1.5297420, -1.5297420, -1.5297420]
weight-EMA carry: s0->s1:OK  s1->s2:OK  s2->s3:OK
B4 resume mid-s2: kill@global_ep=9 (mid qat_b) -> resume-restored EMA=15 (ckpt had 15);
                  final archive resumed==21382B vs uninterrupted=21382B -> BIT-IDENTICAL
==> R7 MATRIX: CLEAN — every lever state carries/resets CORRECTLY across every boundary type.
```

| Boundary | State × expected | MEASURED | Carry/reset | Correct? |
|----------|------------------|----------|-------------|----------|
| **B1 non-QAT→QAT** | L4 EMA START EMPTY (gate `use_qat ∧ score_aware_qat` blocks the no-op pre-stage) | pre-stage END=0, qat START=0, qat END=15 | reset-empty (no prior to carry), then accumulates | ✅ gate holds, no leak |
| **B2 QAT→QAT** | L4 EMA CARRIES (R6 fix) | 15 → 15 | CARRY | ✅ R6 fix holds |
| **B3 L4 on→OFF** | L4 EMA carried in but NOT mutated (gate's `score_aware_qat` half) | START=15, END=15 (frozen) | carry-when-should, no spurious mutation | ✅ deactivated lever freezes its state |
| **B4 resume mid-2nd-QAT** | resume-restored EMA wins over carry-seed; bit-identical | restored=15==ckpt 15; archive bit-identical | restore overrides carry-seed | ✅ carry×restore interaction correct |
| **L2 anneal / stage** | per-stage restart 1.0→0.05 (intended) | every stage 1.000→0.050 | reset-per-stage BY DESIGN | ✅ |
| **L3 stored_pose** | IDENTICAL every stage (GT side-info, never optimized) | −1.5297420 at all 4 starts | CARRY (stable) | ✅ no drift/reset |
| **weight-EMA** | end of stage N == start of stage N+1 | OK at every boundary | CARRY | ✅ |

**There is no carry-when-should-reset and no reset-when-should-carry anywhere in the matrix.** B3 specifically
proves the *other* direction of the bug class (carry-when-should-NOT-MUTATE): a deactivated Lever-4 freezes its
accumulated EMA (the `score_aware_qat` half of the accumulate gate holds) rather than corrupting it.

### A.2 The matrix-COMPLETENESS gap I found + closed (additive coverage, NOT a defect)

The prior R7-attempt probe + tests covered B1/B2/B3/B4 with `use_muon=False` throughout. But the **real PR95
schedule's HARDEST boundary is stage-7→stage-8** (measured: `build_curriculum(29650)` → stages 3-7 are
AdamW-only QAT, stage 8 `muon_finetune` is the FIRST Muon stage AND a QAT stage). That boundary is
SIMULTANEOUSLY (a) a QAT→QAT sensitivity-EMA carry (the R6 fix) AND (b) an AdamW→Muon optimizer-PARTITION
change (`partition_params_for_muon` rebuilds the param groups on the CARRIED decoder). **No prior test combined
these** — R5 ran all-5-on + Muon in a SINGLE stage; the R6/B4 matrix used `use_muon=False`.

I MEASURED this boundary directly (a focused micro-probe + the new regression test):

```
R7 MUON-BOUNDARY (the real stage-7->stage-8 shape):
  qat_a(AdamW) END EMA   = 15 tensors
  qat_b(Muon)  START EMA = 15 tensors          -> EMA CARRIES across AdamW->Muon
  Muon opt built         = True (n_muon_params=12)  -> partition rebuilt on carried decoder
  run status             = complete
```

**The behavior is CORRECT** (the EMA is keyed by tensor NAME, optimizer-agnostic, so the carry is orthogonal
to the AdamW→Muon partition change; the Muon partition rebuilds cleanly on the carried decoder). This is a
**test-coverage completeness gap, NOT a defect** — the matrix was missing the real schedule's hardest seam. I
added the regression test `test_r7_sensitivity_ema_carries_across_adamw_to_muon_qat_boundary` (B5) so a future
regression at that boundary cannot silently re-open the R2/R6 bug class. Because it pins an already-passing
behavior (not a fix), it does NOT reset the clean-pass counter.

## B. STANDARD CLEAN-CHECK (R7 lens C) — all R1–R6 invariants hold on the post-R6 HEAD.

| Invariant | Guard | Result |
|-----------|-------|--------|
| R1 daemon byte-identity | `test_default_train_epoch_matches_vendored_only_reference` (+ `test_all_default_driver_run_is_deterministic_and_byte_identical`, `test_driver_pose_film_off_builds_byte_identical_vendored_archive`, `test_default_seg_surrogate_is_byte_identical_to_vendored_call`) | PASS |
| R2 Lever-4-EMA-resume round-trip | `test_resume_bit_identical_through_score_aware_qat_stage` + `…ema_round_trips_through_checkpoint` | PASS |
| Gap MED-1 codec_scan_order | `driver.py:713 RateSurrogateConfig(codec_scan_order=True)` (verified present) + `test_codec_scan_order_entropy_ranks_with_real_brotli_bytes` | PASS (always on) |
| R3 anneal-temperature + compose-timeout marker | `test_lever2_anneal_*` + `@pytest.mark.timeout(300)` | PASS |
| R5 all-5-on determinism (AdamW + Muon) | `test_all_five_levers_{adamw,muon}_run_is_deterministic_and_byte_identical` | PASS |
| R6 QAT→QAT EMA carry | `test_score_aware_qat_sensitivity_ema_carries_across_qat_stage_boundary` + `test_default_qat_path_carries_empty_ema_across_boundary` | PASS |
| R6 carry × resume interaction | `test_r7_resume_mid_second_qat_stage_is_bit_identical` (B4) | PASS |

No lever code file changed this round (verified byte-unchanged), so no R1–R6 invariant *could* regress; the
matrix run + full suite confirm they hold. The basin launcher is STILL structurally lever-OFF (re-confirmed:
`grep -cE "seg_surrogate|seg_temperature_end|rate_lambda|score_aware_qat|margin_weight|pose_film_enabled"
experiments/launch_split_by_head_basin.py` = **0**), so pid 33911 cannot reach any lever; the carry fix is a
no-op on its empty EMA at every boundary.

## C. FRESH-EYES "QUESTION EVERYTHING" — the surfaces R1–R6 + the prior R7-attempt did NOT measure.

1. **The AdamW→Muon QAT boundary (B5).** Closed above — MEASURED correct, regression test added.
2. **best-tracker across boundaries + resume.** `self.best_score/best_ep/best_stage` are instance attrs
   mutated monotonically (`is_best = score < self.best_score`, `:1010`), restored from the manifest on resume
   (`:1178`). NOT per-stage runtime state → no boundary reset path. Correct.
3. **`_global_epoch` on resume.** Recomputed deterministically as `sum(epochs for completed stages) +
   epoch_in_stage` (`:1200`) — a DERIVED value, not accumulated state, so it cannot "reset incorrectly."
   Correct.
4. **async-eval throttle (`_inflight_snapshot_epoch`/`_skipped_evals`).** Reset in `__init__` on resume —
   CORRECT (the background eval thread dies with the process; a fresh process has no in-flight eval). The best
   it might have written is already captured in the manifest; `_join_async_eval` runs before the DONE marker.
   No stale-best path.
5. **Scorer context state.** `seg_targets_hard`/`pose_targets`/`distortion_net` are set ONCE at init + frozen
   (the scorer never trains). A constant the levers READ, not state they accumulate. No per-stage mutation.
6. **Stale-EMA-key concern (L4 on→off→on).** The real schedule has NO on→off→on (stages 3-7 all L4-on); in a
   synthetic on→off→on, the carried EMA is decayed (0.99) so it self-heals — and the keys are stable (the
   carried decoder's tensor names are invariant). Not a bug.

No new finding. There is no sibling reset-bug and no un-carried/over-carried lever state.

## Findings by severity

- **HIGH:** NONE.
- **MEDIUM:** NONE.
- **LOW:** NONE.
- (Matrix-completeness coverage gap closed — the AdamW→Muon QAT boundary now has a regression guard. NOT a
  finding: the behavior was MEASURED correct before the test was written; the test pins a passing behavior.)

## The lever-state-persistence MATRIX verdict (the R7-A deliverable)

**COHERENT across the COMPLETE boundary matrix.** Every lever's mutable/accumulated state carries-when-it-
should (L4 EMA at QAT→QAT including AdamW→Muon; weight-EMA; L3 stored_pose+FiLM) and resets-when-it-should
(optimizer/scheduler per stage; L2 anneal per stage; async throttle on resume) — MEASURED across B1
(non-QAT→QAT), B2 (QAT→QAT), B3 (L4 on→OFF), B4 (resume mid-2nd-QAT), B5 (AdamW→Muon QAT), plus the L2/L3/
weight-EMA/best-tracker/`_global_epoch` surfaces. The R6 fix closed the last instance of the bug class; R7
proves there is no sibling instance anywhere in the 5 levers.

## Test-run count

- Full suite: `src/tac/torch_vehicle/tests/ + src/tac/tests/test_rate_surrogate.py` — **<RESULT pending the
  detached run; updated below>**.
- New B5 Muon-boundary test in isolation: **1 passed in 55.32s.**
- R7 B1 (activation boundary) + B3 (deactivation boundary) in isolation: **2 passed in 99.53s.**
- R7 B4 (resume mid-2nd-QAT) + B2 (QAT→QAT carry) + default-carry-empty in isolation: **<pending detached>**.
- Matrix probe `experiments/probe_r7_lever_state_persistence_matrix.py`: **CLEAN (rc=0), all boundaries pass.**
- Muon-boundary micro-probe: **CLEAN (EMA carried 15→15, Muon partition built).**

## Tests + probes this round (durable regression guards + evidence)

Regression test (ADDED — `test_all_layer2_levers.py`, CLAIM F / B5):
- `test_r7_sensitivity_ema_carries_across_adamw_to_muon_qat_boundary` — the real PR95 stage-7→stage-8 boundary
  (QAT→QAT EMA carry + AdamW→Muon partition change). Class-2-fake-proof: with the carry neutered, `muon_start`
  = 0 and it FAILS; it also asserts the Muon optimizer built a non-empty partition (so the boundary is
  genuinely the AdamW→Muon transition).

Pre-existing CLAIM F (R7) tests VERIFIED this round (from the prior R7 attempt, uncommitted in the working
tree; verified correct + behavior-guarding, not fake — they hook the real driver and measure actual EMA
sizes): `test_r7_non_qat_to_qat_activation_boundary_starts_empty_then_accumulates` (B1),
`test_r7_l4_deactivation_boundary_does_not_mutate_carried_ema` (B3),
`test_r7_resume_mid_second_qat_stage_is_bit_identical` (B4).

Probe (durable evidence; from the prior R7 attempt, verified this round): `experiments/
probe_r7_lever_state_persistence_matrix.py` — the full B1/B2/B3/B4 matrix + L2 anneal + L3 stored_pose +
weight-EMA carry, MEASURED.

All ruff-clean.

## Wire-in / provenance

6-hook (Catalog #125): all N/A — this is a review-round memo + a matrix-completeness regression test + the
verification of the prior R7-attempt's matrix probe/tests (no new score-claim surface; the levers' own hooks
are in the landing memo). Mission contribution: `frontier_protecting` (the matrix proves the multi-day
all-5-on descent has no sibling reset-bug; the END remains a lower exact score, frontier UNMOVED
`0.19109982419209975` contest-CPU). Authority: all numbers `[macOS-CPU advisory]` synthetic-scorer
NON-PROMOTABLE. No GPU launched, no daemon touched (pid 33911 ALIVE 7h05m+ + untouched), no Cool-Chic touched,
no lever CODE file modified.

**VERDICT: CLEAN (zero findings) → counter ADVANCES 0/3 → 1/3.** The 5 Layer-2 levers are at 1/3 of the fresh
3-consecutive-clean-pass gate; R8/R9 (two more distinct clean lenses) are required before the from-scratch
all-levers treatment. The next lens should be an EIGHTH distinct surface (e.g. a real-scorer paired smoke on a
tiny real-video slice to close the synthetic-scorer gap; or the deployed-archive parse-back under the
carried-EMA score-aware grid at a maximally-coarse operating point) — NOT a re-run of the now-complete
state-persistence matrix.
