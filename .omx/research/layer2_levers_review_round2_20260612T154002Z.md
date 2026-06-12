# Recursive adversarial review — ROUND 2 of the 5 Layer-2 levers (2026-06-12)

**Reviewer:** R2 subagent (author ≠ reviewer). R1 (`layer2_levers_independent_audit_20260612T151829Z.md`,
commit `4cbd9676a`) verified all 5 levers are REAL + no HIGH + daemon-safe + 97 tests pass with a **NO-FAKE
lens**. R2 has a **DIFFERENT lens: runtime/phase/edge-case FRAGILITY** — the bug class that survives a static
NO-FAKE audit and only bites mid-run (the protocol exists because of auto-kill-at-epoch-200 +
OOM-fix-silently-bypassed).
**Scope:** VERIFY + TEST; small fixes allowed. Did NOT touch `src/tac/substrates/cool_chic/**` (Track B),
the basin daemon (pid 33911), or its out-dir. Daemon confirmed ALIVE + untouched (4h+ elapsed, default config).
**Authority:** every in-loop / synthetic number here is `[macOS-CPU advisory]` NON-PROMOTABLE; the levers
land MEANS, the exact frontier is UNMOVED (`0.19109982`). Mission contribution: `frontier_protecting`
(closes a resume-fidelity defect that would corrupt a multi-day Lever-4-ON run).

## CLEAN-PASS VERDICT: **NOT-CLEAN → counter STAYS 0/3.**

R2 found ONE MEDIUM runtime/resume finding (Lever-4 sensitivity EMA not persisted). Per the protocol
("the counter resets to 0 whenever a round finds any issue"), R2 is NOT a clean pass. The finding was
FIXED + committed (`6e0d8feff`) this round per the protocol's "fix immediately" rule, so **R3 starts from
fixed code** and is the next chance to begin the clean-pass count.

---

## Findings by severity

### MEDIUM-R2-1 (FIXED this round) — Lever-4 score-aware-QAT sensitivity EMA is NOT persisted across resume

**Scenario (lens C — resume/checkpoint).** `_StageRuntime.tensor_sensitivity_ema` (`driver.py:339`,
`default_factory=dict`) is the per-tensor `s_t = ‖∂S/∂w_t‖` EMA that drives Lever-4's per-tensor INT8 grid
(`per_tensor_levels_from_sensitivity`). On a multi-day Lever-4-ON (`--levers all`) run that checkpoints
mid-stage and resumes (the exact case the "LONG RESUMABLE SATURATION SWEEPS" + "Durable detached daemons"
non-negotiables target), the EMA was **silently reset to empty** because:
- `checkpoint.save_checkpoint`'s blob had no `tensor_sensitivity_ema` key (`checkpoint.py:97-108`, pre-fix);
- `driver._capture_state` did not include it (`driver.py:791-811`, pre-fix);
- `driver._restore_into` did not restore it (`driver.py:813-840`, pre-fix);
- `_build_stage_runtime` rebuilds it fresh-empty every stage.

**Consequence:** post-resume, `apply_score_aware_qat(decoder, sens=None-because-empty)` falls back
bit-identically to **uniform-127** for the steps until `accumulate_tensor_sensitivity` re-seeds the EMA
(~hundreds of steps at decay=0.99) → the descent trajectory is **NOT bit-identical** to the uninterrupted
run when Lever-4 is on. This is the "sec/epoch/eval-row reset" class the protocol item C names. It is
**MEDIUM, not HIGH**: not a crash, not byte-corruption, not silent-wrong-output; self-healing; and the
**DEFAULT path (Lever-4 OFF) is unaffected** — the basin daemon (pid 33911: vendored CE + no-FiLM) has an
empty EMA always, so its resume is unchanged (proven below).

**Empirical proof of the defect (pre-fix probe, $0 CPU):** trained 5 epochs Lever-4-ON → live EMA = 14
non-trivial tensors; `_capture_state` blob had `tensor_sensitivity_ema` key = **False**; post-`_restore_into`
EMA = **0 tensors**. Demonstrated reset.

**FIX (committed `6e0d8feff`):** thread the EMA (a plain `dict[str,float]`) through
`save_checkpoint` blob (`checkpoint.py`), `_capture_state` (`dict(rt.tensor_sensitivity_ema)`), and
`_restore_into` (`if sens: rt.tensor_sensitivity_ema.update(...)`). **Backward-compatible + daemon-safe:**
default path → empty dict → round-trips empty → `if sens:` falsy → no-op (exactly today's behavior); legacy
checkpoints (no key) → `merged.get(...)` is None → no-op.

**Regression tests added (`test_driver_resume.py`):**
1. `test_resume_bit_identical_through_score_aware_qat_stage` — kill@3 / resume / finish a 6-epoch
   score-aware-QAT stage; final decoder+latents+EMA+AdamW state must equal the uninterrupted reference at
   **atol=0**. Requires the EMA to survive (else the post-resume quant grid diverges).
2. `test_score_aware_qat_sensitivity_ema_round_trips_through_checkpoint` — direct STATE assertion: the
   persisted EMA is non-empty + finite, and `_restore_into` restores it verbatim into a fresh runtime.

**Proven to GUARD the bug (not a constant-test):** with the restore logic temporarily neutered, BOTH new
tests FAIL (`assert {} == {'blocks.0': ...}`); with the fix, both PASS. Class-2-fake-proof.

### Lens A — NaN/inf under a LONG compose-all-five run: **CLEAN (no finding).**

Ran an **80-epoch** synthetic compose-all-five (`--levers all` equivalent: seg surrogate + T-anneal 1.0→0.05
+ rate surrogate on FiLM+vendored weights + score-aware-QAT EMA every epoch + margin-weight + C1a + use_qat),
asserting finiteness of loss, pose, **live decoder weights, EMA shadow, latents, AND the sensitivity EMA**
EVERY epoch. Result: **ALL 80 EPOCHS FINITE.** Specific sub-checks the static read predicted and the run
confirmed:
- **Anneal T→0.05 (`softmax(pred/0.05)` = `softmax(pred·20)`):** `F.softmax` is internally max-stable → no
  overflow; output stays a valid simplex; soft-cosine stays in [0,1]. T_min hit 0.05 cleanly.
- **Zero-init FiLM fc2 feeding the rate soft-hist (R1's LOW-2):** `conditional_weight_entropy` skips tensors
  with `w.abs().max() < max_abs_floor=1e-12` (`rate_surrogate.py:152-154`), so an exactly-zero fc2 is
  skipped; a near-zero fc2 normalizes to the grid and the bounded-exponent soft-assign + `eps=1e-12`
  denominators keep it finite. Confirmed no NaN/inf. **LOW-2 closed.**
- **QAT sensitivity EMA when `‖∂S/∂w‖→0`:** `accumulate_tensor_sensitivity` uses `g.norm().item()` → 0.0
  (finite); `_rank_normalize` routes all-equal/zero to the uniform 0.5 → base-127 fallback;
  `_fake_quantize_n` uses `scale = ma/n if ma>0 else 1.0` → no div-by-zero; `min_abs_levels=16` floors the
  grid. No inf path.

### Lens B — phase interactions across the 8 stages: **CLEAN.**

- **Anneal restarts per stage (correct by design):** `seg_temperature_for_epoch(spec, epoch_in_stage)` uses
  the WITHIN-stage `epoch_in_stage` and `spec.epochs` — the cosine is per-stage (matches the design intent;
  each stage anneals its own surrogate temperature). The driver passes the 0-based within-stage `epoch`
  (`driver.py:1220-1225`). No global/per-stage confusion.
- **`--levers all` enables QAT-stage levers on every stage ≥ fork (correct):** `score_aware_qat=True` rides
  along on non-QAT stages but `_train_one_epoch` only enters the QAT block under `spec.use_qat` (stage 4),
  so it is a **no-op (byte-identical) on non-QAT stages** — confirmed in code + the launch docstring.
- **No un-phase-gated threshold:** the levers are coefficient-scaled (rate_lambda, margin τ, anneal
  endpoints), not threshold-compared against phase-varying metrics. The one phase-varying quantity (the
  sensitivity EMA) is now correctly carried across the QAT stage boundary by the fix.

### Lens C — resume/checkpoint (the other four levers): **CLEAN (only Lever-4 had the gap).**

- **Lever-2 anneal:** stateless per-epoch function of `epoch_in_stage`; resume sets
  `start_epoch = resume_pos.epoch_in_stage` (`driver.py:1168`) → the anneal continues from the correct epoch.
  No persistent anneal state to lose.
- **Lever-3 stored_pose + FiLM weights:** the `stored_pose` buffer + `pose_film.*` params live in the
  decoder/EMA `state_dict`, which IS checkpointed (`decoder`/`ema_decoder` blobs) → survive resume.
- **Lever-1 rate surrogate:** stateless (reads live weights/latents each batch) — nothing to persist.
- **Lever-4:** the ONLY stateful lever; **was** the gap; now fixed + tested.

### Lens D — default-override antipattern: **CLEAN.**

`_resolve_lever_overrides` (`launch_l2_combined_attacks.py:202-235`) with `--levers all` genuinely sets every
lever ON (`seg_surrogate=soft_cosine`, `seg_temperature_end=0.05`, `rate_lambda_w=1e-3`,
`rate_lambda_lat=1e-3`, `score_aware_qat=True`, `margin_weight_tau=2.0`; pose-FiLM via `cfg`).
`_build_combined_curriculum` applies `replace(spec, **lever_overrides)` to the active+later stages. The
self-test asserts `passed = ... and rate_active and anneal_active and qat_active and margin_active` when
`all_on`. No lever is silently dead on the real launch path. **Verified by running `--levers all --self-test`
→ all five reported active = true, PASS.** (Note: `_spec_from_stage_config` defaults every lever OFF, so the
vendored curriculum / basin daemon is fully inert — correct.)

### Lens E — edge cases: **CLEAN.**

- `seg_temperature_end == seg_temperature` → cosine returns the constant start temp; `lo==hi` clamp safe.
- `rate_lambda` large → bounded entropy term (≤ log2(255) bits), degrades smoothly, no NaN.
- QAT uniform-sensitivity tensor → `_rank_normalize` all-0.5 → base-127 fallback (bit-identical), verified.
- The 80-epoch run used `n_pairs=8`, `batch_size=4` (a partial last batch of 0 — `range(0,8,4)` is clean;
  the `max(nb,1)` guard covers an empty-epoch edge). No batch-size-1 crash path in the lever code.

### Lens F — carry-forward MEDIUM pre-A/B probes: **noted as A/B criteria (not run; out of R2 time scope).**

R1's MEDIUM-1 (Lever-1 scan-order proxy vs real brotli byte stream) and MEDIUM-2 (Lever-4 indirect-effect)
remain open as **A/B gating criteria** — they are train/deploy proxy gaps, NOT runtime fragility, so they are
outside R2's lens and correctly deferred to the paired A/B (the empirical bit-spend proof per Catalog #304).
No new runtime risk from either.

---

## Resume-safety verdict (the multi-day from-scratch run)

**SAFE after the fix.** Post-`6e0d8feff`, ALL FOUR mutable lever-relevant states round-trip through a
death/resume bit-identically: decoder/latents/EMA shadow + AdamW/Muon momentum + LR-scheduler + RNG (already
covered by `test_driver_resume.py`) AND now the Lever-4 score-sensitivity EMA. A multi-day `--levers all`
run can checkpoint/resume without lever-state corruption. The DEFAULT path (and the live basin daemon) is
byte-identical — confirmed by `test_default_train_epoch_matches_vendored_only_reference` +
`test_driver_pose_film_off_builds_byte_identical_vendored_archive` (both still pass).

## Test-run result

```
.venv/bin/python -m pytest src/tac/torch_vehicle/tests/ src/tac/tests/test_rate_surrogate.py -q
→ 99 passed in 138.6s   (97 R1 baseline + 2 new R2 regression tests, 0 failures)
```
Byte-identity/default subset (8 tests incl. the daemon-resume-safety proofs): **8 passed**.
Pre-fix proof: the 2 new tests **FAIL** with the restore neutered (genuine guards, not constant-tests).

## LOW-1 doc-fix note (R1's finding, carried forward — trivial)

R1's LOW-1 stands: the rate surrogate runs **per-batch** (called from `_weight_regularizers` inside the
per-batch loop, `driver.py:611/626`), NOT "once per epoch" as the design/landing memos say. Harmless
(C1a is also per-batch; default-OFF) — a wording nit only. No code change; flagging for the landing memo.

## Fixes committed this round

- **`6e0d8feff`** — `torch_vehicle: persist Lever-4 score-aware-QAT sensitivity EMA through checkpoint/resume`
  (3 files, +94 lines: `checkpoint.py` blob key, `driver.py` capture+restore, `test_driver_resume.py` 2 tests).

## Wire-in / provenance

6-hook (Catalog #125): all N/A — this is a review-round memo + a resume-fidelity bug fix (no new score-claim
surface). The fix is `frontier_protecting` (prevents a corrupt multi-day Lever-4 trajectory). Authority: all
numbers `[macOS-CPU advisory]` NON-PROMOTABLE. No GPU launched, no daemon touched, no Cool-Chic touched.
