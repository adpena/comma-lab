# Recursive adversarial review — ROUND 11 of the 5 Layer-2 levers (2026-06-12)

**Reviewer:** Partner-A (author ≠ reviewer). The SEAL requires 3 FRESH consecutive clean rounds.
Prior FRESH count: R9 CLEAN → 1/3; **R10 NOT-CLEAN** (1 LOW lever-code finding — the Lever-2 cold-anneal
seg-gradient floor — fixed via 2-landing) → **fresh count RESET to 0/3**. So R11 begins the fresh count
again: a clean R11 advances 0/3 → 1/3.

**R11 has the ELEVENTH, distinct lens: RESUME × ANNEAL-SCHEDULE RESTORE.** The live distortion arm runs an
async-eval daemon with the seg-temperature anneal active (1.0 → 0.05). If the daemon dies and resumes
mid-stage, does `seg_temperature_for_epoch` reconstruct the EXACT annealed temperature for the resumed
epoch (so the resumed loss is bit-identical to the no-death run)? And do the pose-FiLM state + Lever-4
sensitivity EMA + Muon partition ALL restore correctly across a resume at an arbitrary mid-stage epoch
WHERE the anneal is at a non-trivial intermediate temperature? The prior B4 resume test (R7) covered
resume-mid-2nd-QAT but at a point where the anneal had reached the FLOOR — it did NOT exercise the
intermediate-anneal restore.

**Scope:** VERIFY + TEST (two additive resume×anneal regression tests — they PIN an already-correct behavior,
NOT a defect fix). Did NOT touch any lever CODE file (`driver.py` / `curriculum.py` / `score_aware_qat.py` /
`rate_surrogate.py` / `pose_film.py` / `scorer_context.py` / `checkpoint.py` are byte-unchanged this round —
the only edit is `tests/test_all_layer2_levers.py`). Did not touch Track B, the live distortion arm (out-dir
`experiments/results/distortion_arm_l235_20260612T205102Z`, confirmed ALIVE at global_epoch 517+), or its
out-dir.

**Authority:** every number here is `[macOS-CPU advisory]` NON-PROMOTABLE (synthetic scorer, RESEARCH-ONLY);
the levers land MEANS, the exact frontier is UNMOVED (`0.19109982419209975` contest-CPU). Mission
contribution: `frontier_protecting` (the resume×anneal bit-identity proof guards the live multi-day arm's
crash-resume path — a daemon that dies mid-anneal and resumes reproduces the identical descent, so a crash
costs no signal).

## CLEAN-PASS VERDICT: **CLEAN → fresh counter ADVANCES 0/3 → 1/3.**

R11 found **ZERO findings** (no HIGH, no MEDIUM, no LOW). A daemon crash MID an all-5-on anneal stage —
at a point where the annealed temperature is a NON-trivial intermediate value (not the start 1.0, not the
floor) — resumes to a BIT-IDENTICAL archive, MEASURED on both (a) a single-stage all-5-on anneal stage and
(b) the multi-stage AdamW→Muon stage-7→8 shape (the JOINT of resume-mid-QAT + AdamW→Muon partition rebuild
+ intermediate-anneal restore + L4-EMA restore + pose-FiLM restore). Per the protocol ("a round with zero
issues is a clean pass"), R11 advances the fresh counter to **1/3**.

The two tests ADDED this round PIN an already-correct behavior (MEASURED bit-identical before the tests were
written), so they do NOT reset the counter — they extend the resume-bit-identity matrix to the
intermediate-anneal case the prior B4 test did not cover.

---

## A. THE RESUME × ANNEAL-SCHEDULE-RESTORE MATRIX (the headline R11 lens) — MEASURED, CLEAN.

### A.0 The mechanism (why the anneal restores exactly)

`seg_temperature_for_epoch(spec, epoch_in_stage)` is a PURE function of `(spec, epoch_in_stage)` — it
holds NO accumulated state. The annealed temperature is therefore RECONSTRUCTED from the restored
`epoch_in_stage`, never persisted. On resume, `resume_pos.epoch_in_stage` (loaded from the checkpoint
`position`) feeds the per-stage epoch loop, which calls `seg_temperature_for_epoch(spec, epoch_in_stage)`
with the SAME epoch index the uninterrupted run used at that point — so the temperature is identical.

### A.1 Single-stage all-5-on resume mid-anneal (the live arm's shape) — MEASURED bit-identical

A 6-epoch all-5-on stage (soft_cosine + anneal 1.0→0.2 + margin τ=2 + rate + QAT + score-aware + pose-FiLM),
killed at `epoch_in_stage=3`, resumed:

```
annealed temps over the stage: [1.0, 0.9236, 0.7236, 0.4764, 0.2764, 0.2]
kill at epoch_in_stage=3 -> annealed T there = 0.4764 (INTERMEDIATE: not 1.0, not 0.2)
checkpoint position: TorchCheckpointPosition(stage_index=0, epoch_in_stage=3)
ref = 21369 B   resumed = 21369 B   BIT-IDENTICAL = True
```

The resumed epoch reconstructs T=0.4764 (the intermediate annealed value), and the archive is
bit-identical. **The annealed temperature restores exactly across a mid-stage resume.**

### A.2 Multi-stage AdamW→Muon resume mid-anneal (the JOINT case) — MEASURED bit-identical

A 3-stage all-5-on curriculum (AdamW-QAT s0 → AdamW-QAT s1 → Muon-QAT s2 — the real PR95 stage-7→8 shape,
anneal per stage), killed MID the Muon stage s2 at a non-floor annealed temperature, resumed:

```
s2 annealed temps: [1.0, 0.7625, 0.2875, 0.05]   kill at s2 epoch 2 -> T = 0.2875 (intermediate)
ckpt position: TorchCheckpointPosition(stage_index=2, epoch_in_stage=2)   L4 EMA tensors: 15
ref = 21388 B   resumed = 21388 B   BIT-IDENTICAL = True
```

This is the JOINT of FIVE things crossing the resume: (1) resume-mid-QAT (B4), (2) the AdamW→Muon
partition rebuild on the carried decoder (B5), (3) the intermediate-anneal temperature restore (R11),
(4) the Lever-4 sensitivity EMA restore (15 tensors, R2+R6), (5) the pose-FiLM state restore (Lever-3).
ALL FIVE restore correctly — the resumed Muon-QAT anneal stage reproduces the bit-identical archive.

## B. STANDARD CLEAN-CHECK (R11 lens) — all R1–R10 invariants hold on the post-R10 HEAD.

The full lever suite (run detached, SIGURG-proof):

```
.venv/bin/python -m pytest src/tac/torch_vehicle/tests/test_all_layer2_levers.py -q --timeout=600
→ 88 passed in 717.72s (0:11:57)   (SUITE_EXIT=0)
```

No lever code file changed this round (the only edit is the test file), so no R1–R10 invariant *could*
regress from my change; the suite confirms they hold (R1 daemon byte-identity, R2/R6 EMA carry+resume,
R3 anneal, R5 all-5-on determinism, R7 boundary matrix B1–B5, R9 anneal-tail numerical stability, R10
gradient-floor guard + real-scorer composition). The live distortion arm is structurally untouched.

## C. FRESH-EYES "QUESTION EVERYTHING" — the resume×anneal surfaces R1–R10 did NOT measure.

1. **Does the annealed temperature restore on resume?** Yes — `seg_temperature_for_epoch` is pure, so the
   restored `epoch_in_stage` reconstructs the exact intermediate temperature (0.4764 / 0.2875 measured).
2. **Does the prior B4 test cover this?** No — B4 resumes mid-qat_b where the anneal (end=0.05 over 4
   epochs) has reached near the floor; R11 resumes at an INTERMEDIATE temperature (the case that would
   catch a temperature-restore regression).
3. **Does pose-FiLM state restore mid-anneal?** Yes — it rides the carried/restored decoder state_dict
   (the persistent `stored_pose` buffer + FiLM params); the bit-identical archive proves it.
4. **Does the AdamW→Muon partition rebuild interact with the anneal restore?** No bad interaction — the
   Muon partition rebuilds on the carried decoder, the L4 EMA restores (15 tensors), AND the anneal
   reconstructs its intermediate temperature; the JOINT is bit-identical.
5. **Could the temperature depend on accumulated state (a hidden-state regression)?** No — it is a pure
   function; the R11 tests would FAIL (different temperature → different archive) if a future change made
   it stateful. The intermediate-T assertion ensures the test exercises a non-trivial restore.

No new finding. The resume×anneal path is bit-identical across the single-stage and multi-stage Muon cases.

## Findings by severity

- **HIGH:** NONE.
- **MEDIUM:** NONE.
- **LOW:** NONE.
- (Coverage gap closed — the resume-mid-INTERMEDIATE-anneal case (single-stage + AdamW→Muon multi-stage)
  now has a bit-identity regression guard. NOT a finding: MEASURED bit-identical before the tests were
  written; the tests pin a passing behavior + assert the kill-epoch T is genuinely intermediate.)

## Test-run count

- New R11 resume×anneal tests (`r11_resume`): **2 passed in 133.74s.**
- Full lever suite: **88 passed in 717.72s (0:11:57), 0 failures** (all R1–R10 invariants + the 2 new R11 resume×anneal tests; the longer wall-clock vs R10 is daemon CPU contention, NOT a slowdown in the levers).
- R11 probes (single-stage + multi-stage Muon resume mid-anneal): both BIT-IDENTICAL (21369==21369,
  21388==21388).

## Tests this round (durable regression guards)

Added to `test_all_layer2_levers.py` (R11 lens):
- `test_r11_resume_mid_anneal_all5_is_bit_identical` — single-stage all-5-on, killed at an INTERMEDIATE
  annealed temperature (asserted strictly between floor and start), resumed bit-identical.
- `test_r11_resume_into_muon_stage_with_anneal_is_bit_identical` — the JOINT AdamW→Muon + resume-mid-QAT +
  intermediate-anneal + L4-EMA + pose-FiLM restore, resumed bit-identical.
- `_r11_all5_anneal_stage` helper — the live-arm-shaped all-5-on anneal stage.

All ruff-clean.

## Wire-in / provenance

6-hook (Catalog #125): all N/A — review-round memo + additive resume-bit-identity regression guards (no new
score-claim surface). Mission contribution: `frontier_protecting` (the resume×anneal bit-identity proof
guards the live multi-day arm's crash-resume path; a daemon death mid-anneal costs no signal; the END
remains a lower exact score, frontier UNMOVED `0.19109982419209975` contest-CPU). Authority: all numbers
`[macOS-CPU advisory]` synthetic NON-PROMOTABLE. No GPU launched, no daemon touched (distortion arm out-dir
separate + untouched), no Cool-Chic touched, no lever CODE file modified.

**VERDICT: CLEAN (zero findings) → fresh counter ADVANCES 0/3 → 1/3.** The resume×anneal-schedule restore is
bit-identical across the single-stage all-5-on case AND the multi-stage AdamW→Muon stage-7→8 shape (the
joint of partition rebuild + L4-EMA + intermediate-anneal + pose-FiLM restore). The 5 Layer-2 levers are
at **1/3 of the (post-R10-reset) fresh 3-consecutive-clean-pass gate** — TWO more distinct clean lenses
(R12, R13) are required to SEAL. ITEM A is therefore **NOT-SEALED** this session: R9 clean, R10 caught a
real LOW lever finding (reset), R11 clean — the honest fresh count is 1/3.
