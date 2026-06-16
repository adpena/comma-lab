# KD-warm-start actuator + FiLM rgb_0 decoupling refinement (P1a of the bind-all build-out)

**Date:** 2026-06-16T21:05:40Z · **Authority:** `[contest-CPU advisory]` — NO score claim
(this is a TRAIN-TIME priming primitive; the exact d_seg/d_pose that pick BEST still run the
full scorer on the CPU authority on the byte-closed archive; pointer 0.19110 UNMOVED).
**Spend:** $0, CPU/code only. No GPU touched; no training launched; no running job disturbed.

This lands P1a of the bind-all production build-out
(`.omx/research/production_readiness_bind_all_ingredients_20260616.md`): the wall-clock
resolution that lets the re-tapered (solved-taper) decoder inherit the converged basin's
knowledge feasibly, since the full PR95 29k-epoch from-scratch curriculum is ~weeks on MPS.
Every new field is DEFAULT-OFF → byte-identical to the live from-0 / warm-start paths.

## Part A — FiLM rgb_0 decoupling refinement (`cfg.pose_film_rgb0_pose_trainable`)

**The fix.** The base `pose_film_trunk_stopgrad` restores ALL non-FiLM params — INCLUDING
the `rgb_0` head — to their seg-only grad, which FREEZES `rgb_0` w.r.t. the pose objective.
But the contest SegNet reads ONLY frame-1 (`rgb_1`), so `∂d_seg/∂(rgb_0 params) = 0` exactly
(rgb_0 only writes f0, which SegNet never sees). The pose loss can therefore train `rgb_0`
(the pose-conditioned frame-0 head) with ZERO d_seg cost, giving the pose objective strictly
more capacity ({FiLM path + rgb_0}) to hold d_pose.

**Mechanism.** `_non_film_grad_params(decoder, latents)` (the seg-only-restore set used by
`_split_by_head_backward`) now EXCLUDES rgb_0's params when `pose_film_rgb0_pose_trainable`
is on. rgb_0 is resolved precisely via a new `_rgb0_param_ids(decoder)` helper that mirrors
the `_film_param_ids` id-set pattern — it resolves the actual `rgb_0` Module
(`decoder.rgb_0` on the FiLM wrapper, or `decoder.rgb_0` directly on a bare decoder) and
keys on its parameter `id()`s. Default-OFF keeps rgb_0 in the seg-only set, byte-identical to
the base trunk-stopgrad A/B. A `__post_init__` guard refuses the flag without
`pose_film_trunk_stopgrad=True` (it only exists as a refinement of that set — no silent
no-op).

**The proof (test, not assertion).**
`test_rgb0_refinement_preserves_dseg_decoupling_and_trains_rgb0`:
- ∂d_seg/∂(pose-objective) STILL = 0 — every shared param EXCEPT rgb_0 (trunk + skips +
  blocks + refine + **rgb_1** + latents) keeps the seg-only grad BIT-IDENTICAL (compared
  against a seg-only backward; the pose backward left zero residue there);
- rgb_0 NOW carries the pose gradient (its grad DIFFERS from seg-only — was frozen before).
`test_base_trunk_stopgrad_freezes_rgb0` is the behavioral contrast (flag OFF ⇒ rgb_0 IS
restored to seg-only = frozen), proving the refinement is not a no-op.

## Part B — KD-warm-start actuator (`cfg.kd_warm_start_dir`)

**The linchpin.** A re-taper changes the decoder channel shapes, so a strict-decoder
warm-start of the vendored basin FAILS. The resolution:
- **Latents: direct warm-start.** The basin latents `best_ema_latents.pt` are
  `(n_pairs, 28)` — taper-INDEPENDENT (only decoder channels change) — so they load DIRECTLY
  as the re-taper student's stage-0 init (`load_kd_warm_start_latents`).
- **Decoder: KD-distill.** A FROZEN teacher = the basin's vendored-taper decoder
  (`best_ema_decoder.pt`, built as the plain vendored `HNeRVDecoder` at the basin's
  base_ch/latent_dim, `eval()` + `requires_grad_(False)`, rendered under `no_grad`). The
  student = the re-tapered (solved-taper, possibly +FiLM-v2) decoder. A KD WARM-UP phase
  (first `kd_warm_epochs` of stage 0, default 300) minimizes the frame-MSE between the
  student pair frames and the teacher pair frames rendered on the SAME latents. After the
  warm-up the normal score-aware curriculum (oomph + FiLM + QAT + Muon + …) continues from
  the distilled student.

**Composition.** Built on the EXISTING config: `taper_channels` (the student is the solved
taper), `pose_film_enabled`/`version=2`/`pose_film_trunk_stopgrad`, the oomph curriculum,
pose cadence — the KD warm-up is a PREFIX phase; the rest of the curriculum is unchanged.
The `__post_init__` `taper requires pose_film_enabled=False` constraint was RELAXED to allow
`taper + pose_film_version==2` (the bind-all combo) — the v2 residual wrapper reads only the
inner decoder's public surface (`channels[-1]` / `stem` / `blocks` / `skips` / `ps` /
`refine` / `rgb_0` / `rgb_1`), all of which `ConfigurableTaperHNeRVDecoder` exposes; v1 stem-
injection stays refused on a re-taper (untested + couples d_pose into d_seg).

**NO-FAKE (the distillation actually runs).**
- `test_kd_step_lowers_frame_mse_toward_teacher`: a few KD epochs measurably lower the
  student-vs-teacher frame-MSE (>= 5% relative drop — not numerical noise; if the KD were
  fake the loss would not drop).
- `test_kd_teacher_unchanged_after_kd_step`: the frozen teacher's weights are bit-identical
  before/after a KD step (only the student trains).
- `test_kd_latents_load_directly_for_retaper`: the basin latents load into a student whose
  taper DIFFERS from the vendored taper (the strict-decoder load would fail — only KD
  carries the basin in).
- `test_kd_warm_up_then_curriculum_continues_e2e` / `test_kd_warm_up_emas_track_distilled_
  student`: the warm-up is a PREFIX then the score-aware curriculum runs to a DONE marker,
  and the trained EMA shadow differs from a fresh init (the distillation + curriculum moved
  the weights).
- `test_kd_resume_round_trip`: a kill mid-stage-0 (post-KD) resumes and matches an
  uninterrupted run's final EMA bit-for-bit — KD is correctly idempotent (re-run on a fresh
  start, SKIPPED on a resume that owns a checkpoint; the KD's own seeded CPU Generator does
  not perturb the global RNG stream the curriculum consumes).
- `test_kd_none_is_byte_identical_to_from0`: two from-0 runs with KD off are bit-identical
  (the machinery is a true no-op when off).

## Reused machinery (SEARCH-AND-FAMILIARIZE)

- **`warm_start_dir` latent path** (`driver._load_warm_start_into`) — confirmed it already
  handles the latents direct-load (the strict DECODER load is what fails for a re-taper, NOT
  the latents). `load_kd_warm_start_latents` mirrors its tolerant bare-tensor / dict-wrapped
  / shape-strict load.
- **`_film_param_ids` id-set pattern** — `_rgb0_param_ids` mirrors it (id()-based, decoupled
  from name strings).
- **`_split_by_head_backward` snapshot+restore** — Part A is a one-line change to the set it
  snapshots (exclude rgb_0); the routing is unchanged.
- **`ConfigurableTaperHNeRVDecoder` + `PoseFiLMHNeRVWrapperV2`** — the student backends,
  reused verbatim; the v2 wrapper already wraps whatever `_new_vendored_decoder` builds.
- **KD loss search:** `tac.archive.scorer_distill` distills a renderer's features into tiny
  SCORER heads (renderer→scorer-proxy); `experiments/train_distill.py` trains a renderer
  from rendered-frame targets but is a monolithic CLI bound to the renderer/SegMap stack —
  NEITHER is a decoder-architecture frame-KD for an HNeRV warm-start. So `kd_frame_mse` is a
  focused, documented clean frame-MSE (the simplest objective that makes the re-tapered
  student reproduce the basin's rendered pairs; the score-aware curriculum that follows
  supplies the SegNet/PoseNet awareness).

## Part C — sensitivity-spine single-source finding (verified, NOT rebuilt)

The driver-side QAT/codec level allocation reads from ONE sensitivity source — confirmed:
- The single source is `rt.tensor_sensitivity_ema` (`s_t = ||∂S/∂w_t||`), accumulated once
  per step by `accumulate_tensor_sensitivity(decoder, rt.tensor_sensitivity_ema, …)` from
  the score-domain `w.grad` (driver L~1346, only when `score_aware_qat`).
- **QAT bits (training-time):** `apply_score_aware_qat(decoder, sens)` reads
  `rt.tensor_sensitivity_ema` (driver L~1232).
- **Variable-level export codec (rate-attack):** `levels_from_sensitivity_for_codec(
  sensitivity, weight_keys)` reads the SAME `rt.tensor_sensitivity_ema` snapshot (driver
  L~1846), and the docstring states it is "the SAME rank-norm band the score-aware QAT
  trained the decoder to be robust at."
- Both go through the SAME rank-norm band (`_rank_normalize` in `score_aware_qat` /
  `variable_level_codec`). The EMA is carried across stages (the QAT→QAT-boundary carry) and
  round-tripped on resume — one coherent signal.

**Single-source contract (driver side): CONFIRMED.** Capacity (taper channels, the
structural lever) is the one allocation that is NOT computed from `tensor_sensitivity_ema` at
runtime — it is set offline by the solved taper `[22,16,15,14,15,14,10]` derived from the
gate-2 sensitivity map. So the spine is: ONE offline gate-2 sensitivity map → taper
(channels), and ONE online `tensor_sensitivity_ema` → {QAT bits, codec levels}. The bind-all
spec's "compute sensitivity ONCE and fan out" is honored on the driver side via the single
`tensor_sensitivity_ema`; whether the OFFLINE gate-2 map and the ONLINE EMA are the same
rank-norm band (so taper + bits + levels truly all follow one map) is a codec/probe-side
question = P1b/P2 scope — NOT refactored here (the codec is out of scope for P1a). No
redundant recompute exists on the driver side.

## Recommended composed-launcher flags (the bind-all production run)

```
--kd-warm-start-dir <converged_vendored_basin_best_dir> \
--taper-channels 22,16,15,14,15,14,10 \
--pose-film-v2 --pose-film-trunk-stopgrad --pose-film-rgb0-pose-trainable \
--pose-grad-every-k 1
```
(plus the existing score-aware levers: oomph sharp soft_cosine seg, progressive-QAT + Muon +
C1a + σ in the refinement, `--ema-warmup`, and — for the rate attack — the variable-level
codec / Lever-4 export, enabled at the FINAL export per its own discipline). NOTE: the
config field is `kd_warm_start_dir`; the launcher CLI wiring of `--kd-warm-start-dir` /
`--pose-film-rgb0-pose-trainable` is a thin follow-on (the driver contract is landed here).

## Files

- `src/tac/torch_vehicle/kd_warm_start.py` — NEW: `load_kd_warm_start_latents`,
  `build_frozen_teacher`, `kd_frame_mse`, `kd_warm_up_decoder`.
- `src/tac/torch_vehicle/driver.py` — config fields `pose_film_rgb0_pose_trainable`,
  `kd_warm_start_dir`, `kd_warm_epochs`, `kd_warm_lr`, `kd_warm_train_latents`;
  `_rgb0_param_ids`; `_non_film_grad_params` rgb_0 exclusion; `_run_kd_warm_up`; `run()` KD
  stage-0 init + prefix-phase wiring; relaxed taper+FiLM-v2 `__post_init__` constraint +
  KD guards.
- `src/tac/torch_vehicle/tests/test_kd_warm_start.py` — NEW: 27 NO-FAKE tests.
- `src/tac/torch_vehicle/tests/test_configurable_taper_decoder.py` — updated the
  taper+FiLM validation test for the relaxed v2-allowed / v1-refused contract.

## Wire-in hooks (per Catalog #125; this is a TRAIN-TIME actuator, no score claim)
1. sensitivity-map — N/A (Part C VERIFIES the existing single-source spine; no new map).
2. Pareto — N/A (no byte/score claim; the KD warm-up has no archive effect).
3. bit-allocator — N/A (KD does not allocate bits; it primes the decoder).
4. cathedral autopilot — N/A (a train-time priming flag, not an archive-deployable lane).
5. continual-learning posterior — N/A (no empirical score anchor; advisory, $0).
6. probe-disambiguator — N/A (no 2-interpretation design tension; the KD is a single
   documented frame-MSE; the rgb_0 refinement is proved EXACT by test, not ambiguous).
```
