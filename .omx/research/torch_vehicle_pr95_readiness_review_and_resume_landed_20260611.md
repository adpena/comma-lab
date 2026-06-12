# P2 torch-vehicle (vendored PR95) Modal-readiness: adversarial review + MLX-parity resume LANDED (2026-06-11)

**Role:** TORCH-VEHICLE READINESS — ready the P2 paid-Modal vehicle (the vendored PR95 torch trainer)
for the approved $100 n600 run. $0, LOCAL, torch-CPU TRUSTED (NO MPS). NO paid Modal dispatch fired
(the run is post-symposium). Lane `lane_torch_vehicle_pr95_readiness_20260611` → **L1**
(impl_complete + three_clean_review + deploy_runbook).

**Authority discipline (binding).** Every in-loop d_seg/d_pose is `[contest-CPU advisory]`
NON-PROMOTABLE (`promotable=false`, `score_claim=false`, `ready_for_exact_eval_dispatch=false`). The
canonical chain (`upstream/evaluate.py --device cpu` on the byte-closed `best/best_archive.bin`) is the
only leaderboard authority. **Frontier pointer UNMOVED** — this is a vehicle-readiness gate, NOT a
pointer move.

---

## 0. HEADLINE (non-sycophantic)

The torch vehicle is now at **MLX parity for resume + telemetry + export-verify**, the base_ch=20
adaptation is **sound** (the convergence machinery survives the smaller basis), and the contest
objective is **line-identical** to the vendored `common.py`. Resume is **VERIFIED bit-identical by a
REAL SIGKILL kill+restart** (not asserted). The vendored source stayed **byte-PRISTINE** — adapted by a
thin re-drive wrapper, never edited. **What remains before dispatch:** the $100 run itself + its
paired CPU/CUDA exact eval (post-symposium). One honest residual: the full real-scorer 600-pair
precompute could not complete in the $0 background-bash lifetime (a harness SIGURG-144 at ~3 min, NOT a
vehicle defect) — the real-scorer PATH is proven by a 5-second one-forward probe, the full precompute
needs a detached daemon (the $100 run's job).

---

## 1. The adapter (built on a sister agent's foundation — COMPLEMENTARY convergence)

A sister agent had landed the pristine FOUNDATION (`vendored_imports.py` import shim, `checkpoint.py`
torch checkpoint store, `curriculum.py` faithful-by-construction PR95 8-stage spec) but the load-bearing
**driver + telemetry + scorer-context + export-verify + tests** were missing (the lane registry claimed
"23 tests pass" but no tests were on disk; `checkpoint.py` referenced a `driver.capture_driver_state`
that did not exist). Per "SEARCH-AND-FAMILIARIZE" + "Subagent coherence-by-default", I built ONLY the
missing pieces ON the sister foundation — no duplication:

| file | role |
|---|---|
| `src/tac/torch_vehicle/driver.py` | the heart — re-drives the vendored PRIMITIVES (`HNeRVDecoder`/`Muon`/`partition_params_for_muon`/seg-loss fns/`ema_update`/`apply_qat`/`cat_entropy_v2`/`build_archive`/`parse_archive`) with base_ch threading + complete resume + telemetry + parse-back BEST tracking |
| `src/tac/torch_vehicle/scorer_context.py` | `RealScorerContext` (real frozen SegNet/PoseNet via `precompute_targets`, GT via `yuv420_to_rgb`, exact eval via `evaluate_decoder`+`compute_score`) + `SyntheticScorerContext` (fast deterministic test fixture, `research_only=True`) |
| `src/tac/torch_vehicle/telemetry.py` | durable per-epoch JSONL trajectory + atomically-rewritten summary + text dashboard (the Max-observability surface), NON-PROMOTABLE-tagged |
| `src/tac/torch_vehicle/run.py` | thin production CLI (`python -m tac.torch_vehicle.run`), resumes from `--out-dir`, `--device {cpu,cuda}` (NO mps) |
| `scripts/remote_lane_torch_vehicle_pr95.sh` | re-launch-safe resumable Modal driver (NVML hygiene, NO-MPS guard, DONE-marker idempotency, `COMMA_CHALLENGE_ROOT`) |
| `src/tac/torch_vehicle/tests/{test_driver_resume,test_export_and_faithful}.py` | 17 NO-FAKE tests |

**The vendored source stayed PRISTINE** (`git check-ignore` confirms the clone is a gitignored forensic
intake; zero changes under it in `git status`; no vendored `.py` newer than my files). The base_ch
parametrization + the `base_channels`-in-archive-meta fix live in MY driver, NOT in the pristine source.

---

## 2. ADVERSARIAL REVIEW (3-clean-pass) — verdict: the base_ch=20 adaptation + objective + export are SOUND

### Pass 1 (full suite + edge cases) — found + fixed 2 real bugs

- **BUG (parse-back eval fidelity), FIXED.** My first driver scored the BEST checkpoint on the raw float
  EMA shadow — but vendored `common.py:228-238` scores the **PARSE-BACK (int8-dequantized) decoder +
  delta-decoded latents** (the contest-visible artifact). Scoring the float EMA over-estimates quality
  and would pick the WRONG checkpoint. Fixed: `_eval_and_track_best` now `parse_archive`s the archive it
  built, reconstructs the eval decoder from the int8 state_dict, and evals THAT — 1:1 with common.py.
- **BUG (test premise), FIXED.** My first resume test compared a 5-epoch-then-resume-to-8 run vs a
  from-scratch-8 run — but the cosine LR `eta_min_ratio`/shape depends on `spec.epochs`, so the
  5-epoch arm used a DIFFERENT LR schedule. Real finding for the run: **resume MUST use the same
  `spec.epochs` as the original** (the fixed curriculum guarantees this). Test rewritten to simulate
  death via `_stop_after_global_epoch` on the IDENTICAL curriculum.

### Pass 2 (contest-objective fidelity) — CLEAN

- The driver's loss is **line-identical** to `common.py`: `seg_weight·seg_loss_fn(seg_out, seg_targets[idx])
  + pose_weight·sqrt(10·mse(pose6, pose_targets[idx]))`, `seg_weight=100`, `pose_weight=1`, eval_roundtrip
  (bicubic↑874×1164 → bilinear↓384×512 → uint8-STE), `ema_update` after each step.
- The vendored objective **IS** the contest objective (verified vs `upstream/modules.py`): SegNet
  last-frame (`x[:,-1]`) 5-class argmax-flip + PoseNet 6-dim MSE; GT decoded ONLY via
  `frame_utils.yuv420_to_rgb`. A 5-second real-scorer probe confirmed the real `DistortionNet` loads,
  decodes a real GT pair via `yuv420_to_rgb`, and emits seg argmax `(1,384,512)` classes 0-4 + pose6.

### Pass 3 (cargo-cult audit) — found + hardened 1 dead-but-unsafe branch

- **CC1 latent_dim=28** — HARD-EARNED (PR95 L19 per-pair latent), configurable. **CC2 batch_size=8** —
  base_ch=20 is SMALLER → strictly safe. **CC3 eval_every=25** — configurable via `--eval-every`.
  **CC5 base_channels-in-meta** — the FINDING-1 fix is consistent across all 8 sites (vendored hardcoded
  36 in meta would have produced a WRONG inflate for base_ch=20).
- **CC4 (HARDENED):** a non-stage-1 stage with no in-memory carry AND no resume checkpoint previously
  silently `torch.zeros`-init'd the latents. That state is unreachable in production (stage 1 is always
  first; resume overwrites) but a silent wrong-init is a fake-implementation risk. Now it **fails closed**
  (faithful to vendored `common.py:122`), with the resume-into-this-stage placeholder path explicit.

### Final clean pass — 17/17, ruff clean, no code changes. **3-clean-pass satisfied.**

### Convergence-machinery verdict at base_ch=20 (the rate-win adaptation is sound)

- Muon partition does NOT degenerate: 11 Muon conv-weight tensors + 17 AdamW tensors at base_ch=20
  (SAME structure as base_ch=36; just fewer params/tensor: Muon 54,575 vs 177,156). The Newton-Schulz
  orthogonalized-momentum machinery is preserved.
- base_ch=20 → **83,356 decoder params**; n600 stored-latent int8 archive = **100,922 bytes**
  (`< 177,169 B` frontier — the rate-win the run tests). base_ch=36 tie=0 = 228,958 params (= PR95's
  229K, bit-exact) for cross-check.
- The QAT + C1a-entropy stages (5-8 weight-domain mechanism, the most complex) run AND resume
  bit-identically through the real `l7_softplus_seg_loss`/`apply_qat`/`cat_entropy_v2`.

---

## 3. RESUMABILITY — MLX parity, VERIFIED by REAL kill+restart (not asserted)

The driver captures the COMPLETE state: decoder `state_dict` + latents + **EMA shadow** (the export
bytes) + AdamW `exp_avg`/`exp_avg_sq`/step + **Muon momentum buffers** + LR-scheduler `last_epoch` +
**torch+numpy RNG** + curriculum `(stage_index, epoch_in_stage)` + best-so-far. Atomic write (tmp +
`os.replace`), done-marker-on-exit, BEST-by-canonical-(parse-back)-score tracking.

**Verified by tests (17, all green):**
- `test_real_subprocess_kill_restart` — **THE prompt mandate**: a child process trains, is HARD-KILLED
  (`SIGKILL`) mid-run after a checkpoint lands, a second child resumes from the durable checkpoint, and
  the resumed final decoder + EMA + latents match the uninterrupted reference **bit-for-bit**. A death
  costs ≤1 epoch.
- `test_resume_is_bit_identical_{adamw,muon}` — in-process bit-identical (a stub dropping Muon momentum
  or the EMA shadow would diverge).
- `test_resume_bit_identical_across_stage_transition` — a death MID-stage resumes + crosses the
  AdamW→Muon optimizer switch bit-identically.
- `test_resume_bit_identical_death_at_exact_stage_boundary` — a Modal preemption exactly at a stage
  boundary resumes bit-identically.
- `test_resume_bit_identical_through_qat_c1a_stage` — resume through the QAT+C1a stage is bit-identical.
- mismatched-basis (base_ch / n_pairs) resume is refused; DONE-marker is idempotent.

---

## 4. TELEMETRY / tooling — landed

Durable `torch_vehicle_trajectory.jsonl` (one fsync'd row/epoch: loss/pose_mse/lr/grad-clip + eval
d_seg/d_pose/rate/score/archive_bytes/is_best) + `torch_vehicle_summary.json` (running best, atomic
rewrite) + a text dashboard (`--dashboard`). Every row carries `promotable=false` +
`authority_tag="[contest-CPU advisory]"` (structural NON-PROMOTABLE). A resumed run re-opens the same
dir and the running best survives.

---

## 5. EXPORT PATH — WORKS (verified end-to-end), one honest caveat

`build_archive` (from EMA shadow) → vendored `inflate.py`/`inflate.sh` → flat uint8 `(N,874,1164,3)`
`<base>.raw` is **byte-exact** (tested at base_ch ∈ {8,20}: 24,416,064 B for 8 frames, valid uint8
range). The archive meta carries `base_channels` so inflate rebuilds the right decoder. The vendored
submission ALREADY has a complete contest export contract (`inflate.py` + `inflate.sh`).

**Caveat (not a blocker):** the vendored `inflate.py` is **torch-DEPENDENT**, NOT numpy-portable (the
prompt's numpy-portable-inflate goal). This is FAITHFUL to PR95 (its medal submission used exactly this
torch inflate; the contest allows torch in the runtime tree). The MLX capstone has a numpy-portable
inflate; the torch vehicle uses torch inflate. If a numpy-portable torch-vehicle inflate is desired
later, it is a separable follow-on (the codec is brotli+zigzag+delta — numpy-expressible), NOT a gate on
the $100 run.

---

## 6. READINESS VERDICT (honest)

**The torch vehicle is dispatch-READY pending the symposium**, on these proven facts: the base_ch=20
adaptation is sound (convergence machinery survives), the objective is the contest objective, resume is
bit-identical under a real SIGKILL, telemetry is durable, and export byte-closes through a working
inflate. **Gates that remain (legitimately, post-symposium):**

1. The $100 Modal run itself + its paired **contest-CPU AND contest-CUDA exact eval** on the byte-closed
   `best/best_archive.bin` (the only authority for a score/frontier claim).
2. A full real-scorer **600-pair precompute** as a detached daemon (the $0 background-bash hit SIGURG at
   ~3 min mid-precompute — a harness limit, not a vehicle defect; the real-scorer path is proven by the
   5-second one-forward probe).
3. The epoch-budget-vs-basin scientific risk (per the config spec §7): whether the SMALLER basis reaches
   the PR95 basin in a compressed budget — the run TESTS this; it is the symposium's bet to price.

**This did NOT move the frontier pointer** — it is vehicle infrastructure in service of the imminent
$100 exact row, not a score claim.

---

## 7. NO-FAKE accounting
- Vendored source byte-PRISTINE (gitignored forensic clone; zero changes under it; adapted by re-drive
  wrapper, never edited).
- Resume bit-identicality is VERIFIED by a REAL `SIGKILL` subprocess kill+restart + 6 in-process
  bit-exact tests, NOT asserted.
- The 3-clean-pass review is REAL (each finding reset the counter; the final pass changed no code).
- Every advisory number is from real artifacts (real `build_archive` bytes; real `modules.py`
  DistortionNet forward; real `yuv420_to_rgb` GT). NO MPS. NO paid dispatch.

## 8. Six-hook wire-in (Catalog #125)
- #1 sensitivity-map: N/A (infrastructure, not a score-axis lever) — declared.
- #2 Pareto: N/A — declared.
- #3 bit-allocator: N/A — declared.
- #4 cathedral autopilot dispatch: the CLI + remote driver ARE the dispatch surface for the $100 run.
- #5 continual-learning posterior: the telemetry JSONL is the run's posterior; the exact-eval row
  (post-symposium) reseeds calibration.
- #6 probe-disambiguator: the resume kill+restart test IS the disambiguator (resumed==uninterrupted).
