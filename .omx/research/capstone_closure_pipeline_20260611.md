# Capstone closure pipeline: archive -> numpy-inflate -> contest-CPU-predict -> armed eval-packet

**Date:** 2026-06-11
**Author:** capstone-closure-pipeline subagent (`capstone-closure-pipeline-20260611`)
**Lane:** `lane_capstone_closure_pipeline_20260611` (claimed; `local_pipeline_build`, no paid dispatch)
**Operator ask:** build + verify the end-to-end pipeline that turns a capstone int8 archive into a banked
contest-CPU exact-row CANDIDATE — the step between "trained vehicle" and "moved pointer". NO paid dispatch
(packet armed only). Do not touch the running daemons (capstone 48-pair pid 72123, the 2x2 ablation
pid 85721, atlas).

**Did the exact frontier pointer move?** No. This is the critical-path *closure tooling* + an end-to-end
*verification* on an existing archive, not an exact-eval row. The contest exact eval (the armed, un-fired
packet) remains the only score authority.

---

## 0. TL;DR

- **Built** `tools/capstone_archive_to_contest_candidate.py` — a thin orchestrator that chains the EXISTING
  pieces (reuse, not reinvent) into the 4-step closure: archive.zip -> ACTUAL numpy inflate on disk ->
  contest-faithful local score from the inflated frames -> drift-corrected predicted contest-CPU score ->
  ARMED (un-fired) paired-eval dispatch packet.
- **Verified end-to-end** on the existing `capstone_vq_index_smoke_b20_n8/archive.zip` (base_ch=20, n=8,
  int8, vq_index) AND on `capstone_smoke_stored_latent_b20_n8/archive.zip` (stored_latent carrier) — both
  carriers run clean through the carrier-agnostic decode/render path.
- **Agreement check (NO-FAKE):** the pipeline's disk-frame `d_seg` matches the trainer's render-resolution
  `reloaded_int8` advisory **bit-for-bit** (0.5072727203369141 == 0.5072727203369141); `d_pose` differs by
  ~1.2% (107.15 disk vs 108.42 render-res) — the expected A3 camera-resolution uint8 round-trip fidelity
  the disk-frame path captures and the render-res reloaded advisory does not. The disk-frame number is the
  MORE faithful contest predictor.
- **Rate term is contest-exact:** recomputed from the actual `archive.zip` file size (not the payload), with
  denominator `37_545_489` = `sum(upstream/videos/*)` (verified == `evaluate.py`'s `uncompressed_size`).
- **9 NO-FAKE tests** (8 in the suite + the agreement assertion) all pass; ruff clean.
- **Chain is built + verified.** The moment the base_ch=24 @ 600-pair run emits an archive, one command
  produces its predicted contest-CPU score + the armed (un-fired) ~$0.12 eval packet.

---

## 1. The 4-step chain (what the tool does, reusing existing pieces)

| Step | What | Reused component | Output |
|---|---|---|---|
| 1 | ARCHIVE -> ACTUAL numpy inflate on disk (bicubic A3 camera upscale to 874x1164) | `tac.capstone_vq_nerv.inflate.{_read_archive_and_config, decode_archive, render_all_camera_frames}` | `(N, 874, 1164, 3)` uint8 `.raw` on disk (the exact bytes `inflate.sh` writes for the evaluator) |
| 2 | INFLATED FRAMES -> contest-faithful LOCAL score | `tac.mlx_pr95_port.score_bridge.TorchScorerBridge` (`exact_d_seg`/`exact_d_pose`, eval_roundtrip A2) + `tac.score_aware_loop.targets` GT cache | `d_seg`, `d_pose`, `rate=zip_bytes/37_545_489`, `S=100*d_seg+sqrt(10*d_pose)+25*rate` `[macOS-CPU advisory]` |
| 3 | LOCAL -> PREDICTED contest-Linux-x86_64-CPU score | RUNG-B drift (`local_to_contest_scorer_drift_ladder` §2/§3.2; bias +1.05e-5, guard 3e-6) mirroring `local_cpu_contest_drift.conservative_projected_contest_score` | point + conservative projection + submit rule (conservative < frontier_CPU) `[predicted contest-CPU]` |
| 4 | -> ARMED (un-fired) paired-eval packet | `tools/claim_lane_dispatch.py` + `experiments/modal_auth_eval_cpu.py` + `src/tac/capstone_vq_nerv/runtime/inflate.sh` | lane-claim cmd + Modal Linux-x86_64-CPU eval cmd on EXACT archive bytes + ~$0.12 cost; `armed` iff sub-frontier; `fired=False` |

### Why scoring the on-disk camera frames (not the render-res reloaded advisory) is the A2+A3 closure

`advisory.py::score_reloaded_int8_archive` re-scores the int8 archive at **render resolution** (384x512, no
camera round-trip) deliberately, to isolate the int8/fp16 quantization (A2). The contest path is stricter:
`inflate.py` writes **camera-resolution** uint8 frames (bicubic-up to 874x1164, A3), the evaluator reads
those off disk and bilinear-downsamples to 384x512, then applies eval_roundtrip. This tool feeds the
**on-disk camera frames** into `bridge.exact_d_seg` — which resizes camera->scorer (matching the evaluator's
downsample) and applies eval_roundtrip — so the resulting `S` is the honest predictor of
`inflate.sh -> evaluate.py` with BOTH A2 (int8 reload) AND A3 (camera bicubic + uint8 clamp) closed.

The empirical receipt that A3 is real and captured: `d_seg` is identical to the render-res advisory (argmax
is stable), but `d_pose` shifts ~1.2% (107.15 vs 108.42) — the PoseNet regression head IS sensitive to the
camera uint8 round-trip, exactly as the drift ladder predicts (pose is the noise-sensitive term).

---

## 2. End-to-end verification (on which archive)

Command (n8 vq_index smoke; n8 GT-targets cache already on disk, so no GT decode):

```bash
.venv/bin/python tools/capstone_archive_to_contest_candidate.py \
    --archive experiments/results/capstone_vq_index_smoke_b20_n8/archive.zip \
    --targets-cache experiments/results/capstone_gt_targets_cache \
    --frontier-cpu-score 0.191099824 \
    --out-dir experiments/results/capstone_closure/verify_n8_vq_index
```

Result (`capstone_closure/verify_n8_vq_index/closure_record.json`):

| Quantity | Value | Note |
|---|---:|---|
| `[1]` inflated frames on disk | `(16, 874, 1164, 3)` uint8 | the real contest decode path ran |
| `[2]` local `d_seg` | 0.50727272 | == trainer reloaded-int8 d_seg (bit-exact) |
| `[2]` local `d_pose` | 107.15054 | disk-frame (A3 captured); render-res advisory was 108.42 |
| `[2]` rate term `25*rate` | 0.06067706 | == 25 * 91126 / 37_545_489 (contest-exact, ZIP file size) |
| `[2]` local `S` | 83.521807 | `[macOS-CPU advisory]`, NON-PROMOTABLE |
| `[3]` predicted contest-CPU (conservative) | 83.521799 | `[predicted contest-CPU]`, prediction only |
| `[3]` conservative_beats_frontier | False | (n8 smoke is nowhere near frontier — expected) |
| `[4]` eval packet | `armed=False`, `observe_only` | correctly NOT armed for a non-candidate |

**This is PIPELINE verification only** (num_pairs=8 != 600, so it is honestly flagged
`full_600_pair_faithful=false` and is NOT a faithful contest score — the prompt explicitly allows the n8
output for plumbing verification). The same chain runs unchanged on the stored_latent carrier
(`verify_n8_stored_latent`: S=86.85, both branches of `decode_archive` exercised).

### Predicted-vs-numpy-inflate S agreement

The "predicted-vs-numpy-inflate S agreement" is, by construction, the RUNG-B drift offset only: predicted
`S = local_S - bias + guard`, so |predicted - local| = |guard - bias| = 7.5e-6 score (the conservative
projection sits 7.5e-6 BELOW the local number because bias > guard). The numpy-inflate local `S` and the
predicted contest-CPU `S` agree to within that 7.5e-6 cushion — the drift correction is a tiny, one-sided,
HNeRV-class-bounded offset, not a large re-scaling. (The large, separate CUDA-CPU RUNG-C gap is never
applied to a CPU-axis prediction; the tool predicts only the CPU leaderboard axis.)

---

## 3. The exact armed (un-fired) ~$0.12 eval command for the 600-pair candidate

When the base_ch=24 @ 600-pair run emits `archive.zip` at `<RUN_DIR>/archive.zip`, run the closure tool:

```bash
.venv/bin/python tools/capstone_archive_to_contest_candidate.py \
    --archive <RUN_DIR>/archive.zip \
    --targets-cache experiments/results/capstone_gt_targets_cache \
    --frontier-cpu-score 0.191099824 \
    --out-dir experiments/results/capstone_closure/<candidate_id> \
    --candidate-id <candidate_id>
```

It prints the contest-faithful local `S`, the predicted contest-CPU `S`, and emits the packet. If
`conservative_beats_frontier=True` (a real sub-frontier candidate), the packet is `armed=True` and the two
commands below are the ARMED (un-fired) ~$0.12 contest-CPU exact eval (fire only on a real sub-frontier
candidate or explicit operator greenlight; HARVEST-OR-LOSE within 24h after firing):

```bash
# 1. lane-claim FIRST (cross-agent coordination + HARVEST-OR-LOSE)
.venv/bin/python tools/claim_lane_dispatch.py claim \
    --lane-id lane_capstone_<candidate_id>_contest_cpu_eval_20260611 \
    --platform modal --agent claude:capstone_closure --instance modal_cpu --status active \
    --notes "contest-CPU exact eval of capstone candidate <candidate_id> sha <sha12> (<bytes> B); HARVEST-OR-LOSE within 24h"

# 2. the canonical Linux x86_64 CPU exact eval on the EXACT archive bytes (~$0.12, 60-120 min)
PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach \
    experiments/modal_auth_eval_cpu.py \
    --archive <RUN_DIR>/archive.zip \
    --expected-archive-sha256 <full_sha256> \
    --inflate-sh src/tac/capstone_vq_nerv/runtime/inflate.sh \
    --output-dir experiments/results/capstone_closure/<candidate_id>/modal_cpu_eval \
    --lane-id lane_capstone_<candidate_id>_contest_cpu_eval_20260611 \
    --claim-agent claude:capstone_closure \
    --detach --provider-detach-ack
```

The tool emits these two commands verbatim (with the real sha256 + byte count + candidate id substituted)
in `eval_packet.{lane_claim_command, modal_cpu_eval_command}`. **Cost estimate ~$0.12** (Modal CPU
container ~$0.06/hr, 600-sample eval 60-120 min — per the drift memo §4.1). The contest-CPU result_json is
the ONLY score authority; the closure tool's predicted `S` is the predictor + the trigger.

---

## 4. NO-FAKE / authority notes

- The numpy-inflate `S` is the contest-faithful LOCAL number, tagged `[macOS-CPU advisory]` —
  NON-PROMOTABLE (macOS is NOT 1:1 contest CPU). `score_claim=false`, `promotion_eligible=false`,
  `rank_or_kill_eligible=false`, `ready_for_exact_eval_dispatch=false`.
- The drift-corrected number is a PREDICTION tagged `[predicted contest-CPU]` — NOT a score claim
  (`authority=false_authority_prediction_only_exact_eval_is_arbiter`). The ONLY authority is the actual
  contest eval (the armed packet).
- Pure numpy decode of the REAL archive bytes (`tac.capstone_vq_nerv.inflate`); torch-CPU scorer bridge
  only; **no MPS anywhere** (the CLI fails closed on `--device mps`).
- Rate term uses the actual `archive.zip` file size and the verified `37_545_489` denominator (==
  `evaluate.py`'s `uncompressed_size`), so the rate contribution is contest-exact.

---

## 5. Solver / system wire-in (results become intelligence)

Per CLAUDE.md "Results must become system intelligence" — the 6 unified-Lagrangian hooks:
1. **Sensitivity-map** — N/A (this tool measures/predicts a whole-archive score; no per-byte importance
   change).
2. **Pareto constraint** — N/A (it predicts a score + arms an eval; it does not add a rate/seg/pose
   feasible-set constraint).
3. **Bit-allocator hook** — N/A (no per-tensor importance change).
4. **Cathedral autopilot dispatch hook** — ACTIVE in spirit: the §4 eval packet IS the capstone-class
   CPU-axis spend trigger (the drift-corrected conservative projection vs the CPU frontier), the same
   eureka-rule shape as `local_cpu_contest_drift`. The tool emits the armed dispatch packet directly.
5. **Continual-learning posterior** — TRIGGERED on the empirical anchor: when the armed packet is fired and
   the contest-CPU result lands, it becomes a same-archive paired anchor that can reseed
   `local_cpu_contest_drift` (refining the capstone-class RUNG-B bias from "HNeRV-inferred" to "measured on
   our substrate").
6. **Probe-disambiguator** — ACTIVE: the disk-frame-vs-render-res `d_pose` gap (107.15 vs 108.42) IS the
   disambiguator between "A2 int8-quant only" and "A2+A3 full camera round-trip" — the tool measures the
   stricter, more faithful number.

**`research_only=true`** for the verification rows (n8 smoke, not 600-pair). The tool itself is a wired,
tested production helper (one production CLI + 8 tests). Integration blocker for a real exact-eval row: the
base_ch=24 @ 600-pair archive (in flight on the daemon) + operator greenlight to fire the ~$0.12 packet on a
real sub-frontier candidate.

---

## 6. Reproduce / sources
- Tool: `tools/capstone_archive_to_contest_candidate.py`. Tests:
  `src/tac/tests/test_capstone_archive_to_contest_candidate.py` (8 pass).
- Reused: `tac.capstone_vq_nerv.{inflate, export, numpy_reference, advisory}`,
  `tac.mlx_pr95_port.score_bridge.TorchScorerBridge`, `tac.score_aware_loop.targets`,
  `src/tac/capstone_vq_nerv/runtime/inflate.sh`, `experiments/modal_auth_eval_cpu.py`,
  `tools/claim_lane_dispatch.py`.
- Drift correction: `.omx/research/local_to_contest_scorer_drift_ladder_and_correction_20260611.md`,
  `src/tac/optimization/local_cpu_contest_drift.py`.
- Contest formula: `upstream/evaluate.py:63-92` (rate = archive.zip.size / sum(upstream/videos/*) = /37_545_489;
  `score = 100*segnet + sqrt(10*posenet) + 25*rate`).
- Verification outputs: `experiments/results/capstone_closure/verify_n8_vq_index/closure_record.json`,
  `experiments/results/capstone_closure/verify_n8_stored_latent/closure_record.json`.
- Frontier: `.omx/state/canonical_frontier_pointer.json` (CPU 0.191099824).
