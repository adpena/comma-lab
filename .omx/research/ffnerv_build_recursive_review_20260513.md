# FFNERV-BUILD recursive adversarial review — 2026-05-13

**Scope:** Canonicalize `experiments/train_substrate_ff_nerv.py` to import the
seven shared helpers from `tac.substrates._shared.trainer_skeleton` per the
2026-05-13 FFNERV-BUILD directive ("USE canonical helper from byte one") and
the CANON-DEDUP-1 audit. The substrate package (`src/tac/substrates/ff_nerv/`),
the operator-authorize wrapper, the YAML recipe, and the remote-lane driver
were already landed by WAVE-3 / FFFF / WAVE-A-1 prior sessions; the only
outstanding BUILD gap was the trainer still carrying inlined copies of the
canonical helpers.

## Pass 1 — Yousfi + Fridrich + Hotz

- **Yousfi**: TIER_1 manifest still declares all 6 contest-relevant flags
  (--video-path, --output-dir, --epochs, --batch-size, --upstream-dir,
  --device) via AnnAssign per Catalog #151; refactor did not strip the
  required_input_file or generator_command metadata. PASS.
- **Fridrich**: HNeRV parity lesson L4 honored (inflate.py 90 LOC, <= 100 base
  budget); lesson L1 (score-aware) honored via `FfnervScoreAwareLoss` →
  `score_aware_common.score_pair_components` (Catalog #164). PASS.
- **Hotz**: Refactor deleted ~70 LOC of duplication; trainer dropped from
  1216 to ~1146 LOC; helpers now reviewable in 30 s. PASS.

**Verdict:** 0 issues. Counter 1/3.

## Pass 2 — Shannon + Dykstra + MacKay

- **Shannon**: Score-domain Lagrangian shape unchanged (alpha*B(theta)/N +
  beta*d_seg + gamma*sqrt(d_pose)); rate term `_archive_bytes_proxy_closed_form`
  still computes (fp16 decoder + int16 latents) bytes. PASS.
- **Dykstra**: Feasibility (epochs >= 0, batch_size >= 1, val_pair_count >= 1)
  preserved at argparse; cosine-annealing scheduler intact. PASS.
- **MacKay**: Continual-learning posterior update via `posterior_update_locked`
  (Catalog #128 atomic fcntl) preserved at line ~975 of the refactored
  `_full_main`. Bayesian custody intact. PASS.

**Verdict:** 0 issues. Counter 2/3.

## Pass 3 — Quantizr + Selfcomp + Contrarian

- **Quantizr**: EMA(decay=0.997) update after every `optimizer.step()` still in
  place; snapshot+restore pattern at val intact; EMA shadow → archive bytes;
  live weights never shipped. PASS.
- **Selfcomp**: Archive grammar unchanged (FFV1 monolithic 0.bin via
  `pack_archive`); inflate-time IDCT2 basis rebuild from cfg honored;
  deterministic ZIP construction (fixed_ts + ZIP_DEFLATED) preserved per
  Catalog #19. PASS.
- **Contrarian**: The refactor is NOT a no-op — it deletes ~70 LOC and
  centralizes the seven helpers behind a single import block. The two real
  risks are (a) substrate-tag collision in `importlib` (mitigated: the
  canonical helper uses `f"pact_{substrate_tag}_upstream_frame_utils"` with
  SUBSTRATE_TAG="ff_nerv", unique across the 14 trainers) and (b)
  `device_or_die` error-string substring drift (mitigated: canonical helper
  still emits the "[ff_nerv]" tag via substrate_tag). PASS.

**Verdict:** 0 issues. Counter 3/3 — CLEAN.

## Wire-in declaration (CLAUDE.md Catalog #125)

- Sensitivity-map contribution: **N/A** — pure trainer canonicalization, no new
  signal added to the planner; rate/seg/pose sensitivities unchanged.
- Pareto constraint: **N/A** — Pareto feasibility (rate + seg + pose) is the
  trainer's loss already; refactor does not change its shape.
- Bit-allocator hook: **N/A** — no new per-tensor importance signal.
- Cathedral autopilot dispatch hook: **wired (existing recipe)** — recipe
  `substrate_ff_nerv_modal_a100_dispatch.yaml` already declares
  dispatch_enabled=false (defer-by-routing) and predicted_score_target=0.19;
  autopilot already knows about this lane.
- Continual-learning posterior update: **wired (existing trainer path)** —
  `posterior_update_locked` runs at full-main step 13 on successful CUDA
  auth eval.
- Probe-disambiguator: **N/A** — the substrate's design verdict is
  unambiguous (frequency-domain DCT renderer with 64x64 cutoff prior).

## Cross-refs

- 2026-05-13 FFNERV-BUILD directive (parent session)
- CLAUDE.md "Beauty, simplicity, and developer experience"
- CLAUDE.md Catalog #151 (`check_operator_wrapper_threads_trainer_tier_required_flags`)
- CLAUDE.md Catalog #152 (`check_operator_wrapper_validates_required_input_files_pre_dispatch`)
- CLAUDE.md Catalog #164 (`check_substrate_score_aware_loss_calls_preprocess_input_before_scorer`)
- CLAUDE.md Catalog #146 (`check_phase1_trainer_runtime_emits_contest_compliant_inflate`)
- `tac.substrates._shared.trainer_skeleton` (canonical helper landed by CANON-DEDUP-1)
- Reference template: `experiments/train_substrate_hi_nerv.py` (same canonicalization pattern, prior session)
- Landing memo: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_ffnerv_build_LANDED_20260513.md`
