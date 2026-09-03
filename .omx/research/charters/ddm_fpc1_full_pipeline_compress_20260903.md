# ddm_fpc1_full_pipeline_compress — the from-raw-video compress pipeline: ONE resumable driver that runs training → token field → joint edit admission → in-compile compensation → pose re-solve → the five lossless stages → archive, with `--device {mps,cpu,cuda}` and per-clip auto-configuration (task #1390, the PR #140 public TODO, operator-ordered)

## MANDATE

Operator 2026-09-03 verbatim: *"the full from raw video compress script and any related ports or
work necessary"* + *"continue with all"*. PR #140 (live) publicly promises three things after
submission — `submissions/semantic_joint_ctxmix/README.md` lines 60–67: a `--device` flag for the
solve stages; a full-pipeline mode from the raw video; per-video auto-configuration. The shipped
`compress.py` (f20b5e4baf) is the LOSSLESS TAIL only: it decodes the token field from the pinned
base archive `df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080` (180,456 B) and
replays FX5→DX2→GB1→LB1→AFR1 to the exact 180,002 B archive. Everything BEFORE that base — training,
token encode, joint edit admission, compensation, pose re-solve — ran as separate experiment scripts
on the Apple-silicon stack. This arm binds them into one driver. It is ALSO the substrate every
frontier successor needs (SCMDL's G/M refit retrains the HPAC model; the born-object burns retrain the
renderer): a pipeline that can regenerate the archive from the video is how a new object gets a row.

## WHAT WE ALREADY HOLD (inventory first — do not rebuild what exists; cite what you reuse)

- Training prefix custody: `/Volumes/VertigoDataTier/pact/ddm_pr130_train_20260809/{checkpoints,reports,caches,artifacts}`
  (renderer checkpoints BS2/BS8/BS16.pt, `hpac_init.pt`, gt cache `caches/gt_cache_600.pt`;
  reports pin init `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/artifacts/checkpoints/semantic_renderer_w96_b4_qat4_12k.pt`).
  `repro_repo/{code,recipe,scripts,artifacts,tests}` there is the credited PR #130 reproduction repo —
  INVENTORY its training entry points and recipe; that is the credited "PR #130/#135 training scripts".
- Our lifted trainer + MLX parity harness: `experiments/ddm_mx1_pr130_semantic_renderer.py`
  (080e20d8ac; argparse already has `--device`, `--scorer {upstream,proxy}`, `--bits`, caches, steps,
  lr, seed, microbatch policy). Born-object trainer family: `experiments/ddm_qbt1_qbflow_trainer.py`
  (83cb7d9b27). Token encode on Metal: custody `ddm_pr130_encode_tokens_metal_20260809` on Vertigo.
- Solve stages (each a real script with its own CLI — chain them, do not re-implement):
  joint edit admission `experiments/ddm_fcd1_incompile_schur.py` (1326458d5b) + the jg5 select/codes
  subcommands `experiments/ddm_jg5_pose_resolve_on_edited_renders.py` (63a509f928); in-compile
  compensation `experiments/ddm_qs5_resolve_compensation.py` (95b817fa98); zero-byte shipping-object
  pose re-solve `experiments/ddm_up2_shipping_pose_solve.py` (39a04c3214, `solve --gt-cache --axis`).
- Lossless tail: `submissions/semantic_joint_ctxmix/compress.py` — import its stage functions as a
  library (it is importable; `STAGE_PINS` at line 1350; `stage_runtime` at 1695). The packet tree is
  FROZEN (PR #140 live): the driver lives OUTSIDE it.
- Scorer ports: torch CPU authority-advisory (`upstream/evaluate.py --device cpu` is the only local
  score); MLX/Metal scorer adapters `src/tac/local_acceleration/mlx_scorer_adapters.py`,
  `metal_segnet_conv.py`; torch-MPS gradient path `tac.torch_mps_compat.patch_scorer_for_mps()`
  (MPS = training GRADIENT only, NEVER a score — CLAUDE.md train/authority split).

## SCOPE

1. **Driver:** `experiments/semantic_joint_ctxmix_pipeline.py` (+ package `src/tac/semantic_pipeline/`
   for the reusable stage contracts; thin CLI in experiments). Stages are typed, resumable
   (`--resume-from`, per-stage checkpoints, atomic tmp+rename, EMA shadow saved per the EMA
   non-negotiable), seeded, provenance-stamped (git sha, seed, config, upstream sha, hardware axis,
   per-stage input/output sha256 + bytes). Storage waterfall/preflight before any bulky stage
   (Vertigo 185 GiB free; AP ~30 GiB — refuse below 1.5 GiB free). Auto-clean hook for scratch.
2. **`--device {mps,cpu,cuda}`** for every solve stage: MPS via the existing patch; CPU torch; CUDA
   path written against the same torch code (cannot be executed here — mark it UNTESTED-HERE in the
   provenance, never claim it works). No `mps`-fallback default: the device is explicit; refuse if
   the requested device is unavailable (FORBIDDEN PATTERN: device-selection fallback chains).
3. **`--mode replay`**: reproduces the exact 180,002 B archive from the pinned base by calling the
   shipped tail — this is the REGRESSION GATE (sha must equal `cbb8d928…d405bf25`).
4. **`--mode full`**: from `upstream/videos/0.mkv` (GT decoded ONLY via `frame_utils.yuv420_to_rgb`,
   never PyAV rgb24): decode+cache → train renderer (init from the pinned checkpoint by default;
   `--from-scratch` optional) → encode token field → joint admission → compensation → pose re-solve →
   the five lossless stages → archive → receiver parse-back identity (inflate via the SHIPPED
   `inflate.py` copied to a temp runtime; 600/600 raw identity vs the driver's own render) → local
   advisory score through `upstream/evaluate.py --device cpu` on n≤8 pairs for the SMOKE (label
   `[macOS-CPU advisory]`, `score_claim=false`).
5. **Per-clip auto-config:** frame count, resolution, pair structure (seq_len=2 non-overlapping),
   measured from the input container; every per-video constant in the chained scripts becomes a
   config field derived from the probe, with the 0.mkv values as the regression check.
6. **Acceptance (the test command that must go green):**
   `.venv/bin/python -m pytest src/tac/tests/test_semantic_pipeline.py -q` covering: replay-mode sha
   identity (uses the retained base at
   `/Volumes/VertigoDataTier/pact/ddm_g8s_single_run_reproof/store_v2/retained/` or the g8r
   compliance stage — locate, pin, refuse if absent); full-mode n=2-pair smoke end-to-end on
   `--device cpu` producing a receiver-closed archive that inflates with the shipped inflate.py;
   auto-config probe on 0.mkv equals the pinned constants; resume-from-checkpoint bit-identity on a
   stage boundary; device-refusal on an unavailable device.
7. **The full n600 training run is NOT run in-session** — it is a multi-hour launch: emit a governed
   launch ticket (`tools/launch_witness_run.py`-style or the queue's launch surface) for MAIN with the
   exact argv, projected wall-clock and peak RSS (`tools/witness_memory_preflight.py` pattern), so the
   fresh-measurement archive row can be bought later. Say plainly in the memo: the full-mode archive
   will NOT be bit-identical to AFR1 and its score needs fresh measurement (PR #140's own wording).

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. `submissions/semantic_joint_ctxmix/` READ-ONLY (live PR tree; import from it,
  never edit it). No scorer weights or GT tables in any archive (rule 118; the receiver-consumption
  bijection). NO Modal fire; NO n600 scorer run (MAIN's lane) — emit fire orders/launch tickets.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes;
  `ruff` clean; tests green (`verify-landing` skill chain).
- ALWAYS KEEP THE PAYLOAD; every stage output (caches, checkpoints, token fields, overlays, archives)
  retained under `/Volumes/VertigoDataTier/pact/ddm_fpc1_full_pipeline_compress/` with sha256+bytes
  in the stage receipt. DETACHED >30-MIN COMPUTE rule applies (ONLY via `.venv/bin/python tools/launch_detached_process.py --output-dir <run_dir> --done-receipt <name> -- <cmd...>`; hand-rolled detaches are guard-blocked) to any stage you actually run.
- eval_roundtrip inside every training loss; EMA on every trainer; differentiable YUV6 patched
  before scorer construction (CLAUDE.md non-negotiables) — these are inherited from the reused
  scripts; VERIFY each is on in the chained config and record the verification.
- Never invent a CLI flag for a chained script: `grep add_argument` first; add a subset test.
- Do not fork the canonical helpers without the falling-rule rationale (Catalog #290).

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_g8s_single_run_reproof_20260903.md` — the pinned-base five-stage replay is the ONLY proven
  reproduction (4,140.9 s); the training prefix has never been replayed end-to-end. This arm
  builds the replay of the prefix; it does not claim bit-identity for it.
- `ddm_afr1_cpu_axis_timeout_verdict_20260902.md` — contest-CPU inflation exceeds 1,800 s; CUDA is
  required for authority; the driver's local score is ADVISORY only.
- `ddm_g8r_compress_adversarial_review_20260902.md` — exact-inventory census: any unmanifested file
  inside the packet dir makes compress.py refuse; keep the driver and its outputs OUTSIDE the tree.
- `ddm_wc2_qbr1_bug_wallclock_realization_audit_20260902.md` — WC2-F1: `EMA(..., warmup=True)`
  executes a different law than a sealed constant-decay config; the driver's training stage MUST
  resolve the EMA law from the config and construct the EMA to match it (sister arm ddm_wc3 owns the
  QBR1 cure; this arm owns the pipeline's own EMA construction — do not touch qbr1 files).
- `n205_oom_is_verdict_batch_spike_not_accum_loop_chunk_verdict_20260702` (memory) — chunk every
  full-P scorer forward (`--verdict-batch`, default 32); never batch 600 pairs at once.

## OPTIMAL FORM

- Family exemplar: the shipped lossless tail, reference
  `submissions/semantic_joint_ctxmix/compress.py` (commit f20b5e4baf; receipt
  `.omx/research/ddm_g8s_single_run_reproof_20260903.md`, exact 180,002 B rebuild) and the canonical
  resumable pipeline standard `experiments/pipeline.py` (profile-driven, seeded, provenance JSON) as
  the driver reference form; the training exemplar is the retained
  `ddm_pr130_train_20260809/reports/BS16.json` run (verdict PASS) on the lifted trainer
  `experiments/ddm_mx1_pr130_semantic_renderer.py` (commit 080e20d8ac).
- SCOPE reductions declared per row: the in-arm smoke is n=2 pairs, tens of steps (legal — SCOPE);
  MECHANISM reductions FORBIDDEN: no proxy loss in place of the scorer-aware loss, no skipped
  eval_roundtrip, no synthetic fixtures — the smoke runs on real 0.mkv frames.
- **PRIOR-LAW PREDICTION (falsifiable):** the driver's `--mode replay` reproduces the AFR1 archive
  bit-exactly (deterministic lossless tail); `--mode full` at n=2 produces a receiver-closed archive
  whose inflate identity holds 2/2 pairs. FALSIFIER: any replay sha mismatch, or an inflate identity
  failure on the driver's own render — count it plainly; it means a chained script carries hidden
  per-run state the pipeline must expose.

## DELIVERABLE

`.omx/research/ddm_fpc1_full_pipeline_compress_20260903.md` — typed rows: per stage {inputs sha,
outputs sha+bytes, device, seconds, resumable?, verified non-negotiables}, the acceptance test
transcript, the n600 launch ticket, the honest boundary (what is UNTESTED-HERE: cuda), RECALL
EVIDENCE, NEXT_IF_RESUMED, LIVE-HYPOTHESES, DEAD-ENDS. Commit via the serializer. Cite
`docs/operating_manual_craft_handoff.md`. End with the own-vehicle frontier line.
