# ddm_fpc3_chunked_n600_trainer — the from-raw-video pipeline's population run made READY: a crash-resumable CHUNKED n=600 training stage (all 600 pairs, seeded stratified chunk order, per-chunk checkpoints with the EMA shadow), a numeric memory preflight receipt, and the governed launch ticket MAIN fires after the QBR1 burn releases the Metal slot (task #1390; operator-ordered "full from raw video compress script")

## MANDATE

Operator 2026-09-03: *"the full from raw video compress script and any related ports or work necessary"*
+ standing GO. fpc2 (`ddm_fpc2_full_pipeline_ports_20260903.md`, landed by MAIN) cleared the six blockers
and proved the n=2 full mode end-to-end: fresh archive 180,496 B, driver/receiver raw identical
(12,208,032 B), advisory n=2 S 0.18395 `[macOS-CPU advisory]` (a contiguous-prefix PLUMBING smoke —
never a score). Its NEXT_IF_RESUMED owes ONE thing before the population run: "land the crash-
resumable chunked n=600 trainer, rerun system-aware memory preflight, claim the scorer lane, then
execute the retained CUDA launch ticket." The Metal slot is OWNED for ~18 h by the QBR1 six-cell burn
(the frontier-critical discriminator); this arm builds everything so MAIN can fire the moment it frees.

## SCOPE

1. **Chunked population trainer** in `src/tac/semantic_pipeline/stages/train.py` (the bounded n=2 path
   stays as the smoke): all 600 pairs; chunk order = seeded RANDOM/stratified permutation via
   `tac.subset_selection.select(..., mode=...)` (never a video-order prefix — the hook refuses undeclared
   prefixes; the n=2 smoke carries `SUBSET_SELECTION_OK` waivers, the population path must not need one);
   per-chunk checkpoints written atomically (tmp+rename) with the EMA SHADOW (not live weights), cursor,
   RNG state, and every cfg key the resume path needs; `--resume-from` continues bit-faithfully (prove it:
   interrupt-after-chunk-k + resume == uninterrupted, on a 3-chunk smoke — the QBR1 resume-smoke pattern);
   the verdict/scorer forward chunked (`--verdict-batch`, default 32; never 600 at once — the #205 OOM law).
2. **Semantics preserved from the bounded path**: EMA law resolved through `ema_decay_run_geometry_v1`
   (executable == sealed; the wc3 strict gate `check_ema_executable_law_matches_sealed_law` must stay
   at live count 0 — no literal `warmup=`), explicit target lineage, eval_roundtrip in the loss,
   differentiable YUV6 before scorer construction, update order, archive construction, receiver contract.
3. **System-aware memory preflight receipt**: project peak RSS from the real config (the resume smoke
   measured 45.5 GB peak RSS at B=16 with scorers — cite it) using the `tools/witness_memory_preflight.py`
   pattern; refuse projections > 0.70×RAM; write a numeric receipt.
4. **Governed launch ticket** for MAIN: exact argv through `tools/launch_detached_process.py` (with
   `--derive-resource-budgets --measured-peak-rss-gib <measured> --measured-thread-need <n>
   --walltime-cap-s <derived>` and a `--done-receipt`), projected wall-clock from the measured
   seconds/update × updates, storage projection (Vertigo for retention; AP reserve 8 GiB respected), the
   scorer-lane claim id placeholder, and the fire trigger "Metal slot released by the QBR1 burn". No
   launch from the arm.
5. **Tests**: extend `src/tac/tests/test_semantic_pipeline.py`: chunk-order is seeded-random (not prefix);
   resume identity on the 3-chunk smoke; memory-preflight refusal path; ticket schema. Acceptance:
   `.venv/bin/python -m pytest src/tac/tests/test_semantic_pipeline.py -q` green; ruff clean.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY; `submissions/semantic_joint_ctxmix/` READ-ONLY (live PR #140 tree).
- NO n600 training, NO Metal/MPS use (the burn owns the device — a second Metal fire violates the
  one-Metal-fire law), NO scorer-lane run, NO Modal from the arm. The 3-chunk resume smoke runs on
  CPU at n≤6 pairs.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD under `/Volumes/VertigoDataTier/pact/ddm_fpc3_chunked_n600_trainer/`.
- DETACHED >30-MIN COMPUTE ONLY via `.venv/bin/python tools/launch_detached_process.py --output-dir <run_dir> --done-receipt <name> -- <cmd...>`.
- File ownership: `src/tac/semantic_pipeline/**`, `experiments/semantic_joint_ctxmix_pipeline.py`,
  `src/tac/tests/test_semantic_pipeline.py`. Do not touch `experiments/ddm_qbr1_*`, `confound_gates.py`.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_fpc2_full_pipeline_ports_20260903.md` — the n=2 result is a prefix plumbing smoke; firing the
  n600 ticket before a chunked consumer + lane claim is CLOSED (its own dead-end list).
- `ddm_wc2_qbr1_bug_wallclock_realization_audit_20260902.md` + `ddm_wc3_qbr1_ema_law_cure_20260903.md`
  — the EMA-law mismatch class and its strict gate.
- `n205_oom_is_verdict_batch_spike_not_accum_loop_chunk_verdict_20260702` (memory) — chunk the verdict.
- `prefix_bias_sign_inverts_between_seg_and_pose_20260803` (memory) — pose prefixes read 2.54–4.21×
  harder than the population: the population order must be seeded-random/stratified, never a prefix.
- `feedback_never_launch_non_resumable_per_stage_checkpoints_20260627` (memory) — loop-end-only saving
  is FORBIDDEN; EMA shadow per checkpoint.

## OPTIMAL FORM

- Family exemplar: the QBR1 burn's crash-resumable trainer with its PASSED resume smoke, reference
  `experiments/ddm_qbr1_born_fairform_burn_prep.py` (sealed commit 106d0dd0a094; receipt
  `/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/resume_smoke/RESUME_SMOKE_RESULT.json`, 4/4 equal),
  and fpc2's bounded trainer `src/tac/semantic_pipeline/stages/train.py` (commit 4608f607b8, landed).
- SCOPE reductions: the resume smoke is 3 chunks at n≤6 pairs on CPU (legal). MECHANISM reductions
  FORBIDDEN: no prefix chunk order, no EMA-less checkpoints, no loop-end-only saving, no unchunked
  verdict forward.
- **PRIOR-LAW PREDICTION (falsifiable):** with explicit cursor/RNG/EMA state, interrupt-after-chunk-k +
  resume reproduces the uninterrupted 3-chunk run bit-identically (live, EMA, archive). FALSIFIER: any
  hash differs — count it plainly; it names hidden per-run state.

## DELIVERABLE

`.omx/research/ddm_fpc3_chunked_n600_trainer_20260903.md` — the chunked stage contract, the resume-
identity receipt, the memory-preflight receipt, the governed launch ticket (exact argv), RECALL EVIDENCE,
NEXT_IF_RESUMED, LIVE-HYPOTHESES, DEAD-ENDS. Commit via the serializer. Cite
`docs/operating_manual_craft_handoff.md`. End with the own-vehicle frontier line.
