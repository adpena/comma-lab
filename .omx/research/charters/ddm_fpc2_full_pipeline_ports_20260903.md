# ddm_fpc2_full_pipeline_ports — clear fpc1's six named full-mode blockers so `--mode full` runs end-to-end on real frames: parameterize QS5, add honored `--device` to FCD1/JG5/QS5/UP2, build the pipeline-side fresh-archive receiver (CPU-capable for smokes), resolve the F26 prefix gate, fix the trainer EMA/device contract, make target lineage an explicit config (task #1390; operator-ordered)

## MANDATE

Operator 2026-09-03: *"the full from raw video compress script and any related ports or work
necessary"* + standing GO. `ddm_fpc1_full_pipeline_compress_20260903.md` (arm fpc1, landed by MAIN
from its fallback bundle, commit `2296c6bad8`) built the real substrate — `src/tac/semantic_pipeline/
{pipeline,contracts}.py` + `experiments/semantic_joint_ctxmix_pipeline.py` + 5 tests, probe of
0.mkv (1,200 frames, 1164×874, 600 pairs), replay mode revalidated the exact 180,002 B AFR1 archive
— and then FAILED CLOSED at full mode with six INSTANCE-scope blockers
(`/Volumes/VertigoDataTier/pact/ddm_fpc1_full_pipeline_compress/full/RESULT.json`). This arm ports
each one. PR #140's public TODO promises exactly this.

## THE SIX BLOCKERS → THE SIX PORTS (each a typed row in the deliverable)

1. `QS5_INSTANCE_PINNED` — `experiments/ddm_qs5_resolve_compensation.py` embeds QS4/CP135 paths and
   N=600. PORT: extract its compensation kernel into `src/tac/semantic_pipeline/stages/compensation.py`
   with explicit `archive / clip-config / device / pair-scope` inputs; the old script keeps working
   (regression: its default run reproduces its retained receipt hash on the pinned inputs).
2. `SOLVE_DEVICE_FLAGS_ABSENT` — FCD1/JG5/QS5/UP2 CLIs expose no `--device`. PORT: add a REAL,
   honored `--device {cpu,mps,cuda}` to each (grep `add_argument` first; add the subset test); MPS via
   `tac.torch_mps_compat.patch_scorer_for_mps()` as the GRADIENT device only; refuse if unavailable
   (no fallback chain — FORBIDDEN PATTERN); every score/verdict computed on CPU torch.
3. `SHIPPED_RECEIVER_FRESH_ARCHIVE_REFUSAL` — the shipped `inflate.py` pins the AFR1 sha/size and
   refuses CPU. PORT: `src/tac/semantic_pipeline/receiver.py` — a parameterized COPY of the shipped
   runtime (cpr1 + f26) that accepts a fresh archive by declared sha/size, decodes on CPU for
   n≤8-pair smokes and on CUDA for full runs, and is byte-identity-tested against the shipped
   entrypoint on the AFR1 archive (n=2 pairs: identical raw output). The shipped tree stays FROZEN.
4. `PREFIX_RUNTIME_UNREACHABLE` — `runtime/f26_inflate.py`'s `F26_ADVISORY_PAIR_LIMIT` requires
   native-hpac while the runtime refuses every token decoder except python. PORT: in the pipeline
   receiver copy, make the prefix path consistent (python token decoder allowed under the pair
   limit); document the contradiction; never edit the shipped file.
5. `TRAINER_DEVICE_CONTRACT_MISMATCH` — `ddm_mx1_pr130_semantic_renderer.py` torch-smoke forces CPU
   and lacks an EMA shadow; mlx-train has MLX cpu/gpu semantics and constructs EMA only with a
   controller policy. PORT: a `train` stage contract with EMA ALWAYS constructed (`tac.training.EMA`,
   decay resolved through `ema_decay_run_geometry_v1`; executable law == sealed law — the WC2-F1
   lesson; coordinate with arm ddm_wc3's gate, do not duplicate its files), eval_roundtrip inside the
   loss, differentiable YUV6 patched before scorer construction, per-stage checkpoints with the EMA
   shadow, `--device` honored on the torch path; the MLX path stays a separate, labeled substrate.
6. `TARGET_CACHE_LINEAGE_CONFOUND` — the strict raw-video graph feeds one fresh DALI cache while the
   retained selected lineage used AV-like semantic targets + DALI carrier/HPAC/token targets. PORT:
   `target_lineage` becomes an explicit config field per target kind (`semantic: av|dali`,
   `carrier/hpac/token: dali`), default = the retained selected lineage, stamped in provenance;
   silent mixing refuses. Cite `#1142` via `ddm_cpu1_gt_lineage_attribution` (experiments/
   ddm_cpu1_gt_lineage_attribution.py) for the GT-decode fork.

## ACCEPTANCE (the commands that must go green)

- `.venv/bin/python -m pytest src/tac/tests/test_semantic_pipeline.py -q` (extend it): replay-mode
  sha identity (unchanged); receiver identity on AFR1 n=2 (pipeline receiver == shipped raw bytes);
  `--mode full --smoke --pairs 2 --device cpu` on REAL 0.mkv frames runs every stage end-to-end and
  yields a receiver-closed archive that the pipeline receiver inflates 2/2 identically to the
  driver's own render; resume-from-checkpoint bit-identity at a stage boundary; device refusal on an
  unavailable device; target-lineage refusal on silent mixing; each ported CLI's flag-subset test.
- `ruff` clean; two genuine review passes per `.py`; serializer commits with post-edit shas.
- A local advisory score on the 2-pair smoke via `upstream/evaluate.py --device cpu`, labeled
  `[macOS-CPU advisory]`, `score_claim=false` — present in the memo as a smoke receipt, never a score.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY; `submissions/semantic_joint_ctxmix/` READ-ONLY (live PR #140 tree).
- NO n600 training, NO scorer-lane run, NO Modal, NO Metal from the arm — emit the governed n600
  launch ticket for MAIN (argv, projected wall-clock, peak RSS via the `witness_memory_preflight`
  pattern) as fpc1 specified.
- ALWAYS KEEP THE PAYLOAD under `/Volumes/VertigoDataTier/pact/ddm_fpc2_full_pipeline_ports/`.
- DETACHED >30-MIN COMPUTE ONLY via `.venv/bin/python tools/launch_detached_process.py --output-dir <run_dir> --done-receipt <name> -- <cmd...>`.
- Never invent a flag; never fork a canonical helper without the falling-rule rationale.
- File ownership: you own `src/tac/semantic_pipeline/**`, `experiments/semantic_joint_ctxmix_pipeline.py`,
  `src/tac/tests/test_semantic_pipeline.py`, and ADDITIVE `--device`/kernel-extraction edits to
  the four solve scripts. Do not edit `experiments/ddm_qbr1_*`, `tac/training.py`, or
  `confound_gates.py` (arm ddm_wc3 owns them).

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_fpc1_full_pipeline_compress_20260903.md` — the six blockers and the replay-only proof.
- `ddm_wc2_qbr1_bug_wallclock_realization_audit_20260902.md` — WC2-F1 EMA law mismatch: the
  trainer port's EMA must execute the sealed law.
- `ddm_g8r_compress_adversarial_review_20260902.md` — exact-inventory census: nothing new inside
  the packet dir.
- `ddm_afr1_cpu_axis_timeout_verdict_20260902.md` — CPU inflation of the full archive exceeds
  1,800 s: the CPU receiver is for ≤8-pair smokes only.
- `n205_oom_is_verdict_batch_spike_not_accum_loop_chunk_verdict_20260702` (memory) — chunk every
  full-P scorer forward.
- fpc1's note: the delegated codex-in-codex implementation failed on an app-server permission error
  twice; implement directly.

## OPTIMAL FORM

- Family exemplar: fpc1's landed substrate, reference `src/tac/semantic_pipeline/pipeline.py`
  (commit 2296c6bad8, receipt `.omx/research/ddm_fpc1_full_pipeline_compress_20260903.md`) and the
  shipped lossless tail `submissions/semantic_joint_ctxmix/compress.py` (f20b5e4baf); trainer
  exemplar `experiments/ddm_mx1_pr130_semantic_renderer.py` (080e20d8ac) with the retained
  `ddm_pr130_train_20260809/reports/BS16.json` PASS run.
- SCOPE reductions: n=2-pair smokes, tens of steps (legal). MECHANISM reductions FORBIDDEN: no
  proxy loss, no skipped eval_roundtrip, no EMA-less trainer, no synthetic frames, no
  device-fallback chain, no lineage mixing.
- **PRIOR-LAW PREDICTION (falsifiable):** with the six ports, the 2-pair full-mode smoke on CPU
  completes end-to-end with 2/2 receiver identity, and replay mode still reproduces
  `cbb8d928…d405bf25`. FALSIFIER: any stage still fails closed for a reason outside the six
  named blockers — count it plainly as a seventh.

## DELIVERABLE

`.omx/research/ddm_fpc2_full_pipeline_ports_20260903.md` — the six port rows (before/after,
tests), the smoke transcript with stage timings and the advisory receipt, the n600 launch ticket,
RECALL EVIDENCE, NEXT_IF_RESUMED, LIVE-HYPOTHESES, DEAD-ENDS. Commit via the serializer. Cite
`docs/operating_manual_craft_handoff.md`. End with the own-vehicle frontier line.
