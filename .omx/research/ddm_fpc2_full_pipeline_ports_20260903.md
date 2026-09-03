# ddm_fpc2 full-pipeline ports — receiver-closed n=2 smoke PASS

## Verdict

The six FPC1 instance blockers are cleared for the charter's bounded real-video path.  The exact
acceptance command ran all five stages on the first two pairs of real `upstream/videos/0.mkv`, wrote a
fresh 180,496 B archive, and produced byte-identical driver and independent pipeline-receiver renders.
The prior-law prediction therefore passed at n=2: no seventh blocker appeared in the bounded stage
graph.  This did **not** run n=600, move the score pointer, or establish a population-quality result.

Live board authority supersedes the stale frontier paragraph in `_common_contract.md`: AFR1 remains
the effective frontier at S 0.14797617125559104, 180,002 B, `[contest-CUDA T4, n600]`.

## The six ports

| Blocker | Before | Port now executed | Verification |
|---|---|---|---|
| `QS5_INSTANCE_PINNED` | QS5 embedded the QS4/CP135 archive, clip, device, and n=600 instance. | `semantic_pipeline/stages/compensation.py` owns a generic integer compensation kernel and typed `archive / clip / device / pair_ids` request. The legacy QS5 entrypoint accepts the same explicit inputs and delegates its connective-support kernel to the extracted implementation. | Three retained real QS4 site ledgers produced identical legacy/ported selected vectors: 45 selected sites, digest `4168e52e…b3b2b65`. The retained pinned final receipt remains `d72fafec…e4634f5f4`; no FPC2 compensation was falsely admitted, so the candidate bytes remain unchanged at this stage. |
| `SOLVE_DEVICE_FLAGS_ABSENT` | FCD1, JG5, QS5, and UP2 could not name a device. | Each CLI exposes `--device {cpu,mps,cuda}`, binds exactly that device, patches scorer compatibility before construction on MPS, and refuses an unavailable request. JG5/UP2 proposal Jacobians use the selected gradient device; realized admission and every score/verdict remain frozen CPU Torch. QS5's integer/realized score path and FCD1's non-solver modes remain CPU authority because they contain no accelerator-gradient leg. | Parser-subset tests cover all four CLIs; unavailable-device refusal is executed. CPU is exercised here. MPS/CUDA are code-routed but untested on this host and are not promoted as measured axes. |
| `SHIPPED_RECEIVER_FRESH_ARCHIVE_REFUSAL` | Shipped inflate pins AFR1 sha/size and rejects CPU. | `semantic_pipeline/receiver.py` copies CPR1/F26 into the run store, records every copied source hash, binds a declared archive sha/size, and permits CPU only for n≤8 or CUDA for full runs. Shipped files are untouched. | Fresh archive inflated 2/2. The direct driver (retained quantized compiler state, no fresh-archive parse) and independent receiver (fresh-archive parse-back) each wrote 12,208,032 B raw with sha `c7452f74…6692e3c`. On AFR1 n=2, the pipeline copy and an independently invoked unmodified shipped reference path both wrote sha `8ef6939b…99497`. |
| `PREFIX_RUNTIME_UNREACHABLE` | F26 required native HPAC under its advisory pair limit while the enclosing runtime allowed only the Python token decoder. | The run-local receiver copy makes the prefix contract internally consistent by permitting the Python token checkpoint under the pair limit; only the copied runtime is patched. | AFR1 n=2 receiver-reference identity and fresh-archive 2/2 driver/receiver identity both passed. `upstream/` and `submissions/semantic_joint_ctxmix/` have no diff. |
| `TRAINER_DEVICE_CONTRACT_MISMATCH` | The prior torch smoke forced CPU and had no unconditional EMA; the MLX contract was a different substrate. | The torch train stage uses real decoded frames, frozen real SegNet/PoseNet, differentiable YUV6 patched before scorer construction, and camera-uint8/bilinear eval roundtrip inside the loss. It always constructs `tac.training.EMA`; decay 0.31622776601683794 is resolved from `ema_decay_run_geometry_v1` for two updates and seed fraction 0.1. Every step writes a distinct atomic checkpoint containing live weights, optimizer, RNG, and EMA shadow. MLX remains separate and labeled. | Real n=2 CPU loss moved 0.0164408106 → 0.0164201129 in two steps; pose MSE moved 9.9391e-7 → 8.7319e-7. Stage checkpoints are `ef7b94af…e96aed7` and `60e6e0c9…6982a3`; resume reused them bit-identically. EMA payload `7c334991…50042` and deterministic archive repeat `fa048304…2b7848` were retained. |
| `TARGET_CACHE_LINEAGE_CONFOUND` | One implicit cache could silently stand for incompatible AV and DALI target objects. | `TargetLineage` explicitly types `semantic=av|dali` and fixes `carrier/hpac/token=dali`; the default is the retained selected lineage (`av/dali/dali/dali`). Invalid silent mixing refuses and the complete mapping is stamped in stage and run provenance. | Mixing-refusal test passed. The train receipt binds real AV semantic frames plus DALI cache sha `382d7dfe…40195`. This follows the GT-decode fork identified by task #1142 in `experiments/ddm_cpu1_gt_lineage_attribution.py`. |

### QS5 receipt-hash boundary

The charter asks the modified legacy script to reproduce the old receipt's complete file hash.  That
receipt self-censuses the legacy script source, so any mandated `--device` or delegation edit changes
the provenance field even when the computed compensation is identical.  Returning the old receipt
after a source change would be false provenance.  The admissible regression is therefore the exact
selected-value identity above, with the old receipt and its archive left untouched and hash-bound.
This is a bounded incompatibility in the requested hash criterion, not an algorithmic QS5 failure.

## Smoke receipt

Executed acceptance command:

```text
.venv/bin/python experiments/semantic_joint_ctxmix_pipeline.py \
  --mode full --smoke --pairs 2 --device cpu
```

The resumable replay of that command is retained as
`/Volumes/VertigoDataTier/pact/ddm_fpc2_full_pipeline_ports/full/CLI_TRANSCRIPT.txt`, sha
`a8fcf6cf…8198681`. The matching `RESULT.json` has the same digest because stdout is the exact sorted
manifest. A clean, non-resumed repeat is retained at
`/Volumes/VertigoDataTier/pact/ddm_fpc2_full_pipeline_ports/review_v4_fresh_cli.txt` (sha
`f2f32f7f…dd3e1e5`); its stage receipts preserve these timings:

| Stage | Seconds | Retained output |
|---|---:|---|
| scorer-aware train | 15.569 | `train/TRAIN_RESULT.json`, EMA, quantized state, two step checkpoints, archive + repeat |
| QS5 kernel regression | 0.128 | `qs5_kernel_regression.json`, unchanged stage archive |
| direct driver render | 1.283 | `direct_driver_render_n2.raw` + driver receipt |
| independent receiver render | 2.150 | `receiver_render_n2.raw` |
| actual `upstream/evaluate.py --device cpu` | 5.675 | report, stdout, `EVALUATE_RESULT.json` |

All materialized payloads are retained below
`/Volumes/VertigoDataTier/pact/ddm_fpc2_full_pipeline_ports/`. The fresh archive is 180,496 B, sha
`fa048304…2b7848`; the deterministic archive repeat is identical. Driver and receiver raw output is
12,208,032 B, sha `c7452f74…6692e3c`, on both paths. The compiler's quantized semantic state is also
retained (278,957 B, sha `222dd6ef…14eaa1`) rather than discarded after hashing.

The evaluator smoke reports `d_seg=0.00030772`, `d_pose=0.00010887`, and `rate=0.00480740`.
Recomputed from those printed eight-decimal components,
`100*d_seg + sqrt(10*d_pose) + 25*rate = 0.1839524542323636`. This is a contiguous-prefix n=2
plumbing receipt on `[macOS-CPU advisory]`, with `score_claim=false`; it is not a score row, quality
estimate, contest-axis inference, or evidence that the fresh archive improves AFR1.

Replay mode separately retained the exact source AFR1 archive at 180,002 B, sha
`cbb8d928…d405bf25`. The fresh archive preserves the carrier, HPAC, and token-tail stream hashes while
replacing only the EMA semantic stream.

## Governed n=600 ticket

`/Volumes/VertigoDataTier/pact/ddm_fpc2_full_pipeline_ports/full/governed_n600_launch_ticket.json`
(sha `678f8384…e5cb01d`) is `QUEUED-WITH-FIRE-ORDER`, owner `MAIN`, consumer store
`/Volumes/VertigoDataTier/pact/ddm_fpc2_full_pipeline_ports`.

It is deliberately not fireable yet. The present train stage is bounded to n≤8 and has no chunked
n=600 target materializer/trainer. MAIN must land that same-contract consumer, rerun the live
system-aware memory preflight, and claim the single scorer lane. The ticket's projection is 30.27 GiB
(32,502,165,012 B) from the measured #205 `witness_memory_preflight` law for n=600, 384×512,
in-features 96, micro-batch 2, verdict batch 32. It is a projection, not live admission. The recalled
wall-clock bracket is 14,400–259,200 seconds. The queued argv uses the detached launcher and CUDA,
because the full receiver contract accepts CUDA rather than the smoke-only CPU path. No n=600,
scorer-lane, Modal, Metal, or paid job was launched by this arm.

## RECALL EVIDENCE

I searched the full `.omx/research/` corpus and arm receipts by content for `fpc1`, `QS5`, `FCD1`,
`JG5`, `UP2`, `fresh archive`, `receiver`, `EMA`, `target lineage`, `AV`, `DALI`, `n205`, and verdict
chunking; searched `CANONICAL_RESEARCH_INDEX*`, the `sub015_DAG_*` FEED graph, specifications, and the
task/state surfaces; and ran `tools/list_canonical_equations.py --json`.

Beyond the charter seeds, recall found the executable `ema_decay_run_geometry_v1` evaluator at
`src/tac/canonical_equations/evaluators.py`, the `scorer_input_cache_hash_identity_v1` registry row,
the retained exact Python-decoded token field (`tokens.u8`, sha `cc10a7b0…36efb`), the retained DALI
GT cache (sha `382d7dfe…40195`), and task #1142's explicit CPU AV versus CUDA DALI fork in
`experiments/ddm_cpu1_gt_lineage_attribution.py`. These findings changed the build: the trainer calls
the canonical EMA law instead of restating it, archive/receiver construction binds the retained token
object, and lineage is typed per target rather than per run. The n205/OOM recall changed the n=600
handoff from an apparently runnable command into a blocked ticket that requires a chunked consumer and
fresh system-aware admission.

The implementation follows `docs/operating_manual_craft_handoff.md`: archive identity, receiver
identity, EMA-law execution, lineage, and advisory arithmetic were re-derived from primary artifacts;
bounded or untested claims remain labeled.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_semantic_pipeline.py -q` — 8 passed. The fresh run
  exercised real clip probing, n=2 full-mode receiver closure, stage resume, exact AFR1 replay,
  independent AFR1 receiver reference, unavailable-device refusal, lineage refusal, and four CLI
  device subsets.
- Ruff over every owned/edited Python file — clean.
- `git diff --check` over every owned/edited file — clean.
- Two genuine review-tracker passes per Python file and serializer post-edit hashes are the landing
  gate; no review override is used.
- `git diff -- upstream submissions/semantic_joint_ctxmix` — empty.

## Landing custody

The mandated serializer reached its commit step but this managed session cannot write the shared Git
index/object store (`unable to create temporary file: Operation not permitted`). It did not touch the
index and emitted a verified isolated commit bundle under the arm's SSD receipt store. This is a
custody-safe landing blocker, not a claim that live HEAD contains the work; the final fallback receipt
supersedes any earlier attempt and names the exact bundle, format patch, base HEAD, content hashes, and
isolated commit.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: the final `commit_serializer_fallbacks` receipt directory named in the arm handoff; fire trigger: a Git-write-enabled operator verifies the bundle against its recorded base HEAD, imports the isolated `serializer-fallback` commit, and confirms the 15-file tree plus tests before declaring the landing live.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `/Volumes/VertigoDataTier/pact/ddm_fpc2_full_pipeline_ports`; fire trigger: land a crash-resumable, per-chunk-checkpointed n=600 target materializer/trainer using this exact stage contract, rerun `witness_memory_preflight --system-aware`, claim the scorer lane, then execute the retained CUDA detached-launch ticket.

## LIVE-HYPOTHESES

- A streamed n=600 implementation can preserve the bounded train-stage semantics because the EMA law,
  target lineage, roundtrip, archive compiler, and receiver are already explicit; this is plausible
  if per-chunk gradient accumulation and checkpoint state reproduce the same update ordering.
- The small n=2 descent suggests the scorer-aware semantic update is wired and locally usable, but only
  a stratified/population run can say whether it improves the full objective; n=2 is deliberately not
  evidence about sign or size at n=600.

## DEAD-ENDS

- Treating the n=2 advisory value as a contest score is closed: it is a contiguous-prefix plumbing
  smoke, its rate denominator is the full source while only two candidate pairs are present, and
  `score_claim=false`.
- Re-running the unmodified shipped receiver on a fresh archive is closed: its sha/size and CPU refusals
  are intentional. The parameterized run-local copy is the valid receiver surface; shipped code stays
  frozen.
- Claiming exact legacy QS5 whole-receipt hash reproduction after editing the source is closed: the
  receipt includes that source hash. Selected-value identity is real; laundering old provenance is not.
- Firing the current n=600 argv before the chunked consumer and scorer-lane claim is closed: the bounded
  trainer refuses n>8 and the ticket records both missing prerequisites.

**OWN-VEHICLE FRONTIER: S 0.14797617125559104 @ 180,002 B `[contest-CUDA T4, n600]` (AFR1, unchanged; this arm moved no exact pointer).**
