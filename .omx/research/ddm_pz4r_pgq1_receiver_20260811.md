# DDM PZ4R — PGQ1-conditioned receiver

**Verdict: DIRECT-RECEIVER-BYTE-CLOSED; REALIZED-SCORER-PENDING.** The residual-free receiver is real and causally consumes PGQ1, but it has not produced a full rendered n600 video or any scorer result. The sole scorer is occupied by PS135, so PZ4R is `QUEUED-WITH-A-FIRE-ORDER` rather than scored.

## Result first

| Surface | Result | Authority |
|---|---:|---|
| Selected map | `target_quadratic_previous_f10_q20` | full-n600 coefficient-fit selection; scorer-free |
| Counted PZ4R carrier | **20,869 B**, SHA-256 `99b20f780ff604536691d49c3bd52f4a6737f46ea03a81f89805c79b7b6e1eb5` | retained raw bytes |
| Brotli-q9 carrier | **18,969 B**, SHA-256 `064f0f13eb4bcd3af7069dece734f2d37352959647fec14aa8f040ef048d5440` | retained real coder output |
| Candidate archive | **183,137 B**, SHA-256 `c408adf9101bb19a363039a5e0f7185aabce8f31edb6787e2deaf6d0fe6738f4` | byte-closed; not scorer-measured |
| Repeat archive | **183,137 B**, same SHA-256 | byte-identical repeat |
| Delta versus LC2 | **−4,089 B** from 187,226 B | exact archive bytes |
| Delta versus the PGQ-only envelope | **+15,132 B** from 168,005 B | exact archive bytes; envelope reconciled away |
| Ordinary coefficient-code R² | **0.4860878253721611** | full 7,200-code in-sample selection surrogate |
| Coefficient MAE / RMSE | **277.60527777777776 / 372.72068220645394** | ordinary signed, non-circular code error |
| Exact coefficient codes | **8 / 7,200** | intentionally non-exact receiver |
| Endpoint-wrap crossings | **0** | selected f10 candidate |
| Retained payload records | **140** | all candidates plus repeats and causal proof |

Primary receipt: `/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/direct_v6/PZ4R_RESULT.json`, 50,470 B, SHA-256 `85c24a269b2380bb3ddf1d1a2ea39779d7541d0ea266f289b280d6dbe876c6c3`.

Shipping-shaped archive and runtime: `/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/direct_v6/submission/`.

The exact outer fixed term is 164,168 B, so this row is `164,168 + 18,969 = 183,137 B`. The earlier 168,005 B object removed the whole CPR1 packet and substituted PGQ1 without supplying any way to generate LC2 basis/coefficients. It was explicitly a rate envelope, not a receiver candidate. The real direct receiver retains the 13,101-byte LC2 basis and counts a fixed-point target-to-coefficient map; that is why the measured archive is 15,132 B above the envelope.

## What the receiver does

The direct rung is:

`strict PGQ1 (600×6) decode → decoded-output fixed-point features → counted residual-free map → counted LC2 basis → existing LC2 renderer`

The PZ4R section contains the exact LC2 basis component, exact PGQ1 bytes, predictor state, coefficient scales, and a hash of the predicted coefficient codes. It contains no CPR1 packet, coefficient residual, scorer weights, PoseNet, ground-truth table, or hidden per-frame coefficient table.

The strict PGQ1 path consumes all three parts of the gauge—codes, scale, and compensation matrix—by reconstructing the actual `(600,6)` output object before fixed-point prediction. A mutation of one retained PGQ code changed predicted coefficients and changed the retained rendered pair-0 slave-frame bytes:

- selected frame tensor SHA-256: `a2e666100a52e3e13f8727dd57f7961349e89c63971c47cc9927661f9bd83978`
- mutated frame tensor SHA-256: `a5fa64b5a27aac417b1be5f86e444fe85ebe615ca94d5c7c3bd96b16d670546b`

The public-runtime parse reconstructed basis shape `(12,3,24,32)` and coefficient shape `(600,12)` byte-identically to the direct parser. The basis SHA-256 is `7a6576e991a068e084ffc12f6377b9bfcc00fd2529eb8df27c424921f3c3933b`; the predicted coefficient-array SHA-256 is `e3234c9092fb649f58d40b89494be6388bd73a2bc2eac4cbc0f2d7d120d015da`. Semantic, HPAC, temporal, and token bytes parse back unchanged from LC2.

The dependency manifest was regenerated from the actual tree rather than copied from LC2. It binds eight Python modules plus one shell entrypoint, the exact nine source hashes, four third-party dependency families, and the counted archive. Linux closure remains queued and is not inferred from the local parser check.

## Sealed scorer-free sweep

Selection used ordinary decoded-coefficient error because LC2 renders signed coefficient amplitudes linearly. Signed 12-bit storage is not a circular physical coordinate: `2047` and `-2048` are 4,095 renderer units apart, not one.

| Candidate | Archive B | Carrier q9 B | Ordinary R² | Wrap crossings |
|---|---:|---:|---:|---:|
| `target_f8_q20` | 181,788 | 17,620 | 0.2869545478583707 | 0 |
| `target_f10_q20` | 181,791 | 17,623 | 0.2903439713627972 | 0 |
| `target_f12_q20` | 181,787 | 17,619 | 0.2902746049990653 | 0 |
| `target_f16_q20` | 181,786 | 17,618 | 0.2903246408734730 | 0 |
| `target_quadratic_f8_q20` | 182,757 | 18,589 | 0.4758967136119475 | 1 |
| `target_quadratic_f10_q20` | 182,685 | 18,517 | 0.4734523703339607 | 1 |
| `target_previous_f8_q20` | 182,303 | 18,135 | 0.2864620589041826 | 0 |
| `target_previous_f10_q20` | 182,284 | 18,116 | 0.2903559895292321 | 0 |
| `target_previous_f12_q20` | 182,286 | 18,118 | 0.2901924271702376 | 0 |
| `target_previous_f16_q20` | 182,254 | 18,086 | 0.2902319400201164 | 0 |
| `target_quadratic_previous_f8_q20` | 183,221 | 19,053 | 0.4840700984870227 | 1 |
| **`target_quadratic_previous_f10_q20`** | **183,137** | **18,969** | **0.4860878253721611** | **0** |

All twelve raw carriers, q9 streams, model packs, members, archives, parse arrays, and deterministic repeats remain under `direct_v6/retained/candidates/`. The initial unversioned store and `direct_v2` through `direct_v5` also remain retained. They are apparatus history, not candidate authority.

## Measured, not measured, and authority boundary

Measured on `[macOS-CPU scorer-free receiver build]`:

- strict PGQ1 parse and retained-output equality;
- full `(600,12)` prediction, ordinary coefficient metrics, and winner selection;
- exact counted carrier/archive bytes and SHA-256 values;
- byte-identical carrier/archive repeats;
- archive grammar and semantic/HPAC/temporal/token parse-back;
- direct/public receiver array equality;
- coefficient parity with LC2 is false: only 8 of 7,200 decoded codes are equal;
- one PGQ mutation changing both predicted coefficients and retained rendered pair-0 bytes;
- completed-resume revalidation of source bindings, every retained record, runtime tree, manifest, stage receipts, and final receipt;
- `ruff`, `py_compile`, and **9 passing tests**;
- two review-apparatus passes on every new Python file and an independent final review with no findings.

Not measured:

- a full n600 rendered raw video;
- `d_seg` or `d_pose` through the frozen scorers;
- a macOS advisory score, contest-CPU score, or contest-CUDA score;
- decode wall time under the 30-minute contest budget;
- Linux dependency/runtime closure;
- full rendered-frame parity with LC2; it is not expected, but was not measured and is not claimed.

Therefore this work moved no exact score and did not move the canonical pointer. The candidate is 3,115 B below the current 186,252 B cp135 archive, but no score comparison follows from rate alone.

## RECALL EVIDENCE

The bounded recall pass queried research, equations, DAG, task, and docs stores with these exact concepts:

1. `PGQ1 target conditioned receiver PoseNet output consumer`
2. `stored target pose sidecar FiLM conditioned renderer`
3. `PoseNet six outputs inverse renderer direct consumption`
4. `LC2 CPR1 carrier output gauge realization gap`
5. `pose target bank 600 6 receiver`
6. `Quantizr stored target sidecar joint descent`
7. `PGQ1 receiver pose output target conditioned`
8. `realization gap actual S R GT`
9. `neutral gray pose inverse render`
10. `receiver forward parity pose carrier`

The canonical equation registry supplied `gap_decomposition_against_demonstrated_floor_v1`, `pose_sqrt_concave_coupling_sidecar_v1`, and the receiver admission/realization laws; no PGQ/PZ3/PZ4-specific registry entry was found in that bounded scope.

Recall changed the execution plan in three load-bearing ways:

- PGQ1 was confirmed to reconstruct only LC2's banked PoseNet outputs, not frames, basis, or coefficients. A parser swap into LC2 could not render.
- PZ3's negative closes only the exact-residual frozen-basis formulation. Its 194,120 B archive and 9,767 B residual do not close a residual-free predictor that deliberately changes frames.
- The G91 prohibition closes treating PoseNet outputs directly as a physical SE(3) warp control. It does not prohibit a counted map through LC2's real basis/renderer.

No existing lawful scorer-free PoseNet-output-to-frame consumer was found in the bounded recalled stores. The smallest real direct rung was therefore the counted residual-free coefficient map implemented here, and the 168,005 B expectation was withdrawn before scoring.

## Review corrections and retained failed attempts

Independent review caught two candidate-invalidating classes before any scorer launch:

- `direct_v3` selected f8 using circular 12-bit error, understating a 4,095-unit renderer error as one. That archive is withdrawn. Ordinary non-circular error selects f10.
- early completed-resume logic trusted stage booleans and stale source/runtime metadata. The final v2 schema re-hashes live sources, receipts, all retained payloads, runtime files, manifest, selected winner, and public parse; it also rejects truthy non-boolean stage flags.

The copied LC2 dependency manifest was also withdrawn because it described the old module denominator and hashes. `direct_v6` regenerates and validates the manifest from the shipped tree.

Nothing from the initial unversioned attempt or `direct_v2`–`direct_v5` was deleted. Their bytes are retained so the correction chain is auditable, but none should be scored or promoted.

## Borrowed substrate and original work

Borrowed and counted where applicable:

- LC2 semantic, HPAC, temporal, and token streams;
- LC2's exact 13,101-byte basis component and renderer;
- LC2/RC1 archive grammar, ANS token path, and dependency entrypoint;
- PZ3's fixed-point predictor serialization and basis parser primitives;
- PZ4P's exact PGQ1 winner bytes.

Original to this rung:

- strict scorer-free PGQ1 runtime decode;
- PZ4R residual-free carrier grammar and predicted-code binding;
- decoded-output feature path that makes PGQ codes, scale, and compensation causal;
- checked Python-integer accumulation before int64 conversion;
- full retained fractional-bit/feature sweep and ordinary-error selection;
- LC2 runtime dispatch patch, truthful runtime-manifest regeneration, public parse-back, mutation proof, immutable resume validation, and regressions.

Implementation commit: `c7b9387b9638a490194a0be121fdca0e624a0759` (`ddm pz4r: byte-close residual-free PGQ1 receiver [no-triality] [p0-ledger-ok]`). Post-edit hashes:

- `experiments/ddm_pz4r_pgq1_receiver.py`: `49273606011bb518ecbbbdc416ace204c75d3896c1ec137be534183bb6a752ea`
- `experiments/ddm_pz4r_runtime/pose_gauge_receiver.py`: `f472a6fac4719ab8945a5870c51c986bc7d82c93ab5aad3e0e44e5314bc493d6`
- `experiments/tests/test_ddm_pz4r_pgq1_receiver.py`: `685da99a1008ba027f63e7b44eac3a6d42d5fa4d51639214297a04cedbf8075b`

## Scorer custody and exact fire order

The fleet summary printed `scorer slot: free`, but the physical and dispatch authorities disagree:

- `.omx/state/active_lane_dispatch_claims.md` has PS135 claim `lane_ddm_ps135_pose_resolve_20260810 / ddm_ps135_lc2_joint_pose_n600` in `running_full_n600_local_cpu`;
- PID 43675 currently holds `/Volumes/VertigoDataTier/pact/.locks/ddm_full_n600_scorer.lock`;
- PS135 was still advancing retained scorer chunks when polled.

PZ4R therefore did not claim a lane or run a scorer. `tools/codex_arm_queue.py` was updated to `ddm_pz4r_pgq1_receiver -> queued`. Do not close PS135's claim; its owner must harvest a terminal receipt and write the terminal dispatch row.

Before claiming PZ4R, run the governed Vertigo → APDataStore waterfall in the same shell. This requires 5,000,000,000 bytes for the retained raw/checkpoint/log workload plus an 8 GiB post-run reserve, creates no local-disk fallback, and persists the admission receipt outside scratch:

```sh
PZ4R_STORAGE_PLAN=/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/direct_v6/queued_full_n600_storage_plan.json
PZ4R_WORKLOAD_SUBDIR=ddm_pz4_joint_target_conditioned_receiver/direct_v6/exact_decode_eval/c408adf9101bb19a363039a5e0f7185aabce8f31edb6787e2deaf6d0fe6738f4

.venv/bin/python tools/plan_experiment_storage.py \
  --output "$PZ4R_STORAGE_PLAN" \
  --storage-plan-path "$PZ4R_STORAGE_PLAN" \
  --workload-subdir "$PZ4R_WORKLOAD_SUBDIR" \
  --reserve-free-gb 8 \
  --requested-bytes 5000000000 \
  --create

PZ4R_EVAL_ROOT=$(.venv/bin/python -c \
  'import json,sys; value=json.load(open(sys.argv[1])); assert not value["blockers"] and value["selected_workload_root"]; print(value["selected_workload_root"])' \
  "$PZ4R_STORAGE_PLAN")
```

After all fire conditions below hold, claim PZ4R with:

```sh
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id lane_ddm_pz4r_pgq1_receiver_20260811 \
  --platform macos_cpu_local \
  --instance-job-id ddm_pz4r_direct_v6_full_n600_local_cpu \
  --agent codex:ddm_pz4r \
  --status running_full_n600_local_cpu \
  --notes "direct_v6 archive c408adf9101bb19a363039a5e0f7185aabce8f31edb6787e2deaf6d0fe6738f4; full n600 retained decode plus both metric axes through upstream evaluate.py; macOS-CPU advisory"
```

Then run the canonical archive → public `inflate.sh` → untouched `upstream/evaluate.py` path under the atomic BSD fleet lock and Python 3.11 dependency closure:

```sh
: "${PZ4R_EVAL_ROOT:?run the retained storage-waterfall preflight first}"
mkdir -p "$PZ4R_EVAL_ROOT/token_checkpoint"

PYTHONPATH=/Users/adpena/Projects/pact/src:/Volumes/VertigoDataTier/pact/ddm_fx1_dependency_closure_20260809/runtime_deps_final \
PR130_TOKEN_CACHE="$PZ4R_EVAL_ROOT/token_checkpoint/tokens.npz" \
PR130_TOKEN_RECEIPT="$PZ4R_EVAL_ROOT/token_checkpoint/tokens.receipt.json" \
PR130_BROTLI_CLI=/opt/homebrew/bin/brotli \
PR130_RUNTIME_DEPS_DIR=/Volumes/VertigoDataTier/pact/ddm_fx1_dependency_closure_20260809/runtime_deps_final \
PR130_INFLATE_DEVICE=cpu \
OMP_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 \
VECLIB_MAXIMUM_THREADS=4 \
/usr/bin/lockf -t 0 -k /Volumes/VertigoDataTier/pact/.locks/ddm_full_n600_scorer.lock \
  /Users/adpena/Projects/pact/upstream/.venv/bin/python -u \
  /Users/adpena/Projects/pact/experiments/contest_auth_eval.py \
  --archive /Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/direct_v6/submission/archive.zip \
  --inflate-sh /Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/direct_v6/submission/inflate.sh \
  --upstream-dir /Users/adpena/Projects/pact/upstream \
  --upstream-python /Users/adpena/Projects/pact/upstream/.venv/bin/python \
  --video-names-file /Users/adpena/Projects/pact/upstream/public_test_video_names.txt \
  --device cpu \
  --work-dir "$PZ4R_EVAL_ROOT" \
  --json-out "$PZ4R_EVAL_ROOT/contest_auth_eval.json" \
  --keep-work-dir \
  --inflate-timeout 1800 \
  --evaluate-timeout 1800 \
  >"$PZ4R_EVAL_ROOT/launcher.log" 2>&1
```

The command retains raw output, periodic/final token checkpoints, logs, and both scorer components. Its result is `[macOS-CPU advisory]`, never contest authority. On success, close only PZ4R's claim as `completed_full_n600_local_cpu` and cite the receipt plus archive SHA. On failure, use `failed_full_n600_local_cpu`, retain every partial artifact/log, and move any retry to a distinct attempt path rather than overwriting it.

## Stage-2 design only

No joint training was launched. If the direct row fails its realized falsifier, the next formulation is a receiver trained with the six decoded PGQ outputs in the render/training loop, with every learned/video-derived basis, coefficient state, and conditioner counted. It must preserve the same strict parser, mutation, repeat, parse-back, full-video, and both-metric-axis gates. A post-hoc sidecar that never shapes frames is not this successor.

Cross-lineage use on cp135/PR135 is also design-only. No LC2 coefficient-fit number transfers. A successor must first custody the exact cp135/PR135 decoded output bank and carrier basis/coefficients, refit and byte-close a new counted map against those exact objects, and then measure its own realized row. Reusing PZ4R weights or claiming the LC2 rate/fit on cp135 is forbidden.

## NEXT_IF_RESUMED

- **Disposition: QUEUED-WITH-A-FIRE-ORDER; owner: `codex:ddm_pz4r` or the next explicit scorer-lane claimant; consumer store: the `selected_workload_root` in `/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/direct_v6/queued_full_n600_storage_plan.json`; fire trigger: PS135 has a retained terminal receipt and terminal dispatch row, the retained Vertigo → APDataStore plan has no blockers and preserves the 5 GB workload plus 8 GiB reserve, a fresh `codex_arm_queue.py status` says scorer-free, PZ4R successfully claims `lane_ddm_pz4r_pgq1_receiver_20260811`, and the exact `lockf` command above atomically acquires the physical fleet lock.** Run one retained full-n600 public-runtime decode and both-metric-axis upstream evaluation; write a terminal PZ4R dispatch row from the retained receipt.
- **Disposition: QUEUED-WITH-A-FIRE-ORDER; owner: future PZ4R joint-receiver training arm; consumer store: `/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/joint_v1/`; fire trigger: the direct-v6 full-n600 receipt measures `d_pose > 4e-5` or Seg collateral greater than `0.002` score units, and a counted joint-conditioned design passes resumability/storage preflight.** Train only then; retain every stage checkpoint and candidate, and do not transfer the ancestor stored-target number.
- **Disposition: QUEUED-WITH-A-FIRE-ORDER; owner: future cp135/PR135 refit arm; consumer store: `/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/cp135_refit/`; fire trigger: direct-v6 has a retained realized receipt, exact cp135/PR135 decoded-output and carrier banks are under custody, and a scorer lane is available after a new lineage-specific byte-close.** Refit from the exact cross-lineage objects; transfer apparatus only, never LC2 weights or measurements.

## LIVE-HYPOTHESES

- The residual-free PGQ-conditioned LC2 map may retain enough pose to beat LC2 despite only 0.4861 coefficient R², because the objective depends on PoseNet outputs after rendering rather than coefficient parity and the selected map consumes the incumbent frames' own output gauge.
- The 4,089-byte rate saving may pay for moderate realized pose/Seg degradation, because the candidate is genuinely below LC2 in exact archive bytes; only the queued nonlinear score decomposition can determine the exchange.
- A joint-conditioned receiver may outperform the direct map if it shapes frame preimages rather than predicting incumbent coefficients, because PGQ output preimages are many-to-one and coefficient regression unnecessarily inherits LC2 coordinates.
- A cp135-specific refit may combine PZ4R's receiver mechanism with the current rate frontier, because the apparatus is lineage-agnostic even though all weights, banks, rate, and fit measurements must be rebuilt.

## DEAD-ENDS

- **FORMULATION:** Treating PGQ1 or the 168,005-byte envelope as a rendered candidate is closed: PGQ1 has no basis/coefficients and the unchanged LC2 receiver cannot render it.
- **INSTANCE:** Reusing the retained PZ3 weights on LC2/PGQ1 is closed: PZ3 maps official-DALI-GT PZ2 codes, while PZ4R maps LC2-decoded PoseNet outputs.
- **FORMULATION:** Retaining an exact coefficient residual is closed: PZ3 already measured that formulation at 194,120 B, 3,068 B above its base.
- **FORMULATION:** Feeding six PoseNet outputs directly into an SE(3) physical warp is closed by G91's poor affine fit and admissibility rule.
- **FORMULATION:** Selecting with modular 12-bit coefficient error is closed: it hides renderer discontinuities and selected the withdrawn f8 archive; ordinary error selects f10.
- **INSTANCE:** Scoring the initial unversioned attempt or `direct_v2`–`direct_v5` is closed: they are retained correction history, while only `direct_v6` has the final metric, resume, and manifest gates.
- **INSTANCE (current state):** Calling the scorer now is closed: PS135 owns the active dispatch and physical fleet lock, regardless of the stale queue summary's free-slot line.

Own-vehicle frontier unchanged: cp135 `S=0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`; PZ4R moved no exact score and did not reach sub-0.15.
