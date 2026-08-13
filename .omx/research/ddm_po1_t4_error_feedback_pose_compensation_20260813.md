---
arm: ddm_po1_t4_error_feedback_pose_compensation
date: 2026-08-13
status: APPARATUS_READY_ROUND1_QUEUED_TO_MAIN
research_only: true
score_claim: false
pointer_moved: false
authority: scorer-free build verification only
---

# DDM PO1 — T4 error-feedback Pose compensation

PO1 is ready to run, but no scorer or Modal lane was fired from this arm. The common contract assigns
the component and exact T4 fires to MAIN and this arm did not own the sole n600 scorer slot. The delivered
unit is therefore the complete retained worker, damped solver, candidate repacker, adjudicator, and exact
conditional fire order—not a pose result.

The effective frontier remains **CP135 `S = 0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`**.
The own-vehicle frontier remains **LC2 `S = 0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`**.
This scorer-free apparatus unit moved neither pointer and did not reach sub-0.15.

## What landed

- `experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py` now has reusable exact frozen-PoseNet
  loading and scoring. Each pass retains the preprocessed PoseNet input, the full 12-output tensor, the
  scored first-six vectors, GT RGB source batches, per-batch receipts, and an n600 aggregate vector.
- `experiments/ddm_po1_t4_pose_feedback_worker.py` runs one exact receiver, GT/candidate SegNet fields,
  GT PoseNet, decoded PoseNet, and a same-job decoded repeat. It emits float32-instrument `d_pose`, exact
  integer Seg flips/denominator, per-pair error/noise vectors, and the pre-registered F1 diagnostic.
- `experiments/ddm_po1_modal_t4_pose_feedback.py` is a governed resumable T4 dispatcher. K=1 is projected
  as `466.0 + 5*39.405 + 300 = 963.025 s`, explicitly a projection from prior T4 component times, not a
  PO1 measurement. Its storage preflight reserves all **32,097,566,400 payload bytes plus 4 GiB**.
- `experiments/ddm_po1_t4_error_feedback_pose_compensation.py` consumes the retained T4 residual, uses a
  local CPU PoseNet 6x12 STE Jacobian only as a preconditioner, takes one damped signed-int12 step, skips
  the five non-identity selector pairs, persists every pair/Jacobian/checkpoint, and creates a new archive
  plus runtime without changing CP135. It also adjudicates Round 2 using exact Seg field identity, Pose
  decrease, same-object rate, F2 realization, and the third-round gate.
- `experiments/tests/test_ddm_po1_t4_error_feedback_pose_compensation.py` covers vector retention, repeat
  noise, damped direction and int12 clipping, exact CP135 identity rebuild, and F3 field-identity closure.

Borrowed-substrate accounting: the CP135/F26 receiver, CPR1/CAP1 carrier, F0E1 selector, upstream PoseNet,
and the ExperimentBook CAP1 encoder/damped-GN formulation are granted ancestors. PO1-original work is the
same-T4 vector/repeat instrument, T4-residual/local-J cross-axis solve contract, variable-CAP1 receiver-safe
repacker, resumable retention surface, and F1/F2/F3 adjudication. No borrowed row is relabeled as original.

## Scorer-free measurements and custody

All rows below are **`[macOS-CPU scorer-free byte-close/receiver-parseback] TEST FIXTURE ONLY`**. They are
not Seg/Pose scores and are not promotion eligible.

| Fixture | Changed codes / 7,200 | Archive bytes | SHA-256 | Receiver result |
|---|---:|---:|---|---|
| CP135 identity rebuild | 0 | 186,252 | `6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6` | exact byte identity |
| pair 0, dimension 0, +1 | 1 | 186,251 | `d822ad3f16e4bcefbb5c8fd7373ffee27cf6c191369d6c02d7111f82eb13ab02` | parsed, packed CAP1 |
| seed 1234, iid integer offsets [-16,16] | 6,982 | 186,316 | `ce5ba0ab27f036887b5916fc5f92afb1313166b747c8eefb72d3988db744de4d` | parsed, variable canonical CAP1 |

The two mutated fixtures and their complete candidate runtimes are retained under
`/Volumes/VertigoDataTier/pact/ddm_po1_20260813/retained/verification_recovered/`; the machine manifest is
`MANIFEST.json`. During verification I initially built those two mutations in an auto-cleaned temporary
directory. That violated the keep-every-payload rule. I deterministically rebuilt both into the SSD tier,
confirmed the same SHAs (`d822…` and `ce5b…`), and retained configs, candidates, runtimes, and parse-back
receipts. No such temporary fixture is being used as evidence without recovered payload custody.

## Round 1 fire order

Disposition: **QUEUED-WITH-A-FIRE-ORDER**. Owner: **MAIN**. Remote consumer store:
`comma-ddm-po1-pose-feedback-retained/ddm_po1_round1_cp135_20260813`. Local consumer store after harvest:
`/Volumes/VertigoDataTier/pact/ddm_po1_20260813/round1_cp135`. Fire trigger: MAIN owns the sole component
lane, no other full-n600 scorer is active, and CP135 still verifies as 186,252 bytes / SHA `6eb1…edb6`.

```bash
.venv/bin/modal run --detach experiments/ddm_po1_modal_t4_pose_feedback.py::main \
  --archive /Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/archive.zip \
  --runtime /Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime \
  --expected-archive-bytes 186252 \
  --expected-archive-sha256 6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6 \
  --output-dir /Volumes/VertigoDataTier/pact/ddm_po1_20260813/dispatch/round1_cp135 \
  --run-id ddm_po1_round1_cp135_20260813 \
  --resume-from ddm_po1_round1_cp135_20260813 \
  --round-ordinal 1 \
  --lane-id ddm_po1_t4_pose_feedback_round1 \
  --instance-job-id modal:ddm_po1_round1_cp135_20260813 \
  --claim-agent main:ddm_po1 \
  --detach --provider-detach-ack
```

Recover the detached call, then download the exact solver inputs:

```bash
.venv/bin/python experiments/ddm_po1_modal_t4_pose_feedback.py recover \
  --output-dir /Volumes/VertigoDataTier/pact/ddm_po1_20260813/dispatch/round1_cp135
.venv/bin/modal volume get --force comma-ddm-po1-pose-feedback-retained \
  ddm_po1_round1_cp135_20260813/FINAL_RESULT.json \
  /Volumes/VertigoDataTier/pact/ddm_po1_20260813/round1_cp135/FINAL_RESULT.json
.venv/bin/modal volume get --force comma-ddm-po1-pose-feedback-retained \
  ddm_po1_round1_cp135_20260813/retained/pose_vectors/ \
  /Volumes/VertigoDataTier/pact/ddm_po1_20260813/round1_cp135/retained/pose_vectors/
.venv/bin/modal volume get --force comma-ddm-po1-pose-feedback-retained \
  ddm_po1_round1_cp135_20260813/retained/fields/ \
  /Volumes/VertigoDataTier/pact/ddm_po1_20260813/round1_cp135/retained/fields/
.venv/bin/modal volume get --force comma-ddm-po1-pose-feedback-retained \
  ddm_po1_round1_cp135_20260813/retained/raw/candidate/0.raw \
  /Volumes/VertigoDataTier/pact/ddm_po1_20260813/round1_cp135/retained/raw/candidate/0.raw
```

## Conditional solve and Round 2

If Round 1 returns `FEEDBACK_USABLE`, the local solve is
**QUEUED-WITH-A-FIRE-ORDER** to owner `ddm_po1 local solver`; consumer store
`/Volumes/VertigoDataTier/pact/ddm_po1_20260813/solve/attempt1`; fire trigger: all four downloaded Round 1
payload groups verify against `FINAL_RESULT.json` and no local full-n600 scorer owns the fleet lock.

```bash
.venv/bin/python experiments/ddm_po1_t4_error_feedback_pose_compensation.py solve \
  --round1 /Volumes/VertigoDataTier/pact/ddm_po1_20260813/round1_cp135 \
  --output /Volumes/VertigoDataTier/pact/ddm_po1_20260813/solve/attempt1 \
  --resume-from ddm_po1_attempt1_20260813 \
  --damping 0.01 --max-code-step 32
```

The completed `SOLVE_RESULT.json` writes the Round 2 command with the actual new archive bytes and SHA;
there are no placeholders. Round 2 remains owner **MAIN**, consumer store
`/Volumes/VertigoDataTier/pact/ddm_po1_20260813/dispatch/round2_candidate`, and fires only after the sole T4
component lane is clear. Then run `adjudicate` on the downloaded Round 1 and Round 2 stores. An admitted
result writes the exact paired auth-eval fire order. F1/F3 and non-improving rows are **FOLDED**. F2 writes
one exact higher-damping local retry command, but no third T4 fire because the independent third-round rule
requires at least 50% realization.

## Gate semantics

- F1 closes the mechanism when same-job repeat RMS is at least half decoded-vs-GT RMS for more than
  300/600 pairs. This is a pre-registered operational meaning of “comparable,” not a post-hoc threshold.
- Admission requires T4 `d_pose` lower, the complete 600x384x512 candidate Seg field byte-identical to
  Round 1, and `Delta S < 0` after recomputing Pose+Seg+rate from components and exact archive bytes.
- F2 is `realized_gain / predicted_gain < 0.20`, including a sign inversion. It gets one retained local
  retry at damping 0.04 / radius 16 and then closes; it cannot authorize a third T4 row.
- A third feedback row is allowed only when realization is at least 0.50. No code path promotes on a local
  CPU Jacobian, local score, projection, printed two-decimal evaluator score, or changed Seg field.

## RECALL EVIDENCE

Searched the full memo corpus, canonical-equation registry, canonical index/DAG/SPEC surfaces, and task
ledgers using content queries including `pose compensation`, `error.feedback`, `quantize.then.compensate`,
`joint posenet`, `coefficient.*pose`, `F14`, `jacob`, and `gauss`.

Beyond the charter seeds, the recall found:

- `ddm_rvs1_realization_survival_harvest_20260811.md`: quantize-then-compensate is the named missing
  realization mechanism, so PO1 keeps the integer lattice and solves after exact receiver realization.
- `ddm_hc1_hy1_container_push_20260812.md` plus `ddm_ps135_pass4_exact_row_harvest_20260812.md`: local CPU
  Pose improvements can invert badly on T4. This changed the solver from local-score acceptance to
  T4-residual origin plus same-instrument Round 2 only.
- ExperimentBook F14/F24/F26 sources: an existing 6x12 STE Jacobian, damped solve, CAP1 encoder, selector,
  and fixed archive builder already exist. This prevented a new proxy carrier or invented solver.
- `ddm_cp135_rate_compose_20260810.md`: CP135’s winning split-model qualities are `[10,11,11]`, and its
  packed CAP1 metadata assumes one exact old length. This caused the receiver-safe variable canonical
  CAP1 fallback instead of falsely forcing new coefficients into the old fixed section size.
- `ddm_js1b_cuda_custody_adjudication_20260813.md`: current local-vs-T4 field drift makes same-instrument
  field identity the Seg gate. PO1 therefore compares complete T4 fields, not local flip counts.
- The canonical equations registry supplied the exact score marginal law and same-object rate term, but no
  directly governing T4 error-feedback coefficient law. No equation displaced the measured two-round
  protocol. The broad registry output was bounded with content filtering; the scoped negative is only for
  those searched equation terms.
- The queue/ledger surfaces identify `ddm_po1` as live and MAIN as T4 fire owner. The Round 1, Round 2,
  and conditional paired-exact lane IDs were pre-registered at L0 through `tools/lane_maturity.py`; no
  active lane was claimed because this arm did not dispatch.

## Verification

- `ruff check`: all five touched Python files pass.
- `py_compile`: all five touched Python files pass.
- Focused PO1 tests: **5 passed**.
- Payload-retention AST gate over the four runnable files: **0 findings**.
- CP135 identity build: exact 186,252 bytes / `6eb1…edb6`.
- Both changed-code verification archives parse through their retained candidate runtimes; all bytes kept.
- Two explicit `review_tracker.py mark-file` passes per Python file, then policy check: **90 entities,
  0 violations**.
- The repository-wide developer preflight was **17/25 green, 8 red**. The red gates resolve to existing
  unrelated surfaces: `probe_outcomes_ledger.py`, `submission_chain.py`, legacy ad-hoc launch scripts,
  the AGENTS terminal-claim wording, 124 older landing memos, 14 older unregistered lane references,
  56 substrate scorer-contract debts, and 21 trainer pose-default debts. PO1's three literal lane IDs are
  registered and the PO1 memo uses the research-only wire-in opt-out; this unit does not claim the dirty
  repository's global preflight green.
- A combined PO1+older-JS1B test command produced 11 passes and 2 failures in existing
  `load_cuda_argmax_bundle` fixture expectations (C1 custody SHA and missing C1 file). PO1 does not touch
  that loader; the narrower PO1 suite is green. This unit does not claim the unrelated JS1B suite green.

Post-edit source SHA-256:

- `experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py` — `542afd38d101cf43e3c778a8b758f0295822e8dd8212a64619639aa9c1f02d71`
- `experiments/ddm_po1_t4_pose_feedback_worker.py` — `6b022a71a7f799397f23058c85f7a84ece3c6c30f953f81efb008ede01ffa6ec`
- `experiments/ddm_po1_modal_t4_pose_feedback.py` — `0445eda1d11809e647a6d4d3acbbcd42163516578bef3c4ff6849c3021ac1874`
- `experiments/ddm_po1_t4_error_feedback_pose_compensation.py` — `d16630c4dc820cf89da52d4c61afbb91ac5f162fafa873f2a7525c0c87217dac`
- `experiments/tests/test_ddm_po1_t4_error_feedback_pose_compensation.py` — `cfea46912f946fc3c93fed6fc95022dcd368dc301135204760c43dbee147f7ea`

## LIVE-HYPOTHESES

- CP135’s residual T4 pose error is systematic enough that a local-J-preconditioned step will reduce it on
  the same T4 instrument. This is plausible because the residual comes from exact T4 vectors while the
  Jacobian supplies direction only, directly addressing the measured CPU-to-CUDA transfer failure.
- Most changed coefficient streams will retain CP135’s packed CAP1 length and near-neutral rate. Single and
  constant mutations retained the old physical length; the canonical fallback also bounds the failure to
  a small real byte delta when Rice length changes.
- Same-job repeat noise will be below half the CP135 error on most pairs. Frozen eval-mode PoseNet with
  identical decoded bytes should be deterministic, but this remains untested on the actual T4/DALI job.

## DEAD-ENDS

- Local CPU score improvement as admission: closed by prior T4 inversion evidence; local CPU is direction
  only, never the verdict.
- Forcing every changed coefficient stream into CP135’s old fixed packed length: closed because seeded
  changed codes produced a 22,226-byte physical section. The receiver-safe canonical fallback is required.
- Updating the five non-identity F0E1 selector pairs with the plain carrier Jacobian: closed for this
  formulation because the local derivative omits the integer pixel-mode map; those pairs are frozen.
- Advancing after any Seg field movement: closed by F3; complete field identity is mandatory.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store:
  `/Volumes/VertigoDataTier/pact/ddm_po1_20260813/round1_cp135`; fire trigger: sole T4 component lane clear
  and CP135 bytes/SHA reverified; action: run and harvest the exact Round 1 command above.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: ddm_po1 local solver; consumer store:
  `/Volumes/VertigoDataTier/pact/ddm_po1_20260813/solve/attempt1`; fire trigger: Round 1 status
  `FEEDBACK_USABLE` and all downloaded records verify; action: run the pinned damped solve above.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store:
  `/Volumes/VertigoDataTier/pact/ddm_po1_20260813/dispatch/round2_candidate`; fire trigger:
  `SOLVE_RESULT.json` status `CANDIDATE_READY_FOR_T4_ROUND2`, sole component lane clear, and generated
  archive bytes/SHA reverified; action: execute the exact generated Round 2 command.

## ADDENDUM 1 — Round-1 T4 result (2026-08-13, MAIN harvest)

Round 1 ran to COMPLETE on Modal T4 (run `ddm_po1_round1_cp135_20260813`, detached dispatch
46 s, job ~18 min). Status: **FEEDBACK_USABLE**. The three headline numbers:

- `repeat_noise_mse = 0.0` — the T4 PoseNet forward is **bit-deterministic in-job**. Two
  independent forwards over all 600 pairs produced byte-identical first-six vectors.
- `d_pose_decoded_first == d_pose_decoded_repeat == 6.885642960696714e-06` — independently
  confirms cp135's composed-row d_pose from a fresh decode + fresh scorer load.
- `noise_comparable_pairs 0/600`, `f1_instrument_floor_closed: false` — the F1 falsifier
  (instrument noise floor comparable to the signal) did NOT fire. The remaining d_pose is
  100% signal; the compensation solve faces **no noise-floor constraint**.

Custody: ~32 GB retained on volume `comma-ddm-po1-pose-feedback-retained` at
`/ddm_po1_retained/ddm_po1_round1_cp135_20260813` (raw frames, GT RGB, seg logits/fields,
pose vectors). Seg field sha `7648ad42…` matches the sa1/js1b BASE_FIELD_RECORD exactly —
three independent T4 decodes of cp135 now agree byte-for-byte on the argmax field.

Local consumer store staged at `/Volumes/VertigoDataTier/pact/ddm_po1_20260813/round1_cp135`:
FINAL_RESULT.json + 5 pose-vector npy (sha-verified) + candidate argmax field (sha-verified)
+ retained/raw/candidate/0.raw (3.66 GB, downloaded from the volume — the T4-decoded bytes
the solve linearizes around, NOT a local re-decode, per the sign-flip custody law).

Worker defect fixes applied to both sa1 and po1 workers pre-fire (commits a3eb572dae,
99e4c05f99): `sys.dont_write_bytecode = True` (bytecode self-pollution refused the end-of-run
upstream sha) and volatile `storage_preflight` excluded from the stage-00 checkpoint (its
free_bytes drift made byte-identical resume structurally unsatisfiable).

Next: damped local-Jacobian compensation solve (`solve --resume-from
ddm_po1_solve_attempt1_20260813`, damping 0.01) → Round-2 T4 dispatch on the candidate.

## ADDENDUM 2 — Round-2b T4 verdict: REJECTED (2026-08-13, MAIN adjudication)

**Verdict: the gated damped local-Jacobian candidate is DEAD-CONFIRMED at instance scope.**
Round-2b receipt (`dispatch/round2b_candidate/PO1_FINAL_RESULT.json`, execution_status COMPLETE,
archive sha `5d89150d61a0…` 186,335 B — the exact attempt3 candidate):

- **Realized d_pose 5.6857839808799326e-05** [contest-CUDA T4 component-only, n600], first and
  repeat bit-identical (repeat_noise_mse 0.0 — zero noise floor again).
- Base d_pose 6.885642960696714e-06 → realized **8.257× WORSE**. Predicted gain was +4.4656e-7
  (d_pose → 6.439e-6); realized −4.9972e-5. **Realization ratio ≈ −112×.**
- ΔS ≈ **+0.0156** (pose √-term 0.008298 → 0.023845, rate +83 B). Independent arm (hv1) derived
  +0.015602 from the same receipt before this adjudication — agreement.

**Mechanism (the instrument finding, na2 law: a negative measures the instrument).** The per-pair
acceptance gate consumed the LOCAL linearized prediction, and the accepted per-pair gains
(~1e-9…1e-8 each, +4.47e-7 total) sit far below the model's own measured forward-mismatch floor
(9.36e-6 vs residual RMS 1.589e-3, ~1% trust). The gate was filtering noise: every acceptance was
unfalsifiable by the instrument that proposed it. Attempt1's ungated +55.6% already showed int16
quantization dominates the continuous step (quantum-floor law, third measurement); the gate cured
the PREDICTED regression but could not create real signal below the model's error floor.
**Law sharpened: a local-model acceptance gate is only valid for candidate steps whose predicted
magnitude exceeds the model's measured forward-mismatch floor.** The v17 doctrine (realize one
step → re-linearize) would have rejected this at step 1; the one-shot batch of 3,867 quantized
coefficient changes was outside the validity radius.

**Custody:** all Round-2b payloads retained on the Modal volume run-root
(`ddm_po1_round2b_candidate_20260813`: raw, seg inputs, logits, pose vectors, batch receipts —
shas in the receipt). Lane claim closed `failed_candidate_rejected_on_t4`. Poller pid 39514
stopped (log preserved); the recover-idempotence defect (duplicate terminal ledger rows) is
hv1's queued fix row.

**Spend:** Round-2b ≈ $0.16; po1 arc total ≈ $2.2 of the #381 $20 envelope.

**Routing:** the pose leg's iterative local-Jacobian family is CLOSED on this vehicle at this
operating point. Next pose lead per the hv1 fresh-eyes review (memo sha `1e071f66…`, commit
eb7771b1cd): **PZ4R** — the receiver-closed 183,137 B recode (−4,089 B measured, distortion
unmeasured) — one retained full-n600 public-runtime evaluation decides it. hv1 also corrected
js7's pose stack budget: the published 1.3e-7 is arithmetically wrong; the candidate-specific
allowance is ~1.2–1.3e-6 (≈10× looser), which softens the seg-leg composition constraint.
