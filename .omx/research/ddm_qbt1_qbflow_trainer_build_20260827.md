# DDM QBT1 — real QBFLOW scorer-in-loop trainer build

Date: 2026-08-27  
Disposition: **BUILD COMPLETE; STAGE 03/04 QUEUED FOR MAIN; STAGE 05 BLOCKED ON A REAL SAME-BUDGET QBW1 CONTROL**  
Authority boundary: no Metal launch, no Modal launch, no full-n600 scorer job, no contest evaluation, and no frontier movement occurred in this arm.

## Result

The missing real trainer now exists at `experiments/ddm_qbt1_qbflow_trainer.py` and consumes the frozen QBF1 packet ABI without editing it. The bounded n=1 CPU smoke executed the real receiver twin, camera round trip, uint8 STE, frozen SegNet/PoseNet graph, joint expected-flip/pose objective, optimizer step, canonical EMA update, atomic checkpoint, real-coder re-encode, parse-back, stage-04 precision surface, retained scorer outputs, and stage-05 arithmetic. Resume re-created the same archive byte-for-byte.

The implementation is launchable only through an authorized compiled config. The emitted config is deliberately unclaimed: draft validation passes, while live launch validation refuses with `heavy training is not authorized`.

## Trainer inventory

- `experiments/ddm_qbt1_qbflow_trainer.py`
  - Exact-shape differentiable Torch twin of `experiments/ddm_qbflow_packet.py`; a focused test compares signed interfaces, class logits, RGB, and pose12 against the frozen NumPy receiver.
  - All three QBFLOW escapes are active: joint boundary/interior birth with pose from step zero, receiver-derived Road probability/tangent conditioning, and the dedicated 8/16/24/32 along-tangent comb. Step slopes/centers are trainable; there is no fixed-high-beta hosc.
  - Realization path is QBF1 render at 384x512, bicubic expansion to 874x1164, uint8 STE, then the frozen scorers' differentiable preprocessing, including the PoseNet yuv6/no-grad cure from `load_differentiable_scorers`.
  - Seg uses the w96b expected-flip-margin law with a pinned 0.15 to 0.05 schedule. The realized scorer term and the native signed-interface term use the same law. Pose6 MSE and its score geometry are active on optimizer update zero.
  - The sealed no2 n32 set is split into two 16-pair chunks, each containing twelve weight-15 and four weight-30 pairs. The expected-flip and pose terms consume those weights. This avoids both the forbidden >30 materialization and the biased 30+2 optimizer schedule.
  - EMA decay is resolved through canonical equation `ema_decay_run_geometry_v1`. Checkpoints retain the live model, optimizer, Python/NumPy/Torch RNG state, EMA law/state, and the EMA shadow used for inference.
  - Periodic and stage-end checkpoints are atomic and distinct. Every checkpoint is followed by a complete QBF1 re-encode that retains raw sections, every coder candidate and deterministic repeat, packet/repeat, archive/repeat, reset records, hashes, and parse-back evidence.
  - Stage 04 accumulates a real receiver-plus-scorer gradient at the EMA inference state, probes role-specific prequantization, and byte-closes every option through QBF1. The gradient metric is only a shortlist proxy; the code forbids adoption without a realized scorer A/B.
  - Stage 05 implements the no2 section-5 HT estimator and exact score arithmetic. Its QBW1 leg requires a QBW1 family receipt, an exact archive, all 32 retained pair payloads, identical pair set and budget, and recomputed component arithmetic. A scalar-only or fabricated control cannot pass.
  - `run-config` refuses heavy execution unless the config is authorized, the exact sealed n32 set and 16-pair chunks are preserved, device is MPS, and both Metal and scorer lane claim IDs are present.
- `experiments/tests/test_ddm_qbt1_qbflow_trainer.py`
  - Eight focused tests cover frozen NumPy/Torch forward parity, the hard chunk ceiling, equal no2 chunk mass, expected-flip behavior, frozen-shape prequantization, checkpoint restoration, fail-closed/positive control gate arithmetic, and unclaimed launch config refusal.

No file under `upstream/` was modified. The separately owned untracked WD3 reference file was read and SHA-pinned (`6a567db93c9947e63b5fb022411dd583ce848ccb22e3fe0e2393fe58c94a86df`) but is not a runtime import and is not part of this landing.

## RECALL EVIDENCE

The recall covered the full requested surfaces before implementation:

- `.omx/research/` content searches for `QBFLOW`, `QBF1`, `expected.flip`, `margin`, `comb`, `Road condition`, `QBW1`, `same-budget`, `EMA`, `chunk`, and `materialization`; the QBFLOW and QBW1 rate receipts, packet schema, no2 section 5, w96b aligned verdict, and WD3 reference form were opened at their operative sections.
- `.venv/bin/python tools/list_canonical_equations.py --json` returned 449 canonical rows; `ema_decay_run_geometry_v1` was selected and is resolved at config compilation rather than copied as a literal.
- `.omx/research/CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, specs/design docs, task-ledger hits, `.omx/state/main_hot_state.md`, and the live lane registry were searched for the QBFLOW/QBW1/EMA/scorer surfaces.
- The actual `upstream/evaluate.py`, `src/tac/scorer.py`, `tac.training.EMA`, the QBF1 encoder/decoder/NumPy receiver, the retained GT-cache layout, and the sealed fire-order JSON were inspected rather than inferred from memos.

Findings beyond the charter seeds changed the build in three ways. First, no2's exact weights exposed that a superficially legal 30+2 schedule would overweight two samples, so the compiled n32 schedule uses two equal-mass 16-pair chunks and weighted objectives. Second, the existing QBW1 stage-02 receipt closes only its rate leg (`B_hat=389,362 B`, 251,376 B over the cap) and has no real renderer/scorer control payload, so stage 05 refuses it rather than fabricating a smoke control. Third, the current QBFLOW packet already carries complete n600 latent framing, so every trained EMA checkpoint is re-encoded through that exact object; initialized rate is never transferred as a trained result.

## Bounded smoke receipt

Custody root: `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/smoke_n1_20260827_final/`  
Command: `.venv/bin/python experiments/ddm_qbt1_qbflow_trainer.py smoke --output /Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/smoke_n1_20260827_final --pairs 1 --steps 1`

- Result: 17.456813999917358 s, 189 MiB retained, 1,269 files; `RESULT.json` is 29,522 B, SHA-256 `5bcd918f18ac2058d259d7e2797a3a0fee9f8b0670b0b123331a962b59ee6400`.
- Config SHA-256: `cec05405f5df129e3e0aa086beb5ad2ba7782dddd604560b9b06d25f1250c569`.
- Stage-03 end checkpoint: 1,605,597 B, SHA-256 `f07a4b89084d4779d57bea731e79fdcd601f1d67bae089e23488f8bf858223d5`.
- Re-encoded archive: 107,539 B, SHA-256 `0bde78015b71a1fb7807b8bcd0b93173249bd484d59fab1d09cb311b5fe57ebe`; archive repeat is byte-identical.
- Resume identity: PASS. Reloading live state, optimizer, RNG, and EMA re-encoded the same 107,539-B archive with the same SHA-256.
- Stage-05 retained pair payload: 8,464,966 B, SHA-256 `a5112080fa3edb96c7f73e2e3b742c30dbe01e28e97e9fa879c5423555f9ed65`; it contains camera uint8 frames, SegNet logits/argmax, PoseNet pose6, and both targets.
- Payload-retention gate: zero findings for the trainer and tests.
- The n=1 gate is intentionally non-admitted: estimator status is `UNWEIGHTED_BOUNDED_SMOKE_MEAN_ONLY`, selection count is 1, and the real QBW1 control is absent. Its distortion and `S_hat` are mechanism-smoke numbers, not a family verdict or score claim.

Stage-04's n=1 8-bit probes retained all coder payloads. Boundary-flow prequantization reduced the complete archive by 643 B and step-transition by 334 B on this initialized one-step object, while several other roles increased bytes. These are **not adopted**: the receipt labels them first-order shortlist signals without a realized scorer A/B.

## Memory and schedule projection

Axis: `[derived CPU-smoke projection; Metal peak not measured]`.

- Observed n=1 baseline RSS: 320,159,744 B.
- Observed n=1 peak RSS: 3,681,157,120 B.
- Real compiled materialization chunk: 16 pairs; hard ceiling remains 30.
- Projected n32 peak: 92,084,098,826 B.
- Fire-order ceiling: 124,554,051,584 B.
- Projected headroom: 32,469,952,758 B. This passes only the derived projection; a live Metal preflight remains mandatory.
- CPU-linear upper schedule projection: 36,310.173119828105 s for 130 optimizer updates. This is explicitly not a Metal timing measurement and must be replaced at fire.

## Verification and two-pass review

- `.venv/bin/python -m pytest -q experiments/tests/test_ddm_qbt1_qbflow_trainer.py` → `8 passed`.
- `.venv/bin/python -m ruff check experiments/ddm_qbt1_qbflow_trainer.py experiments/tests/test_ddm_qbt1_qbflow_trainer.py` → PASS.
- `.venv/bin/python -m py_compile experiments/ddm_qbt1_qbflow_trainer.py` → PASS.
- `tac.payload_retention_gate.check_no_measure_and_discard_payload(...)` over both files → zero findings.
- Review tracker pass `qbt1-pass1-final`: 53 trainer entities and 9 test entities reviewed.
- Review tracker pass `qbt1-pass2`: 53 trainer entities and 9 test entities reviewed.
- Review receipt: `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/TWO_PASS_REVIEW_RECEIPT.json`, 814 B, SHA-256 `ecda31dd6360b8c84757b1ef03b6d0607e60822d9737e8b557023dec852d53fc`; `REVIEW_GATE_OVERRIDE` was not used.

Pass 1 corrected frozen-forward semantics, the biased 30+2 schedule, EMA-state sensitivity, and weak control custody. Pass 2 added no2 sample weighting, made HT completeness order/count exact, stabilized the EMA LawRef comparison without discarding its identity, removed WD3 as a clean-checkout runtime dependency, and separated the stage-03/04 fire trigger from the stage-05 control trigger.

## Compiled launch request

- Request: `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/COMPILED_LAUNCH_REQUEST.json`, SHA-256 `3844fcc4cdb7bfe4acad66126e9dc564e4cc256f8c87ed255693eb4980b5194c`.
- Config: `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/COMPILED_N32_CONFIG.json`, SHA-256 `802a06034e2a86a39dbe14e8dffab3d154da87e928f2834d93da371d7c26c1a8`.
- Disposition: `QUEUED_STAGE03_04_FIRE_STAGE05_BLOCKED`.
- Owner: MAIN QBFLOW joint-training owner.
- Consumer store: `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow`.
- Stage-03/04 fire trigger: MAIN verifies committed hashes against the review receipt, confirms no duplicate active lane and no full-n600 scorer job, claims the Metal and scorer lanes, and reruns live storage plus <=116-GiB admission.
- Stage-05 fire trigger: the governed n32 result exists and a real retained same-budget QBW1 receipt passes custody, pair-set, budget, and score-arithmetic validation.
- Draft validation: PASS. Live launch validation: `REFUSED: heavy training is not authorized`.
- Invocation counts in this arm: Metal 0, Modal 0, contest evaluation 0, full-n600 scorer 0.

## Conclusions and boundaries

The build mandate is complete: the QBF1 ABI has a real scorer-in-loop trainer, tests, two-pass review, retained bounded-smoke proof, memory projection, and an unfired compiled request. This did **not** lower an exact score. It did **not** test whether joint-from-birth escapes the QBFLOW distortion wall. It did **not** produce the mandatory same-budget QBW1 control. Those are MAIN fire/consumer obligations, not results of this build arm.

## LIVE-HYPOTHESES

- Joint pose6 descent from update zero may make the interior/RGB field pose-legible before the family enters the historical post-hoc wall; this is plausible because official PoseNet gradients reached the rendered frames in the retained smoke and the two n32 chunks now preserve the no2 mass.
- Boundary-flow and step-transition precision may yield rate cuts after training; this is plausible because both reduced real complete QBF1 bytes in the bounded smoke, but only a realized scorer A/B can admit either option.
- The n32 job may fit below the Metal watermark with 16-pair materialization; this is plausible because the conservative projection is 92,084,098,826 B versus the 124,554,051,584-B ceiling, but it remains unmeasured on Metal.

## DEAD-ENDS

- A 30+2 optimizer split is closed for this trainer because it overweights the two-pair chunk; the replacement is two equal-size, equal-no2-mass 16-pair chunks.
- Treating the existing QBW1 stage-02 rate receipt as the same-budget scorer control is closed because it has no real renderer/scorer outputs or 32 retained control payloads and is over the archive cap on its own measured rate leg.
- Adopting a role precision solely from first-order sensitivity is closed because the proxy does not prove realized Seg/Pose survival; every option remains `NO_ADOPTION_WITHOUT_REALIZED_SCORER_AB`.
- Fixed-high-beta hosc and serialized boundary payloads remain closed by prior family evidence and the frozen QBF1 ABI; this trainer uses trainable finite steps and receiver-generated Road conditioning instead.

Own-vehicle frontier unchanged: gb1 — S 0.14811799921260607 @ 180,215 B `[contest-CUDA T4 n600]`.
