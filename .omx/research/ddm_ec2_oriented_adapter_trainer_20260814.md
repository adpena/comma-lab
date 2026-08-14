# DDM EC2 — oriented latent-adapter T4 trainer and sealed fire order

**Disposition:** `QUEUED-WITH-A-FIRE-ORDER` — build and seal complete; MAIN has not fired Modal.

**Verdict scope:** implementation/seal. No trained adapter, T4 endpoint, packaged candidate,
Seg/Pose component row, exact score, or pointer movement was produced by this arm.

## Result

The EC1 blocker `true_cuda_trainer_implemented_by_producer: false` is closed at the
implementation and seal boundary. The new worker is a real scorer-in-loop trainer:

1. It restores the exact CP135 `SemanticTokenRenderer(96)` from the pinned 186,252-byte archive.
2. It injects the counted EC1 conditioner after `token_embed + coord_mix` and before the same four
   CP135 nonlinear `TokenBlock`s.
3. It applies EC1's exact int8-weight/float16-bias quantizers in the training loop with an STE.
4. It renders through the CP135 public camera operator: bilinear `384x512 -> 874x1164`, camera
   clamp/round to uint8, then bilinear down to the frozen upstream SegNet grid.
5. It optimizes the adapter against the exact retained T4 GT labels and CP135 error field.
6. Its endpoint reloads the serialized counted module through
   `ec1_latent_conditioner.load_conditioner`; the full-n600 field therefore measures the bytes that
   will be put in the archive, not a float-EMA proxy.

This is still a component trainer. Only the later EC1 package plus RE1T/JS1B component chain and
`upstream/evaluate.py` may establish a score.

## Landed surfaces

| Surface | Purpose | Post-seal SHA-256 |
|---|---|---|
| `experiments/ddm_ec2_oriented_adapter_trainer_worker.py` | real T4/QAT/SegNet trainer, resume, retention, endpoint | `5e8a18650d1412d0f3ec4bd6bae61eb9413a1279f130e16323d9e805267aea6e` |
| `experiments/ddm_ec2_modal_oriented_adapter_trainer.py` | new Modal app, immutable requests, claims, recovery, control gate | `ea68289d5864c00fb9e448b46f1e521d7d6c44f3c6e3bd52be84b33a3dc6aeb4` |
| `experiments/tests/test_ddm_ec2_oriented_adapter_trainer.py` | structural, byte-closure, retention, and gate tests | `6cd40ff43ded7694dc9b1e4f8c464a233608ed33c490f125fe58810cea122d0c` |

The source hashes in the sealed requests match the first two rows exactly. The EC1 source pins are
also embedded: design commit `fa29eb9ea17d3bfd5138478470600f322050634d`, EC1 final SHA
`bb0a6582745492dc77e4dc8a6556248bea5cc4084b06de028a4b1aa2aec76bd3`, and predecessor fire-order
SHA `0d403be3b5af461c9e6e8c9caf77066b126f22be853c51d85509d0bcc8a6185c`.

## Actual training contract

### Byte closure

- QAT forward uses the same per-tensor int8 weight scale and float16 bias storage as
  `serialize_module`.
- A structural parity test proves QAT-forward and serialized receiver parse-back are bit-identical.
- Every stage packages both live and EMA modules into deterministic archives and repeat archives.
- The final endpoint consumes the final EMA **serialized parse-back model**, not the float shadow.
- Break-even uses the measured EC1 archive efficiency `0.785 flips/B` against the exact archive
  delta. For EC1's design price `+1,707 B`, the bar remains 1,340 flips.

### Objective and population

The target is the exact T4 GT argmax field, SHA
`91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248`. The base field is SHA
`7648ad42e9f21942f86e81b97cabf46b710af747bba0909f7837ef3891232727`; direct full-field comparison
is 34,970 flips over 117,964,800 pixels.

Each of three stages visits all 600 pairs exactly once through a deterministic 20-stratum hardness
permutation. No prefix is used. Weighted CE is anchored to the measured error prevalence:

`117,964,800 / 34,970 = 3,373.314269373749`.

The stage ratios move from error birth to collateral preservation:

| Stage | Steps | LR | Base-error weight | Correct-pixel weight | Approx. total error:correct mass |
|---|---:|---:|---:|---:|---:|
| `10_target_birth` | 600 | `1e-3` | `3373.314269373749` | `0.25` | `4:1` |
| `20_balanced_descent` | 600 | `3e-4` | `3373.314269373749` | `1.0` | `1:1` |
| `30_collateral_finish` | 600 | `1e-4` | `843.3285673434373` | `1.0` | `1:4` |

The endpoint is full n600 at batch 16 and retains the complete argmax field plus all batch-level
pre-R, camera, scorer-input, logit, target, and base-error payloads.

### Wall-clock derivation

The charter's measured upper anchor is 900 seconds for an n600 T4 field pass at batch 16:

- `900 / 600 = 1.5 s/pair`.
- Training budget is 1,800 optimizer steps.
- A conservative autograd work factor of three forward-equivalent passes gives
  `1,800 * 1.5 * 3 = 8,100 s`.
- Full-n600 endpoint reserve: 900 s.
- checkpoint/package reserve: 900 s.
- projected total: 9,900 s.
- Modal hard cap: 10,800 s = 3 T4-hours.
- price cap: approximately $1.80 at the charter's $0.60/T4-hour assumption.

The factor of three is an analytic forward/backward work allowance, not a measured EC2 trainer
runtime. The request labels it as an assumption; the retained worker log and stage timings will
replace it when MAIN fires. The worker checkpoint-pauses at 10,500 seconds so the Modal timeout has
300 seconds to commit the last complete live/EMA checkpoint.

### Resume and payload retention

- The same immutable `run_id` is the required `--resume-from` value.
- Every optimizer step writes all materialized receiver/scorer tensors, then a distinct live and
  EMA checkpoint. Stage-boundary checkpoint pairs are also distinct and never overwrite prior
  stages.
- The live checkpoint includes model, optimizer, Python/NumPy/Torch/CUDA RNG state, and the complete
  typed training config. The EMA checkpoint includes the shadow, warmup/update count, decay, and the
  same config.
- Narrow crash windows replay into exact-byte verification. Existing retained payloads are not
  overwritten; a mismatch blocks.
- Endpoint payloads are batch-atomic. A full n600 argmax file is assembled only after all batch
  receipts exist.
- The mounted Modal volume keeps all training payloads. Local harvest commands pre-create their
  destination directories. No cleanup or deletion is authorized by this seal.

CUDA reproducibility mirrors the CP135 public receiver rail: seeded Python/NumPy/Torch/CUDA RNG,
IEEE fp32 products, no cuDNN autotuning, and the receiver's existing non-forced deterministic-kernel
choice. The exact serialized endpoint field and deterministic archive repeat are the verdict
surfaces.

## Sealed custody

Root:

`/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned/ddm_ec1_20260814/main_cuda/`

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `SEAL_RESULT.json` | 4,546 | `4bb41e64b375303992c8199afd6725b7ff4847cee571eed05ecc1daede95c8ab` |
| `SEALED_FIRE_ORDER.json` | 8,443 | `f5f7782591639ce55d6b70af99c0e0cac37c42f1bdb6aef6c275766081f6834c` |
| `SEALED_ORIENTED_REQUEST.json` | 8,678 | `b1177cb5a8fb69c3a8b4edbdf0febebabba0ce7b48cf5b9a5e34d6f58e94ab0f` |
| `SEALED_CLASS_ONLY_REQUEST.json` | 8,714 | `241587bf129140caa5d9ba6bc6dcda6e02f90c504340038eb2ba4312c1d56c9e` |
| `SEALED_UNDIRECTED_REQUEST.json` | 8,714 | `f57749b69e2b3042cc7397c8137a66cd0346f8c7a5cfa60c40d2a96d0069e722` |
| `fire_inputs/cp135.archive.zip` | 186,252 | `6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6` |
| `fire_inputs/cp135.runtime.zip` | 50,654 | `db5c10959a21c9ae327de0683bca9ddbacb868deadc79db9925d3c924180ddda` |
| `fire_inputs/decoded_tokens_n600.npy` | 117,964,928 | `03f5379d70e4bbd88e125cfbfb785cf5473315c70a5b78661fa426bb3e96e0f4` |

The SSD preflight measured 48,204,193,792 free bytes against 2,265,448,576 required for local fire
inputs and reserve. The structural toy gate is retained under `toy_gate/`; it proves exact zero
identity and nonzero semantic movement on synthetic 8x12 tokens. It is explicitly not scorer or
empirical evidence.

`SEAL_RESULT.json` records `modal_dispatched=false`. This arm did not claim a scorer lane or contact
Modal.

## Dispatch topology

Dispatch 1 is physically restricted to `family="oriented"`. Its entry point rejects any other
family before claiming or spawning. The second entry point holds the equal-parameter
`class_only`/`undirected` controls and refuses to spawn unless the harvested oriented
`SELECTED_RESULT.json` contains `clears_oriented_break_even=true`. Both control request hashes are
explicit CLI arguments; the control gate cannot accept a modified request by hashing it at fire
time.

The exact commands, lane IDs, single-flight wiring, recovery commands, Modal volume paths, harvest
commands, EC1 package command, and RE1T/JS1B consumer order are in `SEALED_FIRE_ORDER.json`.

## Verification

- `12 passed` — `experiments/tests/test_ddm_ec2_oriented_adapter_trainer.py`.
- `ruff check` passed on worker, dispatcher, and tests.
- `py_compile` passed on worker and dispatcher.
- Two `review_tracker.py mark-file ... --status reviewed` passes recorded all 78 entities across the
  three new Python files (37 worker, 23 dispatcher, 18 tests per pass).
- Modal CLI loaded both `::main` and `::controls` and rendered every sealed argument as required.
- `tools/check_dispatch_cli_shell_hazards.py --strict` passed for the dispatcher.
- P0 payload-retention scan: 0 findings over 2 implementation files.
- `git diff --check` passed on all three code/test files.
- Upstream was read only and remains outside this patch.

No T4 worker, SegNet forward, candidate package, PoseNet pass, or evaluator was run. The only newly
executed tensors were the retained synthetic toy-gate tensors. Therefore there is no measured EC2
flip count and no score claim.

## RECALL EVIDENCE

Searched the full local research/code corpus with content queries including `implicit edge`,
`oriented`, `TokenBlock`, `589814`, `8380`, `EC1`, `RE1T`, `JS1B`, `SA1`, `post-render`,
`conditioned semantic`, `EMA`, and `deepest home context`; inspected the canonical equation registry
via `tools/list_canonical_equations.py --json`; searched `CANONICAL_RESEARCH_INDEX*`, the sub-0.15
DAG, hot state, task/queue surfaces, and lane records.

Charter seeds read in full included the EC1 memo, analyzer/packager, runtime, pinned fire order, and
RE1T/JS1B dispatcher/worker family. Beyond those seeds:

- `ddm_js1c_cuda_custody_stage0` showed the explicit post-render overlay worsened the T4 base. This
  kept EC2 at the pre-TokenBlock latent home and prevented a cheaper but already-refuted overlay
  trainer.
- `ddm_re1` showed a tiny same-object semantic edit can survive the real receiver but delivered only
  a two-flip-scale result. This supported differentiating through the real semantic object while
  rejecting a fixed-amplitude edit as the endpoint.
- `ddm_ec3_t4_targeted_events` closed the singleton/one-anchor event instance, while leaving learned
  coupled amplitudes open. This changed EC2 from a seeded capacity reference into a learned counted
  adapter with a full-population endpoint.
- The canonical EMA equation required stage-aware decay from the 1,800-step count and warmup; the
  deepest-home context equation reinforced injection at the earliest receiver-closed semantic home.
- The hot state supersedes the common contract's stale frontier line and says Seg must supply at
  least roughly 0.004 score units after the lossless rate axis closed. That kept the build directly
  aimed at the only EC1 mechanism capable of buying a Seg row.

I did not find, in the searched equation/index/DAG/task scopes, an already implemented resumable
T4 trainer for EC1's counted latent adapter. Reusing the RE1T/JS1B custody and dispatch substrate was
therefore appropriate; rebuilding its claim/volume patterns was not.

## Boundary and frontier

This unit completed a means, not the mission end. The effective and own-vehicle pointers are
unmoved. The live own-vehicle frontier remains **LC2 S=0.16959899569230852 @ 187,226 B
[contest-CUDA T4, n600]**.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN sole Modal scorer-lane router; consumer store:
  `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned/ddm_ec1_20260814/main_cuda/`;
  fire trigger: confirm no duplicate active EC2 oriented lane and Modal availability, verify the
  sealed request/source hashes, then execute `first_dispatch.exact_argv` from
  `SEALED_FIRE_ORDER.json`.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN EC1 packager and RE1T/JS1B measurement consumer;
  consumer store: the same `main_cuda/` root; fire trigger: recover the oriented run, verify its
  selected archive repeat and full-n600 retained endpoint, then execute the sealed harvest/package
  commands followed by the existing RE1T/JS1B component chain.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN sole Modal scorer-lane router; consumer store:
  `main_cuda/dispatch_controls/`; fire trigger: and only if the harvested oriented
  `SELECTED_RESULT.json` proves `clears_oriented_break_even=true`, execute the sealed second dispatch
  for `class_only` and `undirected`.

## LIVE-HYPOTHESES

- The oriented decoded-token context is still the best live EC1 family because its cross-fit AUROC
  and 8,380-error top mass substantially exceed the 1,340-flip price bar, and the conditioner acts
  before nonlinear spatial mixing where a nonzero control already moved real receiver values.
- QAT plus exact parsed-module endpoint may retain enough of that selectivity to clear break-even;
  this is plausible because the adapter's archive shape and quantizer are already only about 1.7 KB,
  but it remains untested until the T4 run.
- A learned amplitude can outperform the fixed tiny semantic edits that survived with negligible
  benefit, because EC3 closed singleton/fixed-event instances rather than this coupled latent field.
- If oriented clears, the two equal-parameter controls can separate orientation information from
  generic added capacity on the same T4 receiver and schedule.

## DEAD-ENDS

- Local CPU or MPS training as authority is closed by the measured local/T4 axis mismatch and the
  charter's explicit authority boundary.
- Prefix-sampled endpoint verdicts are closed; the worker uses deterministic full-population
  stratification for training and full n600 for the endpoint.
- Float-model endpoint evidence is closed; it would not prove the counted int8/float16 module, so
  the endpoint now reloads serialized parse-back bytes.
- Post-render explicit overlays are closed for this route by JS1C/SA1 T4 evidence; EC2 injects only
  at the latent pre-TokenBlock home.
- Seeded nonzero EC1 modules are closed as candidate evidence; they remain structural capacity
  controls only and are never scored as trained candidates.
- Firing `class_only` or `undirected` before an oriented break-even win is closed by the separate
  hard-gated entry point, not left to operator convention.
