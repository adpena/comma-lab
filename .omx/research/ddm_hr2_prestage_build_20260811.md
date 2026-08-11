# ddm_hr2 prestage build — camera-uint8 closure and fail-closed HR1 apparatus

Date: 2026-08-11

## Outcome

HR2 completed all six scorer-free prestaging items in its charter. The shared differentiable
round-trip now has a typed `camera_uint8` ordering with typed `bicubic` and `bilinear` lift
kernels. Its legacy default remains the old operation order and is covered by byte-identity
tests. The four HR1 arm descriptions, checkpoint/resume/payload manifests, content binder, and
shape-only memory compiler exist as typed apparatus, but deliberately compile to no command and
cannot execute until terminal objects, real consumers, and fresh memory receipts are bound.

This is an implementation-blocker closure, not a candidate, score, or frontier movement. It did
not run SegNet, PoseNet, Modal, an HPAC encoder, a terminal renderer, an arm, or an exact evaluator.

## MEASURED CONTROL

The authoritative retained run is
`/Volumes/VertigoDataTier/pact/ddm_hr2_prestage_build_20260811/retained_v2` on
`[scorer-free pixel/apparatus control]`. It decoded two real frames from `upstream/videos/0.mkv`
through the canonical `frame_utils.yuv420_to_rgb` path.

| Control | Measured result |
|---|---:|
| Bicubic-lift training camera values vs independent public-order reference | 6,104,016 / 6,104,016 exactly equal |
| Bilinear-lift training camera values vs independent public-order reference | 6,104,016 / 6,104,016 exactly equal |
| Bicubic full input-gradient tensor vs CPU-torch reference | 1,179,648 values, max relative error 0 |
| Bilinear full input-gradient tensor vs CPU-torch reference | 1,179,648 values, max relative error 0 |
| Legacy-order vs camera-uint8 scorer-grid RGB values changed | 1,126,626 values |
| Legacy-order vs camera-uint8 RGB-channel argmax changed | 8,288 pixels |
| Process maximum RSS | 413,007,872 B; below the 1 GiB cap |
| Retained payload tree | 34 records, 65,160,742 B; records SHA-256 `0e5e94b8205b7bcbd2865e4170d3f425e764492acca62cf23286e0dcac380914` |
| Tree-manifest file SHA-256 | `0ad9a31966cb8f026e3e2262aa563e1937e39f2ccc7c021dae10192fcb348f16` |

The nonzero prior-law prediction passed only for a scorer-free RGB-channel argmax proxy. That is
evidence that operation placement changes real pixels; it is not a SegNet argmax measurement and
does not establish a change in `d_seg`, `d_pose`, or score. The charter's pixel falsifier did not
fire: both typed lift kernels produced camera bytes exactly equal to their independent receiver-order
references on both retained real frames.

An earlier retained run remains at
`/Volumes/VertigoDataTier/pact/ddm_hr2_prestage_build_20260811/retained`. It is superseded for source
binding because review fixes changed source hashes. Its arrays compare byte-identically with
`retained_v2`; it was retained rather than deleted under the payload law.

## IMPLEMENTED SURFACES

- `src/tac/differentiable_eval_roundtrip.py` adds `EvalRoundTripOrdering`,
  `CameraLiftKernel`, and the exact hard-forward camera-uint8 path. The default remains legacy.
- `src/tac/witness_dsl/hr1_prestage.py` defines four distinct immutable arm configurations and
  program factories. Each result has empty `argv`, no bound consumer, and
  `execution_allowed=false`.
- The same module defines typed payload, checkpoint, and resume manifests with canonical hashes,
  frozen/trained-state invariants, and same-directory atomic JSON replacement.
- The binder stream-hashes the HY1 memo, solved-token payload, HPAC stream, relevant sources, and
  read-only public intake. Seven ps135 terminal roles are typed unresolved values with no fake path,
  size, or hash.
- The memory compiler reports tensor-storage lower bounds of 153,203,472 B for arms A-C and
  157,135,632 B for arm D. These are not peak-memory projections. Every arm returns `REFUSE` until
  exact shapes and a matching fresh real-config memory probe exist.
- `experiments/ddm_hr2_prestage_build.py` is an SSD-only, storage-preflighted, scorer-free control
  runner that retains each materialized array plus hashes, source bindings, command, and a tree
  manifest.

The authoritative result binds the edited helper at SHA-256
`9b550354f38a7422fdc53ee918f6c6a07e5eeb2f543d535920d05fb2740afcec`, the prestage module at
`b9c4291662c65a77ff49935b9b9af69380fcac8737b15532df86fa691ccf9eaa`, and the runner at
`93b7eeb309b628af2c528eb6317741009411b9552adb808d45cdd11793c636ae`.

## BINDING AND CUSTODY

The binder verified the HY1 solved tokens as 117,964,800 B with SHA-256
`2b0bdfc38a131ab1ebc3a2c2153a79b1ba23be0037adda66d01ab56f29f4fed5` and the retained HPAC stream
as 114,717 B with SHA-256
`9def0a4ba849757d473ba2a23cb0fd5370f2566355e5a5cfd398f847349636e8`.
The public PR130 intake was read only. The binding manifest is terminal because it names unresolved
terminal objects explicitly and sets global execution permission false; it does not invent
placeholders.

Checkpoint manifests require distinct stage-encoded checkpoints and complete state. Frozen decode
forbids optimizer, EMA, and trainable state. Trained arms require optimizer and EMA state. Resume
manifests bind configuration and checkpoint hashes. These are apparatus guarantees tested in
isolation; no claim is made that a real arm has resumed.

## VERIFICATION AND REVIEW

- `src/tac/witness_dsl/tests/test_hr1_prestage.py`: 33 passed.
- `src/tac/tests/test_differentiable_eval_roundtrip.py`: 53 passed.
- Non-MLX integration selection covering the training-loop and Z7 wiring: 43 passed, 11 warnings.
- Ruff on all five Python implementation/test files passed with `RUF043` ignored only for two
  pre-existing regex-literal style findings in the old round-trip test file.
- `git diff --check` passed. The payload-retention preflight reported zero findings on the runner,
  helper, and prestage module.
- A larger MLX selection reached 44 passes and 11 failures. Every failure was the managed host's
  `metal::load_device` no-device error in `test_pr95_hnerv_mlx_training.py`; MLX/Metal integration
  therefore remains unverified here rather than being reported as a code failure or a pass.

The first adversarial review found two apparatus defects: the compiled typed config could be mutated
after hashing, and RSS was sampled twice. The config now exposes freshly decoded immutable content,
and the runner captures RSS once. All relevant checks and the retained real-frame control were rerun
after those fixes. A second review found no new defect. Review-tracker receipts and serializer custody
are the landing gate, not mechanism evidence.

## RECALL EVIDENCE

The recall covered the full `.omx/research` corpus, `CANONICAL_RESEARCH_INDEX*`, the canonical
sub-0.15 DAG, current hot state, task bridge, design/spec surfaces, and the canonical-equation listing.
Queries included `camera uint8`, `roundtrip`, `uint8`, `bilinear`, `bicubic`, `realization`, `HPAC`,
`checkpoint`, `resume`, and the HR1/HR2 identifiers.

The important beyond-charter finding was the already-recorded BH1 diagnosis in the canonical DAG:
the old trainer quantized after the scorer-grid downsample while evaluation quantized at camera
resolution, with a historical estimate of about 196 flips per frame. The DAG also records the prior
`apply_contest_faithful_roundtrip_nhwc` correction. This changed the plan from inventing a new operator
to exposing the settled operator ordering as a typed shared helper, preserving its legacy default,
and testing both the incumbent public receiver's bilinear lift and the canonical bicubic lift. No
re-measurement of the historical SegNet claim was performed because this charter owns no scorer slot.

## HONEST FRONTIER STATUS

The effective frontier remains cp135 at **S = 0.16195513827824176 @ 186,252 B**
`[contest-CUDA T4, n600]`. The own-vehicle frontier remains LC2 at
**S = 0.16959899569230852 @ 187,226 B** `[contest-CUDA T4, adjudicated, n600]`.
HR2 moved neither pointer.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: js1 realization successor; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hr1_preflight/content_bindings/`; fire trigger: the ps135 terminal safe-run receipt lands, after which bind exact terminal paths, bytes, and hashes before wiring any consumer.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: js1 realization successor with the governed launcher; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hr1_preflight/memory_probes/`; fire trigger: exact terminal tensor shapes and real arm consumers are bound, then run fresh per-arm real-config probes and admit only matching receipts.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: js1/#995 scorer successor; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hy1_solved_carriage/stage0_v14/`; fire trigger: terminal binding, lane claim, and fresh memory clearance all pass, then test the camera-uint8 placement with the real frozen scorer before any arm launch.

## LIVE-HYPOTHESES

- Camera-resolution quantization is likely scorer-material on the terminal vehicle because it changed
  1,126,626 scorer-grid RGB values and 8,288 RGB-channel argmax pixels over two real frames. Only a
  frozen-scorer test can establish whether those changes cross SegNet cells.
- Bicubic and bilinear lifts can select different camera bytes while both obeying camera-uint8 ordering.
  Keeping both typed should make terminal/public-receiver comparison attributable instead of folding
  a kernel mismatch into the quantization-order verdict.
- Consumer wiring should be mechanically low-risk because every unresolved dependency is typed and
  every program currently refuses execution, but this remains untested until the ps135 objects exist.

## DEAD-ENDS

- Treating the legacy helper as camera-faithful is closed: its operation order is deliberately distinct,
  and the retained real-frame control found nonzero pixel differences.
- Changing the default round-trip semantics is closed: existing callers require byte-identical legacy
  behavior and must opt into `camera_uint8` explicitly.
- Treating shape-only storage sums as launch clearance is closed: they omit allocator, model, optimizer,
  autograd, and runtime peaks, so the compiler refuses without a fresh matching probe.
- Treating typed arm schemas as runnable arms is closed: they have empty command vectors, no consumers,
  unresolved terminal bindings, and false execution permission.
- Treating the pixel control or RGB-channel argmax proxy as SegNet, PoseNet, score, or frontier evidence
  is closed by construction: no scorer or evaluator was loaded.
