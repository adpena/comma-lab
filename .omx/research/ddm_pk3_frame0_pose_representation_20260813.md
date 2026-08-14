# DDM PK3 — frame-0 pose representation

PK3 built a real, receiver-consumed frame-0 pose representation on CP135 and
closed its byte surface. The selected child is `186,283 B`, only `+31 B` over
CP135, with a `42 B` counted temporal-control payload. Its archive repeat is
byte-identical, its overlay parses back exactly, its expanded `600 x 12`
signed-int12 coefficient lattice is receiver-exact, and nine frame-0 samples
were rendered through the real CP135 uint8/R surface.

PK3 did **not** measure a pose improvement. This arm did not own the scorer
slot and made zero PoseNet, SegNet, Modal, contest-CPU, or contest-CUDA calls.
The only pose signal is a deliberately non-authoritative `n=9` toy bracket.
That bracket improved in-sample but worsened leave-one-pair-out, so it is not a
bankable result. One full-n600 dual-axis measurement is sealed and queued to
adjudicate the exact bytes; it was not fired.

## Result

| Surface | Result | Authority |
|---|---:|---|
| Exact selected archive | `186,283 B`, SHA-256 `bf38b800...e18fda` | `[macOS-CPU scorer-free exact bytes]` |
| Delta from CP135 | `+31 B`; rate debit `+2.0641627546787313e-5 S` | `[macOS-CPU scorer-free exact bytes]` |
| Counted P0J1 payload | `42 B`, SHA-256 `4578fbab...59d97` | `[macOS-CPU scorer-free exact bytes]` |
| Representation | 6 uniform temporal knots, 12 CP135 coefficient dimensions, 4 nonzero signed four-bit controls | receiver-closed structure |
| Expanded lattice | 480 nonzero coordinate deltas; 0 signed-int12 violations | `[macOS-CPU scorer-free receiver parse-back]` |
| Archive repeat | byte-identical | `[macOS-CPU scorer-free exact bytes]` |
| Real frame-0 render smoke | pairs `17,105,133,176,178,510,517,523,532`; retained `27,468,200 B` uint8 array | `[macOS-CPU scorer-free real CP135 render/R]` |
| Toy base pose MSE | `1.6211002068381195e-4` on 9 non-random hard pairs | `TOY-BRACKET`, not population d_pose |
| Toy selected in-sample MSE | `1.3060700734834473e-4`, `-19.43%` | `TOY-BRACKET` |
| Toy selected LOPO MSE | `1.8918371289280262e-4`, `+16.70%` | `TOY-BRACKET`; counter-signal |
| PoseNet / SegNet / complete S | **not measured** | no claim |

The exact archive is at
`/Volumes/VertigoDataTier/pact/ddm_pk3_20260813/compiled_candidates/k06_r1e-04_g0.50/archive.zip`.
The complete receipt is
`/Volumes/VertigoDataTier/pact/ddm_pk3_20260813/FINAL_RESULT.json`.
The EU4 consumer pointer is
`/Volumes/VertigoDataTier/pact/ddm_eu4_pose1000_joint_20260813/retained/PK3_HANDOFF.json`.

## What the representation is

`P0J1` stores a small video-derived lattice of signed four-bit controls at
uniformly spaced temporal knots. Generic receiver code expands those controls
by deterministic integer linear interpolation to 600 pairs and adds the result
to CP135's real 12-dimensional frame-0 signed-int12 coefficient lattice. The
receiver rejects reserved nibbles, aliases, trailing bytes, geometry drift, and
int12 overflow.

The payload does not contain PoseNet targets, pixels, scorer weights, a learned
spatial basis, or a hidden per-pair target table. The spatial basis and public
render are the already-shipped CP135 receiver object. The counted controls were
solved to reduce the retained GT-minus-CP135 Pose residual through retained
PoseNet Jacobians, then rounded into the exact receiver lattice. This is the
solve-first Gauss-Newton form requested by the charter; descent remains a
possible finisher only after an exact-current Jacobian surface exists.

## Reach curve and verdict

The deterministic ladder materialized and retained `54/54` candidates:
6 knot counts x 3 ridge values x 3 gains. Exact archive deltas ranged from
`-11 B` to `+113 B`. Of those candidates, `29/54` were full-n600 int12-valid;
`23/54` were both valid and nonzero. Every one of those 23 improved the
in-sample linear model, but **0/23** improved leave-one-pair-out against the toy
base.

The valid in-sample byte/MSE Pareto points were:

| Candidate | Exact delta bytes | Toy in-sample MSE | Toy LOPO MSE |
|---|---:|---:|---:|
| `k08_r1e-04_g0.50` | `-2` | `1.197173440958005e-4` | `1.938618193520014e-4` |
| `k08_r1e-06_g0.25` | `+6` | `9.933293047901181e-5` | `2.014230988814875e-4` |
| `k12_r1e-06_g1.00` | `+33` | `7.287366134824945e-6` | `3.9084129827421366e-4` |
| `k08_r1e-06_g1.00` | `+45` | `5.128277820725149e-6` | `3.428655184049801e-4` |

The selected `k06_r1e-04_g0.50` point is not on that in-sample Pareto curve.
It was selected because it has the least-bad LOPO error among the exact-valid,
nonzero, in-sample-improving candidates. That is the conservative choice for
one exact adjudication, not evidence of generalization.

Verdict scope: **INSTANCE / TOY-BRACKET** — this exact P0J1 representation fit
from nine non-random QS1 Jacobian pairs does not provide a reliable local pose
prediction. The representation family is not closed: the Jacobians were
measured with JS6 candidate frame-1 partners rather than the exact CP135 base
frame-1, and the sample is too small and biased to bank a negative. A full
exact row can still adjudicate this exact archive without transferring the toy
model into a score claim.

At the live CP135 contest-CUDA reference `d_pose=6.885642960696714e-6`, a
`+31 B` child needs only `d_pose < 6.851428816430828e-6`, a `0.497%` reduction,
to beat its rate debit if Seg is unchanged. The charter's stronger
`d_pose <= 3.44e-6` goal remains unmeasured.

## RECALL EVIDENCE

The recall pass searched the full `.omx/research/` corpus by content with:
`frame0 pose|frame_0 pose|#715|ddm_p1|PK2|PZ2|PZ4|PZ4R|direct_v6|PO1|ps135|hy1|#249|solve don't train|QS1|PoseNet Jacobian`.
It also queried `.omx/research/CANONICAL_RESEARCH_INDEX*`, the
`sub015_DAG_*` FEED blocks, design/SPEC files, task-ledger rows, and the full
canonical-equation registry through
`.venv/bin/python tools/list_canonical_equations.py --json`, filtered for
`pose|frame0|Jacobian|scorer obligation|carrier|rate|score`.

The charter's named seeds were confirmed, and recall found four constraints
beyond their short descriptions:

- `.omx/research/codex_findings_ddm_p1_frame0_pose_quotient_carrier_20260725T143303Z_codex.md`
  closes the old shared, quantized, parent-additive 24x32 PCA chart at
  formulation scope: its best rank-1 row had `d_pose=19.894931...` at `3,520 B`,
  and higher rank crossed the uint8 trust region. This changed the child from
  another stored spatial basis to a tiny control lattice over CP135's already
  shipped real basis.
- `.omx/research/ddm_pk2_pose_carrier_representation_20260809/RESULTS.md`
  prices the older physical carrier at 23,384 B, with basis and coefficient
  sections dominating. This made reusing CP135's basis and counting only the
  correction controls load-bearing.
- `.omx/research/ddm_pz4r_full_n600_eval_20260813.md` closes the exact
  direct-v6 instance: its receiver-closed archive reached `d_pose=0.631014...`.
  Together with PZ2's cheap stored-target packet, this prevented a target table
  or target-conditioned predictor from being relabeled as the new carrier.
- `.omx/research/ddm_qs1_frame0_schur_coupled_solve_20260813.md` exposes the
  real CP135 `600 x 12` signed-int12 actuator and nine unique retained
  Jacobian pairs. It enabled a real receiver build, but its Jacobians are bound
  to candidate frame-1 partners. That changed the planned local result from an
  advisory measurement to an explicit toy bracket and forced exact evaluation
  to remain the verdict.

The registered `posenet_luma_chroma_sensitivity_asymmetry_v1` and
`scorer_obligation_matrix_factorization_v1` laws reaffirmed that frame 0 is the
Seg-free place for the pose actuator. The registered
`pose_jacobian_basin_conditioning_v1` law added the reachability warning:
Jacobian conditioning is necessary, not sufficient, and a stale or narrow
Jacobian surface cannot certify the target tube.

No already-built sub-1,000 B, jointly solved, exact-CP135 frame-0 pose
representation with an n600 d_pose receipt was found in those searched scopes.

## Reproducibility, retention, and boundaries

- Run store: `/Volumes/VertigoDataTier/pact/ddm_pk3_20260813/`; the preferred
  SSD preflight passed with a 4 GiB expected run plus 8 GiB reserve.
- Seed `135`; source pins, fit, receiver compile, and fire order have separate
  checkpoints. The runner resumes from the durable output store.
- All 54 control arrays, expanded lattices, overlays, carrier sources, Brotli
  streams, model sections, archive members, and archives are retained with
  bytes and SHA-256. The selected archive, repeat archive, code lattice,
  runtime tree, runtime bundle, and rendered frame-0 samples are also retained.
- The first execution failed closed when the model winner overflowed signed
  int12. Its partial compile was not deleted; its 26-file tree is certified in
  `RETAINED_PARTIAL_COMPILE_RECEIPT.json`. Selection now requires zero
  full-n600 int12 violations.
- The sealed request and its three retained fire inputs were reloaded through
  the actual dispatcher validator. No job was launched and no lane was claimed.
- Measured now: exact bytes, exact coder/container deltas, deterministic repeat,
  exact overlay and coefficient parse-back, and real frame-0 rendering for the
  nine smoke pairs.
- Not measured now: PoseNet d_pose, SegNet d_seg, public full-n600 decode,
  complete S, contest-CPU, or contest-CUDA. Structural frame-0 Seg freedom is
  not substituted for the owed exact decode/component receipt.
- The common contract's scorer ownership overrides the charter's local scorer
  rung for this arm: `owns_scorer=false`, scorer calls `0`, and the scorer step
  is queued.

The live effective pointer remains CP135
`S=0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`.
The own-vehicle frontier remains LC2
`S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: `MAIN` sole scorer-lane router; consumer
  store: `/Volumes/VertigoDataTier/pact/ddm_pk3_20260813`; fire trigger: MAIN
  confirms no active n600 exact-eval/Modal lane, claims
  `ddm_pk3_dual_axis_n600_20260813`, verifies the sealed request and all three
  input SHAs, then executes `exact_command_argv` from `SEALED_FIRE_ORDER.json`.
  Harvest both Pose-vector passes and the Seg field, recompute complete S from
  exact components and archive bytes, and promote only if d_pose, Seg, rate, and
  complete-score gates all pass.

## LIVE-HYPOTHESES

- The exact `+31 B` child may still clear its unusually low `0.497%` pose
  break-even even though the toy LOPO screen worsened. This is plausible because
  the reused Jacobians are bound to different frame-1 partners and the nine
  sampled pairs are not population-representative; only the sealed n600 row can
  resolve the sign.
- A stratified-random `n>=32` Jacobian bank measured on the exact CP135 base
  pair object may make the same P0J1 representation generalize. This is
  plausible because the current in-sample fits show the 12-dimensional actuator
  can move the retained residual, while LOPO identifies sampling/object binding,
  not byte capacity, as the immediate failure.
- A bounded exact integer descent after a fresh Gauss-Newton fit may recover the
  remaining pose tube without increasing payload. This is plausible because
  the receiver payload stores the lattice controls, not the solver, and QS1
  already showed exact integer finishing can cancel local pose residuals on its
  bound objects.

## DEAD-ENDS

- Reusing the old P1 shared 24x32 PCA basis is closed at FORMULATION scope: its
  receiver-realized ranks missed the pose tube by orders of magnitude and
  crossed the uint8 trust region.
- Retrying PZ4R direct-v6 or relabeling a compressed `600 x 6` target table as a
  pose carrier is closed: direct-v6 measured `d_pose=0.631014...`, and stored
  targets are not a jointly solved receiver representation.
- Selecting a control payload from in-sample linear error alone is closed for
  this instance: `23/23` valid nonzero candidates looked better in-sample while
  `0/23` improved leave-one-pair-out.
- Allowing a toy-model winner to reach compile without a full-n600 signed-int12
  check is closed: the first run failed on a saturated coefficient; selection
  now fails closed on any lattice violation and the partial bytes remain
  certified.
- Treating the nine reused QS1 Jacobian pairs as an n600 advisory result is
  closed: they are non-random and bound to different frame-1 objects, so they
  bank neither a positive nor a family-level negative.
