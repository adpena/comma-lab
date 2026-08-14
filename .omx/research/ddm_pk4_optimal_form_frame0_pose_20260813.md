# DDM PK4 — optimal-form frame-0 pose control

Date: 2026-08-13

Authority: scorer-free source/receiver/preflight evidence only

Score claim: `false`

Pointer moved: `false`

## Conclusion first

PK4 built the real optimal-form execution surface but did not measure the
Jacobian bank or pose reach curve. The exact CP135 object and all scorer/runtime
sources are pinned, the random n64 floor is sealed, every materialized payload
is routed to the retained SSD store, and each byte rung has a fail-closed
no-fire order. The local Metal probe then failed with
`RuntimeError: [metal::load_device] No Metal device available`; the active lane
summary also showed an existing full-n600 Modal scorer job. PK4 therefore made
zero scorer calls and launched no local, Modal, CPU-authority, or CUDA-authority
work.

This is an execution blocker, not a negative verdict on frame-0 linear-response
control. No `d_pose(bytes)` curve, generalization sign, candidate archive, or
score was measured. The effective pointer remains CP135 at
**`S=0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`**, archive SHA-256
`6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`.
The own-vehicle frontier remains LC2 at
**`S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`**.
This unit did not achieve goal progress.

## Result

| Surface | Result | Authority |
|---|---:|---|
| Exact CP135 archive pin | `186,252 B`; SHA-256 `6eb1a3b7...edb6` | retained source custody |
| Exact CP135 decoded raw pin | `3,662,409,600 B`; SHA-256 `a641d1ef...ed47` | retained source custody |
| Source pins | `15/15` present and hashed | scorer-free preflight |
| Sampling | seeded temporal-stratified random `n64`: `48` train, `16` untouched holdout | deterministic plan, seed `20260813` |
| Storage | `103,079,215,104 B` expected + `8,589,934,592 B` reserve; passed with `227,303,604,224 B` free | SSD preflight |
| Metal | **FAIL**: no Metal device in this headless/sandboxed session | direct MLX materialization probe |
| Scorer ownership | not owned; another n600 lane active | retained lane summary |
| PoseNet / SegNet calls | `0 / 0` | run receipt |
| Jacobian bank | `0/64` pairs | not measured |
| Generalization gates | `0/3` | not measured |
| Byte-closed rungs | `0/3` | compile structurally prohibited before gate |
| Exact evaluations | `0` | not measured |
| Frontier movement | none | no score claim |

The retained sample is not a prefix. Its holdout set is
`[16, 62, 94, 123, 182, 219, 254, 295, 307, 360, 407, 437, 479, 488, 550, 571]`,
one random member from each of 16 temporal strata. The remaining 48 selected
pairs are train-only. This obeys the measured m96 pose-prefix-bias law rather
than treating an anti-conservative prefix as population evidence.

## Built execution surface

`experiments/ddm_pk4_optimal_form_frame0_pose.py` is a real resumable runner,
not a proxy implementation. Its `prepare` command never constructs a scorer.
Its `measure` command refuses before scorer construction unless both a live
Metal arithmetic probe and a current `MAIN` single-flight ownership receipt
pass. When admitted, it:

1. renders the exact CP135 signed-int12 frame-0 actuator and pairs it with the
   exact CP135 decoded frame 1;
2. retains every code batch, rendered uint8 frame, two-frame PoseNet input,
   preprocessed YUV6 tensor, and six-vector before summarizing it;
3. builds central-difference `6 x 12` Jacobians and a fresh damped GN step;
4. finishes every per-pair solve by strict exact int12 coordinate descent;
5. fits each rung only from train-pair exact-GN deltas, tuning 20
   ridge/gain cells on leave-one-pair-out modeled reduction;
6. evaluates the selected rung twice on the untouched holdout; and
7. compiles only if LOPO is positive and held-out pose-MSE reduction is
   positive and at least twice the repeat-derived pair-noise RMS.

The compile entrypoint independently reloads and validates that typed gate
receipt. A direct call cannot bypass it. A passing rung receives a deterministic
archive repeat, exact receiver parse-back, and an unchanged-worker T4 request
whose helper structurally hardcodes `local_pose_delta: 0.0` and
`pose_unmeasured: true`. `MAIN` remains the only fire owner and each request is
capped at `$0.16`.

`experiments/ddm_pk4_frame0_pose_overlay_runtime.py` adds strict counted P0J2.
PK3's P0J1 used a one-byte knot count with a 64-knot semantic cap, so it could
not express the charter's approximately 1 KB rung. P0J2 uses a versioned uint16
knot count while preserving signed-four-bit controls, canonical nibble checks,
deterministic integer interpolation, trailing-byte rejection, and full-n600
int12 validation. The three planned raw counted payloads are:

| Rung | Knots x dimensions | Exact raw P0J2 bytes | Current disposition |
|---|---:|---:|---|
| `rung_42` | `6 x 12` | `43 B` | `BLOCKED_BEFORE_SCORER_LAUNCH_NO_METAL` |
| `rung_250` | `40 x 12` | `247 B` | `BLOCKED_BEFORE_SCORER_LAUNCH_NO_METAL` |
| `rung_1000` | `165 x 12` | `997 B` | `BLOCKED_BEFORE_SCORER_LAUNCH_NO_METAL` |

These are representation capacities, not measured archive deltas or pose
results. The actual full-container byte count is computed only after a rung
passes generalization and compiles.

## Recall consumed

- PK3's 54-candidate compile/custody pipeline, deterministic ZIP, Brotli-q11
  carrier splice, runtime parse-back, int12 saturation guard, and retained
  payload law were reused. Its model was not reused: the n9 pairs were
  non-random, their Jacobians used changed JS6 frame-1 partners, and `23/23`
  in-sample winners became `0/23` LOPO winners.
- The #715 P1 shared low-rank carrier and matched control closed that old
  stored-basis formulation: rank 1 reached `d_pose=19.8949` at `3,520 B`, and
  rank 6 used `21,045 B`. PK4 therefore reuses CP135's already-counted spatial
  actuator and stores only a tiny temporal control lattice.
- #249/#251 established the necessary mechanism law: Pose inverse-solving must
  keep uint8 and the real resize/receiver round trip in the solve; MLX is a
  search/research instrument, never score authority. Their image-space
  per-pair payload route was rate-prohibitive and was not resurrected.
- PZ4A's best variable-precision rung saved only `500 B` gross and paid
  `2,732 B` for allocation metadata on its tested object. That precision map is
  not transferred here; PK4 tunes a fixed canonical four-bit lattice and
  measures the complete archive if admitted.
- `tac.lie` supplies generic SE(3)/B-spline mechanics but no current-object
  CP135 control or d_pose receipt. It remains a possible nonlinear successor,
  not evidence for the linear rung.
- PO1/PZ4R supplied the unchanged worker family and exact component custody,
  not a reusable compensation vector. Their failures reinforce that every
  control must bind to the exact child object.
- EU4 supplied the load-bearing economics: Pose is `69.38%` of the gap; a
  `1,000 B` representation breaks even at `15.41%` pose reduction and halving
  CP135 d_pose projects about `-0.0017635663 S`. Those are planning thresholds,
  not PK4 measurements.
- A bounded search found an n600 Jacobian bank for LC2/PS135, but that vehicle's
  semantic state, carrier, archive, and frame objects differ from CP135. It was
  rejected rather than relabeled as an exact-object bank.

## Retention and reproducibility

Retained root: `/Volumes/VertigoDataTier/pact/ddm_pk4_20260813/`.

The scorer-free command was:

```text
.venv/bin/python experiments/ddm_pk4_optimal_form_frame0_pose.py prepare --output /Volumes/VertigoDataTier/pact/ddm_pk4_20260813 --resume-from /Volumes/VertigoDataTier/pact/ddm_pk4_20260813 --sample-count 64
```

It returned `0`. The MLX atexit hook repeated the already-retained no-device
exception after the typed blocker was written; it did not change the process
return code or launch a scorer.

Load-bearing receipts:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `FINAL_RESULT.json` | `2,223` | `e1389ffcf137234847c28c7409faacb4277faf03dd1e54f61d7b31f7b85ee669` |
| `checkpoints/stage_00_source_preflight_8e6728b15d055ba8.json` | `4,580` | `6c4f714353b2c3d678eec31befd7734b8c8deaec74f13784778ea038af3ee5f8` |
| `checkpoints/stage_05_sample_plan.json` | `892` | `27158f13b63339e607719732a29b98bd915e00552ece14a327364f7468e76261` |
| `retained/preflight/METAL_PROBE_AT_PREPARE.json` | `301` | `3ace288f2d8e3c60f4762cc1114c546bba3369f60eee8a6179b31fbec056f818` |
| `LOCAL_METAL_FIRE_ORDER.json` | `1,266` | `9a53ccf1d0e557bed6ac266514619a08a19fbdf9b7a1d8774bc0e5dc5d77cb59` |
| `rung_42/SEALED_NO_FIRE_ORDER_PREFLIGHT.json` | `615` | `ea59067778441a88b5fd954f65501d10a563b9fb39bb8dc1bdd7eecba94f9bdc` |
| `rung_250/SEALED_NO_FIRE_ORDER_PREFLIGHT.json` | `617` | `cd82a2761d73a04a5fe6187d95da046d5e3f2ea214a1f7ee687a798e87aefb3c` |
| `rung_1000/SEALED_NO_FIRE_ORDER_PREFLIGHT.json` | `618` | `279845ae3c0c9a260297df0fd1b3d3325100d36f6ae0b2dceedc20a38947a9f1` |

Every future Jacobian stage has its own immutable batch and pair result. Live
progress uses atomic `STATE.json`; completed source, sample, ownership, Metal,
bank, parity, tuning, gate, compile, and rung checkpoints have distinct names.
The expected storage scales with the requested sample census, so asking for
n600 cannot silently reuse the n64 free-space assumption.

## Verification and honesty boundary

- Focused suite: `10 passed` for overlay canonicality, exact rung lengths,
  int12 rejection, deterministic stratified sampling, train-only fitting,
  positive-plus-two-noise-RMS gating, LOPO gating, compile refusal, no-fire
  orders, structural pose placeholders, and scorer-construction ordering.
- Python compilation passed for both implementation modules and their test.
- Payload-retention audit: `0` findings across `2/2` PK4 implementation files.
- Review tracker: two reviewed passes over all three Python files; the second
  pass included fail-closed compile, per-rung tuning, scaled storage, MLX parity,
  resumption, and assumption-challenge checks.
- Repository developer preflight: **RED, 8/25 declared gates**. Individual
  adjudication found no PK4 path in any finding. The failures are existing
  broad repository debt: one state writer, one authoritative-tag site, 25
  legacy launchers, one AGENTS claim-helper documentation issue, 124 historical
  landing memos, eight other unregistered lane references, 56 substrate loss
  files, and 21 substrate trainer defaults. This unit does not claim a clean
  global preflight and did not modify those unrelated surfaces.
- Protected files named by the common contract were unchanged.

Measured now: exact source bytes/hashes, SSD capacity, deterministic sample
membership, direct Metal availability, lane state, P0J2 raw lengths, tests, and
typed no-fire custody.

Not measured now: any PK4 Jacobian, GN result, MLX/CPU pose vector, LOPO sign,
held-out pose reduction, pair-noise RMS, compiled archive bytes, public decode,
`d_seg`, `d_pose`, complete `S`, contest-CPU, or contest-CUDA. The generic code
and tests are means; they are not narrated as score progress.

The shared assumption challenged in review is that a globally smooth temporal
control lattice is the right compression chart. Violating it with a nonlinear
or joint-descent chart could unlock a real gain because physical motion and the
PoseNet response are nonlinear. That alternative is not opened by this blocker:
the charter first requires the correct exact-object linear test, and only a
negative generalization result should route the successor away from this family.

## NEXT_IF_RESUMED

- `BLOCKED-BEFORE-SCORER-LAUNCH` — owner: `MAIN` local scorer-lane router;
  consumer store: `/Volumes/VertigoDataTier/pact/ddm_pk4_20260813/`; fire
  trigger: a non-headless session passes the retained MLX Metal arithmetic
  probe, the current active n600 scorer/Modal lane is terminal, and MAIN writes
  a current single-flight ownership receipt at
  `/Volumes/VertigoDataTier/pact/ddm_pk4_20260813/SCORER_OWNERSHIP_RECEIPT.json`;
  then execute `exact_command_argv` from `LOCAL_METAL_FIRE_ORDER.json` and
  resume from the same store.
- `CONDITIONAL-ROUTE` — owner: `MAIN`; consumer store:
  `/Volumes/VertigoDataTier/pact/ddm_pk4_20260813/`; fire trigger: only after all
  three correct-bank rungs complete; if LOPO remains non-positive or every
  exact held-out reduction fails the two-noise-RMS gate, record the measured
  linear ceiling and route the next representation to nonlinear/joint descent
  without another linear-lattice iteration.

## LIVE-HYPOTHESES

- Exact CP135 frame partners plus fresh GN will remove PK3's binding mismatch;
  this is plausible because PK3 optimized Jacobians taken against changed JS6
  frame-1 objects, while PoseNet consumes both frames jointly.
- The approximately 250 B rung will turn LOPO positive; it has materially more
  temporal capacity than the 43 B rung without the 997 B rung's high-variance
  underdetermination, matching the prior-law prediction.
- Train-only ridge/gain tuning followed by untouched exact holdout will select
  a sparser, more stable control than PK3's in-sample ladder; PK3's `23/23` to
  `0/23` reversal makes this regularization/generalization seam load-bearing.
- If the linear family fails, nonlinear SE(3)/joint descent can still work;
  `tac.lie` supplies a generic motion chart and EU4 shows enough Pose score
  value to pay for a sub-1 KB representation, but neither claim is yet measured
  on the CP135 object.

## DEAD-ENDS

- Firing PK3's 42 B child is closed: its n9 model used biased pairs and wrong
  frame-1 objects, and every valid nonzero in-sample winner failed LOPO.
- Reusing the LC2/PS135 n600 Jacobian bank is closed: it belongs to different
  archive bytes, semantic state, carrier codes, decoded frames, and base pose.
- Reusing an old compensation vector after any child-object change is closed:
  PoseNet is a two-frame scorer and PO1/QS4 showed stale local control can invert.
- Contiguous-prefix sampling is closed: m96 measured the pose prefixes as
  2.54–4.21 times harder than the population, the exact false-negative shape.
- P0J1 for the 1 KB rung is closed: its 64-knot semantic cap cannot represent
  the requested capacity. P0J2 is the strict counted replacement.
- CPU fallback, MPS authority, or a borrowed pose number is closed for this
  execution: the charter requires local Metal compute, and only later matched
  contest-axis measurements can support a score claim.
- In-sample-only selection and compile-before-generalization are closed in code:
  the compile entrypoint requires a custody-verified positive LOPO and exact
  held-out two-noise-RMS gate receipt.
