# DDM RE1T — RE1 Round-1 T4 sign-gate dispatch prep (2026-08-13)

## Result

**READY_TO_FIRE_BY_MAIN.** The scorer-free arm adapted the proven SA1 T4 chain, retained every
fire input, sealed the exact request and detached command at the named consumer store, and did not
call Modal. The final seal is source-bound to implementation commit `27c828d1d11fbc1d88139abd06e96738ea23eb27`.

- Disposition: **QUEUED-WITH-A-FIRE-ORDER**.
- Owner: **MAIN sole scorer-lane router**.
- Consumer store:
  `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/probability_object_race/ddm_re1_20260813/full_n600_exact/round_01_singleton_best`.
- Request: `RE1T_T4_REQUEST.json`, 11,882 B, SHA-256
  `6d343139e9ebf7dd4cc1bca3ffb5d6b4b40f4c5d0c58b6d93219bb98ac5efb79`.
- Fire order: `RE1T_T4_FIRE_ORDER.json`, 5,138 B, SHA-256
  `b0ae14d1b4cda37a6e6e2601a1d0f0433dcefb8872ca354fa9ee0c9fd28900f2`.
- Fresh run/resume identity: `ddm_re1_round1_t4_gate_20260813`.
- Estimated spend: approximately **$0.16**, budget ledger **#381**, charter-recalled spend to
  date approximately **$2.4**. These are recalled planning values, not a new invoice or price.
- `score_claim: false`; `promotion_eligible: false`; pointer moved: **no**.

The exact MAIN command is:

```text
.venv/bin/modal run --detach experiments/ddm_re1t_modal_t4_sign_gate.py::main --sealed-request /Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/probability_object_race/ddm_re1_20260813/full_n600_exact/round_01_singleton_best/RE1T_T4_REQUEST.json --fire-input-dir /Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/probability_object_race/ddm_re1_20260813/full_n600_exact/round_01_singleton_best/re1t_t4_fire_inputs --expected-request-sha256 6d343139e9ebf7dd4cc1bca3ffb5d6b4b40f4c5d0c58b6d93219bb98ac5efb79 --output-dir /Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/probability_object_race/ddm_re1_20260813/full_n600_exact/round_01_singleton_best/re1t_t4_dispatch --detach --provider-detach-ack
```

## Sealed custody

The local seal rehashed the candidate, the complete runtime tree, and every distinct path-bearing
record in the RE1X blocker before writing the request. The runtime-pin guard parses the actual
candidate `inflate.py`, proves both candidate constants are compared in the fail-closed input
guard, and proves that guard is called.

| retained fire input | bytes | SHA-256 |
|---|---:|---|
| candidate `archive.zip` | 186,252 | `7be3eb94b229306278a6ed204e2c716d7aafa98f6f93c82a5d2be18822467dfa` |
| candidate runtime ZIP, archive excluded and supplied separately | 50,707 | `980219a74f6486753eba32238d47633b36ac65e576caa552ed7ae7b364d8379e` |
| RE1X blocker receipt | 11,846 | `197cfd2883e2c23c9f3e39cbe4fe1ce1b24953ebc81467de851a8281e4273a76` |

The candidate runtime has 25 files and tree SHA-256
`63b93187e83cb310d68031a2b08b65b1a5e2103e830cede4941a7d3df604dc75`.
Its `inflate.py` is 1,902 B, SHA-256
`c23f03d61c174f34e8f5f65935132d8b70e0171de718622874a842c86e225478`,
and pins the candidate archive at 186,252 B and `7be3eb94…7dfa`, not CP135. The blocker
receipt itself and all 28 distinct nested records rehashed successfully. Repeated preparation
produced the same 50,707-byte runtime bundle.

The final local storage preflight measured 137,144,168,448 B free against 67,357,669 B required
for the 248,805 B fire-input set plus a 64 MiB reserve. The remote contract blocks unless it can
retain 7,555,248,128 B of exact receiver/scorer payload plus a 4 GiB reserve. It retains raw,
SegNet input tensors, logits, the full argmax field, worker log, request, and every stage
checkpoint. Volatile free-space measurements are deliberately not placed in checkpoint digests.

The sealed loader rejects request drift, input-census drift, source drift, authority escalation,
candidate/runtime drift, and loss of the explicit Pose-unknown law. The remote worker independently
rehashes its own source and the inherited JS1B worker against the sealed request. It sets
`sys.dont_write_bytecode = True` before scorer imports so an upstream `__pycache__` cannot invalidate
the end-of-run snapshot.

## Gate and authority boundary

The retained same-instrument controls are the full-population T4 fields: GT SHA
`91d3ff11…e77248`, CP135 SHA `7648ad42…32727`, and CP135 base 34,970 flips out of
117,964,800 pixels. Candidate and base archives are byte-equal, so exact rate delta is zero.
With Pose explicitly unknown, the request stores `local_pose_delta: 0.0` only alongside
`pose_unmeasured: true`. The zero is a placeholder, not a measurement.

The derived Seg-only sign threshold is therefore one flip: at most 34,969 candidate flips is a
**provisional Seg admission**, worth `8.477105034722222e-7` score units per removed flip. It is not
a composition or score admission. A provisional result requires retained same-axis PoseNet-vector
measurement under the PZ4R law before any composition claim. If the candidate field is identical
to CP135, the local adjudicator returns `DEAD_INSTANCE_RECEIVER_NULL_IDENTICAL_TO_CP135`, folds the
instance, and forbids a pose job. The remote worker only measures and retains fields; all mixed-axis
arithmetic and disposition happen locally after harvest.

## What was measured and what was not

**Measured locally, scorer-free:** archive bytes/hashes; runtime-tree census/hash; runtime archive
pin; blocker receipt plus 28-record rehash; deterministic runtime ZIP; exact fire-input bytes;
source hashes; request/fire-order hashes; local storage headroom; dry-run seal and sealed-loader
round trip. The dry run reported `modal_fired: false`.

**Not measured:** candidate CUDA public decode, candidate raw frames, candidate T4 SegNet field or
flip count, PoseNet vectors or delta, contest-CPU, complete `upstream/evaluate.py` score, a frontier
delta, or any spend. No candidate quality conclusion follows from this prep.

The effective custodial frontier remains **CP135 S=0.16195513827824176 @ 186,252 B
[contest-CUDA T4, n600]**. The own-vehicle frontier remains **LC2 S=0.16959899569230852 @
187,226 B [contest-CUDA T4, n600]**. The stale advisory frontier in the common-contract snapshot
was not used in place of the live board.

## RECALL EVIDENCE

Recall searched the full `.omx/research/` corpus and arm receipts; canonical equation output from
`tools/list_canonical_equations.py --json`; `CANONICAL_RESEARCH_INDEX_20260629.md`; the current
`sub015_DAG_*` FEED; v7.5/v8 SPECs; CN5/task-ledger rows; lane and Modal-call stores; and
`main_hot_state.md`. Content queries included `ddm_re1`, `re1x`, `round_01_singleton_best`,
`7be3eb94`, `63b93187`, `receiver-null`, `runtime pin`, `20260813g`, `dont_write_bytecode`,
`PoseNet vector`, `pz4r`, `CUDA-locked public`, and `probability object`.

Beyond the charter seeds, recall changed the build as follows:

- RE1 proved only entropy-receiver closure for the one-cell Round-1 candidate at zero byte delta;
  it explicitly lacked public raw, SegNet, PoseNet, and complete S. This prevented token/parse-back
  equality from being used as a score proxy.
- RE1X retained the exact non-CUDA error and a 28-record blocker. Rehashing that whole set became a
  pre-seal gate rather than trusting the receipt headline.
- The SA1 addendum identified two separate real failures: upstream bytecode self-pollution and
  volatile storage fields inside immutable checkpoints. Both cures are present in the new worker.
- PO1 showed a candidate can parse back while its public runtime still pins the old CP135 archive.
  RE1T therefore verifies both the new SHA and byte count are actively compared before sealing.
- PZ4R showed receiver/parse-back success does not preserve Pose. RE1T therefore records Pose as
  unmeasured, makes any Seg admission provisional, and folds the pose job on exact receiver-null.
- CN5 and the DAG still described the missing empirical RE1 row, but no canonical equation supplied
  a substitute for the T4 field. The queued CUDA measurement remains the shortest lawful test.

No already-complete RE1 Round-1 public-receiver T4 field or Pose result was found in those scopes.

## Validation and landing

- New RE1T focused suite: **10 passed**.
- RE1X lineage suite: **7 passed**.
- Ruff and Python compilation: passed for dispatcher, worker, and tests.
- P0 payload-retention detector: zero findings across both new implementation files.
- Two clean post-fix review-tracker passes cover 17 dispatcher entities, 5 worker entities, and
  13 test entities. The implementation landed through the serializer as `27c828d1d1` with
  post-edit SHA checks and `[no-triality] [p0-ledger-ok]`.
- The inherited SA1 suite produced 5 passes and one environment timeout while
  `torch.testing.assert_close` lazily imported SymPy; a direct SymPy import was still incomplete
  after more than 120 seconds and was interrupted. No tensor comparison ran in that failure.
- The inherited JS1B suite produced 6 passes and its two already-documented fixture-drift failures:
  both attempt to read a missing synthetic `c1_target_argmax_n600.npy`. RE1T does not edit either
  inherited file. These broader-suite failures are not hidden and are not called green.

The serializer hook examined all three staged Python files for subset-selection and guarded-constant
issues and found none. Its fast mode examined zero full developer gates, so this is not a claim that
the entire dirty shared repository passes the full gate set.

## NEXT_IF_RESUMED

- **Disposition:** `QUEUED-WITH-A-FIRE-ORDER`. **Owner:** `MAIN sole scorer-lane router`.
  **Consumer store:** `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/probability_object_race/ddm_re1_20260813/full_n600_exact/round_01_singleton_best`.
  **Fire trigger:** MAIN confirms no competing full-n600 scorer or Modal single-flight, re-verifies
  request SHA `6d343139…efb79`, archive SHA `7be3eb94…7dfa`, and runtime-tree SHA
  `63b93187…dc75`, then executes the exact command in `RE1T_T4_FIRE_ORDER.json` and harvests with
  its recorded recover command.

## LIVE-HYPOTHESES

- The RE1 probability edit is receiver-null and will yield a candidate T4 argmax field identical
  to CP135. This is plausible because the decoded token plane, semantic plane, HP3 state, residual,
  and other downstream physical components closed identically in RE1; only the exact CUDA public
  renderer can decide it.
- The single categorical semantic-cell change survives the public renderer and removes at least one
  T4 Seg error at equal bytes. This is plausible because it is a full lattice change rather than a
  sub-LSB continuous perturbation, but the earlier favorable number came from an alternate component
  surface and cannot establish the T4 sign.

## DEAD-ENDS

- Local CPU/Metal public decode is closed for this exact instance: the hash-pinned F26 runtime
  rejects non-CUDA execution before raw materialization.
- Runtime patching, CP135-raw substitution, and entropy/token equality as a scorer proxy are closed:
  each changes the tested object or bypasses the public receiver.
- Reusing the failed RE1X run identity is closed because resume custody must remain byte-identical;
  RE1T uses the fresh `ddm_re1_round1_t4_gate_20260813` identity.
- Treating the Pose placeholder as measured, scoring remotely, or promoting a Seg-only gate is
  closed by the request/worker/local-adjudication guards.
- If the T4 candidate field is identical to CP135, this Round-1 instance is closed and no Pose job
  may fire; another characterization pass on the same receiver-null object would add no decision.
