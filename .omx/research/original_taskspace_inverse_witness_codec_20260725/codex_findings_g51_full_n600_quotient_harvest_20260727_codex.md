# G51 full-n600 quotient harvest — completed signal recovered

Date: 2026-07-27  
Lane: `lane_g51_conditional_selected_preimage_quotient_profiler_20260726`  
Axis: `[encoder-only exact-byte quotient diagnostic]`  
Pointer delta: none; the dynamic official target remains `0.172`

## Outcome

The full G51 run was already complete on the Vertigo SSD even though the
durable G51 findings still said it had not been launched. All 50 immutable
12-pair chunks, the aggregate, and the tool receipt exist. The aggregate
profiles all 600 pairs, passes its internal canonical-JSON self-hash, and binds
the exact fresh V15 P, selected-preimage teacher, batch-16 planning coordinate,
and implementation sources used by the run.

Exact aggregate custody:

```text
path          /Volumes/VertigoDataTier/pact/taskspace_conditional_quotient_profile_n600_fresh_v15_20260726_r2/aggregate_receipt.json
file bytes    631924
file SHA-256  e1a9497c3af61e0183a899f9bbaab2c9fa0c257cf9e24c1d7c40ac063bf71f2f
self SHA-256  129a5ce0de652311d8391809ea1576063139c80897519effe3875b07f1e8a612
chunk root    cb03e068f0ef6061aecea2e447b61ac865d0a2263ed1e1677344ac68a8d9e496
run binding   3e3e6b0cd6aefb91a1cc8ac1cb46b05b64fee03689d876d59979dbe66206055c
chunks        50/50
pairs         600/600
```

This closes the launch-status ambiguity and recovers an orphaned negative with
proper verdict scope.

## Exact result

No tested exact local/block basis is remotely close to the 53,622-byte
conditional headroom at the canonical batch-16 low-distortion coordinate:

| Representation | Best exact block coder | Bytes |
|---|---|---:|
| common/differential signed | C0B LZMA | 400,035,534 |
| separate signed residual | C0B LZMA | 411,217,998 |
| separate Y0/Y1 XOR | C0B LZMA | 418,598,640 |
| interleaved Y0/Y1 XOR | C0B LZMA | 419,224,202 |
| Y1 semantic + Y0 given Y1 | C0B LZMA | 420,418,711 |
| pair-temporal signed | C0B LZMA | 432,349,494 |

The best direct basis is about 7,461 times the current conditional headroom.
This is not a near-miss that entropy-coder tuning can repair. It falsifies the
idea that the solved selected-preimage planes themselves are the payload.

The scope is narrower and more useful:

- it rejects these six lossless local/block coordinate systems;
- it does not reject the full-lattice teacher as a source of task-space signal;
- it does not prove that a learned quotient is mandatory;
- it points to a nonlocal analytic/generative factorization of the teacher,
  followed only by an irreducible trained quotient if exact scorer/value-per-byte
  evidence requires it.

## Structural synthesis

The recovered result clarifies the codec boundary:

```text
full-lattice selected preimage = teacher / oracle
not payload

teacher
  -> evaluator quotient
  -> G90 task-weighted costate projection
  -> G89 class-complete shared physical program
  -> G88 conditional pose trajectory
  -> exact archive bytes + complete n600 row
  -> G83 nonlinear whole-state argmin
```

The 400 MB result is therefore not a reason to retreat to marginal tuning. It
is evidence that the missing map is an induction/distillation operator from
the solved evaluator witness into a population-global physical grammar. G89
now supplies the required class-complete grammar and G90 supplies the scorer
cotangent projection that G51 explicitly lacked.

The next experiment should not train pixels or recompress the full residual.
It should fit the smallest shared G89/G88 program whose realized-through-R
effects explain the high-value G90 directions, materialize prefix-closed exact
archives, and let G83 arbitrate the coupled Seg/Pose/rate equation.

## Resumability regression found and fixed

The governed harvest reopen exposed a separate exact bug. Commit `b84b4c6d94`
changed `_implementation_sources()` from semantic labels to recursive
path-keyed identities, while `_build_input_binding()` still indexed the
removed key `v15_receiver`. A current-source resume crashed before custody
reopen with:

```text
KeyError: 'v15_receiver'
```

Commit `e6535801e6` now looks up the exact
`src/tac/optimization/direct_description_carrier_compose.py` path in the
recursive closure and fails with a typed custody error if absent. Eight focused
tests pass. The historical completed aggregate remains bound to its recorded
source hashes; it is not relabeled as a current-source rerun.

## Triality

DSL:

```text
TeacherWitness -> ProjectedPhysicalProgram(G89 semantic, G88 pose)
```

DAG:

```text
recover 50/50 G51 checkpoints
  -> verify aggregate/file/self hashes
  -> classify six exact bases
  -> feed G90 costates
  -> induce population-global G89/G88 state family
  -> exact archive/eval
  -> G83
```

Equations:

```text
Q* = argmin_Q 100*d_seg(Q) + sqrt(10*d_pose(Q))
                 + 25*archive_bytes(Q)/37_545_489

subject to Q being receiver-closed and derived from counted program bytes.
```

Machine-readable harvest:
`g51_full_n600_quotient_harvest_receipt_20260727.json`.
