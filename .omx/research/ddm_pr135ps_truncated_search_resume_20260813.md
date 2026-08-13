# PR135PS truncated-search adjudication — source-converged, no candidate (2026-08-13)

## Result

**FOLDED at the exact-reference-form instance scope.** The retained PR135 source refutes the
charter's premise that the final F26 search stopped after eight passes while still accepting.
F26's eighth pass accepted **0 / 595 active rows** across 14,280 row/direction slots. Three `+1`
moves were domain-invalid because their active coefficients were already `+2047`, leaving
**0 / 14,277 valid proposals**. Its own solver then sets `converged = true` and breaks. The
configured budget was 12 passes, so pass 8 was not a cap.

No ninth-pass candidate was manufactured, no scorer or evaluator ran, no Modal job fired, and no
score or frontier moved. The sealed fire-order is a machine-readable refusal with no command or
run ID. Any search outside this exact signed-int12 singleton neighborhood is a declared mechanism
extension, not a resumed PR135 pass.

## Pinned source and custody

| Object | Bytes | SHA-256 / revision | Finding |
| --- | ---: | --- | --- |
| ExperimentBook | — | git `f229b26735dffc53fdf1ac9987ac7c303298d028`, clean | Retained upstream source tree |
| `README.md` | 26,060 | `b8a05bdb88ce11c369ce15eae089b2d7870f00756d2d2a7784be0a72d791ab0c` | F26 trajectory and zero-accept convergence statement |
| `solve_f26_iterative_joint_carrier.py` | 16,166 | `f69c242748d5289db237c5f7a1b0492901ec1e183edad35bbeef31d4015c3bee` | Exact acceptance, checkpoint, and stopping logic |
| PR135 archive | 186,724 | `12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004` | Final F26 bytes |
| live CP135 archive | 186,252 | `6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6` | Current composed base |

The scorer-free parse-back proves that CP135 retains PR135's exact canonical carrier: 22,307
canonical bytes, SHA-256 `709ea928c2d73c599a9cffa85d9ea4f4cedab2940594f4b8ca39e4c60fd3a1d4`.
The 14-byte F0E1 selector is also byte-identical, SHA-256
`67d43d9050b1005ef04ef8f0e5657d10bd3cbd3920a9874449c9876a676b9a17`. CP135's HPAC and token
streams differ, as expected from the later lossless composition; that does not change the int12
search state.

## Exact trajectory and stopping rule

The retained F26 source reports this accepted-row trajectory:

| Pass | Accepted rows / 595 | Source-retained per-pass pose delta | Source-retained byte delta |
| ---: | ---: | --- | --- |
| 1 | 412 | not present in the retained ExperimentBook tree | not present in the retained ExperimentBook tree |
| 2 | 187 | not present in the retained ExperimentBook tree | not present in the retained ExperimentBook tree |
| 3 | 72 | not present in the retained ExperimentBook tree | not present in the retained ExperimentBook tree |
| 4 | 39 | not present in the retained ExperimentBook tree | not present in the retained ExperimentBook tree |
| 5 | 15 | not present in the retained ExperimentBook tree | not present in the retained ExperimentBook tree |
| 6 | 9 | not present in the retained ExperimentBook tree | not present in the retained ExperimentBook tree |
| 7 | 2 | not present in the retained ExperimentBook tree | not present in the retained ExperimentBook tree |
| 8 | **0** | **0, derived from no accepted code change** | **0, derived from deterministic rebuild with unchanged codes** |

The solver evaluates `best_error < errors - 1e-15`, counts accepted rows, writes the pass archive
checkpoint, marks convergence when that count is zero, persists state, and breaks. This is the
algorithm's derived stopping rule. Extending an arbitrary pass cap is appropriate only while the
last completed pass still accepts; it cannot override an already-fired convergence condition.

The “still accepting at pass 8” language matches the earlier **F23** trajectory
`357, 186, 82, 29, 13, 8, 2, 1`, but that solver then ran pass 9 and accepted zero. Final PR135 is
F26, whose own pass 8 was already the zero-accept pass. The charter appears to have conflated the
two stages.

## OPTIMAL FORM

The reference form remains PR135's own F26 all-12 exact signed-int12 singleton descent. The audit
did not substitute a surrogate, change its Jacobian, change its objective, shrink its proposal set,
or infer convergence from a fixed cap. The exact implementation itself establishes convergence in
its tested one-step neighborhood. A radius-greater-than-one move, coupled move, new global start,
basis edit, FiLM edit, or changed objective is explicitly outside this verdict.

## Dual-axis admission and sealed fire order

Disposition: **FOLDED / REFUSED_NO_CANDIDATE**. There is no archive to score, so the fire command and
run ID are null and the cost is $0. The existing CP135 adapted runtime remains pinned at
`/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime`; no candidate-specific runtime was
needed.

The family law remains binding for any future mechanism extension: admission must measure the
SegNet field and PoseNet first-six vectors on the same decoded archive. Preferred worker:
js6b-extended. Fallback: re1t followed by a same-archive pose-vector leg. This arm does not own or
claim the scorer lane.

## What was measured and what was not

- **Measured `[source/custody, scorer-free]`:** exact file sizes and hashes; clean retained source
  revision; F26 accepted-row counts; exact solver contract; PR135-to-CP135 canonical carrier and
  selector identity; retained receipt hashes.
- **Derived from exact solver semantics:** pass-8 acceptance rate 0 / 595; proposal denominator
  14,277 valid proposals out of 14,280 slots; pass-8 pose and byte deltas are zero because no code changed and the archive builder is
  deterministic.
- **Not measured in this arm:** RGB decode, SegNet, PoseNet, `upstream/evaluate.py`, contest-CPU,
  contest-CUDA, a new archive size, a new score, or promotion eligibility.
- **Bounded absence:** per-pass F26 pose/byte records were not found in the retained ExperimentBook
  tree or the bounded PR135 intake search. They are not reconstructed or invented here.

## RECALL EVIDENCE

Sources searched, beyond the charter's named receipts:

- Content queries `PR135|F26|int12|compensation|coordinate descent|pose` across
  `.omx/research/`, `CANONICAL_RESEARCH_INDEX*`, the sub-0.15 DAG FEED blocks, and the task ledger;
  canonical equations were also listed and searched for trajectory/acceptance/coordinate/pose
  surfaces.
- Primary retained ExperimentBook README, F26/F23 solver sources, git history state, and bounded
  searches for F26 `state.json`, `report.json`, checkpoints, and accepted-row logs.
- CP135 receipt/final result, fd135 decomposition, pi135 intake facts, re1's dual-axis family law,
  and PO1's later T4-residual/local-J compensation result.

Findings beyond the seeds changed the plan materially:

1. F26 is source-converged, so the requested exact continuation is not runnable without violating
   the reference algorithm.
2. F23 explains the stale “pass 8 still accepting” fact, but its ninth pass already reached zero.
3. fd135 already queues the genuinely distinct current-base joint/global-start family under
   `MAIN/#995 successor`; creating another queue row here would duplicate ownership.
4. PO1 later closed the damped T4-residual/local-J iteration at instance scope after a realized
   8.257x pose regression. That negative does not prove larger exact neighborhoods dead, but it
   removes “repeat local-J feedback” as a substitute continuation.
5. re1 independently requires the dual-axis same-archive gate for any future candidate; this is
   preserved in the sealed refusal instead of firing a pose-only check.

No canonical-equation or DAG entry displaced the source solver's zero-accept stopping rule.

## Retained store and verification

Retained store: `/Volumes/VertigoDataTier/pact/ddm_pr135ps_20260813/retained/`

- `SOURCE_CONVERGENCE_RECEIPT_v2.json`: 3,176 bytes, SHA-256
  `82302064fc0f783a5ad7a2a7d37d742eaf66a94fcd4831bce6b1fc76b1319d0a`.
- `SEALED_DUAL_AXIS_FIRE_ORDER_REFUSAL_v2.json`: 877 bytes, SHA-256
  `f131f400096eb9a49d26d61453d487ddc3396f5369967b5713ef1d368f31de1b`.
- `MANIFEST_v2.json`: 924 bytes, SHA-256
  `f7841c4d667fcefd62b86820bbc86ad1c66a66d85613fb0d14d6d023d46a3aed`.

The original v1 receipt/refusal/manifest remain in the same store as superseded evidence; v2
corrects the proposal denominator by separating all 14,280 slots from 14,277 domain-valid moves.

The verifier is idempotent: it refuses to overwrite any retained checkpoint with different bytes.
It created no archive payload and deleted or moved nothing. The already-retained PR135 and CP135
archive paths and hashes are referenced in the manifest.

The live effective pointer remains **S = 0.16195513827824176 @ 186,252 B
`[contest-CUDA T4, n600, prior MAIN-adjudicated]`**. This arm did not remeasure it.

Own-vehicle frontier remains **S = 0.16959899569230852 @ 187,226 B
`[contest-CUDA T4, n600, prior MAIN-adjudicated lc2]`**.

## NEXT_IF_RESUMED

- **FOLDED** — owner: `MAIN/#995 successor`; consumer store:
  `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/`; fire trigger: a declared
  radius-greater-than-one, coupled-move, new-global-start, basis, or FiLM mechanism extension is
  source-pinned, deterministic, resumable, payload-retaining, receiver-closed, and the scorer lane
  is clear; action: consume the existing current-base joint-solve queue and admit only through the
  same-archive dual-axis gate.

## LIVE-HYPOTHESES

- Coupled int12 moves or a radius-greater-than-one neighborhood may escape the singleton local
  optimum because zero accepted singletons proves only one-step coordinate stationarity, not joint
  stationarity.
- New global carrier/FiLM starts may reach a better basin because fd135 found that the published
  book never searched those starts on the current base; this is plausible but is a mechanism
  extension, not a resumed ninth pass.

## DEAD-ENDS

- Exact F26 pass 9 on the retained CP135 carrier: closed at this instance because pass 8 already
  tested all 14,277 domain-valid singleton proposals across 14,280 slots and accepted zero.
- “PR135 stopped after eight passes while still accepting”: refuted for final F26; it conflates
  F23's eighth improving pass with F26's eighth zero-accept pass.
- A scorer/Modal fire from this arm: closed because there is no new candidate archive, and the
  required dual-axis gate must not be invoked on unchanged base bytes.
- Rebranding a changed Jacobian/objective or the already-negative PO1 local-J feedback loop as a
  PR135 resume: closed by the optimal-form boundary and the PO1 instance result.
