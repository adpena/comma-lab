# PZ3 pose receiver realization

## Conclusion

The 2,860-byte PZ2 target packet does not replace PR130's pose carrier through an exact-residual
receiver. The smallest real PZ3R archive is **194,120 B**, which is **3,068 B larger** than the
191,052 B PR130 base. The public receiver reconstructs the frozen basis and coefficients exactly,
and its 120 stratified slave frames are byte-identical to the measured PR130 control, so the loss
terms do not improve. The pre-registered verdict is **REALIZATION-LIMITED** for this formulation.

No contest pointer moved. No new scorer or exact-evaluation job ran.

## Measured result

| object | bytes | SHA-256 | axis / authority |
|---|---:|---|---|
| PR130 base archive | 191,052 | `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd` | pinned contest-CUDA,DALI,n600 base |
| selected `target_q12` carrier | 26,196 | `7f32aeae0f50fea2d3c8cd9b2a781ded94afd4c30585b1c740492b3a55d1e3ac` | `[macOS-CPU byte-closure; scorer-free]` |
| selected `target_q12` archive | **194,120** | `8213e0a580a557042b1a678903bc16504cbaaa1c781ed70e21d9c932b0cd3bdd` | `[macOS-CPU byte-closure; scorer-free]` |
| repeated selected archive | 194,120 | `8213e0a580a557042b1a678903bc16504cbaaa1c781ed70e21d9c932b0cd3bdd` | byte-identical determinism repeat |
| CPR1 n120 slave frames | 366,241,088 | `062bedca79e4f463e1b92c863a51bda2590f6033d1c6a2384ae310296cdf9948` | `[macOS-CPU advisory]`, retained |
| PZ3R n120 slave frames | 366,241,088 | `062bedca79e4f463e1b92c863a51bda2590f6033d1c6a2384ae310296cdf9948` | `[macOS-CPU advisory]`, retained |

The selected carrier contains 56 B of framing, the exact 13,101 B CPR1 basis component, the exact
2,860 B PZ2 packet, 412 B of counted fixed-point predictor state, and a 9,767 B exact coefficient
residual. PR130's original coefficient component was 9,953 B. Conditioning saves only 186 B there,
then adds 3,328 B of target, predictor, and framing state, making the carrier 3,142 B larger.

Twelve byte-closed predictor cells were materialized: target-linear, target-quadratic,
target-plus-previous-coefficients, and target-quadratic-plus-previous-coefficients, each at Q12,
Q16, and Q20. Their full archives span 194,120–195,220 B. Target-linear Q12 is the minimum.

## Real receiver proof

PZ3R is a public PR130 receiver branch. It decodes the PZ2 streams, runs counted integer
fixed-point prediction, decodes a counted Rice residual, verifies the reconstructed coefficient
hash, and then hands the reconstructed arrays to PR130's actual frame renderer. It does not load
PoseNet, SegNet, ground truth, or an uncounted video-derived table.

The selected target packet is causal: flipping target code `[0,0]` changes the coefficient
prediction, and the unchanged residual fails the coefficient-integrity hash. The packet is not a
decorative archive member.

The parsed PZ3R basis and coefficient tensors are byte-identical to CPR1. On the pinned seeded
stratified selection (`seed=20260809`, n=120, receipt SHA-256
`2e2778fd65e69c2af3ddcd1bff1bed3db3737a54743df89393e1e8f673a90f99`), all retained slave-frame
bytes are identical. Therefore the existing measured PK2 control transfers by same-output identity:

- `d_pose = 0.000020148457994650926`
- `d_seg = 0.00028962030773982406`
- `S = 0.17241309716202663` at 194,120 B
- axis: `[macOS-CPU advisory; exact-output identity to measured PK2 n120 control]`
- `score_claim=false`; no new scorer run

Using only the pinned contest-CUDA,DALI,n600 base components and the measured rate action gives
`S = 0.17418415276007526`, a **+0.0020428552681788226** regression. This is
`[DERIVED same-output rate action over contest-CUDA,DALI,n600 base components; no new exact eval]`,
not a new contest score. It exceeds the charter's 0.16110432236983460 falsifier threshold.

## Verdict scope

**FORMULATION negative:** exact-residual realization of the PZ2 target packet through PR130's frozen
carrier basis is REALIZATION-LIMITED. This closes the idea that the 2,860 B targets can simply
replace the current carrier while an exact residual preserves its frames. It does not kill a
jointly trained target-conditioned receiver that changes the rendered frames and removes most of
the incumbent coefficient description.

The prior PZ2 scorer queue row is `FOLDED`: a new scorer run on the identical output would repeat
the existing PK2 control, and the measured rate action is already worse. The only successor is
queued in `PZ3_NEXT_ACTIONS.jsonl` with a fire order.

## RECALL EVIDENCE

Sources searched:

- `tools/corpus_query.py` across research, equations, memory, DAG, council, tasks, and docs for
  `PR130 pose targets six PoseNet outputs target-conditioned receiver preimage carrier basis coefficients realization`
- `tools/corpus_query.py` across the same stores for
  `stored target pose packet decoder real frames PoseNet preimage direct precision receiver`
- `tools/list_canonical_equations.py --json`, filtered for pose and receiver surfaces
- content searches across `.omx/research`, canonical research indexes, `sub015_DAG_*`, design
  specs, task ledgers, `experiments`, and `src`

Beyond the charter seeds, recall found:

- G95's population pose-preimage chart: shared static bases were expensive and its pair-0 rank
  ladder had large pose error. This ruled out treating a new static basis as a free bridge.
- PI2's per-pair Jacobian observation: target preimages are strongly pair-dependent. Its proposal
  depended on receiver-time scorer access, which is forbidden by the current receiver boundary.
- P1's shared low-rank frame-0 actuator and M1's pose-from-embedding MLP instance both lost on
  pose/rate. These closed a repeat of those fixed shared formulations.
- PK2 had already measured 49 PR130 carrier representations. Its lossless coefficient predictors
  were larger than CPR1, but none conditioned on the PZ2 target codes. That changed this arm to an
  exact conditional-entropy test before any new nonlinear receiver training.
- SC1's 2,039 B terminal pose field belongs to a different vehicle and was not transferred as a
  PR130 number.

## Borrowed-substrate accounting and boundaries

Borrowed off-the-shelf PR130 substrate: semantic renderer, frozen carrier basis, coefficient scales,
HPAC, token stream, CPR1 entropy primitives, and public frame renderer. PZ3-original work: PZ2
packet decoder, counted fixed-point target-conditioned predictor, exact residual packet, PZ3R
public receiver dispatch, and byte-closed receiver proof.

- Upstream and the intake clone remained read-only.
- No Modal, paid dispatch, MPS authority, n600 scorer, contest-CPU, or contest-CUDA run occurred.
- All 67 materialized payload entries, including every candidate and repeat plus both rendered
  n120 tensors, are retained under `/Volumes/VertigoDataTier/pact/ddm_pz3_20260810/retained/` with
  bytes and SHA-256 in `PZ3_RESULT.json`.
- The run is resumable from its canonical checkpoint and preserves preflight, materialization,
  verification, and finalization stage checkpoints.

Own-vehicle frontier: **S = 0.17214129749189644 @ 191,052 B**
`[contest-CUDA,DALI,n600 inherited PR130 base; PZ3 did not move it]`.

## LIVE-HYPOTHESES

- A jointly trained compact target-conditioned receiver may make the 2,860 B packet useful because
  it can change the rendered frames and delete the 9,767 B exact residual instead of reproducing
  incumbent coefficients.

## DEAD-ENDS

- Exact target-conditioned residual coding on the frozen PR130 carrier: all twelve cells were
  larger than CPR1; the best archive regressed by 3,068 B.
- A new scorer run for the selected exact-output candidate: folded because frames are byte-identical
  to the already measured control and the rate action is worse.
- Static shared pose-preimage bases and the existing pose-from-embedding MLP instance: prior
  vehicle-specific measurements already closed those formulations on pose/rate.
