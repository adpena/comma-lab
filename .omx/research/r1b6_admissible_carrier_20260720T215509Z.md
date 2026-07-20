# R1b6 admissible carrier — receiver-bound singleton verdict

`lane_id=r1b6_admissible_carrier` · `[macOS-CPU advisory]` ·
`score_claim=false` · `promotion_eligible=false` · pointer
`0.1910828242 [contest-CPU] UNMOVED`.

## Verdict

`MEASURED_PREFIX_RECEIVER_BOUND_SINGLETON_NONPOSITIVE_STOP_NO_N600`

Verdict scope: exact Fisher-ordered prefix n16, 512 requested Road–Lane-first
sites, 498 exact-feasible same-rounded-bin singleton preimages, absolute-write
R1K1 replay, sealed R1b4 deterministic double decode, and hard CPU-Torch batch
16 only.  This is not n600, not compact-binary-v2 parse-back, not production
rank-4 custody, not a contest score, and not a boundary/full-kernel family
negative.

The measured arm worsened the hard oracle.  Baseline -> candidate changed
`10,002 -> 10,009` SegNet flips, `d_pose 131.1549471 -> 131.1581207`, and
prefix nonrate `36.53327596 -> 36.53393663`.  Therefore realized recovery is
`-0.00066067449 S` and the formulation-scoped break-even is **0 bytes**.  No
n600 run is justified for this absolute-write/source-closest-sign formulation.

## Break-even remeasurement

| row | realized recovery | realization | break-even bytes | authority |
|---|---:|---:|---:|---|
| inherited R2b fixed-magnitude n600 | `+0.00123323166 S` | `9.4621217%` decisions | `1,852.091296 B` empirical (`1,852.091427 B` callable) | existing n600 formulation-scoped anchor |
| R1b6 R1B4 absolute replay n16 | `-0.00066067449 S` combined; `-0.00022252401 S` Seg | `-1.4056225%` Seg vs scheduled prefix recovery | `0 B` after nonnegative clamp | measured prefix; not canonical numeric replacement |

The canonical equation remains
`B = max(0, Delta S_realized) * 37,545,489 / 25`.  Registry row 765 appends a
`domain_refined` event for `realization_breakeven_bytes_v1`: the positive R2b
anchor cannot transfer to R1B4 absolute replay without a fresh hard oracle,
and an n16 row cannot be used as an n600 anchor.  The old numeric anchor is
preserved; its domain is narrowed, not overwritten.

## Receiver and byte decomposition

Both arms sealed and decoded twice byte-identically.  Candidate decode was
`35.535 s`, below the `1,800 s` gate; receiver search invocations were exactly
zero.  The candidate archive is `120,542 B` (SHA-256 `193c4cb5dee2200b8...`),
or `+26,198 B` versus the `94,344 B` control.

| section | baseline compressed | candidate compressed | delta |
|---|---:|---:|---:|
| manifest | `1,154 B` | `1,155 B` | `+1 B` |
| zero windowed-curvelet boundary | `395 B` | `395 B` | `0 B` |
| R1K1 replay | `179 B` | `23,069 B` | `+22,890 B` |
| xi0 float16 payload | `1,135 B` | `1,135 B` | `0 B` |
| whole archive vs receiver baseline | `97,651 B` | `120,542 B` | `+22,891 B` |

The replay carried `5,976` absolute writes for `498` sites (`60,008 B` raw),
with 14 exact-preimage-infeasible sites rejected rather than coerced.  The replay cost
is about `45.97 B/site` after ZIP compression, before any claim of useful
realization; measured realization was negative.

## Alphabet decision

`R1B6_COMPACT_BINARY_V2_PROJECTION_KILLED_CURRENT_FORMULATION`

Verdict scope: the existing 1,273-byte compact-binary-v2 projection and this
source-closest-sign absolute replay only; no compact cell-grammar or carrier
family negative.

The 1,273-byte number remains a projection with `parse_back=false`,
`receiver_bound=false`, and no actuating site grammar.  The receiver-bound
implementation available today costs `26,198 B` total carrier delta and is
hard-oracle negative on the measured prefix.  It cannot be promoted by
substituting the ideal Road–Lane cell-identity floor (`810.0446 B / 5,193
known sites`), because that floor excludes locations, transport, headers,
receiver, and realization.  Other-edge and nonedge strata were not reached;
the Road–Lane-first stop rule fired first.

## Compiler blockers after the stop

- `R1B2_RANK4_FIRST_ORDER_REALIZED_SECANT_CUSTODY_ABSENT`: unchanged at
  production scope.  This prefix gives a real endpoint effect, but not distinct
  typed Frechet and realized-secant tensors for all 16,319 moderate cells.
- `R1B6_FULL_KERNEL_ABSOLUTE_REPLAY_PREFIX_PROVEN_N600_MDL_REPLAY_ABSENT`:
  narrowed.  Search-free replay and decode time are measured, but n600 MDL
  selection, compact replay, and hard-oracle-positive admission are absent.
- `VJP_FULL_SIDECAR_REHASH_DEFERRED_UNTIL_PRODUCER_INPUTS_PRESENT`: unchanged
  and fail-closed; no cheaper producer input appeared.

The honest successor tool was run end-to-end through the receiver and hard
oracle.  `tools/compile_r1b2_mdl_xi0.py` was not sent fabricated P1/P2
manifests after the rate/realization stop fired.

## Gates and control comparison

- `d_seg <= 3.39e-4`: FAIL on prefix candidate (`0.0031817754`); prefix scope,
  not an n600 candidate verdict.
- fixed-C1 archive cap `216,222 B`: byte-only PASS (`120,542 B`), but the joint
  gate fails because distortion and realization fail.
- decode `<=1,800 s`: PASS (`35.535 s`, n16).
- `<=477.8 B/pair`: the full archive is `200.90 B/pair` only if divided by 600,
  but that division is inadmissible for an n16 receiver-smoke archive; no gate
  credit is taken.
- `94,344 B / 0.00351579 / 127.36588 / S 36.10276` remains the n600 control.
  The prefix row is not mixed into that n600 score.

## Durable evidence, cleanup, and failed-attempt custody

- Machine measurement:
  `.omx/research/r1b6_admissible_carrier_prefix_n16_20260720.json`, SHA-256
  `7bdcffeb838478de3f75dd5b3ad572383175d42b368b006a26e5351175fc1684`.
- SSD candidate:
  `/Volumes/VertigoDataTier/pact/evidence/r1b6_admissible_carrier_20260720/prefix_n16_512_v2/candidate_sealed.zip`.
- Success-only raw scratch was hash-certified in the receipt and deleted after
  receipt fsync.  The first failed attempt stopped on an infeasible singleton
  before its cleanup receipt existed; its approximately 186 MiB SSD directory
  `prefix_n16_512` is deliberately retained.  Exact blocker:
  `R1B6_FIRST_ATTEMPT_NO_FAILURE_CLEANUP_RECEIPT_KEEP_BYTES`.

## STORES CONSULTED

Delegated wrapped authority; full repository operating contracts; R1b5 memo
and receipt; G1/G3 measurement memo; canonical frontier and lane/subagent
state; R1b2 compiler and tests; R1b3 producer custody; R1b4 receiver, memo, and
tests; R2b exact singleton solver source at commit `98515407bd`; SHA-bound
38,077-row Fisher ordering; C2 control archive/decoder; n600 target raw; frozen
hard CPU-Torch scorer.

## MAIN landing review required

MAIN must independently inspect the exact numerator proof, infeasible-cell
skip, Fisher-order preservation, R1B4 seal/double-decode custody, hard-oracle
arithmetic, domain-refinement row, and scoped stop verdict.  Rerun focused
tests before landing; do not promote the n16 advisory row or the 1,273-byte
projection as n600/contest authority.
