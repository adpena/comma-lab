# G51 findings — conditional selected-preimage quotient profiler

Date: 2026-07-26  
Lane: `lane_g51_conditional_selected_preimage_quotient_profiler_20260726`  
Axis: `[encoder-only exact-byte quotient diagnostic]`  
Pointer delta: **none**; effective competitive target remains the dynamic
canonical pointer, currently 0.172. No archive was built or evaluated.

## Outcome

The production, resumable full-n600 profiler is implemented and its strict
zero-chunk preflight passed on the Vertigo SSD. The full 600-pair measurement
was deliberately not launched by G51.

The profiler compares the strict counted V15 semantic base against the exact C1
two-plane selected preimage in 50 immutable 12-pair stages at factor-2 scorer
coordinates. It races six exactly reversible layouts:

1. separate Y0/Y1 XOR;
2. interleaved Y0/Y1 XOR;
3. separate signed residuals;
4. common/differential signed residuals;
5. pair-temporal signed residuals with explicit chunk resets; and
6. scorer-asymmetric Y1 semantic base plus Y0|Y1 pose enhancement.

For every layout it records raw bytes, zlib-9 block bytes, the existing C0B raw
LZMA bytes, per-pair marginals, entropy, zero runs, channel/class conditionals,
and exact round-trip receipts. Stage 00, every Stage 10 chunk, and Stage 20 are
write-once/self-hashed and independently resumable. No scorer weights, target
planes, or teacher state are serialized into a candidate.

## Correct planning coordinate and premise falsification

G51 initially inherited the historical batch-32 MS1 arithmetic:

- d_seg `0.0001519690619574653`;
- d_pose `0.00010184327939026322`;
- 187,562 bytes below 0.172 and 154,522 below 0.15;
- 53,621 / 20,581 bytes of conditional headroom over V15's 133,941 bytes.

That is not the current primary coordinate. Proactive recall found the
pre-existing canonical batch-16 artifact
`c1_live_target_debt_n600_batch16.json` (file SHA-256
`0db8e47a994cad5367e5eb3028055e667bc4caf3f174026d13171be662e7fbd3`):

- d_seg `0.00015196058485243054`;
- d_pose `0.00010184347386600314`;
- 187,563 bytes strictly below 0.172 and 154,523 below 0.15;
- 53,622 / 20,582 bytes of conditional headroom over V15.

Its same-decoded-raw contest-CPU receipt corroborates rounded component
distances. G54 later independently replayed the same raw at batch 16 and found
d_seg `0.00015196057211142033`, d_pose `0.0001018434704747051`; its distortion
term differs from the canonical primary by only `-1.805437170927273e-9`.
G54 is therefore independent corroboration only. The typed binding records
`PREEXISTING_CANONICAL_BATCH16_PRIMARY_G54_INDEPENDENT_CORROBORATION_NO_NOVELTY`
so future consumers cannot rediscover or misattribute this premise.

The current 53,622-byte value is still conditional planning arithmetic, not an
archive budget or score claim. Only exact ZIP bytes and exact evaluator replay
can move the pointer.

## Forest-level verdict

The structural question is now executable but not yet measured: does the
selected-preimage debt have a cheap exact quotient in one of the six tested
coordinates?

- If a block-coded exact basis is below 53,622 bytes, the shortest route is to
  materialize that basis through the existing receiver seam, measure exact ZIP
  bytes, public-double-decode, and exact-evaluate the same archive.
- If none is below 53,622, the result falsifies only these six local/block
  layouts. It does **not** prove that a learned irreducible quotient is
  mandatory. The next race is nonlocal analytic/generative factorization
  versus a learned residual, trained only on debt left after the analytic
  predictor, temporal structure, and entropy contexts.
- Sub-0.15 is stricter: a direct quotient must fit 20,582 conditional bytes at
  the same distortion premise, or improve distortion enough to buy more rate
  through the joint score equation.

No basis winner exists until the governed full n600 profile completes.

## Functional-operator / HOPE-compatible surface

The profiler exposes exact scorer-asymmetric output groups:

- Y1 semantic delta, visible to SegNet and PoseNet;
- Y0|Y1 enhancement, visible only to PoseNet.

It records exact output-space energies and block-coded marginals. Its ambient
two-group Gram is diagnostic only: disjoint plane coordinates force the cross
term to zero and therefore erase the evaluator's nonlinear coupling.

A task-weighted quotient-atom Gram and low-rank merge, prune, or macro-segment
eviction ranking are
`BLOCKED_MISSING_SCORER_COSTATE_EFFECTS`. Required inputs are receiver/R/frozen
SegNet and PoseNet JVP/VJP effect vectors plus same-object ZIP marginals.
This block is honest and scoped; class labels and residual energies cannot
substitute for scorer-induced function distance.

The HOPE PH-1/BatchNorm closed forms are forbidden because V9/V15 uses the
source-incompatible FiLM-conditioned `tanh(sin)` trunk. Static parameter count
is never accepted as rate. Functional diagnostics generate proposals only;
whole-object archive bytes and frozen-scorer verification decide.

## Triality

- DSL: closed typed config and input binding; six exact representation IDs;
  batch-16 primary/corroboration/historical roles; immutable stage schemas;
  research/candidate/score fences.
- DAG: strict V15 parse -> bounded camera render -> exact factor-2 resize ->
  C1 plane chunk -> six exact transforms -> compressor/statistics race ->
  immutable aggregate -> receiver-closed archive builder -> public double
  decode -> exact evaluator. G51 lands through the aggregate and launches none.
- Equations: `r0=Y0_target-Y0_base`, `r1=Y1_target-Y1_base`;
  `c=floor((r0+r1)/2)`, `d=r1-r0`,
  `r0=c-floor(d/2)`, `r1=r0+d`; and
  `S=100*d_seg+sqrt(10*d_pose)+25*B/37,545,489`. No independent arbitrary
  Seg/Pose/rate threshold is introduced.

## Six downstream hooks

1. `SENSITIVITY_MAP`: typed incompatible until scorer JVP/VJP costates exist;
   class-conditioned residual statistics remain available without authority
   upgrade.
2. `PARETO_CONSTRAINT`: conditional exact block-byte basis rows only; no
   archive admission.
3. `BIT_ALLOCATOR`: per-pair exact block-coded marginals are ready as
   encoder-side diagnostics.
4. `CATHEDRAL_AUTOPILOT_DISPATCH`: Tier-A observability only; no dispatch.
5. `CONTINUAL_LEARNING_POSTERIOR`: blocked because the profiler and canonical
   batch-16 debt are non-promotable planning evidence.
6. `PROBE_DISAMBIGUATOR`: the aggregate emits the exact-basis winner/failure
   route and explicitly keeps learned necessity `NOT_PROVEN`.

## Readiness receipt and verification

Durable preflight:

`/Volumes/VertigoDataTier/pact/taskspace_conditional_quotient_profile_n600_batch16_primary_20260726/preflight_receipt.json`

- internal preflight receipt self-hash:
  `bbef588d6ab89b74db4274508f8fd3c4c2fe8537e3524f70dfd0322e18b48e17`;
- durable file SHA-256:
  `fedcbe40bf7e59b006cb1d14be1a2b6822377ccd21988b5c4c7fbc99f8ef141f`;
- Vertigo storage gate passed with 1 GiB reserved;
- every configured artifact and internal receipt hash reopened;
- current batch-16 headroom derived as 53,622 / 20,582;
- `pair_rendering_started=false`, `chunks_profiled=0`;
- `full_n600_launch_authorized_by_this_receipt=false`.

Verification:

```text
ruff: PASS
targeted implementation + reused-seam pytest: 19 passed
```

## Governed next command

The full profile remains an explicit main-agent/governor decision:

```bash
.venv/bin/python tools/profile_taskspace_conditional_quotient_n600.py \
  .omx/research/configs/taskspace_conditional_quotient_n600_20260726.json
```

It is SSD-first, 12-pair bounded, atomic, and resumes from every completed
chunk. The immediate blocker to the forest verdict is simply that this real
n600 job has not been launched. The task-weighted functional-ranking extension
has the separate, nonblocking scorer-costate blocker described above.

## STORES CONSULTED

- `CLAUDE.md` / `AGENTS.md`, `PROGRAM.md`, top-10 project memory, directives;
- `.omx/state/canonical_frontier_pointer.json`;
- lane registry/progress and current G48/G50/G54 coordination;
- canonical batch-16 C1 debt receipt and same-raw contest-CPU crosscheck;
- independent G54 batch-16 replay receipt;
- historical G48 V15/MS1 coordinate compatibility receipt;
- strict C1 prepare receipt/chunk custody;
- fresh G46 batch-16 teacher materialization receipt;
- C0B semantic quotient, V15 receiver, factor-2 lattice, and frozen evaluator
  seams.
