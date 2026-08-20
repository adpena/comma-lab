QS2 landed in commit `d77fb69efc`.

The exact QS1 compensation object now costs **+34 B / 5.67 B per pair**, down from +77 B / 12.83 B per pair. Its measured density is **0.941 flips/B**, clearing the 0.785 breakeven law.

Key artifacts:

- [Decision memo](/Users/adpena/Projects/pact/.omx/research/ddm_qs2_compensation_rate_rung_20260813.md)
- [Candidate archive](/Volumes/VertigoDataTier/pact/ddm_qs2_20260813/candidate/archive.zip): 186,286 B, SHA `0bb74f1d…`
- [Validated R2 fire order](/Volumes/VertigoDataTier/pact/ddm_qs2_20260813/SEALED_FIRE_ORDER.json), request SHA `aedcd98e…`
- Retained store: `/Volumes/VertigoDataTier/pact/ddm_qs2_20260813/`

The candidate exactly reproduces the QS1 semantic, token, and signed-int12 code lattice. Reusing QS1’s matched T4 components gives a pre-admission projection of **ΔS = −4.3749179e-6**, but this is not a new measured score. Modal was not fired and the pointer did not move.

The required per-pair postmortem and waterfill remain blocked because the retained T4 argmax field is only on the unreachable Modal volume. I did not rederive it with SegNet. The exact recovery command and expected SHA are preserved in `PER_PAIR_POSTMORTEM_BLOCKER.json`.

Verification: 9 focused tests passed, Ruff and compilation passed, 256 randomized codec round-trips passed, all 48 coder candidates and six payload classes per candidate passed custody checks, deterministic resume reproduced the key hashes, and two review-tracker passes were recorded. Developer preflight remained red on eight pre-existing repository-wide gates; none of their returned violations referenced QS2.

Effective frontier remains **S=0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]**.  
Own-vehicle frontier remains **LC2 S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]**.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN sole scorer-lane router; consumer store: `/Volumes/VertigoDataTier/pact/ddm_qs2_20260813`; fire trigger: no active n600 exact-eval/Modal lane and every R2 sealed-input SHA verifies, then execute `exact_command_argv` from the fire order and let the dispatcher self-claim.
- **QUEUED-AT-HARVEST** — owner: MAIN; consumer store: `/Volumes/VertigoDataTier/pact/ddm_qs2_20260813`; fire trigger: the R2 argmax field and matched base/GT fields are locally available, then compute the retained-field per-pair decomposition and waterfill without rerunning SegNet.

## LIVE-HYPOTHESES

- R2 will preserve QS1’s −32 net flips and small Pose leakage because the semantic, token, and code lattices are identical; only full CUDA receiver execution remains untested.
- Dead-zone step 2 may dominate: it costs only +9 B and improved local Pose, but its altered lattice needs matched T4 measurement.
- Per-pair waterfilling or quantum-floor engineering may improve the 16.9% realization efficiency because 189 changed pixels collapse to 32 net flips across heterogeneous supports.

## DEAD-ENDS

- The original +77 B QS1 closure is closed as an instance: 0.416 flips/B misses breakeven.
- Dead-zone steps 3 and 4 are closed on the local Pose instance because their Pose penalties erase their byte savings.
- Local archive equality or local Pose ordering cannot substitute for matched T4 semantic preservation.
- Recomputing the missing argmax field is closed; recover or harvest the retained field.
- R1 is dead because its request violates the unchanged worker’s Pose-placeholder contract. Only R2 is live.