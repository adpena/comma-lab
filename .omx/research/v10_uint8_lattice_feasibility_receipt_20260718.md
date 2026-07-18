# V10 factor 2 uint8 lattice feasibility receipt — 2026-07-18

- Lane: `lane_v10_uint8_lattice_20260718`
- Task: #532 / #520-adjacent
- Role: SOLVE; `training_bytes=0`
- Axis: `[macOS-CPU advisory subset]`
- Verdict scope: six deterministic real n600-cache frame-1 instances,
  canonical uint8-to-`A`, frozen CPU Torch SegNet; no PoseNet, full n600,
  complete receiver, contest CPU/CUDA, score, rank, family-kill, or promotion
  claim
- Pointer: `0.1910828242 [contest-CPU Linux x86_64]` **UNMOVED**

## Executive disposition

**MEASURED — selected-instance affine and hard Seg winner-cell feasibility
passed.** All `3,538,944` RGB channel-blocks were `FEASIBLE_EXACT`, all six
decoded frames had exact integer-numerator equality through `A`, and frozen
SegNet produced zero candidate argmax mismatches. The valid
`clip(round(B(y)))` comparator produced 520 mismatches and maximum scorer-plane
error `63.824981689453125`; the exact lattice candidate recovered all 520 with
zero regressions and maximum float-parity error `8.526512829121202e-14`.

**MEASURED — not a rate win.** The aggregate U8LFS sidecar is `11,346,894`
bytes. It is an incremental measurement sidecar, not `archive.zip`, not a
receiver-complete sufficient statistic, and not a contest score. The result
establishes a reusable local factor-2 primitive, not V10 completeness.

**Disposition:** factor 2 may move from `MISSING` to
`HAVE (advisory local primitive)` with strict certificate `PARTIAL`. Full n600,
Pose/both-frame interaction, receiver/archive rate closure, contest-axis replay,
and independent MAIN adoption remain literal blockers.

## Algorithm form adjudication

| form | honest disposition | evidence/reason |
|---|---|---|
| full frozen-SegNet MILP | rejected for this primitive | the camera-to-feature map is nonlinear; an affine-head MILP would certify only a surrogate subproblem |
| randomized or adjacent-corner repair | heuristic only | it can propose a valid uint8 point but cannot exhaust the bounded preimage or prove a negative |
| lattice-Dykstra | initializer/proposal only | the discrete lattice plus nonlinear CNN violates the convex convergence assumptions needed for a global certificate |
| exact rational gcd-pruned bounded DFS | selected | canonical factor-2 supports are disjoint, reducing each cell/channel to one finite four-variable bounded Diophantine equation |

**DERIVED.** Exact half-pixel tap numerators are formed from geometry, not
rationalized floats. Suffix range bounds and suffix gcd congruences prune the
finite DFS. `INFEASIBLE_EXHAUSTIVE` is scoped to one affine block;
`NOT_FOUND_BUDGET` is unknown. Proof status is orthogonal to returned-candidate
provenance. The hard Seg oracle is separately evaluated only after durable
uint8 parse-back.

## Exact command and deterministic environment

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
PYTHONPATH=src:tools:experiments:. .venv/bin/python \
tools/measure_uint8_lattice_feasibility.py \
  --sample-pairs 6 \
  --output .omx/research/v10_uint8_lattice_feasibility_receipt_20260718.json \
  --sidecar /Volumes/VertigoDataTier/pact/evidence/v10_uint8_lattice_20260718/candidate_n6_landing.u8lfs \
  --state /Volumes/VertigoDataTier/pact/evidence/v10_uint8_lattice_20260718/candidate_n6_landing.state.json \
  --stage-dir /Volumes/VertigoDataTier/pact/evidence/v10_uint8_lattice_20260718/candidate_n6_landing.stages \
  --cpu-threads 1 --max-nodes-per-block 4096
```

Environment: Python `3.13.12`, NumPy `1.26.4`, Torch `2.12.1`, macOS arm64,
CPU device, one Torch/OMP/MKL/Accelerate thread, seed `20260718`. Runtime was
`83.11827645800076 s`. Pair selection was recomputed without candidate-outcome
peeking: six equal temporal strata, alternating within-stratum fragility
quantiles, deterministic pair-index ties. Pair IDs were
`[90, 175, 277, 381, 424, 573]`.

## Custody

| object | bytes | SHA-256 |
|---|---:|---|
| machine receipt JSON |  | `665ce8ecd789a863eb85fa181f11746f292626b9283f82be9d90cf10b7905779` |
| repaired solver module |  | `5039902d8de519f74065dfcc39812d2d7f2336602c537009374e122ebca79e2b` |
| focused tests |  | `c4a532f7ba5c6baa2e375c88fcfbe30292adab3cb8c14374e9790d8e87e19969` |
| measurement tool |  | `51103ef9a97f52da0ec2f7bf370f55ee54ae1746238e7cd69d6c92d8ed40327a` |
| `gt_n600.npz` |  | `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6` |
| frozen `segnet.safetensors` |  | `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6` |
| pinned upstream `modules.py` |  | `065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa` |
| pinned upstream `frame_utils.py` |  | `d689aca7d263997cb2fb980d6098d503f955e56e8642cd0a04cc437f0ffdab90` |
| aggregate U8LFS sidecar | 11,346,894 | `e54162d4689a420c5dc5bca63069f0b6ddd6839c10af57e6d0873c8eeb4d9c31` |
| resumable state |  | `eed75a62a169ddc8e6324b4a61eb7f1f1929deae4729f0506e9492bf7c4ad1a8` |

Every pair-stage payload is preserved on the SSD. The aggregate sidecar parsed
back to all six exact decoded-frame and embedded-payload hashes. The executed
`modules.py` and `frame_utils.py` paths exactly matched the pinned paths whose
bytes are hashed above. The canonical pointer surfaces and the sacred result
tree metadata were byte/hash unchanged before versus after. Independent
sidecar and stage-directory storage checks selected the same
`/Volumes/VertigoDataTier/pact` filesystem with `824,820,822,016` free bytes;
both passed their fail-closed `104,519,392`-byte requirement.

The first successful n6 receipt was bound to the pre-review module and therefore
was not reused as final evidence. Its exact bytes were losslessly moved to
`/Volumes/VertigoDataTier/pact/evidence/v10_uint8_lattice_20260718/prereview_receipt_solver_cdd5a6cc.json`;
the machine-readable move manifest has SHA-256
`5de6f8add4b6b6f77a63cd07f7653cc568f84509ea3354a8ab379fe733bce5fc`.

A second hash-bound receipt reproduced all metrics after the first repair set,
but a later full-landing review found malformed public exact inputs could still
obtain false certificates. Those receipt bytes are likewise preserved at
`/Volumes/VertigoDataTier/pact/evidence/v10_uint8_lattice_20260718/preseal_receipt_solver_4584f124.json`;
its move manifest SHA-256 is
`3e5ca99c9f21dd8d6b245f309ed6c3a179bda149d393965475fd8bcee445de51`.
Neither of those first two superseded receipts is cited as final landing
evidence.

A third receipt reproduced the final scientific metrics on solver SHA
`5897df7b...`, then PASS 2 exposed adjacent public-API numeric-custody and
overflow gaps. Its exact bytes are preserved at
`/Volumes/VertigoDataTier/pact/evidence/v10_uint8_lattice_20260718/preseal_receipt_solver_5897df7b.json`
(SHA-256 `afdbdd761c033e625869adbbd9ec069c0495b8e1c15ea92d89e22e45f1518f5d`).
The certified move manifest SHA-256 is
`03b097f6655d363043abbaf96807d5829b08d76dea4319a65e8a83b3448913c5`.
It too is explicitly superseded and is not cited as final landing evidence.

A fourth fresh receipt bound the hardened solver but preceded fail-closed
resume-row and executed-module-path validation in the measurement wrapper. Its
scientific rows were independently re-scored, but it is not landing evidence.
The bytes are preserved at
`/Volumes/VertigoDataTier/pact/evidence/v10_uint8_lattice_20260718/preseal_receipt_tool_b32ebb8c.json`
(SHA-256 `88e9b9b31f347ddcb7f5090dec4fa0941cd9b3e557f2dd87781ce20cbf2534a7`);
its certified move manifest SHA-256 is
`f8b92478705607f888da2b7ae9030fb41043fb1804c615a85b20afd5ac70c22d`.
None of the four superseded receipts is cited as final landing evidence.

## Aggregate arm comparison

| arm | legal/shippable here? | max `|A(frame)-y|` | mean absolute residual, pair-mean | mismatches / 1,179,648 | `d_seg` |
|---|---|---:|---:|---:|---:|
| real minimum-norm `B(y)` | no; float diagnostic | `8.5265e-14` | `1.4677e-15` | 1 | `8.4771e-7` |
| `clip(round(B(y)))` | uint8 comparator | `63.824981689453125` | `0.1954703522` | 520 | `0.0004408094618` |
| exact uint8 lattice | yes as local sidecar frame | `8.5265e-14` | `8.6109e-16` | 0 | `0` |

The exact solver visited `24,355,043` DFS nodes. It returned
`3,538,944/3,538,944` exact proof blocks and exact-provenance candidates,
`0` heuristic blocks, `0` budget exits, and `0` exhaustively infeasible blocks.
All six frame aggregates were `FEASIBLE_EXACT`; decoded exact numerator residual
was identically zero. The real minimum-norm lift had 5,758 out-of-gamut camera
coordinates before the bounded solve.

### Per-class hard Seg comparison

| target class | target pixels | clip-round mismatches | clip-round `d_seg` | exact-lattice mismatches / `d_seg` |
|---:|---:|---:|---:|---:|
| 0 | 270,157 | 233 | `0.000862461458` | 0 / `0` |
| 1 | 6,625 | 37 | `0.005584905660` | 0 / `0` |
| 2 | 587,273 | 168 | `0.000286067979` | 0 / `0` |
| 3 | 15,273 | 76 | `0.004976101617` | 0 / `0` |
| 4 | 300,320 | 6 | `0.0000199786894` | 0 / `0` |

### Pair-stage receipts

| pair | clip flips | exact flips | out-of-gamut real coords | DFS nodes | stage bytes | stage SHA-256 |
|---:|---:|---:|---:|---:|---:|---|
| 90 | 80 | 0 | 1,048 | 4,160,666 | 1,949,237 | `8e3aa596e0d00b119674a8d16216e89013051e0a6dbc8dc4b92e204ee956bd4a` |
| 175 | 110 | 0 | 611 | 3,985,203 | 1,866,478 | `577f3c1d76793999a882f26fd1bbce9ff98cbad68dc0d27b18d090343b1355e8` |
| 277 | 35 | 0 | 312 | 4,019,021 | 1,865,835 | `071d4480a692566dd442cf9d40ab15246f25ee0562574d7015baf356904f4d21` |
| 381 | 86 | 0 | 589 | 4,066,955 | 1,867,479 | `d33d270672814a4512f8e8cda0b1e1e1fc7b3d3c8a12bc5e78fbb8ebe22bc5c7` |
| 424 | 114 | 0 | 1,682 | 4,063,627 | 1,892,668 | `2a12b93d5427ca55daa2a6dd90bfa094dc215b1d62436b73fa393d95c9e75739` |
| 573 | 95 | 0 | 1,516 | 4,059,571 | 1,904,888 | `cf92bb2617cbaa1bb97a8dd837269a53f465fd533118e12874be759c4805d92a` |

Each exact decoded candidate differed from the hidden source frame. The solver
boundary received only `y`, exact numerator custody, and an assertion-equal
`B(y)`; source camera bytes were deleted before the solve call.

## Named confounds and controls

Named confound: soft margin movement masquerading as a real argmax-cell flip
after uint8/resize/parse-back.

- **Positive, MEASURED:** each source uint8 frame was independently U8LF
  serialized/parsed and re-scored in the same frozen-SegNet process; bytes and
  argmax matched for all pairs. Cache-versus-same-run source labels disagreed at
  zero pixels.
- **Candidate, MEASURED:** candidate admission used decoded uint8 hard argmax;
  all 520 clip-failed cells held and no clip-held cell regressed.
- **Negative, MEASURED plumbing control:** an always-false hard oracle with no
  proposals returned `STALLED_UNKNOWN`, not feasibility or infeasibility.
- **Behavioral test control:** a decoded real-size uint8 frame went through the
  canonical `A` parity check and actual frozen CPU SegNet; the test ran rather
  than skipping.
- **Source-copy control:** independent review demonstrated the original public
  reference API could return a hidden source. The repaired API now recomputes
  `B(y)`, refuses any unequal reference, and ignores the asserted reference for
  construction. The n6 receipt was rerun against the repaired module hash.
- **Exact-certificate input control:** later reviews reproduced false or stale
  certificates from coercible public inputs, caller-widened tolerance, mutable
  result arrays, overflowed arithmetic, and changing hard-oracle obligation
  shapes. The final API refuses boolean/complex/string coercion, non-integral
  exact fields, non-finite values, negative or widened tolerance, more than the
  four factor-2 taps, forged supports, out-of-lattice integer inputs,
  non-finite preimages/debt, oversized serializer payloads, and changing oracle
  shapes. Certificate-bound arrays are bytes-backed and cannot be reopened for
  writes. All 99 focused tests pass, and the n6 receipt was rerun against the
  final module hash.
- **Resume/source-custody control:** a final wrapper review showed that stored
  checkpoint metrics and an earlier `sys.path` entry could outrun byte/path
  custody. Resume now discards stored scientific rows and re-derives every
  completed pair through the deterministic solver, exact `A`, preserved stage,
  and frozen SegNet; it also binds executed scorer paths and aggregate payload
  hashes. A final-tool pair-90 fresh/resume smoke reproduced 80 clip failures,
  `4,160,666` nodes, and identical scientific/stage rows while reporting one
  re-derived pair and `stored_pair_metric_fields_reused=false` (resume receipt
  SHA-256 `141ecf6371733405529658e282d4bc60b6a2cacb6c0d1169d1cf24389972509e`).

## NO-FAKE labels and scoped negatives

- **MEASURED:** exact-search counters, decoded numerator residual, hard Seg
  totals/per-class rows, clip-failure recovery, payload bytes/hashes, runtime,
  parse-back, and unchanged custody surfaces.
- **DERIVED:** disjoint factorization and bounded-Diophantine certificate logic.
- **INFERRED:** success on this subset makes factor-2 feasibility plausible on
  more n600 targets; it does not establish that extrapolation.
- **SCOPED NEGATIVE:** the raw sidecar is not rate-competitive as measured. This
  does not kill compressed lattice grammars, procedural receivers, alternative
  target ownership, or the V10 family.

## Triality and system intelligence

- DSL: typed solver API plus bounded measurement CLI; no trainer argv or launch
  authority.
- DAG: `.omx/research/v10_uint8_lattice_DAG_FEED_20260718.md`.
- Equation: candidate
  `bounded_uint8_resize_preimage_cell_feasibility_v1` is recorded only in a
  temporary JSONL; it is not in the canonical registry.
- System consumer: factor-2 completeness records the primitive as advisory and
  preserves every remaining adoption blocker. No posterior, score, dispatch,
  or pointer ledger was mutated.

## Remaining blockers

1. Full n600 frozen-SegNet replay through the governed launcher.
2. PoseNet and frame-0/frame-1 interaction through the shared `A`.
3. Counted receiver/archive/inflate custody and a real rate benefit.
4. Exact contest-CPU and contest-CUDA replay on identical archive bytes.
5. Independent MAIN review/adoption remains mandatory; any recorded
   branch-local three-clean-pass seal does not substitute for it.

## Round-1 self-review

1. Rejected the incomplete adjacent-corner certificate before implementation.
2. Preserved proof status separately from fallback candidate provenance.
3. Reran measurement after independent review found a source-copy-capable API;
   the old receipt remains recoverable but is not landing evidence.
4. Kept the 11.35 MB sidecar out of archive/score language.
5. Kept the one non-uint8 float-arm flip diagnostic separate from shippable
   decoded uint8 authority.
6. Scoped the result to Seg/frame1/subset/macOS CPU and left the family open.
7. Mutated neither the sacred run tree nor the frontier pointer.
8. Rejected malformed, overflowed, mutable, or shape-changing public inputs
   before hashing the final receipt; no coercion- or stale-array-derived
   certificate is admitted.

## Stores consulted

`CLAUDE.md`; `AGENTS.md`; v7.5/v8 operating specs; V10 source of truth; exact
factorization/completeness/constructive-solve memos; operating manual;
`PROGRAM.md`; vehicle OS; fresh-eyes contract; latest sister
findings/design/council surfaces; `reports/latest.md`; canonical frontier,
lane, subagent, equation, probe, and dispatch state; frozen cache/upstream
scorer; live per-arm and broadcast inboxes.

MAIN must independently reopen the exact code/artifact hashes, source-copy
barrier, decoded scorer path, rate language, factor-2 disposition, and remaining
blockers before landing.
