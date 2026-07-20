# Prerequisite surfaces flush — receiver-closed local build

**UTC:** 2026-07-20T17:10:50Z

**Lane:** `lane_prereq_surfaces_flush_20260720`

**Implementation commit:** `2000338eef9c36054944ea60297aa16ff1e2de7e`

**Authority:** local CPU BUILD/readiness only; `research_only=true`; no launch, score,
rank, or promotion authority

**Pointer:** `0.1910828242 [contest-CPU Linux x86_64]` **UNMOVED**

## Outcome

Three prerequisite surfaces are built and locally ready. The fourth remains blocked
with a byte-verified custody receipt; no missing full-n600 derivative field was
invented.

| surface | disposition | exact local evidence | consumer |
|---|---|---|---|
| matched continuous pre-step -> bounded uint8 -> parsed hard oracle | **BUILT / READY** | shared exact solver; 12/12 canary blocks `FEASIBLE_EXACT`; 106 parsed bytes; one fresh parsed-byte oracle call; `HARD_ACCEPT=true` | three preregistered regmax probes |
| frozen rank-4 valid-cell prototype bank | **BUILT / READY** | real frozen head SHA `68956e...91b6`; rank 4; fp32 reconstruction max error `2.9802322387695312e-8` <= sealed floor `5.960464477539063e-8`; cells 0..4 | regmax probe 3; r1b opportunistically |
| Aurenhammer/tropical/zero-sum same-coder comparator | **BUILT / READY** | identical PDW2 serializer + Brotli q11; exact identity on five strict prototype witnesses; coded bytes 134 / 137 / 131 | regmax probe 2 |
| real n600 M1 positive-anisotropic band | **BLOCKED / NO ARTIFACT EMITTED** | all 24 retained VJP sidecars rehashed, 3,641,507,444 bytes total; 24/600 logical pairs covered; current real n600 selector has 17,926 baseline flips and 16,751 evaluated decisions but no hashed 38,077-candidate measured EV ordering | M1 relaunch gate; r1b opportunistically |

Manifest:
`.omx/research/prereq_surfaces_flush_20260720/manifest.json`, SHA-256
`4a8330e79353cfd45dd3c7580c55013a878b6e72fb3e242223c82a0dc03f766f`.

## Surface 1 — matched preimage adapter

`matched_continuous_to_uint8_hard_accept` accepts a camera-space or scorer-plane
continuous proposal from sparsemax, entropy/Cole-Hopf softmax, or Hopfield
memory-prox. The proposal is clipped/lifted and supplied only as
`preferred_preimage` ordering to the existing
`DisjointResizeOperator.solve_uint8`; target numerators, bounded Diophantine search,
proof status, U8LF serialization/parse-back, and terminal oracle authority remain
unchanged.

The structural canary is **MEASURED local readiness**, not a frozen-scorer probe
verdict:

- exact target-numerator SHA:
  `9049b3a1d1c6957f2f431388a8704b9edfca0097d2fd9e82b5dda17db36aac2e`;
- aggregate status `FEASIBLE_EXACT`, 12 exact blocks, zero claimed score;
- parsed payload 106 bytes, SHA
  `2272021451979a2eb2da05e313bc06e77d5791df7963035ed8517749eac50f0d`;
- the oracle is invoked exactly once on parsed uint8 bytes; its satisfied/margin
  hashes are recorded;
- the separate real frozen-SegNet n6 anchor is SHA-pinned at
  `665ce8...5779`: 6/6 exact decoded numerator frames and `d_seg=0.0` on pairs
  `[90,175,277,381,424,573]`. It is explicitly `ANCHOR_ONLY`, not a rerun of a
  regmax continuous proposal.

Each future regmax A/B must therefore supply its own fresh frozen-SegNet oracle to
the adapter. A soft score or the historical anchor cannot admit a member.

## Surface 2 — frozen rank-4 prototype bank

The builder reads the real SHA-pinned `segmentation_head.0` weights, removes the
shared affine gauge, derives the canonical row-difference basis, and enumerates the
active sets of the strict-cell minimum-L2 problem. It emits immutable rank-4 arrays,
one margin-1 witness per cell, and a PDW2 reference-class gauge packet.

- quotient rank: 4 (**DERIVED from real frozen weights**);
- fp32 reconstruction max error: `2.9802322387695312e-8` (**MEASURED local**), one
  half of the sealed fp32 floor;
- prototype labels: `[0,1,2,3,4]`; margins:
  `[0.9999999403953552,0.9999999403953552,1.0,1.0,1.0]`;
- affine-weight SHA:
  `44a33ca5b57a452affa98e5686c6cccfab72b7309c48d7911bc14bd4e4714675`;
- prototype SHA:
  `5ce0458949acb1cde21022aef7bf642b4491ddb03d1ab66838866b20cb7b162f`;
- PDW2 packet: 142 bytes, SHA
  `756d5a70893ce23d6b1929807e5a0efadf50b2ed68527ed2c4029eadd98b6467`.

## Surface 3 — generic same-coder comparator

`serialize_affine_cell_candidate_same_coder` is the generic callable boundary:
any finite five-cell affine representative plus caller-supplied strict prototype
witnesses passes through the same PDW2 reference-gauge serializer and Brotli q11.
The three requested representatives are generated as distinct shared-affine gauges:

| representative | operational derivation | packet bytes | Brotli-q11 bytes | exact prototype identity |
|---|---|---:|---:|:---:|
| Aurenhammer min-generator LP | coordinatewise midpoint solving the declared shared-affine L-infinity min-generator LP | 142 | 134 | yes |
| tropical residuation principal | coordinatewise max-plus principal normalization | 142 | 137 | yes |
| zero-sum/min-norm | shared-affine L2 projection onto zero-sum gauge | 142 | 131 | yes |

“Exact cell identity” is scoped to the five frozen-head strict prototype witnesses;
it is not an unmeasured global equivalence claim over all feature space.

## Surface 4 — exact fail-closed blocker

The auditor consumed the two composed real VJP custody manifests:

- `chunk_000_010_024_composed/manifest.json`, SHA
  `3d1218...0694`, 12 sidecars / 1,820,970,986 bytes;
- `chunk_012_017_019_023_025_composed/manifest.json`, SHA
  `200e8c...e94`, 12 sidecars / 1,820,536,458 bytes.

Every referenced sidecar byte was rehashed. Each retained record carries winner,
rival, `seg_q`, measured local Lipschitz, head-pair norm, and Pose `J_x/J_y` tensor
hashes. The union covers 24 unique logical pair ids and leaves 576/600 absent.

The contemporaneous real n600 sparse-selector receipt was ingested as one immutable
byte snapshot, SHA
`c86d15ff905c1de912c6a66fe63c2122bd69115a536f70265c59d9ae8cd34a68`,
8,195 bytes. It reports 17,926 baseline hard-oracle flips, 16,751 candidate
evaluations, `hard_gate_pass=false`, and a Fisher-margin tie-break label, but no
hashed `measured_reverse_waterfill_highest_ev_first` artifact over exactly 38,077
candidates.

Therefore the exact blockers are:

1. `INCOMPLETE_PAIR_LOCAL_VJP_CUSTODY`: 24/600 present, 576 missing;
2. `EXACT_38077_CANDIDATE_EV_FIELD_ABSENT`: the inspected real n600 selector is not
   the required hash-bound measured Fisher EV ordering.

No radii store and no `c2_anisotropic_band_artifact.v1` manifest were emitted. The
24 sidecars remain valid advisory pair-local evidence; this negative is scoped only
to full-n600 positive-band assembly. M1 remains prohibited from firing.

## Verification and review

- `114 passed`:
  `test_prereq_surfaces.py` plus the full existing uint8-lattice regression suite;
- Ruff: all five touched Python surfaces clean;
- `git diff --check`: clean;
- two bounded Sol-high passes recorded: structural correctness, then adversarial
  NO-FAKE/custody review;
- review fixes included the generic candidate serializer, hash-bound measured-EV
  requirement, receipt-derived anchor numbers, per-surface consumers/pointer, and
  exact-byte JSON snapshot hashing for the live SSD receipt.

The three stretch regmax probes were not rerun. The remaining budget was spent on
the real surface-4 SSD custody audit and the live-receipt race fix. Their designs are
unchanged; MAIN may route them only after adopting surfaces 1–3.

## Triality and system wire-in

- **Equations:** consumes, without re-registering,
  `bounded_uint8_resize_preimage_cell_feasibility_v1` and
  `segnet_head_rank4_linear_flipdist_v1`; the comparator records its operational LP,
  tropical, and zero-sum gauges directly in receipts.
- **DSL/control:** no flags were invented. The adapter/generic serializer are callable
  typed Python boundaries; existing regmax probes remain the owners of treatment
  configuration and fresh hard-oracle execution.
- **DAG:** `FEED-prereq-surfaces-flush-20260720` in
  `prereq_surfaces_flush_DAG_FEED_20260720T171050Z.md` connects the prior N-A probe
  verdicts and M1 refusal to these readiness gates.
- **Sensitivity/Pareto/bit allocation/autopilot:** no score or per-byte marginal was
  measured, so no sensitivity, Pareto, allocator, or autopilot admission row is
  appended. M1 dispatch remains false. Reusable signal is encoded in modules,
  regression tests, SHA-pinned receipts, and the exact blocker.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; operating manual; v7.5 operating contract; `PROGRAM.md`;
regmax probe memo/manifest at commits `35ef3613ca..14bbca8317`; M1 governed refusal
branch and blocker receipt; current frontier/lane/task/subagent state; central DAG
FEED-R1R2; frozen SegNet weights; real n600 GT cache; both composed 24-sidecar VJP
manifests and bytes; current r2b n600 selector receipt; prior n6 exact-lattice
frozen-SegNet receipt; task inbox through `2026-07-19T19:48:01Z`.

## MAIN landing requirement

This branch is not repository authority. MAIN must review
`8680c8e4f6ed2fe4186be7d9e60ebb9459183cc7..codexwt/prereq_surfaces_flush_20260720T163539Z`,
rerun the focused tests, verify the receipt manifest and surface-4 SSD snapshot
custody, and merge explicitly. After merge, surface 4 remains blocked and the score
pointer remains unchanged.
