# ddm_pi136 leaderboard breadth intake — public PRs after #132

**Date:** 2026-08-10  
**Arm:** `ddm_pi136`  
**Scope:** public breadth after the 2026-07-25 PR129–132 intake; PR135 depth is delegated to `ddm_pi135`  
**Authority:** source/archive reverse engineering only; this arm did not claim or run a scorer slot

## Conclusion

The bounded post-132 census contains PR133, PR134, PR135, and PR136. PR135 remains wholly with the
sister depth intake. This breadth pass therefore intakes PR133, PR134, and PR136. It did not find a PR
number above 136 or a new non-PR leaderboard row in the searched GitHub/pointer surfaces as of
2026-08-10. The two non-PR controls in the current pointer, `baseline_fast` and `no_compress`, were
already present in May snapshots and are not new since the 2026-07-25 cutoff.

Only PR133 transfers directly to the current vehicle: its full-n600 pose-coordinate re-solve is a real
same-lineage improvement, while its three-atom coarse-basis quantization is only a narrow secondary
gain. PR134 is a genuinely different classical AV1 plus scorer-sidechannel family, but its candidate is
dominated and its most useful exact-grid law is already ours. PR136 is a PR95 vehicle carrying the PR112
adaptive order-0 range-coder idea that Pact already absorbed; its included build path is also not a valid
reproducer. No breadth result moved the exact pointer.

## Scope, custody, and authority

- GitHub PR creation-date search for `created:>=2026-07-25` returned exactly PR133–136. PR135 is linked
  below and is not duplicated here.
- The canonical pointer snapshot at 2026-08-10T16:53:24Z has 65 rows: 63 PR-linked rows and the two old
  non-PR controls above. This is a bounded absence claim about those searched surfaces, not a claim that
  no other public artifact exists anywhere.
- Shell GitHub DNS was unavailable and the in-app browser had no browser instance. Source custody is
  therefore a complete, commit-pinned, read-only GitHub API snapshot of each changed submission tree,
  not a detached Git clone. This limitation does not affect the source-file mechanism reads, but it is
  recorded so the snapshots are not mislabeled as clones.
- All materialized source and archive payloads are retained read-only under
  `/Volumes/VertigoDataTier/pact/pr_breadth_intake_20260810/`. Nothing materialized by this arm was
  discarded.
- Source snapshot receipts:
  - PR133: head `7314df52a233f767e164f8de9d312af739692381`; 11 files; 86,432 bytes;
    deterministic inventory digest `9a58adb092b51c0ff356ad20933674144613a9a6b44b4819aa1c546d2728654e`.
  - PR134: head `f5267220e3ef72853001becf1b4dcb7754027839`; 22 files; 78,495 bytes;
    deterministic inventory digest `b124cba83e31628f9590e5fccea7861b44669fd49e90a792373e64a7d595f350`.
  - PR136: head `95d1b49b21c4d0a596bcd47c6ca2edd8c15b5b48`; 22 files; 70,589 bytes;
    deterministic inventory digest `7f26d9f9d4aa62a238fb776a03a0330b04e1cdc678fd4df52a89c04a9c131858`.
- Every archive byte count below marked **MEASURED** comes from the retained exact file. PR scores were
  not replayed by this arm. Bot values remain bot values; author-reported values remain external.

## Ranked cross-PR transfer table

The comparison base is lc2: 187,226 bytes, `d_seg=0.00029662`, `d_pose=0.00002332`,
`S=0.16959899569230852` `[contest-CUDA, exact upstream evaluator, n600]`, archive SHA-256
`f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45`.

| Rank | Public row | Public evidence and delta versus lc2 | Transfer verdict | Borrowed-substrate accounting | Consumer and disposition |
|---:|---|---|---|---|---|
| 1 | PR133 `cpr1_cbq_matched8` | Bot CUDA: 190,212 B, `d_seg=.00029660`, `d_pose=.00000896`; recomputed `S=.16578009084423384` from displayed components `[external official-bot contest-CUDA, n600]`. Delta: **−.00381890484807468 S**, +2,986 B. Exact archive is in custody. | **TRANSFERS**: full-n600 exact-coordinate re-solving and joint basis/coordinate search. The CBQ novelty itself supplied only 828 B and a small pose improvement against its matched eight-pass control. | PR130 renderer, HPAC, token stream, carrier format, and coefficient-search apparatus are borrowed. The re-solved values are learned/video-derived and must remain counted. PR133 is not ours-original. | **#995: FOLDED** into the already-queued current-base joint screen/freeze/frame-0 re-solve successor. **#984: FOLDED** as evidence, not a duplicate PR133 build. **#1009: FOLDED** by link to the PR135 depth lineage. |
| 2 | PR134 `metricwarp_av1` | Author CPU report: 464,856 B, `d_seg=.00531573`, `d_pose=.00094302`; recomputed `S=.9382107463768723` `[external source-reported CPU, n600]`. Delta: +.7686117506845638 S, +277,630 B. Exact archive is in custody. No bot row. | **NEW-MECHANISM** as a packaged classical AV1 plus correction family; **DOMINATED** as a candidate/vehicle. Exact-grid inversion and layered score selection are **ALREADY-OURS**. This is an instance/formulation verdict, not a family kill. | SVT-AV1 is a generic borrowed codec. The correction payloads are video-derived and counted. Pact independently has the disjoint-support exact-grid law and selector/waterfill machinery. | **#984/#995: FOLDED** as an implementation reference only; no candidate fire. **#1009:** no depth action. |
| 3 | PR136 `hnerv_rc` | Author CPU report: 177,998 B, `d_seg=.00057163`, `d_pose=.00002856`; recomputed `S=.19258426607726234` `[external source-reported CPU, n600]`. Delta: +.02298527038495382 S, −9,228 B. No bot row and no archive custody. | Adaptive order-0 coder is **ALREADY-OURS** through PR112; HNeRV vehicle is **DOMINATED**. Its public archive remains unavailable, so this verdict covers source mechanism and reported candidate, not exact wire anatomy. | PR95 `hnerv_muon` decoder/curriculum and PR112-style adaptive coder are borrowed. Pact already implemented and exact-tested the latter. | **#984/#995/#1009: FOLDED**; do not rebuild or launch. Archive custody alone is **QUEUED-WITH-A-FIRE-ORDER** for evidence closure. |

The ranking is by expected value to our live vehicle, not by public score alone. PR134 ranks over PR136
because it contributes a distinct packaged family and a clear receiver implementation, even though the
candidate score is much worse. Neither warrants a scorer run.

## PR133 — `cpr1_cbq_matched8 submission (0.1658)`

### Identity and score evidence

- PR: <https://github.com/commaai/comma_video_compression_challenge/pull/133>
- Author: `JasonMo123`
- State at intake: open, unmerged
- Head: `7314df52a233f767e164f8de9d312af739692381`
- Official bot comment:
  <https://github.com/commaai/comma_video_compression_challenge/pull/133#issuecomment-5228727041>
- CUDA row: 190,212 bytes, `d_seg=0.00029660`, `d_pose=0.00000896`
  `[external official-bot contest-CUDA, n600]`. The score recomputed from those rounded displayed
  components is `0.16578009084423384`. The author gives the compatible rounding interval
  `[0.16577695, 0.16578323]` and self-report `0.165780`.
- CPU row: none found in PR body or bot comments. One earlier bot invocation failed before the successful
  CUDA result.

### Exact archive custody and anatomy

The exact archive is retained at
`/Volumes/VertigoDataTier/pact/pr_breadth_intake_20260810/pr133/archive/archive.zip`.

- `archive.zip`: **190,212 bytes MEASURED**, SHA-256
  `051baf408f57fae3b343d6ee218ab963d070b3935ceb0b2f412c93a53cf3fab0`; ZIP integrity passes.
- One `ZIP_STORED` member `p`: 190,112 bytes, SHA-256
  `fee99107d82b0e6b0bcf1babea6e57fa674031efba182cec87e2ee6bccb1c444`.
- ZIP container overhead: 100 bytes.
- `p` is exactly `[models_len:u32=73128][models.xz:73128][tokens.range:116980]`.
- `models.xz`: 73,128 bytes, SHA-256
  `683c99ab5fd695871f430578d72528de7091a14469230ba59a3e625c1bcd145d`.
- `tokens.range`: 116,980 bytes, SHA-256
  `948379872ff81a4e5d948ec301c143be00ebd0033544c8abdfb4af0f4c4a15eb`.
- Decompressed `models.raw`: 82,743 bytes, SHA-256
  `ee533ce4b7aab41719f5697bbc75a9e28d1612b00bb9a98d14f39501c131cb87`.
- `models.raw` is exactly `[semantic_len:u32][carrier_len:u32][semantic_renderer][pose_carrier][hpac]`:
  - length header: 8 bytes;
  - semantic renderer: 40,252 bytes, SHA-256
    `9b98360bd56918b5a414ace375c29790b7fe9f7f55cf423c0564ef4e62a39b99`;
  - pose carrier: 22,304 bytes, SHA-256
    `080aaf3206e1afc1449c8deff14362bb0b910d937df15106bfb53befe6d5045e`;
  - HPAC model: 20,179 bytes, SHA-256
    `b07fff73fac41c5fec2d8acbfd7c43c518852696f18d95cf7465fc6ed7510b58`.
- Carrier bit anatomy: 152-byte fixed prefix, 12,277-byte basis payload carrying 98,213 coded bits,
  and 9,875-byte coefficient payload carrying 79,000 coded bits.

Every listed materialized payload is retained read-only beside the archive with its exact byte identity.

### Nearest-ancestor diff and mechanism

The nearest ancestor is PR130, not a new family. Against the retained PR130 base archive
`0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`
(191,052 bytes), PR133 is 840 bytes smaller. The decoded token stream, semantic renderer, and HPAC
model are byte-identical. All raw-model change is in the pose carrier:

| Section | PR130 | PR133 | Delta |
|---|---:|---:|---:|
| Archive | 191,052 | 190,212 | −840 |
| Stored member | 190,952 | 190,112 | −840 |
| `models.xz` | 73,968 | 73,128 | −840 |
| `models.raw` | 83,493 | 82,743 | −750 |
| Pose carrier | 23,054 | 22,304 | −750 |
| Carrier basis payload | 13,017 B / 104,135 bits | 12,277 B / 98,213 bits | −740 B / −5,922 bits |
| Carrier coefficient payload | 9,885 B / 79,076 bits | 9,875 B / 79,000 bits | −10 B / −76 bits |

The remaining 90-byte archive gain is LZMA interaction. The mechanism coarsens support for basis atoms
2, 5, and 9 from five bits to four bits, then re-solves all int12 per-pair carrier coordinates through
exact PoseNet coordinate search over eight full n600 passes. There is no new runtime or outer codec.

The source's matched-effort control is important. Eight passes with the unchanged basis reached
`d_pose=.0000080833270`, 191,040 bytes, and 205 accepted moves; CBQ reached
`d_pose=.0000078771227`, 190,212 bytes, and 140 accepted moves. Against that control, CBQ accounts for
only 828 bytes and about 2.55% pose improvement. The dominant improvement relative to public PR130 is
the inherited full-n600 coefficient re-solve, and neither eight-pass search had converged.

Directly quantizing the basis without re-solving the coefficients is closed at the tested formulation:
the author's 100-pair atom-9 control worsened pose from `.00002076` to `.00060345`. The transferable
unit is therefore joint basis-and-coordinate optimization, not independent coarse quantization.

### Transfer to lc2

The displayed bot components imply the following delta versus lc2:

- segmentation contribution: −0.000002;
- pose contribution: −0.005805159682;
- rate contribution: +0.001988254834 from +2,986 bytes;
- total: −0.003818904848.

This is a real external bot result that beats lc2's exact score, but it is not our exact row and did not
move our pointer. A purely arithmetic cross-state projection applying lc2's −3,826-byte PR130 rate delta
to PR133 yields 186,386 bytes and `S=.1632325145`; that is explicitly a **PROJECTION**, not a byte-closed
candidate, and remains about .00123 above PR135's displayed `.162`. It must not be promoted without an
actual current-container composition, retained archive, and exact replay.

## PR134 — `Add metricwarp_av1`

### Identity and score evidence

- PR: <https://github.com/commaai/comma_video_compression_challenge/pull/134>
- Author: `bzlvkv`
- State at intake: closed, unmerged
- Head: `f5267220e3ef72853001becf1b4dcb7754027839`
- Bot CPU row: none found.
- Bot CUDA row: none found.
- Author/source report: 464,856 bytes, `d_seg=.00531573`, `d_pose=.00094302`, recomputed
  `S=.9382107463768723` `[external source-reported CPU, n600]`.

### Exact archive custody and anatomy

The retained archive is
`/Volumes/VertigoDataTier/pact/pr_breadth_intake_20260810/pr134/archive/archive.zip`.

- `archive.zip`: **464,856 bytes MEASURED**, SHA-256
  `9cb7c817f69b5d63192589344870eb41bd14b27a1669e7738a00af90c93e9d7d`; ZIP integrity passes.
- Four `ZIP_STORED` members total 464,448 bytes; ZIP overhead is 408 bytes:
  - `finalA.obu`: 456,276 bytes, SHA-256
    `02bb30cb1ec032a473fa9d550bf1e7550f56d9888073eb9815b84212c9c0649b`;
  - `corr.bin`: 1,686 bytes, SHA-256
    `769c65eac9cff72068a2cb17048bcf068cc9bbb59a991432d96656c7f254889c`;
  - `segfix.bin`: 5,617 bytes, SHA-256
    `03b335015d31084c109526db1431ed79f91f1a9db83dfc61bfc3cfaaf3e0e8bc`;
  - `manifest.json`: 869 bytes, SHA-256
    `b7b6bb924fd56bc2d0e27106eb4649f6dd7a41e387073fc59df190889da041a1`.
- Decompressed `corr.raw`: 3,600 bytes, SHA-256
  `b6c1436b8007b3cc21818a43c975ce1c7791a98a8940c23221a87bff52dc5332`;
  exactly 600×6 int8. It has 596 nonzero pair rows and 2,102 nonzero scalar values; zero rows are
  235, 297, 393, and 423.
- Decompressed `segfix.raw`: 10,803 bytes, SHA-256
  `973352694cd511e23ed043ecad033cc063eec3b88527f1e1fc1be1c69a97d0fa`;
  561 nonempty pair records and 2,280 four-byte tile actions. The size closes exactly as
  `3*561 + 4*2280 = 10,803`.

The PR body says 562 of 600 pairs keep segmentation fixes, but the retained wire parses to 561 nonempty
records. A logically selected no-op pair or a report/wire mismatch could explain the difference; the
archive supports 561, so this memo does not silently promote 562 to a measured wire fact.

### Vehicle and mechanism

This is a different classical family, not a PR130 polish step. It encodes full-resolution 10-bit video
as a single-keyframe raw SVT-AV1 OBU at preset 2 / CRF 56, then applies two counted sidechannels:

1. On even frames, per-pair six-parameter warp/bias/gain corrections are searched through the exact
   scorer geometry with uint8 rounding in the loop to reduce PoseNet error.
2. On odd frames, sparse tile RGB corrections target segmentation flips.
3. A layer mixer selects the better per-pair candidate.

The receiver exploits private, pairwise-disjoint 2×2 source supports to invert the evaluator's bilinear
downsample at scorer coordinates. It fills metrically invisible pixels for cosmetic continuity. The
source report says full resolution beat a downscaled encode, 22.7–23% of native pixels are blind to the
scorer grid, pose fell from about 1.23 to .000943, about 17% of flipped pixels were repaired, and decode
took about 35 seconds on CPU. Those performance statements remain external; the exact receiver mapping
and wire parser were verified from the pinned source snapshot.

### Transfer to lc2

As a candidate, PR134 is dominated: +277,630 bytes and +.768611750685 score versus lc2 from the external
components. That is an **INSTANCE/FORMULATION** verdict only. It does not kill classical video plus sparse
score correction as a family.

The most valuable ideas are not novel to Pact. The full-corpus recall found the same disjoint-support
exact-grid actuator and 22.70% joint blind-coordinate geometry in
`src/tac/optimization/ddm_bp2_blind_pose_actuator.py`, `ddm_ll1_window_solve.py`, and
`ADVISORY_evaluator_video_geometry_20260710.md`; exact rounding/realized-through-R is also standing law.
Layered per-pair score selection is already present in the selector/waterfill corpus. PR134 is useful as
a compact independent receiver implementation, but it supplies no reason to fire a duplicate build or
scorer row.

## PR136 — `hnerv_rc: 0.19258 (CPU axis) — adaptive range coder on the hnerv_muon pipeline`

### Identity and score evidence

- PR: <https://github.com/commaai/comma_video_compression_challenge/pull/136>
- Author: `JPL11`
- State at intake: closed, unmerged
- Head: `95d1b49b21c4d0a596bcd47c6ca2edd8c15b5b48`
- Bot CPU row: none found.
- Bot CUDA row: none found.
- Author/source CPU report: 177,998 bytes, `d_seg=.00057163`, `d_pose=.00002856`, recomputed
  `S=.19258426607726234` `[external source-reported CPU, n600]`.
- The author also reports a GPU/DALI result near `.23`, without an exact bot row or sufficient components;
  it is not used in this comparison.

### Archive boundary

The release archive could not be acquired. The GitHub repository connector exposes source but not the
release asset, shell DNS was unavailable, the browser instance was unavailable, and a bounded exact-size
search of the Vertigo and AP custody tiers found no 177,998-byte candidate. No archive SHA-256 is
published in the PR body. Consequently, **no PR136 byte count or section size is marked measured here**.

Source code defines the expected archive as ZIP member `0.bin`, with internal layout
`[version:u8][meta_len:u32][brotli_meta][decoder_len:u32][decoder][latent_len:u32][latent]`. That is a
source-level wire schema, not an exact section map. Exact member bytes, component lengths, compression
interactions, and hashes remain blocked on archive custody.

### Nearest ancestor, mechanism, and reproduction defect

The vehicle is directly PR95 `hnerv_muon`: the same approximately 229K-parameter HNeRV decoder and the
same eight-stage CE → softplus → smooth → QAT → L7/C1a → lambda → sigma → Muon pipeline. Its changes are
a better retrain from random (author reports about .1931 before the coder) and a v2 per-tensor adaptive
order-0 constriction range coder. The model starts uniform, increments the observed symbol by a fixed
eight, and transmits no frequency table. Latents receive related delta/zigzag low/high coding.

The source report says Brotli was 163,237 bytes, a static entropy estimate was 160,387 bytes, and the
coder saved about 1.1 KB. The code documentation separately says decoder range coding was 161,736 bytes,
1,501 bytes below Brotli. Because no exact archive is in custody, this internal source inconsistency is
left explicit rather than resolved by assumption.

The included `compress.sh` invokes `python -m submissions.hnerv_muon.src.train`, then searches under the
`hnerv_rc` checkpoint tree. The included `src/train.py` usage also names `hnerv_muon` and explicitly says
there is no resume or mid-pipeline shortcut. Therefore the checked-in compression path cannot reproduce
the claimed `hnerv_rc` artifact as written and violates Pact's P0 resumability/per-stage-checkpoint
contract. It must not be launched. The decoder also reconstructs a 256-way probability vector and a
categorical model per symbol; a public review comment flagged likely inflate-time cost, but exact runtime
cannot be adjudicated without the artifact and a governed replay.

The author's asymmetric frame-0 pose-carrier attempt added about 8 KB and landed near .196 on this HNeRV
vehicle. That is an external **FORMULATION** negative for that vehicle, not a family verdict about
frame-0 pose actuation.

### Transfer to lc2

The coder is not new to Pact. PR136 explicitly follows PR112, and the full-corpus recall found the prior
absorption receipt `leapfrog_pr112_absorb_recode_verdict_20260610.md`: per-tensor adaptive 256-ary
constriction coding, geometric priors, and causal latent coding were already incorporated into an exact
177,169-byte / `S=.19109982` CPU row at that time. The canonical equations also retain PR103 range coding,
and the repository has `src/tac/lossless/range_coder.py`.

Against current lc2's cross-axis anchor, PR136's external CPU components are 9,228 bytes smaller but
.022985270385 score worse; the distortion loss dominates. The mechanism is **ALREADY-OURS**, the vehicle
is a forbidden PR95 reskin rather than the task-space witness, and the included producer is broken and
non-resumable. No build, training run, composer fire, or scorer row follows from PR136. Only exact archive
custody remains as a low-priority evidence-closure task.

## PR135 sister handoff

PR135 depth belongs to `ddm_pi135`; this memo does not repeat it. Use:

- `.omx/research/ddm_pi135_pr135_intake_20260810.md`
- `.omx/research/pr135_pr133_direct_intake_facts_20260810.md`

PR133 evidence above is folded into #1009 only by cross-link. Any PR133/PR135 composition or current-base
successor must consume the sister's exact custody and decision record rather than rebuilding depth here.

## Consumer routing and follow-on dispositions

- **#995 roadmap — FOLDED.** Consume PR133's full-n600 exact-coordinate re-solve and joint
  basis/coordinate search in the existing current-base screen → freeze → frame-0 re-solve successor.
  Do not fire an isolated PR133 clone, and do not treat three-atom CBQ as the main gain.
- **#984 composed campaign — FOLDED.** PR134's exact-grid receiver is an independent reference for an
  actuator we already own; PR136 contributes no new lossless composer. No duplicate campaign row fires.
- **#1009 PR135 depth — FOLDED.** Link the exact PR133 anatomy and matched-control attribution into the
  sister lineage record. Do not duplicate PR135 analysis.
- **Lossless-pack composer — FOLDED.** PR133 changes its counted pose carrier, not the outer lossless
  token stream; PR134's AV1 package is dominated; PR136's adaptive coder is already absorbed from PR112.
  Breadth provides no new reason to fire a composer.
- **PR136 archive custody — QUEUED-WITH-A-FIRE-ORDER.** Fire only when the exact release binary becomes
  reachable through a functioning network/browser path or an operator-provided mirror. Retain it first,
  hash it, parse all sections, and append a superseding custody receipt without launching a scorer.

## RECALL EVIDENCE

Recall was performed before the transfer verdicts, beyond the charter's seed documents.

### Sources and queries

- Canonical equation registry:
  `.venv/bin/python tools/list_canonical_equations.py --json`, queried for `range coder`, `adaptive`,
  `order0`, `Brotli`, `pose carrier`, `coefficient`, `quantization`, `bilinear`, `grid`, `rounding`,
  `AV1`, `HPAC`, and `CPR1`.
- Graph-memory surfaces: `.omx/research/CANONICAL_RESEARCH_INDEX*` and `sub015_DAG_*`, searched by the
  same mechanism terms and PR numbers.
- Full `.omx/research/` and source corpus content searches for:
  - `compensability|exact coefficient|CBQ|PR133`;
  - `exact-grid|pairwise-disjoint|metric-invisible|PR134`;
  - `adaptive order-0|fixed increment|hnerv_rc|PR136`.
- Intake and lineage sources beyond the charter: the PR135/PR133 direct facts; PR112 intake and
  `leapfrog_pr112_absorb_recode_verdict_20260610.md`; evaluator-geometry/disjoint-support exact-grid
  receipts; PR103/PR112 canonical equations; the public PR129–132 intake; and the broader PR112–127
  full-stack intake.
- External source authority: commit-pinned GitHub PR133, PR134, and PR136 changed submission trees,
  their bodies/comments, and the PR133 bot comment linked above.

### What recall changed

- The charter predicted PR133 and PR134 as same-lineage polish. Source/archive custody showed PR133 is
  that polish, but PR134 is a genuinely different classical AV1 family. Its candidate still does not
  beat .19.
- PR134's highest-value exact-grid/disjoint-support mechanism is already ours, so the initial
  `NEW-MECHANISM` impression was narrowed to its packaged family and no build was fired.
- PR136's coder is an explicit PR112 descendant already absorbed by Pact. That changed a possible codec
  port into `ALREADY-OURS`, and its broken non-resumable PR95 producer closed the vehicle path.
- PR133's matched eight-pass control showed that the large improvement is inherited coefficient search,
  not CBQ. Routing therefore goes to #995's joint re-solve successor rather than a standalone
  quantization experiment.

## Boundaries and goal status

- **MEASURED:** exact PR133 and PR134 archive bytes, hashes, ZIP membership, retained decoded-section
  bytes/hashes, PR133-to-PR130 byte anatomy, and PR134 correction-stream record counts.
- **NOT MEASURED:** any PR score by this arm; any PR136 archive byte or exact section; any PR133+lc2
  composition; any CPU/CUDA parity inference; any scorer runtime.
- Public bot/source scores are external and axis-labeled. PR133's displayed-component score is a formula
  recomputation from rounded bot components, not a local replay.
- No full-n600 scorer job, training run, archive mutation, or paid dispatch was launched. This breadth
  intake is a means, not a score result. The exact pointer stayed at lc2 and remains above the sub-.15
  goal.

## NEXT_IF_RESUMED

- **PR133 joint re-solve — disposition: FOLDED; owner: #995 roadmap owner; consumer store: the governed
  current-base joint screen/freeze/frame-0 re-solve successor store named by #995; fire trigger: the
  PR135/current-base producer is byte-closed, resumable, per-stage checkpointed, and has the fleet scorer
  slot.** Apply coefficient re-solving jointly with any basis quantization; retain every candidate
  payload and do not launch a PR133-only clone.
- **PR136 archive custody — disposition: QUEUED-WITH-A-FIRE-ORDER; owner: next public-intake custody
  owner; consumer store: `/Volumes/VertigoDataTier/pact/pr_breadth_intake_20260810/pr136/archive/` plus
  this memo; fire trigger: the exact release binary becomes reachable through a functioning network or
  browser path, or the operator supplies a mirror.** Retain first, record exact SHA-256 and bytes, parse
  the exact section map, and append a superseding custody receipt; no scorer fire.

## LIVE-HYPOTHESES

- PR133's unfinished full-n600 coordinate search may still improve the current PR135/lc2-derived carrier
  because both its unchanged-basis control and CBQ arm were accepting moves after eight passes. This is
  plausible from the matched control, but it must be tested on the current byte-closed object rather
  than transferred as a number.
- Jointly re-solving coefficients after basis quantization may expose a small rate/pose Pareto gain on
  the current vehicle. Direct quantization fails, but PR133 demonstrates that compensating coefficients
  can rescue selected atoms; the useful hypothesis is joint optimization, not CBQ alone.
- PR134's exact-grid receiver may still be useful as a compact independent parity reference when an
  existing Pact actuator needs a receiver audit. The law is already ours, so this is plausible only as
  verification leverage, not as a new score vehicle.
- PR136's exact archive, if recovered, may clarify the reported 1.1 KB versus 1,501-byte coder discrepancy
  and actual decode cost. That can improve provenance, but the mechanism/vehicle verdict is unlikely to
  change because PR112 absorption and the broken producer are source-level facts.

## DEAD-ENDS

- Independent coarse basis quantization without coefficient re-solving is closed at the tested PR133
  formulation: the 100-pair atom-9 control increased pose error by about 29×.
- Treating CBQ as the main PR133 gain is closed: the matched eight-pass control attributes only 828 bytes
  and about 2.55% pose improvement to CBQ; inherited coefficient search dominates.
- Building PR134 as a current candidate is closed at this instance/formulation: it is 277,630 bytes and
  .7686 score worse than lc2, while its transferable exact-grid law is already implemented in Pact.
- Porting PR136's adaptive order-0 coder is closed: Pact already absorbed the PR112 mechanism, and PR136
  supplies no new exact archive evidence.
- Re-running PR136's HNeRV training path is closed: it is a PR95 reskin, the included producer invokes the
  wrong submission, and it explicitly lacks resume/mid-pipeline support.
- Claiming a pointer move from this intake is closed: no score was run. The own exact frontier remains
  lc2 at **S=0.16959899569230852, 187,226 bytes `[contest-CUDA, n600]`**.
