# Contest-Faithful Swarm Execution - 2026-05-02 Codex

## Scope

This ledger records the contest-faithful branch only. PR70/PR69-style
code-payload or malformed-ZIP behavior is useful reverse-engineering evidence,
but it is not a score claim and is not a submission target unless the organizer
explicitly changes the accounting rule.

## Current Frontier

- Active internal A++ frontier: C-067 / Apogee candidate.
- Score: `0.31561703078448233`.
- Archive bytes: `276214`.
- Archive SHA-256:
  `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`.
- Exact evidence:
  `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.adjudicated.json`.
- Archive:
  `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip`.
- Evidence grade: A++ exact Tesla T4 CUDA, `600` samples,
  `promotion_eligible=true`, component gates passed.
- Anatomy: charged PR67 mask segment `219472` bytes, C-059 model segment
  `55965` bytes, C-059 pose segment `677` bytes, plus ZIP overhead.

C-067 supersedes C-063 (`0.3156230307844823`), C-059
(`0.3157055307844823`), C-058 (`0.3157555307844823`), and C-057
(`0.3157562807844823`). The remaining unchanged-distortion gap to sub-`0.300`
is `23454` bytes (`23455` buffered), so the main path must be a material
charged representation/decoder/atom-allocation move rather than scalar byte
polish.

## Public Faithful Reference

The official visible leaderboard checked on 2026-05-02 lists `qpose14` / PR #63
at rounded `0.32` as the top visible video-compression entry. PR67 remains an
external public-floor-basin design/input reference for our C-067 fixed-slice
work, but any PR67 text or non-T4 replay is external context until exact archive
custody and contest-equivalent evaluation are established.

C-067's `0.31561703078448233` exact T4 score would round into the visible
`0.32` band, and it is below the rounded-component estimate recorded for the
visible #1 row. The public rank is still not claimed until Apogee is submitted
and accepted on the public leaderboard.

## Active Swarm

- Noether: charged QZS/QP pose-packer implementation surface.
- Popper: PR65/PR67 atom water-fill and component-safe postprocess planner.
- Zeno: contest-faithful report, writeup, and submission-pipeline narrative.
- Herschel: charged mask grammar, ego-motion, and foveation Grand Council with
  recursive senior-engineer greenup.

All workers were instructed to preserve shared dirty state, avoid remote jobs,
and record changed files.

## Highest-EV Strict Experiments

1. **QZS4/QP2 charged format leap.** Move PR70/PR67 codec lessons into
   charged archive payloads: tensor ordering, grouped bit-depth search,
   arithmetic/range-coded pose atoms, and deterministic single-blob packing.
   Exact gate: build complete archive, fast H100 diagnostic eval, T4 promotion
   only if expected delta is material.

2. **PR65 atom water-fill.** Do not apply PR65 postprocess wholesale. Rank
   atom families by expected component benefit per charged byte, then test
   complete stacked archives. Exact gate: no additive promotion without a
   stacked CUDA archive.

3. **Q-FAITHFUL five-stage QAT++ export.** Continue the A100 training path
   but require deterministic charged export closure before any exact score
   claim. This is the highest-upside learned route to true sub-0.3.

4. **Charged mask grammar/foveation lane.** Replace or shrink the large mask
   stream with geometry-aware atoms: ego-motion, directional anisotropic
   foveation, connected components, temporal boundary grammar, and learned
   selectors. All grammar tables and learned state are charged bytes.

5. **Report/submission automation.** Every candidate that reaches A/A++ must
   automatically emit archive bytes, SHA, component distances, score recompute,
   hardware, manifests, provenance, report text, and claim-matrix row.

## Hard Boundaries

- No malformed ZIPs for strict score claims.
- No task-specific constants, model weights, mask data, pose data, generated
  lookup tables, or compressed payloads in uncharged source files.
- H100 results are proposal ranking only unless the contest-equivalent policy
  changes. T4/equivalent exact eval remains the promotion gate.
- PR70/PR69 remain external exploit forensics, not stack inputs.

## Next Dispatch Rule

Do not spend another T4 job on a one-byte or scalar-only pose checkpoint unless
it is already complete and needs harvest. The next T4 should promote a material
format/grammar candidate. Fast H100/A100 work should be used for candidate
construction, exact diagnostic screens, and Q-FAITHFUL/grammar training.

---

## Update - 2026-05-02T03:26Z QZS4 Block128 Negative And Repacker Hardening

The first QZS4/QP1 material-format diagnostic was harvested from H100:

- Candidate:
  `experiments/results/qzs4_qp2_stack_screen_c058_20260502/qzs4_maskfirst_qp1/archive.zip`
- Archive bytes: `273247`.
- Archive SHA-256:
  `32ce9cd3ebbbfa6b3468dc9ba7ac31f0fbd2802a8eb25f6bdd44c137d9f41c69`.
- Exact diagnostic artifact:
  `experiments/results/vast_harvest/h100_diag_qzs4_maskfirst_qp1_c058_20260502T0321Z/contest_auth_eval.json`.
- Score: `1.5244097988910252`.
- Components: PoseNet `0.156837`, SegNet `0.0009012`.
- Evidence grade: A-negative diagnostic CUDA, not T4 promotion.

Interpretation:

- The block128 byte-only QZS4 selection is narrowly retired as a measured
  implementation. It saved bytes but destroyed PoseNet geometry.
- This does **not** kill the charged QZS/QP packer family. It proves block-size
  and packed-runtime selection must be component-bounded, not byte-only.
- The next correct experiment is a bounded QZS3 block-size threshold sweep
  around the current frontier: b32 as layout-equivalence control, then b48 and
  b64 before considering any larger block or QZS4 variant.

Two repacker hardening fixes now protect the next sweep:

- Deployed single-blob frontier archives are valid sources. The repacker
  unpacks `p`/`renderer_payload.bin*` through the runtime payload unpacker and
  records `source_runtime_contract` provenance.
- `--qzs3-block-size` is honored for existing QZS3 sources. When the requested
  block size differs from the source block size, the repacker decodes and
  re-encodes instead of silently producing a no-op archive.

Current byte-screened follow-ups:

- `qzs3_b32_maskfirst_qp1_fix1`: `276347` bytes, SHA
  `cf44aa7fdb13b9ab6236aefde1f5f58e915d7dfa99128246235443841396c6ab`.
- `qzs3_b48_maskfirst_qp1_fix1`: `275151` bytes, SHA
  `852d30eca1f97949022760c9017d78984327cda17acbc559aa41aac63020ea40`.
- `qzs3_b64_maskfirst_qp1_fix1`: `274489` bytes, SHA
  `f917b14a2c67433b026fcdfc3fe18f4fda538a0ae556ef57ba70bc10301fe56c`.

Dispatch decision: run the three candidates sequentially on the warm H100 for
fast diagnostic exact CUDA evidence. Promote to T4 only if the component trace
stays near C-058 and the byte win is real.

---

## Update - 2026-05-02T03:36Z QZS3 Block Threshold Sweep

The bounded H100 sweep completed and was harvested locally under:

`experiments/results/vast_harvest/h100_diag_qzs3_block_sweep_c058_fix1_20260502T0330Z/`

Results:

| candidate | bytes | sha256 | H100 exact diagnostic score | PoseNet | SegNet | decision |
|---|---:|---|---:|---:|---:|---|
| b32 mask-first QP1 | `276347` | `cf44aa7fdb13b9ab6236aefde1f5f58e915d7dfa99128246235443841396c6ab` | `0.31508425622241343` | `0.0004909` | `0.00061012` | T4 promotion queued |
| b48 mask-first QP1 | `275151` | `852d30eca1f97949022760c9017d78984327cda17acbc559aa41aac63020ea40` | `2.1095761693454476` | `0.34243444` | `0.00075866` | A-negative, narrow retire for C-058 |
| b64 mask-first QP1 | `274489` | `f917b14a2c67433b026fcdfc3fe18f4fda538a0ae556ef57ba70bc10301fe56c` | `2.364080478962302` | `0.44303787` | `0.00076463` | A-negative, narrow retire for C-058 |

Interpretation:

- The geometry cliff is sharp: QZS3 b32 preserves or improves components on
  H100, while b48/b64 collapse PoseNet despite saving `1271`/`1933` bytes.
- For this frontier anchor, byte-only QZS block enlargement is unsafe above
  b32. Future QZS block experiments need component-aware mixed block sizes,
  not global block-size escalation.
- b32 is the only promotion candidate from this sweep. Lightning T4 job
  `exact_eval_qzs3_b32_maskfirst_qp1_fix1_t4_20260502T0331Z` is queued with
  expected bytes/SHA, cu124 inflate-side Torch pinning, adjudication, and
  component trace.

Bug/preflight note:

- The first T4 submit was blocked before spend because T4/g4dn exact evals must
  pin `INFLATE_TORCH_SPEC=torch==2.5.1+cu124`. This is a harness success, not a
  lane failure.
- The job-specific active-claim guard also blocked submit until a Lightning
  claim row was added. This is a coordination success; the helper currently
  enforces one active row per lane, so the row was added manually while the
  H100 sweep claim remained active.

---

## Update - 2026-05-02T03:47Z PVR1 Pose-Residual Atom Screen

A small charged pose-residual atom screen was built on top of the safe b32
packed-layout candidate:

- `top16`: `276607` bytes, SHA
  `0399ce47746cc8ed6d09a68c688e04a53e5d2c76d87ec0721acd8731d9b2939d`.
- `top32`: `276383` bytes, SHA
  `a59430492b89a8ec78c65140b747829d4c9dc2da5e868850a76c54fed1738dd3`.
- `top64`: `276417` bytes, SHA
  `e94ba0c6a3bf623fcd5436a833675b20fd82ccf1a2c5ffeadad56c2487f9b5d8`.
- `top128`: `276606` bytes, SHA
  `2f2b329be4a02315630db595d24e1a9ad82b18112a4db92c93a319dc35123ef2`.

Only top32/top64 were H100 exact-screened because top16/top128 were clearly
byte-regressive under the current C-058/b32 operating point. Harvest root:

`experiments/results/vast_harvest/h100_diag_qzs3_b32_pvr1_top32_top64_c058_20260502T0342Z/`

Results:

- top32: score `0.31510825622241345`, PoseNet `0.0004909`, SegNet
  `0.00061012`, bytes `276383`.
- top64: score `0.3151310062224134`, PoseNet `0.0004909`, SegNet
  `0.00061012`, bytes `276417`.

Decision:

- Do not T4-promote these raw-residual PVR1 variants. They did not improve
  components over b32 and only added bytes.
- This is a narrow measured negative for the current raw-residual-magnitude
  selector, not for pose side information generally. The next pose atom selector
  must be scorer-informed: hard pair, active subspace, Fisher/Hessian/Jacobian,
  or exact pair-delta weighted.

Tooling note:

- The child-claim helper hardening worked in real use:
  `--allow-parallel --child-of exact_eval_qzs3_b32_maskfirst_qp1_fix1_t4_20260502T0331Z
  --parallel-reason h100_pose_residual_screen_while_t4_promotion_runs`.

---

## Update - 2026-05-02T03:57Z Small-Block Fidelity Screen

The opposite side of the QZS3 block-size threshold was tested: smaller blocks
cost bytes but might improve renderer fidelity. Byte-screen:

- b16: `278665` bytes, SHA
  `04298c0506b323ab0e702e90a649f490059a8e9eeac32c7c5abe17369522f79f`.
- b24: `277282` bytes, SHA
  `981fed3993e59d3d1d52269e545ed0e0c59cf01b95b313fccbe3ed98207cf466`.

H100 exact diagnostic harvest root:

`experiments/results/vast_harvest/h100_diag_qzs3_smallblock_b16_b24_c058_20260502T0352Z/`

Results:

- b16: score `0.48225686102498067`, PoseNet `0.00550256`, SegNet
  `0.0006213`, bytes `278665`.
- b24: score `1.3137273302325199`, PoseNet `0.1119598`, SegNet `0.00070986`,
  bytes `277282`.

Decision:

- Do not promote b16/b24. For this C-058 anchor, global QZS3 block size is
  sharply tuned: b32 is the only surviving global block size among
  `16,24,32,48,64,128`.
- This points to mixed/local block allocation or learned quantization if we
  revisit renderer packing, not another global scalar block sweep.

---

## Update - 2026-05-02T03:50Z C-059 T4 Promotion Landed

The b32/mask-first/QP1 packed-layout candidate promoted on exact Lightning T4:

- Job: `exact_eval_qzs3_b32_maskfirst_qp1_fix1_t4_20260502T0331Z`.
- Archive bytes: `276347`.
- Archive SHA-256:
  `cf44aa7fdb13b9ab6236aefde1f5f58e915d7dfa99128246235443841396c6ab`.
- Recomputed score: `0.3157055307844823`.
- Components: PoseNet `0.00049637`, SegNet `0.00061244`.
- Evidence:
  `experiments/results/lightning_batch/exact_eval_qzs3_b32_maskfirst_qp1_fix1_t4_20260502T0331Z/contest_auth_eval.adjudicated.json`.
- Component trace:
  `experiments/results/lightning_batch/exact_eval_qzs3_b32_maskfirst_qp1_fix1_t4_20260502T0331Z/component_trace.json`.
- Evidence grade: A++ contest T4, `600` samples, `gpu_t4_match=true`,
  `promotion_eligible=true`, component gates passed.

SDK/custody note:

- The Lightning SDK status stream regressed nonterminal `Running -> Pending`
  after cost accrued, so the local queue state correctly marked
  `REMOTE_STATUS_RECONCILIATION_REQUIRED`.
- The state-derived artifact mirror already contained a complete terminal
  exact-eval packet. `harvest-ssh` validated archive bytes, SHA, JSON, and
  artifact mirror integrity locally before this row was promoted.

Interpretation:

- C-059 supersedes C-058 as the strict internal A++ frontier.
- The score improvement versus C-058 is `-0.00005`, exactly the charged rate
  value of `75` bytes under the contest formula. The next material move must be
  scorer-informed local atoms, charged mask grammar/foveation, Q-FAITHFUL
  export, or mixed/local renderer quantization. Do not spend T4 on another
  global scalar block sweep.

---

## Update - 2026-05-02T04:10Z C-059 Scorer-Weighted Pose Atom Policy

Implemented a deterministic non-promotable planner:

- Tool:
  `experiments/plan_scorer_weighted_pose_atoms.py`.
- Tests:
  `src/tac/tests/test_plan_scorer_weighted_pose_atoms.py`.
- Output:
  `experiments/results/pose_atom_plan_c059_20260502/pose_atom_policies.json`.

The planner reads C-059 exact auth-eval custody, C-059 component trace, public
PR67/PR63/PR64 traces when present, active-subspace metadata, and atom ledgers.
It emits ranked pair atoms and policies with `score_claim=false` and
`promotion_eligible=false`.

Top policy dispatched for H100 diagnostic search:

- Policy: `c_059_pose_atoms_top032`.
- Selected pairs:
  `164,64,130,112,97,153,70,198,420,289,166,435,78,418,87,159,67,46,496,290,156,37,103,143,546,191,107,129,111,221,426,406`.
- Formula-only expected net utility: `0.00042796260663844623`.
- Dispatch claim:
  `35985850:pact_ls_c059_weighted_pairs_top32_20260502T0410Z`.
- Remote output:
  `experiments/results/line_search_c059_weighted_pairs_top32_20260502T0410Z/`.

Important actuator correction:

- The planner's pair list is not a raw PVR1 residual-magnitude list. Current
  QP1 drops non-velocity pose columns, so residual top-K on the decoded current
  pose is structurally unable to help. The immediate actuator is pair-window
  QP1 velocity line search on the ranked hard pairs, then repack any accepted
  archive through the C-059 mask-first single-blob layout before exact eval.

---

## Update - 2026-05-02T15:15Z C-067 Observability And PMG/SJ-KL Tranche

Current frontier supersession:

- Active internal A++ frontier: C-067.
- Score: `0.31561703078448233`.
- Archive bytes: `276214`.
- Archive SHA-256:
  `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`.
- Exact evidence:
  `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.adjudicated.json`.
- Evidence grade: A++ exact Tesla T4 CUDA, `600` samples,
  component gates passed, promotion eligible.
- Anatomy: PR67 mask segment `219472` bytes, C-059 renderer/model segment
  `55965` bytes, C-059 pose segment `677` bytes. The local claim is for the
  exact charged archive bytes; the PR67 mask segment remains external-source
  attributed.

Byte-accounting observability:

- C-067 profile:
  `experiments/results/c067_archive_byte_accounting_20260502/archive_byte_accounting.md`.
- C-067 PNG profile:
  `experiments/results/c067_self_compression_profile_20260502/archive_byte_accounting.png`.
- Observability report:
  `reports/yousfi_fridrich_observability_20260502/observability_report.md`.
- Exact unchanged-distortion gap to sub-`0.300`: `23454` bytes, with a
  buffered planning target of `23455` bytes and target archive size `252759`.
- Stream pressure: `masks.mkv` is `219472` bytes, `renderer.bin` is `55965`
  bytes, `optimized_poses.bin` is `677` bytes, and ZIP overhead is only
  `100` bytes. Nested compression showed `0` best savings on the C-067 streams.

Interpretation:

- The profile is `empirical`/control-plane evidence only. It does not promote
  or rank a candidate.
- Generic self-compression is exhausted for the current byte grammar. Any
  sub-`0.300` path must change the charged representation/decoder grammar or
  trade bytes for measured component improvement inside exact archive custody.
- Pose bytes cannot close the gap alone; renderer-only work would need an
  implausibly large cut unless paired with a new representation. Mask-side
  representation changes remain the largest rate lever but are component-risky.

PMG atomtop4068 result:

- Candidate:
  `experiments/results/pmg_hotspot_candidate_c067_atomtop4068_20260502/archive.zip`.
- Byte profile:
  `experiments/results/pmg_hotspot_candidate_c067_atomtop4068_20260502/archive_byte_accounting.md`
  and `.png`.
- Exact diagnostic artifact:
  `experiments/results/lightning_batch/exact_eval_pmg_hotspot_atomtop4068_l40sdiag_20260502T1445Z/contest_auth_eval.json`.
- Archive bytes/SHA: `195762`,
  `2567dc04185cf20775f1f6c088395aa8df9e4484daa8b25001e940d62a5d6497`.
- Recomputed score: `28.41411894150047`.
- Components: PoseNet `62.34251404`, SegNet `0.03315286`.
- Hardware: NVIDIA L40S CUDA, `600` samples, `gpu_t4_match=false`.
- Evidence grade: A-negative scoped forensic / L40S diagnostic. It is not a
  T4 promotion row and cannot rank against C-067.

Interpretation:

- PMG atomtop4068 saved `80452` bytes versus C-067 but catastrophically
  collapsed PoseNet. The narrow measured implementation retired here is PMG
  row-span/row-run residual rescue at this atom scale, not charged mask grammar,
  learned topology, atom planning, or pose-conditioned residual coding.
- Do not queue another PMG row-run-only T4 promotion. The next mask-side work
  must change atom semantics toward multimask reconciliation, slot-aware
  JointFrameGenerator repair, learned topology, or pose-conditioned residuals.

SJ-KL runtime/tensor-prep status:

- Runtime integration is now a build/runtime support surface, not score
  evidence: charged `sjkl.bin` is handled by the robust-current runtime path
  and remains additive/optional with shape checks and no scorer imports.
- The target-slot bug was fixed in the production command contract: the runtime
  applies residuals to JointFrameGenerator pair slot `0` (`fake1`), so tensor
  prep and the next builder command now use `--target-slot 0` and record
  `target_slot=0`.
- Smoke tensor-prep manifest:
  `experiments/results/sjkl_tensor_prep_c067_smoke_20260502/sjkl_pair_tensor_prep_manifest.json`.
  It prepared `4` pairs with `gt_pairs_btchw` shape `[4, 2, 3, 384, 512]`
  and `renderer_target_slot_chw` shape `[4, 3, 384, 512]`.
- Full tensor-prep manifest:
  `experiments/results/sjkl_tensor_prep_c067_full_20260502/sjkl_pair_tensor_prep_manifest.json`.
  It prepared `600` pairs with `gt_pairs_btchw` shape
  `[600, 2, 3, 384, 512]` and `renderer_target_slot_chw` shape
  `[600, 3, 384, 512]`.
- Evidence grade: `build_tensor_prep_only`, `score_claim=false`,
  `promotion_eligible=false`.

Next tranche:

- Build `sjkl.bin` from the full prepared tensors using the manifest's
  `next_command_template`, then package it as a charged archive member.
- Before any GPU dispatch, require local decode parity, validator allowlist,
  deterministic manifest, payload closure, target-slot assertion, and no
  scorer-loads at inflate time.
- Exact CUDA eval may be requested only after the archive records its own
  bytes, SHA, runtime tree hash, sample count, component fields, and source
  tensor/provenance hashes.

2026-05-02 SJ-KL C067 swarm continuation:

- Spawned three bounded workers for SJ-KL packing, remote/runbook hardening,
  and paper/report integration. All returned without launching spend.
- Added deterministic C067/public-floor SJ-KL archive packer:
  `experiments/build_sjkl_c067_archive.py`.
- Added remote fail-closed driver:
  `scripts/remote_lane_sjkl_c067.sh`.
- Added focused tests:
  `src/tac/tests/test_build_sjkl_c067_archive.py` and
  `src/tac/tests/test_remote_lane_sjkl_c067_script.py`.
- Hardened `experiments/build_sjkl_residual.py` so CUDA builds fail closed
  unless explicit advisory non-CUDA mode is requested; manifests now record
  requested/actual device, advisory status, anchor indices, seed, basis SHA,
  coefficient-block SHA, total SHA, input tensor hashes, and `score_claim=false`.
- Fixed a critical pre-dispatch integration bug: an early remote driver version
  added `sjkl.bin` beside C067's packed ZIP member `p`. That is not a
  runtime-consumable charged payload for C067. The driver now calls
  `experiments/build_sjkl_c067_archive.py`, then fails if `sjkl.bin` is absent
  from `payload_member_names.output_logical_runtime_members` or if
  `score_affecting_payload_charged_in_archive` is not true.
- Real C067 structural smoke passed:
  `experiments/results/sjkl_c067_packer_contract_smoke_20260502T151210Z/`.
  Synthetic valid `sjkl.bin` was packed into the actual C067 `p` payload; the
  unpacker recovered `renderer.bin`, `masks.mkv`, `optimized_poses.bin`, and
  `sjkl.bin`. This remains structural custody evidence only, not score.
- Lightning Studio staging completed with manifest
  `.omx/state/sjkl_c067_stage_20260502T151337Z_lightning_source_manifest.json`:
  `1285` files, `2853284720` bytes, including the `2.7GB` renderer tensor and
  C067 source archive SHA
  `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`.
- Active dispatch claim recorded:
  `sjkl_c067_l40sdiag_20260502T151434Z`.
- Lightning L40S diagnostic Batch job submitted:
  `.omx/state/sjkl_c067_l40sdiag_20260502T151434Z_batch_jobs.json`.
  Initial SDK status: `Pending`, cost `$0.00`.
- Because the L40S parent remained pending at zero cost, a single bounded
  RTX PRO child hedge was claimed and queued:
  `sjkl_c067_rtxprodiag_20260502T151756Z`, state
  `.omx/state/sjkl_c067_rtxprodiag_20260502T151756Z_batch_jobs.json`.
  This is not an independent scientific lane; it is a wall-clock hedge on the
  same source and driver. Stop or close the loser after the first completed
  CUDA packet is harvested.

Verification:

- `bash -n scripts/remote_lane_sjkl_c067.sh`.
- `py_compile` on SJ-KL builder/packer/tensor-prep/tests.
- `36 passed` for SJ-KL remote-script, archive-packer, basis/runtime, and
  tensor-prep focused tests.
- `ruff check` passed for the new archive packer and new remote-script tests.
- `git diff --check` passed for touched SJ-KL and report files.

Current branch rule:

- If the L40S job produces exact JSON with stable components and useful score
  delta, queue a T4/equivalent promotion on the exact same archive bytes.
- If it collapses, preserve the exact JSON/logs/manifests as A-negative and use
  the component trace to tune SJ-KL basis dimension, anchor pairs, alpha bits,
  or target-slot/renderer coupling before any T4 spend.

2026-05-02T15:29Z public-floor refresh and negative local screens:

- Online/public floor refresh:
  - PR #67 `qpose14_qzs3_filmq9g_slsb1_r55` remains the relevant faithful
    rounded `0.31` public target. Its GitHub PR report records CUDA,
    `600` samples, PoseNet `0.00048597`, SegNet `0.00061000`, archive
    `276564` bytes, and a rounded final score `0.31`. The PR comments describe
    QZS3 grouped variable-bit-depth quantization, delta+VLQ pose encoding, and
    a single-blob payload.
  - PR #65 `henosis_qz_n3z_r25_clean` remains the relevant rounded `0.32`
    comparator. Its PR report records PoseNet `0.00035283`, SegNet
    `0.00070896`, archive `284425` bytes, rounded score `0.32`, and an
    approximate exact local score `0.31968005`.
  - C-067 remains our current internal exact A++ frontier at
    `0.31561703078448233`, so it is between PR #67 and PR #65 but not below
    the faithful rounded `0.31` public floor.

- Block-FP/QBF1 local sweep:
  - Tool: `experiments/build_blockfp_c067_archive.py`.
  - Output:
    `experiments/results/blockfp_c067_candidate_20260502T_qbf1_sweep/blockfp_c067_summary.json`.
  - Sweep block sizes: `16,24,32,48,64,96,128,256,512,1024`.
  - Best local archive bytes: `280588` at block size `32`, SHA
    `b578afb324261dabd5492083e9ea46b3b26c0ee99b2ae1298f360d58c2a36b8a`.
  - Delta vs C-067: `+4374` bytes, formula-only rate delta
    `+0.0029124670609563773`.
  - Branch decision: do not exact-eval QBF1-v1 on C-067. The blocker is byte
    economics, not runtime structure. Renderer self-compression remains open,
    but it needs a different trained/entropy-coded weight codec rather than
    this local QBF1 wrapper.

- Foveal/ego hard-pair PMG byte screens:
  - Active subspace outputs:
    `experiments/results/c067_cmg3_rowspan_escape_atoms_20260502/stride1_dynamic_ego_foveal_subspaces.json`
    and sibling stride/static summaries.
  - Stable high-density pair manifold begins with
    `69,290,67,285,289,70,286,164,292,294,293,284`.
  - Built byte-only PMG variants using this pair manifold with
    `residual_atom_count=512`:
    - `pairs20`: `197234` bytes, SHA
      `291d6fd521c5ea124dbc977381b1033d1aff65bdd2063d92c18cd0c7f73fb69a`,
      final mask disagreement `0.014881981743706597`.
    - `pairs40`: `213404` bytes, SHA
      `47d06aeaa1161a6df88371b462c5c3318931cf381353e435acd8d8b2ddfe6adb`,
      final mask disagreement `0.014497858683268229`.
    - `pairs64`: `230996` bytes, SHA
      `23db0ab66eea610b2255b748cf455cac59e22fa21483a8d05405da6783f99c93`,
      final mask disagreement `0.013985850016276042`.
  - Branch decision: do not dispatch these PMG row-span variants. The best
    exact diagnostic in this family, PMG atomtop4068, already collapsed at
    lower disagreement `0.012382837931315104`; these foveal-pair byte screens
    are dominated. The measured row-span/row-run PMG implementation is retired
    as a promotion path; mask work should move to learned topology,
    multimask reconciliation, analog soft-LUT decoders, or pose-conditioned
    residual semantics.

- Live SJ-KL status:
  - `sjkl_c067_l40sdiag_20260502T151434Z`: Running, cost observed about
    `$0.396`, remote heartbeat shows Stage 2 CUDA `build_sjkl_residual.py`
    with L40S GPU utilization up to `99%`.
  - `sjkl_c067_rtxprodiag_20260502T151756Z`: Running, cost observed about
    `$0.327`, remote heartbeat shows Stage 2 CUDA `build_sjkl_residual.py`
    with RTX PRO GPU utilization around `97-100%`.
  - Both jobs have persisted `remote_lane_sjkl_c067.log`,
    `heartbeat.log`, and `source_artifact_manifest.json`, but no
    `sjkl_manifest.json`, packed archive manifest, or `contest_auth_eval.json`
    yet. Continue polling; stop the lower-value duplicate after the first
    completed CUDA packet is harvested.

2026-05-02T15:55Z fixed-slice public-floor isolation repair and relaunch:

- Live public leaderboard refresh at `2026-05-02T15:58Z`:
  https://comma.ai/leaderboard currently lists visible lossy video compression
  leaders as `0.32 qpose14 #63`, `0.33 unified_brotli #64`, and
  `0.33 quantizr #55`. The earlier repo-captured PR67 artifact remains useful
  as reverse-engineering material, but it is not the currently visible #1
  leaderboard row in this live refresh. Dispatch decisions must not assume
  PR67 is the active visible leaderboard leader without rechecking.
- A critical fixed-slice custody bug was found before corrected GPU spend:
  `experiments/build_fixedslice_segment_mix_candidates.py` had trusted the
  runtime metadata member order when slicing raw public fixed-slice payloads.
  The public PR67/QZS3/QP1 wire order is `masks.mkv`, `renderer.bin`, then
  `optimized_poses.bin`, even when a runtime wrapper reports logical members in
  another order. The first four `20260502T1539Z` diagnostic jobs were stopped
  while still zero-cost/pending and are classified
  `cancelled_invalid_archive_builder_raw_order_bug`; they are not score
  evidence.
- Permanent fixes landed:
  - `experiments/build_fixedslice_segment_mix_candidates.py` slices by the raw
    fixed-slice wire contract, validates segment names, duplicate metadata,
    raw SHA-256, decoded SHA-256, and total payload consumption.
  - `experiments/archive_bit_budget_profiler.py` now requires all public
    fixed-slice slices to brotli-decompress and pass decoded magic checks
    before reporting public PR63/PR67 anatomy. It no longer accepts a partial
    split or a brittle total-length band as anatomy truth.
  - `AGENTS.md` now records this as a durable contest-compliance rule.
- Focused verification:
  - `py_compile` on the fixed-slice builder, archive profiler, and tests.
  - `10 passed` for
    `src/tac/tests/test_build_fixedslice_segment_mix_candidates.py` plus
    `src/tac/tests/test_archive_bit_budget_profiler.py`.
  - Additional profiler-only rerun: `7 passed`.
- Corrected deterministic candidate output:
  `experiments/results/fixedslice_segment_mix_c067_pr67_20260502T1545Z/`.
  Corrected anatomy:
  - C067 and PR67 share the exact same charged mask segment:
    `219472` bytes, SHA
    `d1ae0d39c848e5715b74bb0122269066a4a1ab60ba9e12f34e70fd62ac136d87`.
  - C067 renderer: `55965` bytes, SHA
    `2544dfe50d2631a56600cbfc7342e4a18c97f953526fc51425be0d62f97e22ad`.
  - PR67 renderer: `56093` bytes, SHA
    `8d8863c32a264f28f199898a689b4d49ce96ebd12831164882b6babec93cbb9b`.
  - C067 pose: `677` bytes, SHA
    `c998610e82bb46a686a073e8a1847987de1346f86852a1c60cfa3244a1ad43c5`.
  - PR67 pose: `899` bytes, SHA
    `83767bbd10ae72c3237e468351eb9c465a954e25a6af04a0e0cb84d1f7af9b51`.
- Corrected byte candidates:
  - `pr67mask_c067renderer_c067pose`: `276214` bytes, archive SHA
    `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`.
    This is byte-identical to C067 and is a custody check only; no eval.
  - `c067mask_pr67renderer_c067pose`: `276342` bytes, archive SHA
    `0f0b1f84f5c263f5fe8f7d79eba160afa5f8e6f37424273c409bdc560066dc4d`.
  - `c067mask_c067renderer_pr67pose`: `276436` bytes, archive SHA
    `434ed7c8e084854a53a96ab8e64b6cb3f59535a3839f1485e95f4df7fe6799ab`.
  - `c067mask_pr67renderer_pr67pose`: `276564` bytes, archive SHA
    `4f66052846429aa0048e0de6743513bcfb3ce46bcf8504977d920741eb8bbdc8`.
- Archive bit-budget profile:
  `experiments/results/archive_bit_budget_profile_c067_pr67_mix_20260502T1548Z/`.
  The profiler reports all corrected candidates as
  `public_pr67_qzs3_qp1_fixed_slices` with decoded guesses
  `brotli_av1_obu_mask_stream`, `brotli_qzs3_renderer`, and
  `brotli_qp1_pose`.
- Lightning staging:
  `.omx/state/fixedslice_segment_mix_c067_pr67_fix1_20260502T1552Z_manifest.json`,
  remote-verified `1299` files and `22859195` bytes.
- Corrected exact-CUDA diagnostic jobs queued with active claims:
  - `exact_eval_fixedslice_pr67renderer_c067rest_fix1_l40s_20260502T1553Z`
    on L40S, expected archive SHA
    `0f0b1f84f5c263f5fe8f7d79eba160afa5f8e6f37424273c409bdc560066dc4d`.
  - `exact_eval_fixedslice_pr67pose_c067rest_fix1_l40s_20260502T1553Z`
    on L40S, expected archive SHA
    `434ed7c8e084854a53a96ab8e64b6cb3f59535a3839f1485e95f4df7fe6799ab`.
  - `exact_eval_fixedslice_pr67renderer_pose_c067mask_fix1_rtxpro_20260502T1553Z`
    on RTX PRO, expected archive SHA
    `4f66052846429aa0048e0de6743513bcfb3ce46bcf8504977d920741eb8bbdc8`.
  Current refreshed status at submission time: all three `Pending`, zero cost.
- Because the combined renderer+pose archive is the only whole-stream
  fixed-slice candidate with direct public-floor relevance, a T4-equivalent
  confirmation hedge was also queued:
  `exact_eval_fixedslice_pr67renderer_pose_c067mask_fix1_t4_20260502T1558Z`,
  state
  `.omx/state/fixedslice_segment_mix_c067_pr67_fix1_t4_batch_jobs_20260502T1558Z.json`.
  The first g4dn submit correctly failed closed until the inflate-side Torch
  runtime was pinned for the older T4 driver. The submitted job records
  `INFLATE_TORCH_SPEC=torch==2.5.1+cu124`,
  `UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124`, and
  `UV_INDEX_STRATEGY=unsafe-best-match`. Initial status: `Pending`, zero cost.
- SJ-KL RTX PRO hedge status:
  `sjkl_c067_rtxprodiag_20260502T151756Z` failed after building a charged
  `sjkl.bin` and deterministic archive, then failing Stage 0 NVDEC preflight
  before scoring. Built archive bytes: `315630`, SHA
  `7efc496e7bb995e7ce2b8419307edddcb4bb20a83a42db4ae459e47bb16d74de`.
  `sjkl.bin` bytes: `38515`, SHA
  `04e95e588d6ec874809994856c56a5940e7b8e93e1d9e531056ff1505fb94e9f`.
  Classification: infrastructure/NVDEC failure, not scorer evidence. The L40S
  SJ-KL diagnostic remains the live scientific path.

Branch rule:

- If any fixed-slice diagnostic improves components enough to overcome its
  added bytes, use its component trace to update the C067/PR67 pose-renderer
  subspace and promote only on T4/equivalent identical bytes.
- If fixed-slice diagnostics are neutral or worse, the public floor edge is
  not a whole-stream renderer/pose transfer; shift to pair/local atom transfer
  over the same decoded traces.
- If SJ-KL L40S lands positive, stop treating it as a diagnostic and queue a
  T4/equivalent promotion on the exact emitted archive bytes. If it collapses,
  use the component trace to change basis dimension, anchor set, and coefficient
  budget rather than retiring SJ-KL broadly.

## 2026-05-02T16:30Z - SJ-KL L40S harvest and harness hardening

SJ-KL C067 L40S diagnostic completed and was harvested locally from the
state-derived Lightning artifact path:

- Job: `sjkl_c067_l40sdiag_20260502T151434Z`
- Local mirror:
  `experiments/results/lightning_batch/sjkl_c067_l40sdiag_20260502T151434Z/`
- Archive bytes: `315515`
- Archive SHA-256:
  `2e12b9adc552a00b7c956ee2bddae139968b2fe3036bcb4aab8bb10f236ea066`
- Charged `sjkl.bin`: `38505` bytes, SHA-256
  `bfd90839190dc52bf9db4ae6bd0cfa26b50114c4a6ec6c3451657a4e8b2aa59a`
- CUDA hardware: L40S, diagnostic only, `gpu_t4_match=false`
- Component result: PoseNet `0.00527809`, SegNet `0.00061036`
- Recomputed score: `0.5008654410618838`

Classification: A-negative diagnostic. The payload was loaded and applied to
the JointFrameGenerator pair path, but PoseNet regressed sharply against C067.
This is not a T4-promotion candidate. It does not broadly kill SJ-KL; it
retunes the next SJ-KL design toward pose-local atoms, signed trust regions,
smaller basis magnitude, and component-response-ranked anchor pairs rather
than global full-frame fake0 correction.

Permanent hardening landed in this slice:

- `sjkl.bin` exact-eval dispatches now default to `SJKL_REQUIRE_APPLIED=1` in
  `scripts/remote_lane_sjkl_c067.sh`; charged SJ-KL payloads that never affect
  a renderer pair fail as harness failures instead of becoming score evidence.
- `submissions/robust_current/inflate_renderer.py` records SJ-KL applied/skip
  counts and fails the strict contract when requested.
- `submissions/robust_current/inflate_renderer.py` no longer falls back from
  `INFLATE_MASK_SOURCE=archive` to SegNet extraction when archive masks are
  missing. Missing charged masks are now a packaging failure; development
  fallback requires explicit `INFLATE_MASK_SOURCE=segnet`.
- `submissions/robust_current/unpack_renderer_payload.py` rejects ambiguous
  packed payload containers instead of selecting by precedence.
- `submissions/robust_current/inflate.sh` writes
  `renderer_payload_unpack_summary.json`, and `experiments/contest_auth_eval.py`
  attaches that unpack summary to exact-eval provenance.
- `scripts/launch_lightning_batch_job.py` and
  `src/tac/deploy/lightning/batch_jobs.py` now pass `--cloud-account` through
  exact/component/sensitivity submissions so fast-chip routing does not silently
  resolve against the wrong provider account.
- `experiments/contest_auth_eval.py` now accepts
  `--expected-runtime-tree-sha256`, and Lightning exact-eval commands can pass
  it via queue metadata key `expected_runtime_tree_sha256`; this makes runtime
  drift a fail-closed custody condition for promotion-sensitive runs.

Focused verification:

- `py_compile` on touched runtime, launcher, and evaluator files passed.
- Focused tests: `16 passed in 1.77s`.

Open live eval:

- `exact_eval_fixedslice_pr67renderer_c067rest_fix2_t4_20260502T1623Z`
  is now `Running` on T4. This is the repair of the earlier renderer-only
  fixed-slice parser-boundary failure and remains the next harvest target.

## 2026-05-02T16:55Z - Fixed-slice T4 harvest and IMP-10 status

The fixed-slice PR67-renderer-only transfer finished and was harvested from the
state-derived Lightning artifact path:

- Job: `exact_eval_fixedslice_pr67renderer_c067rest_fix2_t4_20260502T1623Z`
- Local mirror:
  `experiments/results/lightning_batch/exact_eval_fixedslice_pr67renderer_c067rest_fix2_t4_20260502T1623Z/`
- Archive bytes: `276342`
- Archive SHA-256:
  `0f0b1f84f5c263f5fe8f7d79eba160afa5f8e6f37424273c409bdc560066dc4d`
- CUDA hardware: Tesla T4, `gpu_t4_match=true`
- Component result: PoseNet `0.00053798`, SegNet `0.00061248`
- Recomputed score: `0.31859986991619027`
- Delta versus C067 T4 frontier: `+0.0029828391317079372`

Classification: A++ T4 exact evidence, no-frontier. The component gates passed,
but the PR67 renderer-only fixed-slice transfer worsened PoseNet enough to lose
against C067. This result narrows the public-floor transfer hypothesis: the
advantage is not a whole renderer stream replacement on the C067 rest-of-archive.
The next useful transfer work is pair-local/atom-local packing, pose, and
payload-container anatomy, not another whole-stream renderer swap.

IMP-10 latest status:

- IMP was considered recently in the phase-14 council plan as Lane 17, with a
  10-cycle iterative magnitude-pruning mechanism and cycle-1 overnight dispatch
  proposal. That plan targeted the older Lane G v3 score band, not the current
  C067 public-floor anchor.
- `scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh` has a hardened
  design: 10 cycles, 20% pruning per cycle, revert-on-regression auth-eval
  checkpoints, real fine-tune stage after the pruning stub, NVDEC/preflight
  guards, and export to a contest archive.
- The actually harvested IMP attempt is not usable score evidence. It failed
  around cycle 0 on a checkpoint/model shape mismatch and earlier loop evidence
  was contaminated by a stub/synthetic-training path. There is no current
  contest-faithful completed 10-cycle IMP score.

Current decision: IMP remains plausible only as a renderer-compression or
sparsity-oracle lane on the current C067/JointFrameGenerator/QZS3 anchor. The
old rate forecast of about `-0.10` was based on shrinking a much larger Lane G
renderer; C067 already has a much smaller public-floor-style renderer payload,
so the C067-era rate-only upside is closer to a meaningful but smaller
`~0.02-0.035` if distortion remains flat. That is still leaderboard-relevant
near `0.3156`, but it should run behind or alongside Block-FP/self-compression,
not displace higher-EV full-stack lanes.

Recommended IMP relaunch shape:

- Create an `IMP-C067-QZS3-bridge` lane instead of rerunning the old Lane G
  script as-is.
- Anchor on the exact C067 JointFrameGenerator/runtime contract and current
  packed renderer format.
- Run a 2-cycle H100 smoke with sensitivity-weighted pruning, deterministic
  export, byte screen, and exact CUDA diagnostic on the emitted archive.
- Launch the full 10-cycle H100 burn only if cycle 1/2 preserves PoseNet/SegNet
  and demonstrates material byte reduction under the current C067 packer.
- If the score path regresses, keep IMP as a learned sparsity mask or pruning
  prior for Block-FP/self-compression rather than treating it as a standalone
  promotion candidate.

## 2026-05-02T17:05Z - C067-anchored IMP bridge byte-screen and L40S diagnostic wave

The old IMP lane was not reused as-is because it targeted the pre-C067 Lane G
renderer family. A new C067/QZS3/JointFrameGenerator bridge builder was added
and run locally as empirical byte-screen evidence only:

- Builder: `experiments/build_imp_c067_bridge_candidates.py`
- Source archive: C067 fixed-slice archive, `276214` bytes, SHA-256
  `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`
- Output root:
  `experiments/results/imp_c067_bridge_candidates_20260502/`
- Source renderer format: QZS3, source renderer bytes `59288`
- Prunable tensor count: `42`; prunable scalar count: `76216`
- Evidence grade: empirical byte-screen only. `score_claim=false`,
  `promotion_eligible=false`.

Best byte-screen points, all QZS3 block size `128`:

| candidate | sparsity | archive bytes | delta bytes vs C067 | formula-only rate delta | SHA-256 |
|---|---:|---:|---:|---:|---|
| `imp_c01_qzs3_b0128` | `0.20008922011126273` | `271746` | `-4468` | `-0.0029750578025498618` | `1a1cf6fc797ba176cfb38b60778049403eb9cd35993f7f260be2647c05f04b4f` |
| `imp_c02_qzs3_b0128` | `0.36021307861866275` | `267153` | `-9061` | `-0.006033347974239995` | `c5b886fa0c39a399a2ccd2f7ce213145f0034aa48bafad5f9eb76580e4c1ac7f` |
| `imp_c05_qzs3_b0128` | `0.672562191665792` | `256193` | `-20021` | `-0.013331162100458992` | `87d65c7e3c279ef5ef5830240ba9de6bfe43d5f8b25d29147f4bb01ea9c60d7f` |
| `imp_c10_qzs3_b0128` | `0.8927915398341556` | `244623` | `-31591` | `-0.021035150188082514` | `4910824c0464fef1d7aa355bf0b82de0a358539cbc2aab68217230f4f6d6f38a` |

Interpretation: the cycle-10 byte win is large enough to cross the unchanged-
distortion sub-0.3 threshold, but it is an untrained no-score pruning bridge.
It cannot promote, rank, or update the frontier until exact CUDA evaluation of
the exact archive bytes lands. Its value is that it makes renderer sparsity a
measured C067-era variable instead of an old Lane G forecast.

Lightning staging and dispatch:

- Staged source/artifact manifest:
  `.omx/state/imp_c067_bridge_exactdiag_20260502T1700Z_manifest.json`
- Manifest verified remotely: `1351` files, `29711553` bytes.
- Strict Lightning doctor passed before dispatch:
  `.omx/state/lightning_doctor_imp_c067_20260502T1700Z.json`
- L40S diagnostic exact-eval jobs queued with adjudication and component trace:
  - `exact_eval_imp_c067_imp_c01_qzs3_b0128_l40sdiag_20260502T1700Z`
  - `exact_eval_imp_c067_imp_c02_qzs3_b0128_l40sdiag_20260502T1700Z`
  - `exact_eval_imp_c067_imp_c05_qzs3_b0128_l40sdiag_20260502T1700Z`
  - `exact_eval_imp_c067_imp_c10_qzs3_b0128_l40sdiag_20260502T1700Z`
- State file:
  `.omx/state/imp_c067_bridge_exactdiag_l40s_batch_jobs_20260502T1700Z.json`

Branch rule:

- If any IMP bridge candidate preserves PoseNet/SegNet enough to beat C067 on
  L40S, immediately rerun the identical archive bytes on T4/equivalent and use
  its component trace to choose the next pruning/training mask.
- If low cycles preserve but high cycles collapse, run a trained H100
  sensitivity-weighted IMP continuation around the highest safe sparsity band.
- If all no-train cycles collapse, do not broadly retire IMP. Treat the traces
  as evidence that untrained global magnitude pruning is too blunt, then reuse
  the byte/compressibility signal as a learned sparsity prior for Block-FP,
  Q-FAITHFUL successor, or hotspot-preserving mask geometry.
- T4 remains promotion-only; L40S results are diagnostic unless explicitly
  superseded by T4/equivalent identical-byte confirmation.

## 2026-05-02T17:10Z - C067 QZS3 block-size self-compression diagnostic

The renderer self-compression v2 planner found a deterministic low-risk
reblock candidate inside the current C067/JointFrameGenerator/QZS3 payload.
This is empirical byte-screen evidence until exact CUDA lands.

- Planner:
  `experiments/plan_c067_renderer_self_compression_v2.py`
- Planner output:
  `experiments/results/c067_renderer_self_compression_v2_20260502/plan.json`
- Deterministic archive builder:
  `experiments/build_mixed_qzs_block_candidate.py`
- Candidate archive:
  `experiments/results/c067_qzs3_b512_candidate_20260502/global_b512/archive.zip`
- Source archive: C067 fixed-slice archive, `276214` bytes, SHA-256
  `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`
- Candidate archive bytes: `271886`
- Candidate SHA-256:
  `4271b2c855fddc089ae590392caec1d92cf408228664c4a9a0249d3b375e9d43`
- Byte delta versus C067: `-4328`
- Evidence grade before CUDA: empirical byte-screen only. `score_claim=false`,
  `promotion_eligible=false`.

Lightning staging and dispatch:

- Staged source/artifact manifest:
  `.omx/state/c067_qzs3_b512_exactdiag_20260502T1710Z_manifest.json`
- Manifest verified remotely: `1300` files, `22444692` bytes.
- L40S diagnostic exact-eval job:
  `exact_eval_c067_qzs3_b512_l40sdiag_20260502T1710Z`
- State file:
  `.omx/state/c067_qzs3_b512_l40sdiag_batch_jobs_20260502T1710Z.json`
- Latest refreshed status at `2026-05-02T17:06:51Z`: `Pending`, zero cost,
  identity by name only, so artifact validation remains required before any
  scientific use.

Branch rule:

- If qzs3-b512 preserves PoseNet/SegNet and beats C067 on L40S, immediately
  queue the identical archive bytes on T4/equivalent.
- If it is neutral or slightly negative but component-safe, use it as a
  composable reblock atom only when a larger mask/renderer stack needs the
  byte margin.
- If it collapses, classify it as a C067 QZS3 block-size distortion failure,
  not as a broad renderer self-compression kill.

## 2026-05-02T17:20Z - Hardening and fast diagnostic hedge while CUDA runs

Active exact-eval orchestration:

- IMP cycle 1, 2, and 5 L40S diagnostic jobs refreshed to `Running`.
- IMP cycle 10 L40S diagnostic and qzs3-b512 L40S diagnostic remained
  `Pending`.
- Because cycle 10 is the highest-upside byte-screen point, one additional
  A100 diagnostic hedge was queued on identical cycle-10 archive bytes:
  `exact_eval_imp_c067_imp_c10_qzs3_b0128_a100diag_fix1_20260502T1720Z`.
  The first A100 submit attempt failed before GPU spend because the Lightning
  SDK requires an explicit `--user` or `--org` when `--teamspace` is supplied;
  the fix1 launch passed `--user adpena` and queued successfully.

Bug-class hardening:

- `experiments/contest_auth_eval.py` now includes a static repo-local `src/tac`
  import closure in the inflate/runtime manifest and folds those hashes into
  `runtime_tree_sha256`. This prevents identical archive bytes from being
  compared as pure archive evidence when runtime helper code changed.
- `scripts/launch_lightning_batch_job.py` now fails closed before remote
  preflight when a non-dry-run Studio submit specifies `--teamspace` without
  `--user` or `--org`. This prevents a late SDK traceback after supply-chain
  work has already consumed wall-clock.
- The same launcher now treats all `completed_*` dispatch-claim statuses as
  terminal, matching `tools/claim_lane_dispatch.py`. This prevents
  `completed_empirical_no_score` and `completed_a_negative_*` rows from being
  misread as still-active claims.
- Deterministic ZIP hardening landed in the fallback archive builders:
  central deterministic ZIP helpers in `src/tac/submission_archive.py`,
  `scripts/compress_archive.py` no longer uses `ZipFile.write()` over
  `os.walk`, the `compress.sh` fallback inline Python routes through the
  central builder, and preflight now scans these paths. Hidden/system ZIP
  members and symlinks fail closed instead of silently entering archives.
- Focused verification passed:
  `23 passed` for the new Lightning identity preflight plus contest auth-eval
  custody tests, plus `2 passed` for dispatch-claim terminal/identity
  regressions, plus `15 passed` for deterministic ZIP/archive-builder
  regressions; `py_compile` passed for touched Python; `bash -n` passed for
  `compress.sh`; `git diff --check` passed on the touched files.

Branch rule:

- If the A100 cycle-10 diagnostic lands first and survives component gates,
  immediately submit the identical `244623` byte archive on T4/equivalent.
- If A100 cycle 10 collapses catastrophically, stop waiting for the L40S
  duplicate cycle-10 unless lower IMP cycles show a safe band that needs
  interpolation.

External design intake while waiting on CUDA:

- The user's "Telescope foveation" reference maps to arXiv:2604.06332,
  "Telescope: Learnable Hyperbolic Foveation for Ultra-Long-Range Object
  Detection" by Ewen/Rivkin/Bijelic/Heide, submitted 2026-04-07.
  Source: https://arxiv.org/abs/2604.06332
- Relevant external idea: learnable hyperbolic/foveated resampling for
  autonomous-driving long-range signal allocation. Contest use is strictly
  design motivation: treat foveation centers, anisotropic radii, ego-motion
  fields, horizon/road priors, and low-dimensional subspace coordinates as
  planning atoms with byte/scorer break-even records.
- Evidence grade: external. No archive score, promotion, or paper claim may
  rely on this paper without a contest-closed archive and exact CUDA eval.

Public leaderboard refresh:

- Source: https://comma.ai/leaderboard
- Current displayed lossy video compression top entries at refresh time:
  `0.32 qpose14` (PR #63), `0.33 unified_brotli` (PR #64), `0.33 quantizr`
  (PR #55), `0.37 fp4_mask_gen` (PR #62), `0.38 selfcomp` (PR #56).
- These are rounded public-display values. Our current internal C067 A++
  anchor remains `0.31561703078448233`, which is below displayed `0.32`
  rounding, but this ledger does not claim public leaderboard placement until
  the candidate is submitted and accepted.

## 2026-05-02T17:45Z - IMP raw bridge closed, hotspot/multires stack moves next

Exact CUDA diagnostics harvested:

- `imp_c01_qzs3_b0128` on L40S: archive `271746` bytes, SHA
  `1a1cf6fc797ba176cfb38b60778049403eb9cd35993f7f260be2647c05f04b4f`,
  score `1.355401799409314`, PoseNet `0.11587474`, SegNet `0.00098006`.
  Evidence grade: A-negative scoped forensic.
- `imp_c02_qzs3_b0128` on L40S: archive `267153` bytes, SHA
  `c5b886fa0c39a399a2ccd2f7ce213145f0034aa48bafad5f9eb76580e4c1ac7f`,
  score `5.528075689026167`, PoseNet `2.68607259`, SegNet `0.00167456`.
  Evidence grade: A-negative scoped forensic.
- `imp_c05_qzs3_b0128` on L40S: archive `256193` bytes, SHA
  `87d65c7e3c279ef5ef5830240ba9de6bfe43d5f8b25d29147f4bb01ea9c60d7f`,
  score `36.79093785806109`, PoseNet `77.92242432`, SegNet `0.08705761`.
  Evidence grade: A-negative scoped forensic.
- `imp_c10_qzs3_b0128` on L40S: archive `244623` bytes, SHA
  `4910824c0464fef1d7aa355bf0b82de0a358539cbc2aab68217230f4f6d6f38a`,
  score `78.9903710766749`, PoseNet `79.82865143`, SegNet `0.50573522`.
  Evidence grade: A-negative scoped forensic.

Interpretation:

- Raw no-train IMP pruning is not a current score path. It has useful byte
  headroom but destroys renderer/scorer geometry before even moderate pruning.
  Reactivation requires learned, QAT, Fisher/Hessian, or component-response
  weighted pruning; byte-only magnitude cycles are measured-implementation
  retired.
- The duplicate A100 cycle-10 hedge was stopped while pending at zero cost
  after the L40S cycle-10 collapse landed.

Additional exact diagnostic:

- `global_b512` QZS3 block-size self-compression on L40S: archive `271886`
  bytes, SHA `4271b2c855fddc089ae590392caec1d92cf408228664c4a9a0249d3b375e9d43`,
  score `2.2397462747539274`, PoseNet `0.3804819`, SegNet `0.00108114`.
  Evidence grade: A-negative scoped forensic.
- Naive global QZS3 reblocking is not a stack atom against C067. Future
  self-compression must be block-local, learned/QAT, or sensitivity-weighted.

Hotspot geometry compiler:

- A class-contract bug surfaced before dispatch: the hotspot planner emitted
  class `0` atoms, but `build_cmg3_adaptive_runs_candidate.py` consumes
  nonzero CMG3A row-run classes only. This is now a permanent planner-side
  guard; class `0` is rejected with
  `builder_incompatible_class_id_for_cmg3a_nonzero_row_run`.
- Regenerated policy
  `experiments/results/c067_hotspot_mask_geometry_compiler_20260502/c067_hotspot_mask_geometry_plan.json`
  selects `128` class-2 hotspot atoms and rejects `196` class-0 atoms.
- Built C067-anchored candidate
  `experiments/results/c067_hotspot_mask_geometry_compiler_20260502/candidate_top0128_c067_anchor/archive.zip`,
  archive `129857` bytes, SHA
  `e8bd5b202efd9bc4f60f83a2a14879449ec2b4d9b241ed8a7cfe682dd567eb0b`.
  Formula-only rate delta vs C067 is `-0.09745311880210163`; pixel
  disagreement is `0.03538155449761285`.
- Lightning L40S diagnostic submitted:
  `exact_eval_c067_hotspot_geometry_top0128_l40sdiag_20260502T1733Z`.
  This is high-risk diagnostic evidence only; if it survives component gates
  and beats C067, the identical bytes must be replayed on T4/equivalent.

Multi-resolution / multi-pass planner:

- Kepler produced the bounded planning-only apogee/C067 stack planner:
  `experiments/plan_c067_multiresolution_stack_candidates.py` and
  `experiments/results/c067_multiresolution_stack_planner_20260502/c067_multiresolution_stack_plan.json`.
- Policies are explicitly `score_claim=false`, typed by resolution layer, and
  ordered as pass 0 anchor, pass 1 coarse/global, pass 2 high-res repair,
  pass 3 entropy/packer, optional pass 4 pose/runtime co-adaptation.
- The planner intentionally emits no builder commands because no current
  repo builder consumes the full multi-pass stack byte-closed. That is a
  correct blocker, not an implementation failure.

Public supplement / site hygiene:

- Public name preference recorded as `apogee`; use lower-case for public
  artifacts and prefixes such as `apogee_c067_*` or `apogee_yf_*` internally.
- Lightning.ai notebooks and Cloudflare Pages are acceptable public supplement
  hosts, but private provider state stays out of GitHub/site artifacts.
- `AGENTS.md` now has a durable public-release hygiene protocol, `.gitignore`
  excludes future local `.omx/state/*.json`, raw/private reports, and `.env*`
  surfaces, and `check_public_release_hygiene()` scans intended public docs
  for local absolute paths, private Studio links, concrete Vast SSH endpoints,
  and API/private-key patterns. Full repo preflight wires the guard warn-only
  because legacy custody ledgers intentionally preserve private local evidence;
  release tooling should call it strict over the explicit publish surface.

Greenup and observability:

- Focused verification after the public-release/preflight patch:
  `.venv/bin/python -m py_compile ...` passed for touched Python, focused
  pytest passed `12 passed in 0.53s`, and `git diff --check` passed on the
  touched files.
- Strict public-release scan over `AGENTS.md`, `reports/writeup_working.md`,
  and `reports/latest.md` found `0` violations. This is a publish-surface
  hygiene signal only, not a claim about legacy raw custody ledgers.
- Byte-accounting refreshed for C067:
  `experiments/results/c067_archive_byte_accounting_20260502/archive_byte_accounting_refreshed.{json,md,png}`.
  The unchanged-distortion sub-0.300 gap is `23454` bytes, or `23455` bytes
  with a one-byte buffer.
- Byte-accounting added for the hotspot candidate:
  `experiments/results/c067_hotspot_mask_geometry_compiler_20260502/candidate_top0128_c067_anchor/archive_byte_accounting.{json,md,png}`.
  The candidate is `129857` bytes with `masks.cmg3` as the largest stream and
  `renderer.bin` second. This is observability evidence only; it remains
  non-promotable until exact CUDA auth eval lands.

Exact hotspot result:

- `exact_eval_c067_hotspot_geometry_top0128_l40sdiag_20260502T1733Z` harvested
  through the state-derived Lightning path after a `Running -> Pending`
  telemetry regression. Terminal artifacts were present and validated from the
  SDK artifact mirror.
- Archive `129857` bytes, SHA
  `e8bd5b202efd9bc4f60f83a2a14879449ec2b4d9b241ed8a7cfe682dd567eb0b`,
  L40S CUDA score `28.98169780735597`, PoseNet `39.20018387`, SegNet
  `0.09096195`, sample count `600`.
- Evidence grade: A-negative scoped forensic. Promotion eligible: false.
  Component trace cross-check matched `contest_auth_eval.json`.
- Branch decision: retire this measured standalone CMG3A hotspot grammar
  implementation as a score path. It proves byte headroom exists, but the
  scorer geometry cliff is catastrophic. Next mask work must use a learned or
  multimask/reconciler trust-region selector with exact byte-closed archive
  composition, not another large standalone grammar replacement.

Apogee multi-pass bridge:

- Bernoulli implemented
  `experiments/build_c067_multiresolution_stack_candidate.py` plus focused
  tests. The tool consumes the C067/apogee multi-resolution planner and emits
  deterministic, argparse-validated standalone builder commands while marking
  full-stack composition blocked.
- Output manifest:
  `experiments/results/c067_multiresolution_stack_builder_20260502/c067_multiresolution_stack_build_manifest.json`.
- It correctly emits no archive, launches no jobs, keeps `score_claim=false`,
  and `--require-byte-closed-stack` fails closed until a real byte-closed stack
  composer exists for overlapping mask/repair/packer/pose passes.

## Update - 2026-05-02T18:20Z Apogee Public Supplement And Naming Hardening

Online control-plane refresh:

- Upstream contest README/leaderboard and PR template were refreshed from
  public sources. The public PR must provide submission name, `archive.zip`
  link, `report.txt`, GPU-required answer, compression-script answer, and
  comments. The current visible leaderboard top row is rounded `0.32`
  (`qpose14` / PR #63); Apogee/C067 is still an internal exact A++ result until
  public submission.
- Lightning Studios remain appropriate for an interactive notebook/app
  supplement, but public links must be sanitized and intentionally published.
  Do not copy private Studio URLs or provider job pages into GitHub, the
  report, notebook, or Cloudflare bundle.
- Cloudflare Pages Direct Upload is the preferred static supplement host.
  Wrangler deploys the prebuilt `reports/graphs/site/` folder; current Pages
  single-asset limit is 25 MiB, so large videos/figures must be compressed,
  split, or hosted as separate release assets.

Implementation landed:

- `docs/submission_template.md` now mirrors the upstream PR fields and uses
  `Apogee` plus public placeholders:
  `${APOGEE_ARCHIVE_ZIP_URL}`, `${LIGHTNING_SUPPLEMENT_URL}`,
  `${CLOUDFLARE_PAGES_URL}`, `${APOGEE_RELEASE_MANIFEST}`.
- `docs/runbooks/apogee_public_supplement_20260502.md` records the supplement
  plan, notebook sections, Cloudflare constraints, and strict publish gate.
- `.omx/research/apogee_public_supplement_plan_20260502_codex.md` records the
  research refresh and naming decisions.
- `reports/graphs/deploy_cloudflare_pages.md` now uses project
  `apogee-comma-video`, documents Direct Upload constraints, and includes the
  strict public-release hygiene command.
- `reports/graphs/build_public_site_bundle.py` now creates the deployable
  `reports/graphs/public_site/` bundle by redacting local paths, private
  Lightning URLs, Vast endpoints, and token-like strings from the internal
  generated site, then enforcing the Cloudflare asset-size and strict hygiene
  gates.
- `notebooks/apogee_lightning_supplement.ipynb` now exists as a sanitized
  no-output Lightning notebook skeleton that recomputes score terms, plots byte
  anatomy, and states the charged-atom interpretation.
- `src/tac/preflight.py` now includes `notebooks` in the default public-release
  scan surface so future notebook leakage is caught before publication.
- Sanitized public bundle build result: `reports/graphs/public_site/`, `74`
  files, `474` redactions, `0` strict hygiene violations, with
  `comparison/comparison.gif` omitted because it is `114724141` bytes and
  exceeds the Cloudflare Pages current single-asset limit.

Allowed use:

- These are report/submission readiness and public-facing reproducibility
  surfaces. They do not change the current score frontier and do not promote
  any candidate. They make the final Apogee PR faster and safer once the last
  optimization wave lands.

Live optimization state:

- Line-search C067 L40S diagnostic
  `line_search_c067_basis_hotpairs_l40sdiag_20260502T1803Z` failed before
  producing lane payload/eval artifacts. Only `run.log` exists, ending during
  `bootstrap_runtime_deps`; cost recorded by Lightning SDK was `$0.3179`.
  Failure classification:
  `experiments/results/lightning_batch/line_search_c067_basis_hotpairs_l40sdiag_20260502T1803Z/failure_classification.json`.
  Allowed use: infrastructure failure forensics only; no score, no line-search
  method evidence, no promotion.
- Guarded SJ-KL v2 diagnostic submitted on Lightning L40S as
  `sjkl_c067_v2_k4_a5_cap32k_l40s_20260502T181718Z`, state
  `.omx/state/sjkl_c067_v2_k4_a5_cap32k_l40s_20260502T181718Z_batch_jobs.json`.
  Source manifest
  `.omx/state/sjkl_c067_v2_stage_20260502T181621Z_manifest.json` verified
  `1307` files and `2853671851` bytes on the remote Studio. This run is capped
  at `SJKL_MAX_BYTES=32768`, requires `SJKL_REQUIRE_APPLIED=1`, and remains
  L40S diagnostic-only until a material archive is replayed on T4/equivalent.

## Update - 2026-05-02T18:36Z Optimization Focus And Q-FAITHFUL Pose-Contract Hardening

The active priority is score movement, not public-supplement polish. Current
score-reduction pressure remains on guarded SJ-KL v2 while infrastructure bugs
that would invalidate high-EV successor lanes are made fail-closed.

Live optimization:

- `sjkl_c067_v2_k4_a5_cap32k_l40s_20260502T181718Z` is running on Lightning
  L40S with GPU at 100%. Remote artifacts currently show tensor preparation
  complete and CUDA basis construction in progress:
  `computing SJ-KL basis: k=4, n_anchors=20, grid=32x24`.
- No archive, `contest_auth_eval.json`, or score evidence has landed from this
  v2 run yet. It remains diagnostic-only until a byte-closed archive is exact
  CUDA evaluated and, if material, replayed on T4/equivalent.

Worker conclusions integrated:

- Block-FP/self-compression local screen produced no immediate dispatchable
  score candidate. The byte-winning global-QZS3 branch is already scoped
  exact-negative in the same family, and the safer component-aware branch is
  byte-regressive versus C067 before scorer effects.
- Q-FAITHFUL old snapshots remain exact-CUDA negative, dominated by PoseNet
  collapse. A real bug class was identified and fixed fail-closed locally:
  `variant=quantizr_faithful` must no longer train with silent zero-pose
  fallback.

Hardening landed:

- `src/tac/experiments/train_renderer.py` now requires
  `--qfaithful-training-poses` for `variant=quantizr_faithful`, loads the
  deployed pose stream through `tac.submission_archive.load_optimized_poses`,
  validates exact pair count, pose dimension, nonzero content, and SHA-256,
  passes the per-pair pose tensor into Q-FAITHFUL forward/eval, disables
  unaudited horizontal flip augmentation for pose-conditioned Q-FAITHFUL
  training, and stores a promotable `training_pose_contract` in checkpoints.
- `scripts/remote_lane_q_faithful_jointgen.sh` now builds the half-frame mask
  archive explicitly, passes the deployed pose stream to training, exports raw
  QFAI as `renderer.bin` so inflate can dispatch by file magic, writes a
  brotli sidecar only for byte research, and refuses export if the checkpoint
  lacks a promotable pose contract.
- `AGENTS.md` now records the durable Q-FAITHFUL pose-contract and raw-QFAI
  deployment rule so future retraining does not reintroduce the zero-pose or
  compressed-under-raw-member bug class.

Verification:

- `py_compile`: `train_renderer.py`, Q-FAITHFUL snapshot loop, and focused
  tests passed.
- `bash -n`: Q-FAITHFUL and SJ-KL remote lane scripts passed.
- Focused tests: Q-FAITHFUL launcher + snapshot loop `19 passed`; block-FP
  planner/builder `6 passed`.

Allowed next use:

- This does not unblock Q-FAITHFUL GPU retraining by itself; retraining gates
  and current score-EV still apply. It makes a future successor checkpoint
  export fail closed unless it proves the same nonzero deployed pose contract.

## Update - 2026-05-02T19:10Z Minimal-Wall-Clock Multimask And SJ-KL Evidence

Current score frontier is unchanged:

- C-067 / Apogee A++ T4 score `0.31561703078448233`, archive `276214` bytes,
  SHA `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`.
- The unchanged-distortion byte gap to score `<0.300` is about `23454` bytes.

CMG3A multimask SHA bug class:

- The first CMG3A reconciler exact-eval wave failed before scorer execution,
  not on score. The archive header recorded typed-array SHA values in
  `source_mask_u8_sha256` / `reconstructed_mask_u8_sha256`, while the runtime
  correctly verified raw uint8 decoded-mask bytes.
- Stale pre-fix failures are classified as
  `failed_prefix_cmg3_header_hash_mismatch_no_score`, not as multimask score
  evidence:
  - `exact_eval_c067_multimask_reconciler_extra072k_l40sdiag_20260502T1840Z`
    failed on the same raw-SHA mismatch.
  - `exact_eval_c067_multimask_reconciler_extra065k_l40sdiag_20260502T1848Z`
    failed on the same raw-SHA mismatch.
  - `exact_eval_c067_multimask_reconciler_extra065k_t4_20260502T1852Z`
    failed on Tesla T4 after CUDA preflight with manifest
    `5af320c7ff15d299c69f3098acbca82238be039d3a579a97ba4cf64315f3254c`
    versus runtime decoded
    `feb59cab6084da9caecc1669cb127fe8e79df7ddd20515c3b476b1f5a4922bfe`.
    Failure logs are locally mirrored under
    `experiments/results/lightning_batch/exact_eval_c067_multimask_reconciler_extra065k_t4_20260502T1852Z/`.
- Permanent fix landed in the builder and tests: CMG3 headers now record raw
  uint8 mask SHA values, and a runtime decode regression test verifies that the
  manifest SHA matches `_decode_cmg3_nonzero_row_runs()` output.

Fixed CMG3A candidates now under exact eval:

- `extra065000` fixed archive:
  - bytes `244760`
  - SHA `b475e9d44fa4816c332c58e9db975dd49d47b58106c771f941880c4a5e4610c2`
  - formula-only rate delta versus C-067 `-0.020943927512`
  - reconstructed-vs-source disagreement `0.003480266995`
  - L40S diagnostic job
    `exact_eval_c067_multimask_reconciler_extra065k_fix1_l40sdiag_20260502T1903Z`
    completed and was harvested:
    - score `7.93854179795029`
    - PoseNet `5.43459129`
    - SegNet `0.00403598`
    - GPU `NVIDIA L40S`, `gpu_t4_match=false`
    - component trace cross-check passed.
  - Classification: scoped A-negative diagnostic for this measured CMG3A
    extra-run count. It proves the fixed header works through inflate and CUDA,
    but the mask geometry collapses PoseNet.
  - T4 promotion-grade job
    `exact_eval_c067_multimask_reconciler_extra065k_fix1_t4_20260502T1903Z`
    was stopped after the identical L40S bytes collapsed. Final SDK status is
    `Stopped`, cost about `$0.0253`, no independent score signal lost.
- `extra072000` fixed archive:
  - bytes `260308`
  - SHA `3c2ef34160a6b14fab361e20f86098e1d4e8950814bff91405d8a9c784f246c9`
  - formula-only rate delta versus C-067 `-0.010591152508`
  - reconstructed-vs-source disagreement `0.002994011773`
  - L40S diagnostic job
    `exact_eval_c067_multimask_reconciler_extra072k_fix1_l40sdiag_20260502T1910Z`
    completed and was harvested:
    - score `6.711537625859189`
    - PoseNet `3.82889199`
    - SegNet `0.00350405`
    - GPU `NVIDIA L40S`, `gpu_t4_match=false`
    - component trace cross-check passed.
  - Classification: scoped A-negative diagnostic for this measured CMG3A
    extra-run count. It confirms the SHA-header bug is fixed and the runtime
    path is contest-evaluable, but the geometry still collapses PoseNet.
- Decision rule: use `extra065000` T4 if it lands component-safe and improves
  C-067; this condition failed. Use `extra072000` only as diagnostic/synergy
  evidence; this condition also failed. Do not T4-promote any plain CMG3A
  run-count candidate solely on byte math. The next mask-grammar dispatch must
  add hotspot/pose-preserving structure, learned reconciliation, or a different
  representation contract before spending T4.

Threshold CMG3A byte-screen:

- Built deterministic fixed-header threshold candidates in
  `experiments/results/c067_multimask_reconciliation_20260502_cmg3a_reconciler_threshold_fix1/`.
- Relevant records:
  - `extra071000`: `257529` bytes, rate-only delta `-0.012441574539`,
    disagreement `0.003056310018`.
  - `extra070000`: `255740` bytes, rate-only delta `-0.013632796206`,
    disagreement `0.003124584622`.
  - `extra069000`: `253691` bytes, rate-only delta `-0.014997141201`,
    disagreement `0.003193138970`.
  - `extra068000`: `251782` bytes, SHA
    `e34590923f8466e625b0514b8e9670cebda8817d8d1b2583dcb421c00102bb6c`,
    rate-only delta `-0.016268265943`, disagreement `0.003263066610`.
  - `extra067000`: `249462` bytes, rate-only delta `-0.017813058714`,
    disagreement `0.003334206475`.
  - `extra066000`: `246949` bytes, rate-only delta `-0.019486362263`,
    disagreement `0.003406516181`.
- `extra068000` is the tight sub-0.300 threshold candidate at unchanged
  distortion, but it is not dispatched yet because `extra065000` already
  collapsed and `extra072000` is the safer active diagnostic. Dispatch
  `extra068000` only if `extra072000` shows a non-collapse regime or if a
  stack-specific reason emerges.

SJ-KL v2 diagnostic:

- `sjkl_c067_v2_k4_a5_cap32k_l40s_20260502T181718Z` completed on L40S and is
  locally harvested without the multi-GB tensor sidecars.
- Exact diagnostic JSON:
  - score `0.4386126293286599`
  - archive bytes `296999`
  - archive SHA
    `37baf8a9bcc975c8eee5167bda20f4a3051d0f9b9a5efd8d71c3c275f1de88f0`
  - SegNet `0.00061036`
  - PoseNet `0.00323342`
  - GPU `NVIDIA L40S`, `gpu_t4_match=false`
- Classification: scoped A-negative diagnostic for the measured SJ-KL v2
  implementation. The charged residual is applied and the payload is under the
  cap, but PoseNet regression plus rate growth make it non-promotable. This
  does not kill SJ-KL as a family; next SJ-KL work must be smaller,
  hotspot-gated, and pose-safe before any T4 spend.

Worker outputs integrated:

- Lane12/NeRV infrastructure support landed only as fail-closed parser work;
  `.omx/state/lane12_nerv_l2_clearance.json` remains absent, so retraining is
  still blocked.
- SJ-KL and Q-FAITHFUL hardening workers added guardrails for byte caps,
  runtime-apply proof, EMA export preference, and nonzero deployed pose
  contracts. Those are infrastructure guards, not score claims.

## Update - 2026-05-02T19:45Z Minimal-Wall-Clock Guardrail And Branch Decision

Provider/liveness state:

- `scripts/reconcile_vast_dispatch_state.py --json` and
  `scripts/verify_vast_instances.py --json` both show no live Vast instances.
  The tracked Q-FAITHFUL A100 `35986044` is gone, not a hidden active run.
- Lightning status for
  `exact_eval_c067_multimask_reconciler_extra072k_fix1_l40sdiag_20260502T1910Z`
  is `Completed`; the stopped duplicate
  `exact_eval_c067_multimask_reconciler_extra065k_fix1_t4_20260502T1903Z`
  is `Stopped`.

Permanent guard landed:

- Added a strict preflight bug-class guard in `src/tac/preflight.py`:
  `check_cmg3a_remote_dispatch_requires_pose_safety`.
- The guard scans remote dispatch surfaces and rejects new CMG3A
  `--target-body-bytes` / `--target-extra-runs` spend unless the command has a
  pose-safety selector (`--field-policy-json`, hard-pair/frame selection,
  class weights) or an explicit `CMG3A_POSE_COLLAPSE_REVIEWED:<reason>` marker.
- This blocks the exact failure class seen in the fixed extra065/extra072
  results: correct archive custody and runtime decode, but catastrophic
  PoseNet collapse from plain run-count/body-byte compression.

Negative-informed hotspot plan:

- Refreshed the hotspot mask-geometry compiler with both fixed CMG3A exact
  negatives as inputs:
  `experiments/results/c067_hotspot_mask_geometry_compiler_20260502/next_pose_safe_plan_after_extra065_072_negatives.json`.
- Inputs included four C067 trace/dynamic ego-foveal atom ledgers plus the
  extra065/extra072 exact CUDA JSONs and candidate manifests.
- The refreshed compiler emits only two surviving policies:
  - `c067_hotspot_poseguard_neg2_top0008`
  - `c067_hotspot_poseguard_neg2_top0012`
- Policies with 16 or more atoms are filtered as
  `known_negative_full_residual_like_pair_spread` /
  `known_negative_full_residual_like_frame_spread`.
- Decision: these 8/12-atom policies are valid planning signals but too small
  to bridge the sub-0.300 gap. Do not dispatch them as standalone exact evals
  unless they become part of a larger learned/stacked archive.

Verification:

- `py_compile` passed for `src/tac/preflight.py` and
  `src/tac/tests/test_preflight_meta_bugs.py`.
- `src/tac/tests/test_preflight_meta_bugs.py`: `250 passed`.
- Focused CMG3A/Yousfi-Fridrich tests: `19 passed`.
- Live repo scan for the new CMG3A guard: no current remote dispatch
  violations.

Next branch:

- Stop spending on plain CMG3A run-count, PMG row-span, global QZS block, and
  broad SJ-KL residuals as standalone lanes.
- Highest-EV remaining routes are:
  1. learned/pose-conditioned mask topology or Q-FAITHFUL successor with
     nonzero pose contract;
  2. SJ-KL successor only if hotspot-gated and much smaller than v2;
  3. renderer/self-compression only after a byte-positive contract exists;
  4. low-risk pose/packer micro-gains only in parallel, never as the main path.

## Update - 2026-05-02T20:08Z SJ-KL v3 Dispatch And Public Payload Accounting

Current exact frontier is unchanged:

```text
C067 / Apogee
score 0.31561703078448233
bytes 276214
sha256 226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a
evidence experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.adjudicated.json
```

SJ-KL v3 wall-clock hedge:

- Added prepared-tensor reuse support to `scripts/remote_lane_sjkl_c067.sh`.
  When `SJKL_PREPARED_TENSOR_DIR` is set, the remote lane requires
  `renderer_target_slot_chw.pt` and `gt_pairs_btchw.pt`, records their
  manifests, and skips the expensive tensor-prep stage.
- Added `scripts/remote_lane_sjkl_c067_v3_cap8k.sh` for the smaller
  hotspot/pose-safe successor: `k=2`, `alpha_bits=4`,
  `n_anchor_pairs=8`, `max_bytes=8192`, using the already-built full tensor
  prep artifact.
- Vast A100 dispatch was attempted after lane claim but provider account
  credit blocked create with zero GPU spend; the claim was closed as
  `failed_provider_account_credit_no_gpu_spend`.
- Lightning L40S diagnostic job is active:
  `sjkl_c067_v3_k2_a4_cap8k_l40s_20260502T2001Z`.
  State file:
  `.omx/state/sjkl_c067_v3_k2_a4_cap8k_l40s_20260502T2001Z_batch_jobs.json`.
  It is diagnostic only, not T4 promotion, until exact JSON and component
  gates show a candidate worth byte-identical T4 confirmation.

Prepared tensor custody:

```text
manifest experiments/results/sjkl_tensor_prep_c067_full_20260502/tensor_manifest.json
manifest_sha256 77a06cd9a16e988cd095f10971ac1c842cff9088d8407f6c56597bd1aecc7e53
gt_pairs_btchw.pt 675M e0ec6826bf7737923b8a547ba87ba3915ba6d61de64bfda3e370a64ff84c0500
renderer_target_slot_chw.pt 1.3G 31c7564b1bcbcade5e2fe19d164b1db3f993b5a2cf84114dd39ff1bfaa4a02df
Lightning source manifest .omx/state/sjkl_c067_v3_stage_20260502T2001Z_manifest.json
```

Public payload accounting:

- ZIP-level accounting now handles invalid/nonstandard archives without
  aborting:
  `experiments/results/archive_byte_profile_20260502/public_and_c067_profile.{json,md}`.
- Stream-level accounting now profiles PR65 compact `x` bundles:
  `experiments/results/public_archive_byte_accounting_20260502/pr65/archive_byte_accounting.{json,md}`.
- The public PR65/PR67/C067 comparison confirms there is no useful outer ZIP
  overhead left. PR65's additional value is a charged postprocess/control
  grammar (`qpost.randmulti`, `qpost.post`, small shift/frac/bias/region
  streams), not generic recompression.

Decision implication:

- PR65-style qpost remains a possible learned/pair-local repair layer after a
  mask-rate win, but exact diagnostics already make blind global qpost
  transfer a measured-bad standalone shape.
- IMP-C067 no-train pruning is not active: existing L40S diagnostics show
  `imp_c01_qzs3_b0128` score `1.355401799409314` and deeper cycles collapse
  further. Future IMP value requires trained sparse export, not current
  no-train pruning.

Verification:

```text
.venv/bin/python -m pytest \
  src/tac/tests/test_archive_byte_profile.py \
  src/tac/tests/test_profile_archive_byte_accounting.py \
  src/tac/tests/test_qzs3_postprocess_candidate.py \
  src/tac/tests/test_remote_lane_sjkl_c067_script.py -q

27 passed, 1 warning

bash -n scripts/remote_lane_sjkl_c067.sh scripts/remote_lane_sjkl_c067_v3_cap8k.sh
passed
```

Telescope/foveation refresh:

- Current external paper: Ewen, Rivkin, Bijelic, Heide, "Telescope:
  Learnable Hyperbolic Foveation for Ultra-Long-Range Object Detection",
  arXiv:2604.06332, submitted 2026-04-07, project link through Princeton
  Light. External source:
  `https://arxiv.org/abs/2604.06332`.
- Relevant transfer: the paper's useful principle for this contest is learned
  hyperbolic resampling for autonomous-driving long-range structure, not a
  fixed radial zoom. Our exact negatives already show naive geometry changes
  can catastrophically move PoseNet, so Telescope-style foveation should enter
  only as a learnable/scorer-gated prior for atom selection, mask topology, or
  low-dimensional ego-motion fields, with all transform parameters charged and
  exact-evaluated.

## Update - 2026-05-02T20:43Z C067 Micro-Mask Exact Negatives And Save12k Diagnostic

Current exact frontier remains unchanged:

```text
C067 / Apogee
score 0.31561703078448233
bytes 276214
sha256 226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a
evidence experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.adjudicated.json
```

C067 byte profile:

- Generated byte-only profile:
  `experiments/results/c067_archive_byte_profile_20260502/archive_byte_profile.{json,md}`.
- The deploy archive is a single stored member `p` with total ZIP size
  `276214` and estimated ZIP overhead `100` bytes.
- Decision: there is no meaningful outer-ZIP slack. Remaining rate progress
  must come from the packed `p` grammar, stream payloads, decoder/self-
  compression, or scorer-safe charged correction atoms.

Micro-mask AV1 exact diagnostics:

- `exact_eval_c067_micro_mask_crf52_hf32_l40sdiag_20260502T2020Z`
  landed as an L40S diagnostic negative:
  score `1.9740232607944193`, bytes `242718`, archive SHA
  `4c8c37752766ecc1d44cd3582286d446fdda008abc8bd75a7b1e5f29f1f8c272`,
  PoseNet `0.23345281`, SegNet `0.00284491`.
- `exact_eval_c067_micro_mask_crf56_hf32_l40sdiag_20260502T2022Z`
  landed as an L40S diagnostic negative:
  score `2.5639286320922166`, bytes `194240`, archive SHA
  `f4867ec2261b1568c18078043cea7466145df1d8dbb1809d253ee2fbe17937db`,
  PoseNet `0.42404205`, SegNet `0.00375364`.
- Both artifacts were re-adjudicated after the non-T4 promotion-gate fix; they
  are exact CUDA diagnostics with `scientific_score_eligible=true` but
  `promotion_eligible=false` because L40S is not T4/equivalent.

Planner-derived trust-region state:

- Local planner output:
  `experiments/results/c067_micro_mask_reencode_plan_20260502/micro_mask_reencode_plan.json`.
- `save05k` byte-screen was byte-regressive (`289931` bytes after runtime
  build) and was not dispatched.
- `save08k` packed archive is `257942` bytes, SHA
  `a6b8f9c66084be1d98c010e753ec5cd629210a1fde6804dd66956ad5215ac923`,
  with formula-only rate delta `-0.012166574791` vs C067. This cannot cross
  sub-0.300 without additional distortion gain.
- `save12k` packed archive is `233612` bytes, SHA
  `d31be8b1e56ec277d50c023d19bb9311c28b9a9dfe6d6e675e3bce4488fbb40b`,
  with formula-only rate delta `-0.028366923121` vs C067. This is large
  enough to cross sub-0.300 only if component penalty stays below about
  `0.01275`.
- `exact_eval_c067_micro_mask_save12k_l40sdiag_20260502T2034Z` is running on
  Lightning L40S with component trace enabled. It is diagnostic only; T4
  promotion is allowed only if exact components are safe and the score
  materially improves.

Scientific interpretation:

- Plain protected AV1 mask reencode is a measured bad implementation shape:
  sub-percent class-pixel disagreement still causes catastrophic PoseNet
  movement.
- This is not a broad mask-rate or grayscale/learned-topology kill. The next
  mask-rate experiment should be charged postdecode repair atoms, PR65-style
  control/qpost protection repurposed as a learned repair layer, or a
  learned topology/INR/NeRV successor once Lane 12 infrastructure clears.
- The save12k result should be harvested even if it collapses because its
  component trace can train the next marginal-benefit-per-byte repair selector.

Verification:

```text
.venv/bin/python -m py_compile \
  scripts/adjudicate_contest_auth_eval.py \
  experiments/build_protected_mask_reencode_candidate.py \
  experiments/plan_c067_micro_mask_reencode.py \
  src/tac/archive_byte_profile.py \
  experiments/profile_archive_bytes.py \
  src/tac/tests/test_remote_auth_eval_hardening.py \
  src/tac/tests/test_build_protected_mask_reencode_candidate.py \
  src/tac/tests/test_plan_c067_micro_mask_reencode.py \
  src/tac/tests/test_archive_byte_profile.py

.venv/bin/python -m pytest \
  src/tac/tests/test_build_protected_mask_reencode_candidate.py \
  src/tac/tests/test_plan_c067_micro_mask_reencode.py \
  src/tac/tests/test_remote_auth_eval_hardening.py \
  src/tac/tests/test_archive_byte_profile.py -q

60 passed, 1 warning
```

## 2026-05-02T21:05Z - C067 postdecode charged-repair diagnostics

Save12k protected AV1 landed as an exact L40S diagnostic negative:

- job:
  `exact_eval_c067_micro_mask_save12k_l40sdiag_20260502T2034Z`
- archive bytes: `233612`
- archive SHA-256:
  `d31be8b1e56ec277d50c023d19bb9311c28b9a9dfe6d6e675e3bce4488fbb40b`
- exact CUDA score: `1.5782530550433744`
- PoseNet distance: `0.13292921`
- SegNet distance: `0.00269751`
- component trace SHA-256:
  `34630e6ce5f11a36dfd47bf0961d0b7c8181e0616a99d7a8ba547c887ca265db`

This is a scoped implementation negative for plain protected AV1 mask
re-encoding, not a mask-rate family kill. The decisive forensic clue is that
the intended protected regions still carried almost all postdecode residual
pixels after lossy AV1: `359711` total changed class pixels, `339460` inside
the protected set, and only `20251` outside. Protection before lossy video
encode is therefore not the same as postdecode scorer-geometry protection.

The immediate successor is charged postdecode AMR1 repair, where every
correction bit is inside `archive.zip`:

- `top10` trace-frame repair:
  `experiments/results/c067_postdecode_mask_repair_candidate_20260502/save12k_trace_frame_sweep/top10/archive.zip`
  - bytes: `251122`
  - SHA-256:
    `447dc22798c287fbb26ba1a9bf63d925fc328d281e51988fc07148b325e0c4fd`
  - compressed repair bytes: `11472`
  - selected atoms: `50`
  - formula-only rate delta vs C067: `-0.016707732851741524`
  - Lightning L40S diagnostic:
    `exact_eval_c067_postdecode_repair_save12k_top10_l40sdiag_20260502T2054Z`
    harvested as a diagnostic negative:
    - exact CUDA score: `1.4680100510237182`
    - PoseNet distance: `0.10732614`
    - SegNet distance: `0.00264815`
    - component trace SHA-256:
      `5b4d28329c60cfc29e955bca70d33dd136c2ec36264dbe9eae05d818c09434f1`
    - adjudication:
      `scientific_score_eligible=true`, `promotion_eligible=false`,
      `hardware_promotion_gate_triggered=true`.
    - verdict: no T4 promotion; use the trace as atom-response evidence.
- `budget8000` compressed-byte waterfill repair:
  `experiments/results/c067_postdecode_mask_repair_candidate_20260502/save12k_budget_sweep/budget8000/archive.zip`
  - bytes: `247414`
  - SHA-256:
    `97a25794da10e2d778b13fd44a6a45c623981a5619ec1ed8ea6a88798ed267f3`
  - compressed repair bytes: `7764`
  - selected atoms: `16`
  - formula-only rate delta vs C067: `-0.019176737849918534`
  - Lightning L40S diagnostic:
    `exact_eval_c067_postdecode_repair_save12k_budget8000_l40sdiag_20260502T2101Z`
    harvested as a diagnostic negative:
    - exact CUDA score: `1.5492577842107278`
    - PoseNet distance: `0.125453`
    - SegNet distance: `0.00264457`
    - component trace SHA-256:
      `5406f8bc4b05592064031bdc0f3e616e9bd9ae3ff6bc0f50c67e271c31790a8b`
    - adjudication:
      `scientific_score_eligible=true`, `promotion_eligible=false`,
      `hardware_promotion_gate_triggered=true`.
    - verdict: no T4 promotion; the compressed-byte prefix is not dense enough.

Top10 did reduce the save12k collapse from score `1.5782530550433744` to
`1.4680100510237182`, but the repair density is still far below what is
needed for C067 or sub-0.300. The top remaining pose pairs after top10 are:

```text
411, 212, 68, 461, 457, 597, 484, 365, 282, 85,
469, 139, 222, 290, 71, 357, 215, 280, 75, 240,
446, 429, 456, 437, 528, 423, 373, 41, 441, 343,
274, 311, 276, 256, 81, 135, 512, 309, 270, 174
```

The top remaining SegNet pairs are:

```text
522, 517, 540, 518, 159, 532, 488, 596, 492, 479,
178, 524, 584, 536, 520, 542, 200, 467, 151, 510,
67, 594, 300, 170, 459, 45, 534, 566, 538, 494
```

The next selector should not simply add more atoms by the same trace-frame
rule. It should price candidate repair atoms against this measured
post-repair trace and ask whether each atom can buy down PoseNet/SegNet faster
than the rate term `25 / 37545489` per byte.

Two high-EV risky lanes were split into dedicated workers because the sparse
repair branch is producing signal but not frontier movement:

- Renderer self-compression / Block-FP / Selfcomp-style transplant against
  the C067 JointFrameGenerator payload. This targets the missing sub-0.300
  gap directly through renderer/packer bytes.
- SJ-KL / low-dimensional residual / water-fill stacking redesign. This
  targets residual structure with a byte cap after the prior v2/v3 evidence
  showed the first implementation was too large or failed its cap guard.

The trace-driven water-fill planner also produced non-promotable frame-class
and pair-class plans:

- frame-class plan:
  `experiments/results/c067_postdecode_mask_repair_candidate_20260502/c067_postdecode_mask_repair_waterfill_frame_class_plan.json`
  with `124` atoms and `45` builder units.
- pair-class plan:
  `experiments/results/c067_postdecode_mask_repair_candidate_20260502/c067_postdecode_mask_repair_waterfill_pair_class_plan.json`
  with `256` atoms and `204` builder units.

One additional high-signal repair diagnostic was launched from the pair-class
water-fill plan:

- archive:
  `experiments/results/c067_postdecode_mask_repair_candidate_20260502/save12k_waterfill_pair_sweep/budget4000/archive.zip`
- bytes: `243422`
- SHA-256:
  `25f890f449796f79c0da246758d05684c40ccce8b29a3bc2322b8958fa7ae489`
- compressed repair bytes: `3772`
- selected atoms: `4`
- selected pair/class atoms:
  `pair0079_class1`, `pair0230_class1`, `pair0216_class1`,
  `pair0153_class1`
- formula-only rate delta vs C067: `-0.025223403003220974`
- Lightning L40S diagnostic:
  `exact_eval_c067_postdecode_repair_save12k_pairwaterfill4k_l40sdiag_20260502T2114Z`
  is queued.

Decision rule: if pair-waterfill4k still scores far above C067, stop spending
GPU on sparse AMR1 repair as a primary sub-0.300 path and keep it only as a
component-trace generator for a learned repair layer. Main effort then stays
on renderer self-compression and SJ-KL/low-rank residual stacking.

The byte-budgeted selector is deliberately anti-arbitrary: it binary-searches
the largest deterministic atom prefix whose compressed AMR1 payload fits the
declared byte budget. Exact component traces from the two diagnostics become
the next water-fill reward surface if the first atom set is still too
distortion-heavy.

Verification:

```text
.venv/bin/python -m py_compile \
  experiments/build_c067_postdecode_mask_repair_candidate.py \
  src/tac/tests/test_build_c067_postdecode_mask_repair_candidate.py \
  scripts/adjudicate_contest_auth_eval.py

.venv/bin/python -m pytest \
  src/tac/tests/test_build_c067_postdecode_mask_repair_candidate.py -q

5 passed, 1 warning
```

## 2026-05-02T21:30Z - Two-lane high-risk push after sparse-repair negatives

Score claim: `false` for this section except exact artifacts explicitly cited
above.

The sparse postdecode repair branch is still useful as a component-trace
generator, but it has not produced frontier movement. The work is therefore
split into two higher-EV lanes while the final pair-waterfill diagnostic runs.

### Lane A: renderer self-compression / Block-FP / QZS3 contract

Returned local screen:

- planner:
  `experiments/plan_c067_renderer_self_compression_v2.py`
- local report:
  `experiments/results/lane_a_blockfp_c067_20260502_local_screen/fail_closed_contract_report.json`
- best archive-byte win:
  `global_b4096`, `270878` bytes,
  SHA-256
  `3d78ddeeb8cb9bd3031e3d2256370f729dc5a61753d1ac5ceecb0778774d9934`
- dispatch decision: no exact-eval dispatch from this screen.

Reason: the only byte-winning candidates are global QZS3 reblocks, and the
same family already has exact L40S CUDA negative evidence via
`exact_eval_c067_qzs3_b512_l40sdiag_20260502T1710Z`, score
`2.2397462747539274`, PoseNet `0.3804819`. The component-aware variants avoid
the known collapse mode but are byte-regressive (`+68` bytes or worse), and the
QBF1 screen is also byte-regressive. This does not kill renderer
self-compression; it closes this measured reblock implementation and redirects
the large renderer-byte bet toward trained/self-compressed exports rather than
untrained global quantizer changes.

### Lane B: SJ-KL sparse low-dimensional canary

Returned local screen:

- byte screen:
  `experiments/results/sjkl_c067_sparse_repack_screen_20260502T_local/byte_screen_summary.json`
- exact-eval recommendation:
  `experiments/results/sjkl_c067_sparse_repack_screen_20260502T_local/exact_eval_recommendation.json`
- strict runtime archive sanity:
  `experiments/results/sjkl_c067_sparse_repack_screen_20260502T_local/runtime_sanity_summary.json`

The smallest canary was queued for exact CUDA diagnostic because it spends only
`1241` extra bytes, carries `sjkl.bin=467` bytes, and needs only `0.00082633`
component-score improvement to break even:

- job:
  `exact_eval_sjkl_c067_sparse_k1g08_p32_a3_rtxprodiag_20260502T2123Z`
- archive:
  `experiments/results/sjkl_c067_sparse_repack_screen_20260502T_local/k1_g08x06_p32_a3_gain003125/pack/archive.zip`
- archive bytes: `277455`
- archive SHA-256:
  `f4146e137d9e2e12d1f7c8d26609f24e0e2bd0d19e5c0534f90c761c35658825`
- runtime guard: `SJKL_REQUIRE_APPLIED=1`
- machine route: Lightning `g7e.12xlarge` / RTX PRO diagnostic, same Studio
  cloud account as the source snapshot.

The first attempted RTX PRO submit against the GCP cloud account failed before
GPU spend because Lightning rejected cross-cloud Studio/job execution. The
claim was closed as `cancelled_renamed_no_gpu_spend`, then the same archive
was submitted on `lightning-public-prod` / `g7e.12xlarge` after doctor passed.

### Lane C: Lane 12 learned-mask unblock

Lane 12 remains the biggest plausible rate lever, but retraining remains
fail-closed until a legitimate L2 clearance packet exists. A worker was
assigned to make the unblock criteria executable rather than bypassing them:

- planned tool:
  `experiments/plan_lane12_l2_unblock.py`
- planned output:
  `experiments/results/lane12_l2_unblock_readiness_20260502/`
- rule: produce a fail-closed readiness report and redesign recipe; do not
  write `.omx/state/lane12_nerv_l2_clearance.json` unless every launcher gate
  is actually satisfied.

### Active exact diagnostics

- `exact_eval_c067_postdecode_repair_save12k_pairwaterfill4k_l40sdiag_20260502T2114Z`
  landed as a clean L40S diagnostic negative:
  - exact CUDA score: `1.5721465219411697`
  - archive bytes: `243422`
  - archive SHA-256:
    `25f890f449796f79c0da246758d05684c40ccce8b29a3bc2322b8958fa7ae489`
  - PoseNet distance: `0.1304232`
  - SegNet distance: `0.00268032`
  - component trace SHA-256:
    `fb9db3281ef0b570be2e77ac9dd3e4f7b92644ae03753139355dce2bb3437e6c`
  - adjudication:
    `scientific_score_eligible=true`, `promotion_eligible=false`,
    `hardware_promotion_gate_triggered=true`.
  - decision: no T4 promotion.
- `exact_eval_sjkl_c067_sparse_k1g08_p32_a3_rtxprodiag_20260502T2123Z`
  landed as an RTX PRO exact CUDA diagnostic:
  - exact CUDA score: `0.3161753677975756`
  - archive bytes: `277455`
  - archive SHA-256:
    `f4146e137d9e2e12d1f7c8d26609f24e0e2bd0d19e5c0534f90c761c35658825`
  - PoseNet distance: `0.00049541`
  - SegNet distance: `0.00061044`
  - score delta vs C067: `+0.0005583370130932686`
  - component trace SHA-256:
    `52716125ecb7ce402e9e8c4a81a1db16916612f10401b76df6fa447335ed592b`
  - adjudication:
    `scientific_score_eligible=true`, `promotion_eligible=false`,
    `hardware_promotion_gate_triggered=true`.
  - decision: no T4 promotion.

Postdecode AMR1 branch decision: stop spending exact-eval GPUs on sparse AMR1
as a primary sub-0.300 path. It produced high-value component traces but did
not buy down PoseNet fast enough. Any future use should be either a learned
repair layer, a training target for a differentiable selector, or a sparse
auxiliary inside a larger learned/topology codec; it should not dispatch more
static frame/pair/class prefix variants by itself.

SJ-KL branch decision: the sparse `SJK2` canary is component-positive but
score-negative. The exact result proves the residual direction has the correct
sign in this basin, but its measured component benefit is only about one third
of the rate break-even. Future SJ-KL work should optimize benefit-per-byte,
not merely add more rows: either shrink the 467-byte payload, select pairs from
the exact component trace, share a smaller basis across harder pairs, or fold
SJ-KL into a learned renderer/mask residual where the basis bytes amortize over
a larger component win. Do not T4-promote this archive.

Verification:

```text
.venv/bin/python -m pytest \
  src/tac/tests/test_plan_c067_renderer_self_compression_v2.py \
  src/tac/tests/test_build_mixed_qzs_block_candidate.py \
  src/tac/tests/test_build_blockfp_c067_archive.py \
  src/tac/tests/test_sjkl_basis.py \
  src/tac/tests/test_inflate_renderer_sjkl_runtime.py \
  src/tac/tests/test_build_sjkl_c067_archive.py -q

57 passed in 4.49s

.venv/bin/python -m pytest \
  src/tac/tests/test_plan_c067_postdecode_mask_repair_waterfill.py \
  src/tac/tests/test_build_c067_postdecode_mask_repair_candidate.py -q

8 passed, 1 warning
```

## 2026-05-02T21:53Z - Big-change gates plus trace-selected SJ-KL wave

Score claim: `false` for this section. The A++ frontier remains C067 /
Apogee at score `0.31561703078448233`, `276214` bytes, SHA-256
`226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`.

### SJ-KL explicit-pair metabug fixed

The first sparse SJ-KL canary was component-positive but rate-negative. The
next useful canary must preserve absolute pair identity, because the component
trace and hard-pair ledgers are expressed in contest pair indices, not packed
row positions.

Implemented fix:

- `experiments/build_sjkl_residual.py`
  - added explicit pair-index parsing from comma/space strings or JSON/text
    files;
  - rejects `--pair-selection explicit` combined with `--max-encoded-pairs`;
  - maps absolute contest pair indices back to source rows and records both
    requested and selected pair indices;
  - forces sparse alpha-block packing for explicit pair selections;
  - fixed `apply_sjkl_at_decode` to respect sparse absolute pair indices.
- `src/tac/tests/test_sjkl_basis.py`
  - covers explicit pair repack custody, conflict/missing-pair fail-closed
    behavior, and decode-time sparse pair-index application.

This permanently closes the "low-dimensional residual rows silently lose
absolute pair identity" bug class for SJ-KL experiments.

### Trace-selected SJ-KL exact diagnostics queued

Both candidates were built from the RTX PRO full SJ-KL tensor artifact
`experiments/results/lightning_batch/sjkl_c067_rtxprodiag_20260502T151756Z/build/sjkl.bin`
with source SHA-256
`04e95e588d6ec874809994856c56a5940e7b8e93e1d9e531056ff1505fb94e9f`,
then staged through manifest
`.omx/state/sjkl_c067_trace_selected_stage_20260502T214429Z_manifest.json`.

Queued Lightning RTX PRO diagnostic jobs:

- `exact_eval_sjkl_c067_tracepos9_rtxprodiag_20260502T2146Z`
  - archive:
    `experiments/results/sjkl_c067_trace_selected_repack_20260502T_local/old_selected_positive9/pack/archive.zip`
  - archive bytes: `277626`
  - archive SHA-256:
    `3c5a4bb287b194b2fdcff014d60c22891299c2db5e31c87dcb26cbe5efcadf23`
  - `sjkl.bin` bytes: `387`
  - selected pairs: `332,555,553,376,352,320,576,556,244`
  - runtime guard: `SJKL_REQUIRE_APPLIED=1`
- `exact_eval_sjkl_c067_tracetop16_rtxprodiag_20260502T2146Z`
  - archive:
    `experiments/results/sjkl_c067_trace_selected_repack_20260502T_local/trace_top16/pack/archive.zip`
  - archive bytes: `277547`
  - archive SHA-256:
    `6dabf8cff8a3528e1f51085f7574783198573589a6d200ffa288116af93d9bac`
  - `sjkl.bin` bytes: `422`
  - selected pairs: `153,156,37,111,382,77,326,579,87,133,140,172,441,136,347,452`
  - runtime guard: `SJKL_REQUIRE_APPLIED=1`

Both were still Lightning `Pending` with zero cost at the first refresh after
submit. Promotion rule: no T4 duplicate unless exact CUDA components beat the
payload rate cost.

`exact_eval_sjkl_c067_tracepos9_rtxprodiag_20260502T2146Z` then landed and
was harvested through `harvest-ssh`:

- exact CUDA diagnostic score: `0.31627277727763115`
- archive bytes: `277626`
- archive SHA-256:
  `3c5a4bb287b194b2fdcff014d60c22891299c2db5e31c87dcb26cbe5efcadf23`
- PoseNet distance: `0.00049518`
- SegNet distance: `0.00061044`
- score delta vs C067: `+0.0006557464931488122`
- component trace SHA-256:
  `646d57c2e533b6f9764cedb1938c6cf4ddc563946ac3ad57161a007a0b761709`
- component trace cross-check: `all_match=true`
- hardware: RTX PRO, not T4/equivalent.

Decision: no T4 promotion. This is another component-positive / rate-negative
SJ-KL result; it strengthens the sign evidence but says the next SJ-KL action
must optimize benefit per charged byte, not simply select a different small
set of pairs. A local allocator worker was assigned to convert the sparse and
tracepos9 component traces into explicit pair/payload-shrink recommendations.

`exact_eval_sjkl_c067_tracetop16_rtxprodiag_20260502T2146Z` also landed and
was harvested from the SDK artifact mirror after a nonterminal SDK status
regression (`Running -> Pending`):

- exact CUDA diagnostic score: `0.3162181456257131`
- archive bytes: `277547`
- archive SHA-256:
  `6dabf8cff8a3528e1f51085f7574783198573589a6d200ffa288116af93d9bac`
- PoseNet distance: `0.00049515`
- SegNet distance: `0.00061044`
- score delta vs C067: `+0.0006011148412307654`
- component trace SHA-256:
  `fe8af77b4cc11b001c08e3cd51e778eec1f228a0cee4b56886836e6f972b04b6`
- component trace cross-check: `all_match=true`
- hardware: RTX PRO, not T4/equivalent.

Decision: no T4 promotion. This is the best SJ-KL trace-selected candidate so
far, but it is still score-negative. The useful scientific result is stable
component-positive sign: the basis nudges both PoseNet and SegNet in the right
direction, but current payload accounting is still too expensive by roughly a
factor of three for standalone promotion.

### Lane12 / Alpha learned-mask gate hardened

The Lane12 decoded-baseline unblock worker fixed a real contract mismatch:
Alpha-Geo contracts hash decoded masks as canonical `torch.uint8`, while
`experiments/train_nerv_mask.py` was hashing the same labels as `torch.int64`.
The trainer now validates decoded-baseline mask custody against the same dtype
contract that Alpha-Geo emits, while still casting labels to `long` at sampling
time.

New fail-closed artifact:

- `experiments/results/lane12_l2_unblock_readiness_20260502/decoded_baseline_build_preflight.json`
  - `decoded_baseline_contract_preflight_passed=true`
  - decoded mask SHA-256:
    `cce3a986341c40df9b9ebca24ff96e16c4b41b40b388dc2af86161ba76e2b4e9`
  - `ready_for_build_only_remote_training=false`

Lane12 remains blocked by evidence, not tooling:

- `.omx/state/lane12_nerv_l2_clearance.json` is missing.
- no current Alpha-Geo geometry JSON passes
  `diagnostic=alpha_geo_0_nerv_geometry`, empirical non-proxy evidence, full
  `1200x384x512`, and `pass_fail.overall_pass=true`.

### Geometry-safe C067 topology gate

The C067 geometry-safe mask/topology v2 gate reviewed `26` current PMG/CMG3A/
multimask/micro-mask/AMR1 same-family candidates and found `0` dispatchable:

- plan:
  `experiments/results/c067_geometry_safe_mask_topology_v2_20260502/c067_geometry_safe_mask_topology_v2_plan.json`
- ledger:
  `.omx/research/c067_geometry_safe_mask_topology_v2_20260502_codex.md`

Decision: do not run more same-family static mask topology replacements from
this pool. The next mask big move must preserve C067 global decoded geometry
and only charge delta/overlay atoms inside a strict trust region with
catastrophic-pair vetoes.

The follow-up decoded-baseline delta/overlay mask-topology planner was then
implemented:

- planner:
  `experiments/plan_c067_decoded_delta_overlay_mask_topology.py`
- artifact:
  `experiments/results/c067_decoded_delta_overlay_mask_topology_20260502/c067_decoded_delta_overlay_mask_topology_plan.json`
- result: `4` donor overlays screened, `0` safe payload specs, `0`
  dispatchable candidates.

The selected trust-region pairs `79,153,212,216,230` all overlap the
catastrophic exact-negative pair set, so the planner correctly failed closed.
No remote dispatch is warranted from this donor/pair set.

### Active workers after this checkpoint

- `Sagan the 2nd`: Alpha-Geo geometry-evidence unlock infrastructure; no GPU
  dispatch and no clearance-state writes.
- `Hegel the 2nd`: decoded-baseline delta/overlay mask-topology builder or
  planner; no SJ-KL edits and no remote dispatch.

Verification:

```text
.venv/bin/python -m py_compile \
  experiments/build_sjkl_residual.py \
  experiments/preflight_lane12_decoded_baseline_build.py \
  experiments/plan_c067_geometry_safe_mask_topology_v2.py \
  experiments/plan_lane12_l2_unblock.py \
  experiments/train_nerv_mask.py \
  src/tac/tests/test_sjkl_basis.py \
  src/tac/tests/test_preflight_lane12_decoded_baseline_build.py \
  src/tac/tests/test_plan_c067_geometry_safe_mask_topology_v2.py

.venv/bin/python -m pytest \
  src/tac/tests/test_sjkl_basis.py \
  src/tac/tests/test_inflate_renderer_sjkl_runtime.py \
  src/tac/tests/test_build_sjkl_c067_archive.py -q

39 passed in 1.06s

.venv/bin/python -m pytest \
  src/tac/tests/test_preflight_lane12_decoded_baseline_build.py \
  src/tac/tests/test_plan_lane12_l2_unblock.py \
  src/tac/tests/test_plan_c067_geometry_safe_mask_topology_v2.py -q

13 passed in 0.57s
```

## 2026-05-02T22:05Z - SJ-KL trace benefit-per-byte allocator

Local-only SJ-KL allocator planner implemented; no remote dispatch.

- planner:
  `experiments/plan_sjkl_trace_benefit_allocator.py`
- focused tests:
  `src/tac/tests/test_plan_sjkl_trace_benefit_allocator.py`
- artifact:
  `experiments/results/sjkl_trace_benefit_allocator_20260502/sjkl_trace_benefit_plan.json`
- schema: `sjkl_trace_benefit_allocator_plan_v1`
- score claim: `false`

Inputs were the C067 A++ T4 component trace, the two RTX PRO exact SJ-KL
diagnostics, local pack/repack manifests, and the SJ-KL byte-screen summaries.
The planner fails closed on missing trace samples, mismatched archive
SHA/bytes, source-archive mismatch, and pack/repack `sjkl.bin` custody drift.

Measured classification:

- `sparse_k1_g08x06_p32_a3_gain003125`: score delta `+0.000558337013`,
  component benefit `+0.000268162987`, rate cost `+0.0008265`, break-even
  archive delta `402` bytes, required shrink `839` bytes. Classification:
  `score_negative_component_positive`.
- `trace_selected_positive9`: score delta `+0.000655746493`, component benefit
  `+0.000284503507`, rate cost `+0.00094025`, break-even archive delta `427`
  bytes, required shrink `985` bytes. Classification:
  `score_negative_component_positive`.

Planning consequence: the current exact SJ-KL traces are useful component
signal, not frontier movement. The selected-pair benefit for the sparse policy
was negative (`-0.000043159085486`), while the old positive-9 policy selected
only `+0.000007574346049`; most observed component benefit came from unselected
global-response pairs. Do not promote either exact diagnostic or spend another
T4 on the same payload shape.

Top next local recommendations:

1. Explicit pair policy over consensus positive trace deltas:
   `153,156,37,111,133,326,579,77,382,140,172,87,196,530,452,558`.
   Avoid measured negative-response pairs
   `246,105,154,12,501,354,284,417` unless a new exact trace reverses them.
2. Payload shrink gate: no SJ-KL successor should promote unless the packed
   archive delta is under about `402` bytes for the measured benefit level, or
   component benefit grows materially. Required shrink from the best exact
   diagnostic is at least `839` bytes.
3. Existing local `trace_top16` byte screen already targets a close top-positive
   set (`153,156,37,111,382,77,326,579,87,133,140,172,441,136,347,452`) with
   `422` charged `sjkl.bin` bytes but `1333` archive-delta bytes. It remains
   empirical planning evidence only until rebuilt/shrunk and exact CUDA scored.

Verification:

```text
.venv/bin/python -m py_compile \
  experiments/plan_sjkl_trace_benefit_allocator.py \
  src/tac/tests/test_plan_sjkl_trace_benefit_allocator.py

.venv/bin/python -m pytest \
  src/tac/tests/test_plan_sjkl_trace_benefit_allocator.py -q

3 passed in 0.12s
```

## 2026-05-02T22:20Z - Sub-0.24 requirement ledger

Local-only Shannon/Yousfi-Fridrich floor requirement planner implemented; no
remote dispatch and no score claim.

- planner: `experiments/plan_shannon_floor_sub024_requirements.py`
- focused tests:
  `src/tac/tests/test_plan_shannon_floor_sub024_requirements.py`
- artifact:
  `experiments/results/shannon_floor_sub024_requirements_20260502/shannon_floor_sub024_requirements.json`
- schema: `shannon_floor_sub024_requirements_v1`
- evidence grade: `derivation_plus_exact_frontier_inputs`

Inputs:

- C067 A++ T4 adjudicated eval:
  `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.adjudicated.json`
- C067 archive byte accounting:
  `experiments/results/c067_archive_byte_accounting_20260502/archive_byte_accounting.json`

Frontier terms from the exact C067 packet:

- archive bytes: `276214`
- exact archive SHA-256:
  `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`
- SegNet score term: `0.061244`
- PoseNet score term: `0.07045353078448233`
- distortion score term: `0.13169753078448232`
- exact byte-formula rate term: `0.18391956487768743`

Requirement results:

- sub-0.300 at unchanged distortion requires archive `<=252759` bytes, or
  `23455` bytes saved.
- sub-0.240 at unchanged distortion requires archive `<=162650` bytes, or
  `113564` bytes saved.
- sub-0.240 at unchanged archive bytes requires distortion score term
  `<=0.05608043512231256`, a `0.07561709566216976` reduction from C067.

Stream implications for sub-0.240:

- `masks.mkv` only: target mask stream `105908` bytes, requiring `51.74%`
  savings from current `219472` bytes.
- `renderer.bin` only: impossible; current renderer stream is only `55965`
  bytes.
- renderer to `11000` bytes plus mask: requires another `68599` mask bytes
  saved, leaving target mask stream `150873` bytes.
- pose stream upper bound is `677` bytes / score span `0.0004507865`; it is
  polish only.

Decision: sub-0.24 is not reachable by another pose/SJ-KL/postfilter
micro-lane unless that micro-lane becomes payload-free or is fused into a
large learned stream. The non-negotiable large moves are:

1. geometry-preserving mask topology or learned mask decoder;
2. real non-surrogate trained renderer self-compression;
3. stacked mask+renderer+positive atom archive with its own exact CUDA eval.

Verification:

```text
.venv/bin/python -m py_compile \
  experiments/plan_shannon_floor_sub024_requirements.py \
  src/tac/tests/test_plan_shannon_floor_sub024_requirements.py

.venv/bin/python -m pytest \
  src/tac/tests/test_plan_shannon_floor_sub024_requirements.py -q

3 passed in 0.08s
```

## 2026-05-02T22:35Z - Predictive mask grammar runtime-readiness planner

Local-only PMG/CMG3 archive-readiness planner implemented; no remote dispatch
and no score claim.

- planner:
  `experiments/plan_predictive_mask_grammar_runtime_readiness.py`
- focused tests:
  `src/tac/tests/test_plan_predictive_mask_grammar_runtime_readiness.py`
- artifact:
  `experiments/results/predictive_mask_grammar_runtime_readiness_20260502/predictive_mask_grammar_runtime_readiness_plan.json`
- schema: `predictive_mask_grammar_runtime_readiness_plan_v1`
- verdict: `blocked_local_only_archive_readiness`

Key findings:

- The `row_span_stride4_class_predictor` probe has `63212` compressed bytes and
  formula-only sub-0.24 headroom of `42696` bytes before decoder, packer,
  validator, runtime, and distortion costs.
- A local CMG3 decoder/packer route exists for `masks.cmg3`, including row-span,
  hotspot-residual, and nonzero-run modes.
- The probe itself is still byte-probe-only: it is not a reviewed archive
  member, uses probe compressor label `lzma6` rather than the runtime enum, and
  has no local inflate parity or exact CUDA artifact.
- Existing CMG3/PMG exact diagnostics are same-family blockers for dispatch:
  seven configured artifacts are present and all show PoseNet and/or SegNet
  collapse. The strongest byte-only warning is `c067_cmg3_nonzero_top1_t4`,
  which is sub-0.24-byte-sufficient at `128028` bytes but collapses to PoseNet
  `39.78416443`, SegNet `0.09192351`.

Decision: do not dispatch PMG/CMG3 row-span or row-run variants from byte
screening alone. The ranked next action is a local decoded-mask geometry parity
gate, then canonical CMG3 re-emission only if that gate explains why the
candidate escapes the measured collapse family.

Verification:

```text
python3 -m py_compile \
  experiments/plan_predictive_mask_grammar_runtime_readiness.py \
  src/tac/tests/test_plan_predictive_mask_grammar_runtime_readiness.py

.venv/bin/python -m pytest \
  src/tac/tests/test_plan_predictive_mask_grammar_runtime_readiness.py -q

4 passed in 0.10s
```

## 2026-05-02T23:05Z - PMG/row-span remote-dispatch preflight hardening

Permanent bug-class guard landed for the PMG/row-span exact-negative replay
surface; no remote dispatch and no score claim.

- preflight implementation: `src/tac/preflight.py`
- focused tests: `src/tac/tests/test_preflight_meta_bugs.py`
- forensic replay script guard:
  `scripts/remote_lane_pmg_hotspot_c067_eval.sh`
- durable protocol update: `AGENTS.md`

Guard behavior:

- `check_pmg_remote_dispatch_requires_geometry_escape` scans remote dispatch
  scripts for PMG/row-span mask-grammar evals that would spend GPU through
  `remote_archive_only_eval.sh`, `launch_lightning_batch_job.py`, or
  `contest_auth_eval.py`.
- A dispatch must now carry one of:
  `PMG_GEOMETRY_ESCAPE_REVIEWED`, `PMG_EXACT_NEGATIVE_REPLAY_GUARD`,
  `--geometry-escape-json`, `--pose-safe-plan-json`, or
  `--learned-mask-contract-json`.
- The existing `remote_lane_pmg_hotspot_c067_eval.sh` is now guarded for
  historical forensic replay only via `ALLOW_REPLAY_EXACT_NEGATIVE_PMG=1`.

Reason:

- The predictive-mask planner confirmed that PMG/CMG3 byte headroom is real,
  but existing exact CUDA diagnostics are same-family blockers until a local
  decoded-geometry parity/escape proof exists.
- This guard blocks the meta-bug class "byte-only row-span candidate burns
  remote exact-eval time despite known PoseNet-collapse evidence."

Verification:

```text
.venv/bin/python -m py_compile \
  src/tac/preflight.py \
  src/tac/tests/test_preflight_meta_bugs.py

.venv/bin/python -m pytest \
  src/tac/tests/test_preflight_meta_bugs.py -q

253 passed in 65.62s

.venv/bin/python - <<'PY'
from pathlib import Path
from tac.preflight import check_pmg_remote_dispatch_requires_geometry_escape
violations = check_pmg_remote_dispatch_requires_geometry_escape(
    repo_root=Path('.'), strict=True, verbose=True
)
print({'violations': len(violations)})
PY

[pmg-geometry-escape] OK: remote PMG dispatches carry geometry-escape review
{'violations': 0}

bash -n scripts/remote_lane_pmg_hotspot_c067_eval.sh
git diff --check -- src/tac/preflight.py \
  src/tac/tests/test_preflight_meta_bugs.py \
  scripts/remote_lane_pmg_hotspot_c067_eval.sh
```

## 2026-05-02T23:20Z - PMG residual economics gate

Local-only residual-economics planner implemented; no remote dispatch and no
score claim.

- planner: `experiments/plan_predictive_mask_residual_economics.py`
- focused tests: `src/tac/tests/test_plan_predictive_mask_residual_economics.py`
- artifact:
  `experiments/results/predictive_mask_residual_economics_20260502/predictive_mask_residual_economics.json`
- schema: `predictive_mask_residual_economics_v1`
- decision: `residual_rowspan_not_sub024_viable`

Key findings:

- Best byte point is PMG stride8: `102456` archive bytes, but decoded-mask
  disagreement remains `0.02741742451985677`.
- Best geometry point is PMG stride2 with all `600` pairs protected: exact
  mask reconstruction, but archive bytes are `635102`.
- No current PMG residual point satisfies both a `0.001` decoded-geometry trust
  threshold and the sub-0.24 unchanged-distortion byte target of `162650`.

Decision: row-span residual coding can be byte-small or geometry-safe, but not
both under the current representation. The next highest-EV implementation must
be a learned/pose-conditioned geometry-preserving mask decoder; residual bytes
are fallback budget and training-signal economics, not the primary path.

Verification:

```text
.venv/bin/python experiments/plan_predictive_mask_residual_economics.py

{"best_byte": 102456, "best_geometry": 635102, "decision": "residual_rowspan_not_sub024_viable", "joint_passes": 0, ...}

.venv/bin/python -m py_compile \
  experiments/plan_predictive_mask_residual_economics.py \
  src/tac/tests/test_plan_predictive_mask_residual_economics.py

.venv/bin/python -m pytest \
  src/tac/tests/test_plan_predictive_mask_residual_economics.py -q

3 passed in 0.09s

git diff --check -- experiments/plan_predictive_mask_residual_economics.py \
  src/tac/tests/test_plan_predictive_mask_residual_economics.py
```

## 2026-05-02T23:35Z - Lane 12 full-CUDA override forwarding hardening

Local-only infrastructure hardening; no remote dispatch and no score claim.

- Patched `scripts/remote_lane_nerv.sh` to forward `NERV_STEPS`,
  `NERV_EVAL_EVERY`, and `NERV_WEIGHT_DTYPE` into
  `experiments/train_nerv_mask.py` as `--steps`, `--eval-every`, and
  `--weight-dtype`.
- Patched `scripts/wave_omega_2_nerv_full_cuda.sh` so it no longer blocks just
  because the profile default has fewer steps than the requested full-CUDA
  burn. It now verifies the inner wrapper has override forwarding.
- Recorded the override values in `remote_lane_nerv.sh` provenance so a long
  burn can be audited against its intended full-CUDA recipe.
- Added durable `AGENTS.md` protocol and focused tests guarding this bug class.

This removes a concrete config/metabug: a nominal 60K-step NeRV burn could have
silently used the shorter profile default or been blocked by stale wrapper
logic. Lane 12 retraining remains closed until the L2 clearance packet and
promotion-threshold Alpha-Geo evidence exist.

Verification:

```text
bash -n scripts/remote_lane_nerv.sh
bash -n scripts/wave_omega_2_nerv_full_cuda.sh

.venv/bin/python -m py_compile \
  src/tac/tests/test_lane12_nerv_dependency_closure.py

.venv/bin/python -m pytest \
  src/tac/tests/test_lane12_nerv_dependency_closure.py -q

16 passed in 2.63s

git diff --check -- scripts/remote_lane_nerv.sh \
  scripts/wave_omega_2_nerv_full_cuda.sh \
  src/tac/tests/test_lane12_nerv_dependency_closure.py \
  AGENTS.md
```

## 2026-05-02T23:55Z - CDO1 decoded-mask overlay runtime contract

Local-only implementation and hardening; no remote dispatch and no score
claim.

Laplace's read-only review identified the next implementable mask-side path as
a byte-closed CDO1 overlay over a decoded base mask stream, not another global
PMG/CMG row-span replacement. The runtime contract is now first-class:

- `submissions/robust_current/inflate_renderer.py` now supports optional
  charged CDO1 sidecars: `masks.cdo1`, `masks.cdo1.zlib`,
  `masks.cdo1.xz`, and `masks.cdo1.br`.
- Runtime verifies CDO1 magic/version/schema, shape, sorted/non-overlapping
  run records, base mask SHA, selected-pixel count, reconstructed mask SHA,
  and preserves `_half_frame_only` metadata after overlay.
- `experiments/build_c067_decoded_delta_overlay_candidate.py` builds a
  deterministic byte-closed archive by adding a CDO1 sidecar to a base archive
  and records raw/compressed payload custody with `score_claim=false`.
- Archive allowlists were updated in `experiments/contest_auth_eval.py`,
  `src/tac/submission_archive.py`,
  `experiments/build_renderer_packed_payload_archive.py`, and
  `submissions/robust_current/unpack_renderer_payload.py`.
- The CDO1 planner now puts `run_count` and
  `reconstructed_mask_u8_sha256` in the payload header, so runtime can reject
  no-op/misapplied overlays.

Current real-data CDO1 plan remains fail-closed:

```text
.venv/bin/python experiments/plan_c067_decoded_delta_overlay_mask_topology.py --force

{"candidate_count": 4, "decision": "fail_closed_no_safe_overlay_specs", "dispatchable_candidate_count": 0, ...}
```

This is the correct outcome: CDO1 runtime closure is available, but the current
default donor/trust set overlaps exact-negative catastrophic pairs. The next
score-relevant work is to generate a learned or reversed-base geometry-preserve
CDO1 plan that repairs a smaller base stream toward the C067 basin without
repeating PMG/CMG global-replacement collapse.

Verification:

```text
.venv/bin/python -m py_compile \
  submissions/robust_current/inflate_renderer.py \
  experiments/plan_c067_decoded_delta_overlay_mask_topology.py \
  experiments/build_c067_decoded_delta_overlay_candidate.py \
  src/tac/submission_archive.py \
  src/tac/tests/test_cdo1_decoded_delta_overlay.py

.venv/bin/python -m pytest \
  src/tac/tests/test_cdo1_decoded_delta_overlay.py \
  src/tac/tests/test_plan_c067_decoded_delta_overlay_mask_topology.py -q

7 passed in 0.62s

.venv/bin/python -m pytest \
  src/tac/tests/test_renderer_packed_payload.py \
  src/tac/tests/test_remote_auth_eval_hardening.py -q

66 passed in 1.80s

.venv/bin/python -m pytest \
  src/tac/tests/test_build_c067_postdecode_mask_repair_candidate.py \
  src/tac/tests/test_build_charged_mask_grammar_candidate.py -q

29 passed, 1 warning in 0.69s

git diff --check -- submissions/robust_current/inflate_renderer.py \
  experiments/plan_c067_decoded_delta_overlay_mask_topology.py \
  experiments/build_c067_decoded_delta_overlay_candidate.py \
  src/tac/submission_archive.py \
  src/tac/tests/test_cdo1_decoded_delta_overlay.py \
  experiments/contest_auth_eval.py \
  submissions/robust_current/unpack_renderer_payload.py \
  experiments/build_renderer_packed_payload_archive.py
```

## 2026-05-03T00:10Z - Reversed-base CDO1 economics screen

Local-only planner implemented and run; no remote dispatch and no score claim.

- planner: `experiments/plan_c067_reversed_base_cdo1_overlay_economics.py`
- focused tests:
  `src/tac/tests/test_plan_c067_reversed_base_cdo1_overlay_economics.py`
- artifact:
  `experiments/results/c067_reversed_base_cdo1_overlay_economics_20260502/c067_reversed_base_cdo1_overlay_economics.json`
- schema: `c067_reversed_base_cdo1_overlay_economics_v1`
- decision: `byte_headroom_but_geometry_blocked`

Key findings:

- Best byte-headroom family is `cmg3_nonzero_top1` plus a small CDO1 overlay
  selected by trace/waterfill pairs. It estimates `132996` to `144036` archive
  bytes, which is below the unchanged-distortion sub-0.24 byte gate.
- That byte headroom is not dispatchable: residual decoded-mask disagreement
  remains `0.03521` to `0.03541`, far above the `0.001` geometry trust gate.
- Exact geometry repair is possible but byte-regressive: the smallest full
  repair-to-C067 plan is `cmg3a_body200__full_repair...`, estimated `455683`
  bytes, `+179469` bytes versus C067.
- No reversed-base CDO1 candidate passes either the joint sub-0.24+geometry gate
  or the joint sub-0.30+geometry gate.

Decision: CDO1 is now a correct runtime/contract primitive, but the available
CMG3/CMG3A bases are not the right compressed latent by themselves. The next
mask-side path must learn or construct a base whose residual to the C067 basin
is much lower before overlay, rather than repairing a heavily collapsed global
grammar with sparse waterfill atoms.

Verification:

```text
.venv/bin/python -m py_compile \
  experiments/plan_c067_reversed_base_cdo1_overlay_economics.py \
  src/tac/tests/test_plan_c067_reversed_base_cdo1_overlay_economics.py

.venv/bin/python -m pytest \
  src/tac/tests/test_plan_c067_reversed_base_cdo1_overlay_economics.py -q

2 passed in 0.16s

.venv/bin/python experiments/plan_c067_reversed_base_cdo1_overlay_economics.py --force

{"byte_headroom_geometry_blocked_count": 4, "candidate_count": 20, "decision": "byte_headroom_but_geometry_blocked", "geometry_safe_byte_regressive_count": 4, "joint_sub0240_geometry_count": 0, "joint_sub0300_geometry_count": 0, ...}
```

## 2026-05-02T23:04Z - Swarm recovery triage and CDO1 lower-bound gate

Recovered the compaction-interrupted swarm state and closed the read-only
subagents after ingesting their findings. Both independent reviews converged
on the same control decision:

- CDO1/reversed-base mask topology is the only current non-retraining lane
  with sub-0.30 byte leverage, but available CMG3/CMG3A bases fail geometry.
- SJ-KL diagnostics have the right component sign but bad payload economics.
- Renderer self-compression is blocked by the absence of a real non-surrogate
  trained JointFrameGenerator export; current QBF1 screens are source-renderer
  surrogates or byte-regressive.

I extended the reversed-base CDO1 economics planner with a deterministic
`geometry_threshold_longest_runs_to_residual_gate` lower-bound policy. This
selects whole changed runs in longest-run order until the residual decoded-mask
disagreement drops below the configured gate. It gives a concrete lower bound
on charged CDO1 overlay bytes needed to make each smaller base geometry-safe.

Real-data result:

```text
.venv/bin/python experiments/plan_c067_reversed_base_cdo1_overlay_economics.py --force

{"candidate_count": 24, "joint_sub0240_geometry_count": 0,
 "joint_sub0300_geometry_count": 0, "byte_headroom_geometry_blocked_count": 4,
 "geometry_safe_byte_regressive_count": 8,
 "decision": "byte_headroom_but_geometry_blocked"}
```

Key lower-bound numbers:

- `cmg3a_body200` needs about `97340` LZMA CDO1 bytes to reach residual
  disagreement `0.000999976264`, for estimated archive `356819` bytes.
- `cmg3_nonzero_top2` needs about `158172` overlay bytes, estimated archive
  `382700` bytes.
- `cmg3_nonzero_top1` needs about `280164` overlay bytes, estimated archive
  `408764` bytes.

Decision: current smaller CMG3/CMG3A bases cannot be repaired into the C067
basin cheaply enough. The next mask-topology work must improve the compressed
base itself, then use CDO1 as a small trust-region correction layer.

I also fixed a stale-evidence bug in
`experiments/plan_c067_bigmove_nontrain_candidates.py`: the planner had not
mapped `save12k_exact_trace_pair_waterfill_budget4000` to its already-harvested
exact negative, so it could re-rank that policy as dispatchable from first-order
benefit alone. The refreshed triage now records it as exact-negative with score
`1.5721465219411697`, archive `243422` bytes, and `dispatchable=false`.

The big-move triage now ingests CDO1 lower-bound economics and renderer export
readiness directly. It reports all current structural coverage while leaving
every blocked candidate in `no_dispatch` state:

```text
.venv/bin/python experiments/plan_c067_bigmove_nontrain_candidates.py

{"candidate_count": 35, ...}

coverage:
  cdo1_reversed_base_mask_topology=true
  renderer_self_compression=true
  multiresolution_multimask_reconciliation=true
  scorer_weighted_mask_topology_repair_atoms=true
  sjkl_existing_active_diagnostic_not_duplicated=true
```

Hardened bug classes:

- stale exact-negative postdecode repair reappearing as dispatchable;
- duplicate micro-mask candidate IDs in ranked triage;
- CDO1 byte-headroom rows treated as eval-ready without a residual-geometry
  lower-bound gate;
- renderer source-surrogate exports being mistaken for trained renderer
  self-compression.

Verification:

```text
.venv/bin/python -m py_compile \
  experiments/plan_c067_reversed_base_cdo1_overlay_economics.py \
  src/tac/tests/test_plan_c067_reversed_base_cdo1_overlay_economics.py \
  experiments/plan_c067_bigmove_nontrain_candidates.py \
  src/tac/tests/test_plan_c067_bigmove_nontrain_candidates.py

.venv/bin/python -m pytest \
  src/tac/tests/test_plan_c067_reversed_base_cdo1_overlay_economics.py \
  src/tac/tests/test_plan_c067_bigmove_nontrain_candidates.py -q

6 passed

.venv/bin/python -m pytest \
  src/tac/tests/test_plan_trained_renderer_export_unlock.py \
  src/tac/tests/test_preflight_trained_renderer_transplant.py \
  src/tac/tests/test_build_blockfp_c067_archive.py \
  src/tac/tests/test_build_mixed_qzs_block_candidate.py -q

25 passed in 3.93s
```

## 2026-05-03T00:33Z - Packed CDO1 archive-shape hardening

The C067/Apogee frontier archive is a single stored `p` member, not an expanded
`renderer.bin`/`masks.mkv`/`optimized_poses.bin` ZIP. The initial CDO1 builder
could build overlay archives only from expanded runtime members. That was a
deployment-shape bug class: a future byte/geometry-safe CDO1 payload could have
failed before exact eval or been measured in the wrong archive container.

Patch landed:

- `experiments/build_c067_decoded_delta_overlay_candidate.py` now accepts packed
  base archives by unpacking them through
  `submissions/robust_current/unpack_renderer_payload.py`, adding the charged
  CDO1 logical member, and optionally repacking the output as a single payload
  via `experiments/build_renderer_packed_payload_archive.py`.
- New CLI flags: `--pack-output-payload`, `--packed-payload-member-name`,
  `--packed-payload-format`, and `--brotli-quality`.
- The manifest records the base runtime-member contract, unpack summary,
  expanded-candidate SHA/bytes, packed payload report, final archive SHA/bytes,
  and `score_claim=false`.
- `AGENTS.md` now records the durable rule: packed CDO1 candidates must round
  trip through the reviewed pack/unpack path before any remote dispatch.

Verification:

```text
.venv/bin/python -m py_compile \
  experiments/build_c067_decoded_delta_overlay_candidate.py \
  src/tac/tests/test_cdo1_decoded_delta_overlay.py

.venv/bin/python -m pytest src/tac/tests/test_cdo1_decoded_delta_overlay.py -q

4 passed in 0.52s
```

Score impact: none yet. This is a permanent preflight/runtime-shape fix, not a
new score claim. The CDO1 economics gate still says existing CMG3/CMG3A bases
are geometry-safe only after byte-regressive overlays; the next score-moving
mask work must create a better compressed base before this packed overlay path
is worth exact CUDA spend.

## 2026-05-03T00:45Z - Trained-renderer scanner exact-negative correction

The renderer self-compression/export unlock scanner was too broad in the wrong
direction: it could say no non-surrogate QZS3/MQZ/QBF export candidate existed
even though real Q-FAITHFUL artifacts had been harvested under
`lane_q_faithful_retrain_20260501`. That is a blocker-class bug because it
turns "present but exact-negative or not preflighted" into "missing", which can
send future agents toward stale H100 retraining instead of the real geometry
failure.

Patch landed:

- `experiments/plan_trained_renderer_export_unlock.py` now scans
  `experiments/results/lane_q_faithful_retrain_20260501` by default.
- QFAI is a known trained renderer magic. Raw QFAI exports are classified as
  present but requiring packing/transcoding plus transplant preflight.
- Packed Q-FAITHFUL QZS3/QP-style artifacts are classified with archive bytes,
  renderer SHA, payload format, pose codec, stacking break-even targets, and
  nearby exact CUDA records when matching harvested eval JSON exists.
- The readiness verdict now distinguishes "no non-surrogate trained-renderer
  archive passed preflight" from "no non-surrogate export exists".
- `AGENTS.md` records the durable rule: trained-renderer scanners must classify
  real Q-FAITHFUL artifacts as present/blocked or present/exact-negative, never
  missing.

Current regenerated plan:

```text
.venv/bin/python experiments/plan_trained_renderer_export_unlock.py

candidate_count=131
non_surrogate_candidate_count=32
h100_ready_preflight_count=0
readiness.blockers=["no non-surrogate trained-renderer archive passed preflight"]
```

Examples include Q-FAITHFUL packed exports around `273048` to `276651` bytes.
Those are byte-interesting, but exact CUDA matches such as the RP2/QPose14
snapshot score `8.420915711675079`, so they are not dispatchable as current
score candidates. The next renderer lane is successor training with a proven
nonzero pose/geometry contract, not a rerun of these collapsed snapshots.

Verification:

```text
.venv/bin/python -m py_compile \
  experiments/plan_trained_renderer_export_unlock.py \
  src/tac/tests/test_plan_trained_renderer_export_unlock.py

.venv/bin/python -m pytest src/tac/tests/test_plan_trained_renderer_export_unlock.py -q

5 passed in 0.48s
```

## 2026-05-03T01:15Z - Q-FAITHFUL successor pose/geometry preflight gate

Added a local-only preflight gate for future Q-FAITHFUL successor burns:
`experiments/preflight_qfaithful_successor_geometry_contract.py`.

The gate fails closed unless provenance plus exact archive bytes prove:
nonzero deployed pose stream, pose pair count, pose SHA, `eval_roundtrip=True`,
`zero_pose_fallback_allowed=false`, runtime-readable QFAI/QZS3 renderer bytes,
charged geometry member preservation, runtime geometry consumption when zoom or
foveation is configured, and `score_claim=false`. It never loads scorers and
never dispatches GPU work.

Current generated report:

```text
experiments/results/qfaithful_successor_geometry_contract_20260503/qfaithful_successor_geometry_contract_preflight.json
verdict=blocked_no_h100_dispatch
training_dispatch_allowed=false
remote_gpu_dispatch_performed=false
score_claim=false
blockers=[
  "eval_roundtrip_true_not_proven",
  "training_pose_contract_missing",
  "zoom_warp_geometry_not_consumed_by_runtime"
]
```

This is a no-score local contract gate. The existing 2146 direct zoom archive is
runtime-readable QZS3 and has nonzero decoded poses, but it remains blocked for
a successor H100 burn because the provenance does not yet prove the training
pose/roundtrip/geometry-consumption contract required by `AGENTS.md`.

## 2026-05-03T00:28Z - Compact SJ-KL Q6/min-RPK1 diagnostic dispatch

Fermat's SJ-KL shrink worker landed a real byte-closed archive candidate:

- archive: `experiments/results/sjkl_c067_shrink_q6_minrpk1_20260502T_worker/pack/archive.zip`
- bytes: `276999`, delta `+785` vs C067
- archive SHA: `7977f3ee5d6d744f5818d358c13424a1f19f6bf8b6604af56d37c13912159922`
- `sjkl.bin`: `250` bytes, SHA `13f605bfd9ad950807d410c8371f20fd2b1c3d9c04bb59cc6bf07d474dcc78bb`
- selected pairs: `153,156,37,111,382,77,326,579,87,133,140,172,441,136,347,452`
- formula-only rate delta: `+0.0005226992782009045`

Break-even math is unfavorable but close enough for a diagnostic: the exact
trace-top16 ancestor had component-positive evidence but only about `0.000286`
score benefit before rate. The compact Q6 variant must roughly double that
component benefit to move the frontier. Because SJ-KL is the only current
near-frontier component-positive family, I queued one RTX PRO exact-CUDA
canary, not a T4 promotion:

- job: `exact_eval_sjkl_c067_q6_minrpk1_rtxprodiag_20260502T2327Z`
- state: `.omx/state/exact_eval_sjkl_c067_q6_minrpk1_rtxprodiag_20260502T2327Z_batch_jobs.json`
- source manifest: `.omx/state/exact_eval_sjkl_c067_q6_minrpk1_rtxprodiag_20260502T2327Z_manifest.json`
- dispatch claim: `sjkl_c067_q6_minrpk1_canary`
- env: `SJKL_REQUIRE_APPLIED=1`
- evidence grade until harvest: `queued_exact_cuda_diagnostic`, `score_claim=false`

Local verification before queue:

```text
.venv/bin/python -m pytest src/tac/tests/test_sjkl_basis.py src/tac/tests/test_build_sjkl_c067_archive.py src/tac/tests/test_plan_sjkl_trace_benefit_allocator.py -q
35 passed in 1.36s
```

The first submit attempt failed before spend because the Lightning source
manifest omitted the baseline JSON referenced in queue metadata. The second
staging manifest included that baseline JSON explicitly and passed remote
manifest verification: `file_count=1348`, `total_bytes=23303290`.

## 2026-05-03T00:31Z - Python loop hotpath profile for atom-compiler speed

Generated a deterministic engineering-only loop profile to guide vectorization
work for the next mask-atom compilers:

```text
.venv/bin/python experiments/profile_python_loop_hotpaths.py \
  --output-json experiments/results/python_loop_hotpaths_20260503/python_loop_hotpaths.json \
  --limit 40

loop_count=4606
score_claim=false
```

Top static hotpaths:

- `experiments/plan_cmg3_pixel_lagrangian_atoms.py:864`
- `experiments/plan_predictive_mask_hotspot.py:380`
- `experiments/plan_predictive_mask_hotspot.py:381`
- `src/tac/mask_entropy_coder.py:319`
- `experiments/build_cmg3_adaptive_runs_candidate.py:404`

Interpretation: if we need faster local search for geometry-preserving mask
topology, the first optimization target is the CMG3/predictive-mask atom
planning surface. This is engineering profile evidence only; it is not score
evidence and does not justify GPU dispatch.

Verification:

```text
.venv/bin/python -m py_compile experiments/profile_python_loop_hotpaths.py \
  src/tac/tests/test_profile_python_loop_hotpaths.py
.venv/bin/python -m pytest src/tac/tests/test_profile_python_loop_hotpaths.py -q

3 passed in 0.08s
```

## 2026-05-03T00:36Z - C067 atom-response taxonomy table

Halley generated and I verified a deterministic C067 atom-response table:

- JSON:
  `experiments/results/c067_atom_response_table_20260503/c067_atom_response_table.json`
- Markdown:
  `experiments/results/c067_atom_response_table_20260503/c067_atom_response_table.md`
- `artifact_count=47`, `family_count=10`, `score_claim=false`

I added stable JSON aliases (`classification`, `exact_count`,
`best_byte_delta`, `best_nonrate_delta`, `collapse_count`, and
`evidence_grade`) so downstream jq/planner code can consume the same fields the
human Markdown table displays.

Machine-readable family verdicts:

- `SJ-KL residual`: `component_positive_byte_regressive`, best non-rate delta
  `-0.0002866351587692403`, best byte delta `+1241`; keep shrinking/selecting
  SJ-KL atoms and exact-screen only compact variants.
- `CMG mask grammar`, `PMG hotspot mask grammar`, `micro mask reencode`,
  `multimask reconciliation`, `postdecode mask repair`, `renderer compression`,
  and `hotspot geometry`: `collapse_or_exact_negative`; do not dispatch the
  same family again without a geometry-escape proof.
- `CDO1 decoded-mask overlay`: `planning_only`; do not dispatch until residual
  geometry, runtime closure, and byte screen pass together.
- `fixed-slice segment mix`: diagnostic-positive only on non-T4 for one row;
  do not promote without matching T4 confirmation.

Verification:

```text
.venv/bin/python -m py_compile experiments/build_c067_atom_response_table.py \
  src/tac/tests/test_build_c067_atom_response_table.py
.venv/bin/python -m pytest src/tac/tests/test_build_c067_atom_response_table.py -q
.venv/bin/python experiments/build_c067_atom_response_table.py \
  --output-dir experiments/results/c067_atom_response_table_20260503

3 passed in 0.08s
artifact_count=47 family_count=10 score_claim=false
```

## 2026-05-02T23:59Z - SJ-KL sibling breakthrough diagnostic and T4 promotion queue

The compact SJ-KL Q6/min-RPK1 archive was repacked into a top-level sibling
layout instead of modifying the packed C067 `p` payload. This preserves the
C067 payload bytes exactly and charges `sjkl.bin` as an archive member consumed
by the runtime under `SJKL_REQUIRE_APPLIED=1`.

Implementation and local verification:

- `experiments/build_sjkl_c067_archive.py` supports
  `--archive-layout top_level_sibling`.
- `src/tac/tests/test_build_sjkl_c067_archive.py` covers preservation of the
  source payload plus charged sibling `sjkl.bin`.
- Focused verification:

```text
.venv/bin/python -m py_compile experiments/build_sjkl_c067_archive.py \
  src/tac/tests/test_build_sjkl_c067_archive.py experiments/build_sjkl_residual.py
.venv/bin/python -m pytest src/tac/tests/test_build_sjkl_c067_archive.py \
  src/tac/tests/test_inflate_renderer_sjkl_runtime.py -q
.venv/bin/python -m pytest src/tac/tests/test_preflight_trained_renderer_transplant.py \
  src/tac/tests/test_build_blockfp_c067_archive.py \
  src/tac/tests/test_plan_trained_renderer_export_unlock.py \
  src/tac/tests/test_qbf1_renderer_codec.py -q

14 passed
22 passed
```

Built archive:

- Path:
  `experiments/results/sjkl_c067_q6_top_level_sibling_20260503/archive.zip`
- Bytes: `276556`, delta `+342` versus C067.
- SHA-256:
  `a576960be12fdcec1cc76257d5a49cd4102476c7e461150847c845ba0cceab6d`
- `sjkl.bin`: `250` bytes.
- Formula-only rate cost versus C067: `+0.0002277237619677826`.

RTX PRO exact CUDA diagnostic:

- Job:
  `exact_eval_sjkl_c067_q6_sibling_rtxprodiag_20260502T2346Z`
- Local artifact:
  `experiments/results/lightning_batch/exact_eval_sjkl_c067_q6_sibling_rtxprodiag_20260502T2346Z/contest_auth_eval.adjudicated.json`
- Score: `0.31556098781392106`.
- Delta versus C067 A++ T4 frontier: `-0.000056042970561276384`.
- Archive bytes/SHA matched the candidate exactly.
- Components: PoseNet `0.00049519`, SegNet `0.00061044`.
- Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition, CUDA, `600` samples.
- Evidence grade: exact CUDA `A score-grade`, not A++ promotion, because this
  hardware is not T4/equivalent.

This is the first post-C067 exact CUDA diagnostic in this tranche that actually
beats the current frontier. I therefore queued a byte-identical T4 promotion
rerun immediately:

- Job:
  `exact_eval_sjkl_c067_q6_sibling_t4_20260502T2357Z`
- State:
  `.omx/state/exact_eval_sjkl_c067_q6_sibling_t4_20260502T2357Z_batch_jobs.json`
- Machine: `g4dn.xlarge` / T4.
- Archive bytes/SHA:
  `276556`,
  `a576960be12fdcec1cc76257d5a49cd4102476c7e461150847c845ba0cceab6d`.
- T4-specific inflate env:
  `INFLATE_TORCH_SPEC=torch==2.5.1+cu124`,
  `UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124`,
  `UV_INDEX_STRATEGY=unsafe-best-match`,
  `INFLATE_REQUIRE_CUDA=1`.
- Status at this ledger update: `Pending`, zero cost.

The initial T4 submit failed closed before spend because the exact-eval wrapper
requires a CUDA-12-compatible inflate Torch pin for g4dn/T4. This is the desired
metabug protection; the corrected submit includes the explicit cu124 pin.

One additional distinct SJ-KL sibling hedge is queued, not as a duplicate:

- Job:
  `exact_eval_sjkl_c067_consensus32_g0046875_sibling_rtxprodiag_20260502T2354Z`
- Archive:
  `experiments/results/sjkl_c067_consensus_top32_g0046875_sibling_20260503/archive.zip`
- Bytes/SHA:
  `276617`,
  `8e10d635d55edaaf809d28b1a4259a891d8abb4630ca874b973d77775ffcbdbf`.
- Status at this ledger update: `Pending`, zero cost.

Decision rule:

- If the T4 rerun confirms the RTX PRO improvement, promote this archive as the
  new A++ frontier and update the claim matrix/submission packet.
- If T4 regresses, preserve both exact packets and treat SJ-KL as
  hardware-sensitive until the component traces explain the delta.
- If the consensus32 diagnostic starts before T4 completes, harvest it, but do
  not let it delay the T4 promotion path.

## 2026-05-02T23:59Z - Block-FP trained-renderer transplant unblocked locally, but not a byte lever

Linnaeus reviewed Block-FP/readiness read-only, then I refreshed the local
gates. The implementation surface is sound:

- `experiments/build_blockfp_c067_archive.py`
- `experiments/preflight_trained_renderer_transplant.py`
- `experiments/plan_trained_renderer_export_unlock.py`
- `src/tac/qbf1_renderer_codec.py`
- QBF1 runtime branch in `submissions/robust_current/inflate_renderer.py`

The refreshed unlock scan still says no preflight-passed non-surrogate export
was available by default, but it surfaced a small accepted-format Q-FAITHFUL
QZS3 renderer export. I ran the transplant preflight:

```text
.venv/bin/python experiments/preflight_trained_renderer_transplant.py \
  --source-archive experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip \
  --renderer-export experiments/results/lane_q_faithful_retrain_20260501/remote_artifacts/postprocess_fixed_snapshot_20260501T2146Z_fix1/qzs3_pr64_qpose14_unpacked_20260501T2159Z/renderer.bin \
  --output-dir experiments/results/trained_renderer_blockfp_preflight_20260503_qfaithful_qzs3_2146 \
  --block-sizes 64,128,256,512,1024 \
  --force
```

Result:

- Preflight ready: `true`.
- Renderer export SHA:
  `00ce395fc6495a47d26e3844537859fdc2a1bc38d121ea5f4fb2610179d26e46`.
- Best QBF1 archive:
  `experiments/results/trained_renderer_blockfp_preflight_20260503_qfaithful_qzs3_2146/trained_qbf1_b0512/archive.zip`
- Bytes/SHA:
  `283432`,
  `6174424516a93aa35852cd4dc22b9132516fc38f824b77562b3f91b3f55f58dc`.
- Delta versus C067: `+7218` bytes.
- Score claim: `false`.

Conclusion: Block-FP transplant plumbing is now locally unblocked for a
non-surrogate export, but this specific archive is byte-negative. It should
only be exact-screened as a learned-renderer component-improvement diagnostic,
not as a rate-only path to sub-0.300. The immediate promotion priority remains
the q6 SJ-KL T4 rerun.

## 2026-05-03T00:05Z - SJ-KL consensus32 diagnostic harvested, weaker than q6

The lower-gain consensus sibling hedge completed on RTX PRO and was harvested
through the state-derived Lightning path:

- Job:
  `exact_eval_sjkl_c067_consensus32_g0046875_sibling_rtxprodiag_20260502T2354Z`
- Local artifact:
  `experiments/results/lightning_batch/exact_eval_sjkl_c067_consensus32_g0046875_sibling_rtxprodiag_20260502T2354Z/contest_auth_eval.adjudicated.json`
- Archive bytes/SHA:
  `276617`,
  `8e10d635d55edaaf809d28b1a4259a891d8abb4630ca874b973d77775ffcbdbf`
- Exact CUDA diagnostic score:
  `0.31560031673416666`
- Delta versus C067 A++ T4:
  `-0.000016714050315669304`
- Components:
  PoseNet `0.00049517`, SegNet `0.00061044`, `600` samples.
- Hardware:
  NVIDIA RTX PRO 6000 Blackwell Server Edition; not T4/equivalent.

This confirms that the SJ-KL sibling family is producing real component-safe
improvements, but the q6 sibling remains the better promotion candidate:
`0.31556098781392106` diagnostic versus `0.31560031673416666` diagnostic.
No consensus32 T4 should be submitted until the q6 T4 promotion result is
harvested or fails to confirm.

Prepared promotion packet command surface for a q6 T4-confirmed artifact:

```text
.venv/bin/python scripts/build_contest_submission_packet.py \
  --artifact-dir experiments/results/lightning_batch/<confirmed_q6_t4_job> \
  --expected-archive-sha256 a576960be12fdcec1cc76257d5a49cd4102476c7e461150847c845ba0cceab6d \
  --expected-archive-size-bytes 276556 \
  --expected-samples 600 \
  --planner-ledger .omx/research/contest_faithful_swarm_execution_20260502_codex.md \
  --next-action-tranche docs/runbooks/contest_faithful_submission_next_tranche_20260502.md
```

The primary q6 T4 job remains the active score authority candidate:
`exact_eval_sjkl_c067_q6_sibling_t4_20260502T2357Z`.

## 2026-05-03T00:25Z - q6 T4 truth landed negative; C067 remains A++ frontier

The q6 SJ-KL sibling improved on RTX PRO diagnostic but did not confirm on
T4/equivalent hardware. The primary T4 job was harvested through the
state-derived Lightning path:

- Job:
  `exact_eval_sjkl_c067_q6_sibling_t4_20260502T2357Z`
- Local artifact:
  `experiments/results/lightning_batch/exact_eval_sjkl_c067_q6_sibling_t4_20260502T2357Z/contest_auth_eval.adjudicated.json`
- Archive bytes/SHA:
  `276556`,
  `a576960be12fdcec1cc76257d5a49cd4102476c7e461150847c845ba0cceab6d`
- Exact CUDA T4 score:
  `0.3158419419767293`
- Delta versus C067 A++ T4:
  `+0.0002249111922469904`
- Components:
  PoseNet `0.00049633`, SegNet `0.00061244`, `600` samples.
- Hardware:
  Tesla T4, `gpu_t4_match=true`.
- Runtime contract:
  strict SJ-KL apply proof was present in `auth_eval.log`.

This is A++ exact negative evidence for this exact SJ-KL payload and selection.
It is not a broad SJ-KL kill: the RTX PRO diagnostic improvement was real, but
the T4 result says the margin is too small and/or hardware-sensitive to promote.
Future SJ-KL work must either shrink the payload below the current byte burden
or produce a component benefit materially larger than the T4 drift observed
here.

The duplicate T4 hedge
`exact_eval_sjkl_c067_q6_sibling_t4aws_20260503T0002Z` was stopped after the
primary T4 truth landed. Final refreshed state was `Stopped`, cost
approximately `$0.16306111`, machine `T4`. The active dispatch claims for q6
primary, q6 hedge, and consensus32 diagnostic were closed with non-frontier or
duplicate-stop statuses.

I built a metadata-only non-frontier packet for this negative so the artifact
is reproducible without implying a score/rank claim:

- Packet directory:
  `experiments/results/submission_packet_sjkl_q6_t4_negative_20260503/`
- Manifest:
  `submission_packet_manifest.json`
- Checklist:
  `submission_packet_checklist.md`
- Captured stdout:
  `.omx/state/submission_packet_sjkl_q6_t4_negative_20260503.stdout.json`

The packet builder now self-protects against a class of silent no-op charged
residual bugs: any archive containing top-level `sjkl.bin` requires
`auth_eval.log` proof that the payload loaded, applied to
`JointFrameGenerator fake1`, and passed `SJKL_REQUIRE_APPLIED`. Focused packet
tests are green: `8 passed`.

## 2026-05-03T00:27Z - High-risk postdecode repair diagnostic queued on RTX PRO

The exact C067 sub-`0.300` unchanged-distortion crossing requires about
`23455` bytes of additional rate savings, so q6-class byte/pose polish is not
enough. I built one aggressive postdecode AMR1 repair candidate to test whether
larger, trace-guided repair can preserve enough geometry while keeping a real
mask-byte reduction.

The first build attempt against the packed C067 deploy archive failed closed:
the builder rejected packed member `p` as an unexpected runtime member. I kept
that guard intact and rebuilt from the unpacked C067 runtime directory:

- Base runtime:
  `experiments/results/c067_fixedslice_unpacked_runtime_20260502/unpacked`
- Lossy mask archive:
  `experiments/results/c067_micro_mask_reencode_plan_20260502/builds/c067_micro_av1_mask_reencode_save12k/archive.zip`
- Output directory:
  `experiments/results/c067_postdecode_mask_repair_candidate_20260503/save12k_exact_trace_pair_waterfill_budget16000/`
- Candidate archive:
  `archive.zip`
- Candidate bytes/SHA:
  `255342`,
  `cfb1861e3288d2bf42f7bf2ef5a1368efb682a4d8dbe783207fed3bef9b9e90e`
- Rate-only delta versus C067:
  approximately `-0.022992775510261697`.
- Repair member:
  `alpha4_residual_repair.amr1.xz`, `15692` compressed bytes,
  `76079` raw AMR1 bytes.
- Selected repair:
  `23` atoms, `12545` pixels, `6608` runs.

This is explicitly high-risk diagnostic evidence-seeking, not a promotion
claim. Smaller postdecode repairs have collapsed before; this candidate asks
whether a materially larger water-fill repair budget escapes the collapse
while preserving enough of the mask byte win to matter for sub-`0.300`.

The diagnostic was staged with a reproducible Lightning source/artifact
manifest and queued on RTX PRO:

- Manifest:
  `.omx/state/exact_eval_c067_postdecode_repair_save12k_pairwaterfill16k_rtxprodiag_20260503T0021Z_manifest.json`
- Manifest verification:
  `REMOTE_MANIFEST_VERIFY OK`, `1351` files, `23399323` bytes,
  manifest SHA `88ace989b81e0c362227fc8be7e691e97445bcbfbd72ec7558df1de5af08dcb4`.
- Job:
  `exact_eval_c067_postdecode_repair_save12k_pairwaterfill16k_rtxprodiag_20260503T0021Z`
- Machine:
  `g7e.4xlarge` / RTX PRO.
- Initial status:
  `Pending`, zero cost.

Decision rule:

- If this diagnostic is component-safe and materially improves C067, rerun the
  identical archive bytes on T4/equivalent immediately.
- If it collapses, preserve the component trace and convert the result into a
  sharper learned/multimask/geometry-preserving selector; do not spend T4 on
  the failed implementation.
- If it is close but not enough, use its per-pair trace as atom-response input
  for a learned or Lagrangian water-fill selector instead of hand-expanding
  AMR1 repair blindly.

## 2026-05-03T00:42Z - Postdecode repair negative, packet hardening, and CDO1 local-overlay dispatch

The high-risk `save12k_pair_waterfill_budget16000` postdecode AMR1 repair
diagnostic completed on RTX PRO and is a scoped exact negative:

- Job:
  `exact_eval_c067_postdecode_repair_save12k_pairwaterfill16k_rtxprodiag_20260503T0021Z`
- Local artifact:
  `experiments/results/lightning_batch/exact_eval_c067_postdecode_repair_save12k_pairwaterfill16k_rtxprodiag_20260503T0021Z/contest_auth_eval.adjudicated.json`
- Archive bytes/SHA:
  `255342`,
  `cfb1861e3288d2bf42f7bf2ef5a1368efb682a4d8dbe783207fed3bef9b9e90e`
- Exact CUDA diagnostic score:
  `1.556012052216008`
- Components:
  PoseNet `0.12643525`, SegNet `0.00261556`, `600` samples.
- Decision:
  no T4 promotion. The measured implementation saved bytes but still collapsed
  pose geometry, so the trace is selector evidence only.

Permanent packet hardening now covers the charged-runtime bug class exposed by
the current wave. `scripts/build_contest_submission_packet.py` detects charged
payloads both as top-level ZIP members and inside packed RPK1 payloads, then
requires `auth_eval.log` proof for the three charged contracts currently in
play:

- SJ-KL: loaded, applied to `JointFrameGenerator fake1`, strict contract passed.
- CDO1: `Applied CDO1 decoded-mask overlay`.
- AMR1: `Applied Alpha residual repair`.

This prevents a submission packet from silently blessing an archive whose
charged side information was packaged but never consumed. Focused packet tests
are green after the expanded contract checks: `13 passed`.

CDO1 planning was also tightened. The decoded-delta topology planner previously
used all catastrophic exact-negative families as a global veto, which over-vetoed
local CDO1 overlays whenever unrelated global replacement lanes touched the same
pairs. The planner now has a controlled
`--catastrophic-family-groups` filter so same-family veto remains strict while
local overlay candidates can be inspected without pretending unrelated family
failures are identical evidence. Focused CDO1 planner tests are green:
`5 passed`.

The same-family-veto CDO1 run found no dispatchable sub-`0.300` candidate, but
it did produce three safe local overlay specs under the `4 KB` trust budget:

- `cmg3_rowspan_stride1`: `2244` compressed payload bytes, selected pairs
  `[79, 153, 212, 216, 230]`.
- `cmg3_nonzero_top2`: `3360` compressed payload bytes.
- `cmg3a_body200`: `3592` compressed payload bytes.

I built the smallest local-overlay archive over C067 as a runtime-primitive
diagnostic, not as a byte-screen-positive frontier candidate:

- Spec:
  `experiments/results/c067_decoded_delta_overlay_mask_topology_20260503_samefamily_veto4k/safe_specs/cmg3_rowspan_stride1/overlay_spec.json`
- Archive:
  `experiments/results/c067_cdo1_local_overlay_candidate_20260503/cmg3_rowspan_stride1_pairs79_153_212_216_230/archive.zip`
- Archive bytes/SHA:
  `279393`,
  `a385a60bac628010040eae4db7dbb5f469402072d069586f49b21798266606ff`
- Byte delta versus C067:
  `+3179`; break-even component improvement is about `0.002116`.

The CDO1 diagnostic is staged and submitted on RTX PRO with archive custody,
remote supply-chain scan, source manifest, component trace, and adjudication:

- Manifest:
  `.omx/state/exact_eval_c067_cdo1_local_overlay_pairs5_rtxprodiag_20260503T0038Z_manifest.json`
- Manifest verification:
  `1370` files, `208344418` bytes, manifest SHA
  `26e7993a84c1642937f437a4808e05126198e85ab11d9d6d5e85f6d4c989c308`.
- Job:
  `exact_eval_c067_cdo1_local_overlay_pairs5_rtxprodiag_20260503T0038Z`
- Initial status:
  `Pending`, zero cost at first refresh.

Decision rule:

- If CDO1 materially improves components enough to overcome the `3179` byte
  penalty, rerun identical bytes on T4/equivalent.
- If it is neutral or negative, keep CDO1 as a correct charged-runtime primitive
  but stop testing local overlays that are byte-regressive before scorer effects.
- If it shows localized component benefit but not enough net score, feed the
  per-pair trace back into the CDO1/reversed-base economics planner rather than
  broadening the overlay by hand.

## 2026-05-03T00:55Z - CDO1 local-overlay diagnostic harvested; reversed-base CDO1 queued

The scoped CDO1 local-overlay runtime diagnostic over C067 has exact CUDA RTX
PRO evidence and is a measured-implementation negative:

- Job:
  `exact_eval_c067_cdo1_local_overlay_pairs5_rtxprodiag_20260503T0038Z`
- Local artifact:
  `experiments/results/lightning_batch/exact_eval_c067_cdo1_local_overlay_pairs5_rtxprodiag_20260503T0038Z/contest_auth_eval.adjudicated.json`
- Archive bytes/SHA:
  `279393`,
  `a385a60bac628010040eae4db7dbb5f469402072d069586f49b21798266606ff`
- Exact CUDA diagnostic score:
  `0.3402091350029083`
- Components:
  PoseNet `0.00085165`, SegNet `0.00061888`, `600` samples.
- Delta versus C067 A++:
  `+0.024592104218425948`.
- Decision:
  no T4 promotion. The runtime primitive applied and stayed within sanity
  gates, but the selected local overlay increased bytes by `3179` and damaged
  PoseNet enough to miss the break-even component delta.

The active high-upside follow-up is the reversed-base CDO1 diagnostic. This
archive uses the exact CDO1 primitive over the byte-cheap
`cmg3_nonzero_top1` base rather than adding an overlay to the already-tight
C067 archive:

- Builder:
  `experiments/build_c067_reversed_base_cdo1_candidate.py`
- Manifest:
  `experiments/results/c067_reversed_base_cdo1_candidate_20260503/cmg3_nonzero_top1_pairwaterfill4k/c067_reversed_base_cdo1_candidate_manifest.json`
- Base archive:
  `experiments/results/lightning_batch/exact_eval_c067_cmg3_nonzero_top1_t4_20260502T1100Z/archive.zip`
- Output archive:
  `experiments/results/c067_reversed_base_cdo1_candidate_20260503/cmg3_nonzero_top1_pairwaterfill4k/archive.zip`
- Archive bytes/SHA:
  `132618`,
  `78fb9b6d4466f3fe32e755676a45eb6ddf8b170214be44f55262a777d3bdcd7b`
- CDO1 payload:
  compressed `4396` bytes, compressed SHA
  `f1398cae9ebc2a06ad0a8af1a1dcfc8436c6e470ee687407c48c328e1fad3bdb`,
  raw `13427` bytes, raw SHA
  `3e0f19d8bec0608bd6ecaeec96be413e782f8dcba19f612e0f937d2b4e1476c8`.
- Lightning job:
  `exact_eval_c067_reversed_base_cdo1_top1_pairwaterfill4k_rtxprodiag_20260503T0050Z`
- State path:
  `.omx/state/exact_eval_c067_reversed_base_cdo1_top1_pairwaterfill4k_rtxprodiag_20260503T0050Z_batch_jobs.json`
- Source manifest:
  `.omx/state/exact_eval_c067_reversed_base_cdo1_top1_pairwaterfill4k_rtxprodiag_20260503T0050Z_manifest.json`
- Initial status:
  `Pending`, zero cost on RTX PRO at latest refresh.

Decision rule:

- If the reversed-base diagnostic lands component-safe and below C067, queue
  identical bytes on T4/equivalent immediately.
- If it remains far above C067, preserve the component trace and use it to
  tighten the CDO1 trust-region/economics planner; do not broaden CDO1 by hand.
- The current CDO1 evidence does not kill the primitive. It says local overlay
  on top of C067 is byte- and PoseNet-regressive, while reversed-base CDO1 is
  still the only live non-training archive with enough rate headroom to test a
  true sub-0.30 move.

## 2026-05-03T01:25Z - Corrected half-frame CDO1 overlay harvested; no T4 promotion

The pair-basis bug was fixed at the manifest/header/dispatch-preflight level,
then the corrected local-overlay diagnostic was exact-evaluated on RTX PRO.
This is a clean A-negative scoped forensic result, not a frontier candidate:

- Job:
  `exact_eval_c067_cdo1_local_overlay_pairbasisfix_rowspan_rtxprodiag_20260503T0110Z`
- Local artifact:
  `experiments/results/lightning_batch/exact_eval_c067_cdo1_local_overlay_pairbasisfix_rowspan_rtxprodiag_20260503T0110Z/contest_auth_eval.adjudicated.json`
- Archive bytes/SHA:
  `278330`,
  `7adff8fc4a0bf9fc2d86c113a333dd23c18f12627fc065233908f2f1f2577717`
- Exact CUDA diagnostic score:
  `0.3608887235320785`
- Components:
  PoseNet `0.00130321`, SegNet `0.00061402`, `600` samples.
- Delta versus C067 A++:
  `+0.04527169274759618`.
- Evidence grade:
  RTX PRO exact CUDA diagnostic, non-promotion hardware, `promotion_eligible=false`.
- Decision:
  no T4 promotion. Even after correcting `half_frame_pair_index`, local CDO1
  overlay on top of C067 increases bytes and damages PoseNet far beyond its
  break-even component delta.

Permanent hardening landed in the local source tree before further dispatches:

- CDO1 payloads must record `pair_index_basis` and sorted integer
  `selected_pair_indices`; missing or invalid basis is rejected.
- Exact-eval submit preflight validates staged CDO1 archive manifests and
  requires queue metadata basis to match the archive manifest.
- Local smoke suffix allowlist now includes `.cdo1`, `.cdo1.xz`,
  `.cdo1.zlib`, and `.cdo1.br` alongside AMR1 compressed forms.
- Focused CDO1/Lightning/smoke tests passed for the new guard.

Interpretation:

- The corrected local overlay proves the CDO1 runtime primitive applies
  charged mask deltas through the canonical inflate path.
- It also proves that byte-regressive local overlays over C067 are not the
  shortest path to sub-0.30.
- CDO1 remains interesting only where it replaces a byte-cheap base and spends
  residual bits by marginal component benefit per byte. The active follow-up is
  a corrected reversed-base CDO1 screen with explicit half-frame basis and a
  touched-frame preflight, not broader hand-selected local overlay.

## 2026-05-03T01:44Z - Recovery pass: Lane 12 blocker clarified; trained-QBF1 diagnostic queued

Recovered after context loss and refreshed the active surfaces instead of
trusting stale claims.

Lane 12 / NeRV:

- `experiments/plan_lane12_l2_unblock.py` refreshed
  `experiments/results/lane12_l2_unblock_readiness_20260502/lane12_l2_unblock_readiness.refresh_20260503.json`.
- `experiments/preflight_lane12_decoded_baseline_build.py` refreshed
  `experiments/results/lane12_l2_unblock_readiness_20260502/decoded_baseline_build_preflight.refresh_20260503.json`.
- Decoded-baseline contract consumption, runtime closure, NRV/QZS3 parser
  presence, and launcher guards pass.
- Retraining remains blocked. This is not just missing paperwork:
  `.omx/state/lane12_nerv_l2_clearance.json` is absent and there is no passing
  `alpha_geo_0_nerv_geometry` promotion geometry JSON.
- Current geometry evidence still shows global disagreement
  `0.012303928799099393`, pair-transition disagreement
  `0.009507171571470149`, and boundary-2px disagreement
  `0.14883144511692872`, which blocks both retraining unblock and exact-eval
  dispatch.

Corrected reversed-base CDO1 worker artifacts were verified locally:

- Tools:
  `experiments/plan_c067_reversed_base_cdo1_overlay_economics.py`,
  `experiments/build_c067_reversed_base_cdo1_candidate.py`.
- Focused tests:
  `6 passed in 0.23s`.
- Byte-closed worker candidates preserve `pair_index_basis=half_frame_pair_index`
  and selected pairs `[79,153,212,216,230]`, but the measured bases are
  catastrophic exact negatives. Do not dispatch these unless a new geometry
  gate or operator decision deliberately accepts the high-risk diagnostic.

Trained renderer / Block-FP:

- Current QBF1 transplant preflight found one byte-closed candidate:
  `trained_qbf1_b0512`, archive bytes `283432`, SHA
  `6174424516a93aa35852cd4dc22b9132516fc38f824b77562b3f91b3f55f58dc`.
- It is `7218` bytes larger than C067, so it requires a component improvement
  to earn promotion. The diagnostic is still valuable because it tests a
  renderer-runtime mechanism rather than another mask-geometry cliff.
- Lightning doctor and manifest staging passed:
  `.omx/state/trained_qbf1_b0512_h100diag_20260503T0200Z_doctor.json`,
  `.omx/state/trained_qbf1_b0512_h100diag_20260503T0200Z_manifest.json`.
- H100 submit attempts did not create a job: explicit Lambda cloud account
  mismatched the Studio, and symbolic `H100` was rejected by the Studio AWS
  cluster. Symbolic `RTXP_6000` was also rejected. This bug class is now
  guarded in `scripts/launch_lightning_batch_job.py`: Studio exact-eval submit
  fails early for known symbolic non-T4 accelerator names and suggests concrete
  provider classes.
- Active queued diagnostic:
  `exact_eval_trained_qbf1_b0512_g7e_20260503T0142Z` on `g7e.4xlarge`
  (`RTXP_6000`), state in `.omx/state/lightning_batch_jobs.json`, currently
  `Pending` at zero cost at first refresh.

Stale claim recovery:

- Closed `pr67_same_h100_component_trace_refresh` as completed external H100
  component trace; not own-score or T4 evidence.
- Closed `c063_same_h100_component_trace` as completed H100 diagnostic trace;
  not T4 evidence.
- Closed `public_pr65_exact_trace_h100_fix1` as failed external runtime
  mismatch superseded by the compatibility trace.
- No hidden T4 A++ result below C067 was found in the read-only audit.

Next decision rule:

- Harvest `exact_eval_trained_qbf1_b0512_g7e_20260503T0142Z` as soon as
  artifacts land.
- Promote to T4 only if the exact CUDA diagnostic component improvement pays
  for the `7218` byte rate penalty and beats C067 after recomputation.
- If it fails, close current surrogate/transplant Block-FP as negative and
  focus the next large move on a real non-surrogate trained renderer export or
  a geometry-safe mask representation that can pass Lane 12's promotion
  geometry gates.

## 2026-05-03T01:54Z - QBF1 transplant harvested; no duplicate SJ-KL rerun

The trained-QBF1 renderer transplant diagnostic completed and was harvested
through the state-derived Lightning SSH path with archive byte/SHA validation.
It is a clean scoped forensic negative, not a frontier candidate:

- Job:
  `exact_eval_trained_qbf1_b0512_g7e_20260503T0142Z`
- Local artifact:
  `experiments/results/lightning_batch/exact_eval_trained_qbf1_b0512_g7e_20260503T0142Z/contest_auth_eval.adjudicated.json`
- Archive bytes/SHA:
  `283432`,
  `6174424516a93aa35852cd4dc22b9132516fc38f824b77562b3f91b3f55f58dc`
- Exact CUDA diagnostic score:
  `17.72267562501643`
- Components:
  PoseNet `29.82484055`, SegNet `0.0026408`, `600` samples.
- Hardware:
  RTX PRO 6000 Blackwell Server Edition, not T4-equivalent promotion hardware.
- Adjudication:
  `REGRESSION_AND_COMPONENT_GATE_AND_SANE_SCORE_REVIEW_REQUIRED`,
  `promotion_eligible=false`, PoseNet component gate failed, sane-score gate
  failed.
- Decision:
  no T4 promotion. This retires only the measured QBF1 transplant config. It
  does not kill renderer self-compression, but any successor must prove local
  renderer/runtime parity and pose safety before exact-eval spend.

The SJ-KL tiny-payload worker returned a deterministic sibling archive:

- Archive:
  `experiments/results/sjkl_c067_tiny_payload_sibling_20260503_codex/pack/archive.zip`
- Bytes/SHA:
  `276556`,
  `a576960be12fdcec1cc76257d5a49cd4102476c7e461150847c845ba0cceab6d`
- Byte delta versus C067:
  `+342`, inside the byte break-even envelope.

This archive is byte-identical to the q6 sibling that already has T4 exact
evidence from `exact_eval_sjkl_c067_q6_sibling_t4_20260502T2357Z`:
score `0.3158419419767293`, worse than C067 by `+0.00022491119224697`.
Therefore no duplicate dispatch is warranted. SJ-KL remains a valid runtime
contract and coefficient-family search target, but only a smaller or more
beneficial payload should be exact-evaluated again.

Swarm respawn:

- `Lorentz the 2nd` owns a no-dispatch Lane 12 geometry-gate repair atom
  planner/profiler.
- `Erdos the 2nd` owns a no-dispatch renderer transplant pose-safety preflight
  so QBF1-class collapse is caught before GPU spend.

## 2026-05-03T01:56Z - Field-selected CMG3A poseguard diagnostic queued

Built two local byte-screen archives from the hard-pair/foveal field plan
after the `extra065`/`extra072` exact negatives:

- `c067_hotspot_poseguard_neg2_top0008`:
  bytes `137945`, SHA
  `0282ae84be5fc0d72d783a0295888016732200610875cee6bb15d4fe5980643c`,
  formula-only rate delta versus C067 `-0.09206765158924951`.
- `c067_hotspot_poseguard_neg2_top0012`:
  bytes `137987`, SHA
  `19beb6cdc0424bc423633625e630f94ce16e013e62fd7a574beefd23aa161821`,
  formula-only rate delta versus C067 `-0.09203968551321838`.

Top12 was selected for one fast diagnostic because it has slightly lower
decoded-mask disagreement than top8 for only `42` extra bytes. This is
high-risk because same-family coarse-mask archives have exact PoseNet-collapse
negatives, but it is not an arbitrary rerun: the selected atoms come from
`experiments/results/c067_hotspot_mask_geometry_compiler_20260502/next_pose_safe_plan_after_extra065_072_negatives.json`,
which was built specifically from the hard-pair/foveal geometry profile.

Queued Lightning RTX PRO diagnostic:

- Job:
  `exact_eval_cmg3a_poseguard_top0012_g7e_20260503T0200Z`
- State:
  `.omx/state/exact_eval_cmg3a_poseguard_top0012_g7e_20260503T0200Z_batch_jobs.json`
- Source manifest:
  `.omx/state/cmg3a_poseguard_top0012_g7e_20260503T0200Z_manifest.json`
- Doctor:
  `.omx/state/cmg3a_poseguard_top0012_g7e_20260503T0200Z_doctor.json`
- Local artifact mirror:
  `experiments/results/lightning_batch/exact_eval_cmg3a_poseguard_top0012_g7e_20260503T0200Z`
- Initial status:
  `Pending`, zero cost.

Decision rule:

- If exact CUDA score is below C067 or close enough that a T4 replay could
  plausibly invert the hardware drift, queue identical bytes on T4.
- If it collapses, preserve the component trace and use it to tighten the
  field allocator instead of dispatching more coarse CMG3A variants blindly.

## 2026-05-03T02:00Z - Worker greenup integrated; exact-eval harvest metadata hardened

Lane 12 geometry worker result:

- Added local-only planner:
  `experiments/plan_lane12_geometry_gate_repair_atoms.py`
- Artifact:
  `experiments/results/lane12_geometry_gate_repair_atoms_20260503/lane12_geometry_gate_repair_atoms.json`
- Artifact SHA:
  `2e0d465603f4c035d14a7a7d2e1776f95a611f5faf883d4b4b4051fb2f600a00`
- Output:
  `62` geometry repair atoms, no archive, no remote dispatch, no score claim.
- Interpretation:
  this creates the next charged-builder target language for Lane 12 geometry
  unlock. It does not clear retraining or exact-eval gates.

Renderer transplant safety worker result:

- Added local fail-closed preflight:
  `experiments/preflight_renderer_transplant_pose_safety.py`
- Artifact:
  `experiments/results/trained_renderer_blockfp_preflight_20260503_qfaithful_qzs3_2146/trained_qbf1_b0512/pose_safety_preflight.json`
- Result:
  `safe_for_exact_eval_dispatch=false`, failure class
  `renderer_transplant_pose_safety_failed`, fail-closed reason
  `render_output_parity_unsafe`.
- Diagnostic:
  masks and poses were byte-identical to C067; only `renderer.bin` changed.
  Source-vs-candidate local rendered output parity was catastrophically unsafe
  before scorer involvement, with mean absolute output delta
  `72.17086791992188`, RMS `87.61087004921963`, and max absolute delta
  `254.8052520751953` on sampled pairs.
- Interpretation:
  this explains the QBF1 exact CUDA PoseNet collapse and blocks further
  trained-renderer exact-eval dispatch until raw export and compressed export
  both pass source-runtime output parity.

Permanent hardening:

- `src/tac/deploy/lightning/batch_jobs.py` now assigns a deterministic default
  `local_artifact_dir` for exact-eval specs:
  `experiments/results/lightning_batch/<job-name>`.
- This prevents the QBF1 harvest failure mode where completed artifacts had a
  valid remote output dir but the state record lacked a local mirror dir.
- Focused Lightning tests passed for the new default and the existing symbolic
  non-T4 Studio-machine guard.

Active remote:

- `exact_eval_cmg3a_poseguard_top0012_g7e_20260503T0200Z` advanced from
  `Pending` to `Running` on RTX PRO at zero reported cost in the latest
  refresh. Harvest remains the next score-critical action.

## 2026-05-03T02:05Z - CMG3A poseguard exact diagnostic harvested

`exact_eval_cmg3a_poseguard_top0012_g7e_20260503T0200Z` completed and was
harvested through the state-derived Lightning SSH path.

- Archive bytes/SHA:
  `137987`,
  `19beb6cdc0424bc423633625e630f94ce16e013e62fd7a574beefd23aa161821`
- Exact CUDA diagnostic score:
  `29.16496751549104`
- Components:
  PoseNet `39.62880325`, SegNet `0.09166103`, `600` samples.
- Hardware:
  RTX PRO 6000 Blackwell Server Edition, not T4-equivalent promotion hardware.
- Adjudication:
  `REGRESSION_AND_COMPONENT_GATE_AND_SANE_SCORE_REVIEW_REQUIRED`,
  `promotion_eligible=false`, PoseNet and SegNet component gates failed, sane
  score gate failed.
- Local artifact:
  `experiments/results/lightning_batch/exact_eval_cmg3a_poseguard_top0012_g7e_20260503T0200Z/contest_auth_eval.adjudicated.json`
- Component trace:
  `experiments/results/lightning_batch/exact_eval_cmg3a_poseguard_top0012_g7e_20260503T0200Z/component_trace.json`

Interpretation:

- The field-selected top12 atom policy did not rescue the coarse CMG3A
  topology. The archive saved `138227` bytes, but the logical mask/pose basin
  collapse dominated by roughly `+28.85` score.
- Do not dispatch more same-family coarse CMG3A/PMG variants from unchanged
  byte screens. The next mask-topology work must either preserve geometry
  before byte savings, use the Lane 12 repair-atom planner to build a charged
  geometry gate candidate, or move to a learned representation with local
  geometry/parity gates.
- This is a measured-implementation negative only; it does not kill
  multi-resolution, foveation, or learned mask representation as families.

## 2026-05-03T02:30Z - Lane12 CDO1 bridge bounded; canonical-score hardening landed

Lane12 geometry repair builder:

- Added charged no-score builder:
  `experiments/build_lane12_geometry_gate_repair_candidate.py`
- Added focused tests:
  `src/tac/tests/test_build_lane12_geometry_gate_repair_candidate.py`
- The builder consumes
  `experiments/results/lane12_geometry_gate_repair_atoms_20260503/lane12_geometry_gate_repair_atoms.json`,
  applies selected policy atoms as a charged `masks.cdo1[.xz]` overlay over
  the existing Lane12 `masks.nrv`, delegates archive construction to
  `experiments/build_c067_decoded_delta_overlay_candidate.py`, and records
  `score_claim=false`, `promotion_eligible=false`, and
  `exact_eval_claim=false`.

Real policy result:

- Command:
  `.venv/bin/python experiments/build_lane12_geometry_gate_repair_candidate.py --policy-id lane12_geometry_gate_budget_16384b --overlay-compressor lzma_xz --output-dir experiments/results/lane12_geometry_gate_repair_candidate_20260503`
- Archive:
  `experiments/results/lane12_geometry_gate_repair_candidate_20260503/lane12_geometry_gate_budget_16384b/archive.zip`
- Bytes/SHA:
  `301693`,
  `b4bc5a571952a595c8410f1ccb609ec0dacb511b4bae80cd1f448bc934503bc2`
- Repair payload:
  `20098` pixels, `1955` runs, raw CDO1 `18678` bytes, compressed CDO1
  `5108` bytes.
- Geometry gate:
  global disagreement improved only from `0.012303928799099393` to
  `0.01221874237060547`; residual disagreement remained `2882763` pixels.
- Decision:
  `dispatch_allowed=false`. No exact-eval dispatch; this is a local geometry
  bound, not score evidence.

Full-repair economic bound:

- Full Lane12 CDO1 repair over all `2902861` disagreement pixels would require
  `613745` runs, raw CDO1 `5526606` bytes, zlib `2316147` bytes, or xz
  `1443024` bytes.
- The xz overlay alone adds about `+0.9608504499701682` score in the rate term.
- Decision:
  full CDO1-over-NeRV repair is economically impossible for sub-0.30. Lane12
  remains useful only if the representation itself changes or a learned/compact
  geometry-preserving repair code replaces raw CDO1.

C067 byte-accounting refresh:

- Regenerated
  `experiments/results/archive_byte_profile_c067_resume_20260503/profile.{json,md}`
  from the A++ C067 frontier archive and adjudicated T4 JSON.
- Canonical score source:
  `score_recomputed_from_components`, score `0.31561703078448233`.
- Unchanged-distortion sub-0.300 crossing requires `23454` fewer bytes;
  buffered target archive size is `252759` bytes.
- Stream allocation remains:
  `masks.mkv` `219472` bytes, `renderer.bin` `55965` bytes,
  `optimized_poses.bin` `677` bytes.
- Decision:
  simple nested compression is exhausted; sub-0.30 needs either about `10.7%`
  mask-stream reduction with scorer geometry intact, or a structurally safe
  renderer-byte win. Pose bytes cannot close the gap alone.

Permanent rounded-score bug-class hardening:

- `experiments/contest_auth_eval.py` now emits additive canonical fields:
  `canonical_score`, `canonical_score_source`,
  `reported_final_score_display_rounded`, `score_rounding_abs_delta`, and
  `score_reported_rounded_differs_from_canonical`.
- The console summary now prints canonical score separately from rounded
  upstream display score.
- `experiments/profile_archive_byte_accounting.py` now prefers
  `canonical_score` / `score_recomputed_from_components` and records
  `target_gap.score_source`, so byte-gap math does not silently rank rounded
  `final_score`.
- Focused verification passed:
  py-compile on touched files, `30 passed` for
  `src/tac/tests/test_contest_auth_eval.py` and
  `src/tac/tests/test_profile_archive_byte_accounting.py`, plus
  `git diff --check` on the touched score-hardening files.

## 2026-05-03T02:48Z - Renderer-only path triaged; micro-dispatch guard added

The byte/self-compression worker reviewed the C067 archive and confirmed the
current high-EV no-mask-geometry conclusion:

- `masks.mkv` is already the dominant stream at `219472` charged bytes, but
  exact/lossless recoding probes are byte-regressive and lossy/coarse topology
  variants are exact negatives.
- `optimized_poses.bin` is only `677` bytes and cannot close the `23454` byte
  unchanged-distortion sub-0.300 gap.
- Generic/nested recompression is exhausted.
- Therefore the credible `>=23500` byte path that preserves C067 decoded mask
  geometry is renderer-only: learned bit-depth/self-compressed renderer,
  trained IMP/sparse recovery, or slim/factorized teacher-student JFG.
- Worker memo:
  `.omx/research/c067_byte_self_compression_opportunity_review_20260503_codex.md`

Refreshed local planning artifacts:

- Byte-accounting review:
  `experiments/results/c067_archive_byte_accounting_20260503_review/archive_byte_accounting.{json,md}`
- Renderer self-compression v2 plan:
  `experiments/results/c067_renderer_self_compression_v2_20260503_review/plan.json`
- Trained renderer export unlock plan:
  `experiments/results/trained_renderer_export_unlock_20260503_review/plan.json`

Renderer planner conclusion:

- Naive global QZS3 reblocks are fail-closed by exact CUDA negative evidence
  from `exact_eval_c067_qzs3_b512_l40sdiag_20260502T1710Z`, where the byte win
  came with PoseNet collapse.
- Current QBF1-v1/v2 byte models are not dispatchable: the loader-ready v1
  format is byte-regressive and the v2 self-describing byte model has no
  reviewed decoder plus remains above the current QZS3 raw slice.
- The only concrete local QZS/MQZ survivor is
  `mixed_local_component_aware_v1_frame2_all64`, saving `87` renderer-stream
  bytes. This is polish-only, not a sub-0.30 lane.

Permanent micro-dispatch guard:

- `experiments/plan_c067_renderer_self_compression_v2.py` now has a
  `--min-dispatch-renderer-byte-savings` gate, default `1024`.
- Its dispatch recommendation now records
  `min_dispatch_renderer_byte_savings` and `best_renderer_byte_savings`.
- The refreshed C067 plan now reports
  `exact_cuda_dispatch_warranted=false` for the `87`-byte MQZ1 candidate and
  explicitly says to prioritize learned/trained renderer compression.
- Focused verification passed:
  py-compile, `5 passed` for
  `src/tac/tests/test_plan_c067_renderer_self_compression_v2.py`, and
  `git diff --check` on the touched planner/test files.

Decision:

- Do not spend exact-eval budget on micro renderer reblocking while the main
  gap is `23454` bytes.
- Next implementation target is a contest-compliant trained renderer
  self-compression lane with fixed C067 masks/poses, deterministic export,
  local transplant/pose-safety preflight, and exact CUDA only after a concrete
  archive has material byte headroom.

## 2026-05-03T03:05Z - Renderer transplant gate is now pose-safe by construction

Recovered the learned renderer self-compression worker memo:

- Memo:
  `.omx/research/c067_learned_renderer_self_compression_lane_20260503_worker.md`
- Worker conclusion:
  the credible large renderer-only move is a fresh fixed-mask/fixed-pose
  Q-FAITHFUL/JFG training burn, not reuse of the stale `trained_qbf1_b0512`
  transplant or another local QZS3 reblock.

Permanent dispatch hardening landed:

- `experiments/preflight_trained_renderer_transplant.py` now requires a matching
  `experiments/preflight_renderer_transplant_pose_safety.py` JSON for the exact
  source/candidate archive SHA pair before it emits dispatchable Lightning exact
  eval commands for a non-surrogate renderer export.
- The gate blocks when the pose-safety report is missing, failed, stale,
  score-claiming, remote-dispatched, schema-mismatched, or SHA-mismatched.
- The generated exact-eval command shape now uses concrete `g7e.4xlarge`
  instead of the previously failed symbolic accelerator name.
- `experiments/plan_trained_renderer_export_unlock.py` now treats older ready
  preflight summaries without the pose-safety gate as blocked.
- `AGENTS.md` now records the durable protocol: renderer transplants that keep
  `masks.mkv` and `optimized_poses.bin` but replace `renderer.bin` must pass
  local runtime output-parity/pose-safety before exact-eval dispatch.

Real stale-transplant review artifact:

- Reran the Q-Faithful-derived transplant preflight with the existing failed
  pose-safety report:
  `experiments/results/trained_renderer_blockfp_preflight_20260503_qfaithful_qzs3_2146_posegate_review/trained_renderer_blockfp_preflight.json`
- Result:
  `h100_lightning_readiness.ready=false`,
  `best_dispatchable_after_pose_safety=null`.
- Best byte candidate remains `trained_qbf1_b0512`, bytes `283432`, SHA
  `6174424516a93aa35852cd4dc22b9132516fc38f824b77562b3f91b3f55f58dc`, but its
  pose-safety gate is failed with blocker `render_output_parity_unsafe`.
- Refreshed unlock plan:
  `experiments/results/trained_renderer_export_unlock_20260503_posegate_review/plan.json`
  reports `blocked_no_h100_dispatch`, `h100_ready_preflight_count=0`.

Verification:

- `py_compile` passed for the trained-renderer preflight, unlock planner, and
  their focused tests.
- `src/tac/tests/test_preflight_trained_renderer_transplant.py` and
  `src/tac/tests/test_plan_trained_renderer_export_unlock.py`: `11 passed`.
- `git diff --check` passed for the touched hardening files and generated
  review JSON.

Decision:

- No remote dispatch from this slice.
- The next score-moving renderer lane must train a fresh renderer against the
  exact C067 charged masks/poses, then pass raw-export and compressed-export
  pose-safety gates before any exact CUDA spend.

IMP bridge recovery:

- Reconciled the stale no-train IMP bridge claims in
  `.omx/state/active_lane_dispatch_claims.md`.
- Existing L40S exact CUDA diagnostics are all A-negative/no-promotion:
  cycle1 `1.355401799409314` at `271746` bytes, cycle2
  `5.528075689026167` at `267153` bytes, cycle5 `36.79093785806109` at
  `256193` bytes, and cycle10 `78.9903710766749` at `244623` bytes.
- Decision:
  do not resurrect no-train IMP bridge. IMP remains relevant only as a trained
  sparse-renderer/self-compression route with the same fixed-mask/fixed-pose
  pose-safety gates as other renderer transplants.

## 2026-05-03T02:49Z - Fixed C067 renderer-burn prep is deterministic and gate-aware

Added a local/no-dispatch bridge from the exact C067 A++ archive to a fresh
fixed-mask/fixed-pose renderer training burn:

- Tool:
  `experiments/prepare_c067_fixed_renderer_burn.py`
- Focused test:
  `src/tac/tests/test_prepare_c067_fixed_renderer_burn.py`
- Real manifest:
  `experiments/results/c067_fixed_renderer_burn_prep_20260503/c067_qfaithful_fixedmask_fixedpose_seed20260503/fixed_c067_renderer_burn_manifest.json`

Manifest custody:

- Source archive bytes: `276214`
- Source archive SHA:
  `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`
- Extracted fixed runtime members:
  - `masks.mkv`: `223385` bytes,
    SHA `a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb`
  - `renderer.bin`: `59288` bytes,
    SHA `5657593aec0bf380f0bd614578cc4a76da8589b6ef8ce0331c28b6a4d6658efb`
  - `optimized_poses.bin`: `7200` bytes,
    SHA `5236cf75cc95f6a9731b35f4e2fb1211aa99bfcc69e5964869edd19a7af3df9f`

The generated shell script records deterministic training and snapshot command
shapes for `q_faithful_dilated_88k`, fixed C067 masks, fixed C067 poses, QZS3
snapshot export, no exact eval, and no dispatch claim. The preparer fails
closed on missing/mismatched C067 archive/member hashes.

The real manifest also embeds static argparse dry-run proof:

- `train_renderer`: `OK — 1 required flag(s) all present`
- `q_faithful_snapshot_loop`: `OK — 4 required flag(s) all present`

Gate state:

- `training_dispatch_gate.cleared_for_retraining_dispatch=false`
- Blocker:
  `lane12_l2_clearance_packet_missing_or_unreadable`
- Refreshed Lane12 unblock report:
  `experiments/results/lane12_l2_unblock_readiness_20260503_resume/lane12_l2_unblock_readiness.refresh2_20260503.json`
- Current readiness remains false: no clearance packet, no passing
  promotion-threshold Alpha-Geo geometry JSON, and no usable pose-regeneration
  provenance.

Bug class fixed during review:

- The first script renderer would have single-quoted `$PWD`, causing the
  snapshot loop to receive a literal workspace string. The shell renderer now
  preserves `$PWD` expansion and the focused test asserts this explicitly.
- Long-running command flag drift is now checked inside the generated manifest
  via `tools/argparse_dryrun.py`, rather than as an external operator habit.

Decision:

- This does not dispatch training and does not change the score frontier.
- It removes local ceremony and checksum risk from the highest-EV renderer
  lane. Once the retraining gate is legitimately cleared or an explicit audited
  operator override is recorded, this manifest is the handoff into the fresh
  fixed-mask/fixed-pose H100/H200 burn.

Cross-agent dispatch hygiene hardening:

- `tools/claim_lane_dispatch.py` now treats `refused_dispatch*`,
  `stale_superseded*`, and `stopped_*` status families as terminal.
- Focused regression coverage:
  `src/tac/tests/test_claim_lane_dispatch.py::test_claim_helper_treats_stale_refused_and_stopped_families_as_terminal`
- Rationale:
  several historical rows used precise but newer terminal vocabulary, which the
  helper previously treated as active conflicts. That could falsely block a
  valid dispatch or push an operator toward `--force`.
- `.omx/state/active_dispatches.md` also had six stale rows left under
  `## Active` despite live Vast state being empty. Moved them into the
  completed table as a dated reconciliation row.
- Verification:
  `scripts/reconcile_vast_dispatch_state.py --json` now reports
  `live_count=0`, `active_dispatch_count=0`, `active_missing_live=[]`, and
  `live_missing_active=[]`.

## 2026-05-03T03:10Z - Sub-0.314 Byte Gate Triage And QZS4+PVR1 Diagnostic

Current A++ C067 frontier remains:

- Score: `0.31561703078448233`
- Archive bytes: `276214`
- Archive SHA-256:
  `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`

At unchanged SegNet/PoseNet distortion, crossing `0.314` requires roughly
`2428` byte savings, i.e. a target archive near `273786` bytes or lower.

Evidence and triage:

- `experiments/results/c067_archive_byte_accounting_20260503_review/archive_byte_accounting.md`
  confirms direct nested compression is exhausted for C067:
  `masks.mkv` best nested savings `0`, `renderer.bin` best nested savings `0`,
  and pose bytes are only `677` encoded bytes.
- QZS4 block128 remains a measured exact CUDA negative, not a T4 candidate:
  `273247` bytes, SHA
  `32ce9cd3ebbbfa6b3468dc9ba7ac31f0fbd2802a8eb25f6bdd44c137d9f41c69`,
  H100 diagnostic score `1.5244097988910252`, PoseNet `0.156837`.
- Q-FAITHFUL geometry-closed zoom candidates are already exact-screened
  negatives after the runtime consumed `zoom_scalars.bin`: primary SHA
  `f64dcb3d12db394efa9b0e0f924bb62b6b24f096d66baf9ed83447077d4f9b61`,
  H100 score `22.147631187370024`. Do not T4-promote these snapshots.
- Lane 12 CDO1 geometry-repair candidates were materialized for all planned
  policies under
  `experiments/results/lane12_geometry_gate_repair_candidate_20260503/`.
  They are byte-closed but not dispatchable: the largest policy is
  `301693` bytes and only moves decoded-mask global disagreement from
  `0.012303928799099393` to `0.01221874237060547`, far above the `0.001`
  local unblock gate.
- `experiments/results/c067_bigmove_nontrain_candidate_triage_20260503_sub0314/c067_bigmove_nontrain_candidate_triage.json`
  refreshes the non-retraining big-move triage. The best-looking byte screens
  remain blocked by exact-negative family evidence unless a redesigned
  geometry/pose-safe selector is added.

Diagnostic dispatched:

- Job:
  `exact_eval_qzs4_pr64_pvr1_top64_rtxprodiag_20260503T0309Z`
- Lane claim:
  `qzs4_pr64_pvr1_top64_rtxprodiag`
- Machine: Lightning `g7e.4xlarge` / RTX PRO diagnostic, not T4 promotion.
- Archive:
  `experiments/results/qzs4_qp2_stack_screen_c058_20260502/qzs4_pr64_pvr1_top64/archive.zip`
- Bytes: `273749`
- SHA-256:
  `550f569984c985c40a60f18812c65fd0cc31024f2e50cc3c8ee815edee2d4ce3`
- Source manifest:
  `.omx/state/exact_eval_qzs4_pr64_pvr1_top64_rtxprodiag_20260503T0309Z_manifest.json`
- State:
  `.omx/state/qzs4_pr64_pvr1_top64_rtxprodiag_batch_jobs_20260503T0309Z.json`

Decision:

- This diagnostic is high risk because its parent QZS4 renderer already
  collapsed PoseNet, but it is one of the only byte-valid unresolved variants
  crossing the sub-0.314 unchanged-distortion threshold.
- Do not submit T4 unless this exact RTX PRO JSON shows component survival
  near C067. If it collapses, retire only this QZS4+PVR1 top64 implementation
  and keep the lesson as a block-size/pose-side-info interaction map.

## 2026-05-03T03:46Z - Latest GitHub Leaderboard Intake And PR75 Ablation Wave

Public GitHub/leaderboard refresh:

- Rendered comma leaderboard still rounds the top faithful entries to the
  `0.32`/`0.31` band, but the GitHub PR stream has a newer open faithful
  target:
  `https://github.com/commaai/comma_video_compression_challenge/pull/75`.
- PR #75 `qpose14_r55_segactions_minp (0.31)` reports:
  PoseNet `0.00048653`, SegNet `0.00060686`, bytes `276741`, archive SHA
  `cc1dfa346811dade15044910fa0143e73a7a049b98aeec39b96a53481a0788bd`.
  Formula recompute from the rounded body fields is about
  `0.31470817503416587`, roughly `0.000908856` below the current C067 T4
  frontier, subject to PR-body rounding.
- PR #73 `emir_flatpack` is a lossless packing idea for older qpose14, not a
  better score by itself: its body reports score `0.37`, bytes `281948`, and a
  single-Brotli/permuted stream idea that may still be transplantable as a
  pack-only atom.
- PR #70 `mask_decoder` reports `0.19`, but the author states it moved bytes
  from the archive into `inflate.py`. Treat it as non-faithful exploit
  motivation only, not a contest-compliant target.

PR75 archive anatomy, public artifact:

- Downloaded archive:
  `experiments/results/top_submission_reverse_engineering_20260503_pr75/archive.zip`
- Official archive SHA-256:
  `cc1dfa346811dade15044910fa0143e73a7a049b98aeec39b96a53481a0788bd`
- Official archive bytes: `276741`
- Single ZIP member: `p`, bytes `276641`, SHA
  `959124abd97042983a47163f70c80fd4c0c751dd5e4cf6b5f0f434d9d2fcd66c`
- Fixed slices:
  - mask Brotli bytes `219472`, decoded bytes `223385`, decoded SHA
    `a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb`
  - model Brotli bytes `56034`, decoded bytes `59288`, decoded SHA
    `2333284a73446c3b323948fb883ade0f677baf9ad5d9d06aa1da7bec337bd9c9`
  - SegNet tile-action Brotli bytes `236`, decoded bytes `268`, decoded SHA
    `bfd46b2b481a5064cc1f64b7b1288640c51b89ad6aeb5598408150f7945eac15`
  - QP1 pose Brotli bytes `899`, decoded QP1 bytes `1140`, decoded SHA
    `8d77f6a39d1a84eca78fbe8fa5ddc31f6ada29814b49e32daeb800ea84a015cc`

Runtime and parser support landed locally:

- `submissions/robust_current/unpack_renderer_payload.py` now accepts the
  PR75 fixed-slice public blob and a self-describing `P3` form that carries
  mask/model/tile-action/pose slices.
- `submissions/robust_current/inflate_renderer.py` now consumes charged
  `seg_tile_actions.bin` and applies PR75-style tile deltas to the generated
  second frame before upsampling.
- Focused verification passed:
  `py_compile` on the touched inflate/unpack code and
  `src/tac/tests/test_unpack_renderer_payload_fixedslice.py`
  (`5 passed`).

Five PR75-derived C067 ablations are running as non-promotion diagnostics on
fast Lightning GPU classes. These test scorer effects before spending T4:

| candidate | archive bytes | SHA-256 | state |
|---|---:|---|---|
| `c067_pr75_model_only_fixed` | `276283` | `f0f677d1bc98648fe8f2b4bcd3fcaafe3b4920aa4881a369a16aaedf0550e09b` | `exact_eval_c067_pr75_model_only_fixed_diag_20260503T0346Z` |
| `c067_pr75_pose_only_fixed` | `276436` | `b6ac7822360836230b5039f5c4f30a30e2912ee758a02d792ef0d114e9570a92` | `exact_eval_c067_pr75_pose_only_fixed_diag_20260503T0346Z` |
| `c067_pr75_actions_only_p3` | `276460` | `851a61ddd88c0ddb4a05762f11e900f132fb771fd9dba743ac2026637ddf15d9` | `exact_eval_c067_pr75_actions_only_p3_diag_20260503T0346Z` |
| `c067_pr75_pose_actions_p3` | `276682` | `72359590fc36e50045aeb4f9cb5e5ad259f34baa1addba19be84c44594936ac7` | `exact_eval_c067_pr75_pose_actions_p3_diag_20260503T0346Z` |
| `c067_pr75_model_pose_actions_p3` | `276751` | `f14f3c8d68d5a5c3ed9308217e697c1af004f93fba91974a041df2e9577aec94` | `exact_eval_c067_pr75_model_pose_actions_p3_diag_20260503T0346Z` |

State manifest:

- Staging manifest:
  `.omx/state/pr75_segactions_ablation_20260503T0342Z_manifest.json`
- Candidate matrix:
  `experiments/results/c067_pr75_segactions_stack_candidates_20260503/candidate_matrix.json`

PR73 flatpack transplant screen:

- Tested `RPK1` inside single Brotli with exhaustive member-order permutation
  search for C067/PR75 model/pose/action variants under
  `experiments/results/c067_pr75_flatpack_rpk1_byte_screen_20260503/`.
- Result: byte-regressive for this payload family. Best unchanged C067
  equivalent is `276566` bytes (`+352` vs C067), and the best PR75-action
  variant is `276733` bytes (`+519` vs C067). Do not dispatch these unless a
  different raw representation is introduced.

Decision rule for this wave:

- If any diagnostic result beats C067 and has sane components, immediately
  queue T4/equivalent promotion on identical archive bytes.
- If the tile-action-only or pose-action variants improve SegNet/PoseNet, build
  the next wave as finer learned/tile-action search around those actions rather
  than more generic byte shaving.
- If all PR75 ablations regress, keep the reverse-engineered parser/runtime
  support but pivot the main effort back to renderer self-compression and
  learned scorer-aligned correction atoms.

Harvested diagnostic results, RTX PRO:

| candidate | score | bytes | PoseNet | SegNet | decision |
|---|---:|---:|---:|---:|---|
| `c067_pr75_model_only_fixed` | `0.31879603282228647` | `276283` | `0.00054446` | `0.00061043` | model stream is harmful; no T4 promotion |
| `c067_pr75_pose_only_fixed` | `0.3186901835779409` | `276436` | `0.0005415` | `0.00061036` | pose stream is harmful; no T4 promotion |
| `c067_pr75_actions_only_p3` | `0.3152359074190711` | `276460` | `0.0004954` | `0.00060768` | best diagnostic so far; T4 promotion queued |
| `c067_pr75_pose_actions_p3` | `0.31861366594581353` | `276682` | `0.00054235` | `0.00060738` | pose stream dominates/hurts even with actions |
| `c067_pr75_model_pose_actions_p3` | `0.3154328799180497` | `276751` | `0.00049547` | `0.00060766` | beats C067 diagnostic, but worse than actions-only; T4 replay already queued |

New T4/equivalent promotion jobs:

- `exact_eval_c067_pr75_model_pose_actions_p3_t4_20260503T0358Z`
  - Archive bytes `276751`
  - SHA `f14f3c8d68d5a5c3ed9308217e697c1af004f93fba91974a041df2e9577aec94`
  - State `.omx/state/c067_pr75_model_pose_actions_p3_t4_batch_jobs_20260503T0358Z.json`
  - Note: first submit was correctly blocked by the g4dn/T4 inflate Torch
    preflight; rerun records `INFLATE_TORCH_SPEC=torch==2.5.1+cu124`.
- `exact_eval_c067_pr75_actions_only_p3_t4_20260503T0401Z`
  - Archive bytes `276460`
  - SHA `851a61ddd88c0ddb4a05762f11e900f132fb771fd9dba743ac2026637ddf15d9`
  - State `.omx/state/c067_pr75_actions_only_p3_t4_batch_jobs_20260503T0401Z.json`
  - Note: this is the current highest-EV promotion candidate because it keeps
    the C067 renderer/pose streams and charges only the PR75 SegNet
    tile-action atom.

Implication:

- PR75's public gain is not primarily from the model bytes; the first measured
  transplant says the model stream hurts PoseNet.
- The charged tile-action payload is now the active mathematical atom class.
  If the T4 result confirms, next wave should search tile/action records as a
  Lagrangian repair problem: action subset, tile scale, pair/frame selection,
  and learned delta dictionary, with every action byte charged inside `p`.

Lagrangian tile-action subset wave:

- Added deterministic builder:
  `experiments/build_pr75_tile_action_subset_candidates.py`
- Inputs:
  - C067 A++ archive and component trace.
  - PR75 public archive action records.
  - `c067_pr75_actions_only_p3` RTX PRO component trace.
- Record scoring proxy:
  `delta = C067_pair_combined_contribution - actions_pair_combined_contribution`.
  This is a cross-hardware first-order selector only; exact CUDA archive eval
  remains the truth.
- Built candidates:

| candidate | records | archive bytes | SHA-256 |
|---|---:|---:|---|
| `c067_pr75_actions_top25_p3` | `25` | `276328` | `2ee07ed8069b3f3cba668f509e71f20f887f501ad1d4e6dc37c23b99ac377747` |
| `c067_pr75_actions_pose_safe_positive_p3` | `27` | `276336` | `b9b7ff1a44c41a7dc85577d426ff1a4d2d1e88a7f8994808f836f42832b35e0a` |
| `c067_pr75_actions_top40_p3` | `40` | `276386` | `9feef7ffaa254f9e5408996a122682757a054144a3000539553786c5292b7d0a` |
| `c067_pr75_actions_positive_p3` | `55` | `276430` | `4134f9c345ae5af7038213c3ea89641939ace01d22926b3614c53a88985b309a` |

Dispatch:

- Staging manifest:
  `.omx/state/pr75_tile_action_subset_20260503T0405Z_manifest.json`
- Fast diagnostic jobs:
  - `exact_eval_c067_pr75_actions_top25_p3_diag_20260503T0405Z`
  - `exact_eval_c067_pr75_actions_pose_safe_positive_p3_diag_20260503T0405Z`
  - `exact_eval_c067_pr75_actions_top40_p3_diag_20260503T0405Z`
  - `exact_eval_c067_pr75_actions_positive_p3_diag_20260503T0405Z`

Decision:

- If any subset beats `actions_only` on diagnostic hardware, T4-promote that
  subset on identical bytes.
- If all subsets lose, the public 67-record PR75 action list is already near a
  local optimum for this dictionary and the next frontier move must learn new
  action dictionaries or optimize action amplitudes, not just prune records.

Amplitude-policy wave:

The PR75 action dictionary is ordered by direction, amplitude, and sign. This
lets us test amplitude as a charged-but-format-compatible variable without a
new runtime:

| candidate | records | archive bytes | SHA-256 |
|---|---:|---:|---|
| `c067_pr75_actions_all_ampminus1_p3` | `67` | `276461` | `54dd2a3aac631e84658a7dcb97efe914850aa8bf1d19d045debc63ceb9b82c51` |
| `c067_pr75_actions_poseharm_ampminus1_p3` | `67` | `276462` | `33598d9aa54a0587fbc3dc9f31d7b6219a744c262e2ae1036a8422490d5c1f95` |
| `c067_pr75_actions_positive_poseharm_ampminus1_p3` | `55` | `276428` | `3900c31458d5ab64349d0041a765e03ad990f9510aa979b046479ad83ca247a2` |

Dispatch:

- Staging manifest:
  `.omx/state/pr75_tile_action_amp_20260503T0410Z_manifest.json`
- Fast diagnostic jobs:
  - `exact_eval_c067_pr75_actions_all_ampminus1_p3_diag_20260503T0410Z`
  - `exact_eval_c067_pr75_actions_poseharm_ampminus1_p3_diag_20260503T0410Z`
  - `exact_eval_c067_pr75_actions_positive_poseharm_ampminus1_p3_diag_20260503T0410Z`

Decision:

- If amplitude shrink wins, promote the winning amplitude policy rather than
  the public action list.
- If amplitude shrink loses but subset pruning wins, continue with subset
  selection and learn a new dictionary later.
- If neither wins, treat the public action list as the current best
  measured implementation and shift larger EV back to new action search or
  renderer self-compression.

## 2026-05-03T04:45Z - PR75 action atom T4 frontier and subset promotions

The first T4 replay of the PR75 action-atom family landed as a real A++
frontier move:

- Candidate: `c067_pr75_actions_only_p3`
- Job: `exact_eval_c067_pr75_actions_only_p3_t4_20260503T0401Z`
- Artifact dir:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_actions_only_p3_t4_20260503T0401Z`
- Score: `0.31553274375536466`
- Archive bytes: `276460`
- Archive SHA-256:
  `851a61ddd88c0ddb4a05762f11e900f132fb771fd9dba743ac2026637ddf15d9`
- PoseNet: `0.00049675`
- SegNet: `0.00060969`
- Samples: `600`
- Hardware: Tesla T4, `promotion_eligible=true`
- Delta versus C067:
  `-0.00008428702911766894`

This supersedes C067 as the current A++ frontier, but it does not close the
PR75 public gap. The public PR75 body fields still recompute to about
`0.31470817503416587`, subject to PR-body rounding. The exact measured lesson
is narrower and useful: charged PR75 tile actions transfer into the C067 basin,
while other PR75 streams are antagonistic.

Full PR75 stream T4 replay landed as non-frontier evidence:

- Candidate: `c067_pr75_model_pose_actions_p3`
- Job: `exact_eval_c067_pr75_model_pose_actions_p3_t4_20260503T0358Z`
- Score: `0.31577689189222813`
- Archive bytes: `276751`
- Archive SHA-256:
  `f14f3c8d68d5a5c3ed9308217e697c1af004f93fba91974a041df2e9577aec94`
- PoseNet: `0.00049757`
- SegNet: `0.00060961`
- Samples: `600`
- Hardware: Tesla T4, component gates passed

Diagnostic subset/action-amplitude harvests:

| candidate | hardware | score | bytes | PoseNet | SegNet | status |
|---|---|---:|---:|---:|---:|---|
| `c067_pr75_actions_top40_p3` | RTX PRO | `0.31521426153808474` | `276386` | `0.00049493` | `0.00060829` | best diagnostic; T4 queued |
| `c067_pr75_actions_top25_p3` | RTX PRO | `0.31523505081518555` | `276328` | `0.00049492` | `0.00060889` | second-best diagnostic; T4 queued |
| `c067_pr75_actions_positive_p3` | L40S | `0.31527415672744813` | `276430` | not promoted | not promoted | diagnostic only |
| `c067_pr75_actions_pose_safe_positive_p3` | L40S | `0.3153341068700835` | `276336` | not promoted | not promoted | diagnostic only |
| `c067_pr75_actions_all_ampminus1_p3` | RTX PRO | `0.3153264988900728` | `276461` | not promoted | not promoted | diagnostic only |
| `c067_pr75_actions_positive_poseharm_ampminus1_p3` | RTX PRO | `0.3152678956257131` | `276428` | not promoted | not promoted | diagnostic only |
| `c067_pr75_actions_poseharm_ampminus1_p3` | L40S | `0.3153122462217741` | `276462` | `0.00049613` | `0.00060791` | diagnostic only |

T4/equivalent promotion harvest:

- `exact_eval_c067_pr75_actions_top40_p3_t4_20260503T0440Z`
  - State:
    `.omx/state/c067_pr75_actions_top40_p3_t4_batch_jobs_20260503T0440Z.json`
  - Artifact dir:
    `experiments/results/lightning_batch/exact_eval_c067_pr75_actions_top40_p3_t4_20260503T0440Z`
  - Score: `0.3155226919767294`
  - Archive bytes: `276386`
  - SHA:
    `9feef7ffaa254f9e5408996a122682757a054144a3000539553786c5292b7d0a`
  - PoseNet: `0.00049633`
  - SegNet: `0.00061038`
  - Hardware: Tesla T4, `600` samples, `promotion_eligible=true`
  - Status: new A++ frontier, superseding C082 by
    `0.000010051778634243068`.
- `exact_eval_c067_pr75_actions_top25_p3_t4_20260503T0440Z`
  - State:
    `.omx/state/c067_pr75_actions_top25_p3_t4_batch_jobs_20260503T0440Z.json`
  - Artifact dir:
    `experiments/results/lightning_batch/exact_eval_c067_pr75_actions_top25_p3_t4_20260503T0440Z`
  - Score: `0.3155351257674013`
  - Archive bytes: `276328`
  - SHA:
    `2ee07ed8069b3f3cba668f509e71f20f887f501ad1d4e6dc37c23b99ac377747`
  - PoseNet: `0.00049616`
  - SegNet: `0.00061101`
  - Hardware: Tesla T4, `600` samples, `promotion_eligible=true`
  - Status: A++ non-frontier.

Decision:

- Promote `c067_pr75_actions_top40_p3` as C088 and use its component trace as
  the next action-subset response surface.
- The T4 replay preserved only a small part of the RTX diagnostic advantage, so
  do not rely on RTX/L40S ordering for micro-ranking. Diagnostics remain useful
  only as cheap filters before T4 replay.
- The public PR75 body-field recompute remains about `0.31470817503416587`,
  so sub-0.314 needs a larger move than record pruning: learned/custom
  tile-action dictionaries, renderer self-compression, or a representation
  stream change that survives exact CUDA.

## 2026-05-03T07:00Z - Current A++ frontier and no-expense renderer burn posture

Latest exact score truth:

- Candidate: `c082_qp1_p6_delta_varint_actions_stream_resweep`
- Job: `exact_eval_c082_qp1_p6_delta_varint_actions_stream_resweep_t4_20260503T0626Z`
- Artifact dir:
  `experiments/results/lightning_batch/exact_eval_c082_qp1_p6_delta_varint_actions_stream_resweep_t4_20260503T0626Z`
- Score: `0.3154889937553647`
- Archive bytes: `276394`
- Archive SHA-256:
  `9b78333dd39c12c986ca7fc02a4bb35a6a61ef5a1cc0c9c9c840820eb840058a`
- PoseNet: `0.00049675`
- SegNet: `0.00060969`
- Samples: `600`
- Hardware: Tesla T4, `promotion_eligible=true`
- Evidence grade: `A++ contest T4`

This supersedes the earlier `c067_pr75_qp1_lag_eval_top67_p6_t4` result
(`0.3154979650614253`) as the current exact frontier.

Immediate pure-rate follow-up:

- Candidate archive:
  `experiments/results/c067_mask_topology_sub314_c088_lossless_repack_20260503/c082_p6_delta_varint_actions_stream_resweep/archive.zip`
- Bytes: `276333`
- SHA-256:
  `30932c684c6b09a7bd2bd248e17fd66a4bc448ed2fe5747f92e74bcceec66681`
- Claim: none yet; it is queued as an exact T4 eval under
  `c082_p6_stream_resweep_276333_t4`.
- Job:
  `exact_eval_c082_p6_stream_resweep_276333_t4_20260503T0705Z`
- Staging manifest:
  `.omx/state/c082_p6_stream_resweep_276333_t4_20260503T0705Z_manifest.json`
- Manifest SHA-256:
  `aa6e997eae6880b89e9a20829e2b0e3e9ecbb72b9b693e59b6902bbe475df274`

Renderer self-compression posture:

- The old Modal A10G/H100 fixed-renderer burns failed before training signal
  because the training loop indexed a 600-frame half-frame mask stream with
  1200-frame full-frame offsets.
- The half-frame mask-index bug is fixed in
  `src/tac/experiments/train_renderer.py` and covered by
  `src/tac/tests/test_train_renderer_half_frame_noise.py`.
- Fresh no-expense retraining hedges are running on Modal:
  - H100: `fc-01KQP9K42CAWJH7XEV4KC0V28M`
  - A100: `fc-01KQP9T1VD14785MG63H7JM5VK`
  - A10G: `fc-01KQP9T19Y7PMDETDN99WDMF2W`
- These are not score evidence. A trained snapshot must still pass transplant
  byte closure, renderer pose-safety preflight, and exact CUDA/T4 auth eval
  before it can affect the frontier.

Decision:

- Keep exact T4 harvest on all already-running P6/action candidates.
- Treat the `276333` C082 repack as the next likely micro-frontier if exact
  CUDA preserves decoded-stream parity.
- Keep the main high-EV spend on renderer self-compression; action/packing
  micro-gains are useful but cannot plausibly close the remaining sub-0.314
  gap alone.

## Frontier update - 2026-05-03T07:12Z

The first completed P6/action follow-up produced a new A++ frontier:

- Candidate: `c067_pr75_qp1_top40_p6`
- Job: `exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z`
- Artifact dir:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z`
- Score: `0.3154707273953505`
- Archive bytes: `276342`
- Archive SHA-256:
  `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8`
- PoseNet: `0.00049601`
- SegNet: `0.00061038`
- Samples: `600`
- Hardware: Tesla T4, `promotion_eligible=true`
- Evidence grade: `A++ contest T4`

This supersedes `c082_qp1_p6_delta_varint_actions_stream_resweep`
(`0.3154889937553647`) by `0.0000182663600142`.

Two other completed P6/action T4 packets are exact non-frontier evidence:

- `c067_pr75_qp1_lag_eval_pose4_top67_p6`: score
  `0.31552758560163663`, bytes `276338`, SHA-256
  `de7d549ec4437b3ed9508c1f62f8a79bef88440ba3db0f4a7de5a2c83fe160ef`.
- `c067_pr75_qp1_pose_safe_positive_ampminus1_p6`: score
  `0.31556196759570776`, bytes `276317`, SHA-256
  `6e6ec4609c6da581b4113c5c06e67970542a9d8e22c4866959fe618aaff2c796`.

Operational read:

- The action-atom compiler is real and T4-transferable, but current P6
  variants are in the `1e-5` improvement regime.
- T4 exact replay reorders trace predictions at this scale; use trace-EV for
  queue ordering only, not claims.
- Keep the C082 `276333` exact eval and renderer-shrink exact evals running,
  but spend the main wall-clock budget on larger levers: renderer
  self-compression burns, safe renderer parity shrink, and byte/packer
  candidates that preserve exact decoded behavior.

## Renderer-shrink A-negative - 2026-05-03T07:22Z

The locally pose-safe renderer-shrink candidate was exact-evaluated on T4 and
L40S. It is a scoped negative, not a frontier move:

- Candidate: `zero_fp4_frame1_head_0.1`
- Archive bytes: `275900`
- Archive SHA-256:
  `bb8d1835303478183465e15b28fd9af8776b4ea1c07cdd6cfe0895032a267b64`
- T4 job:
  `exact_eval_renderer_zero_fp4_frame1_head_010_t4_20260503T0712Z`
- T4 score: `0.5066348615388583`
- T4 PoseNet: `0.00685808`
- T4 SegNet: `0.00061045`
- L40S diagnostic score: `0.5037031267845359`
- L40S PoseNet: `0.00671492`
- Evidence: `A-negative scoped forensic`; promotion disabled by PoseNet gate.

Interpretation:

- Saving several hundred renderer bytes by zeroing the frame1 head is not a
  safe transform despite local pose-safety heuristics.
- The failure is hardware-stable across T4 and L40S and isolated to PoseNet;
  SegNet remains in band.
- Do not broadly retire renderer self-compression. Retire only this measured
  zero-FP4 frame1-head implementation. The next renderer path must be learned
  bit-depth/self-compressed or a strictly localized transform with a stronger
  raw-output/pose-pair proof before exact eval.

Tooling follow-up:

- `scripts/launch_lightning_batch_job.py` was missing a `stop` subcommand when
  the redundant L40S diagnostic became unnecessary. A bounded stop command and
  regression test were added immediately so future agents do not fall back to
  ad hoc provider SDK snippets.

## Sub-0.314 swarm update - 2026-05-03T07:35Z

Current active frontier remains C-089:

- Score: `0.3154707273953505`
- Archive bytes: `276342`
- SHA-256:
  `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8`

Compression council result:

- C-089 needs `2209` bytes saved at unchanged components to break `0.314`.
- Semantic parsing of the C-089 single-member payload now exposes
  `masks.mkv`, `renderer.bin`, `seg_tile_actions.bin`, and
  `optimized_poses.qp1` in the bit-budget profiler.
- Generic nested recompression is exhausted for the visible payloads; the next
  compression win must be semantic: learned renderer/self-compression, mask
  geometry with exact gates, pose/action component improvement, or a true
  representation change.

Worker outputs:

- PR75 action dictionary v2 produced closed local candidates, but the strongest
  candidates are byte/component micro-gains. They remain local/optional until a
  stronger trace or stack reason justifies T4 queue time.
- QP1 active-subspace produced a non-no-op, roundtrip-gated pose candidate:
  `ref_active_combined_top32_s0125`, bytes `276398`, SHA-256
  `944c2ba5af9c2d9c5897e4913f9c476d12d12884d964cdd8531716cf4ec92dc1`,
  proxy score `0.3151098005189211`.

Dispatch:

- Claimed `lane_line_search_pose_refinement` for
  `exact_eval_qp1_active_ref_combined_top32_t4_20260503T0732Z`.
- Staged `1376` files / `24031099` bytes through manifest
  `.omx/state/qp1_active_ref_combined_top32_t4_20260503T0732Z_manifest.json`.
- Submitted exact T4 eval on `g4dn.xlarge` with explicit CUDA-12 torch env pins:
  `INFLATE_TORCH_SPEC=torch==2.5.1+cu124`,
  `UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124`, and
  `UV_INDEX_STRATEGY=unsafe-best-match`.

## Renderer-stack A++ negative and preflight hardening - 2026-05-03T16:05Z

C104 has landed as an exact T4 A-negative, not a frontier:

| candidate | bytes | SHA-256 | PoseNet | SegNet | exact score | verdict |
|---|---:|---|---:|---:|---:|---|
| `c101_renderer_x_top192` | `275683` | `d79d1556b55ba7e36c5aaf91d5b04320587975f1303698d8f1089bd5f399d0f3` | `0.04611362` | `0.00062028` | `0.9246640994742736` | A++ scoped negative |

Evidence:

- Job: `exact_eval_c101_renderer_x_top192_stack_t4_20260503T1540Z`
- Artifact dir:
  `experiments/results/lightning_batch/exact_eval_c101_renderer_x_top192_stack_t4_20260503T1540Z/`
- Adjudication status: `REGRESSION_AND_COMPONENT_GATE_REVIEW_REQUIRED`
- Failure class: PoseNet collapse after renderer-stream replacement.

This retires only the measured C101 renderer-stream transplant configuration.
It does not kill renderer self-compression generally. The useful conclusion is
that renderer replacement must be pose-safety gated before exact eval; byte
closure and runtime unpack validation are not enough.

C105 inherited the same renderer parent and was stopped after C104 landed:

- Job: `exact_eval_c101_renderer_native_action_top64_t4_20260503T1605Z`
- Status: `Stopped`
- Cost: about `$0.0044`
- No score claim and no scientific conclusion beyond "derivative of a measured
  bad parent should not consume T4."

Permanent guard added:

- `experiments/build_c101_renderer_pose_stack_candidate.py` now emits
  `exact_eval_dispatch_gate` in candidate manifests. Renderer-stack candidates
  are fail-closed unless a matching
  `experiments/preflight_renderer_transplant_pose_safety.py` report has
  `safe_for_exact_eval_dispatch=true` for the exact source and candidate archive
  SHA pair.
- `scripts/launch_lightning_batch_job.py` now blocks exact-eval Studio submits
  when a sibling `manifest.json` contains a required dispatch gate that is not
  safe.
- Focused verification:
  `.venv/bin/python -m py_compile experiments/build_c101_renderer_pose_stack_candidate.py scripts/launch_lightning_batch_job.py`
  and
  `.venv/bin/python -m pytest src/tac/tests/test_build_c101_renderer_pose_stack_candidate.py src/tac/tests/test_lightning_batch_jobs.py::test_exact_eval_manifest_dispatch_gate_blocks_renderer_stack_without_pose_safety -q`
  passed with `7 passed`.

Lane 12 current audit:

- Sagan completed a no-dispatch audit at
  `.omx/research/lane12_l2_clearance_current_audit_20260503_codex.md`.
- `decoded_baseline_contract_preflight_passed=true` and
  `runtime_closure.passed=true`, but `passing_geometry_count=0`,
  `usable_pose_regeneration_provenance_count=0`, and
  `.omx/state/lane12_nerv_l2_clearance.json` remains absent.
- The largest existing local Lane 12 CDO1 repair candidate still has
  `global_disagreement_after=0.01221874237060547`, far above the `0.001`
  promotion gate. Lane 12 remains structurally blocked until a real
  promotion-threshold Alpha-Geo JSON and matching pose-regeneration provenance
  exist.

## Active-target preflight correction - 2026-05-03T16:20Z

The mask-packer planner still carried the obsolete sub-`0.314` target in its
rate-only break-even math. That is now a bug class because the active target is
strict `<=0.31`.

Permanent fix:

- `experiments/plan_c091_mask_packer_bigmove.py` now uses
  `DEFAULT_TARGET_SCORE = 0.31`, exposes `--target-score`, and records both
  `target_score` and target-relative break-even byte fields.
- Legacy `sub314_*` JSON keys remain for older ledgers, but they follow the
  explicit target instead of hard-coding `0.314`.
- Focused tests in `src/tac/tests/test_plan_c091_mask_packer_bigmove.py`
  verify explicit `0.314` compatibility and default `0.31` behavior.

Re-run on C102/top192:

```bash
.venv/bin/python experiments/plan_c091_mask_packer_bigmove.py \
  --frontier-archive experiments/results/lightning_batch/exact_eval_c091_next_cem_pose_waterfill_top192_s0125_m06_t4_20260503T1238Z/archive.zip \
  --frontier-eval-json experiments/results/lightning_batch/exact_eval_c091_next_cem_pose_waterfill_top192_s0125_m06_t4_20260503T1238Z/contest_auth_eval.adjudicated.json \
  --output-dir experiments/results/c102_mask_packer_bigmove_20260503_codex \
  --lossy-probe-frames 0 \
  --target-score 0.31
```

Result: exact-lossless/micro packer work is not enough. The best existing
byte-screen candidate still needs about `7365` byte-equivalent component/rate
improvement against the active `0.31` target. This reinforces the current
dispatch priority: renderer training/export with pose-safety, learned/slim JFG
student, and PR65/PR75/PR77 as priors for native atoms rather than direct
transplants.
- Status at submit: `Pending`, zero cost. No score claim until
  `contest_auth_eval.json` is harvested and adjudicated.

Active higher-EV spend:

- Modal fixed-renderer burns remain running on H100, A100, and A10G. Their
  outputs are training artifacts only until recovered, exported, preflighted by
  renderer-transplant pose-safety, packaged into charged archive bytes, and
  exact CUDA evaluated.

## Public-top delta refresh - 2026-05-03T07:42Z

The public leaderboard and PR refresh changed the tactical read:

- Official leaderboard still shows PR67/PR65/PR63 rounded at `0.32`.
- PR67 current release URL downloads as `276620` bytes with SHA-256
  `86c8694adf8bf53a09a2f2162285601be51ae3030572c73d97f85f3db04c85b8`.
- PR67 body text states a different archive: `276741` bytes with SHA-256
  `cc1dfa346811dade15044910fa0143e73a7a049b98aeec39b96a53481a0788bd`.
  Do not conflate those artifacts.
- C-089 is already smaller than both PR67 public variants, so the missing
  sub-0.314 signal is primarily component behavior, not bulk lossless packing.
- PR65/Henosis has much better PoseNet in the body-stated report but worse
  rate and SegNet. Treat its pose/postfilter streams as atom dictionaries for
  C-089 hard-pair corrections, not as a wholesale transplant.

Parser hardening landed from this pass:

- `submissions/robust_current/unpack_renderer_payload.py` now accepts the
  current PR67 `55914` renderer slice and validates fixed-slice splits by
  Brotli/QZS3/QP1 content rather than a single hard-coded model length.
- Focused parser test now covers this slice family.

Immediate consequence:

- After the active QP1 T4 hedge returns, the next high-density local analysis
  should be a three-way raw-output/component trace:
  C-089 vs PR67-current vs PR67-body-cached, followed by PR65 qpost/pose
  atom selection or PR75 action dictionary-v2, whichever has better
  benefit-per-byte and no-op proof.

## Renderer group allocator local negative - 2026-05-03T07:50Z

The renderer tensor/group allocator is now implemented as a local guarded tool,
but the current C-089 renderer does not have a dispatchable hand-built block
allocation:

- Built/preflighted `21` MQZ1 renderer-group candidates.
- Best byte saver:
  `experiments/results/renderer_group_allocator_worker_20260503/group_frame2_head_b128/archive.zip`
  at `276011` bytes, SHA-256
  `19abbe2ba353dee0888c379150392f189d958ccd32faf922e3f793eaed5fa569`.
- This saves `331` bytes, far short of the `2209` unchanged-component target,
  and fails local pose-safety with max absolute output delta `170.275390625`.
- The only local-safe candidates were byte-regressive (`+233` and `+288`
  bytes).

Decision: do not exact-eval this allocator wave. Keep the allocator as a
finishing pass after trained renderer export/transplant; do not spend T4 on
current-renderer group allocation alone.

## PR65 qpost atom exact wave - 2026-05-03T08:00Z

The QP1 active-subspace T4 hedge completed and did not move the frontier:

- job: `exact_eval_qp1_active_ref_combined_top32_t4_20260503T0732Z`
- score: `0.3155101071985553`
- bytes: `276398`
- SHA-256:
  `944c2ba5af9c2d9c5897e4913f9c476d12d12884d964cdd8531716cf4ec92dc1`
- PoseNet: `0.00049604`
- SegNet: `0.00061038`
- hardware: Tesla T4
- evidence: A++ exact T4, component gates passed, no frontier.

This reinforces that scalar/subspace pose polish is not enough by itself.

PR65/Henosis qpost streams are now being tested as charged, pair-filtered atom
subsets against the C-089 frontier. These are exact-score candidates, not score
claims until harvested:

| job | candidate | bytes | SHA-256 | status at queue | rationale |
|---|---|---:|---|---|---|
| `exact_eval_pr65_qpost_bias_poseadv_top032_t4_20260503T0756Z` | `pr65_qpost_bias_poseadv_top032` | `276551` | `91cd1690fdcca47eedbe12dc8bdaba8191210199169449c474730ad30831efbd` | Running | Smallest bias-only point whose public-trace opportunity exceeds the sub-0.314 break-even estimate. |
| `exact_eval_pr65_qpost_bias_poseadv_top064_t4_20260503T0759Z` | `pr65_qpost_bias_poseadv_top064` | `276590` | `656d438c8723482e3500df33d4b7a02d3168b1f42004dfdec4d97ae37281029c` | Running | Higher-upside scale point from the first qpost atom builder. |
| `exact_eval_pr65_qpost_v2_bias_poseadv_top040_t4_20260503T0805Z` | `v2_pr65_qpost_bias_poseadv_top040` | `276561` | `cc44cb245e7772076fd136d457e4b4f549c726db9985a826de2d64b5e29db51e` | Pending | Better-calibrated middle point with planning trace slack `0.00031728568692550784` vs target. |
| `exact_eval_pr65_qpost_v2_bias_poseadv_top080_t4_20260503T0810Z` | `v2_pr65_qpost_bias_poseadv_top080` | `276600` | `5bb174735af9db4f32963933b4c6279777da2b0d90a16332ca339475eabf4da7` | Pending | Aggressive bias-only scale hedge with highest v2 trace slack, still byte-closed and charged. |

All four were claimed under `lane_qzs3_postprocess_sidecar`; top64/top40/top80
are explicit bounded children of the top32 claim. They use the same canonical
Lightning T4 exact-eval path, staged manifests, and CUDA-12 torch env pins as
the current C-089 frontier.

Self-compression nextwave planning also landed:

- Generic nested recompression is exhausted for C-089. The unchanged-distortion
  byte-only target remains `2209` bytes.
- The highest-EV byte path is a trained renderer self-compression transplant,
  but it must wait for Modal artifacts and then pass transplant and pose-safety
  preflight before any exact eval.
- Lossless mask transcode target is `<=217263` bytes if decoded masks are
  preserved exactly; no deployable candidate exists yet.

Immediate decision rule:

- If any qpost atom point beats C-089, promote the best exact T4 result into
  the claim matrix and stack only with byte-preserving/reproducible packers.
- If all qpost points regress, classify the qpost bias-only implementation as
  measured negative and pivot the main lane back to PR75 component-basin
  recovery plus trained renderer transplant.

## PR65 qpost exact harvest and interaction follow-up - 2026-05-03T10:30Z

The four C-089 PR65/Henosis bias-only qpost screens completed on exact Tesla
T4 and are all A++ no-frontier results. Component gates passed, but each point
increased PoseNet enough that the charged qpost sidecar bytes became a net
score regression:

| job | bytes | SHA-256 | PoseNet | SegNet | exact T4 score | delta vs C-089 |
|---|---:|---|---:|---:|---:|---:|
| `exact_eval_pr65_qpost_bias_poseadv_top032_t4_20260503T0756Z` | `276551` | `91cd1690fdcca47eedbe12dc8bdaba8191210199169449c474730ad30831efbd` | `0.00049626` | `0.00061038` | `0.31562772378789217` | `+0.00015699639254168618` |
| `exact_eval_pr65_qpost_bias_poseadv_top064_t4_20260503T0759Z` | `276590` | `656d438c8723482e3500df33d4b7a02d3168b1f42004dfdec4d97ae37281029c` | `0.00049670` | `0.00061038` | `0.3156849465853906` | `+0.00021421919004011025` |
| `exact_eval_pr65_qpost_v2_bias_poseadv_top040_t4_20260503T0805Z` | `276561` | `cc44cb245e7772076fd136d457e4b4f549c726db9985a826de2d64b5e29db51e` | `0.00049632` | `0.00061038` | `0.3156384822569176` | `+0.0001677548615671154` |
| `exact_eval_pr65_qpost_v2_bias_poseadv_top080_t4_20260503T0810Z` | `276600` | `5bb174735af9db4f32963933b4c6279777da2b0d90a16332ca339475eabf4da7` | `0.00049693` | `0.00061038` | `0.3157077620893657` | `+0.0002370346940152035` |

Failure classification: measured implementation/config negative for direct
C-089 bias-only qpost atom sidecars. This does not kill qpost/postfilter
families broadly; it says the public-trace opportunity estimator over-valued
these bias atoms under the robust runtime, and PoseNet reacted adversely.

Follow-up already queued:

- `exact_eval_pr65_qpost_ix_lagtop67_p6_bias_top080_t4_20260503T1024Z`
- archive:
  `experiments/results/pr65_qpost_interaction_worker_20260503/candidates/pr75_lagtop67_p6/ix_pr75_lagtop67_p6_bias_top080/archive.zip`
- bytes: `276610`
- SHA-256:
  `b91f03162758329f97382b42a32f3937c1f7288b589a96fcf820761302a2e51b`
- status at recording: Pending, zero cost.

This interaction job is still justified because it starts from the PR75
lag-top67 P6 action archive instead of C-089 and therefore tests a different
action/qpost basin. If it also regresses, qpost should become a calibration
input for learned/renderer post-correction rather than a direct sub-0.314
dispatch lane.

## Renderer transplant C-089 grammar repair and Q-FAITHFUL 2146 negative - 2026-05-03T10:45Z

The C-089 payload uses the PR75/P6/QP1/action grammar, while the older trained
renderer transplant preflight expected the old `optimized_poses.bin` logical
member. That stale preflight contract blocked evaluation of the highest-EV
renderer self-compression path before it could even run pose-safety.

Permanent additive repair:

- `experiments/build_renderer_shrink_candidate.py` now accepts
  `--renderer-export` for an external QZS3 trained renderer while preserving
  every non-renderer logical member from the source archive.
- The builder preserves the current PR75/P6/QP1/action payload grammar and
  emits explicit renderer-input metadata, direct-export candidates, and
  QZS3-reencoded candidates.
- Regression coverage landed in
  `src/tac/tests/test_build_renderer_shrink_candidate.py`.

Verification:

```text
.venv/bin/python -m py_compile \
  experiments/build_renderer_shrink_candidate.py \
  src/tac/tests/test_build_renderer_shrink_candidate.py

.venv/bin/python -m pytest src/tac/tests/test_build_renderer_shrink_candidate.py -q

2 passed

git diff --check -- \
  experiments/build_renderer_shrink_candidate.py \
  src/tac/tests/test_build_renderer_shrink_candidate.py
```

The repaired builder produced a real byte-crossing C-089 transplant family from
the harvested Q-FAITHFUL 2026-05-01T2146Z renderer:

| candidate | bytes | SHA-256 | byte delta vs C-089 | byte-only note |
|---|---:|---|---:|---|
| `external_qzs3_b1024_pr75_preserved_slices` | `270372` | `57e2640ace74af9dd0b60024e8fdcf906c113958806fcd555f43b2045791e507` | `-5970` | crosses sub-0.314 by bytes if components held |
| `external_qzs3_b0096_pr75_preserved_slices` | `273514` | `9f0ffe769280ff85c4a8ecabcdefc9f3165eca146b7aecd55ebdc9d38aeba86e` | `-2828` | crosses sub-0.314 by bytes if components held |
| `external_qzs3_direct_pr75_preserved_slices` | `277159` | `4e0304b8dcc2dde73816f416596161138856aa8b466d543898be7aace5eb8d13` | `+817` | transplant control, not byte-useful |

All three failed local renderer transplant pose-safety before any remote
dispatch:

| candidate | sampled pairs | mean abs delta | RMS delta | max abs delta | verdict |
|---|---:|---:|---:|---:|---|
| `external_qzs3_b1024_pr75_preserved_slices` | `32` | `72.0155` | `87.1190` | `254.9581` | unsafe |
| `external_qzs3_b0096_pr75_preserved_slices` | `8` | `70.7392` | `86.2799` | `254.9391` | unsafe |
| `external_qzs3_direct_pr75_preserved_slices` | `8` | `72.1968` | `87.6552` | `254.9391` | unsafe |

Failure classification: measured local preflight negative for transplanting the
old Q-FAITHFUL 2146 trained renderer into the C-089 PR75/P6/QP1 archive. This
does not kill trained renderer self-compression. It says old renderer weights
are not geometry-compatible with the current C-089 mask/pose/action basin. The
active Modal fixed-mask/fixed-pose burns remain the correct renderer path,
because their training contract is aligned to C-089 instead of the older
Q-FAITHFUL source archive.

## PR75/minp grammar hardening and stream-ablation wave - 2026-05-03T11:00Z

The current public PR75/minp submission was re-profiled from actual archive
bytes and robust-current runtime parity instead of inferred chat notes.

Public source:

- PR: `https://github.com/commaai/comma_video_compression_challenge/pull/75`
- archive bytes: `276481`
- archive SHA-256:
  `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746`
- payload member: single stored ZIP member `p`, `276381` bytes.

Recovered charged slices:

| stream | charged bytes | decoded/runtime bytes | SHA-256 note |
|---|---:|---:|---|
| `masks.mkv` | `219472` | `223385` | decoded mask stream matches C-089 exactly |
| `renderer.bin` | `55756` | `59288` | differs from C-089 |
| `seg_tile_actions.bin` | `255` | `432` runtime raw4 records | SG2 grouped tile/frame delta-varint; `108` records |
| `optimized_poses.qp1` | `898` | `1140` | differs from C-089 |

Robust-current now parses this current public payload shape and converts SG2
actions to runtime raw4 records. Zeno's sampled CPU raw-output parity passed
bit-exact for public runtime versus robust-current on pairs `33`, `104`, and
`598`; this is not score evidence, but it raises confidence that the remaining
question is stream interaction under exact CUDA scoring.

Exact T4 work currently in flight:

| candidate | archive bytes | SHA-256 | purpose |
|---|---:|---|---|
| `pr75_minp_public_replay` | `276481` | `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746` | reproduce public PR75/minp through robust-current exact CUDA |
| `c089_p6_resweep_pr65_qpost_bias_top032` | `276542` | `3816340f572d21df56fa0ca00e64f51ae8e7f6b7556353d69c7a1453e6d3051f` | qpost microstack sanity after direct qpost negatives |
| `public_renderer_only` | `276171` | `8da9ffe354923dbe34ffa45e261f824a63e21fef45a1cc29fdab6032a6402820` | best byte stream ablation: public renderer plus C-089 masks/actions/pose |
| `public_renderer_actions` | `276270` | `110c3d4eca27fa45bec7bbb3ac38a872b4f0ceb97c7118de851c27d8e47f704a` | coupling test for public renderer plus public SG2 actions |
| `public_renderer_pose` | `276392` | `65796c4d072a1db94d6d243452ce8ec922e7a600a47403f6b81d92c13babde23` | coupling test for public renderer plus public QP1 pose |
| `public_actions_only` | `276479` | `dc9b22624b3f10e3c7cbc65047bb448ecda10cfe1dd5ad3f0e30cc990ec423d8` | isolate public SG2 action contribution against C-089 renderer/pose |
| `public_pose_only` | `276601` | `1758f2cf7425dd222c1abceeffd511c2712c2171997e8b033c9a2efe1dd6b829` | isolate public QP1 pose contribution against C-089 renderer/actions |
| `p6_public_renderer_only` | `276132` | `189301b6b7b8bc7f166ee7ec835bb71ed61ac56570133ac2bfea975b4377dec7` | P6-cleaner public-renderer-only stream mix; `210` bytes under C-089 |
| `p6_public_renderer_pose` | `276353` | `c5d2d83cdfc128cafe4ae59278ba872cc9ef423d5dd32a6ab31b95648343439a` | P6-cleaner public renderer+pose interaction hedge; `39` bytes smaller than the P3 sibling |

P6 stream-mix notes as of 2026-05-03T11:08Z:

- `p6_public_renderer_only` and `p6_public_renderer_pose` are queued through
  manifest-staged Lightning T4 exact-eval jobs with active dispatch claims.
- Public-action P6 mixes remain intentionally skipped: the public SG2 action
  records are not nondecreasing by pair index, so converting them into the P6
  delta-varint wire format would silently change decoded semantics.
- `p6_c089_action_resweep` is a deterministic one-byte smaller encoded-stream
  control, but it has no plausible sub-0.314 headroom by itself and is held out
  of the current exact-eval wave unless a later stack needs the byte.

## PR77 tile-delta public intake - 2026-05-03T11:17Z

PR77 is distinct from the withdrawn PR78 script-payload submission. PR78 is
non-faithful for our purposes because the author withdrew it as a
rules-interpretation payload-relocation submission; it is useful only as a
negative compliance note.

PR77 source:

- PR: `https://github.com/commaai/comma_video_compression_challenge/pull/77`
- archive URL: `https://github.com/user-attachments/files/27314022/archive.zip`
- local archive:
  `experiments/results/top_submission_reverse_engineering_20260503_pr77/archive.zip`
- archive bytes: `276551`
- archive SHA-256:
  `f90880383c95e14d82704f99db9b20944786ae6452a844348638b06c439972af`
- branch clone: `/tmp/pr77-tile-delta`, commit
  `e9512c284dcd233dc5c6a7ed3362a943f8f5e340`.

Recovered PR77 single-member payload anatomy:

| stream | charged bytes | decoded/runtime bytes | SHA-256 note |
|---|---:|---:|---|
| `masks.mkv` | `219472` | `223385` | same decoded mask stream as PR75/minp and C-089 |
| `renderer.bin` | `55756` | `59288` | same public minp renderer stream as PR75/minp |
| `seg_tile_actions.bin` | `325` | `588` runtime raw4 records | PR77 grouped tile-delta action stream |
| `optimized_poses.qp1` | `898` | `1140` | same public minp pose stream as PR75/minp |

Permanent parser hardening:

- `submissions/robust_current/unpack_renderer_payload.py` now includes the
  observed PR77 fixed-slice tuple `(276451, 55756, 325)`.
- `experiments/archive_bit_budget_profiler.py` mirrors that tuple for byte
  attribution.
- Focused parser/profiler tests cover the PR77 grouped tile-delta action
  variant.

Verification:

```text
.venv/bin/python -m py_compile \
  submissions/robust_current/unpack_renderer_payload.py \
  experiments/archive_bit_budget_profiler.py \
  src/tac/tests/test_unpack_renderer_payload_fixedslice.py \
  src/tac/tests/test_archive_bit_budget_profiler.py

.venv/bin/python -m pytest \
  src/tac/tests/test_unpack_renderer_payload_fixedslice.py \
  src/tac/tests/test_archive_bit_budget_profiler.py -q

21 passed, 1 warning
```

Exact replay queued:

| candidate | archive bytes | SHA-256 | purpose |
|---|---:|---|---|
| `pr77_qzs3_tile_delta_r147_public_replay` | `276551` | `f90880383c95e14d82704f99db9b20944786ae6452a844348638b06c439972af` | T4 exact replay/component trace of PR77 tile-delta basin |

No score claim exists until the queued T4 auth eval emits
`contest_auth_eval.json` and adjudication for the exact archive SHA above.

## PR75 replay frontier and renderer-only negative - 2026-05-03T11:30Z

The public PR75/minp replay has now landed as exact T4 evidence through
`robust_current`:

- Candidate: `pr75_minp_public_replay`
- Job: `exact_eval_pr75_minp_public_replay_t4_20260503T1049Z`
- Artifact dir:
  `experiments/results/lightning_batch/exact_eval_pr75_minp_public_replay_t4_20260503T1049Z`
- Score: `0.31516575028285976`
- Archive bytes: `276481`
- Archive SHA-256:
  `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746`
- PoseNet: `0.00049371`
- SegNet: `0.00060804`
- Samples: `600`
- Hardware: Tesla T4, `promotion_eligible=true`
- Evidence grade: `A++ contest T4`

This supersedes C-089 as the exact score anchor by
`0.000304977112490723`. It is an external public replay anchor, not an
original stack breakthrough. At unchanged components, the remaining gap to
`0.314` is about `1751` bytes, so the next useful candidates must either save
mask/renderer/action bytes without component loss or improve PoseNet/SegNet
while staying byte-neutral.

Two more PR75-derived T4 packets landed and did not move the frontier:

- `c089_p6_resweep_pr65_qpost_bias_top032`: score
  `0.3156217237878922`, bytes `276542`, SHA-256
  `3816340f572d21df56fa0ca00e64f51ae8e7f6b7556353d69c7a1453e6d3051f`.
  This closes the qpost microstack as a measured non-frontier point.
- `public_renderer_only`: score `0.31824666346496305`, bytes `276171`,
  SHA-256
  `8da9ffe354923dbe34ffa45e261f824a63e21fef45a1cc29fdab6032a6402820`.
  This is a scoped public-renderer-only negative: it saves bytes, but PoseNet
  worsens to `0.00053746`.

Operational consequence:

- Stop renderer-only same-family hedges unless they include exact new coupling
  evidence. I stopped `p6_public_renderer_only` and Newton's
  `public_renderer_c089_p6_lossless_stream_resweep` at zero cost after the
  exact negative above.
- Keep `public_renderer_actions`, `public_renderer_pose`,
  `p6_public_renderer_pose`, `public_actions_only`, `p6_public_pose_only`, and
  PR77 replay alive because they test coupling or public action/pose basins,
  not public renderer alone.
- Tighten future renderer-transplant interpretation: local pixel-output
  parity preflight is necessary for dispatch hygiene, but it is not sufficient
  to predict PoseNet. Exact CUDA remains the only promotion/ranking signal.

The qpost interaction siblings `top080` and `top040` both completed A++ on T4
but regressed versus C-089:

- `top080`: score `0.3157349872667322`, bytes `276610`, SHA
  `b91f03162758329f97382b42a32f3937c1f7288b589a96fcf820761302a2e51b`.
- `top040`: score `0.3156657157109626`, bytes `276571`, SHA
  `d90912443bc9f60e972a0ac30f190ec1e552fa599a2c0162d8f0f1a99d3bcbd6`.

Failure classification: measured implementation negatives for this qpost
direct/interaction family. They remain useful calibration signal for action
selection and learned post-correction, but they should not consume the main
sub-0.314 exact-eval budget until new component-response evidence changes the
expected value.

Next decision rule:

- If public replay lands at or below the PR75-reported basin, treat public
  renderer/action/pose deltas as the correct target geometry and stack only
  ablations that preserve or improve exact components.
- If `public_renderer_only` holds components, it is immediately near
  sub-0.314 after P6 packing cleanup and any small pose/action byte save.
- If renderer-only fails but renderer+actions or renderer+pose recovers,
  promote the recovered coupling as the next packer/line-search anchor.
- If all public-renderer ablations fail, keep PR75/minp replay as an external
  anatomy reference and shift the main wall-clock lane back to the active
  fixed-mask/fixed-pose renderer self-compression burns.

## PR75 coupled stream ablation harvest - 2026-05-03T11:36Z

Three more PR75-derived stream mixes landed on exact T4. None moved the C-091
public replay anchor:

| candidate | score | bytes | SHA-256 | PoseNet | SegNet | classification |
|---|---:|---:|---|---:|---:|---|
| `public_renderer_actions` | `0.3181017344493514` | `276270` | `110c3d4eca27fa45bec7bbb3ac38a872b4f0ceb97c7118de851c27d8e47f704a` | `0.00053789` | `0.00060804` | A++ scoped negative; public renderer remains pose-toxic even with public actions |
| `public_renderer_pose` | `0.3153386722810012` | `276392` | `65796c4d072a1db94d6d243452ce8ec922e7a600a47403f6b81d92c13babde23` | `0.00049360` | `0.00061044` | A++ non-frontier; improves C-089 but misses C-091 |
| `public_actions_only` | `0.3153942031679369` | `276479` | `dc9b22624b3f10e3c7cbc65047bb448ecda10cfe1dd5ad3f0e30cc990ec423d8` | `0.00049676` | `0.00060817` | A++ non-frontier; improves C-089 but misses C-091 |
| `p6_public_renderer_pose` | `0.3153126722810012` | `276353` | `c5d2d83cdfc128cafe4ae59278ba872cc9ef423d5dd32a6ab31b95648343439a` | `0.00049360` | `0.00061044` | A++ non-frontier; best coupled public-renderer P6 sibling, still worse than C-091 |

Operational consequences:

- Close all harvested stream-ablation claims as `completed_a_pp_no_frontier`
  except the explicit renderer/action negative, which remains a scoped
  measured-implementation negative.
- Keep PR77 replay and `p6_public_pose_only` running because they test action
  grammar and pose-only transfer rather than public-renderer transplant.
- Do not spend more T4 on public-renderer-only or public-renderer-action
  variants unless a new builder changes the actual rendered tensor basin or a
  stricter pose-safety predictor is calibrated against exact CUDA.
- Public renderer+pose is not catastrophic, but its best measured exact score
  is still `0.00014692199814144` worse than C-091. Future use must add either
  real byte savings beyond about `2210` bytes at the same components or a
  component improvement from action/pose optimization.

Online refresh, 2026-05-03:

- Official comma leaderboard still rounds the top faithful lossy-video entries
  to `0.32`; PR67 is listed first, followed by PR65 and PR63.
- PR75/minp reports `276481` bytes and rounded `0.31` with PoseNet
  `0.00048657`, SegNet `0.00060529`; our exact replay of the public archive is
  C-091 at `0.31516575028285976`.
- PR77 reports a same-family QZS3 tile-delta action stream at `276551` bytes
  with PoseNet `0.00049314`, SegNet `0.00060631`; exact replay landed as
  A++ non-frontier evidence at `0.31537874750377204`.
- PR76 is a qpose14 poseq6 entry at rounded `0.34`, not an immediate sub-0.314
  lane.
- PR78's `0.13` script-payload entry is closed and remains non-faithful for
  our purposes because every score-affecting bit must be charged in
  `archive.zip` or fixed contest code.

## PR77 exact replay harvest - 2026-05-03T11:38Z

The PR77 public tile-delta archive is now exact T4 replayed:

- Candidate: `pr77_qzs3_tile_delta_r147_public_replay`
- Job: `exact_eval_pr77_tile_delta_public_replay_t4_20260503T1116Z`
- Artifact dir:
  `experiments/results/lightning_batch/exact_eval_pr77_tile_delta_public_replay_t4_20260503T1116Z`
- Score: `0.31537874750377204`
- Archive bytes: `276551`
- Archive SHA-256:
  `f90880383c95e14d82704f99db9b20944786ae6452a844348638b06c439972af`
- PoseNet: `0.00049588`
- SegNet: `0.00060816`
- Samples: `600`
- Hardware: Tesla T4, `promotion_eligible=true`
- Evidence grade: `A++ contest T4`

Interpretation:

- PR77 improves C-089 by about `0.000091979891578442`, but misses C-091 by
  about `0.00021299722091228`.
- The 147-record tile-delta action grammar is a real action-prior signal, not
  a frontier by itself.
- Use PR77 to guide action subset/dictionary/interaction search only if the
  builder can combine it with C-091/PR75 components or save material bytes
  without reintroducing the renderer-transplant pose penalty.

## PR75 p6 public-pose-only harvest - 2026-05-03T11:42Z

The last live PR75 stream mix also landed on exact T4:

- Candidate: `p6_public_pose_only`
- Job: `exact_eval_pr75_minp_p6_public_pose_only_t4_20260503T1120Z`
- Artifact dir:
  `experiments/results/lightning_batch/exact_eval_pr75_minp_p6_public_pose_only_t4_20260503T1120Z`
- Score: `0.31926623697976314`
- Archive bytes: `276562`
- Archive SHA-256:
  `5e339c212a2cfdee239145a8bebf978177a0770d413fad87c9d62190d0988239`
- PoseNet: `0.00054874`
- SegNet: `0.00061038`
- Samples: `600`
- Hardware: Tesla T4, `promotion_eligible=true`
- Evidence grade: `A++ contest T4`

Interpretation:

- The public pose-only transplant is a measured implementation negative. It
  keeps SegNet near the C-089 reference but damages PoseNet enough to score
  `0.00410048669690338` worse than C-091.
- Do not dispatch more public-pose-only or public-renderer-only stream swaps
  without a new component-basin mechanism. The viable continuation is an
  action/pose/packer candidate that either improves exact components or saves
  at least about `1751` bytes versus C-091 at unchanged components.

## PR77 action-pose mixed candidate dispatch - 2026-05-03T11:44Z

Kant produced a new deterministic local byte-screen over PR77 actions, PR75
mask/renderer streams, and C-089/C-091 pose streams. The best dispatchable
candidate is:

- Candidate: `pr77_actions_pr75mask_renderer_c089pose_fixedslice`
- Archive:
  `experiments/results/pr77_action_pose_mixed_container_20260503_codex/pr77_actions_pr75mask_renderer_c089pose_fixedslice/archive.zip`
- Archive bytes: `276329`
- Archive SHA-256:
  `27866172e76d27113e86a30f722588fd668f81a949be3acbe1e92cddc9a6a1d8`
- Byte delta versus C-091: `-152`
- Break-even still needed versus C-091 for sub-`0.314`: `1599` byte-equivalent,
  or `0.0010645397219851693` score from component improvement.
- Evidence before T4: `empirical_byte_screen_only`; `score_claim=false`.

Dispatch:

- Claim: `pr77_action_pose_fixedslice_t4`
- Job: `exact_eval_pr77_action_pose_fixedslice_t4_20260503T114254Z`
- State:
  `.omx/state/pr77_action_pose_fixedslice_t4_20260503T114254Z_batch_jobs.json`
- Source manifest:
  `.omx/state/pr77_action_pose_fixedslice_t4_20260503T114254Z_manifest.json`
- Local target artifact dir:
  `experiments/results/lightning_batch/exact_eval_pr77_action_pose_fixedslice_t4_20260503T114254Z`

Decision rule:

- If this beats C-091, promote only after state-derived harvest validates the
  exact archive bytes, T4 hardware, adjudicated JSON, and component trace.
- If it misses C-091, use the component trace to decide whether PR77 actions
  are a useful atom prior or only another byte-level stream-packing negative.

## Runtime action grammar hardening - 2026-05-03T11:55Z

While the PR77 mixed candidate was running, a parser split surfaced:

- `submissions/robust_current/unpack_renderer_payload.py` already understood
  PR75/PR77 SG2 grouped tile-action streams and could expand them to runtime
  records.
- `submissions/robust_current/inflate_renderer.py` only accepted already
  expanded 4-byte or 5-byte records, so a future archive carrying `SG2`,
  `TA4`, or `TA5` directly could pass unpack forensics but fail runtime
  inflate.

Permanent fix:

- `inflate_renderer.py` now accepts direct `SG2`, `TA4`, and `TA5` tile-action
  payloads in addition to expanded records.
- `unpack_renderer_payload.py` now preserves the specific self-describing PR75
  action-parser failure reason instead of collapsing malformed P6 actions into
  a generic validation error.
- Focused verification:
  - `py_compile` on the touched runtime/unpacker/test files.
  - `13 passed` for PR75 lossless/action parser, C091-relative action-pose
    matrix, and trained-renderer transplant dispatch readiness tests.

Evidence grade: harness hardening, not score evidence. Operational value:
future action-grammar archives fail closed with useful diagnostics and do not
depend on whether SG2 expansion happened in the unpack stage or in runtime
inflate.

## PR65 pose-transfer public-stream screen - 2026-05-03T11:58Z

Planck's PR65/Henosis anatomy pass found one remaining disjoint public-stream
screen that is cheap enough to run under the no-holds-barred deadline policy:
use PR65's re-encoded QP1 velocity pose stream with C-091's PR75 mask/renderer
and C-089's P6 action stream.

Candidate:

- Candidate: `c091_pr65_pose_qp1_c089_actions_p6`
- Archive:
  `experiments/results/pr65_henosis_stream_transfer_20260503_codex/c091_pr65_pose_qp1_c089_actions_p6/archive.zip`
- Archive bytes: `276346`
- Archive SHA-256:
  `d3913ec75bd1917f16f2bca5e672313f66f81da5446405fc8eccbba757eed79d`
- Byte delta versus C-091: `-135`
- Formula-only score if components matched C-091:
  `0.31507585932418827`
- Component improvement still needed for sub-`0.314`:
  `0.0010758593241882634`

Dispatch:

- Claim: `pr65_pose_qp1_c091_c089_actions_p6_t4`
- Job: `exact_eval_pr65_pose_qp1_c091_c089_actions_p6_t4_20260503T1158Z`
- State:
  `.omx/state/pr65_pose_qp1_c091_c089_actions_p6_t4_20260503T1158Z_batch_jobs.json`
- Manifest:
  `.omx/state/pr65_pose_qp1_c091_c089_actions_p6_t4_20260503T1158Z_manifest.json`
- Status at dispatch: `Pending`, zero cost.

This is not a sub-`0.314` prediction. It is an explicit public pose-basin exact
screen. If it misses C-091, PR65 pose transfer should be treated as scoped
non-frontier evidence unless a new learned/geometry mechanism changes the pose
contract.

## PR77 action-pose fixed-slice exact negative - 2026-05-03T12:01Z

The PR77 action/C-089 pose/C-091 mask-renderer fixed-slice candidate completed
and was harvested through the state-derived Lightning path.

Exact result:

- Job: `exact_eval_pr77_action_pose_fixedslice_t4_20260503T114254Z`
- Artifact dir:
  `experiments/results/lightning_batch/exact_eval_pr77_action_pose_fixedslice_t4_20260503T114254Z`
- Score: `0.318426107391119`
- Archive bytes: `276329`
- Archive SHA-256:
  `27866172e76d27113e86a30f722588fd668f81a949be3acbe1e92cddc9a6a1d8`
- PoseNet: `0.00054190`
- SegNet: `0.00060816`
- Samples: `600`
- Hardware: Tesla T4, `promotion_eligible=true`
- Delta versus C-091: `+0.0032603571082592264`

Interpretation:

- The `152` byte rate win is real but far too small to overcome PoseNet
  regression.
- PR77's 147-record action stream remains useful as an atom prior, but direct
  transplant into C-091's mask/renderer with C-089 pose is a scoped
  non-frontier implementation.
- The next action work should be C-091-native action-atom learning with
  pose-toxicity penalties and exact component-trace feedback, not more direct
  PR77 replay variants.

## C091 patched-runtime control dispatch - 2026-05-03T12:04Z

The PR77 fixed-slice run used the patched runtime tree from the action-grammar
hardening change. C-091 was originally scored under the prior runtime tree.
The change is intended to be behavior-neutral for C-091, but exact custody says
runtime-tree changes must be measured rather than assumed.

Control dispatch:

- Candidate: byte-identical `C091` PR75 public replay archive
- Archive:
  `experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/archive.zip`
- Archive bytes: `276481`
- Archive SHA-256:
  `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746`
- Claim: `c091_current_runtime_control_t4`
- Job: `exact_eval_c091_current_runtime_control_t4_20260503T1204Z`
- State:
  `.omx/state/c091_current_runtime_control_t4_20260503T1204Z_batch_jobs.json`
- Purpose: patched-runtime anchor control; not an improvement lane.

Decision rule: if this differs from C-091 materially, all post-patch deltas
must compare against this control. If it matches within ordinary T4 scorer
noise, the original C-091 score remains usable as the public replay anchor, but
the patched control becomes the local apples-to-apples reference.

## C091 no-dispatch filters from worker swarm - 2026-05-03T12:13Z

Several aggressive local planners finished while the PR65 T4 screen and C091
patched-runtime control were running. They are preserved here as dispatch
filters, not score rows.

Mask/packer big move:

- Tool/artifacts:
  `experiments/plan_c091_mask_packer_bigmove.py`,
  `experiments/results/c091_mask_packer_bigmove_20260503_codex/`
- Exact-lossless mask Brotli repack found only `7` bytes of savings:
  `219465` bytes, SHA
  `1ddf03dad8466797397ead26a0d481766e22e26cbc936251c1e3ed5d2f6eda99`.
- Best ingested byte-screen row is
  `public_renderer_c089_p6_lossless_stream_resweep`, `276124` bytes, SHA
  `337b04040bee1316375bae8b2cfc2f08acddc235e4ee02d37ecfd62c0c831d95`,
  but it still needs `0.0009280386365951299` component-score improvement
  after rate to cross sub-`0.314`.
- Lossy AV1 mask transcode rows that save bytes flip classes, so they are
  planner signal only and not geometry-safe dispatch candidates.

C091-native action atoms:

- Tool/artifacts:
  `experiments/plan_c091_native_action_atoms.py`,
  `experiments/results/c091_native_action_atoms_20260503_codex/`
- The planner classified `257` atoms and found `63` component-positive
  pose-safe atoms, but the conservative component-improvement upper bound is
  only `0.00012712011530774194`.
- The best byte-screen break-even still needs
  `0.0010565494145477472` component-score improvement after rate.
- Exact eval is therefore not justified for current action-only policies.
  Future action work needs a different mechanism, not more direct PR77 replay
  or micro action-dictionary variants.

Public top-submission refresh:

- The official visible leaderboard still rounds PR67/PR65/PR63 at `0.32`.
  PR75/PR77 remain useful public-PR forensics but not visible leaderboard rows
  in the current snapshot.
- The top actionable faithful opportunities are now narrowed to:
  renderer self-compression with pose-safety, PR75 runtime-drift audit, and
  learned PR65/PR67 pose-manifold transfer. PR78/PR70-style payload relocation
  remains quarantined as non-faithful for our submission discipline.

Operational decision: do not queue standalone T4 exact evals for current
mask-packer, action-atom, public-renderer, PR77 replay, or qpost-copy variants.
Spend exact-eval capacity on PR65 pose transfer already running, C091 runtime
control, a renderer-shrink candidate only after pose-safety passes, or a new
candidate whose own planner clears a sub-`0.314` break-even.

## Lane 12 L2 unblock refresh - 2026-05-03T12:13Z

The Lane 12/NeRV retraining gate remains closed after a fresh local readiness
run:

- Command:
  `.venv/bin/python experiments/plan_lane12_l2_unblock.py --output-json experiments/results/lane12_l2_unblock_readiness_20260503_resume/lane12_l2_unblock_readiness.codex_refresh_20260503T121317Z.json`
- Output:
  `experiments/results/lane12_l2_unblock_readiness_20260503_resume/lane12_l2_unblock_readiness.codex_refresh_20260503T121317Z.json`
- `ready_for_retraining_unblock=false`
- `ready_for_exact_eval_dispatch=false`
- `clearance_state_written=false`

This preserves the prior gate: no Lane 12 retraining dispatch until the
clearance packet exists and passes the recorded evidence gates. Local
build-only or planner work remains allowed.

## Sub-0.314 exact-eval dispatch wave - 2026-05-03T12:21Z

Public floor snapshot:

- Official visible leaderboard at `https://comma.ai/leaderboard` still rounds
  PR67, PR65, and PR63 to `0.32`; PR67/PR65 remain the relevant faithful public
  floor references, while PR75/PR77 remain public-PR forensics rather than
  visible leaderboard rows in this snapshot.
- PR75 is a closed public PR for the same `qpose14_r55_segactions_minp`
  archive SHA as our C091 replay:
  `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746`,
  bytes `276481`. Its report lists PoseNet `0.00048657`, SegNet
  `0.00060529`, rounded score `0.31`; formula recomputation from the reported
  rounded components is `0.3143809189609485`. Our T4 replay of the same archive
  remains `0.31516575028285976`, so PR75 report deltas remain a public
  scorer/runtime-drift forensic target, not our current exact score claim.
- PR67 GitHub eval comment reports PoseNet `0.00049341`, SegNet `0.00061248`,
  archive `276564` bytes, rounded score `0.32`. Formula recomputation from
  those rounded components gives about `0.31564376464341976`.
- Our current exact C091 PR75/minp replay anchor remains lower than that
  public report: exact T4 score `0.31516575028285976`, bytes `276481`, SHA
  `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746`.
  The unchanged-component byte gap from C091 to `0.314` is about `1751`
  bytes.

Dispatched exact-eval candidates:

| job | lane | candidate | bytes | SHA-256 | status at dispatch | reason |
|---|---|---|---:|---|---|---|
| `exact_eval_pr65_pose_qp1_c091_c089_actions_p6_t4_fix1_20260503T121925Z` | `pr65_pose_qp1_c091_c089_actions_p6_t4_fix1` | `c091_pr65_pose_qp1_c089_actions_p6` | `276346` | `d3913ec75bd1917f16f2bca5e672313f66f81da5446405fc8eccbba757eed79d` | Pending, zero cost | Same bytes as the failed PR65 pose-QP1 stream screen, relaunched only after the raw4/raw5 action-loader ambiguity was fixed and locally tested. Needs `0.0010758593241882634` component gain for sub-`0.314`; no score claim until T4 harvest. |
| `exact_eval_c091_native_pose_manifold_top128_s025_t4_20260503T122013Z` | `c091_native_pose_manifold_top128_s025_t4` | `c091_native_cem_pose_waterfill_top128_s025` | `276489` | `1c9be2dd14b2607b9a86ecc071d6a37842896d18d62449885dac58d55afbbd64` | Pending, zero cost | Nietzsche worker's C091-native sparse QP1 pose/manifold move. It changes 128 pose pairs, is not a public/prior pose stream copy, passes local stream closure, and has proxy component gain `0.0014904868358843566` versus sub-`0.314` requirement `0.0011710771544847232`. Proxy evidence only until exact CUDA. |

Runtime-control state:

- `exact_eval_c091_current_runtime_control_t4_20260503T1204Z` completed and
  was harvested from state-derived Lightning artifacts.
- Result: exact T4 score `0.31516575028285976`, bytes `276481`, SHA
  `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746`,
  PoseNet `0.00049371`, SegNet `0.00060804`, promotion-eligible in
  adjudication.
- The patched runtime tree hash differs from the original C091 replay
  (`5a56ddfabb170fc85ff20634612e8ace3ec95295e489de527c07f101371e5233`
  versus
  `c816476e31c17ed237644d554801c29964d4f564243021706dbe775a319c3472`),
  but canonical exact score/components are unchanged. Use this as the
  post-patch runtime anchor for candidates queued after the action-grammar
  hardening.

GCP T4 hedge attempt:

- A bounded PR65 GCP T4 hedge was attempted after machine inventory showed a
  short `n1-standard-8` T4 wait.
- The first wrapper submit failed locally before job creation because the GCP
  machine requires an explicit `--cloud-account`.
- The lower-level explicit route also failed before job creation because the
  current Lightning Studio env cloud account does not match
  `gcp-lightning-public-prod`.
- No GPU spend and no score evidence resulted from these hedge attempts. The
  valid AWS/Studio T4 PR65 fix1 job remains Running/Pending as the custody path.

Subagents:

- `Leibniz` is hardening the raw action-payload ambiguity bug class as a
  reusable preflight/validator surface; no remote dispatch authority.
- `Kepler` is searching for one additional public-floor breakthrough candidate
  from PR67/PR65/PR63/PR73/PR77 anatomy; no remote dispatch authority.

## PR75 all-pair parity and no-holds-barred continuation - 2026-05-03T12:33Z

All-pair public PR75 parity completed locally as a custody/control artifact:

- Command:
  `.venv/bin/python experiments/pr75_raw_output_parity.py --public-archive experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/archive.zip --public-inflate-py /tmp/pr75-minp/submissions/qpose14_r55_segactions_minp/inflate.py --output-dir experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/raw_output_parity_all600_cpu_codex_20260503T1213Z --all-pairs --chunk-size 25 --device cpu --force`
- Artifact:
  `experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/raw_output_parity_all600_cpu_codex_20260503T1213Z/pr75_raw_output_parity.json`
- Evidence grade: `empirical_local_raw_parity`; no score claim and no remote
  GPU dispatch.
- Result: all `600` pairs match exactly between the public PR75 inflate path
  and `submissions/robust_current` under the current QP1-preserving runtime.
  The full rendered raw output after actions matches byte-for-byte:
  `3662409600` compared bytes, SHA
  `2cb335b615cb5437f005fa75166266d1e7fdf99853562466bb3a7ddee5996b0a`,
  zero changed bytes. Pair masks and QP1 float32 poses also match exactly.

Implication:

- The PR75 public-report versus current exact-T4 score discrepancy is no
  longer explained by local public-inflate versus robust-current output
  divergence. Treat it as scorer/runtime/report-custody forensics unless an
  exact archive eval proves otherwise.
- The active score path stays focused on C091-compatible byte or component
  movement: PR65 pose-QP1 stream transfer fix1 and the C091-native
  pose/manifold top128-s025 exact T4 run.

Current active T4 jobs after refresh:

| job | status | cost | action |
|---|---|---:|---|
| `exact_eval_pr65_pose_qp1_c091_c089_actions_p6_t4_fix1_20260503T121925Z` | Running | about `$0.027` | harvest immediately when JSON lands; close claim as A++ frontier/no-frontier/failure |
| `exact_eval_c091_native_pose_manifold_top128_s025_t4_20260503T122013Z` | Running | about `$0.026` | harvest immediately when JSON lands; close claim as A++ frontier/no-frontier/failure |

Additional no-remote subagent lanes spawned for wall-clock parallelism:

- `Confucius`: lossless/raw-parity-preserving byte squeeze around the C091/PR75
  anchor; no remote dispatch authority.
- `Plato`: C091-native pose/atom nextwave candidate generation; no remote
  dispatch authority.
- `James`: read-only public-floor and online research refresh; no remote
  dispatch authority.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_seg_tile_actions_preflight.py src/tac/tests/test_renderer_packed_payload.py src/tac/tests/test_lightning_batch_jobs.py src/tac/tests/test_build_pr75_lossless_repack_candidates.py src/tac/tests/test_plan_c091_pose_manifold_bigmove.py src/tac/tests/test_plan_public_floor_next_breakthrough_worker.py -q`
  passed: `145 passed in 1.66s`.

## T4 harvest: PR65 negative and pose-waterfill micro-frontier - 2026-05-03T12:40Z

Harvested two exact T4 jobs through `harvest-ssh --require-adjudication`:

| job | bytes | SHA-256 | PoseNet | SegNet | exact score | verdict |
|---|---:|---|---:|---:|---:|---|
| `exact_eval_pr65_pose_qp1_c091_c089_actions_p6_t4_fix1_20260503T121925Z` | `276346` | `d3913ec75bd1917f16f2bca5e672313f66f81da5446405fc8eccbba757eed79d` | `0.00054426` | `0.00061044` | `0.318825479152544` | A++ scoped negative; direct PR65 pose transfer is pose-toxic versus C091 |
| `exact_eval_c091_native_pose_manifold_top128_s025_t4_20260503T122013Z` | `276489` | `1c9be2dd14b2607b9a86ecc071d6a37842896d18d62449885dac58d55afbbd64` | `0.00049344` | `0.00060804` | `0.3151520345392486` | A++ micro-frontier; C091-native pose-waterfill moves score in the correct direction but only by `1.37e-05` |

Claim rows closed:

- `pr65_pose_qp1_c091_c089_actions_p6_t4_fix1` ->
  `completed_a_pp_no_frontier`
- `c091_native_pose_manifold_top128_s025_t4` ->
  `completed_a_pp_frontier_micro`

Operational consequence:

- PR65 direct pose stream transfer is retired only for this measured
  configuration; keep PR65 as a basis/qpost prior.
- C091-native pose-waterfill is now exact-evidenced but underpowered at the
  top128 setting. The already queued top192 setting remains justified as the
  aggressive next exact CUDA test because it is same-family, non-duplicate, and
  has larger proxy benefit.
- The active exact frontier is now C-101:
  `0.3151520345392486`, bytes `276489`, SHA
  `1c9be2dd14b2607b9a86ecc071d6a37842896d18d62449885dac58d55afbbd64`.

Additional dispatch:

- Queued `exact_eval_c091_next_cem_pose_waterfill_top192_s0125_m06_t4_20260503T1238Z`
  on Lightning T4 with manifest
  `.omx/state/c091_next_cem_pose_waterfill_top192_s0125_m06_t4_20260503T1238Z_manifest.json`.
- Candidate: bytes `276485`, SHA
  `79091f2c3f0c30ef3ca512808f3adc0306010e7f57fed3a09b3664c16fea4ea8`,
  `192` changed pose pairs, mask/renderer/actions preserved, proxy gain
  `0.001848983402485367`.
- Status at queue: Pending, zero cost. No score claim until harvest.

## Lossless byte-squeeze closure - 2026-05-03T12:48Z

Confucius completed the local-only lossless/raw-parity byte screen under
`experiments/results/lossless_byte_squeeze_worker_20260503/` with builder
`experiments/build_lossless_byte_squeeze_worker_20260503.py`.

Result:

- No dispatchable raw-parity/lossless sub-`0.314` byte candidate was found.
- Strict raw-parity stream changes only expose `8` inner bytes: mask `-7`,
  SG2 action wire `-1`, renderer/pose `0`.
- The parser-clean P3 form is `+2` bytes versus the anchor, and the headerless
  `-8` fixed-slice probe fails closed in `unpack_renderer_payload`, as it
  should.

Top artifacts:

| candidate | bytes | SHA-256 | status |
|---|---:|---|---|
| `anchor_stored_zip_rewrite_payload_identical` | `276481` | `2227c1cdc64defc88e88a31c2410a1ba88edac204274624d2aeae6ca9b4b1b3f` | payload-identical control; no byte win |
| `anchor_p3_best_lossless_streams_negative` | `276483` | `33d36fcc79dc3fbd567286f099b37e293bdab951f224114f64fdcdaa5b01e5db` | decoded parity passed, but byte-regressive |
| `anchor_raw_fixedslice_short_streams_parser_rejected` | `276473` | `7594f59fb70ec7100b4bb43b9b8e0f86165477d89ab06eda49695ca65fe5e399` | intentionally invalid; unpacker rejects |
| `anchor_p3_qzs3_block128_nonparity_probe` | `273695` | `409524e95afe1de5e34b2280a93fdde9fef34e4c9f707f0eeb714dd06bae8156` | parser/preflight passed but renderer bytes changed; not a raw-parity candidate |

Decision:

- Do not dispatch raw-parity byte squeeze; the byte ceiling is too small.
- Do not spend T4 on `anchor_p3_qzs3_block128_nonparity_probe` while top192
  pose-waterfill is running, because renderer bytes change and prior naive
  QZS3 reblocks failed pose-safety. This remains a possible diagnostic cliff
  probe only if renderer self-compression needs a fast exact negative.

Verification reported by worker:

- `py_compile` on `experiments/build_lossless_byte_squeeze_worker_20260503.py`
- Builder rerun with `--force`
- local extraction/unpack preflight
- `.venv/bin/python -m pytest src/tac/tests/test_unpack_renderer_payload_fixedslice.py src/tac/tests/test_profile_pr75_minp_archive.py -q`
  passed: `15 passed`.

## New 0.31 target and renderer-pose stack bridge - 2026-05-03T15:42Z

The active target is now strict `<= 0.31`, not merely sub-`0.314`. From the
current A++ parent, this requires a structural jump, not another pose
micro-sweep.

Fresh exact T4 harvests:

| job | bytes | SHA-256 | PoseNet | SegNet | exact score | verdict |
|---|---:|---|---:|---:|---:|---|
| `exact_eval_c091_next_cem_pose_waterfill_top192_s0125_m06_t4_20260503T1238Z` | `276485` | `79091f2c3f0c30ef3ca512808f3adc0306010e7f57fed3a09b3664c16fea4ea8` | `0.00049337` | `0.00060804` | `0.31514430182167497` | A++ current micro-frontier |
| `exact_eval_c091_next_cem_pose_waterfill_top160_s01875_m08_t4_fix1_20260503T1302Z` | `276488` | `26f0336ecee5f363e108f9f930def1382ea1b0734fa07d1f13822fc3d7d1721c` | `0.00049337` | `0.00060804` | `0.315146301821675` | A++ non-frontier sibling |

The top192 result improves C-101 by only about `7.7e-06`. This confirms that
C091-native pose-waterfill is directionally correct but far too shallow by
itself for `0.31`.

Naive public action/pose transplants are exact negatives already:

| candidate | bytes | SHA-256 | PoseNet | SegNet | exact score | verdict |
|---|---:|---|---:|---:|---:|---|
| `pr77_actions_pr75mask_renderer_c089pose_fixedslice` | `276329` | `27866172e76d27113e86a30f722588fd668f81a949be3acbe1e92cddc9a6a1d8` | `0.00054190` | `0.00060816` | `0.318426107391119` | A++ scoped negative; PR77 action/pose transplant is pose-toxic |
| `c067_pr75_actions_pose_safe_positive_ampminus1_p6` | `276317` | `6e6ec4609c6da581b4113c5c06e67970542a9d8e22c4866959fe618aaff2c796` | `0.00049595` | `0.00061150` | `0.31556196759570776` | A++ scoped negative; PR75 action subset worsens components despite byte win |

Therefore the immediate action/SegNet lane must be native atom synthesis or a
learned/reweighted selector against the current C102/top192 basin. Do not
duplicate direct PR77/PR75 action transplants.

Built deterministic renderer-pose stack bridge:

- Candidate: `c101_renderer_x_top192`
- Archive: `experiments/results/renderer_pose_stack_worker_20260503/c101_renderer_x_top192/archive.zip`
- Bytes: `275683`
- SHA-256: `d79d1556b55ba7e36c5aaf91d5b04320587975f1303698d8f1089bd5f399d0f3`
- Stream selection: top192 masks/actions/pose plus C101 renderer-shrink stream.
- Runtime unpack validation: passed; single-member `p`; no sidecars; decoded
  mask/actions/pose match top192 and decoded renderer matches the renderer
  self-compression source.
- Projected unchanged-component score versus top192: about `0.31461`, so this
  is a composability/bridge eval, not a `0.31` breakthrough by itself.

Queued exact T4 bridge eval:

- Job: `exact_eval_c101_renderer_x_top192_stack_t4_20260503T1540Z`
- State: `.omx/state/c101_renderer_x_top192_stack_t4_20260503T1540Z_batch_jobs.json`
- Claim: `c101_renderer_pose_stack_top192_t4`
- Status at submit: Pending, zero cost.

Structural lane status:

- Gauss prepared the fixed-renderer burn handoff at
  `experiments/results/c091_fixed_renderer_burn_sub031_readiness_20260503_worker/`.
- Break-even for strict `0.31` from the top192 parent is `7726` bytes saved at
  unchanged components, max archive bytes `268759`.
- Modal renderer exports are still running; transplant dispatch remains
  fail-closed until an export lands, a deterministic archive is built, exact
  source/candidate pose-safety passes, and a dispatch claim is recorded.

## P6 stack composability and C105 dispatch - 2026-05-03T15:55Z

The `<= 0.31` target remains active. The obsolete sub-`0.314` target must not
drive triage except as historical context.

Two scoped workers returned:

- Native C102/top192 action atoms built three deterministic P6 archives under
  `experiments/results/c101_native_action_atom_worker_20260503/`. The strongest
  byte candidate was
  `c101_top192_actions_consensus_positive_top64_ampfit_p6`, `276332` bytes,
  SHA `53601f0f4fba1b4b6a2ed8fe3489eca235c9c5ffaf7f69029751328b62e274f7`.
  Its own manifest estimates only about `6.2e-05` positive support, far below
  the roughly `0.00504` component improvement needed for `<=0.31`, so it is a
  stackable diagnostic rather than a primary breakthrough.
- Lane 12 readiness is now parser/primitive-contract clean but still blocked:
  no promotion-grade Alpha-Geo geometry JSON and no completed/custody-clean
  pose-regeneration provenance. The write attempt correctly refused to create
  `.omx/state/lane12_nerv_l2_clearance.json`.

Builder hardening:

- `experiments/build_c101_renderer_pose_stack_candidate.py` now accepts P6
  action/pose sources while preserving the source wire grammar and
  `action_record_count`. This fixed the P3-only stack-builder composability bug
  exposed when trying to combine C101 renderer bytes with Helmholtz's P6 native
  action archive.
- Focused verification passed:
  `.venv/bin/python -m py_compile experiments/build_c101_renderer_pose_stack_candidate.py`
  and
  `.venv/bin/python -m pytest src/tac/tests/test_build_c101_renderer_pose_stack_candidate.py -q`
  with `4 passed`.

Built and queued C105:

| candidate | bytes | SHA-256 | status |
|---|---:|---|---|
| `c101_renderer_top192_x_native_actions_top64` | `275530` | `fccd60ae2d5298c8c98e5d63e67a590847ccddad2768bae277570909c2b4ce43` | queued T4 exact screen |

The C105 stack uses C102/top192 mask and pose, C101 smaller renderer, and native
top64 P6 action atoms. Runtime unpack validation passed and decoded streams
match the selected sources. Formula-only unchanged-component projection is about
`0.31451`, still not enough for `<=0.31`; this is a composability/interaction
screen while the larger Modal renderer self-compression burns continue.

Dispatch/custody:

- Claim: `c101_renderer_native_action_top64_t4`
- Job: `exact_eval_c101_renderer_native_action_top64_t4_20260503T1605Z`
- State:
  `.omx/state/c101_renderer_native_action_top64_t4_20260503T1605Z_batch_jobs.json`
- Manifest:
  `.omx/state/c101_renderer_native_action_top64_t4_20260503T1605Z_manifest.json`
- Initial submit attempts failed before GPU spend on missing adjudication fields,
  missing explicit Lightning teamspace, missing T4 torch wheel pins, and
  unavailable `T4_SMALL`; the final queued job uses `g4dn.xlarge`,
  `INFLATE_TORCH_SPEC=torch==2.5.1+cu124`,
  `UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124`, and
  `UV_INDEX_STRATEGY=unsafe-best-match`.

## PR79 public-floor intake and exact replay queue - 2026-05-03T16:16Z

Active target remains strict `<=0.31`; obsolete `0.314` thresholds must not
drive dispatch decisions.

Latest public intake:

- Official comma leaderboard still shows the visible merged top rounded at
  `0.32`.
- Public PR79 `qpose14_r55_segactions_minp v2` is open and reports a CUDA
  600-sample score body with:
  - archive bytes `277388`
  - archive SHA-256
    `01dc02badf851d99108fd92c570271f36f74cc5424c6d2a8f1b499cb4d1c3446`
  - PoseNet `0.00048721`
  - SegNet `0.00059224`
  - recomputed formula score `0.31372571308675656`
- Treat PR79 as a public PR-body target until exact replay or leaderboard
  acceptance; C-102 remains our exact T4 score truth.

Parser/runtime hardening:

- Downloaded PR79 archive to
  `experiments/results/top_submission_reverse_engineering_20260503_pr79/archive.zip`
  and verified the public SHA above.
- Added the measured PR79 fixed-slice grammar to the forensic profiler and
  robust unpacker:
  - payload bytes `277288`
  - `masks.mkv.br` `219472` bytes
  - `renderer.bin.br` `55756` bytes
  - `seg_tile_actions.br` `1162` bytes
  - `optimized_poses.qp1.br` `898` bytes
- PR79 action stream profile:
  - SG2 grouped tile-frame-delta-varint
  - `672` action records
  - `295` unique pairs
  - `45` unique tiles
  - decoded runtime record bytes `2688`
- Selected-pair raw-output parity against the public PR79 inflator is exact for
  pairs `0,12,33,54,104,598`, including action-bearing pairs. This is local
  runtime evidence only, but it clears the semantic parser/runtime gate for an
  exact T4 replay.
- A local CPU all-600 parity pass was started with `--fast-fail` and showed no
  mismatch before interruption, but it was intentionally stopped after several
  minutes because exact CUDA replay was already running and the local proof was
  lower wall-clock EV than keeping the workstation responsive for harvest and
  PR79 action-subset building. This is not a failure of PR79 parity; the
  completed selected-pair parity remains the recorded local gate.

Verification:

- `.venv/bin/python -m py_compile experiments/profile_pr75_minp_archive.py submissions/robust_current/unpack_renderer_payload.py experiments/pr75_raw_output_parity.py`
- `.venv/bin/python -m pytest src/tac/tests/test_profile_pr75_minp_archive.py src/tac/tests/test_unpack_renderer_payload_fixedslice.py -q`
  passed with `17 passed`.

Queued exact T4 replay:

- Lane claim: `pr79_public_replay_t4`
- Job: `exact_eval_pr79_minp_v2_public_replay_t4_20260503T1615Z`
- Machine: `g4dn.xlarge` / Lightning `T4_SMALL`
- State:
  `.omx/state/pr79_minp_v2_public_replay_t4_20260503T1615Z_batch_jobs.json`
- Source manifest:
  `.omx/state/pr79_minp_v2_public_replay_t4_20260503T1615Z_manifest.json`
- Initial status: `Pending`, zero cost.
- A attempted GCP T4 hedge on `n1-standard-8` failed before job creation and
  before spend because the active Lightning Studio cloud account did not match
  `gcp-lightning-public-prod`. Terminal claim:
  `failed_pre_submit_cloud_account_mismatch`.

PR79-vs-C-102 action anatomy:

- C-102/top192 action stream: `108` SG2 records, `255` charged bytes,
  `106` unique pairs, `21` unique tiles.
- PR79 action stream: `672` SG2 records, `1162` charged bytes, `295` unique
  pairs, `45` unique tiles.
- Record overlap: `105` exact records shared, `3` C-102-only records, `567`
  PR79-only records. This makes PR79 an expanded action-waterfill basin, not a
  replacement renderer/mask basin.
- A read/write worker is building deterministic PR79 action-subset candidates
  locally under `experiments/results/pr79_action_subset_worker_20260503/`; no
  remote dispatch is authorized from that planner output alone.

Decision rule after harvest:

- If PR79 exact replay reproduces near its body score, immediately diff PR79 vs
  C-102 by stream and component trace and use PR79's 672-action policy as the
  next action-selection basin.
- If PR79 exact replay regresses, preserve it as exact public-body drift
  evidence and still mine its action stream only through native C-102 atom
  builders, not direct transplant claims.

### 2026-05-03 PR79 exact T4 replay landed

The PR79 public replay has been harvested through the state-derived Lightning
SSH artifact path and is valid A++ contest-T4 evidence:

- Job: `exact_eval_pr79_minp_v2_public_replay_t4_20260503T1615Z`
- Artifact dir:
  `experiments/results/lightning_batch/exact_eval_pr79_minp_v2_public_replay_t4_20260503T1615Z/`
- Archive bytes: `277388`
- Archive SHA-256:
  `01dc02badf851d99108fd92c570271f36f74cc5424c6d2a8f1b499cb4d1c3446`
- Score: `0.31457805357318636`
- PoseNet distance: `0.00049415`
- SegNet distance: `0.00059581`
- Hardware: Tesla T4, CUDA, `600` samples
- Promotion eligible: `true`
- Delta vs C-102: `-0.00056624824848861`

This becomes the exact internal frontier, but it does **not** reproduce the PR
body's formula-only `0.31372571308675656`. Exact CUDA replay is the score
authority. Remaining pure-byte gap from this frontier to `0.31` is about
`6875.41` charged bytes at unchanged distortion; a safer `0.3095` buffer needs
about `7626.32` bytes. PR79's expanded SG2 action basin remains useful profile
feedback, but the next dispatch must either remove roughly seven kilobytes
without changing output or produce a measured SegNet/PoseNet improvement with
charged bytes.

Dispatch claim closure:

- Lane: `pr79_public_replay_t4`
- Status row: `completed_score=0.31457805357318636`

Immediate follow-on work:

- Run PR73-style flatpack transfer only if the emitted runtime is byte-closed
  and local selected-output parity passes.
- Continue Modal fixed-renderer burns and apply the renderer-transplant
  pose-safety gate before any exact eval.
- Mine PR79's `567` extra action records as score-profile atoms, not as a
  direct transplant claim.

### 2026-05-03 PR79-native H100 action-search dispatch

Rationale:

- PR79 exact T4 replay is the new internal frontier, but its public-body score
  did not reproduce. The useful delta is PR79's action-search basin: `672` SG2
  records and materially lower SegNet than C-102, with a small PoseNet/rate
  tax.
- Wrapper/flatpack forensics found no large lossless byte lever in this
  archive family. Beating `0.31` therefore requires score-affecting search,
  not just ZIP or fixed-slice cleanup.

Implementation:

- Added `scripts/remote_lane_pr79_segaction_search.sh`.
- It clones PR79's public branch
  `dev/qpose14-r55-segactions-minp-v2`, verifies the published archive SHA,
  runs `probe_more_seg_actions_minp.py` and `optimize_action_subset.py` on
  CUDA, writes charged candidate archives plus manifests, and marks all output
  as proxy/search-only until robust parity and exact T4 auth eval.

Harness bug and permanent fix:

- First Modal H100 call `fc-01KQQBM0FQ949YVSV1T4E0VNF4` failed in `2s` before
  method work because the Modal training image lacked Python `brotli`, which
  PR79's public scripts import at startup.
- Fixed the bug class by adding `brotli` to `experiments/modal_train_lane.py`
  image dependencies and adding a defensive dependency bootstrap in
  `scripts/remote_lane_pr79_segaction_search.sh`.
- Focused checks passed:
  - `bash -n scripts/remote_lane_pr79_segaction_search.sh`
  - `py_compile experiments/modal_train_lane.py`
  - `17 passed` for Modal-focused tests
  - `25 passed` for PR79/public parser/forensics/action-subset tests
  - `git diff --check` over the touched surfaces

Active relaunch:

- Lane: `pr79_segaction_search_h100`
- Modal call id: `fc-01KQQC1Q7VMM9XKNR84JQT660T`
- Label: `pr79_segaction_search_h100_fix2_20260503T1653Z`
- Status: running after two pre-search harness failures were classified and
  fixed
- Recover command:
  `.venv/bin/python experiments/modal_recover_lane.py --call-id fc-01KQQC1Q7VMM9XKNR84JQT660T`

Second harness bug:

- Fix1 call `fc-01KQQBXQYJTW7HHJ30FNHTG9CH` reached CUDA setup but failed
  before search because the shallow public clone contained invalid/LFS-pointer
  model assets; `safetensors` raised `header too large`.
- The lane now replaces the cloned public `models/`, `videos/`, and public-test
  name files with local upstream assets before running PR79's public scripts.

No score claim exists from this lane. If it produces a promising archive, the
next gates are robust-current local parity, lane claim, and exact T4 CUDA auth
eval on the identical bytes.

### 2026-05-03 PR79 action-search fix4 greenup

Fix2 and fix3 both failed before method work and were classified as harness
failures, not PR79-search negatives:

- Fix2 call `fc-01KQQC1Q7VMM9XKNR84JQT660T` failed inside Modal DALI/NVDEC with
  `nvml internal driver error`. The lane was patched to avoid Modal NVDEC by
  using PyAV decode.
- Fix3 call `fc-01KQQC50S3BR3EFM55NX8KPK9C` then failed because PR79's public
  `AVVideoDataset` asserts against a CUDA device. The lane now patches both
  public search scripts at the dataset construction site so `AVVideoDataset`
  is instantiated with `torch.device("cpu")` while SegNet inference and search
  tensors remain on CUDA.

Permanent guard/fix surface:

- `scripts/remote_lane_pr79_segaction_search.sh` now uses local upstream
  assets, CPU/PyAV decode, CUDA inference/search, and defensive dependency
  bootstrap.
- `experiments/modal_train_lane.py` carries `brotli` in the Modal image
  dependency set.
- Dispatch claims have terminal rows for the three pre-search failures, so
  future agents do not mistake them for active spend or scientific negatives.

Verification:

- `bash -n scripts/remote_lane_pr79_segaction_search.sh`
- `17 passed` for Modal-focused tests
- `27 passed` for PR79 flatpack/public parser/action-subset tests

Active call:

- Lane: `pr79_segaction_search_h100`
- Modal call id: `fc-01KQQCBTFYJMA24Q37V7K8F1FF`
- Label: `pr79_segaction_search_h100_fix4_20260503T1700Z`
- Status: running as proxy/search-only work; no score claim.
- Recover command:
  `.venv/bin/python experiments/modal_recover_lane.py --call-id fc-01KQQCBTFYJMA24Q37V7K8F1FF`

Flatpack transfer worker result:

- PR73-style `Brotli(RPK1)` flatpack transfer is byte-regressive for all
  runtime-closed PR79/C102 variants screened so far and should not consume T4
  exact-eval capacity without a component-improvement rationale.
- The PR79-aware micro-packer found
  `public_renderer_c089_p6_lossless_stream_resweep` at `276124` bytes
  (`-218` versus C089), but it changes `renderer.bin`; it must pass
  `experiments/preflight_renderer_transplant_pose_safety.py` against exact
  source/candidate SHAs before any exact-eval dispatch.

### 2026-05-03 PR79-wide action-search hedge

Because the default PR79 action search may converge inside the same local
basin, a second no-expense H100 search was dispatched with a deliberately wider
action budget:

- Lane: `pr79_segaction_search_h100_wide`
- Modal call id: `fc-01KQQCJB3YAXX7A9VYJ2DA2WJX`
- Label: `pr79_segaction_search_h100_wide_20260503T1705Z`
- Parameters:
  - `PR79_ACTION_TOP_TILES=8`
  - `PR79_ACTION_PASSES=5`
  - `PR79_ACTION_MAX_ACTIONS=3200`
  - `PR79_ACTION_PROBE_MIN_GAIN=0.0000008`
  - `PR79_ACTION_SUBSET_PASSES=5`
  - `PR79_ACTION_SUBSET_MIN_GAIN=0.00000005`
- Status: running as proxy/search-only work; no score claim.

One local launch command failed before dispatch because `--env-overrides` takes
a comma-separated string, not trailing positional `KEY=value` arguments. That
was an operator/CLI invocation bug with zero remote spend; the actual dispatch
uses the real comma-separated argparse contract.

### 2026-05-03 latest public-state refresh

Live refresh:

- `https://comma.ai/leaderboard` still rounds the visible video-compression
  top entry to `0.32` for PR67/PR65/PR63. The displayed table is rounded and
  not sufficient to rank exact sub-0.32 variants.
- GitHub PR API shows the newest relevant open submissions are PR79
  `qpose14_r55_segactions_minp v2 (0.31)` and PR77
  `qzs3_tile_delta_r147 (0.31)`.
- PR79 remains the exact internal frontier after T4 replay:
  `0.31457805357318636`, bytes `277388`, SHA
  `01dc02badf851d99108fd92c570271f36f74cc5424c6d2a8f1b499cb4d1c3446`.
- PR77's current attachment is the same `276551` byte archive already replayed
  exactly on T4 as `0.31537874750377204`, SHA
  `f90880383c95e14d82704f99db9b20944786ae6452a844348638b06c439972af`.
  It is useful as a tile-delta design prior but not the current frontier.
- PR78 and PR70-style payload relocation remain non-contest-faithful for our
  custody standard; they are not score lanes.

Immediate public-intel conclusion: run PR79's own public action-search basin
hard, then compress/grammar-pack the action stream to fund more actions. The
remaining gap to `0.31` is too large for ZIP overhead or PR73 flatpack alone.

### 2026-05-03 PR79 action-search harvest triage tool

Added `experiments/triage_pr79_action_search_results.py` so a recovered Modal
action-search result can be turned immediately into an auditable archive matrix
without manual byte/SHA inspection. The tool records:

- archive bytes and SHA-256 for `probe_archive.zip` and `archive_optimized.zip`
- byte/rate delta versus the exact PR79 T4 frontier
- projected score if components were unchanged
- remaining component-score improvement required to beat the frontier and
  target `0.31`
- no-op/source-preserving status when available from manifests
- whether the archive may proceed to the parity gate before a T4 lane claim

It deliberately emits `score_claim=false` and `promotion_eligible=false`; exact
CUDA auth eval remains the only score source. This closes the wall-clock gap
between a Modal H100 completion and a T4 dispatch decision.

Verification:

- `py_compile experiments/triage_pr79_action_search_results.py`
- `bash -n scripts/remote_lane_pr79_segaction_search.sh`
- `23 passed` for triage, Modal, and PR79 action-subset focused tests
- `git diff --check` over touched surfaces

### 2026-05-03 PR79 SegNet-pool action-search hedge

The PR79 public probe supports optional `--pose-gate` and `--pose-check`.
Default and wide H100 calls keep those gates enabled. To escape that local
minimum, `scripts/remote_lane_pr79_segaction_search.sh` now accepts
`PR79_ACTION_POSE_FLAGS`, defaulting to `--pose-gate --pose-check`.

An aggressive third H100 search was dispatched with empty
`PR79_ACTION_POSE_FLAGS`, so the probe can discover a broader SegNet action
pool. The downstream `optimize_action_subset.py` stage still evaluates net
score before writing `archive_optimized.zip`, and all artifacts remain
proxy-only until exact CUDA auth eval.

- Lane: `pr79_segaction_search_h100_segpool`
- Modal call id: `fc-01KQQD2TPVPWRJTBQ78R1N84J4`
- Label: `pr79_segaction_search_h100_segpool_20260503T1712Z`
- Parameters:
  - `PR79_ACTION_TOP_TILES=12`
  - `PR79_ACTION_PASSES=4`
  - `PR79_ACTION_MAX_ACTIONS=4500`
  - `PR79_ACTION_PROBE_MIN_GAIN=0.0000005`
  - `PR79_ACTION_SUBSET_PASSES=5`
  - `PR79_ACTION_SUBSET_MIN_GAIN=0.00000005`
  - `PR79_ACTION_POSE_FLAGS=`
- Status: running as proxy/search-only work; no score claim.

Additional focused verification after adding configurable pose flags:

- `bash -n scripts/remote_lane_pr79_segaction_search.sh`
- `9 passed` for Modal and triage focused tests
- `git diff --check` over touched surfaces

Public PR refresh:

- PR69 `houdini` release asset is an empty `22` byte ZIP. It is not a usable
  contest artifact.
- PR76 `qpose14_poseq6` reports `0.34` and `288567` bytes, so it is not a
  near-term frontier replay.
- PR74 `ph4ntom_drv` reports `0.35` and `321311` bytes. Its FiLM/QAT/writeup
  is useful architecture evidence for longer renderer work, but not an exact
  replay candidate for beating `0.31`.

### 2026-05-03 continuation: PR79 exact frontier and no-holds action search

Live public refresh at `2026-05-03T17:18Z`:

- Official visible comma leaderboard still lists the rounded faithful frontier
  as `0.32` for PR67/PR65/PR63. Newer PR75/PR77/PR79 rounded `0.31` entries
  are active GitHub floor signals but not visible leaderboard rows yet.
- Latest GitHub PR API ordering has PR77 open
  `qzs3_tile_delta_r147 (0.31)`, PR79 open
  `qpose14_r55_segactions_minp v2 (0.31)`, and PR75 closed
  `qpose14_r55_segactions_minp (0.31)`.
- Current internal exact custody truth remains PR79 T4 replay:
  score `0.31457805357318636`, bytes `277388`, archive SHA
  `01dc02badf851d99108fd92c570271f36f74cc5424c6d2a8f1b499cb4d1c3446`.
  The public PR79 body score is treated as a target/report, not a claim,
  because exact T4 replay is the authority.

Subagent returns:

- Hume/Franklin/Mencius/Halley completed and were closed after harvest.
- Hume confirmed PR79/PR75/PR77 share mask, renderer, and pose streams; only
  action bytes differ. PR79 has `1162` charged action bytes and `672` decoded
  SG2 records.
- Franklin confirmed ZIP overhead is already the strict single-member minimum;
  byte-only path to `<=0.31` is not present without changing stream semantics.
- Mencius ranked remaining top levers as PR79-native action water-fill,
  trained/pose-safe renderer shrink, and bounded pose/action interactions.
- Halley implemented `pr79_s1_fixed_lossless_actions`, a runtime-closed PR79
  action-stream repack preserving decoded action SHA while saving `41` archive
  bytes.

Queued exact eval:

- Lane claim: `pr79_s1_fixed_lossless_actions_t4`
- Job: `exact_eval_pr79_s1_fixed_lossless_actions_t4_20260503T171722Z`
- Archive:
  `experiments/results/pr79_action_lossless_repack_20260503_codex/pr79_s1_fixed_lossless_actions/archive.zip`
- Bytes/SHA: `277347`,
  `d61527a43218b87871fd869dcb92b6875e99482bc28a8fdaf879caf6d8cfc4eb`
- Staging: manifest verified `1434` files, `25203673` bytes, remote manifest
  OK.
- Status at first refresh: Lightning `Pending`, cost `0.0`.
- This is a micro byte-custody anchor, not the main `<=0.31` path.

Active PR79 H100 searches:

- Default/fix4: `fc-01KQQCBTFYJMA24Q37V7K8F1FF`
- Wide: `fc-01KQQCJB3YAXX7A9VYJ2DA2WJX`
- SegPool/no probe pose gate: `fc-01KQQD2TPVPWRJTBQ78R1N84J4`
- Ultra/no probe pose gate: `fc-01KQQDNZN67ZPPW379NZHRHWTJ`

The ultra lane is intentionally risky local-minimum escape:
`TOP_TILES=20`, `PASSES=8`, `MAX_ACTIONS=12000`,
`SUBSET_PASSES=8`, `PR79_ACTION_POSE_FLAGS=`. It is proxy/search-only; any
candidate must pass triage, parity, lane claim, and exact T4 auth eval before
score discussion.

### 2026-05-03 continuation: S2 action codec and renderer-shrink negative

PR79 action codec v2:

- Candidate: `pr79_s2_fixed_adaptive_actions`
- Archive:
  `experiments/results/pr79_action_dictionary_repack_v2_20260503_codex/pr79_s2_fixed_adaptive_actions/archive.zip`
- Bytes/SHA: `277321`,
  `5740aca7e255b00093154eb1823b5b6207d8795f8eb287d35758c4cda438ec68`
- Decoded action SHA:
  `a48bd4e49f8928158756610fd8094e8fb1611a2040121611055266f840faf13f`
- Delta: `-67` bytes versus PR79, `-26` bytes versus S1.
- Codec: S2 split metadata/delta Brotli plus adaptive arithmetic-coded action
  IDs.
- Exact eval job queued:
  `exact_eval_pr79_s2_fixed_adaptive_actions_t4_20260503T173023Z`.
- Staging: manifest verified `1440` files, `25326649` bytes, remote manifest
  OK.
- Status at first refresh: Lightning `Pending`, cost `0.0`.

The S2 job is a deliberate parallel hedge against S1 because it is strictly
smaller and runtime-closed; it remains a micro byte-custody lane, not the main
`<=0.31` route.

Renderer-shrink local findings:

- PR79 coarse QZS3 reblocks can save up to `1345` bytes but fail local
  `render_output_parity_unsafe`, so they are not exact-eval candidates.
- A parity-constrained tensor-local search screened 10 transforms before
  archive build and emitted no archive candidates. The closest transform
  changed renderer bytes but still failed strict render parity
  (`mean_abs=0.039225`, `rms=0.091138`, `max_abs=1.633507`).
- Dispatch recommendation is `do_not_dispatch`; the next renderer path must be
  learned/parity-constrained, not another coarse reblock.

Planning-only interaction findings:

- PR79/C102 pose-action interaction analyzer ranked pair/frame/action/pose
  atoms and emitted machine-readable policy inputs, but the best policy proxy
  benefit was only about `0.00011974`, far below the `<=0.31` break-even.
- This is a stop signal for tiny pose/action atom dispatch and reinforces the
  H100 PR79 action-search lanes as the current highest-EV route.

### 2026-05-03 continuation: PR79 parity closure and S1 exact micro-frontier

Latest public/leaderboard refresh:

- Official visible comma leaderboard still lists the rounded video-compression
  frontier as `0.32` for PR67/PR65/PR63. Newer GitHub PR77 and PR79 are open
  public `0.31` floor signals but not visible official leaderboard rows at the
  time of refresh.
- GitHub PR79 reports `277388` bytes, SHA
  `01dc02badf851d99108fd92c570271f36f74cc5424c6d2a8f1b499cb4d1c3446`,
  PoseNet `0.00048721`, SegNet `0.00059224`, rounded score `0.31`. Our exact
  T4 replay remains the authority for internal claims.
- GitHub PR77 reports `276551` bytes, PoseNet `0.00049314`, SegNet
  `0.00060631`, rounded score `0.31`. Exact T4 replay scored
  `0.31537874750377204`, bytes `276551`, SHA
  `f90880383c95e14d82704f99db9b20944786ae6452a844348638b06c439972af`,
  PoseNet `0.00049588`, SegNet `0.00060816`; useful as action-basin signal,
  not a frontier.

PR79 runtime parity control:

- Artifact:
  `experiments/results/top_submission_reverse_engineering_20260503_pr79/raw_output_parity_all600_cpu_codex_20260503T1728Z/pr75_raw_output_parity.json`
- Evidence grade: `empirical_local_raw_parity`.
- It checked all `600` pairs in `30` chunks over `945.96s`.
- Public PR79 inflate and `robust_current` raw outputs match exactly for every
  chunk after actions; native tensors after actions also match exactly. This
  closes the local raw-output parity confound for current-runtime PR79 replay.
  Exact CUDA auth eval remains the score authority.

PR79 S1 exact result:

- Job: `exact_eval_pr79_s1_fixed_lossless_actions_t4_20260503T171722Z`
- Archive:
  `experiments/results/pr79_action_lossless_repack_20260503_codex/pr79_s1_fixed_lossless_actions/archive.zip`
- Bytes/SHA: `277347`,
  `d61527a43218b87871fd869dcb92b6875e99482bc28a8fdaf879caf6d8cfc4eb`
- Exact Tesla T4 score: `0.3145508035731863`
- Components: PoseNet `0.00049415`, SegNet `0.00059581`, `600` samples.
- Status: A++ contest-T4, promotion eligible. It is a pure byte improvement
  over PR79 exact replay by `41` bytes and `0.00002725` score.

PR79 S2 exact status:

- Job: `exact_eval_pr79_s2_fixed_adaptive_actions_t4_20260503T173023Z`
- Archive bytes/SHA: `277321`,
  `5740aca7e255b00093154eb1823b5b6207d8795f8eb287d35758c4cda438ec68`
- Exact Tesla T4 score: `0.31453355357318635`
- Components: PoseNet `0.00049415`, SegNet `0.00059581`, `600` samples.
- Status: A++ contest-T4, promotion eligible. Lightning status regressed to
  `Pending` after JSON landed, so the state-derived harvest copied
  `contest_auth_eval.json` and the deterministic local adjudicator produced
  `contest_auth_eval.adjudicated.json` plus `adjudication_provenance.json`.
- S2 is the current exact internal frontier. It saves `67` archive bytes versus
  PR79 exact replay and `26` bytes versus S1. At unchanged components the
  remaining pure-byte gap to `0.31` is still about `6808` bytes, so this is a
  custody polish lane; the main breakthrough path remains PR79-native H100
  action search or a larger representation/renderer change.

Renderer transplant high-byte check:

- Candidate:
  `experiments/results/renderer_transplant_qfaithful2146_on_c089_20260503_codex/external_qzs3_b1024_pr75_preserved_slices/archive.zip`
- Bytes/SHA: `270372`,
  `57e2640ace74af9dd0b60024e8fdcf906c113958806fcd555f43b2045791e507`
- Byte delta versus source C089-style archive: `-5970`, close to the pure-byte
  gap.
- Required pose-safety preflight:
  `experiments/results/renderer_transplant_qfaithful2146_on_c089_20260503_codex/external_qzs3_b1024_pr75_preserved_slices/pose_safety_preflight_codex.json`
- Result: `safe_for_exact_eval_dispatch=false`, failure class
  `renderer_transplant_pose_safety_failed`, mean abs `72.1234`, RMS `87.2603`,
  max abs `254.8358`.
- Decision: do not dispatch exact eval for this transplant. The byte win is
  real, but the renderer geometry is catastrophically different under the
  mandated local gate.

### 2026-05-03 continuation: public target refresh, tile16 H100 hedge, and triage hardening

Public target refresh:

- Official comma leaderboard still visibly lists the video-compression frontier
  rounded at `0.32` for PR67/PR65/PR63, with deadline
  `2026-05-04T11:59:00Z`.
- GitHub PR API now reports PR79 closed at `2026-05-03T17:48:28Z`, while PR77
  remains open. PR79's public body remains a rounded `0.31` target, but our
  internal score claim is only the exact T4 S2 artifact above.
- Read-only public-intel subagent Copernicus returned the same high-EV
  ordering: PR79-native action search/action codec, pose-safe renderer shrink,
  PR65 pose active-subspace mining, PR65 postprocess atoms, and PR73/PR74
  architecture transfer only as implementation fuel. PR70/PR78 payload
  relocation remains a non-faithful guard/no-go source only.

Additional H100 search:

- Claimed and dispatched `pr79_segaction_search_h100_tile16` on Modal H100.
- Concrete call id: `fc-01KQQFP05K18159RMFCKGTPZ08`.
- Label: `pr79_segaction_search_h100_tile16_20260503T175501Z`.
- Parameters: `PR79_ACTION_TILE=16`, `TOP_TILES=16`, `PASSES=5`,
  `MAX_ACTIONS=16000`, `SUBSET_PASSES=8`, no probe-time pose flags,
  `BATCH_SIZE=20`.
- Rationale: this is not a duplicate of the active 32x32 PR79 action searches.
  It probes finer SegNet correction atoms and accepts higher action-stream
  cost if component gain is large enough. It is proxy/search-only; no score
  claim can be made until a recovered archive passes triage/parity and exact
  T4 CUDA auth eval.
- Modal recovery currently reports all five PR79 action-search calls still
  running:
  `fc-01KQQCBTFYJMA24Q37V7K8F1FF`,
  `fc-01KQQCJB3YAXX7A9VYJ2DA2WJX`,
  `fc-01KQQD2TPVPWRJTBQ78R1N84J4`,
  `fc-01KQQDNZN67ZPPW379NZHRHWTJ`,
  `fc-01KQQFP05K18159RMFCKGTPZ08`.

Triage hardening:

- Patched `experiments/triage_pr79_action_search_results.py` so its default
  frontier is the current PR79 S2 exact T4 frontier, not the stale PR79 public
  replay:
  score `0.31453355357318635`, bytes `277321`, SHA
  `5740aca7e255b00093154eb1823b5b6207d8795f8eb287d35758c4cda438ec68`.
- Added a focused regression test in
  `src/tac/tests/test_triage_pr79_action_search_results.py`.
- Verification: `10 passed` for the triage/Modal focused tests, Python compile
  passed for the touched triage/recovery tools, and `git diff --check` passed
  for the touched files.

Active local workers:

- Maxwell: local-only PR65/Henosis pose advantage isolation against PR79/S2.
- Singer: local-only PR79/S2 mask stream and full-archive byte reduction.
- Galileo: local-only PR65/Henosis postprocess atom mining and
  PR79-compatible candidate design.

### 2026-05-03 continuation: tile8 H100 hedge and local atom negatives

Additional no-holds-barred H100 search:

- Claimed and dispatched `pr79_segaction_search_h100_tile8` on Modal H100.
- Concrete call id: `fc-01KQQG4ESDVSE4SW1QGBFYR450`.
- Label: `pr79_segaction_search_h100_tile8_20260503T180253Z`.
- Parameters: `PR79_ACTION_TILE=8`, `TOP_TILES=12`, `PASSES=3`,
  `MAX_ACTIONS=20000`, `PROBE_MIN_GAIN=0.00000025`,
  `SUBSET_PASSES=8`, `SUBSET_MIN_GAIN=0.00000002`, no probe-time pose flags,
  `BATCH_SIZE=16`.
- Rationale: this is the finest PR79-native action grid dispatched so far. It
  tests whether smaller tile atoms can recover enough SegNet benefit density
  to offset a larger action stream. This is search/proxy evidence only until
  a recovered archive passes triage/parity and exact T4 CUDA auth eval.
- Modal recovery currently reports all six PR79 action-search calls still
  running:
  `fc-01KQQCBTFYJMA24Q37V7K8F1FF`,
  `fc-01KQQCJB3YAXX7A9VYJ2DA2WJX`,
  `fc-01KQQD2TPVPWRJTBQ78R1N84J4`,
  `fc-01KQQDNZN67ZPPW379NZHRHWTJ`,
  `fc-01KQQFP05K18159RMFCKGTPZ08`,
  `fc-01KQQG4ESDVSE4SW1QGBFYR450`.

Local postprocess atom result:

- Galileo built three PR79/S2 archives with charged PR65 postprocess atoms:
  `pr79_s2_pr65_qpost_bias_pair598` (`277474` bytes,
  SHA `81fff847a0507642b2f7f24b3555ba769bf73dae05fc42d8591416f3f1fe945e`),
  `pr79_s2_pr65_qpost_bias_region_pair598` (`277498` bytes,
  SHA `bf1c4bec6837447528c5b9639ae25fcbd1edf5d3e18b84421f62606ccb9d02be`),
  and `pr79_s2_pr65_qpost_post_pair104` (`277475` bytes,
  SHA `92a8c418045466a21b11afc138b3dfe4a0b9024ce3f425e9c981b379b4333d78`).
- Artifact summary:
  `experiments/results/pr79_pr65_postprocess_atoms_20260503_worker/candidate_summary.json`.
- Evidence grade: `empirical_archive_closed_raw_delta_proof_only`.
- Decision: `do_not_dispatch`. The artifacts are not no-ops, but the family has
  seven exact T4 negative qpost screens and these candidates require about
  `0.00127` component gain just to clear `0.314`. That is not a good T4 spend
  while PR79-native action searches are live.
- Verification: `10 passed` for the PR79/PR65 qpost focused test set; Python
  compile passed for the worker builder.

Local mask-body byte result:

- Singer's first available control artifact,
  `experiments/results/pr79_mask_body_reduction_20260503_worker/p3_lossless_rebrotli_mask_control/manifest.json`,
  is a lossless mask-body rebrotli control at `277398` bytes, SHA
  `652f11c8c59df3e89c816c6730bbd81e9cefc60d195cc2800212b07c4d6f03ca`.
- The decoded mask SHA matches PR79, and non-mask runtime members are preserved,
  but the archive is `+77` bytes versus current S2 and fails the byte/preflight
  recommendation gate. Decision: no exact eval dispatch.

Current coordination:

- Spawned Curie for current source-cited public-floor reverse engineering
  focused on PR79/PR77 and author artifacts.
- Spawned Leibniz for PR77 tile-delta/action-transfer local candidate building.
- These workers must not dispatch remote jobs; any future exact eval candidate
  still requires a fresh lane claim and exact T4 CUDA auth eval.

### 2026-05-03 continuation: high-EV exact screens and TG1 tile-size hardening

Public frontier reconciliation:

- Curie refreshed the public target table. The official comma leaderboard page
  still visibly lists rounded `0.32`, but PR79 and PR77 are live/open PR-level
  rounded `0.31` targets.
- PR79 public body reports `277388` bytes, PoseNet `0.00048721`, SegNet
  `0.00059224`, recomputed `0.3137257131`. Our local exact T4 replay/S2
  frontier is `0.31453355357318635`, `277321` bytes, PoseNet `0.00049415`,
  SegNet `0.00059581`.
- This confirms a live runtime/component-parity reconciliation gap of about
  `0.00080784` versus PR79 public body. Do not treat rounded titles as exact
  score truth; use them as target signals and keep exact CUDA custody local.

TG1 fine-grid action hardening:

- Found a real bug class in the aggressive tile16/tile8 PR79 searches:
  the public search scripts accept `--tile`, but the robust runtime applied
  `seg_tile_actions` with a fixed `SEG_TILE_SIZE=32`.
- Added backward-compatible charged `TG1` tile-size headers for non-32 action
  streams, with runtime/unpacker/preflight support. Existing 32x32 streams
  remain unchanged.
- Also fixed SG2 fine-grid decode to preserve 5-byte record semantics when
  tile ids exceed 255, including the case where decoded length is divisible by
  both 4 and 5.
- Patched `scripts/remote_lane_pr79_segaction_search.sh` so future non-32
  search outputs are wrapped with charged `TG1` metadata. Already-running
  tile16/tile8 calls were launched before this wrapper patch, so their
  recovered artifacts must be rewrapped/parity-checked before exact eval.
- Verification: Python compile passed for the touched runtime/preflight files,
  `bash -n` passed for the remote lane script, and `14 passed` for the focused
  tile-action/preflight/PR77/mask-body tests.

High-EV mask-body exact screen:

- Singer produced
  `protected_c067_micro_mask_reencode_threshold_ladder_20260503_save05k_crf53`,
  archive `260866` bytes, SHA
  `5e49a972ab5d77d8a8eeedc9dd36a309bf8533ad2d2e2b9b96ad30448691d338`.
- It preserves renderer/actions/pose and passes archive/parser validation; the
  byte delta versus PR79/S2 is large enough to cross `0.31` if component
  damage is controlled.
- Modal T4 exact attempts failed twice after strict inflate succeeded because
  upstream DALI evaluation hit NVML internal driver error before
  `contest_auth_eval.json`. Failure class: `failed_infra_nvml_dali_no_score`;
  no method/score conclusion.
- Rerouted through Lightning T4:
  `exact_eval_pr79_mask_body_crf53_t4_20260503T1828Z`, currently `Pending`
  at zero cost after manifest verification.
- Because the first Lightning T4 job remained pending at zero cost, queued one
  wall-clock hedge on `g4dn.2xlarge`:
  `exact_eval_pr79_mask_body_crf53_t4_2x_20260503T1824Z`, same archive bytes
  and SHA. This is a duplicate confirmation path, not a second scientific
  hypothesis; stop/ignore the duplicate once the first exact JSON lands.

PR77 action-transfer exact screen:

- Leibniz produced
  `replace_pr79_with_pr77_pair_overlap_s2_p3_on_pr79`, archive `276474` bytes,
  SHA `9140ea901f85dde755cb448109c9225700822515fccb1eb40f57230face04835`.
- The archive changes only `seg_tile_actions.bin`; unchanged-component score
  estimate is `0.31396957103989187`, with `0.00056398` component-regression
  tolerance versus PR79/S2.
- Modal T4 exact screen hit the same DALI/NVML failure after strict inflate.
  Rerouted through Lightning T4:
  `exact_eval_pr77_pair_overlap_on_pr79_s2_t4_20260503T1828Z`, currently
  `Pending` at zero cost after manifest verification.

Active queue summary:

- Lightning exact screens pending: mask-body CRF53 on T4 small, mask-body
  CRF53 T4 2x hedge, and PR77 pair-overlap action transfer.
- Modal H100 action searches still running:
  `fc-01KQQCBTFYJMA24Q37V7K8F1FF`,
  `fc-01KQQCJB3YAXX7A9VYJ2DA2WJX`,
  `fc-01KQQD2TPVPWRJTBQ78R1N84J4`,
  `fc-01KQQDNZN67ZPPW379NZHRHWTJ`,
  `fc-01KQQFP05K18159RMFCKGTPZ08`,
  `fc-01KQQG4ESDVSE4SW1QGBFYR450`.

### 2026-05-03 continuation: PR79 trust-region bracket after CRF53 cliff

Leaderboard and public-intel refresh:

- Live comma leaderboard lists PR79 `qpose14_r55_segactions_minp` as the
  rounded `0.31` public target. The PR body reports `277388` bytes, SHA
  `01dc02badf851d99108fd92c570271f36f74cc5424c6d2a8f1b499cb4d1c3446`,
  PoseNet `0.00048721`, SegNet `0.00059224`, and rounded final score `0.31`.
- Read-only public review re-confirmed that PR79's actionable signal is
  metric-pruned SegNet tile actions plus compact single-payload packing. PR77
  remains useful as tile/action prior data, not as a full action transplant;
  PR78-style payload relocation remains non-faithful validator-hardening only.

Exact results harvested:

- `exact_eval_pr79_mask_body_crf53_t4_20260503T1828Z`: A-negative T4 exact,
  archive `260866` bytes, SHA
  `5e49a972ab5d77d8a8eeedc9dd36a309bf8533ad2d2e2b9b96ad30448691d338`,
  score `1.5192235429996075`, PoseNet `0.11985313`, SegNet `0.00250749`.
  The T4 2x hedge reproduced the same score/component collapse. Scope:
  measured CRF53 mask-body implementation/config only; this is a sharp
  geometry cliff, not a broad mask-body family kill.
- `exact_eval_pr77_pair_overlap_on_pr79_s2_t4_20260503T1828Z`: A++ T4 exact,
  archive `276474` bytes, SHA
  `9140ea901f85dde755cb448109c9225700822515fccb1eb40f57230face04835`,
  score `0.31532892827214076`, PoseNet `0.00049476`, SegNet `0.00060897`.
  Promotion-eligible but worse than the PR79/S2 exact frontier; use as PR77
  action-prior evidence, not a frontier candidate.

Next-byte/trust-region generator hardening:

- Extended `experiments/build_pr79_next_byte_candidates.py` to screen multiple
  trusted mask-body rows instead of only the matrix recommendation. Generated
  deterministic CRF53/CRF52/CRF51/CRF50 candidates with single stored `p`,
  action-loader guards, action semantics parity, source mask-body readiness,
  and deterministic rebuild evidence.
- Focused verification: `py_compile` passed and
  `src/tac/tests/test_build_pr79_next_byte_candidates.py` reports `3 passed`.

New exact screens queued:

- CRF52 raw trust-region bracket:
  `exact_eval_pr79_mask_body_crf52_t4_20260503T1835Z`, archive `273141`
  bytes, SHA `bc11b5e06f66571f9476dab3ad965f28cb42c63612ce2e0feb42d91c0e1aad27`.
  Status at record time: Running.
- CRF52 PR79-action-preserving flatpack:
  `exact_eval_pr79_nextbyte_crf52_pr79action_flatpack_t4_20260503T1848Z`,
  archive `259288` bytes, SHA
  `6aa256d0f855f950cb03bc25358ce23929eecb2f60ce33eafffad87581d1907a`.
  It changes only `masks.mkv` versus PR79 decoded semantics, preserves
  renderer/pose/actions, and has unchanged-component estimate `0.3025261191`.
  Status at record time: Pending.
- CRF51 PR79-action-preserving flatpack:
  `exact_eval_pr79_nextbyte_crf51_pr79action_flatpack_t4_20260503T1852Z`,
  archive `270941` bytes, SHA
  `45d821206fef0f173036d959c78a0db620ef92e975ee6941430404daf8d60c71`.
  It is a safer cliff-mapping bracket with unchanged-component estimate
  `0.3102853735`. Status at record time: Pending.

Decision rule:

- Do not dispatch additional CRF53-derived mask-body candidates without a new
  repair/geometry mechanism; exact evidence shows catastrophic PoseNet
  collapse.
- If CRF52 raw and CRF52 flatpack are component-safe, CRF52 flatpack becomes
  the immediate sub-0.31 submission candidate path. If CRF52 collapses and
  CRF51 survives, use CRF51 as the new exact anchor and push action-atom/pose
  line-search deltas for the remaining gap.

CRF52 raw bracket harvest:

- `exact_eval_pr79_mask_body_crf52_t4_20260503T1835Z`: A-negative T4 exact,
  archive `273141` bytes, SHA
  `bc11b5e06f66571f9476dab3ad965f28cb42c63612ce2e0feb42d91c0e1aad27`,
  score `1.4869182415517563`, PoseNet `0.11406384`, SegNet `0.00237038`,
  `600` samples on Tesla T4. This saves only `4180` bytes versus the
  PR79/S2 frontier but catastrophically violates the PoseNet gate. Scope:
  measured raw CRF52 mask-body implementation/config only. The evidence now
  says the un-repaired raw mask-body CRF ladder has no safe shoulder between
  CRF52 and CRF53; preserve remaining spend for PR79-action-preserving,
  repaired, or learned/atomized variants.

Updated active exact screens:

- CRF52 PR79-action-preserving flatpack
  `exact_eval_pr79_nextbyte_crf52_pr79action_flatpack_t4_20260503T1848Z`
  is Running at the latest refresh. This remains the highest-EV live packet
  because it preserves PR79 renderer/pose/actions while replacing only the
  charged mask body.
- CRF51 PR79-action-preserving flatpack
  `exact_eval_pr79_nextbyte_crf51_pr79action_flatpack_t4_20260503T1852Z`
  has advanced to Running. This is the safer byte bracket if CRF52 still
  crosses the geometry cliff.

CRF52 action-preserving flatpack harvest:

- `exact_eval_pr79_nextbyte_crf52_pr79action_flatpack_t4_20260503T1848Z`:
  A-negative T4 exact, archive `259288` bytes, SHA
  `6aa256d0f855f950cb03bc25358ce23929eecb2f60ce33eafffad87581d1907a`,
  score `1.4685522121816597`, PoseNet `0.11215021`, SegNet `0.00236893`,
  `600` samples on Tesla T4. Preserving PR79 actions and pose stream did not
  rescue the CRF52 mask-body shrink; PoseNet still collapses while SegNet
  stays within the forensic gate. This narrows the cliff to the mask body
  representation itself, not the action sidecar. Do not dispatch `save05k`
  CRF52 variants without a repair/geometry mechanism.

Current branch after CRF52 negatives:

- Wait for `exact_eval_pr79_nextbyte_crf51_pr79action_flatpack_t4_20260503T1852Z`.
  If CRF51 is safe, it becomes the exact anchor for the smaller `save05k_crf51`
  near-neighbor and PR79 action/pose micro-stack. If CRF51 also collapses, stop
  un-repaired CRF mask-body shrink and move spend to PR79 action search,
  QP1/QPV1 pose atoms, and explicit repair/multimask mechanisms.

CRF51 action-preserving flatpack harvest:

- `exact_eval_pr79_nextbyte_crf51_pr79action_flatpack_t4_20260503T1852Z`:
  A-negative T4 exact, archive `270941` bytes, SHA
  `45d821206fef0f173036d959c78a0db620ef92e975ee6941430404daf8d60c71`,
  score `1.4240150820059325`, PoseNet `0.10408684`, SegNet `0.00223377`,
  `600` samples on Tesla T4. This confirms the measured un-repaired PR79
  mask-body CRF shrink branch has no viable safe shoulder at CRF51/52/53 under
  the current half-frame runtime contract.

Measured retirement:

- Retire only the measured implementation family:
  un-repaired PR79 mask-body CRF shrink candidates that preserve PR79
  renderer/pose/actions but replace the mask body. Do not dispatch
  `save05k_crf51`, CRF50, or CRF52 variants from this family without an
  explicit geometry repair, multimask reconciliation layer, learned mask-body
  replacement, or exact preflight proving the PoseNet cliff has been removed.
- Keep the evidence as training signal: the mask-byte savings are real, but
  PoseNet collapses while SegNet remains within the forensic gate. The next
  archive builder should treat this as a high-density repair/multimask target,
  not as a broad proof against mask compression.

New infrastructure landed by parallel workers:

- PR79 next-byte candidate builder now supports deterministic CRF/action-source
  filters, action-parity dispatch gating, body-byte profiles, and stricter
  provenance. Focused tests: `4 passed`.
- QP19/QPV1 parser/planner support landed for future PR77-style pose/container
  candidates. Current public PR77 artifact still parses as fixed-slice QP1, so
  QPV1 support is infrastructure only until a concrete QP19/QPV1 archive is
  generated and exact-evaluated.

Next high-EV branch:

- Main path: harvest running Modal PR79 action searches and exact-evaluate only
  candidates that preserve action semantics and have byte/action closure.
- Orthogonal path: QP1/QPV1 active-subspace pose atoms and PR67-style line
  search on the PR79/S2 frontier or any action-improved anchor.
- Repair path: build explicit mask repair/multimask candidates around the
  CRF51/52 component traces instead of more raw CRF shrink.

## 2026-05-03T19:20Z - PR77 overlap action exact-eval wave and dispatch guard

Live public target refresh:

- Official `https://comma.ai/leaderboard` still lists PR79
  `qpose14_r55_segactions_minp` at rounded `0.31` on 2026-05-03.
- Internal exact frontier remains PR79/S2, A++ T4 score
  `0.31453355357318635`, archive bytes `277321`, SHA-256
  `5740aca7e255b00093154eb1823b5b6207d8795f8eb287d35758c4cda438ec68`.

Exact-eval dispatches queued:

| job | lane | candidate | bytes | SHA-256 | status | reason |
|---|---|---|---:|---|---|---|
| `exact_eval_pr77_tile_overlap_on_pr79_s2_t4_20260503T1918Z` | `pr77_tile_overlap_on_pr79_s2_t4` | `replace_pr79_with_pr77_tile_overlap_s2_p3_on_pr79` | `276526` | `1043991267ff9ce33df05cdfb7ff865debb907c5f0b4b3a23f57aab553f9806c` | Pending, zero cost | Distinct PR77 action-subset transfer; if components are unchanged, rate-only score is about `0.31400419570545424`, beating PR79/S2 by `0.0005293578677321262`. |
| `exact_eval_pr77_all_overlap_on_pr79_s2_t4_20260503T1918Z` | `pr77_all_overlap_on_pr79_s2_t4` | `replace_pr79_with_pr77_all_s2_p3_on_pr79` | `276535` | `0300f8baa75b7cb864a8c91dc77fd057a38bde2ecd4d736bff3d92c6fcde78f1` | Pending, zero cost | Adjacent PR77 overlap action-subset hypothesis; if components are unchanged, rate-only score is about `0.31401018843603234`, beating PR79/S2 by `0.0005233651371540267`. |

Both jobs were staged through `scripts/lightning_exact_eval_repro.py` with
manifest verification, local and remote supply-chain scans, active dispatch
claims, expected archive SHA/bytes, PR79/S2 component reference, component
trace, and adjudication. No score claim until exact T4 JSON is harvested.

Subagent hardening integrated:

- `experiments/build_pr79_next_byte_candidates.py` now refuses to mark CRF51/50
  next-byte candidates as exact-screen dispatchable unless their source
  mask-body row is itself exact-eval-ready after lane claim.
- Focused test: `src/tac/tests/test_build_pr79_next_byte_candidates.py`.
- Verification from worker: `.venv/bin/python -m pytest
  src/tac/tests/test_build_pr79_next_byte_candidates.py -q` -> `4 passed`.
- This preserves local CRF byte screens but prevents known PoseNet-cliff
  families from leaking into T4 spend without a repaired source contract.

Active nonlocal search:

- Modal H100 PR79 action searches still running at this refresh:
  `fc-01KQQCJB3YAXX7A9VYJ2DA2WJX`,
  `fc-01KQQD2TPVPWRJTBQ78R1N84J4`,
  `fc-01KQQDNZN67ZPPW379NZHRHWTJ`,
  `fc-01KQQFP05K18159RMFCKGTPZ08`,
  `fc-01KQQG4ESDVSE4SW1QGBFYR450`.

Next exact loop:

1. Refresh and harvest the two PR77-overlap T4 jobs first.
2. If either lands below PR79/S2, update the claim matrix and use its component
   trace as the next action-atom selector.
3. If both regress, preserve them as scoped PR77 action-transfer negatives and
   wait for Modal H100 action-search candidates before spending more T4.
4. Continue excluding unrepaired CRF mask-body shrink candidates from dispatch.

## 2026-05-03T19:34Z - Public PR82/PR81 sub-0.30 intake and replay dispatch

Leaderboard and public-target refresh:

- Official `https://comma.ai/leaderboard` still publishes PR79
  `qpose14_r55_segactions_minp` at rounded `0.31`.
- Two late public PRs now claim lower scores but are not yet official
  leaderboard entries at this refresh:
  - PR82 `henosis_frontier`, claimed score `0.298321723132154`.
  - PR81 `qzs3_range_mask`, claimed score `0.2878245063016197`.
- These are external/self-reported until exact CUDA replay of the exact public
  archive bytes lands.

Static custody intake:

| PR | local intake | archive bytes | archive SHA-256 | public source commit | member anatomy |
|---|---|---:|---|---|---|
| PR82 | `experiments/results/public_pr82_henosis_frontier_intake_20260503_codex` | `296789` | `a0e07c360223c1dd3d3b92263225d38d542e218e83d095ad9b91bf872f94c6e4` | `4f460810faac2bf8f774cb9d5ee9d1041389be20` | one stored member `x`, member SHA `1c3377d09ddfddea488be791875605d1877003da4b0bd4d5c752e82d45fd8077` |
| PR81 | `experiments/results/public_pr81_qzs3_range_mask_intake_20260503_codex` | `215960` | `cd01378a52688fe00ee1bfb898c67695aed89a7b3ded602b597eb7fc3031d7fc` | `854b397e9fca912fb864614387912f25805fd669` | one stored member `p`, magic `QMA9`, member SHA `c59524610474c89e5a41433f47d2bb881f878e694f853db1377272699f9eb3e9` |

Replay adapters:

- Public source branches were cloned under each intake directory.
- `replay_submission/` directories preserve public `inflate.py` and decoder
  source bytes, plus an inert empty `config.env` required by the hardened
  Lightning exact-eval runtime-closure guard.
- PR82 `inflate.py` replay SHA matches raw public source:
  `0809f704f1623d0e694621129abced77eaac052ae35469e207ef06938e4be1ad`.
- PR81 `range_mask_codec.cpp` replay SHA matches raw public source:
  `26b44194c9b364ae3e4b7a30832204c008324df91bc12281d30cee30c43b4676`.

Exact replay dispatches:

| job | lane | status at dispatch | purpose |
|---|---|---|---|
| `exact_eval_public_pr82_henosis_frontier_t4_20260503T1932Z` | `public_pr82_henosis_frontier_t4_replay` | Pending | T4 replay of public PR82 archive/source, adjudicated against PR79/S2. |
| `exact_eval_public_pr81_qzs3_range_mask_t4_20260503T1932Z` | `public_pr81_qzs3_range_mask_t4_replay` | Pending | T4 replay of public PR81 archive/source, adjudicated against PR79/S2. |

Both jobs use the shared staged manifest
`.omx/state/public_pr81_pr82_replay_t4_20260503T1930Z_manifest.json`, expected
archive SHA/bytes, active dispatch claims, CUDA auth eval, component traces,
and adjudication.

Harness bug class fixed during dispatch:

- First PR81 submit was blocked before GPU because the exact-eval submitter
  always ran the `seg_tile_actions` packed-payload validator and misclassified
  public `QMA9` as a robust-current `RPK1` payload.
- Fixed `scripts/launch_lightning_batch_job.py` so the `seg_tile_actions`
  archive preflight remains mandatory for `submissions/robust_current/inflate.sh`
  but is skipped for explicit external/public inflate runtimes.
- Added regression coverage in
  `src/tac/tests/test_seg_tile_actions_preflight.py` for public/external
  runtime replay.
- Verification:
  `.venv/bin/python -m pytest src/tac/tests/test_seg_tile_actions_preflight.py -q`
  -> `7 passed`.

Transferable design signal from PR82 static intake:

- Single stored ZIP member with a compact internal length table.
- Joint frame-pair generator: one mask stream, pose-conditioned first-frame
  head, static second-frame head.
- Custom `QH0` FP4-ish weight packing with high/low byte splitting and
  per-block scales.
- Tiny pose stream plus charged postprocess fields: stage choices, shift,
  fractional shifts, RGB bias, region bias, and sparse/random multi-pattern
  overlays.
- This is a direct candidate architecture for our next sub-0.31 implementation
  work if exact replay validates the public score basin.

Transferable design signal from PR81 static intake:

- The dominant byte lever is a deterministic semantic range-mask stream, not
  AV1/CRF grayscale. Static decode produced `117964800` raw mask bytes from
  `159011` charged bytes, with the public `QMA9` header recording 600 frames
  at stored `512x384`.
- Renderer weights are split/reordered by entropy family before Brotli:
  compressed chunks `37086`, `3035`, and `15604` bytes reassemble to QZS3-like
  FP4 generator weights.
- Pose is an `899` byte Brotli `QP1` stream with only one useful pose channel
  populated; channels 1-5 are zero-filled.
- A `225` byte 3-bit router action stream selects 600 per-pair postprocess
  choices.
- The immediate implementation implication, pending exact replay, is to treat
  semantic mask entropy coding as a typed archive contract and calibrate it
  with exact component response, rather than continuing unrepaired CRF mask
  shrink. This directly answers the grayscale-collapse lesson: preserve the
  renderer's mask manifold and entropy-code the labels/control stream instead
  of introducing geometric CRF artifacts.

### 2026-05-03T19:43Z - PR77 overlap transfer exact negatives and PR81/QMA9 planning contract

PR77 overlap transfer exact-eval harvest:

| candidate | evidence | score | delta vs PR79/S2 | bytes | SHA-256 | conclusion |
|---|---|---:|---:|---:|---|---|
| `replace_pr79_with_pr77_tile_s2_p3_on_pr79` | A++ T4 exact CUDA | `0.31533632816891066` | `+0.0008027745957243093` | `276526` | `1043991267ff9ce33df05cdfb7ff865debb907c5f0b4b3a23f57aab553f9806c` | measured implementation negative |
| `replace_pr79_with_pr77_all_s2_p3_on_pr79` | A++ T4 exact CUDA | `0.315367997503772` | `+0.0008344439305856266` | `276535` | `0300f8baa75b7cb864a8c91dc77fd057a38bde2ecd4d736bff3d92c6fcde78f1` | measured implementation negative |

Both archives preserve component gates and save about `786-795` bytes relative
to PR79/S2, but the SegNet/PoseNet regression costs more than the rate gain.
This retires only the measured direct PR77-overlap transplant configuration.
It does not retire PR77 tile-delta ideas, overlap actions, or semantic-mask
entropy coding as families.

Dispatch-claim hygiene:

- `pr77_tile_overlap_on_pr79_s2_t4` closed with
  `completed_exact_negative`.
- `pr77_all_overlap_on_pr79_s2_t4` closed with
  `completed_exact_negative`.

PR81/QMA9 planning contract:

- Added planning-only profiler output:
  `experiments/results/public_pr81_qzs3_range_mask_intake_20260503_codex/pr81_qma9_semantic_range_mask_profile.json`.
- Evidence grade: `external/planning_only`; `score_claim=false`;
  `dispatch_performed=false`; `gpu_required=false`.
- Archive custody in profile: `215960` bytes,
  SHA-256 `cd01378a52688fe00ee1bfb898c67695aed89a7b3ded602b597eb7fc3031d7fc`.
- QMA9 mask stream: `600 x 512 x 384`, `158991` bitstream bytes,
  `159011` packed mask bytes, `117964800` decoded mask bytes.
- Optional local C++ decode-hash validation succeeded: decoded mask SHA-256
  `c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a`.
- Focused verification: `.venv/bin/python -m pytest
  src/tac/tests/test_profile_pr81_qma9_range_mask_contract.py` -> `4 passed`.

Current public replay state after refresh:

- `exact_eval_public_pr82_henosis_frontier_t4_20260503T1932Z`: Running on T4.
- `exact_eval_public_pr81_qzs3_range_mask_t4_20260503T1932Z`: Running on T4.

Next action remains harvest-first: exact public replay results decide whether
PR82 postprocess controls, PR81 semantic range coding, or their composition
becomes the immediate implementation target.

### 2026-05-03T20:02Z - Public PR81/PR82 replay dependency fix and resubmission

The first public replay wave reached T4 preflight but failed before scoring:

- `exact_eval_public_pr81_qzs3_range_mask_t4_20260503T1932Z`
- `exact_eval_public_pr82_henosis_frontier_t4_20260503T1932Z`

Failure class: `harness/runtime_dependency_preflight`. Both jobs had CUDA/T4
runner preflight evidence, but public inflate runtime needed optional Brotli
support that was not installed in the exact-eval environment. This is not
method evidence and produced no score claim.

Permanent hardening landed in the Lightning exact-eval command builder:

- exact-eval now accepts declared optional public-inflate runtime packages from
  `INFLATE_BROTLI_SPEC` and `INFLATE_AV_SPEC`.
- the job emits `lightning_inflate_runtime_bootstrap.json` before scoring.
- `PATH` is updated so public `inflate.sh` calls to `python` resolve through
  the exact-eval venv.
- focused verification passed:
  `.venv/bin/python -m pytest
  src/tac/tests/test_lightning_batch_jobs.py::test_exact_cuda_eval_command_installs_declared_external_inflate_deps
  src/tac/tests/test_lightning_batch_jobs.py::test_exact_cuda_eval_command_is_json_and_cuda_only
  src/tac/tests/test_seg_tile_actions_preflight.py -q`
  -> `9 passed`.

Restaged source and public replay closure through manifest custody:

- manifest:
  `.omx/state/public_pr81_pr82_replay_depsfix_t4_20260503T195657Z_manifest.json`
- remote verification: `1468` files, `26098729` bytes,
  manifest SHA-256
  `392ac38fc44b0463029a7845203aa9d94931d456b2d74803c64682a57cbda022`.
- the manifest explicitly includes both public `replay_submission/` directories,
  not only the archives.

Resubmitted exact T4 replays:

| job | archive bytes | archive SHA-256 | declared runtime deps | state |
|---|---:|---|---|---|
| `exact_eval_public_pr81_qzs3_range_mask_t4_depsfix_20260503T195657Z` | `215960` | `cd01378a52688fe00ee1bfb898c67695aed89a7b3ded602b597eb7fc3031d7fc` | `brotli==1.2.0` | Pending at zero cost after first refresh |
| `exact_eval_public_pr82_henosis_frontier_t4_depsfix_20260503T195657Z` | `296789` | `a0e07c360223c1dd3d3b92263225d38d542e218e83d095ad9b91bf872f94c6e4` | `brotli==1.2.0`, `av==17.0.1` | Pending at zero cost after first refresh |

Dispatch claims are active under:

- `public_pr81_qzs3_range_mask_t4_replay_depsfix`
- `public_pr82_henosis_frontier_t4_replay_depsfix`

Current exact frontier remains PR79/S2 until these replays produce adjudicated
CUDA JSON: score `0.31453355357318635`, bytes `277321`, archive SHA-256
`5740aca7e255b00093154eb1823b5b6207d8795f8eb287d35758c4cda438ec68`.

### 2026-05-03T20:22Z - Public replay fast diagnostics and Lightning namespace guard

Current public replay queue state:

- `exact_eval_public_pr81_qzs3_range_mask_t4_depsfix_20260503T195657Z`:
  Running on T4.
- `exact_eval_public_pr82_henosis_frontier_t4_depsfix_20260503T195657Z`:
  SDK telemetry regressed `Running -> Pending`, so local status is
  `REMOTE_STATUS_RECONCILIATION_REQUIRED` until a terminal status or harvested
  artifact supersedes it.
- `exact_eval_public_pr81_qzs3_range_mask_l40s_fastdiag_20260503T200750Z`:
  Running on L40S as diagnostic-only evidence.
- `exact_eval_public_pr82_henosis_frontier_l40s_fastdiag_20260503T200750Z`:
  Running on L40S as diagnostic-only evidence.

H100 diagnostic hedge attempt:

- Lightning inventory showed Nebius H100 availability, but Studio-backed submit
  failed before job creation because the selected Studio namespace belongs to a
  different cloud account than `lightning-nebius-prod`.
- Failure class: `failed_predispatch_cloud_account_mismatch`; no GPU job was
  created and no H100 spend occurred.
- Durable guardrail added to `AGENTS.md`: Studio-backed Lightning Batch Jobs
  can only run on the cloud account attached to that Studio namespace; inventory
  on another account is not dispatch authority.

Subagent signal now feeding the next local lanes:

- QMA9/PR81: full-stream model accounting is tight
  (`1271926.81158` modeled bits vs `1271928` bitstream bits); up-gate cost
  dominates. Naive extra fallback gates screen negative. The next local
  implementation target is vertical-copy/block/run escape or context-backoff
  redesign with raw-mask parity before any exact eval.
- PR82/Henosis: randmulti has 72 replay groups; 35 fit current `QPS1/NM2`, 37
  need PR82-native semantics or a narrowed exact-parity subset. Existing local
  NM2 screens remain dispatch-closed.
- Observability: current signal table has 177 planning rows from 12 sources and
  surfaces `pair_0230` from a PR79 CRF51 trace as a high break-even component
  atom. This is allocator input only, not score evidence.

### 2026-05-03T20:27Z - Public PR81/PR82 replay harvest

The dependency-fixed public replay wave harvested cleanly.

Promotion-grade T4 results:

| archive | hardware | score | bytes | SHA-256 | PoseNet | SegNet | status |
|---|---|---:|---:|---|---:|---:|---|
| PR81 QMA9/range-mask | Tesla T4 | `0.2812078926981712` | `215960` | `cd01378a52688fe00ee1bfb898c67695aed89a7b3ded602b597eb7fc3031d7fc` | `0.000584` | `0.00060989` | `A++`, promotion eligible, new exact frontier |
| PR82 Henosis | Tesla T4 | `0.2983246102939779` | `296789` | `a0e07c360223c1dd3d3b92263225d38d542e218e83d095ad9b91bf872f94c6e4` | `0.0001894` | `0.00057185` | `A++`, promotion eligible, non-frontier vs PR81 |

Diagnostic L40S replays:

| archive | hardware | score | status |
|---|---|---:|---|
| PR81 QMA9/range-mask | L40S | `0.2813662442992855` | diagnostic only, non-promotion |
| PR82 Henosis | L40S | `0.29956215478043985` | diagnostic only, non-promotion |

Claim rows were closed with terminal statuses in
`.omx/state/active_lane_dispatch_claims.md`.

Current exact frontier is now PR81/QMA9 at `0.2812078926981712`. This
supersedes PR79/S2 by `-0.03332566087501515` and crosses both the sub-0.31 and
sub-0.30 targets under exact Tesla T4 evidence. The next score route is
PR81-native mask entropy/runtime improvement and PR82-control transfer on top
of the PR81 basin.

Immediate stack math:

- PR81 rate term at `215960` bytes is `0.14379889951626412`.
- PR82 exact T4 components contribute SegNet `0.05718500000000001` and PoseNet
  `0.043520110293977884`.
- A hypothetical contest-legal stack with PR81 archive bytes and PR82 component
  distances would score `0.244504009810242`.
- This is planning math only, but it defines the next high-EV implementation
  target: preserve PR81/QMA9 mask compactness while importing the PR82/Henosis
  model/pose/control quality through charged, deterministic runtime contracts.

### 2026-05-03T20:50Z - PR81+PR82 runtime-bridge greenup

No remote dispatch was performed in this slice.

Implemented robust-runtime support for the PR81 public single-member payload
shape and refreshed the fail-closed PR81+PR82 stack manifests.

Runtime bridge work:

- `submissions/robust_current/unpack_renderer_payload.py` now parses PR81
  `p` payloads into typed charged members: `masks.qma9`, PR81 reordered QZS3
  `renderer.bin`, raw `optimized_poses.qp1`, and `router_actions.3bit`.
- `submissions/robust_current/inflate_renderer.py` now supports QMA9 masks,
  PR81 reordered QZS3 model restoration, and PR81 3-bit router actions in the
  normal JointFrameGenerator inflate loop.
- `submissions/robust_current/range_mask_codec.cpp` was promoted into the
  robust runtime so QMA9 decode uses the fast C++ path when available; the
  pure-Python QMA9 decoder remains a correctness fallback.
- Worker `Averroes` added QRM1 runtime decode support in
  `submissions/robust_current/apply_qzs3_postprocess.py`. Generic and
  raw-frame-only PR82 specials are supported; mask-dependent PR82 specials
  still fail closed.

Refreshed stack candidates:

| candidate | bytes | planning score if PR82 components carry | dispatch state |
|---|---:|---:|---|
| `pr81_qma9_pr82_qps1_controls_all600` | `218621` | `0.24627586048450012` | blocked on local raw-output parity/delta proof |
| `pr81_qma9_pr82_qps1_nm2_generic_randmulti` | `223392` | `0.249452673549846` | blocked on local raw-output parity/delta proof |
| `pr81_qma9_pr82_qps1_qrm1_all072_randmulti` | `232580` | `0.2555705856111325` | blocked on raw-output parity plus mask-dependent QRM1 specials |
| `pr81_qma9_pr82_qps1_controls_qrm1_all072` | `235111` | `0.2572558746214847` | blocked on raw-output parity plus mask-dependent QRM1 specials |

Verification:

- `.venv/bin/python -m py_compile` on touched runtime/test files passed.
- C++ compile smoke for `submissions/robust_current/range_mask_codec.cpp`
  passed and prints the expected usage string.
- Focused pytest passed: `45 passed in 3.03s`.
- Local unpack smoke on
  `pr81_qma9_pr82_qps1_controls_all600/archive.zip` emitted
  `masks.qma9`, `renderer.bin` (`Q81R`), `optimized_poses.qp1` (`QP1`), and
  `router_actions.3bit`, while preserving `qpost.bin`.
- `git diff --check` passed on the touched surfaces.

Next score-critical unblock: attach local raw-output parity/delta proof for
the smallest runtime-compatible candidate, then claim and dispatch T4 exact
eval only if the proof shows a real charged output change and no unsupported
runtime branch.
