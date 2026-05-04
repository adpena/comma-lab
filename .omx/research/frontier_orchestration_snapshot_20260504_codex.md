# Frontier Orchestration Snapshot - 2026-05-04

This is a control-plane snapshot, not a score ledger replacement.  Exact score
claims remain attached to their `contest_auth_eval` artifacts.

## Current Exact Anchor

- Archive: PR85 adaptive masking joint frame model replay.
- Evidence grade: `A++` T4 exact CUDA auth eval.
- Artifact:
  `experiments/results/lightning_batch/exact_eval_public_pr85_adaptive_masking_joint_frame_model_t4_20260503T2332Z/contest_auth_eval.adjudicated.json`
- Score:
  `0.25806611029397786`
- Archive bytes:
  `236328`
- Archive SHA-256:
  `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`
- SegNet:
  `0.00057185`
- PoseNet:
  `0.0001894`
- T4 runtime tree SHA-256:
  `d008db50f8a8165a9c7a5954e8eeec443878dbeb7a892e1d455515b4586b1a73`

## Latest Public Frontier Signal

Fresh GitHub API and README refresh during this turn shows PR85 as the current
observed public README leader by rounded score, with PR86 second:

- README master commit:
  `e84851da32108fcadf243c54d091cc71dc150c0e`
- README artifact:
  `experiments/results/public_leaderboard_score_trajectory_20260504_worker/downloads/master_README.md`
- README SHA-256:
  `010021c244403ed3fc6bc5db5bf907e386d9ce37b074dce0a411a5423e97e176`

Observed README top five:

1. PR85 `adaptive_masking_joint_frame_model`: rounded score `0.26`
2. PR86 `jas0xf_adversarial_neural_representation`: rounded score `0.27`
3. PR84 `adaptive_range_mask_no_router`: rounded score `0.28`
4. PR81 `qzs3_range_mask`: rounded score `0.28`
5. PR79 `qpose14_r55_segactions_minp`: rounded score `0.31`

Live API refresh after the worker snapshot also shows one new open scored PR:

- PR90:
  <https://github.com/commaai/comma_video_compression_challenge/pull/90>
- Title:
  `add qrepro submission(0.28)`
- State:
  `open`
- Head SHA:
  `cce857392701e73861ad513d34906faba523f719`
- Updated at:
  `2026-05-04T04:16:45Z`
- Current action:
  public intake/reverse-engineering worker spawned; no score claim until a
  replayable archive and exact CUDA evidence exist.
- Read-only API/body triage:
  PR90 reports `218080` bytes, PoseNet `0.00041977`, SegNet `0.00068872`,
  and score `0.278872`; this is worse than the current PR85 exact anchor.
- Downloaded archive triage:
  one stored zip member `p`, `218080` total bytes, payload `217980` bytes,
  SHA-256 `608ea0355e60faad97b046c27644205d05120ac85ab3e8a99543a75a4ab2dd2d`.
- Architecture signal:
  semantic geometry mask stream, FP4 renderer, explicit pose/bias controls,
  and about `4106` bytes of low-frequency QRGB residual controls.  Treat QRGB
  and semantic-geometry decomposition as transfer ideas; do not treat PR90 as
  a frontier replacement.

PR86 remains the latest merged public submission and an important architecture
signal:

- PR:
  <https://github.com/commaai/comma_video_compression_challenge/pull/86>
- Title:
  `jas0xf_adversarial_neural_representation (0.27)`
- Head SHA:
  `0eabe354f09b7490fd1cbb2b05a9102ab528d4d4`
- Merge commit:
  `14bcede815306415a0005c3cd98804151bce4049`
- Merged at:
  `2026-05-04T03:36:55Z`

Local PR86 replay remains failed-closed before score JSON.  Its archive is
external design signal until full decode/reencode parity and exact CUDA replay
exist.

Open-PR watch:

- PR70 is open but classified by the worker as non-faithful by author note
  because it uses a source-embedded payload pattern.
- PR72 is open WIP with no parsed score.
- PR90 through PR100 currently return GitHub API `404`.

Public-trajectory artifact:

- `experiments/results/public_leaderboard_score_trajectory_20260504_worker/public_leaderboard_score_trajectory_20260504_worker.json`

## Current Matrix

- Matrix:
  `experiments/results/pr85_full_stack_opportunity_matrix_20260504_orchestrator_final6/pr85_full_stack_opportunity_matrix.json`
- Stable digest:
  `a2191b4115578651ab74a662dcf1304b8442a88ba4878b7b36adcb80d679e6ce`

Top stack plan order:

1. `qma9_native_run_grammar_or_table_reduction`
   - bytes at stake: `159011`
   - status: unblocked planning family, but requires full-stream deterministic
     encoder or explicit charged runtime grammar.
2. `pr86_hpac_pr85_mask_contract_port`
   - bytes at stake: `15369`
   - status: blocked on PR86 full decode/reencode byte parity.
3. `scorer_gradient_pair_atom_policy`
   - bytes at stake: `517` for the current top pair break-even.
   - status: pair-action lowering exists but is blocked on grounded
     stream/value action evidence and a non-noop archive-changing path.
4. `pr89_final_bias_stack_on_pr85`
   - bytes at stake: `380`
   - status: blocked unless exact component benefit exceeds charged bytes.

New guardrail:

- `qma9_alternate_neighbor_table_grammar_screen` is now recorded as
  `empirical_alt_grammar_full_stream_no_byte_win`.
- Best screened alternate was `adaptive9up2left2`, `161034` bytes,
  `+2023` vs source QMA9.
- Do not redispatch screened neighbor/table modes.
- `qma9_qrg1_row_run_grammar_screen` is now recorded as
  `empirical_qrg1_full_stream_no_byte_win`.
- Best QRG1 screened mode was `row_rle_lzma6`, `462176` bytes,
  `+303165` vs source QMA9.
- Do not dispatch QRG1 row-run grammars without a materially different
  transform, not just another compressor level.
- `pr86_hpac_probability_contract_variants` is now recorded as
  `fail_closed_probability_variants_no_full_decode`.
- Four source/dtype/perfect-mode variants produced no full decode and no
  byte-exact reencode; best prefix was `source_float32_perfect_false` at
  `30513` decoded symbols.
- Do not port PR86 HPAC to PR85 until submitted-token full decode plus
  byte-exact reencode parity is proven.
- Matrix discovery now supports absolute glob patterns in both default
  discovery and explicit `--input` overrides.  This permanently fixes the
  `/tmp/pr85_pair_action_candidates_*` orchestration bug exposed by final5.

## Worker State

Closed earlier workers:

- `019df129-5bdf-76e3-94a5-78d0cdbd5a05`: public PR/score-trajectory intake.
- `019df129-769f-79c0-b968-a507ed61211b`: PR85 QMA9 run-grammar compiler;
  no byte-positive QRG1 candidate.
- `019df12b-ac6c-72c1-8eda-6b50ac156928`: loop-hotpath/vectorization hardening.
- `019df12d-1edc-7c23-a9ba-b93f91382175`: PR86 HPAC probability-contract
  recovery; no full-decode or byte-parity variant.

Active current workers:

- PR85 QRGB transfer planner from PR90 residual-control signal.

Closed current workers:

- PR85 fixed-runtime atom substrate/readiness:
  landed `fixed_runtime_bridge.atom_substrate`, `custody_expectations`, and
  `atom_edit_guard` in the preflight.  Bridge substrate remains ready, but PR85
  atom edits must now prove candidate-vs-source non-noop segment changes and
  expected archive/member SHA custody before exact-eval dispatch.
- PR85 explicit pair-action lowering:
  landed a deterministic lowering surface.  Current artifacts emit eight
  blocked specs for pairs `192`, `60`, `164`, `197`, `70`, `496`, `106`, and
  `522`; exact eval remains locked because the pair-gradient plans still lack
  grounded stream/value action evidence and non-noop archive-changing paths.
- PR90 public qrepro intake:
  replayable enough for public intake, strict one-member archive, bounded CPU
  smoke wrote evaluator-shaped raw bytes before an intentional timeout.  Best
  transfer idea is QRGB-style low-frequency residual controls; full transplant
  is lower priority because reported components regress against PR85.

No active worker is authorized to dispatch GPU work.  Any exact eval or remote
training must first pass the Level-2 dispatch claim protocol.

## New Profiling Artifacts

Archive bit budget profile:

- `experiments/results/archive_bit_budget_public_frontier_20260504_orchestrator/profile.json`
- Profiles PR85, PR86, and PR89 public archives.
- PR85 single payload has no generic self-compression win under the current
  probes.
- PR86 `tokens.bin` and `hpac.pt.ppmd` show no generic recompression win;
  `meta.pt` has `930` byte generic savings but PR86 replay is not valid yet.

Python loop hotpath profile:

- `experiments/results/python_loop_hotpaths_20260504_orchestrator/profile.json`
- Loop count:
  `5857`
- Top active local-iteration bottlenecks:
  - `experiments/plan_cmg3_pixel_lagrangian_atoms.py`
  - `experiments/plan_predictive_mask_hotspot.py`
  - `submissions/robust_current/apply_qzs3_postprocess.py`
  - `src/tac/mask_entropy_coder.py`
  - QMA9 run-escape builders

This is engineering telemetry only; it is not score evidence.

CMG3A local-speed follow-up:

- Worker `019df12b-ac6c-72c1-8eda-6b50ac156928` vectorized
  `_ranked_nonzero_runs()` in `experiments/build_cmg3_adaptive_runs_candidate.py`.
- Equivalence test added against the base per-row scanner.
- Reported microbench:
  `3.76x` speedup on mask-like long-run rows.
- Focused verification after integration:
  `19 passed, 1 expected duplicate-ZIP warning`.
- Remote-dispatch decision:
  no CMG3A exact eval should be launched from this alone.  Prior exact CMG3
  candidates are far outside the PR85 basin:
  - T4 CMG3 top1 score `29.22356762140238`
  - T4 CMG3 top2 score `9.415026478717273`
  - L40S CMG3A body200 score `7.014092131069795`

This speedup is useful for future atom/compiler iterations, not a current
frontier score path.

## Next Dispatch Rule

Do not spend T4/remote GPU on any archive until one of these happens:

1. QMA9 run grammar emits a byte-positive candidate with decoded-token parity,
   deterministic archive closure, and explicit runtime contract.
2. PR86 HPAC variant proves full submitted-token decode plus byte-exact
   reencode.
3. Pair-gradient atoms are lowered into explicit charged PR85 actions with
   local parity.
4. Public PR refresh exposes a newer merged contest-faithful archive that can
   be replayed exactly.

Once one of those gates passes, claim the lane with
`tools/claim_lane_dispatch.py claim ...`, then run canonical
`archive.zip -> inflate.sh -> upstream/evaluate.py` CUDA eval.

## 2026-05-04T04:50Z Active Exact-Eval Wave

The PR85 QRGB transfer lane crossed the dispatch gate through explicit
PR85-native `bias`/`region` pair-action evidence, non-noop archive-byte
mutation, fixed-runtime preflight, and Level-2 lane claims.  This wave is the
current fastest exact-score path because it changes charged bytes inside the
best known faithful archive while preserving the PR85 replay runtime.

Queued T4/equivalent exact evals:

- `exact_eval_pr85_qrgb_f1_bias_pair_0060_t4_20260504T0450Z`
  - state:
    `.omx/state/pr85_qrgb_f1_bias_pair_0060_t4_20260504T0450Z_batch_jobs.json`
  - candidate archive:
    `experiments/results/pr85_qrgb_pair_atom_archive_candidates_20260504_codex/pr85_qrgb_f1_bias_pair_0060/archive.zip`
  - bytes: `236336`
  - sha256:
    `81fb8d715e37966ead2764f21846909f4bd570f2bfdc5469c53a83ded495bc81`
- `exact_eval_pr85_qrgb_f1_bias_pair_0164_t4_20260504T0450Z`
  - state:
    `.omx/state/pr85_qrgb_f1_bias_pair_0164_t4_20260504T0450Z_batch_jobs.json`
  - candidate archive:
    `experiments/results/pr85_qrgb_pair_atom_archive_candidates_20260504_codex/pr85_qrgb_f1_bias_pair_0164/archive.zip`
  - bytes: `236335`
  - sha256:
    `d5e2a7904f3a9f5333220670e9fe7a99a8c665f16a63b2af90f5c366202fde9e`
- `exact_eval_pr85_qrgb_f1_region_pair_0197_t4_20260504T0450Z`
  - state:
    `.omx/state/pr85_qrgb_f1_region_pair_0197_t4_20260504T0450Z_batch_jobs.json`
  - candidate archive:
    `experiments/results/pr85_qrgb_pair_atom_archive_candidates_20260504_codex/pr85_qrgb_f1_region_pair_0197/archive.zip`
  - bytes: `236335`
  - sha256:
    `236751af46a9c98fa286ecfe613c23a2b96bffbe31784da052304701e02b71c6`

Shared staging manifest:

- `.omx/state/pr85_qrgb_pair_atoms_t4_20260504T0450Z_manifest.json`
- remote verification: `REMOTE_MANIFEST_VERIFY OK`
- staged files: `1116`
- staged bytes: `17594414`

Baseline for comparison:

- PR85 faithful exact T4 score:
  `0.25806611029397786`
- PR85 archive bytes: `236328`
- PR85 archive sha256:
  `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`

Decision rule:

- If any QRGB pair atom lowers exact score, promote it to the frontier stack
  substrate and run pair-combination search with exact-eval confirmation.
- If all three regress, preserve the component deltas as signed atom-training
  labels for the scorer-gradient/QRGB action learner and shift the next dispatch
  to QMA9 native grammar or PR86 HPAC only if those workers remove their parity
  blockers.

## 2026-05-04T04:56Z Worker Closure And Diagnostic Hedges

Worker closures:

- PR86 HPAC-to-PR85 contract port is fail-closed.
  - plan:
    `experiments/results/public_pr86_intake_20260504_codex/pr86_hpac_pr85_contract_port_plan.json`
  - plan sha256:
    `d8aae2bcfe2dafc892c3c29604a197ce30aa88c485ed41978ff060bd49b7d6fe`
  - stable digest:
    `58b964de20b7a885d4983ea116d3968bdf12615897a5b37dc9bf1571d98e374a`
  - blocker:
    `pr86_hpac_pr85_mask_contract_port`
  - status:
    `blocked_fail_closed`
  - reason:
    PR86 full HPAC decode/reencode parity still fails and the PR85 HPAC parity
    probe is blocked by an entropy decode assertion.
- PR85 QMA9 native grammar/table reduction is fail-closed.
  - summary:
    `experiments/results/pr85_qma9_native_grammar_candidates_20260504_codex/candidate_summary.json`
  - summary sha256:
    `7f1ec6ba1a32ef693d318df8656729080ffd515e30e23d807d7cc3ef412d4182`
  - blocker:
    `no_deterministic_byte_closed_pr85_qma9_native_run_or_table_candidate`
  - reason:
    QMA9 source has no post-declared bytes, no trailing zero bytes, and prefix
    trims through 16 bytes decode to the wrong token SHA under the current
    runtime.  Prior row-run/table/mode screens are byte-negative or
    runtime-locked.

Both closures are useful engineering signal but neither emits a candidate
archive, and neither should consume remote GPU until its named blocker is
removed by a byte-closed, parity-proven implementation.

Additional diagnostic hedges:

- `exact_eval_pr85_qrgb_f1_bias_pair_0060_l40sdiag_20260504T0458Z`
- `exact_eval_pr85_qrgb_f1_bias_pair_0164_l40sdiag_20260504T0458Z`
- `exact_eval_pr85_qrgb_f1_region_pair_0197_l40sdiag_20260504T0458Z`

These use the same exact candidate archive bytes as the T4 wave on L40S
machine class `g6e.4xlarge`.  They are explicitly diagnostic-only and cannot
supersede T4/equivalent promotion evidence, but they can provide component sign
signal while the T4 queue is pending.

## 2026-05-04T05:00Z Public PR91 Frontier Intake

New public PR:

- PR: https://github.com/commaai/comma_video_compression_challenge/pull/91
- title: `Hpac coder hybrid`
- author: `ottokunkel`
- claimed exact score: `0.24879480490416128`
- claimed archive bytes: `222404`
- claimed components:
  - PoseNet: `0.00018940`
  - SegNet: `0.00057185`
- claim summary:
  PR86 HPAC mask compressor over PR85 masks, with fallback to the old PR when
  compressor fails.

Local custody:

- archive:
  `experiments/results/public_pr91_intake_20260504_codex/archive.zip`
- archive sha256:
  `4c16d04c746c981feb902e4dd508ffadaf3615e532d351993c3d2f6eccda1b4f`
- replay runtime:
  `experiments/results/public_pr91_intake_20260504_codex/replay_submission/hpac_coder_hybrid/`
- intake ledger:
  `.omx/research/public_pr91_hpac_hybrid_intake_20260504_codex.md`

Archive anatomy:

- one stored ZIP member: `x`
- member bytes: `222304`
- member sha256:
  `5c213c61cc4d29b62286063bfdcb97e812af6b06c0021aeaecc8bc46644e17bf`
- mask payload: `HPM1`, `145087` bytes
- model payload: `57074` bytes
- pose payload: `1487` bytes
- post/shift/frac/frac2/frac3/bias/region/randmulti total:
  `18656` bytes plus `24` byte header

Exact replay dispatch:

- T4 job:
  `exact_eval_public_pr91_hpac_hybrid_t4_20260504T0504Z`
- T4 state:
  `.omx/state/public_pr91_hpac_hybrid_t4_20260504T0504Z_batch_jobs.json`
- T4 status at queue:
  `Pending`, zero cost
- L40S diagnostic job:
  `exact_eval_public_pr91_hpac_hybrid_l40sdiag_20260504T0506Z`
- L40S state:
  `.omx/state/public_pr91_hpac_hybrid_l40sdiag_20260504T0506Z_batch_jobs.json`
- L40S status at queue:
  `Pending`, zero cost
- T4 hedge job:
  `exact_eval_public_pr91_hpac_hybrid_t4_g4dn2x_20260504T0509Z`
- T4 hedge state:
  `.omx/state/public_pr91_hpac_hybrid_t4_g4dn2x_20260504T0509Z_batch_jobs.json`
- T4 hedge status at queue:
  `Pending`, zero cost

Priority change:

- PR91 exact replay is now the primary frontier confirmation path.
- If confirmed on T4, optimization target becomes PR91's HPAC-hybrid basin,
  not raw PR85.
- PR85 QRGB atom singleton results remain useful as signed transfer evidence,
  but should not monopolize the next implementation tranche if PR91 confirms.

## 2026-05-04T05:06Z PR85 QRGB Combo Readiness

The QRGB combo worker built local byte-closed PR85 combo archives without
remote dispatch and with fixed-runtime preflight ready.  These are not score
claims; exact singleton evidence should decide which combo, if any, deserves
T4.

Artifacts:

- builder:
  `experiments/build_pr85_qrgb_pair_atom_combo_candidates.py`
- tests:
  `src/tac/tests/test_build_pr85_qrgb_pair_atom_combo_candidates.py`
- planning:
  `experiments/results/pr85_qrgb_pair_atom_combo_candidates_20260504_worker/combo_planning.json`
- action spec:
  `experiments/results/pr85_qrgb_pair_atom_combo_candidates_20260504_worker/action_spec_combos.json`

Built combos:

- `0060+0164`
  - bytes: `236336`
  - sha256:
    `90de17b17a946671c6e3aac36cc6773dd37ef10a0e064a80283caf9820bdd013`
- `0060+0197`
  - bytes: `236337`
  - sha256:
    `05f3e9e8f980abcd10ac2d6471308aef740b317d4cdde29ddb040b686ed77c18`
- `0164+0197`
  - bytes: `236336`
  - sha256:
    `101a26365e1b4e289bc18fccf5b25e37381870c248fafb085ff8d4a0804c025c`
- `0060+0164+0197`
  - bytes: `236337`
  - sha256:
    `708d89f158c4b08641a502f15c58865b571262dcc8bc142bf0c723758ba5d286`

Verification reported by worker:

- combo unit tests: passed
- py_compile: passed

Dispatch policy:

- Do not queue combos before singleton exact evidence unless PR91 replay fails
  and wall-clock pressure justifies a risky PR85-only branch.
- If one or more singleton atoms improves PR85 on exact T4, queue the matching
  combo immediately with a Level-2 dispatch claim.

## 2026-05-04T05:14Z PR85 QRGB Singleton Exact Results

The three PR85 QRGB singleton atom archives completed on T4/equivalent and
were harvested through the state-derived SSH path.  All three are A++ exact
CUDA evidence and all three regress slightly versus the PR85 public T4 anchor.

Results:

- `pr85_qrgb_f1_bias_pair_0060`
  - T4 score: `0.2580739080216157`
  - delta versus PR85: `+0.000007797727637870455`
  - bytes: `236336`
  - sha256:
    `81fb8d715e37966ead2764f21846909f4bd570f2bfdc5469c53a83ded495bc81`
  - artifact:
    `experiments/results/lightning_batch/exact_eval_pr85_qrgb_f1_bias_pair_0060_t4_20260504T0450Z/contest_auth_eval.adjudicated.json`
- `pr85_qrgb_f1_bias_pair_0164`
  - T4 score: `0.25808234771935784`
  - delta versus PR85: `+0.000016237425379983517`
  - bytes: `236335`
  - sha256:
    `d5e2a7904f3a9f5333220670e9fe7a99a8c665f16a63b2af90f5c366202fde9e`
  - artifact:
    `experiments/results/lightning_batch/exact_eval_pr85_qrgb_f1_bias_pair_0164_t4_20260504T0450Z/contest_auth_eval.adjudicated.json`
- `pr85_qrgb_f1_region_pair_0197`
  - T4 score: `0.2580777531130102`
  - delta versus PR85: `+0.00001164281903232034`
  - bytes: `236335`
  - sha256:
    `236751af46a9c98fa286ecfe613c23a2b96bffbe31784da052304701e02b71c6`
  - artifact:
    `experiments/results/lightning_batch/exact_eval_pr85_qrgb_f1_region_pair_0197_t4_20260504T0450Z/contest_auth_eval.adjudicated.json`

The L40S hedges completed before stop could take effect and were mirrored as
diagnostic-only duplicate evidence:

- `0060` L40S score: `0.25931028308646`
- `0164` L40S score: `0.25931066136375286`
- `0197` L40S score: `0.2593129178321656`

Decision:

- Do not dispatch the prepared PR85 QRGB combo archives from this singleton
  wave.  The singleton exact evidence is directionally negative and the L40S
  diagnostics agree that the basin worsens.
- Keep the QRGB builder and artifacts as transfer machinery only.  Reconsider
  them if PR91 exact replay confirms a different HPAC mask basin and a
  PR91-specific builder proves byte closure without weakening the PR85 scorer
  gradient guard.

## 2026-05-04T05:19Z PR91 Replay Failed Before Score

The PR91 public report claims exact score `0.24879480490416128`, but our
canonical replay of the downloaded PR91 archive did not inflate successfully.

Observed failures:

- T4 hedge
  `exact_eval_public_pr91_hpac_hybrid_t4_g4dn2x_20260504T0509Z` failed before
  `contest_auth_eval.json` with terminal class `inflate_returncode_failure`.
- L40S diagnostic
  `exact_eval_public_pr91_hpac_hybrid_l40sdiag_20260504T0506Z` failed with the
  same HPM1 entropy decode assertion.
- Primary T4
  `exact_eval_public_pr91_hpac_hybrid_t4_20260504T0504Z` was stopped as
  redundant; refreshed status `Stopped`, cost about `0.022694444`, no artifacts
  visible.

The shared log signature:

```text
AssertionError: Tried to decode from compressed data that is invalid for the employed entropy model.
```

Decision:

- Do not promote PR91 locally.
- Treat PR91 as external leaderboard/PR intelligence until HPM1 parity is
  repaired or explained.
- Continue optimizing from the exact PR85 T4 anchor and use PR91 anatomy as
  evidence that the mask stream is the active byte lever.

## 2026-05-04T06:01Z PR85 QRGB Randmulti Exact Negative

The last live QRGB singleton, `pr85_qrgb_f2_randglobal_pair_0192`, completed on
both T4/equivalent Lightning packets and reproduced the same exact negative.

- T4 score: `0.25826470562795345`
- delta versus PR85: `+0.00019859533397559304`
- bytes: `236616`
- sha256:
  `228f8dff9e14bc7d3cdd445d6c7d73ed1818c0facecaa21e97ab71a523b2da40`
- SegNet: `0.00057187`
- PoseNet: `0.00018944`
- sample count: `600`
- evidence:
  `experiments/results/lightning_batch/exact_eval_pr85_qrgb_f2_randglobal_pair_0192_t4_20260504T0536Z/contest_auth_eval.adjudicated.json`
- duplicate evidence:
  `experiments/results/lightning_batch/exact_eval_pr85_qrgb_f2_randglobal_pair_0192_t4_g4dn2x_20260504T0544Z/contest_auth_eval.adjudicated.json`

Decision:

- Close the QRGB singleton wave as measured-implementation negative for PR85.
- Do not dispatch QRGB combo or STBM+QRGB stack variants from this evidence;
  the only currently live exact score-mover remains the STBM1BR lossless mask
  recode.

## 2026-05-04T06:30Z STBM1BR Lossless Mask Recode Exact Positive

The corrected PR85 STBM1BR lossless mask-recode hedge completed on
T4/equivalent hardware and is now the verified internal frontier.

- Job:
  `exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z`
- Artifact:
  `experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z/contest_auth_eval.adjudicated.json`
- Adjudication:
  `experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z/adjudication_provenance.json`
- Score: `0.25369011029397787`
- Delta versus PR85 exact T4 `0.25806611029397786`:
  `-0.004375999999999991`
- Archive bytes/SHA-256:
  `229756`,
  `c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6`
- Components: SegNet `0.00057185`, PoseNet `0.0001894`, samples `600`
- Hardware/evidence: Tesla T4, `A++ contest T4`, promotion eligible

Interpretation:

- The exact result realizes the predicted pure-rate win. The mask decode parity
  was correct: SegNet and PoseNet stayed unchanged at contest JSON precision.
- PR85 non-mask pure-rate search found no additional byte-negative candidate in
  its local scope, so the remaining material score levers are mask-codec
  replacement/parity recovery and geometry-aware PR85-token coding, not generic
  ZIP/member repacking.
- PR91/HPM1 remains the largest known public rate signal (`13924` byte
  opportunity versus PR85, about `-0.009271420063` score if neutral), but it is
  blocked on the real HPM1 probability/token-generation contract.

Immediate next gates:

- Primary duplicate
  `exact_eval_pr85_stbm1br_stbm_runtime_t4_20260504T0610Z` is confirmed
  `Stopped` at cost `0.056366667` after the hedge success.
- Update report/frontier surfaces to anchor on `0.25369011029397787`.
- Continue PR91/HPM1 parity recovery and Rust/STBM decode lowering in parallel.
- Do not dispatch STBM+QRGB from the existing stack builder: QRGB standalone
  exact evidence is negative.

## 2026-05-04T07:25Z Public Frontier Refresh And PR92 Anatomy

Live GitHub PR refresh through the repository API shows the newest relevant
open submissions are:

- PR94 `optimization_qpose_josema`, updated `2026-05-04T07:17:11Z`.
  Reported evidence is local Mac MPS, so it is not a CUDA score anchor.
- PR92 `qzs3_range_joint_r258 (0.26)`, updated
  `2026-05-04T06:24:44Z`.  Public report gives PoseNet `0.00018963`,
  SegNet `0.00057675`, and archive bytes `236516`.
- PR91 `Hpac coder hybrid`, still open, remains external/intake until the HPM1
  entropy contract replay failure is repaired.

PR92 score recomputation from its published components:

```text
100 * 0.00057675
+ sqrt(10 * 0.00018963)
+ 25 * 236516 / 37545489
= 0.2587078229986317
```

This is worse than the current exact T4 frontier `0.25369011029397787`, but it
is close enough to mine for stackable side-channel ideas.

Local PR92 intake artifacts:

- Archive:
  `experiments/results/public_pr92_intake_20260504_codex/archive.zip`
- Archive SHA-256:
  `f0dedeb7ad3c019ab3f4e2dea317f3192408e47a573897adb208030709d01490`
- Runtime files:
  `experiments/results/public_pr92_intake_20260504_codex/inflate.py`
  (`c04cba8431f076838031f013aa005cdb7f530133d5d1856bfe0f02def3f11261`)
  and
  `experiments/results/public_pr92_intake_20260504_codex/range_mask_codec.cpp`
  (`94cd1a86111fb6d34b6e12d37c624bd5938df0fbc6c4c24c8d40c5a83fcb016b`).

PR92 ZIP members:

- `x`: `235952` bytes
- `a`: `386` bytes, magic `RSB1`

PR92 compact member `x` parses as a PR85-family single-blob bundle with:

| segment | bytes | first bytes |
| --- | ---: | --- |
| mask | `159011` | `514d413958020000` |
| model | `57074` | `b1a8843f8097a162` |
| pose | `1487` | `a2e120a29a3463fe` |
| post | `1400` | `e22b61464ace67fc` |
| shift | `226` | `424b00ff20803ad5` |
| frac | `106` | `421600009836dc8d` |
| frac2 | `149` | `424b00bf91727574` |
| frac3 | `154` | `424b00ff919235bc` |
| tail | `16321` | `424b00ff919235bc` |

Interpretation:

- PR92 does not beat the current exact frontier, but its anatomy confirms the
  active public direction: PR85-family JointFrameGenerator plus range-coded QMA9
  mask, compact pose, small correction streams, and tiny action sidecars.
- The `159011` byte mask is the same charged QMA9 mask scale that our residual
  sufficient-program profiler uses as the native-token source.  Direct sparse
  residual coding is therefore not a simple immediate win; it should feed
  native/Rust context design and learned curriculum.
- PR94's bundled HPAC files are byte-identical to the PR91 HPAC runtime files
  already mirrored locally, so PR94 does not provide a new HPAC probability
  contract.

Immediate next action:

- Build a strict PR92-vs-PR85/STBM sidecar anatomy comparator that recognizes
  `RSB1`, PR85 compact bundle variants, and QMA9/randmulti payloads, then use
  it to decide whether PR92's `a`/tail side-action streams are stackable onto
  the STBM1BR frontier without changing mask semantics.

## 2026-05-04T08:05Z STBM1BR + PR92 RMB1 Randmulti Candidate

PR92's byte-only intake showed that its only byte-positive PR85-family segment
versus PR85/STBM is the `randmulti` recode:

- STBM/PR85 randmulti: `16101` bytes,
  SHA `c624372e60a0851c4c427dc333a60dcc5d6657ba8ed56951612e7c9d7be7629f`
- PR92 `RMB1` randmulti: `15825` bytes,
  SHA `4b10018eab64d8755da3def355881f51f2d450c9f19dd1457a6ec813cddd6f7c`
- Both decode to identical headerless sparse rows:
  `27105` bytes,
  SHA `87bcc720c1e80afb9adad5ee01477423ced526f31c54d461d69dbf26e08eecc9`

Runtime support was added for raw `RMB1` randmulti segments in
`submissions/robust_current/apply_qzs3_postprocess.py`; shared decode/parity
helpers were added to `src/tac/pr85_bundle.py`.

Built deterministic candidate:

- Candidate id: `pr85_stbm1br_plus_pr92_rmb1_randmulti`
- Archive:
  `experiments/results/pr85_stbm1br_rmb1_randmulti_20260504_codex/pr85_stbm1br_plus_pr92_rmb1_randmulti/archive.zip`
- Archive bytes/SHA-256:
  `229480`,
  `f8d2dff12004fe15bdedefcd3f9574fab97f22c302fa1417a265c325468ad774`
- Manifest:
  `experiments/results/pr85_stbm1br_rmb1_randmulti_20260504_codex/pr85_stbm1br_plus_pr92_rmb1_randmulti/manifest.json`
- Pre-submission compliance:
  `experiments/results/pr85_stbm1br_rmb1_randmulti_20260504_codex/pr85_stbm1br_plus_pr92_rmb1_randmulti/pre_submission_compliance.json`

Fail-closed checks passed:

- source archive matches current STBM A++ SHA/bytes;
- PR92 archive matches public SHA;
- STBM mask bytes unchanged;
- randmulti is a non-noop byte change;
- randmulti decoded rows are identical;
- archive is deterministic single-member `ZIP_STORED`;
- no scorer, GPU, or dispatch occurred during build.

Expected pure-rate effect if exact components remain unchanged:

```text
delta_bytes = -276
delta_score = 25 * -276 / 37,545,489 = -0.0001837770710617193
projected_score = 0.25369011029397787 - 0.0001837770710617193
                = 0.25350633322291617
```

Decision:

- This is small, but it is a deterministic lossless recode of a charged stream
  with exact row parity and clean pre-submission compliance. It justifies one
  T4 exact CUDA eval after a Level-2 dispatch claim.

## 2026-05-04T07:45Z STBM1BR + PR92 RMB1 Randmulti Exact-Eval Dispatch

Queued the byte-closed standalone PR92 `RMB1` randmulti recode stacked on the
A++ `PR85_STBM1BR` archive. This is a pure-rate hypothesis, not a score claim
until exact CUDA returns.

- Candidate archive:
  `experiments/results/pr85_stbm1br_rmb1_randmulti_20260504_codex/pr85_stbm1br_plus_pr92_rmb1_randmulti/archive.zip`
- Bytes/SHA-256:
  `229480`,
  `f8d2dff12004fe15bdedefcd3f9574fab97f22c302fa1417a265c325468ad774`
- Source frontier bytes/SHA-256:
  `229756`,
  `c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6`
- Decoded randmulti row SHA-256:
  `87bcc720c1e80afb9adad5ee01477423ced526f31c54d461d69dbf26e08eecc9`
- Expected rate-only delta if components are unchanged:
  `-0.0001837770710617193`
- Predicted exact score if components are unchanged:
  approximately `0.25350633322291617`
- Lane claim:
  `pr85_stbm1br_pr92_rmb1_randmulti`
- Lightning job:
  `exact_eval_pr85_stbm1br_plus_pr92_rmb1_randmulti_t4_g4dn2x_20260504T0743Z`
- Source manifest:
  `.omx/state/pr85_stbm1br_rmb1_randmulti_t4_g4dn2x_20260504T0743Z_manifest.json`
- Repro plan:
  `.omx/state/pr85_stbm1br_rmb1_randmulti_t4_g4dn2x_20260504T0743Z_lightning_exact_eval_repro_plan.json`
- Local artifact dir:
  `experiments/results/lightning_batch/exact_eval_pr85_stbm1br_plus_pr92_rmb1_randmulti_t4_g4dn2x_20260504T0743Z`
- Runtime:
  `submissions/robust_current/inflate.sh`, using the current STBM1BR and RMB1
  decode support rather than the older dedicated STBM replay inflater.

Pre-dispatch local verification:

- `py_compile` passed for the candidate builder, bundle helpers, robust runtime
  postprocess, and Lightning staging/submit helpers.
- Focused pytest passed: `30 passed, 1 warning` for bundle, RMB1 runtime,
  candidate builder, and pre-submission compliance tests.
- `scripts/pre_submission_compliance_check.py` passed for the exact archive
  bytes and SHA-256.
- Lightning source/artifact staging verified `1621` files and `28861754` bytes;
  local and remote supply-chain scans were clean.

Decision context:

- This exact eval is cheap and byte-closed, so it can run while PR91/HPM1 parity
  recovery continues.
- It is not a substitute for PR91/HPM1: HPM1 still represents the larger known
  rate opportunity (`-7352` bytes versus STBM1BR) if parity/runtime gates are
  recovered.

## 2026-05-04T07:49Z RMB1 Builder Runtime-Contract Hardening

After dispatch, hardened the canonical local RMB1 builder so future candidate
builds fail closed if the selected production runtime cannot consume the exact
archive representation.

Changed production-facing guard:

- `experiments/build_pr85_stbm1br_rmb1_randmulti_candidate.py` now records a
  `runtime_support` block in `manifest.json` and requires:
  - `submissions/robust_current/inflate_renderer.py` exists;
  - `STBM1BR_MAGIC` and `_load_masks_from_stbm1br` are present;
  - `submissions/robust_current/apply_qzs3_postprocess.py` exists;
  - `_decode_rmb1_randmulti_payload` and the `RMB1` dispatch branch are present;
  - the qpost runtime contains no scorer-load markers.
- The focused candidate test asserts this guard is present and green.
- Added `--candidate-id` and `--stdout` compatibility wiring so workers can use
  the canonical builder without inventing nearby variants.

Verification:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_build_pr85_stbm1br_rmb1_randmulti_candidate.py \
  src/tac/tests/test_pr85_bundle.py \
  src/tac/tests/test_qzs3_postprocess_qrm1_runtime.py -q
# 27 passed, 1 duplicate-ZIP warning from an intentional fail-closed test

.venv/bin/python experiments/build_pr85_stbm1br_rmb1_randmulti_candidate.py --stdout
# archive remains 229480 bytes, SHA f8d2dff12004fe15bdedefcd3f9574fab97f22c302fa1417a265c325468ad774

git diff --check -- \
  experiments/build_pr85_stbm1br_rmb1_randmulti_candidate.py \
  src/tac/tests/test_build_pr85_stbm1br_rmb1_randmulti_candidate.py \
  .omx/research/frontier_orchestration_snapshot_20260504_codex.md
# clean
```

The queued Lightning job was staged before this local builder-hardening change;
the archive bytes are unchanged and the remote exact-eval runtime was already
`submissions/robust_current/inflate.sh`, which contains the required STBM1BR and
RMB1 branches. The hardening is for future reproducible builds and audits, not a
new score claim.

## 2026-05-04T07:52Z Dispatch-Claim Parallel Reason Hardening

Fixed another recurring CLI fragility class in the Level-2 dispatch helper.
`tools/claim_lane_dispatch.py claim --allow-parallel` previously rejected
human-readable `--parallel-reason` values containing spaces even though the
ledger notes field can safely normalize whitespace. This caused a rejected hedge
claim during the RMB1 T4 queue hedge.

Change:

- `parallel_reason` now permits spaces while still rejecting empty strings,
  leading/trailing whitespace, control characters, and Markdown table
  separators.
- Added a regression test proving a human-readable parallel reason is accepted
  and written safely into `.omx/state/active_lane_dispatch_claims.md`.

Verification:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_claim_lane_dispatch.py \
  src/tac/tests/test_build_pr85_stbm1br_rmb1_randmulti_candidate.py -q
# 15 passed

.venv/bin/python -m py_compile \
  tools/claim_lane_dispatch.py \
  experiments/build_pr85_stbm1br_rmb1_randmulti_candidate.py \
  src/tac/tests/test_claim_lane_dispatch.py \
  src/tac/tests/test_build_pr85_stbm1br_rmb1_randmulti_candidate.py
# passed
```

RMB1 exact-eval hedge status after this fix:

- `exact_eval_pr85_stbm1br_plus_pr92_rmb1_randmulti_t4_g4dn2x_20260504T0743Z`:
  `Running`.
- `exact_eval_pr85_stbm1br_plus_pr92_rmb1_randmulti_t4_g4dn1x_20260504T0748Z`:
  `Pending` at zero cost.

## 2026-05-04T07:54Z PR91/HPM1 Blocker Reclassified

PR91/HPM1 remains the biggest known rate opportunity, but it is not exact-eval
ready.

Worker-local diagnostics added:

- `experiments/replay_pr91_hpm1_mask.py` gained a local-only
  `--probability-variant-matrix` mode.
- `src/tac/tests/test_pr91_hpm1_codec.py` gained a regression for the
  fail-closed blocker.
- Diagnostics:
  - `experiments/results/public_pr91_intake_20260504_codex/diagnostics/pr91_hpm1_preflight_frame0_20260504_current_codex.json`
  - `experiments/results/public_pr91_intake_20260504_codex/diagnostics/pr91_hpm1_probability_variant_matrix_frame0_20260504_current_codex.json`
  - `experiments/results/public_pr91_intake_20260504_codex/diagnostics/pr91_hpm1_pr85_stbm_fusion_plan_20260504_current_codex.json`

Current facts:

- PR91 archive bytes/SHA-256:
  `222404`,
  `4c16d04c746c981feb902e4dd508ffadaf3615e532d351993c3d2f6eccda1b4f`
- PR85_STBM1BR frontier bytes/SHA-256:
  `229756`,
  `c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6`
- Byte fusion versus STBM is clean: only `mask` changes, archive delta `-7352`
  bytes.
- Runtime/parity blocker: HPM1 decode fails closed at
  `frame=0, group=10, symbol_in_group=191` after `5951` decoded symbols.
- `dispatch_unlocked=false`, `pr91_ready_for_exact_eval=false`, `score_claim=false`.

Decision:

- Do not queue PR91/HPM1 exact eval until full HPM1 decode plus mask parity is
  proven.
- A new focused worker is assigned to the entropy-contract mismatch. The main
  thread continues harvesting RMB1 exact evals and hardening production gates.

## 2026-05-04T07:58Z SSH Harvest Not-Ready Hardening

Fixed the early-harvest bug class found while polling the running RMB1 exact
eval. Previously, `harvest-ssh` raised a raw `CalledProcessError` when the
provider artifact subdirectory was not present yet. That is normal for a
pending/running job and should not be recorded as method evidence or an infra
failure.

Change:

- `src/tac/deploy/lightning/batch_jobs.py` now returns a nonterminal
  `ARTIFACT_NOT_READY` diagnostic when the remote exact-eval artifact directory
  is missing.
- `harvest_ssh_artifacts()` returns that diagnostic without writing a local
  mirror, without appending `artifact_failures`, and without changing the job
  record status.
- Added regression coverage in `src/tac/tests/test_lightning_batch_jobs.py`.

Verification:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_lightning_batch_jobs.py::test_harvest_ssh_artifacts_records_empty_remote_dir_as_infra_failure \
  src/tac/tests/test_lightning_batch_jobs.py::test_harvest_ssh_artifacts_reports_missing_remote_dir_as_not_ready \
  src/tac/tests/test_lightning_batch_jobs.py::test_harvest_ssh_artifacts_records_partial_missing_score_json_as_infra_failure -q
# 3 passed

.venv/bin/python -m py_compile \
  src/tac/deploy/lightning/batch_jobs.py \
  src/tac/tests/test_lightning_batch_jobs.py
# passed
```

Live retry against the running RMB1 T4 exact eval now returns:

```json
{"status":"ARTIFACT_NOT_READY","reason":"remote Lightning exact-eval artifact directory is not present; the job may still be pending/running or provider artifact persistence may not have completed"}
```

This is a harness hardening result only; no score claim.

## 2026-05-04T08:01Z OSS/Production Custody Guard Integration

Integrated the production hardening pass from the OSS/readiness worker.

New guards:

- `scripts/pre_submission_compliance_check.py` requires auth-eval custody to
  include a 64-hex `runtime_tree_sha256` when an auth-eval artifact is supplied.
  This prevents publishing a score packet whose archive bytes are known but
  whose inflate/runtime tree is not auditable.
- `experiments/build_pr85_stbm1br_rmb1_randmulti_candidate.py` now verifies its
  duplicate/canonical PR92-RMB1 builder constants do not drift from the active
  lane/tool/schema contract.
- `tools/claim_lane_dispatch.py` accepts human-readable audited parallel reasons
  while preserving table-safety guards.
- SSH harvest returns nonterminal `ARTIFACT_NOT_READY` instead of throwing when
  a running job has not persisted artifacts yet.

Integrated verification:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_pre_submission_compliance_check.py \
  src/tac/tests/test_build_pr85_stbm1br_rmb1_randmulti_candidate.py \
  src/tac/tests/test_claim_lane_dispatch.py \
  src/tac/tests/test_lightning_batch_jobs.py::test_harvest_ssh_artifacts_reports_missing_remote_dir_as_not_ready -q
# 20 passed

.venv/bin/python -m py_compile \
  scripts/pre_submission_compliance_check.py \
  experiments/build_pr85_stbm1br_rmb1_randmulti_candidate.py \
  tools/claim_lane_dispatch.py \
  src/tac/deploy/lightning/batch_jobs.py \
  src/tac/tests/test_pre_submission_compliance_check.py \
  src/tac/tests/test_build_pr85_stbm1br_rmb1_randmulti_candidate.py \
  src/tac/tests/test_claim_lane_dispatch.py \
  src/tac/tests/test_lightning_batch_jobs.py
# passed

git diff --check -- touched hardening paths
# clean
```

This is harness/production hardening only. Current score frontier remains the
A++ `PR85_STBM1BR` exact T4 result until the queued RMB1 exact eval returns.

## 2026-05-04T08:05Z Current Frontier Release Gate Recheck

Re-ran the hardened pre-submission compliance gate against the current A++
`PR85_STBM1BR` frontier after adding the auth-eval runtime-tree requirement.

Command:

```bash
.venv/bin/python scripts/pre_submission_compliance_check.py \
  --archive experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z/archive.zip \
  --submission-dir submissions/robust_current \
  --expected-archive-sha256 c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6 \
  --expected-archive-size-bytes 229756 \
  --auth-eval-json experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z/contest_auth_eval.adjudicated.json \
  --output-json experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z/pre_submission_compliance_check_runtime_tree_required.json
```

Result: `status=passed`, no failed checks. The stricter custody gate does not
break the current release fallback.
