# Shannon-Floor Paper Rigor / Writeup Blueprint

**Date:** 2026-04-30
**Scope:** contest-paper-quality blueprint for documenting the Shannon-floor push without inflating evidence.
**Allowed score claims:** only the current Grade A/A++ ledger in `contest_grade_all_lane_results_audit_20260430.md` plus stricter later Codex progress addenda and exact archive-custody notes.
**Current A++ frontier:** C-058 QZS3/QP1 PR67-informed active-subspace byte micro-frontier on Tesla T4, score `0.3157555307844823`, archive bytes `276,422`, archive SHA `5145fb57be574b85639856d239420ffa35e605e32664f93e06753b120b21633f`, exact path `archive.zip -> inflate.sh -> upstream/evaluate.py`. C-057 remains the substantive anisotropic-basis anchor for the PR67 comparison narrative; C-058 is a one-byte exact successor for strict latest packets.
**Hard boundary:** no Grade A/A++ claim is allowed without exact archive CUDA evidence. Predictions, byte-only reports, smoke tests, CPU/MPS/proxy runs, and memory-only results are paper-routing evidence only.

---

## 1. Paper Contract

The paper/writeup must read like a reproducible compression-system paper, not an experiment diary. Every claim must be paired with an evidence grade, an artifact path, a formula or measurement procedure, and an adversarial review status.

### Primary claim classes

| Claim class | Example wording allowed | Evidence required | Forbidden wording before evidence |
|---|---|---|---|
| Grade A score result | "Lane X achieves recomputed score Y under the local CUDA score-grade ledger." | Exact archive, SHA match, CUDA eval report, 600 samples, component recomputation | "contest-identical" unless Grade A++ gate passes |
| Grade A++ contest-ready result | "C-058 QZS3/QP1 active-subspace byte micro-frontier is the current 1:1 contest-grade frontier at recomputed `0.3157555307844823`; C-057 is the PR67 comparison anchor." | Grade A plus clean manifest, payload closure, `archive.zip -> inflate.sh -> upstream/evaluate.py`, T4/equivalent device, 30-minute inflate budget | Any final-submission claim without A++ evidence |
| Empirical component result | "Component Y saved N bytes in a local archive builder." | Exact component report, manifest/hash if archive-shaped, no neural score claim | "improves score" |
| Derivation | "Saving 100,000 bytes changes the rate term by `0.06659`." | Formula and arithmetic | "measured improvement" |
| Prediction / design hypothesis | "Sensitivity weighting is expected to reduce PoseNet regression risk." | Stated mechanism, planned falsification gate | "validated", "frontier", "floor-moving" |
| External research | "HAWQ-style sensitivity allocation motivates the channel-allocation objective." | Citation/source doc plus contest translation | External benchmark as contest evidence |

### Evidence grade vocabulary

Use exactly these tags in the paper tables:

| Grade | Meaning | Can rank lanes? | Can promote? / Can kill? |
|---|---|---:|---:|
| `A++` | 1:1 contest-grade exact archive evidence | Yes | Promote yes; method/family kill still requires scoped review/proof |
| `A` | local CUDA score-grade exact archive evidence | Yes, with hardware caveat | Promote locally with caveat; method/family kill still requires scoped review/proof |
| `A-negative` | exact archive CUDA evidence showing a measured implementation regresses | No | Retire measured implementation/config only; method/family kill requires separate three-pass scope review |
| `B` | diagnostic CUDA evidence with incomplete artifact custody or schema | No | No |
| `empirical` | byte, round-trip, loss, smoke, or partial component measurement | No | No |
| `derivation` | formula-only conclusion | No | No |
| `prediction` | forward-looking hypothesis or model forecast | No | No |
| `external` | outside paper/OSS/leaderboard intake | No | No |
| `invalid` | CPU/MPS/proxy/no-op/stale/unreproducible score claim | No | No |

---

## 2. Source-Of-Truth Map

The writeup must cite internal source docs by role, not by memory.

| Source doc | Role in final paper | Claims that may be imported |
|---|---|---|
| `contest_grade_all_lane_results_audit_20260430.md` | Evidence ledger and scoring standard | Grade A rows, Grade A++ gate, demotion rules |
| `all_scores_forensic_audit_20260430.md` | Failure taxonomy | Engineering/config/methodology bug classes; no new score ranking |
| `shannon_floor_execution_readiness_20260430.md` | Execution plan | Ordering, promotion gates, review loop |
| `grand_council_paradigm_shift_to_shannon_floor_20260430.md` | Hypothesis inventory | α/β/γ framing only as strategy/prediction unless separately measured |
| `external_research_intake_shannon_floor_20260430.md` | Literature and OSS intake | Copy/Translate/Watch mapping; no external score claims |
| `codex_source_doc_structure_and_compliance_map_20260430.md` | Traceability layer | Implementation-to-source alignment rules |
| `*_codex_progress.md` ledgers | Implementation progress | Landed readiness work, test commands, remaining blockers |

All score tables in the paper must trace back to `contest_grade_all_lane_results_audit_20260430.md` or a later file with the same or stricter Grade A schema.

### Evidence taxonomy and allowed verbs

| Grade | Allowed verbs | Disallowed verbs | Required caveat |
|---|---|---|---|
| `A++` | achieves, improves, regresses, promotes, kills, is contest-ready | predicted, expected | Exact T4/equivalent 1:1 archive evidence must be named. |
| `A` | achieves locally, improves locally, regresses locally, ranks current local frontier | contest-identical, final, production-ready | State hardware caveat and missing A++ items. |
| `A-negative` | diagnoses exact regression, supports redesign | ranks frontier, kills family/paradigm, permanently retires before scope review | State exact scope, unresolved engineering explanations, review-pass count, and revival condition. |
| `B` | diagnoses, indicates, motivates, warns | ranks, promotes, kills | Explain missing artifact/provenance field. |
| `empirical` | saves bytes, round-trips, smokes, measures component property | improves score, preserves score | State no exact neural eval exists. |
| `derivation` | implies by formula, bounds, computes rate term | measures, validates | State formula-only, no archive eval. |
| `prediction` | hypothesizes, forecasts, prioritizes, dispatches | achieves, proves, validates | State falsification gate. |
| `external` | motivates, suggests, transfers, compares conceptually | proves contest result | State outside benchmark cannot rank this contest. |
| `invalid` | is excluded, is stale, is non-authoritative | supports | State why it is invalid. |

---

## 3. Mathematical Core

### Contest objective

The paper's methods and ablations must reduce to the contest score:

```text
score = 100 * seg_dist
      + sqrt(10 * pose_dist)
      + 25 * archive_bytes / 37,545,489
```

Required reporting for every scored archive:

| Field | Requirement |
|---|---|
| `archive_bytes` | Exact byte count of submitted archive |
| `seg_dist` | Full 600-sample evaluator field |
| `pose_dist` | Full 600-sample evaluator field |
| `score_reported` | Stored evaluator score |
| `score_recomputed` | Recomputed from formula |
| `score_delta` | `score_reported - score_recomputed` |
| `n_samples` | Must be 600 or equivalent full public-test count |

### Rate arithmetic

Use the rate slope for all byte claims:

```text
rate_points(bytes) = 25 * bytes / 37,545,489
delta_score_per_100KB = 25 * 100,000 / 37,545,489 = 0.06659
```

Any paper claim such as "saves N bytes" must also include:

```text
delta_rate_score = 25 * N / 37,545,489
```

This is a derivation, not a score improvement, until neural distortion is measured through exact archive eval.

### Dykstra ceiling bound

Dykstra/ADMM language is a feasibility discipline over intersecting
constraints, not proof that additive component deltas are realizable. For any
sub-`0.30` claim:

```text
archive_bytes <= floor(0.30 * 37,545,489 / 25) = 450,545
```

The older PFP16 frontier at `686,635` bytes already contributed about `0.4572`
rate points, and its non-rate score was about `0.5868`; that made it a
historical contest-grade baseline, not a Shannon-floor architecture. The newer
QZS3/QP1 r8 frontier is byte-scale-competitive at `276,426` bytes, but still
needs additional scorer-distortion reduction before any sub-`0.30` claim. The
floor path needs both rate discipline and component improvement, with every
stack measured as its own archive.

### Distortion sensitivity

Report local distortion slopes around the Grade A anchor:

```text
delta_score_seg = 100 * delta_seg_dist
delta_score_pose ~= (5 / sqrt(10 * pose_dist_anchor)) * delta_pose_dist
```

The derivative is a local diagnostic only. It cannot replace full archive scoring when distortion changes are not infinitesimal.

### Allocation objective for sensitivity-aware codecs

For β methods, state the proposed allocation objective as:

```text
minimize   sum_c sensitivity_c * quant_error_c(bits_c)
subject to sum_c payload_bytes(bits_c) + side_info_bytes <= budget_bytes
```

Where `sensitivity_c` must be measured on a calibration split and verified on a holdout split before it can guide a production archive.

Required sensitivity artifact fields:

| Field | Requirement |
|---|---|
| channel or block id | Stable across encode/decode builds |
| calibration split id | Exact frame/sample list |
| holdout split id | Exact frame/sample list |
| scorer target | SegNet, PoseNet, or combined score surrogate |
| gradient/Fisher/Hessian estimator | Formula and seed |
| normalization | Per-layer or global normalization rule |
| CV stability metric | Distance between train and holdout sensitivities |
| artifact hash | SHA-256 of saved sensitivity artifact |

---

## 4. Paper Outline

### Title candidates

Use a sober systems-paper title. Avoid claiming the Shannon floor was reached unless a Grade A++ archive proves it.

1. `Evidence-Gated Neural Archive Compression for a Two-Scorer Video Challenge`
2. `Toward the Shannon Floor: Artifact-Custodial Compression Under Segmentation and Pose Scorers`
3. `Score-Aware Rate Allocation for Neural Video Challenge Archives`

### Abstract blueprint

1. State the contest objective and artifact constraints.
2. State the current verified baseline/frontier only from the Grade A ledger.
3. State the methodological contribution: evidence-gated archive compression, sensitivity-aware allocation, and reproducibility-first scoring.
4. State that proposed α/β/γ components are promoted only after exact archive eval.
5. Include no ungraded score claims.

### Main sections

| Section | Purpose | Required tables/figures |
|---|---|---|
| 1. Problem and Objective | Define archive, inflate, scorers, and score formula | Score decomposition figure |
| 2. Evidence Standard | Explain Grade A/A++ and demotion policy | Evidence-grade ladder; artifact custody checklist |
| 3. Baseline and Verified Frontier | Present only Grade A ledger rows | Grade A score table; manifest table |
| 4. Rate-Distortion Decomposition | Show byte slopes and distortion slopes | Per-stream byte budget table; formula box |
| 5. α Mask Payload Hypotheses | Describe NeRV/VQ/wavelet/STC candidates as hypotheses | Planned ablation table, no score claims |
| 6. β Sensitivity-Aware Compression | Define sensitivity maps and allocation objective | Sensitivity artifact schema; CV stability table |
| 7. γ Joint Stack and Entropy Coding | Define stack readiness and MDL/ADMM role | Stack gate table; side-info accounting table |
| 8. Reproducibility | Exact artifact layout and commands | Reproduction manifest template |
| 9. Adversarial Review | Review gates and kill/promote rules | Review-pass ledger |
| 10. Limitations | CPU/MPS/proxy invalidity, hardware caveats, failed lanes | Failure taxonomy table |

---

## 5. Claim-To-Evidence Matrix

The paper should maintain a machine-checkable table with one row per claim.

| Claim id | Claim text | Grade | Evidence path | Formula / method | Review status | Allowed use |
|---|---|---|---|---|---|---|
| C-001 | Lane G v3 PFP16 was the 2026-04-30 historical A++ contest-grade frontier at recomputed `1.043987524793892`. | `A++` | `contest_grade_all_lane_results_audit_20260430_codex_progress.md`; `experiments/results/lane_g_v3_pfp16/pfp16_a_plus_plus_t4_20260430T1620Z_codex/contest_auth_eval.json` | Contest score formula, exact archive row, T4/equivalent provenance | A++ evidence landed; superseded by QZS3/QP1 r8 for current frontier wording | `historical_lesson` |
| C-009 | QZS3/QP1 r8 was the prior A++ frontier at `0.3159064496962538` on exact archive SHA `c68b7522d4d2c8a89771e491c5956b0fb3460e744b3adba9f410f053783044b1`; it is superseded by C-057 for current-frontier wording. | `A++` | `experiments/results/lightning_batch/exact_eval_line_search_qzs3_qp1_r8_t4_20260502T0110Z/contest_auth_eval.adjudicated.json`; `.omx/research/grand_council_stacking_full_pipeline_qzs3_pose_20260502_codex.md` | Contest score formula, exact archive bytes `276426`, T4 exact eval | A++ evidence landed; superseded by C-057 | `historical_lesson` |
| C-011 | C-057 QZS3/QP1 anisotropic basis continuation is the current A++ frontier at `0.3157562807844823` on exact archive SHA `63e6213ae154b5b5ce164829c15e675ad6d7819a9bdb1e8c9b2f099374fa7009`. | `A++` | `experiments/results/lightning_batch/exact_eval_line_search_qzs3_qp1_basis_r13_t4_20260502T0200Z/contest_auth_eval.adjudicated.json`; `.omx/research/shannon_floor_claim_matrix_20260430_codex.md` | Contest score formula, exact archive bytes `276423`, T4 exact eval | A++ evidence landed; active frontier unless superseded by later exact T4/equivalent archive | `rank_frontier` |
| C-012 | PR68/PR69/PR70 rule-boundary submissions are compliance lessons, not score evidence; PR70 explicitly moved score-affecting bytes into `inflate.py`, and PR69 has no filled maintainer eval report at inspection time. | `invalid` / `external_quarantine` | Public PR #68/#69/#70; `.omx/research/top_submission_reverse_engineering_canonical_repro_20260501_codex.md` | Payload-closure policy and external-source inspection | Must not enter ranked result tables unless strict charged-payload exact eval supersedes quarantine | `invalid_do_not_use` |
| C-013 | C-058 is a one-byte exact A++ successor to C-057 at `0.3157555307844823`; use C-058 for strict latest-frontier packets and C-057 for PR67 narrative comparison. | `A++` | `experiments/results/lightning_batch/exact_eval_line_search_qzs3_qp1_active_fix2_t4_20260502T0250Z/contest_auth_eval.adjudicated.json`; `.omx/research/shannon_floor_claim_matrix_20260430_codex.md` | Contest score formula, exact archive bytes `276422`, T4 exact eval | A++ evidence landed; active frontier unless superseded by later exact T4/equivalent archive | `rank_frontier` |
| C-010 | Sub-`0.30` likely requires PR65-style postprocess/side-channel atoms or an equivalent component-improving atom family, not only scalar QP1 pose line search. | `prediction` | `.omx/research/meta_lagrangian_scientific_rigor_thesis_review_20260502_codex.md`; public [PR #65](https://github.com/commaai/comma_video_compression_challenge/pull/65); public [PR #67](https://github.com/commaai/comma_video_compression_challenge/pull/67) | Rate/pose/seg decomposition and missing-score analysis | Must be falsified or confirmed by exact archive eval of charged atoms | `dispatch_priority` |
| C-002 | Saving 100,000 archive bytes changes the rate term by `0.06659`. | `derivation` | This blueprint; contest formula | `25 * 100000 / 37545489` | Math review required | `dispatch_priority` |
| C-003 | PFP16 improved Lane G v3 by about `0.004878` recomputed score after exact T4 archive eval. | `A++` | `contest_grade_all_lane_results_audit_20260430_codex_progress.md` | Full score recomputation; archive bytes `694,074 -> 686,635` | A++ evidence landed; source-bundle provenance review still required | `promote_candidate` |
| C-004 | Materially changed INR/NeRV-style mask payload variants remain α hypotheses, but Lane 12 `jsonfix40` is negative evidence for the current implementation. | `prediction` | `shannon_floor_execution_readiness_20260430.md`; Lane 12 exact negative report | Planned boundary/inflate/loss redesign plus exact score eval | Alpha review must address Lane 12 failure before rerun | `design_motivation` only |
| C-005 | Sensitivity maps are required before β allocation can promote a codec. | `derivation` | `external_research_intake_shannon_floor_20260430.md`; progress ledgers | Allocation objective and required CV stability | Beta review pending | `design_motivation` |
| C-006 | CPU/MPS/proxy scores cannot promote, kill, or rank lanes. | `derivation` | `contest_grade_all_lane_results_audit_20260430.md` | Evidence-grade standard | Audit review complete in source doc | `invalid_do_not_use` |
| C-007 | Lane 12 NeRV `jsonfix40` produced an exact CUDA regression at recomputed `26.03719330455429`; its prior byte-saving hypothesis is not score-valid. | `A-negative` | `contest_grade_all_lane_results_audit_20260430_codex_progress.md`; `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/contest_auth_eval.json` | Full score recomputation: PoseNet `49.77849960`, SegNet `0.03528685`, bytes `296,478` | Retire this implementation/config pending three-pass Grand Council scope review; do not generalize to all INR mask codecs | `historical_lesson` |
| C-008 | OWV3/Fisher Modal smoke produced a larger archive (`912,971` bytes, `+218,897` vs Lane G v3) and is suspicious negative smoke evidence only. | `empirical` | `experiments/results/lane_lane_g_v3_owv3_fisher_smoke_20260430_codex_modal/lane_g_v3_owv3_fisher_stack_results/build_provenance.json` | Rate-only derivation; no exact neural eval | Requires encoder/overhead/config review before any method conclusion | `design_motivation` |

Every future claim row must include an explicit `Allowed use` value from:

```text
rank_frontier
promote_candidate
kill_lane
dispatch_priority
design_motivation
historical_lesson
invalid_do_not_use
```

---

## 6. Reproducibility Artifact Pack

Every scored candidate must have a self-contained directory, for example:

```text
experiments/results/<lane_tag>/
  archive.zip
  contest_auth_eval.json
  provenance.json
  archive_manifest.txt
  archive_sha256.txt
  inflate_stdout.log
  inflate_stderr.log
  evaluate_stdout.log
  evaluate_stderr.log
  config.env.snapshot
  inflate.sh.snapshot
  git_status_short.txt
  build_command.txt
  eval_command.txt
  environment.txt
  review/
    pass_01_math.md
    pass_02_scorer_sensitivity.md
    pass_03_artifact_custody.md
```

### Required provenance fields

| Field | Required content |
|---|---|
| `lane_tag` | Unique experiment tag |
| `archive_path` | Local path to exact archive |
| `archive_sha256` | SHA-256 of exact archive |
| `archive_bytes` | Exact byte size |
| `manifest` | Archive members, sizes, CRCs if available |
| `build_command` | Full command and environment variables |
| `eval_command` | Full command and environment variables |
| `device` | CUDA device name; T4/equivalent flag for A++ |
| `n_samples` | Full eval sample count |
| `score_components` | seg, pose, bytes, reported score, recomputed score |
| `inflate_duration_seconds` | Needed for A++ budget evidence |
| `payload_closure` | Statement that all score-relevant payloads are inside archive or fixed contest code |
| `source_code_ref` | Git commit or explicit dirty-worktree diff manifest |

### Archive closure checks

| Check | Required result |
|---|---|
| No hidden files | No `.DS_Store`, `__MACOSX`, editor temp files, debug payloads |
| No sidecars | No score-relevant artifact loaded outside `archive.zip` |
| Inflate path | `archive.zip -> inflate.sh -> upstream/evaluate.py` |
| Raw output | Exact expected `.raw` cardinality and byte sizes |
| Scorer integrity | No upstream scorer modification |
| Manifest diff | Candidate manifest diffed against expected lane contract |

---

## 7. Ablation Tables

These are templates. Headline result and promotion ablation score cells require Grade A or A++ evidence. Lower-grade diagnostic rows may appear only when explicitly marked non-ranking and paired with a negative-result lesson.

### Verified frontier table

| Lane | Grade | Archive SHA | Bytes | Seg | Pose | Reported score | Recomputed score | Hardware caveat |
|---|---|---|---:|---:|---:|---:|---:|---|
| QZS3/QP1 C-058 active-subspace micro-frontier | `A++` | `5145fb57be574b85639856d239420ffa35e605e32664f93e06753b120b21633f` | 276,422 | 0.00061244 | 0.00049637 | 0.3157555307844823 | 0.3157555307844823 | Lightning AI Tesla T4; strict current frontier |
| QZS3/QP1 C-057 anisotropic basis | `A++` | `63e6213ae154b5b5ce164829c15e675ad6d7819a9bdb1e8c9b2f099374fa7009` | 276,423 | 0.00061244 | 0.00049637 | 0.3157562807844823 | 0.3157562807844823 | Lightning AI Tesla T4; PR67 comparison anchor |
| QZS3/QP1 line-search r8 | `A++` | `c68b7522d4d2c8a89771e491c5956b0fb3460e744b3adba9f410f053783044b1` | 276,426 | 0.00061244 | 0.00049846 | 0.3159064496962538 | 0.3159064496962538 | Lightning AI Tesla T4; historical frontier |
| Lane G v3 PFP16 | `A++` | `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f` | 686,635 | 0.00400656 | 0.00346442 | 1.04 | 1.043987524793892 | Historical A++ frontier superseded by QZS3/QP1 |
| Lane G v3 | `A` | `9b20...6870b` per Grade A ledger | 694,074 | 0.00400846 | 0.00345458 | 1.05 | 1.048866 | CUDA RTX 4090, not T4-matched |

### Component ablation table

| Component | Baseline archive SHA | Candidate archive SHA | Grade | Byte delta | Rate-term delta | Seg delta | Pose delta | Score delta | Promotion decision |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| PFP16 | Lane G v3 `9b20...6870b` | `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f` | `A++` | -7,439 | -0.004953 | approx -0.00000190 | approx +0.00000984 | approx -0.004878 | Current deployable frontier after paper/provenance bundle review |
| NeRV mask Lane 12 `jsonfix40` | Lane G v3/PFP16 depending on comparison | `864549cc648f0b3a023076c11812ccd0f10b1d013ed3fd6bb24d20bbcde85c97` | `A-negative` exact regression | -390,157 vs PFP16 bytes | -0.2598 vs PFP16 bytes | +0.03127855 vs PFP16 | +49.77503940 vs PFP16 | +24.9931 vs PFP16 | Do not promote; measured implementation/config retired only; alpha redesign remains live |
| OWV3/Fisher Modal smoke | Lane G v3 `9b20...6870b` | `710cba0c7c490b13db8b0aee897dd0f33cb8b66a6ed229466bf0d1aea392f5a3` | `empirical` suspicious negative | +218,897 | +0.1457545 | no exact eval | no exact eval | rate-only predicted `+0.1458` | Fail closed; diagnose encoder overhead/config before any rerun or method conclusion |
| IMP renderer | TBD | TBD | `empirical` | TBD | `25 * byte_delta / 37545489` | TBD | TBD | TBD | No decision before exact eval |
| Logit-margin mask training | TBD | TBD | `prediction` | TBD | `25 * byte_delta / 37545489` | TBD | TBD | TBD | No decision before exact eval |

### Sensitivity-map ablation table

| Artifact | Estimator | Calibration split | Holdout split | Stability metric | Protected fraction | Byte overhead | Exact archive grade | Decision |
|---|---|---|---|---:|---:|---:|---|---|
| Sensitivity map v1 | TBD | TBD | TBD | TBD | TBD | TBD | `empirical` | Cannot guide promotion before CV stability |

### Stack ablation table

| Stack | Components individually Grade A? | Stack archive SHA | Side-info bytes | Net byte delta | Seg delta | Pose delta | Recomputed score | Decision |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Materially redesigned INR-mask + PFP16 | No | TBD | TBD | TBD | TBD | TBD | TBD | Do not run as promotion stack until the redesigned mask component is measured |
| Materially redesigned INR-mask + OWV3 | No | TBD | TBD | TBD | TBD | TBD | TBD | Gate on measured α and β components |
| Full γ coordinator | No | TBD | TBD | TBD | TBD | TBD | TBD | Deferred until at least two measured components exist |

### Negative-results table

| Lane/result | Evidence grade | Failure class | Root cause | Repro artifact | Lesson allowed in paper |
|---|---|---|---|---|---|
| KL distill variants | `invalid` for future dispatch | Method/config failure per current instructions | Overweighted KL caused PoseNet collapse | See project ledgers | Do not use KL distill loss mode |
| Lane 12 NeRV `jsonfix40` | `A-negative` exact regression | implementation / representation failure for this archive, scope review pending | Byte savings were overwhelmed by PoseNet and SegNet collapse | `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/contest_auth_eval.json` | Byte-only α wins are invalid until exact archive score proves distortion survived |
| OWV3/Fisher Modal smoke | `empirical` suspicious negative | implementation-smoke / overhead regression | Top-3 Fisher/build-only path produced a larger archive; no exact eval | `experiments/results/lane_lane_g_v3_owv3_fisher_smoke_20260430_codex_modal/lane_g_v3_owv3_fisher_stack_results/build_provenance.json` | OWV3 needs encoder/overhead/config review before another promotion run |
| Ω-W-V2 stack | `B` diagnostic CUDA | scorer-sensitivity failure | Rate save was erased by PoseNet regression; exact archive not preserved locally | `experiments/results/lane_g_v3_omega_w_v2_stack_landed/contest_auth_eval.json` | Renderer codecs need PoseNet-sensitive channel protection |
| Lane GP v2/v3/v4 | `derivation` | approach killed for smooth-basis pose fit | Pose dims 1-5 are white-noise-like, not smooth | `lane_gp_class_forensic_audit_20260430.md`; all-scores audit | Do not spend on polynomial/spline/DCT low-K pose fits without new mathematical evidence |
| UNIWARD v8 | `invalid` | engineering bug | Computed payload was discarded; anchor mask copied into archive | all-scores forensic audit | Do not treat no-op archive copies as codec evidence |
| SegMapTrainer OOM cohort | `empirical` | engineering bug | Full rendered tensor materialization exceeded GPU memory | all-scores forensic audit; Round 6-10 adversarial docs | Bad runs are mostly engineering failures, not method falsifications |
| CPU/MPS/proxy scores | `invalid` for ranking | Methodology bug | Non-authoritative device/path | Contest-grade audit | Never rank/promote/kill |
| Byte-only lanes | `empirical` | Incomplete evidence | No exact CUDA score | Component reports | Use for dispatch priority only |

---

## 8. Adversarial Review Gates

No strategic promotion or kill enters the paper until three consecutive clean review passes are recorded.

### Gate sequence

| Gate | Reviewer stance | Required checks | Failure action |
|---|---|---|---|
| R1 Math / Shannon | Score arithmetic and rate-distortion logic | Recompute score, byte slope, distortion slope, side-info cost | Fix formula or demote claim |
| R2 Scorer sensitivity | SegNet/PoseNet failure modes | Boundary errors, PoseNet amplification, sensitivity-map CV stability | Add ablation or protect channel/region |
| R3 Artifact custody | Archive identity and compliance | SHA, manifest, payload closure, exact eval path | Rebuild/recover archive |
| R4 Engineering | Silent defaults and no-op defense | Dead flags, encode-discard, fallback paths, dependency closure | Patch and rerun |
| R5 Optimization / stack | Additivity and allocation validity | Component independence, KKT/waterline, side-info overhead | Run standalone component first |
| R6 Mitigation / leaderboard reverse-engineering | Rescue paths before scoped retirement | Stacking options, hybrid residuals, fallback routing, leaderboard-style stream allocation hypotheses, full-stack archive analogies | Record redesign options; no broad kill |

### Review ledger row

```text
review_id:
lane_tag:
archive_sha256:
claim_ids:
gate:
reviewer_stance:
status: pass | fail | blocked
findings:
required_rerun_or_patch:
next_review_due:
```

Promotion rule:

```text
promote_or_kill_allowed = Grade in {A, A++} and three_clean_reviews
retire_measured_implementation_allowed = Grade == A-negative and three_clean_reviews
                         and clean_review_passes >= 3
                         and no unresolved artifact-custody finding
```

---

## 9. Figures To Produce

| Figure | Inputs | Purpose | Evidence gate |
|---|---|---|---|
| Score decomposition bar | Grade A ledger | Show seg/pose/rate contribution of verified frontier | Grade A only |
| Evidence ladder | This blueprint + contest audit | Show why CPU/MPS/proxy are demoted | Policy |
| Archive manifest diagram | Exact archive manifest | Show payload closure | A/A++ candidate |
| Rate slope plot | Formula | Show byte leverage by stream | Derivation |
| Sensitivity heatmap | Sensitivity artifact | Show β allocation map | Empirical until exact eval |
| Ablation waterfall | Grade A/A++ ablation rows | Show measured component deltas | Grade A/A++ only |
| Failure taxonomy matrix | Forensic audit | Show what was bug vs method failure | No score ranking |

---

## 10. Paper-Quality Repro Commands

Commands in the writeup must be copy-pastable and tied to artifacts. Use placeholders only in the blueprint; replace with exact paths in the final paper.

```bash
# Build exact archive
.venv/bin/python <builder>.py \
  --input <inputs> \
  --output experiments/results/<lane_tag>/archive.zip \
  --provenance experiments/results/<lane_tag>/provenance.json

# Record hash and manifest
shasum -a 256 experiments/results/<lane_tag>/archive.zip \
  > experiments/results/<lane_tag>/archive_sha256.txt
zipinfo -l experiments/results/<lane_tag>/archive.zip \
  > experiments/results/<lane_tag>/archive_manifest.txt

# Exact contest-style eval
.venv/bin/python experiments/contest_auth_eval.py \
  --archive experiments/results/<lane_tag>/archive.zip \
  --out experiments/results/<lane_tag>/contest_auth_eval.json
```

The final paper must include the exact commands used for every reported Grade A/A++ row, not generic placeholders.

---

## 11. Limitations And Non-Claims

Required limitations section:

1. Grade A is not Grade A++: local CUDA score-grade evidence does not prove T4/equivalent contest identity or inflate-budget compliance.
2. External papers and leaderboard observations motivate designs only; they are not evidence for this contest archive.
3. CPU, MPS, proxy, stale, and byte-only outputs cannot rank, promote, or kill lanes.
4. Sensitivity maps can overfit calibration data; holdout stability is mandatory.
5. Stack deltas are not additive until standalone components are exact-evaled and then measured together.
6. Rate savings can be erased by SegNet/PoseNet distortion; every byte claim needs full component deltas before promotion.
7. No Shannon-floor attainment claim is allowed unless an exact Grade A++ archive demonstrates it.

---

## 12. Final Writeup Readiness Checklist

| Item | Done? | Blocking source |
|---|---|---|
| All headline score rows are Grade A/A++ only | No | `contest_grade_all_lane_results_audit_20260430.md` |
| Every claim has grade, artifact, formula, review status | No | Claim matrix in this blueprint |
| Every candidate has exact archive SHA and manifest | No | Repro artifact pack |
| All byte claims include rate-term arithmetic | No | Section 3 |
| All sensitivity claims include calibration/holdout evidence | No | Section 3 / Section 7 |
| All stack claims include standalone component grades | No | Stack ablation table |
| Three clean adversarial review passes for promoted claims | No | Review ledger |
| A++ wording restricted to A++ candidates | No | Contest-grade audit |

Until every item is resolved, the output is a blueprint or progress report, not a final contest-paper result.

---

## 13. Parallel Writeup Workstream Protocol

This document is the writeup workstream contract while implementation continues. It must be kept independent from lane code so paper rigor can advance without blocking builders.

### Writeup lanes

| Workstream | Can run now? | Inputs | Output | Stop condition |
|---|---:|---|---|---|
| Evidence ledger normalization | Yes | Contest-grade audit and codex progress ledgers | One canonical score/eval table | Any row lacks archive SHA, device, or recomputed score |
| Negative-result ledger | Yes | All-scores forensic audit; adversarial rounds 6-10 | Failure taxonomy and lessons | Any negative result is used to over-kill a broader class |
| Mathematical proof obligations | Yes | Score formula, rate/differential math, allocation objectives | Proof appendix checklist | Any formula is treated as measurement |
| Deterministic reproduction appendix | Yes | Exact artifacts and builders | Commands, manifest schema, environment capture | Any command uses placeholders in final paper |
| Compliance appendix | Yes | Submission rules, contest audit, archive manifests | A++ gate table | Any sidecar or scorer patch remains unresolved |
| Results narrative | Partially | Grade A/A++ rows only | Abstract/results section | Any B/empirical/prediction row enters the headline result |
| Future-work / hypothesis narrative | Yes | Grand Council and external intake | Clearly tagged alpha/beta/gamma hypotheses | Any prediction is phrased as achieved |

### Source update rule

When a new exact eval lands, update in this order:

1. `contest_grade_all_lane_results_audit_20260430.md` or a stricter successor.
2. The relevant `*_codex_progress.md` ledger.
3. This blueprint's claim matrix, frontier table, component ablation row, and negative-result ledger.
4. The paper draft only after the row is graded and review status is explicit.

The writeup author must never infer a frontier from `experiments/results/` alone. The frontier is whatever the contest-grade ledger says after artifact-custody review.

---

## 14. Mathematical Proof Obligations

These are the paper claims that require either formal proof, reproducible arithmetic, or a named limitation. Do not bury them as prose.

| Proof id | Claim needing proof | Current status | Required evidence/proof | Allowed paper wording before proof |
|---|---|---|---|---|
| M-001 | Contest score recomputation is exact for every result row. | Required for every row | Script or table recomputing `100*seg + sqrt(10*pose) + 25*bytes/37545489` from raw JSON fields | "recomputed under the contest formula" only after row-level check |
| M-002 | `100KB -> 0.06659` rate points. | Proven by formula | Arithmetic with denominator `37,545,489` | "rate-term arithmetic" |
| M-003 | Local PoseNet derivative around the frontier is `5/sqrt(10*pose_anchor)`. | Derivation only | Chain-rule derivation plus explicit local-only caveat | "local diagnostic slope" |
| M-004 | Sensitivity allocation objective is a valid proxy for score-preserving compression. | Unproven | Calibration/holdout CV stability plus standalone exact archive eval | "allocation hypothesis" |
| M-005 | Stack deltas can be composed. | Unproven | Standalone Grade A rows and stack Grade A row, with side-info accounting | "planned stack measurement" |
| M-006 | NeRV/INR mask representation preserves scorer-relevant geometry. | Falsified for Lane 12 `jsonfix40`; open for materially different variants | Boundary-local ablation, exact archive eval, no PoseNet/SegNet collapse | "future variant hypothesis" only |
| M-007 | OWV3 protects PoseNet-sensitive renderer channels. | Unproven | CUDA sensitivity artifact, holdout stability, exact archive eval vs OWV2 failure mode | "PoseNet-protection hypothesis" |
| M-008 | Grade A local CUDA result is contest-identical. | False in general; proven only for PFP16 after separate A++ T4 evidence | A++ T4/equivalent rerun, inflate-budget proof, manifest/payload closure | Never say contest-identical from Grade A alone |
| M-009 | A negative run kills a whole architectural family. | Usually unproven | Multiple independent implementations or a mathematical impossibility argument | Kill only the measured implementation unless proof is stronger |

### Formalizable lemmas

The following are small enough to make machine-checkable or script-checkable:

| Lemma | Suggested checker | Required output |
|---|---|---|
| Score formula recomputes each JSON row within tolerance | Python script reading `contest_auth_eval.json` | CSV of row id, reported, recomputed, delta |
| Rate delta is linear in archive bytes | Tiny unit test or appendix arithmetic | `delta_rate = 25 * delta_bytes / 37545489` |
| Payload closure equals archive manifest membership | Zip manifest checker | List of required members and unexpected members |
| A++ gate implies Grade A gate | Markdown proof or test over schema predicates | Predicate implication table |
| Sensitivity split disjointness | Python checker over frame ids | No overlap and stable seed/hash |

Do not cite the existing Lean adaptive-weight proof as evidence for adaptive rebalancing value. It proves a vacuous identity after `T^2` cancellation. It may be cited only as a historical proof-process note or for any still-valid quantization lemma, if that lemma is separately tied to a current lane.

---

## 15. Deterministic Reproducibility Appendix

Every paper result must be reproducible from an immutable artifact directory. The appendix should contain exact paths and commands, not prose descriptions.

### Directory contract

```text
experiments/results/<lane_tag>/<eval_id>/
  archive.zip
  contest_auth_eval.json
  provenance.json
  remote_provenance.json            # optional; never overrides contest_auth_eval.json
  archive_sha256.txt
  archive_manifest.txt
  archive_zipinfo_verbose.txt
  inflate_stdout.log
  inflate_stderr.log
  evaluate_stdout.log
  evaluate_stderr.log
  build_command.txt
  eval_command.txt
  python_version.txt
  pip_freeze.txt
  torch_cuda_environment.txt
  nvidia_smi.txt
  git_commit.txt
  git_status_short.txt
  dirty_diff_stat.txt
  scorer_integrity_sha256.txt
  payload_closure.md
  review/
    math_review.md
    scorer_sensitivity_review.md
    artifact_custody_review.md
```

### Deterministic archive rules

| Rule | Requirement |
|---|---|
| Entry ordering | Lexicographic or manifest-defined order, never Python set iteration |
| Entry timestamps | Fixed timestamp for every zip member |
| Permissions | Fixed member permissions |
| Compression mode | Recorded exactly, including level |
| Source mtimes | Not used for archive member metadata |
| Payload hashes | SHA-256 for every member and archive |
| Rebuild check | Rebuilding from same inputs produces identical archive SHA or documents why not |

### Reproduction command block template

```bash
# 1. Verify archive identity
shasum -a 256 experiments/results/<lane_tag>/<eval_id>/archive.zip
cat experiments/results/<lane_tag>/<eval_id>/archive_sha256.txt

# 2. Verify manifest
zipinfo -l experiments/results/<lane_tag>/<eval_id>/archive.zip \
  > /tmp/<lane_tag>_archive_manifest_check.txt
diff -u experiments/results/<lane_tag>/<eval_id>/archive_manifest.txt \
  /tmp/<lane_tag>_archive_manifest_check.txt

# 3. Re-run exact eval
PYTHONPATH=src:upstream .venv/bin/python experiments/contest_auth_eval.py \
  --archive experiments/results/<lane_tag>/<eval_id>/archive.zip \
  --out experiments/results/<lane_tag>/<eval_id>/rerun_contest_auth_eval.json

# 4. Recompute score from JSON components
PYTHONPATH=src .venv/bin/python scripts/adjudicate_contest_auth_eval.py \
  --eval-json experiments/results/<lane_tag>/<eval_id>/rerun_contest_auth_eval.json \
  --archive experiments/results/<lane_tag>/<eval_id>/archive.zip
```

The final paper must replace `<lane_tag>` and `<eval_id>` with exact values. Placeholders are allowed in this blueprint only.

---

## 16. Exact Artifact And Eval Table Schema

The final paper should use one table schema for all results. Avoid free-form result bullets.

### `paper_results.csv`

| Column | Type | Required | Description |
|---|---|---:|---|
| `row_id` | string | yes | Stable paper row id, e.g. `R-PFP16-20260430T1353Z` |
| `lane_tag` | string | yes | Human-readable lane name |
| `artifact_dir` | path | yes | Directory containing all reproducibility files |
| `archive_path` | path | yes | Exact archive path |
| `archive_sha256` | hex string | yes | SHA-256 of exact archive |
| `archive_bytes` | int | yes | Exact archive size |
| `manifest_status` | enum | yes | `clean`, `unexpected_members`, `missing_members`, `unknown` |
| `payload_closure_status` | enum | yes | `closed`, `sidecar`, `unknown` |
| `eval_json_path` | path | yes | Exact `contest_auth_eval.json` |
| `device_kind` | enum | yes | `cuda`, `t4_equivalent_cuda`, `cpu`, `mps`, `unknown` |
| `gpu_model` | string | yes | Exact model from provenance |
| `gpu_t4_match` | bool | yes | True only for A++ hardware requirement |
| `inflate_seconds` | float | yes for A++ | Runtime for inflate path |
| `n_samples` | int | yes | Must be 600 for Grade A/A++ |
| `seg_dist` | float | yes | Eval component |
| `pose_dist` | float | yes | Eval component |
| `rate_points` | float | yes | `25*archive_bytes/37545489` |
| `score_reported` | float | yes | Stored report value |
| `score_recomputed` | float | yes | Formula recomputation |
| `score_delta` | float | yes | Reported minus recomputed |
| `grade` | enum | yes | One of the evidence tags in Section 2 |
| `allowed_use` | enum | yes | One of the allowed-use values in Section 5 |
| `review_pass_count` | int | yes | Consecutive clean adversarial passes |
| `unresolved_findings` | int | yes | Must be 0 for promotion |
| `source_ledger` | path | yes | Ledger authorizing the grade |

### `paper_component_ablation.csv`

| Column | Type | Required | Description |
|---|---|---:|---|
| `component_id` | string | yes | Stable component id |
| `baseline_row_id` | string | yes | Row in `paper_results.csv` |
| `candidate_row_id` | string | yes | Row in `paper_results.csv` or `pending` |
| `component_type` | enum | yes | `pose`, `mask`, `renderer`, `entropy`, `stack`, `training_loss`, `harness` |
| `grade` | enum | yes | Evidence grade |
| `byte_delta` | int | yes if measured | Candidate bytes minus baseline bytes |
| `rate_delta` | float | yes if measured | `25*byte_delta/37545489` |
| `seg_delta` | float | yes if measured | Candidate seg minus baseline seg |
| `pose_delta` | float | yes if measured | Candidate pose minus baseline pose |
| `score_delta` | float | yes if measured | Candidate score minus baseline score |
| `side_info_bytes` | int | yes for codecs | Codec overhead inside archive |
| `promotion_decision` | enum | yes | `promote`, `hold`, `rerun`, `negative`, `invalid` |
| `decision_reason` | string | yes | One-sentence evidence-grounded reason |

### `paper_negative_results.csv`

| Column | Type | Required | Description |
|---|---|---:|---|
| `negative_id` | string | yes | Stable id |
| `lane_or_result` | string | yes | Measured lane/result |
| `evidence_grade` | enum | yes | Evidence tag |
| `failure_class` | enum | yes | `approach_killed`, `engineering_bug`, `config_bug`, `methodology_bug`, `legitimate_regression`, `indeterminate` |
| `scope_of_kill` | enum | yes | `this_run`, `this_implementation`, `this_lane_family`, `mathematical_class` |
| `root_cause` | string | yes | Short cause |
| `artifact_path` | path | yes if available | Exact report/log/archive |
| `can_rank` | bool | yes | Usually false unless Grade A/A++ |
| `can_promote_or_kill` | bool | yes | True only under evidence policy |
| `paper_lesson` | string | yes | Allowed wording |
| `revival_condition` | string | yes | What evidence would reopen the result |

### `paper_external_context.csv`

| Column | Type | Required | Description |
|---|---|---:|---|
| `external_id` | string | yes | Stable id, e.g. `EXT-PR67` |
| `source_url` | url | yes | Public PR, leaderboard, or upstream source |
| `reported_score` | string/float | yes if present | Public score signal; keep rounded/exact distinction |
| `archive_bytes` | int | yes if present | Publicly reported or locally measured external bytes |
| `component_values` | object | yes if present | Public PoseNet/SegNet fields when available |
| `evidence_tag` | enum | yes | `external`, `external_quarantine`, or `invalid` |
| `allowed_use` | enum | yes | `design_signal`, `target_comparison`, `compliance_lesson`, `invalid_do_not_use` |
| `local_reproduction_status` | enum | yes | `none`, `byte_anatomy`, `cuda_diagnostic`, `exact_promoted` |
| `quarantine_reason` | string | yes if quarantined | Why it cannot rank or promote |

---

## 17. Contest Compliance Appendix

This appendix is mandatory for any final submission candidate. It should be boring and mechanical.

### A++ compliance checklist

| Check | Required evidence | Current C-057 status |
|---|---|---|
| Exact archive preserved | `archive.zip` and SHA file | Required for final packet; C-057 SHA is recorded |
| Clean manifest | `archive_manifest.txt`, no hidden/debug files | Required final audit |
| Payload closure | No score-relevant sidecars | Required final audit |
| Inflate dispatch | Captured `inflate.sh` and `config.env` | Required final packet |
| Upstream scorer | `archive.zip -> inflate.sh -> upstream/evaluate.py` | Yes for A++ eval path |
| Full sample count | `n_samples=600` | Yes per C-057 exact row |
| CUDA | Provenance device CUDA | Yes |
| T4/equivalent | `gpu_t4_match=true` or official equivalent | Yes: Tesla T4 |
| Inflate budget | Under 1800 seconds on contest-equivalent host | Must be co-located in final C-057 packet |
| Score recomputation | Formula delta within tolerance | Required final adjudication row |
| Report custody | JSON, logs, manifest, provenance co-located | Must be consolidated before final submission/paper packet |
| Three clean reviews | Review ledger with no unresolved findings | Not complete |

### Compliance non-claims

Use these exact constraints in the final paper:

1. "Grade A" means local CUDA score-grade evidence, not contest-identity evidence.
2. "A++" is reserved for exact archive, manifest, payload closure, T4/equivalent hardware, inflate-budget, and upstream-scorer proof.
3. Neural artifacts must be inside `archive.zip` unless they are fixed contest code.
4. `remote_provenance.json` is never more authoritative than `contest_auth_eval.json`.
5. Any scorer modification, renderer shortcut, sidecar payload, or CPU/MPS/proxy eval demotes the result.

---

## 18. Current Evidence Gaps For The Paper

The next evidence needed before a strong final writeup:

| Gap | Needed artifact | Why it matters |
|---|---|---|
| C-058/C-057 final publication bundle | Exact archive copy/manifest, source or staged-tree manifest, structured timing note, review signoff | Converts the current strict frontier and PR67 comparison anchor into a paper/deploy packet |
| Lane 12 postmortem packet | Boundary/error visualizations plus payload/inflate contract review | Prevents overgeneralizing the `jsonfix40` failure |
| OWV3 sensitivity redesign truth point | CUDA Fisher/sensitivity artifact, CV stability, archive eval, and no size-regression guard trip | Tests the beta hypothesis after Modal smoke showed archive bloat |
| Active lane harvest | Lane 19, SA, H-V3, HM-S exact evals or failure logs | Updates negative/result ledger without memory-only claims |
| Deterministic rebuild proof | Rebuild C-057 archive and compare SHA, or document why byte-identical rebuild is impossible | Strengthens artifact custody |
| Review ledger | Three clean adversarial passes for any promoted result | Prevents paper claims from outrunning audit |
| Figure data files | CSVs generated from the schemas above | Makes plots reproducible rather than hand-entered |

Until these gaps close, the paper should be framed as an evidence-gated systems report with a current A++ C-058/C-057 QZS3/QP1 contest-grade frontier, not as a Shannon-floor attainment paper.

---

## 19. 2026-05-02 Meta-Lagrangian Thesis Review Addendum

The paper narrative now has a newer A++ frontier than the original 2026-04-30
blueprint. The current verified result to write around is C-057 QZS3/QP1
anisotropic basis continuation at `0.3157562807844823` on exact T4 archive SHA
`63e6213ae154b5b5ce164829c15e675ad6d7819a9bdb1e8c9b2f099374fa7009`.
Public source links for this addendum: the
[challenge repository](https://github.com/commaai/comma_video_compression_challenge),
[PR #67](https://github.com/commaai/comma_video_compression_challenge/pull/67),
and [PR #65](https://github.com/commaai/comma_video_compression_challenge/pull/65).

This changes the paper emphasis from "large renderer frontier plus future
rate allocation" to "public-floor archive basin plus charged atom allocation."
The core formalism should be:

```text
Score(A) = 100*Sg(A) + sqrt(10*P(A)) + 25*B(A)/37,545,489
lambda_rate = 25/37,545,489

utility(a | A, X) =
  E[component_score_saved(a | A, X)]
  - lambda_rate*charged_bytes(a | A, X)
  - beta*uncertainty(a | A, X)
  - gamma*interaction_risk(a | A, X)
  + eta*synergy(a | X)
```

The KKT/waterline language is allowed only as a planning relaxation. In the
actual archive problem, atoms are discrete and interact through decoder logic,
Brotli layout, PoseNet, and SegNet. Therefore every accepted water-fill point
must become its own deterministic archive and pass exact CUDA auth eval before
it enters result tables.

The thesis-advisor position is:

- Write C-058 as the strict current exact frontier and C-057 as the PR67
  comparison anchor.
- Write r8/C-056 and earlier r13 diagnostics as historical/superseded unless a
  later terminal exact T4/equivalent artifact supersedes C-057.
- Write public PR #67 and PR #65 as external design signals, not local proof.
- Write PR #68/#69/#70 as exploit or rule-boundary quarantine, not result
  evidence.
- Do not overstate sub-`0.30`. It likely needs PR65-style postprocess or
  side-channel atoms, or an equivalent component-improving atom family, not
  only scalar QP1 line search.
- Require deterministic archive custody for every claim: exact archive copy,
  SHA, bytes, manifest, provenance, eval command, CUDA device, sample count,
  logs, component recomputation, and review status.

The detailed study note is
`.omx/research/meta_lagrangian_scientific_rigor_thesis_review_20260502_codex.md`.
