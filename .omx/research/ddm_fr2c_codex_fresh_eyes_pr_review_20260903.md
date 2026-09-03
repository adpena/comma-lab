# ddm_fr2c — Codex fresh-eyes final PR review

Date: 2026-09-03  
Review object: `experiments/results/ddm_fr2_final_review_20260903/pr_body_snapshot_v3_FINAL.md`  
Verified object SHA-256: `7c7816a6aba02cc0659bb62b4d7da8970dc81f03c80f8ea3fc74740326bec58b`  
Repository/tree commit reviewed: `5c8a36526424e9538eaddbed643faea297ff8982`  
Mode: scorer-free, read-only review; no packet, PR, release, archive, pointer, or upstream mutation  
Tokens: `[no-triality] [p0-ledger-ok]`

## Verdict

**STOP before submission for one policy decision.** The technical packet and the body’s measured
core are internally consistent, but the live contest policy forbids having coding agents write all
submitted code. The repository’s own provenance says this packet was agent-authored end to end.
Operator-authored public prose cures the separate public-description clause, not the code-authorship
clause. No disclosure sentence can change that historical fact. Submit only after the operator can
truthfully establish compliance from evidence or obtains maintainer pre-clearance; otherwise use a
non-PR publication route.

If that external policy blocker is resolved, this is a narrow final edit, not a rewrite: apply the
five wording recommendations below. The score, archive identity, T4 row, compression boundary,
mechanism accounting, inherited lineage, independent-development dates, and the already cured
fr2 R1–R5 issues all verify.

## Ranked recommendations

| Rank | Grade | Conclusion and exact recommendation | Exact evidence / receipt |
|---:|---|---|---|
| 1 | **BLOCKER** | **The recorded code-authorship history conflicts with the live policy’s banned-use clause.** The policy allows assistants to work on parts, but bans using them to “write all of the code”; the project’s policy intake records the candidate as agent-authored end to end. The body’s disclosure does not establish that the submitter wrote and read most of the submitted code. There is **no text-only cure**. Disposition: stop; the operator must choose no contest PR, maintainer pre-clearance, or submission only if the operator can truthfully demonstrate compliance from the final history. | Live upstream README, `coding agents and LLMs policy`, verified 2026-09-03: <https://github.com/commaai/comma_video_compression_challenge#coding-agents-and-llms-policy>; `.omx/research/ddm_llm_policy_intake_20260831.md:13-18,23-44`; `.omx/research/ddm_pq13_pr_body_refresh_verdict_20260901.md:80-109,155-168,182-188`; body line 78. |
| 2 | **RECOMMEND** | **Two personal-ownership phrases exceed what this repository can verify and sharpen the policy hazard.** `UNVERIFIABLE-FROM-REPO`: “using my own MLX/Metal ports” and “this PR and the innovations involved are mine.” Replace the first with **“using this project’s MLX/Metal ports”**. Replace the final paragraph with exactly: **“I used coding agents (Claude as orchestrator of Codex subagents) extensively as research and engineering tools for the work behind this submission. The linked repository contains the resulting prompts, implementation history, experiment receipts, and provenance records.”** This improves factual precision but does not cure rank 1. | Body lines 43 and 78; `.omx/research/ddm_llm_policy_intake_20260831.md:25-30`; `.omx/research/ddm_pq13_pr_body_refresh_verdict_20260901.md:101-109,165-168`. The repository proves artifacts and provenance, not the body’s ultimate personal-ownership conclusion. |
| 3 | **RECOMMEND** | **The small fitted pose-layer bullet converts an advisory gate into an exact-score claim.** `UNVERIFIABLE-FROM-REPO`: “did not improve the score.” The measured result was a stratified n64 local CPU screen; no complete S or contest evaluation ran. Replace the bullet exactly with: **“Small fitted pose-correction layers: in a stratified n64 local CPU screen, the 43-byte, 247-byte, and 997-byte variants were modeled-positive but held-out pose-negative or neutral, so none passed the gate to compile or exact contest evaluation.”** | Body line 76; `.omx/research/ddm_pk4_optimal_form_frame0_pose_verdict_20260813.md:1-24` (`score_claim=false`, 43/247/997 B, held-out worse/zero/worse). |
| 4 | **RECOMMEND** | **“Equivalent quality” is an untested forecast in both public surfaces.** `UNVERIFIABLE-FROM-REPO`: no raw-video-to-terminal full-pipeline run exists. Replace the body’s second TODO bullet exactly with: **“A full-pipeline mode that starts from the raw video, runs the credited PR #130/#135 training scripts, then applies the solve and packaging stages. Because neural training is not bit-for-bit reproducible across GPUs, the resulting archive’s score and quality would need fresh measurement.”** Apply the same wording to README lines 62–65 so the two public surfaces remain consistent. | Body line 48; `submissions/semantic_joint_ctxmix/README.md:60-66`; `.omx/research/ddm_g8s_single_run_reproof_20260903.md:3-18,26-38` proves only the pinned-base five-stage lossless replay. |
| 5 | **RECOMMEND** | **The entropy-coder bullet overstates the current-object receipt.** `UNVERIFIABLE-FROM-REPO` for the exact AFR1 token stream: its 113,411-byte context-conditioned stream was not re-raced against the stated entropy bound after the last two context changes. Replace the bullet exactly with: **“Swapping or re-tuning the entropy coder: on the measured shipped-family bodies, 25 Brotli/LZMA configurations found no saving on the frozen model sections, and generic recoding added 5 bytes on the then-current token stream. These fixed-probability coder swaps lost; changing the context model is a different axis.”** | Body line 73; `.omx/research/ddm_jt23_coder_collection_compose_verdict_20260826.md:25-72`; `.omx/research/ddm_afc1_address_free_census_20260831.md:24-41,74-79,96-106` identifies the later 113,411-byte AFR1 stream without a fresh generic-coder/entropy-bound race. |
| 6 | **RECOMMEND** | **The token-reordering bullet universalizes a formulation result.** The receipt proves a full n600 seeded permutation inside the shipped fixed model’s only admissible within-group reorder class; cross-group reorder is a different trained model. Replace the bullet exactly with: **“Reordering the token stream before coding: within the shipped fixed model’s only lossless reorder class, a full-n600 seeded within-group permutation changed the stream bytes but not its length (113,777 bytes to 113,777 bytes). Cross-group ordering would require training a different context model.”** | Body line 74; `.omx/research/ddm_rr9_reorder_refit_20260824.md:1-32,109-160`; `.omx/research/ddm_gf1_generator_form_capacity_verdict_20260830.md:36-65` shows the opposite sign on a generic-coder stream, so “only helps coders that have no context model” is not a general theorem. |
| 7 | **VERIFIED-OK** | **The authority row, arithmetic, archive identity, and timing are sound.** Recomputed from printed scorer components plus exact bytes: `0.020139 + 0.007981227975693965 + 0.11985594327989708 = 0.14797617125559104`; the Decimal recompute differs only beyond the binary-float display. The retained archive is 180,002 B, SHA-256 `cbb8d928…d405bf25`, with one stored 179,902-byte member and exactly 100 B ZIP overhead. The current entrypoint produced the same row on Tesla T4, Linux x86_64, CUDA, n600, evaluator threads 2, in 532.3 s inflation plus 40.6 s evaluation. | Body lines 12-37,53; `experiments/results/ddm_g8v1_gen8_tree_cuda_reproof_20260903/t4_row/MODAL_REMOTE_RESULT.json:1-84`; `.omx/research/ddm_g8v1_gen8_tree_cuda_reproof_20260903.md:3-23`; retained archive `/Volumes/VertigoDataTier/pact/ddm_g8s_single_run_reproof/store_v2/retained/archive.zip`. |
| 8 | **VERIFIED-OK** | **The generation-8 packet and compression boundary are accurately described.** All 39 manifest rows hash clean; `compress.py` rebuilt the exact terminal archive from pinned base `df7fd266…e2080` in one retained run (4,140.9 s), and it refuses without/mismatching the base. The apparent six receipt rows contain a sibling GB1 pointer control; the terminal chain has five sequential lossless transformations and removes 454 B from 180,456 B to 180,002 B. Current README and body agree about CUDA inflation, CPU timeout/no CPU score, local Apple-silicon solve, pinned-base replay, and post-submission TODOs. | `submissions/semantic_joint_ctxmix/MANIFEST.sha256` (39/39 passed via `shasum -c`); `submissions/semantic_joint_ctxmix/README.md:34-86`; `.omx/research/ddm_g8s_single_run_reproof_20260903.md:3-38`; `/Volumes/VertigoDataTier/pact/ddm_g8s_single_run_reproof/store_v2/retained/run_1/RESULT.json`. |
| 9 | **VERIFIED-OK** | **The competitive, inherited, innovation, and independent-concurrency claims have named receipts.** The same-axis public bar is PR #135 at 0.162 (author-reported unrounded 0.16226842169958583). The current lineage is PR130 → PR133 refinement → PR135. Joint admission is 455/573 proposed edits; the carrier is re-solved on the admitted candidate’s own renders; in-compile compensation and the zero-byte shipping-object pose re-solve are real; five later lossless moves total −454 B with output identity. Git proves the stored pose-target commit on 2026-04-11 and boundary-math direct-partition commit on 2026-06-10. | Body lines 53-64; `.omx/research/ddm_pi135_pr135_intake_20260810.md`; `.omx/research/ddm_jg5_pose_resolve_on_edited_renders_20260819.md:150-184,220-249`; `.omx/research/ddm_qs5_resolve_compensation_20260813.md`; `.omx/research/ddm_up2_shipping_object_pose_solve_20260819.md`; `.omx/research/ddm_pq1_submission_packet_prep_20260815/GENERATION_LOG.md`; commits `fea4a953f9f4b9fa9147cad7cb253bb3f2824a48` and `752a30cdb9732eed5d48a91d1760361ca2eaf51d`. |
| 10 | **VERIFIED-OK** | **The remaining four unsuccessful-family summaries are supportable at the body’s stated scope.** Learned-tensor quantization was worse across the tested ladder; the measured ep60 small renderer’s pose cost exceeded its rate credit by tens of times; the best current parametric Lane carriage costs 36,044 B against a 21,699 B bar (1.661×); and explicit per-site correction/address forms lose while the shipped address-free conditioning is real. These are scoped empirical summaries, not global family theorems. | Body lines 70-72,75; `.omx/research/ddm_wd2_ep60_advisory_refusal_verdict_20260815.md:3-49`; `.omx/research/ddm_afr1_pointer_move_and_no_toy_erratum_20260831.md:54-82`; `.omx/research/ddm_ad2_addressing_cost_decomposition_20260822.md:47-120,122-171`; `.omx/research/ddm_afc1_address_free_census_20260831.md:5-22,112-127`. |

## Known fr2 cures verified, not refiled

- The reviewed v3 SHA and HEAD match the respawn amendment exactly.
- The old “solved across 573 pairs” wording is gone; v3 says proposals are per pair and 455 of
  573 proposed edits were admitted.
- The old campaign-count framing is gone; v3 does not use “twenty-three.”
- The lineage now says PR #135 builds on PR #133, which refined PR #130’s vehicle; the packet
  README says the same thing.
- The pinned base archive and refusal-without-it are explicit in both body and README.
- The research commit `1c9fbbf58716eb0f26bcdf2a91e3c89d0e4efdde` is on `origin/main`; the packet
  README records public verification on 2026-09-03.
- “Attached” is intentionally realized by the fork release link and is not a defect under the
  operator amendment.

## NO-FAKE and skeptical-reader result

No payload is hidden in free code, no advisory/MPS number is presented as a contest score, and the
current public runtime has a direct current-tree T4 receipt. The body clearly credits the learned
vehicle and claims only the decision/lossless layer. A skeptical competitor can reproduce the
pinned lossless chain if given the pinned base, but cannot reproduce training from raw video today;
that is disclosed. The five wording recommendations above remove the places where prose currently
outruns the measured scope. Rank 1 is different: it is an eligibility fact, not a technical wording
defect.

## RECALL EVIDENCE

Recall was run before adjudication across `.omx/research/`, all canonical research indexes, the
sub-0.15 DAG, `main_hot_state.md`, canonical task status, docs, the packet, and the full canonical
equation registry (457 rows from `.venv/bin/python tools/list_canonical_equations.py --json`). Content
queries included `semantic_joint_ctxmix`, the full archive SHA, `0.14797617125559104`, `455 of 573`,
`coding agents policy`, `Yousfi`, `PR #130/#133/#135/#138`, `quantization`, `smaller network`,
`parametric lane`, `entropy bound`, `reorder`, `address correction`, `small fitted pose`, the two
independent-development dates, and the evaluated commit.

Beyond the charter seeds, recall changed the review in four ways:

1. The live README policy, the 2026-08-31 enforcement intake, and pq13’s policy adjudication exposed
   the one submission blocker. This is not among v3’s cured prose rows.
2. PK4 showed that the small-pose-layer claim is an n64 local gate, not an exact-score result; this
   created rank 3.
3. AFC1 showed that the exact AFR1 token stream postdates JT23’s 25-coder race; this narrowed the
   current-object entropy wording in rank 5.
4. RR9 plus GF1 showed that reordering is zero-byte only inside the shipped fixed-model class and
   can save bytes on a generic-coder object; this created rank 6’s exact scope.

No additional contradiction was found in the canonical equations, index, DAG, task, or docs
surfaces. The packet’s score law remains `100*d_seg + sqrt(10*d_pose) + 25*bytes/37,545,489`.
The exact current archive, current T4 report, and current public packet all agree.

Fresh-eyes sequencing boundary: the prior Opus memo was not opened before this table was frozen.
One broad recall command accidentally printed isolated matching lines from that embargoed path
(an old R2 excerpt and a score-verification excerpt); no surrounding analysis or prior ranking was
read or adopted. This is recorded because “fresh eyes” should not be overstated. The complete prior
memo is read only after the independent table above exists, and the cross-model diff is appended
below.

## Measurement and frontier boundary

Measured in this review: file/tree/archive hashes and sizes, ZIP member anatomy, manifest 39/39,
Git ancestry/dates, exact score arithmetic, retained compression receipts, and receipt-to-prose
consistency. Consumed authority: the direct current-tree `[contest-CUDA T4 n600]` evaluator receipt.
Not measured: any new SegNet/PoseNet output, CPU score, training rerun, public fetchback, or policy
exception. Boundary: review-only; no score or pointer movement.

Own-vehicle frontier remains AFR1: **S `0.14797617125559104` @ `180,002 B`
`[contest-CUDA T4 n600]`**, archive SHA-256
`cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`; unchanged by this review.

## Cross-model diff

The prior memo was reviewed only after the independent table above was persisted. It reviewed an
earlier body (`pr_body_snapshot.md`) and tree (`dba0d7a951`), so its cured findings are not defects
in the v3/`5c8a365264` object.

| Topic | Opus v1 | Codex fr2c | Adjudication |
|---|---|---|---|
| Old R1: README denied current-tree T4 proof | BLOCKER | Verified cured | **Agree on the old defect; v3 tree fixes it.** Current README lines 78–82 now carries the direct g8v1 row. Do not refile. |
| Old R2: “solved across 573 pairs” | BLOCKER | Verified cured | **Agree on the old defect; v3 body line 59 now uses “proposed per pair” and “455 of 573 proposed edits.”** Do not refile. |
| Old R3: unexplained “twenty-three” | RECOMMEND | Verified cured | **Agree.** V3 removes the number and keeps the receipted five-post-freeze/454-byte statement. |
| Old R4: missing base-archive boundary | RECOMMEND | Verified cured | **Agree.** V3 body line 41 and README lines 45–49 now give refusal behavior, full base SHA/bytes, and availability. |
| Old R5: public evidence commit unverified | RECOMMEND | Verified cured | **Agree.** Commit `1c9fbbf5…` is on `origin/main`; README lines 74–76 records the 2026-09-03 check. |
| Old R6–R9: arithmetic, PR135 label, priority dates, five-stage/−454 B identity | VERIFIED-OK | VERIFIED-OK ranks 7–9 | **Full agreement.** Fr2c additionally binds the claims to the current-tree T4 and g8s receipts. |
| Old R10: punctuation, lineage grammar, ambiguous dates | NIT | Verified cured | **Agree.** V3 has one closing parenthesis, singular “PR #135 … builds,” and ISO dates. |
| Coding-agent policy | Opus said the direct disclosure was the right posture and made no recommendation | BLOCKER rank 1 plus wording rank 2 | **Disagree.** Disclosure is good evidence hygiene but not eligibility. The live policy bans agent-written all-code submissions, and the project’s own intake records end-to-end agent authorship. This is the only stop condition. |
| Untested/scoped prose | Not raised | Ranks 3–6 | **Codex-only additions.** “Equivalent quality,” exact-score language for PK4, current-AFR1 entropy-bound wording, and the universal reorder statement each outrun the named receipt; all have narrow replacement text. |

Net: the models agree on every measured core claim and on every old defect after accounting for
snapshot drift. Codex adds one policy blocker and four evidence-scope corrections plus one
ownership-wording correction. No Opus-only live defect survives in v3.

## NEXT_IF_RESUMED

- **BLOCKED-ON-OPERATOR-POLICY-DECISION** — owner: operator; consumer store: task #1363 policy
  ruling consumed by #1111/final PR gate; fire trigger: the operator audits the final submitted-code
  history and either demonstrates compliance with the live all-code clause, obtains maintainer
  pre-clearance, or chooses the no-PR route.
- **QUEUED-AFTER-POLICY-CLEARANCE** — owner: operator final-text editor; consumer store: the
  operator-owned PR body plus `submissions/semantic_joint_ctxmix/README.md`; fire trigger: rank 1 is
  resolved, then independently apply or reject ranks 2–6 while keeping body/README wording aligned.
- **QUEUED-AFTER-POLICY-AND-TEXT-CLEARANCE** — owner: MAIN/operator publication gate; consumer
  store: hosted-archive fetchback receipt and PR #1111 publication receipt; fire trigger: policy is
  cleared, final text is operator-approved, the release asset fetches back as 180,002 B with full
  SHA-256 `cbb8d928…d405bf25`, and the operator gives explicit publish authority.

## LIVE-HYPOTHESES

- Maintainer pre-clearance may admit a fully disclosed research submission even though the literal
  all-code clause appears to bar it; this is plausible only because the challenge remains open for
  “fun” as well as hiring, and it is untested until the maintainer answers before publication.
- A raw-video full-pipeline rerun may land near the frozen AFR1 quality despite different bytes;
  the deterministic lossless tail is proven, but the neural training prefix is hardware-sensitive,
  so only a fresh terminal evaluator row can establish the result.
- The final AFR1 113,411-byte token stream may remain at the fixed-probability coder floor after its
  last context changes; predecessor races and the unchanged RC64 construction make this plausible,
  but an exact-current raw-stream race is absent and is unnecessary for this submission.

## DEAD-ENDS

- Disclosure-only policy cure: closed. The live policy contains a categorical banned-use clause;
  honest prose cannot retroactively change code authorship.
- Refiling fr2 R1–R5 or the old lineage grammar: closed. V3 and commit `5c8a365264` contain the
  requested cures, and the packet agrees.
- Treating the 0.15 display as the exact score: closed. The exact component/byte recompute is
  0.14797617125559104 and the evaluator explicitly prints only two decimals.
- Treating PK4’s small pose layers as an exact-score negative: closed. It is a local n64 held-out
  gate with no contest evaluation.
- Treating RR9 as a theorem about arbitrary reordering/context models: closed. It measures the
  shipped fixed-model within-group class; cross-group order changes the trained model.
- Treating the prior JT23 token coder race as a fresh exact-AFR1 race: closed. AFR1’s later context
  stages changed the token stream to 113,411 B, so only a predecessor/family-scoped statement is
  receipted.
