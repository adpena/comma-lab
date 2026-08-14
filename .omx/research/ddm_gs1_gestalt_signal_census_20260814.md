# DDM GS1 gestalt signal census — 2026-08-14

## Verdict

No newly found mechanism changes the live routing. The census found four
signal-bearing findings in the relevance-enriched top 30, but they are the
already-live EC2 oriented adapter, JS1 promoted Stage-0 decomposition, MT1 T4
sign gate, and MC36 candidate C. The author sweep produced one lesson-only
transfer and no adopt/race row. PR136 is an adaptive order-0 histogram coder,
not a new context model, and its mechanism is already covered by our measured
same-state coder closure. The correct action is therefore to preserve the
existing fire order, not create another branch.

This arm was scorer-free. It ran no Metal, MPS, Modal, paid job, renderer,
training, `upstream/evaluate.py`, or full-n600 scorer forward. It materialized
no candidate payload. All source and corpus work was read-only. Counts below
are `[macOS-CPU filesystem census; scorer-free]`; public-author rows are
`[public-source inspection]`; score effects are explicitly derived from
already-measured components.

The live anchors used here supersede the stale frontier line in the common
contract:

- effective frontier: CP135, `S=0.16195513827824176`, `186,252 B`,
  `[contest-CUDA T4, n600]`, archive SHA-256
  `6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`;
- own-vehicle frontier: LC2, `S=0.16959899569230852`, `187,226 B`,
  `[contest-CUDA T4, n600]`;
- CP135 component anchor: `34,970` Seg flips of `117,964,800`,
  `d_pose=6.885642960696714e-6`, `186,252 B`, exact rate contribution
  `0.124017561736910658`.

## LEG A — newest-author public-profile sweep

### Source boundary and method

I applied the #414 method to the six newly named authors: exact-handle searches
over GitHub profiles/repositories and gists, project/package indexes, papers,
blogs, and the public PR pages. I ignored username collisions unless a source
linked back to the exact handle. Direct public sources included the
[codexblack ExperimentBook](https://github.com/codexblack/CommaVideoCompressionChallenge_ExperimentBook),
[PR135](https://github.com/commaai/comma_video_compression_challenge/pull/135),
[PR133](https://github.com/commaai/comma_video_compression_challenge/pull/133),
[JPL11's evlab package](https://pypi.org/project/evlab/0.2.0a1/),
[VeigaPunk's GitHub profile](https://github.com/VeigaPunk), and
[VeigaPunk's Hugging Face activity](https://huggingface.co/VeigaPunk/activity/all).
The searchable public web did not expose a uniquely attributable gist,
paper/blog, or additional relevant repository for JasonMo123, bzlvkv, or
Tibo-vagenBird in this bounded search. That is a scoped absence, not a claim
that no such source exists.

Ryan Li (`ryanli0070`) is explicitly covered by
`.omx/research/pr_authors_github_intake_20260710.md` (#413): six repositories
were inspected and its contest-margin-surrogate lesson was banked. Per the
charter, I did not resweep him.

### Ranked transfer rows

| rank | author / public evidence | typed disposition | transfer and named consumer |
|---:|---|---|---|
| 1 | `JPL11` / `evlab`: event streams, voxel grids, time surfaces, denoising, corruptions, and reproducible conversion/benchmark recipes | **LESSON-ONLY** | The event-support and corruption protocol is a useful diagnostic pattern for `MICRO35/mc36`: stratify sparse edits by temporal support and test deletion/jitter sensitivity before authority fire. It supplies no contest receiver, compression representation, byte price, or score mechanism. **FOLDED** as test-design guidance; MC36 already has the stronger built/parsed candidate. |
| 2 | `codexblack` / ExperimentBook plus PR135 | **NOTHING-FOUND** | PI135 and FD135 already read all 231 repository files and decomposed the 48-markdown workbook. I found no additional public mechanism beyond that consumed corpus. Named consumer: `#984`; action: none beyond the already-composed CP135/EC2 campaign. |
| 3 | `JasonMo123` / PR133 and exact-handle public search | **NOTHING-FOUND** | The three-basis `5→4` bit change and eight full-600 coordinate passes were already source-inspected and folded by PI136/PI135. Named consumer: `#984`; no duplicate build or race. |
| 4 | `bzlvkv` / PR134 and exact-handle public search | **NOTHING-FOUND** | No uniquely attributable off-PR compression source was found in the searched public surfaces. PR134's AV1-plus-correction mechanism is dominated as a candidate and its exact-grid/layered-selection lesson is already consumed by RVS1/PI136. Named consumer: `#984`; no fire. |
| 5 | `VeigaPunk` / public GitHub and Hugging Face profiles | **NOTHING-FOUND** | The visible inventory is mainly agent/orchestration, Rust/tooling, and openpilot-adjacent work; the Hugging Face harness repository has no model card establishing a transferable contest mechanism. Named consumer: `pz4 QAT`; no rate-aware QAT or receiver row follows from this profile. |
| 6 | `Tibo-vagenBird` / exact-handle public search | **NOTHING-FOUND** | No uniquely attributable repository, gist, paper, or blog relevant to `#978` or the contest receiver was found in the searched surfaces. A similarly themed Brno thesis result belonged to another author and was excluded. Named consumer: `#978`; no fire. |

There is no **ADOPT-NOW** or **RACE** row. The one lesson-only row changes how
one might stress-test sparse events; it does not change the MC36 candidate,
byte gate, scorer order, or consumer store.

## LEG B — PR136 mechanism verification

### Source receipt

The retained, commit-pinned PR136 source is head
`95d1b49b21c4d0a596bcd47c6ca2edd8c15b5b48` under
`/Volumes/VertigoDataTier/pact/pr_breadth_intake_20260810/pr136/source_snapshot/`.
The two decisive files are:

- `submissions/hnerv_rc/src/codec_rc.py`: `06dd2ffba5a3c6a693e803a76ed1babfc92bba2f2d0acae0d0144fa070f6a94f`;
- `submissions/hnerv_rc/README.md`: `aa156f00374f961c0b0409290e5d47b2c276837e6948120bbeb149d8b9291e4c`.

The actual coder initializes a 256-symbol uniform count vector separately for
each tensor stream, turns the current counts into a `float64` categorical
distribution for each next symbol, range-codes it through `constriction`, and
increments only the observed symbol by `8`. The decoder repeats the same
updates. Latent low and high byte planes use the same procedure. The range
queue spans tensors, but the probability counts reset per tensor. The README's
phrase “integer count tables” is not literal code behavior: the implementation
uses `float64` counts/probabilities with `perfect=False`. I did not independently
test its cross-host portability claim.

This is an adaptive order-0 empirical histogram. It can exploit changing
single-symbol marginals within a tensor. It cannot condition on an LZ match,
neighbor, temporal state, previous-symbol context, class edge, or decoded
semantic state. It is therefore not a new match-structure or multi-token
probability mechanism.

### Comparison against both closure legs

| required comparison | retained evidence | PR136 result |
|---|---|---|
| memoryless bound | `.omx/research/ddm_pr130_reproduce_20260809/SEMANTIC_SECTION_NO_MEMORYLESS_SLACK.md` at serializer commit `fec6dae38b`: token ANS `114,860 B` versus explicit model cross-entropy `114,852 B` (`+8 B`); semantic Brotli `35,033 B` versus H0 `36,805 B` (`-1,772 B`); pose same-state recode `+4 B` worse; HPAC `14,962 B` versus H0 `16,567 B` (`-1,605 B`) | PR136 only approaches an evolving order-0 marginal. Current semantic and HPAC sections already beat their H0 bounds through match structure; tokens are within 8 B of their explicit model; pose has no same-state win. No section exposes a PR136-shaped gap. |
| real context/model races | #918 explicit-token coder closure; `.omx/research/ddm_sv2_smevr_base_rule_race_20260803.md` (`+5,183 B` for SMEVR on the tested IX2 object, with the law that rank is not match structure); `.omx/research/ddm_hp1_20260806/RECEIPT.md` (learned prior plus counted model `456,166 B` versus shipped `341,296 B`, `+114,870 B`, on the tested <=10K static-context family) | PR136 adds neither stronger match structure nor an amortizable new context model. It does not reopen those instance/formulation closures. |

**PR136 disposition: CONFIRMED / CONSUMED-VIA-MEMO.** PI136's one-line
adjudication is correct at the mechanism level. PR136 is the PR95 HNeRV vehicle
plus longer training and a PR112-style adaptive order-0 range coder. The
reported source row is `S=0.19258426607726234`, `177,998 B`,
`[external author-reported CPU, n600]`; this arm did not measure that score and
does not have the exact archive. No implementation, transfer race, rebuild, or
scorer fire is justified.

The only remaining PR136 action is evidence custody, not mechanism discovery:
if the exact public binary becomes reachable, retain it, hash it, parse it, and
resolve the README's roughly `1.1 KB` statement against `codec_rc.py`'s
`1,501 B` statement. That action cannot move the routing without new exact
wire evidence.

## LEG C — own-corpus residue triage

### Denominators and sampling

The charter's frozen surfaces are `1,354` un-routed top-level `ddm_*`
artifacts and `4,906` dated markdown documents containing a negative token but
no literal `verdict_scope`. These are raw discovery surfaces, not additive
finding counts: a markdown file can belong to both. The `4,906` count comes
from `.omx/research/ddm_na2_negative_audit_20260803.md`; NA4 supplied the
separate measured warning that prefix selection is not a stable population
proxy. At final census time, concurrent repository growth made the live counts
`1,366` total vehicle artifacts, `10` harvested/routed, `1,356` unharvested,
and `5,187` live untagged negative-bearing dated markdown files under the same
NA2 text rule. I preserve the charter's frozen denominator for the promised
answer and report the live snapshot only as drift, not as a silent replacement.

I formed a deduplicated finding pool from the two surfaces, excluding machine
payloads and treating a top-level directory as one artifact. Within each live
consumer stratum (`JS1`, `#978/MT1`, `MICRO35/MC36`, `#984`, `pz4 QAT`), I
ranked parseable-date recency together with route-specific content relevance,
removed false-hit mentions after reading, and selected six distinct findings.
This produced 30 relevance-enriched findings, never a prefix. For a check on
ranking bias, seed `ddm_gs1_20260814_route_stratified_v1` selected two further
positive-relevance findings per stratum from below the cutoff, ten total.

### Top 30 finding-level grades

`SIGNAL-BEARING` means the source still has an unconsumed, typed action. It
does not mean the live board missed it. `CONSUMED-VIA-MEMO` names the later
memo that already owns the finding. `INERT` means no current-route action
survived.

| # | stratum | finding source | grade | consumer and fire order / consuming memo |
|---:|---|---|---|---|
| 1 | JS1 | `ddm_gca1_graph_calculus_crosswalk_20260813.md` | **CONSUMED-VIA-MEMO** | Its oriented class-interface route is implemented/sealed by EC1 then EC2; its graph-energy and heat-kernel side rows remain folded/conditional. |
| 2 | JS1 | `ddm_ec2_oriented_adapter_trainer_20260814.md` | **SIGNAL-BEARING** | Consumer: `#984` through EC2's retained `main_cuda/` store. Fire after MAIN proves the sole scorer lane free, verifies the sealed hashes, and runs oriented first; controls fire only after an oriented break-even win. |
| 3 | JS1 | `ddm_ec1_implicit_edge_conditioning_20260814.md` | **CONSUMED-VIA-MEMO** | EC2 closes EC1's missing true-CUDA trainer and owns the next fire. |
| 4 | JS1 | `ddm_js1_stage0_per_edge_20260812.md` | **SIGNAL-BEARING** | Consumer: JS1 joint line at `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/`. Fire only on a promoted exact-base T4 axis with retained full-n600 fields; the Mac field is instance-blocked by axis mismatch. |
| 5 | JS1 | `ddm_hr1_realization_engineering_20260811.md` | **CONSUMED-VIA-MEMO** | CN4, RHO1, EC1, EC2, and RFO1 narrowed its broad realization design into typed current routes. |
| 6 | JS1 | `ddm_cn4_arc_consolidation_20260811.md` | **CONSUMED-VIA-MEMO** | Its seven-memo dispositions and lossless closure are folded into RFO1 and the current JS1/#984 queues. |
| 7 | #978/MT1 | `ddm_mt1_978_multitoken_screen_20260814.md` | **SIGNAL-BEARING** | Consumer: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/multitoken_978/ddm_mt1_20260814/optimal_form_r2/retained/t4_sign_gate_r1/`. Fire the sealed T4 sign gate only when the #978 scorer lane is free; joint training is contingent on `positive_t4_sign=true`. |
| 8 | #978/MT1 | `ddm_bg2_postmortem_execute_20260814.md` | **CONSUMED-VIA-MEMO** | Its bilinear formulation is folded; its named multi-token successor is the executed MT1 screen. |
| 9 | #978/MT1 | `ddm_hv1_fresh_eyes_hybrid_review_20260813.md` | **CONSUMED-VIA-MEMO** | RFO1, MT1, and CN4 consume its current probability-object, pose, rate, and receiver-treatment rows. |
| 10 | #978/MT1 | `ddm_op1r_20260809/OP1R_PATH.md` | **CONSUMED-VIA-MEMO** | PI135/FD135/coder closure and RFO1 supersede its PR130 path ordering on the CP135 object. |
| 11 | #978/MT1 | `ddm_cl1_capacity_20260809/BLOCKED_RECEIPT.md` | **INERT** | No CL1 rung was measured; the current matched evidence defaults self-orient off, and EC1/EC2/MT1 provide the live formulations. |
| 12 | #978/MT1 | `ddm_ty1_20260806/RECEIPT.md` | **CONSUMED-VIA-MEMO** | Its 37-group toy/optimal-form inventory is already reflected in HV1/LV2/RFO1; no unconsumed #978 candidate remains in the row. |
| 13 | MICRO35/MC36 | `ddm_rfo1_fresh_hybrid_compose_20260814.md` | **CONSUMED-VIA-MEMO** | Its MICRO35 route was built by MC35 and repaired by MC36; its MT1 route was executed separately. |
| 14 | MICRO35/MC36 | `ddm_mc35_micro35_union_build_20260814.md` | **CONSUMED-VIA-MEMO** | MC36 built both pre-registered successors and found candidate C; the original +44 B / pose-failing instance stays folded. |
| 15 | MICRO35/MC36 | `ddm_mc36_micro35_variants_20260814.md` | **SIGNAL-BEARING** | Consumer: `/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/dispatch/ddm_mc36_dual_axis_t4_r1`. Fire first among small exact candidates when MAIN proves the n600 lane free and claims `ddm_mc36_dual_axis_t4_r1` against the sealed archive/runtime hashes. |
| 16 | MICRO35/MC36 | `ddm_eu4_fresh_eyes_fractal_composition_20260813.md` | **CONSUMED-VIA-MEMO** | Its sub-band micro composition was concretized by RFO1→MC35→MC36; the separate pose-1000 idea is not a built candidate. |
| 17 | MICRO35/MC36 | `ddm_errata_8dp_band_instrument_mixing_20260813.md` | **CONSUMED-VIA-MEMO** | QS2 and RFO1 use the corrected report-ULP boundary; the retracted sign is not a new route. |
| 18 | MICRO35/MC36 | `ddm_qs2_compensation_rate_rung_20260813.md` | **CONSUMED-VIA-MEMO** | QS2 is a component of the built MC35/MC36 unions; no standalone pointer fire survives. |
| 19 | #984 | `ddm_qj1_followon_backlog_join_20260804.md` | **INERT** | This is an older backlog join, not current hot-state authority; current consumers are resolved through LV2/CN4/CN5/RFO1. |
| 20 | #984 | `ddm_oh1_20260807/OH1_FINDINGS.md` | **CONSUMED-VIA-MEMO** | HR1/CN4/CN5 and LV2 consume its bounded orphan-harvest rows. |
| 21 | #984 | `ddm_hr1_20260809T031504Z/HR1_FINDINGS.md` | **CONSUMED-VIA-MEMO** | CN4/CN5/LV2 adjudicate the routed findings and prevent a second campaign branch. |
| 22 | #984 | `ddm_cb2_20260809T125105Z/CB2_FINDINGS.md` | **CONSUMED-VIA-MEMO** | CN4/CN5 and the same-state coder closure absorb its codebook/coder findings. |
| 23 | #984 | `ddm_ty2_20260806/RECEIPT.md` | **CONSUMED-VIA-MEMO** | LV2's full terminal-campaign reconciliation consumes the synergy inventory at its stated scopes. |
| 24 | #984 | `ddm_deferral_queue_ledger_20260729.md` | **INERT** | It is a historical parking surface. The hot state and typed later receipts, not stale open labels, govern the campaign. |
| 25 | pz4 QAT | `ddm_rvs1_realization_survival_harvest_20260811.md` | **CONSUMED-VIA-MEMO** | RHO1, PZ4P/PZ4R, CN4, and RFO1 use its survival constraints and no additive rate credit. |
| 26 | pz4 QAT | `ddm_pp2_20260809T121528Z/PP2_FINDINGS.md` | **CONSUMED-VIA-MEMO** | OP1R, PI135/FD135, and LV2 consume its PR130-path findings. |
| 27 | pz4 QAT | `ddm_pr130_reproduce_20260809/SEMANTIC_SECTION_NO_MEMORYLESS_SLACK.md` | **CONSUMED-VIA-MEMO** | PI135, CN4, and RFO1 use this as the current same-state coder closure; it blocks, rather than creates, another coder route. |
| 28 | pz4 QAT | `ddm_pz4p_pose_gauge_preproof_20260811.md` | **CONSUMED-VIA-MEMO** | PZ4R tested the direct-v6 receiver instance; RFO1 keeps rate-aware PZ4-QAT held behind a parser-equal `>=2,000 B` byte pre-proof. |
| 29 | pz4 QAT | `ddm_fd135_fractal_decomposition_20260810.md` | **CONSUMED-VIA-MEMO** | PI135/RFO1/CN4 consume its 231-file decomposition and lossless non-wins. |
| 30 | pz4 QAT | `ddm_sg2_20260810/SG2_FINDINGS.md` | **INERT** | Its prior Seg-axis actuator audit supplies no current parser-equal PZ4-QAT object or unconsumed fire order. |

Top-sample disposition counts are `4/30 SIGNAL-BEARING` (`13.33%`),
`22/30 CONSUMED-VIA-MEMO` (`73.33%`), and `4/30 INERT` (`13.33%`). All four
signal-bearing rows were already named on the live routing surface; the census
did not discover a fifth route.

### Stratified calibration and projected residue rate

The ten below-cutoff calibration findings were:

- JS1: `pr95_dseg_30k_convergence_deepmath_20260630T1542Z.md` and
  `triality_reconcile_session_20260702T235337Z.md`;
- #978/MT1: `ddm_rr17_20260807/ROUND17_FINDINGS.md` and
  `pr91_hpm1_transfer_stack_worker_20260504.md`;
- MICRO35/MC36: `einstein_kolmogorov_ultra_DAG_FEED_20260721.md` and
  `papers_checked_geometry_of_noise_2602_18428_20260714.md`;
- #984: `boundary_inverse_custody_20260721T052100Z.md` and
  `cathedral_zero_cost_planning_row_hardening_20260516_codex.md`;
- pz4 QAT: `pact_nerv_vq_maturity_audit_for_codebook_investment_20260609.md`
  and `full_pipeline_self_compression_nextwave_worker_20260503.md`.

Calibration result: `0/10 SIGNAL-BEARING`. Four were consumed by later route
memos and six were inert because they were apparatus, literature nulls,
superseded directional-basis claims, or retired-vehicle work. With zero hits,
the honest point projection for the below-cutoff, route-positive remainder is
`0%`. If those ten were exchangeable independent draws, the one-sided exact
95% upper sensitivity bound would be
`1 - 0.05^(1/10) = 25.8866%`. They are equally allocated across route strata,
not proportional to the raw and overlapping discovery surfaces, so that bound
must not be presented as a confidence bound for all `1,354 + 4,906` raw units.
The defensible conclusion is narrower: signal is concentrated at the top of
the current-route ranking, and this sample supplies no evidence of an
additional route below it. The enriched `13.33%` rate is not a population
estimate either.

## Gestalt clause — routing and arithmetic

Nothing in A or B earns a score term. In C, MC36 candidate C is the only
already-built local row that passes every pre-authority gate: `37` fewer Seg
flips, `+17 B`, and `delta d_pose=-1.4632967835484165e-10` on its retained local
advisory surface. At the CP135 anchor,

`delta S = -37*(100/117,964,800) + 17*(25/37,545,489)
          + sqrt(10*(6.885642960696714e-6 - 1.4632967835484165e-10))
          - sqrt(10*6.885642960696714e-6)
        = -2.013385878838082e-5`.

This is a projection, not a score: CPU-to-T4 scorer transfer remains untested.
The exchange rate is `8.4771050347e-7 S/flip` versus
`6.6585895312e-7 S/B`, or `0.785479 flips/B`. MT1 measured zero net heldout
flips locally and therefore stays behind its sealed T4 sign gate. EC2 and JS1
remain the higher-capacity Seg routes aimed at the roughly `0.004` score-unit
need; this census supplies no new effect size for them. Consequently the
ordering stays: harvest/fire already-sealed current routes under the sole-lane
law, use MC36 as the smallest exact candidate, and do not add an author-derived,
PR136-coder, or historical-corpus branch.

## RECALL EVIDENCE

The recall pass covered the full required surfaces before adjudication:

- governing state: `PROGRAM.md`, `AGENTS.md`, `CLAUDE.md`,
  `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, the
  charter, and the common contract;
- author precedents: `.omx/research/pr_authors_github_intake_20260710.md`
  (#413), `.omx/research/pr95_110_authors_github_intake_20260710.md` (#414),
  and the EH1 receipt/table/cure surfaces under `.omx/research/ddm_eh1_20260806/`;
- newest-PR custody: `ddm_pi135_pr135_intake_20260810.md`,
  `ddm_pi136_leaderboard_breadth_intake_20260810.md`, and
  `ddm_fd135_fractal_decomposition_20260810.md`;
- coder closure queries over `memoryless`, `order-0`, `SMEVR`, `range coder`,
  `context`, and `learned prior`, yielding the four-section PR130 closure,
  #918, SV2, HP1, and the R7/SMEVR receipt family;
- gestalt/current-route queries over `js1`, `#978`, `mt1`, `MICRO35`, `mc36`,
  `#984`, `pz4`, `probability object`, `implicit edge`, and `oriented`, yielding
  RFO1 plus later GCA1→EC1→EC2, BG2→MT1, and MC35→MC36 receipts;
- bounded scoped-sweep receipts: LV2's complete terminal-campaign leg set,
  LX1's 26-row crosswalk, WL1's witness-line transfer table and explicit
  anti-orphan limitation, and RHO1's receipt-only survival-prior scope. I cite
  those bounded absences and did not resweep them;
- residue/sampling receipts: NA2's frozen `4,906` count, NA4's selector-bias
  result, VH2's top-level corpus partition law, and
  `src/tac/probe_outcomes_ledger.py::partition_vehicle_corpus/coverage`;
- canonical equations: full JSON registry query, then exact searches for
  `gap_decomposition_against_demonstrated_floor_v1`,
  `radius2_multistart_singleton_escape_v1`,
  `ddm_et4_twelfth_move_solver_carriage_split_v1`, and
  `ddm_et5_restricted_carriage_family_fold_v1`;
- graph/task surfaces: `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks,
  canonical task status, harness task-list bridge, and current queue/hot-state
  rows for JS1, #978, #984, MICRO35/MC36, and pz4.

What appeared beyond the charter seeds was decisive for bookkeeping but not
for a new route: the charter predated the terminal BG2→MT1 and MC35→MC36
receipts and the sealed EC2 trainer. Those later artifacts changed the census
grades from “candidate follow-on” to either consumed or already-live
signal-bearing. The canonical equations and scoped sweep receipts supplied no
new current-object candidate beyond those seeds in the searched scopes.

## Verification and boundaries

- The author dispositions are scoped to public sources findable by exact-handle
  search on 2026-08-14. Private, deleted, unindexed, or unlinked work is outside
  the claim.
- PR136 was verified from retained source, not its missing exact archive. No
  reported PR136 score or byte count was reproduced here.
- The residue grades are finding-level readings. Raw artifact/document counts
  are discovery denominators, not evidence counts, and their overlap was not
  added into a fake population size.
- The projected unsampled rate is explicitly selection-limited. No family
  absence is inferred from ten calibration rows.
- No payload existed in memory that was measured then discarded. No persisted
  evidence path points to `/tmp`.
- No exact frontier pointer moved. This unit is a completeness answer and
  routing audit, not goal progress.

OWN-VEHICLE FRONTIER: unchanged at LC2, `S=0.16959899569230852`, `187,226 B`,
`[contest-CUDA T4, n600]`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN sole scorer-lane router; consumer store: `/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/dispatch/ddm_mc36_dual_axis_t4_r1`; fire trigger: no active full-n600 scorer lane, MAIN has claimed `ddm_mc36_dual_axis_t4_r1`, and the sealed archive/runtime hashes match; action: run the retained dual-axis T4 worker and harvest every required payload.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN EC2 trainer/scorer-lane router; consumer store: the sealed EC2 `main_cuda/` retained root named in `.omx/research/ddm_ec2_oriented_adapter_trainer_20260814.md`, then `#984`; fire trigger: no duplicate EC2 lane, Modal is available, and all sealed request/source hashes pass; action: fire oriented first and suppress controls unless it clears the registered break-even gates.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN #978 scorer-lane router; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/multitoken_978/ddm_mt1_20260814/optimal_form_r2/retained/t4_sign_gate_r1/`; fire trigger: higher-priority active exact rows are terminal, the #978 T4 lane is free, and the sealed hashes match; action: run only the sign gate, then fold unless `positive_t4_sign=true`.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN/JS1 promoted-axis owner; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/`; fire trigger: the selected exact base is terminal and a 1:1 T4 lane can retain full-n600 raw/logit/argmax fields; action: rebuild the directed Stage-0 decomposition on that authority axis before any JS1 learned-conditioning verdict.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: next public-intake custody owner; consumer store: `/Volumes/VertigoDataTier/pact/pr_breadth_intake_20260810/pr136/archive/`; fire trigger: the exact PR136 release binary becomes publicly reachable; action: retain, hash, and parse it to resolve source byte-reporting provenance, with no scorer fire or mechanism rerace.

## LIVE-HYPOTHESES

- MC36 candidate C may preserve its local `-2.013385878838082e-5` projected sign on T4 because it is receiver-closed, repeat-identical, and clears all three pre-authority gates; CPU-to-T4 Seg/Pose transfer is the unresolved risk.
- EC2's oriented implicit conditioner may buy materially more than the micro rows because it changes learned receiver-native representation at the event support rather than adding an explicit overlay; its counted module must still amortize at the complete-container boundary.
- MT1 could show a CUDA sign reversal despite its local zero-net-flip result because LC2 established sign-relevant CPU/CUDA component drift; only the sealed T4 sign gate can decide that narrow formulation.
- The signal-bearing residue rate may fall sharply below the relevance cutoff because the stratified calibration found zero of ten; a future census is warranted only after material route or corpus drift, not to resample the unchanged frozen surface.
- A recovered PR136 archive may explain the `~1.1 KB` versus `1,501 B` source discrepancy through container/header interactions, but it is unlikely to reveal a new context mechanism because the retained encoder and decoder fully specify adaptive order-0 state.

## DEAD-ENDS

- The newest-author public profiles yielded no new adopt-now or race mechanism. JPL11's event-camera work is lesson-only; the other five searched profiles added nothing beyond already-consumed PR evidence in the bounded public scope.
- PR136 is not a hidden context coder. Its per-tensor adaptive order-0 histogram neither beats the current section-specific memoryless/match-structure closure nor reopens the measured context/model races.
- Recounting artifacts as findings is closed. The raw `ddm_*` and untagged-document surfaces overlap, and most high-ranked material is consumed by later memos.
- Prefix or pure-recency sampling is closed by the NA2/NA4 selection warnings. This census used route strata, deduplication, and a separately seeded below-cutoff calibration.
- The BG1 bilinear gate, original MC35 union, standalone QS2 fire, same-state coder hunting, and direct PZ4-v6 receiver route remain closed at their stated instance/formulation scopes; the census found no new evidence that changes those dispositions.
