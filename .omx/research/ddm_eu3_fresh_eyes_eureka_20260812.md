# DDM EU3 fresh-eyes eureka hunt

Date: 2026-08-12  
Arm: `ddm_eu3`  
Outcome: **three representation-changing leads survive; no score was measured and no frontier moved**  
Authority boundary: research/derivation only; **no scorer, renderer, archive mutation, or Modal job ran**  
Axis: arithmetic is anchored to `cp135` at `S=0.16195513827824176`, `186,252 B`,
`[contest-CUDA T4, n600]`; all proposed gains remain unmeasured until that same axis closes them.

## Bottom line

The fresh derivation and the live corpus point to the same narrow opening from different directions:
stop treating the shipped semantic token plane as fixed data to be compressed. Treat **the token plane,
its task-equivalent RGB realization, the frame-0 pose carrier, and the HP3 probability object as one
discrete rate-distortion object**. The highest-value experiment is a CUDA-native alternating solve that
accepts semantic events only after their realized Seg gain, nonlinear Pose cost, and actual RC64/model
byte cost are known.

This is not a claim that the route works. The strongest measured clue is narrower: C1 changed 27,351
semantic sites while the HP3 token stream grew by only 6 B, so this semantic plane is not intrinsically
rate-bound on that trial. The candidate failed because its realization made Seg and Pose much worse,
not because HP3 could not describe the changed labels. That converts the live problem from “find a
better lossless coder” to “find task-improving semantic representatives inside cheap HP3 cells.”

The exact pointer remains `cp135`; the own-vehicle pointer remains `lc2 S=0.16959899569230852 @
187,226 B [contest-CUDA T4, n600]`. This unit did not achieve THE GOAL.

## Phase 1 — independent derivation from the evaluator

Only `upstream/evaluate.py`, `upstream/modules.py`, and the fact that archive bytes are charged were
used for this phase.

The objective is

`S = 100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489`.

The Seg term observes only the second frame through a five-class argmax at `384x512`. The Pose term
observes both frames through the first six PoseNet outputs after the evaluator's YUV conversion. RGB
fidelity has no independent value. Therefore the ideal contest codec is a quotient codec over evaluator
sufficient statistics:

1. encode a last-frame decision-cell representative for the required Seg partition;
2. encode six pair-level Pose sufficient statistics or a cheaper jointly trained preimage;
3. let a deterministic receiver choose an RGB preimage in the intersection of those evaluator cells;
4. spend bits only on boundary events, pose corrections, and exceptions whose realized score benefit
   exceeds their exact byte price.

Frame 0 is structurally Seg-free, so it is the natural control surface for cancelling Pose changes caused
by frame-1 Seg events. The mathematically natural local solve is a Schur complement: for a proposed
frame-1 change `dx1`, choose frame-0 carrier coordinates `dc0` so
`J_pose,0 dc0 ~= -J_pose,1 dx1`, then accept or reject the aggregate pair update using the nonlinear
Pose term and actual archive bytes. This is only a proposal generator; the exact T4 forward and receiver
remain the verdict.

The ideal code is thus neither an RGB codec nor a fixed-label entropy model. It is an
**entropy-constrained evaluator-preimage compiler**.

## Exact price sheet

At `n600`, there are `600*384*512 = 117,964,800` Seg symbols.

| Quantity | Exact arithmetic | Meaning |
|---|---:|---|
| one robust net-corrected Seg pixel | `100/117,964,800 = 8.4771050347e-7 S` | benefit before Pose spill |
| one archive byte | `25/37,545,489 = 6.6585895312e-7 S` | cost |
| Seg break-even | `1.273108 B/net flip` | equivalently `0.78548` net flips/B |
| cp135 gap to 0.15 | `0.0119551382782 S` | measured-pointer arithmetic |
| cp135 Pose term | `sqrt(10*6.88e-6) = 0.0082945765413 S` | rounded component from the exact row |
| flips needed, bytes fixed, Pose unchanged | `ceil(0.0119551382782 / 8.477105e-7) = 14,103` | hard Seg-only target |
| flips needed after Pose reaches zero, bytes fixed | `ceil((0.0119551382782-0.0082945765413)/8.477105e-7) = 4,319` | joint route target |
| bytes needed after Pose reaches zero, Seg fixed | `ceil(0.0036605617369 / 6.6585895e-7) = 5,498 B` | representation-rate-only target |

All flip counts mean **net corrected pixels after regressions**, on the exact CUDA scorer. Local margins or
event counts do not substitute for them.

## Phase 2 — transformed-state diff

The common contract's embedded frontier narrative is stale. The live typed state supersedes it:

- `cp135` is the effective frontier at `0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`.
- lossless recoding on this base is closed: RC64 already beats the tested ANS route, and the container
  and CAP1 savings are banked; rate work must change the representation or probability object;
- `ps135` generation 3 is still solving Pose; `xi2` is actively encoding its full-scale xi-context row;
  neither has a terminal verdict in the searched stores, so this arm does not duplicate or pre-claim it;
- `js7` proved an end-to-end semantic-event chain can reach an exact T4 row, but the 44-event stack lost
  `+0.00147 S`; its pose budget was about ten times too loose and its n32 Seg projection had the wrong
  sign;
- `ec1` produced 200/200 receiver-effective event proposals, but its standalone sparse payload was
  44,410 B and global adjacent-partition coding lost badly to intra-frame coding;
- `hc1` is the decisive transformed fact: direct C1 substitution produced an exact-decode terminal
  archive of 187,046 B, only 794 B over cp135, while its token stream was only 6 B larger. Yet exact
  `S=0.4044688` because `d_seg=0.00047310` and `d_pose=0.00541086`. **The semantic edit was cheap to
  describe and wrong to render.**
- `se1` closed class-wide hidden-4 amplitude packets only. Its best local benefits were below the
  current robust CUDA-disagreement floor `delta=0.0803604`; event-local representations remain open;
- `tf1` closed global temporal raster persistence/XOR/xi coding for this representation. Event sparsity
  survives, but belongs inside the probability object rather than a new standalone sidecar;
- `dg1` found a local topology-stable bending-energy residual suitable for ranking events. It did not
  prove a score gain and does not reopen global elastica/Willmore coding.

The independent derivation wanted score-equivalent representatives; the corpus says that representative
selection, not label entropy, is now the binding gap. The changed state is therefore:

`fixed labels -> compress` **becomes** `propose task-equivalent labels/realizations -> jointly price -> compress`.

## Ranked eureka table

Projected arithmetic below is a **falsifiable scenario, not an expected result**. No row is promoted by
analogy or literature ratios.

| Rank | Claim + honesty label | Hard falsifier | Cheapest admissible probe | Named consumer | Projected delta-S arithmetic |
|---:|---|---|---|---|---:|
| 1 | **Joint semantic-token/HP3 shipping-axis RDO** — `HYPOTHESIS`; optimize F26 token events, their receiver realization, HP3 prior/model bytes, and exact score together instead of coding fixed C1 labels | On held-out robust T4 events, no stack has negative exact `delta-S`, or refitting HP3 makes the total model+token+container rate exceed `1.273108 B/net flip` | Reuse EC1's 200 receiver-effective proposals; batch exact T4 singleton/pair aggregates, refit the real HP3/RC64 object for survivors, retain every candidate payload | MAIN/JS1 joint solve; `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/probability_object_race/` | With future Pose near zero, `4,800` net flips and `+500 B`: `-0.0040690104 +0.0003329295 -0.0082945765 = -0.0120306575`; `S=0.1499244808` (`CONJECTURE`) |
| 2 | **Frame-0 Schur pose antidote for frame-1 Seg events** — `HYPOTHESIS`; solve existing carrier coordinates per pair so accepted semantic events do not consume the terminal Pose gain | Exact nonlinear T4 Pose increase plus carrier bytes is at least the Seg benefit for every aggregate, or the compensator only works below the robust CUDA floor | On the same batched events, aggregate by pair, solve quantized frame-0 carrier deltas, then evaluate Pose and bytes jointly; no standalone sidecar | ps135 terminal carrier -> MAIN/JS1; `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/` | It earns no independent projected gain. Its bar is to preserve a future `-0.0082945765 S` Pose leg while rank 1 supplies at least `4,319` net flips |
| 3 | **Counted ordered-class-pair microtexture atom bank** — `HYPOTHESIS`; learn a small set of R-surviving, event-local AC atoms offline from SegNet, select them deterministically from decoded class-pair transitions, and ship/count every learned constant | No robust T4 net flips after R; Pose spill dominates; or actual bank+selector bytes exceed `1.273108 B/net flip` | Learn a tiny fixed catalog for the 20 ordered class pairs, test it on EC1 event locations, and byte-close the bank before any score claim | EC1/JS1 realization search; `/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/follow_on/realized_acceptance_200/` | Illustrative `540 B` bank + `4,800` net flips + future Pose zero: `-0.0040690104 +0.0003595638 -0.0082945765 = -0.0120040231`; `S=0.1499511152` (`CONJECTURE`) |
| 4 | **Predict HP3's 600x8 frame embedding from already decoded carrier/semantic state** — `HYPOTHESIS`; code only integer residuals | Whole-archive bytes do not fall after model/header effects, or the predictor changes token decode | Scorer-free fit on the retained frame embedding and decoded carrier summaries; rebuild and compare exact archive bytes | Probability-object race after xi2 verdict; same rank-1 store | `1,000 B` real saving would be `-0.0006658590 S`; no expected saving asserted |
| 5 | **xi-context** — `LIVE/MEASURED-LEG-A, TERMINAL UNKNOWN`; do not duplicate | Full-scale range bytes fail its registered `<=114,381 B` bar | Let the live resumable encode finish; inspect its retained terminal receipt | XI2 -> MAIN rate compose | Its `2,335 B` bar is only `-0.0015547807 S`; it cannot cross 0.15 alone |
| 6 | **Bending-energy residual as an event-ranking feature** — `MEASURED-LOCAL/HYPOTHESIS-SHIPPING` | Held-out ranking does not enrich robust T4 winners, or exact stacks do not improve | Add one feature to rank-1 proposal scoring; no separate codec | DG1 -> EC1/JS1 | `UNKNOWN`; it must improve realized net flips at unchanged bytes |
| 7 | **Evaluator-equivalent representative selection (G17 implementation debt)** — `SPECIFIED, NOT EXECUTABLE` | Search cannot produce a receiver-closed representative better than the current realization at the same token state | Fold the representative selector into rank 1; do not create a parallel arm | MAIN/JS1 | Same `1.273108 B/net flip` gate; no separate projection |
| 8 | **Runtime scorer or CUDA-ULP steering** — `DEAD` | Already falsified/forbidden: scorer use at inflate is not admissible; weak local gains fall below CUDA disagreement | None | none | Raw scorer weights total `94,338,452 B`; verbatim rate alone would be `+62.8161 S`, before archive compression; no legal score route |

## Recursive treatment of the top three

### 1. Joint semantic-token/HP3 shipping-axis RDO

**Why it could be real.** HC1 separated rate from realization more cleanly than any earlier receipt:
27,351 changed sites, `+6 B` in the token stream, `+794 B` in the complete archive, and catastrophic
task distortion. HP3 is already an image-dependent probability object, but it was fit to a selected token
plane rather than used inside selection. The new operation is an alternating discrete solve:

1. propose event-local token/realization moves from EC1, DG1, and the corrected JS2 CUDA gauge;
2. measure robust realized Seg/Pose effects on T4;
3. refit the actual HP3 prior and encode with RC64;
4. accept the aggregate only if the complete archive's exact `delta-S` is negative;
5. repeat from the accepted state, never from an unshipped surrogate.

The proposal is consistent with local information-bottleneck and indirect-RD equations, but it does not
inherit any numerical win from them. Per-image latent/prior optimization is also a known compression
pattern: image-dependent local entropy models adapt a prior to one image, and universal-encoder RDO
optimizes latents and side information per input. Those papers support plausibility only; Pact's exact
rates and score must be remeasured ([Minnen et al.](https://arxiv.org/abs/1805.12295),
[Zhao et al.](https://openaccess.thecvf.com/content/CVPR2021W/CLIC/papers/Zhao_A_Universal_Encoder_Rate_Distortion_Optimization_Framework_for_Learned_Compression_CVPRW_2021_paper.pdf)).

**Recursive branches.** If event quality is good but HP3 bytes rise, use rank 4's decoded-state predictor
and XI2's terminal context only after XI2 lands. If HP3 bytes stay cheap but task gains fail, the defect is
representative realization; hand the exact same events to rank 3. If singleton events win but stacks lose,
fit pairwise interaction terms and solve aggregates rather than tightening a scalar gate.

**Kill condition.** Formulation-close only the tested proposal catalog when no held-out robust T4 event
survives through exact receiver, Pose, and complete-archive pricing. Do not kill task-equivalent token-plane
optimization from one catalog.

### 2. Frame-0 Schur pose antidote

**Why it could be real.** The evaluator grants a Seg-free frame while Pose couples both frames. JS2b
showed that the existing carrier compensation machinery can materially move subset Pose, although its
weak class-amplitude event catalog produced no robust Seg win. The reusable mechanism is the carrier
solve, not that catalog. Decoder-side correlated information is known to improve ordinary compression,
and joint task/rate optimization is established in task-aware compression, but neither result transfers a
Pact gain ([Ayzik and Avidan](https://arxiv.org/abs/2001.04753),
[Recognition-aware compression](https://arxiv.org/abs/2202.00198)).

**Recursive branches.** Start with the linear Schur proposal, quantize to the shipped carrier grid, then
run a short exact nonlinear coordinate search. If one event at a time is unstable, solve the full pair
aggregate. If a pair is carrier-saturated, reject its Seg events rather than adding a new stored sidecar.

**Kill condition.** Close only “post-event compensation in the current carrier basis” if exact nonlinear
Pose plus byte cost consumes the event stack's Seg benefit on held-out pairs. A failure does not kill
joint training of Pose and Seg in the shipping vehicle.

### 3. Counted ordered-class-pair microtexture atoms

**Why it could be real.** Existing equations and receipts say partition identity alone is insufficient:
texture is load-bearing, global gratings are closed, and class-wide amplitude packets are too weak. That
leaves spatially local, ordered-class-pair texture as the missing representative degree of freedom.
Universal perturbations can alter semantic-segmentation outputs, including targeted label behavior, so a
small shared atom family is not physically absurd; this is an existence clue, not a transfer
([Metzen et al.](https://arxiv.org/abs/1704.05712)).

**Recursive branches.** First test one source-independent analytic AC catalog. If it produces robust
events, it is generic receiver code. If it fails, learn a tiny scorer-derived catalog offline and **count
the serialized bank**. If class-pair atoms are insufficient, condition on a coarse curvature bin derived
from decoded token geometry before adding stored selectors. Every branch retains each payload and records
bank bytes, selectors, decode equality, Seg, and Pose.

**Kill condition.** Close the tested atom family when no R-surviving robust T4 gain exists or actual
bank+selector bytes fail the break-even gate. Do not generalize that verdict to all event-local
realizations.

## Scorer-weight economics and compliance

Three distinct cases must not be blurred:

1. **Full scorer at inflate: dead.** The repository's strict rule forbids scorer loading in the receiver;
   README rule 118 requires large neural artifacts in `archive.zip`. The two raw safetensor files total
   94,338,452 B. Their verbatim rate term would be `62.8161 S` before considering any compression, and
   runtime scorer use is outside the approved receiver contract anyway.
2. **Video-trained surrogate or scorer-derived learned atom bank: counted.** It can be economically
   rational only if its serialized bank, selectors, and resulting token changes beat the exact
   `1.273108 B/net flip` gate after Pose. Rank 3 uses this conservative treatment.
3. **Source-independent generic algorithm/constants: potentially free, not presumed free.** Analytic
   rasterizers and deterministic generic code fit the free-code rule. Architecture/scorer-derived
   constants baked into code are compliance-sensitive. Until an explicit ruling says otherwise, count
   them. No hide-data-in-code or uncounted scorer surrogate is proposed here.

The exploitable object is therefore scorer **geometry learned offline and economically summarized**, not
scorer weights at runtime.

## CUDA determinism verdict

The corrected JS2 custody shows that promoted and local decoded scorer inputs agree to within six Seg
pixels while the forward results differed substantially across CPU/CUDA. That makes CPU-local scores a
proposal gauge, not a promotion surface. The live robust-disagreement floor `delta=0.0803604` is a reason
to reject tie steering, not a source of free gain. A legal route is CUDA-native optimization against
retained T4 scorer inputs with robust-margin admission; deliberately harvesting ULP/tie instability is
closed.

## Archive/Kolmogorov slack

The remaining slack is not a container trick. CP135's complete archive is `70,825 B` model, `96 B`
residual, `115,231 B` RC64 tokens, and `100 B` ZIP floor. HP3's step-2 probability-object change was only
`-8 B` net because model savings moved into tokens. The opened objects are therefore:

- a different task-equivalent token plane that is cheaper under the same probability family;
- a probability family conditioned on information the receiver already decodes;
- removal/prediction of HP3's per-frame embedding from decoded carrier/semantic state;
- a jointly optimized model/token state, rather than independent model pruning.

In the searched scope, I did not find an exact prior experiment predicting the 600x8 HP3 frame embedding
from the shipped carrier/semantic state and then comparing complete-archive bytes. That bounded absence
is why rank 4 survives. Dictionary-based entropy models and video RD autoencoders offer external
existence proofs for shared dictionaries and joint latent/prior learning, not numerical transfers
([Lu et al.](https://openaccess.thecvf.com/content/CVPR2025/html/Lu_Learned_Image_Compression_with_Dictionary-based_Entropy_Model_CVPR_2025_paper.html),
[Habibian et al.](https://openaccess.thecvf.com/content_ICCV_2019/html/Habibian_Video_Compression_With_Rate-Distortion_Autoencoders_ICCV_2019_paper.html)).

XI2's registered 2,335 B win bar is valuable but would move only `-0.0015547807 S`; it cannot replace
the Seg leg. A pure representation-rate route would need 5,498 B after Pose reaches zero.

## Cross-PR splice verdict

The only justified splice is **functional, then jointly reclosed**:

`cp135 rate package + ps135 terminal pose carrier + JS1 semantic events + HP3/RC64 refit`.

No literal byte splice receives credit. Direct C1-to-HP3 substitution is already measured harmful; the
new state must retrain/refit the interacting realization and probability object, parse back through the
actual receiver, and score the same archive bytes. The result remains “ours composed on a granted PR135
base,” not an own-vehicle row. Only a separately closed own-vehicle archive can move the `lc2` pointer.

## RECALL EVIDENCE

### Governing and live authority read

- `PROGRAM.md`; `CLAUDE.md`/`AGENTS.md`; `docs/operating_manual_craft_handoff.md`;
  `.omx/state/main_hot_state.md`; the charter and common contract.
- Current exact anchor: `.omx/research/ddm_cp135_rate_compose_20260810.md`.
- Fresh required receipts: `ddm_js7_exact_row_verdict_20260812.md`,
  `ddm_ec1_event_coordinate_producer_20260812.md`,
  `ddm_tf1_theoretical_floor_and_beyond_20260812.md`,
  `ddm_dg1_pinkall_elastica_crosswalk_20260812.md`,
  `ddm_hc1_hy1_container_push_20260812.md`, and
  `ddm_se1_shipping_axis_survival_resolve_20260812.md`.
- Probability-object and CUDA custody: `ddm_hp3_20260810/FINAL_REPORT.md`,
  `ddm_xi2_xi_context_full_scale_promotion_20260812.md`,
  `ddm_js2_implicit_edge_conditioning_20260812.md`,
  `ddm_js2b_edge_conditioning_relative_gauge_20260812.md`,
  `ddm_rr1_20260809/RECALL_AUDIT.md`, and `ddm_js1_stage0_per_edge_20260812.md`.
- Live XI2 retained store was inspected read-only: terminal model XZ is 15,100 B and the resumable encode
  had reached 525 frames; no `FULL_SCALE_RESULT` existed at inspection time. It remains LIVE, not won.

### Equation registry consulted

`tools/list_canonical_equations.py --json` was searched for exact objective/marginal, indirect-RD,
information-bottleneck, rate-MDL, Pose square-root coupling/null-space/YUV sensitivity, Seg rank/flip
distance, realization necessity, textured-power-diagram sufficiency, palette/context ceiling, temporal
transport, weight-entropy, ideal argmax-cell bytes, predict-project admissibility, quotient-functional,
gap-decomposition, and multistart equations. Load-bearing concepts used here are the exact score
marginal, frame-0 Pose control surface, indirect-RD quotient, texture necessity, and real
receiver-preimage requirement. No stored numerical ratio was transferred.

### Full-corpus search families and beyond-seed changes

The research index, DAG/specs, design indexes, active-lane registry, canonical equations, and research
corpus were searched for: `HP3/probability object/frame_embed`, `score-aware entropy/semantic RDO`,
`evaluator-equivalent representative/G17`, `scorer weights/derived constants/rule 118`, `CUDA
disagreement/ULP/tie`, `event coding/curvature/bending`, `pose null/frame0/Schur`, `cross-PR splice`, and
`microtexture/universal perturbation/ordered class pair`.

Beyond the charter's seed receipts, that search changed the answer in four ways:

1. JS2's original 44.1% discrepancy was corrected by scorer-input custody: decode disagreement is only
   six pixels; the remaining gap is the CPU-vs-CUDA SegNet forward. This killed ULP/tie harvesting and
   promoted robust T4 batching.
2. RR1 showed HP3 already consumes previous semantic state, causal patch context, coordinates, frame
   embedding, and SPM. Rank 1 is therefore joint state/prior optimization, not a rediscovery of generic
   context modeling. Rank 4 uses an untested cross-section: deriving frame embedding from decoded state.
3. G17 evaluator-equivalent representative selection is already specified but remains executable debt.
   It is folded into rank 1 rather than claimed as a new separate vehicle.
4. Older scorer-specialized packet-compiler discussions contained compliance-sensitive “free constant”
   ideas. The current strict scorer and rule-118 language overrules any casual free-byte assumption;
   rank 3 counts learned atoms by default.

I did not find, in those searched scopes, a receiver-closed T4 experiment jointly optimizing an F26
semantic token plane and HP3's probability object, or an exact whole-archive frame-embedding predictor.
Those are bounded absence statements, not claims of global novelty.

## What is and is not measured

**Measured and inherited:** cp135 exact score/bytes; lc2 own-vehicle row; JS7 exact loss; HC1 exact
archive/score and token-byte delta; HP3 receiver equality; EC1 receiver-effective proposal count and
sidecar bytes; TF1 codec comparisons; SE1 weak-margin close; scorer model raw file sizes; current XI2
checkpoint/progress presence.

**Derived here:** exact byte and net-flip prices; sub-0.15 thresholds; evaluator quotient; frame-0 Schur
proposal; ranking and conditional scenario arithmetic.

**Not measured:** any robust gain from joint token/HP3 optimization; any Pose preservation from the
Schur antidote; any microtexture atom; any frame-embedding prediction saving; any XI2 terminal byte win;
all projected sub-0.15 scenarios.

## Borrowed-substrate accounting

Borrowed: PR135/CP135 archive base, F26 semantic representation, HP3 format and RC64 coder, EC1/C1 event
producers, JS2 CUDA custody/instrumentation, and the ps135 carrier machinery.

Original in this arm: the evaluator-only quotient derivation; the transformed-state conclusion that C1
is realization-bound rather than token-rate-bound; the joint semantic-plane/probability-object action;
the frame-0 Schur composition; the counted ordered-class-pair microtexture proposal; and the bounded
frame-embedding cross-section. These are research hypotheses, not implementations or score claims.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN/JS1 shipping-axis joint-solve owner; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/probability_object_race/`; fire trigger: ps135 generation 3 lands a terminal retained carrier, MAIN owns the sole T4 scorer lane, and XI2 is not using a conflicting consumer; batch EC1 events on exact T4, jointly refit HP3/RC64, retain every payload, and admit only complete-archive negative `delta-S`.
- **FOLDED** — owner: MAIN/JS1 pose-conditioning owner; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/`; fire trigger: rank 1 finds at least one robust receiver-effective Seg event on a pair; solve and quantize the frame-0 Schur antidote inside rank 1 before stack admission, with exact nonlinear Pose and carrier-byte pricing.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: next EC1/JS1 realization-search owner; consumer store: `/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/follow_on/realized_acceptance_200/`; fire trigger: rank 1's cheap semantic events remain task-harmful under current realization; test an analytic atom catalog first, then a tiny explicitly counted learned ordered-class-pair bank, retaining every candidate.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: next probability-object rate owner; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/probability_object_race/`; fire trigger: XI2 writes its terminal full-scale receipt; scorer-free fit an integer predictor for HP3's 600x8 frame embedding from already decoded carrier/semantic state and accept only a smaller byte-identical complete archive.

## LIVE-HYPOTHESES

- Jointly choosing the semantic token plane and HP3 probability object can find task-better states that
  remain cheap to code, because HC1's 27,351-site change cost only 6 token bytes even though its chosen
  realization was task-worse.
- Frame-0 carrier coordinates can cancel the Pose effect of frame-1 Seg events, because frame 0 is
  Seg-free and the existing compensation machinery has already shown Pose actuation; only exact
  nonlinear T4 closure can establish useful rank.
- Ordered-class-pair local microtexture atoms can supply the texture information missing from pure
  partition edits, while sharing enough structure to beat the `1.273108 B/net flip` gate.
- HP3's per-frame embedding may be partly predictable from state already decoded by the carrier/token
  receiver, because the current representation stores both descriptions of per-frame semantic regime;
  the complete-archive test has not been found in the searched scope.
- DG1's bending residual may enrich robust event ranking when used only as a feature, because it is local
  and topology-stable where global curvature codecs failed.

## DEAD-ENDS

- Lossless coder/container retuning on CP135: closed by the banked RC64, container, CAP1, and exact
  ancestry results; only representation-changing rate work remains.
- Direct C1 semantic substitution: closed for that realization by HC1's exact `S=0.4044688`; the
  probability object survived, the rendered representative did not.
- Global temporal adjacent-partition coding: closed for TF1's persistence/XOR/xi formulations because
  all were larger than intra-frame coding.
- Standalone EC1 sparse sidecar: closed at 44,410 B for its tested representation; event coordinates may
  survive only inside the joint probability object or deterministic decoded geometry.
- Class-wide hidden-4 amplitude packets: closed for SE1's tested formulation because gains stayed below
  the robust CUDA-disagreement floor and no exact rung was justified.
- JS2b's weak two-FiLM seed catalog: closed for that catalog because all beneficial local flips were
  below the robust floor and the exact candidate added bytes without robust flips.
- Global grating/elastica/Willmore realization: closed by the prior texture and DG1 evidence; only local
  topology-stable features/atoms remain live.
- Runtime SegNet/PoseNet or uncounted learned scorer surrogate: forbidden and economically noncompetitive;
  scorer-derived learned content must be counted.
- CUDA ULP/tie harvesting: closed because the corrected custody locates the discrepancy in the
  CPU-vs-CUDA forward and weak gains are not robust; T4-native robust optimization is the legitimate path.
- Literal cross-PR byte splicing: closed as a claim; interacting carrier, semantic, and probability states
  must be jointly refit, receiver-closed, and scored on the exact resulting archive.
