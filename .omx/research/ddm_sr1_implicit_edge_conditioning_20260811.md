# ddm_sr1 — implicit edge conditioning for the semantic-token vehicle

## Verdict

The charter's RATE falsifier fired. On the full retained n600 PR135/F26 token
population (117,964,800 events), the best genuinely additional decoder-known
edge context saved **2 charged bytes** against the 114,706 B live F26 token
stream, or **0.00174%**. A pose-to-token context cost **43 charged bytes**.
Both candidates decoded every symbol exactly, reached an empty terminal state,
and reproduced byte-identically in independent deterministic encodes. This is
far below the preregistered 1% threshold (1,147.06 B), so the implicit-
conditioning RATE route is CLOSED at FORMULATION scope for additive probability
calibration from the tested causal-edge and scalar-pose contexts.

This does **not** kill implicit conditioning as a distortion mechanism. The
remaining live route is to expose a decoder-derived edge state inside js1's
joint semantic/pose solve so the same counted model capacity is spent on
SegNet-relevant boundaries. No edge labels or contour stream may be shipped.
That route has no honest projected `d_seg` number yet; it must earn one through
the actual rendered candidate and scorer lane owned outside sr1.

The exact frontier is unmoved. This arm produced a section-level coder result,
not an `archive.zip`, rendered video, scorer result, or contest row.

## Authority and scope

- Axis: `[macOS-CPU advisory, scorer-free n600 token entropy]`.
- Population: all 600 frames and 117,964,800 retained discrete token events;
  selection used even-frame fit / odd-frame holdout, followed by a full-n600
  refit only for the selected tables.
- Pinned probability object: PR135/F26 full-n600 exported probability codes.
- Real coder: the checkpointable native RC64 implementation, using the same
  recurrence as the retained baseline.
- Comparator: 114,706 B F26 token stream. Charges below include the complete
  correction table and eight bytes of composite framing.
- Verdict scope: **FORMULATION**, specifically small additive calibration
  tables over the exact F26 lattice using (a) already-decoded current-frame
  group topology or (b) one already-carried pose coefficient's sign/delta-sign.
  This is not a FAMILY negative on learned joint cross-stream conditioning.
- No scorer, Modal, MPS score authority, public-eval claim, or transfer from a
  foreign image/video corpus was used.

## Measured full-n600 results

| candidate | new decoder-known context | holdout gain before table | selected table | real token bytes | charged composite bytes | charged delta vs F26 | falsifier |
|---|---|---:|---:|---:|---:|---:|---|
| causal edge | predicted class crossed with no prior group / unanimous same / unanimous different / mixed among groups already decoded at the site | +752.509 bits | 112 B, int8 | 114,584 B | **114,704 B** | **−2 B (−0.00174%)** | fires |
| pose cross-stream | predicted class crossed with the selected carried pose coefficient's delta sign (dimension 8), selected among 24 sign/delta-sign variants | −30.936 bits | 37 B, int4 | 114,704 B | **114,749 B** | **+43 B** | fires |

The causal row's full-fit NLL gain was 1,018.308 bits, but its odd-frame
holdout net was −143.491 bits after the table charge. The two-byte real-coder
win therefore should not be interpreted as robust generalization. The pose row
was already negative on held-out frames. The baseline achieved rate is
0.00777899848 bits/event; the table-free NLL is 917,643.6804 bits.

For both rows, decode returned the retained event stream SHA-256
`8eb51ab7a2884c9d7b6e73ee60f78ded38c691d6b82e639b75dddec6e0ac1366`
and spatial-token SHA-256
`c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`,
with 25 stage checkpoints and an empty terminal coder state.

## Ranked mechanism table

| rank | mechanism | route (Δbytes equal seg or Δseg equal bytes) | evidence class | $0 probe result/design | js1-stage-1 consumer hook |
|---:|---|---|---|---|---|
| 1 | Joint nonlinear edge-state modulation of the semantic proposal, using edges derived from the shipping-base decode and already-carried pose/carrier state | **Δseg at equal bytes**; no projected value is claimed | OPEN HYPOTHESIS, supported by current-vehicle explicit/implicit fork plus off-the-shelf learned-context existence proofs; unmeasured on our scorer | Stage 0 retains a shipping-base per-edge decomposition; stage 1 conditions its proposal/trust region on that state without serializing a mask, and compares only complete retained candidates | Feed the derived edge state into the global semantic/pose solve before quantization; admit only a complete receiver-closed candidate whose joint `d_seg`, `d_pose`, and bytes beat control |
| 2 | Edge-conditioned adaptive precision/capacity routing over the existing semantic basis, FiLM, or int12 parameters | **Δseg at equal bytes** by reallocating existing counted parameters, or representation-changing Δbytes only if the complete model shrinks | DERIVED from the score/rate breakeven law and current edge-local error geometry; current-vehicle scorer result absent | Reuse the stage-0 edge sensitivity and ps135 sensitivity surface to exchange parameter bits between low- and high-boundary-debt regions at fixed total model bytes | Make the allocator a stage-boundary treatment inside js1; retain every complete allocation and forbid admission from proxy loss alone |
| 3 | Learned space-channel/cross-stream context inside joint training: semantic group state conditions later semantic/pose proposals, and pose/carrier state conditions semantic boundary work | primarily **Δseg at equal bytes**; any learned context parameters are counted | OFF-THE-SHELF existence proof plus corpus-derived mapping; no numeric transfer | Implement only after ranks 1–2 expose the same decoder-known state; compare to an equal-parameter control, not to a model with fewer counted weights | Treat semantic and pose latent groups as the uneven channel groups; charge all learned mixing parameters and preserve causal receiver order |
| 4 | Additive current-frame causal-edge calibration on the exact F26 probabilities | **Δbytes at equal seg: −2 B section-level**, not an archive delta | MEASURED `[macOS-CPU advisory, scorer-free n600 token entropy]`; FORMULATION CLOSED | Completed full-n600 real-coder probe and deterministic repeat | Negative guard: do not add this standalone table to js1; only a representation-changing joint model may revisit the underlying state |
| 5 | Additive scalar-pose-to-token calibration on the exact F26 probabilities | **Δbytes at equal seg: +43 B section-level** | MEASURED `[macOS-CPU advisory, scorer-free n600 token entropy]`; FORMULATION CLOSED | Completed 24-variant selection, full-n600 real-coder probe, and deterministic repeat | Negative guard: do not bolt scalar sign tables onto the F26 entropy model |
| 6 | Checkerboard/reordering alone over the existing HPAC group order | no honest Δbytes prediction; no new information is introduced | DERIVED CLOSED INSTANCE for a standalone rate lever | No second full encode: PR135 already has causal within-patch groups and the measured rows show negligible residual gain from added local topology | Reordering is allowed only if required by ranks 1–3 and must beat the same complete-object control |
| 7 | Explicit PE3 labels, chain codes, contours, or quadtree masks | prior current-vehicle result is scorer-negative despite a byte-positive section | MEASURED prior negative for the EXPLICIT family on this vehicle | No rerun: pk1/lc1 already found ideal substitution worsened all 32 pairs; PE3 costs 74,408 B | Never transmit these as the conditioning object; a decoder-derived edge feature may still be used internally by ranks 1–3 |

At the score-rate slope recorded in `seg_rate_breakeven_v1`, one byte is worth
about `6.66e-9` in `d_seg` (150.18 B per `1e-6 d_seg`). The causal row's two
bytes therefore have no practical bearing on the required roughly −0.004 seg
movement. That comparison is DERIVED, not a scorer measurement.

## Research sweep and transfer boundary

- The Checkerboard Context Model rearranges spatial decoding into anchor and
  non-anchor passes while retaining nearly the same image RD performance; it is
  an execution/order design, not evidence that our already-causal F26 stream
  contains another kilobyte. [He et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/He_Checkerboard_Context_Model_for_Efficient_Learned_Image_Compression_CVPR_2021_paper.html)
- ELIC combines uneven latent channel groups with space-channel context. ChARM
  conditions later latent channels on earlier decoded channels and adds latent
  residual prediction. These establish that learned group-to-group context can
  improve image codecs, but their percentage savings do not transfer to this
  117,964,800-event semantic-token object. [ELIC, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/He_ELIC_Efficient_Learned_Image_Compression_With_Unevenly_Grouped_Space-Channel_Contextual_CVPR_2022_paper.pdf)
  [ChARM, 2020](https://arxiv.org/abs/2007.08739)
- Autoregressive and hierarchical priors can be complementary in learned image
  compression. PR130/F26 already instantiates that broad idea through HPAC,
  frame embedding, spatial group context, and a previous-frame boundary
  calibration; sr1 therefore probed only context not already in that lattice.
  [Minnen, Ballé, and Toderici, NeurIPS 2018](https://papers.nips.cc/paper_files/paper/2018/hash/53edebc543333dfbf7c5933af792c9c4-Abstract.html)
- Diverse-context and context-modulation video codecs support the plausibility
  of jointly learned spatial/temporal context and oriented temporal modulation.
  Their relevant lesson is architectural: modulation must participate in the
  learned representation, not arrive as a tiny post-hoc calibration table.
  [DCVC diverse contexts, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Neural_Video_Compression_With_Diverse_Contexts_CVPR_2023_paper.html)
  [context modulation, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Tang_Neural_Video_Compression_with_Context_Modulation_CVPR_2025_paper.html)
- Semantic-map codecs built around chain codes, quadtrees, Markov contour
  contexts, and shared-boundary skipping transmit an explicit boundary object.
  They are useful existence proofs for boundary grammar, but map to the explicit
  family already closed on this vehicle, not to a free implicit context.
  [CC-SMC code](https://github.com/Yang-Runyu/CC-SMC)
  [context-adaptive extended chain coding, 2026](https://arxiv.org/abs/2603.03073)

The ranked recommendation does not borrow any paper's percentage. Every paper
operates on a different source, objective, and probability object. The only
current-vehicle numeric conclusion comes from sr1's retained real-coder rows.

## Corpus crosswalk

The corpus resolves the apparent contradiction between “explicit overlays are
dead” and “implicit conditioning is open”:

- pk1/lc1 killed a transmitted, explicit PE3 label substitution: its 74,408 B
  section was receiver-closed, but the ideal n32 substitution worsened every
  pair, from 27,382 to 40,266 errors, dominated by Lane-over-Road mistakes.
- fd135 therefore scoped the negative to explicit overlays and left joint
  implicit conditioning open.
- sg2 says shipping per-edge Seg decomposition cannot be reconstructed from the
  retained scored candidate, so js1 stage 0 remains the necessary source of a
  current shipping-base edge state.
- RR1 and the PR135 runtime show that the live model already consumes the
  previous semantic frame, causal within-patch neighborhoods, patch coordinates,
  a learned 600×8 frame embedding, coarse SPM context, and a counted previous-
  frame boundary-bucket × predicted-class int6 table. Thus a generic “add edge
  context” proposal would duplicate live machinery.
- CPC1's exact causal partition replacement cost 255,288 B at n600, more than
  twice the PR130 stream, closing a standalone partition-codec replacement.
- CR1 compressed a selected edge-support object from 575,095 B to 464,557 B,
  but that object is not the full token stream or renderer; it supports boundary
  grammar as a feature, not an archive-rate claim.
- LP135/HM1 closed recurrence-only coder tweaks and require model-plus-token
  pricing. F26's shipping 100 B boundary table and sr1's charged tables follow
  that accounting.
- SN1's edge-band telemetry came from another vehicle/axis. It supports only
  the geometry prior; none of its percentages were transferred to this verdict.

## RECALL EVIDENCE

Sources searched before design:

- Full corpus stores through `tools/corpus_query.py`: research (8,350 rows),
  equations (880), memory (2,105), DAG (915), council (297), tasks (531), and
  docs (96).
- Exact query strings included `implicit conditioning`, `per-edge`, `PE3`,
  `edge-conditioned`, `context model`, `partition entropy`, `grammar-v2`,
  `Lane-over-Road`, `HPAC boundary predicted`, `causal context partition`,
  `cross-stream`, `semantic token`, `checkerboard`, `ELIC`, `channel
  conditional`, and `segmentation map compression`.
- Canonical equations registry and direct records for
  `ddm_lp1_deepest_home_context_waterfill_v1`,
  `seg_rate_breakeven_v1`,
  `partition_temporal_transport_amortization_jitter_bound_v1`,
  `scorer_conditional_joint_rate_distortion_floor_v1`,
  `wyner_ziv_decoder_side_information_conditional_entropy_savings_v1`, and the
  rate/MDL records.
- Canonical research index, sub-0.15 DAG/FEED blocks, task ledger, fd135,
  pk1/lc1, sn1, sg2/#906, RR1, CPC1, CR1, LP135/HM1, F26/CP135, and the js1
  charter.
- Online primary papers and official OSS/project surfaces for checkerboard,
  ELIC/ChARM, combined hierarchical/autoregressive priors, DCVC, context
  modulation, and semantic-map boundary coding. Unofficial ELIC ports were not
  treated as implementation authority.

Beyond the charter's seeds, recall found three decision-changing facts:

1. F26 already ships a 100 B previous-frame-boundary × predicted-class
   calibration, so the broad edge-rate proposal was already implemented.
2. CPC1 had already made an exact full-n600 causal partition replacement and
   lost badly on bytes, so no replacement codec was rebuilt.
3. CR1's large edge-support compression win concerns only a selected support
   object, preventing an invalid transfer to the complete semantic-token wire.

These facts narrowed the executable probe to additional decoder-known contexts
on the exact retained F26 probability lattice and moved the highest-ranked live
mechanism from post-hoc rate calibration to joint distortion conditioning.

## Payload custody and reproducibility

All materialized outputs are retained under
`/Volumes/APDataStore/pact/ddm_sr1_implicit_edge_20260811/`; APDataStore was
selected by the storage waterfall because Vertigo had only about 26 GiB free
while APDataStore had about 929 GiB. The retained tree is about 1.0 GiB and
contains every table, candidate stream, full decoded symbol/token payload,
atomic analysis receipt, and per-24-frame checkpoint. No materialized payload
was deleted.

Key retained artifacts:

- analysis receipt:
  `/Volumes/APDataStore/pact/ddm_sr1_implicit_edge_20260811/ANALYSIS_RESULT.json`
- causal composite: 114,704 B, SHA-256
  `64248ddcb04c8c96b40abcef01c21018a136489c3a2c10b844aba8ef40020d6c`
- pose composite: 114,749 B, SHA-256
  `45889ed19b7300fa40299361c9343b75b965d662f5a70e0144d736827ee6cc21`
- deterministic repeats:
  `/Volumes/APDataStore/pact/ddm_sr1_implicit_edge_20260811/determinism_repeat/retained/candidates/`
  with byte-identical composite and token SHAs for both candidates.

The runner is restartable from atomic checkpoints and preserves 25 checkpoints
per full encode. The first encode attempt stopped before any payload existed
because the pinned library lacked decoder/checkpoint symbols; it was corrected
to the exact rc64p library before the measured run. A concurrent repeat exposed
that cleanup could remove a sibling process's fresh atomic temp file; the runner
now protects atomic temps younger than 24 hours, and both repeats then completed
byte-identically.

Reproduction commands:

```bash
.venv/bin/python experiments/ddm_sr1_implicit_edge_preprobe.py analyze \
  --output /Volumes/APDataStore/pact/ddm_sr1_implicit_edge_20260811
.venv/bin/python experiments/ddm_sr1_implicit_edge_preprobe.py encode \
  --candidate causal_edge \
  --output /Volumes/APDataStore/pact/ddm_sr1_implicit_edge_20260811
.venv/bin/python experiments/ddm_sr1_implicit_edge_preprobe.py decode \
  --candidate causal_edge \
  --output /Volumes/APDataStore/pact/ddm_sr1_implicit_edge_20260811
.venv/bin/python experiments/ddm_sr1_implicit_edge_preprobe.py encode \
  --candidate pose_cross_stream \
  --output /Volumes/APDataStore/pact/ddm_sr1_implicit_edge_20260811
.venv/bin/python experiments/ddm_sr1_implicit_edge_preprobe.py decode \
  --candidate pose_cross_stream \
  --output /Volumes/APDataStore/pact/ddm_sr1_implicit_edge_20260811
```

## Originality and unmeasured boundaries

sr1's original work is the additional-context formulation, the full-n600
selection/encode/decode apparatus, and the measured negative. The PR130/PR135
vehicle, HPAC probability lattice, retained DT1 symbol corpus, F26 export, and
rc64p coder are borrowed in-repo substrate and are not claimed as sr1 work.

Not measured here: any new `d_seg`, `d_pose`, rendered frames, uint8/resize/
parse-back survival, complete archive bytes, zip framing effect, CPU/CUDA exact
score, or js1 stage-1 candidate. The two-byte section result cannot move either
the effective pointer or the own-vehicle frontier.

Current effective composed pointer remains `S=0.16195513827824176 @ 186,252 B`
`[contest-CUDA T4, n600]`, archive SHA-256
`6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`.
Own-vehicle frontier remains `S=0.16959899569230852 @ 187,226 B`
`[contest-CUDA T4, n600]`.

## NEXT_IF_RESUMED

- `implicit_joint_distortion_conditioning` — **QUEUED-WITH-A-FIRE-ORDER**; owner: js1 stage-1 / #995 successor; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned/`; fire trigger: ps135 `SOLVE` has landed, js1 stage 0 has retained the shipping-base per-edge decomposition, the candidate ships no explicit edge mask, and the runner retains and jointly prices every complete semantic/pose/archive candidate against an equal-parameter control.

## LIVE-HYPOTHESES

- Decoder-derived edge state can improve `d_seg` at fixed model bytes when it
  changes the **joint proposal and allocation**, because learned video codecs
  benefit from representation-level context while sr1 only falsified post-hoc
  probability calibration.
- Edge-conditioned bit/capacity routing may dominate uniform precision because
  the score obligation is boundary-local and the counted byte budget can be
  redistributed without adding a transmitted mask.
- Rich carrier-to-semantic modulation may still help even though scalar pose
  sign failed, because a single sign destroys almost all geometry; the live
  hypothesis requires the full already-carried state inside joint training and
  must charge any learned mixing parameters.

## DEAD-ENDS

- Standalone additive causal-edge calibration on the F26 probability lattice:
  closed at FORMULATION scope because the full-n600 charged real-coder gain was
  only 2 B (0.00174%), far below the 1% falsifier, with weak held-out support.
- Standalone scalar pose-sign/delta-sign calibration: closed at FORMULATION
  scope because the selected full-n600 candidate cost 43 B and its odd-frame
  holdout was negative before table charge.
- Explicit PE3/contour/mask transmission on this vehicle: do not retry; prior
  receiver-closed evidence was scorer-negative on all 32 pairs and cost 74,408 B.
- Exact causal partition-codec replacement: do not retry; CPC1 already measured
  255,288 B at n600, more than twice the live PR130 semantic stream.
- Checkerboard or group reordering as a standalone rate idea: no new side
  information is created, while HPAC already consumes causal group context.
- Treating CR1's selected-support compression or foreign-paper percentage gains
  as a full-wire projection: invalid because neither measures this complete
  probability object and receiver.
