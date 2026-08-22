# DDM XS1 cross-section conditioning — nonzero hindsight information, 45 B causal win

The legally pre-token sections contain measurable association with the DX2 token field, but the
incumbent learned context class captures almost none of it causally. The best real, zero-stored-state
extension is the fixed semantic-renderer probe: **113,732 B**, only **45 B** below the byte-identical
113,777 B control. That is **0.03955%** of the token stream and **0.10618%** of the 42,382 B fixed-
distortion demand, versus the charter prediction of at least 5,689 B. The prediction is falsified by
126.4x. No MAIN fire-order is emitted, no candidate is fired, and the frontier does not move.

`date_utc: 2026-08-22` · `arm: ddm_xs1_cross_section_conditioning` ·
`axis: [macOS-CPU advisory / scorer-free exact lossless measurement]` · `score_claim: false` ·
`promotion_eligible: false` · `Modal/scorer/Metal spend: $0` ·
`retention: /Volumes/VertigoDataTier/pact/ddm_xs1_cross_section_conditioning/measurement_v1`

## Result first

All rows code the exact same **117,964,800** tokens in the incumbent frame → 190-group → raster
order. Each challenger is the shipped 19-member learned corrector plus one cross-section member; its
tables start empty at encoder and decoder. The extended-model description cost is therefore **0 B**:
the rule and fixed quantizer are generic receiver code, and no learned table, fitted scalar, sidecar,
or per-position hint is stored. Each candidate was decoded independently back to TO2's source SHA-256
`cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`.

| extra conditioning member | plug-in `I(token; section | incumbent arg,qbin)` | real token bytes | archive Δ | net model cost | percent of token stream saved | percent of 42,382 B demand | projected decode wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| none — exact control | — | **113,777** | 0 B | 0 B | 0% | 0% | 498.000 s measured incumbent wall |
| compensated carrier RGB222 | 18,960.34 bits = 2,370.04 B | 113,752 | **−25 B** | 0 B | 0.02197% | 0.05899% | 499.767 s |
| selector mode | 459.65 bits = 57.46 B | 113,779 | **+2 B** | 0 B | −0.00176% | −0.00472% | 501.835 s |
| fixed all-zero semantic-renderer probe RGB222 | 78,359.68 bits = 9,794.96 B | **113,732** | **−45 B** | 0 B | **0.03955%** | **0.10618%** | **498.227 s** |
| measured joint carrier + probe + selector | 59,400.27 bits = 7,425.03 B | 113,747 | **−30 B** | 0 B | 0.02637% | 0.07078% | 501.320 s |

The joint row is measured directly. It is not the sum of the three legs, consistent with JX1's
measured warning that unions are not additive. Its feature is deliberately fixed and dense enough for
the inherited online estimator: carrier RGB222 and probe RGB222 are each reduced from 64 to 16 bins,
then crossed with selector mode capped at 3, for 1,024 joint bins.

The incumbent body is exactly reproduced: 113,777 B =
**0.007715996636284722 bits/position**, stream SHA-256
`e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5`; the repacked control archive
is byte-identical to DX2 at 180,368 B and SHA-256
`976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`. This satisfies the TO2/AD2
reference-form control before any challenger delta is admitted.

## Exact decode order and legal conditioning boundary

The receiver order was read from the shipped DX2 runtime, not inferred from section order:

1. `read_residual_archive` opens ZIP member `p`, decodes RX1/model framing, and returns the HPAC blob,
   semantic renderer weights, carrier/selector payload, compact residual table, compensation overlay,
   and RC64 token body.
2. Before token decode, `f26_inflate.inflate_archive` splits the selector from the carrier, restores the
   canonical carrier, decodes basis and coefficients, applies compensation, decodes semantic weights,
   and loads the semantic renderer.
3. `decode_production_tokens` then materializes HPAC, applies the compact residual table, and decodes
   the token stream over `g=(x mod 64)+2*(y mod 64)` in frame/group/raster order under the shipped
   19-member hit-event mixer plus within-miss law.
4. Only after token decode does `renderer.render_video` call the semantic renderer on the decoded
   token field. The selector is applied to rendered frame 0 after rendering.

Therefore the compensated carrier state, selector indices, semantic weights, HPAC state, and compact
residual table are legal pre-token inputs. Actual semantic RGB is **illegal and circular** because it
consumes the token field being decoded; final selector-modified frame-0 RGB is also post-token. XS1
uses a fixed all-zero-token semantic probe to expose position/frame information in the already-decoded
renderer weights without reading the target tokens. The full probe RGB, carrier RGB, and every fixed
categorical map are retained; no feature existed only as a scalar.

HPAC and the compact residual table are containment controls, not new challenger features. Both are
already inputs to the incumbent emitted probability row, so `I(token; HPAC/table | incumbent emitted
law)=0` by construction. Offering either again would duplicate the incumbent rather than expose a
cross-section dependency.

## What the mutual-information numbers mean

The table's MI statistic is the full-field plug-in conditional mutual information over exactly
117,964,800 positions. Its explicit incumbent context statistic is the class argmax and the shipped
64-bin surprise index recomputed from the **post-19-member emitted hit probability**. For section `Z`,
the retained counts are `N[C,Z,Y]`, and the reported quantity is:

`sum N(c,z,y)/N * log2(N(c,z,y) N(c) / (N(c,z) N(c,y)))`.

Those are hindsight descriptive upper bounds, not available coder bytes. They use counts from the
whole clip, have no prequential penalty, and would require a stored video-derived table if shipped as
fitted distributions. The real RC64 rows are the causal authority: empty tables, incumbent update
order, and no transmitted state. The 9,794.96 B semantic-probe association collapsing to a 45 B real
win says the association is largely static/hindsight, redundant with adaptive state at decision time,
or inaccessible to this one-member online estimator. It must not be booked as 9,795 B of supply.

The joint diagnostic being smaller than the semantic-only diagnostic is not evidence of negative
information. The joint feature uses a coarser fixed 1,024-bin projection to avoid a starved model; it
is a separate measured formulation, not a mathematical union identity.

## Same-class extension and rule-118 ledger

The reference implementation is the receiver's own `FreeCorrector`: 19 inherited hit-event members,
the inherited mixer contexts and weights, and the inherited within-miss law over the same 190 HPAC
groups. XS1 appends exactly one `MixerFamily`. Its cell is:

`(predicted_class, inherited_surprise_bin, legal_section_bin)`.

The new member starts at mixer weight zero, so every candidate begins at the incumbent law and must
earn its departure online. Its counts, hits, expected probability mass, mixer weight, causal spatial
state, temporal state, and within-miss state are captured structurally at every 25-frame checkpoint.
The frame-575 resume was exercised and completed bit-faithfully.

| form | already-decoded input read by generic code | stored addition | rule-118 verdict |
|---|---|---:|---|
| carrier | compensated basis/coefficients rendered at the token grid, fixed RGB222 | 0 B | legal/free |
| selector | exact decoded selector mode, broadcast over its pair | 0 B | legal/free |
| semantic probe | semantic weights + public coordinates + public frame index + fixed all-zero token input, fixed RGB222 | 0 B | legal/free |
| joint | fixed cross of the three legal maps | 0 B | legal/free |

The per-position maps retained for measurement are **not** candidate sidecars. Candidate receiver code
recomputes them from sections it already decoded. If a future implementation stored any map or fitted
table, those bytes would be charged and these 0 B entries would no longer apply.

## Decode cost and both score currencies

The five-path verification harness took 2,111.15 s because it ran five correctors/decoders serially
inside every group, flushed five 117,964,800-byte outputs, and wrote five checkpoint sets. That is not
a single candidate receiver wall. Timed single-path model+decoder components were 278.176 s for the
incumbent, 278.403 s semantic, 279.943 s carrier, 281.495 s joint, and 282.011 s selector. Conservatively
adding only the nonnegative component delta to the measured 498 s receiver wall gives 498.227–501.835 s,
leaving at least **1,298.165 s** under the 1,800 s budget. Decode time does not reject these rows; byte
utility does.

The best 45 B lossless saving is worth exactly
`45 * 25 / 37,545,489 = 2.9963652890497712e-5 S`, or **35.35 d_seg cells** at the n600 denominator.
At fixed DX2 distortion it would reduce the archive from 180,368 B to 180,323 B and the remaining
strict byte demand from 42,382 B to **42,337 B**. It is only 0.10618% of that demand. Read through the
zero-distortion reframe, it reduces the unphysical 149.7 B gap by 45 B but still leaves about 104.7 B,
before confronting the impossible premise of eliminating all distortion. No score was run and no
score movement is claimed.

## Adversarial assumption review

The shared assumption inside XS1 is narrower than RB1's independence assumption: useful cross-section
dependence must be visible through a fixed low-cardinality per-position feature and learnable by one
additional inherited hit-event member with the incumbent update law. Violating that assumption with a
generic nonlinear feature extractor, several complementary members, or a different causal recency law
could unlock part of the 9,795 B semantic-probe hindsight gap. That is plausible enough for the one
research-only XS2 queue below, but it has **zero banked credit** until a causal prequential diagnostic
recovers at least 5,689 B without stored state.

The second challenged assumption is that “0 B model description” could hide video-derived information
in receiver code. It does not here: the rules are fixed RGB bit truncation, a fixed all-zero token
input, public frame/coordinate inputs, fixed selector handling, and empty online tables. No fitted
constant, table, payload, or per-position code literal crosses into the receiver. The selected
formulation is design-time information, but the decoder algorithm remains generic under rule 118.

Finally, the retained `candidate_*.zip` files are rate objects, not standalone promoted submissions:
the unchanged shipped DX2 receiver does not contain the extra member. XS1's independent mirror decoder
proves the paired algorithm and stream invert exactly, but a future fire would still require a copied
candidate runtime, parse-back, and exact evaluation. Because no row meets the rate trigger, that
receiver integration was correctly not built and no archive is called frontier-ready.

## Verdict and disposition

`verdict_scope: formulation`

**Verdict: FALSIFIED at FORMULATION scope.** Every real extension lands within 0.04% of the 113,777 B
incumbent, far inside the charter's approximately 2% falsifier and nowhere near its 5% / 5,689 B
prediction. RB1's per-stream isolation assumption is operationally vindicated for these legal fixed
carrier, selector, semantic-probe, and measured-joint features under the incumbent one-extra-member
online model. This closes the named XS1 formulations; it does not prove mathematical conditional
independence against every possible generic feature transform or multi-member causal learner.

- **FOLDED (formulation):** carrier RGB222 member. Owner XS1; consumer
  `measurement_v1/RESULT.json`; reason: 25 B is 0.05899% of demand.
- **FOLDED (formulation):** selector-mode member. Owner XS1; same consumer; reason: real stream grew 2 B.
- **FOLDED (formulation):** all-zero semantic-probe RGB222 member. Owner XS1; same consumer; reason:
  best row saves only 45 B, 126.4x below the prediction.
- **FOLDED (formulation):** measured joint member. Owner XS1; same consumer; reason: 30 B, not the sum
  of leg-wise plug-in MI.
- **QUEUED-WITH-A-FIRE-ORDER (research only, not MAIN):** a successor may test whether the large gap
  between retained hindsight MI and realized bytes is estimator starvation by adding a fixed generic
  multi-member/recency law that reads the same legal maps and starts empty. Owner: successor XS2.
  Consumer store: `/Volumes/VertigoDataTier/pact/ddm_xs2_cross_section_prequential/`. Fire trigger:
  first prove on a retained causal prequential diagnostic that the new law can recover at least
  **5,689 B** with **0 stored bytes**, then run the same baseline-identity, n600 real recode, and full
  inversion gates. Otherwise fold without another n600 neural replay.

No sealed MAIN fire-order exists because no row meets the trigger. `main_fire_status=DO_NOT_FIRE`.

## Retention, controls, and boundaries

All artifacts live on the required Vertigo tier. `RESULT.json` SHA-256 is
`2ae20a1ff2cf38bebc293e7c11a2301c50d3be3e9e78a9ff4a45109bff8d5598`; the measurement tree is 2.7 GiB.
It contains the pinned inputs, full legal RGB probes, fixed maps, incumbent context map, every
candidate stream and archive, per-frame bit ledgers, final corrector states, every 25-frame checkpoint,
five independently decoded fields, per-block and aggregate MI counts, build inputs, and a complete
manifest. The final measurement script and all 32 Python files imported from the AP-hosted DX2 runtime
are separately hash-verified and copied under `retained/source/`; the prior reviewed script snapshot is
preserved under `retained/source/history/`.

The first completed encoding pass used `route_b`'s transport envelope
`R6D1 || raw body || u32 alignment` as though it were the RX1 token body. The required control refused
at byte 0. The entire failed payload set was retained under `retained/failed_framed_encode_v0/` with
`FAILED_FRAMED_CONTROL.json`. The correction reads the C encoder's authoritative raw byte count; it
does not strip trailing zeros heuristically. The corrected baseline body and archive then passed exact
SHA controls.

- **Measured here:** exact legal feature payloads; exact baseline stream/archive identity; four real
  RC64 candidate bodies and archives; five full inversions; plug-in conditional-MI counts over the full
  denominator; component decode time on this macOS CPU.
- **Projected, explicitly:** 498 s full-wall extensions from measured local component deltas; score-
  equivalent arithmetic for a hypothetical lossless fire. Neither is an exact contest row.
- **Not measured:** d_seg, d_pose, scorer output, contest CPU/CUDA time, any alternative feature
  transform, any multi-member successor, or any candidate after MAIN composition.
- **Negative scope:** formulation, on DX2, for the four fixed legal features and one-extra-member
  incumbent-class extensions. Not a family proof against all cross-section conditioning.
- **Custody:** shipped receiver and `upstream/` remained read-only; the JO r9 directory was not touched;
  no Modal, Metal, or scorer job ran.

## RECALL EVIDENCE

The recall searched `.omx/research/` and the canonical indexes/DAG/task ledgers for `113,777`,
`cross-section`, `mutual information`, `conditional entropy`, `19 members`, `RB1`, `CX3`, `TO2`,
`AD2`, `JX1`, `joint coding`, and `free corrector`; it also ran the canonical-equations JSON lister for
rate, entropy, and archive equations. Receiver custody was read directly from DX2's
`runtime/f26_inflate.py`, `runtime/residual_archive.py`, `runtime/free_corrector.py`, its FX1/FX2
parents, and `cpr1/inflate.py`.

Findings beyond the charter seeds changed execution:

- The older `ddm_xs1_cross_section_joint_coding_20260818.md` tested generic Brotli concatenation on a
  different SZ1 body and lost 205–472 B across all orders. That prevented drifting into byte-level
  section concatenation and kept the challenger inside the incumbent learned model class.
- RR1/RR2 showed that an empty, generic, already-decoded-state corrector can produce a real lossless
  win and supplied the exact mirror/checkpoint pattern. That changed the plan from an entropy-only MI
  memo to real RC64 bodies plus independent inversion.
- FS2's predictor traces share the DX2 probability digests but retain only argmax/u-index summaries,
  not the post-19-member emitted law. That ruled out treating those convenient arrays as the incumbent
  context and forced the exact replay that produced the retained emitted-context map.
- Receiver order made actual semantic RGB circular. This changed “renderer output” into the fixed
  all-zero-token probe of already-decoded renderer weights and excluded final selector output.
- JX1's consumed 3.705x non-additivity evidence changed the plan from adding per-section MI to a
  separately measured joint feature and real joint stream.

Source ownership remained explicit: TO2 owns the reused decoded checkpoint and 113,777 B control;
AD2 owns the implicit addressing/anatomy decomposition and receipt
`80124acd71ff63d4d9379b87674d1a976e1aa73857b4062a1c9ea2afb1b73511`; CX3 owns the named self-history
ladder and its 125,210 B best model-inclusive negative; RB1 owns the 0 B isolated coder allocation and
the assumption XS1 tested; JX1 owns the non-additivity warning. None of their borrowed numbers was
converted into XS1 supply.

**OWN-VEHICLE FRONTIER: DX2 remains S = 0.14821987563243377 @ 180,368 B `[contest-CUDA T4 n600]`; XS1 moved it by 0.**
