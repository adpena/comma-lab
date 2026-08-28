# ddm_qbt2 class-birth curriculum — early formulation falsifier

Date: 2026-08-27  
Disposition: **EARLY FORMULATION FALSIFIER; TYPED REFUSAL; NO BUILD; NO FIRE ORDER**  
Verdict scope: **FORMULATION** — the chartered frozen-terminal-head, closed-form RGB-prototype initialization through the real SegNet path. The QBFLOW family and a birth-first curriculum using an honestly nonlinear prototype solve remain open.  
Authority: structural/source proof on the frozen real-path implementation; no scorer row, training, Metal, Modal, contest evaluation, or score claim.

## RESULT

The required Build A operation is not a valid operation on the cited artifacts. The frozen SegNet terminal head has a deterministic closed-form five-cell prototype construction only in its four-dimensional **feature quotient**. Those prototypes are shape `(5, 4)` and are inputs to a reduced affine head model. They are not RGB triples and there is no head-only inverse that maps them through SegNet's nonlinear RGB-to-feature body.

The real RGB prototypes cited as precedent were not solved closed-form:

- FP1's `(5, 3)` palette is optimized with Adam and per-pixel cross-entropy through the complete frozen CPU SegNet on sampled GT. Its memo labels the result “32-pair/100-step CE.”
- SQ1's “margin-optimal paint” is an Adam solve over a dense pixel delta through the complete frozen SegNet, with two starts. It is neither a closed-form class palette nor a terminal-head inverse.
- v14 hard-codes the counted Movable RGB triple `(107, 0, 114)` and inherits the other role colors from an existing receiver. It does not construct five frozen-head-derived RGB prototypes.

The scorer-semantics premise is also misstated: `SegNet.preprocess_input` selects the last frame and bilinearly resizes it. It performs no normalization. The nonlinear `smp.Unet('tu-efficientnet_b2', classes=5)` body then maps RGB to the terminal head's 144-channel convolutional feature domain.

QBFLOW adds a second type mismatch. Its native `class_logits` are computed from coarse logits and signed interfaces, but the RGB renderer does not consume them. RGB is produced by `render_state -> render_out_w/render_out_b -> sigmoid`, with six outputs for two frames. Replacing only that output mapping could fit a **given** palette to observed render states, but it cannot, from the frozen terminal SegNet head alone, guarantee that every native class region paints its prototype. Such a fit would be a separate data-dependent regression, not the chartered closed-form construction.

Implementing any of the following under the requested name would therefore be a NO-FAKE violation: relabeling FP1's trained palette as closed-form; casting the four-dimensional quotient prototypes as RGB; replacing the nonlinear SegNet body by a local linear proxy; or calling an optimizer/exhaustive RGB search a closed-form frozen-head solve.

The charter explicitly permits an early FORMULATION falsifier report instead of the normal deliverable. That branch fires here, before a scorer invocation. Build B is gated on Build A, so no CE stage, config/schema change, storage launcher change, bounded smoke, or sealed r3 request was created.

## EARLY-GATE RECEIPT

| Requested observation | Result | Boundary |
|---|---:|---|
| Closed-form frozen-head RGB prototypes | **REFUSED** | The real closed-form bank is `(5,4)` in terminal-head feature-quotient coordinates, not `(5,3)` RGB. |
| 32-pair render through R/uint8/SegNet | **NOT RUN** | No valid initialized object existed; substituting a trained, hand-picked, searched, or proxy palette would change the mechanism. |
| Before/after per-class table | **NOT AVAILABLE** | No scorer row was measured. A fabricated or inherited table would be false evidence. |
| Road evoked at initialization | **UNTESTED** | This receipt does not claim Road cannot be evoked by an honest nonlinear initialization. |
| `>=4/5` classes with within-class error `<60%` | **UNTESTED** | The empirical gate was not entered. |
| Build B / stage-03a | **NOT AUTHORIZED BY THE FAILED GATE** | Birth-first CE remains a live mechanism under a corrected charter. |

This is a formulation/type falsifier, not an empirical QBFLOW distortion verdict. It does not promote the qbt1 instance result to family scope and does not close the CE-before-margin law.

## PRIMARY EVIDENCE

Arm-start HEAD: `afab7befbbfe991d86a6140212df969ff394e6b3`. HEAD advanced concurrently after the read phase; the serializer must bind this new file to its post-edit SHA rather than absorb any other path.

| Surface | Evidence | SHA-256 |
|---|---|---|
| Real scorer semantics | `upstream/modules.py`: SegNet is an EfficientNet-B2 U-Net; preprocessing is last-frame selection plus bilinear resize, with no normalization. | `065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa` |
| Actual FP1 RGB solve | `experiments/ddm_fp1_class_field_projection.py::solve_prototypes`: Adam minimizes CE through R and the full frozen SegNet on sampled GT. | `c2b22289480538bb8cb4db50a7a4e0b1d947e6c65245e09ad591d024b71fadf4` |
| FP1 receipt | `.omx/research/ddm_fp1_class_field_projection_20260731.md`: prototype row explicitly says `32-pair/100-step CE`. | `b594de4b53d58a1535466f8dc94f14b6fbb87c4d16d8be53b01089996aeef42d` |
| Actual SQ1 paint solve | `experiments/ddm_sq1_stage_decomposition_and_solved_paint.py::solve_margin_optimal_paint`: Adam over dense pixel deltas. | `79973df8b467fe87ba8e6b923d7fe10b7942207191fd1b81dc500e95da2af9f7` |
| Actual v14 color construction | `tools/measure_ddm_v14_realization_fidelity.py`: fixed Movable triple and inherited receiver colors. | `aa618c0a8cb9c34e197526d82fb49663ee7dd8e3a4ff070429b54f2bade650c1` |
| Genuine head-only closed form | `src/tac/boundary_math/prereq_surfaces.py::build_frozen_rank4_prototype_bank`: minimum-norm prototypes in a rank-4 affine quotient. | `9a1cc2df638410c579217f81d9030badc407d18e6c62022ccebda2e99c9de297` |
| QBFLOW renderer wiring | `experiments/ddm_qbt1_qbflow_trainer.py`: native class logits and RGB renderer are separate; RGB is a six-channel nonlinear readout of render state. | `fa5251eaf0ab81c8fa8dafb19b22780397d92374733efa8a978467179a67e5fe` |
| qbt1 build substrate | `.omx/research/ddm_qbt1_qbflow_trainer_build_20260827.md`. | `4d31cbcbcc1fec2ffd65a6b6614bffb3f1b768973bfe969cc90d6566306cedbb` |
| qbt1 working verdict observed | `.omx/research/ddm_qbt1_r1_r2_qbflow_verdict_20260827.md`; this file was already modified outside this arm and was not edited here. | `b546dafc57e7f61a22b6a5cb09b9b4cede39a597b92dedc6c236940bbc5f6c7e` |

No file under `upstream/` was changed. The frozen QBF1 ABI was not changed. The user's pre-existing dirty files and untracked WD3 runner were left untouched.

## RECALL EVIDENCE

Recall preceded the verdict and covered more than the charter's named seeds:

- Content searches over `.omx/research/`, `docs/`, `src/`, `experiments/`, configs, the canonical research index, `sub015_DAG_*` FEED blocks, and task-ledger surfaces used `QBFLOW`, `QBF1`, `class birth`, `prototype color`, `margin-optimal paint`, `CE births`, `event-triggered`, `rare class`, `Road`, `Lane`, `Movable`, `rank4`, and `SegNet head` queries.
- `.venv/bin/python tools/list_canonical_equations.py --json` was searched for class-birth, CE, margin, rank-4 head, and EMA surfaces. The registry contains the rank-4 terminal-head law but did not supply an RGB inverse through the nonlinear SegNet body.
- The actual FP1, SQ1, v14, rank-4 prerequisite-surface, QBFLOW trainer, and upstream scorer implementations were inspected rather than accepting memo labels.
- QBT1's build/verdict, the canonical index/DAG surfaces, the live board, and the common contract were checked for inherited authority and custody boundaries.

Beyond the charter seeds, FP1 supplied the decisive correction: its advertised “solved” RGB palette came from a 32-pair, 100-step CE optimization. The prerequisite-surface implementation supplied the second correction: the real closed-form frozen-head prototypes live in a four-dimensional affine feature quotient. Those findings changed the plan from “implement Build A and measure” to the chartered early-stop branch. The broader corpus also preserves class-birth and chroma/rare-class laws, but none creates a terminal-head-only RGB inverse; those laws therefore remain successor inputs rather than permission to fake this initialization.

## STORAGE, PAYLOAD, AND EXECUTION BOUNDARY

Live `df -k` at `2026-08-28T03:02:52Z`:

| Tier | Available | Capacity |
|---|---:|---:|
| `/Volumes/APDataStore/pact` | 64,520,320 KiB = 61.531372 GiB | 97% used |
| `/Volumes/VertigoDataTier/pact` | 8,754,128 KiB = 8.348587 GiB | 100% rounded used |
| `/Users/adpena/Projects/pact` | 371,917,840 KiB = 354.688492 GiB | 81% used |

The AP repack was changing free space during the read window, so these values are a timestamp-local observation, not a launch guarantee. No r3 demand projection exists because no valid r3 object was compiled. Consequently storage admission remains fail-closed: free space was not compared with an invented demand.

No run materialized a payload. Nothing was discarded, moved, or deleted. Nothing was written to either SSD. Invocation counts: CPU scorer 0; training 0; Metal 0; Modal 0; full-n600 scorer 0; contest evaluation 0.

## VERIFICATION AND REVIEW

- Source/mechanism cross-check: PASS. The three claimed RGB precedents were traced to their primary implementations.
- QBF1 ABI custody: PASS; no ABI file changed.
- `upstream/` custody: PASS; `git diff -- upstream/` was empty.
- Staged-index custody before the landing: PASS; the index was empty.
- Python review requirement: **NOT APPLICABLE**; this arm changed no `.py` file.
- Tests and bounded smoke: **NOT RUN**; there is no implementation to validate under the failed formulation gate.
- Score arithmetic: **NOT APPLICABLE**; no distortion, rate, archive, or score row was produced. `score_claim=false`.

## PROVEN LEGS THAT TRANSFER

The qbt1 pose and rate evidence is not invalidated. A corrected successor may retain the frozen QBF1 archive/coder, real render-to-R-to-uint8 scorer path, checkpoint/resume/EMA discipline, real per-checkpoint re-encode, and pose-active joint descent. What does **not** transfer is the claim that FP1/v14 provides a closed-form RGB initialization from the terminal head.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN QBFLOW curriculum owner; consumer store: `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/`; fire trigger: MAIN issues a corrected charter that explicitly chooses either (a) FP1's palette as a provenance-labeled, video-derived CE-trained inherited initialization plus a separately defined QBFLOW renderer fit, or (b) a newly authorized nonlinear frozen-SegNet RGB solve, then redoes the 32-pair `$0` gate, derives the full on-disk r3 demand, verifies live AP free space exceeds it, claims the Metal/scorer lanes, and only then permits stage-03a.

## LIVE-HYPOTHESES

- An honestly inherited FP1 palette may birth Road/Lane/Movable before QBFLOW training. This is plausible because the palette was optimized through the real frozen SegNet and R path, but it is video-derived and its transfer into QBFLOW's renderer has not been measured.
- CE on realized scorer logits before the margin law may birth all five classes while preserving the proven pose leg. This is plausible from the corpus's birth-then-sharpen law and qbt1's separate pose/rate success, but no qbt2 row tested it.
- A data-dependent fit from QBFLOW render states to an admitted RGB palette may be possible without changing QBF1 shapes. This is plausible because the existing six-channel output map has the right RGB dimensionality, but native class logits are not its direct inputs and exact region-wise fit is unproven.
- Joint pose loss during class birth may expose or refute a pose-seg interior conflict. This remains plausible because both objectives act through rendered interior photometry; qbt1 established correlation in the shared surface, not causation.

## DEAD-ENDS

- Calling FP1's palette “closed-form from the frozen head” is closed because its primary implementation uses Adam, sampled GT, and CE through the full scorer.
- Treating the rank-4 head prototypes as RGB colors is closed because their coordinate space is four-dimensional terminal-head feature quotient, not three-dimensional RGB.
- Citing v14 as an all-class closed-form color constructor is closed because its implementation hard-codes only the Movable triple and inherits the remaining colors.
- Solving the head while omitting the nonlinear SegNet body is closed for the requested real-path claim because it does not prove any RGB value reaches the desired terminal-head cell.
- Re-running qbt1's margin law from step zero is closed at INSTANCE scope because 4,670 flat realized steps already showed it does not birth the missing classes.
- Launching Build B after this failed Build A formulation is closed under the present charter because Build B is downstream of the `$0` gate.

Own-vehicle frontier unchanged: gb1 — S 0.14811799921260607 @ 180,215 B `[contest-CUDA T4 n600]`.
