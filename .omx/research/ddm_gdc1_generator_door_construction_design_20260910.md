# ddm_gdc1 — construct the program, then count what it cannot generate

`[no-triality] [p0-ledger-ok]` · owner `ddm_gdc1` · `research_only=true` ·
axis `[macOS-CPU scorer-free exact-field measurement, n600]` · score claim false.

## Result first

The charter's typed 137,986 B held-distortion cap is stale at move 43. Re-deriving from the actual
move-43 components gives a strict archive cap of **154,507 B**, so the live rate demand at held
distortion is **25,959 B**, not 42,480 B. The replaceable token-tail envelope is 119,969 B, leaving a
strict **94,010 B** program-plus-correction gate.

The no-training ordered-scanline construction is **FORMULATION-NO-GO**. Its best full-n600 point is
K=6: **223,494 B program + 105,628 B real coded residual = 329,122 B**, which is **235,112 B over** the
94,010 B gate. It exactly closes back to the move-43 token field, but loses on bytes. No scorer,
training, Modal, MPS, candidate archive, or pointer move occurred.

The surviving construction is a categorical Cool-Chic-style distillation of the measured K=8
scanline teacher. If it reproduces that teacher, its target mismatch count is already known:
88,304. It must compress the teacher's 306,042 B scanline packet to at most **68,322 B** before the
transferred 0.2909 B/mismatch screen can pass. This is a concrete training construction, not a bound,
but it requires MAIN GO and was not launched.

## Move-43 bar, re-derived

The pointer components reproduce the recorded score:

```text
r = 25 / 37,545,489 = 6.658589531221714e-7 S/B
D = 100*0.00010345 + sqrt(10*0.00000459)
  = 0.01711995387438173
S(180,466) = D + r*180,466 = 0.13728485570852753
B_real(<0.12) = (0.12-D)/r = 154,507.26560515573 B
strict integer archive cap = 154,507 B
```

At 154,507 B, S is 0.11999982314442906; at 154,508 B, S is 0.12000048900338219.
The charter's typed 137,986 B would score 0.10899916737989766 at this distortion, proving that the
number inherited an older distortion premise.

Source inspection of the physical move-43 archive gives:

| Current component | Bytes | Share of archive |
|---|---:|---:|
| Token arithmetic payload | 119,833 | 66.4020% |
| Complete replaceable tail envelope: token payload + 35 B mixer + 100 B residual + 1 B alignment | **119,969** | **66.4773%** |
| Fixed compressed-model/container remainder | **60,497** | 33.5227% |
| Archive | **180,466** | 100% |

The exact held-distortion successor inequality is therefore

```text
60,497 + P_program + R_exact < 154,507.26560515573
P_program + R_exact <= 94,010 integer bytes.
```

Using GF1's deliberately generous transferred price only as a first screen:

```text
P_program + 0.2909*M < 94,010.26560515573.
```

At the 47,779 B HG1 reference packet, the mismatch budget is 158,924. At proposed packet sizes of
50,000 / 56,000 / 60,000 / 64,000 B, the respective budgets are 151,290 / 130,664 / 116,913 /
103,163. The tail must shrink by 25,959/119,969 = **21.6381%**. The 0.2909 coefficient is not a
successor measurement; the scanline result below shows why every admitted candidate must also pay a
real coded residual.

## Gestalt: what the program must generate

The receiver cannot be assumed to possess a semantic partition, the original video, six pose targets,
or an oracle boundary graph. EB2 leaves it only the bytes in the new packet plus generic code, and the
current carrier remains a separately counted positioned coefficient lattice. A legal generator must
therefore produce the ordered five-class field from its own counted sufficient statistic. Persistence,
one shared scene, rigid translation, or a previous-plane warp cannot conjure births, occlusion changes,
thin Lane markings, or fine islands; those forms are already formulation-closed by EB2, GF2, and the
motion/repair work.

The retained HG1 output and move-43 field make the missing gestalt spatially explicit. HG1 misses
**1,331,953 / 117,964,800 = 1.12911%** cells. The error is boundary-heavy but not Lane-only:

| Target class | HG1 mismatches | Share of HG1 mismatches |
|---|---:|---:|
| Road | 634,673 | 47.65% |
| Lane | 320,506 | 24.06% |
| Undrivable | 262,380 | 19.70% |
| Movable | 1,902 | 0.14% |
| MyCar | 112,492 | 8.45% |

Exactly 680,366 errors (51.08%) lie on a target crack edge; 931,011 (69.90%) are within one pixel,
1,039,973 (78.08%) within two, and 1,156,723 (86.84%) within four. No errors occur in rows 0–127;
967,259 occur in rows 128–255 and 364,694 in rows 256–383. Road plus Undrivable supply 897,053 errors,
the old horizon form's dominant debt. The movable-box renderer has only 1,902 target-Movable misses but
falsely paints **498,058** Road/Lane/Undrivable/MyCar cells as Movable. MyCar contributes another
112,492. The needed program is thus an occlusion-aware *partition generator*: it must place all-class
separatrices, preserve the narrow nonlinear Lane orbit, suppress false object fill, and represent bottom
hood shape. A more accurate horizon alone or a bigger class-blind block overlay cannot do this.

This is consistent with the earlier two-scale picture: coarse scene topology is cheap, while fine
boundary/island content is volatile. It is also consistent with the carrier work: the carrier buys a
particular spatial lattice point, not a generic low-rank subspace that can be replaced without checking
realized output. The generator and carrier are distinct obligations.

## Construction 1 — ordered scanline partition program (measured)

**Original vehicle form.** Each row is represented by at most K ordered constant-class runs. The encoder
uses an exact dynamic program over true row transitions to minimize Hamming error. The packet stores row
run counts, start classes, row-to-row delta-coded transition positions, and destination classes. The
receiver inverts deltas and rasterizes the complete field. This is not BND2/BND3's side channel for coder
misses and not WS1's persistent curve-identity worldsheet: it directly generates every token row and has
no incumbent plane beneath it.

**Anchor.** PolyLaneNet demonstrates that road-lane geometry admits small polynomial descriptions, while
DiffVG supplies the broader precedent for fitting and deterministically rasterizing filled polynomial
paths with occlusion. The GDC1 implementation deliberately tests the simpler integer scanline chart before
paying any differentiable-training burn: [PolyLaneNet](https://arxiv.org/abs/2004.10924) and
[DiffVG](https://people.csail.mit.edu/tzumao/diffvg/).

**Counted/free split.** All video-selected transitions and classes are counted. Only the generic exact-DP
fitter, delta transform, codec calls, and raster loop are free. No source-selected
constant is embedded in code.

**Pre-run prediction.** The source-only census found 584,354 runs over 230,400 rows; 214,011 rows
(92.8868%) already have at most eight runs. The preregistered K=8 prediction was 80,000–300,000
mismatches, a 60–100 KB packet, and a central 132,362 B transferred-price total. The mismatch prediction
was sound; the packet prediction was not.

**$0 falsifier and composition.** Full n600 `count_nonzero(packet_render != move43_field)`, followed by
the same Brotli-q11/zlib-9/LZMA2-extreme coder roster and a real HG1 residual race. The packet plus exact
residual replaces the current 119,969 B tail; the 60,497 B non-tail and positioned carrier/model sections
stay fixed.

| K | Packet B | Mismatches | 0.2909×M B | Projected replacement B |
|---:|---:|---:|---:|---:|
| 4 | 133,426 | 673,602 | 195,950.82 | 329,376.82 |
| **6** | **223,494** | **234,295** | **68,156.42** | **291,650.42** |
| 8 | 306,042 | 88,304 | 25,687.63 | 331,729.63 |
| 12 | 421,886 | 9,311 | 2,708.57 | 424,594.57 |
| 16 | 459,394 | 561 | 163.19 | 459,557.19 |
| 24 | 475,002 | 0 | 0 | 475,002.00 |

K=6 wins the preregistered projected selection. Its real residual race measured 119,336 B
frame-raster, 116,072 B class-frame-raster, and **105,628 B tile64-time**. The winner costs
0.450833 B/mismatch, so GF1's transferred rate understated this residual by 37,471.58 B. The physical
replacement is **329,122 B**, 235,112 B over the gate. Every point's LZMA2 packet repeated byte-identically,
every receiver render equalled its fitted render, and the winning correction decoded exactly to the
move-43 field.

**Decode risk.** Maximum measured full-n600 token-field decode was **3.6342 s**, safely below 1,260 s.
RGB inflation/evaluator time was not measured because the byte gate failed before candidate build.

**Verdict.** `FORMULATION-NO-GO` for this exact-DP ordered scanline grammar and K roster. Packet bytes alone
exceed 94,010 B even at K=4. The negative does not close compression-aware transition-sheet factorization,
learned implicit programs, SDF atom dictionaries, or a different generated object.

## Construction 2 — categorical Cool-Chic distillation of the K=8 teacher

**Original vehicle form.** Overfit one tiny multiresolution decoder to output five token logits over
`(pair,y,x)`, but distill the already-retained K=8 scanline render rather than RGB. The teacher has only
88,304 target mismatches. The construction asks whether a learned implicit description can remove the
scanline packet's explicit address cost while retaining its useful field. This is not the earlier Pact
Cool-Chic RGB substrate or a PR106 residual sidecar; its output object is the move-43 categorical token
plane, and it changes the live tail leg directly.

**Anchor.** COOL-CHIC is an open-source overfitted coordinate codec whose receiver evaluates a compact
learned mapping at pixel coordinates; the original paper reports a 629-parameter decoder. The relevant
mechanism is the content-specific decoder, not its RGB loss: [paper](https://arxiv.org/abs/2212.05458),
[official repository](https://github.com/Orange-OpenSource/Cool-Chic).

**Counted/free split.** Count multiresolution integer latent grids, quantized synthesis weights/biases,
scales, entropy tables, and any teacher-correction section. Keep only fixed interpolation, depthwise
convolutions, integer activation/argmax, and packet parsing in free code. Every trained value and selected
seed is counted.

**Predicted mismatch derivation.** Exact teacher reproduction fixes M at **88,304**. At that M the maximum
packet is `floor(94,010.2656 - 0.2909*88,304) = 68,322 B`. A central 60,000 B packet gives an 85,687.63 B
transferred-price total. Equivalently, the implicit decoder must compress the 306,042 B teacher program by
at least **4.4794×**. If decoder-versus-teacher error is E, target error is bounded above by 88,304+E;
the actual target `count_nonzero`, not this loose bound, decides admission. This is a hypothesis, not a
measured n600 neural point.

**Falsifier.** After a governed fit, decode the physical packet on all n600, count against the pinned field,
race a real exact residual, and require `packet + real_residual <= 94,010 B`. A packet-size proxy or entropy
loss cannot pass. No training was launched here.

**Decode risk.** The proposed integer decoder uses two 3-D latent grids and a narrow depthwise-separable
synthesis stack. A full n600 receiver decode must be ≤900 s before composition, reserving at least 360 s
for the existing runtime/evaluator path. The 1,260 s archive ceiling is absolute. The public COOL-CHIC
complexity result is only a plausibility anchor; it is not transferred timing evidence.

**Composition.** Replace the 306,042 B explicit K=8 scanline packet with the learned packet; retain the
same correction-to-move43 target and fixed 60,497 B non-tail. If the combined tail passes, the resulting
token field is bit-identical to move43 and held distortion is valid before any scorer call.

## Construction 3 — occlusion-aware tropical SDF atom dictionary

**Original vehicle form.** Learn a shared bank of local signed-distance primitives whose max-plus class
logits define the five-class partition. Per-pair integer coefficient/visibility codes move, gate, and
compose the atoms; explicit layer order handles occlusion. Road/Undrivable and MyCar use broad atoms, Lane
gets curve-tangent thin atoms, and Movable uses compact supported atoms. This differs from v8's one global
Laguerre diagram: atoms are local, class-protected, occlusion-aware, and temporally reused. It also differs
from GC1's class-blind dyadic paints and WS1's explicit full boundary graph.

**Anchor.** DeepSDF establishes compact learned continuous signed-distance functions; `msdfgen` is an
open-source deterministic multi-channel SDF rasterizer for vector shapes. These supply an implementable
implicit field/raster basis, not free source information: [DeepSDF](https://github.com/facebookresearch/DeepSDF),
[msdfgen](https://github.com/Chlumsky/msdfgen).

**Counted/free split.** Count all fitted atom coordinates/control points, class ownership, scales, per-pair
integer coefficient/visibility codes, and residuals. Generic SDF primitives, max-plus composition,
tile-culling, argmax, and integer rasterization are free.

**Predicted mismatch derivation.** The HG1 class ledger supplies a transparent, ambitious prior: remove
93% of Road+Undrivable errors, 85% of Lane errors, and 90% of MyCar errors while leaving Movable errors
unchanged. That gives
`0.07*(634,673+262,380) + 0.15*320,506 + 1,902 + 0.10*112,492 = 124,021` predicted
mismatches. At a proposed 50,000 B packet, the transferred total is **86,077.71 B**, under the gate; the
mismatch budget is 151,290. Those percentage reductions are HYPOTHESIS, not evidence.

**Falsifier.** Fit on the full n600 target, physical-code the bank and coefficient lattice with every fitted
parameter counted, render the real packet, and run the same count/residual race. Reject if actual packet plus
actual residual exceeds 94,010 B. A continuous SDF loss or a few-frame power-diagram fit cannot decide it.

**Decode risk.** Cap the bank at 96 atoms, carry integer bounding tiles, and require full-n600 raster under
600 s. A dense 96×117,964,800 evaluation is forbidden; receiver tile-culling must be proved from packet
bytes. The cap leaves existing runtime headroom.

**Composition.** Replace all four HG1 semantic streams with one local implicit partition program and exact
correction, while preserving the current positioned carrier/model leg. It attacks the generated object,
not the carrier's basis or the closed current-field arithmetic model.

## Governed GDC2 launch spec — written, not launched

**Status: SPEC-ONLY; STOPPED before training. MAIN GO is required.** Construction 2 is first because it has
a retained full-n600 teacher and an exact rate target; Construction 3 still depends on unmeasured classwise
reduction assumptions.

- Lane ID: `ddm_gdc2_categorical_coolchic_k8_distill_20260910`; owner MAIN-assigned training arm;
  `research_only=true`, `score_claim=false` until exact token closure and a separately allocated scorer.
- Inputs: move43 field SHA `78e57545…`, K=8 teacher render/packet facts from GDC1 RESULT, source commit
  containing the reviewed implementation, seed **20260910**, and no other source-derived constants.
- Output/custody root: `/Volumes/VertigoDataTier/pact/ddm_gdc2_categorical_coolchic_k8_distill/` with
  storage waterfall, predicted output plus 40 GiB reserve, certify-or-block, and no deletion.
- Fixed architecture: int8 latent `z0[75,24,32,4]`, int8 latent `z1[150,12,16,2]`; trilinear coordinate
  interpolation; five width-24 depthwise-separable 3×3 blocks; five-logit integer head. All latents,
  weights, biases, and quantization scales are counted. The physical raw sections are raced unchanged
  through the same Brotli-q11/zlib-9/LZMA2-extreme roster used here.
- Backend: MLX training is research signal only; a NumPy-fp32/integer receiver is verdict authority.
  Every stage checks MLX/NumPy logit parity and exact argmax identity. No MPS score claim.
- Determinism: seed Python/NumPy/MLX from 20260910, deterministic tile schedule generated once and retained,
  one recorded config, no unrecorded autotuning. Same config+checkpoint must reproduce packet SHA.
- Stage A: 10,000 AdamW steps, lr 3e-3 cosine to 3e-4, 16 seeded blocks of 8 pairs × 64×64 sites/step, categorical CE
  against the K=8 teacher. Stage B: 8,000 steps, lr 1e-3 to 1e-4, add measured symbol-rate proxy and sweep
  fixed lambda `{1e-4,3e-4,1e-3}` as three separately retained branches. Stage C: 4,000 QAT steps at lr
  1e-4 with integer receiver in the loop. EMA 0.999 is the exported authority at every stage.
- Resume: atomic optimizer/RNG/EMA/config checkpoint every 250 steps, distinct stage-end checkpoints, and
  `--resume-from` exact continuation. Never overwrite an earlier stage or branch. Every stage end retains
  raw latents/weights, all three coded packets plus repeats, full-n600 render, hashes, and decoder timing.
- Early stop: after Stage B, stop a lambda branch only if another retained branch Pareto-dominates it in
  both physical packet bytes and full-n600 target mismatches at two consecutive checkpoints. Numerical
  divergence also stops at the next checkpoint. Do not kill a branch from entropy loss or mismatch alone.
- Admission: `packet <= 68,322 B` at exact K=8 teacher identity is the clean screen. The authority gate is
  always `packet + best real exact residual <= 94,010 B`, receiver parse-back exact, deterministic repeat,
  and full-n600 decode ≤900 s. Only then may MAIN allocate a scorer/candidate archive step.

No operator burn authority was inferred from this memo. The launch remains queued, not fired.

## RECALL EVIDENCE

The charter seeds were read in their required order: GS3 Addenda 17–24, EB2, GF1, BZ2D §7, MD1–MD4,
the-cross, PC1/PC2, and the Lane dimensionality/rate memos. Own recall then searched:

- `.venv/bin/python tools/list_canonical_equations.py --json` for
  `generator|partition|lane|tropical|scanline|contour|spline|implicit|quadtree|boundary`;
- `.omx/research/`, arm receipts, specs, and designs by content for the same terms plus
  `Cool-Chic|worldsheet|Laguerre|SDF|static dynamic|topology`;
- `CANONICAL_RESEARCH_INDEX*` and `sub015_DAG_*` for generator-description and recursive-fractal FEEDs;
- canonical task/lane/live state for `ddm_gdc1|ddm_gf1|ddm_gb2|ddm_bnd1|ddm_bnd2|ddm_bnd3|Cool-Chic`.

Beyond-seed findings changed the construction slate:

1. GF2's global 923,953-mismatch lower bound closed one shared field plus rigid shifts; GC1's best dyadic
   overlay still cost 401,537 B with real residual. These removed static-scene and class-blind block forms.
2. LTG1 measured 221,717 B for exact Lane contour and 239,818 B for its best three-knot+residual form;
   WS1 measured 918,904 B for the optimalized explicit worldsheet family. These removed another explicit
   Lane/curve grammar from the candidate slate.
3. v8's global Laguerre diagram flattened near 0.7% disagreement while its per-class hybrid still left the
   boundary annulus. That changed Construction 3 from one global diagram to local class-protected SDF atoms.
4. BND2/BND3 found losing charged segment formulations but explicitly did not close the boundary family.
   That licensed a generated object while forbidding free edge/address assumptions.
5. The older Pact Cool-Chic lane reports contain a compact overfit-decoder signal, but no move43 n600
   categorical packet. They are a mechanism anchor only, not a transferable byte/mismatch result.
6. The external primary/official sources above confirm implementable coordinate codecs, polynomial lane
   models, differentiable vector rasterization, and SDF decoders. No external result supplies a byte price
   for this source object.

This bounded recall did not find an already measured move43 program-plus-residual below 94,010 B. It does
not claim global nonexistence.

## Retention, verification, and boundaries

- Move43 archive: `/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/candidate/candidate_runtime/archive.zip`,
  180,466 B, SHA `7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e`.
- Target: `/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/retained/fields/pass6.u8`, 117,964,800 B,
  SHA `78e57545439515eb29f806cc5a5f7d8b14acf658955cdd7561debb4edf3b7db6`.
- HG1 reference render: `/Volumes/APDataStore/pact/ddm_gf1_generator_form_on_lb1_field/retained/generated_from_lb1_fit.u8`,
  117,964,800 B, SHA `4026c4e2c805beb5b79be2879bb4a84311655d0d7d80dbc766654847522a5d19`.
- GDC1 store: `/Volumes/VertigoDataTier/pact/ddm_gdc1_generator_door/scanline_v1/`; 167 manifested files,
  1,648,591,638 declared bytes. RESULT SHA `3c249f8f…`; MANIFEST SHA `deead883…`. Independent rehash found
  zero missing/bad entries. Terminal resume reverified all six packet renders and exact residual closure.
- Source checks: Ruff clean; 6 focused tests pass; two review-tracker passes completed; 32/32 source/test
  entities report reviewed before serializer.
- The initial `nice -n 10` request was refused by the OS; computation continued at inherited priority. Two
  non-authority terminal-resume attempts then failed closed (fresh-launch free-space distinction, followed
  by mutable resume timing). The final terminal path performs read-only manifest and scientific replay.
- Not measured: SegNet/PoseNet/R, d_seg, d_pose, candidate archive bytes, contest CPU/CUDA score, or RGB
  inflate time. The field is a token-plane object; no score is inferred from token mismatches.

Equation leg: the held-distortion bar is exact application of the registered contest score law, not a new
universal equation. The six-point scanline curve is an empirical anchor candidate only; registering it as a
family law would overstate FORMULATION scope. Solver hooks for scorer sensitivity, tensor allocation,
posterior update, and deployment are N/A because this is research-only and losing. The consumer is the GDC2
construction gate and MAIN's generator-door routing.

Composition frontier remains **S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600] (move 43)**.
GDC1 did not move it and did not achieve sub-0.12.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN-assigned GDC2 arm; consumer store:
  `/Volumes/VertigoDataTier/pact/ddm_gdc2_categorical_coolchic_k8_distill/` plus this memo/receipt; fire
  trigger: MAIN explicitly authorizes the training burn against the fixed governed spec above. Implement,
  review, and run the categorical K=8 distillation; do not allocate a scorer unless physical packet plus
  real exact residual is ≤94,010 B and full-n600 decode is ≤900 s.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `.omx/state/main_hot_state.md`; fire trigger:
  harvest of GDC1's verified serializer artifact in a Git-writable context. Correct the stale 42,480 B
  generator-door premise to the move43 held-distortion demand of 25,959 B and mark the finite scanline form
  FOLDED without editing unrelated live-arm state.

## LIVE-HYPOTHESES

- A categorical Cool-Chic decoder can preserve the K=8 teacher's 88,304 mismatches while compressing its
  explicit 306,042 B addresses to ≤68,322 B. Overfitted coordinate decoders are built to exchange explicit
  samples for compact learned structure, and this teacher supplies an exact full-n600 target; the required
  4.4794× compression remains unmeasured and aggressive.
- Local class-protected SDF atoms may beat both global Laguerre and explicit worldsheet forms because 86.84%
  of HG1 errors lie within four pixels of a boundary, while the bank can share local boundary primitives
  temporally without storing a full persistent curve graph. Its 124,021-mismatch prediction depends on
  unmeasured classwise reductions and needs a separately authorized fit.
- The K=8 scanline point is a useful teacher even though its own packet loses: it already reaches 0.07486%
  field disagreement and isolates the remaining error mostly to Lane (63,143/88,304), creating a concrete
  target for an implicit address compressor rather than another vague generator-capacity argument.

## DEAD-ENDS

- **FORMULATION:** the exact-DP ordered scanline grammar at K=4/6/8/12/16/24 is closed. Every physical
  packet alone exceeds 94,010 B; the best complete point is 329,122 B with real residual.
- **Premise:** 137,986 B is not the move43 held-distortion archive cap, and 42,480 B is not the move43 rate
  demand. The correct strict cap/demand are 154,507 B and 25,959 B.
- **Transfer:** GF1's 0.2909 B/mismatch cannot be treated as the actual successor residual rate. K=6 measured
  0.450833 B/mismatch with the best tested real order.
- **Reuse:** persistence, one shared field plus rigid warp, class-blind dyadic capacity, one global Laguerre
  diagram, exact Lane contour, and the registered explicit worldsheet are already closed formulations in
  the recalled corpus. Repeating them under a new generator name does not reopen the door.
