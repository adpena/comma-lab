# DDM-SM3 semantic-section representation receipt

DDM-SM3 completed the scorer-free half of its charter and queued the scorer-bound half. It
decomposed the shipped semantic state exactly, materialized eight complete deterministic
archives, retained every packed payload, and proved archive parse-back plus 38/38 tensor equality
to each candidate's packer state. It did not run or race the scorer slot held by `ddm_ai1`.

There is no winner and no score claim. The six new `SM3R` candidates have real bytes but no
measured `d_seg` or `d_pose`; selecting the smallest archive would be fake. The PR130 base remains
`S = 0.172141297491896447 @ 191,052 B [contest-CUDA, DALI GT, n600]`.

## Exact section decomposition

The apparent `66,339 * 4 / 8 = 33,169.5 B` estimate is not the shipped representation. The exact
40,252-byte semantic blob is:

| Stored object | Denominator | Raw bytes | Share of raw semantic |
|---|---:|---:|---:|
| Signed q4 codes for rank-at-least-2 tensors | 63,936 parameters | 31,968 | 79.42% |
| Per-axis fp16 q4 scales | 1,739 scale values | 3,478 | 8.64% |
| Rank-below-2 tensors stored directly as fp16 | 2,403 parameters | 4,806 | 11.94% |
| **Total** | **66,339 parameters** | **40,252** | **100%** |

The `7,082.5 B` gap from naive all-parameter q4 is therefore representational, not coder slack:
vectors are fp16 and every q4 tensor carries fp16 scales. Standalone Brotli q11 produces 35,033 B,
while the exact leave-one-out full-archive marginal is 36,580 B because the shipped model bundle
uses its own joint LZMA/outer framing context.

The raw bytes by exact tensor family are:

| Tensor family | Raw bytes |
|---|---:|
| `coord_mix.weight` | 4,992 |
| four `blocks.*.pw.weight` tensors | 19,200 |
| `frame_embed.weight` | 2,416 |
| four `blocks.*.film.weight` tensors | 4,608 |
| four `blocks.*.dw.weight` tensors | 2,496 |
| `head.weight` | 1,302 |
| `token_embed.weight` | 432 |
| all 22 rank-below-2 tensors | 4,806 |
| **Total** | **40,252** |

The complete per-tensor table, each raw payload, and each standalone Brotli-q11 payload are under
`/Volumes/VertigoDataTier/pact/ddm_sm3_20260810/final_v3/retained/baseline/`.

## Real archive race

All byte and rate columns below are measured from retained complete `archive.zip` files on the
`[scorer-free exact archive bytes and exact semantic parse-back]` axis. The break-even column is a
derived upper bound on permissible `d_seg` increase only if `d_pose` is unchanged; it is not a
measurement.

| Candidate | Representation | Archive B | Delta B | Rate-only delta S | Max delta d_seg if pose unchanged | Receiver |
|---|---|---:|---:|---:|---:|---|
| `legacy_q4_control` | shipped q4 | 191,052 | 0 | 0 | 0 | landed |
| `sd1_selected_mixed_q3q4` | four q3 tensors, twelve q4 | 190,204 | -848 | -0.0005646484 | 0.0000056465 | landed |
| `vector_vq32` | one fp16 32-value codebook plus 5-bit indices for 2,403 fp16 vectors | 188,124 | -2,928 | -0.0019496350 | 0.0000194964 | research only |
| `scale_vq32` | one fp16 32-value codebook plus 5-bit indices for 1,739 scales | 189,444 | -1,608 | -0.0010707012 | 0.0000107070 | research only |
| `vector_scale_vq32` | both codebooks | 186,404 | -4,648 | -0.0030949124 | 0.0000309491 | research only |
| `pointwise_lowrank_r32` | q4 SVD factors for `coord_mix` plus four pointwise matrices | 184,780 | -6,272 | -0.0041762674 | 0.0000417627 | research only |
| `film_row_prune_keep75` | retain highest-norm 75% of rows in Film blocks 1-3 | 190,328 | -724 | -0.0004820819 | 0.0000048208 | research only |
| `film_row_prune_keep50` | retain highest-norm 50% of rows in Film blocks 1-3 | 189,552 | -1,500 | -0.0009987884 | 0.0000099879 | research only |

The q4 control rebuilt byte-identically at SHA-256
`0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`.
Every candidate was independently rebuilt twice, and every archive retained its semantic bytes,
both complete archives, decoded canonical state, SHA-256, byte count, and receipt.

The landed SD1 evidence was folded instead of rerun. On its prior matched n600
`[macOS-CPU advisory; retained official-Ada target]` measurement, q4 had
`d_seg=0.00028616163465711804` and the selected mixed allocation had
`d_seg=0.00028756883409288193`; its semantic-leg delta was
`-0.00042392844867121244 S`. SR1 commit `58f62cd22f` now proves that counted format through the
receiver. Pose is still unmeasured, so even this existing row is not a full winner.

## Proxy diagnostics and decision boundary

Weight-space diagnostics were computed only from already-retained payloads. They are explicitly
not score authority. Relative L2 error versus q4 ranged from 0.00447 for 75% Film-row retention to
0.0920 for joint vector/scale VQ. Low-rank r32 was 0.0786 and removed 23.2%-27.4% of squared energy
from each selected matrix. Yet the already-measured SD1 candidate has a much larger 0.3084 relative
L2 error while barely moving `d_seg`. That observation closes weight MSE or singular energy as a
selection rule: only the frozen SegNet path can rank these candidates.

Consequently, `pointwise_lowrank_r32` is the byte leader, not the winner. The durable queue uses a
seeded stratified-random n120 screen, never a prefix, followed by n600 for every survivor. No new
format is receiver work until it survives that scorer gate.

## Custody and verification

- Full SSD result: `/Volumes/VertigoDataTier/pact/ddm_sm3_20260810/final_v3/SM3_RESULT.json`,
  SHA-256 `475630c49cbb2f65d2237a7fdd68afdf1324ad64f309c6196b912d2248432dea`.
- Exact decomposition: `/Volumes/VertigoDataTier/pact/ddm_sm3_20260810/final_v3/SECTION_DECOMPOSITION.json`,
  SHA-256 `91e8eea62d85fa97bebf17580e0e0375e679289f9be7ea88235fb4a8540e9ac8`.
- Materializer source SHA-256:
  `6da0aa13649f8f22082723d71e4844495fd9f9aa4897c1bcd87c55ecb81fa7b2`.
- Reused SD1 source: commit `600af8ef7d5f4573f6b3793d7a946fe5bf10d4d5`.
- PR130 intake source: commit `e34f31bc4969042c0051ac81aa3c56884419a231`.
- The final materialization and completed-stage resume replay both exited zero under `safe_run`.
- `tac.payload_retention_gate`: zero findings.
- Focused SM3 tests: 7 passed. Focused SR1/CX2 receiver tests: 8 passed.
- No source under `upstream/` or the read-only intake clone changed. Token bytes were unchanged in
  every rebuilt archive.

## RECALL EVIDENCE

The recall searched the full corpus, not only charter seeds: `.omx/research/` memos and arm
receipts, `CANONICAL_RESEARCH_INDEX*`, every `sub015_DAG_*` FEED, the canonical equation registry,
the task/hot-state and active-claim surfaces, Git history, PR130 intake source, packers, receivers,
and evaluator code. Representative content queries were:

```text
.venv/bin/python tools/list_canonical_equations.py --json | rg -i 'semantic|quantiz|per.tensor|allocation|low.rank|codebook|prun|PR130'
rg -n -i 'PR130.{0,80}(semantic|quant|low.rank|factor|codebook|VQ|prun)|SD1M|heterogeneous_per_tensor' .omx/research
rg -n -i 'shared.{0,30}(codebook|VQ)|cross.tensor|low.rank.{0,40}(semantic|renderer|weight)|structured.prun' .omx/research src experiments
rg -n 'ddm_ai1|ddm_sd1|ddm_sr1|ddm_cx2' .omx/state .omx/research
git log --all --grep='PR130\|semantic\|allocation\|receiver' -i
```

Beyond the seeds, recall found four load-bearing facts that changed the plan:

1. SD1 had already completed the real per-tensor q3/q4/q5 race and selected a measured four-q3
   allocation. SM3 folded that row instead of duplicating scorer work.
2. SR1 landed during this run and closed the counted SD1 receiver path. SM3 updated its receipt but
   did not touch SR1's shared runtime files.
3. Historical #336 predicted a favorable allocation from separable marginals but measured a large
   joint regression. Therefore every SM3 candidate is jointly replayed; tensor or proxy marginals
   cannot select it.
4. The earlier witness cross-tensor receipt closed only a post-hoc exact shared-codebook
   formulation on another checkpoint. It did not kill lossy VQ-in-loop or this PR130 object, so
   SM3 measured VQ32 bytes but withheld a verdict pending the scorer.

The bounded corpus search did not find a PR130-specific measured low-rank, shared-VQ, scale-VQ, or
structured-pruning row before SM3. That is scoped absence in the listed corpus, not a claim of
global nonexistence.

## Boundaries

- Measured here: exact raw section anatomy, real complete archive bytes, deterministic double
  builds, unchanged non-semantic bytes, semantic parse-back, 38/38 tensor equality to each packer
  state, and retained custody.
- Consumed but not remeasured here: SD1's q4 and selected-mixed n600 `d_seg` rows and SR1's receiver
  proof.
- Not measured: new-candidate `d_seg`, any candidate's paired `d_pose`, full `S`, exact
  `upstream/evaluate.py` for a new candidate, contest-CPU, or contest-CUDA.
- Verdict scope: no representation family is killed or adopted. Each new row is an unmeasured
  instance. Uniform post-hoc q3/q5 are closed only on the SD1 master.
- Mission result: the exact pointer did not move. This arm produced means and a queued measurement,
  not the sub-0.15 end.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — owner: MAIN scorer owner; consumer store: `/Volumes/VertigoDataTier/pact/ddm_sm3_20260810/final_v3/scorer/n120/`; fire trigger: `ddm_ai1` releases the scorer slot, the active-claims ledger has no conflicting scorer job, and the DDM-SM3 lane is claimed.** Run the fixed seeded-stratified n120 q4-versus-six-candidate `d_seg` screen from `SM3_SCORER_QUEUE.json`, retain all outputs, and admit only negative semantic-leg rows.
- **QUEUED-WITH-A-FIRE-ORDER — owner: MAIN scorer owner; consumer store: `/Volumes/VertigoDataTier/pact/ddm_sm3_20260810/final_v3/scorer/n600/`; fire trigger: at least one new candidate passes the n120 gate and the scorer lane remains exclusively claimed.** Rerun every survivor at n600 in chunks no larger than 120 with the matched q4 control.
- **QUEUED-WITH-A-FIRE-ORDER — owner: PR130 receiver/runtime successor; consumer store: `/Volumes/VertigoDataTier/pact/ddm_sm3_20260810/final_v3/receiver/`; fire trigger: a new `SM3R` candidate has a negative measured n600 semantic-leg delta.** Add only that winning mode to the counted DV1 receiver and prove legacy, SD1M, and selected-mode parse-back without editing `upstream/`.
- **QUEUED-WITH-A-FIRE-ORDER — owner: MAIN full-candidate evaluator; consumer store: `/Volumes/VertigoDataTier/pact/ddm_sm3_20260810/final_v3/evaluation/`; fire trigger: the winner is receiver-readable, paired pose replay leaves total delta S negative, the scorer lane is claimed, and any remote execution has operator authorization.** Run the paired n600 exact evaluator row, recompute S from components, and retain the exact archive and receipts.

## LIVE-HYPOTHESES

- `vector_vq32` is the cleanest first survivor candidate: it saves 2,928 complete-archive bytes,
  changes only the 2,403 fp16 vector parameters, and has 0.00665 relative L2 error. It can tolerate
  a measured `d_seg` increase of 1.95e-5 if pose is unchanged, about 6.8% of q4 `d_seg`.
- `film_row_prune_keep75` may preserve the semantic partition because it removes the lowest-norm
  rows and has the smallest weight-space perturbation, but its 724-byte saving permits only a
  4.82e-6 `d_seg` increase. Its plausible mechanism and tight budget make the scorer decisive.
- `pointwise_lowrank_r32` may exploit redundancy across the five largest matrices and has the
  largest 6,272-byte saving, but its 23%-27% selected-matrix energy loss makes it high-risk. It is
  still worth the fixed n120 race because SD1 demonstrated that global weight error is a poor
  predictor of semantic argmax behavior.
- Joint vector/scale VQ may trade its 4,648-byte saving for acceptable scorer drift after
  representation-aware QAT; the post-hoc row is a necessary screen, not a verdict on trained VQ.

## DEAD-ENDS

- `66,339 * 4 / 8` as the semantic-section price is closed: exact bytes show 4,806 B of fp16
  vectors and 3,478 B of scales in addition to 31,968 B of q4 codes.
- A better lossless coder over the unchanged semantic state is closed by the existing memoryless
  and real-coder receipts; this charter must make a smaller decoded object.
- Choosing the smallest archive or the lowest weight MSE is closed: neither measures SegNet or
  PoseNet, and SD1 already demonstrates a large inversion between weight L2 and `d_seg`.
- Uniform post-hoc q3 and q5 on the PR130 master are closed by SD1's n600 measurements; only its
  jointly replayed mixed allocation survived the semantic-leg gate.
- Treating another vehicle's exact shared-codebook negative as a PR130 VQ family verdict is closed;
  the prior verdict was formulation- and checkpoint-scoped.
- Running the scorer now is closed by ownership, not by scientific exhaustion: `ddm_ai1` holds the
  slot and the common contract requires DDM-SM3 to queue rather than race it.
