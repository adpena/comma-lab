# ddm_pz4p — learned pose-gauge QAT pre-proof

**Verdict:** `PASS — FIRE pz4 to MAIN routing`  
**Axis:** `[macOS-CPU scorer-free banked-output MSE + exact real-coder rate envelope]`  
**Population:** all `600/600` pairs; reference shape `(600, 6)` float32  
**Scorer work:** zero PoseNet/SegNet forward passes; zero Modal jobs  
**Authority boundary:** representation pre-proof only; no receiver-closed candidate, rendered
`d_pose`, exact score, or pointer movement

## Result

| Gate quantity | Required | Measured winner | Verdict |
|---|---:|---:|---|
| Reconstruction MSE vs the pinned bank | `< 2.5e-6` | **`1.0985637375134246e-6`** | pass |
| Exact whole-container rate-envelope saving vs LC2 | `>= 2,000 B` | **`19,221 B`** (`187,226 -> 168,005 B`) | pass |
| Gauge bytes after the shipped carrier coder | — | **`3,837 B`** (Brotli 1.2.0 q9) | measured |
| Gauge wire bytes before the carrier coder | — | `5,588 B` | measured |
| Passing grid cells | — | `19 / 330` | measured |

The selected cell is `r6_b12_global`: full output rank `6`, signed 12-bit coefficients, one learned
global scale, eight hard quantizer-in-loop fit rounds, and an exact least-squares compensation solve
after every round. Its retained wire payload is SHA-256
`b903c7f0e6100e3602e414fbc261725aa5026fa6e1c6af8fba104ded867b9cac`; the real q9 stream is
`3,837 B`, SHA-256 `dcc60591992ec381f402f4923ec7b9efe2e80e311d39e2c12172018303d19acc`.
The deterministic rate envelope is `168,005 B`, SHA-256
`66d142c7db35f3762be4d810b7549bd94bebe77120fa5dd53173937d5c6d2620`.

This is also a `19,547 B` reduction relative to PK2's `23,384 B` measured pose marginal if the PGQ1
q9 stream is compared directly with that marginal. The load-bearing gate uses the stricter exact
whole-container delta, not this mixed-accounting comparison.

## What the surrogate means

The reference is the exact banked first six PoseNet outputs produced by the shipped LC2 carrier for
all 600 pairs. PGQ1 reconstructs those six values; it does not reconstruct CPR1 coefficients or
frames. Thus the measured MSE answers one question only: how many counted bytes are needed to preserve
the shipped carrier's output-side pose gauge under a perfect future consumer?

The rate envelope replaces CPR1 by PGQ1 inside LC2's exact selected semantic stream, exact selected
HPAC stream, ANS token stream, `split_brotli_cx2` packing, carrier Brotli q9, and deterministic stored
ZIP path. The control rebuilt the pinned LC2 archive byte-for-byte at `187,226 B`, SHA-256
`f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45`. The unchanged LC2 receiver
cannot parse PGQ1. Every file is therefore named `not_receiver_candidate`, and none is an archive or
score claim.

## Custody

- LC2 archive: `187,226 B`, SHA-256
  `f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45`.
- LC2 raw CPR1 carrier: `23,054 B`, SHA-256
  `a05d0985ca5a8d5110bd5bf5be39f238c6f89640b8a8bb888a3e1269bdf636e4`.
- LC2-native PoseNet output bank: `14,528 B`, SHA-256
  `23319e2f0406040ee5d9e904daacc1017f8da44a02e7c259055e72c937515312`; semantic array SHA-256
  `e80dcb0b4ce6afb7ac74db91dc29ce9cbbce09acbd0058f9450364a40f4ebfe2`.
- Bank receipt: SHA-256
  `4deb10bd46296f893e85bda024e572128b0abb303286b73144432c05560920fa`; it binds `600/600`, complete,
  retained outputs and the exact decoded LC2 raw provider.
- Authoritative result:
  `/Volumes/VertigoDataTier/pact/ddm_pz4p_20260811/preproof_v3/FINAL_RESULT.json`, `147,460 B`, SHA-256
  `40d788ed27e79610d66914972b1750a00115da32db07667217ee811c12c9699c`.
- Winner receipt:
  `/Volumes/VertigoDataTier/pact/ddm_pz4p_20260811/preproof_v3/candidates/r6_b12_global/receipt.json`,
  `6,742 B`, SHA-256 `30885816938a15e8f9b63a291ec6d60b190b0ea77a8939b0736ffe27f1f83f2d`.

The development v1 store, source-bound v2 store, and authoritative hardened v3 store are all retained,
`180 MB` each. V2 superseded v1 by binding the runner source SHA in every candidate and using explicit
finite deterministic matrix products. V3 supersedes v2 because it binds the final reviewed runner,
including canonical-config, finite-wire, source-immutability, path-containment, and free-space guards.
No earlier payload was deleted.

## Design and measurement

The measured grid is the full Cartesian product:

- ranks `1, 2, 3, 4, 5, 6`;
- signed depths `3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15` (all sub-int16);
- cells `global`, `per_rank`, `block100_rank`, `block50_rank`, `block25_rank`.

For each of the 330 cells, the fit started from the real bank's centered SVD. Training evaluated only
hard decoded quantized coefficients. Each of eight rounds alternated bounded scale adaptation, exact
floor/ceil rounding over all rank combinations per pair, and the mandatory PR133-style compensation
solve. The best hard state was retained. PGQ1 stores the scales, compensated map, and bit-packed signed
codes. The reported MSE is recomputed after a literal PGQ1 parse and float32 decode.

Every cell retained eight independently loadable stage checkpoints, raw and repeated PGQ1 payloads,
decoded output arrays, codes, scales, compensation state, q9 and repeated q9 streams, rate envelope and
repeat, plus a receipt with SHA-256 and byte count. A resume audit verified all 330 completed receipts.
An independent audit verified 3,300 retained file records and all 2,640 stage checkpoints, reproduced
the winner MSE exactly from a literal PGQ1 decode, and proved the winner's raw, q9, and envelope repeats
byte-identical.

### Rank falsifier

| Rank | Best measured MSE in its 55-cell slice | Gate status |
|---:|---:|---|
| 1 | `4.877486051217807e-4` | fail |
| 2 | `2.1486146179279815e-4` | fail |
| 3 | `6.639265231708469e-5` | fail |
| 4 | `1.511110584732217e-5` | fail |
| 5 | `5.202855334725472e-6` | fail |
| 6 | `3.818996557942058e-10` | pass |

Rank 5 is still `2.08x` above the MSE gate at 15 bits with 25-pair adaptive cells. The old rank-2
prediction is therefore falsified on this current surrogate instance. This does not kill low-rank pose
codecs on other pose tensors; it closes ranks `<=5` for this exact LC2-output reconstruction gate and
tested PGQ1 formulation.

## Research leg

The design follows four primary-source lessons:

- [LSQ](https://openreview.net/pdf?id=rkgO66VKDS) learns step sizes inside quantized training rather
  than fixing a post-training scale. This motivated scale adaptation inside every decoded-gauge round.
- [AdaRound](https://proceedings.mlr.press/v119/nagel20a.html) shows nearest rounding is not generally
  task-optimal. PGQ1 therefore searches floor/ceil assignments against output reconstruction error.
- [HAWQ](https://openaccess.thecvf.com/content_ICCV_2019/html/Dong_HAWQ_Hessian_AWare_Quantization_of_Neural_Networks_With_Mixed-Precision_ICCV_2019_paper.html)
  motivates heterogeneous precision/allocation for differently sensitive cells. The sweep prices global,
  per-rank, and three temporal cell granularities separately.
- [LQ-LoRA](https://openreview.net/forum?id=xw29VvOMmU) and
  [SVDQuant](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f34f0630c33be15b8c89426bb8056798-Abstract-Conference.html)
  support measuring low-rank and quantized components jointly. Here the evidence rejected low rank and
  selected full rank, which is why the grid rather than the literature prior is authoritative.

PR133/PR135 supplied the project-local controlling lesson: raw carrier-basis quantization worsened pose
about 29-fold until coefficients were re-solved. The compensation solve is therefore part of every PGQ1
fit round, not a post-hoc repair or optional ablation.

## RECALL EVIDENCE

Recall queried the full corpus with `tools/corpus_query.py` over `pose gauge`, `pose carrier`, `CPR1`,
`pk2`, `dxi`, `low-rank pose codec`, `semantic pose`, `quantization toolbox`, and explicit `#140/rank-2`
terms. It also searched `.omx/research/` receipts by content, canonical indexes, the sub-0.15 DAG,
canonical equation registry, task/queue ledgers, and live hot state.

Findings beyond the charter seeds changed the plan:

- PK3 had already priced 432 full-archive frozen post-hoc gauge candidates with the real coder. Its best
  admissible point saved only `64 B`, `1,936 B` short of this gate. That closed repeating reciprocal,
  permutation, and frozen-gauge searches, but its stated scope excluded learned QAT/retraining.
- The canonical index supersedes the charter's old “rank-2/254 is a safe 2.7x cut” shorthand. #140's
  final correction found rank-2 net-negative under the nonlinear pose term and only a modest `525 B`
  rank-4/511 Pareto win on a different FiLM-STORE tensor. This forced ranks 1 through 6 into the current
  grid and removed rank-2 from the prior.
- PK2 showed exact low-rank-plus-residual CPR1 cost `+4,316 B` and no frozen representation win. This
  moved PGQ1 to the output-gauge representation instead of repacking frozen CPR1 fields.
- FD135 showed the current carrier's learned-state movement dominates its small header gaps; PS135
  supplied the complete exact LC2 output bank and current carrier/coder custody. These facts made the
  banked-output surrogate and exact LC2 coder path the narrow lawful pre-proof.

## Verification and boundaries

- `19 passed` for the dedicated no-fake tests; Ruff check and bytecode compilation clean.
- The mandatory payload-retention detector reported zero findings for the runner.
- Runner SHA-256: `2896c98c372feeac224d91f29b6ed7ef189a71fa278c72b69845ebd6e234097f`.
- Test SHA-256: `391b07730fea8fd42141a84277d3d1c2b5576622c5f48a554252f4f83ebd49eb`.
- The code contains no scorer import or scorer execution surface. No MPS, Modal, renderer, decoder-output
  materialization, or `upstream/evaluate.py` call ran.
- The exact LC2 output bank is consumed read-only. No GT target was derived and no scorer lane was claimed.
- The stage checkpoints and all candidate payloads are on Vertigo with byte and SHA custody; the state
  file supports `--resume-from` and a completed resume audit passed.
- The rate envelope is exact bytes through the real shipped coder but deliberately not receiver-closed.
  It cannot establish that a compact renderer consumes PGQ1, that rendered frames reproduce its values,
  or that PGQ1 transfers unchanged from LC2 to CP135/PR135.
- No archive was submitted and no score was measured. The derived LC2 byte-only rate action is
  `-0.012798474937961256 S`, but it is not a score prediction because receiver realization is entirely
  unmeasured.
- Protected files, `upstream/`, the staged index, and unrelated dirty work were untouched.

## pz4 fire-ready row

**Disposition:** `FIRED-TO-MAIN-ROUTING`.  
**Owner:** `MAIN`, which must assign the full `ddm_pz4_joint_target_conditioned_receiver` arm and scorer
lane only after receiver preflight.  
**Consumer store:** `/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/`.  
**Satisfied trigger:** PGQ1 `r6_b12_global` pre-proves `19,221 B >= 2,000 B` exact rate-envelope savings
and `1.0985637375134246e-6 < 2.5e-6` decoded bank-output MSE.  
**Admission for any scorer fire:** a deterministic resumable receiver must consume the retained PGQ1
bytes, remove the frozen CPR1 pose carrier/residual rather than coexist with it, retain every stage and
payload, close exact parse/render on all 600 pairs, and beat its current-base full-archive ceiling before
claiming a scorer lane.

Own-vehicle frontier remains **cp135 `S = 0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`**.
This scorer-free pre-proof did not move it or itself reach sub-0.15.

## NEXT_IF_RESUMED

- **FIRED-TO-MAIN-ROUTING** — owner: `MAIN` assigning `ddm_pz4_joint_target_conditioned_receiver`;
  consumer store: `/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/`; fire
  trigger: already satisfied by retained PGQ1 `r6_b12_global` at `19,221 B` exact rate-envelope savings
  and `1.0985637375134246e-6` bank-output MSE; build the resumable receiver and refuse scoring unless it
  removes CPR1 and closes below the current-base archive ceiling.

## LIVE-HYPOTHESES

- A jointly trained PGQ1-conditioned receiver can turn most of the measured `19,221 B` envelope saving
  into a real rate win because the counted gauge preserves the shipped carrier's six PoseNet outputs
  below the preregistered surrogate MSE gate, while generic receiver code is free. This remains plausible
  but untested because no renderer has consumed PGQ1.
- Refitting PGQ1 to a banked CP135/PR135 output tensor may preserve a similar byte scale because the stored
  object is still only 600 by 6 values. It must be remeasured because the current proof is pinned to LC2,
  and no cross-lineage numeric transfer is allowed.

## DEAD-ENDS

- Rank-2 as the load-bearing current-gauge compressor: closed for this LC2 bank and PGQ1 reconstruction
  gate; its best 15-bit adaptive MSE is `2.1486e-4`, about 86 times the gate.
- Any rank at or below five: closed for this instance and formulation; rank 5 bottoms at
  `5.202855334725472e-6`, above the strict `2.5e-6` gate even at 15 bits.
- Frozen post-hoc CPR1 gauges and low-rank-plus-exact-residual repacks: do not retry; PK3 saved only `64 B`
  and PK2's exact low-rank-plus-residual form cost `+4,316 B`.
- Treating the retained rate envelope as a submission candidate or score row: forbidden; the unchanged
  receiver cannot parse PGQ1, and no rendered PoseNet output has been measured.

## ADD.1 (2026-08-11, MAIN) — SUPERSESSION: the 168,005 B envelope is NON-RENDERABLE (pz4r)

ddm_pz4r (commits c7b9387b96 / 5c962530fe, memo `.omx/research/ddm_pz4r_pgq1_receiver_20260811.md`)
built the real receiver and adjudicated this memo's headline: **the 19,221 B envelope does not
survive realization.** FORMULATION verdict: PGQ1 (or lc2 minus CPR1 plus PGQ1) cannot render —
the removed CPR1 carrier holds basis/coefficient data the decoder requires. The realized
residual-free archive is **183,137 B = −4,089 B vs lc2** (21.3% of the envelope), sha
c408adf910…, gauge `target_quadratic_previous_f10_q20`, coefficient surrogate R² 0.4861.
Read every "19,221 B" / "168,005 B" figure above as a NON-RENDERABLE upper bound on the
container arithmetic, never a candidate saving. The pre-proof's own boundary paragraph
("forbidden; the unchanged receiver cannot parse PGQ1") was correct and is now the operative
scope. Realized (d_seg, d_pose) on the 183,137 B candidate: QUEUED behind the ps135b scorer
lane (fire-order in pz4r's NEXT_IF_RESUMED).
