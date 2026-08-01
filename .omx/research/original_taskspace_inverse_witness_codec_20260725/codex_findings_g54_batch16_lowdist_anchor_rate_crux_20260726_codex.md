# G54 — Independent batch-16 replay corroborates the existing macro anchor

Date: 2026-07-26  
Lane: `lane_original_taskspace_inverse_witness_codec_capstone_20260726`  
Verdict scope: historical C1 full-lattice selected solution, exact realized bytes, frozen batch-16 scorer geometry  
Authority: mixed; rounded component cross-check is `[contest-CPU Linux x86_64]`, precise replay is `[macOS-CPU deterministic batch16 scorer advisory]`  
Research only: `true`  
Pointer delta: `0` — no changed archive and no frontier claim

## Outcome first

The canonical n600 batch-16 anchor already existed at
`.omx/research/original_taskspace_inverse_witness_codec_20260725/c1_live_target_debt_n600_batch16.json`
(SHA-256 `0db8e47a994cad5367e5eb3028055e667bc4caf3f174026d13171be662e7fbd3`).
It had already closed the central conclusion: the full-lattice solution is not
failing because its scorer cell is too expensive; it is failing because its
dense representation is approximately 2,184 times larger than the conditional
archive ceiling at the current `0.172` frontier target.

The new complete 38-stage batch-16 replay is an **independent reproduction**,
not a newly discovered coordinate. It measured:

- `d_seg = 0.00015196057211142033`;
- `d_pose = 0.0001018434704747051`;
- distortion score term `D = 0.04710898099989964`;
- strict `<0.172` ceiling `187,563 B`;
- strict `<0.15` ceiling `154,523 B`.

The replay differs from the prior canonical anchor by only
`-1.8054371678233316e-9` distortion score units. The historical C1 archive is
`409,526,925 B`. The forest-level decision remains: preserve the selected
evaluator preimage while replacing its storage coordinate. Do not spend the
next cycle trying to lower already-negligible distortion before the quotient
has a plausible path below roughly 188 KB.

The premise that a precise batch-16 anchor was absent was falsified. The
separate premise-falsification memo records the recall failure and routing fix.

## Evidence and authority correction

The durable replay is:

`/Volumes/VertigoDataTier/pact/c1_batch16_exact_replay_20260726/11_batch_replay_receipt.json`

- file SHA-256: `2d117579c13bf8209c3d6ed8d884f86b7f902a8e1b15a82e146ab4afbe2a49cf`;
- sealed receipt SHA-256: `691f641a19db499a4fd0a18d7226921bbc395a6d05e8186c1747c4faf3124949`;
- preflight SHA-256:
  `14967588535bb371a8d6de6fec29be1d153611d4feee5752aa62cefe4750b55f`;
- 38 atomic scorer stages, stage-root
  `c452d0fee233fcf18c9833114873ac005f28a2f261d3b80ef8325c6614b372d5`;
- exact archive/raw/source/upstream recursive closure was hash-bound;
- batch size 16, final partial batch 8, 600 pairs / 1,200 frames.

The v1 receipt calls itself `exact-upstream-mirror advisory`. That label is too
strong: the launcher explicitly set deterministic Torch algorithms, four Torch
threads, one interop thread, and disabled MKLDNN. It reproduced the frozen
scorer code, weights, batch geometry, aggregation DAG, and exact input bytes,
but not necessarily unmodified upstream host-kernel dispatch. The result is
therefore relabeled here as `[macOS-CPU deterministic batch16 scorer advisory]`.

The independent historical Linux x86-64 contest-CPU receipt corroborates the
same basin at its available display precision:

`/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719/modal_contest_cpu/harvest_fc01KXXRAR/contest_auth_eval.json`

It reports `d_seg=0.00015196`, `d_pose=0.00010184`, exact archive SHA
`e4cd154f79a30e2b1d759af0d26e54444d22807f81700565e475392eae064f42`,
and canonical score `272.73427793588485` from the rounded components. The
precise local values must not be promoted as Linux authority, but they are
well inside the rounded contest-CPU coordinate and are fit for allocator
planning.

One provenance deficit is explicit: v1 did not hash-close the replay harness
implementation itself. The completed result is retained because its 38 stage
receipts, exact inputs, frozen scorer closure, and independent contest-CPU
cross-check remain useful, but it is not promotion evidence. The replay tool is
now v2: it hash-closes its implementation, labels the axis correctly, owns
writable NumPy buffers, and records deterministic-kernel controls. Four tests
and Ruff pass.

## Geometry and arithmetic

The prior precise batch-32 anchor had
`D32 = 0.04710980004607969`. Batch 16 changes that by only
`-8.190461800519633e-7` score units. Thus batch geometry was a necessary
authority closure, not the missing structural gain.

For this measured coordinate:

`S(B) = 0.04710898099989964 + 25*B/37,545,489`.

Hence:

| bytes | score | decision |
|---:|---:|---|
| 187,563 | 0.17199948382435346 | strictly beats 0.172 |
| 187,564 | 0.1720001496833066 | does not beat 0.172 |
| 154,523 | 0.14999950401319692 | strictly beats 0.15 |
| 154,524 | 0.15000016987215004 | does not beat 0.15 |

At the V15 semantic-program reference size `133,941 B`, preserving this
distortion would score `0.1362947950400364`, leaving `53,622 B` to the 0.172
ceiling or `20,582 B` to the 0.15 ceiling. These are coupled counterfactual
budgets, not independent rate gates.

## Macro composition

The coherent codec stack is now a three-coordinate construction:

1. **Semantic program:** a fresh V15-like original base predicts the task-space
   trajectory compactly.
2. **Selected-preimage quotient:** encode only what moves the base into a cheap
   evaluator-equivalent cell. Use scorer asymmetry explicitly: temporal `Y1`
   semantic base and conditional `Y0 | Y1` pose enhancement.
3. **Exact generic realization:** V10 factor-2 integer machinery in the decoder
   expands compact selected state into the required 1,200-frame camera video.

The next arbitration is full n600 and whole-object: direct interleaved lossy
planes versus layered `Y1 + (Y0|Y1)` versus semantic-program residual. The
controller varies quantization, frame/group allocation, enhancement admission,
and entropy layout jointly; it does not impose arbitrary Seg, Pose, or rate
targets. Every point is evaluated by
`100*d_seg + sqrt(10*d_pose) + 25*bytes/37,545,489` after receiver parse-back.

If none enters the conditional rate neighborhood, its error decomposition is
the training target: learn only the irreducible quotient after analytic
prediction, factorization, temporal coding, and entropy contexts have removed
the describable portion.

## Triality

- DSL: the selected-preimage packet carries semantic base identity, analytic
  factor sections, optional learned quotient sections, and receiver guards.
- DAG: semantic base -> selected-plane predictor -> layered/direct codec race ->
  V10 integer realization -> 1,200-frame raw -> frozen scorer -> byte/score
  controller.
- Equations: `S(B)=D+25B/N`; selected-preimage optimization is
  `min_{program,quotient} S(R(decode(program,quotient)))`, with learned payload
  admitted only for the residual not captured by the analytic program.

## STORES CONSULTED

- `.omx/state/canonical_frontier_pointer.json`;
- C1 batch-16 replay preflight, 38 stage receipts, and final receipt on the SSD;
- historical C1 contest-CPU auth-eval receipt;
- G46 batch-geometry audit;
- G48 V15/MS1 coordinate compatibility audit;
- G49 selected-preimage program ABI;
- G50 lossy selected-preimage codec audit;
- `src/tac/contest_score.py` and frozen `upstream/evaluate.py` closure.

## Pointer-delta honesty

This work moves no pointer. It converts the last uncertain distortion premise
into an exact planning coordinate and makes the next build decision sharper.
Frontier lowering begins only when a fresh original receiver-closed archive is
below the dynamic pointer on authoritative contest CPU/CUDA evaluation.
