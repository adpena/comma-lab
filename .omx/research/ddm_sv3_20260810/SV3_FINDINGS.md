# DDM-SV3 unmeasured semantic screen

The joint vector/scale VQ32 sibling is damaged and withdrawn before the scorer. The prior SD1
mixed q3/q4 sibling survives the cheap screen but is not bankable until Pose and total score are
measured. The corrected bankable ledger therefore stays at **-2,424 B**, not -7,072 B or any
composition that includes an unmeasured semantic reduction. The PR130 base remains
`S = 0.172141297491896447 @ 191,052 B [contest-CUDA, DALI GT, n600]`; SV3 did not move it.

## Per-candidate result

All new measurements in this table are
`[scorer-free exact reconstructed weights and retained RGB24 RAW bytes; n600 pairs / 1,200 frames]`.
Even frames are the pose carrier; odd frames are the semantic render. The positive control is the
already-refuted low-rank candidate.

| candidate | semantic dB | composed archive B | weight rel-L2 | even mean abs | odd mean abs | odd RMS | odd max | odd bytes changed | odd RGB pixels changed | screen disposition |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| low-rank r32 control | -6,272 | 182,364 | 0.07860 | 0 | 67.6227 | 87.6419 | 244 | 99.2113% | 99.9993% | positive control fired; prior scorer refutation reproduced |
| joint vector/scale VQ32 | -4,648 | 183,992 | 0.09202 | 0 | 6.2547 | 8.1272 | 98 | 94.0310% | 99.9697% | **WITHDRAWN**, no scorer |
| SD1 mixed q3/q4 | -848 | 187,788 | 0.30840 | 0 | 0.9909 | 1.4496 | 25 | 64.0646% | 92.1321% | survives cheap screen; **QUEUED**, not banked |

All three candidates leave even frames byte-identical to the base. The instrument therefore
reproduces the known low-rank attribution and localizes both sibling effects to the odd semantic
frames. VQ32 crosses the predeclared raw gate (`mean_abs >= 5`, or widespread change with
`RMS >= 5`), so its post-hoc instance is not clean enough to consume a scorer pass. Mixed stays
below that gate. Its pixel changes are widespread, however, and this screen cannot tell whether
PoseNet amplifies them; only the matched full scorer can decide.

## Instrument resolution

- The weight pass compares every reconstructed fp32 value in all 38 tensors. It can see any
  stored-value change and reports per-tensor and global error. It cannot see activation sensitivity,
  rendered pixels, SegNet argmax, or PoseNet output. The 0.05 alert is triage-only.
- Weight L2 is not a safety ranking: mixed has the largest global value (0.3084) despite its prior
  measured favorable semantic-leg delta. That observed inversion is why neither sibling was killed
  from weights alone.
- The RAW pass compares all 3,662,409,600 RGB24 bytes, by parity, and can see exact byte/pixel
  changes and their magnitudes. It cannot see scorer sensitivity to small changes. Passing it means
  “not catastrophically damaged at this resolution,” not “score-safe.”
- The positive control fired at both levels. An instrument that missed it would have stopped the arm.

## Corrected bankable ledger

| candidate | section | delta bytes | status |
|---|---|---:|---|
| AI1 ANS + temporal reversion | tokens | -2,416 | BANKABLE, lossless RAW |
| HP3 requant frame-embed step2 | HPAC | -8 | BANKABLE, lossless RAW |
| SM3 pointwise low-rank r32 | semantic | -6,272 | REFUTED |
| SM3 joint vector/scale VQ32 | semantic | -4,648 | WITHDRAWN at RAW screen |
| SD1 mixed q3/q4 | semantic | -848 | QUEUED, pose and full S unmeasured |

**Bankable = -2,424 B**, a rate-only delta of -0.0016140421 S and 7.29% of the 33,252-byte
sub-0.15 rate target. The survivor is excluded until its matched total delta S is negative.

## Scorer routing

`SV3_SCORER_QUEUE.json` contains only the mixed survivor. SV3 did not claim the scorer slot. The
active-claims ledger still labels `lane_ddm_sd2_seg_decomposition_20260810` active, but the bounded
inspection of `/Volumes/APDataStore/pact/ddm_sd2_20260810/matched_local_n600/` found no advancement
beyond its retention preflight and a progress timestamp of 2026-08-10T04:01:45Z. That is not proof
the job is dead. MAIN must resume, complete, or release that claim before SV3 fires. If SD2 produces
retained component outputs on byte-identical RAWs first, SV3 folds them instead of duplicating work.

## Custody

- Full screen: `/Volumes/VertigoDataTier/pact/ddm_sv3_20260810/final/screen.json`, 68,699 B,
  SHA-256 `11d566d618f83e4190b8f135ca5c7f48ace08b60d15f300df2e44630aa62512a`.
- VQ32 RAW: 3,662,409,600 B, SHA-256
  `9dda985fcb3a492ff6ab661d0bdab035dd3ea0e3ab6d333209f3802f798c33b0`.
- Mixed RAW: 3,662,409,600 B, SHA-256
  `3319a2bddb98a93dc4552d1ccde8f404767bf3939985fae23c58098996ee541d`.
- Both candidate archives, repeat archives, model fields, token fields, parse-back states, token
  checkpoints, logs, RAWs, and receipts remain under
  `/Volumes/VertigoDataTier/pact/ddm_sv3_20260810/`. Nothing materialized by SV3 was discarded.
- Reused receiver/materializer: `experiments/ddm_cp2_composition_receiver_and_harness.py` at
  commit `58d270898002cde052b4ad34506b14984db06d49`, source SHA-256
  `194120dd62cf36563514130b3d31af0cc0d630d67f58c15687010f3369401d4e`; runtime sources are pinned
  in each build receipt. Reused SM3 commit `d3650d6c68764385cad2d32faa394af7c87360c6` and SD1 commit
  `600af8ef7d5f4573f6b3793d7a946fe5bf10d4d5`.

## RECALL EVIDENCE

The recall searched the full required surfaces before adjudication:

```text
.venv/bin/python tools/list_canonical_equations.py --json | rg -i 'semantic|quantiz|per.tensor|allocation|low.rank|codebook|vector|scale|pr130'
rg -n -i 'ddm_sm3|ddm_sd1|vector_scale_vq32|pointwise_lowrank_r32|selected_mixed_n600' .omx/research/CANONICAL_RESEARCH_INDEX* .omx/research/sub015_DAG_* .omx/state .omx/research/*SPEC* .omx/research/charters
rg -n -i 'vector.?scale|VQ32|mixed q3.?q4|pointwise low.?rank|SM3R|selected_mixed_n600' .omx/research src tools experiments reports docs .omx/state
```

Beyond the charter seeds, the search found that SD1 had already measured mixed's semantic leg,
SM3 had already recorded the weight-L2 inversion that makes weights non-decisive, CP2 had already
landed exact receiver-readable composed archives, and SD2 had already claimed the same mixed scorer
surface. Those facts changed the plan: SV3 reused the real receiver, treated weights as triage rather
than a verdict, decoded only the two owed siblings, and queued no duplicate scorer job. The canonical
equation registry did not supply a PR130-specific score-safety law for these post-hoc representations;
no equation was added.

## Boundaries

- Measured here: exact reconstructed-weight errors, real receiver-decoded RAW hashes, full n600
  parity statistics, complete archive bytes, and positive-control behavior.
- Consumed without remeasurement: low-rank's prior `[macOS-CPU advisory; AV GT; n600]` scorer
  refutation and SD1 mixed's prior `[macOS-CPU advisory; retained official-Ada target; n600]`
  semantic-leg delta.
- Not measured: new SegNet or PoseNet outputs, total S for mixed, contest-CPU, contest-CUDA, or any
  frontier move. VQ32's negative is instance-scoped to this post-hoc joint 32-entry formulation.
- Mission result: the exact pointer did not move. This screen prevented one bad scorer spend and
  routed one survivor; it did not achieve the sub-0.15 end.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — owner: MAIN scorer owner; consumer store: `/Volumes/VertigoDataTier/pact/ddm_sv3_20260810/scorer/sd1_selected_mixed_q3q4/`; fire trigger: MAIN resolves the existing SD2 claim, no conflicting n600 scorer job remains, SV3 claims the sole lane, and storage/retention preflight passes.** Run or fold the matched full-n600 mixed row; bank the 848 B only if total delta S is negative.
- **QUEUED-WITH-A-FIRE-ORDER — owner: MAIN exact-eval owner; consumer store: `/Volumes/VertigoDataTier/pact/ddm_sv3_20260810/exact_eval/sd1_selected_mixed_q3q4/`; fire trigger: the matched local total delta S is negative, exact receiver/archive hashes remain pinned, the contest lane is claimed, and remote execution is authorized.** Run the exact retained archive through `upstream/evaluate.py` and retain all payloads.
- **QUEUED-WITH-A-FIRE-ORDER — owner: PR130 resumable-QAT successor; consumer store: `/Volumes/VertigoDataTier/pact/ddm_sv3_20260810/vq_qat/`; fire trigger: the mixed full row fails or leaves a material rate gap, and a compatible resumable q4 checkpoint plus counted SM3R receiver is available.** Test representation-aware vector/scale VQ with preserved per-stage checkpoints; do not reopen the refuted post-hoc VQ32 instance.

## LIVE-HYPOTHESES

- Mixed q3/q4 may remain favorable after Pose because its odd-frame perturbation is low magnitude
  (mean 0.991, RMS 1.450) and its prior semantic-leg delta is already negative. This is plausible,
  not measured; PoseNet consumes continuous pixels.
- Representation-aware VQ training may rescue the VQ family because this screen refutes only a
  post-hoc codebook over a q4-trained checkpoint. Joint QAT could shape the renderer around the
  codebook, but no bytes or score are banked for that hypothesis.

## DEAD-ENDS

- Banking VQ32 from its -4,648-byte count is closed: its odd frames change at mean 6.255 with
  94.03% of bytes affected, while even frames are exactly unchanged.
- Using weight L2 as the screen verdict is closed: mixed has 0.3084 relative L2 and is far cleaner
  in RAW than VQ32 at 0.0920 or low-rank at 0.0786.
- Re-running the low-rank scorer is closed: the cheap instrument reproduces its exact parity
  signature and the prior full scorer already refuted it.
- Treating mixed as bankable before Pose is closed: the RAW screen cannot measure PoseNet.
