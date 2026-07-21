# M1 byte-close first n600 row — FORMULATION-NEGATIVE efficacy, rate-only win vs scored control

**UTC:** 2026-07-21T03:00:00Z  
**Lane:** `m1_byteclose_closer`  
**Verdict:** `MEASURED_FORMULATION_NEGATIVE_EFFICACY_RATE_ONLY_IMPROVEMENT_VS_SCORED_CONTROL`  
**Verdict scope:** `FORMULATION` — the first positive-anisotropic M1 firing after three one-epoch stages is receiver-identical to the scored 94,344-byte control and improves neither `d_seg` nor `d_pose`; no banded-generator, windowed-curvelet, or shared-receiver family negative  
**Authority:** `[macOS-CPU advisory]`; exact hard CPU-Torch through-R measurement, not a contest score  
**Pointer:** `0.19108 [contest-CPU]` **UNMOVED**

## Outcome

The 90,566-byte composed M1 candidate beats the 94,344-byte control on the local advisory action by exactly the rate term. It decodes to the **same 3,662,409,600 receiver bytes** as the control, so `d_seg`, `d_pose`, every per-class contribution, and every per-pair tail value are identical. The candidate lowers the counted archive by 3,778 bytes and the advisory composite by about `0.0025156151249`.

| row | archive B | B/pair | d_seg | d_pose | Seg term | Pose term | Rate term | advisory S |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| settled control | 94,344 | 157.240000 | 0.003515794640406966 | 127.36588287353516 | 0.3515794640406966 | 35.688357047296975 | 0.06281979707335814 | 36.10275630841103 |
| M1 candidate | 90,566 | 150.943333 | 0.003515794640406966 | 127.36588287353516 | 0.3515794640406966 | 35.688357047296975 | 0.06030418194846257 | 36.10024069328613 |
| candidate − control | **−3,778** | **−6.296667** | **0** | **0** | **0** | **0** | **−0.002515615124895563** | **−0.002515615124899284** |

The plane-proximity question is therefore answered precisely: **`Δd_pose = 0.0`**. Banding preserved Pose for this archive because it preserved every receiver byte, not because a separately actuated plane-proximity change was shown pose-null.

This is a firm **FORMULATION-NEGATIVE efficacy verdict**. The three-epoch firing did not actuate a better witness. Its only advantage over the scored control is payload compressibility.

### Later MAIN base-comparison directive

MAIN sent a high-priority conditional prediction at `2026-07-21T02:53:55Z`: check the candidate against the 83,838-byte base materialization and, if raw-identical, classify the row as control distortions plus 6,728 rate bytes. The existing SHA-bound base materialization receipt resolves that condition without another decode:

| row | archive B | raw SHA-256 |
|---|---:|---|
| base materialization | 83,838 | `8565df10cbff8f86f02233fd20ececd74857a0d3806caf278a385a4d5421dcae` |
| M1 candidate | 90,566 | `dbfcdcfa9c2ea361cfa51eb6b6e26379b20ad5591fb0fe399ace496315628a97` |

The candidate is **+6,728 B versus the base but not raw-identical to it**. Therefore the directive’s conditional equality premise is falsified; no base `d_seg`, `d_pose`, or `S` equality is inferred. The candidate is instead exactly raw-identical to the separately scored 94,344-byte control, so the formulation-negative efficacy verdict remains binding there.

## Binding harness and deterministic receiver proof

The aggregate row reused `tools/measure_r1b_boundary_generator_n600.py` byte-for-byte. Its SHA-256 is `400d914eae031e748b0ee70ca9130ac07bf0062365c517fd974602db948428fb`, identical to the file at control-producing commit `aa2cb910c9128e79e392a19e49b61f31484bf499`. The current worktree has no diff to that file. Both rows use seed 1234, batch 16, eight CPU threads, eight decode workers, the same GT video, and the same frozen scorer hashes.

Three candidate reconstructions and the settled control all have raw SHA-256 `dbfcdcfa9c2ea361cfa51eb6b6e26379b20ad5591fb0fe399ace496315628a97`:

| reconstruction | elapsed | raw identity | label |
|---|---:|---|---|
| candidate decode 1 | 753.163059 s launch→receipt | exact | **MEASURED** |
| candidate decode 2 | 738.52 s `/usr/bin/time real` | exact | **MEASURED** |
| unchanged exact-harness decode | 742.0773589611053 s | exact | **MEASURED** |
| settled control receipt | prior 738.9803147315979 s | exact same hash | **MEASURED settled** |

Both fresh independent decodes are byte-identical. Canonical NumPy and parse-back scorer SHA-256 are both `f09527720969b6552a29db13ff68efe3fd55c908ebac44dd829cfa6b3ec3f6f8`; factor-2 equality is exact. Worst fresh decode is 753.163059 seconds, leaving 1,046.836941 seconds under the 1,800-second budget.

The unchanged hard scorer completed 600 pairs in 38 batches: decode 742.0773589611053 s, scoring 503.4817280769348 s, total 1,248.1239140033722 s. The inherited receipt calls the embedded R1b boundary-coordinate candidate “absent”; that schema-local wording does not deny that the whole archive measured here is the M1 composed candidate.

## Exact class decomposition

The diagnostic wrapper observes the two SegNet outputs already produced inside official `DistortionNet.compute_distortion`; it neither replaces nor modifies the scorer. The canonical residual-target helper then performs the label comparison. Integer closure is exact: `414,740` mismatches both by class sum and per-pair sum over `117,964,800` cells. The exact rational `d_seg=0.0035157945421006942` differs from the official float32 aggregate by only `9.830627197698893e-11`.

Because candidate and control raw bytes are identical, candidate class values below are **MEASURED** and control values are **DERIVED EXACT** under the deterministic scorer. Contributions sum to the exact rational overall `d_seg`.

| class | d_seg contribution, each arm | conditional error | area fraction | GT pixels | mismatch pixels |
|---|---:|---:|---:|---:|---:|
| Road | 0.001174714830186632 | 0.0050561820915886475 | 0.23233238220214844 | 27,407,043 | 138,575 |
| Lane | 0.0012633260091145834 | 0.21578277508220647 | 0.005854619344075521 | 690,639 | 149,028 |
| Undriv | 0.00044615003797743053 | 0.000900993715778545 | 0.49517552693684896 | 58,413,282 | 52,630 |
| Movable | 0.00044422573513454864 | 0.03588447777035934 | 0.012379328409830729 | 1,460,325 | 52,403 |
| MyCar | 0.0001873779296875 | 0.0007369594043191543 | 0.2542581431070964 | 29,993,511 | 22,104 |

Lane is the largest overall contribution and has by far the largest conditional error. This is a reusable targeting signal, not permission for blanket repairs: the operator’s Fisher/margin and realized reverse-waterfill stop remain binding.

## Per-pair tails

The candidate tails are **MEASURED**; the control tails are **DERIVED EXACT** from identical raw bytes and deterministic scorer custody.

### Top 8 by d_seg

| rank | pair | d_seg, each arm | mismatches |
|---:|---:|---:|---:|
| 1 | 522 | 0.0057525634765625 | 1,131 |
| 2 | 515 | 0.00531005859375 | 1,044 |
| 3 | 572 | 0.0052947998046875 | 1,041 |
| 4 | 517 | 0.0052897133864462376 | 1,040 |
| 5 | 518 | 0.0052591958083212376 | 1,034 |
| 6 | 510 | 0.0050608315505087376 | 995 |
| 7 | 74 | 0.0049997963942587376 | 983 |
| 8 | 566 | 0.0049947104416787624 | 982 |

### Top 8 by d_pose

| rank | pair | d_pose, each arm |
|---:|---:|---:|
| 1 | 523 | 170.88963317871094 |
| 2 | 21 | 163.10006713867188 |
| 3 | 90 | 162.15255737304688 |
| 4 | 1 | 161.16334533691406 |
| 5 | 7 | 159.992919921875 |
| 6 | 24 | 158.2308349609375 |
| 7 | 49 | 156.4541473388672 |
| 8 | 41 | 155.85140991210938 |

## Gate verdicts

| gate | candidate | limit | margin | verdict |
|---|---:|---:|---:|---|
| d_seg | 0.003515794640406966 | ≤ 0.000339 | −0.0031767946404069663; 10.3711× limit | **FAIL** |
| bytes per pair | 150.94333333333333 | ≤ 477.8 | +326.8566666666667 | **PASS** |
| mission fixed cap | 90,566 B | ≤ 216,222 B | +125,656 B | **PASS** |
| decode | 753.163059 s worst | ≤ 1,800 s | +1,046.836941 s | **PASS** |
| advisory S vs control | 36.10024069328613 | < 36.10275630841103 | −0.002515615124899284 | **PASS, rate only** |
| joint mission box | — | all required | d_seg fails | **FAIL** |

The unchanged harness still emits a legacy fixed-C1 cap of 216,223 B. This mission’s explicit 216,222 B cap is evaluated above. The one-byte contract drift cannot affect this candidate’s pass.

## Mechanism and scoped next move

The trained payload changed materially before the receiver:

- all 4,800 fp16 code elements changed;
- code nonzeros fell from 4,800 to 2,400 and unique values from 3,755 to 1,995;
- all 12 quotient-head elements changed;
- `ipe_codes.f16` compressed by 3,781 B while the manifest grew 3 B, for net archive `−3,778 B`;
- nevertheless, **zero receiver output bytes changed**.

Thus this three-epoch firing remained inside the realized receiver/decode uint8 dead zone while improving compressibility. This is a firm `FORMULATION-NEGATIVE` efficacy result. The family remains open. Named reformulations are: longer `band_fit`; band-width tuning until actual receiver-output actuation is measured; Fisher-margin reverse-waterfill using the corrected realized inner Jacobian with the measured marginal rate stop; receiver-closed `eval_roundtrip` STE; r1b7's measured-positive fixed-magnitude parametrization; and a hotter schedule. The current trainer does call `torch_uint8`, whose output is supplied by `Uint8STE`, before computing `band_loss`; therefore MAIN's literal “NO uint8-STE in training loss” premise is false at this commit. The remaining gap is the lack of a **full receiver-closed** roundtrip STE, not the absence of all uint8 STE. The WARN-only no-Fourier transition remains unchanged; this local no-regression row is not operator-GO to flip defaults or remove the governed control.

Because local advisory `S` beats the control, the archive is **eligible for MAIN’s #578(c) Modal exact-eval review**. This arm did not dispatch. Contest CPU/CUDA custody, promotion, and pointer movement remain owed.

## Resumability, receipts, and triality

The decomposition preserved 38 fsync-complete batch rows in a 574,282-byte JSONL checkpoint, SHA-256 `6202f02f388522d087d6981428ba9f680d8add30ed4c6a182399b3e58777f653`, under contract SHA-256 `beb54f656f2c85be43e1916864b40a85855ae4090b7b99b8321b71f19d5f8fa6`. A torn final append falls back only to the last valid row; contract drift refuses resume. The two full raw decodes remain preserved on the SSD tier; no evidence bytes were deleted.

Primary receipts:

- candidate build: `/Volumes/VertigoDataTier/pact/evidence/m1_byteclose_20260721/build_receipt.json`, SHA-256 `0ef9bd6061ef6cd288b5ab8c140b04b57268d8ee01ae0d61a4250168045b75e2`;
- exact aggregate: `/Volumes/VertigoDataTier/pact/evidence/m1_byteclose_20260721/exact_candidate_harness_20260721.json`, SHA-256 `20d01dac12d8d96c7e20dca44aad1079c9c43e4a5ff92789214a3510faa0ba17`;
- decomposition: `/Volumes/VertigoDataTier/pact/evidence/m1_byteclose_20260721/hard_oracle_decomposition_20260721.json`, SHA-256 `c15423c5316c61297cc7dd1f15df7168d83e46724f8326d2fbfbfac9b214cca5`;
- training: `/Volumes/VertigoDataTier/pact/evidence/m1_curvelet_binding_closer_20260721/materializer_output/m1_c2_curvelet_full_n600_ready_20260721.training_receipt.json`, SHA-256 `9606d01fea0a8153708937031f78973de2b36a6841a75c54b7f4fda01f09cfd8`.

Triality disposition:

- **DSL:** no trainer/controller flag was invented or changed. The candidate consumes the existing typed M1 positive-anisotropic configuration and three sealed stages.
- **Equations:** the existing action `100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37,545,489` is applied without a new law claim; Fisher/margin reverse-waterfill remains the next acquisition law.
- **DAG:** archive → double decode → exact aggregate → decomposed class/tails → per-gate verdict is now a durable, hash-bound edge. The separate DAG FEED prevents the result from remaining chat-only.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; the delegated authority file; latest Codex findings/session summary; R1b4/R1b5/R1b6 memos and receipts; settled control receipt; exact harness lineage at `aa2cb910...`; candidate build/training/base-materialization receipts; exact trainer and `Uint8STE` source; lane, subagent, frontier, probe, council, and task-status canonical stores; broadcast inbox through `2026-07-19T19:48:01Z`; per-arm inbox through `2026-07-21T02:53:55Z`.

## MAIN landing requirement

This branch is not repository authority. MAIN must review the complete branch diff, verify every external receipt/hash, rerun the focused tests and review gates, and decide whether to merge. Only MAIN may claim/dispatch #578(c). A merge must not reinterpret this advisory rate-only row as a contest score, family promotion, operator-GO, or pointer move.
