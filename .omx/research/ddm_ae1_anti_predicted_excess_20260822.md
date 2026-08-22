# ddm_ae1 anti-predicted excess — exact gross exists, both priced static routes lose net

Date: 2026-08-22  
Axis: `[macOS-CPU advisory / scorer-free exact retained-field measurement]`  
Status: `MEASURED_GROSS_AND_PRICED_NET_CEILINGS`  
Score claim: **false** · pointer moved: **false** · scorer/evaluator run: **none**  
`research_only=true`  
`verdict_scope`: **INSTANCE** for the exact excess on DX2/BL1; **FORMULATION** for explicit stored flags and the two counted static uniform-overlay models. The zero-stored causal online-member family remains unmeasured.

# FORMALIZATION_PENDING:This is a one-instance retained-field allocation and signalling-price measurement, not a new predictive equation; all arithmetic and complete denominators are machine-readable in the retained RESULT.json.

## Conclusion first

**MEASURED: 93,580 / 117,964,800 positions cost more than `log2(5)`. Their exact gross excess is 213,162.383261 bits = 26,645.297908 B. The best real explicit flag costs 130,228 B, so that route's NET ceiling is −103,582.702092 B. A counted global uniform overlay selects alpha=0 and nets −14 B; a counted 190-group overlay gains only 10.531230 modelled bytes, costs 45 real descriptor bytes, and nets −34.468770 B.**

The 26,645.30 B gross is material—62.8694% of the 42,382 B sub-0.12 demand—but it is an allocation ceiling, not a saving. Both mechanisms this arm could price without building a receiver lose after their own signalling/model-growth column. The preregistered `>5,000 B NET` prediction is falsified for these two formulations.

Lane is the largest class contribution but misses the preregistered majority threshold: **12,462.135023 B, 46.7705% of gross excess**, from 33,339 positions. That is 29.4043% of the 42,382 B demand. Lane remains the unique shared rate/distortion class-level location, but this result is not a seg-error-position lever and does not make Lane representation changes lossless or safe.

No token stream was recoded or admitted. The real-coded flag streams decode exactly back to the 117,964,800-position overshoot mask; every member descriptor parses back exactly. Therefore no token-array inversion claim is made. A future receiver mechanism would still have to decode byte-for-byte to TO2's `cc10a7b0…` array before any physical byte claim.

## Pinned controls and BL1 reproduction gate

The script refused drift before measuring and reused BL1's primary cost field rather than re-instrumenting the decoder.

| input | bytes | SHA-256 |
|---|---:|---|
| DX2 archive | 180,368 | `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` |
| physical RC64 stream | 113,777 | `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5` |
| TO2 decoded token field | 117,964,800 | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` |
| BL1 primary cost field | 943,718,400 | `99d7833d55a9aa128f67cfc125a10ba90cffaa714de0b88801aa43b8b16e2c86` |
| BL1 RESULT | 318,937 | `f8835acf27c3b46bf95f7cd1954e08d72d591854f8f78ac6c902889a064b6621` |
| BL1 MANIFEST | 56,421 | `0b2ca8ec51738b6e7ee5940d262be7226457fcd5a4f8e56f4bfb5b98184a59ac` |
| n600 GT argmax | 117,964,928 | `91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248` |

The gate passed before the first new excess stage:

| BL1 control | reproduced | BL1 target | disposition |
|---|---:|---:|---|
| top 0.1% bit share | 52.9506876901% | 52.9506876901% | PASS |
| top 1% bit share | 96.3238419085% | 96.3238419085% | PASS |
| Gini | 0.9951593787014741 | 0.9951593787014772 | PASS, floating reduction order only |
| Lane positions | 690,754 | 690,754 | PASS |
| Lane bits | 305,463.96947306156 | 305,463.96947306144 | PASS, floating reduction order only |
| Lane share | 33.5597511452% | 33.5597511452% | PASS |

The independently sorted field is 943,718,400 B, sha `c4d21e36f386bae56213fa905517eee47cc76b59f40f0d2635f384a607271a8f`.

## Exact gross excess

For BL1's exact selected integer-frequency costs `c_i`, this arm computes only:

`E = sum_i max(0, c_i - log2(5))`, with `log2(5) = 2.321928094887362`.

| quantity | measured n600 |
|---|---:|
| denominator | 117,964,800 positions |
| positions with `c_i > log2(5)` | 93,580 (0.0793287489%) |
| total cost carried by those positions | 430,448.414380 bits = 53,806.051798 B |
| mean total cost per overshoot | 4.599790707 bits |
| **gross excess** | **213,162.383261 bits = 26,645.297908 B** |
| mean gross excess per overshoot | 2.277862612 bits |
| gross / 42,382 B demand | **62.8693736%** |

### Per GT class

| GT class | class positions | overshoot positions | overshoot rate in class | gross excess (B) | share of gross |
|---|---:|---:|---:|---:|---:|
| Road | 27,407,372 | 32,038 | 0.116896% | 7,269.246560 | 27.2815% |
| **Lane** | **690,754** | **33,339** | **4.826465%** | **12,462.135023** | **46.7705%** |
| Undrivable | 58,413,067 | 9,590 | 0.016418% | 1,913.812193 | 7.1826% |
| Movable | 1,460,386 | 10,765 | 0.737134% | 3,176.888700 | 11.9229% |
| MyCar | 29,993,221 | 7,848 | 0.026166% | 1,823.215430 | 6.8425% |

Lane owns 35.6262% of overshoot positions but 46.7705% of excess, so its overshoots are also more expensive than average. This is a class-level overlap only. MS9's asymmetric join remains binding: seg-error positions are not the same object.

### Per frame/time

The retained `frame_rows` table has all **600 / 600** frames. The largest rows are:

| frame | overshoot positions | gross excess (B) |
|---:|---:|---:|
| 0 | 292 | 143.484531 |
| 522 | 293 | 134.178390 |
| 65 | 252 | 124.533273 |
| 70 | 266 | 100.544128 |
| 69 | 230 | 89.043009 |
| 515 | 238 | 87.078747 |
| 66 | 202 | 86.716731 |
| 63 | 208 | 81.051750 |
| 67 | 212 | 80.975641 |
| 72 | 221 | 78.396587 |

The top 10 / 25 / 60 / 120 frames carry only 3.7755% / 7.7276% / 15.8237% / 28.2397% of gross. Two visible runs, frames 63–74 and 515–522, carry 973.803147 B and 622.815813 B respectively. Time is not sparse enough to make a stored frame/run signal cheap.

### Per inherited group

The retained `group_rows` table has all **190 / 190** groups for `g=(x mod 64)+2*(y mod 64)`. Its largest rows are:

| group | positions | overshoot positions | gross excess (B) |
|---:|---:|---:|---:|
| 62 | 921,600 | 954 | 249.620878 |
| 60 | 892,800 | 892 | 244.810626 |
| 58 | 864,000 | 853 | 244.479624 |
| 61 | 892,800 | 798 | 238.988490 |
| 64 | 921,600 | 816 | 238.687250 |
| 120 | 921,600 | 792 | 233.958553 |
| 59 | 864,000 | 818 | 232.938471 |
| 57 | 835,200 | 777 | 232.267710 |

The top 10 / 25 / 50 / 95 groups carry 8.8839% / 20.8275% / 39.0024% / 67.7851% of gross. Group identity localizes some excess but does not isolate it.

## Which contexts overshoot

The FS2 trace join is a conditioning read on the same token field, not a replay of DX2's final 70-byte corrector improvement. Its predictor argmax sha is `93cdf71d…`; its `u_index` sha is `74470f44…`.

| relation to predictor argmax | positions | overshoot positions | gross excess (B) |
|---|---:|---:|---:|
| hit | 117,736,970 | **0** | **0** |
| miss | 227,830 | **93,580** | **26,645.297908** |

Thus every overshoot is a miss, and 41.0745% of misses overshoot. By **predicted** class, Road owns 16,302.914266 B (61.18496% of gross), even though by **realized GT** class Lane is largest at 46.7705%. The highest fine contexts are predicted-Road cells in groups around 48–62 over several `u64` bins; the best individual context carries only 15.813 B. The complete `group × predicted class × u64` denominator is 60,800 cells, 48,395 active, and is retained in both array and row form.

Plain-language interpretation: these are wrong-class events where the model's hit-vs-miss machinery is confident and the residual non-argmax mass does not put enough probability on the class that occurs. Lane/Road confusions are a leading component, but the mass is diffuse across frame, group, and fine context.

## Why the incumbent 19 members do not impose a uniform cap

The actual DX2 runtime, not a surrogate compressor, answers this:

- `fx2_model_axis_corrector.py:615-641` freezes 19 named families. They are context views of the predicted-class hit event; none is a categorical five-way uniform floor.
- `fx1_logistic_mixer_corrector.py:15-23` uses a learned geometric/log-odds mix, which can sharpen beyond any arithmetic average. Lines 84–88 establish that its weights start fixed and update only from already-decoded symbols, so their state costs zero stored bytes.
- `fx1_logistic_mixer_corrector.py:652-670` sets the corrected argmax probability `q`, then scales the other four columns proportionally. Calibrating `P(hit)` does not guarantee each particular miss symbol probability is at least 1/5.
- MA1's final `free_corrector.py:251-288` reweights the four non-argmax columns while preserving their total miss mass. It learns relative miss classes, but it still contains no hard `p_k >= 1/5` rule.

That is why the residual is real even after 19 members and MA1. It is also why a blanket uniform blend can lose overall: helping 93,580 rare misses perturbs 117,871,220 positions that are already below the uniform cost.

## Signalling and model-growth price

### Explicit stored flag

Flags are serialized in inherited 190-group event order in two forms, coded with raw, Brotli q11, zlib9, and raw LZMA1-1MiB. Every output has a byte-identical deterministic repeat and decodes back to the exact overshoot mask.

| flag representation | raw (B) | Brotli q11 (B) | zlib9 (B) | LZMA1 (B) |
|---|---:|---:|---:|---:|
| event packbits | 14,745,624 | 185,736 | 199,959 | 216,277 |
| event delta-ULEB | 161,885 | **130,228** | 145,133 | 131,547 |

Winner: delta-ULEB + Brotli q11, 130,228 B, sha `432dea257b08ce24ac9756483a4162614f01bc875cb43d65e38f2c62060143df`. It costs 11.13298 signalling bits per overshoot against only 2.27786 gross excess bits per overshoot.

| gross credit | counted signal | **NET ceiling** | share of demand |
|---:|---:|---:|---:|
| 26,645.297908 B | 130,228 B | **−103,582.702092 B** | **−244.4026%** |

This explicit-flag formulation is CLOSED on the pinned DX2 instance.

### Counted static uniform-overlay descriptions

These extend the incumbent selected-symbol probabilities as `q=(1-alpha)p+alpha/5`; they do not replace the incumbent. Alpha is learned over all n600 positions, quantized to u16, serialized with an `AE1M` header, real-coded four ways, repeated, and parsed back. The credit is a selected-probability model-code-length diagnostic, **not** a finite-precision RC64 stream.

| overlay | learned contexts | modelled credit (B) | best real descriptor (B) | **NET model ceiling (B)** |
|---|---:|---:|---:|---:|
| global | 1 (`alpha=0`) | −0.000000001 | 14 raw | **−14.000000001** |
| inherited group | 190 | 10.531230 | 45 zlib9 | **−34.468770** |

The group descriptor winner sha is `b7d56e567617378698252bc86f233e12b3f8ef9db768ee2313816e8ad94ba4e2`. Only 14 group alpha codes are nonzero, and each is the minimum `1/65535`; this is direct evidence that a static uniform floor is almost entirely rejected by the loss.

These counted static-overlay formulations are CLOSED. They do not close a new causal online member whose weights are regenerated from decoded symbols and therefore add zero stored bytes under rule 118. That mechanism's realized gain is `UNKNOWN_NOT_BUILT_OR_REPLAYED`, as required by this arm's no-build boundary.

## Prior-law adjudication

| preregistered claim | measured | verdict |
|---|---:|---|
| gross excess >10,000 B | 26,645.297908 B | PASS |
| Lane >50% of gross excess | 46.7705% | FAIL |
| learned uniform/escape member >5,000 B NET | global −14.000 B; group −34.469 B on tested counted static overlays | FAIL for tested formulations |

The prediction is mixed at the allocation layer and negative at the priced static-mechanism layer. The narrow honest verdict is not “the incumbent always respects the alphabet bound”; it demonstrably does not. It is: **the residual exists, but explicit location signalling costs 4.89× the gross credit, and static uniform mixing is rejected by the overwhelmingly good positions.**

## MA1 recall verdict

**PARTIALLY ANSWERED, NOT DUPLICATED.** MA1 measured the within-miss relative law on an earlier live body: 223,694 misses, 104.584 B realized online, about 180 B hindsight-reachable, and withdrew the vacuous 77,241 B “miss reservoir.” It did not threshold BL1's final reconciled per-position field at `log2(5)`, compute exact overshoot excess, or price explicit escape flags. This arm therefore measured the open delta only.

MA1 changed the plan in two ways: no categorical replacement of the strong prior, and no claim that all miss entropy is recoverable. AE1 extended selected probabilities with a uniform component and kept all conclusions scoped to the 26,645 B overshoot ceiling.

## RECALL EVIDENCE

Searched before pricing or implementation:

- Full `.omx/research/` and arm receipts with content queries including `uniform|escape|log2(5)|worse than uniform|miss cost|within-miss|alphabet bound|context mix|selected probability`.
- `.omx/research/CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, design/SPEC files, task ledgers, `main_hot_state.md`, and `tools/list_canonical_equations.py --json` with `token|range|context|entropy|direction|marginal` queries.
- BL1, MA1, CX3, EF1, TO2, LQ1, MS9, FX1/FX2/FX5, and the actual DX2 receiver sources.

Findings beyond the charter's seed list:

1. FS2 retained the same-token-field predictor argmax and `u_index` traces. That added the denominator-complete hit/miss and `group × predicted class × u64` characterization instead of guessing “rare context.”
2. The current receiver's 19 families and zero-stored causal weight law were recovered from the actual runtime. This changed “model-blob growth” into two separate cases: counted static descriptors versus an unbuilt rule-118 online member.
3. MA1 did not answer worse-than-uniform; it closed most of a different within-miss multiplicative reservoir. That prevented a duplicate miss-sector mechanism and kept AE1 on the BL1 delta.
4. CX3's 48,395-cell learned-predictor join is exactly the same active-cell count reproduced here. Its best named challenger was already 11,433 B worse than the incumbent, reinforcing that fine context labels are not free descriptions.
5. No canonical equation directly encoded anti-predicted excess or an alphabet-floor saving. The registry search changed no arithmetic; this memo carries the scoped `FORMALIZATION_PENDING` waiver rather than inventing a general law from one stream.

## Retention, failure custody, and verification

All bulk receipts are on **`/Volumes/VertigoDataTier/pact`**, as required. The successful root is:

`/Volumes/VertigoDataTier/pact/ddm_ae1_anti_predicted_excess/measurement_v2`

| retained object | bytes | SHA-256 |
|---|---:|---|
| complete excess field | 943,718,400 | `45f94cdaeeda86a7f4e467af1f182c73a2c5de76d08ed7c0a22c3b0f8af879ed` |
| complete overshoot mask | 14,745,600 | `a1fadb5a966343f79649dcd4af892e373868bb93cf6ab2347fd1f3ef4a274d18` |
| Lane overshoot mask | 14,745,600 | `5d67dbcbfe342c9e3c3ef057d214c058cd2a07e41290d564f9fc92a69a1d8c9b` |
| predictor-context count array | 972,928 | `c8934f46fff04a677ced01beb025d42f971f1ee2bf02e26c47c4939dc8814aac` |
| predictor-context bit array | 972,928 | `cbdc74cf543b941b0c0383ece4c522dc8d02e1cd96b478adcb220607ca843709` |
| RESULT | 12,647,100 | `0554bbad599be921651dcde7174772527dfd801e388bb072a08448330b5d6a6e` |
| MANIFEST | 39,655 | `20c160b7eb78d8ee4806adca10bb9176da84b25d82034bf6f96eccfdf935a9d3` |

The manifest covers **131 artifacts / 3,068,789,515 bytes**. `--verify-completed` rehashed all 131 and returned `VERIFIED_COMPLETE`. Six 100-frame stage receipts and 25 preserved uniform-member optimizer checkpoints (initial state plus 24 bisection iterations) make the run disk-resumable. The exact implementation measured is `experiments/ddm_ae1_anti_predicted_excess.py`, sha `adb025024bd41429392d03eeff328a7033a1069c3f58b775e18cafdf16274447`.

The first run root, `measurement_v1`, passed BL1 reproduction but hit a NumPy `uint64`-to-`bincount` type refusal before its first stage receipt. Its already-materialized bytes were not discarded: `FAILURE_MANIFEST.json` records 12 artifacts / 1,115,759,088 B, status `FAILED_RETAINED_NOT_A_MEASUREMENT_VERDICT`, sha `89bcfb774542fb20e4c9c950cea94324120f2b339d15cc425eb2a86a5b092c3a`. v2 fixed the bounded context IDs to `int64` and independently re-passed BL1.

## Authority and integration boundaries

- MEASURED exactly: BL1 reproduction, threshold count, gross excess, all class/frame/group/context denominators, retained real flag/descriptor sizes, deterministic repeats, and parse-backs.
- MODELLED only: selected-probability code-length credit of the two static overlays. No finite-precision RC64 token stream was produced for them.
- NOT measured: an actual 20th causal online member, physical token-stream shrinkage, receiver timing, archive size, score, d_seg, or d_pose.
- No scorer, Modal, Metal, upstream mutation, shipped receiver mutation, or jo1-r9 access occurred.
- Sensitivity map / Pareto / bit allocator / autopilot hooks are N/A: this is a lossless allocation measurement with no deployable candidate. Continual-learning and probe hooks remain research-only because the sole live interpretation requires a mechanism build expressly outside this charter.

## Successor fire order

- **FOLDED, not launched:** owner `unassigned lossless-coder successor`; consumer store `this memo + measurement_v2/RESULT.json`; fire only on a new MAIN/operator charter authorizing a receiver mechanism. First gate: add exactly one uniform member to the incumbent 19-member causal mixer, keep zero stored state, full n600 encode/decode to TO2 sha `cc10a7b0…`, retain the physical RC64 stream and deterministic repeat, and proceed only if real net gain is positive after receiver runtime.

Own-vehicle frontier: `S = 0.14821987563243377 @ 180,368 B [contest-CUDA T4 n600]` (DX2; AE1 did not move it).
