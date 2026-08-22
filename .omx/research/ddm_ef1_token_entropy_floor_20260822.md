# DDM EF1 token entropy floor — rich generic estimators turn above 365,322 B; no nontrivial lower bound closes every predictor

**MEASURED `[macOS-CPU advisory / scorer-free lossless diagnostic]`, n600:** no tested estimator
reaches the shipped **113,777 B** token stream, much less the required **71,395 B**. The best
challenger is **ZPAQ v7.15 method 5 at 365,322 B**, exact-invertible and deterministic after generic
journal-timestamp canonicalization. It is **251,545 B worse than shipped** and **293,927 B above
demand**. PPMd bottoms at **402,241 B at order 32** and then worsens through order 64. ZPAQ's
prefix-average rate bottoms at 400 frames and rises at 600, while its last marginal rate also rises.

**Adjudication:** charter outcome **(b), scoped to the two measured estimator families**. Their curves
turn well above 71,395 B, so the raw-symbol PPMd suffix-context and ZPAQ generic context-mixing routes
are closed on measured evidence for this exact field. The stronger question—whether *any possible
predictor* can reach 71,395 B—remains **INCONCLUSIVE**. Achieved code sizes are upper bounds, the
normalized compression curve is an estimate, and EF1 establishes no nontrivial lower bound. Calling
365,322 B or its normalized rate an information-theoretic floor would be the mechanism fake this
charter forbids.

`verdict_scope = FAMILY` for PPMd7 variant-H raw-suffix contexts at orders 1–64 with a 256 MiB
adaptive model, and `FAMILY` for ZPAQ v7.15 methods 1–5 on the raw frame-raster field.
`verdict_scope = UNKNOWN` for differently trained HPAC networks, models using the continuous
five-class probability vector, new exact token representations, and unrestricted predictors.

## Reproduced pins and demand

EF1 reused TO2's retained decoded array. It did not decode DX2 again.

| object | bytes | SHA-256 | status |
|---|---:|---|---|
| DX2 archive | 180,368 | `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` | MATCH |
| TO2 decoded token field | 117,964,800 | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` | MATCH |
| shipped RC64 token stream | 113,777 | `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5` | MATCH |
| TO2 token checkpoint receipt | 3,511 | `c0c05971396ff066c16cc0a82a46c5fe3e99a9c0000b4a93933e4bb2a57359f9` | MATCH; binds all three pins |

The denominator is exactly **117,964,800 five-class positions**. Recomputed from retained bytes:

- incumbent density: `8*113777/117964800 = 0.007715996636` bits/position;
- strict fixed-distortion archive ceiling: **137,986 B**;
- required archive and token-stream cut: `180368-137986 = 113777-71395 = 42,382 B`;
- token target: **71,395 B**, `0.004841783312` bits/position, **62.74994%** of incumbent density.

RB1 owns the fixed-representation coder census and measures **0 B** remaining across the seven DX2
regions. AD2 owns addressing: raster sites are already implicit/free; its QPAIR win is on NR1, not
this DX2 stream. TO2 owns serialization order and measures its nine generic exact forms at
196.07%–686.94% worse. CX3 owns named summaries and measures its best model-inclusive row at
125,210 B, with a 117,224 B hindsight data term before model cost. EF1 does not re-race those axes.

## Real estimator race

Every row below is a retained coded payload. Every primary payload has a retained repeat; every
admitted row decodes to a retained byte array with the exact `cc10...3eefb` source SHA-256.
`description/control B` is stated separately and remains included in total bytes.

### High-order adaptive model

Estimator: `pyppmd 1.3.1`, PPMd7 variant H adaptive arithmetic, **256 MiB runtime model**, raw
frame-raster `uint8` symbols. The adaptive model is rebuilt causally and transmits **0 B**. Each total
includes a **16 B** packet header carrying variant, order, memory, and exact decoded length.

| max order | total coded B | description/header B | delta vs 113,777 B | bits/position | native decode s | exact inverse |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 415,045 | 16 | +301,268 | 0.028147040473 | 3.678 | PASS |
| 2 | 415,045 | 16 | +301,268 | 0.028147040473 | 3.705 | PASS |
| 3 | 412,989 | 16 | +299,212 | 0.028007609049 | 3.673 | PASS |
| 4 | 411,303 | 16 | +297,526 | 0.027893269857 | 3.672 | PASS |
| 6 | 408,456 | 16 | +294,679 | 0.027700195312 | 3.679 | PASS |
| 8 | 407,199 | 16 | +293,422 | 0.027614949544 | 3.684 | PASS |
| 12 | 405,755 | 16 | +291,978 | 0.027517022027 | 3.707 | PASS |
| 16 | 405,059 | 16 | +291,282 | 0.027469821506 | 3.684 | PASS |
| 20 | 404,158 | 16 | +290,381 | 0.027408718533 | 3.709 | PASS |
| 24 | 403,185 | 16 | +289,408 | 0.027342732747 | 3.698 | PASS |
| 28 | 402,565 | 16 | +288,788 | 0.027300686306 | 3.709 | PASS |
| **32** | **402,241** | **16** | **+288,464** | **0.027278713650** | **3.707** | **PASS** |
| 40 | 402,412 | 16 | +288,635 | 0.027290310330 | 3.779 | PASS |
| 48 | 403,471 | 16 | +289,694 | 0.027362128364 | 3.775 | PASS |
| 56 | 405,035 | 16 | +291,258 | 0.027468193902 | 3.810 | PASS |
| 64 | 407,038 | 16 | +293,261 | 0.027604031033 | 3.847 | PASS |

The order curve is not merely slow improvement at the compute boundary. It reaches its minimum at
order 32, then worsens monotonically at 40, 48, 56, and 64. Raw suffix context has become too sparse
to pay its learning cost. This is a measured family turn, not a lower bound on other predictors.

### General-purpose context mixing

Estimator: `/opt/homebrew/bin/zpaq` **v7.15**, single thread, `-noattributes`, methods 1–5. ZPAQ is
the existing repository's named strong local context-mixing baseline. It transmits no learned
video-specific model. The separate control is an exact ZPAQ archive of a zero-byte file under the
same method: **784 B** for methods 1–4 and **1,015 B** for method 5. That control includes fixed
archive/model-description/framing bytes; it is not claimed to isolate pure model code, and it is not
subtracted from the achieved total.

| ZPAQ method | total coded B | description/control B | delta vs 113,777 B | bits/position | native decode s | exact inverse |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 924,857 | 784 | +811,080 | 0.062720879449 | 4.536 | PASS |
| 2 | 713,480 | 784 | +599,703 | 0.048385959201 | 4.506 | PASS |
| 3 | 372,443 | 784 | +258,666 | 0.025257907444 | 24.076 | PASS |
| 4 | 372,443 | 784 | +258,666 | 0.025257907444 | 23.774 | PASS |
| **5** | **365,322** | **1,015** | **+251,545** | **0.024774983724** | **106.115** | **PASS** |

ZPAQ journals the wall-clock update time. Primary and repeat raw archives therefore differed only in
generic ASCII journal timestamps on some rungs. EF1 retains both raw archives, canonicalizes every
`jDCYYYYMMDDHHMMSS` field to a fixed generic timestamp, retains both canonical archives, and requires
ZPAQ's own integrity test plus exact source inversion afterward. The counted canonical form is the
same size as raw. Method 5's counted payload SHA-256 is
`b6a0e81a05c138f4b13c390ad07671345291ed9847bf9ca4ee9738472d7478c6`.

The best generic challenger is **3.21086× the incumbent density** and **5.11691× the target density**.
It is not advanced. Its measured 106.1 s local CPU decode fits 1,800 s on this advisory host, but no
contest-runtime ZPAQ bundle or contest-hardware timing was built because bytes already lose by
251,545 B. Runtime is not the rejection mechanism.

## Compression-based entropy-rate convergence curve

This is ZPAQ method 5 over contiguous prefixes of the exact source array. `Achieved bpp` uses total
archive bytes and is an achieved finite-string code rate. `Estimate excluding control` subtracts the
fixed 1,015 B zero-input control and is explicitly an **estimate**, not an achieved packet and not a
lower bound. `Marginal bpp` is the real added archive bits divided by added positions since the prior
prefix. Every prefix source, stream, repeat, and exact decoded output is retained.

| frames | positions | achieved B | achieved bpp | estimate excluding control bpp | marginal bpp |
|---:|---:|---:|---:|---:|---:|
| 8 | 1,572,864 | 5,913 | 0.030075073242 | 0.024912516276 | — |
| 16 | 3,145,728 | 10,890 | 0.027694702148 | 0.025113423665 | 0.025314331055 |
| 32 | 6,291,456 | 21,315 | 0.027103424072 | 0.025812784831 | 0.026512145996 |
| 64 | 12,582,912 | 41,276 | 0.026242574056 | 0.025597254435 | 0.025381724040 |
| 128 | 25,165,824 | 79,828 | 0.025376637777 | 0.025053977966 | 0.024510701497 |
| 256 | 50,331,648 | 151,874 | 0.024139722188 | 0.023978392283 | 0.022902806600 |
| **400** | **78,643,200** | **234,416** | **0.023846028646** | **0.023742777507** | **0.023323906793** |
| **600** | **117,964,800** | **365,322** | **0.024774983724** | **0.024706149631** | **0.026632893880** |

The curve's honest shape is **flattened and turned**, not still descending at n600. Average achieved
rate improves only 1.22% from 256 to 400 frames, then worsens 3.90% from 400 to 600. The marginal
rate rises from 0.022903 to 0.023324 to 0.026633 bits/position over the last three increments. The
source is a nonstationary frame prefix, so this is convergence evidence for this compressor/source
pair, not an asymptotic source theorem.

## Upper bound, estimate, and lower bound labels

- **Achieved sizes / upper bounds:** every PPMd and ZPAQ total above. A real lossless code of that
  size exists and decodes exactly. Therefore each is an upper bound on the shortest code within an
  unrestricted search, never a lower bound.
- **Compression-based estimates:** ZPAQ total/control-normalized bits per position, prefix marginal
  bits per position, and the observed 0.023–0.027 tail band. They describe this estimator's empirical
  rate and convergence shape only.
- **Lower bound:** EF1 establishes **no nontrivial lower bound** on this one fixed field under an
  unrestricted predictor. The only universally defensible bound supplied here is the useless
  nonnegativity bound of 0 B. No result rules out a shorter content-aware program or learned model.

Consequently **71,395 B is not reached and is strongly contradicted by the tested rich generic
families, but it is not proved impossible in principle**. The permanent information-theoretic closure
requested by the charter is unavailable without a defined predictor class plus a valid converse/lower
bound for that class.

## Prior-law verdict and campaign consequence

The prior-law prediction is **MIXED**:

- **FALSIFIED:** richer generic estimators did not land below CX3's 125,210 B named-summary result or
  within a few percent of shipped. The best was 365,322 B, 221.09% above shipped.
- **SUPPORTED at FAMILY scope:** both estimator curves turn far above 71,395 B. More raw suffix order
  and more generic context-mixing effort do not supply the 42,382 B demand.
- **UNPROVEN:** the conclusion that no better predictor can exist. The incumbent domain-specific
  learned 19-member HPAC law beats the best generic estimator by 251,545 B, direct evidence that
  predictor class matters more than generic richness here.

Thus no receiver integration, archive build, scorer, Metal, Modal, or authority-eval follow-on fires.
The fixed DX2 token route is closed for PPMd/raw-suffix and ZPAQ/generic-context families. A future
lossless attempt must change the **predictor class or exact representation**, not rerun a generic
compressor, order, summary, addressing, or wire-coder race.

## RECALL EVIDENCE

I searched `.omx/research/`, arm receipts, `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks,
the canonical task ledger, active lane claims, main hot state, design/spec surfaces, source code, and
the canonical-equations JSON. Content queries included `context mix`, `zpaq`, `paq8`, `ppmd`,
`prediction by partial matching`, `context tree weighting`, `entropy rate`, `high-order adaptive`,
`HPAC`, `RC64`, `71,395`, and `113,777`. No duplicate active EF1 lane or canonical equation
superseding the source-coding label discipline was found in those scopes.

Beyond the charter seeds, recall found and changed the plan as follows:

- `experiments/ddm_rc2_20260810/run_ppmd_reference_race.py` and
  `.omx/research/ddm_rc2_20260810/RC2_FINAL_REPORT.md` already raced PPMd orders 2–16, but only on
  20–117 KB serialized PR130 sections—not the 117,964,800-symbol DX2 field. EF1 reused its counted
  parameter-header and retain-before-measure pattern, but did not transfer its negative.
- `ddm_dc1_decode_budget_conditional_coding_20260816.md` showed an older-object adaptive context curve
  turning when samples/context became sparse. That made an explicit order curve and a prefix
  convergence curve mandatory; EF1 extended PPMd until the order-32 turn was observed rather than
  stopping at order 16.
- `src/tac/lossless/codecs.py` and `profiles.py` name ZPAQ method 5 as the repository's strong local
  context-mixing baseline and label it local-only without a bundled runtime. EF1 therefore measured
  real method-1…5 payloads and decode time, while refusing a contest-runtime claim.
- The live `ddm_ni1_nr1_k32_receiver_distortion_20260822.md` says its n600 shipping-receiver score is
  **not measured**, contrary to the charter's inherited sentence that both lossy representations were
  already measured dead. EF1 does not use NI1's unmeasured distortion as evidence. RI1's exact RC1
  instance is measured dead; LQ1 remains formulation-inconclusive; JX1's 3.705× receipt establishes
  non-additivity but corrects its direction to favorable joint compensation. None changes this
  scorer-free lossless race.

## Retention, verification, and boundaries

All new receipts use the required tier:
`/Volumes/VertigoDataTier/pact/ddm_ef1_token_entropy_floor/measurement_v1/`. EF1 wrote nothing to
APDataStore and created no local scratch.

- Authoritative `MANIFEST.json`: 171,153 B, SHA-256
  `f0b960b4f1a9604ab35830eed7a1318ce334c51873fa38672d9b88ca74102edd`.
- `INVENTORY.json`: 73,247 B, SHA-256
  `f874fddf1d8ed2493fd976ddd9d10f1fc0fbc64b5bb02e5d7ec1611591da2d7a`; 282 listed artifacts
  totaling **2,870,386,984 B**, plus the inventory itself.
- Independent post-run verification rehashed all 282 listed artifacts and found zero mismatches.
- Independent candidate verification covered **21 full-field streams** and **8 prefix rows**: every
  primary/repeat pair matched, every decoded receipt matched its source, and every full-field decoded
  SHA equaled `cc10...3eefb`.
- The default runner replay completed from disk without recompression and refuses finalization unless
  all 16 PPMd orders, all 5 ZPAQ methods, and all 8 prefix points are present.
- Two genuine review passes, `ruff`, `py_compile`, adversarial curve assertions, and the
  measure-and-discard preflight all pass. No shipped receiver, `upstream/`, jo1-r9, scorer, archive,
  or shared hot-state file was modified.

## NEXT_IF_RESUMED

- `ef1_continuous_probability_predictor`; disposition=`QUEUED-WITH-A-FIRE-ORDER`; owner=`MAIN assigns a nonduplicate learned-HPAC/model lane`; consumer store=`/Volumes/VertigoDataTier/pact/<claimed_probability_predictor_lane>/RESULT.json`; fire trigger=`MAIN records the lane claim, pins the cc10...3eefb field and an exact receiver-available continuous five-class probability trace, and the runner retains counted model + coded stream + repeats + exact decoded field; receiver integration fires only if model+token <=84,910 B (equivalently token <=71,395 B with the current 13,515 B model) and contest-hardware decode is measured <=1,800 s`.

## LIVE-HYPOTHESES

- A content-aware learned predictor consuming the continuous five-class probability vector could beat
  generic raw-symbol context models because shipped HPAC is already 3.21086× denser than EF1's best
  generic estimator. It remains plausible but must pay its complete model bytes and invert exactly.
- The useful information is not a longer raw suffix: PPMd's turn at order 32 and ZPAQ's 400→600
  reversal say the remaining surprises are keyed to domain coordinates/probabilities that generic
  contexts do not expose. A changed learned predictor class is therefore more plausible than more
  order, memory, or generic compute.

## DEAD-ENDS

- PPMd7 variant-H raw-suffix contexts, 256 MiB, orders 1–64: minimum 402,241 B at order 32; every
  tested higher order worsens.
- ZPAQ v7.15 generic context mixing, methods 1–5: best 365,322 B; method 5's prefix-average and
  marginal rates turn upward at the full field.
- Re-racing hand-named summaries, serialization order, addressing, or fixed-probability wire coders:
  CX3, TO2, AD2, and RB1 already own and close those scoped cells.
- Treating an achieved compressed size, a zero-input-control-normalized rate, or a visually flat
  curve as an information-theoretic lower bound: none is a converse, so none can permanently prove
  71,395 B impossible.

**Own-vehicle frontier: UNMOVED at DX2 S=0.14821987563243377 @ 180,368 B `[contest-CUDA T4 n600]`, archive SHA-256 `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`.**
