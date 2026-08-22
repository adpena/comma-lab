# ddm_bl1 — the top 1% of DX2 token positions carry 96.3238% of the shipped HPAC/RC64 model cost

**Disposition:** `MEASURED-RECONCILED / STRONGLY-CONCENTRATED`,
`verdict_scope=INSTANCE:DX2_archive_976f706d_n600_shipped_HPAC_RC64_law`.
The registered diffuse falsifier did not fire. Exactly 1,179,648 of 117,964,800 positions carry
876,748.548 / 910,209.281 modeled bits, or **96.323842%**. The top 0.1% already carries
**52.950688%**. Gini is **0.995159**.

This is an allocation, not a lower bound, a new representation, or a score. No archive byte changed,
no scorer ran, and the own-vehicle frontier did not move.

## Shipped-stream reproduction and reconciliation

All charter pins reproduced before the decode:

| object | bytes | SHA-256 |
|---|---:|---|
| DX2 `archive.zip` | 180,368 | `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` |
| TO2 decoded tokens | 117,964,800 | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` |
| TO2 checkpoint receipt | 3,511 | `c0c05971396ff066c16cc0a82a46c5fe3e99a9c0000b4a93933e4bb2a57359f9` |
| RC64 token stream | 113,777 | `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5` |

The physical stream is 910,216 bits over 117,964,800 positions, or
**0.007715996636 bits/position**. The instrument traversed the shipped frame -> 190-group -> raster
decode, called the unmodified shipped C decoder, and compared every decoded group with TO2's retained
field. The complete instrumented output reproduced TO2 byte-for-byte at the same decoded-field SHA.
The final decoder read position was 910,272, exactly TO2's receipt; the 56-bit difference is
arithmetic-decoder lookahead into defined zero fill, not transmitted payload.

The shipped C path converts each float32 row to a five-symbol integer frequency row summing to
`2^31`. The primary field records the actual law used by RC64:

`cost[i] = -log2(integer_frequency[i, decoded_symbol] / 2^31)`.

| accounting surface | bits | bytes | denominator / check |
|---|---:|---:|---|
| shipped RC64 payload | 910,216.000000 | 113,777.000000 | 117,964,800 positions |
| **primary integer-frequency costs** | **910,209.280609** | **113,776.160076** | 117,964,800 positions |
| stream minus primary | **6.719391** | **0.839924** | required `<9` bits; PASS |
| float32 probability-input costs | 910,209.432143 | 113,776.179018 | auxiliary aligned field |
| Python float64 model replay | 910,209.432143 | 113,776.179018 | equals retained same-stream DC1 ledger |

The `<9`-bit gate is the classic finite-precision arithmetic-interval overhead `<2` bits plus the
shipped encoder's final partial-byte padding `<7` bits. The measured 6.719391-bit difference passes.
The float64 replay also equals the 600-frame DC1 ledger to the retained double, while the float32
probability field differs by only 3.1e-9 total bits. These independent controls make an
unreconciled/reimplemented probability field untenable.

TO2 used the native C corrector speed port. The instrument used the shipped Python fallback because
its complete adaptive state is checkpointable; the shipped native binding declares and gates this as
the bit-identical reference law. The stronger empirical controls are that every decoded symbol, every
600-frame float64 ledger row, the full decoded SHA, and the final RC64 state matched.

## Full distribution

The primary field has minimum cost `2.687229994e-9` bits, maximum 31 bits, mean
`0.007715939675` bits/position, and Gini `0.9951593787`.

### Concentration curve

| top position set | positions / 117,964,800 | bits / 910,209.281 | bit fraction | byte-equivalent |
|---|---:|---:|---:|---:|
| 0.1% | 117,965 / 117,964,800 | 481,962.074 / 910,209.281 | **52.950688%** | 60,245.259 B |
| 1% | 1,179,648 / 117,964,800 | 876,748.548 / 910,209.281 | **96.323842%** | 109,593.569 B |
| 5% | 5,898,240 / 117,964,800 | 907,318.863 / 910,209.281 | 99.682445% | 113,414.858 B |
| 10% | 11,796,480 / 117,964,800 | 909,307.074 / 910,209.281 | 99.900879% | 113,663.384 B |
| 50% | 58,982,400 / 117,964,800 | 910,209.092 / 910,209.281 | 99.999979% | 113,776.136 B |

The deterministic exactly-1% target uses threshold `0.045361161525` bits. Strictly higher-cost
positions are taken first; threshold ties are resolved by global `(frame,y,x)` raster order. Its
packed mask is retained, so the target is a payload rather than a scalar description.

### Complete cost histogram

Every row uses 117,964,800 positions and 910,209.281 primary bits as denominators.

| cost interval, bits `[low, high)` | positions / 117,964,800 | position fraction | bits / 910,209.281 | bit fraction |
|---|---:|---:|---:|---:|
| `[0, 1e-9)` | 0 / 117,964,800 | 0.000000% | 0.000 / 910,209.281 | 0.000000% |
| `[1e-9, 1e-8)` | 58,442,368 / 117,964,800 | 49.542209% | 0.183 / 910,209.281 | 0.000020% |
| `[1e-8, 1e-7)` | 13,338,306 / 117,964,800 | 11.307022% | 0.450 / 910,209.281 | 0.000049% |
| `[1e-7, 1e-6)` | 6,489,129 / 117,964,800 | 5.500903% | 2.491 / 910,209.281 | 0.000274% |
| `[1e-6, 1e-5)` | 8,134,366 / 117,964,800 | 6.895587% | 39.732 / 910,209.281 | 0.004365% |
| `[1e-5, 1e-4)` | 17,503,546 / 117,964,800 | 14.837940% | 589.737 / 910,209.281 | 0.064791% |
| `[1e-4, 1e-3)` | 8,685,680 / 117,964,800 | 7.362942% | 2,726.946 / 910,209.281 | 0.299596% |
| `[1e-3, 1e-2)` | 3,316,045 / 117,964,800 | 2.811046% | 11,026.525 / 910,209.281 | 1.211427% |
| `[1e-2, 0.05)` | 921,377 / 117,964,800 | 0.781061% | 21,254.216 / 910,209.281 | 2.335091% |
| `[0.05, 0.1)` | 282,487 / 117,964,800 | 0.239467% | 20,089.881 / 910,209.281 | 2.207172% |
| `[0.1, 0.25)` | 289,671 / 117,964,800 | 0.245557% | 46,555.897 / 910,209.281 | 5.114856% |
| `[0.25, 0.5)` | 180,707 / 117,964,800 | 0.153187% | 64,366.757 / 910,209.281 | 7.071644% |
| `[0.5, 1)` | 152,169 / 117,964,800 | 0.128995% | 109,346.416 / 910,209.281 | 12.013327% |
| `[1, 2)` | 116,380 / 117,964,800 | 0.098657% | 162,820.121 / 910,209.281 | 17.888207% |
| `[2, 4)` | 71,518 / 117,964,800 | 0.060627% | 199,032.665 / 910,209.281 | 21.866693% |
| `[4, 8)` | 31,353 / 117,964,800 | 0.026578% | 169,114.747 / 910,209.281 | 18.579765% |
| `[8, 16)` | 9,278 / 117,964,800 | 0.007865% | 95,449.358 / 910,209.281 | 10.486529% |
| `[16, 32)` | 420 / 117,964,800 | 0.000356% | 7,793.159 / 910,209.281 | 0.856194% |

Almost half the population costs between 1e-9 and 1e-8 bits and contributes 0.000020% of the total.
The mass is not merely nonuniform; it is effectively absent over most positions.

## The exactly-1% target set

The target is **1,179,648 positions**, **876,748.548 bits**, or **109,593.569 B** of gross incumbent
model cost. That byte-equivalent is an oracle ceiling on this set, not predicted savings: no candidate
is measured, and a real receiver must still model every position and count every video-derived byte.

### By DALI GT class

Classes use the exact MS9/DALI GT field, not decoded-token identity. Lane is deliberately on its own
row.

| GT class | positions / 117,964,800 | area | bits / 910,209.281 | all-class bpp | enrichment vs 0.00771594 mean | positions in top 1% / 1,179,648 | bits in top 1% / 876,748.548 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Road | 27,407,372 / 117,964,800 | 23.2335% | 361,105.229 / 910,209.281 | 0.013175 | 1.71x | 564,488 / 1,179,648 | 338,649.597 / 876,748.548 |
| **Lane** | **690,754 / 117,964,800** | **0.5856%** | **305,463.969 / 910,209.281** | **0.442218** | **57.31x** | **271,112 / 1,179,648** | **302,350.519 / 876,748.548** |
| Undrivable | 58,413,067 / 117,964,800 | 49.5174% | 103,472.046 / 910,209.281 | 0.001771 | 0.23x | 162,090 / 1,179,648 | 99,576.014 / 876,748.548 |
| Movable | 1,460,386 / 117,964,800 | 1.2380% | 92,960.609 / 910,209.281 | 0.063655 | 8.25x | 113,360 / 1,179,648 | 91,132.165 / 876,748.548 |
| MyCar | 29,993,221 / 117,964,800 | 25.4256% | 47,207.428 / 910,209.281 | 0.001574 | 0.20x | 68,598 / 1,179,648 | 45,040.252 / 876,748.548 |

Lane alone carries **33.559751%** of all bits from **0.585559%** of positions. It carries 34.485431%
of the top-1% target's bits. Road plus Lane carries 73.111055% of target bits. Movable is the other
strongly enriched class at 8.25x mean bpp. The prior prediction that Lane costs materially more per
position is confirmed on this object.

### By time

| frames | positions / 117,964,800 | bits / 910,209.281 | marginal bpp | prefix bpp through block end | top-1% bits / 876,748.548 |
|---|---:|---:|---:|---:|---:|
| 0–99 | 19,660,800 / 117,964,800 | 158,174.655 / 910,209.281 | 0.008045 | 0.008045 | 152,556.791 / 876,748.548 |
| 100–199 | 19,660,800 / 117,964,800 | 155,649.458 / 910,209.281 | 0.007917 | 0.007981 | 149,692.123 / 876,748.548 |
| 200–299 | 19,660,800 / 117,964,800 | 136,654.440 / 910,209.281 | 0.006951 | 0.007638 | 131,182.660 / 876,748.548 |
| 300–399 | 19,660,800 / 117,964,800 | 134,010.440 / 910,209.281 | **0.006816** | **0.007432 minimum** | 129,092.684 / 876,748.548 |
| 400–499 | 19,660,800 / 117,964,800 | 150,701.004 / 910,209.281 | 0.007665 | 0.007479 | 145,551.856 / 876,748.548 |
| 500–599 | 19,660,800 / 117,964,800 | 175,019.284 / 910,209.281 | **0.008902** | 0.007716 | 168,672.433 / 876,748.548 |

The shipped model shows the same qualitative late-clip degradation EF1 saw generically. Its prefix
average bottoms through frame 399, then rises. The last 100-frame marginal is 30.60% above the
300–399 marginal and contains 19.2384% of target bits. Frame 522 is the single most expensive frame
at 2,991.980 bits / 196,608 positions = 0.015218 bpp; frames 517–519 also sit in the top ten.

### By HPAC group and raster site

The complete receipt contains all 190 group rows and a retained 384x512 spatial aggregate. Group
sizes differ under `g=(x mod 64)+2*(y mod 64)`, so both per-position and total-target views matter.

| group | positions / 117,964,800 | all bits / 910,209.281 | bpp | top-1% positions / 1,179,648 | top-1% bits / 876,748.548 |
|---:|---:|---:|---:|---:|---:|
| 0 | 28,800 / 117,964,800 | 1,685.031 / 910,209.281 | **0.058508** | 1,914 / 1,179,648 | 1,631.094 / 876,748.548 |
| 2 | 57,600 / 117,964,800 | 1,677.742 / 910,209.281 | 0.029127 | 1,897 / 1,179,648 | 1,615.085 / 876,748.548 |
| 1 | 28,800 / 117,964,800 | 576.522 / 910,209.281 | 0.020018 | 477 / 1,179,648 | 558.632 / 876,748.548 |
| 4 | 86,400 / 117,964,800 | 1,632.881 / 910,209.281 | 0.018899 | 2,211 / 1,179,648 | 1,581.645 / 876,748.548 |
| 6 | 115,200 / 117,964,800 | 1,894.039 / 910,209.281 | 0.016441 | 2,280 / 1,179,648 | 1,839.845 / 876,748.548 |

The groups carrying most target bits are different because they have larger populations: group 60
has 8,945.032 target bits, group 62 has 8,902.838, and group 58 has 8,727.740, each about 1% of the
target. No single HPAC group dominates the target.

The most expensive repeated raster sites are `(y=286,x=384,g=60)` at 535.988 bits / 600 positions
= 0.893314 bpp; `(286,448,60)` at 485.203 / 600 = 0.808672 bpp; and `(292,0,72)` at
431.727 / 600 = 0.719546 bpp. The full top-100 site table and full site aggregate remain in the
machine receipt; this memo does not promote three sites into a global mechanism claim.

## Exact MS9 Seg-error join

MS9 landed while this arm was active. Its final-capture receipt changed SHA after the charter, so the
instrument refused the stale receipt and rebound only after verifying the unchanged exact DX2 archive,
GT/label identities, 23,757 numerator, mask manifest, and packed final-error mask.

| population | positions / 117,964,800 | bits / 910,209.281 | bpp | enrichment vs mean |
|---|---:|---:|---:|---:|
| all positions | 117,964,800 / 117,964,800 | 910,209.281 / 910,209.281 | 0.007716 | 1.00x |
| MS9 final Seg errors | **23,757 / 117,964,800** | **47,927.054 / 910,209.281** | **2.017387** | **261.46x** |

The spatial coincidence is strong but asymmetric:

- 21,548 / 23,757 Seg-error positions (**90.7017%**) lie inside the global top 1%;
- those overlaps carry 47,893.520 / 47,927.054 Seg-error-position bits (**99.9300%**);
- yet all Seg-error locations carry only **5.265498%** of stream-model bits, or 5,990.882 B;
- Seg errors are only 23,757 / 1,179,648 = **2.013906%** of the top-1% target population.

Therefore the axes are not disjoint: a sharply enriched joint location set exists. They are also not
the same target: 94.73% of rate mass lies outside current Seg-error locations. This is
`[contest-CUDA T4 component-only exact field replay]` spatial evidence, not causality. No intervention
was measured, and cheap-to-code / expensive-to-code remains distinct from score importance.

## Prior-law verdict and campaign implication

All three falsifiable sub-predictions are confirmed on this exact object:

1. top 1% `>50%`: measured **96.3238%**;
2. Lane materially higher bpp: measured **57.31x** mean;
3. late-clip rise: prefix minimum through frame 399, last marginal **0.008902 bpp**.

The `<25%` diffuse falsifier is decisively false. A globally uniform improvement is not the only
possible shape; the incumbent's cost is a named, retained tail. The broad target is the exactly-1%
mask, with Road/Lane carrying 73.11% of its bits and frames 500–599 carrying the largest 100-frame
share. The maximum gross incumbent cost available on that set is 109,593.569 B. This says where a
successor must earn savings and how large its oracle ceiling is. It does **not** say how, and this arm
proposes no mechanism.

The inherited negative boundaries remain intact:

- EF1's ZPAQ/PPMd FAMILY negatives remain generic-estimator negatives, not floors.
- CX3's named summaries remain worse model-inclusive challengers; this arm did not re-race contexts.
- TO2's orderings remain substitutes for the HPAC law; this arm did not reorder.
- RB1's seven coder forms remain 0 B on their isolated streams; this arm read the incumbent coder.
- LQ1 still requires a collateral column for any Lane-targeted successor. Location is not licence.
- VF1's absence of a token-sensitivity corpus remains true as a sensitivity statement. This is a rate
  cost field, not evaluator sensitivity.

## RECALL EVIDENCE

The pre-measurement recall searched `.omx/research` by content, the canonical research indexes,
`sub015_DAG_*` FEED blocks, `canonical_task_status.jsonl`, active claims, `main_hot_state.md`, and the
canonical-equation catalog. Terms included `DX2`, `HPAC`, `RC64`, `113777`, `910216`, `per-position`,
`ideal bits`, `bits_per_frame`, `Lane`, `token allocation`, `seg error`, and the exact stream/field
SHAs. The shipped decoder, corrector, encoder-finish source, TO2 custody tree, MS9 mask manifest, and
prior DC1/DC1S instruments were then read directly.

The beyond-charter finding that changed execution was DC1S. It had already replayed the same
113,777-byte token member over n600, retained a 910,209.4321425341-bit per-frame ledger, and proved a
20-frame corrector checkpoint architecture. Its checkpoints kept all group totals and only the
non-MAP probability rows needed by its hash search; they did **not** retain all 117,964,800 rows, so
they could not answer this charter. This arm reused the ledger as a per-frame positive control and
adopted the proven complete-state checkpoint discipline instead of falsely calling DC1S a full field.

The canonical equation search found no same-object per-position allocation law. A generic
uniform-byte leverage equation was not transferred because it describes score leverage, not this
receiver's probability mass. The recall therefore changed controls and resumability, not the
registered concentration decision.

## Custody, reproducibility, and verification

- Instrument: `experiments/ddm_bl1_per_position_bit_allocation.py`. The exact executed source is
  retained under the measurement root at SHA-256
  `26e8298ebd14a7c2db9463359366d4b6f4ce0f0154a65fa44dbae71d5691da32`. The committed file differs
  only by an inline `gitleaks:allow` comment identifying TO2's public decoded-token SHA as a content
  digest; no executable token changed. Stage receipts remain bound to the retained executed bytes.
- Durable root: `/Volumes/VertigoDataTier/pact/ddm_bl1_per_position_bit_allocation/measurement_v1/`;
  total retained size 3.8 GiB. APDataStore was read-only source custody, never the output tier.
- Primary field: little-endian float64, shape `(600,384,512)`, 943,718,400 B, SHA-256
  `99d7833d55a9aa128f67cfc125a10ba90cffaa714de0b88801aa43b8b16e2c86`.
- Auxiliary float32-input cost field: little-endian float64, same shape/bytes, SHA-256
  `4a79b8e079976f4166f3e9c31dcf512dd0918235c97144736c3b0d97a36aee4d`.
- Instrumented decoded field: uint8, 117,964,800 B, SHA-256
  `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`.
- Exactly-1% packed mask: 14,745,600 B, SHA-256
  `f48cd9d61c4580dda23dc1ff4c7504009612863760ad962c578c190114ce0bdf`.
- `RESULT.json`: 318,937 B, SHA-256
  `f8835acf27c3b46bf95f7cd1954e08d72d591854f8f78ac6c902889a064b6621`.
- `MANIFEST.json`: 56,421 B, SHA-256
  `0b2ca8ec51738b6e7ee5940d262be7226457fcd5a4f8e56f4bfb5b98184a59ac`.
- Thirty distinct 20-frame stages preserve both cost payloads, decoded tokens, full 127-array
  adaptive-corrector state, RC64 decoder state, prior frame, hashes, and source binding.
- A resume-only repeat revalidated every stage and reproduced the result and manifest SHAs exactly.
- An independent audit rehashed all 191 manifest artifacts, re-summed 910,209.280609 bits, recounted
  exactly 1,179,648 target positions and 23,757 MS9 errors, and confirmed all 30 stage bounds.
- `py_compile`, Ruff, decoder-state self-test, two genuine review-tracker passes, and the targeted
  ALWAYS KEEP THE PAYLOAD audit passed; the latter found 0 findings in 1/1 source file.
- No upstream mutation, receiver-custody mutation, jo1 r9 access, scorer, Modal, Metal, archive build,
  index/stash operation, or `main_hot_state.md` edit occurred. Lane claim/closure used the canonical
  append-only dispatch tool.

## Follow-on disposition

`QUEUED-WITH-A-FIRE-ORDER`: MAIN owns a separately chartered successor, not this measurement arm.
Its consumer store is `/Volumes/VertigoDataTier/pact/<new-claimed-rate-lane>/RESULT.json`. The fire
trigger is a non-duplicate lane claim plus a pre-registered, receiver-closed candidate whose complete
video-derived payload is counted, whose retained inverse/collateral surfaces consume this primary
field and exact top-1% mask, and whose local byte-close gate beats 113,777 B before any scorer request.
LQ1 collateral and the MS9 overlap/non-overlap columns are mandatory. No candidate satisfying that
trigger exists in this arm.

OWN-VEHICLE FRONTIER: UNMOVED — DX2 remains S=0.14821987563243377 @ 180,368 B [contest-CUDA T4 n600], archive SHA-256 976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674; ddm_bl1 made no score claim.
