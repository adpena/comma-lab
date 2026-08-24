# ddm_msr1 — the manufactured Seg error is a BALANCED two-way exchange across five class interfaces, and that balance closes every boundary-moving zero-byte actuator at an oracle ceiling of 1 pixel

**Disposition:** `REPRODUCED-N600 / FAMILY-BOUND-DERIVED / ZERO-BYTE-BOUNDARY-FAMILY-CLOSED`
**verdict_scope:** `FAMILY:boundary-moving-actuators × INSTANCE:dx2 (archive sha 976f706d…, 180,368 B) × n600`
— the bound is actuator-independent inside the boundary-moving family; it does **not** bind actuators
outside that family (see §6 for the two that survive).
**axes:** support and terminal field `[contest-CUDA T4 component field, n600]`; native/preuint8/uint8
intermediate logits `[macOS-CPU advisory, n600]`. **score claim: false.** No archive changed, no scorer
fired, no pointer moved.

## Result first

**The manufactured Seg error is 90.12% BALANCED bidirectional flow.** Of the 21,493 manufactured
pixels, 19,370 pair off as `a→b` against an equal-and-opposite `b→a` across the same class-pair
interface. Only **2,123 (9.88%)** are net imbalance.

This is not a statistic — it is a **family bound**. Any actuator that moves the painted boundary of
interface (a,b) in one direction over a region fixes the `a→b` flow there and, one for one, deepens
the `b→a` flow. Balanced flow inside an actuator's addressing cell is unreachable *by construction*,
whatever the actuator is: learned renderer weights, a hand-written palette bias, a class-confidence
shift, or a shipped mask. So the flow-balance ceiling is **2,123 px = 2,702.8 B = 6.38% of the
42,382 B demand**, before any collateral is charged.

**Charging the real collateral collapses it to essentially zero.** (The flow-balance ceiling above is
over all 21,493 manufactured pixels; the collateral ledger below is over the 16,917 charged to the
native render — the only stage a render-side boundary move can reach. §5.3 states the split exactly.)
The actuator-matched ledger — move
interface (a,b) toward `a`, repair `a→b` manufactured pixels whose frozen-head deficit is below the
move size δ, break currently-correct token-class-`b` pixels touching `a` whose top-1 margin is below
δ — was evaluated over all **20 (interface × direction) rows × 13 values of δ from 0.0005 to 2.0 logits**,
taking the best direction and best δ per interface independently (an oracle no single actuator gets).
The result over the whole search:

> **1 net error removed. 1.27 B. 0.0047% of the manufactured mass. 0.003% of the demand.**

Nineteen of the twenty rows never net positive at any δ. The reason is a constant: the collateral population is
**16.6× to 646.5×** the repairable one, while the frozen scorer's margin field separates the two
populations by only **~36–52×** — and that selectivity does **not** improve as δ→0 (49.6× at δ=0.0005,
42.5× at δ=0.02, 36.0× at δ=0.05, against the 179.4× the shell population ratio requires). The
shortfall never falls below **3.6×** anywhere in four decades of δ (3.6× at 0.0005, 4.2× at
0.02, 5.0× at 0.05 — it widens as δ grows). This is a property of the margin field, not
an artefact of sampling granularity.

**PRIOR-LAW PREDICTION: CONFIRMED, by a wide margin.** The charter predicted the zero-byte-removable
fraction would be "under 20% of the manufactured mass" and set a falsifier at ">50% removable at zero
byte change". Measured ceiling for the boundary-moving family: **0.0047%** with collateral charged,
**9.88%** with collateral set to zero. Both are under 20%; the falsifier is nowhere near met. The
sharp-optimum law survives its first structural (rather than actuator-by-actuator) test.

**A recall finding MAIN must route.** `ddm_mf1_manufactured_seg_repair_20260823.md` (commit
`b0c2869ce4`) had already executed scopes 2, 3 and 4 of this charter — localization by
stage/class/band/margin/boundary-distance, source-level mechanism adjudication, and the identical
ceiling table — and had measured two repair formulations with pose. **The msr1 charter cites neither
mf1 nor its file.** I reproduced mf1's numbers independently (they hold exactly; see §2) and re-aimed
this arm at the one question mf1 explicitly left open: *"this does not close jointly retraining the
counted renderer weights so that the repair is internalized without a shipped mask."* §5 closes it for
the boundary-moving family; §6 states what remains.

**Also: my charter repeats an unsupported number.** Its PRIOR-NEGATIVE-SIGNAL line "rj1 W64 refused
3.51×, pose 97.70%" is not supported by the rj1 memo — mf1 checked this at source and found rj1's
disposition is `MECHANISM-INCOMPLETE-WITHHELD` with all distortion columns UNMEASURED. I do not repeat
the 3.51×/97.70% figures as facts and neither should the next charter.

---

## 1. Currency, cited not re-derived

| quantity | value | source |
|---|---:|---|
| exchange rate | `6.658590e-07` S/B | `ddm_tx1_toolbox_crosswalk_20260819.md` §0 — **CITED** (#1207), not re-derived |
| bytes per Seg flip | `1.273108125702622` B | `100/117,964,800 ÷ 6.658590e-07`, DERIVED from the cited rate |
| demand | 42,382 B | `ddm_fb1_sub012_feasibility_bound_20260823.md` — CITED |
| dx2 archive | 180,368 B, sha `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` | canonical frontier pointer, `effective_frontier` |
| per-pixel address payload | 35,969 B Brotli-q11 | `ddm_mf1_manufactured_seg_repair_20260823.md` — MEASURED there, CITED here |

**Both currencies (per `ddm_tl1_teacher_ledger_20260822.md`):** the demand is *either* 42,382 B shed at
fixed distortion *or* 150 B shed at zero distortion, or any mix at the rate above. Every byte figure in
this memo is the fixed-distortion currency; divide by 282.5 for the equivalent zero-distortion cut.

---

## 2. Scope 1 — the decomposition REPRODUCED at source, not quoted

Producer: `experiments/ddm_msr1_manufactured_seg_characterize.py`. Every figure below is recomputed
from the retained **primary** fields, not from mst1's derived masks and not from any quoted ratio.

| receipt | bytes | sha256 | role |
|---|---:|---|---|
| `retained/inputs/tokens_cpu_stage_complete.u8` | 117,964,800 | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` | `L`, decoded token field |
| `retained/inputs/gt_argmax_n600.npy` | 117,964,928 | `91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248` | `G`, contest-CUDA DALI GT |
| `retained/inputs/cuda_terminal_argmax_n600.npy` | 117,964,928 | `e89e1ac083e5964975a1b4121cd1bc8bd91236256d6922f66c650246d7783c34` | `A`, contest-CUDA terminal |
| `retained/assembled/argmax_native_n600.npy` | 117,964,928 | `c09796746f121d7580080d945f219e7519a91cf7990d1e34e2cc13103b7f49db` | `B1`, native observation |
| `retained/assembled/argmax_preuint8_n600.npy` | 117,964,928 | `fe76c2b9fa6c9607709965ead2550f4ca40abf5f099649822333cb5fa3d9450b` | `B2` |
| `retained/assembled/argmax_uint8_n600.npy` | 117,964,928 | `68f5ad9604090ebdd978d9d3a15c07c04c84671854cbbd7d67969f5b8ce8037c` | `B3` |
| `retained/provenance_sources/archive.zip` | — | `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` | the dx2 body itself |
| `retained/provenance_sources/renderer_source.py` | — | `15fc70a9beb4d3840d54344692fb601229a7c1bc8b5449b666d802d76b9eaff2` | the receiver, read at source |

Store root: `.omx/tmp/arm_receipts_local/ddm_mst1_manufactured_stage_split/capture_r2_local/`.
The three pinned field shas match `ddm_mst1_manufactured_stage_split_20260822.md` exactly, and the
`L` sha matches the charter's own categorical-field pin.

| gate | msr1 recomputed | mst1 / fb1 quoted | agree |
|---|---:|---:|---|
| `L != G` transmitted representation errors | **9,182** | 9,182 | ✅ |
| `A != G` final dx2 errors | **23,757** | 23,757 | ✅ |
| `(A != G) ∧ (L == G)` manufactured | **21,493** | 21,493 | ✅ |
| `(A != G) ∧ (L != G)` survived | **2,264** | 2,264 | ✅ |
| `(A == G) ∧ (L != G)` repaired | **6,918** | 6,918 | ✅ |
| manufactured fraction of final | **90.4702%** | 90.47% | ✅ |
| `d_seg` from `A != G` | **0.00020139058431** | 0.00020139 (dx2 authority row) | ✅ |

Earliest-wrong stage attribution of the 21,493:

| stage | msr1 | mst1 | share | byte ceiling | % of demand |
|---|---:|---:|---:|---:|---:|
| native render + frozen head | **16,917** | 16,917 | **78.7093%** | **21,537.2 B** | **50.82%** |
| float bilinear round trip | **4,030** | 4,030 | 18.7503% | 5,130.6 B | 12.11% |
| uint8 | **544** | 544 | 2.5311% | 692.6 B | 1.63% |
| CPU→CUDA terminal | **2** | 2 | 0.0093% | 2.5 B | 0.006% |

State trajectory vs `G`: `L 9,182 → B1 31,503 → B2 24,523 → B3 23,752 → A 23,757`, identical to mst1.

**Both charter figures reproduce exactly. Nothing in the fb1 re-aim rests on an unreproducible ratio.**
Two independent positive controls back this: (i) the recomputed `d_seg` equals the dx2 authority row to
all printed digits, so these fields really are the shipped object; (ii) the per-chunk lexical ordering
was verified to rebuild the assembled native argmax with **0 mismatched pixels** before any per-chunk
logit was joined to `G` — without that control every margin below would have been silently misaligned.

---

## 3. Scope 2 — the manufactured set characterized

Class indices are **self-detected** from `G`'s own spatial/static signature and refused if outside the
declared band; the measured signature reproduces the canonical comma10k order
`[Road, Lane, Undrivable, Movable, MyCar]` (areas 23.2335 / 0.5856 / 49.5174 / 1.2380 / 25.4256 %;
Undrivable rows 9–182, MyCar rows 291–379; temporal IoU 0.9523 / 0.2531 / 0.9943 / 0.8536 / 0.9930).

### 3.1 Per class — counts, S, and bytes

| GT class | manufactured | S | byte ceiling | % of demand | native-stage | native S | native bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Road | 8,385 | 0.0071081 | 10,675.0 B | 25.19% | 6,196 | 0.0052524 | 7,888.2 B |
| Lane | 5,285 | 0.0044802 | 6,728.4 B | 15.88% | 3,944 | 0.0033434 | 5,021.1 B |
| Undrivable | 4,117 | 0.0034900 | 5,241.4 B | 12.37% | 3,786 | 0.0032094 | 4,820.0 B |
| Movable | 2,952 | 0.0025024 | 3,758.2 B | 8.87% | 2,336 | 0.0019803 | 2,974.0 B |
| MyCar | 754 | 0.0006392 | 959.9 B | 2.26% | 655 | 0.0005553 | 833.9 B |
| **all** | **21,493** | **0.0182198** | **27,362.9 B** | **64.56%** | **16,917** | **0.0143407** | **21,537.2 B** |

Lane density is 7,651.1 manufactured per million Lane pixels against 182.2 body-wide — **42.0×** — the
same concentration mst1 reported, reproduced.

### 3.2 Per region — a codimension-1 shell, and nothing else

Chebyshev distance in `L` to the nearest differently-labelled pixel:

| distance | manufactured | share | cumulative | body pixels at that distance |
|---:|---:|---:|---:|---:|
| 1 | **21,420** | 99.66% | 99.66% | 3,062,050 (2.596% of body) |
| 2 | 57 | 0.27% | 99.93% | 2,478,403 |
| 3 | 12 | 0.06% | 99.98% | 2,288,428 |
| 4–7 | 4 | 0.02% | 100.00% | — |
| ≥ 12 (interior) | **0** | 0.00% | — | 94,876,590 (80.4% of body) |

The manufactured set lives on a one-pixel shell that is 2.596% of the body — an enrichment of
**38.4×**. Zero manufactured pixels sit more than 7 pixels from a token boundary; zero sit in an
interior. This reproduces mf1's 4-neighbour result (99.15%, 45.77×) at the 8-neighbour definition,
which is the wider set, so the two agree.

### 3.3 Per margin band — hairline in the scorer's own units

Native-stage deficit = `native logit[argmax] − native logit[GT class]`, from the retained logits.

| quantile | deficit | | threshold | count | share | ceiling |
|---|---:|---|---|---:|---:|---:|
| q0.10 | 0.0318 | | < 0.05 | 2,682 | 15.85% | 3,414.5 B |
| q0.25 | 0.0781 | | < 0.10 | 5,439 | 32.15% | 6,924.4 B |
| **q0.50** | **0.1621** | | **< 0.25** | **11,512** | **68.05%** | 14,656.0 B |
| q0.75 | 0.2997 | | < 0.50 | 15,258 | 90.19% | 19,425.1 B |
| q0.90 | 0.4955 | | < 1.00 | 16,663 | 98.50% | 21,213.8 B |
| q0.99 | 1.1463 | | < 2.00 | 16,875 | 99.75% | 21,483.7 B |

Mean deficit **0.2320**, against a mean top-1 margin of **5.763** over correct pixels — the
manufactured errors are **24.8× shallower** than a typical correct decision. This independently
reproduces mf1's exact split: deficit ≥ 1 → **254** "deep"; (0.25, 1) → **5,151** "moderate";
≤ 0.25 → **11,512** "hairline". All three match mf1 to the unit, from primary logits.

### 3.4 Per pair — diffuse, not a bad-frame problem

min 8 · median 30 · mean 35.8 · max 237 · **pairs with zero manufactured error: 0** · top-10 pairs
carry 5.94% · top-60 (10% of pairs) carry 22.71%. The error is spread across every frame in the
sequence, so no per-frame actuator (the renderer's 8-dim frame FiLM code is per-pair) has a
concentrated target.

### 3.5 Per interface — the shape that decides everything

Manufactured pixels as directed flows `GT a → terminal b`:

| interface | a→b | b→a | gross | **balanced** | net | balanced share |
|---|---:|---:|---:|---:|---:|---:|
| Road\|Lane | 4,045 | 5,153 | 9,198 | 8,090 | 1,108 | 88.0% |
| Road\|Undrivable | 2,125 | 2,009 | 4,134 | 4,018 | 116 | 97.2% |
| Undrivable\|Movable | 2,102 | 1,839 | 3,941 | 3,678 | 263 | 93.3% |
| Road\|Movable | 1,545 | 1,085 | 2,630 | 2,170 | 460 | 82.5% |
| Road\|MyCar | 670 | 576 | 1,246 | 1,152 | 94 | 92.5% |
| Lane\|MyCar | 99 | 165 | 264 | 198 | 66 | 75.0% |
| Lane\|Movable | 29 | 16 | 45 | 32 | 13 | 71.1% |
| Movable\|MyCar | 12 | 13 | 25 | 24 | 1 | 96.0% |
| Lane\|Undrivable | 4 | 6 | 10 | 8 | 2 | 80.0% |
| Undrivable\|MyCar | 0 | 0 | 0 | 0 | 0 | — |
| **total** | | | **21,493** | **19,370 (90.12%)** | **2,123 (9.88%)** | |

Five interfaces carry **98.4%** of the mass, and every one of them is **near-symmetric in both
directions**. That symmetry is the finding: the renderer is not systematically dilating or eroding any
class — it is placing each boundary with roughly zero-mean one-pixel jitter.

### 3.6 The token field is not the cause

Fraction of each pixel's 17×17 token window where `L == G`: **13,625 of 21,493 (63.4%)** manufactured
pixels had a window in which `L` equals `G` at **every one of 289 cells**; a further 7,505 (34.9%) had
at most ~2 wrong cells (fidelity in [0.99, 0.999)). The renderer had a locally perfect input and still
produced RGB the frozen head misreads. **`L` is exonerated as the mechanism** at the 98.3% level.

---

## 4. Scope 3 — addressable vs structural, with attribution method and falsifier

| # | proposed sub-class | verdict | how attributed | what would falsify the verdict |
|---|---|---|---|---|
| a | **renderer finite capacity** (too few bytes / too small a model) | **NOT SUPPORTED as the mechanism** | Capacity failures produce interior error clouds and confidently-wrong outputs. Measured: **0** manufactured pixels in any token interior (§3.2), and the median deficit is 0.162 against a 5.763 correct-margin mean (§3.3) — the renderer is not confidently wrong anywhere, it is marginally wrong on a shell. | A width/byte sweep on the current vehicle that lowers native manufactured count materially at fixed rate. Note `ddm_wd3` already found fresh-init width distillation FAMILY-NEGATIVE at 65 ep. |
| b | **paint / prototype colours** | **REFUTED, two independent ways** | (i) Source: `SemanticTokenRenderer` has no palette — colours are a learned, coordinate-mixed, frame-FiLM-conditioned CNN output (`renderer_source.py` sha `15fc70a9…`), the same source refutation mf1 made. (ii) Measurement: the interior painted separations are Undrivable\|MyCar **18.25** (closest) carrying **0** manufactured pixels, while Road\|Lane at **156.62** (4th farthest of ten) carries **42.8%** of the mass. Palette proximity and error are, if anything, **anti**-correlated. | An interface found where painted separation is small *and* error is high, or a measured colour-only edit that lowers native manufactured count. |
| c | **quantization** | **BOUNDED AND SMALL** | Camera uint8 is charged **544 px = 2.53% = 692.6 B** by earliest-stage attribution (§2), and uint8 globally *repairs* 1,624 while breaking 853. | A measured uint8-aware candidate that beats the 692.6 B ceiling; the ceiling itself is arithmetic, not opinion. |
| d | **none of the above — frozen-head boundary jitter on a synthetically painted discontinuity** | **CONFIRMED as the mechanism, at the observation level** | The conjunction of §3.2 (99.66% on the one-pixel shell, 0 interior), §3.3 (hairline, 24.8× shallower than correct), §3.5 (90.12% balanced two-way flow across five interfaces), §3.6 (locally perfect token input), and §3.4 (every pair affected). Zero-mean one-pixel boundary jitter is the only reading consistent with all five. | Any of: a manufactured pixel population off the shell; a systematically one-directional interface; a manufactured population with deep margins. None exists in n600. |

**A causal renderer-only / head-only split is NOT claimed.** The frozen SegNet forward and argmax are
inseparable from every RGB observation, exactly as mst1 and mf1 record. "Native render + frozen head"
is one unseparated observation throughout.

---

## 5. Scopes 4 & 5 — the zero-byte ceiling, priced in both currencies

### 5.1 The flow-balance bound (collateral set to zero — generous)

An actuator addresses a set of cells; inside a cell it can move interface (a,b) one way. Balanced flow
inside a cell cancels. Reachable pixels as addressing granularity increases:

| addressing granularity | active (cell × interface) groups | reachable px | repair value | address cost (counting lower bound: 1 sign bit/group) |
|---|---:|---:|---:|---:|
| one sign per interface | 9 | **2,123** | **2,702.8 B** | 1.1 B |
| × pair | 2,927 | 8,473 | 10,787.0 B | 365.9 B |
| × pair × 96×128 cells | 7,689 | 13,051 | 16,615.3 B | 961.1 B |
| × pair × 48×64 | 10,294 | 15,241 | 19,403.4 B | 1,286.8 B |
| × pair × 24×32 | 12,954 | 17,359 | 22,099.9 B | 1,619.2 B |
| × pair × 12×16 | 15,317 | 18,867 | 24,019.7 B | 1,914.6 B |
| × pair × 6×8 | 17,493 | 20,117 | 25,611.1 B | 2,186.6 B |
| × pair × 3×4 | 19,192 | 20,975 | 26,703.4 B | 2,399.0 B |
| × pair × per-pixel | 21,493 | 21,493 | 27,362.9 B | 2,686.6 B — but **MEASURED 35,969 B** (mf1) |

**Read this table honestly.** The counting bound is far too loose: it charges nothing for *locating*
the groups, and mf1's measured Brotli-q11 per-pixel address is **13.4×** the counting bound at that
row. So the ladder does **not** close the family on rate grounds at coarse granularity — at the
interface-global row the address is ~1 B and the oracle value is 2,702.8 B. **The ladder's honest
conclusion is only that the flow-balance ceiling at zero addressing is 2,123 px / 2,702.8 B / 6.38% of
demand, and that mf1's measured per-pixel address (35,969 B) already exceeds even the perfect-repair
value (27,362.9 B) by 8,606 B.**

### 5.2 The collateral ledger — which is what actually closes it

Undirected move of size δ on the whole token-boundary shell (3,062,050 px; 16,866 native-stage
manufactured, 3,026,032 currently correct; collateral ratio **179.4:1**):

| δ (logits) | repaired | broken | net errors removed | net value |
|---:|---:|---:|---:|---:|
| 0.0005 | 26 | 94 | **−68** | −86.6 B |
| 0.001 | 50 | 174 | **−124** | −157.9 B |
| 0.005 | 249 | 1,014 | **−765** | −974.0 B |
| 0.02 | 1,062 | 4,485 | **−3,423** | −4,357.8 B |
| 0.05 | 2,677 | 13,354 | **−10,677** | −13,593.0 B |
| 0.25 | 11,481 | 142,749 | **−131,268** | −167,118.4 B |
| 1.0 | 16,612 | 1,289,726 | **−1,273,114** | −1,620,811.8 B |

Selectivity — the ratio of the repaired *rate* to the broken *rate* — is **49.6× / 51.6× / 44.1× /
42.5× / 36.0×** at δ = 0.0005 / 0.001 / 0.005 / 0.02 / 0.05. It never approaches the **179.4×** the
population ratio demands, and it does not improve toward δ→0. That the shortfall is 3.6× at its
narrowest and widens with δ is why the undirected family cannot be rescued by tuning the move size.

### 5.3 The directional, actuator-matched oracle — the binding bound

The undirected ledger over-charges: a real actuator only touches its own interface, in one direction.
The matched ledger repairs `a→b` pixels on the (a,b) interface and endangers only correct token-class-`b`
pixels touching `a`. Best direction and best δ chosen **independently per interface** (an oracle):

| interface → move toward | repairable | endangered correct | collateral | best net over all 13 δ |
|---|---:|---:|---:|---:|
| Road\|Lane → Road | 3,060 | 599,879 | 196.0× | **0** |
| Road\|Lane → Lane | 3,818 | 858,199 | 224.8× | **0** |
| Road\|Undrivable → Undrivable | 1,917 | 288,141 | 150.3× | **0** |
| Road\|Undrivable → Road | 1,255 | 284,891 | 227.0× | **0** |
| Undrivable\|Movable → Undrivable | 1,837 | 93,120 | 50.7× | **0** |
| Undrivable\|Movable → Movable | 1,476 | 101,878 | 69.0× | **0** |
| Road\|Movable → Road | 1,348 | 88,836 | 65.9× | **0** |
| Road\|Movable → Movable | 813 | 91,332 | 112.3× | **0** |
| Road\|MyCar → Road | 504 | 316,944 | 628.9× | **0** |
| Road\|MyCar → MyCar | 490 | 316,791 | 646.5× | **0** |
| Lane\|MyCar → MyCar | 100 | 5,107 | 51.1× | **0** |
| Lane\|MyCar → Lane | 88 | 5,936 | 67.5× | **0** |
| Lane\|Movable → Lane | 27 | 2,826 | 104.7× | **0** |
| Lane\|Movable → Movable | 14 | 1,466 | 104.7× | **0** |
| Movable\|MyCar → Movable | 11 | 199 | 18.1× | **+1** (δ = 0.05) |
| Movable\|MyCar → MyCar | 11 | 183 | 16.6× | **0** |
| Lane\|Undrivable, Undrivable\|MyCar (4 rows) | ≤ 5 | ≤ 2,515 | 34–629× | **0** |
| **oracle total** | 16,778 | — | — | **+1 px = 1.27 B** |

**Scope of this oracle, stated exactly.** It covers the **native-render stage only** — 16,917 px,
78.71% of the manufactured mass, 21,537.2 B. Its 16,778 repairable pixels are 99.18% of that stage;
the missing 139 have no adjacent token boundary to the class they are misread as, so no boundary move
reaches them at all. The other **4,576 px (21.29%, 5,825.7 B)** are first wrong at the float round
trip (4,030), uint8 (544) or the CPU→CUDA terminal (2) — introduced by frozen operators *after* the
render, so no native boundary move reaches them either, but this instrument does not bound them and
neither did mf1: their repair columns stay **UNMEASURED**. The honest total statement is therefore:
zero-byte boundary-moving reach is bounded at **1 px** on 78.71% of the mass and **UNMEASURED** on the
remaining 21.29%, whose own ceiling is 5,825.7 B even under perfect repair.

### 5.4 Both currencies

| bound | pixels | S | fixed-distortion currency | zero-distortion currency | % of demand |
|---|---:|---:|---:|---:|---:|
| perfect removal of ALL manufactured (oracle, unreachable) | 21,493 | 0.0182198 | 27,362.9 B | 96.9 B | 64.56% |
| perfect removal of the native stage only | 16,917 | 0.0143407 | 21,537.2 B | 76.2 B | 50.82% |
| **flow-balance ceiling, zero collateral, zero addressing** | **2,123** | **0.0017997** | **2,702.8 B** | **9.6 B** | **6.38%** |
| **actuator-matched oracle with real collateral** | **1** | **0.0000008** | **1.27 B** | **0.0045 B** | **0.003%** |

---

## 6. What this does NOT close — stated as scope, not as hope

The bound is a bound on **boundary-moving** actuators. Two things survive it, and one of them I
measured:

1. **Boundary SHARPENING** — raising the frozen head's confidence on *both* sides at once rather than
   moving the boundary. Not covered by §5. **Probed and found without an address.** Cross-boundary
   rendered contrast (largest native-RGB L2 distance to an 8-neighbour of a different token class) is
   mean **236.60** at manufactured shell pixels versus **243.37** at correct shell pixels — a ratio of
   **0.972**, and at the 1st percentile the manufactured pixels are *sharper* (76.81 vs 55.69). The
   distributions coincide. The render's own edge contrast does not discriminate broken from correct
   boundary pixels, so a sharpening actuator has no more of an address than a moving one.
   Producer: `experiments/ddm_msr1_edge_contrast_probe.py`.
2. **A joint re-solve that changes the OBJECT the bound was priced on** — a different token field, a
   different receiver family, or a renderer whose boundary is not a hard categorical step. Per
   `ddm_sy2_composition_synergy_deep_pass_20260823.md` (#1227), a closed leg survives only when another
   leg first changes the object. That is the only door left on this axis, and it is not a repair of dx2.

**And pose, which I did not measure and which binds independently.** PoseNet reads the same master
frame SegNet reads, and per `ddm_pz1` the two scorers make the *identical* resize call, so a
seg-targeted RGB edit lands in PoseNet's input on the same lattice. mf1 MEASURED this on exactly this
support: its best oracle row bought Δ`S_seg` = **−0.00041326** and paid Δ`S_pose` = **+0.01946572** —
pose is **47.1×** the seg gain — even before its 35,969 B address. Every row in §5 is a **Seg-only**
bound with its **pose column UNMEASURED**; adding pose can only make each row worse.

---

## 7. Attacking my own conclusion

- **The one assumption.** §5.2/§5.3 assume a boundary move of magnitude δ reaches both populations.
  Both populations sit exactly one pixel from the same interface, so their exposure is symmetric by
  construction — but the assumption is stated, and it is **falsified by any actuator measured to move
  manufactured deficits without moving correct-pixel margins comparably on the same shell.** No such
  actuator exists in our evidence; mf1's oracle, which edited *only* the manufactured support, still
  introduced 46 errors while fixing 72, which is direct measured evidence that collateral survives
  perfect addressing.
- **Native logits, terminal outcome.** The barrier margins are native-stage `[macOS-CPU advisory]`
  while the manufactured set is defined at the CUDA terminal. R and uint8 are net *repairers*
  (−6,980 and −771), so terminal is mildly more forgiving for **both** populations; there is no
  mechanism by which that shifts the ratio by the required 3.6× or more. Labelled INFERRED, not measured.
- **Double counting.** §5.3 sums per-interface bests, and a correct pixel touching two classes is
  endangered in two rows. That over-credits the actuator's *savings*; since 19 of 20 rows are ≤ 0 it
  cannot change the verdict.
- **Search scope for the negative-existence claim.** "No boundary-moving actuator reaches more than 1
  net pixel" was searched over exactly: 20 (interface × direction) rows × 13 δ ∈ {0.0005 … 2.0}, plus
  9 addressing granularities from interface-global to per-pixel, on the n600 dx2 object. Outside that
  scope I make no claim.
- **Why did five arms miss a free reduction?** They did not. The bound says there is none to miss —
  which is what the sharp-optimum law predicted, now with a mechanism and a number instead of five
  independent refusals.

---

## 8. Verification and boundaries

- `ruff check` and `py_compile` clean on all three instruments; two genuine adversarial review passes
  (pass 2 produced a real fix: the δ search was extended down to 0.0005 because selectivity is best as
  δ→0 and stopping at 0.02 would have left the negative-existence claim an unsearched corner).
- Positive control: per-chunk lexical order rebuilds the assembled native argmax with **0** mismatched
  pixels, so every per-chunk logit is aligned with `G`.
- Cross-arm control: the margin split (254 / 5,151 / 11,512) and the boundary concentration reproduce
  mf1 independently from primary logits; the byte ceilings reproduce mst1 to 0.1 B.
- ALWAYS KEEP THE PAYLOAD: eight derived payloads persisted with sha256 + byte count and **all eight
  re-verified after AppleDouble cleanup** under
  `/Volumes/APDataStore/pact/ddm_msr1_manufactured_seg_reduction/` (2.0 GB):
  `earliest_stage_id` 117,964,928 B `214d9e11…` · `manufactured_support_packbits` 14,745,728 B
  `619f6583…` · `token_boundary_distance` 117,964,928 B `df0c9316…` · `token_window_fidelity`
  471,859,328 B `2c071f62…` · `native_logit_deficit` 471,859,328 B `1fa39a60…` · `native_top1_margin`
  471,859,328 B `e165bbc6…` · `cross_boundary_contrast` 471,859,328 B `d357352b…` · `reach_masks.npz`
  758,485 B `b19148f7…`. Result JSONs: `MSR1_CHARACTERIZE.json` 18,116 B, `MSR1_REACH.json` 65,895 B,
  `MSR1_EDGE.json` 3,764 B. `PYTHONDONTWRITEBYTECODE=1` set; zero `._*` and zero `__pycache__` remain
  on the ExFAT tree.
- **MEASURED:** every gate, stage, class, region, margin, pair, interface, palette and contrast figure.
  **DERIVED:** every byte ceiling (from the CITED exchange rate) and the flow-balance and collateral
  bounds. **INFERRED:** the native→terminal transfer caveat. **UNMEASURED and explicitly so:** the pose
  response of every row in §5, any candidate archive, any exact score.
- No `upstream/` file, no jf1 receipt, no other arm's surface, no Modal or Metal resource was touched.
  Local CPU only.

## NEXT_IF_RESUMED

`QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN. **Do not charter a third boundary-repair actuator on dx2.**
mf1 measured two and refused; msr1 bounds the whole boundary-moving family at 1 pixel and finds the
sharpening family has no address either. The manufactured-seg axis on **this object** is closed for
zero-byte repair. fb1's re-aim of rj2 toward "the 21,537 B the native-render manufactured error is
worth" should be **withdrawn**: that value is real as an accounting identity and unreachable as a
lever. Fire trigger for reopening: an object-changing leg per sy2 (#1227) — a token field or receiver
whose class boundary is not a hard categorical step — after which this bound must be re-derived, not
transferred.

## LIVE-HYPOTHESES

- The 3.6×-and-widening selectivity shortfall is a property of the frozen head's margin field on a painted
  discontinuity and should reproduce on any receiver that paints hard categorical boundaries. If so it
  is a **design constraint on the next receiver**, not a dx2 fact. Testable on any future object with
  the same three instruments.
- The 9.88% net imbalance (2,123 px, 2,702.8 B) is the only part with a coarse, cheap address
  (~1 B). It is currently swamped by collateral, but it is the only sub-population where an actuator
  with near-zero address cost could ever pay — and Road\|Lane alone carries 1,108 of it.

## DEAD-ENDS

- Boundary-moving zero-byte repair of dx2 manufactured Seg: oracle ceiling **1 pixel / 1.27 B** over
  260 searched (interface × direction × δ) combinations. Closed at family scope.
- Palette / prototype-colour explanation: refuted at source *and* by measurement (closest-painted
  interface carries zero error; 42.8% of the error sits at the 3rd-farthest-painted interface).
- Renderer-capacity explanation as the mechanism: unsupported — zero interior manufactured pixels,
  hairline margins throughout.
- Boundary sharpening as an addressed lever: rendered cross-boundary contrast at manufactured pixels
  is 0.972× that at correct pixels — no discrimination.

---

**STORES CONSULTED:** `.omx/research/ddm_mst1_manufactured_stage_split_20260822.md` (the stage split
this memo re-derives) · `.omx/research/ddm_mf1_manufactured_seg_repair_20260823.md` + commit
`b0c2869ce4` + `experiments/ddm_mf1_manufactured_seg_repair.py` (**the recall hit — prior art the
charter did not cite**) · `.omx/research/ddm_fb1_sub012_feasibility_bound_20260823.md` (routing memo,
demand, two-currency arithmetic) · `.omx/research/ddm_tx1_toolbox_crosswalk_20260819.md` §0 (exchange
rate, CITED) · `.omx/research/ddm_w72_distortion_advisory_20260823.md` and
`.omx/research/ddm_rj1_renderer_joint_move_20260823.md` (charter's prior negatives; the rj1 numbers are
**not** repeated, per mf1's source check) · `.omx/research/ddm_sy2_composition_synergy_deep_pass_20260823.md`
(object-change law) · `.omx/research/ddm_ar1b_archive_residue_purchase_20260822.md` (residue map) ·
`.omx/research/ddm_tl1_teacher_ledger_20260822.md` (two currencies) ·
`.omx/state/canonical_frontier_pointer.json` (object identity) · `CLAUDE.md` (canonical class order,
self-detected and re-verified) · `docs/operating_manual_craft_handoff.md` · the retained mst1 capture
store `.omx/tmp/arm_receipts_local/ddm_mst1_manufactured_stage_split/capture_r2_local/` (primary
fields, chunk logits, `renderer_source.py`, `archive.zip`). Repo grep for prior manufactured-Seg
instruments returned `ddm_mf1_manufactured_seg_repair.py` and `ddm_ms9_dx2_seg_manufactured_fraction.py`;
both were read before any instrument was written.

**Own-vehicle frontier: S 0.14821987563243377 @ 180,368 B, archive sha `976f706d…`
`[contest-CUDA]` — UNMOVED by this arm, by design. This arm produced no candidate and fired no scorer.**
