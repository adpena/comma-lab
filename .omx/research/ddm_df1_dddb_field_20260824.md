# ddm_df1 — `dD/dB` on the token half is BIMODAL, not heavy-tailed: an exact zero mode holding 34,674.11 B (81.81% of the demand) that no addressless rule reaches and no address can afford

**Date:** 2026-08-24 · **Arm:** `ddm_df1` · **Pointer:** UNMOVED · **No Modal job fired.**
**Axis:** `[macOS-CPU advisory / scorer-free shipped-receiver instrumentation]` for the field;
`[macOS-CPU advisory / DALI-lineage GT]` for the scorer rows. `score_claim=false` ·
`promotable=false` · no archive built, none promoted.
**verdict_scope:** `INSTANCE:DX2_archive_976f706d_n600_shipped_HPAC_RC64_law_TOKEN_HALF`

---

## 0. Result first

`dD/dB` on the shipped token stream is **not a heavy tail. It is two modes with an exact
boundary**, and the boundary is arithmetic, not empirical:

> `cost_i < 1 bit` ⟺ `p_sel_i > 0.5` ⟺ `p_sel_i` is the unique row maximum ⟺ the receiver's
> argmax already equals the transmitted symbol ⟺ dropping position `i` leaves the decoded field,
> the render, the SegNet argmax **and PoseNet's input** bit-identical ⟺ `dD_i = 0` **exactly**.

Measured over all **117,964,800** positions of the DX2 body:

| mode | positions | share | bytes | share of stream | `dD` |
|---|---:|---:|---:|---:|---|
| **zero** (`argmax == transmitted`) | **117,737,129** | **99.80700%** | **34,674.11** | **30.4757%** | **exactly 0** |
| **positive** (`argmax ≠ transmitted`) | **227,671** | **0.19300%** | **79,102.05** | **69.5243%** | > 0 |
| total | 117,964,800 | | 113,776.16 | | |

**The zero mode is 34,674.11 B = 81.81% of the 42,381.16 B demand.** Searching `.omx/research/*.md`
as of 2026-08-24 I found no larger zero-distortion byte mass recorded on this vehicle; that is the
scope of the search, not a claim that none exists.

**And it is unreachable, twice over:**

1. **No addressless rule isolates it.** A receiver-side rule can only be a function of the coding
   row, i.e. of `pmax`. The largest `pmax` among the positive mode is **0.9999997615814209** — so
   the only strictly-free threshold is `τ = 1.0`, and the float32-saturated cell it selects holds
   **67,955,679 positions carrying 0.0486 bytes** — a twentieth of one byte. *(Measured, not
   inferred: the τ = 1.0 scorer row breaks 0 labels, changes 0 scored cells, and reproduces the
   control `d_seg` to the last digit.)*
2. **No address can afford it.** Isolating the zero mode requires a side channel carrying "is my
   argmax right?", whose floor under the shipped model is `Σᵢ H_b(pmax_i)` =
   **872,907.97 bits = 109,113.50 B**, which is **3.1468×** what the zero mode yields. The zero
   mode is, exactly, the "yes" branch of that same indicator sub-code — already paid for (§4).

**And the route that trades distortion for it loses by an order of magnitude once it is run for
real.** Running `drop(τ = 0.999)` through the shipped encoder in closed loop, then scoring the
receiver's own reconstructed field through the real render and the frozen SegNet, gives — with no
modelled term anywhere — **12,224 real bytes saved, `d_seg` 0.00020135 → 0.00114282, damage ÷
credit 11.567×, candidate S 0.23423 vs 0.14822 shipped.** Static `−log₂p` accounting had predicted
**0.762× — a win** (§6.2, §6.3).

The prediction registered in the charter is **CONFIRMED on both measurable legs** — Gini 0.995
(> 0.8) and co-location AUC **0.999998753** — but **its stated implication is REFUTED.** The
object is not "maximally hard to shrink because the expensive sites are the error sites." It is
hard to shrink for a different and sharper reason: **30.48% of the stream's bytes are the
irreducible cost of confirming 117.7 million guesses that were already right**, and confirmation
cost cannot be addressed away, only *modelled* away.

---

## 1. The granularity, and why this one

`dD/dB` per **token position** — all 117,964,800 of the `(600, 384, 512)` semantic label lattice
that section 4 of the shipped RX1 member carries.

*(Numbers here are DX2's own, re-derived: the DX2 tail is **113,777 B = 63.08%** of its 180,368 B
archive, at **0.0077160 bits/token**. `ddm_tx1` §0's "109,696 B / 62.2% / 0.0074392 bits/token"
belongs to the **up3** body at 176,420 B and does not describe this object — the same cross-body
substitution that put `d_seg = 0.00030309` into circulation for a body whose own T4 receipt says
**0.00020139**.)*

| requirement | why this index satisfies it |
|---|---|
| a REAL operator can address it | the tail is **63.08%** of the archive; `cpr1/inflate.py:318` renders frame `2p+1` from these tokens and `upstream/modules.py:108` (`x = x[:, -1, ...]`) means SegNet sees only that frame, so a token is a near-one-to-one actuator on an argmax cell (`ddm_jg1` §0) |
| `dD` is the SCORER's `D` | `#1127` measured render amplification ~38,700×, so weight-MSE is the wrong space. Here `dD` is a label change pushed through the shipped `SemanticTokenRenderer` and the frozen CPU SegNet |
| `dB` is exact, not modelled | BL1/TB2 reconciled 910,209.280609 modeled bits against 910,216 physical bits with a 6.719391-bit explained residual |

**The operator: `drop(τ)`.** The receiver already computes its coding row before decoding. Where
`pmax ≥ τ` no token is coded and the receiver substitutes its own argmax. Deterministic on both
sides, so it costs **zero address bits** — which is the only reason it is worth studying, given
`ddm_tba1` measured that naming any subset of this object costs more than the subset holds.

### Declared SCOPE reduction (per the charter-time OPTIMAL-FORM law)

The **renderer half** (30,856 B, 17.1% of the archive) is **not fielded here.** A per-group
`(ΔD, ΔB)` table for it already exists — `experiments/ddm_wd3_scorer_aware_width_distillation.py:848-880`,
`quantization_sensitivity_table` — but its `ΔD` is a **first-order squared-gradient proxy**, and
this lineage has a measured proxy-vs-realized gap of **349×** (`ddm_rj1:151`). A field built on it
would not be a scorer-space measurement, which the charter forbids. This is a scope reduction, not
a mechanism reduction: the token half is the larger half, and it is the only half where a real
removal operator has both an exact per-position index and a true scorer `dD`.

---

## 2. What was reused, and what was built

Reused unchanged (the campaign's instrument inventory, not rebuilt):

| object | used for |
|---|---|
| `experiments/ddm_bl1_per_position_bit_allocation.py` | `source_binding` custody gate, `build_decoder`, `load_receiver`, `rc64_costs`, decoder state save/restore |
| TB2 retained `position_rc64_frequency_cost_bits.f64le.bin` | the `dB` field, and the **law-identity gate** |
| TB2 retained `join_fields/{top_1pct, top_10pct, gross_manufactured_native_render_head}` | the joins; my reader reproduces their published counts exactly (28,602 / 1,179,648 / 11,796,480) |
| `experiments/ddm_jg1_seg_solve.py` | `load_semantic_renderer`, `render_frame1`, `load_segnet`, `argmax_from_camera_frames`, `d_seg_per_pair` — the $0 DALI-lineage seg instrument |
| `experiments/ddm_jg2_tail_reencode.py` | `load_route_b`, `compile_rc64`, `load_runtime`, corrector state capture + the divergent-state DETECTOR |
| `ddm_qs3` retained `gt_argmax_n600.npy` | DALI-lineage GT argmax |

Built (two files, `experiments/ddm_df1_drop_field.py` + `experiments/ddm_df1_drop_frontier.py`):
the instrumented replay that retains the receiver-visible **prediction** state BL1/TB2 did not keep
(coding-row argmax, `pmax`, `psecond`), the field analysis, and a **skip-encode** fork of jg2's
encode loop that is exactly one line deep — encode a SUBSET of each group's rows.

### The law-identity gate

DF1's independently re-derived cost field is **byte-identical** to TB2's retained field:

```
df1_cost_sha256 = tb2_cost_sha256 = tb2_declared_sha256
               = 99d7833d55a9aa128f67cfc125a10ba90cffaa714de0b88801aa43b8b16e2c86
```

That is what licenses joining DF1's new prediction fields to TB2/WJ1/BL1 aggregates position for
position. My Gini of `dB` is **0.9951593787014741** against TB2's published
**0.9951593787014772** — agreement to 13 significant figures, the residual being float summation
order.

### One defect I found in my own code before it produced a number

`coding_prediction` first took the row argmax with `np.argsort(...)[:, -1]`. Exact float32 ties at
the row maximum occur at a **measured** rate of **2.5431315104166665e-06** (10 of 3,932,160
positions in frames 0:20) and **7 of those 10 were flips**, so an `argsort` tie-break would have
manufactured drop-operator flips the real receiver never makes. Replaced with `np.argmax`, whose
first-index rule is the receiver's own tie-break because it is the rule `bl1.rc64_costs` already
applies when it balances the winner frequency. The decode was discarded and re-run. Flip count in
frames 0:20 moved 7,728 → 7,725, so the fix did real work.

---

## 3. The field, full population

### 3.1 Concentration and co-location

| statistic | value |
|---|---:|
| Gini of `dB` | **0.9951593787014741** |
| **AUC(`dD > 0` vs `dB`)** | **0.999998753225156** |
| mean bits, positive mode | 2.7795212271316867 |
| mean bits, zero mode | 0.002356035905264548 |
| **ratio** | **1,179.74×** |

AUC is `P(a random positive-mode position costs more than a random zero-mode position) + ½P(tie)`,
computed from a shared 142-bin log-spaced cost histogram (binned, so within-bin order is treated as
tied). **0.5 would mean `dD > 0` carries no information about `dB`. The measured 0.99999875 is
about as close to perfect co-location as a finite sample permits.** The charter's ANTI-correlation
falsifier does **not** fire; it fires in the opposite direction, hard.

The `top_1pct` join is the same statement without a statistic: **all 227,671 positive-mode
positions lie inside TB2's top 1% of cost.** Enrichment **100.000×**, which is the arithmetic
CEILING (`1/0.01`) — the cell is saturated, not merely large.

### 3.2 Joins against bit mass and against manufactured error

| mask | positions | mask bytes | positive-mode positions | positive bytes | **zero-mode bytes inside the mask** | count enrichment |
|---|---:|---:|---:|---:|---:|---:|
| WJ1 `gross_manufactured_native_render_head` | 28,602 | 6,846.8 | 14,437 | 6,221.5 | 625.3 | **261.53×** |
| TB2 `top_1pct` | 1,179,648 | 109,593.6 | **227,671 (all)** | 79,102.0 | **30,491.5** | **100.000× (ceiling)** |
| TB2 `top_10pct` | 11,796,480 | 113,663.4 | 227,671 (all) | 79,102.0 | 34,561.3 | 10.000× (ceiling) |

**Read the third column, not the last.** Even inside the top 1% of cost — the most bit-enriched
1% of the object — **27.8% of the bytes (30,491.5 B) still have `dD` exactly zero.** The expensive
set and the breakable set overlap almost perfectly by *membership*, and yet the expensive set is
still more than a quarter zero-distortion by *mass*. That is the whole tension of this object in
one row.

Expressing the same column as an enrichment of zero-mode BYTE share against the global 30.476%:
`top_10pct` **0.998×** (neutral), `top_1pct` **0.913×** (barely depleted), and WJ1's manufactured
support **0.300× — depleted 3.34-fold.** So the positions where the *renderer* breaks a label are
also positions where the *token model* is unsure. Two independent failure modes co-locate. That is
a real refinement of WJ1's membership finding, and it cuts against the repair route: the render's
manufactured errors sit precisely where transmitted bytes are already being spent hardest.

### 3.3 Per class (DALI GT, canonical comma10k order)

| class | positions | bytes | flip rate | positive bytes | **zero-mode bytes** | flip enrichment |
|---|---:|---:|---:|---:|---:|---:|
| Road | 27,407,372 | 45,138.2 | 0.3282% | 27,364.8 | **17,773.4** | 1.701× |
| Lane | 690,754 | 38,183.0 | **10.9240%** | 30,102.2 | 8,080.8 | **56.601×** |
| Undrivable | 58,413,067 | 12,934.0 | 0.0486% | 8,237.2 | 4,696.8 | 0.252× |
| Movable | 1,460,386 | 11,620.1 | 1.5684% | 8,632.3 | 2,987.8 | 8.126× |
| MyCar | 29,993,221 | 5,900.9 | 0.0366% | 4,765.5 | 1,135.4 | 0.189× |

BL1's "Lane is 0.59% of area and 33.56% of model bits" reproduces exactly (38,183.0 / 113,776.16 =
33.5598%). **DF1 refines it: 78.8% of Lane's bytes are positive-mode** — Lane is expensive because
the model is genuinely wrong there 10.92% of the time, not because it is verbose. **The zero-mode
mass is a Road story, not a Lane story:** Road carries 17,773.4 B of the 34,674.11 B zero mode
(51.3%), Lane only 8,080.8 B (23.3%).

---

## 4. The address bound — tba1's empirical law, derived on this object

The zero mode is **not a function of the coding row.** It depends on the symbol the receiver has
not decoded yet. So isolating it needs a side channel carrying the indicator "is my argmax right?".

Under the shipped model that indicator has probability `pmax`, so its entropy is `H_b(pmax)` and

```
address floor  =  Σᵢ H_b(pmax_i)  =  872,907.97 bits  =  109,113.50 B
zero-mode yield =                                        34,674.11 B
ratio           =                                            3.1468×
```

**Why this is not a coincidence: the zero mode IS half of the address, already paid.** Arithmetic
coding of the 5-way row factors exactly, with no loss:

```
code the indicator "is it the argmax?"      costs  −log₂(pmax)      if yes
                                                   −log₂(1−pmax)    if no
then, only if no, code which of the other 4 costs  −log₂(p_sel/(1−pmax))
                                     total  =      −log₂(p_sel)     either way
```

**The zero mode's 34,674.11 B is precisely the realized "yes" branch of that indicator sub-code.**
The address the campaign would need is the *whole* indicator — both branches, at every position —
whose expected cost under the model is `Σᵢ H_b(pmax_i)`. So the question "can we name the free
set?" is really "can we pay for the whole indicator out of the half of it we already spend?", and
the answer is arithmetic. The measured **3.1468×** says naming the flips costs about 2.15× again
what naming the frees costs.

The inequality behind it: for `p ∈ (0,1)`, `(1−p) > 0` and `log₂(1−p) < 0`, so
`−(1−p)log₂(1−p) > 0` and therefore `H_b(p) > −p·log₂ p` pointwise. Under the model a position is
free with probability `pmax`, and *when* free its cost is exactly `−log₂(pmax)`. Hence

```
E[zero-mode bits]  =  Σᵢ pmax_i·(−log₂ pmax_i)  <  Σᵢ H_b(pmax_i)  =  address floor
```

**In expectation under its own model, the address always costs more than the free set yields**, and
the gap is exactly `Σᵢ (1−pmax_i)·(−log₂(1−pmax_i))` — the cost of naming the *flips*. The measured
3.1468× is that inequality realized. `ddm_tba1`'s "naming any subset costs more than it holds" is
therefore not a regularity on this object; it is a property of it.

**The honest boundary of the theorem.** `Σ H_b(pmax)` is a floor only under *this* model's
marginals. A better model of the indicator could beat it — but a better model of "where am I
wrong" **is** a better token model. The loophole collapses into "train a better model", which is
`ddm_ds1`'s lever, not an addressing trick. The bound is also **conservative**: the 67,955,679
float32-saturated rows contribute 0 to it because the model asserts certainty, and a
better-calibrated model would charge more there.

---

## 5. The addressless frontier

`pmax` bands, split by mode — this is the table that decides reachability:

| `pmax` band | positive positions | positive bytes | zero-mode positions | zero-mode bytes |
|---|---:|---:|---:|---:|
| [0.3, 0.5) | 1,472 | 298.3 | 1,277 | 174.0 |
| [0.5, 0.6) | 54,299 | 8,112.7 | 65,453 | 7,062.3 |
| [0.6, 0.7) | 42,241 | 8,267.0 | 79,786 | 6,166.6 |
| [0.7, 0.8) | 38,285 | 9,969.3 | 117,192 | 5,983.2 |
| [0.8, 0.9) | 37,521 | 13,363.3 | 218,628 | 6,132.8 |
| [0.9, 0.95) | 20,134 | 9,736.3 | 255,973 | 3,420.5 |
| [0.95, 0.99) | 20,282 | 13,653.6 | 816,252 | 3,525.0 |
| [0.99, 0.999) | 9,735 | 10,043.3 | 2,864,302 | 1,680.4 |
| [0.999, 0.9999) | 2,929 | 4,190.2 | 7,022,834 | 414.0 |
| [0.9999, 1] | 773 | 1,468.0 | 106,295,431 | 115.0 |

*(Bands below `pmax = 0.3` hold 1 position and 0.23 B in total and are omitted.)*

**The zero-mode bytes live where the flips live.** 34,559.1 of the 34,674.1 zero-mode bytes sit
below `pmax = 0.9999`, interleaved with 226,898 of the 227,671 flips. At the top band, 106.3 million confidently-correct
positions carry **115.0 B between them** while 773 confidently-*wrong* positions carry **1,468.0 B**
— so **92.7% of the byte saving at `τ = 0.9999` comes from breaking 773 labels**, not from
harvesting free confirmation. At every threshold, the saving is dominated by the flips. That is the
mechanism behind the address bound, seen from the operator side.

---

## 6. The scorer, and the real coder

### 6.1 The two controls both pass

**Forward model.** Re-rendering the shipped tokens through the shipped `SemanticTokenRenderer` and
scoring them with the frozen CPU SegNet against DALI-lineage GT gives
`d_seg = 0.00020134819878472223` (**23,752** disagreeing cells). The body's own contest-CUDA
receipt gives **0.00020139** (23,757 cells). That is **0.99979×** — five cells apart in 117,964,800.
Every `Δd_seg` below is a difference taken on this one validated instrument inside one process, so
numerator and denominator share a lineage. `ddm_rf1` §4 caught a published ratio that divided a
macOS-CPU numerator by a contest-CUDA denominator; this is the structural cure for that class.

*(The control was re-measured in a second, independently launched process after the module was
refactored mid-run, and returned `0.00020134819878472223` / 23,752 cells — identical to the last
digit. That is the proof that the refactor was behaviour-preserving and that the committed code is
the code that produced every row below.)*

**Encoder.** Re-encoding the unedited field through the shipped RC64 encoder along the receiver's
own decode trajectory (`--tau 2.0`, nothing skipped) emitted **113,784 B**, decomposing exactly as

```
4 B TOKEN_MAGIC  +  113,777 B byte-identical to the shipped token stream  +  3 B flush padding
```

with the reconstructed field differing from the shipped field in **0** positions. The coded content
is reproduced byte for byte; the +7 is framing the archive does not store. Drop savings are
therefore taken against the **control's emitted length**, where the framing cancels — never against
the archive's stored length, which would charge the drop 7 bytes it did not cause.

### 6.2 The closed-loop drop at τ = 0.999 — static accounting INVERTS the verdict

The `reencode` stage runs the operator for real: it recomputes the coding row from the live
trajectory, takes THAT row's argmax, encodes only the rows the rule still sends, and feeds the
**reconstructed** symbol back into the corrector and the model context. Encoder and receiver stay
in lockstep, so the emitted stream is a real archive section rather than an accounting estimate.

| quantity at τ = 0.999 | static `−log₂p` accounting | **REAL closed loop** | factor |
|---|---:|---:|---:|
| bytes saved | 6,187.3 | **12,224** *(± ≤3 B)* | **1.976×** |
| token labels broken | 3,702 | **106,711** | **28.83×** |
| rows still sent | 4,642,833 | **3,059,209** | 0.659× |
| **damage ÷ credit** (at 1× render amplification) | **0.762× — a WIN** | **11.114× — a refusal** | **14.59×** |

*(The byte figure is the difference of two emitted payloads, so the 4-byte magic cancels exactly;
only the range coder's terminal flush — whole bytes, `< 1` byte of real content — can differ
between the two runs, hence `± ≤3 B`. Immaterial against an 11× ratio.)*

**Static accounting on this object does not merely misprice the drop operator; it inverts the
verdict.** `ddm_fs2` measured `−log₂p` mispricing token *value* moves by up to 11×; this is the
same class on the *drop* operator, and it lands at 14.59× on the ratio that decides.

**The mechanism, and it is a degenerate feedback loop.** Skipping a token feeds the model its own
prediction. The field it then conditions on is *more* consistent with its own model than the true
field was, so its confidence RISES — the true trajectory sent only **0.659×** as many rows as the
static frontier expected, which is exactly why it saved **1.976×** the bytes. But that confidence
is not accuracy: substitutions rose **28.83×**. **The drop operator decouples confidence from
correctness in the direction that flatters the rate term and punishes the distortion term.** Both
effects are large, they point opposite ways, and the damage wins by 14.59×.

### 6.3 The verdict, end to end — and a render-amplification law nobody had

Scoring the τ = 0.999 closed loop's **own reconstructed field** (sha `aebfb6a9…`, the bytes the
receiver would actually hand SegNet) closes the measurement with no modelled term anywhere:

| | value |
|---|---:|
| control `d_seg` | 0.00020134819878472223 (23,752 cells) |
| **`d_seg` after `drop(τ = 0.999)`** | **0.001142823961046007** (134,813 cells) |
| damage `ΔS_seg` | **0.094147576** |
| credit (12,224 REAL bytes) | 0.008139460 |
| **damage ÷ credit, fully measured** | **11.567×** |
| S of the resulting candidate | **0.23423** vs 0.14822 shipped |

**The addressless drop route is refused by an order of magnitude**, with every term measured: real
encoder bytes, real receiver field, real render, real frozen SegNet, real DALI-lineage GT.

**The render-amplification law: it SATURATES.** Net new scored errors per broken token label,
measured at three perturbation scales on the same instrument:

| labels broken | gross cells changed / label | **net new errors / label** |
|---:|---:|---:|
| 773 (τ = 0.9999, static) | 2.2730 | **1.4036** |
| 3,702 (τ = 0.999, static) | 1.8258 | **1.1807** |
| 106,711 (τ = 0.999, closed loop) | 1.1231 | **1.0408** |

**Render amplification is not a constant and the campaign should stop treating it as one.** A
handful of broken labels costs ~1.40 scored cells each; a hundred thousand costs ~1.04 each. The
mechanism is saturation — once a neighbourhood is already wrong, breaking another token there
cannot break it twice. The practical consequence cuts against targeting: **small, precisely-aimed
token edits are amplified ~35% harder than bulk ones**, so the per-unit damage of a surgical edit
is worse than a bulk average would predict.

**The trap this section closes.** With MEASURED amplification and STATIC bytes, τ = 0.9999 prices
at **0.873×** and τ = 0.999 at **0.899×** — both nominal adoptions. A campaign that priced this
operator on a static field edit, even with a correctly measured render amplification, would have
adopted a change that makes S **58% worse**. Only the closed loop separates them.

---

## 7. The decision number: the cheap-to-remove subset, in both currencies

The charter asks for "the byte-size of any low-`dD/dB` byte-heavy subset". On the token half it is
exact, because the low-`dD/dB` subset is the `dD = 0` subset:

| | value |
|---|---:|
| **zero-mode bytes** | **34,674.11 B** |
| in rate currency | **0.023088070 S** |
| as a share of the 42,381.16 B demand | **81.815%** |
| in distortion currency (÷ 1.273108 B/cell) | **27,235.8 scored argmax cells** |
| scored cells actually wrong on this body | 23,756.9 (`0.00020139 × 117,964,800`) |
| **zero mode ÷ the entire seg debt** | **1.1464×** |
| renderer annihilation ceiling, for comparison | 30,856 B = 72.806% of demand |

Two comparisons make the size legible:

1. **The confirmation entropy is worth 14.6% MORE than perfect segmentation.** Annihilating
   `d_seg` outright — every one of the 23,757 wrong cells repaired — is worth 30,245.1 B. The zero
   mode is 34,674.11 B.
2. **It is a bigger prize than the whole renderer.** The renderer's total annihilation ceiling is
   30,856 B (72.81% of demand); the token stream's zero mode is 34,674.11 B (81.81%).

**What this number is, and is not.** It is the exact byte mass sitting at `dD/dB = 0` — a
*ceiling* on what the confirmation half could yield if the model became certain wherever it is
already right. It is **not** a demonstrated saving. Two things stand between it and the archive,
and only one of them is closed by this arm:

- **Addressing it is closed** (§4): the address floor is 3.1468× the yield, and no addressless
  threshold reaches more than 0.0486 B of it.
- **Modelling it away is open, and priced elsewhere.** The HPAC model is itself 37.7% of the
  archive, so a model good enough to shrink its own confirmation entropy costs bytes to ship.
  That trade is `ddm_cl1`'s capacity curve and **this arm did not re-measure it.** The realizable
  fraction of 34,674.11 B is bounded by it, and is NOT claimed here.

---

## 8. What this implies for `ddm_ds1`

`ddm_ds1` landed today with the right diagnosis — *"the campaign has a thermometer and no
thermostat"* — and with §2.1 already correct that the HPAC token model's weights have
`dD/dB ≡ 0` because the token codec is lossless, concluding they "cannot host the objective."

**DF1 agrees with the fact and inverts its consequence.** `dD/dB ≡ 0` does not disqualify the
token model; it makes it **the only pure-rate lever in the archive**, and DF1 measures what that
lever can win and by which of two distinct mechanisms:

| half of the stream | bytes | removed by | needs new information? |
|---|---:|---|---|
| **confirmation entropy** (zero mode) | **34,674.11** | `pmax → 1` where the model is *already right* — calibration, capacity, conditioning | **no** |
| **correction entropy** (positive mode) | **79,102.05** | predicting correctly where it is currently *wrong* — moving positions between modes | **yes** |

Three consequences for the objective `ds1` is designing:

1. **A `dD/dB`-shaped training term is the wrong shape for the token half.** `dD/dB` is already
   exactly 0 on 30.48% of its bytes, and the term that shrinks it is plain cross-entropy on the
   HPAC model — which that model already minimizes. There is no thermostat to add here; there is a
   better model to train.
2. **The `dD/dB` objective belongs on the renderer**, where `dD/dB` is genuinely > 0 — and DF1
   sizes that arena honestly: 30,856 B, **smaller than the token stream's confirmation entropy
   alone.** `ds1` should know its arena is the smaller of the two before it builds.
3. **Where to aim inside the correction half, if it is aimed at all.** Lane carries 30,102.2 B of
   positive-mode bytes at a **10.92%** flip rate and **56.6×** flip enrichment — it is the one class
   where the model is wrong often enough that better prediction, not better confidence, is the
   binding term. Road is the opposite: 17,773.4 B of zero-mode bytes, 51.3% of the whole
   confirmation prize, at a 0.33% flip rate.

**A harness warning `ds1` should take before it builds.** Any objective that changes what the
token field carries must be evaluated **through the closed-loop trajectory**, never through a
static field edit scored afterwards. §6.2 measured that difference at **14.59× on the ratio that
decides, in the direction that manufactures a false win** — static said 0.762× (adopt), the real
receiver said 11.114× (refuse). An adaptive context model conditions on what it *decoded*, so any
harness that edits the field and re-scores it is measuring a receiver that does not exist. This is
the same genus as `ddm_fs2`'s "price token field levers by REAL re-encode", now measured on the
drop operator and one order of magnitude larger.

---

## 9. The prediction, adjudicated

**Registered:** *"`dD/dB` is also heavy-tailed (Gini > 0.8) AND co-located with bit mass — the
expensive-to-remove sites are the same sites that hold the bits, which would make the object
maximally hard to shrink and would explain every measured refusal in one stroke."*
**Falsifier:** *"Gini < 0.5, OR the high-`dD/dB` set is measurably ANTI-correlated with bit mass."*

| leg | measured | verdict |
|---|---|---|
| Gini > 0.8 | Gini of `dB` = **0.9951593787014741** | **CONFIRMED** |
| co-located with bit mass | AUC = **0.999998753**; **all** 227,671 positive-mode positions inside the top 1% of cost (100.000×, the ceiling); mean-bits ratio **1,179.74×** | **CONFIRMED, at the ceiling** |
| falsifier (Gini < 0.5) | no | did not fire |
| falsifier (anti-correlation) | no — correlation is maximal and positive | did not fire |
| **stated implication** — "maximally hard to shrink" | **REFUTED**: 30.476% of the stream's bytes (34,674.11 B, 81.815% of the demand) sit at `dD` **exactly zero** | **REFUTED** |

The prediction was right about the shape and wrong about what the shape means. Co-location is
near-perfect **by position membership**, and yet the zero mode still holds a third of the bytes,
because the two modes differ in bits-per-position by 1,179.74× — 227,671 positions can own 69.52%
of the mass while 117.7 million positions own the remaining 30.48%. Membership and mass are
different questions and this object answers them differently.

The correct one-stroke explanation of the measured refusals is not "the error is where the bits
are" but the address bound of §4: **the cheap bytes cannot be named for less than they are worth.**

### Corrections owed upward

- The charter prices W72 at **46.3×**. `ddm_rf1` §4 (same day) already corrected that to
  **35.5364×** matched-lineage; 46.3162× divided a macOS-CPU/PyAV numerator by the contest-CUDA
  pointer. The charter carries the superseded figure.
- The charter lists **"nr1 349×"** among `dD/dB` ratios. At source (`ddm_rj1:151`, `ddm_sy2:302`)
  349× is a **proxy-understatement factor** (agreement vs evaluator), not a damage÷credit ratio.
- `dg2` **687.3× / 791.7×** (`ddm_dg2:130,150`) and `rf1` **478.7×** (`ddm_rf1:107,124`) are
  VERIFIED at source as damage÷credit and are quoted correctly.
- This body's seg leg is **0.00020139**, not the 0.00030309 in general circulation — that belongs
  to `ddm_up3`'s 176,420 B body. Same class as the tail-share correction in §1.

---

## 10. verdict_scope, and what is NOT CLAIMED

**verdict_scope:** `INSTANCE:DX2_archive_976f706d_n600_shipped_HPAC_RC64_law_TOKEN_HALF`. The
two-mode partition and every byte mass in it are properties of *this* body's coding rows under
*this* model. A different token model gives a different partition. The address-bound *inequality*
of §4 is general to any model whose coding rows the receiver computes; its *numbers* are this
instance's.

**NOT CLAIMED:**

- **No score claim, no promotion, no submission.** The pointer is unmoved. No archive was built.
- **The 34,674.11 B is a ceiling, not a saving.** Nothing here demonstrates removing any of it.
  The model-size trade that bounds the realizable fraction (`ddm_cl1`'s capacity curve) was **not
  re-measured by this arm**.
- **No claim about the renderer half's `dD/dB` distribution.** Declared SCOPE reduction (§1); its
  only existing field is a first-order gradient proxy with a measured 349× realized gap.
- **No `d_pose` measured under any label-changing drop.** Tokens are a pose actuator too
  (`ddm_jg1` §0), so every flip-inclusive row's damage is a **LOWER BOUND** — pose can only add.
  The zero-mode result is exempt: it leaves the token field bit-identical, so both scorer inputs
  are unchanged and it is pose-exact by construction. Had any τ row come out a net win on
  seg+rate, `d_pose` would have had to be measured before claiming it.
- **The scorer rows are `[macOS-CPU advisory]` with DALI-lineage GT.** The absolute `d_seg` is
  advisory; the deltas are matched-instrument differences taken inside one process.
- **Gini, AUC and every enrichment are this body's.** They transfer to no other object.
- **AUC is computed on a 142-bin log-spaced histogram**, so within-bin orderings are counted as
  ties. It is a binned estimator, not an exact rank statistic.

---

## 11. Retained payload (ALWAYS KEEP THE PAYLOAD, P0)

`/Volumes/APDataStore/pact/ddm_df1_dddb_field/measurement_v1/`

| file | bytes | sha256 |
|---|---:|---|
| `retained/fields/position_coding_argmax.u8.bin` | 117,964,800 | `db498280c22c3aa1b787310e25435116911933216cae558f309f8b10baf7994e` |
| `retained/fields/position_coding_pmax.f32le.bin` | 471,859,200 | `f37e3d8a21d02647437bf950d7a8a75b751c2a9644c7b8ad48aca2833be4794b` |
| `retained/fields/position_coding_psecond.f32le.bin` | 471,859,200 | `72c544044ce735c6f1a3876ef138fba45684f62da1ef3e20bd38aaa316385ae5` |
| `retained/fields/position_rc64_frequency_cost_bits.f64le.bin` | 943,718,400 | `99d7833d55a9aa128f67cfc125a10ba90cffaa714de0b88801aa43b8b16e2c86` |

Plus `analysis/FIELD.json`, `analysis/flip_flags.npy` (14,745,728 B, the packed zero/positive
partition), `analysis/REENCODE_*.json`, `analysis/SCORER*.json`, the 30 per-stage receipts under
`retained/stages/`, and every emitted RC64 stream and reconstructed token field under
`analysis/reencode_work/`. The cost field is byte-identical to TB2's and is therefore cited rather
than duplicated as a second copy; the three prediction fields are new payload and are retained in
full.

**Why the field is retained as `(partition, dB)` and not as a per-position `dD/dB` ratio.** `dD` is
**exact** on the zero mode (it is 0, by the identity in §0) and **only aggregate** on the positive
mode — resolving it per position would need one render + SegNet pass per flipped token, i.e.
227,671 of them. A per-position ratio column would therefore have to spread a measured aggregate
across positions by some assumption, and that column would be an assumption wearing a
measurement's name. What is retained is every exact input — the mode flag and the exact `dB` at
every one of the 117,964,800 positions — so any successor can impose its own `dD` model on the
same field without inheriting mine.

---

## 12. STORES CONSULTED

`.omx/research/*.md` (full grep) · `.omx/state/canonical_frontier_pointer.json` ·
`experiments/results/modal_auth_eval_mirror/contest_auth_eval_ddm_dx2_fx5_cabac_t4_r2.json` ·
`docs/operating_manual_craft_handoff.md` · `CLAUDE.md` · `AGENTS.md` ·
`experiments/{ddm_bl1_per_position_bit_allocation, ddm_tb2_token_bit_attribution,
ddm_wj1_cost_error_position_join, ddm_jg1_seg_solve, ddm_jg2_tail_reencode,
ddm_up2_shipping_pose_solve, ddm_wd3_scorer_aware_width_distillation,
ddm_rc64p_native_cpu_decode/route_b_rc64}.py ·
retained stores `/Volumes/APDataStore/pact/ddm_tb2_token_bit_attribution/measurement_v1/` and
`/Volumes/VertigoDataTier/pact/{ddm_to2_token_ordering_race, ddm_qs3_20260813}/` ·
sister arm `.omx/research/ddm_ds1_cheap_to_shrink_objective_20260824.md`.
