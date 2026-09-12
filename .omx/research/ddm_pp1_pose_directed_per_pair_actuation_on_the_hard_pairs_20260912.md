# ddm_pp1 — POSE-DIRECTED per-pair actuation on the hard pose pairs: the falsifier FIRED, the family closes, and the reason is a FLOOR

Tokens: `[no-triality] [p0-ledger-ok]`

Lane `ddm_pp1_pose_directed_per_pair_actuation_20260912`. Base: move 49, S 0.13632299781031237 @ 179,153 B,
archive sha `73e41a6620bd4ea3aaf236eff9de46391857907527358e8eb40ded0925a1c214`. Axis
`[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage, n600]` for the distortion legs;
bytes EXACT through the shipped SM1S mixer and the shipped Brotli container. **No score is claimed. The
pointer did NOT move and this arm did not build a candidate.** No Modal, no fire, no packet.

---

## 1. The verdict, in one table

The charter's pre-registered falsifier: *"if the resolved per-pair d_pose on the treated pairs does not
fall by ≥ 5 % at seg-neutral admission, the actuator has no supply on the hard pairs and the family
closes there."*

**MEASURED best fall at seg-neutral admission: 0.452 %. The falsifier FIRED by 11.1×.**

| leg | measured | against |
|---|---:|---|
| best seg-neutral resolved pose fall (Actuator A) | **−0.45 %** (pair 88, d1 −1→0) | 5 % falsifier |
| best seg-neutral resolved pose fall (Actuator B) | **−0.25 %** (pair 88, cell (192,258) 0→1) | 5 % falsifier |
| best ΔS on ANY row, seg + pose, before a byte | **−2.590e−06** (pair 66, d1 −1→0, +4 cells) | −2e−05 bar (**7.7× short**) |
| best ΔS at seg NEUTRALITY, before a byte | **−1.534e−06** (pair 73, d2 0→+1, −1 cell) | −2e−05 bar (**13.0× short**) |
| rate fee for ONE changed code, MEASURED on the live container | **+44.00 B mean** = +2.930e−05 S | it alone exceeds the bar |
| whole family composed at seg neutrality (2 of 12 pairs have a move) | **−1.875e−06 S** pose | **26.3× short** of what the +44 B fee plus the bar demand |
| rows admitting once the rate fee is paid | **0 of 24** | — |

Every ΔS above is recomputed from components; the identity control is that the same arithmetic
reproduces the pointer exactly: `100·0.00010287 + √(10·4.55e−06) + 25·179,153/37,545,489 =
0.13632299781031237`.

**The actuator is not inert — it is out-priced.** It moves the resolved pose by up to −8.2 % on one pair
(66, d1 −1→0, 5.8796e−05 → 5.3961e−05), but that move costs 4 seg cells, and the only moves that cost no
cells move the pose by less than half a per cent. The binding fact is §5: **99.4 % of the lattice costs
seg**, and §6: **the resolved pose is a floor the frame_1 render cannot lower**.

---

## 2. The base — reproduced, not inherited (F1)

MEASURED on move 49's own shipped state: its carrier codes read out of its own `archive.zip`, its own
cold parse-back `0.raw` as the decode, its own admitted token field, DALI GT.

| quantity | measured | reference | agreement |
|---|---:|---:|---:|
| n600 mean d_pose | **4.543568679770593e−06** | pass 7's admitted-subset resolved pose **4.543569e−06** | to 7 s.f. |
| " | " | the T4 print **4.55e−06** | ratio **0.99859** |
| median | 3.34503739671486e−07 | — | — |
| max (pair 88) | 2.1283759625946122e−04 | — | **636× the median** |
| top-12 share of the pose mass | **52.49 %** | charter's 52.7 % | — |
| top-12 pair set | 88, 87, 73, 316, 89, 70, 63, 448, 64, 91, 66, 67 | charter's list | **12 / 12 identical** |

The ranking permutes at ranks 4–7 against the charter's, and that is not noise: the charter read move
**48**'s per-pair receipt, and move 49's own admission re-solved the carriers of pairs 70 and 63, which
moved them down. The SET is unchanged, which is the claim the arm needed.

Pose marginal at this operating point, DERIVED: `d/d(d_pose) √(10·d_pose) = 5/√(10·4.5436e−06) = 741.77`
S per unit mean d_pose, i.e. **1.2363 S per unit of ONE pair's d_pose**.

Receipts: `base/POSE_BASE_MOVE49.json`, `base/pose_base_move49.npy`
(sha `ac2b…` recorded in the report).

---

## 3. Controls — four, all PASS, before any claim

| control | what it refuses | result |
|---|---|---|
| **C1** semantic render identity | measuring a look-alike renderer | my batch-1 render of frame 2p+1 reproduces the shipped decode **byte for byte** on pairs 0, 63, 88, 316, 599 — max\|dev\| **0**, **0** mismatched pixels of 3×874×1164 each |
| **C2** carrier render identity | measuring a look-alike frame_0 | render from the shipped carrier codes reproduces the shipped decode's even frames exactly, same 5 pairs, max\|dev\| **0** |
| **C3** SegNet argmax identity | a seg leg that is not the shipped one | my argmax equals the shipped decode's argmax **cell for cell** (0 disagreeing) and the flip counts match: 33/33, 12/12, 27/27, 24/24, 23/23 |
| **C4** unmoved-resolve-is-zero | crediting the move with carrier slack | re-solving pairs **88, 87, 316** on their UNMOVED renders moves d_pose by **exactly 0.000e+00**, 0 coordinates changed |

**C4 is the control that shapes this arm's whole result.** The three hardest pose pairs are already at a
converged carrier optimum, so their 2.13e−04 / 2.12e−04 / 1.29e−04 residual is **not** carrier slack —
it is structural — and any post-move change is the move's own. It also independently re-confirms
pc2/pc3's closure of the carrier-alone door on exactly the pairs where a carrier gain would be worth most.

Receipt: `base/CONTROLS.json`, `all_pass: true`.

---

## 4. The container is not fe1's, and the fee was re-measured on the live one

fe1's loader refused this body, correctly: it reads the semantic section as `brotli → CK2 → RC1 rider`,
and move 49 ships **`SM1S`** — sm1's counted 24-weight arithmetic mixer, the successor coder fe1's own
handoff named. Inheriting fe1's rate law would have been the wrong-CODER class sj1 pass 7 §3 already paid
for once. MEASURED at source on move 49's bytes:

| layer | value |
|---|---|
| semantic member | **29,862 B**, sha `786950a5…` — **byte-identical to move 48's**; pass 7 moved the token tail and the carrier, never the renderer |
| container shape | **Brotli q10 / lgwin 16 + CK2**, identified by BYTE identity (ren2's correction; a length-only tie-break ships different bytes) |
| SM1S rider | 31,451 B, 24 int8 mixer weights carried verbatim |
| SM3R body | 36,130 B, sha matches ren2's pin |
| `frame_embed` run | (600, 8) signed **3-bit** codes, occupancy −3:28 −2:230 −1:892 0:2505 +1:904 +2:216 +3:25 |
| code domain | `rc1.pack_signed_codes` is plain two's complement — **[−4, +3] with NO reserved symbol** (iv1's `−8` reservation belongs to `renderer_weight_codec` on the WANS1 body, a different coder). The shipped table never uses −4, so a move to −4 is the model's first sighting of that symbol. |

**The identity control the whole rate leg rests on:** the shipped codes, pushed back through
`body → sm1.encode → CK2 → brotli(q10, lgwin16)`, reproduce the shipped **member byte for byte**.
Nothing is priced until that passes.

### The fee, MEASURED (scorer-free, exact re-encodes, 8 seeded draws per count)

| N changed codes | mean Δ member | sd | min | max | per code |
|---:|---:|---:|---:|---:|---:|
| 1 | **+44.00 B** | 33.11 | +0 | +90 | +44.00 |
| 2 | +41.38 | 34.97 | +0 | +104 | +20.69 |
| 4 | +31.50 | 30.91 | −4 | +97 | +7.88 |
| 8 | +53.00 | 53.12 | −12 | +141 | +6.62 |
| 16 | +39.62 | 25.57 | −7 | +75 | +2.48 |

Two findings, both reusable:

1. **fe1's structural law survives the coder change.** The fee is a flat **container-break** fee of
   ~+40 B, not a per-code price: the marginal per-code term falls from +44 to +2.48 B between N=1 and
   N=16. The magnitude is lower than fe1's +65–80 B at its shipped shape, so the number moved and the
   SHAPE did not.
2. **The lottery reproduces.** sd **33.11 B** at N=1 against the campaign's standing 34.8 B
   container-break lottery — measured here on a *different* container, which is independent
   confirmation that the spread is the draw and not the coder.

Caveat carried with the number: the **container search was NOT run** in this scan (`--search-container`
off), so the "searched" column is a copy of the shipped-shape column. A real candidate must run the
q/lgwin/ck2 search; fe1 measured that it recovers most of the fee, and that recovery is **owed, not
held**, here. Receipt: `base/RATE_FEE.json`.

---

## 5. Actuator A — the whole single-code lattice, screened on the binding leg first

The admission requires seg NEUTRALITY, and the seg leg needs only a render and an argmax — no carrier
re-solve. Refining a candidate the seg leg will reject spends ~3 minutes to learn nothing, so the whole
lattice was screened at ~4 s per candidate and only the survivors were refined. **An ORDERING change,
not a mechanism change**: every number the admission uses is still realized, and the screen's `d_cells`
is the same measurement the full stage makes.

**672 candidates — all 12 top pairs × all 8 dims × all 7 alternatives. MEASURED:**

| pair | base flips | min Δcells over the 56 moves | seg-neutral moves |
|---:|---:|---:|---:|
| 63 | 12 | **+3** | 0 |
| 64 | 16 | +1 | 0 |
| 66 | 23 | +1 | 0 |
| 67 | 22 | +3 | 0 |
| 70 | 29 | +3 | 0 |
| 73 | 30 | **−1** | **2** |
| 87 | 11 | +2 | 0 |
| 88 | 27 | **0** | **2** |
| 89 | 12 | +3 | 0 |
| 91 | 20 | +1 | 0 |
| 316 | 24 | +2 | 0 |
| 448 | 18 | +1 | 0 |

**4 of 672 (0.60 %) are seg-neutral; 10 of 12 pairs have ZERO.** This is fe1's law
("18,624 of 18,906 single-code moves make d_seg WORSE; the shipped codes sit at a per-pair local
minimum on 95 % of pairs") in a sharper form, on the sub-population that carries the pose term:
**on the hard pose pairs the shipped codes sit at a seg local minimum in 10 cases of 12.**

### The refined survivors (4 seg-neutral + the 2 cheapest per pair = 24 rows, full chain)

| move | Δcells | d_pose base → resolved | ×base | ΔS seg+pose | + mean rate fee |
|---|---:|---|---:|---:|---:|
| 66 d1 −1→0 | +4 | 5.8796e−05 → 5.3961e−05 | **0.9178** | **−2.590e−06** | +2.671e−05 |
| 73 d2 0→+1 | **−1** | 1.6181e−04 → 1.6125e−04 | 0.9966 | **−1.534e−06** | +2.776e−05 |
| 88 d1 −1→0 | **0** | 2.1284e−04 → 2.1188e−04 | 0.9955 | −1.189e−06 | +2.811e−05 |
| 64 d0 +1→0 | +2 | 8.9270e−05 → 8.7026e−05 | 0.9749 | −1.080e−06 | +2.822e−05 |
| 88 d3 +2→−1 | **0** | 2.1284e−04 → 2.1265e−04 | 0.9991 | −2.268e−07 | +2.907e−05 |
| 73 d3 0→−1 | **0** | 1.6181e−04 → 1.6219e−04 | 1.0023 | +4.698e−07 | +2.977e−05 |
| … 18 further rows, all positive ΔS | | | 0.9837 … 1.0630 | +7.94e−07 … +5.91e−06 | |

**0 of 24 admit before the rate leg; 0 of 24 admit after it.** The full table with every row's stale
pose, refine trace and resolved carrier codes is retained at `refine/refine_moves_*.jsonl`.

### The economics, DERIVED at THIS move (binding numbers expire at every pointer move)

A seg-neutral candidate pays only the container fee, so the pose credit must cover the bar plus the fee:

```
required Σ|Δ d_pose| = (2e−05 + 44.00 B × 6.658589531221714e−07) / 1.2363
                     = 3.98756e−05
top-12 pose mass      = 1.4310e−03
required fraction     = 2.79 %          (the charter's falsifier asked for 5 %)
```

So the arm's own economic bar is **milder** than its pre-registered falsifier — and the measurement
misses both. Two readings, both honest:

* **per-pair**: the seg-neutral supply is **0.45 %** of a pair's residual, **6.2× short** of the 2.79 %
  the economics need and 11.1× short of the falsifier;
* **whole-family**: only **2 of 12** pairs have a seg-neutral move at all, and the best one on each
  (73: 5.548e−07, 88: 9.618e−07) totals **1.517e−06** of d_pose — **3.80 % of the requirement, i.e.
  26.3× short**, worth **−1.875e−06 S** in pose against a **+2.930e−05 S** container fee. Across
  DIFFERENT pairs the pose credits DO add (d_pose is per-pair and `frame_embed` is pair-indexed, which
  C4's null control proves cannot leak), so this is the family's whole realizable supply, not a sample.

---

## 6. WHY — the resolved per-pair pose is a FLOOR the frame_1 render cannot lower

Actuator A's single-code sweep is a small neighbourhood. The floor probe asks the structural question
directly: perturb the pair's eight FiLM codes at increasing size — up to **all eight at once**, which
re-renders the whole frame — and measure where the carrier re-solve lands each time.

**24 perturbations over 4 pairs (sizes 2, 4, 8). MEASURED:**

| observation | value |
|---|---|
| stale rise before the re-solve | **1.0× to 740.6×** |
| resolved d_pose after the re-solve | **0.928× to 1.697× base** |
| perturbations landing BELOW base | 9 of 24 |
| best fall found anywhere, at any seg cost | **−7.2 %** (and it cost 17–53 cells) |
| pair 88, 6 perturbations, stale 1.0×–170× | resolved **0.977× – 1.027×** base |

Pair 88 is the cleanest reading: however violently frame_1 moves, the resolved value returns to within
±3 % of the same number.

**The law (MEASURED here, n=23 over 4 pairs; DERIVED mechanism):** the carrier's twelve coefficients
re-aim after any frame_1 change — fe1's §0 law, confirmed again at recoveries up to 740× — and what
they cannot cancel is the component of the pose error outside their twelve-dimensional reach. That
component is a property of the PAIR, not of frame_1's content. So:

* **C4** says the residual is not carrier slack (the unmoved re-solve moves it by exactly 0);
* **this probe** says the residual is not frame_1-reachable (any frame_1 render lands on the same floor);
* therefore **the hard pairs' pose residual lies outside BOTH actuators' reach**, and the only doors left
  are ones that change the reach itself, not the render.

This is the honest generalisation of fe1's law rather than a contradiction of it: *a per-pair render
change is payable on pose* (fe1) **because** the carrier absorbs it — and the same absorption is exactly
what stops that change from BUYING pose.

---

## 7. Actuator B — token edits, the same wall one order of magnitude smaller

A token move changes one cell of the pair's 384×512 plane, so the render change is strictly smaller and
more local than a frame_embed move. Cells were nominated by the pose saliency
`|∂ d_pose/∂ frame_1|` area-pooled onto the token grid — a PROPOSAL only; every nomination was realized
through the shipped render, the frozen argmax and the carrier re-solve.

**16 nominations over 2 pairs (5 saliency cells each, ±1 on the symbol). MEASURED:**

| pair · cell · move | Δcells | d_pose base → resolved | ×base | ΔS seg | ΔS pose |
|---|---:|---|---:|---:|---:|
| 88 (192,258) 0→1 | **0** | 2.1284e−04 → 2.1232e−04 | **0.9975** | +0.000e+00 | −6.451e−07 |
| 88 (191,257) 0→1 | **0** | → 2.1238e−04 | 0.9979 | +0.000e+00 | −5.634e−07 |
| 88 (192,257) 0→1 | **0** | → 2.1245e−04 | 0.9982 | +0.000e+00 | −4.806e−07 |
| 88 (168,262) 2→1 | +3 | → 2.1075e−04 | 0.9902 | +2.543e−06 | −2.578e−06 |
| 88 (167,263) 2→3 | +4 | → 2.1256e−04 | 0.9987 | +3.391e−06 | −3.419e−07 |
| 73 (128,281) 2→3 | +3 | 1.6181e−04 → 1.5933e−04 | **0.9847** | +2.543e−06 | −3.065e−06 |
| 73 (128,281) 2→1 | +5 | → 1.5987e−04 | 0.9881 | +4.239e−06 | −2.390e−06 |
| 73 (129,273) 2→1 | +4 | → 1.8784e−04 | **1.1609** | +3.391e−06 | +3.211e−05 |
| … 8 further rows | +2 … +4 | | 1.0007 … 1.1115 | | |

Three things separate B from A and none rescues it:

1. **Token moves ARE seg-neutral where code moves are not** — **3 of 16** cost zero cells (all on pair
   88), against **4 of 672** for A. That is the expected consequence of locality, and it is a real
   difference worth recording for any later arm.
2. **The pose supply is smaller still**: best seg-neutral fall **−0.25 %**, against A's −0.45 %. Exactly
   what §6 predicts — a whole-frame change cannot lower the floor, so a one-cell change certainly cannot.
3. **The saliency proposal is a weak ranker on pair 73**: its top cells make pose WORSE, up to 1.16×.
   A saliency computed on frame_1 pixels ranks where the pose gradient is large, not where a token symbol
   change moves the render in the descent direction, and the measurement says so plainly. Proposals
   propose; measurements decide.

B's token edits are also NOT free: a token move is priced by the RLC1 token stream, not by the semantic
member, and sj1 pass 7 measured that family at 5.01 bits/token. A −0.25 % pose fall on pair 88 is
`1.2363 × 5.3e−07 = −6.6e−07` S against a bar of −2e−05 — **30× short before a single bit is paid**, so
the token price was not measured for these moves and is not owed: no candidate reaches the pricer.

---

## 8. A + B composed, realized as ONE object

Reporting A+B as A's delta plus B's delta would be the additivity this campaign refuses. It was measured
as one render on pair 88 — A = `frame_embed` d1 −1→0, B = token cell (192,258) 0→1, both individually
seg-neutral:

| variant | Δcells | d_pose base → resolved | ×base | ΔS seg+pose |
|---|---:|---|---:|---:|
| **A** alone | 0 | 2.1284e−04 → 2.1188e−04 | 0.9955 | **−1.189e−06** |
| **B** alone | 0 | 2.1284e−04 → 2.1232e−04 | 0.9975 | **−6.451e−07** |
| **A+B** realized together | 0 | 2.1284e−04 → 2.1226e−04 | 0.9973 | **−7.156e−07** |

| | value |
|---|---:|
| sum of the parts (A + B) | −1.834e−06 |
| **measured A+B** | **−7.156e−07** |
| **measured / sum** | **0.390** |

**A+B delivers 39 % of the sum, and is WORSE than A alone.** The two actuators do not stack because they
are pushing against the same object: the floor of §6. This is the campaign's sub-additivity law reaching
a new surface — not two disjoint edits partially overlapping, but two DIFFERENT actuators competing for
one residual. It also means no composition over more pairs rescues the arithmetic: the seg-neutral
supply is 0.45 % per pair and it does not add.

---

## 9. Falsifier dispositions

| id | statement | disposition |
|---|---|---|
| **charter falsifier** | resolved per-pair d_pose must fall ≥ 5 % at seg-neutral admission | **FIRED.** Best measured fall **0.4519 %** (A, 4 seg-neutral rows) and **0.2451 %** (B, 3 seg-neutral rows). Shortfall **11.06×**. |
| **F1 base + ranking** | the base and the top-12 ranking reproduce on this instrument | **did not fire.** d_pose 4.543568679770593e−06 vs pass 7's 4.543569e−06 and the T4 print at ratio 0.99859; top-12 set 12/12. |
| **C4 unmoved control** | re-solving an untouched pair moves d_pose by exactly 0 | **did not fire** (0.000e+00 on 3 pairs), and it became the arm's load-bearing structural fact. |
| **twins / container control** | the shipped codes repack to the shipped member byte-identically | **did not fire** — member 29,862 B sha `786950a5…`, exact. |
| **base-gate tolerance** | the search's per-pair base must agree with the n600 vector | recalibrated after it fired on float summation order (§10); re-run clean at the MEASURED band. |

---

## 10. The instrument correction this arm owes its own record

The search's first launch died on its own gate: *pair 88 base d_pose 0.00021283725596750921 disagrees
with the n600 base vector 0.00021283759625946122*. The gate was an exact-equality test on a float
re-evaluation, which is the wrong instrument — the n600 vector is measured at PoseNet batch 8 and the
search re-evaluates at batch 1, and up2 §6 already measured that batch shape moves the pose vector.
MAIN corrected it to a MEASURED band rather than a chosen epsilon, and the band was measured:

| quantity | measured |
|---|---:|
| batch-1 repeat, all 16 pairs | **exact** (bit-identical) |
| batch-1 vs batch-8, max **relative** gap on the 12 treated pairs | **3.290e−05** |
| batch-1 vs batch-8, max **absolute** gap, all 16 pairs | **2.586e−09** |
| relative gap on pairs at ~1e−07 d_pose | up to 5.33e−03 — the same ~1e−09 absolute noise over a tiny denominator |
| gate used | **3.29e−04** = 10× the observed maximum on the treated population |

Observed gap per treated pair is recorded on every row (`base_gap_rel_vs_n600`); the largest seen in the
refine set is inside the band by construction or the row would not exist. **The lesson is the
denominator, not the epsilon**: a relative tolerance derived from the whole population would have been
10× too loose for the pairs this arm actually treats, because the noise is absolute and the population
spans 3,000× in magnitude.

Also owned: the first shard launch mis-assigned its pairs because zsh arrays are 1-indexed, so shard 0
received an empty `--pairs` and the last pair set was never launched. Caught by the argument parser in
seconds, no measurement contaminated; the relaunch passes every shard's pairs explicitly.

---

## 11. What this does NOT claim

1. **No score, no candidate, no build.** The pointer is unmoved and no archive was written. Nothing here
   is an exact row.
2. **verdict_scope: FORMULATION → FAMILY, on this object.** What is closed by measurement is
   *pose-directed per-pair actuation through the shipped 3-bit `frame_embed` lattice and through single
   token edits, on move 49's hard pose pairs, admitted at seg neutrality*. The floor probe generalises
   that beyond the single-code neighbourhood (8-code perturbations included), which is why the verdict is
   family-scoped on this object rather than instance-scoped — but it is scoped to THIS carrier
   parametrisation. A carrier with more than twelve coefficients per pair is a different object and this
   arm says nothing about it.
3. **Two-code moves were not swept exhaustively.** The floor probe covers random multi-code perturbations
   at sizes 2, 4 and 8 (23 rows), not the 1,372-point two-code grid per pair. A directed two-code search
   is not ruled out by measurement; it is ruled *unpromising* by the floor, and that distinction is
   preserved.
4. **The container search was not run** on the fee scan (§4), so the +44.00 B is the shipped-shape fee
   and an upper bound on what a searched candidate would pay.
5. **Actuator B's sample is 5 nominations on 1–2 pairs**, not a family sweep. Its claim is bounded to
   "smaller supply than A, and seg-neutral where A is not".
6. **No token-stream price was measured** for the B moves, because none reached the pricer.

---

## 12. What this hands the next arm

1. **The hard pose pairs are unreachable from the render.** C4 (not carrier slack) plus §6 (not
   frame_1-reachable) together say the top-12 residual lives outside both actuators. Any future pose
   arm should stop proposing render-side actuators for these pairs and target the **reach**: more
   carrier DOF for the specific hard pairs, or a different frame_0 parametrisation. Whether extra DOF is
   payable in bytes is exactly what pc2/pc3 priced at a 349.9 B family slack — that door is priced, and
   this arm has now measured that the alternative door is shut.
2. **The pose mass is more concentrated than the top-12 framing suggests**: max/median is 636×, and the
   top 32 pairs hold 72.07 %. A reach-side lever needs to serve only tens of pairs, which is the shape
   the carrier's per-pair addressing already has.
3. **The live semantic container is SM1S at q10/lgwin16+CK2 with a re-measured fee** (§4). Any arm that
   touches `frame_embed` or the renderer member inherits this table and the byte-identity control, and
   must NOT inherit fe1's numbers.
4. **The base-gate lesson** (§10) is a class: a relative tolerance on a population spanning three orders
   of magnitude is not one tolerance. Gate on the band measured for the sub-population you treat.
5. **A reusable harness**: `experiments/ddm_pp1_pose_actuation.py` carries the move-49 body loader, the
   SM1S section with exact member repack, the pose instrument on move 49's own decode, the cheap seg
   screen, the full realized chain, the floor probe, the saliency-nominated token search and the
   composition stage. `experiments/ddm_pp1_verdict.py` recomputes every number from the receipts.

---

## 13. Custody (ALWAYS KEEP THE PAYLOAD)

Store **`/Volumes/APDataStore/pact/ddm_pp1/`** — **31 MB**, 118 files, every one hashed in
`SHA256SUMS.txt` (written after the last measurement; the file itself is excluded from its own listing).
APDataStore rather than Vertigo per the charter's overflow rule: Vertigo held **38 GiB** free at launch,
below the 42 GiB threshold. Every measured payload is kept, not only the winners' — including the rows
that lost.

| path | what |
|---|---|
| `VERDICT.json` | every number in this memo, recomputed from the receipts by `ddm_pp1_verdict.py` |
| `base/POSE_BASE_MOVE49.json` · `base/pose_base_move49.npy` | the n600 per-pair pose base on move 49 and its ranking (F1) |
| `base/CONTROLS.json` | C1–C4, `all_pass: true` |
| `base/BASE_TOLERANCE.json` | the MEASURED batch-1/batch-8 reproduction band behind the gate (§10) |
| `base/RATE_FEE.json` · `base/RATE_FEE_N1.json` | the live-container fee, 40 exact re-encodes + an independent N=1 scan |
| `segscreen/seg_screen_*.jsonl` · `SEG_SCREEN_*.json` | **all 672** single-code candidates with their realized Δcells |
| `refine/refine_moves_*.jsonl` · `REFINE_MOVES_*.json` | the 24 refined survivors: stale pose, resolved pose, the re-solved 12 carrier codes, the refine trace |
| `floor/floor_rows_*.jsonl` · `FLOOR_PROBE_*.json` | the 24 floor-probe perturbations |
| `searchB/search_b_*.jsonl` · `SEARCH_B_*.json` | the 16 token nominations with their saliency and every leg |
| `compose/COMPOSE_88.json` | A, B and A+B on pair 88, each realized, plus the additivity ratio |
| `searchA_partial/*.log` | the 35 rows of the first (aborted) full-refine sweep, retained because they are real measurements of the pose/seg trade-off at large code steps |
| `logs/*.log` · `logs/*.manifest.json` | every detached run's log and its launcher manifest |

Producers: `experiments/ddm_pp1_pose_actuation.py` (stages `base | controls | base-tolerance | rate-fee |
seg-screen | refine-moves | search-a | search-b | floor-probe | compose`) and
`experiments/ddm_pp1_verdict.py`. Both ruff-clean. Nothing in `/Volumes/VertigoDataTier/pact/ddm_sj1_pass7`
was written; the pointer tree was read with `sys.dont_write_bytecode = True` in every run.

<!-- # FORMALIZATION_PENDING: an honest family negative with no exact row; the equations leg is written by tools/pointer_move_packet.py --equations-leg at harvest only when a row exists, and no row exists here -->

Own-vehicle frontier (unchanged by this arm — MAIN fires):
**S 0.13632299781031237 @ 179,153 B [contest-CUDA T4 n600]** (move 49).
