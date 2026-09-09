# ddm_pc2 — widen the CAP1 Rice-k field, then spend it on a carrier rank cut

`[no-triality] [p0-ledger-ok]` · arm `ddm_pc2` · 2026-09-08/09 · lane
`lane_ddm_pc2_carrier_kwidth_rankcut_20260908`

**Axes.** Bytes `[exact local byte arithmetic, receiver-verified parse-back through the public
path]`. d_pose `[macOS-CPU advisory, cpu_torch fp32 authority backend, n600, DALI GT]`.
`score_claim=false`, `promotable=false` for every row below: no contest-CUDA T4 row exists for any
candidate here, and MAIN owns the fire.

**Base.** The live pointer: `S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]`, archive sha
`06c44dc464038649f1cc149f04ac03a518294ffcf49b87d8f66df30eb3c63cd3`, lane
`ddm_sj1_t4_token_predistortion_pass3_20260906`, tree
`/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion/candidate_pass3/candidate_runtime`.
Re-read from `.omx/state/canonical_frontier_pointer.json` before every stage and every seal.

---

## 1. Identity controls — four of them, all on THIS body

Receipts: `/Volumes/VertigoDataTier/pact/ddm_pc2_carrier_kwidth_rankcut/identity/identity.json`.

| control | result |
|---|---|
| container rebuild: shipped codes through `up3.build_archive` | **sha `06c44dc4…`, 181,645 B — bit-identical to the pointer** |
| KW1-patched receiver parses the SHIPPED archive | carrier body, packed metadata and all 600×12 codes **identical** |
| KW1-patched receiver's PUBLIC path (`read_residual_archive` → `materialize_cpr1` → `decode_compact_carrier`) on the shipped bytes | **decodes the pointer's codes exactly** |
| this arm's own builder at r = 12 (drop nothing) | **181,645 B, sha `06c44dc4…`, delta 0** |

**The container shape had to be DERIVED, not inherited, and that is a finding.**
`ddm_up3` pins `BROTLI_QUALITY = 11`, `BROTLI_LGWIN = 24` — its own generation's shipped shape. The
live body is **q = 9, lgwin = 16**. Inheriting up3's constants produced 181,647 B (+2 B) and FAILED
the identity control. The arm now searches the container grid for the option that reproduces the
shipped carrier stream byte-identically and refuses if none does. This is exactly the failure
`up3.build_archive`'s own docstring warns about
([[binding-instruction-numbers-expire-and-nobody-rederives-them]]); the warning was in the code and
the constant was still stale.

**The carrier, re-measured on THIS body** (pc1's figures were the ×1 and ×16-on-rc1 bodies and do not
transfer):

| field | live pointer | pc1's body (§1) |
|---|---:|---:|
| RX1 `reserved` | `0x7a` (CK2-semantic · RR5 · **DX2** · RC1-semantic · RC1-hpac) | — |
| carrier stream (stored) | 18,621 B | 22,031 B |
| carrier body (canonical) | 18,872 B | 22,278 B |
| basis payload | 12,277 B / 98,213 bits | 12,277 B |
| Rice coefficient payload | **6,424 B / 51,385 bits** | 9,830 B / 78,634 bits |
| packed CAP1 metadata | 40 B | 40 B |
| Rice `k` | **all twelve = 5** | 8–9 |
| max \|code\| | 155 | 2,048 |

`ddm_pc1`'s ITEM 3 derivation used the 78,634-bit payload; the live payload is 51,385 bits, and the
live `k` is 5, not 8–9. Both numbers moved under sj1's re-solves. Every figure below is re-measured.

---

## 2. PRIOR-LAW PREDICTION vs MEASURED

The charter pre-registered: *"with a 2-bit k field, the r = 8 rank cut prices at −3,316 ± 400 B … and
re-solves to n600 d_pose ≤ 6.5e-06"*, with falsifiers *"the r = 8 rung's real encode saves < 2,000 B"*
and *"its re-solved d_pose exceeds the bar at every r ≥ 8"*.

| claim | predicted | MEASURED | verdict |
|---|---|---|---|
| k-field widening cost | ≈ +2 B (2-bit field) | **+3 B** (12 × 4-bit absolute) | corrected, §3 |
| r = 8 rate saving | −3,316 ± 400 B | **−6,356 B** | **prediction WRONG by 1.92×, in our favour** |
| r = 8 re-solved d_pose | ≤ 6.5e-06 | **≥ 6.313701e-04** (population floor, 39 pairs) | **prediction WRONG by ≥ 97×** |
| falsifier "saves < 2,000 B" | — | saved 6,356 B | did not fire |
| falsifier "d_pose exceeds the bar at every r ≥ 8" | — | 48.9× past the bar at r = 8, 2.8× at r = 11 | **FIRED** |

**Two corrections to the charter's own premise, both structural.**

1. **A 2-bit field is not enough.** The shipped field is `k_base` (u8) plus **one bit per dimension**,
   so it expresses a span of 2 adjacent `k`. The live columns sit at `k = 5` and a dropped column
   wants `k = 0`: a span of **6**. Two bits give a span of 4 — still short. The honest forms are a
   4-bit *absolute* field (6 B, **+3 B**) or a `k_base` nibble plus twelve 3-bit deltas (5 B, +2 B,
   span 8). This arm ships the absolute form and pays the extra byte (6.66e-07 S) on purpose: with no
   base/span coupling, no re-solve can produce a `k` vector the field cannot hold, so there is no
   fail-closed path to get wrong at seal time.
2. **The prize is nearly twice as large as derived, because the derivation counted only half of it.**
   The charter's −3,316 B came from the coefficient payload and the 4/12 column share. A dropped
   dimension also frees its **basis atom**, and that is the larger half.

---

## 3. Where a dropped dimension's bytes actually are (MEASURED, ablation on the live body)

A dimension is dropped by making its basis atom a CONSTANT symbol and its coefficient column zero.
The receiver needs no change for the basis half: `normalized_basis` centres per atom and divides by
`rms.clamp_min(1e-5)`, so a constant atom becomes exactly the zero field
(`cpr1/inflate.py:290-296`). The coefficient half needs `bias = 0` for that column, which makes every
AR(1) residual identically zero.

Per-dimension ablation, encoded with the shipped riders:

| half | mechanism | measured saving per dimension |
|---|---|---:|
| coefficients, column zeroed, `k` left at 5 | symbol content only; the `k = 5` remainder is still sent as 600 × 5 raw bypass bits | **115 – 178 B** |
| coefficients, column zeroed, `k = 0` | the bypass bits go away; DX2's adaptive terminator context converges to ~0 bits | **490 – 553 B** |
| basis, atom set to a constant symbol | RR5's per-atom adaptive-arithmetic context converges | **726 – 1,154 B** |

**The k-width unlock is the difference between rows 1 and 2, and it is closed-form.** Under DX2 the
Rice remainder is `k` equiprobable bypass bins per symbol, so a zeroed column at parameter `k` costs
`600 · k` bits regardless of its content. Moving a dropped column from the closest `k` the shipped
1-bit field allows (`k = 4`, beside live columns at 5) down to `k = 0` therefore saves exactly
`600 · 4 = 2,400` bits = **300 B per dropped dimension**.

MEASURED, by building both archives:

| r | dropped | archive with wide k | archive at a k the shipped field carries | **k-width unlock** | derived |
|---:|---|---:|---:|---:|---:|
| 11 | [2] | 180,291 | 180,589 | **+298 B** | 300 |
| 10 | [2, 7] | 178,587 | 179,184 | **+597 B** | 600 |
| 9 | [2, 7, 1] | 176,947 | 177,844 | **+897 B** | 900 |
| 8 | [2, 7, 1, 0] | 175,289 | 176,486 | **+1,197 B** | 1,200 |

The closed form predicts 300 B per dimension and the real encode measures 298–300 B — the 1–3 B gap
is the CABAC terminator's own adaptation, not a modelling error. **+1,197 B of archive for +3 B of
metadata: a 399:1 return, and the largest single-lever ratio this campaign has measured.**

---

## 4. The rate ladder, exact

Every row is a real `archive.zip` built through the KW1-patched receiver's own riders and stat-ed;
every one parses back through the public path to exactly its codes. Drop order is ascending realized
coefficient energy, MEASURED on the live body: `[2, 7, 1, 0, 6, 5, 4, 9, 10, 3, 11, 8]` (shares
4.90 % … 16.64 % — the profile is FLAT, there is no free dimension).

| r | dropped | archive B | ΔB vs pointer | ΔS rate | Rice `k` | break-even d_pose |
|---:|---|---:|---:|---:|---|---:|
| 12 | — | 181,645 | 0 | — | all 5 | — |
| 11 | [2] | 180,291 | **−1,354** | −9.015730e-04 | 5,5,0,5,… | see §6 |
| 10 | [2, 7] | 178,587 | **−3,058** | −2.036197e-03 | 5,5,0,…,0,… | see §6 |
| 9 | [2, 7, 1] | 176,947 | **−4,698** | −3.128205e-03 | 5,0,0,… | see §6 |
| 8 | [2, 7, 1, 0] | 175,289 | **−6,356** | −4.232200e-03 | 0,0,0,5,… | see §6 |

Marginal cost per dropped dimension: 1,354 → 1,704 → 1,640 → 1,658 B. Near-linear, as the ablation
predicts (each dimension owns its own basis atom and its own coefficient column, and brotli removes
only ~1 % of the carrier body, so the two halves do not double-count).

The shipped packer REFUSES the rank-cut `k` vector on this body
(`Up3Error: value escapes the 1-bit packed field`) — pc1 §6's container refusal, reproduced here
rather than recalled.

---

## 5. The KW1 format and its versioned reader

`src/tac/kw1_wide_rice_k.py` is the single source of truth and is copied into the candidate runtime,
so encoder and decoder run the same bytes (the RR5/DX2 precedent). The block keeps every shared field
at its shipped offset — including the 32 Huffman lengths the RR5 rider reads — and replaces only the
tail:

| field | shipped `[40 B]` | KW1 `[43 B]` |
|---|---|---|
| `factor_base` / `factors` / `biases` / `lengths` | `[0:37]` | `[0:37]`, unchanged |
| `k_base` | `[37:38]` | dropped |
| `ks` | `[38:40]`, 1 bit/dim over the base | `[37:43]`, **4 bits/dim, absolute** |

Both forms restore to the SAME canonical 80-byte metadata, so every receiver stage below the restore
is bit-identical between them. Selected by RX1 `reserved` bit `0x80` — the only bit the shipped
receiver left unclaimed (`SZ1_RESERVED_KNOWN_BITS = 0x7F`, widened to `0xFF`).

`experiments/ddm_pc2_carrier_receiver_patch.py` applies the smallest diff that carries the block width
as a parameter through the three modules that hard-code 40 (`rr5_arith_basis.split_carrier_body` /
`assemble_carrier_body`, `dx2_cabac_coefficients.packed_ks`, `residual_archive._packed_portion` /
`_restore_packed_cap1_metadata`). Every edit is an exact string replacement that must match its
shipped anchor **exactly once** or the patch refuses, and every width defaults to 40 — so an archive
without the KW1 bit takes exactly the path it takes today. That is the control in §1, row 2.

**A silent-mismatch hazard found and closed while building this.** Python caches `runtime.*` modules
by name, so importing the receiver from the pointer tree and then again from the patched tree
silently returns the FIRST one — which would have made the patch-inertness control a false pass. Every
receiver import in this arm now purges the module cache and then PROVES each module's `__file__` sits
under the requested tree. Same genus as sj1's silent-revert classes (gs3 Addendum 14): a check that
encodes a round-one premise keeps passing after the premise moves.

---

## 6. The base, re-measured on this instrument

600 pairs, `cpu_torch` fp32, DALI GT, on the pointer's own body and its own decode:

    d_pose n600 = 5.09276404439735e-06      pose leg sqrt(10 d) = 0.007136360448013644
    per-pair median 5.05842e-07   min 4.59980e-09   max 2.12838e-04

Receipts: `base/base_d_pose.json`, `base/base_per_pair.npz`, `base/rows_r12_shard*.jsonl`.
Every break-even below is computed against THIS number, not against a recalled 5.1e-06.

| r | rate saving | break-even d_pose | as a multiple of base |
|---:|---:|---:|---:|
| 11 | −1,354 B | 6.460837e-06 | 1.269x |
| 10 | −3,058 B | 8.413580e-06 | 1.652x |
| 9 | −4,698 B | 1.053613e-05 | 2.069x |
| 8 | −6,356 B | 1.292442e-05 | 2.538x |

---

## 7. The rank cut is REFUSED, and the reason is the LATTICE, not the span

### 7a. At first order a rank cut is FREE. Measured.

Before spending solver time on a drop ORDER I measured whether the order can matter at all. For six
strided pairs I formed the pose Jacobian `J = d(pose)/d(coeff)` (6 x 12) at the shipped codes and
asked, per dimension, how much of column j the other eleven already reproduce:

| quantity | result, all twelve dimensions |
|---|---|
| residual share of column j after projection onto the other eleven | **0.000000** |
| increase in the min-norm pose residual when column j is removed | **-2.3e-18 to 7.7e-16** (numerical zero) |

`J` has six rows, so any six or more columns span its whole row space. **Every carrier dimension's
pose-Jacobian column lies exactly in the span of the others, and removing one — or four — costs
nothing at first order.** Three consequences, and the third is the finding:

1. The drop ORDER is irrelevant. Ascending energy, ascending residual share and ascending reach loss
   give three DIFFERENT orderings (`[2,7,1,0,...]`, `[3,4,10,9,...]`, `[4,5,7,0,...]`) with the same
   zero first-order cost. No re-ordering can rescue a rung.
2. There is no low-energy dimension to spend either: the realized-energy profile is FLAT,
   4.90% to 16.64%.
3. **The measured cost is therefore not the SPAN. It is the LATTICE.** The carrier has to land on a
   representable point — signed int12 at a fixed step, rendered through two roundings — that matches
   six PoseNet outputs to ~1e-06. Removing coordinates removes the fine placements that the surviving
   coordinates cannot reach on their own. This turns pc1 section 2b's "positioned subspace" from a
   phrase into a mechanism: **the carrier's bytes buy a LATTICE POINT, not a subspace.**

Receipts: `jacobian/jacobian_leverage.json`.

### 7b. What the solver measured

Every rung ran the masked form of `jg5.refine_pair` — 40 outer rounds, the same step ladder, the same
realized acceptance test, the same materiality stop — with the SEARCH restricted to the retained
dimensions and EVALUATION running the exact 12-dimensional receiver path. Every pair stopped at
`no_improving_step`, a physical stop.

| rung | pairs | subset mean d_pose | x base on the same pairs | projected n600 | x its own break-even |
|---|---:|---:|---:|---:|---:|
| r = 11 (drop 1) | 11 | 1.667243e-05 | 3.6x | 1.8197e-05 | **2.8x** |
| r = 8 (drop 4) | 39 | 9.713386e-03 | 2,733x | 1.3920e-02 | **1,077x** |
| r = 8, retained step x1/2 | 8 | 8.399035e-04 | 136.3x | 6.9428e-04 | **57.2x** |

A subset may only REFUSE, never admit (pc1's rule), so the estimates above are for routing and the
VERDICT rests on the strict population floor `d_pose_n600 >= sum(measured finals)/600`, which no
unmeasured pair can lower because every per-pair d_pose is non-negative.

**r = 8 is REFUSED on that floor, not on an extrapolation.** Over 39 strided pairs (four shards of a
stride-25 partition, never a prefix):

    d_pose_n600  >=  0.3788221 / 600  =  6.313701e-04     against a break-even of 1.292442e-05

That is **48.9x past the bar**, and it is a completed argument on the whole population. Solving was
stopped there: a verdict already decided is not made more decided by more pairs.

The r = 11 row is pruned on the charter's own >2x SCOPE rule with its row recorded. A rigorous floor
there would cost ~222 pair-solves to buy a verdict already visible at 2.8x, and section 7a says the
gentler rung cannot be rescued by re-ordering either.

### 7c. The lattice cure was priced, tried, and does NOT systematically work

If the cost is lattice resolution then the cut should pay for its own cure: the shipped codes reach
only `|code| <= 155` of the +-2047 field, so the retained columns can take a finer step. Each halving
costs one Rice bit per retained symbol (600 bits per retained column) against ~1,590 B per dropped
one. Priced exactly — and **every rung needs a k span of 6 to 8, which the shipped one-bit field
cannot hold**, so KW1 is a precondition of the cure, not an accessory to it:

| r = 8 with retained step | archive B | delta B | max abs code | k span | break-even |
|---|---:|---:|---:|---:|---:|
| x1 (shipped) | 175,289 | -6,356 | 155 | 5 | 1.292442e-05 |
| x1/2 | 175,819 | -5,826 | 310 | **6** | 1.213446e-05 |
| x1/4 | 176,411 | -5,234 | 620 | **7** | 1.128155e-05 |
| x1/8 | 177,019 | -4,626 | 1,240 | **8** | 1.043794e-05 |

MEASURED, paired against the plain cut on the SAME eight pairs: refine x1/2 wins 2.73x on one pair,
sits within +-5% on four, and LOSES on three (0.14x, 0.70x, 0.75x). Refine x1/8 measured 2.9533e-02
on pair 0 — 2,500x worse than the plain cut — because `refine_pair`'s +-2 polish neighbourhood spans
eight times LESS coefficient distance at that step. **The binding constraint is the SOLVER's local
reach, not the lattice alone.** That is pc1 section 7f's non-monotonicity finding read in the other
direction: a coarser lattice hands the search a wider local move, and a finer one takes it away.

### 7d. Verdict

**verdict_scope: FORMULATION — a rank cut on this carrier, at any drop order, with or without a
retained-lattice refinement, solved by `refine_pair`.** The rate side is large and real (-6,356 B at
r = 8, of which +1,197 B is the KW1 field's own contribution). The pose side exceeds its break-even by
20x at r = 8 and by 2.8x at the gentlest rung. What is NOT closed: a solver whose local move scales
with the lattice step (the x1/8 failure is a SEARCH-reach failure, not a representation failure), and
a joint re-solve that moves the basis atoms along with the cut instead of holding them fixed.

---

## 8. ITEM 1 — ADMITTED, and it needs no receiver change at all

pc1 left the twelve per-atom basis scales as "48 recoverable bytes, but NOT free by construction":
they cancel in `normalized_basis` to 1.9073e-06, and dropping them still changed 14 of 24 rendered
`frame_0` images by +-1 uint8. Recovering them by SHRINKING the block needs a second format change,
so I priced the POSE first at zero byte change — and the byte side turned out to be free anyway.

**Setting all twelve basis scales to 1.0 saves 41 bytes with no format change whatsoever**: twelve
identical float32 words compress where twelve distinct ones do not. The block stays 96 bytes, every
offset is untouched, the RX1 `reserved` byte stays `0x7a`, and the receiver ships unmodified.

| | value |
|---|---|
| archive | **181,604 B** (-41 B), sha `6d3717cf8621dbd5d8eb6fbbf7e99f1cb2db1c520088362735fce46cd33bb06a` |
| receiver | SHIPPED, unmodified — no KW1 bit, no patch, no new decode path |
| twin encode | byte-identical |
| parse-back through the public path | returns exactly the shipped codes |
| container sweep | 0 B recovered; the shipped shape wins |
| **d_pose n600** (cpu_torch fp32, DALI GT) | **5.0939863022125565e-06** vs base 5.09276404439735e-06 = **1.000240x** |
| pairs whose d_pose moved | **333 of 600** — pc1's "not neutral by construction" correction, confirmed at n600 |
| dS rate / pose / **net** | -2.7300217e-05 / +8.563080e-07 / **-2.6443909e-05** |
| break-even d_pose | 5.131803e-06; the measured 5.093986e-06 sits at 0.99x of it |
| **verdict** | **ADMIT** — clears the -2e-05 bar by 1.32x |

A sweep of other constants found `basis_scales = coefficient_scales` saves 44 B rather than 41, but
3 bytes (2.0e-06 S) does not buy another 25-minute n600 pose measurement, so the shipped candidate is
the 1.0 form whose pose IS measured. Shrinking the block to 48 B would add another 48 B (-89 B total,
net -5.84e-05) and is left owed, because it needs a second receiver format change and the -41 B row
needs none.

### The seal

Smoke PAIR, all four legs, checked with the seal's OWN `tac.candidate_seal._public_smoke_problems`
rather than a second reading of the contract (`problems: []`):

| leg | role | outcome | seconds | rc |
|---|---|---|---:|---:|
| public path (`f26_inflate.inflate_archive`, CPU) | candidate | REACHED_TOKEN_DECODE | 120.0 | — |
| public path | frontier | REACHED_TOKEN_DECODE | 120.0 | — |
| `bash inflate.sh` (3-arg contest signature) | candidate | REACHED_CUDA_GATE | 1.8 | 1 |
| `bash inflate.sh` | frontier | REACHED_CUDA_GATE | 1.4 | 1 |

    SEAL READY
      /Volumes/VertigoDataTier/pact/ddm_pc2_carrier_kwidth_rankcut/SEAL_ddm_pc2_carrier_kwidth_rankcut.json
      candidate  ddm_pc2_carrier_kwidth_rankcut  [contest_cuda]  SEAL_VALID
      archive    181,604 B  sha 6d3717cf8621dbd5d8eb6fbbf7e99f1cb2db1c520088362735fce46cd33bb06a
      runtime    43 files, 925,422 B, digest 9e96ed4e79b4d0e1…
      seal sha   3016c8aa11aa5de3ffe748895c23c2f097770d55ec3dfc18c666e3b5700ff934
      admit bar  net dS < -2e-05 vs contest_cuda 0.13900438 (tolerance 0)

**MAIN fires; I did not dispatch Modal.**

**Two producer defects found and fixed while building that smoke, both worth keeping.**

*The bound trap.* Declaring one number for both the subprocess timeout and
`public_path_probe_seconds` refuses every timed-out probe, because a timeout records
`elapsed = timeout + epsilon` and the validator refuses `seconds > bound`. A timeout is the EXPECTED
outcome of the token-decode leg on a CPU host, so the trap fires on the normal path, not an edge
case. Probes now run at 0.8x the declared bound.

*The orphan child.* The public-path probe starts a full `inflate_archive`. When its parent died the
child survived under launchd at ~180% CPU, writing into a temp dir nobody reads — MEASURED at 11m52s
of orphan runtime before it was killed. Every probe now starts with `start_new_session=True` and the
bound path `killpg`s the group, so no probe can outlive its caller. The declared bound also dropped
from 750 s to 150 s: the PASS condition is "no exception within the bound" and every pre-decode stage
throws fast, so the bound only has to outlast those stages, never the 25-minute decode.

---

## 9. The fe1 container law, priced on this arm

fe1 measured that on this body an edited RC1 semantic payload pays a flat ~+70 B, and that
re-searching the encoder-only container choices (CK2 interleave, brotli quality, window) recovers
~47-63 B. Every candidate here is now built with that sweep and both prices are recorded.

**MEASURED: the sweep recovers 0 B on every ddm_pc2 candidate.** The shipped shape
(ck2 = False, q = 9, lgwin = 16) wins outright at r = 12, 11, 10, 9, 8 and on the ITEM 1 body. The law
does not reach this arm, and the reason is structural: fe1's tax is a SPARSE edit into a range-coded
stream whose suffix re-randomises, while these edits rewrite whole columns of a section that is
already Rice/CABAC-packed, where the container has no match structure left to lose. Ties now go to
the shipped shape, so the sweep can never buy a moving part for nothing.

---

## 10. What this arm did NOT do

- **No T4 row was fired. Modal was never dispatched; MAIN fires.** Every score-shaped number here is
  local advisory with `score_claim=false` and `promotable=false`.
- **ITEM 4 (the 40-round re-solve as attribution control and zero-byte probe) is NOT measured.** Its
  r = 12 rung exists and the base row in section 6 is its start, but the re-solve itself never ran:
  the CPU went to the rank-cut ladder and then to ITEM 1. It stays the most interesting owed item on
  this carrier, because any d_pose it wins costs ZERO bytes and composes with ITEM 1.
- Only r = 8 carries a rigorous population floor. r = 11, r = 10 and r = 9 were pruned on estimates
  (with their rows recorded) once section 7a showed no re-ordering could rescue them.
- ITEM 1's 48-byte block shrink was not built; only the 41 bytes that need no format change were.
- The KW1 format is BUILT, controlled and proven, but **it does not ship in this arm's candidate**:
  the only admitted row does not need it. It is a mechanism waiting for its consumer.

## 11. Owed items

### ITEM 1 — the zero-byte re-solve (pc1's own ITEM 4)

40-round `refine_pair` over n600 on the ITEM 1 body (`candidate_scales/candidate_runtime`). Any
d_pose it wins costs ZERO bytes, so its whole dS is negative by construction, and it composes
additively with the -41 B scales row. The base it starts from is measured (5.0939863022125565e-06).
~5-7 h at 4 shards. This is the most interesting unmeasured thing left on this carrier.

### ITEM 2 — shrink the scales block from 96 B to 48 B: MEASURED at -7 B, REGISTERED AND SKIPPED

I priced this at the container with no receiver work and no pose run. RR5 and DX2 pass the scales
region through verbatim (`split_carrier_body` slices `[6:102]` into `scales`, `assemble_carrier_body`
writes it back unchanged), so the 48 basis-scale bytes sit verbatim in the ENCODED body at `[6:54]`;
deleting them and re-compressing at the shipped container shape prices exactly what a 48-byte block
would buy.

    carrier stream, 96 B block : 18,580 B
    carrier stream, 48 B block : 18,573 B
    ITEM 2 buys                :     -7 B

**Once all twelve basis scales are identical the 48 raw bytes are already down to 7 B of compressed
match — the -41 B ITEM 1 banks IS 41 of the 48.** So the remaining prize is -7 B = -4.66e-06 S, which
does not clear the -2e-05 admit bar on its own, and it would cost a change to the RENDERER's own
reader (`cpr1/inflate.py::decode_compact_carrier` reads the 12+12 float32 layout), not just the three
container modules KW1 parameterised. That is a materially larger and riskier diff than KW1, on a
candidate whose entire virtue is that it ships an UNMODIFIED receiver, for 0.6% of an already-small
row. Registered and skipped on the measurement, not on a guess.

What WOULD make it worth building: any future candidate that already changes the renderer's carrier
reader for another reason. The 7 bytes are then free to collect.

### ITEM 3 — a solver whose local move scales with the lattice step

The x1/8 retained-lattice refinement failed as a SEARCH, not as a representation: `refine_pair`'s
+-2 polish neighbourhood spans eight times less coefficient distance at that step, and pair 0 landed
2,500x worse than the plain cut. A polish radius expressed in COEFFICIENT units rather than lattice
units would decide whether the rank cut's lattice cure actually works, and it is the only route that
could reopen section 7's FORMULATION-scope refusal.

### ITEM 4 — find the KW1 field's consumer

The format is built, controlled and proven (+3 B, reads both forms, inert on the shipped archive) but
its only measured consumer — the rank cut — is refused, and per-column lattice coarsening does not
need it (the columns' optimal `k` move together; priced at <=1,080 B). Any future carrier edit that
puts columns at genuinely different `k` now has a 43-byte block that can express it. Worth pointing
at the next carrier lever before that lever is designed around a two-value `k` constraint that no
longer exists.

## Frontier

    sj1  S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]   <- the live pointer, unmoved

    ddm_pc2 ITEM 1 candidate (ADVISORY, score_claim=false, promotable=false):
      S 0.13897793405941966 @ 181,604 B  [macOS-CPU advisory projection: measured d_pose n600
      cpu_torch + exact archive bytes + d_seg carried from the sj1 T4 row]
