# ck2 — the eleventh-move candidate: a parameter-free container transform, −657 B at zero distortion

**Arm:** `ddm_ck2` · **Base:** the ck1 pointer, archive
`35c318d541d703708ab06c55473c200bb893491e24bea312e37be42f010677e3` @ 177,182 B,
**S 0.15710198138050818** `[contest-CUDA T4, n600]` (d_seg 0.00030309 · d_pose 7.77e-6 ·
rate 0.11797822103209257).

`verdict_scope`: **INSTANCE** — the ck1 SM3R mode-6 semantic body (36,130 B) and its
compensated carrier body, measured as exact byte counts. The container credits below are
properties of *these* bodies, not of the family. No score is claimed here: the T4 row is
MAIN's to fire.

---

## 1. The candidate

| | value |
|---|---|
| archive | `0aa1cada2ca79ad43a11bfa72be69a5240315e35cf5b4c94665d60d0c3583933` @ **176,525 B** |
| Δ bytes vs ck1 | **−657 B** |
| rate ΔS | **−4.374693e-04** (exact by construction) |
| d_seg ΔS | **0** — decoded state bit-identical |
| d_pose ΔS | **0** — decoded state bit-identical |
| **net ΔS** | **−4.374693e-04** |
| projected S | **0.15666451204830689** |
| vs the −3.5e-6 admit bar | **125.0×** |
| vs ck1's report-8dp error bound (3.336608e-6) | **131.1×** |
| fraction of the 0.00710198 gap to 0.15 | **6.16%** |
| staged runtime | `/Volumes/APDataStore/pact/ddm_ck2/generations/ck2_plane2_r1` (33 files, pristine) |

**The distortion legs are zero by construction, not by measurement, and that is the point.**
This candidate changes only how four already-decided section bodies are laid out before
Brotli. The receiver restores each body byte-for-byte before any parsing, so every
downstream length check, magic check, and field offset sees exactly the bytes it sees
today. There is no seg leg to discount — which matters, because the tenth move realized
only 35% of its projection and the entire miss was a CPU-modelled seg leg
(`cpu_to_cuda_seg_transfer_has_no_law_20260819`). This row carries no such exposure.

**The identity control is the proof.** Built through the same path with the container
transforms off, the tool reproduces the ck1 pointer archive **byte-identically**:
`candidate_nosplit` = `35c318d541d7…` @ 177,182 B. The build path is therefore the one
that produced the T4 row, and the only thing that differs in the candidate is the
serialization.

---

## 2. The mechanism, and why the parameter-free form wins

sz1 shipped `DDM_SZ1_SEMANTIC_METADATA_SPLIT_V1`: a 2-plane byte de-interleave over a
**pinned region** of the semantic body — offset 49, length 8,284 — worth −520 B on sz1's
un-pruned 36,040 B body. Those two integers were fitted to that layout. ck1's SM3R mode-6
row-prune moves the layout, and the fitted constants stop paying.

Measured on the ck1 body (`CK2_RATE_CEILING.json`), semantic section, Brotli q11 lgwin 24:

| serialization | section B | credit |
|---|---:|---:|
| identity (shipped) | 31,469 | — |
| **sz1's pinned constants, carried** | 31,528 | **+59 (a LOSS)** |
| best (offset, length) over a 2,048-step grid | 30,911 | −558 |
| **whole-section 2-plane, no fitted constants** | **30,856** | **−613** |

Two readings, both worth keeping:

1. **The carried constants cost 672 B against the parameter-free form.** That is
   `cross_regime_constant_transfer_genus_finishing_stage` firing inside our own tool,
   priced. sz1's constants were correct for sz1 and are simply not a property of this body.
2. **Fitting the region at all is worse than not fitting it.** The (offset, length) argmax
   loses to the whole-section transform by 55 B. So the winning form carries *nothing* to
   re-solve when the body layout changes again — and nothing that could be read as
   video-derived content smuggled into free runtime code. Rule-118 spotless.

Plane count was swept rather than assumed: k=2 → −613, k=4 → −330, k=6 → −336, k=8 → −37,
k=3/5/12/16 all positive. k=2 is right because the region is interleaved fp16.

The same transform on the carrier body is worth a further **−44 B** on the *compensated*
lattice (22,187 → 22,143). It is measured, not assumed, and its sign is content-dependent:
on the *uncompensated* control lattice the same transform costs +41 B. Two credits over
the same redundancy do not add, and here they do not even share a sign.

**Receiver cost: zero counted bytes.** Two new `reserved` bits (0x02 semantic, 0x04
carrier) on the existing 14-byte RX1M header, which already had an unknown-bit guard that
refuses fail-closed. `reserved == 0` remains byte-identity, so every prior archive takes
exactly the path it takes today.

---

## 3. Where the rest of the field stands on THIS base

Ranked against the ck1 base, with the seg leg discounted as an upper bound wherever a
candidate leans on a CPU-modelled seg delta (the tenth-move law). Sources: the nine
candidate memos swept for this ranking.

| # | candidate | measured on | rate Δ | seg/pose exposure | net on ck1 | gap closed | verdict |
|---:|---|---|---:|---|---:|---:|---|
| **1** | **ck2 whole-section plane2 (this row)** | **ck1, exact** | **−657 B** | **none — bit-identical decode** | **−4.375e-04** | **6.16%** | **STAGED** |
| 2 | fx1 miss-sector relative law | rr4, priced ceiling only | ceiling −1,247 B | none (coder) | ≤ −8.3e-04 | ≤ 11.7% | UNBUILT, priced |
| 3 | iv1 route-1 native WANS1 | rr4 | ~−276 B *if it converged* | unmeasured | ~−1.8e-04 | 2.6% | did NOT converge; FORMULATION |
| 4 | mz2 mixed q3/q4 semantic | e480b 183,502 B | −823 B | **unmeasured, both axes** | indeterminate | — | superseded: ck1 already 3-bits `frame_embed` + `blocks.0.film` |
| 5 | mz2 FiLM row-sparsity ladder | e480b 183,502 B | −130…−2,051 B | unmeasured | — | — | superseded: keep01 (keep 1%) is already IN ck1 |
| 6 | fx1 candidate B `k1_cb16` | rr4 | −341 B code length | none | — | — | superseded: ck1's tail already carries fx2's −711 B build |
| 7 | t1h carrier re-solve, passes 1/2/3 | rr4 | −5…+22 B | **T4-REFUSED: d_pose ×6.31** | +0.0126 measured | — | WITHDRAWN, DO NOT FIRE |
| 8 | iv1 SD1M `frame_embed` pose actuator | rr4→sz1 | +314 B own | **T4-REFUSED: d_pose ×2.73** | +0.00645 measured | — | REFUSED |
| 9 | qs5 partial de-trim + fresh Schur | cp135 | +26 B | −17 realized flips | +2.520e-6 | — | REFUSED on rate |
| 10 | me1 coder rows (7) | rr4 | +67…+1,398 B | 0 by construction | positive | — | REFUSED |
| 11 | mz2 exact fixed-schema forms (4) | e480b | +340 B all four | — | — | — | REFUSED |
| — | qs2 · re1 · sa3 · fx1-A · sz1 split | cp135 / sz1 | — | — | — | — | already CONSUMED or APPLIED in this lineage |

Rows 4–6 are the interesting ones, and the honest reading is that **ck1 ate them**: every
one targets a tensor or a stream that ck1's own row-prune, depth table, or fx2 tail has
already re-represented. Their byte credits were measured on bases 3–6 KB larger and do not
transfer; re-deriving them here would be a joint re-solve, not a byte merge.

**The one live reopening.** `ddm_bp1_section_coding_axis_closed_20260818.md` closed the
section-coding axis at −5 B total, having measured global byte-plane de-interleave at
k∈{2,3,4,8} as a **loss on all three sections at every k**. That verdict is correct and its
own `verdict_scope` says INSTANCE, on sz1's `debb025f…` body. On ck1's row-pruned body the
same transform is **−613 B**. Nothing about bp1 was wrong; the row-prune changed the body,
and a refused family is a calibrated knob rather than a closed door
(`read-closed-negatives-as-actuator-datasheets`). This is the second time in this lineage
that pruning FiLM rows moved a container credit's sign — the first was sz1's own split
going from −520 to +151 on the sa3-edited body.

---

## 4. THE ROUTE question — is there a rate path to −10,666 B?

The gap to 0.15 is 0.00710198, which on the rate axis alone is exactly **−10,666 B**
(archive floor 166,516 B). Asked directly whether the priced inventory reaches it: **no,
and the shortfall is large enough to state plainly rather than round toward.**

Section budget on the ck1 member (177,082 B) with each section's *measured* free-rate
headroom — free meaning zero distortion:

| section | bytes | share | measured free headroom | source |
|---|---:|---:|---:|---|
| tail (fx2 tokens + residual + table codes) | 109,897 | 62.06% | ~−1,247 B ceiling | fx1's miss-sector law, priced not built; SSE/APM measured LOSING 6/6; scan-order already harvested by fx2 |
| semantic | 31,469 | 17.77% | **−613 B (banked here)** | this arm |
| carrier | 22,187 | 12.53% | **−44 B (banked here)** | this arm |
| hpac | 13,515 | 7.63% | 0 | re-Brotli +40; bp1 region byte-plane +7 |

**The container axis on this base is now EXHAUSTED, and all of it is banked here.** All
four sections were swept at k ∈ {2,3,4,5,6,8,12,16}: semantic −613 (k=2), carrier −44
(k=2), hpac 0 (already at its Brotli fixed point; re-Brotli is +40), tail 0 (stored raw and
incompressible — re-Brotli is +5, since it is arithmetic-coded). Total −657 B, all of it in
this candidate. **There is no follow-on container rung on this lineage** — a later arm
should not re-open it unless the body layout changes again, at which point the sweep is one
cheap probe (`experiments/ddm_ck2_rate_ceiling_probe.py`).

**Total credible free-rate reservoir ≈ 1,900 B ≈ 18% of the gap**, of which this row banks
657 B. The remaining ~8,750 B would have to come from lossy representation change, and that
family's exchange rate is measured: `compensated_semantic_edit_exchange_v1` (sa3, T4)
retains **10.5% of rate credit at ×1 mass, 12.6% at ×4**. Buying 8,750 B of *net* score
through that family therefore needs roughly **70,000 B of gross rate credit** against a
31,469 B semantic section and a 22,187 B carrier. sa3's own memo reaches the same wall from
the other side: closing its gap needed 1.20× the whole semantic section, so the family is a
**CONTRIBUTOR, not a route.**

So: the rate axis is the cleanest road *per byte* — exact, immune to the seg-transfer
discount, and it is where this row lives — but its **priced** reservoir on this lineage is
about 2 KB, not 10.7 KB. **Rate alone does not reach 0.15 from the priced inventory.**

**The one place a rate route could still live, stated so it is not lost.** fx1's bit
decomposition of the tail (n600, 117,964,800 positions) measured that 0.190% of positions
are misses and they carry **70.01% of the bits — 78,489 B**, of which **77,241 B is the
cost of *being* a miss**, the quantity the mixer's `q` adapts. That single number is the
largest reservoir anywhere in the archive: it is 43.7% of the whole member and **7.2× the
entire 10,666 B gap**. Realized against it so far: fx1 −560 B, then fx2 −797 B — about
**1.9% extracted**. Closing the gap on rate alone therefore needs roughly an **11%**
improvement in the miss-cost model. Nothing priced supports that today (fx1's remaining
named item is a 1,247 B sector-law ceiling; SSE/APM measured losing 6/6; scan-order is
already harvested), so it is not a route I can rank. But it is the correct place to point a
model-axis arm, and it is unpriced rather than refused — the distinction matters, because
every other section in the table above has been measured to saturation.

The remaining ~82% of the gap is otherwise a representation question — a smaller sufficient
statistic, not a better container — which on current evidence is the js1/js8 joint-nonlinear
line rather than any container or coder rung.

---

## 5. What is owed and what is not claimed

- **No T4 row fired by this arm.** No paid dispatch, no push, no hosting, no PR.
- **The local n600 CPU advisory is the gate**, per the composed-archive class cure: this
  candidate's runtime carries a receiver overlay the base lineage does not have, so it must
  be decoded AND scored end-to-end locally before any paid axis row. Fired as
  `/Volumes/APDataStore/pact/ddm_ck2/advisory/attempt_0001` via the canonical firer.
  Expected: d_seg and d_pose **equal to ck1's advisory to 8dp** (0.00043336 / 0.00014829),
  because the decoded state is bit-identical. A drift on either axis falsifies the
  bit-identity claim and voids the candidate.
- **Parse-back PASS on all six variants** is a structure-level control and does not by
  itself prove a probability-model match — but here nothing in the probability chain moved:
  the tail is spliced byte-identically and the receiver change is a permutation outside the
  arithmetic decoder.
- **The plane2 permutation is duplicated** between the compile tool and the generated
  receiver text, because the receiver ships inside the archive runtime and cannot import
  the repo. The two copies are not trusted by inspection; parse-back decodes through the
  generated receiver and compares recovered codes and semantic sha, so a divergence fails
  the build.

## 6. Custody

All under `/Volumes/APDataStore/pact/ddm_ck2/`:

| path | contents |
|---|---|
| `probe/ceiling_r1/CK2_RATE_CEILING.json` | the section census + per-section headroom + split search |
| `probe/ceiling_r1/sections/` | all four section bodies + the decompressed semantic body, persisted before their lengths were reported |
| `overlay/runtime/residual_archive.py` | the ck2 receiver, sha `c40a7c58b8e6dbb4…` from base `eebd34c96347ad13…` |
| `overlay/CK2_RECEIVER_OVERLAY.json` | anchors applied + round-trip control (synthetic lengths 0–64 and all four real bodies) |
| `compile/build_r1/SA3_REBASE.json` | the six-variant build report incl. the identity control |
| `compile/build_r1/*.zip` | all six archives — every candidate retained, not only the winner |
| `generations/ck2_plane2_r1/` | the staged runtime (33 files, pristine, inflate pin updated) |
| `advisory/attempt_0001/` | the n600 CPU advisory |

Code: `experiments/ddm_ck2_rate_ceiling_probe.py`,
`experiments/ddm_ck2_build_receiver_overlay.py`, and the `ck2_plane2` row plus the
container-variant plan in `experiments/ddm_sa3_rebase_sz1.py`.
