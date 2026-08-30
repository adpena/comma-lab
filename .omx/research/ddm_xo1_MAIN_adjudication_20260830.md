# MAIN adjudication of ddm_xo1 — the intersection verdict CONFIRMED, the demand figure CORRECTED BY A SIGN, and three of the four "byte-feasible" cells are payload counts without a receiver

**Date:** 2026-08-30
**Adjudicates:** `.omx/research/ddm_xo1_cross_successor_object_20260830.md` + `.jsonl`
(commits `f1fe706408` / `f417bf967d`, rc=0, 987 s, 21 rows).
**Axis discipline:** every cross-object sum below is DERIVED-FEASIBILITY, never a score. The lb1 row
is `[contest-CUDA T4 n600]` authority; born/qbz1/bz2/rb1 rows are advisory or scorer-free.

## 1. What I re-derived independently (not quoted)

xo1's headline arithmetic reproduces EXACTLY:

```
zero-distortion archive ceiling = 0.12 · 37,545,489 / 25 = 180,218.3472 B
lb1 ships 180,083 B                     → slack = +135.3472 B  (UNDER the ceiling)
lb1 rate = 25·180,083/37,545,489        = 0.11990987785509999  (< 0.12 ✓)
distortion budget at lb1's bytes        = 0.12 − 0.11990988 = 0.000090122
lb1 actual distortion 0.028120          → needs 312.0× better
```

**This is a SIGN FLIP from the figure I handed the arm.** Memory
`the-demand-has-two-readings-distortion-is-worth-42235-bytes` (MEASURED 2026-08-22) recorded the
zero-distortion gap as **≈150 B OVER** on the then-pointer of 180,368 B. lb1's rate work has since
shed 285 B, and that carried the object across the line: **lb1's rate ALREADY clears sub-0.12 at zero
distortion, by 135 B.** The arm caught this because its charter told it to re-derive against the LIVE
pointer rather than consume the memory's number. That instruction is the only reason the correction
exists; a charter that had said "use 150 B" would have propagated a stale sign all day.

The demand therefore reads, on the live pointer:
- **42,097 B** at current distortion 0.028120, or
- **312× better distortion** at current bytes, or
- any point on the line between, priced at 6.658e-7 S/B.

## 2. The intersection verdict — CONFIRMED at n=3

| body | archive B | vs 137,986 cap | measured distortion (seg+pose) | axis |
|---|---:|---:|---:|---|
| bz2 direct-GT-fit | 100,862 | **−37,124** | **UNMEASURED** | scorer-free byte measurement |
| born-small (qbt2b r10) | 121,928 | −16,058 | 0.327712 | `[macOS-MPS n32-HT advisory]` |
| NR1 K32 | 122,250 | −15,736 | **27.716026** | `[macOS-CPU frozen-scorer advisory]` |
| lb1 pointer | 180,083 | +42,097 | **0.028120** | `[contest-CUDA T4 n600]` authority |

No measured body is both byte-feasible and in the dx2 distortion regime. THE CROSS's two halves are
still held by two different objects. xo1's pre-registered falsifier fired and overshot — recorded
plainly in `ddm_ni1r_nr1_k32_distortion_measured_20260830.md`.

xo1's sharper corollary also holds: born-small and NR1 sit **322 B apart in rate (0.26%) and 84.57×
apart in distortion.** Byte feasibility does not predict distortion among small bodies. There is no
small-body R–D trend to extrapolate along; n=2 spanning two orders of magnitude is not a trend.

## 3. MAIN's own finding — the `byte_feasible` boolean is stronger than the axis label it sits beside

xo1 marks four bodies `byte_feasible: true` with `measured_distortion: "UNMEASURED"`. I opened all
four. **Three of them have no receiver**, so their byte count is a PAYLOAD size, not a contest
archive size:

- **`bz2_direct_fit_bornsmall`** — `archive.zip` (100,862 B, sha `773c7ae3…`) contains exactly ONE
  stored member, `p`, at 100,762 B. There is no `inflate.py`, no runtime dir, and the runtime its
  own fire order names (the bo2/HG1 semantic renderer at
  `…/ddm_tv1_tolerance_curve/runtimes/dx2_shipped`) **no longer exists on disk**. Its own
  `FULL_PACKAGE_RESULT.json` says so honestly: *"native field and bytes close; render/R/uint8/
  frozen-scorer realization is unmeasured."*
- **`rb1_d56_stub` / `rb1_f64_stub`** — the row names say STUB. No trained body exists; the archive
  bytes are a construction estimate.
- `qbz1_fitted_qbt_body` — not separately opened this turn; inherits the same question.

To be fair to the arm: its AXIS labels carry the caveat (`"scorer-free exact byte measurement"`),
and it never claimed these were closed archives. The defect is that a reader consuming the BOOLEAN
column alone gets a stronger claim than the axis supports. Same genus as `#1260` (a retained field
does not carry its own reading semantics) — the column is right, and it is readable wrong.

**Consequence for the census:** the honest count of measured byte-feasible-AND-decodable bodies is
**two** (born-small, NR1), not five. The other three are payload sizes pending a receiver.

## 4. Disposition of the four UNMEASURED bodies

| body | blocker (exact) | owner | disposition |
|---|---|---|---|
| **bz2** | **RENDERER RUNTIME ABSENT** — not storage. Its fire order's trigger conditions are otherwise MET (fcd3 released the scorer lane 22:12Z; archive sha `773c7ae3…` and parseback-field sha `968ffca2…` both re-verified byte-exact by me this turn; AP 11 GiB free vs 3.66 GB needed). What is missing is the decode path. | MAIN | **QUEUED-WITH-FIRE-ORDER, blocker named.** Rank 1 of the four — see §5. |
| **rb1_d56** | **STORAGE, exact shortfall:** trigger requires ≥60,380,026,816 B (56.2 GiB) retained capacity; AP 11 GiB + Vertigo 8.3 GiB = 19.3 GiB → **~36.9 GiB short.** Stub — no archive to score cheaply. | MAIN | **STORAGE-BLOCKED → routes to #1165** (Vertigo reclaim round 2; pk4 cold-move was due 08-27, now past). |
| **rb1_f64** | same as D56 | MAIN | same |
| **qbz1** | receiver status unverified this turn | MAIN | **UNVERIFIED — do not cite as byte-feasible-and-decodable** until its receiver is opened. |

## 5. Why bz2 is rank 1, and what it would decide

bz2 is not merely another census point. It is a **direct GT fit** with native token mismatch
**1.12%** (`FULL_PACKAGE_RESULT.json` `native_full_n.mismatch_fraction = 0.011229510837131076`) —
the best-fitted small body we have, by construction rather than by training luck. And its rate is
0.067160, so its sub-0.12 distortion budget is **0.052840** — the whole target, not half a cross:

- lb1's distortion 0.028120 is INSIDE that budget. If bz2 realized lb1-regime distortion it would
  score ≈ **0.0953**.
- born-small's 0.327712 and NR1's 27.716026 are both far outside it.

So bz2 decides the campaign's live open question in one row: **is small-body distortion a FIT
problem or a RENDER problem?** A near-perfect field that still realizes bad d_seg says RENDER — and
that would kill "build a better small body" as a family by mechanism, not by instance, consistent
with `#1205` (90.47% of dx2's seg error is manufactured) and `#1211` (78.71% appears at the native
render). A near-perfect field that realizes good d_seg says FIT — and bz2 is then a sub-0.12 body.

**Pre-registered prediction (before any measurement exists):** the manufactured-error law predicts
bz2's 1.12% native fit will NOT yield dx2-regime d_seg; the render, not the fit, sets realized
distortion. **FALSIFIER:** bz2's realized seg+pose ≤ 0.052840 ⇒ the fit WAS the problem, the
intersection is NON-EMPTY at n=4, and bz2 is a sub-0.12 candidate outright.

**Cost:** the bo2 predecessor measured 839.5 s / $0 / 3.66 GB inflated raws. bz2 is NOT that cheap
today because the renderer runtime must be rebuilt first — that build is the fire order's real
precondition and it should be priced before it is authorized.

## 6. Successor charter — xo1 wrote none, correctly

xo1 declined to author a successor build charter on the ground that RB1 already owns the distinct
changed-object construction. I adjudicate that REFUSAL CORRECT: writing a second charter for the
same construction would be the duplicate-SoT defect (Catalog #533), and RB1's blocker is storage,
not design. The right next action is the bz2 measurement above plus #1165's reclaim — not a new
object derivation.

**Own-vehicle frontier: lb1 — S 0.14803010583079396 @ 180,083 B [contest-CUDA T4, n600], UNMOVED.**
Sub-0.12 gap 0.028030. Demand, corrected this turn: 42,097 B at current distortion, OR 312× better
distortion at current bytes (lb1's rate already clears 0.12 at zero distortion with 135.3472 B to
spare).

---

## 7. CORRECTION — §3 IS WRONG ABOUT bz2, AND WRONG IN THE EXACT GENUS §3 CITES

Appended by MAIN 2026-08-30, ~1 h after landing, on re-verification. APPEND-ONLY per Catalog
#110/#113 — §3 above is preserved as written and is SUPERSEDED for bz2 by this section.

**What §3 claimed:** *"bz2's `archive.zip` contains exactly ONE stored member, `p` … There is no
inflate.py, no runtime dir"* ⇒ *"three of them have no receiver, so their byte count is a PAYLOAD
size, not a contest archive size"* ⇒ *"honest count of byte-feasible-AND-decodable bodies is TWO."*

**What is actually true, parsed at source** (`ddm_hg1_heterogeneous_analytic_generator_gate.parse_complete_archive`,
the function bz2's own packer round-trip-asserts against):

| section | bytes |
|---|---:|
| generator packet (`bornsmall_gt_fit`) | 47,779 |
| **semantic_renderer** | **30,856** |
| **pose_carrier** | **22,010** |
| compact_residual | 96 |
| container overhead | 121 |
| **total** | **100,862** |

The single member `p` is a **packed multi-section body**, not a blob. And the renderer and carrier are
**byte-identical to the live lineage's own** — sha `39d1be52ba629334…` (30,856 B) and
`932b979f5181b331…` (22,010 B), matching `source_sections/semantic_renderer.bin` and
`source_sections/inherited_pose_carrier.bin` exactly. Those are precisely the ar1b figures (`#1213`:
renderer 30,856 · carrier 22,010).

**How I got it wrong:** I read a `zipfile.infolist()` listing, saw one member, and concluded "no
receiver" — without opening the packer that writes it. That is the **`#1260` genus** (*a retained
field does not carry its own reading semantics*) committed **in the same memo that invokes `#1260`
against xo1's boolean column.** The zip listing gave a right total and a wrong meaning. xo1's
`byte_feasible: true` was correct for bz2 and my caution against it was not.

**What bz2 actually is:** lb1's own renderer + lb1's own pose carrier + a **47,779 B GT-fit
generator** replacing dx2's ~127,007 B {token stream + HPAC model}. A **−79,228 B** rate change,
**1.88× the entire 42,097 B demand**, on an object that keeps the pointer's own realization stack.

**Consequences, stated as predictions and not as measurements:**
- Its rate is REAL: 100,862 B = rate 0.067160, **37,124 B under the cap**, sub-0.12 distortion
  budget **0.052840**.
- Because the pose carrier AND renderer are byte-identical to the pointer's, its **d_pose has a
  reason to sit near lb1's 6.37e-6** — the pose machinery is not swapped, only the field it renders.
  That is a hypothesis from shared construction, NOT a transfer (`m143`), and must be measured.
- Its **d_seg is the genuine unknown**, and the generator's native token mismatch is 1.12%.

**Disposition CHANGED: bz2 is rank-1 and its blocker is SMALLER than §4 stated.** Not "renderer
absent" — the renderer is *inside the archive* and on disk. The measurement path is
parse → `unpack_renderer(sections['semantic_renderer'])` → render → R → uint8 → frozen scorers,
and every piece is named: `ddm_rb1_born_small_receiver.unpack_renderer` / `.render_camera_uint8`
already implement it for the sibling body. Fire-order trigger conditions from bz2's own
`FIRE_ORDER.json` are otherwise MET (scorer lane free; archive sha `773c7ae3…` and parseback field
sha `968ffca2…` both re-verified byte-exact; 3.66 GB needed vs 11 GiB AP free; predecessor cost
839.5 s / $0).

**The pre-registered prediction and falsifier in §5 are UNCHANGED and still bind** — this correction
raises the object's value, it does not pre-judge its distortion.

**§4's rb1 rows are unaffected** (they are genuinely stubs, storage-blocked at 36.9 GiB).
`qbz1` remains UNVERIFIED — and given this correction, "unverified" now means *nobody has parsed it*,
not *it lacks a receiver*.

---

## §8 — THE RENDER PATH IS THE SHIPPING RUNTIME, NOT rb1 (MAIN, 2026-08-30, APPEND-ONLY)

§7 named the measurement path as `ddm_rb1_born_small_receiver.unpack_renderer` → `.render_camera_uint8`.
**That naming is WRONG and is corrected here.** §7's *substantive* claim — that bz2 carries the live
lineage's own renderer and pose carrier — is **CONFIRMED at the strongest available level**; only the
named reader was wrong. Three measured corrections, all re-derived at source:

**1. rb1's `unpack_renderer` REFUSES this stream.** Executed: it raises
`RB1ReceiverError("RB1 WD3 renderer stream is invalid")`. Cause isolated by unwrapping the guard —
`wd3_receiver.packet_allocation` raises `WD3ReceiverError("unsupported WD3 header")` because rb1's
`HEADER` requires `MAGIC = b"WD3Q"`. So §7's "every piece is named ... already implement it for the
sibling body" was a *plausible-looking composition that does not execute*. Counted plainly.

**2. bz2's `SOURCE_ROOT` is bs3, not the live lineage.** `ddm_bz2_bornsmall_capacity_ceiling.py:48`:
`SOURCE_ROOT = /Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/retained`. §7 said the sections
were "byte-identical to the LIVE lineage's own" — read literally off bz2's own constants, that is a
**born-small** path, and the claim would be the #1260 genus a third time.

**3. It is byte-identical ANYWAY — proven by substring, not by size.** The live lb1 archive
(`runtime_joint22_patch192/archive.zip`) has ONE member `p`, 180,092 B. Both bz2 sections occur
**VERBATIM inside it**:

| section | bytes | sha256[:16] | offset in live `p` |
|---|---:|---|---:|
| semantic_renderer | 30,856 | `39d1be52ba629334` | **13,529** |
| pose_carrier | 22,010 | `932b979f5181b331` | **44,385** |

bs3 inherited them from the dx2/lb1 lineage (its own filename says `inherited_pose_carrier.bin`), so
the identity is transitive and real. Size-agreement with ar1b (30,856 / 22,010) was *not* the proof —
the substring hit is. **§7's conclusion stands; its named function does not.**

**THE ACTUAL READER** is the shipping runtime itself, `runtime_joint22_patch192/cpr1/inflate.py`:
`unpack_semantic_pose` → magic test `startswith((b"SD1M", b"SM3R"))`; bz2's blob matches NEITHER, so it
takes the else-branch `SEMANTIC_WIDTH_BY_PAYLOAD_BYTES[semantic_bytes]` → `SemanticTokenRenderer(width)`
→ `unpack_semantic(semantic_blob, template)` → `load_state_dict(strict=True)`. **No brotli step.** My
probe's `brotli.decompress` on this blob "succeeded" and produced an `S3\x01\x01`-headed 36,130 B buffer
that matches no magic in the corpus — that output is off the live path entirely and must not be cited.

**FIRE ORDER (unchanged trigger, corrected mechanism).** Render bz2 via: parse bz2 archive
(`hg1.parse_complete_archive`) → `hg1.decode_packet_to_file` for tokens → **live cpr1**
`SemanticTokenRenderer` + `unpack_semantic` on `sections["semantic_renderer"]` → frames → R → uint8 →
frozen SegNet/PoseNet at n600, `[macOS-CPU frozen-scorer advisory]`. Pre-registered falsifier UNCHANGED:
realized seg+pose ≤ **0.052840** ⇒ intersection NON-EMPTY at n=4 and bz2 is a sub-0.12 candidate outright.

**Own-vehicle frontier: lb1 — S 0.14803010583079396 @ 180,083 B [contest-CUDA T4, n600], UNMOVED.**

## §9 — the render path RESOLVED at source; §8's reader claim WITHDRAWN, the identity claim PROVEN EXACTLY

Two more corrections, and then the path executes. This is the fifth and sixth restatement of one
genus (`#1260`: a retained field does not carry its own reading semantics) inside a single
adjudication, so the genus is the real finding, not any one instance.

### §9.1 — CORRECTION 4: `cpr1/inflate.py::unpack_semantic_pose` is NOT the archive's semantic reader

§8 named it. It is wrong. The real chain, read at source:

`runtime/f26_inflate.py::inflate_archive` → `runtime/residual_archive.py::read_residual_archive`
→ `_decode_rx1_models(outer)` → per-stream **brotli** → CK2/RR5/DX2 riders → then
`renderer.SemanticTokenRenderer(96)` + `renderer.unpack_variant_semantic_or_none(parts.semantic_blob, …)`
(`decode_wans1` fallback) → `load_state_dict(strict=True)`.

`unpack_semantic_pose` **is** called (`f26_inflate.py:453-458`) — but only on a **synthetic** buffer:

```python
semantic_width_marker = bytes(40_252)                      # 40,252 ZERO bytes
semantic_pose = struct.pack("<II", len(semantic_width_marker), len(canonical_carrier)) \
              + semantic_width_marker + canonical_carrier
_, basis, coefficients = renderer.unpack_semantic_pose(semantic_pose)
```

Its only job is to hit `SEMANTIC_WIDTH_BY_PAYLOAD_BYTES[40_252] → 96` and pull basis/coefficients out
of the carrier. **That is why no `<II` header with a mapped size exists anywhere in `p`: the framing
is manufactured at decode time and never stored.** My scan reporting "NO live semantic_pose header
with a mapped size found in the first 60k" was CORRECT; its correct reading is *none by design*, not
*wrong runtime*. I drew the second conclusion from the first observation — the genus again.

Also WITHDRAWN from §8: "there is **no brotli step** on the live path." False. `codec == 2 ==
RX1_CODEC_BROTLI`; every model stream is brotli. rb1's `brotli.decompress` would have **succeeded**;
what failed is the next line, because bz2's renderer is an **SM3R** semantic renderer (cpr1 lineage,
`b'SM3R\x01\x06'`), not a `WD3Q` student packet. rb1 is the wrong receiver for a precise reason.

### §9.2 — CORRECTION 5 (custody): §8's offsets were measured on the BASE archive, not the pointer

§8's substring hits came from `runtime_joint22_patch192/archive.zip` = **180,192 B / `ec0dd68f…`**.
The lb1 pointer is **180,083 B / `5b856e66…`**, which lives in `runtime_candidate_native/`. That tree
is *internally consistent* — its `inflate.py` pins exactly the archive beside it — so this is **not**
the `#1237` pin-mismatch class. I simply read the wrong tree and would have published offsets for the
base while claiming them for the pointer.

**Re-measured on the true pointer bytes, the identity holds at the identical offsets**, and this time
it is proven from the container header rather than from `bytes.find`:

| RX1M field | value | stream offset in pointer `p` | bz2 file | identical |
|---|---:|---:|---|---|
| header | 14 B | 0 | — | — |
| `hpac_bytes` | 13,515 | 14 | *(not in bz2)* | — |
| `semantic_bytes` | **30,856** | **13,529** | `semantic_renderer.bin` | **yes** |
| `carrier_bytes` | **22,010** | **44,385** | `inherited_pose_carrier.bin` | **yes** |
| models block | **66,395** | ends at 66,395 | — | — |

The header's own lengths equal bz2's file sizes exactly, at the offsets the header dictates. And
`13,515 / 30,856 / 22,010` reproduces **ar1b's decomposition** (`#1213`) to the byte — that arm mapped
the 66,591 B residue as renderer 30,856 · carrier 22,010 · HPAC 13,515, and this is the same object
seen through the container.

One loose end, stated rather than papered over: `brotli(semantic_stream)` and `parts.semantic_blob`
are both 36,130 B but **not byte-identical** — a rider inside `_decode_rx1_models` transforms it
(`reserved = 0b11010`). Not load-bearing for the identity claim above, and it is precisely why the
render must go through the runtime's own parser instead of any hand-rolled brotli path.

### §9.3 — what bz2 actually is, exactly

bz2 keeps the pointer's **stored** semantic stream and **stored** carrier stream verbatim, and
replaces `{RX1 header + hpac stream}` **13,529 B** + `token_stream` **113,492 B** = **127,021 B**
with a **47,779 B** GT-fit generator packet: **−79,242 B** on that block, **−79,221 B** at the archive
level (container +21 B). That is **1.88× the entire 42,097 B demand**, on the pointer's own
realization stack. Rate 0.067160 @ 100,862 B, **37,124 B under the sub-0.12 cap**, distortion budget
**0.052840** — wide enough to contain lb1's measured 0.028120.

### §9.4 — #1333 fire order, now fully specified (was blocked on exactly this)

Render bz2 through the runtime's own parser, not a hand-composed path:

1. `sys.path.insert(0, runtime_candidate_native_tree)`; parse bz2's archive with `hg1.parse_complete_archive`
   → `hg1.decode_packet_to_file` for the token field.
2. Splice bz2's `semantic_renderer.bin` / `inherited_pose_carrier.bin` back into an **RX1M container**
   (header 14 B, `codec=2`, `reserved=0b11010`, the measured lengths) so `read_residual_archive` and
   its riders run unmodified — this is what makes the renderer/carrier legs byte-identical **by
   construction**, not by assertion.
3. `_decode_rx1_models` → `SemanticTokenRenderer(96)` + `unpack_variant_semantic_or_none` ·
   `split_frame0_selector_carrier` → `materialize_cpr1` → synthetic `<II` pack → basis/coefficients.
4. `render_video` → exact `R` → uint8 → frozen DALI-lineage SegNet/PoseNet at real n600.
5. Label `[macOS-CPU frozen-scorer advisory]`, `score_claim=false`, `promotable=false`; recompute S
   FROM COMPONENTS (`#877`).

**Pre-registered falsifier UNCHANGED:** realized seg+pose ≤ **0.052840** ⇒ THE CROSS intersection is
non-empty at n=4 and bz2 is a sub-0.12 candidate outright. d_pose has a construction-based reason to
sit near lb1's 6.37e-6 (the pose machinery is not swapped) — a HYPOTHESIS, not a transfer (`m143`);
**d_seg is the genuine unknown** (native token mismatch 1.12%).

### §9.5 — the lesson, sharpened by its own recurrence

Six corrections in one adjudication, one mechanism: **a right total with a wrong meaning.** The zip
listing (§7), the size agreement (§8), the missing header (§9.1), the wrong tree (§9.2) — every one
gave a number that reconciled and a reading that did not. §8's lesson stands and gets a corollary:
*a composition that reads plausibly is not one that executes* — **and neither a retained field nor a
runtime directory announces which object it belongs to. Execute the parser; read the container header.**

**Own-vehicle frontier: lb1 — S 0.14803010583079396 @ 180,083 B [contest-CUDA T4, n600], UNMOVED.**
Sub-0.12 gap 0.028030; demand 42,097 B at current distortion, or 312.0× better distortion at current bytes.
