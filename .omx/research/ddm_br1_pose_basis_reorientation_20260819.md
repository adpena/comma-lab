# ddm_br1 — pose carrier basis re-orientation: the move is null, the search was the wall

**Date** 2026-08-19 · **Arm** `ddm_br1` · **Module** `experiments/ddm_br1_pose_basis_reorientation.py`
(commit `5369fa827c`) · **Payloads** `/Volumes/APDataStore/pact/ddm_br1/`
**Axis** `[macOS-CPU advisory]`, frozen CPU-torch PoseNet, **DALI-lineage GT** (the shipping axis).
`score_claim=false`, `promotable=false`. Only `upstream/evaluate.py` on contest hardware is a score.

---

## Headline

The charter's move — re-orient the 12 stored basis dimensions — is **provably null**, and I measured
the proof rather than asserting it: re-mixing the basis leaves the reachable pose correction invariant
to **1.9e-08** (machine precision, 24 random pairs).

But the arm did not close the pose family. It found the family's actual wall was **mis-identified**.
`ddm_up2` reported its n600 solve as CONVERGED and concluded "the remaining wall is the 12-dim basis."
That convergence is scoped to **single-coordinate ±1/±2 moves**. The step the residual actually demands
is **57 to 14,079 int12 code units**, multi-coordinate. The search never travelled that far.

Replacing the search with a damped Gauss-Newton step on the same basis, same lattice, same bytes:

| | d_pose ratio | changed coords | ΔB |
|---|---|---|---|
| `up2` ±2 coordinate descent, n600 | 0.98452 | 970 | 0 |
| `br1` Gauss-Newton + ±2 polish, n12 random | **0.85985** | 44 | **0** |

⚠ That first table is `br1`-vs-`up2` on the **to1** body, which `ddm_up3` has since superseded. The
LANDED result below is measured and byte-closed against the **live** pointer `7ce46fd7…`, on top of
up2's own solved codes.

## THE RESULT — n600, byte-closed against the live pointer

| | live pointer `7ce46fd7` | **br1 candidate** |
|---|---|---|
| d_pose (DALI, n600) | 7.649247e-06 | **6.993157e-06** |
| ratio | — | **0.91423** |
| archive bytes | 176,420 | **176,429** (ΔB **+9**) |
| archive sha256 | `7ce46fd7a845d598…` | `44e9e6507d60bf8b…` |

```
ΔS_pose  = -3.834877e-04
ΔS_rate  = +5.992731e-06      (+9 B)
ΔS_seg   =  0                 (byte-identical odd-frame sections, proven below)
--------------------------------------------
NET ΔS   = -3.774950e-04      vs the -3.5e-06 admit bar  =  107.9x
```

**204 of 600 pairs improved, 0 worsened**, 2,323 of 7,200 coefficients changed across 200 pairs.
Advisory projected score **0.15614877** — `[macOS-CPU advisory]`, `score_claim=false`; only
`upstream/evaluate.py` on contest hardware makes it a score.

**Report bounds, both addends, unequal by construction** (the pose leg's sensitivity grows as d_pose
falls, so they must not be summed into one figure): live ±2.858450e-06, candidate ±2.989533e-06. The net
clears their sum by 64.5×, so the move is resolvable by the T4 8dp report rather than lost in it.

The admission sweep priced every level by BUILDING the archive, and the full set won — bytes are not
monotone in pairs admitted, because brotli and the CK2 container search re-optimise per candidate:

| pairs admitted | archive bytes | ΔB | ΔS_pose | ΔS_rate | net ΔS |
|---|---|---|---|---|---|
| 20 | 176,431 | +11 | −1.5105e-04 | +7.324e-06 | −1.4373e-04 |
| 51 | 176,429 | +9 | −2.4880e-04 | +5.993e-06 | −2.4281e-04 |
| 102 | 176,453 | +33 | −3.4108e-04 | +2.197e-05 | −3.1911e-04 |
| 153 | 176,460 | +40 | −3.7497e-04 | +2.663e-05 | −3.4833e-04 |
| **204 (all)** | **176,429** | **+9** | **−3.8349e-04** | **+5.993e-06** | **−3.7749e-04** |

Splice-ready codes: `retained/byte_close_n600/br1_candidate_codes.npy`
Archive: `retained/byte_close_n600/archives/archive_level_0204.zip` (every level retained, not just the
winner).

---

## F1 — the charter's move is null (measured, not argued)

The receiver renders frame 0 as `einsum(coeff, basis)/sqrt(12)` from 12 fields at 24×32×3
(`cpr1/inflate.py:335-352`). Those 12 fields span a 12-dimensional subspace `S` of the **2304**-dimensional
space of band-limited fields. Re-mixing them by any invertible 12×12 matrix yields a different *basis*
for the *same* `S`.

What decides whether the carrier can reach its own residual is the smallest image perturbation inside `S`
that cancels it:

```
minimise ||dX||   subject to   J dX = -r,   dX in S
```

No basis appears in that statement, so no re-mixing can change its value.

**Measured** (`mode=geometry`, 24 seeded-random pairs, `retained/geometry_n24/GEOMETRY.json`):

- rotation-invariance relative change: **max 1.886e-08**, median 7.42e-09
- control — interpolating the 24×32 basis back reproduces the receiver's `basis_norm` to **1.9e-06**

A rotation cannot buy pose at any byte price. **The charter's move is refused, with proof.**

The non-null generalization exists and is *not* a rotation: re-**choosing** the 12 directions inside the
2304-dim space costs the identical 27,648 stored codes (same count, same bit width) and does move the span.
That remains open, and is now correctly ranked **behind** F3 below.

## F2 — instrument correction: `lstsq` is not the minimum-norm solution

Both systems here are underdetermined (6 equations; 12 or 2304 unknowns). `torch.linalg.lstsq` returns
*a* feasible solution, not the minimum-norm one. Minimum **coefficient**-norm is basis-dependent;
minimum **image**-norm is the basis-free quantity that can be read as a span property.

`ddm_up2`'s `basis_conditioning_probe.py` computed both legs with `lstsq`. Measured on 24 random pairs:

| quantity | value |
|---|---|
| span penalty, true minimum-image-norm | **5.29×** |
| span penalty, lstsq method | 5.98× |
| overstatement | **1.151×** |

This does not overturn up2's direction — the span penalty is real and large — but the number it produced
is not the span property it was read as. Guarded by
`src/tac/tests/test_ddm_br1_min_image_norm.py::test_lstsq_is_not_the_min_image_norm_solution`.

**Sample-scope note.** The same measurement on a *prefix* (pairs 0–7) gives 7.55×, and up2's 6-pair probe
gave 6.4×. Pose prefixes measure 2.54–4.21× harder than the population
(`[[prefix_bias_sign_inverts_between_seg_and_pose_20260803]]`), so 5.29× on a random sample is the
population-honest figure. Every `br1` sub-n600 run samples; `sample_pairs` refuses a prefix by construction.

## F3 — the premise correction: the wall was the SEARCH

`ddm_up2_shipping_pose_solve.py:694-782` proposes only single-coordinate moves from `offsets=(-2,-1,1,2)`
and stops when no such neighbour improves. That is a coordinate-wise ±2 local optimum, not a lattice
optimum. Measured demanded steps span **57 to 14,079 code units** against per-pair range headroom of
roughly 700–1500 — so for a real share of pairs the demanded step **fits in the lattice** and the ±2
search simply could not reach it.

`mode=gn` excludes the search as the binding constraint: it re-linearises at the current lattice point,
forms the minimum-image-norm step, and picks the step fraction by **realized** evaluation through the
exact receiver path. The linear model only proposes; the receiver disposes, so a wrong model cannot
corrupt a reported number.

Result on 12 seeded-random pairs (`work/gn_shipped_n12/SUMMARY.json`):

- d_pose mean 4.915666e-06 → **4.226757e-06**, ratio **0.85985**
- Gauss-Newton alone: ratio 0.88071 — most of the gain is the large step, not the polish
- **8 of 12 pairs improved, 0 worsened**
- 44 changed coordinates; 254 s

The single clearest instance is pair 269: **2.92e-06 → 4.58e-07 (ratio 0.157)** on a demanded step of
only **142** code units — comfortably inside the lattice, and 71× beyond what a ±2 move can travel.

## Byte price — measured, and it is zero

The coefficients are zigzag deltas along the pair axis, Rice-coded per dimension
(`cpr1/inflate.py:239-247`), so larger steps are not free by construction the way up2's ±2 steps were.
Priced with `ddm_t1h_carrier_byte_pricer` under its own control (the pricer must re-encode the shipped
payload exactly, or `br1` refuses to price):

| | shipped | br1 n12 candidate |
|---|---|---|
| Rice payload bits | 78,065 | 78,068 |
| Rice payload bytes | 9,759 | **9,759** |
| Rice parameters `k` | `[9,9,9,8,8,9,9,9,9,9,9,9]` | **unchanged** |
| **ΔB** | — | **0** |

The changes are sparse — 8 pairs, 44 of 7,200 coefficients — so they ride free inside the existing
stream. `mode=price` does rate-aware greedy selection anyway, so if a denser n600 solution does cost
bits, the admitted subset is chosen on net score rather than assumed free.

**Net ΔS (n12 subset, DALI axis): −7.819e-06 at ΔB = 0** — 2.23× the −3.5e-06 bar.

### Report bounds

`upstream/evaluate.py:95` prints d_pose at 8 decimals, and the pose leg's sensitivity **grows** as
d_pose falls: `d(leg)/d(d_pose) = 5/sqrt(10·d_pose)`. Bounds add for deltas and are unequal per row, so
both addends are carried, never a single summed figure
(`[[concavity_helps_when_you_pay_the_axis_upward_20260818]]`). At the pointer's d_pose 7.769e-06 the
half-ULP pose bound is **±8.97e-09** per row; the rate leg is exact at ΔB=0. The n12 net clears its own
bound by ~436×.

---

## The ceiling — how much pose is available at all

Before attributing any remaining gap to the basis, price the ceiling. `mode=ceiling` optimises an
**unconstrained** 24×32×3 frame-0 field — 2304 dof, no 12-dim basis, no int12 lattice — through the exact
receiver path (both rounds, the clamp, the selector) against DALI GT.

Measured on the **live** body, seeded-random pairs:

| | d_pose ratio | n |
|---|---|---|
| free 2304-dof field (the ceiling) | **0.7347** | **120** |
| ~~free 2304-dof field~~ (superseded, thin sample) | ~~0.5148~~ | 6 |
| 12-dim basis + int12 + Gauss-Newton, on top of up2 | 0.9448 | 40 (partial) |
| 12-dim basis + int12 + up2's ±2 search | 0.98452 | 600 |

**The n=6 figure was wrong by a wide margin and is withdrawn.** I wrote the "n=6 is thin for pose" caveat
into this memo and then the caveat fired on my own number: 0.5148 → **0.7347** at n=120. Pose's estimate
band is ~13.4× seg's at equal n (`ddm_fo2h`), which is exactly why the re-measure was owed before anything
was chartered on the gap. Only the n=120 row is load-bearing.

Two things follow:

1. **d_pose is NOT fully cancellable.** First order says 6 equations in 2304 unknowns is exactly solvable,
   yet the realized optimum stops around three-quarters. The uint8 rounds, the clamp, the bicubic
   resampling and PoseNet's own nonlinearity leave a large irreducible floor. Any claim that the carrier
   "should" reach d_pose ≈ 0 is refuted. (0.7347 is the best value this optimizer realized, so it bounds
   the achievable ratio from above; the true floor may sit lower.)
2. **The subspace re-choice prize is real but modest — ~21 points, not 29.** Between Gauss-Newton on the
   shipped span (0.9448, partial) and the free-field ceiling (0.7347) sits roughly 21 points of d_pose
   that the 12-dim span and the lattice leave on the table. Against the ~0.0087 pose leg that is worth
   order 1e-3 S if fully realized — but a re-chosen span captures only part of it, and unlike
   Gauss-Newton it is not obviously free (a re-chosen basis re-quantizes 27,648 codes and changes their
   compressibility).

Ordering stands and is now better supported: Gauss-Newton first (it pays now, at zero bytes), subspace
re-choice second and only with its byte price measured up front.

## Baseline correction — the pointer moved mid-arm

Partway through the n600 run I checked `.omx/state/canonical_frontier_pointer.json` rather than trusting
the charter's framing, and found the body had moved under me:

| | body | S | d_pose (DALI, n600) |
|---|---|---|---|
| what up2 solved, what br1 first measured on | to1 `50e56145…` | 0.15659460 | 7.769484e-06 |
| **live pointer** (`ddm_up3` thirteenth move) | **`7ce46fd7…`** | **0.15652626** | **7.649247e-06** |

`ddm_up3` byte-closed up2's solved codes and landed a T4 row. So a `br1` delta measured from to1
**double-counts a gain that is already banked**. The first n600 launch was stopped at 15/600 for exactly
this reason and preserved as `work/gn_n600_STALE_BASELINE_to1/` rather than deleted — it is still valid
as the head-to-head-vs-up2 comparison, just not as a marginal claim.

The cure is structural, not a note: `load_instrument` now takes the runtime as an **argument** pinned by
expected `archive.zip` sha256 and refuses a body it cannot identify, and every mode defaults to
`LIVE_RUNTIME` / `LIVE_ARCHIVE_SHA256`. Pricing checks the same sha before it will price. A future arm
cannot silently inherit a stale body the way this one nearly did.

Reading the odd frames from the to1 decode remains valid and is not a shortcut: carrier edits touch only
even frames, `br1` reads only `raw[2i+1]`, and up3 measured the decoded tokens byte-identical to the
pointer.

Genus: `[[a_delta_without_its_baseline_is_unanchored_and_baselines_move_20260803]]` — and it fired inside
a single working day, on an arm whose charter named the older body as current.

## Composability with the live `jg3` fleet

Disjoint archive sections, and a clean division of labour:

- `jg3` edits **tokens** → the semantic stream → **frame 1** (the seg master).
- `br1` edits **carrier coefficients** → **frame 0** (the pose carrier).

d_seg is invariant under `br1` **by construction AND verified at the bytes**. SegNet reads only the odd
frame (`upstream/modules.py:108`); `br1` writes only even frames. Rather than rest on that argument, the
finished candidate was diffed against the live pointer section by section:

| archive section | identical to live? | bytes |
|---|---|---|
| `token_stream` (odd-frame tokens) | **yes** | 109,696 |
| `semantic_blob` | **yes** | 36,130 |
| `hpac_blob` | **yes** | 17,952 |
| `compressed_models` (the carrier) | no — the only change | 66,528 |

Every section that produces frame 1 is byte-identical, so d_seg cannot move. Parse-back also confirms the
candidate decodes to exactly the intended codes, differing from the live body in 200 pairs / 2,323
coordinates — the solve, and nothing else. (`token_stream` at 109,696 B matches `ddm_jg2`'s own
byte-identical control, an independent cross-check that this is the same object jg3 is editing.)

The coupling runs one way. PoseNet reads *both* frames, so `jg3`'s token edits change frame 1 and
therefore change the pose residual — `jg3`'s own header records that a token-only seg solve costs ~387×
pose and that re-running the carrier descent recovers it. `br1` is exactly that recovery step, done
properly: its deliverable is **the solver plus a coefficient transform**, not a frozen code table. The
splice order is `jg3` tokens first, then `br1` re-solves coefficients against the edited frame 1.

This is why `br1` deliberately did **not** touch the `jg3` store or its checkpoints, and why the
candidate is a re-runnable transform rather than a fixed overlay.

---

## What is owed

1. ~~n600 GN solve~~ — **DONE**, 9,018 s, 600/600, rc=0.
2. ~~Byte-close~~ — **DONE** through `ddm_up3_carrier_splice`, parse-back verified, ΔB +9 B.
3. **A contest-CUDA T4 row on the candidate.** Everything above is `[macOS-CPU advisory]` on the DALI
   instrument, which reproduces the T4 pose row at 0.9999× but is not a score. The candidate archive
   `44e9e650…` is splice-ready and needs the exact-eval fire that only MAIN owns.
4. **Compose with `jg3`.** The two moves are on disjoint sections; the composed candidate must re-run
   this arm's coefficient transform AFTER jg3's token edits land, because those edits change frame 1 and
   therefore the pose residual. That re-run is cheap and the design is re-runnable for exactly this
   reason.
3. **Subspace re-choice** — MEASURED at n=120 as worth ~21 points of d_pose (Gauss-Newton 0.9448 vs the
   free-field ceiling 0.7347). Charter it after the Gauss-Newton row lands, and price its bytes first: a
   re-chosen basis re-quantizes all 27,648 stored codes and changes their brotli compressibility, so
   unlike the coefficient-only move it is NOT free by construction. Its instrument is PCA over the
   per-pair optimal target fields, not over the steps: the new span must hold the carrier the pairs
   actually want, not the correction from where they happen to sit.

## Genus notes

- **A verdict inherited a scoped premise.** "Converged" meant converged *w.r.t. ±2 single-coordinate
  moves*; four downstream statements read it as a lattice optimum. Sister of
  `[[measured_object_vs_named_object_20260816]]` — the measured object was a neighbourhood, the named
  object was the lattice.
- **The falsifier was vacuous at the incumbent.** up2's basis-penalty probe could not have detected a
  search limitation: it measures span geometry, and span geometry is silent about how far a greedy
  ±2 walk travels. Sister of
  `[[the_denominator_and_the_falsifier_can_both_be_vacuous_20260816]]`.
- **My own test caught my own error.** The first run of
  `test_image_step_is_invariant_under_basis_remixing` failed because I wrote the change-of-basis Jacobian
  as `jac @ inv(rot)` instead of `jac @ rot.T`. The module was right; the test was wrong. The comment
  recording that is kept in the test.

## T4 AUTHORITY ROW — FOURTEENTH POINTER MOVE (MAIN, 2026-08-19 ~19:45Z)

Call `fc-01M0DQECXABB3PBMS4REVT5P76` (T4, n600, 1,305.7 s, ~$0.16), archive
`44e9e6507d60bf8b6429ce066983aa814b23f2f929869aa5a10a8b8dacda5c7d` (176,429 B), seal
`dab3ed9420649d0ee162322213402d8d3a85969eaeb3f12b2c90f2c3b89f15bf` (SEAL_VALID, receiver
re-pinned via tac.candidate_seal.repin_receiver at staging):

- **S = 0.15615242950573233** (recomputed from components; printed 0.16 is display rounding)
- d_seg **0.00030309 — UNCHANGED** (the byte-identity proof held on the shipping axis)
- pose report 7e-06 (advisory realized 6.993157e-06, ratio 0.91423 vs up3)
- rate 176,429 B (+9 vs up3)
- **Net vs up3 pointer 0.15652626435208142: −3.7383e-4 ADMITTED** — advisory projected
  −3.7750e-4; realization error +3.67e-6, inside the summed 8dp report bounds. 107× the
  −3.5e-6 admit bar. Gap to 0.15: 0.00652626 → **0.00615243**.

Pointer + effective_frontier updated (refresh_canonical_frontier --update-local); lane
`lane_ddm_br1_gn_pose_resolve_t4_20260819` closed `completed_harvested_pointer_moved`.

### Fire-chain incident (three refusals, now #1152)

The T4 fire was refused twice with "dispatch produced no spawn record" (rc=5). The real
cause — visible only in a full-capture reproduction — was the entrypoint's pairing gate:
single-axis runs require `--single-axis-waiver-reason`, which the fire calls omitted; the
firer echoes only the last 2000 chars of dispatch output and modal's mount tree filled the
tail, censoring the one-line refusal. Silent-instrument family. Two-landing filed as task
#1152: (a) stage-0 must REQUIRE the waiver when the seal's axis is single, (b) FIRE_REFUSED
must carry the entrypoint's refusal line verbatim, (c) M1 class-population sweep of other
seal-derivable requirements. Side lesson: MAIN's manual lane pre-claim was also wrong —
the canonical chain owns claiming end-to-end (hand-assembled-dispatch class).
