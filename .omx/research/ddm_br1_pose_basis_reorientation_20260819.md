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

Net **ΔS = −7.819e-06 from 12 of 600 pairs alone**, at **zero bytes** — already 2.2× the −3.5e-06
admit bar. The n600 run is in flight.

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

## Composability with the live `jg3` fleet

Disjoint archive sections, and a clean division of labour:

- `jg3` edits **tokens** → the semantic stream → **frame 1** (the seg master).
- `br1` edits **carrier coefficients** → **frame 0** (the pose carrier).

d_seg is invariant under `br1` **by construction**: SegNet reads only the odd frame
(`upstream/modules.py:108`) and `br1` writes only even frames.

The coupling runs one way. PoseNet reads *both* frames, so `jg3`'s token edits change frame 1 and
therefore change the pose residual — `jg3`'s own header records that a token-only seg solve costs ~387×
pose and that re-running the carrier descent recovers it. `br1` is exactly that recovery step, done
properly: its deliverable is **the solver plus a coefficient transform**, not a frozen code table. The
splice order is `jg3` tokens first, then `br1` re-solves coefficients against the edited frame 1.

This is why `br1` deliberately did **not** touch the `jg3` store or its checkpoints, and why the
candidate is a re-runnable transform rather than a fixed overlay.

---

## What is owed

1. **n600 GN solve** — in flight (pid 55772, `work/gn_n600/`, resumable, ~4 h). Then re-price and
   byte-close. The n12 net is a subset measurement, not the population number.
2. **Byte-close** through the up3-cured path. up2's two blockers were one missing transform (the carrier
   is 2-plane byte-interleaved, reserved bit `0x04`, un-interleaved at `residual_archive.py:188` before
   any offset read) plus a stale section pin; that cure exists, so this is packaging, not new measurement.
3. **The ceiling** (`mode=ceiling`, built and committed, not yet run): best realized d_pose for an
   *unconstrained* 2304-dof frame 0. This is the number that decides whether the **subspace re-choice**
   of F1 is worth its bytes, and it should be run before any basis-re-choice work is chartered. Ranking
   it after F3 is deliberate — F3 pays now at zero bytes.

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
