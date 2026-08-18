# ddm_sa2 — the pose-COMPENSATED semantic edit: sa1's reactivation criterion, executed

Axis: **[macOS-CPU advisory frozen CPU-torch PoseNet + exact byte/container]** —
`score_claim=false`, `promotion_eligible=false`. No Modal, no GPU, no lane claim, no
scorer-lane fire. The n600 advisory adjudication is MAIN's, against the bought base leg.

## STAGE-0 VERDICT — FEASIBLE, and by a wide margin

The sa1 family verdict closed lossy **uncompensated** semantic-FiLM edits 3/3 and recorded
two reactivation criteria. Criterion #1 — the qs5 in-compile Schur compensation pointed at a
semantic-tensor edit — is **not just feasible, it clears the bar by ~180×**.

| quantity | required | measured (n600, all pairs) |
|---|---:|---:|
| cancellation of the S2 pose damage (d_pose units) | ≥ **99.34%** | **99.942%** |
| mean remaining per-pair d_pose damage | ≤ 5.40e-6 | **4.89e-7** |
| compensation byte cost | ≤ ~1,069 B | **+36 B** |

A byte-closed candidate is built, staged and sealed: **179,851 B**
(sha `6e5d8bcf804e0e2f…`), **−1,310 B** against the rr4 base, parse-back PASS.
Its same-instrument row projects **net ΔS = −6.3663e-4** against a bar of −3.5e-6.

**The charter's falsifier threshold was wrong and is corrected here.** It named
"residual d_pose above ~2.2e-4 ⇒ INFEASIBLE". The exact ceiling derived from the admit
arithmetic is **d_pose ≤ 1.5299e-4** at S2's bytes (1.5287e-4 at the built candidate's),
not 2.2e-4 — a 2.2e-4 residual would land ΔS_pose ≈ +8.5e-3, twelve times over budget.
Both numbers in the charter's own sentence ("cancel ≈98.8% of the pose damage") are right,
but they are in **different units**: 98.83% in S-units ≡ 99.34% in d_pose-units. The
gauge's units are part of the claim.

**The projection is barely a projection.** With all 600 pairs solved, this arm's frozen CPU
PoseNet reproduces the bought advisory base leg to five significant figures on both anchors
— base d_pose 1.474661e-4 vs 1.4747e-4 (ratio 0.99997) and uncompensated 9.865438e-4 vs
9.8653e-4 (ratio 1.000014). The only unmeasured terms in the candidate's row are the ones
MAIN's harness supplies: d_seg (invariant by construction, §1) and the harness's own
end-to-end reproduction.

## 1. The structural fact the sa1 memo did not have

`cpr1/inflate.py::render_video` writes the two frames of every pair from **disjoint
sources**:

```python
output[2 * (start + offset) + 1] = master_np[offset]   # frame_1 = semantic renderer
...
output[2 * (start + offset)]     = slave_np[offset]    # frame_0 = carrier ONLY
```

`frame_0 = round(bicubic(round(127.5 + 64·Σ_k c[p,k]·basis_k / √12)))`. It contains **no
semantic-renderer content whatsoever**. So:

* the S2 quantization damages **frame_1 only** — verified, not assumed: frame_0 is
  byte-identical between the base and S2 receiver outputs on every checked pair;
* frame_0 is a clean **12-DOF-per-pair signed-int12 actuator** — exactly the qs5 geometry;
* **d_seg is invariant under the compensation BY CONSTRUCTION**: SegNet reads
  `x[:, -1, ...]` (`upstream/modules.py:109`) = frame_1, which the compensation never
  touches. The candidate carries S2's measured d_seg, not a hoped-for one.

sa1 §9.3 said "semantic edits move *both* (the renderer paints both frames)". That is
correct for the **semantic** edit and is why S2's d_seg moved at all — but it hid the
asymmetry that makes the compensation possible: the renderer paints only frame_1, and the
carrier paints only frame_0.

## 2. Instrument controls — all passed before any number was believed

1. **Receiver exactness.** My frame-0 renderer reproduces the shipped receiver's `0.raw`
   frame_0 with **0 mismatched values** on every control pair (24 seeded-random). The first
   attempt was wrong by ~97% of values — the missing basis mean/RMS normalization
   (`cpr1/inflate.py::normalized_basis`). Had I trusted it, every Jacobian would have been
   garbage.
2. **Frame-1-only damage.** `raw_base[2p] == raw_s2[2p]` on every control pair; frame_1 RGB
   rms 1.26–2.41, matching sa1's 1.634 renderer-field screen.
3. **Rate-model control.** My Rice cost model reproduces the shipped coefficient stream's
   79,020 bits **exactly**.
4. **Encoder byte-identity.** Re-encoding the **unchanged** lattice through
   CPR1 → CAP1 → packed-metadata → brotli reproduces the shipped carrier body
   **byte-for-byte**. This control caught a real bug: the archive stores CAP1 fields in
   `STORED_CAP_FIELDS` order (scales before predictor) while the CAP1 blob uses
   `CAP_FIELDS` order. Before the fix my "rate measurement" read +68 B of pure
   field-order damage as if it were physics.

## 3. The solve — qs5's machinery, re-solved in-compile against the edited object

Per pair, on the exact receiver realization at full resolution:

* `base_vector = P6(f0(c), f1_base)`, `event_vector = P6(f0(c), f1_S2)`; `leak = event − base`;
* exact receiver-realized central-difference Jacobian `∂P6/∂c` (6×12) at code steps 64 then 8;
* damped least squares → integer proposal → **multi-scale integer descent** (steps
  64/16/4/1), each scale stopping on one complete non-improving pass;
* objective = `mean((P6(f0(c'), f1_S2) − base_vector)²)` — restoring the base pose vector
  restores base d_pose exactly.

The `JACOBIAN_STEPS` change is the one place I did not copy qs5: a ±1 difference is right for
qs5's ~1e-9 leakages but is **below this actuator's quantization floor** — one int12 unit
moves frame_0 by ~0.006 grey levels, far under the `round()` quantum. Measured reach:
step 1 → 0.71% of pixels move; step 64 → 32%; step 256 → 73% and 1.62 rms, i.e. comparable
to the frame_1 damage itself. The actuator has ample authority; only the probe step needed
to match the regime.

### Measured, n = 600 seeded-random pairs (never a prefix)

| quantity | value |
|---|---:|
| subset mean uncompensated damage | 8.3908e-04 |
| n600 measured damage (S2 − base) | 8.3906e-4 |
| **subset / n600 damage ratio (representativeness control)** | **1.0000** |
| subset mean remaining damage | 4.8923e-07 |
| mean cancellation of d_pose damage | **99.942%** |
| median energy cancellation | 99.961% |
| median / max code move `|Δc|∞` | 20 / 130 |
| pairs ending **better** than base on pose | 236 of 600 |

The representativeness control matters more than the headline: the subset reproduces the
independently-measured n600 damage to within 0.0%, so the subset mean is a credible
estimator of the population mean and not a lucky draw.

## 4. Rate — the compensation is FREE, and the route matters ~100×

Two counted representations exist for a frame-0 code change. Both were priced on real
`archive.zip` bytes, never an entropy estimate:

| route | cost for a 600-pair, ~11-of-12-dimension compensation |
|---|---:|
| sparse overlay section (`Q2C1` caps at 15 pairs; `P1D1` reaches 600) | **~7,000 B** |
| **re-encode the carrier lattice itself** | **−19 … +49 B** |

The lattice wins because the coefficients are *already charged* as Rice-coded temporal
deltas with k=8/9: a perturbation of tens of units rides inside the existing quotient
buckets and changes almost no bits. Measured on real archives: iid ±20 over all 7,200
coordinates costs **−0.7 B**; ±100 costs **+29 B**.

Better still, folding the shipped 7-pair `Q2C1` overlay into the lattice and dropping the
overlay section is a **net −13 B**: the zero-compensation control archive is **179,815 B**
against S2's 179,828 B, parse-back PASS.

**One runtime change is required and it is free.** The shipped reader dispatches the carrier
on two pinned lengths (`PACKED_CAP1_SECTION_BYTES = 22,183` / `CANONICAL = 22,223`), so any
re-encoded lattice — whose Rice residual stream has a different length — is refused. The
patch derives the packed portion's length from its own u24 bit counts instead. That is pure
framing arithmetic carrying no video-derived content, so it is rule-118 clean and costs zero
counted bytes (inflate.py is unsized); qs1 set the precedent with
`_patch_variable_carrier_runtime`. The patched runtime ships inside the candidate generation.

## 5. What was built

**Candidate** — `179,851 B`, sha256 `6e5d8bcf804e0e2f…`,
staged at `/Volumes/APDataStore/pact/ddm_sa1/generations/sa2_compensated_S2/` with its patched runtime and a re-pinned
`inflate.py`. Parse-back through that runtime in a fresh interpreter: **PASS**, decoded
600x12 lattice equal to the solved lattice with max abs deviation 0.

| leg | value |
|---|---:|
| base (rr4) archive | 181,161 B |
| S2 uncompensated archive | 179,828 B |
| zero-compensation control (overlay folded in) | 179,815 B |
| **compensated candidate** | **179,851 B** |
| compensation marginal cost vs control | **+36 B** |
| Δbytes vs base | **-1,310 B** |
| coordinates changed vs the shipped lattice | 6,228 of 7,200 |

Same-instrument admit arithmetic against the bought base leg:

| term | ΔS |
|---|---:|
| rate (-1,310 B) | -8.722752e-04 |
| seg (carried from the S2 row; invariant under a frame-0 edit) | +1.720000e-04 |
| pose budget remaining before the bar | +6.967752e-04 |
| **d_pose ceiling the candidate must clear** | **1.528700e-04** |
| projected compensated d_pose | 1.479592e-04 |
| projected pose ΔS | +6.364546e-05 |
| **projected net ΔS** | **-6.366298e-04** (bar: < −3.5e-6) |

Direct same-instrument read on this arm's own PoseNet (all 600 pairs, no projection):
base d_pose 1.474661e-04 → compensated
1.479554e-04, against uncompensated
9.865438e-04. This arm's instrument reads the bought leg's base within **0.003%** and the
uncompensated row within **0.001%**, so these are the same instrument in every way that
matters; MAIN's harness row is still the one that adjudicates.

Compile-time bindings that fail closed:

* `build` refuses unless the semantic section being packed has the sha256 the compensation
  was solved against — **the qs4 lesson in code**, not in prose;
* parse-back runs in a **fresh interpreter against the runtime tree that ships with the
  candidate**, and asserts the decoded 600×12 lattice equals the solved lattice exactly and
  that the semantic/hpac/token/residual sections are unchanged;
* `pack_cap1_metadata` refuses any predictor/bias/Rice-k value outside the packed domain
  rather than silently emitting a section the receiver would misread.

## 6. Honest limits — what this arm does NOT establish

1. **No score.** Nothing here is a scorer row. Every d_pose number is this arm's own frozen
   CPU PoseNet on rendered pairs, not a harness run; MAIN's n600 advisory run against the
   bought base leg is the measured row, and only `upstream/evaluate.py` on contest hardware
   is an authority.
2. **The compensation is fitted on the CPU instrument, and that is the real risk.** The
   base leg's own receipt shows CPU-vs-T4 pose drift of **21.4×** on identical bytes. The
   compensation was solved against the frozen CPU-torch PoseNet, so the same-instrument
   advisory delta is honest, but **transfer to T4 is unmeasured**. The mechanism is
   physical (a real frame_0 image change nulling a real frame_1-induced pose shift, cancelled
   to 99.9% in output space), which is why I expect it to transfer — but expectation is not
   measurement, and a T4 row is the only thing that settles it.
3. **d_seg is asserted structurally, not re-measured.** The argument (SegNet reads frame_1;
   the compensation touches only frame_0) is exact, but the candidate's d_seg has not been
   run. MAIN's row measures it.
4. **Scope reduction retired.** The charter permitted a subset; the solve ended up covering
   **600 of 600** pairs, so the pose term is a full-population mean, not an extrapolation.
   Every rendered object, PoseNet call, and descent step is the exact shipping receiver
   realization at full resolution.
5. **Full-frame inflate was not run.** Parse-back is verified at the carrier/semantic
   section level through the shipping receiver, in a fresh interpreter; the token stream is
   byte-identical to base, so the ~25-minute token decode path is unchanged by construction.
   MAIN's advisory run exercises the full path.
6. **Human visual fidelity of frame_0 degrades, and that is fine here.** The compensation
   moves frame_0 away from its base appearance by design. frame_0 was never a photograph —
   it is a synthetic 127.5±10-grey-level carrier field — and the contest scores only
   SegNet/PoseNet/bytes. No fidelity claim is lost because none was being made.
7. **The candidate is not the frontier.** The exact pointer is unmoved by this arm. A
   −6.37e-4 advisory projection is roughly 7.5% of the 8.53e-3 gap to sub-0.15 — worth
   firing, nowhere near the goal.
8. **Three of 600 pairs finished at an int12 boundary.** The descent clamps, so those pairs
   may be actuator-limited rather than converged. Their contribution is inside the measured
   mean; it is named because a boundary hit is where a "the actuator has ample authority"
   claim would first break if the damage were larger.

## 7. What this reopens

The sa1 FAMILY verdict stands as written — it scoped itself to **uncompensated** edits, and
it named this exact escape. What changes is the price of the whole sa1 ladder: every one of
the 16 byte-closed rows sa1 built was refused on pose damage of 68–512× its rate credit, and
that damage is now shown to be **~99.9% cancellable at ~zero bytes**. The natural successor
is not this one candidate but the ladder: `sm3r_keep01` carries a −2,889 B credit
(2.2× S2's) and was refused at +0.157; compensated, its admit condition becomes a question
about whether the frame-0 actuator can cancel a **26×** pose collapse rather than a 6.7× one.
That is a different and much harder ask, and it should be measured, not assumed — the
actuator's authority is finite and keep01's damage is 26× larger.

## STORES CONSULTED

`ddm_sa1_advisory_adjudication_20260818.md` (family verdict + reactivation criteria) ·
`ddm_sa1_semantic_carrier_representation_attack_20260817.md` (the 16-row ladder, §9.3) ·
`ddm_qs5_verdict_and_no_toy_enforcement_20260813.md` + `ddm_qs5_resolve_compensation_20260813.md`
(the proven in-compile compensation; the qs4 cross-regime-transfer disaster) ·
`ddm_qs1_frame0_schur_coupled_solve_20260813.md` (the Schur solver, the ±1 Jacobian, the
variable-carrier runtime patch precedent) · `ddm_qs2_compensation_rate_rung_20260813.md` ·
live source: rr4 `candidate_runtime/{cpr1/inflate.py, runtime/f26_inflate.py,
runtime/residual_archive.py, runtime/compensation_overlay.py, cpr1/carrier_codec.py}`,
`upstream/{modules.py, evaluate.py, frame_utils.py}`,
`experiments/ddm_ps1u_carrier_delta_codec.py` (the P1D1 600-pair overlay — priced and
rejected as ~100× more expensive than the lattice route), the pr135 experiment book's
`coefficient_ar1_codec` / `coefficient_predictor`.

## RETAINED

* solve: `/Volumes/APDataStore/pact/ddm_sa1/retained/sa2/n600/` — per-pair `RESULT.json`,
  base/final codes, every evaluated code row and its PoseNet-6 vector, GT pose vectors,
  final frame_0, `AGGREGATE.json`
* build: `/Volumes/APDataStore/pact/ddm_sa1/retained/sa2/build/` — control and candidate
  archives, `CONTROL.json`, `CANDIDATE.json`
* generation: `/Volumes/APDataStore/pact/ddm_sa1/generations/sa2_compensated_S2/`
* sealed order: `/Volumes/APDataStore/pact/ddm_sa1/FIRE_ORDER_sa2.json`

GT camera frames are a deterministic decode of a read-only upstream input and are certified
rebuildable (receipt per shard records the source sha256 and the rebuild entry point); the
derived measurement — the GT PoseNet-6 vector — is retained per pair.

## NEXT_IF_RESUMED

* `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN. Fire the n600 advisory leg on the staged
  generation against the bought rr4 base leg
  (`advisory_n600_cpu/rr4_base/attempt_0002/contest_auth_eval.json`), recompute the
  same-instrument decomposition, and admit only on measured net ΔS < −3.5e-6.
* If the advisory row admits, the T4 question in §6.2 is the next one worth buying: it is
  the only thing that tests whether a CPU-fitted compensation survives the 21.4× instrument
  drift.
* The compensated **ladder** (§7), starting with the highest-credit rows, is the real prize
  and is unowned.
