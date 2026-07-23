# Codex findings — DDM M4 rate floor / Einstein Avenue

UTC: 2026-07-23
Lane: `ddm_m4_rate_floor_einstein_avenue`
Authority: bounded receipt re-derivation only; `$0`; no launch; no new scorer run;
`research_only=true`; `score_claim=false`; pointer unchanged; MAIN landing review required.

## Decisive result

There are three different “floors,” and collapsing them would manufacture certainty:

| Quantity | Bytes | Epistemic status |
|---|---:|---|
| Universal model-independent MDL lower bound | **0** | DERIVED; only sound nonnegative bound because no complete normalized description model or incompressibility certificate exists |
| Smallest n600 receiver-closed archive in the explicit relaxed-box audit (`d_seg<=0.00116`, `d_pose<=0.00161`) | **177,169** | MEASURED prior exact receiver/evaluator row; archive bytes and SHA reverified this lane; **audited-set minimum, not global optimum** |
| Smallest receiver-closed archive in the explicit exact-C1 audit (`d_seg<=0.00015196`, `d_pose<=0.00010184`) | **409,526,925** | MEASURED prior historical `[contest-CPU]`; **audited-set minimum, not global optimum** |

The 177,169-byte row is the real answer to the delegated relaxed receiver box:
`d_seg=0.0005453067355847452`, `d_pose=0.00002930838566754801`, n600,
receiver-closed, exact archive SHA
`cb6cf0ba719a535bf8874b31675a4ec66a893423d320f1e4071a2012cd88a56f`.
It carries a legacy dispatch-provenance gap (mounted code commit and upstream snapshot SHA were
not recorded), so it remains a MAIN-review candidate rather than a new claim from this lane.
The canonical pointer independently has the same 177,169-byte rate scale.

The exact settled-C1 score law is

`S(B) = 100*0.00015196 + sqrt(10*0.00010184) + 25*B/37545489`.

Its strict sub-0.15 continuous crossing is
`154524.647411717548... B`, so the last legal integer byte count is **154,524 B**:

- `S(154524) = 0.1499995689151115...`
- `S(154525) = 0.1500002347740646...`
- 177,169 B at exact C1 distortion would score `0.1650779449085631...`
- required cut from the smallest relaxed-box receiver row: **22,645 B = 12.7815814%**
- exact fixed-C1 pointer knee: **216,223 B**; the relaxed-box row is 39,054 B below it
- #604's 216,207 B is a description-only row, not a receiver-closed archive

**Reachability verdict:** `SUB015_NOT_REACHED_22645_BYTE_GAP_NOT_RULED_OUT`.
There is no current <=154,524-byte receiver-closed n600 row inside the box, but the only universal
lower bound is zero bytes, so sub-0.15 is not honestly ruled out.

## Audited lower-byte controls

The lower-byte alternatives do not displace 177,169 B:

| Row | Bytes | d_seg | d_pose | Why rejected |
|---|---:|---:|---:|---|
| DDM v19b modern low-byte receiver | 137,825 | 0.026594424778 | 163.061176604795 | receiver-closed but misses the relaxed bounds by 22.93x and 101,280x |
| Q-axis int6 n600 | 147,513 | 0.002384 | 0.000294 | d_seg outside box; advisory |
| Q-axis int7 n600 | 174,061 | 0.001537 | 0.000222 | d_seg outside box; advisory |
| Q-axis int8 n600 | 177,169 | 0.000594 | 0.000037 | inside box; same byte floor |
| #575 exact same-container row | 177,169 | 0.000545306736 | 0.000029308386 | inside box; selected floor row |

The 175,801-byte int7 MSE row is n48 only and therefore cannot enter the n600 floor.

## Rule 118 — FREE / NULL / COUNTED

| Disposition | Contents | Rate consequence |
|---|---|---:|
| FREE | generic parser/integrity logic; generic xi integrator; generic power-diagram/chart rasterizer; deterministic seed-to-table expansion; generic receiver/codec interpreter | 0 B |
| NULL | omitted gauge coordinates and deterministic dither with no receiver-visible statistic | 0 B when omitted; not a serialization channel |
| COUNTED | learned/video-fit weights, latents, coefficients, initial conditions; video-derived xi/pose state; chart/topology/event/exception/residual symbols; video-derived palette/camera constants or lookup tables even when hidden in code | exact archive bytes |

#604's content audit found an empty `move-to-counted` bucket and classified its runtime as generic.
The measured compliant reclassification reduction is therefore **0 B**, not 17,864 source bytes,
5,382 compressed counterfactual bytes, or the runtime's full source size. Those counterfactuals
were explicitly rejected by rule 118. No pointer-archive reclassification is inferred from the
S4 runtime audit.

## ker(A) free hiding

The exact resize geometry has domain dimension 1,017,336/channel, rank 196,608/channel, and
nullity 820,728/channel = **80.6742315223%**. The bounded signed-integer primitive basis has a
34.1931390993% reachability lower bound on the measured fixture.

That is geometric freedom, not byte freedom. The measured full-kernel heuristic admitted zero
frames and cut **0 B** versus the existing zero-weight-mask control. The current exact member
representation already serializes range(A), not ker(A), and no parser-consumed counted payload was
removed. Therefore:

- measured counted bytes hideable-for-free: **0 B**
- one-frame raw-coder diagnostic: the already-settled old zero-weight mask was `-183,778 B`
  Brotli versus the original, but this is not a complete archive delta and the new 80.67% kernel
  added **0 B** beyond that control
- `80.67% * archive_bytes` is forbidden arithmetic
- a pose carrier cannot be placed in ker(A) and still affect PoseNet; a video-derived carrier also
  remains COUNTED even if its rendered perturbation is scorer-invisible

## Non-additive pool partition

| Pool | Levers | Relation | Scoped law |
|---|---|---|---|
| `P_REALIZE` | multicoefficient-solve; correction-synergy | **COMPETE** | same realized receiver field; only strict sequential/joint replay gets credit |
| `P_TEMPORAL_DESCRIPTION` | context-arithmetic-code; xi-once-for-pose; chart-canonicalization | **COMPETE** | same post-chart temporal-symbol redundancy; #574 xi-temporal added +7,020 B / +8,508 B after canonicalization |
| `P_FRAME_OWNERSHIP` | frame-separation | **ORTHOGONAL for d_seg only** | SegNet reads frame 1, so frame 0 is seg-free; Pose still couples the pair and byte deltas need joint replay |
| `P_NULL_GAUGE` | ker(A)-hide | **ORTHOGONAL geometry, not rate** | scorer-null placement is distortion-free; measured byte credit remains 0 B |

The v19b strict joint replay proves why singleton addition is invalid:
single-step gain total `0.0374212702`, survived `0.0359100668`, amplified
`0.0804967212`, degraded `0.0015112034`, survival fraction `0.9596164586`.
Across pools, “orthogonal” means a structural constraint decomposition; final byte/score credit is
still admitted only after one same-artifact receiver replay.

## Integer-lattice-native check

The current g2g2 multicoefficient solver is **not lattice-native**. It searches fp32 coefficient
deltas with bounded projected greedy/coordinate methods, then requires
`factor2_uint8_exact` at the receiver gate. Exact validation after projection is not the same as
optimizing over the integer receiver lattice.

The measured n16 absolute-write formulation scheduled `+0.01583 S` recovery but realized
**-1.4%**:

- flips: 10,002 -> 10,009
- realized score delta: `-0.00022162 S`
- scheduled recovery left unmet: `0.01583 - (-0.00022162) = 0.01605162 S`

`0.01605162 S` is an **unrecovered scheduled debt**, not a measured recoverable gain. The result is
scoped to the n16 source-closest-sign absolute-write formulation; it does not authorize n600
extrapolation.

## Re-derivation

```bash
PYTHONPATH=src uv run --frozen --with scipy python \
  tools/derive_ddm_m4_rate_floor.py \
  --verify-receipt \
  .omx/research/ddm_m4_rate_floor_einstein_avenue_20260723_receipt.json
```

Expected: `status=PASS`, `rate_floor_bytes=177169`, `sub015_gap_bytes=22645`.

## STORES CONSULTED

- Delegated authority prompt, stranded inverse-solve spec, and live inbox through the final
  checkpoint.
- `CLAUDE.md` rule 118, non-additive-pool, quantization, and exact-axis contracts.
- #604 Einstein-Kolmogorov memo at commit `5cc81f1172`; consumed, not duplicated.
- #575 exact same-container row and exact SSD archive bytes.
- Q-axis n600 int8/int7/int6 response surface.
- Historical exact-C1 inverse-solve receipt.
- #580 full-kernel receipt; #602 MDL member receipt; #574 xi-temporal receipt.
- DDM v19b strict joint-remeasurement receipt and the m3 frame-separation handoff.

## MAIN landing review

MAIN must review the branch diff and explicitly confirm: (1) the 177,169-byte relaxed-box
audited minimum is not relabeled a global MDL optimum; (2) the exact-C1 409,526,925-byte row is not
conflated with the relaxed acceptance box; (3) the 22,645-byte sub-0.15 gap is the decision number;
(4) FREE reclassification and ker(A) each receive exactly 0 measured byte credit; and (5) the
lattice debt remains n16 formulation-scoped.
