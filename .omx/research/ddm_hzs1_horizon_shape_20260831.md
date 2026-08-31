# ddm_hzs1 horizon shape — the 37.47× attribution is not causal; fixed-paint horizon shape remains 5.041× at the +2 KB ceiling

`axis: [macOS-CPU scorer-free exact count + real generic coders]` · `n=600` ·
`score_claim: false` · `promotable: false` · `verdict_scope: FORMULATION — one Road/Undrivable threshold per column with GF1's later streams fixed`

The pointer did not move. The charter's 893,436-pixel “horizon” mass is a true-class attribution,
not a horizon-shape causal set. Only **142,739** positions are exposed Road↔Undrivable swaps that
the first-painted horizon can change; **750,697 (84.02%)** are Lane/Movable/MyCar paint laid down
after the horizon. The best of the five measured rows within `packet_delta <= +2,000 B` is an
adaptive 17-knot curve: **1,309,255 mismatches, 47,739 B, 428,601.280 projected B = 5.041×** the
85,020 B bar. The k=2 horizon+lane corner is not reachable by horizon shape with downstream paint
fixed, even if Lane is granted perfect rather than merely the required 0.0178% error.

## 1. RESIDUAL CHARACTERIZATION FIRST

The mandatory n600 control passed before the candidate loop:

```text
[macOS-CPU scorer-free exact count]
GF1 generated vs lb1 field:     1,325,033 == 1,325,033  PASS
GF1 generated vs DALI GT field: 1,324,976 == 1,324,976  PASS
GF1 four-stream accounting packet: 47,603 B             PASS
47,603 + 0.2909 * 1,325,033 = 433,055.0997 B = 5.093567×
```

The full 5×5 confusion matrix separates ownership from actuation:

| True Road/Undrivable mismatch source | Positions | Share of 893,436 | Horizon shape can change it? |
|---|---:|---:|---|
| Road predicted Undrivable | 109,881 | 12.30% | yes |
| Undrivable predicted Road | 32,858 | 3.68% | yes |
| Later Lane paint | 160,131 | 17.92% | no |
| Later Movable paint | 491,841 | 55.05% | no |
| Later MyCar paint | 98,725 | 11.05% | no |
| **Total** | **893,436** | **100.00%** | **142,739 yes / 750,697 no** |

This closes exactly. HG1 renders `horizon -> lane -> movable -> mycar`; changing the horizon cannot
erase a later writer. The prior 173,155 B “perfect horizon” number removes all true Road and
Undrivable misses, silently including those later writers. The causal perfect-horizon arithmetic
is instead:

```text
remaining mismatch = 1,325,033 - 142,739 = 1,182,294
47,603 + 0.2909 * 1,182,294 = 391,532.3246 B = 4.605180×
```

The 142,739 causal swaps are geometrically horizon-like but not well summarized by one global
regime:

- Rows 176–207 contain **120,963 (84.74%)** of the exposed swaps; the 176–191 and 192–207 bands
  contain 68,893 and 52,070 respectively. There is essentially no exposed mass outside rows
  160–255.
- Across 60-pair scene blocks, exposed swaps range from **10,190 to 18,160 (1.782×)**. In contrast,
  the much larger true-class attribution ranges from 47,500 to 163,457 because downstream paint,
  not causal shape, carries most of its regime variation.
- The attributed mass is broad across x: the largest 32-column exposed bands are x=320–351
  (16,879), x=416–447 (14,560), and x=288–319 (14,507); the top 20% of individual columns hold
  only 37.33% of attributed mass.
- Against the exact visible-bulk-loss threshold, the retained 17-knot curve is already close:
  **1.476 px RMS / 0.485 px mean absolute**. Removing per-frame shift leaves 1.437 px RMS; affine,
  quadratic, and cubic fits leave 1.415, 1.394, and 1.370 px. A 60-pair scene profile leaves
  1.361 px. Scene-only and low-order fits therefore explain little of this residual.

The current wire was also corrected in scope: it is not a shift-only horizon. `HGH1` stores 17
x knots and 600×17 row values (10,200 row scalars; 20,447 raw B), then linearly interpolates per
frame. Only a vertical shift of that existing curve had been swept previously.

## 2. PARAMETERIZATION CHOSEN FROM THAT RESIDUAL

The measured residual selected a same-wire objective correction, a shared adaptive-knot control,
two interpolation controls, and a uniform density curve through the full 512-column single-graph
oracle. The objective rows minimize exact Road/Undrivable 0–1 error only where the horizon remains
visible after GF1's fixed overlays; ties stay closest to the retained curve. Thus the fit optimizes
the causal quantity rather than the old “topmost Road” proxy.

All byte claims are real races among Brotli q11/lgwin24, zlib 9, and LZMA2/XZ extreme. Every coded
payload and deterministic repeat is retained and decompresses exactly to its raw stream. “Packet”
below means the same **GF1-equivalent four-generator accounting packet** as the 47,603 B control.

| # | Parameterization | Video-derived scalar DOF | Horizon raw / coded B | Packet B (delta) | Mismatch lb1 / DALI | Causal R↔U left | Projected B | /85,020 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | retained 17 linear control | 10,200 | 20,447 / 4,536 | 47,603 (+0) | 1,325,033 / 1,324,976 | 142,739 | 433,055.100 | 5.094× |
| 2 | objective 17 linear | 10,200 | 20,447 / 4,580 | 47,647 (+44) | 1,323,675 / 1,323,614 | 141,381 | 432,704.058 | 5.089× |
| 3 | adaptive 17 linear | 10,217 | 20,447 / 4,672 | 47,739 (+136) | 1,309,255 / 1,309,137 | 126,961 | **428,601.280** | **5.041×** |
| 4 | objective 17 nearest | 10,200 | 20,448 / 4,580 | 47,647 (+44) | 1,487,376 / 1,487,290 | 305,082 | 480,324.678 | 5.650× |
| 5 | objective 17 PCHIP | 10,200 | 20,448 / 4,580 | 47,647 (+44) | 1,326,875 / 1,326,794 | 144,581 | 433,634.938 | 5.100× |
| 6 | objective 33 linear | 19,800 | 39,680 / 7,552 | 50,619 (+3,016) | 1,267,466 / 1,267,407 | 85,172 | 419,324.859 | 4.932× |
| 7 | objective 65 linear | 39,000 | 78,144 / 11,996 | 55,063 (+7,460) | 1,244,563 / 1,244,548 | 62,269 | 417,106.377 | 4.906× |
| 8 | objective 129 linear | 77,400 | 155,072 / 18,280 | 61,347 (+13,744) | 1,221,515 / 1,221,511 | 39,221 | **416,685.714** | **4.901×** |
| 9 | objective 257 linear | 154,200 | 308,928 / 26,324 | 69,391 (+21,788) | 1,202,135 / 1,202,251 | 19,841 | 419,092.071 | 4.929× |
| 10 | objective 512 linear | 307,200 | 615,438 / 35,552 | 78,619 (+31,016) | 1,183,149 / 1,183,484 | **855** | 422,797.044 | 4.973× |

Denominator: **10 parameterizations enumerated after characterization / 10 measured / 0 missing**.
Of these, five satisfy the charter's `<= +2,000 B` packet constraint. The honest measured ceiling
inside that constraint and this declared family is row 3: it removes 15,778 causal swaps (11.05%)
and leaves 1,309,255 total mismatches. It does not meet the charter's proposed “below ~10% of the
gap” falsifier; attributed horizon residual remains 877,658, or 66.24% of the original full gap.

The full 512-column row establishes the single-graph distortion ceiling: it removes 99.40% of the
causal swaps, leaving 855, but immutable downstream intrusion leaves 750,697 true-horizon misses.
Its rate makes it worse than the 129-knot optimum. This ceiling is scoped to one threshold per
column with fixed overlays; it is not a global theorem about multi-interval semantic masks or a
jointly refit generator.

## 3. K=2 REACHABILITY — NOT REACHABLE

The charter's optimistic true-class accounting leaves 113,191 mismatches after perfect Horizon
and perfect Lane. If `H` is remaining attributed horizon error and `delta_B` is packet growth, the
strict bar requires:

```text
H < (85,020 - 47,603 - delta_B) / 0.2909 - 113,191

delta_B =     0: H < 15,433.96
delta_B =   136: H < 14,966.44
delta_B = 2,000: H <  8,558.74
```

The +2 KB condition is a 99.04% reduction from 893,436, much stronger than “below ~10% of the
gap.” The best eligible row leaves H=877,658. More decisively, horizon-only work cannot lower H
below the **750,697 later-stream intrusion floor**, regardless of curve shape.

Granting the best eligible row a literally perfect Lane stream—stronger than the required but
unmeasured 0.0178% Lane error—still gives:

```text
47,739 + 0.2909 * (1,309,255 - 318,406) = 335,976.974 B = 3.952×
```

Even granting the baseline a perfect causal horizon, all 318,406 true-Lane misses, and all 160,498
predicted-Lane false positives leaves 703,390 mismatches and **2.967×** the bar. Therefore the
80,530 B attribution oracle is not a realizable horizon+lane claim under the current paint model:
it also gifts away Movable/MyCar intrusions. No Lane 0.0178% row was measured here, no assumption
of one is promoted, and no scorer fire is warranted from these token-field results.

## 4. RECALL EVIDENCE

The charter seeds were treated as a floor. Exact corpus queries included:

```text
rg -l -i 'horizon' .omx/research --glob '*.md' --glob '*.json' --glob '*.jsonl'
rg -l -i 'road[_ -]boundary|boundary[_ -]road' .omx/research ...
rg -l -i 'per[_ -]column|columnwise|column-wise' .omx/research ...
rg -l -i 'curvature|curve|spline|polynomial|polyfit|bezier' .omx/research ...
rg -l -i 'scene[_ -]condition|scene regime|regime[_ -]condition|tilt' .omx/research ...
rg -n -i 'horizon|road.?boundary|per.?column|polyline|curvature|scene.?condition|tilt|ddm_gf1|ddm_hzs1' .omx/state/canonical_task_status.jsonl
.venv/bin/python tools/list_canonical_equations.py --json
```

Beyond the charter's seeds:

- `v8_increment1_design_draft_20260709.md:58-74` already measured a dominant n600 horizon as a
  degree-3 curve plus per-frame intercept, median 1.46 px over 425/512 columns. The corrected
  full-n600 real-zlib price is 4,167 B (`ddm_fc3_prereg_triage_and_residue_20260821.md:130-168`).
  Therefore a cubic top arc is a control, not HZS1 novelty.
- `t5_crucible3/SPEC_v8.1_20260709.md:414-459` measured hard degree-2/3 lateral envelopes as
  negative (0.100403 to 0.119845/0.119060, about 20 px fit residual) and curve-relative residual
  coding at only 0.99×/0.90×. Those exact formulations were not re-run.
- `frozen_partition_topology_ego_deformation_20260623.md:99-171` found effective curve rank about
  4.07 but ego variables explained only R² about 0.23–0.38. `road_horizon_joint_containment_20260627T074201Z.md:24-58,108-149`
  found a two-float half-plane 2.6× worse than a static mask because Road/Undrivable was not a
  clean half-plane. These ruled out ego/tilt or one low-order arc as the sole candidate.
- `ddm_rd2_hg1_rate_distortion_curve_20260824.md:111-170` found coherent residual orderings cheaper
  than pixel order, but its oracle-tile selection is not receiver-shippable and was not credited.
- `.omx/state/canonical_task_status.jsonl:566` already assigns task 1181 the prefix-derived
  Road↔Undrivable price cure; no duplicate task was created.
- The canonical equations registry returned related V8/lane/rate analogues, but no equation
  validates a shape model on GF1's current residual. In the scoped GF1/V8/HG1 horizon corpus, no
  measured real-coded scene-conditioned mixture exists.

This recall changed the plan: the instrument first corrected the causal object, kept GF1's 17-knot
wire as the positive control, skipped cubic/ego/scene-only rediscovery, and spent the curve on the
exact visible Road/Undrivable loss plus an explicit density ceiling.

## 5. RECEIPTS, PARSE BOUNDARY, AND CUSTODY

Command:

```text
.venv/bin/python -B experiments/ddm_hzs1_horizon_shape.py --phase measure --out /Volumes/APDataStore/pact/ddm_hzs1_horizon_shape --resume-from /Volumes/APDataStore/pact/ddm_hzs1_horizon_shape
```

Durable root: `/Volumes/APDataStore/pact/ddm_hzs1_horizon_shape/` (1.6 GB retained). Primary facts:

- `CHARACTERIZATION.json`: 0fff0c142db217ed7517b8b4a99c0d6aa3c07e31a32b8e829d1e077c213da590
- `MEASUREMENT.json`: 66d834725b4ca8fdbe4bc3bd00840bb66d7176765def37a1840d438e98251bbf
- `RUN_PROVENANCE_measure.json`: 36277b76d85efbb8da9060b0c26d6cfdbe4526b0ad78f627f93105aee0661e19
- `STAGE_measure_COMPLETE.json`: 8be251bc3583e6bd0e57d3765a2ea854062a33f98d49dc24c1ca0cc065a77d88
- Measurement executable content SHA-256:
  `840038891e47a89e241a4bdea7ada4edf0a1a57867e05277aa3014a1cad2a1c8`.

Each candidate directory retains its horizon raw bytes before render, the three real coder payloads
and deterministic repeats for every stream, the exact four-stream accounting packet, the rendered
117,964,800-byte field, both packed mismatch fields, `RESULT.json`, and a complete checkpoint.
Fixed Lane/Movable/MyCar coder receipts after candidate 1 are byte-identical copies of the fully
re-encoded control receipts; `coder_race` then re-verifies equality and raw decompression. Candidate
horizon streams are independently double-encoded.

The packet boundary matters: GF1's retained packet has four generator streams. The shipped
`hg1.parse_packet` requires five streams including residual and rejects it. HZS1 therefore calls
every 4-stream byte result a **GF1-equivalent accounting packet**, never receiver-valid. Rows 1–3
use the existing `HGH1` parser/renderer. Other rows use the retained `HZS1` prototype parser/renderer
in this instrument and have exact packet parse-back, but are not claimed as shipped receiver rows.

Independent review re-read all 10 packets, all 120 coder payload/repeat pairs (240 files), all 20 packed masks,
the winner selection and frontier formulas; it also confirmed the shipped parser rejects the
4-stream control. No local scorer or Modal job ran, `upstream/` remained read-only, no lb1 runtime or
shipped bytes changed, and token mismatch is not presented as d_seg or an exact contest score.

## 6. VERDICT AND FOLLOW-ON DISPOSITION

The prior-law prediction survives, but for a stronger reason than parameter sharpness: the
37.47× row combined true-class ownership with causal responsibility. **REFUSE** fixed-downstream,
single-threshold horizon shape as a GF1 reactivation route at `<= +2,000 B`; the best row is 5.041×,
and even the high-rate causal oracle remains 4.973×. This does not close a joint generator that owns
later false-positive support.

- **FOLDED** — owner: MAIN; consumer: the existing GF1 reactivation frontier in
  `.omx/state/main_hot_state.md`; fire trigger: a successor explicitly owns downstream
  Lane/Movable/MyCar false-positive support and first demonstrates
  `packet_B + 0.2909*mismatches < 85,020` on retained n600 lb1/DALI fields. Only then should MAIN
  decide whether to spend the scorer lane.
- **FOLDED-EXISTING-TASK** — owner: MAIN; consumer: task 1181 in
  `.omx/state/canonical_task_status.jsonl`; fire trigger: before any prefix-derived
  Road↔Undrivable coder price is cited as n600 authority, run that task's full-n600 or seeded-random
  cure. HZS1 creates no duplicate ticket.

`[contest-CUDA T4, n600] own-vehicle frontier: LB1 — S=0.14803010583079396 @ 180,083 B; HZS1 did not move the pointer.`
