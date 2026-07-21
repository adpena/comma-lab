# Task #610 — wrong levels describe sweep: receiver-rate custody verdict

UTC: 2026-07-21T22:09:16Z  
Lane: `lane_wrong_levels_describe_sweep_610_20260721`  
Authority: research-only, local `$0`, seed 1234, `[macOS-CPU advisory]`; MAIN landing review required  
Pointer: `0.1910828242 [contest-CPU]` **UNMOVED**

## Outcome first

**No admissible byte win can be claimed on the three requested surfaces from
current custody.** The result is not a performance negative on B-splines, ops
grammars, or chart quantization as families. It is a precise receiver-rate
blocker:

- S1 has exact PNTG description bytes and an existing XIP2 `xi -> H -> RGB`
  realizer, but no pure-knot XIP2 coder ID/format or shipped runtime parser can
  feed spline knots into it. The current shipped path accepts only
  `delta_ar|none`, so a knot curve cannot yet report the requested real
  `d_pose <= 1.02e-4`.
- S2's mandated merged audit ran twice and reproduced byte-identically. All
  five unique-home byte fields remain `null` because no common archive parser
  consumes the v8/v9 per-stratum carrier sections.
- S3's seed is already a counted chart-description object, but it is not camera
  receiver closed. The closest real chart receivers either failed a stricter
  n16 two-axis gate or tested only six selected Lane symbols with zero
  admissions. Neither is the requested corrected-trajectory joint Road/Lane
  seed. That formulation still owes receiver-closed n16 admission; full n64
  and n600 were therefore not authorized.

The binding v9 recursion is preserved: store generators once at their unique
lowest sufficient home, derive boundaries and pixels, reuse one `xi`, and count
only parser-consumed video-derived sections. Summing unrelated diagnostic bytes
would violate that law and the seed/archive/description distinction.

## Primary measured table

`null` is a custody result, not zero.

| surface | exact measured control | candidate stream bytes | real receiver result | projected archive bytes | verdict |
|---|---:|---:|---|---:|---|
| **S1 xi curve knots** | PNTG n64 **797 B**; n600 **6,791 B**; S4 PPCS planar trajectory **11,768 raw B / 902 B standalone zlib diagnostic** | **null** | `d_pose=null`; XIP2 `xi -> H -> RGB` exists, but shipped `delta_ar|none` parsing cannot consume a pure-knot packet | **null** | `BLOCKED_NO_PURE_KNOT_XIP2_RUNTIME_PARSEBACK` |
| **S2 per-stratum ops grammar** | R3 component control **180,196 B**; S4 event alphabet **181,904 B**; literal S2 events **39,836 B** | **null** for Road/Lane/Undrivable/Movable/MyCar unique-home sections | no parser-consumed v9 candidate | **null** | `NO_VERDICT_RECEIVER_RATE_CUSTODY` |
| **S3 chart seed quantization** | PPCS seed **884,872 raw B / 78,969 B standalone zlib diagnostic** | **null** for the requested corrected-xi joint seed | closest full-screw n16 control: **348 B Brotli terminal chart-coefficient packet** (not a seed, receiver payload, or archive), `d_seg=0.016764005`, `d_pose=181.987147`, gate failed; not the requested formulation | **null** | `BLOCKED_BEFORE_N64_CORRECTED_CHART_RECEIVER_ABSENT` |
| **composition** | S4 archive control **451,191 B**; R3 description control **216,207 B** | **null** | no surface admitted | **null** | `THREE_SURFACES_BLOCKED_RECEIVER_RATE_CUSTODY_FAMILIES_OPEN` |

The 216,207-byte description row is `36,011 B` base entropy plus `180,196 B`
admitted components; it is not an archive. The 451,191-byte S4 archive is a
deterministic receiver control with `d_seg=0.60198647` and
`d_pose=163.11865234`; it is not a viable witness. The pre-registered targets
are 216,300 B for the pointer box and 154,600 B for sub-0.15; the R3
measurement receipt independently used a 216,222-byte box and remains labeled
as such in the machine-readable receipt.

## S1 — exact pose-description bytes, no real knot verdict

The frozen n600 cache is hash-pinned at
`cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`.
Canonical PNTG encoding of its `(600,6)` PoseNet targets is 6,791 B
(`ccea4eb9...0014`); the n64 prefix is 797 B (`807ba877...0742`). These are
exact description-space rows, reproduced in memory through the canonical
`fp32 -> fp16 -> zlib9 -> PNTG` layout.

PNTG rows are frozen PoseNet six-vector targets, not translation-first
Lie-algebra `xi`; the byte rows are therefore a target-description control, not
a directly interchangeable knot baseline. A valid comparison must name the
exact `PoseTargetEgoEstimator` or `xi_from_pose_calibration` channel mapping and
calibration, then lift 600 pair twists to 601 absolute poses from a fixed root.

They are not members of S4. S4 instead carries a planar PPCS trajectory with
10 Catmull-Rom controls and 254 AR residual rows. Its receiver applies that
trajectory to the cell field and then emits identical frame 0/frame 1 bytes,
so that archive's measured pose admission is false. Separately, the repository
already has an XIP2 store-nothing receiver that decodes `xi`, derives `H`, and
warps distinct frame 0 before scorer evaluation. Its CLI accepts only
`delta_ar|none`, and shipped `_xip_parse` treats nonzero coder IDs as the
delta-AR layout; it cannot consume a spline/pure-knot coder. Fitting knots and
comparing six decoded scalars alone would still be a proxy. The exact missing
seam is:

`PoseNet targets -> named calibration/channels -> translation-first xi -> fixed-root 601-pose lift -> pure-knot XIP2 coder ID/section -> shipped parseback -> existing H/RGB realizer -> real R -> PoseNet`.

This lane did not reuse `spline_residual` as a claimed win because its lossless
residual prediction overlaps sibling `xi_temporal_delta_coder_574`.

## S2 — per-stratum controls, but unique-home bytes remain null

The exact R3 component control is 180,196 B (`32b41a7d...e68d`). Its packed
class accounting is:

| class | boundary / interior control bytes | total |
|---|---:|---:|
| Road | 829 / 130,736 | **131,565** |
| Lane | 359 / 7,803 | **8,162** |
| Undrivable | 273 / 39,406 | **39,679** |
| Movable | 0 / 0 | **0** |
| MyCar | 0 / 790 | **790** |

Those are literal component controls, not v9 unique-home ops sections. The
mandated `measure_per_stratum_recursive_fractal_optimal.py` from merged commit
`8fa6581f74` was executed twice without modification. The full output is
124,731 B, SHA-256 `1d4b4f6f...d4b5c`; the compact output is 23,414 B,
SHA-256 `68b20a2a...b46c`; consecutive runs were byte-identical.

Its blockers are exact: missing candidate archive, encoder/parser/receiver
modules, n600 scorer receipt, exact evaluator receipt, parser-consumed section
registry, and unique-home byte attribution. The historical 65,172-byte
`0.bin` is a diagnostic, not an archive.

The c2 trained-witness residual signal still determines priority: 90.6% is
edge flicker, with Lane 38.0%, Road 29.2%, Undrivable 9.8%, Movable 9.6%, and
MyCar 4.0%. These are allocation weights, not byte fields. Any future stream
must use semantic self-detection and the built class-specific surfaces, project
onto `range(A)`, use the shared PMF/range backend, and consume corrected `xi`
as external custody. It must not invent another temporal predictor.

## S3 — corrected ground custody blocks the requested chart seed

The PPCS B2 seed is 884,872 raw B, SHA-256 `a21dde38...56b`; its standalone
zlib-9 size is 78,969 B. It is a description object, not a complete archive,
and all 600 factor-2 seed checks fail.

BEV-v2's positive hood control establishes that the transform code executes,
but Road and Lane retain n600 p50 residuals of 39.022618 px and 47.119248 px.
All stored rotations are identity, so calibration-versus-geometry attribution
is unidentifiable. Pure static ground charts are forbidden; a new candidate
must use the corrected absolute trajectory and carry both scorer planes.

Two real-receiver controls were recalled, not re-run:

- The full-screw chart-level formulation produced a 348-byte Brotli terminal
  chart-coefficient packet at n16, not a complete seed, receiver payload, or
  archive. It measured `d_seg=0.01676400495` and `d_pose=181.9871473`. It
  improved pose weakly but increased chart bytes 29.37% and harmed Seg, so the
  pre-registered n16 gate correctly stopped before n64.
- G2CS1 costs 20 B per Lane symbol over a 121,128-byte base. All six selected
  n64-subset candidates were factor-2 exact and double-decode identical, but
  zero were semantically exact or inside the pose tube; n600 was refused. The
  row is not a full n64 result and uses a relative-motion proxy, so it does not
  answer the corrected joint Road/Lane question.

Task #532's 62.74 initial plane error (63.82498 in the later six-pair receipt)
remains a warning, not a chart-family kill. The missing artifact is a
parser-consumed joint Road/Lane chart packet with corrected-trajectory
realization, range/uint8 feasibility, distinct frame 0/1, and receiver-closed
n16 admission first; only then may it advance to full n64.

## Composition and sibling boundary

There is no candidate composition: stream bytes, projected archive bytes,
`d_seg`, and `d_pose` remain `null`. Using
`451191 + compression_ratio * diagnostic_delta` would be fake because none of
the diagnostic packets replaces a named S4 section. A deterministic S2/S4
rebuild is the first point where a projected archive may become an exact
archive row.

No temporal-predictor work was duplicated. If future S1/S2 representation
sections admit, they may consume `xi_temporal_delta_coder_574` as an external
predictor and measure the interaction; no additive savings are assumed here.

## Triality, hygiene, and pointer delta

- **DAG:** `wrong_levels_describe_sweep_610_DAG_FEED_20260721.md` records the
  fail-closed gate ordering.
- **Equations:** `wrong_levels_describe_sweep_610_canonical_equations_20260721.md`
  records consumed laws and explains why a null byte verdict registers no new
  empirical law.
- **DSL:** no new lever. There is no parser/receiver bijection to expose; all
  reused v8/v9 surfaces remain default-OFF.
- **Storage:** fresh output is small JSON under
  `/Volumes/VertigoDataTier/pact/evidence/wrong_levels_610_20260721/`; no raw
  decode, scorer scratch, archive mutation, or cleanup was created. Retired
  controls were harvest-signal-only.
- **Python review:** no Python source was changed. The required existing audit
  tool was consumed exactly from its reviewed merge; no code-review gate was
  bypassed.

Pointer delta: **none**. MAIN must independently review every `null`, verify the
mandatory audit hashes, and confirm the refusal to project archive bytes before
merging.
