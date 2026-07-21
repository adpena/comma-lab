# Per-stratum recursive-fractal optimal treatment — measured custody verdict

UTC: 2026-07-21T19:12:17Z
Lane: `lane_per_stratum_recursive_fractal_optimal_20260721`
Authority: research-only, local `$0`, `[macOS-CPU advisory]`; MAIN landing review required
Pointer: `0.1910828242 [contest-CPU]` **UNMOVED**

## Outcome first

**NO_VERDICT_RECEIVER_RATE_CUSTODY.** The settled v8/v9 treatments cannot be
composed or byte-measured from this checkout: the #503 `DecisionCarrierBundle`
encoder, parser, live receiver, and alternate archive do not exist in the current
tree or Git object history. The #503 receipt itself records
`candidate_archive_present=false`, `full_n600_measured=false`, and
`exact_evaluator_called=false`. Treating its 65,172-byte current-receiver diagnostic
or its zero-group deltas as a composed v9 archive would be fake.

Two extant archives were re-audited as controls. M1 is below the 154,600-byte cap but
does not reproduce the full scorer cells; S4 has deterministic n600 receiver closure
but is 2.9184× the cap and has catastrophic scorer error. Neither closes the requested
richness-preserving describe line.

The Road/Lane calibration split is also **unidentifiable from current custody**.
BEV-v2 persists no independently observed pixel homography. Its settled `s_r=0`
makes every calibrated twist rotation and every absolute-pose rotation exactly zero;
`pitch_rad=-0.05` is carried as metadata but `xi_from_pose_calibration` does not use
pitch to construct `R`. Decomposing a homography generated from that same
translation-only pose would be circular, not a measurement of missing pitch/yaw.

## Exact byte-closed controls

| row | archive bytes | cap ratio | receiver/decode custody | hard n600 result | richness verdict |
|---|---:|---:|---|---|---|
| M1/C2 control | **90,566** | **0.5858085×** | archive `a386a854…edf8c`; real n600 decode; chunked 38-batch hard CPU-torch oracle; seed 1234 | `d_seg=0.0035157945421` exact rational; `d_pose=127.3658829`; 414,740/117,964,800 label mismatches | **FAIL current instance**; not full argmax/Pose closure |
| S4 composition control | **451,191** | **2.9184411×** | archive `d84f2fe0…1696ed`; repo/standalone n64+n600 byte identity; n600 stream `01f45813…83f8` | `d_seg=0.60198647`; `d_pose=163.11865234` | **FAIL current instance**; same-frame realization and no frame-0 carrier |
| #503 diagnostic | 65,172-byte `0.bin` only | 0.4215524× of cap if it were an archive, which it is not | no alternate encoder, payload, parse-back, or archive | full n600 not run | **NO_VERDICT**; not an admissible byte row |
| requested v9 composition | **null** | **null** | required payload/receiver absent | required n600 scorer row absent | **NO_VERDICT_RECEIVER_RATE_CUSTODY** |

All byte counts above are exact integer file/ZIP counts. The cap arithmetic is:

\[
25(154600)/37545489 = 0.10294179415268769,
\]

so a 154,600-byte archive has only
`0.15 - rate = 0.04705820584731231` score units left for
`100*d_seg + sqrt(10*d_pose)`. M1's low byte count cannot compensate for its Pose
term; S4 already exceeds the entire score target on rate alone.

## Road/Lane ground-frame cell: what is and is not measured

### Settled BEV-v2 measurements recalled

| scale | gate/stratum | p50 residual | p90 residual | fraction at ≤1 px | status |
|---|---|---:|---:|---:|---|
| n64 | bottom-connected MyCar hood | 0 px | 0 px | 0.922991 | **MEASURED positive control** |
| n600 | bottom-connected MyCar hood | 0 px | 0 px | 0.913057 | **MEASURED positive control** |
| n600 | Road ground signature | 39.022618 px | 180.509541 px | 0.043093 | **MEASURED negative for exact G1 translation-only chart** |
| n600 | Lane ground signature | 47.119248 px | 186.702513 px | 0.043713 | **MEASURED negative for exact G1 translation-only chart** |

The new stage audit reads all 600 n600 stages. It measures maximum
`|R-I|=0`, zero non-identity absolute rotations, maximum twist rotation magnitude
zero, and zero nonzero rotation transitions. (The stored absolute poses do contain
translation.) Therefore the input has no rotation signal for an `R`-versus-`t`
decomposition to identify.

`CalibratedGeometry` is exercised only as a custody canary with explicit scorer-space
`K=(fx=400.3, fy=399.5, cx=256, cy=192)`: identity `H` must decompose to identity
`R`, zero `t`, and zero pose. Its defaults are intentionally not used: settled #326
found they mix native `fx=910, pp=(582,437)` with a 512×384 working shape and emit
rotation-first `[omega,t]`, while the live `tac.lie` carrier uses translation-first
`[rho,omega]`.

### OpenPilot sensitivity bound, not causal attribution

Using the already-reconciled `v_h=174`, `h=1.22 m`, and scorer `K`, the OpenPilot
vanishing-point equations are

\[
u_{vp}=c_x+f_x\tan(yaw),\qquad
v_{vp}=c_y-f_y\tan(pitch)/\cos(yaw).
\]

The nominal horizon corresponds to pitch `0.04502587 rad` (2.57979°). OpenPilot's
calibration spread thresholds (pitch 4°, yaw 2°) induce at most about 28.11 px vertical
and 13.98 px horizontal VP displacement; the corner norm is 31.39 px. By comparison,
the Road p50 has one-axis horizon equivalents of at least 5.543° pitch or 5.568° yaw,
and Lane at least 6.678° pitch or 6.713° yaw. The broader validity windows can span
the residuals, but validity is not an error posterior.

These are **DERIVED sensitivity bounds**, not explained fractions. Both
`calibration_explained_fraction` and `genuine_geometry_fraction` remain `null`.
An exact split needs independently observed correspondences/flow or custodied
OpenPilot `cameraOdometry`/`liveCalibration`, scorer-resolution `K`, convention-safe
adaptation, and held-out `R`-only/`t`-only counterfactuals. Semantic boundaries alone
are not persistent point identities.

### OpenPilot Lane DOF correction

OpenPilot v0.9.7/current does **not** emit polynomial lane lines. It emits four lane
lines and two road edges as 33 sampled `(x,y,z)` points on fixed forward samples, with
probability/std metadata. Only the path surface uses a quartic polynomial helper.
The repo's cubic/polynomial `LaneLine` is a compression abstraction fit to the SegNet
lane mask, not an OpenPilot-native DOF claim. #327 nevertheless settled the relevant
repo abstraction: `v_h=174` beats 188/192 on the measured n600 lane-band error, and
changing camera height 1.2→1.22 has zero band-error effect.

## Per-stratum recursive-fractal treatment table

The treatments below are **SETTLED-RECALL** from SPEC_v8/#503, not re-derived. c2 is
the measured 0.01328 palette-vehicle residual taxonomy. M1 columns are the exact
n600 hard-oracle output control. “Bytes” is deliberately null until a receiver parses
a unique-home section for that stratum.

| stratum | D1 frame | D2 basis | D3 temporal | D4 quantization / lambda | D5 boundary recursion | c2 residual debt | M1 mismatch / d_seg contribution | measured unique-home bytes | current verdict |
|---|---|---|---|---|---|---:|---:|---:|---|
| Road | calibration-corrected ground/BEV **conditional on independent R custody** | shared Road+Undrivable bulk field; spend on separatrices | one `se(3)` ξ plus receiver-proven correction | highest flip-mass priority; chart coefficients; shallow side per pair | localized tangent atoms on Road–Lane/MyCar/Undriv annuli; rank-4/Fisher margin DOF | 0.004684 edge + 0.001586 near = **0.006270** | 138,575 / **0.0011747148** | **null** | ground transform unresolved; family open |
| Lane | ground curve chart, with the repo analytic band treated as a compression model | analytic curve/band + dash grammar; not raster and not mislabelled as OpenPilot polynomial | ξ transport + dash phase + sparse curvature/fork corrections | thin-class high precision; Lane shallow on Road–Lane; coefficient quantization only after uint8 survival | along-tangent 4–8 px boundary atoms; ragged edge remains explicit | **0.001098** edge | 149,028 / **0.0012633260** | **null** | SPEC_v8's 1–2 KB is a prior estimate, not composed bytes |
| Undrivable | image/horizon chart plus shared ground boundary where adjacent to Road | low-frequency region plus shared Road boundary field | slow-drift horizon knots / shared ξ transport | low interior precision; spend on Road-side shallow horizon edge | horizon annulus shares one edge packet with Road; no double charge | **0.000585** edge+near | 52,630 / **0.0004461500** | **null** | near-free interior is recalled, exact edge bytes absent |
| Movable | per-object frame | sparse islands / object cells with boundary-defined identity | per-object tracks plus ξ; measured far persistence 0.865 | both sides fragile on Movable pairs; per-object coefficient waterfill | object-border tangent atoms and sparse topology/birth/death correction | 0.003222 far + 0.001101 near + 0.000614 edge = **0.004937** | 52,403 / **0.0004442257** | **null** | largest dynamic carrier; no aggregate parser/receiver |
| MyCar | ego-image frame | static bottom-connected hood mask plus rim | store once; sparse rim correction | lowest post-seed debt; image-coordinate quantization | one static rim boundary with localized correction | **0.000140** edge | 22,104 / **0.0001873779** | **null** | SPEC_v8's 0.1–0.5 KB is a prior estimate, not composed bytes |

Inter-class saddles add c2 `0.000100` (0.75%) and must live on their incident edge
packets, not become a sixth duplicated class carrier. The table's five M1 rows sum
exactly to 414,740 mismatch pixels and the exact rational n600 `d_seg`.

## Nine-dimensional composition custody

| #503 dimension | settled optimal form | exact composed bytes | custody result |
|---|---|---:|---|
| pixel | DecisionCarrierBundle generator/tie/ξ/Pose6/chroma state | null | module/encoder/parser absent |
| class | v8 five edge-centric carriers; merge→diff→correct | null | individual helpers exist; no common parser-consumed archive grammar |
| boundary | smooth interior + localized curvelet/shearlet annulus PoU | null | no literal receiver-boundary PoU byte A/B; Fourier remains governed control |
| frame | keyframe + deterministic warp + sparse correction | null | packet concepts exist; no #503 real-frame receiver section |
| pair | one ξ, Pose6 tangent only if needed | null | #359/#425 label-space byte rows are not through-R v9 closure |
| epoch | stage-boundary metric anneal | null | training schedule, not a measured shipped section |
| chroma | decision palette + luma-null sparse correction | null | bounded n1 secants only; full n600 consume absent |
| scale | multiresolution PoU with unique lowest home | null | derived law; encoder absent |
| frequency | smooth interior + localized 4–8 px tangent atoms | null | no equal-byte receiver-closed selection |

The recursive composition law—partition of unity, generators not sampled
boundaries, unique home, RGB only at the scorer boundary, stage-boundary annealing—is
still the correct design contract. This audit closes only the premise that current
bytes already instantiate it: they do not.

## Richness, scope, and actionable blocker

- **Full argmax exactness is required.** M1's 414,740 mismatches and S4's much larger
  error prevent either control from standing in for the requested exact partition.
- **Pose closure is jointly required.** A good Seg-only row with catastrophic Pose is
  not a witness-equivalent archive.
- The Road/Lane result kills only the exact G1 translation-only chart. It does not kill
  independently calibrated ego/ground transforms, analytic bands, or localized edge
  carriers.
- The smallest honest next gate is not another design memo: materialize one actual
  #503 archive whose standalone receiver consumes every unique-home section, then run
  n64 admission and the existing chunked n600 hard oracle. Independently, Road/Lane
  needs observed-H/live-calibration custody before causal R/t attribution.

## Triality and stores consulted

- **DSL:** no new lever. Existing #503/v8 contracts remain default-OFF; missing
  parser/consumer bijection is preserved as a blocker.
- **DAG:** `per_stratum_recursive_fractal_optimal_DAG_FEED_20260721.md` records the
  fail-closed gate ordering.
- **Equations:** consumes `argmax_native_vjp_fidelity_v1`,
  `perclass_stratum_residual_carrier_taxonomy_v1`, rank-4 Laguerre head closure,
  unique-home/PoU laws, and the exact score action. No new empirical equation is
  registered from a null byte verdict.
- **STORES CONSULTED:** CLAUDE.md; AGENTS.md; PROGRAM.md; v7.5 §8; SPEC_v8; #503
  build spec and receipt; c2 taxonomy/equation; #325/#326/#327 OpenPilot audits;
  BEV-v2 n64/n600 SSD receipts and 600 stages; M1 archive/exact harness/chunked
  decomposition; S4 archive/standalone parity/hard scorer receipts; U1 and closed
  scorer findings; live lane/subagent registries and both delegation inboxes.

## Pointer delta

None. This is a measured custody/blocker artifact, not a score or promotion claim.
MAIN must independently review the code, exact input hashes, null-byte handling, and
the refusal to infer a calibration split before merging this branch.

## Reproduction receipt

The read-only production command was run twice after the evidence-binding pass and
produced byte-identical outputs:

- full SSD receipt: 124,744 bytes, SHA-256
  `bbab41d92c0cc05a88a0c107f04d4b9112b24ca63a39d329dfaa660549a9af2c`;
- compact repository receipt: 23,427 bytes, SHA-256
  `b45c80c06d146b92808cf50ad1d6b98e4be699e4d8ee14472dc36c88a1becb49`.

The receipt directly binds the M1 counted-archive build, two independent exact
decodes, all 38 preserved hard-oracle checkpoints, and the final decomposition. It
also binds the S4 archive build, n600 repo/standalone parity checkpoint, direct
advisory-eval checkpoint, and aggregate measurement receipt. The volatile observed
free-space quantity is deliberately omitted after enforcing the SSD threshold, so a
change in unrelated disk use cannot mutate the scientific receipt.
