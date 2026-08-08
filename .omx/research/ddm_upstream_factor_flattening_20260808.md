# The flattened upstream factor graph — every surface that plays into S (operator 08-08: "items I never named explicitly but expected you to find")

Tags: [no-triality] [p0-ledger-ok]. Status per node: **MEASURED** (receipt exists) ·
**FOUND-UNCONSUMED** (analyzed, no shipping-path consumer — the authority-poor class) ·
**OPEN** (never measured — honest debt). Consumers: ddm_lx1 mandatory coverage · #984
arithmetic sheet · the vehicle-argmax check.

S = 100·d_seg + sqrt(10·d_pose) + 25·|archive.zip|/rate_denominator

## RATE term
| node | status | receipt / debt |
|---|---|---|
| archive.zip stat().st_size only — inflate.py/sh NOT sized | MEASURED | evaluate.py:63; rule-118 boundary |
| rate DENOMINATOR is DYNAMIC — rglob('*') over videos/, not the 37,545,489 constant | MEASURED | us1, Catalog #812 guard; W = 4·denom/117,964,800 derived |
| ZIP container micro-overhead (member headers, central dir, name lengths, method flags) | MEASURED | #79 packaging audit |
| double-compression interplay (coded payload under ZIP deflate can ADD bytes; store-mode) | FOUND-UNCONSUMED | #79-era; re-audit at composed byte-close |
| 30-min budget = WHOLE CI job → decode-side free-compute headroom | MEASURED | #835; funds decode-side prediction |

## SEG term (chain: archive → inflate env → uint8 frames → D → SegNet → argmax vs GT-argmax)
| node | status | receipt / debt |
|---|---|---|
| D operator exact semantics: bilinear antialias=False, stride 2.276 → DISJOINT 2×2 sampling, 4 private camera px/scorer px | MEASURED | m86 |
| 22.70% of camera px blind to BOTH scorers · 80.67% resize-nullity DOF · ~52% range(A)-complement | MEASURED | #839 four canonical names, #580 projector |
| SegNet reads LAST frame only → frame_0 structurally seg-free (obligation 8.5e-9) | MEASURED | Unit C; the frame-role asymmetry DOF |
| uint8 quantum = the actuator lattice (±0.5; breaks float nullity exactness Δ=62.74) | MEASURED | #532 |
| BATCH SHAPE is part of the forward instrument (oneDNN FP order flips tie-adjacent argmax, batch-1 vs batch-16) | MEASURED | et4 pair-17 |
| argmax tie semantics — ties measured 0.00000%, diagram well-defined | MEASURED | mf1 F1b |
| eval-mode BN running stats = fixed per-channel affine operators → capacity codebook | FOUND-UNCONSUMED | #725 built, consumer unbound → lx1 |
| GT-label FLICKER floors (GT's own frame-to-frame instability = per-class distortion floor) | MEASURED | fl1; the tolerance-shaping floors |
| per-class structure: canonical comma10k order · err ∝ area^−1.26 · ONE graph w/ Road hub (87.8% of flips) | MEASURED | class-order law, ddm_pc2/m91 |
| GT video decode lineage (yuv420_to_rgb law; PyAV rgb24 = phantom pose ~100×) | MEASURED | GT decode law |
| pair blocking: 600 non-overlapping; hardness 79× spread across 60-pair blocks; prefix bias seg 0.96× / pose 2.5–4.2× / rate ~neutral | MEASURED | m88/m96/na4 |
| CPU-vs-CUDA GT gap mechanism (DALI/NVDEC vs PyAV chroma siting) | **OPEN** | #906 — needs CUDA+DALI; owner needed before any CUDA-axis claim |
| kernel/thread determinism of the authority forward (code,weights,threads,batch pinned) | MEASURED | et4 pinning law; MPS 23×/2× drift laws |

## POSE term
| node | status | receipt / debt |
|---|---|---|
| d_pose is RELATIVE — target = PoseNet(decoded ORIGINAL pair), not physical pose; true-GT frame_0 scores 3.05–16.66 | MEASURED | m87 |
| BOTH scorers share the SAME D (interpolate to segnet size FIRST, then yuv6) — no different-resize arguments exist | MEASURED | ddm_pz1 (corrected our own docs) |
| yuv6 basis: 4 subsampled luma + 2 chroma → chroma feeds pose; warp-resampling attenuates null fields only 1.662× | MEASURED | pz1 |
| sqrt(10·d_pose) nonlinearity → marginal value flips at pose_avg ~2.5e-4 (operating-point law) | MEASURED | CLAUDE.md operating-point table |
| PoseNet hydra head outputs 12 dims, only [:6] scored → 6 UNUSED output dims (free slack in the head) | FOUND-UNCONSUMED | noted early, never exploited; lx1 row owed |
| pose-null frame_1 subspace: seg-reachable; float-null but NOT exact under integer actuator | MEASURED | #837/sq1 |
| Q3 placement law: seg spend in the pose-null cone cannot damage pose (exact kernel) | MEASURED | #889 sharpened |

## RUNTIME / ENV ("everything upstream uses")
| node | status | receipt / debt |
|---|---|---|
| dependency closure: uv.lock +3 drift, 2 lost symlinks, torch pin, PyAV version, brotli precedent + fail-closed fallback | MEASURED | #836, us2, e4 |
| scorer WEIGHTS three-way class: operator-property=FREE · weights=ECONOMIC (73MB wall, small-excerpt end UNPRICED) · clip-data=COUNTED | MEASURED + **OPEN(small end)** | us2; lx1 row owed |
| GT source statistics: temporally-correlated scene blocks (drives prefix bias + temporal prediction value) | MEASURED | m88 |
| videos/ dir contents fix the denominator (not manipulable — guard only) | MEASURED | #812 |

## The consumption law (why this memo exists)
Nearly every node above was FOUND — the recursive-fractal analysis largely happened. The
failure class was AUTHORITY/CONSUMPTION (see memory
`pr130-postmortem-mispriced-carriage-rerouted-months` + `zero-gravitational-pull` law): nodes
orbited the incumbent vehicle instead of composing into the design the map demanded. Every
FOUND-UNCONSUMED and OPEN row here must exit lx1 with a named consumer or an explicit
does-not-pay verdict. OPEN rows: #906 (CUDA GT gap — hardware-gated owner), scorer-weight
small-excerpt pricing, ZIP double-compression re-audit at composed byte-close, PoseNet unused
6-dim slack.
