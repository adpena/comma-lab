# Score-native amortized POSE carrier — verdict (task #57)

**Subagent:** `task57_pose_carrier`. **Authority of every number below:** `[local CPU-torch advisory]`
— exact upstream PoseNet/SegNet (`DistortionNet`) on CPU, GT decoded via
`upstream/frame_utils.yuv420_to_rgb` ONLY, scores recomputed from components (the rounded field
lies). `[macOS research-signal]` for the carrier forward (numpy↔torch RGB parity within 1 LSB).
**NOT** the contest 600-sample harness → non-promotable per the GOAL authority ladder. `$0` spend,
no GPU, no paid dispatch, **NO MPS**. `promotable=false`, `score_claim=false`,
`ready_for_exact_eval_dispatch=false`.

**Spec executed:** #57 — close the pose-carrying-appearance blocker the #56 verdict named (the
palette frame1 collapses pose; d_pose 12.66). Build the AMORTIZED luma carrier (not the #56 dead-end
per-pair 17 MB raw store), RD-sweep it, assemble the byte-closed candidate, measure the full advisory
S, and (conditionally) launch the paired exact eval. Source: `score_native_first_candidate_20260610T112433Z.md`
(#56 verdict), `score_native_pose_carrier_DESIGN_20260610T123000Z.md` (this task's pre-registration).

---

## 0. PRE-REGISTRATION (written BEFORE the measurement; see DESIGN memo)

**PREDICTION:** a score-aware learned AMORTIZED luma carrier reaches a lower d_pose at a given byte
budget than naive bilinear low-res (frame0-only f8 = 0.074), and the RD curve is monotone-decreasing.

**KILL CRITERION:** if no amortized operating point reaches a d_pose where the FULL S beats the
frontier / hits sub-0.15 — i.e. the luma needed for pose precision costs more rate than it saves vs
the 177 KB frontier decoder — record the finding + reactivate via lever C (jointly-trained smaller
amortizer that ALSO carries seg) OR a frame0-warp pose carrier.

**RESULT vs pre-registration:** PREDICTION **HALF-CONFIRMED** (the carrier DOES beat naive low-res:
d_pose 0.0036 vs 0.074, a 20× win, frame0-only) **HALF-REFUTED** (the RD curve is **NON-monotone** —
the tiny 13 KB carrier BEAT the small 49 KB one: 0.0036 vs 0.0158; more capacity trained WORSE). The
**KILL/DEFER criterion FIRES** (this is a DEFER, not a kill, per "Forbidden premature KILL"): no
operating point gets d_pose to the tube level (2.9e-5) — the best is 0.0036, **124× above** — and the
structural blocker is now precisely located (frame1's dual constraint, §3).

---

## 1. The amortized pose carrier WORKS in isolation (the #57 deliverable)

The carrier (`AmortizedLumaCarrier`: a coordinate-INR conditioned on (pair, x, y) → camera-res RGB,
trained score-aware vs the EXACT PoseNet with differentiable rgb_to_yuv6 + eval_roundtrip) reaches:

| frame0 | frame1 | d_pose | pose_term sqrt(10·) | note |
|---|---|---:|---:|---|
| GT0 | GT1 | 0.00000 | 0.0000 | sanity |
| **carrier0 (13 KB)** | **GT1** | **0.00359** | **0.1894** | the carrier in isolation |
| naive f8 lowres0 | GT1 | 0.073626 | 0.858 | the naive baseline (the carrier BEATS it 20×) |
| naive f4 lowres0 | GT1 | 0.011572 | 0.340 | even the carrier (0.0036) beats f4 |

**The carrier IS a real, working, amortized pose section** — 13,068 bytes total (the whole INR + per-
pair mod, int8+brotli), reducing frame0's pose contribution to d_pose 0.0036. numpy↔torch RGB parity
within 1 LSB (the portability contract holds; the inflate-time decode is pure numpy, scorer-free).

## 2. The RD curve (the empirical crux — NON-monotone, ceiling ~0.0036)

`score_native_luma_carrier_rd_sweep.py`, 4 pairs, 80 epochs/point, frame0-only, exact CPU PoseNet:

| capacity | params | carrier bytes | rate (4pair) | d_pose | pose_term |
|---|---:|---:|---:|---:|---:|
| **tiny** (h48/m16/f16) | 13,747 | **13,068** | 0.0087 | **0.00359** | **0.1894** |
| small (h96/m32/f32) | 53,603 | 49,157 | 0.0327 | 0.01577 | 0.3972 |
| medium (h160/m48/f48) | 181,955 | 160,538 | 0.1069 | 0.01437 | 0.3791 |
| large (h256/m64/f64) | 430,339 | 366,873 | 0.2443 | 0.00351 | 0.1875 |

**The curve is NOT monotone-decreasing** (`rd_monotone_decreasing=False`) — the tiny config (FEWEST
bytes, 13 KB) reached d_pose 0.0036; small (49 KB) AND medium (160 KB) BOTH trained to WORSE d_pose
(0.016 / 0.014); the large config (367 KB, **28× the tiny bytes**) only TIES tiny (0.0035). More
capacity / bytes does NOT buy lower d_pose: the pose loss is unstable at higher capacity (training-MSE
bounced 0.024→0.124 small, 0.027→1.77→0.023 medium, 0.016→0.101→0.021 large across epochs). The
**ceiling of the amortized-INR pose section on this video is ~0.0035** (frame0-only, best_pose_term
0.187), **~120× above the frontier's tube** (2.9e-5) — and even that best d_pose gives pose_term 0.187
≈ the ENTIRE frontier 0.191. The decisive reading: a coordinate-INR cannot amortize the 2-frame luma
motion to tube precision at ANY capacity in this family; 28× more bytes (tiny→large) buys NO pose gain.

## 3. THE STRUCTURAL BLOCKER (the sharp #57 diagnosis): frame1's DUAL constraint

The decisive coupling measurement (4 pairs, exact PoseNet) — isolating which frame's pose
contribution dominates in the ACTUAL candidate (carrier frame0 + palette frame1):

| frame0 | frame1 | d_pose |
|---|---|---:|
| GT0 | GT1 | 0.00000 |
| carrier0 | GT1 | 0.00359 |
| GT0 | **palette1** | **12.14364** |
| carrier0 | palette1 (the real candidate) | 1.93189 |

**PoseNet reads BOTH frames. The pose carrier (frame0) is solved (0.0036). But the palette frame1
— the d_seg carrier — ALONE destroys pose (d_pose 12.14).** This is the move beyond #56: #56 found
the palette frame1 collapses pose; #57 PROVES the frame0 pose carrier is recoverable AND isolates
that the *remaining* pose collapse is entirely frame1's. The score-native representation of frame1
(a piecewise-constant palette painted from the seg-generator argmax) is **pose-blind**: it has no
luma texture/motion for PoseNet. frame1 must simultaneously (a) land the SegNet argmax (d_seg) AND
(b) be a high-fidelity RGB frame (d_pose). The palette/argmax representation cannot do both.

## 4. The full byte-closed candidate (the assembled deliverable)

`experiments/results/score_native_candidate_pose_20260610T124641Z/` — archive.zip 85,590 B sha
`b971fe74…`, **lossless parity all_match=True over 4 pairs** (archive parse-back frame sha == direct
forward), scorer-free `inflate.py` decodes BOTH frames per pair (seg generator → palette frame1 +
luma carrier → frame0; NO scorer at inflate). 5-section monolithic member grammar
(`SCNP1`: seg-cfg+gen, luma-cfg+carrier, palette, pose-traj).

| section | bytes |
|---|---:|
| seg generator (lever_b INR, int8+brotli) | ~65,000 |
| luma carrier (the #57 INR, int8+brotli) | ~13,000 |
| palette (15) + pose-traj (~6,650) | ~6,665 |
| **archive.zip total** | **85,590** |

**Full advisory S (4 pairs, exact CPU scorer, recomputed from components):**

| term | value |
|---|---:|
| d_seg (palette frame1 SegNet argmax) | 0.0642 |
| d_pose (carrier0 + palette1) | 2.6749 |
| seg_term (100·d_seg) | 6.421 |
| pose_term (sqrt(10·d_pose)) | 5.172 |
| rate_term (25·85590/D) | 0.057 |
| **advisory S** | **11.65** |

vs frontier **0.19110**. **Does NOT beat the frontier; does NOT hit sub-0.15.** The eval gate
("advisory S beats frontier OR sub-0.15") is **NOT met** → **NO paired exact eval launched** (correct
fail-closed: do not spend $ to confirm a non-improvement). $0 spent.

## 5. VERDICT: DEFER-to-frame1-dual-fidelity (NOT kill; the pose-carrier sub-problem is SOLVED)

Per CLAUDE.md "Forbidden premature KILL" + Catalog #307 IMPLEMENTATION-LEVEL: the amortized pose
carrier (the #57 named build) is a **real, working, byte-closed success in isolation** (d_pose 0.0036,
13 KB, beats naive low-res 20×). The paradigm is intact. The SPECIFIC composition (palette frame1 +
carrier frame0) is falsified on the FULL S because of the newly-isolated structural coupling: **frame1
must carry BOTH d_seg and pose, and the score-native palette/argmax representation of frame1 is
pose-blind**. The KILL criterion's prediction is confirmed: pose is NOT cheaply amortizable to tube
precision via a coordinate-INR (best 0.0036 vs tube 2.9e-5, a fidelity ceiling), AND the dominant
remaining pose debt is frame1's, not frame0's.

**Why the score-native rate win does not survive the pose constraint at this representation:** the
frontier's 177 KB HNeRV decoder amortizes BOTH frames as high-fidelity RGB (so both d_seg AND d_pose
land). The score-native carrier amortizes frame1 as a *label map* (cheap, but pose-blind) + frame0 as
a low-fidelity INR (cheap, d_pose 0.0036 ceiling). The −59% rate win is real, but it is bought by
discarding exactly the frame1 luma fidelity PoseNet's tube requires. The two are coupled: you cannot
have the palette-cheap frame1 AND the tube-precise pose.

**Reactivation criteria (priority-ordered; the next builds):**
1. **frame1 = high-fidelity RGB INR that ALSO lands the SegNet argmax (lever C, the dominant
   reactivation):** replace the palette frame1 with a per-pair RGB carrier (like the #57 luma carrier
   but for frame1) trained JOINTLY against BOTH SegNet (d_seg) and PoseNet (d_pose). This converges
   toward a unified frame decoder — i.e. back toward HNeRV — and the open question is whether the
   coordinate-INR family can reach the frontier's d_seg=5.6e-4 + d_pose=2.9e-5 at < 177 KB. The RD
   ceiling found here (pose ~0.0036 for the INR family) suggests it likely CANNOT at the
   coordinate-INR capacity; a convolutional per-pair-latent decoder (HNeRV-class) is the structurally
   expressive carrier. **This is the honest conclusion: the score-native seg+palette representation is
   dominated on pose by a full-RGB decoder, exactly the HNeRV-parity-discipline lesson 5 (full
   renderer, not single-component slot).**
2. **frame0-warp pose carrier:** instead of generating frame0 pixels, store a warp/optical-flow field
   from frame1→frame0 (the pose signal IS the inter-frame motion). Needs a high-fidelity frame1 first
   (blocked by #1), so it is downstream of the frame1 fix.
3. **Joint seg+pose INR on frame1 with a wider/conv carrier:** test whether a higher-capacity (conv,
   not coordinate-MLP) per-pair carrier can break the 0.0036 pose ceiling — the non-monotone RD curve
   says coordinate-MLP capacity does not help; a different architecture family is required.

## 6. Wire-in (Catalog #125)
1. **sensitivity-map — ACTIVE:** the pose-coupling table (§3) is the new sensitivity input: frame1
   carries 12.14 of the candidate's pose debt vs frame0's 0.0036 — frame1 is the dominant pose lever,
   not frame0. The waterfiller (#54) consumes this: the pose marginal is concentrated in frame1's
   luma fidelity, which is coupled to the (SegNet-constrained) seg carrier.
2. **Pareto — ACTIVE:** the RD curve (§2) maps the carrier's {d_pose, bytes} surface and establishes
   it is NON-monotone (capacity does not buy pose below ~0.0036). The Pareto-feasible move is NOT a
   bigger coordinate-INR; it is a different frame1 representation (lever C / HNeRV-class).
3. **bit-allocator — ACTIVE:** the byte breakdown (§4) is the literal allocator; the finding is that
   allocating 13 KB to a frame0 pose carrier is well-spent (0.0036) but the 65 KB seg+palette frame1
   is mis-allocated for pose (it carries no pose fidelity).
4. **cathedral-autopilot — gate NOT met:** advisory S 11.65 ≫ frontier; no paired-eval dispatch.
5. **continual-learning — ACTIVE:** this reseeds the planner: (a) the amortized pose carrier is a real
   working primitive (d_pose 0.0036, 13 KB); (b) the RD ceiling of the coordinate-INR pose family is
   ~0.0036 (124× above the tube) — NON-monotone in capacity; (c) the binding constraint is frame1's
   DUAL (seg+pose) fidelity, which the palette/argmax representation cannot satisfy; (d) the
   score-native seg-carrier representation is dominated on pose by a full-RGB per-pair decoder
   (HNeRV-parity lesson 5).
6. **probe-disambiguator — RESOLVED:** "can an amortized learned carrier beat the naive low-res pose
   ceiling?" → YES (0.0036 vs 0.074, 20×). "can it reach the tube (2.9e-5)?" → NO (ceiling ~0.0036,
   non-monotone). "is frame0 or frame1 the dominant pose debt in the score-native candidate?" →
   frame1 (12.14 vs 0.0036). The next probe: a JOINT seg+pose frame1 carrier (lever C).

## 7. Deliverables + cross-references
- **Module (NO-FAKE, tested):** `src/tac/boundary_math/amortized_luma_carrier.py` (INR + numpy-portable
  forward + quant byte-accounting + checkpoint I/O) + 17 behavior tests
  (`tests/test_amortized_luma_carrier.py`: 15 module + 2 assembly round-trip; constant-frame negative
  control FAILS the variance test; byte cost tracks capacity; numpy↔dequant parity; member parse-back
  lossless). 86 boundary_math tests green; ruff clean.
- **Tools:** `tools/score_native_train_luma_carrier.py` (score-aware trainer vs exact PoseNet,
  differentiable rgb_to_yuv6 + eval_roundtrip), `tools/score_native_luma_carrier_rd_sweep.py` (the RD
  curve), `tools/score_native_assemble_pose_carrier_candidate.py` (byte-closed 5-section archive +
  scorer-free inflate + lossless parity + exact advisory S).
- **Byte-closed candidate:** `experiments/results/score_native_candidate_pose_20260610T124641Z/`
  (archive.zip 85,590 B sha `b971fe74…`, scorer-free inflate.py, decoded_frames/, manifest.json with
  parity proof all_match=True + advisory S row).
- **RD sweep artifacts:** `/Volumes/VertigoDataTier/pact/score_native_luma_carrier_20260610/`
  (rd_sweep_4p/rd_sweep.json + per-capacity carrier.npz checkpoints, SSD tier).
- **Cross-refs:** `score_native_first_candidate_20260610T112433Z.md` (#56 verdict this advances) ·
  `score_native_pose_carrier_DESIGN_20260610T125000Z.md` (pre-registration) ·
  `information_theoretic_floor_report_v1_20260610T102335Z.md` (S_floor) ·
  `upstream/{modules.py,frame_utils.py}` (the scorer facts) · CLAUDE.md HNeRV-parity lesson 5
  (full renderer not single-component slot — the canonical articulation of this verdict).
