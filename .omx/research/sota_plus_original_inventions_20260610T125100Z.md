# SOTA grounding + ORIGINAL inventions exploiting our MEASURED-oracle geometry (Task #59)

**Subagent:** `task59_sota_inventions` · UTC 2026-06-10 · **DESIGN ONLY (no code).**
**Authority:** every external number is tagged **EXTERNAL** (a paper/OSS *claim*, NOT our authority per
NO-FAKE class 8). Every in-repo claim is verified-present (NO-FAKE: only what the code honors). No
score claim, `promotable=false`, no dispatch, $0. Operator 2026-06-10: full authority to research
online + "invent our own bleeding edge."

**The frame (from the three pre-reads):** the problem is CLOSED — `A* = argmin_A S(A)` over a frozen
oracle {modules.py, evaluate.py, 0.mkv}. We have done what no competitor has: **MEASURED the oracle's
geometry** — the 80.67% resize null space, the SegNet top-2 **margin field** + Jacobian-difference
**polytope** coefficients, the per-pair PoseNet pixel-**Jacobian atlas** (`evaluator_response_atlas`),
the **flip-map** (#35/#36/#51), and the closed-form **water level λ\* = 25/D = 6.66e-7 score/byte**.
That measured geometry is a *private oracle model* generic SOTA cannot have, because no published codec
has the contest's frozen S/P weights and has differentiated them. **Every invention below is generic
SOTA + our private oracle measurement as the side information that makes it win.**

**The live state of play (pre-reads):** rate is already broken byte-closed (**72,217 B = −59%** vs the
177,169 B frontier, lossless parity proven, `score_native_first_candidate`). **Pose is THE live
blocker** — the palette appearance section collapses d_pose to 12.66 (sqrt term = 11.25 alone). The
floor report MEASURED: `S_floor = 0.118`, **RATE-dominated**; storing the partition *directly* LOSES to
amortization (0.169 > 0.118); pose-OUTPUT entropy is only ~1.5 KB. ⟹ **the door below S_floor is a
smaller amortizer; the door to a usable score TODAY is an amortized pose carrier (#57).**

---

## AREA (a) — Contour / region-MDL coding of the SegNet argmax partition

### SOTA baseline (cited)
- **MDL texture+boundary compression on a RAG** — Mobahi, Rao, Yang, Sastry, Ma, *"Segmentation of
  Natural Images by Texture and Boundary Compression"* (IJCV 2011, arXiv 1006.3679 [EXTERNAL]): the
  canonical coding-theoretic segmentation — agglomerative **region merge on a region-adjacency graph**,
  minimizing total code length = within-region model bits + **adaptive chain-code boundary bits**. This
  is exactly our `region_merge.py` lineage.
- **Context Adaptive Extended Chain Coding for Semantic Map Compression** (arXiv 2603.03073, 2024
  [EXTERNAL]): the *current* SOTA for semantic/label-map coding — an **extended chain code** for
  long-range contour transitions + **context-adaptive (Markov) arithmetic entropy coding**, claimed
  **~18% bitrate reduction over prior SOTA** [EXTERNAL claim, not verified by us].
- **JBIG2 / MQ-coder** (bi-level standard [EXTERNAL]): context-modeled adaptive arithmetic coding of
  binary masks — the floor for coding each of our 5 class indicator masks; subsumed by our MEASURED
  temporal-context coder (floor report §2: 253,413 B = the tightest achievable partition-direct floor).
- **STC / S-UNIWARD** — Filler-Judas-Fridrich syndrome-trellis codes + Holub-Fridrich universal
  distortion (the contest's own inverse-steganalysis theory [EXTERNAL]): assign a **per-element cost of
  change**, then STC **embeds a payload while minimizing Σ cost**, approaching the payload-limited
  entropy bound. Present in-repo: `tac.codec.syndrome_trellis_codec`, `tac.uniward_delta`.

### OUR ORIGINAL INVENTION — the **Margin-Weighted Contour Coder (MWCC)**: a contour coder whose
per-edge code cost is conditioned on the **SegNet top-2 margin** at that boundary edge.
Generic chain/contour coders (Mobahi, the 2024 ECC, JBIG2) treat every crack-edge as equally costly to
code and equally costly to *omit* — they have no notion of which boundary pixels actually flip the
argmax. **Our coder conditions the arithmetic model's context on the measured margin field**: where the
SegNet top-2 margin is large (boundary is *certain* — the argmax will not flip under the resize/quant
null space), the contour is **predictable** (a near-deterministic continuation, costing ≈0 bits) and
**omittable** (dropping/coarsening it cannot create a flip, because the margin absorbs the perturbation
— this is the §4 polytope free-budget `b(p)=m(p)/‖g_p‖` made into a *coding* prior). We **only spend
contour bits along fragile, small-margin boundary segments** — the 91% of boundary that is margin<0.5.

Formally: replace `region_merge.py`'s **uniform** 1.27-B/flip contour cost with a **margin-weighted**
per-edge cost `c(e) = -log2 P(edge | margin-context)`, and feed the surviving fragile contour into an
STC encoder using `cost(p) = margin(p)/‖g_p‖` as the S-UNIWARD-style change-cost (lever D). STC then
minimizes Σ cost while embedding exactly the flip-corrections that pay rent, approaching the
**1.27 B/flip water level from below** — the only way seg-repair beats λ\* (closed-spec §10).

### Why our measured geometry makes it win
The 2024 ECC and Mobahi code the partition geometry *as drawn* — they have no oracle. **We have the
margin field and the polytope coefficients**, so we know which boundary edges are score-irrelevant
(large margin = inside the argmax cell regardless of how we code them) and which are the *entire* signal
(margin→0). A generic coder spends uniform bits on all ~2,700 crack-steps/frame; **MWCC spends bits only
where flipping is possible**, conditioning both the *entropy model* and the *drop decision* on the
measured margin. No competitor can build this — it requires the differentiated frozen oracle.

### Feasibility / risk
- **Feasibility: HIGH.** `contour_codec.py` already *defers* the tighter arithmetic coder to STC/UNIWARD
  (verified: "do NOT hand-roll an arithmetic chain-coder here ... STC/UNIWARD"); `margin_polytope.py`
  emits the per-pixel budget; `region_merge.py` does the 1.27-B/flip cut at uniform cost; STC + UNIWARD
  codecs exist. MWCC is the *fusion* of pieces already present — a margin-context arithmetic model + STC
  with the margin cost map.
- **Risk (HONEST):** the floor report MEASURED that partition-direct storage **loses to amortization on
  rate** (0.169 > 0.118). ⟹ **MWCC is NOT a standalone rate win** — it is a *residual-repair* coder: it
  prices the boundary SOLVE's corrections on top of an amortized base (the score-native generator's
  contiguous residual, which IS efficiently chain-codable, §1 of the score-native memo). Its EV is in
  driving the score-native candidate's seg term down below the water level, not in replacing the
  decoder. Risk it stays above 1.27 B/flip even with margin weighting (the STC-clean-source DEFER
  measured uniform-cost STC at 2.4-2.6× brotli) — **margin-weighting is exactly the cost-map that DEFER
  said was the missing reactivation lever**, so this is the test of that hypothesis.

### Build recommendation: **BUILD A $0 SMOKE FIRST** — margin-weighted STC cost on the score-native
candidate's *contiguous* residual vs uniform brotli; target the DEFER's >=5% bar at < 1.27 B/flip.

---

## AREA (b) — Waterfilling / Lagrangian R-D + optimal-transport allocation

### SOTA baseline (cited)
- **Classical waterfilling / Lagrangian RDO** (Shoham-Gersho; Ortega-Ramchandran [EXTERNAL]): optimal
  bit allocation equalizes the marginal `∂D/∂R` across all sources at a single multiplier λ — the KKT
  stationarity condition. Modern frame-level rate-control (Gu et al. 2024 [EXTERNAL]) predicts per-frame
  R-D-λ then allocates globally within a GOP.
- **R(D) ↔ Optimal Transport** — *"On a Relation Between the Rate-Distortion Function and Optimal
  Transport"* (arXiv 2307.00246 [EXTERNAL]): R(D) is computable via **entropic OT**, and Blahut-Arimoto
  ≈ **Sinkhorn-Knopp** matrix scaling. Sinkhorn (Cuturi 2013 [EXTERNAL]) solves a global transport
  budget by iterative marginal-scaling — closed-form plan `P* = diag(u) K diag(v)`.

### OUR ORIGINAL INVENTION — the **Closed-Form λ\*-Equalizing Allocator with Pose-as-Global-OT-Budget.**
Generic RDO *estimates* per-source R-D slopes by encoding at several λ and fitting curves (a sweep).
**We have the EXACT closed-form marginals** from the measured oracle, so allocation becomes a *solve*,
not a sweep:
- **seg (linear `100·d_seg`):** every flip is worth exactly `100/N = 8.48e-7`; pays rent iff its
  contour-repair cost < `(100/N)/λ\* = 1.27 B/flip` — a closed-form admission test, no sweep.
- **pose (concave `sqrt(10·d_pose)`, GLOBAL pool):** here is the original move. Because pose is
  **pooled-before-sqrt**, a pose-error reduction on *any* pair trades 1:1 with any other — it is a
  **single global budget**, and the marginal `5/sqrt(10·d_pose)` is shared. This is an **optimal
  transport problem**: distribute a fixed pose-byte budget `B_pose` across 600 pairs to minimize total
  d_pose, where the per-pair cost-to-reduce is the **PoseNet Jacobian norm** from the atlas. Solve it
  with **Sinkhorn** (the R(D)↔OT identity): the transport plan moves bytes to the pairs with the
  steepest `Δd_pose/Δb` (smallest Jacobian-conditioned cost) until every funded pair's marginal equals
  λ\*/(5/sqrt(10·d_pose)) — a **single Sinkhorn scaling**, closed-form, no per-pair sweep.

This makes the meta-Lagrangian/Pareto solver (already partly built: `evaluator_action_waterfill`,
`joint_p18_p19_waterfill`, V3 exact-ΔS waterfiller) consume **closed-form measured marginals** instead
of swept estimates ⟹ the waterfilling becomes the KKT SOLVE the closed-spec §10 specifies.

### Why our measured geometry makes it win
SOTA RDO sweeps because it cannot evaluate `∂D/∂R` without encoding. **Our atlas gives the per-pair
PoseNet Jacobian norm directly** (the exact `Δd_pose` per unit byte spent on that pair's pose-relevant
luma) and the polytope gives the per-flip seg marginal — so we equalize at the analytic λ\*=6.66e-7 in
*one* allocation, exploiting the pooled-before-sqrt structure as a literal OT budget that no codec
without the differentiated oracle can see.

### Feasibility / risk
- **Feasibility: HIGH (the allocator) / MEDIUM (the pose-OT solve).** The λ\* admission test and
  seg side are pure arithmetic over existing fields (`margin_polytope`, `region_merge` already encode
  1.27). The pose-OT Sinkhorn needs the per-pair `Δd_pose/Δb` cost vector, which requires the *amortized
  pose carrier* (area c) to define what "spend a byte on pair i's pose" even means.
- **Risk:** pose marginals couple through the *shared carrier weights* (amortization), so the per-pair
  cost is not perfectly separable — Sinkhorn assumes a fixed cost matrix. Mitigate by alternating
  (carrier train → re-measure atlas Jacobian → re-allocate), i.e. an EM/Dykstra outer loop.

### Build recommendation: **BUILD the closed-form seg-side allocator NOW** (it is arithmetic over
existing measured fields and directly feeds the score-native candidate); **DESIGN the pose-OT layer to
consume area (c)'s carrier** — it cannot be built before the Jacobian-native carrier exists.

---

## AREA (c) — Amortized neural carrier of pose-relevant 2-frame luma MOTION  ⟵ feeds #57 (THE LIVE PRIZE)

### SOTA baseline (cited)
- **NeRV / HNeRV / NVRC** (Chen 2021; Chen 2023; NVRC 2024 [EXTERNAL]): index→frame INRs with
  quantization-aware training + entropy coding; FFNeRV (flow-guided) claims to beat H.264/HEVC
  [EXTERNAL]. The frontier 177-KB decoder is this class.
- **Ballé entropy bottleneck + scale hyperprior** (2018 [EXTERNAL]): the canonical learned-codec rate
  model `bits = -log2 p_y(y)`.
- **Image/Video Coding for Machines — the decisive SOTA** — Fernandez-Menduina et al. (USC),
  *"Feature-Preserving Rate-Distortion Optimization in Image Coding for Machines"* (arXiv 2408.07028 /
  2504.02216, 2024-2025 [EXTERNAL]): RDO against a **task feature extractor's Jacobian**. Taylor-expand
  the task net `f`: `‖f(x)−f(x̂)‖² ≈ ‖J_f(x)(x̂−x)‖²` — the **Input-Dependent Squared Error (IDSE)**, a
  quadratic metric in the compression residual weighted by the Jacobian Gram matrix; evaluated in the
  transform domain via random projections (`B = S·J_f·D`). **Their limitation (EXTERNAL, key):** the
  Jacobian is *computed per-image at encode time as a high-bit-rate Taylor approximation* with a
  block-diagonal Gram assumption.

### OUR ORIGINAL INVENTION — the **PoseNet-TUBE-Native Carrier (PTNC):** an amortized luma-motion INR
trained against the **measured PoseNet pixel-Jacobian atlas** so it spends bits *only* in the input
directions PoseNet's 6 scored dims are sensitive to, holding d_pose at the frontier ε≈3e-5 at minimal
byte. This is the IDSE idea — but our Jacobian is the **MEASURED, PRE-COMPUTED atlas of the FROZEN
oracle** (`evaluator_response_atlas` already has the per-pair PoseNet frame-channel pixel-Jacobian
field), not a per-image Taylor approximation. And we apply it where it matters most: **frame0 is
SegNet-INVISIBLE** (SegNet reads `x[:,-1,...]` = frame1 only), so frame0 is a *free pose-only carrier* —
the PTNC paints frame0's luma (and frame1's pose-relevant luma, in the seg null space) purely in the
PoseNet-sensitive subspace.

**The mechanism that beats both the frontier and the current `amortized_luma_carrier`:** the current
carrier (verified in-repo) trains with **pose-MSE** — it tries to reproduce the *appearance* that yields
the right pose, paying for every luma pixel. **PTNC replaces pose-MSE with the Jacobian-projected
loss**: project the carrier residual onto the PoseNet Jacobian's 6-scored-dim row space and penalize
ONLY that projection — `L_pose = ‖J_P · (carrier − GT)‖²` restricted to the 6 scored dims (literally
IDSE with our measured `J_P`). The carrier is then **free to be wrong everywhere the Jacobian is zero**
(the pose null space — most of the luma field), so it amortizes to a tiny weight set: it need only carry
the **pose-relevant motion tube**, a 600×(low-rank) object, not a 600-frame appearance. The floor report
MEASURED the pose-OUTPUT entropy at ~1.5 KB — **PTNC is the carrier that realizes that floor** by
storing only the Jacobian-row-space motion, not the 17-MB raw appearance (the #56 dead end) nor the
frontier's 177 KB that pays for appearance it doesn't need.

### Why our measured geometry makes it win
USC's IDSE *approximates* the Jacobian per-image and assumes high bit-rate + block-diagonal Gram. **We
have the EXACT measured Jacobian atlas of the frozen PoseNet** (no approximation, no per-image cost), so
(1) the projection is exact, not Taylor; (2) we exploit the **frame0-SegNet-invisibility** + the
**80.67% resize null space** — two free subspaces no general codec knows about — to place all
pose-carrying luma where it costs zero seg and zero appearance fidelity; (3) the pooled-before-sqrt pose
structure (area b) lets us allocate the carrier's capacity across pairs by the atlas Jacobian norm. The
result is the **smallest possible pose carrier**, which is precisely the open lever the floor report
named as "the only door below S_floor: a smaller amortizer."

### Feasibility / risk
- **Feasibility: MEDIUM-HIGH.** `amortized_luma_carrier.py` exists (the INR scaffold, numpy-portable
  forward, MLX training path) — PTNC is a **loss-function + capacity change** on it (swap pose-MSE for
  Jacobian-projected loss; restrict carrier to frame0 + seg-null of frame1). The atlas Jacobian exists.
  MLX-first per CLAUDE.md; numpy reference for portability; CPU-torch scorer for d_pose; **NO MPS**.
- **Risk:** (1) the measured atlas Jacobian is at the *GT operating point*; as the carrier moves the
  input, the Jacobian shifts (the Taylor validity USC flagged) — mitigate with a trust region + re-measure
  (the area-b alternating loop). (2) PoseNet's `rgb_to_yuv6` is `@torch.no_grad()`/in-place — the
  Jacobian must come from the differentiable-YUV6 path (`tac.differentiable_eval_roundtrip`, per CLAUDE.md
  eval_roundtrip non-negotiable) or from the already-measured atlas. (3) it is a research-signal until
  byte-closed + paired CUDA+CPU exact eval — the score-native candidate's pose collapse is the gate it
  must clear (d_pose 12.66 → ε).

### Build recommendation: **THIS IS THE #1 BUILD.** It is the live blocker (#57). Start with a $0 MLX
smoke: train PTNC with the Jacobian-projected loss on N pairs, measure d_pose on the CPU-torch scorer vs
the palette's 12.66 and the GT floor 0.0; **pre-register kill = d_pose stays > ~0.1 at < 30 KB amortized.**

---

## AREA (d) — Rust codec crates + runtime-less native 'core representation grammar' decoder  ⟵ feeds #58

### SOTA baseline (cited)
- **constriction** (bamler-lab, Rust+Python, Oct-2024 active [EXTERNAL]): composable ANS + range coders,
  **no_std + WASM** support, "< 0.1% above the theoretical minimum bit rate" [EXTERNAL claim]. Already in
  our `runtime-rs/Cargo.lock` (alongside `brotli`).
- **The runtime-less native-decoder pattern** — our own `runtime-rs/crates/tac-boundary-decode`
  (verified present): a FIXED, rate-free Rust decoder (contour decode, d_seg popcount, connected
  components) with **bit-identical golden-vector parity gates** against the Python oracle and a
  payload-cleanliness discipline (NO video-derived constant in the binary, per CLAUDE.md "Native
  eval-time runtime discipline"). Sister crates: `tac-packet-compiler`, `qma-codec`, `stbm1br-codec`,
  `zipwire`, `residual-codec`.

### OUR ORIGINAL INVENTION — the **Core Representation Grammar interpreter vs Fixed-Form decoder
bake-off**, deciding the carrier-runtime route empirically.
Two routes from the closed-spec §6:
- **(a) grammar + tiny interpreter:** archive = a *program* in a small domain grammar
  `{region-fill(label, contour), luma-field(jacobian-coeffs), pose(traj)}`; inflate is a ~few-hundred-line
  Rust interpreter. Flexible (one decoder handles any candidate's section mix), carries interpreter
  overhead in the binary (rate-free) but the *program* may have grammar overhead per archive.
- **(b) fixed-form decoder:** a hard-coded rasterizer (the `tac-boundary-decode` shape) that reads
  fixed-offset sections. Smallest archive (no grammar tokens), but a new section type = a new decoder.
**The invention is the explicit MEASUREMENT** — encode the *same* score-native candidate both ways and
compare (archive bytes, inflate wall-clock on the contest 30-min budget, binary size). My **prediction
from the floor report:** because the carrier is now THREE small sections (contour partition + PTNC luma +
pose traj) and the partition is the dominant payload, the **grammar tokens are a negligible fraction**,
so route (a)'s flexibility likely costs ~0 bytes while saving us from re-porting a decoder per candidate
— *unless* the archive shrinks below ~10 KB (then grammar overhead matters and (b) wins). The
measured-geometry hook: the grammar's `luma-field` opcode carries **Jacobian-basis coefficients** (the
PTNC representation), so the interpreter rasterizes the pose tube directly from the atlas basis — a
representation no generic codec runtime supports.

### Why our measured geometry makes it win
A generic Rust image decoder rasterizes pixels. **Our grammar's primitives are the scored objects
themselves** (argmax partition contours + Jacobian-basis luma + pose trajectory) — the decoder emits a
witness that lands in the argmax cell + pose tube *by construction*, carrying no appearance. The fixed
rate-free decoder means archive bytes = pure payload; the parity-gate discipline (already in the crate)
guarantees the native decoder reproduces the Python oracle byte-for-byte (the contest-compliance gate).

### Feasibility / risk
- **Feasibility: MEDIUM.** The fixed-form decoder for the *current* sections largely exists
  (`tac-boundary-decode`: contour + d_seg + components). The grammar interpreter + the PTNC luma opcode +
  the bake-off harness are new. Native discipline (oracle parity, payload-clean, no scorer load) is
  established.
- **Risk:** premature — the sections aren't settled until area (c) lands the pose carrier. Building the
  grammar before the carrier representation is fixed is the wrong order (the closed-spec §9 routes the
  carrier-runtime *after* the seg-core + pose-carrier prove out). LOW EV until #57 resolves.

### Build recommendation: **DEFER the bake-off until area (c) settles the pose-carrier representation;**
meanwhile, keep `tac-boundary-decode` parity-gated and EXTEND it with the PTNC luma opcode *as a design
stub* so the grammar's hardest primitive is scoped.

---

## SUMMARY TABLE

| Area | SOTA baseline (cited, EXTERNAL) | OUR original invention | Predicted edge (measured-geometry reason) | Feeds task |
|---|---|---|---|---|
| (a) contour/region-MDL | Mobahi IJCV2011 RAG-merge; 2024 Context-Adaptive Extended Chain Coding (arXiv 2603.03073, ~18% claim); JBIG2/MQ; STC/UNIWARD | **Margin-Weighted Contour Coder (MWCC)** — per-edge cost + drop decision conditioned on SegNet top-2 margin; STC with margin cost-map | Spends contour bits ONLY on margin<0.5 fragile boundary (91% is certain/omittable via the polytope free-budget); the cost-map the STC DEFER said was missing. Target < 1.27 B/flip | #59 seg-core → score-native (#55/#56), Lever F |
| (b) waterfilling/OT | Shoham-Gersho/Ortega-Ramchandran Lagrangian RDO; R(D)↔OT Sinkhorn (arXiv 2307.00246) | **Closed-form λ\*-equalizing allocator + pose-as-global-OT-budget (Sinkhorn over per-pair atlas Jacobian)** | Exact closed-form marginals (no sweep) at analytic λ\*=6.66e-7; pooled-before-sqrt pose = literal OT budget solved in one Sinkhorn | #54 waterfiller, meta-Lagrangian solver |
| (c) amortized pose carrier | NeRV/HNeRV/NVRC; Ballé hyperprior; **USC IDSE Jacobian-RDO (arXiv 2408.07028)** | **PoseNet-TUBE-Native Carrier (PTNC)** — INR trained against MEASURED PoseNet Jacobian atlas, painting only frame0 (SegNet-invisible) + seg-null luma in the pose-sensitive subspace | EXACT measured oracle Jacobian (not per-image Taylor); exploits frame0-invisibility + 80.67% resize-null + pooled-pose → smallest pose carrier, realizes the ~1.5 KB pose floor | **#57 (LIVE PRIZE)**, the door below S_floor |
| (d) Rust runtime-less decoder | constriction (no_std/WASM ANS); `tac-boundary-decode` fixed-form pattern | **Grammar-interpreter vs fixed-form bake-off**, grammar `luma-field` opcode = Jacobian-basis coeffs | Decoder primitives ARE the scored objects (cell+tube by construction), zero appearance; parity-gated oracle equivalence = contest-compliant | #58 |

---

## TOP-3 HIGHEST-EV ORIGINAL INVENTIONS (the report-back)

1. **PTNC — PoseNet-TUBE-Native Carrier (area c)** → **feeds #57, the live prize.** Highest EV: pose is
   the *single binding blocker* on a candidate that already won rate by −59% byte-closed. The
   measured-Jacobian-projected loss + frame0-invisibility exploit is genuinely original (USC's IDSE
   approximates the Jacobian per-image; ours is the exact frozen-oracle atlas), and the floor report
   proves a ~1.5 KB pose carrier is information-theoretically possible. **READY TO BUILD NOW** as a $0
   MLX smoke on `amortized_luma_carrier.py` (loss swap + capacity restriction). Pre-registered kill:
   d_pose > ~0.1 at < 30 KB.
2. **Closed-form λ\*-equalizing allocator + pose-OT (area b)** → **feeds #54 / the meta-Lagrangian
   solver.** Turns the waterfiller from a sweep into a KKT solve using marginals we already measured; the
   seg-side is **buildable now** (pure arithmetic over `margin_polytope`/`region_merge`). The pose-OT
   Sinkhorn layer waits for PTNC to define per-pair pose cost — so it composes directly with invention 1.
3. **MWCC — Margin-Weighted Contour Coder (area a)** → **feeds the score-native seg-core (#55/#56) +
   Lever F.** It is the precise test of the STC-clean-source DEFER's named reactivation lever (a
   detector-informed cost map), on a base whose residual is now *contiguous* (chain-codable). **NEEDS A
   $0 SMOKE** (margin-weighted STC vs uniform brotli, the >=5% / <1.27 B/flip bar) before any build
   commitment — honest because partition-direct storage is MEASURED to lose, so MWCC's EV is strictly as
   a residual-repair coder, not a standalone rate win.

(Area d — the Rust grammar bake-off — is **deliberately deferred**: low EV until the pose-carrier
representation settles, per closed-spec §9 routing. Keep `tac-boundary-decode` parity-gated; scope the
PTNC luma opcode as a stub.)

**Build-readiness flags:** (c) PTNC — **build now** ($0 MLX smoke). (b) seg-allocator — **build now**
(arithmetic); pose-OT — **after (c)**. (a) MWCC — **$0 smoke first**, then build if it clears the
DEFER's bar. (d) — **defer + stub**.

---

## WIRE-IN (Catalog #125)
1. **sensitivity-map — ACTIVE.** MWCC margin cost-map + PTNC Jacobian projection + the closed-form
   per-axis marginals are direct sensitivity inputs the waterfiller (#54) consumes.
2. **Pareto — ACTIVE.** This memo re-ranks the offensive levers: pose-carrier (c) is the binding Pareto
   move (rate is off the frontier vertex by −59%, pose is the cliff); the allocator (b) computes the
   achievable {d_seg, d_pose, B} surface at λ\*.
3. **bit-allocator — ACTIVE.** Invention (b) IS the bit-allocator (closed-form λ\*-equalization + pose-OT).
4. **cathedral autopilot — N/A.** Design-only surface; no archive bytes emitted (the PTNC smoke is the
   next dispatch surface, gated on advisory d_pose improvement).
5. **continual-learning — ACTIVE.** Seeds the planner: SOTA IDSE confirms the Jacobian-RDO direction is
   correct and our measured atlas is the strict improvement; the partition-direct-loses-to-amortization
   floor demotes any standalone partition-storage rate play; pose-OT is the unbuilt allocator layer.
6. **probe-disambiguator — ACTIVE.** Two pre-registered $0 probes named: (c) "does Jacobian-projected
   loss beat pose-MSE at lower byte?" and (a) "does margin-weighted STC beat uniform brotli at <1.27
   B/flip on the contiguous residual?".

## CROSS-REFERENCES
`closed_spec_boundary_math_system_of_equations_20260610.md` (§4 polytope, §6 carrier-runtime, §10 water
level) · `information_theoretic_floor_report_v1_20260610T102335Z.md` (S_floor=0.118, partition-direct
loses, pose ~1.5 KB) · `score_native_first_candidate_20260610T112433Z.md` (−59% rate, pose-collapse
blocker, #57 reactivation) · `lever_b_score_native_argmax_smoke_verdict_20260610.md` ·
`stc_clean_source_mask_delta_disambiguator_probe` DEFER (the cost-map reactivation lever MWCC tests) ·
in-repo verified: `src/tac/boundary_math/{margin_polytope,contour_codec,region_merge,amortized_luma_carrier}.py`,
`src/tac/optimization/{evaluator_response_atlas,evaluator_action_waterfill,joint_p18_p19_waterfill,resize_null_preimage}.py`,
`src/tac/codec/syndrome_trellis_codec.py`, `src/tac/uniward_delta.py`,
`runtime-rs/crates/tac-boundary-decode/` · EXTERNAL: arXiv 1006.3679, 2603.03073, 2307.00246,
2408.07028, 2504.02216; constriction (bamler-lab); JBIG2; NVRC 2024 — all EXTERNAL claims, not our authority.
```
