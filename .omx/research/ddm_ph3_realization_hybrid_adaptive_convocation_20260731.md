---
schema: ddm_ph3_convocation.v1
date_utc: 2026-07-31
arm: MAIN (8th standing convocation, operator-prompted; $0 analysis over live receipts)
lane_id: "lane_ddm_ph3_realization_hybrid_adaptive_20260731"
research_only: true
score_claim: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
operator_verbatim: "realization crux and all other negative findings and the path forward? Also what about two plane versus single and certain pairs doing better with different techniques, can we use hybrid approach here, and also maybe there is more optimal? Are we using fixed length or coarse to fine and dynamic adaptive and such?"
operating_state: "v4c MEASURED S 0.992972 (seg 0.431179 + pose 0.322250 + rate 0.239543; 359,750B) [macOS-CPU advisory]; fidelity law 2 anchors (3e-6, 1.38e-4); bars 0.172141 / 0.15"
council_tier: T2
council_verdict: PROCEED
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_dissent:
  - member: Contrarian
    verbatim: "Coarse-to-fine was measured DOMINATED at the token level (QA07 nested rungs). Do not adopt adaptive precision as a law — RACE it per stream; the pose-stream prediction rests on measured misallocation, not on a measured win."
council_assumption_adversary_verdict:
  - assumption: "realized acceptance is the only admissible acceptor on this vehicle"
    classification: HARD-EARNED
    rationale: "5 independent instruments (pi2 tangent 3-10x over-predict; QA05 structured<noise; QA11 overturned through coder+render; kl1 law-coders lose; fd2 uint8 gap) + 2 fidelity-law anchors on the constructive side"
  - assumption: "fixed 6xf16/pair pose storage is adequate"
    classification: CARGO-CULTED (measured-misallocated)
    rationale: "pi2: dim0 f16-MARGINAL (1 ulp ~ 0.040 S at v4b operating point) while dims 3-5 value-null; per-pair d_pose spread spans orders of magnitude — the flat grid provably misallocates on BOTH axes"
---

# ddm_ph3 — 8th convocation: the realization crux resolved into a doctrine · the hybrid selector generalized · fixed-length declared cargo-cult

**POINTER HONESTY FIRST: `0.1910828242 [contest-CPU]` UNMOVED.** All rows advisory. Operating row:
v4c measured 0.992972 (seg 0.431 now the largest axis · pose 0.322 · rate 0.240).

## §1 THE REALIZATION CRUX — the negatives compress to ONE law with a constructive half

**The measured negative pile (5 independent instruments, one mechanism):** pi2 (tangent over-
predicts realized Δpose 3-10×; min-L2 atoms sub-LSB); QA05 (structured rank-1 edits < the 72-
control noise floor); gr1 (QA11 sensitivity law INSTANCE-OVERTURNED through real coder+render);
kl1 (spline/AR/rank-1 law-coders LOSE to byte-plane on white solver fields ×3); fd2/#532 (uint8
breaks range(A) exactness, Δ=62.74 vs 1.7e-13). **The measured positive pile (same mechanism,
other side):** the v4b/v4c chain — GN with REALIZED monotone acceptance at shipped quanta +
min-selection fallbacks — predicted the real evaluator to 3e-6 and 1.38e-4 (two anchors).

**Schmidhuber's compression of all of it (the doctrine):** *the realization wall is not a wall to
analysis — it is a wall to LINEARIZATION.* Proxies (gradients, sensitivity maps, laws, ranks) are
PROPOSAL GENERATORS; the uint8/coder/render path is the only ACCEPTOR; anything accepted through
the realized path composes to instrument grade (the fidelity law is the proof). Consequence for
the path forward: stop spending arms trying to make proxies authoritative (that family is closed
at this vehicle — 5 receipts); spend on making proposals CHEAP (MPS-gradient, atlas-ranked) and
realized acceptance CHEAP (batched realized evals, chunked scorers). The campaign already runs
this loop; §1 makes it doctrine.

**Fridrich's design rule from the same receipts:** the acceptor lives on the uint8 lattice, so
admissible actuators must move ≥1 LSB **coherently over sufficient support** — the two-plane warp
won because 6 params move ~half the frame coherently; steering atoms died because they were
sub-LSB. *Coherence × support beats amplitude* — rank actuator families by (LSBs moved × support
area × coherence), not by gradient norm.

**Kolmogorov/MDL on the three white fields (the deepest wonder):** the solver launders smooth
physics into white parameters because the solution SET is degenerate (rotation/translation
aliasing = gauge freedom in solution space) and the solver picks an arbitrary member. The cure is
NOT post-hoc coding (measured dead ×3) — it is **min-entropy member selection**: among near-
equivalent realized optima, pick the one on the smooth/predictable manifold — same distortion,
lower field entropy. This is #602's min-description-polytope-member idea applied to the pose
field, and it is UNMEASURED at this vehicle. ($0-cheap probe: re-run the tail solve with a tiny
tie-breaking penalty toward the temporal-neighbor prediction, INSIDE realized acceptance — accept
only ΔS≤0 moves — then measure field entropy through the real coder.) → QA70.

**Dykstra on the negatives generally:** each negative carved the feasible set — corrections dead,
token-finer dominated, chroma-blind pose, rung-A-global harmful, tier-2 token-only — and what
remains after the carving is EXACTLY the live plan: base-quality seg (structural rungs), realized
per-pair pose portfolio, adaptive allocation. The negatives were the search; honor them by not
re-entering closed rooms (verdict-scope ladders all recorded).

## §2 HYBRID — the selector already proved the thesis; generalize it to a PER-PAIR EXPERT MENU

**Measured evidence that per-pair technique choice pays:** the 1-bit selector (single vs two-
plane) is already a 2-expert mixture and it is load-bearing (always-two-plane blows up ~10 pairs;
best-of is monotone-safe); v4c re-solve: 188/250 pick two-plane, 62 keep single; rung-C Movable
plane: MIXED (10/17 improve, 6 degrade) = exactly an expert that needs a selector; rung-A shear:
per-pair wins, global hurts = same signature; qa45's 6 static-mask losers are a typed class
(below-horizon boundary). Different pairs ARE best served by different physics — measured, not
conjectured. This is the Netflix-per-content-RD archetype (already in the codec-identity memory)
landing at the PAIR level.

**The generalization (v4d pose grammar): expert menu E = {single, two-plane, +Movable-depth,
+per-pair-shear, (+steering-coeffs for the hard core)} with encode-side best-of under JOINT
realized pricing** — expert k costs bytes(params_k) + selector entropy; accept the argmin realized
S contribution per pair. Selector cost: entropy-coded expert IDs (usage is heavily skewed →
≪log2|E| bits/pair average). Admission rule per expert (Hotz): an expert enters the menu only if
its realized win on its winning pairs exceeds its param+selector bytes at water — no menu bloat.

**"Maybe there is more optimal" — yes, two rungs above the discrete menu:** (a) the CONTINUOUS
family view — experts are quantized points of one parametric family (number of planes k_p, plane
inverse-depths as per-pair scalars); the menu is its cheap discretization, and per-pair k_p with
tiny depth scalars may dominate fixed experts; (b) the RECEIVER-DERIVED view (pi2's surviving
path) — the receiver computes per-pair structure (Jacobian basis, depth layout) from decoded
content for FREE, so the shipped selector can shrink toward "exceptions only" (predict the expert
from decoded frame_1; ship only disagreements). Both are races, not adoptions. → QA68.

## §3 FIXED-LENGTH IS A MEASURED CARGO-CULT — but coarse-to-fine must be RACED, not assumed

**Current truth: every stream is FIXED-length** (pose 6×f16/pair flat; (a,b) 2×f16 flat; selector
1 bit; tokens mod-16 with cell-drop = the only spatial adaptivity, and it is the measured rate
frontier). **The misallocation is measured on two axes:** per-DIM (pi2: dim0 f16-marginal — one
ulp ≈ 0.040 S at v4b — while p3-5 are value-null; the flat f16 grid gives every dim identical
precision) and per-PAIR (post-v4c d_pose spans orders of magnitude across pairs; hard pairs are
precision-starved, easy pairs waste bits).

**The pantheon's design (Shannon/MacKay), with the Contrarian's measured brake welded on:**
1. **Per-dim variable quanta** — dim0 offset-coded finer (QA65, the owed v4d rung), p1/p2 at f16,
   p3-5 coarse-or-derived: a few hundred bytes swing + a distortion win. Predicted-strong
   (misallocation measured); race the exact quanta.
2. **Per-pair coarse-to-fine REFINEMENT waterfill** — solve all pairs at a coarse base quantum;
   REFINE only pairs whose realized marginal ΔS/Δbyte clears water, greedy-under-realized-
   remeasure (the v19c saturation pattern applied to pose precision). This is dynamic-adaptive
   allocation with the acceptor in the loop — never a schedule, always a race.
3. **The brake (Contrarian, binding):** gr1 measured nested precision rungs DOMINATED at the
   TOKEN level (drop-to-base beat every intermediate). Coarse-to-fine is NOT a law — it won
   nowhere yet; it is a per-stream RACE whose pose-stream prior is good because the misallocation
   is measured. Falsifier per stream: refinement curve never clears water → fixed-at-optimal-
   single-quantum wins, record and close.
4. **Archive-level joint waterfill** — one KKT allocation across ALL live pools (pose precision ×
   cell restore × photometric × expert menu × selector) at realized exchange rates, now cheap to
   measure because the fidelity chain is trusted. The pools are non-additive (standing law); the
   waterfill is the composition instrument. → QA69.

## §4 What they WONDER (open, cheap, decisive)
1. (Shannon) Does min-entropy member selection (§1/QA70) make the pose field codable at equal
   realized d_pose? One number decides whether "white solver fields" was a solver artifact.
2. (Tao) The 17-pair hard core is still untyped geometrically (QA48) — after v4c, does it shrink,
   and is the residue epipole-in-view? Type before curing.
3. (MacKay) Where is the realized pose floor? p3v2 bound says ≤1e-3-class; current mean 0.0104 —
   is the remaining 0.32 contribution precision-starved (adaptive fixes it) or content-limited?
   The per-pair refinement curve answers this as a byproduct.
4. (Assumption-Adversary) Seg is now the largest axis (0.431). Is the pose-first reflex now
   inertia? The next heavy slot arguably belongs to the seg structural rung (QA24 coarser
   re-burn, operator-GO class) in PARALLEL with cheap v4d pose rungs — not after them.

## §5 ROUTING (defer-at-source, same commit)
- QA68 per-pair expert-menu hybrid (menu + admission rule + entropy-coded selector + receiver-
  predicted-selector race) · QA69 realized bit-allocation (per-dim quanta + per-pair refinement
  waterfill + archive-level joint KKT; per-stream RACE discipline w/ QA07 brake) · QA70
  min-entropy member selection probe ($0-cheap, inside realized acceptance).
- v4d arm charter = QA65 + QA66 + QA68 + QA69 (+QA70 rider) under the §1 doctrine (proposals
  cheap, acceptance realized, gates verify); seg structural rung (QA24 re-burn) surfaced to
  operator as the PARALLEL heavy slot per §4.4.

## §6 ADDENDUM (operator 07-31 "the quantum issue") — the pantheon on quantization: not a wall, a MEDIUM

**The measured quantum stack (three lattices, one issue):** (1) uint8 pixels at the decode boundary
(realization death for sub-LSB atoms — pi2; #532 range(A) exactness broken Δ=62.74); (2) f16
storage quanta on shipped params (dim0 f16-MARGINAL, 1 ulp ≈ 0.040 S — pi2; QA65 offset cure
riding v4d); (3) the solve-lattice interaction (monotone acceptance AT q16 = the solver only
visits lattice points). Prior partial cures already in receipts: camera-Q8 sub-quantum staging
(j5) · quarter-quantum realized caps (#518/j4) · #149 pre-R placement (set flips at camera res
BEFORE the averaging) · fd2's named quantum-arithmetic fork.

**FRIDRICH/YOUSFI (the inverse-steganalysis reading — the deepest cut):** the contest IS inverse
steganalysis, and quantization boundaries are precisely where steganographic technique lives.
Three mechanisms, all classical steg/halftoning transplants:
1. **The resize divides the quantum.** R averages ~k camera pixels per scorer pixel (k≈30 area).
   A COHERENT pattern of ±1-LSB roundings across a support realizes EFFECTIVE sub-LSB moves at
   the scorer's input — spatial DITHERING / error diffusion through the averaging operator. The
   realization wall is not 1 LSB; it is ~1/k LSB for coherent dithered actuators. **Named
   reopening: the steering-atom family (pi2, killed at naive rounding) re-enters at
   DITHERED-REALIZATION scope** — the atoms died because rounding was treated as given, not
   designed.
2. **The rounding-direction field is a near-free actuator.** Pixels whose pre-round fractional
   part sits near 0.5 can round either way at negligible cost (wet-paper-codes logic: use the
   "dry" pixels). We ship descriptions, not pixels — so the field is steered by SUB-EFFECT-SCALE
   perturbations of spatially-LOCAL description DOF (tokens are cell-local: a tiny token nudge
   flips the rounding pattern of its cell). Cell-granular, ~zero marginal bytes at f16.
3. **tt1 is already the discovery instrument for this**: a gradient on the pre-uint8 values sees
   the threshold crossings; realized acceptance verifies each one. The joint TTO will find
   dithering implicitly IF token DOF are in the loop — watch for accepted steps whose continuous
   magnitude is sub-LSB but whose realized effect is not (the signature).
**TAO / THE LATTICE LINE (ms1/ms2 lineage):** rounding = CVP in the SCORER metric on a severely
anisotropic lattice (head cond 24.8, per-dim sensitivity spread ~600×). Independent per-param
rounding is Babai rounding — provably suboptimal on anisotropic lattices. Part of the measured
3-10× tangent-overshoot may be a ROUNDING artifact, not intrinsic realization: solve CONTINUOUS,
then JOINT-round the coupled 6-dim block in the scorer metric (race vs the current
solve-on-lattice). Cheap race, bounded pairs.
**SHANNON/MACKAY/LLOYD:** quantizer design is a solved discipline being applied naively — f16 is
a value-space log quantizer, but sensitivity is not proportional to value. Per-dim COMPANDING
(Lloyd-Max in the measured d_pose-vs-Δparam curve) is QA65/QA69's principled completion; the
refinement waterfill's realized curve IS the quantizer-design instrument. And the quantum is the
RATE DIAL — coarser where the curve is flat, finer where f16-marginal; never uniform.
**THE DECISIVE $0 WONDER (stage attribution):** is the current pose residual (contribution 0.322)
bounded by f16 storage (curable: offset/companding — QA65 measures), by uint8-through-R (curable:
dithering — QA72), or by content (neither)? Solve a bounded set at CONTINUOUS precision, then
re-quantize stage-by-stage (fp32 → f16 → uint8-through-R) and attribute the realized loss per
stage. One afternoon; aims every quantum lever.
**CONTRARIAN (binding):** every dithering/CVP claim passes the same realized acceptor as
everything else; token-nudge collateral (seg flips) is jointly priced; the steg transplants are
EASIER here than in steganography (we fight an exact forward simulation, not a detector) but that
is an argument for measuring fast, not for believing early.
**Routing:** QA72 (dithered realization + rounding-field steering + stage-attribution probe +
continuous-solve-then-CVP-round race) → ledger, defer-at-source; tt1 messaged to instrument the
threshold-crossing signature; QA65/QA69 unchanged (they are the storage-lattice half).

## §7 ADDENDUM (operator 07-31, STANDING STEER) — comprehensive view: no axis priority; joint sensitivity + synergies + order of operations

**Operator verbatim:** "make sure now that we're getting smaller and smaller not to necessarily
prioritize pose or seg but take a comprehensive view and synergies and order of operations and
all physics and photometrics and all dimensions and layers and types upstream and all
dependencies and weights and hyperplanes and channels are sensitive to."

**Why this binds NOW (measured):** at S 0.993 the axes are near-parity (seg 0.431 · pose 0.322 ·
rate 0.240) — marginal prices have converged, so the next −0.1 lives wherever the JOINT exchange
rate says, not where the last win was. Axis-reflex (pose-first inertia, seg-first reflex) becomes
the dominant allocation error exactly here. The non-additive-pools LAW says per-axis pursuit
double-counts shared budgets; the Knee-A arc measured the cross-coupling both ways (seg cell
drops COST pose; pose re-solve PAID for seg damage).

**The standing law (binding on every charter from here):** no rung fires on axis identity; every
rung fires on its JOINT realized exchange rate read from ONE unified table. The three legs:
1. **THE JOINT SENSITIVITY ATLAS AT THE LIVE BASE** — unify every measured scorer-internal
   surface: SegNet rank-4 head hyperplanes + margin-Fisher Gram + BN-channel structure (sn1/#725)
   + ker(A) resize null (shared both scorers) + PoseNet per-dim Jacobians/quadratic (pi2/ms4) +
   photometric response (pm1) + composite-R Hessian — RE-ANCHORED at the v4c/v4d base (the
   staleness confound: prior bundles were measured at dead bases; freshness-at-consumption).
2. **ONE ARCHIVE-LEVEL JOINT WATERFILL** across all pools (token quanta/cell set · pose precision
   · photometric · expert menu · selector) at REALIZED exchange rates → the ranked next-rung
   table. This is QA69-leg-3 promoted from a v4d rung to THE standing allocator.
3. **THE ORDER-OF-OPERATIONS DAG** — dependencies are physical (motion → per-depth projection →
   photometric → uint8; base changes invalidate pose solves; token nudges move rounding fields)
   and re-solve costs are real; the allocator must emit ORDER, not just amounts (oc1's
   "ORDER is important" lifted from build-time to campaign-time).
Consumer: the costate organ's duty queue reads THIS table; MAIN charters read it before spawning.
→ QA73 + ddm_ja1 arm. Sister laws: non-additive-pools · meet-it-where-it-is 4-clause ·
holistic-check-ins-are-facets · decompose-every-headline.

## §8 ADDENDUM (operator 07-31, 9th convocation + STANDING GO) — SegNet: the amortization gap is the crux; the re-burn fires

**Measured SegNet state:** d_seg 0.00431179 at the v4c gate (0.431 S = largest axis) · renderer burn
endpoint 0.0038892 FLAT (full base) · byte pool SATURATED (gr1 knee ±move = +0.05 S; corrections
break-even ×5) · **the exact solve PROVED d_seg 0.00116 is reachable on this video** (ms2r_r3
in-box, 291 MB value-materialized = rate-dead) · exact structure: rank-4 head hyperplanes + flip
distance d=|m|/‖Δw‖ · stride-2 stem, ERF r50≈85 px (region-reader) · flips ~50% Road / 19% Lane
(77% skip-limited) / 13% Undrivable · hood static IoU 0.994.

**SCHMIDHUBER'S FRAME (the crux in one number):** renderer endpoint 0.00389 vs solved 0.00116 =
**3.35× AMORTIZATION GAP**. The solve is the existence proof; the renderer is the predict stage
that fails to reproduce what the solve proved reachable. SegNet's remaining descent is not "more
training" and not "more bytes" (both measured saturated) — it is CLOSING THE AMORTIZATION GAP.
Three ordered rungs:
1. **QA74 — TYPE THE RESIDUAL ($0, FIRST — the pose-collapse playbook applied to seg):** decompose
   the live 0.00431 residual by {margin depth at the rank-4 head · class · spatial stationarity ·
   renderer-vs-SOLVE disagreement}. The last is the decisive column: pixels where the SOLVE also
   fails are target/GT-floor (concede); pixels where the solve succeeds and the renderer fails
   are the amortization gap (attack). This typed split did for pose what no capacity sweep did.
2. **QA24 — COARSER-FROM-BIRTH RE-BURN (FIRES NOW under standing GO):** gr1 measured that post-hoc
   dropping the low-|g| half of cells is near-free → a from-birth coarse grid re-allocates ALL
   training capacity onto cells that matter and can only beat the −0.098 post-hoc bound. Grid
   DERIVED from gr1's curve (not assumed); **solve-INIT tokens** (lv1's headline A/B — start the
   burn AT the solved object's projection, not from noise); sched1 event-driven schedule; full
   launch non-negotiables (governed launcher · DSL-hash · memory preflight · resumable + per-stage
   checkpoints · EMA shadow). Invalidation budgeted: post-burn pose re-solve (~3.5 h proven) +
   photometric re-fit (~35 min proven) — the capacity pool runs PARALLEL to v4d per ja1.
3. **QA75 — SOLVE-DISTILLATION (pn1-S5 revived, the named unfired lever):** the solve provides
   PERFECT per-pixel targets ON THIS VIDEO — train/finish the renderer to match the SOLVED frames
   (KD #74 lineage, reopened at this vehicle by rv1) instead of fighting argmax-CE against GT.
   Sidesteps the CE-vs-argmax wall; fires as the burn's finishing stage or post-burn, informed by
   QA74's typed split (distill ONLY the amortization-gap regions).
**TISHBY/CB1 (per-class carriers, the seg expert-menu):** hood is static (IoU 0.994, 25.6% of
frame) — a ~0.5 KB static carrier relieves the renderer of a quarter of the frame; Lane (77%
skip-limited) needs either a dedicated thin-structure carrier or the #149 pre-R sub-pixel
placement. cb1's byte-closed carriers exist; QA74's typing decides which classes earn one.
**WONDERS:** (Tao) is Lane's 77% skip-limit SegNet's own stride-2 resolution (→ concede or pre-R
place) or the renderer's reach (→ QA24 cures it)? QA74 answers via the solve column. (MacKay) the
old 0.0053 "GT-flicker floor" is construction-specific and already beaten — the true floor
question is the solve column, not that number. (Contrarian, binding) QA24's prediction rests on a
post-hoc bound; the from-birth claim is a RACE — falsifier: re-burn endpoint ≥ cell_drop50's
0.00431 at matched bytes → coarse-from-birth closes at INSTANCE, solve-distillation becomes the
lead rung.
**Routing:** QA74 ($0, fires inside the sg1 arm first) · QA24 FIRES under the standing GO just
granted (sg1 arm stages through the governed chain and launches) · QA75 rowed as the follow-on
rung. Sister: ja1 table (QA24 = the only seg descent, capacity pool, parallel).
