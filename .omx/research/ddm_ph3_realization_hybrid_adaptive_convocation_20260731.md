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
